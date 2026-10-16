//! 特征匹配（M4）：配对生成 → 描述子精确 2-NN 匹配 → F/H 几何验证 → 落库。
//!
//! 对齐 COLMAP 4.2.1 的三个匹配器（`exhaustive_matcher` / `sequential_matcher` /
//! `matches_importer --match_type pairs`）：
//! `controllers/feature_matching.cc` + `controllers/feature_matching_utils.cc` +
//! `estimators/pairing.cc` + `feature/sift.cc` 的 CPU 分支。
//!
//! - 配对：exhaustive（`block_size = 50` 的块配对，逐块迭代并去重）、sequential
//!   （按图像名排序，`overlap = 10` 的二次间隔 `1, 2, 4, ...`）、imported
//!   （`pair_list` 文本每行 `name1 name2`，按 `block_size = 1225` 分批）；
//! - 已完成的图像对（`matches` 与 `two_view_geometries` 都在）直接跳过；只缺一半的
//!   先删掉已有的一半再重算（COLMAP `FeatureMatcherController::Match` 的原语义，
//!   其中"只有原始匹配"时复用匹配、只补几何验证）；
//! - 描述子：两向 2-NN + `max_ratio = 0.8` 比值检验 + `max_distance = 0.7`
//!   （阈值 = `512² · 0.7²`）+ 交叉验证，输出按第一张图的下标升序；
//! - 落库：原始匹配不足 `min_num_inliers = 15` 时 `matches` 写空 blob；几何验证
//!   （M4：F/H 路径）失败或被裁到不足 15 条内点时 `two_view_geometries` 写
//!   `DEGENERATE` 空载荷行；描述子缺失时只写空的 `matches` 行、不写几何行；
//! - 全程 CPU，不读 GPU 选项（`gpu_index` 仅进日志）。
//!
//! 刻意差异（见方案文档 M4 决策）：
//!
//! - **描述子检索精确**：4.2.1 的 CPU 路径在 ≥512 个描述子时用 faiss IVF 近似检索，
//!   这里统一精确暴力 2-NN → 只会"更准"；
//! - **CPU 路径不施加 `max_num_matches`**：与 4.2.1 的 CPU 运行时一致（该上限只用于
//!   GPU 与 guided matching，本项目都不走）；
//! - **闭环检测（vocabulary tree）是 no-op 预留**：4.2.1 在 `loop_detection = 1` 时会再
//!   跑一遍词袋配对（需要 `vocab_tree_path`）；自研后端 M4 只日志提示、不生成这些配对
//!   （方案 D5/M6 预留），也绝不调用 `vocabulary::prepare`；
//! - **RANSAC 采样确定性**：每对图像按 `pair_id` 派生随机种子（4.2.1 默认每次运行不同）。

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use nalgebra::{DMatrix, Vector2};
use rayon::prelude::*;
use rusqlite::Connection;

use crate::error::{Result, SplatError};

use super::format::database::{self, ImageRow, TwoViewGeometry};
use super::runtime::ProgressSink;
use super::two_view::{self, Rng, VerifyOptions};

/// 阶段名（前端 `runtime.phase`；M7 补中文文案）。
pub const PHASE: &str = "matching";

/// `ExhaustivePairingOptions::block_size` 默认值。
const EXHAUSTIVE_BLOCK_SIZE: usize = 50;
/// `SequentialPairingOptions::overlap` 默认值（项目显式传 `--SequentialMatching.overlap 10`）。
const SEQUENTIAL_OVERLAP: usize = 10;
/// `ImportedPairingOptions::block_size` 默认值。
const IMPORTED_BLOCK_SIZE: usize = 1225;

/// 配对策略（对应三个匹配器）。
#[derive(Clone, Debug)]
pub enum PairStrategy {
    /// `sequential_matcher`；`loop_detection` 在 M4 是 no-op 预留。
    Sequential { loop_detection: bool },
    /// `exhaustive_matcher`。
    Exhaustive,
    /// `matches_importer --match_type pairs`（`match_list` 文本）。
    Imported { match_list: PathBuf },
}

#[derive(Clone, Debug)]
pub struct MatchingRequest {
    pub database: PathBuf,
    pub strategy: PairStrategy,
}

/// SIFT 描述子匹配参数（`SiftMatchingOptions` 默认值）。
#[derive(Clone, Copy, Debug)]
struct SiftMatchOptions {
    max_ratio: f32,
    max_distance: f32,
    cross_check: bool,
}

impl Default for SiftMatchOptions {
    fn default() -> Self {
        Self {
            max_ratio: 0.8,
            max_distance: 0.7,
            cross_check: true,
        }
    }
}

// ---------------------------------------------------------------- 配对生成

/// 配对生成器（逐个"块"产出，与 COLMAP 的 `Next()` 同构）。
enum PairGenerator {
    Exhaustive {
        image_ids: Vec<u32>,
        block_size: usize,
        start_idx1: usize,
        start_idx2: usize,
        finished: bool,
    },
    Sequential {
        image_ids: Vec<u32>,
        overlap: usize,
        image_idx: usize,
        finished: bool,
    },
    Imported {
        batches: Vec<Vec<[u32; 2]>>,
        index: usize,
    },
}

impl PairGenerator {
    fn new(images: &[ImageRow], strategy: &PairStrategy, sink: &ProgressSink) -> Result<Self> {
        match strategy {
            PairStrategy::Exhaustive => {
                let image_ids: Vec<u32> = images.iter().map(|image| image.image_id).collect();
                sink.log(format!(
                    "Generating exhaustive image pairs (block_size = {EXHAUSTIVE_BLOCK_SIZE})"
                ));
                Ok(Self::Exhaustive {
                    image_ids,
                    block_size: EXHAUSTIVE_BLOCK_SIZE,
                    start_idx1: 0,
                    start_idx2: 0,
                    finished: false,
                })
            }
            PairStrategy::Sequential { loop_detection } => {
                // COLMAP `SequentialPairGenerator::GetOrderedImageIds`：按图像名排序。
                let mut ordered = images.to_vec();
                ordered.sort_by(|left, right| left.name.cmp(&right.name));
                let image_ids: Vec<u32> = ordered.iter().map(|image| image.image_id).collect();
                sink.log(format!(
                    "Generating sequential image pairs (overlap = {SEQUENTIAL_OVERLAP}, \
                     quadratic_overlap = true, loop_detection = {})",
                    u8::from(*loop_detection)
                ));
                if *loop_detection {
                    sink.log(
                        "注意：自研后端 M4 尚未实现词袋闭环检测，本轮只为相邻图像配对（预留接口）",
                    );
                }
                Ok(Self::Sequential {
                    image_ids,
                    overlap: SEQUENTIAL_OVERLAP,
                    image_idx: 0,
                    finished: false,
                })
            }
            PairStrategy::Imported { match_list } => {
                let pairs = read_image_pairs_text(match_list, images, sink)?;
                sink.log(format!(
                    "Importing matches for {} image pairs (block_size = {IMPORTED_BLOCK_SIZE})",
                    pairs.len()
                ));
                let batches = pairs
                    .chunks(IMPORTED_BLOCK_SIZE)
                    .map(|chunk| chunk.to_vec())
                    .collect();
                Ok(Self::Imported { batches, index: 0 })
            }
        }
    }

    /// 预计产出的图像对数（进度显示用）。
    fn total_pairs(&self) -> u64 {
        match self {
            Self::Exhaustive { image_ids, .. } => {
                let n = image_ids.len() as u64;
                n * n.saturating_sub(1) / 2
            }
            Self::Sequential {
                image_ids, overlap, ..
            } => {
                let n = image_ids.len();
                let mut total = 0u64;
                for index in 0..n {
                    for offset in 0..*overlap {
                        if index + (1usize << offset) < n {
                            total += 1;
                        } else {
                            break;
                        }
                    }
                }
                total
            }
            Self::Imported { batches, .. } => batches.iter().map(|batch| batch.len() as u64).sum(),
        }
    }

    /// 下一个"块"（COLMAP `Next()`；返回空批次表示结束）。
    fn next_batch(&mut self, sink: &ProgressSink) -> Vec<[u32; 2]> {
        match self {
            Self::Exhaustive {
                image_ids,
                block_size,
                start_idx1,
                start_idx2,
                finished,
            } => {
                if *finished {
                    return Vec::new();
                }
                let size = image_ids.len();
                let num_blocks = size.div_ceil(*block_size);
                let end_idx1 = (size.min(*start_idx1 + *block_size)).saturating_sub(1);
                let end_idx2 = (size.min(*start_idx2 + *block_size)).saturating_sub(1);
                sink.log(format!(
                    "Processing block [{}/{}, {}/{}]",
                    *start_idx1 / *block_size + 1,
                    num_blocks,
                    *start_idx2 / *block_size + 1,
                    num_blocks
                ));
                let mut pairs = Vec::new();
                for idx1 in *start_idx1..=end_idx1 {
                    for idx2 in *start_idx2..=end_idx2 {
                        let block_id1 = idx1 % *block_size;
                        let block_id2 = idx2 % *block_size;
                        // 与 COLMAP 一致：每个无序对只生成一次。
                        if (idx1 > idx2 && block_id1 <= block_id2)
                            || (idx1 < idx2 && block_id1 < block_id2)
                        {
                            pairs.push([image_ids[idx1], image_ids[idx2]]);
                        }
                    }
                }
                *start_idx2 += *block_size;
                if *start_idx2 >= size {
                    *start_idx2 = 0;
                    *start_idx1 += *block_size;
                }
                if *start_idx1 >= size {
                    *finished = true;
                }
                pairs
            }
            Self::Sequential {
                image_ids,
                overlap,
                image_idx,
                finished,
            } => {
                if *finished {
                    return Vec::new();
                }
                sink.log(format!(
                    "Processing image [{}/{}]",
                    *image_idx + 1,
                    image_ids.len()
                ));
                let image_id1 = image_ids[*image_idx];
                let mut pairs = Vec::new();
                for offset in 0..*overlap {
                    let image_idx2 = *image_idx + (1usize << offset);
                    if image_idx2 < image_ids.len() {
                        pairs.push([image_id1, image_ids[image_idx2]]);
                    } else {
                        break;
                    }
                }
                *image_idx += 1;
                if *image_idx >= image_ids.len() {
                    *finished = true;
                }
                pairs
            }
            Self::Imported { batches, index } => {
                let batch = batches.get(*index).cloned().unwrap_or_default();
                *index += 1;
                batch
            }
        }
    }
}

/// `ReadImagePairsText`：跳过空行与 `#` 注释；未知图像名报错跳过；按 `pair_id` 去重。
fn read_image_pairs_text(
    path: &std::path::Path,
    images: &[ImageRow],
    sink: &ProgressSink,
) -> Result<Vec<[u32; 2]>> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        SplatError::Process(format!("读取配对列表 {} 失败：{error}", path.display()))
    })?;
    let name_to_id: HashMap<&str, u32> = images
        .iter()
        .map(|image| (image.name.as_str(), image.image_id))
        .collect();
    let mut pairs = Vec::new();
    let mut seen = HashSet::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split(' ');
        let name1 = fields.next().unwrap_or("").trim();
        let name2 = fields.next().unwrap_or("").trim();
        let Some(&image_id1) = name_to_id.get(name1) else {
            sink.log(format!("Image {name1} does not exist."));
            continue;
        };
        let Some(&image_id2) = name_to_id.get(name2) else {
            sink.log(format!("Image {name2} does not exist."));
            continue;
        };
        if image_id1 == image_id2 {
            // COLMAP 的 `ImagePairToPairId` 不接受同一张图像；列表里出现时直接跳过。
            continue;
        }
        let pair = database::pair_id(image_id1, image_id2);
        if seen.insert(pair) {
            pairs.push([image_id1, image_id2]);
        }
    }
    Ok(pairs)
}

// ---------------------------------------------------------------- 特征缓存

/// 一对图像匹配 / 验证需要的全部数据。
struct ImageFeatures {
    points: Vec<Vector2<f64>>,
    /// `None`：库里没有（完整）描述子 → 走 COLMAP 的"缺描述子直通"路径。
    descriptors: Option<DMatrix<f32>>,
    /// 相机宽高（水印检测用）。
    camera: (u32, u32),
}

/// 本次运行的图像对涉及的图像（带淘汰，避免整库常驻）。
#[derive(Default)]
struct FeatureStore {
    images: HashMap<u32, Arc<ImageFeatures>>,
    camera_dims: HashMap<u32, (u32, u32)>,
}

impl FeatureStore {
    fn get(&self, image_id: u32) -> &Arc<ImageFeatures> {
        self.images
            .get(&image_id)
            .expect("特征缓存必须先 load")
    }

    /// 只保留 `keep` 里的图像，并把缺的读进来（串行查库，SQLite 单写者）。
    fn load(&mut self, connection: &Connection, keep: &HashSet<u32>) -> Result<()> {
        self.images.retain(|image_id, _| keep.contains(image_id));
        for &image_id in keep {
            if self.images.contains_key(&image_id) {
                continue;
            }
            let Some((_, camera_id)) = read_image_row(connection, image_id)? else {
                return Err(SplatError::Process(format!(
                    "图像 {image_id} 从数据库中消失（匹配期间不应发生）"
                )));
            };
            let camera = match self.camera_dims.get(&camera_id) {
                Some(dims) => *dims,
                None => {
                    let camera_row = database::read_camera(connection, camera_id)?;
                    let dims = camera_row
                        .map(|camera| (camera.width as u32, camera.height as u32))
                        .unwrap_or((0, 0));
                    self.camera_dims.insert(camera_id, dims);
                    dims
                }
            };

            let keypoints = database::read_keypoints(connection, image_id)?;
            let descriptors = database::read_descriptors(connection, image_id)?;
            let features = match (keypoints, descriptors) {
                (Some(keypoints), Some((rows, data))) if rows == keypoints.len() && rows > 0 => {
                    let points = keypoints
                        .iter()
                        .map(|keypoint| Vector2::new(keypoint.x as f64, keypoint.y as f64))
                        .collect();
                    let values: Vec<f32> = data.iter().map(|&value| value as f32).collect();
                    let descriptors = DMatrix::from_row_slice(rows, 128, &values);
                    ImageFeatures {
                        points,
                        descriptors: Some(descriptors),
                        camera,
                    }
                }
                // 键点 / 描述子缺一个（或行数对不上）：当作"没有描述子"处理。
                _ => ImageFeatures {
                    points: Vec::new(),
                    descriptors: None,
                    camera,
                },
            };
            self.images.insert(image_id, Arc::new(features));
        }
        Ok(())
    }
}

fn read_image_row(connection: &Connection, image_id: u32) -> Result<Option<(String, u32)>> {
    connection
        .query_row(
            "SELECT name, camera_id FROM images WHERE image_id = ?1",
            [image_id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as u32)),
        )
        .map(Some)
        .or_else(|error| match error {
            rusqlite::Error::QueryReturnedNoRows => Ok(None),
            other => Err(SplatError::Process(format!("COLMAP 数据库操作失败：{other}"))),
        })
}

// ---------------------------------------------------------------- 描述子匹配

/// 两向 2-NN + 比值检验 + 交叉验证（对应 `FindBestMatchesIndex` 的语义）。
fn match_descriptors(
    descriptors1: &DMatrix<f32>,
    descriptors2: &DMatrix<f32>,
    options: &SiftMatchOptions,
) -> Vec<[u32; 2]> {
    if descriptors1.nrows() == 0 || descriptors2.nrows() == 0 {
        return Vec::new();
    }
    let matches_1to2 = best_matches_one_way(descriptors1, descriptors2, options);
    if !options.cross_check {
        return matches_1to2
            .iter()
            .enumerate()
            .filter_map(|(index, &other)| {
                (other >= 0).then_some([index as u32, other as u32])
            })
            .collect();
    }
    let matches_2to1 = best_matches_one_way(descriptors2, descriptors1, options);
    matches_1to2
        .iter()
        .enumerate()
        .filter_map(|(index, &other)| {
            if other < 0 || matches_2to1[other as usize] != index as i32 {
                None
            } else {
                Some([index as u32, other as u32])
            }
        })
        .collect()
}

/// `FindBestMatchesOneWayIndex` 的精确版：每行找最近与次近（近的在前）。
///
/// `max_distance` 阈值用 `512² · max_distance²`（描述子经 L1 归一化 ×512，L2 范数 = 512）；
/// 比值检验保持 COLMAP 的 `sqrt(best) >= ratio · sqrt(second)` 与 `>` 的边界语义。
fn best_matches_one_way(
    query: &DMatrix<f32>,
    database: &DMatrix<f32>,
    options: &SiftMatchOptions,
) -> Vec<i32> {
    let num_query = query.nrows();
    let num_database = database.nrows();
    let mut result = vec![-1i32; num_query];
    if num_query == 0 || num_database == 0 {
        return result;
    }

    const DESCRIPTOR_DIM: usize = 128;
    const SQ_NORM: f32 = (512.0 * 512.0) as f32;
    let max_l2_dist = SQ_NORM * options.max_distance * options.max_distance;
    // 分块计算内积：`||q - d||² = ||q||² + ||d||² - 2·q·d`（与 faiss 的 flat 检索同量级精度）。
    const BLOCK: usize = 256;

    let database_norms = row_norms(database);
    let database_t = database.transpose();
    for start in (0..num_query).step_by(BLOCK) {
        let rows = (num_query - start).min(BLOCK);
        let query_block = query.view((start, 0), (rows, DESCRIPTOR_DIM));
        let query_norms = row_norms(&query_block.into_owned());
        let dots = query_block * &database_t;
        for row in 0..rows {
            let query_norm = query_norms[row];
            let mut best_idx = -1i32;
            let mut best = f32::MAX;
            let mut second = f32::MAX;
            for column in 0..num_database {
                let dist = query_norm + database_norms[column] - 2.0 * dots[(row, column)];
                let dist = dist.max(0.0);
                if dist < best {
                    second = best;
                    best = dist;
                    best_idx = column as i32;
                } else if dist < second {
                    second = dist;
                }
            }
            if best_idx < 0 {
                continue;
            }
            if best > max_l2_dist {
                continue;
            }
            // 保持 `>=`：best == second 时必须被检出（COLMAP 原注释）。
            if best.sqrt() >= options.max_ratio * second.sqrt() {
                continue;
            }
            result[start + row] = best_idx;
        }
    }
    result
}

fn row_norms(matrix: &DMatrix<f32>) -> Vec<f32> {
    (0..matrix.nrows())
        .map(|row| matrix.row(row).norm_squared())
        .collect()
}

// ---------------------------------------------------------------- 图像对处理

/// 一个待处理的图像对。
struct PairJob {
    image_id1: u32,
    image_id2: u32,
    /// `Some`：复用已有原始匹配，只补几何验证（COLMAP 的只缺一半路径）。
    existing_matches: Option<Vec<[u32; 2]>>,
}

/// 一对图像的完整产物（已完成按 `min_num_inliers` 裁剪）。
struct PairOutcome {
    matches: Vec<[u32; 2]>,
    /// `None`：几何行不写（描述子缺失 → COLMAP 的 `UNDEFINED` 直通路径）。
    geometry: Option<TwoViewGeometry>,
}

fn process_pair(
    job: &PairJob,
    store: &FeatureStore,
    match_options: &SiftMatchOptions,
    verify_options: &VerifyOptions,
) -> PairOutcome {
    let features1 = store.get(job.image_id1);
    let features2 = store.get(job.image_id2);
    let (Some(descriptors1), Some(descriptors2)) =
        (&features1.descriptors, &features2.descriptors)
    else {
        // COLMAP：描述子缺失 → 原始匹配为空、`two_view_geometry` 保持 `UNDEFINED`（不落库）。
        return PairOutcome {
            matches: Vec::new(),
            geometry: None,
        };
    };

    let raw_matches = match &job.existing_matches {
        Some(matches) => matches.clone(),
        None => match_descriptors(descriptors1, descriptors2, match_options),
    };

    if raw_matches.len() < verify_options.min_num_inliers {
        // 验证器提前中止：`DEGENERATE` 空载荷行；匹配按 COLMAP 清空。
        return PairOutcome {
            matches: Vec::new(),
            geometry: Some(degenerate_two_view_geometry()),
        };
    }

    let seed = pair_seed(job.image_id1, job.image_id2);
    let verified = two_view::verify(
        &raw_matches,
        &features1.points,
        &features2.points,
        features1.camera,
        features2.camera,
        verify_options,
        &Rng::new(seed),
    );

    let geometry = if verified.config == two_view::DEGENERATE
        || verified.inlier_matches.len() < verify_options.min_num_inliers
    {
        degenerate_two_view_geometry()
    } else {
        TwoViewGeometry {
            config: verified.config,
            inlier_matches: verified.inlier_matches,
            f: verified.f.as_ref().map(two_view::to_row_major),
            e: None,
            h: verified.h.as_ref().map(two_view::to_row_major),
            cam2_from_cam1: None,
        }
    };

    PairOutcome {
        matches: raw_matches,
        geometry: Some(geometry),
    }
}

fn degenerate_two_view_geometry() -> TwoViewGeometry {
    TwoViewGeometry {
        config: two_view::DEGENERATE,
        ..Default::default()
    }
}

fn pair_seed(image_id1: u32, image_id2: u32) -> u64 {
    (database::pair_id(image_id1, image_id2) as u64)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(0x243f_6a88_85a3_08d3)
}

/// 一批里"需要算"的图像对（去重、跳过已完成、半成品先删）。
fn collect_jobs(
    connection: &Connection,
    batch: &[[u32; 2]],
) -> Result<(Vec<PairJob>, usize)> {
    let mut seen = HashSet::with_capacity(batch.len());
    let mut jobs = Vec::new();
    let mut skipped = 0usize;
    for &[image_id1, image_id2] in batch {
        if image_id1 == image_id2 {
            continue;
        }
        if !seen.insert(database::pair_id(image_id1, image_id2)) {
            continue;
        }
        let exists_matches = database::exists_matches(connection, image_id1, image_id2)?;
        let exists_geometry = database::exists_two_view_geometry(connection, image_id1, image_id2)?;
        if exists_matches && exists_geometry {
            skipped += 1;
            continue;
        }
        // 与 COLMAP 相同：只缺一半时先删掉已有的一半，避免主键冲突。
        if exists_geometry {
            database::delete_two_view_geometry(connection, image_id1, image_id2)?;
        }
        if exists_matches {
            let matches = database::read_matches(connection, image_id1, image_id2)?.unwrap_or_default();
            database::delete_matches(connection, image_id1, image_id2)?;
            jobs.push(PairJob {
                image_id1,
                image_id2,
                existing_matches: Some(matches),
            });
        } else {
            jobs.push(PairJob {
                image_id1,
                image_id2,
                existing_matches: None,
            });
        }
    }
    Ok((jobs, skipped))
}

// ---------------------------------------------------------------- 主入口

/// 三个匹配器共用的主体。
pub fn match_images(sink: &ProgressSink, request: &MatchingRequest) -> Result<()> {
    let connection = database::open(&request.database)?;
    let images = database::read_images(&connection)?;
    if images.is_empty() {
        sink.log("特征匹配：数据库里没有任何图像，直接返回");
        return Ok(());
    }

    let match_options = SiftMatchOptions::default();
    let verify_options = VerifyOptions::default();

    let mut generator = PairGenerator::new(&images, &request.strategy, sink)?;
    let total_pairs = generator.total_pairs();
    sink.log(format!(
        "特征匹配开始：{} 张图像，预计 {total_pairs} 个图像对（全程 CPU；\
         描述子 2-NN 精确检索）",
        images.len()
    ));

    let mut done = 0u64;
    let mut skipped = 0u64;
    let mut verified_pairs = 0u64;
    let mut inlier_total = 0u64;
    let mut store = FeatureStore::default();

    loop {
        sink.check_cancelled()?;
        let batch = generator.next_batch(sink);
        if batch.is_empty() {
            break;
        }

        let (jobs, skipped_here) = collect_jobs(&connection, &batch)?;
        skipped += skipped_here as u64;
        if jobs.is_empty() {
            done += batch.len() as u64;
            sink.stage(
                PHASE,
                done,
                Some(total_pairs),
                format!("已匹配 {done}/{total_pairs} 对面（本次跳过 {skipped_here} 对）"),
            );
            continue;
        }

        // 本次涉及的所有图像（含只有描述子缺失的），一次读进缓存。
        let mut involved = HashSet::with_capacity(jobs.len() * 2);
        for job in &jobs {
            involved.insert(job.image_id1);
            involved.insert(job.image_id2);
        }
        store.load(&connection, &involved)?;

        // 并行计算（每对独立；RANSAC 种子由 pair_id 派生，结果与顺序无关）。
        let outcomes: Vec<PairOutcome> = jobs
            .par_iter()
            .map(|job| process_pair(job, &store, &match_options, &verify_options))
            .collect();

        // 串行落库（SQLite 单写者）。
        database::begin_bulk(&connection)?;
        let write_result = (|| -> Result<()> {
            for (job, outcome) in jobs.iter().zip(&outcomes) {
                database::write_matches(&connection, job.image_id1, job.image_id2, &outcome.matches)?;
                if let Some(geometry) = &outcome.geometry {
                    database::write_two_view_geometry(
                        &connection,
                        job.image_id1,
                        job.image_id2,
                        geometry,
                    )?;
                }
            }
            Ok(())
        })();
        match write_result {
            Ok(()) => database::end_bulk(&connection)?,
            Err(error) => {
                let _ = connection.execute("ROLLBACK", []);
                return Err(error);
            }
        }

        for outcome in &outcomes {
            if let Some(geometry) = &outcome.geometry {
                if geometry.config != two_view::DEGENERATE {
                    verified_pairs += 1;
                    inlier_total += geometry.inlier_matches.len() as u64;
                }
            }
        }

        done += batch.len() as u64;
        sink.stage(
            PHASE,
            done,
            Some(total_pairs),
            format!("已匹配 {done}/{total_pairs} 对面"),
        );
    }

    let _ = match_options;
    sink.log(format!(
        "特征匹配完成：处理 {done} 个图像对（跳过 {skipped} 个已完成），\
         几何验证通过 {verified_pairs} 对，内点匹配合计 {inlier_total} 条"
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::colmap::apple::runtime;
    use rusqlite::Connection;
    use tempfile::TempDir;
    use tokio_util::sync::CancellationToken;

    fn sink() -> ProgressSink {
        runtime::test_sink(CancellationToken::new()).unwrap()
    }

    fn images(rows: &[(u32, &str)]) -> Vec<ImageRow> {
        rows.iter()
            .map(|(image_id, name)| ImageRow {
                image_id: *image_id,
                name: (*name).to_string(),
                camera_id: 1,
            })
            .collect()
    }

    // ------------------------------------------------------------ 配对生成

    fn all_pairs(strategy: &PairStrategy, rows: &[ImageRow]) -> Vec<[u32; 2]> {
        let sink = sink();
        let mut generator = PairGenerator::new(rows, strategy, &sink).unwrap();
        let mut pairs = Vec::new();
        loop {
            let batch = generator.next_batch(&sink);
            if batch.is_empty() {
                break;
            }
            pairs.extend(batch);
        }
        pairs
    }

    #[test]
    fn exhaustive_pairs_are_complete_and_unique() {
        let rows = images(&[
            (1, "a.jpg"),
            (2, "b.jpg"),
            (3, "c.jpg"),
            (4, "d.jpg"),
            (5, "e.jpg"),
        ]);
        let pairs = all_pairs(&PairStrategy::Exhaustive, &rows);
        assert_eq!(pairs.len(), 10, "5 张图应有 10 个无序对");
        let unique: HashSet<_> = pairs.iter().map(|pair| database::pair_id(pair[0], pair[1])).collect();
        assert_eq!(unique.len(), pairs.len(), "不应有重复对");
        for pair in &pairs {
            assert_ne!(pair[0], pair[1]);
        }
    }

    #[test]
    fn exhaustive_pairs_span_multiple_blocks_without_duplicates() {
        let rows: Vec<ImageRow> = (1..=120)
            .map(|index| ImageRow {
                image_id: index,
                name: format!("frame_{index:04}.jpg"),
                camera_id: 1,
            })
            .collect();
        let pairs = all_pairs(&PairStrategy::Exhaustive, &rows);
        assert_eq!(pairs.len(), 120 * 119 / 2);
        let unique: HashSet<_> = pairs.iter().map(|pair| database::pair_id(pair[0], pair[1])).collect();
        assert_eq!(unique.len(), pairs.len());
    }

    #[test]
    fn sequential_pairs_use_quadratic_offsets() {
        let rows: Vec<ImageRow> = (1..=20)
            .map(|index| ImageRow {
                image_id: index,
                name: format!("frame_{index:04}.jpg"),
                camera_id: 1,
            })
            .collect();
        let pairs = all_pairs(
            &PairStrategy::Sequential {
                loop_detection: false,
            },
            &rows,
        );
        // 第 1 张：1↔2、1↔3、1↔5、1↔9、1↔17（1+2^k < 20）。
        for (index, pair) in pairs.iter().take(5).enumerate() {
            assert_eq!(pair[0], 1);
            assert_eq!(pair[1], 1 + (1u32 << index));
        }
        // 总数：Σ min(10, 20 - i) for i in 1..=20（1-based）→ 前 10 张各 10 个，其余递减。
        let expected: usize = (1..=20)
            .map(|index| {
                let mut count = 0;
                for offset in 0..10 {
                    if index + (1usize << offset) <= 20 {
                        count += 1;
                    } else {
                        break;
                    }
                }
                count
            })
            .sum();
        assert_eq!(pairs.len(), expected);
    }

    #[test]
    fn sequential_orders_images_by_name() {
        // 名字顺序与编号顺序相反。
        let rows = images(&[(7, "a.jpg"), (3, "b.jpg"), (5, "c.jpg")]);
        let pairs = all_pairs(
            &PairStrategy::Sequential {
                loop_detection: false,
            },
            &rows,
        );
        assert_eq!(pairs[0], [7, 3]);
        assert_eq!(pairs[1], [7, 5]);
        assert_eq!(pairs[2], [3, 5]);
    }

    #[test]
    fn imported_pair_list_skips_comments_and_unknown_names() {
        let directory = TempDir::new().unwrap();
        let list = directory.path().join("pairs.txt");
        std::fs::write(
            &list,
            "# 注释\n\na.jpg b.jpg\nmissing.jpg b.jpg\nb.jpg a.jpg\na.jpg a.jpg\nc.jpg b.jpg\n",
        )
        .unwrap();
        let rows = images(&[(1, "a.jpg"), (2, "b.jpg"), (3, "c.jpg")]);
        let run = sink();
        let pairs = read_image_pairs_text(&list, &rows, &run).unwrap();
        // `b.jpg a.jpg` 与 `a.jpg b.jpg` 同 `pair_id` → 去重；自配对跳过。
        assert_eq!(pairs, vec![[1, 2], [3, 2]]);
    }

    // ------------------------------------------------------------ 描述子匹配

    /// 造 2 个"图像"的描述子：第二张是若干张的重复 + 干扰项。
    fn descriptor_matrix(rows: &[[u8; 4]]) -> DMatrix<f32> {
        let mut data = vec![0u8; rows.len() * 128];
        for (index, row) in rows.iter().enumerate() {
            for (column, &value) in row.iter().enumerate() {
                data[index * 128 + column] = value;
            }
        }
        let values: Vec<f32> = data.iter().map(|&value| value as f32).collect();
        DMatrix::from_row_slice(rows.len(), 128, &values)
    }

    #[test]
    fn descriptor_matching_keeps_mutual_nearest() {
        let descriptors1 = descriptor_matrix(&[[10, 20, 30, 40], [200, 210, 220, 230], [50, 60, 70, 80]]);
        let descriptors2 = descriptor_matrix(&[[10, 20, 30, 41], [50, 61, 70, 80], [90, 100, 110, 120]]);
        let matches = match_descriptors(&descriptors1, &descriptors2, &SiftMatchOptions::default());
        // 第 1 个描述子 ↔ 第 1 个；第 3 个 ↔ 第 2 个；第 2 个（200..）无相互最近邻。
        assert_eq!(matches, vec![[0, 0], [2, 1]]);
    }

    #[test]
    fn descriptor_matching_rejects_ambiguous_ratio() {
        // best = 144（第 4 维差 12）、second = 196（差 14）→ 比值 12/14 ≈ 0.86 ≥ 0.8 → 拒绝。
        let descriptors1 = descriptor_matrix(&[[100, 100, 100, 100]]);
        let descriptors2 = descriptor_matrix(&[[100, 100, 100, 112], [100, 100, 100, 114]]);
        let matches = match_descriptors(&descriptors1, &descriptors2, &SiftMatchOptions::default());
        assert!(matches.is_empty(), "模糊匹配应被比值检验拒绝");
    }

    #[test]
    fn descriptor_matching_enforces_max_distance() {
        // 唯一候选但距离超限（> 512²·0.7²）。
        let descriptors1 = descriptor_matrix(&[[0, 0, 0, 0]]);
        let descriptors2 = descriptor_matrix(&[[255, 255, 255, 255]]);
        let matches = match_descriptors(&descriptors1, &descriptors2, &SiftMatchOptions::default());
        assert!(matches.is_empty(), "超距离的匹配应被拒绝");
    }

    // ------------------------------------------------------------ 端到端

    use crate::engines::colmap::apple::format::camera_model;
    use crate::engines::colmap::apple::{features, test_images};

    /// 端到端夹具：临时目录 + 数据库 + 图像目录（两块图像共用一套相机参数）。
    struct E2eFixture {
        #[allow(dead_code)] // 持有 TempDir，防止目录提前被删。
        directory: TempDir,
        database: PathBuf,
        images: PathBuf,
    }

    impl E2eFixture {
        fn new() -> Self {
            let directory = TempDir::new().unwrap();
            let database = directory.path().join("database.db");
            let images = directory.path().join("frames");
            std::fs::create_dir_all(&images).unwrap();
            Self {
                directory,
                database,
                images,
            }
        }

        fn open(&self) -> Connection {
            database::open_read_only(&self.database).unwrap()
        }

        fn create_database(&self) -> Connection {
            database::open(&self.database).unwrap()
        }

        /// 走 M3 的特征提取（真实图片文件 → 键点 + 描述子 + 相机）。
        fn extract(&self) {
            let request = features::ExtractionRequest {
                database: self.database.clone(),
                images: self.images.clone(),
                masks: None,
                image_list: None,
                max_image_size: None,
                max_features: None,
                existing_camera_id: None,
                gpu_index: None,
            };
            features::extract(&sink(), &request).unwrap();
        }

        fn match_exhaustive(&self) -> Result<()> {
            match_images(
                &sink(),
                &MatchingRequest {
                    database: self.database.clone(),
                    strategy: PairStrategy::Exhaustive,
                },
            )
        }
    }

    /// 同一张合成大图的平移裁剪 → 真值位移 `(dx, dy)`；两张图都应匹配出这组位移。
    fn write_crop_pair(fixture: &E2eFixture) -> (i32, i32) {
        test_images::write_translated_pair(
            &fixture.images.join("frame_0001.png"),
            &fixture.images.join("frame_0002.png"),
            320,
            240,
            40,
            24,
            0x51fa,
        )
    }

    #[test]
    fn end_to_end_matching_writes_verified_pair() {
        let fixture = E2eFixture::new();
        let (dx, dy) = write_crop_pair(&fixture);
        fixture.extract();
        fixture.match_exhaustive().unwrap();

        let connection = fixture.open();
        let matches = database::read_matches(&connection, 1, 2).unwrap().unwrap();
        assert!(matches.len() >= 15, "原始匹配太少：{}", matches.len());

        let geometry = database::read_two_view_geometry(&connection, 1, 2)
            .unwrap()
            .unwrap();
        assert!(
            geometry.config == two_view::UNCALIBRATED
                || geometry.config == two_view::PLANAR_OR_PANORAMIC,
            "平移裁剪场景 config = {}（应为 UNCALIBRATED 或 PLANAR_OR_PANORAMIC）",
            geometry.config
        );
        assert!(
            geometry.inlier_matches.len() >= 15,
            "内点太少：{}",
            geometry.inlier_matches.len()
        );
        assert!(geometry.f.is_some() || geometry.h.is_some());

        // 内点位移应集中在已知平移附近（同纹理、无损 PNG，2 px 容差很宽松）。
        let keypoints1 = database::read_keypoints(&connection, 1).unwrap().unwrap();
        let keypoints2 = database::read_keypoints(&connection, 2).unwrap().unwrap();
        let consistent = geometry
            .inlier_matches
            .iter()
            .filter(|m| {
                let p1 = keypoints1[m[0] as usize];
                let p2 = keypoints2[m[1] as usize];
                ((p2.x - p1.x) - dx as f32).abs() <= 2.0
                    && ((p2.y - p1.y) - dy as f32).abs() <= 2.0
            })
            .count();
        assert!(
            consistent * 10 >= geometry.inlier_matches.len() * 9,
            "内点里只有 {consistent}/{} 条的位移接近真值 ({dx}, {dy})",
            geometry.inlier_matches.len()
        );
    }

    #[test]
    fn rerunning_matching_is_idempotent() {
        let fixture = E2eFixture::new();
        write_crop_pair(&fixture);
        fixture.extract();
        fixture.match_exhaustive().unwrap();

        let snapshot = |connection: &Connection| {
            (
                database::read_matches(connection, 1, 2).unwrap().unwrap(),
                database::read_two_view_geometry(connection, 1, 2)
                    .unwrap()
                    .unwrap(),
            )
        };
        let before = snapshot(&fixture.open());

        // 第二遍：两个表都已完成 → 跳过全部图像对，内容逐位不变。
        fixture.match_exhaustive().unwrap();
        let after = snapshot(&fixture.open());
        assert_eq!(before, after);
    }

    #[test]
    fn missing_descriptors_write_empty_matches_without_geometry() {
        let fixture = E2eFixture::new();
        let connection = fixture.create_database();
        let camera_id = database::insert_camera(
            &connection,
            camera_model::SIMPLE_RADIAL,
            320,
            240,
            &[384.0, 160.0, 120.0, 0.0],
            false,
        )
        .unwrap();
        database::write_image(&connection, 1, "frame_0001.png", camera_id).unwrap();
        database::write_image(&connection, 2, "frame_0002.png", camera_id).unwrap();
        let keypoints: Vec<database::Keypoint> = (0..20)
            .map(|index| {
                database::Keypoint::upright(10.0 + index as f32 * 7.0, 12.0 + index as f32 * 5.0)
            })
            .collect();
        database::write_keypoints(&connection, 1, &keypoints).unwrap();
        database::write_keypoints(&connection, 2, &keypoints).unwrap();
        // 只有第 1 张有描述子。
        database::write_descriptors(&connection, 1, &vec![1u8; 20 * 128]).unwrap();
        drop(connection);

        fixture.match_exhaustive().unwrap();

        let connection = fixture.open();
        assert!(database::exists_matches(&connection, 1, 2).unwrap());
        assert_eq!(
            database::read_matches(&connection, 1, 2).unwrap(),
            Some(Vec::new()),
            "描述子缺失应写空 matches 行"
        );
        assert!(
            database::read_two_view_geometry(&connection, 1, 2)
                .unwrap()
                .is_none(),
            "描述子缺失不应写几何行（UNDEFINED 直通）"
        );
    }

    #[test]
    fn cancelled_matching_fails_with_cancelled() {
        let fixture = E2eFixture::new();
        let connection = fixture.create_database();
        let camera_id = database::insert_camera(
            &connection,
            camera_model::SIMPLE_RADIAL,
            320,
            240,
            &[384.0, 160.0, 120.0, 0.0],
            false,
        )
        .unwrap();
        // 至少一张图像，否则空库会直接成功返回、不经过取消检查。
        database::write_image(&connection, 1, "frame_0001.png", camera_id).unwrap();
        drop(connection);

        let token = CancellationToken::new();
        let run = runtime::test_sink(token.clone()).unwrap();
        token.cancel();
        let result = match_images(
            &run,
            &MatchingRequest {
                database: fixture.database.clone(),
                strategy: PairStrategy::Exhaustive,
            },
        );
        assert!(matches!(result, Err(SplatError::Cancelled)));
    }
}
