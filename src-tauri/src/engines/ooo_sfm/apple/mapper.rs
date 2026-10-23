//! M5d：增量重建主循环（`IncrementalPipeline` + `IncrementalMapper` 等价实现）。
//!
//! 结构与阈值逐项对齐 COLMAP 4.2.1（规格见 `docs/project-notes/recon/m5-mapper.md`）：
//! - [`MapperOptions`]：4.2.1 `IncrementalMapper::Options` 的全字段默认值；
//! - [`PipelineOptions`]：4.2.1 `IncrementalPipelineOptions` 里与调度相关的部分
//!   （全局 BA 触发比例/频率、局部/全局迭代上限、重试与终止条件）；
//! - 主循环：`FindNextImages` → `RegisterNextImage` → `TriangulateImage` →
//!   `IterativeLocalRefinement` →（触发条件满足时）`IterativeGlobalRefinement`；
//!   连续两轮注册失败即退出（4.2.1 `incremental_pipeline.cc:599-724`）。
//!
//! 刻意差异（记录在案）：
//! 1. **不做 structure-less 注册回退**（`structure_less_registration_fallback`）——
//!    本项目的输入是同一设备的连续帧，PnP 失败即换下一张即可。
//! 2. 不实现多模型（`multiple_models`）与快照（`snapshot_frames_freq`）：三个入口
//!    都以"一个模型"为准（`map_from_existing` 显式传 `multiple_models=0`）。
//! 3. 不实现初始化松弛（`kNumInitRelaxations`）与 `init_num_trials` 重试预算之外的
//!    统计回滚；初始化失败即返回"无可注册图像"，由上层按 COLMAP 的失败语义报错。
//! 4. 初始化的相对位姿由 [`super::relative_pose`] 给出（不做焦距联合估计，见该模块）。

use std::collections::HashSet;
use std::path::Path;

use super::ba::{run_bundle_adjustment, BaCamera, BaImage, BaPoint, Gauge, Loss};
use super::camera;
use super::format::database;
use super::format::model::{Camera, Point2D, TrackElement};
use super::pose::estimate_absolute_pose;
use super::reconstruction::{
    filter_all_points3d, CorrespondenceGraph, ImageState, ObservationManager, Reconstruction,
};
use super::relative_pose::relative_pose_from_geometry;
use super::runtime::ProgressSink;
use super::triangulator::{TriangulationOptions, Triangulator};
use super::two_view::{RansacOptions, Rng, DEGENERATE, WATERMARK};

use crate::error::{Result, SplatError};

/// 4.2.1 `IncrementalMapper::Options`（默认值逐项一致）。
#[derive(Clone, Copy, Debug)]
pub struct MapperOptions {
    pub init_min_num_inliers: usize,
    pub init_max_error: f64,
    pub init_max_forward_motion: f64,
    pub init_min_tri_angle: f64,
    pub init_max_reg_trials: usize,
    pub abs_pose_max_error: f64,
    pub abs_pose_min_num_inliers: usize,
    pub abs_pose_min_inlier_ratio: f64,
    pub abs_pose_refine_focal_length: bool,
    pub abs_pose_refine_extra_params: bool,
    pub ba_local_num_images: usize,
    pub ba_local_min_tri_angle: f64,
    pub min_focal_length_ratio: f64,
    pub max_focal_length_ratio: f64,
    pub max_extra_param: f64,
    pub filter_max_reproj_error: f64,
    pub filter_min_tri_angle: f64,
    pub max_reg_trials: usize,
    pub fix_existing_frames: bool,
    pub refine_focal_length: bool,
    pub refine_principal_point: bool,
    pub refine_extra_params: bool,
}

impl Default for MapperOptions {
    fn default() -> Self {
        Self {
            init_min_num_inliers: 100,
            init_max_error: 4.0,
            init_max_forward_motion: 0.95,
            init_min_tri_angle: 16.0,
            init_max_reg_trials: 2,
            abs_pose_max_error: 12.0,
            abs_pose_min_num_inliers: 30,
            abs_pose_min_inlier_ratio: 0.25,
            abs_pose_refine_focal_length: true,
            abs_pose_refine_extra_params: true,
            ba_local_num_images: 6,
            ba_local_min_tri_angle: 6.0,
            min_focal_length_ratio: 0.1,
            max_focal_length_ratio: 10.0,
            max_extra_param: 1.0,
            filter_max_reproj_error: 4.0,
            filter_min_tri_angle: 1.5,
            max_reg_trials: 3,
            fix_existing_frames: false,
            refine_focal_length: true,
            refine_principal_point: false,
            refine_extra_params: true,
        }
    }
}

/// 4.2.1 `IncrementalPipelineOptions` 中与调度相关的字段。
#[derive(Clone, Copy, Debug)]
pub struct PipelineOptions {
    pub ba_local_max_refinements: usize,
    pub ba_local_max_refinement_change: f64,
    pub ba_global_max_refinements: usize,
    pub ba_global_max_refinement_change: f64,
    pub ba_global_frames_ratio: f64,
    pub ba_global_points_ratio: f64,
    pub ba_global_frames_freq: usize,
    pub ba_global_points_freq: usize,
    /// 4.2.1 `kDefaultCeresLocalMaxNumIterations = 25`。
    pub ba_local_max_num_iterations: u32,
    /// 4.2.1 `kDefaultCeresGlobalMaxNumIterations = 50`。
    pub ba_global_max_num_iterations: u32,
    /// 局部 BA 的 Ceres `gradient_tolerance = 10.0`。
    pub ba_local_gradient_tolerance: f64,
    /// 全局 BA 的 Ceres `gradient_tolerance = 1.0`。
    pub ba_global_gradient_tolerance: f64,
    pub max_model_overlap: usize,
    pub min_model_size: usize,
    /// 4.2.1：单个 next_images 列表内连续失败多少张后放弃当前初值对。
    pub min_num_initial_reg_trials: usize,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            ba_local_max_refinements: 2,
            ba_local_max_refinement_change: 0.001,
            ba_global_max_refinements: 5,
            ba_global_max_refinement_change: 0.0005,
            ba_global_frames_ratio: 1.1,
            ba_global_points_ratio: 1.1,
            ba_global_frames_freq: 500,
            ba_global_points_freq: 250_000,
            ba_local_max_num_iterations: 25,
            ba_global_max_num_iterations: 50,
            ba_local_gradient_tolerance: 10.0,
            ba_global_gradient_tolerance: 1.0,
            max_model_overlap: 20,
            min_model_size: 10,
            min_num_initial_reg_trials: 30,
        }
    }
}

/// 两视图几何（mapper 侧需要的字段）。
#[derive(Clone, Debug)]
pub struct PairGeometry {
    pub image1: usize,
    pub image2: usize,
    pub config: i32,
    pub f: Option<[[f64; 3]; 3]>,
    pub h: Option<[[f64; 3]; 3]>,
    /// 内点对应（`(idx1, idx2)`，已按 `idx1` 升序）。
    pub inliers: Vec<(u32, u32)>,
}

/// 从数据库加载的重建上下文。
pub struct Mapper {
    pub recon: Reconstruction,
    pub graph: CorrespondenceGraph,
    pub obs: ObservationManager,
    pub triangulator: Triangulator,
    pub geometries: HashSet<(usize, usize)>,
    pub pairs: Vec<PairGeometry>,
    pub options: MapperOptions,
    pub tri_options: TriangulationOptions,
    pub num_reg_trials: Vec<usize>,
    /// 已注册图像的 id 集合（写模型时用）。
    pub filtered_images: HashSet<usize>,
    /// 已有模型（`map_from_existing` / `map_incremental`）中已注册的图像下标。
    pub existing_images: HashSet<usize>,
}

/// 数据库加载结果 + 图像过滤（`Mapper.image_list_path` 语义）。
pub fn load(
    database: &Path,
    image_names: Option<&HashSet<String>>,
) -> Result<Mapper> {
    let connection = database::open_read_only(database)?;
    let mut camera_rows = connection
        .prepare("SELECT camera_id, model, width, height, params, prior_focal_length FROM cameras")
        .map_err(|error| SplatError::Process(format!("读取 cameras 失败：{error}")))?;
    let mut cameras: Vec<Camera> = Vec::new();
    let mut camera_prior: Vec<bool> = Vec::new();
    let mut rows = camera_rows
        .query([])
        .map_err(|error| SplatError::Process(format!("读取 cameras 失败：{error}")))?;
    while let Some(row) = rows
        .next()
        .map_err(|error| SplatError::Process(format!("读取 cameras 失败：{error}")))?
    {
        let camera_id: u32 = row.get(0).map_err(sqlite)?;
        let model_id: i32 = row.get(1).map_err(sqlite)?;
        let width: i64 = row.get(2).map_err(sqlite)?;
        let height: i64 = row.get(3).map_err(sqlite)?;
        let params: Vec<u8> = row.get(4).map_err(sqlite)?;
        let prior: i64 = row.get(5).map_err(sqlite)?;
        let params = decode_params(&params)?;
        if super::format::camera_model::num_params(model_id) != Some(params.len()) {
            return Err(SplatError::Process(format!(
                "相机 {camera_id} 的参数个数与模型 {model_id} 不符"
            )));
        }
        cameras.push(Camera::new(camera_id, model_id, width as u64, height as u64, params));
        camera_prior.push(prior != 0);
    }

    let image_rows = database::read_images(&connection)?;
    // 稠密下标：`index ↔ image_id`。
    let mut image_id_to_index: std::collections::HashMap<u32, usize> =
        std::collections::HashMap::new();
    let mut states: Vec<ImageState> = Vec::new();
    for row in &image_rows {
        if let Some(filter) = image_names {
            if !filter.contains(&row.name) {
                continue;
            }
        }
        let keypoints = database::read_keypoints(&connection, row.image_id)?;
        let Some(keypoints) = keypoints else {
            continue;
        };
        let camera_index = cameras
            .iter()
            .position(|camera| camera.id == row.camera_id)
            .ok_or_else(|| SplatError::Process(format!("图像 {} 引用了不存在的相机", row.image_id)))?;
        image_id_to_index.insert(row.image_id, states.len());
        states.push(ImageState {
            id: row.image_id,
            name: row.name.clone(),
            camera: camera_index,
            points2d: keypoints
                .iter()
                .map(|keypoint| Point2D::new(keypoint.x as f64, keypoint.y as f64))
                .collect(),
            colors: database::read_feature_colors(&connection, row.image_id)?,
            registered: false,
            qvec: [1.0, 0.0, 0.0, 0.0],
            tvec: [0.0; 3],
        });
    }
    let num_points2d: Vec<usize> = states
        .iter()
        .map(|state| state.points2d.len())
        .collect();
    let recon = Reconstruction::new(cameras, camera_prior, states);
    let mut graph = CorrespondenceGraph::new(&num_points2d);

    // 对应图：只用**非退化几何对的内点**（4.2.1 `DatabaseCache::Create` 语义）。
    let mut geometries = HashSet::new();
    let mut pairs = Vec::new();
    for [image_id1, image_id2] in database::read_verified_pairs(&connection)? {
        let (Some(index1), Some(index2)) = (
            image_id_to_index.get(&image_id1).copied(),
            image_id_to_index.get(&image_id2).copied(),
        ) else {
            continue;
        };
        let Some(geometry) = database::read_two_view_geometry(&connection, image_id1, image_id2)?
        else {
            continue;
        };
        if geometry.config == DEGENERATE || geometry.config == WATERMARK {
            continue;
        }
        let inliers: Vec<(u32, u32)> = geometry
            .inlier_matches
            .iter()
            .map(|pair| (pair[0], pair[1]))
            .collect();
        for (idx1, idx2) in &inliers {
            if (*idx1 as usize) < num_points2d[index1] && (*idx2 as usize) < num_points2d[index2] {
                graph.add_match(index1, *idx1, index2, *idx2);
            }
        }
        geometries.insert((index1, index2));
        pairs.push(PairGeometry {
            image1: index1,
            image2: index2,
            config: geometry.config,
            f: geometry.f,
            h: geometry.h,
            inliers,
        });
    }
    graph.finalize();
    let obs = ObservationManager::new(&graph, &recon);
    let num_images = recon.num_images();
    Ok(Mapper {
        recon,
        graph,
        obs,
        triangulator: Triangulator::new(),
        geometries,
        pairs,
        options: MapperOptions::default(),
        tri_options: TriangulationOptions::default(),
        num_reg_trials: vec![0; num_images],
        filtered_images: HashSet::new(),
        existing_images: HashSet::new(),
    })
}

fn sqlite(error: rusqlite::Error) -> SplatError {
    SplatError::Process(format!("数据库读取失败：{error}"))
}

fn decode_params(data: &[u8]) -> Result<Vec<f64>> {
    if data.len() % 8 != 0 {
        return Err(SplatError::Process(format!(
            "相机参数块长度非法：{}",
            data.len()
        )));
    }
    Ok(data
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("8 字节")))
        .collect())
}

impl Mapper {
    pub fn num_reg_images(&self) -> usize {
        self.recon.num_reg_images()
    }

    pub fn num_points3d(&self) -> usize {
        self.recon.num_points3d()
    }

    fn camera_has_prior(&self, image: usize) -> bool {
        self.recon.camera_prior_focal[self.recon.images[image].camera]
    }

    fn camera_is_bogus(&self, image: usize) -> bool {
        camera::has_bogus_params(
            self.recon.camera_of(image),
            self.options.min_focal_length_ratio,
            self.options.max_focal_length_ratio,
            self.options.max_extra_param,
        )
    }

    // ------------------------------------------------------------ 初始化

    /// 4.2.1 `FindInitialImagePair`：按"有焦距先验优先 + 对应数降序"选种子图，
    /// 再按与种子图的对应数（≥ `init_min_num_inliers`）选第二张，逐对试一次。
    pub fn find_initial_image_pair(&mut self, rng: &mut Rng) -> Option<(usize, usize)> {
        struct Candidate {
            image: usize,
            prior: bool,
            count: usize,
        }
        let mut first: Vec<Candidate> = (0..self.recon.num_images())
            .filter(|image| self.graph.num_correspondences(*image) > 0)
            .map(|image| Candidate {
                image,
                prior: self.camera_has_prior(image),
                count: self.graph.num_correspondences(image),
            })
            .collect();
        first.sort_by(|a, b| b.prior.cmp(&a.prior).then(b.count.cmp(&a.count)));

        let mut tried: HashSet<(usize, usize)> = HashSet::new();
        for candidate in first {
            // 与种子图对应数 ≥ init_min_num_inliers 的候选第二张。
            let mut second: Vec<(usize, usize)> = (0..self.recon.num_images())
                .filter(|image| *image != candidate.image)
                .map(|image| (image, 0usize))
                .collect();
            second.clear();
            let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
            for idx in 0..self.recon.images[candidate.image].points2d.len() as u32 {
                for corr in self.graph.find(candidate.image, idx) {
                    *counts.entry(corr.image).or_insert(0) += 1;
                }
            }
            let mut second: Vec<(usize, usize)> = counts
                .into_iter()
                .filter(|(_, count)| *count >= self.options.init_min_num_inliers)
                .collect();
            second.sort_by(|a, b| {
                let prior_a = self.camera_has_prior(a.0);
                let prior_b = self.camera_has_prior(b.0);
                prior_b.cmp(&prior_a).then(b.1.cmp(&a.1))
            });
            for (image, _) in second {
                let key = (candidate.image, image).min((image, candidate.image));
                if !tried.insert(key) {
                    continue;
                }
                if self.estimate_initial_two_view_geometry(candidate.image, image, rng) {
                    return Some((candidate.image, image));
                }
            }
        }
        None
    }

    /// 4.2.1 `EstimateInitialTwoViewGeometry`：用库里存的两视图几何求相对位姿，
    /// 并检查 `inliers ≥ init_min_num_inliers`、`|t_z| < init_max_forward_motion`、
    /// `tri_angle > init_min_tri_angle`。
    pub fn estimate_initial_two_view_geometry(
        &self,
        image1: usize,
        image2: usize,
        _rng: &mut Rng,
    ) -> bool {
        let Some(geometry) = self
            .pairs
            .iter()
            .find(|pair| {
                (pair.image1 == image1 && pair.image2 == image2)
                    || (pair.image1 == image2 && pair.image2 == image1)
            })
        else {
            return false;
        };
        let (first, second) = if geometry.image1 == image1 {
            (image1, image2)
        } else {
            (image2, image1)
        };
        let points1: Vec<[f64; 2]> = geometry
            .inliers
            .iter()
            .map(|(idx1, _)| {
                let point = self.recon.images[first].points2d[*idx1 as usize];
                [point.x, point.y]
            })
            .collect();
        let points2: Vec<[f64; 2]> = geometry
            .inliers
            .iter()
            .map(|(_, idx2)| {
                let point = self.recon.images[second].points2d[*idx2 as usize];
                [point.x, point.y]
            })
            .collect();
        if points1.len() < self.options.init_min_num_inliers {
            return false;
        }
        let Some(report) = relative_pose_from_geometry(
            geometry.config,
            self.recon.camera_of(first),
            self.recon.camera_of(second),
            &points1,
            &points2,
            geometry.f.as_ref(),
            geometry.h.as_ref(),
            self.camera_has_prior(first),
            self.camera_has_prior(second),
            self.options.min_focal_length_ratio,
            self.options.max_focal_length_ratio,
        ) else {
            return false;
        };
        report.tri_angle_deg > self.options.init_min_tri_angle
            && report.tvec[2].abs() < self.options.init_max_forward_motion
    }

    /// 4.2.1 `RegisterInitialImagePair` + 初始三角化 + 全局 BA + 归一化 + 过滤。
    pub fn register_initial_image_pair(
        &mut self,
        image1: usize,
        image2: usize,
        rng: &mut Rng,
    ) -> Result<bool> {
        let Some(geometry_index) = self
            .pairs
            .iter()
            .position(|pair| {
                (pair.image1 == image1 && pair.image2 == image2)
                    || (pair.image1 == image2 && pair.image2 == image1)
            })
        else {
            return Ok(false);
        };
        let geometry = self.pairs[geometry_index].clone();
        let (first, second) = if geometry.image1 == image1 {
            (image1, image2)
        } else {
            (image2, image1)
        };
        let points1: Vec<[f64; 2]> = geometry
            .inliers
            .iter()
            .map(|(idx1, _)| {
                let point = self.recon.images[first].points2d[*idx1 as usize];
                [point.x, point.y]
            })
            .collect();
        let points2: Vec<[f64; 2]> = geometry
            .inliers
            .iter()
            .map(|(_, idx2)| {
                let point = self.recon.images[second].points2d[*idx2 as usize];
                [point.x, point.y]
            })
            .collect();
        let Some(report) = relative_pose_from_geometry(
            geometry.config,
            self.recon.camera_of(first),
            self.recon.camera_of(second),
            &points1,
            &points2,
            geometry.f.as_ref(),
            geometry.h.as_ref(),
            self.camera_has_prior(first),
            self.camera_has_prior(second),
            self.options.min_focal_length_ratio,
            self.options.max_focal_length_ratio,
        ) else {
            return Ok(false);
        };
        if report.tri_angle_deg <= self.options.init_min_tri_angle
            || report.tvec[2].abs() >= self.options.init_max_forward_motion
        {
            return Ok(false);
        }
        // 第一台相机放在世界系原点。
        self.obs
            .register_image(&self.graph, &mut self.recon, first, [1.0, 0.0, 0.0, 0.0], [0.0; 3]);
        // cam2_from_cam1 → world（第一台为单位阵）。
        self.obs.register_image(
            &self.graph,
            &mut self.recon,
            second,
            report.qvec,
            report.tvec,
        );

        // 初始三角化（4.2.1：初始对用 `init_min_tri_angle` 作为 `tri_options.min_angle`）。
        let mut tri_options = self.tri_options;
        tri_options.min_angle = self.options.init_min_tri_angle;
        self.triangulator.triangulate_image(
            &self.graph,
            &mut self.recon,
            &mut self.obs,
            &tri_options,
            first,
            rng,
        );
        self.triangulator
            .triangulate_image(&self.graph, &mut self.recon, &mut self.obs, &tri_options, second, rng);
        let dbg = std::env::var("OOO_MAP_DEBUG").is_ok();
        if dbg {
            eprintln!(
                "[map] 初始三角化得到 {} 点（min_angle={:.1}°，内点 {}）",
                self.recon.num_points3d(),
                tri_options.min_angle,
                points1.len()
            );
        }
        if self.recon.num_points3d() == 0 {
            return Ok(false);
        }
        let mean_error = |mapper: &Self| -> f64 {
            let mut sum = 0.0;
            let mut count = 0usize;
            for point in &mapper.recon.points {
                for element in &point.track {
                    let image = mapper.recon.image_index[&element.image_id];
                    let observation = mapper.recon.images[image].points2d[element.point2d_idx as usize];
                    sum += camera::reprojection_error(
                        mapper.recon.camera_of(image),
                        mapper.recon.model_of(image),
                        &mapper.recon.images[image].qvec,
                        &mapper.recon.images[image].tvec,
                        point.xyz,
                        [observation.x, observation.y],
                    );
                    count += 1;
                }
            }
            if count == 0 { 0.0 } else { sum / count as f64 }
        };
        if dbg {
            eprintln!("[map] BA 前平均重投影误差 = {:.4} px", mean_error(self));
        }
        let ba_ok = self.adjust_global_bundle(PipelineOptions::default(), true);
        if dbg {
            eprintln!("[map] BA 后平均重投影误差 = {:.4} px", mean_error(self));
            let c0 = self.recon.projection_center_of(0);
            let c1 = self.recon.projection_center_of(1);
            let baseline = (0..3)
                .map(|axis| (c1[axis] - c0[axis]).powi(2))
                .sum::<f64>()
                .sqrt();
            let mut angles: Vec<f64> = Vec::new();
            for point in &self.recon.points {
                angles.push(
                    super::triangulator::triangulation_angle(c0, c1, point.xyz).to_degrees(),
                );
            }
            angles.sort_by(|a, b| a.partial_cmp(b).expect("角度有限"));
            eprintln!(
                "[map] BA 后基线 = {baseline:.4}，三角角 min={:.2}° 中位={:.2}° max={:.2}°",
                angles.first().copied().unwrap_or(0.0),
                angles.get(angles.len() / 2).copied().unwrap_or(0.0),
                angles.last().copied().unwrap_or(0.0)
            );
        }
        if dbg {
            eprintln!(
                "[map] 初始全局 BA ok={ba_ok} → {} 点 / {} 图像",
                self.recon.num_points3d(),
                self.recon.num_reg_images()
            );
        }
        self.recon.normalize(10.0);
        let filtered = self.filter_points();
        if dbg {
            eprintln!(
                "[map] normalize + 过滤 {} 个观测后 → {} 点 / {} 图像",
                filtered,
                self.recon.num_points3d(),
                self.recon.num_reg_images()
            );
        }
        if self.recon.num_reg_images() == 0 || self.recon.num_points3d() == 0 {
            return Ok(false);
        }
        let ok = self.recon.num_points3d() >= self.options.abs_pose_min_num_inliers;
        if dbg {
            eprintln!("[map] 初始对可用 = {ok}（点数 {}，阈值 {}）", self.recon.num_points3d(), self.options.abs_pose_min_num_inliers);
        }
        Ok(ok)
    }

    // ------------------------------------------------------------ 注册

    /// 4.2.1 `FindNextImages`：`MIN_UNCERTAINTY` 排序（可见性金字塔分数降序），
    /// 两桶（未试过且未被过滤 / 其余）。
    pub fn find_next_images(&self) -> Vec<usize> {
        if std::env::var("OOO_MAP_DEBUG").is_ok() {
            let detail: Vec<String> = (0..self.recon.num_images())
                .map(|image| {
                    format!(
                        "{image}:{}",
                        self.obs.num_visible_points3D(image)
                    )
                })
                .collect();
            eprintln!(
                "[next] 可见 3D 点数 {}（下限 {}，已注册 {} 图 / {} 点）",
                detail.join(" "),
                self.options.abs_pose_min_num_inliers,
                self.num_reg_images(),
                self.num_points3d()
            );
        }
        let mut primary: Vec<(usize, usize)> = Vec::new();
        let mut secondary: Vec<(usize, usize)> = Vec::new();
        for image in 0..self.recon.num_images() {
            if self.recon.images[image].registered {
                continue;
            }
            if self.obs.num_visible_points3D(image) < self.options.abs_pose_min_num_inliers {
                continue;
            }
            if self.num_reg_trials[image] >= self.options.max_reg_trials {
                continue;
            }
            let rank = self.obs.point3d_visibility_score(image);
            if !self.filtered_images.contains(&image) && self.num_reg_trials[image] == 0 {
                primary.push((image, rank));
            } else {
                secondary.push((image, rank));
            }
        }
        primary.sort_by(|a, b| b.1.cmp(&a.1));
        secondary.sort_by(|a, b| b.1.cmp(&a.1));
        primary
            .into_iter()
            .chain(secondary)
            .map(|(image, _)| image)
            .collect()
    }

    /// 4.2.1 `RegisterNextImage`。
    pub fn register_next_image(&mut self, image: usize, rng: &mut Rng) -> Result<bool> {
        if self.recon.num_reg_images() == 0 {
            return Ok(false);
        }
        self.num_reg_trials[image] += 1;
        if self.obs.num_visible_points3D(image) < self.options.abs_pose_min_num_inliers {
            return Ok(false);
        }
        // 2D-3D 对应（来自对应图：对方特征点已有三维点）。
        let mut points2d: Vec<[f64; 2]> = Vec::new();
        let mut points3d: Vec<[f64; 3]> = Vec::new();
        let mut point_ids: Vec<usize> = Vec::new();
        let mut seen: HashSet<usize> = HashSet::new();
        for point2d_idx in 0..self.recon.images[image].points2d.len() as u32 {
            for index in 0..self.graph.find(image, point2d_idx).len() {
                let corr = self.graph.find(image, point2d_idx)[index];
                if !self.recon.images[corr.image].registered {
                    continue;
                }
                let Some(point) = self.recon.point_index_at(corr.image, corr.point2d_idx) else {
                    continue;
                };
                if !seen.insert(point) {
                    continue;
                }
                if self.camera_is_bogus(corr.image) {
                    continue;
                }
                let observation = self.recon.images[image].points2d[point2d_idx as usize];
                points2d.push([observation.x, observation.y]);
                points3d.push(self.recon.points[point].xyz);
                point_ids.push(point);
            }
        }
        if std::env::var("OOO_MAP_DEBUG").is_ok() {
            eprintln!(
                "[reg] 图像 {image}: 可见点 {} / 观测 {} → 2D-3D 对应 {}（下限 {}）",
                self.obs.num_visible_points3D(image),
                self.obs.num_observations(image),
                points2d.len(),
                self.options.abs_pose_min_num_inliers
            );
        }
        if points2d.len() < self.options.abs_pose_min_num_inliers {
            return Ok(false);
        }

        // 内参精化规则（4.2.1 `incremental_mapper.cc:360-412`）。
        let camera_index = self.recon.images[image].camera;
        let prior = self.recon.camera_prior_focal[camera_index];
        let reg_before = self
            .registered_count_for_camera(camera_index);
        let refine_focal = self.options.abs_pose_refine_focal_length
            && self.options.refine_focal_length
            && reg_before == 0
            && !prior;
        let refine_extra =
            self.options.abs_pose_refine_extra_params && self.options.refine_extra_params;

        let mut params = self.recon.cameras[camera_index].params.clone();
        let model = camera::model_of(&self.recon.cameras[camera_index]);
        let options = RansacOptions {
            max_error: self.options.abs_pose_max_error,
            min_inlier_ratio: self.options.abs_pose_min_inlier_ratio,
            ..RansacOptions::default()
        };
        let report = estimate_absolute_pose(
            &points2d,
            &points3d,
            model,
            &mut params,
            refine_focal,
            refine_extra,
            &options,
            rng,
        );
        if std::env::var("OOO_MAP_DEBUG").is_ok() {
            eprintln!(
                "[reg] 图像 {image}: RANSAC success={} 内点 {} / {}（阈值 {:.2} px）",
                report.success,
                report.num_inliers,
                points2d.len(),
                self.options.abs_pose_max_error
            );
        }
        if !report.success || report.num_inliers < self.options.abs_pose_min_num_inliers {
            return Ok(false);
        }
        // 内参被更新后先写回（后续过滤/BA 都要用）。
        self.recon.cameras[camera_index].params = params;

        self.obs.register_image(
            &self.graph,
            &mut self.recon,
            image,
            report.qvec,
            report.tvec,
        );
        // 续上已有 track。
        let mut cursor = 0usize;
        for point2d_idx in 0..self.recon.images[image].points2d.len() as u32 {
            if self.recon.point_index_at(image, point2d_idx).is_some() {
                continue;
            }
            let mut target: Option<usize> = None;
            for index in 0..self.graph.find(image, point2d_idx).len() {
                let corr = self.graph.find(image, point2d_idx)[index];
                if !self.recon.images[corr.image].registered {
                    continue;
                }
                if let Some(point) = self.recon.point_index_at(corr.image, corr.point2d_idx) {
                    target = Some(point);
                    break;
                }
            }
            let _ = &mut cursor;
            if let Some(point) = target {
                let element = TrackElement {
                    image_id: self.recon.images[image].id,
                    point2d_idx,
                };
                if self.obs.add_observation(&self.graph, &mut self.recon, point, element) {
                    self.triangulator.add_modified(point);
                }
            }
        }
        Ok(true)
    }

    fn registered_count_for_camera(&self, camera_index: usize) -> usize {
        self.recon
            .registered
            .iter()
            .filter(|image| self.recon.images[**image].camera == camera_index)
            .count()
    }

    // ------------------------------------------------------------ 三角化与过滤

    pub fn triangulate_image(&mut self, image: usize, rng: &mut Rng) -> usize {
        let options = self.tri_options;
        self.triangulator.triangulate_image(
            &self.graph,
            &mut self.recon,
            &mut self.obs,
            &options,
            image,
            rng,
        )
    }

    pub fn filter_points(&mut self) -> usize {
        filter_all_points3d(
            &self.graph,
            &mut self.recon,
            &mut self.obs,
            self.options.filter_max_reproj_error,
            self.options.filter_min_tri_angle,
        )
    }

    // ------------------------------------------------------------ BA

    /// 4.2.1 `IterativeLocalRefinement`：局部 bundle 上迭代 BA + 补全/合并/过滤，
    /// 变化率 < `ba_local_max_refinement_change` 即停；第一轮用鲁棒损失、后续 TRIVIAL。
    pub fn iterative_local_refinement(
        &mut self,
        pipeline: PipelineOptions,
        image: usize,
    ) -> bool {
        let local_bundle = self.find_local_bundle(image);
        let mut loss = Loss::SoftL1(1.0);
        let mut changed = 0usize;
        for round in 0..pipeline.ba_local_max_refinements {
            let images = local_bundle.clone();
            self.adjust_local_bundle(pipeline, &images, loss);
            loss = Loss::Trivial;
            let merged = self
                .triangulator
                .complete_and_merge_tracks(&self.graph, &mut self.recon, &mut self.obs, &self.tri_options);
            let filtered = self.filter_points();
            changed = merged + filtered;
            let ratio = changed as f64 / (self.recon.compute_num_observations().max(1) as f64);
            if ratio < pipeline.ba_local_max_refinement_change {
                break;
            }
            let _ = round;
        }
        self.triangulator.clear_modified();
        changed > 0
    }

    /// 4.2.1 `FindLocalBundle`：参考图 + 共视最多的 `ba_local_num_images - 1` 张。
    pub fn find_local_bundle(&self, image: usize) -> Vec<usize> {
        let mut shared: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for point2d_idx in 0..self.recon.images[image].points2d.len() as u32 {
            let Some(point) = self.recon.point_index_at(image, point2d_idx) else {
                continue;
            };
            for element in &self.recon.points[point].track {
                if element.image_id == self.recon.images[image].id {
                    continue;
                }
                let other = self.recon.image_index[&element.image_id];
                *shared.entry(other).or_insert(0) += 1;
            }
        }
        let mut overlapping: Vec<(usize, usize)> = shared.into_iter().collect();
        overlapping.sort_by(|a, b| b.1.cmp(&a.1));
        let limit = self.options.ba_local_num_images.saturating_sub(1);
        let mut bundle: Vec<usize> = overlapping
            .into_iter()
            .take(limit)
            .map(|(image, _)| image)
            .collect();
        bundle.push(image);
        bundle
    }

    fn adjust_local_bundle(
        &mut self,
        pipeline: PipelineOptions,
        images: &[usize],
        loss: Loss,
    ) {
        if images.is_empty() {
            return;
        }
        let point_ids: Vec<usize> = (0..self.recon.points.len()).collect();
        self.run_bundle_adjustment(images, &point_ids, loss, Gauge::ThreePoints, pipeline);
    }

    /// 4.2.1 `AdjustGlobalBundle`：全部已注册图像 + 全部点，`TWO_CAMS_FROM_WORLD`。
    pub fn adjust_global_bundle(&mut self, pipeline: PipelineOptions, quick: bool) -> bool {
        let images: Vec<usize> = self.recon.registered.clone();
        if images.len() < 2 {
            return false;
        }
        let point_ids: Vec<usize> = (0..self.recon.points.len()).collect();
        let iterations = pipeline.ba_global_max_num_iterations * if quick { 2 } else { 1 };
        let summary = self.run_bundle_adjustment(
            &images,
            &point_ids,
            Loss::Trivial,
            Gauge::TwoCamsFromWorld,
            PipelineOptions {
                ba_global_max_num_iterations: iterations,
                ..pipeline
            },
        );
        summary
    }

    /// 4.2.1 `IterativeGlobalRefinement`：补全/合并 → 重三角化 → 全局 BA →
    /// 归一化 → 补全/合并 + 过滤，变化率 < 阈值即停。
    pub fn iterative_global_refinement(&mut self, pipeline: PipelineOptions, rng: &mut Rng) {
        self.triangulator.complete_and_merge_tracks(
            &self.graph,
            &mut self.recon,
            &mut self.obs,
            &self.tri_options,
        );
        self.triangulator.retriangulate(
            &self.graph,
            &mut self.recon,
            &mut self.obs,
            &self.tri_options,
            rng,
        );
        for _ in 0..pipeline.ba_global_max_refinements {
            let before = self.recon.compute_num_observations();
            self.adjust_global_bundle(pipeline, false);
            self.recon.normalize(10.0);
            let changed = self.triangulator.complete_and_merge_tracks(
                &self.graph,
                &mut self.recon,
                &mut self.obs,
                &self.tri_options,
            ) + self.filter_points();
            if (changed as f64) / (before.max(1) as f64) < pipeline.ba_global_max_refinement_change {
                break;
            }
        }
        self.triangulator.clear_modified();
    }

    /// 把选定图像/点转成 M6 的 BA 问题，跑完再写回。
    fn run_bundle_adjustment(
        &mut self,
        images: &[usize],
        points: &[usize],
        loss: Loss,
        gauge: Gauge,
        pipeline: PipelineOptions,
    ) -> bool {
        // 相机（本项目通常 1 台；按需精化内参）。
        let mut ba_cameras: Vec<BaCamera> = self
            .recon
            .cameras
            .iter()
            .map(|camera| BaCamera {
                id: camera.id,
                model_id: camera.model_id,
                params: camera.params.clone(),
                constant: false,
            })
            .collect();
        // 图像：只放 bundle 内的已注册图像。
        let mut ba_images: Vec<BaImage> = Vec::new();
        let mut image_row_of: Vec<Option<usize>> = vec![None; self.recon.num_images()];
        for image in images {
            if !self.recon.images[*image].registered {
                continue;
            }
            let row = ba_images.len();
            image_row_of[*image] = Some(row);
            ba_images.push(BaImage {
                id: self.recon.images[*image].id,
                camera: self.recon.images[*image].camera,
                qvec: self.recon.images[*image].qvec,
                tvec: self.recon.images[*image].tvec,
                constant_pose: self.options.fix_existing_frames
                    && self.existing_images.contains(image),
            });
        }
        if ba_images.is_empty() {
            return false;
        }
        // 点：只放被 bundle 图像观测到、且未被过滤的点。
        let mut ba_points: Vec<BaPoint> = Vec::new();
        let mut point_row_of: Vec<Option<usize>> = vec![None; self.recon.num_points3d()];
        for point in points {
            if *point >= self.recon.points.len() {
                continue;
            }
            let mut observations = Vec::new();
            for element in &self.recon.points[*point].track {
                let image = self.recon.image_index[&element.image_id];
                let Some(row) = image_row_of[image] else {
                    continue;
                };
                let observation = self.recon.images[image].points2d[element.point2d_idx as usize];
                observations.push((row, observation.x, observation.y));
            }
            if observations.len() < 2 {
                continue;
            }
            point_row_of[*point] = Some(ba_points.len());
            ba_points.push(BaPoint {
                xyz: self.recon.points[*point].xyz,
                observations,
                constant: false,
            });
        }
        if ba_points.is_empty() {
            return false;
        }
        let options = super::ba::BaOptions {
            loss,
            max_iterations: if gauge == Gauge::ThreePoints {
                pipeline.ba_local_max_num_iterations
            } else {
                pipeline.ba_global_max_num_iterations
            },
            refine_focal_length: self.options.refine_focal_length,
            refine_principal_point: self.options.refine_principal_point,
            refine_extra_params: self.options.refine_extra_params,
            gauge,
            function_tolerance: 0.0,
            gradient_tolerance: if gauge == Gauge::ThreePoints {
                pipeline.ba_local_gradient_tolerance
            } else {
                pipeline.ba_global_gradient_tolerance
            },
        };
        let summary = run_bundle_adjustment(&mut ba_cameras, &mut ba_images, &mut ba_points, &options);
        if !summary.cost.is_finite() {
            return false;
        }
        // 写回。
        for (index, camera) in self.recon.cameras.iter_mut().enumerate() {
            camera.params = ba_cameras[index].params.clone();
        }
        for (image, row) in image_row_of.iter().enumerate() {
            let Some(row) = *row else { continue };
            self.recon.images[image].qvec = ba_images[row].qvec;
            self.recon.images[image].tvec = ba_images[row].tvec;
        }
        for (point, row) in point_row_of.iter().enumerate() {
            let Some(row) = *row else { continue };
            self.recon.points[point].xyz = ba_points[row].xyz;
            self.recon.points[point].error = -1.0;
        }
        true
    }

    // ------------------------------------------------------------ 产物

    /// 更新每点误差（4.2.1 `UpdatePoint3DErrors`）。
    pub fn update_point_errors(&mut self) {
        self.recon.update_point3d_errors();
    }

    /// 写 `sparse/<model_id>/`（`cameras.bin` / `images.bin` / `points3D.bin`）。
    pub fn write_model(&self, directory: &Path) -> Result<()> {
        let (cameras, images, points) = self.recon.export_model();
        super::format::model::write_model(directory, &cameras, &images, &points)
    }
}

// ---------------------------------------------------------------- 主循环

/// 主循环返回值。
#[derive(Clone, Debug)]
pub struct RunReport {
    pub num_reg_images: usize,
    pub num_points3d: usize,
    pub num_cameras: usize,
}

/// 4.2.1 `IncrementalPipeline::Reconstruct` 主循环。
pub fn run(mapper: &mut Mapper, sink: &ProgressSink, rng: &mut Rng) -> Result<RunReport> {
    let pipeline = PipelineOptions::default();
    sink.log("=> Finding good initial image pair");
    let Some((image1, image2)) = mapper.find_initial_image_pair(rng) else {
        return Err(SplatError::Process(
            "找不到可用的初始图像对（内点或三角角不足）".into(),
        ));
    };
    sink.log(format!(
        "Registering initial image pair #{} and #{}",
        mapper.recon.images[image1].id,
        mapper.recon.images[image2].id
    ));
    if !mapper.register_initial_image_pair(image1, image2, rng)? {
        return Err(SplatError::Process("初始图像对初始化失败".into()));
    }
    sink.log(format!(
        "=> Initial model: {} registered, {} points",
        mapper.num_reg_images(),
        mapper.num_points3d()
    ));

    let mut ba_prev_frames = mapper.num_reg_images();
    let mut ba_prev_points = mapper.num_points3d();
    let mut success = true;
    let mut previous_success = true;
    loop {
        sink.check_cancelled()?;
        previous_success = success;
        success = false;
        let candidates = mapper.find_next_images();
        for (index, image) in candidates.iter().enumerate() {
            sink.log(format!(
                "Registering image #{} (num_reg_frames={})",
                mapper.recon.images[*image].id,
                mapper.num_reg_images()
            ));
            sink.log(format!(
                "=> Image sees {} / {} points",
                mapper.obs.num_visible_points3D(*image),
                mapper.obs.num_observations(*image)
            ));
            if mapper.register_next_image(*image, rng)? {
                success = true;
                break;
            }
            sink.log("=> Could not register, trying another image.");
            if index + 1 >= pipeline.min_num_initial_reg_trials
                && mapper.num_reg_images() < pipeline.min_model_size
            {
                break;
            }
        }
        if std::env::var("OOO_MAP_DEBUG").is_ok() {
            eprintln!(
                "[loop] success={success} 已注册 {} 图像 / {} 点",
                mapper.num_reg_images(),
                mapper.num_points3d()
            );
        }
        if success {
            let image = mapper.recon.registered.last().copied().expect("刚注册过");
            mapper.triangulate_image(image, rng);
            mapper.iterative_local_refinement(pipeline, image);
            // 4.2.1 `CheckRunGlobalRefinement`。
            let triggered = mapper.num_reg_images() as f64
                >= pipeline.ba_global_frames_ratio * ba_prev_frames as f64
                || mapper.num_reg_images() >= pipeline.ba_global_frames_freq + ba_prev_frames
                || mapper.num_points3d() as f64
                    >= pipeline.ba_global_points_ratio * ba_prev_points as f64
                || mapper.num_points3d() >= pipeline.ba_global_points_freq + ba_prev_points;
            if triggered {
                mapper.iterative_global_refinement(pipeline, rng);
                ba_prev_frames = mapper.num_reg_images();
                ba_prev_points = mapper.num_points3d();
            }
        } else if previous_success {
            // 连续两轮失败前的一轮：再跑一次全局精化（4.2.1 pipeline:716-723）。
            mapper.iterative_global_refinement(pipeline, rng);
        }
        if !(success || previous_success) {
            break;
        }
    }

    // 收尾：最后一次全局精化（4.2.1 pipeline:730-737）。
    if mapper.num_reg_images() > 0
        && (mapper.num_reg_images() != ba_prev_frames || mapper.num_points3d() != ba_prev_points)
    {
        mapper.iterative_global_refinement(pipeline, rng);
    }
    mapper.update_point_errors();
    if mapper.num_reg_images() == 0 {
        return Err(SplatError::Process("没有图像被注册".into()));
    }
    sink.log(format!(
        "=> Reconstruction: {} registered images, {} points",
        mapper.num_reg_images(),
        mapper.num_points3d()
    ));
    Ok(RunReport {
        num_reg_images: mapper.num_reg_images(),
        num_points3d: mapper.num_points3d(),
        num_cameras: mapper.recon.cameras.len(),
    })
}
// ---------------------------------------------------------------- 阶段入口

/// 三个入口共用的请求（对应 `colmap mapper` 的参数组合）。
pub struct MappingRequest {
    pub database: std::path::PathBuf,
    /// 模型输出目录（`--output_path`）：写 `sparse/0/{cameras,images,points3D}.bin`。
    pub output: std::path::PathBuf,
    /// 图像列表（`--Mapper.image_list_path`；`None` = 全部图像）。
    pub image_list: Option<std::path::PathBuf>,
    /// `--input_path`：续建用的既有模型目录。
    pub input_model: Option<std::path::PathBuf>,
    /// `--Mapper.tri_ignore_two_view_tracks 0`（允许两视图 track）。
    pub allow_two_view_tracks: bool,
    /// `--Mapper.fix_existing_frames 1`（续建时固定既有位姿）。
    pub fix_existing_frames: bool,
    /// 内参精化开关（`--Mapper.ba_refine_* 0`）。
    pub refine_intrinsics: bool,
}

/// 阶段实现：加载 → 增量重建 → 写 `sparse/<n>/`。
pub fn map_sparse(sink: &ProgressSink, request: &MappingRequest) -> Result<RunReport> {
    sink.stage(
        "mapping",
        0,
        Some(1),
        "正在增量重建相机轨迹".to_string(),
    );
    let mut rng = Rng::new(0x6f6f_5f73_666d);
    let mut mapper = load(&request.database, None)?;
    mapper.options.fix_existing_frames = request.fix_existing_frames;
    mapper.options.refine_focal_length = request.refine_intrinsics;
    mapper.options.refine_principal_point = false;
    mapper.options.refine_extra_params = request.refine_intrinsics;
    mapper.options.abs_pose_refine_focal_length = request.refine_intrinsics;
    mapper.options.abs_pose_refine_extra_params = request.refine_intrinsics;
    mapper.tri_options.ignore_two_view_tracks = !request.allow_two_view_tracks;

    // `--input_path`：载入既有模型并固定其位姿。
    let mut continuation = false;
    if let Some(input_model) = &request.input_model {
        continuation = load_existing_model(&mut mapper, input_model)?;
        sink.log(format!(
            "=> Continuing from existing model: {} registered images",
            mapper.num_reg_images()
        ));
    } else if let Some(image_list) = &request.image_list {
        // `image_list` 限定新图像（续建场景）：不在列表里的图像不参与。
        let names = read_image_list(image_list)?;
        let keep: HashSet<String> = names.into_iter().collect();
        let mut filtered = load(&request.database, Some(&keep))?;
        filtered.options = mapper.options;
        filtered.tri_options = mapper.tri_options;
        mapper = filtered;
    }

    if !continuation {
        let report = run(&mut mapper, sink, &mut rng)?;
        mapper.write_model(&request.output.join("0"))?;
        sink.stage("mapping", 1, Some(1), "增量重建完成".to_string());
        return Ok(report);
    }

    // 续建：既有位姿固定，只注册新图像。
    continuation_run(&mut mapper, sink, &mut rng, &request.output.join("0"))
}

/// 读 `--Mapper.image_list_path`（每行一个图像名，忽略空行）。
fn read_image_list(path: &Path) -> Result<HashSet<String>> {
    let content = std::fs::read_to_string(path).map_err(|error| {
        SplatError::Process(format!("读取图像列表 {} 失败：{error}", path.display()))
    })?;
    Ok(content
        .lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect())
}

/// 从 `sparse/<n>/{cameras,images,points3D}.bin` 载入既有模型并注册之。
fn load_existing_model(mapper: &mut Mapper, input_model: &Path) -> Result<bool> {
    // `input_path` 既可能是模型目录，也可能是包含 `0/` 的 sparse 根。
    let directory = if input_model.join("images.bin").exists() {
        input_model.to_path_buf()
    } else {
        let candidate = input_model.join("0");
        if candidate.join("images.bin").exists() {
            candidate
        } else {
            return Err(SplatError::Process(format!(
                "既有模型 {} 里没有 images.bin",
                input_model.display()
            )));
        }
    };
    let cameras = super::format::model::read_cameras(&directory.join("cameras.bin"))?;
    let images = super::format::model::read_images(&directory.join("images.bin"))?;
    let points = super::format::model::read_points3d(&directory.join("points3D.bin"))?;

    // 相机：以库里的为准（参数更精细），既有模型只补同名相机的位姿无关项。
    for camera in &cameras {
        let Some(index) = mapper.recon.cameras.iter().position(|item| item.id == camera.id)
        else {
            continue;
        };
        mapper.recon.cameras[index].params = camera.params.clone();
    }
    let mut registered = 0usize;
    for image in &images {
        let Some(index) = mapper.recon.image_index.get(&image.id).copied() else {
            continue;
        };
        if mapper.recon.images[index].points2d.len() != image.points2d.len() {
            continue;
        }
        // 特征点坐标以数据库为准（三维点下标按轨道重建）。
        mapper.obs.register_image(
            &mapper.graph,
            &mut mapper.recon,
            index,
            image.qvec,
            image.tvec,
        );
        mapper.existing_images.insert(index);
        registered += 1;
    }
    // 既有轨道：按 point3D 的 track 逐条挂回（依赖上面的 point3D 指针）。
    for point in &points {
        let index = match mapper.recon.points.len() {
            0 => break,
            _ => (point.id as usize).saturating_sub(1),
        };
        if index >= mapper.recon.points.len() {
            continue;
        }
        mapper.recon.points[index] = point.clone();
        let track = point.track.clone();
        let Some(first) = track.first().copied() else {
            continue;
        };
        let Some(image) = mapper.recon.image_index.get(&first.image_id).copied() else {
            continue;
        };
        mapper.obs.set_observation_as_triangulated(
            &mapper.graph,
            &mapper.recon,
            image,
            first.point2d_idx,
            false,
        );
    }
    Ok(registered > 0)
}

/// 续建主循环（固定既有位姿，只注册新图像）。
fn continuation_run(
    mapper: &mut Mapper,
    sink: &ProgressSink,
    rng: &mut Rng,
    output: &Path,
) -> Result<RunReport> {
    let pipeline = PipelineOptions::default();
    let mut success = true;
    let mut previous = true;
    loop {
        sink.check_cancelled()?;
        previous = success;
        success = false;
        for image in mapper.find_next_images() {
            sink.log(format!(
                "Registering image #{} (num_reg_frames={})",
                mapper.recon.images[image].id,
                mapper.num_reg_images()
            ));
            if mapper.register_next_image(image, rng)? {
                success = true;
                break;
            }
        }
        if std::env::var("OOO_MAP_DEBUG").is_ok() {
            eprintln!(
                "[loop] success={success} 已注册 {} 图像 / {} 点",
                mapper.num_reg_images(),
                mapper.num_points3d()
            );
        }
        if success {
            let image = mapper.recon.registered.last().copied().expect("刚注册过");
            mapper.triangulate_image(image, rng);
            mapper.iterative_local_refinement(pipeline, image);
        }
        if !(success || previous) {
            break;
        }
    }
    mapper.iterative_global_refinement(pipeline, rng);
    mapper.update_point_errors();
    if mapper.num_reg_images() == 0 {
        return Err(SplatError::Process("没有图像被注册".into()));
    }
    mapper.write_model(output)?;
    Ok(RunReport {
        num_reg_images: mapper.num_reg_images(),
        num_points3d: mapper.num_points3d(),
        num_cameras: mapper.recon.cameras.len(),
    })
}

// ---------------------------------------------------------------- 测试

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::colmap::apple::ba::CameraModelId;
    use crate::engines::colmap::apple::runtime::test_sink;
    use crate::engines::colmap::apple::two_view::UNCALIBRATED;
    use tempfile::TempDir;
    use tokio_util::sync::CancellationToken;

    struct Rng(u64);
    impl Rng {
        fn uniform(&mut self, min: f64, max: f64) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            min + (max - min) * ((self.0 >> 11) as f64 / (1u64 << 53) as f64)
        }
    }

    /// 合成数据库：`num_images` 台相机绕一圈看 `num_points` 个三维点。
    ///
    /// 相机是**纯平移**（沿圆周切向），朝向固定 —— 位姿可解析验证；
    /// 每对相邻图像的匹配 = 所有共视点的精确索引对应（无误匹配）。
    fn synthetic_database(
        directory: &TempDir,
        num_images: usize,
        num_points: usize,
        seed: u64,
    ) -> Vec<([f64; 4], [f64; 3])> {
        let database = directory.path().join("database.db");
        let connection = database::create(&database).unwrap();
        let width = 1000u64;
        let height = 800u64;
        let focal = 900.0;
        let camera = Camera::new(
            1,
            CameraModelId::SimpleRadial as i32,
            width,
            height,
            vec![focal, width as f64 / 2.0, height as f64 / 2.0, 0.0],
        );
        database::write_camera(&connection, &camera, true).unwrap();

        let mut rng = Rng(seed);
        // 场景是一面沿 x 铺开的「墙」：x ∈ [-0.5, 8.2]（覆盖 0…4.2 的机位）、
        // y ∈ [-0.4, 0.4]、z ∈ [1.2, 1.8]（浅景深 → 三角角 ≈ 18°…28°）。
        //
        // 旧夹具是「点集中在 x ∈ [-1.5,1.5]、相机每帧横移 2.0」：第 4 帧起相机已经
        // 完全转到场景侧面，**看到 0 个点**，所以只能注册 3 张图（夹具的 bug，不是
        // mapper 的）。现在横向步长 0.6 + 每帧 0.02 rad 偏航，相邻共视 ≈ 100 点。
        let points: Vec<[f64; 3]> = (0..num_points)
            .map(|_| {
                [
                    rng.uniform(-0.5, 8.2),
                    rng.uniform(-0.4, 0.4),
                    rng.uniform(1.2, 1.8),
                ]
            })
            .collect();
        // 相机**横向平移 + 轻微偏航**（像手持侧向拍摄）。
        // 两个细节缺一不可：
        // 1) 必须有横向分量：纯前向平移 `|t_z| = 1` 会被
        //    `init_max_forward_motion = 0.95` 直接否掉（4.2.1 同款判定）；
        // 2) 必须有旋转：`R = I` 的纯平移下 `E = [t]×` 只确定**方向的正负**
        //    （`[−t]× = −[t]×`），两种符号都可能全部 cheiral → 基线方向二义。
        let poses: Vec<([f64; 4], [f64; 3])> = (0..num_images)
            .map(|index| {
                let yaw = 0.02 * index as f64;
                let (sin, cos) = yaw.sin_cos();
                let rotation = [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]];
                let center = [0.6 * index as f64, 0.0, 0.0];
                let translation = [
                    -(rotation[0][0] * center[0]
                        + rotation[0][1] * center[1]
                        + rotation[0][2] * center[2]),
                    -(rotation[1][0] * center[0]
                        + rotation[1][1] * center[1]
                        + rotation[1][2] * center[2]),
                    -(rotation[2][0] * center[0]
                        + rotation[2][1] * center[1]
                        + rotation[2][2] * center[2]),
                ];
                (
                    super::super::linalg::rotation_to_quaternion(&rotation),
                    translation,
                )
            })
            .collect();
        // 写图像 + 特征点。
        let mut image_ids = Vec::new();
        for index in 0..num_images {
            let name = format!("frame_{index:04}.jpg");
            let image_id = database::insert_image(&connection, &name, camera.id).unwrap();
            image_ids.push(image_id);
            let mut keypoints = Vec::new();
            for xyz in &points {
                let pose = poses[index];
                let pixel = camera::world_to_img(
                    &camera,
                    CameraModelId::SimpleRadial,
                    &pose.0,
                    &pose.1,
                    *xyz,
                )
                .unwrap();
                keypoints.push(database::Keypoint::upright(pixel[0] as f32, pixel[1] as f32));
            }
            database::write_keypoints(&connection, image_id, &keypoints).unwrap();
        }
        // 共视关系（哪些点能被两台相机同时看到）。
        let visible: Vec<Vec<bool>> = poses
            .iter()
            .map(|pose| {
                points
                    .iter()
                    .map(|xyz| {
                        camera::world_to_img(
                            &camera,
                            CameraModelId::SimpleRadial,
                            &pose.0,
                            &pose.1,
                            *xyz,
                        )
                        .map(|pixel| {
                            pixel[0] > 8.0 && pixel[0] < 992.0 && pixel[1] > 8.0 && pixel[1] < 792.0
                        })
                        .unwrap_or(false)
                    })
                    .collect()
            })
            .collect();
        // 匹配 + F（用真值相对位姿构造 E 再转 F，保证初始化可用）。
        for i in 0..num_images {
            for j in (i + 1)..num_images {
                let mut matches: Vec<[u32; 2]> = Vec::new();
                for point in 0..num_points {
                    if visible[i][point] && visible[j][point] {
                        matches.push([point as u32, point as u32]);
                    }
                }
                if matches.len() < 60 {
                    continue;
                }
                database::write_matches(&connection, image_ids[i], image_ids[j], &matches)
                    .unwrap();
                let f = fundamental_between(&camera, &poses[i], &poses[j], &matches, &points);
                let geometry = database::TwoViewGeometry {
                    config: UNCALIBRATED,
                    inlier_matches: matches,
                    f: Some(f),
                    e: None,
                    h: None,
                    cam2_from_cam1: None,
                };
                database::write_two_view_geometry(&connection, image_ids[i], image_ids[j], &geometry)
                    .unwrap();
            }
        }
        drop(connection);
        poses
    }

    /// 由真值相对位姿构造像素域的 F（`F = K⁻ᵀ E K⁻¹`）。
    fn fundamental_between(
        camera: &Camera,
        pose1: &([f64; 4], [f64; 3]),
        pose2: &([f64; 4], [f64; 3]),
        matches: &[[u32; 2]],
        points: &[[f64; 3]],
    ) -> [[f64; 3]; 3] {
        // E = [t_rel]ₓ，t_rel = R1ᵀ(t2 − t1)（R 为 `cam_from_world`）。
        let r1 = super::super::linalg::quaternion_to_rotation_matrix(&pose1.0);
        let delta = [
            pose2.1[0] - pose1.1[0],
            pose2.1[1] - pose1.1[1],
            pose2.1[2] - pose1.1[2],
        ];
        let t = [
            r1[0][0] * delta[0] + r1[1][0] * delta[1] + r1[2][0] * delta[2],
            r1[0][1] * delta[0] + r1[1][1] * delta[1] + r1[2][1] * delta[2],
            r1[0][2] * delta[0] + r1[1][2] * delta[1] + r1[2][2] * delta[2],
        ];
        let e = [[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]];
        let k = camera::calibration_matrix(camera);
        let k_inv = inverse3(k);
        let mut k_inv_t = k_inv;
        for row in 0..3 {
            for col in 0..row {
                let value = k_inv_t[row][col];
                k_inv_t[row][col] = k_inv_t[col][row];
                k_inv_t[col][row] = value;
            }
        }
        let mut f = [[0.0f64; 3]; 3];
        let tmp = mat3_mul(&k_inv_t, &mat3_mul(&e, &k_inv));
        for row in 0..3 {
            for col in 0..3 {
                f[row][col] = tmp[row][col];
            }
        }
        // 归一化（4.2.1 也只存归一化后的矩阵）。
        let norm = (f[0][0] * f[0][0] + f[1][1] * f[1][1]).sqrt().max(1e-12);
        for row in 0..3 {
            for col in 0..3 {
                f[row][col] /= norm;
            }
        }
        let _ = (matches, points);
        f
    }

    fn inverse3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        let inv = 1.0 / det;
        let mut out = [[0.0f64; 3]; 3];
        out[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv;
        out[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv;
        out[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv;
        out[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv;
        out[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv;
        out[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv;
        out[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv;
        out[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv;
        out[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv;
        out
    }

    fn normalize3(v: [f64; 3]) -> [f64; 3] {
        let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / norm, v[1] / norm, v[2] / norm]
    }

    fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let mut out = [[0.0f64; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += a[row][k] * b[k][col];
                }
                out[row][col] = sum;
            }
        }
        out
    }

    /// 端到端：合成 8 图环形场景 → 全图注册 → 写 `sparse/0` → 读回校验。
    ///
    /// 依赖链：初始对选择 → 相对位姿 → 初始三角化 → 初始全局 BA → **注册新图像**
    /// （`pose.rs` 的 LO-RANSAC + EPnP）→ 三角化 → 局部/全局 BA。
    #[test]
    fn reconstructs_a_synthetic_ring_scene() {
        let directory = TempDir::new().unwrap();
        let num_images = 8;
        let num_points = 800;
        synthetic_database(&directory, num_images, num_points, 0x5eed);
        let request = MappingRequest {
            database: directory.path().join("database.db"),
            output: directory.path().join("sparse"),
            image_list: None,
            input_model: None,
            allow_two_view_tracks: false,
            fix_existing_frames: false,
            refine_intrinsics: true,
        };
        let sink = test_sink(CancellationToken::new()).unwrap();
        let report = map_sparse(&sink, &request).unwrap();
        assert_eq!(
            report.num_reg_images, num_images,
            "全部图像都应注册：{report:?}"
        );
        assert!(report.num_points3d > 100, "点数 {report:?}");

        // 产物可读回，且点误差（像素级平均重投影误差）足够小。
        let model_dir = directory.path().join("sparse").join("0");
        let images = super::super::format::model::read_images(&model_dir.join("images.bin")).unwrap();
        let points =
            super::super::format::model::read_points3d(&model_dir.join("points3D.bin")).unwrap();
        assert_eq!(images.len(), num_images);
        assert_eq!(points.len(), report.num_points3d);
        let mean_error: f64 = points.iter().map(|point| point.error).sum::<f64>()
            / points.len().max(1) as f64;
        assert!(
            mean_error < 1.0,
            "平均重投影误差应 < 1 像素，实际 {mean_error}"
        );
        // 相机中心应**共线且等距**（横向平移轨迹）。
        //
        // 注意不能断言「y/z 分量为 0」：重建只确定到 7 自由度相似变换，
        // 初始对是图像 (5,6) 而不是 (0,1)，所以解出来的坐标系相对真值世界系
        // 有一个未知的旋转 —— 旧断言在这里是错的。这里改成规范（gauge-free）检查。
        let centers: Vec<[f64; 3]> = images
            .iter()
            .map(|image| camera::projection_center(&image.qvec, &image.tvec))
            .collect();
        let centroid = {
            let mut sum = [0.0f64; 3];
            for center in &centers {
                for axis in 0..3 {
                    sum[axis] += center[axis];
                }
            }
            [sum[0] / centers.len() as f64, sum[1] / centers.len() as f64, sum[2] / centers.len() as f64]
        };
        let scatter = {
            let mut covariance = nalgebra::Matrix3::<f64>::zeros();
            for center in &centers {
                let d = [center[0] - centroid[0], center[1] - centroid[1], center[2] - centroid[2]];
                for (r, dr) in d.iter().enumerate() {
                    for (c, dc) in d.iter().enumerate() {
                        covariance[(r, c)] += dr * dc;
                    }
                }
            }
            covariance
        };
        let svd = nalgebra::linalg::SVD::try_new(scatter, true, true, 1e-12, 200)
            .expect("中心散布矩阵的 SVD");
        let u = svd.u.expect("SVD 的 U");
        let axis = [u[(0, 0)], u[(1, 0)], u[(2, 0)]]; // 最大方差方向
        let mut offsets: Vec<f64> = Vec::with_capacity(centers.len());
        let mut offsets_sq: Vec<f64> = Vec::with_capacity(centers.len());
        for center in &centers {
            let d = [
                center[0] - centroid[0],
                center[1] - centroid[1],
                center[2] - centroid[2],
            ];
            let along = (0..3).map(|k| d[k] * axis[k]).sum::<f64>();
            offsets.push(along);
            offsets_sq.push((0..3).map(|k| d[k] * d[k]).sum::<f64>() - along * along);
        }
        let order: Vec<usize> = {
            let mut order: Vec<usize> = (0..offsets.len()).collect();
            order.sort_by(|&a, &b| offsets[a].partial_cmp(&offsets[b]).expect("投影有限"));
            order
        };
        let spacing = (offsets[order[order.len() - 1]] - offsets[order[0]])
            / (centers.len() - 1) as f64;
        for (&index, &perpendicular_sq) in order.iter().zip(offsets_sq.iter()) {
            assert!(
                perpendicular_sq.sqrt() < 0.02 * spacing,
                "相机中心应共线：偏离 {:.4}（步长 {spacing:.4}）",
                perpendicular_sq.sqrt()
            );
        }
        for pair in order.windows(2) {
            let step = offsets[pair[1]] - offsets[pair[0]];
            assert!(
                (step / spacing - 1.0).abs() < 0.05,
                "相机间距应等距：{spacing:.4} vs {step:.4}"
            );
        }
        let _ = &centers;
    }

    #[test]
    fn image_without_keypoints_does_not_panic_during_load() {
        // 真实数据处理常见：某帧空白/全模糊 → 0 个特征点。ObservationManager 构造
        // 时曾用 `graph.find(image, 0)`“占位”，对这类图像直接越界 panic。
        let directory = TempDir::new().unwrap();
        let connection = database::create(&directory.path().join("database.db")).unwrap();
        let camera = Camera::new(
            1,
            CameraModelId::SimpleRadial as i32,
            100,
            80,
            vec![90.0, 50.0, 40.0, 0.0],
        );
        database::write_camera(&connection, &camera, true).unwrap();
        let with_features = database::insert_image(&connection, "a.jpg", camera.id).unwrap();
        database::write_keypoints(
            &connection,
            with_features,
            &[database::Keypoint::upright(10.0, 10.0)],
        )
        .unwrap();
        let empty = database::insert_image(&connection, "empty.jpg", camera.id).unwrap();
        database::write_keypoints(&connection, empty, &[]).unwrap();
        drop(connection);

        let request = MappingRequest {
            database: directory.path().join("database.db"),
            output: directory.path().join("sparse"),
            image_list: None,
            input_model: None,
            allow_two_view_tracks: false,
            fix_existing_frames: false,
            refine_intrinsics: true,
        };
        let sink = test_sink(CancellationToken::new()).unwrap();
        // 没有任何几何对：应干净地报“找不到初始图像对”，而不是越界 panic。
        let error = map_sparse(&sink, &request).unwrap_err();
        assert!(
            error.to_string().contains("初始图像对"),
            "错误信息应指向初始对：{error}"
        );
    }

    #[test]
    fn missing_two_view_geometry_fails_cleanly() {
        // 没有几何对时必须给出明确错误（而不是 panic 或空模型）。
        let directory = TempDir::new().unwrap();
        let connection = database::create(&directory.path().join("database.db")).unwrap();
        let camera = Camera::new(
            1,
            CameraModelId::SimpleRadial as i32,
            1000,
            800,
            vec![900.0, 500.0, 400.0, 0.0],
        );
        database::write_camera(&connection, &camera, true).unwrap();
        let image_id = database::insert_image(&connection, "a.jpg", camera.id).unwrap();
        database::write_keypoints(&connection, image_id, &[database::Keypoint::upright(10.0, 10.0)])
            .unwrap();
        drop(connection);
        let request = MappingRequest {
            database: directory.path().join("database.db"),
            output: directory.path().join("sparse"),
            image_list: None,
            input_model: None,
            allow_two_view_tracks: false,
            fix_existing_frames: false,
            refine_intrinsics: true,
        };
        let sink = test_sink(CancellationToken::new()).unwrap();
        let error = map_sparse(&sink, &request).unwrap_err();
        assert!(
            error.to_string().contains("初始图像对"),
            "错误信息应指向初始对：{error}"
        );
    }

    /// 临时诊断：直接对真实数据库跑 mapper（不经过 runtime 的 catch_unwind），
    /// 让 panic 带出位置与回溯。用法：
    /// `OOO_MAP_DEBUG_DB=<database.db> RUST_BACKTRACE=1 cargo test --features backend-ooo-sfm --lib debug_real_database -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn debug_real_database() {
        let Ok(database) = std::env::var("OOO_MAP_DEBUG_DB") else {
            eprintln!("未设置 OOO_MAP_DEBUG_DB");
            return;
        };
        let directory = TempDir::new().unwrap();
        let request = MappingRequest {
            database: std::path::PathBuf::from(database),
            output: directory.path().to_path_buf(),
            image_list: None,
            input_model: None,
            allow_two_view_tracks: false,
            fix_existing_frames: false,
            refine_intrinsics: true,
        };
        let sink = test_sink(CancellationToken::new()).unwrap();
        match map_sparse(&sink, &request) {
            Ok(report) => eprintln!("完成：{report:?}"),
            Err(error) => eprintln!("失败：{error}"),
        }
    }

}

