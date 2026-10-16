//! 特征提取（M3）：自研 SIFT + COLMAP 兼容的 `database.db` 落库。
//!
//! 行为对齐 COLMAP 4.2.1 的 `feature_extractor`
//! （`controllers/image_reader.cc` + `controllers/feature_extraction.cc`）：
//!
//! - 图像按名字排序（给 `--image_list_path` 时按列表内容排序 —— COLMAP 也会排序）；
//! - 已存在且键点、描述子齐全的图像直接跳过（幂等续跑）；半成品先清掉旧特征再重提；
//! - `single_camera`：所有图像共用一个 `SIMPLE_RADIAL` 相机；首张成功图像优先采用
//!   EXIF 焦点长度（`prior_focal_length = 1`），没有则 `f = 1.2 × max(w, h)`
//!   （`prior_focal_length = 0`）；`cx = w/2`、`cy = h/2`、`k = 0`；
//! - `existing_camera_id`（增量重拍）：复用既有相机，不新建、不改尺寸；
//! - 图像读不出来 / 掩码缺失 / 尺寸与单相机不一致：warning 后跳过该图
//!   （COLMAP 的 `BITMAP_ERROR` / `MASK_ERROR` / `CAMERA_SINGLE_DIM_ERROR` 同样只跳过）；
//! - 掩码取 `<mask_path>/<name>.png`，退化到 `<mask_path>/<stem>.png`；黑色像素
//!   （灰度 0）处的特征点被剔除（COLMAP `MaskFeatures`，按整数坐标取样）；
//! - 缩略图：长边缩到 `max_image_size`，特征点坐标再还原到相机尺寸
//!   （COLMAP `Bitmap::Thumbnail` + `ScaleKeypoints`）；
//! - 每张图像写 `images` / `keypoints` / `descriptors` / `frames` + `frame_data`
//!   （4.2.1 的特征提取本身就会写 rig/frame）。

use std::path::{Path, PathBuf};

use rayon::prelude::*;
use rusqlite::Connection;

use crate::error::{Result, SplatError};

use super::{
    exif,
    format::{
        camera_model,
        database::{self, Keypoint},
        model::Camera,
    },
    runtime::ProgressSink,
    sift::{self, GrayImage, SiftOptions},
};

/// 进度 `phase`（前端 `RuntimePanel` 的文案键，M7 补文案；未知键回退成"运行中"）。
pub const PHASE: &str = "featureExtraction";

/// COLMAP `ImageReader.default_focal_length_factor`。
const DEFAULT_FOCAL_LENGTH_FACTOR: f64 = 1.2;
/// COLMAP `SiftExtractionOptions::max_num_features` 默认值。
const DEFAULT_MAX_FEATURES: u32 = 8192;
/// 每个并行批次的目标内存预算（柏格：上采样金字塔 6 张 f32 平面）。
const MEMORY_BUDGET_BYTES: usize = 2 * 1024 * 1024 * 1024;
/// 探测不到图像尺寸时的保守估计（长边 = 该值）。
const FALLBACK_LONG_EDGE: usize = 3200;

/// COLMAP 支持、且本项目会产出的图像扩展名。
const IMAGE_EXTENSIONS: [&str; 3] = ["jpg", "jpeg", "png"];

/// 特征提取请求（对应 COLMAP `feature_extractor` 的相关参数）。
///
/// 字段是 owned（`run_job` 的闭包要求 `'static`）。
pub struct ExtractionRequest {
    pub database: PathBuf,
    pub images: PathBuf,
    pub masks: Option<PathBuf>,
    pub image_list: Option<PathBuf>,
    pub max_image_size: Option<u32>,
    pub max_features: Option<u32>,
    /// 增量重拍：复用该相机（COLMAP `--ImageReader.existing_camera_id`）。
    pub existing_camera_id: Option<u32>,
    /// 记录用：自研后端当前用 CPU（Metal 变体预留在 M6 之后）。
    pub gpu_index: Option<u32>,
}

/// 提取入口。
pub fn extract(sink: &ProgressSink, request: &ExtractionRequest) -> Result<()> {
    sink.check_cancelled()?;
    let names = collect_image_names(request)?;
    if names.is_empty() {
        return Err(SplatError::Process(format!(
            "特征提取：{} 下没有可用图像（支持 jpg/jpeg/png）",
            request.images.display()
        )));
    }
    sink.log(format!(
        "特征提取开始：图像 {} 张，database {}",
        names.len(),
        request.database.display()
    ));
    sink.log(format!(
        "  图像目录:        {}",
        request.images.display()
    ));
    if let Some(masks) = &request.masks {
        sink.log(format!("  掩码目录:        {}", masks.display()));
    }
    if let Some(list) = &request.image_list {
        sink.log(format!("  图像列表:        {}", list.display()));
    }
    if let Some(max_image_size) = request.max_image_size {
        sink.log(format!("  最大边长:        {max_image_size}"));
    }
    sink.log(format!(
        "  最多特征数:      {}",
        request.max_features.unwrap_or(DEFAULT_MAX_FEATURES)
    ));
    if let Some(gpu_index) = request.gpu_index {
        sink.log(format!(
            "  gpu_index:       {gpu_index}（自研后端当前走 CPU）"
        ));
    }

    let connection = database::open(&request.database)?;
    let options = SiftOptions {
        max_num_features: request.max_features.unwrap_or(DEFAULT_MAX_FEATURES) as usize,
        ..Default::default()
    };

    // 增量重拍：复用既有相机与其 rig。
    let mut camera = match request.existing_camera_id {
        Some(camera_id) => {
            let camera = database::read_camera(&connection, camera_id)?.ok_or_else(|| {
                SplatError::Process(format!(
                    "增量特征提取：数据库 {} 里没有相机 {camera_id}",
                    request.database.display()
                ))
            })?;
            sink.log(format!(
                "  复用相机:        #{} - {} {}×{}",
                camera.id,
                camera_model::name(camera.model_id).unwrap_or("UNKNOWN"),
                camera.width,
                camera.height
            ));
            Some(camera)
        }
        None => None,
    };
    let mut rig_id = match camera.as_ref() {
        Some(camera) => Some(database::ensure_rig_for_camera(&connection, camera.id)?),
        None => None,
    };

    let total = names.len();
    let chunk_size = parallel_chunk_size(&names, request);
    let mut index = 0usize;
    let mut features_total = 0usize;

    for chunk in names.chunks(chunk_size) {
        sink.check_cancelled()?;
        // 1) 串行查库：已完成的图像跳过（COLMAP `ImageReader::Next` 的 IMAGE_EXISTS）。
        let mut work: Vec<&String> = Vec::new();
        for name in chunk {
            if let Some((image_id, _)) = database::read_image_by_name(&connection, name)? {
                if database::has_keypoints_and_descriptors(&connection, image_id)? {
                    index += 1;
                    sink.log(format!("Processed file [{index}/{total}]"));
                    sink.log(format!("  Name:            {name}"));
                    sink.log(format!("  {name} IMAGE_EXISTS（已有特征，跳过）"));
                    sink.stage(
                        PHASE,
                        index as u64,
                        Some(total as u64),
                        format!("已处理 {index}/{total} 张图像"),
                    );
                    continue;
                }
                // 半成品（键点/描述子缺一个）：清掉旧特征重提，避免主键冲突。
                database::delete_features(&connection, image_id)?;
            }
            work.push(name);
        }

        // 2) 并行提取（每张图像内存占用可控，见 `parallel_chunk_size`）。
        let prepared: Vec<std::result::Result<Prepared, Skipped>> = work
            .par_iter()
            .map(|name| prepare_image(sink, request, name, &options))
            .collect();

        // 3) 串行写库（SQLite 单写者，与 COLMAP 的 writer 线程同构）。
        for (name, result) in work.into_iter().zip(prepared) {
            index += 1;
            sink.log(format!("Processed file [{index}/{total}]"));
            sink.log(format!("  Name:            {name}"));
            let prepared = match result {
                Ok(prepared) => prepared,
                Err(Skipped::Cancelled) => return Err(SplatError::Cancelled),
                Err(skipped) => {
                    sink.log(format!("  {name} {}", skipped.text()));
                    sink.stage(
                        PHASE,
                        index as u64,
                        Some(total as u64),
                        format!("已处理 {index}/{total} 张图像"),
                    );
                    continue;
                }
            };
            for note in &prepared.notes {
                sink.log(format!("  {name} {note}"));
            }
            let written = write_prepared(
                sink,
                &connection,
                &mut camera,
                &mut rig_id,
                &prepared,
            )?;
            if let Some(written) = written {
                features_total += written;
            }
            sink.stage(
                PHASE,
                index as u64,
                Some(total as u64),
                format!("已处理 {index}/{total} 张图像"),
            );
        }
    }
    sink.log(format!(
        "特征提取完成：{} 张图像，共 {} 个特征",
        total, features_total
    ));
    Ok(())
}

/// 一批（并行）里最多同时处理几张图像：按内存预算与 CPU 核数取小。
fn parallel_chunk_size(names: &[String], request: &ExtractionRequest) -> usize {
    let threads = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(4);
    // 用第一张图像的尺寸估算（视频帧尺寸一致；混合尺寸时只是估计，不影响正确性）。
    let (width, height) = match names.first() {
        Some(name) => {
            let path = request.images.join(name);
            image::image_dimensions(&path).unwrap_or((FALLBACK_LONG_EDGE as u32, FALLBACK_LONG_EDGE as u32))
        }
        None => (FALLBACK_LONG_EDGE as u32, FALLBACK_LONG_EDGE as u32),
    };
    let (width, height) = (width as usize, height as usize);
    let long_edge = width.max(height);
    let (width, height) = if let Some(max_image_size) = request.max_image_size {
        let max_image_size = max_image_size as usize;
        if long_edge > max_image_size {
            let scale = max_image_size as f64 / long_edge as f64;
            (
                ((width as f64 * scale).round() as usize).max(1),
                ((height as f64 * scale).round() as usize).max(1),
            )
        } else {
            (width, height)
        }
    } else {
        (width, height)
    };
    // 首八度上采样 2×；金字塔同时常驻 6 张 f32 平面。
    let per_image = 6 * 4 * (width * 2) * (height * 2);
    let by_memory = MEMORY_BUDGET_BYTES / per_image.max(1);
    by_memory.clamp(1, threads)
}

/// 单张图像的处理结果（"提取线程"部分：读图 → 掩码 → 缩略图 → SIFT）。
struct Prepared {
    name: String,
    /// 原图尺寸（= 相机尺寸）。
    width: u64,
    height: u64,
    /// 缩略图尺寸（`features` 当前的坐标系）。
    thumbnail_width: usize,
    thumbnail_height: usize,
    features: Vec<sift::SiftFeature>,
    mask: Option<Mask>,
    notes: Vec<String>,
    /// EXIF 焦点长度（像素）；`None` = 无先验（相机用默认系数建，`prior_focal_length = 0`）。
    exif_focal_length: Option<f64>,
}

/// 灰度掩码（非零 = 保留）。
struct Mask {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl Mask {
    /// COLMAP `MaskFeatures`：按**整数坐标**取样；越界视为不保留。
    fn keeps(&self, x: f32, y: f32) -> bool {
        let (x, y) = (x as isize, y as isize);
        if x < 0 || y < 0 || x >= self.width as isize || y >= self.height as isize {
            return false;
        }
        self.pixels[y as usize * self.width + x as usize] != 0
    }
}

/// 跳过该图像的原因（文案对齐 COLMAP `ImageReader::StatusToString`）。
enum Skipped {
    Bitmap(String),
    Mask(String),
    Cancelled,
}

impl Skipped {
    fn text(&self) -> String {
        match self {
            Skipped::Bitmap(reason) => format!("BITMAP_ERROR（无法读取图像：{reason}）"),
            Skipped::Mask(reason) => format!("MASK_ERROR（掩码不可用：{reason}）"),
            Skipped::Cancelled => "CANCELLED".to_string(),
        }
    }
}

/// 读图 → 掩码 → 缩略图 → SIFT（可在并行线程里跑）。
fn prepare_image(
    sink: &ProgressSink,
    request: &ExtractionRequest,
    name: &str,
    options: &SiftOptions,
) -> std::result::Result<Prepared, Skipped> {
    if sink.is_cancelled() {
        return Err(Skipped::Cancelled);
    }
    let path = request.images.join(name);
    let mut bitmap = load_gray(&path).map_err(Skipped::Bitmap)?;
    let (width, height) = (bitmap.width as u64, bitmap.height as u64);
    // EXIF 焦点长度（4.2.1 `ImageReader` 的相机分支：有效则用 EXIF，否则默认系数）。
    let exif_focal_length = exif::file_focal_length(&path, width, height);
    let mask = match &request.masks {
        Some(masks) => Some(load_mask(masks, name).map_err(Skipped::Mask)?),
        None => None,
    };
    // COLMAP `Bitmap::Thumbnail(max_image_size)`：长边缩到上限（缩小前等效低通）。
    let (width_limit, height_limit) = match request.max_image_size {
        Some(max_image_size) => {
            let max_image_size = max_image_size as usize;
            if bitmap.width <= max_image_size && bitmap.height <= max_image_size {
                (bitmap.width, bitmap.height)
            } else {
                let scale = max_image_size as f64 / bitmap.width.max(bitmap.height) as f64;
                (
                    ((bitmap.width as f64 * scale).round() as usize).max(1),
                    ((bitmap.height as f64 * scale).round() as usize).max(1),
                )
            }
        }
        None => (bitmap.width, bitmap.height),
    };
    if (width_limit, height_limit) != (bitmap.width, bitmap.height) {
        bitmap = sift::resize(&bitmap, width_limit, height_limit);
    }
    // 注意：这里的尺寸是**缩略图**尺寸，`features` 的坐标就在这个坐标系里。
    let (thumbnail_width, thumbnail_height) = (bitmap.width, bitmap.height);
    let extraction = sift::extract(&bitmap, options);
    Ok(Prepared {
        name: name.to_string(),
        width,
        height,
        thumbnail_width,
        thumbnail_height,
        features: extraction.features,
        mask,
        notes: extraction.notes,
        exif_focal_length,
    })
}

/// 写库：相机 → 图像行 → 特征点/描述子 → frame；返回写入的特征数（跳过的图返回 `None`）。
fn write_prepared(
    sink: &ProgressSink,
    connection: &Connection,
    camera: &mut Option<Camera>,
    rig_id: &mut Option<u32>,
    image: &Prepared,
) -> Result<Option<usize>> {
    // 首张成功图像建相机（COLMAP `ImageReader::Next` 的相机分支）。
    if camera.is_none() {
        // EXIF 有焦距则用它并记 `prior_focal_length = 1`（4.2.1 语义）；否则默认系数。
        let has_prior_focal_length = image.exif_focal_length.is_some();
        let focal_length = image.exif_focal_length.unwrap_or(
            DEFAULT_FOCAL_LENGTH_FACTOR * image.width.max(image.height) as f64,
        );
        let params = [
            focal_length,
            image.width as f64 / 2.0,
            image.height as f64 / 2.0,
            0.0,
        ];
        let camera_id = database::insert_camera(
            connection,
            camera_model::SIMPLE_RADIAL,
            image.width,
            image.height,
            &params,
            has_prior_focal_length,
        )?;
        sink.log(format!(
            "  Camera:          #{camera_id} - SIMPLE_RADIAL（新建，{}×{}）",
            image.width, image.height
        ));
        *camera = Some(Camera {
            id: camera_id,
            model_id: camera_model::SIMPLE_RADIAL,
            width: image.width,
            height: image.height,
            params: params.to_vec(),
        });
        *rig_id = Some(database::ensure_rig_for_camera(connection, camera_id)?);
    }
    let camera = camera.as_ref().expect("相机已在上面创建");

    // 尺寸必须与相机一致，否则跳过该图（COLMAP `CAMERA_SINGLE_DIM_ERROR`）。
    if (camera.width, camera.height) != (image.width, image.height) {
        sink.log(format!(
            "  {} CAMERA_SINGLE_DIM_ERROR（图像 {}×{} 与相机 {}×{} 不一致）",
            image.name, image.width, image.height, camera.width, camera.height
        ));
        return Ok(None);
    }

    // 图像行：已存在则复用编号（COLMAP 只在 `!ExistsImage` 时插入）。
    let image_id = match database::read_image_by_name(connection, &image.name)? {
        Some((image_id, existing_camera_id)) => {
            if existing_camera_id != camera.id {
                sink.log(format!(
                    "  {} IMAGE_CAMERA_MISMATCH（图像已在库中，相机会 #{existing_camera_id} ≠ #{})",
                    image.name, camera.id
                ));
                return Ok(None);
            }
            image_id
        }
        None => database::insert_image(connection, &image.name, camera.id)?,
    };

    // 特征点：缩略图坐标 → 相机坐标（COLMAP `ScaleKeypoints`），再按掩码剔除。
    let scale_x = camera.width as f32 / image.thumbnail_width as f32;
    let scale_y = camera.height as f32 / image.thumbnail_height as f32;
    let mut keypoints = Vec::with_capacity(image.features.len());
    let mut descriptors = Vec::with_capacity(image.features.len() * sift::DESCRIPTOR_DIM);
    for feature in &image.features {
        let x = feature.x * scale_x;
        let y = feature.y * scale_y;
        if let Some(mask) = &image.mask {
            if !mask.keeps(x, y) {
                continue;
            }
        }
        keypoints.push(Keypoint {
            x,
            y,
            shape: [
                feature.shape[0] * scale_x,
                feature.shape[1] * scale_y,
                feature.shape[2] * scale_x,
                feature.shape[3] * scale_y,
            ],
        });
        descriptors.extend_from_slice(&feature.descriptor);
    }

    database::begin_bulk(connection)?;
    database::write_keypoints(connection, image_id, &keypoints)?;
    database::write_descriptors(connection, image_id, &descriptors)?;
    if let Some(rig_id) = *rig_id {
        database::insert_frame(connection, rig_id, image_id, camera.id)?;
    }
    database::end_bulk(connection)?;

    sink.log(format!(
        "  Dimensions:      {} x {}",
        image.width, image.height
    ));
    sink.log(format!(
        "  Camera:          #{} - {}",
        camera.id,
        camera_model::name(camera.model_id).unwrap_or("UNKNOWN")
    ));
    if let Some(focal_length) = camera.params.first() {
        sink.log(format!("  Focal Length:    {focal_length:.2}px"));
    }
    sink.log(format!("  Features:        {} (SIFT)", keypoints.len()));
    if image.mask.is_some() {
        sink.log("  Mask:            Yes");
    }
    Ok(Some(keypoints.len()))
}

/// 按名字收集图像（相对路径、`/` 分隔、字典序）。
fn collect_image_names(request: &ExtractionRequest) -> Result<Vec<String>> {
    let mut names = Vec::new();
    match &request.image_list {
        Some(list) => {
            let text = std::fs::read_to_string(list).map_err(|error| {
                SplatError::Process(format!("读取图像列表 {} 失败：{error}", list.display()))
            })?;
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                names.push(line.to_string());
            }
            // COLMAP 对未排序的列表会先排序再处理（顺序即 image_id 顺序）。
            if names.windows(2).any(|pair| pair[0] > pair[1]) {
                names.sort();
            }
        }
        None => {
            collect_recursive(&request.images, &request.images, &mut names)?;
            names.sort();
        }
    }
    Ok(names)
}

/// 递归收集图像文件（相对 `root` 的路径）。
fn collect_recursive(root: &Path, directory: &Path, names: &mut Vec<String>) -> Result<()> {
    let entries = std::fs::read_dir(directory).map_err(|error| {
        SplatError::Process(format!("读取图像目录 {} 失败：{error}", directory.display()))
    })?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            SplatError::Process(format!("读取图像目录 {} 失败：{error}", directory.display()))
        })?;
        let path = entry.path();
        if entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false) {
            collect_recursive(root, &path, names)?;
            continue;
        }
        let is_image = path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| {
                let extension = extension.to_ascii_lowercase();
                IMAGE_EXTENSIONS.contains(&extension.as_str())
            })
            .unwrap_or(false);
        if !is_image {
            continue;
        }
        let relative = path.strip_prefix(root).unwrap_or(&path);
        let relative = relative
            .components()
            .map(|component| component.as_os_str().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/");
        names.push(relative);
    }
    Ok(())
}

/// 读图为灰度 `f32`（COLMAP `Bitmap::Read(as_rgb = false)` + `CloneAsGrey`）。
fn load_gray(path: &Path) -> std::result::Result<GrayImage, String> {
    let decoded = image::open(path).map_err(|error| error.to_string())?;
    let (width, height) = (decoded.width() as usize, decoded.height() as usize);
    if width == 0 || height == 0 {
        return Err("图像尺寸为 0".to_string());
    }
    // RGB（丢弃 alpha，与 COLMAP 一致）→ Rec.709 灰度，四舍五入到 u8。
    let rgb = decoded.to_rgb8();
    let mut pixels = Vec::with_capacity(width * height);
    for pixel in rgb.pixels() {
        let [r, g, b] = pixel.0;
        let gray = (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32 + 0.5) as u8;
        pixels.push(gray as f32 / 255.0);
    }
    Ok(GrayImage::new(width, height, pixels))
}

/// 读掩码：`<mask_path>/<name>.png`，退化到 `<mask_path>/<stem>.png`（COLMAP 的两种命名）。
fn load_mask(masks: &Path, name: &str) -> std::result::Result<Mask, String> {
    let direct = masks.join(format!("{name}.png"));
    let replaced = masks.join(replace_extension(name, "png"));
    let path = if direct.is_file() {
        direct
    } else if replaced.is_file() {
        replaced
    } else {
        return Err(format!(
            "找不到 {name}.png（已尝试 {} 与 {}）",
            direct.display(),
            replaced.display()
        ));
    };
    let decoded = image::open(&path).map_err(|error| format!("{}：{error}", path.display()))?;
    let gray = decoded.to_luma8();
    Ok(Mask {
        width: gray.width() as usize,
        height: gray.height() as usize,
        pixels: gray.into_raw(),
    })
}

/// 把扩展名换成 `extension`（没有扩展名时追加）。
fn replace_extension(name: &str, extension: &str) -> String {
    match name.rsplit_once('.') {
        Some((stem, _)) if !stem.is_empty() => format!("{stem}.{extension}"),
        _ => format!("{name}.{extension}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio_util::sync::CancellationToken;

    // 门面生效后，本文件的逻辑路径是 `engines::colmap::apple::*`。
    use crate::engines::colmap::apple::runtime;
    use crate::engines::colmap::apple::test_images::{
        write_image, write_jpeg_with_exif_focal_35mm,
    };

    struct Fixture {
        directory: TempDir,
        database: PathBuf,
        images: PathBuf,
    }

    impl Fixture {
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

        fn request(&self) -> ExtractionRequest {
            ExtractionRequest {
                database: self.database.clone(),
                images: self.images.clone(),
                masks: None,
                image_list: None,
                max_image_size: None,
                max_features: None,
                existing_camera_id: None,
                gpu_index: None,
            }
        }

        fn open(&self) -> Connection {
            database::open_read_only(&self.database).unwrap()
        }
    }

    fn count(connection: &Connection, sql: &str) -> i64 {
        connection.query_row(sql, [], |row| row.get(0)).unwrap()
    }

    fn sink() -> ProgressSink {
        runtime::test_sink(CancellationToken::new()).unwrap()
    }

    #[test]
    fn extracts_features_into_database() {
        let fixture = Fixture::new();
        for index in 1..=3 {
            write_image(
                &fixture.images.join(format!("frame_{index:04}.jpg")),
                160,
                120,
                0x1000 + index,
            );
        }
        let run = sink();
        extract(&run, &fixture.request()).unwrap();

        let connection = fixture.open();
        // 相机：SIMPLE_RADIAL，f = 1.2 × max(w, h)，cx = w/2，cy = h/2，k = 0。
        assert_eq!(count(&connection, "SELECT COUNT(*) FROM cameras"), 1);
        let camera = database::read_camera(&connection, 1).unwrap().unwrap();
        assert_eq!(camera.model_id, camera_model::SIMPLE_RADIAL);
        assert_eq!((camera.width, camera.height), (160, 120));
        assert_eq!(camera.params, vec![192.0, 80.0, 60.0, 0.0]);
        // 无 EXIF 的测试图 → 无先验焦距。
        assert_eq!(
            count(
                &connection,
                "SELECT prior_focal_length FROM cameras WHERE camera_id = 1"
            ),
            0
        );

        // 图像行：按名字排序、编号连续（image_id 顺序 = 处理顺序）。
        assert_eq!(count(&connection, "SELECT COUNT(*) FROM images"), 3);
        for index in 1..=3 {
            let name = format!("frame_{index:04}.jpg");
            let (image_id, camera_id) =
                database::read_image_by_name(&connection, &name).unwrap().unwrap();
            assert_eq!(image_id, index);
            assert_eq!(camera_id, 1);
        }

        // 每张图像都有键点与描述子，且行数一致。
        for image_id in 1..=3u32 {
            let keypoints = database::read_keypoints(&connection, image_id).unwrap().unwrap();
            let (rows, descriptors) =
                database::read_descriptors(&connection, image_id).unwrap().unwrap();
            assert!(
                keypoints.len() > 50,
                "合成图应检出足够多的特征：{}",
                keypoints.len()
            );
            assert_eq!(rows, keypoints.len());
            assert_eq!(descriptors.len(), keypoints.len() * sift::DESCRIPTOR_DIM);
            assert!(descriptors.iter().any(|value| *value > 0));
            for keypoint in &keypoints {
                assert!(keypoint.x >= 0.0 && keypoint.x <= 160.0);
                assert!(keypoint.y >= 0.0 && keypoint.y <= 120.0);
            }
        }

        // 每张图像一个 frame（4.2.1 的 rig/frame 兼容）。
        assert_eq!(count(&connection, "SELECT COUNT(*) FROM rigs"), 1);
        assert_eq!(count(&connection, "SELECT COUNT(*) FROM rig_sensors"), 1);
        assert_eq!(count(&connection, "SELECT COUNT(*) FROM frames"), 3);
        assert_eq!(count(&connection, "SELECT COUNT(*) FROM frame_data"), 3);
        let (rig_id, ref_sensor_id): (i64, i64) = connection
            .query_row(
                "SELECT rig_id, ref_sensor_id FROM rigs",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(rig_id, 1);
        assert_eq!(ref_sensor_id, 1);
    }

    #[test]
    fn camera_uses_exif_focal_length_and_prior_flag() {
        // EXIF `FocalLengthIn35mmFilm = 50`：f = 50 / 43.27 × 对角线，prior = 1。
        let fixture = Fixture::new();
        write_jpeg_with_exif_focal_35mm(&fixture.images.join("exif.jpg"), 200, 100, 50);
        extract(&sink(), &fixture.request()).unwrap();

        let connection = fixture.open();
        let camera = database::read_camera(&connection, 1).unwrap().unwrap();
        assert_eq!((camera.width, camera.height), (200, 100));
        let expected = 50.0 / 43.27 * (200.0_f64 * 200.0 + 100.0_f64 * 100.0).sqrt();
        assert!(
            (camera.params[0] - expected).abs() < 1e-9,
            "f = {}（期望 {expected}）",
            camera.params[0]
        );
        assert_eq!(camera.params[1], 100.0);
        assert_eq!(camera.params[2], 50.0);
        assert_eq!(
            count(
                &connection,
                "SELECT prior_focal_length FROM cameras WHERE camera_id = 1"
            ),
            1
        );
    }

    #[test]
    fn extraction_is_idempotent() {
        let fixture = Fixture::new();
        write_image(&fixture.images.join("frame_0001.jpg"), 160, 120, 1);
        let run = sink();
        extract(&run, &fixture.request()).unwrap();
        let before = {
            let connection = fixture.open();
            database::read_keypoints(&connection, 1).unwrap().unwrap()
        };
        // 第二次：IMAGE_EXISTS 跳过，特征点逐位不变。
        extract(&run, &fixture.request()).unwrap();
        let connection = fixture.open();
        let after = database::read_keypoints(&connection, 1).unwrap().unwrap();
        assert_eq!(before, after);
        assert_eq!(count(&connection, "SELECT COUNT(*) FROM images"), 1);
        assert_eq!(count(&connection, "SELECT COUNT(*) FROM keypoints"), 1);
    }

    #[test]
    fn max_image_size_scales_keypoints_back_to_camera() {
        let fixture = Fixture::new();
        write_image(&fixture.images.join("frame_0001.jpg"), 160, 120, 3);
        let mut request = fixture.request();
        request.max_image_size = Some(80);
        extract(&sink(), &request).unwrap();
        let connection = fixture.open();
        let keypoints = database::read_keypoints(&connection, 1).unwrap().unwrap();
        assert!(!keypoints.is_empty());
        let max_x = keypoints.iter().map(|k| k.x).fold(0.0f32, f32::max);
        let max_y = keypoints.iter().map(|k| k.y).fold(0.0f32, f32::max);
        // 缩略图是 80×60，坐标必须还原到相机尺寸（160×120）而不是留在缩略图坐标系。
        assert!(max_x > 80.0, "坐标未还原：max_x = {max_x}");
        assert!(max_y > 60.0, "坐标未还原：max_y = {max_y}");
        assert!(keypoints.iter().all(|k| k.x <= 160.0 && k.y <= 120.0));
    }

    #[test]
    fn mask_removes_keypoints() {
        let fixture = Fixture::new();
        write_image(&fixture.images.join("frame_0001.jpg"), 160, 120, 1);
        write_image(&fixture.images.join("frame_0002.jpg"), 160, 120, 2);
        // 掩码：左半黑（剔除）、右半白（保留）。
        let masks = fixture.directory.path().join("masks");
        std::fs::create_dir_all(&masks).unwrap();
        let mut mask = image::GrayImage::new(160, 120);
        for y in 0..120 {
            for x in 0..160 {
                mask.put_pixel(x, y, image::Luma([if x < 80 { 0 } else { 255 }]));
            }
        }
        // 第 1 张用 `<name>.png` 命名，第 2 张用 `<stem>.png`（COLMAP 的两种命名都要认）。
        mask.save(masks.join("frame_0001.jpg.png")).unwrap();
        mask.save(masks.join("frame_0002.png")).unwrap();

        let mut request = fixture.request();
        request.masks = Some(masks);
        extract(&sink(), &request).unwrap();

        let connection = fixture.open();
        for image_id in 1..=2u32 {
            let keypoints = database::read_keypoints(&connection, image_id).unwrap().unwrap();
            assert!(keypoints.iter().any(|k| k.x >= 80.0));
            assert!(
                keypoints.iter().all(|k| k.x >= 80.0),
                "掩码左侧的特征点应被剔除"
            );
        }
    }

    #[test]
    fn missing_mask_skips_image_with_warning() {
        let fixture = Fixture::new();
        write_image(&fixture.images.join("frame_0001.jpg"), 160, 120, 1);
        let masks = fixture.directory.path().join("masks");
        std::fs::create_dir_all(&masks).unwrap();
        let mut request = fixture.request();
        request.masks = Some(masks);
        // COLMAP 遇缺掩码只 warning 跳过，整体仍成功。
        extract(&sink(), &request).unwrap();
        let connection = fixture.open();
        assert_eq!(count(&connection, "SELECT COUNT(*) FROM images"), 0);
        assert_eq!(count(&connection, "SELECT COUNT(*) FROM cameras"), 0);
    }

    #[test]
    fn unreadable_image_is_skipped() {
        let fixture = Fixture::new();
        write_image(&fixture.images.join("frame_0001.jpg"), 160, 120, 1);
        std::fs::write(fixture.images.join("frame_0002.jpg"), b"not an image").unwrap();
        extract(&sink(), &fixture.request()).unwrap();
        let connection = fixture.open();
        // 只有能读的那张进了库。
        assert_eq!(count(&connection, "SELECT COUNT(*) FROM images"), 1);
        assert!(database::read_image_by_name(&connection, "frame_0002.jpg")
            .unwrap()
            .is_none());
    }

    #[test]
    fn image_list_limits_processed_images() {
        let fixture = Fixture::new();
        for index in 1..=3 {
            write_image(
                &fixture.images.join(format!("frame_{index:04}.jpg")),
                160,
                120,
                0x2000 + index,
            );
        }
        let list = fixture.directory.path().join("images.txt");
        // 故意乱序：COLMAP 会先排序再处理。
        std::fs::write(&list, "frame_0003.jpg\nframe_0001.jpg\n").unwrap();
        let mut request = fixture.request();
        request.image_list = Some(list);
        extract(&sink(), &request).unwrap();
        let connection = fixture.open();
        assert_eq!(count(&connection, "SELECT COUNT(*) FROM images"), 2);
        let (first_id, _) = database::read_image_by_name(&connection, "frame_0001.jpg")
            .unwrap()
            .unwrap();
        let (second_id, _) = database::read_image_by_name(&connection, "frame_0003.jpg")
            .unwrap()
            .unwrap();
        assert_eq!(first_id, 1, "列表未排序：image_id 顺序应与排序后的名字一致");
        assert_eq!(second_id, 2);
    }

    #[test]
    fn incremental_reuses_existing_camera() {
        let fixture = Fixture::new();
        write_image(&fixture.images.join("frame_0001.jpg"), 160, 120, 1);
        write_image(&fixture.images.join("frame_0002.jpg"), 160, 120, 4);
        let run = sink();
        extract(&run, &fixture.request()).unwrap();
        let existing = {
            let connection = fixture.open();
            database::read_keypoints(&connection, 1).unwrap().unwrap()
        };

        // 增量：新增一张图像，复用相机 1。
        write_image(&fixture.images.join("frame_0003.jpg"), 160, 120, 5);
        let list = fixture.directory.path().join("reshoot.txt");
        std::fs::write(&list, "frame_0003.jpg\n").unwrap();
        let mut request = fixture.request();
        request.image_list = Some(list);
        request.existing_camera_id = Some(1);
        extract(&run, &request).unwrap();

        let connection = fixture.open();
        assert_eq!(count(&connection, "SELECT COUNT(*) FROM cameras"), 1);
        assert_eq!(count(&connection, "SELECT COUNT(*) FROM images"), 3);
        let (image_id, camera_id) = database::read_image_by_name(&connection, "frame_0003.jpg")
            .unwrap()
            .unwrap();
        assert_eq!(camera_id, 1);
        assert!(database::read_keypoints(&connection, image_id)
            .unwrap()
            .unwrap()
            .len()
            > 50);
        // 既有图像的特征没被改动。
        assert_eq!(
            database::read_keypoints(&connection, 1).unwrap().unwrap(),
            existing
        );
        // 三张图像都有 frame，但 rig 只有一个（复用）。
        assert_eq!(count(&connection, "SELECT COUNT(*) FROM frames"), 3);
        assert_eq!(count(&connection, "SELECT COUNT(*) FROM rigs"), 1);
    }

    #[test]
    fn reusable_camera_dimension_mismatch_skips_image() {
        let fixture = Fixture::new();
        write_image(&fixture.images.join("frame_0001.jpg"), 160, 120, 1);
        write_image(&fixture.images.join("frame_0002.jpg"), 200, 150, 2);
        extract(&sink(), &fixture.request()).unwrap();
        let connection = fixture.open();
        // 尺寸不一致的那张被跳过（COLMAP CAMERA_SINGLE_DIM_ERROR 语义）。
        assert_eq!(count(&connection, "SELECT COUNT(*) FROM images"), 1);
        assert_eq!(count(&connection, "SELECT COUNT(*) FROM frames"), 1);
    }

    #[test]
    fn cancelled_sink_stops_immediately() {
        let fixture = Fixture::new();
        write_image(&fixture.images.join("frame_0001.jpg"), 160, 120, 1);
        let cancel = CancellationToken::new();
        cancel.cancel();
        let run = runtime::test_sink(cancel).unwrap();
        let error = extract(&run, &fixture.request()).unwrap_err();
        assert!(matches!(error, SplatError::Cancelled));
        // 取消发生在建库之前：不该留下半成品数据库。
        assert!(!fixture.database.exists());
    }

    #[test]
    fn missing_existing_camera_is_an_error() {
        let fixture = Fixture::new();
        write_image(&fixture.images.join("frame_0001.jpg"), 160, 120, 1);
        let mut request = fixture.request();
        request.existing_camera_id = Some(7);
        let error = extract(&sink(), &request).unwrap_err();
        assert!(error.to_string().contains("相机 7"));
    }
}

