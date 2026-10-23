//! Apple 平台后端（macOS）。
//!
//! 九个阶段都接到 [`runtime::run_job`] 上（日志 / observer / 取消 / 错误语义齐全）；
//! 算法本体按 M3–M6 逐段落地：M3（特征提取）已实现，其余阶段在落地前以明确的
//! `SplatError::UnsupportedEngine` 失败，而不是 panic。
//! 见 `docs/project-notes/ooo-sfm-implementation-plan.md`。

// 求解器本体由 M5 的 mapper / 位姿精化接入；在那之前仅测试使用。
#[allow(dead_code)]
pub(crate) mod ba;
pub(crate) mod camera;
pub(crate) mod exif;
pub(crate) mod features;
pub(crate) mod gpu;
// M5 起被 mapper / 三角化广泛使用；先按 M6 消费范围标注。
#[allow(dead_code)]
pub(crate) mod linalg;
#[allow(dead_code)] // M5：`*.bin` 读取器在自研建图落地时接入
pub(crate) mod format;
pub(crate) mod mapper;
pub(crate) mod matching;
// M5b：绝对位姿（PnP）——mapper 的注册路径使用。
pub(crate) mod pose;
pub(crate) mod reconstruction;
pub(crate) mod relative_pose;
pub(crate) mod runtime;
pub(crate) mod sift;
#[cfg(test)]
pub(crate) mod test_images;
pub(crate) mod triangulator;
pub(crate) mod two_view;

use std::path::{Path, PathBuf};

use crate::{
    error::{Result, SplatError},
    process::{ProcessManager, ProcessObserver},
};

use runtime::run_job;

/// 尚未落地的阶段（M4–M6）：失败信息指明对应里程碑。
fn not_yet(milestone: &'static str) -> SplatError {
    SplatError::UnsupportedBackend(format!("ooo-sfm 尚未实现该阶段（{milestone}）"))
}

/// 日志"参数区"的一行（取代子进程版的 CLI 参数列表）。
fn describe(pairs: impl IntoIterator<Item = (&'static str, String)>) -> Vec<String> {
    pairs
        .into_iter()
        .map(|(name, value)| format!("{name}: {value}"))
        .collect()
}

/// `Option` 的可读形式（"-" 表示未设置）。
fn optional<T: ToString>(value: Option<T>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string())
}

#[allow(clippy::too_many_arguments)]
pub async fn extract_features(    executable: &Path,
    database: &Path,
    images: &Path,
    masks: Option<&Path>,
    max_image_size: Option<u32>,
    max_features: Option<u32>,
    log: PathBuf,
    manager: &ProcessManager,
    observer: Option<ProcessObserver>,
    gpu_index: Option<u32>,
) -> Result<()> {
    let request = features::ExtractionRequest {
        database: database.to_path_buf(),
        images: images.to_path_buf(),
        masks: masks.map(Path::to_path_buf),
        image_list: None,
        max_image_size,
        max_features,
        existing_camera_id: None,
        gpu_index,
    };
    run_job(
        manager,
        executable,
        &describe([
            ("stage", "feature_extraction".into()),
            ("database", database.display().to_string()),
            ("images", images.display().to_string()),
            ("masks", masks.map(|path| path.display().to_string()).unwrap_or_else(|| "-".into())),
            ("max_image_size", optional(max_image_size)),
            ("max_features", optional(max_features)),
            ("gpu_index", optional(gpu_index)),
        ]),
        log,
        observer,
        move |sink| features::extract(&sink, &request),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn extract_features_quality_v2(
    executable: &Path,
    database: &Path,
    images: &Path,
    masks: Option<&Path>,
    image_list: Option<&Path>,
    max_image_size: u32,
    max_features: u32,
    log: PathBuf,
    manager: &ProcessManager,
    observer: Option<ProcessObserver>,
    gpu_index: Option<u32>,
) -> Result<()> {
    let request = features::ExtractionRequest {
        database: database.to_path_buf(),
        images: images.to_path_buf(),
        masks: masks.map(Path::to_path_buf),
        image_list: image_list.map(Path::to_path_buf),
        max_image_size: Some(max_image_size),
        max_features: Some(max_features),
        existing_camera_id: None,
        gpu_index,
    };
    run_job(
        manager,
        executable,
        &describe([
            ("stage", "feature_extraction (quality v2)".into()),
            ("database", database.display().to_string()),
            ("images", images.display().to_string()),
            ("masks", masks.map(|path| path.display().to_string()).unwrap_or_else(|| "-".into())),
            ("image_list", image_list.map(|path| path.display().to_string()).unwrap_or_else(|| "-".into())),
            ("max_image_size", max_image_size.to_string()),
            ("max_features", max_features.to_string()),
            ("gpu_index", optional(gpu_index)),
        ]),
        log,
        observer,
        move |sink| features::extract(&sink, &request),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn extract_incremental_features(
    executable: &Path,
    database: &Path,
    images: &Path,
    image_list: &Path,
    existing_camera_id: u32,
    masks: Option<&Path>,
    max_image_size: Option<u32>,
    max_features: Option<u32>,
    log: PathBuf,
    manager: &ProcessManager,
    observer: Option<ProcessObserver>,
    gpu_index: Option<u32>,
) -> Result<()> {
    let request = features::ExtractionRequest {
        database: database.to_path_buf(),
        images: images.to_path_buf(),
        masks: masks.map(Path::to_path_buf),
        image_list: Some(image_list.to_path_buf()),
        max_image_size,
        max_features,
        existing_camera_id: Some(existing_camera_id),
        gpu_index,
    };
    run_job(
        manager,
        executable,
        &describe([
            ("stage", "feature_extraction (incremental)".into()),
            ("database", database.display().to_string()),
            ("images", images.display().to_string()),
            ("image_list", image_list.display().to_string()),
            ("existing_camera_id", existing_camera_id.to_string()),
            ("masks", masks.map(|path| path.display().to_string()).unwrap_or_else(|| "-".into())),
            ("max_image_size", optional(max_image_size)),
            ("max_features", optional(max_features)),
            ("gpu_index", optional(gpu_index)),
        ]),
        log,
        observer,
        move |sink| features::extract(&sink, &request),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn match_sequential(
    executable: &Path,
    database: &Path,
    log: PathBuf,
    manager: &ProcessManager,
    observer: Option<ProcessObserver>,
    gpu_index: Option<u32>,
    loop_detection: bool,
) -> Result<()> {
    let request = matching::MatchingRequest {
        database: database.to_path_buf(),
        strategy: matching::PairStrategy::Sequential { loop_detection },
    };
    run_job(
        manager,
        executable,
        &describe([
            ("stage", "sequential_matching".into()),
            ("database", database.display().to_string()),
            ("loop_detection", loop_detection.to_string()),
            ("gpu_index", optional(gpu_index)),
        ]),
        log,
        observer,
        move |sink| matching::match_images(&sink, &request),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn match_exhaustive(
    executable: &Path,
    database: &Path,
    log: PathBuf,
    manager: &ProcessManager,
    observer: Option<ProcessObserver>,
    gpu_index: Option<u32>,
) -> Result<()> {
    let request = matching::MatchingRequest {
        database: database.to_path_buf(),
        strategy: matching::PairStrategy::Exhaustive,
    };
    run_job(
        manager,
        executable,
        &describe([
            ("stage", "exhaustive_matching".into()),
            ("database", database.display().to_string()),
            ("gpu_index", optional(gpu_index)),
        ]),
        log,
        observer,
        move |sink| matching::match_images(&sink, &request),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn match_pairs(
    executable: &Path,
    database: &Path,
    pair_list: &Path,
    log: PathBuf,
    manager: &ProcessManager,
    observer: Option<ProcessObserver>,
    gpu_index: Option<u32>,
) -> Result<()> {
    let request = matching::MatchingRequest {
        database: database.to_path_buf(),
        strategy: matching::PairStrategy::Imported {
            match_list: pair_list.to_path_buf(),
        },
    };
    run_job(
        manager,
        executable,
        &describe([
            ("stage", "matches_importer (pairs)".into()),
            ("database", database.display().to_string()),
            ("pair_list", pair_list.display().to_string()),
            ("gpu_index", optional(gpu_index)),
        ]),
        log,
        observer,
        move |sink| matching::match_images(&sink, &request),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn map(
    executable: &Path,
    database: &Path,
    images: &Path,
    output: &Path,
    allow_two_view_tracks: bool,
    automatic_optimization: bool,
    log: PathBuf,
    manager: &ProcessManager,
    observer: Option<ProcessObserver>,
    gpu_index: Option<u32>,
) -> Result<()> {
    let request = mapper::MappingRequest {
        database: database.to_path_buf(),
        output: output.to_path_buf(),
        image_list: None,
        input_model: None,
        allow_two_view_tracks,
        fix_existing_frames: false,
        refine_intrinsics: true,
    };
    let _ = (images, automatic_optimization, gpu_index);
    run_job(
        manager,
        executable,
        &describe([
            ("stage", "mapper".into()),
            ("database", database.display().to_string()),
            ("images", images.display().to_string()),
            ("output", output.display().to_string()),
            ("allow_two_view_tracks", allow_two_view_tracks.to_string()),
        ]),
        log,
        observer,
        move |sink| mapper::map_sparse(&sink, &request).map(|_| ()),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn map_from_existing(
    executable: &Path,
    database: &Path,
    images: &Path,
    input_model: &Path,
    output: &Path,
    automatic_optimization: bool,
    log: PathBuf,
    manager: &ProcessManager,
    observer: Option<ProcessObserver>,
    gpu_index: Option<u32>,
) -> Result<()> {
    let request = mapper::MappingRequest {
        database: database.to_path_buf(),
        output: output.to_path_buf(),
        image_list: None,
        input_model: Some(input_model.to_path_buf()),
        allow_two_view_tracks: false,
        fix_existing_frames: false,
        refine_intrinsics: true,
    };
    let _ = (images, automatic_optimization, gpu_index);
    run_job(
        manager,
        executable,
        &describe([
            ("stage", "mapper (from existing model)".into()),
            ("database", database.display().to_string()),
            ("input_path", input_model.display().to_string()),
            ("output", output.display().to_string()),
        ]),
        log,
        observer,
        move |sink| mapper::map_sparse(&sink, &request).map(|_| ()),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn map_incremental(
    executable: &Path,
    database: &Path,
    images: &Path,
    input: &Path,
    output: &Path,
    image_list: &Path,
    automatic_optimization: bool,
    log: PathBuf,
    manager: &ProcessManager,
    observer: Option<ProcessObserver>,
    gpu_index: Option<u32>,
) -> Result<()> {
    let request = mapper::MappingRequest {
        database: database.to_path_buf(),
        output: output.to_path_buf(),
        image_list: Some(image_list.to_path_buf()),
        input_model: Some(input.to_path_buf()),
        allow_two_view_tracks: false,
        // `--Mapper.fix_existing_frames 1` + `--Mapper.ba_refine_* 0`。
        fix_existing_frames: true,
        refine_intrinsics: false,
    };
    let _ = (images, automatic_optimization, gpu_index);
    run_job(
        manager,
        executable,
        &describe([
            ("stage", "mapper (incremental)".into()),
            ("database", database.display().to_string()),
            ("input_path", input.display().to_string()),
            ("image_list", image_list.display().to_string()),
            ("output", output.display().to_string()),
        ]),
        log,
        observer,
        move |sink| mapper::map_sparse(&sink, &request).map(|_| ()),
    )
    .await
}
