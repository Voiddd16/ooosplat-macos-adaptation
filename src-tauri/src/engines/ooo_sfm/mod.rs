//! ooo-sfm：项目自研的 SfM 领域层（平台无关）。
//!
//! 本文件经 `engines/mod.rs` 的 `#[cfg(feature = "backend-ooo-sfm")] + #[path]` 门面，
//! 以 `engines::colmap` 的名义对外暴露；公开签名与 `engines/colmap.rs` **逐字一致**，
//! 因此 `pipeline/runner.rs` 等调用点一行不改。
//!
//! 平台相关实现位于 `ooo_sfm/<platform>/`（当前仅 `apple`）。
//! 设计文档：`docs/project-notes/ooo-sfm-implementation-plan.md`。

use std::path::{Path, PathBuf};

use crate::{
    error::Result,
    process::{ProcessManager, ProcessObserver},
};

#[cfg(target_vendor = "apple")]
pub(crate) mod apple;

// M7：自研后端的健康 / 能力报告（替代主分支对 COLMAP 二进制的探测）。
pub mod health;

#[cfg(not(target_vendor = "apple"))]
compile_error!(
    "ooo-sfm 目前只有 Apple 平台后端，不要在非 Apple 平台启用 feature `ooo-sfm`（见 docs/project-notes/ooo-sfm-implementation-plan.md）"
);

// ---------------------------------------------------------------------------
// 引擎身份（原语义：COLMAP 二进制的版本 / CLI 家族）
// ---------------------------------------------------------------------------

/// 自研后端版本串。原语义是"上游锁定版本"（COLMAP 4.2.1），此处改为 ooo-sfm 自身版本。
pub const LOCKED_VERSION: &str = "ooo-sfm 0.1.0";

/// 自研后端没有上游 commit；保留常量以维持公开接口。
pub const LOCKED_COMMIT: &str = "";

/// ooo-sfm 随本仓库一起编译，天然可信，恒为 `true`。
pub fn is_locked_build(_help: &str) -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ColmapCliFamily {
    Legacy39,
    Modern4,
}

impl ColmapCliFamily {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Legacy39 => "COLMAP 3.9 CLI",
            Self::Modern4 => "COLMAP 4.x CLI",
        }
    }
}

/// 自研后端没有 CLI，能力集合按 COLMAP 4.x 家族等价报告（沿用既有前端语义）。
pub fn detect_cli_family(_feature_help: &str, _matching_help: &str) -> Option<ColmapCliFamily> {
    Some(ColmapCliFamily::Modern4)
}

/// 后端自报的身份与能力（供 `engines::backend` 契约层使用）。
pub fn backend_profile() -> crate::engines::backend::BackendProfile {
    crate::engines::backend::BackendProfile {
        identity: crate::contracts::BackendIdentity {
            id: "ooo-sfm".into(),
            display_name: "自研 SfM 后端".into(),
            version: LOCKED_VERSION.to_string(),
            family: crate::contracts::BackendFamily::InProcess,
        },
        capabilities: crate::contracts::BackendCapabilities {
            stages: vec![
                crate::contracts::BackendStage::FeatureExtraction,
                crate::contracts::BackendStage::IncrementalFeatureExtraction,
                crate::contracts::BackendStage::SequentialMatching,
                crate::contracts::BackendStage::ExhaustiveMatching,
                crate::contracts::BackendStage::PairMatching,
                crate::contracts::BackendStage::SparseMapping,
                crate::contracts::BackendStage::MappingFromExisting,
                crate::contracts::BackendStage::IncrementalMapping,
            ],
            // 闭环检测（词袋树）尚未实现；如实上报，不假装支持。
            supports_loop_detection: false,
            supports_two_view_tracks: true,
        },
    }
}

/// 进程内后端没有需要校验的外部二进制。
pub fn require_verified_cli(_executable: &Path) -> Result<()> {
    Ok(())
}

// ---------------------------------------------------------------------------
// 稀疏重建流水线（签名 = engines/colmap.rs，实现 = 平台后端）
// ---------------------------------------------------------------------------

/// 把 CLI 契约里的相对路径解析为进程内后端可用的路径。
///
/// `pipeline/runner.rs` 有意向 COLMAP 传 `../colmap_frames` 这类相对路径：外部
/// CLI 的工作目录是数据库所在目录（`work/colmap`，见 `engines/colmap.rs` 各阶段
/// 传给 `run_colmap` 的 `working_directory`），既让命令行保持简短，也避免 Windows
/// 上非 ASCII 的绝对路径进入 COLMAP 的位图加载器。进程内后端没有进程工作目录，
/// 因此在门面处统一以 `database.parent()` 为基准展开 —— 与 CLI 的解析结果一致。
fn resolve_path(database: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    database
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(path)
}

#[allow(clippy::too_many_arguments)]
pub async fn extract_features(
    executable: &Path,
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
    let images = resolve_path(database, images);
    let masks = masks.map(|path| resolve_path(database, path));
    apple::extract_features(
        executable,
        database,
        &images,
        masks.as_deref(),
        max_image_size,
        max_features,
        log,
        manager,
        observer,
        gpu_index,
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
    let images = resolve_path(database, images);
    let masks = masks.map(|path| resolve_path(database, path));
    let image_list = image_list.map(|path| resolve_path(database, path));
    apple::extract_features_quality_v2(
        executable,
        database,
        &images,
        masks.as_deref(),
        image_list.as_deref(),
        max_image_size,
        max_features,
        log,
        manager,
        observer,
        gpu_index,
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
    let images = resolve_path(database, images);
    let image_list = resolve_path(database, image_list);
    let masks = masks.map(|path| resolve_path(database, path));
    apple::extract_incremental_features(
        executable,
        database,
        &images,
        &image_list,
        existing_camera_id,
        masks.as_deref(),
        max_image_size,
        max_features,
        log,
        manager,
        observer,
        gpu_index,
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
    apple::match_sequential(
        executable,
        database,
        log,
        manager,
        observer,
        gpu_index,
        loop_detection,
    )
    .await
}

pub async fn match_exhaustive(
    executable: &Path,
    database: &Path,
    log: PathBuf,
    manager: &ProcessManager,
    observer: Option<ProcessObserver>,
    gpu_index: Option<u32>,
) -> Result<()> {
    apple::match_exhaustive(executable, database, log, manager, observer, gpu_index).await
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
    let pair_list = resolve_path(database, pair_list);
    apple::match_pairs(
        executable,
        database,
        &pair_list,
        log,
        manager,
        observer,
        gpu_index,
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
    let images = resolve_path(database, images);
    let output = resolve_path(database, output);
    apple::map(
        executable,
        database,
        &images,
        &output,
        allow_two_view_tracks,
        automatic_optimization,
        log,
        manager,
        observer,
        gpu_index,
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
    let images = resolve_path(database, images);
    let input_model = resolve_path(database, input_model);
    let output = resolve_path(database, output);
    apple::map_from_existing(
        executable,
        database,
        &images,
        &input_model,
        &output,
        automatic_optimization,
        log,
        manager,
        observer,
        gpu_index,
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
    let images = resolve_path(database, images);
    let input = resolve_path(database, input);
    let output = resolve_path(database, output);
    let image_list = resolve_path(database, image_list);
    apple::map_incremental(
        executable,
        database,
        &images,
        &input,
        &output,
        &image_list,
        automatic_optimization,
        log,
        manager,
        observer,
        gpu_index,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// runner 传给 COLMAP 的是 `../colmap_frames` 这类相对路径（相对数据库目录）。
    /// 进程内后端必须在门面处按同一基准展开，否则真实任务会在特征提取阶段报
    /// “读取图像目录 ../colmap_frames 失败”。
    #[test]
    fn relative_cli_paths_resolve_against_the_database_directory() {
        let database = Path::new("project/work/colmap/database.db");
        assert_eq!(
            resolve_path(database, Path::new("../colmap_frames")),
            Path::new("project/work/colmap").join("../colmap_frames")
        );
        assert_eq!(
            resolve_path(database, Path::new("../masks")),
            Path::new("project/work/colmap").join("../masks")
        );
    }

    #[test]
    fn absolute_paths_pass_through_unchanged() {
        let database = Path::new("project/work/colmap/database.db");
        let absolute = std::env::temp_dir().join("colmap_frames");
        assert_eq!(resolve_path(database, &absolute), absolute);
    }

    /// 真实任务复刻：runner 给 COLMAP 传 `../colmap_frames`（相对 `work/colmap`）。
    /// 进程内后端必须按数据库目录展开，否则真实任务会在特征提取阶段报
    /// “读取图像目录 ../colmap_frames 失败”。
    #[tokio::test]
    async fn quality_v2_extraction_accepts_the_runner_relative_image_path() {
        let temporary = tempfile::tempdir().unwrap();
        let colmap = temporary.path().join("work/colmap");
        let frames = temporary.path().join("work/colmap_frames");
        tokio::fs::create_dir_all(&colmap).await.unwrap();
        tokio::fs::create_dir_all(&frames).await.unwrap();
        for index in 1..=2 {
            apple::test_images::write_image(
                &frames.join(format!("frame_{index:04}.jpg")),
                160,
                120,
                0x1000 + index,
            );
        }
        let database = colmap.join("database.db");

        extract_features_quality_v2(
            Path::new("ooo-sfm"),
            &database,
            Path::new("../colmap_frames"),
            None,
            None,
            1600,
            8192,
            temporary.path().join("extract.log"),
            &ProcessManager::new(),
            None,
            None,
        )
        .await
        .unwrap();

        let connection = rusqlite::Connection::open(&database).unwrap();
        let images: i64 = connection
            .query_row("SELECT COUNT(*) FROM images", [], |row| row.get(0))
            .unwrap();
        assert_eq!(images, 2, "相对路径必须解析到 work/colmap_frames 并入库");
    }
}
