//! ooo-sfm：项目自研的 SfM 领域层（平台无关）。
//!
//! 本文件经 `engines/mod.rs` 的 `#[cfg(feature = "ooo-sfm")] + #[path]` 门面，
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

/// 进程内后端没有需要校验的外部二进制。
pub fn require_verified_cli(_executable: &Path) -> Result<()> {
    Ok(())
}

// ---------------------------------------------------------------------------
// 稀疏重建流水线（签名 = engines/colmap.rs，实现 = 平台后端）
// ---------------------------------------------------------------------------

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
    apple::extract_features(
        executable,
        database,
        images,
        masks,
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
    apple::extract_features_quality_v2(
        executable,
        database,
        images,
        masks,
        image_list,
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
    apple::extract_incremental_features(
        executable,
        database,
        images,
        image_list,
        existing_camera_id,
        masks,
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
    apple::match_pairs(
        executable,
        database,
        pair_list,
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
    apple::map(
        executable,
        database,
        images,
        output,
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
    apple::map_from_existing(
        executable,
        database,
        images,
        input_model,
        output,
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
    apple::map_incremental(
        executable,
        database,
        images,
        input,
        output,
        image_list,
        automatic_optimization,
        log,
        manager,
        observer,
        gpu_index,
    )
    .await
}
