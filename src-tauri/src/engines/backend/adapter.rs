//! 后端契约 → 具体实现的适配层。
//!
//! 这里只有一个实现体，因为**被适配的两侧由 `engines/mod.rs` 的 `#[path]` 门面保证
//! 同名同签名**：feature 关闭时适配 `engines/colmap.rs`（外部 CLI），开启时适配
//! `engines/ooo_sfm/mod.rs`（进程内）。适配层因此只需要写一次，两边不可能漂移。
//!
//! 契约里没有的东西（可执行文件路径、自动优化开关）都在 [`BackendConfig`] 里，
//! 它们是**装配参数**，不是领域概念。

use async_trait::async_trait;

use super::{
    BackendConfig, FeatureExtraction, IncrementalFeatureExtraction, IncrementalMapping,
    JobContext, Matching, PairMatching, QualityFeatureExtraction, SparseMapping, SfMBackend,
};
use crate::contracts::{BackendCapabilities, BackendIdentity};
use crate::error::Result;

/// 被选中的后端（外部 CLI 或进程内，由 feature 决定）。
pub struct SelectedBackend {
    /// 外部 CLI 的可执行文件（进程内后端忽略）。
    executable: std::path::PathBuf,
}

impl SelectedBackend {
    pub fn new(config: BackendConfig) -> Self {
        Self {
            executable: config.executable.unwrap_or_default(),
        }
    }
}

#[async_trait]
impl SfMBackend for SelectedBackend {
    fn identity(&self) -> BackendIdentity {
        crate::engines::colmap::backend_profile().identity
    }

    fn capabilities(&self) -> BackendCapabilities {
        crate::engines::colmap::backend_profile().capabilities
    }

    fn require_ready(&self) -> Result<()> {
        // 两侧语义不同但都由门面实现：外部 CLI 校验二进制存在，进程内后端是 no-op。
        crate::engines::colmap::require_verified_cli(&self.executable)
    }

    async fn extract_features(
        &self,
        context: &JobContext<'_>,
        request: FeatureExtraction,
    ) -> Result<()> {
        crate::engines::colmap::extract_features(
            &self.executable,
            &request.database,
            &request.images,
            request.masks.as_deref(),
            request.max_image_size,
            request.max_features,
            context.log.clone(),
            context.manager,
            context.observer.clone(),
            context.accelerator_index,
        )
        .await
    }

    async fn extract_features_quality_v2(
        &self,
        context: &JobContext<'_>,
        request: QualityFeatureExtraction,
    ) -> Result<()> {
        let base = request.base;
        crate::engines::colmap::extract_features_quality_v2(
            &self.executable,
            &base.database,
            &base.images,
            base.masks.as_deref(),
            base.image_list.as_deref(),
            base.max_image_size.unwrap_or_default(),
            request.planned_max_features,
            context.log.clone(),
            context.manager,
            context.observer.clone(),
            context.accelerator_index,
        )
        .await
    }

    async fn extract_incremental_features(
        &self,
        context: &JobContext<'_>,
        request: IncrementalFeatureExtraction,
    ) -> Result<()> {
        let base = request.base;
        crate::engines::colmap::extract_incremental_features(
            &self.executable,
            &base.database,
            &base.images,
            base.image_list.as_deref().unwrap_or(std::path::Path::new("")),
            base.existing_camera_id.unwrap_or_default(),
            base.masks.as_deref(),
            base.max_image_size,
            base.max_features,
            context.log.clone(),
            context.manager,
            context.observer.clone(),
            context.accelerator_index,
        )
        .await
    }

    async fn match_sequential(&self, context: &JobContext<'_>, request: Matching) -> Result<()> {
        crate::engines::colmap::match_sequential(
            &self.executable,
            &request.database,
            context.log.clone(),
            context.manager,
            context.observer.clone(),
            context.accelerator_index,
            request.loop_detection,
        )
        .await
    }

    async fn match_exhaustive(&self, context: &JobContext<'_>, request: Matching) -> Result<()> {
        crate::engines::colmap::match_exhaustive(
            &self.executable,
            &request.database,
            context.log.clone(),
            context.manager,
            context.observer.clone(),
            context.accelerator_index,
        )
        .await
    }

    async fn match_pairs(&self, context: &JobContext<'_>, request: PairMatching) -> Result<()> {
        crate::engines::colmap::match_pairs(
            &self.executable,
            &request.database,
            &request.pair_list,
            context.log.clone(),
            context.manager,
            context.observer.clone(),
            context.accelerator_index,
        )
        .await
    }

    async fn map_sparse(&self, context: &JobContext<'_>, request: SparseMapping) -> Result<()> {
        crate::engines::colmap::map(
            &self.executable,
            &request.database,
            &request.images,
            &request.output,
            request.allow_two_view_tracks,
            request.automatic_optimization,
            context.log.clone(),
            context.manager,
            context.observer.clone(),
            context.accelerator_index,
        )
        .await
    }

    async fn map_from_existing(
        &self,
        context: &JobContext<'_>,
        request: SparseMapping,
    ) -> Result<()> {
        crate::engines::colmap::map_from_existing(
            &self.executable,
            &request.database,
            &request.images,
            &request.input_model,
            &request.output,
            request.automatic_optimization,
            context.log.clone(),
            context.manager,
            context.observer.clone(),
            context.accelerator_index,
        )
        .await
    }

    async fn map_incremental(
        &self,
        context: &JobContext<'_>,
        request: IncrementalMapping,
    ) -> Result<()> {
        crate::engines::colmap::map_incremental(
            &self.executable,
            &request.database,
            &request.images,
            &request.input_model,
            &request.output,
            &request.image_list,
            request.automatic_optimization,
            context.log.clone(),
            context.manager,
            context.observer.clone(),
            context.accelerator_index,
        )
        .await
    }
}
