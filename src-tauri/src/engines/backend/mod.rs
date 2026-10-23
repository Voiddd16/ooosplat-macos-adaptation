//! 稀疏重建**后端抽象层**。
//!
//! # 为什么有这一层
//!
//! 以前 pipeline 直接调用 `engines::colmap::*`（外部 CLI）与 `engines::ooo_sfm::*`
//! （自研进程内）—— 两边靠"同名同签名"硬凑，调用点里全是 COLMAP 专属概念：
//! 可执行文件路径、`gpu_index`、`automatic_optimization`……换后端等于换一套世界观。
//!
//! 现在 pipeline 只认 [`SfMBackend`] 这一个 trait：
//!
//! * **契约中性**：请求结构体里没有"可执行文件"、"CUDA 设备号"这类实现概念；
//! * **编译期选定**：[`active()`] 按 cargo feature 决定用哪个实现（见
//!   `engines/backend/mod.rs` 的 `compile_error!` 守卫：同时开两个后端直接编译失败）；
//! * **可测**：`Arc<dyn SfMBackend>` 可以在测试里塞假实现。
//!
//! # feature 与后端的对应
//!
//! | feature | 后端 | 形态 |
//! |---|---|---|
//! | （都不开） | COLMAP CLI | 外部进程 |
//! | `backend-colmap` | COLMAP CLI | 外部进程（显式） |
//! | `backend-ooo-sfm` | ooo-sfm | 进程内（Apple 平台） |
//! | `backend-ooo-sfm-metal` | ooo-sfm + Metal | 进程内 + GPU 内核 |
//!
//! 同时开 `backend-colmap` 与 `backend-ooo-sfm` 是**编译错误**（后端不唯一）。

use std::path::PathBuf;
use std::sync::Arc;

use crate::contracts::{BackendCapabilities, BackendIdentity};
use crate::error::Result;
use crate::process::{ProcessManager, ProcessObserver};

pub mod adapter;
pub mod request;

pub use request::{
    FeatureExtraction, IncrementalFeatureExtraction, IncrementalMapping, Matching,
    PairMatching, QualityFeatureExtraction, SparseMapping,
};

/// 一次后端作业的运行环境（与后端实现无关）。
///
/// 只包含"所有后端都需要的东西"：日志落到哪、进度往哪报、进程由谁托管。
/// **不含**可执行文件路径 —— 那是 CLI 后端自己的构造参数。
pub struct JobContext<'a> {
    /// 日志文件。
    pub log: PathBuf,
    /// 结构化进度观察者（`None` = 不上报）。
    pub observer: Option<ProcessObserver>,
    /// 进程托管器：外部后端用它拉子进程，进程内后端用它取取消令牌与生命周期观察者。
    pub manager: &'a ProcessManager,
    /// 加速器设备号（用哪个加速器）——中性概念：外部后端翻译成自己的设备参数，
    /// 进程内后端自行决定是否使用。
    pub accelerator_index: Option<u32>,
}

impl<'a> JobContext<'a> {
    pub fn new(
        log: PathBuf,
        observer: Option<ProcessObserver>,
        manager: &'a ProcessManager,
        accelerator_index: Option<u32>,
    ) -> Self {
        Self {
            log,
            observer,
            manager,
            accelerator_index,
        }
    }
}

/// 稀疏重建后端的统一契约。
///
/// 阶段划分与 4.2.1 的 CLI 子命令一一对应（特征提取 / 匹配 / 重建），
/// 但**参数是中性领域概念**，不是 CLI 开关。
#[async_trait::async_trait]
pub trait SfMBackend: Send + Sync {
    /// 身份（版本、形态、给界面显示的名字）。
    fn identity(&self) -> BackendIdentity;

    /// 能力清单（支持哪些阶段、是否支持闭环检测/两视图 track）。
    fn capabilities(&self) -> BackendCapabilities;

    /// 启动前的就绪校验（缺依赖、版本不符等）。
    fn require_ready(&self) -> Result<()>;

    /// 特征提取。
    async fn extract_features(
        &self,
        context: &JobContext<'_>,
        request: FeatureExtraction,
    ) -> Result<()>;

    /// 特征提取（带规划器：整目录一次性规划特征预算）。
    async fn extract_features_quality_v2(
        &self,
        context: &JobContext<'_>,
        request: QualityFeatureExtraction,
    ) -> Result<()>;

    /// 增量特征提取（补拍：新旧画面一起提取，复用既有相机）。
    async fn extract_incremental_features(
        &self,
        context: &JobContext<'_>,
        request: IncrementalFeatureExtraction,
    ) -> Result<()>;

    /// 顺序匹配（视频链路）。
    async fn match_sequential(&self, context: &JobContext<'_>, request: Matching) -> Result<()>;

    /// 穷举匹配（图片链路）。
    async fn match_exhaustive(&self, context: &JobContext<'_>, request: Matching) -> Result<()>;

    /// 指定帧对匹配。
    async fn match_pairs(&self, context: &JobContext<'_>, request: PairMatching) -> Result<()>;

    /// 稀疏重建。
    async fn map_sparse(&self, context: &JobContext<'_>, request: SparseMapping) -> Result<()>;

    /// 基于既有模型续建。
    async fn map_from_existing(
        &self,
        context: &JobContext<'_>,
        request: SparseMapping,
    ) -> Result<()>;

    /// 增量重建（补拍：既有模型 + 新画面）。
    async fn map_incremental(
        &self,
        context: &JobContext<'_>,
        request: IncrementalMapping,
    ) -> Result<()>;
}

/// 后端自报的身份与能力（由两侧实现各自提供）。
#[derive(Debug, Clone)]
pub struct BackendProfile {
    pub identity: BackendIdentity,
    pub capabilities: BackendCapabilities,
}

/// 后端构造参数（**装配层**的东西，不属于契约）。
///
/// 这些是"把后端跑起来"所需的接线信息，pipeline 不应该直接看到它们。
#[derive(Debug, Clone, Default)]
pub struct BackendConfig {
    /// 外部 CLI 后端的可执行文件路径（进程内后端忽略）。
    pub executable: Option<PathBuf>,
    /// 是否允许后端自动挑选自己的优化后端（COLMAP 的 Caspar/Ceres 自动选择）。
    pub automatic_optimization: bool,
    /// 加速器设备号（COLMAP 用 `--cuda_device` 之类；进程内后端忽略）。
    pub accelerator_index: Option<u32>,
}

/// 编译期选定的后端。
///
/// * `backend-ooo-sfm` 开启 → 自研进程内后端；
/// * 否则 → 外部 COLMAP CLI 后端（`backend-colmap` 显式开启，或默认）。
///
/// 两者同时开启是**编译错误**（见本文件顶部的 `compile_error!`）。
pub fn active(config: BackendConfig) -> Arc<dyn SfMBackend> {
    // 具体实现由 `engines/mod.rs` 的 `#[path]` 门面按 feature 决定；
    // 适配层只写一次，两侧签名由门面保证一致。
    Arc::new(crate::engines::backend::adapter::SelectedBackend::new(config))
}

#[cfg(all(feature = "backend-colmap", feature = "backend-ooo-sfm"))]
compile_error!(
    "后端不唯一：同时启用了 `backend-colmap` 与 `backend-ooo-sfm`。\
     请只选一个（自研后端：`--features backend-ooo-sfm`；外部 COLMAP：`--features backend-colmap`）。"
);

/// 当前编译期选中的后端标识（诊断/日志用）。
pub const fn active_backend_id() -> &'static str {
    #[cfg(feature = "backend-ooo-sfm")]
    {
        "ooo-sfm"
    }
    #[cfg(not(feature = "backend-ooo-sfm"))]
    {
        "colmap"
    }
}

#[cfg(all(feature = "backend-ooo-sfm", not(target_vendor = "apple")))]
compile_error!(
    "`backend-ooo-sfm` 目前只有 Apple 平台实现；其它平台请使用 `backend-colmap`。"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{BackendFamily, ComponentKind};

    /// 契约自检：后端身份与能力必须自洽（两个实现共用同一组断言）。
    #[test]
    fn active_backend_reports_a_coherent_identity() {
        let backend = active(BackendConfig::default());
        let identity = backend.identity();
        assert!(!identity.id.is_empty());
        assert!(!identity.display_name.is_empty());
        assert!(!identity.version.is_empty());
        let capabilities = backend.capabilities();
        assert!(
            !capabilities.stages.is_empty(),
            "后端必须声明至少一个阶段"
        );
        // 身份必须可序列化（前端契约）。
        let json = serde_json::to_string(&identity).expect("序列化后端身份");
        assert!(json.contains("\"id\""), "{json}");
    }

    /// 组件类别契约（前端只认这几类）。
    #[test]
    fn component_kinds_are_the_neutral_set() {
        assert_eq!(ComponentKind::Media.label(), "media");
        assert_eq!(ComponentKind::Reconstruction.label(), "reconstruction");
        assert_eq!(ComponentKind::Training.label(), "training");
    }

    /// 两个实现都必须满足：家族与形态自洽。
    #[test]
    fn backend_family_matches_implementation() {
        let backend = active(BackendConfig::default());
        let family = backend.identity().family;
        #[cfg(feature = "backend-ooo-sfm")]
        assert_eq!(family, BackendFamily::InProcess);
        #[cfg(not(feature = "backend-ooo-sfm"))]
        assert_eq!(family, BackendFamily::ExternalCli);
    }
}
