//! 前端 ⇄ 后端的**稳定契约**。
//!
//! 这一层是前端唯一能看到的词汇表：只描述"**哪一类能力**"和"**当前状态**"，
//! 不含 COLMAP / Brush / NVIDIA / CUDA / 某个可执行文件等具体实现概念。
//!
//! 具体的计算后端（外部 CLI / 自研进程内）由 [`crate::engines::backend`] 在
//! **编译期**按 cargo feature 选定；后端实现的任何细节都不得穿透到本模块 ——
//! 这是"前端与计算后端解耦"的硬边界：
//!
//! * 后端换了（COLMAP → 自研），前端契约**一字不改**；
//! * 前端只依赖本模块的语义（`ComponentKind::Reconstruction` 而不是 "colmap"）。
//!
//! 命名约定：序列化一律 `camelCase`（前端 TS 类型与之对应，见 `src/types/pipeline.ts`）。

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// 组件（承担计算的可执行单元）
// ---------------------------------------------------------------------------

/// 承担计算的组件**类别**。
///
/// 刻意不区分"ffmpeg"与"ffprobe"、不区分"colmap"与"自研"：前端只需要知道
/// "媒体工具在不在" / "重建引擎在不在" / "训练器在不在"。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ComponentKind {
    /// 媒体处理（抽帧、探测视频）。
    Media,
    /// 稀疏重建（特征、匹配、三角化、平差）。
    Reconstruction,
    /// 高斯训练 / 查看器。
    Training,
}

impl ComponentKind {
    /// 稳定的中性标签（前端 i18n 的 key 后缀，避免在 TS 里硬编码实现名）。
    pub const fn label(self) -> &'static str {
        match self {
            Self::Media => "media",
            Self::Reconstruction => "reconstruction",
            Self::Training => "training",
        }
    }
}

// ---------------------------------------------------------------------------
// 加速状态
// ---------------------------------------------------------------------------

/// 加速模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AccelerationMode {
    Cpu,
    Gpu,
}

/// 加速可用性。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AccelerationState {
    Ready,
    TemporarilyUnavailable,
    Unavailable,
}

/// 加速状态的**中性**原因码。
///
/// 去掉了 CUDA / NVIDIA 专属语义（原 `ColmapUnavailable` / `NvidiaSmiNotFound` /
/// `MacOsCpuOnly` 等），改成"为什么是这个结论"的通用分类。原文案走 `reason` 字段。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AccelerationReason {
    /// 加速器可用。
    GpuReady,
    /// 平台/配置只提供 CPU（例如没有受支持的加速后端）。
    CpuOnly,
    /// 组件本身缺失，无法探测。
    ComponentUnavailable,
    /// 加速器存在但运行要求不满足（驱动/能力不足）。
    RequirementsUnavailable,
    /// 探测过程本身失败。
    ProbeFailed,
    /// 探测超时。
    ProbeTimeout,
    /// 没有找到任何加速器。
    NoAccelerator,
    /// 驱动版本未知。
    DriverVersionUnknown,
    /// 驱动版本过旧。
    DriverTooOld,
    /// 计算能力未知。
    ComputeCapabilityUnknown,
    /// 计算能力不足。
    ComputeCapabilityTooLow,
}

/// 加速器设备信息（通用字段：Metal / CUDA / 其它后端都能填）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcceleratorDevice {
    pub index: u32,
    pub name: String,
    /// 后端运行时的版本（CUDA 驱动版本 / Metal feature set）。
    pub driver_version: String,
    /// 后端能力等级（CUDA compute capability / Metal 家族）。
    pub compute_capability: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_memory_mb: Option<u64>,
}

/// 加速器运行要求。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcceleratorRequirements {
    pub minimum_driver_version: String,
    pub minimum_compute_capability: String,
}

/// 加速状态（前端唯一的加速数据源）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AccelerationStatus {
    pub mode: AccelerationMode,
    pub state: AccelerationState,
    pub reason_code: AccelerationReason,
    /// 人类可读的原因（前端用 `reason` 做本地化兜底）。
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<AcceleratorDevice>,
    pub requirements: AcceleratorRequirements,
    /// 探测到的加速器数量。
    pub detected_device_count: usize,
}

impl AccelerationStatus {
    /// 是否真的跑在加速器上。
    pub const fn accelerated(&self) -> bool {
        matches!(self.mode, AccelerationMode::Gpu)
    }

    /// 兼容旧接口名（`use_gpu`）。
    pub const fn use_gpu(&self) -> bool {
        self.accelerated()
    }

    /// 只有通过兼容性检查的加速器才计入可用显存；CPU 回退可能仍保留诊断信息，
    /// 但不得据此选择更高显存的配置档。
    pub fn usable_gpu_total_memory_mb(&self) -> Option<u64> {
        self.use_gpu()
            .then(|| {
                self.device
                    .as_ref()
                    .and_then(|device| device.total_memory_mb)
            })
            .flatten()
    }

    /// 规划阶段可以沿用上一次成功的探测结果；任务执行永远走新鲜探测。
    pub fn planning_gpu_total_memory_mb(&self) -> Option<u64> {
        (self.use_gpu()
            && matches!(
                self.state,
                AccelerationState::Ready | AccelerationState::TemporarilyUnavailable
            ))
        .then(|| {
            self.device
                .as_ref()
                .and_then(|device| device.total_memory_mb)
        })
        .flatten()
    }

    /// 加速器设备号（供后端选择设备）。
    pub fn gpu_index(&self) -> Option<u32> {
        self.use_gpu()
            .then(|| self.device.as_ref().map(|device| device.index))
            .flatten()
    }
}

// ---------------------------------------------------------------------------
// 组件状态
// ---------------------------------------------------------------------------

/// 单个组件的可用状态（`check_components` 命令的返回元素）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ComponentStatus {
    pub kind: ComponentKind,
    /// 能否启动。
    pub ready: bool,
    /// 加速状态（只有重建/训练类组件有）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acceleration: Option<AccelerationStatus>,
    /// 详情（缺失原因、版本提示等）。
    pub detail: String,
    /// 组件位置（诊断用；前端不依赖它做逻辑判断）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
}

// ---------------------------------------------------------------------------
// 重建后端身份 / 能力
// ---------------------------------------------------------------------------

/// 后端实现形态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BackendFamily {
    /// 外部命令行程序。
    ExternalCli,
    /// 进程内实现。
    InProcess,
}

/// 重建后端的身份。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BackendIdentity {
    /// 稳定标识（例如 `colmap` / `ooo-sfm`）——用于遥测与诊断，**前端不要用它做逻辑分支**。
    pub id: String,
    /// 展示名（前端直接显示，避免在 TS 里硬编码实现名）。
    pub display_name: String,
    /// 版本串。
    pub version: String,
    pub family: BackendFamily,
}

/// 重建后端支持的阶段。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BackendStage {
    FeatureExtraction,
    IncrementalFeatureExtraction,
    SequentialMatching,
    ExhaustiveMatching,
    PairMatching,
    SparseMapping,
    MappingFromExisting,
    IncrementalMapping,
}

/// 重建后端的能力清单。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BackendCapabilities {
    pub stages: Vec<BackendStage>,
    /// 是否支持闭环检测（词袋树）。
    pub supports_loop_detection: bool,
    /// 是否支持两视图 track。
    pub supports_two_view_tracks: bool,
}

// ---------------------------------------------------------------------------
// 流水线
// ---------------------------------------------------------------------------

/// 流水线里的**组件类别**（事件与错误里标注"这一步是谁干的"）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PipelineComponent {
    System,
    Media,
    Reconstruction,
    Training,
}

/// 失败分类（前端据此选文案与建议；不含具体实现名）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FailureKind {
    /// 重建输入有问题（数据、匹配、初始对……）。
    ReconstructionInput,
    /// 重建存储/写出失败。
    ReconstructionStorage,
    /// 训练加速器不足。
    TrainingAccelerator,
    /// 训练设备掉线。
    TrainingDeviceLost,
    /// 训练数据集问题。
    TrainingDataset,
    /// 媒体处理失败。
    Media,
    /// 其它系统级失败。
    System,
}

/// 失败分类的中性标签（前端 i18n key 后缀）。
impl FailureKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::ReconstructionInput => "reconstructionInput",
            Self::ReconstructionStorage => "reconstructionStorage",
            Self::TrainingAccelerator => "trainingAccelerator",
            Self::TrainingDeviceLost => "trainingDeviceLost",
            Self::TrainingDataset => "trainingDataset",
            Self::Media => "media",
            Self::System => "system",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_labels_are_stable() {
        // 前端 i18n key 依赖这些字符串，改动要同步 `src/i18n/index.tsx`。
        assert_eq!(ComponentKind::Media.label(), "media");
        assert_eq!(ComponentKind::Reconstruction.label(), "reconstruction");
        assert_eq!(ComponentKind::Training.label(), "training");
    }

    #[test]
    fn failure_labels_are_stable() {
        assert_eq!(FailureKind::ReconstructionInput.label(), "reconstructionInput");
        assert_eq!(FailureKind::TrainingDeviceLost.label(), "trainingDeviceLost");
    }

    #[test]
    fn serialization_uses_camel_case() {
        let json = serde_json::to_string(&ComponentStatus {
            kind: ComponentKind::Reconstruction,
            ready: true,
            acceleration: None,
            detail: String::new(),
            location: None,
        })
        .expect("序列化");
        assert!(json.contains("\"kind\":\"reconstruction\""), "{json}");
        assert!(json.contains("\"ready\":true"), "{json}");
    }
}
