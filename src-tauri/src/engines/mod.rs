pub mod backend;
pub mod brush;

// ooo-sfm 门面：feature 关闭 = 主分支 COLMAP 后端；开启 = 自研后端（决策 D1/D6）。
// 两条分支的文件同名同签名 → 调用点（pipeline/runner.rs 等）一行不改。
#[cfg(not(feature = "backend-ooo-sfm"))]
pub mod colmap;

#[cfg(feature = "backend-ooo-sfm")]
#[path = "ooo_sfm/mod.rs"]
pub mod colmap;

#[cfg(not(feature = "backend-ooo-sfm"))]
pub mod colmap_ba;

#[cfg(feature = "backend-ooo-sfm")]
#[path = "ooo_sfm/ba.rs"]
pub mod colmap_ba;

pub mod ffmpeg;
pub mod ffprobe;
pub mod health;

// 契约类型统一来自 `crate::contracts`（前端可见的中性词汇表）。
pub use crate::contracts::{
    AcceleratorDevice, AcceleratorRequirements, AccelerationMode, AccelerationReason,
    AccelerationState, AccelerationStatus, BackendCapabilities, BackendFamily, BackendIdentity,
    BackendStage, ComponentKind, ComponentStatus, FailureKind,
};
pub use health::{EnginePaths, ToolKind, ToolStatus};
