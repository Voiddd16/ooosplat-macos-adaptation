pub mod brush;

// ooo-sfm 门面：feature 关闭 = 主分支 COLMAP 后端；开启 = 自研后端（决策 D1/D6）。
// 两条分支的文件同名同签名 → 调用点（pipeline/runner.rs 等）一行不改。
#[cfg(not(feature = "ooo-sfm"))]
pub mod colmap;

#[cfg(feature = "ooo-sfm")]
#[path = "ooo_sfm/mod.rs"]
pub mod colmap;

#[cfg(not(feature = "ooo-sfm"))]
pub mod colmap_ba;

#[cfg(feature = "ooo-sfm")]
#[path = "ooo_sfm/ba.rs"]
pub mod colmap_ba;

pub mod ffmpeg;
pub mod ffprobe;
pub mod health;

pub use health::{
    AccelerationReasonCode, AccelerationRequirements, ColmapAccelerationStatus, ColmapBackend,
    EngineKind, EnginePaths, EngineStatus, GpuDetectionState, GpuDeviceInfo,
};
