//! 自研后端的健康 / 能力报告（M7）。
//!
//! 主分支的 `engines/health.rs` 通过**探测 COLMAP 二进制**（`-h` 嗅探 + 路径检查）来
//! 回答"引擎能不能用、走 CPU 还是 GPU"。自研后端是**进程内**的，没有二进制可探测，
//! 所以这里反过来：由后端自己声明能力，`health.rs` 在 `feature = "backend-ooo-sfm"`
//! 下直接取用本文件的结果。
//!
//! 字段与语义沿用 `crate::contracts` 的**中性**结构（`AccelerationStatus` /
//! `ComponentStatus`），前端零改动。

use crate::contracts::{
    AcceleratorDevice, AcceleratorRequirements, AccelerationMode, AccelerationReason,
    AccelerationState, AccelerationStatus, BackendCapabilities, BackendFamily, BackendIdentity,
    BackendStage, ComponentKind, ComponentStatus,
};

use super::LOCKED_VERSION;

/// 自研后端对外声明的阶段能力。
///
/// 主分支靠 `feature_extractor -h` / `sequential_matcher -h` / ... 的嗅探来确认 CLI
/// 家族；自研后端直接给出这份清单，`health.rs` 不再需要跑子进程。
pub const CAPABILITIES: &[BackendStage] = &[
    BackendStage::FeatureExtraction,
    BackendStage::IncrementalFeatureExtraction,
    BackendStage::SequentialMatching,
    BackendStage::ExhaustiveMatching,
    BackendStage::PairMatching,
    BackendStage::SparseMapping,
    BackendStage::MappingFromExisting,
    BackendStage::IncrementalMapping,
];

/// 数据库 / 模型的**输入版本**（对应 COLMAP 的 `COLMAP_INPUT_VERSION`）。
///
/// 上游 4.2.1 的 `database.db` schema 版本是 20；自研后端沿用同一 schema（读写都在
/// `apple/format/database.rs`），因此这里报告同一个值 —— M9 的"输入版本纳入自研
/// 版本"要做的正是把这两个值的**来源**分开：版本号属于格式层，后端版本属于引擎层。
pub const INPUT_VERSION: i64 = 20;

/// 后端身份（供 `engines::backend` 契约层与界面展示）。
pub fn identity() -> BackendIdentity {
    BackendIdentity {
        id: "ooo-sfm".into(),
        display_name: "自研 SfM 后端".into(),
        version: LOCKED_VERSION.to_string(),
        family: BackendFamily::InProcess,
    }
}

/// 后端能力清单。
pub fn capabilities() -> BackendCapabilities {
    BackendCapabilities {
        stages: CAPABILITIES.to_vec(),
        // 闭环检测（词袋树）尚未实现；如实上报，不假装支持。
        supports_loop_detection: false,
        supports_two_view_tracks: true,
    }
}

/// Metal 后端是否编译进来（编译期）。
pub fn metal_compiled_in() -> bool {
    super::apple::gpu::compiled_in()
}

/// Metal 后端是否真的在用（运行期：设备存在 + 内核编译成功）。
pub fn metal_active() -> bool {
    super::apple::gpu::active()
}

/// 自研后端的加速状态。
///
/// * Metal 可用 → `GpuReady`（`mode = Gpu`，device 名为 Apple 芯片名；`driver_version`
///   报 Metal feature set 是我们能拿到的最接近的等价物；`compute_capability` 留空）。
/// * 否则 → `CpuOnly`，与主分支 macOS 分支的既有语义一致（D2 前端零改）。
pub fn acceleration() -> AccelerationStatus {
    let requirements = AcceleratorRequirements {
        minimum_driver_version: String::new(),
        minimum_compute_capability: String::new(),
    };
    if metal_active() {
        return AccelerationStatus {
            mode: AccelerationMode::Gpu,
            state: AccelerationState::Ready,
            reason_code: AccelerationReason::GpuReady,
            reason: "ooo-sfm 自研后端：Metal 计算内核已就绪".into(),
            device: Some(AcceleratorDevice {
                index: 0,
                name: metal_device_name().unwrap_or_else(|| "Apple GPU".into()),
                // Metal 没有"驱动版本"的概念；`device.os_version()` 在 metal crate
                // 里是私有方法，拿不到系统版本，所以这里报一个固定、诚实的标签。
                driver_version: "Metal".into(),
                compute_capability: String::new(),
                total_memory_mb: None,
            }),
            requirements,
            detected_device_count: 0,
        };
    }
    AccelerationStatus {
        mode: AccelerationMode::Cpu,
        state: AccelerationState::Unavailable,
        reason_code: if metal_compiled_in() {
            // 编译进来了但设备/内核不可用：说明白，别让前端以为是"没编 GPU"。
            AccelerationReason::ProbeFailed
        } else {
            AccelerationReason::CpuOnly
        },
        reason: if metal_compiled_in() {
            "ooo-sfm 自研后端：Metal 内核不可用，已回退 CPU".into()
        } else {
            "ooo-sfm 自研后端：未启用 `backend-ooo-sfm-metal` feature，走 CPU".into()
        },
        device: None,
        requirements,
        detected_device_count: 0,
    }
}

/// 自研后端的组件状态（进程内，没有可执行文件要校验）。
///
/// `location` 仍按调用方给的路径回填，只用于界面显示"配置指向哪里"，不代表要存在。
pub fn component_status(path: std::path::PathBuf) -> ComponentStatus {
    let acceleration = acceleration();
    let accelerated = acceleration.accelerated();
    ComponentStatus {
        kind: ComponentKind::Reconstruction,
        ready: true,
        acceleration: Some(acceleration),
        detail: format!(
            "自研后端（进程内，无外部二进制）；能力：{} 项；加速：{}",
            CAPABILITIES.len(),
            if accelerated { "Metal" } else { "CPU" }
        ),
        location: Some(path.display().to_string()),
    }
}

fn metal_device_name() -> Option<String> {
    super::apple::gpu::device_name()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_cover_the_pipeline_stages() {
        for stage in [
            BackendStage::FeatureExtraction,
            BackendStage::SequentialMatching,
            BackendStage::ExhaustiveMatching,
            BackendStage::SparseMapping,
            BackendStage::IncrementalMapping,
        ] {
            assert!(CAPABILITIES.contains(&stage), "能力清单缺少 {stage:?}");
        }
    }

    #[test]
    fn component_status_is_always_ready() {
        let status = component_status(std::path::PathBuf::from("/nonexistent/colmap"));
        assert!(status.ready);
        assert_eq!(status.kind, ComponentKind::Reconstruction);
        assert!(!status.detail.is_empty());
    }

    #[test]
    fn acceleration_matches_the_metal_feature() {
        let status = acceleration();
        if metal_compiled_in() && metal_active() {
            assert!(status.accelerated());
            assert!(status.device.is_some(), "GPU 状态必须带设备信息");
        } else {
            assert!(!status.accelerated());
            assert!(status.device.is_none());
        }
    }
}
