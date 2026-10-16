# 后端加速器管理机制

## 1. 三个独立的加速域

| 加速域 | 谁决定 | 探测方式 | 现状 |
| --- | --- | --- | --- |
| COLMAP SIFT 特征/匹配 | OOOSplat | `nvidia-smi` 主动探测 | NVIDIA-only（驱动≥580、CC≥7.5），否则 CPU |
| Caspar BA（全局光束法平差） | OOOSplat | 真实执行探测（8 点 fixtures 跑 `bundle_adjuster`） | 依赖 COLMAP 的 GPU index，独立阈值 + 熔断 |
| Brush 训练后端 | **Brush 自己** | OOOSplat 不探测，从日志回读设备 | 跨厂商：Win/Linux 用 Vulkan，macOS 用 Metal |

关键点：**COLMAP 是"受管加速器"（OOOSplat 探测并传参），Brush 是"自治加速器"
（后端选择在引擎内部，OOOSplat 只在特定场景轻推 + 读回结果）**。

## 2. 统一接口约定的实际形态

没有 trait 多态接口（全库唯一的 trait 是 `FrameSelectionStrategy`），统一性由四样东西保证：

### 2.1 单一数据契约 `ColmapAccelerationStatus`

`src-tauri/src/engines/health.rs:92-143`：

```rust
pub struct ColmapAccelerationStatus {
    backend: ColmapBackend,               // Cpu | Gpu
    detection_state: GpuDetectionState,   // Ready | TemporarilyUnavailable | Unavailable
    reason_code: AccelerationReasonCode,  // 12 种机器可读原因码
    reason: String,                       // 人类可读（前端本地化）
    device: Option<GpuDeviceInfo>,        // index/name/driver/compute_capability/total_memory_mb
    requirements: AccelerationRequirements, // 来自引擎 manifest 的最低驱动/CC
    detected_nvidia_device_count: usize,
}
```

消费方只通过 `use_gpu()` / `gpu_index()` / `usable_gpu_total_memory_mb()` /
`planning_gpu_total_memory_mb()` 使用，不关心探测过程。

### 2.2 探测服务与统一策略 `AccelerationProbeService`

`health.rs:153-173, 825-891`，全局单例 + 三档策略：

| 策略 | 场景 | 行为 |
| --- | --- | --- |
| `Cached` | 启动时检查引擎 | 10 分钟 TTL，`try_lock` 不排队 |
| `Refresh` | `check_colmap_acceleration` | 单次强制刷新 |
| `Task` | 任务执行前 | 3 次退避重试，**从不用缓存兜底** |

### 2.3 决策纯函数 `choose_acceleration`

`health.rs:956-1052`：`Vec<GpuDeviceInfo> + 要求 → 状态`，按优先级给出
reason code（Ready → ComputeCapabilityTooLow → … → DriverVersionUnknown）。
阈值来自 manifest 的 `cudaCompatibility` 块，非硬编码。

### 2.4 端到端传播

`EngineStatus.acceleration` → `PipelineEvent.acceleration`（`runner.rs:254`）
→ 前端 appStore → `TaskConfiguration.tsx` 显示。前端类型镜像在
`src/types/pipeline.ts:8-24`。

## 3. 消费点

| 消费点 | 位置 | 用途 |
| --- | --- | --- |
| COLMAP 特征/匹配参数 | `colmap.rs:82-116` | CLI 家族适配 + 传 gpu_index |
| Caspar BA | `colmap_ba.rs:40-119` | 独立执行探测 + 失败熔断 |
| Brush 启动策略 | `brush.rs:42-124` | 仅 Windows + 唯一 NVIDIA 时注入离散 GPU 环境变量 |
| Brush 训练 profile | `presets/quality.rs` | 按 `usable_gpu_total_memory_mb()` 选档 |
| Brush 设备回读 | `pipeline/runtime.rs:100-101` | 从日志解析实际设备 |
| 资源采样 | `process/resources.rs:100-158` | nvidia-smi 采样 GPU 占用（macOS 标 `unsupported`） |
| 诊断报告 | `diagnostics/hardware.rs:121-159` | Windows WMI / macOS system_profiler / Linux lspci |

## 4. 跨平台准备情况

### 已具备的接缝

- 数据结构全部 serde，可增补字段（`GpuDeviceInfo.total_memory_mb` 已有 `#[serde(default)]` 先例）；
- 阈值数据驱动（每平台一份 manifest）；
- `AccelerationReasonCode` 封闭枚举，可扩展；
- `ProbePolicy` 机制与"探什么"解耦；
- Brush 域天然多平台（Metal/Vulkan）；
- 诊断采集已有三平台分化模式。

### 阻碍扩展的耦合

- **无行为抽象**：`nvidia-smi` 调用硬编码在三处（`health.rs` / `resources.rs` / `diagnostics/hardware.rs`）；
- **`ColmapBackend` 只有 `Cpu|Gpu`，无 vendor 维度**；`GpuDeviceInfo` 无 vendor 字段；
- **CUDA 语义泄漏进通用层**：`compute_capability`、`detected_nvidia_device_count`、`NvidiaSmiNotFound`；
- **命名与归属不一致**：状态叫 `ColmapAccelerationStatus`，却被 Brush profile 选择依赖；
- **macOS 是硬短路而非另一种探测实现**（`health.rs:577`、`colmap_ba.rs:41/73`）。

### 若支持 AMD/Intel/macOS-Metal 的改造路径

1. 泛化为 vendor 无关的 `AccelerationStatus`（加 `vendor` 字段），
   `ColmapBackend` 扩为 `{Cpu, Cuda, Rocm, Metal, Vulkan, …}`；
2. 定义探测抽象（trait 或函数注册表），把三处 nvidia-smi 收敛为 `GpuProbe` 实现；
3. requirements 继续走 manifest（每 vendor 一个 compatibility 块）；
4. 消费点基本已满足（COLMAP 参数按 vendor 映射、Brush 域基本不动）。

## 5. macOS / Metal 现状与结论

- macOS 的 COLMAP 内置构建是 **CPU-only**（`engines/colmap-runtime.json` 中
  `cuda:false, caspar:false`），不是 OOOSplat 管理层的适配问题，而是**上游 COLMAP
  没有 Metal/Vulkan 的 GPU 实现**（GPU SIFT 只有 CUDA，OpenGL/SiftGPU 老路径已移除）；
- 因此"给 COLMAP/Caspar 适配 Metal"不是改 OOOSplat 的 API（CLI 不变），而是写引擎
  内部的 Metal 内核（人月级 fork 开发）；Caspar 的 Metal 收益也被项目自身基准
  （`docs/caspar-*-report-*.md`）证明不稳定；
- 与"能否在 M1 上跑"是两件事：COLMAP CPU 本来就能在 M1 跑，卡点是系统下限
  （项目全线锁 **macOS 15.0**：`tauri.macos.conf.json`、`manifest.macos.json`、
  `build-*-macos.*`）；
- **现成可做的高收益小改动**：解开 `usable_gpu_total_memory_mb()` 在 macOS 返回
  `None` 的问题（`health.rs:111` → `quality.rs:275-285`），让 Brush 在 Apple Silicon
  上按统一内存选更高 profile；以及补 macOS 的 GPU 资源采样（目前 `unsupported`）。
- 若要在 macOS 上游加速 SIFT，可参考 **`byplay-io/colmap-metal`**（COLMAP 的 Metal
  SIFT fork，用法与标准 COLMAP 一致），但只加速特征提取，且需并入自家 fork 长期维护。
