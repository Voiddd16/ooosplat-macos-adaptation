# 整体架构与核心流程

## 1. 技术栈

| 层 | 技术 |
| --- | --- |
| 桌面外壳 | Tauri 2（Windows / macOS arm64 / Ubuntu 24.04） |
| 前端 | React 19 + TypeScript + Zustand 5 + PlayCanvas（WebGPU 优先 / WebGL2 回退）+ mediabunny（视频编码） |
| 后端 | Rust（约 3.3 万行，crate `ooo-splat`），tokio 异步运行时 |
| 原生引擎 | 内置 FFmpeg/FFprobe、COLMAP 4.2.1（fork，含 Caspar）、OOOBrush `brush-cli` 1.0.0 |
| 数据/服务 | 本地文件（项目目录）+ 可选遥测/诊断后端（PostgreSQL 迁移在 `db/`，集成源码在 `server/`） |

## 2. 分层结构

```
前端 React (src/)
  App.tsx            任务工作区 / 创建任务 / 全部对话框编排
  GaussianViewer/     PlayCanvas 预览与编辑（选择、变换、裁剪、导出）
  stores/             Zustand：appStore（任务运行态）、gaussianTransformStore（编辑态）
  lib/backend.ts      唯一的 Tauri 边界（invoke / listen / 等待器）
        │  IPC（invoke + 事件）
后端 Rust (src-tauri/src/)
  commands/           Tauri 命令层（约 40 个 IPC）
  tasks/              应用级任务注册表（GUI/MCP/CLI 共用；单活动任务，持久化）
  pipeline/runner.rs  核心流水线编排（检查点复用、OOM 重试、Bridge 补帧）
  engines/            FFmpeg / COLMAP / Brush 封装 + GPU 探测
  process/            子进程管理（Job Object / process group、取消、日志泵）
  project/            项目目录、原子写、检查点 schema
  planner/ 等         Bridge 补帧规划、PLY 变换、遥测、诊断、MCP
        │  子进程调用
原生引擎 (engines/)
  FFmpeg / COLMAP 4.2.1 / OOOBrush —— 二进制不入 Git，按 SHA-256 四级锁定
```

## 3. 核心业务流程（生成任务）

```
前端 startPipeline → create_gui_task + start_gui_task
 → TaskService.start_inner（单活动任务、tasks.json 原子持久化、250ms 节流通知）
 → PipelineRunner.generate
   1. 引擎体检（锁定版本校验 + NVIDIA 驱动/CC 探测）
   2. 建项目目录（source/ work/ logs/ + project.json + state.json）
   3. ProbingVideo      FFprobe 探测（旋转/alpha/帧率）
   4. PlanningFrames    抽帧规划（planner 或均匀抽帧）
   5. ExtractingFrames  FFmpeg 抽帧 / 图片硬链接复用 + Alpha Mask
   6. ExtractingFeatures COLMAP 特征提取（GPU/CPU 自动选择）
   7. Matching         视频顺序匹配(+闭环) / 图片穷举匹配
   8. Reconstructing    COLMAP 增量重建（Caspar BA，失败回退 Ceres）
   9. Validating        注册率校验；不足时 Bridge 补帧重试（失败回滚）
   10. TrainingSplats   Brush 训练（按显存选 profile，OOM 降档重试）
   11. Exporting        校验 PLY → 原子发布 final.ply
 → TaskService.finish → task-update 事件 → 前端 waitForSharedTask 收到结果
```

阶段状态机见 `src-tauri/src/pipeline/state.rs`，检查点 schema 见
`src-tauri/src/project/metadata.rs`（`PipelineStateFile`）。

## 4. 关键机制

- **断点续跑**：检查点写入 `state.json`（features/matching/reconstruction/brush 完成标记），
  重启后从最近阶段继续；损坏则回退重跑。
- **子进程管理**：Windows 用 Job Object（KILL_ON_JOB_CLOSE），Unix 用 process group，
  SIGTERM→3s→SIGKILL 三连取消；stdout/stderr 分流写日志。
- **降级链**：WebGPU→WebGL2、COLMAP CUDA→CPU、Caspar BA→Ceres、Brush OOM 降档、
  Bridge 失败回滚。
- **任务拥有权**：传输层（GUI/MCP/CLI）不持有生成 Future，统一由 `TaskService` 管理。

## 5. 引擎供应链（简述）

- 二进制不进 Git，正式运行时来自固定 Release 资产，四级锁定（上游 commit +
  归档 SHA-256 + 内部 SHA256SUMS + BUILD-INFO）。
- 安装/校验脚本在 `scripts/`，CI 在 `.github/workflows/`（三平台各自
  前端测试 + 引擎严格校验 + Rust lint/test + 打包 + 安装包内容检查）。
- 详见 [`../../engines/README.md`](../../engines/README.md)。

## 6. 平台差异化现状

| 平台 | FFmpeg | COLMAP | Brush |
| --- | --- | --- | --- |
| Windows x64 | 内置（BtbN LGPL） | 内置，CUDA + Caspar | 自选 Vulkan 后端 |
| Linux x86_64 | 系统 `/usr/bin` | 内置，CUDA + Caspar | 自选 Vulkan 后端 |
| macOS 15+ arm64 | 内置（源码构建） | 内置，**CPU-only（Ceres）** | 自选 **Metal** 后端 |

macOS 上 COLMAP 为 CPU-only 构建（`engines/colmap-runtime.json` 中 `cuda:false, caspar:false`），
下限锁定 macOS 15.0。
