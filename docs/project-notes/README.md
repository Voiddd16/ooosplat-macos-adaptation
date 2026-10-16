# OOOSplat 项目理解文档

本目录用于沉淀对 OOOSplat 整个项目的理解，供开发与排查参考。内容基于对当前工作树
（版本 0.6.1，commit `c72207e`）源码的阅读整理，可能与后续版本存在差异。

## 目录

| 文档 | 内容 |
| --- | --- |
| [architecture.md](./architecture.md) | 整体架构、技术栈、核心业务流程、模块地图 |
| [colmap-usage.md](./colmap-usage.md) | **项目对 COLMAP 功能的使用清单**（子命令、参数、读取的产物、未用功能） |
| [accelerator-management.md](./accelerator-management.md) | 后端加速器（COLMAP/Caspar/Brush）的管理机制、统一接口约定与跨平台现状 |
| [colmap-dependency-map.md](./colmap-dependency-map.md) | COLMAP 依赖图（谁依赖、依赖二进制还是格式）与替换接口契约 |
| [ooo-sfm-implementation-plan.md](./ooo-sfm-implementation-plan.md) | **自研 SfM 后端（ooo-sfm）替换 COLMAP 的完整实施方案**（分阶段计划） |

## 一句话定位

OOOSplat 是一款本地优先的桌面应用：把普通环绕拍摄的视频/图片序列，一键转换为
3D Gaussian Splatting，并内置预览编辑与多格式导出。前端 React + Tauri，后端 Rust，
原生引擎为内置的 FFmpeg / COLMAP（fork） / OOOBrush。

## 相关文档（仓库既有）

- 产品与路线：[`../../README.md`](../../README.md)、[`../../ROADMAP.md`](../../ROADMAP.md)
- 自动优化参数：[`../pipeline-optimization-config.zh-CN.md`](../pipeline-optimization-config.zh-CN.md)
- MCP 协议：[`../mcp-v1.md`](../mcp-v1.md)
- 引擎策略：[`../../engines/README.md`](../../engines/README.md)
