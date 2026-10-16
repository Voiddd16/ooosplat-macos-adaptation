# COLMAP 依赖图与替换接口契约

> 用途：评估用"自研最小子集工具"替换外部 COLMAP 的影响面。基于版本 0.6.1 源码整理。

## 0. 核心结论

1. **项目内没有任何第三方工具在代码/二进制层面依赖 COLMAP。**
2. **Brush 依赖的是 COLMAP 的"文件格式"（稀疏模型 + 图像布局），不是 COLMAP 的二进制或库。**
   Brush 只把数据集目录当作输入，读取标准的 `sparse/0/{cameras,images,points3D}.bin`。
   因此只要替换工具**产出同格式的稀疏模型**，Brush 完全不受影响。
3. **FFmpeg 与 COLMAP 无任何依赖**（仅目录命名 `colmap_frames` 沿用习惯）。
4. Caspar 与离线词袋树是 **COLMAP fork 的一部分**，会随 COLMAP 一起被替换（或按需保留）。
5. 替换工具必须同时满足 **CLI 契约** 与 **数据格式契约**（见第 3 节）。

## 1. 依赖分类总览

| 依赖类型 | 受影响文件/位置 | 替换时是否必须改 |
| --- | --- | --- |
| 二进制 / CLI 调用 | `engines/colmap.rs`、`engines/colmap_ba.rs`、`engines/colmap/vocabulary.rs`、`engines/health.rs`、`pipeline/runner.rs` | 是（替换为调用新工具） |
| 数据格式（读） | `reconstruction/colmap_model.rs`、`reconstruction/validator.rs`、`pipeline/runner.rs` | 保持兼容则不用改 |
| 数据格式（写，给 Brush） | `pipeline/runner.rs::prepare_brush_dataset` | 输出格式一致则不用改 |
| SQLite 数据库（读） | `engines/colmap_ba.rs`、`pipeline/runner.rs::validate_database_camera` | 视是否保留 Caspar/补拍校验 |
| 引擎路径与身份校验 | `engines/health.rs`、`project/manager.rs` | 是 |
| 构建 / 打包 / CI | `engines/*.json`、`scripts/*`、`.github/workflows/*`、`src-tauri/tauri.*.conf.json` | 是 |
| 前端展示 | `src/types/pipeline.ts`、`App.tsx`、`TaskConfiguration.tsx`、`i18n/index.tsx` 等 | 视状态字段是否保留 |
| 文档 / 法律 / 遥测 | `docs/*`、`licenses/*`、`telemetry/*`、`db/migrations/*`、`server/*` | 视情况 |

## 2. 二进制 / CLI 依赖清单（功能面）

| 调用点 | COLMAP 子命令 | 特征 |
| --- | --- | --- |
| `runner.rs:2268/2283/3239` | `feature_extractor` | 普通 / 自动优化 / 补拍增量三种参数 |
| `runner.rs:2351` | `sequential_matcher` | 视频顺序匹配 + 闭环 |
| `runner.rs:2341/1846` | `exhaustive_matcher` | 图片 / 补拍穷举匹配 |
| `runner.rs:3282` | `matches_importer` | Bridge 补帧 `--match_type pairs` |
| `runner.rs:2397/3308/1905` | `mapper` | 普通 / 从已有模型 / 补拍增量 |
| `colmap_ba.rs:216` | `bundle_adjuster` | 仅 Caspar 能力探测 |
| `health.rs:482-487` | 上述 6 个命令 `-h` | CLI 家族与必需参数探测 |

引擎身份与路径：

- `engines/health.rs:467` 缺失状态、`:513`/`:617` 引擎状态、`is_locked_build`（`:533`）要求
  `-h` 输出含 `COLMAP 4.2.1 (Commit bd1fcf6…)`；
- `engines/colmap.rs:13-14` 锁定版本/commit；`:118-124` `require_verified_cli`（仅检查文件存在）；
- `engines/health.rs:605` 触发 Caspar 能力探测；
- `project/manager.rs:28-55` 工作目录 `work/colmap`、`work/colmap_frames`。

## 3. 替换工具必须实现的接口契约

### 3.1 CLI 契约（被 `detect_cli_family` 与参数构造依赖）

替换工具需满足二者之一，并暴露对应参数名（`colmap.rs:45-116`）：

- **Modern4**：`feature_extractor --FeatureExtraction.use_gpu/--FeatureExtraction.max_image_size`、
  `*_matcher --FeatureMatching.use_gpu/--FeatureMatching.gpu_index`
- **Legacy39**：`--SiftExtraction.*` / `--SiftMatching.*`

必需的参数（来自 `health.rs:633-660` 的必需选项检查）：

| 命令 | 必需参数 |
| --- | --- |
| `feature_extractor` | `--database_path`、`--image_path`、`--ImageReader.*`（camera_model/single_camera 等） |
| `sequential_matcher` | `--database_path`、`--SequentialMatching.overlap` |
| `exhaustive_matcher` | `--database_path` |
| `matches_importer` | `--database_path`、`--match_list_path`、`--match_type` |
| `mapper` | `--database_path`、`--image_path`、`--output_path`、`--Mapper.*`（含 tri_ignore_two_view_tracks / ba_* / image_list_path / fix_existing_frames） |
| `bundle_adjuster` | `--input_path`、`--output_path` |

此外 `is_locked_build` 会校验 `-h` 首行含 `COLMAP 4.2.1 (Commit bd1fcf65…)`。
替换工具要么伪造该身份串，要么项目改走 `local-colmap` feature / 调整身份校验。

本项目实际传入的参数全集见 [`colmap-usage.md`](./colmap-usage.md) 第 2 节。

### 3.2 数据格式契约（输出，被本项目与 Brush 读取）

**稀疏模型二进制**（`reconstruction/colmap_model.rs`、`validator.rs`）：

- `sparse/<n>/cameras.bin`：`u64` 数量；每台相机 `u32 id` + `i32 model_id` + `u64 width` +
  `u64 height` + `f64` 参数（数量由模型决定）。模型枚举：0 SIMPLE_PINHOLE(3)、1 PINHOLE(4)、
  2 SIMPLE_RADIAL(4)、3 RADIAL(5)、4 OPENCV(8)、5 OPENCV_FISHEYE(8)、6 FULL_OPENCV(12)、
  7 FOV(5)、8 SIMPLE_RADIAL_FISHEYE(4)、9 RADIAL_FISHEYE(5)、10 THIN_PRISM_FISHEYE(12)、
  11 RAD_TAN_THIN_PRISM_FISHEYE(16)。
- `sparse/<n>/images.bin`：`u64` 数量；每张图 `u32 id` + `7×f64`（qvec 4 + tvec 3）+
  `u32 camera_id` + 以 `\0` 结尾的 name + `u64` 点数 + `n×24` 字节的 2D/3D 对应。
- `sparse/<n>/points3D.bin`：`u64` 数量（本项只读计数，用于注册率校验）。
- `rigs.bin` / `frames.bin`（COLMAP 4.x 可选）：若存在必须同时存在，`reusable_model_files`
  会一并保留用于复用（`colmap_model.rs:145-181`）。

**给 Brush 的数据集布局**（`runner.rs:4559-4580`，Brush `train` 以该目录为位置参数）：

```
dataset/
  images/<帧文件>            # 硬链接或复制
  sparse/0/{cameras,images,points3D}.bin
```

**注册率校验**（`reconstruction/validator.rs`）：`images.bin` 计数 = 注册图像数，
`points3D.bin` 计数 = 三维点数，需 > 0；注册率阈值来自自动优化配置。

**多模型**（`runner.rs:4434-4500`）：从 `sparse/` 下选择最佳子模型
（`best_sparse_model*`），因此输出目录需符合 `sparse/<n>/` 约定。

**进度解析**：matching/mapper/feature 的输出行会被 observer 解析为进度/日志。

### 3.3 SQLite 数据库契约（读）

- `engines/colmap_ba.rs:121-138`：读 `keypoints`、`two_view_geometries` 表的 `rows` 列求和，
  用于 Caspar 阈值判断（**仅自动优化路径**，非必需）。
- `runner.rs:4383-4417`：读 `cameras` 表的 `camera_id / model / width / height`，
  用于补拍时校验原项目相机一致性。

若替换工具使用自研中间数据库，这两处读取需要相应调整（或保留兼容 schema）。

### 3.4 辅助资源

- 离线词袋树 `vocab_tree_faiss_flickr100K_words256K.bin`（72,412,636 字节 + SHA-256），
  路径与哈希来自 `engines/colmap-runtime.json`，校验逻辑在 `engines/colmap/vocabulary.rs`。
  若替换工具自研闭环检测，可去掉该资源及其校验。

## 4. 构建 / 打包 / CI 依赖

| 位置 | 内容 |
| --- | --- |
| `engines/manifest.json` / `manifest.linux.json` / `manifest.macos.json` | COLMAP 版本/哈希/CUDA 兼容性锁定 |
| `engines/colmap-build.json` | 源码构建锁（commit、CUDA、Caspar、vcpkg、CMake/Ninja） |
| `engines/colmap-runtime.json` | 运行时发布锁（三平台归档、词汇树） |
| `scripts/setup-colmap-runtime.mjs`、`brush-runtime.mjs`、`setup-engines*.{ps1,sh}`、`local-tauri.mjs` | 安装与本地开发 |
| `scripts/build-colmap-{windows.ps1,linux.sh,macos.sh}`、`colmap-macos-common.sh`、`package-colmap-*.mjs`、`collect-linux-runtime.mjs` | 源码构建与打包 |
| `scripts/colmap-runtime.mjs`、`verify-colmap-lock.mjs`、`verify-engines-*`、`smoke-colmap-image-io.mjs`、`verify-macos-colmap-runtime.mjs` | 校验与冒烟 |
| `scripts/verify-licenses.mjs` / `.ps1`、`licenses/THIRD_PARTY_NOTICES.txt` | 许可证合规 |
| `.github/workflows/{windows,ubuntu,macos,colmap-engines,macos-engines}.yml` | CI（含 `verify:colmap-lock`、COLMAP 冒烟、安装包内容检查） |
| `src-tauri/tauri.{conf,windows.conf,linux.conf,macos.conf}.json` | 打包引擎资源与许可证文件 |
| `src-tauri/Cargo.toml:14` | feature `local-colmap`（仅放宽本地身份校验） |
| `.gitignore` | 忽略 `engines/**` 二进制 |

## 5. 前端 / 展示层依赖

| 文件 | 内容 |
| --- | --- |
| `src/types/pipeline.ts:2,6-24` | `EngineKind "colmap"`、`ColmapBackend`、`ColmapAccelerationStatus`、`EngineStatus.colmapCliFamily` |
| `src/app/App.tsx`、`src/components/TaskConfiguration.tsx`、`SharedTaskDetail.tsx` | 加速状态/引擎状态展示 |
| `src/i18n/index.tsx` | COLMAP/GPU 相关文案 |
| `src/lib/backend.ts:75` | `checkColmapAcceleration`（`check_colmap_acceleration`） |
| `src/lib/pipelineError.ts`、`errorSummary.ts` | COLMAP 失败分类文案 |
| `src/stores/appStore.ts` | `colmapAcceleration` / `taskColmapAcceleration` |

## 6. 其他

- 遥测：`telemetry/event.rs`、`telemetry/mod.rs`（错误码/阶段，字段名含 colmap/planner 语义）。
- 数据库迁移：`db/migrations/20260928_...planner_evaluations.sql`（列含 planner/brush/SfM 指标，不直接依赖 COLMAP 二进制）。
- 服务端集成：`server/telemetry/planner-evaluation.schema.ts`、`server/diagnostics/*`。
- 文档：`README*`、`ROADMAP*`、`docs/*`（用户可见的系统要求/流程说明）。

## 7. 替换时的最小改动集（建议顺序）

1. 新增替换工具的 Rust 封装（替代 `engines/colmap.rs` 的 6 个子命令接口）；
2. 保持 `sparse/0/*.bin` + `images/` 输出布局不变（Brush 与校验零改动）；
3. 决定是否保留：Caspar BA（`colmap_ba.rs`）、闭环词袋树（`vocabulary.rs`）、
   `database.db` 的两处读取；
4. 调整 `health.rs` 的引擎身份/CLI 家族校验与 `EnginePaths`；
5. 更新 `engines/*.json`、`scripts/*`、`.github/workflows/*`、`tauri.*.conf.json`、
   `verify-licenses` 与许可证文件；
6. 前端 `EngineKind`/加速状态按新工具语义调整（或保留字段以复用 UI）。
