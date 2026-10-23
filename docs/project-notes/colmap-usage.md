# OOOSplat 对 COLMAP 的功能使用清单

> 结论先行：OOOSplat 只把 COLMAP 当作**纯 CLI 工具**使用，覆盖 **6 个子命令**
> （`feature_extractor` / `sequential_matcher` / `exhaustive_matcher` /
> `matches_importer` / `mapper` / `bundle_adjuster`），并且**只用稀疏重建（SfM）+ GPU SIFT**，
> **不使用稠密重建（MVS）**。它不使用 COLMAP GUI，也不链接 COLMAP 的 C++ 库，只解析
> CLI 的输出文件（SQLite 数据库与二进制稀疏模型）。

锁定版本：**COLMAP 4.2.1（fork，commit `bd1fcf65…`，自编译 CLI-only）**，见
`src-tauri/src/engines/colmap.rs:13-14`。

## 1. 子命令总览

| COLMAP 子命令 | OOOSplat 封装函数 | 用途 |
| --- | --- | --- |
| `feature_extractor` | `extract_features` / `extract_features_quality_v2` / `extract_incremental_features` | SIFT 特征提取（普通 / 自动优化 / 补拍增量） |
| `sequential_matcher` | `match_sequential` | 视频帧顺序匹配（含可选闭环检测） |
| `exhaustive_matcher` | `match_exhaustive` | 图片序列穷举匹配 |
| `matches_importer` | `match_pairs` | Bridge 补帧的指定配对匹配（`--match_type pairs`） |
| `mapper` | `map` / `map_from_existing` / `map_incremental` | 增量 SfM 稀疏重建（含从已有模型续建、补拍增量） |
| `bundle_adjuster` | `colmap_ba::probe` | 仅用于 Caspar GPU BA 能力的真实执行探测 |

此外，引擎体检会对上述前 6 个命令逐个执行 `-h`，用于**探测 CLI 家族**与必需参数，
见 `src-tauri/src/engines/health.rs:482-487`、`colmap_command_has_required_options`。

### 1.1 CLI 家族自动适配

COLMAP 4.x 与 3.9 的 SIFT GPU 参数名不同，OOOSplat 运行时探测并适配
（`colmap.rs:45-116`）：

| 家族 | 特征提取参数 | 匹配参数 |
| --- | --- | --- |
| `Modern4`（≥ 4.x） | `--FeatureExtraction.use_gpu` / `.gpu_index` | `--FeatureMatching.use_gpu` / `.gpu_index` |
| `Legacy39`（3.9） | `--SiftExtraction.use_gpu` / `.gpu_index` | `--SiftMatching.use_gpu` / `.gpu_index` |

GPU 不可用时传 `use_gpu=0` 且**不传 index**（`colmap.rs:492, 618`）。

## 2. 各阶段参数详解

### 2.1 特征提取 `feature_extractor`

参数构造：`feature_extraction_args`（`colmap.rs:469-508`）、
`incremental_feature_extraction_args`（`colmap.rs:535-571`）。

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `--ImageReader.camera_model` | `SIMPLE_RADIAL` | 固定相机模型 |
| `--ImageReader.single_camera` | `1` | 视频/图片序列共享同一相机 |
| `--ImageReader.mask_path` | 目录 | 透明素材的黑白蒙版（仅透明视频/PNG 生成） |
| `--ImageReader.existing_camera_id` | id | 补拍：复用已有相机（`colmap.rs:556`） |
| `--image_list_path` | 文件 | 仅对清单内图像提取（自动优化子集 / 补拍新增帧） |
| `--FeatureExtraction.max_image_size` 或 `--SiftExtraction.max_image_size` | px | 长边上限（按 CLI 家族选名，`colmap.rs:510-526`） |
| `--SiftExtraction.max_num_features` | N | 特征数上限（`colmap.rs:527-532`） |
| `use_gpu` + `gpu_index` | 见 1.1 | GPU/CPU 切换 |

- 算法为 **SIFT**（COLMAP 默认；未启用 ONNX 的 ALIKED/LoMa）。
- `extract_features_quality_v2` 会带 `max_image_size` 与 `max_features`
  （`colmap.rs:298-332`），由自动优化按档位决定。

### 2.2 特征匹配

**顺序匹配（视频）** `match_sequential`（`colmap.rs:372-404`、`573-604`）：

| 参数 | 值 |
| --- | --- |
| `--SequentialMatching.overlap` | `10` |
| `--SequentialMatching.loop_detection` | `1`（自动优化开启时）/ `0` |
| `--SequentialMatching.loop_detection_period` | `10` |
| `--SequentialMatching.loop_detection_num_images` | `50` |
| `--SequentialMatching.vocab_tree_path` | 内置离线词袋树路径 |

**穷举匹配（图片）** `match_exhaustive`（`colmap.rs:406-431`）：仅 `--database_path`
+ GPU 参数。

**指定配对匹配（Bridge 补帧）** `match_pairs`（`colmap.rs:433-467`）：
`matches_importer --match_list_path <pair_list> --match_type pairs`。

### 2.3 稀疏重建 `mapper`

参数构造：`mapper_args`（`colmap.rs:706-733`）、`incremental_mapper_args`（`colmap.rs:795-823`）。

**普通重建 `map`**：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `--database_path` / `--image_path` / `--output_path` | 路径 | 标准输入输出 |
| `--Mapper.tri_ignore_two_view_tracks` | `0` | 条件性放开两视图轨迹（高画质/透明场景） |

**从已有模型续建 `map_from_existing`**：
`--input_path <model>` + `--Mapper.multiple_models 0`（`colmap.rs:720-725`），
基线模型不覆盖，输出到独立目录便于回滚。

**补拍增量 `map_incremental`**（`colmap.rs:795-823`）：

| 参数 | 值 |
| --- | --- |
| `--input_path` | 已有稀疏模型 |
| `--Mapper.image_list_path` | 新增帧清单 |
| `--Mapper.fix_existing_frames` | `1` |
| `--Mapper.ba_refine_focal_length` | `0` |
| `--Mapper.ba_refine_principal_point` | `0` |
| `--Mapper.ba_refine_extra_params` | `0` |

**BA 后端选择（Caspar）** `with_ba` + `colmap_ba::append_mapper_options`
（`colmap_ba.rs:300-323`）：当 Caspar 探测通过时追加

```
--Mapper.ba_local_backend CERES --Mapper.ba_global_backend CASPAR
--Mapper.ba_gpu_index <N> --log_level 2
```

即 **局部 Ceres + 全局 Caspar**。失败时回退并显式设 `ba_local/global_backend CERES`
（`colmap.rs:160-231`、`colmap_ba.rs:316-323`）。

### 2.4 光束法平差探测 `bundle_adjuster`

`colmap_ba::probe`（`colmap_ba.rs:175-246`）在临时目录用内置 **3 相机 / 8 点**
fixtures 真实执行一次：

```
bundle_adjuster --input_path <fixture> --output_path <tmp>
  --BundleAdjustment.backend CASPAR --BundleAdjustmentCaspar.gpu_index <N> --log_level 2
```

通过条件：进程成功 + 产出 `points3D.bin` + 日志含 `Creating Caspar bundle adjuster`。
失败则永久回退 Ceres（进程内熔断，`colmap_ba.rs:113-119`）。

## 3. 读取的 COLMAP 产物与数据结构

| 产物 | OOOSplat 读取方式 | 用途 |
| --- | --- | --- |
| `database.db`（SQLite） | rusqlite 只读，统计 `keypoints` 与 `two_view_geometries` 表的 `rows` 之和 | Caspar 阈值判断（`colmap_ba.rs:121-138`，阈值 keypoints≥2,000,000 且 verified_matches≥2,000,000） |
| `sparse/*/cameras.bin` | 自行解析二进制（`reconstruction/colmap_model.rs`） | 相机模型/尺寸（枚举含 SIMPLE_PINHOLE/PINHOLE/SIMPLE_RADIAL/RADIAL/OPENCV/OPENCV_FISHEYE 等） |
| `sparse/*/images.bin` | 自行解析二进制 | 注册图像计数、相机归属 |
| `sparse/*/points3D.bin` | 读取计数 | 三维点数、注册率校验（`reconstruction/validator.rs`） |
| 词袋树 `vocab_tree_faiss_flickr100K_words256K.bin` | 随引擎分发，SHA-256 + 字节数校验（`colmap/vocabulary.rs`） | 顺序匹配闭环检测 |
| 多模型目录 | `best_sparse_model_with_input_count` 选择最佳 | 选择稀疏模型 |

进度与日志：解析 COLMAP 的输出行（matching/mapper/feature extractor observer）。

## 4. 未使用的 COLMAP 功能

| 功能 | 为什么不用 |
| --- | --- |
| 稠密重建 `image_undistorter` / `patch_match_stereo` / `stereo_fusion` | 下游是 Brush 训练高斯泼溅，不需要稠密 MVS/网格 |
| `model_converter` / `model_analyzer` / `model_splitter` | 直接读二进制模型，不需要格式转换/分析 |
| GUI | 项目自带 Tauri 前端 |
| ONNX 特征（ALIKED / LoMa） | 构建时禁用 `ONNX_ENABLED` |
| 独立词袋匹配 `vocab_tree_matcher` 作为主匹配 | 只用顺序匹配内的闭环检测 |
| `point_triangulator` | 未使用 |

## 5. 相关源文件索引

| 文件 | 职责 |
| --- | --- |
| `src-tauri/src/engines/colmap.rs` | 子命令封装、CLI 家族适配、参数构造、Ceres 回退 |
| `src-tauri/src/engines/colmap_ba.rs` | Caspar BA 能力探测、阈值判断、熔断、mapper 参数追加 |
| `src-tauri/src/engines/colmap/vocabulary.rs` | 离线词袋树校验与路径解析 |
| `src-tauri/src/engines/health.rs` | 引擎体检、CLI 家族探测、GPU 加速状态 |
| `src-tauri/src/reconstruction/colmap_model.rs` | 解析 `cameras.bin` / `images.bin` |
| `src-tauri/src/reconstruction/validator.rs` | 注册率/三维点校验 |
| `src-tauri/src/reconstruction/ply.rs` | 发布前 PLY 头校验 |
| `src-tauri/src/planner/bridge_backfill.rs` | 补帧规划与配对清单（供 `matches_importer`） |
| `src-tauri/src/pipeline/runner.rs` | 各阶段调用编排 |
