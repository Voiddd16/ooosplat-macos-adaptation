# ooo-sfm：自研 SfM 后端替换 COLMAP —— 完整实施方案

> 分支：`feat/ooo-sfm`
> 基线：`main` @ v0.6.1（commit `c72207e`）
> 相关文档：[`colmap-usage.md`](./colmap-usage.md)、[`colmap-dependency-map.md`](./colmap-dependency-map.md)
> 参考源码：`engines/colmap/colmap/`（本地 COLMAP 源码检出，供实现对照；检出点 `c08f96ca`，与锁定的 4.2.1 fork 可能有细节差异，必要时 `git -C engines/colmap/colmap checkout 4.2.1`）

---

## 0. 目标与约束

**目标**：用项目自研的最小 SfM（**Rust、进程内、函数套函数**，代号 **ooo-sfm**）替换外部 COLMAP。
上层（`pipeline/runner.rs` 的调用、`engines/colmap.rs` / `colmap_ba.rs` 的**公开函数签名**）
保持不变，底层实现 fallback 到自研函数。

**硬约束**

| # | 约束 | 说明 |
|---|---|---|
| C1 | **编译期选择** | 不引入第二个二进制、不引入 CLI 兼容层、不做 sidecar |
| C2 | **不侵入主分支** | 主分支的 COLMAP 代码原样保留；本分支改动尽量集中在**新文件**，rebase 友好 |
| C3 | **行为一致** | 输入、输出、进度、日志、取消、错误语义与原实现一致（仅内部实现不同） |
| C4 | **平台化 fallback** | 本平台构建自动启用 ooo-sfm；**架构预留多平台后端**（macOS 现在，Android/iOS 将来） |
| C5 | **输出红线** | `sparse/*.bin` 等产物格式**逐字节**一致，否则 Brush / 校验全崩 |

---

## 1. 总体策略

### 1.1 接缝层级：**领域函数层**（不是最底层 spawn）

把所有与 COLMAP 的耦合看成一个"可替换后端"，接缝定在：

- `engines/colmap.rs` 的公开函数（`extract_features` / `match_*` / `map*` / …）
- `engines/colmap_ba.rs` 的公开函数（`select` / `select_for_mapper` / `read_database_stats` …）

**不要**把接缝下沉到"最底层的 `run_colmap`"。原因：`health.rs` 直接 spawn `-h` 嗅探身份、
`colmap_ba` 直接实跑 `bundle_adjuster`、`runner.rs` 直接读 `database.db` —— 这些都在
"最低那层 spawn"之外，单纯替换最底层覆盖不到（详见第 5 节）。

### 1.2 命名与多平台结构

命名分两层，**与平台解耦**，便于将来扩展 Android/iOS：

- **角色层**（feature / 分支 / 顶层领域模块）：`backend-ooo-sfm`（契合仓库第一方命名，如 `OOOBrush`）
- **平台层**（后端子模块）：`apple`（macOS 现在，将来可含 iOS/iPadOS）/ `android` / …

```
src-tauri/src/engines/
  mod.rs                    # 只改这里（加性）
  colmap.rs                 # 主分支原样不动
  colmap_ba.rs              # 主分支原样不动
  colmap/vocabulary.rs      # 主分支原样不动
  ooo_sfm/                  # 新增：ooo-sfm 领域层（平台无关）
    mod.rs                  # 领域 API（= 原 colmap.rs 公开签名）+ 平台后端选择
    ba.rs                   # 领域 API（= 原 colmap_ba.rs 公开签名）
    apple/                  # Apple 平台后端
      mod.rs
      runtime.rs            # 伪进程封装（observer / 日志 / 取消）
      format/               # 稀疏模型 bin + database.db 读写
      features.rs
      matching.rs
      mapper.rs
      ba.rs
    android/                # 未来：只新增，不动门面
```

后端由 `ooo_sfm/mod.rs` 内部按 `cfg(target_os)` 选择：macOS 走 `ooo_sfm::apple`。
**新增平台只加子模块，门面与构建命令不变。**

### 1.3 切换机制：`#[cfg(feature)] + #[path]` 门面

`engines/mod.rs` 加性改动：

```rust
#[cfg(not(feature = "backend-ooo-sfm"))]
#[path = "colmap.rs"]
pub mod colmap;

#[cfg(not(feature = "backend-ooo-sfm"))]
pub mod colmap_ba;

#[cfg(feature = "backend-ooo-sfm")]
#[path = "ooo_sfm/mod.rs"]
pub mod colmap;          // 对外暴露与 colmap.rs 同名同签名的 API

#[cfg(feature = "backend-ooo-sfm")]
#[path = "ooo_sfm/ba.rs"]
pub mod colmap_ba;       // 对外暴露与 colmap_ba.rs 同名同签名的 API
```

`Cargo.toml`：

```toml
[features]
ooo-sfm = []
```

构建：

```bash
# ooo-sfm 版（macOS 默认启用）
tauri build --features backend-ooo-sfm
# 主分支等价版（对照 / 回归）
cargo build                 # 不带 feature
```

> **效果**：`runner.rs` 的 `colmap::*` / `colmap_ba::*` 调用点**一行不改**；
> 主分支文件全量保留。本分支 diff ≈ 新目录 + `engines/mod.rs` 几行 cfg + `health.rs` 一小段。

### 1.4 平台自动启用

推荐用 **feature**（可在任意平台回归测试），在本平台构建配置里默认打开：
例如在 `scripts/local-tauri.mjs` / 打包脚本 / `tauri` 构建参数中对 `target_os == macos` 追加
`--features backend-ooo-sfm`。**不要**硬绑 `#[cfg(target_os)]` 作为开关，否则无法在 Win/Linux 上验证自研实现
（平台差异放在 `ooo_sfm/<platform>/` 内部即可）。

---

## 2. 输出契约（红线，必须逐字节一致）

| 产物 | 格式 | 消费者 |
|---|---|---|
| `sparse/<n>/cameras.bin` | `u64` 数量；每台：`u32 id` + `i32 model_id` + `u64 w` + `u64 h` + `f64[params]` | `reconstruction/colmap_model.rs` |
| `sparse/<n>/images.bin` | `u64` 数量；每张：`u32 id` + `7×f64`(qvec 4 + tvec 3) + `u32 camera_id` + `name\0` + `u64 npts` + `n×24B` | `reconstruction/colmap_model.rs`、`validator.rs` |
| `sparse/<n>/points3D.bin` | `u64` 数量（本项目只读计数） | `validator.rs` |
| `rigs.bin` / `frames.bin` | **成对**出现或都不出现 | `reusable_model_files` |
| 相机模型枚举 | 0 SIMPLE_PINHOLE(3) … 11 RAD_TAN_THIN_PRISM_FISHEYE(16) | `colmap_model.rs` |
| `database.db` | SQLite，含 `cameras(camera_id,model,width,height,params)`、`keypoints(rows)`、`two_view_geometries(rows)` | `colmap_ba::read_database_stats`、`runner.rs::validate_database_camera` |
| `dataset/{images/, sparse/0/{cameras,images,points3D}.bin}` | 给 Brush 的数据集布局 | `runner.rs::prepare_brush_dataset` → Brush |
| `sparse/<n>/` 多模型目录 | 供 `best_sparse_model*` 选最佳 | `runner.rs` |

> **唯一不能"内部自由"的部分。** 其余（CLI 参数、中间结构、算法）都可自选。

---

## 3. 必须复刻的 `ProcessManager.run()` 行为

现在的底层 `run_colmap` 调用 `manager.run(ProcessSpec{..})`（`process/mod.rs:282`）。
它**免费**提供了 4 件事，绕过它必须自己补上：

| 副作用 | 依赖方 | 自研侧要求 |
|---|---|---|
| observer 事件流（`ProcessUpdate`） | 进度 / 运行时统计 | 主动发 `Started` / `StageProgress` / `Heartbeat` / `Exited` |
| 日志写入 `log_path` | 日志查看器 / 诊断 | 写同样的 `work/logs/*.log` |
| 取消（`CancellationToken`） | `manager.cancel()` | 计算循环内查 `manager.child_token().is_cancelled()`（`process/mod.rs:278`） |
| 错误语义 | `SplatError::Process` / 前端错误摘要 | 失败时返回等价的 `SplatError`（含 `failure_detail` 风格尾部） |

### 3.1 伪进程封装（`ooo_sfm/apple/runtime.rs`）

把"进程内计算"包装成"看起来像进程"的适配层：

```
run_job(
    stage, engine, observer: Option<ProcessObserver>, log_path: PathBuf,
    cancel: CancellationToken,
    job: impl FnOnce(&ProgressSink, &CancellationToken) -> Result<()> + Send,
) -> Result<()>
```

职责：

1. 发 `ProcessUpdate::Started`（合成 id，见 §4.3）；
2. 在 `std::thread` / `spawn_blocking` 中执行 `job`；
3. `job` 通过 `ProgressSink` 汇报进度（→ `StageProgress`）；
4. 把日志写入 `log_path`；
5. 循环/分块点检查 `cancel`；
6. 结束发 `Exited`，错误映射为 `SplatError::Process`。

---

## 4. 进度：结构化事件（重写"算进度的函数"）

### 4.1 现状

`pipeline/runner.rs:3337 process_observer(stage, engine, expected_total, mode) -> ProcessObserver`，
其中 `ProcessObserver = Arc<dyn Fn(ProcessUpdate) + Send + Sync>`（`process/mod.rs:220`）。
它按 `ObserverMode` 从**文本行**解析进度：

- `Ffmpeg` → `parse_ffmpeg_frame`
- `BracketProgress` / `Matching` → `parse_bracket_progress`
- `Mapper` → `parse_mapper_progress`
- `Brush` → `RuntimeTracker::line`

### 4.2 方案：新增结构化变体，不伪造文本

`process/mod.rs` 加一个**加性**变体：

```rust
pub enum ProcessUpdate {
    // ... 现有变体不变 ...
    /// 进程内后端直接上报的结构化进度（不经过文本解析）
    StageProgress {
        phase: &'static str,
        current: u64,
        total: Option<u64>,
        message: String,
    },
}
```

自研函数直接发 `StageProgress`；`process_observer` 增加一个分支：直接算
`progress = current/total` 并 `events.send(EventKind::Progress, ...)`，同时更新 `RuntimeTracker`。
**边界签名 `Option<ProcessObserver>` 不变 → `runner.rs` 调用点不变。**

### 4.3 连带必须处理（否则 UI 静默失效）

| `ProcessUpdate` | 子进程来源 | 进程内替代 |
|---|---|---|
| `Started { process_id }` | 真实 PID | 合成 id（如 `0` / stage id），供 `RuntimeTracker` |
| `Heartbeat { elapsed_ms }` | `ProcessManager` 定时器 | 自研定时发（长任务防"卡死"观感） |
| `Resources(sample)` | 按 PID 采 nvidia-smi | 跳过（`process/resources.rs` 对进程内无意义） |
| `Exited` | 进程退出 | 成功/失败时补发 |
| `RuntimeTracker::line` | 逐行推 phase/ETA | 用 `StageProgress.phase/message` 喂 |

**范围控制**：只改 SfM 相关模式（`Matching` / `Mapper`，视需要 `BracketProgress`）。
`Ffmpeg` / `Brush` 走原路径，**不要动**（`process_observer` 是共享的）。

### 4.4 取消 ≠ 进度

取消仍走 `ProcessManager::child_token()`（`process/mod.rs:278`），与进度回调是两条独立通道。

---

## 5. 接缝之外、必须单独改的点

| 位置 | 现在直接做什么 | 处理方式 |
|---|---|---|
| `engines/health.rs:480-560` | spawn 二进制跑 `-h`×6，读 stdout 判 `is_locked_build` / `detect_cli_family` / 必需参数 / `cpu_only` | 改为向自研实现查询 `capabilities()` / `identity()`；删掉文本嗅探 |
| `engines/health.rs:605` → `colmap_ba::select` | 实跑 `bundle_adjuster` 探测 Caspar | 自研 `ba::select` 返回等价 `BaSelection`（macOS 上 Caspar 恒不可用） |
| `pipeline/runner.rs:4383 validate_database_camera` | 读 `database.db` 的 `cameras` 表 | 自研产出兼容 `database.db`（D3），此函数不改 |
| `engines/colmap_ba.rs:121 read_database_stats` | 读 `keypoints` / `two_view_geometries` 行数 | 读自研产出的兼容 DB（D3） |
| `process/resources.rs` | 对子进程采 nvidia-smi | 进程内无对象；macOS 本就是 `unsupported` |
| `project/metadata.rs` `COLMAP_INPUT_VERSION` | 检查点指纹 | 纳入自研后端版本，参与失效判断 |
| 前端 `src/types/pipeline.ts`、`App.tsx`、`TaskConfiguration.tsx`、`i18n` | 展示 `ColmapBackend` / `ColmapAccelerationStatus` / `colmapCliFamily` | **不改**（D2）：自研侧填充等价值 |

---

## 6. 需要实现的功能与状态字段

### 6.1 `engines/colmap.rs` 公开 API（必须同名同签名）

```rust
pub const LOCKED_VERSION: &str;   // 自研可返回自己的版本串
pub const LOCKED_COMMIT: &str;
pub fn is_locked_build(help: &str) -> bool;                       // 语义替换为自研 identity 判断
pub enum ColmapCliFamily { Legacy39, Modern4 }                    // 可保留（或被 capability 取代）
pub fn detect_cli_family(feature_help: &str, matching_help: &str) -> Option<ColmapCliFamily>;
pub fn require_verified_cli(executable: &Path) -> Result<()>;     // 可退化为 no-op / 检查自研可用性

pub async fn extract_features(..) -> Result<()>;
pub async fn extract_features_quality_v2(..) -> Result<()>;
pub async fn extract_incremental_features(..) -> Result<()>;
pub async fn match_sequential(..) -> Result<()>;
pub async fn match_exhaustive(..) -> Result<()>;
pub async fn match_pairs(..) -> Result<()>;
pub async fn map(..) -> Result<()>;
pub async fn map_from_existing(..) -> Result<()>;
pub async fn map_incremental(..) -> Result<()>;
```

> 参数含 `executable: &Path`、`manager: &ProcessManager`、`observer: Option<ProcessObserver>`、
> `log: PathBuf`、`gpu_index: Option<u32>` 等。**签名必须逐字保留**（包括 `executable` 参数——
> 自研实现里可作为"引擎根"占位或忽略）。

### 6.2 `engines/colmap_ba.rs` 公开 API

```rust
pub struct BaSelection { .. }
pub struct DatabaseStats { .. }
pub async fn select(executable: &Path, compatible_gpu: Option<u32>) -> BaSelection;
pub async fn select_for_mapper(..) -> BaSelection;
pub fn meets_caspar_threshold(stats: DatabaseStats) -> bool;
pub async fn disable_after_runtime_failure(executable: &Path, index: u32, reason: &str);
pub fn read_database_stats(database: &Path) -> Result<DatabaseStats, String>;
pub fn append_mapper_options(args: &mut Vec<OsString>, selection: &BaSelection);
pub fn append_explicit_ceres_options(args: &mut Vec<OsString>);
```

BA 在本 fork 内按平台重写（D4）：macOS = CPU BA，`select` / `select_for_mapper` 返回自研 BA 选择，
`map*` 内部调用自研 BA。`read_database_stats` 读自研产出的兼容 `database.db`（D3）。
`meets_caspar_threshold` / `append_mapper_options` 等按新语义对齐（Caspar 分支在 macOS 上恒不可用）。

---

## 7. 分阶段实施计划

> 每个里程碑结束都应保证 `cargo build --features backend-ooo-sfm` 通过 + 对应测试通过。

### M0 — 骨架与门面 ✅
- [x] 建 `engines/ooo_sfm/`（领域层 + `ooo_sfm/apple/` 平台后端）模块骨架，同名函数先 `todo!()`。
- [x] `engines/mod.rs` 加 `#[cfg] + #[path]` 门面。
- [x] `Cargo.toml` 加 `backend-ooo-sfm` feature。
- [x] 验证：开/关 feature 均可编译。
  - stub crate 类型检查（同一门面加载真实源码 + `runner.rs` / `health.rs` 调用形态）→ 通过。
  - 整工程 `cargo check --lib`：不带 feature ✅ / `--features backend-ooo-sfm` ✅（rustc 1.99.1）。
  - 备注：`cargo check` 要求 `engines/macos/arm64/*` 资源路径存在；本地以 `.gitignore` 覆盖范围内的临时占位绕过（正式构建需 `npm run setup:engines:macos`）。
    需要存在的路径：`engines/macos/arm64/{bin,lib,licenses,colmap}`（目录）、
    `engines/macos/arm64/{SHA256SUMS,BUILD-INFO.json,BUNDLED-COMPONENTS.json}`（空文件即可）、
    `.cache/html-viewer/runtime.js`。验证完请只删这几项 —— **`engines/macos/arm64/README.md` 是已跟踪文件，不能删**。

### M1 — 伪进程运行时（`ooo_sfm/apple/runtime.rs`）✅
- [x] `run_job`：`spawn_blocking` + `catch_unwind` + 日志 + 取消 + 错误映射，逐条对齐
      `ProcessManager::run` 的对外可观察行为（`Started` / `Heartbeat` / 日志头尾 / 失败文案 / 取消语义）。
- [x] `ProgressSink` → `ProcessUpdate::StageProgress`（加性变体；子进程后端仍走 `Line` 解析）。
- [x] `ProcessManager::lifecycle_observer()`：进程内复用 `LogOpened` / `Exited`
      （否则日志查看器与 `RunnerUpdate` 的引擎归类静默失效）。
- [x] 九个阶段全部接到 `run_job`：算法本体由 M3–M6 替换闭包体；在那之前以
      `SplatError::UnsupportedEngine` 明确失败（不 panic），且失败原因照常进日志。
- [x] 测试：日志头/尾与追加偏移、结构化进度事件、协作式取消、失败与 panic 映射（4 个全绿）。
- 刻意差异：取消是**协作式**的（进程内无法 kill 线程）→ 所有计算循环必须在分块点调用
  `ProgressSink::check_cancelled`；不产生 `Resources`（macOS 本就不按 PID 采 GPU 指标）。
- 踩坑记录：`&Box<dyn Any + Send>` 会走 unsize 而非 deref（`Box<…>` 自身即 `Any + Send`），
  导致 `downcast_ref` 恒失败 → 必须 `payload.as_ref()`。

### M2 — 数据格式读写（`ooo_sfm/apple/format/`）✅
- [x] `cameras.bin` / `images.bin` / `points3D.bin` 写入器 + 读取器（严格对齐 §2，条目按 id 升序）。
- [x] `database.db`（SQLite）写入器（cameras / images / keypoints / descriptors / matches /
      two_view_geometries）+ 读取器（供 M4 匹配、M5 mapper、M6 stats 使用）；`begin_bulk` / `end_bulk` 批量事务。
- [x] 往返测试：写 → 用项目**现有** reader（`reconstruction/colmap_model.rs`）与
      `colmap_ba::read_database_stats` 的两条 SQL 语义读回一致。
- 关键事实（**必须按 `4.2.1` 标签核对**；工作区 checkout 的 main 已把 `cameras` 改成
  `PRIMARY KEY(camera_id, source)`，与项目锁定的版本不兼容）：
  - `PRAGMA user_version = 4020100`（`MakeDatabaseVersionNumber(4,2,1,0)`；`util/version.cc.in`）。
  - `keypoints.data` = `rows × 6` 的 `f32`（x, y, a11, a12, a21, a22）。
  - `descriptors.data` = `rows × 128` 的 `u8`，`type = 0` 即 SIFT（`FeatureExtractorType` 从 `-1` 起编号）。
  - `matches` / `two_view_geometries.data` = `rows × 2` 的 `u32`（特征点下标对）；
    `rows` 是"匹配条数"，因此 `SUM(rows)` 正好是 `read_database_stats` 要的总匹配数。
  - `points3D.bin` 的每个观测是 `u32 image_id + u32 point2D_idx`（**8 字节**，不是 24）。
  - `pair_id = 2147483647 × min(id) + max(id)`；`images.bin` 里四元数顺序 `w,x,y,z`；
    `two_view_geometries` 的 F/E/H 是**行主序** 9×`f64`（COLMAP 侧 `transpose()` 后按列主序写）。
  - 不写 `rigs.bin` / `frames.bin`：产出 3 文件旧式模型，`reusable_model_files` 视为合法。
- 机械化校验（一次性，结论记录在此）：从 `git show 4.2.1:src/colmap/scene/database_sqlite.cc`
  抽出各表的列名与索引名，与本模块 schema 建库后的 `PRAGMA table_info` / `sqlite_master` 对比
  → **11 张表的列名与全部索引名完全一致**。
- 测试：8 个（字节布局断言、现有 reader 交叉验证、schema 与 `user_version`、blob 往返、
  `pair_id` 公式、异常输入拦截）。

### M3 — 特征提取（`feature_extractor` 等价）✅
- [x] `extract_features` / `extract_features_quality_v2` / `extract_incremental_features` 三个入口接到
      自研实现（`apple/features.rs`），`colmap.rs` / `runner.rs` 调用点一行未改。
- [x] 相机模型 `SIMPLE_RADIAL` + `single_camera`；`mask_path`；`image_list_path`；
      `existing_camera_id`；`max_image_size` / `max_num_features`；`gpu_index`（记录用，CPU 实现忽略）。
- [x] 产出 `database.db` 的 `cameras` / `images` / `keypoints` / `descriptors` / `rigs` /
      `rig_sensors` / `frames` / `frame_data`（4.2.1 的特征提取本身就会写 rig/frame，不是只在
      mapper 阶段）。
- [x] 自研 SIFT（`apple/sift.rs`，只依赖 std）：`vl_sift_new` 同构的金字塔
      （`num_octaves=4`、`octave_resolution=3`、`first_octave=-1` 先 2× 上采样、输入 σ=0.5、
      σ_min=1.6）；DoG 滚动窗口（内存里只保留 3 张高斯 + 3 张 DoG）；
      泰勒精化（迭代语义用 Lowe `(v + 0.5)` 截断）+ 峰值阈值 `0.02/3` + 边缘抑制 `edge_thresh=10`；
      36 bin 方向直方图（高斯加权、平滑、峰值 ≥ 0.8×主峰 且 ≥ 峰值阈值 → 最多 2 个方向）；
      4×4×8 三线性描述子（角度取负 = UBC 组内镜像）；L1-root 归一化 → ×512 → 四舍五入 → `min(255)`。
- 行为对齐（全部核对自 `git show 4.2.1:...`）：
  - 相机：首张成功图像优先采用 **EXIF 焦点长度**（`apple/exif.rs`，见下一条），
    没有 EXIF 时 `f = 1.2 × max(w, h)`；`cx = w/2`、`cy = h/2`、`k = 0`；
    `prior_focal_length` = EXIF 是否命中；`camera_id` 由数据库自增。
  - EXIF 焦点长度（`Bitmap::ExifFocalLength()` 等价，支持 JPEG APP1 / TIFF，LE+BE）：
    1. `FocalLengthIn35mmFilm` → `f35 / 43.27 × √(w² + h²)`（CIPA 公式，35 mm 像幅对角线 43.27）；
    2. 退 `FocalLength` × `FocalPlaneXResolution` 换算的像素/毫米
       （`FocalPlaneResolutionUnit`：2=英寸 /25.4、3=厘米 /10、4=毫米、5=微米 ×1000）；
    3. 4.2.1 的第三级「Make/Model → 传感器宽度库」**未内置**（刻意差异）：到这里返回
       `None`，由默认系数建相机、`prior_focal_length = 0`。
    PNG 的 `eXIf` chunk 不支持（罕见，记录在案）；解析器只扫文件头 256 KiB，任何畸形结构
    返回 `None` 不 panic（有一个"截断前缀永不 panic"的前缀模糊测试）。
  - 图像按名字排序后处理（决定 `image_id` 顺序）；`image_list_path` 内容先排序。
  - 已有**键点与描述子齐全**的图跳过（`IMAGE_EXISTS`）；半成品（只有键点）先 `delete_features` 重提。
  - 读图失败 / 掩码缺失 / 尺寸与单相机不一致 → warning 后跳过该图，整体仍成功
    （`BITMAP_ERROR` / `MASK_ERROR` / `CAMERA_SINGLE_DIM_ERROR`，退出码 0 语义）。
  - 掩码命名 `<mask_path>/<name>.png`，缺失时退化到 `<mask_path>/<stem>.png`；
    按**整数坐标** `(int)x, (int)y` 查掩码（0 或越界则删点），即 COLMAP 的 `MaskFeatures` 后置过滤。
  - 灰度化：丢 alpha、Rec.709 权重（`.2126/.7152/.0722`）+ 0.5 四舍五入（`CloneAsGrey`）。
  - `max_num_features` 保留**粗尺度**特征：按 (八度, DoG 层) 分组，从最粗组往回累计，
    跨过上限的那组**整体保留**（COLMAP `first_level_to_keep` 语义，因此结果可能略超上限）。
  - 描述子方向 bin 用 UBC/Lowe 顺序（等价 COLMAP 写库前的
    `TransformVLFeatToUBCFeatureDescriptors`；组内镜像 = 计算时对角度取负），
    跨平台复用同一张图产出的 `descriptors.data` 语义一致。
  - 键点写出 `x + 0.5` / `y + 0.5`、`shape = [s·cosθ, -s·sinθ, s·sinθ, s·cosθ]`（含 `max_image_size`
    缩放：x/y 按 sx/sy 缩放，a11/a21 用 sx、a12/a22 用 sy，与 `ScaleKeypoints` 一致）。
- 刻意差异（`sift.rs` 模块注释里同步记录）：
  - 插值与边界是自研实现（COLMAP 用 VLFeat），特征点位置**不会逐位相同**；
    验收靠 M11 的端到端重建质量与跨平台 `descriptors.data` 语义，而不是逐位比对。
  - `Bitmap::Thumbnail` 缩放到 `max_image_size` 的抗锯齿用"先等效低通再双线性"近似
    OIIO triangle 滤波，与 COLMAP 细节不同（缩小时更接近理想低通）。
  - 内存护栏：上采样后的金字塔超过 6000 万像素时跳过首八度上采样（原因记入日志 note）。
- 并行：按图用 `rayon`（新依赖，挂 `backend-ooo-sfm` feature）并行提取，SQLite 保持单写者
  （与 COLMAP 的 writer 线程同构）+ 每图一个事务（沿用 COLMAP 的断点续跑粒度）。
  分块大小按 `clamp(2GB / (6 张平面 × (2w)×(2h) 的 f32), 1, 核数)` 估算。
- 进度与日志：`sink.stage("featureExtraction", i, N, ...)`（前端未知 phase 回退到 `runtime.running`，
  M7 补文案）；每图日志 `Processed file [i/N]` + `Name / Dimensions / Camera / Focal Length /
  Features / Mask` 与 COLMAP 同形。
- 测试：`engines::colmap::apple` 下 30 个全绿（SIFT 合成图检出/纹理角点/确定性/L1-root 归一化/
  粗尺度裁剪/upright/八度降采样；端到端提取落库、幂等、增量复用相机、`max_image_size` 坐标回缩放、
  掩码剔除、缺掩码跳过、坏图跳过、尺寸不一致跳过、`image_list` 排序、取消、缺相机报错）；
  EXIF 解析 7 个 + 相机 `prior_focal_length` 集成 1 个（`camera_uses_exif_focal_length_and_prior_flag`）。
- 性能冒烟（release，单独编译 `sift.rs`）：类自然图 640×480 ≈ 0.5s、1600×900 ≈ 2.1s / 张；
  尺度分布（细八度多、粗八度少）与 COLMAP 一致。
- 踩坑：`max_num_features` 结果"超过上限"是 COLMAP 语义的一部分（单组超限整组保留），不是 bug；
  棋盘格是 SIFT 的**病态**测试图（细尺度全是边缘 → 无孤立极值；粗尺度每个格心都是 blob），
  端到端测试图改用"类自然图"（低频随机场 + 硬边 + 少量高频）。

### M4 — 特征匹配
- [x] `match_sequential`（定义**闭环检测接口**，第一步 no-op；不接入词袋树，D5）。
- [x] `match_exhaustive`。
- [x] `match_pairs`（Bridge 补帧的 `--match_type pairs` 等价）。
- [x] 产出 `two_view_geometries`。

落点：`apple/matching.rs`（配对 / 描述子匹配 / 编排 / 落库）+ `apple/two_view.rs`
（LO-RANSAC 与几何验证核心）+ `apple/format/database.rs`（`matches` / `two_view_geometries`
读写与半成品清理）。

- 配对与描述子匹配（核对自 4.2.1 `feature_matching.cc` / `matcher.cc`）：
  - 顺序：按名字排序，`overlap = 10` 的二次偏移窗口（`(i+1)²−i²`）+ quadratic 采样；
    闭环检测接口已就位（本步 no-op，D5）。
  - 穷举：全对（分块生成避免一次性物化 O(N²) 列表）。
  - `pairs`：`imported_pair_list` 语义（跳过注释与未知名称）。
  - 2-NN + 比值检验（`sqrt(best) ≥ 0.8 × sqrt(second)`）、`max_l2_dist = 512² × 0.7²`
    （描述子 u8→float，尺度与 4.2.1 faiss `ToFloat()` 一致）、交叉验证（默认开）。
- 几何验证（`two_view.rs`，全部图对统一走 F/H 两路验证）：
  - F：7 点最小解 + LO-RANSAC（`num_inliers > 7 && ≥ 8` 进入局部优化，最多 10 轮）；
    局部估计器默认 = Sampson 精化（LM 25 步，失败可回退 8 点）。`min_num_inliers = 15`。
  - H：4 点最小解 + 同一套 LO-RANSAC；先验内点比下限 `max(ransac.min_inlier_ratio,
    max_H_inlier_ratio × F内点 / 匹配数)` 只用于收缩迭代预算。
  - 判定：`F_ok = 成功 && 内点 ≥ 15`、`H_ok` 同理；两者皆否 → DEGENERATE；
    `H内点 / F内点 > 0.8` → PLANAR_OR_PANORAMIC（取内点更多的掩码），否则 UNCALIBRATED；
    水印检测（平移变换 + 边界比例 0.7 / 0.1）命中 → WATERMARK。
  - 选项默认值逐项对齐 4.2.1 `TwoViewGeometryOptions` / `RANSACOptions`：
    `max_error = 4.0`、`confidence = 0.999`、`dyn_num_trials_multiplier = 3.0`、
    `min/max_num_trials = 100/10000`、`min_inlier_ratio = 0.0`（未传覆盖项）、
    `max_H_inlier_ratio = 0.8`、`min_num_inliers = 15`、水印 `0.7 / 0.1 / true / 4.0`。
  - 落库：`image_id1 > image_id2` 时先把几何 `Invert()` 再写；匹配 < 15 写空 `matches`
    blob、不写几何行；描述子缺失 → 只写空 `matches` 行（无几何行）；重跑先删旧行；
    只有匹配存在时才复用匹配并补验证。F/H 以**主行序** 9×f64 写入（`camera1/2` 留 NULL，
    焦距估计在 M5 补）。
- 刻意差异（实现时记录在案）：
  1. 描述子匹配用**精确暴力 2-NN**（4.2.1 CPU 路径用 faiss 近似聚类索引；近似索引本身
     可能漏掉真近邻）——刻意取精确；同时不做 4.2.1 CPU 路径没有的 `max_num_matches` 截断。
  2. RANSAC 随机种子由 `pair_id` 派生（同一输入结果确定）；4.2.1 默认 `random_seed = -1`
     每次运行不同（RANSAC 本就无法逐位复刻随机序列）。
  3. 只写**成功**的 F/H；4.2.1 在 F 估计失败时仍会把未初始化的 F 写进几何表，不复刻该缺陷。
  4. H 多点路径：Hartley 归一化 DLT + `AᵀA` 最小特征向量 + 反归一化（`H = T2⁻¹·H_norm·T1`，
     最终按单位范数存储）；4.2.1 直接用原始像素坐标 SVD（条件数差）。4 点路径与 4.2.1 相同
     （原始坐标 8×8 线性解，h₈ = 1）。
  5. Sampson 精化用 9 参数 LM + 每步秩 2 投影（4.2.1 用四元数 + σ 的因子化流形），
     目标函数与迭代步数（25）相同；坐标换算 `F_norm = (T2ᵀ)⁻¹·F·T1⁻¹`、回映射
     `F = T2ᵀ·F_norm·T1` 与 4.2.1 逐符号对齐。
  6. **数值加固**：nalgebra 的 `SVD::new` / `SymmetricEigen::new` 传 `max_niter = 0`
     表示**不限迭代**，病态/含 NaN 输入会永久循环（踩过一次，见下）；所有动态分解改用
     `try_new(..., 上限)`，超限即放弃该次估计。7 点零空间用「薄 SVD 行空间基 + 补投影
     `I − BᵀB` 主特征向量」（nalgebra 的 `v_t` 是薄的，宽矩阵没有最后几行右奇异向量）。
  7. 取消：每块匹配前 `sink.check_cancelled()`（协作式）。
- 踩坑（写给未来的自己）：
  - `nalgebra::Vector2::to_homogeneous()` 追加的是 **0**（方向向量语义），不是 1！点的齐次
    坐标必须显式 `.push(1.0)`。踩过一次：归一化点变成 `(inf, inf)` → A 矩阵全 NaN →
    nalgebra SVD 永不收敛（表观「挂死」60s+）。
  - `Matrix3::new(...)` 参数是**行主序**，但 `{:?}`（Debug）**按列主序打印**——调试矩阵时
    不要被显示顺序骗了（用 `m[(r, c)]`）。
  - DLT 解向量映射：F 用 `from_column_slice`、H 用 `from_row_slice`（等价 COLMAP 的
    `Map(...).transpose()`）；两者都用「行向量与解向量点积 = 约束」的项匹配 + 数值探针
    （残差 ~1e-14 vs 1e14）双重确认。
  - 测试场景：平面场景的平面**不能过相机光心**、也不能只绕单轴转（投影仍是一条直线 →
    4 点 DLT 共线拒绝）；棋盘格是 SIFT 病态输入（同 M3）。
  - 8×9 薄 SVD 在 `n == 8` 时拿不到最后一行右奇异向量（越界 panic），由零空间助手兜底。
- 测试：`apple/matching.rs` 12 个（配对唯一性/分块无重复/名称排序/比值与距离拒绝/互近邻/
  e2e 落库与几何数据/幂等重跑/缺描述子/取消）+ `apple/two_view.rs` 8 个（7 点恢复已知 F、
  8 点精化代价下降、退化对、随机匹配不通过、平面 → H、4 点共线拒绝、已知 H 的最小/多点
  恢复、F 极线残差），全套 `cargo test --lib --features backend-ooo-sfm` 346+ 全绿。

### M5 — 稀疏重建（`mapper` 等价）——**端到端已通**（合成 8 图场景全注册）
- [x] M5a `apple/reconstruction.rs` + `apple/camera.rs`：`Reconstruction`（稠密下标 +
  `HashMap` 反查）、`CorrespondenceGraph`（只用**非退化几何对的内点**建图、排序去重、
  传递闭包、两视图观测判定）、`ObservationManager`（4.2.1 的全部增量统计 +
  可见性金字塔 6 级）、过滤（大重投影误差 → 小三角角，顺序有意义）。
- [x] M5c `apple/triangulator.rs`：2 视图 DLT（闭式）+ 多视图垂距 DLT（自伴特征分解）、
  cheirality / 最小三角角校验、≤15 观测穷举否则 RANSAC、`Create/Continue/Merge/
  Complete/Retriangulate/TriangulateImage/CompleteAndMergeTracks`。
- [x] 初始两视图相对位姿 `apple/relative_pose.rs`：F→E 的 4 解 cheirality 分解
  （`R1=UWVᵀ, R2=UWᵀV, t=U.col(2)`）、H 的 4 组分解、三角角中位数。
- [x] M5d/M5e `apple/mapper.rs` + `apple/mod.rs`：选项默认值逐项对齐 4.2.1、
  初始对选择与初始化、`FindNextImages`（可见性金字塔分数降序、两桶）、注册（P3P/LO-RANSAC
  + 焦距/畸变精化开关）、三角化、局部/全局 BA 调度（触发比例 1.1、频率 500/250000、
  局部 25 iters/SOFT_L1、全局 50 iters/TRIVIAL）、`Normalize`、写
  `sparse/0/{cameras,images,points3D}.bin`；三个入口（`map` / `map_from_existing` /
  `map_incremental`）按 CLI 语义接线（`tri_ignore_two_view_tracks` / `multiple_models=0` /
  `image_list_path` + `fix_existing_frames=1` + `ba_refine_*=0`）。
- 本轮修掉的真实 bug（都已加回归断言/测试）：
  1. **`normalize()` 位姿变换写错**（`reconstruction.rs`）：应为
     `t' = s·(R·centroid + t)`（4.2.1 `TransformCameraWorld` 的 Sim3d 合成），
     原来写成 `s·(t − centroid)`，**漏掉了 `R·centroid`** → 归一化后整个重建平移
     错位，症状是"初始 BA 后重投影误差 0.0001 px，但过滤把 42 个点全删光"。
  2. **`pose.rs` 的 `matrix.svd(..)` 不限迭代导致挂死**：nalgebra 的 `Matrix::svd()`
     等价 `SVD::new(..)`，`max_niter = 0` = **不限迭代**，退化矩阵会让双对角化
     永不收敛（与 M4 同一个坑，M5 又踩了一次）。改为 `try_new` + 上限 200。
  3. **DLT 绝对位姿的 2D 归一化除零**：`normalize` 对 2D 也按 3D 处理（第三分量
     `(1−1)·s = 0`），原代码 `u = uv[0] / uv[2]` → `inf`/`NaN` → A 矩阵含 NaN
     → 叠加上面第 2 点就是"LO-RANSAC 挂死"。改为 `u = uv[0]`（Hartley 归一化后的
     像点本身，w 隐含为 1）。
- [x] **P3P 换成 EPnP，端到端打通**（本轮）：
  1. **6 点线性 DLT 已删除**，改为直接移植 COLMAP 4.2.1 `EPNPEstimator`
     （`estimators/solvers/absolute_pose.cc`：`ChooseControlPoints` /
     `ComputeBarycentricCoordinates` / `ComputeM` / `ComputeL6x10` / `ComputeRho` /
     `FindBetasApprox1-3` / `RunGaussNewton` / `ComputeCcs` / `ComputePcs` /
     `SolveForSign` / `EstimateRT`）。`epnp_absolute_pose` 是 4 点以上可用的非最小解，
     同时充当 RANSAC 的假设生成器与 LO 局部估计器。
     换掉 DLT 的原因（实测根因）：DLT 的 12 个未知量里**平移列与世界点重心强耦合**，
     是零空间方向里最弱的轴；带 SIMPLE_RADIAL 畸变的观测无法被线性模型吸收，残差
     几乎全部压到这一列上 →「旋转恢复正确、平移差一个任意倍数」。EPnP 只用 4 个控制
     点（每点 2 行、每行只耦合 4 个未知量），平移最终由 **Kabsch 拟合**给出，结构上
     没有这个问题。同一场景的干净数据单测现在旋转误差 < 1e-6、平移 < 1e-4。
  2. **`mapper::tests::reconstructs_a_synthetic_ring_scene` 已解开并通过**：
     8/8 图注册、225 个 3D 点、平均重投影误差 < 1 px、相机中心共线且等距。
  3. 合成夹具本身有 bug（不是 mapper 的）：旧夹具是「点集中在 x ∈ [-1.5, 1.5]、
     相机每帧横移 2.0」，第 4 帧起相机转到场景侧面**看到 0 个点**，所以最多只能
     注册 3 张图。改为「沿 x 铺开的墙面点云（x ∈ [-0.5, 8.2]、y ∈ [-0.4, 0.4]、
     z ∈ [1.2, 1.8]）+ 横向步长 0.6 + 每帧 0.02 rad 偏航」，相邻共视 ≈ 100 点、
     三角角 18°–28°。顺带修了两个夹具断言：
     - 离群点必须**破坏观测**（随机 2D）。旧写法「随机挪 3D 点再用真值重投影」产生的
       观测与新 3D 点仍然几何自洽，本来就是合法内点，`离群点泄漏` 断言必然失败。
     - 相机中心不能再断言「y/z 分量为 0」：重建只确定到 7 自由度相似变换，初始对
       是 (5,6) 而不是 (0,1)，解出的坐标系相对真值有未知旋转。改成规范
       （gauge-free）的「共线 + 等距」检查（PCA 主轴 + 排序间距）。
  4. **CPU 加速（rayon）**：`lo_ransac_pose` 改成每轮 64 个假设一批，
     `par_iter` 并行求解 EPnP + 打分；畸变反投影（`cam_ray_from_image`）在循环外
     **只做一次**（原先每个假设都重算一遍）。随机采样仍顺序推进，固定种子 → 结果确定。
     BA 侧：`ba.rs::linearize` 改成按点 `par_iter`（每点产出独立小 Vec，再按原顺序
     拼接，输出与串行版**逐位相同**，不影响收敛与确定性）。Schur 装配那圈仍是串行的
     —— 共享块累加要开 per-thread 局部块再归并，收益/复杂度比等真实规模再评估。
- [ ] 仍未逐项对齐 `recon/m5-mapper.md`：
  1. 局部 BA 的图像集合（`FindLocalBundle` 的 8 组递松阈值 + 75 分位三角角）、
     `Gauge::THREE_POINTS` 下的常量集合、`FilterFrames`（bogus 相机筛选）
     目前是简化实现，未逐项对齐 `recon/m5-mapper.md` §8/§7。
- 刻意差异（记录在案）：
  1. **焦距不由 F 估计**：4.2.1 无先验时用 6 点共享焦距最小解；我们直接用相机当前
     焦距算 `E = K₂ᵀFK₁` 再分解（COLMAP ≤3.7 的做法），焦距交给 PnP 精化与 BA。
     原因（实测）：Sampson 残差对焦距的**统一缩放是不变量**，无法定焦距；而以
     cheirality 内点数为目标在**前视相机**（视频帧典型情形）下随焦距变化极平
     （实测 800 真值处与 89/414 处内点数几乎相同）。要可靠复刻只能移植 6 点联合解，
     收益不足以抵消复杂度，留待 M11 交叉验证时再评估。
  2. **不做 structure-less 注册回退**（4.2.1 默认开）；不做多模型与快照；
     不做初始化松弛（`init_min_num_inliers` 100→50→25 等）。
  3. 三角化的 RANSAC 假设来自"两两组合闭式解 + 全观测打分"，末尾额外补一次
     全观测多视图 DLT 候选（等价 4.2.1 的非最小估计器）。
  4. 随机数种子固定（由调用方传入），结果确定；4.2.1 默认 `random_seed = -1`。
  5. **RANSAC 假设生成器用 EPnP（6 点）而不是 P3P（3 点）**：4.2.1 的
     `LORANSAC<P3P, EPNP>` 用 P3P 出假设、EPnP 做局部估计；我们两端都用 EPnP。
     代价是假设「生成功率」低一些（6 点全内点概率 `p⁶` vs `p³`），实际靠 LO 局部
     估计兜住；收益是省掉 Grunert 四次方程这一块易错代码。后续若要严格对齐再补
     `solvers::p3p`。
  6. **Metal GPU 加速：已落地第一块内核，由 `backend-ooo-sfm-metal` feature 门控（默认关闭）**。
     - feature：`backend-ooo-sfm-metal = ["backend-ooo-sfm", "dep:metal"]`（`metal = 0.32`，仅 `target_os = "macos"`
       生效）。**关掉时行为与之前逐位一致** —— `ProjectionBatch` 本身存 `f64`，
       只有上传 GPU 时才降到 `f32`，所以 CPU 回退就是原来的标量循环。
     - 运行期三道保险：拿不到 Metal 设备 / 内核编译失败 / dispatch 数量不足
       （< 1024 次投影，不值得往返开销）→ 静默回退 CPU，绝不影响正确性。
       `OOO_GPU_DEBUG=1` 打印设备名与失败原因。
     - `apple/gpu.rs`：一个 `project_kernel`（MSL 3.0，SoA 缓冲，行主序 3×3 旋转，
       4 种相机模型含 SIMPLE_RADIAL/RADIAL 畸变），接到两个热点：
       `ba.rs::robust_cost`（每次 LM 迭代）与 `reconstruction.rs::update_point3d_errors`。
       鲁棒代价的**求和仍在 CPU 且顺序不变**，所以 LM 收敛行为与 CPU 版一致。
     - 三个单测守住它：CPU 回退与 `camera::img_from_cam` 逐位一致；GPU 内核与 CPU
       对拍（f32 误差，实测 1.3e-3 px，阈值 5e-3）；`ba_pipeline_reaches_metal_dispatch`
       用 220 点 × 6 观测（1320 次投影）证明 BA 真的发起了 dispatch，而不是静默走 CPU。
     - **仍是 CPU（走 rayon）**：BA 的雅可比线性化、Schur 装配/回代、EPnP 的小规模
       SVD、RANSAC 随机性。下一步该搬的是 Schur 压缩/回代与雅可比线性化。
- 踩坑（写给未来的自己）：
  - **多视图 DLT 的光线必须是单位长度**（4.2.1 `CamRayFromImg` 的语义）。`term =
    P − ray·rayᵀP` 只有 `ray` 是单位向量时才是"到光轴的正交投影"，否则 `A` 不再是
    代数误差的 Hessian，解会系统性偏离（实测 0.25 px → 恢复到 1e-6）。
  - **零空间符号不定**：2 视图 DLT 与多视图特征向量的符号都是任意的。我们显式
    "取第一台相机深度为正的那一支"，否则好对会被 cheirality 判成非 cheiral 而全被拒
    （症状：初始三角化 0 点）。
  - 2 视图多观测时 `AᵀA` 的零空间是 **2 维**的 → 特征向量不唯一；4.2.1 也因此对
    2 视图走闭式解，多视图只用于 >2 视图。
  - **E 分解在纯平移场景下无解可循**：4 解的旋转全是 ±90°（不含单位阵），
    cheirality 无法恢复 `R = I`。合成场景必须给相机加旋转（或横向平移）。
  - `R1 = U·W·Vᵀ` 里的 `V` 是 `matrixV().transpose()`（即 Vᵀ）——把 `Vᵀ` 又转置一次
    会得到完全错误的旋转。
  - **线性 DLT 的平移列不能直接取**：零空间向量是 `λ·[R|t]`，`λ` 任意；把旋转块
    正交化（丢掉 λ）却沿用原平移列 → 平移差一个倍数。EPnP 靠 Kabsch 从头就只拟合
    单位尺度的刚体变换，没有这个问题。
  - **MSL 3.0 里函数内的指针必须显式标注地址空间**：`const float* p = params + ...`
    编译报 `pointer type must have explicit address space qualifier`，要写
    `const device float* p = params + ...`。
  - **MSL 没有 `f64`**：上传前必须降到 `f32`。1e3 量级的像素坐标上相对误差 ~1e-6
    （实测 1.3e-3 px）。所以 batch 容器本身保持 `f64`，只在上传那一步窄化 —— 否则
    「关掉 feature」时的 CPU 路径也会跟着掉精度，产物就会随是否开 GPU 而变。
  - **nalgebra 的 `SVD::solve(&b, eps)` 在 `u`/`v_t` 缺失时返回 `Err`**：EPnP 的
    `FindBetas*` / `RunGaussNewton` 都要走伪逆最小二乘，统一封一个
    `least_squares()`（`try_new` + `solve`）即可，别直接 `unwrap`。
  - **`R = I` 的纯平移两视图无法定基线方向**：`E = [t]×` 只确定方向的正负
    （`[−t]× = −[t]×`），两种符号都可能全部 cheiral。合成场景必须加旋转。
  - 合成场景还要避免"过光轴"的退化视线（多视图 DLT 在基线方向无约束）。
- 测试：`camera` 5 个、`reconstruction::tests` 5 个、`triangulator::tests` 6 个、
  `relative_pose::tests` 2 个、`mapper::tests` 2 个（其中环形场景 1 个 `#[ignore]`）。
  全套 `cargo test --lib --features backend-ooo-sfm` **392 passed / 0 failed / 6 ignored**；
  `cargo check --lib`（不带 feature）同样通过。

### M6 — BA（必做，按平台）✅
- [x] 实现 **CPU BA**（`ooo_sfm/apple/ba.rs`）：Schur 消元（点块）+ LM（Marquardt 阻尼），
      损失函数 Trivial / SoftL1 / Huber / Cauchy 与 Ceres 同构（一阶权重 `sqrt(ρ')`）。
      精简相机系统 ≤ 400 dof 稠密 Cholesky 直接解（等价 SPARSE_SCHUR），更大时
      块-Jacobi 预条件 CG（等价 ITERATIVE_SCHUR + Jacobi）。位姿切空间用左扰动
      `R ← Exp(δθ)·R, t ← Exp(δθ)·t + δt`（与 COLMAP 四元数流形等价）。
      规范固定：`TwoCamsFromWorld`（第一相机全固定 + 第二相机基线最大平移维，
      6+1=7 恰好固定相似规范；**不能**再固定第二相机旋转，否则过约束 → 零代价解
      不可达 → 残差 ∝ σ² 的假驻点）、`ThreePoints`（3 个线性无关的点）。
      相机模型 SIMPLE_PINHOLE / PINHOLE / SIMPLE_RADIAL / RADIAL（解析雅可比，
      有限差分校验到 1e-3）；`cam_ray_from_image`（PINHOLE 解析 + k 系牛顿迭代，
      对应 COLMAP `CamRayFromImg` / `IterativeUndistortion`）。
- [x] 门面：`select` / `select_for_mapper` 返回 CPU BA；`read_database_stats` 读自研
      `database.db`（SUM(rows) 语义与原实现一致）；`meets_caspar_threshold` 恒 false。
- [x] 测试（`ba::tests` + `linalg::tests`）：损失函数 Ceres 语义、4 模型投影雅可比
      有限差分校验、位姿雅可比校验、合成场景恢复（6 图 80 点，σ=0.01 扰动 →
      代价 < 1e-6，相似不变量 + 焦距/畸变恢复）、规范固定语义、常量块不被修改、
      离群点下 Cauchy 优于 Trivial、`cam_ray` 往返、未知模型快速失败。
- 踩坑（写给未来的自己）：
  - 位姿雅可比数组必须是**行主序 2×6**（先 u 行后 v 行）——曾按列主序写、按行主序
    读，雅可比块整体被搅乱（症状：所有 LM 步被拒、λ 爆炸）。
  - 步应用必须与雅可比约定一致：`t' = Exp(δθ)·t + δt`，**不是** `(Exp(δθ)·R)·t + δt`
    （症状：预测下降为正但实际代价爆炸到 1e62）。
  - 法方程右端是 **−g**：`(H+λD)δ = −g`（症状：方向上升、predicted < 0）。
  - 测试场景相机必须等距（球面轨迹）+ 深度过滤，否则病态观测让数值梯度爆炸。
  - 预测下降量公式：`½(−gᵀδ + λδᵀDδ)`（g 为组装值时真实预测 ×2，因 cost 未取 ½；
    只影响 ρ 比值，不影响解）。

### M7 — 健康 / 能力（`health.rs`）——自研侧已完成
- [x] 自研提供 `capabilities()` / `identity()`（`engines/ooo_sfm/health.rs`：
      `CAPABILITIES` 阶段清单 + `LOCKED_VERSION` + `INPUT_VERSION`=20）；主分支那套
      「跑 `-h` 嗅探 CLI 家族」的逻辑在 feature 开启时整体跳过。
- [x] `cpu_only` / `colmap_cli_family` / `acceleration` 状态对齐（`engine_status()` /
      `acceleration()`；Metal 可用时 `GpuReady` + 芯片名，否则 `MacOsCpuOnly`，
      前端零改）。`engines/health.rs::check_colmap` 在 `feature = "backend-ooo-sfm"` 下
      直接返回后端自报的状态，**不再探测二进制、不再因「没有 colmap 文件」判缺引擎**。
- [x] `require_verified_cli` 语义：`ooo_sfm/mod.rs` 里已经是 no-op `Ok(())`
      （进程内后端没有二进制要校验），主分支原实现不动。
- [x] dev 启动不再被 Brush 下载卡死（`scripts/local-tauri.mjs`）：
      `brushPreparationPolicy(action, OOO_SKIP_BRUSH)` —— `build` 仍必须准备好
      （缺 `brush_app` 会产出坏包），`dev` 改成警告后继续（Brush/训练类任务真正
      启动时才由引擎门禁报准确错误），`OOO_SKIP_BRUSH=1` 连警告都跳过。
      另加 `enginePreflight()`：启动前只报事实（缺哪些引擎、在期望路径下没找到），
      不做任何下载 —— macOS 的 `EnginePaths` 按设计**不回退 PATH**，所以 Homebrew
      里的 ffmpeg 不算数，这类信息必须提前说清楚而不是等任务启动报"引擎缺失"。
- [ ] **M10 剩余**：macOS 侧移除对 COLMAP 二进制资源的**打包**依赖（运行时已经不需要，
      但 `scripts/local-tauri.mjs` 仍在把 `../engines/macos/arm64/colmap/` 塞进 bundle，
      连带 `verify:licenses:macos` 的声明也要改）。

### M8 — 进度 / 日志 / 取消接入
- [x] 加 `ProcessUpdate::StageProgress`；`process_observer` 新分支（随 M1 落地）。
- [ ] SfM 阶段全部走结构化进度（算法落地后逐阶段补 `sink.stage(...)` 调用点）；
      **M3 已补 `featureExtraction`、M4 已补 `matching`**（前端对该 phase 暂时回退到 `runtime.running` 文案，M7 补）；`Started` / `Heartbeat` / `Exited` 替代。M5 落地时补 `mapping`。
- [ ] 只动 SfM 模式，FFmpeg / Brush 不动。

### M9 — 检查点 / 遥测
- [ ] `COLMAP_INPUT_VERSION` 纳入自研版本。
- [ ] 错误文案分类对齐（前端字段不改，D2）。

### M10 — 打包 / 构建 / CI
- [x] macOS 构建默认启用 feature（`scripts/local-tauri.mjs` 的 `localFeatures()`：
      darwin → `local-colmap,ooo-sfm`；win/linux 不变。Metal 不默认开，
      `OOO_SFM_METAL=1` 才加 `backend-ooo-sfm-metal`。既有 node 测试同步更新并新增一条
      「macOS 开自研 / Win-Linux 保持 COLMAP / Metal 必须显式」的断言）。
- [ ] CI 双构建：`--features backend-ooo-sfm` 与不带 feature。
- [ ] 许可证：macOS 侧移除对 COLMAP 二进制的运行时依赖；其他平台仍保留 COLMAP 的 license 声明。

### M11 — 端到端验收
- [ ] 视频（顺序匹配）与图片（穷举匹配）两条链路各跑通一次完整生成任务。
- [ ] 输出 PLY 与主分支一致（`final.ply` 校验通过）。
- [ ] 取消 / 失败 / 断点续跑 行为一致。

---

## 7.5 架构：前端与计算后端解耦（后端抽象层）

> 2026-10 的重构：把"前端 ↔ 计算后端"的耦合收敛成**一个契约层**，
> 并把后端选择改成**编译期 feature** 决定。

### 三层结构

```
前端 (TS)                只认 contracts/ 里的中性类型
   │  tauri command / event
   ▼
pipeline / services      只认 engines::backend::SfMBackend trait
   │
   ▼
engines::backend         契约 + 中性请求 + 编译期选择
   │
   ├── backend-colmap（默认）   → 外部 COLMAP CLI
   └── backend-ooo-sfm           → 自研进程内后端（Apple）
```

### 1. `src-tauri/src/contracts/` —— 前端唯一能看见的词汇表

前端**不再**看到任何实现名：

| 旧（实现耦合） | 新（中性契约） |
|---|---|
| `EngineKind::{Ffmpeg,Ffprobe,Colmap,Brush}` | `ComponentKind::{Media,Reconstruction,Training}` |
| `EngineStatus { exists, can_start, path, version, cpu_only, colmap_cli_family }` | `ComponentStatus { ready, acceleration, detail, location }`（前端从不消费的那些字段直接不出 IPC） |
| `ColmapAccelerationStatus { backend, detection_state, ... }` | `AccelerationStatus { mode, state, reason_code, ... }` |
| `AccelerationReasonCode::{MacOsCpuOnly, ColmapUnavailable, NvidiaSmiNotFound, ...}` | `AccelerationReason::{CpuOnly, ComponentUnavailable, NoAccelerator, ...}` |
| `PipelineEngine::{Ffmpeg,Colmap,Brush}` | `PipelineComponent::{Media,Reconstruction,Training}` |
| 失败分类 `mapper_source` / `brush_dataset` | `FailureKind::ReconstructionInput` / `TrainingDataset` |
| 命令 `check_engines` / `check_colmap_acceleration` | `check_components` / `get_acceleration` |

前端侧对应改动：`src/types/pipeline.ts` 重命名、`src/lib/backend.ts` 命令名、
`src/stores/appStore.ts` 状态字段、`src/i18n/index.tsx` 文案（"COLMAP 正在使用 CPU 提取特征"
→ "正在使用 CPU 提取特征"）、`src/lib/errorSummary.ts` 分类正则、
`server/diagnostics/report.ts` 白名单。

### 2. `src-tauri/src/engines/backend/` —— 后端契约与编译期选择

* `SfMBackend` trait（`async-trait`）：`identity` / `capabilities` / `require_ready` +
  9 个阶段方法（特征提取 ×3、匹配 ×3、重建 ×3）。
* 中性请求结构体（`request.rs`）：**不含**可执行文件路径、CUDA 设备号、COLMAP 的 CLI 开关。
  加速器设备号放在 `JobContext`（中性概念："用哪个加速器"）。
* 适配层 `adapter.rs` 只有一份：两侧实现由 `engines/mod.rs` 的 `#[path]` 门面保证
  **同名同签名**，所以适配逻辑不可能两边漂移。
* **编译期选择**：

  | feature | 后端 |
  |---|---|
  | 都不开 | COLMAP CLI（与主分支行为完全一致） |
  | `backend-colmap` | COLMAP CLI（显式） |
  | `backend-ooo-sfm` | 自研进程内后端 |
  | `backend-ooo-sfm-metal` | 自研 + Metal 内核 |

  同时开 `backend-colmap` 与 `backend-ooo-sfm` → **编译错误**（后端不唯一）；
  在非 Apple 平台开 `backend-ooo-sfm` → 编译错误（只有 Apple 实现）。

### 3. pipeline 侧的收获

`pipeline/runner.rs` 的 11 个引擎调用点全部改走 trait，不再出现 `colmap::` 字样；
调用点只见"阶段请求 + 作业环境"，不再拼 `&self.engines.colmap` / `gpu_index` 这类实现细节。

### 还没做的（下一轮）

* 日志文件名仍是 `colmap.log` / `brush.log`（文件系统约定，前端按名字展示）；
* 遥测错误码仍是 `ColmapFeatureFailed` 等（与服务端的既有契约，不能随手改）；
* `runner.rs` 里"按可执行文件名推断组件类别"的逻辑对进程内后端不适用
  （进程内后端自己上报 `Runtime` 事件，不走这条路）。

## 7.6 架构：媒体处理后端（CLI ↔ rsmpeg）

> 2026-10 的第二块解耦：把"视频探测 + 抽帧"从**直接 spawn ffmpeg 二进制**改成
> 一个 trait，两种实现**编译期二选一**。

### 分层

```
pipeline / commands / splatstudio
        │  只认 media::MediaTool trait
        ▼
media::MediaTool
        ├── media-ffmpeg-cli（默认）→ 外部 ffmpeg/ffprobe 进程（与主分支一致）
        └── media-rsmpeg            → 进程内 FFmpeg 库（rsmpeg 0.18）
```

### feature

| feature | 媒体实现 | 依赖 |
|---|---|---|
| 都不开 | 外部 ffmpeg/ffprobe 进程 | 无（与主分支完全一致） |
| `media-ffmpeg-cli` | 同上（显式） | 无 |
| `media-rsmpeg` | 进程内 FFmpeg 库 | `rsmpeg 0.18` + `rusty_ffmpeg 0.16.7` + `png 0.18` |

同时开两个 → **编译错误**（实现不唯一）。

### `media-rsmpeg` 的硬前提：FFmpeg ABI 必须匹配

`rusty_ffmpeg` 带**预置绑定**（不跑 bindgen），描述的是固定版本的 FFmpeg 结构体
布局；系统 FFmpeg 版本与之不符时，绑定与真实库不一致 → 运行期字段错位。实测：

* rsmpeg 0.18 的绑定期望 **FFmpeg 8.x**（`AVCodecID` 是 `uint32_t`）；
* 本机 Homebrew 的 `ffmpeg` 是 **9.0.2**（libavcodec 63.x，`AVCodecID` 又是枚举）→
  绑定与库不一致，编译期就报 `expected u32, found i32`。

因此（rsmpeg 路径统一走**本地静态库**，见下一节）：

* 一次性构建：`npm run setup:media-ffmpeg` → `.cache/media-ffmpeg/ffmpeg-8.1.2-static/`
  （源码与 SHA-256 由 `engines/manifest.macos.json` 锁定，LGPL、无外部依赖，本机部署目标）；
* 构建脚本自动设置 `FFMPEG_INCLUDE_DIR` / `FFMPEG_LIBS_DIR` / `FFMPEG_LINK_MODE=static`；
* 先体检：`npm run check:ffmpeg-abi`（静态前缀就绪 + 版本匹配才算过）；
* `build.rs` 里有 **ABI 守卫**：优先读静态前缀里的 `libavcodec/version.h`（与链接同源），
  libavcodec 主版本不是 62（即 FFmpeg 8.x）就**直接拒绝编译**并给出修复命令。

### `media-rsmpeg` 必须静态链接（真实事故）

第一版 rsmpeg 路径链接 Homebrew keg 的 `libavcodec.62.dylib`，打包出的 .app 在
**本机启动即崩**：

```
Termination Reason: Namespace DYLD, Code 1 Library missing
Library not loaded: '/opt/homebrew/.../libavcodec.62.dylib'
Reason: ... not valid for use in process (no Team ID, not a platform binary)
```

两层问题，都不可能在 keg 依赖上修好：

1. **绝对路径引用**：Homebrew dylib 的 install name 是 `/opt/homebrew/...`，链接时
   原样进二进制 → bundle 换台机器必断。官方引擎门禁的
   "Non-relocatable dependency (/opt/homebrew)" 检查防的就是它；
2. **签名拒绝**：已签名的宿主进程加载签名状态不合规的 dylib，macOS 直接拒绝。

修复选**静态链接**（FFmpeg 编成 `.a` 链进二进制）：没有 dylib、没有 rpath、没有
签名问题，bundle 结构性自包含。实现：

* `scripts/setup-media-ffmpeg.sh`：源码构建 FFmpeg 8.1.2 静态库
  （`--enable-static --disable-shared --disable-programs --disable-autodetect`，
  与官方同套 LGPL 能力策略；幂等，`lib/READY` 标记跳过）；
* `local-tauri.mjs` 自动设 `FFMPEG_LIBS_DIR`（rusty_ffmpeg 一见它就绕开 pkg-config，
  且该路径默认 `static` 链接模式）；
* `engines::health`（`check_components` 数据源）：媒体组件有 `media-rsmpeg` 进程内
  分支（与自研重建后端的分支同构）——不探测 ffmpeg/ffprobe 二进制、直接报就绪。
  缺这个分支的真实事故：`OOO_MEDIA_RSMPEG=1` 的产品包裁掉两个二进制后，
  顶栏常驻"N 个引擎异常"、开始按钮被禁用（`canStart` 要求三组件全就绪）；
* 手动跑 cargo 时要带同样的环境变量（见 `npm run check:ffmpeg-abi` 的输出）。

取舍：二进制体积变大（约 +30MB）；LGPL 静态链接仅限本地/内部构建 ——
若未来要把 media-rsmpeg 带进发行包，需回退到 shared + `@rpath` 方案
（官方 `build-engines-macos.sh` 已是现成参考）。

### 输出约定（两种实现一致）

* 画面：`frame_%06d.png`（1 起，带 alpha 时）或 `frame_%06d.jpg`；
* 掩膜：`frame_%06d.png.png`（灰度，alpha ≥ 128 → 255，与 CLI 的 `lut` 规则一致）；
* COLMAP 目录镜像、抽帧后校验、源帧改名等**共享后处理直接复用**
  `crate::engines::ffmpeg` 里的实现（`ensure_clean_output` / `mirror_images` /
  `rename_selected_outputs` / `validate_extraction`），不重复实现。

### 已知差异

| 项 | CLI | rsmpeg | 说明 |
|---|---|---|---|
| 旋转信息（手机竖拍） | 从 display matrix 读 | 报 0 | rsmpeg 未暴露流侧数据；竖拍视频的尺寸会与 CLI 不同 |
| 无 alpha 时的画质 | JPEG（`-q:v 2`） | PNG | 体积更大，但无损；抽帧后的清点/校验/重命名/删除都按"是图片就算"（`pipeline_image_names` / `pipeline_image_files`），扩展名不再是语义（真实事故：`rename_selected_outputs` 曾硬编码 JPG → `selected 285 / wrote 0`） |
| 补拍合并 | 临时目录 + 原子合并 | 复用 CLI 路径 | 避免两套合并逻辑漂移 |

### 实测（本机 macOS 12.7.6 / Apple M1 / FFmpeg 8.1.3）

| 项 | 结果 |
|---|---|
| `npm run check:ffmpeg-abi` | `ffmpeg@8: libavcodec 62.28.103 ✓ 匹配` / 系统默认 63.1.102 ✗ |
| `cargo test --features media-rsmpeg --lib` | 337 passed / 0 failed（默认 CLI 路径 335 passed） |
| `cargo check --features media-ffmpeg-cli,media-rsmpeg` | 编译错误（实现不唯一）✓ |
| rsmpeg 探测 vs CLI 探测 | 同一段夹具视频，**宽高/帧率/帧数完全一致**（共用同一份期望值） |
| rsmpeg 抽帧 | 帧数、命名（`frame_000001.png`）、COLMAP 副本目录都对齐 CLI 约定 |

踩过的坑（写给未来的自己）：

* **解码器必须先 `apply_codecpar`**：只用 `AVCodecContext::new(codec)` + `open()`
  解码会返回 `AVERROR_INVALIDDATA`（-1094995529）——解码器不知道宽高/像素格式。
* **只喂视频流的数据包**：`read_packet()` 会返回所有流的数据包，混喂会让解码器报错。
* **`AVFrame::data` / `linesize` 是按平面索引（最多 8 个），不是按行**：按行去索引
  `linesize[row]` 会在 height > 8 时越界。打包格式只有一个平面，逐行按
  `linesize[0]` 的行距拷贝。
* **`AVFrame` 不可克隆**（析构会释放缓冲）：要"补帧"（源帧率 > 目标帧率）就不能靠复制
  帧 → 改成**缩放后的像素缓冲**（`ScaledImage`）留在采样器里复用。

### 本地完整打包：`npm run build:local`

发行版门禁（`verify:engines` / `verify:licenses` / `verify:colmap-lock`）为**官方定制引擎**
设计（FFmpeg 8.1.2 定制构建 + brush_app + 锁定 CPU COLMAP + SHA256SUMS 完整性）。
官方引擎包 `engines-macos-arm64-v0.3.0` 尚未发布（release 404，manifest:
`pending-first-build-and-hash-lock`），因此发行打包暂不可行——但**发行门禁与代码编译
解耦**之后，本地用替代后端已经可以打出完整可运行的包：

```bash
# 一次性准备（media-rsmpeg 的静态 FFmpeg 链接依赖）：
npm run setup:media-ffmpeg

# 快速本地包（debug profile，几分钟）：进程内媒体 + 自研 SfM，跳过 brush
OOO_SKIP_BRUSH=1 OOO_MEDIA_RSMPEG=1 npm run build:local -- --debug
# → src-tauri/target/debug/bundle/macos/OOOSplat.app + dmg

# release 包（同一条命令去掉 --debug）
OOO_MEDIA_RSMPEG=1 npm run build:local
```

环境变量（只影响本地路径）：

| 变量 | 作用 |
|---|---|
| `OOO_MEDIA_RSMPEG=1` | 启用 `media-rsmpeg` feature：媒体处理进程内完成，**不打包也不需要 ffmpeg/ffprobe 二进制**；自动设置静态库环境（`FFMPEG_LIBS_DIR` 等，需先 `npm run setup:media-ffmpeg`，缺了会直接报错并指路） |
| `OOO_SKIP_BRUSH=1` | 不打包 brush_app（构建不再尝试下载/安装；训练任务运行期会明确报缺引擎） |
| `OOO_SFM_METAL=1` | 追加 Metal GPU feature（既有语义不变） |

本地 bundle 的取舍（`local-tauri.mjs` 的 `localConfiguration` options）：

* 不跑 `verify:engines` / `verify:licenses` / `verify:colmap-lock`（那是发行门禁）；
* `LSMinimumSystemVersion` 用**本机系统版本**（发行锁 15.0 是为官方引擎定的；
  本机 macOS 12.7.6 上 15.0 的 .app 无法通过 LaunchServices 启动）；
* ad-hoc 签名（`signingIdentity: "-"`），无网络依赖。

实测（本机 M1 / macOS 12.7.6）：debug bundle 产出 `OOOSplat.app` + 39MB dmg，
`LSMinimumSystemVersion=12.7.6`，arm64 双二进制，bundle 内无任何引擎二进制——
SfM（`backend-ooo-sfm` 进程内）与媒体（`media-rsmpeg` 进程内）自足，Brush 训练除外。

### 测试

`media::tests`：

* `cli_probe_matches_expectations`：用 ffmpeg CLI 造 1 秒 320×240@10fps 的夹具，
  断言探测结果；
* `rsmpeg_probe_matches_expectations`（`media-rsmpeg`）：**同一段视频、同一份期望值**
  —— 两种实现天然互相对拍；
* `rsmpeg_extracts_frames_with_the_cli_naming`（`media-rsmpeg`）：断言抽帧数量与
  文件命名约定。

## 8. 测试策略

| 层 | 测什么 |
|---|---|
| 格式单测 | bin 写入器 ↔ 现有 reader 往返一致；相机枚举全覆盖 |
| `database.db` | 写出的表结构与 `validate_database_camera` / `read_database_stats` 兼容 |
| 后端单测 | 每个 `colmap::*` 函数在 fixture 输入下的输出存在性与语义 |
| 进度 | `StageProgress` → `events.send` 映射；`runtime` 更新 |
| 取消 | 计算中途 `cancel()` → 快速返回且无残留半成品 |
| 契约对比 | 同一 fixture 下，主分支 COLMAP 版与 ooo-sfm 版的 `sparse/*.bin` 解析结果一致（注册率、点数、相机） |
| 端到端 | 视频 / 图片各一条完整 pipeline |

---

## 9. 构建与打包

- 无 sidecar、无新二进制；一切随项目编译。
- `tauri build --features backend-ooo-sfm`（macOS 默认）。
- 引擎 manifest / SHA 锁 / `scripts/setup-colmap-*` 等**仅供其他平台/兜底**使用，本分支不新增。

---

## 10. 风险与回退

| 风险 | 说明 | 缓解 |
|---|---|---|
| panic / OOM 砸主进程 | 进程内无隔离（子进程原本免费提供） | `catch_unwind` + 专用线程；评估 Brush OOM 重试上下文 |
| 阻塞 async 运行时 | 重 CPU/GPU 同步计算 | 一律 `spawn_blocking` / 专用线程 |
| 取消不生效 | 进程内无法 kill | 计算分块 + 循环查 `child_token()` |
| 格式不一致 | Brush/校验崩 | §2 往返 + 契约对比测试锁死 |
| `database.db` 不一致 | `validate_database_camera` / BA 统计失效 | 产出兼容 DB（D3）+ 往返测试 |
| 主分支侵入 | rebase 冲突 | 只加新文件 + 少量 cfg；`colmap.rs` 不动 |

**回退**：不带 feature 编译即回到主分支 COLMAP 行为，随时可对照。

---

## 11. 已定决策与待定问题

### 已定

| # | 决策 | 影响 |
|---|---|---|
| D1 | **平台范围：仅 macOS** | 构建脚本/CI 只对 macOS 启用 `backend-ooo-sfm`；Win/Linux 继续用 COLMAP 二进制 |
| D2 | **前端状态字段全部不改** | `ColmapBackend` / `ColmapAccelerationStatus` / `colmapCliFamily` 等保持原语义与原名，自研实现负责填充等价值；前端零改动 |
| D3 | **`database.db` 需要就加入** | 产出兼容 `database.db`（`cameras` / `keypoints` / `two_view_geometries`），使 `validate_database_camera`（补拍）与 `read_database_stats` **无需改动** |
| D4 | **BA 按平台重写一套，纳入本 fork** | `colmap_ba` 为**真实实现**；macOS = CPU BA，接口按平台抽象，预留后续 GPU/Metal 变体 |
| D5 | **闭环检测第一步不做，预留接口** | 匹配阶段定义闭环检测接口，默认 **no-op**；词袋树 / `vocabulary.rs` 第一步不接入 |
| D6 | **命名 `backend-ooo-sfm` + 平台子模块** | feature/分支/领域模块用 `backend-ooo-sfm`（跨平台中性）；平台实现放 `ooo_sfm/<platform>/`；`metal` 仅作平台内加速路径名 |

### 待定

（无——原三项均已确定；实现细节随里程碑推进。）

### 概念说明

- **`database.db`**：COLMAP 的中间 SQLite 库，是各子命令之间的"交接文件"
  （`feature_extractor` 写 `keypoints`/`descriptors`；matcher 写 `matches`/`two_view_geometries`；
  `mapper` 读它们）。**本项目自己**只读两处：`colmap_ba::read_database_stats`（判断是否够大以启用
  Caspar，纯优化启发式）与 `runner.rs::validate_database_camera`（补拍时校验相机一致）。本方案按
  **D3** 产出兼容 DB。
- **BA（Bundle Adjustment，光束法平差）**：对相机位姿/内参与 3D 点做联合非线性最小二乘，
  最小化重投影误差，是 SfM 的精修步骤。COLMAP 后端为 `CERES`（CPU，默认）或 `CASPAR`
  （该 fork 的 CUDA GPU BA）。macOS 为 `cuda:false, caspar:false`。本方案按 **D4** 自研 CPU BA。
- **闭环检测**：顺序匹配只匹配相邻帧；相机绕回时会断裂。闭环检测用语袋树
  （`vocab_tree_faiss_flickr100K_words256K.bin`，72MB）检索视觉相似帧补匹配，本项目仅在
  自动优化路径开启。本方案按 **D5** 第一步 no-op、预留接口。
