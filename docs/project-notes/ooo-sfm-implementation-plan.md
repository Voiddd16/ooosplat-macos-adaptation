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

- **角色层**（feature / 分支 / 顶层领域模块）：`ooo-sfm`（契合仓库第一方命名，如 `OOOBrush`）
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
#[cfg(not(feature = "ooo-sfm"))]
#[path = "colmap.rs"]
pub mod colmap;

#[cfg(not(feature = "ooo-sfm"))]
pub mod colmap_ba;

#[cfg(feature = "ooo-sfm")]
#[path = "ooo_sfm/mod.rs"]
pub mod colmap;          // 对外暴露与 colmap.rs 同名同签名的 API

#[cfg(feature = "ooo-sfm")]
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
tauri build --features ooo-sfm
# 主分支等价版（对照 / 回归）
cargo build                 # 不带 feature
```

> **效果**：`runner.rs` 的 `colmap::*` / `colmap_ba::*` 调用点**一行不改**；
> 主分支文件全量保留。本分支 diff ≈ 新目录 + `engines/mod.rs` 几行 cfg + `health.rs` 一小段。

### 1.4 平台自动启用

推荐用 **feature**（可在任意平台回归测试），在本平台构建配置里默认打开：
例如在 `scripts/local-tauri.mjs` / 打包脚本 / `tauri` 构建参数中对 `target_os == macos` 追加
`--features ooo-sfm`。**不要**硬绑 `#[cfg(target_os)]` 作为开关，否则无法在 Win/Linux 上验证自研实现
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

> 每个里程碑结束都应保证 `cargo build --features ooo-sfm` 通过 + 对应测试通过。

### M0 — 骨架与门面 ✅
- [x] 建 `engines/ooo_sfm/`（领域层 + `ooo_sfm/apple/` 平台后端）模块骨架，同名函数先 `todo!()`。
- [x] `engines/mod.rs` 加 `#[cfg] + #[path]` 门面。
- [x] `Cargo.toml` 加 `ooo-sfm` feature。
- [x] 验证：开/关 feature 均可编译。
  - stub crate 类型检查（同一门面加载真实源码 + `runner.rs` / `health.rs` 调用形态）→ 通过。
  - 整工程 `cargo check --lib`：不带 feature ✅ / `--features ooo-sfm` ✅（rustc 1.99.1）。
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
- 并行：按图用 `rayon`（新依赖，挂 `ooo-sfm` feature）并行提取，SQLite 保持单写者
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
  恢复、F 极线残差），全套 `cargo test --lib --features ooo-sfm` 346+ 全绿。

### M5 — 稀疏重建（`mapper` 等价）
- [ ] `map` / `map_from_existing` / `map_incremental`。
- [ ] 输出 `sparse/<n>/{cameras,images,points3D}.bin`（+ 可选 rigs/frames 成对）。
- [ ] 注册率语义与 `validator.rs` 对齐。
- [ ] `allow_two_view_tracks` 语义等价。

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

### M7 — 健康 / 能力（`health.rs`）
- [ ] 自研提供 `capabilities()` / `identity()`；替换 `-h` 嗅探。
- [ ] `cpu_only` / `colmap_cli_family` / `acceleration` 状态对齐（macOS 复用 `MacOsCpuOnly`；D2 前端零改）。
- [ ] `require_verified_cli` 语义。

### M8 — 进度 / 日志 / 取消接入
- [x] 加 `ProcessUpdate::StageProgress`；`process_observer` 新分支（随 M1 落地）。
- [ ] SfM 阶段全部走结构化进度（算法落地后逐阶段补 `sink.stage(...)` 调用点）；
      **M3 已补 `featureExtraction`、M4 已补 `matching`**（前端对该 phase 暂时回退到 `runtime.running` 文案，M7 补）；`Started` / `Heartbeat` / `Exited` 替代。M5 落地时补 `mapping`。
- [ ] 只动 SfM 模式，FFmpeg / Brush 不动。

### M9 — 检查点 / 遥测
- [ ] `COLMAP_INPUT_VERSION` 纳入自研版本。
- [ ] 错误文案分类对齐（前端字段不改，D2）。

### M10 — 打包 / 构建 / CI
- [ ] macOS 构建默认启用 feature（构建脚本）。
- [ ] CI 双构建：`--features ooo-sfm` 与不带 feature。
- [ ] 许可证：macOS 侧移除对 COLMAP 二进制的运行时依赖；其他平台仍保留 COLMAP 的 license 声明。

### M11 — 端到端验收
- [ ] 视频（顺序匹配）与图片（穷举匹配）两条链路各跑通一次完整生成任务。
- [ ] 输出 PLY 与主分支一致（`final.ply` 校验通过）。
- [ ] 取消 / 失败 / 断点续跑 行为一致。

---

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
- `tauri build --features ooo-sfm`（macOS 默认）。
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
| D1 | **平台范围：仅 macOS** | 构建脚本/CI 只对 macOS 启用 `ooo-sfm`；Win/Linux 继续用 COLMAP 二进制 |
| D2 | **前端状态字段全部不改** | `ColmapBackend` / `ColmapAccelerationStatus` / `colmapCliFamily` 等保持原语义与原名，自研实现负责填充等价值；前端零改动 |
| D3 | **`database.db` 需要就加入** | 产出兼容 `database.db`（`cameras` / `keypoints` / `two_view_geometries`），使 `validate_database_camera`（补拍）与 `read_database_stats` **无需改动** |
| D4 | **BA 按平台重写一套，纳入本 fork** | `colmap_ba` 为**真实实现**；macOS = CPU BA，接口按平台抽象，预留后续 GPU/Metal 变体 |
| D5 | **闭环检测第一步不做，预留接口** | 匹配阶段定义闭环检测接口，默认 **no-op**；词袋树 / `vocabulary.rs` 第一步不接入 |
| D6 | **命名 `ooo-sfm` + 平台子模块** | feature/分支/领域模块用 `ooo-sfm`（跨平台中性）；平台实现放 `ooo_sfm/<platform>/`；`metal` 仅作平台内加速路径名 |

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
