use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use serde::Deserialize;

use crate::{
    engines::colmap::{detect_cli_family, ColmapCliFamily},
    process::{ProcessManager, ProcessSpec},
};

// 契约类型（前端可见，中性命名）统一放在 `crate::contracts`。
pub use crate::contracts::{
    AcceleratorDevice, AcceleratorRequirements, AccelerationMode, AccelerationReason,
    AccelerationState, AccelerationStatus, ComponentKind, ComponentStatus,
};

/// 被探测的**工具**（内部细节，不进 IPC）。
///
/// 刻意保留"具体工具"粒度：ffmpeg 与 ffprobe 要分别探测，COLMAP 还要探测它的
/// 6 个子命令。聚合成前端可见的 [`ComponentKind`] 在 `component_status()` 里做。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    Ffmpeg,
    Ffprobe,
    Reconstruction,
    Training,
}

/// 单个工具的探测结果（内部）。
#[derive(Debug, Clone)]
pub struct ToolStatus {
    pub kind: ToolKind,
    pub path: PathBuf,
    pub exists: bool,
    pub can_start: bool,
    pub version: Option<String>,
    pub cpu_only: Option<bool>,
    pub acceleration: Option<AccelerationStatus>,
    pub detail: String,
}

impl ToolStatus {
    /// 聚合成前端可见的组件状态。
    ///
    /// 丢弃的字段（`version` / `cpu_only` / `colmap_cli_family`）前端从不消费，
    /// 而它们都是实现专属概念 —— 不出 IPC 边界。
    pub fn component_status(&self) -> ComponentStatus {
        ComponentStatus {
            kind: match self.kind {
                ToolKind::Ffmpeg | ToolKind::Ffprobe => ComponentKind::Media,
                ToolKind::Reconstruction => ComponentKind::Reconstruction,
                ToolKind::Training => ComponentKind::Training,
            },
            ready: self.exists && self.can_start,
            acceleration: self.acceleration.clone(),
            detail: self.detail.clone(),
            location: Some(self.path.display().to_string()),
        }
    }
}

const DEFAULT_MINIMUM_DRIVER: &str = "580.00";
const DEFAULT_MINIMUM_COMPUTE_CAPABILITY: &str = "7.5";
const NVIDIA_SMI_TIMEOUT: Duration = Duration::from_secs(5);
const ACCELERATION_CACHE_TTL: Duration = Duration::from_secs(10 * 60);
const ACCELERATION_DIAGNOSTIC_LIMIT: u64 = 512 * 1024;
const ACCELERATION_OUTPUT_LIMIT: usize = 4 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbePolicy {
    Cached,
    Refresh,
    Task,
}

#[derive(Clone, Default)]
pub struct AccelerationProbeService {
    state: Arc<tokio::sync::Mutex<AccelerationProbeState>>,
    run_lock: Arc<tokio::sync::Mutex<()>>,
}

#[derive(Default)]
struct AccelerationProbeState {
    last_good: Option<(Instant, AccelerationStatus)>,
}

fn acceleration_probe_service() -> &'static AccelerationProbeService {
    static SERVICE: OnceLock<AccelerationProbeService> = OnceLock::new();
    SERVICE.get_or_init(AccelerationProbeService::default)
}

#[derive(Debug, Clone)]
pub struct EnginePaths {
    pub root: PathBuf,
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    pub colmap: PathBuf,
    pub brush: PathBuf,
}

impl EnginePaths {
    pub fn from_root(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        #[cfg(windows)]
        let paths = Self {
            ffmpeg: root.join("ffmpeg").join("ffmpeg.exe"),
            ffprobe: root.join("ffmpeg").join("ffprobe.exe"),
            colmap: root.join("colmap").join("bin").join("colmap.exe"),
            brush: root.join("brush").join("brush_app.exe"),
            root,
        };
        #[cfg(target_os = "macos")]
        let paths = Self {
            ffmpeg: root.join("bin").join("ffmpeg"),
            ffprobe: root.join("bin").join("ffprobe"),
            colmap: macos_colmap_path(&root),
            brush: root.join("bin").join("brush_app"),
            root,
        };
        #[cfg(all(not(windows), not(target_os = "macos")))]
        let paths = Self {
            ffmpeg: root.join("ffmpeg"),
            ffprobe: root.join("ffprobe"),
            colmap: root.join("linux").join("colmap").join("bin").join("colmap"),
            brush: root.join("linux").join("brush").join("brush_app"),
            root,
        };
        paths
    }

    fn from_candidates(root: PathBuf) -> Self {
        #[cfg(windows)]
        let paths = {
            // Windows releases are self-contained. Never let a missing bundled
            // executable silently select an unrelated program from PATH.
            Self::from_root(root)
        };

        #[cfg(target_os = "macos")]
        let paths = {
            // macOS releases are self-contained. Homebrew and PATH are build-time
            // conveniences only and must never become end-user dependencies.
            Self::from_root(root)
        };

        #[cfg(all(not(windows), not(target_os = "macos")))]
        let paths = {
            let defaults = Self::from_root(root.clone());
            Self {
                ffmpeg: resolve_engine(
                    "OOOSPLAT_FFMPEG",
                    std::slice::from_ref(&defaults.ffmpeg),
                    "ffmpeg",
                ),
                ffprobe: resolve_engine(
                    "OOOSPLAT_FFPROBE",
                    std::slice::from_ref(&defaults.ffprobe),
                    "ffprobe",
                ),
                colmap: defaults.colmap,
                // Do not silently choose an old official Brush from PATH.
                // Explicit diagnostic overrides are checked for the new CLI below.
                brush: std::env::var_os("OOOSPLAT_BRUSH")
                    .map(PathBuf::from)
                    .unwrap_or(defaults.brush),
                root,
            }
        };
        paths
    }

    pub fn discover(resource_dir: Option<&Path>) -> Self {
        if let Some(value) = std::env::var_os("OOOSPLAT_ENGINE_DIR") {
            return Self::from_candidates(value.into());
        }

        let current = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let candidates = vec![
            resource_dir.map(|path| path.join("engines")),
            Some(current.join("engines")),
            Some(current.join("..").join("engines")),
        ];
        #[cfg(target_os = "macos")]
        let mut candidates = candidates
            .into_iter()
            .map(|candidate| candidate.map(|path| path.join("macos").join("arm64")))
            .collect::<Vec<_>>();
        #[cfg(not(target_os = "macos"))]
        let mut candidates = candidates;
        if let Ok(executable) = std::env::current_exe() {
            #[cfg(target_os = "macos")]
            candidates.extend(
                executable
                    .ancestors()
                    .skip(1)
                    .map(|ancestor| Some(ancestor.join("engines").join("macos").join("arm64"))),
            );
            #[cfg(not(target_os = "macos"))]
            candidates.extend(
                executable
                    .ancestors()
                    .skip(1)
                    .map(|ancestor| Some(ancestor.join("engines"))),
            );
        }
        let fallback = if cfg!(target_os = "macos") {
            current.join("engines").join("macos").join("arm64")
        } else {
            current.join("engines")
        };
        let root = candidates
            .into_iter()
            .flatten()
            .find(|path| path.is_dir())
            .unwrap_or(fallback);
        Self::from_candidates(root)
    }

    /// 全部组件状态（前端 `check_components` 命令的数据源）。
    pub async fn check_all(&self) -> Vec<ComponentStatus> {
        self.check_all_with_policy(ProbePolicy::Cached).await
    }

    /// 任务启动前的组件状态（`ProbePolicy::Task` = 不吃缓存）。
    pub async fn check_all_for_task(&self) -> Vec<ComponentStatus> {
        self.check_all_with_policy(ProbePolicy::Task).await
    }

    /// 探测并聚合成**中性组件**。
    ///
    /// 前端只看到三类组件（媒体 / 重建 / 训练）；ffmpeg 与 ffprobe 这类工具粒度
    /// 留在内部，聚合规则见 [`merge_media`]。
    async fn check_all_with_policy(&self, policy: ProbePolicy) -> Vec<ComponentStatus> {
        // media-rsmpeg（进程内）：媒体能力在编译期已链接进二进制（FFmpeg 8.x 静态库，
        // build.rs 的 ABI 守卫已核过版本），**没有外部 ffmpeg/ffprobe 可执行文件要探测**
        // —— 与重建组件的自研后端同构：不能因为 bundle 里没有二进制就判缺失。
        // 缺这个分支的真实后果：`OOO_MEDIA_RSMPEG=1` 的产品包资源裁掉了两个二进制，
        // 顶栏常驻"N 个引擎异常"、开始按钮被禁用。
        #[cfg(feature = "media-rsmpeg")]
        let media = in_process_media_status(&self.ffmpeg);
        #[cfg(not(feature = "media-rsmpeg"))]
        let media = {
            let (ffmpeg, ffprobe) = tokio::join!(
                check_basic(ToolKind::Ffmpeg, &self.ffmpeg, &["-version"]),
                check_basic(ToolKind::Ffprobe, &self.ffprobe, &["-version"]),
            );
            merge_media(ffmpeg, ffprobe)
        };
        let (reconstruction, training) = tokio::join!(
            check_colmap(&self.colmap, &self.root, policy),
            check_brush(&self.brush),
        );
        vec![
            media,
            reconstruction.component_status(),
            training.component_status(),
        ]
    }
}

/// 进程内媒体实现（`media-rsmpeg`）的组件状态。
///
/// 与自研重建后端分支同构：media-rsmpeg 没有外部可执行文件，ffmpeg 路径仅作
/// 展示（组件就绪时运行期不会读取它）。
#[cfg(feature = "media-rsmpeg")]
fn in_process_media_status(ffmpeg_path: &Path) -> ComponentStatus {
    ComponentStatus {
        kind: ComponentKind::Media,
        ready: true,
        acceleration: None,
        detail: "媒体处理在进程内完成（rsmpeg + 静态链接 FFmpeg 8.x）；无需外部 ffmpeg/ffprobe"
            .into(),
        location: Some(ffmpeg_path.display().to_string()),
    }
}

/// 把 ffmpeg 与 ffprobe 两个工具聚合成一个"媒体"组件。
///
/// 两个工具都必须就绪才算媒体能力可用（抽帧与探测缺一不可）。
fn merge_media(ffmpeg: ToolStatus, ffprobe: ToolStatus) -> ComponentStatus {
    let ready = ffmpeg.exists
        && ffmpeg.can_start
        && ffprobe.exists
        && ffprobe.can_start;
    let detail = match (ffmpeg.can_start, ffprobe.can_start) {
        (true, true) => "媒体工具就绪".to_string(),
        (true, false) => format!("ffprobe 不可用：{}", ffprobe.detail),
        (false, true) => format!("ffmpeg 不可用：{}", ffmpeg.detail),
        (false, false) => format!("ffmpeg：{}；ffprobe：{}", ffmpeg.detail, ffprobe.detail),
    };
    ComponentStatus {
        kind: ComponentKind::Media,
        ready,
        acceleration: None,
        detail,
        location: Some(ffmpeg.path.display().to_string()),
    }
}

#[cfg(any(target_os = "macos", test))]
fn macos_colmap_path(root: &Path) -> PathBuf {
    let standalone = root.join("colmap").join("bin").join("colmap");
    if standalone.is_file() {
        standalone
    } else {
        root.join("bin").join("colmap")
    }
}

#[cfg(all(not(windows), not(target_os = "macos")))]
fn resolve_engine(env_name: &str, managed: &[PathBuf], executable_name: &str) -> PathBuf {
    if let Some(path) = std::env::var_os(env_name) {
        return path.into();
    }
    managed
        .iter()
        .find(|path| path.is_file())
        .cloned()
        .or_else(|| find_on_path(executable_name))
        .unwrap_or_else(|| managed[0].clone())
}

#[cfg(all(not(windows), not(target_os = "macos")))]
fn find_on_path(executable_name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|directory| directory.join(executable_name))
        .find(|candidate| candidate.is_file())
}

fn missing(kind: ToolKind, path: &Path) -> ToolStatus {
    ToolStatus {
        kind,
        path: path.to_path_buf(),
        exists: false,
        can_start: false,
        version: None,
        cpu_only: None,
        acceleration: None,
        detail: format!("未找到 {}", path.display()),
    }
}

async fn check_basic(kind: ToolKind, path: &Path, args: &[&str]) -> ToolStatus {
    if !path.is_file() {
        return missing(kind, path);
    }
    let manager = ProcessManager::new();
    let result = manager
        .run(ProcessSpec {
            executable: path.to_path_buf(),
            args: args.iter().map(OsString::from).collect(),
            working_directory: path.parent().map(Path::to_path_buf),
            log_path: None,
            observer: None,
        })
        .await;

    match result {
        Ok(output) => {
            let combined = format!("{}\n{}", output.stdout, output.stderr);
            let first_line = combined
                .lines()
                .find(|line| !line.trim().is_empty())
                .map(|line| line.trim().to_owned());
            ToolStatus {
                kind,
                path: path.to_path_buf(),
                exists: true,
                can_start: output.success,
                version: first_line,
                cpu_only: None,
                acceleration: None,
                    detail: if output.success {
                    "引擎可启动".into()
                } else {
                    format!("帮助命令退出码：{:?}", output.exit_code)
                },
            }
        }
        Err(error) => ToolStatus {
            kind,
            path: path.to_path_buf(),
            exists: true,
            can_start: false,
            version: None,
            cpu_only: None,
            acceleration: None,
            detail: error.to_string(),
        },
    }
}

async fn check_brush(path: &Path) -> ToolStatus {
    let mut status = check_basic(ToolKind::Training, path, &["--version"]).await;
    if !status.can_start {
        return status;
    }
    if status.version.as_deref() != Some("brush-cli 1.0.0") {
        status.can_start = false;
        status.detail = "需要 OOOBrush ooo-v1.0.0 的无界面 brush-cli；请重新准备 Brush".into();
        return status;
    }
    let manager = ProcessManager::new();
    match manager
        .run(ProcessSpec {
            executable: path.to_path_buf(),
            args: vec![OsString::from("--help")],
            working_directory: path.parent().map(Path::to_path_buf),
            log_path: None,
            observer: None,
        })
        .await
    {
        Ok(output) => {
            let help = format!("{}\n{}", output.stdout, output.stderr);
            let missing: Vec<_> = super::brush::REQUIRED_CLI_FLAGS
                .iter()
                .filter(|flag| !help.split_whitespace().any(|token| token == **flag))
                .copied()
                .collect();
            status.can_start = output.success && missing.is_empty();
            status.detail = if status.can_start {
                "OOOBrush ooo-v1.0.0 无界面 CLI 可启动；GPU 训练将在任务中验证".into()
            } else {
                format!(
                    "OOOBrush CLI 参数检查失败，缺失参数：{}，退出码：{:?}",
                    missing.join(", "),
                    output.exit_code
                )
            };
        }
        Err(error) => {
            status.can_start = false;
            status.detail = error.to_string();
        }
    }
    status
}

async fn check_colmap(path: &Path, engines_root: &Path, probe_policy: ProbePolicy) -> ToolStatus {
    // 自研后端（`feature = "backend-ooo-sfm"`）：进程内实现，没有可执行文件要探测 ——
    // 也不能因为「路径下没有 colmap 二进制」就判成引擎缺失（macOS 上自研后端
    // 本来就不带 COLMAP）。能力与加速状态由后端自己声明。
    #[cfg(feature = "backend-ooo-sfm")]
    {
        let _ = (engines_root, probe_policy);
        let component = super::colmap::health::component_status(path.to_path_buf());
        return ToolStatus {
            kind: ToolKind::Reconstruction,
            path: path.to_path_buf(),
            exists: true,
            can_start: true,
            version: Some(super::colmap::LOCKED_VERSION.to_string()),
            cpu_only: component
                .acceleration
                .as_ref()
                .map(|acceleration| !acceleration.accelerated()),
            acceleration: component.acceleration,
            detail: component.detail,
        };
    }
    #[cfg(not(feature = "backend-ooo-sfm"))]
    if !path.is_file() {
        let mut status = missing(ToolKind::Reconstruction, path);
        status.acceleration = Some(cpu_status(
            AccelerationReason::ComponentUnavailable,
            format!("未找到 COLMAP：{}", path.display()),
            None,
            requirements_or_default(engines_root),
        ));
        return status;
    }
    let manager = ProcessManager::new();
    let mut help = String::new();
    let mut feature_help = String::new();
    let mut matching_help = String::new();
    let mut successful = true;
    for args in [
        vec!["feature_extractor", "-h"],
        vec!["sequential_matcher", "-h"],
        vec!["exhaustive_matcher", "-h"],
        vec!["matches_importer", "-h"],
        vec!["mapper", "-h"],
        vec!["bundle_adjuster", "-h"],
    ] {
        let command_name = args[0];
        match manager
            .run(ProcessSpec {
                executable: path.to_path_buf(),
                args: args.into_iter().map(OsString::from).collect(),
                working_directory: path.parent().map(Path::to_path_buf),
                log_path: None,
                observer: None,
            })
            .await
        {
            Ok(output) => {
                let command_help = format!("{}\n{}", output.stdout, output.stderr);
                successful &= output.success
                    && colmap_command_has_required_options(command_name, &command_help);
                match command_name {
                    "feature_extractor" => feature_help = command_help.clone(),
                    "sequential_matcher" => matching_help = command_help.clone(),
                    _ => {}
                }
                help.push_str(&command_help);
            }
            Err(error) => {
                return ToolStatus {
                    kind: ToolKind::Reconstruction,
                    path: path.to_path_buf(),
                    exists: true,
                    can_start: false,
                    version: None,
                    cpu_only: None,
                    acceleration: Some(cpu_status(
                        AccelerationReason::ComponentUnavailable,
                        format!("COLMAP 无法启动：{error}"),
                        None,
                        requirements_or_default(engines_root),
                    )),
                            detail: error.to_string(),
                }
            }
        }
    }

    let cli_family = detect_cli_family(&feature_help, &matching_help);
    let locked_build = super::colmap::is_locked_build(&feature_help);
    successful &= colmap_identity_accepted(locked_build);
    successful &= cli_family.is_some();
    #[cfg(target_os = "macos")]
    let cpu_only = Some(true);
    #[cfg(not(target_os = "macos"))]
    let cpu_only = {
        let lower = help.to_ascii_lowercase();
        let explicit_cpu = [
            "cuda: no",
            "cuda support: no",
            "without cuda",
            "no cuda support",
        ]
        .iter()
        .any(|marker| lower.contains(marker));
        let bundled_cuda =
            lower.contains("with cuda") || path.parent().is_some_and(runtime_contains_cuda);
        if bundled_cuda {
            Some(false)
        } else if explicit_cpu {
            Some(true)
        } else {
            None
        }
    };
    let first_line = help
        .lines()
        .find(|line| !line.trim().is_empty())
        .map(|line| line.trim().to_owned());
    let unavailable_reason = if cli_family.is_none() {
        "COLMAP 缺少 OOOSplat 所需的 SIFT 提取或匹配参数".into()
    } else if !colmap_identity_accepted(locked_build) {
        "COLMAP 版本/commit 不匹配；请安装锁定的 COLMAP 4.2.1 引擎包".into()
    } else {
        "COLMAP 必需命令无法正常启动".into()
    };
    let acceleration = if !successful {
        cpu_status(
            AccelerationReason::ComponentUnavailable,
            unavailable_reason,
            None,
            requirements_or_default(engines_root),
        )
    } else if cfg!(target_os = "macos") {
        cpu_status(
            AccelerationReason::CpuOnly,
            "macOS Alpha 当前内置 COLMAP CPU 构建；Brush 仍会使用可用的 Metal 后端".into(),
            None,
            requirements_or_default(engines_root),
        )
    } else if cpu_only != Some(false) {
        cpu_status(
            AccelerationReason::RequirementsUnavailable,
            "内置 COLMAP 未检测到完整 CUDA 运行时，已使用 CPU".into(),
            None,
            requirements_or_default(engines_root),
        )
    } else {
        probe_acceleration(engines_root, probe_policy).await
    };
    let family_label = cli_family.map_or("不支持的 CLI", ColmapCliFamily::label);
    #[cfg(target_os = "macos")]
    let detail = format!("三个必需命令可启动；{family_label}；macOS arm64 CPU-only 构建");
    #[cfg(not(target_os = "macos"))]
    let detail = match cpu_only {
        Some(true) => {
            format!("三个必需命令可启动；{family_label}；帮助输出明确报告无 CUDA")
        }
        Some(false) => format!("{family_label}；{}", acceleration.reason),
        None => format!("三个必需命令可启动；{family_label}；未明确报告 CUDA 构建状态"),
    };
    let ba = super::colmap_ba::select(path, acceleration.gpu_index()).await;
    let detail = if !locked_build && cfg!(feature = "local-colmap") {
        format!(
            "{detail}；本地开发模式接受能力兼容的非锁定 COLMAP；{}",
            ba.detail
        )
    } else if !locked_build {
        "COLMAP 版本/commit 不匹配；请安装锁定的 COLMAP 4.2.1 引擎包".into()
    } else {
        format!("{detail}；{}", ba.detail)
    };
    ToolStatus {
        kind: ToolKind::Reconstruction,
        path: path.to_path_buf(),
        exists: true,
        can_start: successful,
        version: first_line,
        cpu_only,
        acceleration: Some(acceleration),
        detail,
    }
}

fn colmap_identity_accepted(locked_build: bool) -> bool {
    locked_build || cfg!(feature = "local-colmap")
}

fn colmap_command_has_required_options(command: &str, help: &str) -> bool {
    let required: &[&str] = match command {
        "feature_extractor" => &[
            "--database_path",
            "--image_path",
            "--image_list_path",
            "--ImageReader.camera_model",
            "--ImageReader.mask_path",
            "--ImageReader.existing_camera_id",
        ],
        "sequential_matcher" => &["--database_path", "--SequentialMatching.overlap"],
        "exhaustive_matcher" => &["--database_path"],
        "matches_importer" => &["--database_path", "--match_list_path", "--match_type"],
        "mapper" => &[
            "--database_path",
            "--image_path",
            "--input_path",
            "--output_path",
            "--Mapper.image_list_path",
            "--Mapper.fix_existing_frames",
        ],
        "bundle_adjuster" => &["--input_path", "--output_path"],
        _ => return false,
    };
    required.iter().all(|option| help.contains(option))
}

#[cfg(any(not(target_os = "macos"), test))]
fn runtime_contains_cuda(directory: &Path) -> bool {
    let mut found = [false; 2];
    scan_cuda_runtime(directory, &mut found);
    found.into_iter().all(|present| present)
}

#[cfg(any(not(target_os = "macos"), test))]
fn scan_cuda_runtime(directory: &Path, found: &mut [bool; 2]) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() && !path.is_symlink() {
            scan_cuda_runtime(&path, found);
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        found[0] |= name.contains("cudart64_") || name.starts_with("libcudart.so");
        found[1] |= name.contains("curand64_") || name.starts_with("libcurand.so");
    }
}

/// 加速状态探测（强制刷新）。
pub async fn refresh_acceleration(paths: &EnginePaths) -> AccelerationStatus {
    check_colmap(&paths.colmap, &paths.root, ProbePolicy::Refresh)
        .await
        .acceleration
        .unwrap_or_else(|| {
            cpu_status(
                AccelerationReason::ComponentUnavailable,
                "无法读取 COLMAP 加速状态，已使用 CPU".into(),
                None,
                requirements_or_default(&paths.root),
            )
        })
}

/// 加速状态探测（带缓存）。
pub async fn detect_acceleration(paths: &EnginePaths) -> AccelerationStatus {
    check_colmap(&paths.colmap, &paths.root, ProbePolicy::Cached)
        .await
        .acceleration
        .unwrap_or_else(|| {
            cpu_status(
                AccelerationReason::ComponentUnavailable,
                "无法读取 COLMAP 加速状态，已使用 CPU".into(),
                None,
                requirements_or_default(&paths.root),
            )
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct NumericVersion(u32, u32);

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EngineManifest {
    #[serde(default)]
    engines: Vec<ManifestEngine>,
    colmap: Option<ManifestEngine>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ManifestEngine {
    name: String,
    cuda_compatibility: Option<ManifestCudaCompatibility>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ManifestCudaCompatibility {
    #[serde(alias = "minimumDriver")]
    minimum_windows_driver: String,
    minimum_compute_capability: String,
}

#[derive(Debug, Clone, Copy)]
enum ProbeError {
    NotFound,
    Failed,
    Timeout,
    NoGpu,
    InvalidOutput,
}

impl ProbeError {
    const fn is_transient(self) -> bool {
        matches!(self, Self::Failed | Self::Timeout | Self::InvalidOutput)
    }
}

fn default_requirements() -> AcceleratorRequirements {
    AcceleratorRequirements {
        minimum_driver_version: DEFAULT_MINIMUM_DRIVER.into(),
        minimum_compute_capability: DEFAULT_MINIMUM_COMPUTE_CAPABILITY.into(),
    }
}

fn requirements_or_default(engines_root: &Path) -> AcceleratorRequirements {
    load_requirements(engines_root).unwrap_or_else(|_| default_requirements())
}

fn load_requirements(engines_root: &Path) -> std::result::Result<AcceleratorRequirements, String> {
    let path = engines_root.join(if cfg!(target_os = "linux") {
        "manifest.linux.json"
    } else {
        "manifest.json"
    });
    let bytes =
        std::fs::read(&path).map_err(|error| format!("无法读取 {}：{error}", path.display()))?;
    let manifest: EngineManifest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("无法解析 {}：{error}", path.display()))?;
    let compatibility = manifest
        .engines
        .into_iter()
        .chain(manifest.colmap)
        .find(|engine| engine.name.eq_ignore_ascii_case("COLMAP"))
        .and_then(|engine| engine.cuda_compatibility)
        .ok_or_else(|| "引擎清单缺少 COLMAP cudaCompatibility".to_string())?;
    parse_version(&compatibility.minimum_windows_driver)
        .ok_or_else(|| "引擎清单中的最低驱动版本无效".to_string())?;
    parse_version(&compatibility.minimum_compute_capability)
        .ok_or_else(|| "引擎清单中的最低 Compute Capability 无效".to_string())?;
    Ok(AcceleratorRequirements {
        minimum_driver_version: compatibility.minimum_windows_driver,
        minimum_compute_capability: compatibility.minimum_compute_capability,
    })
}

fn cpu_status(
    reason_code: AccelerationReason,
    reason: String,
    device: Option<AcceleratorDevice>,
    requirements: AcceleratorRequirements,
) -> AccelerationStatus {
    AccelerationStatus {
        mode: AccelerationMode::Cpu,
        state: AccelerationState::Unavailable,
        reason_code,
        reason,
        device,
        requirements,
        detected_device_count: 0,
    }
}

async fn probe_acceleration(engines_root: &Path, policy: ProbePolicy) -> AccelerationStatus {
    let requirements = match load_requirements(engines_root) {
        Ok(requirements) => requirements,
        Err(error) => {
            return cpu_status(
                AccelerationReason::RequirementsUnavailable,
                format!("{error}，已为兼容性使用 CPU"),
                None,
                default_requirements(),
            )
        }
    };
    acceleration_probe_service()
        .probe(requirements, policy)
        .await
}

impl AccelerationProbeService {
    async fn probe(
        &self,
        requirements: AcceleratorRequirements,
        policy: ProbePolicy,
    ) -> AccelerationStatus {
        if policy == ProbePolicy::Cached {
            if let Some(status) = self.fresh_status().await {
                return status;
            }
        }

        // Serialize nvidia-smi calls. A white-screen recovery and a source
        // analysis can otherwise start several copies at the same time. A
        // planning-only request never waits behind a slow startup probe.
        let _run_guard = if policy == ProbePolicy::Cached {
            match self.run_lock.try_lock() {
                Ok(guard) => guard,
                Err(_) => return transient_probe_status(ProbeError::Failed, requirements, None),
            }
        } else {
            self.run_lock.lock().await
        };
        if policy == ProbePolicy::Cached {
            if let Some(status) = self.fresh_status().await {
                return status;
            }
        }

        let attempts = if policy == ProbePolicy::Refresh { 1 } else { 3 };
        let mut last_error = ProbeError::Failed;
        for attempt in 0..attempts {
            match probe_gpu_devices().await {
                Ok(devices) => {
                    let status = choose_acceleration(devices, requirements.clone());
                    self.state.lock().await.last_good = Some((Instant::now(), status.clone()));
                    append_acceleration_diagnostic(&format!(
                        "attempt={} outcome=success backend={:?} devices={}",
                        attempt + 1,
                        status.mode,
                        status.detected_device_count
                    ));
                    return status;
                }
                Err(error) => {
                    last_error = error;
                    append_acceleration_diagnostic(&format!(
                        "attempt={} outcome={error:?}",
                        attempt + 1
                    ));
                    if !error.is_transient() {
                        return probe_error_status(error, requirements);
                    }
                    if attempt + 1 < attempts {
                        tokio::time::sleep(Duration::from_secs((attempt + 1) as u64)).await;
                    }
                }
            }
        }

        let cached = if policy == ProbePolicy::Task {
            None
        } else {
            self.fresh_status().await
        };
        transient_probe_status(last_error, requirements, cached)
    }

    async fn fresh_status(&self) -> Option<AccelerationStatus> {
        self.state
            .lock()
            .await
            .last_good
            .as_ref()
            .filter(|(recorded, _)| recorded.elapsed() <= ACCELERATION_CACHE_TTL)
            .map(|(_, status)| status.clone())
    }
}

fn transient_probe_status(
    error: ProbeError,
    requirements: AcceleratorRequirements,
    cached: Option<AccelerationStatus>,
) -> AccelerationStatus {
    let fallback = probe_error_status(error, requirements);
    if let Some(mut status) = cached {
        status.state = AccelerationState::TemporarilyUnavailable;
        status.reason_code = fallback.reason_code;
        status.reason = format!(
            "NVIDIA 显卡暂时无法检测；继续显示本次运行中最近一次有效结果。{}",
            fallback.reason
        );
        status
    } else {
        fallback
    }
}

fn probe_error_status(
    error: ProbeError,
    requirements: AcceleratorRequirements,
) -> AccelerationStatus {
    let (reason_code, reason) = match error {
        ProbeError::NotFound => (
            AccelerationReason::NoAccelerator,
            "未检测到 NVIDIA 驱动，已使用 CPU",
        ),
        ProbeError::Timeout => (
            AccelerationReason::ProbeTimeout,
            "NVIDIA 显卡检测超时，已使用 CPU",
        ),
        ProbeError::NoGpu => (
            AccelerationReason::NoAccelerator,
            "未检测到 NVIDIA 显卡，已使用 CPU",
        ),
        ProbeError::InvalidOutput => (
            AccelerationReason::ComputeCapabilityUnknown,
            "无法读取显卡驱动或 Compute Capability，已使用 CPU",
        ),
        ProbeError::Failed => (
            AccelerationReason::ProbeFailed,
            "NVIDIA 显卡检测失败，已使用 CPU",
        ),
    };
    let mut status = cpu_status(reason_code, reason.into(), None, requirements);
    if error.is_transient() {
        status.state = AccelerationState::TemporarilyUnavailable;
    }
    status
}

fn choose_acceleration(
    devices: Vec<AcceleratorDevice>,
    requirements: AcceleratorRequirements,
) -> AccelerationStatus {
    let detected_device_count = devices.len();
    let minimum_driver = parse_version(&requirements.minimum_driver_version)
        .expect("validated acceleration requirement");
    let minimum_compute = parse_version(&requirements.minimum_compute_capability)
        .expect("validated acceleration requirement");

    let mut compatible = devices
        .iter()
        .filter_map(|device| {
            let driver = parse_version(&device.driver_version)?;
            let compute = parse_version(&device.compute_capability)?;
            (driver >= minimum_driver && compute >= minimum_compute).then_some((
                compute,
                device.index,
                device.clone(),
            ))
        })
        .collect::<Vec<_>>();
    compatible.sort_by(|left, right| right.0.cmp(&left.0).then(left.1.cmp(&right.1)));
    if let Some((_, _, device)) = compatible.into_iter().next() {
        return AccelerationStatus {
            mode: AccelerationMode::Gpu,
            state: AccelerationState::Ready,
            reason_code: AccelerationReason::GpuReady,
            reason: format!(
                "已启用 {}（驱动 {}，Compute Capability {}）",
                device.name, device.driver_version, device.compute_capability
            ),
            device: Some(device),
            requirements,
            detected_device_count,
        };
    }

    let driver_eligible = devices.iter().filter(|device| {
        parse_version(&device.driver_version).is_some_and(|version| version >= minimum_driver)
    });
    let mut low_compute = driver_eligible
        .clone()
        .filter_map(|device| {
            parse_version(&device.compute_capability)
                .filter(|version| *version < minimum_compute)
                .map(|version| (version, device.clone()))
        })
        .collect::<Vec<_>>();
    low_compute.sort_by_key(|item| std::cmp::Reverse(item.0));
    if let Some((_, device)) = low_compute.into_iter().next() {
        return cpu_status(
            AccelerationReason::ComputeCapabilityTooLow,
            format!(
                "{} 的 Compute Capability {} 低于最低要求 {}，已使用 CPU",
                device.name, device.compute_capability, requirements.minimum_compute_capability
            ),
            Some(device),
            requirements,
        );
    }
    if let Some(device) = devices.iter().find(|device| {
        parse_version(&device.driver_version).is_some_and(|version| version >= minimum_driver)
            && parse_version(&device.compute_capability).is_none()
    }) {
        return cpu_status(
            AccelerationReason::ComputeCapabilityUnknown,
            format!("无法读取 {} 的 Compute Capability，已使用 CPU", device.name),
            Some(device.clone()),
            requirements,
        );
    }

    let mut old_driver = devices
        .iter()
        .filter_map(|device| parse_version(&device.driver_version).map(|v| (v, device.clone())))
        .collect::<Vec<_>>();
    old_driver.sort_by_key(|item| std::cmp::Reverse(item.0));
    if let Some((_, device)) = old_driver.into_iter().next() {
        return cpu_status(
            AccelerationReason::DriverTooOld,
            format!(
                "NVIDIA 驱动 {} 低于最低要求 {}，已使用 CPU",
                device.driver_version, requirements.minimum_driver_version
            ),
            Some(device),
            requirements,
        );
    }
    let device = devices.into_iter().next();
    cpu_status(
        AccelerationReason::DriverVersionUnknown,
        "无法读取 NVIDIA 驱动版本，已使用 CPU".into(),
        device,
        requirements,
    )
}

pub(crate) async fn diagnostic_gpu_devices() -> Vec<AcceleratorDevice> {
    probe_gpu_devices().await.unwrap_or_default()
}

async fn probe_gpu_devices() -> std::result::Result<Vec<AcceleratorDevice>, ProbeError> {
    let candidate = nvidia_smi_candidates()
        .into_iter()
        .find(|candidate| candidate.is_file());
    let Some(candidate) = candidate else {
        append_acceleration_diagnostic("outcome=NotFound path=nvidia-smi");
        return Err(ProbeError::NotFound);
    };
    let manager = ProcessManager::new();
    let run = manager.run(ProcessSpec {
        executable: candidate.clone(),
        args: vec![
            OsString::from("--query-gpu=index,name,driver_version,compute_cap,memory.total"),
            OsString::from("--format=csv,noheader,nounits"),
        ],
        working_directory: None,
        log_path: None,
        observer: None,
    });
    tokio::pin!(run);
    let result = match tokio::time::timeout(NVIDIA_SMI_TIMEOUT, &mut run).await {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => {
            append_acceleration_diagnostic(&format!(
                "path={} process_error={}",
                candidate.display(),
                bounded_diagnostic(&error.to_string())
            ));
            return Err(ProbeError::Failed);
        }
        Err(_) => {
            manager.cancel();
            let _ = run.await;
            append_acceleration_diagnostic(&format!(
                "path={} timeout_ms={}",
                candidate.display(),
                NVIDIA_SMI_TIMEOUT.as_millis()
            ));
            return Err(ProbeError::Timeout);
        }
    };
    append_acceleration_diagnostic(&format!(
        "path={} exit={:?} stdout={} stderr={}",
        candidate.display(),
        result.exit_code,
        bounded_diagnostic(&result.stdout),
        bounded_diagnostic(&result.stderr)
    ));
    if !result.success {
        return Err(ProbeError::Failed);
    }
    parse_nvidia_smi_csv(&result.stdout)
}

fn bounded_diagnostic(value: &str) -> String {
    let normalized = value.replace(['\r', '\n'], " ");
    if normalized.len() <= ACCELERATION_OUTPUT_LIMIT {
        return normalized;
    }
    let mut end = ACCELERATION_OUTPUT_LIMIT;
    while !normalized.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &normalized[..end])
}

fn acceleration_diagnostic_path() -> Option<PathBuf> {
    #[cfg(windows)]
    let root = std::env::var_os("LOCALAPPDATA").map(PathBuf::from);
    #[cfg(target_os = "macos")]
    let root = std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join("Library").join("Logs"));
    #[cfg(all(not(windows), not(target_os = "macos")))]
    let root = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".local").join("state"))
        });
    root.map(|root| root.join("SplatStudio").join("engine-health.log"))
}

fn append_acceleration_diagnostic(message: &str) {
    use std::io::Write;

    let Some(path) = acceleration_diagnostic_path() else {
        return;
    };
    let Some(parent) = path.parent() else {
        return;
    };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    if path
        .metadata()
        .is_ok_and(|metadata| metadata.len() >= ACCELERATION_DIAGNOSTIC_LIMIT)
    {
        let _ = std::fs::write(&path, []);
    }
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(file, "{timestamp} {message}");
    }
}

fn parse_nvidia_smi_csv(output: &str) -> std::result::Result<Vec<AcceleratorDevice>, ProbeError> {
    let mut devices = Vec::new();
    for line in output.lines().filter(|line| !line.trim().is_empty()) {
        let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
        if fields.len() < 5 {
            return Err(ProbeError::InvalidOutput);
        }
        let index = fields[0]
            .parse::<u32>()
            .map_err(|_| ProbeError::InvalidOutput)?;
        devices.push(AcceleratorDevice {
            index,
            name: fields[1..fields.len() - 3].join(", "),
            driver_version: fields[fields.len() - 3].to_string(),
            compute_capability: fields[fields.len() - 2].to_string(),
            total_memory_mb: fields[fields.len() - 1].parse().ok(),
        });
    }
    if devices.is_empty() {
        Err(ProbeError::NoGpu)
    } else {
        Ok(devices)
    }
}

fn parse_version(value: &str) -> Option<NumericVersion> {
    let mut parts = value.trim().split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    Some(NumericVersion(major, minor))
}

pub(crate) fn nvidia_smi_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    #[cfg(windows)]
    match std::env::var_os("SystemRoot") {
        Some(root) => candidates.push(PathBuf::from(root).join("System32").join("nvidia-smi.exe")),
        None => candidates.push(PathBuf::from(r"C:\Windows\System32\nvidia-smi.exe")),
    }
    #[cfg(windows)]
    let executable_name = "nvidia-smi.exe";
    #[cfg(not(windows))]
    let executable_name = "nvidia-smi";
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            candidates.push(dir.join(executable_name));
        }
    }
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    fn requirements() -> AcceleratorRequirements {
        default_requirements()
    }

    /// 自研后端（`feature = "backend-ooo-sfm"`）：**没有 COLMAP 二进制也必须报就绪**。
    ///
    /// 这条守的是真实卡点：`runner.rs::verify_pipeline_engines` 会遍历
    /// `check_all_for_task()` 的结果，只要 COLMAP 的 `exists == false` 就直接
    /// `EngineMissing` —— 而自研后端是进程内的，路径下本来就不该有二进制。
    #[cfg(feature = "backend-ooo-sfm")]
    #[tokio::test]
    async fn self_hosted_backend_needs_no_colmap_binary() {
        let missing = std::path::PathBuf::from("/definitely/not/here/colmap");
        let status = check_colmap(&missing, Path::new("/definitely/not/here"), ProbePolicy::Task).await;
        assert!(status.exists, "自研后端不应被判为缺引擎：{status:?}");
        assert!(status.can_start, "自研后端应可启动：{status:?}");
        assert!(status.exists && status.can_start, "自研后端应就绪：{status:?}");
        assert!(status.acceleration.is_some(), "自研后端应上报加速状态");
        assert!(status.detail.contains("自研后端"), "{}", status.detail);
        // 加速状态也必须存在（前端零改动依赖它），且与 Metal feature 一致。
        let acceleration = status.acceleration.expect("自研后端必须上报加速状态");
        assert_eq!(
            acceleration.use_gpu(),
            crate::engines::colmap::health::metal_active()
        );
    }

    /// 进程内媒体（`media-rsmpeg`）：**没有 ffmpeg/ffprobe 二进制也必须报就绪**。
    ///
    /// 对称于 `self_hosted_backend_needs_no_colmap_binary`（重建组件的进程内分支）。
    /// 缺这条分支的真实事故：`OOO_MEDIA_RSMPEG=1` 的产品包把两个二进制裁掉了，
    /// 媒体组件永远 `ready=false` → 顶栏常驻"N 个引擎异常"、前端 `canStart`
    /// 因 missingComponents 非空被禁用，任务侧也会 EngineMissing。
    #[cfg(feature = "media-rsmpeg")]
    #[tokio::test]
    async fn in_process_media_needs_no_cli_binaries() {
        let paths = EnginePaths::from_candidates(PathBuf::from(
            "/definitely/not/here/macos/arm64",
        ));
        assert!(!paths.ffmpeg.is_file());
        assert!(!paths.ffprobe.is_file());
        let statuses = paths.check_all().await;
        let media = statuses
            .iter()
            .find(|status| status.kind == ComponentKind::Media)
            .expect("媒体组件必须存在");
        assert!(media.ready, "进程内媒体不应要求 CLI 二进制：{media:?}");
        assert!(media.detail.contains("进程内"), "{}", media.detail);
    }

    /// 反向保证：不带 feature 时 COLMAP 缺失仍然要判缺（主分支行为不变）。
    #[cfg(not(feature = "backend-ooo-sfm"))]
    #[tokio::test]
    async fn missing_colmap_binary_is_still_reported() {
        let status = check_colmap(
            std::path::Path::new("/definitely/not/here/colmap"),
            Path::new("/definitely/not/here"),
            ProbePolicy::Task,
        )
        .await;
        assert!(!status.exists);
    }

    #[cfg(all(not(windows), not(target_os = "macos")))]
    #[test]
    fn finds_executable_on_path() {
        assert!(find_on_path("cargo").is_some());
    }

    #[cfg(all(not(windows), not(target_os = "macos")))]
    #[test]
    fn linux_colmap_is_managed_without_path_fallback() {
        let paths = EnginePaths::from_root("/opt/ooosplat-engines");
        assert_eq!(
            paths.colmap,
            PathBuf::from("/opt/ooosplat-engines/linux/colmap/bin/colmap")
        );
        let discovered = EnginePaths::from_candidates(PathBuf::from("/missing/engines"));
        assert_eq!(
            discovered.colmap,
            PathBuf::from("/missing/engines/linux/colmap/bin/colmap")
        );
        // FFmpeg is not installed on every contributor machine or CI runner, so assert
        // the resolver contract instead of requiring the binary: an explicit override
        // wins, then PATH, and otherwise the managed path is kept so engine health can
        // report the exact file it expected.
        let expected = std::env::var_os("OOOSPLAT_FFMPEG")
            .map(PathBuf::from)
            .or_else(|| find_on_path("ffmpeg"))
            .unwrap_or_else(|| PathBuf::from("/missing/engines/ffmpeg"));
        assert_eq!(discovered.ffmpeg, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_candidates_stay_inside_the_managed_arm64_root() {
        let root = PathBuf::from("/missing/ooosplat-engines/macos/arm64");
        let paths = EnginePaths::from_candidates(root.clone());
        assert_eq!(paths.ffmpeg, root.join("bin").join("ffmpeg"));
        assert_eq!(paths.ffprobe, root.join("bin").join("ffprobe"));
        assert_eq!(paths.colmap, root.join("bin").join("colmap"));
        assert_eq!(paths.brush, root.join("bin").join("brush_app"));
    }

    #[cfg(windows)]
    #[test]
    fn windows_candidates_stay_inside_the_managed_root() {
        let root = PathBuf::from(r"Z:\missing\ooosplat-engines");
        let paths = EnginePaths::from_candidates(root.clone());
        assert_eq!(paths.ffmpeg, root.join("ffmpeg").join("ffmpeg.exe"));
        assert_eq!(paths.ffprobe, root.join("ffmpeg").join("ffprobe.exe"));
        assert_eq!(
            paths.colmap,
            root.join("colmap").join("bin").join("colmap.exe")
        );
        assert_eq!(paths.brush, root.join("brush").join("brush_app.exe"));
    }

    #[tokio::test]
    #[ignore = "Requires npm run setup:brush on a matching native host"]
    async fn managed_ooobrush_cli_health_smoke() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("engines");
        #[cfg(target_os = "macos")]
        let root = root.join("macos").join("arm64");
        let paths = EnginePaths::from_root(root);
        let status = check_brush(&paths.brush).await;
        assert!(
            status.can_start,
            "{}: {}",
            paths.brush.display(),
            status.detail
        );
        assert_eq!(status.version.as_deref(), Some("brush-cli 1.0.0"));
    }

    fn device(index: u32, driver: &str, compute: &str) -> AcceleratorDevice {
        AcceleratorDevice {
            index,
            name: format!("GPU {index}"),
            driver_version: driver.into(),
            compute_capability: compute.into(),
            total_memory_mb: Some(8_192),
        }
    }

    #[test]
    fn parses_nvidia_smi_csv() {
        let devices = parse_nvidia_smi_csv(
            "0, NVIDIA GeForce RTX 3060 Ti, 560.81, 8.6, 8192\n1, NVIDIA RTX 4090, 560.81, 8.9, 24564\n",
        )
        .unwrap();
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].name, "NVIDIA GeForce RTX 3060 Ti");
        assert_eq!(devices[1].compute_capability, "8.9");
        assert_eq!(devices[0].total_memory_mb, Some(8_192));
    }

    #[test]
    fn unavailable_memory_does_not_hide_a_compatible_gpu() {
        let devices =
            parse_nvidia_smi_csv("0, NVIDIA GeForce RTX 3060 Ti, 560.81, 8.6, N/A\n").unwrap();
        assert_eq!(devices[0].total_memory_mb, None);
        let devices = devices
            .into_iter()
            .map(|mut device| {
                device.driver_version = DEFAULT_MINIMUM_DRIVER.into();
                device
            })
            .collect();
        assert_eq!(
            choose_acceleration(devices, requirements()).mode,
            AccelerationMode::Gpu
        );
    }

    #[test]
    fn rejects_malformed_nvidia_smi_output() {
        assert!(matches!(
            parse_nvidia_smi_csv("not a gpu row"),
            Err(ProbeError::InvalidOutput)
        ));
        assert!(matches!(parse_nvidia_smi_csv("\n"), Err(ProbeError::NoGpu)));
    }

    #[test]
    fn accepts_exact_compatibility_boundaries() {
        let status = choose_acceleration(vec![device(0, "580.00", "7.5")], requirements());
        assert_eq!(status.mode, AccelerationMode::Gpu);
        assert_eq!(status.reason_code, AccelerationReason::GpuReady);
        assert_eq!(status.usable_gpu_total_memory_mb(), Some(8_192));
        assert_eq!(status.detected_device_count, 1);
    }

    #[test]
    fn rejects_old_driver_and_low_compute_capability() {
        let old_driver = choose_acceleration(vec![device(0, "579.99", "8.6")], requirements());
        assert_eq!(old_driver.mode, AccelerationMode::Cpu);
        assert_eq!(old_driver.reason_code, AccelerationReason::DriverTooOld);
        assert_eq!(
            old_driver.device.as_ref().unwrap().total_memory_mb,
            Some(8_192)
        );
        assert_eq!(old_driver.usable_gpu_total_memory_mb(), None);
        assert_eq!(old_driver.planning_gpu_total_memory_mb(), None);

        let old_gpu = choose_acceleration(vec![device(0, "580.00", "7.4")], requirements());
        assert_eq!(old_gpu.mode, AccelerationMode::Cpu);
        assert_eq!(
            old_gpu.reason_code,
            AccelerationReason::ComputeCapabilityTooLow
        );
    }

    #[test]
    fn selects_highest_compute_capability_then_lowest_index() {
        let status = choose_acceleration(
            vec![
                device(2, "580.00", "8.6"),
                device(1, "580.00", "8.9"),
                device(0, "580.00", "8.9"),
            ],
            requirements(),
        );
        assert_eq!(status.detected_device_count, 3);
        assert_eq!(status.device.unwrap().index, 0);
    }

    #[test]
    fn maps_probe_failures_to_conservative_cpu_reasons() {
        for (error, reason) in [
            (
                ProbeError::NotFound,
                AccelerationReason::NoAccelerator,
            ),
            (ProbeError::Failed, AccelerationReason::ProbeFailed),
            (ProbeError::Timeout, AccelerationReason::ProbeTimeout),
            (ProbeError::NoGpu, AccelerationReason::NoAccelerator),
            (
                ProbeError::InvalidOutput,
                AccelerationReason::ComputeCapabilityUnknown,
            ),
        ] {
            let status = probe_error_status(error, requirements());
            assert_eq!(status.mode, AccelerationMode::Cpu);
            assert_eq!(status.reason_code, reason);
            assert_eq!(
                status.state,
                if error.is_transient() {
                    AccelerationState::TemporarilyUnavailable
                } else {
                    AccelerationState::Unavailable
                }
            );
        }
    }

    #[test]
    fn transient_probe_keeps_a_recent_result_for_planning_only() {
        let cached = choose_acceleration(vec![device(0, "580.00", "8.6")], requirements());
        let status = transient_probe_status(ProbeError::Timeout, requirements(), Some(cached));
        assert_eq!(status.mode, AccelerationMode::Gpu);
        assert_eq!(
            status.state,
            AccelerationState::TemporarilyUnavailable
        );
        assert_eq!(status.planning_gpu_total_memory_mb(), Some(8_192));
    }

    #[tokio::test]
    async fn cached_planning_probe_never_waits_behind_an_active_probe() {
        let service = AccelerationProbeService::default();
        let _active_probe = service.run_lock.lock().await;
        let status = service.probe(requirements(), ProbePolicy::Cached).await;
        assert_eq!(status.mode, AccelerationMode::Cpu);
        assert_eq!(
            status.state,
            AccelerationState::TemporarilyUnavailable
        );
    }

    #[test]
    fn requires_the_complete_locked_cuda_runtime_set() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("cudart64_12.dll"), []).unwrap();
        std::fs::write(directory.path().join("curand64_10.dll"), []).unwrap();
        assert!(runtime_contains_cuda(directory.path()));
    }

    #[test]
    fn macos_prefers_standalone_colmap_without_changing_other_engine_paths() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let legacy = root.join("bin").join("colmap");
        assert_eq!(macos_colmap_path(root), legacy);
        let standalone = root.join("colmap").join("bin").join("colmap");
        std::fs::create_dir_all(standalone.parent().unwrap()).unwrap();
        std::fs::create_dir_all(legacy.parent().unwrap()).unwrap();
        std::fs::write(&legacy, []).unwrap();
        std::fs::write(&standalone, []).unwrap();
        assert_eq!(macos_colmap_path(root), standalone);
        std::fs::remove_file(&standalone).unwrap();
        assert_eq!(macos_colmap_path(root), legacy);
    }

    #[test]
    fn locked_colmap_identity_is_always_accepted() {
        assert!(colmap_identity_accepted(true));
    }

    #[test]
    fn local_colmap_capability_check_rejects_missing_options() {
        let mapper = [
            "--database_path",
            "--image_path",
            "--input_path",
            "--output_path",
            "--Mapper.image_list_path",
            "--Mapper.fix_existing_frames",
        ]
        .join("\n");
        assert!(colmap_command_has_required_options("mapper", &mapper));
        assert!(!colmap_command_has_required_options(
            "mapper",
            &mapper.replace("--Mapper.fix_existing_frames", "")
        ));
        assert!(!colmap_command_has_required_options("unknown", &mapper));
    }

    #[test]
    #[cfg(not(feature = "local-colmap"))]
    fn formal_build_rejects_an_unlocked_colmap_identity() {
        assert!(!colmap_identity_accepted(false));
    }

    #[test]
    #[cfg(feature = "local-colmap")]
    fn local_build_accepts_an_unlocked_colmap_identity() {
        assert!(colmap_identity_accepted(false));
    }
}
