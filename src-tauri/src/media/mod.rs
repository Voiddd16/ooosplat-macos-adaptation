//! 媒体处理（视频探测 + 抽帧）的**后端抽象**。
//!
//! # 为什么有这一层
//!
//! 原来 pipeline 直接 spawn `ffprobe` / `ffmpeg` 二进制。两套实现各有代价：
//!
//! * CLI：多一次进程启动 + 一次完整的解码/封装往返；参数用字符串表达，容易和
//!   上层语义脱节（"采样帧率"、"目标尺寸"、"要不要 alpha mask"）。
//! * 库（rsmpeg）：直接链接 FFmpeg，省掉进程与临时文件；但要求本机 FFmpeg 的
//!   **ABI 与 FFI 绑定的版本严格一致**（见 `media-rsmpeg` feature 的说明）。
//!
//! 现在两种实现都实现同一个 [`MediaTool`] trait，**编译期二选一**：
//!
//! | feature | 实现 | 形态 |
//! |---|---|---|
//! | 都不开 | [`cli::FfmpegCliMedia`] | 外部进程（与主分支行为一致） |
//! | `media-ffmpeg-cli` | 同上（显式） | 外部进程 |
//! | `media-rsmpeg` | [`rsmpeg::RsmpegMedia`] | 进程内（链接 FFmpeg 库） |
//!
//! 同时开两个 → **编译错误**（实现不唯一）。
//!
//! 上层（pipeline / planner）只认 trait：请求结构体是**中性领域概念**
//! （"按多少帧率采样"、"要哪些源帧"、"要不要 alpha 掩膜"），不是 CLI 参数。

use std::path::{Path, PathBuf};

use async_trait::async_trait;

use crate::contracts::BackendIdentity;
use crate::engines::ffmpeg::FrameExtractionResult;
use crate::error::Result;
use crate::process::{ProcessManager, ProcessObserver};
use crate::video::{FramePlan, PlannedFrame, VideoInfo};

pub mod cli;

#[cfg(feature = "media-rsmpeg")]
pub mod rsmpeg;

/// 媒体作业的运行环境（与实现无关）。
pub struct MediaContext<'a> {
    /// 进程托管器：CLI 实现用它拉子进程；库实现用它取取消令牌与日志。
    pub process_manager: &'a ProcessManager,
    /// 进度观察者。
    pub observer: Option<ProcessObserver>,
    /// 日志文件。
    pub log_path: Option<PathBuf>,
}

/// 抽帧请求的公共部分。
#[derive(Debug, Clone)]
pub struct FrameExtraction {
    /// 输入视频。
    pub input: PathBuf,
    /// 抽出的画面目录。
    pub output_directory: PathBuf,
    /// COLMAP 用的画面副本目录（现有 CLI 语义：两份都要有）。
    pub colmap_directory: PathBuf,
    /// alpha 掩膜目录。
    pub mask_directory: PathBuf,
    /// 输入是否带 alpha 通道。
    pub has_alpha: bool,
    /// 目标尺寸（`None` = 按 CLI 默认规则缩放到长边 ≤ 1920）。
    pub target_dimensions: Option<(u32, u32)>,
}

/// 均匀采样（按帧率抽帧）。
#[derive(Debug, Clone)]
pub struct UniformFrameExtraction {
    pub base: FrameExtraction,
    /// 抽帧计划（`sampling_fps` 等）。
    pub plan: FramePlan,
}

/// 指定源帧（Quality v2 / 补拍）。
#[derive(Debug, Clone)]
pub struct SelectedFrameExtraction {
    pub base: FrameExtraction,
    /// 选中的源帧（按源帧序号）。
    pub selected_frames: Vec<PlannedFrame>,
}

/// 媒体处理后端的统一契约。
#[async_trait]
pub trait MediaTool: Send + Sync {
    /// 身份（给界面与遥测看）。
    fn identity(&self) -> BackendIdentity;

    /// 探测视频信息。
    async fn probe_video(
        &self,
        context: &MediaContext<'_>,
        input: &Path,
    ) -> Result<VideoInfo>;

    /// 均匀采样抽帧。
    async fn extract_uniform_frames(
        &self,
        context: &MediaContext<'_>,
        request: UniformFrameExtraction,
    ) -> Result<FrameExtractionResult>;

    /// 指定源帧抽帧。
    async fn extract_selected_frames(
        &self,
        context: &MediaContext<'_>,
        request: SelectedFrameExtraction,
    ) -> Result<FrameExtractionResult>;

    /// 补拍：抽到临时目录后合并进已有画面集。
    async fn extract_additional_frames(
        &self,
        context: &MediaContext<'_>,
        request: SelectedFrameExtraction,
    ) -> Result<FrameExtractionResult>;
}

/// 构造参数（**装配层**的东西，不属于契约）。
#[derive(Debug, Clone, Default)]
pub struct MediaConfig {
    /// `ffmpeg` 可执行文件（CLI 实现用）。
    pub ffmpeg: Option<PathBuf>,
    /// `ffprobe` 可执行文件（CLI 实现用）。
    pub ffprobe: Option<PathBuf>,
}

/// 编译期选定的媒体实现。
pub fn active(config: MediaConfig) -> std::sync::Arc<dyn MediaTool> {
    #[cfg(feature = "media-rsmpeg")]
    {
        // 进程内实现不需要可执行文件路径。
        let _ = config;
        std::sync::Arc::new(rsmpeg::RsmpegMedia::new())
    }
    #[cfg(not(feature = "media-rsmpeg"))]
    {
        std::sync::Arc::new(cli::FfmpegCliMedia::new(config))
    }
}

#[cfg(all(feature = "media-ffmpeg-cli", feature = "media-rsmpeg"))]
compile_error!(
    "媒体实现不唯一：同时启用了 `media-ffmpeg-cli` 与 `media-rsmpeg`。\
     请只选一个（FFmpeg 库：`--features media-rsmpeg`；外部进程：`--features media-ffmpeg-cli`）。"
);

#[cfg(all(
    feature = "media-rsmpeg",
    not(all(target_os = "macos", target_arch = "aarch64"))
))]
compile_error!(
    "`media-rsmpeg` 目前只在 macOS / Apple Silicon 上验证过（FFmpeg 8.x，Homebrew `ffmpeg@8`）；\
     其它平台请使用 `media-ffmpeg-cli`。"
);

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// 契约自检：两种实现都要能自报身份，且形态字段自洽。
    #[test]
    fn active_media_reports_identity() {
        let media = active(MediaConfig::default());
        let identity = media.identity();
        assert!(!identity.id.is_empty());
        assert!(!identity.display_name.is_empty());
        let json = serde_json::to_string(&identity).expect("序列化");
        assert!(json.contains("\"family\""), "{json}");
    }

    // ── 两种实现共用的期望值（互相对拍的基准）──────────────────────────
    const EXPECTED_WIDTH: u32 = 320;
    const EXPECTED_HEIGHT: u32 = 240;
    const EXPECTED_FPS: f64 = 10.0;
    const EXPECTED_FRAMES: u64 = 10;

    /// 生成一段 1 秒、320×240、10fps 的测试视频（用 ffmpeg CLI 造夹具）。
    ///
    /// 没有 ffmpeg 可执行文件时返回 `None` → 相关测试跳过（CI 的精简镜像可能没有）。
    fn fixture_video(directory: &Path) -> Option<PathBuf> {
        let Some(ffmpeg) = which("ffmpeg") else {
            return None;
        };
        let output = std::process::Command::new(&ffmpeg)
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=320x240:rate=10",
                "-pix_fmt",
                "yuv420p",
            ])
            .arg(directory.join("fixture.mp4"))
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        Some(directory.join("fixture.mp4"))
    }

    fn temp_dir() -> Option<tempfile::TempDir> {
        Some(tempfile::tempdir().expect("临时目录"))
    }

    /// 在 PATH 里找可执行文件（返回**绝对路径** —— 引擎的进程托管器要求路径存在，
    /// 相对名会被判成"缺引擎"）。
    fn which(name: &str) -> Option<PathBuf> {
        let path = std::env::var_os("PATH")?;
        std::env::split_paths(&path)
            .map(|directory| directory.join(name))
            .find(|candidate| candidate.is_file())
    }

    /// CLI 实现：探测结果必须与期望一致。
    #[tokio::test]
    async fn cli_probe_matches_expectations() {
        let Some(directory) = temp_dir() else {
            return;
        };
        let Some(video) = fixture_video(directory.path()) else {
            eprintln!("跳过：没有可用的 ffmpeg 可执行文件");
            return;
        };
        let manager = ProcessManager::new();
        let Some(ffprobe) = which("ffprobe") else {
            eprintln!("跳过：PATH 里没有 ffprobe");
            return;
        };
        let media = cli::FfmpegCliMedia::new(MediaConfig {
            ffmpeg: Some(PathBuf::from("ffmpeg")),
            ffprobe: Some(ffprobe),
        });
        let info = media
            .probe_video(
                &MediaContext {
                    process_manager: &manager,
                    observer: None,
                    log_path: None,
                },
                &video,
            )
            .await
            .expect("探测");
        assert_eq!(info.width, EXPECTED_WIDTH, "宽度不符：{info:?}");
        assert_eq!(info.height, EXPECTED_HEIGHT, "高度不符：{info:?}");
        assert!((info.fps - EXPECTED_FPS).abs() < 0.01, "帧率不符：{info:?}");
        assert!(
            info.total_frames == EXPECTED_FRAMES || info.total_frames == 0,
            "帧数不符：{info:?}"
        );
    }

    /// rsmpeg 实现：同一段视频必须给出**同样的**结果（两种实现互相对拍）。
    #[cfg(feature = "media-rsmpeg")]
    #[tokio::test]
    async fn rsmpeg_probe_matches_expectations() {
        let Some(directory) = temp_dir() else {
            return;
        };
        let Some(video) = fixture_video(directory.path()) else {
            eprintln!("跳过：没有可用的 ffmpeg 可执行文件（无法造夹具）");
            return;
        };
        let manager = ProcessManager::new();
        let media = rsmpeg::RsmpegMedia::new();
        let info = media
            .probe_video(
                &MediaContext {
                    process_manager: &manager,
                    observer: None,
                    log_path: None,
                },
                &video,
            )
            .await
            .expect("探测");
        assert_eq!(info.width, EXPECTED_WIDTH, "宽度不符：{info:?}");
        assert_eq!(info.height, EXPECTED_HEIGHT, "高度不符：{info:?}");
        assert!((info.fps - EXPECTED_FPS).abs() < 0.01, "帧率不符：{info:?}");
        assert!(
            info.total_frames == EXPECTED_FRAMES || info.total_frames == 0,
            "帧数不符：{info:?}"
        );
    }

    /// 抽帧：两种实现都要按同一套命名与数量产出画面。
    #[cfg(feature = "media-rsmpeg")]
    #[tokio::test]
    async fn rsmpeg_extracts_frames_with_the_cli_naming() {
        let Some(directory) = temp_dir() else {
            return;
        };
        let Some(video) = fixture_video(directory.path()) else {
            eprintln!("跳过：没有可用的 ffmpeg 可执行文件（无法造夹具）");
            return;
        };
        let output = directory.path().join("frames");
        let colmap_output = directory.path().join("colmap_frames");
        let masks = directory.path().join("masks");
        let manager = ProcessManager::new();
        let media = rsmpeg::RsmpegMedia::new();
        let mut plan = crate::video::FramePlan::default();
        plan.sampling_fps = 5.0;
        let result = media
            .extract_uniform_frames(
                &MediaContext {
                    process_manager: &manager,
                    observer: None,
                    log_path: None,
                },
                UniformFrameExtraction {
                    base: FrameExtraction {
                        input: video,
                        output_directory: output.clone(),
                        colmap_directory: colmap_output.clone(),
                        mask_directory: masks.clone(),
                        has_alpha: false,
                        target_dimensions: None,
                    },
                    plan,
                },
            )
            .await
            .expect("抽帧");
        assert_eq!(result.frame_count, EXPECTED_FRAMES / 2, "帧数不符：{result:?}");
        assert_eq!(result.width, EXPECTED_WIDTH);
        assert_eq!(result.height, EXPECTED_HEIGHT);
        // 命名约定：frame_000001.png（1 起）。
        assert!(output.join("frame_000001.png").is_file(), "缺少首帧");
        assert!(colmap_output.join("frame_000001.png").is_file(), "缺少 COLMAP 副本");
    }
}
