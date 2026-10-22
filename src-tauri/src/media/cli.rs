//! 外部 `ffmpeg` / `ffprobe` 二进制实现（默认路径，行为与主分支完全一致）。
//!
//! 这一层是**薄适配器**：真正的抽帧/探测逻辑仍在 `crate::engines::{ffmpeg, ffprobe}`，
//! 这里只把 trait 的中性请求翻译成 CLI 调用。

use std::path::{Path, PathBuf};

use async_trait::async_trait;

use super::{MediaContext, MediaTool, SelectedFrameExtraction, UniformFrameExtraction};
use crate::contracts::{BackendFamily, BackendIdentity};
use crate::engines::ffmpeg::FrameExtractionResult;
use crate::error::Result;
use crate::video::VideoInfo;

/// 用外部 ffmpeg/ffprobe 做媒体处理。
pub struct FfmpegCliMedia {
    ffmpeg: PathBuf,
    ffprobe: PathBuf,
}

impl FfmpegCliMedia {
    pub fn new(config: super::MediaConfig) -> Self {
        Self {
            ffmpeg: config.ffmpeg.unwrap_or_else(|| PathBuf::from("ffmpeg")),
            ffprobe: config.ffprobe.unwrap_or_else(|| PathBuf::from("ffprobe")),
        }
    }
}

#[async_trait]
impl MediaTool for FfmpegCliMedia {
    fn identity(&self) -> BackendIdentity {
        BackendIdentity {
            id: "ffmpeg-cli".into(),
            display_name: "FFmpeg（外部进程）".into(),
            version: String::new(),
            family: BackendFamily::ExternalCli,
        }
    }

    async fn probe_video(
        &self,
        context: &MediaContext<'_>,
        input: &Path,
    ) -> Result<VideoInfo> {
        crate::engines::ffprobe::probe_video(
            &self.ffprobe,
            input,
            context.log_path.clone(),
            context.process_manager,
        )
        .await
    }

    async fn extract_uniform_frames(
        &self,
        context: &MediaContext<'_>,
        request: UniformFrameExtraction,
    ) -> Result<FrameExtractionResult> {
        let base = request.base;
        crate::engines::ffmpeg::extract_uniform_frames(
            &self.ffmpeg,
            &base.input,
            &base.output_directory,
            &base.colmap_directory,
            &base.mask_directory,
            &request.plan,
            base.has_alpha,
            base.target_dimensions,
            context.log_path.clone(),
            context.process_manager,
            context.observer.clone(),
        )
        .await
    }

    async fn extract_selected_frames(
        &self,
        context: &MediaContext<'_>,
        request: SelectedFrameExtraction,
    ) -> Result<FrameExtractionResult> {
        let base = request.base;
        crate::engines::ffmpeg::extract_selected_frames(
            &self.ffmpeg,
            &base.input,
            &base.output_directory,
            &base.colmap_directory,
            &base.mask_directory,
            &request.selected_frames,
            base.has_alpha,
            base.target_dimensions,
            context.log_path.clone(),
            context.process_manager,
            context.observer.clone(),
        )
        .await
    }

    async fn extract_additional_frames(
        &self,
        context: &MediaContext<'_>,
        request: SelectedFrameExtraction,
    ) -> Result<FrameExtractionResult> {
        let base = request.base;
        crate::engines::ffmpeg::extract_additional_frames(
            &self.ffmpeg,
            &base.input,
            &base.output_directory,
            &base.colmap_directory,
            &base.mask_directory,
            &request.selected_frames,
            base.has_alpha,
            base.target_dimensions,
            context.log_path.clone(),
            context.process_manager,
            context.observer.clone(),
        )
        .await
    }
}
