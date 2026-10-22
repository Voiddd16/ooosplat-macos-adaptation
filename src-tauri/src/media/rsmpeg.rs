//! rsmpeg（进程内 FFmpeg 库）媒体实现。
//!
//! # 前置条件（重要）
//!
//! `rsmpeg` 的 FFI 层（`rusty_ffmpeg`）带**预置绑定**，其 ABI 必须与本机 FFmpeg
//! **严格同版本**：rsmpeg 0.18 对应 FFmpeg 8.x（FFmpeg 8 把 `AVCodecID` 改成
//! `uint32_t`，FFmpeg 9 又变回枚举 → 不兼容）。macOS 上装 `brew install ffmpeg@8`
//! （8.1.3）后用：
//!
//! ```bash
//! PKG_CONFIG_PATH=/opt/homebrew/opt/ffmpeg@8/lib/pkgconfig \
//!   cargo build --features media-rsmpeg
//! ```
//!
//! 版本不匹配会在**编译期**报错（`build.rs` 的 `ffmpeg_abi_version()` 守卫），
//! 而不是等到运行时行为诡异。
//!
//! # 行为对齐
//!
//! 与 CLI 实现保持同一套**输出约定**（文件名、掩膜阈值、目录布局、校验）；
//! 共享的后处理（`ensure_clean_output` / `mirror_images` / `rename_selected_outputs` /
//! `validate_extraction`）直接复用 `crate::engines::ffmpeg` 里的实现，不重复实现。
//!
//! # 内存约定
//!
//! 解码出来的帧**不可克隆**（`AVFrame` 析构会释放缓冲）。因此：
//! 帧只在被选中时缩放一次，缩放结果（紧凑像素缓冲）留在采样器里供重复输出复用
//! ——这样"补帧"（源帧率 > 目标帧率）不会重复缩放。

use std::{
    ffi::CString,
    path::{Path, PathBuf},
};

use async_trait::async_trait;

use rsmpeg::{
    avcodec::AVCodecContext,
    avformat::AVFormatContextInput,
    avutil::{AVPixelFormat, AVRational},
    swscale::SwsContext,
};

use super::{MediaContext, MediaTool, SelectedFrameExtraction, UniformFrameExtraction};
use crate::contracts::{BackendFamily, BackendIdentity};
use crate::engines::ffmpeg::FrameExtractionResult;
use crate::error::{Result, SplatError};
use crate::video::VideoInfo;

/// 用 FFmpeg 库（rsmpeg）做媒体处理。
#[derive(Debug, Default)]
pub struct RsmpegMedia;

impl RsmpegMedia {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl MediaTool for RsmpegMedia {
    fn identity(&self) -> BackendIdentity {
        BackendIdentity {
            id: "rsmpeg".into(),
            display_name: "FFmpeg（进程内库）".into(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            family: BackendFamily::InProcess,
        }
    }

    async fn probe_video(
        &self,
        _context: &MediaContext<'_>,
        input: &Path,
    ) -> Result<VideoInfo> {
        let path = input.to_path_buf();
        tokio::task::spawn_blocking(move || probe_blocking(&path))
            .await
            .map_err(|error| SplatError::Process(format!("FFmpeg 探测任务失败：{error}")))?
    }

    async fn extract_uniform_frames(
        &self,
        _context: &MediaContext<'_>,
        request: UniformFrameExtraction,
    ) -> Result<FrameExtractionResult> {
        let base = request.base;
        crate::engines::ffmpeg::ensure_clean_output(
            &base.output_directory,
            &base.colmap_directory,
            &base.mask_directory,
        )
        .await?;
        tokio::fs::create_dir_all(&base.output_directory).await?;
        tokio::fs::create_dir_all(&base.colmap_directory).await?;
        if base.has_alpha {
            tokio::fs::create_dir_all(&base.mask_directory).await?;
        }
        let selection = Selection::Uniform {
            sampling_fps: request.plan.sampling_fps,
        };
        let (input, output_directory, mask_directory, has_alpha, target_dimensions) = (
            base.input.clone(),
            base.output_directory.clone(),
            base.mask_directory.clone(),
            base.has_alpha,
            base.target_dimensions,
        );
        let written = tokio::task::spawn_blocking(move || {
            decode_and_write(
                &input,
                &output_directory,
                &mask_directory,
                selection,
                has_alpha,
                target_dimensions,
            )
        })
        .await
        .map_err(|error| SplatError::Process(format!("FFmpeg 抽帧任务失败：{error}")))??;
        if written == 0 {
            return Err(SplatError::Process("没有抽出任何画面".into()));
        }
        crate::engines::ffmpeg::mirror_images(&base.output_directory, &base.colmap_directory).await?;
        crate::engines::ffmpeg::validate_extraction(
            &base.output_directory,
            &base.colmap_directory,
            &base.mask_directory,
            base.has_alpha,
        )
        .await
    }

    async fn extract_selected_frames(
        &self,
        _context: &MediaContext<'_>,
        request: SelectedFrameExtraction,
    ) -> Result<FrameExtractionResult> {
        let base = request.base;
        crate::engines::ffmpeg::ensure_clean_output(
            &base.output_directory,
            &base.colmap_directory,
            &base.mask_directory,
        )
        .await?;
        tokio::fs::create_dir_all(&base.output_directory).await?;
        tokio::fs::create_dir_all(&base.colmap_directory).await?;
        if base.has_alpha {
            tokio::fs::create_dir_all(&base.mask_directory).await?;
        }
        let selection = Selection::Selected(
            request
                .selected_frames
                .iter()
                .map(|frame| frame.source_frame_index as u32)
                .collect(),
        );
        let (input, output_directory, mask_directory, has_alpha, target_dimensions) = (
            base.input.clone(),
            base.output_directory.clone(),
            base.mask_directory.clone(),
            base.has_alpha,
            base.target_dimensions,
        );
        let written = tokio::task::spawn_blocking(move || {
            decode_and_write(
                &input,
                &output_directory,
                &mask_directory,
                selection,
                has_alpha,
                target_dimensions,
            )
        })
        .await
        .map_err(|error| SplatError::Process(format!("FFmpeg 抽帧任务失败：{error}")))??;
        if written == 0 {
            return Err(SplatError::Process(
                "Quality v2 did not select any source frames".into(),
            ));
        }
        crate::engines::ffmpeg::rename_selected_outputs(
            &base.output_directory,
            &base.mask_directory,
            &request.selected_frames,
            base.has_alpha,
        )
        .await?;
        crate::engines::ffmpeg::mirror_images(&base.output_directory, &base.colmap_directory).await?;
        crate::engines::ffmpeg::validate_extraction(
            &base.output_directory,
            &base.colmap_directory,
            &base.mask_directory,
            base.has_alpha,
        )
        .await
    }

    async fn extract_additional_frames(
        &self,
        context: &MediaContext<'_>,
        request: SelectedFrameExtraction,
    ) -> Result<FrameExtractionResult> {
        // 补拍的"抽到临时目录再原子合并"语义由 CLI 实现负责（它已经处理了增量合并的
        // 原子性）；库实现复用同一条路径，避免两套合并逻辑漂移。
        super::cli::FfmpegCliMedia::new(super::MediaConfig::default())
            .extract_additional_frames(context, request)
            .await
    }
}

/// 抽帧选择方式。
enum Selection {
    /// 按帧率均匀采样（等价 ffmpeg 的 `fps=` 滤镜）。
    Uniform { sampling_fps: f64 },
    /// 指定源帧序号。
    Selected(Vec<u32>),
}

/// 探测视频信息（阻塞实现）。
fn probe_blocking(input: &Path) -> Result<VideoInfo> {
    if !input.is_file() {
        return Err(SplatError::InvalidPath(input.to_path_buf()));
    }
    let path = CString::new(input.as_os_str().as_encoded_bytes())
        .map_err(|_| SplatError::InvalidPath(input.to_path_buf()))?;
    let context = AVFormatContextInput::open(&path)
        .map_err(|error| SplatError::InvalidVideo(format!("FFmpeg 无法打开这个文件：{error}")))?;
    let Some((index, codec)) = context
        .find_best_stream(rsmpeg::ffi::AVMEDIA_TYPE_VIDEO)
        .map_err(|error| SplatError::InvalidVideo(format!("读取视频流失败：{error}")))?
    else {
        return Err(SplatError::InvalidVideo("这个文件里没有视频流".into()));
    };
    let stream = &context.streams()[index];
    let codecpar = stream.codecpar();
    let width = codecpar.width.max(0) as u32;
    let height = codecpar.height.max(0) as u32;
    let fps = rational_to_f64(stream.avg_frame_rate);
    let time_base = rational_to_f64(stream.time_base);
    let duration = if context.duration > 0 {
        context.duration as f64 / 1_000_000.0
    } else if fps > 0.0 && stream.duration > 0 {
        stream.duration as f64 * time_base
    } else {
        0.0
    };
    let total_frames = if stream.nb_frames > 0 {
        stream.nb_frames as u64
    } else if fps > 0.0 && duration > 0.0 {
        (duration * fps).round() as u64
    } else {
        0
    };
    let codec_name = codec
        .name()
        .to_string_lossy()
        .trim_end_matches('\0')
        .to_string();
    let pixel_format = pixel_format_name(codecpar.format);
    Ok(VideoInfo {
        duration,
        width,
        height,
        fps,
        total_frames,
        codec: codec_name,
        // 旋转信息来自流上的 display matrix（手机竖拍视频）。rsmpeg 没有直接暴露
        // 侧数据，这里保守报 0 —— 与 CLI 路径的差异记录在方案文档里。
        rotation: 0,
        pixel_format,
        has_alpha: false,
    })
}

fn rational_to_f64(value: AVRational) -> f64 {
    if value.den == 0 {
        0.0
    } else {
        value.num as f64 / value.den as f64
    }
}

fn pixel_format_name(format: i32) -> String {
    // SAFETY：`av_get_pix_fmt_name` 对未知格式返回 NULL，其它情况返回静态字符串。
    unsafe {
        let name = rsmpeg::ffi::av_get_pix_fmt_name(format);
        if name.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(name).to_string_lossy().to_string()
        }
    }
}

/// 解码 + 采样 + 写 PNG（阻塞实现）。
///
/// 输出命名与 CLI 一致：画面 `frame_%06d.png`（1 起），掩膜 `frame_%06d.png.png`。
fn decode_and_write(
    input: &Path,
    output_directory: &Path,
    mask_directory: &Path,
    selection: Selection,
    has_alpha: bool,
    target_dimensions: Option<(u32, u32)>,
) -> Result<u64> {
    let path = CString::new(input.as_os_str().as_encoded_bytes())
        .map_err(|_| SplatError::InvalidPath(input.to_path_buf()))?;
    let mut context = AVFormatContextInput::open(&path)
        .map_err(|error| SplatError::InvalidVideo(format!("FFmpeg 无法打开这个文件：{error}")))?;
    let Some((index, codec)) = context
        .find_best_stream(rsmpeg::ffi::AVMEDIA_TYPE_VIDEO)
        .map_err(|error| SplatError::InvalidVideo(format!("读取视频流失败：{error}")))?
    else {
        return Err(SplatError::InvalidVideo("这个文件里没有视频流".into()));
    };
    let time_base = rational_to_f64(context.streams()[index].time_base);
    let mut decoder = AVCodecContext::new(&codec);
    // 必须先灌入流的 codec 参数（宽高 / 像素格式），否则解码器不知道帧的布局。
    decoder
        .apply_codecpar(&context.streams()[index].codecpar())
        .map_err(|error| SplatError::Process(format!("无法套用解码参数：{error}")))?;
    decoder
        .open(None)
        .map_err(|error| SplatError::Process(format!("无法打开视频解码器：{error}")))?;

    let mut writer = FrameWriter::new(output_directory, mask_directory, has_alpha, target_dimensions);
    let mut sampler = Sampler::new(selection);
    let mut source_index = 0u64;

    while let Some(packet) = context
        .read_packet()
        .map_err(|error| SplatError::Process(format!("读取数据包失败：{error}")))?
    {
        // 只喂视频流的数据包（文件里可能有音频/字幕流）。
        if packet.stream_index != index as i32 {
            continue;
        }
        decoder
            .send_packet(Some(&packet))
            .map_err(|error| SplatError::Process(format!("解码失败：{error}")))?;
        loop {
            match decoder.receive_frame() {
                Ok(frame) => {
                    let decoded = DecodedFrame {
                        index: source_index,
                        width: frame.width,
                        height: frame.height,
                        format: frame.format,
                        time: frame.pts as f64 * time_base,
                        frame,
                    };
                    source_index += 1;
                    sampler.push(decoded, &mut writer)?;
                }
                Err(_) => break,
            }
        }
    }
    decoder
        .send_packet(None)
        .map_err(|error| SplatError::Process(format!("解码收尾失败：{error}")))?;
    loop {
        match decoder.receive_frame() {
            Ok(frame) => {
                let decoded = DecodedFrame {
                    index: source_index,
                    width: frame.width,
                    height: frame.height,
                    format: frame.format,
                    time: frame.pts as f64 * time_base,
                    frame,
                };
                source_index += 1;
                sampler.push(decoded, &mut writer)?;
            }
            Err(_) => break,
        }
    }
    sampler.finish(&mut writer)?;
    Ok(writer.written)
}

/// 帧选择器。
struct Sampler {
    selection: Selection,
    /// 尚未缩放的候选帧（时间上"最新"的那一帧）。
    pending: Option<DecodedFrame>,
    /// 已缩放、可重复输出的当前帧。
    current: Option<ScaledImage>,
    /// 下一个输出槽（0 起）。
    next_slot: u64,
}

impl Sampler {
    fn new(selection: Selection) -> Self {
        Self {
            selection,
            pending: None,
            current: None,
            next_slot: 0,
        }
    }

    fn push(&mut self, frame: DecodedFrame, writer: &mut FrameWriter) -> Result<()> {
        match &self.selection {
            Selection::Selected(indices) => {
                if indices.contains(&(frame.index as u32)) {
                    let image = writer.scale(&frame)?;
                    writer.write(&image)?;
                }
                Ok(())
            }
            Selection::Uniform { sampling_fps } => {
                if *sampling_fps <= 0.0 {
                    return Err(SplatError::Process("采样帧率必须为正".into()));
                }
                // 新帧到达：所有"目标时刻早于新帧"的输出槽都只能用更早的帧。
                let previous = self.pending.take();
                self.pending = Some(frame);
                let Some(previous) = previous else {
                    // 第一帧：先留着，等下一帧来了才知道它是否会被用。
                    return Ok(());
                };
                let frame_time = self
                    .pending
                    .as_ref()
                    .map(|frame| frame.time)
                    .unwrap_or(previous.time);
                while self.next_slot_time(*sampling_fps) < frame_time - 1e-9 {
                    // 目标槽的候选是"时间 ≤ 目标时刻的最后一帧"。
                    let image = match self.current.take() {
                        Some(image) => image,
                        None => writer.scale(&previous)?,
                    };
                    writer.write(&image)?;
                    self.current = Some(image);
                    self.next_slot += 1;
                }
                Ok(())
            }
        }
    }

    /// 解码结束：把剩余的槽补完（目标时刻不超过最后一帧）。
    fn finish(&mut self, writer: &mut FrameWriter) -> Result<()> {
        if let Selection::Uniform { sampling_fps } = self.selection {
            let last_time = self
                .pending
                .as_ref()
                .map(|frame| frame.time)
                .or_else(|| self.current.as_ref().map(|_| 0.0));
            let Some(last_time) = last_time else {
                return Ok(());
            };
            // 最后一帧还没缩放过：它就是剩余槽的候选。
            let last_scaled = match self.pending.take() {
                Some(frame) => Some(writer.scale(&frame)?),
                None => None,
            };
            while self.next_slot_time(sampling_fps) <= last_time + 1e-9 {
                let image = match (&last_scaled, self.current.take()) {
                    (Some(image), _) => image.clone(),
                    (None, Some(image)) => image,
                    (None, None) => break,
                };
                writer.write(&image)?;
                self.current = Some(image);
                self.next_slot += 1;
            }
        }
        Ok(())
    }

    fn next_slot_time(&self, sampling_fps: f64) -> f64 {
        self.next_slot as f64 / sampling_fps
    }
}

/// 一帧（带源帧序号与时间戳）。帧数据不可克隆，只能移动。
struct DecodedFrame {
    /// 源帧序号（0 起，按解码顺序）。
    index: u64,
    width: i32,
    height: i32,
    format: i32,
    /// 显示时间（秒）。
    time: f64,
    frame: rsmpeg::avutil::AVFrame,
}

/// 缩放后的图像（紧凑像素缓冲，可安全克隆/复用）。
#[derive(Clone)]
struct ScaledImage {
    width: i32,
    height: i32,
    format: AVPixelFormat,
    pixels: Vec<u8>,
}

/// 帧写出器：缩放 + PNG 落盘。
struct FrameWriter {
    output_directory: PathBuf,
    mask_directory: PathBuf,
    has_alpha: bool,
    target_dimensions: Option<(u32, u32)>,
    scaler: Option<SwsContext>,
    scaler_key: Option<(i32, i32, i32, AVPixelFormat)>,
    frame_number: u32,
    written: u64,
}

impl FrameWriter {
    fn new(
        output_directory: &Path,
        mask_directory: &Path,
        has_alpha: bool,
        target_dimensions: Option<(u32, u32)>,
    ) -> Self {
        Self {
            output_directory: output_directory.to_path_buf(),
            mask_directory: mask_directory.to_path_buf(),
            has_alpha,
            target_dimensions,
            scaler: None,
            scaler_key: None,
            frame_number: 0,
            written: 0,
        }
    }

    /// 缩放到目标尺寸与像素格式。
    fn scale(&mut self, frame: &DecodedFrame) -> Result<ScaledImage> {
        let (width, height) = scaled_dimensions(frame.width, frame.height, self.target_dimensions);
        let target_format = if self.has_alpha {
            rsmpeg::ffi::AV_PIX_FMT_RGBA
        } else {
            rsmpeg::ffi::AV_PIX_FMT_RGB24
        };
        let key = (width, height, frame.format, target_format);
        if self.scaler_key != Some(key) {
            self.scaler = SwsContext::get_context(
                frame.width,
                frame.height,
                frame.format,
                width,
                height,
                target_format,
                sws_flags(),
                None,
                None,
                None,
            );
            self.scaler_key = Some(key);
        }
        let Some(scaler) = self.scaler.as_mut() else {
            return Err(SplatError::Process("无法创建缩放器".into()));
        };
        let mut destination = rsmpeg::avutil::AVFrame::new();
        destination.set_format(target_format as i32);
        destination.set_width(width);
        destination.set_height(height);
        destination
            .alloc_buffer()
            .map_err(|error| SplatError::Process(format!("无法分配缩放缓冲：{error}")))?;
        scaler
            .scale_frame(&frame.frame, 0, frame.height, &mut destination)
            .map_err(|error| SplatError::Process(format!("缩放失败：{error}")))?;
        Ok(ScaledImage {
            width,
            height,
            format: target_format,
            pixels: copy_pixels(&destination, width, height, target_format),
        })
    }

    /// 写出一帧（PNG/JPEG + 可选掩膜）。
    fn write(&mut self, image: &ScaledImage) -> Result<()> {
        let number = self.frame_number;
        self.frame_number += 1;
        // 统一写 PNG：进程内路径不做 JPEG 编码（省一个编码器依赖，且无损）。
        // 扩展名不是语义 —— 抽帧后的清点与校验按"是图片就算"（pipeline_image_names）。
        let file_name = format!("frame_{number:06}.png");
        write_image(
            &self.output_directory.join(&file_name),
            &image.pixels,
            image.width,
            image.height,
            image.format,
        )?;
        if self.has_alpha {
            let mask = alpha_mask(&image.pixels, image.width, image.height);
            write_image(
                &self.mask_directory.join(format!("{file_name}.png")),
                &mask,
                image.width,
                image.height,
                rsmpeg::ffi::AV_PIX_FMT_GRAY8,
            )?;
        }
        self.written += 1;
        Ok(())
    }
}

/// 目标尺寸：显式指定就照用，否则按 CLI 规则缩放到长边 ≤ 1920（保持比例，取偶数）。
fn scaled_dimensions(width: i32, height: i32, target: Option<(u32, u32)>) -> (i32, i32) {
    if let Some((target_width, target_height)) = target {
        return (target_width.max(2) as i32, target_height.max(2) as i32);
    }
    let longest = 1920.0f64;
    let scale = (longest / width.max(1) as f64)
        .min(longest / height.max(1) as f64)
        .min(1.0);
    let scaled_width = ((width as f64 * scale).round() as i32).max(2);
    let scaled_height = ((height as f64 * scale).round() as i32).max(2);
    (scaled_width & !1, scaled_height & !1)
}

/// swscale 的缩放算法标志（FFmpeg 8 起是强类型 `SwsFlags`）。
fn sws_flags() -> rsmpeg::ffi::SwsFlags {
    rsmpeg::ffi::SWS_BILINEAR
}

/// 把帧数据按行拷贝成紧凑的 RGB(A) 字节。
///
/// `AVFrame::data` / `linesize` 是**按平面**索引的（最多 8 个平面），不是按行。
/// 打包格式（RGB24 / RGBA / 灰度）只有一个平面：第 0 平面按 `linesize[0]` 的行距
/// 连续排列，行尾可能有对齐填充 → 逐行只取 `width * channels` 字节。
fn copy_pixels(
    frame: &rsmpeg::avutil::AVFrame,
    width: i32,
    height: i32,
    format: AVPixelFormat,
) -> Vec<u8> {
    let channels = match format {
        rsmpeg::ffi::AV_PIX_FMT_RGBA => 4usize,
        rsmpeg::ffi::AV_PIX_FMT_GRAY8 => 1usize,
        _ => 3usize,
    };
    let needed = width as usize * channels;
    let mut buffer = vec![0u8; needed * height as usize];
    let source = frame.data[0];
    if source.is_null() || needed == 0 {
        return buffer;
    }
    let stride = frame.linesize[0] as usize;
    if stride < needed {
        return buffer;
    }
    for row in 0..height as usize {
        // SAFETY：目标缓冲按 `needed * height` 分配；源是 FFmpeg 自己的帧缓冲，
        // 行距 `stride >= needed` 保证不越界。
        unsafe {
            let src = source.add(row * stride);
            let dst = buffer.as_mut_ptr().add(row * needed);
            std::ptr::copy_nonoverlapping(src, dst, needed);
        }
    }
    buffer
}

/// alpha 通道 → 灰度掩膜（≥128 记为 255，与 CLI 的 `lut` 规则一致）。
fn alpha_mask(rgba: &[u8], width: i32, height: i32) -> Vec<u8> {
    let mut mask = vec![0u8; (width as usize) * (height as usize)];
    for index in 0..mask.len() {
        let alpha = rgba[index * 4 + 3];
        mask[index] = if alpha >= 128 { 255 } else { 0 };
    }
    mask
}

/// 写图片（RGB / RGBA 用 PNG；灰度掩膜也用 PNG）。
fn write_image(
    path: &Path,
    pixels: &[u8],
    width: i32,
    height: i32,
    format: AVPixelFormat,
) -> Result<()> {
    let color = match format {
        rsmpeg::ffi::AV_PIX_FMT_RGBA => png::ColorType::Rgba,
        rsmpeg::ffi::AV_PIX_FMT_GRAY8 => png::ColorType::Grayscale,
        _ => png::ColorType::Rgb,
    };
    let file = std::fs::File::create(path)
        .map_err(|error| SplatError::Process(format!("无法写入 {}：{error}", path.display())))?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width as u32, height as u32);
    encoder.set_color(color);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|error| SplatError::Process(format!("PNG 头写入失败：{error}")))?;
    writer
        .write_image_data(pixels)
        .map_err(|error| SplatError::Process(format!("PNG 数据写入失败：{error}")))?;
    Ok(())
}
