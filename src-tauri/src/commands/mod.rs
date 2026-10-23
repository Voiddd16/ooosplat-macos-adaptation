use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tauri::{ipc::InvokeBody, Emitter, Manager, State};
use tauri_plugin_opener::OpenerExt;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tokio::sync::Mutex;
use uuid::Uuid;

pub mod html_export;
pub mod model_export;

use crate::{
    contracts::{AccelerationStatus, ComponentStatus, FailureKind},
    engines::{
        ffprobe::probe_video,
        health::{
            detect_acceleration, refresh_acceleration,
        },
        EnginePaths,
    },
    error::{Result, SplatError},
    pipeline::{
        estimate::{
            estimate_runtime_for_images_with_brush_and_resolution,
            estimate_runtime_with_brush_and_resolution, RuntimeEstimate,
        },
        runner::{
            PipelineFailureContext, PipelineResult, PipelineRunner, ReshootInputInfo,
            ReshootInputType, ReshootSourceInfo,
        },
        PipelineComponent, PipelineStage,
    },
    presets::{
        resolve_brush_training_preset, resolve_brush_training_preset_for_plan,
        resolve_planner_resolution_plan, Quality,
    },
    process::ProcessManager,
    project::{
        catalog::{self, AppSettings, ProjectOverview, ProjectSummary},
        manager::atomic_write_json,
        GaussianCrop, GaussianEditing, GaussianTransform, PipelineStateFile, ProjectInputType,
        ProjectStatus,
    },
    reconstruction::{
        edit_mask::{
            cleanup_old_masks, count_deleted, mask_path, packed_mask_bytes, read_mask,
            remove_edit_files, write_mask_atomic,
        },
        ply::inspect_gaussian_ply,
        splat_transform::{export_transformed_ply_with_edits, GaussianExportEdits},
    },
    telemetry::{TelemetryPreferences, TelemetryService},
    video::{
        analyze_image_sequence, create_image_plan, FramePlan, FrameSelectionStrategy,
        ImageSequenceInfo, QualityV2FrameSelection, UniformRatioFrameSelection, VideoInfo,
    },
};

pub type PipelineController = crate::tasks::TaskService;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PipelineCommandError {
    pub(crate) code: &'static str,
    pub(crate) message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) failed_stage: Option<PipelineStage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) component: Option<PipelineComponent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) failure_kind: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) project_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) project_path: Option<Box<PathBuf>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logs_directory: Option<Box<PathBuf>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) failure_id: Option<Box<Uuid>>,
}

impl From<SplatError> for PipelineCommandError {
    fn from(error: SplatError) -> Self {
        Self {
            code: if matches!(error, SplatError::Cancelled) {
                "cancelled"
            } else {
                "pipeline_failed"
            },
            message: error.to_string(),
            failed_stage: None,
            component: None,
            failure_kind: None,
            project_id: None,
            project_path: None,
            logs_directory: None,
            failure_id: None,
        }
    }
}

impl PipelineCommandError {
    pub(crate) fn from_runner(error: &SplatError, runner: &PipelineRunner) -> Self {
        let message = error.to_string();
        let cancelled = matches!(error, SplatError::Cancelled);
        let PipelineFailureContext {
            failed_stage,
            project_id,
            project_path,
            logs_directory,
        } = runner.failure_context();
        let (component, failure_kind) = classify_pipeline_failure(failed_stage, &message);
        Self {
            code: if cancelled {
                "cancelled"
            } else {
                "pipeline_failed"
            },
            message,
            failed_stage,
            component,
            failure_kind: if cancelled {
                None
            } else {
                failure_kind.map(|kind| kind.label())
            },
            project_id: project_id.map(|value| value.to_string()),
            project_path: project_path.map(Box::new),
            logs_directory: logs_directory.map(Box::new),
            failure_id: None,
        }
    }
}

fn classify_pipeline_failure(
    stage: Option<PipelineStage>,
    message: &str,
) -> (Option<PipelineComponent>, Option<FailureKind>) {
    let lower = message.to_ascii_lowercase();
    match stage {
        Some(PipelineStage::Reconstructing | PipelineStage::ValidatingReconstruction) => {
            let source_error = [
                "no initial image pair",
                "no good initial image pair",
                "could not find a good initial image pair",
                "failed to find an initial image pair",
                "failed to register",
                "could not register",
                "no images registered",
                "registered 0 images",
                "discarding reconstruction",
                "failed to create any sparse model",
                "did not produce a usable sparse model",
                "未生成完整的稀疏模型",
                "稀疏重建没有可用的注册图像或三维点",
                "稀疏重建输出不完整",
            ]
            .iter()
            .any(|needle| lower.contains(needle));
            let database_io_error = ["io error", "i/o error"]
                .iter()
                .any(|needle| lower.contains(needle))
                && ["database", "sqlite", "filesystem", "project file"]
                    .iter()
                    .any(|needle| lower.contains(needle));
            let storage_error = database_io_error
                || [
                    "database is locked",
                    "database locked",
                    "unable to open database",
                    "failed to open database",
                    "cannot open database",
                    "unable to open database file",
                    "read-only database",
                    "readonly database",
                    "no space left",
                    "disk full",
                    "input/output error",
                    "being used by another process",
                    "used by another process",
                    "sharing violation",
                    "access denied",
                    "permission denied",
                    "数据库被锁定",
                    "无法打开数据库",
                    "磁盘空间不足",
                    "磁盘已满",
                    "另一个进程正在使用",
                    "拒绝访问",
                    "权限不足",
                ]
                .iter()
                .any(|needle| lower.contains(needle));
            (
                Some(PipelineComponent::Reconstruction),
                Some(if storage_error {
                    FailureKind::ReconstructionStorage
                } else {
                    FailureKind::ReconstructionInput
                }),
            )
        }
        Some(PipelineStage::TrainingSplats) => {
            let device_lost = [
                "devicelost",
                "device lost",
                "device_lost",
                "parent device is lost",
                "vk_error_device_lost",
                "dxgi_error_device_removed",
                "显卡设备连接中断",
            ]
            .iter()
            .any(|needle| lower.contains(needle));
            let dataset_error = [
                "early eof",
                "failed to load dataset",
                "io error",
                "i/o error",
                "no such file",
                "access denied",
                "permission denied",
            ]
            .iter()
            .any(|needle| lower.contains(needle));
            (
                Some(PipelineComponent::Training),
                Some(if device_lost {
                    FailureKind::TrainingDeviceLost
                } else if dataset_error {
                    FailureKind::TrainingDataset
                } else {
                    FailureKind::TrainingAccelerator
                }),
            )
        }
        Some(PipelineStage::ProbingVideo | PipelineStage::ExtractingFrames) => {
            (Some(PipelineComponent::Media), None)
        }
        Some(PipelineStage::ExtractingFeatures | PipelineStage::Matching) => {
            (Some(PipelineComponent::Reconstruction), None)
        }
        _ => (Some(PipelineComponent::System), None),
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DroppedInputInfo {
    input_type: ProjectInputType,
}

async fn classify_dropped_input_path(path: &Path) -> Result<DroppedInputInfo> {
    let metadata = tokio::fs::metadata(path)
        .await
        .map_err(|error| SplatError::Process(format!("拖入的素材不存在或无法访问：{error}")))?;
    if metadata.is_dir() {
        return Ok(DroppedInputInfo {
            input_type: ProjectInputType::Images,
        });
    }
    if !metadata.is_file() {
        return Err(SplatError::Process(
            "仅支持 MP4/MOV 视频文件或图片文件夹".into(),
        ));
    }
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match extension.as_str() {
        "mp4" | "mov" => Ok(DroppedInputInfo {
            input_type: ProjectInputType::Video,
        }),
        "jpg" | "jpeg" | "png" => Err(SplatError::Process(
            "请拖入包含 JPG、JPEG 或 PNG 的文件夹，不支持单张图片".into(),
        )),
        _ => Err(SplatError::Process(
            "仅支持 MP4/MOV 视频文件或图片文件夹".into(),
        )),
    }
}

/// 构造媒体工具（编译期选定的实现）。
fn media_tool(paths: &crate::engines::EnginePaths) -> std::sync::Arc<dyn crate::media::MediaTool> {
    crate::media::active(crate::media::MediaConfig {
        ffmpeg: Some(paths.ffmpeg.clone()),
        ffprobe: Some(paths.ffprobe.clone()),
    })
}

/// 媒体作业环境。
fn media_context(
    process_manager: &ProcessManager,
    observer: Option<crate::process::ProcessObserver>,
) -> crate::media::MediaContext<'_> {
    crate::media::MediaContext {
        process_manager,
        observer,
        log_path: None,
    }
}

#[tauri::command]
pub async fn classify_dropped_input(path: String) -> Result<DroppedInputInfo> {
    classify_dropped_input_path(Path::new(&path)).await
}

#[derive(Default)]
pub struct PreviewController {
    active: Mutex<Option<GaussianPreviewSession>>,
    lifecycle: Mutex<()>,
    metadata_write: Mutex<()>,
    export: std::sync::Arc<Mutex<()>>,
    model_export: Mutex<Option<std::sync::Arc<model_export::ModelExportSession>>>,
    video_export: Mutex<Option<GaussianVideoExportSession>>,
    html_export: Mutex<Option<html_export::HtmlExportSession>>,
    edit_save: Mutex<Option<GaussianEditSaveSession>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppRuntimeStatus {
    pipeline_running: bool,
    pipeline_project_id: Option<String>,
    pipeline_workspace_task_id: Option<String>,
    pipeline_run_elapsed_ms: u64,
    pipeline_elapsed_offset_ms: u64,
    preview_project_id: Option<String>,
    acceleration: Option<AccelerationStatus>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectTaskLogLine {
    source: String,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectTaskDetail {
    project: ProjectSummary,
    input_type: Option<ProjectInputType>,
    video_projection: Option<crate::video::panorama::VideoProjection>,
    input_projection: Option<crate::video::panorama::VideoProjection>,
    source_path: PathBuf,
    projects_root: PathBuf,
    planner_enabled: Option<bool>,
    estimated_frames: Option<u64>,
    stage: Option<PipelineStage>,
    progress: Option<f64>,
    input_images: Option<u64>,
    registered_images: Option<u64>,
    video: Option<VideoInfo>,
    image_sequence: Option<ImageSequenceInfo>,
    source_project_id: Option<String>,
    logs: Vec<ProjectTaskLogLine>,
    imported: Option<ImportedProjectDetail>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportedProjectDetail {
    source_model_path: PathBuf,
    format: crate::project::ImportedGaussianFormat,
    source_available: bool,
    colmap_root: Option<PathBuf>,
    sparse_model_path: Option<PathBuf>,
    database_available: bool,
    images_available: bool,
    reshoot_eligible: bool,
    reshoot_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProjectLocation {
    Project,
    Logs,
    Source,
}

#[derive(Debug, Clone)]
struct GaussianPreviewSession {
    project_id: Uuid,
    asset_paths: Vec<PathBuf>,
    read_token: Option<Uuid>,
    read_path: Option<PathBuf>,
    read_size: Option<u64>,
    read_modified_ms: Option<u64>,
}

#[derive(Debug, Clone)]
struct GaussianEditSaveSession {
    edit_id: Uuid,
    project_id: Uuid,
    base_revision: u64,
    next_revision: u64,
    splat_count: u64,
    crop: Option<GaussianCrop>,
}

#[derive(Debug, Clone)]
struct GaussianVideoExportSession {
    export_id: Uuid,
    project_id: Uuid,
    destination: PathBuf,
    temporary: PathBuf,
    orientation: VideoOrientation,
    edit_revision: u64,
    transform: GaussianTransform,
    cancel: tokio_util::sync::CancellationToken,
    running: bool,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoOrientation {
    #[default]
    Portrait,
    Landscape,
}

impl VideoOrientation {
    fn dimensions(self) -> (u32, u32) {
        match self {
            Self::Portrait => (1080, 1920),
            Self::Landscape => (1920, 1080),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GaussianPreviewDescriptor {
    project_id: Uuid,
    model_path: PathBuf,
    asset_path: PathBuf,
    format: &'static str,
    file_size: u64,
    splat_count: u64,
    transform: GaussianTransform,
    editing: GaussianEditing,
    edit_mask_asset_path: Option<PathBuf>,
    read_token: Option<Uuid>,
    reference_view: Option<crate::reconstruction::reference_view::TrainingReferenceView>,
    reference_view_unavailable: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GaussianEditDraft {
    crop: Option<GaussianCrop>,
    base_revision: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GaussianEditSaveReservation {
    edit_id: Uuid,
    expected_mask_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GaussianExportProgress {
    project_id: Uuid,
    processed_splats: u64,
    total_splats: u64,
    progress: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GaussianExportResult {
    path: PathBuf,
    file_size: u64,
    splat_count: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GaussianVideoExportReservation {
    export_id: Uuid,
    destination_path: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GaussianVideoExportResult {
    path: PathBuf,
    file_size: u64,
    width: u32,
    height: u32,
    fps: u32,
    duration_ms: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbeAndPlan {
    projection_detection: Option<crate::video::panorama::ProjectionDetection>,
    input_type: ProjectInputType,
    video: Option<VideoInfo>,
    image_sequence: Option<ImageSequenceInfo>,
    plan: FramePlan,
    estimate: RuntimeEstimate,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoAnalysis {
    video: VideoInfo,
    projection_detection: crate::video::panorama::ProjectionDetection,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InputAnalysis {
    input_type: ProjectInputType,
    video: Option<VideoInfo>,
    image_sequence: Option<ImageSequenceInfo>,
    projection_detection: crate::video::panorama::ProjectionDetection,
}

async fn analyze_media(engine_paths: &EnginePaths, input: &Path) -> Result<InputAnalysis> {
    static CACHE: std::sync::OnceLock<
        tokio::sync::Mutex<std::collections::VecDeque<(String, InputAnalysis)>>,
    > = std::sync::OnceLock::new();
    let identity = crate::video::panorama::source_identity(input)?;
    // Single flight, bounded cache; quality/planner changes never re-open image headers.
    let mut cache = CACHE.get_or_init(Default::default).lock().await;
    if let Some((_, value)) = cache.iter().find(|(key, _)| key == &identity) {
        return Ok(value.clone());
    }
    let value = if input.is_dir() {
        let path = input.to_path_buf();
        let analysis =
            tokio::task::spawn_blocking(move || crate::video::image_projection::analyze(&path))
                .await
                .map_err(|e| SplatError::Process(e.to_string()))??;
        InputAnalysis {
            input_type: ProjectInputType::Images,
            video: None,
            image_sequence: Some(analysis.info),
            projection_detection: analysis.detection,
        }
    } else {
        let (video, json) = crate::engines::ffprobe::probe_video_metadata(
            &engine_paths.ffprobe,
            input,
            None,
            &ProcessManager::new(),
        )
        .await?;
        let mut detection = crate::video::panorama::detect_metadata(
            &json,
            &video,
            input
                .extension()
                .is_some_and(|s| s.eq_ignore_ascii_case("mp4")),
        );
        detection.source_identity = identity.clone();
        InputAnalysis {
            input_type: ProjectInputType::Video,
            video: Some(video),
            image_sequence: None,
            projection_detection: detection,
        }
    };
    if crate::video::panorama::source_identity(input)? != identity {
        return Err(SplatError::Process(
            "素材文件已发生变化，请重新分析后再生成".into(),
        ));
    }
    if cache.len() >= 8 {
        cache.pop_front();
    }
    cache.push_back((identity, value.clone()));
    Ok(value)
}

#[tauri::command]
pub async fn analyze_input(app: tauri::AppHandle, path: String) -> Result<InputAnalysis> {
    analyze_media(&paths_for_app(&app), Path::new(&path)).await
}

async fn input_analysis_for_plan(
    engines: &EnginePaths,
    input: &Path,
    previous: Option<InputAnalysis>,
) -> Result<InputAnalysis> {
    // Same identity contract as the legacy VideoAnalysis interface. Drafts can
    // replan without a second read even after the bounded process cache evicts them.
    if let Some(previous) = previous {
        let shape_matches = match previous.input_type {
            ProjectInputType::Images => {
                input.is_dir() && previous.image_sequence.is_some() && previous.video.is_none()
            }
            ProjectInputType::Video => {
                input.is_file() && previous.video.is_some() && previous.image_sequence.is_none()
            }
        };
        if shape_matches
            && previous.projection_detection.source_identity
                == crate::video::panorama::source_identity(input)?
        {
            return Ok(previous);
        }
    }
    analyze_media(engines, input).await
}

async fn video_analysis_for_plan(
    ffprobe: &Path,
    ffmpeg: &Path,
    input: &Path,
    previous: Option<VideoAnalysis>,
) -> Result<VideoAnalysis> {
    if let Some(previous) = previous {
        if previous.projection_detection.source_identity
            == crate::video::panorama::source_identity(input)?
        {
            return Ok(previous);
        }
    }
    let manager = ProcessManager::new();
    let (video, json) =
        crate::engines::ffprobe::probe_video_metadata(ffprobe, input, None, &manager).await?;
    let mut projection_detection = crate::video::panorama::detect_metadata(
        &json,
        &video,
        input
            .extension()
            .is_some_and(|s| s.eq_ignore_ascii_case("mp4")),
    );
    projection_detection.source_identity = crate::video::panorama::source_identity(input)?;
    let _ = ffmpeg;
    Ok(VideoAnalysis {
        video,
        projection_detection,
    })
}

pub(crate) fn paths_for_app(app: &tauri::AppHandle) -> EnginePaths {
    EnginePaths::discover(app.path().resource_dir().ok().as_deref())
}

#[tauri::command]
/// 组件就绪状态（媒体 / 重建 / 训练）。
///
/// 命令名是**中性**的：前端不需要知道背后是 COLMAP 还是自研后端。
pub async fn check_components(app: tauri::AppHandle) -> Vec<ComponentStatus> {
    paths_for_app(&app).check_all().await
}

#[tauri::command]
/// 当前加速状态（与后端无关的中性结构）。
pub async fn get_acceleration(app: tauri::AppHandle) -> AccelerationStatus {
    refresh_acceleration(&paths_for_app(&app)).await
}

#[tauri::command]
pub async fn probe_and_plan(
    app: tauri::AppHandle,
    path: String,
    quality: Quality,
    planner_enabled: Option<bool>,
    video_analysis: Option<VideoAnalysis>,
    input_analysis: Option<InputAnalysis>,
) -> std::result::Result<ProbeAndPlan, SplatError> {
    let engine_paths = paths_for_app(&app);
    let samples = catalog::runtime_samples().await;
    let input = PathBuf::from(path);
    let planner_enabled = planner_enabled.unwrap_or(true);
    if input.is_dir() {
        let analysis = input_analysis_for_plan(&engine_paths, &input, input_analysis).await?;
        let image_sequence = analysis
            .image_sequence
            .ok_or_else(|| SplatError::InvalidPath(input.clone()))?;
        let plan = create_image_plan(&image_sequence, &quality.preset());
        let acceleration = detect_acceleration(&engine_paths).await;
        let resolution = planner_enabled.then(|| {
            resolve_planner_resolution_plan(
                quality,
                acceleration.planning_gpu_total_memory_mb(),
                image_sequence.width,
                image_sequence.height,
                false,
            )
        });
        let brush = resolution.map_or_else(
            || {
                resolve_brush_training_preset(
                    quality,
                    false,
                    acceleration.planning_gpu_total_memory_mb(),
                    image_sequence.width.max(image_sequence.height),
                    0,
                )
            },
            |resolution| {
                resolve_brush_training_preset_for_plan(
                    quality,
                    acceleration.planning_gpu_total_memory_mb(),
                    image_sequence.width.max(image_sequence.height),
                    0,
                    &resolution,
                )
            },
        );
        let estimate = estimate_runtime_for_images_with_brush_and_resolution(
            image_sequence.image_count,
            image_sequence.width.max(image_sequence.height),
            &plan,
            quality,
            &samples,
            Some(&brush),
            resolution.map(|plan| plan.policy_version),
            resolution
                .map(|plan| plan.working_long_edge())
                .unwrap_or_else(|| image_sequence.width.max(image_sequence.height)),
        );
        Ok(ProbeAndPlan {
            input_type: ProjectInputType::Images,
            projection_detection: Some(analysis.projection_detection),
            video: None,
            image_sequence: Some(image_sequence),
            plan,
            estimate,
        })
    } else {
        let video_analysis = if input_analysis.is_some() {
            let analysis = input_analysis_for_plan(&engine_paths, &input, input_analysis).await?;
            Some(VideoAnalysis {
                video: analysis
                    .video
                    .ok_or_else(|| SplatError::InvalidPath(input.clone()))?,
                projection_detection: analysis.projection_detection,
            })
        } else {
            video_analysis
        };
        let VideoAnalysis {
            video,
            projection_detection,
        } = video_analysis_for_plan(
            &engine_paths.ffprobe,
            &engine_paths.ffmpeg,
            &input,
            video_analysis,
        )
        .await?;
        let plan = if planner_enabled {
            QualityV2FrameSelection.create_plan(&video, &quality.preset())
        } else {
            UniformRatioFrameSelection.create_plan(&video, &quality.preset())
        };
        let acceleration = detect_acceleration(&engine_paths).await;
        let (source_width, source_height) = if video.rotation.rem_euclid(180) == 90 {
            (video.height, video.width)
        } else {
            (video.width, video.height)
        };
        let resolution = planner_enabled.then(|| {
            resolve_planner_resolution_plan(
                quality,
                acceleration.planning_gpu_total_memory_mb(),
                source_width,
                source_height,
                true,
            )
        });
        let brush = resolution.map_or_else(
            || {
                resolve_brush_training_preset(
                    quality,
                    false,
                    acceleration.planning_gpu_total_memory_mb(),
                    source_width.max(source_height),
                    0,
                )
            },
            |resolution| {
                resolve_brush_training_preset_for_plan(
                    quality,
                    acceleration.planning_gpu_total_memory_mb(),
                    source_width.max(source_height),
                    0,
                    &resolution,
                )
            },
        );
        let estimate = estimate_runtime_with_brush_and_resolution(
            &video,
            &plan,
            quality,
            &samples,
            Some(&brush),
            resolution.map(|plan| plan.policy_version),
            resolution
                .map(|plan| plan.working_long_edge())
                .unwrap_or_else(|| source_width.max(source_height)),
        );
        Ok(ProbeAndPlan {
            input_type: ProjectInputType::Video,
            projection_detection: Some(projection_detection),
            video: Some(video),
            image_sequence: None,
            plan,
            estimate,
        })
    }
}

fn resume_checkpoint_fraction(state: &PipelineStateFile) -> (f64, &'static str) {
    if state.brush_complete {
        (0.98, "结果发布")
    } else if state.reconstruction_complete {
        (0.60, "Brush 训练")
    } else if state.matching_complete {
        (0.45, "相机重建")
    } else if state.features_complete {
        (0.32, "顺序匹配")
    } else if state
        .frames
        .as_ref()
        .and_then(|frames| frames.extracted_frames)
        .is_some_and(|count| count > 0)
    {
        (0.20, "特征提取")
    } else {
        (0.0, "画面提取")
    }
}

fn resolve_checkpoint_brush(
    state: &PipelineStateFile,
    quality: Quality,
    detected_total_memory_mb: Option<u64>,
    source_long_edge: u32,
    initial_sfm_points: u64,
) -> Result<crate::presets::ResolvedBrushTrainingPreset> {
    if let Some(saved) = state.brush_training.resolved {
        return Ok(saved);
    }
    if let Some(panorama) = &state.panorama {
        return crate::presets::resolve_panorama_training_preset(
            quality,
            state.planner_enabled,
            detected_total_memory_mb,
            panorama.face_size,
            initial_sfm_points,
        );
    }
    Ok(state.resolution_plan.map_or_else(
        || {
            resolve_brush_training_preset(
                quality,
                state.planner_enabled,
                detected_total_memory_mb,
                source_long_edge,
                initial_sfm_points,
            )
        },
        |resolution| {
            resolve_brush_training_preset_for_plan(
                quality,
                detected_total_memory_mb,
                source_long_edge,
                initial_sfm_points,
                &resolution,
            )
        },
    ))
}

#[tauri::command]
pub async fn estimate_project_runtime(
    app: tauri::AppHandle,
    project_id: Uuid,
) -> Result<RuntimeEstimate> {
    let (project, metadata) = catalog::load_registered_project(project_id).await?;
    let state_bytes = tokio::fs::read(project.join("state.json")).await?;
    let state: PipelineStateFile = serde_json::from_slice(&state_bytes)?;
    let saved_plan = state.frames.as_ref().map(|frames| FramePlan {
        retention_ratio: frames.retention_ratio,
        sampling_fps: frames.sampling_fps,
        estimated_frames: frames
            .extracted_frames
            .unwrap_or(frames.estimated_frames)
            .max(1),
        selected_frames: frames.selected_frames.clone(),
        candidate_frames: frames.candidate_frames.clone(),
        rescue_max_frames: frames.rescue_max_frames,
        ..FramePlan::default()
    });
    let samples = catalog::runtime_samples().await;
    let acceleration = detect_acceleration(&paths_for_app(&app)).await;
    let detected_total_memory_mb = acceleration.planning_gpu_total_memory_mb();
    let mut estimate = match metadata.input_type {
        ProjectInputType::Video => {
            let video = match state.video.clone() {
                Some(video) => video,
                None => {
                    probe_video(
                        &paths_for_app(&app).ffprobe,
                        &metadata.source_path,
                        None,
                        &ProcessManager::new(),
                    )
                    .await?
                }
            };
            let plan = saved_plan.unwrap_or_else(|| {
                UniformRatioFrameSelection.create_plan(&video, &metadata.quality.preset())
            });
            let brush = state.brush_training.resolved.map(Ok).unwrap_or_else(|| {
                resolve_checkpoint_brush(
                    &state,
                    metadata.quality,
                    detected_total_memory_mb,
                    video.width.max(video.height),
                    metadata
                        .output
                        .as_ref()
                        .map(|output| output.points_3d)
                        .unwrap_or(0),
                )
            })?;
            estimate_runtime_with_brush_and_resolution(
                &video,
                &plan,
                metadata.quality,
                &samples,
                Some(&brush),
                state.resolution_policy_version,
                state
                    .resolution_plan
                    .map(|resolution| resolution.working_long_edge())
                    .unwrap_or_else(|| video.width.max(video.height)),
            )
        }
        ProjectInputType::Images => {
            let image_sequence = match state.image_sequence.clone() {
                Some(info) => info,
                None => tokio::task::spawn_blocking({
                    let source = metadata.source_path.clone();
                    move || analyze_image_sequence(&source)
                })
                .await
                .map_err(|error| SplatError::Process(format!("图片序列分析任务失败：{error}")))??,
            };
            let plan = saved_plan
                .unwrap_or_else(|| create_image_plan(&image_sequence, &metadata.quality.preset()));
            let brush = state.brush_training.resolved.map(Ok).unwrap_or_else(|| {
                resolve_checkpoint_brush(
                    &state,
                    metadata.quality,
                    detected_total_memory_mb,
                    image_sequence.width.max(image_sequence.height),
                    metadata
                        .output
                        .as_ref()
                        .map(|output| output.points_3d)
                        .unwrap_or(0),
                )
            })?;
            estimate_runtime_for_images_with_brush_and_resolution(
                image_sequence.image_count,
                image_sequence.width.max(image_sequence.height),
                &plan,
                metadata.quality,
                &samples,
                Some(&brush),
                state.resolution_policy_version,
                state
                    .resolution_plan
                    .map(|resolution| resolution.working_long_edge())
                    .unwrap_or_else(|| image_sequence.width.max(image_sequence.height)),
            )
        }
    };
    let previous_duration = metadata.duration_ms.unwrap_or(0);
    let (completed_fraction, next_stage) = resume_checkpoint_fraction(&state);
    let remaining_fraction = 1.0 - completed_fraction;
    let remaining = |total: u64| ((total as f64 * remaining_fraction).round() as u64).max(1_000);
    estimate.estimated_ms = previous_duration.saturating_add(remaining(estimate.estimated_ms));
    estimate.lower_bound_ms = previous_duration.saturating_add(remaining(estimate.lower_bound_ms));
    estimate.upper_bound_ms = previous_duration.saturating_add(remaining(estimate.upper_bound_ms));
    estimate.basis = format!(
        "{}；已计入此前耗时，预计从{next_stage}阶段继续",
        estimate.basis
    );
    Ok(estimate)
}

#[tauri::command]
pub async fn get_project_overview(
    state: State<'_, PipelineController>,
) -> std::result::Result<ProjectOverview, SplatError> {
    let mut overview = catalog::get_overview().await?;
    state
        .reconcile_projects(&overview.projects)
        .await
        .map_err(|error| SplatError::Process(error.to_string()))?;
    let tasks = state
        .all()
        .await
        .map_err(|e| SplatError::Process(e.to_string()))?;
    for project in &mut overview.projects {
        if let Some(task) = tasks.iter().find(|t| t.project_id == Some(project.id)) {
            project.status = task.status.project_status();
        }
    }
    if state.active.lock().await.is_none() {
        for project in &mut overview.projects {
            if project.status == ProjectStatus::Running {
                project.status = ProjectStatus::Interrupted;
            }
        }
    }
    Ok(overview)
}

fn persisted_task_stage(state: &PipelineStateFile, status: ProjectStatus) -> PipelineStage {
    match status {
        ProjectStatus::Completed => PipelineStage::Completed,
        _ if state.brush_complete => PipelineStage::Exporting,
        _ if state.reconstruction_complete => PipelineStage::TrainingSplats,
        _ if state.matching_complete => PipelineStage::Reconstructing,
        _ if state.features_complete => PipelineStage::Matching,
        _ if state
            .frames
            .as_ref()
            .and_then(|frames| frames.extracted_frames)
            .is_some_and(|count| count > 0) =>
        {
            PipelineStage::ExtractingFeatures
        }
        _ => PipelineStage::ExtractingFrames,
    }
}

async fn read_project_log_tail(logs_directory: &Path) -> Result<Vec<ProjectTaskLogLine>> {
    const MAX_FILE_BYTES: u64 = 64 * 1024;
    const MAX_TOTAL_BYTES: usize = 256 * 1024;
    const MAX_LINES: usize = 500;
    let mut files = Vec::new();
    if !logs_directory.is_dir() {
        return Ok(Vec::new());
    }
    let mut entries = tokio::fs::read_dir(logs_directory).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if !entry.file_type().await?.is_file()
            || path.extension().and_then(|value| value.to_str()) != Some("log")
        {
            continue;
        }
        let metadata = entry.metadata().await?;
        let modified = metadata.modified().ok();
        files.push((modified, path, metadata.len()));
    }
    files.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    let mut lines = Vec::new();
    let mut total_bytes = 0_usize;
    for (_, path, length) in files {
        let mut file = tokio::fs::File::open(&path).await?;
        let start = length.saturating_sub(MAX_FILE_BYTES);
        file.seek(std::io::SeekFrom::Start(start)).await?;
        let mut bytes = Vec::with_capacity((length - start) as usize);
        file.read_to_end(&mut bytes).await?;
        let text = String::from_utf8_lossy(&bytes);
        let source = path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("system")
            .to_string();
        let mut file_lines = text.lines();
        if start > 0 {
            let _ = file_lines.next();
        }
        for message in file_lines {
            if message.is_empty() {
                continue;
            }
            total_bytes = total_bytes.saturating_add(message.len());
            lines.push(ProjectTaskLogLine {
                source: source.clone(),
                message: message.to_string(),
            });
            while lines.len() > MAX_LINES || total_bytes > MAX_TOTAL_BYTES {
                if let Some(removed) = lines.first() {
                    total_bytes = total_bytes.saturating_sub(removed.message.len());
                }
                lines.remove(0);
            }
        }
    }
    Ok(lines)
}

#[tauri::command]
pub async fn get_project_task_detail(
    controller: State<'_, PipelineController>,
    project_id: String,
) -> Result<ProjectTaskDetail> {
    let id = parse_project_id(&project_id)?;
    let (project_root, metadata, mut project) =
        catalog::load_registered_project_summary(id).await?;
    if let Some(task) = controller
        .all()
        .await
        .map_err(|e| SplatError::Process(e.to_string()))?
        .into_iter()
        .find(|t| t.project_id == Some(id))
    {
        project.status = task.status.project_status();
    }
    let imported_detail = metadata
        .imported
        .as_ref()
        .map(|imported| ImportedProjectDetail {
            source_model_path: imported.source_model_path.clone(),
            format: imported.source_format,
            source_available: project.source_available.unwrap_or(false),
            colmap_root: imported
                .colmap
                .as_ref()
                .map(|value| value.root_path.clone()),
            sparse_model_path: imported
                .colmap
                .as_ref()
                .map(|value| value.sparse_model_path.clone()),
            database_available: imported
                .colmap
                .as_ref()
                .and_then(|value| value.database_path.as_ref())
                .is_some_and(|path| path.is_file()),
            images_available: imported
                .colmap
                .as_ref()
                .and_then(|value| value.images_path.as_ref())
                .is_some_and(|path| path.is_dir()),
            reshoot_eligible: imported
                .colmap
                .as_ref()
                .is_some_and(|value| value.reshoot_eligible),
            reshoot_reason: imported
                .colmap
                .as_ref()
                .and_then(|value| value.reshoot_reason.clone()),
        });
    let is_imported = imported_detail.is_some();
    let state = if is_imported {
        None
    } else {
        let state_bytes = tokio::fs::read(project_root.join("state.json")).await?;
        Some(serde_json::from_slice::<PipelineStateFile>(&state_bytes)?)
    };
    let progress = state.as_ref().map(|state| {
        if project.status == ProjectStatus::Completed {
            100.0
        } else {
            resume_checkpoint_fraction(state).0 * 100.0
        }
    });
    let stage = state
        .as_ref()
        .map(|state| persisted_task_stage(state, project.status));
    let logs = if is_imported {
        Vec::new()
    } else {
        read_project_log_tail(&project_root.join("logs")).await?
    };
    let input_images = metadata.output.as_ref().map(|output| output.input_images);
    let registered_images = metadata
        .output
        .as_ref()
        .map(|output| output.registered_images);
    Ok(ProjectTaskDetail {
        project,
        input_type: (!is_imported).then_some(metadata.input_type),
        video_projection: project_video_projection(state.as_ref(), metadata.input_type),
        input_projection: if is_imported {
            None
        } else if metadata.input_type == ProjectInputType::Images {
            Some(if state.as_ref().is_some_and(|s| s.panorama.is_some()) {
                crate::video::panorama::VideoProjection::Equirectangular
            } else {
                crate::video::panorama::VideoProjection::Perspective
            })
        } else {
            project_video_projection(state.as_ref(), metadata.input_type)
        },
        source_path: metadata.source_path,
        projects_root: project_root
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default(),
        planner_enabled: state.as_ref().map(|state| state.planner_enabled),
        estimated_frames: state
            .as_ref()
            .and_then(|state| state.frames.as_ref())
            .map(|frames| frames.estimated_frames),
        stage,
        progress,
        input_images,
        registered_images,
        video: state.as_ref().and_then(|state| state.video.clone()),
        image_sequence: state
            .as_ref()
            .and_then(|state| state.image_sequence.clone()),
        source_project_id: metadata
            .reshoot
            .as_ref()
            .map(|reshoot| reshoot.source_project_id.to_string()),
        logs,
        imported: imported_detail,
    })
}

fn project_video_projection(
    state: Option<&PipelineStateFile>,
    input_type: ProjectInputType,
) -> Option<crate::video::panorama::VideoProjection> {
    use crate::video::panorama::VideoProjection;
    if input_type != ProjectInputType::Video {
        return None;
    }
    let state = state?;
    if state.panorama.is_some() {
        return Some(VideoProjection::Equirectangular);
    }
    match state.video_projection {
        VideoProjection::Equirectangular => Some(VideoProjection::Equirectangular),
        VideoProjection::Perspective => Some(VideoProjection::Perspective),
        VideoProjection::Auto if state.video.is_some() => Some(VideoProjection::Perspective),
        VideoProjection::Auto => None,
    }
}

#[tauri::command]
pub async fn set_projects_root(
    projects_root: String,
) -> std::result::Result<AppSettings, SplatError> {
    catalog::save_projects_root(PathBuf::from(projects_root)).await
}

#[tauri::command]
pub async fn set_planner_enabled(enabled: bool) -> std::result::Result<AppSettings, SplatError> {
    catalog::save_planner_enabled(enabled).await
}

#[tauri::command]
pub async fn initialize_telemetry(
    telemetry: State<'_, TelemetryService>,
) -> std::result::Result<TelemetryPreferences, SplatError> {
    telemetry.initialize().await
}

#[tauri::command]
pub async fn set_telemetry_consent(
    telemetry: State<'_, TelemetryService>,
    enabled: bool,
) -> std::result::Result<TelemetryPreferences, SplatError> {
    telemetry.set_consent(enabled).await
}

#[cfg(test)]
async fn record_command_failure(
    state: &PipelineController,
    error: PipelineCommandError,
    paths: &[PathBuf],
) -> PipelineCommandError {
    state.capture_failure(error, paths).await
}

fn service_command_error(error: crate::tasks::ServiceError) -> PipelineCommandError {
    let mut command = PipelineCommandError::from(SplatError::Process(error.to_string()));
    if error.code == "TASK_BUSY" {
        command.code = "TASK_BUSY";
    }
    command
}

async fn wait_task_result(
    state: &PipelineController,
    id: Uuid,
) -> std::result::Result<PipelineResult, PipelineCommandError> {
    let task = state.wait(id).await.map_err(service_command_error)?;
    if let Some(result) = task.result {
        return serde_json::from_value(result)
            .map_err(|e| PipelineCommandError::from(SplatError::Json(e)));
    }
    if let Some(error) = task.error {
        return Err(PipelineCommandError {
            code: if task.status == crate::tasks::TaskStatus::Cancelled {
                "cancelled"
            } else {
                "pipeline_failed"
            },
            message: error.message,
            failed_stage: error.failed_stage,
            component: error.component,
            failure_kind: None,
            project_id: task.project_id.map(|v| v.to_string()),
            project_path: task.project_path.map(Box::new),
            logs_directory: None,
            failure_id: error.failure_id.map(Box::new),
        });
    }
    Err(PipelineCommandError::from(SplatError::Process(
        "任务未产生结果".into(),
    )))
}

#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn create_gui_task(
    state: State<'_, PipelineController>,
    path: String,
    quality: Quality,
    projects_root: String,
    planner_enabled: Option<bool>,
    workspace_task_id: Option<String>,
    video_projection: Option<crate::video::panorama::VideoProjection>,
    video_source_identity: Option<String>,
    input_projection: Option<crate::video::panorama::VideoProjection>,
    input_source_identity: Option<String>,
    image_matching_strategy: Option<crate::project::ImageMatchingStrategy>,
) -> std::result::Result<crate::tasks::TaskRecord, crate::tasks::ServiceError> {
    let video_projection = input_projection.or(video_projection);
    let video_source_identity = input_source_identity.or(video_source_identity);
    if let Some(expected) = &video_source_identity {
        let identity = crate::video::panorama::source_identity(std::path::Path::new(&path))
            .map_err(|e| crate::tasks::ServiceError::new("INPUT_CHANGED", e.to_string()))?;
        if &identity != expected {
            return Err(crate::tasks::ServiceError::new(
                "INPUT_CHANGED",
                "素材文件已发生变化，请重新分析后再生成",
            ));
        }
    }
    let id = workspace_task_id
        .map(|v| Uuid::parse_str(&v))
        .transpose()
        .map_err(|_| crate::tasks::ServiceError::new("INVALID_ARGUMENT", "工作区任务 ID 无效"))?;
    state
        .create(
            id,
            path.into(),
            quality,
            projects_root.into(),
            planner_enabled.unwrap_or(true),
            "gui",
            None,
            Some(serde_json::json!({"videoProjection":video_projection.unwrap_or_default(), "videoSourceIdentity":video_source_identity, "imageMatchingStrategy":image_matching_strategy.unwrap_or_default()})),
        )
        .await
}
#[tauri::command]
pub async fn start_gui_task(
    app: tauri::AppHandle,
    state: State<'_, PipelineController>,
    telemetry: State<'_, TelemetryService>,
    imports: State<'_, crate::imports::ImportController>,
    task_id: Uuid,
) -> std::result::Result<crate::tasks::StartReceipt, crate::tasks::ServiceError> {
    let _admission = imports.admission().await;
    if imports.is_active().await {
        return Err(crate::tasks::ServiceError::new(
            "TASK_BUSY",
            "已有 Gaussian 导入正在进行，请等待完成后再启动任务",
        ));
    }
    state
        .start(
            task_id,
            paths_for_app(&app),
            Some(telemetry.inner().clone()),
        )
        .await
}
#[tauri::command]
pub async fn get_shared_tasks(
    state: State<'_, PipelineController>,
) -> std::result::Result<Vec<crate::tasks::TaskRecord>, crate::tasks::ServiceError> {
    state.all().await
}
#[tauri::command]
pub async fn read_shared_task_logs(
    state: State<'_, PipelineController>,
    request: crate::tasks::logs::LogRequest,
) -> std::result::Result<crate::tasks::logs::LogPage, crate::tasks::ServiceError> {
    let task = state.get(request.task_id).await?;
    tokio::task::spawn_blocking(move || crate::tasks::logs::read(&task, &request))
        .await
        .map_err(|_| crate::tasks::ServiceError::new("LOG_READ_FAILED", "日志读取异常"))?
}
#[tauri::command]
pub async fn get_shared_task(
    state: State<'_, PipelineController>,
    task_id: Uuid,
) -> std::result::Result<crate::tasks::TaskRecord, crate::tasks::ServiceError> {
    state.get(task_id).await
}
#[tauri::command]
pub async fn cancel_shared_task(
    state: State<'_, PipelineController>,
    task_id: Uuid,
    run_id: Uuid,
) -> std::result::Result<crate::tasks::TaskRecord, crate::tasks::ServiceError> {
    state.cancel(task_id, run_id).await
}
#[tauri::command]
pub async fn resume_gui_task(
    app: tauri::AppHandle,
    state: State<'_, PipelineController>,
    telemetry: State<'_, TelemetryService>,
    imports: State<'_, crate::imports::ImportController>,
    project_id: Uuid,
) -> std::result::Result<crate::tasks::StartReceipt, crate::tasks::ServiceError> {
    let _admission = imports.admission().await;
    if imports.is_active().await {
        return Err(crate::tasks::ServiceError::new(
            "TASK_BUSY",
            "已有 Gaussian 导入正在进行，请等待完成后再继续任务",
        ));
    }
    state
        .resume(
            project_id,
            paths_for_app(&app),
            Some(telemetry.inner().clone()),
        )
        .await
}

#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn start_pipeline(
    app: tauri::AppHandle,
    state: State<'_, PipelineController>,
    telemetry: State<'_, TelemetryService>,
    imports: State<'_, crate::imports::ImportController>,
    path: String,
    quality: Quality,
    projects_root: String,
    planner_enabled: Option<bool>,
    workspace_task_id: Option<String>,
) -> std::result::Result<PipelineResult, PipelineCommandError> {
    let _admission = imports.admission().await;
    if imports.is_active().await {
        return Err(PipelineCommandError::from(SplatError::Process(
            "已有 Gaussian 导入正在进行，请等待完成后再启动任务".into(),
        )));
    }
    let id = workspace_task_id
        .map(|v| Uuid::parse_str(&v))
        .transpose()
        .map_err(|_| {
            PipelineCommandError::from(SplatError::Process("工作区任务 ID 无效".into()))
        })?;
    let task = state
        .create(
            id,
            path.into(),
            quality,
            projects_root.into(),
            planner_enabled.unwrap_or(true),
            "gui",
            None,
            None,
        )
        .await
        .map_err(service_command_error)?;
    state
        .start(
            task.task_id,
            paths_for_app(&app),
            Some(telemetry.inner().clone()),
        )
        .await
        .map_err(service_command_error)?;
    wait_task_result(state.inner(), task.task_id).await
}
#[tauri::command]
pub async fn resume_pipeline(
    app: tauri::AppHandle,
    state: State<'_, PipelineController>,
    telemetry: State<'_, TelemetryService>,
    project_id: String,
) -> std::result::Result<PipelineResult, PipelineCommandError> {
    let id = parse_project_id(&project_id)?;
    let receipt = state
        .resume(id, paths_for_app(&app), Some(telemetry.inner().clone()))
        .await
        .map_err(service_command_error)?;
    wait_task_result(state.inner(), receipt.task_id).await
}
#[tauri::command]
pub async fn cancel_pipeline(state: State<'_, PipelineController>) -> Result<()> {
    if let Some(task) = state
        .running()
        .await
        .map_err(|e| SplatError::Process(e.to_string()))?
    {
        if let Some(run) = task.run_id {
            state
                .cancel(task.task_id, run)
                .await
                .map_err(|e| SplatError::Process(e.to_string()))?;
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn get_app_runtime_status(
    pipeline: State<'_, PipelineController>,
    preview: State<'_, PreviewController>,
) -> Result<AppRuntimeStatus> {
    let active_pipeline = pipeline.active.lock().await;
    let pipeline_running = active_pipeline.is_some();
    let pipeline_identity = active_pipeline
        .as_ref()
        .and_then(|runner| runner.current_project_identity());
    let pipeline_run_elapsed_ms = active_pipeline
        .as_ref()
        .map(|runner| runner.elapsed_ms())
        .unwrap_or(0);
    let pipeline_elapsed_offset_ms = active_pipeline
        .as_ref()
        .map(|runner| runner.elapsed_offset_ms())
        .unwrap_or(0);
    let acceleration = active_pipeline
        .as_ref()
        .and_then(|runner| runner.current_acceleration());
    drop(active_pipeline);
    let active_task = pipeline
        .running()
        .await
        .map_err(|e| SplatError::Process(e.to_string()))?;
    let preview_project_id = preview
        .active
        .lock()
        .await
        .as_ref()
        .map(|session| session.project_id.to_string());
    Ok(AppRuntimeStatus {
        pipeline_running,
        pipeline_project_id: pipeline_identity.map(|value| value.0.to_string()),
        pipeline_workspace_task_id: active_task.map(|task| task.task_id.to_string()),
        pipeline_run_elapsed_ms,
        pipeline_elapsed_offset_ms,
        preview_project_id,
        acceleration,
    })
}

#[tauri::command]
pub async fn open_project_location(
    app: tauri::AppHandle,
    project_id: String,
    location: ProjectLocation,
) -> Result<()> {
    let id = parse_project_id(&project_id)?;
    let (project, _) = catalog::load_registered_project(id).await?;
    let target = match location {
        ProjectLocation::Project => project,
        ProjectLocation::Logs => project.join("logs"),
        ProjectLocation::Source => {
            let (_, metadata) = catalog::load_registered_project(id).await?;
            let source = metadata
                .imported
                .ok_or_else(|| SplatError::Process("该项目没有外部源文件".into()))?
                .source_model_path;
            source
                .parent()
                .map(Path::to_path_buf)
                .ok_or(SplatError::InvalidPath(source))?
        }
    };
    if !target.is_dir() {
        return Err(SplatError::InvalidPath(target));
    }
    app.opener()
        .open_path(target.to_string_lossy().into_owned(), None::<String>)
        .map_err(|error| SplatError::Process(format!("无法打开文件夹：{error}")))
}

#[tauri::command]
pub async fn delete_project(
    app: tauri::AppHandle,
    state: State<'_, PipelineController>,
    preview: State<'_, PreviewController>,
    project_id: String,
) -> Result<()> {
    if state.active.lock().await.is_some() {
        return Err(SplatError::Process("任务运行期间不能删除项目".into()));
    }
    let id =
        Uuid::parse_str(&project_id).map_err(|_| SplatError::Process("项目 ID 无效".into()))?;
    let _lifecycle = preview.lifecycle.lock().await;
    if preview
        .html_export
        .lock()
        .await
        .as_ref()
        .is_some_and(|session| session.project_id == id)
        || preview
            .video_export
            .lock()
            .await
            .as_ref()
            .is_some_and(|session| session.project_id == id)
    {
        return Err(SplatError::Process("请先完成或取消导出再删除项目".into()));
    }
    let session = {
        let mut active = preview.active.lock().await;
        if active
            .as_ref()
            .is_some_and(|session| session.project_id == id)
        {
            active.take()
        } else {
            None
        }
    };
    if let Some(session) = session {
        discard_preview_asset(&app, session).await;
    }

    let mut edit_save = preview.edit_save.lock().await;
    if edit_save
        .as_ref()
        .is_some_and(|session| session.project_id == id)
    {
        *edit_save = None;
    }
    drop(edit_save);
    catalog::delete_project(id).await?;
    state
        .mark_project_deleted(id)
        .await
        .map_err(|e| SplatError::Process(e.to_string()))
}

fn parse_project_id(project_id: &str) -> Result<Uuid> {
    Uuid::parse_str(project_id).map_err(|_| SplatError::Process("项目 ID 无效".into()))
}

fn preview_client_path(path: &Path) -> PathBuf {
    let value = path.to_string_lossy();
    if let Some(unc) = value.strip_prefix(r"\\?\UNC\") {
        PathBuf::from(format!(r"\\{unc}"))
    } else if let Some(local) = value.strip_prefix(r"\\?\") {
        PathBuf::from(local)
    } else {
        path.to_path_buf()
    }
}

async fn create_preview_asset(project_root: &Path, source: &Path) -> Result<PathBuf> {
    let directory = project_root.join("work").join("preview");
    tokio::fs::create_dir_all(&directory).await?;
    let asset_path = directory.join(format!("preview-{}.ply", Uuid::new_v4().simple()));

    // A per-session hard link gives the asset protocol a fresh URL without copying a large PLY.
    // Some network filesystems do not support hard links, so retain a copy fallback for UNC roots.
    if let Err(link_error) = tokio::fs::hard_link(source, &asset_path).await {
        tokio::fs::copy(source, &asset_path)
            .await
            .map_err(|copy_error| {
                SplatError::Process(format!(
                    "无法准备高斯泼溅预览文件（硬链接：{link_error}；复制：{copy_error}）"
                ))
            })?;
    }
    Ok(asset_path)
}

async fn discard_preview_asset(app: &tauri::AppHandle, session: GaussianPreviewSession) {
    // `forbid_file` is permanent for the lifetime of this Tauri scope. It is therefore only safe
    // for the unique session alias, never for final.ply, which must be previewable again later.
    let parent = session
        .asset_paths
        .first()
        .and_then(|path| path.parent())
        .map(Path::to_path_buf);
    for asset_path in session.asset_paths {
        let _ = app.asset_protocol_scope().forbid_file(&asset_path);
        let _ = tokio::fs::remove_file(asset_path).await;
    }
    if let Some(parent) = parent {
        let _ = tokio::fs::remove_dir(parent).await;
    }
}

#[tauri::command]
pub async fn prepare_gaussian_preview(
    app: tauri::AppHandle,
    state: State<'_, PreviewController>,
    project_id: String,
) -> Result<GaussianPreviewDescriptor> {
    let _lifecycle = state.lifecycle.lock().await;
    if state.html_export.lock().await.is_some()
        || state.video_export.lock().await.is_some()
        || state.model_export.lock().await.is_some()
    {
        return Err(SplatError::Process(
            "导出期间不能重新加载或切换预览项目".into(),
        ));
    }
    let id = parse_project_id(&project_id)?;
    let (project_root, path, metadata) = catalog::registered_final_ply_for_project(id).await?;
    let reference_root = project_root.clone();
    let imported = metadata.imported.is_some();
    let (reference_view, reference_view_unavailable) = tokio::task::spawn_blocking(move || {
        crate::reconstruction::reference_view::training_reference_view(&reference_root, imported)
    })
    .await
    .unwrap_or((None, Some("unavailable")));
    let info = inspect_gaussian_ply(&path)?;
    let transform = metadata.transform.validate()?;
    let editing = metadata.editing.validate(info.splat_count)?;
    let edit_mask = read_mask(&project_root, editing).map_err(|error| {
        SplatError::Process(format!(
            "Gaussian 编辑数据已损坏或与源文件不兼容：{error}。可在预览中重置编辑状态后恢复。"
        ))
    })?;

    let previous = state.active.lock().await.take();
    if let Some(previous) = previous {
        discard_preview_asset(&app, previous).await;
    }
    let direct_external = metadata
        .imported
        .as_ref()
        .is_some_and(|imported| imported.canonical_model_path.is_none());
    let read_token = direct_external.then(Uuid::new_v4);
    let asset_path = if direct_external {
        path.clone()
    } else {
        let asset = create_preview_asset(&project_root, &path).await?;
        app.asset_protocol_scope()
            .allow_file(&asset)
            .map_err(|error| {
                let _ = std::fs::remove_file(&asset);
                SplatError::Process(format!("无法开放本地 PLY 预览资源：{error}"))
            })?;
        asset
    };
    let mut asset_paths = if direct_external {
        Vec::new()
    } else {
        vec![asset_path.clone()]
    };
    let edit_mask_asset_path = if editing.revision > 0 || editing.deleted_count > 0 {
        let directory = project_root.join("work").join("preview");
        tokio::fs::create_dir_all(&directory).await?;
        let path = directory.join(format!("preview-{}.mask.bin", Uuid::new_v4().simple()));
        tokio::fs::write(&path, &edit_mask).await?;
        app.asset_protocol_scope()
            .allow_file(&path)
            .map_err(|error| {
                let _ = std::fs::remove_file(&path);
                SplatError::Process(format!("无法开放 Gaussian 编辑位图资源：{error}"))
            })?;
        asset_paths.push(path.clone());
        Some(preview_client_path(&path))
    } else {
        None
    };
    *state.active.lock().await = Some(GaussianPreviewSession {
        project_id: id,
        asset_paths,
        read_token,
        read_path: direct_external.then_some(path.clone()),
        read_size: direct_external.then_some(info.file_size),
        read_modified_ms: metadata
            .imported
            .as_ref()
            .filter(|_| direct_external)
            .and_then(|imported| imported.source_fingerprint.modified_ms),
    });

    Ok(GaussianPreviewDescriptor {
        project_id: id,
        model_path: preview_client_path(&path),
        asset_path: preview_client_path(&asset_path),
        format: "ply",
        file_size: info.file_size,
        splat_count: info.splat_count,
        transform,
        editing,
        edit_mask_asset_path,
        read_token,
        reference_view,
        reference_view_unavailable,
    })
}

#[tauri::command]
pub async fn read_gaussian_preview_chunk(
    state: State<'_, PreviewController>,
    request: tauri::ipc::Request<'_>,
) -> Result<tauri::ipc::Response> {
    let token = request
        .headers()
        .get("x-ooosplat-preview-token")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| Uuid::parse_str(value).ok())
        .ok_or_else(|| SplatError::Process("Gaussian 预览读取令牌无效".into()))?;
    let offset = request
        .headers()
        .get("x-ooosplat-offset")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| SplatError::Process("Gaussian 预览读取偏移无效".into()))?;
    let length = request
        .headers()
        .get("x-ooosplat-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1_024_000)
        .min(1_024_000);
    let active = state.active.lock().await;
    let session = active
        .as_ref()
        .filter(|session| session.read_token == Some(token))
        .ok_or_else(|| SplatError::Process("Gaussian 预览读取会话已结束".into()))?;
    let path = session
        .read_path
        .clone()
        .ok_or_else(|| SplatError::Process("Gaussian 预览读取会话已结束".into()))?;
    let file_size = session
        .read_size
        .ok_or_else(|| SplatError::Process("Gaussian 预览读取会话已结束".into()))?;
    let expected_modified_ms = session.read_modified_ms;
    if length == 0 || offset >= file_size {
        return Err(SplatError::Process("Gaussian 预览读取范围无效".into()));
    }
    let remaining = usize::try_from(file_size - offset).unwrap_or(usize::MAX);
    let length = length.min(remaining);
    let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
        let metadata = path.metadata()?;
        let modified_ms = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|value| value.as_millis() as u64);
        if metadata.len() != file_size
            || expected_modified_ms.is_some_and(|expected| modified_ms != Some(expected))
        {
            return Err(SplatError::Process(
                "外部 Gaussian 文件在预览加载期间发生变化，请重新打开项目".into(),
            ));
        }
        let mut file = std::fs::File::open(path)?;
        use std::io::{Read, Seek};
        file.seek(std::io::SeekFrom::Start(offset))?;
        let mut bytes = vec![0_u8; length];
        let read = file.read(&mut bytes)?;
        bytes.truncate(read);
        Ok(bytes)
    })
    .await
    .map_err(|error| SplatError::Process(error.to_string()))??;
    Ok(tauri::ipc::Response::new(bytes))
}

#[tauri::command]
pub async fn release_gaussian_preview(
    app: tauri::AppHandle,
    state: State<'_, PreviewController>,
    project_id: String,
) -> Result<()> {
    let _lifecycle = state.lifecycle.lock().await;
    let id = parse_project_id(&project_id)?;
    let model_id = state
        .model_export
        .lock()
        .await
        .as_ref()
        .filter(|session| session.project_id == id)
        .map(|session| session.id);
    if let Some(export_id) = model_id {
        model_export::cancel_gaussian_model_export(state.clone(), export_id).await?;
    }
    let mut html_export = state.html_export.lock().await;
    if let Some(session) = html_export
        .as_ref()
        .filter(|session| session.project_id == id)
    {
        session.cancel.cancel();
        if !session.running {
            *html_export = None;
        }
    }
    drop(html_export);
    let session = {
        let mut active = state.active.lock().await;
        if active
            .as_ref()
            .is_some_and(|session| session.project_id == id)
        {
            active.take()
        } else {
            None
        }
    };
    if let Some(session) = session {
        discard_preview_asset(&app, session).await;
    }

    let mut edit_save = state.edit_save.lock().await;
    if edit_save
        .as_ref()
        .is_some_and(|session| session.project_id == id)
    {
        *edit_save = None;
    }
    drop(edit_save);

    let mut video_export = state.video_export.lock().await;
    if let Some(session) = video_export
        .as_ref()
        .filter(|session| session.project_id == id)
    {
        session.cancel.cancel();
        if !session.running {
            *video_export = None;
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn begin_gaussian_edit_save(
    state: State<'_, PreviewController>,
    project_id: String,
    edit_state: GaussianEditDraft,
) -> Result<GaussianEditSaveReservation> {
    let project_id = parse_project_id(&project_id)?;
    let (_, source, metadata) = catalog::registered_final_ply_for_project(project_id).await?;
    let info = inspect_gaussian_ply(&source)?;
    let current = metadata.editing.validate(info.splat_count)?;
    if current.revision != edit_state.base_revision {
        return Err(SplatError::Process(format!(
            "Gaussian 编辑状态已更新（当前修订 {}，提交基于 {}），请重新加载后再试",
            current.revision, edit_state.base_revision
        )));
    }
    if let Some(crop) = edit_state.crop {
        crop.validate()?;
    }
    if !state
        .active
        .lock()
        .await
        .as_ref()
        .is_some_and(|session| session.project_id == project_id)
    {
        return Err(SplatError::Process(
            "项目当前未在预览中打开，无法保存编辑".into(),
        ));
    }
    let mut active = state.edit_save.lock().await;
    if active.is_some() {
        return Err(SplatError::Process("已有 Gaussian 编辑正在保存".into()));
    }
    let edit_id = Uuid::new_v4();
    let next_revision = current.revision.saturating_add(1);
    *active = Some(GaussianEditSaveSession {
        edit_id,
        project_id,
        base_revision: current.revision,
        next_revision,
        splat_count: info.splat_count,
        crop: edit_state.crop,
    });
    Ok(GaussianEditSaveReservation {
        edit_id,
        expected_mask_bytes: packed_mask_bytes(info.splat_count)?,
    })
}

#[tauri::command]
pub async fn commit_gaussian_edit_save(
    state: State<'_, PreviewController>,
    request: tauri::ipc::Request<'_>,
) -> Result<GaussianEditing> {
    let edit_id = request
        .headers()
        .get("x-ooosplat-edit-id")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| Uuid::parse_str(value).ok())
        .ok_or_else(|| SplatError::Process("Gaussian 编辑保存令牌无效".into()))?;
    let mask = match request.body() {
        InvokeBody::Raw(bytes) => bytes.as_slice(),
        _ => {
            return Err(SplatError::Process(
                "Gaussian 编辑位图必须通过原始二进制 IPC 提交".into(),
            ))
        }
    };
    let mut active = state.edit_save.lock().await;
    if !active
        .as_ref()
        .is_some_and(|session| session.edit_id == edit_id)
    {
        return Err(SplatError::Process(
            "Gaussian 编辑保存会话不存在或已结束".into(),
        ));
    }
    // Taking the matching session while retaining the mutex guard guarantees every
    // success or failure closes this token and no second save can overtake it.
    let session = active.take().expect("matching session checked above");
    if mask.len() != packed_mask_bytes(session.splat_count)? {
        return Err(SplatError::Process(
            "Gaussian 编辑位图长度与源文件不一致".into(),
        ));
    }
    let deleted_count = count_deleted(mask, session.splat_count)?;
    let _write_guard = state.metadata_write.lock().await;
    let (root, _, mut metadata) =
        catalog::registered_final_ply_for_project(session.project_id).await?;
    if metadata.editing.revision != session.base_revision {
        return Err(SplatError::Process(
            "保存期间编辑状态已发生变化，请重新加载".into(),
        ));
    }
    let editing = GaussianEditing {
        crop: session.crop,
        revision: session.next_revision,
        source_splat_count: session.splat_count,
        deleted_count,
    };
    // A process crash can leave a future revision file behind before project.json
    // was published. It is not referenced by current metadata and is safe to replace.
    let destination = mask_path(&root, editing.revision);
    if destination.exists() {
        std::fs::remove_file(&destination)?;
    }
    write_mask_atomic(&root, editing.revision, editing.source_splat_count, mask)?;
    metadata.schema_version = crate::project::metadata::schema_version();
    metadata.editing = editing;
    if let Err(error) = atomic_write_json(&root.join("project.json"), &metadata).await {
        let _ = std::fs::remove_file(mask_path(&root, editing.revision));
        return Err(error);
    }
    cleanup_old_masks(&root, editing.revision);
    drop(active);
    Ok(editing)
}

#[tauri::command]
pub async fn reset_gaussian_edits(
    state: State<'_, PreviewController>,
    project_id: String,
) -> Result<GaussianEditing> {
    let id = parse_project_id(&project_id)?;
    let mut active = state.edit_save.lock().await;
    if active
        .as_ref()
        .is_some_and(|session| session.project_id == id)
    {
        *active = None;
    }
    drop(active);
    let _write_guard = state.metadata_write.lock().await;
    let (root, source, mut metadata) = catalog::registered_final_ply_for_project(id).await?;
    let splat_count = inspect_gaussian_ply(&source)?.splat_count;
    metadata.schema_version = crate::project::metadata::schema_version();
    metadata.editing = GaussianEditing {
        source_splat_count: splat_count,
        ..GaussianEditing::default()
    };
    atomic_write_json(&root.join("project.json"), &metadata).await?;
    // The metadata switch is the authoritative reset. Old mask files are now
    // unreachable, so a cleanup failure must not prevent reopening final.ply.
    let _ = remove_edit_files(&root);
    Ok(metadata.editing)
}

#[tauri::command]
pub async fn save_gaussian_transform(
    state: State<'_, PreviewController>,
    project_id: String,
    transform: GaussianTransform,
) -> Result<GaussianTransform> {
    let id = parse_project_id(&project_id)?;
    let transform = transform.validate()?;
    let _write_guard = state.metadata_write.lock().await;
    let (root, _, mut metadata) = catalog::registered_final_ply_for_project(id).await?;
    metadata.schema_version = crate::project::metadata::schema_version();
    metadata.model = "final.ply".into();
    metadata.transform = transform;
    atomic_write_json(&root.join("project.json"), &metadata).await?;
    Ok(transform)
}

#[tauri::command]
pub async fn export_transformed_gaussian(
    app: tauri::AppHandle,
    state: State<'_, PreviewController>,
    project_id: String,
    transform: GaussianTransform,
    edit_revision: Option<u64>,
) -> Result<GaussianExportResult> {
    let id = parse_project_id(&project_id)?;
    let transform = transform.validate()?;
    let _export_guard = state.export.lock().await;
    let (root, source, metadata) = catalog::registered_final_ply_for_project(id).await?;
    let source_info = inspect_gaussian_ply(&source)?;
    let editing = metadata.editing.validate(source_info.splat_count)?;
    if edit_revision.is_some_and(|revision| revision != editing.revision) {
        return Err(SplatError::Process(format!(
            "编辑状态尚未保存完成（导出请求修订 {:?}，当前修订 {}）",
            edit_revision, editing.revision
        )));
    }
    let mask = read_mask(&root, editing)?;
    let emitter = app.clone();
    let (path, info) = tokio::task::spawn_blocking(move || {
        export_transformed_ply_with_edits(
            &source,
            &root,
            transform,
            GaussianExportEdits {
                crop: editing.crop,
                deleted_mask: Some(&mask),
            },
            |processed, total| {
                let _ = emitter.emit(
                    "gaussian-export-progress",
                    GaussianExportProgress {
                        project_id: id,
                        processed_splats: processed,
                        total_splats: total,
                        progress: if total == 0 {
                            0.0
                        } else {
                            processed as f64 / total as f64 * 100.0
                        },
                    },
                );
            },
        )
    })
    .await
    .map_err(|error| SplatError::Process(format!("Gaussian 导出线程失败：{error}")))??;
    Ok(GaussianExportResult {
        path,
        file_size: info.file_size,
        splat_count: info.splat_count,
    })
}

const GAUSSIAN_VIDEO_FPS: u32 = 30;
const GAUSSIAN_VIDEO_DURATION_MS: u64 = 23_000;
const MAX_GAUSSIAN_VIDEO_BYTES: usize = 1024 * 1024 * 1024;

#[cfg(test)]
fn next_gaussian_video_path(root: &Path) -> PathBuf {
    next_gaussian_video_path_for_orientation(root, VideoOrientation::Portrait)
}

fn next_gaussian_video_path_for_orientation(root: &Path, orientation: VideoOrientation) -> PathBuf {
    let stem = match orientation {
        VideoOrientation::Portrait => "preview",
        VideoOrientation::Landscape => "preview-landscape",
    };
    for number in 1_u32.. {
        let name = if number == 1 {
            format!("{stem}.mp4")
        } else {
            format!("{stem}-{number}.mp4")
        };
        let candidate = root.join(name);
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!("video export numbering is unbounded")
}

fn contains_mp4_ftyp(bytes: &[u8]) -> bool {
    bytes
        .get(..bytes.len().min(64))
        .is_some_and(|header| header.windows(4).any(|window| window == b"ftyp"))
}

#[tauri::command]
pub async fn begin_gaussian_video_export(
    state: State<'_, PreviewController>,
    project_id: String,
    orientation: Option<VideoOrientation>,
    edit_revision: Option<u64>,
) -> Result<GaussianVideoExportReservation> {
    let _lifecycle = state.lifecycle.lock().await;
    let project_id = parse_project_id(&project_id)?;
    let (root, _, metadata) = catalog::registered_final_ply_for_project(project_id).await?;
    if edit_revision.is_some_and(|revision| revision != metadata.editing.revision) {
        return Err(SplatError::Process(
            "编辑状态尚未保存完成，请重试导出".into(),
        ));
    }

    let active = state.active.lock().await;
    if active
        .as_ref()
        .is_none_or(|session| session.project_id != project_id)
    {
        return Err(SplatError::Process(
            "该项目当前未在 OOOSplat 预览中打开，无法导出视频。".into(),
        ));
    }
    drop(active);

    if state.html_export.lock().await.is_some() || state.model_export.lock().await.is_some() {
        return Err(SplatError::Process(
            "已有模型或 HTML 正在导出，请等待或取消。".into(),
        ));
    }

    let mut video_export = state.video_export.lock().await;
    if video_export.is_some() {
        return Err(SplatError::Process(
            "已有一个高斯预览视频正在导出，请等待其完成或先取消。".into(),
        ));
    }

    let export_id = Uuid::new_v4();
    let orientation = orientation.unwrap_or_default();
    let destination = next_gaussian_video_path_for_orientation(&root, orientation);
    let temporary = root.join(format!(".ooosplat-preview-{export_id}.mp4.tmp"));
    *video_export = Some(GaussianVideoExportSession {
        export_id,
        project_id,
        destination: destination.clone(),
        temporary,
        orientation,
        edit_revision: metadata.editing.revision,
        transform: metadata.transform,
        cancel: tokio_util::sync::CancellationToken::new(),
        running: false,
    });

    Ok(GaussianVideoExportReservation {
        export_id,
        destination_path: preview_client_path(&destination),
    })
}

async fn write_gaussian_video(
    session: &GaussianVideoExportSession,
    bytes: &[u8],
) -> Result<GaussianVideoExportResult> {
    if session.cancel.is_cancelled() {
        return Err(SplatError::Cancelled);
    }
    if bytes.is_empty() {
        return Err(SplatError::Process("视频编码器返回了空文件。".into()));
    }
    if bytes.len() > MAX_GAUSSIAN_VIDEO_BYTES {
        return Err(SplatError::Process("视频文件超过 1 GB 安全限制。".into()));
    }
    if !contains_mp4_ftyp(bytes) {
        return Err(SplatError::Process(
            "视频编码结果不是有效的 MP4 文件（缺少 ftyp 标识）。".into(),
        ));
    }
    if session.destination.exists() {
        return Err(SplatError::Process(format!(
            "视频目标文件已存在，请重新开始导出：{}",
            session.destination.display()
        )));
    }

    let write_result = async {
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&session.temporary)
            .await?;
        for chunk in bytes.chunks(1024 * 1024) {
            if session.cancel.is_cancelled() {
                return Err(SplatError::Cancelled);
            }
            file.write_all(chunk).await?;
        }
        file.flush().await?;
        file.sync_all().await?;
        drop(file);
        if session.cancel.is_cancelled() {
            return Err(SplatError::Cancelled);
        }
        let temporary = session.temporary.clone();
        let destination = session.destination.clone();
        tokio::task::spawn_blocking(move || {
            html_export::publish_html_file(&temporary, &destination)
        })
        .await
        .map_err(|error| SplatError::Process(format!("视频发布失败：{error}")))??;
        let _ = tokio::fs::remove_file(&session.temporary).await;
        Result::<()>::Ok(())
    }
    .await;

    if let Err(error) = write_result {
        let _ = tokio::fs::remove_file(&session.temporary).await;
        return Err(error);
    }

    Ok(GaussianVideoExportResult {
        path: preview_client_path(&session.destination),
        file_size: bytes.len() as u64,
        width: session.orientation.dimensions().0,
        height: session.orientation.dimensions().1,
        fps: GAUSSIAN_VIDEO_FPS,
        duration_ms: GAUSSIAN_VIDEO_DURATION_MS,
    })
}

#[tauri::command]
pub async fn commit_gaussian_video_export(
    state: State<'_, PreviewController>,
    request: tauri::ipc::Request<'_>,
) -> Result<GaussianVideoExportResult> {
    let export_id = request
        .headers()
        .get("x-ooosplat-export-id")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| Uuid::parse_str(value).ok())
        .ok_or_else(|| SplatError::Process("视频导出令牌无效。".into()))?;
    let bytes = match request.body() {
        InvokeBody::Raw(bytes) => bytes.as_slice(),
        _ => {
            return Err(SplatError::Process(
                "视频必须通过原始二进制 IPC 提交。".into(),
            ))
        }
    };

    let mut video_export = state.video_export.lock().await;
    let session = video_export
        .as_mut()
        .filter(|session| session.export_id == export_id && !session.running)
        .ok_or_else(|| SplatError::Process("视频导出会话不存在或已经结束。".into()))?;
    session.running = true;
    let session = session.clone();
    drop(video_export);
    let _metadata_guard = state.metadata_write.lock().await;
    let result = async {
        let (_, _, metadata) =
            catalog::registered_final_ply_for_project(session.project_id).await?;
        if metadata.editing.revision != session.edit_revision
            || metadata.transform != session.transform
        {
            return Err(SplatError::Process(
                "导出期间编辑状态发生变化，请重试".into(),
            ));
        }
        write_gaussian_video(&session, bytes).await
    }
    .await;
    let mut video_export = state.video_export.lock().await;
    if video_export
        .as_ref()
        .is_some_and(|session| session.export_id == export_id)
    {
        *video_export = None;
    }
    result
}

#[tauri::command]
pub async fn cancel_gaussian_video_export(
    state: State<'_, PreviewController>,
    export_id: String,
) -> Result<()> {
    let export_id = Uuid::parse_str(&export_id)
        .map_err(|_| SplatError::Process("视频导出令牌无效。".into()))?;
    let mut video_export = state.video_export.lock().await;
    if let Some(session) = video_export
        .as_ref()
        .filter(|session| session.export_id == export_id)
    {
        session.cancel.cancel();
        if !session.running {
            *video_export = None;
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn export_ply(source_path: String, destination_path: String) -> Result<u64> {
    let source = catalog::validate_registered_final_ply(Path::new(&source_path)).await?;
    let destination = PathBuf::from(destination_path);
    if destination
        .extension()
        .is_none_or(|extension| !extension.eq_ignore_ascii_case("ply"))
    {
        return Err(SplatError::InvalidPath(destination));
    }
    Ok(tokio::fs::copy(source, destination).await?)
}

#[cfg(test)]
mod tests {
    #[test]
    fn panorama_estimates_use_recipe_but_saved_runs_keep_their_parameters() {
        use super::*;
        for images in [false, true] {
            let mut state = PipelineStateFile::created(Quality::High);
            state.planner_enabled = true;
            state.panorama = Some(serde_json::from_value(serde_json::json!({
                "version": 1, "sourceSha256": "fixture", "sfmWidth": 4096,
                "sfmHeight": 2048, "faceSize": 1024, "fieldOfViewDegrees": 92.0,
                "imageSources": if images { serde_json::json!([]) } else { serde_json::Value::Null },
            })).unwrap());
            let resolved =
                resolve_checkpoint_brush(&state, Quality::High, Some(8192), 4096, 10_192).unwrap();
            assert_eq!(resolved.preset.total_steps, 45_000);
            assert_eq!(resolved.preset.max_splats, Some(3_000_000));
            let old = crate::presets::resolve_brush_training_preset(
                Quality::Fast,
                true,
                Some(8192),
                512,
                10_192,
            );
            state.brush_training.resolved = Some(old);
            assert_eq!(
                resolve_checkpoint_brush(&state, Quality::High, Some(24_576), 4096, 10_192)
                    .unwrap(),
                old
            );
            // A panorama task saved before the manual-cap change must also
            // retain its former 10M cap rather than adopting the new 3M cap.
            let mut old_panorama = crate::presets::resolve_panorama_training_preset(
                Quality::High,
                false,
                Some(8192),
                1024,
                10_192,
            )
            .unwrap();
            old_panorama.configured_max_splats = Some(10_000_000);
            old_panorama.preset.max_splats = Some(10_000_000);
            let saved = serde_json::to_vec(&old_panorama).unwrap();
            state.brush_training.resolved = Some(serde_json::from_slice(&saved).unwrap());
            assert_eq!(
                resolve_checkpoint_brush(&state, Quality::High, Some(24_576), 4096, 10_192)
                    .unwrap(),
                old_panorama
            );
        }
    }

    #[tokio::test]
    async fn folder_preflight_cache_revalidates_membership_and_never_needs_engines() {
        let root = tempfile::tempdir().unwrap();
        for name in ["2.png", "10.png"] {
            image::RgbImage::new(64, 32)
                .save(root.path().join(name))
                .unwrap();
        }
        let engines = crate::engines::EnginePaths::from_root(root.path().join("missing-engines"));
        let first = super::analyze_media(&engines, root.path()).await.unwrap();
        assert!(first.projection_detection.requires_choice);
        let second = super::analyze_media(&engines, root.path()).await.unwrap();
        assert_eq!(
            crate::video::image_projection::header_reads_for_test(root.path()),
            2
        );
        // More drafts than the cache capacity must still replan without header reads.
        for _ in 0..9 {
            let other = tempfile::tempdir().unwrap();
            for name in ["1.png", "2.png"] {
                image::RgbImage::new(64, 32)
                    .save(other.path().join(name))
                    .unwrap();
            }
            super::analyze_media(&engines, other.path()).await.unwrap();
        }
        super::input_analysis_for_plan(&engines, root.path(), Some(first.clone()))
            .await
            .unwrap();
        assert_eq!(
            crate::video::image_projection::header_reads_for_test(root.path()),
            2
        );
        assert_eq!(
            first.projection_detection.source_identity,
            second.projection_detection.source_identity
        );
        image::RgbImage::new(64, 32)
            .save(root.path().join("3.png"))
            .unwrap();
        let changed = super::analyze_media(&engines, root.path()).await.unwrap();
        assert_ne!(
            first.projection_detection.source_identity,
            changed.projection_detection.source_identity
        );
        assert_eq!(changed.image_sequence.unwrap().image_count, 3);
    }

    #[tokio::test]
    #[ignore = "Requires bundled FFprobe and a real ordinary/panorama video path"]
    async fn fast_video_detection_benchmark_and_single_probe_cache() {
        let source = std::path::PathBuf::from(
            std::env::var_os("OOOSPLAT_DETECTION_VIDEO").expect("Set real video"),
        );
        let engines = crate::engines::EnginePaths::from_root(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("engines"),
        );
        let start = std::time::Instant::now();
        let first = super::analyze_media(&engines, &source).await.unwrap();
        eprintln!(
            "video cold: {} us; confirmed={}, choice={}",
            start.elapsed().as_micros(),
            first.projection_detection.confirmed_erp,
            first.projection_detection.requires_choice
        );
        let mut absent = engines.clone();
        absent.ffprobe = std::path::PathBuf::from("must-not-execute-on-cache-hit");
        let start = std::time::Instant::now();
        let second = super::analyze_media(&absent, &source).await.unwrap();
        eprintln!(
            "video hot: {} us, zero FFprobe on cache hit",
            start.elapsed().as_micros()
        );
        assert_eq!(first.video, second.video);
    }

    #[tokio::test]
    #[ignore = "Header-only preflight cold/cache-hit timing, isolated PNG fixtures"]
    async fn fast_folder_detection_cache_benchmark() {
        for count in [100, 1000] {
            let root = tempfile::tempdir().unwrap();
            for i in 0..count {
                image::RgbImage::new(64, 32)
                    .save(root.path().join(format!("{i}.png")))
                    .unwrap();
            }
            let engines = crate::engines::EnginePaths::from_root(root.path().join("must-not-run"));
            let before = crate::video::image_projection::header_reads_for_test(root.path());
            let start = std::time::Instant::now();
            super::analyze_media(&engines, root.path()).await.unwrap();
            let cold = start.elapsed().as_micros();
            let middle = crate::video::image_projection::header_reads_for_test(root.path());
            let start = std::time::Instant::now();
            super::analyze_media(&engines, root.path()).await.unwrap();
            let hot = start.elapsed().as_micros();
            let after = crate::video::image_projection::header_reads_for_test(root.path());
            assert_eq!(middle - before, count);
            assert_eq!(after - middle, 0);
            eprintln!("{count} PNG preflight: cold {cold} us / {} header reads; cached {hot} us / {} header reads; FFprobe=0",middle-before,after-middle);
        }
    }

    #[tokio::test]
    async fn quality_replanning_reuses_video_analysis_without_starting_ffprobe() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("视频.mp4");
        std::fs::write(&input, b"identity-only fixture, not a decodable video").unwrap();
        let previous: super::VideoAnalysis = serde_json::from_value(serde_json::json!({
            "video": { "duration": 10.0, "width": 3840, "height": 1920, "fps": 30.0,
                "totalFrames": 300, "codec": "h264", "rotation": 0 },
            "projectionDetection": { "sourceIdentity": crate::video::panorama::source_identity(&input).unwrap(),
                "confirmedErp": true, "requiresChoice": false, "supportedErp": true,
                "reason": "metadata", "previewJpeg": [] }
        })).unwrap();
        let missing_engine = root.path().join("must-not-run.exe");
        let result = super::video_analysis_for_plan(
            &missing_engine,
            &missing_engine,
            &input,
            Some(previous.clone()),
        )
        .await
        .unwrap();
        assert_eq!(result.video, previous.video);
        assert!(result.projection_detection.confirmed_erp);
        // Changed content, size, mtime or path must not reuse a stale detection.
        std::fs::write(&input, b"changed").unwrap();
        assert!(super::video_analysis_for_plan(
            &missing_engine,
            &missing_engine,
            &input,
            Some(previous)
        )
        .await
        .is_err());
        assert!(
            super::video_analysis_for_plan(&missing_engine, &missing_engine, &input, None)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    #[ignore = "Requires OOOSPLAT_PANORAMA_DETAIL_TEST_PROJECT and the bundled engines; read-only real input verification"]
    async fn real_panorama_metadata_and_saved_detail_without_preview_extraction() {
        let project = std::path::PathBuf::from(
            std::env::var_os("OOOSPLAT_PANORAMA_DETAIL_TEST_PROJECT")
                .expect("Set a real panorama project path"),
        );
        let state: crate::project::PipelineStateFile =
            serde_json::from_slice(&std::fs::read(project.join("state.json")).unwrap()).unwrap();
        let engines = crate::pipeline::runner::default_engine_paths(Some(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("engines"),
        ));
        let input = project.join("source/input.mp4");
        let manager = crate::process::ProcessManager::new();
        let video = crate::engines::ffprobe::probe_video(&engines.ffprobe, &input, None, &manager)
            .await
            .unwrap();
        let detection = crate::video::panorama::detect(
            &engines.ffprobe,
            &engines.ffmpeg,
            &input,
            &video,
            &manager,
            false,
        )
        .await
        .unwrap();
        assert!(detection.confirmed_erp && !detection.requires_choice);
        assert!(!detection.source_identity.is_empty() && detection.preview_jpeg.is_empty());
        assert_eq!(
            super::project_video_projection(Some(&state), crate::project::ProjectInputType::Video),
            Some(crate::video::panorama::VideoProjection::Equirectangular)
        );
        eprintln!(
            "Real metadata/detail verified: {}x{}, confirmed ERP, no JPEG extraction",
            video.width, video.height
        );
    }
    #[test]
    fn historical_video_projection_comes_from_saved_configuration_not_dimensions() {
        use super::project_video_projection;
        use crate::presets::Quality;
        use crate::project::{PipelineStateFile, ProjectInputType};
        use crate::video::panorama::VideoProjection;
        let mut state = PipelineStateFile::created(Quality::Fast);
        assert_eq!(
            project_video_projection(Some(&state), ProjectInputType::Video),
            Some(VideoProjection::Perspective)
        );
        state.video_projection = VideoProjection::Equirectangular;
        assert_eq!(
            project_video_projection(Some(&state), ProjectInputType::Video),
            Some(VideoProjection::Equirectangular)
        );
        assert_eq!(
            project_video_projection(Some(&state), ProjectInputType::Images),
            None
        );
        state.video_projection = VideoProjection::Auto;
        assert_eq!(
            project_video_projection(Some(&state), ProjectInputType::Video),
            None
        );
        assert_eq!(
            project_video_projection(None, ProjectInputType::Video),
            None
        );
    }
    use super::{
        classify_dropped_input_path, classify_pipeline_failure, contains_mp4_ftyp,
        create_preview_asset, next_gaussian_video_path, persisted_task_stage, preview_client_path,
        read_project_log_tail, write_gaussian_video, GaussianVideoExportSession,
        PipelineCommandError, VideoOrientation,
    };
    use crate::error::SplatError;
    use crate::pipeline::{PipelineComponent, PipelineStage};
    use crate::contracts::FailureKind;
    use crate::presets::Quality;
    use crate::project::{PipelineStateFile, ProjectStatus};
    use std::{fs, path::Path};
    use tempfile::tempdir;
    use uuid::Uuid;

    #[tokio::test]
    async fn dropped_input_classification_accepts_video_and_directories() {
        let root = tempdir().unwrap();
        let images = root.path().join("中文 图片");
        fs::create_dir(&images).unwrap();
        let video = root.path().join("clip.MOV");
        fs::write(&video, b"video").unwrap();
        assert_eq!(
            classify_dropped_input_path(&images)
                .await
                .unwrap()
                .input_type,
            crate::project::ProjectInputType::Images
        );
        assert_eq!(
            classify_dropped_input_path(&video)
                .await
                .unwrap()
                .input_type,
            crate::project::ProjectInputType::Video
        );
    }

    #[tokio::test]
    async fn dropped_input_classification_rejects_single_images_unknown_and_missing_paths() {
        let root = tempdir().unwrap();
        for name in ["single.jpeg", "notes.txt"] {
            let path = root.path().join(name);
            fs::write(&path, b"x").unwrap();
            assert!(classify_dropped_input_path(&path).await.is_err());
        }
        assert!(
            classify_dropped_input_path(&root.path().join("missing.mp4"))
                .await
                .is_err()
        );
    }

    #[test]
    fn pipeline_errors_have_stable_machine_readable_codes() {
        let cancelled =
            serde_json::to_value(PipelineCommandError::from(SplatError::Cancelled)).unwrap();
        assert_eq!(cancelled["code"], "cancelled");
        let failed = serde_json::to_value(PipelineCommandError::from(SplatError::Process(
            "boom".into(),
        )))
        .unwrap();
        assert_eq!(failed["code"], "pipeline_failed");
        assert!(failed["message"].as_str().unwrap().contains("boom"));
    }

    #[test]
    fn pipeline_command_error_stays_below_clippy_large_error_threshold() {
        assert!(std::mem::size_of::<PipelineCommandError>() <= 128);
    }

    #[test]
    fn persisted_task_stage_preserves_terminal_status_and_checkpoint_position() {
        let mut state = PipelineStateFile::created(Quality::Balanced);
        state.features_complete = true;
        state.matching_complete = true;
        assert_eq!(
            persisted_task_stage(&state, ProjectStatus::Interrupted),
            PipelineStage::Reconstructing
        );
        assert_eq!(
            persisted_task_stage(&state, ProjectStatus::Failed),
            PipelineStage::Reconstructing
        );
        assert_eq!(
            persisted_task_stage(&state, ProjectStatus::Cancelled),
            PipelineStage::Reconstructing
        );
    }

    #[tokio::test]
    async fn historical_log_tail_is_bounded_to_the_latest_five_hundred_lines() {
        let root = tempdir().expect("temporary log directory");
        let logs = root.path().join("logs");
        fs::create_dir_all(&logs).unwrap();
        let content = (0..620)
            .map(|index| format!("line-{index:04}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(logs.join("brush.log"), content).unwrap();

        let lines = read_project_log_tail(&logs).await.unwrap();
        assert_eq!(lines.len(), 500);
        assert_eq!(lines.first().unwrap().message, "line-0120");
        assert_eq!(lines.last().unwrap().message, "line-0619");
        assert!(lines.iter().all(|line| line.source == "brush"));
    }

    #[tokio::test]
    async fn failure_snapshots_cover_generic_errors_but_never_cancelled_tasks() {
        let controller = super::PipelineController::default();
        let cancelled =
            super::record_command_failure(&controller, SplatError::Cancelled.into(), &[]).await;
        assert!(cancelled.failure_id.is_none());
        let failed = super::record_command_failure(
            &controller,
            SplatError::Process("image import failed".into()).into(),
            &[],
        )
        .await;
        assert!(failed.failure_id.is_some());
        assert_eq!(failed.code, "pipeline_failed");
        for (stage, engine) in [
            (PipelineStage::ProbingVideo, PipelineComponent::Media),
            (PipelineStage::ExtractingFrames, PipelineComponent::Media),
            (PipelineStage::ExtractingFeatures, PipelineComponent::Reconstruction),
            (PipelineStage::Matching, PipelineComponent::Reconstruction),
            (
                PipelineStage::ValidatingReconstruction,
                PipelineComponent::Reconstruction,
            ),
            (PipelineStage::Exporting, PipelineComponent::System),
        ] {
            assert_eq!(
                classify_pipeline_failure(Some(stage), "failure").0,
                Some(engine)
            );
        }
    }

    #[test]
    fn classifies_mapper_and_brush_failures_for_plain_language_guidance() {
        assert_eq!(
            classify_pipeline_failure(
                Some(PipelineStage::Reconstructing),
                "Could not find a good initial image pair",
            ),
            (Some(PipelineComponent::Reconstruction), Some(FailureKind::ReconstructionInput)),
        );
        assert_eq!(
            classify_pipeline_failure(
                Some(PipelineStage::TrainingSplats),
                "IO error while loading dataset: early eof",
            ),
            (Some(PipelineComponent::Training), Some(FailureKind::TrainingDataset)),
        );
        assert_eq!(
            classify_pipeline_failure(
                Some(PipelineStage::TrainingSplats),
                "Device lost while allocating a buffer",
            ),
            (Some(PipelineComponent::Training), Some(FailureKind::TrainingDeviceLost)),
        );

        let real_mapper_log = r#"
            Loading database
            Loading cameras...
            Loading matches...
            Finding good initial image pair
            No good initial image pair found.
            Discarding reconstruction because it is too small
            Failed to create any sparse model
        "#;
        assert_eq!(
            classify_pipeline_failure(Some(PipelineStage::Reconstructing), real_mapper_log),
            (Some(PipelineComponent::Reconstruction), Some(FailureKind::ReconstructionInput)),
        );
        assert_eq!(
            classify_pipeline_failure(
                Some(PipelineStage::Reconstructing),
                "SQLite error: database is locked",
            ),
            (Some(PipelineComponent::Reconstruction), Some(FailureKind::ReconstructionStorage)),
        );
        assert_eq!(
            classify_pipeline_failure(
                Some(PipelineStage::Reconstructing),
                "COLMAP did not produce a usable sparse model: file not found",
            ),
            (Some(PipelineComponent::Reconstruction), Some(FailureKind::ReconstructionInput)),
        );
        assert_eq!(
            classify_pipeline_failure(
                Some(PipelineStage::ValidatingReconstruction),
                "稀疏重建没有可用的注册图像或三维点",
            ),
            (Some(PipelineComponent::Reconstruction), Some(FailureKind::ReconstructionInput)),
        );
        assert_eq!(
            classify_pipeline_failure(
                Some(PipelineStage::Reconstructing),
                "I/O error while reading feature descriptors",
            ),
            (Some(PipelineComponent::Reconstruction), Some(FailureKind::ReconstructionInput)),
        );
        assert_eq!(
            classify_pipeline_failure(
                Some(PipelineStage::Reconstructing),
                "I/O error while opening SQLite database",
            ),
            (Some(PipelineComponent::Reconstruction), Some(FailureKind::ReconstructionStorage)),
        );
    }

    #[test]
    fn preview_path_keeps_regular_paths() {
        assert_eq!(
            preview_client_path(Path::new(r"C:\Projects\场景\final.ply")),
            Path::new(r"C:\Projects\场景\final.ply")
        );
    }

    #[test]
    fn preview_path_removes_windows_verbatim_prefix() {
        assert_eq!(
            preview_client_path(Path::new(r"\\?\C:\Projects\场景\final.ply")),
            Path::new(r"C:\Projects\场景\final.ply")
        );
    }

    #[test]
    fn preview_path_restores_unc_prefix() {
        assert_eq!(
            preview_client_path(Path::new(r"\\?\UNC\server\share\final.ply")),
            Path::new(r"\\server\share\final.ply")
        );
    }

    #[tokio::test]
    async fn preview_assets_use_a_fresh_session_path_without_reusing_final_ply() {
        let root = tempdir().expect("temporary project");
        let source = root.path().join("final.ply");
        fs::write(&source, b"ply preview fixture").expect("source ply");

        let first = create_preview_asset(root.path(), &source)
            .await
            .expect("first preview asset");
        let second = create_preview_asset(root.path(), &source)
            .await
            .expect("second preview asset");

        assert_ne!(first, second);
        assert_ne!(first, source);
        assert!(first.starts_with(root.path().join("work").join("preview")));
        assert_eq!(
            fs::read(first).expect("first preview contents"),
            b"ply preview fixture"
        );
        assert_eq!(
            fs::read(second).expect("second preview contents"),
            b"ply preview fixture"
        );
    }

    #[test]
    fn video_path_uses_the_first_available_preview_number() {
        let root = tempdir().expect("temporary project");
        assert_eq!(
            next_gaussian_video_path(root.path()),
            root.path().join("preview.mp4")
        );
        fs::write(root.path().join("preview.mp4"), b"existing").expect("first video");
        fs::write(root.path().join("preview-2.mp4"), b"existing").expect("second video");
        assert_eq!(
            next_gaussian_video_path(root.path()),
            root.path().join("preview-3.mp4")
        );
    }

    #[test]
    fn mp4_validation_requires_ftyp_near_the_start() {
        expect_ftyp(true, b"\0\0\0\x18ftypisom\0\0\0\0");
        expect_ftyp(false, b"not-an-mp4");
        let mut late = vec![0_u8; 80];
        late[70..74].copy_from_slice(b"ftyp");
        expect_ftyp(false, &late);
    }

    fn expect_ftyp(expected: bool, bytes: &[u8]) {
        assert_eq!(contains_mp4_ftyp(bytes), expected);
    }

    #[tokio::test]
    async fn video_publish_is_atomic_and_never_overwrites_an_existing_export() {
        let root = tempdir().expect("temporary project");
        let destination = root.path().join("preview.mp4");
        let temporary = root.path().join(".preview.tmp");
        let session = GaussianVideoExportSession {
            export_id: Uuid::new_v4(),
            project_id: Uuid::new_v4(),
            destination: destination.clone(),
            temporary: temporary.clone(),
            orientation: VideoOrientation::Portrait,
            edit_revision: 0,
            transform: crate::project::GaussianTransform::default(),
            cancel: tokio_util::sync::CancellationToken::new(),
            running: false,
        };
        let bytes = b"\0\0\0\x18ftypisom\0\0\0\0payload";

        let result = write_gaussian_video(&session, bytes)
            .await
            .expect("valid MP4 is published");
        assert_eq!(result.path, destination);
        assert_eq!(fs::read(&destination).expect("published bytes"), bytes);
        assert!(!temporary.exists());

        let error = write_gaussian_video(&session, bytes)
            .await
            .expect_err("existing video must not be overwritten");
        assert!(error.to_string().contains("已存在"));
        assert_eq!(fs::read(&destination).expect("original export"), bytes);
    }

    #[tokio::test]
    async fn landscape_video_uses_its_own_numbering_and_real_dimensions() {
        let root = tempdir().unwrap();
        let first = super::next_gaussian_video_path_for_orientation(
            root.path(),
            VideoOrientation::Landscape,
        );
        assert_eq!(first.file_name().unwrap(), "preview-landscape.mp4");
        let session = GaussianVideoExportSession {
            export_id: Uuid::new_v4(),
            project_id: Uuid::new_v4(),
            destination: first.clone(),
            temporary: root.path().join(".landscape.tmp"),
            orientation: VideoOrientation::Landscape,
            edit_revision: 0,
            transform: crate::project::GaussianTransform::default(),
            cancel: tokio_util::sync::CancellationToken::new(),
            running: false,
        };
        let result = write_gaussian_video(&session, b"\0\0\0\x18ftypisom\0\0\0\0payload")
            .await
            .unwrap();
        assert_eq!(
            (result.width, result.height, result.fps, result.duration_ms),
            (1920, 1080, 30, 23000)
        );
        assert_eq!(
            super::next_gaussian_video_path_for_orientation(
                root.path(),
                VideoOrientation::Landscape
            )
            .file_name()
            .unwrap(),
            "preview-landscape-2.mp4"
        );
        assert_eq!(
            next_gaussian_video_path(root.path()).file_name().unwrap(),
            "preview.mp4"
        );
    }

    #[tokio::test]
    async fn invalid_mp4_is_rejected_without_leaving_a_temporary_file() {
        let root = tempdir().expect("temporary project");
        let session = GaussianVideoExportSession {
            export_id: Uuid::new_v4(),
            project_id: Uuid::new_v4(),
            destination: root.path().join("preview.mp4"),
            temporary: root.path().join(".preview.tmp"),
            orientation: VideoOrientation::Portrait,
            edit_revision: 0,
            transform: crate::project::GaussianTransform::default(),
            cancel: tokio_util::sync::CancellationToken::new(),
            running: false,
        };

        write_gaussian_video(&session, b"not-an-mp4")
            .await
            .expect_err("invalid MP4 must be rejected");
        assert!(!session.destination.exists());
        assert!(!session.temporary.exists());
        session.cancel.cancel();
        assert!(matches!(
            write_gaussian_video(&session, b"\0\0\0\x18ftypisom\0\0\0\0payload").await,
            Err(SplatError::Cancelled)
        ));
        assert!(!session.destination.exists());
        assert!(!session.temporary.exists());
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IncrementalReshootRequest {
    source_project_id: String,
    reshoot_path: String,
    input_type: ReshootInputType,
    projects_root: String,
    workspace_task_id: Option<String>,
    quality: Option<Quality>,
    planner_enabled: Option<bool>,
}

#[tauri::command]
pub async fn inspect_reshoot_source(
    app: tauri::AppHandle,
    project_id: String,
) -> std::result::Result<ReshootSourceInfo, SplatError> {
    let project_id =
        Uuid::parse_str(&project_id).map_err(|_| SplatError::Process("原项目 ID 无效".into()))?;
    PipelineRunner::new(paths_for_app(&app), |_| {})
        .inspect_reshoot_source(project_id)
        .await
}

#[tauri::command]
pub async fn probe_reshoot_input(
    app: tauri::AppHandle,
    project_id: String,
    path: String,
    input_type: ReshootInputType,
) -> std::result::Result<ReshootInputInfo, SplatError> {
    let project_id =
        Uuid::parse_str(&project_id).map_err(|_| SplatError::Process("原项目 ID 无效".into()))?;
    PipelineRunner::new(paths_for_app(&app), |_| {})
        .probe_reshoot_input(project_id, Path::new(&path), input_type)
        .await
}

#[tauri::command]
pub async fn start_gui_reshoot(
    app: tauri::AppHandle,
    state: State<'_, PipelineController>,
    telemetry: State<'_, TelemetryService>,
    imports: State<'_, crate::imports::ImportController>,
    request: IncrementalReshootRequest,
) -> std::result::Result<crate::tasks::StartReceipt, crate::tasks::ServiceError> {
    let _admission = imports.admission().await;
    if imports.is_active().await {
        return Err(crate::tasks::ServiceError::new(
            "TASK_BUSY",
            "已有 Gaussian 导入正在进行，请等待完成后再启动补拍",
        ));
    }
    let source = Uuid::parse_str(&request.source_project_id)
        .map_err(|_| crate::tasks::ServiceError::new("INVALID_ARGUMENT", "原项目 ID 无效"))?;
    let (project, metadata) = catalog::load_registered_project(source).await?;
    if (request.input_type == ReshootInputType::Images) != Path::new(&request.reshoot_path).is_dir()
    {
        return Err(crate::tasks::ServiceError::new(
            "INVALID_INPUT",
            "补拍输入类型与路径不一致",
        ));
    }
    let checkpoint: PipelineStateFile =
        serde_json::from_slice(&tokio::fs::read(project.join("state.json")).await?)?;
    let imported = metadata.imported.is_some();
    let quality = if imported {
        request.quality.unwrap_or(Quality::Balanced)
    } else {
        metadata.quality
    };
    let planner_enabled = if imported {
        request.planner_enabled.unwrap_or(true)
    } else {
        checkpoint.planner_enabled
    };
    let id = request
        .workspace_task_id
        .map(|v| Uuid::parse_str(&v))
        .transpose()
        .map_err(|_| crate::tasks::ServiceError::new("INVALID_ARGUMENT", "工作区任务 ID 无效"))?;
    let task = state
        .create(
            id,
            request.reshoot_path.into(),
            quality,
            request.projects_root.into(),
            planner_enabled,
            "gui",
            None,
            None,
        )
        .await?;
    state.mark_reshoot(task.task_id, source).await?;
    state
        .start(
            task.task_id,
            paths_for_app(&app),
            Some(telemetry.inner().clone()),
        )
        .await
}
#[tauri::command]
pub async fn start_incremental_reshoot_pipeline(
    app: tauri::AppHandle,
    state: State<'_, PipelineController>,
    telemetry: State<'_, TelemetryService>,
    imports: State<'_, crate::imports::ImportController>,
    request: IncrementalReshootRequest,
) -> std::result::Result<PipelineResult, PipelineCommandError> {
    let receipt = start_gui_reshoot(app, state.clone(), telemetry, imports, request)
        .await
        .map_err(service_command_error)?;
    wait_task_result(state.inner(), receipt.task_id).await
}
