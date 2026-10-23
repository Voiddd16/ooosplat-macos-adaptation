use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;

use crate::{
    contracts::ComponentKind,
    engines::backend::request::{
        FeatureExtraction, IncrementalFeatureExtraction, IncrementalMapping, Matching,
        PairMatching, QualityFeatureExtraction, SparseMapping,
    },
    engines::{
        brush, colmap,
        ffmpeg::validate_extraction,
        ffprobe::probe_video,
        EnginePaths,
    },
    error::{Result, SplatError},
    pipeline::{
        effectiveness::{
            PlannerBridgeMetrics, PlannerBridgeStatus, PlannerEffectivenessTracker,
            PlannerFinalResultMetrics, PlannerFramePlanMetrics, PlannerGpuVendor,
            PlannerInitialReconstructionMetrics,
        },
        estimate::{
            estimate_calibrated_brush_stage_ms, estimate_calibrated_brush_stage_ms_for_images,
            estimate_runtime, estimate_runtime_for_images, RuntimeEstimate,
        },
        progress::stage_progress_range,
        runtime::{RuntimeSnapshot, RuntimeTracker},
        EventKind, EventLevel, PipelineComponent, PipelineEvent, PipelineStage,
    },
    planner::{
        bridge_trigger_ratio, plan_bridge_backfill, read_registered_source_indices,
        write_bridge_pair_list, BridgeBackfillStatus,
    },
    presets::{
        resolve_brush_training_preset, resolve_brush_training_preset_for_plan,
        resolve_panorama_training_preset, resolve_planner_resolution_plan, Quality,
        ResolvedBrushTrainingPreset, PLANNER_RESOLUTION_POLICY_VERSION,
    },
    process::{ProcessManager, ProcessObserver, ProcessUpdate},
    project::{
        catalog, manager::atomic_replace_file, FrameState, PipelineStateFile,
        ProjectImportObserver, ProjectInputType, ProjectManager, ProjectMetadata, ProjectOutput,
        ProjectPaths, ProjectStatus, ReshootProvenance, ReshootState, COLMAP_INPUT_VERSION,
    },
    reconstruction::{
        colmap_model::{
            read_registered_images, read_single_camera, reusable_model_files, ColmapCamera,
        },
        ply::inspect_gaussian_ply,
        validator::{ReconstructionQuality, ReconstructionReport, ReconstructionValidator},
    },
    video::{
        prepare_scanned_image_sequence, scaled_video_dimensions, scan_image_sequence,
        validate_prepared_image_sequence, validate_reshoot_image_sequence, video_can_scale_to,
        FramePlan, FrameSelectionStrategy, ImagePreparationObserver, ImagePreparationPhase,
        ImageSequenceInfo, ImageSequenceNaming, PlannedFrame, PreparedImageSequence,
        QualityV2FrameSelection, UniformRatioFrameSelection, VideoInfo,
    },
};

pub struct PreparedFrames {
    pub input_type: ProjectInputType,
    pub video: Option<VideoInfo>,
    pub image_sequence: Option<ImageSequenceInfo>,
    pub plan: FramePlan,
    pub extracted_frames: u64,
    pub image_format: String,
    pub mask_count: u64,
    pub has_alpha: bool,
    pub working_width: u32,
    pub working_height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PipelineResult {
    pub project_id: String,
    pub project_path: PathBuf,
    pub final_ply: PathBuf,
    pub file_size: u64,
    pub splat_count: u64,
    pub input_images: u64,
    pub registered_images: u64,
    pub registered_ratio: f64,
    pub points_3d: u64,
    pub duration_ms: u64,
    pub completed_at: chrono::DateTime<Utc>,
    pub warning: Option<String>,
    pub logs_directory: PathBuf,
    #[serde(skip)]
    pub(crate) source_duration_seconds: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReshootSourceInfo {
    pub project_id: String,
    pub project_name: String,
    pub quality: Quality,
    pub planner_enabled: bool,
    pub camera_id: u32,
    pub camera_model: String,
    pub width: u64,
    pub height: u64,
    pub source_image_count: u64,
    pub eligible: bool,
    pub reason: Option<String>,
    pub imported: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReshootInputType {
    Video,
    Images,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReshootInputInfo {
    pub input_type: ReshootInputType,
    pub image_count: Option<u64>,
    pub duration: Option<f64>,
    pub prepared_width: u32,
    pub prepared_height: u32,
    pub estimated_frames: u64,
    pub has_alpha: bool,
    pub mask_count: u64,
    pub compatible: bool,
    pub incompatibility_reason: Option<String>,
    pub estimate: RuntimeEstimate,
}

#[derive(Clone)]
struct EventSink {
    context: Arc<std::sync::Mutex<OperationContext>>,
    bound_operation: Option<super::event::PipelineOperation>,
    // 0 ordinary; 1 panorama; 2 face preparation; 3 validation; 4 frame timing.
    panorama_phase: Arc<AtomicU64>,
    panorama_images: Arc<AtomicBool>,
    emit: Arc<dyn Fn(PipelineEvent) + Send + Sync>,
    sequence: Arc<AtomicU64>,
    last_progress_milli_percent: Arc<AtomicU64>,
    last_stage: Arc<std::sync::Mutex<Option<PipelineStage>>>,
    dispatch: Arc<std::sync::Mutex<()>>,
    started: Instant,
}

#[derive(Default)]
struct OperationContext {
    serial: u64,
    stage: Option<PipelineStage>,
    substage: Option<String>,
    current: Option<super::event::PipelineOperation>,
    attempts: Vec<(PipelineStage, u32)>,
}

impl EventSink {
    fn substage(&self, stage: PipelineStage) -> Option<String> {
        match (stage, self.panorama_phase.load(Ordering::Relaxed)) {
            (PipelineStage::ExtractingFrames, 4) => Some("panoramaTimestamps"),
            (PipelineStage::ExtractingFrames, 1) => {
                Some(if self.panorama_images.load(Ordering::Relaxed) {
                    "panoramaImagesPreparation"
                } else {
                    "panoramaExtraction"
                })
            }
            (PipelineStage::Reconstructing, 1) => {
                Some(if self.panorama_images.load(Ordering::Relaxed) {
                    "panoramaImagesReconstruction"
                } else {
                    "panoramaReconstruction"
                })
            }
            (PipelineStage::TrainingSplats, 2) => Some("panoramaTrainingPreparation"),
            (PipelineStage::TrainingSplats, 3) => Some("panoramaTrainingValidation"),
            (PipelineStage::TrainingSplats, _) => Some("training"),
            _ => None,
        }
        .map(str::to_owned)
    }

    fn begin_process(&self, stage: PipelineStage) -> Self {
        let _dispatch = self.dispatch.lock().unwrap_or_else(|p| p.into_inner());
        let mut context = self.context.lock().unwrap_or_else(|p| p.into_inner());
        let attempt =
            if let Some((_, count)) = context.attempts.iter_mut().find(|(key, _)| *key == stage) {
                *count += 1;
                *count
            } else {
                context.attempts.push((stage, 1));
                1
            };
        context.serial += 1;
        let operation = super::event::PipelineOperation {
            id: context.serial,
            attempt,
        };
        context.stage = Some(stage);
        context.substage = self.substage(stage);
        context.current = Some(operation.clone());
        let mut scoped = self.clone();
        scoped.bound_operation = Some(operation);
        scoped
    }
    #[allow(clippy::too_many_arguments)]
    fn send(
        &self,
        stage: PipelineStage,
        component: Option<PipelineComponent>,
        kind: EventKind,
        level: EventLevel,
        stage_progress: Option<f32>,
        indeterminate: bool,
        message: impl Into<String>,
        current: Option<u64>,
        total: Option<u64>,
        unit: Option<&str>,
    ) {
        let _dispatch = self.dispatch.lock().unwrap_or_else(|p| p.into_inner());
        let mut context = self.context.lock().unwrap_or_else(|p| p.into_inner());
        if self
            .bound_operation
            .as_ref()
            .is_some_and(|op| context.current.as_ref().map(|active| active.id) != Some(op.id))
        {
            return;
        }
        let substage = self.substage(stage);
        if kind == EventKind::Stage
            && (context.stage != Some(stage) || context.substage != substage)
        {
            context.serial += 1;
            context.current = Some(super::event::PipelineOperation {
                id: context.serial,
                attempt: 0,
            });
            context.stage = Some(stage);
            context.substage = substage.clone();
        }
        let operation = context.current.clone();
        let (stage, substage) = if matches!(kind, EventKind::Log | EventKind::Heartbeat) {
            (context.stage.unwrap_or(stage), context.substage.clone())
        } else {
            (stage, substage)
        };
        drop(context);
        if kind == EventKind::Stage
            && !matches!(
                stage,
                PipelineStage::Completed | PipelineStage::Failed | PipelineStage::Cancelled
            )
        {
            *self
                .last_stage
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(stage);
        }
        let phase = self.panorama_phase.load(Ordering::Relaxed);
        let (start, end) = if stage == PipelineStage::TrainingSplats && phase != 0 {
            if phase == 2 || phase == 3 {
                (60.0, 66.0)
            } else {
                (66.0, 98.0)
            }
        } else {
            stage_progress_range(stage)
        };
        let progress = stage_progress
            .map(|value| start + (end - start) * value.clamp(0.0, 1.0))
            .unwrap_or(start);
        let progress = if stage != PipelineStage::Completed {
            progress.min(99.9)
        } else {
            progress
        };
        self.last_progress_milli_percent.fetch_max(
            (progress.max(0.0) * 1_000.0).round() as u64,
            Ordering::Relaxed,
        );
        let stage_percent = stage_progress.map(|value| {
            if stage == PipelineStage::TrainingSplats && phase != 0 {
                (start + (end - start) * value.clamp(0.0, 1.0) - 60.0) / 38.0 * 100.0
            } else {
                value.clamp(0.0, 1.0) * 100.0
            }
        });
        let progress = self.last_progress_milli_percent.load(Ordering::Relaxed) as f32 / 1_000.0;
        (self.emit)(PipelineEvent {
            operation,
            task_id: None,
            run_id: None,
            revision: 0,
            sequence: self.sequence.fetch_add(1, Ordering::Relaxed) + 1,
            timestamp: Utc::now(),
            kind,
            level,
            stage,
            substage,
            component,
            progress,
            stage_progress: stage_percent,
            indeterminate,
            message: message.into(),
            current,
            total,
            unit: unit.map(str::to_owned),
            elapsed_ms: self.started.elapsed().as_millis() as u64,
            acceleration: None,
            runtime: None,
        });
    }

    fn stage(&self, stage: PipelineStage, progress: f32, message: impl Into<String>) {
        self.send(
            stage,
            Some(PipelineComponent::System),
            EventKind::Stage,
            EventLevel::Info,
            Some(progress),
            false,
            message,
            None,
            None,
            None,
        );
    }

    fn acceleration(&self, status: crate::engines::AccelerationStatus) {
        let _dispatch = self
            .dispatch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        (self.emit)(PipelineEvent {
            operation: None,
            task_id: None,
            run_id: None,
            revision: 0,
            sequence: self.sequence.fetch_add(1, Ordering::Relaxed) + 1,
            timestamp: Utc::now(),
            kind: EventKind::Capability,
            level: if status.use_gpu() {
                EventLevel::Info
            } else {
                EventLevel::Warning
            },
            stage: PipelineStage::Created,
            substage: None,
            component: Some(PipelineComponent::Reconstruction),
            progress: 0.0,
            stage_progress: None,
            indeterminate: false,
            message: status.reason.clone(),
            current: None,
            total: None,
            unit: None,
            elapsed_ms: self.started.elapsed().as_millis() as u64,
            acceleration: Some(status),
            runtime: None,
        });
    }

    fn runtime(&self, stage: PipelineStage, component: PipelineComponent, snapshot: RuntimeSnapshot) {
        let _dispatch = self
            .dispatch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let context = self.context.lock().unwrap_or_else(|p| p.into_inner());
        if self
            .bound_operation
            .as_ref()
            .is_some_and(|op| context.current.as_ref().map(|active| active.id) != Some(op.id))
        {
            return;
        }
        (self.emit)(PipelineEvent {
            operation: context.current.clone(),
            task_id: None,
            run_id: None,
            revision: 0,
            sequence: self.sequence.fetch_add(1, Ordering::Relaxed) + 1,
            timestamp: Utc::now(),
            kind: EventKind::Runtime,
            level: EventLevel::Info,
            stage,
            substage: context.substage.clone(),
            component: Some(component),
            progress: self.last_progress_milli_percent.load(Ordering::Relaxed) as f32 / 1_000.0,
            stage_progress: None,
            indeterminate: true,
            message: String::new(),
            current: None,
            total: None,
            unit: None,
            elapsed_ms: self.started.elapsed().as_millis() as u64,
            acceleration: None,
            runtime: Some(snapshot),
        });
    }

    fn terminal(&self, error: &SplatError) {
        let cancelled = matches!(error, SplatError::Cancelled);
        let _dispatch = self
            .dispatch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        (self.emit)(PipelineEvent {
            operation: self
                .context
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .current
                .clone(),
            task_id: None,
            run_id: None,
            revision: 0,
            sequence: self.sequence.fetch_add(1, Ordering::Relaxed) + 1,
            timestamp: Utc::now(),
            kind: EventKind::Stage,
            level: if cancelled {
                EventLevel::Warning
            } else {
                EventLevel::Error
            },
            stage: if cancelled {
                PipelineStage::Cancelled
            } else {
                PipelineStage::Failed
            },
            substage: None,
            component: Some(PipelineComponent::System),
            progress: self.last_progress_milli_percent.load(Ordering::Relaxed) as f32 / 1_000.0,
            stage_progress: None,
            indeterminate: false,
            message: error.to_string(),
            current: None,
            total: None,
            unit: None,
            elapsed_ms: self.started.elapsed().as_millis() as u64,
            acceleration: None,
            runtime: None,
        });
    }
}

#[derive(Debug, Clone)]
pub struct PipelineFailureContext {
    pub failed_stage: Option<PipelineStage>,
    pub project_id: Option<uuid::Uuid>,
    pub project_path: Option<PathBuf>,
    pub logs_directory: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct ActiveProjectContext {
    project_id: uuid::Uuid,
    workspace_task_id: Option<uuid::Uuid>,
    project_path: PathBuf,
    logs_directory: PathBuf,
}

#[derive(Debug, Clone)]
pub enum RunnerUpdate {
    Configuration(serde_json::Value),
    Project(Box<crate::project::ProjectMetadata>),
    LogOpened {
        path: PathBuf,
        offset: u64,
    },
    ProcessExited {
        component: PipelineComponent,
        exit_code: Option<i32>,
    },
}

pub struct PipelineRunner {
    image_matching_strategy: crate::project::ImageMatchingStrategy,
    video_projection: crate::video::panorama::VideoProjection,
    lifecycle: Option<Arc<dyn Fn(RunnerUpdate) + Send + Sync>>,
    input_boundary: Option<PathBuf>,
    engines: EnginePaths,
    /// 稀疏重建后端（编译期选定的实现，契约见 `engines::backend`）。
    backend: Arc<dyn crate::engines::backend::SfMBackend>,
    /// 媒体处理实现（编译期选定：外部 ffmpeg 进程 / 进程内 FFmpeg 库）。
    media: Arc<dyn crate::media::MediaTool>,
    process_manager: ProcessManager,
    events: EventSink,
    active_project: Arc<std::sync::Mutex<Option<ActiveProjectContext>>>,
    current_acceleration: Arc<std::sync::Mutex<Option<crate::engines::AccelerationStatus>>>,
    elapsed_offset_ms: AtomicU64,
    workspace_task_id: Option<uuid::Uuid>,
    planner_enabled: bool,
    effectiveness: Option<PlannerEffectivenessTracker>,
}

fn inherit_reshoot_planner_settings(source: &PipelineStateFile, target: &mut PipelineStateFile) {
    target.planner_enabled = source.planner_enabled;
    target.resolution_policy_version = source.resolution_policy_version;
}

fn reshoot_project_name(source_name: &str) -> String {
    format!("{source_name}_补拍")
}

fn image_matching_policy(
    input: ProjectInputType,
    strategy: crate::project::ImageMatchingStrategy,
    quality: Quality,
    planner_enabled: bool,
) -> (bool, bool) {
    let sequential = input == ProjectInputType::Video
        || strategy == crate::project::ImageMatchingStrategy::Sequential;
    (
        sequential,
        sequential && quality.loop_detection_enabled(planner_enabled),
    )
}

impl PipelineRunner {
    pub fn new(engines: EnginePaths, emit: impl Fn(PipelineEvent) + Send + Sync + 'static) -> Self {
        Self::new_with_planner(engines, true, emit)
    }

    pub fn new_with_planner(
        engines: EnginePaths,
        planner_enabled: bool,
        emit: impl Fn(PipelineEvent) + Send + Sync + 'static,
    ) -> Self {
        Self {
            video_projection: crate::video::panorama::VideoProjection::Auto,
            image_matching_strategy: crate::project::ImageMatchingStrategy::default(),
            backend: crate::engines::backend::active(crate::engines::backend::BackendConfig {
                executable: Some(engines.colmap.clone()),
                automatic_optimization: true,
                accelerator_index: None,
            }),
            media: crate::media::active(crate::media::MediaConfig {
                ffmpeg: Some(engines.ffmpeg.clone()),
                ffprobe: Some(engines.ffprobe.clone()),
            }),
            engines,
            lifecycle: None,
            input_boundary: None,
            process_manager: ProcessManager::new(),
            events: EventSink {
                context: Default::default(),
                bound_operation: None,
                panorama_phase: Arc::new(AtomicU64::new(0)),
                panorama_images: Arc::new(AtomicBool::new(false)),
                emit: Arc::new(emit),
                sequence: Arc::new(AtomicU64::new(0)),
                last_progress_milli_percent: Arc::new(AtomicU64::new(0)),
                last_stage: Arc::new(std::sync::Mutex::new(None)),
                dispatch: Arc::new(std::sync::Mutex::new(())),
                started: Instant::now(),
            },
            active_project: Arc::new(std::sync::Mutex::new(None)),
            current_acceleration: Arc::new(std::sync::Mutex::new(None)),
            elapsed_offset_ms: AtomicU64::new(0),
            workspace_task_id: None,
            planner_enabled,
            effectiveness: None,
        }
    }

    /// 构造一次后端作业的运行环境。
    ///
    /// 这是 pipeline 与后端之间**唯一**的交接点：日志位置、进度观察者、进程托管器、
    /// 加速器选择。后端拿不到可执行文件路径，也不需要知道 pipeline 的内部状态。
    fn job_context(
        &self,
        log: PathBuf,
        observer: Option<crate::process::ProcessObserver>,
        accelerator_index: Option<u32>,
    ) -> crate::engines::backend::JobContext<'_> {
        crate::engines::backend::JobContext::new(
            log,
            observer,
            &self.process_manager,
            accelerator_index,
        )
    }

    /// 构造媒体作业的运行环境（与实现无关）。
    fn media_context(
        &self,
        log_path: Option<PathBuf>,
        observer: Option<crate::process::ProcessObserver>,
    ) -> crate::media::MediaContext<'_> {
        crate::media::MediaContext {
            process_manager: &self.process_manager,
            observer,
            log_path,
        }
    }

    pub fn with_input_boundary(mut self, path: Option<PathBuf>) -> Self {
        self.input_boundary = path;
        self
    }

    pub fn with_video_projection(
        mut self,
        projection: crate::video::panorama::VideoProjection,
    ) -> Self {
        self.video_projection = projection;
        self
    }

    pub fn with_image_matching_strategy(
        mut self,
        strategy: crate::project::ImageMatchingStrategy,
    ) -> Self {
        self.image_matching_strategy = strategy;
        self
    }

    pub fn with_lifecycle_observer(
        mut self,
        observer: impl Fn(RunnerUpdate) + Send + Sync + 'static,
    ) -> Self {
        let observer: Arc<dyn Fn(RunnerUpdate) + Send + Sync> = Arc::new(observer);
        let observe = observer.clone();
        self.process_manager =
            self.process_manager
                .with_lifecycle(Arc::new(move |update| match update {
                    ProcessUpdate::LogOpened { path, offset } => {
                        observe(RunnerUpdate::LogOpened { path, offset })
                    }
                    ProcessUpdate::Exited {
                        executable,
                        exit_code,
                    } => {
                        let name = executable.file_stem().unwrap_or_default().to_string_lossy();
                        // 从可执行文件名推断组件类别：进程内后端没有可执行文件，
                        // 由 runtime 侧自己上报（`Runtime` 事件），不会走这里。
                        let component = if name.contains("brush") {
                            PipelineComponent::Training
                        } else if name.contains("colmap") {
                            PipelineComponent::Reconstruction
                        } else {
                            PipelineComponent::Media
                        };
                        observe(RunnerUpdate::ProcessExited { component, exit_code });
                    }
                    _ => {}
                }));
        self.lifecycle = Some(observer);
        self
    }

    pub fn with_effectiveness_tracker(mut self, tracker: PlannerEffectivenessTracker) -> Self {
        self.effectiveness = Some(tracker);
        self
    }

    pub(crate) fn record_frame_metrics(
        &self,
        state: &PipelineStateFile,
        selected_count: u64,
        plan: Option<&FramePlan>,
    ) {
        let Some(tracker) = &self.effectiveness else {
            return;
        };
        let (source_width, source_height, source_item_count) =
            match (state.video.as_ref(), state.image_sequence.as_ref()) {
                (Some(video), _) => (
                    u64::from(video.width),
                    u64::from(video.height),
                    video.total_frames,
                ),
                (_, Some(images)) => (
                    u64::from(images.width),
                    u64::from(images.height),
                    images.image_count,
                ),
                _ => return,
            };
        let frames = state.frames.as_ref();
        let is_reshoot = state.reshoot.is_some();
        let is_planned_video = state.planner_enabled && state.video.is_some();
        let initial_selected_count = if is_reshoot {
            selected_count
        } else {
            frames
                .map(|frames| frames.initial_extracted_frames)
                .filter(|count| *count > 0)
                .unwrap_or(selected_count)
        };
        let has_candidates = is_planned_video && !is_reshoot;
        let candidate_count = if has_candidates {
            plan.map(|plan| plan.candidate_frames.len() as u64)
                .or_else(|| frames.map(|frames| frames.candidate_frames.len() as u64))
                .filter(|count| *count > 0)
                .unwrap_or(initial_selected_count)
        } else {
            initial_selected_count
        };
        let preset = state.preset.preset();
        let effective_fps = |count| {
            state
                .video
                .as_ref()
                .map(|video| count as f64 / video.duration.max(0.001))
        };
        tracker.record_frame_plan(PlannerFramePlanMetrics {
            source_width,
            source_height,
            source_item_count,
            configured_target_fps: is_planned_video.then_some(preset.initial_fps).flatten(),
            configured_candidate_fps: has_candidates.then_some(preset.rescue_max_fps).flatten(),
            effective_target_fps: effective_fps(initial_selected_count),
            effective_candidate_fps: (!is_reshoot)
                .then(|| effective_fps(candidate_count))
                .flatten(),
            initial_selected_count,
            candidate_count,
            minimum_frame_override_applied: is_planned_video
                && if is_reshoot {
                    state.video.as_ref().is_some_and(|video| {
                        preset.initial_fps.is_some_and(|fps| {
                            crate::video::minimum_selected_frames() as f64
                                / video.duration.max(0.001)
                                > fps
                        })
                    })
                } else {
                    plan.map(|plan| plan.minimum_frame_override_applied)
                        .or_else(|| frames.map(|frames| frames.minimum_frame_override_applied))
                        .unwrap_or(false)
                },
            minimum_frame_target_unreachable: is_planned_video
                && source_item_count < crate::video::minimum_selected_frames(),
        });
    }

    pub(crate) fn record_reconstruction_metrics(
        &self,
        report: &ReconstructionReport,
        acceleration: &crate::engines::AccelerationStatus,
        allow_two_view_tracks: bool,
        is_reshoot: bool,
    ) {
        let Some(tracker) = &self.effectiveness else {
            return;
        };
        tracker.record_gpu_vendor(if acceleration.detected_device_count > 0 {
            PlannerGpuVendor::Nvidia
        } else if cfg!(target_os = "macos") {
            PlannerGpuVendor::Apple
        } else {
            PlannerGpuVendor::Unknown
        });
        tracker.record_initial_reconstruction(PlannerInitialReconstructionMetrics {
            input_images: report.input_images,
            registered_images: report.registered_images,
            points_3d: report.points_3d,
            backend: acceleration.mode,
            allow_two_view_tracks,
        });
        if is_reshoot {
            tracker.record_bridge(PlannerBridgeMetrics {
                status: PlannerBridgeStatus::NotApplicable,
                trigger_ratio: bridge_trigger_ratio(),
                initial_input_images: report.input_images,
                initial_registered_images: report.registered_images,
                initial_points_3d: report.points_3d,
                final_input_images: report.input_images,
                final_registered_images: report.registered_images,
                final_points_3d: report.points_3d,
                ..Default::default()
            });
        }
    }

    pub(crate) fn record_brush_metrics(&self, state: &PipelineStateFile) {
        if let Some(tracker) = &self.effectiveness {
            tracker.restore_brush(
                state.brush_training.initial_resolved,
                state.brush_training.resolved,
                state.brush_training.oom_retry_used,
            );
        }
    }

    pub(crate) fn record_final_metrics(&self, output: &ProjectOutput) {
        if let Some(tracker) = &self.effectiveness {
            tracker.record_final_result(PlannerFinalResultMetrics {
                input_images: output.input_images,
                registered_images: output.registered_images,
                points_3d: output.points_3d,
                splat_count: output.splat_count,
                ply_size_bytes: output.file_size,
            });
        }
    }

    pub fn with_workspace_task_id(mut self, workspace_task_id: Option<uuid::Uuid>) -> Self {
        self.workspace_task_id = workspace_task_id;
        self
    }

    pub fn current_project_identity(&self) -> Option<(uuid::Uuid, Option<uuid::Uuid>)> {
        self.active_project
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .map(|project| (project.project_id, project.workspace_task_id))
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.events.started.elapsed().as_millis() as u64
    }

    pub fn elapsed_offset_ms(&self) -> u64 {
        self.elapsed_offset_ms.load(Ordering::Relaxed)
    }

    pub fn cancel(&self) {
        self.process_manager.cancel();
    }

    pub fn emit_terminal(&self, error: &SplatError) {
        self.events.terminal(error);
    }

    pub fn failure_context(&self) -> PipelineFailureContext {
        let failed_stage = *self
            .events
            .last_stage
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let project = self
            .active_project
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        PipelineFailureContext {
            failed_stage,
            project_id: project.as_ref().map(|value| value.project_id),
            project_path: project.as_ref().map(|value| value.project_path.clone()),
            logs_directory: project.map(|value| value.logs_directory),
        }
    }

    pub fn current_acceleration(&self) -> Option<crate::engines::AccelerationStatus> {
        self.current_acceleration
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// 任务启动前的组件校验。
    ///
    /// 只认**中性组件类别**（媒体 / 重建 / 训练）：具体是哪个二进制、哪个后端，
    /// 由 `engines::health` 与 `engines::backend` 各自负责回答。
    pub async fn verify_pipeline_engines(
        &self,
    ) -> Result<crate::engines::AccelerationStatus> {
        let statuses = self.engines.check_all_for_task().await;
        for required in [
            ComponentKind::Media,
            ComponentKind::Reconstruction,
            ComponentKind::Training,
        ] {
            let status = statuses
                .iter()
                .find(|status| status.kind == required)
                .expect("all component kinds returned");
            if !status.ready {
                let location = status
                    .location
                    .clone()
                    .unwrap_or_else(|| required.label().to_string());
                // 组件在但起不来 → EngineStart；组件根本不在 → ComponentMissing。
                let present = std::path::Path::new(&location).exists();
                return Err(if present {
                    SplatError::EngineStart {
                        component: required.label().to_string(),
                        detail: status.detail.clone(),
                    }
                } else {
                    SplatError::ComponentMissing(location)
                });
            }
        }
        // 重建后端的就绪校验（外部 CLI 校验二进制；进程内后端是 no-op）。
        self.backend.require_ready()?;
        brush::require_verified_cli(&self.engines.brush)?;
        let acceleration = statuses
            .into_iter()
            .find(|status| status.kind == ComponentKind::Reconstruction)
            .and_then(|status| status.acceleration)
            .ok_or_else(|| {
                SplatError::UnsupportedBackend("无法确定重建后端的加速状态".into())
            })?;
        *self
            .current_acceleration
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(acceleration.clone());
        Ok(acceleration)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn prepare_frames(
        &self,
        input: &Path,
        quality: Quality,
        output: &Path,
        colmap_output: &Path,
        masks: &Path,
        logs: Option<&Path>,
        planner_enabled: bool,
        probed_video: Option<VideoInfo>,
        target_dimensions: Option<(u32, u32)>,
        panorama: bool,
    ) -> Result<PreparedFrames> {
        self.events
            .stage(PipelineStage::ProbingVideo, 0.0, "正在读取视频信息");
        let video = match probed_video {
            Some(video) => video,
            None => {
                self.media
                    .probe_video(
                        &self.media_context(logs.map(|path| path.join("ffprobe.log")), None),
                        input,
                    )
                    .await?
            }
        };
        let probe_message = if video.has_alpha {
            format!(
                "视频 {:.1} 秒 · {:.2} FPS · {}×{} · 检测到 Alpha 通道（{}）",
                video.duration, video.fps, video.width, video.height, video.pixel_format
            )
        } else {
            format!(
                "视频 {:.1} 秒 · {:.2} FPS · {}×{}",
                video.duration, video.fps, video.width, video.height
            )
        };
        self.events
            .stage(PipelineStage::ProbingVideo, 1.0, probe_message);

        self.events
            .stage(PipelineStage::PlanningFrames, 0.0, "正在规划均匀抽帧");
        let mut plan = if planner_enabled {
            QualityV2FrameSelection.create_plan(&video, &quality.preset())
        } else {
            UniformRatioFrameSelection.create_plan(&video, &quality.preset())
        };
        // A 2:1 target is set only by the persisted panorama branch. Keep real
        // decoder PTS and indices, including a separate rescue candidate pool.
        if panorama {
            target_dimensions.ok_or_else(|| SplatError::Process("全景抽帧缺少目标尺寸".into()))?;
            self.events.panorama_phase.store(4, Ordering::Relaxed);
            self.events.send(
                PipelineStage::ExtractingFrames,
                Some(PipelineComponent::Media),
                EventKind::Stage,
                EventLevel::Info,
                None,
                true,
                "正在读取视频帧时间信息",
                None,
                None,
                None,
            );
            let times = crate::video::panorama::actual_frame_plan(
                &self.engines.ffprobe,
                input,
                &mut plan,
                &self.process_manager,
            )
            .await?;
            self.events.panorama_phase.store(1, Ordering::Relaxed);
            self.events
                .stage(PipelineStage::ExtractingFrames, 0.0, "正在提取全景画面");
            tokio::fs::write(
                output
                    .parent()
                    .unwrap_or(output)
                    .join("panorama-frame-times.json"),
                serde_json::to_vec_pretty(&times)?,
            )
            .await?;
        }
        self.events.stage(
            PipelineStage::PlanningFrames,
            1.0,
            format!(
                "预计提取 {} 帧（目标 {:.2} FPS，候选池 {} 帧 / 最高约 {:.2} FPS）",
                plan.estimated_frames,
                plan.sampling_fps,
                plan.candidate_frames.len(),
                plan.candidate_frames.len() as f64 / video.duration.max(f64::EPSILON)
            ),
        );

        self.events.stage(
            PipelineStage::ExtractingFrames,
            0.0,
            if video.has_alpha {
                "正在同步提取透明 PNG 画面与掩膜"
            } else {
                "FFmpeg 开始提取画面"
            },
        );
        let observer = self.process_observer(
            PipelineStage::ExtractingFrames,
            PipelineComponent::Media,
            Some(plan.estimated_frames),
            ObserverMode::Ffmpeg,
        );
        let extraction = if planner_enabled || panorama {
            self.media
                .extract_selected_frames(
                    &self.media_context(
                        logs.map(|path| path.join("ffmpeg.log")),
                        Some(observer.clone()),
                    ),
                    crate::media::SelectedFrameExtraction {
                        base: crate::media::FrameExtraction {
                            input: input.to_path_buf(),
                            output_directory: output.to_path_buf(),
                            colmap_directory: colmap_output.to_path_buf(),
                            mask_directory: masks.to_path_buf(),
                            has_alpha: video.has_alpha,
                            target_dimensions,
                        },
                        selected_frames: plan.selected_frames.clone(),
                    },
                )
                .await?
        } else {
            self.media
                .extract_uniform_frames(
                    &self.media_context(
                        logs.map(|path| path.join("ffmpeg.log")),
                        Some(observer.clone()),
                    ),
                    crate::media::UniformFrameExtraction {
                        base: crate::media::FrameExtraction {
                            input: input.to_path_buf(),
                            output_directory: output.to_path_buf(),
                            colmap_directory: colmap_output.to_path_buf(),
                            mask_directory: masks.to_path_buf(),
                            has_alpha: video.has_alpha,
                            target_dimensions,
                        },
                        plan: plan.clone(),
                    },
                )
                .await?
        };
        if let Some(expected) = target_dimensions {
            if (extraction.width, extraction.height) != expected {
                return Err(SplatError::Process(format!(
                    "FFmpeg produced {}x{} frames; expected {}x{}",
                    extraction.width, extraction.height, expected.0, expected.1
                )));
            }
        }
        self.events.stage(
            PipelineStage::ExtractingFrames,
            1.0,
            if extraction.has_alpha {
                format!(
                    "已提取 {} 张透明 PNG 和 {} 张 Mask",
                    extraction.frame_count, extraction.mask_count
                )
            } else {
                format!("已提取 {} 帧", extraction.frame_count)
            },
        );
        Ok(PreparedFrames {
            input_type: ProjectInputType::Video,
            video: Some(video),
            image_sequence: None,
            plan,
            extracted_frames: extraction.frame_count,
            image_format: extraction.image_format.as_str().into(),
            mask_count: extraction.mask_count,
            has_alpha: extraction.has_alpha,
            working_width: extraction.width,
            working_height: extraction.height,
        })
    }

    pub async fn prepare_images(
        &self,
        input: &Path,
        quality: Quality,
        output: &Path,
        colmap_output: &Path,
        masks: &Path,
        probe_already_complete: bool,
    ) -> Result<PreparedFrames> {
        if !probe_already_complete {
            self.events
                .stage(PipelineStage::ProbingVideo, 0.0, "正在快速读取图片头");
        }
        let source = input.to_path_buf();
        let cancellation = self.process_manager.child_token();
        let scan = crate::presets::spawn_pipeline_blocking(move || {
            scan_image_sequence(&source, Some(&cancellation))
        })
        .await
        .map_err(|error| SplatError::Process(format!("图片序列分析任务失败：{error}")))??;
        let image_sequence = scan.info.clone();
        self.events.stage(
            PipelineStage::ProbingVideo,
            1.0,
            format!(
                "图片序列 {} 张 · {}×{}{}",
                image_sequence.image_count,
                image_sequence.width,
                image_sequence.height,
                if image_sequence.has_alpha {
                    " · 检测到 Alpha 通道"
                } else {
                    ""
                }
            ),
        );
        let plan = crate::video::create_image_plan(&image_sequence, &quality.preset());
        self.events.stage(
            PipelineStage::PlanningFrames,
            1.0,
            format!("将处理全部 {} 张图片", image_sequence.image_count),
        );
        self.events.stage(
            PipelineStage::ExtractingFrames,
            0.0,
            if image_sequence.has_alpha {
                "正在建立画面链接并检查 Alpha 通道"
            } else {
                "正在建立画面链接"
            },
        );
        let frames = output.to_path_buf();
        let colmap_frames = colmap_output.to_path_buf();
        let mask_root = masks.to_path_buf();
        let events = self.events.clone();
        let observer: ImagePreparationObserver = Arc::new(move |progress| {
            let message = match progress.phase {
                ImagePreparationPhase::LinkingFrames => format!(
                    "正在建立画面链接或复制 {}/{} 张",
                    progress.current, progress.total
                ),
                ImagePreparationPhase::InspectingAlpha => format!(
                    "正在检测 Alpha 并生成 Mask {}/{} 张",
                    progress.current, progress.total
                ),
                ImagePreparationPhase::WritingOpaqueMasks => format!(
                    "正在补全不透明 Mask {}/{} 张",
                    progress.current, progress.total
                ),
                ImagePreparationPhase::Validating if progress.current == progress.total => {
                    "图片与 Mask 完整性校验完成".into()
                }
                ImagePreparationPhase::Validating => "正在校验图片与 Mask".into(),
            };
            events.send(
                PipelineStage::ExtractingFrames,
                Some(PipelineComponent::System),
                EventKind::Stage,
                EventLevel::Info,
                Some(progress.stage_progress),
                false,
                message,
                Some(progress.current),
                Some(progress.total),
                Some("images"),
            );
        });
        let cancellation = self.process_manager.child_token();
        let prepared = crate::presets::spawn_pipeline_blocking(move || {
            prepare_scanned_image_sequence(
                scan,
                &frames,
                &colmap_frames,
                &mask_root,
                ImageSequenceNaming::Primary,
                Some(observer),
                Some(&cancellation),
            )
        })
        .await
        .map_err(|error| SplatError::Process(format!("图片序列准备任务失败：{error}")))??;
        self.events.stage(
            PipelineStage::ExtractingFrames,
            1.0,
            if prepared.has_alpha {
                format!(
                    "已准备 {} 张图片和 {} 张 Mask",
                    prepared.image_count, prepared.mask_count
                )
            } else {
                format!("已准备 {} 张图片", prepared.image_count)
            },
        );
        let working_width = image_sequence.width;
        let working_height = image_sequence.height;
        Ok(PreparedFrames {
            input_type: ProjectInputType::Images,
            video: None,
            image_sequence: Some(image_sequence),
            plan,
            extracted_frames: prepared.image_count,
            image_format: "images".into(),
            mask_count: prepared.mask_count,
            has_alpha: prepared.has_alpha,
            working_width,
            working_height,
        })
    }

    async fn prepare_panorama_images(
        &self,
        input: &Path,
        quality: Quality,
        config: &crate::video::panorama::PanoramaCheckpoint,
        frames: &Path,
        colmap: &Path,
        masks: &Path,
    ) -> Result<PreparedFrames> {
        let source = input.to_path_buf();
        let config = config.clone();
        let (frames, colmap, masks) = (
            frames.to_path_buf(),
            colmap.to_path_buf(),
            masks.to_path_buf(),
        );
        let cancellation = self.process_manager.child_token();
        let events = self.events.clone();
        crate::presets::spawn_pipeline_blocking(move || {
            let scan = crate::video::image_projection::analyze(&source)?;
            let plan = crate::video::create_image_plan(&scan.info, &quality.preset());
            events.stage(
                PipelineStage::PlanningFrames,
                1.0,
                format!("将处理全部 {} 张图片", scan.info.image_count),
            );
            events.stage(
                PipelineStage::ExtractingFrames,
                0.0,
                format!("正在准备全景图片 0/{}", scan.info.image_count),
            );
            let prepared = crate::video::image_projection::prepare(
                &source,
                &config,
                &frames,
                &colmap,
                &masks,
                &cancellation,
                |current, total| {
                    events.send(
                        PipelineStage::ExtractingFrames,
                        Some(PipelineComponent::System),
                        EventKind::Stage,
                        EventLevel::Info,
                        Some(current as f32 / total.max(1) as f32),
                        false,
                        format!("正在准备全景图片 {current}/{total}"),
                        Some(current),
                        Some(total),
                        Some("images"),
                    );
                },
            )?;
            Ok(PreparedFrames {
                input_type: ProjectInputType::Images,
                video: None,
                image_sequence: Some(scan.info),
                plan,
                extracted_frames: prepared.image_count,
                image_format: "images".into(),
                mask_count: prepared.mask_count,
                has_alpha: prepared.has_alpha,
                working_width: config.sfm_width,
                working_height: config.sfm_height,
            })
        })
        .await
        .map_err(|e| SplatError::Process(e.to_string()))?
    }

    pub async fn generate(
        &self,
        input: &Path,
        quality: Quality,
        projects_root: &Path,
    ) -> Result<PipelineResult> {
        self.generate_with_manager(
            input,
            quality,
            ProjectManager::with_root(projects_root.to_path_buf()),
        )
        .await
    }

    pub async fn generate_for_diagnostics(
        &self,
        input: &Path,
        quality: Quality,
        projects_root: &Path,
    ) -> Result<PipelineResult> {
        self.generate_with_manager(
            input,
            quality,
            ProjectManager::for_diagnostics(projects_root.to_path_buf()),
        )
        .await
    }

    async fn generate_with_manager(
        &self,
        input: &Path,
        quality: Quality,
        project_manager: ProjectManager,
    ) -> Result<PipelineResult> {
        let project_manager = project_manager.with_input_boundary(self.input_boundary.clone());
        let acceleration = self.verify_pipeline_engines().await?;
        self.events.acceleration(acceleration.clone());
        let (paths, mut metadata) = if input.is_dir() {
            self.events
                .stage(PipelineStage::ProbingVideo, 0.0, "正在快速读取图片头");
            let events = self.events.clone();
            let observer: ProjectImportObserver = Arc::new(move |progress| {
                let ratio = if progress.total == 0 {
                    1.0
                } else {
                    progress.current as f32 / progress.total as f32
                };
                events.send(
                    PipelineStage::ProbingVideo,
                    Some(PipelineComponent::System),
                    EventKind::Stage,
                    EventLevel::Info,
                    Some(0.1 + 0.9 * ratio),
                    false,
                    format!("正在导入图片 {}/{} 张", progress.current, progress.total),
                    Some(progress.current),
                    Some(progress.total),
                    Some("images"),
                );
            });
            project_manager
                .create_with_progress(
                    input,
                    quality,
                    Some(observer),
                    Some(self.process_manager.child_token()),
                )
                .await?
        } else {
            project_manager.create(input, quality).await?
        };
        metadata.workspace_task_id = self.workspace_task_id;
        project_manager
            .write_metadata(&paths.metadata, &metadata)
            .await?;
        let mut state = project_manager.read_state(&paths.state).await?;
        state.video_projection = self.video_projection;
        if state.input_type == ProjectInputType::Images {
            state.image_matching_strategy = self.image_matching_strategy;
        }
        state.configuration = Some((*crate::presets::pipeline_optimization_config()).clone());
        state.planner_enabled = self.planner_enabled;
        state.resolution_policy_version = self
            .planner_enabled
            .then_some(PLANNER_RESOLUTION_POLICY_VERSION);
        project_manager.write_state(&paths.state, &state).await?;
        self.execute_project(project_manager, paths, &mut metadata, state, &acceleration)
            .await
    }

    /// 从已完成项目派生一个高清补拍项目。
    ///
    /// 源项目**只读**：它的输入画面与 final.ply 从不会被移动或覆盖。派生项目拥有自己的
    /// 画面目录，由"源项目的画面 + 用户提供的补拍素材"融合而成，随后重跑整条重建流水线。
    pub async fn inspect_reshoot_source(
        &self,
        project_id: uuid::Uuid,
    ) -> Result<ReshootSourceInfo> {
        let (project, metadata) = catalog::load_registered_project(project_id).await?;
        let mut info = ReshootSourceInfo {
            project_id: project_id.to_string(),
            project_name: metadata.name.clone(),
            quality: metadata.quality,
            planner_enabled: false,
            camera_id: 0,
            camera_model: String::new(),
            width: 0,
            height: 0,
            source_image_count: 0,
            eligible: false,
            reason: None,
            imported: metadata.imported.is_some(),
        };
        if !catalog::project_is_durably_completed(&project, &metadata).await {
            info.reason = Some("只有已完成且结果完整的项目可以进行高清补拍".into());
            return Ok(info);
        }
        if let Some(imported) = &metadata.imported {
            let Some(colmap) = imported.colmap.as_ref() else {
                info.reason = Some("外部项目没有可复用的 COLMAP 数据".into());
                return Ok(info);
            };
            if !colmap.reshoot_eligible {
                info.reason = colmap
                    .reshoot_reason
                    .clone()
                    .or_else(|| Some("外部 COLMAP 数据不满足高清补拍条件".into()));
                return Ok(info);
            }
            let colmap = colmap.clone();
            let parsed = match crate::presets::spawn_pipeline_blocking(move || {
                crate::imports::revalidate_imported_colmap(&colmap)
            })
            .await
            .map_err(|error| SplatError::Process(format!("读取外部相机信息失败：{error}")))?
            {
                Ok(parsed) => parsed,
                Err(error) => {
                    info.reason = Some(error.to_string());
                    return Ok(info);
                }
            };
            info.camera_id = parsed.0.id;
            info.camera_model = parsed.0.model;
            info.width = parsed.0.width;
            info.height = parsed.0.height;
            info.source_image_count = parsed.1.len() as u64;
            info.eligible = true;
            return Ok(info);
        }
        let source_state: PipelineStateFile =
            serde_json::from_slice(&tokio::fs::read(project.join("state.json")).await?)?;
        info.planner_enabled = source_state.planner_enabled;
        if source_state.panorama.is_some() {
            info.reason = Some("全景项目暂不支持高清补拍。".into());
            return Ok(info);
        }
        if source_state
            .frames
            .as_ref()
            .and_then(|frames| frames.colmap_input_version)
            != Some(COLMAP_INPUT_VERSION)
        {
            info.reason = Some(
                "原项目使用旧版或未知的 COLMAP 图片处理语义。为避免混用特征缓存，请先使用当前版本从原素材重新生成项目，再进行补拍。"
                    .into(),
            );
            return Ok(info);
        }
        let frames = project.join("work/frames");
        let colmap_frames = project.join("work/colmap_frames");
        if !colmap_frames.is_dir() {
            info.reason = Some("原项目缺少 COLMAP 专用图片，无法安全进行增量补拍。".into());
            return Ok(info);
        }
        if let Err(error) = validate_current_colmap_input_snapshot(&project, &source_state).await {
            info.reason = Some(format!(
                "原项目的 COLMAP 专用图片不完整或与原始帧不一致，无法安全进行增量补拍：{error}"
            ));
            return Ok(info);
        }
        let database = project.join("work/colmap/database.db");
        if !database.is_file()
            || std::fs::metadata(&database).map_or(true, |value| value.len() == 0)
        {
            info.reason = Some("原项目缺少相机重建数据库，无法进行增量补拍".into());
            return Ok(info);
        }
        let reconstruction =
            match reusable_reconstruction(&project, &frames, metadata.output.as_ref()).await {
                Ok(value) => value,
                Err(error) => {
                    info.reason = Some(format!("原项目的相机重建结果不可用：{error}"));
                    return Ok(info);
                }
            };
        let expected_registered_images = reconstruction.report.registered_images;
        let model = reconstruction.model;
        let database_for_validation = database.clone();
        let parsed = crate::presets::spawn_pipeline_blocking(move || {
            let camera = read_single_camera(&model)?;
            let images = read_registered_images(&model)?;
            validate_database_camera(&database_for_validation, &camera)?;
            Ok::<_, SplatError>((camera, images))
        })
        .await
        .map_err(|error| SplatError::Process(format!("读取原项目相机信息失败：{error}")))??;
        if parsed.1.len() as u64 != expected_registered_images {
            info.reason = Some("原项目的注册图片记录与稀疏模型统计不一致".into());
            return Ok(info);
        }
        if parsed.1.iter().any(|image| image.camera_id != parsed.0.id) {
            info.reason = Some("原项目包含多个相机，暂时不能进行同相机增量补拍".into());
            return Ok(info);
        }
        info.camera_id = parsed.0.id;
        info.camera_model = parsed.0.model;
        info.width = parsed.0.width;
        info.height = parsed.0.height;
        info.source_image_count = parsed.1.len() as u64;
        info.eligible = true;
        Ok(info)
    }

    pub async fn probe_reshoot_input(
        &self,
        project_id: uuid::Uuid,
        input: &Path,
        input_type: ReshootInputType,
    ) -> Result<ReshootInputInfo> {
        let source = self.inspect_reshoot_source(project_id).await?;
        if !source.eligible {
            return Err(SplatError::Process(
                source
                    .reason
                    .unwrap_or_else(|| "原项目不能进行高清补拍".into()),
            ));
        }
        let (source_project, source_metadata) =
            catalog::load_registered_project(project_id).await?;
        ensure_reshoot_input_is_new(
            &source_project,
            &source_metadata.source_path,
            input,
            input_type,
        )
        .await?;
        let samples = catalog::runtime_samples().await;
        let (image_count, duration, width, height, estimated_frames, has_alpha, estimate) =
            match input_type {
                ReshootInputType::Images => {
                    let path = input.to_path_buf();
                    let images = crate::presets::spawn_pipeline_blocking(move || {
                        crate::video::analyze_image_sequence(&path)
                    })
                    .await
                    .map_err(|error| SplatError::Process(format!("分析补拍图片失败：{error}")))??;
                    let plan = crate::video::create_image_plan(&images, &source.quality.preset());
                    let estimate = estimate_runtime_for_images(
                        source.source_image_count + images.image_count,
                        images.width.max(images.height),
                        &FramePlan {
                            retention_ratio: 1.0,
                            sampling_fps: 0.0,
                            estimated_frames: source.source_image_count + images.image_count,
                            ..FramePlan::default()
                        },
                        source.quality,
                        &samples,
                    );
                    (
                        Some(images.image_count),
                        None,
                        images.width,
                        images.height,
                        plan.estimated_frames,
                        images.has_alpha,
                        estimate,
                    )
                }
                ReshootInputType::Video => {
                    let video =
                        self.media
                            .probe_video(&self.media_context(None, None), input)
                            .await?;
                    let plan =
                        UniformRatioFrameSelection.create_plan(&video, &source.quality.preset());
                    let target_width = u32::try_from(source.width).unwrap_or(u32::MAX);
                    let target_height = u32::try_from(source.height).unwrap_or(u32::MAX);
                    let can_scale = video_can_scale_to(&video, target_width, target_height);
                    let dimensions = if can_scale {
                        (target_width, target_height)
                    } else {
                        scaled_video_dimensions(&video, target_width.max(target_height))
                    };
                    let estimate = estimate_runtime(
                        &video,
                        &FramePlan {
                            retention_ratio: plan.retention_ratio,
                            sampling_fps: plan.sampling_fps,
                            estimated_frames: source.source_image_count + plan.estimated_frames,
                            ..FramePlan::default()
                        },
                        source.quality,
                        &samples,
                    );
                    (
                        None,
                        Some(video.duration),
                        dimensions.0,
                        dimensions.1,
                        plan.estimated_frames,
                        video.has_alpha,
                        estimate,
                    )
                }
            };
        let compatible = u64::from(width) == source.width && u64::from(height) == source.height;
        Ok(ReshootInputInfo {
            input_type,
            image_count,
            duration,
            prepared_width: width,
            prepared_height: height,
            estimated_frames,
            has_alpha,
            mask_count: if has_alpha { estimated_frames } else { 0 },
            compatible,
            incompatibility_reason: (!compatible).then(|| {
                format!(
                    "补拍画面必须与原项目保持相同分辨率：需要 {}×{}，当前为 {}×{}",
                    source.width, source.height, width, height
                )
            }),
            estimate,
        })
    }

    pub async fn generate_incremental_reshoot(
        &self,
        source_project_id: uuid::Uuid,
        reshoot_input: &Path,
        input_type: ReshootInputType,
        projects_root: &Path,
        requested_quality: Quality,
    ) -> Result<PipelineResult> {
        let source = self.inspect_reshoot_source(source_project_id).await?;
        if !source.eligible {
            return Err(SplatError::Process(
                source
                    .reason
                    .unwrap_or_else(|| "原项目不能进行高清补拍".into()),
            ));
        }
        if input_type == ReshootInputType::Images {
            self.events
                .stage(PipelineStage::ProbingVideo, 0.0, "正在快速读取补拍图片头");
        }
        let input = self
            .probe_reshoot_input(source_project_id, reshoot_input, input_type)
            .await?;
        if !input.compatible {
            return Err(SplatError::Process(
                input
                    .incompatibility_reason
                    .unwrap_or_else(|| "补拍素材与原项目相机尺寸不一致".into()),
            ));
        }
        let acceleration = self.verify_pipeline_engines().await?;
        self.events.acceleration(acceleration.clone());
        let (source_root, source_metadata) =
            catalog::load_registered_project(source_project_id).await?;
        let source_state: PipelineStateFile = if source_metadata.imported.is_some() {
            let mut state =
                PipelineStateFile::created_for(requested_quality, ProjectInputType::Images);
            state.planner_enabled = self.planner_enabled;
            state
        } else {
            serde_json::from_slice(&tokio::fs::read(source_root.join("state.json")).await?)?
        };
        let quality = if source_metadata.imported.is_some() {
            requested_quality
        } else {
            source_metadata.quality
        };
        let project_manager = ProjectManager::with_root(projects_root.to_path_buf());
        let (paths, mut metadata) = if input_type == ReshootInputType::Images {
            let events = self.events.clone();
            let observer: ProjectImportObserver = Arc::new(move |progress| {
                let ratio = if progress.total == 0 {
                    1.0
                } else {
                    progress.current as f32 / progress.total as f32
                };
                events.send(
                    PipelineStage::ProbingVideo,
                    Some(PipelineComponent::System),
                    EventKind::Stage,
                    EventLevel::Info,
                    Some(0.1 + 0.9 * ratio),
                    false,
                    format!(
                        "正在导入补拍图片 {}/{} 张",
                        progress.current, progress.total
                    ),
                    Some(progress.current),
                    Some(progress.total),
                    Some("images"),
                );
            });
            project_manager
                .create_with_progress(
                    reshoot_input,
                    quality,
                    Some(observer),
                    Some(self.process_manager.child_token()),
                )
                .await?
        } else {
            project_manager.create(reshoot_input, quality).await?
        };
        let stored_source = metadata.source_path.clone();
        metadata.name = reshoot_project_name(&source_metadata.name);
        metadata.workspace_task_id = self.workspace_task_id;
        metadata.transform = source_metadata.transform;
        metadata.editing = Default::default();
        metadata.reshoot = Some(ReshootProvenance {
            source_project_id,
            source_project_path: source_root.clone(),
            source_final_ply: catalog::project_model_path(&source_root, &source_metadata)
                .unwrap_or_else(|| source_root.join("final.ply")),
            reshoot_source_path: stored_source,
            camera_id: source.camera_id,
            camera_model: source.camera_model,
            width: source.width,
            height: source.height,
            has_alpha: input.has_alpha,
            mask_count: input.mask_count,
            source_image_count: source.source_image_count,
            reshoot_frame_count: input.estimated_frames,
            registered_reshoot_count: 0,
        });
        project_manager
            .write_metadata(&paths.metadata, &metadata)
            .await?;
        let mut state = project_manager.read_state(&paths.state).await?;
        if source_metadata.imported.is_some() {
            state.planner_enabled = self.planner_enabled;
        } else {
            inherit_reshoot_planner_settings(&source_state, &mut state);
        }
        state.reshoot = Some(ReshootState {
            source_image_count: source.source_image_count,
            reshoot_frame_count: input.estimated_frames,
            mask_count: input.mask_count,
            has_alpha: input.has_alpha,
            ..Default::default()
        });
        project_manager.write_state(&paths.state, &state).await?;
        self.execute_project(project_manager, paths, &mut metadata, state, &acceleration)
            .await
    }

    pub async fn resume(&self, project_id: uuid::Uuid) -> Result<PipelineResult> {
        let acceleration = self.verify_pipeline_engines().await?;
        self.events.acceleration(acceleration.clone());
        let (project, mut metadata) = catalog::load_registered_project(project_id).await?;
        if catalog::project_is_durably_completed(&project, &metadata).await {
            return Err(SplatError::Process("该项目已经完成，无需继续".into()));
        }
        let source_available = match metadata.input_type {
            ProjectInputType::Video => metadata.source_path.is_file(),
            ProjectInputType::Images => metadata.source_path.is_dir(),
        };
        if !source_available {
            return Err(SplatError::Process("项目源素材缺失，无法继续".into()));
        }
        let paths = ProjectPaths::existing(project_id, project.clone());
        let project_manager = ProjectManager::with_root(
            project
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| project.clone()),
        );
        let state = project_manager.read_state(&paths.state).await?;
        if state.preset != metadata.quality {
            return Err(SplatError::Process(
                "项目档位与检查点不一致，无法安全继续".into(),
            ));
        }
        self.execute_project(project_manager, paths, &mut metadata, state, &acceleration)
            .await
    }

    async fn execute_project(
        &self,
        project_manager: ProjectManager,
        paths: ProjectPaths,
        metadata: &mut ProjectMetadata,
        state: PipelineStateFile,
        acceleration: &crate::engines::AccelerationStatus,
    ) -> Result<PipelineResult> {
        if let Some(observer) = &self.lifecycle {
            observer(RunnerUpdate::Project(Box::new(metadata.clone())));
            observer(RunnerUpdate::Configuration(
                crate::tasks::checkpoint_snapshot(&state),
            ));
        }
        let lifecycle = self.lifecycle.clone();
        let project_manager = project_manager.with_checkpoint_observer(Arc::new(move |state| {
            if let Some(observer) = &lifecycle {
                observer(RunnerUpdate::Configuration(
                    crate::tasks::checkpoint_snapshot(state),
                ));
            }
        }));
        let previous_duration = metadata.duration_ms.unwrap_or(0);
        self.elapsed_offset_ms
            .store(previous_duration, Ordering::Relaxed);
        *self
            .active_project
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(ActiveProjectContext {
            project_id: paths.id,
            workspace_task_id: metadata.workspace_task_id,
            project_path: paths.project.clone(),
            logs_directory: paths.logs.clone(),
        });
        let started = Instant::now();
        metadata.status = ProjectStatus::Running;
        metadata.started_at = Some(Utc::now());
        metadata.completed_at = None;
        metadata.failure_message = None;
        project_manager
            .write_metadata(&paths.metadata, metadata)
            .await?;
        let result = if metadata.reshoot.is_some() {
            self.run_incremental_reshoot_project(
                &project_manager,
                &paths,
                metadata,
                state,
                acceleration,
            )
            .await
        } else {
            self.run_project(&project_manager, &paths, metadata, state, acceleration)
                .await
        };

        if let Err(error) = &result {
            let cancelled = matches!(error, SplatError::Cancelled);
            metadata.status = if cancelled {
                ProjectStatus::Cancelled
            } else {
                ProjectStatus::Failed
            };
            metadata.completed_at = Some(Utc::now());
            metadata.duration_ms =
                Some(previous_duration.saturating_add(started.elapsed().as_millis() as u64));
            metadata.failure_message = Some(error.to_string());
            let _ = project_manager
                .write_metadata(&paths.metadata, metadata)
                .await;
            let mut state = project_manager
                .read_state(&paths.state)
                .await
                .unwrap_or_else(|_| {
                    PipelineStateFile::created_for(metadata.quality, metadata.input_type)
                });
            state = mark_state_terminal(state, cancelled);
            let _ = project_manager.write_state(&paths.state, &state).await;
        }
        result
    }

    async fn run_incremental_reshoot_project(
        &self,
        project_manager: &ProjectManager,
        paths: &ProjectPaths,
        metadata: &mut ProjectMetadata,
        mut state: PipelineStateFile,
        acceleration: &crate::engines::AccelerationStatus,
    ) -> Result<PipelineResult> {
        let provenance = metadata
            .reshoot
            .clone()
            .ok_or_else(|| SplatError::Process("补拍项目缺少来源信息".into()))?;
        let mut checkpoint = state.reshoot.clone().unwrap_or_default();
        let source_root = &provenance.source_project_path;
        let source_metadata: ProjectMetadata =
            serde_json::from_slice(&tokio::fs::read(source_root.join("project.json")).await?)?;
        let (source_frames, source_colmap_frames, source_database, source_model) = if let Some(
            imported,
        ) =
            &source_metadata.imported
        {
            let colmap = imported.colmap.as_ref().ok_or_else(|| {
                SplatError::Process("外部导入项目没有可复用的 COLMAP 数据".into())
            })?;
            if !colmap.reshoot_eligible {
                return Err(SplatError::Process(
                    colmap
                        .reshoot_reason
                        .clone()
                        .unwrap_or_else(|| "外部 COLMAP 数据不满足高清补拍条件".into()),
                ));
            }
            let images = colmap
                .images_path
                .clone()
                .filter(|path| path.is_dir())
                .ok_or_else(|| SplatError::Process("外部 COLMAP 项目缺少原始图片".into()))?;
            let database = colmap
                .database_path
                .clone()
                .filter(|path| path.is_file())
                .ok_or_else(|| SplatError::Process("外部 COLMAP 项目缺少 database.db".into()))?;
            (
                images.clone(),
                images,
                database,
                colmap.sparse_model_path.clone(),
            )
        } else {
            let state: PipelineStateFile =
                serde_json::from_slice(&tokio::fs::read(source_root.join("state.json")).await?)?;
            if state
                .frames
                .as_ref()
                .and_then(|frames| frames.colmap_input_version)
                != Some(COLMAP_INPUT_VERSION)
            {
                return Err(SplatError::Process("原项目使用旧版或未知的 COLMAP 图片处理语义，不能与新版补拍特征缓存混用。请先从原素材重新生成基线项目。".into()));
            }
            validate_current_colmap_input_snapshot(source_root, &state)
                .await
                .map_err(|error| {
                    SplatError::Process(format!(
                        "原项目的 COLMAP 专用图片不完整或与原始帧不一致，不能作为补拍基线：{error}"
                    ))
                })?;
            let frames = source_root.join("work/frames");
            let colmap_frames = source_root.join("work/colmap_frames");
            let reconstruction =
                reusable_reconstruction(source_root, &frames, source_metadata.output.as_ref())
                    .await?;
            (
                frames,
                colmap_frames,
                source_root.join("work/colmap/database.db"),
                reconstruction.model,
            )
        };
        archive_incompatible_colmap_input_cache(paths, &state).await?;
        if !source_database.is_file()
            || std::fs::metadata(&source_database).map_or(true, |value| value.len() == 0)
        {
            return Err(SplatError::Process(
                "原项目缺少可复用的 COLMAP 数据库".into(),
            ));
        }
        let source_model_for_validation = source_model.clone();
        let source_database_for_validation = source_database.clone();
        if let Some(colmap) = source_metadata
            .imported
            .as_ref()
            .and_then(|imported| imported.colmap.clone())
        {
            crate::presets::spawn_pipeline_blocking(move || {
                crate::imports::revalidate_imported_colmap(&colmap).map(|_| ())
            })
            .await
            .map_err(|error| SplatError::Process(format!("复检外部 COLMAP 数据失败：{error}")))??;
        }
        crate::presets::spawn_pipeline_blocking(move || {
            let camera = read_single_camera(&source_model_for_validation)?;
            validate_database_camera(&source_database_for_validation, &camera)
        })
        .await
        .map_err(|error| SplatError::Process(format!("校验补拍基线模型失败：{error}")))??;
        let source_model_files = reusable_model_files(&source_model)?;
        let base_model = paths.colmap.join("base-model");
        let database = paths.colmap.join("database.db");
        let base_database = paths.colmap.join("base-database.db");
        let reshoot_masks = paths.work.join("reshoot-masks");
        let reshoot_list = paths.colmap.join("reshoot-images.txt");
        let mapper_list = paths.colmap.join("mapper-images.txt");

        let snapshot_frames_valid = count_image_files(&paths.frames)
            .await
            .is_ok_and(|count| count >= provenance.source_image_count);
        let snapshot_colmap_frames_valid = count_image_files(&paths.colmap_frames)
            .await
            .is_ok_and(|count| count >= provenance.source_image_count);
        let snapshot_database_valid = tokio::fs::metadata(&database)
            .await
            .is_ok_and(|value| value.is_file() && value.len() > 0);
        let base_database_valid = tokio::fs::metadata(&base_database)
            .await
            .is_ok_and(|value| value.is_file() && value.len() > 0);
        if !checkpoint.source_snapshot_complete
            || checkpoint.colmap_input_version != Some(COLMAP_INPUT_VERSION)
            || !snapshot_frames_valid
            || !snapshot_colmap_frames_valid
            || !snapshot_database_valid
            || !base_database_valid
            || !model_snapshot_matches(&source_model_files, &base_model)
        {
            self.events.stage(
                PipelineStage::ExtractingFrames,
                0.0,
                "正在复用原项目的相机与重建结果",
            );
            reset_directory(&paths.frames).await?;
            reset_directory(&paths.colmap_frames).await?;
            reset_directory(&paths.colmap).await?;
            reset_directory(&reshoot_masks).await?;
            copy_image_files(&source_frames, &paths.frames).await?;
            copy_image_files(&source_colmap_frames, &paths.colmap_frames).await?;
            tokio::fs::copy(&source_database, &base_database).await?;
            tokio::fs::copy(&base_database, &database).await?;
            tokio::fs::create_dir_all(&base_model).await?;
            for source in &source_model_files {
                let name = source.file_name().expect("model file has name");
                tokio::fs::copy(&source, base_model.join(name)).await?;
            }
            checkpoint = ReshootState {
                source_snapshot_complete: true,
                source_image_count: provenance.source_image_count,
                reshoot_frame_count: provenance.reshoot_frame_count,
                mask_count: provenance.mask_count,
                has_alpha: provenance.has_alpha,
                colmap_input_version: Some(COLMAP_INPUT_VERSION),
                ..Default::default()
            };
            state.reshoot = Some(checkpoint.clone());
            project_manager.write_state(&paths.state, &state).await?;
        }

        if checkpoint.supplemental_frames_complete
            && (checkpoint.colmap_input_version != Some(COLMAP_INPUT_VERSION)
                || validate_reshoot_image_sequence(
                    &paths.frames,
                    &paths.colmap_frames,
                    &reshoot_masks,
                    checkpoint.reshoot_frame_count,
                    checkpoint.has_alpha,
                )
                .is_err())
        {
            checkpoint.supplemental_frames_complete = false;
            checkpoint.supplemental_features_complete = false;
            checkpoint.incremental_matching_complete = false;
            checkpoint.incremental_reconstruction_complete = false;
            state.brush_complete = false;
        }
        let prepared = if checkpoint.supplemental_frames_complete {
            validate_reshoot_image_sequence(
                &paths.frames,
                &paths.colmap_frames,
                &reshoot_masks,
                checkpoint.reshoot_frame_count,
                checkpoint.has_alpha,
            )?
        } else {
            remove_reshoot_images(&paths.frames).await?;
            remove_reshoot_images(&paths.colmap_frames).await?;
            reset_directory(&reshoot_masks).await?;
            self.events.stage(
                PipelineStage::ExtractingFrames,
                0.0,
                if provenance.has_alpha {
                    "正在检查补拍 Alpha 通道并按需生成 Mask"
                } else {
                    "正在准备补拍画面"
                },
            );
            let prepared_result: Result<PreparedImageSequence> = async {
                Ok(match metadata.input_type {
                ProjectInputType::Images => {
                    let source = metadata.source_path.clone();
                    let cancellation = self.process_manager.child_token();
                    let scan = crate::presets::spawn_pipeline_blocking(move || {
                        scan_image_sequence(&source, Some(&cancellation))
                    })
                    .await
                    .map_err(|error| SplatError::Process(format!("补拍图片扫描失败：{error}")))??;
                    state.image_sequence = Some(scan.info.clone());
                    let frames = paths.frames.clone();
                    let colmap_frames = paths.colmap_frames.clone();
                    let masks = reshoot_masks.clone();
                    let events = self.events.clone();
                    let observer: ImagePreparationObserver = Arc::new(move |progress| {
                        let message = match progress.phase {
                            ImagePreparationPhase::LinkingFrames => format!(
                                "正在建立补拍画面链接或复制 {}/{} 张",
                                progress.current, progress.total
                            ),
                            ImagePreparationPhase::InspectingAlpha => format!(
                                "正在检测补拍 Alpha 并生成 Mask {}/{} 张",
                                progress.current, progress.total
                            ),
                            ImagePreparationPhase::WritingOpaqueMasks => format!(
                                "正在补全补拍不透明 Mask {}/{} 张",
                                progress.current, progress.total
                            ),
                            ImagePreparationPhase::Validating
                                if progress.current == progress.total =>
                            {
                                "补拍图片与 Mask 完整性校验完成".into()
                            }
                            ImagePreparationPhase::Validating => "正在校验补拍图片与 Mask".into(),
                        };
                        events.send(
                            PipelineStage::ExtractingFrames,
                            Some(PipelineComponent::System),
                            EventKind::Stage,
                            EventLevel::Info,
                            Some(progress.stage_progress),
                            false,
                            message,
                            Some(progress.current),
                            Some(progress.total),
                            Some("images"),
                        );
                    });
                    let cancellation = self.process_manager.child_token();
                    crate::presets::spawn_pipeline_blocking(move || {
                        prepare_scanned_image_sequence(
                            scan,
                            &frames,
                            &colmap_frames,
                            &masks,
                            ImageSequenceNaming::Reshoot,
                            Some(observer),
                            Some(&cancellation),
                        )
                    })
                    .await
                    .map_err(|error| SplatError::Process(format!("补拍图片准备失败：{error}")))??
                }
                ProjectInputType::Video => {
                    let temporary = paths.work.join("reshoot-extract");
                    let temporary_colmap = paths.work.join("reshoot-extract-colmap");
                    let temporary_masks = paths.work.join("reshoot-extract-masks");
                    reset_directory(&temporary).await?;
                    reset_directory(&temporary_colmap).await?;
                    reset_directory(&temporary_masks).await?;
                    let supplemental_video = self
                        .media
                        .probe_video(
                            &self.media_context(Some(paths.logs.join("ffprobe.log")), None),
                            &metadata.source_path,
                        )
                        .await?;
                    state.video = Some(supplemental_video.clone());
                    let target_width = u32::try_from(provenance.width).map_err(|_| {
                        SplatError::Process("Reshoot target width exceeds supported range".into())
                    })?;
                    let target_height = u32::try_from(provenance.height).map_err(|_| {
                        SplatError::Process("Reshoot target height exceeds supported range".into())
                    })?;
                    if !video_can_scale_to(&supplemental_video, target_width, target_height) {
                        return Err(SplatError::Process(format!(
                            "Reshoot video cannot be scaled to the source camera resolution {}x{} without upscaling or changing aspect ratio",
                            target_width, target_height
                        )));
                    }
                    let extracted = self
                        .prepare_frames(
                            &metadata.source_path,
                            metadata.quality,
                            &temporary,
                            &temporary_colmap,
                            &temporary_masks,
                            Some(&paths.logs),
                            state.planner_enabled,
                            Some(supplemental_video),
                            Some((target_width, target_height)),
                            false,
                        )
                        .await?;
                    move_extracted_reshoot(
                        &temporary,
                        &temporary_colmap,
                        &temporary_masks,
                        &paths.frames,
                        &paths.colmap_frames,
                        &reshoot_masks,
                        extracted.has_alpha,
                    )
                    .await?;
                    validate_reshoot_image_sequence(
                        &paths.frames,
                        &paths.colmap_frames,
                        &reshoot_masks,
                        extracted.extracted_frames,
                        extracted.has_alpha,
                    )?
                }
                })
            }
            .await;
            let prepared = match prepared_result {
                Ok(prepared) => prepared,
                Err(error) => {
                    remove_reshoot_images(&paths.frames).await?;
                    remove_reshoot_images(&paths.colmap_frames).await?;
                    reset_directory(&reshoot_masks).await?;
                    return Err(error);
                }
            };
            validate_image_dimensions(
                &paths.frames,
                "reshoot_",
                provenance.width,
                provenance.height,
            )
            .await?;
            checkpoint.supplemental_frames_complete = true;
            checkpoint.reshoot_frame_count = prepared.image_count;
            checkpoint.mask_count = prepared.mask_count;
            checkpoint.has_alpha = prepared.has_alpha;
            checkpoint.colmap_input_version = Some(COLMAP_INPUT_VERSION);
            if let Some(reshoot) = metadata.reshoot.as_mut() {
                reshoot.reshoot_frame_count = prepared.image_count;
                reshoot.mask_count = prepared.mask_count;
                reshoot.has_alpha = prepared.has_alpha;
            }
            state.stage = PipelineStage::ExtractingFrames;
            state.reshoot = Some(checkpoint.clone());
            project_manager.write_state(&paths.state, &state).await?;
            project_manager
                .write_metadata(&paths.metadata, metadata)
                .await?;
            self.events.send(
                PipelineStage::ExtractingFrames,
                Some(PipelineComponent::System),
                EventKind::Progress,
                EventLevel::Info,
                Some(1.0),
                false,
                format!("补拍画面准备完成 · {} 张", prepared.image_count),
                Some(prepared.image_count),
                Some(prepared.image_count),
                Some("images"),
            );
            prepared
        };
        self.record_frame_metrics(&state, prepared.image_count, None);
        let reshoot_names = image_names_with_prefix(&paths.colmap_frames, "reshoot_").await?;
        write_image_list(&reshoot_list, &reshoot_names).await?;

        let backend_label = if acceleration.use_gpu() { "GPU" } else { "CPU" };
        let sfm_preset = metadata.quality.preset();
        if !checkpoint.supplemental_features_complete {
            tokio::fs::copy(&base_database, &database).await?;
            self.events.stage(
                PipelineStage::ExtractingFeatures,
                0.0,
                format!("正在使用 {backend_label} 提取补拍画面的信息"),
            );
            self.backend
                .extract_incremental_features(
                    &self.job_context(
                        paths.logs.join("colmap.log"),
                        Some(self.process_observer(
                    PipelineStage::ExtractingFeatures,
                    PipelineComponent::Reconstruction,
                    Some(prepared.image_count),
                    ObserverMode::BracketProgress,
                )),
                        acceleration.gpu_index(),
                    ),
                    IncrementalFeatureExtraction {
                        base: FeatureExtraction {
                            database: database.clone(),
                            images: Path::new("../colmap_frames").to_path_buf(),
                            image_list: Some(Path::new("reshoot-images.txt").to_path_buf()),
                            existing_camera_id: Some(provenance.camera_id),
                            masks: prepared.has_alpha.then(|| Path::new("../reshoot-masks").to_path_buf()),
                            max_image_size: (!state.planner_enabled).then_some(sfm_preset.sfm_max_image_size),
                            max_features: Some(if state.planner_enabled { sfm_preset.planned_sfm_max_features } else { sfm_preset.sfm_max_features }),
                        },
                    },
                )
                .await?;
            checkpoint.supplemental_features_complete = true;
            state.stage = PipelineStage::ExtractingFeatures;
            state.reshoot = Some(checkpoint.clone());
            project_manager.write_state(&paths.state, &state).await?;
        }

        if !checkpoint.incremental_matching_complete {
            self.events
                .stage(PipelineStage::Matching, 0.0, "正在补充新旧画面之间的联系");
            self.backend
                .match_exhaustive(
                    &self.job_context(
                        paths.logs.join("colmap.log"),
                        Some(self.process_observer(
                    PipelineStage::Matching,
                    PipelineComponent::Reconstruction,
                    Some(prepared.image_count),
                    ObserverMode::Matching,
                )),
                        acceleration.gpu_index(),
                    ),
                    Matching {
                        database: database.clone(),
                        loop_detection: false,
                    },
                )
                .await?;
            self.events.send(
                PipelineStage::Matching,
                Some(PipelineComponent::Reconstruction),
                EventKind::Progress,
                EventLevel::Info,
                Some(1.0),
                false,
                "补拍图像匹配完成",
                None,
                None,
                None,
            );
            checkpoint.incremental_matching_complete = true;
            state.stage = PipelineStage::Matching;
            state.reshoot = Some(checkpoint.clone());
            project_manager.write_state(&paths.state, &state).await?;
        }

        let sparse = paths.colmap.join("sparse");
        if checkpoint.incremental_reconstruction_complete
            && best_sparse_model(&paths.frames, &sparse).await.is_err()
        {
            checkpoint.incremental_reconstruction_complete = false;
            state.brush_complete = false;
        }
        if !checkpoint.incremental_reconstruction_complete {
            let base = base_model.clone();
            let original_names =
                crate::presets::spawn_pipeline_blocking(move || read_registered_images(&base))
                    .await
                    .map_err(|error| {
                        SplatError::Process(format!("读取原项目画面列表失败：{error}"))
                    })??
                    .into_iter()
                    .map(|image| image.name)
                    .collect::<Vec<_>>();
            let mut mapper_names = original_names;
            mapper_names.extend(reshoot_names.iter().cloned());
            write_image_list(&mapper_list, &mapper_names).await?;
            reset_directory(&sparse).await?;
            self.events.stage(
                PipelineStage::Reconstructing,
                0.0,
                "正在把补拍画面注册到原相机重建中",
            );
            self.backend
                .map_incremental(
                    &self.job_context(
                        paths.logs.join("colmap.log"),
                        Some(self.process_observer(
                    PipelineStage::Reconstructing,
                    PipelineComponent::Reconstruction,
                    Some(mapper_names.len() as u64),
                    ObserverMode::Mapper,
                )),
                        acceleration.gpu_index(),
                    ),
                    IncrementalMapping {
                        database: database.clone(),
                        images: Path::new("../colmap_frames").to_path_buf(),
                        input_model: Path::new("base-model").to_path_buf(),
                        output: sparse.clone(),
                        image_list: Path::new("mapper-images.txt").to_path_buf(),
                        automatic_optimization: state.planner_enabled,
                    },
                )
                .await?;
            checkpoint.incremental_reconstruction_complete = true;
            state.stage = PipelineStage::Reconstructing;
            state.reshoot = Some(checkpoint.clone());
            project_manager.write_state(&paths.state, &state).await?;
        }

        let (model, report) = best_sparse_model(&paths.frames, &sparse).await?;
        // Incremental Mapper does not enable two-view tracks or bridge backfill.
        self.record_reconstruction_metrics(&report, acceleration, false, true);
        let model_copy = model.clone();
        let registered_reshoot_count =
            crate::presets::spawn_pipeline_blocking(move || read_registered_images(&model_copy))
                .await
                .map_err(|error| SplatError::Process(format!("读取补拍注册结果失败：{error}")))??
                .into_iter()
                .filter(|image| image.name.starts_with("reshoot_"))
                .count() as u64;
        if registered_reshoot_count == 0 {
            return Err(SplatError::Process("补拍画面没有成功加入原项目。请使用同一设备和镜头重新拍摄，并确保新画面与原画面有足够重叠。".into()));
        }
        checkpoint.registered_reshoot_count = registered_reshoot_count;
        state.reconstruction_complete = true;
        state.reshoot = Some(checkpoint.clone());
        project_manager.write_state(&paths.state, &state).await?;
        if let Some(provenance) = metadata.reshoot.as_mut() {
            provenance.registered_reshoot_count = registered_reshoot_count;
            provenance.reshoot_frame_count = prepared.image_count;
            provenance.mask_count = prepared.mask_count;
            provenance.has_alpha = prepared.has_alpha;
        }

        if state.brush_complete && brush_candidate(&paths.brush).is_none() {
            state.brush_complete = false;
        }
        let candidate = if state.brush_complete {
            self.record_brush_metrics(&state);
            brush_candidate(&paths.brush)
                .ok_or_else(|| SplatError::Process("Brush 检查点文件缺失".into()))?
        } else {
            reset_directory(&paths.brush).await?;
            let dataset = prepare_brush_dataset(&paths.brush, &paths.frames, &model).await?;
            let source_long_edge = provenance
                .width
                .max(provenance.height)
                .min(u64::from(u32::MAX)) as u32;
            let resolved_brush = state.brush_training.resolved.unwrap_or_else(|| {
                resolve_brush_training_preset(
                    metadata.quality,
                    state.planner_enabled,
                    acceleration.usable_gpu_total_memory_mb(),
                    source_long_edge,
                    report.points_3d,
                )
            });
            if !state.brush_training.oom_retry_used {
                state
                    .brush_training
                    .initial_resolved
                    .get_or_insert(resolved_brush);
            }
            state.brush_training.resolved = Some(resolved_brush);
            self.record_brush_metrics(&state);
            project_manager.write_state(&paths.state, &state).await?;
            self.events.send(
                PipelineStage::TrainingSplats,
                Some(PipelineComponent::Training),
                EventKind::Stage,
                EventLevel::Info,
                None,
                true,
                "正在使用全部已注册画面重新训练高斯",
                Some(0),
                Some(resolved_brush.preset.total_steps as u64),
                Some("iterations"),
            );
            let candidate = self
                .train_brush_with_oom_retry(
                    project_manager,
                    paths,
                    &mut state,
                    metadata.quality,
                    acceleration,
                    &dataset,
                    report.points_3d,
                    resolved_brush,
                    crate::pipeline::estimate::estimate_brush_stage_ms(metadata.quality),
                )
                .await?;
            state.brush_complete = true;
            state.stage = PipelineStage::TrainingSplats;
            project_manager.write_state(&paths.state, &state).await?;
            candidate
        };

        let ply = inspect_gaussian_ply(&candidate)?;
        let final_ply = paths.project.join("final.ply");
        atomic_replace_file(&candidate, &final_ply).await?;
        state.stage = PipelineStage::Completed;
        project_manager.write_state(&paths.state, &state).await?;
        let completed_at = Utc::now();
        let duration_ms = metadata.duration_ms.unwrap_or(0).saturating_add(
            metadata
                .started_at
                .map(|started| (completed_at - started).num_milliseconds().max(0) as u64)
                .unwrap_or(0),
        );
        let reshoot_ratio = registered_reshoot_count as f64 / prepared.image_count.max(1) as f64;
        let warning = (reshoot_ratio < crate::reconstruction::good_registered_ratio()).then(|| format!(
            "补拍画面成功加入 {registered_reshoot_count}/{}，结果已生成，但部分补拍画面未能找到与原项目的联系。",
            prepared.image_count
        ));
        metadata.status = ProjectStatus::Completed;
        metadata.completed_at = Some(completed_at);
        metadata.duration_ms = Some(duration_ms);
        metadata.output = Some(ProjectOutput {
            final_ply: final_ply.clone(),
            file_size: ply.file_size,
            splat_count: ply.splat_count,
            input_images: report.input_images,
            registered_images: report.registered_images,
            registered_ratio: report.registered_ratio,
            points_3d: report.points_3d,
        });
        project_manager
            .write_metadata(&paths.metadata, metadata)
            .await?;
        if let Some(output) = &metadata.output {
            self.record_final_metrics(output);
        }
        self.events
            .stage(PipelineStage::Completed, 1.0, "高清补拍处理完成");
        Ok(PipelineResult {
            project_id: paths.id.to_string(),
            project_path: paths.project.clone(),
            final_ply,
            file_size: ply.file_size,
            splat_count: ply.splat_count,
            input_images: report.input_images,
            registered_images: report.registered_images,
            registered_ratio: report.registered_ratio,
            points_3d: report.points_3d,
            duration_ms,
            completed_at,
            warning,
            logs_directory: paths.logs.clone(),
            source_duration_seconds: None,
        })
    }

    async fn temporary_png_focal_preflight(
        &self,
        project_manager: &ProjectManager,
        paths: &ProjectPaths,
        state: &mut PipelineStateFile,
    ) -> Result<Option<crate::video::temporary_png_focal_workaround::FocalOverride>> {
        use crate::video::temporary_png_focal_workaround::{detect, Request};
        let focal = detect(
            &self.engines.colmap,
            &self.process_manager,
            Request {
                images: &paths.colmap_frames,
                image_list: None,
                ordinary_images: true,
                camera_model: "SIMPLE_RADIAL",
                single_camera: true,
                explicit_params: None,
                existing_camera_id: None,
                calibrated: false,
            },
        )
        .await;
        let message = if let Some(focal) = &focal {
            let invalidated = apply_temporary_png_focal_checkpoint(
                project_manager,
                paths,
                state,
                &focal.checkpoint,
            )
            .await?;
            format!("[TemporaryPngFocal] enabled=true image={} focalLength={} xResolution={} unit={} cameraParams={} cacheInvalidated={invalidated}",
                focal.checkpoint.image_name, focal.raw_focal_length, focal.raw_x_resolution,
                focal.unit, focal.camera_params())
        } else {
            "[TemporaryPngFocal] enabled=false (input/runtime not confirmed applicable)".to_owned()
        };
        self.events.send(
            // A preflight log must not advance progress before a lifecycle stage.
            PipelineStage::Created,
            Some(PipelineComponent::System),
            EventKind::Log,
            EventLevel::Info,
            None,
            true,
            message,
            None,
            None,
            None,
        );
        Ok(focal)
    }

    async fn run_project(
        &self,
        project_manager: &ProjectManager,
        paths: &ProjectPaths,
        metadata: &mut ProjectMetadata,
        mut state: PipelineStateFile,
        acceleration: &crate::engines::AccelerationStatus,
    ) -> Result<PipelineResult> {
        let quality = metadata.quality;
        let mut panorama_video = None;
        if metadata.input_type == ProjectInputType::Video
            && state.video_projection != crate::video::panorama::VideoProjection::Perspective
        {
            let video = probe_video(
                &self.engines.ffprobe,
                &metadata.source_path,
                Some(paths.logs.join("ffprobe.log")),
                &self.process_manager,
            )
            .await?;
            let detection = crate::video::panorama::detect(
                &self.engines.ffprobe,
                &self.engines.ffmpeg,
                &metadata.source_path,
                &video,
                &self.process_manager,
                false,
            )
            .await?;
            if crate::video::panorama::resolve(state.video_projection, &detection)? {
                let input = metadata.source_path.clone();
                let info = video.clone();
                let cancellation = self.process_manager.child_token();
                let config = crate::presets::spawn_pipeline_blocking(move || {
                    crate::video::panorama::PanoramaCheckpoint::new_cancellable(
                        &input,
                        quality,
                        &info,
                        &cancellation,
                    )
                })
                .await
                .map_err(|e| SplatError::Process(e.to_string()))??;
                if state
                    .panorama
                    .as_ref()
                    .is_some_and(|previous| previous != &config)
                {
                    let sfm_changed = state.panorama.as_ref().is_some_and(|previous| {
                        previous.source_sha256 != config.source_sha256
                            || previous.sfm_width != config.sfm_width
                            || previous.sfm_height != config.sfm_height
                    });
                    if sfm_changed {
                        state.frames = None;
                        state.features_complete = false;
                        state.matching_complete = false;
                        state.reconstruction_complete = false;
                        state.bridge_backfill = Default::default();
                    }
                    state.brush_complete = false;
                    state.brush_training = Default::default();
                }
                state.panorama = Some(config);
                state.video = Some(video.clone());
                panorama_video = Some(video);
                project_manager.write_state(&paths.state, &state).await?;
            } else if state.panorama.is_some() {
                return Err(SplatError::Process(
                    "源视频投影已发生变化，不能复用球面重建；请将修改后的素材创建为新任务".into(),
                ));
            }
        }
        if metadata.input_type == ProjectInputType::Images
            && state.video_projection != crate::video::panorama::VideoProjection::Perspective
        {
            let source = metadata.source_path.clone();
            let cancellation = self.process_manager.child_token();
            let requested = state.video_projection;
            let config = crate::presets::spawn_pipeline_blocking(move || {
                let scan = crate::video::image_projection::analyze(&source)?;
                if requested == crate::video::panorama::VideoProjection::Auto && !scan.detection.confirmed_erp {
                    if scan.detection.requires_choice {
                        return Err(SplatError::InvalidVideo("无法判断图片类型。请选择普通图片或全景图片。当前不支持双鱼眼、双目、VR180、裁切全景和厂商私有原片。".into()));
                    }
                    return Ok(None);
                }
                crate::video::image_projection::checkpoint(&source, quality, &cancellation).map(Some)
            }).await.map_err(|e| SplatError::Process(e.to_string()))??;
            if let Some(config) = config {
                if state.panorama.as_ref() != Some(&config) {
                    // A changed source, orientation or projection must never reuse old features.
                    state.frames = None;
                    state.features_complete = false;
                    state.matching_complete = false;
                    state.reconstruction_complete = false;
                    state.bridge_backfill = Default::default();
                    state.brush_complete = false;
                    state.brush_training = Default::default();
                }
                state.panorama = Some(config);
                state.video_projection = crate::video::panorama::VideoProjection::Equirectangular;
                project_manager.write_state(&paths.state, &state).await?;
            }
        }
        self.events.panorama_images.store(
            state
                .panorama
                .as_ref()
                .is_some_and(|p| p.image_sources.is_some()),
            Ordering::Relaxed,
        );
        self.events
            .panorama_phase
            .store(u64::from(state.panorama.is_some()), Ordering::Relaxed);
        if state.input_type != metadata.input_type {
            return Err(SplatError::Process(
                "项目输入类型与检查点不一致，无法安全继续".into(),
            ));
        }
        archive_incompatible_colmap_input_cache(paths, &state).await?;
        // Temporary #73 preflight MUST precede checkpoint/database reuse.
        let ordinary_images = metadata.input_type == ProjectInputType::Images
            && metadata.imported.is_none()
            && metadata.reshoot.is_none()
            && state.panorama.is_none();
        let mut focal_checked = false;
        let mut focal_override = None;
        if ordinary_images
            && state.frames.as_ref().is_some_and(|frames| {
                frames.colmap_input_version == Some(COLMAP_INPUT_VERSION)
                    && frames.extracted_frames.is_some_and(|count| count > 0)
            })
        {
            focal_override = self
                .temporary_png_focal_preflight(project_manager, paths, &mut state)
                .await?;
            focal_checked = true;
        }
        recover_interrupted_publish(paths, &state).await?;
        normalize_checkpoints(paths, &mut state).await?;
        project_manager.write_state(&paths.state, &state).await?;
        let uses_current_resolution_policy = state.planner_enabled
            && state.resolution_policy_version == Some(PLANNER_RESOLUTION_POLICY_VERSION);
        let mut preprobed_video = None;
        if state.panorama.is_none()
            && uses_current_resolution_policy
            && state.resolution_plan.is_none()
            && metadata.input_type == ProjectInputType::Video
        {
            let video = self
                .media
                .probe_video(
                    &self.media_context(Some(paths.logs.join("ffprobe.log")), None),
                    &metadata.source_path,
                )
                .await?;
            let (source_width, source_height) = oriented_video_dimensions(&video);
            let resolution = resolve_planner_resolution_plan(
                quality,
                acceleration.usable_gpu_total_memory_mb(),
                source_width,
                source_height,
                true,
            );
            state.video = Some(video.clone());
            state.resolution_plan = Some(resolution);
            project_manager.write_state(&paths.state, &state).await?;
            preprobed_video = Some(video);
        }
        let target_dimensions = state
            .panorama
            .as_ref()
            .map(|p| (p.sfm_width, p.sfm_height))
            .or_else(|| {
                state
                    .resolution_plan
                    .map(|plan| (plan.working_width, plan.working_height))
            });
        let mut prepared =
            if let Some(prepared) = prepared_frames_from_checkpoint(paths, &state).await? {
                self.events.stage(
                    PipelineStage::ExtractingFrames,
                    1.0,
                    format!("已复用 {} 帧检查点", prepared.extracted_frames),
                );
                prepared
            } else {
                focal_checked = false;
                focal_override = None;
                reset_directory(&paths.frames).await?;
                reset_directory(&paths.colmap_frames).await?;
                reset_directory(&paths.masks).await?;
                reset_directory(&paths.colmap).await?;
                reset_directory(&paths.brush).await?;
                let prepared = match metadata.input_type {
                    ProjectInputType::Video => {
                        self.prepare_frames(
                            &metadata.source_path,
                            quality,
                            &paths.frames,
                            &paths.colmap_frames,
                            &paths.masks,
                            Some(&paths.logs),
                            state.planner_enabled,
                            panorama_video
                                .or(preprobed_video)
                                .or_else(|| state.video.clone()),
                            target_dimensions,
                            state.panorama.is_some(),
                        )
                        .await?
                    }
                    ProjectInputType::Images => {
                        if let Some(config) = &state.panorama {
                            self.prepare_panorama_images(
                                &metadata.source_path,
                                quality,
                                config,
                                &paths.frames,
                                &paths.colmap_frames,
                                &paths.masks,
                            )
                            .await?
                        } else {
                            self.prepare_images(
                                &metadata.source_path,
                                quality,
                                &paths.frames,
                                &paths.colmap_frames,
                                &paths.masks,
                                state.image_sequence.is_some(),
                            )
                            .await?
                        }
                    }
                };
                state.input_type = prepared.input_type;
                state.video = prepared.video.clone();
                state.image_sequence = prepared.image_sequence.clone();
                if state.panorama.is_none()
                    && uses_current_resolution_policy
                    && state.resolution_plan.is_none()
                {
                    state.resolution_plan = Some(resolve_planner_resolution_plan(
                        quality,
                        acceleration.usable_gpu_total_memory_mb(),
                        prepared.working_width,
                        prepared.working_height,
                        false,
                    ));
                }
                let mut frames = FrameState::from(&prepared.plan);
                frames.extracted_frames = Some(prepared.extracted_frames);
                frames.initial_extracted_frames = prepared.extracted_frames;
                frames.image_format = Some(prepared.image_format.clone());
                frames.mask_count = Some(prepared.mask_count);
                frames.has_alpha = prepared.has_alpha;
                frames.colmap_input_version = Some(COLMAP_INPUT_VERSION);
                state.frames = Some(frames);
                state.features_complete = false;
                state.matching_complete = false;
                state.reconstruction_complete = false;
                state.brush_complete = false;
                state.stage = PipelineStage::ExtractingFrames;
                project_manager.write_state(&paths.state, &state).await?;
                prepared
            };
        if ordinary_images && !focal_checked {
            focal_override = self
                .temporary_png_focal_preflight(project_manager, paths, &mut state)
                .await?;
        }
        let camera_params = focal_override.as_ref().map(|focal| focal.camera_params());
        let source_duration_seconds = prepared.video.as_ref().map(|video| video.duration);
        self.record_frame_metrics(&state, prepared.extracted_frames, Some(&prepared.plan));

        let database = paths.colmap.join("database.db");
        let sparse = paths.colmap.join("sparse");
        let colmap_log = paths.logs.join("colmap.log");
        let preset = quality.preset();
        let sfm_max_image_size =
            state
                .panorama
                .as_ref()
                .map(|p| p.sfm_width)
                .unwrap_or_else(|| {
                    state
                        .resolution_plan
                        .map(|plan| plan.sfm_max_image_size)
                        .unwrap_or(if state.planner_enabled {
                            preset.planned_sfm_max_image_size
                        } else {
                            preset.sfm_max_image_size
                        })
                });
        if let Some(config) = &state.panorama {
            tokio::fs::create_dir_all(&paths.colmap).await?;
            tokio::fs::write(
                paths.colmap.join("panorama-camera.json"),
                serde_json::to_vec(config)?,
            )
            .await?;
        }
        if let Some(resolution) = state.resolution_plan {
            self.events.send(
                PipelineStage::ExtractingFeatures,
                Some(PipelineComponent::System),
                EventKind::Log,
                EventLevel::Info,
                None,
                true,
                format!(
                    "[ResolutionPlan] version={} source={}x{} working={}x{} sfmMax={} brushMax={} profile={}",
                    resolution.policy_version,
                    prepared
                        .video
                        .as_ref()
                        .map(|video| oriented_video_dimensions(video).0)
                        .or_else(|| prepared.image_sequence.as_ref().map(|images| images.width))
                        .unwrap_or(prepared.working_width),
                    prepared
                        .video
                        .as_ref()
                        .map(|video| oriented_video_dimensions(video).1)
                        .or_else(|| prepared.image_sequence.as_ref().map(|images| images.height))
                        .unwrap_or(prepared.working_height),
                    prepared.working_width,
                    prepared.working_height,
                    resolution.sfm_max_image_size,
                    resolution.brush_initial_max_resolution,
                    resolution.brush_initial_profile.label(),
                ),
                None,
                None,
                None,
            );
        }
        // COLMAP's bundled bitmap loader cannot reliably open non-ASCII absolute
        // paths on Windows. The process working directory is work/colmap, so this
        // ASCII-only relative path preserves Unicode/UNC project roots without
        // moving any project data outside the project directory.
        let colmap_images = Path::new("../colmap_frames");
        let colmap_masks = prepared.has_alpha.then_some(Path::new("../masks"));

        let backend_label = if acceleration.use_gpu() { "GPU" } else { "CPU" };
        let gpu_index = acceleration.gpu_index();
        if state.features_complete {
            self.events.stage(
                PipelineStage::ExtractingFeatures,
                1.0,
                "已复用特征提取检查点",
            );
        } else {
            reset_directory(&paths.colmap).await?;
            if let Some(config) = &state.panorama {
                tokio::fs::write(
                    paths.colmap.join("panorama-camera.json"),
                    serde_json::to_vec(config)?,
                )
                .await?;
            }
            self.events.stage(
                PipelineStage::ExtractingFeatures,
                0.0,
                format!("正在使用 {backend_label} 提取特征"),
            );
            let observer = Some(self.process_observer(
                PipelineStage::ExtractingFeatures,
                PipelineComponent::Reconstruction,
                Some(prepared.extracted_frames),
                ObserverMode::BracketProgress,
            ));
            if state.planner_enabled {
                self.backend
                .extract_features_quality_v2(
                    &self.job_context(
                        colmap_log.clone(),
                        observer,
                        gpu_index,
                    ),
                    QualityFeatureExtraction {
                        base: FeatureExtraction {
                            database: database.clone(),
                            images: colmap_images.to_path_buf(),
                            masks: colmap_masks.map(|path| path.to_path_buf()),
                            image_list: None,
                            existing_camera_id: None,
                            max_image_size: Some(sfm_max_image_size),
                            max_features: Some(preset.planned_sfm_max_features),
                        },
                        planned_max_features: preset.planned_sfm_max_features,
                    },
                )
                .await?;
            } else {
                self.backend
                .extract_features(
                    &self.job_context(
                        colmap_log.clone(),
                        observer,
                        gpu_index,
                    ),
                    FeatureExtraction {
                        database: database.clone(),
                        images: colmap_images.to_path_buf(),
                        masks: colmap_masks.map(|path| path.to_path_buf()),
                        image_list: None,
                        existing_camera_id: None,
                        max_image_size: Some(sfm_max_image_size),
                        max_features: Some(preset.sfm_max_features),
                    },
                )
                .await?;
            }
            state.stage = PipelineStage::ExtractingFeatures;
            state.features_complete = true;
            project_manager.write_state(&paths.state, &state).await?;
            self.events.stage(
                PipelineStage::ExtractingFeatures,
                1.0,
                format!("{backend_label} 特征提取完成"),
            );
        }

        // Resumes deliberately consult the persisted strategy, never the new-run builder.
        let (sequential_matching, loop_detection) = image_matching_policy(
            prepared.input_type,
            state.image_matching_strategy,
            quality,
            state.planner_enabled,
        );
        if state.matching_complete {
            self.events.stage(
                PipelineStage::Matching,
                1.0,
                if !sequential_matching {
                    "已复用穷举匹配检查点"
                } else {
                    "已复用顺序匹配检查点"
                },
            );
        } else {
            self.events.stage(
                PipelineStage::Matching,
                0.0,
                format!(
                    "COLMAP 正在进行 {backend_label} {}",
                    if !sequential_matching {
                        "穷举匹配"
                    } else if loop_detection {
                        "顺序匹配（已开启闭环检测）"
                    } else {
                        "顺序匹配"
                    }
                ),
            );
            let observer = Some(self.process_observer(
                PipelineStage::Matching,
                PipelineComponent::Reconstruction,
                Some(prepared.extracted_frames),
                ObserverMode::Matching,
            ));
            if !sequential_matching {
                colmap::match_exhaustive(
                    &self.engines.colmap,
                    &database,
                    colmap_log.clone(),
                    &self.process_manager,
                    observer,
                    gpu_index,
                )
                .await?;
            } else {
                self.backend
                .match_sequential(
                    &self.job_context(
                        colmap_log.clone(),
                        observer,
                        gpu_index,
                    ),
                    Matching {
                        database: database.clone(),
                        loop_detection: loop_detection,
                    },
                )
                .await?;
            }
            state.stage = PipelineStage::Matching;
            state.matching_complete = true;
            project_manager.write_state(&paths.state, &state).await?;
            self.events.stage(
                PipelineStage::Matching,
                1.0,
                if !sequential_matching {
                    "穷举匹配完成"
                } else if loop_detection {
                    "顺序匹配（已开启闭环检测）完成"
                } else {
                    "顺序匹配完成"
                },
            );
        }

        let allow_two_view_tracks = should_allow_two_view_tracks(
            state.planner_enabled,
            preset.sfm_allow_two_view_tracks,
            prepared.has_alpha,
        );
        if state.reconstruction_complete {
            self.events
                .stage(PipelineStage::Reconstructing, 1.0, "已复用相机重建检查点");
        } else {
            reset_directory(&sparse).await?;
            self.events.stage(
                PipelineStage::Reconstructing,
                0.0,
                if allow_two_view_tracks {
                    "正在增量重建相机轨迹（High：允许两视图三角化）"
                } else {
                    "正在增量重建相机轨迹"
                },
            );
            self.backend
                .map_sparse(
                    &self.job_context(
                        colmap_log.clone(),
                        Some(self.process_observer(
                    PipelineStage::Reconstructing,
                    PipelineComponent::Reconstruction,
                    Some(prepared.extracted_frames),
                    ObserverMode::Mapper,
                )),
                        gpu_index,
                    ),
                    SparseMapping {
                        database: database.clone(),
                        images: colmap_images.to_path_buf(),
                        input_model: PathBuf::new(),
                        output: sparse.clone(),
                        allow_two_view_tracks: allow_two_view_tracks,
                        automatic_optimization: state.planner_enabled,
                    },
                )
                .await?;
            state.stage = PipelineStage::Reconstructing;
            state.reconstruction_complete = true;
            project_manager.write_state(&paths.state, &state).await?;
            self.events
                .stage(PipelineStage::Reconstructing, 1.0, "增量重建完成");
        }

        self.events.stage(
            PipelineStage::ValidatingReconstruction,
            0.0,
            "正在核验注册率和三维点",
        );
        let initial_input_images = state
            .frames
            .as_ref()
            .map(|frames| frames.initial_extracted_frames)
            .filter(|count| *count > 0)
            .unwrap_or(prepared.extracted_frames);
        let (initial_model, initial_report) =
            best_sparse_model_with_input_count(&sparse, initial_input_images).await?;
        self.record_reconstruction_metrics(
            &initial_report,
            acceleration,
            allow_two_view_tracks,
            false,
        );
        let initial_report_for_metrics = initial_report.clone();
        let (model, report) = self
            .maybe_run_bridge_backfill(
                project_manager,
                paths,
                metadata,
                &mut state,
                &mut prepared,
                &database,
                colmap_images,
                colmap_masks,
                &colmap_log,
                gpu_index,
                initial_model,
                initial_report,
            )
            .await?;
        if let Some(tracker) = &self.effectiveness {
            let bridge_applicable =
                state.planner_enabled && prepared.input_type == ProjectInputType::Video;
            let plan = state.bridge_backfill.plan.as_ref();
            tracker.record_bridge(PlannerBridgeMetrics {
                status: if bridge_applicable {
                    PlannerBridgeStatus::from(state.bridge_backfill.status)
                } else {
                    PlannerBridgeStatus::NotApplicable
                },
                trigger_ratio: bridge_trigger_ratio(),
                available_budget: plan.map(|value| value.available_budget).unwrap_or(0),
                requested_frames: plan
                    .map(|value| value.selected_frame_indices.len() as u64)
                    .unwrap_or(0),
                added_frames: if matches!(
                    state.bridge_backfill.status,
                    BridgeBackfillStatus::Completed | BridgeBackfillStatus::FailedRolledBack
                ) {
                    plan.map(|value| value.selected_frame_indices.len() as u64)
                        .unwrap_or(0)
                } else {
                    0
                },
                internal_bridge_count: plan.map(|value| value.internal_bridge_count).unwrap_or(0),
                edge_extension_count: plan.map(|value| value.edge_extension_count).unwrap_or(0),
                duration_ms: state.bridge_backfill.duration_ms,
                initial_input_images: initial_report_for_metrics.input_images,
                initial_registered_images: initial_report_for_metrics.registered_images,
                initial_points_3d: initial_report_for_metrics.points_3d,
                final_input_images: report.input_images,
                final_registered_images: report.registered_images,
                final_points_3d: report.points_3d,
                adopted: state.bridge_backfill.status == BridgeBackfillStatus::Completed,
            });
        }
        let warning = (report.quality == ReconstructionQuality::Warning).then(|| {
            format!(
                "注册率 {:.1}%：低于 {:.1}%，将继续训练，但结果质量可能受影响",
                report.registered_ratio * 100.0,
                crate::reconstruction::good_registered_ratio() * 100.0
            )
        });
        self.events.stage(
            PipelineStage::ValidatingReconstruction,
            1.0,
            format!(
                "注册 {}/{} 张 · 三维点 {}",
                report.registered_images, report.input_images, report.points_3d
            ),
        );

        let candidate = if state.brush_complete {
            self.record_brush_metrics(&state);
            self.events.stage(
                PipelineStage::TrainingSplats,
                1.0,
                "已复用 Brush 训练检查点",
            );
            brush_candidate(&paths.brush)
                .ok_or_else(|| SplatError::Process("Brush 检查点文件缺失，无法继续发布".into()))?
        } else {
            // Freeze the recipe before cache preparation, including on cancel.
            if state.brush_training.resolved.is_none() {
                if let Some(config) = &state.panorama {
                    let resolved = resolve_panorama_training_preset(
                        quality,
                        state.planner_enabled,
                        brush::BrushGpuLaunchPolicy::from_acceleration(acceleration)
                            .known_training_memory_mb(acceleration),
                        config.face_size,
                        report.points_3d,
                    )?;
                    state
                        .brush_training
                        .initial_resolved
                        .get_or_insert(resolved);
                    state.brush_training.resolved = Some(resolved);
                    project_manager.write_state(&paths.state, &state).await?;
                }
            }
            reset_directory(&paths.brush).await?;
            let dataset = if let Some(config) = &state.panorama {
                self.events.panorama_phase.store(2, Ordering::Relaxed);
                self.events.send(
                    PipelineStage::TrainingSplats,
                    Some(PipelineComponent::System),
                    EventKind::Stage,
                    EventLevel::Info,
                    None,
                    true,
                    "正在检查可复用的全景训练缓存",
                    None,
                    None,
                    None,
                );
                let config = config.clone();
                let frames = paths.frames.clone();
                let masks = paths.masks.clone();
                let model = model.clone();
                let root = paths.work.join("panorama-training");
                let cancel = self.process_manager.child_token();
                let events = self.events.clone();
                crate::presets::spawn_pipeline_blocking(move || {
                    crate::video::panorama::prepare_training_cache(
                        &frames,
                        &masks,
                        &model,
                        &root,
                        &config,
                        &cancel,
                        |update| {
                            let (current, total, reused) = match update {
                                crate::video::panorama::TrainingCacheUpdate::Faces {
                                    completed,
                                    total,
                                    reused,
                                } => (completed, total, reused),
                                crate::video::panorama::TrainingCacheUpdate::Validating => {
                                    events.panorama_phase.store(3, Ordering::Relaxed);
                                    events.send(
                                        PipelineStage::TrainingSplats,
                                        Some(PipelineComponent::System),
                                        EventKind::Stage,
                                        EventLevel::Info,
                                        Some(0.95),
                                        true,
                                        "正在校验训练图片和相机数据",
                                        None,
                                        None,
                                        None,
                                    );
                                    return;
                                }
                            };
                            if reused && config.image_sources.is_some() {
                                events.send(PipelineStage::TrainingSplats,Some(PipelineComponent::System),EventKind::Log,EventLevel::Info,None,false,"已校验并复用训练缓存",None,None,None);
                            }
                            events.send(
                                PipelineStage::TrainingSplats,
                                Some(PipelineComponent::System),
                                EventKind::Stage,
                                EventLevel::Info,
                                Some(if current < total {
                                    (current as f32 / total.max(1) as f32) * 0.95
                                } else {
                                    0.95
                                }),
                                false,
                                if config.image_sources.is_some() {
                                    format!("正在准备训练图片：全景图片 {}/{}，六面图片 {current}/{total}", current / 6, total / 6)
                                } else { format!(
                                    "{}：全景帧 {}/{}，六面图片 {current}/{total}",
                                    if reused {
                                        "已校验并复用训练缓存"
                                    } else {
                                        "正在准备训练图片"
                                    },
                                    current / 6,
                                    total / 6
                                ) },
                                Some(current),
                                Some(total),
                                Some(if config.image_sources.is_some() { "panoramaImageFaces" } else { "panoramaFaces" }),
                            )
                        },
                    )
                })
                .await
                .map_err(|e| SplatError::Process(e.to_string()))??
            } else {
                prepare_brush_dataset(&paths.brush, &paths.frames, &model).await?
            };
            if state.panorama.is_some() {
                self.events.panorama_phase.store(3, Ordering::Relaxed);
                self.events.stage(
                    PipelineStage::TrainingSplats,
                    1.0,
                    "训练图片和相机数据已校验",
                );
                self.events.panorama_phase.store(1, Ordering::Relaxed);
            }
            let source_long_edge = prepared
                .video
                .as_ref()
                .map(|video| video.width.max(video.height))
                .or_else(|| {
                    prepared
                        .image_sequence
                        .as_ref()
                        .map(|images| images.width.max(images.height))
                })
                .unwrap_or(1);
            let detected_total_memory_mb = if state.panorama.is_some() {
                brush::BrushGpuLaunchPolicy::from_acceleration(acceleration)
                    .known_training_memory_mb(acceleration)
            } else {
                acceleration.usable_gpu_total_memory_mb()
            };
            let mut resolved_brush = if let Some(saved) = state.brush_training.resolved {
                saved
            } else if let Some(config) = &state.panorama {
                resolve_panorama_training_preset(
                    quality,
                    state.planner_enabled,
                    detected_total_memory_mb,
                    config.face_size,
                    report.points_3d,
                )?
            } else {
                state.resolution_plan.map_or_else(
                    || {
                        resolve_brush_training_preset(
                            quality,
                            state.planner_enabled,
                            detected_total_memory_mb,
                            source_long_edge,
                            report.points_3d,
                        )
                    },
                    |resolution| {
                        resolve_brush_training_preset_for_plan(
                            quality,
                            detected_total_memory_mb,
                            source_long_edge,
                            report.points_3d,
                            &resolution,
                        )
                    },
                )
            };
            if let Some(config) = &state.panorama {
                resolved_brush.preset.max_resolution =
                    resolved_brush.preset.max_resolution.min(config.face_size);
            }
            if !state.brush_training.oom_retry_used {
                state
                    .brush_training
                    .initial_resolved
                    .get_or_insert(resolved_brush);
            }
            state.brush_training.resolved = Some(resolved_brush);
            self.record_brush_metrics(&state);
            project_manager.write_state(&paths.state, &state).await?;
            let runtime_samples = catalog::runtime_samples().await;
            let estimated_brush_duration_ms = match (&prepared.video, &prepared.image_sequence) {
                (Some(video), _) => estimate_calibrated_brush_stage_ms(
                    video,
                    &prepared.plan,
                    quality,
                    &runtime_samples,
                    Some(&resolved_brush),
                    state.resolution_policy_version,
                    prepared.working_width.max(prepared.working_height),
                ),
                (_, Some(images)) => estimate_calibrated_brush_stage_ms_for_images(
                    images.image_count,
                    images.width.max(images.height),
                    &prepared.plan,
                    quality,
                    &runtime_samples,
                    Some(&resolved_brush),
                    state.resolution_policy_version,
                    prepared.working_width.max(prepared.working_height),
                ),
                _ => return Err(SplatError::Process("项目输入信息不完整".into())),
            };
            self.events.send(
                PipelineStage::TrainingSplats,
                Some(PipelineComponent::Training),
                EventKind::Stage,
                EventLevel::Info,
                None,
                true,
                format!(
                    "Brush 训练开始（使用可用图形后端）· {} iterations · 最大分辨率 {} · 预计约 {}",
                    resolved_brush.preset.total_steps,
                    resolved_brush.preset.max_resolution,
                    format_duration(estimated_brush_duration_ms)
                ),
                Some(0),
                Some(resolved_brush.preset.total_steps as u64),
                Some("iterations"),
            );
            let candidate = self
                .train_brush_with_oom_retry(
                    project_manager,
                    paths,
                    &mut state,
                    quality,
                    acceleration,
                    &dataset,
                    report.points_3d,
                    resolved_brush,
                    estimated_brush_duration_ms,
                )
                .await?;
            state.stage = PipelineStage::TrainingSplats;
            state.brush_complete = true;
            project_manager.write_state(&paths.state, &state).await?;
            self.events
                .stage(PipelineStage::TrainingSplats, 1.0, "Brush 训练完成");
            candidate
        };

        self.events
            .stage(PipelineStage::Exporting, 0.0, "正在校验并发布 final.ply");
        let ply = inspect_gaussian_ply(&candidate)?;
        let final_ply = paths.project.join("final.ply");
        atomic_replace_file(&candidate, &final_ply).await?;
        state.stage = PipelineStage::Completed;
        project_manager.write_state(&paths.state, &state).await?;

        let completed_at = Utc::now();
        let duration_ms = metadata.duration_ms.unwrap_or(0).saturating_add(
            metadata
                .started_at
                .map(|started| (completed_at - started).num_milliseconds().max(0) as u64)
                .unwrap_or(0),
        );
        metadata.status = ProjectStatus::Completed;
        metadata.completed_at = Some(completed_at);
        metadata.duration_ms = Some(duration_ms);
        metadata.output = Some(ProjectOutput {
            final_ply: final_ply.clone(),
            file_size: ply.file_size,
            splat_count: ply.splat_count,
            input_images: report.input_images,
            registered_images: report.registered_images,
            registered_ratio: report.registered_ratio,
            points_3d: report.points_3d,
        });
        project_manager
            .write_metadata(&paths.metadata, metadata)
            .await?;

        if let Some(output) = &metadata.output {
            self.record_final_metrics(output);
        }

        self.events.stage(
            PipelineStage::Exporting,
            1.0,
            format!("已发布 {} 个 Splat", ply.splat_count),
        );
        self.events
            .stage(PipelineStage::Completed, 1.0, "全部处理完成");
        Ok(PipelineResult {
            project_id: paths.id.to_string(),
            project_path: paths.project.clone(),
            final_ply,
            file_size: ply.file_size,
            splat_count: ply.splat_count,
            input_images: report.input_images,
            registered_images: report.registered_images,
            registered_ratio: report.registered_ratio,
            points_3d: report.points_3d,
            duration_ms,
            completed_at,
            warning,
            logs_directory: paths.logs.clone(),
            source_duration_seconds,
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn train_brush_with_oom_retry(
        &self,
        project_manager: &ProjectManager,
        paths: &ProjectPaths,
        state: &mut PipelineStateFile,
        quality: Quality,
        acceleration: &crate::engines::AccelerationStatus,
        dataset: &Path,
        initial_sfm_points: u64,
        resolved: ResolvedBrushTrainingPreset,
        _estimated_duration_ms: u64,
    ) -> Result<PathBuf> {
        self.log_brush_profile("BrushProfile", &resolved, EventLevel::Info);
        let gpu_launch_policy = brush::BrushGpuLaunchPolicy::from_acceleration(acceleration);
        self.events.send(
            PipelineStage::TrainingSplats,
            Some(PipelineComponent::Training),
            EventKind::Log,
            EventLevel::Info,
            None,
            true,
            format!("[BrushGpu] {}", gpu_launch_policy.log_summary()),
            None,
            None,
            None,
        );
        let first_log = if state.brush_training.oom_retry_used {
            paths.logs.join("brush-retry.log")
        } else {
            paths.logs.join("brush.log")
        };
        let first = brush::train(
            &self.engines.brush,
            dataset,
            &paths.brush,
            brush::BrushTrainingOptions {
                preset: resolved.preset,
                panorama: resolved.panorama.is_some(),
                log_path: first_log,
                gpu_launch_policy: &gpu_launch_policy,
            },
            &self.process_manager,
            Some(self.process_observer(
                PipelineStage::TrainingSplats,
                PipelineComponent::Training,
                Some(resolved.preset.total_steps as u64),
                ObserverMode::Brush,
            )),
        )
        .await;

        match first {
            Ok(candidate) => Ok(candidate),
            Err(SplatError::BrushOutOfMemory(detail))
                if (quality == Quality::High || resolved.panorama.is_some())
                    && state.planner_enabled
                    && !state.brush_training.oom_retry_used =>
            {
                let Some(downgraded) = brush_oom_fallback(
                    quality,
                    state.planner_enabled,
                    state.brush_training.oom_retry_used,
                    resolved,
                    initial_sfm_points,
                ) else {
                    return Err(SplatError::BrushOutOfMemory(detail));
                };
                state.brush_training.oom_retry_used = true;
                state.brush_training.resolved = Some(downgraded);
                if let Some(tracker) = &self.effectiveness {
                    tracker.record_brush(downgraded, true);
                }
                project_manager.write_state(&paths.state, state).await?;
                self.events.send(
                    PipelineStage::TrainingSplats,
                    Some(PipelineComponent::Training),
                    EventKind::Log,
                    EventLevel::Warning,
                    None,
                    true,
                    format!(
                        "[BrushRetry] reason=OutOfMemory from={} to={}，复用现有 SfM 和训练数据重试一次",
                        resolved.profile.label(),
                        downgraded.profile.label()
                    ),
                    None,
                    None,
                    None,
                );
                self.log_brush_profile("BrushRetryProfile", &downgraded, EventLevel::Warning);
                brush::train(
                    &self.engines.brush,
                    dataset,
                    &paths.brush,
                    brush::BrushTrainingOptions {
                        preset: downgraded.preset,
                        panorama: downgraded.panorama.is_some(),
                        log_path: paths.logs.join("brush-retry.log"),
                        gpu_launch_policy: &gpu_launch_policy,
                    },
                    &self.process_manager,
                    Some(self.process_observer(
                        PipelineStage::TrainingSplats,
                        PipelineComponent::Training,
                        Some(downgraded.preset.total_steps as u64),
                        ObserverMode::Brush,
                    )),
                )
                .await
            }
            Err(error) => Err(error),
        }
    }

    fn log_brush_profile(
        &self,
        label: &str,
        resolved: &ResolvedBrushTrainingPreset,
        level: EventLevel,
    ) {
        let densification = resolved.preset.densification;
        if let Some(snapshot) = resolved.panorama {
            self.events.send(
                PipelineStage::TrainingSplats, Some(PipelineComponent::Training),
                EventKind::Log, level, None, true,
                format!("[PanoramaBrushRecipe] version={} quality={} automaticOptimization={} memoryMaxSplats={:?} initialPoints={} initialPointsOverride={} growthStart=0 shDegree=3",
                    snapshot.recipe_version, snapshot.quality, snapshot.automatic_optimization,
                    snapshot.memory_max_splats, snapshot.initial_points, snapshot.initial_points_override),
                None, None, None,
            );
        }
        self.events.send(
            PipelineStage::TrainingSplats,
            Some(PipelineComponent::Training),
            EventKind::Log,
            level,
            None,
            true,
            format!(
                "[{label}] profile={} vramMiB={} resolution={} steps={} threshold={} fraction={} growthStop={} configuredMaxSplats={} effectiveMaxSplats={}",
                resolved.profile.label(),
                resolved
                    .detected_total_memory_mb
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "unknown".into()),
                resolved.preset.max_resolution,
                resolved.preset.total_steps,
                densification
                    .map(|value| value.growth_grad_threshold.to_string())
                    .unwrap_or_else(|| "default".into()),
                densification
                    .map(|value| value.growth_select_fraction.to_string())
                    .unwrap_or_else(|| "default".into()),
                densification
                    .map(|value| value.growth_stop_iter.to_string())
                    .unwrap_or_else(|| "default".into()),
                resolved
                    .configured_max_splats
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "default".into()),
                resolved
                    .preset
                    .max_splats
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "default".into()),
            ),
            None,
            None,
            None,
        );
    }

    #[allow(clippy::too_many_arguments)]
    async fn maybe_run_bridge_backfill(
        &self,
        project_manager: &ProjectManager,
        paths: &ProjectPaths,
        metadata: &ProjectMetadata,
        state: &mut PipelineStateFile,
        prepared: &mut PreparedFrames,
        database: &Path,
        colmap_images: &Path,
        colmap_masks: Option<&Path>,
        colmap_log: &Path,
        gpu_index: Option<u32>,
        initial_model: PathBuf,
        initial_report: ReconstructionReport,
    ) -> Result<(PathBuf, ReconstructionReport)> {
        if !state.planner_enabled || prepared.input_type != ProjectInputType::Video {
            return Ok((initial_model, initial_report));
        }

        match state.bridge_backfill.status {
            BridgeBackfillStatus::NotNeeded
            | BridgeBackfillStatus::NoBudget
            | BridgeBackfillStatus::FailedRolledBack => {
                return Ok((initial_model, initial_report));
            }
            BridgeBackfillStatus::Completed => {
                if let Some(relative) = state.bridge_backfill.selected_model.as_deref() {
                    let model = paths.colmap.join(relative);
                    if let Ok(report) = ReconstructionValidator::validate_with_input_images(
                        prepared.extracted_frames,
                        &model,
                    ) {
                        return Ok((model, report));
                    }
                }
                state.bridge_backfill.status = BridgeBackfillStatus::FailedRolledBack;
                project_manager.write_state(&paths.state, state).await?;
                return Ok((initial_model, initial_report));
            }
            BridgeBackfillStatus::NotEvaluated | BridgeBackfillStatus::Running => {}
        }

        if initial_report.registered_ratio >= bridge_trigger_ratio() {
            state.bridge_backfill.status = BridgeBackfillStatus::NotNeeded;
            project_manager.write_state(&paths.state, state).await?;
            self.events.stage(
                PipelineStage::ValidatingReconstruction,
                1.0,
                format!(
                    "初始注册 {}/{} 张（{:.1}%）达到 {:.1}% 阈值，无需 Bridge Backfill",
                    initial_report.registered_images,
                    initial_report.input_images,
                    initial_report.registered_ratio * 100.0,
                    bridge_trigger_ratio() * 100.0
                ),
            );
            return Ok((initial_model, initial_report));
        }

        let bridge_plan = if state.bridge_backfill.status == BridgeBackfillStatus::Running {
            state.bridge_backfill.plan.clone().unwrap_or_default()
        } else {
            let registered = match read_registered_source_indices(&initial_model) {
                Ok(indices) => indices,
                Err(error) => {
                    state.bridge_backfill.status = BridgeBackfillStatus::FailedRolledBack;
                    project_manager.write_state(&paths.state, state).await?;
                    self.events.send(
                        PipelineStage::ValidatingReconstruction,
                        Some(PipelineComponent::System),
                        EventKind::Log,
                        EventLevel::Warning,
                        Some(1.0),
                        false,
                        format!(
                            "Bridge Backfill 无法读取初始注册时间线，继续使用初始模型：{error}"
                        ),
                        None,
                        None,
                        None,
                    );
                    return Ok((initial_model, initial_report));
                }
            };
            let plan = plan_bridge_backfill(
                &prepared.plan,
                &registered,
                initial_report.registered_images,
            );
            state.bridge_backfill.plan = Some(plan.clone());
            state.bridge_backfill.initial_model = Some(relative_model_path(paths, &initial_model));
            if plan.selected_frame_indices.is_empty() {
                state.bridge_backfill.status = BridgeBackfillStatus::NoBudget;
                project_manager.write_state(&paths.state, state).await?;
                self.events.send(
                    PipelineStage::ValidatingReconstruction,
                    Some(PipelineComponent::System),
                    EventKind::Log,
                    EventLevel::Warning,
                    Some(1.0),
                    false,
                    format!(
                        "初始注册 {}/{} 张（{:.1}%）低于 {:.1}%，但 Bridge Backfill 没有可用补帧预算",
                        initial_report.registered_images,
                        initial_report.input_images,
                        initial_report.registered_ratio * 100.0,
                        bridge_trigger_ratio() * 100.0
                    ),
                    None,
                    None,
                    None,
                );
                return Ok((initial_model, initial_report));
            }
            state.bridge_backfill.status = BridgeBackfillStatus::Running;
            project_manager.write_state(&paths.state, state).await?;
            plan
        };

        let additional = bridge_plan
            .selected_frame_indices
            .iter()
            .filter_map(|index| {
                prepared
                    .plan
                    .candidate_frames
                    .iter()
                    .find(|frame| frame.source_frame_index == *index)
                    .cloned()
            })
            .collect::<Vec<_>>();
        if additional.len() != bridge_plan.selected_frame_indices.len() {
            state.bridge_backfill.status = BridgeBackfillStatus::FailedRolledBack;
            project_manager.write_state(&paths.state, state).await?;
            self.events.send(
                PipelineStage::ValidatingReconstruction,
                Some(PipelineComponent::System),
                EventKind::Log,
                EventLevel::Warning,
                Some(1.0),
                false,
                "Bridge Backfill 检查点与候选池不一致，继续使用初始模型",
                None,
                None,
                None,
            );
            return Ok((initial_model, initial_report));
        }

        if let Some(first) = bridge_plan.selection_trace.first() {
            self.events.send(
                PipelineStage::ValidatingReconstruction,
                Some(PipelineComponent::System),
                EventKind::Log,
                EventLevel::Info,
                Some(0.0),
                false,
                format!(
                    "Bridge Backfill 最长未注册区 {}..{}，选择中点候选帧 {}",
                    first.gap_start_frame_index,
                    first.gap_end_frame_index,
                    first.selected_frame_index
                ),
                Some(1),
                Some(additional.len() as u64),
                Some("frames"),
            );
        }
        self.events.send(
            PipelineStage::ValidatingReconstruction,
            Some(PipelineComponent::System),
            EventKind::Log,
            EventLevel::Warning,
            Some(0.0),
            false,
            format!(
                "Bridge Backfill 已触发：初始注册={}/{}（{:.1}%），剩余预算={}，补帧={}，内部桥接={}，边缘延伸={}",
                bridge_plan.initial_registered_images,
                bridge_plan.initial_input_images,
                bridge_plan.initial_registration_ratio * 100.0,
                bridge_plan.available_budget,
                additional.len(),
                bridge_plan.internal_bridge_count,
                bridge_plan.edge_extension_count
            ),
            Some(additional.len() as u64),
            Some(bridge_plan.available_budget),
            Some("frames"),
        );

        let bridge_started = Instant::now();
        let attempt = self
            .execute_bridge_attempt(
                project_manager,
                paths,
                metadata,
                state,
                prepared,
                database,
                colmap_images,
                colmap_masks,
                colmap_log,
                gpu_index,
                &initial_model,
                &initial_report,
                &additional,
            )
            .await;
        match attempt {
            Ok((model, report)) => {
                let bridge_duration_ms = bridge_started.elapsed().as_millis() as u64;
                state.bridge_backfill.status = BridgeBackfillStatus::Completed;
                state.bridge_backfill.selected_model = Some(relative_model_path(paths, &model));
                state.bridge_backfill.final_registered_images = Some(report.registered_images);
                state.bridge_backfill.final_points_3d = Some(report.points_3d);
                state.bridge_backfill.duration_ms = Some(bridge_duration_ms);
                project_manager.write_state(&paths.state, state).await?;
                self.events.send(
                    PipelineStage::ValidatingReconstruction,
                    Some(PipelineComponent::System),
                    EventKind::Log,
                    EventLevel::Info,
                    Some(1.0),
                    false,
                    format!(
                        "Bridge Backfill 完成：注册 {}（{:.1}%）→ {}（{:.1}%），三维点 {} → {}，耗时 {}；采用增量模型",
                        initial_report.registered_images,
                        initial_report.registered_ratio * 100.0,
                        report.registered_images,
                        report.registered_ratio * 100.0,
                        initial_report.points_3d,
                        report.points_3d,
                        format_duration(bridge_duration_ms)
                    ),
                    Some(report.registered_images),
                    Some(report.input_images),
                    Some("images"),
                );
                Ok((model, report))
            }
            Err(error) => {
                let bridge_duration_ms = bridge_started.elapsed().as_millis() as u64;
                remove_bridge_outputs(
                    &paths.frames,
                    &paths.colmap_frames,
                    &paths.masks,
                    &additional,
                    prepared.has_alpha,
                )
                .await;
                let added = additional
                    .iter()
                    .map(|frame| frame.source_frame_index)
                    .collect::<HashSet<_>>();
                prepared
                    .plan
                    .selected_frames
                    .retain(|frame| !added.contains(&frame.source_frame_index));
                prepared.extracted_frames = bridge_plan.initial_input_images;
                prepared.mask_count = if prepared.has_alpha {
                    bridge_plan.initial_input_images
                } else {
                    0
                };
                prepared.plan.estimated_frames = bridge_plan.initial_input_images;
                if let Some(video) = prepared.video.as_ref() {
                    prepared.plan.retention_ratio =
                        bridge_plan.initial_input_images as f64 / video.total_frames.max(1) as f64;
                    prepared.plan.sampling_fps =
                        bridge_plan.initial_input_images as f64 / video.duration.max(0.001);
                }
                update_frame_checkpoint(state, prepared);
                state.bridge_backfill.status = BridgeBackfillStatus::FailedRolledBack;
                state.bridge_backfill.selected_model = state.bridge_backfill.initial_model.clone();
                state.bridge_backfill.duration_ms = Some(bridge_duration_ms);
                project_manager.write_state(&paths.state, state).await?;
                self.events.send(
                    PipelineStage::ValidatingReconstruction,
                    Some(PipelineComponent::System),
                    EventKind::Log,
                    EventLevel::Warning,
                    Some(1.0),
                    false,
                    format!(
                        "Bridge Backfill 失败并回退可用初始模型（耗时 {}）：{error}",
                        format_duration(bridge_duration_ms)
                    ),
                    Some(initial_report.registered_images),
                    Some(initial_report.input_images),
                    Some("images"),
                );
                Ok((initial_model, initial_report))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_bridge_attempt(
        &self,
        project_manager: &ProjectManager,
        paths: &ProjectPaths,
        metadata: &ProjectMetadata,
        state: &mut PipelineStateFile,
        prepared: &mut PreparedFrames,
        database: &Path,
        colmap_images: &Path,
        colmap_masks: Option<&Path>,
        colmap_log: &Path,
        gpu_index: Option<u32>,
        initial_model: &Path,
        initial_report: &ReconstructionReport,
        additional: &[PlannedFrame],
    ) -> Result<(PathBuf, ReconstructionReport)> {
        self.events.stage(
            PipelineStage::ValidatingReconstruction,
            0.0,
            format!("Bridge Backfill 正在提取 {} 张新增帧", additional.len()),
        );
        let extraction = self
            .media
            .extract_additional_frames(
                &self.media_context(
                    Some(paths.logs.join("ffmpeg-bridge.log")),
                    Some(self.process_observer(
                        PipelineStage::ValidatingReconstruction,
                        PipelineComponent::Media,
                        Some(additional.len() as u64),
                        ObserverMode::Ffmpeg,
                    )),
                ),
                crate::media::SelectedFrameExtraction {
                    base: crate::media::FrameExtraction {
                        input: metadata.source_path.clone(),
                        output_directory: paths.frames.clone(),
                        colmap_directory: paths.colmap_frames.clone(),
                        mask_directory: paths.masks.clone(),
                        has_alpha: prepared.has_alpha,
                        target_dimensions: Some((prepared.working_width, prepared.working_height)),
                    },
                    selected_frames: additional.to_vec(),
                },
            )
            .await?;
        for frame in additional {
            if !prepared
                .plan
                .selected_frames
                .iter()
                .any(|selected| selected.source_frame_index == frame.source_frame_index)
            {
                prepared.plan.selected_frames.push(frame.clone());
            }
        }
        prepared
            .plan
            .selected_frames
            .sort_by_key(|frame| frame.source_frame_index);
        prepared.extracted_frames = extraction.frame_count;
        prepared.mask_count = extraction.mask_count;
        prepared.plan.estimated_frames = extraction.frame_count;
        prepared.plan.retention_ratio = prepared
            .video
            .as_ref()
            .map(|video| extraction.frame_count as f64 / video.total_frames.max(1) as f64)
            .unwrap_or(prepared.plan.retention_ratio);
        if let Some(video) = prepared.video.as_ref() {
            prepared.plan.sampling_fps = extraction.frame_count as f64 / video.duration.max(0.001);
        }
        update_frame_checkpoint(state, prepared);
        project_manager.write_state(&paths.state, state).await?;

        let extension = if prepared.has_alpha { "png" } else { "jpg" };
        let image_list = paths.colmap.join("bridge-images.txt");
        let image_list_text = additional
            .iter()
            .map(|frame| format!("frame_{:010}.{extension}", frame.source_frame_index))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        tokio::fs::write(&image_list, image_list_text).await?;
        let preset = metadata.quality.preset();
        let sfm_max_image_size = state
            .resolution_plan
            .map(|plan| plan.sfm_max_image_size)
            .unwrap_or(if state.planner_enabled {
                preset.planned_sfm_max_image_size
            } else {
                preset.sfm_max_image_size
            });
        self.events.stage(
            PipelineStage::ValidatingReconstruction,
            0.0,
            format!(
                "Bridge Backfill 正在为 {} 张新增帧提取特征",
                additional.len()
            ),
        );
        colmap::extract_features_quality_v2(
            &self.engines.colmap,
            database,
            colmap_images,
            colmap_masks,
            Some(&image_list),
            sfm_max_image_size,
            preset.planned_sfm_max_features,
            paths.logs.join("colmap-bridge-features.log"),
            &self.process_manager,
            Some(self.process_observer(
                PipelineStage::ValidatingReconstruction,
                PipelineComponent::Reconstruction,
                Some(additional.len() as u64),
                ObserverMode::BracketProgress,
            )),
            gpu_index,
            None,
        )
        .await?;

        let all_indices = prepared
            .plan
            .selected_frames
            .iter()
            .map(|frame| frame.source_frame_index)
            .collect::<Vec<_>>();
        let added_indices = additional
            .iter()
            .map(|frame| frame.source_frame_index)
            .collect::<Vec<_>>();
        let pair_list = paths.colmap.join("bridge-pairs.txt");
        let pair_count =
            write_bridge_pair_list(&pair_list, &all_indices, &added_indices, prepared.has_alpha)?;
        if pair_count == 0 {
            return Err(SplatError::Process(
                "Bridge Backfill did not produce any local matching pairs".into(),
            ));
        }
        self.events.stage(
            PipelineStage::ValidatingReconstruction,
            0.0,
            format!("Bridge Backfill 正在匹配 {pair_count} 组局部帧对"),
        );
        self.backend
                .match_pairs(
                    &self.job_context(
                        paths.logs.join("colmap-bridge-matching.log"),
                        Some(self.process_observer(
                PipelineStage::ValidatingReconstruction,
                PipelineComponent::Reconstruction,
                Some(pair_count as u64),
                ObserverMode::Matching,
            )),
                        gpu_index,
                    ),
                    PairMatching {
                        database: database.to_path_buf(),
                        pair_list: pair_list.clone(),
                    },
                )
                .await?;

        let bridge_sparse = paths.colmap.join("sparse-bridge");
        reset_directory(&bridge_sparse).await?;
        self.events.stage(
            PipelineStage::ValidatingReconstruction,
            0.0,
            "Bridge Backfill 正在复用初始模型继续增量重建",
        );
        let input_model = initial_model
            .strip_prefix(&paths.colmap)
            .unwrap_or(initial_model);
        self.backend
                .map_from_existing(
                    &self.job_context(
                        colmap_log.with_file_name("colmap-bridge-mapper.log"),
                        Some(self.process_observer(
                PipelineStage::ValidatingReconstruction,
                PipelineComponent::Reconstruction,
                Some(prepared.extracted_frames),
                ObserverMode::Mapper,
            )),
                        gpu_index,
                    ),
                    SparseMapping {
                        database: database.to_path_buf(),
                        images: colmap_images.to_path_buf(),
                        input_model: input_model.to_path_buf(),
                        output: bridge_sparse.clone(),
                        allow_two_view_tracks: false,
                        automatic_optimization: state.planner_enabled,
                    },
                )
                .await?;
        let (model, report) =
            best_sparse_model_with_input_count(&bridge_sparse, prepared.extracted_frames).await?;
        if report.registered_images < initial_report.registered_images {
            return Err(SplatError::Process(format!(
                "Bridge model registered fewer images than the initial model ({} < {})",
                report.registered_images, initial_report.registered_images
            )));
        }
        Ok((model, report))
    }

    fn process_observer(
        &self,
        stage: PipelineStage,
        component: PipelineComponent,
        expected_total: Option<u64>,
        mode: ObserverMode,
    ) -> ProcessObserver {
        let events = self.events.clone();
        let mapper_count = Arc::new(AtomicU64::new(0));
        let runtime = Arc::new(std::sync::Mutex::new(RuntimeTracker::new()));
        let matching_notice_elapsed = Arc::new(AtomicU64::new(0));
        let process_active = Arc::new(AtomicBool::new(false));
        let cancellation = self.process_manager.child_token();
        let operation = Arc::new(std::sync::Mutex::new(None));
        Arc::new(move |update| {
            let mut scoped = events.clone();
            if matches!(update, ProcessUpdate::Started { .. }) {
                scoped = events.begin_process(stage);
                *operation.lock().unwrap_or_else(|p| p.into_inner()) =
                    scoped.bound_operation.clone();
            } else {
                scoped.bound_operation =
                    operation.lock().unwrap_or_else(|p| p.into_inner()).clone();
            }
            let events = scoped;
            match update {
                ProcessUpdate::LogOpened { .. } => {}
                ProcessUpdate::Exited { .. } => {
                    process_active.store(false, Ordering::Relaxed);
                }
                ProcessUpdate::Started { process_id } => {
                    let mut tracker = runtime
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    *tracker = RuntimeTracker::new();
                    mapper_count.store(0, Ordering::Relaxed);
                    matching_notice_elapsed.store(0, Ordering::Relaxed);
                    process_active.store(true, Ordering::Relaxed);
                    tracker.snapshot.process_id = process_id;
                    if mode != ObserverMode::Brush {
                        tracker.snapshot.phase = "running".into();
                    }
                    events.send(
                        stage,
                        Some(component),
                        EventKind::Stage,
                        EventLevel::Info,
                        None,
                        true,
                        format!("进程已启动 · PID {process_id}"),
                        None,
                        expected_total,
                        None,
                    );
                    events.runtime(stage, component, tracker.refresh(Instant::now()));
                }
                ProcessUpdate::Resources(sample) => {
                    let mut tracker = runtime
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if tracker.snapshot.process_id != sample.process_id {
                        return;
                    }
                    tracker.snapshot.resources = Some(sample);
                    events.runtime(stage, component, tracker.refresh(Instant::now()));
                }
                ProcessUpdate::Heartbeat { elapsed_ms } => {
                    let mut tracker = runtime
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let snapshot = tracker.refresh(Instant::now());
                    let quiet_ms = snapshot.last_output_age_ms;
                    events.runtime(stage, component, snapshot);
                    if mode == ObserverMode::Matching
                        && process_active.load(Ordering::Relaxed)
                        && !cancellation.is_cancelled()
                        && matching_quiet_notice_due(&matching_notice_elapsed, elapsed_ms, quiet_ms)
                    {
                        let message = format!(
                            "图像匹配进程仍在运行，最近 {} 秒无新输出。此步骤可能耗时较长，请耐心等待",
                            quiet_ms / 1000
                        );
                        events.send(
                            stage,
                            Some(PipelineComponent::System),
                            EventKind::Log,
                            EventLevel::Info,
                            None,
                            true,
                            message,
                            None,
                            None,
                            None,
                        );
                    }
                }
                ProcessUpdate::Line { stream: _, line } => {
                    if line.is_empty() {
                        return;
                    }
                    matching_notice_elapsed.store(0, Ordering::Relaxed);
                    let brush = mode == ObserverMode::Brush;
                    let actual = {
                        let mut tracker = runtime
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        let actual = tracker.line(&line, Instant::now(), brush);
                        events.runtime(stage, component, tracker.refresh(Instant::now()));
                        actual
                    };
                    if mode == ObserverMode::Matching {
                        if let Some(activity) = matching_activity(&line) {
                            events.send(
                                stage,
                                Some(component),
                                EventKind::Progress,
                                EventLevel::Info,
                                None,
                                true,
                                friendly_engine_line(&line),
                                None,
                                None,
                                Some(activity),
                            );
                        }
                    }
                    let parsed = match mode {
                        ObserverMode::Ffmpeg => parse_ffmpeg_frame(&line).map(|current| {
                            (
                                current,
                                expected_total,
                                format!("FFmpeg 已输出 {current} 帧"),
                            )
                        }),
                        ObserverMode::BracketProgress => {
                            parse_bracket_progress(&line).map(|(current, total)| {
                                (current, Some(total), friendly_engine_line(&line))
                            })
                        }
                        ObserverMode::Matching => {
                            parse_matching_progress(&line).map(|(current, total)| {
                                (current, Some(total), friendly_engine_line(&line))
                            })
                        }
                        ObserverMode::Mapper => {
                            parse_mapper_progress(&line, &mapper_count, expected_total)
                        }
                        ObserverMode::Brush => actual.map(|(current, total)| {
                            (
                                current,
                                Some(total),
                                format!("Brush 训练 · {current}/{total} 步"),
                            )
                        }),
                    };
                    if let Some((current, total, message)) = parsed {
                        let progress = total
                            .filter(|value| *value > 0)
                            .filter(|_| {
                                mode != ObserverMode::Matching && mode != ObserverMode::Mapper
                            })
                            .map(|value| current as f32 / value as f32);
                        events.send(
                            stage,
                            Some(component),
                            EventKind::Progress,
                            EventLevel::Info,
                            progress.map(|value| {
                                if brush {
                                    value.min(0.99)
                                } else {
                                    value.min(0.999)
                                }
                            }),
                            progress.is_none(),
                            message,
                            Some(current),
                            total,
                            Some(if brush {
                                "steps"
                            } else if mode == ObserverMode::Matching {
                                matching_unit(&line)
                            } else if mode == ObserverMode::Mapper {
                                "registeredImages"
                            } else {
                                "images"
                            }),
                        );
                    } else if mode == ObserverMode::Brush {
                        events.send(
                            stage,
                            Some(component),
                            EventKind::Log,
                            if line.contains(" WARN ") {
                                EventLevel::Warning
                            } else if line.contains(" ERROR ")
                                || line.contains("panicked at")
                                || line.starts_with("Error:")
                            {
                                EventLevel::Error
                            } else {
                                EventLevel::Info
                            },
                            None,
                            true,
                            friendly_engine_line(&line),
                            None,
                            expected_total,
                            None,
                        );
                    } else if is_useful_line(&line) {
                        events.send(
                            stage,
                            Some(component),
                            EventKind::Log,
                            EventLevel::Info,
                            None,
                            true,
                            friendly_engine_line(&line),
                            None,
                            expected_total,
                            None,
                        );
                    }
                }
                ProcessUpdate::StageProgress {
                    phase,
                    current,
                    total,
                    message,
                } => {
                    {
                        let mut tracker = runtime
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        tracker.snapshot.phase = phase.into();
                        events.runtime(stage, component, tracker.refresh(Instant::now()));
                    }
                    let progress = total
                        .filter(|value| *value > 0)
                        .map(|value| current as f32 / value as f32);
                    events.send(
                        stage,
                        Some(component),
                        EventKind::Progress,
                        EventLevel::Info,
                        progress,
                        progress.is_none(),
                        message,
                        Some(current),
                        total,
                        None,
                    );
                }
            }
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ObserverMode {
    Ffmpeg,
    BracketProgress,
    Matching,
    Mapper,
    Brush,
}

fn matching_quiet_notice_due(last_notice: &AtomicU64, elapsed_ms: u64, quiet_ms: u64) -> bool {
    if quiet_ms < 15_000 || elapsed_ms == 0 {
        return false;
    }
    let previous = last_notice.load(Ordering::Relaxed);
    (previous == 0 || elapsed_ms.saturating_sub(previous) >= 30_000)
        && last_notice
            .compare_exchange(previous, elapsed_ms, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
}

fn parse_ffmpeg_frame(line: &str) -> Option<u64> {
    line.strip_prefix("frame=")?.trim().parse().ok()
}

fn parse_bracket_progress(line: &str) -> Option<(u64, u64)> {
    if !line.contains("Processed file [") && !line.contains("Processed image [") {
        return None;
    }
    parse_counter_brackets(line)
}

fn parse_matching_progress(line: &str) -> Option<(u64, u64)> {
    if ![
        "Matching block [",
        "Matching image [",
        "Processing image [",
        "Indexing image [",
    ]
    .iter()
    .any(|marker| line.contains(marker))
    {
        return None;
    }
    parse_counter_brackets(line)
}

fn matching_unit(line: &str) -> &'static str {
    if line.contains("Indexing image [") {
        "matchingIndexImages"
    } else if line.contains("Processing image [") || line.contains("Matching image [") {
        "matchingImages"
    } else {
        "matchingBatches"
    }
}

fn matching_activity(line: &str) -> Option<&'static str> {
    let lower = line.to_ascii_lowercase();
    if lower.contains("vocabulary tree")
        || lower.contains("loop detection")
        || lower.contains("loop closure")
    {
        Some("matchingLoop")
    } else if lower.contains("generating sequential image pairs") {
        Some("matchingSequential")
    } else if lower.contains("matching local") || lower.contains("importing image pairs") {
        Some("matchingLocal")
    } else {
        None
    }
}

fn parse_counter_brackets(line: &str) -> Option<(u64, u64)> {
    let open = line.find('[')?;
    let close = line[open + 1..].find(']')? + open + 1;
    let value = &line[open + 1..close];
    let (current, total) = value.split_once('/')?;
    let current = current.trim().parse().ok()?;
    let total = total.trim().parse().ok()?;
    (total > 0 && current <= total).then_some((current, total))
}

fn parse_mapper_progress(
    line: &str,
    counter: &AtomicU64,
    expected_total: Option<u64>,
) -> Option<(u64, Option<u64>, String)> {
    if line.contains("Registering initial image pair")
        || line.contains("Initializing with image pair")
    {
        counter.store(0, Ordering::Relaxed);
        return Some((0, expected_total, friendly_engine_line(line)));
    }
    let reported_count = value_after(line, "num_reg_frames=")
        .or_else(|| value_after(line, "num_reg_frames ="))
        .and_then(|value| value.parse::<u64>().ok());
    let lower = line.to_ascii_lowercase();
    if lower.contains("retriangulation") || lower.contains("global bundle adjustment") {
        if let Some(current) = reported_count {
            counter.fetch_max(current, Ordering::Relaxed);
        }
        let current = counter.load(Ordering::Relaxed);
        if current > 0 {
            return Some((current, expected_total, friendly_engine_line(line)));
        }
        return None;
    }
    if let Some(current) = reported_count {
        counter.store(current, Ordering::Relaxed);
        return Some((current, expected_total, format!("已注册 {current} 张图像")));
    }
    None
}

fn value_after<'a>(line: &'a str, marker: &str) -> Option<&'a str> {
    let value = line.split_once(marker)?.1;
    Some(
        value
            .split_whitespace()
            .next()?
            .trim_matches(|ch: char| !ch.is_ascii_digit()),
    )
}

fn is_useful_line(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    [
        "bundle",
        "register",
        "triang",
        "elapsed",
        "warning",
        "error",
        "writing",
        "loading",
        "matching",
        "loop detection",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn friendly_engine_line(line: &str) -> String {
    const MAX: usize = 360;
    let mut value = line.trim().to_string();
    if value.chars().count() > MAX {
        value = value.chars().take(MAX).collect::<String>() + "…";
    }
    value
}

fn format_duration(milliseconds: u64) -> String {
    let seconds = milliseconds / 1_000;
    format!(
        "{:02}:{:02}:{:02}",
        seconds / 3_600,
        (seconds % 3_600) / 60,
        seconds % 60
    )
}

fn brush_oom_fallback(
    quality: Quality,
    planner_enabled: bool,
    retry_used: bool,
    resolved: ResolvedBrushTrainingPreset,
    initial_sfm_points: u64,
) -> Option<ResolvedBrushTrainingPreset> {
    ((quality == Quality::High || resolved.panorama.is_some()) && planner_enabled && !retry_used)
        .then(|| resolved.downgrade_after_oom(initial_sfm_points))
        .flatten()
}

fn should_allow_two_view_tracks(
    planner_enabled: bool,
    preset_allows_two_view_tracks: bool,
    has_transparency_mask: bool,
) -> bool {
    planner_enabled && preset_allows_two_view_tracks && has_transparency_mask
}

fn checkpoint_stage(state: &PipelineStateFile) -> PipelineStage {
    if state.brush_complete {
        PipelineStage::TrainingSplats
    } else if state.reconstruction_complete {
        PipelineStage::Reconstructing
    } else if state.matching_complete {
        PipelineStage::Matching
    } else if state.features_complete {
        PipelineStage::ExtractingFeatures
    } else if state
        .frames
        .as_ref()
        .and_then(|frames| frames.extracted_frames)
        .is_some_and(|count| count > 0)
    {
        PipelineStage::ExtractingFrames
    } else {
        PipelineStage::Created
    }
}

async fn count_image_files(directory: &Path) -> Result<u64> {
    let directory = directory.to_path_buf();
    crate::presets::spawn_pipeline_blocking(move || {
        Ok::<u64, SplatError>(crate::video::list_images(&directory)?.len() as u64)
    })
    .await
    .map_err(|error| SplatError::Process(format!("无法统计输入画面：{error}")))?
}

async fn ensure_reshoot_input_is_new(
    source_project: &Path,
    stored_source: &Path,
    input: &Path,
    input_type: ReshootInputType,
) -> Result<()> {
    let canonical_input = tokio::fs::canonicalize(input).await.ok();
    let canonical_project = tokio::fs::canonicalize(source_project).await.ok();
    if canonical_input
        .as_ref()
        .zip(canonical_project.as_ref())
        .is_some_and(|(input, project)| input.starts_with(project))
    {
        return Err(original_reshoot_input_error());
    }

    let is_original = match input_type {
        ReshootInputType::Video => {
            stored_source.is_file() && files_have_same_content(stored_source, input).await?
        }
        ReshootInputType::Images => {
            let same_as_source = stored_source.is_dir()
                && image_sequences_share_content(stored_source, input).await?;
            if same_as_source {
                true
            } else {
                let prepared_source_frames = source_project.join("work/frames");
                prepared_source_frames.is_dir()
                    && image_sequences_share_content(&prepared_source_frames, input).await?
            }
        }
    };
    if is_original {
        Err(original_reshoot_input_error())
    } else {
        Ok(())
    }
}

fn original_reshoot_input_error() -> SplatError {
    SplatError::Process("补拍时不能再次使用原素材，请选择新拍摄的视频或图片序列。".into())
}

async fn image_sequences_share_content(left: &Path, right: &Path) -> Result<bool> {
    if !left.is_dir() || !right.is_dir() {
        return Ok(false);
    }
    let left = left.to_path_buf();
    let right = right.to_path_buf();
    let (left, right) = crate::presets::spawn_pipeline_blocking(move || {
        Ok::<_, SplatError>((
            crate::video::list_images(&left)?,
            crate::video::list_images(&right)?,
        ))
    })
    .await
    .map_err(|error| SplatError::Process(format!("比较补拍素材失败：{error}")))??;
    let mut source_fingerprints: HashMap<(u64, u64), Vec<PathBuf>> = HashMap::new();
    for path in left {
        source_fingerprints
            .entry(file_fingerprint(&path).await?)
            .or_default()
            .push(path);
    }
    for path in right {
        if let Some(candidates) = source_fingerprints.get(&file_fingerprint(&path).await?) {
            for source in candidates {
                if files_have_same_content(source, &path).await? {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

async fn file_fingerprint(path: &Path) -> Result<(u64, u64)> {
    let mut file = tokio::fs::File::open(path).await?;
    let length = file.metadata().await?.len();
    let mut hash = 0xcbf29ce484222325_u64;
    let mut buffer = vec![0_u8; 256 * 1024];
    loop {
        let read = file.read(&mut buffer).await?;
        if read == 0 {
            return Ok((length, hash));
        }
        for byte in &buffer[..read] {
            hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
        }
    }
}

async fn files_have_same_content(left: &Path, right: &Path) -> Result<bool> {
    if !left.is_file() || !right.is_file() {
        return Ok(false);
    }
    let left_metadata = tokio::fs::metadata(left).await?;
    let right_metadata = tokio::fs::metadata(right).await?;
    if left_metadata.len() != right_metadata.len() {
        return Ok(false);
    }

    let mut left_file = tokio::fs::File::open(left).await?;
    let mut right_file = tokio::fs::File::open(right).await?;
    let mut left_buffer = vec![0_u8; 256 * 1024];
    let mut right_buffer = vec![0_u8; 256 * 1024];
    loop {
        let left_read = left_file.read(&mut left_buffer).await?;
        let right_read = right_file.read(&mut right_buffer).await?;
        if left_read != right_read || left_buffer[..left_read] != right_buffer[..right_read] {
            return Ok(false);
        }
        if left_read == 0 {
            return Ok(true);
        }
    }
}

fn mark_state_terminal(mut state: PipelineStateFile, cancelled: bool) -> PipelineStateFile {
    state.stage = if cancelled {
        PipelineStage::Cancelled
    } else {
        PipelineStage::Failed
    };
    state
}

async fn copy_image_files(source: &Path, destination: &Path) -> Result<()> {
    tokio::fs::create_dir_all(destination).await?;
    for image in crate::video::list_images(source)? {
        let name = image
            .file_name()
            .ok_or_else(|| SplatError::InvalidPath(image.clone()))?;
        tokio::fs::copy(&image, destination.join(name)).await?;
    }
    Ok(())
}

async fn remove_reshoot_images(directory: &Path) -> Result<()> {
    if !directory.is_dir() {
        return Ok(());
    }
    let mut entries = tokio::fs::read_dir(directory).await?;
    while let Some(entry) = entries.next_entry().await? {
        if entry.file_name().to_string_lossy().starts_with("reshoot_") && entry.path().is_file() {
            tokio::fs::remove_file(entry.path()).await?;
        }
    }
    Ok(())
}

async fn move_extracted_reshoot(
    extracted_frames: &Path,
    extracted_colmap_frames: &Path,
    extracted_masks: &Path,
    frames: &Path,
    colmap_frames: &Path,
    masks: &Path,
    has_alpha: bool,
) -> Result<()> {
    tokio::fs::create_dir_all(frames).await?;
    tokio::fs::create_dir_all(colmap_frames).await?;
    if has_alpha {
        tokio::fs::create_dir_all(masks).await?;
    }
    let mut pending = Vec::new();
    for (index, source) in crate::video::list_images(extracted_frames)?
        .iter()
        .enumerate()
    {
        let extension = source
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("jpg");
        let old_name = source
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| SplatError::InvalidPath(source.clone()))?;
        let new_name = format!("reshoot_{:06}.{extension}", index + 1);
        pending.push((source.clone(), frames.join(&new_name)));
        pending.push((
            extracted_colmap_frames.join(old_name),
            colmap_frames.join(&new_name),
        ));
        if has_alpha {
            pending.push((
                extracted_masks.join(format!("{old_name}.png")),
                masks.join(format!("{new_name}.png")),
            ));
        }
    }
    for (source, destination) in &pending {
        if !source.is_file() || destination.exists() {
            return Err(SplatError::Process(format!(
                "补拍三路图片无法安全合并：{} -> {}",
                source.display(),
                destination.display()
            )));
        }
    }
    let mut moved = Vec::new();
    for (source, destination) in pending {
        if let Err(error) = tokio::fs::rename(&source, &destination).await {
            for (original, target) in moved.into_iter().rev() {
                let _ = tokio::fs::rename(target, original).await;
            }
            return Err(error.into());
        }
        moved.push((source, destination));
    }
    Ok(())
}

async fn image_names_with_prefix(directory: &Path, prefix: &str) -> Result<Vec<String>> {
    Ok(crate::video::list_images(directory)?
        .into_iter()
        .filter_map(|path| {
            path.file_name()
                .and_then(|value| value.to_str())
                .map(str::to_owned)
        })
        .filter(|name| name.starts_with(prefix))
        .collect())
}

async fn write_image_list(path: &Path, names: &[String]) -> Result<()> {
    let mut value = names.join("\n");
    value.push('\n');
    tokio::fs::write(path, value).await?;
    Ok(())
}

async fn validate_image_dimensions(
    directory: &Path,
    prefix: &str,
    expected_width: u64,
    expected_height: u64,
) -> Result<()> {
    let directory = directory.to_path_buf();
    let prefix = prefix.to_owned();
    crate::presets::spawn_pipeline_blocking(move || {
        use image::GenericImageView;
        for path in crate::video::list_images(&directory)? {
            let name = path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default();
            if !name.starts_with(&prefix) {
                continue;
            }
            let dimensions = image::ImageReader::open(&path)
                .map_err(|error| {
                    SplatError::Process(format!("无法读取补拍画面 {}：{error}", path.display()))
                })?
                .with_guessed_format()
                .map_err(|error| {
                    SplatError::Process(format!("无法识别补拍画面 {}：{error}", path.display()))
                })?
                .decode()
                .map_err(|error| {
                    SplatError::Process(format!("补拍画面损坏 {}：{error}", path.display()))
                })?
                .dimensions();
            if u64::from(dimensions.0) != expected_width
                || u64::from(dimensions.1) != expected_height
            {
                return Err(SplatError::Process(format!(
                    "补拍画面 {} 的分辨率为 {}×{}，需要与原项目保持 {}×{}",
                    name, dimensions.0, dimensions.1, expected_width, expected_height
                )));
            }
        }
        Ok(())
    })
    .await
    .map_err(|error| SplatError::Process(format!("补拍画面校验失败：{error}")))?
}

/// Temporary #73 hook. Never archive the (potentially hard-linked) pictures,
/// masks, final.ply or edits. Persist the reuse barrier BEFORE moving caches.
async fn apply_temporary_png_focal_checkpoint(
    manager: &ProjectManager,
    paths: &ProjectPaths,
    state: &mut PipelineStateFile,
    checkpoint: &crate::video::temporary_png_focal_workaround::Checkpoint,
) -> Result<bool> {
    if state.temporary_png_focal.as_ref() == Some(checkpoint) {
        return Ok(false);
    }
    let mut caches = Vec::new();
    for (path, name) in [(&paths.colmap, "colmap"), (&paths.brush, "brush")] {
        if path.exists() {
            if tokio::fs::symlink_metadata(path)
                .await?
                .file_type()
                .is_symlink()
            {
                return Err(SplatError::InvalidPath(path.clone()));
            }
            if tokio::fs::read_dir(path)
                .await?
                .next_entry()
                .await?
                .is_some()
            {
                caches.push((path.clone(), name));
            }
        }
    }
    let invalidated = !caches.is_empty()
        || state.features_complete
        || state.matching_complete
        || state.reconstruction_complete
        || state.brush_complete;
    state.features_complete = false;
    state.matching_complete = false;
    state.reconstruction_complete = false;
    state.brush_complete = false;
    state.brush_training = Default::default();
    state.bridge_backfill = Default::default();
    state.stage = PipelineStage::ExtractingFrames;
    state.temporary_png_focal = None;
    manager.write_state(&paths.state, state).await?;
    if !caches.is_empty() {
        let archive = paths
            .work
            .join("cache-archive")
            .join(format!("temporary-png-focal-v1-{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&archive).await?;
        let mut moved = Vec::new();
        for (source, name) in caches {
            let destination = archive.join(name);
            if let Err(error) = tokio::fs::rename(&source, &destination).await {
                for (original, saved) in moved.into_iter().rev() {
                    // State stays non-reusable even if rollback itself fails.
                    let _ = tokio::fs::rename(saved, original).await;
                }
                return Err(SplatError::Process(format!(
                    "Cannot archive temporary PNG focal cache: {error}"
                )));
            }
            moved.push((source, destination));
        }
    }
    state.temporary_png_focal = Some(checkpoint.clone());
    manager.write_state(&paths.state, state).await?;
    Ok(invalidated)
}

async fn archive_incompatible_colmap_input_cache(
    paths: &ProjectPaths,
    state: &PipelineStateFile,
) -> Result<()> {
    let frame_cache_incompatible = state.frames.as_ref().is_some_and(|frames| {
        frames.extracted_frames.is_some()
            && frames.colmap_input_version != Some(COLMAP_INPUT_VERSION)
    });
    let reshoot_cache_incompatible = state.reshoot.as_ref().is_some_and(|reshoot| {
        (reshoot.source_snapshot_complete
            || reshoot.supplemental_frames_complete
            || reshoot.supplemental_features_complete
            || reshoot.incremental_matching_complete
            || reshoot.incremental_reconstruction_complete)
            && reshoot.colmap_input_version != Some(COLMAP_INPUT_VERSION)
    });
    if !frame_cache_incompatible && !reshoot_cache_incompatible {
        return Ok(());
    }

    let archive_root = paths
        .work
        .join("cache-archive")
        .join(format!("colmap-input-vunknown-{}", uuid::Uuid::new_v4()));
    tokio::fs::create_dir_all(&archive_root).await?;
    let candidates = vec![
        (paths.frames.clone(), "frames"),
        (paths.colmap_frames.clone(), "colmap_frames"),
        (paths.masks.clone(), "masks"),
        (paths.colmap.clone(), "colmap"),
        (paths.brush.clone(), "brush"),
        (paths.work.join("reshoot-masks"), "reshoot-masks"),
    ];
    let mut moved: Vec<(PathBuf, PathBuf)> = Vec::new();
    for (source, name) in candidates {
        if !source.exists() {
            continue;
        }
        let destination = archive_root.join(name);
        if let Err(error) = tokio::fs::rename(&source, &destination).await {
            for (original, archived) in moved.into_iter().rev() {
                let _ = tokio::fs::rename(archived, original).await;
            }
            let _ = tokio::fs::remove_dir_all(&archive_root).await;
            return Err(SplatError::Process(format!(
                "无法保存旧版 COLMAP 缓存 {}：{error}",
                source.display()
            )));
        }
        moved.push((source, destination));
    }
    Ok(())
}

async fn validate_current_colmap_input_snapshot(
    project: &Path,
    state: &PipelineStateFile,
) -> Result<PreparedImageSequence> {
    let frame_state = state
        .frames
        .as_ref()
        .ok_or_else(|| SplatError::Process("项目缺少已准备图片的检查点记录".into()))?;
    if frame_state.colmap_input_version != Some(COLMAP_INPUT_VERSION) {
        return Err(SplatError::Process(
            "项目使用旧版或未知的 COLMAP 图片处理语义".into(),
        ));
    }
    let expected_count = frame_state
        .extracted_frames
        .filter(|count| *count > 0)
        .ok_or_else(|| SplatError::Process("项目缺少已准备图片数量".into()))?;
    let frames = project.join("work/frames");
    let colmap_frames = project.join("work/colmap_frames");
    let masks = project.join("work/masks");
    let has_alpha = frame_state.has_alpha;
    crate::presets::spawn_pipeline_blocking(move || {
        validate_prepared_image_sequence(&frames, &colmap_frames, &masks, expected_count, has_alpha)
    })
    .await
    .map_err(|error| SplatError::Process(format!("校验 COLMAP 图片快照失败：{error}")))?
}

async fn normalize_checkpoints(paths: &ProjectPaths, state: &mut PipelineStateFile) -> Result<()> {
    let frames_complete = prepared_frames_from_checkpoint(paths, state)
        .await?
        .is_some();
    if !frames_complete {
        state.video = None;
        state.image_sequence = None;
        state.frames = None;
        state.bridge_backfill = Default::default();
    }

    let database_complete = tokio::fs::metadata(paths.colmap.join("database.db"))
        .await
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0);
    state.features_complete = frames_complete && state.features_complete && database_complete;
    state.matching_complete = state.features_complete && state.matching_complete;
    state.reconstruction_complete = state.matching_complete
        && state.reconstruction_complete
        && best_sparse_model(&paths.frames, &paths.colmap.join("sparse"))
            .await
            .is_ok();
    if !state.reconstruction_complete {
        state.brush_training = Default::default();
    }
    state.brush_complete = state.reconstruction_complete
        && state.brush_complete
        && brush_candidate(&paths.brush)
            .and_then(|path| inspect_gaussian_ply(&path).ok())
            .is_some();
    state.stage = checkpoint_stage(state);
    Ok(())
}

async fn prepared_frames_from_checkpoint(
    paths: &ProjectPaths,
    state: &PipelineStateFile,
) -> Result<Option<PreparedFrames>> {
    let Some(frames) = state.frames.as_ref() else {
        return Ok(None);
    };
    if frames.colmap_input_version != Some(COLMAP_INPUT_VERSION) {
        return Ok(None);
    }
    let Some(extracted_frames) = frames.extracted_frames.filter(|count| *count > 0) else {
        return Ok(None);
    };
    let plan = FramePlan {
        retention_ratio: frames.retention_ratio,
        sampling_fps: frames.sampling_fps,
        estimated_frames: frames.estimated_frames,
        selected_frames: frames.selected_frames.clone(),
        candidate_frames: frames.candidate_frames.clone(),
        rescue_max_frames: frames.rescue_max_frames,
        minimum_frame_override_applied: frames.minimum_frame_override_applied,
    };
    match state.input_type {
        ProjectInputType::Video => {
            let Some(video) = state.video.clone() else {
                return Ok(None);
            };
            let has_alpha = frames.has_alpha || video.has_alpha;
            let Ok(extraction) =
                validate_extraction(&paths.frames, &paths.colmap_frames, &paths.masks, has_alpha)
                    .await
            else {
                return Ok(None);
            };
            if extraction.frame_count != extracted_frames
                || frames
                    .image_format
                    .as_deref()
                    .is_some_and(|format| format != extraction.image_format.as_str())
                || frames
                    .mask_count
                    .is_some_and(|count| count != extraction.mask_count)
            {
                return Ok(None);
            }
            if state.resolution_plan.is_some_and(|resolution| {
                (extraction.width, extraction.height)
                    != (resolution.working_width, resolution.working_height)
            }) {
                return Ok(None);
            }
            Ok(Some(PreparedFrames {
                input_type: ProjectInputType::Video,
                video: Some(video),
                image_sequence: None,
                plan,
                extracted_frames,
                image_format: extraction.image_format.as_str().into(),
                mask_count: extraction.mask_count,
                has_alpha: extraction.has_alpha,
                working_width: extraction.width,
                working_height: extraction.height,
            }))
        }
        ProjectInputType::Images => {
            let Some(image_sequence) = state.image_sequence.clone() else {
                return Ok(None);
            };
            let frames_dir = paths.frames.clone();
            let colmap_frames_dir = paths.colmap_frames.clone();
            let masks_dir = paths.masks.clone();
            let has_alpha = frames.has_alpha;
            let Ok(prepared) = crate::presets::spawn_pipeline_blocking(move || {
                validate_prepared_image_sequence(
                    &frames_dir,
                    &colmap_frames_dir,
                    &masks_dir,
                    extracted_frames,
                    has_alpha,
                )
                .map(|prepared| (prepared, image_sequence))
            })
            .await
            .map_err(|error| SplatError::Process(format!("图片检查点校验失败：{error}")))?
            else {
                return Ok(None);
            };
            let (prepared, image_sequence) = prepared;
            if frames
                .mask_count
                .is_some_and(|count| count != prepared.mask_count)
            {
                return Ok(None);
            }
            let (working_width, working_height) = state
                .panorama
                .as_ref()
                .map(|p| (p.sfm_width, p.sfm_height))
                .unwrap_or((image_sequence.width, image_sequence.height));
            Ok(Some(PreparedFrames {
                input_type: ProjectInputType::Images,
                video: None,
                image_sequence: Some(image_sequence),
                plan,
                extracted_frames,
                image_format: "images".into(),
                mask_count: prepared.mask_count,
                has_alpha: prepared.has_alpha,
                working_width,
                working_height,
            }))
        }
    }
}

fn brush_candidate(root: &Path) -> Option<PathBuf> {
    [root.join("final.ply.tmp"), root.join("final.ply.tmp.ply")]
        .into_iter()
        .find(|path| path.is_file())
}

async fn recover_interrupted_publish(
    paths: &ProjectPaths,
    state: &PipelineStateFile,
) -> Result<()> {
    if state.stage == PipelineStage::Completed
        || !state.brush_complete
        || brush_candidate(&paths.brush).is_some()
    {
        return Ok(());
    }
    let orphan = paths.project.join("final.ply");
    if !orphan.is_file() {
        return Ok(());
    }
    let inspect_path = orphan.clone();
    if crate::presets::spawn_pipeline_blocking(move || inspect_gaussian_ply(&inspect_path))
        .await
        .map_err(|error| SplatError::Process(format!("PLY 恢复校验任务失败：{error}")))?
        .is_err()
    {
        return Ok(());
    }
    tokio::fs::create_dir_all(&paths.brush).await?;
    atomic_replace_file(&orphan, &paths.brush.join("final.ply.tmp")).await
}

async fn reset_directory(path: &Path) -> Result<()> {
    if path.exists() {
        tokio::fs::remove_dir_all(path).await?;
    }
    tokio::fs::create_dir_all(path).await?;
    Ok(())
}

#[derive(Debug)]
struct ReusableReconstruction {
    model: PathBuf,
    report: ReconstructionReport,
}

#[derive(Debug)]
struct ReusableModelCandidate {
    model: PathBuf,
    report: ReconstructionReport,
    candidate_id: Option<String>,
}

async fn reusable_reconstruction(
    project: &Path,
    frames: &Path,
    output: Option<&ProjectOutput>,
) -> Result<ReusableReconstruction> {
    let project = project.to_path_buf();
    let frames = frames.to_path_buf();
    let output = output.cloned();
    crate::presets::spawn_pipeline_blocking(move || {
        reusable_reconstruction_blocking(&project, &frames, output.as_ref())
    })
    .await
    .map_err(|error| SplatError::Process(format!("读取可复用重建模型失败：{error}")))?
}

fn reusable_reconstruction_blocking(
    project: &Path,
    frames: &Path,
    output: Option<&ProjectOutput>,
) -> Result<ReusableReconstruction> {
    let colmap = project.join("work/colmap");
    let selected_candidate = selected_reconstruction_candidate(project)?;
    let mut models = Vec::new();

    collect_reusable_models(frames, &colmap.join("sparse"), None, &mut models)?;
    let candidates = colmap.join("candidates");
    if candidates.is_dir()
        && !std::fs::symlink_metadata(&candidates)?
            .file_type()
            .is_symlink()
    {
        for entry in std::fs::read_dir(&candidates)? {
            let entry = entry?;
            if entry.file_type()?.is_symlink() || !entry.file_type()?.is_dir() {
                continue;
            }
            let id = entry.file_name().to_string_lossy().into_owned();
            if !safe_candidate_id(&id) {
                continue;
            }
            collect_reusable_models(frames, &entry.path(), Some(id), &mut models)?;
        }
    }

    if models.is_empty() {
        return Err(SplatError::Process(
            "原项目缺少可复用的 COLMAP 稀疏重建模型（已检查 sparse 与 candidates）".into(),
        ));
    }
    models.sort_by(|left, right| left.model.cmp(&right.model));

    let selected_models = selected_candidate.as_deref().map(|selected| {
        models
            .iter()
            .enumerate()
            .filter(|(_, candidate)| candidate.candidate_id.as_deref() == Some(selected))
            .map(|(index, _)| index)
            .collect::<Vec<_>>()
    });
    let pool = selected_models
        .as_ref()
        .filter(|indices| !indices.is_empty())
        .cloned()
        .unwrap_or_else(|| (0..models.len()).collect());
    let exact = output.map(|output| {
        pool.iter()
            .copied()
            .filter(|index| {
                let report = &models[*index].report;
                report.registered_images == output.registered_images
                    && report.points_3d == output.points_3d
            })
            .collect::<Vec<_>>()
    });
    let pool = exact
        .as_ref()
        .filter(|indices| !indices.is_empty())
        .unwrap_or(&pool);
    let best = pool
        .iter()
        .copied()
        .max_by(|left, right| compare_reusable_models(&models[*left], &models[*right]))
        .expect("a non-empty model pool");
    let selected = models.swap_remove(best);
    Ok(ReusableReconstruction {
        model: selected.model,
        report: selected.report,
    })
}

fn compare_reusable_models(
    left: &ReusableModelCandidate,
    right: &ReusableModelCandidate,
) -> std::cmp::Ordering {
    left.report
        .registered_images
        .cmp(&right.report.registered_images)
        .then_with(|| left.report.points_3d.cmp(&right.report.points_3d))
        .then_with(|| right.model.cmp(&left.model))
}

fn collect_reusable_models(
    frames: &Path,
    root: &Path,
    candidate_id: Option<String>,
    output: &mut Vec<ReusableModelCandidate>,
) -> Result<()> {
    if !root.is_dir() {
        return Ok(());
    }
    if std::fs::symlink_metadata(root)?.file_type().is_symlink() {
        return Ok(());
    }
    if let Some(report) = validate_reusable_model(frames, root) {
        output.push(ReusableModelCandidate {
            model: root.to_path_buf(),
            report,
            candidate_id: candidate_id.clone(),
        });
    }
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() || !file_type.is_dir() {
            continue;
        }
        let path = entry.path();
        if let Some(report) = validate_reusable_model(frames, &path) {
            output.push(ReusableModelCandidate {
                model: path,
                report,
                candidate_id: candidate_id.clone(),
            });
        }
    }
    Ok(())
}

fn validate_reusable_model(frames: &Path, model: &Path) -> Option<ReconstructionReport> {
    reusable_model_files(model).ok()?;
    ReconstructionValidator::validate(frames, model).ok()
}

fn selected_reconstruction_candidate(project: &Path) -> Result<Option<String>> {
    for path in [
        project.join("state.json"),
        project.join("logs/planner.json"),
    ] {
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            continue;
        };
        let selected = value
            .pointer("/planner/bestReconstructionId")
            .or_else(|| value.pointer("/bestReconstructionId"))
            .and_then(serde_json::Value::as_str);
        let Some(selected) = selected else {
            continue;
        };
        if !safe_candidate_id(selected) {
            return Err(SplatError::Process(
                "项目记录的 COLMAP 候选模型标识不安全，已拒绝访问".into(),
            ));
        }
        return Ok(Some(selected.to_owned()));
    }
    Ok(None)
}

fn safe_candidate_id(value: &str) -> bool {
    let mut components = Path::new(value).components();
    matches!(components.next(), Some(std::path::Component::Normal(_)))
        && components.next().is_none()
        && value != "."
        && value != ".."
        && !value.contains(['/', '\\', ':'])
}

fn validate_database_camera(database: &Path, camera: &ColmapCamera) -> Result<()> {
    use rusqlite::OptionalExtension;

    let connection = rusqlite::Connection::open_with_flags(
        database,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|error| SplatError::Process(format!("无法读取原项目 COLMAP 数据库：{error}")))?;
    let stored = connection
        .query_row(
            "SELECT model, width, height, params FROM cameras WHERE camera_id = ?1",
            [camera.id],
            |row| {
                Ok((
                    row.get::<_, i32>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, Vec<u8>>(3)?,
                ))
            },
        )
        .optional()
        .map_err(|error| SplatError::Process(format!("校验原项目相机数据库失败：{error}")))?;
    let Some((model, width, height, params)) = stored else {
        return Err(SplatError::Process(format!(
            "原项目 COLMAP 数据库缺少相机 ID {}",
            camera.id
        )));
    };
    let (chunks, remainder) = params.as_chunks::<8>();
    let stored_params = chunks
        .iter()
        .map(|bytes| f64::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    if model != camera.model_id
        || width != camera.width as i64
        || height != camera.height as i64
        || !remainder.is_empty()
        || stored_params != camera.params
    {
        return Err(SplatError::Process(
            "原项目 COLMAP 数据库与最终稀疏模型的相机信息不一致".into(),
        ));
    }
    Ok(())
}

fn model_snapshot_matches(source_files: &[PathBuf], snapshot: &Path) -> bool {
    source_files.iter().all(|source| {
        let Some(name) = source.file_name() else {
            return false;
        };
        let destination = snapshot.join(name);
        match (std::fs::metadata(source), std::fs::metadata(destination)) {
            (Ok(source), Ok(destination)) => {
                source.is_file() && destination.is_file() && source.len() == destination.len()
            }
            _ => false,
        }
    })
}

async fn best_sparse_model(
    frames: &Path,
    sparse: &Path,
) -> Result<(PathBuf, ReconstructionReport)> {
    let frames = frames.to_path_buf();
    let sparse = sparse.to_path_buf();
    crate::presets::spawn_pipeline_blocking(move || best_sparse_model_blocking(&frames, &sparse))
        .await
        .map_err(|error| SplatError::Process(format!("稀疏模型校验任务失败：{error}")))?
}

async fn best_sparse_model_with_input_count(
    sparse: &Path,
    input_images: u64,
) -> Result<(PathBuf, ReconstructionReport)> {
    let sparse = sparse.to_path_buf();
    crate::presets::spawn_pipeline_blocking(move || {
        best_sparse_model_with_input_count_blocking(&sparse, input_images)
    })
    .await
    .map_err(|error| SplatError::Process(format!("Sparse model validation task failed: {error}")))?
}

fn best_sparse_model_blocking(
    frames: &Path,
    sparse: &Path,
) -> Result<(PathBuf, ReconstructionReport)> {
    // COLMAP normally writes numbered children such as `sparse/0`, while
    // continuation from `--input_path` may write directly to the requested
    // directory. Accept both layouts and keep the model with most registered
    // images.
    let mut best = ReconstructionValidator::validate(frames, sparse)
        .ok()
        .map(|report| (sparse.to_path_buf(), report));
    for entry in std::fs::read_dir(sparse)? {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        if let Ok(report) = ReconstructionValidator::validate(frames, &path) {
            if best
                .as_ref()
                .is_none_or(|(_, current)| report.registered_images > current.registered_images)
            {
                best = Some((path, report));
            }
        }
    }
    best.ok_or_else(|| SplatError::Process("COLMAP 未生成完整的稀疏模型".into()))
}

fn best_sparse_model_with_input_count_blocking(
    sparse: &Path,
    input_images: u64,
) -> Result<(PathBuf, ReconstructionReport)> {
    let mut best = ReconstructionValidator::validate_with_input_images(input_images, sparse)
        .ok()
        .map(|report| (sparse.to_path_buf(), report));
    for entry in std::fs::read_dir(sparse)? {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        if let Ok(report) = ReconstructionValidator::validate_with_input_images(input_images, &path)
        {
            if best
                .as_ref()
                .is_none_or(|(_, current)| report.registered_images > current.registered_images)
            {
                best = Some((path, report));
            }
        }
    }
    best.ok_or_else(|| SplatError::Process("COLMAP did not produce a usable sparse model".into()))
}

fn relative_model_path(paths: &ProjectPaths, model: &Path) -> String {
    model
        .strip_prefix(&paths.colmap)
        .unwrap_or(model)
        .to_string_lossy()
        .replace('\\', "/")
}

fn update_frame_checkpoint(state: &mut PipelineStateFile, prepared: &PreparedFrames) {
    if let Some(frames) = state.frames.as_mut() {
        frames.extracted_frames = Some(prepared.extracted_frames);
        frames.estimated_frames = prepared.extracted_frames;
        frames.retention_ratio = prepared.plan.retention_ratio;
        frames.sampling_fps = prepared.plan.sampling_fps;
        frames.selected_frames = prepared.plan.selected_frames.clone();
        frames.candidate_frames = prepared.plan.candidate_frames.clone();
        frames.rescue_max_frames = prepared.plan.rescue_max_frames;
        frames.minimum_frame_override_applied = prepared.plan.minimum_frame_override_applied;
        frames.mask_count = Some(prepared.mask_count);
        frames.colmap_input_version = Some(COLMAP_INPUT_VERSION);
    }
}

fn oriented_video_dimensions(video: &VideoInfo) -> (u32, u32) {
    if video.rotation.rem_euclid(180) == 90 {
        (video.height, video.width)
    } else {
        (video.width, video.height)
    }
}

async fn remove_bridge_outputs(
    frames: &Path,
    colmap_frames: &Path,
    masks: &Path,
    additional: &[PlannedFrame],
    has_alpha: bool,
) {
    let extension = if has_alpha { "png" } else { "jpg" };
    for frame in additional {
        let name = format!("frame_{:010}.{extension}", frame.source_frame_index);
        let _ = tokio::fs::remove_file(frames.join(&name)).await;
        let _ = tokio::fs::remove_file(colmap_frames.join(&name)).await;
        if has_alpha {
            let _ = tokio::fs::remove_file(masks.join(format!("{name}.png"))).await;
        }
    }
}

async fn prepare_brush_dataset(root: &Path, frames: &Path, model: &Path) -> Result<PathBuf> {
    let dataset = root.join("dataset");
    let images = dataset.join("images");
    let sparse = dataset.join("sparse").join("0");
    tokio::fs::create_dir_all(&images).await?;
    tokio::fs::create_dir_all(&sparse).await?;
    let mut entries = tokio::fs::read_dir(frames).await?;
    while let Some(entry) = entries.next_entry().await? {
        let source = entry.path();
        if !source.is_file() {
            continue;
        }
        let destination = images.join(entry.file_name());
        if tokio::fs::hard_link(&source, &destination).await.is_err() {
            tokio::fs::copy(&source, &destination).await?;
        }
    }
    for name in ["cameras.bin", "images.bin", "points3D.bin"] {
        tokio::fs::copy(model.join(name), sparse.join(name)).await?;
    }
    Ok(dataset)
}

pub fn default_engine_paths(engine_root: Option<PathBuf>) -> EnginePaths {
    engine_root
        .map(EnginePaths::from_root)
        .unwrap_or_else(|| EnginePaths::discover(None))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_matching_dispatch_preserves_video_and_quality_loop_policy() {
        use crate::project::ImageMatchingStrategy::{Exhaustive, Sequential};
        for quality in [Quality::Fast, Quality::Balanced, Quality::High] {
            for planner in [false, true] {
                assert_eq!(
                    image_matching_policy(ProjectInputType::Images, Exhaustive, quality, planner),
                    (false, false)
                );
                assert_eq!(
                    image_matching_policy(ProjectInputType::Images, Sequential, quality, planner),
                    (true, quality.loop_detection_enabled(planner))
                );
                for strategy in [Sequential, Exhaustive] {
                    assert_eq!(
                        image_matching_policy(ProjectInputType::Video, strategy, quality, planner),
                        (true, quality.loop_detection_enabled(planner))
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn resumed_matching_policy_uses_saved_state_not_new_runner_preference() {
        use crate::project::ImageMatchingStrategy::{Exhaustive, Sequential};
        let directory = tempfile::tempdir().unwrap();
        let manager = ProjectManager::for_diagnostics(directory.path().to_path_buf());
        for strategy in [Sequential, Exhaustive] {
            let mut state = PipelineStateFile::created_for(Quality::Fast, ProjectInputType::Images);
            state.image_matching_strategy = strategy;
            state.matching_complete = true;
            let path = directory.path().join("state.json");
            manager.write_state(&path, &state).await.unwrap();
            let runner = PipelineRunner::new(default_engine_paths(None), |_| {})
                .with_image_matching_strategy(if strategy == Sequential {
                    Exhaustive
                } else {
                    Sequential
                });
            let restored = manager.read_state(&path).await.unwrap();
            assert_ne!(
                restored.image_matching_strategy,
                runner.image_matching_strategy
            );
            assert_eq!(
                image_matching_policy(
                    restored.input_type,
                    restored.image_matching_strategy,
                    restored.preset,
                    restored.planner_enabled
                )
                .0,
                strategy == Sequential
            );
            assert!(restored.matching_complete);
        }
    }

    #[tokio::test]
    #[ignore = "Requires real ERP photos and bundled engines; isolated cache only"]
    async fn bundled_panorama_images_full_pipeline() {
        let input = PathBuf::from(
            std::env::var_os("OOOSPLAT_PANORAMA_IMAGES").expect("Set ERP photo directory"),
        );
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let cache = workspace.join(".cache/panorama-images-v1");
        std::fs::create_dir_all(&cache).unwrap();
        let root = tempfile::tempdir_in(&cache).unwrap().keep();
        let events = Arc::new(std::sync::Mutex::new(Vec::<PipelineEvent>::new()));
        let captured = events.clone();
        let runner = PipelineRunner::new(
            default_engine_paths(Some(workspace.join("engines"))),
            move |event| captured.lock().unwrap().push(event),
        )
        .with_video_projection(crate::video::panorama::VideoProjection::Equirectangular);
        eprintln!("Real image pipeline outputs: {}", root.display());
        let result = runner
            .generate_for_diagnostics(&input, Quality::Fast, &root)
            .await;
        std::fs::write(
            cache.join("events.json"),
            serde_json::to_vec_pretty(&*events.lock().unwrap()).unwrap(),
        )
        .unwrap();
        let result = result.unwrap();
        std::fs::write(
            cache.join("result.json"),
            serde_json::to_vec_pretty(&result).unwrap(),
        )
        .unwrap();
        assert!(
            result.input_images >= 2 && result.registered_images >= 2 && result.splat_count > 0
        );
        let project = result.final_ply.parent().unwrap();
        let state: PipelineStateFile =
            serde_json::from_slice(&std::fs::read(project.join("state.json")).unwrap()).unwrap();
        assert!(state.video.is_none());
        assert_eq!(state.input_type, ProjectInputType::Images);
        assert_eq!(
            state
                .panorama
                .as_ref()
                .unwrap()
                .image_sources
                .as_ref()
                .unwrap()
                .len() as u64,
            result.input_images
        );
        assert!(events
            .lock()
            .unwrap()
            .iter()
            .any(|e| e.unit.as_deref() == Some("panoramaImageFaces")));
        eprintln!(
            "Real ERP photos: {} input, {} registered, {} splats, {} ms; {}",
            result.input_images,
            result.registered_images,
            result.splat_count,
            result.duration_ms,
            result.final_ply.display()
        );
    }

    #[tokio::test]
    #[ignore = "Requires OOOSPLAT_PANORAMA_VIDEO and bundled COLMAP/Brush; full real pipeline in isolated cache"]
    async fn bundled_panorama_progress_full_pipeline() {
        let input = PathBuf::from(
            std::env::var_os("OOOSPLAT_PANORAMA_VIDEO").expect("Set a real stitched ERP video"),
        );
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let root = tempfile::tempdir_in(workspace.join(".cache/panorama-v1"))
            .unwrap()
            .keep();
        let captured = Arc::new(std::sync::Mutex::new(Vec::<PipelineEvent>::new()));
        let output = captured.clone();
        let runner = PipelineRunner::new(
            default_engine_paths(Some(workspace.join("engines"))),
            move |event| output.lock().unwrap().push(event),
        )
        .with_video_projection(crate::video::panorama::VideoProjection::Equirectangular);
        let result = runner
            .generate_for_diagnostics(&input, Quality::Fast, &root)
            .await;
        let events = captured.lock().unwrap();
        std::fs::write(
            workspace.join(".cache/panorama-v1/progress-e2e-events.json"),
            serde_json::to_vec_pretty(&*events).unwrap(),
        )
        .unwrap();
        let result = result.unwrap_or_else(|error| {
            panic!("Real pipeline failed: {error}; outputs: {}", root.display())
        });
        std::fs::write(
            workspace.join(".cache/panorama-v1/progress-e2e-result.json"),
            serde_json::to_vec_pretty(&result).unwrap(),
        )
        .unwrap();
        assert!(events
            .windows(2)
            .all(|pair| pair[1].progress >= pair[0].progress));
        assert!(events
            .iter()
            .filter(|event| event.stage != PipelineStage::Completed)
            .all(|event| event.progress <= 99.9));
        let mut active = None;
        for event in events.iter() {
            if event.kind == EventKind::Stage {
                active = Some(event);
            }
            if event.kind == EventKind::Runtime {
                let active = active.unwrap();
                assert_eq!(event.stage, active.stage);
                assert_eq!(event.substage, active.substage);
                assert_eq!(
                    event.operation.as_ref().unwrap().id,
                    active.operation.as_ref().unwrap().id
                );
            }
        }
        for stage in [
            PipelineStage::ExtractingFeatures,
            PipelineStage::Matching,
            PipelineStage::Reconstructing,
            PipelineStage::TrainingSplats,
        ] {
            assert!(
                events
                    .iter()
                    .any(|event| event.stage == stage && event.kind == EventKind::Runtime),
                "Missing actual engine runtime for {stage:?}"
            );
        }
        assert!(events
            .iter()
            .filter(|event| matches!(
                event.stage,
                PipelineStage::Matching | PipelineStage::Reconstructing
            ) && event.kind == EventKind::Progress)
            .all(|event| event.indeterminate && event.stage_progress.is_none()));
        eprintln!(
            "Real panorama pipeline: {} input frames, {} registered, {} splats, {} ms; PLY: {}",
            result.input_images,
            result.registered_images,
            result.splat_count,
            result.duration_ms,
            result.final_ply.display()
        );
    }

    /// Real engines, no telemetry/catalog writes. The fixture must be a completed
    /// diagnostic project; only isolated copies/derived outputs are modified.
    #[tokio::test]
    #[ignore = "Requires OOOSPLAT_PANORAMA_TEST_PROJECT and the bundled GPU runtime"]
    async fn bundled_panorama_cancel_and_resume() {
        let source = PathBuf::from(
            std::env::var_os("OOOSPLAT_PANORAMA_TEST_PROJECT")
                .expect("Set a completed diagnostic project path"),
        );
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let temporary = tempfile::tempdir_in(workspace.join(".cache/panorama-v1")).unwrap();
        let project = temporary.path().join("resume-project");
        let reuse_faces = std::env::var_os("OOOSPLAT_PANORAMA_RESUME_REUSE_FACES").is_some();
        let original_face_batches = if reuse_faces {
            std::fs::read_dir(source.join("work/panorama-training/images"))
                .unwrap()
                .count()
        } else {
            0
        };
        fn copy_readonly_tree(src: &Path, dst: &Path, reuse_faces: bool) {
            std::fs::create_dir_all(dst).unwrap();
            for e in std::fs::read_dir(src).unwrap() {
                let e = e.unwrap();
                let name = e.file_name();
                if ["panorama-training", "brush", "final.ply", "edit.ply"]
                    .iter()
                    .any(|n| {
                        name == std::ffi::OsStr::new(n)
                            && !(*n == "panorama-training" && reuse_faces)
                    })
                {
                    continue;
                }
                let dest = dst.join(&name);
                if e.file_type().unwrap().is_dir() {
                    copy_readonly_tree(&e.path(), &dest, reuse_faces);
                } else {
                    std::fs::copy(e.path(), dest).unwrap();
                }
            }
        }
        copy_readonly_tree(&source, &project, reuse_faces);
        let paths = ProjectPaths::existing(uuid::Uuid::new_v4(), project.clone());
        let manager = ProjectManager::for_diagnostics(temporary.path().into());
        let mut metadata: ProjectMetadata =
            serde_json::from_slice(&std::fs::read(&paths.metadata).unwrap()).unwrap();
        metadata.id = paths.id;
        metadata.status = ProjectStatus::Cancelled;
        metadata.output = None;
        let mut state = manager.read_state(&paths.state).await.unwrap();
        state.brush_complete = false;
        state.brush_training = Default::default();
        state.stage = PipelineStage::ValidatingReconstruction;
        manager.write_state(&paths.state, &state).await.unwrap();
        manager
            .write_metadata(&paths.metadata, &metadata)
            .await
            .unwrap();
        let sphere = paths.colmap.join("sparse/0/images.bin");
        let before = crate::video::panorama::fingerprint(&sphere).unwrap();
        let before_time = std::fs::metadata(&sphere).unwrap().modified().unwrap();
        let cancel_handle = Arc::new(std::sync::Mutex::new(None::<ProcessManager>));
        let observer = cancel_handle.clone();
        let runner = PipelineRunner::new(
            default_engine_paths(Some(workspace.join("engines"))),
            move |event| {
                if event.substage.as_deref() == Some("panoramaTrainingPreparation")
                    && event.current == Some(12)
                {
                    observer.lock().unwrap().as_ref().unwrap().cancel();
                }
            },
        );
        *cancel_handle.lock().unwrap() = Some(runner.process_manager.clone());
        let acceleration = runner.verify_pipeline_engines().await.unwrap();
        let cancelled = runner
            .execute_project(
                manager.clone(),
                paths.clone(),
                &mut metadata,
                state,
                &acceleration,
            )
            .await;
        assert!(matches!(cancelled, Err(SplatError::Cancelled)));
        assert_eq!(metadata.status, ProjectStatus::Cancelled);
        let cache = paths.work.join("panorama-training/images");
        let count = std::fs::read_dir(&cache).unwrap().count();
        assert_eq!(
            count,
            if reuse_faces {
                original_face_batches
            } else {
                2
            }
        );
        assert!(!std::fs::read_dir(&cache).unwrap().any(|e| e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".partial")));
        let first = std::fs::read_dir(&cache)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path()
            .join("front.png");
        let face_time = std::fs::metadata(&first).unwrap().modified().unwrap();
        let restored = manager.read_state(&paths.state).await.unwrap();
        let image_input = restored.input_type == ProjectInputType::Images;
        assert!(restored.reconstruction_complete);
        assert!(!restored.brush_complete);
        let cancelled_recipe = restored
            .brush_training
            .resolved
            .expect("Recipe saved before preparing faces");
        assert_eq!(cancelled_recipe.panorama.unwrap().recipe_version, 1);
        assert_eq!(cancelled_recipe.preset.total_steps, 12_000);
        let messages = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = messages.clone();
        let captured = Arc::new(std::sync::Mutex::new(Vec::<PipelineEvent>::new()));
        let observed = captured.clone();
        let runner = PipelineRunner::new(
            default_engine_paths(Some(workspace.join("engines"))),
            move |event| {
                observed.lock().unwrap().push(event.clone());
                if event.message.contains("复用") {
                    seen.lock().unwrap().push(event.message);
                }
            },
        );
        let result = runner
            .execute_project(
                manager,
                paths.clone(),
                &mut metadata,
                restored,
                &acceleration,
            )
            .await
            .unwrap();
        assert_eq!(result.registered_images, result.input_images);
        assert!(result.splat_count > 0);
        let completed: PipelineStateFile =
            serde_json::from_slice(&std::fs::read(&paths.state).unwrap()).unwrap();
        assert_eq!(
            completed.brush_training.initial_resolved,
            Some(cancelled_recipe)
        );
        assert_eq!(completed.brush_training.resolved, Some(cancelled_recipe));
        assert_eq!(
            crate::video::panorama::fingerprint(&sphere).unwrap(),
            before
        );
        assert_eq!(
            std::fs::metadata(&sphere).unwrap().modified().unwrap(),
            before_time
        );
        assert_eq!(
            std::fs::metadata(first).unwrap().modified().unwrap(),
            face_time
        );
        let seen = messages.lock().unwrap();
        for text in [
            "帧检查点",
            "特征提取检查点",
            if image_input {
                "穷举匹配检查点"
            } else {
                "顺序匹配检查点"
            },
            "相机重建检查点",
        ] {
            assert!(seen.iter().any(|m| m.contains(text)), "{text}: {seen:?}");
        }
        let events = captured.lock().unwrap();
        std::fs::write(
            workspace.join(".cache/panorama-v1/last-resume-events.json"),
            serde_json::to_vec_pretty(&*events).unwrap(),
        )
        .unwrap();
        for pair in events.windows(2) {
            assert!(
                pair[1].progress >= pair[0].progress,
                "Progress regressed: {:?} {} -> {:?} {}",
                pair[0].kind,
                pair[0].progress,
                pair[1].kind,
                pair[1].progress
            );
        }
        let faces: Vec<_> = events
            .iter()
            .filter(|event| {
                event.unit.as_deref()
                    == Some(if image_input {
                        "panoramaImageFaces"
                    } else {
                        "panoramaFaces"
                    })
            })
            .collect();
        assert!(if image_input {
            events.iter().any(|event| {
                event.kind == EventKind::Log && event.message == "已校验并复用训练缓存"
            })
        } else {
            faces.iter().any(|event| event.message.contains("复用"))
        });
        assert_eq!(
            faces.last().unwrap().current,
            Some(result.registered_images * 6)
        );
        assert_eq!(
            faces.last().unwrap().total,
            Some(result.registered_images * 6)
        );
        assert!(faces.iter().all(|event| event.progress < 66.0));
        let validated = events
            .iter()
            .position(|event| {
                event.substage.as_deref() == Some("panoramaTrainingValidation")
                    && event
                        .stage_progress
                        .is_some_and(|progress| (progress - 600.0 / 38.0).abs() < 0.001)
            })
            .unwrap();
        assert_eq!(events[validated].progress, 66.0);
        assert!(events[validated].current.is_none() && events[validated].total.is_none());
        assert!(events[validated + 1..]
            .iter()
            .any(|event| event.component == Some(PipelineComponent::Training)
                && event.current.is_some()
                && event.substage.as_deref() == Some("training")));
        assert!(events
            .iter()
            .filter(|event| event.stage != PipelineStage::Completed)
            .all(|event| event.progress < 100.0));
        eprintln!("Real cancellation/resume verified: {} ERP frames, {} splats; sphere model and completed face batches unchanged",result.input_images,result.splat_count);
    }

    #[test]
    fn reshoot_project_name_uses_the_short_suffix() {
        assert_eq!(reshoot_project_name("示例项目"), "示例项目_补拍");
    }

    #[test]
    fn two_view_tracks_require_automatic_optimization_and_transparency_masks() {
        assert!(should_allow_two_view_tracks(true, true, true));
        assert!(!should_allow_two_view_tracks(false, true, true));
        assert!(!should_allow_two_view_tracks(true, true, false));
        assert!(!should_allow_two_view_tracks(true, false, true));
    }

    #[test]
    fn reshoot_inherits_source_planner_settings() {
        for planner_enabled in [false, true] {
            let mut source = PipelineStateFile::created(Quality::Balanced);
            source.planner_enabled = planner_enabled;
            source.resolution_policy_version = planner_enabled.then_some(7);
            let mut target = PipelineStateFile::created(Quality::Balanced);
            target.planner_enabled = !planner_enabled;
            target.resolution_policy_version = (!planner_enabled).then_some(3);

            inherit_reshoot_planner_settings(&source, &mut target);

            assert_eq!(target.planner_enabled, planner_enabled);
            assert_eq!(
                target.resolution_policy_version,
                source.resolution_policy_version
            );
        }
    }

    #[test]
    fn missing_checkpoint_metrics_stay_unknown() {
        let tracker = PlannerEffectivenessTracker::default();
        let runner = PipelineRunner::new(default_engine_paths(None), |_| {})
            .with_effectiveness_tracker(tracker.clone());
        let state = PipelineStateFile::created(Quality::High);
        runner.record_frame_metrics(&state, 30, None);
        runner.record_brush_metrics(&state);
        let snapshot = tracker.snapshot();
        assert!(snapshot.frame_plan.is_none());
        assert!(snapshot.initial_reconstruction.is_none());
        assert!(snapshot.bridge.is_none());
        assert!(snapshot.brush.is_none());
        assert!(snapshot.final_result.is_none());
    }

    #[tokio::test]
    async fn image_preparation_emits_monotonic_phase_progress() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("source");
        let frames = temporary.path().join("frames");
        let colmap_frames = temporary.path().join("colmap_frames");
        let masks = temporary.path().join("masks");
        std::fs::create_dir_all(&source).unwrap();
        image::RgbImage::new(2, 2)
            .save(source.join("1.jpg"))
            .unwrap();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([1, 2, 3, 0]))
            .save(source.join("2.png"))
            .unwrap();
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured = events.clone();
        let runner = PipelineRunner::new(EnginePaths::from_root(temporary.path()), move |event| {
            captured.lock().unwrap().push(event);
        });

        let prepared = runner
            .prepare_images(
                &source,
                Quality::Balanced,
                &frames,
                &colmap_frames,
                &masks,
                false,
            )
            .await
            .unwrap();

        assert!(prepared.has_alpha);
        let events = events.lock().unwrap();
        assert!(events
            .iter()
            .any(|event| event.message.starts_with("正在建立画面链接或复制")));
        assert!(events
            .iter()
            .any(|event| event.message.starts_with("正在检测 Alpha 并生成 Mask")));
        assert!(events
            .iter()
            .any(|event| event.message == "图片与 Mask 完整性校验完成"));
        let progress = events
            .iter()
            .filter(|event| event.stage == PipelineStage::ExtractingFrames)
            .map(|event| event.stage_progress.unwrap_or_default())
            .collect::<Vec<_>>();
        assert!(
            progress.windows(2).all(|pair| pair[0] <= pair[1]),
            "stage progress regressed: {progress:?}"
        );
        assert!(events.iter().any(|event| {
            event.unit.as_deref() == Some("images")
                && event.current.is_some()
                && event.total.is_some()
        }));
    }

    fn write_validator_model(root: &Path, registered: u64, points: u64) {
        std::fs::create_dir_all(root).unwrap();
        for (name, count) in [
            ("cameras.bin", 1_u64),
            ("images.bin", registered),
            ("points3D.bin", points),
        ] {
            let mut bytes = count.to_le_bytes().to_vec();
            bytes.extend_from_slice(&[0_u8; 8]);
            std::fs::write(root.join(name), bytes).unwrap();
        }
    }

    #[test]
    fn incremental_sparse_model_is_found_at_the_output_root() {
        let temporary = tempfile::tempdir().unwrap();
        let frames = temporary.path().join("frames");
        let sparse = temporary.path().join("sparse");
        std::fs::create_dir_all(&frames).unwrap();
        std::fs::write(frames.join("frame_000001.jpg"), b"one").unwrap();
        std::fs::write(frames.join("reshoot_000001.png"), b"two").unwrap();
        write_validator_model(&sparse, 2, 50);

        let (model, report) = best_sparse_model_blocking(&frames, &sparse).unwrap();

        assert_eq!(model, sparse);
        assert_eq!(report.registered_images, 2);
        assert_eq!(report.points_3d, 50);
    }

    #[test]
    fn fresh_sparse_model_is_still_found_in_a_numbered_child() {
        let temporary = tempfile::tempdir().unwrap();
        let frames = temporary.path().join("frames");
        let sparse = temporary.path().join("sparse");
        std::fs::create_dir_all(&frames).unwrap();
        std::fs::write(frames.join("frame_000001.jpg"), b"one").unwrap();
        write_validator_model(&sparse.join("0"), 1, 25);

        let (model, report) = best_sparse_model_blocking(&frames, &sparse).unwrap();

        assert_eq!(model, sparse.join("0"));
        assert_eq!(report.registered_images, 1);
    }

    #[test]
    fn reusable_reconstruction_prefers_the_planner_selected_candidate() {
        let temporary = tempfile::tempdir().unwrap();
        let project = temporary.path();
        let frames = project.join("work/frames");
        std::fs::create_dir_all(&frames).unwrap();
        for index in 0..3 {
            std::fs::write(frames.join(format!("frame_{index}.jpg")), b"frame").unwrap();
        }
        let selected = project.join("work/colmap/candidates/selected/1");
        let other = project.join("work/colmap/candidates/other/0");
        write_validator_model(&selected, 2, 20);
        write_validator_model(&other, 3, 30);
        std::fs::write(
            project.join("state.json"),
            br#"{"planner":{"bestReconstructionId":"selected"}}"#,
        )
        .unwrap();
        let output = ProjectOutput {
            registered_images: 3,
            points_3d: 30,
            ..Default::default()
        };

        let resolved = reusable_reconstruction_blocking(project, &frames, Some(&output)).unwrap();

        assert_eq!(resolved.model, selected);
        assert_eq!(resolved.report.registered_images, 2);
    }

    #[test]
    fn reusable_reconstruction_uses_output_metrics_without_a_planner_record() {
        let temporary = tempfile::tempdir().unwrap();
        let project = temporary.path();
        let frames = project.join("work/frames");
        std::fs::create_dir_all(&frames).unwrap();
        for index in 0..3 {
            std::fs::write(frames.join(format!("frame_{index}.jpg")), b"frame").unwrap();
        }
        let expected = project.join("work/colmap/candidates/expected/0");
        let larger = project.join("work/colmap/candidates/larger/0");
        write_validator_model(&expected, 2, 25);
        write_validator_model(&larger, 3, 40);
        let output = ProjectOutput {
            registered_images: 2,
            points_3d: 25,
            ..Default::default()
        };

        let resolved = reusable_reconstruction_blocking(project, &frames, Some(&output)).unwrap();

        assert_eq!(resolved.model, expected);
    }

    #[test]
    fn reusable_reconstruction_rejects_an_unsafe_candidate_id() {
        let temporary = tempfile::tempdir().unwrap();
        let project = temporary.path();
        let frames = project.join("work/frames");
        let sparse = project.join("work/colmap/sparse/0");
        std::fs::create_dir_all(&frames).unwrap();
        std::fs::write(frames.join("frame.jpg"), b"frame").unwrap();
        write_validator_model(&sparse, 1, 10);
        std::fs::write(
            project.join("state.json"),
            br#"{"planner":{"bestReconstructionId":"../outside"}}"#,
        )
        .unwrap();

        let error = reusable_reconstruction_blocking(project, &frames, None).unwrap_err();

        assert!(error.to_string().contains("不安全"));
    }

    #[test]
    fn reusable_reconstruction_reports_missing_models_without_an_io_error() {
        let temporary = tempfile::tempdir().unwrap();
        let frames = temporary.path().join("work/frames");
        std::fs::create_dir_all(&frames).unwrap();

        let error = reusable_reconstruction_blocking(temporary.path(), &frames, None).unwrap_err();

        let message = error.to_string();
        assert!(message.contains("缺少可复用"));
        assert!(!message.contains("os error"));
    }

    #[test]
    fn reshoot_database_camera_must_match_the_selected_model() {
        let temporary = tempfile::tempdir().unwrap();
        let database = temporary.path().join("database.db");
        let connection = rusqlite::Connection::open(&database).unwrap();
        connection
            .execute(
                "CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY, model INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL, params BLOB NOT NULL)",
                [],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO cameras(camera_id, model, width, height, params) VALUES (1, 2, 1920, 1080, ?1)",
                [vec![0_u8; 32]],
            )
            .unwrap();
        drop(connection);
        let camera = ColmapCamera {
            id: 1,
            model_id: 2,
            model: "SIMPLE_RADIAL".into(),
            width: 1920,
            height: 1080,
            params: vec![0.0; 4],
        };

        validate_database_camera(&database, &camera).unwrap();
        let connection = rusqlite::Connection::open(&database).unwrap();
        for malformed in [vec![0_u8; 33], vec![0_u8; 31], vec![0_u8; 24], Vec::new()] {
            connection
                .execute("UPDATE cameras SET params = ?1", [&malformed])
                .unwrap();
            assert!(validate_database_camera(&database, &camera).is_err());
        }
        connection
            .execute("UPDATE cameras SET params = ?1", [vec![0_u8; 32]])
            .unwrap();
        assert!(validate_database_camera(
            &database,
            &ColmapCamera {
                width: 1080,
                ..camera
            },
        )
        .is_err());
    }

    #[test]
    fn model_snapshot_requires_every_source_model_file_at_the_same_size() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("source");
        let snapshot = temporary.path().join("snapshot");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::create_dir_all(&snapshot).unwrap();
        let source_files = [
            "cameras.bin",
            "images.bin",
            "points3D.bin",
            "rigs.bin",
            "frames.bin",
        ]
        .map(|name| {
            let path = source.join(name);
            std::fs::write(&path, name.as_bytes()).unwrap();
            std::fs::write(snapshot.join(name), name.as_bytes()).unwrap();
            path
        });
        assert!(model_snapshot_matches(&source_files, &snapshot));

        std::fs::write(snapshot.join("frames.bin"), b"truncated").unwrap();
        assert!(!model_snapshot_matches(&source_files, &snapshot));
    }

    #[tokio::test]
    async fn reshoot_rejects_the_original_video_by_content() {
        let temporary = tempfile::tempdir().unwrap();
        let project = temporary.path().join("project");
        let stored = project.join("source/input.mov");
        let original = temporary.path().join("original.mov");
        let fresh = temporary.path().join("fresh.mov");
        tokio::fs::create_dir_all(stored.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&stored, b"same-video").await.unwrap();
        tokio::fs::write(&original, b"same-video").await.unwrap();
        tokio::fs::write(&fresh, b"new-video!").await.unwrap();

        assert!(
            ensure_reshoot_input_is_new(&project, &stored, &original, ReshootInputType::Video,)
                .await
                .is_err()
        );
        assert!(
            ensure_reshoot_input_is_new(&project, &stored, &fresh, ReshootInputType::Video,)
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn reshoot_rejects_the_original_image_sequence_by_content() {
        let temporary = tempfile::tempdir().unwrap();
        let project = temporary.path().join("project");
        let stored = project.join("source/images");
        let original = temporary.path().join("original-images");
        let fresh = temporary.path().join("fresh-images");
        let mixed = temporary.path().join("mixed-images");
        for directory in [&stored, &original, &fresh, &mixed] {
            tokio::fs::create_dir_all(directory).await.unwrap();
        }
        for (name, bytes) in [("1.jpg", b"one".as_slice()), ("2.png", b"two".as_slice())] {
            tokio::fs::write(stored.join(name), bytes).await.unwrap();
            tokio::fs::write(original.join(name), bytes).await.unwrap();
            tokio::fs::write(fresh.join(name), bytes).await.unwrap();
        }
        tokio::fs::write(fresh.join("1.jpg"), b"fresh-one")
            .await
            .unwrap();
        tokio::fs::write(fresh.join("2.png"), b"fresh-two")
            .await
            .unwrap();
        tokio::fs::write(mixed.join("1.jpg"), b"one").await.unwrap();
        tokio::fs::write(mixed.join("2.png"), b"fresh-two")
            .await
            .unwrap();

        assert!(ensure_reshoot_input_is_new(
            &project,
            &stored,
            &original,
            ReshootInputType::Images,
        )
        .await
        .is_err());
        assert!(
            ensure_reshoot_input_is_new(&project, &stored, &mixed, ReshootInputType::Images,)
                .await
                .is_err()
        );
        assert!(
            ensure_reshoot_input_is_new(&project, &stored, &fresh, ReshootInputType::Images,)
                .await
                .is_ok()
        );
    }

    #[test]
    fn parses_ffmpeg_progress() {
        assert_eq!(parse_ffmpeg_frame("frame=127"), Some(127));
        assert_eq!(parse_ffmpeg_frame("progress=continue"), None);
    }

    #[test]
    fn reports_the_latest_durable_checkpoint_instead_of_terminal_status() {
        let mut state = PipelineStateFile::created(Quality::Balanced);
        state.stage = PipelineStage::Cancelled;
        assert_eq!(checkpoint_stage(&state), PipelineStage::Created);

        state.frames = Some(FrameState {
            retention_ratio: 0.5,
            sampling_fps: 15.0,
            estimated_frames: 100,
            extracted_frames: Some(100),
            image_format: Some("jpeg".into()),
            mask_count: Some(0),
            has_alpha: false,
            ..FrameState::default()
        });
        state.features_complete = true;
        state.matching_complete = true;
        assert_eq!(checkpoint_stage(&state), PipelineStage::Matching);

        state.reconstruction_complete = true;
        state.brush_complete = true;
        assert_eq!(checkpoint_stage(&state), PipelineStage::TrainingSplats);
    }

    #[test]
    fn terminal_state_preserves_every_checkpoint() {
        let mut state = PipelineStateFile::created(Quality::Balanced);
        state.frames = Some(FrameState {
            retention_ratio: 0.5,
            sampling_fps: 15.0,
            estimated_frames: 100,
            extracted_frames: Some(100),
            image_format: Some("jpeg".into()),
            mask_count: Some(0),
            has_alpha: false,
            ..FrameState::default()
        });
        state.features_complete = true;
        state.matching_complete = true;
        state.reconstruction_complete = true;
        state.brush_complete = true;

        let failed = mark_state_terminal(state.clone(), false);
        let cancelled = mark_state_terminal(state, true);
        for terminal in [failed, cancelled] {
            assert_eq!(
                terminal.frames.as_ref().unwrap().extracted_frames,
                Some(100)
            );
            assert!(terminal.features_complete);
            assert!(terminal.matching_complete);
            assert!(terminal.reconstruction_complete);
            assert!(terminal.brush_complete);
        }
    }

    #[tokio::test]
    async fn frame_checkpoint_requires_every_recorded_frame() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = ProjectPaths::existing(uuid::Uuid::nil(), temporary.path().to_path_buf());
        tokio::fs::create_dir_all(&paths.frames).await.unwrap();
        tokio::fs::create_dir_all(&paths.colmap_frames)
            .await
            .unwrap();
        image::RgbImage::new(2, 2)
            .save(paths.frames.join("frame_000001.jpg"))
            .unwrap();
        image::RgbImage::new(2, 2)
            .save(paths.frames.join("frame_000002.jpg"))
            .unwrap();
        for name in ["frame_000001.jpg", "frame_000002.jpg"] {
            tokio::fs::copy(paths.frames.join(name), paths.colmap_frames.join(name))
                .await
                .unwrap();
        }
        let mut state = PipelineStateFile::created(Quality::Balanced);
        state.video = Some(VideoInfo {
            duration: 1.0,
            width: 1920,
            height: 1080,
            fps: 30.0,
            total_frames: 30,
            codec: "h264".into(),
            rotation: 0,
            pixel_format: "yuv420p".into(),
            has_alpha: false,
        });
        state.frames = Some(FrameState {
            retention_ratio: 0.5,
            sampling_fps: 15.0,
            estimated_frames: 2,
            extracted_frames: Some(2),
            image_format: Some("jpeg".into()),
            mask_count: Some(0),
            has_alpha: false,
            colmap_input_version: Some(COLMAP_INPUT_VERSION),
            ..FrameState::default()
        });

        assert!(prepared_frames_from_checkpoint(&paths, &state)
            .await
            .unwrap()
            .is_some());
        state.frames.as_mut().unwrap().colmap_input_version = Some(1);
        assert!(prepared_frames_from_checkpoint(&paths, &state)
            .await
            .unwrap()
            .is_none());
        state.frames.as_mut().unwrap().colmap_input_version = Some(COLMAP_INPUT_VERSION);
        tokio::fs::remove_file(paths.frames.join("frame_000002.jpg"))
            .await
            .unwrap();
        assert!(prepared_frames_from_checkpoint(&paths, &state)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn interrupted_publish_restores_a_valid_orphan_as_a_brush_checkpoint() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = ProjectPaths::existing(uuid::Uuid::nil(), temporary.path().to_path_buf());
        let valid = b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty float f_dc_0\nproperty float opacity\nproperty float scale_0\nproperty float rot_0\nend_header\n";
        tokio::fs::write(paths.project.join("final.ply"), valid)
            .await
            .unwrap();
        let mut state = PipelineStateFile::created(Quality::Balanced);
        state.brush_complete = true;

        recover_interrupted_publish(&paths, &state).await.unwrap();

        assert!(!paths.project.join("final.ply").exists());
        assert!(paths.brush.join("final.ply.tmp").is_file());
    }

    fn png_focal_checkpoint_fixture() -> crate::video::temporary_png_focal_workaround::Checkpoint {
        crate::video::temporary_png_focal_workaround::Checkpoint {
            version: 1,
            image_name: "frame_000001.png".into(),
            sha256: "fixture".into(),
            width: 8192,
            height: 5464,
            camera_params: [22728.66, 4096.0, 2732.0, 0.0],
        }
    }

    #[tokio::test]
    async fn temporary_png_focal_cache_barrier_preserves_images_and_published_files() {
        let temp = tempfile::tempdir().unwrap();
        let paths = ProjectPaths::existing(uuid::Uuid::nil(), temp.path().to_path_buf());
        let manager = ProjectManager::for_diagnostics(temp.path().to_path_buf());
        for dir in [
            &paths.frames,
            &paths.colmap_frames,
            &paths.masks,
            &paths.colmap,
            &paths.brush,
        ] {
            tokio::fs::create_dir_all(dir).await.unwrap();
            tokio::fs::write(dir.join("sentinel"), b"original")
                .await
                .unwrap();
        }
        for name in ["final.ply", "edit.ply", "editing.json"] {
            tokio::fs::write(paths.project.join(name), b"published")
                .await
                .unwrap();
        }
        let mut state = PipelineStateFile::created_for(Quality::Balanced, ProjectInputType::Images);
        state.frames = Some(FrameState {
            extracted_frames: Some(2),
            ..FrameState::default()
        });
        state.features_complete = true;
        state.matching_complete = true;
        state.reconstruction_complete = true;
        state.brush_complete = true;
        let checkpoint = png_focal_checkpoint_fixture();
        assert!(
            apply_temporary_png_focal_checkpoint(&manager, &paths, &mut state, &checkpoint)
                .await
                .unwrap()
        );
        let saved = manager.read_state(&paths.state).await.unwrap();
        assert!(
            !saved.features_complete
                && !saved.matching_complete
                && !saved.reconstruction_complete
                && !saved.brush_complete
        );
        assert_eq!(saved.temporary_png_focal.as_ref(), Some(&checkpoint));
        assert_eq!(saved.frames.unwrap().extracted_frames, Some(2));
        assert!(!paths.colmap.exists() && !paths.brush.exists());
        for dir in [&paths.frames, &paths.colmap_frames, &paths.masks] {
            assert_eq!(
                tokio::fs::read(dir.join("sentinel")).await.unwrap(),
                b"original"
            );
        }
        for name in ["final.ply", "edit.ply", "editing.json"] {
            assert_eq!(
                tokio::fs::read(paths.project.join(name)).await.unwrap(),
                b"published"
            );
        }
        let mut archives = tokio::fs::read_dir(paths.work.join("cache-archive"))
            .await
            .unwrap();
        let archive = archives.next_entry().await.unwrap().unwrap().path();
        assert!(archives.next_entry().await.unwrap().is_none());
        for dir in ["colmap", "brush"] {
            assert_eq!(
                tokio::fs::read(archive.join(dir).join("sentinel"))
                    .await
                    .unwrap(),
                b"original"
            );
        }
        // Identical policy/input/params is a true no-op, even with new cached features.
        tokio::fs::create_dir_all(&paths.colmap).await.unwrap();
        tokio::fs::write(paths.colmap.join("database.db"), b"new correct database")
            .await
            .unwrap();
        state.features_complete = true;
        assert!(
            !apply_temporary_png_focal_checkpoint(&manager, &paths, &mut state, &checkpoint)
                .await
                .unwrap()
        );
        assert!(state.features_complete && paths.colmap.join("database.db").exists());
        let mut changed = checkpoint.clone();
        changed.sha256 = "changed".into();
        assert!(
            apply_temporary_png_focal_checkpoint(&manager, &paths, &mut state, &changed)
                .await
                .unwrap()
        );
        assert!(!state.features_complete);
    }

    #[tokio::test]
    async fn temporary_png_focal_archive_failure_keeps_a_durable_reuse_barrier() {
        let temp = tempfile::tempdir().unwrap();
        let paths = ProjectPaths::existing(uuid::Uuid::nil(), temp.path().to_path_buf());
        let manager = ProjectManager::for_diagnostics(temp.path().to_path_buf());
        tokio::fs::create_dir_all(&paths.colmap).await.unwrap();
        tokio::fs::write(paths.colmap.join("database.db"), b"old wrong database")
            .await
            .unwrap();
        // A file where the archive directory should be forces a safe failure.
        tokio::fs::write(paths.work.join("cache-archive"), b"blocked")
            .await
            .unwrap();
        let mut state = PipelineStateFile::created_for(Quality::High, ProjectInputType::Images);
        state.features_complete = true;
        state.brush_complete = true;
        let checkpoint = png_focal_checkpoint_fixture();
        assert!(
            apply_temporary_png_focal_checkpoint(&manager, &paths, &mut state, &checkpoint)
                .await
                .is_err()
        );
        let mut saved = manager.read_state(&paths.state).await.unwrap();
        assert!(
            !saved.features_complete
                && !saved.brush_complete
                && saved.temporary_png_focal.is_none()
        );
        assert!(paths.colmap.join("database.db").exists());
        tokio::fs::remove_file(paths.work.join("cache-archive"))
            .await
            .unwrap();
        // Retrying after the interruption cannot certify the old database.
        apply_temporary_png_focal_checkpoint(&manager, &paths, &mut saved, &checkpoint)
            .await
            .unwrap();
        assert!(!paths.colmap.join("database.db").exists());
        assert_eq!(saved.temporary_png_focal, Some(checkpoint));
    }

    #[tokio::test]
    async fn legacy_colmap_input_cache_is_archived_without_touching_published_output() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = ProjectPaths::existing(uuid::Uuid::nil(), temporary.path().to_path_buf());
        tokio::fs::create_dir_all(&paths.work).await.unwrap();
        for directory in [
            &paths.frames,
            &paths.colmap_frames,
            &paths.masks,
            &paths.colmap,
            &paths.brush,
        ] {
            tokio::fs::create_dir_all(directory).await.unwrap();
            tokio::fs::write(directory.join("sentinel"), b"legacy")
                .await
                .unwrap();
        }
        tokio::fs::write(paths.project.join("final.ply"), b"published")
            .await
            .unwrap();
        let mut state = PipelineStateFile::created(Quality::Balanced);
        state.frames = Some(FrameState {
            extracted_frames: Some(2),
            colmap_input_version: Some(1),
            ..FrameState::default()
        });

        archive_incompatible_colmap_input_cache(&paths, &state)
            .await
            .unwrap();

        for directory in [
            &paths.frames,
            &paths.colmap_frames,
            &paths.masks,
            &paths.colmap,
            &paths.brush,
        ] {
            assert!(!directory.exists());
        }
        assert_eq!(
            tokio::fs::read(paths.project.join("final.ply"))
                .await
                .unwrap(),
            b"published"
        );
        let mut archives = tokio::fs::read_dir(paths.work.join("cache-archive"))
            .await
            .unwrap();
        let archive = archives.next_entry().await.unwrap().unwrap().path();
        assert!(archives.next_entry().await.unwrap().is_none());
        for name in ["frames", "colmap_frames", "masks", "colmap", "brush"] {
            assert_eq!(
                tokio::fs::read(archive.join(name).join("sentinel"))
                    .await
                    .unwrap(),
                b"legacy"
            );
        }
    }

    #[tokio::test]
    async fn reshoot_baseline_requires_a_complete_current_colmap_image_snapshot() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = ProjectPaths::existing(uuid::Uuid::nil(), temporary.path().to_path_buf());
        tokio::fs::create_dir_all(&paths.frames).await.unwrap();
        tokio::fs::create_dir_all(&paths.colmap_frames)
            .await
            .unwrap();
        for name in ["frame_000001.png", "frame_000002.png"] {
            image::RgbaImage::from_pixel(2, 2, image::Rgba([1, 2, 3, 255]))
                .save(paths.frames.join(name))
                .unwrap();
            tokio::fs::copy(paths.frames.join(name), paths.colmap_frames.join(name))
                .await
                .unwrap();
        }
        let mut state = PipelineStateFile::created_for(Quality::Balanced, ProjectInputType::Images);
        state.frames = Some(FrameState {
            extracted_frames: Some(2),
            has_alpha: false,
            colmap_input_version: Some(COLMAP_INPUT_VERSION),
            ..FrameState::default()
        });

        state.frames.as_mut().unwrap().colmap_input_version = Some(1);
        assert!(
            validate_current_colmap_input_snapshot(&paths.project, &state)
                .await
                .is_err()
        );
        state.frames.as_mut().unwrap().colmap_input_version = Some(COLMAP_INPUT_VERSION);
        validate_current_colmap_input_snapshot(&paths.project, &state)
            .await
            .unwrap();
        tokio::fs::remove_file(paths.colmap_frames.join("frame_000002.png"))
            .await
            .unwrap();
        assert!(
            validate_current_colmap_input_snapshot(&paths.project, &state)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn reshoot_three_way_merge_preflights_and_moves_every_image_set() {
        let temporary = tempfile::tempdir().unwrap();
        let extracted_frames = temporary.path().join("extracted-frames");
        let extracted_colmap = temporary.path().join("extracted-colmap");
        let extracted_masks = temporary.path().join("extracted-masks");
        let frames = temporary.path().join("frames");
        let colmap_frames = temporary.path().join("colmap-frames");
        let masks = temporary.path().join("masks");
        for directory in [
            &extracted_frames,
            &extracted_colmap,
            &extracted_masks,
            &frames,
            &colmap_frames,
            &masks,
        ] {
            tokio::fs::create_dir_all(directory).await.unwrap();
        }
        for index in 1..=2 {
            let name = format!("frame_{index:06}.png");
            image::RgbaImage::new(2, 2)
                .save(extracted_frames.join(&name))
                .unwrap();
            tokio::fs::copy(extracted_frames.join(&name), extracted_colmap.join(&name))
                .await
                .unwrap();
            image::GrayImage::new(2, 2)
                .save(extracted_masks.join(format!("{name}.png")))
                .unwrap();
        }
        tokio::fs::write(frames.join("reshoot_000002.png"), b"collision")
            .await
            .unwrap();

        assert!(move_extracted_reshoot(
            &extracted_frames,
            &extracted_colmap,
            &extracted_masks,
            &frames,
            &colmap_frames,
            &masks,
            true,
        )
        .await
        .is_err());
        assert!(extracted_frames.join("frame_000001.png").is_file());
        assert!(!frames.join("reshoot_000001.png").exists());

        tokio::fs::remove_file(frames.join("reshoot_000002.png"))
            .await
            .unwrap();
        move_extracted_reshoot(
            &extracted_frames,
            &extracted_colmap,
            &extracted_masks,
            &frames,
            &colmap_frames,
            &masks,
            true,
        )
        .await
        .unwrap();
        for index in 1..=2 {
            let name = format!("reshoot_{index:06}.png");
            assert!(frames.join(&name).is_file());
            assert!(colmap_frames.join(&name).is_file());
            assert!(masks.join(format!("{name}.png")).is_file());
            assert_eq!(
                tokio::fs::read(frames.join(&name)).await.unwrap(),
                tokio::fs::read(colmap_frames.join(&name)).await.unwrap()
            );
        }
    }

    #[tokio::test]
    async fn image_sequence_checkpoint_requires_every_image_and_mask() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = ProjectPaths::existing(uuid::Uuid::nil(), temporary.path().to_path_buf());
        tokio::fs::create_dir_all(&paths.frames).await.unwrap();
        tokio::fs::create_dir_all(&paths.colmap_frames)
            .await
            .unwrap();
        tokio::fs::create_dir_all(&paths.masks).await.unwrap();
        for name in ["frame_000001.png", "frame_000002.jpg"] {
            if name.ends_with(".png") {
                image::RgbaImage::new(2, 2)
                    .save(paths.frames.join(name))
                    .unwrap();
            } else {
                image::RgbImage::new(2, 2)
                    .save(paths.frames.join(name))
                    .unwrap();
            }
            tokio::fs::copy(paths.frames.join(name), paths.colmap_frames.join(name))
                .await
                .unwrap();
            image::GrayImage::new(2, 2)
                .save(paths.masks.join(format!("{name}.png")))
                .unwrap();
        }
        let mut state = PipelineStateFile::created_for(Quality::Balanced, ProjectInputType::Images);
        state.image_sequence = Some(ImageSequenceInfo {
            image_count: 2,
            width: 1920,
            height: 1080,
            has_alpha: true,
            requires_large_sequence_confirmation: false,
        });
        state.frames = Some(FrameState {
            retention_ratio: 1.0,
            sampling_fps: 0.0,
            estimated_frames: 2,
            extracted_frames: Some(2),
            image_format: Some("images".into()),
            mask_count: Some(2),
            has_alpha: true,
            colmap_input_version: Some(COLMAP_INPUT_VERSION),
            ..FrameState::default()
        });

        assert!(prepared_frames_from_checkpoint(&paths, &state)
            .await
            .unwrap()
            .is_some());
        tokio::fs::remove_file(paths.masks.join("frame_000002.jpg.png"))
            .await
            .unwrap();
        assert!(prepared_frames_from_checkpoint(&paths, &state)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn opaque_alpha_channel_sequence_reuses_a_maskless_checkpoint() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = ProjectPaths::existing(uuid::Uuid::nil(), temporary.path().to_path_buf());
        tokio::fs::create_dir_all(&paths.frames).await.unwrap();
        tokio::fs::create_dir_all(&paths.colmap_frames)
            .await
            .unwrap();
        for name in ["frame_000001.png", "frame_000002.png"] {
            image::RgbaImage::from_pixel(2, 2, image::Rgba([1, 2, 3, 255]))
                .save(paths.frames.join(name))
                .unwrap();
            tokio::fs::copy(paths.frames.join(name), paths.colmap_frames.join(name))
                .await
                .unwrap();
        }
        let mut state = PipelineStateFile::created_for(Quality::Balanced, ProjectInputType::Images);
        state.image_sequence = Some(ImageSequenceInfo {
            image_count: 2,
            width: 1920,
            height: 1080,
            has_alpha: true,
            requires_large_sequence_confirmation: false,
        });
        state.frames = Some(FrameState {
            retention_ratio: 1.0,
            sampling_fps: 0.0,
            estimated_frames: 2,
            extracted_frames: Some(2),
            image_format: Some("images".into()),
            mask_count: Some(0),
            has_alpha: false,
            colmap_input_version: Some(COLMAP_INPUT_VERSION),
            ..FrameState::default()
        });

        let prepared = prepared_frames_from_checkpoint(&paths, &state)
            .await
            .unwrap()
            .expect("opaque alpha-channel images should reuse the maskless checkpoint");

        assert!(!prepared.has_alpha);
        assert_eq!(prepared.mask_count, 0);
    }

    #[tokio::test]
    async fn transparent_frame_checkpoint_requires_matching_masks() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = ProjectPaths::existing(uuid::Uuid::nil(), temporary.path().to_path_buf());
        tokio::fs::create_dir_all(&paths.frames).await.unwrap();
        tokio::fs::create_dir_all(&paths.colmap_frames)
            .await
            .unwrap();
        tokio::fs::create_dir_all(&paths.masks).await.unwrap();
        image::RgbaImage::new(2, 2)
            .save(paths.frames.join("frame_000001.png"))
            .unwrap();
        tokio::fs::copy(
            paths.frames.join("frame_000001.png"),
            paths.colmap_frames.join("frame_000001.png"),
        )
        .await
        .unwrap();
        image::GrayImage::new(2, 2)
            .save(paths.masks.join("frame_000001.png.png"))
            .unwrap();
        let mut state = PipelineStateFile::created(Quality::Balanced);
        state.video = Some(VideoInfo {
            duration: 1.0,
            width: 1920,
            height: 1080,
            fps: 30.0,
            total_frames: 30,
            codec: "prores".into(),
            rotation: 0,
            pixel_format: "yuva444p10le".into(),
            has_alpha: true,
        });
        state.frames = Some(FrameState {
            retention_ratio: 0.5,
            sampling_fps: 15.0,
            estimated_frames: 1,
            extracted_frames: Some(1),
            image_format: Some("png".into()),
            mask_count: Some(1),
            has_alpha: true,
            colmap_input_version: Some(COLMAP_INPUT_VERSION),
            ..FrameState::default()
        });

        let prepared = prepared_frames_from_checkpoint(&paths, &state)
            .await
            .unwrap()
            .unwrap();
        assert!(prepared.has_alpha);
        assert_eq!(prepared.mask_count, 1);

        tokio::fs::remove_file(paths.masks.join("frame_000001.png.png"))
            .await
            .unwrap();
        assert!(prepared_frames_from_checkpoint(&paths, &state)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn empty_colmap_database_downgrades_the_feature_checkpoint() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = ProjectPaths::existing(uuid::Uuid::nil(), temporary.path().to_path_buf());
        tokio::fs::create_dir_all(&paths.frames).await.unwrap();
        tokio::fs::create_dir_all(&paths.colmap_frames)
            .await
            .unwrap();
        tokio::fs::create_dir_all(&paths.colmap).await.unwrap();
        image::RgbImage::new(2, 2)
            .save(paths.frames.join("frame_000001.jpg"))
            .unwrap();
        tokio::fs::copy(
            paths.frames.join("frame_000001.jpg"),
            paths.colmap_frames.join("frame_000001.jpg"),
        )
        .await
        .unwrap();
        tokio::fs::write(paths.colmap.join("database.db"), b"")
            .await
            .unwrap();
        let mut state = PipelineStateFile::created(Quality::Balanced);
        state.video = Some(VideoInfo {
            duration: 1.0,
            width: 1920,
            height: 1080,
            fps: 30.0,
            total_frames: 30,
            codec: "h264".into(),
            rotation: 0,
            pixel_format: "yuv420p".into(),
            has_alpha: false,
        });
        state.frames = Some(FrameState {
            retention_ratio: 0.5,
            sampling_fps: 15.0,
            estimated_frames: 1,
            extracted_frames: Some(1),
            image_format: Some("jpeg".into()),
            mask_count: Some(0),
            has_alpha: false,
            colmap_input_version: Some(COLMAP_INPUT_VERSION),
            ..FrameState::default()
        });
        state.features_complete = true;
        state.matching_complete = true;

        normalize_checkpoints(&paths, &mut state).await.unwrap();

        assert!(!state.features_complete);
        assert!(!state.matching_complete);
        assert_eq!(state.stage, PipelineStage::ExtractingFrames);
    }

    #[test]
    fn parses_colmap_file_progress() {
        assert_eq!(
            parse_bracket_progress("Processed file [23/533]"),
            Some((23, 533))
        );
        assert_eq!(
            parse_bracket_progress("Processed image [4/10]"),
            Some((4, 10))
        );
        assert_eq!(parse_bracket_progress("Processing image [4/10]"), None);
        assert_eq!(parse_bracket_progress("Matching block [4/10]"), None);
        assert_eq!(parse_bracket_progress("bundle iteration [4/10]"), None);
    }

    #[test]
    fn matching_silence_notice_has_a_15_second_threshold_and_30_second_limit() {
        let last = AtomicU64::new(0);
        assert!(!matching_quiet_notice_due(&last, 14_999, 14_999));
        assert!(matching_quiet_notice_due(&last, 15_000, 15_000));
        assert!(!matching_quiet_notice_due(&last, 15_000, 15_000));
        assert!(!matching_quiet_notice_due(&last, 44_999, 44_999));
        assert!(matching_quiet_notice_due(&last, 45_000, 45_000));
        assert!(!matching_quiet_notice_due(&last, 99_000, 100));
        last.store(0, Ordering::Relaxed); // An actual output line or fresh process resets silence.
        assert!(!matching_quiet_notice_due(&last, 100_000, 14_999));
        assert!(matching_quiet_notice_due(&last, 100_001, 15_000));
    }

    #[test]
    fn matching_notices_are_system_logs_without_progress_and_stop_on_exit_or_cancel() {
        let received = Arc::new(std::sync::Mutex::new(Vec::<PipelineEvent>::new()));
        let out = received.clone();
        let runner = PipelineRunner::new(default_engine_paths(None), move |e| {
            out.lock().unwrap().push(e)
        });
        let cancelled = Arc::new(std::sync::Mutex::new(Vec::<PipelineEvent>::new()));
        let out = cancelled.clone();
        let cancelled_runner = PipelineRunner::new(default_engine_paths(None), move |e| {
            out.lock().unwrap().push(e)
        });
        let observer = runner.process_observer(
            PipelineStage::Matching,
            PipelineComponent::Reconstruction,
            Some(86),
            ObserverMode::Matching,
        );
        let cancelled_observer = cancelled_runner.process_observer(
            PipelineStage::Matching,
            PipelineComponent::Reconstruction,
            Some(86),
            ObserverMode::Matching,
        );
        observer(ProcessUpdate::Started { process_id: 42 });
        cancelled_observer(ProcessUpdate::Started { process_id: 43 });
        let initial_progress = received.lock().unwrap().last().unwrap().progress;
        // One real quiet window tests the production Instant-based clock, not a mocked age.
        std::thread::sleep(std::time::Duration::from_millis(15_025));
        observer(ProcessUpdate::Heartbeat { elapsed_ms: 15_025 });
        observer(ProcessUpdate::Heartbeat { elapsed_ms: 16_000 });
        observer(ProcessUpdate::Heartbeat { elapsed_ms: 45_025 });
        let notices = || {
            received
                .lock()
                .unwrap()
                .iter()
                .filter(|e| e.message.starts_with("图像匹配进程仍在运行"))
                .count()
        };
        assert_eq!(notices(), 2);
        {
            let events = received.lock().unwrap();
            for event in events
                .iter()
                .filter(|e| e.message.starts_with("图像匹配进程仍在运行"))
            {
                assert_eq!(event.kind, EventKind::Log);
                assert_eq!(event.component, Some(PipelineComponent::System));
                assert_eq!(event.progress, initial_progress);
                assert!(event.stage_progress.is_none());
                assert!(event.current.is_none() && event.total.is_none());
            }
        }
        observer(ProcessUpdate::Exited {
            executable: PathBuf::from("colmap"),
            exit_code: Some(0),
        });
        observer(ProcessUpdate::Heartbeat { elapsed_ms: 75_025 });
        assert_eq!(notices(), 2);
        cancelled_runner.process_manager.cancel();
        cancelled_observer(ProcessUpdate::Heartbeat { elapsed_ms: 15_025 });
        assert!(!cancelled
            .lock()
            .unwrap()
            .iter()
            .any(|e| e.message.starts_with("图像匹配进程仍在运行")));
        observer(ProcessUpdate::Line {
            stream: crate::process::ProcessStream::Stdout,
            line: "Indexing image [1/86]".into(),
        });
        observer(ProcessUpdate::Heartbeat { elapsed_ms: 90_000 });
        assert_eq!(notices(), 2);
    }

    #[test]
    #[ignore = "Requires OOOSPLAT_MATCHING_LOG, an existing real COLMAP matching log"]
    fn replay_real_matching_log_keeps_index_counts_and_unknown_stage_percentage() {
        let path = std::env::var_os("OOOSPLAT_MATCHING_LOG").expect("Set a real COLMAP log");
        let source = std::fs::read_to_string(path).unwrap();
        let captured = Arc::new(std::sync::Mutex::new(Vec::<PipelineEvent>::new()));
        let out = captured.clone();
        let runner = PipelineRunner::new(default_engine_paths(None), move |event| {
            out.lock().unwrap().push(event)
        });
        let observer = runner.process_observer(
            PipelineStage::Matching,
            PipelineComponent::Reconstruction,
            None,
            ObserverMode::Matching,
        );
        observer(ProcessUpdate::Started { process_id: 42 });
        let mut expected_index_lines = 0;
        for line in source.lines().filter(|line| {
            parse_matching_progress(line).is_some() || matching_activity(line).is_some()
        }) {
            expected_index_lines += usize::from(line.contains("Indexing image ["));
            observer(ProcessUpdate::Line {
                stream: crate::process::ProcessStream::Stderr,
                line: line.into(),
            });
        }
        let captured = captured.lock().unwrap();
        let indices: Vec<_> = captured
            .iter()
            .filter(|e| e.unit.as_deref() == Some("matchingIndexImages"))
            .collect();
        assert!(expected_index_lines > 0);
        assert_eq!(indices.len(), expected_index_lines);
        assert!(indices
            .iter()
            .all(|e| e.current.is_some() && e.stage_progress.is_none() && e.indeterminate));
        println!(
            "Real matching replay: {} indexing lines retained; last count {:?}/{:?}",
            indices.len(),
            indices.last().unwrap().current,
            indices.last().unwrap().total
        );
        if let Some(path) = std::env::var_os("OOOSPLAT_MATCHING_EVENTS") {
            std::fs::write(path, serde_json::to_vec(&*captured).unwrap()).unwrap();
        }
    }

    #[test]
    fn panorama_process_context_is_stable_and_matching_counts_do_not_estimate_completion() {
        let captured = Arc::new(std::sync::Mutex::new(Vec::<PipelineEvent>::new()));
        let output = captured.clone();
        let runner = PipelineRunner::new(default_engine_paths(None), move |event| {
            output.lock().unwrap().push(event)
        });
        runner.events.panorama_phase.store(1, Ordering::Relaxed);
        let old = runner.process_observer(
            PipelineStage::Reconstructing,
            PipelineComponent::Reconstruction,
            Some(100),
            ObserverMode::Mapper,
        );
        old(ProcessUpdate::Started { process_id: 42 });
        old(ProcessUpdate::Line {
            stream: crate::process::ProcessStream::Stdout,
            line: "Registering image #50 (num_reg_frames=40)".into(),
        });
        old(ProcessUpdate::Heartbeat { elapsed_ms: 1000 });
        {
            let events = captured.lock().unwrap();
            assert!(events
                .iter()
                .all(|event| event.substage.as_deref() == Some("panoramaReconstruction")));
            let progress = events
                .iter()
                .find(|event| event.kind == EventKind::Progress)
                .unwrap();
            assert!(progress.indeterminate && progress.stage_progress.is_none());
            assert_eq!(progress.current, Some(40));
        }
        let retry = runner.process_observer(
            PipelineStage::Reconstructing,
            PipelineComponent::Reconstruction,
            Some(100),
            ObserverMode::Mapper,
        );
        retry(ProcessUpdate::Started { process_id: 43 });
        let count = captured.lock().unwrap().len();
        old(ProcessUpdate::Heartbeat { elapsed_ms: 2000 });
        assert_eq!(captured.lock().unwrap().len(), count);
        assert_eq!(
            captured
                .lock()
                .unwrap()
                .last()
                .unwrap()
                .operation
                .as_ref()
                .unwrap()
                .attempt,
            2
        );
        let matching = runner.process_observer(
            PipelineStage::Matching,
            PipelineComponent::Reconstruction,
            Some(100),
            ObserverMode::Matching,
        );
        matching(ProcessUpdate::Started { process_id: 44 });
        for line in [
            "Generating sequential image pairs...",
            "Generating image pairs with vocabulary tree...",
            "Processing image [86/86]",
            "Processing image [1/9]",
        ] {
            matching(ProcessUpdate::Line {
                stream: crate::process::ProcessStream::Stdout,
                line: line.into(),
            });
        }
        let events = captured.lock().unwrap();
        let progress: Vec<_> = events
            .iter()
            .filter(|event| {
                event.stage == PipelineStage::Matching && event.kind == EventKind::Progress
            })
            .collect();
        assert_eq!(progress.len(), 4);
        assert!(progress
            .iter()
            .all(|event| event.stage_progress.is_none() && event.indeterminate));
        assert_eq!(progress[0].unit.as_deref(), Some("matchingSequential"));
        assert_eq!(progress[1].unit.as_deref(), Some("matchingLoop"));
        assert_eq!(progress[3].current, Some(1));
    }

    #[test]
    fn parses_mapper_registration() {
        let counter = AtomicU64::new(0);
        let value = parse_mapper_progress(
            "Registering image #90 (num_reg_frames=86)",
            &counter,
            Some(100),
        )
        .unwrap();
        assert_eq!(value.0, 86);
        assert_eq!(counter.load(Ordering::Relaxed), 86);
    }

    #[test]
    fn mapper_refinement_keeps_the_latest_registered_count() {
        let counter = AtomicU64::new(0);
        parse_mapper_progress(
            "Registering image #90 (num_reg_frames=86)",
            &counter,
            Some(100),
        )
        .unwrap();

        let retriangulation = parse_mapper_progress(
            "Retriangulation and Global bundle adjustment",
            &counter,
            Some(100),
        )
        .unwrap();
        assert_eq!(retriangulation.0, 86);
        assert_eq!(retriangulation.1, Some(100));
        assert_eq!(
            retriangulation.2,
            "Retriangulation and Global bundle adjustment"
        );

        let bundle_adjustment =
            parse_mapper_progress("Global bundle adjustment", &counter, Some(100)).unwrap();
        assert_eq!(bundle_adjustment.0, 86);
        assert_eq!(bundle_adjustment.1, Some(100));
    }

    #[test]
    fn mapper_refinement_without_a_registration_count_stays_indeterminate() {
        let counter = AtomicU64::new(0);
        assert!(parse_mapper_progress(
            "Retriangulation and Global bundle adjustment",
            &counter,
            Some(100),
        )
        .is_none());
    }

    #[test]
    fn mapper_registration_uses_the_current_model_count_not_an_accumulated_maximum() {
        let counter = AtomicU64::new(0);
        parse_mapper_progress("num_reg_frames=86", &counter, Some(100)).unwrap();
        parse_mapper_progress("num_reg_frames=91", &counter, Some(100)).unwrap();
        parse_mapper_progress("num_reg_frames=89", &counter, Some(100)).unwrap();

        let value = parse_mapper_progress(
            "Retriangulation and Global bundle adjustment",
            &counter,
            Some(100),
        )
        .unwrap();
        assert_eq!(value.0, 89);
        assert!(parse_mapper_progress("Registering image #5", &counter, Some(100)).is_none());
    }

    #[test]
    fn brush_heartbeat_never_invents_completed_steps() {
        let received = Arc::new(std::sync::Mutex::new(Vec::new()));
        let output = received.clone();
        let runner = PipelineRunner::new(
            EnginePaths::from_root(PathBuf::from("engines")),
            move |event| output.lock().unwrap().push(event),
        );
        let observer = runner.process_observer(
            PipelineStage::TrainingSplats,
            PipelineComponent::Training,
            Some(100),
            ObserverMode::Brush,
        );
        observer(ProcessUpdate::Started { process_id: 42 });
        observer(ProcessUpdate::Heartbeat {
            elapsed_ms: 500_000,
        });
        let events = received.lock().unwrap();
        assert!(events.iter().all(|event| event.current.is_none()));
        assert!(events
            .iter()
            .filter_map(|event| event.runtime.as_ref())
            .all(|runtime| runtime.training.is_none()));
    }

    #[test]
    fn brush_progress_and_retry_use_actual_output_and_fresh_state() {
        let received = Arc::new(std::sync::Mutex::new(Vec::new()));
        let output = received.clone();
        let runner = PipelineRunner::new(
            EnginePaths::from_root(PathBuf::from("engines")),
            move |event| output.lock().unwrap().push(event),
        );
        let observer = runner.process_observer(
            PipelineStage::TrainingSplats,
            PipelineComponent::Training,
            Some(100),
            ObserverMode::Brush,
        );
        observer(ProcessUpdate::Started { process_id: 42 });
        for line in [
            "Training config: total_iters=120 start_iter=20",
            "Training progress: iteration=60 total=120 elapsed_secs=1 lod=0",
            "[INFO brush_process] Export started: iteration 60, final.ply",
        ] {
            observer(ProcessUpdate::Line {
                stream: crate::process::ProcessStream::Stdout,
                line: line.into(),
            });
        }
        {
            let events = received.lock().unwrap();
            let progress = events
                .iter()
                .find(|event| event.kind == EventKind::Progress)
                .unwrap();
            assert_eq!(progress.current, Some(60));
            assert_eq!(progress.total, Some(120));
            assert!(progress.progress < 100.0);
            assert_eq!(
                events
                    .iter()
                    .filter_map(|event| event.runtime.as_ref())
                    .next_back()
                    .unwrap()
                    .phase,
                "exporting"
            );
        }

        let retry = runner.process_observer(
            PipelineStage::TrainingSplats,
            PipelineComponent::Training,
            Some(100),
            ObserverMode::Brush,
        );
        retry(ProcessUpdate::Started { process_id: 43 });
        let events = received.lock().unwrap();
        let snapshot = events
            .iter()
            .filter_map(|event| event.runtime.as_ref())
            .next_back()
            .unwrap();
        assert_eq!(snapshot.process_id, 43);
        assert!(snapshot.training.is_none());
        assert!(snapshot.resources.is_none());
    }

    #[test]
    fn brush_oom_fallback_is_high_only_and_can_be_used_once() {
        let resolved =
            resolve_brush_training_preset(Quality::High, true, Some(12_288), 7_680, 60_000);
        let fallback = brush_oom_fallback(Quality::High, true, false, resolved, 60_000).unwrap();
        assert_eq!(
            fallback.profile,
            crate::presets::BrushTrainingProfile::HighStandard
        );
        assert!(brush_oom_fallback(Quality::High, true, true, fallback, 60_000).is_none());
        assert!(brush_oom_fallback(Quality::Balanced, true, false, resolved, 60_000).is_none());
        assert!(brush_oom_fallback(Quality::High, false, false, resolved, 60_000).is_none());
    }

    #[test]
    fn panorama_oom_retry_is_all_quality_but_still_once_and_optimization_only() {
        for quality in [Quality::Fast, Quality::Balanced, Quality::High] {
            let r =
                resolve_panorama_training_preset(quality, true, Some(8192), 512, 10_192).unwrap();
            let retry = brush_oom_fallback(quality, true, false, r, 10_192).unwrap();
            assert_eq!(retry.preset.max_splats, r.preset.max_splats.map(|n| n / 2));
            assert!(brush_oom_fallback(quality, true, true, retry, 10_192).is_none());
            assert!(brush_oom_fallback(quality, false, false, r, 10_192).is_none());
            let fixed =
                resolve_panorama_training_preset(quality, false, Some(8192), 512, 10_192).unwrap();
            assert!(brush_oom_fallback(quality, true, false, fixed, 10_192).is_none());
        }
    }

    #[test]
    fn event_sequence_is_strictly_increasing() {
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured = events.clone();
        let sink = EventSink {
            context: Default::default(),
            bound_operation: None,
            panorama_phase: Arc::new(AtomicU64::new(0)),
            panorama_images: Arc::new(AtomicBool::new(false)),
            emit: Arc::new(move |event| captured.lock().unwrap().push(event.sequence)),
            sequence: Arc::new(AtomicU64::new(0)),
            last_progress_milli_percent: Arc::new(AtomicU64::new(0)),
            last_stage: Arc::new(std::sync::Mutex::new(None)),
            dispatch: Arc::new(std::sync::Mutex::new(())),
            started: Instant::now(),
        };
        sink.stage(PipelineStage::Created, 0.0, "created");
        sink.stage(PipelineStage::ProbingVideo, 0.0, "probing");
        assert_eq!(*events.lock().unwrap(), vec![1, 2]);
    }

    #[test]
    fn terminal_event_keeps_the_last_real_progress_and_clears_stage_progress() {
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured = events.clone();
        let sink = EventSink {
            panorama_phase: Arc::new(AtomicU64::new(0)),
            panorama_images: Arc::new(AtomicBool::new(false)),
            context: Default::default(),
            bound_operation: None,
            emit: Arc::new(move |event| captured.lock().unwrap().push(event)),
            sequence: Arc::new(AtomicU64::new(0)),
            last_progress_milli_percent: Arc::new(AtomicU64::new(0)),
            last_stage: Arc::new(std::sync::Mutex::new(None)),
            dispatch: Arc::new(std::sync::Mutex::new(())),
            started: Instant::now(),
        };
        sink.stage(PipelineStage::TrainingSplats, 0.5, "training");
        sink.terminal(&SplatError::Process("boom".into()));

        let events = events.lock().unwrap();
        assert_eq!(events[0].progress, 79.0);
        assert_eq!(events[1].progress, 79.0);
        assert_eq!(events[1].stage_progress, None);
        assert_eq!(events[1].stage, PipelineStage::Failed);
        assert_eq!(
            *sink.last_stage.lock().unwrap(),
            Some(PipelineStage::TrainingSplats)
        );
    }

    #[test]
    fn panorama_preparation_and_training_use_distinct_progress_ranges() {
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured = events.clone();
        let sink = EventSink {
            context: Default::default(),
            bound_operation: None,
            panorama_phase: Arc::new(AtomicU64::new(2)),
            panorama_images: Arc::new(AtomicBool::new(false)),
            emit: Arc::new(move |event| captured.lock().unwrap().push(event)),
            sequence: Arc::new(AtomicU64::new(0)),
            last_progress_milli_percent: Arc::new(AtomicU64::new(0)),
            last_stage: Arc::new(std::sync::Mutex::new(None)),
            dispatch: Arc::new(std::sync::Mutex::new(())),
            started: Instant::now(),
        };
        sink.stage(PipelineStage::TrainingSplats, 0.5, "faces");
        sink.panorama_phase.store(3, Ordering::Relaxed);
        sink.stage(PipelineStage::TrainingSplats, 1.0, "validated");
        sink.panorama_phase.store(1, Ordering::Relaxed);
        sink.stage(PipelineStage::TrainingSplats, 0.0, "training");
        sink.stage(PipelineStage::TrainingSplats, 0.5, "training");
        sink.stage(
            PipelineStage::TrainingSplats,
            0.1,
            "late or restarted progress",
        );
        sink.stage(
            PipelineStage::TrainingSplats,
            0.513123,
            "fractional progress",
        );
        sink.send(
            PipelineStage::TrainingSplats,
            Some(PipelineComponent::Training),
            EventKind::Log,
            EventLevel::Info,
            None,
            true,
            "log without progress",
            None,
            None,
            None,
        );
        sink.stage(PipelineStage::Exporting, 1.0, "published");
        sink.stage(PipelineStage::Completed, 1.0, "completed");
        let events = events.lock().unwrap();
        assert_eq!(
            events
                .iter()
                .map(|event| event.progress)
                .collect::<Vec<_>>(),
            [63.0, 66.0, 66.0, 82.0, 82.0, 82.42, 82.42, 99.9, 100.0]
        );
        assert_eq!(
            events[0].substage.as_deref(),
            Some("panoramaTrainingPreparation")
        );
        assert_eq!(
            events[1].substage.as_deref(),
            Some("panoramaTrainingValidation")
        );
        assert_eq!(events[2].substage.as_deref(), Some("training"));
        assert!((events[1].stage_progress.unwrap() - 600.0 / 38.0).abs() < 0.001);
        assert_eq!(events[2].stage_progress, events[1].stage_progress);
    }
}
