export type Quality = "fast" | "balanced" | "high";
/**
 * 承担计算的组件类别（**中性**命名）。
 *
 * 刻意不暴露"colmap"/"brush"这类实现名：后端换了（外部 CLI ↔ 自研进程内），
 * 前端契约一字不改。需要知道具体是谁时，读 `BackendIdentity.displayName`。
 */
export type ComponentKind = "media" | "reconstruction" | "training";
export type RunPhase = "idle" | "analyzing" | "running" | "completed" | "failed" | "cancelled";
export type ProjectStatus = "running" | "completed" | "failed" | "cancelled" | "interrupted";

export type AccelerationMode = "cpu" | "gpu";
export type AccelerationState = "ready" | "temporarilyUnavailable" | "unavailable";
/** 加速原因码（**中性**：去掉 CUDA/NVIDIA 专属语义，原文案走 `reason`）。 */
export type AccelerationReasonCode =
  | "gpuReady" | "cpuOnly" | "componentUnavailable" | "requirementsUnavailable"
  | "probeFailed" | "probeTimeout" | "noAccelerator"
  | "driverVersionUnknown" | "driverTooOld" | "computeCapabilityUnknown"
  | "computeCapabilityTooLow";
export interface AcceleratorDevice { index: number; name: string; driverVersion: string; computeCapability: string; totalMemoryMb?: number; }
export interface AcceleratorRequirements { minimumDriverVersion: string; minimumComputeCapability: string; }
export interface AccelerationStatus {
  mode: AccelerationMode;
  state: AccelerationState;
  reasonCode: AccelerationReasonCode;
  reason: string;
  device: AcceleratorDevice | null;
  requirements: AcceleratorRequirements;
  detectedDeviceCount?: number;
}
export interface ComponentStatus {
  kind: ComponentKind;
  ready: boolean;
  acceleration: AccelerationStatus | null;
  detail: string;
  location?: string;
}
/** 重建后端身份（`id` 稳定、`displayName` 直接展示）。 */
export interface BackendIdentity { id: string; displayName: string; version: string; family: "externalCli" | "inProcess"; }
export interface VideoInfo {
  duration: number;
  width: number;
  height: number;
  fps: number;
  totalFrames: number;
  codec: string;
  rotation: number;
  pixelFormat: string;
  hasAlpha: boolean;
}
export type InputType = "video" | "images";
export interface ImageSequenceInfo {
  imageCount: number;
  width: number;
  height: number;
  hasAlpha: boolean;
  requiresLargeSequenceConfirmation: boolean;
}
export interface PlannedFrame { sourceFrameIndex: number; timestampSeconds: number; }
export interface FramePlan {
  retentionRatio: number;
  samplingFps: number;
  estimatedFrames: number;
  selectedFrames?: PlannedFrame[];
  candidateFrames?: PlannedFrame[];
  rescueMaxFrames?: number;
  minimumFrameOverrideApplied?: boolean;
}
export interface RuntimeEstimate {
  estimatedMs: number;
  lowerBoundMs: number;
  upperBoundMs: number;
  confidence: "low" | "medium" | "high";
  sampleCount: number;
  basis: string;
}
export interface ProbeAndPlan {
  inputType: InputType;
  video: VideoInfo | null;
  imageSequence: ImageSequenceInfo | null;
  plan: FramePlan;
  estimate: RuntimeEstimate;
}

export interface ReshootSourceInfo {
  projectId: string;
  projectName: string;
  quality: Quality;
  plannerEnabled: boolean;
  cameraId: number;
  cameraModel: string;
  width: number;
  height: number;
  sourceImageCount: number;
  eligible: boolean;
  reason: string | null;
}

export interface ReshootInputInfo {
  inputType: InputType;
  imageCount: number | null;
  duration: number | null;
  preparedWidth: number;
  preparedHeight: number;
  estimatedFrames: number;
  hasAlpha: boolean;
  maskCount: number;
  compatible: boolean;
  incompatibilityReason: string | null;
  estimate: RuntimeEstimate;
}

export interface PipelineEvent {
  taskId?: string | null; runId?: string | null; revision?: number;
  sequence: number;
  timestamp: string;
  kind: "stage" | "progress" | "log" | "heartbeat" | "capability" | "runtime";
  level: "info" | "warning" | "error";
  stage: string;
  component: "system" | "media" | "reconstruction" | "training" | null;
  progress: number;
  stageProgress: number | null;
  indeterminate: boolean;
  message: string;
  current: number | null;
  total: number | null;
  unit: string | null;
  elapsedMs: number;
  acceleration: AccelerationStatus | null;
  runtime?: RuntimeSnapshot | null;
}

export interface RuntimeSnapshot {
  processId: number;
  phase: string;
  updatedAt: string;
  lastOutputAgeMs: number;
  training: {
    iteration: number | null;
    total: number | null;
    startIter: number;
    lod: number;
    stepsPerSecond: number | null;
    remainingSeconds: number | null;
    splatCount: number | null;
    psnr: number | null;
    ssim: number | null;
  } | null;
  device: string | null;
  backend: string | null;
  config: Record<string, string>;
  resources: {
    processId: number;
    sampledAt: string;
    cpuPercent: number | null;
    memoryBytes: number | null;
    gpuStatus: "available" | "unavailable" | "unsupported";
    gpus: Array<{
      uuid: string;
      name: string;
      utilizationPercent: number | null;
      memoryUsedMib: number | null;
      memoryTotalMib: number | null;
    }>;
  } | null;
}

export interface PipelineResult {
  projectId: string;
  projectPath: string;
  finalPly: string;
  fileSize: number;
  splatCount: number;
  inputImages: number;
  registeredImages: number;
  registeredRatio: number;
  points3d: number;
  durationMs: number;
  completedAt: string;
  warning: string | null;
  logsDirectory: string;
}

export interface ProjectSummary {
  id: string;
  workspaceTaskId?: string | null;
  taskKind?: "generation" | "reshoot";
  name: string;
  status: ProjectStatus;
  projectPath: string;
  finalPly: string | null;
  fileSize: number | null;
  splatCount: number | null;
  createdAt: string;
  completedAt: string | null;
  durationMs: number | null;
  quality: Quality;
  sourceName: string;
  registeredRatio: number | null;
  points3d: number | null;
  failureMessage: string | null;
}

export interface AppSettings { schemaVersion: number; projectsRoot: string; plannerEnabled: boolean; }
export interface ProjectOverview { projectsRoot: string; plannerEnabled: boolean; projects: ProjectSummary[]; }
export interface AppRuntimeStatus {
  pipelineRunning: boolean;
  pipelineProjectId?: string | null;
  pipelineWorkspaceTaskId?: string | null;
  pipelineRunElapsedMs: number;
  pipelineElapsedOffsetMs: number;
  previewProjectId: string | null;
  acceleration: AccelerationStatus | null;
}

export interface ProjectTaskLogLine { source: string; message: string; }
export interface ProjectTaskDetail {
  project: ProjectSummary;
  inputType: InputType;
  sourcePath: string;
  projectsRoot: string;
  plannerEnabled: boolean;
  estimatedFrames: number | null;
  stage: string;
  progress: number;
  inputImages: number | null;
  registeredImages: number | null;
  video: VideoInfo | null;
  imageSequence: ImageSequenceInfo | null;
  sourceProjectId: string | null;
  logs: ProjectTaskLogLine[];
}

export type GaussianFormat = "ply" | "sog" | "spz";
export interface GaussianTransform {
  position: [number, number, number];
  rotation: [number, number, number];
  scale: number;
}
export type GaussianEditorTool = "transform" | "rectangle" | "sphere" | "box";
export type GaussianOrthographicView = "side" | "front" | "top";
export type GaussianCrop =
  | { kind: "sphere"; center: [number, number, number]; radius: number; inverted?: boolean }
  | { kind: "box"; center: [number, number, number]; size: [number, number, number]; inverted?: boolean }
  | null;
export interface GaussianEditState {
  crop: GaussianCrop;
  revision: number;
  sourceSplatCount: number;
  deletedCount: number;
}
export interface GaussianEditSaveSession {
  editId: string;
  expectedMaskBytes: number;
}
export interface GaussianPreviewDescriptor {
  projectId: string;
  modelPath: string;
  assetPath: string;
  format: GaussianFormat;
  fileSize: number;
  splatCount: number;
  transform: GaussianTransform;
  editing: GaussianEditState;
  editMaskAssetPath: string | null;
}
export interface GaussianExportProgress {
  projectId: string;
  processedSplats: number;
  totalSplats: number;
  progress: number;
}
export interface GaussianExportResult {
  path: string;
  fileSize: number;
  splatCount: number;
}

export interface GaussianVideoExportSession {
  exportId: string;
  destinationPath: string;
}

export type VideoOrientation = "portrait" | "landscape";
export interface GaussianHtmlView {
  target: [number, number, number];
  yaw: number;
  pitch: number;
  distance: number;
  horizontalFrameOffset: number;
  projection: number;
  orthoHeight: number;
  orthographicView: GaussianOrthographicView | null;
  fov: number;
  nearClip: number;
  farClip: number;
}
export interface GaussianHtmlExportProgress {
  exportId: string;
  phase: "model" | "packing" | "completed";
  progress: number;
}
export interface GaussianHtmlExportResult {
  path: string;
  fileSize: number;
  splatCount: number;
}

export interface GaussianVideoExportResult {
  path: string;
  fileSize: number;
  width: number;
  height: number;
  fps: number;
  durationMs: number;
}
