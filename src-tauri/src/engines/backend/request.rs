//! 后端契约的**中性请求结构体**。
//!
//! 这些结构体是 pipeline 与后端之间的唯一数据通道。刻意不包含：
//!
//! * 可执行文件路径（那是后端自己的装配参数）；
//! * `gpu_index` / CUDA 设备号（后端自己从加速状态里取）；
//! * COLMAP 的 CLI 开关（`automatic_optimization` / `loop_detection` 之类）——
//!   需要它们的语义被翻译成领域选项，或者由后端自己决定。

use std::path::PathBuf;

/// 特征提取。
#[derive(Debug, Clone, Default)]
pub struct FeatureExtraction {
    /// COLMAP 兼容的 `database.db`。
    pub database: PathBuf,
    /// 图像目录（或帧目录）。
    pub images: PathBuf,
    /// 掩膜目录（可选）。
    pub masks: Option<PathBuf>,
    /// 图像白名单（可选）。
    pub image_list: Option<PathBuf>,
    /// 复用既有相机 id（增量提取时用）。
    pub existing_camera_id: Option<u32>,
    /// 长边上限。
    pub max_image_size: Option<u32>,
    /// 特征点上限。
    pub max_features: Option<u32>,
}

/// 特征提取（规划器模式：先规划整目录的特征预算再提取）。
#[derive(Debug, Clone, Default)]
pub struct QualityFeatureExtraction {
    pub base: FeatureExtraction,
    /// 规划后的特征预算。
    pub planned_max_features: u32,
}

/// 增量特征提取（补拍：既有相机 + 新画面）。
#[derive(Debug, Clone, Default)]
pub struct IncrementalFeatureExtraction {
    pub base: FeatureExtraction,
}

/// 匹配。
#[derive(Debug, Clone, Default)]
pub struct Matching {
    pub database: PathBuf,
    /// 是否启用闭环检测（词袋树）。不支持的后端应当自行忽略并如实上报能力。
    pub loop_detection: bool,
}

/// 指定帧对匹配。
#[derive(Debug, Clone, Default)]
pub struct PairMatching {
    pub database: PathBuf,
    /// 帧对清单。
    pub pair_list: PathBuf,
}

/// 稀疏重建。
#[derive(Debug, Clone, Default)]
pub struct SparseMapping {
    pub database: PathBuf,
    pub images: PathBuf,
    /// 既有模型目录（续建时用；首次重建留空）。
    pub input_model: PathBuf,
    /// 输出目录（`sparse/0/*.bin`）。
    pub output: PathBuf,
    /// 允许两视图 track（等价 COLMAP `--Mapper.tri_ignore_two_view_tracks 0`）。
    pub allow_two_view_tracks: bool,
    /// 允许后端自动挑选自己的优化实现（COLMAP 语义：自动在 Caspar / Ceres 之间选）。
    pub automatic_optimization: bool,
}

/// 增量重建（补拍）。
#[derive(Debug, Clone, Default)]
pub struct IncrementalMapping {
    pub database: PathBuf,
    pub images: PathBuf,
    /// 既有模型目录。
    pub input_model: PathBuf,
    pub output: PathBuf,
    /// 新增画面白名单。
    pub image_list: PathBuf,
    /// 允许后端自动挑选自己的优化实现。
    pub automatic_optimization: bool,
}
