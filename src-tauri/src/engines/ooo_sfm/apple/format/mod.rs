//! COLMAP 兼容的数据格式（macOS 自研后端）。
//!
//! 这是本方案里**唯一不能自由发挥**的部分：产物必须能被
//! `reconstruction/colmap_model.rs`、`reconstruction/validator.rs`、
//! `pipeline/runner.rs`、Brush 数据集准备原样消费（方案文档 §2）。
//!
//! 权威参考（本仓库 `engines/colmap/colmap`，**必须是 4.2.1 标签**：
//! 工作区 checkout 的 main 已把 `cameras` 改成 `PRIMARY KEY(camera_id, source)`，
//! 与项目锁定的 4.2.1 不兼容）：
//!
//! ```text
//! git show 4.2.1:src/colmap/scene/reconstruction_io_binary.cc   # *.bin 布局
//! git show 4.2.1:src/colmap/scene/database_sqlite.cc            # schema 与 blob 布局
//! ```
//!
//! 所有整数/浮点均为**小端**；`*.bin` 的条目按 id 升序写出。

pub mod database;
pub mod model;

/// COLMAP `kMaxNumImages`（`util/types.h`）：`image_pair_id` 的进制。
pub const MAX_NUM_IMAGES: u64 = 2_147_483_647;

/// COLMAP `kInvalidPoint3DId`：`images.bin` 里表示"未三角化"。
pub const INVALID_POINT3D_ID: u64 = u64::MAX;

/// `database.db` 的 `PRAGMA user_version`，取自 4.2.1 的
/// `MakeDatabaseVersionNumber(4, 2, 1, 0)`（`util/version.cc.in`）。
pub const DATABASE_VERSION: i32 = 4_020_100;

/// `descriptors.type`：4.2.1 的 `FeatureExtractorType` 从 `-1` 起编号，
/// `UNDEFINED = -1, SIFT = 0`。
pub const DESCRIPTOR_TYPE_SIFT: i32 = 0;

/// SIFT 描述子维度（`database.db` 的 `descriptors.cols`）。
pub const DESCRIPTOR_COLS: usize = 128;

/// 相机模型枚举与参数个数（`colmap/scene/camera.h` 的 `CameraModelId`）。
pub mod camera_model {
    pub const SIMPLE_PINHOLE: i32 = 0;
    pub const PINHOLE: i32 = 1;
    pub const SIMPLE_RADIAL: i32 = 2;
    pub const RADIAL: i32 = 3;
    pub const OPENCV: i32 = 4;
    pub const OPENCV_FISHEYE: i32 = 5;
    pub const FULL_OPENCV: i32 = 6;
    pub const FOV: i32 = 7;
    pub const SIMPLE_RADIAL_FISHEYE: i32 = 8;
    pub const RADIAL_FISHEYE: i32 = 9;
    pub const THIN_PRISM_FISHEYE: i32 = 10;
    pub const RAD_TAN_THIN_PRISM_FISHEYE: i32 = 11;

    /// `CameraModelNumParams`：每个模型的参数个数。
    pub fn num_params(model_id: i32) -> Option<usize> {
        Some(match model_id {
            SIMPLE_PINHOLE => 3,
            PINHOLE => 4,
            SIMPLE_RADIAL => 4,
            RADIAL => 5,
            OPENCV => 8,
            OPENCV_FISHEYE => 8,
            FULL_OPENCV => 12,
            FOV => 5,
            SIMPLE_RADIAL_FISHEYE => 4,
            RADIAL_FISHEYE => 5,
            THIN_PRISM_FISHEYE => 12,
            RAD_TAN_THIN_PRISM_FISHEYE => 16,
            _ => return None,
        })
    }

    /// `CameraModelName`。
    pub fn name(model_id: i32) -> Option<&'static str> {
        Some(match model_id {
            SIMPLE_PINHOLE => "SIMPLE_PINHOLE",
            PINHOLE => "PINHOLE",
            SIMPLE_RADIAL => "SIMPLE_RADIAL",
            RADIAL => "RADIAL",
            OPENCV => "OPENCV",
            OPENCV_FISHEYE => "OPENCV_FISHEYE",
            FULL_OPENCV => "FULL_OPENCV",
            FOV => "FOV",
            SIMPLE_RADIAL_FISHEYE => "SIMPLE_RADIAL_FISHEYE",
            RADIAL_FISHEYE => "RADIAL_FISHEYE",
            THIN_PRISM_FISHEYE => "THIN_PRISM_FISHEYE",
            RAD_TAN_THIN_PRISM_FISHEYE => "RAD_TAN_THIN_PRISM_FISHEYE",
            _ => return None,
        })
    }
}
