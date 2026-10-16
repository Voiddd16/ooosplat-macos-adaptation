//! ooo-sfm 的 BA（光束法平差）领域层。
//!
//! 本文件经 `engines/mod.rs` 的 `#[cfg(feature = "ooo-sfm")] + #[path]` 门面，
//! 以 `engines::colmap_ba` 的名义对外暴露；公开签名与 `engines/colmap_ba.rs` **逐字一致**。
//!
//! 原实现是"是否启用 Caspar（CUDA GPU BA）"的启发式探测；自研侧按平台重写（决策 D4）：
//! macOS = CPU BA，平台实现在 `ooo_sfm/apple/ba.rs`。M0 为占位，M6 落地。

use std::{ffi::OsString, path::Path};

// 平台后端与 `ooo_sfm/apple/mod.rs` 是同一份实现。
// 注意：本文件在 ooo-sfm 构建下就是 `engines::colmap_ba`，而 `engines::colmap`
// 是同一棵 `ooo_sfm` 树的另一视图，因此这里经它取平台实现，避免源码被编译两次。
#[cfg(target_vendor = "apple")]
use crate::engines::colmap::apple::ba as platform;

/// BA 后端选择结果（与 `engines/colmap_ba.rs::BaSelection` 同构）。
#[derive(Clone, Debug)]
pub struct BaSelection {
    pub gpu_index: Option<u32>,
    pub detail: String,
}

/// `database.db` 的规模统计（与 `engines/colmap_ba.rs::DatabaseStats` 同构）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DatabaseStats {
    pub keypoints: u64,
    pub verified_matches: u64,
}

pub async fn select(executable: &Path, compatible_gpu: Option<u32>) -> BaSelection {
    platform::select(executable, compatible_gpu).await
}

pub async fn select_for_mapper(
    executable: &Path,
    database: &Path,
    compatible_gpu: Option<u32>,
    automatic_optimization: bool,
) -> BaSelection {
    platform::select_for_mapper(executable, database, compatible_gpu, automatic_optimization).await
}

pub fn meets_caspar_threshold(stats: DatabaseStats) -> bool {
    platform::meets_caspar_threshold(stats)
}

pub async fn disable_after_runtime_failure(executable: &Path, index: u32, reason: &str) {
    platform::disable_after_runtime_failure(executable, index, reason).await
}

pub fn read_database_stats(database: &Path) -> Result<DatabaseStats, String> {
    platform::read_database_stats(database)
}

pub fn append_mapper_options(args: &mut Vec<OsString>, selection: &BaSelection) {
    platform::append_mapper_options(args, selection)
}

pub fn append_explicit_ceres_options(args: &mut Vec<OsString>) {
    platform::append_explicit_ceres_options(args)
}
