//! M5c：三角化几何核 + `IncrementalTriangulator` 等价实现。
//!
//! 规格见 `docs/project-notes/recon/m5-mapper.md` §6 / §12（全部阈值核对自
//! COLMAP 4.2.1 `sfm/incremental_triangulator.cc`、`geometry/triangulation.cc`、
//! `estimators/triangulation.cc`）：
//!
//! - 三角化核：2 视图 DLT（`geometry/triangulation.cc:40`）+ 多视图垂距最小特征向量；
//!   校验顺序 = cheirality → 最小三角角（存在任一对视点满足即接受）。
//! - 残差类型：`ANGULAR_ERROR`（create / continue）与 `REPROJECTION_ERROR`
//!   （complete / retriangulate）。
//! - 采样：观测数 ≤ 15 时穷举两两组合（`kExhaustiveSamplingThreshold`），
//!   否则 RANSAC（`max_error = DegToRad(create_max_angle_error)`）。
//! - `Create` / `Continue` / `Merge` / `Complete` / `Retriangulate` /
//!   `TriangulateImage` / `CompleteImage` / `CompleteAndMergeTracks`。
//!
//! 刻意差异（记录在案）：
//! 1. RANSAC 采样用我们自己的 `Rng`（种子由调用方传入 → 结果确定）；
//!    4.2.1 用 `random_seed = -1`（每次运行不同）。
//! 2. 角误差残差按 4.2.1 的 `ANGULAR_ERROR` 语义（视线夹角差的平方），
//!    但没有引入 `loransac` 的 MSAC 打分等价优化 —— 局部优化用的是"最小三角角
//!    满足即接受"的 inlier 掩码（等价于 4.2.1 的 LORANSAC 支持度）。

use std::collections::HashSet;

use nalgebra::DMatrix;

use super::camera::{self, cam_point_from_img};
use super::format::model::TrackElement;
use super::linalg::{quaternion_to_rotation_matrix, rotation_to_quaternion};
use super::reconstruction::{CorrespondenceGraph, ObservationManager, Reconstruction};
use super::two_view::Rng;

// ---------------------------------------------------------------- 选项

/// `IncrementalTriangulator::Options`（默认值逐项对齐 4.2.1）。
#[derive(Clone, Copy, Debug)]
pub struct TriangulationOptions {
    pub max_transitivity: usize,
    pub create_max_angle_error: f64,
    pub continue_max_angle_error: f64,
    pub merge_max_reproj_error: f64,
    pub complete_max_reproj_error: f64,
    pub complete_max_transitivity: usize,
    pub re_max_angle_error: f64,
    pub re_min_ratio: f64,
    pub re_max_trials: usize,
    pub min_angle: f64,
    pub ignore_two_view_tracks: bool,
    pub min_focal_length_ratio: f64,
    pub max_focal_length_ratio: f64,
    pub max_extra_param: f64,
}

impl Default for TriangulationOptions {
    fn default() -> Self {
        Self {
            max_transitivity: 1,
            create_max_angle_error: 2.0,
            continue_max_angle_error: 2.0,
            merge_max_reproj_error: 4.0,
            complete_max_reproj_error: 4.0,
            complete_max_transitivity: 5,
            re_max_angle_error: 5.0,
            re_min_ratio: 0.2,
            re_max_trials: 1,
            min_angle: 1.5,
            ignore_two_view_tracks: true,
            min_focal_length_ratio: 0.1,
            max_focal_length_ratio: 10.0,
            max_extra_param: 1.0,
        }
    }
}

// ---------------------------------------------------------------- 几何核

/// 三角角（弧度）：`min(angle, π − angle)`，4.2.1 `CalculateTriangulationAngle`。
pub fn triangulation_angle(center1: [f64; 3], center2: [f64; 3], xyz: [f64; 3]) -> f64 {
    let ray1 = [xyz[0] - center1[0], xyz[1] - center1[1], xyz[2] - center1[2]];
    let ray2 = [xyz[0] - center2[0], xyz[1] - center2[1], xyz[2] - center2[2]];
    let norm1 = (ray1[0].powi(2) + ray1[1].powi(2) + ray1[2].powi(2)).sqrt();
    let norm2 = (ray2[0].powi(2) + ray2[1].powi(2) + ray2[2].powi(2)).sqrt();
    if !(norm1 > 0.0) || !(norm2 > 0.0) {
        return 0.0;
    }
    let cosine =
        (ray1[0] * ray2[0] + ray1[1] * ray2[1] + ray1[2] * ray2[2]) / (norm1 * norm2);
    let angle = cosine.clamp(-1.0, 1.0).acos();
    angle.min(std::f64::consts::PI - angle)
}

/// 2 视图线性三角化（4.2.1 `TriangulatePoint`）。
///
/// `point1` / `point2` 是**归一化**像平面坐标（`cam_ray.hnormalized()`，不含内参）——
/// 与 4.2.1 `TriangulationEstimator` 里 2 视图分支的调用方式一致。
///
/// ⚠️ 零空间是一条过原点的直线，返回解的**符号不确定**（SVD 任意取一向量）。
/// 4.2.1 也一样：真正的判据是随后的 cheirality 检查
/// （`HasPointPositiveDepth`，本模块的 `estimate_triangulation` 里做）。
pub fn triangulate_two_view(
    qvec1: &[f64; 4],
    tvec1: &[f64; 3],
    point1: [f64; 2],
    qvec2: &[f64; 4],
    tvec2: &[f64; 3],
    point2: [f64; 2],
) -> Option<[f64; 3]> {
    let r1 = quaternion_to_rotation_matrix(qvec1);
    let r2 = quaternion_to_rotation_matrix(qvec2);
    // A 的 4 行：u·P3 − P1、v·P3 − P2（两视图各 2 行）。
    let mut a = DMatrix::<f64>::zeros(4, 4);
    for row in 0..2 {
        // A 的第 row 行 = u·P.row(2) − P.row(row)（4 维行向量）。
        // 第 4 项是 `u·t_z − t_x`（注意符号，4.2.1 `geometry/triangulation.cc:52`）。
        let term1 = [
            r1[0][2] * point1[row] - r1[row][0],
            r1[1][2] * point1[row] - r1[row][1],
            r1[2][2] * point1[row] - r1[row][2],
            point1[row] * tvec1[2] - tvec1[row],
        ];
        let term2 = [
            r2[0][2] * point2[row] - r2[row][0],
            r2[1][2] * point2[row] - r2[row][1],
            r2[2][2] * point2[row] - r2[row][2],
            point2[row] * tvec2[2] - tvec2[row],
        ];
        for col in 0..4 {
            a[(row, col)] = term1[col];
            a[(2 + row, col)] = term2[col];
        }
    }
    let svd = nalgebra::linalg::SVD::try_new(a, false, true, 5.0 * f64::EPSILON, 200)?;
    // `v_t` = Vᵀ；第 3 行即第 4 个右奇异向量（`V.col(3)`）。
    let v_t = svd.v_t?;
    let homogeneous = [v_t[(3, 0)], v_t[(3, 1)], v_t[(3, 2)], v_t[(3, 3)]];
    if homogeneous[3] == 0.0 {
        return None;
    }
    let mut xyz = [
        homogeneous[0] / homogeneous[3],
        homogeneous[1] / homogeneous[3],
        homogeneous[2] / homogeneous[3],
    ];
    if !xyz.iter().all(|value| value.is_finite()) {
        return None;
    }
    // 零空间符号不定：取第一台相机深度为正的那一支（4.2.1 里由后续 cheirality
    // 检查隐式完成，这里显式定号，避免整对被误判为非 cheiral）。
    if depth_in_first_view(qvec1, tvec1, xyz) < 0.0 {
        for value in &mut xyz {
            *value = -*value;
        }
    }
    Some(xyz)
}

/// 点在第一台相机下的深度（`[R|t]` 第三行 · `[X;1]`）。
fn depth_in_first_view(qvec: &[f64; 4], tvec: &[f64; 3], xyz: [f64; 3]) -> f64 {
    let r = quaternion_to_rotation_matrix(qvec);
    r[2][0] * xyz[0] + r[2][1] * xyz[1] + r[2][2] * xyz[2] + tvec[2]
}

/// 多视图线性三角化（最小化垂距 L2；自伴特征分解取最小特征向量，
/// 4.2.1 `TriangulateMultiViewPoint`）。
pub fn triangulate_multi_view(
    first_qvec: &[f64; 4],
    first_tvec: &[f64; 3],
    projections: &[[[f64; 4]; 3]],
    rays: &[[f64; 3]],
) -> Option<[f64; 3]> {
    // 首观测相机的位姿（用于符号定号）。
    let (recon_qvec_first, recon_tvec_first) = (first_qvec, first_tvec);
    // 4.2.1 `TriangulationDltTerm`：`term = P − ray·rayᵀP`（3×4），
    // 累加 `termᵀ·term` 得到对称 4×4 法方程。
    let mut normal = [[0.0f64; 4]; 4];
    for (projection, ray) in projections.iter().zip(rays) {
        // rayᵀP 是 1×4：第 j 项 = Σ_k ray_k·P[k][j]。
        let mut ray_dot_p = [0.0f64; 4];
        for j in 0..4 {
            ray_dot_p[j] = ray[0] * projection[0][j]
                + ray[1] * projection[1][j]
                + ray[2] * projection[2][j];
        }
        // term 行 i：[P[i][j] − ray_i·rayᵀP[j]]
        let mut term = [[0.0f64; 4]; 3];
        for i in 0..3 {
            for j in 0..4 {
                term[i][j] = projection[i][j] - ray[i] * ray_dot_p[j];
            }
        }
        for j in 0..4 {
            for l in 0..4 {
                let mut sum = 0.0;
                for i in 0..3 {
                    sum += term[i][j] * term[i][l];
                }
                normal[j][l] += sum;
            }
        }
    }
    let matrix = nalgebra::SMatrix::<f64, 4, 4>::from_column_slice(&[
        normal[0][0], normal[1][0], normal[2][0], normal[3][0], //
        normal[0][1], normal[1][1], normal[2][1], normal[3][1], //
        normal[0][2], normal[1][2], normal[2][2], normal[3][2], //
        normal[0][3], normal[1][3], normal[2][3], normal[3][3],
    ]);
    let eigen = nalgebra::linalg::SymmetricEigen::try_new(matrix, 5.0 * f64::EPSILON, 100 * 4)?;
    let mut smallest = 0;
    for index in 1..4 {
        if eigen.eigenvalues[index] < eigen.eigenvalues[smallest] {
            smallest = index;
        }
    }
    let vector = eigen.eigenvectors.column(smallest);
    let xyz = [vector[0], vector[1], vector[2]];
    let w = vector[3];
    if w.abs() < f64::EPSILON {
        // 点在无穷远（4.2.1 同样把 `w == 0` 视为失败）。
        return None;
    }
    let point = [xyz[0] / w, xyz[1] / w, xyz[2] / w];
    if !point.iter().all(|value| value.is_finite()) {
        return None;
    }
    // 特征向量符号不定：与 2 视图一致，统一到"首观测相机深度为正"的那一支。
    if depth_in_first_view(recon_qvec_first, recon_tvec_first, point) < 0.0
    {
        return Some([-point[0], -point[1], -point[2]]);
    }
    Some(point)
}

/// 观测视线（4.2.1 `Camera::CamRayFromImg` 语义：**单位长度**的相机系光线）。
///
/// ⚠️ 单位长度很重要：`TriangulationDltTerm` 里的 `P − ray·rayᵀP` 只有在
/// `ray` 是单位向量时才是"到光轴的正交投影"，否则 `A` 不再是代数误差的
/// Hessian，多视图三角化会系统性偏离。
fn ray_of(recon: &Reconstruction, image: usize, point2d_idx: u32) -> Option<[f64; 3]> {
    let point2d = recon.images[image].points2d[point2d_idx as usize];
    let ray = cam_point_from_img(
        recon.model_of(image),
        &recon.camera_of(image).params,
        point2d.x,
        point2d.y,
    )?;
    let norm = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
    if !(norm > 0.0) {
        return None;
    }
    Some([ray[0] / norm, ray[1] / norm, ray[2] / norm])
}

/// 投影矩阵前三行（`P = [R|t]` 的 3×4）。
fn projection_rows(qvec: &[f64; 4], tvec: &[f64; 3]) -> [[f64; 4]; 3] {
    let r = quaternion_to_rotation_matrix(qvec);
    [
        [r[0][0], r[0][1], r[0][2], tvec[0]],
        [r[1][0], r[1][1], r[1][2], tvec[1]],
        [r[2][0], r[2][1], r[2][2], tvec[2]],
    ]
}

/// 三角化观测集：≥2 个观测，采样两两组合（≤15 观测穷举，否则 RANSAC），
/// 返回通过 cheirality + 最小三角角校验的模型。
fn estimate_triangulation(
    recon: &Reconstruction,
    observations: &[TrackElement],
    min_tri_angle: f64,
    max_error: f64,
    angular: bool,
    rng: &mut Rng,
) -> Option<([f64; 3], Vec<bool>)> {
    if observations.len() < 2 {
        return None;
    }
    // 残差：角误差（度）or 像素重投影误差。
    let residual = |xyz: [f64; 3], element: &TrackElement| -> f64 {
        let image = recon.image_index[&element.image_id];
        if angular {
            let Some(ray) = ray_of(recon, image, element.point2d_idx) else {
                return f64::MAX;
            };
            let center = recon.projection_center_of(image);
            // 观测视线（相机系 → 世界系方向；`cam = R·world + t` ⇒ 方向用 Rᵀ 变换）。
            let r = quaternion_to_rotation_matrix(&recon.images[image].qvec);
            let world_ray = [
                r[0][0] * ray[0] + r[1][0] * ray[1] + r[2][0] * ray[2],
                r[0][1] * ray[0] + r[1][1] * ray[1] + r[2][1] * ray[2],
                r[0][2] * ray[0] + r[1][2] * ray[1] + r[2][2] * ray[2],
            ];
            let point_ray = [xyz[0] - center[0], xyz[1] - center[1], xyz[2] - center[2]];
            let norm1 = (world_ray[0].powi(2) + world_ray[1].powi(2) + world_ray[2].powi(2)).sqrt();
            let norm2 =
                (point_ray[0].powi(2) + point_ray[1].powi(2) + point_ray[2].powi(2)).sqrt();
            if !(norm1 > 0.0) || !(norm2 > 0.0) {
                return f64::MAX;
            }
            let cosine =
                (world_ray[0] * point_ray[0] + world_ray[1] * point_ray[1] + world_ray[2] * point_ray[2])
                    / (norm1 * norm2);
            cosine.clamp(-1.0, 1.0).acos().to_degrees()
        } else {
            let point2d = recon.images[image].points2d[element.point2d_idx as usize];
            camera::reprojection_error(
                recon.camera_of(image),
                recon.model_of(image),
                &recon.images[image].qvec,
                &recon.images[image].tvec,
                xyz,
                [point2d.x, point2d.y],
            )
        }
    };
    let cheirality_ok = |xyz: [f64; 3]| -> bool {
        observations.iter().all(|element| {
            let image = recon.image_index[&element.image_id];
            let point = camera::cam_from_world_point(
                &recon.images[image].qvec,
                &recon.images[image].tvec,
                xyz,
            );
            point[2] >= 1e-6
        })
    };
    let angle_ok = |xyz: [f64; 3]| -> bool {
        let mut centers: Vec<[f64; 3]> = observations
            .iter()
            .map(|element| {
                let image = recon.image_index[&element.image_id];
                recon.projection_center_of(image)
            })
            .collect();
        centers.sort_by(|a, b| a.partial_cmp(b).expect("投影中心有限"));
        for i in 0..centers.len() {
            for j in 0..i {
                if triangulation_angle(centers[i], centers[j], xyz) >= min_tri_angle {
                    return true;
                }
            }
        }
        false
    };
    // 观测 → (相机系光线, 投影矩阵)。
    let ray_and_projection = |element: &TrackElement| -> Option<([f64; 3], [[f64; 4]; 3])> {
        let image = recon.image_index[&element.image_id];
        let ray = ray_of(recon, image, element.point2d_idx)?;
        Some((ray, projection_rows(&recon.images[image].qvec, &recon.images[image].tvec)))
    };
    let estimate_from_pair = |pair: (usize, usize)| -> Option<[f64; 3]> {
        let (ray_a, _) = ray_and_projection(&observations[pair.0])?;
        let (ray_b, _) = ray_and_projection(&observations[pair.1])?;
        let image_a = recon.image_index[&observations[pair.0].image_id];
        let image_b = recon.image_index[&observations[pair.1].image_id];
        triangulate_two_view(
            &recon.images[image_a].qvec,
            &recon.images[image_a].tvec,
            [ray_a[0] / ray_a[2], ray_a[1] / ray_a[2]],
            &recon.images[image_b].qvec,
            &recon.images[image_b].tvec,
            [ray_b[0] / ray_b[2], ray_b[1] / ray_b[2]],
        )
    };
    // 非最小候选：用**全部**观测的多视图 DLT（4.2.1 `TriangulateMultiViewPoint`）。
    let estimate_from_all = || -> Option<[f64; 3]> {
        if observations.len() <= 2 {
            return None;
        }
        let mut rays = Vec::with_capacity(observations.len());
        let mut projections = Vec::with_capacity(observations.len());
        for element in observations {
            let (ray, projection) = ray_and_projection(element)?;
            rays.push(ray);
            projections.push(projection);
        }
        let first_image = recon.image_index[&observations[0].image_id];
        triangulate_multi_view(
            &recon.images[first_image].qvec,
            &recon.images[first_image].tvec,
            &projections,
            &rays,
        )
    };

    let mut best: Option<([f64; 3], Vec<bool>)> = None;
    let mut best_inliers = 0usize;

    let count = observations.len();
    if count <= 15 {
        // 穷举所有两两组合（4.2.1 `kExhaustiveSamplingThreshold = 15`）。
        for i in 0..count {
            for j in 0..i {
                if let Some(xyz) = estimate_from_pair((i, j)) {
                    if !cheirality_ok(xyz) || !angle_ok(xyz) {
                        continue;
                    }
                    let mask: Vec<bool> = observations
                        .iter()
                        .map(|element| residual(xyz, element) <= max_error)
                        .collect();
                    let inliers = mask.iter().filter(|value| **value).count();
                    if inliers > best_inliers {
                        best_inliers = inliers;
                        best = Some((xyz, mask));
                    }
                }
            }
        }
    } else {
        // RANSAC：最大 10000 次，置信度 0.999（4.2.1 `EstimateTriangulationOptions`）。
        const MAX_TRIALS: usize = 10_000;
        let mut trials = 100usize.min(MAX_TRIALS);
        while trials > 0 {
            let first = rng.range(observations.len());
            let mut second = rng.range(observations.len());
            if first == second {
                second = (second + 1) % observations.len();
            }
            if let Some(xyz) = estimate_from_pair((first, second)) {
                if cheirality_ok(xyz) && angle_ok(xyz) {
                    let mask: Vec<bool> = observations
                        .iter()
                        .map(|element| residual(xyz, element) <= max_error)
                        .collect();
                    let inliers = mask.iter().filter(|value| **value).count();
                    if inliers > best_inliers {
                        best_inliers = inliers;
                        best = Some((xyz, mask));
                    }
                }
            }
            // 自适应收缩：内点比 p 时一次采中两个内点的概率为 p²。
            let ratio = best_inliers as f64 / count as f64;
            let probability = ratio * ratio;
            if probability > 0.0 {
                let needed = ((1.0 - 0.999f64).ln() / (1.0 - probability).ln()).ceil() as usize;
                trials = trials.min(needed).min(MAX_TRIALS);
            }
            trials = trials.saturating_sub(1);
        }
    }

    // 最后再补一次"全观测多视图 DLT"候选（等价 4.2.1 的非最小估计器）。
    if let Some(xyz) = estimate_from_all() {
        if cheirality_ok(xyz) && angle_ok(xyz) {
            let mask: Vec<bool> = observations
                .iter()
                .map(|element| residual(xyz, element) <= max_error)
                .collect();
            let inliers = mask.iter().filter(|value| **value).count();
            if inliers > best_inliers {
                best = Some((xyz, mask));
            }
        }
    }
    best
}

// ---------------------------------------------------------------- 三角化器

/// `IncrementalTriangulator` 等价实现（可变点集合由 mapper 维护）。
pub struct Triangulator {
    pub modified: HashSet<usize>,
    re_trials: std::collections::HashMap<(usize, usize), usize>,
    merge_trials: HashSet<(usize, usize, usize, u32)>,
}

impl Default for Triangulator {
    fn default() -> Self {
        Self {
            modified: HashSet::new(),
            re_trials: std::collections::HashMap::new(),
            merge_trials: HashSet::new(),
        }
    }
}

impl Triangulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_modified(&mut self, point: usize) {
        self.modified.insert(point);
    }

    pub fn clear_modified(&mut self) {
        self.modified.clear();
    }

    pub fn modified_points(&self, recon: &Reconstruction) -> Vec<usize> {
        self.modified
            .iter()
            .copied()
            .filter(|point| *point < recon.points.len())
            .collect()
    }

    fn camera_is_bogus(recon: &Reconstruction, image: usize, options: &TriangulationOptions) -> bool {
        camera::has_bogus_params(
            recon.camera_of(image),
            options.min_focal_length_ratio,
            options.max_focal_length_ratio,
            options.max_extra_param,
        )
    }

    /// 4.2.1 `TriangulateImage`：创建新点 + 续上已有点（逐特征点）。
    pub fn triangulate_image(
        &mut self,
        graph: &CorrespondenceGraph,
        recon: &mut Reconstruction,
        obs: &mut ObservationManager,
        options: &TriangulationOptions,
        image: usize,
        rng: &mut Rng,
    ) -> usize {
        if !recon.images[image].registered || Self::camera_is_bogus(recon, image, options) {
            return 0;
        }
        let mut added = 0usize;
        for point2d_idx in 0..recon.images[image].points2d.len() as u32 {
            let correspondences =
                graph.find_transitive(image, point2d_idx, options.max_transitivity);
            if correspondences.is_empty() {
                continue;
            }
            let observations: Vec<TrackElement> = correspondences
                .iter()
                .filter(|corr| {
                    recon.images[corr.image].registered
                        && !Self::camera_is_bogus(recon, corr.image, options)
                })
                .map(|corr| TrackElement {
                    image_id: recon.images[corr.image].id,
                    point2d_idx: corr.point2d_idx,
                })
                .collect();
            if observations.is_empty() {
                continue;
            }
            let has_triangulated = recon.point_index_at(image, point2d_idx).is_some();
            if has_triangulated {
                // 先续（可能失败），再把参考观测加入后创建新点。
                let reference = TrackElement {
                    image_id: recon.images[image].id,
                    point2d_idx,
                };
                added += self.continue_track(
                    graph,
                    recon,
                    obs,
                    options,
                    image,
                    point2d_idx,
                    &reference,
                    &observations,
                );
            }
            let mut with_reference = observations.clone();
            with_reference.push(TrackElement {
                image_id: recon.images[image].id,
                point2d_idx,
            });
            added += self.create(graph, recon, obs, options, image, point2d_idx, &with_reference, rng);
        }
        added
    }

    /// 4.2.1 `Create`：从未三角化的对应里建新点。
    #[allow(clippy::too_many_arguments)]
    fn create(
        &mut self,
        graph: &CorrespondenceGraph,
        recon: &mut Reconstruction,
        obs: &mut ObservationManager,
        options: &TriangulationOptions,
        image: usize,
        point2d_idx: u32,
        observations: &[TrackElement],
        rng: &mut Rng,
    ) -> usize {
        let candidates: Vec<TrackElement> = observations
            .iter()
            .copied()
            .filter(|element| {
                let target = recon.image_index[&element.image_id];
                recon.point_index_at(target, element.point2d_idx).is_none()
            })
            .collect();
        if candidates.len() < 2 {
            return 0;
        }
        if options.ignore_two_view_tracks
            && candidates.len() == 2
            && graph.is_two_view_observation(
                recon.image_index[&candidates[0].image_id],
                candidates[0].point2d_idx,
            )
        {
            return 0;
        }
        let estimated = estimate_triangulation(
            recon,
            &candidates,
            options.min_angle.to_radians(),
            options.create_max_angle_error,
            true,
            rng,
        );
        if std::env::var("OOO_TRI_DEBUG").is_ok() {
            match &estimated {
                Some((xyz, mask)) => eprintln!(
                    "[tri] create {} 候选 → {:?}，内点 {}/{}",
                    candidates.len(),
                    xyz,
                    mask.iter().filter(|v| **v).count(),
                    mask.len()
                ),
                None => eprintln!("[tri] create {} 候选 → 无解", candidates.len()),
            }
        }
        let Some((xyz, mask)) = estimated else {
            return 0;
        };
        let track: Vec<TrackElement> = candidates
            .iter()
            .zip(&mask)
            .filter(|(_, keep)| **keep)
            .map(|(element, _)| *element)
            .collect();
        if track.len() < 2 {
            return 0;
        }
        let point = obs.add_point3d(graph, recon, xyz, recon.mean_track_color(&track), track.clone());
        self.modified.insert(point);
        let mut added = track.len();
        // 4.2.1：剩余候选仍够 3 个时递归再试一次（拆成不同 track）。
        if candidates.len() - track.len() >= 3 {
            let remaining: Vec<TrackElement> = candidates
                .iter()
                .zip(&mask)
                .filter(|(_, keep)| !**keep)
                .map(|(element, _)| *element)
                .collect();
            added += self.create(graph, recon, obs, options, image, point2d_idx, &remaining, rng);
        }
        added
    }

    /// 4.2.1 `Continue`：把参考观测挂到角重投影误差最小的已有三维点上。
    #[allow(clippy::too_many_arguments)]
    fn continue_track(
        &mut self,
        graph: &CorrespondenceGraph,
        recon: &mut Reconstruction,
        obs: &mut ObservationManager,
        options: &TriangulationOptions,
        image: usize,
        point2d_idx: u32,
        reference: &TrackElement,
        observations: &[TrackElement],
    ) -> usize {
        let mut best: Option<(f64, usize)> = None;
        for element in observations {
            let target = recon.image_index[&element.image_id];
            let Some(point) = recon.point_index_at(target, element.point2d_idx) else {
                continue;
            };
            let point2d = recon.images[image].points2d[point2d_idx as usize];
            let error = camera::reprojection_error(
                recon.camera_of(image),
                recon.model_of(image),
                &recon.images[image].qvec,
                &recon.images[image].tvec,
                recon.points[point].xyz,
                [point2d.x, point2d.y],
            );
            if best.map(|(current, _)| error < current).unwrap_or(true) {
                best = Some((error, point));
            }
        }
        let Some((error, point)) = best else {
            return 0;
        };
        if error > options.continue_max_angle_error {
            return 0;
        }
        let _ = reference;
        if obs.add_observation(graph, recon, point, *reference) {
            self.modified.insert(point);
            1
        } else {
            0
        }
    }

    /// 4.2.1 `CompleteImage`：为未三角化的特征点补全 track。
    pub fn complete_image(
        &mut self,
        graph: &CorrespondenceGraph,
        recon: &mut Reconstruction,
        obs: &mut ObservationManager,
        options: &TriangulationOptions,
        image: usize,
        rng: &mut Rng,
    ) -> usize {
        if !recon.images[image].registered {
            return 0;
        }
        let mut completed = 0usize;
        for point2d_idx in 0..recon.images[image].points2d.len() as u32 {
            if recon.point_index_at(image, point2d_idx).is_some() {
                let point = recon.point_index_at(image, point2d_idx).expect("上一步判定");
                let track = recon.points[point].track.clone();
                if track.len() < 2 {
                    continue;
                }
                completed += self.complete_point(graph, recon, obs, options, point);
                continue;
            }
            if options.ignore_two_view_tracks && graph.is_two_view_observation(image, point2d_idx)
            {
                continue;
            }
            let correspondences =
                graph.find_transitive(image, point2d_idx, options.max_transitivity);
            let candidates: Vec<TrackElement> = correspondences
                .iter()
                .filter(|corr| {
                    recon.images[corr.image].registered
                        && !Self::camera_is_bogus(recon, corr.image, options)
                })
                .map(|corr| TrackElement {
                    image_id: recon.images[corr.image].id,
                    point2d_idx: corr.point2d_idx,
                })
                .collect();
            let any_triangulated = candidates.iter().any(|element| {
                let target = recon.image_index[&element.image_id];
                recon.point_index_at(target, element.point2d_idx).is_some()
            });
            if candidates.is_empty() || any_triangulated {
                continue;
            }
            let mut with_reference = candidates;
            with_reference.push(TrackElement {
                image_id: recon.images[image].id,
                point2d_idx,
            });
            let Some((xyz, mask)) = estimate_triangulation(
                recon,
                &with_reference,
                options.min_angle.to_radians(),
                options.complete_max_reproj_error,
                false,
                rng,
            ) else {
                continue;
            };
            let track: Vec<TrackElement> = with_reference
                .iter()
                .zip(&mask)
                .filter(|(_, keep)| **keep)
                .map(|(element, _)| *element)
                .collect();
            if track.len() < 2 {
                continue;
            }
            let point = obs.add_point3d(
                graph,
                recon,
                xyz,
                recon.mean_track_color(&track),
                track.clone(),
            );
            self.modified.insert(point);
            completed += track.len();
        }
        completed
    }

    /// 4.2.1 `Complete`：以现有 track 为种子做传递补全。
    pub fn complete_point(
        &mut self,
        graph: &CorrespondenceGraph,
        recon: &mut Reconstruction,
        obs: &mut ObservationManager,
        options: &TriangulationOptions,
        point: usize,
    ) -> usize {
        let mut completed = 0usize;
        let mut visited: HashSet<(u32, u32)> = HashSet::new();
        for element in &recon.points[point].track {
            visited.insert((element.image_id, element.point2d_idx));
        }
        let mut frontier: Vec<(usize, u32)> = recon.points[point]
            .track
            .iter()
            .map(|element| {
                (
                    recon.image_index[&element.image_id],
                    element.point2d_idx,
                )
            })
            .collect();
        let max_error = options.complete_max_reproj_error;
        let min_angle = options.min_angle.to_radians();
        for _ in 0..options.complete_max_transitivity {
            let mut next: Vec<(usize, u32)> = Vec::new();
            for (image, point2d_idx) in &frontier {
                for index in 0..graph.find(*image, *point2d_idx).len() {
                    let corr = graph.find(*image, *point2d_idx)[index];
                    if !visited.insert((recon.images[corr.image].id, corr.point2d_idx)) {
                        continue;
                    }
                    if !recon.images[corr.image].registered
                        || Self::camera_is_bogus(recon, corr.image, options)
                        || recon.point_index_at(corr.image, corr.point2d_idx).is_some()
                    {
                        continue;
                    }
                    let element = TrackElement {
                        image_id: recon.images[corr.image].id,
                        point2d_idx: corr.point2d_idx,
                    };
                    let xyz = recon.points[point].xyz;
                    let observation = recon.images[corr.image].points2d[corr.point2d_idx as usize];
                    let error = camera::reprojection_error(
                        recon.camera_of(corr.image),
                        recon.model_of(corr.image),
                        &recon.images[corr.image].qvec,
                        &recon.images[corr.image].tvec,
                        xyz,
                        [observation.x, observation.y],
                    );
                    // 最小三角角校验（与新点同标准）。
                    let mut ok = error <= max_error;
                    if ok {
                        let center_new = recon.projection_center_of(corr.image);
                        ok = recon.points[point].track.iter().any(|element| {
                            let other = recon.image_index[&element.image_id];
                            triangulation_angle(center_new, recon.projection_center_of(other), xyz)
                                >= min_angle
                        });
                    }
                    if ok && obs.add_observation(graph, recon, point, element) {
                        self.modified.insert(point);
                        completed += 1;
                        next.push((corr.image, corr.point2d_idx));
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        completed
    }

    /// 4.2.1 `Merge`：把两条 track 合成一条（长度加权位置 + 重投影校验）。
    pub fn merge_point(
        &mut self,
        graph: &CorrespondenceGraph,
        recon: &mut Reconstruction,
        obs: &mut ObservationManager,
        options: &TriangulationOptions,
        point: usize,
    ) -> usize {
        let track = recon.points[point].track.clone();
        for element in &track {
            let image = recon.image_index[&element.image_id];
            for index in 0..graph.find(image, element.point2d_idx).len() {
                let corr = graph.find(image, element.point2d_idx)[index];
                if corr.image == image || !recon.images[corr.image].registered {
                    continue;
                }
                let Some(other) = recon.point_index_at(corr.image, corr.point2d_idx) else {
                    continue;
                };
                if other == point {
                    continue;
                }
                let key = (
                    point.min(other),
                    point.max(other),
                    corr.image,
                    corr.point2d_idx,
                );
                if !self.merge_trials.insert(key) {
                    continue;
                }
                let xyz1 = recon.points[point].xyz;
                let xyz2 = recon.points[other].xyz;
                let weight1 = recon.points[point].track.len() as f64;
                let weight2 = recon.points[other].track.len() as f64;
                let mut merged = [0.0; 3];
                for axis in 0..3 {
                    merged[axis] = (weight1 * xyz1[axis] + weight2 * xyz2[axis]) / (weight1 + weight2);
                }
                let ok = (recon.points[point].track.iter().chain(recon.points[other].track.iter()))
                    .all(|element| {
                        let image = recon.image_index[&element.image_id];
                        let observation = recon.images[image].points2d[element.point2d_idx as usize];
                        camera::reprojection_error(
                            recon.camera_of(image),
                            recon.model_of(image),
                            &recon.images[image].qvec,
                            &recon.images[image].tvec,
                            merged,
                            [observation.x, observation.y],
                        ) <= options.merge_max_reproj_error
                    });
                if !ok {
                    continue;
                }
                let (first, second) = if point < other {
                    (point, other)
                } else {
                    (other, point)
                };
                let Some(new_point) = obs.merge_points3d(graph, recon, first, second) else {
                    continue;
                };
                self.modified.insert(new_point);
                let length = recon.points[new_point].track.len();
                self.merge_point(graph, recon, obs, options, new_point);
                return length;
            }
        }
        0
    }

    /// 4.2.1 `CompleteAndMergeTracks`（对全部点：先 merge 再 complete）。
    pub fn complete_and_merge_tracks(
        &mut self,
        graph: &CorrespondenceGraph,
        recon: &mut Reconstruction,
        obs: &mut ObservationManager,
        options: &TriangulationOptions,
    ) -> usize {
        let mut changed = 0usize;
        let mut points: Vec<usize> = (0..recon.points.len()).collect();
        points.sort_unstable_by(|a, b| b.cmp(a));
        for point in points {
            if point >= recon.points.len() {
                continue;
            }
            changed += self.merge_point(graph, recon, obs, options, point);
        }
        let points: Vec<usize> = (0..recon.points.len()).collect();
        for point in points {
            changed += self.complete_point(graph, recon, obs, options, point);
        }
        changed
    }

    /// 4.2.1 `Retriangulate`：对三角化率过低的图像对重新三角化。
    pub fn retriangulate(
        &mut self,
        graph: &CorrespondenceGraph,
        recon: &mut Reconstruction,
        obs: &mut ObservationManager,
        options: &TriangulationOptions,
        rng: &mut Rng,
    ) -> usize {
        let mut count = 0usize;
        let pairs = graph.image_pairs();
        for (image1, image2) in pairs {
            let key = (image1, image2);
            let Some(stats) = obs.pairs.get(&key) else {
                continue;
            };
            if stats.num_total_corrs == 0 {
                continue;
            }
            let ratio = stats.num_tri_corrs as f64 / stats.num_total_corrs as f64;
            if ratio >= options.re_min_ratio {
                continue;
            }
            if !recon.images[image1].registered || !recon.images[image2].registered {
                continue;
            }
            let trials = self.re_trials.entry(key).or_insert(0);
            if *trials >= options.re_max_trials {
                continue;
            }
            *trials += 1;
            if Self::camera_is_bogus(recon, image1, options) || Self::camera_is_bogus(recon, image2, options)
            {
                continue;
            }
            for (idx1, idx2) in graph.matches_between(image1, image2) {
                let point1 = recon.point_index_at(image1, idx1);
                let point2 = recon.point_index_at(image2, idx2);
                match (point1, point2) {
                    // 两侧都有点：不合并（4.2.1 同样跳过）。
                    (Some(_), Some(_)) => {}
                    (Some(_point), None) | (None, Some(_point)) => {
                        let target = if point1.is_some() { image2 } else { image1 };
                        let target_idx = if point1.is_some() { idx2 } else { idx1 };
                        let reference = TrackElement {
                            image_id: if point1.is_some() {
                                recon.images[image1].id
                            } else {
                                recon.images[image2].id
                            },
                            point2d_idx: if point1.is_some() { idx1 } else { idx2 },
                        };
                        let re_options = TriangulationOptions {
                            continue_max_angle_error: options.re_max_angle_error,
                            ..*options
                        };
                        count += self.continue_track(
                            graph,
                            recon,
                            obs,
                            &re_options,
                            target,
                            target_idx,
                            &reference,
                            &[reference],
                        );
                    }
                    (None, None) => {
                        let observations = vec![
                            TrackElement {
                                image_id: recon.images[image1].id,
                                point2d_idx: idx1,
                            },
                            TrackElement {
                                image_id: recon.images[image2].id,
                                point2d_idx: idx2,
                            },
                        ];
                        // 4.2.1 刻意沿用 `options` 的角阈值，不用 re 的 5°。
                        count += self.create(graph, recon, obs, options, image1, idx1, &observations, rng);
                    }
                }
            }
        }
        count
    }
}

/// 4.2.1 `Reconstruction::Normalize` 用的四元数工具（供 mapper 复用）。
pub fn identity_pose() -> ([f64; 4], [f64; 3]) {
    ([1.0, 0.0, 0.0, 0.0], [0.0; 3])
}

/// 由 `R,t` 组装 `qvec/tvec`（mapper 里手工构造相对位姿时用）。
pub fn pose_from_rotation_translation(rotation: [[f64; 3]; 3], translation: [f64; 3]) -> ([f64; 4], [f64; 3]) {
    (rotation_to_quaternion(&rotation), translation)
}

/// `Point3D` 的误差字段是否有值（4.2.1 `Point3D::HasError`）。
pub fn has_error(error: f64) -> bool {
    error >= 0.0
}

/// 投影矩阵第三行（测试用）。
#[cfg(test)]
fn third_row(projection: [[f64; 4]; 3]) -> [f64; 3] {
    [projection[2][0], projection[2][1], projection[2][2]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::colmap::apple::ba::CameraModelId;
    use crate::engines::colmap::apple::format::model::{Camera, Point2D};
    use crate::engines::colmap::apple::reconstruction::{ImageState, Reconstruction};

    fn camera() -> Camera {
        Camera::new(
            1,
            CameraModelId::SimpleRadial as i32,
            1000,
            800,
            vec![800.0, 500.0, 400.0, 0.0],
        )
    }

    /// 三台平移相机（非共线基线）看 5 个已知三维点，生成精确像素观测。
    ///
    /// 基线必须**非共线**，否则多视图 DLT 在基线方向退化（线性三角化的固有性质）。
    fn multi_view_fixture() -> (Reconstruction, Vec<[f64; 3]>) {
        let points: Vec<[f64; 3]> = (0..5)
            .map(|index| {
                [
                    -1.5 + 0.6 * index as f64,
                    -0.4 + 0.2 * index as f64,
                    3.5 + 0.25 * index as f64,
                ]
            })
            .collect();
        let qvec = rotation_to_quaternion(&[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let tvecs = [[0.0, 0.0, 0.0], [-0.5, -0.3, -0.2], [-0.2, 0.4, -0.3]];
        let project = |tvec: [f64; 3], xyz: [f64; 3]| -> Point2D {
            let pixel = camera::world_to_img(
                &camera(),
                CameraModelId::SimpleRadial,
                &qvec,
                &tvec,
                xyz,
            )
            .unwrap();
            Point2D::new(pixel[0], pixel[1])
        };
        let images: Vec<ImageState> = tvecs
            .iter()
            .enumerate()
            .map(|(index, tvec)| ImageState {
                id: index as u32 + 1,
                name: format!("{index}.jpg"),
                camera: 0,
                points2d: points.iter().map(|xyz| project(*tvec, *xyz)).collect(),
                colors: Vec::new(),
                registered: true,
                qvec,
                tvec: *tvec,
            })
            .collect();
        (Reconstruction::new(vec![camera()], vec![false], images), points)
    }

    #[test]
    fn two_view_triangulation_recovers_known_points() {
        let (recon, points) = multi_view_fixture();
        for (index, xyz) in points.iter().enumerate() {
            let ray1 = ray_of(&recon, 0, index as u32).unwrap();
            let ray2 = ray_of(&recon, 1, index as u32).unwrap();
            let recovered = triangulate_two_view(
                &recon.images[0].qvec,
                &recon.images[0].tvec,
                [ray1[0] / ray1[2], ray1[1] / ray1[2]],
                &recon.images[1].qvec,
                &recon.images[1].tvec,
                [ray2[0] / ray2[2], ray2[1] / ray2[2]],
            )
            .unwrap();
            // 函数内部已把符号定到"第一台相机深度为正"的那一支。
            for axis in 0..3 {
                assert!(
                    (recovered[axis] - xyz[axis]).abs() < 1e-6,
                    "axis {axis}: {} vs {}",
                    recovered[axis],
                    xyz[axis]
                );
            }
        }
    }

    #[test]
    fn triangulation_angle_matches_hand_computation() {
        // 两个视点在 x 轴上，点在中间正上方：夹角 90°。
        let angle = triangulation_angle([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
        // 共线视点 → 角度 0。
        let angle = triangulation_angle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]);
        assert!(angle < 1e-9);
    }

    #[test]
    fn multi_view_triangulation_reprojects_exactly() {
        // 3 视图（两视图时 AᵀA 零空间是 2 维的，特征向量不唯一 —— 4.2.1 因此
        // 对 2 视图走闭式解，见 `triangulate_two_view`）。
        let (recon, points) = multi_view_fixture();
        let index = 0usize;
        let xyz = points[index];
        let observations: Vec<TrackElement> = (0..3)
            .map(|image| TrackElement {
                image_id: recon.images[image].id,
                point2d_idx: index as u32,
            })
            .collect();
        let rays: Vec<[f64; 3]> = observations
            .iter()
            .map(|element| {
                let image = recon.image_index[&element.image_id];
                ray_of(&recon, image, element.point2d_idx).unwrap()
            })
            .collect();
        let projections: Vec<[[f64; 4]; 3]> = observations
            .iter()
            .map(|element| {
                let image = recon.image_index[&element.image_id];
                projection_rows(&recon.images[image].qvec, &recon.images[image].tvec)
            })
            .collect();
        let recovered = triangulate_multi_view(
            &recon.images[0].qvec,
            &recon.images[0].tvec,
            &projections,
            &rays,
        )
        .unwrap();
        // 函数内部已把符号定到"首观测相机深度为正"，直接比较。
        for axis in 0..3 {
            assert!(
                (recovered[axis] - xyz[axis]).abs() < 1e-6,
                "axis {axis}: {} vs {}",
                recovered[axis],
                xyz[axis]
            );
        }
    }

    #[test]
    fn estimate_triangulation_finds_consistent_subset() {
        let (recon, points) = multi_view_fixture();
        // 3 个观测属于同一个三维点（index 1），第 4 个是离群观测（index 3 的点）。
        let observations = vec![
            TrackElement { image_id: recon.images[0].id, point2d_idx: 1 },
            TrackElement { image_id: recon.images[1].id, point2d_idx: 1 },
            TrackElement { image_id: recon.images[2].id, point2d_idx: 1 },
            TrackElement { image_id: recon.images[0].id, point2d_idx: 3 },
        ];
        let mut rng = Rng::new(0x1234_5678);
        let (xyz, mask) = estimate_triangulation(
            &recon,
            &observations,
            1.5f64.to_radians(),
            2.0,
            true,
            &mut rng,
        )
        .expect("存在一致的三元组时应三角化成功");
        assert!(mask.iter().filter(|keep| **keep).count() >= 3, "内点掩码 {mask:?}");
        for axis in 0..3 {
            assert!(
                (xyz[axis] - points[1][axis]).abs() < 1e-6,
                "axis {axis}: {} vs {}",
                xyz[axis],
                points[1][axis]
            );
        }
    }

    #[test]
    fn estimate_triangulation_rejects_inconsistent_observations() {
        let (recon, _points) = multi_view_fixture();
        // 三个观测分属不同三维点 → 没有一致解。
        let observations = vec![
            TrackElement { image_id: recon.images[0].id, point2d_idx: 0 },
            TrackElement { image_id: recon.images[1].id, point2d_idx: 2 },
            TrackElement { image_id: recon.images[2].id, point2d_idx: 4 },
        ];
        let mut rng = Rng::new(0xabcd_ef01);
        assert!(estimate_triangulation(
            &recon,
            &observations,
            1.5f64.to_radians(),
            2.0,
            true,
            &mut rng,
        )
        .is_none());
    }

    #[test]
    fn triangulation_angle_of_third_row_is_consistent() {
        let rows = projection_rows(&[1.0, 0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        let third = third_row(rows);
        assert_eq!(third, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn point_index_of_rejects_invalid_ids() {
        use super::super::reconstruction::point_index_of;
        assert_eq!(point_index_of(0), None);
        assert_eq!(point_index_of(super::super::format::INVALID_POINT3D_ID), None);
        assert_eq!(point_index_of(7), Some(6));
    }
}