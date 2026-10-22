//! 绝对位姿估计（M5b 第一版）。
//!
//! - 最小解：6 点 DLT（Hartley 归一化）估计 `[R|t]`，替代 COLMAP 4.2.1 的
//!   P3P/EPnP/P4PF（刻意差异，见下）；LO-RANSAC 框架与 `two_view.rs` 同构。
//! - 无焦距先验时的 P4PF 路径暂缓：本项目相机恒有 DB 焦距（EXIF 或 1.2×边长），
//!   先以该焦距做 RANSAC，焦距误差交给 [`refine_absolute_pose`] 精化
//!   （`abs_pose_refine_focal_length=true`）。
//! - [`refine_absolute_pose`] 复用 M6 的 BA 引擎：单图位姿可变、点常量、
//!   Cauchy(1.0) 损失、主点恒定 —— 与 COLMAP `RefineAbsolutePose` 语义对齐。
//!
//! 刻意差异（记录在案，对应 `recon/m5-mapper.md` §4）：
//! 1. DLT-6pt 替代 P3P 最小解 + EPnP 局部优化（RANSAC 迭代上限同为 10000；
//!    内点比高时 DLT 采样开销可接受；后续可升级 P3P）。
//! 2. 焦距在 RANSAC 阶段不参与估计（P4PF 延后），依赖 LM 精化阶段的
//!    `refine_focal_length`。

use super::ba::{
    run_bundle_adjustment, BaCamera, BaImage, BaOptions, BaPoint, CameraModelId, Loss,
};

/// 动态 SVD 的迭代上限。
///
/// ⚠️ 踩坑：`matrix.svd(..)` 内部等价于 `SVD::new(..)`，而 `max_niter = 0`
/// 表示**不限迭代** —— 退化矩阵会让双对角化永不收敛（表观"挂死"）。
/// 因此这里一律用 `try_new` 并给出上限（M4 `two_view.rs` 同款约束）。
const SVD_MAX_NITER: usize = 200;
use super::linalg::{quaternion_to_rotation_matrix, rotation_to_quaternion};
use super::ba::cam_ray_from_image;
use super::two_view::{RansacOptions, Rng};
use rayon::prelude::*;

/// 一次绝对位姿估计的结果。
#[derive(Clone, Debug)]
pub struct AbsolutePoseReport {
    pub success: bool,
    /// `cam_from_world`（qvec `w,x,y,z` + tvec）。
    pub qvec: [f64; 4],
    pub tvec: [f64; 3],
    pub num_inliers: usize,
    pub inlier_mask: Vec<bool>,
}

/// 从 2D-3D 对应估计绝对位姿（LO-RANSAC + 位姿精化）。
#[allow(clippy::too_many_arguments)]
pub fn estimate_absolute_pose(
    points2d: &[[f64; 2]],
    points3d: &[[f64; 3]],
    model: CameraModelId,
    camera_params: &mut Vec<f64>,
    refine_focal_length: bool,
    refine_extra_params: bool,
    options: &RansacOptions,
    rng: &mut Rng,
) -> AbsolutePoseReport {
    let mut report = lo_ransac_pose(points2d, points3d, model, camera_params, options, rng);
    if !report.success {
        return report;
    }
    // 位姿精化（Cauchy 鲁棒损失；主点恒定；按需精化焦距/畸变）。
    let refined = refine_absolute_pose(
        &report.qvec,
        &report.tvec,
        points2d,
        points3d,
        &report.inlier_mask,
        model,
        camera_params,
        refine_focal_length,
        refine_extra_params,
    );
    if let Some((qvec, tvec, num_inliers, mask)) = refined {
        report.qvec = qvec;
        report.tvec = tvec;
        report.num_inliers = num_inliers;
        report.inlier_mask = mask;
    }
    report
}

/// 6 点 DLT 绝对位姿（Hartley 归一化）。
/// EPnP（Lepetit et al. 2009）绝对位姿 —— 移植 COLMAP 4.2.1
/// `EPNPEstimator`（`src/colmap/estimators/solvers/absolute_pose.cc`）。
///
/// 输入是**归一化光线**（`cam_ray_from_image` 的输出，等价于 COLMAP 的
/// `Point2DWithRay::camera_ray`）与对应的 3D 点。
///
/// 为什么不用 6 点线性 DLT（上一版的做法）：DLT 的 12 个未知量里平移列与世界点
/// 重心强耦合，是零空间方向里最弱的轴；带 SIMPLE_RADIAL 畸变的观测会让残差几乎全部
/// 压到这一列上，表现为「旋转恢复正确、平移差一个任意倍数」。EPnP 只用 4 个控制点
/// （每点 2 行、每行只耦合 4 个未知量），平移最终由 Kabsch（对位姿 SVD）拟合给出，
/// 结构上就没有这个问题。COLMAP 的 LO-RANSAC 局部估计器同样用 EPNP。
///
/// 4.2.1 的最小采样是 P3P（`poselib::p3p`），我们暂用 6 点 EPNP 作假设生成器；
/// 见 `docs/project-notes/ooo-sfm-implementation-plan.md` 的差异清单。
fn epnp_absolute_pose(
    rays: &[[f64; 3]],
    points3d: &[[f64; 3]],
) -> Option<([f64; 4], [f64; 3])> {
    let n = points3d.len();
    if n < 4 || rays.len() != n {
        return None;
    }
    if rays
        .iter()
        .chain(points3d.iter())
        .any(|value| value.iter().any(|component| !component.is_finite()))
    {
        return None;
    }
    let cws = choose_control_points(points3d)?;
    let alphas = barycentric_coordinates(&cws, points3d)?;

    // M（3n × 12）：每个点 3 行，每行只耦合 4 个控制点分量。
    let mut m = nalgebra::DMatrix::<f64>::zeros(3 * n, 12);
    for i in 0..n {
        let ray = rays[i];
        for j in 0..4 {
            let alpha = alphas[i][j];
            m[(3 * i, 3 * j)] = 0.0;
            m[(3 * i, 3 * j + 1)] = -alpha * ray[2];
            m[(3 * i, 3 * j + 2)] = alpha * ray[1];
            m[(3 * i + 1, 3 * j)] = alpha * ray[2];
            m[(3 * i + 1, 3 * j + 1)] = 0.0;
            m[(3 * i + 1, 3 * j + 2)] = -alpha * ray[0];
            m[(3 * i + 2, 3 * j)] = -alpha * ray[1];
            m[(3 * i + 2, 3 * j + 1)] = alpha * ray[0];
            m[(3 * i + 2, 3 * j + 2)] = 0.0;
        }
    }

    // MᵀM 的零空间（12×12）→ 控制点在相机系的坐标。
    let mt_m = m.transpose() * m;
    let svd = nalgebra::linalg::SVD::try_new(
        mt_m,
        true,
        true,
        5.0 * f64::EPSILON,
        SVD_MAX_NITER,
    )?;
    let u = svd.u?;

    let l6x10 = compute_l6x10(&u);
    let rho = compute_rho(&cws);

    // 三组 betas 初值（1/2/3 控制点项），每组跑 5 步 Gauss–Newton，取归一化射线
    // 空间重投影误差最小者 —— 与 4.2.1 `ComputeRT` 后的择优逻辑一致。
    let mut best: Option<(f64, [[f64; 3]; 3], [f64; 3])> = None;
    for approx in 1..=3u32 {
        let mut betas = find_betas(&l6x10, &rho, approx)?;
        run_betas_gauss_newton(&l6x10, &rho, &mut betas);
        let (rotation, translation) = estimate_rt_from_betas(&u, &betas, &alphas, points3d);
        if rotation.iter().flatten().any(|value| !value.is_finite())
            || translation.iter().any(|value| !value.is_finite())
        {
            continue;
        }
        let error = normalized_ray_error(&rotation, &translation, rays, points3d);
        if best.as_ref().is_none_or(|(best_error, _, _)| error < *best_error) {
            best = Some((error, rotation, translation));
        }
    }
    let (_, rotation, translation) = best?;
    Some((rotation_to_quaternion(&rotation), translation))
}

/// `ChooseControlPoints`（4.2.1）：`cws[0]` 取重心，其余三个沿协方差的主轴。
fn choose_control_points(points3d: &[[f64; 3]]) -> Option<[[f64; 3]; 4]> {
    let n = points3d.len() as f64;
    let mut centroid = [0.0f64; 3];
    for point in points3d {
        for axis in 0..3 {
            centroid[axis] += point[axis];
        }
    }
    for value in centroid.iter_mut() {
        *value /= n;
    }
    let mut covariance = nalgebra::Matrix3::<f64>::zeros();
    for point in points3d {
        let d = [
            point[0] - centroid[0],
            point[1] - centroid[1],
            point[2] - centroid[2],
        ];
        for (r, dr) in d.iter().enumerate() {
            for (c, dc) in d.iter().enumerate() {
                covariance[(r, c)] += dr * dc;
            }
        }
    }
    let svd = nalgebra::linalg::SVD::try_new(
        covariance,
        true,
        true,
        5.0 * f64::EPSILON,
        SVD_MAX_NITER,
    )?;
    let u = svd.u?;
    let mut control = [[0.0f64; 3]; 4];
    control[0] = centroid;
    for i in 1..4 {
        let scale = (svd.singular_values[i - 1] / n).sqrt();
        if !scale.is_finite() {
            return None;
        }
        for axis in 0..3 {
            control[i][axis] = centroid[axis] + scale * u[(axis, i - 1)];
        }
    }
    Some(control)
}

/// `ComputeBarycentricCoordinates`（4.2.1）：控制点退化（共线/共面且秩 < 3）时返回 `None`。
fn barycentric_coordinates(
    control: &[[f64; 3]; 4],
    points3d: &[[f64; 3]],
) -> Option<Vec<[f64; 4]>> {
    let mut cc = nalgebra::Matrix3::<f64>::zeros();
    for row in 0..3 {
        for column in 0..3 {
            cc[(row, column)] = control[column + 1][row] - control[0][row];
        }
    }
    let cc_inverse = cc.try_inverse()?;
    if cc_inverse.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut alphas = Vec::with_capacity(points3d.len());
    for point in points3d {
        let delta = [
            point[0] - control[0][0],
            point[1] - control[0][1],
            point[2] - control[0][2],
        ];
        let mut alpha = [0.0f64; 4];
        for j in 0..3 {
            alpha[j + 1] = cc_inverse[(j, 0)] * delta[0]
                + cc_inverse[(j, 1)] * delta[1]
                + cc_inverse[(j, 2)] * delta[2];
        }
        alpha[0] = 1.0 - alpha[1] - alpha[2] - alpha[3];
        alphas.push(alpha);
    }
    Some(alphas)
}

/// `ComputeL6x10`（4.2.1）：10 个二阶项的方程系数。
fn compute_l6x10(u: &nalgebra::DMatrix<f64>) -> [[f64; 10]; 6] {
    // Ut(11 - i, 3 * j + k) = Uᵀ 的元素 = U(3 * j + k, 11 - i)。
    let ut = |i: usize, j: usize, k: usize| -> f64 { u[(3 * j + k, 11 - i)] };
    let mut dv = [[[0.0f64; 3]; 6]; 4];
    for i in 0..4 {
        let mut a = 0usize;
        let mut b = 1usize;
        for slot in dv[i].iter_mut() {
            for k in 0..3 {
                slot[k] = ut(i, a, k) - ut(i, b, k);
            }
            b += 1;
            if b > 3 {
                a += 1;
                b = a + 1;
            }
        }
    }
    let mut l6x10 = [[0.0f64; 10]; 6];
    for i in 0..6 {
        let dot = |x: &[f64; 3], y: &[f64; 3]| -> f64 {
            (0..3).map(|k| x[k] * y[k]).sum()
        };
        l6x10[i] = [
            dot(&dv[0][i], &dv[0][i]),
            2.0 * dot(&dv[0][i], &dv[1][i]),
            dot(&dv[1][i], &dv[1][i]),
            2.0 * dot(&dv[0][i], &dv[2][i]),
            2.0 * dot(&dv[1][i], &dv[2][i]),
            dot(&dv[2][i], &dv[2][i]),
            2.0 * dot(&dv[0][i], &dv[3][i]),
            2.0 * dot(&dv[1][i], &dv[3][i]),
            2.0 * dot(&dv[2][i], &dv[3][i]),
            dot(&dv[3][i], &dv[3][i]),
        ];
    }
    l6x10
}

/// `ComputeRho`（4.2.1）：6 个控制点两两距离平方。
fn compute_rho(control: &[[f64; 3]; 4]) -> [f64; 6] {
    let mut rho = [0.0f64; 6];
    let mut index = 0;
    for i in 0..4 {
        for j in i + 1..4 {
            rho[index] = (0..3)
                .map(|axis| {
                    let d = control[i][axis] - control[j][axis];
                    d * d
                })
                .sum();
            index += 1;
        }
    }
    rho
}

/// 最小二乘解 `min ‖A x − b‖`（对秩亏/病态系统取伪逆解）。
fn least_squares(a: &nalgebra::DMatrix<f64>, b: &nalgebra::DVector<f64>) -> Option<nalgebra::DVector<f64>> {
    let svd = nalgebra::linalg::SVD::try_new(a.clone(), true, true, 1e-12, SVD_MAX_NITER)?;
    svd.solve(b, 1e-12).ok()
}

/// `FindBetasApprox1/2/3`（4.2.1）：三组不同项数的近似解，作为 Gauss–Newton 初值。
fn find_betas(l6x10: &[[f64; 10]; 6], rho: &[f64; 6], approx: u32) -> Option<[f64; 4]> {
    let rho_vector = nalgebra::DVector::from_iterator(rho.len(), rho.iter().copied());
    let columns: &[usize] = match approx {
        1 => &[0, 1, 3, 6],
        2 => &[0, 1, 2],
        _ => &[0, 1, 2, 3, 4],
    };
    let mut a = nalgebra::DMatrix::<f64>::zeros(6, columns.len());
    for row in 0..6 {
        for (j, column) in columns.iter().enumerate() {
            a[(row, j)] = l6x10[row][*column];
        }
    }
    let solution = least_squares(&a, &rho_vector)?;

    let mut betas = [0.0f64; 4];
    match approx {
        1 => {
            let first = solution[0];
            if first < 0.0 {
                betas[0] = (-first).sqrt();
                for j in 0..3 {
                    betas[j + 1] = -solution[j + 1] / betas[0];
                }
            } else {
                betas[0] = first.sqrt();
                for j in 0..3 {
                    betas[j + 1] = solution[j + 1] / betas[0];
                }
            }
        }
        2 => {
            if solution[0] < 0.0 {
                betas[0] = (-solution[0]).sqrt();
                betas[1] = if solution[2] < 0.0 { (-solution[2]).sqrt() } else { 0.0 };
            } else {
                betas[0] = solution[0].sqrt();
                betas[1] = if solution[2] > 0.0 { solution[2].sqrt() } else { 0.0 };
            }
            if solution[1] < 0.0 {
                betas[0] = -betas[0];
            }
        }
        _ => {
            if solution[0] < 0.0 {
                betas[0] = (-solution[0]).sqrt();
                betas[1] = if solution[2] < 0.0 { (-solution[2]).sqrt() } else { 0.0 };
            } else {
                betas[0] = solution[0].sqrt();
                betas[1] = if solution[2] > 0.0 { solution[2].sqrt() } else { 0.0 };
            }
            if solution[1] < 0.0 {
                betas[0] = -betas[0];
            }
            betas[2] = solution[3] / betas[0];
        }
    }
    if betas.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some(betas)
}

/// `RunGaussNewton`（4.2.1）：5 步 Gauss–Newton 精化 betas。
fn run_betas_gauss_newton(l6x10: &[[f64; 10]; 6], rho: &[f64; 6], betas: &mut [f64; 4]) {
    let mut a = nalgebra::DMatrix::<f64>::zeros(6, 4);
    let mut b = nalgebra::DVector::<f64>::zeros(6);
    for _ in 0..5 {
        for i in 0..6 {
            let l = &l6x10[i];
            a[(i, 0)] = 2.0 * l[0] * betas[0] + l[1] * betas[1] + l[3] * betas[2] + l[6] * betas[3];
            a[(i, 1)] = l[1] * betas[0] + 2.0 * l[2] * betas[1] + l[4] * betas[2] + l[7] * betas[3];
            a[(i, 2)] = l[3] * betas[0] + l[4] * betas[1] + 2.0 * l[5] * betas[2] + l[8] * betas[3];
            a[(i, 3)] = l[6] * betas[0] + l[7] * betas[1] + l[8] * betas[2] + 2.0 * l[9] * betas[3];
            b[i] = rho[i]
                - (l[0] * betas[0] * betas[0]
                    + l[1] * betas[0] * betas[1]
                    + l[2] * betas[1] * betas[1]
                    + l[3] * betas[0] * betas[2]
                    + l[4] * betas[1] * betas[2]
                    + l[5] * betas[2] * betas[2]
                    + l[6] * betas[0] * betas[3]
                    + l[7] * betas[1] * betas[3]
                    + l[8] * betas[2] * betas[3]
                    + l[9] * betas[3] * betas[3]);
        }
        let Some(step) = least_squares(&a, &b) else {
            return;
        };
        for j in 0..4 {
            betas[j] += step[j];
        }
        if betas.iter().any(|value| !value.is_finite()) {
            return;
        }
    }
}

/// `ComputeCcs` + `ComputePcs` + `SolveForSign` + `EstimateRT`（4.2.1）。
fn estimate_rt_from_betas(
    u: &nalgebra::DMatrix<f64>,
    betas: &[f64; 4],
    alphas: &[[f64; 4]],
    points3d: &[[f64; 3]],
) -> ([[f64; 3]; 3], [f64; 3]) {
    let mut control_camera = [[0.0f64; 3]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..3 {
                control_camera[j][k] += betas[i] * u[(3 * j + k, 11 - i)];
            }
        }
    }
    let mut points_camera: Vec<[f64; 3]> = Vec::with_capacity(points3d.len());
    for alpha in alphas {
        let mut point = [0.0f64; 3];
        for (k, control) in control_camera.iter().enumerate() {
            for axis in 0..3 {
                point[axis] += alpha[k] * control[axis];
            }
        }
        points_camera.push(point);
    }
    // 深度约定：第一个点必须在相机前方。
    if points_camera[0][2] < 0.0 {
        for control in control_camera.iter_mut() {
            for value in control.iter_mut() {
                *value = -*value;
            }
        }
        for point in points_camera.iter_mut() {
            for value in point.iter_mut() {
                *value = -*value;
            }
        }
    }
    // Kabsch：最小化 Σ |R·(X − X̄) − (Y − Ȳ)|² 的刚体旋转。
    let n = points3d.len() as f64;
    let mut mean_camera = [0.0f64; 3];
    let mut mean_world = [0.0f64; 3];
    for i in 0..points3d.len() {
        for axis in 0..3 {
            mean_camera[axis] += points_camera[i][axis];
            mean_world[axis] += points3d[i][axis];
        }
    }
    for axis in 0..3 {
        mean_camera[axis] /= n;
        mean_world[axis] /= n;
    }
    let mut abt = nalgebra::Matrix3::<f64>::zeros();
    for i in 0..points3d.len() {
        let camera = [
            points_camera[i][0] - mean_camera[0],
            points_camera[i][1] - mean_camera[1],
            points_camera[i][2] - mean_camera[2],
        ];
        let world = [
            points3d[i][0] - mean_world[0],
            points3d[i][1] - mean_world[1],
            points3d[i][2] - mean_world[2],
        ];
        for (r, dr) in camera.iter().enumerate() {
            for (c, dc) in world.iter().enumerate() {
                abt[(r, c)] += dr * dc;
            }
        }
    }
    // R = U·Vᵀ；det < 0（反射）时把 V 的第三列取反。
    let mut rotation = [[0.0f64; 3]; 3];
    if let Some(svd) = nalgebra::linalg::SVD::try_new(abt, true, true, 1e-12, SVD_MAX_NITER) {
        if let (Some(left), Some(right)) = (svd.u, svd.v_t) {
            // tr(U·Vᵀ) < 0 说明是反射，翻转 V 的第三列（4.2.1 同款修正）。
            let trace = (0..3)
                .map(|r| {
                    (0..3)
                        .map(|c| (0..3).map(|k| left[(r, k)] * right[(k, c)]).sum::<f64>())
                        .sum::<f64>()
                })
                .sum::<f64>();
            let flip = if trace < 0.0 { -1.0 } else { 1.0 };
            for r in 0..3 {
                for c in 0..3 {
                    let factor = if c == 2 { flip } else { 1.0 };
                    rotation[r][c] = (0..3)
                        .map(|k| left[(r, k)] * right[(k, c)] * factor)
                        .sum();
                }
            }
        }
    }
    let mut translation = [0.0f64; 3];
    for r in 0..3 {
        let mut sum = mean_camera[r];
        for k in 0..3 {
            sum -= rotation[r][k] * mean_world[k];
        }
        translation[r] = sum;
    }
    (rotation, translation)
}



/// 归一化光线空间的重投影误差（角度域），用于三组 betas 择优。
fn normalized_ray_error(
    rotation: &[[f64; 3]; 3],
    translation: &[f64; 3],
    rays: &[[f64; 3]],
    points3d: &[[f64; 3]],
) -> f64 {
    let mut error = 0.0;
    for i in 0..points3d.len() {
        let point = points3d[i];
        let mut projected = [0.0f64; 3];
        for r in 0..3 {
            projected[r] = rotation[r][0] * point[0]
                + rotation[r][1] * point[1]
                + rotation[r][2] * point[2]
                + translation[r];
        }
        let norm = (projected[0].powi(2) + projected[1].powi(2) + projected[2].powi(2)).sqrt();
        let ray_norm =
            (rays[i][0].powi(2) + rays[i][1].powi(2) + rays[i][2].powi(2)).sqrt();
        if !(norm > 1e-12) || !(ray_norm > 1e-12) {
            return f64::INFINITY;
        }
        let dot = (0..3)
            .map(|axis| projected[axis] * rays[i][axis])
            .sum::<f64>()
            / (norm * ray_norm);
        let sin = (1.0 - dot * dot).max(0.0).sqrt();
        error += sin * sin;
    }
    error
}


/// 完整投影（K[R|t]·X），K 从相机模型参数构造。
fn project_with_model(model: CameraModelId, params: &[f64], rotation: &[[f64; 3]; 3], t: &[f64; 3], x: [f64; 3]) -> Option<[f64; 2]> {
    let u = [
        rotation[0][0] * x[0] + rotation[0][1] * x[1] + rotation[0][2] * x[2] + t[0],
        rotation[1][0] * x[0] + rotation[1][1] * x[1] + rotation[1][2] * x[2] + t[1],
        rotation[2][0] * x[0] + rotation[2][1] * x[1] + rotation[2][2] * x[2] + t[2],
    ];
    if u[2] < 1e-9 {
        return None;
    }
    let z_safe = u[2].max(1e-9);
    let (fx, fy, cx, cy, k1, k2) = match model {
        CameraModelId::SimplePinhole => (params[0], params[0], params[1], params[2], 0.0, 0.0),
        CameraModelId::Pinhole => (params[0], params[1], params[2], params[3], 0.0, 0.0),
        CameraModelId::SimpleRadial => (params[0], params[0], params[1], params[2], params[3], 0.0),
        CameraModelId::Radial => (params[0], params[0], params[1], params[2], params[3], params[4]),
    };
    let xp = u[0] / z_safe;
    let yp = u[1] / z_safe;
    let r2 = xp * xp + yp * yp;
    let scale = 1.0 + k1 * r2 + k2 * r2 * r2;
    Some([fx * xp * scale + cx, fy * yp * scale + cy])
}

/// RANSAC 最小采样点数（EPnP 的下限是 4，取 6 与 COLMAP `kAbsolutePoseMinNumSize` 一致）。
const RANSAC_SAMPLE_SIZE: usize = 6;
/// 每轮并行评估的假设数。COLMAP 用 OpenMP 的 `ParallelFor` 同样把迭代切成批。
const RANSAC_BATCH: usize = 64;

/// MSAC 支持度：返回 `(内点掩码, 内点数, 截断平方误差和)`。
fn score_pose(
    rotation: &[[f64; 3]; 3],
    tvec: &[f64; 3],
    points2d: &[[f64; 2]],
    points3d: &[[f64; 3]],
    model: CameraModelId,
    camera_params: &[f64],
    max_residual: f64,
) -> (Vec<bool>, usize, f64) {
    let mut inliers = vec![false; points2d.len()];
    let mut count = 0usize;
    let mut score = 0.0;
    for i in 0..points2d.len() {
        let residual = project_with_model(model, camera_params, rotation, tvec, points3d[i]).map(
            |projection| {
                let dx = projection[0] - points2d[i][0];
                let dy = projection[1] - points2d[i][1];
                dx * dx + dy * dy
            },
        );
        match residual {
            Some(r2) if r2 <= max_residual => {
                inliers[i] = true;
                count += 1;
                score += r2;
            }
            _ => score += max_residual,
        }
    }
    (inliers, count, score)
}

/// LO-RANSAC 主循环（MSAC 支持度 + 内点集上 EPnP 局部重估）。
///
/// CPU 加速：畸变反投影（`cam_ray_from_image`）在循环外只做一次；每 `RANSAC_BATCH`
/// 个假设用 rayon 并行求解与打分（随机采样仍按顺序推进，种子决定的结果保持确定）。
fn lo_ransac_pose(
    points2d: &[[f64; 2]],
    points3d: &[[f64; 3]],
    model: CameraModelId,
    camera_params: &[f64],
    options: &RansacOptions,
    rng: &mut Rng,
) -> AbsolutePoseReport {
    let mut failed = AbsolutePoseReport {
        success: false,
        qvec: [1.0, 0.0, 0.0, 0.0],
        tvec: [0.0; 3],
        num_inliers: 0,
        inlier_mask: Vec::new(),
    };
    let n = points2d.len();
    if n < RANSAC_SAMPLE_SIZE || points3d.len() != n {
        return failed;
    }
    let max_residual = options.max_error * options.max_error;
    // 自适应迭代数（与 two_view.rs 同款公式）。
    let compute_trials = |inlier_ratio: f64| -> usize {
        let denominator = 1.0 - (1.0 - inlier_ratio).powi(RANSAC_SAMPLE_SIZE as i32);
        if denominator <= 1e-12 {
            return options.max_num_trials;
        }
        let numer = (1.0 - options.confidence).ln();
        let trials = numer / denominator.ln();
        if !trials.is_finite() || trials <= 0.0 {
            return options.max_num_trials;
        }
        (trials.ceil() as usize).min(options.max_num_trials)
    };

    // 畸变反投影只做一次：EPnP、局部重估、择优全部复用这批光线。
    let mut kept: Vec<usize> = Vec::with_capacity(n);
    let mut rays: Vec<[f64; 3]> = Vec::with_capacity(n);
    let mut kept2d: Vec<[f64; 2]> = Vec::with_capacity(n);
    let mut kept3d: Vec<[f64; 3]> = Vec::with_capacity(n);
    for i in 0..n {
        let Some(ray) = cam_ray_from_image(model, camera_params, points2d[i][0], points2d[i][1])
        else {
            continue;
        };
        if !(ray[2] > 0.0) || ray.iter().any(|value| !value.is_finite()) {
            continue;
        }
        kept.push(i);
        rays.push(ray);
        kept2d.push(points2d[i]);
        kept3d.push(points3d[i]);
    }
    let m = rays.len();
    if m < RANSAC_SAMPLE_SIZE {
        return failed;
    }

    let mut best_count = 0usize;
    let mut best_score = f64::INFINITY;
    let mut best_model: Option<([f64; 4], [f64; 3])> = None;
    let mut max_trials = options.max_num_trials;
    let mut trial = 0usize;

    while trial < max_trials {
        let take = RANSAC_BATCH.min(max_trials - trial);
        // 顺序采样（保证固定种子 → 结果确定），并行求解与打分。
        let mut samples: Vec<[usize; RANSAC_SAMPLE_SIZE]> = Vec::with_capacity(take);
        for _ in 0..take {
            let mut sample = [0usize; RANSAC_SAMPLE_SIZE];
            let mut filled = 0usize;
            while filled < RANSAC_SAMPLE_SIZE {
                let candidate = (rng.next_u64() as usize) % m;
                if !sample[..filled].contains(&candidate) {
                    sample[filled] = candidate;
                    filled += 1;
                }
            }
            samples.push(sample);
        }
        trial += take;

        let hypotheses: Vec<Option<([f64; 4], [f64; 3], usize, f64)>> = samples
            .par_iter()
            .map(|sample| {
                let sample_rays: Vec<[f64; 3]> = sample.iter().map(|&i| rays[i]).collect();
                let sample_xyz: Vec<[f64; 3]> = sample.iter().map(|&i| kept3d[i]).collect();
                let (qvec, tvec) = epnp_absolute_pose(&sample_rays, &sample_xyz)?;
                let rotation = quaternion_to_rotation_matrix(&qvec);
                let (_, count, score) = score_pose(
                    &rotation,
                    &tvec,
                    &kept2d,
                    &kept3d,
                    model,
                    camera_params,
                    max_residual,
                );
                Some((qvec, tvec, count, score))
            })
            .collect();

        for hypothesis in hypotheses.into_iter().flatten() {
            let (qvec, tvec, count, score) = hypothesis;
            if count < RANSAC_SAMPLE_SIZE {
                continue;
            }
            if count > best_count || (count == best_count && score < best_score) {
                best_count = count;
                best_score = score;
                best_model = Some((qvec, tvec));
                max_trials = compute_trials((count as f64 / m as f64).max(1e-6));
            }
        }
    }

    let Some((mut qvec, mut tvec)) = best_model else {
        return failed;
    };
    let (mut best_inliers, _, _) = score_pose(
        &quaternion_to_rotation_matrix(&qvec),
        &tvec,
        &kept2d,
        &kept3d,
        model,
        camera_params,
        max_residual,
    );

    // 局部优化：内点集上重跑 EPnP（与 4.2.1 LO-RANSAC 的 local estimator 同款），
    // 最多 10 轮，内点数不降且误差和不升即接受。
    for _ in 0..10 {
        let inlier_rays: Vec<[f64; 3]> = (0..m)
            .filter(|&i| best_inliers[i])
            .map(|i| rays[i])
            .collect();
        let inlier_xyz: Vec<[f64; 3]> = (0..m)
            .filter(|&i| best_inliers[i])
            .map(|i| kept3d[i])
            .collect();
        let Some((q_refined, t_refined)) = epnp_absolute_pose(&inlier_rays, &inlier_xyz) else {
            break;
        };
        let rotation = quaternion_to_rotation_matrix(&q_refined);
        let (refined_inliers, refined_count, refined_score) = score_pose(
            &rotation,
            &t_refined,
            &kept2d,
            &kept3d,
            model,
            camera_params,
            max_residual,
        );
        if refined_count >= best_count && refined_score <= best_score {
            best_count = refined_count;
            best_score = refined_score;
            best_inliers = refined_inliers;
            qvec = q_refined;
            tvec = t_refined;
            if refined_count == m {
                break;
            }
        } else {
            break;
        }
    }

    let mut inlier_mask = vec![false; n];
    for (j, &original) in kept.iter().enumerate() {
        inlier_mask[original] = best_inliers[j];
    }
    failed.success = true;
    failed.qvec = qvec;
    failed.tvec = tvec;
    failed.num_inliers = best_count;
    failed.inlier_mask = inlier_mask;
    failed
}

/// 位姿精化（复用 M6 BA 引擎：Cauchy(1.0)、主点恒定、点常量）。
/// 返回 `(qvec, tvec, num_inliers, inlier_mask)`；失败返回 `None`（保持原位姿）。
#[allow(clippy::too_many_arguments)]
fn refine_absolute_pose(
    qvec: &[f64; 4],
    tvec: &[f64; 3],
    points2d: &[[f64; 2]],
    points3d: &[[f64; 3]],
    inlier_mask: &[bool],
    model: CameraModelId,
    camera_params: &mut [f64],
    refine_focal_length: bool,
    refine_extra_params: bool,
) -> Option<([f64; 4], [f64; 3], usize, Vec<bool>)> {
    let model_id = match model {
        CameraModelId::SimplePinhole => 0,
        CameraModelId::Pinhole => 1,
        CameraModelId::SimpleRadial => 2,
        CameraModelId::Radial => 3,
    };
    let mut camera = BaCamera {
        id: 0,
        model_id,
        params: camera_params.to_vec(),
        constant: false,
    };
    let mut images = vec![BaImage {
        id: 0,
        camera: 0,
        qvec: *qvec,
        tvec: *tvec,
        constant_pose: false,
    }];
    let mut points: Vec<BaPoint> = (0..points2d.len())
        .filter(|&i| inlier_mask[i])
        .map(|i| BaPoint {
            xyz: points3d[i],
            observations: vec![(0, points2d[i][0], points2d[i][1])],
            constant: true,
        })
        .collect();
    if points.is_empty() {
        return None;
    }
    let options = BaOptions {
        loss: Loss::Cauchy(1.0),
        max_iterations: 100,
        refine_focal_length,
        refine_principal_point: false,
        refine_extra_params,
        gauge: super::ba::Gauge::None,
        function_tolerance: 0.0,
        gradient_tolerance: 1.0,
    };
    run_bundle_adjustment(std::slice::from_mut(&mut camera), &mut images, &mut points, &options);

    // 用精化后的位姿/内参重新计算内点。
    let rotation = quaternion_to_rotation_matrix(&images[0].qvec);
    let max_residual = 12.0_f64 * 12.0; // abs_pose_max_error
    let mut mask = vec![false; points2d.len()];
    let mut count = 0usize;
    for i in 0..points2d.len() {
        if let Some(proj) = project_with_model(model, &camera.params, &rotation, &images[0].tvec, points3d[i]) {
            let dx = proj[0] - points2d[i][0];
            let dy = proj[1] - points2d[i][1];
            if dx * dx + dy * dy <= max_residual {
                mask[i] = true;
                count += 1;
            }
        }
    }
    camera_params.copy_from_slice(&camera.params);
    Some((images[0].qvec, images[0].tvec, count, mask))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::linalg::{mat3_mat3, exp_rotation_vector};

    fn make_observations(
        qvec: [f64; 4],
        tvec: [f64; 3],
        params: &[f64],
        points3d: &[[f64; 3]],
    ) -> Vec<[f64; 2]> {
        let rotation = quaternion_to_rotation_matrix(&qvec);
        points3d
            .iter()
            .map(|&x| project_with_model(CameraModelId::SimpleRadial, params, &rotation, &tvec, x).expect("点在前方"))
            .collect()
    }

    /// EPnP 在带 SIMPLE_RADIAL 畸变 + 无噪声的观测上必须恢复出精确位姿。
    ///
    /// 这是上一版 6 点线性 DLT 失败的那个场景（旋转对、平移差一个倍数），
    /// 现在由 EPnP 接手：平移来自 Kabsch 拟合，没有零空间任意尺度的问题。
    #[test]
    fn epnp_recovers_known_pose_from_clean_data() {
        let params = vec![800.0, 500.0, 375.0, 0.01];
        // 点云在相机前方（z ∈ [2, 6]），避免深度为负。
        let points3d: Vec<[f64; 3]> = (0..30)
            .map(|i| {
                let t = i as f64 * 0.7;
                [1.5 * t.sin(), 1.2 * (t * 0.6).cos(), 4.0 + 2.0 * t.cos()]
            })
            .collect();
        let qvec = super::super::linalg::rotation_vector_to_quaternion(&[0.05, -0.04, 0.03]);
        let tvec = [0.2, -0.1, 0.15];
        let points2d = make_observations(qvec, tvec, &params, &points3d);
        let rays: Vec<[f64; 3]> = points2d
            .iter()
            .map(|p| cam_ray_from_image(CameraModelId::SimpleRadial, &params, p[0], p[1]).expect("光线"))
            .collect();

        let (q_est, t_est) = epnp_absolute_pose(&rays, &points3d).expect("EPnP 成功");
        // 旋转矩阵差（四元数双覆盖，比较矩阵）。
        let r_true = quaternion_to_rotation_matrix(&qvec);
        let r_est = quaternion_to_rotation_matrix(&q_est);
        let mut max_err = 0.0f64;
        for i in 0..3 {
            for j in 0..3 {
                max_err = max_err.max((r_true[i][j] - r_est[i][j]).abs());
            }
        }
        assert!(max_err < 1e-6, "旋转恢复失败：{max_err}");
        for axis in 0..3 {
            assert!(
                (t_est[axis] - tvec[axis]).abs() < 1e-4,
                "平移恢复失败：{t_est:?} vs {tvec:?}"
            );
        }
    }

    #[test]
    fn absolute_pose_survives_outliers() {
        let params = vec![800.0, 500.0, 375.0, 0.01];
        let mut points3d: Vec<[f64; 3]> = (0..80)
            .map(|i| {
                let t = i as f64 * 0.37;
                [1.6 * t.sin(), 1.1 * (t * 0.53).cos(), 4.0 + 2.5 * (t * 0.21).cos()]
            })
            .collect();
        // 用 ground truth 位姿造观测。
        let qvec_true = super::super::linalg::rotation_vector_to_quaternion(&[0.04, 0.03, -0.02]);
        let tvec_true = [0.1, 0.05, -0.08];
        let mut points2d = make_observations(qvec_true, tvec_true, &params, &points3d);

        // 注入 20% 离群点：**观测**随机化（3D 点保持不动）。
        //
        // 注意不能改成"随机挪动 3D 点再用真值位姿重投影"——那样 2D 观测与新的 3D 点
        // 仍然几何自洽，反而是合法内点（旧夹具的 bug：`离群点泄漏` 断言必然失败）。
        let mut rng = Rng::new(7);
        let num_outliers = points3d.len() / 5;
        for k in 0..num_outliers {
            let unit = |rng: &mut Rng| ((rng.next_u64() >> 11) as f64) * (1.0 / (1u64 << 53) as f64);
            points2d[k] = [40.0 + 920.0 * unit(&mut rng), 30.0 + 690.0 * unit(&mut rng)];
        }
        let mut camera_params = params.clone();
        let options = RansacOptions {
            max_error: 12.0,
            confidence: 0.9999,
            min_inlier_ratio: 0.25,
            max_num_trials: 10000,
            ..RansacOptions::default()
        };
        let mut rng = Rng::new(42);
        let report = estimate_absolute_pose(
            &points2d,
            &points3d,
            CameraModelId::SimpleRadial,
            &mut camera_params,
            true,
            true,
            &options,
            &mut rng,
        );
        assert!(report.success, "位姿估计应成功");
        assert!(
            report.num_inliers >= (points3d.len() - num_outliers) - 3,
            "内点数应≈非离群数：{} (期望 ≥ {})",
            report.num_inliers,
            points3d.len() - num_outliers
        );
        // 位姿恢复（比较旋转矩阵与平移）。
        let r_true = quaternion_to_rotation_matrix(&qvec_true);
        let r_est = quaternion_to_rotation_matrix(&report.qvec);
        let mut max_err = 0.0f64;
        for i in 0..3 {
            for j in 0..3 {
                max_err = max_err.max((r_true[i][j] - r_est[i][j]).abs());
            }
        }
        assert!(max_err < 1e-3, "旋转恢复失败：{max_err}");
        for axis in 0..3 {
            assert!(
                (report.tvec[axis] - tvec_true[axis]).abs() < 5e-3,
                "平移恢复失败：{:?} vs {:?}",
                report.tvec,
                tvec_true
            );
        }
        // 内点掩码应排除离群点（前 num_outliers 个）。
        let outlier_leak = report.inlier_mask[..num_outliers].iter().filter(|&&v| v).count();
        assert!(outlier_leak <= 2, "离群点泄漏过多：{outlier_leak}");
    }

    #[test]
    fn insufficient_points_fail_cleanly() {
        let points2d = [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0]];
        let points3d = [[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [1.0, 1.0, 5.0]];
        let options = RansacOptions::default();
        let mut rng = Rng::new(1);
        let mut camera_params = vec![800.0, 500.0, 375.0, 0.01];
        let report = estimate_absolute_pose(
            &points2d,
            &points3d,
            CameraModelId::SimpleRadial,
            &mut camera_params,
            false,
            false,
            &options,
            &mut rng,
        );
        assert!(!report.success);
        let _ = mat3_mat3;
        let _ = exp_rotation_vector;
    }


}