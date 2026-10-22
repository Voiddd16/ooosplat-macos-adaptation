//! M5d 支撑：初始化的两视图相对位姿（`EstimateTwoViewGeometryPose` 等价）。
//!
//! 对齐 COLMAP 4.2.1 `estimators/two_view_geometry.cc:890-1042`：
//! - `UNCALIBRATED`：先用焦距把 F 转成 E，再做 4 解 cheirality 分解；
//! - `PLANAR_OR_PANORAMIC`：分解单应得到 4 个"旋转 + 平移 + 平面法向"，用
//!   cheirality + 正深度挑一个（`|t|² < 1e-12` → PANORAMIC）；
//! - `tri_angle`：存活对应光线的夹角**中位数**。
//!
//! 刻意差异（记录在案，对应 `recon/m5-mapper.md` §5）：
//! 4.2.1 在**无焦距先验**时用 6 点共享焦距最小解（`solvers/relpose_shared_focal.cc`）
//! 在 LO-RANSAC 内联合估焦距与相对位姿。我们**不估焦距**：直接用相机当前焦距
//! （EXIF 先验或 `1.2 × max(w,h)` 默认值）算 `E = K₂ᵀFK₁` 再做 cheirality 分解 ——
//! 即 COLMAP ≤3.7 的做法，焦距随后由 `abs_pose_refine_focal_length` 的 PnP 精化
//! 与 `ba_refine_focal_length` 的全局 BA 逐步优化。
//!
//! 为什么不用"1D 搜索焦距"：Sampson 残差对焦距的**统一缩放是不变量**（分子分母同乘
//! f²），而以 cheirality 内点数为目标在**前视相机**（视频帧的典型情形）下随焦距
//! 变化极平（实测 800 真值处与 89/414 处内点数几乎相同），无法定位焦距。
//! 要可靠复刻只能移植 6 点联合最小解，收益（初值精度）不足以抵消复杂度 ——
//! 留待 M11 与 COLMAP 交叉验证时再评估。

use super::camera::{self, cam_point_from_img};
use super::format::model::Camera;
use super::linalg::{
    mat3_vec3, quaternion_to_rotation_matrix, rotation_to_quaternion, transpose3,
};
use super::triangulator::triangulate_two_view;
use super::two_view::{DEGENERATE, PLANAR_OR_PANORAMIC, UNCALIBRATED};

/// 相对位姿估计结果。
#[derive(Clone, Copy, Debug)]
pub struct RelativePoseReport {
    /// `cam2_from_cam1`（四元数 `w,x,y,z` + 平移）。
    pub qvec: [f64; 4],
    pub tvec: [f64; 3],
    /// 内点数（cheirality 通过的对应数）。
    pub num_inliers: usize,
    /// 三角角（度，中位数）。
    pub tri_angle_deg: f64,
    /// 是否纯旋转（PANORAMIC）。
    pub panoramic: bool,
    /// 估计出的焦距（像素）；`None` = 用相机既有焦距。
    pub focal_length: Option<f64>,
}

/// 由 `f` 构造标定矩阵（单焦距 + 主点）。
pub fn calibration_with_focal(width: u64, height: u64, focal: f64) -> [[f64; 3]; 3] {
    [[focal, 0.0, width as f64 / 2.0], [0.0, focal, height as f64 / 2.0], [0.0, 0.0, 1.0]]
}

/// `E = K₂ᵀ · F · K₁`（`EssentialFromFundamentalMatrix`）。
pub fn essential_from_fundamental(k1: [[f64; 3]; 3], k2: [[f64; 3]; 3], f: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    mat3_mul(&transpose3(&k2), &mat3_mul(&f, &k1))
}

fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += a[row][k] * b[k][col];
            }
            out[row][col] = sum;
        }
    }
    out
}

/// 特征矩阵的 4 个（旋转、平移）解（4.2.1 `DecomposeEssentialMatrix`：
/// `R1 = U·W·V`、`R2 = U·Wᵀ·V`、`t = U.col(2)`，组合 `(R1,t) (R2,t) (R1,−t) (R2,−t)`）。
pub fn decompose_essential(e: &[[f64; 3]; 3]) -> Vec<([[f64; 3]; 3], [f64; 3])> {
    let svd = nalgebra::linalg::SVD::try_new(
        nalgebra::Matrix3::from_row_slice(&flatten_row(e)),
        true,
        true,
        5.0 * f64::EPSILON,
        200,
    )
    .expect("3×3 SVD");
    let to_array = |m: &nalgebra::Matrix3<f64>| -> [[f64; 3]; 3] {
        [
            [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
            [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
            [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
        ]
    };
    let mut u = to_array(&svd.u.expect("U"));
    // 4.2.1 用的是 `V = matrixV().transpose()`（即 Vᵀ），`R = U·W·V`。
    let mut v = to_array(&svd.v_t.expect("Vᵀ"));
    // 4.2.1：先修正行列式符号，保证 R1/R2 都是真旋转。
    if det3(&u) < 0.0 {
        u = scale3(&u, -1.0);
    }
    if det3(&v) < 0.0 {
        v = scale3(&v, -1.0);
    }
    let w = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let w_t = transpose3(&w);
    let r1 = mat3_mul(&mat3_mul(&u, &w), &v);
    let r2 = mat3_mul(&mat3_mul(&u, &w_t), &v);
    let t = normalize3([u[0][2], u[1][2], u[2][2]]);
    vec![
        (r1, t),
        (r2, t),
        (r1, [-t[0], -t[1], -t[2]]),
        (r2, [-t[0], -t[1], -t[2]]),
    ]
}

fn flatten_row(m: &[[f64; 3]; 3]) -> [f64; 9] {
    [
        m[0][0], m[0][1], m[0][2], //
        m[1][0], m[1][1], m[1][2], //
        m[2][0], m[2][1], m[2][2],
    ]
}

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn scale3(m: &[[f64; 3]; 3], factor: f64) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = m[i][j] * factor;
        }
    }
    out
}

fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm > 0.0 {
        [v[0] / norm, v[1] / norm, v[2] / norm]
    } else {
        v
    }
}

/// 单应的 4 组解（Malis–Vargas / INRIA；与 OpenCV `decomposeHomographyMat` 一致）。
///
/// 输入是**去掉内参后**的单应 `K₂⁻¹ H K₁`。分解前必须按中间奇异值归一
/// （`σ₂ = 1`）—— 漏掉这一步会让 R/t 被尺度污染（真实事故：相邻视频帧被解出
/// 114° 旋转，初始三角化全零）。
pub fn decompose_homography(h: &[[f64; 3]; 3]) -> Vec<([[f64; 3]; 3], [f64; 3], [f64; 3])> {
    // removeScale：H /= σ₂。
    let matrix = nalgebra::Matrix3::from_row_slice(&flatten_row(h));
    let Some(svd) =
        nalgebra::linalg::SVD::try_new(matrix, false, false, 5.0 * f64::EPSILON, 200)
    else {
        return Vec::new();
    };
    let sigma2 = svd.singular_values[1];
    if !(sigma2 > 0.0) {
        return Vec::new();
    }
    let hn = {
        let scale = 1.0 / sigma2;
        let mut out = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = h[i][j] * scale;
            }
        }
        out
    };
    // S = HᵀH − I；纯旋转（S ≈ 0）没有平移，交给上层拒绝。
    let mut s = mat3_mul(&transpose3(&hn), &hn);
    for i in 0..3 {
        s[i][i] -= 1.0;
    }
    let norm_inf = s
        .iter()
        .flatten()
        .fold(0.0f64, |acc, value| acc.max(value.abs()));
    if norm_inf < 1e-3 {
        return vec![(hn, [0.0; 3], [0.0; 3])];
    }

    let opposite_of_minor = |m: &[[f64; 3]; 3], row: usize, col: usize| -> f64 {
        let x1 = if col == 0 { 1 } else { 0 };
        let x2 = if col == 2 { 1 } else { 2 };
        let y1 = if row == 0 { 1 } else { 0 };
        let y2 = if row == 2 { 1 } else { 2 };
        m[y1][x2] * m[y2][x1] - m[y1][x1] * m[y2][x2]
    };
    let m00 = opposite_of_minor(&s, 0, 0);
    let m11 = opposite_of_minor(&s, 1, 1);
    let m22 = opposite_of_minor(&s, 2, 2);
    if m00 < 0.0 || m11 < 0.0 || m22 < 0.0 {
        return Vec::new();
    }
    let rt_m00 = m00.sqrt();
    let rt_m11 = m11.sqrt();
    let rt_m22 = m22.sqrt();
    let m01 = opposite_of_minor(&s, 0, 1);
    let m12 = opposite_of_minor(&s, 1, 2);
    let m02 = opposite_of_minor(&s, 0, 2);
    let signd = |value: f64| if value >= 0.0 { 1.0 } else { -1.0 };
    let e12 = signd(m12);
    let e02 = signd(m02);
    let e01 = signd(m01);

    let n_s00 = s[0][0].abs();
    let n_s11 = s[1][1].abs();
    let n_s22 = s[2][2].abs();
    let indx = if n_s00 < n_s11 {
        if n_s11 < n_s22 {
            2
        } else {
            1
        }
    } else if n_s00 < n_s22 {
        2
    } else {
        0
    };

    let mut npa = [0.0f64; 3];
    let mut npb = [0.0f64; 3];
    match indx {
        0 => {
            npa[0] = s[0][0];
            npb[0] = s[0][0];
            npa[1] = s[0][1] + rt_m22;
            npb[1] = s[0][1] - rt_m22;
            npa[2] = s[0][2] + e12 * rt_m11;
            npb[2] = s[0][2] - e12 * rt_m11;
        }
        1 => {
            npa[0] = s[0][1] + rt_m22;
            npb[0] = s[0][1] - rt_m22;
            npa[1] = s[1][1];
            npb[1] = s[1][1];
            npa[2] = s[1][2] - e02 * rt_m00;
            npb[2] = s[1][2] + e02 * rt_m00;
        }
        _ => {
            npa[0] = s[0][2] + e01 * rt_m11;
            npb[0] = s[0][2] - e01 * rt_m11;
            npa[1] = s[1][2] + rt_m00;
            npb[1] = s[1][2] - rt_m00;
            npa[2] = s[2][2];
            npb[2] = s[2][2];
        }
    }

    let trace_s = s[0][0] + s[1][1] + s[2][2];
    let v_sq = 1.0 + trace_s - m00 - m11 - m22;
    if !(v_sq > 0.0) {
        return Vec::new();
    }
    let v = 2.0 * v_sq.sqrt();
    if !(v > 0.0) {
        return Vec::new();
    }
    let esii = signd(s[indx][indx]);
    let r = (2.0 + trace_s + v).sqrt();
    let n_t = (2.0 + trace_s - v).sqrt();
    let na = normalize3(npa);
    let nb = normalize3(npb);
    let half_nt = 0.5 * n_t;
    let esii_t_r = esii * r;
    let ta_star = [
        half_nt * (esii_t_r * nb[0] - n_t * na[0]),
        half_nt * (esii_t_r * nb[1] - n_t * na[1]),
        half_nt * (esii_t_r * nb[2] - n_t * na[2]),
    ];
    let tb_star = [
        half_nt * (esii_t_r * na[0] - n_t * nb[0]),
        half_nt * (esii_t_r * na[1] - n_t * nb[1]),
        half_nt * (esii_t_r * na[2] - n_t * nb[2]),
    ];
    let rotation_from = |tstar: [f64; 3], n: [f64; 3]| -> [[f64; 3]; 3] {
        // R = H·(I − (2/v)·tstar·nᵀ)
        let mut m = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = (if i == j { 1.0 } else { 0.0 }) - (2.0 / v) * tstar[i] * n[j];
            }
        }
        let mut rotation = mat3_mul(&hn, &m);
        if det3(&rotation) < 0.0 {
            rotation = scale3(&rotation, -1.0);
        }
        rotation
    };
    let mut out = Vec::with_capacity(4);
    for (tstar, n) in [(ta_star, na), (tb_star, nb)] {
        let rotation = rotation_from(tstar, n);
        let t = mat3_vec3(&rotation, tstar);
        out.push((rotation, t, n));
        out.push((rotation, [-t[0], -t[1], -t[2]], [-n[0], -n[1], -n[2]]));
    }
    out
}

/// 归一化像平面坐标（`z = 1`）。
fn normalized_point(camera: &Camera, point: [f64; 2]) -> Option<[f64; 3]> {
    cam_point_from_img(camera::model_of(camera), &camera.params, point[0], point[1])
}

/// cheirality 计数：对每个对应做两视图三角化，统计两台相机深度都为正的数量
/// （4.2.1 `CheckCheirality`）。旧实现把平移直接加到光线方向上，维度错误，
/// 会在纯旋转/近平面的 H 分解里选错分支。
fn count_cheiral(
    r: &[[f64; 3]; 3],
    t: [f64; 3],
    rays1: &[[f64; 3]],
    rays2: &[[f64; 3]],
) -> (usize, Vec<bool>) {
    let qvec = rotation_to_quaternion(r);
    let identity = [1.0, 0.0, 0.0, 0.0];
    let zero = [0.0; 3];
    let mask: Vec<bool> = rays1
        .iter()
        .zip(rays2)
        .map(|(a, b)| {
            let Some(xyz) = triangulate_two_view(
                &identity,
                &zero,
                [a[0] / a[2], a[1] / a[2]],
                &qvec,
                &t,
                [b[0] / b[2], b[1] / b[2]],
            ) else {
                return false;
            };
            let cam2 = camera::cam_from_world_point(&qvec, &t, xyz);
            xyz[2] > 1e-6 && cam2[2] > 1e-6
        })
        .collect();
    let count = mask.iter().filter(|value| **value).count();
    (count, mask)
}

/// 三角角中位数（4.2.1 `tri_angle`：存活对应光线的夹角中位数）。
pub fn median_tri_angle_deg(
    qvec: &[f64; 4],
    tvec: &[f64; 3],
    points1: &[[f64; 3]],
    points2: &[[f64; 3]],
) -> f64 {
    let r = quaternion_to_rotation_matrix(qvec);
    // cam1 视线方向在世界系里；cam2 的视线要用 `Rᵀ` 变换回 cam1 系。
    let mut angles: Vec<f64> = Vec::with_capacity(points1.len());
    for (a, b) in points1.iter().zip(points2) {
        let b_in_cam1 = mat3_vec3(&transpose3(&r), *b);
        let angle = angle_between(*a, b_in_cam1);
        angles.push(angle.min(std::f64::consts::PI - angle));
    }
    let _ = tvec;
    if angles.is_empty() {
        return 0.0;
    }
    angles.sort_by(|a, b| a.partial_cmp(b).expect("角度有限"));
    angles[angles.len() / 2].to_degrees()
}

fn angle_between(a: [f64; 3], b: [f64; 3]) -> f64 {
    let norm1 = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let norm2 = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    if !(norm1 > 0.0) || !(norm2 > 0.0) {
        return 0.0;
    }
    let cosine = ((a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (norm1 * norm2)).clamp(-1.0, 1.0);
    cosine.acos()
}

/// 从两视图几何（F 或 H）求相对位姿（4.2.1 `EstimateTwoViewGeometryPose`）。
///
/// `inlier_points1/2` 是内点对应的像素坐标；`focal` / `h` 按 `config` 取用。
pub fn relative_pose_from_geometry(
    config: i32,
    camera1: &Camera,
    camera2: &Camera,
    inlier_points1: &[[f64; 2]],
    inlier_points2: &[[f64; 2]],
    f: Option<&[[f64; 3]; 3]>,
    h: Option<&[[f64; 3]; 3]>,
    _camera1_has_prior: bool,
    _camera2_has_prior: bool,
    _min_focal_length_ratio: f64,
    _max_focal_length_ratio: f64,
) -> Option<RelativePoseReport> {
    if inlier_points1.len() < 2 || inlier_points1.len() != inlier_points2.len() {
        return None;
    }
    match config {
        UNCALIBRATED => {
            let f = f?;
            // 直接用相机当前内参（焦距由后续 PnP 精化 / BA 继续优化）。
            let k1 = camera::calibration_matrix(camera1);
            let k2 = camera::calibration_matrix(camera2);
            let e = essential_from_fundamental(k1, k2, *f);
            let rays1: Vec<[f64; 3]> = inlier_points1
                .iter()
                .filter_map(|point| normalized_point(camera1, *point))
                .collect();
            let rays2: Vec<[f64; 3]> = inlier_points2
                .iter()
                .filter_map(|point| normalized_point(camera2, *point))
                .collect();
            if rays1.len() != rays2.len() || rays1.is_empty() {
                return None;
            }
            // E 在标定系里：把归一化光线还原到像素系再判 cheirality。
            let _ = (k1, k2);
            let candidates = decompose_essential(&e);
            let mut best: Option<(usize, [[f64; 3]; 3], [f64; 3])> = None;
            for (r, t) in candidates {
                let (count, _) = count_cheiral(&r, t, &rays1, &rays2);
                if best.as_ref().map(|(current, _, _)| count > *current).unwrap_or(true) {
                    best = Some((count, r, t));
                }
            }
            let (num_inliers, r, t) = best?;
            if num_inliers == 0 {
                return None;
            }
            let qvec = rotation_to_quaternion(&r);
            // 三角角用归一化光线算（世界系下等价）。
            let tri_angle_deg = median_tri_angle_deg(&qvec, &t, &rays1, &rays2);
            Some(RelativePoseReport {
                qvec,
                tvec: t,
                num_inliers,
                tri_angle_deg,
                panoramic: false,
                focal_length: None,
            })
        }
        PLANAR_OR_PANORAMIC => {
            let h = h?;
            let k1 = camera::calibration_matrix(camera1);
            let k2 = camera::calibration_matrix(camera2);
            let k2_inv = invert3(k2)?;
            let k1_inv = invert3(k1)?;
            // 去标定：H_cam = K₂⁻¹ H K₁
            let h_cam = mat3_mul(&k2_inv, &mat3_mul(h, &k1));
            let rays1: Vec<[f64; 3]> = inlier_points1
                .iter()
                .filter_map(|point| normalized_point(camera1, *point))
                .collect();
            let rays2: Vec<[f64; 3]> = inlier_points2
                .iter()
                .filter_map(|point| normalized_point(camera2, *point))
                .collect();
            if rays1.len() != rays2.len() || rays1.is_empty() {
                return None;
            }
            let mut best: Option<(usize, [[f64; 3]; 3], [f64; 3])> = None;
            for (r, t, _normal) in decompose_homography(&h_cam) {
                let (count, mask) = count_cheiral(&r, t, &rays1, &rays2);
                let all_cheiral = mask.iter().all(|keep| *keep);
                let score = if all_cheiral { count + rays1.len() } else { count };
                if best
                    .as_ref()
                    .map(|(current, _, _)| score > *current)
                    .unwrap_or(true)
                {
                    best = Some((count, r, t));
                }
            }
            let (num_inliers, r, t) = best?;
            let qvec = rotation_to_quaternion(&r);
            let tri_angle_deg = median_tri_angle_deg(&qvec, &t, &rays1, &rays2);
            let panoramic = t[0] * t[0] + t[1] * t[1] + t[2] * t[2] < 1e-12;
            Some(RelativePoseReport {
                qvec,
                tvec: t,
                num_inliers,
                tri_angle_deg: if panoramic { 0.0 } else { tri_angle_deg },
                panoramic,
                focal_length: None,
            })
        }
        DEGENERATE | _ => None,
    }
}

fn invert3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = det3(&m);
    if det.abs() < 1e-18 {
        return None;
    }
    let inverse = 1.0 / det;
    let mut out = [[0.0f64; 3]; 3];
    out[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inverse;
    out[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inverse;
    out[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inverse;
    out[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inverse;
    out[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inverse;
    out[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inverse;
    out[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inverse;
    out[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inverse;
    out[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inverse;
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::colmap::apple::ba::CameraModelId;

    fn camera() -> Camera {
        Camera::new(
            1,
            CameraModelId::SimpleRadial as i32,
            1000,
            800,
            vec![800.0, 500.0, 400.0, 0.0],
        )
    }

    /// 合成两视图：随机三维点 → 像素观测 + 真值 E 的分解。
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / (1u64 << 53) as f64
        }
    }

    fn scene(seed: u64) -> ([f64; 3], Vec<([f64; 2], [f64; 2])>) {
        let mut rng = Rng(seed);
        let focal = 800.0;
        let center2 = [0.6, 0.1, 0.2];
        // 朝向：两相机平行（纯平移），最简单且可解析验证。
        let r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let t = [
            -center2[0],
            -center2[1],
            -center2[2],
        ];
        let mut pairs = Vec::new();
        for _ in 0..40 {
            let xyz = [
                (rng.next() - 0.5) * 4.0,
                (rng.next() - 0.5) * 3.0,
                3.0 + rng.next() * 3.0,
            ];
            let pixel1 = camera::world_to_img(&camera(), CameraModelId::SimpleRadial, &rotation_to_quaternion(&r), &[0.0, 0.0, 0.0], xyz).unwrap();
            let in2 = [
                r[0][0] * xyz[0] + r[0][1] * xyz[1] + r[0][2] * xyz[2] + t[0],
                r[1][0] * xyz[0] + r[1][1] * xyz[1] + r[1][2] * xyz[2] + t[1],
                r[2][0] * xyz[0] + r[2][1] * xyz[1] + r[2][2] * xyz[2] + t[2],
            ];
            let pixel2 = camera::world_to_img(&camera(), CameraModelId::SimpleRadial, &rotation_to_quaternion(&r), &t, xyz).unwrap();
            let _ = (focal, in2);
            pairs.push(([pixel1[0], pixel1[1]], [pixel2[0], pixel2[1]]));
        }
        (center2, pairs)
    }

    #[test]
    fn essential_decomposition_finds_known_translation_axis() {
        // 纯平移的 E = [t]×：SVD 必然给出 σ = (1, 1, 0)，所以平移方向 = U 第三列。
        let (center2, pairs) = scene(7);
        let t = [-center2[0], -center2[1], -center2[2]];
        let e = [[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]];
        let candidates = decompose_essential(&e);
        assert_eq!(candidates.len(), 4);
        let truth = normalize3([t[0], t[1], t[2]]);
        let mut matched = 0;
        for (r, translation) in candidates {
            // 4 支的平移只有两个方向（±t）。
            let dot = translation[0] * truth[0]
                + translation[1] * truth[1]
                + translation[2] * truth[2];
            assert!(dot.abs() > 0.999, "平移应与基线平行：{dot}");
            if dot > 0.0 {
                matched += 1;
            }
            // 旋转必须是真旋转（det = +1）。
            assert!((det3(&r) - 1.0).abs() < 1e-6, "det(R) = {}", det3(&r));
            // E = [t]× 的几何：R 是绕基线方向 ±90° 的旋转，因此 R·t̂ = ±t̂。
            let rotated = mat3_vec3(&r, truth);
            let along = rotated[0] * truth[0] + rotated[1] * truth[1] + rotated[2] * truth[2];
            assert!(along.abs() > 0.999, "R 应绕基线轴旋转：{along}");
        }
        assert_eq!(matched, 2, "两个方向各两支");
        assert_eq!(pairs.len(), 40);
    }

    #[test]
    fn homography_decomposition_recovers_known_pose() {
        // 平面 n·X = d（相机 1 系），相机 2 有已知 R、t：x2 ∝ (R + t·nᵀ/d)·x1。
        let angle = 20.0f64.to_radians();
        let r = [
            [angle.cos(), 0.0, angle.sin()],
            [0.0, 1.0, 0.0],
            [-angle.sin(), 0.0, angle.cos()],
        ];
        let t = [0.5, 0.1, 0.2];
        let n = [0.2f64, -0.1, 0.97];
        let norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        let n = [n[0] / norm, n[1] / norm, n[2] / norm];
        let d = 2.0;
        let mut h = r;
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] += t[i] * n[j] / d;
            }
        }
        let candidates = decompose_homography(&h);
        assert!(!candidates.is_empty(), "应有候选解");
        let truth = normalize3(t);
        let mut best_dot = 0.0f64;
        let mut best_rotation_error = f64::MAX;
        for (candidate_r, candidate_t, _) in &candidates {
            let candidate_norm = (candidate_t[0] * candidate_t[0]
                + candidate_t[1] * candidate_t[1]
                + candidate_t[2] * candidate_t[2])
                .sqrt();
            if candidate_norm <= 0.0 {
                continue;
            }
            let dot = (candidate_t[0] * truth[0]
                + candidate_t[1] * truth[1]
                + candidate_t[2] * truth[2])
                / candidate_norm;
            if dot.abs() > 0.99 {
                let mut rotation_error = 0.0f64;
                for i in 0..3 {
                    for j in 0..3 {
                        rotation_error =
                            rotation_error.max((candidate_r[i][j] - r[i][j]).abs());
                    }
                }
                if rotation_error < best_rotation_error {
                    best_rotation_error = rotation_error;
                }
            }
            best_dot = best_dot.max(dot.abs());
        }
        assert!(best_dot > 0.99, "平移方向应被覆盖：{best_dot}");
        assert!(
            best_rotation_error < 1e-6,
            "旋转应与真值一致：误差 {best_rotation_error}"
        );
    }

    #[test]
    fn relative_pose_from_fundamental_recovers_translation() {
        let (center2, pairs) = scene(23);
        let focal = 800.0;
        let t = [-center2[0], -center2[1], -center2[2]];
        let e = [[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]];
        let k = calibration_with_focal(1000, 800, focal);
        let k_inv = invert3(k).unwrap();
        let f = mat3_mul(&transpose3(&k_inv), &mat3_mul(&e, &k_inv));
        let points1: Vec<[f64; 2]> = pairs.iter().map(|(a, _)| *a).collect();
        let points2: Vec<[f64; 2]> = pairs.iter().map(|(_, b)| *b).collect();
        let report = relative_pose_from_geometry(
            UNCALIBRATED,
            &camera(),
            &camera(),
            &points1,
            &points2,
            Some(&f),
            None,
            true,
            true,
            0.1,
            10.0,
        )
        .expect("应解出相对位姿");
        assert!(report.num_inliers >= points1.len() / 2);
        let truth = normalize3([t[0], t[1], t[2]]);
        let dot = report.tvec[0] * truth[0] + report.tvec[1] * truth[1] + report.tvec[2] * truth[2];
        assert!(dot.abs() > 0.99, "平移方向应与真值一致：{dot}");
        assert!(
            report.tri_angle_deg > 0.5,
            "三角角 {}",
            report.tri_angle_deg
        );
    }
}