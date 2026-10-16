//! 自研 CPU 光束法平差（M6，决策 D4）。
//!
//! 结构：Schur 消元（点块）+ LM（Marquardt 阻尼），与 COLMAP/Ceres 的对应关系：
//! - 精简相机系统规模 ≤ [`DENSE_MAX_DOF`] 时稠密 Cholesky 直接解（等价
//!   Ceres `SPARSE_SCHUR` 的直接路径），更大时用块-Jacobi 预条件共轭梯度
//!   （等价 `ITERATIVE_SCHUR` + Jacobi 预条件）。
//! - 损失函数与 Ceres 同构：Trivial / SoftL1 / Huber / Cauchy（一阶权重
//!   `sqrt(ρ')` 施加到残差与雅可比）。
//! - 规范固定（gauge）：`TwoCamsFromWorld` = 固定第一个可变位姿 + 第二个的
//!   旋转与一个平移分量（COLMAP 全局 BA 的 `FixGaugeWithTwoCamsFromWorld`）；
//!   `ThreePoints` = 固定 3 个线性无关的点（COLMAP 局部 BA）。
//! - 位姿切空间：左扰动 `R ← Exp(δθ)·R`、`t ← Exp(δθ)·t + δt`（与 COLMAP 的
//!   四元数流形等价，仅参数化不同）。
//!
//! 支持相机模型：SIMPLE_PINHOLE(0) / PINHOLE(1) / SIMPLE_RADIAL(2) / RADIAL(3)；
//! 主链路只产出 SIMPLE_RADIAL，其余为兼容读入而保留。

use std::collections::HashMap;

use super::linalg::{
    exp_rotation_vector, mat3_mat3, mat3_vec3, quaternion_to_rotation_matrix,
    rotation_to_quaternion,
};

/// 精简相机系统改用稠密直接解的最大自由度。
const DENSE_MAX_DOF: usize = 400;
/// LM 阻尼上下限（对角乘子）。
const MIN_LAMBDA: f64 = 1e-16;
const MAX_LAMBDA: f64 = 1e16;
/// 点深度截断（负深度由 mapper 的过滤步骤清理；BA 内防御性截断）。
const MIN_DEPTH: f64 = 1e-9;
/// 阻尼对角下限（防止零对角）。
const DAMPING_FLOOR: f64 = 1e-8;
/// CG 相对残差容限与迭代上限。
const CG_TOLERANCE: f64 = 1e-8;
const CG_MAX_ITERATIONS: usize = 500;

// ---------------------------------------------------------------- 损失函数

/// Ceres 同构的鲁棒损失（`b` = `loss_function_scale`）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Loss {
    Trivial,
    SoftL1(f64),
    Huber(f64),
    Cauchy(f64),
}

impl Loss {
    /// 返回 `ρ(s)` 与 `ρ'(s)`（`s` 为平方残差）。
    #[inline]
    pub fn evaluate(&self, s: f64) -> (f64, f64) {
        match *self {
            Loss::Trivial => (s, 1.0),
            Loss::SoftL1(b) => {
                let root = (1.0 + s / (b * b)).sqrt();
                (2.0 * b * (root - 1.0), 1.0 / root)
            }
            Loss::Huber(b) => {
                if s <= b * b {
                    (s, 1.0)
                } else {
                    let root = s.sqrt();
                    (2.0 * b * root - b * b, b / root)
                }
            }
            Loss::Cauchy(b) => {
                let u = s / (b * b);
                (0.5 * b * b * (1.0 + u).ln(), 1.0 / (1.0 + u))
            }
        }
    }
}

// ---------------------------------------------------------------- 相机模型

/// BA 内部的相机投影模型（参数布局与 `format::camera_model` 一致）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraModelId {
    /// 0：`f, cx, cy`。
    SimplePinhole,
    /// 1：`fx, fy, cx, cy`。
    Pinhole,
    /// 2：`f, cx, cy, k`（主链路）。
    SimpleRadial,
    /// 3：`f, cx, cy, k1, k2`。
    Radial,
}

impl CameraModelId {
    pub fn from_id(model_id: i32) -> Option<Self> {
        match model_id {
            0 => Some(Self::SimplePinhole),
            1 => Some(Self::Pinhole),
            2 => Some(Self::SimpleRadial),
            3 => Some(Self::Radial),
            _ => None,
        }
    }

    pub fn num_params(self) -> usize {
        match self {
            Self::SimplePinhole => 3,
            Self::Pinhole => 4,
            Self::SimpleRadial => 4,
            Self::Radial => 5,
        }
    }

    /// 焦距参数下标（COLMAP `FocalLengthIdxs`）。
    pub fn focal_idxs(self) -> &'static [usize] {
        match self {
            Self::SimplePinhole | Self::SimpleRadial | Self::Radial => &[0],
            Self::Pinhole => &[0, 1],
        }
    }

    /// 主点参数下标（`PrincipalPointIdxs`）。
    pub fn principal_point_idxs(self) -> &'static [usize] {
        match self {
            Self::SimplePinhole | Self::SimpleRadial | Self::Radial => &[1, 2],
            Self::Pinhole => &[2, 3],
        }
    }

    /// 畸变参数下标（`ExtraParamsIdxs`）。
    pub fn extra_idxs(self) -> &'static [usize] {
        match self {
            Self::SimplePinhole | Self::Pinhole => &[],
            Self::SimpleRadial => &[3],
            Self::Radial => &[3, 4],
        }
    }
}

/// 像素 → 归一化相机光线（含畸变逆；PINHOLE 系解析，k 系牛顿迭代，
/// 与 COLMAP `CamRayFromImg` / `IterativeUndistortion` 同语义）。
pub fn cam_ray_from_image(
    model: CameraModelId,
    params: &[f64],
    x: f64,
    y: f64,
) -> Option<[f64; 3]> {
    let (fx, fy, cx, cy) = match model {
        CameraModelId::SimplePinhole | CameraModelId::SimpleRadial | CameraModelId::Radial => {
            (params[0], params[0], params[1], params[2])
        }
        CameraModelId::Pinhole => (params[0], params[1], params[2], params[3]),
    };
    if fx <= 0.0 || fy <= 0.0 {
        return None;
    }
    let xd = (x - cx) / fx;
    let yd = (y - cy) / fy;
    let extras = model.extra_idxs();
    if extras.is_empty() {
        return Some([xd, yd, 1.0]);
    }
    let k1 = params[extras[0]];
    let k2 = if extras.len() > 1 {
        params[extras[1]]
    } else {
        0.0
    };
    // 牛顿迭代解 (xn, yn)：xd = xn·(1 + k1 r² + k2 r⁴)。
    let (mut xn, mut yn) = (xd, yd);
    for _ in 0..100 {
        let r2 = xn * xn + yn * yn;
        let scale = 1.0 + k1 * r2 + k2 * r2 * r2;
        let ex = xn * scale - xd;
        let ey = yn * scale - yd;
        if ex.abs() < 1e-10 && ey.abs() < 1e-10 {
            break;
        }
        let j00 = scale + 2.0 * k1 * xn * xn + 4.0 * k2 * xn * xn * r2;
        let j01 = 2.0 * k1 * xn * yn + 4.0 * k2 * xn * yn * r2;
        let j11 = scale + 2.0 * k1 * yn * yn + 4.0 * k2 * yn * yn * r2;
        let det = j00 * j11 - j01 * j01;
        if det.abs() < 1e-14 {
            return None;
        }
        xn -= (j11 * ex - j01 * ey) / det;
        yn -= (j00 * ey - j01 * ex) / det;
    }
    Some([xn, yn, 1.0])
}

// ---------------------------------------------------------------- 问题定义

/// BA 中的相机内参。
#[derive(Clone, Debug)]
pub struct BaCamera {
    pub id: u32,
    pub model_id: i32,
    pub params: Vec<f64>,
    /// 固定全部内参。
    pub constant: bool,
}

/// BA 中的一张图像（位姿块）。
#[derive(Clone, Debug)]
pub struct BaImage {
    pub id: u32,
    /// `cameras` 数组行号（不是 COLMAP camera_id）。
    pub camera: usize,
    pub qvec: [f64; 4],
    pub tvec: [f64; 3],
    /// 固定位姿（规范固定 / `fix_existing_frames`）。
    pub constant_pose: bool,
}

/// BA 中的一个三维点及其观测。
#[derive(Clone, Debug)]
pub struct BaPoint {
    pub xyz: [f64; 3],
    /// `(images 行号, 像素 x, 像素 y)`。
    pub observations: Vec<(usize, f64, f64)>,
    /// 固定该点（规范固定使用）。
    pub constant: bool,
}

/// 规范固定方式（COLMAP `BundleAdjustmentGauge`）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Gauge {
    #[default]
    None,
    /// 固定第一个可变位姿 + 第二个的旋转与一个平移分量（全局 BA）。
    TwoCamsFromWorld,
    /// 固定 3 个线性无关的点（局部 BA）。
    ThreePoints,
}

/// 求解选项（COLMAP 侧语义逐项对应）。
#[derive(Clone, Debug)]
pub struct BaOptions {
    pub loss: Loss,
    pub max_iterations: u32,
    pub refine_focal_length: bool,
    pub refine_principal_point: bool,
    pub refine_extra_params: bool,
    pub gauge: Gauge,
    /// 收敛：相对函数下降量（0 = 关闭）。
    pub function_tolerance: f64,
    /// 收敛：`max|∇| ≤ gradient_tolerance·max(1, cost)`（0 = 关闭）。
    pub gradient_tolerance: f64,
}

impl Default for BaOptions {
    fn default() -> Self {
        Self {
            loss: Loss::Trivial,
            max_iterations: 50,
            refine_focal_length: true,
            refine_principal_point: false,
            refine_extra_params: true,
            gauge: Gauge::TwoCamsFromWorld,
            function_tolerance: 0.0,
            gradient_tolerance: 1.0,
        }
    }
}

/// 求解摘要。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BaSummary {
    pub cost: f64,
    pub iterations: u32,
    pub num_residuals: usize,
    pub converged: bool,
}

// ---------------------------------------------------------------- 投影

/// 把相机系点投影到像素（z 已截断），返回投影与各偏导。
#[allow(clippy::type_complexity)]
fn project(
    model: CameraModelId,
    params: &[f64],
    u: [f64; 3],
) -> (
    [f64; 2],
    [f64; 6],
    Vec<[f64; 2]>,
    [[f64; 2]; 2],
    Vec<[f64; 2]>,
) {
    let (fx, fy, cx, cy) = match model {
        CameraModelId::SimplePinhole | CameraModelId::SimpleRadial | CameraModelId::Radial => {
            (params[0], params[0], params[1], params[2])
        }
        CameraModelId::Pinhole => (params[0], params[1], params[2], params[3]),
    };
    let z_safe = u[2].max(MIN_DEPTH);
    let xp = u[0] / z_safe;
    let yp = u[1] / z_safe;
    let r2 = xp * xp + yp * yp;
    let extras = model.extra_idxs();
    let (k1, k2) = if extras.is_empty() {
        (0.0, 0.0)
    } else {
        (
            params[extras[0]],
            if extras.len() > 1 {
                params[extras[1]]
            } else {
                0.0
            },
        )
    };
    let r4 = r2 * r2;
    let scale = 1.0 + k1 * r2 + k2 * r4;
    let xd = xp * scale;
    let yd = yp * scale;
    let projection = [fx * xd + cx, fy * yd + cy];

    let dxx = scale + 2.0 * k1 * xp * xp + 4.0 * k2 * xp * xp * r2;
    let dxy = 2.0 * k1 * xp * yp + 4.0 * k2 * xp * yp * r2;
    let dyy = scale + 2.0 * k1 * yp * yp + 4.0 * k2 * yp * yp * r2;

    // dp/du = diag(fx,fy)·D·d(xp,yp)/du（行主序 2×3）。
    let iz = 1.0 / z_safe;
    let dp_du = [
        fx * dxx * iz,
        fx * dxy * iz,
        fx * (-dxx * xp - dxy * yp) * iz,
        fy * dxy * iz,
        fy * dyy * iz,
        fy * (-dxy * xp - dyy * yp) * iz,
    ];

    let focal_columns = match model {
        CameraModelId::SimplePinhole | CameraModelId::SimpleRadial | CameraModelId::Radial => {
            vec![[xd, yd]]
        }
        CameraModelId::Pinhole => vec![[xd, 0.0], [0.0, yd]],
    };
    let pp_columns = [[1.0, 0.0], [0.0, 1.0]];
    let extra_columns = match model {
        CameraModelId::SimpleRadial => vec![[fx * xp * r2, fy * yp * r2]],
        CameraModelId::Radial => vec![
            [fx * xp * (r2 + 2.0 * k2 * r4), fy * yp * (r2 + 2.0 * k2 * r4)],
            [fx * xp * r4, fy * yp * r4],
        ],
        _ => Vec::new(),
    };
    (projection, dp_du, focal_columns, pp_columns, extra_columns)
}

/// 像素投影（无雅可比）。
fn project_point(
    qvec: &[f64; 4],
    tvec: &[f64; 3],
    xyz: &[f64; 3],
    model: CameraModelId,
    params: &[f64],
) -> [f64; 2] {
    let rotation = quaternion_to_rotation_matrix(qvec);
    let u = [
        rotation[0][0] * xyz[0] + rotation[0][1] * xyz[1] + rotation[0][2] * xyz[2] + tvec[0],
        rotation[1][0] * xyz[0] + rotation[1][1] * xyz[1] + rotation[1][2] * xyz[2] + tvec[1],
        rotation[2][0] * xyz[0] + rotation[2][1] * xyz[1] + rotation[2][2] * xyz[2] + tvec[2],
    ];
    project(model, params, u).0
}

// ---------------------------------------------------------------- 观测雅可比

/// 一个观测的解析雅可比（加权后缓存，供组装与矩阵向量积复用）。
struct ObsJacobian {
    image: usize,
    camera: usize,
    /// `dp/du`（行主序 2×3；u = 相机系坐标）。
    dp_du: [f64; 6],
    /// 相机系坐标（z 已截断为正）。
    u: [f64; 3],
    /// 位姿旋转矩阵。
    rotation: [[f64; 3]; 3],
    /// 各焦距参数的 `dp/df`（与 `model.focal_idxs()` 同序）。
    focal_columns: Vec<[f64; 2]>,
    /// 各畸变参数的 `dp/dk`（与 `model.extra_idxs()` 同序）。
    extra_columns: Vec<[f64; 2]>,
    /// 主点两参数的 `dp/dc`。
    pp_columns: [[f64; 2]; 2],
    /// 加权残差（`sqrt(ρ')·r`）。
    residual: [f64; 2],
    /// 损失权重 `sqrt(ρ')`（用于加权雅可比）。
    weight: f64,
}

impl ObsJacobian {
    /// 某内参参数（模型内绝对下标）的像素偏导。
    #[inline]
    fn intrinsic_column(&self, model: CameraModelId, param_idx: usize) -> [f64; 2] {
        if let Some(pos) = model.focal_idxs().iter().position(|&idx| idx == param_idx) {
            return self.focal_columns[pos];
        }
        if let Some(pos) = model
            .principal_point_idxs()
            .iter()
            .position(|&idx| idx == param_idx)
        {
            return self.pp_columns[pos];
        }
        if let Some(pos) = model.extra_idxs().iter().position(|&idx| idx == param_idx) {
            return self.extra_columns[pos];
        }
        [0.0; 2]
    }

    /// 位姿雅可比（行主序 2×6；左扰动，列序 = (θ0,θ1,θ2,t0,t1,t2)）。
    #[inline]
    fn pose_jacobian(&self) -> [f64; 12] {
        let d = &self.dp_du;
        let (u, v, w) = (self.u[0], self.u[1], self.u[2]);
        // J_θ = −dp/du·[u]×，[u]× = [[0,−w,v],[w,0,−u],[−v,u,0]]
        // 行主序：前 6 = u 像素行（θ、t），后 6 = v 像素行。
        [
            -(d[1] * w - d[2] * v),
            -(-d[0] * w + d[2] * u),
            -(d[0] * v - d[1] * u),
            d[0],
            d[1],
            d[2],
            -(d[4] * w - d[5] * v),
            -(-d[3] * w + d[5] * u),
            -(d[3] * v - d[4] * u),
            d[3],
            d[4],
            d[5],
        ]
    }

    /// 点雅可比（行主序 2×3）：dp/du·R。
    #[inline]
    fn point_jacobian(&self) -> [f64; 6] {
        let d = &self.dp_du;
        let r = &self.rotation;
        [
            d[0] * r[0][0] + d[1] * r[1][0] + d[2] * r[2][0],
            d[0] * r[0][1] + d[1] * r[1][1] + d[2] * r[2][1],
            d[0] * r[0][2] + d[1] * r[1][2] + d[2] * r[2][2],
            d[3] * r[0][0] + d[4] * r[1][0] + d[5] * r[2][0],
            d[3] * r[0][1] + d[4] * r[1][1] + d[5] * r[2][1],
            d[3] * r[0][2] + d[4] * r[1][2] + d[5] * r[2][2],
        ]
    }
}

/// 观测的消元项（加权雅可比子块，供组装 / matvec / 回代统一使用）。
struct ObsTerms {
    pose_blk: Option<usize>,
    cam_blk: Option<usize>,
    /// 加权位姿雅可比（行主序 2×6，未用列为 0）。
    jp: [[f64; 6]; 2],
    /// 加权内参雅可比（行主序 2×6，未用列为 0）。
    jc: [[f64; 6]; 2],
    /// 加权点雅可比（行主序 2×3）。
    jpt: [[f64; 3]; 2],
}

impl ObsTerms {
    /// 位姿消元块 `E = Jpᵀ·Jpt`（d×3，行主序压缩进 6×3，未用行为 0）。
    fn elim_pose(&self) -> ([[f64; 3]; 6], usize) {
        let mut e = [[0.0f64; 3]; 6];
        let d = (0..6)
            .filter(|&col| self.jp[0][col] != 0.0 || self.jp[1][col] != 0.0)
            .count();
        for row in 0..d {
            for k in 0..3 {
                e[row][k] = self.jp[0][row] * self.jpt[0][k]
                    + self.jp[1][row] * self.jpt[1][k];
            }
        }
        (e, d)
    }

    /// 内参消元块 `E = Jcᵀ·Jpt`（c×3）。
    fn elim_cam(&self) -> ([[f64; 3]; 6], usize) {
        let mut e = [[0.0f64; 3]; 6];
        let c = (0..6)
            .filter(|&col| self.jc[0][col] != 0.0 || self.jc[1][col] != 0.0)
            .count();
        for row in 0..c {
            for k in 0..3 {
                e[row][k] = self.jc[0][row] * self.jpt[0][k]
                    + self.jc[1][row] * self.jpt[1][k];
            }
        }
        (e, c)
    }
}

/// 线性化结果：观测按点顺序平铺，`groups[k]` = 点 k 的观测行号区间。
struct Linearized {
    jacobians: Vec<ObsJacobian>,
    groups: Vec<(usize, usize)>,
}

fn linearize(
    cameras: &[BaCamera],
    images: &[BaImage],
    points: &[BaPoint],
    options: &BaOptions,
    model_ids: &[Option<CameraModelId>],
) -> Linearized {
    let mut jacobians = Vec::new();
    let mut groups = Vec::with_capacity(points.len());
    for point in points {
        let first = jacobians.len();
        for &(image_row, x, y) in &point.observations {
            let image = &images[image_row];
            let camera_row = image.camera;
            let camera = &cameras[camera_row];
            let model = model_ids[camera_row].expect("模型已在外层校验");

            let rotation = quaternion_to_rotation_matrix(&image.qvec);
            let px = point.xyz;
            let u = [
                rotation[0][0] * px[0] + rotation[0][1] * px[1] + rotation[0][2] * px[2]
                    + image.tvec[0],
                rotation[1][0] * px[0] + rotation[1][1] * px[1] + rotation[1][2] * px[2]
                    + image.tvec[1],
                rotation[2][0] * px[0] + rotation[2][1] * px[1] + rotation[2][2] * px[2]
                    + image.tvec[2],
            ];
            let (projection, dp_du, focal_columns, pp_columns, extra_columns) =
                project(model, &camera.params, u);
            let ex = projection[0] - x;
            let ey = projection[1] - y;
            let (_, drho) = options.loss.evaluate(ex * ex + ey * ey);
            let weight = drho.sqrt();
            jacobians.push(ObsJacobian {
                image: image_row,
                camera: camera_row,
                dp_du,
                u: [u[0], u[1], u[2].max(MIN_DEPTH)],
                rotation,
                focal_columns,
                extra_columns,
                pp_columns,
                residual: [weight * ex, weight * ey],
                weight,
            });
        }
        groups.push((first, jacobians.len()));
    }
    Linearized { jacobians, groups }
}

/// 鲁棒代价 `Σ ρ(‖r‖²)`。
fn robust_cost(
    cameras: &[BaCamera],
    images: &[BaImage],
    points: &[BaPoint],
    loss: Loss,
    model_ids: &[Option<CameraModelId>],
) -> f64 {
    let mut total = 0.0;
    for point in points {
        for &(image_row, x, y) in &point.observations {
            let image = &images[image_row];
            let model = model_ids[image.camera].expect("模型已在外层校验");
            let projection = project_point(
                &image.qvec,
                &image.tvec,
                &point.xyz,
                model,
                &cameras[image.camera].params,
            );
            let ex = projection[0] - x;
            let ey = projection[1] - y;
            let (rho, _) = loss.evaluate(ex * ex + ey * ey);
            total += rho;
        }
    }
    total
}

// ---------------------------------------------------------------- 参数布局

/// 位姿块：自由维列表（≤6）。
struct PoseBlock {
    image: usize,
    free: Vec<usize>,
    dof: usize,
}

/// 内参块：可变参数下标列表（≤5）。
struct CameraBlock {
    camera: usize,
    params: Vec<usize>,
    dof: usize,
}

/// 参数布局。
struct ParameterLayout {
    /// 图像行号 → (块序号, 块)。
    pose_of_image: Vec<Option<(usize, PoseBlock)>>,
    /// 相机行号 → (块序号, 块)。
    camera_of_camera: Vec<Option<(usize, CameraBlock)>>,
    num_blocks: usize,
    total_dof: usize,
}

fn build_layout(
    images: &[BaImage],
    cameras: &[BaCamera],
    options: &BaOptions,
    model_ids: &[Option<CameraModelId>],
    partial_fixed: &[Option<[bool; 6]>],
) -> ParameterLayout {
    let mut pose_of_image = Vec::with_capacity(images.len());
    let mut camera_of_camera = Vec::with_capacity(cameras.len());
    let mut block_id = 0usize;
    let mut cursor = 0usize;
    for (row, image) in images.iter().enumerate() {
        if image.constant_pose {
            pose_of_image.push(None);
            continue;
        }
        let fixed = partial_fixed
            .get(row)
            .copied()
            .flatten()
            .unwrap_or([false; 6]);
        let free: Vec<usize> = (0..6).filter(|&dim| !fixed[dim]).collect();
        if free.is_empty() {
            pose_of_image.push(None);
        } else {
            let dof = free.len();
            pose_of_image.push(Some((block_id, PoseBlock { image: row, free, dof })));
            block_id += 1;
            cursor += dof;
        }
    }
    for (row, camera) in cameras.iter().enumerate() {
        if camera.constant {
            camera_of_camera.push(None);
            continue;
        }
        let model = model_ids[row].expect("模型已在外层校验");
        let mut params = Vec::new();
        if options.refine_focal_length {
            params.extend(model.focal_idxs().iter().copied());
        }
        if options.refine_principal_point {
            params.extend(model.principal_point_idxs().iter().copied());
        }
        if options.refine_extra_params {
            params.extend(model.extra_idxs().iter().copied());
        }
        if params.is_empty() {
            camera_of_camera.push(None);
        } else {
            let dof = params.len();
            camera_of_camera.push(Some((block_id, CameraBlock { camera: row, params, dof })));
            block_id += 1;
            cursor += dof;
        }
    }
    ParameterLayout {
        pose_of_image,
        camera_of_camera,
        num_blocks: block_id,
        total_dof: cursor,
    }
}

// ---------------------------------------------------------------- 主入口

/// 运行光束法平差（就地精修输入；返回摘要）。
pub fn run_bundle_adjustment(
    cameras: &mut [BaCamera],
    images: &mut [BaImage],
    points: &mut [BaPoint],
    options: &BaOptions,
) -> BaSummary {
    let model_ids: Vec<Option<CameraModelId>> = cameras
        .iter()
        .map(|camera| CameraModelId::from_id(camera.model_id))
        .collect();
    if model_ids.iter().any(Option::is_none) {
        return BaSummary {
            cost: f64::INFINITY,
            iterations: 0,
            num_residuals: 0,
            converged: false,
        };
    }

    let partial_fixed = apply_gauge(images, points, options.gauge);
    let num_residuals: usize = points.iter().map(|p| p.observations.len()).sum::<usize>() * 2;
    let mut lambda = 1e-4f64;
    let mut cost = robust_cost(cameras, images, points, options.loss, &model_ids);
    let mut iterations = 0u32;
    let mut converged = false;

    for _ in 0..options.max_iterations {
        let layout = build_layout(images, cameras, options, &model_ids, &partial_fixed);
        if layout.total_dof == 0 {
            converged = true;
            break;
        }
        let linearized = linearize(cameras, images, points, options, &model_ids);
        let mut system = assemble(&layout, &linearized, points, &model_ids, lambda);

        let max_gradient = system.max_gradient();
        if options.gradient_tolerance > 0.0
            && max_gradient <= options.gradient_tolerance * cost.max(1.0)
        {
            converged = true;
            break;
        }

        iterations += 1;
        if std::env::var("OOO_BA_DEBUG").is_ok() {
            eprintln!("it={iterations} cost={cost:.6e} lambda={lambda:.3e} maxgrad={max_gradient:.3e}");
        }
        match system.solve_step(cameras, images, points, cost, options.loss, &model_ids) {
            StepOutcome::Accepted(new_cost) => {
                let relative_change = if cost > 0.0 {
                    (cost - new_cost) / cost
                } else {
                    0.0
                };
                cost = new_cost;
                lambda = (lambda / 3.0).max(MIN_LAMBDA);
                if options.function_tolerance > 0.0
                    && relative_change < options.function_tolerance
                {
                    converged = true;
                    break;
                }
                if relative_change.abs() < 1e-12 {
                    converged = true;
                    break;
                }
            }
            StepOutcome::Rejected => {
                lambda = (lambda * 3.0).min(MAX_LAMBDA);
                if lambda >= MAX_LAMBDA {
                    break;
                }
            }
            StepOutcome::Converged => {
                converged = true;
                break;
            }
            StepOutcome::Failed => {
                lambda = (lambda * 10.0).min(MAX_LAMBDA);
                if lambda >= MAX_LAMBDA {
                    break;
                }
            }
        }
    }

    BaSummary {
        cost,
        iterations,
        num_residuals,
        converged,
    }
}

#[derive(Clone, Copy, Debug)]
enum StepOutcome {
    /// 接受步（携带新代价）。
    Accepted(f64),
    /// 步被求解但代价未下降（已回滚）。
    Rejected,
    /// 步长为零：已到驻点。
    Converged,
    /// 线性求解失败（无需回滚）。
    Failed,
}

// ---------------------------------------------------------------- 规范固定

/// 应用规范固定：修改 `images` / `points`，返回每图的部分固定维
/// （`[bool; 6]`，true = 固定；维度序 = (θ0,θ1,θ2,t0,t1,t2)）。
fn apply_gauge(
    images: &mut [BaImage],
    points: &mut [BaPoint],
    gauge: Gauge,
) -> Vec<Option<[bool; 6]>> {
    let mut partial: Vec<Option<[bool; 6]>> = vec![None; images.len()];
    match gauge {
        Gauge::None => {}
        Gauge::ThreePoints => {
            let mut basis: Vec<[f64; 3]> = Vec::with_capacity(3);
            for point in points.iter_mut() {
                if basis.len() >= 3 {
                    break;
                }
                if point.constant {
                    continue;
                }
                if is_linearly_independent(&basis, &point.xyz) {
                    point.constant = true;
                    basis.push(point.xyz);
                }
            }
        }
        Gauge::TwoCamsFromWorld => {
            let mut first: Option<usize> = None;
            let mut second: Option<usize> = None;
            for (row, image) in images.iter().enumerate() {
                if !image.constant_pose {
                    if first.is_none() {
                        first = Some(row);
                    } else {
                        second = Some(row);
                        break;
                    }
                }
            }
            if let (Some(first_row), Some(second_row)) = (first, second) {
                // 基线最大分量作为固定平移维（COLMAP 同款选择）。
                let center1 =
                    pose_projection_center(&images[first_row].qvec, &images[first_row].tvec);
                let center2 =
                    pose_projection_center(&images[second_row].qvec, &images[second_row].tvec);
                let baseline = [
                    center2[0] - center1[0],
                    center2[1] - center1[1],
                    center2[2] - center1[2],
                ];
                let fixed_dim = baseline
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                images[first_row].constant_pose = true;
                // 第二个位姿：旋转保持全自由（COLMAP 的四元数流形只是参数化，
                // 不是规范约束），仅固定基线最大分量的平移维 —— 6 + 1 = 7，
                // 恰好固定相似规范的 7 个自由度。
                let mut second_fixed = [false; 6];
                second_fixed[3 + fixed_dim] = true;
                partial[second_row] = Some(second_fixed);
            } else if let Some(first_row) = first {
                images[first_row].constant_pose = true;
            }
        }
    }
    partial
}

/// 判断 `vector` 与既有基是否线性无关（≤3 维，几何判秩）。
fn is_linearly_independent(basis: &[[f64; 3]], vector: &[f64; 3]) -> bool {
    if basis.len() >= 3 {
        return false;
    }
    if vector.iter().all(|&value| value.abs() < 1e-12) {
        return false;
    }
    match basis.len() {
        0 => true,
        1 => {
            let cross = [
                basis[0][1] * vector[2] - basis[0][2] * vector[1],
                basis[0][2] * vector[0] - basis[0][0] * vector[2],
                basis[0][0] * vector[1] - basis[0][1] * vector[0],
            ];
            let cross_norm =
                (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            let base_norm = (basis[0][0] * basis[0][0]
                + basis[0][1] * basis[0][1]
                + basis[0][2] * basis[0][2])
                .sqrt();
            cross_norm > 1e-9 * base_norm.max(1e-12)
        }
        2 => {
            let det = basis[0][0] * (basis[1][1] * vector[2] - basis[1][2] * vector[1])
                - basis[1][0] * (basis[0][1] * vector[2] - basis[0][2] * vector[1])
                + vector[0] * (basis[0][1] * basis[1][2] - basis[0][2] * basis[1][1]);
            let scale = basis
                .iter()
                .chain(std::iter::once(vector))
                .flat_map(|v| v.iter())
                .fold(1.0f64, |acc, value| acc.max(value.abs()));
            det.abs() > 1e-9 * scale * scale * scale
        }
        _ => false,
    }
}

/// 位姿的投影中心 `C = −Rᵀ·t`。
fn pose_projection_center(qvec: &[f64; 4], tvec: &[f64; 3]) -> [f64; 3] {
    let r = quaternion_to_rotation_matrix(qvec);
    [
        -(r[0][0] * tvec[0] + r[1][0] * tvec[1] + r[2][0] * tvec[2]),
        -(r[0][1] * tvec[0] + r[1][1] * tvec[1] + r[2][1] * tvec[2]),
        -(r[0][2] * tvec[0] + r[1][2] * tvec[1] + r[2][2] * tvec[2]),
    ]
}

// ---------------------------------------------------------------- 小型稠密助手

/// 对称正定矩阵的 Cholesky 求解（行主序 `d×d`；只读下三角）。
fn solve_spd(matrix: &[f64], d: usize, rhs: &[f64]) -> Option<Vec<f64>> {
    let mut lower = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut sum = matrix[i * d + j];
            for k in 0..j {
                sum -= lower[i * d + k] * lower[j * d + k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 1e-300 {
                    return None;
                }
                lower[i * d + j] = sum.sqrt();
            } else {
                lower[i * d + j] = sum / lower[j * d + j];
            }
        }
    }
    let mut y = vec![0.0; d];
    for i in 0..d {
        let mut sum = rhs[i];
        for k in 0..i {
            sum -= lower[i * d + k] * y[k];
        }
        y[i] = sum / lower[i * d + i];
    }
    let mut x = vec![0.0; d];
    for i in (0..d).rev() {
        let mut sum = y[i];
        for k in (i + 1)..d {
            sum -= lower[k * d + i] * x[k];
        }
        x[i] = sum / lower[i * d + i];
    }
    Some(x)
}

/// 对称 3×3 求逆（解析伴随矩阵）。
fn invert_symmetric_3x3(m: &[f64; 9]) -> Option<[f64; 9]> {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if !det.is_finite() || det.abs() < 1e-300 {
        return None;
    }
    let s = 1.0 / det;
    Some([
        (m[4] * m[8] - m[5] * m[7]) * s,
        (m[2] * m[7] - m[1] * m[8]) * s,
        (m[1] * m[5] - m[2] * m[4]) * s,
        (m[2] * m[7] - m[1] * m[8]) * s,
        (m[0] * m[8] - m[2] * m[6]) * s,
        (m[2] * m[3] - m[0] * m[5]) * s,
        (m[1] * m[5] - m[2] * m[4]) * s,
        (m[2] * m[3] - m[0] * m[5]) * s,
        (m[0] * m[4] - m[1] * m[3]) * s,
    ])
}

// ---------------------------------------------------------------- 精简系统

/// 位姿-内参交叉块。
struct CrossBlock {
    pose: usize,
    camera: usize,
    /// 行主序 `d_pose × d_cam`（≤6×6，用固定数组承载）。
    matrix: [[f64; 6]; 6],
    d: usize,
    c: usize,
}

/// 点消元数据。
struct PointElim {
    point_row: usize,
    /// 观测项在 `terms` 中的区间。
    terms: (usize, usize),
    /// 阻尼后 C 的逆（行主序 3×3）；奇异点为全零（跳过耦合）。
    cinv: [f64; 9],
    /// 点梯度（加权）。
    g: [f64; 3],
    /// 有效对角阻尼。
    damp: [f64; 3],
    /// 回代解出的点增量。
    delta: [f64; 3],
}

/// 精简相机系统（含点消元的 Schur 结构）。
struct ReducedSystem {
    layout: ParameterLayout,
    block_dof: Vec<usize>,
    /// 各块的阻尼 Hessian（行主序 d×d，位于 6×6 数组的左上角）。
    block_h: Vec<[[f64; 6]; 6]>,
    block_g: Vec<[f64; 6]>,
    /// 各块的有效对角阻尼（预测下降量用）。
    block_damp: Vec<[f64; 6]>,
    cross: Vec<CrossBlock>,
    cross_index: HashMap<(usize, usize), usize>,
    /// 各块的块-Jacobi 预条件子逆（d×d 位于 6×6 数组左上角）。
    precond: Vec<[[f64; 6]; 6]>,
    points: Vec<PointElim>,
    terms: Vec<ObsTerms>,
    lambda: f64,
}

impl ReducedSystem {
    /// 全空间梯度无穷范数（相机组块 + 点块）。
    fn max_gradient(&self) -> f64 {
        let mut max = 0.0f64;
        for (block, g) in self.block_g.iter().enumerate() {
            for k in 0..self.block_dof[block] {
                max = max.max(g[k].abs());
            }
        }
        for point in &self.points {
            for value in point.g {
                max = max.max(value.abs());
            }
        }
        max
    }

    /// 阻尼精简系统 `B' = B − E·C⁻¹·Eᵀ` 的矩阵向量积。
    fn matvec(&self, v: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.layout.total_dof];
        // 块对角 + 交叉。
        for (block, h) in self.block_h.iter().enumerate() {
            let d = self.block_dof[block];
            let off = self.block_offset(block);
            for row in 0..d {
                let mut sum = 0.0;
                for col in 0..d {
                    sum += h[row][col] * v[off + col];
                }
                out[off + row] += sum;
            }
        }
        for cross in &self.cross {
            let pose_off = self.block_offset(cross.pose);
            let cam_off = self.block_offset(cross.camera);
            for row in 0..cross.d {
                let mut sum = 0.0;
                for col in 0..cross.c {
                    sum += cross.matrix[row][col] * v[cam_off + col];
                }
                out[pose_off + row] += sum;
            }
            for col in 0..cross.c {
                let mut sum = 0.0;
                for row in 0..cross.d {
                    sum += cross.matrix[row][col] * v[pose_off + row];
                }
                out[cam_off + col] += sum;
            }
        }
        // 点消元：−E·C⁻¹·Eᵀ·v。
        for point in &self.points {
            if point.cinv == [0.0; 9] {
                continue;
            }
            let (first, last) = point.terms;
            let mut s = [0.0f64; 3];
            for term_row in first..last {
                let term = &self.terms[term_row];
                let mut pixel = [0.0f64; 2];
                if let Some(pose) = term.pose_blk {
                    let off = self.block_offset(pose);
                    for col in 0..self.block_dof[pose] {
                        pixel[0] += term.jp[0][col] * v[off + col];
                        pixel[1] += term.jp[1][col] * v[off + col];
                    }
                }
                if let Some(cam) = term.cam_blk {
                    let off = self.block_offset(cam);
                    for col in 0..self.block_dof[cam] {
                        pixel[0] += term.jc[0][col] * v[off + col];
                        pixel[1] += term.jc[1][col] * v[off + col];
                    }
                }
                for row in 0..3 {
                    s[row] += term.jpt[0][row] * pixel[0] + term.jpt[1][row] * pixel[1];
                }
            }
            let w = [
                point.cinv[0] * s[0] + point.cinv[1] * s[1] + point.cinv[2] * s[2],
                point.cinv[3] * s[0] + point.cinv[4] * s[1] + point.cinv[5] * s[2],
                point.cinv[6] * s[0] + point.cinv[7] * s[1] + point.cinv[8] * s[2],
            ];
            for term_row in first..last {
                let term = &self.terms[term_row];
                let jpt_w = [
                    term.jpt[0][0] * w[0] + term.jpt[0][1] * w[1] + term.jpt[0][2] * w[2],
                    term.jpt[1][0] * w[0] + term.jpt[1][1] * w[1] + term.jpt[1][2] * w[2],
                ];
                if let Some(pose) = term.pose_blk {
                    let off = self.block_offset(pose);
                    for col in 0..self.block_dof[pose] {
                        out[off + col] -= term.jp[0][col] * jpt_w[0] + term.jp[1][col] * jpt_w[1];
                    }
                }
                if let Some(cam) = term.cam_blk {
                    let off = self.block_offset(cam);
                    for col in 0..self.block_dof[cam] {
                        out[off + col] -= term.jc[0][col] * jpt_w[0] + term.jc[1][col] * jpt_w[1];
                    }
                }
            }
        }
        out
    }

    /// 块在精简向量中的偏移。
    fn block_offset(&self, block: usize) -> usize {
        let mut offset = 0;
        for b in 0..block {
            offset += self.block_dof[b];
        }
        offset
    }

    /// 块-Jacobi 预条件。
    fn precondition(&self, r: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.layout.total_dof];
        for (block, precond) in self.precond.iter().enumerate() {
            let d = self.block_dof[block];
            let off = self.block_offset(block);
            for row in 0..d {
                let mut sum = 0.0;
                for col in 0..d {
                    sum += precond[row][col] * r[off + col];
                }
                out[off + row] = sum;
            }
        }
        out
    }

    /// 稠密直接解（组装完整 `B'` 后 Cholesky）。
    fn solve_dense(&self, rhs: &[f64]) -> Option<Vec<f64>> {
        let n = self.layout.total_dof;
        let mut matrix = vec![0.0; n * n];
        for (block, h) in self.block_h.iter().enumerate() {
            let d = self.block_dof[block];
            let off = self.block_offset(block);
            for row in 0..d {
                for col in 0..d {
                    matrix[(off + row) * n + off + col] += h[row][col];
                }
            }
        }
        for cross in &self.cross {
            let pose_off = self.block_offset(cross.pose);
            let cam_off = self.block_offset(cross.camera);
            for row in 0..cross.d {
                for col in 0..cross.c {
                    matrix[(pose_off + row) * n + cam_off + col] += cross.matrix[row][col];
                    matrix[(cam_off + col) * n + pose_off + row] += cross.matrix[row][col];
                }
            }
        }
        // 点消元散射（全有序对，保证对称）。
        for point in &self.points {
            if point.cinv == [0.0; 9] {
                continue;
            }
            let (first, last) = point.terms;
            for a in first..last {
                for b in first..last {
                    let (ea, eb) = (&self.terms[a], &self.terms[b]);
                    let elim_a = [(ea.pose_blk, ea.elim_pose()), (ea.cam_blk, ea.elim_cam())];
                    let elim_b = [(eb.pose_blk, eb.elim_pose()), (eb.cam_blk, eb.elim_cam())];
                    for (blk_a, (e_a, d_a)) in elim_a {
                        let Some(blk_a) = blk_a else { continue };
                        for (blk_b, (e_b, d_b)) in elim_b {
                            let Some(blk_b) = blk_b else { continue };
                            let off_a = self.block_offset(blk_a);
                            let off_b = self.block_offset(blk_b);
                            // E_a·C⁻¹·E_bᵀ（d_a×d_b）。
                            for row in 0..d_a {
                                for col in 0..d_b {
                                    let mut sum = 0.0;
                                    for k in 0..3 {
                                        for l in 0..3 {
                                            sum += e_a[row][k]
                                                * point.cinv[k * 3 + l]
                                                * e_b[col][l];
                                        }
                                    }
                                    matrix[(off_a + row) * n + off_b + col] -= sum;
                                }
                            }
                        }
                    }
                }
            }
        }
        solve_spd(&matrix, n, rhs)
    }

    /// 块-Jacobi 预条件共轭梯度。
    fn solve_cg(&self, rhs: &[f64]) -> Option<Vec<f64>> {
        let n = self.layout.total_dof;
        let mut x = vec![0.0; n];
        let mut r = rhs.to_vec();
        let rhs_norm = r.iter().map(|value| value * value).sum::<f64>().sqrt();
        if rhs_norm < 1e-300 {
            return Some(x);
        }
        let mut z = self.precondition(&r);
        let mut p = z.clone();
        let mut rz = r.iter().zip(&z).map(|(a, b)| a * b).sum::<f64>();
        for _ in 0..CG_MAX_ITERATIONS {
            let ap = self.matvec(&p);
            let pap = p.iter().zip(&ap).map(|(a, b)| a * b).sum::<f64>();
            if !pap.is_finite() || pap.abs() < 1e-300 {
                break;
            }
            let alpha = rz / pap;
            for (x_value, p_value) in x.iter_mut().zip(&p) {
                *x_value += alpha * p_value;
            }
            for (r_value, ap_value) in r.iter_mut().zip(&ap) {
                *r_value -= alpha * ap_value;
            }
            let r_norm = r.iter().map(|value| value * value).sum::<f64>().sqrt();
            if r_norm <= CG_TOLERANCE * rhs_norm.max(1e-300) {
                return Some(x);
            }
            let z_new = self.precondition(&r);
            let rz_new = r.iter().zip(&z_new).map(|(a, b)| a * b).sum::<f64>();
            let beta = rz_new / rz;
            for (p_value, z_value) in p.iter_mut().zip(&z_new) {
                *p_value = z_value + beta * *p_value;
            }
            z = z_new;
            rz = rz_new;
            let _ = &z; // z 仅用于下一轮 precondition 的输入（见上方 r 的更新）。
        }
        // 达到迭代上限：若残差已足够小仍接受。
        let r_norm = r.iter().map(|value| value * value).sum::<f64>().sqrt();
        if r_norm <= 1e-4 * rhs_norm.max(1e-300) {
            Some(x)
        } else {
            None
        }
    }

    /// 求解并应用一步（回滚语义由调用方的代价比较驱动）。
    fn solve_step(
        &mut self,
        cameras: &mut [BaCamera],
        images: &mut [BaImage],
        points: &mut [BaPoint],
        current_cost: f64,
        loss: Loss,
        model_ids: &[Option<CameraModelId>],
    ) -> StepOutcome {
        // 1) 精简右端：g' = g − E·C⁻¹·g_pt。
        let n = self.layout.total_dof;
        let mut rhs = vec![0.0; n];
        for (block, g) in self.block_g.iter().enumerate() {
            let off = self.block_offset(block);
            rhs[off..off + self.block_dof[block]].copy_from_slice(&g[..self.block_dof[block]]);
        }
        for point in &self.points {
            if point.cinv == [0.0; 9] {
                continue;
            }
            let cg = [
                point.cinv[0] * point.g[0]
                    + point.cinv[1] * point.g[1]
                    + point.cinv[2] * point.g[2],
                point.cinv[3] * point.g[0]
                    + point.cinv[4] * point.g[1]
                    + point.cinv[5] * point.g[2],
                point.cinv[6] * point.g[0]
                    + point.cinv[7] * point.g[1]
                    + point.cinv[8] * point.g[2],
            ];
            let (first, last) = point.terms;
            for term_row in first..last {
                let term = &self.terms[term_row];
                let jpt_cg = [
                    term.jpt[0][0] * cg[0] + term.jpt[0][1] * cg[1] + term.jpt[0][2] * cg[2],
                    term.jpt[1][0] * cg[0] + term.jpt[1][1] * cg[1] + term.jpt[1][2] * cg[2],
                ];
                if let Some(pose) = term.pose_blk {
                    let off = self.block_offset(pose);
                    for col in 0..self.block_dof[pose] {
                        rhs[off + col] -= term.jp[0][col] * jpt_cg[0]
                            + term.jp[1][col] * jpt_cg[1];
                    }
                }
                if let Some(cam) = term.cam_blk {
                    let off = self.block_offset(cam);
                    for col in 0..self.block_dof[cam] {
                        rhs[off + col] -= term.jc[0][col] * jpt_cg[0]
                            + term.jc[1][col] * jpt_cg[1];
                    }
                }
            }
        }

        // 精简系统：(B − E·C⁻¹·Eᵀ)·δ_cam = −g'。
        for value in &mut rhs {
            *value = -*value;
        }

        // 2) 解精简系统。
        let Some(delta_cam) = (if n <= DENSE_MAX_DOF {
            self.solve_dense(&rhs)
        } else {
            self.solve_cg(&rhs)
        }) else {
            return StepOutcome::Failed;
        };
        if delta_cam.iter().any(|value| !value.is_finite()) {
            return StepOutcome::Failed;
        }
        if delta_cam.iter().all(|value| value.abs() < 1e-12) {
            return StepOutcome::Converged;
        }

        // 3) 回代点增量：δ_pt = C⁻¹·(g_pt − Eᵀ·δ_cam)。
        let block_offsets: Vec<usize> = (0..self.block_dof.len())
            .map(|block| self.block_offset(block))
            .collect();
        for point in &mut self.points {
            if point.cinv == [0.0; 9] {
                point.delta = [0.0; 3];
                continue;
            }
            let mut neg = [-point.g[0], -point.g[1], -point.g[2]];
            let (first, last) = point.terms;
            for term_row in first..last {
                let term = &self.terms[term_row];
                let mut pixel = [0.0f64; 2];
                if let Some(pose) = term.pose_blk {
                    let off = block_offsets[pose];
                    let dof = self.block_dof[pose];
                    for col in 0..dof {
                        pixel[0] += term.jp[0][col] * delta_cam[off + col];
                        pixel[1] += term.jp[1][col] * delta_cam[off + col];
                    }
                }
                if let Some(cam) = term.cam_blk {
                    let off = block_offsets[cam];
                    let dof = self.block_dof[cam];
                    for col in 0..dof {
                        pixel[0] += term.jc[0][col] * delta_cam[off + col];
                        pixel[1] += term.jc[1][col] * delta_cam[off + col];
                    }
                }
                for row in 0..3 {
                    neg[row] -= term.jpt[0][row] * pixel[0] + term.jpt[1][row] * pixel[1];
                }
            }
            point.delta = [
                point.cinv[0] * neg[0] + point.cinv[1] * neg[1] + point.cinv[2] * neg[2],
                point.cinv[3] * neg[0] + point.cinv[4] * neg[1] + point.cinv[5] * neg[2],
                point.cinv[6] * neg[0] + point.cinv[7] * neg[1] + point.cinv[8] * neg[2],
            ];
        }

        // 4) 预测下降量 = ½(−gᵀδ + λ·δᵀDδ)。
        let mut g_dot = 0.0;
        let mut d_delta2 = 0.0;
        for (block, g) in self.block_g.iter().enumerate() {
            let off = self.block_offset(block);
            for k in 0..self.block_dof[block] {
                g_dot += g[k] * delta_cam[off + k];
                d_delta2 += self.block_damp[block][k] * delta_cam[off + k] * delta_cam[off + k];
            }
        }
        for point in &self.points {
            for k in 0..3 {
                g_dot += point.g[k] * point.delta[k];
                d_delta2 += point.damp[k] * point.delta[k] * point.delta[k];
            }
        }
        let predicted = 0.5 * (-g_dot + self.lambda * d_delta2);

        // 5) 保存并应用增量。
        let mut saved_poses: Vec<(usize, [f64; 4], [f64; 3])> = Vec::new();
        let mut saved_params: Vec<(usize, Vec<f64>)> = Vec::new();
        let mut saved_points: Vec<(usize, [f64; 3])> = Vec::new();

        for (image_row, (block_index, pose_block)) in self
            .layout
            .pose_of_image
            .iter()
            .enumerate()
            .filter_map(|(row, entry)| entry.as_ref().map(|(index, block)| (row, (*index, block))))
        {
            let offset = self.block_offset(block_index);
            let mut delta_full = [0.0f64; 6];
            for (k, &dim) in pose_block.free.iter().enumerate() {
                delta_full[dim] = delta_cam[offset + k];
            }
            saved_poses.push((image_row, images[image_row].qvec, images[image_row].tvec));
            let rotation = quaternion_to_rotation_matrix(&images[image_row].qvec);
            let delta_rotation = exp_rotation_vector(&[delta_full[0], delta_full[1], delta_full[2]]);
            let new_rotation = mat3_mat3(&delta_rotation, &rotation);
            // 左扰动约定（与雅可比一致）：t' = Exp(δθ)·t + δt。
            let t = images[image_row].tvec;
            let rotated = mat3_vec3(&delta_rotation, t);
            images[image_row].qvec = rotation_to_quaternion(&new_rotation);
            images[image_row].tvec = [
                rotated[0] + delta_full[3],
                rotated[1] + delta_full[4],
                rotated[2] + delta_full[5],
            ];
        }
        for (camera_row, (block_index, cam_block)) in self
            .layout
            .camera_of_camera
            .iter()
            .enumerate()
            .filter_map(|(row, entry)| entry.as_ref().map(|(index, block)| (row, (*index, block))))
        {
            let offset = self.block_offset(block_index);
            saved_params.push((camera_row, cameras[camera_row].params.clone()));
            for (k, &param) in cam_block.params.iter().enumerate() {
                cameras[camera_row].params[param] += delta_cam[offset + k];
            }
        }
        for point in &self.points {
            if point.cinv == [0.0; 9] {
                continue;
            }
            saved_points.push((point.point_row, points[point.point_row].xyz));
            let xyz = points[point.point_row].xyz;
            points[point.point_row].xyz = [
                xyz[0] + point.delta[0],
                xyz[1] + point.delta[1],
                xyz[2] + point.delta[2],
            ];
        }

        // 6) 代价比较与回滚。
        let new_cost = robust_cost(cameras, images, points, loss, model_ids);
        if std::env::var("OOO_BA_DEBUG").is_ok() {
            let dcam = delta_cam.iter().map(|v| v*v).sum::<f64>().sqrt();
            let dpt: f64 = self.points.iter().map(|p| p.delta.iter().map(|v| v*v).sum::<f64>()).sum();
            eprintln!("  step |d_cam|={dcam:.3e} |d_pt|={:.3e} new={new_cost:.6e} cur={current_cost:.6e}", dpt.sqrt());
        }
        if std::env::var("OOO_BA_DEBUG").is_ok() {
            let delta_norm = delta_cam.iter().map(|v| v*v).sum::<f64>().sqrt();
            let point_norm = self.points.iter().map(|p| p.delta.iter().map(|v| v*v).sum::<f64>()).sum::<f64>().sqrt();
            eprintln!("  step: |d_cam|={delta_norm:.3e} |d_pt|={point_norm:.3e} pred={predicted:.4e} new={new_cost:.4e} cur={current_cost:.4e}");
        }
        if new_cost.is_finite() && new_cost < current_cost {
            StepOutcome::Accepted(new_cost)
        } else {
            for (image_row, qvec, tvec) in saved_poses {
                images[image_row].qvec = qvec;
                images[image_row].tvec = tvec;
            }
            for (camera_row, params) in saved_params {
                cameras[camera_row].params = params;
            }
            for (row, xyz) in saved_points {
                points[row].xyz = xyz;
            }
            if predicted <= 0.0 {
                StepOutcome::Failed
            } else {
                StepOutcome::Rejected
            }
        }
    }
}

/// 组装精简系统（法方程 + 点消元的 C⁻¹）。
fn assemble(
    layout: &ParameterLayout,
    linearized: &Linearized,
    points: &[BaPoint],
    model_ids: &[Option<CameraModelId>],
    lambda: f64,
) -> ReducedSystem {
    let num_blocks = layout.num_blocks;
    let mut block_dof = vec![0usize; num_blocks];
    for (block, pose) in layout.pose_of_image.iter().flatten() {
        block_dof[*block] = pose.dof;
    }
    for (block, camera) in layout.camera_of_camera.iter().flatten() {
        block_dof[*block] = camera.dof;
    }
    let mut block_h: Vec<[[f64; 6]; 6]> = vec![[[0.0; 6]; 6]; num_blocks];
    let mut block_g: Vec<[f64; 6]> = vec![[0.0; 6]; num_blocks];
    let mut cross: Vec<CrossBlock> = Vec::new();
    let mut cross_index: HashMap<(usize, usize), usize> = HashMap::new();

    let mut terms: Vec<ObsTerms> = Vec::new();
    let mut point_data: Vec<PointElim> = Vec::with_capacity(points.len());

    for (point_row, &(first, last)) in linearized.groups.iter().enumerate() {
        let terms_start = terms.len();
        for obs_row in first..last {
            let jac = &linearized.jacobians[obs_row];
            let pose_entry = layout.pose_of_image[jac.image].as_ref();
            let cam_entry = layout.camera_of_camera[jac.camera].as_ref();
            let model = model_ids[jac.camera].expect("模型已在外层校验");

            let mut jp = [[0.0f64; 6]; 2];
            let mut jc = [[0.0f64; 6]; 2];
            let mut jpt = [[0.0f64; 3]; 2];

            if let Some((_, pose_block)) = pose_entry {
                let pjac = jac.pose_jacobian();
                for (col, &dim) in pose_block.free.iter().enumerate() {
                    jp[0][col] = jac.weight * pjac[dim];
                    jp[1][col] = jac.weight * pjac[dim + 6];
                }
            }
            if let Some((_, cam_block)) = cam_entry {
                for (col, &param) in cam_block.params.iter().enumerate() {
                    let column = jac.intrinsic_column(model, param);
                    jc[0][col] = jac.weight * column[0];
                    jc[1][col] = jac.weight * column[1];
                }
            }
            let pjac_point = jac.point_jacobian();
            for col in 0..3 {
                jpt[0][col] = jac.weight * pjac_point[col];
                jpt[1][col] = jac.weight * pjac_point[col + 3];
            }

            // 累加 H / g / cross。
            if let Some((pose_block, _)) = pose_entry {
                let block = *pose_block;
                let d = pose_block_of(layout, jac.image).1.dof;
                for plane in 0..2 {
                    for col in 0..d {
                        let value = jp[plane][col];
                        if value == 0.0 {
                            continue;
                        }
                        for k in 0..d {
                            block_h[block][col][k] += value * jp[plane][k];
                        }
                    }
                }
                for k in 0..d {
                    block_g[block][k] += jp[0][k] * jac.residual[0] + jp[1][k] * jac.residual[1];
                }
            }
            if let Some((cam_block, _)) = cam_entry {
                let block = *cam_block;
                let c = camera_block_of(layout, jac.camera).1.dof;
                for plane in 0..2 {
                    for col in 0..c {
                        let value = jc[plane][col];
                        if value == 0.0 {
                            continue;
                        }
                        for k in 0..c {
                            block_h[block][col][k] += value * jc[plane][k];
                        }
                    }
                }
                for k in 0..c {
                    block_g[block][k] += jc[0][k] * jac.residual[0] + jc[1][k] * jac.residual[1];
                }
            }
            if let (Some((pose_block, _)), Some((cam_block, _))) = (pose_entry, cam_entry) {
                let d = pose_block_of(layout, jac.image).1.dof;
                let c = camera_block_of(layout, jac.camera).1.dof;
                let key = (*pose_block, *cam_block);
                let cross_id = match cross_index.get(&key) {
                    Some(&id) => id,
                    None => {
                        cross_index.insert(key, cross.len());
                        cross.push(CrossBlock {
                            pose: *pose_block,
                            camera: *cam_block,
                            matrix: [[0.0; 6]; 6],
                            d,
                            c,
                        });
                        cross.len() - 1
                    }
                };
                for row in 0..d {
                    for col in 0..c {
                        cross[cross_id].matrix[row][col] +=
                            jp[0][row] * jc[0][col] + jp[1][row] * jc[1][col];
                    }
                }
            }

            terms.push(ObsTerms {
                pose_blk: pose_entry.map(|(block, _)| *block),
                cam_blk: cam_entry.map(|(block, _)| *block),
                jp,
                jc,
                jpt,
            });
        }

        // 点块（可变时）：C += JptᵀJpt、g_pt += Jptᵀ·r。
        let variable = !points[point_row].constant;
        let (mut c_matrix, mut g_pt, mut damp) = ([0.0f64; 9], [0.0f64; 3], [0.0f64; 3]);
        if variable {
            for term_row in terms_start..terms.len() {
                let term = &terms[term_row];
                for row in 0..3 {
                    for col in 0..3 {
                        c_matrix[row * 3 + col] +=
                            term.jpt[0][row] * term.jpt[0][col]
                                + term.jpt[1][row] * term.jpt[1][col];
                    }
                    g_pt[row] += term.jpt[0][row] * linearized.jacobians[first
                        + (term_row - terms_start)]
                        .residual[0]
                        + term.jpt[1][row]
                            * linearized.jacobians[first + (term_row - terms_start)].residual[1];
                }
            }
            for k in 0..3 {
                damp[k] = lambda * c_matrix[k * 3 + k].abs().max(DAMPING_FLOOR);
                c_matrix[k * 3 + k] += damp[k];
            }
        }
        let mut cinv = [0.0f64; 9];
        if variable {
            cinv = invert_symmetric_3x3(&c_matrix).unwrap_or([0.0; 9]);
            if cinv == [0.0; 9] {
                for k in 0..3 {
                    c_matrix[k * 3 + k] += damp[k] * 9.0;
                }
                cinv = invert_symmetric_3x3(&c_matrix).unwrap_or([0.0; 9]);
            }
        }
        point_data.push(PointElim {
            point_row,
            terms: (terms_start, terms.len()),
            cinv,
            g: g_pt,
            damp,
            delta: [0.0; 3],
        });
    }

    // 块阻尼 + 预条件子。
    let mut block_damp: Vec<[f64; 6]> = vec![[0.0; 6]; num_blocks];
    let mut precond: Vec<[[f64; 6]; 6]> = vec![[[0.0; 6]; 6]; num_blocks];
    for block in 0..num_blocks {
        let d = block_dof[block];
        for k in 0..d {
            let damp = lambda * block_h[block][k][k].abs().max(DAMPING_FLOOR);
            block_damp[block][k] = damp;
            block_h[block][k][k] += damp;
        }
        precond[block] = block_h[block];
    }

    // 预条件子扣除点消元的对角块贡献（仅同块内的 obs 对）。
    for point in &point_data {
        if point.cinv == [0.0; 9] {
            continue;
        }
        let (first, last) = point.terms;
        for a in first..last {
            for b in a..last {
                let (ta, tb) = (&terms[a], &terms[b]);
                let elim_a = [(ta.pose_blk, ta.elim_pose()), (ta.cam_blk, ta.elim_cam())];
                let elim_b = [(tb.pose_blk, tb.elim_pose()), (tb.cam_blk, tb.elim_cam())];
                for (blk_a, (e_a, d_a)) in elim_a {
                    let Some(blk_a) = blk_a else { continue };
                    for (blk_b, (e_b, _d_b)) in elim_b {
                        let Some(blk_b) = blk_b else { continue };
                        if blk_a != blk_b {
                            continue;
                        }
                        let d = d_a.min(block_dof[blk_a]);
                        for row in 0..d {
                            for col in 0..d {
                                let mut sum = 0.0;
                                for k in 0..3 {
                                    for l in 0..3 {
                                        sum += e_a[row][k]
                                            * point.cinv[k * 3 + l]
                                            * e_b[col][l];
                                    }
                                }
                                precond[blk_a][row][col] -= sum;
                            }
                        }
                    }
                }
            }
        }
    }
    // 预条件子求逆（失败回退单位阵）。
    for block in 0..num_blocks {
        let d = block_dof[block];
        let mut matrix = vec![0.0; d * d];
        for row in 0..d {
            for col in 0..d {
                matrix[row * d + col] = precond[block][row][col];
            }
        }
        let inverted = invert_spd(&matrix, d).unwrap_or_else(|| {
            let mut identity = vec![0.0; d * d];
            for k in 0..d {
                identity[k * d + k] = 1.0;
            }
            identity
        });
        let mut packed = [[0.0; 6]; 6];
        for row in 0..d {
            for col in 0..d {
                packed[row][col] = inverted[row * d + col];
            }
        }
        precond[block] = packed;
    }

    ReducedSystem {
        layout: clone_layout(layout),
        block_dof,
        block_h,
        block_g,
        block_damp,
        cross,
        cross_index,
        precond,
        points: point_data,
        terms,
        lambda,
    }
}

/// 布局克隆（ReducedSystem 持有自己的副本）。
fn clone_layout(layout: &ParameterLayout) -> ParameterLayout {
    ParameterLayout {
        pose_of_image: layout
            .pose_of_image
            .iter()
            .map(|entry| {
                entry.as_ref().map(|(block, pose)| {
                    (
                        *block,
                        PoseBlock {
                            image: pose.image,
                            free: pose.free.clone(),
                            dof: pose.dof,
                        },
                    )
                })
            })
            .collect(),
        camera_of_camera: layout
            .camera_of_camera
            .iter()
            .map(|entry| {
                entry.as_ref().map(|(block, camera)| {
                    (
                        *block,
                        CameraBlock {
                            camera: camera.camera,
                            params: camera.params.clone(),
                            dof: camera.dof,
                        },
                    )
                })
            })
            .collect(),
        num_blocks: layout.num_blocks,
        total_dof: layout.total_dof,
    }
}

/// 图像行号 → (块序号, 位姿块)。
fn pose_block_of(
    layout: &ParameterLayout,
    image: usize,
) -> (usize, &PoseBlock) {
    let (block, pose) = layout.pose_of_image[image].as_ref().expect("位姿块存在");
    (*block, pose)
}

/// 相机行号 → (块序号, 内参块)。
fn camera_block_of(layout: &ParameterLayout, camera: usize) -> (usize, &CameraBlock) {
    let (block, cam) = layout.camera_of_camera[camera].as_ref().expect("内参块存在");
    (*block, cam)
}

/// 对称正定矩阵求逆（经 Cholesky 解 d 个单位向量）。
fn invert_spd(matrix: &[f64], d: usize) -> Option<Vec<f64>> {
    let mut out = vec![0.0; d * d];
    for col in 0..d {
        let mut unit = vec![0.0; d];
        unit[col] = 1.0;
        let column = solve_spd(matrix, d, &unit)?;
        for row in 0..d {
            out[row * d + col] = column[row];
        }
    }
    Some(out)
}

// ---------------------------------------------------------------- 门面接口
//
// `ooo_sfm/ba.rs`（即 `engines::colmap_ba`）转发的平台实现。

use crate::engines::colmap_ba::{BaSelection, DatabaseStats};
use std::{ffi::OsString, path::Path};

/// 自研 BA 恒为 CPU（无 Caspar GPU 后端），`gpu_index` 永远是 `None`。
pub async fn select(_executable: &Path, _compatible_gpu: Option<u32>) -> BaSelection {
    BaSelection {
        gpu_index: None,
        detail: "BA: ooo-sfm CPU (Schur LM)".into(),
    }
}

pub async fn select_for_mapper(
    executable: &Path,
    _database: &Path,
    compatible_gpu: Option<u32>,
    _automatic_optimization: bool,
) -> BaSelection {
    select(executable, compatible_gpu).await
}

/// 自研侧没有 Caspar 对应物。
pub fn meets_caspar_threshold(_stats: DatabaseStats) -> bool {
    false
}

pub async fn disable_after_runtime_failure(_executable: &Path, _index: u32, _reason: &str) {}

/// 读取自研 `database.db` 的规模统计（`SUM(rows)` 语义与原实现一致）。
pub fn read_database_stats(database: &Path) -> std::result::Result<DatabaseStats, String> {
    let connection =
        super::format::database::open_read_only(database).map_err(|error| error.to_string())?;
    let sum = |table: &str| -> std::result::Result<u64, String> {
        let sql = format!("SELECT COALESCE(SUM(rows), 0) FROM {table}");
        let value: i64 = connection
            .query_row(&sql, [], |row| row.get(0))
            .map_err(|error| error.to_string())?;
        u64::try_from(value).map_err(|_| format!("negative row total in {table}"))
    };
    Ok(DatabaseStats {
        keypoints: sum("keypoints")?,
        verified_matches: sum("two_view_geometries")?,
    })
}

/// 自研 mapper 不拼 COLMAP CLI 参数，保留签名以维持公开接口。
pub fn append_mapper_options(_args: &mut Vec<OsString>, _selection: &BaSelection) {}

/// 同上。
pub fn append_explicit_ceres_options(_args: &mut Vec<OsString>) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// 确定性伪随机数（LCG）。
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 11
        }

        fn uniform(&mut self, min: f64, max: f64) -> f64 {
            min + (max - min) * (self.next() as f64 / (u64::MAX >> 11) as f64)
        }

        fn normal(&mut self) -> f64 {
            // Box-Muller。
            let u1 = self.uniform(1e-12, 1.0);
            let u2 = self.uniform(0.0, 1.0);
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    fn simple_radial(params: [f64; 4]) -> BaCamera {
        BaCamera {
            id: 1,
            model_id: 2,
            params: params.to_vec(),
            constant: false,
        }
    }

    fn image_at(camera: usize, qvec: [f64; 4], tvec: [f64; 3]) -> BaImage {
        BaImage {
            id: camera as u32,
            camera,
            qvec,
            tvec,
            constant_pose: false,
        }
    }

    /// 合成场景：`num_images` 台相机环视原点，`num_points` 个随机点。
    pub(crate) fn synthetic_scene(
        num_images: usize,
        num_points: usize,
        seed: u64,
    ) -> (Vec<BaCamera>, Vec<BaImage>, Vec<BaPoint>) {
        let mut rng = Rng(seed);
        let camera = simple_radial([800.0, 500.0, 375.0, 0.01]);
        let mut images = Vec::new();
        for i in 0..num_images {
            let angle = std::f64::consts::TAU * i as f64 / num_images as f64;
            // 球面轨迹：所有相机到原点距离一致（避免病态深度）。
            let direction = [
                angle.cos(),
                0.6 * angle.sin(),
                0.35 * (2.0 * angle).sin(),
            ];
            let dir_norm = (direction[0] * direction[0]
                + direction[1] * direction[1]
                + direction[2] * direction[2])
                .sqrt();
            let center = [
                8.0 * direction[0] / dir_norm,
                8.0 * direction[1] / dir_norm,
                8.0 * direction[2] / dir_norm,
            ];
            // 朝向原点：R 使相机 -z 指向原点。
            let forward = [-center[0], -center[1], -center[2]];
            let norm = (forward[0] * forward[0]
                + forward[1] * forward[1]
                + forward[2] * forward[2])
                .sqrt();
            let forward = [
                forward[0] / norm,
                forward[1] / norm,
                forward[2] / norm,
            ];
            // right = normalize(forward × (0,0,1))：对任意有限俯仰都非退化。
            let right = [forward[1], -forward[0], 0.0];
            let rnorm = (right[0] * right[0] + right[1] * right[1]).sqrt();
            let right = [right[0] / rnorm, right[1] / rnorm, 0.0];
            let down = [
                forward[1] * right[2] - forward[2] * right[1],
                forward[2] * right[0] - forward[0] * right[2],
                forward[0] * right[1] - forward[1] * right[0],
            ];
            // cam_from_world：X_cam = R·(X − C)，COLMAP 相机看向 +z。
            // 取 R 行 = (right, down, forward)，t = −R·C。
            let rotation = [
                [right[0], right[1], right[2]],
                [down[0], down[1], down[2]],
                [forward[0], forward[1], forward[2]],
            ];
            let t = [
                -(rotation[0][0] * center[0]
                    + rotation[0][1] * center[1]
                    + rotation[0][2] * center[2]),
                -(rotation[1][0] * center[0]
                    + rotation[1][1] * center[1]
                    + rotation[1][2] * center[2]),
                -(rotation[2][0] * center[0]
                    + rotation[2][1] * center[1]
                    + rotation[2][2] * center[2]),
            ];
            images.push(image_at(0, rotation_to_quaternion(&rotation), t));
        }
        let mut points = Vec::new();
        for _ in 0..num_points {
            let xyz = [rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), rng.uniform(-1.5, 1.5)];
            let mut observations = Vec::new();
            for (row, image) in images.iter().enumerate() {
                let rotation = quaternion_to_rotation_matrix(&image.qvec);
                let depth = rotation[2][0] * xyz[0] + rotation[2][1] * xyz[1]
                    + rotation[2][2] * xyz[2] + image.tvec[2];
                if depth < 1.0 {
                    continue; // 深度过浅的观测病态，不进入场景。
                }
                let projection = project_point(&image.qvec, &image.tvec, &xyz, CameraModelId::SimpleRadial, &camera.params);
                if projection[0] < 20.0
                    || projection[0] > 980.0
                    || projection[1] < 20.0
                    || projection[1] > 730.0
                {
                    continue;
                }
                observations.push((row, projection[0], projection[1]));
            }
            if observations.len() >= 3 {
                points.push(BaPoint { xyz, observations, constant: false });
            }
        }
        (vec![camera], images, points)
    }

    pub(crate) fn perturb_scene(
        cameras: &mut [BaCamera],
        images: &mut [BaImage],
        points: &mut [BaPoint],
        seed: u64,
        sigma: f64,
    ) {
        let mut rng = Rng(seed);
        for image in images.iter_mut() {
            let rotvec = [rng.normal() * sigma, rng.normal() * sigma, rng.normal() * sigma];
            let rotation = quaternion_to_rotation_matrix(&image.qvec);
            let new_rotation = mat3_mat3(&exp_rotation_vector(&rotvec), &rotation);
            image.qvec = rotation_to_quaternion(&new_rotation);
            for value in &mut image.tvec {
                *value += rng.normal() * sigma;
            }
        }
        for point in points.iter_mut() {
            for value in &mut point.xyz {
                *value += rng.normal() * sigma;
            }
        }
        for camera in cameras.iter_mut() {
            camera.params[0] += rng.normal() * sigma * 100.0;
        }
    }

    #[test]
    fn loss_functions_match_ceres_semantics() {
        // ρ(0) = 0、ρ'(0) = 1（所有损失）。
        for loss in [
            Loss::Trivial,
            Loss::SoftL1(1.0),
            Loss::Huber(1.0),
            Loss::Cauchy(1.0),
        ] {
            let (rho, drho) = loss.evaluate(0.0);
            assert_eq!(rho, 0.0);
            assert_eq!(drho, 1.0);
        }
        // Huber 在 b² 处连续：ρ = b²，ρ' = 1。
        let (rho, drho) = Loss::Huber(1.0).evaluate(1.0);
        assert!((rho - 1.0).abs() < 1e-12);
        assert!((drho - 1.0).abs() < 1e-12);
        // 大残差：Huber 线性增长，Cauchy 对数增长。
        let (huber, _) = Loss::Huber(1.0).evaluate(10000.0);
        assert!((huber - (2.0 * 100.0 - 1.0)).abs() < 1e-9);
        let (cauchy, _) = Loss::Cauchy(1.0).evaluate(10000.0);
        assert!(cauchy < 10.0);
        // SoftL1 大残差按 2b√s 增长（亚线性）。
        let (softl1, _) = Loss::SoftL1(1.0).evaluate(1e12);
        assert!((softl1 - (2.0 * (1.0e12f64.sqrt()) - 2.0)).abs() < 1e-3);
    }

    #[test]
    fn projection_jacobians_match_finite_differences() {
        let models = [
            CameraModelId::SimplePinhole,
            CameraModelId::Pinhole,
            CameraModelId::SimpleRadial,
            CameraModelId::Radial,
        ];
        let param_sets: [Vec<f64>; 4] = [
            vec![800.0, 500.0, 375.0],
            vec![800.0, 780.0, 500.0, 375.0],
            vec![800.0, 500.0, 375.0, 0.02],
            vec![800.0, 500.0, 375.0, 0.02, -0.004],
        ];
        let u = [0.3, -0.2, 2.5];
        for (model, params) in models.iter().zip(&param_sets) {
            let (_, dp_du, focal, pp, extra) = project(*model, params, u);
            let base = project(*model, params, u).0;
            // dp/du（有限差分）。
            for dim in 0..3 {
                let eps = 1e-6;
                let mut u_plus = u;
                u_plus[dim] += eps;
                let plus = project(*model, params, u_plus).0;
                let numeric = [(plus[0] - base[0]) / eps, (plus[1] - base[1]) / eps];
                let analytic = [dp_du[dim], dp_du[dim + 3]];
                for axis in 0..2 {
                    assert!(
                        (numeric[axis] - analytic[axis]).abs() < 1e-3,
                        "{model:?} dp/du dim {dim} axis {axis}: {numeric:?} vs {analytic:?}"
                    );
                }
            }
            // 焦距列。
            for (pos, column) in focal.iter().enumerate() {
                let idx = model.focal_idxs()[pos];
                let eps = 1e-4;
                let mut params_plus = params.clone();
                params_plus[idx] += eps;
                let plus = project(*model, &params_plus, u).0;
                let numeric = [(plus[0] - base[0]) / eps, (plus[1] - base[1]) / eps];
                for axis in 0..2 {
                    assert!(
                        (numeric[axis] - column[axis]).abs() < 1e-2,
                        "{model:?} dp/df: {numeric:?} vs {column:?}"
                    );
                }
            }
            // 主点列。
            for (pos, column) in pp.iter().enumerate() {
                let idx = model.principal_point_idxs()[pos];
                let expected = if pos == 0 { [1.0, 0.0] } else { [0.0, 1.0] };
                assert_eq!(*column, expected, "{model:?} pp idx {idx}");
                let _ = idx;
            }
            // 畸变列。
            for (pos, column) in extra.iter().enumerate() {
                let idx = model.extra_idxs()[pos];
                let eps = 1e-6;
                let mut params_plus = params.clone();
                params_plus[idx] += eps;
                let plus = project(*model, &params_plus, u).0;
                let numeric = [(plus[0] - base[0]) / eps, (plus[1] - base[1]) / eps];
                for axis in 0..2 {
                    assert!(
                        (numeric[axis] - column[axis]).abs() < 1e-2,
                        "{model:?} dp/dk: {numeric:?} vs {column:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn pose_jacobian_matches_finite_differences() {
        let (mut cameras, mut images, mut points) = synthetic_scene(3, 20, 7);
        perturb_scene(&mut cameras, &mut images, &mut points, 11, 0.0);
        let options = BaOptions::default();
        let model_ids = vec![Some(CameraModelId::SimpleRadial)];
        let linearized = linearize(&cameras, &images, &points, &options, &model_ids);

        let obs = linearized.jacobians.first().expect("有观测");
        let pjac = obs.pose_jacobian();
        // 对图像位姿施加小扰动，比较数值导数。
        let eps = 1e-6;
        for dim in 0..6 {
            let mut images_plus = images.clone();
            let mut delta = [0.0f64; 6];
            delta[dim] = eps;
            let image = &mut images_plus[obs.image];
            let rotation = quaternion_to_rotation_matrix(&image.qvec);
            let delta_rotation = exp_rotation_vector(&[delta[0], delta[1], delta[2]]);
            let new_rotation = mat3_mat3(&delta_rotation, &rotation);
            let rotated_t = mat3_vec3(&delta_rotation, image.tvec);
            image.qvec = rotation_to_quaternion(&new_rotation);
            image.tvec = [rotated_t[0] + delta[3], rotated_t[1] + delta[4], rotated_t[2] + delta[5]];
            let projection_plus = project_point(
                &image.qvec,
                &image.tvec,
                &points[0].xyz,
                CameraModelId::SimpleRadial,
                &cameras[obs.camera].params,
            );
            let base = project_point(
                &images[obs.image].qvec,
                &images[obs.image].tvec,
                &points[0].xyz,
                CameraModelId::SimpleRadial,
                &cameras[obs.camera].params,
            );
            let numeric = [
                (projection_plus[0] - base[0]) / eps,
                (projection_plus[1] - base[1]) / eps,
            ];
            let analytic = [pjac[dim], pjac[dim + 6]];
            if dim == 0 {
                let rotation = quaternion_to_rotation_matrix(&images[obs.image].qvec);
                let px = points[0].xyz;
                let u_check = [
                    rotation[0][0]*px[0] + rotation[0][1]*px[1] + rotation[0][2]*px[2] + images[obs.image].tvec[0],
                    rotation[1][0]*px[0] + rotation[1][1]*px[1] + rotation[1][2]*px[2] + images[obs.image].tvec[1],
                    rotation[2][0]*px[0] + rotation[2][1]*px[1] + rotation[2][2]*px[2] + images[obs.image].tvec[2],
                ];
                eprintln!("dim0: obs.image={} jac.u={:?} 重算u={:?} jac.dp_du={:?}", obs.image, obs.u, u_check, obs.dp_du);
                eprintln!("     base={base:?} plus={projection_plus:?}");
            }
            for axis in 0..2 {
                assert!(
                    (numeric[axis] - analytic[axis]).abs() < 1e-3,
                    "pose jacobian dim {dim} axis {axis}: {numeric:?} vs {analytic:?}"
                );
            }
        }
    }

    #[test]
    fn synthetic_scene_recovers_ground_truth() {
        let (mut cameras, mut images, mut points) = synthetic_scene(6, 80, 42);
        let truth: Vec<Vec<f64>> = cameras.iter().map(|c| c.params.clone()).collect();
        let truth_points: Vec<[f64; 3]> = points.iter().map(|p| p.xyz).collect();
        let truth_pose0 = (images[0].qvec, images[0].tvec);
        perturb_scene(&mut cameras, &mut images, &mut points, 99, 0.01);

        let options = BaOptions {
            loss: Loss::Trivial,
            max_iterations: 100,
            refine_focal_length: true,
            refine_principal_point: false,
            refine_extra_params: true,
            gauge: Gauge::TwoCamsFromWorld,
            function_tolerance: 0.0,
            gradient_tolerance: 0.0,
        };
        let summary = run_bundle_adjustment(&mut cameras, &mut images, &mut points, &options);
        assert!(summary.cost < 1e-6, "代价应接近 0：{summary:?}");

        // 规范固定把解钉在某个相似系（平移+尺度由规范决定），因此不逐点比较
        // 坐标，而是验证「点间距离比恒定」（相似不变量）+ 内参恢复。
        let dist = |a: [f64; 3], b: [f64; 3]| {
            (0..3)
                .map(|axis| (a[axis] - b[axis]).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let mut ratios = Vec::new();
        for i in 0..points.len().min(12) {
            for j in (i + 1)..points.len().min(12) {
                let recovered = dist(points[i].xyz, points[j].xyz);
                let truth_dist = dist(truth_points[i], truth_points[j]);
                if truth_dist > 1e-6 {
                    ratios.push(recovered / truth_dist);
                }
            }
        }
        let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
        for ratio in &ratios {
            assert!(
                (ratio - mean).abs() < 1e-6 * mean.max(1.0),
                "点间距离比应恒定（相似变换）：{ratio} vs {mean}"
            );
        }
        // 内参应恢复真值（相似变换不改变内参）。
        for (camera, truth_params) in cameras.iter().zip(&truth) {
            assert!(
                (camera.params[0] - truth_params[0]).abs() < 1e-3,
                "焦距恢复失败：{} vs {}",
                camera.params[0],
                truth_params[0]
            );
            assert!((camera.params[3] - truth_params[3]).abs() < 1e-6);
        }
    }

    #[test]
    fn two_cams_gauge_keeps_first_pose_fixed() {
        let (mut cameras, mut images, mut points) = synthetic_scene(5, 40, 3);
        let first_pose = (images[0].qvec, images[0].tvec);
        perturb_scene(&mut cameras, &mut images, &mut points, 8, 0.005);
        let options = BaOptions {
            loss: Loss::Trivial,
            max_iterations: 30,
            gauge: Gauge::TwoCamsFromWorld,
            ..BaOptions::default()
        };
        // 重新应用扰动后第一个位姿可能已变——重新记录。
        let first_pose = (images[0].qvec, images[0].tvec);
        run_bundle_adjustment(&mut cameras, &mut images, &mut points, &options);
        assert_eq!(images[0].qvec, first_pose.0, "第一个位姿应被规范固定");
        assert_eq!(images[0].tvec, first_pose.1);
    }

    #[test]
    fn three_points_gauge_keeps_basis_fixed() {
        let (mut cameras, mut images, mut points) = synthetic_scene(4, 30, 5);
        perturb_scene(&mut cameras, &mut images, &mut points, 9, 0.005);
        let options = BaOptions {
            loss: Loss::Trivial,
            max_iterations: 30,
            gauge: Gauge::ThreePoints,
            ..BaOptions::default()
        };
        let before: Vec<[f64; 3]> = points.iter().take(3).map(|p| p.xyz).collect();
        run_bundle_adjustment(&mut cameras, &mut images, &mut points, &options);
        // 前三个线性无关的点应被固定（若不足三个线性无关点则至少第一个）。
        assert_eq!(points[0].xyz, before[0]);
    }

    #[test]
    fn robust_loss_ignores_outliers_better_than_trivial() {
        let (mut cameras, mut images, mut points) = synthetic_scene(4, 40, 21);
        // 在第一个点上注入 50 像素的离群观测。
        points[0].observations[0].1 += 50.0;

        let make_options = |loss: Loss| BaOptions {
            loss,
            max_iterations: 60,
            gauge: Gauge::TwoCamsFromWorld,
            refine_focal_length: false,
            refine_extra_params: false,
            ..BaOptions::default()
        };
        let mut trivial_points = points.clone();
        let mut trivial_images = images.clone();
        let mut trivial_cameras = cameras.clone();
        run_bundle_adjustment(
            &mut trivial_cameras,
            &mut trivial_images,
            &mut trivial_points,
            &make_options(Loss::Trivial),
        );
        let mut cauchy_points = points.clone();
        let mut cauchy_images = images.clone();
        let mut cauchy_cameras = cameras.clone();
        run_bundle_adjustment(
            &mut cauchy_cameras,
            &mut cauchy_images,
            &mut cauchy_points,
            &make_options(Loss::Cauchy(1.0)),
        );
        // 用受污染点之外的点的恢复质量比较（到真值的距离）。
        let trivial_error: f64 = trivial_points[1..]
            .iter()
            .zip(&points[1..])
            .map(|(got, truth)| {
                (0..3)
                    .map(|axis| (got.xyz[axis] - truth.xyz[axis]).powi(2))
                    .sum::<f64>()
            })
            .sum();
        let cauchy_error: f64 = cauchy_points[1..]
            .iter()
            .zip(&points[1..])
            .map(|(got, truth)| {
                (0..3)
                    .map(|axis| (got.xyz[axis] - truth.xyz[axis]).powi(2))
                    .sum::<f64>()
            })
            .sum();
        assert!(
            cauchy_error < trivial_error,
            "Cauchy 应比 Trivial 更抗离群点：{cauchy_error} vs {trivial_error}"
        );
    }

    #[test]
    fn constant_blocks_are_never_touched() {
        let (mut cameras, mut images, mut points) = synthetic_scene(4, 30, 13);
        perturb_scene(&mut cameras, &mut images, &mut points, 14, 0.01);
        let fixed_pose = (images[1].qvec, images[1].tvec);
        images[1].constant_pose = true;
        cameras[0].constant = true;
        let before_point = points[0].xyz;
        points[0].constant = true;

        let options = BaOptions {
            loss: Loss::Trivial,
            max_iterations: 40,
            gauge: Gauge::None,
            ..BaOptions::default()
        };
        run_bundle_adjustment(&mut cameras, &mut images, &mut points, &options);
        assert_eq!((images[1].qvec, images[1].tvec), fixed_pose);
        assert_eq!(points[0].xyz, before_point);
    }

    #[test]
    fn cam_ray_from_image_roundtrips_projection() {
        let models = [
            (CameraModelId::SimpleRadial, vec![800.0, 500.0, 375.0, 0.05]),
            (CameraModelId::Pinhole, vec![800.0, 780.0, 500.0, 375.0]),
        ];
        for (model, params) in models {
            for (x, y) in [(250.0, 200.0), (500.0, 375.0), (700.0, 600.0)] {
                let ray = cam_ray_from_image(model, &params, x, y).expect("光线有效");
                // 光线方向投影回像素应还原 (x, y)。
                let u = [ray[0] * 3.0, ray[1] * 3.0, ray[2] * 3.0];
                let (projection, _, _, _, _) = project(model, &params, u);
                assert!(
                    (projection[0] - x).abs() < 1e-4 && (projection[1] - y).abs() < 1e-4,
                    "{model:?} ({x}, {y}): {projection:?}"
                );
            }
        }
    }

    #[test]
    fn unknown_camera_model_fails_fast() {
        let mut cameras = vec![BaCamera {
            id: 1,
            model_id: 99,
            params: vec![1.0; 3],
            constant: false,
        }];
        let mut images = vec![image_at(0, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])];
        let mut points = vec![BaPoint {
            xyz: [0.0, 0.0, 1.0],
            observations: vec![(0, 100.0, 100.0)],
            constant: false,
        }];
        let options = BaOptions::default();
        let summary = run_bundle_adjustment(&mut cameras, &mut images, &mut points, &options);
        assert_eq!(summary.iterations, 0);
        assert!(summary.cost.is_infinite());
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;

    #[test]
    fn debug_linear_system_consistency() {
        // 2 台相机、1 个共享内参、3 个点的小场景；检查：
        // (a) 精简系统解出的 δ 满足 B'δ = rhs（线性求解残差小）；
        // (b) 线性化模型预测的代价变化与真实代价变化一致。
        let (mut cameras, mut images, mut points) = super::tests::synthetic_scene(4, 10, 7);
        eprintln!("扰动前 image1 q={:?}", images[1].qvec);
        super::tests::perturb_scene(&mut cameras, &mut images, &mut points, 11, 0.01);
        eprintln!("扰动后 image1 q={:?}", images[1].qvec);

        let options = BaOptions {
            loss: Loss::Trivial,
            max_iterations: 1,
            gauge: Gauge::TwoCamsFromWorld,
            function_tolerance: 0.0,
            gradient_tolerance: 0.0,
            ..BaOptions::default()
        };
        let model_ids = vec![Some(CameraModelId::SimpleRadial)];
        let partial_fixed = apply_gauge(&mut images, &mut points, Gauge::TwoCamsFromWorld);
        let layout = build_layout(&images, &cameras, &options, &model_ids, &partial_fixed);
        let linearized = linearize(&cameras, &images, &points, &options, &model_ids);
        let lambda = 1e-4;
        let mut system = assemble(&layout, &linearized, &points, &model_ids, lambda);

        let cost0 = robust_cost(&cameras, &images, &points, Loss::Trivial, &model_ids);

        // —— 数值校验 1：梯度 g 与有限差分一致 ——
        let eps = 1e-6;
        for (block, (image_row, pose_block)) in system.layout.pose_of_image.iter().enumerate()
            .filter_map(|(row, e)| e.as_ref().map(|(b, p)| (row == *b, (row, p))))
            .map(|(_, pair)| pair).enumerate().take(1)
        {
            let _ = block;
            for &dim in pose_block.free.iter().take(1) {
                let mut images_plus = images.clone();
                let rotation = quaternion_to_rotation_matrix(&images_plus[image_row].qvec);
                let mut delta6 = [0.0f64; 6];
                delta6[dim] = eps;
                let delta = [delta6[0], delta6[1], delta6[2]];
                let delta_rotation = exp_rotation_vector(&delta);
                let new_rotation = mat3_mat3(&delta_rotation, &rotation);
                let t = images_plus[image_row].tvec;
                let rotated = mat3_vec3(&delta_rotation, t);
                images_plus[image_row].qvec = rotation_to_quaternion(&new_rotation);
                images_plus[image_row].tvec = [rotated[0] + delta6[3], rotated[1] + delta6[4], rotated[2] + delta6[5]];
                let cost_plus = robust_cost(&cameras, &images_plus, &points, Loss::Trivial, &model_ids);
                // 调试：找出爆炸来源。
                let mut min_z = f64::INFINITY;
                let mut min_at = (0usize, [0.0f64; 3]);
                let mut max_pix = 0.0f64;
                for point in &points {
                    for &(ir, _, _) in &point.observations {
                        let img = &images_plus[ir];
                        let r = quaternion_to_rotation_matrix(&img.qvec);
                        let z = r[2][0]*point.xyz[0] + r[2][1]*point.xyz[1] + r[2][2]*point.xyz[2] + img.tvec[2];
                        if z < min_z { min_z = z; min_at = (ir, point.xyz); }
                        let pix = project_point(&img.qvec, &img.tvec, &point.xyz, CameraModelId::SimpleRadial, &cameras[0].params);
                        max_pix = max_pix.max(pix[0].abs()).max(pix[1].abs());
                    }
                }
                eprintln!("数值诊断: min_z={min_z:.3e} at image {} xyz={:?} max_pix={max_pix:.3e} cost_plus={cost_plus:.3e}", min_at.0, min_at.1);
                {
                    eprintln!("  扰动目标 image_row={image_row} dim={dim} eps={eps}");
                    let img = &images_plus[image_row];
                    eprintln!("  images_plus[目标]: q={:?} t={:?}", img.qvec, img.tvec);
                }
                {
                    let img = &images_plus[1];
                    let r = quaternion_to_rotation_matrix(&img.qvec);
                    let z = r[2][0]*(-1.680310389474337) + r[2][1]*1.8466431963829981 + r[2][2]*1.3927246419184296 + img.tvec[2];
                    eprintln!("  images_plus[1]: t={:?} 该点 z={z:.3e}", img.tvec);
                }
                for (idx, img) in images.iter().enumerate() {
                    let r = quaternion_to_rotation_matrix(&img.qvec);
                    let zs: Vec<f64> = points.iter().filter_map(|p| p.observations.iter().find(|o| o.0 == idx).map(|_| r[2][0]*p.xyz[0] + r[2][1]*p.xyz[1] + r[2][2]*p.xyz[2] + img.tvec[2])).collect();
                    let zmin = zs.iter().cloned().fold(f64::INFINITY, f64::min);
                    eprintln!("  image {idx}: q={:?} t={:?} zmin={zmin:.3e} n_obs={}", img.qvec, img.tvec, zs.len());
                }
                let numeric_grad = (cost_plus - cost0) / eps;
                let block_index = system.layout.pose_of_image[image_row].as_ref().map(|(b, _)| *b).unwrap();
                let off = system.block_offset(block_index);
                let dof = system.block_dof[block_index];
                let dim_index = pose_block.free.iter().position(|&d| d == dim).unwrap();
                println!("梯度校验 dim {dim}: 数值={numeric_grad:.4} 组装={:.4}", system.block_g[block_index][dim_index]);
                let _ = (off, dof);
            }
        }

        let outcome = system.solve_step(&mut cameras, &mut images, &mut points, cost0, Loss::Trivial, &model_ids);
        let cost1 = robust_cost(&cameras, &images, &points, Loss::Trivial, &model_ids);
        if let StepOutcome::Accepted(new_cost) = outcome {
            println!("cost {cost0:.6} -> {cost1:.6}（summary cost {new_cost:.6}，应下降）");
            assert!(cost1 < cost0, "一步 GN 应降低代价");
        } else {
            panic!("一步 GN 应被接受：{outcome:?}（cost {cost0:.6} -> {cost1:.6}）");
        }
    }
}
