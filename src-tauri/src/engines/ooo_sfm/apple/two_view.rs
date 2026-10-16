//! 双视图几何验证（M4）：LO-RANSAC + 7 点基础矩阵（Sampson 精化）+ DLT 单应 + 水印检测。
//!
//! 逐条对齐 COLMAP 4.2.1 `src/colmap/estimators/two_view_geometry.cc` 的
//! **未标定路径**（`EstimateUncalibratedTwoViewGeometry`）与
//! `src/colmap/optim/{ransac,loransac,support_measurement,random_sampler}.h`：
//!
//! - 基础矩阵：`LORANSAC<7 点, Sampson 精化, MSAC>`，`max_error = 4.0`（平方残差阈值 16）、
//!   `confidence = 0.999`、`min_num_trials = 100`、`max_num_trials = 10000`、
//!   `min_inlier_ratio = 0.25`（构造时按 `ComputeNumTrials(0.25·1e5, 1e5)` 收缩上限）；
//! - 单应：同样的 LO-RANSAC（DLT 既是假设估计器也是局部估计器），但把
//!   `min_inlier_ratio` 提高到 `max(0.25, 0.8 · F 内点数 / 匹配数)`（预算搜索）；
//! - 停止准则：`ComputeNumTrials` 用无放回抽样的精确概率（Schönberger–Larsson–Pollefeys 2025），
//!   每次改进最优模型后收缩迭代上限；
//! - 支持度：MSAC（`MEstimatorSupportMeasurer`）——得分 = Σ min(残差, 阈值)，内点 = 残差 ≤ 阈值；
//! - 局部优化：只在内点数 > 7 且 ≥ 8 时进入，最多递归 10 轮（每轮用当前内点集精化）；
//! - 判定：两个报告都\"不成功或不足 `min_num_inliers`\"→ `DEGENERATE`；
//!   否则 `H 内点数 / F 内点数 > 0.8` → `PLANAR_OR_PANORAMIC`（内点数多的那个模型出掩码），
//!   否则 `UNCALIBRATED`（F 掩码）；内点匹配保持输入匹配的顺序；
//! - 水印检测：内点中\"同时落在两张图边界带之内\"（`border = 0.1 · 对角线`）的比例 ≥ 0.7 时，
//!   再做平移模型 RANSAC（`max_error = 4.0`、`min_inlier_ratio = 0.7`），通过则 `WATERMARK`；
//! - `min_inlier_ratio = 0`（项目不传覆盖项）→ 跳过内点比例阈值检查；
//! - `compute_relative_pose = false` → 不估计相对位姿（`qvec`/`tvec` 留 NULL）。
//!
//! 刻意差异（见方案文档 M4 决策）：
//!
//! - **所有图像对都走未标定路径**：4.2.1 在\"同一相机、无焦距先验、针孔模型\"时会走共享焦距
//!   本质矩阵求解器（`EstimateSharedFocalTwoViewGeometry`，需要 poselib 的 6 点求解器）；
//!   自研实现统一用 F/H（等价于 COLMAP 3.x 对未知内参图像对的行为）。落库的 `config`
//!   仍是 `UNCALIBRATED`/`PLANAR_OR_PANORAMIC`；`F` 正是 view graph 标定需要的输入
//!   （4.2.1 的注释点名了这一点）；`camera1`/`camera2` 留 NULL，焦距估计交给自研 mapper；
//! - **F/H 只在估计成功时落库**：4.2.1 在 F 失败但 H 成功时会写入默认构造的 F（未初始化内存），
//!   这里不复刻该缺陷；
//! - **Sampson 精化用等价参数化**：4.2.1 用 Bartoli–Sturm 的 7 参数（两个 SO(3) + 奇异值比）
//!   做无约束最小二乘；这里对 9 个元素做 Levenberg–Marquardt（每步尺度归一 + 秩 2 投影），
//!   目标函数同为平方 Sampson 误差 → 同一极小点；
//! - **随机采样确定性**：`Rng` 由调用方按图像对派生种子，同一输入每次运行结果一致
//!   （4.2.1 `random_seed = -1` 时每次运行不同；RANSAC 本就不可能逐位复刻随机序列）。

use nalgebra::{DMatrix, Matrix3, SMatrix, SVector, Vector2, Vector3};

/// 精化用的定长类型（9 个 F 元素）。
type SMat9 = SMatrix<f64, 9, 9>;
type SVector9 = SVector<f64, 9>;

/// 动态矩阵分解的迭代上限（不透传 nalgebra 的 `max_niter = 0`「不限迭代」）。
/// 行为良好的输入迭代数是常数级；触发上限只会让个别 RANSAC 样本被跳过。
const SVD_MAX_NITER: usize = 200 * 9;
const EIG_MAX_NITER: usize = 100 * 9;

/// `TwoViewGeometry::ConfigurationType`（只列本模块会产出的值）。
pub const DEGENERATE: i32 = 1;
pub const UNCALIBRATED: i32 = 3;
pub const PLANAR_OR_PANORAMIC: i32 = 6;
pub const WATERMARK: i32 = 7;

/// COLMAP `RANSACOptions`（本项目固定用 `TwoViewGeometryOptions` 的默认值）。
#[derive(Clone, Copy, Debug)]
pub struct RansacOptions {
    /// 内点判定的最大误差（内部平方后当残差阈值用）。
    pub max_error: f64,
    /// 先验最小内点比，用于收缩最大迭代次数。
    pub min_inlier_ratio: f64,
    pub confidence: f64,
    pub dyn_num_trials_multiplier: f64,
    pub min_num_trials: usize,
    pub max_num_trials: usize,
}

impl Default for RansacOptions {
    fn default() -> Self {
        Self {
            max_error: 4.0,
            min_inlier_ratio: 0.25,
            confidence: 0.999,
            dyn_num_trials_multiplier: 3.0,
            min_num_trials: 100,
            max_num_trials: 10_000,
        }
    }
}

/// COLMAP `TwoViewGeometryOptions`（只保留本项目可达的字段，取默认值）。
#[derive(Clone, Copy, Debug)]
pub struct VerifyOptions {
    pub min_num_inliers: usize,
    /// 项目不传覆盖项：0（跳过内点比例阈值检查）。
    pub min_inlier_ratio: f64,
    pub max_h_inlier_ratio: f64,
    pub watermark_min_inlier_ratio: f64,
    pub watermark_border_size: f64,
    pub detect_watermark: bool,
    pub watermark_detection_max_error: f64,
    /// `use_sampson_refinement`：默认 true（局部估计器 = Sampson 精化；false 时 = 8 点法）。
    pub use_sampson_refinement: bool,
    pub ransac: RansacOptions,
}

impl Default for VerifyOptions {
    fn default() -> Self {
        Self {
            min_num_inliers: 15,
            min_inlier_ratio: 0.0,
            max_h_inlier_ratio: 0.8,
            watermark_min_inlier_ratio: 0.7,
            watermark_border_size: 0.1,
            detect_watermark: true,
            watermark_detection_max_error: 4.0,
            use_sampson_refinement: true,
            ransac: RansacOptions::default(),
        }
    }
}

/// 验证结果（对应 COLMAP `TwoViewGeometry` 里要落库的部分）。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Verified {
    pub config: i32,
    pub f: Option<Matrix3<f64>>,
    pub h: Option<Matrix3<f64>>,
    pub inlier_matches: Vec<[u32; 2]>,
}

/// 把 3×3 矩阵转成数据库用的行主序数组。
pub fn to_row_major(matrix: &Matrix3<f64>) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = matrix[(row, col)];
        }
    }
    out
}

/// 确定性 PRNG（xorshift64*）。只保证均匀与可复现，不复刻 COLMAP 的 mt19937 序列。
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed | 0x9e37_79b9_7f4a_7c15,
        }
    }

    /// 派生一个子流（同一图像对里 F/H/水印各自用一个流，互不干扰）。
    pub fn derive(&self, salt: u64) -> Self {
        Self::new(self.state.rotate_left(17) ^ salt.wrapping_mul(0xbf58_476d_1ce4_e5b9))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// `[0, 1)` 的均匀实数。
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
    }

    /// `[min, max]` 的均匀整数（对应 `RandomUniformInteger`）。
    fn uniform_index(&mut self, min: usize, max: usize) -> usize {
        debug_assert!(min <= max);
        let span = max - min + 1;
        let value = (self.next_f64() * span as f64) as usize;
        min + value.min(span - 1)
    }
}

/// `RandomSampler`：部分 Fisher–Yates，每次取前 `k` 个作为最小样本。
struct Sampler {
    indices: Vec<usize>,
}

impl Sampler {
    fn new(num_samples: usize) -> Self {
        Self {
            indices: (0..num_samples).collect(),
        }
    }

    fn sample(&mut self, k: usize, rng: &mut Rng, out: &mut Vec<usize>) {
        let last = self.indices.len() - 1;
        for i in 0..k {
            let j = rng.uniform_index(i, last);
            self.indices.swap(i, j);
        }
        out.clear();
        out.extend_from_slice(&self.indices[..k]);
    }
}

/// `RANSAC::ComputeNumTrials`：无放回抽样的精确概率（Schönberger–Larsson–Pollefeys 2025）。
fn compute_num_trials(
    min_samples: usize,
    num_inliers: usize,
    num_samples: usize,
    confidence: f64,
    multiplier: f64,
) -> usize {
    let prob_failure = 1.0 - confidence;
    if prob_failure <= 0.0 {
        return usize::MAX;
    }

    let mut prob_inlier = 1.0_f64;
    for i in 0..min_samples {
        prob_inlier *= (num_inliers as f64 - i as f64) / (num_samples as f64 - i as f64);
    }

    let prob_outlier = 1.0 - prob_inlier;
    if prob_outlier <= 0.0 {
        return 1;
    }
    if prob_outlier == 1.0 {
        return usize::MAX;
    }

    let trials = (prob_failure.ln() / prob_outlier.ln() * multiplier).ceil();
    if !trials.is_finite() || trials <= 0.0 {
        return 0;
    }
    trials as usize
}

/// 模型的鲁棒估计器（最小样本数 + 从最小样本估计 + 平方残差）。
trait ModelEstimator {
    const MIN_SAMPLES: usize;

    fn estimate(
        &self,
        points1: &[Vector2<f64>],
        points2: &[Vector2<f64>],
    ) -> Vec<Matrix3<f64>>;

    fn residuals(
        &self,
        points1: &[Vector2<f64>],
        points2: &[Vector2<f64>],
        model: &Matrix3<f64>,
        out: &mut Vec<f64>,
    );
}

/// LO-RANSAC 的局部估计器（在全部内点上改进 / 重估模型）。
trait LocalModelEstimator: ModelEstimator {
    fn refine(
        &self,
        points1: &[Vector2<f64>],
        points2: &[Vector2<f64>],
        model: &mut Matrix3<f64>,
    ) -> bool;
}

/// MSAC 支持度（`MEstimatorSupportMeasurer`）。
#[derive(Clone, Copy, Debug)]
struct MsacSupport {
    num_inliers: usize,
    score: f64,
}

impl MsacSupport {
    fn evaluate(residuals: &[f64], max_residual: f64) -> Self {
        let mut support = Self {
            num_inliers: 0,
            score: 0.0,
        };
        for &residual in residuals {
            if residual <= max_residual {
                support.num_inliers += 1;
                support.score += residual;
            } else {
                support.score += max_residual;
            }
        }
        support
    }
}

fn is_left_better(left: &MsacSupport, right: &MsacSupport) -> bool {
    left.score < right.score
}

/// 一次 RANSAC/LO-RANSAC 的结果（对应 `LORANSAC::Report`）。
#[derive(Clone, Debug)]
struct Report {
    success: bool,
    num_inliers: usize,
    score: f64,
    inlier_mask: Vec<bool>,
    model: Option<Matrix3<f64>>,
}

impl Report {
    fn failed() -> Self {
        Self {
            success: false,
            num_inliers: 0,
            score: f64::MAX,
            inlier_mask: Vec::new(),
            model: None,
        }
    }
}

/// `LORANSAC::Estimate`：RANSAC + 递归局部优化（最多 10 轮扩展）。
fn loransac<E: ModelEstimator, L: LocalModelEstimator>(
    estimator: &E,
    local_estimator: &L,
    points1: &[Vector2<f64>],
    points2: &[Vector2<f64>],
    options: &RansacOptions,
    rng: &mut Rng,
) -> Report {
    let num_samples = points1.len();
    debug_assert_eq!(num_samples, points2.len());

    let mut report = Report::failed();
    if num_samples < E::MIN_SAMPLES {
        return report;
    }

    let max_residual = options.max_error * options.max_error;

    // 构造函数：按先验内点比收缩迭代上限。
    const PRETEND_NUM_SAMPLES: usize = 100_000;
    let dyn_max_num_trials = compute_num_trials(
        E::MIN_SAMPLES,
        (options.min_inlier_ratio * PRETEND_NUM_SAMPLES as f64) as usize,
        PRETEND_NUM_SAMPLES,
        options.confidence,
        options.dyn_num_trials_multiplier,
    );
    let max_num_trials = options.max_num_trials.min(dyn_max_num_trials);

    let mut sampler = Sampler::new(num_samples);
    let mut best = MsacSupport {
        num_inliers: 0,
        score: f64::MAX,
    };
    let mut best_model: Option<Matrix3<f64>> = None;
    let mut dyn_limit = max_num_trials;

    let mut sampled = Vec::with_capacity(E::MIN_SAMPLES);
    let mut sample1 = Vec::with_capacity(E::MIN_SAMPLES);
    let mut sample2 = Vec::with_capacity(E::MIN_SAMPLES);
    let mut residuals = Vec::with_capacity(num_samples);
    let mut trial = 0usize;
    let mut abort = false;

    while trial < max_num_trials && !abort {
        sampler.sample(E::MIN_SAMPLES, rng, &mut sampled);
        sample1.clear();
        sample2.clear();
        for &index in &sampled {
            sample1.push(points1[index]);
            sample2.push(points2[index]);
        }

        for model in estimator.estimate(&sample1, &sample2) {
            estimator.residuals(points1, points2, &model, &mut residuals);
            let support = MsacSupport::evaluate(&residuals, max_residual);

            if is_left_better(&support, &best) {
                // 先在当前模型上做局部优化（4.2.1 在临界区外做同一件事）。
                let mut local_support = support;
                let mut local_model = model;

                if support.num_inliers > E::MIN_SAMPLES
                    && support.num_inliers >= L::MIN_SAMPLES
                {
                    const MAX_LOCAL_TRIALS: usize = 10;
                    for _ in 0..MAX_LOCAL_TRIALS {
                        let mut inlier1 = Vec::with_capacity(num_samples);
                        let mut inlier2 = Vec::with_capacity(num_samples);
                        for (index, &residual) in residuals.iter().enumerate() {
                            if residual <= max_residual {
                                inlier1.push(points1[index]);
                                inlier2.push(points2[index]);
                            }
                        }
                        if inlier1.len() < L::MIN_SAMPLES {
                            break;
                        }

                        let mut refined = local_model;
                        if !local_estimator.refine(&inlier1, &inlier2, &mut refined) {
                            break;
                        }
                        local_estimator.residuals(points1, points2, &refined, &mut residuals);

                        let refined_support = MsacSupport::evaluate(&residuals, max_residual);
                        if is_left_better(&refined_support, &local_support) {
                            local_support = refined_support;
                            local_model = refined;
                        } else {
                            break;
                        }
                        // 下一轮用精化后的内点集重新提取（`residuals` 已是精化模型的残差）。
                    }
                }

                if is_left_better(&local_support, &best) {
                    best = local_support;
                    best_model = Some(local_model);
                    dyn_limit = compute_num_trials(
                        E::MIN_SAMPLES,
                        best.num_inliers,
                        num_samples,
                        options.confidence,
                        options.dyn_num_trials_multiplier,
                    );
                }
            }

            if trial >= dyn_limit && trial >= options.min_num_trials {
                abort = true;
                break;
            }
        }

        trial += 1;
    }

    let Some(model) = best_model else {
        return report;
    };

    report.score = best.score;
    report.model = Some(model);
    if best.num_inliers < E::MIN_SAMPLES {
        return report;
    }
    report.success = true;

    // 最优模型的残差 → 内点掩码（残差函数在假设/局部估计器间一致）。
    estimator.residuals(points1, points2, &model, &mut residuals);
    report.inlier_mask = residuals.iter().map(|&r| r <= max_residual).collect();
    report.num_inliers = report.inlier_mask.iter().filter(|&&v| v).count();
    report
}

// ---------------------------------------------------------------- 点工具

/// COLMAP `CenterAndNormalizeImagePoints`（返回归一化点与 `normed_from_orig`）。
fn center_and_normalize(points: &[Vector2<f64>]) -> (Vec<Vector2<f64>>, Matrix3<f64>) {
    debug_assert!(!points.is_empty());
    let n = points.len() as f64;
    let mut centroid = Vector2::zeros();
    for point in points {
        centroid += point;
    }
    centroid /= n;

    let mut rms_mean_dist = 0.0;
    for point in points {
        rms_mean_dist += (point - centroid).norm_squared();
    }
    let rms_mean_dist = (rms_mean_dist / n).sqrt();

    let scale = std::f64::consts::SQRT_2 / rms_mean_dist;
    let normed_from_orig = Matrix3::new(
        scale,
        0.0,
        -scale * centroid.x,
        0.0,
        scale,
        -scale * centroid.y,
        0.0,
        0.0,
        1.0,
    );
    let normed = points
        .iter()
        .map(|point| {
            // 注意：不能写 `point.to_homogeneous()` —— nalgebra 的 `Vector::to_homogeneous`
            // 追加的是 0（方向向量语义），点的齐次坐标必须显式追加 1。
            let homogeneous = normed_from_orig * point.push(1.0);
            Vector2::new(homogeneous.x / homogeneous.z, homogeneous.y / homogeneous.z)
        })
        .collect();
    (normed, normed_from_orig)
}

/// COLMAP `ComputeSquaredSampsonError`（单点）。
fn squared_sampson_error(x1: &Vector3<f64>, x2: &Vector3<f64>, f: &Matrix3<f64>) -> f64 {
    let epipolar_line1 = f * x1;
    let num = x2.dot(&epipolar_line1);
    let denom = x2.dot(&f.column(0)) * x2.dot(&f.column(0))
        + x2.dot(&f.column(1)) * x2.dot(&f.column(1))
        + epipolar_line1.x * epipolar_line1.x
        + epipolar_line1.y * epipolar_line1.y;
    if denom == 0.0 {
        return f64::MAX;
    }
    num * num / denom
}

fn sampson_residuals(
    points1: &[Vector2<f64>],
    points2: &[Vector2<f64>],
    model: &Matrix3<f64>,
    out: &mut Vec<f64>,
) {
    out.clear();
    out.extend(points1.iter().zip(points2).map(|(p1, p2)| {
        squared_sampson_error(&p1.push(1.0), &p2.push(1.0), model)
    }));
}

/// 9 系数行（对方程 `x2ᵀ F x1 = 0`，F 按列主序展平）：`[x1·x2, x1·y2, x1, y1·x2, y1·y2, y1, x2, y2, 1]`。
fn fundamental_row(x1: f64, y1: f64, x2: f64, y2: f64) -> [f64; 9] {
    [
        x1 * x2,
        x1 * y2,
        x1,
        y1 * x2,
        y1 * y2,
        y1,
        x2,
        y2,
        1.0,
    ]
}

// ---------------------------------------------------------------- 基本矩阵

/// `A`（k×9，k ≤ 9）零空间的 `dim` 维正交基。
///
/// nalgebra 的 SVD 是薄的（`v_t` 只有 `min(k, 9)` 行），宽矩阵拿不到最后几行右奇异向量，
/// 所以先取行空间的正交基 `B`（= `v_t` 的行），再对补投影 `I − BᵀB`（秩 = `dim`，
/// 特征值平坦：1 / 0，良态）求绝对值最大的 `dim` 个特征向量。
///
/// 与 COLMAP 取满分解末几列等价：零空间的任意正交基都可以参数化同一组解
/// （三次方程只是在零空间里换了个基，解出的 F 集合相同）。
///
/// `SVD::try_new` / `SymmetricEigen::try_new` 的迭代上限是必要的：nalgebra 的
/// `*::new` 传的 `max_niter = 0` 表示**不限迭代**，病态输入下测试会挂在迭代里。
fn null_space9(a: &DMatrix<f64>, dim: usize) -> Option<Vec<SVector<f64, 9>>> {
    debug_assert!((1..=2).contains(&dim));
    let svd = nalgebra::linalg::SVD::try_new(
        a.clone(),
        false,
        true,
        5.0 * f64::EPSILON,
        SVD_MAX_NITER,
    )?;
    let v_t = svd.v_t?; // k×9，行正交
    let projector = v_t.transpose() * &v_t; // 9×9，秩 k 的（行空间）投影
    if projector.nrows() != 9 || projector.ncols() != 9 {
        return None;
    }
    let mut complement = SMat9::identity();
    for row in 0..9 {
        for col in 0..9 {
            complement[(row, col)] -= projector[(row, col)];
        }
    }

    let eigen = nalgebra::linalg::SymmetricEigen::try_new(complement, f64::EPSILON, EIG_MAX_NITER)?;
    let mut order: Vec<usize> = (0..9).collect();
    order.sort_by(|&left, &right| {
        eigen.eigenvalues[right]
            .abs()
            .partial_cmp(&eigen.eigenvalues[left].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // 主特征值必须确实接近 1（即 A 确有 dim 维零空间）。
    if eigen.eigenvalues[order[0]].abs() < 0.5 {
        return None;
    }
    let mut basis = Vec::with_capacity(dim);
    for &index in &order[..dim] {
        basis.push(SVector::<f64, 9>::from_column_slice(
            eigen.eigenvectors.column(index).as_slice(),
        ));
    }
    Some(basis)
}

/// `FundamentalMatrixSevenPointEstimator`：7 点问题（最多 3 个解）。
struct FundamentalSevenPoint;

impl FundamentalSevenPoint {
    fn estimate_models(points1: &[Vector2<f64>], points2: &[Vector2<f64>]) -> Vec<Matrix3<f64>> {
        debug_assert_eq!(points1.len(), 7);
        debug_assert_eq!(points2.len(), 7);

        let mut a = DMatrix::<f64>::zeros(7, 9);
        for i in 0..7 {
            let row = fundamental_row(points1[i].x, points1[i].y, points2[i].x, points2[i].y);
            for (j, value) in row.into_iter().enumerate() {
                a[(i, j)] = value;
            }
        }

        // 7 个方程 9 个未知数 → 2 维零空间。
        let Some(nullspace) = null_space9(&a, 2) else {
            return Vec::new();
        };
        let mut f1 = nullspace[0];
        let f2 = nullspace[1];
        f1 -= &f2;

        // det(λ·f1 + f2) 展开成 λ 的三次多项式（系数逐项对齐 COLMAP 原式；
        // 向量下标 = 矩阵按列主序展平，3..8 对应第 1、2 列）。
        let t0 = f1[4] * f1[8] - f1[5] * f1[7];
        let t1 = f1[3] * f1[8] - f1[5] * f1[6];
        let t2 = f1[3] * f1[7] - f1[4] * f1[6];
        let t3 = f2[4] * f2[8] - f2[5] * f2[7];
        let t4 = f2[3] * f2[8] - f2[5] * f2[6];
        let t5 = f2[3] * f2[7] - f2[4] * f2[6];

        let mut coeffs = [0.0f64; 4];
        coeffs[0] = f1[0] * t0 - f1[1] * t1 + f1[2] * t2;
        if coeffs[0].abs() < 1e-16 {
            return Vec::new();
        }
        coeffs[1] = f2[0] * t0 - f2[1] * t1 + f2[2] * t2
            - f2[3] * (f1[1] * f1[8] - f1[2] * f1[7])
            + f2[4] * (f1[0] * f1[8] - f1[2] * f1[6])
            - f2[5] * (f1[0] * f1[7] - f1[1] * f1[6])
            + f2[6] * (f1[1] * f1[5] - f1[2] * f1[4])
            - f2[7] * (f1[0] * f1[5] - f1[2] * f1[3])
            + f2[8] * (f1[0] * f1[4] - f1[1] * f1[3]);
        coeffs[2] = f1[0] * t3 - f1[1] * t4 + f1[2] * t5
            - f1[3] * (f2[1] * f2[8] - f2[2] * f2[7])
            + f1[4] * (f2[0] * f2[8] - f2[2] * f2[6])
            - f1[5] * (f2[0] * f2[7] - f2[1] * f2[6])
            + f1[6] * (f2[1] * f2[5] - f2[2] * f2[4])
            - f1[7] * (f2[0] * f2[5] - f2[2] * f2[3])
            + f1[8] * (f2[0] * f2[4] - f2[1] * f2[3]);
        coeffs[3] = f2[0] * t3 - f2[1] * t4 + f2[2] * t5;
        let c0 = coeffs[0];
        for value in &mut coeffs[1..] {
            *value /= c0;
        }

        let (roots, num_roots) = find_cubic_polynomial_roots(coeffs[1], coeffs[2], coeffs[3]);
        let mut models = Vec::with_capacity(num_roots);
        for &root in &roots[..num_roots] {
            let mut f = &f1 * root + &f2;
            let norm = f.norm();
            if !(norm > 0.0) || !norm.is_finite() {
                continue;
            }
            f /= norm;
            models.push(Matrix3::from_column_slice(f.as_slice()));
        }
        models
    }
}

impl ModelEstimator for FundamentalSevenPoint {
    const MIN_SAMPLES: usize = 7;

    fn estimate(
        &self,
        points1: &[Vector2<f64>],
        points2: &[Vector2<f64>],
    ) -> Vec<Matrix3<f64>> {
        Self::estimate_models(points1, points2)
    }

    fn residuals(
        &self,
        points1: &[Vector2<f64>],
        points2: &[Vector2<f64>],
        model: &Matrix3<f64>,
        out: &mut Vec<f64>,
    ) {
        sampson_residuals(points1, points2, model, out)
    }
}

/// `FindCubicPolynomialRoots`（单步 Newton 精化，逐行对齐）。
fn find_cubic_polynomial_roots(c2: f64, c1: f64, c0: f64) -> ([f64; 3], usize) {
    const TWO_PI_OVER_3: f64 = 2.094_395_102_393_195_3;
    const FOUR_PI_OVER_3: f64 = 4.188_790_204_786_390_5;

    let c2_over_3 = c2 / 3.0;
    let a = c1 - c2 * c2_over_3;
    let mut b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
    let mut c = b * b / 4.0 + a * a * a / 27.0;

    let mut real = [0.0f64; 3];
    let num_roots;
    if c > 0.0 {
        c = c.sqrt();
        b *= -0.5;
        real[0] = (b + c).cbrt() + (b - c).cbrt() - c2_over_3;
        num_roots = 1;
    } else {
        c = 3.0 * b / (2.0 * a) * (-3.0 / a).sqrt();
        let d = 2.0 * (-a / 3.0).sqrt();
        let acos_over_3 = c.acos() / 3.0;
        real[0] = d * acos_over_3.cos() - c2_over_3;
        real[1] = d * (acos_over_3 - TWO_PI_OVER_3).cos() - c2_over_3;
        real[2] = d * (acos_over_3 - FOUR_PI_OVER_3).cos() - c2_over_3;
        num_roots = 3;
    }

    for value in real.iter_mut().take(num_roots) {
        let x = *value;
        let x2 = x * x;
        let x3 = x * x2;
        let dx = -(x3 + c2 * x2 + c1 * x + c0) / (3.0 * x2 + 2.0 * c2 * x + c1);
        *value += dx;
    }
    (real, num_roots)
}

/// `FundamentalMatrixEightPointEstimator`：归一化 8 点法（含秩 2 投影）。
struct FundamentalEightPoint;

impl FundamentalEightPoint {
    fn estimate_models(points1: &[Vector2<f64>], points2: &[Vector2<f64>]) -> Vec<Matrix3<f64>> {
        if points1.len() < 8 || points1.len() != points2.len() {
            return Vec::new();
        }
        let (normed1, normed_from_orig1) = center_and_normalize(points1);
        let (normed2, normed_from_orig2) = center_and_normalize(points2);

        let mut a = DMatrix::<f64>::zeros(points1.len(), 9);
        for i in 0..points1.len() {
            let row = fundamental_row(normed1[i].x, normed1[i].y, normed2[i].x, normed2[i].y);
            for (j, value) in row.into_iter().enumerate() {
                a[(i, j)] = value;
            }
        }

        // 零空间（8×9 的薄 SVD 拿不到最后一行右奇异向量，8 点样本时走零空间助手）。
        let model_vector = if points1.len() == 8 {
            let Some(nullspace) = null_space9(&a, 1) else {
                return Vec::new();
            };
            nullspace[0]
        } else {
            let svd = nalgebra::linalg::SVD::try_new(
                a,
                false,
                true,
                5.0 * f64::EPSILON,
                SVD_MAX_NITER,
            );
            let Some(svd) = svd else {
                return Vec::new();
            };
            let Some(v_t) = svd.v_t else {
                return Vec::new();
            };
            SVector::<f64, 9>::from_column_slice(v_t.row(8).transpose().as_slice())
        };
        let mut model = Matrix3::from_column_slice(model_vector.as_slice());

        // 秩 2 投影：最小奇异值置零。
        normalize_and_project(&mut model);
        if !model.iter().all(|v| v.is_finite()) {
            return Vec::new();
        }

        let model = normed_from_orig2.transpose() * model * normed_from_orig1;
        vec![model]
    }
}

impl ModelEstimator for FundamentalEightPoint {
    const MIN_SAMPLES: usize = 8;

    fn estimate(
        &self,
        points1: &[Vector2<f64>],
        points2: &[Vector2<f64>],
    ) -> Vec<Matrix3<f64>> {
        Self::estimate_models(points1, points2)
    }

    fn residuals(
        &self,
        points1: &[Vector2<f64>],
        points2: &[Vector2<f64>],
        model: &Matrix3<f64>,
        out: &mut Vec<f64>,
    ) {
        sampson_residuals(points1, points2, model, out)
    }
}

impl LocalModelEstimator for FundamentalEightPoint {
    fn refine(
        &self,
        points1: &[Vector2<f64>],
        points2: &[Vector2<f64>],
        model: &mut Matrix3<f64>,
    ) -> bool {
        let models = Self::estimate_models(points1, points2);
        if let Some(best) = models.first() {
            *model = *best;
            true
        } else {
            false
        }
    }
}

/// 秩 2 投影 + 尺度归一（`||F|| = 1`）。
fn normalize_and_project(model: &mut Matrix3<f64>) {
    let svd = nalgebra::linalg::SVD::new(*model, true, true);
    let (Some(u), Some(v_t)) = (svd.u, svd.v_t) else {
        return;
    };
    let mut singular_values = svd.singular_values;
    singular_values[2] = 0.0;
    let projected = u * Matrix3::from_diagonal(&singular_values) * v_t;
    let norm = projected.norm();
    if norm > 0.0 && norm.is_finite() {
        *model = projected / norm;
    }
}

/// 平方 Sampson 误差（9 系数向量形式，列主序展平）。
fn sampson_cost(points1: &[Vector2<f64>], points2: &[Vector2<f64>], model: &Matrix3<f64>) -> f64 {
    let mut cost = 0.0;
    for (p1, p2) in points1.iter().zip(points2) {
        let residual = squared_sampson_error(&p1.push(1.0), &p2.push(1.0), model);
        if residual.is_finite() {
            cost += residual;
        } else {
            return f64::MAX;
        }
    }
    cost
}

/// `RefineFundamentalMatrixSampson`：平方 Sampson 误差的局部最小二乘精化。
///
/// 4.2.1 用 Bartoli–Sturm 的 7 参数（U、V 两个四元数 + 奇异值比）做无约束最小二乘；
/// 这里在 9 个元素上做 Levenberg–Marquardt，每步做尺度归一 + 秩 2 投影。目标函数相同
/// （牛顿步的雅可比不同但都精确二阶逼近），直达同一极小点。
fn refine_fundamental_matrix_sampson(
    points1: &[Vector2<f64>],
    points2: &[Vector2<f64>],
    f: &mut Matrix3<f64>,
) -> bool {
    if points1.len() < 8 || points1.len() != points2.len() {
        return false;
    }

    // 与 8 点法一致的居中归一化 + 两视图公共尺度（Sampson 误差只对公共尺度不变）。
    let (mut normed1, mut normed_from_orig1) = center_and_normalize(points1);
    let (mut normed2, mut normed_from_orig2) = center_and_normalize(points2);
    let scale1 = normed_from_orig1[(0, 0)];
    let scale2 = normed_from_orig2[(0, 0)];
    let scale = (scale1 * scale2).sqrt();
    let factor1 = scale / scale1;
    let factor2 = scale / scale2;
    for point in &mut normed1 {
        *point *= factor1;
    }
    for point in &mut normed2 {
        *point *= factor2;
    }
    for row in 0..2 {
        normed_from_orig1.row_mut(row).scale_mut(factor1);
        normed_from_orig2.row_mut(row).scale_mut(factor2);
    }

    // 把 F 换到归一化坐标系（8 点法回映射 `T2ᵀ·F_norm·T1` 的逆：
    // `F_norm = (T2ᵀ)⁻¹·F·T1⁻¹`，对应 4.2.1 的
    // `normed_from_orig2.transpose().inverse() * F * normed_from_orig1.inverse()`）。
    let Some(inv1) = normed_from_orig1.try_inverse() else {
        return false;
    };
    let Some(inv2) = normed_from_orig2.try_inverse() else {
        return false;
    };
    let mut model = inv2.transpose() * (*f) * inv1;
    let svd = nalgebra::linalg::SVD::new(model, false, false);
    let singular_values = svd.singular_values;
    const MIN_SINGULAR_VALUE_RATIO: f64 = 1e-12;
    if !(singular_values[0] > 0.0)
        || singular_values[1] <= MIN_SINGULAR_VALUE_RATIO * singular_values[0]
    {
        return false;
    }
    normalize_and_project(&mut model);

    // Gauss–Newton/LM：目标 = Σ 平方 Sampson 误差（纯最小二乘，鲁棒性来自 RANSAC 的内点集）。
    // 雅可比与 9×9 法方程都用栈上定长矩阵（RANSAC 会高频调用本函数）。
    let num_points = normed1.len();
    let mut residual = vec![0.0f64; num_points];
    let mut jacobian = vec![SVector9::zeros(); num_points];
    let mut cost = sampson_cost(&normed1, &normed2, &model);
    if !cost.is_finite() {
        return false;
    }

    let mut lambda = 1e-4_f64;
    for _ in 0..25 {
        let mut jtj = SMat9::zeros();
        let mut jtr = SVector9::zeros();
        for i in 0..num_points {
            let x1h = normed1[i].push(1.0);
            let x2h = normed2[i].push(1.0);
            let fx1 = model * x1h;
            let ftx2 = model.transpose() * x2h;
            let num = x2h.dot(&fx1);
            let denom = fx1.x * fx1.x + fx1.y * fx1.y + ftx2.x * ftx2.x + ftx2.y * ftx2.y;
            if !(denom > 0.0) || !denom.is_finite() {
                return false;
            }
            // ∂num/∂F 与 ∂denom/∂F（3×3，按列主序展平后与 9 维参数一致）。
            let dnum = x2h * x1h.transpose();
            let padded_fx1 = Vector3::new(fx1.x, fx1.y, 0.0);
            let padded_ftx2 = Vector3::new(ftx2.x, ftx2.y, 0.0);
            let ddenom = 2.0 * (padded_fx1 * x1h.transpose() + x2h * padded_ftx2.transpose());
            let dr = (2.0 * num * dnum * denom - num * num * ddenom) / (denom * denom);
            residual[i] = num * num / denom;
            let j = SVector9::from_column_slice(dr.as_slice());
            jacobian[i] = j;
            jtj += j * j.transpose();
            jtr += j * residual[i];
        }

        let mut step_taken = false;
        for _ in 0..10 {
            let mut damped = jtj;
            for d in 0..9 {
                damped[(d, d)] += lambda * jtj[(d, d)].max(1e-12);
            }
            let Some(delta) = damped.lu().solve(&jtr) else {
                lambda *= 10.0;
                continue;
            };
            let mut candidate = model;
            for j in 0..9 {
                candidate[j] -= delta[j];
            }
            normalize_and_project(&mut candidate);
            if !candidate.iter().all(|v| v.is_finite()) {
                lambda *= 10.0;
                continue;
            }
            let candidate_cost = sampson_cost(&normed1, &normed2, &candidate);
            if candidate_cost < cost {
                model = candidate;
                cost = candidate_cost;
                lambda = (lambda * 0.5).max(1e-12);
                step_taken = true;
                break;
            }
            lambda *= 10.0;
        }

        if !step_taken {
            break;
        }
    }

    if !model.iter().all(|v| v.is_finite()) {
        return false;
    }
    *f = normed_from_orig2.transpose() * model * normed_from_orig1;
    true
}

/// 局部估计器：Sampson 精化（`use_sampson_refinement = true`，默认）。
struct FundamentalSampsonLocal;

impl ModelEstimator for FundamentalSampsonLocal {
    const MIN_SAMPLES: usize = 8;

    fn estimate(
        &self,
        points1: &[Vector2<f64>],
        points2: &[Vector2<f64>],
    ) -> Vec<Matrix3<f64>> {
        FundamentalEightPoint::estimate_models(points1, points2)
    }

    fn residuals(
        &self,
        points1: &[Vector2<f64>],
        points2: &[Vector2<f64>],
        model: &Matrix3<f64>,
        out: &mut Vec<f64>,
    ) {
        sampson_residuals(points1, points2, model, out)
    }
}

impl LocalModelEstimator for FundamentalSampsonLocal {
    fn refine(
        &self,
        points1: &[Vector2<f64>],
        points2: &[Vector2<f64>],
        model: &mut Matrix3<f64>,
    ) -> bool {
        refine_fundamental_matrix_sampson(points1, points2, model)
    }
}

// ---------------------------------------------------------------- 单应

/// `HomographyMatrixEstimator`：DLT + 平方变换误差。
///
/// - 4 点：原始坐标下固定 h₈ = 1、解 8×8 线性系统（与 4.2.1 相同）；
/// - 多点：Hartley 归一化后取 `AᵀA` 的最小特征向量（秩检查容差与 4.2.1 的
///   `svd.rank() < 8` 对齐），再反归一化 `H = T2⁻¹ H_norm T1` 并按单位范数存储。
///   矩阵以**行主序**展平成 9 维向量（`from_row_slice`），与 COLMAP 的
///   `Map(h.data()).transpose()` 一致；多点路径的归一化是刻意的数值改进
///   （4.2.1 用原始像素直接 SVD，条件数差且建立在 nalgebra 上会不收敛）。
struct HomographyMatrix;

impl HomographyMatrix {
    fn has_collinear_triplet(points: &[Vector2<f64>]) -> bool {
        let is_collinear = |i: usize, j: usize, k: usize| {
            const MIN_NORMALIZED_AREA_SQUARED: f64 = 1e-24;
            let delta1 = points[j] - points[i];
            let delta2 = points[k] - points[i];
            let scale_squared = delta1.norm_squared() * delta2.norm_squared();
            let area = delta1.x * delta2.y - delta1.y * delta2.x;
            scale_squared == 0.0 || area * area <= MIN_NORMALIZED_AREA_SQUARED * scale_squared
        };
        is_collinear(0, 1, 2)
            || is_collinear(0, 1, 3)
            || is_collinear(0, 2, 3)
            || is_collinear(1, 2, 3)
    }

    /// DLT 约束矩阵：h 按行主序展平，行 `2i` 为
    /// `[x, y, 1, 0, 0, 0, −x'x, −x'y, −x']`，行 `2i+1` 同理。
    fn constraint_matrix(points1: &[Vector2<f64>], points2: &[Vector2<f64>]) -> DMatrix<f64> {
        let mut a = DMatrix::<f64>::zeros(2 * points1.len(), 9);
        for i in 0..points1.len() {
            let s = points1[i];
            let d = points2[i];
            a[(2 * i, 0)] = s.x;
            a[(2 * i, 1)] = s.y;
            a[(2 * i, 2)] = 1.0;
            a[(2 * i, 6)] = -d.x * s.x;
            a[(2 * i, 7)] = -d.x * s.y;
            a[(2 * i, 8)] = -d.x;
            a[(2 * i + 1, 3)] = s.x;
            a[(2 * i + 1, 4)] = s.y;
            a[(2 * i + 1, 5)] = 1.0;
            a[(2 * i + 1, 6)] = -d.y * s.x;
            a[(2 * i + 1, 7)] = -d.y * s.y;
            a[(2 * i + 1, 8)] = -d.y;
        }
        a
    }

    fn estimate_models(points1: &[Vector2<f64>], points2: &[Vector2<f64>]) -> Vec<Matrix3<f64>> {
        let num_points = points1.len();
        if num_points < 4 || num_points != points2.len() {
            return Vec::new();
        }
        if num_points == 4
            && (Self::has_collinear_triplet(points1) || Self::has_collinear_triplet(points2))
        {
            return Vec::new();
        }

        let h = if num_points == 4 {
            // 齐次坐标最后一个系数固定为 1，解 8×8 线性系统。
            let a = Self::constraint_matrix(points1, points2);
            let m = a.view((0, 0), (8, 8));
            let b = -a.view((0, 8), (8, 1)).into_owned();
            let Some(solution) = m.lu().solve(&b) else {
                return Vec::new();
            };
            if !solution.iter().all(|v| v.is_finite()) {
                return Vec::new();
            }
            let mut vector = [0.0_f64; 9];
            for (j, value) in solution.iter().enumerate() {
                vector[j] = *value;
            }
            vector[8] = 1.0;
            Matrix3::from_row_slice(&vector)
        } else {
            // 归一化 DLT：Hartley 归一化后对 AᵀA 求最小特征向量，再反归一化。
            let (normed1, normed_from_orig1) = center_and_normalize(points1);
            let (normed2, normed_from_orig2) = center_and_normalize(points2);
            let a = Self::constraint_matrix(&normed1, &normed2);
            let matrix = a.transpose() * &a;
            let matrix = SMat9::from_column_slice(matrix.as_slice());
            let Some(eigen) =
                nalgebra::linalg::SymmetricEigen::try_new(matrix, f64::EPSILON, EIG_MAX_NITER)
            else {
                return Vec::new();
            };

            // 秩检查（与 COLMAP `svd.rank() < 8` 对齐：容差 = eps·9·最大奇异值）。
            let mut sigmas = [0.0_f64; 9];
            for (index, value) in eigen.eigenvalues.iter().enumerate() {
                sigmas[index] = value.max(0.0).sqrt();
            }
            let max_sv = sigmas.iter().cloned().fold(0.0, f64::max);
            let tolerance = f64::EPSILON * 9.0 * max_sv;
            let rank = sigmas.iter().filter(|&&value| value > tolerance).count();
            if rank < 8 {
                return Vec::new();
            }
            let mut smallest = 0;
            for index in 1..9 {
                if sigmas[index] < sigmas[smallest] {
                    smallest = index;
                }
            }
            let h_norm = Matrix3::from_row_slice(eigen.eigenvectors.column(smallest).as_slice());
            let Some(inverse2) = normed_from_orig2.try_inverse() else {
                return Vec::new();
            };
            let mut h = inverse2 * h_norm * normed_from_orig1;
            // 4.2.1 的零向量是单位范数，这里对齐同一尺度后再做行列式检查。
            let norm = h.norm();
            if !(norm > 0.0) || !h.iter().all(|v| v.is_finite()) {
                return Vec::new();
            }
            h /= norm;
            h
        };

        if !(h.determinant().abs() >= 1e-8) {
            return Vec::new();
        }
        vec![h]
    }
}

impl ModelEstimator for HomographyMatrix {
    const MIN_SAMPLES: usize = 4;

    fn estimate(
        &self,
        points1: &[Vector2<f64>],
        points2: &[Vector2<f64>],
    ) -> Vec<Matrix3<f64>> {
        Self::estimate_models(points1, points2)
    }

    fn residuals(
        &self,
        points1: &[Vector2<f64>],
        points2: &[Vector2<f64>],
        model: &Matrix3<f64>,
        out: &mut Vec<f64>,
    ) {
        out.clear();
        for (source, destination) in points1.iter().zip(points2) {
            let pd = model * source.push(1.0);
            let dx = destination.x - pd.x / pd.z;
            let dy = destination.y - pd.y / pd.z;
            out.push(dx * dx + dy * dy);
        }
    }
}

impl LocalModelEstimator for HomographyMatrix {
    fn refine(
        &self,
        points1: &[Vector2<f64>],
        points2: &[Vector2<f64>],
        model: &mut Matrix3<f64>,
    ) -> bool {
        let models = Self::estimate_models(points1, points2);
        if let Some(best) = models.first() {
            *model = *best;
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------- 平移（水印）

/// `TranslationTransformEstimator<2>`。
struct TranslationTransform;

impl ModelEstimator for TranslationTransform {
    const MIN_SAMPLES: usize = 1;

    fn estimate(
        &self,
        points1: &[Vector2<f64>],
        points2: &[Vector2<f64>],
    ) -> Vec<Matrix3<f64>> {
        if points1.is_empty() || points1.len() != points2.len() {
            return Vec::new();
        }
        let mut mean_src = Vector2::zeros();
        let mut mean_dst = Vector2::zeros();
        for (p1, p2) in points1.iter().zip(points2) {
            mean_src += p1;
            mean_dst += p2;
        }
        mean_src /= points1.len() as f64;
        mean_dst /= points1.len() as f64;
        let translation = mean_dst - mean_src;
        vec![Matrix3::new(
            1.0,
            0.0,
            translation.x,
            0.0,
            1.0,
            translation.y,
            0.0,
            0.0,
            1.0,
        )]
    }

    fn residuals(
        &self,
        points1: &[Vector2<f64>],
        points2: &[Vector2<f64>],
        model: &Matrix3<f64>,
        out: &mut Vec<f64>,
    ) {
        out.clear();
        for (p1, p2) in points1.iter().zip(points2) {
            let dx = p2.x - p1.x - model[(0, 2)];
            let dy = p2.y - p1.y - model[(1, 2)];
            out.push(dx * dx + dy * dy);
        }
    }
}

impl LocalModelEstimator for TranslationTransform {
    fn refine(
        &self,
        points1: &[Vector2<f64>],
        points2: &[Vector2<f64>],
        model: &mut Matrix3<f64>,
    ) -> bool {
        let models = self.estimate(points1, points2);
        if let Some(best) = models.first() {
            *model = *best;
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------- 主入口

/// `EstimateUncalibratedTwoViewGeometry`（不含本项目未启用的两条分支：
/// 内点比例阈值 `min_inlier_ratio = 0`、相对位姿 `compute_relative_pose = false`）。
pub fn verify(
    matches: &[[u32; 2]],
    points1: &[Vector2<f64>],
    points2: &[Vector2<f64>],
    camera1: (u32, u32),
    camera2: (u32, u32),
    options: &VerifyOptions,
    rng: &Rng,
) -> Verified {
    let mut geometry = Verified::default();

    if matches.len() < options.min_num_inliers {
        geometry.config = DEGENERATE;
        return geometry;
    }

    let matched1: Vec<Vector2<f64>> =
        matches.iter().map(|m| points1[m[0] as usize]).collect();
    let matched2: Vec<Vector2<f64>> =
        matches.iter().map(|m| points2[m[1] as usize]).collect();

    let mut ransac_options = options.ransac;
    if options.min_inlier_ratio > 0.0 {
        ransac_options.min_inlier_ratio = options.min_inlier_ratio;
    }

    let f_report = if options.use_sampson_refinement {
        loransac(
            &FundamentalSevenPoint,
            &FundamentalSampsonLocal,
            &matched1,
            &matched2,
            &ransac_options,
            &mut rng.derive(0x46),
        )
    } else {
        loransac(
            &FundamentalSevenPoint,
            &FundamentalEightPoint,
            &matched1,
            &matched2,
            &ransac_options,
            &mut rng.derive(0x46),
        )
    };

    // 单应的搜索预算：内点数达不到 max_H_inlier_ratio × F 内点数的话反正会被丢弃。
    let mut h_ransac_options = ransac_options;
    let h_floor = options.max_h_inlier_ratio * f_report.num_inliers as f64 / matches.len() as f64;
    h_ransac_options.min_inlier_ratio = ransac_options.min_inlier_ratio.max(h_floor);
    let h_report = loransac(
        &HomographyMatrix,
        &HomographyMatrix,
        &matched1,
        &matched2,
        &h_ransac_options,
        &mut rng.derive(0x48),
    );

    let f_ok = f_report.success && f_report.num_inliers >= options.min_num_inliers;
    let h_ok = h_report.success && h_report.num_inliers >= options.min_num_inliers;
    if !f_ok && !h_ok {
        geometry.config = DEGENERATE;
        return geometry;
    }

    // F/H 只在估计成功时写出（4.2.1 会写入默认构造的 F，这里不复刻该缺陷）。
    geometry.f = if f_report.success {
        f_report.model
    } else {
        None
    };
    geometry.h = if h_report.success {
        h_report.model
    } else {
        None
    };

    let h_f_inlier_ratio = if f_report.num_inliers > 0 {
        h_report.num_inliers as f64 / f_report.num_inliers as f64
    } else {
        f64::INFINITY
    };
    let mut num_inliers = f_report.num_inliers;
    let mut best_inlier_mask = &f_report.inlier_mask;
    if h_f_inlier_ratio > options.max_h_inlier_ratio {
        geometry.config = PLANAR_OR_PANORAMIC;
        if h_report.num_inliers >= f_report.num_inliers {
            num_inliers = h_report.num_inliers;
            best_inlier_mask = &h_report.inlier_mask;
        }
    } else {
        geometry.config = UNCALIBRATED;
    }

    geometry.inlier_matches = matches
        .iter()
        .zip(best_inlier_mask)
        .filter_map(|(m, inlier)| inlier.then_some(*m))
        .collect();

    if options.min_inlier_ratio > 0.0 {
        let inlier_ratio = num_inliers as f64 / matches.len() as f64;
        if inlier_ratio < options.min_inlier_ratio {
            geometry.config = DEGENERATE;
            geometry.inlier_matches.clear();
            geometry.f = None;
            geometry.h = None;
            return geometry;
        }
    }

    let watermark = options.detect_watermark
        && detect_watermark_matches(
            camera1,
            &matched1,
            camera2,
            &matched2,
            num_inliers,
            best_inlier_mask,
            options,
            &rng.derive(0x4a),
        );
    if watermark {
        geometry.config = WATERMARK;
    }

    geometry
}

/// `DetectWatermarkMatches`。
#[allow(clippy::too_many_arguments)]
fn detect_watermark_matches(
    camera1: (u32, u32),
    points1: &[Vector2<f64>],
    camera2: (u32, u32),
    points2: &[Vector2<f64>],
    num_inliers: usize,
    inlier_mask: &[bool],
    options: &VerifyOptions,
    rng: &Rng,
) -> bool {
    if num_inliers == 0 {
        return false;
    }
    let diagonal1 = ((camera1.0 as f64).powi(2) + (camera1.1 as f64).powi(2)).sqrt();
    let diagonal2 = ((camera2.0 as f64).powi(2) + (camera2.1 as f64).powi(2)).sqrt();
    let border1 = options.watermark_border_size * diagonal1;
    let border2 = options.watermark_border_size * diagonal2;
    let inside = |point: Vector2<f64>, width: u32, height: u32, border: f64| {
        point.x >= border
            && point.y >= border
            && point.x <= width as f64 - border
            && point.y <= height as f64 - border
    };

    let mut inlier1 = Vec::with_capacity(num_inliers);
    let mut inlier2 = Vec::with_capacity(num_inliers);
    let mut num_matches_in_border = 0usize;
    for (index, &inlier) in inlier_mask.iter().enumerate() {
        if !inlier {
            continue;
        }
        inlier1.push(points1[index]);
        inlier2.push(points2[index]);
        if !inside(points1[index], camera1.0, camera1.1, border1)
            && !inside(points2[index], camera2.0, camera2.1, border2)
        {
            num_matches_in_border += 1;
        }
    }

    let matches_in_border_ratio = num_matches_in_border as f64 / num_inliers as f64;
    if matches_in_border_ratio < options.watermark_min_inlier_ratio {
        return false;
    }

    let mut ransac_options = options.ransac;
    ransac_options.max_error = options.watermark_detection_max_error;
    ransac_options.min_inlier_ratio = options.watermark_min_inlier_ratio;
    let report = loransac(
        &TranslationTransform,
        &TranslationTransform,
        &inlier1,
        &inlier2,
        &ransac_options,
        &mut rng.derive(0x4c),
    );

    let inlier_ratio = report.num_inliers as f64 / num_inliers as f64;
    inlier_ratio >= options.watermark_min_inlier_ratio
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng() -> Rng {
        Rng::new(0x1234_5678_9abc_def0)
    }

    /// 合成场景：固定内参 + 两个相机（绕 y 轴 4° + 平移），随机 3D 点投影。
    fn scene(seed: u64, num_points: usize) -> (Vec<Vector2<f64>>, Vec<Vector2<f64>>) {
        let mut rng = Rng::new(seed);
        let focal = 1000.0;
        let principal = Vector2::new(640.0, 360.0);
        let angle = 4.0_f64.to_radians();
        let (s, c) = angle.sin_cos();
        let rotation = Matrix3::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c);
        let translation = Vector3::new(0.2, 0.0, 0.05);

        let mut points1 = Vec::new();
        let mut points2 = Vec::new();
        while points1.len() < num_points {
            let x = (rng.next_f64() - 0.5) * 4.0;
            let y = (rng.next_f64() - 0.5) * 4.0;
            let z = 3.0 + rng.next_f64() * 5.0;
            let point = Vector3::new(x, y, z);
            let p1 = Vector2::new(principal.x + focal * x / z, principal.y + focal * y / z);
            let projected = rotation * point + translation;
            let p2 = Vector2::new(
                principal.x + focal * projected.x / projected.z,
                principal.y + focal * projected.y / projected.z,
            );
            if p1.x > 0.0 && p1.x < 1280.0 && p1.y > 0.0 && p1.y < 720.0
                && p2.x > 0.0 && p2.x < 1280.0 && p2.y > 0.0 && p2.y < 720.0
            {
                points1.push(p1);
                points2.push(p2);
            }
        }
        (points1, points2)
    }

    #[test]
    fn seven_point_recovers_a_known_fundamental_matrix() {
        let (points1, points2) = scene(7, 60);
        let matches: Vec<[u32; 2]> = (0..points1.len() as u32).map(|i| [i, i]).collect();
        let verified = verify(
            &matches,
            &points1,
            &points2,
            (1280, 720),
            (1280, 720),
            &VerifyOptions::default(),
            &rng(),
        );
        assert_eq!(verified.config, UNCALIBRATED);
        assert!(verified.f.is_some());
        assert!(
            verified.inlier_matches.len() >= points1.len() - 2,
            "内点 {} / {}",
            verified.inlier_matches.len(),
            points1.len()
        );
        // 极线约束确实成立：x2ᵀ F x1 ≈ 0。
        let f = verified.f.unwrap();
        for pair in &verified.inlier_matches {
            let x1 = points1[pair[0] as usize].push(1.0);
            let x2 = points2[pair[1] as usize].push(1.0);
            let residual = x2.dot(&(f * x1)).abs();
            let scale = (f * x1).xy().norm().max(1e-12);
            assert!(residual / scale < 1e-6, "极线残差 {residual}");
        }
    }

    #[test]
    fn degenerate_pairs_are_marked_degenerate() {
        let (points1, points2) = scene(11, 10);
        let matches: Vec<[u32; 2]> = (0..points1.len() as u32).map(|i| [i, i]).collect();
        let verified = verify(
            &matches,
            &points1,
            &points2,
            (1280, 720),
            (1280, 720),
            &VerifyOptions::default(),
            &rng(),
        );
        assert_eq!(verified.config, DEGENERATE);
        assert!(verified.inlier_matches.is_empty());
        assert!(verified.f.is_none() && verified.h.is_none());
    }

    #[test]
    fn random_matches_do_not_verify() {
        let mut rng = rng();
        let points1: Vec<Vector2<f64>> = (0..80)
            .map(|_| Vector2::new(rng.next_f64() * 1280.0, rng.next_f64() * 720.0))
            .collect();
        let points2: Vec<Vector2<f64>> = (0..80)
            .map(|_| Vector2::new(rng.next_f64() * 1280.0, rng.next_f64() * 720.0))
            .collect();
        let matches: Vec<[u32; 2]> = (0..30).map(|i| [i, i]).collect();
        let verified = verify(
            &matches,
            &points1,
            &points2,
            (1280, 720),
            (1280, 720),
            &VerifyOptions::default(),
            &rng,
        );
        assert_eq!(verified.config, DEGENERATE);
    }

    #[test]
    fn planar_scene_is_detected_as_homography() {
        let mut rng = rng();
        // 平面场景：平面 `y = 0.12·x + 0.5` 上的 3D 点投影到两个相机（H 成立）。
        // 注意两点：平面不能过相机光心（否则成像仍是直线，共线退化）；也不能只让
        // 相机绕 y 轴转 —— 平面上若取 y 只随 x 变化、又过光心，同样投影成直线。
        let focal = 1000.0;
        let principal = Vector2::new(640.0, 360.0);
        let angle = 6.0_f64.to_radians();
        let (s, c) = angle.sin_cos();
        let rotation = Matrix3::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c);
        let translation = Vector3::new(0.4, 0.0, 0.0);
        let mut points1 = Vec::new();
        let mut points2 = Vec::new();
        while points1.len() < 60 {
            let x = (rng.next_f64() - 0.5) * 8.0;
            let z = 3.0 + rng.next_f64() * 9.0;
            let point = Vector3::new(x, 0.12 * x + 0.5, z);
            let p1 = Vector2::new(principal.x + focal * x / z, principal.y + focal * point.y / z);
            let projected = rotation * point + translation;
            let p2 = Vector2::new(
                principal.x + focal * projected.x / projected.z,
                principal.y + focal * projected.y / projected.z,
            );
            if p1.x > 0.0 && p1.x < 1280.0 && p1.y > 0.0 && p1.y < 720.0
                && p2.x > 0.0 && p2.x < 1280.0 && p2.y > 0.0 && p2.y < 720.0
            {
                points1.push(p1);
                points2.push(p2);
            }
        }
        let matches: Vec<[u32; 2]> = (0..points1.len() as u32).map(|i| [i, i]).collect();
        let verified = verify(
            &matches,
            &points1,
            &points2,
            (1280, 720),
            (1280, 720),
            &VerifyOptions::default(),
            &rng,
        );
        assert_eq!(verified.config, PLANAR_OR_PANORAMIC);
        assert!(verified.h.is_some());
        assert!(verified.inlier_matches.len() >= 55);
    }

    #[test]
    fn homography_estimator_recovers_known_projection() {
        // 已知 H 上采样：4 点（最小解）与 60 点（归一化 DLT）两条路径都必须精确还原。
        let h = Matrix3::new(1.01, 0.02, -30.0, -0.01, 0.99, 12.0, 0.0002, 0.0001, 1.0);
        let mut rng = rng();
        let mut points1 = Vec::new();
        let mut points2 = Vec::new();
        for _ in 0..60 {
            let p = Vector2::new(
                100.0 + rng.next_f64() * 1000.0,
                80.0 + rng.next_f64() * 550.0,
            );
            let q = h * p.push(1.0);
            points1.push(p);
            points2.push(Vector2::new(q.x / q.z, q.y / q.z));
        }
        for count in [4usize, 60] {
            let models = HomographyMatrix::estimate_models(&points1[..count], &points2[..count]);
            assert_eq!(models.len(), 1, "count = {count} 应有唯一解");
            let estimated = models[0];
            for i in 0..count {
                let q = estimated * points1[i].push(1.0);
                let error = (q.x / q.z - points2[i].x).powi(2) + (q.y / q.z - points2[i].y).powi(2);
                assert!(error < 1e-6, "count = {count}, i = {i}: 误差 {error}");
            }
        }
    }

    #[test]
    fn homography_with_collinear_triplet_is_rejected() {
        let points1 = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(2.0, 0.0),
            Vector2::new(0.0, 1.0),
        ];
        let points2 = points1.clone();
        assert!(HomographyMatrix::estimate_models(&points1, &points2).is_empty());
    }

    #[test]
    fn refinement_improves_sampson_cost() {
        let (mut points1, mut points2) = scene(23, 50);
        // 无噪声数据上 8 点法已几乎精确，加入亚像素噪声让 Gauss–Newton 有改进空间。
        let mut noise = Rng::new(0x5eed_5eed);
        let mut jitter = |points: &mut Vec<Vector2<f64>>| {
            for point in points.iter_mut() {
                point.x += (noise.next_f64() - 0.5) * 1.0;
                point.y += (noise.next_f64() - 0.5) * 1.0;
            }
        };
        jitter(&mut points1);
        jitter(&mut points2);
        let matches: Vec<[u32; 2]> = (0..points1.len() as u32).map(|i| [i, i]).collect();
        let (m1, m2): (Vec<_>, Vec<_>) = matches
            .iter()
            .map(|m| (points1[m[0] as usize], points2[m[1] as usize]))
            .unzip();
        let mut f = FundamentalEightPoint::estimate_models(&m1, &m2)
            .into_iter()
            .next()
            .expect("8 点法应有解");
        let before = sampson_cost(&m1, &m2, &f);
        assert!(refine_fundamental_matrix_sampson(&m1, &m2, &mut f));
        let after = sampson_cost(&m1, &m2, &f);
        assert!(after < before, "精化后代价应下降：{before} → {after}");
    }

    #[test]
    fn seven_point_estimator_returns_valid_models() {
        let (points1, points2) = scene(31, 7);
        let models = FundamentalSevenPoint::estimate_models(&points1[..7], &points2[..7]);
        assert!(!models.is_empty(), "7 点法应给出至少一个解");
        // 至少一个模型能把 7 个样本点全部拟合上（平方 Sampson 误差足够小）。
        let fits = models.iter().any(|model| {
            points1[..7].iter().zip(&points2[..7]).all(|(p1, p2)| {
                squared_sampson_error(&p1.push(1.0), &p2.push(1.0), model) < 1e-6
            })
        });
        assert!(fits, "7 点法给出的解应拟合全部样本");
        for model in &models {
            assert!((model.norm() - 1.0).abs() < 1e-9, "F 应为单位范数");
        }
    }
}
