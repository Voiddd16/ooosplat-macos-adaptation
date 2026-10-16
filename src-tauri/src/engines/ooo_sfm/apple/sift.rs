//! 自研 SIFT 特征提取（Lowe 1999/2004 的标准实现）。
//!
//! 参数与尺度空间语义对齐 COLMAP 4.2.1 的 CPU 路径（`feature/sift.cc`，底层是 VLFeat）：
//! `first_octave = -1`（金字塔前先 2× 上采样）、`num_octaves = 4`、
//! `octave_resolution = 3`、`peak_threshold = 0.02/3`、`edge_threshold = 10`、
//! `max_num_orientations = 2`；描述子按 `L1RootNormalizeFeatureDescriptors`
//! （L1 归一化 → 逐元素开方）后 ×512 量化为 `u8`（`round(512·x)` 截断到 255）。
//!
//! 与 COLMAP 的已知差异（都在算法实现层面，不影响产物格式）：
//! - 插值/边界处理是自研实现（COLMAP 用 VLFeat），特征点位置不会逐位相同；
//! - `max_num_features` 同样保留**粗尺度**特征（COLMAP 注释："keeping larger-scale features"）；
//! - 内存护栏：上采样后的金字塔超过 [`MAX_PYRAMID_PIXELS`] 时跳过首八度上采样（记入 note）。
//!
//! 描述子的 8 个方向 bin 采用 UBC/Lowe 顺序（与 COLMAP 存库前做的
//! `TransformVLFeatToUBCFeatureDescriptors` 等价：组内 bin 镜像），
//! 这样其他平台上的 COLMAP 直接拿本库的 `descriptors.data` 匹配也能用（方案 D3）。

use std::f32::consts::PI;

/// SIFT 描述子维度。
pub const DESCRIPTOR_DIM: usize = 128;

/// 尺度空间最小模糊。
const SIGMA_MIN: f32 = 1.6;
/// 假定输入图像已具备的模糊（COLMAP/VLFeat 约定）。
const INPUT_SIGMA: f32 = 0.5;
/// 上采样后单张金字塔图的最大像素数（内存护栏，见模块注释）。
const MAX_PYRAMID_PIXELS: usize = 60_000_000;
/// 八度最小边长（再小就没有意义）。
const MIN_OCTAVE_SIZE: usize = 8;

const ORIENTATION_BINS: usize = 36;
const ORIENTATION_SMOOTH_PASSES: usize = 2;
const ORIENTATION_SIGMA_FACTOR: f32 = 1.5;
const ORIENTATION_RADIUS_FACTOR: f32 = 3.0;
const ORIENTATION_PEAK_RATIO: f32 = 0.8;

const DESC_WIDTH: usize = 4;
const DESC_BINS: usize = 8;
const DESC_SCALE_FACTOR: f32 = 3.0;

const MAX_REFINE_STEPS: usize = 5;

/// 提取参数（默认值与 COLMAP `SiftExtractionOptions` 一致）。
#[derive(Clone, Debug)]
pub struct SiftOptions {
    /// 最多保留的特征数（0 = 不限制）；超限时保留粗尺度特征。
    pub max_num_features: usize,
    /// 首个八度；负数表示先做 `2^-first_octave` 倍上采样。
    pub first_octave: i32,
    pub num_octaves: usize,
    pub octave_resolution: usize,
    pub peak_threshold: f32,
    pub edge_threshold: f32,
    /// 每个关键点最多分配几个方向（`upright = false` 时生效）。
    pub max_num_orientations: usize,
    /// 是否固定方向为 0（upright SIFT）。
    pub upright: bool,
}

impl Default for SiftOptions {
    fn default() -> Self {
        Self {
            max_num_features: 8192,
            first_octave: -1,
            num_octaves: 4,
            octave_resolution: 3,
            peak_threshold: 0.02 / 3.0,
            edge_threshold: 10.0,
            max_num_orientations: 2,
            upright: false,
        }
    }
}

/// 灰度平面（`f32`，值域 `[0, 1]`）。像素 `(x, y)` 的中心在连续坐标 `(x + 0.5, y + 0.5)`。
#[derive(Clone, Debug)]
pub struct GrayImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<f32>,
}

impl GrayImage {
    pub fn new(width: usize, height: usize, pixels: Vec<f32>) -> Self {
        assert_eq!(pixels.len(), width * height, "灰度图像素个数与尺寸不符");
        Self {
            width,
            height,
            pixels,
        }
    }

    /// 最近邻取样（越界用边缘像素补齐，与卷积的边界处理一致）。
    #[inline]
    fn at(&self, x: isize, y: isize) -> f32 {
        let x = x.clamp(0, self.width as isize - 1) as usize;
        let y = y.clamp(0, self.height as isize - 1) as usize;
        self.pixels[y * self.width + x]
    }

    /// 中心差分梯度。
    #[inline]
    fn gradient(&self, x: isize, y: isize) -> (f32, f32) {
        (
            0.5 * (self.at(x + 1, y) - self.at(x - 1, y)),
            0.5 * (self.at(x, y + 1) - self.at(x, y - 1)),
        )
    }

    /// 双线性取样（连续坐标，中心约定）。
    fn bilinear(&self, x: f32, y: f32) -> f32 {
        let xc = x - 0.5;
        let yc = y - 0.5;
        let x0 = xc.floor();
        let y0 = yc.floor();
        let fx = xc - x0;
        let fy = yc - y0;
        let (x0, y0) = (x0 as isize, y0 as isize);
        let top = self.at(x0, y0) * (1.0 - fx) + self.at(x0 + 1, y0) * fx;
        let bottom = self.at(x0, y0 + 1) * (1.0 - fx) + self.at(x0 + 1, y0 + 1) * fx;
        top * (1.0 - fy) + bottom * fy
    }
}

/// 一个 SIFT 特征；坐标/尺度都在**输入图像**的像素坐标系里。
#[derive(Clone, Debug)]
pub struct SiftFeature {
    /// 像素中心约定下的横坐标（与 COLMAP `FeatureKeypoint::x` 同义）。
    pub x: f32,
    pub y: f32,
    /// 高斯尺度（输入图像像素）。
    pub scale: f32,
    /// 主方向（弧度，`[0, 2π)`）。
    #[allow(dead_code)] // 方向已经编码在 shape 里；这里留字段给 M4 的可视化/匹配用
    pub orientation: f32,
    /// `a11, a12, a21, a22`（COLMAP 约定：`a11 = s·cos`、`a12 = -s·sin`、
    /// `a21 = s·sin`、`a22 = s·cos`）。
    pub shape: [f32; 4],
    /// `|D̂|`（DoG 响应绝对值，仅用于内部排序/裁剪）。
    #[allow(dead_code)] // M4 的匹配阈值/排序可能要用
    pub response: f32,
    pub descriptor: [u8; DESCRIPTOR_DIM],
}

/// 提取结果。
#[derive(Clone, Debug, Default)]
pub struct Extraction {
    pub features: Vec<SiftFeature>,
    /// 运行说明（交给上层写日志）。
    pub notes: Vec<String>,
}

/// 在灰度图上提取 SIFT 特征。
pub fn extract(image: &GrayImage, options: &SiftOptions) -> Extraction {
    let mut notes = Vec::new();
    let resolution = options.octave_resolution.max(1);

    // 首八度：按 first_octave 做 2 的幂次上采样（VLFeat 的语义）。
    let mut upsample = 1usize;
    if options.first_octave < 0 {
        upsample = 1usize << ((-options.first_octave).min(3) as usize);
    }
    let mut base = if upsample > 1 {
        upsample_bilinear(image, upsample)
    } else {
        image.clone()
    };
    if base.width * base.height > MAX_PYRAMID_PIXELS {
        notes.push(format!(
            "图像 {}×{} 上采样后超过内存护栏（{} 像素），跳过首八度上采样",
            image.width, image.height, MAX_PYRAMID_PIXELS
        ));
        base = image.clone();
        upsample = 1;
    }

    // 把输入等效模糊推到 SIGMA_MIN。
    let input_sigma = INPUT_SIGMA * upsample as f32;
    let mut current = blur(&base, sigma_delta(SIGMA_MIN, input_sigma));
    // 本八度像素 → 输入图像坐标的比例。
    let mut pixel_scale = 1.0 / upsample as f32;

    // 按 (八度, DoG 层) 分组收集：顺序是"先细后粗"，裁剪时保留尾部（粗尺度）。
    let mut groups: Vec<Vec<SiftFeature>> = Vec::new();
    for _ in 0..options.num_octaves.max(1) {
        if current.width.min(current.height) < MIN_OCTAVE_SIZE {
            break;
        }
        let Some(next) = extract_octave(&current, resolution, options, pixel_scale, &mut groups)
        else {
            break;
        };
        current = next;
        pixel_scale *= 2.0;
    }

    let mut features = retain_coarse(groups, options.max_num_features);
    // 稳定排序：坐标升序，便于测试与排查（写库顺序与 COLMAP 一样不做要求）。
    features.sort_by(|a, b| {
        (a.y, a.x, a.scale)
            .partial_cmp(&(b.y, b.x, b.scale))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Extraction { features, notes }
}

/// 按 `max_num_features` 保留**粗尺度**特征。
///
/// COLMAP 的语义（`sift.cc`）：按 (八度, DoG 层) 分组，从最粗的组往回累计，
/// 跨过阈值的那个组整体保留（因此结果可能略多于上限）。
fn retain_coarse(groups: Vec<Vec<SiftFeature>>, max_num_features: usize) -> Vec<SiftFeature> {
    let total: usize = groups.iter().map(Vec::len).sum();
    let first = if max_num_features > 0 && total > max_num_features {
        let mut count = 0usize;
        let mut first = 0usize;
        for (index, group) in groups.iter().enumerate().rev() {
            count += group.len();
            first = index;
            if count > max_num_features {
                break;
            }
        }
        first
    } else {
        0
    };
    groups
        .into_iter()
        .skip(first)
        .flatten()
        .collect::<Vec<_>>()
}

/// 处理一个八度；返回下一个八度的底图（`None` 表示不再有下一个八度）。
///
/// 内存上只保留 3 张高斯层 + 3 张 DoG（滚动窗口），而不是整层金字塔。
fn extract_octave(
    base: &GrayImage,
    resolution: usize,
    options: &SiftOptions,
    pixel_scale: f32,
    groups: &mut Vec<Vec<SiftFeature>>,
) -> Option<GrayImage> {
    let num_layers = resolution + 3;
    let (width, height) = (base.width, base.height);
    let blank = || GrayImage::new(width, height, vec![0.0; width * height]);
    // gauss[k % 3] 依次表示第 k 层高斯图；dog[k % 3] 依次表示第 k 张 DoG。
    let mut gauss = [blank(), blank(), blank()];
    let mut dog = [blank(), blank(), blank()];

    for layer in 0..num_layers {
        if layer == 0 {
            gauss[0].pixels.copy_from_slice(&base.pixels);
        } else {
            let previous = (layer - 1) % 3;
            let current = layer % 3;
            let sigma_previous = SIGMA_MIN * 2f32.powf((layer - 1) as f32 / resolution as f32);
            let sigma_current = SIGMA_MIN * 2f32.powf(layer as f32 / resolution as f32);
            // `previous != current`，用 `split_at_mut` 拿两块不重叠的可变借用。
            let (source, destination) = if previous < current {
                let (head, tail) = gauss.split_at_mut(current);
                (&head[previous], &mut tail[0])
            } else {
                let (head, tail) = gauss.split_at_mut(previous);
                (&tail[0], &mut head[current])
            };
            blur_into(
                source,
                destination,
                sigma_delta(sigma_current, sigma_previous),
            );
            subtract_into(destination, source, &mut dog[(layer - 1) % 3]);
        }

        // 有了 dog[layer-1] 之后，可以检测 dog[layer-2]（需要三张 DoG）。
        if layer >= 3 {
            let index = layer - 2;
            let previous = (index - 1) % 3;
            let current = index % 3;
            let next = (index + 1) % 3;
            // 描述子取样层：l ∈ {index, index+1}，两张都在窗口里。
            let mut group = Vec::new();
            detect_scale_space(
                &dog[previous],
                &dog[current],
                &dog[next],
                &gauss[index % 3],
                &gauss[(index + 1) % 3],
                index,
                resolution,
                options,
                pixel_scale,
                &mut group,
            );
            groups.push(group);
        }
    }

    // 下一八度的底图 = 第 resolution 层高斯图降采样 2×（模糊正好是 2·SIGMA_MIN）。
    Some(downsample2(&gauss[resolution % 3]))
}

/// 在 DoG 层 `index` 上找极值点，做精化/方向/描述子，追加到 `group`。
#[allow(clippy::too_many_arguments)]
fn detect_scale_space(
    previous: &GrayImage,
    current: &GrayImage,
    next: &GrayImage,
    layer_lo: &GrayImage,
    layer_hi: &GrayImage,
    index: usize,
    resolution: usize,
    options: &SiftOptions,
    pixel_scale: f32,
    group: &mut Vec<SiftFeature>,
) {
    let (width, height) = (current.width, current.height);
    if width < 3 || height < 3 {
        return;
    }
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let Some(extremum) = refine_extremum(previous, current, next, x, y, options) else {
                continue;
            };
            // DoG `index` 位于高斯层 `index` 与 `index + 1` 之间。
            let level = index as f32 + 0.5 + extremum.offset_level;
            let sigma = SIGMA_MIN * 2f32.powf(level / resolution as f32);
            let sampling = if level.round() as usize <= index {
                layer_lo
            } else {
                layer_hi
            };
            let (cx, cy) = (x as f32 + extremum.offset_x, y as f32 + extremum.offset_y);
            let orientations = if options.upright {
                vec![0.0f32]
            } else {
                orientation_peaks(
                    sampling,
                    cx,
                    cy,
                    sigma,
                    options.max_num_orientations,
                )
            };
            for orientation in orientations {
                let scale = sigma * pixel_scale;
                group.push(SiftFeature {
                    x: (cx + 0.5) * pixel_scale,
                    y: (cy + 0.5) * pixel_scale,
                    scale,
                    orientation,
                    shape: [
                        scale * orientation.cos(),
                        -scale * orientation.sin(),
                        scale * orientation.sin(),
                        scale * orientation.cos(),
                    ],
                    response: extremum.response,
                    descriptor: descriptor(sampling, cx, cy, sigma, orientation),
                });
            }
        }
    }
}

/// 精化结果：相对检测点的亚像素偏移与响应。
struct Extremum {
    offset_x: f32,
    offset_y: f32,
    offset_level: f32,
    response: f32,
}

/// Lowe 的极值精化（`interpExtrema`）：泰勒展开求极值 → 峰值阈值 → 边缘抑制。
fn refine_extremum(
    previous: &GrayImage,
    current: &GrayImage,
    next: &GrayImage,
    x: usize,
    y: usize,
    options: &SiftOptions,
) -> Option<Extremum> {
    let center = current.at(x as isize, y as isize);
    let mut is_max = true;
    let mut is_min = true;
    {
        let mut consider = |value: f32| {
            if value > center {
                is_max = false;
            }
            if value < center {
                is_min = false;
            }
        };
        for (dx, dy) in [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ] {
            consider(current.at(x as isize + dx, y as isize + dy));
        }
        for plane in [previous, next] {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    consider(plane.at(x as isize + dx, y as isize + dy));
                }
            }
        }
    }
    if !is_max && !is_min {
        return None;
    }

    let (max_x, max_y) = (current.width as isize - 2, current.height as isize - 2);
    let (mut ix, mut iy) = (x as isize, y as isize);
    let mut level_offset = 0.0f32;
    for _ in 0..MAX_REFINE_STEPS {
        let (gradient, hessian) = derivatives(previous, current, next, ix, iy);
        let solved = solve3(&hessian, &gradient)?;
        if solved[0].abs() < 0.5 && solved[1].abs() < 0.5 && solved[2].abs() < 0.5 {
            break;
        }
        // 与 Lowe 一致：`(int)(v + 0.5)`（向零截断）。
        ix += (solved[0] + 0.5) as isize;
        iy += (solved[1] + 0.5) as isize;
        level_offset += solved[2];
        if ix < 1 || iy < 1 || ix > max_x || iy > max_y {
            return None;
        }
        if level_offset.abs() > 1.5 {
            return None;
        }
    }

    // 用最终位置重算一次，得到响应与边缘抑制所需的二阶导。
    let (gradient, hessian) = derivatives(previous, current, next, ix, iy);
    let offset = solve3(&hessian, &gradient)?;
    let value = current.at(ix, iy);
    let response = (value
        + 0.5 * (gradient[0] * offset[0] + gradient[1] * offset[1] + gradient[2] * offset[2]))
        .abs();
    if response < options.peak_threshold {
        return None;
    }
    let trace = hessian[0][0] + hessian[1][1];
    let determinant = hessian[0][0] * hessian[1][1] - hessian[0][1] * hessian[0][1];
    if determinant <= 0.0 {
        return None;
    }
    let edge_threshold = options.edge_threshold.max(1.0);
    if trace * trace / determinant >= (edge_threshold + 1.0) * (edge_threshold + 1.0) / edge_threshold
    {
        return None;
    }

    Some(Extremum {
        offset_x: (ix - x as isize) as f32 + offset[0],
        offset_y: (iy - y as isize) as f32 + offset[1],
        offset_level: level_offset + offset[2],
        response,
    })
}

/// `D` 的一阶/二阶导（三张 DoG 的中心像素在 `current` 的 `(x, y)`）。
fn derivatives(
    previous: &GrayImage,
    current: &GrayImage,
    next: &GrayImage,
    x: isize,
    y: isize,
) -> ([f32; 3], [[f32; 3]; 3]) {
    let center = current.at(x, y);
    let gradient = [
        0.5 * (current.at(x + 1, y) - current.at(x - 1, y)),
        0.5 * (current.at(x, y + 1) - current.at(x, y - 1)),
        0.5 * (next.at(x, y) - previous.at(x, y)),
    ];
    let dxx = current.at(x + 1, y) + current.at(x - 1, y) - 2.0 * center;
    let dyy = current.at(x, y + 1) + current.at(x, y - 1) - 2.0 * center;
    let dss = next.at(x, y) + previous.at(x, y) - 2.0 * center;
    let dxy = 0.25
        * (current.at(x + 1, y + 1) - current.at(x + 1, y - 1) - current.at(x - 1, y + 1)
            + current.at(x - 1, y - 1));
    let dxs = 0.25
        * (next.at(x + 1, y) - next.at(x - 1, y) - previous.at(x + 1, y) + previous.at(x - 1, y));
    let dys = 0.25
        * (next.at(x, y + 1) - next.at(x, y - 1) - previous.at(x, y + 1) + previous.at(x, y - 1));
    (
        gradient,
        [[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]],
    )
}

/// 解对称 3×3 线性方程组 `h · x = -g`；奇异时返回 `None`。
fn solve3(h: &[[f32; 3]; 3], g: &[f32; 3]) -> Option<[f32; 3]> {
    let determinant = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
        - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
        + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
    if determinant.abs() < 1e-10 {
        return None;
    }
    let inverse = 1.0 / determinant;
    let c00 = (h[1][1] * h[2][2] - h[1][2] * h[2][1]) * inverse;
    let c01 = (h[0][2] * h[2][1] - h[0][1] * h[2][2]) * inverse;
    let c02 = (h[0][1] * h[1][2] - h[0][2] * h[1][1]) * inverse;
    let c11 = (h[0][0] * h[2][2] - h[0][2] * h[2][0]) * inverse;
    let c12 = (h[0][2] * h[1][0] - h[0][0] * h[1][2]) * inverse;
    let c22 = (h[0][0] * h[1][1] - h[0][1] * h[1][0]) * inverse;
    Some([
        -(c00 * g[0] + c01 * g[1] + c02 * g[2]),
        -(c01 * g[0] + c11 * g[1] + c12 * g[2]),
        -(c02 * g[0] + c12 * g[1] + c22 * g[2]),
    ])
}

/// 方向直方图（36 bin）→ 主方向（最多 `max_peaks` 个，按强度排序）。
fn orientation_peaks(
    image: &GrayImage,
    center_x: f32,
    center_y: f32,
    sigma: f32,
    max_peaks: usize,
) -> Vec<f32> {
    if max_peaks == 0 {
        return Vec::new();
    }
    let gaussian_sigma = ORIENTATION_SIGMA_FACTOR * sigma;
    let radius = ((ORIENTATION_RADIUS_FACTOR * gaussian_sigma).round() as isize).max(1);
    let center_x = center_x.round() as isize;
    let center_y = center_y.round() as isize;
    let mut histogram = [0.0f32; ORIENTATION_BINS];
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let x = center_x + dx;
            let y = center_y + dy;
            if x < 1 || y < 1 || x + 1 >= image.width as isize || y + 1 >= image.height as isize {
                continue;
            }
            let (gx, gy) = image.gradient(x, y);
            let magnitude = (gx * gx + gy * gy).sqrt();
            if magnitude <= 0.0 {
                continue;
            }
            let weight = (-((dx * dx + dy * dy) as f32)
                / (2.0 * gaussian_sigma * gaussian_sigma))
                .exp();
            let angle = gy.atan2(gx).rem_euclid(2.0 * PI);
            let position = angle * ORIENTATION_BINS as f32 / (2.0 * PI);
            let bin = position.floor() as isize;
            let fraction = position - bin as f32;
            let low = bin.rem_euclid(ORIENTATION_BINS as isize) as usize;
            let high = (low + 1) % ORIENTATION_BINS;
            let contribution = magnitude * weight;
            histogram[low] += contribution * (1.0 - fraction);
            histogram[high] += contribution * fraction;
        }
    }
    // 平滑（Lowe：2 轮 [1, 1, 1] / 3）。
    for _ in 0..ORIENTATION_SMOOTH_PASSES {
        let mut smoothed = [0.0f32; ORIENTATION_BINS];
        for index in 0..ORIENTATION_BINS {
            let previous = histogram[(index + ORIENTATION_BINS - 1) % ORIENTATION_BINS];
            let next = histogram[(index + 1) % ORIENTATION_BINS];
            smoothed[index] = (previous + histogram[index] + next) / 3.0;
        }
        histogram = smoothed;
    }
    let maximum = histogram.iter().copied().fold(0.0f32, f32::max);
    if maximum <= 0.0 {
        return vec![0.0];
    }
    let mut peaks: Vec<(f32, f32)> = Vec::new();
    for index in 0..ORIENTATION_BINS {
        let previous = histogram[(index + ORIENTATION_BINS - 1) % ORIENTATION_BINS];
        let next = histogram[(index + 1) % ORIENTATION_BINS];
        if histogram[index] <= previous
            || histogram[index] <= next
            || histogram[index] < ORIENTATION_PEAK_RATIO * maximum
        {
            continue;
        }
        let denominator = previous - 2.0 * histogram[index] + next;
        let interpolation = if denominator.abs() > 1e-6 {
            0.5 * (previous - next) / denominator
        } else {
            0.0
        };
        let angle = (2.0 * PI * (index as f32 + interpolation) / ORIENTATION_BINS as f32)
            .rem_euclid(2.0 * PI);
        peaks.push((histogram[index], angle));
    }
    peaks.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    peaks
        .into_iter()
        .take(max_peaks)
        .map(|(_, angle)| angle)
        .collect()
}

/// 128 维描述子（4×4 空间 × 8 方向，三线性插值；方向 bin 为 UBC/Lowe 顺序）。
fn descriptor(
    image: &GrayImage,
    center_x: f32,
    center_y: f32,
    sigma: f32,
    orientation: f32,
) -> [u8; DESCRIPTOR_DIM] {
    let hist_width = DESC_SCALE_FACTOR * sigma;
    let radius = (hist_width * 2f32.sqrt() * (DESC_WIDTH as f32 + 1.0) * 0.5).round() as isize;
    let cos_t = orientation.cos();
    let sin_t = orientation.sin();
    let exp_scale = -1.0 / (2.0 * (DESC_WIDTH as f32 / 2.0).powi(2) * hist_width * hist_width);
    let center_x = center_x.round() as isize;
    let center_y = center_y.round() as isize;
    let mut histogram = [[[0.0f32; DESC_BINS]; DESC_WIDTH]; DESC_WIDTH];
    for i in -radius..=radius {
        for j in -radius..=radius {
            let c_rot = j as f32 * cos_t - i as f32 * sin_t;
            let r_rot = j as f32 * sin_t + i as f32 * cos_t;
            let c_hist = c_rot / hist_width + DESC_WIDTH as f32 / 2.0 - 0.5;
            let r_hist = r_rot / hist_width + DESC_WIDTH as f32 / 2.0 - 0.5;
            if c_hist <= -1.0
                || c_hist >= DESC_WIDTH as f32
                || r_hist <= -1.0
                || r_hist >= DESC_WIDTH as f32
            {
                continue;
            }
            let x = center_x + j;
            let y = center_y + i;
            if x < 1 || y < 1 || x + 1 >= image.width as isize || y + 1 >= image.height as isize {
                continue;
            }
            let (gx, gy) = image.gradient(x, y);
            let magnitude = (gx * gx + gy * gy).sqrt();
            // 方向 bin 取镜像（等价 COLMAP 的 VLFeat → UBC 变换）。
            let angle = (orientation - gy.atan2(gx)).rem_euclid(2.0 * PI);
            let position = DESC_BINS as f32 * angle / (2.0 * PI);
            let bin = position.floor() as isize;
            let bin_fraction = position - bin as f32;
            let bin_low = bin.rem_euclid(DESC_BINS as isize) as usize;
            let bin_high = (bin_low + 1) % DESC_BINS;
            let row = r_hist.floor() as isize;
            let column = c_hist.floor() as isize;
            let row_fraction = r_hist - row as f32;
            let column_fraction = c_hist - column as f32;
            let weight =
                ((c_rot * c_rot + r_rot * r_rot) * exp_scale).exp() * magnitude;
            for (row_index, row_weight) in [(row, 1.0 - row_fraction), (row + 1, row_fraction)] {
                if row_index < 0 || row_index >= DESC_WIDTH as isize {
                    continue;
                }
                for (column_index, column_weight) in
                    [(column, 1.0 - column_fraction), (column + 1, column_fraction)]
                {
                    if column_index < 0 || column_index >= DESC_WIDTH as isize {
                        continue;
                    }
                    let weight = weight * row_weight * column_weight;
                    let cell = &mut histogram[row_index as usize][column_index as usize];
                    cell[bin_low] += weight * (1.0 - bin_fraction);
                    cell[bin_high] += weight * bin_fraction;
                }
            }
        }
    }

    // 量化：L1 归一化 → 开方 → ×512 → 四舍五入 → 截断到 u8。
    let mut values = [0.0f32; DESCRIPTOR_DIM];
    let mut l1 = 0.0f32;
    for (index, value) in values.iter_mut().enumerate() {
        let row = index / (DESC_WIDTH * DESC_BINS);
        let column = (index / DESC_BINS) % DESC_WIDTH;
        let bin = index % DESC_BINS;
        *value = histogram[row][column][bin];
        l1 += value.abs();
    }
    if l1 > 0.0 {
        for value in values.iter_mut() {
            *value = (*value / l1).max(0.0).sqrt();
        }
    }
    let mut out = [0u8; DESCRIPTOR_DIM];
    for (index, value) in values.iter().enumerate() {
        out[index] = (512.0 * value).round().clamp(0.0, 255.0) as u8;
    }
    out
}

// ---------------------------------------------------------------- 图像操作

/// `sigma_delta(a, b) = sqrt(a² - b²)`（`a <= b` 时返回 0，表示不用再模糊）。
fn sigma_delta(a: f32, b: f32) -> f32 {
    let delta = a * a - b * b;
    if delta <= 1e-6 {
        0.0
    } else {
        delta.sqrt()
    }
}

/// 高斯核（半径 = ceil(3σ)）。
fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil().max(1.0) as isize;
    let mut kernel = Vec::with_capacity((2 * radius + 1) as usize);
    let denominator = 2.0 * sigma * sigma;
    let mut sum = 0.0f32;
    for offset in -radius..=radius {
        let value = (-((offset * offset) as f32) / denominator).exp();
        kernel.push(value);
        sum += value;
    }
    for value in kernel.iter_mut() {
        *value /= sum;
    }
    kernel
}

/// 可分离高斯模糊（边界用最近邻延拓）；`dst` 可以与 `src` 尺寸不同时不允许。
fn blur(image: &GrayImage, sigma: f32) -> GrayImage {
    let mut output = GrayImage::new(image.width, image.height, vec![0.0; image.pixels.len()]);
    blur_into(image, &mut output, sigma);
    output
}

/// 就地可分离高斯模糊（`src` 与 `dst` 尺寸必须相同，且不能是同一块内存）。
fn blur_into(src: &GrayImage, dst: &mut GrayImage, sigma: f32) {
    debug_assert_eq!((src.width, src.height), (dst.width, dst.height));
    if sigma <= 1e-3 {
        dst.pixels.copy_from_slice(&src.pixels);
        return;
    }
    let kernel = gaussian_kernel(sigma);
    let radius = (kernel.len() / 2) as isize;
    let (width, height) = (src.width, src.height);
    // 横向。
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f32;
            for (offset, weight) in kernel.iter().enumerate() {
                sum += weight * src.at(x as isize + offset as isize - radius, y as isize);
            }
            dst.pixels[y * width + x] = sum;
        }
    }
    // 纵向（逐列取到临时缓冲，避免读写冲突）。
    let mut column = vec![0.0f32; height + 2 * radius as usize];
    for x in 0..width {
        for (index, value) in column.iter_mut().enumerate() {
            let y = index as isize - radius;
            *value = dst.pixels[y.clamp(0, height as isize - 1) as usize * width + x];
        }
        for y in 0..height {
            let mut sum = 0.0f32;
            for (offset, weight) in kernel.iter().enumerate() {
                sum += weight * column[y + offset];
            }
            dst.pixels[y * width + x] = sum;
        }
    }
}

/// `dst = a - b`。
fn subtract_into(a: &GrayImage, b: &GrayImage, dst: &mut GrayImage) {
    debug_assert_eq!((a.width, a.height), (dst.width, dst.height));
    for (index, value) in dst.pixels.iter_mut().enumerate() {
        *value = a.pixels[index] - b.pixels[index];
    }
}

/// 2× 双线性上采样（`factor` 为 2 的幂）。
fn upsample_bilinear(image: &GrayImage, factor: usize) -> GrayImage {
    let width = image.width * factor;
    let height = image.height * factor;
    let mut pixels = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let source_x = (x as f32 + 0.5) / factor as f32;
            let source_y = (y as f32 + 0.5) / factor as f32;
            pixels[y * width + x] = image.bilinear(source_x, source_y);
        }
    }
    GrayImage::new(width, height, pixels)
}

/// 2× 降采样（2×2 均值，中心约定一致）。
fn downsample2(image: &GrayImage) -> GrayImage {    let width = (image.width / 2).max(1);
    let height = (image.height / 2).max(1);
    let mut pixels = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let (x0, y0) = ((2 * x) as isize, (2 * y) as isize);
            let sum = image.at(x0, y0)
                + image.at(x0 + 1, y0)
                + image.at(x0, y0 + 1)
                + image.at(x0 + 1, y0 + 1);
            pixels[y * width + x] = sum / 4.0;
        }
    }
    GrayImage::new(width, height, pixels)
}

/// 缩放到指定尺寸。
///
/// 缩小时先按比例做一次等效低通（近似 COLMAP 用 OIIO triangle 滤波下采样的抗锯齿效果），
/// 再做双线性重采样；放大时仅双线性。
pub fn resize(image: &GrayImage, width: usize, height: usize) -> GrayImage {
    if width == image.width && height == image.height {
        return image.clone();
    }
    let scale_x = image.width as f32 / width as f32;
    let scale_y = image.height as f32 / height as f32;
    let scale = scale_x.min(scale_y);
    let source = if scale > 1.0 {
        blur(image, 0.5 * (scale - 1.0))
    } else {
        image.clone()
    };
    let mut pixels = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let source_x = (x as f32 + 0.5) * scale_x;
            let source_y = (y as f32 + 0.5) * scale_y;
            pixels[y * width + x] = source.bilinear(source_x, source_y);
        }
    }
    GrayImage::new(width, height, pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 合成图：渐变背景 + 若干高斯亮斑（平滑图，只有亮斑中心是极值点）。
    fn blobs(width: usize, height: usize, centers: &[(f32, f32)]) -> GrayImage {
        let mut pixels = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let mut value = 0.2 + 0.3 * (x as f32 / width as f32);
                for (cx, cy) in centers {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    value += 0.6 * (-(dx * dx + dy * dy) / (2.0 * 3.0 * 3.0)).exp();
                }
                pixels[y * width + x] = value.clamp(0.0, 1.0);
            }
        }
        GrayImage::new(width, height, pixels)
    }

    /// 合成图：棋盘格 + 正弦波纹（角点很多，特征数量接近真实照片的量级）。
    fn texture(width: usize, height: usize, phase: f32) -> GrayImage {
        let mut pixels = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let (fx, fy) = (x as f32, y as f32);
                let checker = if ((fx / 16.0).floor() as i32 + (fy / 16.0).floor() as i32) % 2 == 0
                {
                    0.27
                } else {
                    -0.27
                };
                let wave = 0.15 * ((fx + phase) * 0.05).sin() * (fy * 0.04).cos();
                pixels[y * width + x] = (0.5 + checker + wave).clamp(0.0, 1.0);
            }
        }
        GrayImage::new(width, height, pixels)
    }

    fn options() -> SiftOptions {
        SiftOptions {
            num_octaves: 3,
            max_num_features: 0,
            ..Default::default()
        }
    }

    #[test]
    fn finds_synthetic_blobs() {
        let centers = [(64.0f32, 64.0f32), (180.0, 90.0), (100.0, 190.0)];
        let image = blobs(256, 256, &centers);
        let extraction = extract(&image, &options());
        // 平滑图只有亮斑中心是 DoG 极值点（每个中心可能有 1–2 个方向）。
        assert!(
            extraction.features.len() >= centers.len(),
            "每个亮斑至少应有一个特征点：{}",
            extraction.features.len()
        );
        for (cx, cy) in centers {
            let near = extraction
                .features
                .iter()
                .filter(|feature| (feature.x - cx).abs() < 4.0 && (feature.y - cy).abs() < 4.0)
                .count();
            assert!(near > 0, "亮斑 ({cx}, {cy}) 附近应有特征点");
        }
        for feature in &extraction.features {
            assert!(feature.x > 0.0 && feature.x < 256.0);
            assert!(feature.y > 0.0 && feature.y < 256.0);
            assert!(feature.scale > 0.0);
            assert!((0.0..2.0 * PI).contains(&feature.orientation));
            // 形状矩阵与 scale/方向一致。
            let expected = [
                feature.scale * feature.orientation.cos(),
                -feature.scale * feature.orientation.sin(),
                feature.scale * feature.orientation.sin(),
                feature.scale * feature.orientation.cos(),
            ];
            for (actual, expected) in feature.shape.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn finds_texture_corners() {
        let image = texture(256, 256, 0.0);
        let extraction = extract(&image, &options());
        assert!(
            extraction.features.len() > 200,
            "棋盘纹理应检出几百个特征：{}",
            extraction.features.len()
        );
        // 尺度分布应覆盖多个八度（说明 2× 上采样与金字塔在正常工作）。
        let finest = extraction
            .features
            .iter()
            .map(|feature| feature.scale)
            .fold(f32::INFINITY, f32::min);
        let coarsest = extraction
            .features
            .iter()
            .map(|feature| feature.scale)
            .fold(0.0f32, f32::max);
        assert!(
            coarsest / finest > 3.0,
            "尺度范围应有多个八度：{finest} … {coarsest}"
        );
    }

    #[test]
    fn extraction_is_deterministic() {
        let image = texture(192, 160, 1.5);
        let first = extract(&image, &options());
        let second = extract(&image, &options());
        assert_eq!(first.features.len(), second.features.len());
        for (a, b) in first.features.iter().zip(&second.features) {
            assert_eq!((a.x, a.y, a.scale, a.orientation), (b.x, b.y, b.scale, b.orientation));
            assert_eq!(a.descriptor, b.descriptor);
        }
    }

    #[test]
    fn max_num_features_keeps_coarse_scales() {
        let image = texture(320, 240, 3.0);
        let full = extract(
            &image,
            &SiftOptions {
                num_octaves: 4,
                max_num_features: 0,
                ..Default::default()
            },
        );
        let capped = extract(
            &image,
            &SiftOptions {
                num_octaves: 4,
                max_num_features: 40,
                ..Default::default()
            },
        );
        assert!(full.features.len() > 400);
        assert!(capped.features.len() < full.features.len());
        // 保留的组可能整体越过上限（COLMAP 的 `first_level_to_keep` 语义）。
        assert!(capped.features.len() <= full.features.len());
        let finest = full
            .features
            .iter()
            .map(|feature| feature.scale)
            .fold(f32::INFINITY, f32::min);
        let capped_finest = capped
            .features
            .iter()
            .map(|feature| feature.scale)
            .fold(f32::INFINITY, f32::min);
        assert!(
            capped_finest > finest,
            "裁剪后应丢掉最细尺度（{capped_finest} vs {finest}）"
        );
    }

    #[test]
    fn upright_has_zero_orientation() {
        let image = texture(160, 160, 2.0);
        let extraction = extract(
            &image,
            &SiftOptions {
                upright: true,
                max_num_features: 0,
                ..Default::default()
            },
        );
        assert!(!extraction.features.is_empty());
        for feature in &extraction.features {
            assert_eq!(feature.orientation, 0.0);
        }
    }

    #[test]
    fn descriptors_are_l1_root_normalized() {
        let image = texture(200, 200, 0.5);
        let extraction = extract(&image, &options());
        assert!(!extraction.features.is_empty());
        for feature in extraction.features.iter().take(32) {
            let norm: f32 = feature
                .descriptor
                .iter()
                .map(|value| {
                    let value = *value as f32 / 512.0;
                    value * value
                })
                .sum();
            // L1 归一化 + 开方 ⇒ 平方和 = 1（量化与 255 截断只会让它变小）。
            assert!(
                (0.6..=1.05).contains(&norm),
                "描述子平方和应在 1 附近：{norm}"
            );
            assert!(feature.descriptor.iter().any(|value| *value > 0));
        }
    }

    #[test]
    fn downsampled_next_octave_keeps_dimensions() {
        let image = blobs(200, 100, &[(100.0, 50.0)]);
        let extraction = extract(
            &image,
            &SiftOptions {
                num_octaves: 4,
                first_octave: 0,
                max_num_features: 0,
                ..Default::default()
            },
        );
        // first_octave = 0 时不应有上采样 note。
        assert!(extraction.notes.is_empty());
        assert!(extraction.features.iter().all(|feature| feature.scale >= 0.5));
    }
}
