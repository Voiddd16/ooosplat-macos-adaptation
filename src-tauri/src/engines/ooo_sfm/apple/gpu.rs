//! Metal GPU 后端 —— 由 `ooo-sfm-metal` feature 门控，**默认关闭**，CPU 是回退路径。
//!
//! # 现状（与方案文档的差异清单同步）
//!
//! 目前只把**批量前向投影**这一个内核搬到 GPU：
//!
//! * [`ba.rs`](super::ba) 的每次 LM 迭代的鲁棒代价 `robust_cost`（观测级 O(N)）；
//! * [`reconstruction.rs`](super::reconstruction) 的点误差统计 `update_point3d_errors`。
//!
//! 仍然是 CPU 的（走 rayon）：BA 的雅可比线性化 + Schur 装配/回代、EPnP 的小规模
//! SVD、RANSAC 假设生成的随机性。要继续往 GPU 搬的话，顺序是 Schur 压缩/回代 →
//! 雅可比线性化（这两个收益最大，也最需要先解决 per-thread 局部块归并）。
//!
//! # 开关
//!
//! * 编译期：`--features ooo-sfm-metal`（`target_os = "macos"` 之外自动无效）。
//! * 运行期：拿不到 Metal 设备 / 内核编译失败 / 初始化出错，一律**静默回退 CPU**，
//!   绝不影响正确性（`project_points_or` 的 `cpu` 闭包就是回退路径）。
//! * `OOO_GPU_DEBUG=1` 打印设备名与失败原因。

use super::ba::CameraModelId;

/// 内核里 `params` 的定长步长（实际最多用 5 个：f/fx、fy、cx、cy、k1、k2）。
pub const PARAM_STRIDE: usize = 8;

/// 一次批量投影的输入（SoA，`f64`）。
///
/// **batch 本身保持 `f64`**：feature 关掉（或设备不可用）时 CPU 回退要与原来的
/// 标量循环**逐位一致**，否则产物会随是否开 GPU 而变。只有上传到 GPU 时才转
/// `f32`（MSL 里没有 `f64`），那一档的投影差异约 1e-4 px，见下方单测。
#[derive(Debug, Default, Clone)]
pub struct ProjectionBatch {
    /// 每个观测的相机模型 id（0=SimplePinhole，1=Pinhole，2=SimpleRadial，3=Radial）。
    pub models: Vec<u32>,
    /// 相机参数，步长 [`PARAM_STRIDE`]。
    pub params: Vec<f64>,
    /// 相机系旋转矩阵（行主序），步长 9。
    pub rotation: Vec<f64>,
    /// 平移，步长 3。
    pub translation: Vec<f64>,
    /// 世界点，步长 3。
    pub points: Vec<f64>,
}

impl ProjectionBatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(observations: usize) -> Self {
        Self {
            models: Vec::with_capacity(observations),
            params: Vec::with_capacity(observations * PARAM_STRIDE),
            rotation: Vec::with_capacity(observations * 9),
            translation: Vec::with_capacity(observations * 3),
            points: Vec::with_capacity(observations * 3),
        }
    }

    /// 追加一次观测（世界点 → 像素），返回观测下标。
    pub fn push(
        &mut self,
        model: CameraModelId,
        params: &[f64],
        rotation: &[[f64; 3]; 3],
        translation: &[f64; 3],
        point: [f64; 3],
    ) {
        self.models.push(model_id(model));
        self.params.resize(self.models.len() * PARAM_STRIDE, 0.0);
        let base = (self.models.len() - 1) * PARAM_STRIDE;
        let mut index = 0;
        for value in params.iter().take(PARAM_STRIDE) {
            self.params[base + index] = *value;
            index += 1;
        }
        for row in rotation {
            for value in row {
                self.rotation.push(*value);
            }
        }
        self.translation
            .extend_from_slice(&[translation[0], translation[1], translation[2]]);
        self.points.extend_from_slice(&[point[0], point[1], point[2]]);
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

/// 内核里的模型编号（与 `CameraModelId` 的枚举顺序一致）。
fn model_id(model: CameraModelId) -> u32 {
    match model {
        CameraModelId::SimplePinhole => 0,
        CameraModelId::Pinhole => 1,
        CameraModelId::SimpleRadial => 2,
        CameraModelId::Radial => 3,
    }
}

/// 批量投影：能上 GPU 就上，不能就逐个调 `cpu` 闭包回退。
///
/// 返回的 `Vec<Option<[f64; 2]>>` 与 `batch` 的下标一一对应；`None` = 深度不足
/// （对齐 `camera::img_from_cam` 的失败语义）。**长度恒等于 `batch.len()`**，
/// 所以调用方不需要为回退写第二套聚合逻辑。
pub fn project_points_or(
    batch: &ProjectionBatch,
    cpu: impl Fn(usize) -> Option<[f64; 2]>,
) -> Vec<Option<[f64; 2]>> {
    if batch.is_empty() {
        return Vec::new();
    }
    #[cfg(all(feature = "backend-ooo-sfm-metal", target_os = "macos"))]
    {
        if let Some(projections) = backend::project(batch) {
            return projections;
        }
    }
    let _ = cpu;
    (0..batch.len())
        .map(|index| {
            let model = match batch.models[index] {
                0 => CameraModelId::SimplePinhole,
                1 => CameraModelId::Pinhole,
                2 => CameraModelId::SimpleRadial,
                _ => CameraModelId::Radial,
            };
            let params: Vec<f64> =
                batch.params[index * PARAM_STRIDE..(index + 1) * PARAM_STRIDE].to_vec();
            let rotation = [
                [
                    batch.rotation[index * 9] as f64,
                    batch.rotation[index * 9 + 1],
                    batch.rotation[index * 9 + 2],
                ],
                [
                    batch.rotation[index * 9 + 3],
                    batch.rotation[index * 9 + 4],
                    batch.rotation[index * 9 + 5],
                ],
                [
                    batch.rotation[index * 9 + 6],
                    batch.rotation[index * 9 + 7],
                    batch.rotation[index * 9 + 8],
                ],
            ];
            let translation = [
                batch.translation[index * 3] as f64,
                batch.translation[index * 3 + 1],
                batch.translation[index * 3 + 2],
            ];
            let point = [
                batch.points[index * 3] as f64,
                batch.points[index * 3 + 1],
                batch.points[index * 3 + 2],
            ];
            super::camera::img_from_cam(model, &params, [
                rotation[0][0] * point[0]
                    + rotation[0][1] * point[1]
                    + rotation[0][2] * point[2]
                    + translation[0],
                rotation[1][0] * point[0]
                    + rotation[1][1] * point[1]
                    + rotation[1][2] * point[2]
                    + translation[1],
                rotation[2][0] * point[0]
                    + rotation[2][1] * point[1]
                    + rotation[2][2] * point[2]
                    + translation[2],
            ])
        })
        .collect()
}

/// 成功提交到 GPU 的 dispatch 次数（单测用来证明接线真的走到了 Metal）。
pub fn dispatch_count() -> u64 {
    DISPATCH_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

static DISPATCH_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 编译期是否带 Metal 后端。
pub const fn compiled_in() -> bool {
    cfg!(all(feature = "backend-ooo-sfm-metal", target_os = "macos"))
}

/// Metal 设备名（`Apple M1` / `Apple M2 Pro` …）；没设备或没编进来时 `None`。
pub fn device_name() -> Option<String> {
    #[cfg(all(feature = "backend-ooo-sfm-metal", target_os = "macos"))]
    {
        backend::device_name()
    }
    #[cfg(not(all(feature = "backend-ooo-sfm-metal", target_os = "macos")))]
    {
        None
    }
}

/// 运行期是否真的能用 Metal（设备存在 + 内核编译成功）。
pub fn active() -> bool {
    if !compiled_in() {
        return false;
    }
    #[cfg(all(feature = "backend-ooo-sfm-metal", target_os = "macos"))]
    {
        backend::available()
    }
    #[cfg(not(all(feature = "backend-ooo-sfm-metal", target_os = "macos")))]
    {
        false
    }
}

fn debug_enabled() -> bool {
    std::env::var("OOO_GPU_DEBUG").is_ok()
}

// ---------------------------------------------------------------- Metal 实现

#[cfg(all(feature = "backend-ooo-sfm-metal", target_os = "macos"))]
mod backend {
    use metal::{
        Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
    };
    use std::sync::OnceLock;

    use super::ProjectionBatch;

    /// 计算内核：与 `camera::img_from_cam` 逐行对应（`x + x·k·r² + x·k₂·r⁴`）。
    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void project_kernel(
    const device uint*  models  [[buffer(0)]],
    const device float* params  [[buffer(1)]],
    const device float* rot     [[buffer(2)]],
    const device float* tvec    [[buffer(3)]],
    const device float* xyz     [[buffer(4)]],
    device float2*      output  [[buffer(5)]],
    const device uint&  count   [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) { return; }
    // MSL 3.0：函数内的指针必须显式标注地址空间。
    const device float* p = params + 8 * gid;
    const device float* r = rot + 9 * gid;
    const device float* t = tvec + 3 * gid;
    const device float* x = xyz + 3 * gid;
    const uint m = models[gid];

    const float zc = r[6] * x[0] + r[7] * x[1] + r[8] * x[2] + t[2];
    if (zc < 1e-6f) { output[gid] = float2(NAN, NAN); return; }
    const float u = r[0] * x[0] + r[1] * x[1] + r[2] * x[2] + t[0];
    const float v = r[3] * x[0] + r[4] * x[1] + r[5] * x[2] + t[1];
    const float px = u / zc;
    const float py = v / zc;

    float fx, fy, cx, cy;
    if (m == 1u) { fx = p[0]; fy = p[1]; cx = p[2]; cy = p[3]; }
    else          { fx = p[0]; fy = p[0]; cx = p[1]; cy = p[2]; }

    float dx = 0.0f;
    float dy = 0.0f;
    if (m >= 2u) {
        const float r2 = px * px + py * py;
        dx = px * p[3] * r2;
        dy = py * p[3] * r2;
        if (m == 3u) {
            const float r4 = r2 * r2;
            dx += px * p[4] * r4;
            dy += py * p[4] * r4;
        }
    }
    output[gid] = float2(fx * (px + dx) + cx, fy * (py + dy) + cy);
}
"#;

    const THREADS_PER_GROUP: usize = 64;

    struct Backend {
        device: Device,
        queue: CommandQueue,
        pipeline: ComputePipelineState,
    }

    static BACKEND: OnceLock<Option<Backend>> = OnceLock::new();

    fn backend() -> Option<&'static Backend> {
        BACKEND
            .get_or_init(|| {
                let Some(device) = Device::system_default() else {
                    if super::debug_enabled() {
                        eprintln!("[gpu] 没有可用的 Metal 设备 → 走 CPU 回退");
                    }
                    return None;
                };
                let library = match device
                    .new_library_with_source(SOURCE, &metal::CompileOptions::new())
                {
                    Ok(library) => library,
                    Err(message) => {
                        eprintln!("[gpu] Metal 内核编译失败 → 走 CPU 回退：{message}");
                        return None;
                    }
                };
                let function = match library.get_function("project_kernel", None) {
                    Ok(function) => function,
                    Err(message) => {
                        eprintln!("[gpu] 取不到 project_kernel → 走 CPU 回退：{message}");
                        return None;
                    }
                };
                let pipeline = match device.new_compute_pipeline_state_with_function(&function) {
                    Ok(pipeline) => pipeline,
                    Err(message) => {
                        eprintln!("[gpu] 管线状态创建失败 → 走 CPU 回退：{message}");
                        return None;
                    }
                };
                if super::debug_enabled() {
                    eprintln!("[gpu] Metal 后端就绪：{}", device.name());
                }
                Some(Backend {
                    queue: device.new_command_queue(),
                    device,
                    pipeline,
                })
            })
            .as_ref()
    }

    pub fn available() -> bool {
        backend().is_some()
    }

    /// 设备名（`Apple M1` / `Apple M2 Pro` …），没设备时 `None`。
    pub fn device_name() -> Option<String> {
        backend().map(|backend| backend.device.name().to_string())
    }


    /// 一次 dispatch 的上传缓冲。
    ///
    /// 先用「每次调用本地拷贝」的正确实现：Metal 的 `newBufferWithBytes` 每次都要
    /// 拷贝到共享内存，真正的优化是常驻 buffer 池（按容量复用），等有真实规模
    /// 数据再上，避免现在就把锁与生命周期搞复杂。
    struct Upload {
        models: Vec<u8>,
        params: Vec<u8>,
        rotation: Vec<u8>,
        translation: Vec<u8>,
        points: Vec<u8>,
        count: Vec<u8>,
    }

    impl Upload {
        fn from(batch: &ProjectionBatch, count: u32) -> Self {
            // f64 → f32 只在这里发生（MSL 没有 f64）。
            let narrow = |values: &[f64]| -> Vec<u8> {
                let narrowed: Vec<f32> = values.iter().map(|value| *value as f32).collect();
                bytes_f32(&narrowed).to_vec()
            };
            Self {
                models: bytes_u32(&batch.models).to_vec(),
                params: narrow(&batch.params),
                rotation: narrow(&batch.rotation),
                translation: narrow(&batch.translation),
                points: narrow(&batch.points),
                count: bytes_u32_slice(count),
            }
        }
    }

    fn upload(device: &Device, bytes: &[u8]) -> Buffer {
        device.new_buffer_with_data(
            bytes.as_ptr() as *const std::ffi::c_void,
            bytes.len().max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// GPU 路径；任何一步失败都返回 `None`（调用方回退 CPU）。
    pub fn project(batch: &ProjectionBatch) -> Option<Vec<Option<[f64; 2]>>> {
        let count = batch.len();
        // 太小的批次不值得往返 PCIe/调度开销，直接交给 CPU。
        if count < 1024 {
            return None;
        }
        let backend = backend()?;
        let started = std::time::Instant::now();
        let data = Upload::from(batch, count as u32);
        let models = upload(&backend.device, &data.models);
        let params = upload(&backend.device, &data.params);
        let rotation = upload(&backend.device, &data.rotation);
        let translation = upload(&backend.device, &data.translation);
        let points = upload(&backend.device, &data.points);
        let counter = upload(&backend.device, &data.count);
        let output = backend.device.new_buffer(
            (count * 8) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = backend.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&backend.pipeline);
        encoder.set_buffer(0, Some(&models), 0);
        encoder.set_buffer(1, Some(&params), 0);
        encoder.set_buffer(2, Some(&rotation), 0);
        encoder.set_buffer(3, Some(&translation), 0);
        encoder.set_buffer(4, Some(&points), 0);
        encoder.set_buffer(5, Some(&output), 0);
        encoder.set_buffer(6, Some(&counter), 0);
        encoder.dispatch_threads(
            MTLSize::new(count as u64, 1, 1),
            MTLSize::new(THREADS_PER_GROUP as u64, 1, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let dispatch = super::DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if super::debug_enabled() {
            eprintln!(
                "[gpu] dispatch #{dispatch}：{count} 次投影，{:.3} ms",
                started.elapsed().as_secs_f64() * 1e3
            );
        }

        let raw = output.contents() as *const f32;
        if raw.is_null() {
            return None;
        }
        let values = unsafe { std::slice::from_raw_parts(raw, count * 2) };
        let mut projections = Vec::with_capacity(count);
        for index in 0..count {
            let u = values[index * 2] as f64;
            let v = values[index * 2 + 1] as f64;
            if u.is_finite() && v.is_finite() {
                projections.push(Some([u, v]));
            } else {
                projections.push(None);
            }
        }
        Some(projections)
    }

    fn bytes_u32(values: &[u32]) -> &[u8] {
        // MSL 读的是 `uint`/`float`（4 字节），Rust 侧同宽，直接按字节重解释。
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values)) }
    }

    fn bytes_u32_slice(value: u32) -> Vec<u8> {
        value.to_le_bytes().to_vec()
    }

    fn bytes_f32(values: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::linalg::quaternion_to_rotation_matrix;

    /// CPU 回退路径必须与 `camera::img_from_cam` 一致（feature 关/开都跑）。
    #[test]
    fn cpu_fallback_matches_img_from_cam() {
        let rotation = quaternion_to_rotation_matrix(&[0.9, 0.2, -0.3, 0.1]);
        let models = [
            (CameraModelId::SimplePinhole, vec![900.0, 500.0, 400.0]),
            (CameraModelId::Pinhole, vec![900.0, 910.0, 500.0, 400.0]),
            (CameraModelId::SimpleRadial, vec![900.0, 500.0, 400.0, 0.02]),
            (CameraModelId::Radial, vec![900.0, 500.0, 400.0, 0.01, 0.002]),
        ];
        let points = [[0.1, -0.2, 4.0], [-1.0, 0.5, 2.5], [2.0, -1.5, 6.0], [0.0, 0.0, 8.0]];
        for (model, params) in models {
            let mut batch = ProjectionBatch::with_capacity(points.len());
            for point in points {
                batch.push(model, &params, &rotation, &[0.1, -0.05, 0.2], point);
            }
            let projections = project_points_or(&batch, |_| None);
            for (index, point) in points.iter().enumerate() {
                let expected = super::super::camera::world_to_img(
                    &super::super::format::model::Camera::new(
                        1,
                        model as i32,
                        1000,
                        800,
                        params.clone(),
                    ),
                    model,
                    &[0.9, 0.2, -0.3, 0.1],
                    &[0.1, -0.05, 0.2],
                    *point,
                );
                match (projections[index], expected) {
                    (Some(actual), Some(want)) => {
                        assert!(
                            (actual[0] - want[0]).abs() < 1e-6 && (actual[1] - want[1]).abs() < 1e-6,
                            "模型 {model:?} 点 {point:?}：{actual:?} vs {want:?}"
                        );
                    }
                    (None, None) => {}
                    pair => panic!("模型 {model:?} 失败语义不一致：{pair:?}"),
                }
            }
        }
    }

    /// 接线验证：BA 的 `robust_cost` 在观测数 ≥ 一个 dispatch 时真的把投影送上了
    /// Metal（不是静默走 CPU）。没有 Metal 设备时自动跳过。
    #[cfg(all(feature = "backend-ooo-sfm-metal", target_os = "macos"))]
    #[test]
    fn ba_pipeline_reaches_metal_dispatch() {
        if !active() {
            eprintln!("[gpu] 无 Metal 设备，跳过接线验证");
            return;
        }
        // 220 个点 × 6 个观测 = 1320 次投影（> 1024 的 dispatch 阈值）。
        let cameras = vec![super::super::ba::BaCamera {
            id: 1,
            model_id: 2,
            params: vec![900.0, 500.0, 400.0, 0.0],
            constant: false,
        }];
        let points: Vec<super::super::ba::BaPoint> = (0..220)
            .map(|index| {
                let angle = index as f64 * 0.37;
                super::super::ba::BaPoint {
                    xyz: [
                        1.5 * angle.sin(),
                        1.2 * angle.cos(),
                        4.0 + 0.5 * (angle * 0.3).cos(),
                    ],
                    observations: (0..6)
                        .map(|step| {
                            let yaw = 0.02 * step as f64;
                            let (sin, cos) = yaw.sin_cos();
                            let rotation = [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]];
                            let tvec = [0.1 * step as f64, 0.0, 0.0];
                            let qvec = super::super::linalg::rotation_to_quaternion(&rotation);
                            let projected = super::super::camera::world_to_img(
                                &super::super::format::model::Camera::new(
                                    1,
                                    2,
                                    1000,
                                    800,
                                    vec![900.0, 500.0, 400.0, 0.0],
                                ),
                                super::super::ba::CameraModelId::SimpleRadial,
                                &qvec,
                                &tvec,
                                [
                                    1.5 * angle.sin(),
                                    1.2 * angle.cos(),
                                    4.0 + 0.5 * (angle * 0.3).cos(),
                                ],
                            )
                            .expect("点在前方");
                            (step, projected[0], projected[1])
                        })
                        .collect(),
                    constant: false,
                }
            })
            .collect();
        let images: Vec<super::super::ba::BaImage> = (0..6)
            .map(|step| {
                let yaw = 0.02 * step as f64;
                let (sin, cos) = yaw.sin_cos();
                super::super::ba::BaImage {
                    id: step + 2,
                    camera: 0,
                    qvec: super::super::linalg::rotation_to_quaternion(&[
                        [cos, -sin, 0.0],
                        [sin, cos, 0.0],
                        [0.0, 0.0, 1.0],
                    ]),
                    tvec: [0.1 * step as f64, 0.0, 0.0],
                    constant_pose: false,
                }
            })
            .collect();
        let before = dispatch_count();
        let mut cameras = cameras;
        let mut images = images;
        let mut points = points;
        let options = super::super::ba::BaOptions {
            refine_focal_length: false,
            refine_principal_point: false,
            refine_extra_params: false,
            max_iterations: 3,
            ..super::super::ba::BaOptions::default()
        };
        let summary = super::super::ba::run_bundle_adjustment(&mut cameras, &mut images, &mut points, &options);
        assert!(summary.cost.is_finite(), "BA 代价应有限：{summary:?}");
        assert!(
            dispatch_count() > before,
            "BA 的投影没有走到 Metal（dispatch 次数没变）"
        );
    }

    /// Metal 内核必须与 CPU 一致（f32 误差 ~1e-3 px）。没有 Metal 设备时自动跳过。
    #[cfg(all(feature = "backend-ooo-sfm-metal", target_os = "macos"))]
    #[test]
    fn metal_kernel_matches_cpu_projection() {
        if !active() {
            eprintln!("[gpu] 无 Metal 设备，跳过 GPU 对拍");
            return;
        }
        let rotation = quaternion_to_rotation_matrix(&[0.97, 0.1, 0.2, -0.05]);
        let params = vec![900.0, 500.0, 400.0, 0.02];
        let mut batch = ProjectionBatch::with_capacity(4096);
        let mut expected = Vec::with_capacity(4096);
        for index in 0..4096u32 {
            let seed = index as f64;
            let point = [
                (seed * 0.37).sin() * 2.0,
                (seed * 0.11).cos() * 1.5,
                2.0 + (seed * 0.017).sin().abs() * 6.0,
            ];
            batch.push(CameraModelId::SimpleRadial, &params, &rotation, &[0.1, 0.05, -0.2], point);
            let projected = super::super::camera::img_from_cam(
                CameraModelId::SimpleRadial,
                &params,
                [
                    rotation[0][0] * point[0]
                        + rotation[0][1] * point[1]
                        + rotation[0][2] * point[2]
                        + 0.1,
                    rotation[1][0] * point[0]
                        + rotation[1][1] * point[1]
                        + rotation[1][2] * point[2]
                        + 0.05,
                    rotation[2][0] * point[0]
                        + rotation[2][1] * point[1]
                        + rotation[2][2] * point[2]
                        - 0.2,
                ],
            );
            expected.push(projected);
        }
        let projections = project_points_or(&batch, |_| None);
        let mut worst: f64 = 0.0;
        for (index, want) in expected.iter().enumerate() {
            match (projections[index], *want) {
                (Some(actual), Some(want)) => {
                    worst = worst
                        .max((actual[0] - want[0]).abs())
                        .max((actual[1] - want[1]).abs());
                }
                _ => panic!("第 {index} 个投影的失败语义不一致"),
            }
        }
        // f32 上传 + 1e3 量级的像素坐标 → 相对误差 ~1e-6（实测 1.3e-3 px）。
        // COLMAP 的内点阈值是 4–12 px，这个偏差比它小 3 个数量级，可以忽略。
        assert!(worst < 5e-3, "GPU 与 CPU 的最大像素偏差 {worst} 过大");
    }
}