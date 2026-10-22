//! 相机模型工具（M5）：投影 / 反投影 / 标定矩阵 / 退化参数判定。
//!
//! 语义对齐 COLMAP 4.2.1：`sensor/models.h`（`ImgFromCam`、`CamFromImg`、
//! `CamRayFromImg`）与 `scene/projection.cc`（重投影误差）。SIMPLE_RADIAL 是
//! 本项目唯一会用到的模型，但 PINHOLE / SIMPLE_PINHOLE / RADIAL 一并支持，
//! 与 M6 的 BA 引擎（`super::ba::CameraModelId`）共用同一套模型编号。

use super::ba::{cam_ray_from_image, CameraModelId};
use super::format::model::Camera;
use super::linalg::quaternion_to_rotation_matrix;

/// 4.2.1 `models.h:279` 的 `kMinDepth`（投影深度下限）。
const MIN_DEPTH: f64 = 1e-6;

/// `f, cx, cy[, k…]` / `[fx, fy, cx, cy, …]` 拆包。
pub fn intrinsics(model: CameraModelId, params: &[f64]) -> (f64, f64, f64, f64) {
    match model {
        CameraModelId::SimplePinhole | CameraModelId::SimpleRadial | CameraModelId::Radial => {
            (params[0], params[0], params[1], params[2])
        }
        CameraModelId::Pinhole => (params[0], params[1], params[2], params[3]),
    }
}

/// 标定矩阵 `K`（4.2.1 `Camera::CalibrationMatrix`；SIMPLE_RADIAL 的 k 不进 K）。
pub fn calibration_matrix(camera: &Camera) -> [[f64; 3]; 3] {
    let model = CameraModelId::from_id(camera.model_id).expect("相机模型已校验");
    let (fx, fy, cx, cy) = intrinsics(model, &camera.params);
    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
}

/// 相机系点 → 像素（4.2.1 `CameraModelImgFromCam`；深度不足返回 `None`）。
pub fn img_from_cam(model: CameraModelId, params: &[f64], point: [f64; 3]) -> Option<[f64; 2]> {
    if point[2] < MIN_DEPTH {
        return None;
    }
    let (fx, fy, cx, cy) = intrinsics(model, params);
    let x = point[0] / point[2];
    let y = point[1] / point[2];
    let extra = model.extra_idxs();
    let mut dx = 0.0;
    let mut dy = 0.0;
    if extra.is_empty() {
        // 无畸变。
    } else if extra.len() == 1 {
        let r2 = x * x + y * y;
        dx = x * params[extra[0]] * r2;
        dy = y * params[extra[0]] * r2;
    } else {
        let r2 = x * x + y * y;
        let (k1, k2) = (params[extra[0]], params[extra[1]]);
        dx = x * k1 * r2 + x * k2 * r2 * r2;
        dy = y * k1 * r2 + y * k2 * r2 * r2;
    }
    Some([fx * (x + dx) + cx, fy * (y + dy) + cy])
}

/// 像素 → 相机系点（单位深度；畸变用牛顿迭代反解，4.2.1 `IterativeUndistortion`）。
pub fn cam_point_from_img(
    model: CameraModelId,
    params: &[f64],
    x: f64,
    y: f64,
) -> Option<[f64; 3]> {
    cam_ray_from_image(model, params, x, y)
}

/// 世界点 → 像素（`cam_from_world` 为 `qvec w,x,y,z` + `tvec`）。
pub fn world_to_img(
    camera: &Camera,
    model: CameraModelId,
    qvec: &[f64; 4],
    tvec: &[f64; 3],
    xyz: [f64; 3],
) -> Option<[f64; 2]> {
    let rotation = quaternion_to_rotation_matrix(qvec);
    let point = [
        rotation[0][0] * xyz[0] + rotation[0][1] * xyz[1] + rotation[0][2] * xyz[2] + tvec[0],
        rotation[1][0] * xyz[0] + rotation[1][1] * xyz[1] + rotation[1][2] * xyz[2] + tvec[1],
        rotation[2][0] * xyz[0] + rotation[2][1] * xyz[1] + rotation[2][2] * xyz[2] + tvec[2],
    ];
    img_from_cam(model, &camera.params, point)
}

/// 像素重投影误差（欧氏距离，4.2.1 `CalculateSquaredReprojectionError` 的开方形式）。
pub fn reprojection_error(
    camera: &Camera,
    model: CameraModelId,
    qvec: &[f64; 4],
    tvec: &[f64; 3],
    xyz: [f64; 3],
    observation: [f64; 2],
) -> f64 {
    match world_to_img(camera, model, qvec, tvec, xyz) {
        Some(projection) => {
            let dx = projection[0] - observation[0];
            let dy = projection[1] - observation[1];
            (dx * dx + dy * dy).sqrt()
        }
        // 4.2.1 投影失败时返回 `double::max()`（等价无穷大误差）。
        None => f64::MAX,
    }
}

/// 投影中心（世界系，`C = −Rᵀt`；4.2.1 `Image::ProjectionCenter`）。
pub fn projection_center(qvec: &[f64; 4], tvec: &[f64; 3]) -> [f64; 3] {
    let rotation = quaternion_to_rotation_matrix(qvec);
    [
        -(rotation[0][0] * tvec[0] + rotation[1][0] * tvec[1] + rotation[2][0] * tvec[2]),
        -(rotation[0][1] * tvec[0] + rotation[1][1] * tvec[1] + rotation[2][1] * tvec[2]),
        -(rotation[0][2] * tvec[0] + rotation[1][2] * tvec[1] + rotation[2][2] * tvec[2]),
    ]
}

/// `cam_from_world` 作用到世界点（相机系坐标）。
pub fn cam_from_world_point(qvec: &[f64; 4], tvec: &[f64; 3], xyz: [f64; 3]) -> [f64; 3] {
    let rotation = quaternion_to_rotation_matrix(qvec);
    [
        rotation[0][0] * xyz[0] + rotation[0][1] * xyz[1] + rotation[0][2] * xyz[2] + tvec[0],
        rotation[1][0] * xyz[0] + rotation[1][1] * xyz[1] + rotation[1][2] * xyz[2] + tvec[1],
        rotation[2][0] * xyz[0] + rotation[2][1] * xyz[1] + rotation[2][2] * xyz[2] + tvec[2],
    ]
}

/// 4.2.1 `Camera::HasBogusParams`：焦距比 `[min, max]` 越界或畸变过大。
pub fn has_bogus_params(
    camera: &Camera,
    min_focal_length_ratio: f64,
    max_focal_length_ratio: f64,
    max_extra_param: f64,
) -> bool {
    let model = CameraModelId::from_id(camera.model_id).expect("相机模型已校验");
    let max_size = camera.width.max(camera.height).max(1) as f64;
    for index in model.focal_idxs() {
        let focal = camera.params[*index];
        if focal < min_focal_length_ratio * max_size || focal > max_focal_length_ratio * max_size {
            return true;
        }
    }
    for index in model.extra_idxs() {
        if camera.params[*index].abs() > max_extra_param {
            return true;
        }
    }
    false
}

/// 相机模型枚举的辅助（供 mapper 判定 SIMPLE_RADIAL 等）。
pub fn model_of(camera: &Camera) -> CameraModelId {
    CameraModelId::from_id(camera.model_id).expect("相机模型已校验")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera(model_id: i32, width: u64, height: u64, params: &[f64]) -> Camera {
        Camera::new(1, model_id, width, height, params.to_vec())
    }

    #[test]
    fn projection_round_trips_through_rays() {
        let cameras = [
            camera(2, 1000, 800, &[800.0, 500.0, 400.0, 0.01]),
            camera(1, 1000, 800, &[800.0, 810.0, 500.0, 400.0]),
        ];
        for camera in cameras {
            let model = model_of(&camera);
            for point in [[0.2, -0.3, 4.0], [-1.0, 0.5, 6.0]] {
                let pixel = img_from_cam(model, &camera.params, point).unwrap();
                let ray = cam_point_from_img(model, &camera.params, pixel[0], pixel[1]).unwrap();
                // `cam_ray_from_image` 返回 z = 1 的归一化光线（不是单位长度）。
                let scale = point[2] / ray[2];
                for axis in 0..3 {
                    assert!((ray[axis] * scale - point[axis]).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn negative_depth_is_not_projectable() {
        let camera = camera(2, 1000, 800, &[800.0, 500.0, 400.0, 0.0]);
        let model = model_of(&camera);
        assert!(img_from_cam(model, &camera.params, [0.0, 0.0, -1.0]).is_none());
        assert_eq!(
            reprojection_error(
                &camera,
                model,
                &[1.0, 0.0, 0.0, 0.0],
                &[0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [10.0, 10.0],
            ),
            f64::MAX
        );
    }

    #[test]
    fn projection_center_matches_rigid_transform() {
        // 位姿：绕 y 轴 90°，平移 (1, 2, 3)。
        let qvec = super::super::linalg::rotation_vector_to_quaternion(&[
            0.0,
            std::f64::consts::FRAC_PI_2,
            0.0,
        ]);
        let center = projection_center(&qvec, &[1.0, 2.0, 3.0]);
        let point = cam_from_world_point(&qvec, &[1.0, 2.0, 3.0], center);
        for value in point {
            assert!(value.abs() < 1e-12);
        }
    }

    #[test]
    fn bogus_params_detect_out_of_range_focal_and_distortion() {
        let sane = camera(2, 1000, 800, &[1200.0, 500.0, 400.0, 0.05]);
        assert!(!has_bogus_params(&sane, 0.1, 10.0, 1.0));
        let too_short = camera(2, 1000, 800, &[50.0, 500.0, 400.0, 0.0]);
        assert!(has_bogus_params(&too_short, 0.1, 10.0, 1.0));
        let too_wide = camera(2, 1000, 800, &[20000.0, 500.0, 400.0, 0.0]);
        assert!(has_bogus_params(&too_wide, 0.1, 10.0, 1.0));
        let too_distorted = camera(2, 1000, 800, &[1200.0, 500.0, 400.0, 1.5]);
        assert!(has_bogus_params(&too_distorted, 0.1, 10.0, 1.0));
    }

    #[test]
    fn calibration_matrix_matches_intrinsics() {
        let camera = camera(2, 1000, 800, &[800.0, 500.0, 400.0, 0.02]);
        let k = calibration_matrix(&camera);
        assert_eq!(k[0][0], 800.0);
        assert_eq!(k[1][1], 800.0);
        assert_eq!(k[0][2], 500.0);
        assert_eq!(k[1][2], 400.0);
        assert_eq!(k[2], [0.0, 0.0, 1.0]);
    }
}