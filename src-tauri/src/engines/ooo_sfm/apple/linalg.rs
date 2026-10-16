//! 自研 SfM 的共享几何助手（M5/M6）。
//!
//! 只依赖 `std`：四元数 ↔ 旋转矩阵、旋转矢量（SO(3) exp/log）、刚体变换
//! （`Rigid3d`，四元数顺序与本项目其余部分一致为 `w,x,y,z`）。

/// 刚体变换（`cam_from_world` 语义：世界点 → 相机系）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rigid3d {
    /// `(w, x, y, z)`，单位四元数。
    pub qvec: [f64; 4],
    pub tvec: [f64; 3],
}

impl Rigid3d {
    pub const IDENTITY: Rigid3d = Rigid3d {
        qvec: [1.0, 0.0, 0.0, 0.0],
        tvec: [0.0, 0.0, 0.0],
    };

    pub fn new(qvec: [f64; 4], tvec: [f64; 3]) -> Self {
        Self { qvec, tvec }
    }

    pub fn from_rotation_matrix(rotation: &[[f64; 3]; 3], tvec: [f64; 3]) -> Self {
        Self {
            qvec: rotation_to_quaternion(rotation),
            tvec,
        }
    }

    /// 旋转矩阵 `R = R(qvec)`。
    pub fn rotation_matrix(&self) -> [[f64; 3]; 3] {
        quaternion_to_rotation_matrix(&self.qvec)
    }

    /// 变换点：`R·X + t`。
    pub fn apply(&self, point: [f64; 3]) -> [f64; 3] {
        let rotation = self.rotation_matrix();
        [
            rotation[0][0] * point[0] + rotation[0][1] * point[1] + rotation[0][2] * point[2]
                + self.tvec[0],
            rotation[1][0] * point[0] + rotation[1][1] * point[1] + rotation[1][2] * point[2]
                + self.tvec[1],
            rotation[2][0] * point[0] + rotation[2][1] * point[1] + rotation[2][2] * point[2]
                + self.tvec[2],
        ]
    }

    /// 投影中心（世界系）：`C = −Rᵀ·t`。
    pub fn projection_center(&self) -> [f64; 3] {
        let inverse_rotation = transpose3(&self.rotation_matrix());
        [
            -(inverse_rotation[0][0] * self.tvec[0]
                + inverse_rotation[0][1] * self.tvec[1]
                + inverse_rotation[0][2] * self.tvec[2]),
            -(inverse_rotation[1][0] * self.tvec[0]
                + inverse_rotation[1][1] * self.tvec[1]
                + inverse_rotation[1][2] * self.tvec[2]),
            -(inverse_rotation[2][0] * self.tvec[0]
                + inverse_rotation[2][1] * self.tvec[1]
                + inverse_rotation[2][2] * self.tvec[2]),
        ]
    }

    /// `self⁻¹`：`q⁻¹` + `−Rᵀ·t`。
    pub fn inverse(&self) -> Rigid3d {
        let rotation = self.rotation_matrix();
        let inverse_rotation = transpose3(&rotation);
        Rigid3d {
            qvec: [
                self.qvec[0],
                -self.qvec[1],
                -self.qvec[2],
                -self.qvec[3],
            ],
            tvec: [
                -(inverse_rotation[0][0] * self.tvec[0]
                    + inverse_rotation[0][1] * self.tvec[1]
                    + inverse_rotation[0][2] * self.tvec[2]),
                -(inverse_rotation[1][0] * self.tvec[0]
                    + inverse_rotation[1][1] * self.tvec[1]
                    + inverse_rotation[1][2] * self.tvec[2]),
                -(inverse_rotation[2][0] * self.tvec[0]
                    + inverse_rotation[2][1] * self.tvec[1]
                    + inverse_rotation[2][2] * self.tvec[2]),
            ],
        }
    }

    /// `self * other`（先 `other` 后 `self`）。
    pub fn multiply(&self, other: &Rigid3d) -> Rigid3d {
        let rotation = mat3_mat3(&self.rotation_matrix(), &other.rotation_matrix());
        Rigid3d {
            qvec: rotation_to_quaternion(&rotation),
            tvec: self.apply(other.tvec),
        }
    }

    /// 归一化四元数（返回新值）。
    pub fn normalized(&self) -> Rigid3d {
        Rigid3d {
            qvec: normalize_quaternion(&self.qvec),
            tvec: self.tvec,
        }
    }
}

// ---------------------------------------------------------------- 四元数

/// 归一化（零向量时退回单位四元数；w 拉成非负保证双覆盖语义稳定）。
pub fn normalize_quaternion(qvec: &[f64; 4]) -> [f64; 4] {
    let norm = (qvec[0] * qvec[0]
        + qvec[1] * qvec[1]
        + qvec[2] * qvec[2]
        + qvec[3] * qvec[3])
        .sqrt();
    if norm < 1e-15 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let mut out = [
        qvec[0] / norm,
        qvec[1] / norm,
        qvec[2] / norm,
        qvec[3] / norm,
    ];
    if out[0] < 0.0 {
        out = [-out[0], -out[1], -out[2], -out[3]];
    }
    out
}

/// `R = R(qvec)`（`w,x,y,z`）。
pub fn quaternion_to_rotation_matrix(qvec: &[f64; 4]) -> [[f64; 3]; 3] {
    let (w, x, y, z) = (qvec[0], qvec[1], qvec[2], qvec[3]);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// `qvec = R(q)`（Shepperd 数值稳定分支）。
pub fn rotation_to_quaternion(rotation: &[[f64; 3]; 3]) -> [f64; 4] {
    let m = rotation;
    let trace = m[0][0] + m[1][1] + m[2][2];
    let (w, x, y, z);
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        w = 0.25 * s;
        x = (m[2][1] - m[1][2]) / s;
        y = (m[0][2] - m[2][0]) / s;
        z = (m[1][0] - m[0][1]) / s;
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        w = (m[2][1] - m[1][2]) / s;
        x = 0.25 * s;
        y = (m[0][1] + m[1][0]) / s;
        z = (m[0][2] + m[2][0]) / s;
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        w = (m[0][2] - m[2][0]) / s;
        x = (m[0][1] + m[1][0]) / s;
        y = 0.25 * s;
        z = (m[1][2] + m[2][1]) / s;
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        w = (m[1][0] - m[0][1]) / s;
        x = (m[0][2] + m[2][0]) / s;
        y = (m[1][2] + m[2][1]) / s;
        z = 0.25 * s;
    }
    normalize_quaternion(&[w, x, y, z])
}

// ---------------------------------------------------------------- SO(3) exp/log

/// 旋转矢量 → 旋转矩阵（Rodrigues，零角用泰勒展开保精度）。
pub fn exp_rotation_vector(rotvec: &[f64; 3]) -> [[f64; 3]; 3] {
    let theta2 = rotvec[0] * rotvec[0] + rotvec[1] * rotvec[1] + rotvec[2] * rotvec[2];
    if theta2 < 1e-16 {
        // θ ≈ 0：R ≈ I + [ω]× + ½[ω]×²
        let (wx, wy, wz) = (rotvec[0], rotvec[1], rotvec[2]);
        let half = 0.5;
        return [
            [1.0 - half * (wy * wy + wz * wz), -wz + half * wx * wy, wy + half * wx * wz],
            [wz + half * wx * wy, 1.0 - half * (wx * wx + wz * wz), -wx + half * wy * wz],
            [-wy + half * wx * wz, wx + half * wy * wz, 1.0 - half * (wx * wx + wy * wy)],
        ];
    }
    let theta = theta2.sqrt();
    let axis = [rotvec[0] / theta, rotvec[1] / theta, rotvec[2] / theta];
    let (sin, cos) = theta.sin_cos();
    let k = 1.0 - cos;
    let (ax, ay, az) = (axis[0], axis[1], axis[2]);
    [
        [ax * ax * k + cos, ax * ay * k - az * sin, ax * az * k + ay * sin],
        [ay * ax * k + az * sin, ay * ay * k + cos, ay * az * k - ax * sin],
        [az * ax * k - ay * sin, az * ay * k + ax * sin, az * az * k + cos],
    ]
}

/// 旋转矩阵 → 旋转矢量（SO(3) log，角度域 [0, π]；病态输入返回近似值）。
pub fn log_rotation_matrix(rotation: &[[f64; 3]; 3]) -> [f64; 3] {
    let cos_theta = ((rotation[0][0] + rotation[1][1] + rotation[2][2]) - 1.0) / 2.0;
    let cos_theta = cos_theta.clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    if theta < 1e-7 {
        // 小角：R ≈ I + [ω]×，ω_i = (R[j][k] − R[k][j]) / 2
        return [
            0.5 * (rotation[2][1] - rotation[1][2]),
            0.5 * (rotation[0][2] - rotation[2][0]),
            0.5 * (rotation[1][0] - rotation[0][1]),
        ];
    }
    if theta > std::f64::consts::PI - 1e-4 {
        // θ ≈ π：由对角元恢复轴长、非对角元定符号（最大分量取正基准）。
        let mut axis = [
            ((rotation[0][0] + 1.0) / 2.0).max(0.0).sqrt(),
            ((rotation[1][1] + 1.0) / 2.0).max(0.0).sqrt(),
            ((rotation[2][2] + 1.0) / 2.0).max(0.0).sqrt(),
        ];
        if rotation[2][1] < 0.0 {
            axis[1] = -axis[1];
        }
        if rotation[0][2] < 0.0 {
            axis[2] = -axis[2];
        }
        if rotation[0][1] * axis[0] * axis[1] < 0.0 {
            axis[0] = -axis[0];
        }
        return [axis[0] * theta, axis[1] * theta, axis[2] * theta];
    }
    let factor = theta / (2.0 * theta.sin());
    [
        factor * (rotation[2][1] - rotation[1][2]),
        factor * (rotation[0][2] - rotation[2][0]),
        factor * (rotation[1][0] - rotation[0][1]),
    ]
}

/// 四元数 → 旋转矢量。
pub fn quaternion_to_rotation_vector(qvec: &[f64; 4]) -> [f64; 3] {
    log_rotation_matrix(&quaternion_to_rotation_matrix(&normalize_quaternion(qvec)))
}

/// 旋转矢量 → 四元数。
pub fn rotation_vector_to_quaternion(rotvec: &[f64; 3]) -> [f64; 4] {
    rotation_to_quaternion(&exp_rotation_vector(rotvec))
}

// ---------------------------------------------------------------- 小工具

/// `3×3` 矩阵乘向量。
pub fn mat3_vec3(matrix: &[[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

/// `3×3` 矩阵乘 `3×3` 矩阵。
pub fn mat3_mat3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (row, out_row) in out.iter_mut().enumerate() {
        for col in 0..3 {
            out_row[col] =
                a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col];
        }
    }
    out
}

/// 矩阵转置。
pub fn transpose3(matrix: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn quaternion_roundtrip_through_rotation_vector() {
        let cases = [
            [1.0, 0.0, 0.0, 0.0],
            [0.9659, 0.1494, -0.1494, 0.1494], // ~25°
            [0.7071, 0.0, 0.7071, 0.0],        // 90° 绕 y
            [0.0, 0.5773, 0.5773, 0.5773],     // 180°
        ];
        for qvec in cases {
            let rotvec = quaternion_to_rotation_vector(&qvec);
            let back = rotation_vector_to_quaternion(&rotvec);
            // 双覆盖：比较旋转矩阵而非四元数分量。
            let original = quaternion_to_rotation_matrix(&qvec);
            let returned = quaternion_to_rotation_matrix(&back);
            for i in 0..3 {
                for j in 0..3 {
                    // 手写四元数未精确归一化，容差放宽到 1e-3。
                    assert!(
                        (original[i][j] - returned[i][j]).abs() < 1e-3,
                        "q={qvec:?} back={back:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn exp_log_roundtrip_random_rotations() {
        let rotvecs = [
            [0.1, -0.2, 0.3],
            [1.2, 0.4, -0.8],
            [2.9, 0.01, -0.03], // 接近 π
            [0.0, 0.0, 0.0],
        ];
        for rotvec in rotvecs {
            let rotation = exp_rotation_vector(&rotvec);
            let back = log_rotation_matrix(&rotation);
            for i in 0..3 {
                assert!((back[i] - rotvec[i]).abs() < 1e-6, "{rotvec:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn rigid3d_inverse_and_compose() {
        let pose = Rigid3d::new(
            normalize_quaternion(&[0.8, 0.2, -0.4, 0.4]),
            [1.5, -2.0, 3.0],
        );
        let identity = pose.multiply(&pose.inverse());
        for i in 0..4 {
            assert!(approx(identity.qvec[i], Rigid3d::IDENTITY.qvec[i], 1e-9));
        }
        for i in 0..3 {
            assert!(identity.tvec[i].abs() < 1e-9);
        }

        let point = [0.3, -1.2, 2.5];
        let to_camera = pose.apply(point);
        let to_world = pose.inverse().apply(to_camera);
        for i in 0..3 {
            assert!((to_world[i] - point[i]).abs() < 1e-9);
        }

        let center = pose.projection_center();
        // 投影中心变换到相机系后应为 0。
        let center_in_camera = pose.apply(center);
        for value in center_in_camera {
            assert!(value.abs() < 1e-9);
        }
    }

    #[test]
    fn rotation_matrix_matches_quaternion_formula() {
        let qvec = normalize_quaternion(&[0.5, -0.5, 0.5, 0.5]);
        let rotation = quaternion_to_rotation_matrix(&qvec);
        // Rᵀ·R = I
        let transposed = transpose3(&rotation);
        let product = mat3_mat3(&transposed, &rotation);
        for row in 0..3 {
            for col in 0..3 {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!((product[row][col] - expected).abs() < 1e-9);
            }
        }
        assert!(approx(
            rotation_to_quaternion(&rotation)[0],
            qvec[0],
            1e-9
        ));
    }
}

#[cfg(test)]
mod quaternion_stress_tests {
    use super::*;

    #[test]
    fn quaternion_roundtrip_trace_zero_rotations() {
        // trace = 0 / 负 trace 的旋转（Shepperd 分支全覆盖）。
        let cases = [
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        ];
        for rotation in cases {
            let qvec = rotation_to_quaternion(&rotation);
            let back = quaternion_to_rotation_matrix(&qvec);
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        (back[i][j] - rotation[i][j]).abs() < 1e-9,
                        "R={rotation:?} q={qvec:?} back={back:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn quaternion_roundtrip_random_perturbations() {
        let base = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]];
        for k in 0..200 {
            let t = k as f64 * 0.031;
            let rotvec = [0.01 * t.sin(), 0.01 * t.cos(), 0.008 * (2.0 * t).sin()];
            let rotation = mat3_mat3(&exp_rotation_vector(&rotvec), &base);
            let qvec = rotation_to_quaternion(&rotation);
            let back = quaternion_to_rotation_matrix(&qvec);
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        (back[i][j] - rotation[i][j]).abs() < 1e-9,
                        "k={k} R={rotation:?} q={qvec:?} back={back:?}"
                    );
                }
            }
        }
    }
}
