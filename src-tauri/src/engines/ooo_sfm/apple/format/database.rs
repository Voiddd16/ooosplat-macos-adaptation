//! `database.db`（SQLite）的读写。
//!
//! schema 与 blob 布局逐字对齐 COLMAP 4.2.1
//! （`src/colmap/scene/database_sqlite.cc`），包括 `PRAGMA user_version`，
//! 这样产物既能被本项目读取，也能被其他平台上的 COLMAP 继续使用（方案 D3）。
//!
//! 列/blob 关键约定：
//! - `keypoints.data`：`rows × 6` 的 `f32`（x, y, a11, a12, a21, a22）。
//! - `descriptors.data`：`rows × 128` 的 `u8`；`type = 0` 表示 SIFT。
//! - `matches.data` / `two_view_geometries.data`：`rows × 2` 的 `u32`（特征点下标对）。
//! - `two_view_geometries` 的 F/E/H 按**行主序**存 9 个 `f64`
//!   （COLMAP 侧是 `M.transpose()` 再按列主序写，等价于行主序）。
//! - `pair_id = kMaxNumImages * min(id1, id2) + max(id1, id2)`。

use std::path::Path;

use rusqlite::{Connection, OpenFlags, OptionalExtension};

use super::{
    camera_model, model::Camera, DATABASE_VERSION, DESCRIPTOR_COLS, DESCRIPTOR_TYPE_SIFT,
    MAX_NUM_IMAGES,
};
use crate::error::{Result, SplatError};

fn sqlite(error: rusqlite::Error) -> SplatError {
    SplatError::Process(format!("COLMAP 数据库操作失败：{error}"))
}

/// `keypoints.data` 的一个特征点（`shape` = a11, a12, a21, a22）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    pub shape: [f32; 4],
}

impl Keypoint {
    /// 各向同性（无仿射形状）的特征点，形状矩阵取单位阵。
    pub fn upright(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            shape: [1.0, 0.0, 0.0, 1.0],
        }
    }
}

/// `two_view_geometries` 一行（`cam2_from_cam1` 的四元数顺序为 `w,x,y,z`）。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TwoViewGeometry {
    /// COLMAP `TwoViewGeometry::ConfigurationType`（`CALIBRATED = 1` 等）。
    pub config: i32,
    /// 几何验证后的内点匹配（特征点下标对）。
    pub inlier_matches: Vec<[u32; 2]>,
    pub f: Option<[[f64; 3]; 3]>,
    pub e: Option<[[f64; 3]; 3]>,
    pub h: Option<[[f64; 3]; 3]>,
    pub cam2_from_cam1: Option<([f64; 4], [f64; 3])>,
}

impl TwoViewGeometry {
    /// COLMAP `TwoViewGeometry::Invert()`：把几何关系交换成 `(id2, id1)` 方向。
    pub fn inverted(&self) -> Self {
        let transpose = |matrix: &[[f64; 3]; 3]| {
            let mut out = [[0.0; 3]; 3];
            for row in 0..3 {
                for col in 0..3 {
                    out[row][col] = matrix[col][row];
                }
            }
            out
        };
        // 3×3 伴随矩阵除以行列式。
        let invert_matrix = |m: &[[f64; 3]; 3]| -> Option<[[f64; 3]; 3]> {
            let cofactor = |row: usize, col: usize| {
                let rows: Vec<usize> = (0..3).filter(|&r| r != row).collect();
                let cols: Vec<usize> = (0..3).filter(|&c| c != col).collect();
                let minor = m[rows[0]][cols[0]] * m[rows[1]][cols[1]]
                    - m[rows[0]][cols[1]] * m[rows[1]][cols[0]];
                if (row + col) % 2 == 0 {
                    minor
                } else {
                    -minor
                }
            };
            let adjugate = |row: usize, col: usize| cofactor(col, row);
            let det = m[0][0] * cofactor(0, 0) + m[0][1] * cofactor(0, 1) + m[0][2] * cofactor(0, 2);
            if det.abs() < 1e-300 {
                return None;
            }
            let mut out = [[0.0; 3]; 3];
            for row in 0..3 {
                for col in 0..3 {
                    out[row][col] = adjugate(row, col) / det;
                }
            }
            Some(out)
        };
        // 位姿取逆：q⁻¹ = (w, -x, -y, -z)，R⁻¹ = R(q⁻¹)，t⁻¹ = -R⁻¹·t。
        let inverse_pose = |pose: &([f64; 4], [f64; 3])| -> ([f64; 4], [f64; 3]) {
            let (qvec, tvec) = pose;
            let (w, x, y, z) = (qvec[0], -qvec[1], -qvec[2], -qvec[3]);
            let rotate = |point: [f64; 3]| {
                let v = [point[0], point[1], point[2]];
                // v' = v + 2·w·(u×v) + 2·(u×(u×v))，u = (x, y, z)。
                let cross = |a: [f64; 3], b: [f64; 3]| {
                    [
                        a[1] * b[2] - a[2] * b[1],
                        a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0],
                    ]
                };
                let u = [x, y, z];
                let uv = cross(u, v);
                let uuv = cross(u, uv);
                [
                    v[0] + 2.0 * (w * uv[0] + uuv[0]),
                    v[1] + 2.0 * (w * uv[1] + uuv[1]),
                    v[2] + 2.0 * (w * uv[2] + uuv[2]),
                ]
            };
            let rotated = rotate(*tvec);
            (
                [w, x, y, z],
                [-rotated[0], -rotated[1], -rotated[2]],
            )
        };

        Self {
            config: self.config,
            inlier_matches: self
                .inlier_matches
                .iter()
                .map(|pair| [pair[1], pair[0]])
                .collect(),
            f: self.f.as_ref().map(transpose),
            e: self.e.as_ref().map(transpose),
            h: self.h.as_ref().and_then(invert_matrix),
            cam2_from_cam1: self.cam2_from_cam1.as_ref().map(inverse_pose),
        }
    }
}

/// `pair_id`：与 COLMAP `ImagePairToPairId` 一致（内部会排序，id 必须不等）。
pub fn pair_id(image_id1: u32, image_id2: u32) -> i64 {
    assert_ne!(image_id1, image_id2, "pair_id 需要两个不同图像");
    let (low, high) = if image_id1 < image_id2 {
        (image_id1, image_id2)
    } else {
        (image_id2, image_id1)
    };
    (MAX_NUM_IMAGES * u64::from(low) + u64::from(high)) as i64
}

/// `ImagePairToPairId` 的逆（`PairIdToImagePair`）。
pub fn pair_image_ids(pair: i64) -> (u32, u32) {
    let pair = pair as u64;
    (
        (pair / MAX_NUM_IMAGES) as u32,
        (pair % MAX_NUM_IMAGES) as u32,
    )
}

// ---------------------------------------------------------------- 生命周期

/// 新建（或覆盖）一个 COLMAP 兼容数据库。
pub fn create(database: &Path) -> Result<Connection> {
    if let Some(parent) = database.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // COLMAP 的 schema 不保证能覆盖旧文件中的残留定义（例如旧版 inlier_matches）。
    // 自研后端每次都是全量重建，直接删掉旧文件最干净。
    let _ = std::fs::remove_file(database);
    let connection = Connection::open(database).map_err(sqlite)?;
    create_schema(&connection)?;
    Ok(connection)
}

/// 只读打开（读取既有项目数据库用）。
pub fn open_read_only(database: &Path) -> Result<Connection> {
    Connection::open_with_flags(
        database,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(sqlite)
}

/// 打开（不存在则新建）数据库，等价 COLMAP `Database::Open`：
/// 既有内容保留（幂等续跑），缺失的 schema 补齐。
pub fn open(database: &Path) -> Result<Connection> {
    if let Some(parent) = database.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let connection = Connection::open(database).map_err(sqlite)?;
    create_schema(&connection)?;
    Ok(connection)
}

/// 建表 + 写 `user_version`。语句与 4.2.1 的 `CreateTables` 等价。
pub fn create_schema(connection: &Connection) -> Result<()> {
    connection
        .execute_batch(&format!(
            r#"
CREATE TABLE IF NOT EXISTS rigs
   (rig_id               INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,
    ref_sensor_id        INTEGER                             NOT NULL,
    ref_sensor_type      INTEGER                             NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS rig_ref_sensor_assignment ON
   rigs(ref_sensor_id, ref_sensor_type);
CREATE TABLE IF NOT EXISTS rig_sensors
   (rig_id               INTEGER                             NOT NULL,
    sensor_id            INTEGER                             NOT NULL,
    sensor_type          INTEGER                             NOT NULL,
    sensor_from_rig      BLOB,
FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE);
CREATE UNIQUE INDEX IF NOT EXISTS rig_sensor_assignment ON
   rig_sensors(sensor_id, sensor_type);
CREATE TABLE IF NOT EXISTS cameras
   (camera_id            INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,
    model                INTEGER                             NOT NULL,
    width                INTEGER                             NOT NULL,
    height               INTEGER                             NOT NULL,
    params               BLOB,
    prior_focal_length   INTEGER                             NOT NULL);
CREATE TABLE IF NOT EXISTS frames
   (frame_id             INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,
    rig_id               INTEGER                             NOT NULL,
    FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS frame_data
   (frame_id             INTEGER                             NOT NULL,
    data_id              INTEGER                             NOT NULL,
    sensor_id            INTEGER                             NOT NULL,
    sensor_type          INTEGER                             NOT NULL,
    FOREIGN KEY(frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE);
CREATE UNIQUE INDEX IF NOT EXISTS frame_sensor_assignment ON
   frame_data(data_id, sensor_type);
CREATE TABLE IF NOT EXISTS images
   (image_id   INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,
    name       TEXT                                NOT NULL UNIQUE,
    camera_id  INTEGER                             NOT NULL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {max_images}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));
CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name);
CREATE TABLE IF NOT EXISTS pose_priors
   (pose_prior_id              INTEGER  PRIMARY KEY  NOT NULL,
    corr_data_id               INTEGER               NOT NULL,
    corr_sensor_id             INTEGER               NOT NULL,
    corr_sensor_type           INTEGER               NOT NULL,
    position                   BLOB,
    position_covariance        BLOB,
    gravity                    BLOB,
    coordinate_system          INTEGER               NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS pose_prior_data_assignment ON
   pose_priors(corr_data_id, corr_sensor_id, corr_sensor_type);
CREATE TABLE IF NOT EXISTS keypoints
   (image_id  INTEGER  PRIMARY KEY  NOT NULL,
    rows      INTEGER               NOT NULL,
    cols      INTEGER               NOT NULL,
    data      BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS descriptors
   (image_id      INTEGER  PRIMARY KEY  NOT NULL,
    type          INTEGER               NOT NULL,
    rows          INTEGER               NOT NULL,
    cols          INTEGER               NOT NULL,
    data          BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS matches
   (pair_id  INTEGER  PRIMARY KEY  NOT NULL,
    rows     INTEGER               NOT NULL,
    cols     INTEGER               NOT NULL,
    data     BLOB);
CREATE TABLE IF NOT EXISTS two_view_geometries
   (pair_id  INTEGER  PRIMARY KEY  NOT NULL,
    rows     INTEGER               NOT NULL,
    cols     INTEGER               NOT NULL,
    data     BLOB,
    config   INTEGER               NOT NULL,
    F        BLOB,
    E        BLOB,
    H        BLOB,
    qvec     BLOB,
    tvec     BLOB,
    camera1  BLOB,
    camera2  BLOB);
"#,
            max_images = MAX_NUM_IMAGES
        ))
        .map_err(sqlite)?;
    // 迁移检查依赖这个版本号；写错会让 COLMAP 认为需要迁移甚至拒绝读取。
    connection
        .execute_batch(&format!("PRAGMA user_version = {DATABASE_VERSION};"))
        .map_err(sqlite)
}

/// 批量写入前调用（COLMAP 的 `Database::BeginTransaction` 等价）。
pub fn begin_bulk(connection: &Connection) -> Result<()> {
    connection.execute_batch("BEGIN").map_err(sqlite)
}

pub fn end_bulk(connection: &Connection) -> Result<()> {
    connection.execute_batch("COMMIT").map_err(sqlite)
}

// ---------------------------------------------------------------- 写入

pub fn write_camera(
    connection: &Connection,
    camera: &Camera,
    prior_focal_length: bool,
) -> Result<()> {
    if camera_model::num_params(camera.model_id) != Some(camera.params.len()) {
        return Err(SplatError::Process(format!(
            "相机 {} 使用了未知模型编号 {} 或参数个数不符",
            camera.id, camera.model_id
        )));
    }
    if u64::from(camera.id) >= MAX_NUM_IMAGES {
        return Err(SplatError::Process(format!(
            "相机编号 {} 超出 COLMAP 上限",
            camera.id
        )));
    }
    let mut params = Vec::with_capacity(camera.params.len() * 8);
    for param in &camera.params {
        params.extend_from_slice(&param.to_le_bytes());
    }
    connection
        .execute(
            "INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![
                camera.id,
                camera.model_id,
                camera.width as i64,
                camera.height as i64,
                params,
                prior_focal_length as i64,
            ],
        )
        .map_err(sqlite)?;
    Ok(())
}

pub fn write_image(connection: &Connection, image_id: u32, name: &str, camera_id: u32) -> Result<()> {
    if u64::from(image_id) >= MAX_NUM_IMAGES {
        return Err(SplatError::Process(format!(
            "图像编号 {image_id} 超出 COLMAP 上限"
        )));
    }
    connection
        .execute(
            "INSERT INTO images(image_id, name, camera_id) VALUES (?1, ?2, ?3)",
            rusqlite::params![image_id, name, camera_id],
        )
        .map_err(sqlite)?;
    Ok(())
}

/// 插入相机并返回数据库分配到的编号。
///
/// 对应 COLMAP `Database::WriteCamera(camera)`（`use_camera_id = false`）：
/// 特征提取新建相机时编号由数据库自增分配（首个相机通常是 1）。
pub fn insert_camera(
    connection: &Connection,
    model_id: i32,
    width: u64,
    height: u64,
    params: &[f64],
    prior_focal_length: bool,
) -> Result<u32> {
    if camera_model::num_params(model_id) != Some(params.len()) {
        return Err(SplatError::Process(format!(
            "相机使用了未知模型编号 {model_id} 或参数个数不符"
        )));
    }
    let mut blob = Vec::with_capacity(params.len() * 8);
    for param in params {
        blob.extend_from_slice(&param.to_le_bytes());
    }
    connection
        .execute(
            "INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length)
             VALUES (NULL, ?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                model_id,
                width as i64,
                height as i64,
                blob,
                prior_focal_length as i64,
            ],
        )
        .map_err(sqlite)?;
    let camera_id = connection.last_insert_rowid();
    if camera_id <= 0 || camera_id as u64 >= MAX_NUM_IMAGES {
        return Err(SplatError::Process(format!(
            "相机编号 {camera_id} 超出 COLMAP 上限"
        )));
    }
    Ok(camera_id as u32)
}

/// 插入图像并返回数据库分配到的编号（COLMAP `WriteImage(image)` 的自增路径）。
pub fn insert_image(connection: &Connection, name: &str, camera_id: u32) -> Result<u32> {
    connection
        .execute(
            "INSERT INTO images(image_id, name, camera_id) VALUES (NULL, ?1, ?2)",
            rusqlite::params![name, camera_id],
        )
        .map_err(sqlite)?;
    let image_id = connection.last_insert_rowid();
    if image_id <= 0 || image_id as u64 >= MAX_NUM_IMAGES {
        return Err(SplatError::Process(format!(
            "图像编号 {image_id} 超出 COLMAP 上限"
        )));
    }
    Ok(image_id as u32)
}

/// 按名字读图像（编号 + 相机编号），对应 COLMAP `Database::ReadImageWithName`。
pub fn read_image_by_name(connection: &Connection, name: &str) -> Result<Option<(u32, u32)>> {
    connection
        .query_row(
            "SELECT image_id, camera_id FROM images WHERE name = ?1",
            [name],
            |row| Ok((row.get::<_, i64>(0)? as u32, row.get::<_, i64>(1)? as u32)),
        )
        .optional()
        .map_err(sqlite)
}

/// 该图像是否已有键点与描述子（特征提取的"已处理则跳过"判定，
/// 对应 COLMAP `ExistsKeypoints && ExistsDescriptors`）。
pub fn has_keypoints_and_descriptors(connection: &Connection, image_id: u32) -> Result<bool> {
    let count: i64 = connection
        .query_row(
            "SELECT (EXISTS(SELECT 1 FROM keypoints WHERE image_id = ?1))
                  + (EXISTS(SELECT 1 FROM descriptors WHERE image_id = ?1))",
            [image_id],
            |row| row.get(0),
        )
        .map_err(sqlite)?;
    Ok(count == 2)
}

/// 删除某图像的特征（重新提取用；COLMAP 靠主键冲突暴露半成品状态，这里更稳）。
pub fn delete_features(connection: &Connection, image_id: u32) -> Result<()> {
    connection
        .execute("DELETE FROM keypoints WHERE image_id = ?1", [image_id])
        .map_err(sqlite)?;
    connection
        .execute("DELETE FROM descriptors WHERE image_id = ?1", [image_id])
        .map_err(sqlite)?;
    Ok(())
}

pub fn write_keypoints(connection: &Connection, image_id: u32, keypoints: &[Keypoint]) -> Result<()> {
    let mut data = Vec::with_capacity(keypoints.len() * 6 * 4);
    for keypoint in keypoints {
        for value in [
            keypoint.x,
            keypoint.y,
            keypoint.shape[0],
            keypoint.shape[1],
            keypoint.shape[2],
            keypoint.shape[3],
        ] {
            data.extend_from_slice(&value.to_le_bytes());
        }
    }
    connection
        .execute(
            "INSERT INTO keypoints(image_id, rows, cols, data) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![image_id, keypoints.len() as i64, 6_i64, data],
        )
        .map_err(sqlite)?;
    Ok(())
}

/// `descriptors` 表；`data` 是行主序的 `rows × 128` 的 `u8`。
pub fn write_descriptors(connection: &Connection, image_id: u32, data: &[u8]) -> Result<()> {
    if data.len() % DESCRIPTOR_COLS != 0 {
        return Err(SplatError::Process(format!(
            "描述子字节数 {} 不是 {} 的整数倍",
            data.len(),
            DESCRIPTOR_COLS
        )));
    }
    connection
        .execute(
            "INSERT INTO descriptors(image_id, type, rows, cols, data)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                image_id,
                DESCRIPTOR_TYPE_SIFT,
                (data.len() / DESCRIPTOR_COLS) as i64,
                DESCRIPTOR_COLS as i64,
                data.to_vec(),
            ],
        )
        .map_err(sqlite)?;
    Ok(())
}

pub fn write_matches(connection: &Connection, image_id1: u32, image_id2: u32, matches: &[[u32; 2]]) -> Result<()> {
    let data = matches_blob(matches);
    connection
        .execute(
            "INSERT INTO matches(pair_id, rows, cols, data) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![
                pair_id(image_id1, image_id2),
                matches.len() as i64,
                2_i64,
                data
            ],
        )
        .map_err(sqlite)?;
    Ok(())
}

pub fn write_two_view_geometry(
    connection: &Connection,
    image_id1: u32,
    image_id2: u32,
    geometry: &TwoViewGeometry,
) -> Result<()> {
    // 与 COLMAP `SqliteDatabase::WriteTwoViewGeometry` 一致：pair_id 是有序的，
    // 写入前把 `(id1 > id2)` 方向的几何关系取逆。
    let geometry = if image_id1 > image_id2 {
        geometry.inverted()
    } else {
        geometry.clone()
    };
    connection
        .execute(
            "INSERT INTO two_view_geometries
                (pair_id, rows, cols, data, config, F, E, H, qvec, tvec)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![
                pair_id(image_id1, image_id2),
                geometry.inlier_matches.len() as i64,
                2_i64,
                matches_blob(&geometry.inlier_matches),
                geometry.config,
                geometry.f.map(|matrix| matrix_blob(&matrix)),
                geometry.e.map(|matrix| matrix_blob(&matrix)),
                geometry.h.map(|matrix| matrix_blob(&matrix)),
                geometry
                    .cam2_from_cam1
                    .map(|(qvec, _)| vector_blob(&qvec)),
                geometry
                    .cam2_from_cam1
                    .map(|(_, tvec)| vector_blob(&tvec)),
            ],
        )
        .map_err(sqlite)?;
    Ok(())
}

// ---------------------------------------------------------------- rig / frame（4.2.1 兼容）

/// `SensorType::CAMERA`。
const SENSOR_TYPE_CAMERA: i64 = 0;

/// 读相机所属的 rig（对应 COLMAP `Database::ReadRigWithSensor` 的相机分支）。
pub fn read_rig_for_camera(connection: &Connection, camera_id: u32) -> Result<Option<u32>> {
    let rig_id = connection
        .query_row(
            "SELECT rig_id FROM rig_sensors WHERE sensor_id = ?1 AND sensor_type = ?2",
            rusqlite::params![camera_id, SENSOR_TYPE_CAMERA],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(sqlite)?;
    Ok(rig_id.map(|rig_id| rig_id as u32))
}

/// 保证相机有 rig（没有就按 4.2.1 的写法新建），返回 `rig_id`。
///
/// 4.2.1 的特征提取会给每个相机建一个 rig、给每张图像建一个 frame；
/// 本项目自己的代码不读这两张表，但其他平台上的 COLMAP 会（方案 D3）。
pub fn ensure_rig_for_camera(connection: &Connection, camera_id: u32) -> Result<u32> {
    if let Some(rig_id) = read_rig_for_camera(connection, camera_id)? {
        return Ok(rig_id);
    }
    connection
        .execute(
            "INSERT INTO rigs(rig_id, ref_sensor_id, ref_sensor_type) VALUES (NULL, ?1, ?2)",
            rusqlite::params![camera_id, SENSOR_TYPE_CAMERA],
        )
        .map_err(sqlite)?;
    let rig_id = connection.last_insert_rowid() as u32;
    connection
        .execute(
            "INSERT INTO rig_sensors(rig_id, sensor_id, sensor_type, sensor_from_rig)
             VALUES (?1, ?2, ?3, NULL)",
            rusqlite::params![rig_id, camera_id, SENSOR_TYPE_CAMERA],
        )
        .map_err(sqlite)?;
    Ok(rig_id)
}

/// 写一个 frame（自增编号）及其 `frame_data`（相机 → 图像），返回 `frame_id`。
pub fn insert_frame(
    connection: &Connection,
    rig_id: u32,
    image_id: u32,
    camera_id: u32,
) -> Result<u32> {
    connection
        .execute(
            "INSERT INTO frames(frame_id, rig_id) VALUES (NULL, ?1)",
            rusqlite::params![rig_id],
        )
        .map_err(sqlite)?;
    let frame_id = connection.last_insert_rowid() as u32;
    connection
        .execute(
            "INSERT INTO frame_data(frame_id, data_id, sensor_id, sensor_type)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![frame_id, image_id, camera_id, SENSOR_TYPE_CAMERA],
        )
        .map_err(sqlite)?;
    Ok(frame_id)
}

// ---------------------------------------------------------------- 读取

pub fn read_camera(connection: &Connection, camera_id: u32) -> Result<Option<Camera>> {
    let row = connection
        .query_row(
            "SELECT model, width, height, params FROM cameras WHERE camera_id = ?1",
            [camera_id],
            |row| {
                Ok((
                    row.get::<_, i32>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, Option<Vec<u8>>>(3)?,
                ))
            },
        )
        .optional()
        .map_err(sqlite)?;
    let Some((model_id, width, height, params)) = row else {
        return Ok(None);
    };
    let params = params.unwrap_or_default();
    if params.len() % 8 != 0 {
        return Err(SplatError::Process(format!(
            "相机 {camera_id} 的参数块长度非法：{}",
            params.len()
        )));
    }
    let params = params
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    Ok(Some(Camera {
        id: camera_id,
        model_id,
        width: width as u64,
        height: height as u64,
        params,
    }))
}

pub fn read_keypoints(connection: &Connection, image_id: u32) -> Result<Option<Vec<Keypoint>>> {
    let row = connection
        .query_row(
            "SELECT rows, cols, data FROM keypoints WHERE image_id = ?1",
            [image_id],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                ))
            },
        )
        .optional()
        .map_err(sqlite)?;
    let Some((rows, cols, data)) = row else {
        return Ok(None);
    };
    if cols != 6 {
        return Err(SplatError::Process(format!(
            "特征点块列数 {cols} 不是 6（可能来自非 4.2.1 版本写入）"
        )));
    }
    let rows = usize::try_from(rows)
        .map_err(|_| SplatError::Process("特征点行数非法".into()))?;
    if data.len() != rows * 6 * 4 {
        return Err(SplatError::Process(format!(
            "特征点块长度 {} 与 {rows} 行不符",
            data.len()
        )));
    }
    let values = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    Ok(Some(
        values
            .chunks_exact(6)
            .map(|value| Keypoint {
                x: value[0],
                y: value[1],
                shape: [value[2], value[3], value[4], value[5]],
            })
            .collect(),
    ))
}

/// 返回 `(rows, 行主序描述子字节)`；未写过则 `None`。
pub fn read_descriptors(connection: &Connection, image_id: u32) -> Result<Option<(usize, Vec<u8>)>> {
    let row = connection
        .query_row(
            "SELECT rows, cols, data FROM descriptors WHERE image_id = ?1",
            [image_id],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                ))
            },
        )
        .optional()
        .map_err(sqlite)?;
    let Some((rows, cols, data)) = row else {
        return Ok(None);
    };
    let rows = usize::try_from(rows).map_err(|_| SplatError::Process("描述子行数非法".into()))?;
    if cols as usize != DESCRIPTOR_COLS || data.len() != rows * DESCRIPTOR_COLS {
        return Err(SplatError::Process(format!(
            "描述子块 {rows}×{cols}（{} 字节）与 SIFT 布局不符",
            data.len()
        )));
    }
    Ok(Some((rows, data)))
}

pub fn read_matches(
    connection: &Connection,
    image_id1: u32,
    image_id2: u32,
) -> Result<Option<Vec<[u32; 2]>>> {
    let row = connection
        .query_row(
            "SELECT rows, cols, data FROM matches WHERE pair_id = ?1",
            [pair_id(image_id1, image_id2)],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                ))
            },
        )
        .optional()
        .map_err(sqlite)?;
    row.map(|(rows, cols, data)| decode_matches(rows, cols, &data))
        .transpose()
}

pub fn read_two_view_geometry(
    connection: &Connection,
    image_id1: u32,
    image_id2: u32,
) -> Result<Option<TwoViewGeometry>> {
    let row = connection
        .query_row(
            "SELECT rows, cols, data, config, F, E, H, qvec, tvec
             FROM two_view_geometries WHERE pair_id = ?1",
            [pair_id(image_id1, image_id2)],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                    row.get::<_, i32>(3)?,
                    row.get::<_, Option<Vec<u8>>>(4)?,
                    row.get::<_, Option<Vec<u8>>>(5)?,
                    row.get::<_, Option<Vec<u8>>>(6)?,
                    row.get::<_, Option<Vec<u8>>>(7)?,
                    row.get::<_, Option<Vec<u8>>>(8)?,
                ))
            },
        )
        .optional()
        .map_err(sqlite)?;
    let Some((rows, cols, data, config, f, e, h, qvec, tvec)) = row else {
        return Ok(None);
    };
    let cam2_from_cam1 = match (qvec, tvec) {
        (Some(qvec), Some(tvec)) => {
            let qvec = decode_vector(&qvec)?;
            let tvec = decode_vector(&tvec)?;
            if qvec.len() != 4 || tvec.len() != 3 {
                return Err(SplatError::Process(
                    "two_view_geometries 的相对位姿长度非法".into(),
                ));
            }
            Some((
                [qvec[0], qvec[1], qvec[2], qvec[3]],
                [tvec[0], tvec[1], tvec[2]],
            ))
        }
        _ => None,
    };
    Ok(Some(TwoViewGeometry {
        config,
        inlier_matches: decode_matches(rows, cols, &data)?,
        f: f.map(|blob| decode_matrix(&blob)).transpose()?,
        e: e.map(|blob| decode_matrix(&blob)).transpose()?,
        h: h.map(|blob| decode_matrix(&blob)).transpose()?,
        cam2_from_cam1,
    }))
}

/// `images` 表的一行（匹配 / 建图的输入）。
#[derive(Clone, Debug, PartialEq)]
pub struct ImageRow {
    pub image_id: u32,
    pub name: String,
    pub camera_id: u32,
}

/// 所有图像，按编号升序（COLMAP `Database::ReadAllImageIds` 的等价物）。
pub fn read_images(connection: &Connection) -> Result<Vec<ImageRow>> {
    let mut statement = connection
        .prepare("SELECT image_id, name, camera_id FROM images ORDER BY image_id")
        .map_err(sqlite)?;
    let rows = statement
        .query_map([], |row| {
            Ok(ImageRow {
                image_id: row.get::<_, i64>(0)? as u32,
                name: row.get::<_, String>(1)?,
                camera_id: row.get::<_, i64>(2)? as u32,
            })
        })
        .map_err(sqlite)?;
    let mut images = Vec::new();
    for row in rows {
        images.push(row.map_err(sqlite)?);
    }
    Ok(images)
}

/// 该图像对是否已有原始匹配行（对应 COLMAP `ExistsMatches`；`rows = 0` 也算存在）。
pub fn exists_matches(connection: &Connection, image_id1: u32, image_id2: u32) -> Result<bool> {
    connection
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM matches WHERE pair_id = ?1)",
            [pair_id(image_id1, image_id2)],
            |row| row.get::<_, i64>(0),
        )
        .map(|exists| exists != 0)
        .map_err(sqlite)
}

/// 该图像对是否已有几何验证行（对应 COLMAP `ExistsTwoViewGeometry`）。
pub fn exists_two_view_geometry(
    connection: &Connection,
    image_id1: u32,
    image_id2: u32,
) -> Result<bool> {
    connection
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM two_view_geometries WHERE pair_id = ?1)",
            [pair_id(image_id1, image_id2)],
            |row| row.get::<_, i64>(0),
        )
        .map(|exists| exists != 0)
        .map_err(sqlite)
}

/// 删除某图像对的原始匹配（半成品重算用，对应 COLMAP `DeleteMatches`）。
pub fn delete_matches(connection: &Connection, image_id1: u32, image_id2: u32) -> Result<()> {
    connection
        .execute(
            "DELETE FROM matches WHERE pair_id = ?1",
            [pair_id(image_id1, image_id2)],
        )
        .map_err(sqlite)?;
    Ok(())
}

/// 删除某图像对的几何验证行（半成品重算用，对应 COLMAP `DeleteTwoViewGeometry`）。
pub fn delete_two_view_geometry(
    connection: &Connection,
    image_id1: u32,
    image_id2: u32,
) -> Result<()> {
    connection
        .execute(
            "DELETE FROM two_view_geometries WHERE pair_id = ?1",
            [pair_id(image_id1, image_id2)],
        )
        .map_err(sqlite)?;
    Ok(())
}

/// 几何验证通过的所有图像对（升序，元素为 `(较小 id, 较大 id)`）。
pub fn read_verified_pairs(connection: &Connection) -> Result<Vec<[u32; 2]>> {
    let mut statement = connection
        .prepare("SELECT pair_id FROM two_view_geometries ORDER BY pair_id")
        .map_err(sqlite)?;
    let rows = statement
        .query_map([], |row| row.get::<_, i64>(0))
        .map_err(sqlite)?;
    let mut pairs = Vec::new();
    for row in rows {
        let pair = pair_image_ids(row.map_err(sqlite)?);
        if pair.0 != pair.1 {
            pairs.push([pair.0, pair.1]);
        }
    }
    Ok(pairs)
}

// ---------------------------------------------------------------- blob 编码

fn matches_blob(matches: &[[u32; 2]]) -> Vec<u8> {
    let mut data = Vec::with_capacity(matches.len() * 8);
    for pair in matches {
        data.extend_from_slice(&pair[0].to_le_bytes());
        data.extend_from_slice(&pair[1].to_le_bytes());
    }
    data
}

fn decode_matches(rows: i64, cols: i64, data: &[u8]) -> Result<Vec<[u32; 2]>> {
    if cols != 2 {
        return Err(SplatError::Process(format!("匹配块列数 {cols} 不是 2")));
    }
    let rows = usize::try_from(rows).map_err(|_| SplatError::Process("匹配行数非法".into()))?;
    if data.len() != rows * 8 {
        return Err(SplatError::Process(format!(
            "匹配块长度 {} 与 {rows} 行不符",
            data.len()
        )));
    }
    Ok(data
        .chunks_exact(8)
        .map(|chunk| {
            [
                u32::from_le_bytes(chunk[0..4].try_into().unwrap()),
                u32::from_le_bytes(chunk[4..8].try_into().unwrap()),
            ]
        })
        .collect())
}

/// 3×3 矩阵按行主序存 9 个 `f64`（见模块文档）。
fn matrix_blob(matrix: &[[f64; 3]; 3]) -> Vec<u8> {
    let mut data = Vec::with_capacity(9 * 8);
    for row in matrix {
        for value in row {
            data.extend_from_slice(&value.to_le_bytes());
        }
    }
    data
}

fn decode_matrix(blob: &[u8]) -> Result<[[f64; 3]; 3]> {
    let values = decode_vector(blob)?;
    if values.len() != 9 {
        return Err(SplatError::Process(format!(
            "矩阵块长度 {} 不是 9 个 f64",
            values.len()
        )));
    }
    let mut matrix = [[0.0_f64; 3]; 3];
    for (index, value) in values.into_iter().enumerate() {
        matrix[index / 3][index % 3] = value;
    }
    Ok(matrix)
}

fn vector_blob(values: &[f64]) -> Vec<u8> {
    let mut data = Vec::with_capacity(values.len() * 8);
    for value in values {
        data.extend_from_slice(&value.to_le_bytes());
    }
    data
}

fn decode_vector(blob: &[u8]) -> Result<Vec<f64>> {
    if blob.len() % 8 != 0 {
        return Err(SplatError::Process(format!(
            "向量块长度 {} 不是 8 的整数倍",
            blob.len()
        )));
    }
    Ok(blob
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    // `colmap_ba::read_database_stats` 目前还是占位实现（M6 落地）；这里直接复刻它的两条 SQL，
    // 确保数据库满足 `colmap_ba::read_database_stats` 的读取语义。
    fn stats(connection: &Connection) -> (i64, i64) {
        let sum = |table: &str| -> i64 {
            connection
                .query_row(
                    &format!("SELECT COALESCE(SUM(rows), 0) FROM {table}"),
                    [],
                    |row| row.get(0),
                )
                .unwrap()
        };
        (sum("keypoints"), sum("two_view_geometries"))
    }

    #[test]
    fn creates_the_colmap_schema_and_version() {
        let directory = tempfile::tempdir().unwrap();
        let database = directory.path().join("database.db");
        let connection = create(&database).unwrap();

        let version: i32 = connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, DATABASE_VERSION);

        let mut statement = connection
            .prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
            .unwrap();
        let tables = statement
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        for expected in [
            "cameras",
            "descriptors",
            "frame_data",
            "frames",
            "images",
            "keypoints",
            "matches",
            "pose_priors",
            "rig_sensors",
            "rigs",
            "sqlite_sequence",
            "two_view_geometries",
        ] {
            assert!(tables.iter().any(|table| table == expected), "缺表 {expected}");
        }
    }

    #[test]
    fn round_trips_cameras_features_and_geometry() {
        let directory = tempfile::tempdir().unwrap();
        let connection = create(&directory.path().join("database.db")).unwrap();
        let camera = Camera::new(
            1,
            camera_model::SIMPLE_RADIAL,
            1920,
            1080,
            vec![1580.5, 960.0, 540.0, -0.02],
        );
        begin_bulk(&connection).unwrap();
        write_camera(&connection, &camera, true).unwrap();
        write_image(&connection, 1, "frame_000001.jpg", 1).unwrap();
        write_image(&connection, 2, "frame_000002.jpg", 1).unwrap();
        let keypoints = vec![
            Keypoint::upright(10.5, 20.25),
            Keypoint {
                x: 30.0,
                y: 40.0,
                shape: [0.5, 0.25, -0.25, 0.5],
            },
        ];
        write_keypoints(&connection, 1, &keypoints).unwrap();
        write_keypoints(&connection, 2, &keypoints[..1]).unwrap();
        write_descriptors(&connection, 1, &[7_u8; DESCRIPTOR_COLS * 2])
            .unwrap();
        write_matches(&connection, 1, 2, &[[0, 1]]).unwrap();
        write_two_view_geometry(
            &connection,
            1,
            2,
            &TwoViewGeometry {
                config: 1,
                inlier_matches: vec![[1, 0]],
                e: Some([[0.0, -0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]]),
                cam2_from_cam1: Some(([0.5, 0.5, 0.5, 0.5], [1.0, 2.0, 3.0])),
                ..Default::default()
            },
        )
        .unwrap();
        end_bulk(&connection).unwrap();

        assert_eq!(read_camera(&connection, 1).unwrap(), Some(camera));
        assert_eq!(read_keypoints(&connection, 1).unwrap(), Some(keypoints));
        assert_eq!(
            read_keypoints(&connection, 2).unwrap(),
            Some(vec![Keypoint::upright(10.5, 20.25)])
        );
        assert_eq!(
            read_descriptors(&connection, 1).unwrap(),
            Some((2, vec![7_u8; DESCRIPTOR_COLS * 2]))
        );
        assert_eq!(
            read_matches(&connection, 2, 1).unwrap(),
            Some(vec![[0, 1]]),
            "pair_id 与顺序无关"
        );
        assert_eq!(read_verified_pairs(&connection).unwrap(), vec![[1, 2]]);
        let geometry = read_two_view_geometry(&connection, 1, 2).unwrap().unwrap();
        assert_eq!(geometry.config, 1);
        assert_eq!(geometry.inlier_matches, vec![[1, 0]]);
        assert_eq!(geometry.h, None);
        assert_eq!(
            geometry.e,
            Some([[0.0, -0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]])
        );
        assert_eq!(geometry.cam2_from_cam1, Some(([0.5, 0.5, 0.5, 0.5], [1.0, 2.0, 3.0])));

        // 与 `colmap_ba::read_database_stats` 的两条 SQL 语义一致。
        assert_eq!(stats(&connection), (3, 1));
    }

    #[test]
    fn pair_ids_match_the_colmap_formula() {
        assert_eq!(pair_id(1, 2), (MAX_NUM_IMAGES + 2) as i64);
        assert_eq!(pair_id(2, 1), pair_id(1, 2));
        assert_eq!(pair_image_ids(pair_id(3, 9)), (3, 9));
        assert_eq!(pair_image_ids(pair_id(9, 3)), (3, 9));
    }

    #[test]
    fn rejects_inconsistent_payloads() {
        let directory = tempfile::tempdir().unwrap();
        let connection = create(&directory.path().join("database.db")).unwrap();
        // 描述子字节数不是 128 的整数倍。
        assert!(write_descriptors(&connection, 1, &[0_u8; 100]).is_err());
        // 图像编号超出 COLMAP 的 CHECK 约束上限（`kMaxNumImages` 本身不可用）。
        assert!(write_image(&connection, MAX_NUM_IMAGES as u32, "a.jpg", 1).is_err());
        // 相机参数个数与模型不符：`Camera::new` 之外直接构造的模型也必须被拦下。
        let inconsistent = Camera {
            id: 1,
            model_id: camera_model::SIMPLE_RADIAL,
            width: 1,
            height: 1,
            params: vec![0.0; 3],
        };
        assert!(write_camera(&connection, &inconsistent, false).is_err());
    }
}
