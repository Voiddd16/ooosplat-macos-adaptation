//! `sparse/<n>/{cameras,images,points3D}.bin` 的读写。
//!
//! 布局逐字节对齐 COLMAP 4.2.1（`src/colmap/scene/reconstruction_io_binary.cc`）：
//!
//! ```text
//! cameras.bin : u64 n; n × { u32 camera_id, i32 model_id, u64 width, u64 height,
//!                            f64 params[num_params(model_id)] }
//! images.bin  : u64 n; n × { u32 image_id, f64 qw,qx,qy,qz,tx,ty,tz, u32 camera_id,
//!                            <name> 0x00, u64 num_points2D,
//!                            num_points2D × { f64 x, f64 y, u64 point3D_id } }   // 24 字节/点
//! points3D.bin: u64 n; n × { u64 point3D_id, f64 x,y,z, u8 r,g,b, f64 error,
//!                            u64 track_length,
//!                            track_length × { u32 image_id, u32 point2D_idx } }  // 8 字节/观测
//! ```
//!
//! 条目按 id 升序写出（与 COLMAP 的 `ExtractSortedIds` 一致），因此可执行文件
//! 层面的输出与 COLMAP 逐字节一致。rigs/frames 不写出：本项目产出的是
//! 「一台共享相机」的 3 文件模型，`reusable_model_files` 视为合法旧式模型。

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

use super::{camera_model, INVALID_POINT3D_ID};
use crate::error::{Result, SplatError};

/// `cameras.bin` 的一台相机。
#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    pub id: u32,
    pub model_id: i32,
    pub width: u64,
    pub height: u64,
    pub params: Vec<f64>,
}

impl Camera {
    /// 参数个数与模型不符时立即失败（写坏模型会静默污染整条链路）。
    pub fn new(id: u32, model_id: i32, width: u64, height: u64, params: Vec<f64>) -> Self {
        let expected = camera_model::num_params(model_id).unwrap_or(params.len());
        assert_eq!(
            params.len(),
            expected,
            "相机模型 {} 需要 {} 个参数",
            camera_model::name(model_id).unwrap_or("UNKNOWN"),
            expected
        );
        Self {
            id,
            model_id,
            width,
            height,
            params,
        }
    }
}

/// `images.bin` 里的一个特征点（`point3d_id` 为 `INVALID_POINT3D_ID` 表示未三角化）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
    pub point3d_id: u64,
}

impl Point2D {
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            x,
            y,
            point3d_id: INVALID_POINT3D_ID,
        }
    }
}

/// `images.bin` 的一张已注册图像（四元数顺序 `w,x,y,z`）。
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub id: u32,
    pub qvec: [f64; 4],
    pub tvec: [f64; 3],
    pub camera_id: u32,
    pub name: String,
    pub points2d: Vec<Point2D>,
}

/// `points3D.bin` 的一个观测；`point2d_idx` 是目标图像 `points2d` 的下标。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrackElement {
    pub image_id: u32,
    pub point2d_idx: u32,
}

/// `points3D.bin` 的一个三维点。
#[derive(Clone, Debug, PartialEq)]
pub struct Point3D {
    pub id: u64,
    pub xyz: [f64; 3],
    pub color: [u8; 3],
    pub error: f64,
    pub track: Vec<TrackElement>,
}

// ---------------------------------------------------------------- 写入

pub fn write_cameras(path: &Path, cameras: &[Camera]) -> Result<()> {
    let mut writer = writer(path)?;
    write_u64(&mut writer, cameras.len() as u64)?;

    let mut order = (0..cameras.len()).collect::<Vec<_>>();
    order.sort_by_key(|index| cameras[*index].id);
    for index in order {
        let camera = &cameras[index];
        if camera_model::num_params(camera.model_id) != Some(camera.params.len()) {
            return Err(SplatError::Process(format!(
                "相机 {} 使用了未知模型编号 {} 或参数个数不符",
                camera.id, camera.model_id
            )));
        }
        write_u32(&mut writer, camera.id)?;
        write_i32(&mut writer, camera.model_id)?;
        write_u64(&mut writer, camera.width)?;
        write_u64(&mut writer, camera.height)?;
        for param in &camera.params {
            write_f64(&mut writer, *param)?;
        }
    }
    flush(writer)
}

pub fn write_images(path: &Path, images: &[Image]) -> Result<()> {
    let mut writer = writer(path)?;
    write_u64(&mut writer, images.len() as u64)?;

    let mut order = (0..images.len()).collect::<Vec<_>>();
    order.sort_by_key(|index| images[*index].id);
    for index in order {
        let image = &images[index];
        if image.name.as_bytes().contains(&0) {
            return Err(SplatError::Process(format!(
                "图像名称包含 NUL 字节：{}",
                image.name
            )));
        }
        write_u32(&mut writer, image.id)?;
        for value in image.qvec {
            write_f64(&mut writer, value)?;
        }
        for value in image.tvec {
            write_f64(&mut writer, value)?;
        }
        write_u32(&mut writer, image.camera_id)?;
        writer.write_all(image.name.as_bytes())?;
        writer.write_all(&[0])?;
        write_u64(&mut writer, image.points2d.len() as u64)?;
        for point in &image.points2d {
            write_f64(&mut writer, point.x)?;
            write_f64(&mut writer, point.y)?;
            write_u64(&mut writer, point.point3d_id)?;
        }
    }
    flush(writer)
}

pub fn write_points3d(path: &Path, points: &[Point3D]) -> Result<()> {
    let mut writer = writer(path)?;
    write_u64(&mut writer, points.len() as u64)?;

    let mut order = (0..points.len()).collect::<Vec<_>>();
    order.sort_by_key(|index| points[*index].id);
    for index in order {
        let point = &points[index];
        write_u64(&mut writer, point.id)?;
        for value in point.xyz {
            write_f64(&mut writer, value)?;
        }
        writer.write_all(&point.color)?;
        write_f64(&mut writer, point.error)?;
        write_u64(&mut writer, point.track.len() as u64)?;
        for element in &point.track {
            write_u32(&mut writer, element.image_id)?;
            write_u32(&mut writer, element.point2d_idx)?;
        }
    }
    flush(writer)
}

/// 写出完整模型（`cameras.bin` + `images.bin` + `points3D.bin`）。
pub fn write_model(
    directory: &Path,
    cameras: &[Camera],
    images: &[Image],
    points: &[Point3D],
) -> Result<()> {
    std::fs::create_dir_all(directory)?;
    write_cameras(&directory.join("cameras.bin"), cameras)?;
    write_images(&directory.join("images.bin"), images)?;
    write_points3d(&directory.join("points3D.bin"), points)
}

// ---------------------------------------------------------------- 读取

pub fn read_cameras(path: &Path) -> Result<Vec<Camera>> {
    let mut reader = reader(path)?;
    let count = read_u64(&mut reader)?;
    let mut cameras = Vec::new();
    for _ in 0..count {
        let id = read_u32(&mut reader)?;
        let model_id = read_i32(&mut reader)?;
        let width = read_u64(&mut reader)?;
        let height = read_u64(&mut reader)?;
        let num_params = camera_model::num_params(model_id).ok_or_else(|| {
            SplatError::Process(format!("相机 {id} 使用了未知模型编号 {model_id}"))
        })?;
        let mut params = Vec::with_capacity(num_params);
        for _ in 0..num_params {
            params.push(read_f64(&mut reader)?);
        }
        cameras.push(Camera {
            id,
            model_id,
            width,
            height,
            params,
        });
    }
    Ok(cameras)
}

pub fn read_images(path: &Path) -> Result<Vec<Image>> {
    let mut reader = reader(path)?;
    let count = read_u64(&mut reader)?;
    let capacity = usize::try_from(count)
        .map_err(|_| SplatError::Process("COLMAP 图像数量超出可读取范围".into()))?;
    let mut images = Vec::with_capacity(capacity);
    for _ in 0..count {
        let id = read_u32(&mut reader)?;
        let mut qvec = [0.0_f64; 4];
        for value in &mut qvec {
            *value = read_f64(&mut reader)?;
        }
        let mut tvec = [0.0_f64; 3];
        for value in &mut tvec {
            *value = read_f64(&mut reader)?;
        }
        let camera_id = read_u32(&mut reader)?;
        let name = read_name(&mut reader)?;
        let num_points = read_u64(&mut reader)?;
        let capacity = usize::try_from(num_points)
            .map_err(|_| SplatError::Process("COLMAP 特征点数量超出可读取范围".into()))?;
        let mut points2d = Vec::with_capacity(capacity);
        for _ in 0..num_points {
            let x = read_f64(&mut reader)?;
            let y = read_f64(&mut reader)?;
            let point3d_id = read_u64(&mut reader)?;
            points2d.push(Point2D { x, y, point3d_id });
        }
        images.push(Image {
            id,
            qvec,
            tvec,
            camera_id,
            name,
            points2d,
        });
    }
    Ok(images)
}

pub fn read_points3d(path: &Path) -> Result<Vec<Point3D>> {
    let mut reader = reader(path)?;
    let count = read_u64(&mut reader)?;
    let capacity = usize::try_from(count)
        .map_err(|_| SplatError::Process("COLMAP 三维点数量超出可读取范围".into()))?;
    let mut points = Vec::with_capacity(capacity);
    for _ in 0..count {
        let id = read_u64(&mut reader)?;
        let xyz = [
            read_f64(&mut reader)?,
            read_f64(&mut reader)?,
            read_f64(&mut reader)?,
        ];
        let mut color = [0_u8; 3];
        reader.read_exact(&mut color)?;
        let error = read_f64(&mut reader)?;
        let track_length = read_u64(&mut reader)?;
        let capacity = usize::try_from(track_length)
            .map_err(|_| SplatError::Process("COLMAP 观测数量超出可读取范围".into()))?;
        let mut track = Vec::with_capacity(capacity);
        for _ in 0..track_length {
            track.push(TrackElement {
                image_id: read_u32(&mut reader)?,
                point2d_idx: read_u32(&mut reader)?,
            });
        }
        points.push(Point3D {
            id,
            xyz,
            color,
            error,
            track,
        });
    }
    Ok(points)
}

/// 读取完整模型。
pub fn read_model(directory: &Path) -> Result<(Vec<Camera>, Vec<Image>, Vec<Point3D>)> {
    Ok((
        read_cameras(&directory.join("cameras.bin"))?,
        read_images(&directory.join("images.bin"))?,
        read_points3d(&directory.join("points3D.bin"))?,
    ))
}

// ---------------------------------------------------------------- 底层 I/O

fn writer(path: &Path) -> Result<BufWriter<File>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(BufWriter::new(File::create(path)?))
}

fn flush(mut writer: BufWriter<File>) -> Result<()> {
    writer.flush()?;
    Ok(())
}

fn reader(path: &Path) -> Result<BufReader<File>> {
    Ok(BufReader::new(File::open(path)?))
}

fn write_u32(writer: &mut impl Write, value: u32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_i32(writer: &mut impl Write, value: i32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_u64(writer: &mut impl Write, value: u64) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_f64(writer: &mut impl Write, value: f64) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32(reader: &mut impl Read) -> Result<i32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f64(reader: &mut impl Read) -> Result<f64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(f64::from_le_bytes(bytes))
}

/// 读 NUL 结尾的 UTF-8 名称；上限与 `colmap_model.rs` 一致。
fn read_name(reader: &mut impl Read) -> Result<String> {
    let mut bytes = Vec::new();
    loop {
        let mut byte = [0_u8; 1];
        reader.read_exact(&mut byte)?;
        if byte[0] == 0 {
            break;
        }
        bytes.push(byte[0]);
        if bytes.len() > 32 * 1024 {
            return Err(SplatError::Process("COLMAP 图像名称异常过长".into()));
        }
    }
    String::from_utf8(bytes).map_err(|_| SplatError::Process("COLMAP 图像名称不是有效 UTF-8".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconstruction::colmap_model;

    fn sample() -> (Vec<Camera>, Vec<Image>, Vec<Point3D>) {
        let cameras = vec![
            Camera::new(
                1,
                camera_model::SIMPLE_RADIAL,
                1920,
                1080,
                vec![1580.5, 960.0, 540.0, -0.02],
            ),
            Camera::new(
                7,
                camera_model::PINHOLE,
                640,
                480,
                vec![500.0, 510.0, 320.0, 240.0],
            ),
        ];
        let images = vec![
            Image {
                id: 3,
                qvec: [0.999_9, 0.01, -0.02, 0.03],
                tvec: [0.1, -0.2, 0.3],
                camera_id: 1,
                name: "frame_000002.jpg".into(),
                points2d: vec![
                    Point2D::new(10.5, 20.25),
                    Point2D {
                        x: 30.0,
                        y: 40.0,
                        point3d_id: 42,
                    },
                ],
            },
            Image {
                id: 1,
                qvec: [1.0, 0.0, 0.0, 0.0],
                tvec: [0.0, 0.0, 0.0],
                camera_id: 1,
                name: "frame_000001.jpg".into(),
                points2d: vec![Point2D::new(1.0, 2.0)],
            },
        ];
        let points = vec![
            Point3D {
                id: 9,
                xyz: [1.5, -2.5, 3.5],
                color: [10, 20, 30],
                error: 0.75,
                track: vec![
                    TrackElement {
                        image_id: 1,
                        point2d_idx: 0,
                    },
                    TrackElement {
                        image_id: 3,
                        point2d_idx: 1,
                    },
                ],
            },
            Point3D {
                id: 2,
                xyz: [0.0, 0.0, 0.0],
                color: [255, 255, 255],
                error: 0.0,
                track: vec![TrackElement {
                    image_id: 3,
                    point2d_idx: 0,
                }],
            },
        ];
        (cameras, images, points)
    }

    #[test]
    fn round_trips_the_full_model() {
        let directory = tempfile::tempdir().unwrap();
        let (cameras, images, points) = sample();
        write_model(directory.path(), &cameras, &images, &points).unwrap();

        let (read_cameras, read_images, read_points) = read_model(directory.path()).unwrap();
        assert_eq!(read_cameras.len(), 2);
        assert_eq!(read_points.len(), 2);
        // 写出按 id 升序，读回也是升序；与输入集合一致（输入本身乱序）。
        let mut cameras_sorted = cameras.clone();
        cameras_sorted.sort_by_key(|camera| camera.id);
        assert_eq!(read_cameras, cameras_sorted);
        let mut images_sorted = images.clone();
        images_sorted.sort_by_key(|image| image.id);
        assert_eq!(read_images, images_sorted);
        let mut points_sorted = points.clone();
        points_sorted.sort_by_key(|point| point.id);
        assert_eq!(read_points, points_sorted);
    }

    /// 与项目**现有** reader（`reconstruction/colmap_model.rs`）交叉验证：
    /// 这是真正的红线消费者。
    #[test]
    fn is_readable_by_the_existing_project_reader() {
        let directory = tempfile::tempdir().unwrap();
        let (mut cameras, images, points) = sample();
        cameras.truncate(1);
        write_model(directory.path(), &cameras, &images, &points).unwrap();

        let camera = colmap_model::read_single_camera(directory.path()).unwrap();
        assert_eq!((camera.id, camera.model_id), (1, camera_model::SIMPLE_RADIAL));
        assert_eq!(camera.model, "SIMPLE_RADIAL");
        assert_eq!((camera.width, camera.height), (1920, 1080));

        let names = colmap_model::read_registered_images(directory.path()).unwrap();
        assert_eq!(
            names
                .iter()
                .map(|image| (image.id, image.camera_id, image.name.as_str()))
                .collect::<Vec<_>>(),
            vec![
                (1, 1, "frame_000001.jpg"),
                (3, 1, "frame_000002.jpg")
            ]
        );
        assert_eq!(
            colmap_model::reusable_model_files(directory.path())
                .unwrap()
                .len(),
            3
        );
    }

    /// 逐字节校验头部布局（防止"能读回但布局不对"的假通过）。
    #[test]
    fn matches_the_colmap_byte_layout() {
        let directory = tempfile::tempdir().unwrap();
        let (cameras, _, _) = sample();
        write_cameras(&directory.path().join("cameras.bin"), &cameras[..1]).unwrap();
        let bytes = std::fs::read(directory.path().join("cameras.bin")).unwrap();

        let mut expected = Vec::new();
        expected.extend_from_slice(&1_u64.to_le_bytes());
        expected.extend_from_slice(&1_u32.to_le_bytes());
        expected.extend_from_slice(&camera_model::SIMPLE_RADIAL.to_le_bytes());
        expected.extend_from_slice(&1920_u64.to_le_bytes());
        expected.extend_from_slice(&1080_u64.to_le_bytes());
        for param in [1580.5_f64, 960.0, 540.0, -0.02] {
            expected.extend_from_slice(&param.to_le_bytes());
        }
        assert_eq!(bytes, expected);
        assert_eq!(bytes.len(), 8 + 4 + 4 + 8 + 8 + 4 * 8);

        // images.bin：每点 24 字节；points3D.bin：每观测 8 字节。
        let (_, images, points) = sample();
        write_images(&directory.path().join("images.bin"), &images[..1]).unwrap();
        let image_bytes = std::fs::read(directory.path().join("images.bin")).unwrap();
        assert_eq!(
            image_bytes.len(),
            8 + 4 + 7 * 8 + 4 + "frame_000002.jpg".len() + 1 + 8 + 2 * 24
        );
        write_points3d(&directory.path().join("points3D.bin"), &points[..1]).unwrap();
        let point_bytes = std::fs::read(directory.path().join("points3D.bin")).unwrap();
        assert_eq!(point_bytes.len(), 8 + 8 + 3 * 8 + 3 + 8 + 8 + 2 * 8);
    }

    #[test]
    fn rejects_unwritable_inputs() {
        let directory = tempfile::tempdir().unwrap();
        let bad_name = Image {
            id: 1,
            qvec: [1.0, 0.0, 0.0, 0.0],
            tvec: [0.0, 0.0, 0.0],
            camera_id: 1,
            name: "bad\0name.jpg".into(),
            points2d: Vec::new(),
        };
        assert!(write_images(&directory.path().join("images.bin"), &[bad_name]).is_err());

        let bad_camera = Camera {
            id: 1,
            model_id: camera_model::SIMPLE_RADIAL,
            width: 1,
            height: 1,
            params: vec![0.0; 3],
        };
        assert!(write_cameras(&directory.path().join("cameras.bin"), &[bad_camera]).is_err());
    }
}
