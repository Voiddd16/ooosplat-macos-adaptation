//! M5a：重建状态（`Reconstruction`）、对应图（`CorrespondenceGraph`）、
//! 观测统计（`ObservationManager` + 可见性金字塔）。
//!
//! 语义逐项对齐 COLMAP 4.2.1（`scene/reconstruction.cc`、`scene/correspondence_graph.cc`、
//! `sfm/observation_manager.cc`、`scene/visibility_pyramid.cc`），规格见
//! `docs/project-notes/recon/m5-mapper.md` §2 / §11 / §12。
//!
//! 与 COLMAP 的结构差异（刻意）：
//! - **稠密下标**：图像 / 相机 / 三维点都用 `Vec` 下标 + `HashMap` 反查表，而不用
//!   `FlatHashMap` —— 本项目单模型最多几百张图像，遍历局部性比哈希更好。
//! - 三维点 id 取 `index + 1`（与 COLMAP 的自增 id 等价，且 `Point2D::point3d_id`
//!   直接存这个 id，`INVALID_POINT3D_ID` 表示未三角化）。
//! - 只实现**单传感器 rig**（本项目的特征提取阶段一张图一个 frame 一个相机），
//!   因此 frame 级 API 退化成 image 级；4.2.1 的 `FixGauge` 等价语义在 `ba.rs` 里。

use std::collections::{HashMap, HashSet};

use super::ba::CameraModelId;
use super::camera::{self, projection_center};
use super::format::model::{Camera, Image, Point2D, Point3D, TrackElement};
use super::format::INVALID_POINT3D_ID;

// ---------------------------------------------------------------- 对应图

/// 一条 2D-2D 对应（稠密下标空间）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Correspondence {
    pub image: usize,
    pub point2d_idx: u32,
}

/// 一对图像的对应统计（4.2.1 `ImagePairStat`）。
#[derive(Clone, Copy, Debug, Default)]
pub struct PairStats {
    /// 该图像对已三角化的对应数。
    pub num_tri_corrs: usize,
    /// 该图像对的对应总数。
    pub num_total_corrs: usize,
}

/// 2D-2D 对应图（4.2.1 `CorrespondenceGraph`）。
///
/// 表结构 `[image][point2D_idx] → Vec<Correspondence>`，构造完成后按 `(image, idx)`
/// 升序排序去重（4.2.1 `CorrespondenceGraph::Finalize`）。
#[derive(Clone, Debug, Default)]
pub struct CorrespondenceGraph {
    table: Vec<Vec<Vec<Correspondence>>>,
    num_corrs: Vec<usize>,
}

impl CorrespondenceGraph {
    /// 用每张图的特征点数建表。
    pub fn new(num_points2d_per_image: &[usize]) -> Self {
        Self {
            table: num_points2d_per_image
                .iter()
                .map(|count| vec![Vec::new(); *count])
                .collect(),
            num_corrs: vec![0; num_points2d_per_image.len()],
        }
    }

    pub fn num_images(&self) -> usize {
        self.table.len()
    }

    /// 插入一条匹配（对称两侧都插入）。
    pub fn add_match(&mut self, image1: usize, idx1: u32, image2: usize, idx2: u32) {
        self.table[image1][idx1 as usize].push(Correspondence {
            image: image2,
            point2d_idx: idx2,
        });
        self.table[image2][idx2 as usize].push(Correspondence {
            image: image1,
            point2d_idx: idx1,
        });
        self.num_corrs[image1] += 1;
        self.num_corrs[image2] += 1;
    }

    /// 排序去重（4.2.1 在构造结束时统一做一次）。
    pub fn finalize(&mut self) {
        for image in &mut self.table {
            for point2d in image.iter_mut() {
                point2d.sort_by_key(|corr| (corr.image, corr.point2d_idx));
                point2d.dedup();
            }
        }
    }

    pub fn find(&self, image: usize, point2d_idx: u32) -> &[Correspondence] {
        &self.table[image][point2d_idx as usize]
    }

    pub fn num_correspondences(&self, image: usize) -> usize {
        self.num_corrs[image]
    }

    /// 4.2.1 `IsTwoViewObservation`：该特征点的每个对应点都只看到这一个对应。
    pub fn is_two_view_observation(&self, image: usize, point2d_idx: u32) -> bool {
        let correspondences = self.find(image, point2d_idx);
        if correspondences.is_empty() {
            return false;
        }
        correspondences
            .iter()
            .all(|corr| self.find(corr.image, corr.point2d_idx).len() == 1)
    }

    /// 4.2.1 `ExtractTransitiveCorrespondences`：BFS 展开 `transitivity` 层，
    /// 结果**不含**起始观测本身，且不含起始观测（去重集合）。
    pub fn find_transitive(
        &self,
        image: usize,
        point2d_idx: u32,
        transitivity: usize,
    ) -> Vec<Correspondence> {
        let start = (image, point2d_idx);
        let mut visited: HashSet<(usize, u32)> = HashSet::new();
        visited.insert(start);
        let mut result: Vec<Correspondence> = Vec::new();
        if transitivity == 0 {
            return result;
        }
        let mut current: Vec<Correspondence> = self.find(image, point2d_idx).to_vec();
        for &corr in &current {
            if visited.insert((corr.image, corr.point2d_idx)) {
                result.push(corr);
            }
        }
        for _ in 1..transitivity {
            let mut next: Vec<Correspondence> = Vec::new();
            for corr in &current {
                for &other in self.find(corr.image, corr.point2d_idx) {
                    if visited.insert((other.image, other.point2d_idx)) {
                        next.push(other);
                        result.push(other);
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            current = next;
        }
        result
    }

    /// 两个图对之间的匹配（`(idx1, idx2)`），顺序按 `idx1` 升序。
    pub fn matches_between(&self, image1: usize, image2: usize) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        for (index, point2d) in self.table[image1].iter().enumerate() {
            for corr in point2d {
                if corr.image == image2 {
                    out.push((index as u32, corr.point2d_idx));
                }
            }
        }
        out
    }

    /// 出现过的图像对与其对应总数（4.2.1 `ObservationManager` 构造时的统计源）。
    pub fn pair_correspondence_counts(&self) -> HashMap<(usize, usize), usize> {
        let mut counts: HashMap<(usize, usize), usize> = HashMap::new();
        for image in 0..self.table.len() {
            for point2d in &self.table[image] {
                for corr in point2d {
                    let key = if image < corr.image {
                        (image, corr.image)
                    } else {
                        (corr.image, image)
                    };
                    // 每对匹配在两侧各出现一次，只在 image < corr.image 时计一次。
                    if image < corr.image {
                        *counts.entry(key).or_insert(0) += 1;
                    }
                }
            }
        }
        counts
    }

    /// 出现过的图像对（去重、稳定顺序）。
    pub fn image_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs: Vec<(usize, usize)> = self
            .pair_correspondence_counts()
            .into_keys()
            .collect();
        pairs.sort_unstable();
        pairs
    }
}

// ---------------------------------------------------------------- 可见性金字塔

/// 4.2.1 `VisibilityPyramid`（6 级，`dim_ℓ = 2^(ℓ+1)`）。
///
/// 分数越小表示可见点分布越均匀（`MIN_UNCERTAINTY` 选择策略用它排序）。
#[derive(Clone, Debug)]
pub struct VisibilityPyramid {
    width: usize,
    height: usize,
    levels: Vec<Vec<u32>>,
    score: usize,
}

impl VisibilityPyramid {
    pub fn new(width: usize, height: usize, num_levels: usize) -> Self {
        let levels = (0..num_levels)
            .map(|level| {
                let dim = 1usize << (level + 1);
                vec![0u32; dim * dim]
            })
            .collect();
        Self {
            width: width.max(1),
            height: height.max(1),
            levels,
            score: 0,
        }
    }

    fn cell_for_point(&self, x: f64, y: f64) -> (usize, usize) {
        let max_dim = 1usize << self.levels.len();
        let cx = ((max_dim as f64 * x / self.width as f64) as usize).clamp(0, max_dim - 1);
        let cy = ((max_dim as f64 * y / self.height as f64) as usize).clamp(0, max_dim - 1);
        (cx, cy)
    }

    pub fn set_point(&mut self, x: f64, y: f64) {
        if self.levels.is_empty() {
            return;
        }
        let (mut cx, mut cy) = self.cell_for_point(x, y);
        for level in (0..self.levels.len()).rev() {
            let dim = 1usize << (level + 1);
            let cell = &mut self.levels[level][cy * dim + cx];
            *cell += 1;
            if *cell == 1 {
                self.score += dim * dim;
            }
            cx >>= 1;
            cy >>= 1;
        }
    }

    pub fn reset_point(&mut self, x: f64, y: f64) {
        if self.levels.is_empty() {
            return;
        }
        let (mut cx, mut cy) = self.cell_for_point(x, y);
        for level in (0..self.levels.len()).rev() {
            let dim = 1usize << (level + 1);
            let cell = &mut self.levels[level][cy * dim + cx];
            *cell = cell.saturating_sub(1);
            if *cell == 0 {
                self.score = self.score.saturating_sub(dim * dim);
            }
            cx >>= 1;
            cy >>= 1;
        }
    }

    pub fn score(&self) -> usize {
        self.score
    }
}

// ---------------------------------------------------------------- 重建状态

/// 一张图像的可变状态（`images.bin` 内容 + 注册位姿）。
#[derive(Clone, Debug)]
pub struct ImageState {
    pub id: u32,
    pub name: String,
    /// 稠密相机下标。
    pub camera: usize,
    pub points2d: Vec<Point2D>,
    /// 与 `points2d` 同下标的特征观测颜色（RGB，来自数据库 `feature_colors`）；
    /// 空表示数据库缺少该表（旧版数据），均值色退化为黑色。
    pub colors: Vec<[u8; 3]>,
    pub registered: bool,
    /// `cam_from_world`（四元数顺序 `w,x,y,z`）。
    pub qvec: [f64; 4],
    pub tvec: [f64; 3],
}

/// 重建状态（4.2.1 `Reconstruction` 的单传感器子集）。
#[derive(Clone, Debug, Default)]
pub struct Reconstruction {
    pub cameras: Vec<Camera>,
    /// `prior_focal_length` 不进 `cameras.bin`，但 mapper 需要它决定是否精化内参。
    pub camera_prior_focal: Vec<bool>,
    pub camera_index: HashMap<u32, usize>,
    pub images: Vec<ImageState>,
    pub image_index: HashMap<u32, usize>,
    pub points: Vec<Point3D>,
    pub registered: Vec<usize>,
}

impl Reconstruction {
    pub fn new(cameras: Vec<Camera>, camera_prior_focal: Vec<bool>, images: Vec<ImageState>) -> Self {
        let camera_index = cameras
            .iter()
            .enumerate()
            .map(|(index, camera)| (camera.id, index))
            .collect();
        let image_index = images
            .iter()
            .enumerate()
            .map(|(index, image)| (image.id, index))
            .collect();
        // 已注册标记（续跑时从既有模型载入）→ `registered` 列表保持一致。
        let registered = images
            .iter()
            .enumerate()
            .filter(|(_, image)| image.registered)
            .map(|(index, _)| index)
            .collect();
        Self {
            cameras,
            camera_prior_focal,
            camera_index,
            images,
            image_index,
            points: Vec::new(),
            registered,
        }
    }

    pub fn num_images(&self) -> usize {
        self.images.len()
    }

    pub fn num_reg_images(&self) -> usize {
        self.registered.len()
    }

    /// track 内各观测的特征颜色均值（跳过黑色与缺失条目）。
    /// COLMAP 语义：`Point3D.color` 取 track 观测色均值；无数据时为黑。
    pub fn mean_track_color(&self, track: &[TrackElement]) -> [u8; 3] {
        let (mut r, mut g, mut b) = (0u64, 0u64, 0u64);
        let mut count = 0usize;
        for element in track {
            let Some(&index) = self.image_index.get(&element.image_id) else {
                continue;
            };
            let Some(color) = self
                .images
                .get(index)
                .and_then(|image| image.colors.get(element.point2d_idx as usize))
            else {
                continue;
            };
            if *color == [0, 0, 0] {
                continue;
            }
            r += color[0] as u64;
            g += color[1] as u64;
            b += color[2] as u64;
            count += 1;
        }
        if count == 0 {
            [0, 0, 0]
        } else {
            [
                (r / count as u64) as u8,
                (g / count as u64) as u8,
                (b / count as u64) as u8,
            ]
        }
    }

    pub fn num_points3d(&self) -> usize {
        self.points.len()
    }

    pub fn camera_of(&self, image: usize) -> &Camera {
        &self.cameras[self.images[image].camera]
    }

    pub fn camera_id_of(&self, image: usize) -> u32 {
        self.cameras[self.images[image].camera].id
    }

    pub fn model_of(&self, image: usize) -> CameraModelId {
        camera::model_of(self.camera_of(image))
    }

    /// 特征点当前的三维点下标（未三角化 → `None`）。
    pub fn point_index_at(&self, image: usize, point2d_idx: u32) -> Option<usize> {
        point_index_of(self.images[image].points2d[point2d_idx as usize].point3d_id)
    }

    pub fn add_point3d(&mut self, xyz: [f64; 3], color: [u8; 3], track: Vec<TrackElement>) -> usize {
        let index = self.points.len();
        let id = index as u64 + 1;
        // 回写 image 侧的 `point3d_id`（不变式：每个特征点最多属于一个三维点）。
        for element in &track {
            let image = self.image_index[&element.image_id];
            self.images[image].points2d[element.point2d_idx as usize].point3d_id = id;
        }
        self.points.push(Point3D {
            id,
            xyz,
            color,
            error: -1.0,
            track,
        });
        index
    }

    /// 给已有特征点追加观测（要求该特征点尚未三角化）。
    pub fn add_observation(&mut self, point: usize, element: TrackElement) -> bool {
        let image = match self.image_index.get(&element.image_id) {
            Some(index) => *index,
            None => return false,
        };
        let slot = &mut self.images[image].points2d[element.point2d_idx as usize];
        if slot.point3d_id != INVALID_POINT3D_ID {
            return false;
        }
        slot.point3d_id = point as u64 + 1;
        self.points[point].track.push(element);
        true
    }

    pub fn delete_point3d(&mut self, point: usize) {
        for element in &self.points[point].track {
            let image = self.image_index[&element.image_id];
            self.images[image].points2d[element.point2d_idx as usize].point3d_id =
                INVALID_POINT3D_ID;
        }
        // 与 COLMAP 一致：整体删除后不保留空洞（末尾交换删除）。点 id 与
        // `Point2D::point3d_id` 都按"下标 + 1"编码，交换后必须把末尾点的 id
        // 和它所有观测的引用改写到新下标，否则按 id 反查会指错点。
        let last = self.points.len() - 1;
        if point != last {
            self.points.swap(point, last);
            self.points[point].id = point as u64 + 1;
            let track = std::mem::take(&mut self.points[point].track);
            for element in &track {
                let image = self.image_index[&element.image_id];
                self.images[image].points2d[element.point2d_idx as usize].point3d_id =
                    point as u64 + 1;
            }
            self.points[point].track = track;
        }
        self.points.pop();
    }

    pub fn delete_observation(&mut self, point: usize, element: TrackElement) {
        let image = self.image_index[&element.image_id];
        self.images[image].points2d[element.point2d_idx as usize].point3d_id =
            INVALID_POINT3D_ID;
        self.points[point].track.retain(|item| *item != element);
    }

    /// 4.2.1 `MergePoints3D`：长度加权平均位置，颜色按"已提取"取平均，track 拼接。
    pub fn merge_points3d(&mut self, point1: usize, point2: usize) -> Option<usize> {
        if point1 == point2 {
            return None;
        }
        let (first, second) = if point1 < point2 {
            (point1, point2)
        } else {
            (point2, point1)
        };
        let a = &self.points[first];
        let b = &self.points[second];
        let weight1 = a.track.len() as f64;
        let weight2 = b.track.len() as f64;
        let total = weight1 + weight2;
        if total == 0.0 {
            return None;
        }
        let mut xyz = [0.0; 3];
        for axis in 0..3 {
            xyz[axis] = (weight1 * a.xyz[axis] + weight2 * b.xyz[axis]) / total;
        }
        let mut color = [0u8; 3];
        let mut color_sum = [0f64; 3];
        let mut color_count = 0f64;
        for point in [a, b] {
            if point.color != [0, 0, 0] {
                for channel in 0..3 {
                    color_sum[channel] += point.color[channel] as f64;
                }
                color_count += 1.0;
            }
        }
        if color_count > 0.0 {
            for channel in 0..3 {
                color[channel] = (color_sum[channel] / color_count).round() as u8;
            }
        }
        let mut track = a.track.clone();
        track.extend(b.track.iter().copied());
        self.delete_point3d(second);
        self.delete_point3d(first);
        Some(self.add_point3d(xyz, color, track))
    }

    /// 注册一张图（要求未注册）。
    pub fn register_image(&mut self, image: usize, qvec: [f64; 4], tvec: [f64; 3]) {
        if self.images[image].registered {
            return;
        }
        self.images[image].registered = true;
        self.images[image].qvec = qvec;
        self.images[image].tvec = tvec;
        self.registered.push(image);
    }

    /// 注销一张图：先删掉它在所有三维点里的观测。
    pub fn de_register_image(&mut self, image: usize) {
        if !self.images[image].registered {
            return;
        }
        let observations: Vec<(usize, TrackElement)> = self
            .points
            .iter()
            .enumerate()
            .filter(|(_, point)| point.track.iter().any(|el| el.image_id == self.images[image].id))
            .map(|(index, point)| {
                (
                    index,
                    point
                        .track
                        .iter()
                        .find(|el| el.image_id == self.images[image].id)
                        .copied()
                        .expect("过滤条件保证存在"),
                )
            })
            .collect();
        for (point, element) in observations {
            if self.points[point].track.len() <= 2 {
                self.delete_point3d(point);
            } else {
                self.delete_observation(point, element);
            }
        }
        self.images[image].registered = false;
        self.images[image].qvec = [1.0, 0.0, 0.0, 0.0];
        self.images[image].tvec = [0.0; 3];
        self.registered.retain(|item| *item != image);
    }

    pub fn compute_num_observations(&self) -> usize {
        self.points.iter().map(|point| point.track.len()).sum()
    }

    pub fn projection_center_of(&self, image: usize) -> [f64; 3] {
        projection_center(&self.images[image].qvec, &self.images[image].tvec)
    }

    /// 4.2.1 `UpdatePoint3DErrors`：逐点平均像素重投影误差。
    pub fn update_point3d_errors(&mut self) {
        // GPU（`ooo-sfm-metal` feature，默认关闭）：先把**所有**观测的投影一次性
        // 算完（`apple/gpu.rs` 的 `project_kernel`），再按 track 归并。观测数不够
        // 一个 dispatch 时内核直接返回 `None`，自动落回下面的 CPU 循环。
        let mut batch = super::gpu::ProjectionBatch::new();
        let mut owner: Vec<usize> = Vec::new();
        let mut observed: Vec<[f64; 2]> = Vec::new();
        for (index, point) in self.points.iter().enumerate() {
            if point.track.is_empty() {
                continue;
            }
            for element in &point.track {
                let Some(&image) = self.image_index.get(&element.image_id) else {
                    continue;
                };
                let observation = self.images[image].points2d[element.point2d_idx as usize];
                let rotation =
                    super::linalg::quaternion_to_rotation_matrix(&self.images[image].qvec);
                batch.push(
                    self.model_of(image),
                    &self.camera_of(image).params,
                    &rotation,
                    &self.images[image].tvec,
                    point.xyz,
                );
                owner.push(index);
                observed.push([observation.x, observation.y]);
            }
        }
        let projections = super::gpu::project_points_or(&batch, |slot| {
            let rotation = super::linalg::quaternion_to_rotation_matrix(&self.images[
                self.image_index[&self.points[owner[slot]].track[slot].image_id]
            ].qvec);
            super::camera::world_to_img(
                &self.camera_of(self.image_index[&self.points[owner[slot]].track[slot].image_id]),
                self.model_of(self.image_index[&self.points[owner[slot]].track[slot].image_id]),
                &self.images[self.image_index[&self.points[owner[slot]].track[slot].image_id]].qvec,
                &self.images[self.image_index[&self.points[owner[slot]].track[slot].image_id]].tvec,
                self.points[owner[slot]].xyz,
            )
            .map(|projection| {
                let _ = rotation;
                projection
            })
        });
        if !owner.is_empty() {
            let mut sums = vec![0.0f64; self.points.len()];
            let mut counts = vec![0usize; self.points.len()];
            for (slot, &point) in owner.iter().enumerate() {
                // 投影失败（深度不足）按 4.2.1 的 `double::max()` 语义计入。
                let error = match projections[slot] {
                    Some(projection) => {
                        let dx = projection[0] - observed[slot][0];
                        let dy = projection[1] - observed[slot][1];
                        (dx * dx + dy * dy).sqrt()
                    }
                    None => f64::MAX,
                };
                sums[point] += error;
                counts[point] += 1;
            }
            for index in 0..self.points.len() {
                if counts[index] > 0 {
                    self.points[index].error = sums[index] / counts[index] as f64;
                } else {
                    self.points[index].error = 0.0;
                }
            }
        }
    }

    /// 4.2.1 `Reconstruction::Normalize`（`use_images = true`，extent 10，
    /// 百分位 0.1 / 0.9）：返回 `(scale, centroid)`。
    pub fn normalize(&mut self, extent: f64) -> Option<(f64, [f64; 3])> {
        if self.registered.len() < 2 {
            return None;
        }
        let mut coords: Vec<[f64; 3]> = self
            .registered
            .iter()
            .map(|index| self.projection_center_of(*index))
            .collect();
        coords.sort_by(|a, b| a.partial_cmp(b).expect("投影中心有限"));
        let percentile = |values: Vec<f64>, fraction: f64| -> f64 {
            if values.is_empty() {
                return 0.0;
            }
            let index = ((values.len() - 1) as f64 * fraction).round() as usize;
            values[index]
        };
        let mut lower = [0.0; 3];
        let mut upper = [0.0; 3];
        let mut centroid = [0.0; 3];
        for axis in 0..3 {
            let values: Vec<f64> = coords.iter().map(|point| point[axis]).collect();
            lower[axis] = percentile(values.clone(), 0.1);
            upper[axis] = percentile(values, 0.9);
            centroid[axis] = 0.5 * (lower[axis] + upper[axis]);
        }
        let diagonal = ((upper[0] - lower[0]).powi(2)
            + (upper[1] - lower[1]).powi(2)
            + (upper[2] - lower[2]).powi(2))
        .sqrt();
        let scale = if diagonal >= f64::EPSILON {
            extent / diagonal
        } else {
            1.0
        };
        // 位姿：4.2.1 `TransformCameraWorld` 的 Sim3d 合成
        // `R' = R`、`t' = s·(t_sim3 + R·t)`，其中 `t_sim3 = −s·centroid`…
        // 直接写清楚：`X_world' = s·(X_world − centroid)` ⟹ `t' = s·(R·centroid + t)`。
        // ⚠️ 漏掉 `R·centroid` 会把整个重建平移错位（症状：归一化后重投影误差爆炸、
        // 过滤把点全删光）。
        for index in 0..self.images.len() {
            if !self.images[index].registered {
                continue;
            }
            let rotation =
                super::linalg::quaternion_to_rotation_matrix(&self.images[index].qvec);
            let rotated_centroid = [
                rotation[0][0] * centroid[0] + rotation[0][1] * centroid[1] + rotation[0][2] * centroid[2],
                rotation[1][0] * centroid[0] + rotation[1][1] * centroid[1] + rotation[1][2] * centroid[2],
                rotation[2][0] * centroid[0] + rotation[2][1] * centroid[1] + rotation[2][2] * centroid[2],
            ];
            for axis in 0..3 {
                self.images[index].tvec[axis] =
                    scale * (self.images[index].tvec[axis] + rotated_centroid[axis]);
            }
        }
        for point in &mut self.points {
            for axis in 0..3 {
                point.xyz[axis] = scale * (point.xyz[axis] - centroid[axis]);
            }
        }
        Some((scale, centroid))
    }

    /// 导出 `cameras.bin` / `images.bin` / `points3D.bin` 用的快照。
    pub fn export_model(&self) -> (Vec<Camera>, Vec<Image>, Vec<Point3D>) {
        let cameras = self.cameras.clone();
        let images = self
            .registered
            .iter()
            .map(|index| {
                let image = &self.images[*index];
                Image {
                    id: image.id,
                    qvec: image.qvec,
                    tvec: image.tvec,
                    camera_id: self.cameras[image.camera].id,
                    name: image.name.clone(),
                    points2d: image.points2d.clone(),
                }
            })
            .collect();
        let points = self
            .points
            .iter()
            .map(|point| {
                let mut point = point.clone();
                point.track.sort_by_key(|element| (element.image_id, element.point2d_idx));
                point
            })
            .collect();
        (cameras, images, points)
    }
}

/// `point3d_id` → 稠密下标（`INVALID_POINT3D_ID` → `None`）。
pub fn point_index_of(point3d_id: u64) -> Option<usize> {
    if point3d_id == INVALID_POINT3D_ID || point3d_id == 0 {
        None
    } else {
        Some((point3d_id - 1) as usize)
    }
}

// ---------------------------------------------------------------- 观测统计

/// 单图统计（4.2.1 `ObservationManager::ImageStat`）。
#[derive(Clone, Debug)]
pub struct ImageStats {
    /// 至少有一个对应点的特征点数。
    pub num_observations: usize,
    /// 对应点总数。
    pub num_correspondences: usize,
    /// 对应点所在图像已注册的对应数（structure-less 打分用）。
    pub num_visible_correspondences: usize,
    /// 至少一个对应点带三维点的特征点数（`FindNextImages` 与 PnP 门槛）。
    pub num_visible_points3D: usize,
    /// 每个特征点有多少对应点已带三维点。
    pub num_correspondences_have_point3D: Vec<usize>,
    pyramid: VisibilityPyramid,
}

impl ImageStats {
    fn new(num_points2d: usize, width: usize, height: usize) -> Self {
        Self {
            num_observations: 0,
            num_correspondences: 0,
            num_visible_correspondences: 0,
            num_visible_points3D: 0,
            num_correspondences_have_point3D: vec![0; num_points2d],
            pyramid: VisibilityPyramid::new(width, height, 6),
        }
    }
}

/// 观测统计（4.2.1 `ObservationManager`；与 `Reconstruction` 分离以避免借用冲突）。
#[derive(Clone, Debug, Default)]
pub struct ObservationManager {
    pub images: Vec<ImageStats>,
    pub pairs: HashMap<(usize, usize), PairStats>,
}

impl ObservationManager {
    /// 4.2.1 `ObservationManager::Observer` 构造：统计对应总数、已注册图像的可见性。
    pub fn new(graph: &CorrespondenceGraph, recon: &Reconstruction) -> Self {
        let mut manager = Self {
            images: (0..recon.images.len())
                .map(|image| {
                    let camera = recon.camera_of(image);
                    ImageStats::new(
                        recon.images[image].points2d.len(),
                        camera.width as usize,
                        camera.height as usize,
                    )
                })
                .collect(),
            pairs: HashMap::new(),
        };
        for ((image1, image2), count) in graph.pair_correspondence_counts() {
            manager.pairs.insert(
                (image1, image2),
                PairStats {
                    num_total_corrs: count,
                    num_tri_corrs: 0,
                },
            );
        }
        for image in 0..recon.images.len() {
            for idx in 0..recon.images[image].points2d.len() {
                let count = graph.find(image, idx as u32).len();
                if count == 0 {
                    continue;
                }
                manager.images[image].num_observations += 1;
                manager.images[image].num_correspondences += count;
                for corr in graph.find(image, idx as u32) {
                    if recon.images[corr.image].registered {
                        manager.images[corr.image].num_visible_correspondences += 1;
                    }
                }
                if recon.point_index_at(image, idx as u32).is_some() {
                    manager.increment_corr_has_point3D(graph, recon, image, idx as u32);
                }
            }
        }
        manager
    }

    fn pair_key(image1: usize, image2: usize) -> (usize, usize) {
        if image1 < image2 {
            (image1, image2)
        } else {
            (image2, image1)
        }
    }

    /// 4.2.1 `IncrementCorrespondenceHasPoint3D`。
    fn increment_corr_has_point3D(
        &mut self,
        graph: &CorrespondenceGraph,
        recon: &Reconstruction,
        image: usize,
        point2d_idx: u32,
    ) {
        let point2d = &recon.images[image].points2d[point2d_idx as usize];
        let stats = &mut self.images[image].num_correspondences_have_point3D
            [point2d_idx as usize];
        *stats += 1;
        if *stats == 1 {
            self.images[image].num_visible_points3D += 1;
            self.images[image].pyramid.set_point(point2d.x, point2d.y);
        }
        let _ = graph;
    }

    /// 4.2.1 `DecrementCorrespondenceHasPoint3D`。
    fn decrement_corr_has_point3D(&mut self, recon: &Reconstruction, image: usize, point2d_idx: u32) {
        let point2d = &recon.images[image].points2d[point2d_idx as usize];
        let stats = &mut self.images[image].num_correspondences_have_point3D
            [point2d_idx as usize];
        *stats = stats.saturating_sub(1);
        if *stats == 0 {
            self.images[image].num_visible_points3D =
                self.images[image].num_visible_points3D.saturating_sub(1);
            self.images[image].pyramid.reset_point(point2d.x, point2d.y);
        }
    }

    /// 4.2.1 `SetObservationAsTriangulated`。
    pub fn set_observation_as_triangulated(
        &mut self,
        graph: &CorrespondenceGraph,
        recon: &Reconstruction,
        image: usize,
        point2d_idx: u32,
        is_continued: bool,
    ) {
        let point3d_id = recon.images[image].points2d[point2d_idx as usize].point3d_id;
        for index in 0..graph.find(image, point2d_idx).len() {
            let corr = graph.find(image, point2d_idx)[index];
            self.increment_corr_has_point3D(graph, recon, corr.image, corr.point2d_idx);
            let corr_point3d_id =
                recon.images[corr.image].points2d[corr.point2d_idx as usize].point3d_id;
            if corr_point3d_id == point3d_id
                && (is_continued || image < corr.image)
                && point3d_id != INVALID_POINT3D_ID
            {
                let key = Self::pair_key(image, corr.image);
                if let Some(stats) = self.pairs.get_mut(&key) {
                    stats.num_tri_corrs += 1;
                }
            }
        }
    }

    /// 4.2.1 `ResetTriObservations`。
    pub fn reset_tri_observations(
        &mut self,
        graph: &CorrespondenceGraph,
        recon: &Reconstruction,
        image: usize,
        point2d_idx: u32,
        is_deleted: bool,
    ) {
        let point3d_id = recon.images[image].points2d[point2d_idx as usize].point3d_id;
        for index in 0..graph.find(image, point2d_idx).len() {
            let corr = graph.find(image, point2d_idx)[index];
            self.decrement_corr_has_point3D(recon, corr.image, corr.point2d_idx);
            let corr_point3d_id =
                recon.images[corr.image].points2d[corr.point2d_idx as usize].point3d_id;
            if corr_point3d_id == point3d_id
                && (!is_deleted || image < corr.image)
                && point3d_id != INVALID_POINT3D_ID
            {
                let key = Self::pair_key(image, corr.image);
                if let Some(stats) = self.pairs.get_mut(&key) {
                    stats.num_tri_corrs = stats.num_tri_corrs.saturating_sub(1);
                }
            }
        }
    }

    pub fn add_point3d(
        &mut self,
        graph: &CorrespondenceGraph,
        recon: &mut Reconstruction,
        xyz: [f64; 3],
        color: [u8; 3],
        track: Vec<TrackElement>,
    ) -> usize {
        let index = recon.add_point3d(xyz, color, track.clone());
        for element in &track {
            let image = recon.image_index[&element.image_id];
            self.set_observation_as_triangulated(graph, recon, image, element.point2d_idx, false);
        }
        index
    }

    pub fn delete_point3d(&mut self, graph: &CorrespondenceGraph, recon: &mut Reconstruction, point: usize) {
        let track = recon.points[point].track.clone();
        for element in &track {
            let image = recon.image_index[&element.image_id];
            self.reset_tri_observations(graph, recon, image, element.point2d_idx, true);
        }
        recon.delete_point3d(point);
    }

    /// 4.2.1 `AddObservation`（track 长度 ≤ 2 时直接删点）。
    pub fn add_observation(
        &mut self,
        graph: &CorrespondenceGraph,
        recon: &mut Reconstruction,
        point: usize,
        element: TrackElement,
    ) -> bool {
        if !recon.add_observation(point, element) {
            return false;
        }
        let image = recon.image_index[&element.image_id];
        self.set_observation_as_triangulated(graph, recon, image, element.point2d_idx, true);
        true
    }

    /// 4.2.1 `DeleteObservation`（track 长度 ≤ 2 时删整点）。
    pub fn delete_observation(
        &mut self,
        graph: &CorrespondenceGraph,
        recon: &mut Reconstruction,
        point: usize,
        element: TrackElement,
    ) {
        if recon.points[point].track.len() <= 2 {
            self.delete_point3d(graph, recon, point);
            return;
        }
        let image = recon.image_index[&element.image_id];
        self.reset_tri_observations(graph, recon, image, element.point2d_idx, false);
        recon.delete_observation(point, element);
    }

    pub fn merge_points3d(
        &mut self,
        graph: &CorrespondenceGraph,
        recon: &mut Reconstruction,
        point1: usize,
        point2: usize,
    ) -> Option<usize> {
        let track1 = recon.points[point1].track.clone();
        let track2 = recon.points[point2].track.clone();
        for element in track1.iter().chain(track2.iter()) {
            let image = recon.image_index[&element.image_id];
            self.reset_tri_observations(graph, recon, image, element.point2d_idx, true);
        }
        let merged = recon.merge_points3d(point1, point2)?;
        let track = recon.points[merged].track.clone();
        for element in &track {
            let image = recon.image_index[&element.image_id];
            self.set_observation_as_triangulated(graph, recon, image, element.point2d_idx, false);
        }
        Some(merged)
    }

    /// 4.2.1 `RegisterFrame`（单传感器 rig 下等于 register image）。
    pub fn register_image(&mut self, graph: &CorrespondenceGraph, recon: &mut Reconstruction, image: usize, qvec: [f64; 4], tvec: [f64; 3]) {
        for idx in 0..recon.images[image].points2d.len() {
            for index in 0..graph.find(image, idx as u32).len() {
                let corr = graph.find(image, idx as u32)[index];
                self.images[corr.image].num_visible_correspondences += 1;
            }
        }
        recon.register_image(image, qvec, tvec);
    }

    /// 4.2.1 `DeRegisterFrame`。
    pub fn de_register_image(&mut self, graph: &CorrespondenceGraph, recon: &mut Reconstruction, image: usize) {
        for idx in 0..recon.images[image].points2d.len() {
            for index in 0..graph.find(image, idx as u32).len() {
                let corr = graph.find(image, idx as u32)[index];
                self.images[corr.image].num_visible_correspondences =
                    self.images[corr.image].num_visible_correspondences.saturating_sub(1);
            }
        }
        // 删除该图像在所有三维点里的观测（track ≤ 2 时连带删点）。
        for idx in 0..recon.images[image].points2d.len() {
            let Some(point) = recon.point_index_at(image, idx as u32) else {
                continue;
            };
            let Some(element) = recon.points[point]
                .track
                .iter()
                .find(|el| el.image_id == recon.images[image].id)
                .copied()
            else {
                continue;
            };
            self.delete_observation(graph, recon, point, element);
        }
        recon.de_register_image(image);
    }

    pub fn num_observations(&self, image: usize) -> usize {
        self.images[image].num_observations
    }

    pub fn num_correspondences(&self, image: usize) -> usize {
        self.images[image].num_correspondences
    }

    pub fn num_visible_correspondences(&self, image: usize) -> usize {
        self.images[image].num_visible_correspondences
    }

    pub fn num_visible_points3D(&self, image: usize) -> usize {
        self.images[image].num_visible_points3D
    }

    /// 4.2.1 `Point3DVisibilityScore`（越小分布越均匀；`MIN_UNCERTAINTY` 排序用）。
    pub fn point3d_visibility_score(&self, image: usize) -> usize {
        self.images[image].pyramid.score()
    }
}

// ---------------------------------------------------------------- 过滤

/// 4.2.1 `FilterPoints3DWithLargeReprojectionError`。
pub fn filter_points_with_large_reprojection_error(
    graph: &CorrespondenceGraph,
    recon: &mut Reconstruction,
    obs: &mut ObservationManager,
    max_error: f64,
    point_ids: &[usize],
) -> usize {
    let mut filtered = 0usize;
    // 逆序遍历：删除采用末尾交换，避免下标漂移。
    for index in (0..point_ids.len()).rev() {
        let point = point_ids[index];
        if point >= recon.points.len() {
            continue;
        }
        let track = recon.points[point].track.clone();
        if track.len() < 2 {
            filtered += track.len();
            obs.delete_point3d(graph, recon, point);
            continue;
        }
        let mut error_sum = 0.0;
        let mut to_delete: Vec<TrackElement> = Vec::new();
        for element in &track {
            let image = recon.image_index[&element.image_id];
            let observation = recon.images[image].points2d[element.point2d_idx as usize];
            let error = camera::reprojection_error(
                recon.camera_of(image),
                recon.model_of(image),
                &recon.images[image].qvec,
                &recon.images[image].tvec,
                recon.points[point].xyz,
                [observation.x, observation.y],
            );
            if error > max_error {
                to_delete.push(*element);
            } else {
                error_sum += error;
            }
        }
        if to_delete.len() + 1 >= track.len() {
            filtered += track.len();
            obs.delete_point3d(graph, recon, point);
        } else if !to_delete.is_empty() {
            let track_len = track.len() as f64;
            for element in to_delete {
                filtered += 1;
                obs.delete_observation(graph, recon, point, element);
            }
            // `delete_observation` 可能删掉整个点（长度 ≤ 2），此时写 error 会越界。
            if point < recon.points.len() && recon.points[point].id == point as u64 + 1 {
                recon.points[point].error = error_sum / track_len;
            }
        }
    }
    filtered
}

/// 4.2.1 `FilterPoints3DWithSmallTriangulationAngle`。
pub fn filter_points_with_small_triangulation_angle(
    graph: &CorrespondenceGraph,
    recon: &mut Reconstruction,
    obs: &mut ObservationManager,
    min_tri_angle_degrees: f64,
    point_ids: &[usize],
) -> usize {
    let min_angle = min_tri_angle_degrees.to_radians();
    let mut filtered = 0usize;
    for index in (0..point_ids.len()).rev() {
        let point = point_ids[index];
        if point >= recon.points.len() {
            continue;
        }
        let xyz = recon.points[point].xyz;
        let track = recon.points[point].track.clone();
        let mut keep = false;
        'outer: for a in 0..track.len() {
            let image_a = recon.image_index[&track[a].image_id];
            let center_a = recon.projection_center_of(image_a);
            for b in 0..a {
                let image_b = recon.image_index[&track[b].image_id];
                let center_b = recon.projection_center_of(image_b);
                if super::triangulator::triangulation_angle(center_a, center_b, xyz) >= min_angle {
                    keep = true;
                    break 'outer;
                }
            }
        }
        if !keep {
            filtered += track.len();
            obs.delete_point3d(graph, recon, point);
        }
    }
    filtered
}

/// 4.2.1 `FilterAllPoints3D`（**先重投影误差、后三角角**，顺序有意义）。
pub fn filter_all_points3d(
    graph: &CorrespondenceGraph,
    recon: &mut Reconstruction,
    obs: &mut ObservationManager,
    max_reprojection_error: f64,
    min_tri_angle_degrees: f64,
) -> usize {
    let point_ids: Vec<usize> = (0..recon.points.len()).collect();
    let mut filtered = filter_points_with_large_reprojection_error(
        graph,
        recon,
        obs,
        max_reprojection_error,
        &point_ids,
    );
    // 第一轮删除会让点数组收缩，重新收集存活点。
    let point_ids: Vec<usize> = (0..recon.points.len()).collect();
    filtered += filter_points_with_small_triangulation_angle(
        graph,
        recon,
        obs,
        min_tri_angle_degrees,
        &point_ids,
    );
    filtered
}

/// 4.2.1 `FilterPoints3DInImages`：只过滤被这些图像观测到的点。
pub fn filter_points3d_in_images(
    graph: &CorrespondenceGraph,
    recon: &mut Reconstruction,
    obs: &mut ObservationManager,
    max_reprojection_error: f64,
    min_tri_angle_degrees: f64,
    images: &[usize],
) -> usize {
    let mut ids: Vec<usize> = Vec::new();
    let image_ids: HashSet<u32> = images
        .iter()
        .map(|index| recon.images[*index].id)
        .collect();
    for (point_index, point) in recon.points.iter().enumerate() {
        if point
            .track
            .iter()
            .any(|element| image_ids.contains(&element.image_id))
        {
            ids.push(point_index);
        }
    }
    let mut filtered = filter_points_with_large_reprojection_error(
        graph,
        recon,
        obs,
        max_reprojection_error,
        &ids,
    );
    let surviving: Vec<usize> = ids
        .into_iter()
        .filter(|index| *index < recon.points.len() && recon.points[*index].id == *index as u64 + 1)
        .collect();
    filtered += filter_points_with_small_triangulation_angle(
        graph,
        recon,
        obs,
        min_tri_angle_degrees,
        &surviving,
    );
    filtered
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::colmap::apple::ba::CameraModelId;

    /// 两张图（各 3 个特征点）+ 一条匹配链，构建最小场景。
    fn fixture() -> (CorrespondenceGraph, Reconstruction, ObservationManager) {
        let cameras = vec![Camera::new(1, CameraModelId::SimpleRadial as i32, 1000, 800, vec![800.0, 500.0, 400.0, 0.0])];
        let prior = vec![false];
        let images = vec![
            ImageState {
                id: 1,
                name: "a.jpg".into(),
                camera: 0,
                points2d: (0..3)
                    .map(|index| Point2D::new(100.0 + 200.0 * index as f64, 300.0 + 50.0 * index as f64))
                    .collect(),
                colors: Vec::new(),
                registered: false,
                qvec: [1.0, 0.0, 0.0, 0.0],
                tvec: [0.0; 3],
            },
            ImageState {
                id: 2,
                name: "b.jpg".into(),
                camera: 0,
                points2d: (0..3)
                    .map(|index| Point2D::new(120.0 + 200.0 * index as f64, 310.0 + 50.0 * index as f64))
                    .collect(),
                colors: Vec::new(),
                registered: false,
                qvec: [1.0, 0.0, 0.0, 0.0],
                tvec: [0.0; 3],
            },
        ];
        let mut recon = Reconstruction::new(cameras, prior, images);
        let _ = &mut recon;
        let mut graph = CorrespondenceGraph::new(&[3, 3]);
        // 匹配链：0↔0、1↔1、2↔2，另外 0-2 号点都看到 1 号点（多视图）。
        graph.add_match(0, 0, 1, 0);
        graph.add_match(0, 1, 1, 1);
        graph.add_match(0, 2, 1, 2);
        graph.add_match(0, 0, 1, 2);
        graph.finalize();
        // `Reconstruction::new` 已按 id 建好反查表（id 1/2 → 下标 0/1）。
        let obs = ObservationManager::new(&graph, &recon);
        (graph, recon, obs)
    }

    #[test]
    fn correspondence_graph_indexes_transitively() {
        let (graph, _, _) = fixture();
        assert_eq!(graph.num_correspondences(0), 4);
        assert_eq!(graph.find(0, 0).len(), 2);
        // 两视图观测：0 号点的每个对应点都只看到它自己。
        assert!(!graph.is_two_view_observation(0, 0));
        // 1 号点只与 1 号点匹配 → 两视图观测。
        assert!(graph.is_two_view_observation(0, 1));
        let matches = graph.matches_between(0, 1);
        assert_eq!(matches.len(), 4);
        // 传递闭包：transitivity=1 只返回直接对应。
        let direct = graph.find_transitive(0, 1, 1);
        assert_eq!(direct.len(), 1);
        assert_eq!((direct[0].image, direct[0].point2d_idx), (1, 1));
    }

    #[test]
    fn observation_stats_track_registration_and_triangulation() {
        let (graph, mut recon, mut obs) = fixture();
        assert_eq!(obs.num_observations(0), 3);
        assert_eq!(obs.num_correspondences(0), 4);
        assert_eq!(obs.num_visible_points3D(0), 0);

        // 注册第 2 张图 → 对应点的"可见对应"计数上升。
        obs.register_image(&graph, &mut recon, 1, [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert_eq!(recon.num_reg_images(), 1);
        assert_eq!(obs.num_visible_correspondences(0), 4);

        // 三角化 0 号点（观测：图 0 的点 1 与图 1 的点 2）。
        let point = obs.add_point3d(
            &graph,
            &mut recon,
            [0.0, 0.0, 5.0],
            [10, 20, 30],
            vec![
                TrackElement { image_id: 1, point2d_idx: 1 },
                TrackElement { image_id: 2, point2d_idx: 2 },
            ],
        );
        assert_eq!(recon.points[point].id, 1);
        assert_eq!(recon.point_index_at(0, 1), Some(point));
        // 可见三维点特征数：图 0 的点 1 与点 0（都看到已三角化的点）。
        assert!(obs.num_visible_points3D(0) >= 1);
        assert!(obs.point3d_visibility_score(0) > 0);

        // 撤销：删点后统计归零。
        obs.delete_point3d(&graph, &mut recon, point);
        assert_eq!(recon.num_points3d(), 0);
        assert_eq!(recon.point_index_at(0, 1), None);
        assert_eq!(obs.num_visible_points3D(0), 0);
        assert_eq!(obs.point3d_visibility_score(0), 0);
    }

    #[test]
    fn deleting_a_point_renumbers_the_swapped_last_point() {
        let (graph, mut recon, mut obs) = fixture();
        obs.register_image(&graph, &mut recon, 1, [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        obs.add_point3d(
            &graph,
            &mut recon,
            [0.0, 0.0, 5.0],
            [1, 2, 3],
            vec![
                TrackElement { image_id: 1, point2d_idx: 0 },
                TrackElement { image_id: 2, point2d_idx: 0 },
            ],
        );
        let second = obs.add_point3d(
            &graph,
            &mut recon,
            [1.0, 1.0, 5.0],
            [4, 5, 6],
            vec![
                TrackElement { image_id: 1, point2d_idx: 1 },
                TrackElement { image_id: 2, point2d_idx: 1 },
            ],
        );
        assert_eq!(second, 1);
        // 删除下标 0（非末尾）：下标 1 的点会被交换过来，id/引用必须改成 1。
        obs.delete_point3d(&graph, &mut recon, 0);
        assert_eq!(recon.num_points3d(), 1);
        assert_eq!(recon.points[0].id, 1);
        assert_eq!(recon.point_index_at(0, 1), Some(0));
        assert_eq!(recon.point_index_at(1, 1), Some(0));
        assert_eq!(recon.point_index_at(0, 0), None);
    }

    #[test]
    fn visibility_pyramid_score_drops_for_spread_points() {
        // 均匀分布：分数低于集中分布。
        let mut spread = VisibilityPyramid::new(1000, 800, 6);
        for x in 0..10 {
            spread.set_point(100.0 * x as f64, 80.0 * x as f64);
        }
        let mut clustered = VisibilityPyramid::new(1000, 800, 6);
        for _ in 0..10 {
            clustered.set_point(500.0, 400.0);
        }
        // 分数随"占据的格子数"增长：分布越散越高（4.2.1 用它当 MIN_UNCERTAINTY rank）。
        assert!(spread.score() > clustered.score());
        for x in 0..10 {
            spread.reset_point(100.0 * x as f64, 80.0 * x as f64);
        }
        assert_eq!(spread.score(), 0);
    }

    #[test]
    fn normalize_moves_camera_centers_and_points() {
        let mut cameras = vec![Camera::new(1, CameraModelId::SimpleRadial as i32, 100, 100, vec![100.0, 50.0, 50.0, 0.0])];
        let mut images: Vec<ImageState> = (0..4)
            .map(|index| ImageState {
                id: index as u32 + 1,
                name: format!("{index}.jpg"),
                camera: 0,
                points2d: Vec::new(),
                colors: Vec::new(),
                registered: true,
                qvec: [1.0, 0.0, 0.0, 0.0],
                // 沿 x 轴等距 10 米（相机看向 +z，光心在 x = −t）。
                tvec: [-10.0 * index as f64, 0.0, 0.0],
            })
            .collect();
        let mut recon = Reconstruction::new(std::mem::take(&mut cameras), vec![true], std::mem::take(&mut images));
        let before: Vec<[f64; 3]> = recon
            .registered
            .iter()
            .map(|index| recon.projection_center_of(*index))
            .collect();
        let Some((_scale, _centroid)) = recon.normalize(10.0) else {
            panic!("至少两张已注册图时应能归一化");
        };
        let after: Vec<[f64; 3]> = recon
            .registered
            .iter()
            .map(|index| recon.projection_center_of(*index))
            .collect();
        // 归一化后包围盒对角线约 10。
        let span = (after[3][0] - after[0][0]).abs();
        assert!((span - 10.0).abs() < 1e-6, "跨度 {span}");
        assert!(before[3][0] > before[0][0]);
    }

    #[test]
    fn merge_points_averages_positions_and_unions_tracks() {
        let (graph, mut recon, mut obs) = fixture();
        let a = obs.add_point3d(
            &graph,
            &mut recon,
            [0.0, 0.0, 4.0],
            [255, 0, 0],
            vec![
                TrackElement { image_id: 1, point2d_idx: 0 },
                TrackElement { image_id: 2, point2d_idx: 2 },
            ],
        );
        let b = obs.add_point3d(
            &graph,
            &mut recon,
            [0.0, 0.0, 8.0],
            [0, 0, 0],
            vec![
                TrackElement { image_id: 1, point2d_idx: 1 },
                TrackElement { image_id: 2, point2d_idx: 1 },
            ],
        );
        let merged = obs.merge_points3d(&graph, &mut recon, a, b).unwrap();
        assert_eq!(recon.points.len(), 1);
        assert_eq!(recon.points[merged].xyz, [0.0, 0.0, 6.0]);
        assert_eq!(recon.points[merged].track.len(), 4);
        // 颜色：黑色视为"未提取"，只取非黑。
        assert_eq!(recon.points[merged].color, [255, 0, 0]);
    }
}