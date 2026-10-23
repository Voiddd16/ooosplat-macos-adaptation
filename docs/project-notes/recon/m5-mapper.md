# M5 规格：COLMAP 增量重建 per-image 核心算法（tag `4.2.1` 冻结参考）

## 0. 基线与命名差异（Rust 实现必须对齐本版本，不要照搬旧 COLMAP 文档）
- 源码路径前缀：`src/colmap/`。本文引用写作 `sfm/incremental_mapper.cc:233` 等，仓库根 `engines/colmap/colmap`，取文件用 `git show 4.2.1:<path>`。
- 该版本是 **frame/rig 化**的 mapper：注册粒度是 **frame**（`Reconstruction::RegisterFrame`），`image.HasPose()` 来自 `image.FramePtr()->HasPose()`（`scene/image.h:272`）。trivial rig 时 frame_id == image_id（`scene/reconstruction.cc:519-525`）。
- 不存在的东西（旧版本有、4.2.1 没有；不要在 Rust 里臆造）：`IncrementalMapperOptions`（实际是 `IncrementalMapper::Options`）、`local_window_size`/`local_window_*`、`CheckRegisterImage`、`ObservationManager::TrackLength/MinTrackLength`、`pose_refinement.{h,cc}`（`RefineAbsolutePose` 在 `estimators/pose.{h,cc}`）、`use_inner_iterations`。
- `tri_*` 参数与 `ignore_two_view_tracks` 属于 `IncrementalTriangulator::Options`（`sfm/incremental_triangulator.h:45`），不属于 mapper options；`min_track_len` 不在 mapper（仅在 `exe/sfm.cc:602` 的 triangulation CLI 与 `BundleAdjustmentOptions::min_track_length=0`，`estimators/bundle_adjustment.h:196`）。
- 参数覆盖链：`IncrementalPipelineOptions` → `Mapper()`/`Triangulation()`/`LocalBundleAdjustment()`/`GlobalBundleAdjustment()`（`controllers/incremental_pipeline.cc:169-287`）。CLI 名字映射见 `controllers/option_manager.cc:725-793`（`Mapper.tri_*`, `Mapper.abs_pose_*` 等）。

## 1. `IncrementalMapper::Options` 全字段与默认值（`sfm/incremental_mapper.h:68-171`）
| 字段 | 默认 | 行 |
|---|---|---|
| `init_min_num_inliers` | 100 | :70 |
| `init_max_error` | 4.0 | :74 |
| `init_max_forward_motion` | 0.95 | :77 |
| `init_min_tri_angle` | 16.0 (deg) | :80 |
| `init_max_reg_trials` | 2 | :83 |
| `abs_pose_max_error` | 12.0 (px) | :86 |
| `abs_pose_min_num_inliers` | 30 | :89 |
| `abs_pose_min_inlier_ratio` | 0.25 | :92 |
| `abs_pose_refine_focal_length` | true | :95 |
| `abs_pose_refine_extra_params` | true | :98 |
| `ba_local_num_images` | 6 | :101 |
| `ba_local_min_tri_angle` | 6 (deg) | :104 |
| `ba_global_ignore_redundant_points3D` | false | :113 |
| `ba_global_ignore_redundant_points3D_min_coverage_gain` | 0.05 | :117 |
| `min_focal_length_ratio` | 0.1 (~130° FOV) | :121 |
| `max_focal_length_ratio` | 10 (~5° FOV) | :122 |
| `max_extra_param` | 1 | :123 |
| `filter_max_reproj_error` | 4.0 (px) | :126 |
| `filter_min_tri_angle` | 1.5 (deg) | :129 |
| `max_reg_trials` | 3 | :132 |
| `fix_existing_frames` | false | :135 |
| `constant_rigs` / `constant_cameras` | 空集合 | :139,:143 |
| `use_prior_position` / `use_robust_loss_on_prior_position` | false / false | :146,:149 |
| `prior_position_loss_scale` | 7.815 (χ²3DOF@95%) | :153 |
| `num_threads` / `random_seed` | -1 / -1 | :156,:159 |
| `image_selection_method` | `MIN_UNCERTAINTY`（另 MAX_VISIBLE_POINTS_NUM / RATIO） | :162-168 |
- `Check()`（`sfm/incremental_mapper.cc:77-100`）：`init_min_num_inliers>0, init_max_error>0, 0<=init_max_forward_motion<=1, init_min_tri_angle>=0, init_max_reg_trials>=1, abs_pose_max_error>0, abs_pose_min_num_inliers>0, 0<=abs_pose_min_inlier_ratio<=1, ba_local_num_images>=2, ba_local_min_tri_angle>=0, ba_global_...gain>=0, min_focal_length_ratio>=0, max_focal_length_ratio>=min, max_extra_param>=0, filter_max_reproj_error>=0, filter_min_tri_angle>=0, max_reg_trials>=1, num_threads>=-1, random_seed>=-1`。
- 统计/报告结构：`LocalBundleAdjustmentReport{num_merged_observations, num_completed_observations, num_filtered_observations, num_adjusted_observations}`（:173-178）；`RegistrationStatistics`：`num_total_reg_images, num_shared_reg_images, init_num_reg_trials, init_image_pairs, num_reg_frames_per_rig, num_reg_images_per_camera, num_registrations, num_reg_trials, num_structure_less_reg_trials`（:329-356）。`RegisterFrameEvent`（`sfm/incremental_mapper.cc:1443-1465`）：per-frame 计数 +1、per-camera 计数 +1、`num_registrations[image]` +1（首次 → total+1，>1 → shared+1）；`DeRegisterFrameEvent`（:1467-1491）反向。

## 2. ObservationManager（`sfm/observation_manager.{h,cc}`）
- 构造（cc:57-98）：① 从 `correspondence_graph_->NumMatchesBetweenAllImages()` 建 `image_pair_stats_[pair].num_total_corrs`；② 每个 image 建 `ImageStat`（`InitImageStat`, cc:163-179：visibility pyramid 6 级、`camera.width/height`、`num_correspondences_have_point3D.resize(NumPoints2D())`）；③ 已注册 image 中 `HasPoint3D()` 的 point2D → `SetObservationAsTriangulated(false)`；④ 同时给每个 corr 的 `num_visible_correspondences += 1`（cc:89-96）。
- `ImageStat`（h:201-223）：`num_observations`(有≥1对应点的点数)、`num_correspondences`(对应总数)、`num_visible_correspondences`(对应点所在 image 已注册)、`num_visible_points3D`(该 image 的 point2D 至少有一个对应点带 3D)、`num_correspondences_have_point3D[per-point]`、`point3D_visibility_pyramid`。`ImagePairStat`（h:55-60）：`num_tri_corrs`（该 image pair 已三角化对应数）、`num_total_corrs`。
- 查询（h:252-272）：`NumObservations` = `num_observations`；`NumCorrespondences`；`NumVisibleCorrespondences`；`NumVisiblePoints3D` = `num_visible_points3D`；**`Point3DVisibilityScore(image)` = visibility pyramid 的 `score_`（即 `GetScore`）**。
- Score 精确公式（`scene/visibility_pyramid.{h,cc}`）：级别 ℓ∈[0,L)，L=6，dim_ℓ=2^(ℓ+1)；`max_score = Σ dim_ℓ⁴`。`CellForPoint(x,y)`：`max_dim=2^L=64`，`cx=clamp(64x/width,0,63)`, `cy=clamp(64y/height,0,63)`（cc:96-105）。`SetPoint`：从最高分辨率级向下 `cx>>=1, cy>>=1`，每级 `level(cy,cx)+=1`，若变 1 → `score_ += level.size() = dim_ℓ²`（cc:52-72）；`ResetPoint` 对称（cc:74-94）。`Point3DVisibilityScore` 返回未归一化的 `size_t`，越小=分布越差；mapper 里 cast 成 `float` 做排序。
- 增量更新（打分依赖，必须逐条复刻）：
  - `IncrementCorrespondenceHasPoint3D(image, idx)`（cc:181-195）：`num_correspondences_have_point3D[idx]+=1`；`0→1` 时 `num_visible_points3D+=1`；`pyramid.SetPoint(xy)`；断言 `num_visible_points3D<=num_observations`。
  - `DecrementCorrespondenceHasPoint3D`（cc:197-214）：`-=1`；`1→0` 时 `num_visible_points3D-=1`；`pyramid.ResetPoint(xy)`。
  - `SetObservationAsTriangulated(image,idx,is_continued)`（cc:216-247）：对每个 corr：`IncrementCorrespondenceHasPoint3D(corr.image, corr.point2D_idx)`；若 **双方 point3D_id 相同** 且 `(is_continued || image_id < corr.image_id)` → `image_pair_stats[pair].num_tri_corrs += 1`（断言 ≤ num_total_corrs）。`ResetTriObservations(...,is_deleted)`（cc:249-276）完全镜像，条件为 `(!is_deleted || image_id < corr.image_id)`。
  - `AddPoint3D(xyz, track)`（cc:278-290）：`reconstruction.AddPoint3D` → 对 track 每元素 `SetObservationAsTriangulated(false)`。`AddObservation(id, el)`（cc:292-297）：reconstruction 后 `Set(true)`。
  - `DeletePoint3D`（cc:299-309）：先对 track 每元素 `ResetTriObservations(deleted=true)`，再 `reconstruction.DeletePoint3D`。`DeleteObservation`（cc:311-326）：`track.Length()<=2` → 整点删除；否则 `ResetTriObservations(deleted=false)` + `reconstruction.DeleteObservation`。
  - `MergePoints3D(a,b)`（cc:328-351）：对两条 track 各 `ResetTriObservations(deleted=true)` → `reconstruction.MergePoints3D` → 对合并后 track `SetObservationAsTriangulated(false)`；`modified` 集合由 triangulator 侧维护。
  - `RegisterFrame`（cc:587-603）：**先**对所有 image 的每个 point2D 的每个 corr `image_stats_[corr.image].num_visible_correspondences += 1`，**再** `reconstruction.RegisterFrame`。`DeRegisterFrame`（cc:605-628）：先对所有 point2D 的 corr `-=1`，再对 `HasPoint3D()` 的 point2D 逐个 `DeleteObservation`，最后 `reconstruction.DeRegisterFrame`。
- 过滤 API：`FilterPoints3D(max_reproj_error,min_tri_angle,ids)` = 大重投影 + 小三角角（cc:353-363）；`FilterPoints3DInImages(...,image_ids)`（cc:365-379，先把这些 image 的 point3D 收集成集合）；`FilterAllPoints3D`（cc:381-393，注释强调**顺序**：先 reproj 后 tri angle）；`FilterPoints3DWithShortTracks(min_len)`（cc:395-407）；`FilterObservationsWithNegativeDepth`（cc:409-433，跳过 `IsSpherical`，用 `HasPointPositiveDepth`）；`FilterPoints3DWithSmallTriangulationAngle`（cc:435-494，`DegToRad`，任一对 `tri_angle>=min` 即保留，否则删整点，计数 += `track.Length()`）；`FilterPoints3DWithLargeReprojectionError`（cc:496-585，见 §12 投影公式）；`FindFramesToFilter(ratios,max_extra,min_num_observations)`（cc:630-655，bogus 相机或观测数 < 阈值即列入）。

## 3. `FindNextImages` / 初始化对选取
- `FindNextImages`（`sfm/incremental_mapper.cc:182-192` → impl `sfm/incremental_mapper_impl.cc:294-364`）：
  1. 选 rank 函数（impl:303-324）：
     - structure_less=true → `NumVisibleCorrespondences(image)`（:304-309）；
     - `MAX_VISIBLE_POINTS_NUM` → `float(NumVisiblePoints3D)`；`MAX_VISIBLE_POINTS_RATIO` → `NumVisiblePoints3D/NumObservations`；`MIN_UNCERTAINTY` → `float(Point3DVisibilityScore)`（impl:86-100）。**注意语义：MIN_UNCERTAINTY 用"越小越好"的分布分数当"越大越好"的 rank，降序排序后该类图像最后被尝试**。
  2. 遍历 `reconstruction.Images()`：跳过 `image.HasPose()`；跳过 `NumVisiblePoints3D < abs_pose_min_num_inliers`（size_t 比较）；跳过 `num_reg_trials[image] >= max_reg_trials`（impl:330-346）。
  3. 若 frame 未在 `filtered_frames_` 且 `num_reg_trials==0` → 第一桶 `image_ranks`，否则第二桶 `other_image_ranks`（impl:348-356）。
  4. `SortAndAppendNextImages`：每桶按 rank **降序** `std::sort`（不保证稳定性 → 同分顺序未定义），先 append 第一桶再第二桶（impl:71-84）。
- 计数递增点：`RegisterNextImage` :273、`RegisterNextStructureLessImage` :684（structure-less 用独立计数器 `num_structure_less_reg_trials`）。
- `FindFirstInitialImage`（impl:104-145）：候选需 ① `NumCorrespondencesForImage>0`；② `init_num_reg_trials[image] < init_max_reg_trials(2)`；③ 未在任何先前重建中注册（`num_registrations==0`）；排序比较函数（impl:54-59）：`has_prior_focal_length` 降序，再 `num_correspondences` 降序。
- `FindSecondInitialImage`（impl:147-187）：统计 image1 的每个 point2D 对应到的 image（要求对方 `num_registrations==0`），要求该计数 `>= init_min_num_inliers(100)`，排序同上一比较函数。
- `FindInitialImagePair`（impl:189-292）：多线程（`num_threads`）；每对只试一次（`init_image_pairs` 集合，pair_id = `ImagePairToPairId`）；命中 `stop=true` 后按提交顺序取第一个成功结果（确定性）；找不到 → `nullopt`。可只给 image1 或 image2（impl:204-225）。成功结果 `InitInfo{image_id1,image_id2,cam2_from_cam1,camera1?,camera2?}`（impl.h:70-76），`SeedEstimatedInitialCameras` 把估计出的内参写回 reconstruction（`sfm/incremental_mapper.cc:49-60,174-179`）。

## 4. `RegisterNextImage` 完整伪代码（`sfm/incremental_mapper.cc:233-490`）
1. 前置：`recon.NumRegFrames()>0`；每个 image 的 frame 的 non-ref sensors 必须有已知 `sensor_from_rig`（:243-248）。
2. **广义帧分支**：若 frame `NumSensors()>1` 且该帧所有 image 的相机 `has_prior_focal_length || num_reg_images_per_camera[cam]>0` 且都 `!HasBogusParams` → 调 `RegisterNextGeneralFrame` 并直接返回（:250-271）；否则继续单相机路径（并可能重置 bogus 相机）。
3. `num_reg_trials[image_id] += 1`（:273）。
4. `NumVisiblePoints3D(image) < abs_pose_min_num_inliers` → return false（:276-282）。
5. **2D-3D 收集**（:288-347）：对 `point2D_idx` 遍历 image 全部点：`corr_point3D_ids` 每点清空；对 `correspondence_graph.FindCorrespondences(image_id, idx)` 每个 corr：跳过 `!corr_image.HasPose()`；跳过 `!corr_point2D.HasPoint3D()`；跳过该 point2D 内重复的 `point3D_id`；跳过 `corr_camera.HasBogusParams(min_focal_length_ratio,max_focal_length_ratio,max_extra_param)`；否则 push `(point2D_idx, point3D_id)` 与 `point2D.xy`、`point3D.xyz`。→ **对应来源是 CorrespondenceGraph（间接 2D-2D→3D），点本身已存在于 reconstruction（track 长度不限，≥1 即可贡献）**。
6. `tri_points2D.size() < abs_pose_min_num_inliers` → return false（:341-347）。
7. **绝对位姿选项**（:360-412）：
   - `abs_pose_options.ransac_options.max_error = abs_pose_max_error`；`min_inlier_ratio = abs_pose_min_inlier_ratio`；`random_seed`。
   - `constant_cameras` 含本相机 → `estimate_focal_length=false, refine_focal_length=false, refine_extra_params=false`。
   - 否则 `num_reg_images_per_camera[cam] > 0` 且相机 `HasBogusParams` → `estimate_focal_length = !has_prior_focal_length`、refine 两者 true；`>0` 且不 bogus → 三者 false；`==0` → **先把 `camera.params` 重置为 DB 原值**，`estimate_focal_length=!has_prior_focal_length`、refine 两者 true。
   - 再被 option 覆盖：`!abs_pose_refine_focal_length` → `estimate_focal_length=false, refine_focal_length=false`；`!abs_pose_refine_extra_params` → `refine_extra_params=false`。
   - `!camera.IsPerspective()` → 三者全 false。
   - 之后对本帧所有 image：若相机 `HasBogusParams` → `params = database_cache.Camera(id).params`（:419-429）。
8. `EstimateAbsolutePose(options, points2D, points3D, &cam_from_world, &camera, &num_inliers, &inlier_mask)`（:434-443）。内部（`estimators/pose.cc:46-112`）：`estimate_focal_length=true` → 先减去主点后用 `P4PFEstimator`（共享焦距取决于 `FocalLengthIdxs().size()==1`）纯 RANSAC，成功则把 `focal_lengths[k]` 写入 `camera->params[focal_length_idxs[k]]`；否则 `LORANSAC<P3PEstimator,EPNPEstimator>`，用 `CamRayFromImg` 光线。`AbsolutePoseEstimationOptions` 默认 `max_error=12.0, min_num_trials=100, max_num_trials=10000, confidence=0.99999`（`estimators/pose.h:47-64`）。
9. `num_inliers < abs_pose_min_num_inliers` → return false（:445-449）。**`abs_pose_min_inlier_ratio` 在本版本只喂给 RANSAC（用于最小迭代数/自适应终止），没有"内点数/对应数 < ratio"的后验检查**。
10. `RefineAbsolutePose(refine_options, inlier_mask, points2D, points3D, &cam_from_world, &camera)`（:455-463，`estimators/pose.cc:160-289`）：Ceres，`CauchyLoss(loss_function_scale=1.0)`；`gradient_tolerance=1.0`、`max_num_iterations=100`、`DENSE_QR`、单线程；主点**永远固定**；按 refine 标志把 focal/extra 参数 blocks 设为 constant，否则 `CreateSubsetManifold`；`IsSolutionUsable()==false` → 注册失败。
11. `image.FramePtr()->SetCamFromWorld(image.CameraId(), cam_from_world)`；`obs_manager_->RegisterFrame(frame_id)`；`RegisterFrameEvent(frame_id)`（:472-475）。
12. **继续 track**（:477-487）：对 inlier 且 `!point2D.HasPoint3D()` 的对应：`obs_manager_->AddObservation(point3D_id, TrackElement(image_id, point2D_idx))`；`triangulator_->AddModifiedPoint3D(point3D_id)`。返回 true。
- `RegisterNextGeneralFrame`（:492-669）：收集**跨 sensor** 的 2D-3D + `tri_camera_idxs` + `cams_from_rig`；`EstimateGeneralizedAbsolutePose(RANSACOptions{max_error=abs_pose_max_error,min_inlier_ratio=abs_pose_min_inlier_ratio,random_seed})`（:596-616，内部把 max_error 折算到各相机（`estimators/generalized_pose.cc:83-89,173-174`））；`num_inliers>=abs_pose_min_num_inliers`；`RefineGeneralizedAbsolutePose`（refine 标志全 false）；`frame.SetRigFromWorld(rig_from_world)`；其余同 11-12。
- `RegisterNextStructureLessImage`（:671-949）：需 `NumRegImages>=2`；`min_num_inliers = 2*abs_pose_min_num_inliers`；`NumVisibleCorrespondences` 必须 ≥ 该值；`max_error = 0.5*abs_pose_max_error`（Sampson 误差更严）、`num_threads`；refine 用 `BundleAdjustmentOptions`（CAUCHY + SILENT），相机重置/refine 规则同 §4.7 但无 `estimate_focal_length`；`EstimateStructureLessAbsolutePose`；先尝试把 2D 点挂到已有 track（`AddObservation`），否则 `EstimateTriangulation{min_tri_angle=filter_min_tri_angle, max_error=abs_pose_max_error}`（REPROJECTION 残差）建 `Track` 并用 `obs_manager.AddPoint3D`；最后对该 image 做一次 BA refine（`BundleAdjustmentConfig` 只有该 image + 新点 variable）。

## 5. `EstimateInitialTwoViewGeometry`（仅初始化对使用；注册后的单邻居不走这里）
- 入口：`sfm/incremental_mapper.cc:1493-1511` → impl `sfm/incremental_mapper_impl.cc:654-747`。调用点：`FindInitialImagePair` 内部对每个候选对（impl:266-268），以及用户显式给定 `init_image_id1/2` 时（`controllers/incremental_pipeline.cc:484-489`）。**没有任何"注册新图像后与单邻居做两视图初始化"的路径**；新图像只用 §4 的 P3P/P4PF。
- 步骤：取出 image1/2 全部 `points2D`，`ExtractMatchesBetweenImages` 取匹配（impl:660-679）。
- `TwoViewGeometryOptions`：`ransac_options.min_num_trials=30`、`max_error=init_max_error(4.0)`、`random_seed`（impl:681-684）。其余默认（`estimators/two_view_geometry.h:45-135`）：`min_num_inliers=15, min_inlier_ratio=0.0, min_E_F_inlier_ratio=0.95, max_H_inlier_ratio=0.8, watermark_min_inlier_ratio=0.7, watermark_border_size=0.1, detect_watermark=true, watermark_detection_max_error=4.0, use_degensac=false, use_sampson_refinement=true, multiple_models=false, force_H_use=false`，RANSAC `confidence=0.999, max_num_trials=10000, min_inlier_ratio=0.25`。
- `EstimateTwoViewGeometry`（`estimators/two_view_geometry.cc:618-700`）分派（`IsCameraCalibrated = IsSpherical() || has_prior_focal_length`，:66）：
  1. 恰一侧 calibrated 且另一侧 `IsPerspectivePinhole` → **one-sided focal**（6 点 LO-RANSAC 联合估计焦距，:655-664, 1400+）；
  2. 任一侧 `IsSpherical` → spherical 路径（:665-669）；
  3. 同一 camera_id 且 `!has_prior_focal_length` 且 pinhole → **shared focal**（:670-678；成功后 config=UNCALIBRATED，设 `E`、`F=K⁻ᵀEK⁻¹`（K=diag(f,f,1)+主点），`camera1=camera2=estimated_camera`，:1304-1329）；
  4. 两侧都有焦距先验 → calibrated（E + F 对比，:679-683, 1148-1169）；
  5. 非 pinhole 且无先验 → 标记 degenerate（:684-694）；
  6. 否则 uncalibrated F 路径（:295-400，H 与 F 比较，`H/F > max_H_inlier_ratio` → PLANAR_OR_PANORAMIC）。
- 之后 `EstimateTwoViewGeometryPose`（impl:691-694；实现 `two_view_geometry.cc:1006-1042` → `EstimateTwoViewGeometryPoseFromCamRays` :890-1002）：CALIBRATED 或 `camera1/camera2` 有值 → `PoseFromEssentialMatrix(E, rays)`；UNCALIBRATED → `E = EssentialFromFundamentalMatrix(K2, F, K1)` → `PoseFromEssentialMatrix`；PLANAR/PANORAMIC/PLANAR_OR_PANORAMIC → `PoseFromHomographyMatrix(H, K1, K2, rays, ..., &normal, &points3D)`（spherical 时 K=I），`|t|²<1e-12` → PANORAMIC（并把 `tri_angle` 置 0，:955-964）。`tri_angle` = 存活对应光线夹角的**中位数**（:976-991），homography 路径用 3D 点的三角角中位数（:992-999）。
- 初始化判定（impl:703-709，必须**同时**满足）：`inlier_matches.size() >= init_min_num_inliers(100)`；`|cam2_from_cam1.translation().z()| < init_max_forward_motion(0.95)`；`tri_angle > DegToRad(init_min_tri_angle)(16°)`。
- 非平凡 rig（任一 frame `NumSensors()>1`）：改走 `EstimateInitialGeneralizedTwoViewGeometry`（impl:541-650；`max_error=init_max_error`、`min_num_trials=30`、`EstimateGeneralizedRelativePose`；要求 `num_inliers>=init_min_num_inliers`；重组成 `orig_cam2_from_orig_cam1 = orig_cam2_from_rig2 * rig2_from_rig1 * Inv(orig_cam1_from_rig1)`），**不恢复内参**（impl:735-737）。
- `init_*` 生效点汇总：`init_max_reg_trials`/`init_min_num_inliers` 见 §3；`init_max_error` 见上；`init_min_tri_angle` 限定初始对 2D 场景（另在 pipeline 里作为初始对 triangulation 的 `tri_options.min_angle`，`incremental_pipeline.cc:497-504`）；`init_image_pairs` 保证每对只试一次；总试数 `init_num_trials=200`（`incremental_pipeline.h:78`）＋外层松弛（见 §12）。

## 6. `IncrementalTriangulator`（`sfm/incremental_triangulator.{h,cc}`）
- `Options` 默认（h:45-91）：`max_transitivity=1, create_max_angle_error=2.0, continue_max_angle_error=2.0, merge_max_reproj_error=4.0, complete_max_reproj_error=4.0, complete_max_transitivity=5, re_max_angle_error=5.0, re_min_ratio=0.2, re_max_trials=1, min_angle=1.5, ignore_two_view_tracks=true, min_focal_length_ratio=0.1, max_focal_length_ratio=10.0, max_extra_param=1.0, random_seed=-1`；`Check()`（cc:70-84）。pipeline 会用全局 `min/max_focal_length_ratio, max_extra_param, random_seed` 覆盖（`incremental_pipeline.cc:187-195`）。
- `TriangulateImage(options, image_id)`（cc:99-157）：1) `ClearCaches()`；2) image 无 pose 或相机 bogus → 返回 0；3) 每个 point2D：`Find(max_transitivity)` 收集 `corrs_data`；空 → 跳过；4) `num_triangulated==0` → 把 ref corr 加入后 `Create`；否则先 `Continue(ref, corrs)` 再把 ref 加入后 `Create`（"创建未续上的点"）。
- `Find`（cc:440-479）：`ExtractTransitiveCorrespondences(image,idx,transitivity)`（`scene/correspondence_graph.cc:230-291`：BFS 去重、**结果里剔除起始观测**）；过滤 `!HasPose()`、相机 bogus；同时统计其中已有 point3D 的数量。
- `Create`（cc:481-540）：候选=corrs 中 `!HasPoint3D()` 者；`<2` → 0；`ignore_two_view_tracks && size==2 && CG.IsTwoViewObservation(first)` → 0（两视图 track 定义：双方各自只有 1 个对应，`correspondence_graph.cc:354-363`）；`EstimateTriangulation` 参数：`min_tri_angle=DegToRad(min_angle)`、`residual_type=ANGULAR_ERROR`、`ransac max_error=DegToRad(create_max_angle_error)`、`random_seed`；用 inlier_mask 建 `Track`；`obs_manager.AddPoint3D`；递归：若 `corrs.size() - track_length >= 3`（`kMinRecursiveTrackLength`）则用同一候选集再 `Create` 一次；返回新增观测数。
- `TriangulateTrack`（cc:39-66）：点数 ≤ **15**（`kExhaustiveSamplingThreshold`）时强制 `min_num_trials = NChooseK(n,2)`（穷举）。
- `EstimateTriangulation`（`estimators/triangulation.cc:154-204`）：LORANSAC（minimal=non-minimal=`TriangulationEstimator(min_tri_angle,residual_type)`，`CombinationSampler`）；`EstimateTriangulationOptions` 默认（`estimators/triangulation.h:124-146`）`min_tri_angle=0`、`residual_type=ANGULAR_ERROR`、`max_error=DegToRad(2.0)`、`confidence=0.9999`、`min_inlier_ratio=0.02`、`max_num_trials=10000`。
- 三角化核（DLT）与校验：`estimate`（triangulation.cc:51-129）：2 视图且均为 perspective → `TriangulatePoint(P1,P2,(u1,v1),(u2,v2))`（`geometry/triangulation.cc:40-66`：A 的 4 行 = `u·P.row(2)-P.row(0)` 等，`JacobiSVD(ComputeFullV)`，`V(3,3)==0` 失败，`xyz = V.col(3).hnormalized()`）；否则多视图 `TriangulateMultiViewPoint`（最小化垂距 L2，自伴特征分解取最小特征向量，`geometry/triangulation.cc:143-189`）；**cheirality**：perspective → `HasPointPositiveDepth`（`P.row(2)·[X;1] >= ε`，`scene/projection.cc:137-141`），非 perspective → `(P·X)·cam_ray > 0`；**最小三角角**：存在任一对视点满足 `CalculateTriangulationAngle >= min_tri_angle` 即接受（`geometry/triangulation.cc:217-226`，取 `min(angle, π-angle)`）。残差（triangulation.cc:131-152）：REPROJECTION → 像素平方重投影误差；ANGULAR → 角误差²。
- `Continue`（cc:542-586）：ref 已有 3D → 0；在所有带 3D 的 corr 中挑 `CalculateAngularReprojectionError(ref.xy, point3D.xyz, ref.image.CamFromWorld(), *ref.camera)` **最小**者；若 `<= DegToRad(continue_max_angle_error)` → `AddObservation`，返回 1。
- `CompleteImage`（cc:159-247）：每个 point2D：有 3D → `Complete(point3D_id)`；`ignore_two_view_tracks && IsTwoViewObservation` → 跳过；`Find` 后有已三角化或空 → 跳过；否则用 `TriangulateTrack`（REPROJECTION，`max_error=complete_max_reproj_error`，`min_tri_angle=DegToRad(min_angle)`）→ 建 track（带 inlier mask）→ `obs_manager.AddPoint3D`，计数 += inlier 数。
- `Complete`（cc:684-770）：以当前 track 元素为种子做 BFS（`complete_visited_` 去重），层数 ≤ `complete_max_transitivity(5)`；每步要求 image `HasPose()`、相机不 bogus、point2D `!HasPoint3D()`、`CalculateSquaredReprojectionError <= complete_max_reproj_error²` → `AddObservation`，成功即 +1 并在未达最大层数时入下一层队列。
- `Merge`（cc:588-682）：遍历 track 的所有 corr（对方 image 有 pose、对方 point2D 有 3D、不是自身）；用规范化 `(min,max)` 对去重（`merge_trials_`）；`merged_xyz = (len1·xyz1 + len2·xyz2)/(len1+len2)`；**合并后两条 track 的每个元素**都必须满足 `CalculateSquaredReprojectionError <= merge_max_reproj_error²`，否则放弃；成功则 `obs_manager.MergePoints3D`（内部删除两个旧点、新建点、重建统计）并对新点递归 `Merge`；返回 `len1+len2`。
- `MergeTracks/CompleteTracks(options, point3D_ids)`（cc:249-305）：对给定 id 集合逐个跑 `Merge/Complete`（mapper 局部 BA 用"变动的点集合"）。
- `Retriangulate`（cc:307-406）：`re_options.continue_max_angle_error = re_max_angle_error(5.0)`；遍历 `obs_manager.ImagePairs()`：`tri_ratio = num_tri_corrs/num_total_corrs < re_min_ratio(0.2)`；两 image 都有 pose；`re_num_trials_[pair] < re_max_trials(1)` 后自增；相机不 bogus；`ExtractMatchesBetweenImages`；对每对匹配：两边都有 3D → 跳过（不 merge）；仅一侧有 3D → `Continue(re_options, 另一侧, {有 3D 那侧})`；都没有 → `Create(options, {c1,c2})`（**故意不用 re_options 的角阈值**，注释 cc:396-398）。
- 变动点集合：`AddModifiedPoint3D`（cc:408-410）、`GetModifiedPoints3D`（先剔除已不存在的点，cc:412-428）、`ClearModifiedPoints3D`（cc:430-432）。

## 7. `FilterFrames` / `FilterPoints`（mapper 侧）
- `FilterFrames`（`sfm/incremental_mapper.cc:1339-1372`）：`NumRegFrames() < 20`（`kMinNumFrames`）→ 不筛；`FindFramesToFilter(min_focal_length_ratio, max_focal_length_ratio, max_extra_param, min_num_observations=1)`；对每个入选 frame：若 `!fix_existing_frames || 不在 existing_frame_ids_` → `obs_manager.DeRegisterFrame` + `DeRegisterFrameEvent` + 加入 `filtered_frames_`（后续 `FindNextImages` 降入第二桶）。**注意是 frame 级，不是 image 级。**
- `FilterPoints`（:1374-1381）：`obs_manager.FilterAllPoints3D(filter_max_reproj_error, filter_min_tri_angle)`。
- 观测剔除：逐 track 元素算误差（`^`§12 公式）；`error > filter_max_reproj_error` → 删该观测；若 `#待删 >= track.Length()-1` → 删整点；否则删观测并 `point3D.error = error_sum / point3D.track.Length()`（`observation_manager.cc:565-581`）；track 长度 `<2` 直接删点（cc:509-513）。
- 点剔除：任一对视点 `tri_angle >= DegToRad(filter_min_tri_angle)` 才保留，否则删整点（cc:435-494）；两视图 track 的"忽略"只发生在 triangulator 的 `Create/CompleteImage`（`ignore_two_view_tracks`），过滤阶段没有专门的两视图逻辑；"min_track_len" 在本版 mapper 不存在。

## 8. 局部 BA
- `FindLocalBundle(options, image_id)`（impl:366-537）：共视统计 `shared_observations`（遍历 image 的 point3D 的 track 元素，排除自己）；降序排序；`num_images = ba_local_num_images - 1`（6-1=5）；若共视图像数 == num_eff 直接返回；否则用 8 组递松阈值（impl:438-447）：三角角阈值 `min_tri_angle_rad / {1,1.5,2,2.5,3,4,5,6}`，最少共享观测 `{0.6,0.6,0.5,0.4,0.3,0.2,0.1,0.1} × image.NumPoints3D()`；图像三角角取与共享 3D 点夹角的 **75 分位**（`kTriangulationAnglePercentile=75`，impl:491-495）；不足则按共视数回填（impl:518-534）。
- `AdjustLocalBundle`（`sfm/incremental_mapper.cc:990-1120`）编号步骤：
  1. `local_bundle = FindLocalBundle(...)`（:1004）。
  2. 若非空：`ba_config.FixGauge(THREE_POINTS)`（:1010）；把 image_id 所在 frame 的所有 image 加入 config，再把 local_bundle 每个 image 的 frame 的所有 image 加入（:1013-1025）。
  3. `fix_existing_frames` → 对属于 `existing_frame_ids_` 的 frame `SetConstantRigFromWorldPose`（:1028-1034）。
  4. rig 外参：若 `constant_rigs` 含该 rig 或该 rig 在 bundle 内的 frame 数 `< num_reg_frames_per_rig[rig]` → 对该 rig 的所有 non-ref sensor `SetConstantSensorFromRigPose`（:1037-1051）。
  5. 相机内参：若 `constant_cameras` 含该 camera 或该 camera 在 config 内的 image 数 `< num_reg_images_per_camera[cam]` → `SetConstantCamIntrinsics`（:1054-1065）。
  6. 变量点：对传入的 `point3D_ids`（通常 = `GetModifiedPoints3D()`）：`!point3D.HasError() || track.Length() <= 15`（`kMaxTrackLength=15`）→ `AddVariablePoint`（:1072-1080）。
  7. `CreateDefaultBundleAdjuster(ba_options, config, reconstruction)` → `Solve()`；`report.num_adjusted_observations = summary->num_residuals / 2`（:1085-1089）。
  8. 未停止时（`check_if_stopped`）：`MergeTracks(tri_options, variable_point3D_ids)` → `CompleteTracks(tri_options, variable_point3D_ids)` → `CompleteImage(tri_options, image_id)`（:1091-1105）。
  9. 过滤（**无论 BA 是否运行都执行**）：`FilterPoints3DInImages(filter_max_reproj_error, filter_min_tri_angle, image_ids)` + `FilterPoints3D(..., point3D_ids)`（:1112-1117）。`image_ids` 为空（local_bundle 空）时第一步过滤为空操作。
- `IterativeLocalRefinement`（:1255-1295）：循环 `ba_local_max_refinements=2` 次；`changed = (merged+completed+filtered)/adjusted`（adjusted==0 → 0）；`changed < ba_local_max_refinement_change=0.001` → break；**第一次迭代后把 loss 改为 TRIVIAL**（:1288-1292）；结束 `ClearModifiedPoints3D()`。
- 传给 BA 的 options 由 `IncrementalPipelineOptions::LocalBundleAdjustment()`（`controllers/incremental_pipeline.cc:197-239`）生成：`print_summary=false`、`backend=CERES`、`refine_focal_length=ba_refine_focal_length(true)`、`refine_principal_point=false`、`refine_extra_params=true`、`refine_sensor_from_rig=true`；Ceres：`function_tolerance=0.0`、`gradient_tolerance=10.0`、`parameter_tolerance=0.0`、`max_num_iterations = ba_local_max_num_iterations>=0 ? 该值 : 25`、`max_linear_solver_iterations=100`、`logging=SILENT`、`num_threads`、`min_num_residuals_for_cpu_multi_threading=50000`、`loss_function_type=SOFT_L1`、`loss_function_scale=1.0`、`use_gpu=false`；`check_if_stopped` 由 pipeline 注入。无 `use_inner_iterations` 概念。

## 9. 全局 BA
- `AdjustGlobalBundle(options, ba_options)`（`sfm/incremental_mapper.cc:1122-1253`）：
  1. `custom_ba_options = ba_options`；若 `NumRegFrames() < 10`（`kMinNumRegFramesForFastBA`）：Ceres `function/gradient/parameter_tolerance /= 10`、`max_num_iterations *= 2`、`max_linear_solver_iterations = 200`（:1127-1138）。**该 custom 版本只用于 pose-prior 分支（:1216）；默认分支用的是原 `ba_options`（:1204-1205）——这是参考实现的既有行为，Rust 端要么等价复刻要么显式记录差异。**
  2. **BA 前**先 `obs_manager.FilterObservationsWithNegativeDepth()`（:1141，去掉负深度观测）。
  3. config：所有已注册 frame 的所有 image `AddImage`；`NumImages()<2` → 警告并 return false（:1144-1158）。
  4. `fix_existing_frames` → `SetConstantRigFromWorldPose`（:1161-1167）；`constant_rigs` → non-ref sensor constant；`constant_cameras` → `SetConstantCamIntrinsics`（:1169-1178）。
  5. 冗余点：`!is_small_reconstruction && ba_global_ignore_redundant_points3D` → `FindRedundantPoints3D(min_coverage_gain(0.05), recon)`（`scene/reconstruction_pruning.cc:88-169`：8×8 图像网格，coverage gain = Σ_track `1/√(n)-1/√(n+1)`，贪心取 gain 最大的点直到 gain ≤ 阈值，其余为冗余）→ 逐个 `IgnorePoint`（:1180-1190）。
  6. `use_prior_position && NumRegisteredPosePriors(pose_priors, config) >= 3` → `CreatePosePriorBundleAdjuster(custom_ba_options, prior_options, ...)`，其中 `use_robust_loss_on_prior_position` → CAUCHY，`prior_options.ceres->prior_position_loss_scale = 7.815`，`alignment_ransac_options.random_seed`；否则 `FixGauge(TWO_CAMS_FROM_WORLD)` + `CreateDefaultBundleAdjuster`（:1192-1221）。
  7. 若启用冗余点优化：`Solve()` 后（可 `check_if_stopped` 提前 return true），重建 config：仅冗余点为 variable，所有 frame `SetConstantRigFromWorldPose`、所有 camera 内参 constant、所有 rig 的 non-ref sensor constant，再 `CreateDefaultBundleAdjuster(ba_options,...)` 并 `Solve()`；返回最后一次 `Solve()->IsSolutionUsable()`（:1223-1252）。
- mapper 侧不控制触发频率与迭代次数，那在 driver：`CheckRunGlobalRefinement`（`controllers/incremental_pipeline.cc:543-555`）= `NumRegFrames >= ba_global_frames_ratio(1.1)*prev || NumRegFrames >= ba_global_frames_freq(500)+prev || NumPoints3D >= ba_global_points_ratio(1.1)*prev || NumPoints3D >= ba_global_points_freq(250000)+prev`；`IterativeGlobalRefinement`（mapper:1297-1337）：先 `CompleteAndMergeTracks` + `Retriangulate`，循环 `ba_global_max_refinements=5`：`AdjustGlobalBundle` → `Normalize()`（当 `normalize_reconstruction && !use_prior_position`）→ `CompleteAndMergeTracks` + `FilterPoints`；`changed = num_changed/num_observations(before BA)`，`< ba_global_max_refinement_change=0.0005` → break。BA 后 pipeline 再跑一次 `FilterFrames`（pipeline:96-98）。全局 BA options（pipeline:241-287）：`function_tolerance=ba_global_function_tolerance(0.0)`、`gradient_tolerance=1.0`、`parameter_tolerance=0.0`、`max_num_iterations = ba_global_max_num_iterations>=0 ? 该值 : 50`、`max_linear_solver_iterations=100`、`loss_function_type=TRIVIAL`（无 robust loss）、其余同局部。

## 10. Finalize（模型收尾）
- mapper **没有** `Finalize`/`FilterImages` 方法；收尾在 driver：子模型循环结束后，若 `NumRegFrames()>0 && (NumRegFrames != ba_prev_num_reg_frames) && (NumPoints3D != ba_prev_num_points)` → 再跑一次 `IterativeGlobalRefinement`（`incremental_pipeline.cc:730-737`）。
- 成功：`reconstruction->UpdatePoint3DErrors()`（`scene/reconstruction.cc:968-984`，逐点像素级平均重投影误差，track 为空时 error=0）→ `mapper.EndReconstruction(false)` → `AlignReconstructionToPriorsOrRigScale` → `Callback(LAST_IMAGE_REG_CALLBACK)`（pipeline:825-832）。
- `EndReconstruction(discard)`（mapper:135-152）：`discard=true` 时对每个已注册 frame `obs_manager.DeRegisterFrame` + `DeRegisterFrameEvent`（**同步回退 total/shared 计数**）；随后释放 triangulator/obs_manager 并 `reconstruction_->TearDown()`（删除未注册 image 与未用 camera，`scene/reconstruction.cc:352`）。
- 返回统计：`NumTotalRegImages()`、`NumSharedRegImages()`（mapper:1421-1427）、`LocalBundleAdjustmentReport` 各计数；被丢弃模型按 `min_model_size`（默认 10；`CustomizeIncrementalPipelineOptions` 把它压到 `min(0.5*NumImages, min_model_size)`，pipeline:50-58）。

## 11. `Reconstruction` 最小 API 与不变式
- 必须实现（`scene/reconstruction.h`）：`Images()/Image(id)/Point3D(id)/Points3D()/NumImages/NumPoints3D/NumRegFrames/NumRegImages/RegFrameIds()/RegImageIds()/Point3DIds()/ExistsPoint3D/ExistsImage`；`AddPoint3D(xyz, track, color) -> id`（新 id = `++max_point3D_id_`，`reconstruction.cc:544-554`）；`AddPoint3D(id, point3D)`（把 track 元素回写到 image 的 point2D，已存在则必须 id 相同，`cc:527-542`）；`AddObservation(id, el)`（要求该 point2D 尚无 3D；`cc:556-566`）；`DeletePoint3D`（清空 track 元素的 point2D 指针后 erase，`cc:605-615`）；`DeleteObservation`（`track.Length()<=2` → 删整点，否则删除元素并清指针，`cc:617-633`）；`MergePoints3D(a,b)`（长度加权的 xyz、颜色平均但黑=未提取不参与、track 拼接、删旧建新，`cc:568-603`）；`RegisterFrame/DeRegisterFrame`（`cc:670-705`：注册需 `frame.HasPose()`，`reg_frame_ids_` 追加 + `num_reg_images_ += 该 frame 的 image 数`；注销先删该 frame 所有 image 的 3D 观测、再 `frame.ResetPose()`）；`Normalize(fixed_scale=false, extent=10.0, min_percentile=0.1, max_percentile=0.9, use_images=true) -> Sim3d`；`Transform/ComputeNumObservations/ComputeMeanTrackLength/UpdatePoint3DErrors/Load(database_cache)/TearDown/Write/Read`。
- `Image`（`scene/image.h:146-173`）：`points2D_`（`xy` + `point3D_id`）、缓存 `num_points3D_`、`camera_ptr_`、`frame_ptr_`；`SetPoint3DForPoint2D`（若原本无 3D 则 `num_points3D_ += 1`，`image.cc:93-101`）、`ResetPoint3DForPoint2D`（有 3D 才 `-= 1`，`image.cc:103-109`）；`CamFromWorld()` = trivial rig 时 `rig_from_world`，否则 `sensor_from_rig * rig_from_world`（`image.h:267-270`，`frame.h:210-217`）；`HasPose()` = `frame_ptr_ != null && frame.HasPose()`。
- `Point3D`（`scene/point3d.h:41-57`）：`xyz`、`error`（默认 `-1` = 未计算，`HasError()`）、`track`、`color`。`Track`（`scene/track.h:52-90`）：插入序 `elements_`，`AddElement/DeleteElement/Length/Reserve`，`TrackElement` 相等= `(image_id, point2D_idx)` 相同。
- 不变式（Rust 端建议直接 assert）：① 对 track 每个元素，`image.Point2D(idx).point3D_id == point3D_id`；② 每个 point2D 至多属于一个 point3D；③ `image.num_points3D_ <= NumPoints2D()`；④ 保留点的 `track.Length() >= 2`（所有过滤路径都会删掉 <2 的点）；⑤ 只有 pose 存在的 image 上的 point2D 才能进入 track（所有三角化入口都先查 `HasPose()`）；⑥ `point3D.track` 中不允许出现同一 `(image_id, point2D_idx)` 两次（`AddObservation` 前置断言 + CG 去重保证）。

## 12. 结束/防环与投影公式
- 本版本**没有** `CheckRegisterImage`。防环/重试机制：
  1. per-image 注册尝试上限 `max_reg_trials=3`（`FindNextImages` impl:343-346 过滤 + `RegisterNextImage` :273 计数；structure-less 有独立计数器 :684）。
  2. `FindNextImages` 两桶：未试过且未被过滤的优先，其余（已试过 / 被过滤 frame）在后（impl:348-361）。
  3. 初始对：`init_max_reg_trials=2` 限制每张图当种子次数，`init_image_pairs` 限制每对只试一次（§3）。
  4. driver 主循环（`controllers/incremental_pipeline.cc:599-724`）：`do{...}while(reg_next_success || prev_reg_next_success)`；若本轮失败且上轮成功 → 立刻跑一次 `IterativeGlobalRefinement` 再进下一轮（:716-723），**连续两轮失败即退出**。
  5. 单个 next_images 列表内若连续失败 `reg_trial >= 30`（`kMinNumInitialRegTrials`）且 `NumRegImages < min_model_size` → 放弃该初值对（:648-655）。
  6. `NumSharedRegImages() >= max_model_overlap(20)` → 结束当前子模型（:710-714）。
  7. structure-less 回退开关（:588-597）：`{true}` / `{false,true}` / `{false}`；其阈值见 §4 末。
  8. 初始化松弛（:423-450）：`kNumInitRelaxations=2`，每轮先 `init_min_num_inliers /= 2`（100→50→25）后 `init_min_tri_angle /= 2`（16→8→4）并 `ResetInitializationStats()`（清 `init_image_pairs`、`init_num_reg_trials`）。
- 投影与误差（SIMPLE_RADIAL，参数顺序 **f, cx, cy, k**，`sensor/models.h:495-503`；1 个焦距 idx、2 个主点 idx、1 个 extra idx）：
  - `ImgFromCam(u,v,w)`（`models.h:1350-1377`）：cheirality `w >= ε`（`HasProjectableDepth`，`models.h:279-284`）否则失败；`x_n=u/w, y_n=v/w`；`r² = x_n²+y_n²`，`du = x_n·k·r², dv = y_n·k·r²`；`X = f·(x_n+du)+cx`，`Y = f·(y_n+dv)+cy`。`CameraModelImgFromCam` 失败返回 `nullopt`（`models.h:2905-2930`）。
  - `CalculateSquaredReprojectionError(point2D, point3D, cam_from_world, camera)`（`scene/projection.cc:56-91`）：非 spherical → `ImgFromCam(P·X)`，失败返回 `double::max()`，否则 `‖proj - point2D‖²`；spherical → `(angular_error · width/(2π))²`（cc:46-52）。
  - 过滤比较用 `sqrt(squared) > filter_max_reproj_error`（`sfm/observation_manager.cc:529-530,565`）；merge/complete 用与阈值的**平方**比较（`sfm/incremental_triangulator.cc:594-595,692-693,742-746`）；`tri_angle` 全部用弧度（`DegToRad` 入口、`RadToDeg` 仅日志）。
