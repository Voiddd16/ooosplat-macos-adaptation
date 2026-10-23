use crate::camera::{calculate_jacobian_clamp_limits, max_render_theta};
use crate::{
    RenderAuxInner, SplatOps,
    camera::Camera,
    dim_check::DimCheck,
    gaussian_splats::{RasterPass, SplatRenderMode},
    get_tile_offset::{CHECKS_PER_ITER, get_tile_offsets},
    kernels,
    render_aux::RenderOutput,
    sh::sh_degree_from_coeffs,
    shaders,
};
use brush_cube::create_tensor;
use brush_scan::prefix_sum;
use brush_sort::radix_argsort;
use burn::backend::TensorMetadata;
use burn::backend::ops::TransactionPrimitive;
use burn::backend::ops::{FloatTensorOps, IntTensorOps, TransactionOps};
use burn::backend::tensor::FloatTensor;
use burn::cubecl::CubeDim;
use burn::cubecl::calculate_cube_count_elemwise;
use burn::tensor::{DType, FloatDType, IntDType};
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::into_contiguous;
use glam::{Vec3, uvec2};
use kernels::types::RasterizeUniformsLaunch;
use std::f32::consts::PI;

#[doc(hidden)]
pub fn calc_tile_bounds(img_size: glam::UVec2) -> glam::UVec2 {
    uvec2(
        img_size.x.div_ceil(shaders::helpers::TILE_WIDTH),
        img_size.y.div_ceil(shaders::helpers::TILE_WIDTH),
    )
}

impl SplatOps for CubeBackend {
    #[allow(clippy::too_many_arguments)]
    async fn render(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        min_scale: FloatTensor<Self>,
        has_min_scale: bool,
        _refine_weight: FloatTensor<Self>,
        _coeffs_grad_sq: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        background: Vec3,
        pass: RasterPass,
    ) -> RenderOutput<Self> {
        assert!(
            img_size[0] > 0 && img_size[1] > 0,
            "Can't render images with 0 size."
        );
        let bwd_info = pass.bwd_info();
        let smooth_cutoff = pass.smooth_cutoff();

        let transforms = into_contiguous(transforms);
        let sh_coeffs = into_contiguous(sh_coeffs);
        let raw_opacities = into_contiguous(raw_opacities);
        let min_scale = into_contiguous(min_scale);

        DimCheck::new()
            .check_dims("transforms", &transforms, &["D".into(), 10.into()])
            .check_dims("sh_coeffs", &sh_coeffs, &["D".into(), "C".into(), 3.into()])
            .check_dims("raw_opacities", &raw_opacities, &["D".into()]);

        let total_splats = transforms.shape()[0] as u32;
        let sh_degree = sh_degree_from_coeffs(sh_coeffs.shape()[1] as u32);
        let mip_splat = matches!(render_mode, SplatRenderMode::Mip);

        // Cull splats beyond the lens' diagonal fov with some margin, but never
        // past the angle where the distortion polynomial folds back on itself:
        // those would project mirrored into the image with huge radii.
        let half_max_render_fov =
            (((camera.fov_x as f32).hypot(camera.fov_y as f32) * 1.05).min(2.0 * PI - 1e-6) * 0.5)
                .min(max_render_theta(&camera.camera_model) as f32);
        let pinhole_params = camera.build_pinhole_params(img_size);

        let mut project_uniforms = shaders::helpers::ProjectUniforms {
            viewmat: glam::Mat4::from(camera.world_to_local()).to_cols_array_2d(),
            camera_model: camera.camera_model,
            half_max_render_fov,
            pinhole_params,
            camera_position: [camera.position.x, camera.position.y, camera.position.z, 0.0],
            img_size: img_size.into(),
            tile_bounds: calc_tile_bounds(img_size).into(),
            sh_degree,
            total_splats,
            num_visible: 0, // num_visible — not yet known.
            jacobian_clamp_limits: calculate_jacobian_clamp_limits(
                img_size,
                pinhole_params,
                camera.camera_model,
            ),
        };

        let device = transforms.device.clone();
        let client = transforms.client.clone();

        let (
            global_from_presort_gid,
            compact_from_global,
            opacities,
            depths,
            intersect_counts,
            max_radius,
            num_visible_buf,
            num_intersections_buf,
        ) = {
            let project_uniforms: &shaders::helpers::ProjectUniforms = &project_uniforms;
            let _span = tracing::trace_span!("ProjectSplats").entered();

            let total_splats = project_uniforms.total_splats as usize;
            let num_visible_buf = Self::int_zeros([1].into(), &device, IntDType::U32);
            let num_intersections_buf = Self::int_zeros([1].into(), &device, IntDType::U32);
            let intersect_counts = Self::int_zeros([total_splats].into(), &device, IntDType::U32);
            let max_radius = Self::float_zeros([total_splats].into(), &device, FloatDType::F32);

            let global_from_presort_gid = create_tensor([total_splats], &device, DType::U32);
            // Written for every splat by the kernel, so no zero-fill.
            let compact_from_global = create_tensor([total_splats], &device, DType::U32);
            let opacities = create_tensor([total_splats], &device, DType::F32);
            let depths = create_tensor([total_splats], &device, DType::F32);

            let uniforms = project_uniforms.to_launch_object();
            let cube_dim = CubeDim::new_1d(kernels::project_forward::WG_SIZE);

            kernels::project_forward::project_forward_kernel::launch(
                &client,
                calculate_cube_count_elemwise(&client, total_splats, cube_dim),
                cube_dim,
                transforms.clone().into_tensor_arg(),
                raw_opacities.clone().into_tensor_arg(),
                min_scale.clone().into_tensor_arg(),
                global_from_presort_gid.clone().into_tensor_arg(),
                compact_from_global.clone().into_tensor_arg(),
                opacities.clone().into_tensor_arg(),
                depths.clone().into_tensor_arg(),
                num_visible_buf.clone().into_tensor_arg(),
                intersect_counts.clone().into_tensor_arg(),
                num_intersections_buf.clone().into_tensor_arg(),
                max_radius.clone().into_tensor_arg(),
                uniforms,
                mip_splat,
                has_min_scale,
                camera.camera_model,
            );
            (
                global_from_presort_gid,
                compact_from_global,
                opacities,
                depths,
                intersect_counts,
                max_radius,
                num_visible_buf,
                num_intersections_buf,
            )
        };

        // Read both atomic counts in one transaction BEFORE the sort.
        let (num_visible, num_intersections) = if total_splats == 0 {
            (0, 0)
        } else {
            let tp = TransactionPrimitive::<Self>::new(
                vec![],
                vec![],
                vec![num_visible_buf, num_intersections_buf],
                vec![],
            );
            let data = <Self as TransactionOps<Self>>::tr_execute(tp)
                .await
                .expect("Failed to read counts");
            let num_visible = data.read_ints[0]
                .clone()
                .try_into_vec::<u32>()
                .expect("num_visible")[0];
            let num_intersections = data.read_ints[1]
                .clone()
                .try_into_vec::<u32>()
                .expect("num_intersections")[0];
            (num_visible, num_intersections)
        };

        project_uniforms.num_visible = num_visible;

        let mip_splat = matches!(render_mode, SplatRenderMode::Mip);
        let img_size: glam::UVec2 = project_uniforms.img_size.into();
        let tile_bounds: glam::UVec2 = project_uniforms.tile_bounds.into();
        let num_visible_sz = (num_visible as usize).max(1);

        let dbg_projection = std::env::var_os("OOO_RASTER_DEBUG").is_some();
        // The GPU radix argsort is unreliable on Metal for full-width (32-bit)
        // keys: it returns a corrupted permutation (duplicates + out-of-range
        // indices), which scrambles the compact splat order and breaks the
        // rasterizer. Sort the visible list and prefix-sum the per-splat tile
        // counts on the CPU instead — a few ~200 KB readbacks and an
        // O(n log n) host sort per render, negligible next to the rasterize
        // itself. OOO_BRUSH_GPU_DEPTH_SORT=1 opts back into the GPU sort.
        let use_gpu_depth_sort = std::env::var_os("OOO_BRUSH_GPU_DEPTH_SORT").is_some();

        let (global_from_compact_gid, cum_tiles_hit) = if num_visible == 0 {
            let empty = Self::int_zeros([0usize].into(), &device, IntDType::U32);
            (empty.clone(), empty)
        } else if use_gpu_depth_sort {
            let depths_slice = Self::float_slice(depths, &[(0..num_visible_sz).into()]);
            let presort_slice =
                Self::int_slice(global_from_presort_gid, &[(0..num_visible_sz).into()]);
            let depths_pre = if dbg_projection {
                Some(depths_slice.clone())
            } else {
                None
            };
            let (_, sorted) = tracing::trace_span!("DepthSort")
                .in_scope(|| radix_argsort(depths_slice, presort_slice, 32));
            if dbg_projection {
                let perm = burn_cubecl::ops::into_data_sync(sorted.clone())
                    .to_vec::<u32>()
                    .expect("perm vec");
                let depths_data = burn_cubecl::ops::into_data_sync(depths_pre.expect("depths"))
                    .to_vec::<f32>()
                    .expect("depths vec");
                let depth_nan = depths_data.iter().filter(|v| v.is_nan()).count();
                let total_splats = project_uniforms.total_splats as usize;
                let oob = perm
                    .iter()
                    .filter(|&&g| (g as usize) >= total_splats)
                    .count();
                let mut sp = perm.clone();
                sp.sort_unstable();
                let dup = sp.windows(2).filter(|w| w[0] == w[1]).count();
                sp.dedup();
                let distinct = sp.len();
                log::info!(
                    "DepthSort(gpu) check: perm_len={} dup={} oob={} distinct={} of {} depth_nan={} visible={}",
                    perm.len(), dup, oob, distinct, total_splats, depth_nan, num_visible
                );
            }
            let compact_counts = Self::int_gather(0, intersect_counts.clone(), sorted.clone());
            let cum = tracing::trace_span!("PrefixSumGaussHits")
                .in_scope(|| prefix_sum(compact_counts));
            if dbg_projection {
                let cum_host = burn_cubecl::ops::into_data_sync(cum.clone())
                    .to_vec::<u32>()
                    .expect("cum vec");
                let monotonic = cum_host.windows(2).all(|w| w[0] <= w[1]);
                let last = cum_host.last().copied().unwrap_or(0);
                log::info!(
                    "CumTilesHit(gpu) check: len={} monotonic={} last={} intersections={}",
                    cum_host.len(), monotonic, last, num_intersections
                );
            }
            (sorted, cum)
        } else {
            let _span = tracing::trace_span!("DepthSortCpu").entered();
            let depths_all = burn_cubecl::ops::into_data_sync(depths.clone())
                .to_vec::<f32>()
                .expect("depths host");
            let presort_all = burn_cubecl::ops::into_data_sync(global_from_presort_gid.clone())
                .to_vec::<u32>()
                .expect("presort host");
            let counts_all = burn_cubecl::ops::into_data_sync(intersect_counts.clone())
                .to_vec::<u32>()
                .expect("counts host");
            let n = num_visible as usize;
            // `depths` is written at the compact slot (`write_id`), i.e. it is
            // slot-indexed over [0, num_visible), not gid-indexed — pair each
            // slot's depth with that slot's splat gid.
            let mut pairs: Vec<(f32, u32)> = depths_all[..n]
                .iter()
                .copied()
                .zip(presort_all[..n].iter().copied())
                .collect();
            // Nearest-first ordering matches the rasterizer's front-to-back
            // alpha compositing. NaN keys (shouldn't occur) sort last.
            pairs.sort_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Greater)
            });
            let mut acc = 0u32;
            let cum_host: Vec<u32> = pairs
                .iter()
                .map(|(_, g)| {
                    acc += counts_all[*g as usize];
                    acc
                })
                .collect();
            let perm_host: Vec<u32> = pairs.iter().map(|(_, g)| *g).collect();
            let sorted = Self::int_from_data(
                burn::tensor::TensorData::new(perm_host, vec![n]),
                &device,
            );
            let cum = Self::int_from_data(
                burn::tensor::TensorData::new(cum_host, vec![n]),
                &device,
            );
            if dbg_projection {
                let perm = burn_cubecl::ops::into_data_sync(sorted.clone())
                    .to_vec::<u32>()
                    .expect("perm vec");
                let total_splats = project_uniforms.total_splats as usize;
                let oob = perm
                    .iter()
                    .filter(|&&g| (g as usize) >= total_splats)
                    .count();
                let mut sp = perm.clone();
                sp.sort_unstable();
                let dup = sp.windows(2).filter(|w| w[0] == w[1]).count();
                sp.dedup();
                let distinct = sp.len();
                let cum_val = burn_cubecl::ops::into_data_sync(cum.clone())
                    .to_vec::<u32>()
                    .expect("cum vec");
                let monotonic = cum_val.windows(2).all(|w| w[0] <= w[1]);
                let last = cum_val.last().copied().unwrap_or(0);
                log::info!(
                    "DepthSort(cpu) check: perm_len={} dup={} oob={} distinct={} of {} | cum_len={} monotonic={} last={} intersections={}",
                    perm.len(), dup, oob, distinct, total_splats,
                    cum_val.len(), monotonic, last, num_intersections
                );
            }
            (sorted, cum)
        };
        let zero_proj = std::env::var_os("OOO_RASTER_ZERO_PROJ").is_some();
        let projected_splats = if dbg_projection || zero_proj {
            // Debug: zero-init so rows the projection kernel never writes are
            // detectable (uninitialized reads would look like legitimate data).
            // `OOO_RASTER_ZERO_PROJ` additionally skips the projection kernel
            // entirely: a correct rasterizer must then output pure background.
            Self::float_zeros(
                [num_visible_sz, kernels::helpers::PROJECTED_LANES_USIZE].into(),
                &device,
                FloatDType::F32,
            )
        } else {
            create_tensor(
                [num_visible_sz, kernels::helpers::PROJECTED_LANES_USIZE],
                &device,
                DType::F32,
            )
        };
        if zero_proj {
            log::info!("Raster zero-proj experiment: projection kernel skipped");
        } else {
        tracing::trace_span!("ProjectVisible").in_scope(|| {
            let uniforms = project_uniforms.to_launch_object();
            let cube_dim = CubeDim::new_1d(kernels::project_visible::WG_SIZE);
            kernels::project_visible::project_visible_kernel::launch(
                &client,
                calculate_cube_count_elemwise(&client, num_visible as usize, cube_dim),
                cube_dim,
                transforms.into_tensor_arg(),
                sh_coeffs.into_tensor_arg(),
                raw_opacities.into_tensor_arg(),
                min_scale.into_tensor_arg(),
                global_from_compact_gid.clone().into_tensor_arg(),
                compact_from_global.clone().into_tensor_arg(),
                projected_splats.clone().into_tensor_arg(),
                uniforms,
                mip_splat,
                has_min_scale,
                sh_degree,
                camera.camera_model,
            );
        });
        if dbg_projection {
            let data = burn_cubecl::ops::into_data_sync(projected_splats.clone())
                .to_vec::<f32>()
                .expect("projection data");
            let lanes = kernels::helpers::PROJECTED_LANES_USIZE;
            let rows = data.len() / lanes.max(1);
            let nan = data.iter().filter(|v| v.is_nan()).count();
            let huge = data.iter().filter(|v| v.abs() > 1.0e6).count();
            let unwritten = (0..rows)
                .filter(|&r| data[r * lanes..(r + 1) * lanes].iter().all(|v| *v == 0.0))
                .count();
            log::info!(
                "ProjectVisible check: rows={} nan={} huge={} unwritten={} of visible={}",
                rows, nan, huge, unwritten, num_visible
            );
            // Lane-level view of the projected data so systematic zeroing or
            // corruption of one column (e.g. colors/alpha) is visible.
            let lane_names = ["xy_x", "xy_y", "conic_x", "conic_y", "conic_z", "color_a", "r", "g", "b"];
            for (lane, name) in lane_names.iter().take(lanes).enumerate() {
                let mut mn = f32::INFINITY;
                let mut mx = f32::NEG_INFINITY;
                let mut sum = 0.0f32;
                let mut nz = 0usize;
                for r in 0..rows {
                    let v = data[r * lanes + lane];
                    mn = mn.min(v);
                    mx = mx.max(v);
                    sum += v;
                    if v != 0.0 {
                        nz += 1;
                    }
                }
                log::info!(
                    "Projected lane {} ({}): min={} max={} mean={} nonzero={}/{}",
                    lane,
                    name,
                    mn,
                    mx,
                    sum / rows.max(1) as f32,
                    nz,
                    rows
                );
            }
        }
        }
        let num_tiles = tile_bounds.x * tile_bounds.y;
        let buffer_size = (num_intersections as usize).max(1);
        let tile_id_from_isect = create_tensor([buffer_size], &device, DType::U32);
        let compact_gid_from_isect = create_tensor([buffer_size], &device, DType::U32);
        tracing::trace_span!("MapGaussiansToIntersect").in_scope(|| {
            let cube_dim = CubeDim::new_1d(kernels::map_gaussians::WG_SIZE);
            kernels::map_gaussians::map_gaussians_to_intersect_kernel::launch(
                &client,
                calculate_cube_count_elemwise(&client, num_visible as usize, cube_dim),
                cube_dim,
                projected_splats.clone().into_tensor_arg(),
                cum_tiles_hit.clone().into_tensor_arg(),
                tile_id_from_isect.clone().into_tensor_arg(),
                compact_gid_from_isect.clone().into_tensor_arg(),
                project_uniforms.tile_bounds[0],
                project_uniforms.tile_bounds[1],
                num_visible,
            );
        });
        let bits = u32::BITS - num_tiles.leading_zeros();
        let raster_debug = std::env::var_os("OOO_RASTER_DEBUG").is_some();
        let (keys_pre, gids_pre) = if raster_debug {
            (
                Some(tile_id_from_isect.clone()),
                Some(compact_gid_from_isect.clone()),
            )
        } else {
            (None, None)
        };
        let (tile_id_from_isect, compact_gid_from_isect) = tracing::trace_span!("Tile sort")
            .in_scope(|| radix_argsort(tile_id_from_isect, compact_gid_from_isect, bits));
        if raster_debug {
            // Diagnostic: compare the (key,gid) multiset before and after the
            // radix sort to prove whether the sort preserves its inputs.
            let mut pre_pairs: Vec<(u32, u32)> = burn_cubecl::ops::into_data_sync(
                keys_pre.expect("pre keys"),
            )
            .to_vec::<u32>()
            .expect("pre keys vec")
            .into_iter()
            .zip(
                burn_cubecl::ops::into_data_sync(gids_pre.expect("pre gids"))
                    .to_vec::<u32>()
                    .expect("pre gids vec"),
            )
            .collect();
            let keys_sorted = burn_cubecl::ops::into_data_sync(tile_id_from_isect.clone())
                .to_vec::<u32>()
                .expect("sorted keys");
            let gids_sorted = burn_cubecl::ops::into_data_sync(compact_gid_from_isect.clone())
                .to_vec::<u32>()
                .expect("sorted gids");
            let mut post_pairs: Vec<(u32, u32)> =
                keys_sorted.iter().copied().zip(gids_sorted.iter().copied()).collect();
            let keys_sorted_ok = keys_sorted.windows(2).all(|w| w[0] <= w[1]);
            pre_pairs.sort_unstable();
            post_pairs.sort_unstable();
            let multiset_mismatch = pre_pairs
                .iter()
                .zip(post_pairs.iter())
                .filter(|(a, b)| a != b)
                .count();
            let gid_oob = gids_sorted
                .iter()
                .filter(|&&g| (g as usize) >= num_visible as usize)
                .count();
            log::info!(
                "Raster sort check: keys_sorted={} multiset_mismatch={} gid_oob={} len_pre={} len_post={} intersections={} splats={}",
                keys_sorted_ok,
                multiset_mismatch,
                gid_oob,
                pre_pairs.len(),
                post_pairs.len(),
                num_intersections,
                project_uniforms.total_splats
            );
        }
        let cube_dim = CubeDim::new_1d(256);
        let tile_offsets = Self::int_zeros(
            [tile_bounds.y as usize, tile_bounds.x as usize, 2].into(),
            &device,
            IntDType::U32,
        );
        tracing::trace_span!("GetTileOffsets").in_scope(|| {
            get_tile_offsets::launch(
                &client,
                calculate_cube_count_elemwise(
                    &client,
                    num_intersections as usize,
                    CubeDim::new_1d(cube_dim.x * CHECKS_PER_ITER),
                ),
                cube_dim,
                num_intersections,
                num_tiles,
                tile_id_from_isect.clone().into_tensor_arg(),
                tile_offsets.clone().into_tensor_arg(),
            );
        });
        if std::env::var_os("OOO_RASTER_DEBUG").is_some() {
            // Diagnostic: force the tile-offset pass to finish and inspect the
            // ranges the rasterizer is about to trust.
            let offsets = burn_cubecl::ops::into_data_sync(tile_offsets.clone());
            let values = offsets.to_vec::<u32>().expect("tile offset data");
            let max = values.iter().copied().max().unwrap_or(0);
            let bad = values
                .iter()
                .filter(|&&value| value > num_intersections)
                .count();
            let non_monotonic = values
                .chunks_exact(2)
                .filter(|pair| pair[0] > pair[1])
                .count();
            // Cross-check against offsets recomputed on the CPU from the
            // sorted (key,gid) pairs: per-tile occurrence counts give the
            // expected [start,end) ranges. Any mismatch means get_tile_offsets
            // itself is wrong, independent of the rasterizer.
            let keys_host = burn_cubecl::ops::into_data_sync(tile_id_from_isect.clone())
                .to_vec::<u32>()
                .expect("sorted keys re-read");
            let tx_count = project_uniforms.tile_bounds[0] as usize;
            let ty_count = project_uniforms.tile_bounds[1] as usize;
            let total_tiles = tx_count * ty_count;
            let mut counts = vec![0u32; total_tiles];
            let mut key_oob = 0usize;
            for &k in &keys_host {
                if (k as usize) < total_tiles {
                    counts[k as usize] += 1;
                } else {
                    key_oob += 1;
                }
            }
            let mut expected = Vec::with_capacity(total_tiles * 2);
            let mut acc = 0u32;
            for &c in &counts {
                expected.push(acc);
                expected.push(acc + c);
                acc += c;
            }
            let mismatches = values
                .iter()
                .zip(expected.iter())
                .filter(|(a, b)| a != b)
                .count();
            let first_mismatch = values
                .iter()
                .zip(expected.iter())
                .position(|(a, b)| a != b);
            log::info!(
                "Raster tile offsets: len={} max={} out_of_range={} non_monotonic={} | vs_expected: mismatch={} first_mismatch={:?} key_oob={} total_tiles={}",
                values.len(),
                max,
                bad,
                non_monotonic,
                mismatches,
                first_mismatch.map(|i| (i, values[i], expected[i])),
                key_oob,
                total_tiles
            );
        }
        let out_dim = if bwd_info { 4 } else { 1 };
        let out_img = if dbg_projection {
            // 判别实验：未写入像素若为旧内存，则零初始化后 huge 必为 0。
            Self::float_zeros(
                [img_size.y as usize, img_size.x as usize, out_dim].into(),
                &device,
                FloatDType::F32,
            )
        } else {
            create_tensor(
                [img_size.y as usize, img_size.x as usize, out_dim],
                &device,
                DType::F32,
            )
        };
        let (out_packed_arg, out_f32_arg) = if bwd_info {
            (create_tensor([1], &device, DType::U32), out_img.clone())
        } else {
            (out_img.clone(), create_tensor([1], &device, DType::F32))
        };
        let total_splats = project_uniforms.total_splats as usize;
        let visible = if bwd_info {
            Self::float_zeros([total_splats].into(), &device, FloatDType::F32)
        } else {
            // Zero-init the dummy — `create_tensor` doesn't initialise, and
            // validate() may read this tensor to check its invariants.
            // Using `float_zeros` makes that read a well-defined no-op.
            Self::float_zeros([1].into(), &device, FloatDType::F32)
        };
        tracing::trace_span!("Rasterize").in_scope(|| {
            let uniforms = RasterizeUniformsLaunch::new(
                project_uniforms.tile_bounds[0],
                project_uniforms.img_size[0],
                project_uniforms.img_size[1],
                num_intersections,
                background.x,
                background.y,
                background.z,
            );
            // One cube per tile, one thread per pixel in it.
            let cube_dim = CubeDim::new_1d(shaders::helpers::TILE_SIZE);
            kernels::rasterize::rasterize_kernel::launch(
                &client,
                calculate_cube_count_elemwise(
                    &client,
                    (num_tiles * shaders::helpers::TILE_SIZE) as usize,
                    cube_dim,
                ),
                cube_dim,
                compact_gid_from_isect.clone().into_tensor_arg(),
                tile_offsets.clone().into_tensor_arg(),
                projected_splats.clone().into_tensor_arg(),
                out_packed_arg.into_tensor_arg(),
                out_f32_arg.into_tensor_arg(),
                global_from_compact_gid.clone().into_tensor_arg(),
                visible.clone().into_tensor_arg(),
                uniforms,
                // Precomputed divisor for the tiles-per-row split in the
                // per-pixel index math.
                project_uniforms.tile_bounds[0],
                bwd_info,
                smooth_cutoff,
                std::env::var_os("OOO_RASTER_PATTERN").is_some(),
            );
        });
        if raster_debug && bwd_info {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static RENDER_N: AtomicUsize = AtomicUsize::new(0);
            let n = RENDER_N.fetch_add(1, Ordering::Relaxed);
            if n < 8 {
                let data = burn_cubecl::ops::into_data_sync(out_img.clone())
                    .to_vec::<f32>()
                    .expect("out f32 data");
                let nan = data.iter().filter(|v| v.is_nan()).count();
                let huge = data.iter().filter(|v| v.abs() > 1.0e6).count();
                let max = data.iter().fold(0.0f32, |m, v| m.max(*v));
                let mut lane_huge = [0usize; 4];
                for (i, v) in data.iter().enumerate() {
                    if v.abs() > 1.0e6 {
                        lane_huge[i & 3] += 1;
                    }
                }
                let samples: Vec<String> = data
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.is_finite() && v.abs() > 1.0e6)
                    .take(5)
                    .map(|(i, v)| format!("(i={i} pix={} lane={} bits=0x{:08x})", i / 4, i & 3, v.to_bits()))
                    .collect();
                log::info!(
                    "Raster out f32: n={} w={} h={} len={} nan={} huge={} huge_lanes={:?} max={}",
                    n,
                    img_size.x,
                    img_size.y,
                    data.len(),
                    nan,
                    huge,
                    lane_huge,
                    max
                );
                if !samples.is_empty() {
                    log::info!("Raster huge samples: {}", samples.join(" "));
                }
                let w = img_size.x as usize;
                let h = img_size.y as usize;
                let mut bytes = Vec::with_capacity(w * h * 3);
                for i in 0..w * h {
                    for c in 0..3 {
                        let v = data[i * 4 + c];
                        let v = if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
                        bytes.push((v * 255.0) as u8);
                    }
                }
                let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
                ppm.extend_from_slice(&bytes);
                let _ = std::fs::create_dir_all("/tmp/brush-dump");
                let _ = std::fs::write(format!("/tmp/brush-dump/render_{n}.ppm"), ppm);
            }
        }
        if raster_debug && !zero_proj {
            // Post-render re-read of `projected`: if the buffer changed between
            // the pre-render validation read and now, some kernel is clobbering
            // pooled memory rather than the rasterizer math being broken.
            let post = burn_cubecl::ops::into_data_sync(projected_splats.clone())
                .to_vec::<f32>()
                .expect("post projection data");
            let post_huge = post.iter().filter(|v| v.abs() > 1.0e6).count();
            let post_nan = post.iter().filter(|v| v.is_nan()).count();
            log::info!(
                "Projected post-render check: len={} nan={} huge={}",
                post.len(),
                post_nan,
                post_huge
            );
        }
        RenderOutput {
            out_img,
            aux: RenderAuxInner {
                num_visible,
                num_intersections,
                visible,
                max_radius,
                opacities,
                tile_offsets,
                img_size,
            },
            projected_splats,
            compact_gid_from_isect,
            project_uniforms,
            global_from_compact_gid,
            compact_from_global,
        }
    }
}
