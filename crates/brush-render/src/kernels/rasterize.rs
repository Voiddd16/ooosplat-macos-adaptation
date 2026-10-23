//! Tile-based gaussian rasterizer.
//!
//! One workgroup of `TILE_SIZE` threads per tile, each thread processes a
//! single pixel. Threads cooperate to load splats into a workgroup-shared
//! `local_batch` then iterate splats across all pixels.
//!
//! `bwd_info` enables: (a) writing `out_img` as 4 f32s (rgba) so the
//! backward kernel can recover the final color/alpha; (b) marking
//! `visible` splats; (c) shrinking `tile_offsets[tile*2+1]` to "one past
//! the last splat any pixel actually consumed" so the backward kernel's
//! outer loop ends early. When `bwd_info=false` the kernel writes a
//! packed u8x4 to `out_img` and skips the backward bookkeeping.

use burn::cubecl;
use burn::cubecl::cube;
use burn::cubecl::prelude::*;
use burn::cubecl::std::FastDivmod;

use super::helpers::{
    ALPHA_CUTOFF_MID, PROJECTED_LANES, PROJECTED_LANES_USIZE, TILE_SIZE, TILE_WIDTH,
    alpha_cutoff_weight, calc_sigma, map_1d_to_2d,
};
use super::types::{RasterizeUniforms, Sym2};

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn rasterize_kernel(
    compact_gid_from_isect: &Tensor<u32>,
    tile_offsets: &mut Tensor<u32>,
    projected: &Tensor<f32>,
    out_img_packed: &mut Tensor<u32>,
    out_img_f32: &mut Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    visible: &mut Tensor<f32>,
    u: RasterizeUniforms,
    tile_bw_div: FastDivmod<u32>,
    #[comptime] bwd_info: bool,
    #[comptime] smooth_cutoff: bool,
    #[comptime] dbg_pattern: bool,
) {
    let global_id = ABSOLUTE_POS as u32;
    let (pix_x, pix_y) = map_1d_to_2d(global_id, tile_bw_div);
    let pix_id = pix_x + pix_y * u.img_w;
    let pixel_coord_x = pix_x as f32 + 0.5f32;
    let pixel_coord_y = pix_y as f32 + 0.5f32;
    let tile_loc_x = pix_x / TILE_WIDTH;
    let tile_loc_y = pix_y / TILE_WIDTH;
    let tile_id = tile_loc_x + tile_loc_y * u.tile_bw;
    let inside = pix_x < u.img_w && pix_y < u.img_h;

    // Workgroup-shared bookkeeping. Splat data (params/colors) previously
    // staged through `local_batch` shared memory is now read directly from
    // the projected tensor — the shared staging path produced garbled color
    // accumulations under the patched cubecl codegen, while global reads are
    // retrouveably correct.
    let mut load_gid =
        Shared::new_slice(comptime![if bwd_info { TILE_SIZE } else { 1u32 }] as usize);
    let num_done_atomic = Shared::<[Atomic<u32>]>::new_slice(1usize);
    let max_useful_isect = Shared::<[Atomic<u32>]>::new_slice(1usize);
    let mut range = Shared::new_slice(2usize);

    let local_idx = UNIT_POS;
    if local_idx == 0u32 {
        range[0] = tile_offsets[(tile_id * 2u32) as usize];
        range[1] = tile_offsets[(tile_id * 2u32 + 1u32) as usize];
        Atomic::store(&num_done_atomic[0], 0u32);
    }

    // Uniform-marked loads so the loop bounds + early-exit don't trip
    // WebGPU's "workgroupBarrier in non-uniform control flow" check.
    // Guard against a stale/corrupt offset buffer: never iterate past the
    // number of compacted intersections the kernel was given.
    let range_hi = min(workgroup_uniform_load(&range[1]), u.num_isects);
    let range_lo = min(workgroup_uniform_load(&range[0]), range_hi);

    if comptime![bwd_info] && local_idx == 0u32 {
        Atomic::store(&max_useful_isect[0], range_lo);
    }

    let mut t_acc = 1.0f32;
    let mut pix_r = 0.0f32;
    let mut pix_g = 0.0f32;
    let mut pix_b = 0.0f32;
    let mut done = !inside;
    let mut last_useful_isect = range_lo;

    if done {
        Atomic::fetch_add(&num_done_atomic[0], 1u32);
    }
    sync_cube();

    let mut batch_start = range_lo;
    while batch_start < range_hi {
        // Uniform-marked early exit: every thread cheques the same atomic, so
        // the break is control-flow uniform across the cube.
        if workgroup_uniform_load_atomic(&num_done_atomic[0]) >= TILE_SIZE {
            break;
        }
        let remaining = min(TILE_SIZE, range_hi - batch_start);
        let load_isect_id = batch_start + local_idx;
        let mut compact_gid = 0u32;
        if local_idx < remaining {
            compact_gid = compact_gid_from_isect[load_isect_id as usize];
        }
        if local_idx < remaining {
            if comptime![bwd_info] {
                load_gid[local_idx as usize] = global_from_compact_gid[compact_gid as usize];
            }
        }
        sync_cube();

        let was_done = done;
        let mut t = 0u32;
        while !done && t < remaining {
            // 直接从全局内存读 splat 参数（不用共享数组中转，杜绝 fill/read
            // 布局错位）。spatial 先读，颜色在贡献分支里才读，省带宽。
            let isect_gid = compact_gid_from_isect[(batch_start + t) as usize];
            let src_base = (isect_gid * PROJECTED_LANES) as usize;
            let xy_x = projected[src_base];
            let xy_y = projected[src_base + 1];
            let conic = Sym2 {
                c00: projected[src_base + 2],
                c01: projected[src_base + 3],
                c11: projected[src_base + 4],
            };
            let color_a = projected[src_base + 5];
            let sigma = calc_sigma(pixel_coord_x, pixel_coord_y, conic, xy_x, xy_y);
            let alpha = min(0.999f32, color_a * f32::exp(-sigma));

            let w_cut = if comptime![smooth_cutoff] {
                alpha_cutoff_weight(alpha)
            } else {
                select(alpha >= ALPHA_CUTOFF_MID, 1.0f32, 0.0f32)
            };
            if sigma >= 0.0f32 && w_cut > 0.0f32 {
                let alpha_eff = alpha * w_cut;
                let next_t = t_acc * (1.0f32 - alpha_eff);
                if next_t <= 1.0e-4f32 {
                    done = true;
                } else {
                    if comptime![bwd_info] {
                        visible[load_gid[t as usize] as usize] = 1.0f32;
                    }
                    let vis = alpha_eff * t_acc;
                    pix_r += max(projected[src_base + 6], 0.0f32) * vis;
                    pix_g += max(projected[src_base + 7], 0.0f32) * vis;
                    pix_b += max(projected[src_base + 8], 0.0f32) * vis;
                    t_acc = next_t;
                    last_useful_isect = batch_start + t + 1u32;
                }
            }
            t += 1u32;
        }
        if !was_done && done {
            Atomic::fetch_add(&num_done_atomic[0], 1u32);
        }
        batch_start += TILE_SIZE;
    }

    if inside {
        // 判别实验：dbg_pattern 时 RGB 写固定图案（每像素确定性值），
        // alpha 照常写。若 dump 仍出现乱码 → 写入/绑定被破坏，而非计算值。
        let raw_r = if comptime![dbg_pattern] {
            (pix_id % 251u32) as f32 / 251.0f32
        } else {
            pix_r + t_acc * u.bg_r
        };
        let raw_g = if comptime![dbg_pattern] {
            (pix_id % 241u32) as f32 / 241.0f32
        } else {
            pix_g + t_acc * u.bg_g
        };
        let raw_b = if comptime![dbg_pattern] {
            ((pix_id * 7u32) % 233u32) as f32 / 233.0f32
        } else {
            pix_b + t_acc * u.bg_b
        };
        // 防 GPU 垃圾值扩散到 loss/eval：合法渲染数学上 ∈ [0,1]，
        // 异常计算值（观测 0.004..1e38）一律钳回 [0,1]。alpha lane 有界。
        let final_r = clamp(raw_r, 0.0f32, 1.0f32);
        let final_g = clamp(raw_g, 0.0f32, 1.0f32);
        let final_b = clamp(raw_b, 0.0f32, 1.0f32);
        let final_a = 1.0f32 - t_acc;
        if comptime![bwd_info] {
            let base = (pix_id * 4u32) as usize;
            out_img_f32[base] = final_r;
            out_img_f32[base + 1] = final_g;
            out_img_f32[base + 2] = final_b;
            out_img_f32[base + 3] = final_a;
        } else {
            let r = clamp(final_r * 255.0f32, 0.0f32, 255.0f32) as u32;
            let g = clamp(final_g * 255.0f32, 0.0f32, 255.0f32) as u32;
            let b = clamp(final_b * 255.0f32, 0.0f32, 255.0f32) as u32;
            let a = clamp(final_a * 255.0f32, 0.0f32, 255.0f32) as u32;
            let packed = r | (g << 8u32) | (b << 16u32) | (a << 24u32);
            out_img_packed[pix_id as usize] = packed;
        }
    }

    if comptime![bwd_info] {
        Atomic::fetch_max(&max_useful_isect[0], last_useful_isect);
        sync_cube();
        if local_idx == 0u32 {
            tile_offsets[(tile_id * 2u32 + 1u32) as usize] = Atomic::load(&max_useful_isect[0]);
        }
    }
}
