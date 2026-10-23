use crate::{
    Emitter,
    config::TrainStreamConfig,
    message::{ProcessMessage, TrainMessage},
    slot::SlotSender,
};
use anyhow::Context;
use brush_dataset::{load_dataset, scene::Scene, scene_loader::SceneLoader};
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_rerun::visualize_tools::VisualizeTools;
use brush_train::{
    RandomSplatsConfig, create_random_splats,
    eval::eval_stats,
    lod::{compute_pup_scores, decimate_to_count},
    msg::RefineStats,
    to_init_splats,
    train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds},
};
use brush_vfs::BrushVfs;
use burn::module::Module;
use rand::SeedableRng;
use std::{path::PathBuf, sync::Arc};

#[allow(unused)]
use std::path::Path;

use tracing::{Instrument, trace_span};
use web_time::{Duration, Instant};

#[allow(clippy::large_stack_frames)]
pub(crate) async fn train_stream(
    vfs: Arc<BrushVfs>,
    train_stream_config: TrainStreamConfig,
    emitter: &Emitter,
    slot: SlotSender<Splats>,
    device: &crate::ProcessDevice,
) -> anyhow::Result<()> {
    log::info!("Start of training stream");

    emitter
        .emit(ProcessMessage::TrainMessage(TrainMessage::TrainConfig {
            config: Box::new(train_stream_config.clone()),
        }))
        .await;

    let visualize = VisualizeTools::new(train_stream_config.rerun_config.rerun_enabled).await;

    let process_config = &train_stream_config.process_config;
    log::info!("Using seed {}", process_config.seed);

    device.seed(process_config.seed);
    let mut rng = rand::rngs::StdRng::seed_from_u64(process_config.seed);

    log::info!("Loading dataset");
    let load_started = Instant::now();
    let load_result = load_dataset(vfs.clone(), &train_stream_config.load_config)
        .instrument(trace_span!("Load dataset"))
        .await
        .with_context(|| {
            format!(
                "Loading dataset failed after {:.3}s",
                load_started.elapsed().as_secs_f64()
            )
        })?;
    let load_elapsed = load_started.elapsed();

    // Emit any warnings from dataset loading.
    for warning in load_result.warnings {
        emitter
            .emit(ProcessMessage::Warning {
                error: anyhow::anyhow!("{warning}"),
            })
            .await;
    }

    let dataset = load_result.dataset;

    log::info!("Log scene to rerun");
    if let Err(error) = visualize.log_scene(
        &dataset.train,
        train_stream_config.rerun_config.rerun_max_img_size,
    ) {
        emitter.emit(ProcessMessage::Warning { error }).await;
    }

    let num_eval_views = dataset.eval.as_ref().map_or(0, |s| s.views.len());
    if let Err(error) = visualize.send_default_blueprint(num_eval_views) {
        emitter.emit(ProcessMessage::Warning { error }).await;
    }

    log::info!("Dataset loaded in {:.3}s", load_elapsed.as_secs_f64());
    emitter
        .emit(ProcessMessage::TrainMessage(TrainMessage::Dataset {
            dataset: dataset.clone(),
        }))
        .await;

    log::info!("Loading initial splats if any.");
    let init_started = Instant::now();
    let estimated_up = dataset.estimate_up();

    // Convert SplatData to Splats using KNN initialization
    let (up_axis, init_splats) = if let Some(msg) = load_result.init_splat {
        // Use loaded splats with KNN init
        let render_mode = train_stream_config
            .train_config
            .render_mode
            .or(msg.meta.render_mode)
            .unwrap_or(SplatRenderMode::Default);
        let max_splats = train_stream_config.train_config.max_splats as usize;
        let original = msg.data.num_splats();
        let data = msg.data.subsample(max_splats);
        if data.num_splats() < original {
            emitter
                .emit(ProcessMessage::Warning {
                    error: anyhow::anyhow!(
                        "Initial point cloud has {original} points, exceeding --max-splats ({max_splats}). Subsampled to {}; the remaining points were discarded. Raise --max-splats to keep more.",
                        data.num_splats()
                    ),
                })
                .await;
        }
        let method = if data.log_scales.is_none() {
            "KNN"
        } else {
            "provided scales"
        };
        log::info!(
            "Initial splat initialization started: {method}, {} points",
            data.num_splats()
        );
        let splats = to_init_splats(data, render_mode, device);
        (msg.meta.up_axis, splats)
    } else {
        // Default: just use random splats
        let render_mode = train_stream_config
            .train_config
            .render_mode
            .unwrap_or(SplatRenderMode::Default);
        log::info!("Starting with random splat config.");
        let cameras: Vec<_> = dataset.train.views.iter().map(|v| v.camera).collect();
        let config = RandomSplatsConfig::new();
        let scene_scale = train_stream_config.train_config.random_init_scene_scale;
        let splats = create_random_splats(
            &config,
            &cameras,
            scene_scale,
            &mut rng,
            render_mode,
            device,
        );
        (None, splats)
    };

    let init_splats = init_splats.with_sh_degree(train_stream_config.model_config.sh_degree);
    log::info!(
        "Initial splats initialized in {:.3}s: {} splats",
        init_started.elapsed().as_secs_f64(),
        init_splats.num_splats()
    );

    let prepare_started = Instant::now();
    log::info!("Trainer preparation started");

    // If the metadata has an up axis prefer that, otherwise estimate the up direction.
    let up_axis = up_axis.or(Some(estimated_up));

    // The trainer owns its working `splats` locally and publishes a
    // clone to the `Slot` after every modification (train
    // step, refine, LOD decimation).
    let mut splats: Splats = init_splats.clone();
    let bounds = get_splat_bounds(init_splats.clone(), BOUND_PERCENTILE).await;
    slot.set(0, splats.clone());
    emitter
        .emit(ProcessMessage::SplatsUpdated {
            up_axis,
            frame: 0,
            total_frames: 1,
            num_splats: init_splats.num_splats(),
            sh_degree: init_splats.sh_degree(),
            scene_scale: bounds.median_size(),
        })
        .await;

    emitter.emit(ProcessMessage::DoneLoading).await;

    // Start with memory cleared out.
    device.memory_cleanup();

    let mut eval_scene = dataset.eval;

    let mut train_duration = Duration::from_secs(0);
    let mut dataloader = SceneLoader::new(
        &dataset.train,
        process_config.seed,
        &train_stream_config.load_config,
    );
    // Per-train-view (world center, focal-px at native res) for the
    // Mip-Splatting 3D filter (always on).
    let mut view_cams: Vec<(glam::Vec3, f32)> = Vec::with_capacity(dataset.train.views.len());
    for view in dataset.train.views.iter() {
        let (w, h) = view.image.dimensions().await.unwrap_or((1, 1));
        let focal = view.camera.focal(glam::uvec2(w, h)).x;
        view_cams.push((view.camera.position, focal));
    }

    let mut trainer = SplatTrainer::new_seeded(
        &train_stream_config.train_config,
        device,
        bounds,
        process_config.seed,
    );
    trainer.set_view_cams(view_cams.clone());
    log::info!(
        "Trainer prepared in {:.3}s",
        prepare_started.elapsed().as_secs_f64()
    );

    // Get the dataset name from the base path (if available) for interpolation.
    let dataset_name = vfs
        .base_path()
        .and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "dataset".to_owned());

    // Interpolate {dataset} in the export path.
    let export_path_str = train_stream_config
        .process_config
        .export_path
        .replace("{dataset}", &dataset_name);

    // Resolve relative to the dataset's parent directory if available, otherwise CWD.
    let base_path = vfs
        .base_path()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));

    let export_path = base_path.join(&export_path_str);
    // Normalize path components
    let export_path: PathBuf = export_path.components().collect();
    let sh_degree = init_splats.sh_degree();

    let training_steps = train_stream_config.train_config.total_train_iters;
    let lod_levels = train_stream_config.train_config.lod_levels;
    let lod_refine_steps = train_stream_config.train_config.lod_refine_steps;
    let mut current_lod: u32 = 0;

    let process_config = &train_stream_config.process_config;

    log::info!("Start training loop.");
    let training_started = Instant::now();
    for iter in process_config.start_iter..train_stream_config.train_config.total_iters() {
        let target_lod = if lod_levels == 0 || iter < training_steps {
            0u32
        } else {
            ((iter - training_steps) / lod_refine_steps + 1).min(lod_levels)
        };

        if target_lod > current_lod {
            #[cfg(not(target_family = "wasm"))]
            {
                let (name, exp_iter, exp_total) = if current_lod == 0 {
                    (process_config.export_name.clone(), iter, training_steps)
                } else {
                    let lod_name = process_config
                        .export_name
                        .replace(".ply", &format!("_lod{current_lod}.ply"));
                    (lod_name, lod_refine_steps, lod_refine_steps)
                };
                let res = export_checkpoint(
                    splats.clone(),
                    &export_path,
                    &name,
                    exp_iter,
                    exp_total,
                    up_axis,
                    train_stream_config.load_config.units_per_meter,
                )
                .await
                .with_context(|| "Export at LOD boundary failed");

                if let Err(error) = res {
                    emitter.emit(ProcessMessage::Warning { error }).await;
                }
            }

            current_lod = target_lod;
            let lod_keep_pct = train_stream_config.train_config.lod_decimation_keep;
            let lod_img_pct = train_stream_config.train_config.lod_image_scale;

            log::info!("LOD {current_lod}/{lod_levels}: Decimating (keep {lod_keep_pct}%)");

            let before = splats.num_splats();
            let target_count = (before as f32 * lod_keep_pct as f32 / 100.0).max(1.0) as u32;

            log::info!("LOD {current_lod}/{lod_levels}: Computing sensitivity scores...");
            let scores = compute_pup_scores(splats.clone(), &dataset.train, device).await;
            splats = decimate_to_count(splats, &scores, target_count).await;
            slot.set(0, splats.clone());

            let after = splats.num_splats();
            log::info!("LOD {current_lod}/{lod_levels}: {before} -> {after} splats");

            device.memory_cleanup();

            let cumulative_scale = (lod_img_pct as f32 / 100.0).powi(current_lod as i32);
            // Only rebuild the loader when the images actually changed size.
            // A rebuild throws away a warm batch cache and re-decodes the
            // whole dataset, which at 100% scale buys nothing.
            if lod_img_pct < 100 {
                let lod_scene = dataset.train.clone().with_image_scale(cumulative_scale);
                dataloader = SceneLoader::new(
                    &lod_scene,
                    process_config.seed,
                    &train_stream_config.load_config,
                );
            }

            let bounds = get_splat_bounds(splats.clone(), BOUND_PERCENTILE).await;
            trainer = SplatTrainer::new_seeded(
                &train_stream_config.train_config,
                device,
                bounds,
                process_config.seed,
            );
            trainer.set_view_cams(view_cams.clone());

            log::info!(
                "LOD {current_lod}/{lod_levels}: Training for {lod_refine_steps} steps (image scale {:.0}%)",
                cumulative_scale * 100.0
            );
        }

        let step_time = Instant::now();
        let first_step = iter == process_config.start_iter;
        if first_step {
            log::info!("First training step started: iteration {}", iter + 1);
        }

        let trace_step = std::env::var_os("OOO_BRUSH_STEP_TRACE").is_some();
        if trace_step {
            log::info!("Waiting for training batch: iteration {}", iter + 1);
        }
        let batch = dataloader
            .next_batch()
            .instrument(trace_span!("Wait for next data batch"))
            .await;
        if trace_step {
            log::info!("Received training batch: iteration {}", iter + 1);
        }

        // Lift splats onto the autodiff graph for this step, run training,
        // then strip back to inner so the viewer slot sees plain splats.
        // `step` immediately replaces `splats` with the returned value, so we
        // can move it here instead of cloning every iteration.
        let diff_splats = splats.train();
        if trace_step {
            log::info!("Training step started: iteration {}", iter + 1);
        }
        let (new_diff_splats, stats) = trainer.step(batch, diff_splats).await;
        splats = new_diff_splats.valid();
        if trace_step {
            log::info!("Training step returned: iteration {}", iter + 1);
        }
        if std::env::var_os("OOO_BRUSH_STEP_SYNC").is_some() {
            let sync_start = Instant::now();
            splats.device().sync().expect("GPU step sync failed");
            log::info!(
                "Step GPU sync took {:.3}s: iteration {}",
                sync_start.elapsed().as_secs_f64(),
                iter + 1
            );
        }
        if first_step {
            // Host elapsed time only: do not synchronize the GPU for logging.
            log::info!(
                "First training step returned in {:.3}s: iteration {}",
                step_time.elapsed().as_secs_f64(),
                iter + 1
            );
        }

        // Phase-local iteration for refine gating
        let phase_iter = if current_lod == 0 {
            iter
        } else {
            (iter - training_steps) % lod_refine_steps
        };
        let phase_total = if current_lod == 0 {
            training_steps
        } else {
            lod_refine_steps
        };
        let phase_progress = (phase_iter as f32 / phase_total as f32).clamp(0.0, 1.0);

        let refine_start = Instant::now();
        let refine = if phase_iter > 0
            && phase_iter.is_multiple_of(train_stream_config.train_config.refine_every)
            && phase_progress <= 0.95
        {
            let (new_splats, refine_stats) = trainer.refine(iter, splats).await;
            splats = new_splats;
            refine_stats
        } else {
            RefineStats {
                num_added: 0,
                num_split_oversized: 0,
                num_split_high_grad: 0,
                num_pruned: 0,
                num_pruned_non_finite: 0,
                total_splats: splats.num_splats(),
            }
        };
        slot.set(0, splats.clone());
        let refine_dur = refine_start.elapsed();

        // We just finished iter 'iter', now starting iter + 1.
        let iter = iter + 1;
        let is_last_step = iter == train_stream_config.train_config.total_iters();

        let step_dur = step_time.elapsed();
        train_duration += step_dur;

        // Do evals. We skip this for LODs as it'd be confusing for rerun, but, could
        // revisit this.
        if current_lod == 0
            && (iter % process_config.eval_every == 0 || iter == training_steps)
            && let Some(eval_scene) = eval_scene.as_mut()
        {
            let save_path = train_stream_config
                .process_config
                .eval_save_to_disk
                .then(|| export_path.clone());

            let eval = run_eval(
                device,
                emitter,
                &visualize,
                splats.clone(),
                iter,
                eval_scene,
                save_path,
                train_stream_config.rerun_config.rerun_max_img_size,
            )
            .await
            .with_context(|| format!("Failed evaluation at iteration {iter}"));

            if let Err(error) = eval {
                emitter.emit(ProcessMessage::Warning { error }).await;
            }
        }

        // Export checkpoints
        #[cfg(not(target_family = "wasm"))]
        {
            let should_export = if current_lod == 0 {
                iter % process_config.export_every == 0 || (is_last_step && lod_levels == 0)
            } else {
                is_last_step
            };
            if should_export {
                let (name, exp_iter, exp_total) = if current_lod == 0 {
                    (process_config.export_name.clone(), iter, training_steps)
                } else {
                    let lod_name = process_config
                        .export_name
                        .replace(".ply", &format!("_lod{current_lod}.ply"));
                    (lod_name, lod_refine_steps, lod_refine_steps)
                };
                let res = export_checkpoint(
                    splats.clone(),
                    &export_path,
                    &name,
                    exp_iter,
                    exp_total,
                    up_axis,
                    train_stream_config.load_config.units_per_meter,
                )
                .await
                .with_context(|| format!("Export at iteration {iter} failed"));

                if let Err(error) = res {
                    emitter.emit(ProcessMessage::Warning { error }).await;
                }
            }
        }

        // --- Rerun logging ---
        {
            let rerun_config = &train_stream_config.rerun_config;
            visualize
                .log_splat_stats(iter, refine.total_splats)
                .unwrap();

            if let Some(every) = rerun_config.rerun_log_splats_every
                && (iter.is_multiple_of(every) || is_last_step)
            {
                visualize.log_splats(iter, splats.clone()).await.unwrap();
            }

            if iter.is_multiple_of(rerun_config.rerun_log_train_stats_every) || is_last_step {
                visualize
                    .log_train_stats(iter, &stats, step_dur)
                    .await
                    .unwrap();
            }

            // The memory query goes through the compute server and stalls
            // behind all queued GPU work — keep it off the hot path unless
            // rerun is actually recording, and then only on the stats cadence.
            if rerun_config.rerun_enabled
                && (iter.is_multiple_of(rerun_config.rerun_log_train_stats_every) || is_last_step)
            {
                visualize.log_memory(iter, &device.memory_pool_usage().unwrap_or_default())?;
            }

            if refine.num_added > 0 {
                visualize
                    .log_refine_stats(iter, &refine, refine_dur)
                    .unwrap();
            }

            // Distribution stats need a GPU read-back, so sample them on a
            // coarser cadence than the per-refine stats.
            if iter.is_multiple_of(rerun_config.rerun_log_distribution_every) || is_last_step {
                visualize
                    .log_splat_distribution_stats(iter, splats.clone())
                    .await
                    .unwrap();
            }
        }

        if refine.num_added > 0 {
            emitter
                .emit(ProcessMessage::TrainMessage(TrainMessage::RefineStep {
                    cur_splat_count: refine.total_splats,
                    iter,
                }))
                .await;
        }

        const UPDATE_EVERY: u32 = 5;
        if iter % UPDATE_EVERY == 0 || is_last_step {
            emitter
                .emit(ProcessMessage::SplatsUpdated {
                    up_axis: None,
                    frame: 0,
                    total_frames: 1,
                    num_splats: refine.total_splats,
                    sh_degree,
                    scene_scale: trainer.bounds().median_size(),
                })
                .await;

            let lod_progress = if current_lod > 0 {
                Some((current_lod, lod_levels))
            } else {
                None
            };

            emitter
                .emit(ProcessMessage::TrainMessage(TrainMessage::TrainStep {
                    iter,
                    total_elapsed: train_duration,
                    lod_progress,
                }))
                .await;
        }

        brush_async::yield_now().await;
    }

    log::info!(
        "Training loop completed in {:.3}s (including evaluation and exports)",
        training_started.elapsed().as_secs_f64()
    );
    emitter
        .emit(ProcessMessage::TrainMessage(TrainMessage::DoneTraining))
        .await;

    Ok(())
}

#[cfg(not(target_family = "wasm"))]
fn eval_output_path(base: &Path, iter: u32, image_path: &Path) -> anyhow::Result<PathBuf> {
    let mut relative = PathBuf::new();
    for component in image_path.components() {
        match component {
            std::path::Component::Normal(part) => relative.push(part),
            std::path::Component::CurDir => {}
            _ => anyhow::bail!(
                "Eval image path must be relative and cannot traverse directories: {}",
                image_path.display()
            ),
        }
    }
    if relative.as_os_str().is_empty() {
        anyhow::bail!("Eval image path is empty");
    }
    relative.set_extension("png");
    Ok(base.join(format!("eval_{iter}")).join(relative))
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;

    #[test]
    fn export_completion_logs_follow_io_result() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "train_stream::tests::export_probe",
                "--nocapture",
            ])
            .env("BRUSH_EXPORT_TEST_PROBE", "1")
            .output()
            .expect("Run isolated export probe");
        assert!(output.status.success(), "{output:?}");
    }

    #[test]
    fn export_probe() -> anyhow::Result<()> {
        if std::env::var_os("BRUSH_EXPORT_TEST_PROBE").is_none() {
            return Ok(());
        }
        struct Capture(std::sync::Mutex<Vec<String>>);
        impl log::Log for Capture {
            fn enabled(&self, _: &log::Metadata<'_>) -> bool {
                true
            }
            fn log(&self, record: &log::Record<'_>) {
                self.0.lock().unwrap().push(record.args().to_string());
            }
            fn flush(&self) {}
        }
        static CAPTURE: Capture = Capture(std::sync::Mutex::new(Vec::new()));
        log::set_logger(&CAPTURE)?;
        log::set_max_level(log::LevelFilter::Info);
        let root = std::env::temp_dir().join(format!("brush-export-probe-{}", std::process::id()));
        std::fs::create_dir(&root)?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        runtime.block_on(async {
            for name in ["export_0010.ply", "export_0010_lod1.ply"] {
                let path = root.join(name);
                let bytes = b"ply\nformat ascii 1.0\nend_header\n";
                log_checkpoint_export(name, 10, async {
                    tokio::fs::write(&path, bytes)
                        .await
                        .context("Writing checkpoint")
                })
                .await?;
                assert_eq!(std::fs::read(&path)?, bytes);
                std::fs::remove_file(path)?;
            }
            let blocked = root.join("blocked");
            std::fs::write(&blocked, b"not a directory")?;
            let error = log_checkpoint_export("failed.ply", 20, async {
                tokio::fs::create_dir_all(&blocked)
                    .await
                    .context("Creating export directory")
            })
            .await
            .unwrap_err();
            let chain = format!("{error:#}");
            assert!(chain.contains("Export failed after"), "{chain}");
            assert!(chain.contains("Creating export directory"), "{chain}");
            assert!(
                error.chain().count() >= 3,
                "Original I/O cause must be preserved"
            );
            log::warn!("{error:#}");
            std::fs::remove_file(blocked)?;
            Ok::<(), anyhow::Error>(())
        })?;
        std::fs::remove_dir(root)?;
        let entries = CAPTURE.0.lock().unwrap();
        assert_eq!(
            entries
                .iter()
                .filter(|line| line.contains("Export started:"))
                .count(),
            3
        );
        assert_eq!(
            entries
                .iter()
                .filter(|line| line.contains("Export succeeded"))
                .count(),
            2
        );
        assert!(
            entries
                .iter()
                .any(|line| line.contains("Export failed after") && line.contains("failed.ply"))
        );
        assert!(
            !entries
                .iter()
                .any(|line| line.contains("Export succeeded") && line.contains("failed.ply"))
        );
        Ok(())
    }

    #[test]
    fn eval_paths_are_safe_and_distinct() {
        let base = Path::new("output");
        let first = eval_output_path(base, 42, Path::new("images/a/frame.jpg")).unwrap();
        let second = eval_output_path(base, 42, Path::new("images/b/frame.jpg")).unwrap();

        assert_eq!(first, Path::new("output/eval_42/images/a/frame.png"));
        assert_eq!(second, Path::new("output/eval_42/images/b/frame.png"));
        assert_ne!(first, second);

        for invalid in ["../frame.jpg", "/tmp/frame.jpg", "."] {
            assert!(eval_output_path(base, 1, Path::new(invalid)).is_err());
        }
    }
}

async fn run_eval(
    device: &burn::tensor::Device,
    emitter: &Emitter,
    visualize: &VisualizeTools,
    splats: Splats,
    iter: u32,
    eval_scene: &Scene,
    save_path: Option<PathBuf>,
    rerun_max_img_size: u32,
) -> Result<(), anyhow::Error> {
    if eval_scene.views.is_empty() {
        return Ok(());
    }

    let mut psnr = 0.0;
    let mut ssim = 0.0;
    let mut count = 0;
    log::info!("Running evaluation for iteration {iter}");

    for (i, view) in eval_scene.views.iter().enumerate() {
        brush_async::yield_now().await;

        let eval_img = view.image.load().await?;
        let sample = eval_stats(
            splats.clone(),
            &view.camera,
            eval_img,
            view.image.alpha_mode(),
            device,
        )
        .await
        .context("Failed to run eval for sample.")?;

        count += 1;
        psnr += sample.psnr.clone().into_scalar_async::<f32>().await?;
        ssim += sample.ssim.clone().into_scalar_async::<f32>().await?;

        #[cfg(not(target_family = "wasm"))]
        if let Some(path) = &save_path {
            let path = eval_output_path(path, iter, view.image.path())?;
            if let Some(parent) = path.parent() {
                tokio::fs::create_dir_all(parent).await.with_context(|| {
                    format!(
                        "Failed to create eval output directory {}",
                        parent.display()
                    )
                })?;
            }
            sample.save_to_disk(&path).await?;
        }

        #[cfg(target_family = "wasm")]
        let _ = save_path;

        visualize
            .log_eval_sample(iter, i as u32, sample, rerun_max_img_size)
            .await?;
    }
    psnr /= count as f32;
    ssim /= count as f32;
    visualize.log_eval_stats(iter, psnr, ssim)?;
    emitter
        .emit(ProcessMessage::TrainMessage(TrainMessage::EvalResult {
            iter,
            avg_psnr: psnr,
            avg_ssim: ssim,
        }))
        .await;

    Ok(())
}

// TODO: Want to support this on WASM somehow. Maybe have user pick a file once,
// and write to it repeatedly?
#[cfg(not(target_family = "wasm"))]
async fn export_checkpoint(
    splats: Splats,
    export_path: &Path,
    export_name: &str,
    iter: u32,
    total_steps: u32,
    up_axis: Option<glam::Vec3>,
    units_per_meter: f32,
) -> Result<(), anyhow::Error> {
    let digits = ((total_steps as f64).log10().floor() as usize) + 1;
    let export_name = export_name.replace("{iter}", &format!("{iter:0digits$}"));
    // Only the filename is added to diagnostic logs, not the resolved directory.
    let display_name = Path::new(&export_name)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    log_checkpoint_export(&display_name, iter, async {
        tokio::fs::create_dir_all(&export_path)
            .await
            .with_context(|| format!("Creating export directory {}", export_path.display()))?;
        // Training runs in metres; write the file back in the dataset's units.
        let splat_data = brush_serde::splat_to_ply(splats.scaled(units_per_meter), up_axis)
            .await
            .context("Serializing splat data")?;
        tokio::fs::write(export_path.join(&export_name), splat_data)
            .await
            .context(format!("Failed to export ply {export_path:?}"))?;
        Ok(())
    })
    .await
}

// A local wrapper around the existing export operation. Failures remain
// Warning messages at the call sites; their context includes elapsed time.
#[cfg(not(target_family = "wasm"))]
async fn log_checkpoint_export(
    name: &str,
    iter: u32,
    export: impl std::future::Future<Output = anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let started = Instant::now();
    log::info!("Export started: iteration {iter}, {name}");
    export.await.with_context(|| {
        format!(
            "Export failed after {:.3}s: iteration {iter}, {name}",
            started.elapsed().as_secs_f64()
        )
    })?;
    log::info!(
        "Export succeeded in {:.3}s: iteration {iter}, {name}",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
