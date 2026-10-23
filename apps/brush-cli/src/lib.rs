#![recursion_limit = "256"]
#![cfg(not(target_family = "wasm"))]

use brush_async::Actor;
use brush_process::DataSource;
use brush_process::RunningProcess;
use brush_process::config::TrainStreamConfig;
use brush_process::create_process;
use brush_process::message::ProcessMessage;
use brush_process::message::TrainMessage;

use clap::{Error, Parser, builder::ArgPredicate, error::ErrorKind};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use indicatif_log_bridge::LogWrapper;
use std::{
    path::Path,
    sync::OnceLock,
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tracing::trace_span;

#[derive(Parser)]
#[command(
    author,
    version,
    arg_required_else_help = false,
    about = "Brush - universal splats"
)]
pub struct Cli {
    /// Source to load from (path or URL).
    #[arg(value_name = "PATH_OR_URL")]
    pub source: Option<DataSource>,

    #[arg(
        long,
        default_value = "true",
        default_value_if("source", ArgPredicate::IsPresent, "false"),
        help = "Spawn a viewer to visualize the training"
    )]
    pub with_viewer: bool,

    #[clap(flatten)]
    pub train_stream: TrainStreamConfig,
}

impl Cli {
    pub fn validate(self) -> Result<Self, Error> {
        if !self.with_viewer && self.source.is_none() {
            return Err(Error::raw(
                ErrorKind::MissingRequiredArgument,
                "When --with-viewer is false, --source must be provided",
            ));
        }
        Ok(self)
    }
}

/// Build the training process described by `args`, or `None` if no source was
/// given. Shared by the standalone CLI binary and brush-app's headless path.
pub fn build_process(args: &Cli) -> Option<RunningProcess> {
    let source = args.source.clone()?;
    let cli_config = args.train_stream.clone();
    Some(create_process(source, async move |init| {
        Some(brush_process::args_file::merge_configs(&init, &cli_config))
    }))
}

/// Install the CLI logger once, sharing its progress display with the UI.
/// The caller's `RUST_LOG` filter is used unchanged.
pub fn init_cli_logging() -> anyhow::Result<MultiProgress> {
    static LOGGING: OnceLock<Result<MultiProgress, String>> = OnceLock::new();
    LOGGING
        .get_or_init(|| {
            let logger = env_logger::Builder::from_default_env()
                .target(env_logger::Target::Stdout)
                .build();
            let level = logger.filter();
            let multi = MultiProgress::new();
            LogWrapper::new(multi.clone(), logger)
                .try_init()
                .map_err(|error| format!("Failed to initialize CLI logger: {error}"))?;
            log::set_max_level(level);
            Ok(multi)
        })
        .as_ref()
        .cloned()
        .map_err(|error| anyhow::anyhow!("{error}"))
}

/// Initialize logging and the backend, then drive `process` to completion.
pub async fn run_headless(
    process: RunningProcess,
    train_stream_config: TrainStreamConfig,
) -> Result<(), anyhow::Error> {
    init_cli_logging()?;
    log::info!("Brush {} (headless)", env!("CARGO_PKG_VERSION"));
    brush_process::burn_init_setup().await;
    run_cli_ui(process, train_stream_config).await
}

/// Run the CLI: pin the trainer stream to a dedicated [`Actor`] thread,
/// drive the indicatif UI on the main task.
pub async fn run_cli_ui(
    mut process: RunningProcess,
    train_stream_config: TrainStreamConfig,
) -> Result<(), anyhow::Error> {
    let sp = init_cli_logging()?;
    log::info!("Compute backend: {:?}", process.device);
    // Pump the trainer stream from a dedicated Actor thread; the
    // indicatif UI loop below consumes its output on the main task.
    let (tx, mut messages) = mpsc::unbounded_channel();
    let trainer = Actor::new("cli-trainer");
    let trainer_task = trainer.run(move || async move {
        while let Some(msg) = process.stream.next().await {
            let failed = msg.is_err();
            if tx.send(msg).is_err() || failed {
                break;
            }
        }
    });

    // Hold the actor for the lifetime of the UI loop; dropping it
    // would kill the pump.
    let _trainer = trainer;

    let main_spinner = ProgressBar::new_spinner().with_style(
        ProgressStyle::with_template("{spinner:.blue} {msg}")
            .expect("Invalid indacitif config")
            .tick_strings(&[
                "🖌️      ",
                "█🖌️     ",
                "▓█🖌️    ",
                "░▓█🖌️   ",
                "•░▓█🖌️  ",
                "·•░▓█🖌️ ",
                " ·•░▓🖌️ ",
                "  ·•░🖌️ ",
                "   ·•🖌️ ",
                "    ·🖌️ ",
                "     🖌️ ",
                "    🖌️ █",
                "   🖌️ █▓",
                "  🖌️ █▓░",
                " 🖌️ █▓░•",
                "🖌️ █▓░•·",
                "🖌️ ▓░•· ",
                "🖌️ ░•·  ",
                "🖌️ •·   ",
                "🖌️ ·    ",
                "🖌️      ",
            ]),
    );

    let stats_spinner = ProgressBar::new_spinner().with_style(
        ProgressStyle::with_template("{spinner:.blue} {msg}")
            .expect("Invalid indicatif config")
            .tick_strings(&["ℹ️", "ℹ️"]),
    );

    // Sized once the process emits its final config: the CLI args alone are
    // wrong when a dataset's args.txt is merged in.
    let train_progress = {
        let bar = ProgressBar::new(0)
        .with_style(
            ProgressStyle::with_template(
                "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg} ({per_sec}, {eta} remaining)",
            )
            .expect("Invalid indicatif config").progress_chars("◍○○"),
        )
        .with_message("Steps");
        sp.add(bar)
    };

    let main_spinner = sp.add(main_spinner);
    main_spinner.enable_steady_tick(Duration::from_millis(120));

    let eval_spinner = sp.add(
        ProgressBar::new_spinner().with_style(
            ProgressStyle::with_template("{spinner:.blue} {msg}")
                .expect("Invalid indicatif config")
                .tick_strings(&["✅", "✅"]),
        ),
    );

    eval_spinner.set_message("waiting for dataset...");

    let stats_spinner = sp.add(stats_spinner);
    stats_spinner.set_message("Starting up");
    log::info!("Starting up");

    if cfg!(debug_assertions) {
        let _ =
            sp.println("ℹ️  running in debug mode, compile with --release for best performance");
    }

    #[allow(unused_mut)]
    let mut duration = Duration::from_secs(0);
    let mut eval_every = train_stream_config.process_config.eval_every;
    let mut done_training = false;
    let mut stream_error = None;
    let mut progress_log = ProgressLog::default();
    let mut total_iters = 0;

    while let Some(msg) = messages.recv().await {
        let _span = trace_span!("CLI UI").entered();

        let msg = match msg {
            Ok(msg) => msg,
            Err(error) => {
                // Don't print the error here. It'll bubble up and be printed as output.
                let _ = sp.println("❌ Encountered an error");
                stream_error = Some(error);
                break;
            }
        };

        match msg {
            ProcessMessage::NewProcess => {
                main_spinner.set_message("Starting process...");
            }
            ProcessMessage::StartLoading { name, training, .. } => {
                if !training {
                    // Display a big warning saying viewing splats from the CLI doesn't make sense.
                    let _ = sp.println("❌ Only training is supported in the CLI (try passing --with-viewer to view a splat)");
                    anyhow::bail!(
                        "Only training is supported in the CLI (use --with-viewer to view a splat)"
                    );
                }
                main_spinner.set_message(format!("Loading {name}..."));
            }
            ProcessMessage::SplatsUpdated { .. } => {}
            ProcessMessage::TrainMessage(train) => match train {
                TrainMessage::TrainConfig { config } => {
                    log_train_config(&config);
                    total_iters = config.train_config.total_iters();
                    train_progress.set_length(config.train_config.total_iters() as u64);
                    eval_every = config.process_config.eval_every;
                }
                TrainMessage::Dataset { dataset } => {
                    let train_views = dataset.train.views.len();
                    let eval_views = dataset.eval.as_ref().map_or(0, |v| v.views.len());
                    log::info!(
                        "Loaded dataset with {train_views} training, {eval_views} eval views",
                    );
                    main_spinner.set_message(format!(
                        "Loading dataset with {train_views} training, {eval_views} eval views",
                    ));
                    if eval_views > 0 {
                        eval_spinner.set_message(format!(
                            "evaluating {eval_views} views every {eval_every} steps",
                        ));
                    } else {
                        eval_spinner.finish_and_clear();
                    }
                }
                TrainMessage::TrainStep {
                    iter,
                    total_elapsed,
                    lod_progress,
                    ..
                } => {
                    if progress_log.should_emit(Instant::now(), iter, total_iters, lod_progress) {
                        log::info!(
                            "Training progress: iteration={iter} total={total_iters} elapsed_secs={:.3} lod={}",
                            total_elapsed.as_secs_f64(),
                            lod_progress.map_or(0, |(lod, _)| lod),
                        );
                    }
                    if let Some((lod, total_lods)) = lod_progress {
                        main_spinner.set_message(format!("LOD {lod}/{total_lods}"));
                    } else {
                        main_spinner.set_message("Training");
                    }
                    train_progress.set_position(iter as u64);
                    duration = total_elapsed;
                }
                TrainMessage::RefineStep {
                    cur_splat_count,
                    iter,
                    ..
                } => {
                    stats_spinner.set_message(format!("Current splat count {cur_splat_count}"));
                    log::info!("Refine iter {iter}, {cur_splat_count} splats.");
                }
                TrainMessage::EvalResult {
                    iter,
                    avg_psnr,
                    avg_ssim,
                } => {
                    log::info!("Eval iter {iter}: PSNR {avg_psnr}, ssim {avg_ssim}");

                    eval_spinner.set_message(format!(
                        "Eval iter {iter}: PSNR {avg_psnr}, ssim {avg_ssim}"
                    ));
                }
                TrainMessage::DoneTraining => done_training = true,
            },
            ProcessMessage::DoneLoading => {
                log::info!("Completed loading.");
                main_spinner.set_message("Completed loading");
                stats_spinner.set_message("Completed loading");
            }
            ProcessMessage::Warning { error } => {
                // Alternate form prints the whole anyhow context chain.
                log::warn!("{error:#}");
                sp.println(format!("⚠️: {error:#}"))?;
            }
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }

    // Await even after DoneTraining: a subsequent panic must never look like
    // a successful run. Actor's handle preserves the original panic payload.
    trainer_task.await;
    if let Some(error) = stream_error {
        return Err(error);
    }
    anyhow::ensure!(done_training, "Training stream closed before DoneTraining");

    let duration_secs = Duration::from_secs(duration.as_secs());
    let _ = sp.println(format!(
        "Training took {}",
        humantime::format_duration(duration_secs)
    ));

    log::info!(
        "Done training! Took {:?}.",
        humantime::format_duration(duration_secs)
    );

    Ok(())
}

#[derive(Default)]
struct ProgressLog {
    last: Option<(Instant, Option<(u32, u32)>)>,
}

#[cfg(test)]
mod progress_tests {
    use super::*;

    #[test]
    fn progress_is_throttled_except_first_last_and_lod_change() {
        let mut progress = ProgressLog::default();
        let now = Instant::now();
        assert!(progress.should_emit(now, 5, 100, None));
        assert!(!progress.should_emit(now + Duration::from_millis(900), 10, 100, None));
        assert!(progress.should_emit(now + Duration::from_secs(1), 15, 100, None));
        assert!(progress.should_emit(now + Duration::from_millis(1100), 20, 100, Some((1, 2))));
        assert!(!progress.should_emit(now + Duration::from_millis(1200), 25, 100, Some((1, 2))));
        assert!(progress.should_emit(now + Duration::from_millis(1300), 100, 100, Some((1, 2))));
    }
}

impl ProgressLog {
    fn should_emit(
        &mut self,
        now: Instant,
        iter: u32,
        total: u32,
        lod: Option<(u32, u32)>,
    ) -> bool {
        let emit = self.last.is_none_or(|(last, previous_lod)| {
            now.duration_since(last) >= Duration::from_secs(1) || previous_lod != lod
        }) || iter == total;
        if emit {
            self.last = Some((now, lod));
        }
        emit
    }
}

fn log_train_config(config: &TrainStreamConfig) {
    let train = &config.train_config;
    let process = &config.process_config;
    log::info!(
        "Training config: train_iters={} total_iters={} start_iter={} lod_levels={} lod_refine_steps={} lod_keep_pct={} lod_image_scale={} max_resolution={} max_splats={} growth_start_iter={} growth_stop_iter={} refine_every={} seed={} eval_every={}",
        train.total_train_iters,
        train.total_iters(),
        process.start_iter,
        train.lod_levels,
        train.lod_refine_steps,
        train.lod_decimation_keep,
        train.lod_image_scale,
        config.load_config.max_resolution,
        train.max_splats,
        train.growth_start_iter,
        train.growth_stop_iter,
        train.refine_every,
        process.seed,
        process.eval_every,
    );
    // Keep relative templates useful, without expanding private parent paths.
    let export_path = if Path::new(&process.export_path).is_absolute() {
        "<absolute directory>"
    } else {
        &process.export_path
    };
    let export_name = Path::new(&process.export_name)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown");
    log::info!(
        "Export config: every={} directory={} name={} eval_save_to_disk={} units_per_meter={}",
        process.export_every,
        export_path,
        export_name,
        process.eval_save_to_disk,
        config.load_config.units_per_meter,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn probe(scenario: &str, filter: &str) -> std::process::Output {
        std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "tests::diagnostic_probe", "--nocapture"])
            .env("BRUSH_CLI_TEST_SCENARIO", scenario)
            .env("RUST_LOG", filter)
            .env("RUST_BACKTRACE", "1")
            .output()
            .expect("Run isolated CLI diagnostic probe")
    }

    #[test]
    fn logger_initialization_and_filters() {
        let output = probe("logging", "info");
        assert!(output.status.success(), "{output:?}");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert_eq!(stdout.matches("CLI logger ready").count(), 1);
        assert!(!stdout.contains("CLI debug probe"));
        assert!(!stdout.contains("CLI trace probe"));

        let output = probe("logging", "error");
        assert!(output.status.success(), "{output:?}");
        assert!(!String::from_utf8_lossy(&output.stdout).contains("CLI logger ready"));

        let output = probe("logging", "error,brush_cli=info");
        assert!(output.status.success(), "{output:?}");
        assert!(String::from_utf8_lossy(&output.stdout).contains("CLI logger ready"));
    }

    #[test]
    fn logs_final_merged_config_and_success() {
        let output = probe("success", "info");
        assert!(output.status.success(), "{output:?}");
        let stdout = String::from_utf8_lossy(&output.stdout);
        for field in [
            "train_iters=17 total_iters=23 start_iter=2",
            "lod_levels=2 lod_refine_steps=3",
            "max_resolution=50",
            "max_splats=321",
            "growth_stop_iter=11",
            "seed=7",
            "every=9",
            "directory=./{dataset}_exports/",
            "Done training!",
        ] {
            assert!(stdout.contains(field), "Missing {field}: {stdout}");
        }
    }

    #[test]
    fn logs_actual_config_after_existing_merge_fallback() {
        let output = probe("merge-fallback", "info");
        assert!(output.status.success(), "{output:?}");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Failed to parse merged config"), "{stdout}");
        assert!(
            stdout.contains("train_iters=23 total_iters=23 start_iter=0 lod_levels=0"),
            "{stdout}"
        );
    }

    #[test]
    fn absolute_export_directory_is_not_logged() {
        let output = probe("absolute-path", "info");
        assert!(output.status.success(), "{output:?}");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("directory=<absolute directory>"),
            "{stdout}"
        );
        assert!(!stdout.contains("private-parent"), "{stdout}");
    }

    #[test]
    fn abnormal_streams_do_not_report_success() {
        for scenario in [
            "closed",
            "error",
            "panic",
            "done-then-panic",
            "done-then-error",
            "ply",
        ] {
            let output = probe(scenario, "info");
            assert!(!output.status.success(), "{scenario}: {output:?}");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(!stdout.contains("Done training!"), "{scenario}: {stdout}");
            assert!(!stdout.contains("Training took"), "{scenario}: {stdout}");
            if scenario.contains("panic") {
                assert!(stderr.contains("original GPU panic probe"), "{stderr}");
                assert!(stderr.contains("cli-trainer"), "{stderr}");
            } else if scenario.contains("error") {
                assert!(stderr.contains("training stage context"), "{stderr}");
                assert!(stderr.contains("original I/O failure"), "{stderr}");
            } else if scenario == "closed" {
                assert!(stderr.contains("before DoneTraining"), "{stderr}");
            } else {
                assert!(stderr.contains("Only training is supported"), "{stderr}");
            }
        }
    }

    #[test]
    fn export_warning_preserves_successful_training_and_error_chain() {
        let output = probe("warning", "info");
        assert!(output.status.success(), "{output:?}");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("Export at iteration 9 failed: disk full"),
            "{stdout}"
        );
        assert!(stdout.contains("Done training!"), "{stdout}");
        assert!(!stdout.contains("Export succeeded"), "{stdout}");
    }

    // Invoked in a subprocess so global logging and panic behavior are tested
    // without changing the parent test runner's environment or panic hook.
    #[test]
    fn diagnostic_probe() -> anyhow::Result<()> {
        let Ok(scenario) = std::env::var("BRUSH_CLI_TEST_SCENARIO") else {
            return Ok(());
        };
        init_cli_logging()?;
        init_cli_logging()?;
        if scenario == "logging" {
            log::info!("CLI logger ready");
            log::debug!("CLI debug probe");
            log::trace!("CLI trace probe");
            return Ok(());
        }

        let mut initial = TrainStreamConfig::default();
        initial.train_config.total_train_iters = 17;
        initial.train_config.lod_levels = 2;
        initial.train_config.lod_refine_steps = 3;
        initial.train_config.max_splats = 321;
        initial.train_config.growth_stop_iter = 11;
        initial.load_config.max_resolution = 50;
        initial.process_config.start_iter = 2;
        initial.process_config.export_every = 9;
        let mut cli = TrainStreamConfig::default();
        if scenario == "merge-fallback" {
            // Preserve merge_configs' existing duplicate-argument fallback;
            // diagnostics must describe what training actually receives.
            cli.train_config.total_train_iters = 23;
        }
        cli.process_config.seed = 7;
        let mut config = brush_process::args_file::merge_configs(&initial, &cli);
        if scenario == "absolute-path" {
            config.process_config.export_path = std::env::temp_dir()
                .join("private-parent")
                .to_string_lossy()
                .into_owned();
            config.process_config.export_name = std::env::temp_dir()
                .join("private-parent")
                .join("export_{iter}.ply")
                .to_string_lossy()
                .into_owned();
        }
        let mut items = vec![Ok(ProcessMessage::TrainMessage(
            TrainMessage::TrainConfig {
                config: Box::new(config),
            },
        ))];
        if scenario == "ply" {
            items.push(Ok(ProcessMessage::StartLoading {
                name: "test.ply".into(),
                source: DataSource::Path("test.ply".into()),
                training: false,
                base_path: None,
            }));
        }
        if scenario == "warning" {
            items.push(Ok(ProcessMessage::Warning {
                error: anyhow::anyhow!("disk full").context("Export at iteration 9 failed"),
            }));
        }
        if matches!(
            scenario.as_str(),
            "success"
                | "merge-fallback"
                | "absolute-path"
                | "warning"
                | "done-then-panic"
                | "done-then-error"
        ) {
            items.push(Ok(ProcessMessage::TrainMessage(TrainMessage::DoneTraining)));
        }
        if matches!(scenario.as_str(), "error" | "done-then-error") {
            items.push(Err(
                anyhow::anyhow!("original I/O failure").context("training stage context")
            ));
        }
        let stream: std::pin::Pin<Box<dyn brush_process::ProcessStream>> =
            if scenario.contains("panic") {
                Box::pin(
                    tokio_stream::iter(items).chain(tokio_stream::once(()).map(|()| {
                        panic!("original GPU panic probe");
                    })),
                )
            } else {
                Box::pin(tokio_stream::iter(items))
            };
        let process = RunningProcess {
            stream,
            splat_view: brush_process::slot::Slot::empty(),
            device: brush_process::default_device(),
        };
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?
            .block_on(run_cli_ui(process, TrainStreamConfig::default()))
    }

    #[test]
    fn parses_source_and_overrides() {
        let cli = Cli::try_parse_from([
            "brush-cli",
            "some/dataset/path",
            "--total-train-iters",
            "50",
            "--eval-split-every",
            "2",
            "--max-resolution",
            "512",
            "--sh-degree",
            "2",
            "--seed",
            "7",
        ])
        .unwrap();

        assert!(matches!(
            &cli.source,
            Some(DataSource::Path(p)) if p == "some/dataset/path"
        ));
        // Passing a source flips the viewer default off.
        assert!(!cli.with_viewer);

        let ts = &cli.train_stream;
        assert_eq!(ts.train_config.total_train_iters, 50);
        assert_eq!(ts.train_config.total_iters(), 50); // No LOD levels by default.
        assert_eq!(ts.load_config.eval_split_every, Some(2));
        assert_eq!(ts.load_config.max_resolution, 512);
        assert_eq!(ts.model_config.sh_degree, 2);
        assert_eq!(ts.process_config.seed, 7);

        // A source without a viewer is a valid combination.
        assert!(cli.validate().is_ok());
    }

    #[test]
    fn parses_url_source() {
        let cli = Cli::try_parse_from(["brush-cli", "https://example.com/data.zip"]).unwrap();
        assert!(matches!(
            &cli.source,
            Some(DataSource::Url(u)) if u == "https://example.com/data.zip"
        ));
    }

    #[test]
    fn defaults_to_viewer_without_source() {
        let cli = Cli::try_parse_from(["brush-cli"]).unwrap();
        assert!(cli.source.is_none());
        assert!(cli.with_viewer);
        // Viewer without a source is valid (brush-app's default mode).
        assert!(cli.validate().is_ok());
    }

    #[test]
    fn viewer_flag_with_source() {
        let cli = Cli::try_parse_from(["brush-cli", "some/path", "--with-viewer"]).unwrap();
        assert!(cli.with_viewer);
        assert!(cli.source.is_some());
        assert!(cli.validate().is_ok());
    }

    #[test]
    fn validate_rejects_headless_without_source() {
        let mut cli = Cli::try_parse_from(["brush-cli"]).unwrap();
        cli.with_viewer = false;
        let Err(err) = cli.validate() else {
            panic!("expected validation error")
        };
        assert_eq!(err.kind(), ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn rejects_unknown_flag() {
        assert!(Cli::try_parse_from(["brush-cli", "--not-a-real-flag"]).is_err());
    }
}
