# Brush

<video src=https://github.com/user-attachments/assets/5756967a-846c-44cf-bde9-3ca4c86f1a4d>A video showing various Brush features and scenes</video>

<p align="center">
  <i>
    Massive thanks to <a href="https://www.youtube.com/@gradeeterna">@GradeEterna</a> for the beautiful scenes
  </i>
</p>

Brush is a 3D reconstruction engine using [Gaussian splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). It works on a wide range of systems: **macOS/windows/linux**, **AMD/Nvidia/Intel** cards, **Android**, and in a **browser**. To achieve this, it uses WebGPU compatible tech and the [Burn](https://github.com/tracel-ai/burn) machine learning framework.

Machine learning for real time rendering has tons of potential, but most ML tools don't work well with it: Rendering requires realtime interactivity, usually involve dynamic shapes & computations, don't run on most platforms, and it can be cumbersome to ship apps with large CUDA deps. Brush on the other hand produces simple dependency free binaries, runs on nearly all devices, without any setup.

[**Try the web demo** <img src="https://cdn-icons-png.flaticon.com/256/888/888846.png" alt="chrome logo" width="24"/>
](https://arthurbrussee.github.io/brush-demo)
_NOTE: Only works on Chrome and Edge. Firefox and Safari are hopefully supported soon)_

[![](https://dcbadge.limes.pink/api/server/https://discord.gg/TbxJST2BbC)](https://discord.gg/TbxJST2BbC)

# Features

## Training

Brush takes in COLMAP data or datasets in the Nerfstudio format. Training is fully supported natively, on mobile, and in a browser. While training you can interact with the scene and see the training dynamics live, and compare the current rendering to input views as the training progresses.

It also supports masking images:
- Images with transparency. This will force the final splat to match the transparency of the input.
- A folder of images called 'masks'. This ignores parts of the image that are masked out.
  Black pixels in the mask are ignored, white pixels are kept. Pass `--invert-masks` if your masks are the other way around.

## Viewer
Brush also works well as a splat viewer, including on the web. It can load .ply & .compressed.ply files. You can stream in data from a URL (for a web app, simply append `?url=`).

Brush also can load .zip of splat files to display them as an animation, or a special ply that includes delta frames (see [cat-4D](https://cat-4d.github.io/) and [Cap4D](https://felixtaubner.github.io/cap4d/)!).

## CLI
Brush can be used as a CLI. Run `brush --help` to get an overview. Every CLI command can work with `--with-viewer` which also opens the UI, for easy debugging.

### CLI diagnostics (OOOBrush fork)

For subprocess consumers, completed training updates also produce `Training progress: iteration=… total=… elapsed_secs=… lod=…` at INFO. Output is throttled to once per second, with immediate first, LOD-change and final reports. The total includes LOD steps and uses the merged configuration. These host-side reports reuse `TrainStep`; they add no GPU synchronization or loss readback. A final step report does not establish process/export success. Consumers must still check the exit status and output file.

Enable INFO logs before starting either headless binary. In PowerShell, for example:

```powershell
$env:RUST_LOG = 'info'
$env:RUST_BACKTRACE = '1'
.\target\release\brush-cli.exe .\dataset --total-train-iters 10
# Alternatively, use the Brush application's headless entrypoint:
.\target\release\brush.exe .\dataset --total-train-iters 10
```

The existing `RUST_LOG` filter is respected; no default INFO or TRACE filter is added. Logs on stdout include the Brush version, GPU initialization and actual adapter information, final settings after merging `args.txt` and CLI arguments, loading/initialization durations, the first training step, and checkpoint export results. Stage timings measure host elapsed time without adding GPU synchronization. Existing progress, splat-count and evaluation output remains available. Errors and Rust panics retain their original context and stderr output; `RUST_BACKTRACE=1` enables Rust backtraces. A missing stage completion helps locate a panic, but does not identify its GPU cause. Training completion does not imply every export succeeded: export failures remain warnings.

OOOSplat captures stdout/stderr, but its default filter mainly enables `cubecl_wgpu`/`burn_wgpu`. To also capture these Brush diagnostics, supply `RUST_LOG=info`, or append `brush_cli=info,brush_process=info` to the existing filter while retaining its GPU directives. This fork does not change OOOSplat's filter. Empty adapter/driver information is reported as `unknown`; total VRAM is not queried. Diagnostic summaries omit absolute export directories, but original error messages and existing logs may contain paths. These diagnostics do not fix the underlying GPU crashes reported in OOOSplat #23/#40.

### Builds and releases (OOOBrush fork)

The [CLI diagnostics, builds and releases workflow](https://github.com/ooolabdev/OOOBrush/actions/workflows/cli-diagnostics.yml) checks pull requests and pushes to `main` without publishing. It tests diagnostics without a GPU and builds Windows x64, Linux x64 and macOS ARM64 artifacts containing both `brush-cli` and `brush` (with `.exe` on Windows), README, LICENSE, `BUILDINFO.json` (source commit and actual binary versions), and internal SHA-256 checksums. Actions artifacts remain available for 14 days. Extract the Windows ZIP or Unix tar.gz inside the platform artifact. Builds are unsigned, macOS is not notarized, and hosted-runner tests do not exercise GPU training.

To publish, push a fork release tag after committing the desired code:

```powershell
git tag ooo-v1.0.0
git push origin ooo-v1.0.0
```

Alternatively, open the workflow's **Run workflow** form, choose a branch and enter the required `release_tag`, such as `ooo-v1.0.0` or `ooo-v1.0.0-rc.1`. An existing tag builds that exact tag; a new tag builds the selected branch's run commit and is created at that SHA by the publish job. These commands are examples, not a claim that this version has already been released. Tags use `ooo-vMAJOR.MINOR.PATCH`; a prerelease suffix automatically marks the Release as a prerelease.

Only after checks and all three builds succeed does the workflow create a draft, upload all three packages plus an external `SHA256SUMS`, download and verify the attachments, and publish to [OOOBrush Releases](https://github.com/ooolabdev/OOOBrush/releases). A failed upload/verification leaves a draft that the same-tag, same-commit run can resume. Already public Releases are never overwritten; use a new version tag. Manual and automatic runs for the same tag share a concurrency group and do not cancel active publication. The workflow uses `GITHUB_TOKEN`, with `contents: write` only in the publish job; repository policy must permit Release creation.

Merge workflow changes into `main` before publishing. GitHub can reject tag/Release creation with `GITHUB_TOKEN` when the selected commit changes workflow files relative to the default branch; such errors fail the job and leave any draft unpublished.

For OOOSplat, use `brush-cli.exe` on Windows or `brush-cli` on Linux/macOS. Verify the downloaded archive against the Release's external `SHA256SUMS`; the extracted package also includes checksums for its files. The existing cargo-dist Release workflow handles other version tags and excludes `ooo-v*`. Preserve that exclusion if regenerating its workflow.

## Rerun

https://github.com/user-attachments/assets/f679fec0-935d-4dd2-87e1-c301db9cdc2c

While training, additional data can be visualized with the excellent [rerun](https://rerun.io/). To install rerun on your machine, please follow their [instructions](https://rerun.io/docs/getting-started/installing-viewer). Open the ./brush_blueprint.rbl in the viewer for best results.

## Building Brush
First install rust 1.88+. You can run tests with `cargo test --all`. Brush uses the wonderful [rerun](https://rerun.io/) for additional visualizations while training, run `cargo install rerun-cli` if you want to use it.

### Windows/macOS/Linux
Use `cargo run --release` from the workspace root to make an optimized build. Use `cargo run` to run a debug build. 

### Web
Brush can be compiled to WASM. Run `npm run dev` to start the demo website using Next.js, see the web directory in app/brush-app/web.

Brush uses [`wasm-pack`](https://drager.github.io/wasm-pack/) to build the WASM bundle. You can also use it without a bundler, see [wasm-pack's documentation](https://drager.github.io/wasm-pack/book/).

WebGPU is still an upcoming standard, and as such, only Chrome 134+ on Windows and macOS is currently supported.

### Android

As a one time setup, make sure you have the Android SDK & NDK installed.
- Check if ANDROID_NDK_HOME and ANDROID_HOME are set
- Add the Android target to rust `rustup target add aarch64-linux-android`
- Install cargo-ndk to manage building a lib `cargo install cargo-ndk`

Each time you change the rust code, run
- `cargo ndk -t arm64-v8a -o crates/brush-app/app/src/main/jniLibs/ build`
- Nb:  Nb, for best performance, build in release mode. This is separate
  from the Android Studio app build configuration.
- `cargo ndk -t arm64-v8a -o crates/brush-app/app/src/main/jniLibs/  build --release`

You can now either run the project from Android Studio (Android Studio does NOT build the rust code), or run it from the command line:
```
./gradlew build
./gradlew installDebug
adb shell am start -n com.splats.app/.MainActivity
```

You can also open this folder as a project in Android Studio and run things from there. Nb: Running in Android Studio does _not_ rebuild the rust code automatically.

## Benchmarks

Rendering and training are generally faster than gsplat. You can run benchmarks of some of the kernels using `cargo bench`.

# Acknowledgements

[**gSplat**](https://github.com/nerfstudio-project/gsplat), for their reference version of the kernels

**Peter Hedman, George Kopanas & Bernhard Kerbl**, for the many discussions & pointers.

**The Burn team**, for help & improvements to Burn along the way

**Raph Levien**, for the [original version](https://github.com/googlefonts/compute-shader-101/pull/31) of the GPU radix sort.

**GradeEterna**, for feedback and their scenes.

# Disclaimer

This is *not* an official Google product. This repository is a forked public version of [the google-research repository](https://github.com/google-research/google-research/tree/master/brush_splat)
