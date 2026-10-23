use std::{env, fs, path::PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo"),
    );
    let endpoint_path = manifest_dir.join("../config/telemetry-endpoint.txt");
    println!("cargo:rerun-if-changed={}", endpoint_path.display());

    let endpoint = fs::read_to_string(&endpoint_path)
        .unwrap_or_else(|error| {
            panic!(
                "failed to read telemetry endpoint from {}: {error}",
                endpoint_path.display()
            )
        })
        .trim()
        .to_owned();
    assert!(
        !endpoint.is_empty(),
        "telemetry endpoint configuration must not be empty"
    );
    println!("cargo:rustc-env=OOOSPLAT_TELEMETRY_ENDPOINT={endpoint}");

    ffmpeg_abi_guard();
    tauri_build::build()
}

/// `media-rsmpeg` 的 ABI 守卫。
///
/// rsmpeg 的 FFI 层（`rusty_ffmpeg`）带**预置绑定**：它描述的是某个固定版本的
/// FFmpeg 结构体布局。系统 FFmpeg 的版本与之不符时，绑定与真实库不一致 → 运行期
/// 行为诡异（段错误、字段错位）。这里在**编译期**校验实际版本并拒绝。
///
/// 版本来源：设了 `FFMPEG_INCLUDE_DIR`（静态库构建路径，`npm run setup:media-ffmpeg`
/// 产出）时直接读同一份 `libavcodec/version.h` —— 与链接用的头文件同源，最可靠；
/// 否则回退 pkg-config 查询。
///
/// 期望：rsmpeg 0.18 ↔ FFmpeg 8.x（libavcodec 62.x）。
fn ffmpeg_abi_guard() {
    if std::env::var("CARGO_FEATURE_MEDIA_RSMPEG").is_err() {
        return;
    }
    let setup_hint = "`media-rsmpeg` 需要本地静态 FFmpeg 8.1.2（进程内解码的链接依赖）。\n\
         先构建一次：npm run setup:media-ffmpeg\n\
         产出 .cache/media-ffmpeg/ffmpeg-8.1.2-static，local-tauri 会自动设置相关环境变量。";
    if let Ok(libs_dir) = std::env::var("FFMPEG_LIBS_DIR") {
        if !PathBuf::from(&libs_dir).join("libavcodec.a").exists() {
            panic!(
                "FFMPEG_LIBS_DIR={libs_dir} 里没有 libavcodec.a。\n\
                 {setup_hint}\n\
                 （若要改用其它前缀，请同时设置 FFMPEG_INCLUDE_DIR 与 FFMPEG_LINK_MODE=static。）"
            );
        }
    }
    let version = match std::env::var("FFMPEG_INCLUDE_DIR") {
        Ok(include_dir) => {
            // FFmpeg 7+ 把主版本拆进 version_major.h（version.h 只有 MINOR/MICRO），
            // 两个文件都读，宏各自匹配。
            let mut text = String::new();
            for header_name in ["libavcodec/version.h", "libavcodec/version_major.h"] {
                let header = PathBuf::from(&include_dir).join(header_name);
                match fs::read_to_string(&header) {
                    Ok(file_text) => {
                        text.push('\n');
                        text.push_str(&file_text);
                    }
                    Err(error) => panic!("读不到 {}: {error}\n{setup_hint}", header.display()),
                }
            }
            let major = version_macro(&text, "LIBAVCODEC_VERSION_MAJOR").unwrap_or(0);
            let minor = version_macro(&text, "LIBAVCODEC_VERSION_MINOR").unwrap_or(0);
            let micro = version_macro(&text, "LIBAVCODEC_VERSION_MICRO").unwrap_or(0);
            format!("{major}.{minor}.{micro}")
        }
        Err(_) => {
            let output = std::process::Command::new("pkg-config")
                .args(["--modversion", "libavcodec"])
                .output();
            let Ok(output) = output else {
                panic!("`media-rsmpeg` 需要 FFmpeg 版本信息（没有 pkg-config，也没设 FFMPEG_INCLUDE_DIR）。\n{setup_hint}");
            };
            if !output.status.success() {
                panic!("`media-rsmpeg` 需要 FFmpeg 版本信息（pkg-config 查询失败，且没设 FFMPEG_INCLUDE_DIR）。\n{setup_hint}");
            }
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        }
    };
    // libavcodec 主版本：FFmpeg 8.x → 62.x，7.x → 61.x，9.x → 63.x。
    let major = version
        .split('.')
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0);
    if major != 62 {
        panic!(
            "`media-rsmpeg` 的预置 ABI 绑定对应 **FFmpeg 8.x**（libavcodec 62.x），\
             但当前报告的是 libavcodec {version}（主版本 {major}）。\n\
             版本不一致会导致运行期行为错乱，因此直接拒绝编译。\n\
             用锁定版本重建：npm run setup:media-ffmpeg\n\
             或者改用外部进程实现：`--features media-ffmpeg-cli`（默认）。"
        );
    }
    println!("cargo:warning=media-rsmpeg 使用 libavcodec {version}（ABI 与 rsmpeg 0.18 匹配）");
}

/// 从 `#define NAME value` 行里取整数值（FFmpeg 8 的 version.h 是纯数字宏）。
fn version_macro(text: &str, name: &str) -> Option<u32> {
    text.lines()
        .map(str::trim)
        .find_map(|line| {
            let rest = line.strip_prefix("#define")?.trim();
            let mut parts = rest.split_whitespace();
            let macro_name = parts.next()?;
            if macro_name != name {
                return None;
            }
            parts.next()?.parse::<u32>().ok()
        })
}
