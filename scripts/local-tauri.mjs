import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { brushLock, hostPlatform, prepareBrush, verifyBrushCli } from "./brush-runtime.mjs";

const workspace = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// Only these explicit local commands bypass COLMAP archive/manifest checks.
// Normal setup, build:bundle, package:windows and application CI are unchanged.
// options（全部默认关闭，纯函数便于测试）：
//   inProcessMedia：启用 media-rsmpeg 时，ffmpeg/ffprobe 二进制不需要打包（进程内实现）。
//   skipBrush：OOO_SKIP_BRUSH=1 时，brush_app 不打包（骨架里没有这个文件，打包会崩）。
//   hostMinimumSystemVersion：本地包不限 15.0 —— 发行 manifest 的 minOS 本是为官方
//     定制引擎定的；本地 bundle 用本机系统版本，否则 macOS 12 上 .app 无法启动。
export function localConfiguration(platform, options = {}) {
  if (!["win32", "linux", "darwin"].includes(platform)) throw new Error(`Unsupported local platform: ${platform}`);
  const { inProcessMedia = false, skipBrush = false, hostMinimumSystemVersion = null } = options;
  const config = { build: { beforeBuildCommand: "npm run build" }, bundle: { resources: {} } };
  const roots = {
    win32: "../engines/colmap",
    linux: "../engines/linux/colmap",
    darwin: "../engines/macos/arm64/colmap",
  };
  const destinations = {
    win32: "engines/colmap",
    linux: "engines/linux/colmap",
    darwin: "engines/macos/arm64/colmap",
  };
  const formalColmapResources = {
    win32: ["../engines/colmap/bin/", "../engines/colmap/lib/", "../engines/colmap/licenses/", "../engines/colmap/BUILD-INFO.json", "../engines/colmap/BUNDLED-COMPONENTS.json", "../engines/colmap/SHA256SUMS"],
    linux: ["../engines/linux/colmap/bin/", "../engines/linux/colmap/lib/", "../engines/linux/colmap/licenses/", "../engines/linux/colmap/BUILD-INFO.json", "../engines/linux/colmap/BUNDLED-COMPONENTS.json", "../engines/linux/colmap/SHA256SUMS"],
    darwin: ["../engines/macos/arm64/colmap/"],
  };
  for (const resource of formalColmapResources[platform]) config.bundle.resources[resource] = null;
  config.bundle.resources[`${roots[platform]}/`] = `${destinations[platform]}/`;
  if (platform === "darwin") {
    const root = "../engines/macos/arm64";
    const localBinaries = {
      [`${root}/bin/ffmpeg`]: "engines/macos/arm64/bin/ffmpeg",
      [`${root}/bin/ffprobe`]: "engines/macos/arm64/bin/ffprobe",
      [`${root}/bin/brush_app`]: "engines/macos/arm64/bin/brush_app",
    };
    // 按本地能力裁剪：进程内媒体实现不吃 ffmpeg/ffprobe；显式跳过就不背 brush。
    if (inProcessMedia) {
      delete localBinaries[`${root}/bin/ffmpeg`];
      delete localBinaries[`${root}/bin/ffprobe`];
    }
    if (skipBrush) {
      delete localBinaries[`${root}/bin/brush_app`];
    }
    Object.assign(config.bundle, {
      macOS: {
        signingIdentity: "-",
        ...(hostMinimumSystemVersion ? { minimumSystemVersion: hostMinimumSystemVersion } : {}),
      },
      resources: {
        ...config.bundle.resources,
        // RFC 7396 nulls remove mixed-runtime resources from the local merge.
        [`${root}/bin/`]: null,
        [`${root}/SHA256SUMS`]: null,
        [`${root}/BUILD-INFO.json`]: null,
        [`${root}/BUNDLED-COMPONENTS.json`]: null,
        ...localBinaries,
      },
    });
  }
  return config;
}

// Feature 列表（纯函数，便于测试）。
//
// `local-colmap`：本地开发用的 COLMAP 构建（主分支既有语义，不变）。
// `backend-ooo-sfm`：**macOS 默认启用自研 SfM 后端**（M10「macOS 构建默认启用 feature」）。
//   自研后端是进程内实现，不需要 COLMAP 二进制；`engines/health.rs` 在该 feature 下
//   也不再探测二进制（见 `engines/ooo_sfm/health.rs`）。
//   Win/Linux 仍走 COLMAP 后端，所以这里不加 —— 与 `ooo_sfm/mod.rs` 的
//   `compile_error!` 一致（非 Apple 平台禁止启用 `backend-ooo-sfm`）。
// Metal（`backend-ooo-sfm-metal`）不默认开：GPU 内核要显式要求，避免有人不知道自己在跑
//   f32 内核。命令行入口读 `OOO_SFM_METAL=1` 打开（见文件末尾）。
export function localFeatures(platform) {
  if (!["win32", "linux", "darwin"].includes(platform)) throw new Error(`Unsupported local platform: ${platform}`);
  return platform === "darwin" ? ["local-colmap", "backend-ooo-sfm"] : ["local-colmap"];
}

// Brush 运行时准备失败时的策略（纯函数，便于测试）。
//
//   build：必须成功 —— 打包会把 brush_app 写进 bundle，缺了就产出一个坏包。
//   dev  ：只警告不阻塞 —— dev 只想把 app 拉起来看 SfM 链路，Brush/训练类任务
//          真正跑起来时引擎门禁会给出准确报错，比在这里因为一次网络抖动
//          ETIMEDOUT 就整个起不来要好。
//   OOO_SKIP_BRUSH=1：显式跳过（连警告都不打）。
export function brushPreparationPolicy(action, skipEnv) {
  if (skipEnv === "1") return "skipped";
  return action === "build" ? "required" : "warn";
}

export function localInvocation(action, platform, extra = [], features = localFeatures(platform), configOptions = {}) {
  if (!["dev", "build"].includes(action)) throw new Error("Usage: node scripts/local-tauri.mjs dev|build [Tauri arguments]");
  // Invocation construction is pure. Only Brush is prepared before execution;
  // manually placed COLMAP and existing FFmpeg are never verified/downloaded.
  return {
    command: process.execPath,
    features: [...features],
    args: [path.join(workspace, "node_modules", "@tauri-apps", "cli", "tauri.js"), action,
      "--features", features.join(","), "--config", JSON.stringify(localConfiguration(platform, configOptions)), ...extra],
  };
}

// 启动前的引擎体检（只报事实，不做任何下载/校验）。
//
// macOS 的 `EnginePaths` 按设计**不**回退 PATH（`from_candidates` 的注释：发行版必须
// 自包含），所以 Homebrew 里的 ffmpeg 不算数 —— 缺就是缺，这里提前说清楚，
// 免得等到任务启动才看到一个笼统的"引擎缺失"。
export function enginePreflight(platform, features) {
  const bin = platform === "darwin"
    ? path.join(workspace, "engines", "macos", "arm64", "bin")
    : platform === "win32"
      ? path.join(workspace, "engines", "colmap", "bin")
      : path.join(workspace, "engines", "linux", "colmap", "bin");
  const selfHosted = features.includes("backend-ooo-sfm");
  // media-rsmpeg：媒体处理在进程内完成，本机不需要 ffmpeg/ffprobe 二进制。
  const inProcessMedia = features.includes("media-rsmpeg");
  const wanted = [
    ...inProcessMedia ? [] : [["ffmpeg", "ffmpeg"], ["ffprobe", "ffprobe"]],
    ...(selfHosted ? [] : [["colmap", "COLMAP"]]),
    ["brush_app", "OOOBrush"],
  ];
  return wanted.map(([file, label]) => ({
    label,
    file,
    present: fs.existsSync(path.join(bin, file)),
  }));
}

function reportEnginePreflight(platform, features) {
  if (!fs.existsSync(path.join(workspace, "engines"))) return;
  const rows = enginePreflight(platform, features);
  const missing = rows.filter(row => !row.present).map(row => row.label);
  if (missing.length === 0) return;
  console.warn(`[preflight] ${platform} 缺少引擎：${missing.join(", ")}（期望位置 engines/*/bin）。`);
  console.warn("[preflight] macOS 不回退 PATH/环境变量：Homebrew 里的 ffmpeg 不算数。");
  console.warn("[preflight] 获取方式见 README 的「引擎」章节；OOOBrush 见 scripts/brush-runtime.mjs。");
}

// 本机系统版本（本地 bundle 用它当 minimumSystemVersion；发行版仍锁 15.0）。
function hostMacosVersion() {
  try {
    return spawnSync("/usr/bin/sw_vers", ["-productVersion"], { encoding: "utf8" }).stdout?.trim() || null;
  } catch {
    return null;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const action = process.argv[2];
  // 环境变量只在这一层读，保持 `localInvocation` / `localFeatures` 是纯函数。
  const features = localFeatures(process.platform);
  if (process.env.OOO_SFM_METAL === "1" && process.platform === "darwin") features.push("backend-ooo-sfm-metal");
  if (process.env.OOO_MEDIA_RSMPEG === "1" && process.platform === "darwin") features.push("media-rsmpeg");

  // media-rsmpeg 在链接期要 FFmpeg 8.x 的头文件/静态库（build.rs 的 ABI 守卫会再校验版本）。
  // 链**静态库**（`npm run setup:media-ffmpeg` 产出的 .a）：Homebrew keg 的 dylib 是
  // 绝对路径引用，会原样打进二进制 —— 打包出的 .app 换台机器就崩（真实事故：dyld
  // "Library not loaded: /opt/homebrew/.../libavcodec.62.dylib"，外加签名拒绝）。
  // 静态链接没有 dylib/rpath/签名问题，bundle 天然自包含。
  if (features.includes("media-rsmpeg")) {
    const prefix = path.join(workspace, ".cache", "media-ffmpeg", "ffmpeg-8.1.2-static");
    if (!fs.existsSync(path.join(prefix, "lib", "READY"))) {
      console.error("[media-rsmpeg] 缺少本地静态 FFmpeg 库 —— 先运行：npm run setup:media-ffmpeg");
      console.error("             （首次编译约 5-10 分钟，之后幂等跳过。ABI 体检：npm run check:ffmpeg-abi。）");
      process.exit(1);
    }
    process.env.FFMPEG_INCLUDE_DIR = path.join(prefix, "include");
    process.env.FFMPEG_LIBS_DIR = path.join(prefix, "lib");
    process.env.FFMPEG_LINK_MODE = "static";
    // PKG_CONFIG_PATH 供 build.rs 的 ABI 守卫与 check:ffmpeg-abi 使用
    // （rusty_ffmpeg 本身走 FFMPEG_LIBS_DIR，不走 pkg-config）。
    const pkgconfig = path.join(prefix, "lib", "pkgconfig");
    process.env.PKG_CONFIG_PATH = process.env.PKG_CONFIG_PATH
      ? `${pkgconfig}:${process.env.PKG_CONFIG_PATH}`
      : pkgconfig;
    console.log(`[media-rsmpeg] 静态 FFmpeg：${prefix}（FFMPEG_LINK_MODE=static，不依赖 Homebrew）。`);
  }

  // 本地 bundle 的资源按本机能力裁剪：进程内媒体不带 ffmpeg/ffprobe；
  // OOO_SKIP_BRUSH=1 不背 brush；minOS 用本机系统版本（发行版锁 15.0 的原因
  // 是官方定制引擎，本地 bundle 不装它）。
  const invocation = localInvocation(action, process.platform, process.argv.slice(3), features, {
    inProcessMedia: features.includes("media-rsmpeg"),
    skipBrush: process.env.OOO_SKIP_BRUSH === "1",
    hostMinimumSystemVersion: process.platform === "darwin" ? hostMacosVersion() : null,
  });
  if (!process.argv.slice(3).some(arg => ["--help", "-h", "--version", "-V"].includes(arg))) {
    const policy = brushPreparationPolicy(action, process.env.OOO_SKIP_BRUSH);
    if (policy === "required") {
      prepareBrush(hostPlatform());
    } else {
      try {
        prepareBrush(hostPlatform());
      } catch (error) {
        const detail = error?.message ?? String(error);
        if (policy === "skipped") {
          console.log(`[OOOBrush] 按 OOO_SKIP_BRUSH=1 跳过准备（${detail}）。`);
        } else {
          console.error("");
          console.error(`[OOOBrush] 准备失败，dev 继续启动：${detail}`);
          console.error("           Brush / 训练类任务会因引擎缺失失败；需要时可用");
          console.error("           `npm run setup:brush` 重试，或手动放置 brush_app。");
          console.error("           只想安静跳过请设 OOO_SKIP_BRUSH=1。");
          console.error("");
        }
      }
    }
    // 本地调试用：允许用自编译的 brush-cli 覆盖官方运行时。仅本地，
    // 因为二进制没有对应的官方哈希；仍校验 CLI 版本和必需参数。
    const localBrush = process.env.OOO_BRUSH_LOCAL_BINARY;
    if (localBrush) {
      const lock = brushLock(hostPlatform());
      const source = path.resolve(workspace, localBrush);
      const destination = path.resolve(workspace, lock.root, lock.binary);
      if (!fs.lstatSync(source).isFile()) throw new Error(`OOO_BRUSH_LOCAL_BINARY is not a file: ${source}`);
      verifyBrushCli(source, lock);
      fs.copyFileSync(source, destination);
      fs.chmodSync(destination, 0o755);
      console.log(`[OOOBrush] 使用本地构建覆盖：${source} -> ${destination}（跳过官方哈希校验）`);
    }
  }
  reportEnginePreflight(process.platform, invocation.features);
  console.log(`Features: ${invocation.features.join(", ")}. Using pinned OOOBrush CLI; manually placed COLMAP skips package verification and downloads.${features.includes("media-rsmpeg") ? " Media runs in-process (rsmpeg), no external ffmpeg/ffprobe needed." : " FFmpeg is unchanged."}`);
  const result = spawnSync(invocation.command, invocation.args, { cwd: workspace, stdio: "inherit", windowsHide: true });
  if (result.error) throw result.error;
  process.exit(result.status ?? 1);
}
