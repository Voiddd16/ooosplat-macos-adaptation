// 媒体后端 ABI 体检：确认本机 FFmpeg 与 rsmpeg 的预置绑定同版本。
//
// rsmpeg 的 FFI 层（rusty_ffmpeg）带**预置绑定**，描述的是固定版本的 FFmpeg 结构体
// 布局。系统 FFmpeg 版本与之不符时，绑定与真实库不一致 → 编译期类型错误
// （`expected u32, found i32`）或运行期字段错位。
//
// rsmpeg 0.18 ↔ FFmpeg 8.x（libavcodec 62.x）。
//
// 首选来源是 `npm run setup:media-ffmpeg` 产出的**静态库**前缀 —— media-rsmpeg
// 的打包构建必须走它：Homebrew keg 的 dylib 是绝对路径引用，会打进二进制，
// 打包出的 .app 换台机器就崩（真实事故：dyld "Library not loaded:
// /opt/homebrew/.../libavcodec.62.dylib" + 签名拒绝）。keg 仅作为信息项列出。
//
// 用法：node scripts/check-ffmpeg-abi.mjs

import { execFileSync } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const workspace = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const staticPrefix = path.join(workspace, ".cache", "media-ffmpeg", "ffmpeg-8.1.2-static");

const EXPECTED_MAJOR = "62"; // libavcodec 主版本：FFmpeg 8.x
const CANDIDATES = [
  { name: "静态库（setup:media-ffmpeg 产出，构建用它）", pkgConfigPath: path.join(staticPrefix, "lib", "pkgconfig") },
  { name: "ffmpeg@8 (Homebrew keg，仅参考)", pkgConfigPath: "/opt/homebrew/opt/ffmpeg@8/lib/pkgconfig" },
  { name: "系统默认（仅参考）", pkgConfigPath: "" },
];

function probe(pkgConfigPath) {
  try {
    const env = { ...process.env };
    if (pkgConfigPath) env.PKG_CONFIG_PATH = pkgConfigPath;
    else delete env.PKG_CONFIG_PATH;
    const version = execFileSync("pkg-config", ["--modversion", "libavcodec"], {
      env,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
    return version;
  } catch {
    return null;
  }
}

let matched = null;
for (const candidate of CANDIDATES) {
  if (candidate.pkgConfigPath && !existsSync(candidate.pkgConfigPath)) continue;
  const version = probe(candidate.pkgConfigPath);
  if (!version) continue;
  const major = version.split(".")[0];
  const mark = major === EXPECTED_MAJOR ? "✓ 匹配" : "✗ 不匹配";
  console.log(`${candidate.name}: libavcodec ${version} ${mark}`);
  if (major === EXPECTED_MAJOR && !matched) matched = candidate;
}

// 构建可用性 = 静态前缀就绪（不只是版本匹配）。
const staticReady = existsSync(path.join(staticPrefix, "lib", "READY")) &&
  existsSync(path.join(staticPrefix, "lib", "libavcodec.a"));

if (!matched || !staticReady) {
  console.error("");
  if (!staticReady) {
    console.error("media-rsmpeg 构建需要的静态 FFmpeg 前缀尚未就绪（.cache/media-ffmpeg/ffmpeg-8.1.2-static）。");
    console.error("执行：npm run setup:media-ffmpeg（首次约 5-10 分钟，之后幂等跳过）。");
  }
  if (matched) {
    console.error("");
    console.error(`版本匹配但来源不可用于打包（${matched.pkgConfigPath || "系统默认"}）。`);
  } else {
    console.error("");
    console.error("没有找到与 rsmpeg 0.18 匹配（FFmpeg 8.x / libavcodec 62.x）的 FFmpeg。");
    console.error("执行：npm run setup:media-ffmpeg（源码与版本由 engines/manifest.macos.json 锁定）。");
  }
  console.error("");
  console.error("然后正常构建：OOO_MEDIA_RSMPEG=1 npm run build:local -- --debug");
  process.exitCode = 1;
} else {
  const archives = readdirSync(path.join(staticPrefix, "lib")).filter((name) => name.endsWith(".a")).length;
  console.error("");
  console.error(`✓ 可以构建：静态前缀就绪（${archives} 个 .a），版本 libavcodec ${probe(path.join(staticPrefix, "lib", "pkgconfig"))}。`);
  console.error("  构建命令：OOO_MEDIA_RSMPEG=1 npm run build:local -- --debug（环境变量自动设置）。");
}
