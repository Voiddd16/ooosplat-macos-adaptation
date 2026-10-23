#!/usr/bin/env bash
set -euo pipefail

# 把 FFmpeg 8.1.2 编成**静态库**，专供 `media-rsmpeg` feature 链接。
#
# # 为什么是静态链接（教训来自一次真实的启动崩溃）
#
# 第一版 media-rsmpeg 链的是 Homebrew keg 的 dylib，结果打包出的 .app 一启动就被
# dyld 击杀：
#
#   Library not loaded: '/opt/homebrew/.../libavcodec.62.dylib'
#   Reason: ... code signature ... not valid for use in process
#
# 两层问题：(1) keg dylib 是**绝对路径**引用，bundle 换台机器就断（官方引擎门禁的
# "Non-relocatable dependency" 检查防的就是它）；(2) 已签名的宿主进程加载签名状态
# 不合规的 dylib 会被 macOS 拒绝。静态链接把这两个问题**结构性消除**：没有 dylib、
# 没有 rpath、没有签名问题，二进制自包含。
#
# # 与官方引擎构建的关系
#
# 官方 bundle 的 FFmpeg（`build-engines-macos.sh`）是 **shared** + `@rpath`，跟着
# 引擎运行时一起分发并受完整性门禁保护——那是发行版语义。这里的静态库只服务
# `media-rsmpeg` 本地构建，不进 `engines/macos/arm64/`，不受门禁管。
# 若未来要把 media-rsmpeg 带进发行包，LGPL 静态链接需要回退到 shared 方案。
#
# 幂等：`lib/READY` 存在即跳过。删除该文件可强制重建。

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "media FFmpeg static build requires an Apple Silicon Mac." >&2
  exit 1
fi

for command_name in clang make shasum tar pkg-config; do
  command -v "$command_name" >/dev/null || { echo "Missing build command: $command_name" >&2; exit 1; }
done

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache="$workspace/.cache/media-ffmpeg"
sources="$cache/sources"
build="$cache/build"
prefix="$cache/ffmpeg-8.1.2-static"

read_manifest() {
  node -e 'const m=require(process.argv[1]); const e=m.engines.find(e=>e.name===process.argv[2]); process.stdout.write(String(e[process.argv[3]]));' \
    "$workspace/engines/manifest.macos.json" "FFmpeg / FFprobe" "$1"
}

source_url="$(read_manifest sourceUrl)"
source_sha256="$(read_manifest sourceSha256)"
# 部署目标用本机系统主版本（发行 manifest 的 15.0 是为官方引擎定的；本地库要能在
# 本机跑，比如 macOS 12.7 → 12.0）。
host_major="$(sw_vers -productVersion | cut -d. -f1)"
deployment_target="${host_major}.0"
jobs="${OOOSPLAT_BUILD_JOBS:-$(sysctl -n hw.logicalcpu)}"

if [[ -f "$prefix/lib/READY" ]]; then
  echo "FFmpeg 8.1.2 static libraries already built: $prefix"
  echo "（强制重建：rm -rf $prefix 后重跑）"
  exit 0
fi

mkdir -p "$sources" "$build"

archive="$sources/ffmpeg-8.1.2.tar.xz"
if [[ ! -f "$archive" ]] || [[ "$(shasum -a 256 "$archive" | awk '{print toupper($1)}')" != "$source_sha256" ]]; then
  echo "Downloading FFmpeg 8.1.2 source from $source_url ..."
  curl --fail --location --retry 3 "$source_url" --output "$archive"
fi
[[ "$(shasum -a 256 "$archive" | awk '{print toupper($1)}')" == "$source_sha256" ]] || {
  echo "Source SHA-256 mismatch: $archive" >&2
  exit 1
}

rm -rf -- "$build/ffmpeg-source" "$prefix"
mkdir -p "$build/ffmpeg-source"
tar -xJf "$archive" -C "$build/ffmpeg-source" --strip-components=1

(
  cd "$build/ffmpeg-source"
  # 与官方构建同一套能力策略：LGPL、无 GPL/非自由组件、无外部依赖
  # （--disable-autodetect）。差异只有两处：静态（--enable-static）+ 不出可执行文件
  # （--disable-programs，进程内解码用不到 CLI）。
  MACOSX_DEPLOYMENT_TARGET="$deployment_target" ./configure \
    --prefix="$prefix" \
    --arch=arm64 \
    --target-os=darwin \
    --cc=clang \
    --enable-static \
    --disable-shared \
    --disable-gpl \
    --disable-nonfree \
    --disable-programs \
    --disable-ffplay \
    --disable-doc \
    --disable-debug \
    --disable-autodetect \
    --extra-cflags="-mmacosx-version-min=$deployment_target" \
    --extra-cxxflags="-mmacosx-version-min=$deployment_target" \
    --extra-ldflags="-mmacosx-version-min=$deployment_target"
  make -j"$jobs"
  make install
)

rm -rf -- "$build/ffmpeg-source"

# 纯静态断言：lib 里不允许出现任何 dylib。
if find "$prefix/lib" -name '*.dylib' | grep -q .; then
  echo "Unexpected shared libraries in static prefix:" >&2
  find "$prefix/lib" -name '*.dylib' >&2
  exit 1
fi

archive_count="$(find "$prefix/lib" -maxdepth 1 -name '*.a' | wc -l | tr -d ' ')"
if [[ "$archive_count" -lt 7 ]]; then
  echo "Expected 7+ static FFmpeg libraries, found $archive_count:" >&2
  ls "$prefix/lib" >&2
  exit 1
fi

# 记录源码指纹（可追溯），然后打就绪标记。
{
  echo "ffmpeg-8.1.2.tar.xz  sha256=$source_sha256"
  echo "deployment-target=$deployment_target"
  echo "policy=LGPL-2.1-or-later static, no external deps"
} > "$prefix/SOURCES.sha256"
touch "$prefix/lib/READY"

echo ""
echo "✓ FFmpeg 8.1.2 static libraries: $prefix/lib"
echo "  ABI check: PKG_CONFIG_PATH=$prefix/lib/pkgconfig pkg-config --modversion libavcodec"
echo "  media-rsmpeg 构建会自动使用这个前缀（local-tauri.mjs 自动设置环境变量）。"
