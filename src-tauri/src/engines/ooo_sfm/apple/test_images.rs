//! 测试用的合成图生成器（`features` / `matching` 的端到端测试共用）。
//!
//! 统一用"类自然图"：低频随机场（双线性放大）+ 硬边 + 少量高频细节。
//! 棋盘格对 SIFT 是病态输入（细尺度全是边缘、粗尺度却全是 blob），不要用。

use std::path::Path;

/// 合成"类自然"RGB 图（`seed` 决定随机纹理）。
pub(crate) fn synthetic_image(width: u32, height: u32, seed: u32) -> image::RgbImage {
    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            (self.0 >> 8) as f32 / 16777216.0
        }
    }
    let mut rng = Rng(seed | 1);
    let cell = 6usize;
    let (cw, ch) = (width as usize / cell + 3, height as usize / cell + 3);
    let coarse: Vec<f32> = (0..cw * ch).map(|_| rng.next()).collect();
    let mut buffer = image::RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / cell as f32;
            let fy = y as f32 / cell as f32;
            let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
            let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
            let top = coarse[y0 * cw + x0] * (1.0 - tx) + coarse[y0 * cw + x0 + 1] * tx;
            let bottom =
                coarse[(y0 + 1) * cw + x0] * (1.0 - tx) + coarse[(y0 + 1) * cw + x0 + 1] * tx;
            let mut value = top * (1.0 - ty) + bottom * ty;
            if (x > width / 4 && x < width / 4 + 3) || (y > height / 2 && y < height / 2 + 3) {
                value -= 0.35;
            }
            value += 0.04 * ((x * 7 + y * 13) as f32).sin();
            let value = (value * 255.0).clamp(0.0, 255.0) as u8;
            buffer.put_pixel(x, y, image::Rgb([value, value, value]));
        }
    }
    buffer
}

/// 把图像写到 `path`（自动建父目录）。
pub(crate) fn save_image(path: &Path, buffer: &image::RgbImage) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    buffer
        .save(path)
        .unwrap_or_else(|error| panic!("写测试图 {} 失败：{error}", path.display()));
}

/// 写一张合成图到 `path`。
pub(crate) fn write_image(path: &Path, width: u32, height: u32, seed: u32) {
    save_image(path, &synthetic_image(width, height, seed));
}

/// 写下同一张合成大图的两个平移裁剪：`path2` 相对 `path1` 的裁剪原点是 `(dx, dy)`。
///
/// 返回特征点的期望位移 `(x2 - x1, y2 - y1) = (-dx, -dy)`（无损 PNG，位移应精确）。
pub(crate) fn write_translated_pair(
    path1: &Path,
    path2: &Path,
    width: u32,
    height: u32,
    dx: u32,
    dy: u32,
    seed: u32,
) -> (i32, i32) {
    let big = synthetic_image(width + dx, height + dy, seed);
    let crop1 = image::imageops::crop_imm(&big, 0, 0, width, height).to_image();
    let crop2 = image::imageops::crop_imm(&big, dx, dy, width, height).to_image();
    save_image(path1, &crop1);
    save_image(path2, &crop2);
    (-(dx as i32), -(dy as i32))
}

/// 写一张带 EXIF `FocalLengthIn35mmFilm` 的 JPEG（`focal_35mm` 单位毫米）。
///
/// 用 `image` 的 JPEG 编码器生成基线 JPEG，再把构造好的 APP1/Exif 段插到 SOI 之后。
pub(crate) fn write_jpeg_with_exif_focal_35mm(
    path: &Path,
    width: u32,
    height: u32,
    focal_35mm: u16,
) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let buffer = synthetic_image(width, height, 0x51f7);
    let mut jpeg = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 92);
    encoder
        .encode_image(&buffer)
        .unwrap_or_else(|error| panic!("编码测试 JPEG 失败：{error}"));

    let tiff = super::exif::build_test_tiff(
        &[(0xA405, super::exif::TestTagValue::Short(focal_35mm))],
        true,
    );
    let mut payload = b"Exif\0\0".to_vec();
    payload.extend_from_slice(&tiff);

    let mut out = Vec::with_capacity(jpeg.len() + payload.len() + 4);
    out.extend_from_slice(&jpeg[..2]); // SOI
    out.extend_from_slice(&[0xFF, 0xE1]);
    out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(&payload);
    out.extend_from_slice(&jpeg[2..]);
    std::fs::write(path, out)
        .unwrap_or_else(|error| panic!("写测试 JPEG {} 失败：{error}", path.display()));
}
