//! EXIF 焦点长度解析（4.2.1 `Bitmap::ExifFocalLength()` 的等价实现）。
//!
//! 优先级与 4.2.1 一致：
//! 1. `FocalLengthIn35mmFilm`（CIPA 公式）：`f35 / 43.27 × √(w² + h²)`；
//! 2. `FocalLength` + `FocalPlaneXResolution` + `FocalPlaneResolutionUnit`
//!    （2 = 英寸、3 = 厘米、4 = 毫米、5 = 微米）：`f_mm × 像素/毫米`；
//! 3. 4.2.1 的第三级是「Make/Model → 传感器宽度」内置库；我们**没有内置该库**
//!    （刻意差异，见方案文档），到这里直接返回 `None`，由调用方回落
//!    `default_focal_length_factor × max(w, h)`（此时 `prior_focal_length = 0`）。
//!
//! 支持 JPEG（APP1/Exif 段）与 TIFF；PNG 的 `eXIf` chunk 不支持（数据集里罕见，
//! 同样记录在案）。解析器只读文件头部固定字节数，遇到任何截断/畸形结构都返回
//! `None`，绝不 panic。

use std::fs::File;
use std::io::Read;
use std::path::Path;

/// 只扫文件头部这么多字节。JPEG 单个 APP1 段受 64 KiB 段长限制，4 倍余量足够；
/// TIFF 的 IFD0 也总在头部。
const HEAD_SCAN_LIMIT: usize = 256 * 1024;

/// 「35 mm 相机」像幅对角线（毫米，36×24 像幅）。
const DIAGONAL_35MM: f64 = 43.27;

/// 从图像文件读 EXIF 焦点长度（像素单位）；`width` / `height` 是**原图**尺寸。
///
/// 仅对 JPEG / TIFF 扩展名尝试读取（避免为 PNG 等白白读一次文件）。
pub fn file_focal_length(path: &Path, width: u64, height: u64) -> Option<f64> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    if !matches!(extension.as_str(), "jpg" | "jpeg" | "tif" | "tiff") {
        return None;
    }
    let mut file = File::open(path).ok()?;
    let mut buffer = vec![0u8; HEAD_SCAN_LIMIT];
    let read = file.read(&mut buffer).ok()?;
    buffer.truncate(read);
    focal_length(&buffer, width, height)
}

/// 从（JPEG 或 TIFF 的）字节流解析 EXIF 焦点长度。
pub fn focal_length(data: &[u8], width: u64, height: u64) -> Option<f64> {
    let tags = TiffTags::parse(tiff_payload(data)?)?;

    // 1) 35 mm 等效焦距。
    if let Some(focal_35mm) = tags.focal_length_35mm {
        if focal_35mm > 0.0 {
            let diagonal = ((width * width + height * height) as f64).sqrt();
            return Some(focal_35mm / DIAGONAL_35MM * diagonal);
        }
    }

    // 2) 物理焦距 + 焦平面分辨率（单位换算与 4.2.1 相同）。
    let focal_length_mm = tags.focal_length_mm.filter(|value| *value > 0.0)?;
    let pixels_per_mm = match (
        tags.focal_plane_x_resolution,
        tags.focal_plane_resolution_unit,
    ) {
        (Some(x_resolution), Some(unit)) if (2..=5).contains(&unit) => match unit {
            2 => x_resolution / 25.4,  // 英寸
            3 => x_resolution / 10.0,  // 厘米
            4 => x_resolution,         // 毫米
            5 => x_resolution * 1000.0, // 微米
            _ => unreachable!("unit 已由 match guard 限定在 2..=5"),
        },
        _ => return None,
    };
    if !(pixels_per_mm > 0.0) {
        return None;
    }
    Some(focal_length_mm * pixels_per_mm)
}

/// 取出 Exif 块内的 TIFF 数据：JPEG 扫 APP1 段，TIFF 直接整块返回。
fn tiff_payload(data: &[u8]) -> Option<&[u8]> {
    if data.starts_with(b"II*\0") || data.starts_with(b"MM\0*") {
        return Some(data);
    }
    if data.len() < 2 || data[0] != 0xFF || data[1] != 0xD8 {
        return None; // 不是 JPEG
    }

    let mut offset = 2usize;
    while offset + 4 <= data.len() {
        if data[offset] != 0xFF {
            return None; // 段结构损坏
        }
        let marker = data[offset + 1];
        match marker {
            // 无长度字段的标记。
            0x01 | 0xD0..=0xD7 | 0xD8 => {
                offset += 2;
                continue;
            }
            // SOS / EOI：Exif 只会在其之前。
            0xDA | 0xD9 => return None,
            _ => {}
        }
        let length = u16::from_be_bytes([data[offset + 2], data[offset + 3]]) as usize;
        if length < 2 {
            return None;
        }
        let end = offset + 2 + length;
        if end > data.len() {
            return None;
        }
        if marker == 0xE1 {
            let payload = &data[offset + 4..end];
            if payload.len() >= 6 && &payload[..6] == b"Exif\0\0" {
                let tiff = &payload[6..];
                if tiff.starts_with(b"II*\0") || tiff.starts_with(b"MM\0*") {
                    return Some(tiff);
                }
            }
            // 非 Exif 的 APP1（XMP 等）：继续找下一个段。
        }
        offset = end;
    }
    None
}

/// 我们需要的 5 个标签（IFD0 与 Exif 子 IFD 合并读取，子 IFD 覆盖）。
#[derive(Default, Clone, Copy)]
struct TiffTags {
    focal_length_35mm: Option<f64>,
    focal_length_mm: Option<f64>,
    focal_plane_x_resolution: Option<f64>,
    focal_plane_resolution_unit: Option<u32>,
}

impl TiffTags {
    fn parse(tiff: &[u8]) -> Option<Self> {
        let little_endian = match tiff.get(..4)? {
            b"II*\0" => true,
            b"MM\0*" => false,
            _ => return None,
        };
        let mut tags = Self::default();
        let ifd0_offset = Self::read_u32(tiff, 4, little_endian)? as usize;
        let exif_ifd_offset = Self::read_ifd(tiff, ifd0_offset, little_endian, &mut tags)?;
        if exif_ifd_offset != 0 {
            // 子 IFD 读取失败不致命：保守使用已读到的 IFD0 标签。
            let _ = Self::read_ifd(tiff, exif_ifd_offset as usize, little_endian, &mut tags);
        }
        Some(tags)
    }

    /// 读一个 IFD；返回指向 Exif 子 IFD 的偏移（0 = 无）。
    fn read_ifd(
        tiff: &[u8],
        offset: usize,
        little_endian: bool,
        tags: &mut Self,
    ) -> Option<u32> {
        let count = Self::read_u16(tiff, offset, little_endian)? as usize;
        let mut exif_ifd_offset = 0u32;
        for index in 0..count {
            let entry = offset
                .checked_add(2)?
                .checked_add(index.checked_mul(12)?)?;
            let tag = Self::read_u16(tiff, entry, little_endian)?;
            let field_type = Self::read_u16(tiff, entry + 2, little_endian)?;
            let value_count = Self::read_u32(tiff, entry + 4, little_endian)?;
            let value_field = entry + 8;
            match tag {
                // ExifIFDPointer
                0x8769 => exif_ifd_offset = Self::read_u32(tiff, value_field, little_endian)?,
                // FocalLengthIn35mmFilm / FocalPlaneResolutionUnit（SHORT ×1）
                0xA405 if field_type == SHORT && value_count == 1 => {
                    tags.focal_length_35mm =
                        Some(Self::read_u16(tiff, value_field, little_endian)? as f64);
                }
                0xA210 if field_type == SHORT && value_count == 1 => {
                    tags.focal_plane_resolution_unit =
                        Some(Self::read_u16(tiff, value_field, little_endian)? as u32);
                }
                // FocalLength / FocalPlaneXResolution（RATIONAL ×1）
                0x920A if field_type == RATIONAL && value_count == 1 => {
                    tags.focal_length_mm =
                        Self::read_rational(tiff, value_field, little_endian);
                }
                0xA20E if field_type == RATIONAL && value_count == 1 => {
                    tags.focal_plane_x_resolution =
                        Self::read_rational(tiff, value_field, little_endian);
                }
                _ => {}
            }
        }
        Some(exif_ifd_offset)
    }

    /// RATIONAL 的值是指向数据区的偏移。
    fn read_rational(tiff: &[u8], value_field: usize, little_endian: bool) -> Option<f64> {
        let offset = Self::read_u32(tiff, value_field, little_endian)? as usize;
        let numerator = Self::read_u32(tiff, offset, little_endian)? as f64;
        let denominator = Self::read_u32(tiff, offset + 4, little_endian)? as f64;
        if denominator == 0.0 {
            return None;
        }
        Some(numerator / denominator)
    }

    fn read_u16(data: &[u8], offset: usize, little_endian: bool) -> Option<u16> {
        let bytes = data.get(offset..offset + 2)?;
        let pair = [bytes[0], bytes[1]];
        Some(if little_endian {
            u16::from_le_bytes(pair)
        } else {
            u16::from_be_bytes(pair)
        })
    }

    fn read_u32(data: &[u8], offset: usize, little_endian: bool) -> Option<u32> {
        let bytes = data.get(offset..offset + 4)?;
        let quad = [bytes[0], bytes[1], bytes[2], bytes[3]];
        Some(if little_endian {
            u32::from_le_bytes(quad)
        } else {
            u32::from_be_bytes(quad)
        })
    }
}

const SHORT: u16 = 3;
const RATIONAL: u16 = 5;

// ---------------------------------------------------------------- 测试

/// 构造测试用 TIFF 的最小标签值。
#[cfg(test)]
pub(crate) enum TestTagValue {
    /// FocalLengthIn35mmFilm / FocalPlaneResolutionUnit。
    Short(u16),
    /// FocalLength / FocalPlaneXResolution。
    Rational(u32, u32),
}

/// 构造一个只含 Exif 子 IFD（经 IFD0 的 0x8769 指针可达）的最小 TIFF。
#[cfg(test)]
pub(crate) fn build_test_tiff(entries: &[(u16, TestTagValue)], little_endian: bool) -> Vec<u8> {
    let write_u16 = |out: &mut Vec<u8>, value: u16| {
        if little_endian {
            out.extend_from_slice(&value.to_le_bytes());
        } else {
            out.extend_from_slice(&value.to_be_bytes());
        }
    };
    let write_u32 = |out: &mut Vec<u8>, value: u32| {
        if little_endian {
            out.extend_from_slice(&value.to_le_bytes());
        } else {
            out.extend_from_slice(&value.to_be_bytes());
        }
    };

    // 布局：头（8）+ IFD0（2 + 12 + 4）+ Exif IFD（2 + 12×n + 4）+ RATIONAL 数据区。
    let ifd0_offset = 8usize;
    let exif_offset = ifd0_offset + 2 + 12 + 4;
    let exif_size = 2 + 12 * entries.len() + 4;
    let mut rational_offset = exif_offset + exif_size;
    let mut rationals = Vec::new();
    let mut out = Vec::new();
    if little_endian {
        out.extend_from_slice(b"II*\0");
    } else {
        out.extend_from_slice(b"MM\0*");
    }
    write_u32(&mut out, ifd0_offset as u32);
    write_u16(&mut out, 1);
    write_u16(&mut out, 0x8769); // ExifIFDPointer
    write_u16(&mut out, 4); // LONG
    write_u32(&mut out, 1);
    write_u32(&mut out, exif_offset as u32);
    write_u32(&mut out, 0); // 无下一个 IFD

    write_u16(&mut out, entries.len() as u16);
    for (tag, value) in entries {
        match value {
            TestTagValue::Short(value) => {
                write_u16(&mut out, *tag);
                write_u16(&mut out, SHORT);
                write_u32(&mut out, 1);
                write_u16(&mut out, *value);
                write_u16(&mut out, 0); // 内联值不足 4 字节的部分补零
            }
            TestTagValue::Rational(numerator, denominator) => {
                write_u16(&mut out, *tag);
                write_u16(&mut out, RATIONAL);
                write_u32(&mut out, 1);
                write_u32(&mut out, rational_offset as u32);
                rationals.push((*numerator, *denominator));
                rational_offset += 8;
            }
        }
    }
    write_u32(&mut out, 0); // 无下一个 IFD
    for (numerator, denominator) in rationals {
        write_u32(&mut out, numerator);
        write_u32(&mut out, denominator);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIDTH: u64 = 4000;
    const HEIGHT: u64 = 3000;

    fn diagonal() -> f64 {
        ((WIDTH * WIDTH + HEIGHT * HEIGHT) as f64).sqrt()
    }

    #[test]
    fn focal_length_35mm_uses_cipa_formula() {
        let tiff = build_test_tiff(&[(0xA405, TestTagValue::Short(50))], true);
        let focal = focal_length(&tiff, WIDTH, HEIGHT).unwrap();
        assert!((focal - 50.0 / 43.27 * diagonal()).abs() < 1e-9);
    }

    #[test]
    fn focal_length_35mm_takes_priority_over_physical() {
        let tiff = build_test_tiff(
            &[
                (0xA405, TestTagValue::Short(28)),
                (0x920A, TestTagValue::Rational(6, 1)),
                (0xA20E, TestTagValue::Rational(1000, 1)),
                (0xA210, TestTagValue::Short(2)),
            ],
            true,
        );
        let focal = focal_length(&tiff, WIDTH, HEIGHT).unwrap();
        assert!((focal - 28.0 / 43.27 * diagonal()).abs() < 1e-9);
    }

    #[test]
    fn physical_focal_length_converts_plane_resolution_units() {
        // 单位 × 每毫米像素：英寸 /25.4、厘米 /10、毫米 ×1、微米 ×1000。
        let expected = [
            (2u16, 35.0 * 1000.0 / 25.4),
            (3, 35.0 * 100.0),
            (4, 35.0 * 1000.0),
            (5, 35.0 * 1_000_000.0),
        ];
        for (unit, expected) in expected {
            let tiff = build_test_tiff(
                &[
                    (0x920A, TestTagValue::Rational(35, 1)),
                    (0xA20E, TestTagValue::Rational(1000, 1)),
                    (0xA210, TestTagValue::Short(unit)),
                ],
                true,
            );
            let focal = focal_length(&tiff, WIDTH, HEIGHT).unwrap();
            assert!((focal - expected).abs() < 1e-6, "单位 {unit}: {focal}");
        }
    }

    #[test]
    fn zero_or_missing_focal_length_yields_none() {
        // 空子 IFD。
        let tiff = build_test_tiff(&[], true);
        assert!(focal_length(&tiff, WIDTH, HEIGHT).is_none());

        // 35 mm 焦距为 0 → 不采用；物理焦距缺失 → None。
        let tiff = build_test_tiff(&[(0xA405, TestTagValue::Short(0))], true);
        assert!(focal_length(&tiff, WIDTH, HEIGHT).is_none());

        // 只有物理焦距、没有分辨率 → None（传感器库未内置）。
        let tiff = build_test_tiff(&[(0x920A, TestTagValue::Rational(35, 1))], true);
        assert!(focal_length(&tiff, WIDTH, HEIGHT).is_none());

        // 分辨率单位超出 2..=5 → None。
        let tiff = build_test_tiff(
            &[
                (0x920A, TestTagValue::Rational(35, 1)),
                (0xA20E, TestTagValue::Rational(1000, 1)),
                (0xA210, TestTagValue::Short(1)),
            ],
            true,
        );
        assert!(focal_length(&tiff, WIDTH, HEIGHT).is_none());

        // 分母为 0 → None。
        let tiff = build_test_tiff(
            &[
                (0x920A, TestTagValue::Rational(35, 0)),
                (0xA20E, TestTagValue::Rational(1000, 1)),
                (0xA210, TestTagValue::Short(2)),
            ],
            true,
        );
        assert!(focal_length(&tiff, WIDTH, HEIGHT).is_none());
    }

    #[test]
    fn big_endian_tiff_parses() {
        let tiff = build_test_tiff(&[(0xA405, TestTagValue::Short(35))], false);
        let focal = focal_length(&tiff, WIDTH, HEIGHT).unwrap();
        assert!((focal - 35.0 / 43.27 * diagonal()).abs() < 1e-9);
    }

    #[test]
    fn jpeg_app1_wrapper_is_scanned() {
        let tiff = build_test_tiff(&[(0xA405, TestTagValue::Short(35))], true);
        let mut jpeg = vec![0xFF, 0xD8];
        // 先放一个非 Exif 的 APP1（模拟 XMP）确保会被跳过。
        let xmp = b"http://ns.adobe.com/xap/1.0/\0<xml/>";
        jpeg.extend_from_slice(&[0xFF, 0xE1]);
        jpeg.extend_from_slice(&((xmp.len() + 2) as u16).to_be_bytes());
        jpeg.extend_from_slice(xmp);
        // 再放 Exif APP1。
        let mut payload = b"Exif\0\0".to_vec();
        payload.extend_from_slice(&tiff);
        jpeg.extend_from_slice(&[0xFF, 0xE1]);
        jpeg.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
        jpeg.extend_from_slice(&payload);
        // 结束标记。
        jpeg.extend_from_slice(&[0xFF, 0xD9]);

        let focal = focal_length(&jpeg, WIDTH, HEIGHT).unwrap();
        assert!((focal - 35.0 / 43.27 * diagonal()).abs() < 1e-9);
    }

    #[test]
    fn malformed_inputs_never_panic() {
        let tiff = build_test_tiff(
            &[
                (0xA405, TestTagValue::Short(50)),
                (0x920A, TestTagValue::Rational(35, 1)),
                (0xA20E, TestTagValue::Rational(1000, 1)),
                (0xA210, TestTagValue::Short(3)),
            ],
            true,
        );
        // 对每个前缀长度都不 panic（返回 Some 或 None 均可）。
        for length in 0..tiff.len() {
            let _ = focal_length(&tiff[..length], WIDTH, HEIGHT);
        }
        // 随机噪声不应 panic。
        let noise: Vec<u8> = (0..256u32)
            .map(|index| (index.wrapping_mul(2654435761) >> 13) as u8)
            .collect();
        let _ = focal_length(&noise, WIDTH, HEIGHT);
        assert!(focal_length(&[], WIDTH, HEIGHT).is_none());
    }
}
