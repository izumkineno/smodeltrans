use crate::{
    backend::{
        contracts::{OcrOutput, OutputPort, RegionRecord, TranslationOutput},
        failure::BackendFailure,
        input::{DecodedImage, MAX_TEXT_BYTES},
    },
    model_support::CancellationToken,
};
use ab_glyph::{Font, FontArc, FontVec, PxScale};
use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
use imageproc::{
    drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut},
    rect::Rect,
};
use std::{
    fs,
    io::Cursor,
    path::{Path, PathBuf},
};

const MAX_PNG_BYTES: usize = 8 * 1024 * 1024;
const MAX_DATA_URL_CHARS: usize = 11_200_000;
const BORDER: Rgba<u8> = Rgba([255, 0, 0, 255]);
const OVERLAY: Rgba<u8> = Rgba([0, 0, 0, 191]);
const WHITE: Rgba<u8> = Rgba([255, 255, 255, 255]);

pub(crate) struct ImageOutput {
    font_path: Option<PathBuf>,
}

impl ImageOutput {
    pub(crate) fn new(font_path: Option<PathBuf>) -> Self {
        Self { font_path }
    }
}

impl OutputPort for ImageOutput {
    fn render(
        &mut self,
        image: &DecodedImage,
        regions: &[RegionRecord],
        target_language: &str,
        cancellation: &CancellationToken,
    ) -> Result<TranslationOutput, BackendFailure> {
        cancellation.check()?;
        let mut rendered: RgbaImage = DynamicImage::ImageRgb8(image.canvas().clone()).into_rgba8();
        let mut translated_text = Vec::with_capacity(regions.len());
        for region in regions {
            cancellation.check()?;
            if region.order == 0 || region.translated_text.trim().is_empty() {
                return Err(BackendFailure::output("输出区域缺少有效顺序或译文"));
            }
            let rect = quad_rect(region.quad_points, rendered.width(), rendered.height())
                .ok_or_else(|| BackendFailure::output("OCR 区域超出图像或几何退化"))?;
            if rect.width() < 8 || rect.height() < 8 {
                return Err(BackendFailure::output("OCR 区域没有足够的标注空间"));
            }
            draw_hollow_rect_mut(&mut rendered, rect, BORDER);
            translated_text.push(clean_annotation(&region.translated_text));
        }
        if !regions.is_empty() {
            let font = resolve_font(self.font_path.as_deref(), &translated_text)?;
            for (region, text) in regions.iter().zip(translated_text.iter()) {
                cancellation.check()?;
                let rect = quad_rect(region.quad_points, rendered.width(), rendered.height())
                    .ok_or_else(|| BackendFailure::output("OCR 区域几何无效"))?;
                let inset = Rect::at(rect.left() + 2, rect.top() + 2).of_size(
                    rect.width().saturating_sub(4),
                    rect.height().saturating_sub(4),
                );
                draw_filled_rect_mut(&mut rendered, inset, OVERLAY);
                let size = (inset.height() as f32 * 0.72).clamp(8.0, 32.0);
                draw_text_mut(
                    &mut rendered,
                    WHITE,
                    inset.left() + 2,
                    inset.top() + 2,
                    PxScale::from(size),
                    &font,
                    text,
                );
            }
        }
        cancellation.check()?;
        let markdown = write_markdown(image.file_name(), target_language, regions)?;
        if markdown.len() > MAX_TEXT_BYTES {
            return Err(BackendFailure::output("Markdown 输出超过 8 MiB 上限"));
        }
        let text = regions
            .iter()
            .map(|region| region.translated_text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        if text.len() > MAX_TEXT_BYTES {
            return Err(BackendFailure::output("文本输出超过 8 MiB 上限"));
        }
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(rendered)
            .write_to(&mut bytes, ImageFormat::Png)
            .map_err(|error| BackendFailure::output(format!("PNG 编码失败：{error}")))?;
        let png = bytes.into_inner();
        if png.len() > MAX_PNG_BYTES {
            return Err(BackendFailure::output("PNG 输出超过 8 MiB 上限"));
        }
        let data_url_len = 22usize
            .checked_add((png.len() + 2) / 3 * 4)
            .ok_or_else(|| BackendFailure::output("PNG data URL 长度溢出"))?;
        if data_url_len > MAX_DATA_URL_CHARS {
            return Err(BackendFailure::output("PNG data URL 超过长度上限"));
        }
        Ok(TranslationOutput {
            annotated_png: png,
            markdown,
            text,
            provider_label: "PP-OCRv5 + Hy-MT2".to_owned(),
            is_translated: !regions.is_empty(),
        })
    }

    fn render_ocr(
        &mut self,
        image: &DecodedImage,
        regions: Vec<RegionRecord>,
        cancellation: &CancellationToken,
    ) -> Result<OcrOutput, BackendFailure> {
        cancellation.check()?;
        let mut rendered: RgbaImage = DynamicImage::ImageRgb8(image.canvas().clone()).into_rgba8();
        let ordered = ordered_regions(&regions);
        let mut recognized_text = Vec::with_capacity(ordered.len());
        for region in &ordered {
            cancellation.check()?;
            if region.order == 0 || region.source_text.trim().is_empty() {
                return Err(BackendFailure::output("输出区域缺少有效顺序或识别文本"));
            }
            let rect = quad_rect(region.quad_points, rendered.width(), rendered.height())
                .ok_or_else(|| BackendFailure::output("OCR 区域超出图像或几何退化"))?;
            if rect.width() < 8 || rect.height() < 8 {
                return Err(BackendFailure::output("OCR 区域没有足够的标注空间"));
            }
            draw_hollow_rect_mut(&mut rendered, rect, BORDER);
            recognized_text.push(clean_annotation(&region.source_text));
        }
        if !ordered.is_empty() {
            let font = resolve_font(self.font_path.as_deref(), &recognized_text)?;
            for (region, text) in ordered.iter().zip(recognized_text.iter()) {
                cancellation.check()?;
                let rect = quad_rect(region.quad_points, rendered.width(), rendered.height())
                    .ok_or_else(|| BackendFailure::output("OCR 区域几何无效"))?;
                let inset = Rect::at(rect.left() + 2, rect.top() + 2).of_size(
                    rect.width().saturating_sub(4),
                    rect.height().saturating_sub(4),
                );
                draw_filled_rect_mut(&mut rendered, inset, OVERLAY);
                let size = (inset.height() as f32 * 0.72).clamp(8.0, 32.0);
                draw_text_mut(
                    &mut rendered,
                    WHITE,
                    inset.left() + 2,
                    inset.top() + 2,
                    PxScale::from(size),
                    &font,
                    text,
                );
            }
        }
        cancellation.check()?;
        let markdown = write_ocr_markdown(image.file_name(), &regions)?;
        if markdown.len() > MAX_TEXT_BYTES {
            return Err(BackendFailure::output("Markdown 输出超过 8 MiB 上限"));
        }
        let text = ordered
            .iter()
            .map(|region| region.source_text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        if text.len() > MAX_TEXT_BYTES {
            return Err(BackendFailure::output("文本输出超过 8 MiB 上限"));
        }
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(rendered)
            .write_to(&mut bytes, ImageFormat::Png)
            .map_err(|error| BackendFailure::output(format!("PNG 编码失败：{error}")))?;
        let png = bytes.into_inner();
        if png.len() > MAX_PNG_BYTES {
            return Err(BackendFailure::output("PNG 输出超过 8 MiB 上限"));
        }
        let data_url_len = 22usize
            .checked_add((png.len() + 2) / 3 * 4)
            .ok_or_else(|| BackendFailure::output("PNG data URL 长度溢出"))?;
        if data_url_len > MAX_DATA_URL_CHARS {
            return Err(BackendFailure::output("PNG data URL 超过长度上限"));
        }
        Ok(OcrOutput {
            annotated_png: png,
            markdown,
            text,
            provider_label: "PP-OCRv5 / Candle".to_owned(),
            regions,
        })
    }
}

fn ordered_regions(regions: &[RegionRecord]) -> Vec<&RegionRecord> {
    let mut ordered = regions.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|region| region.order);
    ordered
}

fn quad_rect(quad: [[i32; 2]; 4], width: u32, height: u32) -> Option<Rect> {
    if width == 0 || height == 0 {
        return None;
    }
    let left = quad
        .iter()
        .map(|point| point[0])
        .min()?
        .clamp(0, width as i32 - 1);
    let top = quad
        .iter()
        .map(|point| point[1])
        .min()?
        .clamp(0, height as i32 - 1);
    let right = quad
        .iter()
        .map(|point| point[0])
        .max()?
        .clamp(0, width as i32 - 1);
    let bottom = quad
        .iter()
        .map(|point| point[1])
        .max()?
        .clamp(0, height as i32 - 1);
    (left < right && top < bottom)
        .then(|| Rect::at(left, top).of_size((right - left + 1) as u32, (bottom - top + 1) as u32))
}

fn clean_annotation(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect()
}

fn resolve_font(path: Option<&Path>, texts: &[String]) -> Result<FontArc, BackendFailure> {
    if let Some(path) = path {
        let bytes = fs::read(path)
            .map_err(|error| BackendFailure::output(format!("无法读取字体：{error}")))?;
        let font = FontVec::try_from_vec(bytes)
            .map(FontArc::from)
            .map_err(|_| BackendFailure::output("字体文件无效"))?;
        if !covers(&font, texts) {
            return Err(BackendFailure::output("显式字体不覆盖全部译文字符"));
        }
        return Ok(font);
    }
    let mut database = fontdb::Database::new();
    database.load_system_fonts();
    let mut candidates = database
        .faces()
        .filter_map(|face| match &face.source {
            fontdb::Source::File(path) => Some((path.clone(), face.id, face.index)),
            _ => None,
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.0.cmp(&right.0).then(left.2.cmp(&right.2)));
    for (_, id, _) in candidates {
        let maybe_font = database.with_face_data(id, |data, face_index| {
            FontVec::try_from_vec_and_index(data.to_vec(), face_index)
                .ok()
                .map(FontArc::from)
        });
        if let Some(font) = maybe_font.flatten().filter(|font| covers(font, texts)) {
            return Ok(font);
        }
    }
    Err(BackendFailure::output("没有系统字体覆盖全部译文字符"))
}

fn covers(font: &FontArc, texts: &[String]) -> bool {
    texts
        .iter()
        .flat_map(|text| text.chars())
        .all(|character| font.glyph_id(character).0 != 0)
}

fn escape_scalar(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 8);
    for character in value.chars() {
        match character {
            '\r' | '\n' => escaped.push_str("\\n"),
            '\\' => escaped.push_str("\\\\"),
            character if character.is_control() => {
                use std::fmt::Write;
                let _ = write!(escaped, "\\u{:04x}", character as u32);
            }
            character if "`*_[]-#>!|{}()".contains(character) => {
                escaped.push('\\');
                escaped.push(character);
            }
            character => escaped.push(character),
        }
    }
    serde_json::to_string(&escaped).unwrap_or_else(|_| "\"\"".to_owned())
}

fn write_markdown(
    file_name: &str,
    target_language: &str,
    regions: &[RegionRecord],
) -> Result<String, BackendFailure> {
    let mut markdown = String::new();
    markdown.push_str("---\nsource_image: ");
    markdown.push_str(&escape_scalar(file_name));
    markdown.push_str("\ntarget_language: ");
    markdown.push_str(&escape_scalar(target_language));
    markdown.push_str(&format!(
        "\nregion_count: {}\n---\n\n# OCR Translation\n",
        regions.len()
    ));
    for region in regions {
        markdown.push_str(&format!(
            "\n## Region {}\n\n- order: {}\n- quad_points: [",
            region.order, region.order
        ));
        for (index, point) in region.quad_points.iter().enumerate() {
            if index > 0 {
                markdown.push_str(", ");
            }
            markdown.push_str(&format!("[{}, {}]", point[0], point[1]));
        }
        markdown.push_str("]\n- source_text: ");
        markdown.push_str(&escape_scalar(&region.source_text));
        markdown.push_str("\n- translated_text: ");
        markdown.push_str(&escape_scalar(&region.translated_text));
        markdown.push('\n');
    }
    Ok(markdown)
}

fn write_ocr_markdown(file_name: &str, regions: &[RegionRecord]) -> Result<String, BackendFailure> {
    let ordered = ordered_regions(regions);
    let mut markdown = String::new();
    markdown.push_str("---\nsource_image: ");
    markdown.push_str(&escape_scalar(file_name));
    markdown.push_str(&format!(
        "\nregion_count: {}\n---\n\n# OCR\n",
        regions.len()
    ));
    for region in ordered {
        markdown.push_str(&format!(
            "\n## Region {}\n\n- order: {}\n- quad_points: [",
            region.order, region.order
        ));
        for (index, point) in region.quad_points.iter().enumerate() {
            if index > 0 {
                markdown.push_str(", ");
            }
            markdown.push_str(&format!("[{}, {}]", point[0], point[1]));
        }
        markdown.push_str("]\n- recognized_text: ");
        markdown.push_str(&escape_scalar(&region.source_text));
        markdown.push('\n');
    }
    Ok(markdown)
}

pub(crate) fn max_data_url_chars() -> usize {
    MAX_DATA_URL_CHARS
}

#[cfg(test)]
mod tests {
    use super::write_ocr_markdown;
    use crate::backend::contracts::RegionRecord;

    #[test]
    fn ocr_markdown_contains_recognized_text_in_reading_order() {
        let regions = vec![
            RegionRecord::untranslated(2, [[10, 10]; 4], "second"),
            RegionRecord::untranslated(1, [[0, 0]; 4], "first"),
        ];
        let markdown = write_ocr_markdown("screen.png", &regions).expect("OCR markdown");

        assert!(markdown.contains("# OCR"));
        assert!(markdown.contains("- recognized_text: \"first\""));
        assert!(markdown.contains("- recognized_text: \"second\""));
        assert!(
            markdown
                .find("recognized_text: \"first\"")
                .expect("first text")
                < markdown
                    .find("recognized_text: \"second\"")
                    .expect("second text")
        );
    }
}
