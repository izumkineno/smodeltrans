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
        let _span = tracing::debug_span!(target: "output", "render", target_language = %target_language, region_count = regions.len()).entered();
        tracing::info!(target: "output", target_language = %target_language, region_count = regions.len(), image_width = image.canvas().width(), image_height = image.canvas().height(), "render translation output started");
        tracing::debug!(target: "output", regions = ?regions.iter().map(|r| (r.order, r.quad_points)).collect::<Vec<_>>(), "ordered region input");
        cancellation.check()?;
        let base: RgbaImage = DynamicImage::ImageRgb8(image.canvas().clone()).into_rgba8();
        // 小图（对话框 502x203≈100k）2x 超采样，否则 1x；保证 CJK 在 10px->20px 光栅，显著抗糊
        let scale: u32 = if (base.width() as u64 * base.height() as u64) < 1_200_000 { 2 } else { 1 };
        let mut rendered: RgbaImage = if scale > 1 {
            image::imageops::resize(&base, base.width()*scale, base.height()*scale, image::imageops::FilterType::Lanczos3)
        } else { base };
        let mut translated_text = Vec::with_capacity(regions.len());
        for region in regions {
            cancellation.check()?;
            if region.order == 0 || region.translated_text.trim().is_empty() {
                return Err(BackendFailure::output("输出区域缺少有效顺序或译文"));
            }
            let q = if scale>1 { region.quad_points.map(|p| [p[0]*scale as i32, p[1]*scale as i32]) } else { region.quad_points };
            let rect = quad_rect(q, rendered.width(), rendered.height())
                .ok_or_else(|| BackendFailure::output("OCR 区域超出图像或几何退化"))?;
            if rect.width() < 8*scale || rect.height() < 8*scale {
                return Err(BackendFailure::output("OCR 区域没有足够的标注空间"));
            }
            // 2x 时画 2px 边框保持视觉一致
            for s in 0..scale { let r = Rect::at(rect.left()+s as i32, rect.top()+s as i32).of_size(rect.width().saturating_sub(s*2), rect.height().saturating_sub(s*2)); draw_hollow_rect_mut(&mut rendered, r, BORDER); }
            translated_text.push(clean_annotation(&region.translated_text));
        }
        if !regions.is_empty() {
            let font = resolve_font(self.font_path.as_deref(), &translated_text)?;
            for (region, text) in regions.iter().zip(translated_text.iter()) {
                cancellation.check()?;
                let q = if scale>1 { region.quad_points.map(|p| [p[0]*scale as i32, p[1]*scale as i32]) } else { region.quad_points };
                let rect = quad_rect(q, rendered.width(), rendered.height())
                    .ok_or_else(|| BackendFailure::output("OCR 区域几何无效"))?;
                let inset = Rect::at(rect.left() + 2*scale as i32, rect.top() + 2*scale as i32).of_size(
                    rect.width().saturating_sub(4*scale),
                    rect.height().saturating_sub(4*scale),
                );
                draw_filled_rect_mut(&mut rendered, inset, OVERLAY);
                // 0.72*inset 高度，CJK 最小 14px@1x => 28px@2x，保证 20px+ 光栅不糊；上限 48px
                let mut size = (inset.height() as f32 * 0.72).clamp(14.0*scale as f32, 48.0*scale as f32);
                // 宽度自适应：按平均 0.60*size 估算，超宽则等比缩小（仍 >=12*scale）
                let est_w = text.chars().count() as f32 * size * 0.60;
                if est_w > inset.width() as f32 - 4.0*scale as f32 && !text.is_empty() {
                    size = ((inset.width() as f32 - 4.0*scale as f32) / (text.chars().count() as f32 * 0.60)).clamp(12.0*scale as f32, size);
                }
                // 长句换行抑制溢出：若仍超宽，draw_text_mut 会截断，至少已缩至可读下限
                draw_text_mut(
                    &mut rendered,
                    WHITE,
                    inset.left() + 2*scale as i32,
                    inset.top() + 2*scale as i32,
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
        let _span = tracing::debug_span!(target: "output", "render_ocr", region_count = regions.len()).entered();
        tracing::info!(target: "output", region_count = regions.len(), image_width = image.canvas().width(), image_height = image.canvas().height(), "render ocr output started");
        tracing::debug!(target: "output", ordered = ?ordered_regions(&regions).iter().map(|r| r.order).collect::<Vec<_>>(), "region ordering for ocr");
        cancellation.check()?;
        let base: RgbaImage = DynamicImage::ImageRgb8(image.canvas().clone()).into_rgba8();
        let scale: u32 = if (base.width() as u64 * base.height() as u64) < 1_200_000 { 2 } else { 1 };
        let mut rendered: RgbaImage = if scale > 1 {
            image::imageops::resize(&base, base.width()*scale, base.height()*scale, image::imageops::FilterType::Lanczos3)
        } else { base };
        let ordered = ordered_regions(&regions);
        let mut recognized_text = Vec::with_capacity(ordered.len());
        for region in &ordered {
            cancellation.check()?;
            if region.order == 0 || region.source_text.trim().is_empty() {
                return Err(BackendFailure::output("输出区域缺少有效顺序或识别文本"));
            }
            let q = if scale>1 { region.quad_points.map(|p| [p[0]*scale as i32, p[1]*scale as i32]) } else { region.quad_points };
            let rect = quad_rect(q, rendered.width(), rendered.height())
                .ok_or_else(|| BackendFailure::output("OCR 区域超出图像或几何退化"))?;
            if rect.width() < 8*scale || rect.height() < 8*scale {
                return Err(BackendFailure::output("OCR 区域没有足够的标注空间"));
            }
            for s in 0..scale { let r = Rect::at(rect.left()+s as i32, rect.top()+s as i32).of_size(rect.width().saturating_sub(s*2), rect.height().saturating_sub(s*2)); draw_hollow_rect_mut(&mut rendered, r, BORDER); }
            recognized_text.push(clean_annotation(&region.source_text));
        }
        if !ordered.is_empty() {
            let font = resolve_font(self.font_path.as_deref(), &recognized_text)?;
            for (region, text) in ordered.iter().zip(recognized_text.iter()) {
                cancellation.check()?;
                let q = if scale>1 { region.quad_points.map(|p| [p[0]*scale as i32, p[1]*scale as i32]) } else { region.quad_points };
                let rect = quad_rect(q, rendered.width(), rendered.height())
                    .ok_or_else(|| BackendFailure::output("OCR 区域几何无效"))?;
                let inset = Rect::at(rect.left() + 2*scale as i32, rect.top() + 2*scale as i32).of_size(
                    rect.width().saturating_sub(4*scale),
                    rect.height().saturating_sub(4*scale),
                );
                draw_filled_rect_mut(&mut rendered, inset, OVERLAY);
                let mut size = (inset.height() as f32 * 0.72).clamp(14.0*scale as f32, 48.0*scale as f32);
                let est_w = text.chars().count() as f32 * size * 0.60;
                if est_w > inset.width() as f32 - 4.0*scale as f32 && !text.is_empty() {
                    size = ((inset.width() as f32 - 4.0*scale as f32) / (text.chars().count() as f32 * 0.60)).clamp(12.0*scale as f32, size);
                }
                draw_text_mut(
                    &mut rendered,
                    WHITE,
                    inset.left() + 2*scale as i32,
                    inset.top() + 2*scale as i32,
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
    tracing::trace!(target: "output", region_count = regions.len(), "ordered_regions called");
    let _span = tracing::trace_span!(target: "output", "ordered_regions", count = regions.len()).entered();
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
    tracing::debug!(target: "output", font_path = ?path, text_count = texts.len(), "resolve_font started");
    if let Some(path) = path {
        match fs::read(path) {
            Ok(bytes) => match FontVec::try_from_vec(bytes).map(FontArc::from) {
                Ok(font) => {
                    if !covers(&font, texts) {
                        tracing::warn!(target: "output", font_path = ?path, "explicit font does not cover all characters, still using it");
                    } else {
                        tracing::info!(target: "output", font_path = ?path, "explicit font resolved and covers text");
                    }
                    return Ok(font);
                }
                Err(_) => {
                    tracing::warn!(target: "output", font_path = ?path, "explicit font invalid, falling back to system font");
                }
            },
            Err(error) => {
                tracing::warn!(target: "output", font_path = ?path, error = %error, "cannot read explicit font, falling back to system font");
            }
        }
    }
    let mut database = fontdb::Database::new();
    database.load_system_fonts();
    // 优先高质量 CJK 字体，避免回退到点阵/低清晰字体导致糊；按 family/postscript 关键字打分
    let priority = |path: &PathBuf, family: &str, ps: &str| -> i32 {
        let p = format!("{} {} {}", path.display(), family, ps).to_lowercase();
        if p.contains("noto sans cjk") || p.contains("noto sans sc") || p.contains("source han sans") { 100 }
        else if p.contains("microsoft yahei") || p.contains("msyh") { 90 }
        else if p.contains("pingfang") { 85 }
        else if p.contains("hiragino") || p.contains("yu gothic") { 80 }
        else if p.contains("noto") { 70 }
        else if p.contains("simsun") || p.contains("simsun") { 10 }
        else if p.contains("unifont") || p.contains("wqy bitmap") { 0 }
        else { 50 }
    };
    let mut candidates = database
        .faces()
        .filter_map(|face| match &face.source {
            fontdb::Source::File(path) => {
                let family = face.families.first().map(|(n, _)| n.clone()).unwrap_or_default();
                Some((path.clone(), face.id, face.index, family, face.post_script_name.clone()))
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        let pl = priority(&left.0, &left.3, &left.4);
        let pr = priority(&right.0, &right.3, &right.4);
        pr.cmp(&pl).then(left.0.cmp(&right.0)).then(left.2.cmp(&right.2))
    });
    tracing::debug!(target: "output", candidate_count = candidates.len(), "system font candidates collected (priority sorted)");
    for (_, id, _, _, _) in candidates {
        let maybe_font = database.with_face_data(id, |data, face_index| {
            FontVec::try_from_vec_and_index(data.to_vec(), face_index)
                .ok()
                .map(FontArc::from)
        });
        if let Some(font) = maybe_font.flatten().filter(|font| covers(font, texts)) {
            tracing::info!(target: "output", font_id = ?id, "system font selected that covers text");
            return Ok(font);
        }
    }
    tracing::warn!(target: "output", text_count = texts.len(), "no system font covers all characters");
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
