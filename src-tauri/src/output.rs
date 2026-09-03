use crate::{
    backend::{
        contracts::{OcrOutput, OutputPort, RegionRecord, TranslationOutput},
        failure::BackendFailure,
        input::{DecodedImage, MAX_TEXT_BYTES},
    },
    model_support::CancellationToken,
};
use ab_glyph::{Font, FontArc, FontVec, PxScale};
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{DynamicImage, ExtendedColorType, ImageEncoder, Rgba, RgbaImage};
use imageproc::{
    drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut},
    rect::Rect,
};
use std::{
    collections::HashMap,
    fs,
    io::Cursor,
    path::{Path, PathBuf},
    sync::{LazyLock, Mutex},
};

const MAX_PNG_BYTES: usize = 8 * 1024 * 1024;
const MAX_DATA_URL_CHARS: usize = 11_200_000;
const BORDER: Rgba<u8> = Rgba([255, 0, 0, 255]);
const OVERLAY: Rgba<u8> = Rgba([0, 0, 0, 191]);
const WHITE: Rgba<u8> = Rgba([255, 255, 255, 255]);
// 浅底气泡（漫画白底对话框）用纸白底+墨字，还原气泡底色；深底沿用黑底白字。
const PAPER: Rgba<u8> = Rgba([255, 255, 255, 255]);
const INK: Rgba<u8> = Rgba([17, 17, 17, 255]);

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
        let (mut rendered, scale) = upsample_for_annotation(base);
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
            let font = self.resolve_font(&translated_text)?;
            for (region, text) in regions.iter().zip(translated_text.iter()) {
                cancellation.check()?;
                let q = if scale>1 { region.quad_points.map(|p| [p[0]*scale as i32, p[1]*scale as i32]) } else { region.quad_points };
                let rect = quad_rect(q, rendered.width(), rendered.height())
                    .ok_or_else(|| BackendFailure::output("OCR 区域几何无效"))?;
                // 气泡级排版：换行+自适应字号+居中+底色自适应（对齐主流），此前单行截断。
                draw_bubble_text(&mut rendered, rect, scale, &font, text);
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
        let png = encode_png_fast(&rendered)?;
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
        let (mut rendered, scale) = upsample_for_annotation(base);
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
            let font = self.resolve_font(&recognized_text)?;
            for (region, text) in ordered.iter().zip(recognized_text.iter()) {
                cancellation.check()?;
                let q = if scale>1 { region.quad_points.map(|p| [p[0]*scale as i32, p[1]*scale as i32]) } else { region.quad_points };
                let rect = quad_rect(q, rendered.width(), rendered.height())
                    .ok_or_else(|| BackendFailure::output("OCR 区域几何无效"))?;
                draw_bubble_text(&mut rendered, rect, scale, &font, text);
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
        let png = encode_png_fast(&rendered)?;
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

/// PNG 编码：Fast 压缩+Sub 过滤。解码像素与默认压缩逐字节一致，只是文件大 10-30%；
/// 默认压缩在 2x 放大后的十几 MB raw 上是秒级开销，是字体缓存之外的第二刀。
fn encode_png_fast(rendered: &RgbaImage) -> Result<Vec<u8>, BackendFailure> {
    let mut bytes = Cursor::new(Vec::new());
    {
        let mut encoder =
            PngEncoder::new_with_quality(&mut bytes, CompressionType::Fast, FilterType::Sub);
        encoder
            .write_image(
                rendered.as_raw(),
                rendered.width(),
                rendered.height(),
                ExtendedColorType::Rgba8,
            )
            .map_err(|error| BackendFailure::output(format!("PNG 编码失败：{error}")))?;
    }
    Ok(bytes.into_inner())
}

/// 2x 超采样（小图 CJK 光栅抗糊）。Triangle 而非 Lanczos3：文字是放大后才光栅的，
/// 上采样只影响背景照片，Triangle 快数倍且文字锐度零损失。
fn upsample_for_annotation(base: RgbaImage) -> (RgbaImage, u32) {
    let scale: u32 = if (base.width() as u64 * base.height() as u64) < 1_200_000 {
        2
    } else {
        1
    };
    let rendered = if scale > 1 {
        image::imageops::resize(
            &base,
            base.width() * scale,
            base.height() * scale,
            image::imageops::FilterType::Triangle,
        )
    } else {
        base
    };
    (rendered, scale)
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
    // 显式换行保留给排版按段换行；其余控制字符折为空格（此前 \n 也被压成空格，框内多行对话只能单行截断）。
    let mut out = String::with_capacity(value.len());
    let mut last_was_newline = true;
    for character in value.chars() {
        if character == '\n' || character == '\r' {
            if !last_was_newline {
                out.push('\n');
            }
            last_was_newline = true;
        } else if character.is_control() {
            if !out.is_empty() && !out.ends_with(' ') && !last_was_newline {
                out.push(' ');
            }
            last_was_newline = false;
        } else {
            out.push(character);
            last_was_newline = false;
        }
    }
    while out.ends_with('\n') || out.ends_with(' ') {
        out.pop();
    }
    out
}
/// CJK/全角按 1.0 字宽估算，其余按 0.55（旧 0.60 一刀切低估 CJK 致溢出），空格按 0.4。
fn char_width_ratio(character: char) -> f32 {
    match character as u32 {
        0x3040..=0x30ff | 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xac00..=0xd7af | 0xf900..=0xfaff
        | 0xff00..=0xffef => 1.0,
        _ if character.is_whitespace() => 0.4,
        _ => 0.55,
    }
}

fn est_line_width_px(text: &str, size: f32) -> f32 {
    text.chars().map(|c| char_width_ratio(c) * size).sum()
}

/// 按气泡框换行：显式换行保留；CJK 可任意断，拉丁优先空格断行；行首空格丢弃。
fn wrap_annotation(text: &str, max_width_px: f32, size: f32) -> Vec<String> {
    let mut lines = Vec::new();
    for paragraph in text.split('\n') {
        let paragraph = paragraph.trim_end();
        if paragraph.is_empty() {
            continue;
        }
        let mut current = String::new();
        let mut width = 0.0f32;
        for character in paragraph.chars() {
            if current.is_empty() && character.is_whitespace() {
                continue;
            }
            let char_w = char_width_ratio(character) * size;
            if width + char_w > max_width_px {
                if let Some(space_pos) = current.rfind(|c: char| c == ' ' || c == '\u{3000}') {
                    let cut = space_pos
                        + current[space_pos..]
                            .chars()
                            .next()
                            .map_or(1, |c| c.len_utf8());
                    let tail = current[cut..].to_owned();
                    current.truncate(space_pos);
                    if !current.trim().is_empty() {
                        lines.push(std::mem::take(&mut current));
                    } else {
                        current.clear();
                    }
                    current = tail;
                    width = est_line_width_px(&current, size);
                } else {
                    if !current.is_empty() {
                        lines.push(std::mem::take(&mut current));
                    }
                    width = 0.0;
                }
                if character.is_whitespace() {
                    continue;
                }
            }
            current.push(character);
            width += char_w;
        }
        if !current.trim().is_empty() {
            lines.push(current);
        }
    }
    lines
}

/// 气泡底色采样：沿框内圈取一圈像素算平均亮度；亮底（漫画白底气泡）用白底黑字，
/// 暗底沿用黑底白字。inpaint 前的轻量对齐，无模型开销。只收坐标，避免依赖 Rect 是否 Copy。
fn bubble_style(image: &RgbaImage, left: i32, top: i32, width: i32, height: i32) -> (Rgba<u8>, Rgba<u8>) {
    if width < 4 || height < 4 {
        return (OVERLAY, WHITE);
    }
    let (img_w, img_h) = (image.width() as i32, image.height() as i32);
    let (step_x, step_y) = ((width / 24).max(1), (height / 24).max(1));
    let mut sum = 0u64;
    let mut count = 0u64;
    let mut sample = |x: i32, y: i32| {
        if x < 0 || y < 0 || x >= img_w || y >= img_h {
            return;
        }
        let channels = image.get_pixel(x as u32, y as u32).0;
        sum += (77u64 * u64::from(channels[0])
            + 150u64 * u64::from(channels[1])
            + 29u64 * u64::from(channels[2]))
            >> 8;
        count += 1;
    };
    for x in (left..left + width).step_by(step_x as usize) {
        sample(x, top);
        sample(x, top + height - 1);
    }
    for y in (top..top + height).step_by(step_y as usize) {
        sample(left, y);
        sample(left + width - 1, y);
    }
    if count == 0 || sum / count <= 150 {
        (OVERLAY, WHITE)
    } else {
        (PAPER, INK)
    }
}

/// 气泡级排版绘制：换行+自适应字号+块居中；字号初值沿用旧 0.72*框高规则，
/// 按高度收敛到能放下所有行为止（下限沿用 12px），仍放不下截尾保留可见行。
/// 与实时浮层 wrapText/drawRegionText 同思想，静态贴图此前单行截断。
fn draw_bubble_text(
    rendered: &mut RgbaImage,
    rect: Rect,
    scale: u32,
    font: &FontArc,
    text: &str,
) {
    let inset = Rect::at(
        rect.left() + 2 * scale as i32,
        rect.top() + 2 * scale as i32,
    )
    .of_size(
        rect.width().saturating_sub(4 * scale),
        rect.height().saturating_sub(4 * scale),
    );
    let (inset_left, inset_top, inset_width, inset_height) = (
        inset.left(),
        inset.top(),
        inset.width() as i32,
        inset.height() as i32,
    );
    let (fill, ink) = bubble_style(rendered, inset_left, inset_top, inset_width, inset_height);
    draw_filled_rect_mut(rendered, inset, fill);
    let pad = 2.0 * scale as f32;
    let max_width = (inset_width as f32 - pad * 2.0).max(8.0 * scale as f32);
    let max_height = (inset_height as f32 - pad * 2.0).max(8.0 * scale as f32);
    // 0.72*inset 高度，CJK 最小 14px@1x => 28px@2x，保证 20px+ 光栅不糊；上限 48px（沿用旧规则）。
    let min_size = 12.0 * scale as f32;
    let mut size = (inset_height as f32 * 0.72).clamp(14.0 * scale as f32, 48.0 * scale as f32);
    let mut lines = wrap_annotation(text, max_width, size);
    if lines.is_empty() {
        return;
    }
    while lines.len() as f32 * size * 1.18 > max_height && size > min_size {
        size = (size - 1.0 * scale as f32).max(min_size);
        lines = wrap_annotation(text, max_width, size);
        if lines.is_empty() {
            return;
        }
    }
    let max_lines = (max_height / (size * 1.18)).floor().max(1.0) as usize;
    if lines.len() > max_lines {
        lines.truncate(max_lines);
    }
    let total_height = lines.len() as f32 * size * 1.18;
    let mut y = inset_top as f32 + ((inset_height as f32 - total_height) / 2.0).max(0.0);
    for line in &lines {
        let line_width = est_line_width_px(line, size);
        let x = inset_left as f32 + ((inset_width as f32 - line_width) / 2.0).max(0.0);
        draw_text_mut(
            rendered,
            ink,
            x as i32,
            y as i32,
            PxScale::from(size),
            font,
            line,
        );
        y += size * 1.18;
    }
}


static FONT_DB: LazyLock<fontdb::Database> = LazyLock::new(|| {
    tracing::info!(target: "output", "system font database cold load (once per process)");
    let mut database = fontdb::Database::new();
    database.load_system_fonts();
    database
});
static EXPLICIT_FONTS: LazyLock<Mutex<HashMap<PathBuf, FontArc>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static FONT_CACHE: LazyLock<Mutex<HashMap<String, FontArc>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// 显式字体按路径常驻：同一文件只读盘解析一次，文件变更需重启进程。
fn cached_explicit_font(path: &Path) -> Result<Option<FontArc>, BackendFailure> {
    let store = LazyLock::force(&EXPLICIT_FONTS);
    if let Some(font) = store
        .lock()
        .map_err(|_| BackendFailure::output("字体缓存锁中毒"))?
        .get(path)
        .cloned()
    {
        tracing::debug!(target: "output", font_path = ?path, "explicit font cache hit");
        return Ok(Some(font));
    }
    let font = match fs::read(path) {
        Ok(bytes) => match FontVec::try_from_vec(bytes).map(FontArc::from) {
            Ok(font) => font,
            Err(_) => {
                tracing::warn!(target: "output", font_path = ?path, "explicit font invalid, falling back to system font");
                return Ok(None);
            }
        },
        Err(error) => {
            tracing::warn!(target: "output", font_path = ?path, error = %error, "cannot read explicit font, falling back to system font");
            return Ok(None);
        }
    };
    store
        .lock()
        .map_err(|_| BackendFailure::output("字体缓存锁中毒"))?
        .insert(path.to_path_buf(), font.clone());
    Ok(Some(font))
}

/// 相同字符集恒选同一字体（候选排序确定），缓存键即去重排序后的字符集；
/// 字体安装变化需重启进程才生效。上限轮转防非常用字符集撑内存。
const MAX_CACHED_FONTS: usize = 8;

fn font_cache_key(texts: &[String]) -> String {
    let mut chars: Vec<char> = texts
        .iter()
        .flat_map(|text| text.chars())
        .filter(|character| !character.is_whitespace())
        .collect();
    chars.sort_unstable();
    chars.dedup();
    chars.into_iter().collect()
}

impl ImageOutput {
    fn resolve_font(&mut self, texts: &[String]) -> Result<FontArc, BackendFailure> {
        // ImageOutput 随引擎按请求重建，实例字段活不过单次请求；跨请求命中靠进程级静态。
        let mut key = self
            .font_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default();
        key.push('|');
        key.push_str(&font_cache_key(texts));
        let cache = LazyLock::force(&FONT_CACHE);
        {
            let guard = cache
                .lock()
                .map_err(|_| BackendFailure::output("字体缓存锁中毒"))?;
            if let Some(font) = guard.get(&key) {
                tracing::debug!(target: "output", text_count = texts.len(), "font cache hit");
                return Ok(font.clone());
            }
        }
        let font = self.select_font_uncached(texts)?;
        {
            let mut guard = cache
                .lock()
                .map_err(|_| BackendFailure::output("字体缓存锁中毒"))?;
            if guard.len() >= MAX_CACHED_FONTS {
                guard.clear();
            }
            guard.insert(key, font.clone());
        }
        Ok(font)
    }

    fn select_font_uncached(&mut self, texts: &[String]) -> Result<FontArc, BackendFailure> {
        if let Some(path) = self.font_path.clone() {
            if let Some(font) = cached_explicit_font(&path)? {
                if !covers(&font, texts) {
                    tracing::warn!(target: "output", font_path = ?path, "explicit font does not cover all characters, still using it");
                } else {
                    tracing::info!(target: "output", font_path = ?path, "explicit font resolved and covers text");
                }
                return Ok(font);
            }
        }
        let database: &fontdb::Database = &FONT_DB;
    // 优先高质量 CJK 字体，避免回退到点阵/低清晰字体导致糊；按 family/postscript 关键字打分
    let priority = |path: &PathBuf, family: &str, ps: &str| -> i32 {
        let p = format!("{} {} {}", path.display(), family, ps).to_lowercase();
        if p.contains("noto sans cjk") || p.contains("noto sans sc") || p.contains("source han sans") { 100 }
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
}

fn covers(font: &FontArc, texts: &[String]) -> bool {
    // 换行/空格不参与覆盖检查：clean_annotation 现保留 \n 供排版分段，
    // wrap 后行内无换行；缺字形空白回退渲染，不应整图报错。
    texts
        .iter()
        .flat_map(|text| text.chars())
        .all(|character| character.is_whitespace() || font.glyph_id(character).0 != 0)
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
