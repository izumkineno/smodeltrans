use super::failure::BackendFailure;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use image::{ImageReader, Limits, RgbImage};
use std::{io::Cursor, path::Path, sync::Arc};

pub(crate) const MAX_BASE64_CHARS: usize = 14_000_000;
pub(crate) const MAX_ENCODED_BYTES: usize = 10 * 1024 * 1024;
pub(crate) const MAX_IMAGE_SIDE: u32 = 8192;
pub(crate) const MAX_IMAGE_PIXELS: u64 = 33_554_432;
pub(crate) const MAX_CANVAS_BYTES: u64 = 128 * 1024 * 1024;
pub(crate) const MAX_REGIONS: usize = 256;
pub(crate) const MAX_TEXT_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Debug)]
pub(crate) struct DecodedImage {
    canvas: Arc<RgbImage>,
    encoded_bytes: Arc<[u8]>,
    file_name: String,
    target_language: String,
}

impl DecodedImage {
    pub(crate) fn canvas(&self) -> &RgbImage {
        self.canvas.as_ref()
    }

    pub(crate) fn canvas_identity(&self) -> usize {
        Arc::as_ptr(&self.canvas) as usize
    }

    pub(crate) fn encoded_bytes(&self) -> &[u8] {
        &self.encoded_bytes
    }

    pub(crate) fn file_name(&self) -> &str {
        &self.file_name
    }

    pub(crate) fn target_language(&self) -> &str {
        &self.target_language
    }
}

pub(crate) fn validate_target_language(value: &str) -> Result<String, BackendFailure> {
    let value = value.trim();
    let scalar_count = value.chars().count();
    if !(1..=64).contains(&scalar_count) {
        return Err(BackendFailure::arguments(
            "targetLanguage must contain 1..=64 Unicode scalar values",
        ));
    }
    Ok(value.to_owned())
}

pub(crate) fn validate_text(value: &str) -> Result<String, BackendFailure> {
    if value.len() > MAX_TEXT_BYTES {
        return Err(BackendFailure::arguments(
            "text exceeds the 8 MiB transport limit",
        ));
    }
    let value = value.trim();
    if value.is_empty() {
        return Err(BackendFailure::arguments("text must not be empty"));
    }
    Ok(value.to_owned())
}

pub(crate) fn decode_image(
    image_base64: &str,
    file_name: String,
    target_language: String,
) -> Result<DecodedImage, BackendFailure> {
    validate_file_name(&file_name)?;
    let target_language = validate_target_language(&target_language)?;
    decode_encoded_image(image_base64, file_name, target_language)
}

pub(crate) fn decode_ocr_image(
    image_base64: &str,
    file_name: String,
) -> Result<DecodedImage, BackendFailure> {
    validate_file_name(&file_name)?;
    decode_encoded_image(image_base64, file_name, String::new())
}

fn decode_encoded_image(
    image_base64: &str,
    file_name: String,
    target_language: String,
) -> Result<DecodedImage, BackendFailure> {
    if image_base64.is_empty() || image_base64.len() > MAX_BASE64_CHARS || !image_base64.is_ascii()
    {
        return Err(BackendFailure::arguments(
            "imageBase64 must be non-empty ASCII within the transport limit",
        ));
    }
    let encoded_bytes = BASE64
        .decode(image_base64)
        .map_err(|_| BackendFailure::arguments("imageBase64 is not valid Base64"))?;
    if encoded_bytes.is_empty() || encoded_bytes.len() > MAX_ENCODED_BYTES {
        return Err(BackendFailure::arguments(
            "encoded image bytes exceed the 10 MiB limit",
        ));
    }

    let mut reader = ImageReader::new(Cursor::new(&encoded_bytes))
        .with_guessed_format()
        .map_err(|_| BackendFailure::arguments("image format could not be identified"))?;
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_SIDE);
    limits.max_image_height = Some(MAX_IMAGE_SIDE);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let image = reader
        .decode()
        .map_err(|_| BackendFailure::arguments("image bytes are not a supported image"))?;
    let pixels = u64::from(image.width())
        .checked_mul(u64::from(image.height()))
        .ok_or_else(|| BackendFailure::arguments("image dimensions overflow"))?;
    if image.width() == 0
        || image.height() == 0
        || image.width() > MAX_IMAGE_SIDE
        || image.height() > MAX_IMAGE_SIDE
        || pixels > MAX_IMAGE_PIXELS
    {
        return Err(BackendFailure::arguments(
            "image dimensions exceed the supported bounds",
        ));
    }
    let canvas = image.to_rgb8();
    let canvas_bytes = pixels
        .checked_mul(3)
        .ok_or_else(|| BackendFailure::arguments("decoded canvas size overflow"))?;
    if canvas_bytes > MAX_CANVAS_BYTES {
        return Err(BackendFailure::arguments(
            "decoded canvas exceeds the memory bound",
        ));
    }
    Ok(DecodedImage {
        canvas: Arc::new(canvas),
        encoded_bytes: Arc::from(encoded_bytes.into_boxed_slice()),
        file_name,
        target_language,
    })
}

fn validate_file_name(file_name: &str) -> Result<(), BackendFailure> {
    if file_name.is_empty()
        || file_name.len() > 255
        || !file_name.is_char_boundary(file_name.len())
        || file_name.chars().any(|character| character.is_control())
        || file_name.contains(['/', '\\', ':'])
    {
        return Err(BackendFailure::arguments(
            "fileName must be a safe UTF-8 basename of at most 255 bytes",
        ));
    }
    let path = Path::new(file_name);
    if path.file_name().and_then(|name| name.to_str()) != Some(file_name)
        || matches!(file_name, "." | "..")
        || file_name.ends_with(['.', ' '])
    {
        return Err(BackendFailure::arguments(
            "fileName is not a valid basename",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{MAX_TEXT_BYTES, decode_ocr_image, validate_target_language, validate_text};

    #[test]
    fn target_language_validation_trims_and_bounds_unicode_scalars() {
        assert_eq!(
            validate_target_language("  English  ").expect("valid target"),
            "English"
        );
        assert!(validate_target_language("   ").is_err());
        assert!(validate_target_language(&"中".repeat(65)).is_err());
    }

    #[test]
    fn text_validation_trims_rejects_empty_and_enforces_byte_bound() {
        assert_eq!(validate_text("  hello\n").expect("valid text"), "hello");
        assert!(validate_text(" \t\r\n").is_err());
        assert!(validate_text(&"x".repeat(MAX_TEXT_BYTES + 1)).is_err());
    }

    #[test]
    fn ocr_decoder_does_not_require_a_target_language() {
        let image = decode_ocr_image(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
            "sample.png".to_owned(),
        )
        .expect("valid PNG");
        assert_eq!(image.target_language(), "");
        assert_eq!(image.file_name(), "sample.png");
    }
}
