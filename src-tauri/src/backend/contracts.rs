use crate::{
    backend::{failure::BackendFailure, input::DecodedImage},
    model_support::CancellationToken,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RegionRecord {
    pub(crate) order: u32,
    pub(crate) quad_points: [[i32; 2]; 4],
    pub(crate) source_text: String,
    pub(crate) translated_text: String,
}

impl RegionRecord {
    pub(crate) fn untranslated(
        order: u32,
        quad_points: [[i32; 2]; 4],
        source_text: impl Into<String>,
    ) -> Self {
        Self {
            order,
            quad_points,
            source_text: source_text.into(),
            translated_text: String::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OcrDocument {
    pub(crate) regions: Vec<RegionRecord>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TranslationRegion {
    pub(crate) order: u32,
    pub(crate) source_text: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TranslatedRegion {
    pub(crate) order: u32,
    pub(crate) translated_text: String,
}

#[derive(Debug)]
pub(crate) struct TranslationOutput {
    pub(crate) annotated_png: Vec<u8>,
    pub(crate) markdown: String,
    pub(crate) text: String,
    pub(crate) provider_label: String,
    pub(crate) is_translated: bool,
}

pub(crate) trait OcrPort: Send {
    fn recognize(
        &mut self,
        image: &DecodedImage,
        cancellation: &CancellationToken,
    ) -> Result<OcrDocument, BackendFailure>;
}

pub(crate) trait HyPort: Send {
    fn translate(
        &mut self,
        regions: &[TranslationRegion],
        target_language: &str,
        cancellation: &CancellationToken,
    ) -> Result<Vec<TranslatedRegion>, BackendFailure>;

    fn loaded(&self) -> bool;
}

pub(crate) trait OutputPort: Send {
    fn render(
        &mut self,
        image: &DecodedImage,
        regions: &[RegionRecord],
        target_language: &str,
        cancellation: &CancellationToken,
    ) -> Result<TranslationOutput, BackendFailure>;
}
