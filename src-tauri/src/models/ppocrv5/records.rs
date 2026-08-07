//! PP-OCRv5-owned region records.
//!
//! These records stay private to the OCR provider until the adapter converts
//! them once into the neutral backend contract.

use crate::backend::contracts::RegionRecord;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PpOcrRegionRecord {
    pub(crate) order: u32,
    pub(crate) quad_points: [[i32; 2]; 4],
    pub(crate) source_text: String,
}

impl PpOcrRegionRecord {
    pub(crate) fn new(
        order: u32,
        quad_points: [[i32; 2]; 4],
        source_text: impl Into<String>,
    ) -> Self {
        Self {
            order,
            quad_points,
            source_text: source_text.into(),
        }
    }

    pub(crate) fn into_contract(self) -> RegionRecord {
        RegionRecord::untranslated(self.order, self.quad_points, self.source_text)
    }
}
