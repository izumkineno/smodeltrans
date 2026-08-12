//! PP-OCRv5-owned region records.
//!
//! These records stay private to the OCR provider until the adapter converts
//! them once into the neutral backend contract.

use crate::backend::contracts::{CharacterRecord, RegionRecord};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PpOcrCharacterRecord {
    pub(crate) order: u32,
    pub(crate) quad_points: [[i32; 2]; 4],
    pub(crate) source_text: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PpOcrRegionRecord {
    pub(crate) order: u32,
    pub(crate) quad_points: [[i32; 2]; 4],
    pub(crate) source_text: String,
    pub(crate) confidence_milli: u16,
    pub(crate) characters: Vec<PpOcrCharacterRecord>,
}

impl PpOcrRegionRecord {
    pub(crate) fn new(
        order: u32,
        quad_points: [[i32; 2]; 4],
        source_text: impl Into<String>,
        confidence_milli: u16,
        characters: Vec<PpOcrCharacterRecord>,
    ) -> Self {
        Self {
            order,
            quad_points,
            source_text: source_text.into(),
            confidence_milli,
            characters,
        }
    }
    pub(crate) fn into_contract(self) -> RegionRecord {
        RegionRecord {
            order: self.order,
            quad_points: self.quad_points,
            source_text: self.source_text,
            confidence_milli: self.confidence_milli,
            translated_text: String::new(),
            characters: self
                .characters
                .into_iter()
                .map(|character| CharacterRecord {
                    order: character.order,
                    quad_points: character.quad_points,
                    source_text: character.source_text,
                })
                .collect(),
        }
    }
}
