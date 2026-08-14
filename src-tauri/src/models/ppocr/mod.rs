//! Native PP-OCR detector/recognizer provider (v5 server/mobile + v6 tiers).

mod adapter;
pub(crate) mod assets;
mod common;
mod geometry;
mod records;
mod v5;
mod v6;

pub(crate) use adapter::PpOcrProvider;
