//! Native PP-OCRv5 detector/recognizer provider.

mod adapter;
pub(crate) mod assets;
mod geometry;
mod model;
mod records;

pub(crate) use adapter::PpOcrV5Provider;
