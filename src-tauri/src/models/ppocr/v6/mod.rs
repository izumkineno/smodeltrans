//! Native PP-OCRv6 detector/recognizer graphs (tiny, small, medium tiers).

mod backbone;
mod graph;
mod head;
mod neck;

pub(crate) use graph::{PpOcrV6Detector, PpOcrV6Recognizer};
