//! PP-OCRv5 server-family adapter: HGNetV2 backbone, DB++ head, SVTR head.

mod backbone;
mod graph;
mod head;
mod neck;

pub(crate) use graph::{PpOcrServerDetector, PpOcrServerRecognizer};
pub(crate) use neck::DetNeck;
