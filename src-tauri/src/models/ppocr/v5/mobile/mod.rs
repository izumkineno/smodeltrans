//! PP-OCRv5 mobile-family adapter: PPLCNetV3 backbone, RSEFPN neck, DB head.

mod backbone;
mod graph;
mod neck;

pub(crate) use graph::{PpOcrMobileDetector, PpOcrMobileRecognizer};
