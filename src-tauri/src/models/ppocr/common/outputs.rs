//! Shape metadata produced by native detector and recognizer graphs.

use candle_core::Tensor;
/// Shape metadata accompanying a native detector probability map.
#[derive(Clone, Debug)]
pub struct DetectorOutput {
    /// PP-OCRv5's final sigmoid probability map, shaped `[batch, 1, height, width]`.
    pub probabilities: Tensor,
    /// Materialized output dimensions for callers that do not want to inspect a Tensor.
    pub shape: Vec<usize>,
}

impl DetectorOutput {
    pub(crate) fn new(probabilities: Tensor) -> Self {
        Self {
            shape: probabilities.dims().to_vec(),
            probabilities,
        }
    }

    /// Borrow the raw probability tensor without doing post-processing.
    pub fn tensor(&self) -> &Tensor {
        &self.probabilities
    }
}

/// Shape metadata accompanying native recognizer per-time-step logits.
#[derive(Clone, Debug)]
pub struct RecognizerOutput {
    /// PP-OCRv5's per-time-step CTC logits tensor, shaped `[batch, time, vocab]`.
    pub logits: Tensor,
    /// Materialized output dimensions for callers that do not want to inspect a Tensor.
    pub shape: Vec<usize>,
}

impl RecognizerOutput {
    pub(crate) fn new(logits: Tensor) -> Self {
        Self {
            shape: logits.dims().to_vec(),
            logits,
        }
    }

    /// Borrow the raw logits tensor without softmax, decoding, or CTC post-processing.
    pub fn tensor(&self) -> &Tensor {
        &self.logits
    }
}
