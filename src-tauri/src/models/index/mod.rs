//! Target-owned Index-Translate GGUF translation provider.
//!
//! GGUF `general.architecture == "qwen35"`（Qwen3.5 hybrid：3 linear + 1 full）×6 + MTP 1 层。
//! linear 层为 Gated DeltaNet（`ssm_*` tensor），full 层为 GQA（`attn_q/k/v`）。
//! 主参照：transformers `Qwen3NextGatedDeltaNet` 前向；prompt 仿写官方 `translate.py:trans_prompt()`。

pub(crate) mod deltanet;
pub(crate) mod full;
pub(crate) mod model;
pub(crate) mod prompt;
pub(crate) mod session;
pub(crate) mod translation;
pub(crate) use translation::{IndexTranslator, load_with_config};
