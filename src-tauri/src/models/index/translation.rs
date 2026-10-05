//! Index 翻译入口（签名对标 `HyTranslator::translate_text`）。
use super::{
    prompt::{strip_think, trans_prompt},
    session::IndexSession,
};
use crate::{
    model_config::{GenerationConfig, MemoryConfig, PromptConfig},
    model_support::CancellationToken,
};
use anyhow::Result;
use candle_core::Device;
use std::path::Path;
pub(crate) struct IndexTranslator {
    session: IndexSession,
}
pub(crate) fn load_with_config(
    model_path: &Path,
    device: &Device,
    _memory: MemoryConfig,
    _generation: GenerationConfig,
    _prompt: PromptConfig,
) -> Result<IndexTranslator> {
    Ok(IndexTranslator {
        session: IndexSession::new(model_path, device)?,
    })
}
impl IndexTranslator {
    /// 纯文本翻译：instTrans prompt → session 贪心解码。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn translate_text(
        &mut self,
        text: &str,
        target_language: &str,
        _prompt: &PromptConfig,
        supplemental_prompt: &str,
        generation: &GenerationConfig,
        cancellation: &CancellationToken,
        on_chunk: impl FnMut(&str) -> Result<()>,
    ) -> Result<String> {
        let extra;
        let hard: &[String] = if supplemental_prompt.is_empty() {
            &[]
        } else {
            extra = vec![supplemental_prompt.to_owned()];
            &extra
        };
        let prompt = trans_prompt(text, target_language, "auto", hard, &[], &[], false);
        let out = self
            .session
            .translate(&prompt, generation, cancellation, on_chunk)?;
        Ok(super::prompt::strip_think(&out))
    }
}

/// e2e：真实 GGUF 在本地时跑中→英三例；缺权重跳过并明确提示。
#[cfg(test)]
mod e2e {
    use super::*;

    fn gguf_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("models")
            .join("Index-Translate-2B.Q4_K_M.gguf")
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn index_e2e_zh_en_smoke() {
        let path = gguf_path();
        if !path.exists() {
            println!("SKIP index_e2e: GGUF not found at {}", path.display());
            return;
        }
        let device = Device::new_cuda(0).expect("index e2e requires CUDA device 0");
        let mut translator = load_with_config(
            &path,
            &device,
            MemoryConfig::default(),
            GenerationConfig::default(),
            PromptConfig::default(),
        )
        .expect("index load failed");
        let cancel = CancellationToken::new_for_test();
        let mut generation = GenerationConfig::default();
        generation.max_new_tokens = 64;
        let prompt = PromptConfig::default();

        let plain = translator
            .translate_text(
                "你好，世界。今天天气不错，我们去公园散步吧。",
                "en",
                &prompt,
                "",
                &generation,
                &cancel,
                |_| Ok(()),
            )
            .expect("plain translation failed");
        let plain_lower = plain.to_lowercase();
        println!("plain translation: {plain}");
        assert!(
            plain_lower.contains("hello") || plain_lower.contains("park"),
            "unexpected plain translation: {plain}"
        );

        let json = translator
            .translate_text(
                "{\"greeting\":\"你好\",\"city\":\"北京\"}",
                "en",
                &prompt,
                "输出严格有效的 JSON，不要加代码块。保留所有 key，只翻译 value。",
                &generation,
                &cancel,
                |_| Ok(()),
            )
            .expect("JSON translation failed");
        println!("json translation: {json}");
        let parsed: serde_json::Value = serde_json::from_str(json.trim())
            .unwrap_or_else(|error| panic!("invalid JSON translation ({error}): {json}"));
        assert!(
            parsed["greeting"]
                .as_str()
                .unwrap_or_default()
                .to_lowercase()
                .contains("hello")
        );
        assert!(
            parsed["city"]
                .as_str()
                .unwrap_or_default()
                .to_lowercase()
                .contains("beijing")
        );

        let glossary = translator
            .translate_text(
                "碳纤维材料轻便而坚固。",
                "en",
                &prompt,
                "术语对照：碳纤维必须翻译为 carbon fiber。",
                &generation,
                &cancel,
                |_| Ok(()),
            )
            .expect("glossary translation failed");
        println!("glossary translation: {glossary}");
        assert!(
            glossary.to_lowercase().contains("carbon fiber"),
            "glossary term missing: {glossary}"
        );
    }
}
