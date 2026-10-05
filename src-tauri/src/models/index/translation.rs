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

    fn fallback_gguf_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("models")
            .join("Index-Translate-2B.f16.gguf")
    }

    /// 流式统计：首 token 时间 / token 数 / 总耗时（一个 token 对应一次 on_chunk 调用）。
    struct StreamStat {
        t0: std::time::Instant,
        first_ms: Option<f64>,
        tokens: u64,
    }

    impl StreamStat {
        fn new() -> Self {
            Self {
                t0: std::time::Instant::now(),
                first_ms: None,
                tokens: 0,
            }
        }

        fn on_chunk(&mut self, chunk: &str) -> Result<()> {
            use std::io::Write as _;
            if self.first_ms.is_none() {
                self.first_ms = Some(self.t0.elapsed().as_secs_f64() * 1000.0);
            }
            self.tokens += 1;
            print!("{chunk}");
            let _ = std::io::stdout().flush();
            Ok(())
        }

        fn report(&self, name: &str, out_chars: usize) {
            let total_s = self.t0.elapsed().as_secs_f64();
            println!(
                "\n{name}: TTFT={:.0}ms tokens={} total={:.2}s tok/s={:.1} ({} chars)",
                self.first_ms.unwrap_or(-1.0),
                self.tokens,
                total_s,
                self.tokens as f64 / total_s.max(1e-6),
                out_chars,
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn index_e2e_zh_en_smoke() {
        let path = gguf_path();
        let path = if path.exists() {
            path
        } else {
            let fallback = fallback_gguf_path();
            if fallback.exists() {
                fallback
            } else {
                println!(
                    "SKIP index_e2e: GGUF not found at {} or {}",
                    path.display(),
                    fallback.display()
                );
                return;
            }
        };
        let device = Device::new_cuda(0).expect("index e2e requires CUDA device 0");
        let t_load = std::time::Instant::now();
        let mut translator = load_with_config(
            &path,
            &device,
            MemoryConfig::default(),
            GenerationConfig::default(),
            PromptConfig::default(),
        )
        .expect("index load failed");
        println!("index_e2e load: {:.2}s", t_load.elapsed().as_secs_f32());
        let cancel = CancellationToken::new_for_test();
        let mut generation = GenerationConfig::default();
        generation.max_new_tokens = 64;
        let prompt = PromptConfig::default();

        let t_total = std::time::Instant::now();
        let mut stat_plain = StreamStat::new();
        let plain = translator
            .translate_text(
                "你好，世界。今天天气不错，我们去公园散步吧。",
                "en",
                &prompt,
                "",
                &generation,
                &cancel,
                |chunk| stat_plain.on_chunk(chunk),
            )
            .expect("plain translation failed");
        stat_plain.report("plain", plain.chars().count());
        let plain_lower = plain.to_lowercase();
        println!("plain translation: {plain}");
        assert!(
            plain_lower.contains("hello") || plain_lower.contains("park"),
            "unexpected plain translation: {plain}"
        );

        let mut stat_json = StreamStat::new();
        let json = translator
            .translate_text(
                "{\"greeting\":\"你好\",\"city\":\"北京\"}",
                "en",
                &prompt,
                "输出严格有效的 JSON，不要加代码块。保留所有 key，只翻译 value。",
                &generation,
                &cancel,
                |chunk| stat_json.on_chunk(chunk),
            )
            .expect("JSON translation failed");
        stat_json.report("json", json.chars().count());
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

        let mut stat_glossary = StreamStat::new();
        let glossary = translator
            .translate_text(
                "碳纤维材料轻便而坚固。",
                "en",
                &prompt,
                "术语对照：碳纤维必须翻译为 carbon fiber。",
                &generation,
                &cancel,
                |chunk| stat_glossary.on_chunk(chunk),
            )
            .expect("glossary translation failed");
        stat_glossary.report("glossary", glossary.chars().count());
        println!("glossary translation: {glossary}");
        assert!(
            glossary.to_lowercase().contains("carbon fiber"),
            "glossary term missing: {glossary}"
        );

        let mut stat_long = StreamStat::new();
        generation.max_new_tokens = 256;
        let long = translator
            .translate_text(
                "北京市是中国的首都，也是一座拥有三千多年建城史的历史文化名城。\
                故宫、天坛、颐和园和八达岭长城都是世界闻名的文化遗产。\
                每年都有数千万游客来到北京，参观这些古迹并品尝地道的北京烤鸭。\
                近年来，北京还大力发展高新技术产业，中关村成为了中国科技创新的重要引擎。\
                便捷的地铁网络连接着城市的每一个角落，让市民的出行更加高效环保。",
                "en",
                &prompt,
                "",
                &generation,
                &cancel,
                |chunk| stat_long.on_chunk(chunk),
            )
            .expect("long-input translation failed");
        stat_long.report("long", long.chars().count());
        println!("long translation: {long}");
        assert!(
            !long.trim().is_empty(),
            "long-input translation came back empty"
        );
        assert!(
            long.to_lowercase().contains("beijing"),
            "long-input translation missing key term: {long}"
        );
        println!(
            "index_e2e long-input tokens: {} (prompt was long-form)",
            stat_long.tokens
        );
        println!(
            "index_e2e total: {:.2}s ({} tokens)",
            t_total.elapsed().as_secs_f32(),
            stat_plain.tokens + stat_json.tokens + stat_glossary.tokens + stat_long.tokens
        );
    }
}
