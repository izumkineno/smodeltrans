use anyhow::{Result, bail, ensure};
use std::collections::HashSet;

pub(crate) const MAX_NEW_TOKENS: usize = 4096;
pub(crate) const MAX_TOP_K: usize = 1024;
pub(crate) const MAX_STOP_TOKENS: usize = 32;
pub(crate) const MAX_STOP_STRINGS: usize = 16;
pub(crate) const MAX_STOP_STRING_CHARS: usize = 128;
pub(crate) const MAX_PROMPT_CHARS: usize = 8192;
pub(crate) const MAX_MEMORY_TOKENS: usize = 262_144;
pub(crate) const MAX_MEMORY_TURNS: usize = 1024;
pub(crate) const MAX_TOKEN_ID: u32 = 1_000_000;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GenerationConfig {
    pub(crate) max_new_tokens: usize,
    pub(crate) sampling: bool,
    pub(crate) temperature: f32,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
    pub(crate) seed: Option<u64>,
    pub(crate) repetition_penalty: f32,
    pub(crate) frequency_penalty: f32,
    pub(crate) stop_tokens: Vec<u32>,
    pub(crate) stop_strings: Vec<String>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 4096,
            sampling: true,
            temperature: 0.7,
            top_k: 20,
            top_p: 0.6,
            seed: None,
            repetition_penalty: 1.05,
            frequency_penalty: 0.0,
            stop_tokens: Vec::new(),
            // Hy 角色边界符：模型若尝试续写下一轮对话应立即截断，避免 7B 等模型在长输出末尾伪造 <｜hy_User｜> 等
            stop_strings: vec![
                "<｜hy_User｜>".to_owned(),
                "<｜hy_Assistant｜>".to_owned(),
                "<｜hy_begin▁of▁sentence｜>".to_owned(),
                "<｜hy_place▁holder▁no▁3｜>".to_owned(),
                "<|hy_User|>".to_owned(),
                "<|hy_Assistant|>".to_owned(),
            ],
        }
    }
}

impl GenerationConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        tracing::debug!(target: "config", max_new_tokens = self.max_new_tokens, sampling = self.sampling, temperature = self.temperature, top_k = self.top_k, top_p = self.top_p, "GenerationConfig::validate started");
        if !(1..=MAX_NEW_TOKENS).contains(&self.max_new_tokens) {
            bail!("generation.max_new_tokens must be in 1..={MAX_NEW_TOKENS}");
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            bail!("generation.temperature must be finite and greater than zero");
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            bail!("generation.top_p must be finite and in (0, 1]");
        }
        if self.top_k > MAX_TOP_K {
            bail!("generation.top_k must be in 0..={MAX_TOP_K}");
        }
        if self.sampling && self.top_k == 0 {
            bail!("generation.top_k must be greater than zero when sampling is enabled");
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            bail!("generation.repetition_penalty must be finite and greater than zero");
        }
        if !self.frequency_penalty.is_finite() || self.frequency_penalty < 0.0 {
            bail!("generation.frequency_penalty must be finite and non-negative");
        }
        if self.stop_tokens.len() > MAX_STOP_TOKENS {
            bail!("generation.stop_tokens must contain at most {MAX_STOP_TOKENS} items");
        }
        if self.stop_tokens.iter().any(|token| *token > MAX_TOKEN_ID) {
            bail!("generation.stop_tokens values must be in 0..={MAX_TOKEN_ID}");
        }
        if self.stop_strings.len() > MAX_STOP_STRINGS {
            bail!("generation.stop_strings must contain at most {MAX_STOP_STRINGS} items");
        }
        let mut seen = HashSet::with_capacity(self.stop_strings.len());
        for stop_string in &self.stop_strings {
            let trimmed = stop_string.trim();
            if trimmed.is_empty() {
                bail!("generation.stop_strings cannot contain an empty string");
            }
            if trimmed.chars().count() > MAX_STOP_STRING_CHARS {
                bail!(
                    "generation.stop_strings values must contain at most {MAX_STOP_STRING_CHARS} characters"
                );
            }
            let _ = seen.insert(trimmed);
        }
        tracing::debug!(target: "config", max_new_tokens = self.max_new_tokens, "GenerationConfig::validate succeeded");
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MemoryConfig {
    pub(crate) enabled: bool,
    pub(crate) max_tokens: usize,
    pub(crate) max_turns: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_tokens: 4096,
            max_turns: 16,
        }
    }
}

impl MemoryConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        tracing::debug!(target: "config", enabled = self.enabled, max_tokens = self.max_tokens, max_turns = self.max_turns, "MemoryConfig::validate started");
        if !(1..=MAX_MEMORY_TOKENS).contains(&self.max_tokens) {
            bail!("memory.max_tokens must be in 1..={MAX_MEMORY_TOKENS}");
        }
        if !(1..=MAX_MEMORY_TURNS).contains(&self.max_turns) {
            bail!("memory.max_turns must be in 1..={MAX_MEMORY_TURNS}");
        }
        tracing::debug!(target: "config", enabled = self.enabled, "MemoryConfig::validate succeeded");
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PromptConfig {
    /// 单一翻译指令模板，支持 `{source_text}` / `{target_lang}` / `{target_language}` / `{format_type}` 占位符。
    /// 为空时使用 Hy-MT2 官方 Default：`Translate the following text into {target_lang}. Note that you should only output the translated result without any additional explanation:\n\n{source_text}`
    /// 包含 `{source_text}` 时为结构化完整提示（覆盖默认），否则作为附加约束拼于默认模板前。
    pub(crate) template: String,
}

impl PromptConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        tracing::debug!(target: "config", template_len = self.template.chars().count(), has_placeholder = self.has_source_text_placeholder(), "PromptConfig::validate started");
        if self.template.trim().chars().count() > MAX_PROMPT_CHARS {
            bail!("prompt.template must contain at most {MAX_PROMPT_CHARS} characters");
        }
        tracing::debug!(target: "config", "PromptConfig::validate succeeded");
        Ok(())
    }

    /// 是否包含 `{source_text}` 占位符，决定是否作为完整模板覆盖默认
    pub(crate) fn has_source_text_placeholder(&self) -> bool {
        self.template.contains("{source_text}")
    }
}
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ModelConfig {
    pub(crate) target_language: String,
    pub(crate) prompt: PromptConfig,
    pub(crate) generation: GenerationConfig,
    pub(crate) memory: MemoryConfig,
}

impl ModelConfig {
    pub(crate) fn from_parts(
        target_language: &str,
        prompt: PromptConfig,
        generation: GenerationConfig,
        memory: MemoryConfig,
    ) -> Result<Self> {
        let _span = tracing::debug_span!(target: "config", "ModelConfig::from_parts", target_language = %target_language).entered();
        tracing::debug!(target: "config", target_language = %target_language, prompt_len = prompt.template.chars().count(), max_new_tokens = generation.max_new_tokens, memory_enabled = memory.enabled, "ModelConfig::from_parts started");
        let target_language = target_language.trim();
        ensure!(
            !target_language.is_empty(),
            "target language must not be empty"
        );
        prompt.validate()?;
        generation.validate()?;
        memory.validate()?;
        tracing::debug!(target: "config", target_language = %target_language, "ModelConfig::from_parts validation passed");
        Ok(Self {
            target_language: target_language.to_owned(),
            prompt,
            generation,
            memory,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{GenerationConfig, MAX_NEW_TOKENS, MAX_PROMPT_CHARS, PromptConfig};

    #[test]
    fn generation_config_validation_boundaries() {
        GenerationConfig::default().validate().unwrap();

        let mut sampled = GenerationConfig {
            sampling: true,
            top_k: 1,
            ..GenerationConfig::default()
        };
        sampled.validate().unwrap();

        sampled.top_k = 0;
        assert!(sampled.validate().is_err());

        let mut invalid = GenerationConfig {
            top_p: 0.0,
            ..GenerationConfig::default()
        };
        assert!(invalid.validate().is_err());

        invalid = GenerationConfig {
            temperature: 0.0,
            ..GenerationConfig::default()
        };
        assert!(invalid.validate().is_err());

        invalid = GenerationConfig {
            max_new_tokens: MAX_NEW_TOKENS + 1,
            ..GenerationConfig::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn generation_defaults_match_requested_hy_parameters() {
        let defaults = GenerationConfig::default();

        assert_eq!(defaults.max_new_tokens, 4096);
        assert!(defaults.sampling);
        assert_eq!(defaults.temperature, 0.7);
        assert_eq!(defaults.top_k, 20);
        assert_eq!(defaults.top_p, 0.6);
        assert_eq!(defaults.repetition_penalty, 1.05);
    }

    #[test]
    fn prompt_config_validation_covers_user_preset() {
        PromptConfig {
            template: "Preserve product names.".to_owned(),
        }
        .validate()
        .unwrap();

        assert!(
            PromptConfig {
                template: "x".repeat(MAX_PROMPT_CHARS + 1),
            }
            .validate()
            .is_err()
        );
    }
}
