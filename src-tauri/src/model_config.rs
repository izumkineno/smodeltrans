use anyhow::{Result, ensure};

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
            max_new_tokens: 128,
            sampling: false,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            stop_tokens: Vec::new(),
            stop_strings: Vec::new(),
        }
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

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PromptConfig {
    pub(crate) system: String,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ModelConfig {
    pub(crate) target_language: String,
    pub(crate) prompt: PromptConfig,
    pub(crate) generation: GenerationConfig,
    pub(crate) memory: MemoryConfig,
}

impl ModelConfig {
    pub(crate) fn for_target(target_language: &str) -> Result<Self> {
        let target_language = target_language.trim();
        ensure!(
            !target_language.is_empty(),
            "target language must not be empty"
        );
        Ok(Self {
            target_language: target_language.to_owned(),
            prompt: PromptConfig::default(),
            generation: GenerationConfig::default(),
            memory: MemoryConfig::default(),
        })
    }
}
