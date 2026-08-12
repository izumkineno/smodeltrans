//! Hy-owned copy-on-write session and conversation memory.

use super::{
    generation::{self, HyGenerationResult, HyStopReason},
    model::{HyGenerationState, HySession},
};
use crate::{
    model_config::{GenerationConfig, MemoryConfig},
    model_support::CancellationToken,
};
use anyhow::{Result, bail};
use candle_core::Device;
use std::path::Path;

#[derive(Clone, Debug)]
struct HyConversationTurn {
    system: String,
    user: String,
    assistant_token_ids: Vec<u32>,
    token_count: usize,
}

#[derive(Clone, Debug)]
struct HyConversationMemory {
    enabled: bool,
    max_tokens: usize,
    max_turns: usize,
    turns: Vec<HyConversationTurn>,
}

impl HyConversationMemory {
    fn from_config(config: MemoryConfig) -> Self {
        Self {
            enabled: config.enabled,
            max_tokens: config.max_tokens,
            max_turns: config.max_turns,
            turns: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.turns.clear();
    }

    fn token_count(&self) -> usize {
        self.turns.iter().map(|turn| turn.token_count).sum()
    }

    fn record_turn(
        &mut self,
        system: &str,
        user: &str,
        prompt_token_count: usize,
        assistant_token_ids: Vec<u32>,
    ) -> bool {
        if !self.enabled
            || self
                .turns
                .iter()
                .any(|turn| memory_text_similarity(&turn.user, user) > 0.80)
        {
            return false;
        }
        let token_count = prompt_token_count.saturating_add(assistant_token_ids.len());
        self.turns.push(HyConversationTurn {
            system: system.to_owned(),
            user: user.to_owned(),
            assistant_token_ids,
            token_count,
        });
        true
    }

    fn trim(&mut self) -> bool {
        if !self.enabled {
            return false;
        }
        let mut trimmed = false;
        while self.turns.len() > 1
            && (self.turns.len() > self.max_turns || self.token_count() > self.max_tokens)
        {
            self.turns.remove(0);
            trimmed = true;
        }
        trimmed
    }

    fn validate_budget(&self) -> Result<()> {
        if self.enabled
            && !self.turns.is_empty()
            && (self.turns.len() > self.max_turns || self.token_count() > self.max_tokens)
        {
            bail!(
                "conversation context exceeds memory budget (tokens {}/{}, turns {}/{})",
                self.token_count(),
                self.max_tokens,
                self.turns.len(),
                self.max_turns
            );
        }
        Ok(())
    }
}

fn memory_text_similarity(left: &str, right: &str) -> f32 {
    let left = normalize_memory_text(left);
    let right = normalize_memory_text(right);
    if left == right {
        return 1.0;
    }
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }

    let left_chars: Vec<char> = left.chars().collect();
    let right_chars: Vec<char> = right.chars().collect();
    if left_chars.len() < 2 || right_chars.len() < 2 {
        return 0.0;
    }
    let mut left_ngrams = character_ngrams(&left_chars);
    let mut right_ngrams = character_ngrams(&right_chars);
    left_ngrams.sort_unstable();
    right_ngrams.sort_unstable();

    let mut left_index = 0;
    let mut right_index = 0;
    let mut intersection = 0usize;
    while left_index < left_ngrams.len() && right_index < right_ngrams.len() {
        match left_ngrams[left_index].cmp(&right_ngrams[right_index]) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
            std::cmp::Ordering::Equal => {
                intersection += 1;
                left_index += 1;
                right_index += 1;
            }
        }
    }
    (2 * intersection) as f32 / (left_ngrams.len() + right_ngrams.len()) as f32
}

fn normalize_memory_text(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn character_ngrams(chars: &[char]) -> Vec<(char, char)> {
    chars.windows(2).map(|pair| (pair[0], pair[1])).collect()
}

/// Loaded Hy resources plus the private state needed to service requests.
pub(crate) struct HySessionDriver {
    model: HySession,
    initial_state: HyGenerationState,
    state: HyGenerationState,
    position: usize,
    replay_pending: bool,
    memory: HyConversationMemory,
}

impl HySessionDriver {
    pub(crate) fn new(model_path: &Path, device: &Device, memory: MemoryConfig) -> Result<Self> {
        let model = HySession::new(model_path, device)?;
        let initial_state = model.new_state()?;
        Ok(Self {
            model,
            initial_state: initial_state.clone(),
            state: initial_state,
            position: 0,
            replay_pending: false,
            memory: HyConversationMemory::from_config(memory),
        })
    }

    /// Clear the retained conversation state while keeping model weights loaded.
    pub(crate) fn reset(&mut self) {
        self.state = self.initial_state.clone();
        self.position = 0;
        self.replay_pending = false;
        self.memory.clear();
    }

    /// Run one request against a cloned state and commit only on success.
    pub(crate) fn respond(
        &mut self,
        system_prompt: &str,
        user_prompt: &str,
        config: &GenerationConfig,
        on_chunk: impl FnMut(&str) -> Result<()>,
        cancellation: &CancellationToken,
    ) -> Result<HyGenerationResult> {
        cancellation
            .check()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let mut state = self.state.clone();
        let mut position = self.position;
        let mut memory = self.memory.clone();
        let mut replay_pending = self.replay_pending;

        if !memory.enabled {
            state = self.initial_state.clone();
            position = 0;
            replay_pending = false;
            memory.clear();
        } else {
            let trimmed = memory.trim();
            memory.validate_budget()?;
            if trimmed || replay_pending {
                state = self.initial_state.clone();
                position = 0;
                replay(&self.model, &mut state, &mut position, &memory)?;
                cancellation
                    .check()
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                replay_pending = false;
            }
        }
        let prompt_token_count = if memory.enabled {
            let count = self
                .model
                .prompt_token_count(&state, system_prompt, user_prompt)?;
            anyhow::ensure!(
                count <= memory.max_tokens,
                "prompt exceeds the configured memory token budget"
            );
            Some(count)
        } else {
            None
        };

        let result = generation::generate(
            &self.model,
            &mut state,
            &mut position,
            system_prompt,
            user_prompt,
            config,
            on_chunk,
            cancellation,
        )?;
        if matches!(result.stop_reason, HyStopReason::Cancelled) {
            return Ok(result);
        }
        if matches!(result.stop_reason, HyStopReason::ContextLength) {
            // The accepted output filled the fixed context. Reset explicitly so
            // the next request cannot reuse a cache that has no append slot.
            self.reset();
            return Ok(result);
        }

        if memory.enabled {
            let recorded = memory.record_turn(
                system_prompt,
                user_prompt,
                prompt_token_count.unwrap_or_default(),
                result.generated_ids.clone(),
            );
            if recorded {
                let trimmed = memory.trim();
                memory.validate_budget()?;
                if trimmed {
                    replay_pending = true;
                }
            }
        } else {
            replay_pending = false;
        }

        self.state = state;
        self.position = position;
        self.replay_pending = replay_pending;
        self.memory = memory;
        Ok(result)
    }
}

fn replay(
    model: &HySession,
    state: &mut HyGenerationState,
    position: &mut usize,
    memory: &HyConversationMemory,
) -> Result<()> {
    for turn in &memory.turns {
        let (_, prompt_ids) = model.prefill(state, &turn.system, &turn.user, *position)?;
        *position = position
            .checked_add(prompt_ids.len())
            .ok_or_else(|| anyhow::anyhow!("generation position overflowed usize"))?;
        model.replay_assistant_tokens(state, &turn.assistant_token_ids, *position)?;
        *position = position
            .checked_add(turn.assistant_token_ids.len())
            .ok_or_else(|| anyhow::anyhow!("generation position overflowed usize"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{HyConversationMemory, memory_text_similarity};
    use crate::model_config::MemoryConfig;

    #[test]
    fn duplicate_memory_text_requires_similarity_above_eighty_percent() {
        assert!(memory_text_similarity("A quick brown fox.", "a quick brown fox!") > 0.80);
        assert!(memory_text_similarity("A quick brown fox.", "Close the window.") <= 0.80);
    }

    #[test]
    fn memory_skips_similar_turns() {
        let mut memory = HyConversationMemory::from_config(MemoryConfig {
            enabled: true,
            max_tokens: 128,
            max_turns: 4,
        });
        assert!(memory.record_turn("system", "Translate: Hello, world.", 4, vec![1, 2]));
        assert!(!memory.record_turn("system", "translate: hello, world!", 4, vec![3, 4]));
        assert_eq!(memory.turns.len(), 1);
    }
}
