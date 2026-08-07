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

    fn record_turn(&mut self, user: &str, user_token_count: usize, assistant_token_ids: Vec<u32>) {
        if !self.enabled {
            return;
        }
        let token_count = user_token_count.saturating_add(assistant_token_ids.len());
        self.turns.push(HyConversationTurn {
            user: user.to_owned(),
            assistant_token_ids,
            token_count,
        });
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
        prompt: &str,
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
            let count = self.model.prompt_token_count(&state, prompt)?;
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
            prompt,
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
            memory.record_turn(
                prompt,
                prompt_token_count.unwrap_or_default(),
                result.generated_ids.clone(),
            );
            let trimmed = memory.trim();
            memory.validate_budget()?;
            if trimmed {
                replay_pending = true;
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
        let (_, prompt_ids) = model.prefill(state, &turn.user, *position)?;
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
