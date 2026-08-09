//! Hy-owned autoregressive generation policy.
//!
//! This module deliberately does not implement or expose a shared adapter
//! contract. It owns the generation loop, stop handling, sampling policy, and
//! output accounting for the Hy model only.

use super::model::{HyGenerationState, HySession};
use crate::{model_config::GenerationConfig, model_support::CancellationToken};
#[cfg(test)]
use anyhow::bail;
use anyhow::{Result, anyhow};
#[cfg(test)]
use std::collections::HashMap;
use std::{
    str,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HyStopReason {
    Eos,
    StopToken,
    StopString,
    MaxTokens,
    ContextLength,
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HyGenerationStats {
    pub(crate) generated_tokens: usize,
    pub(crate) decode_time: Duration,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HyGenerationResult {
    pub(crate) text: String,
    pub(crate) stop_reason: HyStopReason,
    pub(crate) stats: HyGenerationStats,
    pub(crate) generated_ids: Vec<u32>,
}

/// Decoder state backed by tokenizers' stateful stream decoder.
#[derive(Default)]
pub(super) struct HyDecoderState {
    pub(super) token_ids: Vec<u32>,
    pub(super) prefix: String,
    pub(super) prefix_index: usize,
}

pub(super) trait HyTokenRng {
    fn next_f64(&mut self) -> f64;
}

/// Generate one response against exactly the supplied Hy state.
pub(super) fn generate(
    model: &HySession,
    state: &mut HyGenerationState,
    position: &mut usize,
    prompt: &str,
    config: &GenerationConfig,
    mut on_chunk: impl FnMut(&str) -> Result<()>,
    cancellation: &CancellationToken,
) -> Result<HyGenerationResult> {
    validate_config(config)?;
    let started = Instant::now();
    if cancellation.is_cancelled() {
        return make_result(
            String::new(),
            HyStopReason::Cancelled,
            0,
            started,
            Vec::new(),
        );
    }

    model.prepare_penalty_state(state, *position, config)?;
    let (mut logits, prompt_ids) = model.prefill(state, prompt, *position)?;
    *position = position
        .checked_add(prompt_ids.len())
        .ok_or_else(|| anyhow!("generation position overflowed usize"))?;

    let mut decoder = HyDecoderState::default();
    let mut output = Vec::new();
    let mut visible = String::new();
    let mut generated_ids = Vec::new();
    let mut generated_tokens = 0usize;
    let mut rng = XorShift64::new(config.seed.unwrap_or(DEFAULT_SEED));
    let selection = model.prepare_selection(config, &mut rng)?;
    loop {
        if cancellation.is_cancelled() {
            finish_visible_output(&output, &mut visible, &mut on_chunk)?;
            return make_result(
                visible,
                HyStopReason::Cancelled,
                generated_tokens,
                started,
                generated_ids,
            );
        }
        if generated_tokens >= config.max_new_tokens {
            finish_visible_output(&output, &mut visible, &mut on_chunk)?;
            return make_result(
                visible,
                HyStopReason::MaxTokens,
                generated_tokens,
                started,
                generated_ids,
            );
        }
        if !model.has_context_capacity(*position)? {
            finish_visible_output(&output, &mut visible, &mut on_chunk)?;
            return make_result(
                visible,
                HyStopReason::ContextLength,
                generated_tokens,
                started,
                generated_ids,
            );
        }

        let (token_id, device_token) =
            model.select_token(state, &logits, config, &selection, generated_tokens)?;
        if config.stop_tokens.contains(&token_id) {
            finish_visible_output(&output, &mut visible, &mut on_chunk)?;
            return make_result(
                visible,
                HyStopReason::StopToken,
                generated_tokens,
                started,
                generated_ids,
            );
        }
        if model.is_stop_token(token_id) {
            finish_visible_output(&output, &mut visible, &mut on_chunk)?;
            return make_result(
                visible,
                HyStopReason::Eos,
                generated_tokens,
                started,
                generated_ids,
            );
        }

        generated_tokens += 1;
        generated_ids.push(token_id);
        let next_logits = model.step(state, &device_token, *position)?;
        *position = position
            .checked_add(1)
            .ok_or_else(|| anyhow!("generation position overflowed usize"))?;
        output.extend_from_slice(&model.decode_token(&mut decoder, token_id)?);
        if process_output(&output, &mut visible, &config.stop_strings, &mut on_chunk)? {
            return make_result(
                visible,
                HyStopReason::StopString,
                generated_tokens,
                started,
                generated_ids,
            );
        }
        if generated_tokens >= config.max_new_tokens {
            finish_visible_output(&output, &mut visible, &mut on_chunk)?;
            return make_result(
                visible,
                HyStopReason::MaxTokens,
                generated_tokens,
                started,
                generated_ids,
            );
        }

        logits = next_logits;
    }
}

fn make_result(
    text: String,
    stop_reason: HyStopReason,
    generated_tokens: usize,
    started: Instant,
    generated_ids: Vec<u32>,
) -> Result<HyGenerationResult> {
    Ok(HyGenerationResult {
        text,
        stop_reason,
        stats: HyGenerationStats {
            generated_tokens,
            decode_time: started.elapsed(),
        },
        generated_ids,
    })
}

fn validate_config(config: &GenerationConfig) -> Result<()> {
    config.validate()
}

#[cfg(test)]
pub(super) fn select_token_cpu(
    logits: &[f32],
    config: &GenerationConfig,
    prompt_ids: &[u32],
    generated_ids: &[u32],
    rng: &mut dyn HyTokenRng,
) -> Result<u32> {
    if logits.is_empty() {
        bail!("backend returned empty logits");
    }
    let needs_penalties = config.repetition_penalty != 1.0 || config.frequency_penalty != 0.0;
    if !config.sampling && !needs_penalties {
        return logits
            .iter()
            .enumerate()
            .max_by(|(left_id, left), (right_id, right)| {
                left.partial_cmp(right)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| right_id.cmp(left_id))
            })
            .map(|(id, _)| id as u32)
            .ok_or_else(|| anyhow!("backend returned empty logits"));
    }

    let mut adjusted = logits.to_vec();
    let counts = if needs_penalties {
        let mut counts =
            HashMap::<u32, usize>::with_capacity(prompt_ids.len() + generated_ids.len());
        for &token in prompt_ids.iter().chain(generated_ids.iter()) {
            *counts.entry(token).or_default() += 1;
        }
        Some(counts)
    } else {
        None
    };
    for (id, value) in adjusted.iter_mut().enumerate() {
        if !value.is_finite() {
            bail!("backend returned non-finite logits");
        }
        if let Some(counts) = counts.as_ref() {
            if let Some(&count) = counts.get(&(id as u32)) {
                if config.repetition_penalty != 1.0 {
                    if *value >= 0.0 {
                        *value /= config.repetition_penalty;
                    } else {
                        *value *= config.repetition_penalty;
                    }
                }
                if config.frequency_penalty != 0.0 {
                    *value -= config.frequency_penalty * count as f32;
                }
            }
        }
        if !value.is_finite() {
            bail!("backend returned non-finite logits");
        }
    }

    if !config.sampling {
        return adjusted
            .iter()
            .enumerate()
            .max_by(|(left_id, left), (right_id, right)| {
                left.partial_cmp(right)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| right_id.cmp(left_id))
            })
            .map(|(id, _)| id as u32)
            .ok_or_else(|| anyhow!("backend returned empty logits"));
    }

    let mut candidates: Vec<usize> = (0..adjusted.len()).collect();
    if config.top_k > 0 && candidates.len() > config.top_k {
        let top_k = config.top_k;
        candidates.select_nth_unstable_by(top_k - 1, |left, right| {
            adjusted[*right]
                .partial_cmp(&adjusted[*left])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.cmp(right))
        });
        candidates.truncate(top_k);
    }
    candidates.sort_by(|left, right| {
        adjusted[*right]
            .partial_cmp(&adjusted[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    let max_logit = candidates
        .iter()
        .map(|id| adjusted[*id] / config.temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let fallback = candidates
        .first()
        .copied()
        .map(|id| id as u32)
        .ok_or_else(|| anyhow!("backend returned empty logits"))?;
    let mut weighted: Vec<(usize, f64)> = candidates
        .into_iter()
        .map(|id| {
            let value = adjusted[id];
            (id, ((value / config.temperature - max_logit) as f64).exp())
        })
        .collect();
    let total: f64 = weighted.iter().map(|(_, weight)| *weight).sum();
    if !total.is_finite() || total <= 0.0 {
        bail!("backend returned non-finite logits");
    }
    for (_, weight) in &mut weighted {
        *weight /= total;
    }
    if config.top_p < 1.0 {
        let mut cumulative = 0.0;
        let mut keep = 0usize;
        for (_, weight) in &weighted {
            cumulative += *weight;
            keep += 1;
            if cumulative >= f64::from(config.top_p) {
                break;
            }
        }
        weighted.truncate(keep.max(1));
        let retained_total: f64 = weighted.iter().map(|(_, weight)| *weight).sum();
        if !retained_total.is_finite() || retained_total <= 0.0 {
            bail!("backend returned non-finite top-p probability mass");
        }
        for (_, weight) in &mut weighted {
            *weight /= retained_total;
        }
    }
    let threshold = rng.next_f64();
    let mut cumulative = 0.0;
    for (id, weight) in weighted {
        cumulative += weight;
        if threshold < cumulative {
            return Ok(id as u32);
        }
    }
    Ok(fallback)
}

fn process_output(
    output: &[u8],
    visible: &mut String,
    stop_strings: &[String],
    on_chunk: &mut impl FnMut(&str) -> Result<()>,
) -> Result<bool> {
    let text = match str::from_utf8(output) {
        Ok(text) => text,
        Err(error) if error.error_len().is_none() => {
            let valid_up_to = error.valid_up_to();
            str::from_utf8(&output[..valid_up_to])
                .map_err(|_| anyhow!("decoder returned invalid UTF-8 at byte {valid_up_to}"))?
        }
        Err(error) => {
            return Err(anyhow!(
                "decoder returned invalid UTF-8 at byte {}",
                error.valid_up_to()
            ));
        }
    };
    if let Some((start, _)) = find_stop(text, stop_strings) {
        emit_text(&text[..start], visible, true, on_chunk)?;
        return Ok(true);
    }
    let hold_len = longest_stop_prefix(text.as_bytes(), stop_strings);
    let safe_end = text.len().saturating_sub(hold_len);
    emit_text(&text[..safe_end], visible, false, on_chunk)?;
    Ok(false)
}

fn find_stop(text: &str, stop_strings: &[String]) -> Option<(usize, usize)> {
    stop_strings
        .iter()
        .filter_map(|stop| text.find(stop).map(|start| (start, stop.len())))
        .min_by_key(|(start, _)| *start)
}

fn longest_stop_prefix(valid: &[u8], stop_strings: &[String]) -> usize {
    stop_strings
        .iter()
        .map(|stop| {
            let stop_bytes = stop.as_bytes();
            let max = stop_bytes.len().saturating_sub(1).min(valid.len());
            (1..=max)
                .rev()
                .find(|&length| valid.ends_with(&stop_bytes[..length]))
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0)
}

fn emit_text(
    text: &str,
    visible: &mut String,
    final_chunk: bool,
    on_chunk: &mut impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    let mut utf8 = [0u8; 4];
    let mut on_character = |character: char| {
        on_chunk(character.encode_utf8(&mut utf8))?;
        Ok(())
    };
    let result = if final_chunk {
        finish_incremental(text, visible, &mut on_character)
    } else {
        emit_incremental(text, visible, &mut on_character)
    };
    result.map_err(|error| anyhow!("generation output callback failed: {error}"))
}

const HOLD_CHARACTERS: usize = 4;

fn emit_incremental<F>(decoded: &str, emitted: &mut String, on_character: &mut F) -> Result<()>
where
    F: FnMut(char) -> Result<()>,
{
    emit_prefix(decoded, emitted, on_character, false)
}

fn finish_incremental<F>(decoded: &str, emitted: &mut String, on_character: &mut F) -> Result<()>
where
    F: FnMut(char) -> Result<()>,
{
    emit_prefix(decoded, emitted, on_character, true)
}

fn emit_prefix<F>(
    decoded: &str,
    emitted: &mut String,
    on_character: &mut F,
    final_chunk: bool,
) -> Result<()>
where
    F: FnMut(char) -> Result<()>,
{
    let stable_prefix = if final_chunk {
        decoded
    } else {
        decoded
            .char_indices()
            .nth_back(HOLD_CHARACTERS - 1)
            .map(|(index, _)| &decoded[..index])
            .unwrap_or("")
    };
    let delta = stable_prefix
        .strip_prefix(emitted.as_str())
        .ok_or_else(|| anyhow!("tokenizer output changed an already streamed character"))?;
    for character in delta.chars() {
        on_character(character)?;
        emitted.push(character);
    }
    Ok(())
}

fn finish_visible_output(
    output: &[u8],
    visible: &mut String,
    on_chunk: &mut impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    let text = str::from_utf8(output).map_err(|error| {
        anyhow!(
            "decoder returned invalid UTF-8 at byte {}",
            error.valid_up_to()
        )
    })?;
    emit_text(text, visible, true, on_chunk)
}

#[derive(Clone, Copy, Debug)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { DEFAULT_SEED } else { seed },
        }
    }

    fn next_f64(&mut self) -> f64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        // Use the high 53 bits so the sampling threshold is always in [0, 1).
        ((value >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

impl HyTokenRng for XorShift64 {
    fn next_f64(&mut self) -> f64 {
        Self::next_f64(self)
    }
}

const DEFAULT_SEED: u64 = 0x4d595f47454e4552;
