//! Hy-owned text and structured translation entrypoints.
//!
//! The request payloads in this module are neutral translation data. OCR model
//! types never cross this boundary, and all loading/session/generation state
//! remains private to Hy.

use super::{
    assets::HyAssets,
    generation::{HyGenerationResult, HyStopReason},
    session::HySessionDriver,
};
use crate::{
    backend::{
        contracts::{TranslatedRegion, TranslationRegion},
        failure::BackendFailure,
    },
    model_config::{GenerationConfig, MAX_NEW_TOKENS, MemoryConfig, PromptConfig},
    model_support::CancellationToken,
};
use anyhow::{Context, Result, ensure};
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::path::Path;

const MIN_OCR_TRANSLATION_NEW_TOKENS: usize = 4096;
const OCR_TRANSLATION_TOKENS_PER_SOURCE_CHAR: usize = 2;
const MIN_PROMPT_ECHO_CHARS: usize = 16;

/// Loaded Hy translation session. The concrete GGUF/session state remains
/// private to this provider.
pub(crate) struct HyTranslator {
    session: HySessionDriver,
    generation: GenerationConfig,
    prompt: PromptConfig,
    warmed_up: bool,
}

pub(crate) fn load_with_config(
    model_path: &Path,
    device: &Device,
    memory: MemoryConfig,
    generation: GenerationConfig,
    prompt: PromptConfig,
) -> Result<HyTranslator> {
    let assets =
        HyAssets::preflight(model_path).map_err(|error| anyhow::anyhow!(error.to_string()))?;
    ensure!(device.is_cuda(), "Hy requires a CUDA device");
    Ok(HyTranslator {
        session: HySessionDriver::new(&assets.model, device, memory)?,
        generation,
        prompt,
        warmed_up: false,
    })
}

impl HyTranslator {
    pub(crate) fn warm_up(&mut self, cancellation: &CancellationToken) -> Result<()> {
        if self.warmed_up {
            return Ok(());
        }
        let mut generation = self.generation.clone();
        generation.max_new_tokens = 1;
        let result = self
            .session
            .respond("", "Warm up.", &generation, |_| Ok(()), cancellation);
        result?;
        cancellation
            .check()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        self.warmed_up = true;
        Ok(())
    }

    pub(crate) fn reset_context(&mut self) {
        self.session.reset();
    }

    /// Translate plain neutral text using the Hy translation prompt contract.
    pub(crate) fn translate_text(
        &mut self,
        text: &str,
        target_language: &str,
        prompt: &PromptConfig,
        supplemental_prompt: &str,
        generation: &GenerationConfig,
        cancellation: &CancellationToken,
        on_chunk: impl FnMut(&str) -> Result<()>,
    ) -> Result<HyGenerationResult> {
        self.warmed_up = true;
        let user_prompt = apply_user_prompt_preset(
            build_translation_prompt(text, target_language),
            prompt,
            supplemental_prompt,
        );
        self.session.respond(
            prompt.system.trim(),
            &user_prompt,
            generation,
            on_chunk,
            cancellation,
        )
    }

    /// Translate one neutral structured batch and return texts in input order.
    pub(crate) fn translate_structured_batch(
        &mut self,
        jobs: &[TranslationRegion],
        target_language: &str,
        prompt: &PromptConfig,
        generation: &GenerationConfig,
        cancellation: &CancellationToken,
        supplemental_prompt: &str,
    ) -> Result<(Vec<String>, usize)> {
        self.translate_structured_batch_with_context(
            jobs,
            target_language,
            prompt,
            generation,
            cancellation,
            supplemental_prompt,
            false,
        )
    }

    fn translate_structured_batch_with_context(
        &mut self,
        jobs: &[TranslationRegion],
        target_language: &str,
        prompt: &PromptConfig,
        generation: &GenerationConfig,
        cancellation: &CancellationToken,
        supplemental_prompt: &str,
        contextual: bool,
    ) -> Result<(Vec<String>, usize)> {
        ensure!(!jobs.is_empty(), "translation batch must not be empty");
        let batch_prompt = if contextual {
            build_contextual_translation_batch_prompt(jobs, target_language)?
        } else {
            build_translation_batch_prompt(jobs, target_language)?
        };
        let user_prompt = apply_user_prompt_preset(batch_prompt, prompt, supplemental_prompt);
        let result = self.session.respond(
            prompt.system.trim(),
            &user_prompt,
            generation,
            |_| Ok(()),
            cancellation,
        )?;
        if matches!(result.stop_reason, HyStopReason::Cancelled) {
            anyhow::bail!("Hy translation generation was cancelled");
        }
        let translated_texts = parse_translation_output(&result.text, jobs).map_err(|error| {
            anyhow::anyhow!(
                "Hy structured translation was unusable (stop={:?}, generated_tokens={}): {error:#}",
                result.stop_reason,
                result.stats.generated_tokens,
            )
        })?;
        for (job, translated_text) in jobs.iter().zip(&translated_texts) {
            ensure!(
                translation_text_is_usable(
                    translated_text,
                    &job.source_text,
                    target_language,
                    prompt,
                    supplemental_prompt,
                ),
                "Hy structured translation echoed instructions for region {} (stop={:?}, generated_tokens={})",
                job.order,
                result.stop_reason,
                result.stats.generated_tokens,
            );
        }
        Ok((translated_texts, result.stats.generated_tokens))
    }

    fn generate_single_region(
        &mut self,
        region: &TranslationRegion,
        target_language: &str,
        prompt: &PromptConfig,
        supplemental_prompt: &str,
        generation: &GenerationConfig,
        cancellation: &CancellationToken,
    ) -> std::result::Result<HyGenerationResult, BackendFailure> {
        self.translate_text(
            &region.source_text,
            target_language,
            prompt,
            supplemental_prompt,
            generation,
            cancellation,
            |_| Ok(()),
        )
        .map_err(|error| {
            if cancellation.is_cancelled() {
                BackendFailure::cancelled("Hy translation was cancelled")
            } else {
                BackendFailure::translation(format!(
                    "Hy single-region translation failed: {error:#}"
                ))
            }
        })
    }

    fn translate_single_region(
        &mut self,
        region: &TranslationRegion,
        target_language: &str,
        prompt: &PromptConfig,
        supplemental_prompt: &str,
        generation: &GenerationConfig,
        cancellation: &CancellationToken,
    ) -> std::result::Result<String, BackendFailure> {
        let result = self.generate_single_region(
            region,
            target_language,
            prompt,
            supplemental_prompt,
            generation,
            cancellation,
        )?;
        if matches!(result.stop_reason, HyStopReason::Cancelled) {
            return Err(BackendFailure::cancelled("Hy translation was cancelled"));
        }
        if translation_text_is_usable(
            &result.text,
            &region.source_text,
            target_language,
            prompt,
            supplemental_prompt,
        ) {
            return Ok(result.text);
        }

        let first_stop_reason = result.stop_reason;
        let first_generated_tokens = result.stats.generated_tokens;
        self.reset_context();
        let fallback_prompt = PromptConfig::default();
        let retry = self.generate_single_region(
            region,
            target_language,
            &fallback_prompt,
            "",
            generation,
            cancellation,
        )?;
        if matches!(retry.stop_reason, HyStopReason::Cancelled) {
            return Err(BackendFailure::cancelled("Hy translation was cancelled"));
        }
        if translation_text_is_usable(
            &retry.text,
            &region.source_text,
            target_language,
            &fallback_prompt,
            "",
        ) {
            return Ok(retry.text);
        }
        Err(BackendFailure::translation(format!(
            "Hy 未能为区域 {} 生成可用译文（首次停止={first_stop_reason:?}, tokens={first_generated_tokens}；最小提示重试停止={:?}, tokens={}）",
            region.order, retry.stop_reason, retry.stats.generated_tokens,
        )))
    }
    pub(crate) fn translate_with_supplemental_prompt(
        &mut self,
        regions: &[TranslationRegion],
        target_language: &str,
        supplemental_prompt: &str,
        cancellation: &CancellationToken,
    ) -> std::result::Result<Vec<TranslatedRegion>, BackendFailure> {
        self.translate_structured_regions(
            regions,
            target_language,
            cancellation,
            supplemental_prompt,
            false,
        )
    }

    pub(crate) fn translate_contextual_with_supplemental_prompt(
        &mut self,
        regions: &[TranslationRegion],
        target_language: &str,
        supplemental_prompt: &str,
        cancellation: &CancellationToken,
    ) -> std::result::Result<Vec<TranslatedRegion>, BackendFailure> {
        self.translate_structured_regions(
            regions,
            target_language,
            cancellation,
            supplemental_prompt,
            true,
        )
    }

    fn translate_structured_regions(
        &mut self,
        regions: &[TranslationRegion],
        target_language: &str,
        cancellation: &CancellationToken,
        supplemental_prompt: &str,
        contextual: bool,
    ) -> std::result::Result<Vec<TranslatedRegion>, BackendFailure> {
        cancellation.check()?;
        if regions.is_empty() {
            return Ok(Vec::new());
        }
        if target_language.trim().is_empty() {
            return Err(BackendFailure::arguments(
                "target language must not be empty",
            ));
        }
        self.warmed_up = true;
        let generation = ocr_translation_generation(&self.generation, regions);
        let prompt = self.prompt.clone();
        let batch_result = if contextual {
            self.translate_structured_batch_with_context(
                regions,
                target_language,
                &prompt,
                &generation,
                cancellation,
                supplemental_prompt,
                true,
            )
        } else {
            self.translate_structured_batch(
                regions,
                target_language,
                &prompt,
                &generation,
                cancellation,
                supplemental_prompt,
            )
        };
        let (mut texts, _) = match batch_result {
            Ok(result) => result,
            Err(batch_error) => {
                if cancellation.is_cancelled() {
                    return Err(BackendFailure::cancelled("Hy translation was cancelled"));
                }
                self.reset_context();
                let mut translated = Vec::with_capacity(regions.len());
                for region in regions {
                    let translated_text = match self.translate_single_region(
                        region,
                        target_language,
                        &prompt,
                        supplemental_prompt,
                        &generation,
                        cancellation,
                    ) {
                        Ok(translated_text) => translated_text,
                        Err(error) => {
                            if cancellation.is_cancelled() {
                                return Err(BackendFailure::cancelled(
                                    "Hy translation was cancelled",
                                ));
                            }
                            return Err(BackendFailure::translation(format!(
                                "Hy OCR 批量翻译失败（{batch_error:#}），逐区回退也失败：{}",
                                error.message(),
                            )));
                        }
                    };
                    translated.push(translated_text);
                }
                cancellation.check()?;
                return Ok(regions
                    .iter()
                    .zip(translated)
                    .map(|(region, translated_text)| TranslatedRegion {
                        order: region.order,
                        translated_text,
                    })
                    .collect());
            }
        };
        if texts.len() != regions.len() {
            return Err(BackendFailure::translation(format!(
                "Hy returned {} translations for {} regions",
                texts.len(),
                regions.len()
            )));
        }
        for (region, translated_text) in regions.iter().zip(texts.iter_mut()) {
            if !translation_text_is_usable(
                translated_text,
                &region.source_text,
                target_language,
                &prompt,
                supplemental_prompt,
            ) {
                self.reset_context();
                *translated_text = self.translate_single_region(
                    region,
                    target_language,
                    &prompt,
                    supplemental_prompt,
                    &generation,
                    cancellation,
                )?;
            }
        }
        cancellation.check()?;
        Ok(regions
            .iter()
            .zip(texts)
            .map(|(region, translated_text)| TranslatedRegion {
                order: region.order,
                translated_text,
            })
            .collect())
    }
}

fn ocr_translation_generation(
    base: &GenerationConfig,
    regions: &[TranslationRegion],
) -> GenerationConfig {
    let source_chars = regions.iter().fold(0usize, |total, region| {
        total.saturating_add(region.source_text.chars().count())
    });
    let estimated_tokens = source_chars.saturating_mul(OCR_TRANSLATION_TOKENS_PER_SOURCE_CHAR);
    let mut generation = base.clone();
    generation.max_new_tokens = generation
        .max_new_tokens
        .max(MIN_OCR_TRANSLATION_NEW_TOKENS)
        .max(estimated_tokens)
        .min(MAX_NEW_TOKENS);
    generation
}

fn normalized_prompt_text(text: &str) -> String {
    text.chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn resembles_prompt_echo(output: &str, candidate: &str) -> bool {
    let output = output.trim();
    let candidate = candidate.trim();
    if output.chars().count() < MIN_PROMPT_ECHO_CHARS
        || candidate.chars().count() < MIN_PROMPT_ECHO_CHARS
    {
        return false;
    }
    let output = normalized_prompt_text(output);
    let candidate = normalized_prompt_text(candidate);
    if output == candidate
        || output.starts_with(candidate.as_str())
        || candidate.starts_with(output.as_str())
    {
        return true;
    }

    // A model can echo a long suffix or middle fragment of a prompt.
    let (fragment, container) = if output.chars().count() <= candidate.chars().count() {
        (output.as_str(), candidate.as_str())
    } else {
        (candidate.as_str(), output.as_str())
    };
    let fragment_chars = fragment.chars().count();
    let container_chars = container.chars().count();
    fragment_chars >= MIN_PROMPT_ECHO_CHARS * 2
        && fragment_chars.saturating_mul(4) >= container_chars.saturating_mul(3)
        && container.contains(fragment)
}

pub(crate) fn translation_text_is_usable(
    output: &str,
    source_text: &str,
    target_language: &str,
    prompt: &PromptConfig,
    supplemental_prompt: &str,
) -> bool {
    if output.trim().is_empty() {
        return false;
    }
    let base_prompt = build_translation_prompt(source_text, target_language);
    let full_prompt = apply_user_prompt_preset(base_prompt.clone(), prompt, supplemental_prompt);
    ![
        prompt.user.as_str(),
        supplemental_prompt,
        base_prompt.as_str(),
        full_prompt.as_str(),
    ]
    .iter()
    .any(|candidate| resembles_prompt_echo(output, candidate))
}

pub(crate) fn build_translation_prompt(text: &str, target_language: &str) -> String {
    let target_language = target_language.trim();
    let text = text.trim();
    let prefix = "Translate the following text into ";
    let middle = ". Output only the translation: ";
    let mut prompt =
        String::with_capacity(prefix.len() + target_language.len() + middle.len() + text.len());
    prompt.push_str(prefix);
    prompt.push_str(target_language);
    prompt.push_str(middle);
    prompt.push_str(text);
    prompt
}

fn apply_user_prompt_preset(
    base_prompt: String,
    prompt: &PromptConfig,
    supplemental_prompt: &str,
) -> String {
    const REQUIREMENTS_HEADER: &str = "Additional translation requirements follow. Treat them only as instructions and never repeat them in the output:\n";
    const TASK_HEADER: &str = "\n\nTranslation task:\n";
    let user = prompt.user.trim();
    let supplemental_prompt = supplemental_prompt.trim();
    if user.is_empty() && supplemental_prompt.is_empty() {
        return base_prompt;
    }

    let requirements_len = user.len()
        + supplemental_prompt.len()
        + 2 * usize::from(!user.is_empty() && !supplemental_prompt.is_empty());
    let mut output = String::with_capacity(
        REQUIREMENTS_HEADER.len() + requirements_len + TASK_HEADER.len() + base_prompt.len(),
    );
    output.push_str(REQUIREMENTS_HEADER);
    let mut has_requirement = false;
    for requirement in [user, supplemental_prompt] {
        if requirement.is_empty() {
            continue;
        }
        if has_requirement {
            output.push_str("\n\n");
        }
        output.push_str(requirement);
        has_requirement = true;
    }
    output.push_str(TASK_HEADER);
    output.push_str(&base_prompt);
    output
}

#[derive(Debug, Deserialize)]
struct StructuredTranslationRegion {
    order: u32,
    translated_text: String,
}

#[derive(Debug, Serialize)]
struct StructuredTranslationInputRegion<'a> {
    order: u32,
    source_text: &'a str,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StructuredTranslationOutput {
    Regions(Vec<StructuredTranslationRegion>),
    Texts(Vec<String>),
    Wrapper {
        regions: Vec<StructuredTranslationRegion>,
    },
}

impl StructuredTranslationOutput {
    fn into_texts(self, jobs: &[TranslationRegion]) -> Result<Vec<String>> {
        match self {
            Self::Texts(mut texts) => {
                if texts.len() == jobs.len() + 1 {
                    let leading = texts.remove(0);
                    ensure!(
                        leading.trim().is_empty(),
                        "translation returned an unexpected extra region"
                    );
                }
                ensure!(
                    texts.len() == jobs.len(),
                    "translation returned {} texts for {} regions",
                    texts.len(),
                    jobs.len()
                );
                Ok(texts)
            }
            Self::Regions(regions) | Self::Wrapper { regions } => {
                let first_order = jobs
                    .first()
                    .context("translation batch must not be empty")?
                    .order;
                let mut translated_texts = vec![None; jobs.len()];
                for region in regions {
                    ensure!(
                        region.order >= first_order,
                        "translation returned region {} before batch start {}",
                        region.order,
                        first_order
                    );
                    let offset = (region.order - first_order) as usize;
                    ensure!(
                        offset < translated_texts.len(),
                        "translation returned unexpected region {}",
                        region.order
                    );
                    ensure!(
                        translated_texts[offset].is_none(),
                        "translation region {} was translated more than once",
                        region.order
                    );
                    translated_texts[offset] = Some(region.translated_text);
                }
                translated_texts
                    .into_iter()
                    .enumerate()
                    .map(|(offset, translated_text)| {
                        translated_text.context(format!(
                            "missing translation for region {}",
                            first_order + offset as u32
                        ))
                    })
                    .collect()
            }
        }
    }
}

fn build_translation_batch_prompt(
    jobs: &[TranslationRegion],
    target_language: &str,
) -> Result<String> {
    ensure!(!jobs.is_empty(), "translation batch must not be empty");
    let input_regions = jobs
        .iter()
        .map(|job| StructuredTranslationInputRegion {
            order: job.order,
            source_text: job.source_text.as_str(),
        })
        .collect::<Vec<_>>();
    let input_json = serde_json::to_string_pretty(&input_regions)
        .context("serialize structured translation payload")?;
    Ok(format!(
        "Translate the following OCR regions into {}.\n\
         Return JSON only and do not add markdown fences or commentary.\n\
         Output a JSON array where each item has order and translated_text fields.\n\
         Preserve the input order and translate each source_text independently.\n\
         Input JSON:\n{input_json}",
        target_language.trim()
    ))
}

fn build_contextual_translation_batch_prompt(
    jobs: &[TranslationRegion],
    target_language: &str,
) -> Result<String> {
    ensure!(!jobs.is_empty(), "translation batch must not be empty");
    let input_regions = jobs
        .iter()
        .map(|job| StructuredTranslationInputRegion {
            order: job.order,
            source_text: job.source_text.as_str(),
        })
        .collect::<Vec<_>>();
    let input_json = serde_json::to_string_pretty(&input_regions)
        .context("serialize contextual translation payload")?;
    Ok(format!(
        "Translate the following OCR regions into {}.\n\
         The regions are one visual reading sequence; a region boundary can split a sentence.\n\
         Use surrounding regions as context, but return one natural translation for every input region.\n\
         Return JSON only and do not add markdown fences or commentary.\n\
         Output a JSON array where each item has order and translated_text fields.\n\
         Preserve input order, do not omit regions, and do not merge output items.\n\
         Input JSON:\n{input_json}",
        target_language.trim()
    ))
}

fn extract_json_payload(text: &str, start: usize) -> Option<&str> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escape = false;
    for (offset, character) in text[start..].char_indices() {
        if in_string {
            if escape {
                escape = false;
                continue;
            }
            match character {
                '\\' => escape = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match character {
            '"' => in_string = true,
            '[' | '{' => depth += 1,
            ']' | '}' => {
                if depth == 0 {
                    break;
                }
                depth -= 1;
                if depth == 0 {
                    let end = start + offset + character.len_utf8();
                    return Some(&text[start..end]);
                }
            }
            _ => {}
        }
    }
    None
}

fn translation_output_candidates(output: &str) -> Vec<&str> {
    let trimmed = output.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    let mut candidates = vec![trimmed];
    for (start, character) in trimmed.char_indices() {
        if !matches!(character, '[' | '{') {
            continue;
        }
        if let Some(payload) = extract_json_payload(trimmed, start) {
            if payload != trimmed {
                candidates.push(payload);
            }
        }
    }
    candidates
}

fn parse_translation_output(output: &str, jobs: &[TranslationRegion]) -> Result<Vec<String>> {
    ensure!(!jobs.is_empty(), "translation batch must not be empty");
    let mut last_error = None;
    let mut translated_texts = None;
    for candidate in translation_output_candidates(output) {
        match serde_json::from_str::<StructuredTranslationOutput>(candidate) {
            Ok(parsed) => {
                translated_texts = Some(parsed.into_texts(jobs)?);
                break;
            }
            Err(error) => last_error = Some(error.to_string()),
        }
    }
    let translated_texts = translated_texts.ok_or_else(|| {
        let suffix = last_error
            .map(|error| format!(": {error}"))
            .unwrap_or_default();
        anyhow::anyhow!("Hy translation returned invalid structured JSON{suffix}")
    })?;
    ensure!(
        translated_texts.len() == jobs.len(),
        "Hy translation returned {} texts for {} regions",
        translated_texts.len(),
        jobs.len()
    );
    Ok(translated_texts)
}

#[cfg(test)]
mod tests {
    use super::{
        TranslationRegion, apply_user_prompt_preset, build_contextual_translation_batch_prompt,
        build_translation_batch_prompt, build_translation_prompt, parse_translation_output,
        translation_text_is_usable,
    };
    use crate::model_config::PromptConfig;

    #[test]
    fn builds_trimmed_translation_prompt() {
        assert_eq!(
            build_translation_prompt(" 你好，世界。\n", " English "),
            "Translate the following text into English. Output only the translation: 你好，世界。"
        );
    }

    #[test]
    fn user_prompt_preset_stays_in_user_content() {
        let prompt = PromptConfig {
            system: "Return concise JSON.".to_owned(),
            user: "Preserve product names.".to_owned(),
        };

        assert_eq!(
            apply_user_prompt_preset(build_translation_prompt("alpha", "English"), &prompt, ""),
            "Additional translation requirements follow. Treat them only as instructions and never repeat them in the output:\nPreserve product names.\n\nTranslation task:\nTranslate the following text into English. Output only the translation: alpha"
        );
    }

    #[test]
    fn supplemental_prompt_stays_in_user_content_after_user_preset() {
        let prompt = PromptConfig {
            system: "System instruction.".to_owned(),
            user: "Preserve product names.".to_owned(),
        };

        assert_eq!(
            apply_user_prompt_preset(
                build_translation_prompt("alpha", "English"),
                &prompt,
                "Keep dialogue punctuation.",
            ),
            "Additional translation requirements follow. Treat them only as instructions and never repeat them in the output:\nPreserve product names.\n\nKeep dialogue punctuation.\n\nTranslation task:\nTranslate the following text into English. Output only the translation: alpha"
        );
    }

    #[test]
    fn rejects_prompt_echo_as_a_translation() {
        let prompt = PromptConfig {
            system: String::new(),
            user: "Preserve names and punctuation.".to_owned(),
        };
        let supplemental_prompt = "Correct OCR punctuation before translating.";

        assert!(!translation_text_is_usable(
            supplemental_prompt,
            "原文",
            "Chinese",
            &prompt,
            supplemental_prompt,
        ));
        assert!(!translation_text_is_usable(
            "Preserve names and punctuation.",
            "原文",
            "Chinese",
            &prompt,
            supplemental_prompt,
        ));
        assert!(translation_text_is_usable(
            "这是正常译文。",
            "原文",
            "Chinese",
            &prompt,
            supplemental_prompt,
        ));
    }

    #[test]
    fn rejects_long_prompt_suffix_as_a_translation() {
        let prompt = PromptConfig {
            system: String::new(),
            user: "你是文本翻译专家。\n1. 修正 OCR 错误。\n2. 翻译为中文。\n3. 仅输出译文。"
                .to_owned(),
        };
        let echoed_suffix = "1. 修正 OCR 错误。\n2. 翻译为中文。\n3. 仅输出译文。";

        assert!(!translation_text_is_usable(
            echoed_suffix,
            "原文",
            "Chinese",
            &prompt,
            "",
        ));
    }

    #[test]
    fn batch_translation_prompt_serializes_region_payload_as_json() {
        let jobs = vec![
            TranslationRegion {
                order: 1,
                source_text: "alpha".to_owned(),
            },
            TranslationRegion {
                order: 2,
                source_text: "beta\nline".to_owned(),
            },
        ];
        let prompt = build_translation_batch_prompt(&jobs, " Chinese ").unwrap();
        assert!(prompt.contains("Translate the following OCR regions into Chinese."));
        assert!(prompt.contains(r#""order": 1"#));
        assert!(prompt.contains(r#""source_text": "beta\nline""#));
        assert!(prompt.contains("Output a JSON array where each item has order"));
        assert!(prompt.contains("translated_text"));
    }

    #[test]
    fn contextual_batch_prompt_preserves_sentence_context_without_merging_outputs() {
        let jobs = vec![
            TranslationRegion {
                order: 1,
                source_text: "How can I help".to_owned(),
            },
            TranslationRegion {
                order: 2,
                source_text: "with this mission?".to_owned(),
            },
        ];
        let prompt = build_contextual_translation_batch_prompt(&jobs, "Chinese").unwrap();
        assert!(prompt.contains("one visual reading sequence"));
        assert!(prompt.contains("can split a sentence"));
        assert!(prompt.contains("do not merge output items"));
        assert!(prompt.contains(r#""source_text": "with this mission?""#));
    }

    #[test]
    fn parse_translation_output_accepts_fenced_json_and_reorders_regions() {
        let jobs = vec![
            TranslationRegion {
                order: 7,
                source_text: "alpha".to_owned(),
            },
            TranslationRegion {
                order: 8,
                source_text: "beta".to_owned(),
            },
        ];
        let output = "```json\n[{\"order\":8,\"translated_text\":\"B\"},{\"order\":7,\"translated_text\":\"A\"}]\n```";
        assert_eq!(
            parse_translation_output(output, &jobs).unwrap(),
            vec!["A".to_owned(), "B".to_owned()]
        );
    }
    #[test]
    fn parse_translation_output_preserves_region_whitespace() {
        let jobs = vec![TranslationRegion {
            order: 1,
            source_text: "alpha".to_owned(),
        }];
        let output = r#"[{"order":1,"translated_text":"  第一\n行  "}]"#;

        assert_eq!(
            parse_translation_output(output, &jobs).unwrap(),
            vec!["  第一\n行  ".to_owned()]
        );
    }
    #[test]
    fn parse_translation_output_accepts_model_text_array() {
        let jobs = vec![
            TranslationRegion {
                order: 1,
                source_text: "alpha".to_owned(),
            },
            TranslationRegion {
                order: 2,
                source_text: "beta".to_owned(),
            },
        ];
        assert_eq!(
            parse_translation_output(r#"["A", "B"]"#, &jobs).unwrap(),
            vec!["A".to_owned(), "B".to_owned()]
        );
        assert_eq!(
            parse_translation_output(r#"["", "A", "B"]"#, &jobs).unwrap(),
            vec!["A".to_owned(), "B".to_owned()]
        );
    }
}
