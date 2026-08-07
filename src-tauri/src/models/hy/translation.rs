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
        contracts::{HyPort, TranslatedRegion, TranslationRegion},
        failure::BackendFailure,
    },
    model_config::{GenerationConfig, MemoryConfig},
    model_support::CancellationToken,
};
use anyhow::{Context, Result, ensure};
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Loaded Hy translation session. The concrete GGUF/session state remains
/// private to this provider and only the neutral HyPort is exposed.
pub(crate) struct HyTranslator {
    session: HySessionDriver,
    generation: GenerationConfig,
    system_prompt: Option<String>,
}
pub(crate) fn load(
    model_path: &Path,
    device: &Device,
    memory: MemoryConfig,
) -> Result<HyTranslator> {
    load_with_config(
        model_path,
        device,
        memory,
        GenerationConfig::default(),
        None,
    )
}

pub(crate) fn load_with_config(
    model_path: &Path,
    device: &Device,
    memory: MemoryConfig,
    generation: GenerationConfig,
    system_prompt: Option<String>,
) -> Result<HyTranslator> {
    let assets =
        HyAssets::preflight(model_path).map_err(|error| anyhow::anyhow!(error.to_string()))?;
    ensure!(device.is_cuda(), "Hy requires a CUDA device");
    Ok(HyTranslator {
        session: HySessionDriver::new(&assets.model, device, memory)?,
        generation,
        system_prompt,
    })
}

pub(crate) fn load_port(
    model_path: &Path,
    device: &Device,
    memory: MemoryConfig,
    generation: GenerationConfig,
    system_prompt: Option<String>,
) -> std::result::Result<HyTranslator, BackendFailure> {
    if !device.is_cuda() {
        return Err(BackendFailure::device("Hy requires a CUDA device"));
    }
    let assets = HyAssets::preflight(model_path)?;
    HySessionDriver::new(&assets.model, device, memory)
        .map(|session| HyTranslator {
            session,
            generation,
            system_prompt,
        })
        .map_err(|error| {
            let message = error.to_string();
            if message.contains("flash-attn") || message.contains("CUDA") {
                BackendFailure::device(format!("initialize Hy device path: {message}"))
            } else {
                BackendFailure::translation(format!("load Hy GGUF model: {message}"))
            }
        })
}

impl HyTranslator {
    /// Translate plain neutral text using the Hy translation prompt contract.
    pub(crate) fn translate_text(
        &mut self,
        text: &str,
        target_language: &str,
        system_prompt: Option<&str>,
        generation: &GenerationConfig,
        cancellation: &CancellationToken,
        on_chunk: impl FnMut(&str) -> Result<()>,
    ) -> Result<HyGenerationResult> {
        let mut prompt = build_translation_prompt(text, target_language);
        if let Some(system_prompt) = system_prompt {
            let system_prompt = system_prompt.trim();
            if !system_prompt.is_empty() {
                prompt = format!("{system_prompt}\n\n{prompt}");
            }
        }
        self.session
            .respond(&prompt, generation, on_chunk, cancellation)
    }

    /// Translate one neutral structured batch and return texts in input order.
    pub(crate) fn translate_structured_batch(
        &mut self,
        jobs: &[TranslationRegion],
        target_language: &str,
        system_prompt: Option<&str>,
        generation: &GenerationConfig,
        cancellation: &CancellationToken,
    ) -> Result<(Vec<String>, usize)> {
        ensure!(!jobs.is_empty(), "translation batch must not be empty");
        let mut prompt = build_translation_batch_prompt(jobs, target_language)?;
        if let Some(system_prompt) = system_prompt {
            let system_prompt = system_prompt.trim();
            if !system_prompt.is_empty() {
                prompt = format!("{system_prompt}\n\n{prompt}");
            }
        }
        let mut output = String::new();
        let result = self.session.respond(
            &prompt,
            generation,
            |chunk| {
                output.push_str(chunk);
                Ok(())
            },
            cancellation,
        )?;
        if matches!(result.stop_reason, HyStopReason::Cancelled) {
            anyhow::bail!("Hy translation generation was cancelled");
        }
        let translated_texts = parse_translation_output(&output, jobs)?;
        Ok((translated_texts, result.stats.generated_tokens))
    }
    fn translate_single_region(
        &mut self,
        region: &TranslationRegion,
        target_language: &str,
        system_prompt: Option<&str>,
        generation: &GenerationConfig,
        cancellation: &CancellationToken,
    ) -> std::result::Result<String, BackendFailure> {
        let result = self
            .translate_text(
                &region.source_text,
                target_language,
                system_prompt,
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
            })?;
        if matches!(result.stop_reason, HyStopReason::Cancelled) {
            return Err(BackendFailure::cancelled("Hy translation was cancelled"));
        }
        let translated_text = result.text.trim().to_owned();
        if translated_text.is_empty() {
            return Err(BackendFailure::translation(format!(
                "Hy returned empty translation for region {}",
                region.order
            )));
        }
        Ok(translated_text)
    }
}

impl HyPort for HyTranslator {
    fn translate(
        &mut self,
        regions: &[TranslationRegion],
        target_language: &str,
        cancellation: &CancellationToken,
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
        let generation = self.generation.clone();
        let system_prompt = self.system_prompt.clone();
        let batch_result = self.translate_structured_batch(
            regions,
            target_language,
            system_prompt.as_deref(),
            &generation,
            cancellation,
        );
        let (mut texts, _) = match batch_result {
            Ok(result) => result,
            Err(_error) => {
                if cancellation.is_cancelled() {
                    return Err(BackendFailure::cancelled("Hy translation was cancelled"));
                }
                let mut translated = Vec::with_capacity(regions.len());
                for region in regions {
                    translated.push(self.translate_single_region(
                        region,
                        target_language,
                        system_prompt.as_deref(),
                        &generation,
                        cancellation,
                    )?);
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
            if translated_text.trim().is_empty() {
                *translated_text = self.translate_single_region(
                    region,
                    target_language,
                    system_prompt.as_deref(),
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

    fn loaded(&self) -> bool {
        true
    }
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
    Ok(translated_texts
        .into_iter()
        .map(|translated_text| translated_text.trim().to_owned())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{
        TranslationRegion, build_translation_batch_prompt, build_translation_prompt,
        parse_translation_output,
    };

    #[test]
    fn builds_trimmed_translation_prompt() {
        assert_eq!(
            build_translation_prompt(" 你好，世界。\n", " English "),
            "Translate the following text into English. Output only the translation: 你好，世界。"
        );
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
