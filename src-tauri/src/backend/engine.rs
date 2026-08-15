use crate::{
    backend::{
        contracts::{OcrOutput, OcrPort, OutputPort, RegionRecord, TranslationOutput},
        failure::BackendFailure,
        input::DecodedImage,
        settings::{BackendSettings, DeviceKind},
    },
    model_config::{GenerationConfig, MAX_NEW_TOKENS, MemoryConfig, ModelConfig, PromptConfig},
    model_support::CancellationToken,
    models::{hy, ppocr::PpOcrProvider},
    output::ImageOutput,
};
use candle_core::Device as CandleDevice;

#[cfg(feature = "cuda")]
const MIB: usize = 1024 * 1024;
const GPU_RESIDENT_TOTAL_MIB: usize = 8 * 1024;
const GPU_RESIDENT_FREE_MIB: usize = 4 * 1024;
const GPU_BALANCED_FREE_MIB: usize = 2_500;
const LIVE_TRANSLATION_MIN_NEW_TOKENS: usize = 2_048;
const LIVE_TRANSLATION_TOKENS_PER_SOURCE_CHAR: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GpuExecutionPolicy {
    Cpu,
    Resident,
    Balanced,
    Constrained,
}

impl GpuExecutionPolicy {
    fn for_memory(total_mib: usize, free_mib: usize) -> Self {
        if total_mib >= GPU_RESIDENT_TOTAL_MIB && free_mib >= GPU_RESIDENT_FREE_MIB {
            Self::Resident
        } else if free_mib >= GPU_BALANCED_FREE_MIB {
            Self::Balanced
        } else {
            Self::Constrained
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Resident => "gpu_resident",
            Self::Balanced => "gpu_balanced",
            Self::Constrained => "gpu_constrained",
        }
    }

    fn region_parallelism(self, configured: usize) -> usize {
        match self {
            Self::Cpu | Self::Resident => configured,
            Self::Balanced => configured.min(8),
            Self::Constrained => configured.min(4),
        }
        .max(1)
    }

    fn recognizer_batch_pixels(self) -> usize {
        match self {
            Self::Cpu => 48 * 3200 * 4,
            Self::Resident => 48 * 3200 * 16,
            Self::Balanced => 48 * 3200 * 8,
            Self::Constrained => 48 * 3200 * 4,
        }
    }

    fn keeps_models_resident(self) -> bool {
        matches!(self, Self::Resident)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GpuResourceInfo {
    pub(crate) name: String,
    pub(crate) total_memory_mib: u64,
    pub(crate) free_memory_mib: u64,
    pub(crate) execution_mode: &'static str,
}

pub(crate) struct BackendEngine {
    pub(crate) settings: BackendSettings,
    device: CandleDevice,
    gpu_policy: GpuExecutionPolicy,
    ocr: Option<PpOcrProvider>,
    hy: Option<hy::HyTranslator>,
    translator_memory: Option<MemoryConfig>,
    output: ImageOutput,
}

impl BackendEngine {
    pub(crate) fn new(mut settings: BackendSettings) -> Result<Self, BackendFailure> {
        let device = create_device(settings.device_kind)?;
        let fallback_policy = if settings.device_kind == DeviceKind::Cuda {
            GpuExecutionPolicy::Balanced
        } else {
            GpuExecutionPolicy::Cpu
        };
        let gpu_policy = query_gpu_memory(&device)
            .ok()
            .flatten()
            .map(|(_, total_mib, free_mib)| GpuExecutionPolicy::for_memory(total_mib, free_mib))
            .unwrap_or(fallback_policy);
        settings.region_parallelism = gpu_policy.region_parallelism(settings.region_parallelism);
        Ok(Self {
            output: ImageOutput::new(settings.font_path.clone()),
            settings,
            device,
            gpu_policy,
            ocr: None,
            hy: None,
            translator_memory: None,
        })
    }

    pub(crate) fn ocr_loaded(&self) -> bool {
        self.ocr.is_some()
    }

    pub(crate) fn translator_loaded(&self) -> bool {
        self.hy.is_some()
    }

    pub(crate) fn model_states(&self) -> (bool, bool) {
        (self.ocr_loaded(), self.translator_loaded())
    }

    pub(crate) fn load_ocr(&mut self) -> Result<(), BackendFailure> {
        if self.ocr.is_none() {
            if !self.gpu_policy.keeps_models_resident() {
                self.hy = None;
                self.translator_memory = None;
            }
            self.ocr = Some(PpOcrProvider::load(
                &self.settings.detector_model_dir,
                &self.settings.recognizer_model_dir,
                &self.device,
                self.settings.region_parallelism,
                self.gpu_policy.recognizer_batch_pixels(),
            )?);
        }
        Ok(())
    }

    pub(crate) fn load_translator(&mut self, target_language: &str) -> Result<(), BackendFailure> {
        self.load_translator_with_memory(target_language, self.settings.memory.clone())
    }

    fn load_translator_with_memory(
        &mut self,
        target_language: &str,
        memory: MemoryConfig,
    ) -> Result<(), BackendFailure> {
        let config = ModelConfig::from_parts(
            target_language,
            self.settings.prompt.clone(),
            self.settings.generation.clone(),
            memory.clone(),
        )
        .map_err(|error| BackendFailure::arguments(format!("invalid model config: {error:#}")))?;
        if self.hy.is_some() && self.translator_memory.as_ref() != Some(&memory) {
            self.hy = None;
            self.translator_memory = None;
        }
        if self.hy.is_none() {
            if !self.gpu_policy.keeps_models_resident() {
                self.ocr = None;
            }
            let translator = hy::load_with_config(
                &self.settings.hy_model,
                &self.device,
                config.memory,
                config.generation,
                config.prompt,
            )
            .map_err(|error| {
                BackendFailure::asset(format!("load local Hy-MT2 model: {error:#}"))
            })?;
            self.hy = Some(translator);
            self.translator_memory = Some(memory);
        }
        Ok(())
    }

    pub(crate) fn reset_translator_context(&mut self) {
        if let Some(translator) = self.hy.as_mut() {
            translator.reset_context();
        }
    }

    pub(crate) fn prepare_live_pipeline(
        &mut self,
        target_language: &str,
        memory: MemoryConfig,
        cancellation: &CancellationToken,
    ) -> Result<(), BackendFailure> {
        self.load_ocr()?;
        self.ocr
            .as_mut()
            .ok_or_else(|| BackendFailure::internal("PP-OCR provider was not initialized"))?
            .warm_up(cancellation)?;
        self.load_translator_with_memory(target_language, memory)?;
        self.hy
            .as_mut()
            .ok_or_else(|| BackendFailure::internal("Hy translator was not initialized"))?
            .warm_up(cancellation)
            .map_err(|error| {
                BackendFailure::translation(format!("warm up Hy CUDA kernels: {error:#}"))
            })?;
        self.reset_translator_context();
        Ok(())
    }

    pub(crate) fn gpu_resource_info(&self) -> Result<Option<GpuResourceInfo>, BackendFailure> {
        let resources = query_gpu_memory(&self.device).ok().flatten();
        Ok(resources
            .map(|(name, total_mib, free_mib)| GpuResourceInfo {
                name,
                total_memory_mib: u64::try_from(total_mib).unwrap_or(u64::MAX),
                free_memory_mib: u64::try_from(free_mib).unwrap_or(u64::MAX),
                execution_mode: self.gpu_policy.label(),
            })
            .or_else(|| {
                (self.settings.device_kind == DeviceKind::Cuda).then(|| GpuResourceInfo {
                    name: "CUDA".to_owned(),
                    total_memory_mib: 0,
                    free_memory_mib: 0,
                    execution_mode: self.gpu_policy.label(),
                })
            }))
    }

    pub(crate) fn unload_ocr(&mut self) {
        self.ocr = None;
    }

    pub(crate) fn unload_translator(&mut self) {
        self.hy = None;
        self.translator_memory = None;
    }

    pub(crate) fn unload_models(&mut self) {
        self.unload_ocr();
        self.unload_translator();
    }

    pub(crate) fn recognize_regions(
        &mut self,
        image: &DecodedImage,
        cancellation: &CancellationToken,
    ) -> Result<Vec<crate::backend::contracts::RegionRecord>, BackendFailure> {
        cancellation.check()?;
        self.load_ocr()?;
        let ocr = self
            .ocr
            .as_mut()
            .ok_or_else(|| BackendFailure::internal("PP-OCR provider was not initialized"))?;
        let document = ocr.recognize(image, cancellation)?;
        cancellation.check()?;
        if document.regions.len() > crate::backend::input::MAX_REGIONS {
            return Err(BackendFailure::ocr(
                "OCR region count exceeds the supported bound",
            ));
        }
        Ok(document.regions)
    }

    pub(crate) fn translate(
        &mut self,
        image: &DecodedImage,
        cancellation: &CancellationToken,
        mut report_progress: impl FnMut(u8, &'static str),
    ) -> Result<TranslationOutput, BackendFailure> {
        cancellation.check()?;
        report_progress(20, "正在准备 PP-OCR");
        self.load_ocr()?;
        let ocr = self
            .ocr
            .as_mut()
            .ok_or_else(|| BackendFailure::internal("PP-OCR provider was not initialized"))?;
        let document = ocr.recognize(image, cancellation)?;
        cancellation.check()?;
        report_progress(55, "OCR 识别完成");
        if document.regions.len() > crate::backend::input::MAX_REGIONS {
            return Err(BackendFailure::ocr(
                "OCR region count exceeds the supported bound",
            ));
        }

        let mut records = document.regions;
        if records.is_empty() {
            let output =
                self.output
                    .render(image, &records, image.target_language(), cancellation)?;
            report_progress(100, "处理完成");
            return Ok(output);
        }

        self.translate_regions_with_progress(
            &mut records,
            image.target_language(),
            cancellation,
            false,
            "",
            &mut report_progress,
        )?;
        cancellation.check()?;
        report_progress(90, "翻译完成，正在生成标注图");
        let output = self
            .output
            .render(image, &records, image.target_language(), cancellation)?;
        report_progress(100, "翻译完成");
        Ok(output)
    }

    pub(crate) fn translate_regions(
        &mut self,
        records: &mut [RegionRecord],
        target_language: &str,
        cancellation: &CancellationToken,
    ) -> Result<(), BackendFailure> {
        self.translate_regions_with_progress(
            records,
            target_language,
            cancellation,
            false,
            "",
            |_, _| {},
        )
    }

    pub(crate) fn translate_live_regions(
        &mut self,
        records: &mut [RegionRecord],
        target_language: &str,
        supplemental_prompt: &str,
        memory: MemoryConfig,
        cancellation: &CancellationToken,
        mut on_chunk: impl FnMut(u32, &str),
    ) -> Result<(), BackendFailure> {
        if records.is_empty() {
            return Ok(());
        }
        cancellation.check()?;
        self.load_translator_with_memory(target_language, memory)?;
        let prompt = self.settings.prompt.clone();
        let base_generation = self.settings.generation.clone();
        let translator = self
            .hy
            .as_mut()
            .ok_or_else(|| BackendFailure::internal("Hy provider was not initialized"))?;
        for record in records.iter_mut() {
            let order = record.order;
            let source_text = record.source_text.clone();
            let generation = live_translation_generation(&base_generation, &source_text);
            record.translated_text = translate_live_text_with_recovery(
                translator,
                &source_text,
                target_language,
                &prompt,
                supplemental_prompt,
                &generation,
                cancellation,
                |partial| on_chunk(order, partial),
            )?;
            on_chunk(order, &record.translated_text);
        }
        cancellation.check()
    }

    pub(crate) fn translate_live_subtitle(
        &mut self,
        source_text: &str,
        target_language: &str,
        supplemental_prompt: &str,
        memory: MemoryConfig,
        cancellation: &CancellationToken,
        mut on_chunk: impl FnMut(&str),
    ) -> Result<String, BackendFailure> {
        cancellation.check()?;
        self.load_translator_with_memory(target_language, memory)?;
        let prompt = self.settings.prompt.clone();
        let generation = live_translation_generation(&self.settings.generation, source_text);
        let translator = self
            .hy
            .as_mut()
            .ok_or_else(|| BackendFailure::internal("Hy provider was not initialized"))?;
        let text = translate_live_text_with_recovery(
            translator,
            source_text,
            target_language,
            &prompt,
            supplemental_prompt,
            &generation,
            cancellation,
            |partial| on_chunk(partial),
        )?;
        validate_live_subtitle_translation_output(text)
    }

    fn translate_regions_with_progress(
        &mut self,
        records: &mut [RegionRecord],
        target_language: &str,
        cancellation: &CancellationToken,
        contextual: bool,
        supplemental_prompt: &str,
        mut report_progress: impl FnMut(u8, &'static str),
    ) -> Result<(), BackendFailure> {
        if records.is_empty() {
            return Ok(());
        }
        cancellation.check()?;
        self.load_translator(target_language)?;
        report_progress(70, "Hy-MT2 已就绪");
        if !(1..=4).contains(&self.settings.translation_batch_size) {
            return Err(BackendFailure::arguments(
                "translation batch size must be 1..=4",
            ));
        }
        let regions = records
            .iter()
            .map(|record| crate::backend::contracts::TranslationRegion {
                order: record.order,
                source_text: record.source_text.clone(),
            })
            .collect::<Vec<_>>();
        let translator = self
            .hy
            .as_mut()
            .ok_or_else(|| BackendFailure::internal("Hy provider was not initialized"))?;
        let mut translated = Vec::with_capacity(regions.len());
        let total_batches = regions.len().div_ceil(self.settings.translation_batch_size);
        for (batch_index, batch) in regions
            .chunks(self.settings.translation_batch_size)
            .enumerate()
        {
            translated.extend(if contextual {
                translator.translate_contextual_with_supplemental_prompt(
                    batch,
                    target_language,
                    supplemental_prompt,
                    cancellation,
                )?
            } else {
                translator.translate_with_supplemental_prompt(
                    batch,
                    target_language,
                    supplemental_prompt,
                    cancellation,
                )?
            });
            let progress =
                70 + u8::try_from(((batch_index + 1) * 20) / total_batches).unwrap_or(20);
            report_progress(progress, "Hy-MT2 翻译中");
        }
        if translated.len() != records.len() {
            return Err(BackendFailure::translation(
                "Hy returned an incomplete region set",
            ));
        }
        for record in records {
            let Some(result) = translated.iter().find(|item| item.order == record.order) else {
                return Err(BackendFailure::translation(format!(
                    "Hy returned no translation for region {}",
                    record.order
                )));
            };
            if result.translated_text.trim().is_empty() {
                return Err(BackendFailure::translation(format!(
                    "Hy returned empty translation for region {}",
                    record.order
                )));
            }
            record.translated_text = result.translated_text.clone();
        }
        cancellation.check()
    }

    pub(crate) fn translate_text(
        &mut self,
        text: &str,
        target_language: &str,
        supplemental_prompt: &str,
        cancellation: &CancellationToken,
        mut report_progress: impl FnMut(u8, &'static str),
        mut on_chunk: impl FnMut(&str),
    ) -> Result<String, BackendFailure> {
        cancellation.check()?;
        report_progress(20, "正在准备 Hy-MT2");
        self.load_translator(target_language)?;
        report_progress(45, "Hy-MT2 已就绪");
        let prompt = self.settings.prompt.clone();
        let generation = self.settings.generation.clone();
        let translator = self
            .hy
            .as_mut()
            .ok_or_else(|| BackendFailure::internal("Hy provider was not initialized"))?;
        report_progress(70, "Hy-MT2 生成中");
        let result = translator
            .translate_text(
                text,
                target_language,
                &prompt,
                supplemental_prompt,
                &generation,
                cancellation,
                |chunk| {
                    on_chunk(chunk);
                    Ok(())
                },
            )
            .map_err(|error| {
                if cancellation.is_cancelled() {
                    BackendFailure::cancelled("Hy translation was cancelled")
                } else {
                    BackendFailure::translation(format!("Hy text translation failed: {error:#}"))
                }
            })?;
        cancellation.check()?;
        let text = validate_text_translation_output(result.text)?;
        report_progress(100, "翻译完成");
        Ok(text)
    }

    pub(crate) fn ocr(
        &mut self,
        image: &DecodedImage,
        cancellation: &CancellationToken,
        mut report_progress: impl FnMut(u8, &'static str),
    ) -> Result<OcrOutput, BackendFailure> {
        cancellation.check()?;
        report_progress(20, "正在准备 PP-OCR");
        self.load_ocr()?;
        let ocr = self
            .ocr
            .as_mut()
            .ok_or_else(|| BackendFailure::internal("PP-OCR provider was not initialized"))?;
        let document = ocr.recognize(image, cancellation)?;
        cancellation.check()?;
        report_progress(75, "OCR 识别完成");
        if document.regions.len() > crate::backend::input::MAX_REGIONS {
            return Err(BackendFailure::ocr(
                "OCR region count exceeds the supported bound",
            ));
        }
        report_progress(85, "正在生成 OCR 标注图");
        let output = self
            .output
            .render_ocr(image, document.regions, cancellation)?;
        report_progress(100, "OCR 处理完成");
        Ok(output)
    }
}

fn validate_text_translation_output(text: String) -> Result<String, BackendFailure> {
    if text.trim().is_empty() {
        return Err(BackendFailure::translation(
            "Hy returned empty text translation",
        ));
    }
    if text.len() > crate::backend::input::MAX_TEXT_BYTES {
        return Err(BackendFailure::translation(
            "text output exceeds the 8 MiB limit",
        ));
    }
    Ok(text)
}

fn validate_live_subtitle_translation_output(text: String) -> Result<String, BackendFailure> {
    if text.trim().is_empty() {
        return Err(BackendFailure::translation(
            "Hy returned empty live subtitle translation",
        ));
    }
    Ok(text)
}

fn live_translation_generation(base: &GenerationConfig, source_text: &str) -> GenerationConfig {
    let mut generation = base.clone();
    generation.max_new_tokens = generation
        .max_new_tokens
        .max(LIVE_TRANSLATION_MIN_NEW_TOKENS)
        .max(
            source_text
                .chars()
                .count()
                .saturating_mul(LIVE_TRANSLATION_TOKENS_PER_SOURCE_CHAR),
        )
        .min(MAX_NEW_TOKENS);
    generation
}

fn translate_live_text_with_recovery(
    translator: &mut hy::HyTranslator,
    source_text: &str,
    target_language: &str,
    prompt: &PromptConfig,
    supplemental_prompt: &str,
    generation: &GenerationConfig,
    cancellation: &CancellationToken,
    mut on_chunk: impl FnMut(&str),
) -> Result<String, BackendFailure> {
    let mut streamed_text = String::new();
    let initial = translator
        .translate_text(
            source_text,
            target_language,
            prompt,
            supplemental_prompt,
            generation,
            cancellation,
            |chunk| {
                streamed_text.push_str(chunk);
                on_chunk(&streamed_text);
                Ok(())
            },
        )
        .map_err(|error| {
            if cancellation.is_cancelled() {
                BackendFailure::cancelled("Hy translation was cancelled")
            } else {
                BackendFailure::translation(format!("Hy live translation failed: {error:#}"))
            }
        })?;
    cancellation.check()?;
    if hy::translation_text_is_usable(
        &initial.text,
        source_text,
        target_language,
        prompt,
        supplemental_prompt,
    ) {
        return Ok(initial.text);
    }

    let initial_stop_reason = initial.stop_reason;
    let initial_generated_tokens = initial.stats.generated_tokens;
    translator.reset_context();
    streamed_text.clear();
    on_chunk("");
    let retry_prompt = PromptConfig::default();
    let retry = translator
        .translate_text(
            source_text,
            target_language,
            &retry_prompt,
            "",
            generation,
            cancellation,
            |chunk| {
                streamed_text.push_str(chunk);
                on_chunk(&streamed_text);
                Ok(())
            },
        )
        .map_err(|error| {
            if cancellation.is_cancelled() {
                BackendFailure::cancelled("Hy translation was cancelled")
            } else {
                BackendFailure::translation(format!(
                    "Hy minimal-prompt live translation retry failed: {error:#}"
                ))
            }
        })?;
    cancellation.check()?;
    if hy::translation_text_is_usable(&retry.text, source_text, target_language, &retry_prompt, "")
    {
        return Ok(retry.text);
    }
    Err(BackendFailure::translation(format!(
        "Hy 未生成可用实时译文（首次停止={initial_stop_reason:?}, tokens={initial_generated_tokens}；最小提示重试停止={:?}, tokens={}）",
        retry.stop_reason, retry.stats.generated_tokens,
    )))
}

fn create_device(kind: DeviceKind) -> Result<CandleDevice, BackendFailure> {
    match kind {
        DeviceKind::Cpu => Ok(CandleDevice::Cpu),
        DeviceKind::Cuda => {
            #[cfg(not(feature = "cuda"))]
            {
                return Err(BackendFailure::device("CUDA 模式需要编译 feature `cuda`"));
            }
            #[cfg(feature = "cuda")]
            {
                CandleDevice::new_cuda(0).map_err(|error| {
                    BackendFailure::device(format!("初始化 CUDA 设备失败：{error:#}"))
                })
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn query_gpu_memory(
    device: &CandleDevice,
) -> Result<Option<(String, usize, usize)>, BackendFailure> {
    let CandleDevice::Cuda(cuda) = device else {
        return Ok(None);
    };
    let stream = cuda.cuda_stream();
    let context = stream.context();
    let name = context
        .name()
        .map_err(|error| BackendFailure::device(format!("读取 CUDA 设备名称失败：{error}")))?;
    let (free_bytes, total_bytes) = context
        .mem_get_info()
        .map_err(|error| BackendFailure::device(format!("读取 CUDA 显存信息失败：{error}")))?;
    Ok(Some((name, total_bytes / MIB, free_bytes / MIB)))
}

#[cfg(not(feature = "cuda"))]
fn query_gpu_memory(
    _device: &CandleDevice,
) -> Result<Option<(String, usize, usize)>, BackendFailure> {
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::{BackendEngine, DeviceKind, GpuExecutionPolicy, validate_text_translation_output};
    use crate::{
        backend::{input::decode_image, settings::BackendSettings},
        model_config::{GenerationConfig, MemoryConfig, PromptConfig},
        model_support::CancellationToken,
    };
    use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
    use std::{env, fs, path::PathBuf};

    fn settings(device_kind: DeviceKind) -> BackendSettings {
        BackendSettings {
            detector_model_dir: PathBuf::from("missing-detector"),
            recognizer_model_dir: PathBuf::from("missing-recognizer"),
            hy_model: PathBuf::from("missing-hy.gguf"),
            font_path: None,
            target_language: "Chinese".to_owned(),
            region_parallelism: 16,
            translation_batch_size: 4,
            device_kind,
            idle_unload_minutes: 0,
            prompt: PromptConfig::default(),
            generation: GenerationConfig::default(),
            memory: MemoryConfig::default(),
            model_root: PathBuf::from("models"),
            catalog: Default::default(),
        }
    }

    #[test]
    fn text_translation_output_preserves_model_whitespace() {
        let output = "  第一  行\n下一行  ".to_owned();

        assert_eq!(
            validate_text_translation_output(output.clone()).unwrap(),
            output
        );
        assert!(validate_text_translation_output(" \n\t ".to_owned()).is_err());
    }

    #[test]
    fn cpu_engine_is_constructible_but_translation_is_device_gated() {
        let engine = BackendEngine::new(settings(DeviceKind::Cpu)).expect("cpu device");
        assert_eq!(engine.settings.device_kind, DeviceKind::Cpu);
    }

    #[test]
    fn cpu_engine_carries_effective_generation_config() {
        let mut settings = settings(DeviceKind::Cpu);
        settings.generation.max_new_tokens = 64;

        let engine = BackendEngine::new(settings).expect("cpu device");

        assert_eq!(engine.settings.generation.max_new_tokens, 64);
    }

    #[test]
    fn gpu_policy_uses_vram_to_bound_batch_pixels_and_model_residency() {
        let resident = GpuExecutionPolicy::for_memory(24 * 1024, 20 * 1024);
        assert_eq!(resident.label(), "gpu_resident");
        assert_eq!(resident.region_parallelism(16), 16);
        assert_eq!(resident.recognizer_batch_pixels(), 48 * 3200 * 16);
        assert!(resident.keeps_models_resident());

        let balanced = GpuExecutionPolicy::for_memory(12 * 1024, 3 * 1024);
        assert_eq!(balanced.label(), "gpu_balanced");
        assert_eq!(balanced.region_parallelism(16), 8);
        assert_eq!(balanced.recognizer_batch_pixels(), 48 * 3200 * 8);
        assert!(!balanced.keeps_models_resident());

        let constrained = GpuExecutionPolicy::for_memory(8 * 1024, 2 * 1024);
        assert_eq!(constrained.label(), "gpu_constrained");
        assert_eq!(constrained.region_parallelism(16), 4);
        assert_eq!(constrained.recognizer_batch_pixels(), 48 * 3200 * 4);
        assert!(!constrained.keeps_models_resident());
    }
    #[cfg(all(feature = "cuda", feature = "flash-attn"))]
    #[test]
    #[ignore = "requires the staged PP-OCR/Hy assets and a supported CUDA device"]
    fn cuda_native_fixture_smoke() {
        assert_eq!(
            env::var("SMODELTRANS_RUN_CUDA_E2E").as_deref(),
            Ok("1"),
            "set SMODELTRANS_RUN_CUDA_E2E=1 to run the native CUDA fixture"
        );
        let manifest_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let model_root = manifest_root.join("..").join("models");
        let fixture = manifest_root.join("tests/fixtures/ppocrv5/contract-game-ui.png");
        let encoded = BASE64.encode(fs::read(fixture).expect("fixture"));
        let settings = BackendSettings {
            detector_model_dir: model_root.join("ppocrv5/server_det"),
            recognizer_model_dir: model_root.join("ppocrv5/server_rec"),
            hy_model: model_root.join("hy/Hy-MT2-1.8B-Q4_K_M.gguf"),
            font_path: None,
            target_language: "English".to_owned(),
            region_parallelism: 4,
            translation_batch_size: 4,
            device_kind: DeviceKind::Cuda,
            idle_unload_minutes: 0,
            prompt: PromptConfig::default(),
            generation: GenerationConfig::default(),
            memory: MemoryConfig::default(),
            model_root: model_root.clone(),
            catalog: Default::default(),
        };
        let mut engine = BackendEngine::new(settings).expect("CUDA engine");
        let image = decode_image(
            &encoded,
            "contract-game-ui.png".to_owned(),
            "English".to_owned(),
        )
        .expect("decode fixture");
        let result = engine
            .translate(&image, &CancellationToken::new_for_test(), |_, _| {})
            .expect("native OCR/Hy translation");
        assert!(result.is_translated);
        assert!(!result.text.trim().is_empty());
        assert!(result.annotated_png.starts_with(b"\x89PNG"));
    }
}
