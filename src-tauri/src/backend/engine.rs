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
    #[tracing::instrument(level = "info", skip(settings), fields(device_kind = ?settings.device_kind, region_parallelism = settings.region_parallelism, idle_unload_seconds = settings.idle_unload_seconds, font_path = ?settings.font_path.as_ref().map(|p| p.display().to_string())))]
    pub(crate) fn new(mut settings: BackendSettings) -> Result<Self, BackendFailure> {
        tracing::info!(target: "backend::engine", device_kind = ?settings.device_kind, region_parallelism = settings.region_parallelism, font_path = ?settings.font_path.as_ref().map(|p| p.display().to_string()), "BackendEngine::new entry");
        let __start = std::time::Instant::now();
        let device = create_device(settings.device_kind).inspect_err(|e| {
            tracing::error!(target: "backend::engine", error = %e, duration_ms = __start.elapsed().as_millis() as u64, "BackendEngine::new create_device failed");
        })?;
        tracing::debug!(target: "backend::engine", device_kind = ?settings.device_kind, "device created, querying GPU memory");
        let fallback_policy = if settings.device_kind == DeviceKind::Cuda {
            GpuExecutionPolicy::Balanced
        } else {
            GpuExecutionPolicy::Cpu
        };
        let gpu_policy = query_gpu_memory(&device)
            .inspect_err(|e| {
                tracing::warn!(target: "backend::engine", error = %e, "query_gpu_memory failed, using fallback policy");
            })
            .ok()
            .flatten()
            .map(|(_, total_mib, free_mib)| {
                let policy = GpuExecutionPolicy::for_memory(total_mib, free_mib);
                tracing::debug!(target: "backend::engine", total_mib, free_mib, policy = policy.label(), "GPU memory queried, resolved policy");
                policy
            })
            .unwrap_or(fallback_policy);
        tracing::debug!(target: "backend::engine", gpu_policy = gpu_policy.label(), "resolved gpu_policy");
        settings.region_parallelism = gpu_policy.region_parallelism(settings.region_parallelism);
        let out = Self {
            output: ImageOutput::new(settings.font_path.clone()),
            settings,
            device,
            gpu_policy,
            ocr: None,
            hy: None,
            translator_memory: None,
        };
        tracing::info!(target: "backend::engine", duration_ms = __start.elapsed().as_millis() as u64, gpu_policy = out.gpu_policy.label(), region_parallelism = out.settings.region_parallelism, font_path = ?out.settings.font_path.as_ref().map(|p| p.display().to_string()), "BackendEngine::new success");
        Ok(out)
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

    #[tracing::instrument(level = "info", skip(self), fields(gpu_policy = self.gpu_policy.label(), region_parallelism = self.settings.region_parallelism, ocr_loaded_before = self.ocr.is_some()))]
    pub(crate) fn load_ocr(&mut self) -> Result<(), BackendFailure> {
        tracing::info!(target: "backend::engine", gpu_policy = self.gpu_policy.label(), ocr_loaded_before = self.ocr.is_some(), "load_ocr entry");
        let __start = std::time::Instant::now();
        if self.ocr.is_none() {
            tracing::debug!(target: "backend::engine", "load_ocr: ocr not loaded, preparing to load");
            if !self.gpu_policy.keeps_models_resident() {
                tracing::debug!(target: "backend::engine", "load_ocr: gpu policy does not keep models resident, evicting translator");
                self.hy = None;
                self.translator_memory = None;
            }
            tracing::debug!(target: "backend::engine", detector = %self.settings.detector_model_dir.display(), recognizer = %self.settings.recognizer_model_dir.display(), region_parallelism = self.settings.region_parallelism, batch_pixels = self.gpu_policy.recognizer_batch_pixels(), "load_ocr: loading PpOcrProvider");
            let provider = PpOcrProvider::load(
                &self.settings.detector_model_dir,
                &self.settings.recognizer_model_dir,
                &self.device,
                self.settings.region_parallelism,
                self.gpu_policy.recognizer_batch_pixels(),
            )
            .inspect_err(|e| {
                tracing::error!(target: "backend::engine", error = %e, duration_ms = __start.elapsed().as_millis() as u64, "load_ocr: PpOcrProvider::load failed");
            })?;
            self.ocr = Some(provider);
            tracing::info!(target: "backend::engine", duration_ms = __start.elapsed().as_millis() as u64, "load_ocr: provider loaded successfully");
        } else {
            tracing::debug!(target: "backend::engine", "load_ocr: already loaded, skipping");
        }
        tracing::info!(target: "backend::engine", duration_ms = __start.elapsed().as_millis() as u64, ocr_loaded = self.ocr.is_some(), "load_ocr success");
        Ok(())
    }

    #[tracing::instrument(level = "info", skip(self), fields(target_language = %target_language, memory_present = self.translator_memory.is_some()))]
    pub(crate) fn load_translator(&mut self, target_language: &str) -> Result<(), BackendFailure> {
        tracing::info!(target: "backend::engine", target_language = %target_language, "load_translator entry");
        let __start = std::time::Instant::now();
        let res = self.load_translator_with_memory(target_language, self.settings.memory.clone());
        match &res {
            Ok(_) => tracing::info!(target: "backend::engine", target_language = %target_language, duration_ms = __start.elapsed().as_millis() as u64, translator_loaded = self.hy.is_some(), "load_translator success"),
            Err(e) => tracing::error!(target: "backend::engine", target_language = %target_language, error = %e, duration_ms = __start.elapsed().as_millis() as u64, "load_translator failed"),
        }
        res
    }

    #[tracing::instrument(level = "info", skip(self, memory), fields(target_language = %target_language, gpu_policy = self.gpu_policy.label()))]
    fn load_translator_with_memory(
        &mut self,
        target_language: &str,
        memory: MemoryConfig,
    ) -> Result<(), BackendFailure> {
        tracing::info!(target: "backend::engine", target_language = %target_language, "load_translator_with_memory entry (ensure_model_loaded)");
        let __start = std::time::Instant::now();
        tracing::debug!(target: "backend::engine", target_language = %target_language, "load_translator_with_memory: building ModelConfig");
        let config = ModelConfig::from_parts(
            target_language,
            self.settings.prompt.clone(),
            self.settings.generation.clone(),
            memory.clone(),
        )
        .map_err(|error| {
            tracing::error!(target: "backend::engine", target_language = %target_language, error = %error, duration_ms = __start.elapsed().as_millis() as u64, "load_translator_with_memory: invalid model config");
            BackendFailure::arguments(format!("invalid model config: {error:#}"))
        })?;
        tracing::debug!(target: "backend::engine", target_language = %target_language, "ModelConfig built, checking memory mismatch");
        if self.hy.is_some() && self.translator_memory.as_ref() != Some(&memory) {
            tracing::debug!(target: "backend::engine", "load_translator_with_memory: memory config changed, evicting existing translator");
            self.hy = None;
            self.translator_memory = None;
        }
        if self.hy.is_none() {
            if !self.gpu_policy.keeps_models_resident() {
                tracing::debug!(target: "backend::engine", "load_translator_with_memory: gpu policy does not keep models resident, evicting OCR");
                self.ocr = None;
            }
            tracing::debug!(target: "backend::engine", hy_model = %self.settings.hy_model.display(), target_language = %target_language, "load_translator_with_memory: loading Hy translator");
            let translator = hy::load_with_config(
                &self.settings.hy_model,
                &self.device,
                config.memory,
                config.generation,
                config.prompt,
            )
            .map_err(|error| {
                tracing::error!(target: "backend::engine", target_language = %target_language, error = %error, duration_ms = __start.elapsed().as_millis() as u64, "load_translator_with_memory: hy::load_with_config failed");
                BackendFailure::asset(format!("load local Hy-MT2 model: {error:#}"))
            })?;
            self.hy = Some(translator);
            self.translator_memory = Some(memory);
            tracing::info!(target: "backend::engine", target_language = %target_language, duration_ms = __start.elapsed().as_millis() as u64, "load_translator_with_memory: translator loaded");
        } else {
            tracing::debug!(target: "backend::engine", target_language = %target_language, "load_translator_with_memory: translator already loaded");
        }
        tracing::info!(target: "backend::engine", target_language = %target_language, duration_ms = __start.elapsed().as_millis() as u64, translator_loaded = self.hy.is_some(), "load_translator_with_memory success");
        Ok(())
    }

    pub(crate) fn reset_translator_context(&mut self) {
        tracing::debug!(target: "backend::engine", translator_loaded = self.hy.is_some(), "reset_translator_context entry");
        if let Some(translator) = self.hy.as_mut() {
            translator.reset_context();
            tracing::debug!(target: "backend::engine", "reset_translator_context: context reset");
        } else {
            tracing::debug!(target: "backend::engine", "reset_translator_context: no translator loaded, skipping");
        }
    }

    #[tracing::instrument(level = "info", skip(self, memory, cancellation), fields(target_language = %target_language))]
    pub(crate) fn prepare_live_pipeline(
        &mut self,
        target_language: &str,
        memory: MemoryConfig,
        cancellation: &CancellationToken,
    ) -> Result<(), BackendFailure> {
        tracing::info!(target: "backend::engine", target_language = %target_language, "prepare_live_pipeline entry");
        let __start = std::time::Instant::now();
        tracing::debug!(target: "backend::engine", target_language = %target_language, "prepare_live_pipeline: ensuring OCR loaded");
        self.load_ocr().inspect_err(|e| {
            tracing::error!(target: "backend::engine", target_language = %target_language, error = %e, duration_ms = __start.elapsed().as_millis() as u64, "prepare_live_pipeline: load_ocr failed");
        })?;
        tracing::debug!(target: "backend::engine", "prepare_live_pipeline: warming up OCR");
        self.ocr
            .as_mut()
            .ok_or_else(|| {
                tracing::error!(target: "backend::engine", target_language = %target_language, "prepare_live_pipeline: PP-OCR provider was not initialized");
                BackendFailure::internal("PP-OCR provider was not initialized")
            })?
            .warm_up(cancellation)
            .inspect_err(|e| {
                tracing::error!(target: "backend::engine", target_language = %target_language, error = %e, duration_ms = __start.elapsed().as_millis() as u64, "prepare_live_pipeline: ocr warm_up failed");
            })?;
        tracing::debug!(target: "backend::engine", target_language = %target_language, "prepare_live_pipeline: loading translator");
        self.load_translator_with_memory(target_language, memory).inspect_err(|e| {
            tracing::error!(target: "backend::engine", target_language = %target_language, error = %e, duration_ms = __start.elapsed().as_millis() as u64, "prepare_live_pipeline: load_translator_with_memory failed");
        })?;
        tracing::debug!(target: "backend::engine", target_language = %target_language, "prepare_live_pipeline: warming up Hy translator");
        self.hy
            .as_mut()
            .ok_or_else(|| {
                tracing::error!(target: "backend::engine", target_language = %target_language, "prepare_live_pipeline: Hy translator was not initialized");
                BackendFailure::internal("Hy translator was not initialized")
            })?
            .warm_up(cancellation)
            .map_err(|error| {
                tracing::error!(target: "backend::engine", target_language = %target_language, error = %error, duration_ms = __start.elapsed().as_millis() as u64, "prepare_live_pipeline: Hy warm_up failed");
                BackendFailure::translation(format!("warm up Hy CUDA kernels: {error:#}"))
            })?;
        self.reset_translator_context();
        tracing::info!(target: "backend::engine", target_language = %target_language, duration_ms = __start.elapsed().as_millis() as u64, "prepare_live_pipeline success");
        Ok(())
    }

    #[tracing::instrument(level = "info", skip(self))]
    pub(crate) fn gpu_resource_info(&self) -> Result<Option<GpuResourceInfo>, BackendFailure> {
        tracing::debug!(target: "backend::engine", "gpu_resource_info entry");
        let resources = query_gpu_memory(&self.device).inspect_err(|e| {
            tracing::warn!(target: "backend::engine", error = %e, "gpu_resource_info: query_gpu_memory failed");
        }).ok().flatten();
        let out = Ok(resources
            .map(|(name, total_mib, free_mib)| {
                tracing::debug!(target: "backend::engine", name = %name, total_mib, free_mib, "gpu_resource_info: cuda memory info");
                GpuResourceInfo {
                    name,
                    total_memory_mib: u64::try_from(total_mib).unwrap_or(u64::MAX),
                    free_memory_mib: u64::try_from(free_mib).unwrap_or(u64::MAX),
                    execution_mode: self.gpu_policy.label(),
                }
            })
            .or_else(|| {
                (self.settings.device_kind == DeviceKind::Cuda).then(|| {
                    tracing::debug!(target: "backend::engine", "gpu_resource_info: cuda without memory info, fallback");
                    GpuResourceInfo {
                        name: "CUDA".to_owned(),
                        total_memory_mib: 0,
                        free_memory_mib: 0,
                        execution_mode: self.gpu_policy.label(),
                    }
                })
            }));
        tracing::debug!(target: "backend::engine", has_info = out.as_ref().map(|v| v.is_some()).unwrap_or(false), "gpu_resource_info done");
        out
    }

    #[tracing::instrument(level = "info", skip(self))]
    pub(crate) fn unload_ocr(&mut self) {
        tracing::info!(target: "backend::engine", ocr_loaded_before = self.ocr.is_some(), "unload_ocr entry");
        self.ocr = None;
        tracing::info!(target: "backend::engine", "unload_ocr completed");
    }

    #[tracing::instrument(level = "info", skip(self))]
    pub(crate) fn unload_translator(&mut self) {
        tracing::info!(target: "backend::engine", translator_loaded_before = self.hy.is_some(), "unload_translator entry");
        self.hy = None;
        self.translator_memory = None;
        tracing::info!(target: "backend::engine", "unload_translator completed");
    }

    #[tracing::instrument(level = "info", skip(self))]
    pub(crate) fn unload_models(&mut self) {
        tracing::info!(target: "backend::engine", ocr_loaded = self.ocr.is_some(), translator_loaded = self.hy.is_some(), "unload_models entry");
        self.unload_ocr();
        self.unload_translator();
        tracing::info!(target: "backend::engine", "unload_models completed");
    }

    #[tracing::instrument(level = "info", skip(self, image, cancellation), fields(request_id = %image.file_name(), target_language = %image.target_language(), width = image.canvas().width(), height = image.canvas().height()))]
    pub(crate) fn recognize_regions(
        &mut self,
        image: &DecodedImage,
        cancellation: &CancellationToken,
    ) -> Result<Vec<crate::backend::contracts::RegionRecord>, BackendFailure> {
        tracing::info!(target: "backend::engine", request_id = %image.file_name(), target_language = %image.target_language(), width = image.canvas().width(), height = image.canvas().height(), "recognize_regions entry (ocr_image ensure)");
        let __start = std::time::Instant::now();
        tracing::debug!(target: "backend::engine", request_id = %image.file_name(), "recognize_regions: checking cancellation");
        cancellation.check().inspect_err(|e| {
            tracing::error!(target: "backend::engine", request_id = %image.file_name(), error = %e, "recognize_regions: cancelled before ensure");
        })?;
        tracing::debug!(target: "backend::engine", request_id = %image.file_name(), "recognize_regions: ensure model loaded (load_ocr)");
        self.load_ocr().inspect_err(|e| {
            tracing::error!(target: "backend::engine", request_id = %image.file_name(), error = %e, duration_ms = __start.elapsed().as_millis() as u64, "recognize_regions: load_ocr failed");
        })?;
        let ocr = self
            .ocr
            .as_mut()
            .ok_or_else(|| {
                tracing::error!(target: "backend::engine", request_id = %image.file_name(), "recognize_regions: PP-OCR provider was not initialized");
                BackendFailure::internal("PP-OCR provider was not initialized")
            })?;
        tracing::debug!(target: "backend::engine", request_id = %image.file_name(), width = image.canvas().width(), height = image.canvas().height(), "recognize_regions: invoking ppocr recognize (decoding already done, now ppocr)");
        let document = ocr.recognize(image, cancellation).inspect_err(|e| {
            tracing::error!(target: "backend::engine", request_id = %image.file_name(), error = %e, duration_ms = __start.elapsed().as_millis() as u64, "recognize_regions: ocr.recognize failed");
        })?;
        tracing::debug!(target: "backend::engine", request_id = %image.file_name(), region_count = document.regions.len(), "recognize_regions: ppocr completed");
        cancellation.check().inspect_err(|e| {
            tracing::error!(target: "backend::engine", request_id = %image.file_name(), error = %e, "recognize_regions: cancelled after ocr");
        })?;
        if document.regions.len() > crate::backend::input::MAX_REGIONS {
            tracing::error!(target: "backend::engine", request_id = %image.file_name(), region_count = document.regions.len(), max = crate::backend::input::MAX_REGIONS, "recognize_regions: region count exceeds bound");
            return Err(BackendFailure::ocr(
                "OCR region count exceeds the supported bound",
            ));
        }
        tracing::info!(target: "backend::engine", request_id = %image.file_name(), duration_ms = __start.elapsed().as_millis() as u64, region_count = document.regions.len(), "recognize_regions success");
        Ok(document.regions)
    }

    #[tracing::instrument(level = "info", skip(self, image, cancellation, report_progress), fields(request_id = %image.file_name(), target_language = %image.target_language(), width = image.canvas().width(), height = image.canvas().height(), file_name = %image.file_name()))]
    pub(crate) fn translate(
        &mut self,
        image: &DecodedImage,
        cancellation: &CancellationToken,
        mut report_progress: impl FnMut(u8, &'static str),
    ) -> Result<TranslationOutput, BackendFailure> {
        let __span = tracing::info_span!(target: "backend::engine", "translate_request", request_id = %image.file_name(), target_language = %image.target_language(), width = image.canvas().width(), height = image.canvas().height());
        let _guard = __span.enter();
        tracing::info!(target: "backend::engine", request_id = %image.file_name(), target_language = %image.target_language(), width = image.canvas().width(), height = image.canvas().height(), file_name = %image.file_name(), "translate_image entry");
        let __start = std::time::Instant::now();
        tracing::debug!(target: "backend::engine", request_id = %image.file_name(), "translate: checking cancellation and decoding already completed");
        cancellation.check().inspect_err(|e| {
            tracing::error!(target: "backend::engine", request_id = %image.file_name(), error = %e, "translate: cancelled at entry");
        })?;
        report_progress(20, "正在准备 PP-OCR");
        tracing::debug!(target: "backend::engine", request_id = %image.file_name(), "translate: ensure_model_loaded -> load_ocr");
        self.load_ocr().inspect_err(|e| {
            tracing::error!(target: "backend::engine", request_id = %image.file_name(), error = %e, duration_ms = __start.elapsed().as_millis() as u64, "translate: load_ocr failed");
        })?;
        let ocr = self
            .ocr
            .as_mut()
            .ok_or_else(|| {
                tracing::error!(target: "backend::engine", request_id = %image.file_name(), "translate: PP-OCR provider was not initialized");
                BackendFailure::internal("PP-OCR provider was not initialized")
            })?;
        tracing::debug!(target: "backend::engine", request_id = %image.file_name(), width = image.canvas().width(), height = image.canvas().height(), "translate: invoking ppocr recognize");
        let document = ocr.recognize(image, cancellation).inspect_err(|e| {
            tracing::error!(target: "backend::engine", request_id = %image.file_name(), error = %e, duration_ms = __start.elapsed().as_millis() as u64, "translate: ocr.recognize failed");
        })?;
        tracing::debug!(target: "backend::engine", request_id = %image.file_name(), region_count = document.regions.len(), "translate: ppocr completed, regions detected");
        cancellation.check().inspect_err(|e| {
            tracing::error!(target: "backend::engine", request_id = %image.file_name(), error = %e, "translate: cancelled after ocr");
        })?;
        report_progress(55, "OCR 识别完成");
        if document.regions.len() > crate::backend::input::MAX_REGIONS {
            tracing::error!(target: "backend::engine", request_id = %image.file_name(), region_count = document.regions.len(), "translate: region count exceeds bound");
            return Err(BackendFailure::ocr(
                "OCR region count exceeds the supported bound",
            ));
        }

        let mut records = document.regions;
        if records.is_empty() {
            tracing::debug!(target: "backend::engine", request_id = %image.file_name(), "translate: no regions, rendering empty output");
            let output = self
                .output
                .render(image, &records, image.target_language(), cancellation)
                .inspect_err(|e| {
                    tracing::error!(target: "backend::engine", request_id = %image.file_name(), error = %e, duration_ms = __start.elapsed().as_millis() as u64, "translate: output.render failed for empty regions");
                })?;
            tracing::debug!(target: "backend::engine", request_id = %image.file_name(), "translate: output rendering completed (empty)");
            report_progress(100, "处理完成");
            tracing::info!(target: "backend::engine", request_id = %image.file_name(), duration_ms = __start.elapsed().as_millis() as u64, region_count = 0, "translate success (empty)");
            return Ok(output);
        }

        tracing::debug!(target: "backend::engine", request_id = %image.file_name(), region_count = records.len(), target_language = %image.target_language(), "translate: invoking hy translation");
        self.translate_regions_with_progress(
            &mut records,
            image.target_language(),
            cancellation,
            false,
            "",
            &mut report_progress,
        ).inspect_err(|e| {
            tracing::error!(target: "backend::engine", request_id = %image.file_name(), error = %e, duration_ms = __start.elapsed().as_millis() as u64, "translate: hy translation failed");
        })?;
        tracing::debug!(target: "backend::engine", request_id = %image.file_name(), region_count = records.len(), "translate: hy translation completed, rendering output");
        cancellation.check().inspect_err(|e| {
            tracing::error!(target: "backend::engine", request_id = %image.file_name(), error = %e, "translate: cancelled before render");
        })?;
        report_progress(90, "翻译完成，正在生成标注图");
        let output = self
            .output
            .render(image, &records, image.target_language(), cancellation)
            .inspect_err(|e| {
                tracing::error!(target: "backend::engine", request_id = %image.file_name(), error = %e, duration_ms = __start.elapsed().as_millis() as u64, "translate: output.render failed");
            })?;
        tracing::debug!(target: "backend::engine", request_id = %image.file_name(), "translate: output rendering completed");
        report_progress(100, "翻译完成");
        tracing::info!(target: "backend::engine", request_id = %image.file_name(), duration_ms = __start.elapsed().as_millis() as u64, region_count = records.len(), target_language = %image.target_language(), "translate success");
        Ok(output)
    }

    #[tracing::instrument(level = "info", skip(self, records, cancellation, on_chunk), fields(target_language = %target_language, region_count = records.len()))]
    pub(crate) fn translate_live_regions(
        &mut self,
        records: &mut [RegionRecord],
        target_language: &str,
        supplemental_prompt: &str,
        memory: MemoryConfig,
        cancellation: &CancellationToken,
        mut on_chunk: impl FnMut(u32, &str),
    ) -> Result<(), BackendFailure> {
        tracing::info!(target: "backend::engine", target_language = %target_language, region_count = records.len(), "translate_live_regions entry");
        let __start = std::time::Instant::now();
        if records.is_empty() {
            tracing::debug!(target: "backend::engine", target_language = %target_language, "translate_live_regions: empty records, skipping");
            tracing::info!(target: "backend::engine", target_language = %target_language, duration_ms = __start.elapsed().as_millis() as u64, region_count = 0, "translate_live_regions success (empty)");
            return Ok(());
        }
        cancellation.check().inspect_err(|e| {
            tracing::error!(target: "backend::engine", target_language = %target_language, error = %e, "translate_live_regions: cancelled at entry");
        })?;
        tracing::debug!(target: "backend::engine", target_language = %target_language, "translate_live_regions: ensure_model_loaded -> load_translator_with_memory");
        self.load_translator_with_memory(target_language, memory).inspect_err(|e| {
            tracing::error!(target: "backend::engine", target_language = %target_language, error = %e, duration_ms = __start.elapsed().as_millis() as u64, "translate_live_regions: load_translator_with_memory failed");
        })?;
        let prompt = self.settings.prompt.clone();
        let base_generation = self.settings.generation.clone();
        let translator = self
            .hy
            .as_mut()
            .ok_or_else(|| {
                tracing::error!(target: "backend::engine", target_language = %target_language, "translate_live_regions: Hy provider was not initialized");
                BackendFailure::internal("Hy provider was not initialized")
            })?;
        tracing::debug!(target: "backend::engine", target_language = %target_language, region_count = records.len(), "translate_live_regions: starting per-region hy translation");
        for record in records.iter_mut() {
            let __batch_span = tracing::info_span!(target: "backend::engine", "live_region_translate", order = record.order, target_language = %target_language, source_len = record.source_text.chars().count());
            let _g = __batch_span.enter();
            let order = record.order;
            let source_text = record.source_text.clone();
            let generation = live_translation_generation(&base_generation, &source_text);
            tracing::debug!(target: "backend::engine", order, source_text_len = source_text.chars().count(), max_new_tokens = generation.max_new_tokens, "translate_live_regions: translating region");
            record.translated_text = translate_live_text_with_recovery(
                translator,
                &source_text,
                target_language,
                &prompt,
                supplemental_prompt,
                &generation,
                cancellation,
                |partial| on_chunk(order, partial),
            ).inspect_err(|e| {
                tracing::error!(target: "backend::engine", order, target_language = %target_language, error = %e, duration_ms = __start.elapsed().as_millis() as u64, "translate_live_regions: region translation failed");
            })?;
            tracing::debug!(target: "backend::engine", order, translated_len = record.translated_text.chars().count(), "translate_live_regions: region translated");
            on_chunk(order, &record.translated_text);
        }
        cancellation.check().inspect_err(|e| {
            tracing::error!(target: "backend::engine", target_language = %target_language, error = %e, "translate_live_regions: cancelled after loop");
        })?;
        tracing::info!(target: "backend::engine", target_language = %target_language, duration_ms = __start.elapsed().as_millis() as u64, region_count = records.len(), "translate_live_regions success");
        Ok(())
    }

    #[tracing::instrument(level = "info", skip(self, memory, cancellation, on_chunk), fields(target_language = %target_language, source_len = source_text.chars().count()))]
    pub(crate) fn translate_live_subtitle(
        &mut self,
        source_text: &str,
        target_language: &str,
        supplemental_prompt: &str,
        memory: MemoryConfig,
        cancellation: &CancellationToken,
        mut on_chunk: impl FnMut(&str),
    ) -> Result<String, BackendFailure> {
        tracing::info!(target: "backend::engine", target_language = %target_language, source_len = source_text.chars().count(), "translate_live_subtitle entry");
        let __start = std::time::Instant::now();
        cancellation.check().inspect_err(|e| {
            tracing::error!(target: "backend::engine", target_language = %target_language, error = %e, "translate_live_subtitle: cancelled at entry");
        })?;
        tracing::debug!(target: "backend::engine", target_language = %target_language, "translate_live_subtitle: ensure_model_loaded -> load_translator_with_memory");
        self.load_translator_with_memory(target_language, memory).inspect_err(|e| {
            tracing::error!(target: "backend::engine", target_language = %target_language, error = %e, duration_ms = __start.elapsed().as_millis() as u64, "translate_live_subtitle: load_translator_with_memory failed");
        })?;
        let prompt = self.settings.prompt.clone();
        let generation = live_translation_generation(&self.settings.generation, source_text);
        tracing::debug!(target: "backend::engine", target_language = %target_language, max_new_tokens = generation.max_new_tokens, "translate_live_subtitle: generation config prepared");
        let translator = self
            .hy
            .as_mut()
            .ok_or_else(|| {
                tracing::error!(target: "backend::engine", target_language = %target_language, "translate_live_subtitle: Hy provider was not initialized");
                BackendFailure::internal("Hy provider was not initialized")
            })?;
        tracing::debug!(target: "backend::engine", target_language = %target_language, "translate_live_subtitle: invoking hy translation with recovery");
        let text = translate_live_text_with_recovery(
            translator,
            source_text,
            target_language,
            &prompt,
            supplemental_prompt,
            &generation,
            cancellation,
            |partial| on_chunk(partial),
        ).inspect_err(|e| {
            tracing::error!(target: "backend::engine", target_language = %target_language, error = %e, duration_ms = __start.elapsed().as_millis() as u64, "translate_live_subtitle: hy translation failed");
        })?;
        tracing::debug!(target: "backend::engine", target_language = %target_language, text_len = text.chars().count(), "translate_live_subtitle: hy translation completed, validating");
        let validated = validate_live_subtitle_translation_output(text).inspect_err(|e| {
            tracing::error!(target: "backend::engine", target_language = %target_language, error = %e, "translate_live_subtitle: validation failed");
        })?;
        tracing::info!(target: "backend::engine", target_language = %target_language, duration_ms = __start.elapsed().as_millis() as u64, text_len = validated.chars().count(), "translate_live_subtitle success");
        Ok(validated)
    }

    #[tracing::instrument(level = "info", skip(self, records, cancellation, report_progress), fields(target_language = %target_language, region_count = records.len(), contextual = contextual, batch_size = self.settings.translation_batch_size))]
    fn translate_regions_with_progress(
        &mut self,
        records: &mut [RegionRecord],
        target_language: &str,
        cancellation: &CancellationToken,
        contextual: bool,
        supplemental_prompt: &str,
        mut report_progress: impl FnMut(u8, &'static str),
    ) -> Result<(), BackendFailure> {
        tracing::info!(target: "backend::engine", target_language = %target_language, region_count = records.len(), contextual, "translate_regions_with_progress entry");
        let __start = std::time::Instant::now();
        if records.is_empty() {
            tracing::debug!(target: "backend::engine", "translate_regions_with_progress: empty, skipping hy");
            tracing::info!(target: "backend::engine", duration_ms = __start.elapsed().as_millis() as u64, region_count = 0, "translate_regions_with_progress success (empty)");
            return Ok(());
        }
        cancellation.check().inspect_err(|e| {
            tracing::error!(target: "backend::engine", target_language = %target_language, error = %e, "translate_regions_with_progress: cancelled at entry");
        })?;
        tracing::debug!(target: "backend::engine", target_language = %target_language, "translate_regions_with_progress: ensure_model_loaded -> load_translator");
        self.load_translator(target_language).inspect_err(|e| {
            tracing::error!(target: "backend::engine", target_language = %target_language, error = %e, duration_ms = __start.elapsed().as_millis() as u64, "translate_regions_with_progress: load_translator failed");
        })?;
        report_progress(70, "Hy-MT2 已就绪");
        if !(1..=4).contains(&self.settings.translation_batch_size) {
            tracing::error!(target: "backend::engine", target_language = %target_language, batch_size = self.settings.translation_batch_size, "translate_regions_with_progress: invalid batch size");
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
        tracing::debug!(target: "backend::engine", target_language = %target_language, region_count = regions.len(), batch_size = self.settings.translation_batch_size, "translate_regions_with_progress: prepared batch regions");
        let translator = self
            .hy
            .as_mut()
            .ok_or_else(|| {
                tracing::error!(target: "backend::engine", target_language = %target_language, "translate_regions_with_progress: Hy provider was not initialized");
                BackendFailure::internal("Hy provider was not initialized")
            })?;
        let mut translated = Vec::with_capacity(regions.len());
        let total_batches = regions.len().div_ceil(self.settings.translation_batch_size);
        tracing::debug!(target: "backend::engine", target_language = %target_language, total_batches, "translate_regions_with_progress: starting batch loop (hy translation)");
        for (batch_index, batch) in regions
            .chunks(self.settings.translation_batch_size)
            .enumerate()
        {
            let __batch_span = tracing::info_span!(target: "backend::engine", "hy_batch", batch_index, batch_size = batch.len(), target_language = %target_language);
            let _g = __batch_span.enter();
            tracing::debug!(target: "backend::engine", batch_index, batch_len = batch.len(), "translate_regions_with_progress: translating batch");
            let batch_result = if contextual {
                translator.translate_contextual_with_supplemental_prompt(
                    batch,
                    target_language,
                    supplemental_prompt,
                    cancellation,
                )
            } else {
                translator.translate_with_supplemental_prompt(
                    batch,
                    target_language,
                    supplemental_prompt,
                    cancellation,
                )
            };
            let batch_translated = batch_result.inspect_err(|e| {
                tracing::error!(target: "backend::engine", batch_index, target_language = %target_language, error = %e, duration_ms = __start.elapsed().as_millis() as u64, "translate_regions_with_progress: batch translation failed");
            })?;
            tracing::debug!(target: "backend::engine", batch_index, translated_in_batch = batch_translated.len(), "translate_regions_with_progress: batch completed");
            translated.extend(batch_translated);
            let progress =
                70 + u8::try_from(((batch_index + 1) * 20) / total_batches).unwrap_or(20);
            report_progress(progress, "Hy-MT2 翻译中");
        }
        if translated.len() != records.len() {
            tracing::error!(target: "backend::engine", target_language = %target_language, expected = records.len(), got = translated.len(), "translate_regions_with_progress: incomplete region set");
            return Err(BackendFailure::translation(
                "Hy returned an incomplete region set",
            ));
        }
        for record in &mut *records {
            let Some(result) = translated.iter().find(|item| item.order == record.order) else {
                tracing::error!(target: "backend::engine", order = record.order, target_language = %target_language, "translate_regions_with_progress: missing translation for region");
                return Err(BackendFailure::translation(format!(
                    "Hy returned no translation for region {}",
                    record.order
                )));
            };
            if result.translated_text.trim().is_empty() {
                tracing::error!(target: "backend::engine", order = record.order, target_language = %target_language, "translate_regions_with_progress: empty translation for region");
                return Err(BackendFailure::translation(format!(
                    "Hy returned empty translation for region {}",
                    record.order
                )));
            }
            record.translated_text = result.translated_text.clone();
        }
        cancellation.check().inspect_err(|e| {
            tracing::error!(target: "backend::engine", target_language = %target_language, error = %e, "translate_regions_with_progress: cancelled after translation");
        })?;
        tracing::info!(target: "backend::engine", target_language = %target_language, duration_ms = __start.elapsed().as_millis() as u64, region_count = translated.len(), "translate_regions_with_progress success");
        Ok(())
    }

    #[tracing::instrument(level = "info", skip(self, cancellation, report_progress, on_chunk), fields(target_language = %target_language, text_len = text.chars().count(), supplemental_len = supplemental_prompt.chars().count()))]
    pub(crate) fn translate_text(
        &mut self,
        text: &str,
        target_language: &str,
        supplemental_prompt: &str,
        cancellation: &CancellationToken,
        mut report_progress: impl FnMut(u8, &'static str),
        mut on_chunk: impl FnMut(&str),
    ) -> Result<String, BackendFailure> {
        let __span = tracing::info_span!(target: "backend::engine", "translate_text_request", target_language = %target_language, text_len = text.chars().count());
        let _guard = __span.enter();
        tracing::info!(target: "backend::engine", target_language = %target_language, text_len = text.chars().count(), supplemental_len = supplemental_prompt.chars().count(), "translate_text entry");
        let __start = std::time::Instant::now();
        cancellation.check().inspect_err(|e| {
            tracing::error!(target: "backend::engine", target_language = %target_language, error = %e, "translate_text: cancelled at entry");
        })?;
        report_progress(20, "正在准备 Hy-MT2");
        tracing::debug!(target: "backend::engine", target_language = %target_language, "translate_text: ensure_model_loaded -> load_translator");
        self.load_translator(target_language).inspect_err(|e| {
            tracing::error!(target: "backend::engine", target_language = %target_language, error = %e, duration_ms = __start.elapsed().as_millis() as u64, "translate_text: load_translator failed");
        })?;
        report_progress(45, "Hy-MT2 已就绪");
        let prompt = self.settings.prompt.clone();
        let generation = self.settings.generation.clone();
        let translator = self
            .hy
            .as_mut()
            .ok_or_else(|| {
                tracing::error!(target: "backend::engine", target_language = %target_language, "translate_text: Hy provider was not initialized");
                BackendFailure::internal("Hy provider was not initialized")
            })?;
        report_progress(70, "Hy-MT2 生成中");
        tracing::debug!(target: "backend::engine", target_language = %target_language, text_len = text.chars().count(), max_new_tokens = generation.max_new_tokens, "translate_text: invoking hy translate_text");
        let result = translator
            .translate_text(
                text,
                target_language,
                &prompt,
                supplemental_prompt,
                &generation,
                cancellation,
                |chunk| {
                    tracing::trace!(target: "backend::engine", target_language = %target_language, chunk_len = chunk.chars().count(), "translate_text: on_chunk");
                    on_chunk(chunk);
                    Ok(())
                },
            )
            .map_err(|error| {
                if cancellation.is_cancelled() {
                    tracing::warn!(target: "backend::engine", target_language = %target_language, error = %error, duration_ms = __start.elapsed().as_millis() as u64, "translate_text: cancelled");
                    BackendFailure::cancelled("Hy translation was cancelled")
                } else {
                    tracing::error!(target: "backend::engine", target_language = %target_language, error = %error, duration_ms = __start.elapsed().as_millis() as u64, "translate_text: Hy text translation failed");
                    BackendFailure::translation(format!("Hy text translation failed: {error:#}"))
                }
            })?;
        tracing::debug!(target: "backend::engine", target_language = %target_language, generated_len = result.text.chars().count(), "translate_text: hy generation completed, validating");
        cancellation.check().inspect_err(|e| {
            tracing::error!(target: "backend::engine", target_language = %target_language, error = %e, "translate_text: cancelled after generation");
        })?;
        let text = validate_text_translation_output(result.text).inspect_err(|e| {
            tracing::error!(target: "backend::engine", target_language = %target_language, error = %e, "translate_text: validation failed");
        })?;
        report_progress(100, "翻译完成");
        tracing::info!(target: "backend::engine", target_language = %target_language, duration_ms = __start.elapsed().as_millis() as u64, text_len = text.chars().count(), "translate_text success");
        Ok(text)
    }

    #[tracing::instrument(level = "info", skip(self, image, cancellation, report_progress), fields(request_id = %image.file_name(), width = image.canvas().width(), height = image.canvas().height()))]
    pub(crate) fn ocr(
        &mut self,
        image: &DecodedImage,
        cancellation: &CancellationToken,
        mut report_progress: impl FnMut(u8, &'static str),
    ) -> Result<OcrOutput, BackendFailure> {
        let __span = tracing::info_span!(target: "backend::engine", "ocr_request", request_id = %image.file_name(), width = image.canvas().width(), height = image.canvas().height());
        let _guard = __span.enter();
        tracing::info!(target: "backend::engine", request_id = %image.file_name(), width = image.canvas().width(), height = image.canvas().height(), file_name = %image.file_name(), "ocr_image entry");
        let __start = std::time::Instant::now();
        tracing::debug!(target: "backend::engine", request_id = %image.file_name(), "ocr: checking cancellation");
        cancellation.check().inspect_err(|e| {
            tracing::error!(target: "backend::engine", request_id = %image.file_name(), error = %e, "ocr: cancelled at entry");
        })?;
        report_progress(20, "正在准备 PP-OCR");
        tracing::debug!(target: "backend::engine", request_id = %image.file_name(), "ocr: ensure_model_loaded -> load_ocr");
        self.load_ocr().inspect_err(|e| {
            tracing::error!(target: "backend::engine", request_id = %image.file_name(), error = %e, duration_ms = __start.elapsed().as_millis() as u64, "ocr: load_ocr failed");
        })?;
        let ocr = self
            .ocr
            .as_mut()
            .ok_or_else(|| {
                tracing::error!(target: "backend::engine", request_id = %image.file_name(), "ocr: PP-OCR provider was not initialized");
                BackendFailure::internal("PP-OCR provider was not initialized")
            })?;
        tracing::debug!(target: "backend::engine", request_id = %image.file_name(), width = image.canvas().width(), height = image.canvas().height(), "ocr: invoking ppocr recognize (decoding already done)");
        let document = ocr.recognize(image, cancellation).inspect_err(|e| {
            tracing::error!(target: "backend::engine", request_id = %image.file_name(), error = %e, duration_ms = __start.elapsed().as_millis() as u64, "ocr: recognize failed");
        })?;
        tracing::debug!(target: "backend::engine", request_id = %image.file_name(), region_count = document.regions.len(), "ocr: ppocr completed");
        cancellation.check().inspect_err(|e| {
            tracing::error!(target: "backend::engine", request_id = %image.file_name(), error = %e, "ocr: cancelled after recognize");
        })?;
        report_progress(75, "OCR 识别完成");
        if document.regions.len() > crate::backend::input::MAX_REGIONS {
            tracing::error!(target: "backend::engine", request_id = %image.file_name(), region_count = document.regions.len(), "ocr: region count exceeds bound");
            return Err(BackendFailure::ocr(
                "OCR region count exceeds the supported bound",
            ));
        }
        report_progress(85, "正在生成 OCR 标注图");
        tracing::debug!(target: "backend::engine", request_id = %image.file_name(), region_count = document.regions.len(), "ocr: rendering output (output rendering)");
        let output = self
            .output
            .render_ocr(image, document.regions, cancellation)
            .inspect_err(|e| {
                tracing::error!(target: "backend::engine", request_id = %image.file_name(), error = %e, duration_ms = __start.elapsed().as_millis() as u64, "ocr: render_ocr failed");
            })?;
        tracing::debug!(target: "backend::engine", request_id = %image.file_name(), "ocr: output rendering completed");
        report_progress(100, "OCR 处理完成");
        tracing::info!(target: "backend::engine", request_id = %image.file_name(), duration_ms = __start.elapsed().as_millis() as u64, region_count = output.regions.len(), width = image.canvas().width(), height = image.canvas().height(), "ocr success");
        Ok(output)
    }
}

fn validate_text_translation_output(text: String) -> Result<String, BackendFailure> {
    tracing::debug!(target: "backend::engine", text_len = text.chars().count(), "validate_text_translation_output entry");
    if text.trim().is_empty() {
        tracing::error!(target: "backend::engine", "validate_text_translation_output: empty translation");
        return Err(BackendFailure::translation(
            "Hy returned empty text translation",
        ));
    }
    if text.len() > crate::backend::input::MAX_TEXT_BYTES {
        tracing::error!(target: "backend::engine", text_bytes = text.len(), max = crate::backend::input::MAX_TEXT_BYTES, "validate_text_translation_output: exceeds limit");
        return Err(BackendFailure::translation(
            "text output exceeds the 8 MiB limit",
        ));
    }
    tracing::debug!(target: "backend::engine", text_len = text.chars().count(), "validate_text_translation_output success");
    Ok(text)
}

fn validate_live_subtitle_translation_output(text: String) -> Result<String, BackendFailure> {
    tracing::debug!(target: "backend::engine", text_len = text.chars().count(), "validate_live_subtitle_translation_output entry");
    if text.trim().is_empty() {
        tracing::error!(target: "backend::engine", "validate_live_subtitle_translation_output: empty");
        return Err(BackendFailure::translation(
            "Hy returned empty live subtitle translation",
        ));
    }
    tracing::debug!(target: "backend::engine", "validate_live_subtitle_translation_output success");
    Ok(text)
}

fn live_translation_generation(base: &GenerationConfig, source_text: &str) -> GenerationConfig {
    tracing::trace!(target: "backend::engine", source_len = source_text.chars().count(), base_max_new_tokens = base.max_new_tokens, "live_translation_generation entry");
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
    tracing::trace!(target: "backend::engine", max_new_tokens = generation.max_new_tokens, "live_translation_generation computed");
    generation
}

#[tracing::instrument(level = "info", skip(translator, cancellation, on_chunk), fields(target_language = %target_language, source_len = source_text.chars().count()))]
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
    tracing::info!(target: "backend::engine", target_language = %target_language, source_len = source_text.chars().count(), "translate_live_text_with_recovery entry");
    let __start = std::time::Instant::now();
    let mut streamed_text = String::new();
    tracing::debug!(target: "backend::engine", target_language = %target_language, "translate_live_text_with_recovery: initial Hy translate_text");
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
                tracing::warn!(target: "backend::engine", target_language = %target_language, error = %error, duration_ms = __start.elapsed().as_millis() as u64, "translate_live_text_with_recovery: cancelled on initial");
                BackendFailure::cancelled("Hy translation was cancelled")
            } else {
                tracing::error!(target: "backend::engine", target_language = %target_language, error = %error, duration_ms = __start.elapsed().as_millis() as u64, "translate_live_text_with_recovery: initial failed");
                BackendFailure::translation(format!("Hy live translation failed: {error:#}"))
            }
        })?;
    tracing::debug!(target: "backend::engine", target_language = %target_language, generated_tokens = initial.stats.generated_tokens, stop_reason = ?initial.stop_reason, text_len = initial.text.chars().count(), "translate_live_text_with_recovery: initial completed");
    cancellation.check().inspect_err(|e| {
        tracing::error!(target: "backend::engine", target_language = %target_language, error = %e, "translate_live_text_with_recovery: cancelled after initial");
    })?;
    if hy::translation_text_is_usable(
        &initial.text,
        source_text,
        target_language,
        prompt,
        supplemental_prompt,
    ) {
        tracing::info!(target: "backend::engine", target_language = %target_language, duration_ms = __start.elapsed().as_millis() as u64, "translate_live_text_with_recovery: initial usable, success");
        return Ok(initial.text);
    }

    let initial_stop_reason = initial.stop_reason;
    let initial_generated_tokens = initial.stats.generated_tokens;
    tracing::warn!(target: "backend::engine", target_language = %target_language, initial_stop_reason = ?initial_stop_reason, initial_generated_tokens, "translate_live_text_with_recovery: initial not usable, retrying with minimal prompt");
    translator.reset_context();
    streamed_text.clear();
    on_chunk("");
    let retry_prompt = PromptConfig::default();
    tracing::debug!(target: "backend::engine", target_language = %target_language, "translate_live_text_with_recovery: retry with minimal prompt");
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
                tracing::warn!(target: "backend::engine", target_language = %target_language, error = %error, duration_ms = __start.elapsed().as_millis() as u64, "translate_live_text_with_recovery: cancelled on retry");
                BackendFailure::cancelled("Hy translation was cancelled")
            } else {
                tracing::error!(target: "backend::engine", target_language = %target_language, error = %error, duration_ms = __start.elapsed().as_millis() as u64, "translate_live_text_with_recovery: retry failed");
                BackendFailure::translation(format!(
                    "Hy minimal-prompt live translation retry failed: {error:#}"
                ))
            }
        })?;
    tracing::debug!(target: "backend::engine", target_language = %target_language, retry_stop_reason = ?retry.stop_reason, retry_tokens = retry.stats.generated_tokens, "translate_live_text_with_recovery: retry completed");
    cancellation.check().inspect_err(|e| {
        tracing::error!(target: "backend::engine", target_language = %target_language, error = %e, "translate_live_text_with_recovery: cancelled after retry");
    })?;
    if hy::translation_text_is_usable(&retry.text, source_text, target_language, &retry_prompt, "")
    {
        tracing::info!(target: "backend::engine", target_language = %target_language, duration_ms = __start.elapsed().as_millis() as u64, "translate_live_text_with_recovery: retry usable, success");
        return Ok(retry.text);
    }
    tracing::error!(target: "backend::engine", target_language = %target_language, initial_stop_reason = ?initial_stop_reason, initial_generated_tokens, retry_stop_reason = ?retry.stop_reason, retry_tokens = retry.stats.generated_tokens, duration_ms = __start.elapsed().as_millis() as u64, "translate_live_text_with_recovery: both attempts not usable");
    Err(BackendFailure::translation(format!(
        "Hy 未生成可用实时译文（首次停止={initial_stop_reason:?}, tokens={initial_generated_tokens}；最小提示重试停止={:?}, tokens={}）",
        retry.stop_reason, retry.stats.generated_tokens,
    )))
}

#[tracing::instrument(level = "debug", skip(kind), fields(device_kind = ?kind))]
fn create_device(kind: DeviceKind) -> Result<CandleDevice, BackendFailure> {
    tracing::debug!(target: "backend::engine", device_kind = ?kind, "create_device entry");
    let res = match kind {
        DeviceKind::Cpu => Ok(CandleDevice::Cpu),
        DeviceKind::Cuda => {
            #[cfg(not(feature = "cuda"))]
            {
                tracing::error!(target: "backend::engine", "create_device: CUDA feature not enabled");
                return Err(BackendFailure::device("CUDA 模式需要编译 feature `cuda`"));
            }
            #[cfg(feature = "cuda")]
            {
                tracing::debug!(target: "backend::engine", "create_device: initializing CUDA device 0");
                CandleDevice::new_cuda(0).map_err(|error| {
                    tracing::error!(target: "backend::engine", error = %error, "create_device: CUDA init failed");
                    BackendFailure::device(format!("初始化 CUDA 设备失败：{error:#}"))
                })
            }
        }
    };
    match &res {
        Ok(device) => tracing::debug!(target: "backend::engine", device = ?device, "create_device success"),
        Err(e) => tracing::error!(target: "backend::engine", error = %e, "create_device failed"),
    }
    res
}

#[cfg(feature = "cuda")]
#[tracing::instrument(level = "debug", skip(device))]
fn query_gpu_memory(
    device: &CandleDevice,
) -> Result<Option<(String, usize, usize)>, BackendFailure> {
    tracing::debug!(target: "backend::engine", "query_gpu_memory entry (cuda)");
    let CandleDevice::Cuda(cuda) = device else {
        tracing::debug!(target: "backend::engine", "query_gpu_memory: not cuda device, returning None");
        return Ok(None);
    };
    let stream = cuda.cuda_stream();
    let context = stream.context();
    let name = context
        .name()
        .map_err(|error| {
            tracing::error!(target: "backend::engine", error = %error, "query_gpu_memory: read device name failed");
            BackendFailure::device(format!("读取 CUDA 设备名称失败：{error}"))
        })?;
    tracing::debug!(target: "backend::engine", name = %name, "query_gpu_memory: device name read");
    let (free_bytes, total_bytes) = context
        .mem_get_info()
        .map_err(|error| {
            tracing::error!(target: "backend::engine", error = %error, "query_gpu_memory: mem_get_info failed");
            BackendFailure::device(format!("读取 CUDA 显存信息失败：{error}"))
        })?;
    tracing::debug!(target: "backend::engine", name = %name, total_mib = total_bytes / MIB, free_mib = free_bytes / MIB, "query_gpu_memory success");
    Ok(Some((name, total_bytes / MIB, free_bytes / MIB)))
}

#[cfg(not(feature = "cuda"))]
#[tracing::instrument(level = "debug", skip(_device))]
fn query_gpu_memory(
    _device: &CandleDevice,
) -> Result<Option<(String, usize, usize)>, BackendFailure> {
    tracing::debug!(target: "backend::engine", "query_gpu_memory: non-cuda build, returning None");
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
            idle_unload_seconds: 0,
            prompt: PromptConfig::default(),
            generation: GenerationConfig::default(),
            memory: MemoryConfig::default(),
            model_root: PathBuf::from("models"),
            catalog: Default::default(),
            openai_compat: Default::default(),
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
            idle_unload_seconds: 0,
            prompt: PromptConfig::default(),
            generation: GenerationConfig::default(),
            memory: MemoryConfig::default(),
            model_root: model_root.clone(),
            catalog: Default::default(),
            openai_compat: Default::default(),
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
