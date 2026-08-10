use crate::{
    backend::{
        contracts::{HyPort, OcrOutput, OcrPort, OutputPort, RegionRecord, TranslationOutput},
        failure::BackendFailure,
        input::DecodedImage,
        settings::{BackendSettings, DeviceKind},
    },
    model_config::ModelConfig,
    model_support::CancellationToken,
    models::{hy, ppocrv5::PpOcrV5Provider},
    output::ImageOutput,
};
use candle_core::Device as CandleDevice;

pub(crate) struct BackendEngine {
    pub(crate) settings: BackendSettings,
    device: CandleDevice,
    ocr: Option<PpOcrV5Provider>,
    hy: Option<hy::HyTranslator>,
    output: ImageOutput,
}

impl BackendEngine {
    pub(crate) fn new(settings: BackendSettings) -> Result<Self, BackendFailure> {
        let device = create_device(settings.device_kind)?;
        Ok(Self {
            output: ImageOutput::new(settings.font_path.clone()),
            settings,
            device,
            ocr: None,
            hy: None,
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
            self.ocr = Some(PpOcrV5Provider::load(
                &self.settings.detector_model_dir,
                &self.settings.recognizer_model_dir,
                &self.device,
                self.settings.region_parallelism,
            )?);
        }
        Ok(())
    }

    pub(crate) fn load_translator(&mut self, target_language: &str) -> Result<(), BackendFailure> {
        let config = ModelConfig::from_parts(
            target_language,
            self.settings.prompt.clone(),
            self.settings.generation.clone(),
            self.settings.memory.clone(),
        )
        .map_err(|error| BackendFailure::arguments(format!("invalid model config: {error:#}")))?;
        if self.hy.is_none() {
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
        }
        Ok(())
    }

    pub(crate) fn unload_ocr(&mut self) {
        self.ocr = None;
    }

    pub(crate) fn unload_translator(&mut self) {
        self.hy = None;
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
            .ok_or_else(|| BackendFailure::internal("PP-OCRv5 provider was not initialized"))?;
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
        report_progress(20, "正在准备 PP-OCRv5");
        self.load_ocr()?;
        let ocr = self
            .ocr
            .as_mut()
            .ok_or_else(|| BackendFailure::internal("PP-OCRv5 provider was not initialized"))?;
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

        report_progress(60, "正在准备 Hy-MT2");
        self.translate_regions_with_progress(
            &mut records,
            image.target_language(),
            cancellation,
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
        self.translate_regions_with_progress(records, target_language, cancellation, |_, _| {})
    }

    fn translate_regions_with_progress(
        &mut self,
        records: &mut [RegionRecord],
        target_language: &str,
        cancellation: &CancellationToken,
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
            cancellation.check()?;
            translated.extend(translator.translate(batch, target_language, cancellation)?);
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
        cancellation: &CancellationToken,
        mut report_progress: impl FnMut(u8, &'static str),
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
                &generation,
                cancellation,
                |_| Ok(()),
            )
            .map_err(|error| {
                if cancellation.is_cancelled() {
                    BackendFailure::cancelled("Hy translation was cancelled")
                } else {
                    BackendFailure::translation(format!("Hy text translation failed: {error:#}"))
                }
            })?;
        cancellation.check()?;
        let text = result.text.trim().to_owned();
        if text.is_empty() {
            return Err(BackendFailure::translation(
                "Hy returned empty text translation",
            ));
        }
        if text.len() > crate::backend::input::MAX_TEXT_BYTES {
            return Err(BackendFailure::translation(
                "text output exceeds the 8 MiB limit",
            ));
        }
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
        report_progress(20, "正在准备 PP-OCRv5");
        self.load_ocr()?;
        let ocr = self
            .ocr
            .as_mut()
            .ok_or_else(|| BackendFailure::internal("PP-OCRv5 provider was not initialized"))?;
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

fn create_device(kind: DeviceKind) -> Result<CandleDevice, BackendFailure> {
    match kind {
        DeviceKind::Cpu => Ok(CandleDevice::Cpu),
        DeviceKind::Cuda => {
            #[cfg(not(feature = "flash-attn"))]
            {
                return Err(BackendFailure::device(
                    "CUDA 图片翻译需要编译 feature `cuda,flash-attn`",
                ));
            }
            #[cfg(feature = "flash-attn")]
            {
                CandleDevice::new_cuda(0).map_err(|error| {
                    BackendFailure::device(format!("初始化 CUDA 设备失败：{error:#}"))
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BackendEngine, DeviceKind};
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
        }
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
    #[cfg(all(feature = "cuda", feature = "flash-attn"))]
    #[test]
    #[ignore = "requires the staged PP-OCRv5/Hy assets and a supported CUDA device"]
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
            detector_model_dir: model_root.join("ppocrv5/detector"),
            recognizer_model_dir: model_root.join("ppocrv5/recognizer"),
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
