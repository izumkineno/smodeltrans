//! PP-OCR provider adapter and native detector/recognizer pipeline.

use super::{
    assets::{DetectorPostProcess, GraphRole, PpOcrAssets, PpOcrVariant},
    common::{DetectorOutput, RecognizerOutput},
    geometry::{self, DetectorProfile, QuadI, RegionCrop},
    records::{PpOcrCharacterRecord, PpOcrRegionRecord},
    v5::mobile::{PpOcrMobileDetector, PpOcrMobileRecognizer},
    v5::server::{PpOcrServerDetector, PpOcrServerRecognizer},
    v6::{PpOcrV6Detector, PpOcrV6Recognizer},
};
use crate::{
    backend::{
        contracts::{OcrDocument, OcrPort, RegionRecord},
        failure::BackendFailure,
        input::DecodedImage,
    },
    model_support::CancellationToken,
};
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops;
use image::{GrayImage, Luma, Rgb, RgbImage, imageops};
use imageproc::contours::find_contours;
use rayon::prelude::*;
use std::{path::Path, sync::LazyLock, time::Instant};

const MIN_BOX_SIDE: f32 = 3.0;
const MAX_TOTAL_CROP_PIXELS: u64 = 64 * 1024 * 1024;
const MIN_RECOGNIZER_WIDTH: usize = 320;
const PARALLEL_CROP_MIN_QUADS: usize = 4;
const MAX_BATCH_WIDTH_RATIO_NUMERATOR: usize = 3;
const MAX_BATCH_WIDTH_RATIO_DENOMINATOR: usize = 2;
const RECOGNIZER_IMAGE_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const RECOGNIZER_IMAGE_STD: [f32; 3] = [0.5, 0.5, 0.5];
const MAX_RECOGNITION_RESCUE_CANDIDATES: usize = 24;
const RECOGNITION_RESCUE_MAX_SPAN: usize = 3;

fn trace_ocr(message: std::fmt::Arguments<'_>) {
    if std::env::var_os("SMODELTRANS_TRACE_OCR").is_some() {
        eprintln!("[ocr] {message}");
    }
}

static OCR_POOL: LazyLock<rayon::ThreadPool> = LazyLock::new(|| {
    rayon::ThreadPoolBuilder::new()
        .num_threads(
            std::thread::available_parallelism()
                .map(|count| count.get().min(8))
                .unwrap_or(4),
        )
        .thread_name(|index| format!("ocr-{index}"))
        .build()
        .expect("build OCR thread pool")
});

fn ocr_pool() -> &'static rayon::ThreadPool {
    &OCR_POOL
}
/// Batching strategy for a recognizer region pass.
///
/// `Ratio` keeps the width-similarity batching used by the main pass; padding a
/// batch to its widest member does not change CTC decoding.  `ExactWidth`
/// groups only regions whose resized width is identical, so every member's
/// `batch_width` matches what a single-region forward would have used — this
/// keeps character geometry bit-identical for rescue candidates while still
/// collapsing same-width candidates into shared forwards.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RecognitionBatchMode {
    Ratio,
    ExactWidth,
}

#[derive(Debug)]
struct ImageNormalizer {
    mean: Tensor,
    std: Tensor,
}

impl ImageNormalizer {
    fn new(mean: [f32; 3], std: [f32; 3], device: &Device) -> Result<Self> {
        Ok(Self {
            mean: Tensor::from_slice(&mean, (1, 3, 1, 1), device)
                .context("construct image normalization mean")?,
            std: Tensor::from_slice(&std, (1, 3, 1, 1), device)
                .context("construct image normalization standard deviation")?,
        })
    }

    fn apply(&self, image: &Tensor) -> Result<Tensor> {
        image
            .broadcast_sub(&self.mean)?
            .broadcast_div(&self.std)
            .context("normalize image tensor")
    }
}

/// A loaded PP-OCR detector graph from any supported family or tier.
enum DetectorKind {
    V5Server(PpOcrServerDetector),
    V5Mobile(PpOcrMobileDetector),
    V6(PpOcrV6Detector),
}

impl DetectorKind {
    fn load(assets: &PpOcrAssets, device: &Device) -> std::result::Result<Self, BackendFailure> {
        match assets.variant {
            PpOcrVariant::V5Server => PpOcrServerDetector::load(assets, device)
                .map(Self::V5Server)
                .map_err(|error| {
                    BackendFailure::ocr(format!("load PP-OCR detector graph: {error:#}"))
                }),
            PpOcrVariant::V5Mobile => PpOcrMobileDetector::load(assets, device)
                .map(Self::V5Mobile)
                .map_err(|error| {
                    BackendFailure::ocr(format!("load PP-OCR mobile detector graph: {error:#}"))
                }),
            PpOcrVariant::V6Tiny | PpOcrVariant::V6Small | PpOcrVariant::V6Medium => {
                PpOcrV6Detector::load(assets, device)
                    .map(Self::V6)
                    .map_err(|error| {
                        BackendFailure::ocr(format!(
                            "load PP-OCR {} detector graph: {error:#}",
                            assets.variant.label()
                        ))
                    })
            }
        }
    }

    fn forward(&self, input: &Tensor) -> Result<DetectorOutput> {
        match self {
            Self::V5Server(detector) => detector.forward(input),
            Self::V5Mobile(detector) => detector.forward(input),
            Self::V6(detector) => detector.forward(input),
        }
    }
}

/// A loaded PP-OCR recognizer graph from any supported family or tier.
enum RecognizerKind {
    V5Server(PpOcrServerRecognizer),
    V5Mobile(PpOcrMobileRecognizer),
    V6(PpOcrV6Recognizer),
}

impl RecognizerKind {
    fn load(assets: &PpOcrAssets, device: &Device) -> std::result::Result<Self, BackendFailure> {
        match assets.variant {
            PpOcrVariant::V5Server => PpOcrServerRecognizer::load(assets, device)
                .map(Self::V5Server)
                .map_err(|error| {
                    BackendFailure::ocr(format!("load PP-OCR recognizer graph: {error:#}"))
                }),
            PpOcrVariant::V5Mobile => PpOcrMobileRecognizer::load(assets, device)
                .map(Self::V5Mobile)
                .map_err(|error| {
                    BackendFailure::ocr(format!("load PP-OCR mobile recognizer graph: {error:#}"))
                }),
            PpOcrVariant::V6Tiny | PpOcrVariant::V6Small | PpOcrVariant::V6Medium => {
                PpOcrV6Recognizer::load(assets, device)
                    .map(Self::V6)
                    .map_err(|error| {
                        BackendFailure::ocr(format!(
                            "load PP-OCR {} recognizer graph: {error:#}",
                            assets.variant.label()
                        ))
                    })
            }
        }
    }

    fn forward(&self, input: &Tensor) -> Result<RecognizerOutput> {
        match self {
            Self::V5Server(recognizer) => recognizer.forward(input),
            Self::V5Mobile(recognizer) => recognizer.forward(input),
            Self::V6(recognizer) => recognizer.forward(input),
        }
    }
}

/// A loaded PP-OCR detector and recognizer pair.
///
/// We load both graphs from the validated local safetensors assets before the
/// provider is exposed to the backend.  Requests only perform preprocessing,
/// native Candle inference, and deterministic postprocessing.
pub(crate) struct PpOcrProvider {
    detector: DetectorKind,
    recognizer: RecognizerKind,
    variant: PpOcrVariant,
    postprocess: DetectorPostProcess,
    characters: Vec<String>,
    device: Device,
    detector_normalizer: ImageNormalizer,
    recognizer_normalizer: ImageNormalizer,
    region_parallelism: usize,
    max_recognizer_batch_pixels: usize,
    warmed_up: bool,
}

impl PpOcrProvider {
    pub(crate) fn load(
        detector_dir: &Path,
        recognizer_dir: &Path,
        device: &Device,
        region_parallelism: usize,
        max_recognizer_batch_pixels: usize,
    ) -> std::result::Result<Self, BackendFailure> {
        if region_parallelism == 0 {
            return Err(BackendFailure::arguments(
                "PP-OCR region parallelism must be positive",
            ));
        }
        if max_recognizer_batch_pixels == 0 {
            return Err(BackendFailure::arguments(
                "PP-OCR recognizer batch pixel budget must be positive",
            ));
        }
        let detector_assets = PpOcrAssets::preflight(GraphRole::Detector, detector_dir)?;
        let recognizer_assets = PpOcrAssets::preflight(GraphRole::Recognizer, recognizer_dir)?;
        if detector_assets.variant != recognizer_assets.variant {
            return Err(BackendFailure::asset(format!(
                "detector variant {} does not match recognizer variant {}",
                detector_assets.variant.label(),
                recognizer_assets.variant.label()
            )));
        }
        let variant = detector_assets.variant;
        let postprocess = detector_assets.postprocess.ok_or_else(|| {
            BackendFailure::asset("detector assets carry no DBPostProcess parameters")
        })?;
        let characters = recognizer_assets.character_list.clone();
        let detector = DetectorKind::load(&detector_assets, device)?;
        let recognizer = RecognizerKind::load(&recognizer_assets, device)?;
        let detector_normalizer =
            ImageNormalizer::new([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], device).map_err(
                |error| {
                    BackendFailure::ocr(format!(
                        "prepare PP-OCR detector normalization: {error:#}"
                    ))
                },
            )?;
        let recognizer_normalizer =
            ImageNormalizer::new(RECOGNIZER_IMAGE_MEAN, RECOGNIZER_IMAGE_STD, device).map_err(
                |error| {
                    BackendFailure::ocr(format!(
                        "prepare PP-OCR recognizer normalization: {error:#}"
                    ))
                },
            )?;
        Ok(Self {
            detector,
            recognizer,
            variant,
            postprocess,
            characters,
            device: device.clone(),
            detector_normalizer,
            recognizer_normalizer,
            region_parallelism,
            max_recognizer_batch_pixels,
            warmed_up: false,
        })
    }

    pub(crate) fn warm_up(
        &mut self,
        cancellation: &CancellationToken,
    ) -> std::result::Result<(), BackendFailure> {
        if self.warmed_up || !matches!(self.device, Device::Cuda(_)) {
            return Ok(());
        }
        let canvas = RgbImage::from_pixel(320, 96, Rgb([0, 0, 0]));
        let image = DecodedImage::from_rgb_image(canvas, "ppocr-warmup", "English");
        let _ = <Self as OcrPort>::recognize(self, &image, cancellation)?;
        let _ =
            self.recognize_quad(&image, [[0, 0], [319, 0], [319, 95], [0, 95]], cancellation)?;
        self.warmed_up = true;
        Ok(())
    }

    pub(crate) fn recognize_quad(
        &mut self,
        image: &DecodedImage,
        quad: QuadI,
        cancellation: &CancellationToken,
    ) -> std::result::Result<Option<RegionRecord>, BackendFailure> {
        cancellation.check()?;
        let mut records = recognize_regions_with_retry(
            image.canvas(),
            &[quad],
            &self.recognizer,
            &self.characters,
            &self.device,
            &self.recognizer_normalizer,
            1,
            self.max_recognizer_batch_pixels,
            cancellation,
        )
        .map_err(|error| BackendFailure::ocr(format!("recognize OCR region: {error:#}")))?
        .into_iter()
        .map(PpOcrRegionRecord::into_contract)
        .collect::<Vec<_>>();
        cancellation.check()?;
        Ok(records.pop())
    }
}

impl OcrPort for PpOcrProvider {
    fn recognize(
        &mut self,
        image: &DecodedImage,
        cancellation: &CancellationToken,
    ) -> std::result::Result<OcrDocument, BackendFailure> {
        cancellation.check()?;
        let started = Instant::now();
        let canvas = image.canvas();
        let profile = DetectorProfile::for_image(canvas.width(), canvas.height())
            .map_err(|error| BackendFailure::ocr(format!("build detector profile: {error:#}")))?;
        let detector_input_started = Instant::now();
        let detector_input =
            detector_tensor(canvas, profile, &self.device, &self.detector_normalizer).map_err(
                |error| BackendFailure::ocr(format!("prepare detector input: {error:#}")),
            )?;
        let detector_input_ms = detector_input_started.elapsed().as_millis();
        cancellation.check()?;
        let detector_forward_started = Instant::now();
        let detector_output = self
            .detector
            .forward(&detector_input)
            .map_err(|error| BackendFailure::ocr(format!("run detector graph: {error:#}")))?;
        let detector_forward_ms = detector_forward_started.elapsed().as_millis();
        let detector_postprocess_started = Instant::now();
        let quads = detector_quads(&detector_output, profile, self.postprocess).map_err(
            |error| BackendFailure::ocr(format!("postprocess detector output: {error:#}")),
        )?;
        let detector_postprocess_ms = detector_postprocess_started.elapsed().as_millis();
        cancellation.check()?;
        if quads.is_empty() {
            trace_ocr(format_args!(
                "variant={} phase=main detector_input={detector_input_ms}ms detector_forward={detector_forward_ms}ms detector_postprocess={detector_postprocess_ms}ms recognize_total=0ms regions=0 total={}ms",
                self.variant.label(),
                started.elapsed().as_millis()
            ));
            return Ok(OcrDocument {
                regions: Vec::new(),
            });
        }
        let recognize_started = Instant::now();
        let records = recognize_regions_with_retry(
            canvas,
            &quads,
            &self.recognizer,
            &self.characters,
            &self.device,
            &self.recognizer_normalizer,
            self.region_parallelism,
            self.max_recognizer_batch_pixels,
            cancellation,
        )
        .map_err(|error| BackendFailure::ocr(format!("recognize OCR regions: {error:#}")))?;
        let recognize_total_ms = recognize_started.elapsed().as_millis();
        cancellation.check()?;
        let regions = records.len();
        let document = OcrDocument {
            regions: records
                .into_iter()
                .map(PpOcrRegionRecord::into_contract)
                .collect(),
        };
        trace_ocr(format_args!(
            "variant={} phase=main detector_input={detector_input_ms}ms detector_forward={detector_forward_ms}ms detector_postprocess={detector_postprocess_ms}ms recognize_total={recognize_total_ms}ms regions={regions} total={}ms",
            self.variant.label(),
            started.elapsed().as_millis()
        ));
        Ok(document)
    }
}

fn detector_tensor(
    image: &RgbImage,
    profile: DetectorProfile,
    device: &Device,
    normalizer: &ImageNormalizer,
) -> Result<Tensor> {
    if matches!(device, Device::Cuda(_)) {
        let rgb = resized_rgb_tensor(
            image,
            profile.detector_height as usize,
            profile.detector_width as usize,
            device,
        )?;
        let blue = rgb.narrow(1, 2, 1)?;
        let green = rgb.narrow(1, 1, 1)?;
        let red = rgb.narrow(1, 0, 1)?;
        let bgr = Tensor::cat(&[&blue, &green, &red], 1)?;
        return normalizer.apply(&bgr);
    }

    let resized = imageops::resize(
        image,
        profile.detector_width,
        profile.detector_height,
        imageops::FilterType::Triangle,
    );
    let pixel_count = resized.width() as usize * resized.height() as usize;
    let pixels = resized.as_raw();
    let mut values = Vec::with_capacity(pixel_count * 3);
    let source_channels = [
        (2_usize, 0.485_f32, 0.229_f32),
        (1, 0.456, 0.224),
        (0, 0.406, 0.225),
    ];
    for (source_channel, mean, std) in source_channels {
        for index in 0..pixel_count {
            let value = pixels[index * 3 + source_channel];
            values.push((f32::from(value) / 255.0 - mean) / std);
        }
    }
    Tensor::from_vec(
        values,
        (
            1,
            3,
            profile.detector_height as usize,
            profile.detector_width as usize,
        ),
        device,
    )
    .context("construct detector input tensor")
}
fn resized_rgb_tensor(
    image: &RgbImage,
    output_height: usize,
    output_width: usize,
    device: &Device,
) -> Result<Tensor> {
    let input = Tensor::from_slice(
        image.as_raw(),
        (1, image.height() as usize, image.width() as usize, 3),
        device,
    )?
    .permute((0, 3, 1, 2))?
    .contiguous()?
    .to_dtype(DType::F32)?;
    let input = (input / 255.0)?;
    if image.height() as usize == output_height && image.width() as usize == output_width {
        Ok(input)
    } else {
        input
            .upsample_bilinear2d(output_height, output_width, false)
            .context("resize image tensor on GPU")
    }
}

fn detector_contours(
    probabilities: &[f32],
    width: usize,
    height: usize,
    binary_thresh: f32,
) -> Result<Vec<Vec<(usize, usize)>>> {
    ensure!(
        width > 0 && height > 0,
        "detector output has empty dimensions"
    );
    let expected = width
        .checked_mul(height)
        .context("detector output dimensions overflow")?;
    ensure!(
        probabilities.len() == expected,
        "detector probability map size does not match dimensions"
    );
    let width_u32 = u32::try_from(width).context("detector width exceeds image limits")?;
    let height_u32 = u32::try_from(height).context("detector height exceeds image limits")?;
    let bitmap = GrayImage::from_fn(width_u32, height_u32, |x, y| {
        let probability = probabilities[y as usize * width + x as usize];
        if probability > binary_thresh {
            Luma([255_u8])
        } else {
            Luma([0_u8])
        }
    });
    Ok(find_contours::<u32>(&bitmap)
        .into_iter()
        .map(|contour| {
            contour
                .points
                .into_iter()
                .map(|point| (point.x as usize, point.y as usize))
                .collect::<Vec<_>>()
        })
        .filter(|contour| !contour.is_empty())
        .collect())
}

fn component_quad(component: &[(usize, usize)]) -> [[f32; 2]; 4] {
    debug_assert!(!component.is_empty());
    let mut points = component
        .iter()
        .map(|&(x, y)| [x as f64, y as f64])
        .collect::<Vec<_>>();
    points.sort_by(|left, right| {
        left[0]
            .total_cmp(&right[0])
            .then_with(|| left[1].total_cmp(&right[1]))
    });
    points.dedup();
    if points.len() < 2 {
        let point = points.first().copied().unwrap_or([0.0, 0.0]);
        return [[point[0] as f32, point[1] as f32]; 4];
    }
    let cross = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    let mut lower = Vec::with_capacity(points.len());
    for point in points.iter().copied() {
        while lower.len() >= 2
            && cross(lower[lower.len() - 2], lower[lower.len() - 1], point) <= 0.0
        {
            lower.pop();
        }
        lower.push(point);
    }
    let mut upper = Vec::with_capacity(points.len());
    for point in points.iter().rev().copied() {
        while upper.len() >= 2
            && cross(upper[upper.len() - 2], upper[upper.len() - 1], point) <= 0.0
        {
            upper.pop();
        }
        upper.push(point);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    let hull = lower;

    let mut best_area = f64::INFINITY;
    let mut best_quad = [[0.0_f32; 2]; 4];
    for index in 0..hull.len() {
        let next = (index + 1) % hull.len();
        let dx = hull[next][0] - hull[index][0];
        let dy = hull[next][1] - hull[index][1];
        let edge_length = dx.hypot(dy);
        if edge_length <= f64::EPSILON {
            continue;
        }
        let cosine = dx / edge_length;
        let sine = dy / edge_length;
        let mut min_u = f64::INFINITY;
        let mut max_u = f64::NEG_INFINITY;
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for &point in &hull {
            let u = point[0] * cosine + point[1] * sine;
            let v = -point[0] * sine + point[1] * cosine;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let area = (max_u - min_u) * (max_v - min_v);
        if area >= best_area {
            continue;
        }
        let point = |u: f64, v: f64| {
            [
                (u * cosine - v * sine) as f32,
                (u * sine + v * cosine) as f32,
            ]
        };
        best_area = area;
        best_quad = [
            point(min_u, min_v),
            point(max_u, min_v),
            point(max_u, max_v),
            point(min_u, max_v),
        ];
    }
    best_quad
}

/// Match DB postprocessing's target-width branch: narrow inputs round up,
/// while inputs wider than the 320/48 minimum use the processor's integer floor.
fn recognizer_resize_width(crop_width: u32, crop_height: u32) -> Result<u32> {
    ensure!(crop_width > 0 && crop_height > 0, "OCR crop is empty");
    let numerator = u64::from(crop_width)
        .checked_mul(48)
        .context("OCR crop width overflows recognizer target size")?;
    let denominator = u64::from(crop_height);
    let minimum_numerator = 320_u64
        .checked_mul(denominator)
        .context("OCR crop height overflows recognizer target size")?;
    let width = if numerator <= minimum_numerator {
        numerator
            .checked_add(denominator - 1)
            .context("OCR crop target width overflows")?
            / denominator
    } else {
        numerator / denominator
    };
    Ok(width.min(3200) as u32)
}

fn unclipped_quad(quad: [[f32; 2]; 4], ratio: f32) -> [[f32; 2]; 4] {
    let vector = |from: [f32; 2], to: [f32; 2]| [to[0] - from[0], to[1] - from[1]];
    let length = |vector: [f32; 2]| vector[0].hypot(vector[1]);
    let horizontal = vector(quad[0], quad[1]);
    let vertical = vector(quad[0], quad[3]);
    let width = length(horizontal);
    let height = length(vertical);
    if width <= f32::EPSILON || height <= f32::EPSILON {
        return quad;
    }
    let offset = width * height * ratio / (2.0 * (width + height));
    let horizontal = [horizontal[0] / width, horizontal[1] / width];
    let vertical = [vertical[0] / height, vertical[1] / height];
    let center = [
        (quad[0][0] + quad[2][0]) * 0.5,
        (quad[0][1] + quad[2][1]) * 0.5,
    ];
    let half_width = width * 0.5 + offset;
    let half_height = height * 0.5 + offset;
    let point = |horizontal_sign: f32, vertical_sign: f32| {
        [
            center[0]
                + horizontal_sign * half_width * horizontal[0]
                + vertical_sign * half_height * vertical[0],
            center[1]
                + horizontal_sign * half_width * horizontal[1]
                + vertical_sign * half_height * vertical[1],
        ]
    };
    [
        point(-1.0, -1.0),
        point(1.0, -1.0),
        point(1.0, 1.0),
        point(-1.0, 1.0),
    ]
}

fn clip_detector_quad(quad: [[f32; 2]; 4], profile: DetectorProfile) -> [[f32; 2]; 4] {
    let max_x = profile.detector_width.saturating_sub(1) as f32;
    let max_y = profile.detector_height.saturating_sub(1) as f32;
    quad.map(|[x, y]| [x.clamp(0.0, max_x), y.clamp(0.0, max_y)])
}
fn quad_short_side(quad: [[f32; 2]; 4]) -> f32 {
    let side = |left: [f32; 2], right: [f32; 2]| (right[0] - left[0]).hypot(right[1] - left[1]);
    side(quad[0], quad[1]).min(side(quad[0], quad[3]))
}

fn point_in_quad(point: [f64; 2], quad: [[f32; 2]; 4]) -> bool {
    let mut positive = false;
    let mut negative = false;
    for index in 0..4 {
        let current = quad[index];
        let next = quad[(index + 1) % 4];
        let cross = (f64::from(next[0]) - f64::from(current[0]))
            * (point[1] - f64::from(current[1]))
            - (f64::from(next[1]) - f64::from(current[1])) * (point[0] - f64::from(current[0]));
        if cross > 1e-6 {
            positive = true;
        } else if cross < -1e-6 {
            negative = true;
        }
    }
    !(positive && negative)
}

/// Score the filled pre-unclip rectangle, matching DB's polygon mask mean.
fn filled_quad_score(
    probabilities: &[f32],
    width: usize,
    height: usize,
    quad: [[f32; 2]; 4],
) -> f32 {
    let min_x = quad
        .iter()
        .map(|point| point[0].floor())
        .fold(f32::INFINITY, f32::min)
        .max(0.0) as usize;
    let max_x = quad
        .iter()
        .map(|point| point[0].ceil())
        .fold(f32::NEG_INFINITY, f32::max)
        .min(width.saturating_sub(1) as f32) as usize;
    let min_y = quad
        .iter()
        .map(|point| point[1].floor())
        .fold(f32::INFINITY, f32::min)
        .max(0.0) as usize;
    let max_y = quad
        .iter()
        .map(|point| point[1].ceil())
        .fold(f32::NEG_INFINITY, f32::max)
        .min(height.saturating_sub(1) as f32) as usize;
    if min_x > max_x || min_y > max_y {
        return 0.0;
    }
    // cv2.fillPoly receives int32 vertices after subtracting the ROI origin.
    let raster_quad = quad.map(|[x, y]| {
        [
            (x - min_x as f32) as i32 as f32,
            (y - min_y as f32) as i32 as f32,
        ]
    });
    let mut sum = 0.0_f32;
    let mut count = 0usize;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            if point_in_quad([(x - min_x) as f64, (y - min_y) as f64], raster_quad) {
                sum += probabilities[y * width + x];
                count += 1;
            }
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn detector_quads(
    output: &DetectorOutput,
    profile: DetectorProfile,
    postprocess: DetectorPostProcess,
) -> Result<Vec<QuadI>> {
    ensure!(
        output.shape.len() == 4
            && output.shape[0] == 1
            && output.shape[1] == 1
            && output.shape[2] > 0
            && output.shape[3] > 0,
        "detector output must be [1, 1, H, W] with positive dimensions"
    );
    let height = output.shape[2];
    let width = output.shape[3];
    let flat = output.tensor().flatten_all()?.to_vec1::<f32>()?;
    let expected = height
        .checked_mul(width)
        .context("detector output shape overflows")?;
    ensure!(
        flat.len() == expected,
        "detector output tensor size does not match its shape"
    );
    let scale_x = profile.detector_width as f32 / width as f32;
    let scale_y = profile.detector_height as f32 / height as f32;
    let contours = detector_contours(&flat, width, height, postprocess.binary_thresh)?
        .into_iter()
        .take(postprocess.max_candidates)
        .collect::<Vec<_>>();
    let processed = ocr_pool().install(|| {
        contours
            .par_iter()
            .map(|contour| {
                // DB's cv2.minAreaRect is applied to each contour before scoring.
                let quad = component_quad(contour);
                if quad_short_side(quad) < MIN_BOX_SIDE {
                    return Ok(None);
                }
                let score = filled_quad_score(&flat, width, height, quad);
                if !score.is_finite() || score < postprocess.box_thresh {
                    return Ok(None);
                }
                let unclipped = unclipped_quad(quad, postprocess.unclip_ratio);
                if quad_short_side(unclipped) < MIN_BOX_SIDE + 2.0 {
                    return Ok(None);
                }
                let clipped = clip_detector_quad(unclipped, profile);
                let raw = clipped.map(|[x, y]| [x * scale_x, y * scale_y]);
                Ok(Some(geometry::map_detector_quad(raw, profile)?))
            })
            .collect::<Result<Vec<_>>>()
    })?;
    Ok(processed.into_iter().flatten().collect())
}

#[derive(Debug)]
struct RecognitionJob {
    index: usize,
    quad: QuadI,
    crop: RegionCrop,
    resized_width: usize,
}

fn recognition_batch_len(
    widths: impl IntoIterator<Item = usize>,
    max_regions: usize,
    max_batch_pixels: usize,
) -> usize {
    if max_regions == 0 {
        return 0;
    }
    let mut widths = widths.into_iter();
    let Some(first_width) = widths.next() else {
        return 0;
    };
    let batch_width = first_width.max(MIN_RECOGNIZER_WIDTH);
    let mut batch_len = 1;
    for width in widths.take(max_regions - 1) {
        let width = width.max(MIN_RECOGNIZER_WIDTH);
        if width.saturating_mul(MAX_BATCH_WIDTH_RATIO_NUMERATOR)
            < batch_width.saturating_mul(MAX_BATCH_WIDTH_RATIO_DENOMINATOR)
        {
            break;
        }
        let next_len = batch_len + 1;
        let padded_pixels = batch_width.saturating_mul(48).saturating_mul(next_len);
        if padded_pixels > max_batch_pixels {
            break;
        }
        batch_len = next_len;
    }
    batch_len
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DecodedToken {
    token: usize,
    timestep: usize,
}

#[derive(Debug, Eq, PartialEq)]
struct DecodedRecognizer {
    text: String,
    tokens: Vec<DecodedToken>,
    time_steps: usize,
    confidence_milli: u16,
}

fn fill_recognition_tensor(
    crop: &RgbImage,
    resized_width: usize,
    tensor_width: usize,
    values: &mut [f32],
) -> Result<()> {
    ensure!(resized_width > 0, "OCR crop is empty");
    ensure!(
        tensor_width >= resized_width,
        "recognizer batch width is narrower than the crop"
    );
    let plane = 48usize
        .checked_mul(tensor_width)
        .context("recognizer tensor dimensions overflow")?;
    ensure!(
        values.len() == 3usize * plane,
        "recognizer tensor buffer size does not match its shape"
    );
    let resized = imageops::resize(
        crop,
        resized_width as u32,
        48,
        imageops::FilterType::Triangle,
    );
    // RecResizeImg resizes in RGB, then maps each channel with
    // `(pixel / 255 - 0.5) / 0.5`. This differs from the detector's BGR
    // ImageNet contract and preserves the checkpoint's trained input range.
    for channel in 0..3 {
        for y in 0..48 {
            for x in 0..resized_width {
                let pixel = resized.get_pixel(x as u32, y as u32);
                let value = f32::from(pixel[channel]) / 255.0;
                values[channel * plane + y * tensor_width + x] =
                    (value - RECOGNIZER_IMAGE_MEAN[channel]) / RECOGNIZER_IMAGE_STD[channel];
            }
        }
    }
    Ok(())
}


fn recognition_batch_tensor(
    jobs: &[RecognitionJob],
    device: &Device,
    normalizer: &ImageNormalizer,
) -> Result<Tensor> {
    ensure!(!jobs.is_empty(), "recognizer batch is empty");
    let batch_width = jobs
        .iter()
        .map(|job| job.resized_width)
        .max()
        .context("recognizer batch is empty")?
        .max(MIN_RECOGNIZER_WIDTH);
    if matches!(device, Device::Cuda(_)) {
        let mut samples = Vec::with_capacity(jobs.len());
        for job in jobs {
            let resized = resized_rgb_tensor(&job.crop.image, 48, job.resized_width, device)?;
            let normalized = normalizer.apply(&resized)?;
            let padded =
                normalized.pad_with_zeros(3, 0, batch_width.saturating_sub(job.resized_width))?;
            samples.push(padded);
        }
        let samples = samples.iter().collect::<Vec<_>>();
        return Tensor::cat(&samples, 0).context("construct GPU recognizer batch tensor");
    }

    let plane = 48usize
        .checked_mul(batch_width)
        .context("recognizer batch dimensions overflow")?;
    let sample_stride = 3usize
        .checked_mul(plane)
        .context("recognizer batch dimensions overflow")?;
    let mut values = vec![0.0_f32; jobs.len() * sample_stride];
    for (batch_index, job) in jobs.iter().enumerate() {
        let offset = batch_index * sample_stride;
        fill_recognition_tensor(
            &job.crop.image,
            job.resized_width,
            batch_width,
            &mut values[offset..offset + sample_stride],
        )?;
    }
    Tensor::from_vec(values, (jobs.len(), 3, 48, batch_width), device)
        .context("construct recognizer batch tensor")
}

fn decode_recognizer_token_ids(
    token_ids: &[u32],
    characters: &[String],
) -> Result<DecodedRecognizer> {
    let mut text = String::new();
    let mut tokens = Vec::new();
    let mut previous = 0usize;
    for (timestep, &token) in token_ids.iter().enumerate() {
        let token = token as usize;
        if token != 0 && token != previous {
            let character = characters
                .get(token)
                .with_context(|| format!("recognizer token {token} exceeds character list"))?;
            text.push_str(character);
            tokens.push(DecodedToken { token, timestep });
        }
        previous = token;
    }
    Ok(DecodedRecognizer {
        text,
        tokens,
        time_steps: token_ids.len(),
        confidence_milli: 0,
    })
}


fn decode_recognizer_batch_detailed(
    output: &RecognizerOutput,
    characters: &[String],
) -> Result<Vec<DecodedRecognizer>> {
    ensure!(
        output.shape.len() == 3
            && output.shape[0] > 0
            && output.shape[1] > 0
            && output.shape[2] > 0,
        "recognizer output must be [B, T, V] with positive dimensions"
    );
    let batch = output.shape[0];
    let time = output.shape[1];
    let vocabulary = output.shape[2];
    let expected_tokens = batch
        .checked_mul(time)
        .context("recognizer token id count overflowed")?;
    let token_ids = output
        .tensor()
        .argmax(2)?
        .flatten_all()?
        .to_vec1::<u32>()
        .context("copy recognizer batch token ids to CPU")?;
    ensure!(
        token_ids.len() == expected_tokens,
        "recognizer token id count does not match its shape"
    );
    let mut decoded = token_ids
        .chunks_exact(time)
        .map(|row| decode_recognizer_token_ids(row, characters))
        .collect::<Result<Vec<_>>>()?;
    let emission_indices = decoded
        .iter()
        .enumerate()
        .flat_map(|(batch_index, decoded)| {
            decoded.tokens.iter().map(move |token| {
                u32::try_from(batch_index * time + token.timestep)
                    .context("recognizer emission index overflowed")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if emission_indices.is_empty() {
        return Ok(decoded);
    }
    let emission_count = emission_indices.len();
    let indices = Tensor::from_vec(emission_indices, emission_count, output.tensor().device())
        .context("construct recognizer emission indices")?;
    let selected_logits = output
        .tensor()
        .reshape((batch * time, vocabulary))?
        .index_select(&indices, 0)
        .context("select recognizer emission logits")?;
    let confidence = ops::softmax(&selected_logits, 1)?
        .max(1)?
        .to_vec1::<f32>()
        .context("copy recognizer emission confidence to CPU")?;
    ensure!(
        confidence.len() == emission_count,
        "recognizer confidence count does not match emitted tokens"
    );
    let mut offset = 0;
    for decoded in &mut decoded {
        let count = decoded.tokens.len();
        if count == 0 {
            continue;
        }
        let mean = confidence[offset..offset + count].iter().sum::<f32>() / count as f32;
        decoded.confidence_milli = (mean.clamp(0.0, 1.0) * 1_000.0).round() as u16;
        offset += count;
    }
    Ok(decoded)
}


fn character_records(
    job: &RecognitionJob,
    decoded: &DecodedRecognizer,
    characters: &[String],
    batch_width: usize,
    image_width: u32,
    image_height: u32,
) -> Result<Vec<PpOcrCharacterRecord>> {
    let mut tokens = decoded
        .tokens
        .iter()
        .map(|token| {
            characters
                .get(token.token)
                .cloned()
                .map(|text| (token.timestep, text))
                .with_context(|| format!("recognizer token {} exceeds character list", token.token))
        })
        .collect::<Result<Vec<_>>>()?;
    while tokens
        .first()
        .is_some_and(|(_, text)| text.trim_start().is_empty())
    {
        tokens.remove(0);
    }
    if let Some((_, text)) = tokens.first_mut() {
        *text = text.trim_start().to_owned();
    }
    while tokens
        .last()
        .is_some_and(|(_, text)| text.trim_end().is_empty())
    {
        tokens.pop();
    }
    if let Some((_, text)) = tokens.last_mut() {
        *text = text.trim_end().to_owned();
    }
    ensure!(
        tokens
            .iter()
            .map(|(_, text)| text.as_str())
            .collect::<String>()
            == decoded.text.trim(),
        "recognizer character sequence does not match region text"
    );
    if tokens.is_empty() {
        return Ok(Vec::new());
    }

    let valid_time_steps = decoded
        .time_steps
        .checked_mul(job.resized_width)
        .context("recognizer timestep scaling overflowed")?
        .div_ceil(batch_width)
        .max(1);
    let position_time_steps =
        valid_time_steps.max(tokens.last().map(|(timestep, _)| timestep + 1).unwrap_or(1));
    let centers = tokens
        .iter()
        .map(|(timestep, _)| (*timestep as f64 + 0.5) / position_time_steps as f64)
        .collect::<Vec<_>>();
    let crop_width = f64::from(job.crop.image.width());
    let crop_height = f64::from(job.crop.image.height());
    let max_x = f64::from(image_width.saturating_sub(1));
    let max_y = f64::from(image_height.saturating_sub(1));
    let mut records = Vec::with_capacity(tokens.len());
    for (index, ((_, text), center)) in tokens.iter().zip(centers.iter()).enumerate() {
        let left = if index == 0 {
            0.0
        } else {
            (centers[index - 1] + center) * 0.5
        };
        let right = if index + 1 == centers.len() {
            1.0
        } else {
            (center + centers[index + 1]) * 0.5
        };
        let mapped = job.crop.map_output_quad([
            [left * crop_width, 0.0],
            [right * crop_width, 0.0],
            [right * crop_width, crop_height],
            [left * crop_width, crop_height],
        ])?;
        records.push(PpOcrCharacterRecord {
            order: u32::try_from(index + 1).context("OCR character order overflowed")?,
            quad_points: mapped.map(|[x, y]| {
                [
                    x.round().clamp(0.0, max_x) as i32,
                    y.round().clamp(0.0, max_y) as i32,
                ]
            }),
            source_text: text.clone(),
        });
    }
    Ok(records)
}

fn recognized_region_record(
    job: &RecognitionJob,
    decoded: &DecodedRecognizer,
    characters: &[String],
    batch_width: usize,
    image_width: u32,
    image_height: u32,
) -> Result<Option<(QuadI, String, u16, Vec<PpOcrCharacterRecord>)>> {
    let source_text = decoded.text.trim().to_owned();
    if source_text.is_empty() {
        return Ok(None);
    }
    let characters = character_records(
        job,
        decoded,
        characters,
        batch_width,
        image_width,
        image_height,
    )?;
    Ok(Some((
        job.quad,
        source_text,
        decoded.confidence_milli,
        characters,
    )))
}

fn finalize_recognition_records(
    records: Vec<Option<(QuadI, String, u16, Vec<PpOcrCharacterRecord>)>>,
) -> Result<Vec<PpOcrRegionRecord>> {
    let mut output = Vec::with_capacity(records.len());
    for (index, result) in records.into_iter().enumerate() {
        let Some((quad, source_text, confidence_milli, characters)) = result else {
            continue;
        };
        output.push(PpOcrRegionRecord::new(
            index as u32 + 1,
            quad,
            source_text,
            confidence_milli,
            characters,
        ));
    }
    Ok(output)
}

fn recognize_regions(
    image: &RgbImage,
    quads: &[QuadI],
    recognizer: &RecognizerKind,
    characters: &[String],
    device: &Device,
    normalizer: &ImageNormalizer,
    region_parallelism: usize,
    max_batch_pixels: usize,
    batch_mode: RecognitionBatchMode,
    cancellation: &CancellationToken,
) -> Result<Vec<PpOcrRegionRecord>> {
    ensure!(
        region_parallelism > 0,
        "region parallelism must be positive"
    );
    let phase = match batch_mode {
        RecognitionBatchMode::Ratio => "main",
        RecognitionBatchMode::ExactWidth => "rescue",
    };
    let jobs_started = Instant::now();
    let mut jobs = build_recognition_jobs(image, quads, cancellation)?;
    let crop_warp_ms = jobs_started.elapsed().as_millis();
    if jobs.is_empty() {
        trace_ocr(format_args!(
            "phase={phase} crop_warp={crop_warp_ms}ms forward=0ms regions=0"
        ));
        return Ok(Vec::new());
    }
    match batch_mode {
        RecognitionBatchMode::Ratio => {
            if region_parallelism > 1 {
                jobs.sort_by(|left, right| {
                    right
                        .resized_width
                        .max(MIN_RECOGNIZER_WIDTH)
                        .cmp(&left.resized_width.max(MIN_RECOGNIZER_WIDTH))
                        .then_with(|| left.index.cmp(&right.index))
                });
            }
        }
        RecognitionBatchMode::ExactWidth => {
            jobs.sort_by_key(|job| (job.resized_width, job.index));
        }
    }

    let mut records: Vec<Option<(QuadI, String, u16, Vec<PpOcrCharacterRecord>)>> =
        vec![None; jobs.len()];
    cancellation
        .check()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let forward_started = Instant::now();
    let mut batch_start = 0;
    while batch_start < jobs.len() {
        let batch_end = match batch_mode {
            RecognitionBatchMode::Ratio => {
                batch_start
                    + recognition_batch_len(
                        jobs[batch_start..].iter().map(|job| job.resized_width),
                        region_parallelism,
                        max_batch_pixels,
                    )
            }
            RecognitionBatchMode::ExactWidth => {
                let width = jobs[batch_start].resized_width;
                let mut group_end = batch_start + 1;
                while group_end < jobs.len() && jobs[group_end].resized_width == width {
                    group_end += 1;
                }
                batch_start
                    + exact_width_batch_len(width, region_parallelism, max_batch_pixels)
                        .min(group_end - batch_start)
            }
        };
        let batch = &jobs[batch_start..batch_end];
        let batch_width = batch
            .iter()
            .map(|job| job.resized_width)
            .max()
            .context("recognizer batch is empty")?
            .max(MIN_RECOGNIZER_WIDTH);
        let tensor = recognition_batch_tensor(batch, device, normalizer)?;
        let output = recognizer
            .forward(&tensor)
            .with_context(|| format!("recognize OCR batch with {} regions", batch.len()))?;
        let decoded = decode_recognizer_batch_detailed(&output, characters)?;
        cancellation
            .check()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        for (job, decoded) in batch.iter().zip(decoded) {
            records[job.index] = recognized_region_record(
                job,
                &decoded,
                characters,
                batch_width,
                image.width(),
                image.height(),
            )?;
        }
        batch_start = batch_end;
    }
    let forward_ms = forward_started.elapsed().as_millis();
    let records = finalize_recognition_records(records)?;
    trace_ocr(format_args!(
        "phase={phase} crop_warp={crop_warp_ms}ms forward={forward_ms}ms regions={}",
        records.len()
    ));
    Ok(records)
}

fn build_recognition_jobs(
    image: &RgbImage,
    quads: &[QuadI],
    cancellation: &CancellationToken,
) -> Result<Vec<RecognitionJob>> {
    cancellation
        .check()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let crops: Vec<RegionCrop> = if quads.len() >= PARALLEL_CROP_MIN_QUADS {
        let results = ocr_pool().install(|| {
            quads
                .par_iter()
                .enumerate()
                .map(|(index, &quad)| {
                    geometry::crop_region(image, quad)
                        .with_context(|| format!("crop OCR region {}", index + 1))
                })
                .collect::<Vec<_>>()
        });
        results.into_iter().collect::<Result<Vec<_>>>()?
    } else {
        quads
            .iter()
            .enumerate()
            .map(|(index, &quad)| {
                geometry::crop_region(image, quad)
                    .with_context(|| format!("crop OCR region {}", index + 1))
            })
            .collect::<Result<Vec<_>>>()?
    };
    cancellation
        .check()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let mut jobs = Vec::with_capacity(quads.len());
    let mut total_crop_pixels = 0u64;
    for (index, (quad, crop)) in quads.iter().copied().zip(crops).enumerate() {
        let crop_pixels = u64::from(crop.image.width())
            .checked_mul(u64::from(crop.image.height()))
            .context("OCR crop pixel count overflowed")?;
        total_crop_pixels = total_crop_pixels
            .checked_add(crop_pixels)
            .context("OCR crop pixel budget overflowed")?;
        ensure!(
            total_crop_pixels <= MAX_TOTAL_CROP_PIXELS,
            "OCR crop pixel budget exceeded"
        );
        let resized_width =
            recognizer_resize_width(crop.image.width(), crop.image.height())? as usize;
        jobs.push(RecognitionJob {
            index,
            quad,
            crop,
            resized_width,
        });
    }
    Ok(jobs)
}

/// Batch size for a group of same-width rescue crops: pad-free batches never
/// exceed the pixel budget, and every member keeps the `batch_width` a
/// single-region forward would have used.
fn exact_width_batch_len(width: usize, max_regions: usize, max_batch_pixels: usize) -> usize {
    let cap = max_batch_pixels / (48 * width.max(MIN_RECOGNIZER_WIDTH));
    cap.clamp(1, max_regions.max(1))
}

fn recognize_regions_with_retry(
    image: &RgbImage,
    quads: &[QuadI],
    recognizer: &RecognizerKind,
    characters: &[String],
    device: &Device,
    normalizer: &ImageNormalizer,
    region_parallelism: usize,
    max_batch_pixels: usize,
    cancellation: &CancellationToken,
) -> Result<Vec<PpOcrRegionRecord>> {
    let records = recognize_regions(
        image,
        quads,
        recognizer,
        characters,
        device,
        normalizer,
        region_parallelism,
        max_batch_pixels,
        RecognitionBatchMode::Ratio,
        cancellation,
    )?;
    let rescue_quads = recognition_rescue_quads(&records);
    if rescue_quads.is_empty() {
        return Ok(records);
    }

    let rescue_started = Instant::now();
    let rescues = recognize_regions(
        image,
        &rescue_quads,
        recognizer,
        characters,
        device,
        normalizer,
        region_parallelism,
        max_batch_pixels,
        RecognitionBatchMode::ExactWidth,
        cancellation,
    )?;
    trace_ocr(format_args!(
        "phase=rescue rescue_candidates={} rescue_ms={}ms",
        rescue_quads.len(),
        rescue_started.elapsed().as_millis()
    ));
    let mut replacements = Vec::new();
    for rescue in rescues {
        let rescue_index = rescue.order.saturating_sub(1) as usize;
        let Some(&quad) = rescue_quads.get(rescue_index) else {
            continue;
        };
        let original = records
            .iter()
            .filter(|record| quad_contains(quad, record.quad_points))
            .cloned()
            .collect::<Vec<_>>();
        if original.is_empty() || !rescue_improves_recognition(&original, &rescue) {
            continue;
        }
        replacements.push((quad, rescue));
    }
    if replacements.is_empty() {
        return Ok(records);
    }

    let mut resolved = Vec::with_capacity(records.len() + replacements.len());
    let mut inserted = vec![false; replacements.len()];
    for record in records {
        let replacement_index = replacements
            .iter()
            .enumerate()
            .filter(|(_, (quad, _))| quad_contains(*quad, record.quad_points))
            .max_by_key(|(_, (quad, _))| quad_area(*quad))
            .map(|(index, _)| index);
        let Some(replacement_index) = replacement_index else {
            resolved.push(record);
            continue;
        };
        if !inserted[replacement_index] {
            resolved.push(replacements[replacement_index].1.clone());
            inserted[replacement_index] = true;
        }
    }
    for (index, record) in resolved.iter_mut().enumerate() {
        record.order = u32::try_from(index + 1).unwrap_or(u32::MAX);
    }
    Ok(resolved)
}

fn recognition_rescue_quads(records: &[PpOcrRegionRecord]) -> Vec<QuadI> {
    let mut candidates = Vec::new();
    for start in 0..records.len() {
        let mut quad = records[start].quad_points;
        for end in start
            ..records
                .len()
                .min(start.saturating_add(RECOGNITION_RESCUE_MAX_SPAN))
        {
            if end > start {
                quad = bounding_quad(quad, records[end].quad_points);
            }
            if recognition_text_needs_rescue(&records[start..=end])
                && !candidates
                    .iter()
                    .any(|candidate| quad_contains(*candidate, quad))
            {
                candidates.retain(|candidate| !quad_contains(quad, *candidate));
                candidates.push(quad);
                if candidates.len() == MAX_RECOGNITION_RESCUE_CANDIDATES {
                    return candidates;
                }
            }
        }
    }
    candidates
}

fn bounding_quad(left: QuadI, right: QuadI) -> QuadI {
    let points = left.into_iter().chain(right);
    let (min_x, min_y, max_x, max_y) = points.fold(
        (i32::MAX, i32::MAX, i32::MIN, i32::MIN),
        |(min_x, min_y, max_x, max_y), [x, y]| {
            (min_x.min(x), min_y.min(y), max_x.max(x), max_y.max(y))
        },
    );
    [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y],
    ]
}

fn quad_area(quad: QuadI) -> i64 {
    let width = i64::from(quad[2][0] - quad[0][0]).max(0);
    let height = i64::from(quad[2][1] - quad[0][1]).max(0);
    width * height
}

fn quad_contains(container: QuadI, candidate: QuadI) -> bool {
    let [container_left, container_top] = container[0];
    let [container_right, container_bottom] = container[2];
    candidate.into_iter().all(|[x, y]| {
        x >= container_left && x <= container_right && y >= container_top && y <= container_bottom
    })
}

fn recognition_text_needs_rescue(records: &[PpOcrRegionRecord]) -> bool {
    if records.is_empty() {
        return false;
    }
    let text = records
        .iter()
        .map(|record| record.source_text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    recognition_text_needs_retry(&text) || recognition_has_lost_contraction(&text)
}

fn recognition_has_lost_contraction(text: &str) -> bool {
    let words = text
        .split_whitespace()
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    words.windows(2).any(|pair| {
        matches!(
            pair[0].to_ascii_lowercase().as_str(),
            "i" | "you"
                | "we"
                | "they"
                | "he"
                | "she"
                | "it"
                | "that"
                | "there"
                | "what"
                | "who"
                | "where"
                | "how"
        ) && matches!(
            pair[1]
                .trim_matches(|character: char| !character.is_ascii_alphabetic())
                .to_ascii_lowercase()
                .as_str(),
            "ll" | "re" | "ve" | "d" | "m" | "s" | "nt"
        )
    }) || words.iter().any(|word| {
        matches!(
            word.trim_matches(|character: char| !character.is_ascii_alphabetic())
                .to_ascii_lowercase()
                .as_str(),
            "ill"
                | "well"
                | "shell"
                | "hell"
                | "theyre"
                | "youre"
                | "were"
                | "ive"
                | "dont"
                | "cant"
                | "wont"
                | "isnt"
                | "arent"
                | "didnt"
                | "doesnt"
                | "shouldnt"
                | "couldnt"
                | "wouldnt"
        )
    })
}

fn rescue_improves_recognition(
    original: &[PpOcrRegionRecord],
    candidate: &PpOcrRegionRecord,
) -> bool {
    let original_text = original
        .iter()
        .map(|record| record.source_text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let original_confidence = original
        .iter()
        .map(|record| u32::from(record.confidence_milli))
        .sum::<u32>()
        / u32::try_from(original.len()).unwrap_or(1);
    let original_count = original_text
        .chars()
        .filter(|character| character.is_alphanumeric())
        .count();
    let candidate_count = candidate
        .source_text
        .chars()
        .filter(|character| character.is_alphanumeric())
        .count();
    candidate_count.saturating_mul(4) >= original_count.saturating_mul(3)
        && !recognition_has_lost_contraction(&candidate.source_text)
        && (candidate.confidence_milli > original_confidence as u16
            || recognition_has_lost_contraction(&original_text))
}

fn recognition_text_needs_retry(text: &str) -> bool {
    let mut frequencies = [0usize; 26];
    let mut ascii_letters = 0usize;
    let mut longest_run = 0usize;
    let mut current_run = 0usize;
    let mut previous = None;
    let mut has_whitespace = false;
    for character in text.chars() {
        has_whitespace |= character.is_whitespace();
        if !character.is_ascii_alphabetic() {
            previous = None;
            current_run = 0;
            continue;
        }
        let character = character.to_ascii_lowercase();
        frequencies[(character as u8 - b'a') as usize] += 1;
        ascii_letters += 1;
        if previous == Some(character) {
            current_run += 1;
        } else {
            previous = Some(character);
            current_run = 1;
        }
        longest_run = longest_run.max(current_run);
    }
    if ascii_letters < 4 {
        return false;
    }
    let dominant = frequencies.into_iter().max().unwrap_or(0);
    dominant.saturating_mul(5) >= ascii_letters.saturating_mul(3)
        || longest_run >= 3
            && (dominant.saturating_mul(5) >= ascii_letters.saturating_mul(2)
                || !has_whitespace && ascii_letters >= 14)
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_model_pair_tests {
    use super::*;
    use crate::backend::input::DecodedImage;
    use std::path::{Path, PathBuf};

    const MODEL_PAIRS: [(&str, &str, &str); 5] = [
        (
            "v5-server",
            "PP-OCRv5_server_det_safetensors",
            "PP-OCRv5_server_rec_safetensors",
        ),
        (
            "v5-mobile",
            "PP-OCRv5_mobile_det_safetensors",
            "PP-OCRv5_mobile_rec_safetensors",
        ),
        (
            "v6-tiny",
            "PP-OCRv6_tiny_det_safetensors",
            "PP-OCRv6_tiny_rec_safetensors",
        ),
        (
            "v6-small",
            "PP-OCRv6_small_det_safetensors",
            "PP-OCRv6_small_rec_safetensors",
        ),
        (
            "v6-medium",
            "PP-OCRv6_medium_det_safetensors",
            "PP-OCRv6_medium_rec_safetensors",
        ),
    ];

    fn local_model_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("PP-OCR")
            .canonicalize()
            .expect("canonicalize local PP-OCR model root")
    }

    fn fixture() -> DecodedImage {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("ppocrv5")
            .join("italic-dialogue.png");
        let image = image::open(&path)
            .unwrap_or_else(|error| panic!("open {}: {error}", path.display()))
            .to_rgb8();
        DecodedImage::from_rgb_image(image, "ppocr-cuda-fixture", "English")
    }

    const EXPECTED_DIALOGUE: &str = "thatstheonlyproblemyouhavewiththisheadline";

    fn normalized_alphanumeric(text: &str) -> String {
        text.chars()
            .filter(|character| character.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect()
    }

    fn assert_expected_dialogue(
        label: &str,
        route: &str,
        text: &str,
    ) -> std::result::Result<(), String> {
        if normalized_alphanumeric(text).contains(EXPECTED_DIALOGUE) {
            Ok(())
        } else {
            Err(format!(
                "{route} OCR did not contain the fixture dialogue for {label}: {text:?}"
            ))
        }
    }


    fn run_model_pair(
        label: &str,
        detector_dir: &str,
        recognizer_dir: &str,
        root: &Path,
        image: &DecodedImage,
        device: &Device,
        cancellation: &CancellationToken,
    ) -> std::result::Result<(String, String), String> {
        let mut provider = PpOcrProvider::load(
            &root.join(detector_dir),
            &root.join(recognizer_dir),
            device,
            1,
            48 * 320,
        )
        .map_err(|error| format!("load {label} on CUDA: {error:#}"))?;

        let document = provider
            .recognize(image, cancellation)
            .map_err(|error| format!("run {label} image OCR on CUDA: {error:#}"))?;
        let image_text = document
            .regions
            .iter()
            .map(|region| region.source_text.as_str())
            .collect::<String>();
        if !image_text.chars().any(|character| character.is_alphanumeric()) {
            return Err(format!(
                "image OCR produced no recognized text: {image_text:?}"
            ));
        }
        assert_expected_dialogue(label, "image", &image_text)?;

        // The original fixed rectangle cut through the fixture's glyphs.  A
        // detector-produced text quad is a valid real-time selection and keeps
        // this regression independent of incidental fixture layout.
        let live_quad = document
            .regions
            .iter()
            .max_by_key(|region| region.source_text.chars().count())
            .map(|region| region.quad_points)
            .ok_or_else(|| "image OCR produced no selectable text region".to_owned())?;
        let live_region = provider
            .recognize_quad(image, live_quad, cancellation)
            .map_err(|error| format!("run {label} live OCR on CUDA: {error:#}"))?
            .ok_or_else(|| "live OCR produced no region".to_owned())?;
        let live_text = live_region.source_text;
        if !live_text.chars().any(|character| character.is_alphanumeric()) {
            return Err(format!(
                "live OCR produced no recognized text: {live_text:?}"
            ));
        }
        assert_expected_dialogue(label, "live", &live_text)?;
        Ok((image_text, live_text))
    }

    #[test]
    #[ignore = "requires local PP-OCR assets and a CUDA device"]
    fn cuda_all_supported_model_pairs_run_image_and_live_ocr() {
        let device = Device::new_cuda(0).expect("create CUDA device; no CPU fallback is permitted");
        assert!(matches!(device, Device::Cuda(_)));
        let root = local_model_root();
        let image = fixture();
        let cancellation = CancellationToken::new_for_test();
        let mut failures = Vec::new();

        for (label, detector_dir, recognizer_dir) in MODEL_PAIRS {
            match run_model_pair(
                label,
                detector_dir,
                recognizer_dir,
                &root,
                &image,
                &device,
                &cancellation,
            ) {
                Ok((image_text, live_text)) => {
                    eprintln!(
                        "[cuda-ocr] {label}: image={image_text:?}; live={live_text:?}"
                    );
                }
                Err(error) => failures.push(format!("{label}: {error}")),
            }
        }

        assert!(
            failures.is_empty(),
            "CUDA OCR model-pair failures:\n{}",
            failures.join("\n")
        );
    }
}

