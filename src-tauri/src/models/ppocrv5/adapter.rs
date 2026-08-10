//! PP-OCRv5 provider adapter and native detector/recognizer pipeline.

use super::{
    assets::{GraphRole, PpOcrV5Assets},
    geometry::{self, DetectorProfile, QuadI, RegionCrop},
    model::{self, PpOcrV5Detector, PpOcrV5Recognizer, RecognizerOutput},
    records::{PpOcrCharacterRecord, PpOcrRegionRecord},
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
use candle_core::{Device, IndexOp, Tensor};
use image::{GrayImage, Luma, RgbImage, imageops};
use imageproc::contours::find_contours;
use std::path::Path;

const DETECTOR_THRESHOLD: f32 = 0.30;
const MAX_DETECTOR_CANDIDATES: usize = 1000;
const DETECTOR_BOX_THRESHOLD: f32 = 0.60;
const UNCLIP_RATIO: f32 = 1.50;
const MIN_BOX_SIDE: f32 = 3.0;
const MAX_TOTAL_CROP_PIXELS: u64 = 64 * 1024 * 1024;
const RECOGNIZER_IMAGE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const RECOGNIZER_IMAGE_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// A loaded PP-OCRv5 detector and recognizer pair.
///
/// We load both graphs from the validated local safetensors assets before the
/// provider is exposed to the backend.  Requests only perform preprocessing,
/// native Candle inference, and deterministic postprocessing.
pub(crate) struct PpOcrV5Provider {
    detector: PpOcrV5Detector,
    recognizer: PpOcrV5Recognizer,
    characters: Vec<String>,
    device: Device,
    region_parallelism: usize,
}

impl PpOcrV5Provider {
    pub(crate) fn load(
        detector_dir: &Path,
        recognizer_dir: &Path,
        device: &Device,
        region_parallelism: usize,
    ) -> std::result::Result<Self, BackendFailure> {
        if region_parallelism == 0 {
            return Err(BackendFailure::arguments(
                "PP-OCRv5 region parallelism must be positive",
            ));
        }
        let detector_assets = PpOcrV5Assets::preflight(GraphRole::Detector, detector_dir)?;
        let recognizer_assets = PpOcrV5Assets::preflight(GraphRole::Recognizer, recognizer_dir)?;
        let characters = recognizer_assets.character_list.clone();
        let detector = PpOcrV5Detector::load(&detector_assets, device).map_err(|error| {
            BackendFailure::ocr(format!("load PP-OCRv5 detector graph: {error:#}"))
        })?;
        let recognizer = PpOcrV5Recognizer::load(&recognizer_assets, device).map_err(|error| {
            BackendFailure::ocr(format!("load PP-OCRv5 recognizer graph: {error:#}"))
        })?;
        Ok(Self {
            detector,
            recognizer,
            characters,
            device: device.clone(),
            region_parallelism,
        })
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
            1,
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

impl OcrPort for PpOcrV5Provider {
    fn recognize(
        &mut self,
        image: &DecodedImage,
        cancellation: &CancellationToken,
    ) -> std::result::Result<OcrDocument, BackendFailure> {
        cancellation.check()?;
        let canvas = image.canvas();
        let profile = DetectorProfile::for_image(canvas.width(), canvas.height())
            .map_err(|error| BackendFailure::ocr(format!("build detector profile: {error:#}")))?;
        let detector_input = detector_tensor(canvas, profile, &self.device)
            .map_err(|error| BackendFailure::ocr(format!("prepare detector input: {error:#}")))?;
        cancellation.check()?;
        let detector_output = self
            .detector
            .forward(&detector_input)
            .map_err(|error| BackendFailure::ocr(format!("run detector graph: {error:#}")))?;
        let quads = detector_quads(&detector_output, profile).map_err(|error| {
            BackendFailure::ocr(format!("postprocess detector output: {error:#}"))
        })?;
        cancellation.check()?;
        if quads.is_empty() {
            return Ok(OcrDocument {
                regions: Vec::new(),
            });
        }
        let records = recognize_regions_with_retry(
            canvas,
            &quads,
            &self.recognizer,
            &self.characters,
            &self.device,
            self.region_parallelism,
            cancellation,
        )
        .map_err(|error| BackendFailure::ocr(format!("recognize OCR regions: {error:#}")))?;
        cancellation.check()?;
        Ok(OcrDocument {
            regions: records
                .into_iter()
                .map(PpOcrRegionRecord::into_contract)
                .collect(),
        })
    }
}

fn detector_tensor(image: &RgbImage, profile: DetectorProfile, device: &Device) -> Result<Tensor> {
    let resized = imageops::resize(
        image,
        profile.detector_width,
        profile.detector_height,
        imageops::FilterType::Triangle,
    );
    let mut values = Vec::with_capacity((resized.width() * resized.height() * 3) as usize);
    for channel in 0..3 {
        for pixel in resized.pixels() {
            // The processor normalizes RGB channels, then swaps to BGR CHW.
            let source_channel = 2 - channel;
            let value = pixel[source_channel];
            let mean = [0.406_f32, 0.456, 0.485][source_channel];
            let std = [0.225_f32, 0.224, 0.229][source_channel];
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

fn detector_contours(
    probabilities: &[f32],
    width: usize,
    height: usize,
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
        if probability > DETECTOR_THRESHOLD {
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

fn detector_quads(output: &model::DetectorOutput, profile: DetectorProfile) -> Result<Vec<QuadI>> {
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
    let mut quads = Vec::new();
    for contour in detector_contours(&flat, width, height)?
        .into_iter()
        .take(MAX_DETECTOR_CANDIDATES)
    {
        // DB's cv2.minAreaRect is applied to each contour before scoring.
        let quad = component_quad(&contour);
        if quad_short_side(quad) < MIN_BOX_SIDE {
            continue;
        }
        let score = filled_quad_score(&flat, width, height, quad);
        if !score.is_finite() || score < DETECTOR_BOX_THRESHOLD {
            continue;
        }
        let unclipped = unclipped_quad(quad, UNCLIP_RATIO);
        if quad_short_side(unclipped) < MIN_BOX_SIDE + 2.0 {
            continue;
        }
        let clipped = clip_detector_quad(unclipped, profile);
        let raw = clipped.map(|[x, y]| [x * scale_x, y * scale_y]);
        quads.push(geometry::map_detector_quad(raw, profile)?);
    }
    Ok(quads)
}

#[derive(Debug)]
struct RecognitionJob {
    index: usize,
    quad: QuadI,
    crop: RegionCrop,
    resized_width: usize,
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
    // The local Transformers PP-OCRv5 recognizer first converts inputs to RGB,
    // then applies its inherited ImageNet normalization. The detector uses a
    // separate BGR contract, so do not share its preprocessing here.
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

#[cfg(test)]
fn recognition_tensor(crop: &RgbImage, device: &Device) -> Result<Tensor> {
    let resized_width = recognizer_resize_width(crop.width(), crop.height())? as usize;
    let tensor_width = resized_width.max(320);
    let mut values = vec![0.0_f32; 3usize * 48 * tensor_width];
    fill_recognition_tensor(crop, resized_width, tensor_width, &mut values)?;
    Tensor::from_vec(values, (1, 3, 48, tensor_width), device)
        .context("construct recognizer input tensor")
}

fn recognition_batch_tensor(jobs: &[&RecognitionJob], device: &Device) -> Result<Tensor> {
    ensure!(!jobs.is_empty(), "recognizer batch is empty");
    let batch_width = jobs
        .iter()
        .map(|job| job.resized_width)
        .max()
        .context("recognizer batch is empty")?
        .max(320);
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

fn decode_recognizer_row(logits: &Tensor, characters: &[String]) -> Result<DecodedRecognizer> {
    ensure!(
        logits.rank() == 2,
        "recognizer row must be [T, V], got shape {:?}",
        logits.dims()
    );
    let flat = logits.flatten_all()?.to_vec1::<f32>()?;
    let time = logits.dim(0)?;
    let vocab = logits.dim(1)?;
    let expected = time
        .checked_mul(vocab)
        .context("recognizer output shape overflows")?;
    ensure!(
        flat.len() == expected,
        "recognizer output tensor size does not match its shape"
    );
    let mut text = String::new();
    let mut tokens = Vec::new();
    let mut previous = 0usize;
    for (timestep, scores) in flat.chunks(vocab).enumerate() {
        let (token, _) = scores
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| right.0.cmp(&left.0))
            })
            .context("recognizer returned an empty vocabulary")?;
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
        time_steps: time,
    })
}

#[cfg(test)]
fn decode_recognizer(output: &RecognizerOutput, characters: &[String]) -> Result<String> {
    ensure!(
        output.shape.len() == 3 && output.shape[0] == 1 && output.shape[2] > 0,
        "recognizer output must be [1, T, V] with a positive vocabulary"
    );
    Ok(decode_recognizer_row(&output.tensor().i(0)?, characters)?.text)
}

fn decode_recognizer_batch_detailed(
    output: &RecognizerOutput,
    characters: &[String],
) -> Result<Vec<DecodedRecognizer>> {
    ensure!(
        output.shape.len() == 3 && output.shape[0] > 0 && output.shape[2] > 0,
        "recognizer output must be [B, T, V] with a positive vocabulary"
    );
    let mut decoded = Vec::with_capacity(output.shape[0]);
    for row in 0..output.shape[0] {
        decoded.push(decode_recognizer_row(&output.tensor().i(row)?, characters)?);
    }
    Ok(decoded)
}

#[cfg(test)]
fn decode_recognizer_batch(
    output: &RecognizerOutput,
    characters: &[String],
) -> Result<Vec<String>> {
    Ok(decode_recognizer_batch_detailed(output, characters)?
        .into_iter()
        .map(|decoded| decoded.text)
        .collect())
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
) -> Result<Option<(QuadI, String, Vec<PpOcrCharacterRecord>)>> {
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
    Ok(Some((job.quad, source_text, characters)))
}

fn finalize_recognition_records(
    records: Vec<Option<(QuadI, String, Vec<PpOcrCharacterRecord>)>>,
) -> Result<Vec<PpOcrRegionRecord>> {
    let mut output = Vec::with_capacity(records.len());
    for (index, result) in records.into_iter().enumerate() {
        let Some((quad, source_text, characters)) = result else {
            continue;
        };
        output.push(PpOcrRegionRecord::new(
            index as u32 + 1,
            quad,
            source_text,
            characters,
        ));
    }
    Ok(output)
}

fn recognize_regions(
    image: &RgbImage,
    quads: &[QuadI],
    recognizer: &PpOcrV5Recognizer,
    characters: &[String],
    device: &Device,
    region_parallelism: usize,
    cancellation: &CancellationToken,
) -> Result<Vec<PpOcrRegionRecord>> {
    ensure!(
        region_parallelism > 0,
        "region parallelism must be positive"
    );
    let mut jobs = Vec::with_capacity(quads.len());
    let mut total_crop_pixels = 0u64;
    cancellation
        .check()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    for (index, &quad) in quads.iter().enumerate() {
        let crop = geometry::crop_region(image, quad)
            .with_context(|| format!("crop OCR region {}", index + 1))?;
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
    if jobs.is_empty() {
        return Ok(Vec::new());
    }
    if region_parallelism > 1 {
        jobs.sort_by(|left, right| {
            right
                .resized_width
                .cmp(&left.resized_width)
                .then_with(|| left.index.cmp(&right.index))
        });
    }

    let mut records: Vec<Option<(QuadI, String, Vec<PpOcrCharacterRecord>)>> =
        vec![None; jobs.len()];
    cancellation
        .check()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    for chunk in jobs.chunks(region_parallelism) {
        let batch = chunk.iter().collect::<Vec<_>>();
        let batch_width = batch
            .iter()
            .map(|job| job.resized_width)
            .max()
            .context("recognizer batch is empty")?
            .max(320);
        let tensor = recognition_batch_tensor(&batch, device)?;
        let output = recognizer
            .forward(&tensor)
            .with_context(|| format!("recognize OCR batch with {} regions", batch.len()))?;
        let decoded = decode_recognizer_batch_detailed(&output, characters)?;
        cancellation
            .check()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        for (job, decoded) in batch.into_iter().zip(decoded) {
            records[job.index] = recognized_region_record(
                job,
                &decoded,
                characters,
                batch_width,
                image.width(),
                image.height(),
            )?;
        }
    }

    finalize_recognition_records(records)
}

fn recognize_regions_with_retry(
    image: &RgbImage,
    quads: &[QuadI],
    recognizer: &PpOcrV5Recognizer,
    characters: &[String],
    device: &Device,
    region_parallelism: usize,
    cancellation: &CancellationToken,
) -> Result<Vec<PpOcrRegionRecord>> {
    let mut records = recognize_regions(
        image,
        quads,
        recognizer,
        characters,
        device,
        region_parallelism,
        cancellation,
    )?;
    let retry_targets = records
        .iter()
        .enumerate()
        .filter(|(_, record)| recognition_text_needs_retry(&record.source_text))
        .map(|(record_index, record)| (record_index, recognition_retry_quad(record.quad_points)))
        .filter(|(record_index, retry_quad)| records[*record_index].quad_points != *retry_quad)
        .collect::<Vec<_>>();
    if retry_targets.is_empty() {
        return Ok(records);
    }
    let retry_quads = retry_targets
        .iter()
        .map(|(_, quad)| *quad)
        .collect::<Vec<_>>();
    let retries = recognize_regions(
        image,
        &retry_quads,
        recognizer,
        characters,
        device,
        region_parallelism.min(retry_quads.len()).max(1),
        cancellation,
    )?;
    for mut retry in retries {
        let retry_index = retry.order.saturating_sub(1) as usize;
        let Some((record_index, _)) = retry_targets.get(retry_index) else {
            continue;
        };
        let original = &records[*record_index];
        if retry_improves_recognition(&original.source_text, &retry.source_text) {
            retry.order = original.order;
            records[*record_index] = retry;
        }
    }
    Ok(records)
}

fn recognition_retry_quad(quad: QuadI) -> QuadI {
    let interpolate = |from: [i32; 2], to: [i32; 2]| {
        let coordinate =
            |from: i32, to: i32| ((i64::from(from) * 8 + i64::from(to) + 4) / 9) as i32;
        [coordinate(from[0], to[0]), coordinate(from[1], to[1])]
    };
    [
        interpolate(quad[0], quad[3]),
        interpolate(quad[1], quad[2]),
        interpolate(quad[2], quad[1]),
        interpolate(quad[3], quad[0]),
    ]
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

fn retry_improves_recognition(original: &str, candidate: &str) -> bool {
    let meaningful_count = |text: &str| {
        text.chars()
            .filter(|character| character.is_alphanumeric())
            .count()
    };
    let original_count = meaningful_count(original);
    let candidate_count = meaningful_count(candidate);
    candidate_count > 0
        && !recognition_text_needs_retry(candidate)
        && candidate_count.saturating_mul(3) >= original_count.saturating_mul(2)
}
#[cfg(test)]
mod tests {
    use super::{
        DecodedRecognizer, DecodedToken, DetectorProfile, OcrPort, PpOcrV5Provider, RecognitionJob,
        RecognizerOutput, character_records, clip_detector_quad, component_quad, decode_recognizer,
        decode_recognizer_batch, detector_contours, detector_tensor, filled_quad_score,
        finalize_recognition_records, recognition_retry_quad, recognition_tensor,
        recognition_text_needs_retry, recognized_region_record, recognizer_resize_width,
        retry_improves_recognition, unclipped_quad,
    };
    use candle_core::{Device, Tensor};
    use image::{Rgb, RgbImage};

    #[test]
    fn detector_tensor_matches_processor_bgr_channel_order() {
        let image = RgbImage::from_pixel(32, 32, Rgb([255, 128, 0]));
        let profile = DetectorProfile::for_image(32, 32).unwrap();
        let tensor = detector_tensor(&image, profile, &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[1, 3, 32, 32]);
        let values = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let plane = 32 * 32;
        assert!((values[0] - (0.0 - 0.485) / 0.229).abs() < 1e-6);
        assert!((values[plane] - (128.0 / 255.0 - 0.456) / 0.224).abs() < 1e-6);
        assert!((values[2 * plane] - (1.0 - 0.406) / 0.225).abs() < 1e-6);
    }

    #[test]
    fn detector_contours_use_strict_threshold() {
        let contours = detector_contours(&[0.30, 0.3001], 2, 1).unwrap();
        assert_eq!(contours, vec![vec![(1, 0)]]);
    }

    #[test]
    fn detector_contours_are_deterministic() {
        let values = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let first = detector_contours(&values, 6, 2).unwrap();
        let second = detector_contours(&values, 6, 2).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 2);
    }

    #[test]
    fn recognizer_tensor_matches_transformers_rgb_imagenet_normalization() {
        let crop = RgbImage::from_pixel(10, 10, Rgb([255, 0, 127]));
        let tensor = recognition_tensor(&crop, &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[1, 3, 48, 320]);
        let values = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let plane = 48 * 320;
        assert!((values[0] - (1.0 - 0.485) / 0.229).abs() < 1e-6);
        assert_eq!(values[48], 0.0);
        assert!((values[plane] - (0.0 - 0.456) / 0.224).abs() < 1e-6);
        assert!((values[2 * plane] - (127.0 / 255.0 - 0.406) / 0.225).abs() < 1e-6);
    }

    #[test]
    fn recognizer_tensor_keeps_wide_target_width() {
        let crop = RgbImage::from_pixel(1000, 100, Rgb([0, 0, 0]));
        let tensor = recognition_tensor(&crop, &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[1, 3, 48, 480]);
    }
    #[test]
    fn recognizer_target_width_matches_processor_rounding() {
        assert_eq!(recognizer_resize_width(10, 10).unwrap(), 48);
        assert_eq!(recognizer_resize_width(101, 10).unwrap(), 484);
        assert_eq!(recognizer_resize_width(667, 100).unwrap(), 320);
    }

    #[test]
    fn decorative_frame_retry_insets_only_the_vertical_edges() {
        assert_eq!(
            recognition_retry_quad([[0, 0], [470, 0], [470, 36], [0, 36]]),
            [[0, 4], [470, 4], [470, 32], [0, 32]]
        );
    }

    #[test]
    fn repeated_letter_noise_triggers_a_bounded_retry() {
        for text in ["eeeee", "eere", "cereree eee", "Newspaperhalaaanaaa"] {
            assert!(
                recognition_text_needs_retry(text),
                "{text:?} should request a retry"
            );
        }
        for text in ["Newspaper headlines", "committee", "Mad scientist"] {
            assert!(
                !recognition_text_needs_retry(text),
                "{text:?} should remain unchanged"
            );
        }
        assert!(retry_improves_recognition(
            "Newspaperhalaaanaaa",
            "Newspaper headlines"
        ));
        assert!(!retry_improves_recognition(
            "Newspaperhalaaanaaa",
            "Newspaper"
        ));
    }

    #[test]
    fn detector_quad_uses_minimum_area_orientation() {
        let quad = component_quad(&[(0, 1), (1, 0), (2, 1), (1, 2)]);
        let area = (0..4)
            .map(|index| {
                let next = (index + 1) % 4;
                quad[index][0] * quad[next][1] - quad[next][0] * quad[index][1]
            })
            .sum::<f32>()
            .abs()
            * 0.5;
        assert!(
            (area - 2.0).abs() < 1e-4,
            "minimum-area rectangle area: {area}"
        );
    }

    #[test]
    fn detector_db_score_and_unclip_are_deterministic() {
        let mut probabilities = vec![0.0_f32; 25];
        for y in 1..=3 {
            for x in 1..=3 {
                probabilities[y * 5 + x] = 0.75;
            }
        }
        let quad = [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]];
        assert!((filled_quad_score(&probabilities, 5, 5, quad) - 0.75).abs() < 1e-6);

        let expanded = unclipped_quad([[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [0.0, 4.0]], 1.5);
        assert!(expanded[0][0] < -2.0 && expanded[0][1] < -2.0);
        assert!(expanded[2][0] > 12.0 && expanded[2][1] > 6.0);
        let mut fractional = vec![0.0_f32; 16];
        fractional[0] = 1.0;
        let fractional_quad = [[0.9, 0.9], [2.9, 0.9], [2.9, 2.9], [0.9, 2.9]];
        let score = filled_quad_score(&fractional, 4, 4, fractional_quad);
        assert!((score - (1.0 / 9.0)).abs() < 1e-6);
    }

    #[test]
    fn detector_unclip_is_clipped_before_mapping_back_to_source() {
        let profile = DetectorProfile::for_image(1920, 1080).unwrap();
        let expanded = [[-8.0, 40.0], [968.0, 40.0], [968.0, 120.0], [-8.0, 120.0]];
        let clipped = clip_detector_quad(expanded, profile);
        assert_eq!(
            clipped,
            [[0.0, 40.0], [959.0, 40.0], [959.0, 120.0], [0.0, 120.0]]
        );
        let raw =
            clipped.map(|[x, y]| [x * profile.scale_x() as f32, y * profile.scale_y() as f32]);
        assert!(super::geometry::map_detector_quad(raw, profile).is_ok());
    }

    fn recognizer_output(rows: &[[f32; 3]]) -> RecognizerOutput {
        let values = rows
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>();
        let tensor = Tensor::from_vec(values, (1, rows.len(), 3), &Device::Cpu).unwrap();
        RecognizerOutput {
            logits: tensor,
            shape: vec![1, rows.len(), 3],
        }
    }
    fn recognizer_batch_output(rows: &[&[[f32; 3]]]) -> RecognizerOutput {
        let batch = rows.len();
        let time = rows.first().map(|row| row.len()).unwrap();
        let values = rows
            .iter()
            .flat_map(|row| row.iter().flat_map(|scores| scores.iter().copied()))
            .collect::<Vec<_>>();
        let tensor = Tensor::from_vec(values, (batch, time, 3), &Device::Cpu).unwrap();
        RecognizerOutput {
            logits: tensor,
            shape: vec![batch, time, 3],
        }
    }

    #[test]
    fn ctc_batch_decode_preserves_row_order() {
        let output = recognizer_batch_output(&[
            &[[0.1, 0.9, 0.0], [0.1, 0.9, 0.0]],
            &[[0.1, 0.0, 0.9], [0.1, 0.0, 0.9]],
        ]);
        let characters = vec!["blank".to_owned(), "A".to_owned(), "B".to_owned()];
        let texts = decode_recognizer_batch(&output, &characters).unwrap();
        assert_eq!(texts, vec!["A".to_owned(), "B".to_owned()]);
    }

    #[test]
    fn empty_recognition_text_is_skipped_without_character_geometry() {
        let image = RgbImage::new(100, 40);
        let crop =
            super::geometry::crop_region(&image, [[10, 10], [90, 10], [90, 30], [10, 30]]).unwrap();
        let job = RecognitionJob {
            index: 0,
            quad: [[10, 10], [90, 10], [90, 30], [10, 30]],
            crop,
            resized_width: 320,
        };
        let decoded = DecodedRecognizer {
            text: " \n ".to_owned(),
            tokens: Vec::new(),
            time_steps: 1,
        };

        let result =
            recognized_region_record(&job, &decoded, &["blank".to_owned()], 320, 100, 40).unwrap();

        assert!(result.is_none());
    }

    #[test]
    fn finalizing_records_drops_unrecognized_region_slots() {
        let quad = [[10, 10], [90, 10], [90, 30], [10, 30]];
        let records = finalize_recognition_records(vec![
            Some((quad, "first".to_owned(), Vec::new())),
            None,
            Some((quad, "third".to_owned(), Vec::new())),
        ])
        .unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].order, 1);
        assert_eq!(records[0].source_text, "first");
        assert_eq!(records[1].order, 3);
        assert_eq!(records[1].source_text, "third");
    }

    #[test]
    fn ctc_decode_collapses_without_trimming_and_prefers_first_tie() {
        let output = recognizer_output(&[
            [0.1, 0.4, 0.4],
            [0.1, 0.9, 0.1],
            [0.9, 0.1, 0.1],
            [0.1, 0.1, 0.9],
            [0.1, 0.1, 0.9],
        ]);
        let characters = vec!["blank".to_owned(), " A ".to_owned(), "B ".to_owned()];
        let text = decode_recognizer(&output, &characters).unwrap();
        assert_eq!(text, " A B ");
    }
    #[test]
    fn character_records_partition_the_source_region_in_ctc_order() {
        let image = RgbImage::new(100, 40);
        let crop =
            super::geometry::crop_region(&image, [[10, 10], [90, 10], [90, 30], [10, 30]]).unwrap();
        let job = RecognitionJob {
            index: 0,
            quad: [[10, 10], [90, 10], [90, 30], [10, 30]],
            crop,
            resized_width: 320,
        };
        let decoded = DecodedRecognizer {
            text: "AB".to_owned(),
            tokens: vec![
                DecodedToken {
                    token: 1,
                    timestep: 5,
                },
                DecodedToken {
                    token: 2,
                    timestep: 25,
                },
            ],
            time_steps: 40,
        };
        let characters = vec!["blank".to_owned(), "A".to_owned(), "B".to_owned()];

        let records = character_records(&job, &decoded, &characters, 320, 100, 40).unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].source_text, "A");
        assert_eq!(records[1].source_text, "B");
        assert_eq!(records[0].quad_points[0], [10, 10]);
        assert_eq!(records[0].quad_points[2], [41, 30]);
        assert_eq!(records[1].quad_points[0], [41, 10]);
        assert_eq!(records[1].quad_points[2], [90, 30]);
    }

    #[cfg(all(feature = "cuda", feature = "flash-attn"))]
    #[test]
    #[ignore = "requires staged PP-OCRv5 assets and a supported CUDA device"]
    fn cuda_live_ocr_fixtures_recognize_headline_text() {
        assert_eq!(
            std::env::var("SMODELTRANS_RUN_CUDA_E2E").as_deref(),
            Ok("1"),
            "set SMODELTRANS_RUN_CUDA_E2E=1 to run the native CUDA fixture"
        );
        let manifest_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let model_root = manifest_root.join("..").join("models").join("ppocrv5");
        let device = Device::new_cuda(0).expect("CUDA device");
        let mut provider = PpOcrV5Provider::load(
            &model_root.join("detector"),
            &model_root.join("recognizer"),
            &device,
            4,
        )
        .expect("PP-OCRv5 provider");
        let cancellation = super::CancellationToken::new_for_test();

        let newspaper_image =
            image::open(manifest_root.join("tests/fixtures/ppocrv5/newspaper-headlines.png"))
                .expect("newspaper fixture")
                .to_rgb8();
        let newspaper = super::DecodedImage::from_rgb_image(
            newspaper_image,
            "newspaper-headlines.png",
            "English",
        );
        let newspaper_text = provider
            .recognize(&newspaper, &cancellation)
            .expect("newspaper OCR fixture")
            .regions
            .into_iter()
            .map(|region| region.source_text)
            .collect::<Vec<_>>()
            .join("\n");
        for expected in [
            "Newspaper headlines",
            "Adoptions wanted! How to help the..",
            "Mad scientist on the loose! Townsfolks..",
            "A Tale of Two Cities: Maids,",
        ] {
            assert!(
                newspaper_text
                    .lines()
                    .any(|line| line.eq_ignore_ascii_case(expected)),
                "missing {expected:?} in:\n{newspaper_text}"
            );
        }
        let rescued_header = provider
            .recognize_quad(
                &newspaper,
                [[0, 0], [470, 0], [470, 36], [0, 36]],
                &cancellation,
            )
            .expect("recognize framed headline")
            .expect("framed headline text");
        assert_eq!(rescued_header.source_text, "Newspaper headlines");

        let dialogue_image =
            image::open(manifest_root.join("tests/fixtures/ppocrv5/headline-dialogue.png"))
                .expect("dialogue fixture")
                .to_rgb8();
        let dialogue =
            super::DecodedImage::from_rgb_image(dialogue_image, "headline-dialogue.png", "English");
        let dialogue_text = provider
            .recognize(&dialogue, &cancellation)
            .expect("dialogue OCR fixture")
            .regions
            .into_iter()
            .map(|region| region.source_text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            dialogue_text
                .lines()
                .any(|line| line == "That's the only problem you have with this headline?!?!"),
            "recognized text:\n{dialogue_text}"
        );
    }
    #[cfg(all(feature = "cuda", feature = "flash-attn"))]
    #[test]
    #[ignore = "requires staged PP-OCRv5 assets and a supported CUDA device"]
    fn cuda_italic_dialogue_fixture() {
        assert_eq!(
            std::env::var("SMODELTRANS_RUN_CUDA_E2E").as_deref(),
            Ok("1"),
            "set SMODELTRANS_RUN_CUDA_E2E=1 to run the native CUDA fixture"
        );
        let manifest_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let model_root = manifest_root.join("..").join("models").join("ppocrv5");
        let device = Device::new_cuda(0).expect("CUDA device");
        let mut provider = PpOcrV5Provider::load(
            &model_root.join("detector"),
            &model_root.join("recognizer"),
            &device,
            4,
        )
        .expect("PP-OCRv5 provider");
        let cancellation = super::CancellationToken::new_for_test();
        let image = image::open(manifest_root.join("tests/fixtures/ppocrv5/italic-dialogue.png"))
            .expect("italic dialogue fixture")
            .to_rgb8();
        let decoded = super::DecodedImage::from_rgb_image(image, "italic-dialogue.png", "English");
        let text = provider
            .recognize(&decoded, &cancellation)
            .expect("italic dialogue OCR fixture")
            .regions
            .into_iter()
            .map(|region| region.source_text)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(
            text.lines().any(|line| line
                .eq_ignore_ascii_case("That's the only problem you have with this headline?!?!")),
            "recognized text:\n{text}"
        );
    }
    #[cfg(all(feature = "cuda", feature = "flash-attn"))]
    #[test]
    #[ignore = "requires staged PP-OCRv5 assets and a supported CUDA device"]
    fn cuda_low_contrast_multiline_newspaper_fixture() {
        assert_eq!(
            std::env::var("SMODELTRANS_RUN_CUDA_E2E").as_deref(),
            Ok("1"),
            "set SMODELTRANS_RUN_CUDA_E2E=1 to run the native CUDA fixture"
        );
        let manifest_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let model_root = manifest_root.join("..").join("models").join("ppocrv5");
        let device = Device::new_cuda(0).expect("CUDA device");
        let mut provider = PpOcrV5Provider::load(
            &model_root.join("detector"),
            &model_root.join("recognizer"),
            &device,
            4,
        )
        .expect("PP-OCRv5 provider");
        let cancellation = super::CancellationToken::new_for_test();
        let image =
            image::open(manifest_root.join("tests/fixtures/ppocrv5/low-contrast-newspaper.png"))
                .expect("low-contrast newspaper fixture")
                .to_rgb8();
        let decoded =
            super::DecodedImage::from_rgb_image(image, "low-contrast-newspaper.png", "English");
        let text = provider
            .recognize(&decoded, &cancellation)
            .expect("low-contrast newspaper OCR fixture")
            .regions
            .into_iter()
            .map(|region| region.source_text)
            .collect::<Vec<_>>()
            .join("\n")
            .to_ascii_lowercase();
        println!("{text}");

        for expected in [
            "mad scientist on the loose",
            "cantamille afraid to leave their",
            "homes after local loon escapes prison",
            "scantily-clad",
            "accomplice",
        ] {
            assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
        }
    }
}
