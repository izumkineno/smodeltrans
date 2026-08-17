//! Detector geometry, original-coordinate mapping, and official OCR crops.
//!
//! This module deliberately keeps geometry independent of the model adapter.  The
//! detector is allowed to emit either four-point polygons or explicit `xyxy`
//! boxes; everything after that boundary operates in original decoded-image
//! pixels.

use anyhow::{Context, Result, bail, ensure};
use image::{Rgb, RgbImage, imageops};
use std::cmp::Ordering;

pub type QuadI = [[i32; 2]; 4];
pub type QuadF = [[f64; 2]; 4];

const DETECTOR_LONGEST_SIDE: u32 = 960;
const DETECTOR_STRIDE: u32 = 32;
const EPSILON: f64 = 1e-9;
/// The one application-owned detector resize and its inverse coordinate scales.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DetectorProfile {
    pub original_width: u32,
    pub original_height: u32,
    pub detector_width: u32,
    pub detector_height: u32,
}
impl DetectorProfile {
    pub fn for_image(original_width: u32, original_height: u32) -> Result<Self> {
        ensure!(
            original_width > 0 && original_height > 0,
            "image dimensions must be positive"
        );
        let longest = original_width.max(original_height);
        let scale = if longest > DETECTOR_LONGEST_SIDE {
            f64::from(DETECTOR_LONGEST_SIDE) / f64::from(longest)
        } else {
            1.0
        };
        // PPOCRV5ServerDetImageProcessor rounds each input axis to the nearest
        // model stride before inference. Matching that geometry avoids an
        // unsupported detector shape for short subtitle bands.
        let scale_axis = |value: u32| (f64::from(value) * scale) as u32;
        let round_to_stride = |value: u32| {
            ((f64::from(value) / f64::from(DETECTOR_STRIDE))
                .round_ties_even()
                .max(1.0) as u32)
                .saturating_mul(DETECTOR_STRIDE)
        };
        let detector_width = round_to_stride(scale_axis(original_width));
        let detector_height = round_to_stride(scale_axis(original_height));
        Ok(Self {
            original_width,
            original_height,
            detector_width,
            detector_height,
        })
    }

    pub fn scale_x(self) -> f64 {
        f64::from(self.detector_width) / f64::from(self.original_width)
    }

    pub fn scale_y(self) -> f64 {
        f64::from(self.detector_height) / f64::from(self.original_height)
    }
}

/// Convert a detector-space four-point float polygon into a validated,
/// canonical original-image integer polygon.
pub fn map_detector_quad(raw: [[f32; 2]; 4], profile: DetectorProfile) -> Result<QuadI> {
    let sx = profile.scale_x();
    let sy = profile.scale_y();
    let max_x = f64::from(profile.original_width - 1);
    let max_y = f64::from(profile.original_height - 1);
    let mut mapped = [[0_i32; 2]; 4];

    for (index, point) in raw.into_iter().enumerate() {
        ensure!(
            point[0].is_finite() && point[1].is_finite(),
            "detector polygon contains a non-finite coordinate"
        );
        let x = f64::from(point[0]) / sx;
        let y = f64::from(point[1]) / sy;
        ensure!(
            (-EPSILON..=max_x + EPSILON).contains(&x) && (-EPSILON..=max_y + EPSILON).contains(&y),
            "detector polygon maps outside original image bounds"
        );
        let x = x.round();
        let y = y.round();
        ensure!(
            (0.0..=max_x).contains(&x) && (0.0..=max_y).contains(&y),
            "rounded detector polygon maps outside original image bounds"
        );
        mapped[index] = [x as i32, y as i32];
    }
    canonicalize_quad(mapped)
}

/// Canonicalize a valid integer quadrilateral to clockwise screen order with
/// the geometric top-left point first.
pub fn canonicalize_quad(points: QuadI) -> Result<QuadI> {
    for i in 0..4 {
        for j in (i + 1)..4 {
            ensure!(
                points[i] != points[j],
                "quadrilateral contains duplicate points"
            );
        }
    }
    ensure!(
        !has_self_intersection(points),
        "quadrilateral self-intersects"
    );

    let center = [
        points.iter().map(|point| f64::from(point[0])).sum::<f64>() / 4.0,
        points.iter().map(|point| f64::from(point[1])).sum::<f64>() / 4.0,
    ];
    let mut indexed = points
        .into_iter()
        .enumerate()
        .map(|(index, point)| {
            (
                (f64::from(point[1]) - center[1]).atan2(f64::from(point[0]) - center[0]),
                (f64::from(point[0]) - center[0]).hypot(f64::from(point[1]) - center[1]),
                point,
                index,
            )
        })
        .collect::<Vec<_>>();
    indexed.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal))
            .then_with(|| left.2[1].cmp(&right.2[1]))
            .then_with(|| left.2[0].cmp(&right.2[0]))
            .then_with(|| left.3.cmp(&right.3))
    });

    let mut ordered = indexed
        .into_iter()
        .map(|entry| entry.2)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap_or(points);
    if polygon_area(ordered).abs() <= 0.5 {
        bail!("quadrilateral has zero area");
    }
    if polygon_area(ordered) < 0.0 {
        ordered.reverse();
    }
    rotate_top_left(&mut ordered);
    ensure!(
        !has_self_intersection(ordered),
        "quadrilateral self-intersects"
    );
    ensure!(
        polygon_area(ordered).abs() > 0.5,
        "quadrilateral has zero area"
    );
    Ok(ordered)
}

fn rotate_top_left(points: &mut QuadI) {
    let start = (0..4)
        .min_by_key(|&index| {
            (
                i64::from(points[index][0]) + i64::from(points[index][1]),
                points[index][1],
                points[index][0],
            )
        })
        .unwrap_or(0);
    points.rotate_left(start);
}

/// Signed area in screen coordinates.  Positive means clockwise because the
/// y-axis increases downward.
fn polygon_area(points: QuadI) -> f64 {
    let mut area = 0.0;
    for index in 0..4 {
        let next = (index + 1) % 4;
        area += f64::from(points[index][0]) * f64::from(points[next][1])
            - f64::from(points[next][0]) * f64::from(points[index][1]);
    }
    area / 2.0
}

fn has_self_intersection(points: QuadI) -> bool {
    segments_intersect(points[0], points[1], points[2], points[3])
        || segments_intersect(points[1], points[2], points[3], points[0])
}

fn orientation(a: [i32; 2], b: [i32; 2], c: [i32; 2]) -> i64 {
    i64::from(b[0] - a[0]) * i64::from(c[1] - a[1])
        - i64::from(b[1] - a[1]) * i64::from(c[0] - a[0])
}

fn on_segment(a: [i32; 2], b: [i32; 2], p: [i32; 2]) -> bool {
    orientation(a, b, p) == 0
        && p[0] >= a[0].min(b[0])
        && p[0] <= a[0].max(b[0])
        && p[1] >= a[1].min(b[1])
        && p[1] <= a[1].max(b[1])
}

fn segments_intersect(a: [i32; 2], b: [i32; 2], c: [i32; 2], d: [i32; 2]) -> bool {
    let ab_c = orientation(a, b, c);
    let ab_d = orientation(a, b, d);
    let cd_a = orientation(c, d, a);
    let cd_b = orientation(c, d, b);
    (ab_c.signum() != ab_d.signum() && cd_a.signum() != cd_b.signum())
        || (ab_c == 0 && on_segment(a, b, c))
        || (ab_d == 0 && on_segment(a, b, d))
        || (cd_a == 0 && on_segment(c, d, a))
        || (cd_b == 0 && on_segment(c, d, b))
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn f64_area(points: QuadF) -> f64 {
    let mut area = 0.0;
    for index in 0..4 {
        let next = (index + 1) % 4;
        area += points[index][0] * points[next][1] - points[next][0] * points[index][1];
    }
    area / 2.0
}

fn canonicalize_float_quad(mut points: QuadF) -> QuadF {
    let center = [
        points.iter().map(|point| point[0]).sum::<f64>() / 4.0,
        points.iter().map(|point| point[1]).sum::<f64>() / 4.0,
    ];
    points.sort_by(|left, right| {
        (left[1] - center[1])
            .atan2(left[0] - center[0])
            .partial_cmp(&(right[1] - center[1]).atan2(right[0] - center[0]))
            .unwrap_or(Ordering::Equal)
            .then_with(|| left[1].partial_cmp(&right[1]).unwrap_or(Ordering::Equal))
            .then_with(|| left[0].partial_cmp(&right[0]).unwrap_or(Ordering::Equal))
    });
    if f64_area(points) < 0.0 {
        points.reverse();
    }
    let start = (0..4)
        .min_by(|&left, &right| {
            (points[left][0] + points[left][1])
                .partial_cmp(&(points[right][0] + points[right][1]))
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    points[left][1]
                        .partial_cmp(&points[right][1])
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| {
                    points[left][0]
                        .partial_cmp(&points[right][0])
                        .unwrap_or(Ordering::Equal)
                })
        })
        .unwrap_or(0);
    points.rotate_left(start);
    points
}

/// Find the minimum-area rectangle using all pair directions as candidates.
fn minimum_area_rect(points: QuadI) -> QuadF {
    let source = points.map(|point| [f64::from(point[0]), f64::from(point[1])]);
    let mut best_area = f64::INFINITY;
    let mut best = source;
    for i in 0..4 {
        for j in (i + 1)..4 {
            let dx = source[j][0] - source[i][0];
            let dy = source[j][1] - source[i][1];
            let length = dx.hypot(dy);
            if length <= EPSILON {
                continue;
            }
            let cos = dx / length;
            let sin = dy / length;
            let mut min_u = f64::INFINITY;
            let mut max_u = f64::NEG_INFINITY;
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            for point in source {
                let u = point[0] * cos + point[1] * sin;
                let v = -point[0] * sin + point[1] * cos;
                min_u = min_u.min(u);
                max_u = max_u.max(u);
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }
            let area = (max_u - min_u) * (max_v - min_v);
            if area < best_area {
                best_area = area;
                let local = [
                    [min_u, min_v],
                    [max_u, min_v],
                    [max_u, max_v],
                    [min_u, max_v],
                ];
                best = local.map(|point| {
                    [
                        point[0] * cos - point[1] * sin,
                        point[0] * sin + point[1] * cos,
                    ]
                });
            }
        }
    }
    canonicalize_float_quad(best)
}

fn solve_homography(source: QuadF, destination: QuadF) -> Result<[f64; 8]> {
    let mut matrix = [[0.0_f64; 9]; 8];
    for index in 0..4 {
        let x = destination[index][0];
        let y = destination[index][1];
        let u = source[index][0];
        let v = source[index][1];
        let row = index * 2;
        matrix[row] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        matrix[row + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }
    for column in 0..8 {
        let pivot = (column..8)
            .max_by(|&left, &right| {
                matrix[left][column]
                    .abs()
                    .partial_cmp(&matrix[right][column].abs())
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or(column);
        ensure!(
            matrix[pivot][column].abs() > EPSILON,
            "degenerate perspective transform"
        );
        matrix.swap(column, pivot);
        let divisor = matrix[column][column];
        for item in column..9 {
            matrix[column][item] /= divisor;
        }
        for row in 0..8 {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            if factor.abs() <= EPSILON {
                continue;
            }
            for item in column..9 {
                matrix[row][item] -= factor * matrix[column][item];
            }
        }
    }
    Ok(std::array::from_fn(|index| matrix[index][8]))
}

fn cubic_weight(value: f64) -> f64 {
    let x = value.abs();
    const A: f64 = -0.75;
    if x <= 1.0 {
        (A + 2.0) * x * x * x - (A + 3.0) * x * x + 1.0
    } else if x < 2.0 {
        A * x * x * x - 5.0 * A * x * x + 8.0 * A * x - 4.0 * A
    } else {
        0.0
    }
}

fn sample_cubic(image: &RgbImage, x: f64, y: f64) -> Rgb<u8> {
    let x = x.clamp(0.0, f64::from(image.width().saturating_sub(1)));
    let y = y.clamp(0.0, f64::from(image.height().saturating_sub(1)));
    let x_floor = x.floor() as i32;
    let y_floor = y.floor() as i32;
    let max_x = image.width().saturating_sub(1) as i32;
    let max_y = image.height().saturating_sub(1) as i32;
    let x_indices: [usize; 4] =
        std::array::from_fn(|index| (x_floor + index as i32 - 1).clamp(0, max_x) as usize);
    let y_indices: [usize; 4] =
        std::array::from_fn(|index| (y_floor + index as i32 - 1).clamp(0, max_y) as usize);
    let x_weights: [f64; 4] =
        std::array::from_fn(|index| cubic_weight(x - f64::from(x_floor + index as i32 - 1)));
    let y_weights: [f64; 4] =
        std::array::from_fn(|index| cubic_weight(y - f64::from(y_floor + index as i32 - 1)));
    let pixels = image.as_raw();
    let image_width = image.width() as usize;
    let mut channels = [0.0_f64; 3];
    let mut total = 0.0;
    for (dy, &py) in y_indices.iter().enumerate() {
        let row_offset = py * image_width * 3;
        for (dx, &px) in x_indices.iter().enumerate() {
            let weight = x_weights[dx] * y_weights[dy];
            let offset = row_offset + px * 3;
            channels[0] += weight * f64::from(pixels[offset]);
            channels[1] += weight * f64::from(pixels[offset + 1]);
            channels[2] += weight * f64::from(pixels[offset + 2]);
            total += weight;
        }
    }
    if total.abs() > EPSILON {
        for channel in &mut channels {
            *channel /= total;
        }
    }
    Rgb(channels.map(|value| value.round().clamp(0.0, 255.0) as u8))
}

/// A recognition crop plus the inverse transform needed to project recognizer
/// coordinates back onto the original image.
#[derive(Debug)]
pub struct RegionCrop {
    pub image: RgbImage,
    transform: [f64; 8],
    warped_width: u32,
    rotated: bool,
}

impl RegionCrop {
    pub fn map_output_point(&self, point: [f64; 2]) -> Result<[f64; 2]> {
        ensure!(
            point[0].is_finite() && point[1].is_finite(),
            "crop point contains a non-finite coordinate"
        );
        let (x, y) = if self.rotated {
            (f64::from(self.warped_width) - point[1], point[0])
        } else {
            (point[0], point[1])
        };
        let denominator = self.transform[6] * x + self.transform[7] * y + 1.0;
        ensure!(
            denominator.abs() > EPSILON,
            "invalid crop transform denominator"
        );
        Ok([
            (self.transform[0] * x + self.transform[1] * y + self.transform[2]) / denominator,
            (self.transform[3] * x + self.transform[4] * y + self.transform[5]) / denominator,
        ])
    }

    pub fn map_output_quad(&self, quad: QuadF) -> Result<QuadF> {
        let mut mapped = [[0.0_f64; 2]; 4];
        for (index, point) in quad.into_iter().enumerate() {
            mapped[index] = self.map_output_point(point)?;
        }
        Ok(mapped)
    }
}

/// Crop a region from the original RGB image using the official
/// min-area-rectangle → perspective-warp → tall-rotation sequence.
///
/// PaddleOCR uses OpenCV `INTER_CUBIC` for this warp. Keeping its $A=-0.75$
/// cubic kernel preserves narrow strokes, such as dialogue apostrophes, before
/// the recognizer normalizes each crop to 48 pixels high.
pub fn crop_region(image: &RgbImage, quad: QuadI) -> Result<RegionCrop> {
    ensure!(
        image.width() > 0 && image.height() > 0,
        "source image is empty"
    );
    let rectangle = minimum_area_rect(quad);
    let width = distance(rectangle[0], rectangle[1]).floor() as u32;
    let height = distance(rectangle[0], rectangle[3]).floor() as u32;
    ensure!(
        width > 0 && height > 0,
        "quadrilateral crop has zero dimensions"
    );
    let destination = [
        [0.0, 0.0],
        [f64::from(width), 0.0],
        [f64::from(width), f64::from(height)],
        [0.0, f64::from(height)],
    ];
    let transform =
        solve_homography(rectangle, destination).context("unable to solve crop transform")?;
    let mut warped = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let x = f64::from(x);
            let y = f64::from(y);
            let denominator = transform[6] * x + transform[7] * y + 1.0;
            ensure!(
                denominator.abs() > EPSILON,
                "invalid crop transform denominator"
            );
            let source_x = (transform[0] * x + transform[1] * y + transform[2]) / denominator;
            let source_y = (transform[3] * x + transform[4] * y + transform[5]) / denominator;
            warped.put_pixel(x as u32, y as u32, sample_cubic(image, source_x, source_y));
        }
    }
    let rotated = f64::from(height) / f64::from(width) >= 1.5;
    let image = if rotated {
        imageops::rotate270(&warped)
    } else {
        warped
    };
    Ok(RegionCrop {
        image,
        transform,
        warped_width: width,
        rotated,
    })
}
