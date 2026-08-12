use super::contracts::DEFAULT_STABILITY_WAIT_MS;
use crate::backend::contracts::RegionRecord;

use std::{
    sync::{Condvar, Mutex},
    time::Duration,
};

const SIGNATURE_COLUMNS: usize = 48;
const SIGNATURE_ROWS: usize = 28;
const SIGNATURE_SIZE: usize = SIGNATURE_COLUMNS * SIGNATURE_ROWS;
const MEAN_CHANGE_THRESHOLD: f32 = 0.012;
const CELL_CHANGE_THRESHOLD: u8 = 18;
const CHANGED_CELL_RATIO: f32 = 0.02;
#[derive(Debug)]
pub(super) struct OwnedFrame {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) rgb: Vec<u8>,
    pub(super) observed_at_epoch_ms: u64,
    pub(super) roi: crate::backend::live::contracts::LiveRoi,
    pub(super) roi_version: u64,
}

#[derive(Debug, Default)]
pub(super) struct LatestFrameSlot {
    frame: Mutex<Option<OwnedFrame>>,
    changed: Condvar,
}

impl LatestFrameSlot {
    pub(super) fn replace(&self, frame: OwnedFrame) -> bool {
        let mut current = self.frame.lock().unwrap_or_else(|error| error.into_inner());
        let dropped = current.replace(frame).is_some();
        self.changed.notify_one();
        dropped
    }

    pub(super) fn wait_take(&self, timeout: Duration) -> Option<OwnedFrame> {
        let mut current = self.frame.lock().unwrap_or_else(|error| error.into_inner());
        if current.is_none() {
            let (next, _) = self
                .changed
                .wait_timeout(current, timeout)
                .unwrap_or_else(|error| error.into_inner());
            current = next;
        }
        current.take()
    }

    pub(super) fn clear(&self) {
        *self.frame.lock().unwrap_or_else(|error| error.into_inner()) = None;
    }

    pub(super) fn wake(&self) {
        self.changed.notify_all();
    }
}

#[derive(Debug)]
pub(super) struct StabilityScheduler {
    settle_ms: u64,
    signature: Option<[u8; SIGNATURE_SIZE]>,
    stable_since_ms: Option<u64>,
    stable_observations: u8,
    suppressed_until_change: bool,
}

impl Default for StabilityScheduler {
    fn default() -> Self {
        Self::new(DEFAULT_STABILITY_WAIT_MS)
    }
}

impl StabilityScheduler {
    pub(super) fn new(settle_ms: u64) -> Self {
        Self {
            settle_ms,
            signature: None,
            stable_since_ms: None,
            stable_observations: 0,
            suppressed_until_change: false,
        }
    }

    pub(super) fn set_settle_ms(&mut self, settle_ms: u64) {
        if self.settle_ms == settle_ms {
            return;
        }
        self.settle_ms = settle_ms;
        self.reset();
    }

    pub(super) fn observe(&mut self, frame: &OwnedFrame, now_ms: u64) -> bool {
        let Some(signature) = luminance_signature(frame) else {
            self.reset();
            return false;
        };
        let Some(previous) = self.signature.as_ref() else {
            self.signature = Some(signature);
            self.stable_since_ms = Some(now_ms);
            self.stable_observations = 1;
            return false;
        };
        let changed = signatures_differ(previous, &signature);
        self.signature = Some(signature);
        if changed {
            self.stable_since_ms = Some(now_ms);
            self.stable_observations = 1;
            self.suppressed_until_change = false;
        } else {
            self.stable_observations = self.stable_observations.saturating_add(1);
        }
        self.should_probe(now_ms)
    }

    pub(super) fn tick(&mut self, now_ms: u64) -> bool {
        self.should_probe(now_ms)
    }

    pub(super) fn reset(&mut self) {
        let settle_ms = self.settle_ms;
        *self = Self::new(settle_ms);
    }

    fn should_probe(&mut self, now_ms: u64) -> bool {
        if self.suppressed_until_change || self.signature.is_none() || self.stable_observations < 2
        {
            return false;
        }
        let settled = self
            .stable_since_ms
            .is_some_and(|since| now_ms.saturating_sub(since) >= self.settle_ms);
        if settled {
            self.suppressed_until_change = true;
            true
        } else {
            false
        }
    }
}

fn luminance_signature(frame: &OwnedFrame) -> Option<[u8; SIGNATURE_SIZE]> {
    let width = usize::try_from(frame.width).ok()?;
    let height = usize::try_from(frame.height).ok()?;
    let expected_len = width.checked_mul(height)?.checked_mul(3)?;
    if width == 0 || height == 0 || frame.rgb.len() < expected_len {
        return None;
    }
    let mut signature = [0_u8; SIGNATURE_SIZE];
    for row in 0..SIGNATURE_ROWS {
        let y = ((row * height) / SIGNATURE_ROWS).min(height - 1);
        for column in 0..SIGNATURE_COLUMNS {
            let x = ((column * width) / SIGNATURE_COLUMNS).min(width - 1);
            let offset = (y * width + x) * 3;
            let red = u32::from(frame.rgb[offset]);
            let green = u32::from(frame.rgb[offset + 1]);
            let blue = u32::from(frame.rgb[offset + 2]);
            signature[row * SIGNATURE_COLUMNS + column] =
                ((red * 77 + green * 150 + blue * 29) >> 8) as u8;
        }
    }
    Some(signature)
}

fn signatures_differ(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() || left.is_empty() {
        return true;
    }
    let mut difference = 0_u64;
    let mut changed_cells = 0_usize;
    for (&left, &right) in left.iter().zip(right) {
        let cell_difference = left.abs_diff(right);
        difference += u64::from(cell_difference);
        changed_cells += usize::from(cell_difference >= CELL_CHANGE_THRESHOLD);
    }
    let mean_difference = difference as f32 / (left.len() as f32 * 255.0);
    let changed_ratio = changed_cells as f32 / left.len() as f32;
    mean_difference > MEAN_CHANGE_THRESHOLD || changed_ratio > CHANGED_CELL_RATIO
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TextBounds {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

impl TextBounds {
    fn from_quad(quad: [[i32; 2]; 4]) -> Option<Self> {
        let left = quad.iter().map(|point| point[0]).min()?;
        let top = quad.iter().map(|point| point[1]).min()?;
        let right = quad.iter().map(|point| point[0]).max()?;
        let bottom = quad.iter().map(|point| point[1]).max()?;
        (right > left && bottom > top).then_some(Self {
            left,
            top,
            right,
            bottom,
        })
    }

    fn width(self) -> i32 {
        self.right - self.left
    }

    fn height(self) -> i32 {
        self.bottom - self.top
    }

    fn center_y(self) -> f64 {
        f64::from(self.top + self.bottom) / 2.0
    }

    fn area(self) -> i64 {
        i64::from(self.width()) * i64::from(self.height())
    }

    fn overlap_height(self, other: Self) -> i32 {
        (self.bottom.min(other.bottom) - self.top.max(other.top)).max(0)
    }

    fn intersection_area(self, other: Self) -> i64 {
        let width = (self.right.min(other.right) - self.left.max(other.left)).max(0);
        let height = self.overlap_height(other);
        i64::from(width) * i64::from(height)
    }

    fn union(self, other: Self) -> Self {
        Self {
            left: self.left.min(other.left),
            top: self.top.min(other.top),
            right: self.right.max(other.right),
            bottom: self.bottom.max(other.bottom),
        }
    }

    fn into_quad(self) -> [[i32; 2]; 4] {
        [
            [self.left, self.top],
            [self.right, self.top],
            [self.right, self.bottom],
            [self.left, self.bottom],
        ]
    }
}

#[derive(Debug)]
struct PreparedRegion {
    record: RegionRecord,
    bounds: TextBounds,
    original_order: u32,
}

impl PreparedRegion {
    fn estimated_character_width(&self) -> f64 {
        let count = self
            .record
            .source_text
            .chars()
            .filter(|character| !character.is_whitespace())
            .count()
            .max(1);
        f64::from(self.bounds.width()) / count as f64
    }
}

#[derive(Debug)]
struct VisualLine {
    regions: Vec<PreparedRegion>,
    mean_center_y: f64,
    mean_height: f64,
}

impl VisualLine {
    fn new(region: PreparedRegion) -> Self {
        Self {
            mean_center_y: region.bounds.center_y(),
            mean_height: f64::from(region.bounds.height()),
            regions: vec![region],
        }
    }

    fn accepts(&self, region: &PreparedRegion) -> bool {
        let height = self.mean_height.min(f64::from(region.bounds.height()));
        if height <= 0.0 {
            return false;
        }
        let line_bounds = self
            .regions
            .iter()
            .map(|candidate| candidate.bounds)
            .reduce(TextBounds::union)
            .expect("visual line always has a region");
        let overlap = f64::from(line_bounds.overlap_height(region.bounds)) / height;
        let center_delta = (self.mean_center_y - region.bounds.center_y()).abs() / height;
        overlap >= 0.45 || center_delta <= 0.45
    }

    fn push(&mut self, region: PreparedRegion) {
        let count = self.regions.len() as f64;
        self.mean_center_y =
            (self.mean_center_y * count + region.bounds.center_y()) / (count + 1.0);
        self.mean_height =
            (self.mean_height * count + f64::from(region.bounds.height())) / (count + 1.0);
        self.regions.push(region);
    }

    fn leftmost(&self) -> i32 {
        self.regions
            .iter()
            .map(|region| region.bounds.left)
            .min()
            .unwrap_or_default()
    }
}

/// A visual line or tightly connected OCR fragments in deterministic reading order.
#[derive(Debug)]
pub(super) struct LiveOcrGroup {
    regions: Vec<RegionRecord>,
    quad: [[i32; 2]; 4],
}

impl LiveOcrGroup {
    pub(super) fn source_text(&self) -> String {
        join_fragments(
            self.regions
                .iter()
                .map(|region| region.source_text.as_str()),
        )
    }

    #[cfg(test)]
    pub(super) fn into_regions(self) -> Vec<RegionRecord> {
        self.regions
    }

    pub(super) fn into_merged_region(self, source_text: String) -> RegionRecord {
        let confidence_milli = self
            .regions
            .iter()
            .filter_map(|region| (region.confidence_milli > 0).then_some(region.confidence_milli))
            .min()
            .unwrap_or(0);
        let mut characters = self
            .regions
            .into_iter()
            .flat_map(|region| region.characters)
            .collect::<Vec<_>>();
        for (index, character) in characters.iter_mut().enumerate() {
            character.order = u32::try_from(index + 1).unwrap_or(u32::MAX);
        }
        RegionRecord {
            order: 0,
            quad_points: self.quad,
            source_text: normalize_text(&source_text),
            confidence_milli,
            translated_text: String::new(),
            characters,
        }
    }
}

const DUPLICATE_OVERLAP_PERCENT: i64 = 85;
const MIN_FRAGMENT_GAP_PX: i32 = 4;
const MIN_FRAGMENT_GAP_HEIGHT_RATIO: f64 = 1.25;
const MAX_FRAGMENT_GAP_HEIGHTS: i32 = 2;
const MAX_FRAGMENT_OVERLAP_HEIGHT_DIVISOR: i32 = 3;

fn compare_regions(left: &PreparedRegion, right: &PreparedRegion) -> std::cmp::Ordering {
    (
        left.bounds.top,
        left.bounds.left,
        left.bounds.bottom,
        left.bounds.right,
        left.original_order,
    )
        .cmp(&(
            right.bounds.top,
            right.bounds.left,
            right.bounds.bottom,
            right.bounds.right,
            right.original_order,
        ))
}

fn compact_text(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn duplicate_region(left: &PreparedRegion, right: &PreparedRegion) -> bool {
    let smaller_area = left.bounds.area().min(right.bounds.area());
    if smaller_area == 0
        || left.bounds.intersection_area(right.bounds) * 100
            < smaller_area * DUPLICATE_OVERLAP_PERCENT
    {
        return false;
    }
    let left_text = compact_text(&left.record.source_text);
    let right_text = compact_text(&right.record.source_text);
    !left_text.is_empty()
        && !right_text.is_empty()
        && (left_text == right_text
            || left_text.contains(&right_text)
            || right_text.contains(&left_text))
}

fn deduplicate_regions(regions: Vec<PreparedRegion>) -> Vec<PreparedRegion> {
    let mut retained = Vec::with_capacity(regions.len());
    for region in regions {
        let duplicate = retained
            .iter()
            .position(|candidate| duplicate_region(candidate, &region));
        let Some(index) = duplicate else {
            retained.push(region);
            continue;
        };
        let replace = region.record.source_text.chars().count()
            > retained[index].record.source_text.chars().count()
            || (region.record.source_text.chars().count()
                == retained[index].record.source_text.chars().count()
                && region.original_order < retained[index].original_order);
        if replace {
            retained[index] = region;
        }
    }
    retained
}

fn regions_belong_together(left: &PreparedRegion, right: &PreparedRegion) -> bool {
    let gap = right.bounds.left - left.bounds.right;
    let reference_height = left.bounds.height().min(right.bounds.height()).max(1);
    let maximum_overlap = (reference_height / MAX_FRAGMENT_OVERLAP_HEIGHT_DIVISOR).max(2);
    if gap < -maximum_overlap {
        return false;
    }
    let character_width = left
        .estimated_character_width()
        .max(right.estimated_character_width());
    let maximum_gap = (character_width * 1.65)
        .max(f64::from(reference_height) * MIN_FRAGMENT_GAP_HEIGHT_RATIO)
        .round()
        .clamp(
            f64::from(MIN_FRAGMENT_GAP_PX),
            f64::from(
                reference_height
                    .saturating_mul(MAX_FRAGMENT_GAP_HEIGHTS)
                    .max(MIN_FRAGMENT_GAP_PX),
            ),
        ) as i32;
    gap.max(0) <= maximum_gap
}

fn group_from_regions(regions: Vec<PreparedRegion>) -> LiveOcrGroup {
    let bounds = regions
        .iter()
        .map(|region| region.bounds)
        .reduce(TextBounds::union)
        .expect("OCR group always has a region");
    LiveOcrGroup {
        regions: regions.into_iter().map(|region| region.record).collect(),
        quad: bounds.into_quad(),
    }
}

pub(super) fn plan_live_ocr_groups(regions: Vec<RegionRecord>) -> Vec<LiveOcrGroup> {
    let prepared = regions
        .into_iter()
        .filter_map(|mut record| {
            record.source_text = normalize_text(&record.source_text);
            record.translated_text.clear();
            let bounds = TextBounds::from_quad(record.quad_points)?;
            has_translatable_character(&record.source_text).then_some(PreparedRegion {
                original_order: record.order,
                record,
                bounds,
            })
        })
        .collect::<Vec<_>>();
    let mut prepared = deduplicate_regions(prepared);
    prepared.sort_by(compare_regions);

    let mut lines = Vec::<VisualLine>::new();
    for region in prepared {
        let line = lines
            .iter()
            .enumerate()
            .filter(|(_, candidate)| candidate.accepts(&region))
            .min_by(|(_, left), (_, right)| {
                (left.mean_center_y - region.bounds.center_y())
                    .abs()
                    .total_cmp(&(right.mean_center_y - region.bounds.center_y()).abs())
                    .then_with(|| left.leftmost().cmp(&right.leftmost()))
            })
            .map(|(index, _)| index);
        if let Some(index) = line {
            lines[index].push(region);
        } else {
            lines.push(VisualLine::new(region));
        }
    }
    lines.sort_by(|left, right| {
        left.mean_center_y
            .total_cmp(&right.mean_center_y)
            .then_with(|| left.leftmost().cmp(&right.leftmost()))
    });

    let mut groups = Vec::new();
    for mut line in lines {
        line.regions.sort_by(|left, right| {
            (
                left.bounds.left,
                left.bounds.top,
                left.bounds.right,
                left.bounds.bottom,
                left.original_order,
            )
                .cmp(&(
                    right.bounds.left,
                    right.bounds.top,
                    right.bounds.right,
                    right.bounds.bottom,
                    right.original_order,
                ))
        });
        let mut current = Vec::new();
        for region in line.regions {
            let can_join = current
                .last()
                .is_some_and(|previous| regions_belong_together(previous, &region));
            if !current.is_empty() && !can_join {
                groups.push(group_from_regions(std::mem::take(&mut current)));
            }
            current.push(region);
        }
        if !current.is_empty() {
            groups.push(group_from_regions(current));
        }
    }
    groups
}

pub(super) fn finalize_live_regions(regions: &mut Vec<RegionRecord>) {
    regions.retain_mut(|region| {
        region.source_text = normalize_text(&region.source_text);
        has_translatable_character(&region.source_text)
    });
    for (index, region) in regions.iter_mut().enumerate() {
        region.order = u32::try_from(index + 1).unwrap_or(u32::MAX);
        region.translated_text.clear();
        region.characters.sort_by_key(|character| character.order);
        for (character_index, character) in region.characters.iter_mut().enumerate() {
            character.order = u32::try_from(character_index + 1).unwrap_or(u32::MAX);
        }
    }
}

fn ordered_regions(regions: &[RegionRecord]) -> Vec<&RegionRecord> {
    let mut ordered = regions.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|region| {
        let top = region
            .quad_points
            .iter()
            .map(|point| point[1])
            .min()
            .unwrap_or(0);
        let left = region
            .quad_points
            .iter()
            .map(|point| point[0])
            .min()
            .unwrap_or(0);
        let bottom = region
            .quad_points
            .iter()
            .map(|point| point[1])
            .max()
            .unwrap_or(0);
        let right = region
            .quad_points
            .iter()
            .map(|point| point[0])
            .max()
            .unwrap_or(0);
        (top, left, bottom, right, region.order)
    });
    ordered
}

fn ordered_live_regions(regions: &[RegionRecord]) -> Vec<&RegionRecord> {
    let mut ordered = regions.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|region| region.order);
    ordered
}

pub(super) fn normalized_region_text(regions: &[RegionRecord]) -> String {
    ordered_regions(regions)
        .into_iter()
        .map(|region| normalize_text(&region.source_text))
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) fn live_translated_region_text(regions: &[RegionRecord]) -> String {
    ordered_live_regions(regions)
        .into_iter()
        .map(|region| region.translated_text.as_str())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) fn normalized_live_region_text(regions: &[RegionRecord]) -> String {
    ordered_live_regions(regions)
        .into_iter()
        .map(|region| normalize_text(&region.source_text))
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn join_fragments<'a>(fragments: impl IntoIterator<Item = &'a str>) -> String {
    let mut result = String::new();
    for fragment in fragments {
        let fragment = normalize_text(fragment);
        if fragment.is_empty() {
            continue;
        }
        let needs_space = result
            .chars()
            .last()
            .zip(fragment.chars().next())
            .is_some_and(|(left, right)| {
                (left.is_alphanumeric() || matches!(left, ',' | '.' | ';' | ':' | '!' | '?'))
                    && right.is_alphanumeric()
                    && !is_cjk_like(left)
                    && !is_cjk_like(right)
            });
        if needs_space {
            result.push(' ');
        }
        result.push_str(&fragment);
    }
    result
}

fn is_cjk_like(character: char) -> bool {
    matches!(
        character as u32,
        0x3040..=0x30ff
            | 0x3400..=0x4dbf
            | 0x4e00..=0x9fff
            | 0xac00..=0xd7af
            | 0xf900..=0xfaff
    )
}

pub(super) fn normalize_text(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn has_translatable_character(value: &str) -> bool {
    value.chars().any(char::is_alphabetic)
}

pub(super) const fn roi_result_is_current(result_version: u64, current_version: u64) -> bool {
    result_version == current_version
}

#[cfg(test)]
mod tests {
    use super::{
        LatestFrameSlot, OwnedFrame, StabilityScheduler, finalize_live_regions,
        live_translated_region_text, normalized_live_region_text, plan_live_ocr_groups,
        roi_result_is_current,
    };
    use crate::backend::contracts::RegionRecord;
    use crate::backend::live::contracts::LiveRoi;
    use std::time::Duration;

    fn frame(value: u8, version: u64) -> OwnedFrame {
        OwnedFrame {
            width: 24,
            height: 14,
            rgb: vec![value; 24 * 14 * 3],
            observed_at_epoch_ms: 1,
            roi: LiveRoi {
                x: 0,
                y: 0,
                width: 24,
                height: 14,
                client_width: 24,
                client_height: 14,
            },
            roi_version: version,
        }
    }

    #[test]
    fn capacity_one_slot_replaces_unconsumed_frame() {
        let slot = LatestFrameSlot::default();
        assert!(!slot.replace(frame(1, 1)));
        assert!(slot.replace(frame(2, 2)));
        let latest = slot.wait_take(Duration::ZERO).expect("latest frame");
        assert_eq!(latest.rgb[0], 2);
        assert_eq!(latest.roi_version, 2);
        assert!(slot.wait_take(Duration::ZERO).is_none());
    }

    #[test]
    fn scheduler_processes_one_stable_frame_and_reuses_it_until_change() {
        let mut scheduler = StabilityScheduler::new(300);
        assert!(!scheduler.observe(&frame(20, 1), 0));
        assert!(
            !scheduler.tick(300),
            "one frame is not continuous stability"
        );
        assert!(scheduler.observe(&frame(20, 1), 300));
        assert!(!scheduler.tick(2_000));
        assert!(!scheduler.observe(&frame(20, 1), 2_001));
        assert!(!scheduler.tick(2_500));
        assert!(!scheduler.observe(&frame(240, 1), 2_501));
        assert!(
            !scheduler.tick(2_801),
            "changed frame needs a matching successor"
        );
        assert!(scheduler.observe(&frame(240, 1), 2_801));
        assert!(!scheduler.tick(4_000));
    }

    #[test]
    fn scheduler_waits_for_the_configured_stability_window() {
        let mut scheduler = StabilityScheduler::new(800);
        assert!(!scheduler.observe(&frame(20, 1), 0));
        assert!(!scheduler.observe(&frame(20, 1), 400));
        assert!(!scheduler.tick(799));
        assert!(scheduler.tick(800));
    }

    #[test]
    fn scheduler_restarts_wait_after_a_visual_change_without_forcing_ocr() {
        let mut scheduler = StabilityScheduler::new(500);
        assert!(!scheduler.observe(&frame(20, 1), 0));
        assert!(!scheduler.observe(&frame(20, 1), 250));
        assert!(!scheduler.observe(&frame(240, 1), 400));
        assert!(!scheduler.tick(899));
        assert!(scheduler.observe(&frame(240, 1), 900));
    }

    #[test]
    fn scheduler_detects_a_localized_change_after_reusing_a_frame() {
        let mut scheduler = StabilityScheduler::new(300);
        let original = frame(20, 1);
        assert!(!scheduler.observe(&original, 0));
        assert!(scheduler.observe(&original, 300));

        let mut changed = frame(20, 1);
        for y in 4..10 {
            for x in 8..16 {
                let offset = (y * changed.width as usize + x) * 3;
                changed.rgb[offset..offset + 3].fill(240);
            }
        }
        assert!(!scheduler.observe(&changed, 400));
        assert!(!scheduler.tick(700));
        assert!(scheduler.observe(&changed, 700));
    }

    #[test]
    fn live_translated_region_text_preserves_model_whitespace() {
        let mut regions = vec![
            RegionRecord::untranslated(1, [[60, 50], [90, 50], [90, 70], [60, 70]], "third"),
            RegionRecord::untranslated(2, [[50, 10], [90, 10], [90, 30], [50, 30]], "second"),
            RegionRecord::untranslated(3, [[5, 10], [40, 10], [40, 30], [5, 30]], "first"),
        ];
        regions[0].translated_text = "第三  行\n下一行  ".to_owned();
        regions[1].translated_text = "第二".to_owned();
        regions[2].translated_text = " 第一\n行 ".to_owned();

        assert_eq!(
            live_translated_region_text(&regions),
            "第三  行\n下一行  \n第二\n 第一\n行 "
        );
    }

    #[test]
    fn live_grouping_uses_visual_lines_and_only_tight_fragment_gaps() {
        let groups = plan_live_ocr_groups(vec![
            RegionRecord::untranslated(1, [[55, 18], [110, 18], [110, 38], [55, 38]], "world"),
            RegionRecord::untranslated(2, [[5, 20], [50, 20], [50, 40], [5, 40]], "Hello"),
            RegionRecord::untranslated(3, [[5, 70], [62, 70], [62, 90], [5, 90]], "next line"),
            RegionRecord::untranslated(
                4,
                [[100, 70], [160, 70], [160, 90], [100, 90]],
                "unrelated",
            ),
        ]);

        assert_eq!(
            groups
                .iter()
                .map(|group| group.source_text())
                .collect::<Vec<_>>(),
            vec!["Hello world", "next line", "unrelated"]
        );
    }

    #[test]
    fn live_regions_drop_numeric_and_symbol_only_detector_noise() {
        let mut regions = vec![
            RegionRecord::untranslated(1, [[0, 0], [50, 0], [50, 20], [0, 20]], "0-----"),
            RegionRecord::untranslated(2, [[0, 30], [20, 30], [20, 50], [0, 50]], "×"),
            RegionRecord::untranslated(
                3,
                [[0, 60], [180, 60], [180, 80], [0, 80]],
                "Boats are available",
            ),
        ];

        let groups = plan_live_ocr_groups(regions.clone());
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].source_text(), "Boats are available");

        finalize_live_regions(&mut regions);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].order, 1);
        assert_eq!(regions[0].source_text, "Boats are available");
    }

    #[test]
    fn live_grouping_reconstructs_fragmented_newspaper_lines() {
        let groups = plan_live_ocr_groups(vec![
            RegionRecord::untranslated(1, [[5, 20], [92, 20], [92, 40], [5, 40]], "Newspaper"),
            RegionRecord::untranslated(
                2,
                [[90, 20], [162, 20], [162, 40], [90, 40]],
                ": headlines",
            ),
            RegionRecord::untranslated(3, [[5, 50], [77, 50], [77, 70], [5, 70]], "Adoptions"),
            RegionRecord::untranslated(
                4,
                [[75, 50], [164, 50], [164, 70], [75, 70]],
                "wanted! How",
            ),
            RegionRecord::untranslated(5, [[162, 50], [184, 50], [184, 70], [162, 70]], "to"),
            RegionRecord::untranslated(
                6,
                [[182, 50], [254, 50], [254, 70], [182, 70]],
                "help the..",
            ),
            RegionRecord::untranslated(7, [[5, 80], [45, 80], [45, 100], [5, 100]], "Mad"),
            RegionRecord::untranslated(
                8,
                [[43, 80], [132, 80], [132, 100], [43, 100]],
                "scientist on",
            ),
            RegionRecord::untranslated(9, [[130, 80], [168, 80], [168, 100], [130, 100]], "the"),
            RegionRecord::untranslated(
                10,
                [[166, 80], [302, 80], [302, 100], [166, 100]],
                "loose! Townsfolks.",
            ),
            RegionRecord::untranslated(11, [[5, 110], [62, 110], [62, 130], [5, 130]], "A Tale"),
            RegionRecord::untranslated(
                12,
                [[60, 110], [118, 110], [118, 130], [60, 130]],
                "of Two",
            ),
            RegionRecord::untranslated(
                13,
                [[116, 110], [170, 110], [170, 130], [116, 130]],
                "Cities:",
            ),
            RegionRecord::untranslated(
                14,
                [[168, 110], [220, 110], [220, 130], [168, 130]],
                "Maids",
            ),
        ]);

        assert_eq!(
            groups
                .iter()
                .map(|group| group.source_text())
                .collect::<Vec<_>>(),
            vec![
                "Newspaper: headlines",
                "Adoptions wanted! How to help the..",
                "Mad scientist on the loose! Townsfolks.",
                "A Tale of Two Cities: Maids",
            ]
        );
        let mut merged = groups
            .into_iter()
            .map(|group| {
                let source_text = group.source_text();
                group.into_merged_region(source_text)
            })
            .collect::<Vec<_>>();
        finalize_live_regions(&mut merged);
        assert_eq!(merged.len(), 4);
        assert_eq!(
            normalized_live_region_text(&merged),
            "Newspaper: headlines\n\
             Adoptions wanted! How to help the..\n\
             Mad scientist on the loose! Townsfolks.\n\
             A Tale of Two Cities: Maids"
        );
    }

    #[test]
    fn live_grouping_keeps_widely_spaced_fragments_on_their_visual_line() {
        let groups = plan_live_ocr_groups(vec![
            RegionRecord::untranslated(
                1,
                [[5, 20], [162, 20], [162, 40], [5, 40]],
                "Newspaper headlines",
            ),
            RegionRecord::untranslated(2, [[5, 50], [77, 50], [77, 70], [5, 70]], "Adoptions"),
            RegionRecord::untranslated(
                3,
                [[95, 50], [254, 50], [254, 70], [95, 70]],
                "wanted! How to help the",
            ),
            RegionRecord::untranslated(
                4,
                [[5, 80], [302, 80], [302, 100], [5, 100]],
                "Mad scientist on the loose! Townsfolks.",
            ),
            RegionRecord::untranslated(5, [[5, 110], [62, 110], [62, 130], [5, 130]], "A Tale"),
            RegionRecord::untranslated(
                6,
                [[80, 110], [220, 110], [220, 130], [80, 130]],
                "of Two Cities: Maids",
            ),
        ]);

        assert_eq!(
            groups
                .iter()
                .map(|group| group.source_text())
                .collect::<Vec<_>>(),
            vec![
                "Newspaper headlines",
                "Adoptions wanted! How to help the",
                "Mad scientist on the loose! Townsfolks.",
                "A Tale of Two Cities: Maids",
            ]
        );
    }

    #[test]
    fn live_grouping_removes_overlapping_duplicate_regions_and_fixes_reading_order() {
        let groups = plan_live_ocr_groups(vec![
            RegionRecord::untranslated(1, [[101, 18], [160, 18], [160, 38], [101, 38]], "right"),
            RegionRecord::untranslated(2, [[5, 20], [70, 20], [70, 40], [5, 40]], "left"),
            RegionRecord::untranslated(3, [[6, 21], [71, 21], [71, 41], [6, 41]], "left"),
            RegionRecord::untranslated(4, [[5, 70], [90, 70], [90, 90], [5, 90]], "lower"),
        ]);
        let mut records = groups
            .into_iter()
            .flat_map(|group| group.into_regions())
            .collect::<Vec<_>>();
        finalize_live_regions(&mut records);

        assert_eq!(records.len(), 3);
        assert_eq!(normalized_live_region_text(&records), "left\nright\nlower");
        assert_eq!(
            records
                .iter()
                .map(|record| record.order)
                .collect::<Vec<_>>(),
            [1, 2, 3]
        );
    }

    #[test]
    fn stale_roi_results_are_rejected() {
        assert!(roi_result_is_current(4, 4));
        assert!(!roi_result_is_current(3, 4));
    }
}
