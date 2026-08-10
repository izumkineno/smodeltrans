use super::contracts::DEFAULT_STABILITY_WAIT_MS;
use crate::backend::contracts::RegionRecord;

use std::{
    collections::VecDeque,
    sync::{Condvar, Mutex},
    time::Duration,
};

const SIGNATURE_COLUMNS: usize = 24;
const SIGNATURE_ROWS: usize = 14;
const CHANGE_THRESHOLD: f32 = 0.035;
const PROBE_INTERVAL_MS: u64 = 250;
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
    signature: Vec<u8>,
    stable_since_ms: Option<u64>,
    last_probe_ms: Option<u64>,
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
            signature: Vec::new(),
            stable_since_ms: None,
            last_probe_ms: None,
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
        let signature = luminance_signature(frame);
        if self.signature.is_empty() {
            self.signature = signature;
            self.stable_since_ms = Some(now_ms);
            return false;
        }
        let changed = signature_difference(&self.signature, &signature) > CHANGE_THRESHOLD;
        self.signature = signature;
        if changed {
            self.stable_since_ms = Some(now_ms);
            self.suppressed_until_change = false;
            self.last_probe_ms = None;
        }
        self.should_probe(now_ms)
    }

    pub(super) fn tick(&mut self, now_ms: u64) -> bool {
        self.should_probe(now_ms)
    }

    pub(super) fn mark_confirmed(&mut self) {
        self.suppressed_until_change = true;
    }

    pub(super) fn reset(&mut self) {
        let settle_ms = self.settle_ms;
        *self = Self::new(settle_ms);
    }

    fn should_probe(&mut self, now_ms: u64) -> bool {
        if self.suppressed_until_change || self.signature.is_empty() {
            return false;
        }
        if self
            .last_probe_ms
            .is_some_and(|last| now_ms.saturating_sub(last) < PROBE_INTERVAL_MS)
        {
            return false;
        }
        let settled = self
            .stable_since_ms
            .is_some_and(|since| now_ms.saturating_sub(since) >= self.settle_ms);
        if settled {
            self.last_probe_ms = Some(now_ms);
            true
        } else {
            false
        }
    }
}

fn luminance_signature(frame: &OwnedFrame) -> Vec<u8> {
    if frame.width == 0 || frame.height == 0 || frame.rgb.len() < 3 {
        return Vec::new();
    }
    let mut signature = Vec::with_capacity(SIGNATURE_COLUMNS * SIGNATURE_ROWS);
    for row in 0..SIGNATURE_ROWS {
        let y = ((row * frame.height as usize) / SIGNATURE_ROWS)
            .min(frame.height.saturating_sub(1) as usize);
        for column in 0..SIGNATURE_COLUMNS {
            let x = ((column * frame.width as usize) / SIGNATURE_COLUMNS)
                .min(frame.width.saturating_sub(1) as usize);
            let offset = (y * frame.width as usize + x) * 3;
            let red = u32::from(frame.rgb[offset]);
            let green = u32::from(frame.rgb[offset + 1]);
            let blue = u32::from(frame.rgb[offset + 2]);
            signature.push(((red * 77 + green * 150 + blue * 29) >> 8) as u8);
        }
    }
    signature
}

fn signature_difference(left: &[u8], right: &[u8]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 1.0;
    }
    let difference = left
        .iter()
        .zip(right)
        .map(|(left, right)| u64::from(left.abs_diff(*right)))
        .sum::<u64>();
    difference as f32 / (left.len() as f32 * 255.0)
}

#[derive(Debug, Default)]
pub(super) struct TwoProbeConfirmation {
    candidate: Option<String>,
    hits: u8,
    accepted: Option<String>,
}

impl TwoProbeConfirmation {
    pub(super) fn observe(&mut self, text: String) -> Option<String> {
        if self.candidate.as_deref() == Some(text.as_str()) {
            self.hits = self.hits.saturating_add(1);
        } else {
            self.candidate = Some(text.clone());
            self.hits = 1;
        }
        if self.hits < 2 || self.accepted.as_deref() == Some(text.as_str()) {
            return None;
        }
        self.accepted = Some(text.clone());
        Some(text)
    }

    pub(super) fn reset(&mut self) {
        *self = Self::default();
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

pub(super) fn normalized_region_text(regions: &[RegionRecord]) -> String {
    ordered_regions(regions)
        .into_iter()
        .map(|region| normalize_text(&region.source_text))
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) fn normalized_translated_region_text(regions: &[RegionRecord]) -> String {
    ordered_regions(regions)
        .into_iter()
        .map(|region| normalize_text(&region.translated_text))
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) fn normalize_text(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Debug)]
pub(super) struct BoundedCache {
    capacity: usize,
    entries: VecDeque<(String, String)>,
}

impl BoundedCache {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: VecDeque::with_capacity(capacity.max(1)),
        }
    }

    pub(super) fn get(&mut self, key: &str) -> Option<String> {
        let index = self
            .entries
            .iter()
            .position(|(candidate, _)| candidate == key)?;
        let entry = self.entries.remove(index)?;
        let value = entry.1.clone();
        self.entries.push_back(entry);
        Some(value)
    }

    pub(super) fn insert(&mut self, key: String, value: String) {
        if let Some(index) = self
            .entries
            .iter()
            .position(|(candidate, _)| candidate == &key)
        {
            self.entries.remove(index);
        }
        self.entries.push_back((key, value));
        while self.entries.len() > self.capacity {
            self.entries.pop_front();
        }
    }

    pub(super) fn clear(&mut self) {
        self.entries.clear();
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

pub(super) const fn roi_result_is_current(result_version: u64, current_version: u64) -> bool {
    result_version == current_version
}

#[cfg(test)]
mod tests {
    use super::{
        BoundedCache, LatestFrameSlot, OwnedFrame, StabilityScheduler, TwoProbeConfirmation,
        normalized_region_text, normalized_translated_region_text, roi_result_is_current,
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
    fn scheduler_settles_and_confirmation_requires_two_identical_hits() {
        let mut scheduler = StabilityScheduler::default();
        assert!(!scheduler.observe(&frame(20, 1), 0));
        assert!(scheduler.tick(300));
        let mut confirmation = TwoProbeConfirmation::default();
        assert!(confirmation.observe("hello".to_owned()).is_none());
        assert_eq!(
            confirmation.observe("hello".to_owned()),
            Some("hello".to_owned())
        );
        scheduler.mark_confirmed();
        assert!(!scheduler.tick(2_000));
        assert!(!scheduler.observe(&frame(240, 1), 2_001));
        assert!(scheduler.tick(2_301));
    }

    #[test]
    fn scheduler_waits_for_the_configured_stability_window() {
        let mut scheduler = StabilityScheduler::new(800);
        assert!(!scheduler.observe(&frame(20, 1), 0));
        assert!(!scheduler.tick(799));
        assert!(scheduler.tick(800));
    }

    #[test]
    fn scheduler_restarts_wait_after_a_visual_change_without_forcing_ocr() {
        let mut scheduler = StabilityScheduler::new(500);
        assert!(!scheduler.observe(&frame(20, 1), 0));
        assert!(!scheduler.observe(&frame(240, 1), 400));
        assert!(!scheduler.tick(899));
        assert!(scheduler.tick(900));
    }

    #[test]
    fn region_text_is_geometry_ordered_then_whitespace_normalized() {
        let regions = vec![
            RegionRecord::untranslated(1, [[60, 50], [90, 50], [90, 70], [60, 70]], " third "),
            RegionRecord::untranslated(2, [[50, 10], [90, 10], [90, 30], [50, 30]], "second"),
            RegionRecord::untranslated(3, [[5, 10], [40, 10], [40, 30], [5, 30]], " first\nline "),
        ];
        assert_eq!(
            normalized_region_text(&regions),
            "first line\nsecond\nthird"
        );
    }

    #[test]
    fn translated_region_text_preserves_one_line_per_region() {
        let mut regions = vec![
            RegionRecord::untranslated(1, [[60, 50], [90, 50], [90, 70], [60, 70]], "third"),
            RegionRecord::untranslated(2, [[50, 10], [90, 10], [90, 30], [50, 30]], "second"),
            RegionRecord::untranslated(3, [[5, 10], [40, 10], [40, 30], [5, 30]], "first"),
        ];
        regions[0].translated_text = "第三 行".to_owned();
        regions[1].translated_text = "第二".to_owned();
        regions[2].translated_text = "第一\n行".to_owned();

        assert_eq!(
            normalized_translated_region_text(&regions),
            "第一 行\n第二\n第三 行"
        );
    }

    #[test]
    fn cache_is_bounded_and_uses_recent_entries() {
        let mut cache = BoundedCache::new(2);
        cache.insert("a".to_owned(), "A".to_owned());
        cache.insert("b".to_owned(), "B".to_owned());
        assert_eq!(cache.get("a"), Some("A".to_owned()));
        cache.insert("c".to_owned(), "C".to_owned());
        assert_eq!(cache.len(), 2);
        assert!(cache.get("b").is_none());
        assert_eq!(cache.get("a"), Some("A".to_owned()));
    }

    #[test]
    fn stale_roi_results_are_rejected() {
        assert!(roi_result_is_current(4, 4));
        assert!(!roi_result_is_current(3, 4));
    }
}
