use crate::backend::{
    failure::BackendFailure,
    live::{
        contracts::{
            CaptureWindowInfo, LiveConfig, LiveMetrics, LiveRecognitionMode,
            LiveRecognitionSettings, LiveRoi,
        },
        scheduler::{LatestFrameSlot, OwnedFrame},
    },
};
use std::{
    collections::HashSet,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use windows::{
    Win32::{
        Foundation::{HWND, LPARAM, TRUE},
        UI::{
            Input::KeyboardAndMouse::{
                GetAsyncKeyState, VK_CONTROL, VK_F1, VK_MENU, VK_RETURN, VK_SHIFT, VK_SPACE, VK_TAB,
            },
            WindowsAndMessaging::{
                EnumWindows, IsIconic, IsWindow, IsWindowVisible, SW_RESTORE, ShowWindow,
            },
        },
    },
    core::BOOL,
};
use windows_capture::{
    capture::{CaptureControl, Context, GraphicsCaptureApiHandler},
    frame::{DirtyRegion, Frame},
    graphics_capture_api::{GraphicsCaptureApi, InternalCaptureControl},
    settings::{
        ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
        MinimumUpdateIntervalSettings, SecondaryWindowSettings, Settings,
    },
    window::Window,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::backend::live) struct TargetGeometry {
    pub(in crate::backend::live) x: i32,
    pub(in crate::backend::live) y: i32,
    pub(in crate::backend::live) width: u32,
    pub(in crate::backend::live) height: u32,
}

unsafe extern "system" fn collect_top_level_window(hwnd: HWND, windows: LPARAM) -> BOOL {
    let windows = unsafe { &mut *(windows.0 as *mut Vec<HWND>) };
    windows.push(hwnd);
    TRUE
}

fn enumerate_top_level_windows() -> Result<Vec<Window>, BackendFailure> {
    let mut raw_windows: Vec<HWND> = Vec::new();
    unsafe {
        EnumWindows(
            Some(collect_top_level_window),
            LPARAM(std::ptr::addr_of_mut!(raw_windows) as isize),
        )
    }
    .map_err(|error| BackendFailure::internal(format!("无法枚举顶级窗口: {error}")))?;
    Ok(raw_windows
        .into_iter()
        .filter(|hwnd| unsafe {
            IsWindow(Some(*hwnd)).as_bool() && IsWindowVisible(*hwnd).as_bool()
        })
        .map(|hwnd| Window::from_raw_hwnd(hwnd.0))
        .collect())
}

fn is_live_target_window(window: &Window) -> bool {
    let hwnd = HWND(window.as_raw_hwnd());
    unsafe { IsWindow(Some(hwnd)).as_bool() }
}

pub(in crate::backend::live) fn list_target_windows()
-> Result<Vec<CaptureWindowInfo>, BackendFailure> {
    let current_pid = std::process::id();
    let mut seen = HashSet::new();
    let mut targets = Vec::new();
    let windows = enumerate_top_level_windows()?;
    for window in windows {
        let raw = window.as_raw_hwnd() as usize;
        if raw == 0 || !seen.insert(raw) {
            continue;
        }
        let hwnd = HWND(window.as_raw_hwnd());
        let is_minimized = unsafe { IsIconic(hwnd).as_bool() };
        let title = window.title().unwrap_or_default();
        let Ok(process_id) = window.process_id() else {
            continue;
        };
        if process_id == current_pid {
            continue;
        }
        let (width, height) = if is_minimized {
            (0, 0)
        } else {
            let Ok(geometry) = geometry_for_window(&window) else {
                continue;
            };
            (geometry.width, geometry.height)
        };
        targets.push(CaptureWindowInfo {
            id: raw.to_string(),
            title: title.trim().to_owned(),
            process_name: window
                .process_name()
                .unwrap_or_else(|_| "未知进程".to_owned()),
            process_id,
            width,
            height,
            is_minimized,
        });
    }
    targets.sort_by(|left, right| left.title.to_lowercase().cmp(&right.title.to_lowercase()));
    Ok(targets)
}

fn target_from_id(target_id: &str) -> Result<Window, BackendFailure> {
    let raw = target_id
        .parse::<usize>()
        .map_err(|_| BackendFailure::arguments("targetId 不是有效的窗口句柄"))?;
    if raw == 0 {
        return Err(BackendFailure::arguments("targetId 不是有效的窗口句柄"));
    }
    let window = Window::from_raw_hwnd(raw as *mut std::ffi::c_void);
    if !is_live_target_window(&window) {
        return Err(BackendFailure::arguments("目标窗口已关闭或不可捕获"));
    }
    Ok(window)
}

fn geometry_for_window(window: &Window) -> Result<TargetGeometry, BackendFailure> {
    use windows::{
        Win32::{
            Foundation::{HWND, POINT, RECT},
            Graphics::Gdi::ClientToScreen,
            UI::WindowsAndMessaging::GetClientRect,
        },
        core::Error as WindowsError,
    };

    let hwnd = HWND(window.as_raw_hwnd());
    let mut rect = RECT::default();
    unsafe { GetClientRect(hwnd, &mut rect) }
        .map_err(|error| BackendFailure::internal(format!("无法读取目标客户区: {error}")))?;
    let mut origin = POINT {
        x: rect.left,
        y: rect.top,
    };
    if !unsafe { ClientToScreen(hwnd, &mut origin) }.as_bool() {
        return Err(BackendFailure::internal(format!(
            "无法换算目标客户区坐标: {}",
            WindowsError::from_win32()
        )));
    }
    let width = u32::try_from(rect.right.saturating_sub(rect.left))
        .map_err(|_| BackendFailure::internal("目标客户区宽度无效"))?;
    let height = u32::try_from(rect.bottom.saturating_sub(rect.top))
        .map_err(|_| BackendFailure::internal("目标客户区高度无效"))?;
    if width == 0 || height == 0 {
        return Err(BackendFailure::arguments("目标窗口当前不可见或已最小化"));
    }
    Ok(TargetGeometry {
        x: origin.x,
        y: origin.y,
        width,
        height,
    })
}

pub(in crate::backend::live) fn target_geometry(
    target_id: &str,
) -> Result<TargetGeometry, BackendFailure> {
    geometry_for_window(&target_from_id(target_id)?)
}

pub(in crate::backend::live) fn activate_target_window(
    target_id: &str,
) -> Result<(), BackendFailure> {
    use windows::Win32::UI::WindowsAndMessaging::{BringWindowToTop, SetForegroundWindow};

    let window = target_from_id(target_id)?;
    let hwnd = HWND(window.as_raw_hwnd());
    if unsafe { IsIconic(hwnd).as_bool() } {
        let _ = unsafe { ShowWindow(hwnd, SW_RESTORE) };
    }
    if unsafe { !SetForegroundWindow(hwnd).as_bool() } {
        return Err(BackendFailure::arguments(
            "Windows 未允许将目标窗口置于前台；请关闭可能打开的系统菜单后重试",
        ));
    }
    unsafe {
        BringWindowToTop(hwnd)
            .map_err(|error| BackendFailure::internal(format!("目标窗口置顶失败: {error}")))?;
    }
    Ok(())
}
pub(in crate::backend::live) fn target_is_minimized(
    target_id: &str,
) -> Result<bool, BackendFailure> {
    let window = target_from_id(target_id)?;
    Ok(unsafe { IsIconic(HWND(window.as_raw_hwnd())).as_bool() })
}
pub(in crate::backend::live) fn recognition_is_active(settings: &LiveRecognitionSettings) -> bool {
    if settings.mode == LiveRecognitionMode::Automatic {
        return true;
    }
    let Some(virtual_key) = trigger_virtual_key(&settings.trigger_key) else {
        return false;
    };
    unsafe { GetAsyncKeyState(virtual_key) < 0 }
}

/// Recorded keys carry the Windows virtual-key value from Chromium's
/// `KeyboardEvent.keyCode`; only the legacy settings format needs aliases.
fn trigger_virtual_key(key: &str) -> Option<i32> {
    let key = key.trim();
    if let Some(payload) = key.strip_prefix("vk:") {
        let virtual_key = payload.split_once('|').map_or(payload, |(value, _)| value);
        return virtual_key
            .parse::<u16>()
            .ok()
            .filter(|number| (1..=0xff).contains(number))
            .map(i32::from);
    }
    legacy_trigger_virtual_key(key)
}

fn legacy_trigger_virtual_key(key: &str) -> Option<i32> {
    if let Some(letter) = key.strip_prefix("Key") {
        let bytes = letter.as_bytes();
        if bytes.len() == 1 && bytes[0].is_ascii_uppercase() {
            return Some(i32::from(bytes[0]));
        }
    }
    if let Some(number) = key.strip_prefix("Digit") {
        let bytes = number.as_bytes();
        if bytes.len() == 1 && bytes[0].is_ascii_digit() {
            return Some(0x30 + i32::from(bytes[0] - b'0'));
        }
    }
    if let Some(number) = key.strip_prefix("Numpad") {
        let bytes = number.as_bytes();
        if bytes.len() == 1 && bytes[0].is_ascii_digit() {
            return Some(0x60 + i32::from(bytes[0] - b'0'));
        }
    }
    if let Some(number) = key
        .strip_prefix('F')
        .and_then(|number| number.parse::<u16>().ok())
        .filter(|number| (1..=24).contains(number))
    {
        return Some(i32::from(VK_F1.0 + number - 1));
    }
    match key {
        "Escape" => Some(0x1b),
        "CapsLock" => Some(0x14),
        "Shift" | "ShiftLeft" => Some(i32::from(VK_SHIFT.0)),
        "ShiftRight" => Some(0xa1),
        "Ctrl" | "Control" => Some(i32::from(VK_CONTROL.0)),
        "ControlLeft" => Some(0xa2),
        "ControlRight" => Some(0xa3),
        "Alt" | "AltLeft" => Some(i32::from(VK_MENU.0)),
        "AltRight" => Some(0xa5),
        "MetaLeft" => Some(0x5b),
        "MetaRight" => Some(0x5c),
        "ContextMenu" => Some(0x5d),
        "Space" => Some(i32::from(VK_SPACE.0)),
        "Enter" | "NumpadEnter" => Some(i32::from(VK_RETURN.0)),
        "Tab" => Some(i32::from(VK_TAB.0)),
        "Backspace" => Some(0x08),
        "Insert" => Some(0x2d),
        "Delete" => Some(0x2e),
        "Home" => Some(0x24),
        "End" => Some(0x23),
        "PageUp" => Some(0x21),
        "PageDown" => Some(0x22),
        "ArrowUp" => Some(0x26),
        "ArrowDown" => Some(0x28),
        "ArrowLeft" => Some(0x25),
        "ArrowRight" => Some(0x27),
        "PrintScreen" => Some(0x2c),
        "ScrollLock" => Some(0x91),
        "Pause" => Some(0x13),
        "NumLock" => Some(0x90),
        "Minus" => Some(0xbd),
        "Equal" | "NumpadEqual" => Some(0xbb),
        "BracketLeft" => Some(0xdb),
        "BracketRight" => Some(0xdd),
        "Backslash" | "IntlYen" => Some(0xdc),
        "IntlBackslash" => Some(0xe2),
        "Semicolon" => Some(0xba),
        "Quote" => Some(0xde),
        "Backquote" => Some(0xc0),
        "Comma" => Some(0xbc),
        "Period" => Some(0xbe),
        "Slash" | "IntlRo" => Some(0xbf),
        "NumpadDecimal" => Some(0x6e),
        "NumpadDivide" => Some(0x6f),
        "NumpadMultiply" => Some(0x6a),
        "NumpadSubtract" => Some(0x6d),
        "NumpadAdd" => Some(0x6b),
        _ => None,
    }
}

#[derive(Clone)]
struct CaptureFlags {
    latest: Arc<LatestFrameSlot>,
    terminal_error: Arc<Mutex<Option<String>>>,
    config: Arc<Mutex<LiveConfig>>,
    metrics: Arc<Mutex<LiveMetrics>>,
    paused: Arc<AtomicBool>,
}

struct CaptureHandler {
    flags: CaptureFlags,
    scratch: Vec<u8>,
    last_roi_version: Option<u64>,
    was_paused: bool,
}

impl GraphicsCaptureApiHandler for CaptureHandler {
    type Flags = CaptureFlags;
    type Error = String;

    fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
        Ok(Self {
            flags: ctx.flags,
            scratch: Vec::new(),
            last_roi_version: None,
            was_paused: false,
        })
    }

    fn on_frame_arrived(
        &mut self,
        frame: &mut Frame<'_>,
        capture_control: InternalCaptureControl,
    ) -> Result<(), Self::Error> {
        if self
            .flags
            .terminal_error
            .lock()
            .map(|error| error.is_some())
            .unwrap_or(true)
        {
            capture_control.stop();
            return Ok(());
        }
        if self.flags.paused.load(Ordering::Relaxed) {
            self.was_paused = true;
            return Ok(());
        }
        if self.was_paused {
            self.last_roi_version = None;
            self.was_paused = false;
        }
        let config = self
            .flags
            .config
            .lock()
            .map_err(|_| "实时 ROI 锁已损坏".to_owned())?
            .clone();
        if self.last_roi_version == Some(config.roi_version) {
            if let Ok(dirty_regions) = frame.dirty_regions() {
                if dirty_regions_miss_roi(frame.width(), frame.height(), &config, &dirty_regions) {
                    if let Ok(mut metrics) = self.flags.metrics.lock() {
                        metrics.frames_skipped_unchanged =
                            metrics.frames_skipped_unchanged.saturating_add(1);
                    }
                    return Ok(());
                }
            }
        }

        // Window capture frames include the non-client title bar. The live ROI
        // is defined against the target client area, so remove that strip
        // before mapping coordinates and copying pixels.
        let buffer = frame
            .buffer_without_title_bar()
            .map_err(|error| error.to_string())?;
        let frame_width = buffer.width();
        let frame_height = buffer.height();
        let bytes = buffer.as_nopadding_buffer(&mut self.scratch);
        let roi = config
            .roi
            .to_physical(frame_width, frame_height)
            .ok_or_else(|| "实时 ROI 无法映射到捕获帧".to_owned())?;
        let frame_stride = usize::try_from(frame_width)
            .ok()
            .and_then(|width| width.checked_mul(4))
            .ok_or_else(|| "捕获帧行宽溢出".to_owned())?;
        let pixel_count = usize::try_from(roi.width)
            .ok()
            .and_then(|width| {
                usize::try_from(roi.height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or_else(|| "ROI 像素数溢出".to_owned())?;
        let mut rgb = Vec::with_capacity(pixel_count.saturating_mul(3));
        for y in roi.y..roi.y + roi.height {
            let row_start = usize::try_from(y)
                .ok()
                .and_then(|y| y.checked_mul(frame_stride))
                .and_then(|start| {
                    usize::try_from(roi.x)
                        .ok()
                        .and_then(|x| x.checked_mul(4))
                        .and_then(|x| start.checked_add(x))
                })
                .ok_or_else(|| "ROI 行偏移溢出".to_owned())?;
            let row_bytes = usize::try_from(roi.width)
                .ok()
                .and_then(|width| width.checked_mul(4))
                .ok_or_else(|| "ROI 行宽溢出".to_owned())?;
            let row_end = row_start
                .checked_add(row_bytes)
                .ok_or_else(|| "ROI 行终点溢出".to_owned())?;
            let row = bytes
                .get(row_start..row_end)
                .ok_or_else(|| "ROI 超出捕获帧缓冲区".to_owned())?;
            for pixel in row.chunks_exact(4) {
                rgb.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
            }
        }
        let image = image::RgbImage::from_raw(roi.width, roi.height, rgb)
            .ok_or_else(|| "实时捕获帧缓冲区无效".to_owned())?;
        let observed_at_epoch_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX);
        let dropped = self.flags.latest.replace(OwnedFrame {
            width: roi.width,
            height: roi.height,
            image: Arc::new(image),
            observed_at_epoch_ms,
            roi: LiveRoi {
                x: roi.x,
                y: roi.y,
                width: roi.width,
                height: roi.height,
                client_width: frame_width,
                client_height: frame_height,
            },
            roi_version: config.roi_version,
        });
        self.last_roi_version = Some(config.roi_version);
        if let Ok(mut metrics) = self.flags.metrics.lock() {
            metrics.frames_captured = metrics.frames_captured.saturating_add(1);
            if dropped {
                metrics.frames_dropped = metrics.frames_dropped.saturating_add(1);
            }
        }
        Ok(())
    }

    fn on_closed(&mut self) -> Result<(), Self::Error> {
        if let Ok(mut terminal) = self.flags.terminal_error.lock() {
            *terminal = Some("目标窗口已关闭，实时翻译已停止。".to_owned());
        }
        self.flags.latest.wake();
        Ok(())
    }
}

fn dirty_regions_miss_roi(
    frame_width: u32,
    frame_height: u32,
    config: &LiveConfig,
    dirty_regions: &[DirtyRegion],
) -> bool {
    if frame_width != config.client_width || frame_height < config.client_height {
        return false;
    }
    let title_bar_height = frame_height - config.client_height;
    if title_bar_height > 256 {
        return false;
    }
    let Some(roi) = config
        .roi
        .to_physical(config.client_width, config.client_height)
    else {
        return false;
    };
    let roi_left = i64::from(roi.x);
    let roi_top = i64::from(roi.y) + i64::from(title_bar_height);
    let roi_right = roi_left + i64::from(roi.width);
    let roi_bottom = roi_top + i64::from(roi.height);
    !dirty_regions_intersect_rect(dirty_regions, roi_left, roi_top, roi_right, roi_bottom)
}

fn dirty_regions_intersect_rect(
    dirty_regions: &[DirtyRegion],
    roi_left: i64,
    roi_top: i64,
    roi_right: i64,
    roi_bottom: i64,
) -> bool {
    dirty_regions.iter().any(|dirty| {
        if dirty.width <= 0 || dirty.height <= 0 {
            return false;
        }
        let left = i64::from(dirty.x);
        let top = i64::from(dirty.y);
        let right = left + i64::from(dirty.width);
        let bottom = top + i64::from(dirty.height);
        left < roi_right && right > roi_left && top < roi_bottom && bottom > roi_top
    })
}

pub(in crate::backend::live) struct CaptureWorker {
    control: Option<CaptureControl<CaptureHandler, String>>,
}

impl CaptureWorker {
    pub(in crate::backend::live) fn stop(mut self) -> Result<(), BackendFailure> {
        let Some(control) = self.control.take() else {
            return Ok(());
        };
        control
            .stop()
            .map_err(|error| BackendFailure::internal(format!("停止窗口捕获失败: {error}")))
    }
}

pub(in crate::backend::live) fn start_capture(
    target_id: &str,
    latest: Arc<LatestFrameSlot>,
    terminal_error: Arc<Mutex<Option<String>>>,
    config: Arc<Mutex<LiveConfig>>,
    metrics: Arc<Mutex<LiveMetrics>>,
    paused: Arc<AtomicBool>,
) -> Result<CaptureWorker, BackendFailure> {
    let target = target_from_id(target_id)?;
    let dirty_region_settings = if GraphicsCaptureApi::is_dirty_region_supported().unwrap_or(false)
    {
        DirtyRegionSettings::ReportOnly
    } else {
        DirtyRegionSettings::Default
    };
    let settings = Settings::new(
        target,
        CursorCaptureSettings::WithoutCursor,
        DrawBorderSettings::WithoutBorder,
        SecondaryWindowSettings::Exclude,
        MinimumUpdateIntervalSettings::Custom(Duration::from_millis(100)),
        dirty_region_settings,
        ColorFormat::Bgra8,
        CaptureFlags {
            latest,
            terminal_error,
            config,
            metrics,
            paused,
        },
    );
    let control = CaptureHandler::start_free_threaded(settings).map_err(|error| {
        BackendFailure::internal(format!("启动 Windows Graphics Capture 失败: {error}"))
    })?;
    Ok(CaptureWorker {
        control: Some(control),
    })
}

#[cfg(test)]
mod tests {
    use super::{activate_target_window, dirty_regions_intersect_rect, trigger_virtual_key};
    use crate::backend::failure::BackendFailureCode;
    use windows_capture::frame::DirtyRegion;

    #[test]
    fn activating_an_invalid_target_rejects_the_handle_before_calling_windows() {
        let error = activate_target_window("not-a-window-handle")
            .expect_err("malformed handles must not reach the Windows API");

        assert_eq!(error.code(), BackendFailureCode::Arguments);
        assert_eq!(error.message(), "targetId 不是有效的窗口句柄");
    }

    #[test]
    fn dirty_regions_only_wake_live_ocr_when_the_roi_changes() {
        let outside = DirtyRegion {
            x: 0,
            y: 0,
            width: 100,
            height: 50,
        };
        let overlapping = DirtyRegion {
            x: 150,
            y: 120,
            width: 100,
            height: 40,
        };

        assert!(!dirty_regions_intersect_rect(
            &[outside],
            100,
            100,
            200,
            150,
        ));
        assert!(dirty_regions_intersect_rect(
            &[outside, overlapping],
            100,
            100,
            200,
            150,
        ));
    }

    #[test]
    fn recorded_virtual_keys_map_without_a_platform_key_table() {
        assert_eq!(trigger_virtual_key("vk:65|KeyA"), Some(65));
        assert_eq!(trigger_virtual_key("vk:121|F10"), Some(121));
        assert_eq!(trigger_virtual_key("KeyA"), Some(0x41));
        assert_eq!(trigger_virtual_key("Digit7"), Some(0x37));
        assert_eq!(trigger_virtual_key("Numpad3"), Some(0x63));
        assert_eq!(trigger_virtual_key("ArrowLeft"), Some(0x25));
        assert_eq!(trigger_virtual_key("Semicolon"), Some(0xba));
        assert_eq!(trigger_virtual_key("IntlBackslash"), Some(0xe2));
        assert_eq!(trigger_virtual_key("IntlYen"), Some(0xdc));
        assert_eq!(trigger_virtual_key("F8"), Some(0x77));
        assert_eq!(trigger_virtual_key("UnknownKey"), None);
    }
}
