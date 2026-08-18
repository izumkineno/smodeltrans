use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tauri::{Emitter, Manager, PhysicalPosition, PhysicalSize, State};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

use crate::selection;

const DEFAULT_SHORTCUT: &str = "CommandOrControl+Alt+E";
const QUICK_TRANSLATION_WINDOW_LABEL: &str = "quick-translation";
const QUICK_TRANSLATION_EVENT: &str = "quick-translation-request";
const QUICK_TRANSLATION_GAP: i32 = 12;
const DEFAULT_QUICK_TRANSLATION_WIDTH: u32 = 320;
const DEFAULT_QUICK_TRANSLATION_HEIGHT: u32 = 104;
const SUPPORTED_SHORTCUTS: [&str; 4] = [
    DEFAULT_SHORTCUT,
    "CommandOrControl+Alt+T",
    "CommandOrControl+Shift+E",
    "Alt+Shift+E",
];

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct QuickTranslationEvent {
    text: Option<String>,
    error: Option<String>,
    selection_bounds: Option<selection::SelectionBounds>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct QuickTranslationShortcutSettings {
    enabled: bool,
    shortcut: String,
}

pub(crate) struct QuickTranslationShortcutState(Mutex<Option<String>>);

pub(crate) fn setup(app: &mut tauri::App) {
    let registered_shortcut = match register_shortcut(app.handle(), DEFAULT_SHORTCUT) {
        Ok(()) => Some(DEFAULT_SHORTCUT.to_owned()),
        Err(error) => {
            eprintln!("注册快捷翻译快捷键失败（{DEFAULT_SHORTCUT}）：{error}");
            None
        }
    };
    app.manage(QuickTranslationShortcutState(Mutex::new(
        registered_shortcut,
    )));
}

fn register_shortcut(app: &tauri::AppHandle, shortcut: &str) -> Result<(), String> {
    app.global_shortcut()
        .on_shortcut(shortcut, |app, _shortcut, event| {
            if event.state() == ShortcutState::Pressed {
                trigger(app);
            }
        })
        .map_err(|error| error.to_string())
}

fn trigger(app: &tauri::AppHandle) {
    let app = app.clone();
    let spawn_result = std::thread::Builder::new()
        .name("smodeltrans-quick-translation".to_owned())
        .spawn(move || {
            let event = match selection::read_selected_text() {
                Ok(selection) => QuickTranslationEvent {
                    text: Some(selection.text),
                    error: None,
                    selection_bounds: selection.bounds,
                },
                Err(error) => QuickTranslationEvent {
                    text: None,
                    error: Some(error.to_string()),
                    selection_bounds: None,
                },
            };

            if let Some(window) = app.get_webview_window(QUICK_TRANSLATION_WINDOW_LABEL) {
                position_quick_translation_window(&app, &window, event.selection_bounds);
                let _ = window.show();
                let _ = window.set_focus();
            }
            if let Err(error) = app.emit_to(
                QUICK_TRANSLATION_WINDOW_LABEL,
                QUICK_TRANSLATION_EVENT,
                event,
            ) {
                eprintln!("发送快捷翻译选区事件失败：{error}");
            }
        });
    if let Err(error) = spawn_result {
        eprintln!("启动快捷翻译线程失败：{error}");
    }
}

fn position_quick_translation_window(
    app: &tauri::AppHandle,
    window: &tauri::WebviewWindow,
    selection_bounds: Option<selection::SelectionBounds>,
) {
    let selection_bounds = selection_bounds.or_else(|| {
        let cursor = app.cursor_position().ok()?;
        if !cursor.x.is_finite() || !cursor.y.is_finite() {
            return None;
        }
        let left = cursor.x.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        let top = cursor.y.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        Some(selection::SelectionBounds {
            left,
            top,
            right: left.saturating_add(1),
            bottom: top.saturating_add(1),
        })
    });
    let Some(selection_bounds) = selection_bounds else {
        return;
    };
    let Ok(monitors) = app.available_monitors() else {
        return;
    };
    let Some(monitor) = monitors
        .iter()
        .find(|monitor| point_is_in_work_area(monitor, selection_bounds.left, selection_bounds.top))
        .or_else(|| monitors.first())
    else {
        return;
    };

    let window_size = window.outer_size().unwrap_or_else(|_| {
        PhysicalSize::new(
            DEFAULT_QUICK_TRANSLATION_WIDTH,
            DEFAULT_QUICK_TRANSLATION_HEIGHT,
        )
    });
    let window_width = i32::try_from(window_size.width).unwrap_or(i32::MAX);
    let window_height = i32::try_from(window_size.height).unwrap_or(i32::MAX);
    let work_area = monitor.work_area();
    let work_left = work_area.position.x;
    let work_top = work_area.position.y;
    let work_right =
        work_left.saturating_add(i32::try_from(work_area.size.width).unwrap_or(i32::MAX));
    let work_bottom =
        work_top.saturating_add(i32::try_from(work_area.size.height).unwrap_or(i32::MAX));

    let x = clamp_window_position(selection_bounds.left, work_left, work_right, window_width);
    let below_selection = selection_bounds
        .bottom
        .saturating_add(QUICK_TRANSLATION_GAP);
    let above_selection = selection_bounds
        .top
        .saturating_sub(window_height)
        .saturating_sub(QUICK_TRANSLATION_GAP);
    let preferred_y = if below_selection.saturating_add(window_height) <= work_bottom {
        below_selection
    } else if above_selection >= work_top {
        above_selection
    } else {
        below_selection
    };
    let y = clamp_window_position(preferred_y, work_top, work_bottom, window_height);

    let _ = window.set_position(PhysicalPosition::new(x, y));
}

fn point_is_in_work_area(monitor: &tauri::Monitor, x: i32, y: i32) -> bool {
    let work_area = monitor.work_area();
    let right = work_area
        .position
        .x
        .saturating_add(i32::try_from(work_area.size.width).unwrap_or(i32::MAX));
    let bottom = work_area
        .position
        .y
        .saturating_add(i32::try_from(work_area.size.height).unwrap_or(i32::MAX));
    x >= work_area.position.x && x < right && y >= work_area.position.y && y < bottom
}

fn clamp_window_position(preferred: i32, minimum: i32, maximum: i32, window_size: i32) -> i32 {
    let latest = maximum.saturating_sub(window_size.max(0));
    if latest < minimum {
        minimum
    } else {
        preferred.clamp(minimum, latest)
    }
}

fn validate_shortcut(value: &str) -> Result<&str, String> {
    SUPPORTED_SHORTCUTS
        .iter()
        .copied()
        .find(|shortcut| *shortcut == value)
        .ok_or_else(|| "不支持该快捷键组合。".to_owned())
}

#[tauri::command]
pub(crate) fn configure_quick_translation(
    app: tauri::AppHandle,
    settings: QuickTranslationShortcutSettings,
    state: State<'_, QuickTranslationShortcutState>,
) -> Result<(), String> {
    let next_shortcut = validate_shortcut(settings.shortcut.trim())?;
    let current_shortcut = state
        .0
        .lock()
        .map_err(|_| "快捷翻译设置状态不可用。".to_owned())?
        .clone();

    if !settings.enabled {
        if let Some(current) = current_shortcut {
            app.global_shortcut()
                .unregister(current.as_str())
                .map_err(|error| format!("停用快捷翻译快捷键失败：{error}"))?;
            *state
                .0
                .lock()
                .map_err(|_| "快捷翻译设置状态不可用。".to_owned())? = None;
        }
        return Ok(());
    }

    if current_shortcut.as_deref() == Some(next_shortcut) {
        return Ok(());
    }

    register_shortcut(&app, next_shortcut)
        .map_err(|error| format!("注册快捷键失败，可能已被其他程序占用：{error}"))?;

    if let Some(current) = current_shortcut {
        if let Err(error) = app.global_shortcut().unregister(current.as_str()) {
            let _ = app.global_shortcut().unregister(next_shortcut);
            return Err(format!("替换原快捷键失败：{error}"));
        }
    }

    *state
        .0
        .lock()
        .map_err(|_| "快捷翻译设置状态不可用。".to_owned())? = Some(next_shortcut.to_owned());
    Ok(())
}
