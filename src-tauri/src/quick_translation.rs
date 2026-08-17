use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tauri::{Emitter, Manager, State};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

use crate::selection;

const DEFAULT_SHORTCUT: &str = "CommandOrControl+Alt+E";
const QUICK_TRANSLATION_WINDOW_LABEL: &str = "quick-translation";
const QUICK_TRANSLATION_EVENT: &str = "quick-translation-request";
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
                Ok(text) => QuickTranslationEvent {
                    text: Some(text),
                    error: None,
                },
                Err(error) => QuickTranslationEvent {
                    text: None,
                    error: Some(error.to_string()),
                },
            };

            if let Some(window) = app.get_webview_window(QUICK_TRANSLATION_WINDOW_LABEL) {
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
