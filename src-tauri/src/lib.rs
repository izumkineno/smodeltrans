mod backend;
mod model_config;
mod model_support;
mod models;
mod output;

use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let resource_root = app.path().resource_dir().ok();
            let config_path = app
                .path()
                .app_config_dir()
                .ok()
                .map(|directory| directory.join("model-settings.json"));
            let state = backend::BackendState::new_with_resource_root_and_config(
                resource_root,
                config_path,
            );
            state.start_idle_monitor();
            let live_manager = backend::live::LiveSessionManager::new(state.clone());
            app.manage(state);
            app.manage(live_manager);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            backend::commands::get_backend_status,
            backend::commands::get_model_runtime_status,
            backend::commands::control_model,
            backend::commands::update_backend_settings,
            backend::commands::list_model_catalog,
            backend::commands::save_model_catalog,
            backend::commands::translate_image,
            backend::commands::translate_text,
            backend::commands::ocr_image,
            backend::live::list_capture_windows,
            backend::live::begin_live_selection,
            backend::live::confirm_live_selection,
            backend::live::begin_live_roi_update,
            backend::live::cancel_live_selection,
            backend::live::get_live_session_status,
            backend::live::pause_live_session,
            backend::live::resume_live_session,
            backend::live::interrupt_live_translation,
            backend::live::stop_live_session
        ])
        .run(tauri::generate_context!())
        .expect("error while running smodeltrans application");
}
