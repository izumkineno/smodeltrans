use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SelectionReadError {
    NoTextSelection,
    UnsupportedPlatform,
    System(String),
}

impl Display for SelectionReadError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoTextSelection => formatter.write_str("未检测到可访问的文字选区。"),
            Self::UnsupportedPlatform => {
                formatter.write_str("当前平台暂不支持读取其他应用的文字选区。")
            }
            Self::System(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for SelectionReadError {}

pub(crate) fn read_selected_text() -> Result<String, SelectionReadError> {
    #[cfg(target_os = "windows")]
    {
        let worker = std::thread::Builder::new()
            .name("smodeltrans-uia-selection".to_owned())
            .spawn(read_selected_text_on_current_thread)
            .map_err(|error| {
                SelectionReadError::System(format!("启动选区读取线程失败：{error}"))
            })?;
        return worker
            .join()
            .map_err(|_| SelectionReadError::System("选区读取线程异常退出。".to_owned()))?;
    }

    #[cfg(not(target_os = "windows"))]
    {
        Err(SelectionReadError::UnsupportedPlatform)
    }
}

#[cfg(target_os = "windows")]
fn read_selected_text_on_current_thread() -> Result<String, SelectionReadError> {
    use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx};

    unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }
        .ok()
        .map_err(|error| {
            SelectionReadError::System(format!("初始化 UI Automation COM 线程失败：{error}"))
        })?;
    let _com_guard = ComGuard;

    match read_selected_text_with_uia() {
        Ok(Some(text)) => Ok(text),
        Ok(None) => Err(SelectionReadError::NoTextSelection),
        Err(error) => Err(SelectionReadError::System(format!(
            "读取当前文字选区失败：{error}"
        ))),
    }
}

#[cfg(target_os = "windows")]
struct ComGuard;

#[cfg(target_os = "windows")]
impl Drop for ComGuard {
    fn drop(&mut self) {
        unsafe { windows::Win32::System::Com::CoUninitialize() };
    }
}

#[cfg(target_os = "windows")]
const MAX_UIA_ANCESTOR_DEPTH: usize = 16;

#[cfg(target_os = "windows")]
fn read_selected_text_with_uia() -> windows::core::Result<Option<String>> {
    use windows::Win32::System::Com::{CLSCTX_INPROC_SERVER, CoCreateInstance};
    use windows::Win32::UI::Accessibility::{
        CUIAutomation, IUIAutomation, IUIAutomationTextPattern, UIA_TextPatternId,
    };

    let automation: IUIAutomation =
        unsafe { CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER)? };
    let walker = unsafe { automation.RawViewWalker()? };
    let mut current = Some(unsafe { automation.GetFocusedElement()? });

    for _ in 0..MAX_UIA_ANCESTOR_DEPTH {
        let Some(element) = current.take() else {
            break;
        };

        if let Some(text_pattern) = unsafe {
            element
                .GetCurrentPatternAs::<IUIAutomationTextPattern>(UIA_TextPatternId)
                .ok()
        } {
            if let Some(text) = read_selected_text_from_pattern(&text_pattern) {
                return Ok(Some(text));
            }
        }

        current = unsafe { walker.GetParentElement(&element).ok() };
    }

    Ok(None)
}

#[cfg(target_os = "windows")]
fn read_selected_text_from_pattern(
    text_pattern: &windows::Win32::UI::Accessibility::IUIAutomationTextPattern,
) -> Option<String> {
    let ranges = unsafe { text_pattern.GetSelection().ok()? };
    let range_count = unsafe { ranges.Length().ok()? };
    let mut selected_text = String::new();

    for index in 0..range_count {
        let Ok(range) = (unsafe { ranges.GetElement(index) }) else {
            continue;
        };
        let Ok(text) = (unsafe { range.GetText(-1) }) else {
            continue;
        };
        let text = text.to_string();
        if text.chars().any(|character| !character.is_whitespace()) {
            if !selected_text.is_empty() {
                selected_text.push('\n');
            }
            selected_text.push_str(text.trim());
        }
    }

    normalize_selected_text(&selected_text)
}

fn normalize_selected_text(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_owned())
}

#[cfg(test)]
mod tests {
    use super::normalize_selected_text;

    #[test]
    fn rejects_empty_or_whitespace_selection() {
        assert_eq!(normalize_selected_text(""), None);
        assert_eq!(normalize_selected_text(" \n\t"), None);
    }

    #[test]
    fn trims_outer_selection_whitespace_but_preserves_content() {
        assert_eq!(
            normalize_selected_text("  Hello\n世界  "),
            Some("Hello\n世界".to_owned())
        );
    }
}
