use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SelectionBounds {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SelectedText {
    pub text: String,
    pub bounds: Option<SelectionBounds>,
}

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

pub(crate) fn read_selected_text() -> Result<SelectedText, SelectionReadError> {
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
fn read_selected_text_on_current_thread() -> Result<SelectedText, SelectionReadError> {
    use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx};

    unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }
        .ok()
        .map_err(|error| {
            SelectionReadError::System(format!("初始化 UI Automation COM 线程失败：{error}"))
        })?;
    let _com_guard = ComGuard;

    match read_selected_text_with_uia() {
        Ok(Some(selection)) => Ok(selection),
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
fn read_selected_text_with_uia() -> windows::core::Result<Option<SelectedText>> {
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
            if let Some(selection) = read_selected_text_from_pattern(&text_pattern) {
                return Ok(Some(selection));
            }
        }

        current = unsafe { walker.GetParentElement(&element).ok() };
    }

    Ok(None)
}

#[cfg(target_os = "windows")]
fn read_selected_text_from_pattern(
    text_pattern: &windows::Win32::UI::Accessibility::IUIAutomationTextPattern,
) -> Option<SelectedText> {
    let ranges = unsafe { text_pattern.GetSelection().ok()? };
    let range_count = unsafe { ranges.Length().ok()? };
    let mut selected_text = String::new();
    let mut selection_bounds = None;

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
            selection_bounds =
                merge_selection_bounds(selection_bounds, read_text_range_bounds(&range));
        }
    }

    normalize_selected_text(&selected_text).map(|text| SelectedText {
        text,
        bounds: selection_bounds,
    })
}

#[cfg(target_os = "windows")]
fn merge_selection_bounds(
    current: Option<SelectionBounds>,
    next: Option<SelectionBounds>,
) -> Option<SelectionBounds> {
    match (current, next) {
        (Some(current), Some(next)) => Some(SelectionBounds {
            left: current.left.min(next.left),
            top: current.top.min(next.top),
            right: current.right.max(next.right),
            bottom: current.bottom.max(next.bottom),
        }),
        (Some(current), None) => Some(current),
        (None, Some(next)) => Some(next),
        (None, None) => None,
    }
}

#[cfg(target_os = "windows")]
fn read_text_range_bounds(
    range: &windows::Win32::UI::Accessibility::IUIAutomationTextRange,
) -> Option<SelectionBounds> {
    let array = unsafe { range.GetBoundingRectangles().ok()? };
    if array.is_null() {
        return None;
    }

    let bounds = unsafe { selection_bounds_from_safe_array(array) };
    unsafe { destroy_safe_array(array) };
    bounds
}

#[cfg(target_os = "windows")]
unsafe fn selection_bounds_from_safe_array(
    array: *mut windows::Win32::System::Com::SAFEARRAY,
) -> Option<SelectionBounds> {
    let array = unsafe { array.as_ref()? };
    if array.cDims != 1
        || array.cbElements as usize != std::mem::size_of::<f64>()
        || array.pvData.is_null()
    {
        return None;
    }

    let values = unsafe {
        std::slice::from_raw_parts(
            array.pvData.cast::<f64>(),
            array.rgsabound[0].cElements as usize,
        )
    };
    let mut result: Option<(f64, f64, f64, f64)> = None;
    for rectangle in values.chunks_exact(4) {
        let [left, top, width, height] = rectangle else {
            continue;
        };
        if !left.is_finite()
            || !top.is_finite()
            || !width.is_finite()
            || !height.is_finite()
            || *width <= 0.0
            || *height <= 0.0
        {
            continue;
        }
        let right = *left + *width;
        let bottom = *top + *height;
        if !right.is_finite() || !bottom.is_finite() {
            continue;
        }
        result = Some(match result {
            Some((current_left, current_top, current_right, current_bottom)) => (
                current_left.min(*left),
                current_top.min(*top),
                current_right.max(right),
                current_bottom.max(bottom),
            ),
            None => (*left, *top, right, bottom),
        });
    }

    result.map(|(left, top, right, bottom)| SelectionBounds {
        left: left.floor().clamp(i32::MIN as f64, i32::MAX as f64) as i32,
        top: top.floor().clamp(i32::MIN as f64, i32::MAX as f64) as i32,
        right: right.ceil().clamp(i32::MIN as f64, i32::MAX as f64) as i32,
        bottom: bottom.ceil().clamp(i32::MIN as f64, i32::MAX as f64) as i32,
    })
}

#[cfg(target_os = "windows")]
unsafe fn destroy_safe_array(array: *mut windows::Win32::System::Com::SAFEARRAY) {
    #[link(name = "oleaut32")]
    unsafe extern "system" {
        fn SafeArrayDestroy(
            psa: *mut windows::Win32::System::Com::SAFEARRAY,
        ) -> windows::core::HRESULT;
    }

    let _ = unsafe { SafeArrayDestroy(array) };
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
