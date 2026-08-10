#[cfg(not(target_os = "windows"))]
#[path = "platform_stub.rs"]
mod implementation;
#[cfg(target_os = "windows")]
#[path = "platform_windows.rs"]
mod implementation;

pub(super) use implementation::*;
