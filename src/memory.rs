//! Hand memory back to Windows once HouWritr has nothing to do.
//!
//! The tray process stays running all day. After a job it still holds pages
//! it no longer needs: freed heap blocks and pages of the speech engine and
//! audio buffers that were just released. Windows would only take them back
//! under memory pressure; HouWritr returns them as soon as it goes idle, and
//! pages it needs again come back on first use.

use tracing::info;
use windows_sys::Win32::System::Memory::{GetProcessHeap, HeapCompact};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, SetProcessWorkingSetSize};

/// Return free heap memory and trim the working set. Called once per idle
/// period (after the speech engine is released), never on a timer.
pub fn release_idle_memory() {
    // SAFETY: GetProcessHeap returns the process heap, which is serialised,
    // so compacting it while other threads allocate is allowed.
    // GetCurrentProcess returns a pseudo handle that needs no closing, and
    // (usize::MAX, usize::MAX) asks Windows to remove as many pages as
    // possible from the working set.
    let trimmed = unsafe {
        HeapCompact(GetProcessHeap(), 0);
        SetProcessWorkingSetSize(GetCurrentProcess(), usize::MAX, usize::MAX) != 0
    };
    info!(trimmed, "idle: memory handed back to Windows");
}
