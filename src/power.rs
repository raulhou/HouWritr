//! Keep Windows from slowing HouWritr down as a "background" process.
//!
//! Windows applies power throttling (EcoQoS: lower clock speeds) to processes
//! it considers to run in the background, and the tray process that does the
//! transcription, speaker labelling and cleanup never has a window in the
//! foreground. Speaker labelling on the CPU took 62 s in one run and 210 to
//! 230 s in three later runs of the same file with the same code, which fits
//! throttling. The same opt-out is applied to the local model server.

use tracing::{info, warn};
use windows_sys::Win32::Foundation::HANDLE;
use windows_sys::Win32::System::Threading::{
    GetCurrentProcess, ProcessPowerThrottling, SetProcessInformation,
    PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
    PROCESS_POWER_THROTTLING_STATE,
};

/// Ask Windows never to slow `process` down to save power. Returns whether
/// Windows accepted the request.
pub fn opt_out_of_throttling(process: HANDLE) -> bool {
    let state = PROCESS_POWER_THROTTLING_STATE {
        Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        // Control the execution speed setting, and turn it off.
        ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        StateMask: 0,
    };
    // SAFETY: `state` is a valid PROCESS_POWER_THROTTLING_STATE that lives
    // for the call, and its size is passed alongside it.
    unsafe {
        SetProcessInformation(
            process,
            ProcessPowerThrottling,
            (&state as *const PROCESS_POWER_THROTTLING_STATE).cast(),
            std::mem::size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32,
        ) != 0
    }
}

/// [`opt_out_of_throttling`] for HouWritr's own process.
pub fn opt_out_for_this_process() {
    // SAFETY: GetCurrentProcess returns a pseudo handle that needs no closing.
    if opt_out_of_throttling(unsafe { GetCurrentProcess() }) {
        info!("asked Windows not to power throttle this process");
    } else {
        warn!(
            error = %std::io::Error::last_os_error(),
            "could not opt out of Windows power throttling"
        );
    }
}
