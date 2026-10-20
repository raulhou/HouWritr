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
use windows_sys::Win32::System::Power::{GetSystemPowerStatus, SYSTEM_POWER_STATUS};
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

/// What Windows reports about its power saving mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PowerStatus {
    /// Windows' power saving mode (battery saver, called energy saver on
    /// newer Windows) is on.
    pub battery_saver: bool,
}

/// Read the power status. A status Windows does not report counts as power
/// saving off.
pub fn status() -> PowerStatus {
    let mut raw = SYSTEM_POWER_STATUS {
        ACLineStatus: 255,
        BatteryFlag: 0,
        BatteryLifePercent: 0,
        SystemStatusFlag: 0,
        BatteryLifeTime: 0,
        BatteryFullLifeTime: 0,
    };
    // SAFETY: `raw` is a valid SYSTEM_POWER_STATUS that lives for the call.
    if unsafe { GetSystemPowerStatus(&mut raw) } == 0 {
        return PowerStatus::default();
    }
    status_from(raw.SystemStatusFlag)
}

/// Whether power saving is on: the toggle, or Windows' battery saver when
/// `automatic` says to follow it.
pub fn saving_on(manual: bool, automatic: bool, status: PowerStatus) -> bool {
    if automatic {
        status.battery_saver
    } else {
        manual
    }
}

/// `SystemStatusFlag`: 1 when battery saver is on. Whether the PC is plugged
/// in does not matter: Windows itself decides when its power saving mode is
/// on.
fn status_from(system_flag: u8) -> PowerStatus {
    PowerStatus {
        battery_saver: system_flag == 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_flag_one_means_windows_power_saving_is_on() {
        assert!(status_from(1).battery_saver);
        assert!(!status_from(0).battery_saver);
        assert!(!status_from(255).battery_saver);
        assert_eq!(status_from(0), PowerStatus::default());
    }

    #[test]
    fn the_toggle_decides_unless_following_windows() {
        let saver = status_from(1);
        let normal = status_from(0);
        assert!(saving_on(true, false, normal));
        assert!(saving_on(true, false, saver));
        assert!(!saving_on(false, false, saver));
        // Following Windows: only its power saving mode counts, not the
        // toggle.
        assert!(saving_on(false, true, saver));
        assert!(!saving_on(true, true, normal));
    }
}
