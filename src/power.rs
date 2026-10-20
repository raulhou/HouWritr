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

/// What Windows reports about the power supply.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PowerStatus {
    /// Running on the battery (not plugged in).
    pub on_battery: bool,
    /// Windows' battery saver is on.
    pub battery_saver: bool,
}

/// Read the power status. A desktop, or a status Windows does not know, counts
/// as plugged in with battery saver off.
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
    status_from(raw.ACLineStatus, raw.SystemStatusFlag)
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

/// Whether work waits right now: power saving is on and the PC runs on its
/// battery.
pub fn holds_work(manual: bool, automatic: bool, status: PowerStatus) -> bool {
    saving_on(manual, automatic, status) && status.on_battery
}

/// `ACLineStatus`: 0 offline (on battery), 1 online, 255 unknown.
/// `SystemStatusFlag`: 1 when battery saver is on.
fn status_from(line: u8, system_flag: u8) -> PowerStatus {
    PowerStatus {
        on_battery: line == 0,
        battery_saver: system_flag == 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_an_offline_ac_line_means_battery_and_flag_one_means_saver() {
        assert!(status_from(0, 0).on_battery);
        assert!(!status_from(1, 0).on_battery);
        assert!(!status_from(255, 0).on_battery);
        assert!(status_from(0, 1).battery_saver);
        assert!(!status_from(0, 0).battery_saver);
        assert_eq!(status_from(255, 0), PowerStatus::default());
    }

    #[test]
    fn work_waits_only_with_power_saving_on_and_on_battery() {
        let battery = status_from(0, 0);
        let saver = status_from(0, 1);
        let plugged = status_from(1, 0);
        // The toggle decides when it is not automatic.
        assert!(holds_work(true, false, battery));
        assert!(!holds_work(false, false, battery));
        assert!(!holds_work(true, false, plugged));
        // Automatic: only Windows' battery saver counts, not the toggle.
        assert!(!holds_work(true, true, battery));
        assert!(holds_work(false, true, saver));
        assert!(saving_on(false, true, saver));
        assert!(!saving_on(true, true, battery));
    }
}
