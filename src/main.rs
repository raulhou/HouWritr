//! HouWritr: native push to talk voice to text for Windows.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod about;
mod app;
mod asr;
mod audio;
mod autostart;
mod cleanup;
mod copilot_auth;
mod credentials;
mod diagnostics;
mod enhance;
mod hotkey;
mod key_hook;
mod local_llm;
mod media;
mod meeting;
mod model_manager;
mod overlay;
mod paste;
mod paths;
mod power;
mod self_check;
mod settings;
mod settings_ui;
mod single_instance;
mod sounds;
mod transcript_window;
mod tray;
mod worker;

use anyhow::{Context, Result};
use std::ffi::OsStr;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if is_self_check_mode(arguments.iter()) {
        return self_check::run();
    }
    if is_transcript_window_mode(arguments.iter()) {
        let _log_guard = init_tracing(false)?;
        info!(
            version = env!("CARGO_PKG_VERSION"),
            pid = std::process::id(),
            mode = "transcript-window",
            "process starting"
        );
        return transcript_window::run_window();
    }
    let settings_mode = is_settings_mode(arguments.iter());
    let about_mode = is_about_mode(arguments.iter());

    // Settings and About share one subprocess and single-instance boundary.
    if settings_mode || about_mode {
        let settings_instance = single_instance::SingleInstance::acquire_settings()?;
        let _log_guard = init_tracing(false)?;
        info!(
            version = env!("CARGO_PKG_VERSION"),
            pid = std::process::id(),
            architecture = std::env::consts::ARCH,
            mode = "settings",
            "process starting"
        );
        let Some(_settings_instance) = settings_instance else {
            info!("settings window already running; exiting");
            return Ok(());
        };
        let result = settings_ui::run_dialog(about_mode);
        info!("settings process stopping");
        return result;
    }

    let tray_instance = single_instance::SingleInstance::acquire_tray()?;
    // Only the one running tray process tidies old logs; the windows it
    // opens write to the same file and must never delete it.
    let _log_guard = init_tracing(tray_instance.is_some())?;
    let Some(_instance) = tray_instance else {
        info!("tray instance already running; exiting");
        return Ok(());
    };

    info!(
        version = env!("CARGO_PKG_VERSION"),
        pid = std::process::id(),
        architecture = std::env::consts::ARCH,
        mode = "tray",
        "process starting"
    );
    // Install the global low-level keyboard hook before any subsystem starts
    // polling for keys. It tracks physical key state across focus changes,
    // which GetAsyncKeyState cannot.
    key_hook::install_once();
    // The tray process does the heavy work (transcription, speaker labels,
    // cleanup) without ever having a foreground window.
    power::opt_out_for_this_process();
    let result = app::run();
    if let Err(ref failure) = result {
        error!(error = %failure, "tray process stopping after error");
    } else {
        info!("tray process stopping");
    }
    result
}

fn is_settings_mode<T: AsRef<OsStr>>(args: impl IntoIterator<Item = T>) -> bool {
    args.into_iter()
        .any(|arg| arg.as_ref().eq_ignore_ascii_case(OsStr::new("--settings")))
}

fn is_transcript_window_mode<T: AsRef<OsStr>>(args: impl IntoIterator<Item = T>) -> bool {
    args.into_iter().any(|arg| {
        arg.as_ref()
            .eq_ignore_ascii_case(OsStr::new("--transcript-window"))
    })
}

fn is_about_mode<T: AsRef<OsStr>>(args: impl IntoIterator<Item = T>) -> bool {
    args.into_iter()
        .any(|arg| arg.as_ref().eq_ignore_ascii_case(OsStr::new("--about")))
}

fn is_self_check_mode<T: AsRef<OsStr>>(args: impl IntoIterator<Item = T>) -> bool {
    args.into_iter().any(|arg| {
        arg.as_ref()
            .eq_ignore_ascii_case(OsStr::new("--self-check"))
    })
}

fn init_tracing(prune: bool) -> Result<tracing_appender::non_blocking::WorkerGuard> {
    let log_dir = paths::log_dir();
    std::fs::create_dir_all(&log_dir)
        .with_context(|| format!("create log directory {}", log_dir.display()))?;
    if prune {
        diagnostics::prune_logs()?;
    }
    record_crashes(log_dir.clone());

    let file_appender = tracing_appender::rolling::RollingFileAppender::builder()
        .rotation(tracing_appender::rolling::Rotation::DAILY)
        .filename_prefix("openwritr")
        .filename_suffix("log")
        // No `max_log_files`: the library's own pruning counts every
        // `openwritr*.log` file, copies included, by creation time, and ran
        // in every process; it deleted the log the tray was still writing.
        // `diagnostics::prune_logs` does it safely instead.
        .build(&log_dir)
        .with_context(|| format!("initialize log appender in {}", log_dir.display()))?;
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(non_blocking)
        .with_ansi(false)
        .with_thread_ids(true)
        .with_thread_names(true);
    let console_layer =
        cfg!(debug_assertions).then(|| tracing_subscriber::fmt::layer().with_ansi(false));

    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "openwritr=info".into()))
        .with(file_layer)
        .with(console_layer)
        .try_init()
        .context("initialize tracing subscriber")?;

    Ok(guard)
}

/// Write every panic straight to `crash.log` in the log folder before the
/// process aborts (release builds abort on panic). The normal log is written
/// in the background, so its last lines, and the panic itself, were lost.
fn record_crashes(log_dir: std::path::PathBuf) {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        use std::io::Write;
        let thread = std::thread::current();
        let report = format!(
            "{} UTC pid {} thread {}: {}\n{}\n\n",
            utc_now(),
            std::process::id(),
            thread.name().unwrap_or("unnamed"),
            info,
            std::backtrace::Backtrace::force_capture()
        );
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_dir.join("crash.log"))
        {
            let _ = file.write_all(report.as_bytes());
            let _ = file.flush();
        }
        error!("crash: {info}");
        previous(info);
    }));
}

/// The current UTC time as `YYYY-MM-DD hh:mm:ss`, without a date library.
fn utc_now() -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    let (days, rest) = (seconds / 86_400, seconds % 86_400);
    // Days since 1970-01-01 to a calendar date (Howard Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}:{:02}",
        rest / 3_600,
        rest % 3_600 / 60,
        rest % 60
    )
}

#[cfg(test)]
mod tests {
    use super::{is_about_mode, is_self_check_mode, is_settings_mode, is_transcript_window_mode};
    use std::ffi::OsString;

    #[test]
    fn settings_mode_is_selected_before_tray_startup() {
        assert!(is_settings_mode([OsString::from("--settings")]));
        assert!(is_settings_mode([OsString::from("--SETTINGS")]));
        assert!(!is_settings_mode([OsString::from("--unknown")]));
    }

    #[test]
    fn self_check_mode_is_selected_before_tray_startup() {
        assert!(is_self_check_mode([OsString::from("--self-check")]));
        assert!(is_self_check_mode([OsString::from("--SELF-CHECK")]));
        assert!(!is_self_check_mode([OsString::from("--settings")]));
    }

    #[test]
    fn transcript_window_mode_is_selected_before_tray_startup() {
        assert!(is_transcript_window_mode([OsString::from(
            "--transcript-window"
        )]));
        assert!(!is_transcript_window_mode([OsString::from("--settings")]));
    }

    #[test]
    fn about_mode_is_selected_before_tray_startup() {
        assert!(is_about_mode([OsString::from("--about")]));
        assert!(is_about_mode([OsString::from("--ABOUT")]));
        assert!(!is_about_mode([OsString::from("--settings")]));
    }
}
