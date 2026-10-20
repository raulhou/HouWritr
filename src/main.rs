//! HouWritr: meeting recording and audio/video transcription for Windows.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod about;
mod app;
mod asr;
mod autostart;
mod cleanup;
mod copilot_auth;
mod credentials;
mod diagnostics;
mod enhance;
mod jobs_window;
mod local_llm;
mod log_files;
mod media;
mod meeting;
mod model_manager;
mod overlay;
mod paths;
mod power;
mod self_check;
mod settings;
mod settings_ui;
mod single_instance;
mod spool;
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
    if arguments
        .iter()
        .any(|arg| arg.eq_ignore_ascii_case(OsStr::new("--jobs-window")))
    {
        let _log_guard = init_tracing("jobs-window", false)?;
        info!(
            version = env!("CARGO_PKG_VERSION"),
            pid = std::process::id(),
            mode = "jobs-window",
            build = build_id(),
            "process starting"
        );
        return jobs_window::run_window();
    }
    if is_transcript_window_mode(arguments.iter()) {
        let _log_guard = init_tracing("transcript-window", false)?;
        info!(
            version = env!("CARGO_PKG_VERSION"),
            pid = std::process::id(),
            mode = "transcript-window",
            build = build_id(),
            "process starting"
        );
        return transcript_window::run_window();
    }
    let settings_mode = is_settings_mode(arguments.iter());
    let about_mode = is_about_mode(arguments.iter());

    // Settings and About share one subprocess and single-instance boundary.
    if settings_mode || about_mode {
        let settings_instance = single_instance::SingleInstance::acquire_settings()?;
        let _log_guard = init_tracing("settings", false)?;
        info!(
            version = env!("CARGO_PKG_VERSION"),
            pid = std::process::id(),
            architecture = std::env::consts::ARCH,
            mode = "settings",
            build = build_id(),
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
    // Only the one running tray process starts a session log and tidies old
    // logs; the windows it opens write into its file.
    let _log_guard = init_tracing("tray", tray_instance.is_some())?;
    let Some(_instance) = tray_instance else {
        info!("tray instance already running; exiting");
        return Ok(());
    };

    info!(
        version = env!("CARGO_PKG_VERSION"),
        pid = std::process::id(),
        architecture = std::env::consts::ARCH,
        mode = "tray",
        build = build_id(),
        "process starting"
    );
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

fn init_tracing(mode: &str, owner: bool) -> Result<tracing_appender::non_blocking::WorkerGuard> {
    let log_dir = paths::log_dir();
    let sessions = log_files::sessions_dir();
    std::fs::create_dir_all(&sessions)
        .with_context(|| format!("create log directory {}", sessions.display()))?;
    if owner {
        diagnostics::prune_logs()?;
    }
    record_crashes(log_dir.clone());

    // One file per session; the tray's windows append to the tray's file.
    let session_log = log_files::session_log(mode, owner);
    let stem = session_log
        .file_stem()
        .and_then(|stem| stem.to_str())
        .context("session log has no file name")?;
    let file_appender = tracing_appender::rolling::RollingFileAppender::builder()
        .rotation(tracing_appender::rolling::Rotation::NEVER)
        .filename_prefix(stem)
        .filename_suffix("log")
        .build(&sessions)
        .with_context(|| format!("open {}", session_log.display()))?;
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(non_blocking)
        .with_ansi(false)
        .with_thread_ids(true)
        .with_thread_names(true);
    // The same lines, while a recording or transcript is worked on, also go
    // to that recording's own log (tray only: the windows run no jobs).
    let recording_layer = owner.then(|| {
        tracing_subscriber::fmt::layer()
            .with_writer(log_files::RecordingWriter)
            .with_ansi(false)
            .with_thread_ids(true)
            .with_thread_names(true)
    });
    let console_layer =
        cfg!(debug_assertions).then(|| tracing_subscriber::fmt::layer().with_ansi(false));

    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "houwritr=info".into()))
        .with(file_layer)
        .with(recording_layer)
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
            log_files::utc_now(),
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

/// The commit this binary was built from (CI sets `GITHUB_SHA`), so a log
/// shows exactly which build ran; every build shares one version number.
fn build_id() -> &'static str {
    match option_env!("GITHUB_SHA") {
        Some(sha) if sha.len() >= 7 => &sha[..7],
        _ => "local",
    }
}
