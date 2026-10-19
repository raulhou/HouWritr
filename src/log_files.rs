//! Where HouWritr writes its logs.
//!
//! * `logs\sessions\`: one file per run of the app. The settings and
//!   transcript windows the tray opens write into the tray's file, so one
//!   file tells the whole story of a session.
//! * `logs\recordings\`: one file per recording or transcript, holding every
//!   line logged while it was being worked on (a meeting keeps one file
//!   across all its parts).
//!
//! The same lines also go to the session file.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use parking_lot::Mutex;
use tracing::{info, warn};

use crate::paths;

/// Set by the tray for the windows it opens, so they log into its file.
const SESSION_LOG_VARIABLE: &str = "HOUWRITR_SESSION_LOG";

/// Session logs kept; older ones are deleted when the tray starts.
const KEEP_SESSIONS: usize = 30;
/// Recording logs kept; older ones are deleted when the tray starts.
const KEEP_RECORDINGS: usize = 300;

pub fn sessions_dir() -> PathBuf {
    paths::log_dir().join("sessions")
}

pub fn recordings_dir() -> PathBuf {
    paths::log_dir().join("recordings")
}

/// The session file this process logs to: the file of the tray that opened
/// it, or a new one named after the time, `mode` and process id. The tray
/// (`owner`) passes its file on to the windows it opens.
pub fn session_log(mode: &str, owner: bool) -> PathBuf {
    let directory = sessions_dir();
    let inherited = (!owner)
        .then(|| std::env::var_os(SESSION_LOG_VARIABLE))
        .flatten()
        .map(PathBuf::from)
        // Only ever a file in the sessions folder.
        .filter(|path| path.parent() == Some(directory.as_path()) && is_log(path));
    let path = inherited.unwrap_or_else(|| {
        directory.join(format!(
            "{}_{mode}_{}.log",
            file_stamp(&utc_now()),
            std::process::id()
        ))
    });
    if owner {
        std::env::set_var(SESSION_LOG_VARIABLE, &path);
    }
    path
}

/// What a recording log is for; part of its file name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordingKind {
    Dictation,
    File,
    /// One part of a meeting; all parts of meeting `session` share a file.
    Meeting {
        session: u64,
        last: bool,
    },
    /// A background retry of AI cleanup or notes.
    Retry,
}

impl RecordingKind {
    fn label(self) -> &'static str {
        match self {
            RecordingKind::Dictation => "dictation",
            RecordingKind::File => "file",
            RecordingKind::Meeting { .. } => "meeting",
            RecordingKind::Retry => "retry",
        }
    }
}

/// The recording log being written, and the job it belongs to.
struct Current {
    job: u64,
    file: File,
}

static CURRENT: Mutex<Option<Current>> = Mutex::new(None);

/// The file of each meeting still in progress.
static MEETINGS: Mutex<Vec<(u64, PathBuf)>> = Mutex::new(Vec::new());

/// Start the log of job `job`. Every line logged from now until [`end`] (or
/// the next `begin`) also goes to it.
pub fn begin(job: u64, kind: RecordingKind) {
    let path = recording_path(job, kind);
    let opened = fs::create_dir_all(recordings_dir())
        .and_then(|()| OpenOptions::new().create(true).append(true).open(&path));
    // Nothing is logged while the lock is held: the log writer takes it too.
    let failure = match opened {
        Ok(file) => {
            *CURRENT.lock() = Some(Current { job, file });
            None
        }
        Err(error) => {
            *CURRENT.lock() = None;
            Some(error)
        }
    };
    match failure {
        None => {
            info!(id = job, kind = kind.label(), log = %path.display(), "recording log started")
        }
        Some(error) => {
            warn!(error = %error, log = %path.display(), "could not open the recording log")
        }
    }
}

/// Stop writing the log of job `job`, once the app has saved its result.
pub fn end(job: u64) {
    let mut current = CURRENT.lock();
    if current.as_ref().is_some_and(|current| current.job == job) {
        *current = None;
    }
}

fn recording_path(job: u64, kind: RecordingKind) -> PathBuf {
    let new = || {
        recordings_dir().join(format!(
            "{}_{}_{job}.log",
            file_stamp(&utc_now()),
            kind.label()
        ))
    };
    let RecordingKind::Meeting { session, last } = kind else {
        return new();
    };
    let mut meetings = MEETINGS.lock();
    let path = match meetings.iter().find(|(id, _)| *id == session) {
        Some((_, path)) => path.clone(),
        None => {
            let path = new();
            meetings.push((session, path.clone()));
            path
        }
    };
    if last {
        meetings.retain(|(id, _)| *id != session);
    }
    path
}

/// Writes log lines to the current recording log, if there is one.
pub struct RecordingWriter;

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for RecordingWriter {
    type Writer = RecordingWriter;

    fn make_writer(&'a self) -> Self::Writer {
        RecordingWriter
    }
}

impl Write for RecordingWriter {
    fn write(&mut self, line: &[u8]) -> io::Result<usize> {
        if let Some(current) = CURRENT.lock().as_mut() {
            // A full disk must not break logging elsewhere.
            let _ = current.file.write_all(line);
        }
        Ok(line.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Delete the oldest session and recording logs beyond what is kept.
pub fn prune() -> io::Result<()> {
    prune_dir(&sessions_dir(), KEEP_SESSIONS)?;
    prune_dir(&recordings_dir(), KEEP_RECORDINGS)
}

/// Keep the newest `keep` logs in `directory`, newest by the time at the
/// start of the name. Only files named like this module names them count.
fn prune_dir(directory: &Path, keep: usize) -> io::Result<()> {
    if !directory.is_dir() {
        return Ok(());
    }
    let mut logs: Vec<PathBuf> = fs::read_dir(directory)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && is_log(path) && has_stamp(path))
        .collect();
    // The stamp sorts as text: newest first.
    logs.sort_by(|left, right| right.file_name().cmp(&left.file_name()));
    for path in logs.into_iter().skip(keep) {
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

pub fn is_log(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("log"))
}

/// Whether the name starts with a stamp like `2026-10-17_22-17-08Z_`.
fn has_stamp(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let bytes = name.as_bytes();
    bytes.len() > 21
        && bytes[..21]
            .iter()
            .enumerate()
            .all(|(index, byte)| match index {
                4 | 7 | 13 | 16 => *byte == b'-',
                10 | 20 => *byte == b'_',
                19 => *byte == b'Z',
                _ => byte.is_ascii_digit(),
            })
}

/// `2026-10-17 22:17:08` as `2026-10-17_22-17-08Z`, for file names.
fn file_stamp(utc: &str) -> String {
    format!("{}Z", utc.replace(' ', "_").replace(':', "-"))
}

/// The time now in UTC, as `2026-10-17 22:17:08`.
pub fn utc_now() -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    utc_text(seconds)
}

fn utc_text(seconds: u64) -> String {
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
    use super::*;

    #[test]
    fn file_stamps_sort_by_time_and_are_recognised() {
        assert_eq!(utc_text(0), "1970-01-01 00:00:00");
        assert_eq!(utc_text(1_792_275_428), "2026-10-17 22:17:08");
        let stamp = file_stamp(&utc_text(1_792_275_428));
        assert_eq!(stamp, "2026-10-17_22-17-08Z");
        assert!(has_stamp(Path::new(&format!("{stamp}_tray_4388.log"))));
        assert!(!has_stamp(Path::new("houwritr.2026-10-17.log")));
        assert!(!has_stamp(Path::new("2026-10-17_22-17-08Z")));
    }

    #[test]
    fn only_the_oldest_stamped_logs_are_pruned() {
        let temp = tempfile::tempdir().unwrap();
        for minute in 10..20 {
            fs::write(
                temp.path()
                    .join(format!("2026-10-17_22-{minute}-00Z_file_1.log")),
                b"log",
            )
            .unwrap();
        }
        fs::write(temp.path().join("notes.log"), b"mine").unwrap();
        prune_dir(temp.path(), 3).unwrap();
        let mut left: Vec<String> = fs::read_dir(temp.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        left.sort();
        assert_eq!(
            left,
            [
                "2026-10-17_22-17-00Z_file_1.log",
                "2026-10-17_22-18-00Z_file_1.log",
                "2026-10-17_22-19-00Z_file_1.log",
                "notes.log",
            ]
        );
    }

    #[test]
    fn meeting_parts_share_one_log_until_the_last() {
        let first = recording_path(
            1,
            RecordingKind::Meeting {
                session: 77,
                last: false,
            },
        );
        let second = recording_path(
            2,
            RecordingKind::Meeting {
                session: 77,
                last: true,
            },
        );
        assert_eq!(first, second);
        assert!(MEETINGS.lock().iter().all(|(id, _)| *id != 77));
    }
}
