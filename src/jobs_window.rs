//! The Jobs window: everything HouWritr is doing or will do, with a way to
//! stop it.
//!
//! Like the transcript window it is the same exe started with a flag. The
//! tray sends it the whole list as one JSON line whenever the list changes
//! (standard input) and reads back `stop <key>` lines (standard output).

use crate::settings_ui;
use anyhow::{Context, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::os::windows::process::CommandExt;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::time::Duration;
use tracing::warn;

const CREATE_NO_WINDOW: u32 = 0x0800_0000;
const STOP_PREFIX: &str = "stop ";

/// One line of the list.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JobRow {
    /// What a stop request names, e.g. `job:7` or `retry:2`.
    pub key: String,
    /// "File", "Meeting", "Cleanup retry"…
    pub kind: String,
    pub name: String,
    /// "Transcribing 3/12", "Queued", "Next try in about 5 min"…
    pub state: String,
    /// Progress of the current step, when it is known.
    pub fraction: Option<f32>,
    /// What the button says, when the job can be stopped from here.
    pub stop_label: Option<String>,
}

// ---------------------------------------------------------------------------
// Tray side.
// ---------------------------------------------------------------------------

/// The open Jobs window, seen from the tray.
pub struct JobsWindow {
    rows: Sender<Vec<JobRow>>,
    stops: Arc<Mutex<Vec<String>>>,
    open: Arc<AtomicBool>,
    last: Option<Vec<JobRow>>,
}

impl JobsWindow {
    /// Start the window process (on its own thread: starting a process can
    /// block for seconds on Windows ARM64).
    pub fn open() -> std::io::Result<Self> {
        let (rows, incoming) = mpsc::channel();
        let stops = Arc::new(Mutex::new(Vec::new()));
        let open = Arc::new(AtomicBool::new(true));
        let thread_stops = stops.clone();
        let thread_open = open.clone();
        std::thread::Builder::new()
            .name("jobs-window".into())
            .spawn(move || {
                if let Err(error) = pump(incoming, thread_stops) {
                    warn!(error = %error, "jobs window ended early");
                }
                thread_open.store(false, Ordering::Release);
            })?;
        Ok(Self {
            rows,
            stops,
            open,
            last: None,
        })
    }

    /// Whether the window is still open.
    pub fn is_open(&self) -> bool {
        self.open.load(Ordering::Acquire)
    }

    /// Show `rows`, unless the window already shows exactly that.
    pub fn show(&mut self, rows: Vec<JobRow>) {
        if self.last.as_ref() == Some(&rows) {
            return;
        }
        if self.rows.send(rows.clone()).is_err() {
            self.open.store(false, Ordering::Release);
        }
        self.last = Some(rows);
    }

    /// The keys of the jobs the user asked to stop since the last call.
    pub fn take_stops(&self) -> Vec<String> {
        std::mem::take(&mut *self.stops.lock())
    }
}

fn pump(rows: Receiver<Vec<JobRow>>, stops: Arc<Mutex<Vec<String>>>) -> Result<()> {
    let exe = std::env::current_exe().context("resolve current executable")?;
    let mut child = Command::new(exe)
        .arg("--jobs-window")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
        .context("start jobs window")?;
    if let Some(output) = child.stdout.take() {
        let _ = std::thread::Builder::new()
            .name("jobs-window-out".into())
            .spawn(move || {
                for line in BufReader::new(output).lines() {
                    let Ok(line) = line else { break };
                    if let Some(key) = parse_stop(&line) {
                        stops.lock().push(key);
                    }
                }
            });
    }
    let mut stdin = child.stdin.take().context("jobs window stdin")?;
    for list in rows {
        let mut line = serde_json::to_string(&list).context("encode jobs")?;
        line.push('\n');
        // A failed write means the user closed the window.
        if stdin.write_all(line.as_bytes()).is_err() || stdin.flush().is_err() {
            break;
        }
    }
    Ok(())
}

fn parse_stop(line: &str) -> Option<String> {
    line.trim()
        .strip_prefix(STOP_PREFIX)
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_string)
}

// ---------------------------------------------------------------------------
// Window process.
// ---------------------------------------------------------------------------

enum Incoming {
    Rows(Vec<JobRow>),
    ParentClosed,
}

pub fn run_window() -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(settings_ui::scaled_size([560.0, 340.0]))
            .with_min_inner_size(settings_ui::scaled_size([420.0, 220.0]))
            .with_resizable(true)
            .with_title("HouWritr — Jobs"),
        ..Default::default()
    };
    eframe::run_native(
        "HouWritr — Jobs",
        options,
        Box::new(|cc| {
            settings_ui::apply_dark_theme(&cc.egui_ctx);
            let (sender, receiver) = mpsc::channel();
            let context = cc.egui_ctx.clone();
            std::thread::Builder::new()
                .name("jobs-window-stdin".into())
                .spawn(move || {
                    let stdin = std::io::stdin();
                    for line in BufReader::new(stdin.lock()).lines() {
                        let Ok(line) = line else { break };
                        if let Ok(rows) = serde_json::from_str::<Vec<JobRow>>(&line) {
                            if sender.send(Incoming::Rows(rows)).is_err() {
                                return;
                            }
                            context.request_repaint();
                        }
                    }
                    let _ = sender.send(Incoming::ParentClosed);
                    context.request_repaint();
                })
                .ok();
            Ok(Box::new(JobsApp {
                rows: Vec::new(),
                incoming: receiver,
                parent_gone: false,
                stopping: Vec::new(),
            }))
        }),
    )
    .map_err(|error| anyhow::anyhow!("jobs window failed: {error}"))
}

struct JobsApp {
    rows: Vec<JobRow>,
    incoming: Receiver<Incoming>,
    parent_gone: bool,
    /// Keys whose Stop was pressed (the button is not offered again).
    stopping: Vec<String>,
}

impl JobsApp {
    fn drain(&mut self) {
        loop {
            match self.incoming.try_recv() {
                Ok(Incoming::Rows(rows)) => {
                    // Forget presses for jobs that are gone.
                    self.stopping
                        .retain(|key| rows.iter().any(|row| &row.key == key));
                    self.rows = rows;
                }
                Ok(Incoming::ParentClosed) => self.parent_gone = true,
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
    }

    fn request_stop(&mut self, key: &str) {
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{STOP_PREFIX}{key}");
        let _ = out.flush();
        self.stopping.push(key.to_string());
    }
}

impl eframe::App for JobsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain();
        if !self.parent_gone {
            ctx.request_repaint_after(Duration::from_millis(500));
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Jobs");
            ui.add_space(6.0);
            if self.parent_gone {
                settings_ui::inline_error(ui, "HouWritr was closed.");
                return;
            }
            if self.rows.is_empty() {
                ui.weak("Nothing is running or waiting.");
                return;
            }
            let mut stop = None;
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for row in &self.rows {
                        ui.horizontal_wrapped(|ui| {
                            ui.strong(&row.kind);
                            ui.label(&row.name);
                        });
                        ui.horizontal(|ui| {
                            ui.weak(&row.state);
                            if let Some(label) = &row.stop_label {
                                if self.stopping.contains(&row.key) {
                                    ui.weak("Stopping…");
                                } else if ui.small_button(label).clicked() {
                                    stop = Some(row.key.clone());
                                }
                            }
                        });
                        if let Some(fraction) = row.fraction {
                            ui.add(egui::ProgressBar::new(fraction).show_percentage());
                        }
                        ui.separator();
                    }
                });
            if let Some(key) = stop {
                self.request_stop(&key);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_lines_name_one_job() {
        assert_eq!(parse_stop("stop job:7\n"), Some("job:7".into()));
        assert_eq!(parse_stop("stop retry:2"), Some("retry:2".into()));
        assert_eq!(parse_stop("stop "), None);
        assert_eq!(parse_stop("hello"), None);
    }

    #[test]
    fn rows_survive_the_trip_through_json() {
        let rows = vec![JobRow {
            key: "job:1".into(),
            kind: "File".into(),
            name: "talk.mp3".into(),
            state: "Transcribing 2/9".into(),
            fraction: Some(0.2),
            stop_label: Some("Stop".into()),
        }];
        let line = serde_json::to_string(&rows).unwrap();
        assert_eq!(serde_json::from_str::<Vec<JobRow>>(&line).unwrap(), rows);
    }
}
