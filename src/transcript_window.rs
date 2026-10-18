//! Popup window for local-file transcription.
//!
//! The tray process never draws this window itself. Like Settings and About
//! it runs as its own `--transcript-window` subprocess, so a GUI failure can
//! never take down meeting recording. The tray streams [`WindowMsg`]s as
//! JSON lines to the child's stdin; the child only renders them. The window
//! uses native decorations, so minimize / restore / close come from Windows.
//! Closing it never cancels the job: the tray keeps going, saves the file and
//! reveals it as before.

use crate::settings_ui;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::os::windows::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant};
use tracing::warn;

const CREATE_NO_WINDOW: u32 = 0x0800_0000;
/// How much of the finished transcript the popup previews.
pub const PREVIEW_CHARS: usize = 2_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Decoding,
    Queued,
    Diarizing,
    Transcribing,
    CleaningUp,
}

impl Stage {
    fn label(self) -> &'static str {
        match self {
            Stage::Decoding => "Reading audio…",
            Stage::Queued => "Waiting for the engine…",
            Stage::Diarizing => "Identifying speakers",
            Stage::Transcribing => "Transcribing",
            Stage::CleaningUp => "Cleaning up with AI",
        }
    }
}

/// One message from the tray process to the popup, as a JSON line.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WindowMsg {
    Init {
        title: String,
        source_path: String,
        duration: String,
        engine: String,
        cleanup: bool,
        /// Which model does the AI cleanup; empty when unknown.
        #[serde(default)]
        cleanup_engine: String,
        markdown: bool,
        output_path: String,
    },
    Stage {
        stage: Stage,
        done: usize,
        total: usize,
    },
    Done {
        output_path: String,
        words: usize,
        chars: usize,
        preview: String,
        warning: Option<String>,
        /// Number of labelled speakers; `None` when not labelled.
        #[serde(default)]
        speakers: Option<usize>,
    },
    Failed {
        error: String,
    },
    /// AI cleanup of the saved transcript is being retried in the
    /// background: `text` says where it stands. `active` while a retry runs;
    /// `settled` once nothing is left to retry (finished or given up).
    Retry {
        text: String,
        active: bool,
        settled: bool,
    },
}

/// Where a background retry of AI cleanup stands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetryStatus {
    pub text: String,
    pub active: bool,
    pub settled: bool,
}

/// Everything the window shows, as plain data so it is testable without egui.
#[derive(Debug)]
pub struct WindowState {
    pub title: String,
    pub source_path: String,
    pub duration: String,
    pub engine: String,
    pub cleanup: bool,
    pub cleanup_engine: String,
    pub markdown: bool,
    pub output_path: String,
    pub stage: Stage,
    pub done: usize,
    pub total: usize,
    pub finished: Option<Finished>,
    pub error: Option<String>,
    /// A background retry of AI cleanup, after the transcript was saved.
    pub retry: Option<RetryStatus>,
    /// The tray process went away (stdin closed) before the job finished.
    pub parent_gone: bool,
    started: Instant,
    stage_started: Instant,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Finished {
    pub words: usize,
    pub chars: usize,
    pub preview: String,
    pub warning: Option<String>,
    pub speakers: Option<usize>,
}

impl WindowState {
    pub fn new(now: Instant) -> Self {
        Self {
            title: "Transcribing…".into(),
            source_path: String::new(),
            duration: String::new(),
            engine: String::new(),
            cleanup: false,
            cleanup_engine: String::new(),
            markdown: false,
            output_path: String::new(),
            stage: Stage::Decoding,
            done: 0,
            total: 0,
            finished: None,
            error: None,
            retry: None,
            parent_gone: false,
            started: now,
            stage_started: now,
        }
    }

    pub fn apply(&mut self, message: WindowMsg, now: Instant) {
        match message {
            WindowMsg::Init {
                title,
                source_path,
                duration,
                engine,
                cleanup,
                cleanup_engine,
                markdown,
                output_path,
            } => {
                self.title = title;
                self.source_path = source_path;
                self.duration = duration;
                self.engine = engine;
                self.cleanup = cleanup;
                self.cleanup_engine = cleanup_engine;
                self.markdown = markdown;
                self.output_path = output_path;
            }
            WindowMsg::Stage { stage, done, total } => {
                if stage != self.stage {
                    self.stage_started = now;
                }
                self.stage = stage;
                self.done = done;
                self.total = total;
            }
            WindowMsg::Done {
                output_path,
                words,
                chars,
                preview,
                warning,
                speakers,
            } => {
                self.output_path = output_path;
                self.finished = Some(Finished {
                    words,
                    chars,
                    preview,
                    warning,
                    speakers,
                });
                self.error = None;
            }
            WindowMsg::Failed { error } => {
                self.error = Some(error);
            }
            WindowMsg::Retry {
                text,
                active,
                settled,
            } => {
                if active && !self.retry.as_ref().is_some_and(|retry| retry.active) {
                    // A new round: its progress starts from nothing.
                    self.done = 0;
                    self.total = 0;
                    self.stage_started = now;
                }
                self.retry = Some(RetryStatus {
                    text,
                    active,
                    settled,
                });
            }
        }
    }

    /// The transcript is saved but AI cleanup is still being retried.
    pub fn retry_pending(&self) -> bool {
        self.retry.as_ref().is_some_and(|retry| !retry.settled)
    }

    pub fn is_running(&self) -> bool {
        self.finished.is_none() && self.error.is_none()
    }

    /// Progress of the current stage in `0.0..=1.0`, when it is measurable.
    pub fn fraction(&self) -> Option<f32> {
        (self.total > 0).then(|| (self.done as f32 / self.total as f32).clamp(0.0, 1.0))
    }

    /// Seconds left in the current stage, extrapolated from its own pace.
    pub fn eta_secs(&self, now: Instant) -> Option<u64> {
        if self.done == 0 || self.total <= self.done {
            return None;
        }
        let elapsed = now
            .saturating_duration_since(self.stage_started)
            .as_secs_f64();
        let per_item = elapsed / self.done as f64;
        Some((per_item * (self.total - self.done) as f64).round() as u64)
    }

    pub fn elapsed_secs(&self, now: Instant) -> u64 {
        now.saturating_duration_since(self.started).as_secs()
    }
}

pub fn format_clock(total_seconds: u64) -> String {
    let (hours, minutes, seconds) = (
        total_seconds / 3600,
        total_seconds % 3600 / 60,
        total_seconds % 60,
    );
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

/// Word/char counts and a bounded preview for the `Done` message.
pub fn summarize(text: &str) -> (usize, usize, String) {
    let words = word_count(text);
    let chars = text.chars().count();
    let mut preview: String = text.chars().take(PREVIEW_CHARS).collect();
    if chars > PREVIEW_CHARS {
        preview.push('…');
    }
    (words, chars, preview)
}

/// Words in `text`. Chinese and Japanese are written without spaces, so each
/// of their characters counts as a word (the usual convention); other text
/// counts runs between spaces.
fn word_count(text: &str) -> usize {
    let is_ideograph = |c: char| {
        matches!(c,
            '\u{3040}'..='\u{30FF}' // Hiragana, Katakana
            | '\u{3400}'..='\u{4DBF}' // CJK extension A
            | '\u{4E00}'..='\u{9FFF}' // CJK unified ideographs
            | '\u{F900}'..='\u{FAFF}' // CJK compatibility ideographs
        )
    };
    let ideographs = text.chars().filter(|&c| is_ideograph(c)).count();
    let other_words = text
        .split(|c: char| c.is_whitespace() || is_ideograph(c))
        .filter(|word| word.chars().any(char::is_alphanumeric))
        .count();
    ideographs + other_words
}

// ---------------------------------------------------------------------------
// Tray side: spawn the child and stream messages to it.
// ---------------------------------------------------------------------------

/// Launch the popup and forward everything received on `messages` to it.
///
/// Runs on its own thread because `CreateProcess` can block for seconds on
/// ARM64 with real-time scanning. The thread ends when the sender is dropped
/// or the window is closed (write fails); it never kills the child.
pub fn spawn_window_thread(messages: Receiver<WindowMsg>) -> std::io::Result<()> {
    std::thread::Builder::new()
        .name("transcript-window".into())
        .spawn(move || {
            if let Err(error) = pump_messages(messages) {
                warn!(error = %error, "transcript window ended early");
            }
        })
        .map(|_| ())
}

fn pump_messages(messages: Receiver<WindowMsg>) -> Result<()> {
    let exe = std::env::current_exe().context("resolve current executable")?;
    let mut child = Command::new(exe)
        .arg("--transcript-window")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
        .context("start transcript window")?;
    let mut stdin = child.stdin.take().context("transcript window stdin")?;
    for message in messages {
        let mut line = serde_json::to_string(&message).context("encode window message")?;
        line.push('\n');
        // A failed write means the user closed the window; the job goes on.
        if stdin.write_all(line.as_bytes()).is_err() || stdin.flush().is_err() {
            break;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Window process.
// ---------------------------------------------------------------------------

enum Incoming {
    Message(WindowMsg),
    ParentClosed,
}

pub fn run_window() -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(settings_ui::scaled_size([460.0, 360.0]))
            .with_min_inner_size(settings_ui::scaled_size([380.0, 280.0]))
            .with_resizable(true)
            .with_title("HouWritr — Transcribe file"),
        ..Default::default()
    };
    eframe::run_native(
        "HouWritr — Transcribe file",
        options,
        Box::new(|cc| {
            settings_ui::apply_dark_theme(&cc.egui_ctx);
            let (sender, receiver) = mpsc::channel();
            let context = cc.egui_ctx.clone();
            std::thread::Builder::new()
                .name("transcript-window-stdin".into())
                .spawn(move || {
                    let stdin = std::io::stdin();
                    for line in BufReader::new(stdin.lock()).lines() {
                        let Ok(line) = line else { break };
                        if let Ok(message) = serde_json::from_str::<WindowMsg>(&line) {
                            if sender.send(Incoming::Message(message)).is_err() {
                                return;
                            }
                            context.request_repaint();
                        }
                    }
                    let _ = sender.send(Incoming::ParentClosed);
                    context.request_repaint();
                })
                .ok();
            Ok(Box::new(TranscriptWindow {
                state: WindowState::new(Instant::now()),
                incoming: receiver,
                title_applied: false,
                notice: None,
            }))
        }),
    )
    .map_err(|error| anyhow::anyhow!("transcript window failed: {error}"))
}

struct TranscriptWindow {
    state: WindowState,
    incoming: Receiver<Incoming>,
    title_applied: bool,
    notice: Option<String>,
}

impl TranscriptWindow {
    fn drain(&mut self) {
        loop {
            match self.incoming.try_recv() {
                Ok(Incoming::Message(message)) => self.state.apply(message, Instant::now()),
                Ok(Incoming::ParentClosed) => self.state.parent_gone = true,
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
    }

    fn open_output(&mut self) {
        if self.state.output_path.is_empty() {
            return;
        }
        let result = Command::new("explorer.exe")
            .arg(&self.state.output_path)
            .creation_flags(CREATE_NO_WINDOW)
            .spawn();
        if let Err(error) = result {
            self.notice = Some(format!("Could not open the file: {error}"));
        }
    }

    fn reveal_output(&mut self) {
        let path = PathBuf::from(&self.state.output_path);
        if let Err(error) = crate::diagnostics::reveal(&path) {
            self.notice = Some(format!("Could not open the folder: {error}"));
        }
    }
}

impl eframe::App for TranscriptWindow {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain();
        if !self.title_applied && !self.state.source_path.is_empty() {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(format!(
                "HouWritr — {}",
                self.state.title
            )));
            self.title_applied = true;
        }
        let now = Instant::now();
        if (self.state.is_running() || self.state.retry_pending()) && !self.state.parent_gone {
            // Keep the elapsed clock ticking while idle between messages.
            ctx.request_repaint_after(Duration::from_millis(500));
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(&self.state.title);
            ui.add_space(6.0);

            if let Some(error) = &self.state.error {
                settings_ui::inline_error(ui, format!("Transcription failed: {error}"));
            } else if self.state.finished.is_some() {
                match self.state.retry.clone() {
                    Some(retry) if !retry.settled && self.state.parent_gone => {
                        settings_ui::inline_error(
                            ui,
                            "Saved. HouWritr was closed before AI cleanup could be retried.",
                        );
                    }
                    Some(retry) if retry.active => {
                        ui.horizontal_wrapped(|ui| {
                            ui.spinner();
                            ui.label(&retry.text);
                        });
                        if let Some(fraction) = self.state.fraction() {
                            let mut text = format!("{}/{}", self.state.done, self.state.total);
                            if let Some(eta) = self.state.eta_secs(now) {
                                text.push_str(&format!(" · about {} left", format_clock(eta)));
                            }
                            ui.add(egui::ProgressBar::new(fraction).text(text));
                        }
                    }
                    Some(retry) if !retry.settled => {
                        ui.colored_label(egui::Color32::from_rgb(235, 190, 90), &retry.text);
                    }
                    Some(retry) => {
                        ui.colored_label(egui::Color32::from_rgb(110, 200, 140), "Done");
                        ui.weak(&retry.text);
                    }
                    None => {
                        ui.colored_label(egui::Color32::from_rgb(110, 200, 140), "Done");
                    }
                }
            } else if self.state.parent_gone {
                settings_ui::inline_error(ui, "HouWritr was closed before this finished.");
            } else {
                ui.horizontal(|ui| {
                    ui.spinner();
                    let mut line = self.state.stage.label().to_string();
                    if self.state.total > 0 {
                        line.push_str(&format!(" {}/{}", self.state.done, self.state.total));
                    }
                    ui.label(line);
                });
                match self.state.fraction() {
                    Some(fraction) => {
                        ui.add(egui::ProgressBar::new(fraction).show_percentage());
                    }
                    None => {
                        ui.add(egui::ProgressBar::new(0.0).animate(true));
                    }
                }
                let mut timing = format!("Elapsed {}", format_clock(self.state.elapsed_secs(now)));
                if let Some(eta) = self.state.eta_secs(now) {
                    timing.push_str(&format!(" · about {} left", format_clock(eta)));
                }
                ui.weak(timing);
            }

            ui.add_space(8.0);
            ui.separator();
            egui::Grid::new("details")
                .num_columns(2)
                .spacing([14.0, 6.0])
                .show(ui, |ui| {
                    let row = |ui: &mut egui::Ui, name: &str, value: &str| {
                        ui.weak(name);
                        ui.label(if value.is_empty() { "—" } else { value });
                        ui.end_row();
                    };
                    row(ui, "Source", &self.state.source_path);
                    row(ui, "Length", &self.state.duration);
                    row(ui, "Engine", &self.state.engine);
                    let cleanup = if !self.state.cleanup {
                        "Off (raw transcript)".to_string()
                    } else if self.state.cleanup_engine.is_empty() {
                        "On".to_string()
                    } else {
                        format!("On · {}", self.state.cleanup_engine)
                    };
                    row(ui, "AI cleanup", &cleanup);
                    row(
                        ui,
                        "Format",
                        if self.state.markdown {
                            "Markdown (.md)"
                        } else {
                            "Plain text (.txt)"
                        },
                    );
                    row(ui, "Output", &self.state.output_path);
                    if let Some(done) = &self.state.finished {
                        row(
                            ui,
                            "Size",
                            &format!("{} words · {} characters", done.words, done.chars),
                        );
                        if let Some(speakers) = done.speakers {
                            row(ui, "Speakers", &format!("{speakers} identified"));
                        }
                    }
                });

            if let Some(warning) = self
                .state
                .finished
                .as_ref()
                .and_then(|done| done.warning.clone())
            {
                ui.add_space(6.0);
                settings_ui::inline_error(ui, warning);
            }

            if self.state.finished.is_some() {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Open file").clicked() {
                        self.open_output();
                    }
                    if ui.button("Show in folder").clicked() {
                        self.reveal_output();
                    }
                    if ui.button("Close").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                if let Some(notice) = &self.notice {
                    ui.weak(notice);
                }
                ui.add_space(6.0);
                if let Some(done) = &self.state.finished {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new(&done.preview).monospace());
                        });
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(base: Instant, seconds: u64) -> Instant {
        base + Duration::from_secs(seconds)
    }

    #[test]
    fn a_retry_keeps_the_window_from_saying_done() {
        let now = Instant::now();
        let mut state = WindowState::new(now);
        state.apply(
            WindowMsg::Retry {
                text: "AI cleanup unfinished; trying again in 2 min".into(),
                active: false,
                settled: false,
            },
            now,
        );
        state.apply(
            WindowMsg::Done {
                output_path: "a.md".into(),
                words: 1,
                chars: 1,
                preview: String::new(),
                warning: None,
                speakers: None,
            },
            now,
        );
        assert!(state.retry_pending());
        state.apply(
            WindowMsg::Retry {
                text: "Retrying AI cleanup".into(),
                active: true,
                settled: false,
            },
            now,
        );
        assert_eq!((state.done, state.total), (0, 0));
        state.apply(
            WindowMsg::Retry {
                text: "AI cleanup finished".into(),
                active: false,
                settled: true,
            },
            now,
        );
        assert!(!state.retry_pending());
        // The message survives the trip through JSON.
        let line = serde_json::to_string(&WindowMsg::Retry {
            text: "x".into(),
            active: true,
            settled: false,
        })
        .unwrap();
        assert!(matches!(
            serde_json::from_str::<WindowMsg>(&line).unwrap(),
            WindowMsg::Retry { active: true, .. }
        ));
    }

    #[test]
    fn messages_round_trip_as_single_json_lines() {
        let messages = vec![
            WindowMsg::Init {
                title: "talk".into(),
                source_path: "C:\\a\\talk.mp3".into(),
                duration: "12:03".into(),
                engine: "Parakeet CPU".into(),
                cleanup: true,
                cleanup_engine: "Qwen3-4B-Instruct-2507 · this PC (NPU)".into(),
                markdown: false,
                output_path: "C:\\a\\talk.txt".into(),
            },
            WindowMsg::Stage {
                stage: Stage::CleaningUp,
                done: 2,
                total: 9,
            },
            WindowMsg::Done {
                output_path: "C:\\a\\talk.txt".into(),
                words: 10,
                chars: 55,
                preview: "line\nbreak".into(),
                warning: Some("no cleanup".into()),
                speakers: Some(2),
            },
            WindowMsg::Failed {
                error: "boom".into(),
            },
        ];
        for message in messages {
            let line = serde_json::to_string(&message).unwrap();
            assert!(!line.contains('\n'), "embedded newlines must be escaped");
            assert_eq!(serde_json::from_str::<WindowMsg>(&line).unwrap(), message);
        }
    }

    #[test]
    fn stage_changes_reset_progress_pace_and_fraction() {
        let base = Instant::now();
        let mut state = WindowState::new(base);
        assert!(state.is_running());
        assert_eq!(state.fraction(), None);

        state.apply(
            WindowMsg::Stage {
                stage: Stage::Transcribing,
                done: 0,
                total: 4,
            },
            at(base, 1),
        );
        assert_eq!(state.fraction(), Some(0.0));
        assert_eq!(state.eta_secs(at(base, 1)), None);

        state.apply(
            WindowMsg::Stage {
                stage: Stage::Transcribing,
                done: 1,
                total: 4,
            },
            at(base, 11),
        );
        // 10 s for the first window → 3 windows left at 10 s each.
        assert_eq!(state.fraction(), Some(0.25));
        assert_eq!(state.eta_secs(at(base, 11)), Some(30));

        state.apply(
            WindowMsg::Stage {
                stage: Stage::CleaningUp,
                done: 0,
                total: 2,
            },
            at(base, 50),
        );
        assert_eq!(state.eta_secs(at(base, 50)), None);
        assert_eq!(state.elapsed_secs(at(base, 50)), 50);
    }

    #[test]
    fn done_and_failed_end_the_running_state() {
        let base = Instant::now();
        let mut state = WindowState::new(base);
        state.apply(
            WindowMsg::Done {
                output_path: "out.txt".into(),
                words: 3,
                chars: 11,
                preview: "a b c".into(),
                warning: None,
                speakers: None,
            },
            base,
        );
        assert!(!state.is_running());
        assert_eq!(state.output_path, "out.txt");

        let mut failed = WindowState::new(base);
        failed.apply(
            WindowMsg::Failed {
                error: "bad codec".into(),
            },
            base,
        );
        assert!(!failed.is_running());
    }

    #[test]
    fn summaries_count_words_and_bound_the_preview() {
        let (words, chars, preview) = summarize("one two  three");
        assert_eq!((words, chars), (3, 14));
        // Chinese has no spaces: each character is a word.
        assert_eq!(word_count("万物皆可虚拟资料。"), 8);
        assert_eq!(word_count("AI+Skill 是新商机"), 5);
        assert_eq!(preview, "one two  three");

        let long = "x".repeat(PREVIEW_CHARS + 50);
        let (_, chars, preview) = summarize(&long);
        assert_eq!(chars, PREVIEW_CHARS + 50);
        assert_eq!(preview.chars().count(), PREVIEW_CHARS + 1);
        assert!(preview.ends_with('…'));
    }

    #[test]
    fn clock_formats_minutes_and_hours() {
        assert_eq!(format_clock(65), "1:05");
        assert_eq!(format_clock(3_725), "1:02:05");
    }
}
