//! App orchestration: tray + winit event loop + speech engine.
//!
//! HouWritr records meetings and transcribes audio and video files. The tray
//! menu starts both; the work runs on the worker thread and progress shows
//! in the transcript window, the tray and the status pill.

use crate::{
    cleanup::FallbackReason,
    diagnostics, enhance, media,
    meeting::{self, MeetingRecorder},
    model_manager::ModelState,
    overlay::{self, NoticeKind, OverlayViewState, ProcessingPhase},
    power,
    settings::{CredentialHealth, MeetingNotesMode, Settings, SettingsRevision},
    spool::{Spool, SpooledJob},
    transcript_window::{self, WindowMsg},
    tray,
    worker::{
        DiarizationOutcome, JobConfig, JobStage, RepairPlan, Rerun, SessionPart, ShutdownMode,
        Worker, WorkerEvent,
    },
};
use anyhow::Result;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::os::windows::process::CommandExt;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::mpsc;
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{info, warn};
use tray_icon::menu::MenuEvent;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::window::WindowId;

#[derive(Debug, Clone)]
pub enum UserEvent {
    DiagnosticsExported,
    DiagnosticsExportFailed(String),
    /// A file picked from the tray menu was decoded to 16 kHz mono. `window`
    /// feeds the popup that was opened for it when the file was chosen.
    FileDecoded {
        path: PathBuf,
        samples: Vec<f32>,
        output: PathBuf,
        markdown: bool,
        window: mpsc::Sender<WindowMsg>,
        /// Set when the user presses Stop in that popup.
        stop: Arc<AtomicBool>,
    },
    /// The file dialog was dismissed, or decoding failed (`Some(reason)`).
    FileImportEnded(Option<String>),
    /// Answer to "also create meeting notes?" after a recording stopped.
    MeetingNotesChoice(bool),
    Tick,
}

// DETACHED_PROCESS | CREATE_NO_WINDOW — child fully decoupled from parent
const DETACHED_PROCESS: u32 = 0x0000_0008;
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

enum EngineState {
    NotStarted,
    Loading,
    Ready { label: String },
    Failed { error: String },
}

/// Where a long-form (file or meeting) transcript goes, and how to label it.
struct FileJob {
    output: PathBuf,
    markdown: bool,
    /// Structured AI notes were requested; they go to `notes_output`.
    notes: bool,
    notes_output: Option<PathBuf>,
    title: String,
    details: String,
    /// Popup window for this job. `None` (meetings) keeps the overlay pill.
    window: Option<mpsc::Sender<WindowMsg>>,
    /// Set when the user presses Stop in the popup; `stop_sent` once the
    /// worker was told.
    stop: Option<Arc<AtomicBool>>,
    stop_sent: bool,
    /// A non-final part of a rolling meeting transcription: it only feeds the
    /// worker's session and is never shown or delivered on its own.
    background: bool,
    /// Meeting membership, with the settings snapshotted when the meeting
    /// began so every part is processed the same way.
    session: Option<(SessionPart, Settings)>,
    /// The in-progress transcript of a meeting. Background parts append to it
    /// (it is their `output`); the final job deletes it once the finished
    /// transcript is saved.
    partial: Option<PathBuf>,
    /// Set when the job (and its audio) is also kept on disk, because it was
    /// held for power or recovered from an earlier run; its files are deleted
    /// when the job is done.
    spool: Option<u64>,
}

/// What of a [`Repair`] is kept on disk, so that unfinished cleanup retries
/// also survive closing HouWritr or a battery that runs out.
#[derive(serde::Serialize, serde::Deserialize)]
struct RepairRecord {
    id: u64,
    name: String,
    plan: RepairPlan,
    output: PathBuf,
    notes_output: Option<PathBuf>,
    markdown: bool,
    title: String,
    details: String,
    last_written: String,
    best_cleaned: Vec<usize>,
    failed_rounds: u32,
}

/// A job waiting for power saving to end. Its audio is only on disk.
struct HeldJob {
    spool_id: u64,
    job: FileJob,
    kind: String,
}

impl FileJob {
    /// Jobs that must stay out of the status pill's job queue: the
    /// popup shows file jobs, and background meeting parts show nothing.
    fn overlay_free(&self) -> bool {
        self.window.is_some() || self.background
    }

    /// Tell the popup something; a failed send just means it was closed.
    fn tell_window(&self, message: WindowMsg) -> bool {
        self.window
            .as_ref()
            .is_some_and(|window| window.send(message).is_ok())
    }
}

/// What one background retry does for a delivered transcript.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RepairStep {
    /// Clean piece `n` of the transcript again.
    Piece(usize),
    /// Generate the meeting notes that could not be generated before.
    Notes,
}

/// A delivered transcript whose AI cleanup (or notes) could not finish, e.g.
/// because the connection was down. The app keeps what is needed to try again
/// in the background and rewrites the saved files when it succeeds.
struct Repair {
    id: u64,
    name: String,
    plan: RepairPlan,
    output: PathBuf,
    notes_output: Option<PathBuf>,
    markdown: bool,
    title: String,
    details: String,
    /// Exactly what the transcript file held when we last wrote it: if the
    /// file has changed since (the user edited it), it is left alone.
    last_written: String,
    /// Most slices ever cleaned for each piece, to tell progress from repeats.
    best_cleaned: Vec<usize>,
    failed_rounds: u32,
    next_attempt: Instant,
    in_flight: Option<u64>,
    /// The popup of the job that saved the transcript, while it is open.
    window: Option<mpsc::Sender<WindowMsg>>,
}

impl Repair {
    /// What is being retried, for the popup.
    fn what(&self) -> &'static str {
        if self.plan.pieces.iter().any(|piece| piece.raw.is_some()) {
            "AI cleanup"
        } else {
            "AI notes"
        }
    }

    /// Tell the popup where the retry stands; a closed popup is ignored.
    fn tell(&self, text: String, active: bool, settled: bool) {
        if let Some(window) = &self.window {
            let _ = window.send(WindowMsg::Retry {
                text,
                active,
                settled,
            });
        }
    }
}

/// "Transcribing 3/12" and the fraction done, for the Jobs window.
fn progress_text(progress: Option<&(JobStage, usize, usize)>) -> (String, Option<f32>) {
    let Some(&(stage, done, total)) = progress else {
        return ("Starting…".into(), None);
    };
    let label = match stage {
        JobStage::Diarizing => "Identifying speakers",
        JobStage::Transcribing => "Transcribing",
        JobStage::CleaningUp => "Cleaning up with AI",
    };
    if total == 0 {
        return (label.into(), None);
    }
    (
        format!("{label} {done}/{total}"),
        Some((done as f32 / total as f32).clamp(0.0, 1.0)),
    )
}

/// `120` seconds as `2 min`, for the popup.
fn minutes(seconds: u64) -> String {
    format!("{} min", seconds.div_ceil(60).max(1))
}

/// First retry after two minutes, then slower; after five fruitless rounds
/// the text is left as it is.
const REPAIR_BACKOFF_SECONDS: [u64; 5] = [120, 300, 600, 1_200, 1_800];
/// Pause between steps of one productive round.
const REPAIR_STEP_PAUSE: Duration = Duration::from_secs(5);
/// A cheap fingerprint to tell whether saved bytes changed.
fn fingerprint(bytes: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

/// How often the power supply is looked at.
const POWER_CHECK_INTERVAL: Duration = Duration::from_secs(2);
/// How often the main loop looks for a changed settings file.
const SETTINGS_CHECK_INTERVAL: Duration = Duration::from_millis(500);

/// How long to wait before the next round after `failed_rounds` fruitless
/// ones, or `None` to give up.
fn repair_delay(failed_rounds: u32) -> Option<Duration> {
    REPAIR_BACKOFF_SECONDS
        .get(failed_rounds.saturating_sub(1) as usize)
        .map(|seconds| Duration::from_secs(*seconds))
}

/// The next thing to retry: unfinished pieces in order, then the notes.
fn next_repair_step(plan: &RepairPlan) -> Option<RepairStep> {
    plan.pieces
        .iter()
        .position(|piece| piece.raw.is_some())
        .map(RepairStep::Piece)
        .or_else(|| plan.notes_failed.then_some(RepairStep::Notes))
}

/// File content for a transcript: Markdown with a title, or the plain text.
fn render_transcript(markdown: bool, title: &str, details: &str, text: &str) -> String {
    if markdown {
        media::to_markdown(title, details, text)
    } else {
        text.to_string()
    }
}

/// File content for meeting notes, which sit next to the transcript.
fn render_notes(title: &str, details: &str, transcript_name: &str, notes: &str) -> String {
    media::to_markdown(
        &title.replacen("Meeting", "Meeting notes", 1),
        &format!("{details} · transcript: {transcript_name}"),
        notes,
    )
}

/// Overwrite a transcript or notes file the user already has. Atomic, so a
/// crash in the middle never leaves it empty or half written.
fn write_file(path: &std::path::Path, content: &str) -> std::io::Result<()> {
    if let Some(folder) = path
        .parent()
        .filter(|folder| !folder.as_os_str().is_empty())
    {
        std::fs::create_dir_all(folder)?;
    }
    crate::settings::atomic_write_bytes(path, content.as_bytes())
        .map_err(|error| std::io::Error::other(error.to_string()))
}

/// Where a transcript goes when its own folder cannot be written:
/// `Documents\HouWritr\<file name>`.
fn fallback_output(output: &std::path::Path) -> Option<PathBuf> {
    Some(crate::paths::documents_dir()?.join(output.file_name()?))
}

/// "cleaned 21 of 25 sections" when some of the text is still as recognised.
fn cleanup_summary(report: &crate::enhance::CleanupReport) -> Option<String> {
    (report.total > 0 && report.cleaned < report.total)
        .then(|| format!("cleaned {} of {} sections", report.cleaned, report.total))
}

/// What the transcript window says when AI cleanup finished but not every
/// section was cleaned. Not an error: the file is complete and the sections
/// that were not cleaned keep exactly what was recognised.
fn cleanup_notice(report: &crate::enhance::CleanupReport) -> Option<String> {
    if report.total == 0 || report.cleaned + report.partly >= report.total {
        // Everything cleaned, or the rest only in part (said below).
        if report.partly == 0 {
            return None;
        }
    }
    // Neither cleaned, cleaned in part nor unreachable: the cleaned version
    // changed a number or a speaker label even for the smallest piece, or
    // was empty.
    let kept = report
        .total
        .saturating_sub(report.cleaned)
        .saturating_sub(report.partly)
        .saturating_sub(report.unreachable);
    let mut notice = format!(
        "AI cleanup finished: {} of {} sections cleaned.",
        report.cleaned, report.total
    );
    if report.partly > 0 {
        notice.push_str(&format!(
            " {} more in part: a piece of each kept the recognised text because the cleaned version changed a number or a speaker label.",
            report.partly
        ));
    }
    if kept > 0 {
        notice.push_str(&format!(
            " {kept} kept the recognised text because the cleaned version changed a number or a speaker label, even in smaller pieces and when asked again; they are not retried."
        ));
    }
    if report.unreachable > 0 {
        notice.push_str(&format!(
            " {} could not be sent to the model.",
            report.unreachable
        ));
    }
    Some(notice)
}

/// A meeting being recorded: its parts are transcribed as they are cut.
struct MeetingSession {
    id: u64,
    started: meeting::LocalTime,
    /// The transcript so far, kept on disk while recording continues.
    partial: PathBuf,
    /// Settings as they were when the meeting began.
    settings: Settings,
}

/// A finished meeting recording waiting for the "create notes?" answer. The
/// final part is the audio recorded since the last rolling cut.
struct PendingMeeting {
    tail: meeting::Segment,
    session: MeetingSession,
    recorded: Duration,
}

/// What to do about meeting notes when a recording stops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NotesDecision {
    /// Show the "create notes?" question.
    Ask,
    /// Create notes without asking.
    Create,
    /// Transcript only.
    Skip,
    /// The user wants notes (or would be asked) but no AI provider is set
    /// up: transcript only, with a hint.
    SkipNoProvider,
}

/// `may_ask` is false when no question can be shown (quitting the app).
fn notes_decision(mode: MeetingNotesMode, provider_ready: bool, may_ask: bool) -> NotesDecision {
    match mode {
        MeetingNotesMode::Never => NotesDecision::Skip,
        MeetingNotesMode::Always if provider_ready => NotesDecision::Create,
        MeetingNotesMode::Always => NotesDecision::SkipNoProvider,
        MeetingNotesMode::Ask if !may_ask => NotesDecision::Skip,
        MeetingNotesMode::Ask if provider_ready => NotesDecision::Ask,
        MeetingNotesMode::Ask => NotesDecision::SkipNoProvider,
    }
}

/// Rolling parts are queued as pieces of at most this long, so a file queued
/// during a meeting never waits behind a long one.
const MEETING_PIECE_SECONDS: usize = 300;
const MEETING_PIECE_SEARCH_SECONDS: usize = 10;

#[derive(Clone)]
struct OverlayNotice {
    kind: NoticeKind,
    message: String,
    expires_at: Instant,
}

struct OverlayReducer {
    jobs: BTreeMap<u64, ProcessingPhase>,
    active_job: Option<u64>,
    notice: Option<OverlayNotice>,
    last_view: OverlayViewState,
    /// Whether the queued / writing / polishing pill may be shown. It follows
    /// the recording indicator: hidden there, hidden here. Notices (errors,
    /// warnings) are not progress and still show.
    progress_visible: bool,
}

impl Default for OverlayReducer {
    fn default() -> Self {
        Self {
            jobs: BTreeMap::new(),
            active_job: None,
            notice: None,
            last_view: OverlayViewState::Hidden,
            progress_visible: true,
        }
    }
}

impl OverlayReducer {
    fn job_queued(&mut self, id: u64) {
        self.jobs.entry(id).or_insert(ProcessingPhase::Queued);
    }

    fn job_started(&mut self, id: u64) -> bool {
        if self.active_job.is_some_and(|active| active != id) {
            return false;
        }
        let Some(phase) = self.jobs.get_mut(&id) else {
            return false;
        };
        if *phase != ProcessingPhase::Queued {
            return false;
        }
        *phase = ProcessingPhase::Transcribing;
        self.active_job = Some(id);
        true
    }

    fn enhancement_started(&mut self, id: u64) -> bool {
        if self.active_job != Some(id) {
            return false;
        }
        let Some(phase) = self.jobs.get_mut(&id) else {
            return false;
        };
        if *phase != ProcessingPhase::Transcribing {
            return false;
        }
        *phase = ProcessingPhase::Enhancing;
        true
    }

    fn job_finished(&mut self, id: u64) -> bool {
        if self.jobs.remove(&id).is_none() {
            return false;
        }
        if self.active_job == Some(id) {
            self.active_job = None;
        }
        true
    }

    fn show_notice(&mut self, kind: NoticeKind, message: impl Into<String>, now: Instant) {
        let duration = match kind {
            NoticeKind::Success => Duration::from_millis(2500),
            NoticeKind::Info => Duration::from_secs(3),
            NoticeKind::Warning => Duration::from_secs(4),
            NoticeKind::Error => Duration::from_secs(5),
        };
        self.notice = Some(OverlayNotice {
            kind,
            message: message.into(),
            expires_at: now + duration,
        });
    }

    fn desired_view(&mut self, now: Instant) -> OverlayViewState {
        if self
            .notice
            .as_ref()
            .is_some_and(|notice| now >= notice.expires_at)
        {
            self.notice = None;
        }
        if let Some(id) = self.active_job.filter(|_| self.progress_visible) {
            if let Some(phase) = self.jobs.get(&id) {
                let queued = self
                    .jobs
                    .values()
                    .filter(|phase| **phase == ProcessingPhase::Queued)
                    .count();
                return OverlayViewState::processing(id, *phase, queued);
            }
        }
        let queued = self
            .jobs
            .iter()
            .filter(|(_, phase)| **phase == ProcessingPhase::Queued)
            .map(|(id, _)| *id)
            .collect::<Vec<_>>();
        if let Some(id) = queued.first().filter(|_| self.progress_visible) {
            return OverlayViewState::processing(
                *id,
                ProcessingPhase::Queued,
                queued.len().saturating_sub(1),
            );
        }
        if let Some(notice) = &self.notice {
            return OverlayViewState::notice(notice.kind, notice.message.clone());
        }
        OverlayViewState::Hidden
    }

    fn sync(&mut self, controller: &overlay::OverlayController, now: Instant) {
        let view = self.desired_view(now);
        if view != self.last_view {
            if let Err(error) = controller.set_state(view.clone()) {
                warn!(error, "failed to update overlay state");
            } else {
                self.last_view = view;
            }
        }
    }
}

struct State {
    settings: Settings,
    credential_health: CredentialHealth,
    settings_error: Option<String>,
    tray: tray::Tray,
    overlay: overlay::OverlayController,
    overlay_state: OverlayReducer,
    worker: Worker,
    engine_state: EngineState,
    model_state: Option<ModelState>,
    load_generation: u64,
    pending_jobs: usize,
    active_job: Option<u64>,
    shutting_down: bool,
    discarding_jobs: bool,
    hk_stop: Arc<AtomicBool>,
    settings_revision: Option<SettingsRevision>,
    /// When the settings file was last checked for changes (see
    /// [`SETTINGS_CHECK_INTERVAL`]).
    settings_checked_at: Instant,
    proxy: EventLoopProxy<UserEvent>,
    diagnostics_exporting: bool,
    meeting: Option<MeetingRecorder>,
    /// Live meeting input level feeding the recording indicator.
    meeting_level: Arc<AtomicU32>,
    /// The meeting pill: same position as the status pill, shown for the
    /// whole recording except while the status pill is on screen.
    meeting_overlay: overlay::OverlayController,
    meeting_overlay_view: OverlayViewState,
    meeting_overlay_enabled: bool,
    /// Tray toggle for this session only. `None` follows the saved setting;
    /// the tray process never rewrites settings.json itself.
    meeting_indicator_override: Option<bool>,
    /// Recording finished; the notes question is open.
    pending_meeting: Option<PendingMeeting>,
    /// The meeting being recorded, if any.
    meeting_session: Option<MeetingSession>,
    next_meeting_session: u64,
    /// Delivered transcripts waiting for an automatic retry of AI cleanup.
    repairs: Vec<Repair>,
    repair_jobs: HashMap<u64, (u64, RepairStep)>,
    next_repair_id: u64,
    /// Long-form jobs (imported files, meetings) mapped to their output file.
    file_jobs: HashMap<u64, FileJob>,
    file_import_running: bool,
    /// The Jobs window, while it is open.
    jobs_window: Option<crate::jobs_window::JobsWindow>,
    /// Jobs the worker has started, and the latest progress of each.
    running_jobs: HashSet<u64>,
    job_progress: HashMap<u64, (JobStage, usize, usize)>,
    /// Jobs kept on disk until power saving ends.
    spool: Spool,
    held: Vec<HeldJob>,
    /// What Windows said about the power supply at the last check.
    power: power::PowerStatus,
    power_checked_at: Instant,
    /// What the tray's power saving item shows: (on, follows battery saver).
    power_mode_shown: Option<(bool, bool)>,
    /// Fingerprint of the retries last written to disk.
    repairs_saved: u64,
}

pub fn run() -> Result<()> {
    let (settings, credential_health, settings_error, settings_revision) =
        match Settings::load_runtime() {
            Ok(loaded) => (
                loaded.settings,
                loaded.credential_health,
                loaded.settings_error,
                Some(loaded.revision),
            ),
            Err(error) => {
                warn!(error = %error, "settings load failed; using defaults until a valid file is saved");
                (
                    Settings::default(),
                    CredentialHealth::default(),
                    Some(format!("settings could not be loaded: {error}")),
                    Settings::revision().ok(),
                )
            }
        };
    if let Some(message) = &credential_health.message {
        warn!(message, "credential migration needs attention");
    }
    if let Some(error) = &settings_error {
        warn!(
            error,
            "settings validation failed; using defaults until a valid file is saved"
        );
    }
    info!(
        engine = %settings.engine,
        overlay = settings.overlay,
        "settings loaded"
    );
    let tray = tray::Tray::new(&settings)?;
    tray.set_meeting_indicator_visible(settings.show_meeting_indicator);
    let worker = Worker::spawn()?;

    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    let proxy = event_loop.create_proxy();

    // Tells the status pills' threads to stop when HouWritr quits.
    let hk_stop = Arc::new(AtomicBool::new(false));
    spawn_tick_thread(proxy.clone());

    // Status pill (job progress and notices) on its own thread and Win32
    // message loop, so it can never stall the tray loop.
    let overlay = overlay::spawn(
        overlay::OverlayHandles {
            level_x10000: Arc::new(AtomicU32::new(0)),
            stop: hk_stop.clone(),
        },
        settings.overlay,
    )?;

    let meeting_level = Arc::new(AtomicU32::new(0));
    let meeting_overlay = overlay::spawn(
        overlay::OverlayHandles {
            level_x10000: meeting_level.clone(),
            stop: hk_stop.clone(),
        },
        settings.show_meeting_indicator,
    )?;
    let meeting_overlay_enabled = settings.show_meeting_indicator;

    let state = State {
        settings,
        credential_health,
        settings_error,
        tray,
        overlay,
        overlay_state: OverlayReducer::default(),
        worker,
        engine_state: EngineState::NotStarted,
        model_state: None,
        load_generation: 0,
        pending_jobs: 0,
        active_job: None,
        shutting_down: false,
        discarding_jobs: false,
        hk_stop,
        settings_revision,
        settings_checked_at: Instant::now(),
        proxy,
        diagnostics_exporting: false,
        meeting: None,
        meeting_level,
        meeting_overlay,
        meeting_overlay_view: OverlayViewState::Hidden,
        meeting_overlay_enabled,
        meeting_indicator_override: None,
        pending_meeting: None,
        meeting_session: None,
        next_meeting_session: 1,
        repairs: Vec::new(),
        repair_jobs: HashMap::new(),
        next_repair_id: 1,
        file_jobs: HashMap::new(),
        file_import_running: false,
        jobs_window: None,
        running_jobs: HashSet::new(),
        job_progress: HashMap::new(),
        spool: Spool::open(Spool::default_dir()),
        held: Vec::new(),
        power: power::status(),
        power_checked_at: Instant::now(),
        power_mode_shown: None,
        repairs_saved: fingerprint(&[]),
    };

    // Wait mode: loop sleeps until a message arrives (tray click, user event,
    // window event). The tick thread wakes it via EventLoopProxy.
    // No more thread::sleep inside the message pump → tray stays responsive.
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = AppHandler { state };
    app.recover_spooled_jobs();
    app.recover_repairs();
    app.start_engine_load();
    event_loop.run_app(&mut app)?;
    Ok(())
}

fn spawn_tick_thread(proxy: EventLoopProxy<UserEvent>) {
    // Tick the event loop every 100 ms so periodic tasks (tray menu poll,
    // staged engine, settings reload, transcribe-done flag) get serviced
    // even when no input is happening.
    thread::Builder::new()
        .name("tick".into())
        .spawn(move || loop {
            thread::sleep(Duration::from_millis(100));
            if proxy.send_event(UserEvent::Tick).is_err() {
                break;
            }
        })
        .ok();
}

/// Why AI cleanup gave no output, in words for the transcript window and
/// the notice. The log has the details.
fn cleanup_failure_text(reason: &FallbackReason) -> String {
    match reason {
        FallbackReason::UnknownProvider => "the cleanup provider setting is not valid".into(),
        FallbackReason::MissingCredential => "no sign-in or API key was found".into(),
        FallbackReason::CredentialTargetChanged => {
            "the provider changed after recording started".into()
        }
        FallbackReason::InvalidEndpoint => {
            "the server address is not accepted (an API key needs https)".into()
        }
        FallbackReason::EmptyModelId => "no model is set".into(),
        FallbackReason::RequestFailed => {
            "the provider could not be reached or did not answer in time".into()
        }
        FallbackReason::ProviderRejected(status) => {
            format!("the provider rejected the request, HTTP {status}, see the log")
        }
        FallbackReason::ResponseUnparseable => "the provider's answer could not be read".into(),
        FallbackReason::EmptyCandidate => "the provider returned no text".into(),
        FallbackReason::IntegrityRejected(_) => {
            "the cleaned text changed facts or numbers, so it was not used".into()
        }
        FallbackReason::ValidatorError => "the cleaned text could not be checked".into(),
    }
}

struct AppHandler {
    state: State,
}

impl AppHandler {
    fn sync_overlay(&mut self) {
        self.state.overlay_state.progress_visible = self.meeting_indicator_visible();
        self.state
            .overlay_state
            .sync(&self.state.overlay, Instant::now());
    }

    fn show_overlay_notice(&mut self, kind: NoticeKind, message: impl Into<String>) {
        self.state
            .overlay_state
            .show_notice(kind, message, Instant::now());
        self.sync_overlay();
    }

    fn start_engine_load(&mut self) {
        if self.state.shutting_down {
            return;
        }
        let engine_name = self.state.settings.engine.clone();
        match self.state.worker.load(engine_name.clone()) {
            Ok(generation) => {
                self.state.load_generation = generation;
                self.state.engine_state = EngineState::Loading;
                self.state.model_state = None;
                self.state.tray.set_status(
                    tray::IconColor::Transcribing,
                    &format!("HouWritr — loading {engine_name}"),
                );
            }
            Err(error) => {
                self.state.engine_state = EngineState::Failed {
                    error: error.to_string(),
                };
                self.state.tray.set_status(
                    tray::IconColor::Error,
                    "HouWritr — could not start engine loader (see log)",
                );
                warn!(error = %error, "failed to start engine load");
            }
        }
    }
}

impl ApplicationHandler<UserEvent> for AppHandler {
    fn resumed(&mut self, _el: &ActiveEventLoop) {
        info!("event loop ready");
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, _id: WindowId, _ev: WindowEvent) {}

    fn user_event(&mut self, el: &ActiveEventLoop, ev: UserEvent) {
        match ev {
            UserEvent::DiagnosticsExported => {
                self.state.diagnostics_exporting = false;
                self.state
                    .tray
                    .set_tooltip("HouWritr — diagnostics exported");
            }
            UserEvent::DiagnosticsExportFailed(error) => {
                self.state.diagnostics_exporting = false;
                warn!(error = %error, "diagnostics export failed");
                self.state
                    .tray
                    .set_tooltip("HouWritr — diagnostics export failed (see log)");
            }
            UserEvent::FileDecoded {
                path,
                samples,
                output,
                markdown,
                window,
                stop,
            } => {
                self.state.file_import_running = false;
                if stop.load(Ordering::Acquire) {
                    // Stopped while the file was being read.
                    info!(file = %path.display(), "transcription stopped before it started");
                    let _ = window.send(WindowMsg::Stopped);
                } else {
                    self.dispatch_file(path, samples, output, markdown, window, stop);
                }
            }
            UserEvent::MeetingNotesChoice(create_notes) => {
                self.finish_pending_meeting(create_notes);
            }
            UserEvent::FileImportEnded(error) => {
                self.state.file_import_running = false;
                if let Some(error) = error {
                    // The popup already shows the reason; no pill for files.
                    warn!(error = %error, "file import failed");
                    self.state.tray.set_status(
                        tray::IconColor::Error,
                        &format!("HouWritr — could not read file: {}", short_error(&error)),
                    );
                } else {
                    self.update_job_status();
                }
            }
            UserEvent::Tick => self.tick(el),
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        // Wait mode: don't sleep here; tick thread + user events wake us.
        self.tick(el);
        el.set_control_flow(ControlFlow::Wait);
    }
}

impl AppHandler {
    fn tick(&mut self, el: &ActiveEventLoop) {
        // Stop pressed in a transcript window: tell the worker once.
        for (id, job) in self.state.file_jobs.iter_mut() {
            if !job.stop_sent
                && job
                    .stop
                    .as_ref()
                    .is_some_and(|stop| stop.load(Ordering::Acquire))
            {
                job.stop_sent = true;
                info!(id, "transcription stop requested");
                self.state.worker.stop_job(*id);
            }
        }
        while let Some(event) = self.state.worker.try_recv() {
            // What the Jobs window shows.
            match &event {
                WorkerEvent::JobStarted { id } => {
                    self.state.running_jobs.insert(*id);
                }
                WorkerEvent::JobProgress {
                    id,
                    stage,
                    done,
                    total,
                } => {
                    self.state.running_jobs.insert(*id);
                    self.state.job_progress.insert(*id, (*stage, *done, *total));
                }
                WorkerEvent::JobCompleted { id, .. }
                | WorkerEvent::JobFailed { id, .. }
                | WorkerEvent::JobDiscarded { id } => {
                    self.state.running_jobs.remove(id);
                    self.state.job_progress.remove(id);
                }
                _ => {}
            }
            let finished = match &event {
                WorkerEvent::JobCompleted { id, .. }
                | WorkerEvent::JobFailed { id, .. }
                | WorkerEvent::JobDiscarded { id } => Some(*id),
                _ => None,
            };
            self.handle_worker_event(el, event);
            // The job's own log ends once its result is saved.
            if let Some(id) = finished {
                crate::log_files::end(id);
            }
        }
        self.sync_overlay();
        self.sync_meeting_indicator();
        self.update_jobs_window();
        self.poll_power();

        let meeting_status = self
            .state
            .meeting
            .as_ref()
            .map(|meeting| (meeting.limit_reached(), meeting.failure()));
        if let Some((limit_reached, failure)) = meeting_status {
            if limit_reached || failure.is_some() {
                if let Some(failure) = failure {
                    warn!(
                        failure,
                        "meeting capture failed; transcribing what was captured"
                    );
                }
                self.stop_meeting(true);
            }
        }

        self.run_due_repairs();

        // Rolling transcription: five minutes of finished audio go to the
        // transcriber while the recording carries on.
        if self
            .state
            .meeting
            .as_ref()
            .is_some_and(|recorder| !recorder.is_paused() && recorder.roll_due())
        {
            self.roll_meeting_part();
        }

        // Drain tray menu events.
        while let Ok(ev) = MenuEvent::receiver().try_recv() {
            if ev.id == self.state.tray.menu_quit_id {
                self.request_shutdown(el);
                return;
            }
            if ev.id == self.state.tray.menu_cancel_model_id {
                if matches!(self.state.engine_state, EngineState::Loading) {
                    self.state.load_generation = self.state.worker.cancel_load();
                    self.state.engine_state = EngineState::NotStarted;
                    self.state.model_state = Some(ModelState::Cancelled);
                    self.state.tray.set_status(
                        tray::IconColor::Error,
                        "HouWritr — model download cancelled; choose Retry",
                    );
                    info!("model acquisition cancelled by user");
                }
                continue;
            }
            if ev.id == self.state.tray.menu_retry_engine_id {
                self.start_engine_load();
                continue;
            }
            if ev.id == self.state.tray.menu_open_logs_id {
                if let Err(e) = diagnostics::open_logs_dir() {
                    warn!(error = %e, "failed to open logs directory");
                    self.state.tray.set_status(
                        tray::IconColor::Error,
                        "HouWritr — could not open logs (see log)",
                    );
                }
                continue;
            }
            if ev.id == self.state.tray.menu_export_diagnostics_id {
                if !self.state.diagnostics_exporting {
                    self.state.diagnostics_exporting = true;
                    self.state
                        .tray
                        .set_tooltip("HouWritr — exporting diagnostics…");
                    let settings = self.state.settings.clone();
                    let proxy = self.state.proxy.clone();
                    let spawn = thread::Builder::new()
                        .name("diagnostics-export".into())
                        .spawn(move || {
                            let event = match diagnostics::export_bundle(&settings) {
                                Ok(path) => {
                                    info!(file = %path.display(), "diagnostics exported");
                                    if let Err(e) = diagnostics::reveal(&path) {
                                        warn!(error = %e, "failed to reveal diagnostics bundle");
                                    }
                                    UserEvent::DiagnosticsExported
                                }
                                Err(e) => UserEvent::DiagnosticsExportFailed(e.to_string()),
                            };
                            let _ = proxy.send_event(event);
                        });
                    if let Err(e) = spawn {
                        self.state.diagnostics_exporting = false;
                        warn!(error = %e, "failed to start diagnostics export");
                        self.state
                            .tray
                            .set_tooltip("HouWritr — diagnostics export failed (see log)");
                    }
                } else {
                    self.state
                        .tray
                        .set_tooltip("HouWritr — diagnostics export already running");
                }
                continue;
            }
            if ev.id == self.state.tray.menu_transcribe_file_id {
                self.start_file_import();
                continue;
            }
            if ev.id == self.state.tray.menu_jobs_id {
                if self
                    .state
                    .jobs_window
                    .as_ref()
                    .is_some_and(|window| window.is_open())
                {
                    info!("jobs window already open");
                } else {
                    match crate::jobs_window::JobsWindow::open() {
                        Ok(window) => self.state.jobs_window = Some(window),
                        Err(error) => warn!(error = %error, "could not open the jobs window"),
                    }
                    self.update_jobs_window();
                }
                continue;
            }
            if ev.id == self.state.tray.menu_meeting_id {
                if self.state.meeting.is_some() {
                    self.stop_meeting(true);
                } else {
                    self.start_meeting();
                }
                continue;
            }
            if ev.id == self.state.tray.menu_meeting_pause_id {
                self.toggle_meeting_pause();
                continue;
            }
            if ev.id == self.state.tray.menu_meeting_indicator_id {
                self.toggle_meeting_indicator();
                continue;
            }
            if ev.id == self.state.tray.menu_power_saving_id {
                self.toggle_power_saving();
                continue;
            }
            if ev.id == self.state.tray.menu_settings_id {
                info!("opening settings UI subprocess");
                // CRITICAL: spawn from a background thread. CreateProcessW on
                // Windows ARM64 (especially with Defender real-time scanning)
                // can block for several seconds. If we call it from inside the
                // winit pump, the tray's message queue stalls → app goes
                // "Not Responding" → meeting capture stalls. From a worker thread the
                // main pump keeps draining messages while the child boots.
                if let Ok(exe) = std::env::current_exe() {
                    thread::spawn(move || {
                        let _ = std::process::Command::new(exe)
                            .arg("--settings")
                            .stdin(Stdio::null())
                            .stdout(Stdio::null())
                            .stderr(Stdio::null())
                            .creation_flags(DETACHED_PROCESS | CREATE_NO_WINDOW)
                            .spawn();
                    });
                }
                continue;
            }
            if ev.id == self.state.tray.menu_about_id {
                info!("opening About UI subprocess");
                if let Ok(exe) = std::env::current_exe() {
                    thread::spawn(move || {
                        let _ = std::process::Command::new(exe)
                            .arg("--about")
                            .stdin(Stdio::null())
                            .stdout(Stdio::null())
                            .stderr(Stdio::null())
                            .creation_flags(DETACHED_PROCESS | CREATE_NO_WINDOW)
                            .spawn();
                    });
                }
            }
        }

        // Settings hot-reload compares content revisions so deletion is a
        // first-class change and both loops return to the same defaults.
        // Reading and hashing the file on every pass of the loop (at least 10
        // times a second, and on every event) cost disk reads and virus scans
        // all day; twice a second is as prompt as anyone notices.
        if self.state.settings_checked_at.elapsed() >= SETTINGS_CHECK_INTERVAL {
            self.state.settings_checked_at = Instant::now();
            match Settings::revision() {
                Ok(revision) if self.state.settings_revision.as_ref() != Some(&revision) => {
                    match Settings::load_runtime() {
                        Ok(loaded) => {
                            self.state.settings_revision = Some(loaded.revision.clone());
                            self.state.credential_health = loaded.credential_health;
                            if let Some(message) = &self.state.credential_health.message {
                                warn!(message, "credential migration needs attention");
                            }
                            if let Some(error) = loaded.settings_error {
                                warn!(error, "settings reload failed validation; keeping the last valid settings");
                                self.state.settings_error = Some(error);
                                self.update_job_status();
                            } else {
                                let old_engine = self.state.settings.engine.clone();
                                let old_overlay = self.state.settings.overlay;
                                let old_indicator = self.state.settings.show_meeting_indicator;
                                self.state.settings = loaded.settings;
                                self.state.settings_error = None;
                                if self.state.settings.show_meeting_indicator != old_indicator {
                                    // A saved change replaces any tray toggle.
                                    self.state.meeting_indicator_override = None;
                                    self.sync_overlay();
                                    self.sync_meeting_indicator();
                                }
                                if self.state.settings.overlay != old_overlay {
                                    if let Err(error) =
                                        self.state.overlay.set_enabled(self.state.settings.overlay)
                                    {
                                        warn!(error, "failed to update overlay setting");
                                    }
                                    self.sync_overlay();
                                }
                                let new_engine = self.state.settings.engine.clone();
                                if new_engine != old_engine {
                                    info!(
                                        "engine changed: {old_engine} -> {new_engine}; reloading"
                                    );
                                    self.start_engine_load();
                                } else {
                                    self.update_job_status();
                                }
                            }
                        }
                        Err(error) => {
                            warn!(error = %error, "settings reload failed; keeping the last valid settings");
                            self.state.settings_error =
                                Some(format!("settings could not be reloaded: {error}"));
                            self.update_job_status();
                        }
                    }
                }
                Ok(_) => {}
                Err(error) => {
                    let message = format!("settings revision could not be read: {error}");
                    if self.state.settings_error.as_deref() != Some(message.as_str()) {
                        warn!(error = %error, "settings revision check failed; keeping the last valid settings");
                        self.state.settings_error = Some(message);
                        self.update_job_status();
                    }
                }
            }
        }
    }
}

impl AppHandler {
    /// Gate shared by file import and meetings: they need a loaded engine.
    fn require_ready_engine(&mut self) -> bool {
        if self.state.shutting_down {
            return false;
        }
        if matches!(self.state.engine_state, EngineState::Ready { .. }) {
            return true;
        }
        let (color, status) = self.blocked_recording_status();
        self.state.tray.set_status(color, &status);
        self.show_overlay_notice(NoticeKind::Warning, "Engine is not ready yet");
        false
    }

    fn start_file_import(&mut self) {
        if self.state.file_import_running {
            return;
        }
        if !self.require_ready_engine() {
            return;
        }
        self.state.file_import_running = true;
        let proxy = self.state.proxy.clone();
        let markdown = self.state.settings.markdown_transcripts;
        let cleanup = self.state.settings.enhance.mode.is_enabled();
        let cleanup_engine = crate::enhance::cleanup_engine_label(&self.state.settings);
        let engine = match &self.state.engine_state {
            EngineState::Ready { label } => label.clone(),
            _ => self.state.settings.engine.clone(),
        };
        // The Open dialog blocks and decoding is CPU-heavy: neither may run
        // on the tray event loop.
        let spawned = thread::Builder::new()
            .name("file-import".into())
            .spawn(move || {
                let Some(path) = media::pick_audio_file() else {
                    let _ = proxy.send_event(UserEvent::FileImportEnded(None));
                    return;
                };
                let event = import_file(path, markdown, cleanup, cleanup_engine, engine);
                let _ = proxy.send_event(event);
            });
        if let Err(error) = spawned {
            self.state.file_import_running = false;
            warn!(error = %error, "failed to start file import");
        }
    }

    fn dispatch_file(
        &mut self,
        source: PathBuf,
        samples: Vec<f32>,
        output: PathBuf,
        markdown: bool,
        window: mpsc::Sender<WindowMsg>,
        stop: Arc<AtomicBool>,
    ) {
        let name = source
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "audio file".into());
        let title = source
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Transcript".into());
        let seconds = samples.len() / media::TARGET_RATE as usize;
        let job = FileJob {
            output,
            markdown,
            notes: false,
            notes_output: None,
            title,
            details: format!(
                "Transcript of {name} ({}) · HouWritr",
                format_duration(seconds)
            ),
            window: Some(window),
            stop: Some(stop),
            stop_sent: false,
            background: false,
            session: None,
            partial: None,
            spool: None,
        };
        self.dispatch_long_form(samples, job, "file");
    }

    fn dispatch_long_form(&mut self, samples: Vec<f32>, job: FileJob, kind: &str) {
        if self.state.shutting_down {
            job.tell_window(WindowMsg::Failed {
                error: "HouWritr is shutting down".into(),
            });
            return;
        }
        self.state.power = power::status();
        // Jobs go to the worker in the order they came: when power saving has
        // just ended, the waiting ones start before this one.
        if !self.hold_for_power() && self.engine_ready() {
            self.release_held_jobs();
        }
        let job = if self.hold_for_power() || !self.state.held.is_empty() {
            match self.hold_job(&samples, job, kind) {
                Ok(()) => return,
                Err(job) => job,
            }
        } else {
            job
        };
        self.enqueue_long_form(samples, job, kind);
    }

    fn engine_ready(&self) -> bool {
        matches!(self.state.engine_state, EngineState::Ready { .. })
    }

    /// Power saving is on: the saved toggle, or Windows' battery saver when
    /// the setting says to follow it.
    fn power_saving_on(&self) -> bool {
        power::saving_on(
            self.state.settings.power_saving,
            self.state.settings.power_saving_auto,
            self.state.power,
        )
    }

    /// Work waits: power saving is on.
    fn hold_for_power(&self) -> bool {
        self.power_saving_on()
    }

    /// Keep the job and its audio on disk until power saving ends. The job
    /// comes back when it could not be saved: it is then done at once rather
    /// than lost.
    fn hold_job(
        &mut self,
        samples: &[f32],
        mut job: FileJob,
        kind: &str,
    ) -> std::result::Result<(), FileJob> {
        let spooled = SpooledJob {
            id: 0,
            kind: kind.into(),
            output: job.output.clone(),
            markdown: job.markdown,
            notes: job.notes,
            notes_output: job.notes_output.clone(),
            title: job.title.clone(),
            details: job.details.clone(),
            background: job.background,
            session: job.session.clone(),
            partial: job.partial.clone(),
        };
        match self.state.spool.save(&spooled, samples) {
            Ok(id) => {
                let first = self.state.held.is_empty();
                job.spool = Some(id);
                job.tell_window(WindowMsg::Stage {
                    stage: transcript_window::Stage::Queued,
                    done: 0,
                    total: 0,
                });
                info!(id, kind, "job kept on disk until power saving ends");
                self.state.held.push(HeldJob {
                    spool_id: id,
                    job,
                    kind: kind.to_string(),
                });
                if first {
                    let message = if self.state.settings.power_saving_auto {
                        "Windows power saving is on: transcription waits until it is off"
                    } else {
                        "Power saving is on: transcription waits until you turn it off"
                    };
                    self.show_overlay_notice(NoticeKind::Info, message);
                }
                self.update_job_status();
                Ok(())
            }
            Err(error) => {
                warn!(error = %error, kind, "could not keep the job on disk; doing it now");
                Err(job)
            }
        }
    }

    /// Give the held jobs to the worker, oldest first. Their files stay on
    /// disk until each job is done.
    fn release_held_jobs(&mut self) {
        let held = std::mem::take(&mut self.state.held);
        if held.is_empty() {
            return;
        }
        info!(
            count = held.len(),
            "power saving ended: starting the waiting jobs"
        );
        for HeldJob {
            spool_id,
            job,
            kind,
        } in held
        {
            match self.state.spool.load_samples(spool_id) {
                Ok(samples) => self.enqueue_long_form(samples, job, &kind),
                Err(error) => {
                    warn!(error = %error, id = spool_id, "the audio of a waiting job could not be read");
                    job.tell_window(WindowMsg::Failed {
                        error: "the saved audio could not be read".into(),
                    });
                }
            }
        }
        self.update_job_status();
    }

    /// Delete the files of a finished job.
    fn forget_spooled(&self, job: &Option<FileJob>) {
        if let Some(id) = job.as_ref().and_then(|job| job.spool) {
            self.state.spool.remove(id);
        }
    }

    /// Pick up the jobs an earlier run left on disk (the battery ran out,
    /// HouWritr was closed, or the PC restarted before they were done).
    fn recover_spooled_jobs(&mut self) {
        self.state.spool.sweep();
        let (jobs, unreadable) = self.state.spool.list();
        for path in unreadable {
            warn!(file = %path.display(), "a waiting job on disk could not be read; leaving it");
        }
        for spooled in jobs {
            if let Some((part, _)) = &spooled.session {
                self.state.next_meeting_session = self.state.next_meeting_session.max(part.id + 1);
            }
            info!(id = spooled.id, kind = %spooled.kind, "picked up a job left on disk");
            self.state.held.push(HeldJob {
                spool_id: spooled.id,
                kind: spooled.kind,
                job: FileJob {
                    output: spooled.output,
                    markdown: spooled.markdown,
                    notes: spooled.notes,
                    notes_output: spooled.notes_output,
                    title: spooled.title,
                    details: spooled.details,
                    window: None,
                    stop: None,
                    stop_sent: false,
                    background: spooled.background,
                    session: spooled.session,
                    partial: spooled.partial,
                    spool: Some(spooled.id),
                },
            });
        }
    }

    /// Pick up the cleanup retries an earlier run had not finished.
    fn recover_repairs(&mut self) {
        let Some(bytes) = self.state.spool.load_repairs() else {
            return;
        };
        let records: Vec<RepairRecord> = match serde_json::from_slice(&bytes) {
            Ok(records) => records,
            Err(error) => {
                warn!(error = %error, "the saved cleanup retries could not be read; leaving them");
                return;
            }
        };
        for record in records {
            info!(file = %record.output.display(), "picked up an unfinished cleanup retry");
            self.state.next_repair_id = self.state.next_repair_id.max(record.id + 1);
            self.state.repairs.push(Repair {
                id: record.id,
                name: record.name,
                plan: record.plan,
                output: record.output,
                notes_output: record.notes_output,
                markdown: record.markdown,
                title: record.title,
                details: record.details,
                last_written: record.last_written,
                best_cleaned: record.best_cleaned,
                failed_rounds: record.failed_rounds,
                next_attempt: Instant::now() + Duration::from_secs(30),
                in_flight: None,
                window: None,
            });
        }
        self.sync_repairs_to_disk();
    }

    /// Keep the unfinished cleanup retries on disk: written when they change,
    /// deleted when none are left.
    fn sync_repairs_to_disk(&mut self) {
        let bytes = if self.state.repairs.is_empty() {
            Vec::new()
        } else {
            let records: Vec<RepairRecord> = self
                .state
                .repairs
                .iter()
                .map(|repair| RepairRecord {
                    id: repair.id,
                    name: repair.name.clone(),
                    plan: repair.plan.clone(),
                    output: repair.output.clone(),
                    notes_output: repair.notes_output.clone(),
                    markdown: repair.markdown,
                    title: repair.title.clone(),
                    details: repair.details.clone(),
                    last_written: repair.last_written.clone(),
                    best_cleaned: repair.best_cleaned.clone(),
                    failed_rounds: repair.failed_rounds,
                })
                .collect();
            serde_json::to_vec(&records).unwrap_or_default()
        };
        let print = fingerprint(&bytes);
        if print == self.state.repairs_saved {
            return;
        }
        // Whatever happens, do not try again until something changes.
        self.state.repairs_saved = print;
        if bytes.is_empty() {
            self.state.spool.clear_repairs();
        } else if let Err(error) = self.state.spool.save_repairs(&bytes) {
            warn!(error = %error, "could not keep the cleanup retries on disk");
        }
    }

    /// Every couple of seconds: look at the power supply, show the power
    /// saving mode in the tray, and start the waiting jobs once power saving
    /// is off and the engine is ready.
    fn poll_power(&mut self) {
        if self.state.power_checked_at.elapsed() < POWER_CHECK_INTERVAL {
            return;
        }
        self.state.power_checked_at = Instant::now();
        self.sync_repairs_to_disk();
        let held_before = self.hold_for_power();
        self.state.power = power::status();
        let shown = (
            self.power_saving_on(),
            self.state.settings.power_saving_auto,
        );
        if self.state.power_mode_shown != Some(shown) {
            self.state.power_mode_shown = Some(shown);
            self.state.tray.set_power_saving(shown.0, shown.1);
        }
        let holding = self.hold_for_power();
        if holding != held_before {
            info!(
                holding,
                battery_saver = self.state.power.battery_saver,
                "power state changed"
            );
        }
        self.drop_stopped_held_jobs();
        if !holding
            && !self.state.held.is_empty()
            && !self.state.shutting_down
            && self.engine_ready()
        {
            if held_before {
                let count = self.state.held.len();
                self.show_overlay_notice(
                    NoticeKind::Info,
                    format!(
                        "Power saving ended: starting {count} waiting job{}",
                        if count == 1 { "" } else { "s" }
                    ),
                );
            }
            self.release_held_jobs();
        }
    }

    /// Stop pressed in the window of a job that is still waiting for power:
    /// it is dropped, and its files deleted.
    fn drop_stopped_held_jobs(&mut self) {
        let (stopped, waiting): (Vec<HeldJob>, Vec<HeldJob>) = std::mem::take(&mut self.state.held)
            .into_iter()
            .partition(|held| {
                held.job
                    .stop
                    .as_ref()
                    .is_some_and(|stop| stop.load(Ordering::Acquire))
            });
        self.state.held = waiting;
        for held in stopped {
            info!(id = held.spool_id, "waiting job stopped by the user");
            self.state.spool.remove(held.spool_id);
            held.job.tell_window(WindowMsg::Stopped);
        }
    }

    /// The tray toggle was clicked: keep the choice in settings.json.
    fn toggle_power_saving(&mut self) {
        let on = self.state.tray.power_saving_checked();
        match Settings::load().and_then(|mut saved| {
            saved.power_saving = on;
            saved.save().map(|()| saved)
        }) {
            Ok(_) => info!(on, "power saving changed from the tray"),
            Err(error) => warn!(error = %error, "could not save the power saving choice"),
        }
        self.state.settings.power_saving = on;
        self.state.power_mode_shown = None;
        self.state.power_checked_at = Instant::now()
            .checked_sub(POWER_CHECK_INTERVAL)
            .unwrap_or_else(Instant::now);
        self.show_overlay_notice(
            NoticeKind::Info,
            if on {
                "Power saving on: recordings wait before transcription until you turn it off"
            } else {
                "Power saving off"
            },
        );
    }

    fn enqueue_long_form(&mut self, samples: Vec<f32>, job: FileJob, kind: &str) {
        let config = match &job.session {
            Some((part, settings)) => JobConfig::meeting_part(settings.clone(), *part, job.notes),
            None if job.notes => JobConfig::meeting_notes(self.state.settings.clone()),
            None => JobConfig::long_form(self.state.settings.clone()),
        };
        match self
            .state
            .worker
            .enqueue(samples, media::TARGET_RATE, config)
        {
            Ok(id) => {
                let windowed = job.overlay_free();
                job.tell_window(WindowMsg::Stage {
                    stage: transcript_window::Stage::Queued,
                    done: 0,
                    total: 0,
                });
                self.state.file_jobs.insert(id, job);
                self.state.pending_jobs = self.state.pending_jobs.saturating_add(1);
                if !windowed {
                    self.state.overlay_state.job_queued(id);
                }
                info!(id, kind, "long-form transcription job queued");
                self.update_job_status();
                self.sync_overlay();
            }
            Err(error) => {
                warn!(error = %error, "failed to queue long-form transcription");
                let reported = job.tell_window(WindowMsg::Failed {
                    error: format!("could not queue transcription: {error}"),
                });
                if reported {
                    self.state.tray.set_status(
                        tray::IconColor::Error,
                        "HouWritr — could not queue transcription (see log)",
                    );
                } else {
                    self.show_job_warning("Could not queue transcription (see log)");
                }
            }
        }
    }

    /// Whether the meeting recording indicator should be on screen: the saved
    /// setting, unless the tray menu overrode it for this session.
    fn meeting_indicator_visible(&self) -> bool {
        self.state
            .meeting_indicator_override
            .unwrap_or(self.state.settings.show_meeting_indicator)
    }

    /// Push the indicator's visibility and current view to its pill. Cheap:
    /// only sends a command when something changed (about once a second
    /// while recording, for the clock).
    fn sync_meeting_indicator(&mut self) {
        let visible = self.meeting_indicator_visible();
        if visible != self.state.meeting_overlay_enabled {
            match self.state.meeting_overlay.set_enabled(visible) {
                Ok(()) => self.state.meeting_overlay_enabled = visible,
                Err(error) => warn!(error, "failed to update meeting indicator"),
            }
            self.state.tray.set_meeting_indicator_visible(visible);
        }
        // The meeting pill occupies the same spot as the status pill, so it
        // steps aside while that one is on screen (writing or a notice) and
        // returns as soon as it clears.
        let status_pill_showing =
            self.state.settings.overlay && !self.state.overlay_state.last_view.is_hidden();
        let desired = match &self.state.meeting {
            Some(recorder) if !status_pill_showing => {
                OverlayViewState::meeting(recorder.elapsed().as_secs(), recorder.is_paused())
            }
            _ => OverlayViewState::Hidden,
        };
        if desired != self.state.meeting_overlay_view {
            match self.state.meeting_overlay.set_state(desired.clone()) {
                Ok(()) => self.state.meeting_overlay_view = desired,
                Err(error) => warn!(error, "failed to update meeting indicator"),
            }
        }
    }

    fn toggle_meeting_indicator(&mut self) {
        let show = !self.meeting_indicator_visible();
        self.state.meeting_indicator_override = Some(show);
        self.state.tray.set_meeting_indicator_visible(show);
        self.sync_overlay();
        self.sync_meeting_indicator();
        if self.state.meeting.is_some() {
            // Hiding must never be silent while a recording is running.
            let message = if show {
                "Recording indicator shown"
            } else {
                "Indicator hidden · still recording (tray icon is red)"
            };
            self.show_overlay_notice(NoticeKind::Info, message);
        }
    }

    fn start_meeting(&mut self) {
        if self.state.pending_meeting.is_some() {
            self.show_overlay_notice(
                NoticeKind::Warning,
                "Answer the meeting notes question first",
            );
            return;
        }
        if !self.require_ready_engine() {
            return;
        }
        match MeetingRecorder::start(self.state.meeting_level.clone()) {
            Ok(recorder) => {
                let sources = recorder.sources();
                let id = self.state.next_meeting_session;
                self.state.next_meeting_session += 1;
                let started = meeting::LocalTime::now();
                self.state.meeting_session = Some(MeetingSession {
                    id,
                    started,
                    partial: media::unique_path(meeting::partial_output_path(started)),
                    settings: self.state.settings.clone(),
                });
                self.state.meeting = Some(recorder);
                self.state
                    .tray
                    .set_meeting_state(tray::MeetingMenuState::Recording);
                self.state.tray.set_status(
                    tray::IconColor::Recording,
                    &format!("HouWritr — recording meeting ({sources})"),
                );
                let hidden_note = if self.meeting_indicator_visible() {
                    ""
                } else {
                    " (indicator hidden)"
                };
                self.show_overlay_notice(
                    NoticeKind::Success,
                    format!("Recording meeting: {sources}{hidden_note}"),
                );
            }
            Err(error) => {
                warn!(error = %error, "could not start meeting capture");
                self.show_job_warning(&format!(
                    "Could not start meeting recording: {}",
                    short_error(&error.to_string())
                ));
            }
        }
    }

    fn toggle_meeting_pause(&mut self) {
        let Some(recorder) = self.state.meeting.as_mut() else {
            return;
        };
        if recorder.is_paused() {
            let resumed = recorder.resume().map(|()| recorder.sources());
            match resumed {
                Ok(_) => {
                    self.state
                        .tray
                        .set_meeting_state(tray::MeetingMenuState::Recording);
                    self.update_job_status();
                    self.show_overlay_notice(NoticeKind::Success, "Meeting recording resumed");
                }
                Err(error) => {
                    // Still paused: the user can retry or stop and keep what
                    // was recorded so far.
                    warn!(error = %error, "could not resume meeting capture");
                    self.show_job_warning(&format!(
                        "Could not resume the meeting: {}",
                        short_error(&error.to_string())
                    ));
                }
            }
        } else {
            recorder.pause();
            // Nothing more is recorded while paused, so use the time: hand
            // everything recorded so far to the transcriber now.
            let segment = recorder.take_segment(false, meeting::MIN_PAUSE_SEGMENT_SAMPLES);
            self.state
                .tray
                .set_meeting_state(tray::MeetingMenuState::Paused);
            self.update_job_status();
            self.show_overlay_notice(NoticeKind::Info, "Meeting recording paused");
            if let Some(segment) = segment {
                self.queue_meeting_part(segment);
            }
        }
    }

    /// Cut the finished stretch of the recording off for transcription.
    /// Called once an hour of audio is waiting (see `MeetingRecorder::roll_due`).
    fn roll_meeting_part(&mut self) {
        let segment = self
            .state
            .meeting
            .as_mut()
            .and_then(|recorder| recorder.take_segment(true, 1));
        if let Some(segment) = segment {
            info!(
                seconds = segment.samples.len() / media::TARGET_RATE as usize,
                "rolling meeting part cut"
            );
            self.queue_meeting_part(segment);
        }
    }

    /// Queue a finished part of the running meeting as background jobs.
    fn queue_meeting_part(&mut self, segment: meeting::Segment) {
        let Some(session) = &self.state.meeting_session else {
            return;
        };
        let (id, settings, partial) = (
            session.id,
            session.settings.clone(),
            session.partial.clone(),
        );
        let pieces = media::plan_windows(
            &segment.samples,
            media::TARGET_RATE,
            MEETING_PIECE_SECONDS,
            MEETING_PIECE_SEARCH_SECONDS,
        );
        for range in pieces {
            let part = SessionPart {
                id,
                offset_samples: segment.start + range.start,
                last: false,
            };
            let job = FileJob {
                // Where this part's text is appended as soon as it is ready.
                output: partial.clone(),
                markdown: false,
                notes: false,
                notes_output: None,
                title: String::new(),
                details: String::new(),
                window: None,
                stop: None,
                stop_sent: false,
                background: true,
                session: Some((part, settings.clone())),
                partial: None,
                spool: None,
            };
            self.dispatch_long_form(segment.samples[range].to_vec(), job, "meeting part");
        }
    }

    /// End the recording. With `ask_for_notes`, offer to also create AI notes
    /// (only when a provider is configured); otherwise queue the transcript.
    fn stop_meeting(&mut self, ask_for_notes: bool) {
        let Some(recorder) = self.state.meeting.take() else {
            return;
        };
        self.state
            .tray
            .set_meeting_state(tray::MeetingMenuState::Idle);
        let recorded = recorder.elapsed();
        let tail = recorder.finish();
        let session = self.state.meeting_session.take();
        let total_samples = tail.start + tail.samples.len();
        let Some(session) = session else {
            return;
        };
        if total_samples < media::TARGET_RATE as usize {
            self.show_job_warning("Meeting recording was too short to transcribe");
            self.update_job_status();
            return;
        }
        info!(seconds = recorded.as_secs(), "meeting capture stopped");
        let pending = PendingMeeting {
            tail,
            session,
            recorded,
        };

        let provider_ready = enhance::notes_configuration(&self.state.settings).is_ok();
        let decision = notes_decision(
            self.state.settings.meeting_notes,
            provider_ready,
            ask_for_notes,
        );
        match decision {
            NotesDecision::Create | NotesDecision::Skip | NotesDecision::SkipNoProvider => {
                if decision == NotesDecision::SkipNoProvider {
                    self.show_overlay_notice(
                        NoticeKind::Info,
                        "Set up an AI provider in Settings to get meeting notes",
                    );
                }
                self.state.pending_meeting = Some(pending);
                self.finish_pending_meeting(decision == NotesDecision::Create);
                return;
            }
            NotesDecision::Ask => {}
        }

        self.state.pending_meeting = Some(pending);
        self.state.tray.set_status(
            tray::IconColor::Transcribing,
            "HouWritr — meeting recorded; waiting for your answer about notes",
        );
        let proxy = self.state.proxy.clone();
        let asked = thread::Builder::new()
            .name("meeting-notes-prompt".into())
            .spawn(move || {
                let choice = meeting::ask_create_notes(recorded);
                let _ = proxy.send_event(UserEvent::MeetingNotesChoice(choice));
            });
        if let Err(error) = asked {
            warn!(error = %error, "could not show the notes question; saving the transcript only");
            self.finish_pending_meeting(false);
        }
    }

    /// Queue the final part of the recording (and with it the stitched
    /// transcript, and notes if wanted) after the "create notes?" answer.
    fn finish_pending_meeting(&mut self, create_notes: bool) {
        let Some(pending) = self.state.pending_meeting.take() else {
            return;
        };
        let started = pending.session.started;
        let markdown = self.state.settings.markdown_transcripts;
        let mut output = meeting::default_output_path(started);
        if markdown {
            output.set_extension("md");
        }
        let notes_output =
            create_notes.then(|| media::unique_path(meeting::default_notes_path(started)));
        let part = SessionPart {
            id: pending.session.id,
            offset_samples: pending.tail.start,
            last: true,
        };
        let job = FileJob {
            output: media::unique_path(output),
            markdown,
            notes: create_notes,
            notes_output,
            window: None,
            stop: None,
            stop_sent: false,
            background: false,
            partial: Some(pending.session.partial),
            spool: None,
            session: Some((part, pending.session.settings)),
            title: format!(
                "Meeting {:04}-{:02}-{:02} {:02}:{:02}",
                started.year, started.month, started.day, started.hour, started.minute
            ),
            details: format!(
                "Recorded with HouWritr · {}",
                format_duration(pending.recorded.as_secs() as usize)
            ),
        };
        self.dispatch_long_form(pending.tail.samples, job, "meeting");
    }

    /// Report a file/meeting outcome. Windowed jobs only touch the tray; the
    /// popup carries the details, so the overlay pill stays out of it.
    fn report_long_form(&mut self, windowed: bool, warning: bool, message: &str) {
        if windowed {
            let color = if warning {
                tray::IconColor::Error
            } else {
                tray::IconColor::Idle
            };
            self.state
                .tray
                .set_status(color, &format!("HouWritr — {message}"));
        } else if warning {
            self.show_job_warning(message);
        } else {
            self.show_job_success(message);
        }
    }

    fn deliver_file_transcript(
        &mut self,
        mut job: FileJob,
        text: &str,
        notes: Option<String>,
        diarization: DiarizationOutcome,
        cleanup: Option<crate::enhance::CleanupReport>,
        repair: Option<RepairPlan>,
        cleanup_warning: Option<FallbackReason>,
    ) {
        let windowed = job.window.is_some();
        if text.trim().is_empty() {
            job.tell_window(WindowMsg::Failed {
                error: "No speech detected in the audio".into(),
            });
            self.report_long_form(windowed, true, "No speech detected in the audio");
            return;
        }
        let document = render_transcript(job.markdown, &job.title, &job.details, text);
        let text = document.as_str();
        let output = match write_file(&job.output, text) {
            Ok(()) => job.output.clone(),
            Err(error) => {
                warn!(error = %error, file = %job.output.display(), "failed to write transcript");
                // Never lose a long transcription to a bad path: save it in
                // Documents\HouWritr instead.
                let fallback = fallback_output(&job.output);
                match fallback
                    .as_deref()
                    .map(|path| (path, write_file(path, text)))
                {
                    Some((path, Ok(()))) => {
                        warn!(file = %path.display(), "transcript saved in the fallback folder");
                        let path = path.to_path_buf();
                        job.output = path.clone();
                        path
                    }
                    _ => {
                        job.tell_window(WindowMsg::Failed {
                            error: format!("could not save {}: {error}", job.output.display()),
                        });
                        self.report_long_form(
                            windowed,
                            true,
                            &format!(
                                "Could not save transcript ({})",
                                short_error(&error.to_string())
                            ),
                        );
                        return;
                    }
                }
            }
        };
        info!(file = %output.display(), chars = text.chars().count(), "transcript saved");
        // The finished transcript supersedes the in-progress one. (If saving
        // failed above we returned earlier and the partial file is kept.)
        if let Some(partial) = &job.partial {
            let _ = std::fs::remove_file(partial);
        }

        // Meeting notes go next to the transcript as their own file.
        let mut notes_saved = false;
        if let (Some(notes_text), Some(notes_path)) = (notes.as_deref(), job.notes_output.as_ref())
        {
            let transcript_name = output
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            let document = render_notes(&job.title, &job.details, &transcript_name, notes_text);
            let saved = write_file(notes_path, &document);
            match saved {
                Ok(()) => notes_saved = true,
                Err(error) => {
                    warn!(error = %error, file = %notes_path.display(), "failed to write notes");
                }
            }
        }

        let missing = if job.notes && !notes_saved {
            "AI notes"
        } else {
            "AI cleanup"
        };
        // What is still missing is retried in the background; say so.
        let retrying = repair.is_some();
        let summary = cleanup.as_ref().and_then(cleanup_summary);
        if let Some(plan) = repair {
            self.register_repair(&job, plan, text.to_string());
        }
        let (words, chars, preview) = transcript_window::summarize(text);
        // Warnings for the popup: AI problems and unavailable speaker labels.
        let mut warnings = Vec::new();
        if let Some(reason) = &cleanup_warning {
            warnings.push(format!(
                "Saved without {missing}: {}.",
                cleanup_failure_text(reason)
            ));
        }
        let notice = if cleanup_warning.is_none() {
            cleanup.as_ref().and_then(cleanup_notice)
        } else {
            None
        };
        if retrying {
            warnings.push("HouWritr will retry automatically and update the file.".into());
        }
        if let DiarizationOutcome::Failed(reason) = &diarization {
            warnings.push(format!("Speaker labels unavailable: {reason}"));
        }
        if let Some(note) = self.unlabelled_note(&diarization, text) {
            warnings.push(note);
        }
        let speakers = match &diarization {
            DiarizationOutcome::Labelled { speakers } => Some(*speakers),
            _ => None,
        };
        let shown = job.tell_window(WindowMsg::Done {
            output_path: output.display().to_string(),
            words,
            chars,
            preview,
            warning: (!warnings.is_empty()).then(|| warnings.join(" ")),
            notice,
            speakers,
        });
        // The popup has "Show in folder"; only pop Explorer open ourselves
        // when nothing is showing the result (meetings, or popup closed).
        if !shown {
            if let Err(error) = diagnostics::reveal(&output) {
                warn!(error = %error, "failed to reveal transcript");
            }
        }
        let name = output
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "transcript".into());
        let retry_note = if retrying {
            "; retrying automatically"
        } else {
            ""
        };
        if let Some(reason) = cleanup_warning {
            warn!(?reason, "transcript saved without AI output");
            self.report_long_form(
                windowed,
                true,
                &format!(
                    "Saved {name} without {missing} ({}){retry_note}",
                    cleanup_failure_text(&reason)
                ),
            );
        } else if let Some(summary) = &summary {
            self.report_long_form(
                windowed,
                true,
                &format!("Saved {name} · {summary}{retry_note}"),
            );
        } else if let DiarizationOutcome::Failed(reason) = &diarization {
            self.report_long_form(
                windowed,
                true,
                &format!(
                    "Saved {name} without speaker labels ({})",
                    short_error(reason)
                ),
            );
        } else {
            let mut message = if notes_saved {
                format!("Saved {name} and meeting notes")
            } else {
                format!("Saved {name}")
            };
            if let DiarizationOutcome::Labelled { speakers } = &diarization {
                message.push_str(&format!(" · {speakers} speakers"));
            } else if let Some(note) = self.unlabelled_note(&diarization, text) {
                message.push_str(&format!(" · {}", note.trim_end_matches('.')));
            }
            self.report_long_form(windowed, false, &message);
        }
    }

    /// Why a finished file or meeting transcript has no speaker labels, when
    /// that is not an error (switched off, or only one voice heard).
    fn unlabelled_note(&self, diarization: &DiarizationOutcome, text: &str) -> Option<String> {
        match diarization {
            // A retried cleanup reports `NotRequested` for text that already
            // carries labels from the first pass.
            DiarizationOutcome::NotRequested
                if !self.state.settings.diarize_speakers && !text.contains("Speaker 1:") =>
            {
                Some("Speaker labels are off. Turn them on in Settings → Speaker labels.".into())
            }
            DiarizationOutcome::SingleSpeaker => {
                Some("Only one speaker was detected, so no labels were added.".into())
            }
            _ => None,
        }
    }

    /// Remember a delivered transcript whose AI steps can still be retried.
    fn register_repair(&mut self, job: &FileJob, plan: RepairPlan, written: String) {
        let id = self.state.next_repair_id;
        self.state.next_repair_id += 1;
        let name = job
            .output
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "transcript".into());
        info!(id, file = %job.output.display(), "AI cleanup will be retried in the background");
        self.state.repairs.push(Repair {
            id,
            name,
            best_cleaned: vec![0; plan.pieces.len()],
            plan,
            output: job.output.clone(),
            notes_output: job.notes_output.clone(),
            markdown: job.markdown,
            title: job.title.clone(),
            details: job.details.clone(),
            last_written: written,
            failed_rounds: 0,
            next_attempt: Instant::now() + Duration::from_secs(REPAIR_BACKOFF_SECONDS[0]),
            in_flight: None,
            window: job.window.clone(),
        });
        if let Some(repair) = self.state.repairs.last() {
            repair.tell(
                format!(
                    "Saved, but {} is unfinished; trying again in the background in about {}.",
                    repair.what(),
                    minutes(REPAIR_BACKOFF_SECONDS[0])
                ),
                false,
                false,
            );
        }
    }

    /// Start the next retry of every delivered transcript that is due.
    fn run_due_repairs(&mut self) {
        if self.state.shutting_down || self.state.repairs.is_empty() || self.hold_for_power() {
            return;
        }
        let now = Instant::now();
        for index in 0..self.state.repairs.len() {
            let repair = &self.state.repairs[index];
            if repair.in_flight.is_some() || now < repair.next_attempt {
                continue;
            }
            let Some(step) = next_repair_step(&repair.plan) else {
                continue;
            };
            let kind = match step {
                RepairStep::Piece(piece) => Rerun::Clean {
                    raw: repair.plan.pieces[piece].raw.clone().unwrap_or_default(),
                },
                RepairStep::Notes => Rerun::Notes {
                    text: repair.plan.assemble(),
                },
            };
            let repair_id = repair.id;
            let config = JobConfig::rerun(self.state.settings.clone(), kind);
            match self
                .state
                .worker
                .enqueue(Vec::new(), media::TARGET_RATE, config)
            {
                Ok(job_id) => {
                    let repair = &mut self.state.repairs[index];
                    repair.in_flight = Some(job_id);
                    let attempt = repair.failed_rounds + 1;
                    repair.tell(
                        format!(
                            "Retrying {} in the background (try {attempt})…",
                            repair.what()
                        ),
                        true,
                        false,
                    );
                    self.state.repair_jobs.insert(job_id, (repair_id, step));
                }
                Err(error) => {
                    warn!(error = %error, "could not queue a cleanup retry");
                    self.state.repairs[index].next_attempt = now + Duration::from_secs(60);
                }
            }
        }
    }

    /// A background retry finished.
    fn finish_repair_step(
        &mut self,
        job_id: u64,
        text: String,
        notes: Option<String>,
        cleanup: Option<crate::enhance::CleanupReport>,
    ) {
        let Some((repair_id, step)) = self.state.repair_jobs.remove(&job_id) else {
            return;
        };
        let Some(index) = self
            .state
            .repairs
            .iter()
            .position(|repair| repair.id == repair_id)
        else {
            return;
        };
        let mut transcript_changed = false;
        let progressed;
        {
            let repair = &mut self.state.repairs[index];
            repair.in_flight = None;
            match step {
                RepairStep::Piece(piece) => {
                    let report = cleanup.unwrap_or_default();
                    let improved = report.cleaned > repair.best_cleaned[piece];
                    let complete = !report.needs_retry();
                    if improved || complete {
                        repair.plan.pieces[piece].text = text;
                        transcript_changed = true;
                    }
                    if improved {
                        repair.best_cleaned[piece] = report.cleaned;
                    }
                    if complete {
                        repair.plan.pieces[piece].raw = None;
                    }
                    progressed = improved || complete;
                }
                RepairStep::Notes => match (notes, repair.notes_output.clone()) {
                    (Some(notes), Some(path)) => {
                        let transcript_name = repair
                            .output
                            .file_name()
                            .map(|name| name.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        let document =
                            render_notes(&repair.title, &repair.details, &transcript_name, &notes);
                        match write_file(&path, &document) {
                            Ok(()) => {
                                repair.plan.notes_failed = false;
                                progressed = true;
                            }
                            Err(error) => {
                                warn!(error = %error, file = %path.display(), "could not write the retried notes");
                                progressed = false;
                            }
                        }
                    }
                    _ => progressed = false,
                },
            }
        }

        if transcript_changed {
            if let Err(reason) = self.rewrite_repaired_transcript(index) {
                // Most likely the file was edited or moved: leave it alone.
                let repair = self.state.repairs.remove(index);
                repair.tell(
                    format!("Left the file as it is ({reason}); retries stopped."),
                    false,
                    true,
                );
                warn!(file = %repair.output.display(), reason, "stopping cleanup retries");
                self.show_job_warning(&format!(
                    "Left {} as it is ({reason}); automatic cleanup stopped",
                    repair.name
                ));
                return;
            }
        }

        let repair = &mut self.state.repairs[index];
        if repair.plan.is_complete() {
            let repair = self.state.repairs.remove(index);
            repair.tell(
                format!(
                    "{} finished in the background; the file was updated.",
                    repair.what()
                ),
                false,
                true,
            );
            info!(file = %repair.output.display(), "transcript cleanup finished in the background");
            self.show_job_success(&format!("Finished cleaning up {}", repair.name));
        } else if progressed {
            repair.next_attempt = Instant::now() + REPAIR_STEP_PAUSE;
            repair.tell(
                format!("Part of {} done; continuing in a moment…", repair.what()),
                false,
                false,
            );
        } else {
            self.schedule_repair_retry(index);
        }
    }

    /// The provider refused a background retry outright (an unsupported
    /// model, a rejected key...): trying again later would only be refused
    /// again, and the transcript must not be reported as cleaned.
    fn stop_rejected_repair(&mut self, job_id: u64, status: u16) {
        let Some((repair_id, _)) = self.state.repair_jobs.remove(&job_id) else {
            return;
        };
        let Some(index) = self
            .state
            .repairs
            .iter()
            .position(|repair| repair.id == repair_id)
        else {
            return;
        };
        let repair = self.state.repairs.remove(index);
        repair.tell(
            format!("Stopped retrying: the provider rejected the request (HTTP {status})."),
            false,
            true,
        );
        warn!(
            file = %repair.output.display(),
            status,
            "background cleanup stopped: the provider rejected the request"
        );
        self.show_job_warning(&format!(
            "Stopped cleaning up {}: the provider rejected the request (HTTP {status}, see the log)",
            repair.name
        ));
    }

    /// A background retry did not run to completion.
    fn abort_repair_step(&mut self, job_id: u64) {
        let Some((repair_id, _)) = self.state.repair_jobs.remove(&job_id) else {
            return;
        };
        if let Some(index) = self
            .state
            .repairs
            .iter()
            .position(|repair| repair.id == repair_id)
        {
            self.state.repairs[index].in_flight = None;
            self.schedule_repair_retry(index);
        }
    }

    /// Count a fruitless round and wait longer, or give up.
    fn schedule_repair_retry(&mut self, index: usize) {
        let repair = &mut self.state.repairs[index];
        repair.failed_rounds += 1;
        match repair_delay(repair.failed_rounds) {
            Some(delay) => {
                repair.next_attempt = Instant::now() + delay;
                repair.tell(
                    format!(
                        "{} did not finish this time; trying again in about {}.",
                        repair.what(),
                        minutes(delay.as_secs())
                    ),
                    false,
                    false,
                );
            }
            None => {
                let repair = self.state.repairs.remove(index);
                repair.tell(
                    format!(
                        "Could not finish {}; the file keeps the recognised text.",
                        repair.what()
                    ),
                    false,
                    true,
                );
                warn!(file = %repair.output.display(), "giving up on background cleanup");
                self.show_job_warning(&format!(
                    "Could not finish cleaning up {}; it stays as recognised",
                    repair.name
                ));
            }
        }
    }

    /// Rewrite the saved transcript from the repaired pieces, unless the file
    /// was changed by someone else since we wrote it.
    fn rewrite_repaired_transcript(&mut self, index: usize) -> Result<(), String> {
        let repair = &mut self.state.repairs[index];
        let current = std::fs::read_to_string(&repair.output)
            .map_err(|_| "the file was moved or deleted".to_string())?;
        if current != repair.last_written {
            return Err("you have edited it".into());
        }
        let document = render_transcript(
            repair.markdown,
            &repair.title,
            &repair.details,
            &repair.plan.assemble(),
        );
        write_file(&repair.output, &document).map_err(|error| error.to_string())?;
        repair.last_written = document;
        Ok(())
    }

    fn handle_worker_event(&mut self, el: &ActiveEventLoop, event: WorkerEvent) {
        match event {
            // Background retries of AI cleanup / notes never touch the
            // overlay, the job counters or the tray.
            WorkerEvent::JobProgress {
                id, done, total, ..
            } if self.state.repair_jobs.contains_key(&id) => {
                // Only the popup of the retried transcript hears of it.
                let repair_id = self.state.repair_jobs[&id].0;
                if let Some(window) = self
                    .state
                    .repairs
                    .iter()
                    .find(|repair| repair.id == repair_id)
                    .and_then(|repair| repair.window.as_ref())
                {
                    let _ = window.send(WindowMsg::Stage {
                        stage: transcript_window::Stage::CleaningUp,
                        done,
                        total,
                    });
                }
            }
            WorkerEvent::JobStarted { id } | WorkerEvent::EnhancementStarted { id }
                if self.state.repair_jobs.contains_key(&id) => {}
            WorkerEvent::JobCompleted {
                id,
                text,
                cleanup,
                notes,
                enhancement_warning,
                ..
            } if self.state.repair_jobs.contains_key(&id) => {
                if let Some(FallbackReason::ProviderRejected(status)) = enhancement_warning {
                    self.stop_rejected_repair(id, status);
                } else {
                    self.finish_repair_step(id, text, notes, cleanup);
                }
            }
            WorkerEvent::JobFailed { id, .. } | WorkerEvent::JobDiscarded { id }
                if self.state.repair_jobs.contains_key(&id) =>
            {
                self.abort_repair_step(id);
            }
            WorkerEvent::ModelState {
                generation,
                engine,
                state,
            } => {
                if generation != self.state.load_generation || engine != self.state.settings.engine
                {
                    return;
                }
                let color = match &state {
                    ModelState::Failed { .. } | ModelState::Cancelled => tray::IconColor::Error,
                    ModelState::Ready => tray::IconColor::Transcribing,
                    ModelState::Missing
                    | ModelState::Downloading { .. }
                    | ModelState::Verifying => tray::IconColor::Transcribing,
                };
                let status = state.status_text(&engine);
                self.state.model_state = Some(state);
                self.state
                    .tray
                    .set_status(color, &format!("HouWritr — {status}"));
            }
            WorkerEvent::EngineLoading { generation, engine } => {
                if generation != self.state.load_generation || engine != self.state.settings.engine
                {
                    return;
                }
                self.state.engine_state = EngineState::Loading;
                self.state.tray.set_status(
                    tray::IconColor::Transcribing,
                    &format!("HouWritr — loading {engine}"),
                );
            }
            WorkerEvent::EngineReady {
                generation,
                engine,
                label,
            } => {
                if generation != self.state.load_generation || engine != self.state.settings.engine
                {
                    return;
                }
                info!(engine, label, "selected engine ready");
                self.state.engine_state = EngineState::Ready {
                    label: label.clone(),
                };
                self.state.model_state = Some(ModelState::Ready);
                if self.state.pending_jobs == 0 {
                    self.update_job_status();
                }
            }
            WorkerEvent::EngineFailed {
                generation,
                engine,
                error,
            } => {
                if generation != self.state.load_generation || engine != self.state.settings.engine
                {
                    return;
                }
                warn!(engine, error = %error, "selected engine failed to load");
                let short_error = short_error(&error);
                self.state.engine_state = EngineState::Failed { error };
                self.state.tray.set_status(
                    tray::IconColor::Error,
                    &format!("HouWritr — {engine} failed: {short_error}; choose Retry"),
                );
            }
            WorkerEvent::JobStarted { id } if self.is_windowed_job(id) => {
                self.state.active_job = Some(id);
                self.tell_job_window(
                    id,
                    WindowMsg::Stage {
                        stage: transcript_window::Stage::Transcribing,
                        done: 0,
                        total: 0,
                    },
                );
                self.update_job_status();
            }
            WorkerEvent::EnhancementStarted { id } if self.is_windowed_job(id) => {
                self.tell_job_window(
                    id,
                    WindowMsg::Stage {
                        stage: transcript_window::Stage::CleaningUp,
                        done: 0,
                        total: 0,
                    },
                );
                self.state.tray.set_status(
                    tray::IconColor::Transcribing,
                    "HouWritr — cleaning up transcript",
                );
            }
            WorkerEvent::JobProgress {
                id,
                stage,
                done,
                total,
            } => {
                let stage = match stage {
                    JobStage::Diarizing => transcript_window::Stage::Diarizing,
                    JobStage::Transcribing => transcript_window::Stage::Transcribing,
                    JobStage::CleaningUp => transcript_window::Stage::CleaningUp,
                };
                self.tell_job_window(id, WindowMsg::Stage { stage, done, total });
            }
            WorkerEvent::JobStarted { id } => {
                if !self.state.overlay_state.job_started(id) {
                    warn!(id, "ignoring start event for unknown or inactive job");
                    return;
                }
                self.state.active_job = Some(id);
                self.update_job_status();
                self.sync_overlay();
            }
            WorkerEvent::EnhancementStarted { id } => {
                if !self.state.overlay_state.enhancement_started(id) {
                    warn!(id, "ignoring enhancement phase for inactive job");
                    return;
                }
                self.state.tray.set_status(
                    tray::IconColor::Transcribing,
                    &format!(
                        "HouWritr — enhancing; {} job(s) remaining",
                        self.state.pending_jobs
                    ),
                );
                self.sync_overlay();
            }
            WorkerEvent::JobCompleted {
                id,
                text,
                diarization,
                cleanup,
                repair,
                notes,
                enhancement_warning,
            } => {
                let file_output = self.state.file_jobs.remove(&id);
                self.forget_spooled(&file_output);
                let windowed = file_output.as_ref().is_some_and(FileJob::overlay_free);
                let finished = if windowed {
                    self.finish_windowed_job(id)
                } else {
                    self.finish_job(id)
                };
                if !finished {
                    warn!(id, "ignoring completion for tombstoned or unknown job");
                    return;
                }
                self.update_job_status();
                self.sync_overlay();
                if self.state.discarding_jobs {
                    warn!(id, "suppressing completed job during discard shutdown");
                    return;
                }
                if let Some(job) = file_output {
                    if job.background {
                        // A non-final meeting part. The worker keeps its text
                        // for the final stitch; here it only extends the
                        // in-progress transcript on disk.
                        if let Err(error) = meeting::append_partial(&job.output, &text) {
                            warn!(error = %error, "could not extend the in-progress transcript");
                        }
                        return;
                    }
                    self.deliver_file_transcript(
                        job,
                        &text,
                        notes,
                        diarization,
                        cleanup,
                        repair,
                        enhancement_warning,
                    );
                    return;
                }
                // Every job the app starts is a file, a meeting or a retry,
                // all handled above.
                warn!(id, "ignoring completion of a job with no transcript file");
            }
            WorkerEvent::JobFailed { id, error } => {
                let file_job = self.state.file_jobs.remove(&id);
                if !self.state.shutting_down {
                    self.forget_spooled(&file_job);
                }
                let windowed = file_job.as_ref().is_some_and(FileJob::overlay_free);
                let finished = if windowed {
                    self.finish_windowed_job(id)
                } else {
                    self.finish_job(id)
                };
                if !finished {
                    warn!(id, "ignoring failure for tombstoned or unknown job");
                    return;
                }
                warn!(id, error = %error, "transcription job failed");
                if self.state.pending_jobs == 0 {
                    self.state.tray.set_status(
                        tray::IconColor::Error,
                        &format!("HouWritr — transcription failed: {}", short_error(&error)),
                    );
                } else {
                    self.update_job_status();
                }
                if let Some(job) = file_job.filter(FileJob::overlay_free) {
                    job.tell_window(WindowMsg::Failed {
                        error: error.clone(),
                    });
                } else {
                    self.show_overlay_notice(
                        NoticeKind::Error,
                        format!("Transcription failed: {}", short_error(&error)),
                    );
                }
            }
            WorkerEvent::JobDiscarded { id } => {
                let file_job = self.state.file_jobs.remove(&id);
                // Dropped because the app is closing: the job is not done, so
                // its files stay for the next start.
                if !self.state.shutting_down && !self.state.discarding_jobs {
                    self.forget_spooled(&file_job);
                }
                let windowed = file_job.as_ref().is_some_and(FileJob::overlay_free);
                let finished = if windowed {
                    self.finish_windowed_job(id)
                } else {
                    self.finish_job(id)
                };
                if !finished {
                    return;
                }
                if let Some(job) = file_job {
                    if job.stop_sent {
                        job.tell_window(WindowMsg::Stopped);
                        info!(id, "transcription job stopped by the user");
                    } else {
                        job.tell_window(WindowMsg::Failed {
                            error: "Cancelled because HouWritr is quitting".into(),
                        });
                    }
                }
                info!(id, "transcription job discarded");
                self.update_job_status();
                self.sync_overlay();
            }
            WorkerEvent::ShutdownComplete => {
                info!("worker shutdown complete");
                self.exit(el);
            }
        }
    }

    /// Whether `id` is a file job whose progress lives in a popup window
    /// rather than the overlay pill.
    fn is_windowed_job(&self, id: u64) -> bool {
        self.state
            .file_jobs
            .get(&id)
            .is_some_and(FileJob::overlay_free)
    }

    /// Act on Stop presses in the Jobs window and show it the current list.
    fn update_jobs_window(&mut self) {
        let Some(window) = &self.state.jobs_window else {
            return;
        };
        if !window.is_open() {
            self.state.jobs_window = None;
            return;
        }
        for key in window.take_stops() {
            self.stop_from_jobs_window(&key);
        }
        let rows = self.job_rows();
        if let Some(window) = self.state.jobs_window.as_mut() {
            window.show(rows);
        }
    }

    fn stop_from_jobs_window(&mut self, key: &str) {
        if let Some(id) = key
            .strip_prefix("job:")
            .and_then(|id| id.parse::<u64>().ok())
        {
            // The same path as Stop in the job's own window (see `tick`).
            if let Some(stop) = self
                .state
                .file_jobs
                .get(&id)
                .and_then(|job| job.stop.as_ref())
            {
                stop.store(true, Ordering::Release);
            }
        } else if let Some(repair_id) = key
            .strip_prefix("retry:")
            .and_then(|id| id.parse::<u64>().ok())
        {
            let Some(index) = self
                .state
                .repairs
                .iter()
                .position(|repair| repair.id == repair_id)
            else {
                return;
            };
            let repair = self.state.repairs.remove(index);
            if let Some(job_id) = repair.in_flight {
                self.state.worker.stop_job(job_id);
            }
            info!(file = %repair.output.display(), "background cleanup cancelled from the jobs window");
            repair.tell(
                "Retry cancelled; the file keeps the recognised text.".into(),
                false,
                true,
            );
        }
    }

    /// Everything running or waiting, for the Jobs window.
    fn job_rows(&self) -> Vec<crate::jobs_window::JobRow> {
        use crate::jobs_window::JobRow;
        let mut rows = Vec::new();
        if let Some(meeting) = &self.state.meeting {
            rows.push(JobRow {
                key: "meeting".into(),
                kind: "Meeting".into(),
                name: "Recording".into(),
                state: format!(
                    "Recording for {} · stop it from the tray menu",
                    transcript_window::format_clock(meeting.elapsed().as_secs())
                ),
                fraction: None,
                stop_label: None,
            });
        }
        for held in &self.state.held {
            let kind = match &held.job.session {
                Some((part, _)) if part.last => "Meeting (last part)",
                Some(_) => "Meeting part",
                None => "File",
            };
            let name = if held.job.title.is_empty() {
                held.job
                    .output
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default()
            } else {
                held.job.title.clone()
            };
            rows.push(JobRow {
                key: format!("held:{}", held.spool_id),
                kind: kind.into(),
                name,
                state: if self.state.settings.power_saving_auto {
                    "Waiting for Windows power saving to end (saved on disk)".into()
                } else {
                    "Waiting for power saving to be turned off (saved on disk)".into()
                },
                fraction: None,
                stop_label: None,
            });
        }
        let mut ids: Vec<u64> = self.state.file_jobs.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            let job = &self.state.file_jobs[&id];
            let kind = match &job.session {
                Some((part, _)) if part.last => "Meeting (last part)",
                Some(_) => "Meeting part",
                None => "File",
            };
            let (state, fraction) = if job.stop_sent {
                ("Stopping…".to_string(), None)
            } else if self.state.running_jobs.contains(&id) {
                progress_text(self.state.job_progress.get(&id))
            } else {
                ("Queued".to_string(), None)
            };
            rows.push(JobRow {
                key: format!("job:{id}"),
                kind: kind.into(),
                name: job.title.clone(),
                state,
                fraction,
                stop_label: (job.stop.is_some() && !job.stop_sent).then(|| "Stop".into()),
            });
        }
        let now = Instant::now();
        for repair in &self.state.repairs {
            let (state, fraction) = match repair.in_flight {
                Some(job_id) => progress_text(self.state.job_progress.get(&job_id)),
                None => (
                    format!(
                        "Next try in about {}",
                        minutes(repair.next_attempt.saturating_duration_since(now).as_secs())
                    ),
                    None,
                ),
            };
            rows.push(JobRow {
                key: format!("retry:{}", repair.id),
                kind: format!("{} retry", repair.what()),
                name: repair.name.clone(),
                state,
                fraction,
                stop_label: Some("Cancel".into()),
            });
        }
        rows
    }

    fn tell_job_window(&self, id: u64, message: WindowMsg) {
        if let Some(job) = self.state.file_jobs.get(&id) {
            job.tell_window(message);
        }
    }

    /// `finish_job` for jobs the overlay reducer never saw.
    fn finish_windowed_job(&mut self, id: u64) -> bool {
        self.state.pending_jobs = self.state.pending_jobs.saturating_sub(1);
        if self.state.active_job == Some(id) {
            self.state.active_job = None;
        }
        true
    }

    fn finish_job(&mut self, id: u64) -> bool {
        if !self.state.overlay_state.job_finished(id) {
            return false;
        }
        self.state.pending_jobs = self.state.pending_jobs.saturating_sub(1);
        if self.state.active_job == Some(id) {
            self.state.active_job = None;
        }
        true
    }

    fn update_job_status(&self) {
        // While a meeting is being recorded the tray shows *that*, however
        // many earlier parts are being transcribed in the background.
        if let Some(recorder) = &self.state.meeting {
            let (color, state) = if recorder.is_paused() {
                (
                    tray::IconColor::Transcribing,
                    "meeting paused, not recording",
                )
            } else {
                (tray::IconColor::Recording, "recording meeting")
            };
            let queued = if self.state.pending_jobs > 0 {
                format!(
                    " · transcribing earlier parts ({} queued)",
                    self.state.pending_jobs
                )
            } else if !self.state.held.is_empty() {
                format!(
                    " · {} part{} waiting for power",
                    self.state.held.len(),
                    if self.state.held.len() == 1 { "" } else { "s" }
                )
            } else {
                String::new()
            };
            self.state.tray.set_status(
                color,
                &format!("HouWritr — {state} ({}){queued}", recorder.sources()),
            );
            return;
        }
        if self.state.pending_jobs > 0 {
            let noun = if self.state.pending_jobs == 1 {
                "job"
            } else {
                "jobs"
            };
            self.state.tray.set_status(
                tray::IconColor::Transcribing,
                &format!(
                    "HouWritr — transcribing; {} {noun} remaining",
                    self.state.pending_jobs
                ),
            );
        } else if !self.state.held.is_empty() && !self.state.shutting_down {
            let count = self.state.held.len();
            self.state.tray.set_status(
                tray::IconColor::Transcribing,
                &format!(
                    "HouWritr — {count} job{} waiting for power (saved on disk)",
                    if count == 1 { "" } else { "s" }
                ),
            );
        } else if !self.state.shutting_down {
            if let EngineState::Failed { error } = &self.state.engine_state {
                self.state.tray.set_status(
                    tray::IconColor::Error,
                    &format!("HouWritr — engine failed: {}", short_error(error)),
                );
            } else if let Some(issue) = self.configuration_issue() {
                self.state.tray.set_status(
                    tray::IconColor::Error,
                    &format!("HouWritr — {}", short_error(issue)),
                );
            } else {
                match &self.state.engine_state {
                    EngineState::Ready { label } => self
                        .state
                        .tray
                        .set_status(tray::IconColor::Idle, &format!("HouWritr — ready: {label}")),
                    EngineState::Loading => self
                        .state
                        .tray
                        .set_status(tray::IconColor::Transcribing, "HouWritr — loading engine"),
                    EngineState::NotStarted => self
                        .state
                        .tray
                        .set_status(tray::IconColor::Error, "HouWritr — engine not ready"),
                    EngineState::Failed { .. } => unreachable!(),
                }
            }
        }
    }

    fn show_job_success(&mut self, message: &str) {
        self.state
            .tray
            .set_status(tray::IconColor::Idle, &format!("HouWritr — {message}"));
        self.show_overlay_notice(NoticeKind::Success, message);
    }

    fn show_job_warning(&mut self, message: &str) {
        self.show_job_warning_kind(NoticeKind::Warning, message);
    }

    fn show_job_warning_kind(&mut self, kind: NoticeKind, message: &str) {
        self.state
            .tray
            .set_status(tray::IconColor::Error, &format!("HouWritr — {message}"));
        self.show_overlay_notice(kind, message);
    }

    fn configuration_issue(&self) -> Option<&str> {
        self.state
            .settings_error
            .as_deref()
            .or(self.state.credential_health.message.as_deref())
    }

    fn blocked_recording_status(&self) -> (tray::IconColor, String) {
        if let Some(state) = &self.state.model_state {
            let color = match state {
                ModelState::Failed { .. } | ModelState::Cancelled => tray::IconColor::Error,
                _ => tray::IconColor::Transcribing,
            };
            return (
                color,
                format!(
                    "HouWritr — transcription blocked: {}",
                    state.status_text(&self.state.settings.engine)
                ),
            );
        }
        match &self.state.engine_state {
            EngineState::Loading => (
                tray::IconColor::Transcribing,
                "HouWritr — transcription blocked: engine is loading".into(),
            ),
            EngineState::Failed { error } => (
                tray::IconColor::Error,
                format!(
                    "HouWritr — transcription blocked: engine failed: {}",
                    short_error(error)
                ),
            ),
            EngineState::NotStarted => (
                tray::IconColor::Error,
                "HouWritr — transcription blocked: engine is not ready".into(),
            ),
            EngineState::Ready { .. } => (tray::IconColor::Idle, "HouWritr — engine ready".into()),
        }
    }

    fn request_shutdown(&mut self, el: &ActiveEventLoop) {
        if self.state.shutting_down {
            return;
        }
        self.sync_repairs_to_disk();
        // Queue the meeting recorded so far so quitting never loses it; the
        // existing "finish queued jobs?" prompt then covers it.
        self.stop_meeting(false);
        if self.state.pending_meeting.is_some() {
            // Quit while the notes question was open: keep the transcript.
            self.finish_pending_meeting(false);
        }
        info!(pending_jobs = self.state.pending_jobs, "quit requested");
        self.state.shutting_down = true;
        self.state.hk_stop.store(true, Ordering::Relaxed);

        let mode = if self.state.pending_jobs > 0 {
            prompt_shutdown_mode(self.state.pending_jobs)
        } else {
            ShutdownMode::Discard
        };
        self.state.discarding_jobs = mode == ShutdownMode::Discard;
        if mode == ShutdownMode::Discard {
            // A cleanup request can wait minutes on a slow connection; quitting
            // must not wait for it.
            crate::enhance::cancel_requests();
        }
        let status = match mode {
            ShutdownMode::Wait => format!(
                "HouWritr — finishing {} queued job(s) before exit",
                self.state.pending_jobs
            ),
            ShutdownMode::Discard => "HouWritr — discarding queued work and exiting".into(),
        };
        self.state
            .tray
            .set_status(tray::IconColor::Transcribing, &status);
        if let Err(error) = self.state.worker.shutdown(mode) {
            warn!(error = %error, "worker shutdown command failed");
            self.exit(el);
        }
    }

    fn exit(&self, el: &ActiveEventLoop) {
        crate::local_llm::shutdown();
        el.exit();
    }
}

/// Runs on the file-import thread after a file was picked: opens the popup,
/// decodes, and returns the event that hands everything to the tray loop.
fn import_file(
    path: PathBuf,
    markdown: bool,
    cleanup: bool,
    cleanup_engine: String,
    engine: String,
) -> UserEvent {
    let (window, messages) = mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));
    if let Err(error) = transcript_window::spawn_window_thread(messages, stop.clone()) {
        // The job still runs; the result is saved and revealed as usual.
        warn!(error = %error, "could not open the transcript window");
    }
    let output = media::unique_path(path.with_extension(if markdown { "md" } else { "txt" }));
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "audio file".into());
    let init = |duration: String| WindowMsg::Init {
        title: name.clone(),
        source_path: path.display().to_string(),
        duration,
        engine: engine.clone(),
        cleanup,
        cleanup_engine: cleanup_engine.clone(),
        markdown,
        output_path: output.display().to_string(),
    };
    let _ = window.send(init("reading…".into()));
    let _ = window.send(WindowMsg::Stage {
        stage: transcript_window::Stage::Decoding,
        done: 0,
        total: 0,
    });
    match media::decode_file(&path) {
        Ok(samples) => {
            let seconds = samples.len() / media::TARGET_RATE as usize;
            info!(seconds, "file decoded");
            let _ = window.send(init(format_duration(seconds)));
            UserEvent::FileDecoded {
                path,
                samples,
                output,
                markdown,
                window,
                stop,
            }
        }
        Err(error) => {
            let message = format!("{error:#}");
            let _ = window.send(WindowMsg::Failed {
                error: message.clone(),
            });
            UserEvent::FileImportEnded(Some(message))
        }
    }
}

fn format_duration(total_seconds: usize) -> String {
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

fn short_error(error: &str) -> String {
    const MAX_CHARS: usize = 140;
    let mut shortened = error.chars().take(MAX_CHARS).collect::<String>();
    if error.chars().count() > MAX_CHARS {
        shortened.push('…');
    }
    shortened
}

fn prompt_shutdown_mode(pending_jobs: usize) -> ShutdownMode {
    use windows::core::HSTRING;
    use windows::Win32::UI::WindowsAndMessaging::{
        MessageBoxW, IDNO, MB_DEFBUTTON1, MB_ICONQUESTION, MB_SETFOREGROUND, MB_YESNO,
    };

    let message = HSTRING::from(format!(
        "HouWritr is still processing {pending_jobs} transcription job(s).\n\n\
         Choose Yes to wait for them to finish before exiting.\n\
         Choose No to discard queued results and exit after the current native call reaches a safe boundary."
    ));
    let title = HSTRING::from("HouWritr");
    let response = unsafe {
        MessageBoxW(
            None,
            &message,
            &title,
            MB_YESNO | MB_ICONQUESTION | MB_DEFBUTTON1 | MB_SETFOREGROUND,
        )
    };
    if response == IDNO {
        ShutdownMode::Discard
    } else {
        ShutdownMode::Wait
    }
}

#[cfg(test)]
mod tests {
    use super::OverlayReducer;
    use crate::overlay::{NoticeKind, OverlayViewState, ProcessingPhase};
    use std::time::Instant;

    fn file_job(window: bool, background: bool) -> super::FileJob {
        super::FileJob {
            output: std::path::PathBuf::new(),
            markdown: false,
            notes: false,
            notes_output: None,
            title: String::new(),
            details: String::new(),
            window: window.then(|| std::sync::mpsc::channel().0),
            stop: None,
            stop_sent: false,
            background,
            session: None,
            partial: None,
            spool: None,
        }
    }

    #[test]
    fn partly_cleaned_text_gets_a_calm_notice_that_counts_what_was_kept() {
        let report = |total, cleaned, unreachable| crate::enhance::CleanupReport {
            partly: 0,
            total,
            cleaned,
            unreachable,
        };
        assert_eq!(super::cleanup_notice(&report(48, 48, 0)), None);
        assert_eq!(super::cleanup_notice(&report(0, 0, 0)), None);
        let notice = super::cleanup_notice(&report(48, 39, 0)).unwrap();
        assert!(notice.starts_with("AI cleanup finished: 39 of 48 sections cleaned."));
        assert!(notice.contains("9 kept the recognised text"));
        let notice = super::cleanup_notice(&crate::enhance::CleanupReport {
            total: 48,
            cleaned: 40,
            partly: 5,
            unreachable: 0,
        })
        .unwrap();
        assert!(notice.contains("40 of 48 sections cleaned."));
        assert!(notice.contains("5 more in part"));
        assert!(notice.contains("3 kept the recognised text"));
        assert!(!notice.contains("could not be sent"));
        let notice = super::cleanup_notice(&report(48, 30, 8)).unwrap();
        assert!(notice.contains("10 kept"));
        assert!(notice.contains("8 could not be sent"));
    }

    #[test]
    fn repairs_back_off_and_eventually_give_up() {
        let seconds: Vec<Option<u64>> = (1..=6)
            .map(|round| super::repair_delay(round).map(|d| d.as_secs()))
            .collect();
        assert_eq!(
            seconds,
            vec![
                Some(120),
                Some(300),
                Some(600),
                Some(1_200),
                Some(1_800),
                None
            ]
        );
        // Round zero (nothing failed yet) behaves like the first.
        assert_eq!(super::repair_delay(0).map(|d| d.as_secs()), Some(120));
    }

    #[test]
    fn repairs_fix_unfinished_pieces_in_order_then_the_notes() {
        use super::{next_repair_step, RepairStep};
        use crate::worker::{RepairPiece, RepairPlan};
        let piece = |raw: Option<&str>| RepairPiece {
            text: "t".into(),
            raw: raw.map(str::to_string),
        };
        let mut plan = RepairPlan {
            pieces: vec![piece(None), piece(Some("b")), piece(Some("c"))],
            separator: " ".into(),
            unlabel: false,
            notes_failed: true,
        };
        assert_eq!(next_repair_step(&plan), Some(RepairStep::Piece(1)));
        plan.pieces[1].raw = None;
        assert_eq!(next_repair_step(&plan), Some(RepairStep::Piece(2)));
        plan.pieces[2].raw = None;
        assert_eq!(next_repair_step(&plan), Some(RepairStep::Notes));
        plan.notes_failed = false;
        assert_eq!(next_repair_step(&plan), None);
    }

    #[test]
    fn documents_are_rendered_the_same_way_every_time() {
        use super::{render_notes, render_transcript};
        assert_eq!(render_transcript(false, "T", "d", "plain"), "plain");
        let markdown = render_transcript(true, "Meeting 1", "details", "body text.");
        assert!(markdown.starts_with("# Meeting 1\n\n*details*\n\n"));
        let notes = render_notes(
            "Meeting 2026-03-09 14:05",
            "Recorded",
            "Meeting.txt",
            "## Summary",
        );
        assert!(notes.starts_with("# Meeting notes 2026-03-09 14:05\n"));
        assert!(notes.contains("transcript: Meeting.txt"));
    }

    #[test]
    fn the_cleanup_summary_only_appears_when_something_is_left() {
        use super::cleanup_summary;
        use crate::enhance::CleanupReport;
        let report = |total, cleaned| CleanupReport {
            partly: 0,
            total,
            cleaned,
            unreachable: 0,
        };
        assert_eq!(
            cleanup_summary(&report(25, 21)).unwrap(),
            "cleaned 21 of 25 sections"
        );
        assert!(cleanup_summary(&report(25, 25)).is_none());
        assert!(cleanup_summary(&report(0, 0)).is_none());
    }

    #[test]
    fn notes_decision_covers_every_mode() {
        use super::notes_decision;
        use super::NotesDecision::{Ask as AskUser, Create, Skip, SkipNoProvider};
        use crate::settings::MeetingNotesMode::{Always, Ask, Never};
        // Never: no notes, no question, no hint.
        for ready in [true, false] {
            for may_ask in [true, false] {
                assert_eq!(notes_decision(Never, ready, may_ask), Skip);
            }
        }
        // Always: notes whenever a provider is ready, even while quitting.
        assert_eq!(notes_decision(Always, true, true), Create);
        assert_eq!(notes_decision(Always, true, false), Create);
        assert_eq!(notes_decision(Always, false, true), SkipNoProvider);
        // Ask: the question only when it can be shown and notes are possible.
        assert_eq!(notes_decision(Ask, true, true), AskUser);
        assert_eq!(notes_decision(Ask, true, false), Skip);
        assert_eq!(notes_decision(Ask, false, true), SkipNoProvider);
        assert_eq!(notes_decision(Ask, false, false), Skip);
    }

    #[test]
    fn popup_and_background_jobs_stay_out_of_the_status_pill() {
        // The final part of a meeting (shown in the pill) uses it.
        assert!(!file_job(false, false).overlay_free());
        // A file job shows in its popup, a background meeting part nowhere.
        assert!(file_job(true, false).overlay_free());
        assert!(file_job(false, true).overlay_free());
    }

    #[test]
    fn overlay_reducer_enforces_job_queue_notice_priority() {
        let now = Instant::now();
        let mut reducer = OverlayReducer::default();
        reducer.show_notice(NoticeKind::Warning, "old notice", now);
        reducer.job_queued(10);
        reducer.job_queued(11);
        assert_eq!(
            reducer.desired_view(now),
            OverlayViewState::processing(10, ProcessingPhase::Queued, 1)
        );

        assert!(reducer.job_started(10));
        assert_eq!(
            reducer.desired_view(now),
            OverlayViewState::processing(10, ProcessingPhase::Transcribing, 1)
        );

        assert!(reducer.enhancement_started(10));
        assert_eq!(
            reducer.desired_view(now),
            OverlayViewState::processing(10, ProcessingPhase::Enhancing, 1)
        );

        assert!(reducer.job_finished(10));
        assert_eq!(
            reducer.desired_view(now),
            OverlayViewState::processing(11, ProcessingPhase::Queued, 0)
        );
        assert!(reducer.job_finished(11));
        assert_eq!(
            reducer.desired_view(now),
            OverlayViewState::notice(NoticeKind::Warning, "old notice")
        );
    }

    #[test]
    fn progress_follows_the_recording_indicator_but_notices_still_show() {
        let now = Instant::now();
        let mut reducer = OverlayReducer::default();
        reducer.job_queued(1);
        assert!(reducer.job_started(1));
        assert!(reducer.enhancement_started(1));
        reducer.job_queued(2);
        reducer.progress_visible = false;
        assert_eq!(reducer.desired_view(now), OverlayViewState::Hidden);
        reducer.show_notice(NoticeKind::Error, "something failed", now);
        assert_eq!(
            reducer.desired_view(now),
            OverlayViewState::notice(NoticeKind::Error, "something failed")
        );
        reducer.progress_visible = true;
        assert_eq!(
            reducer.desired_view(now),
            OverlayViewState::processing(1, ProcessingPhase::Enhancing, 1)
        );
    }

    #[test]
    fn old_enhancement_and_terminal_events_cannot_clobber_newer_activity() {
        let now = Instant::now();
        let mut reducer = OverlayReducer::default();
        reducer.job_queued(1);
        reducer.job_queued(2);
        assert!(reducer.job_started(1));
        assert!(reducer.job_finished(1));
        assert!(reducer.job_started(2));

        assert!(!reducer.enhancement_started(1));
        assert!(!reducer.job_finished(1));
        assert_eq!(
            reducer.desired_view(now),
            OverlayViewState::processing(2, ProcessingPhase::Transcribing, 0)
        );
    }
}
