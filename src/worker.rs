use crate::asr::{self, Engine};
use crate::cleanup::{EnhanceOutcome, FallbackReason};
use crate::enhance;
use crate::model_manager::{CancellationToken, ModelManager, ModelState};
use crate::settings::Settings;
use anyhow::{anyhow, Context, Result};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use tracing::{info, warn};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShutdownMode {
    Wait,
    Discard,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordingIntent {
    Raw,
    Enhance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeliveryTarget {
    pub hwnd: isize,
    pub process_id: u32,
    pub process_creation_time_100ns: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct JobConfig {
    settings: Settings,
    intent: RecordingIntent,
    delivery_target: Option<DeliveryTarget>,
    /// Imported files and meetings: transcribed in silence-aligned windows
    /// instead of one pass, never LLM-enhanced, never pasted.
    long_form: bool,
    /// Label speakers (long-form jobs only; needs the diarization models).
    diarize: bool,
    /// Also ask the provider for structured meeting notes (implies
    /// `long_form`); the notes are placed above the transcript.
    notes: bool,
}

impl JobConfig {
    pub fn new(
        settings: Settings,
        intent: RecordingIntent,
        delivery_target: Option<DeliveryTarget>,
        _enhancement_disabled: bool,
    ) -> Self {
        Self {
            settings,
            intent,
            delivery_target,
            long_form: false,
            diarize: false,
            notes: false,
        }
    }

    pub fn long_form(settings: Settings) -> Self {
        Self {
            diarize: settings.diarize_speakers,
            settings,
            intent: RecordingIntent::Raw,
            delivery_target: None,
            long_form: true,
            notes: false,
        }
    }

    pub fn meeting_notes(settings: Settings) -> Self {
        Self {
            notes: true,
            ..Self::long_form(settings)
        }
    }

    fn should_enhance(&self) -> bool {
        if self.long_form {
            // Imported files and meetings have no Shift key to consult, so any
            // mode other than Never opts them into cleanup.
            return self.settings.enhance.mode.is_enabled();
        }
        self.intent == RecordingIntent::Enhance && self.settings.enhance.mode.is_enabled()
    }
}

/// Which long-running phase a [`WorkerEvent::JobProgress`] belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobStage {
    Diarizing,
    Transcribing,
    CleaningUp,
}

/// What happened to the optional "label the speakers" step of a job.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiarizationOutcome {
    /// Not requested (dictation, or the setting is off).
    NotRequested,
    /// The transcript carries `Speaker N:` labels for this many speakers.
    Labelled { speakers: usize },
    /// Only one speaker was found, so nothing was labelled.
    SingleSpeaker,
    /// Diarization failed; the transcript is unlabelled. Content-free reason.
    Failed(String),
}

pub enum WorkerEvent {
    ModelState {
        generation: u64,
        engine: String,
        state: ModelState,
    },
    EngineLoading {
        generation: u64,
        engine: String,
    },
    EngineReady {
        generation: u64,
        engine: String,
        label: String,
    },
    EngineFailed {
        generation: u64,
        engine: String,
        error: String,
    },
    JobStarted {
        id: u64,
    },
    EnhancementStarted {
        id: u64,
    },
    /// `done` of `total` windows (transcription) or slices (cleanup) finished.
    /// Only long-form jobs report progress.
    JobProgress {
        id: u64,
        stage: JobStage,
        done: usize,
        total: usize,
    },
    JobCompleted {
        id: u64,
        text: String,
        diarization: DiarizationOutcome,
        /// Structured meeting notes, kept apart from the transcript so the
        /// caller can save them as their own file. `Some` only for notes jobs
        /// whose generation succeeded.
        notes: Option<String>,
        auto_paste: bool,
        enhancement_warning: Option<FallbackReason>,
        delivery_target: Option<DeliveryTarget>,
    },
    JobFailed {
        id: u64,
        error: String,
    },
    JobDiscarded {
        id: u64,
    },
    ShutdownComplete,
}

struct Job {
    id: u64,
    generation: u64,
    samples: Vec<f32>,
    sample_rate: u32,
    config: JobConfig,
}

enum Command {
    Load {
        generation: u64,
        engine: String,
        cancellation: CancellationToken,
    },
    Transcribe(Job),
    Shutdown(ShutdownMode),
}

trait EngineLoader: Send + 'static {
    fn load(
        &mut self,
        engine: &str,
        cancellation: &CancellationToken,
        emit_model_state: &mut dyn FnMut(ModelState),
    ) -> Result<Box<dyn Engine>>;
}

struct RuntimeEngineLoader {
    models: ModelManager,
}

impl RuntimeEngineLoader {
    fn new() -> Result<Self> {
        Ok(Self {
            models: ModelManager::new().map_err(|error| anyhow!(error))?,
        })
    }
}

impl EngineLoader for RuntimeEngineLoader {
    fn load(
        &mut self,
        engine: &str,
        cancellation: &CancellationToken,
        emit_model_state: &mut dyn FnMut(ModelState),
    ) -> Result<Box<dyn Engine>> {
        asr::ensure_engine_supported(engine)?;
        let model_dir = self
            .models
            .ensure(engine, cancellation, emit_model_state)
            .map_err(|error| anyhow!(error))?;
        if cancellation.is_cancelled() {
            return Err(anyhow!("engine load cancelled"));
        }
        asr::load_from_dir(engine, model_dir)
    }
}

pub struct Worker {
    command_tx: Sender<Command>,
    event_rx: Receiver<WorkerEvent>,
    latest_generation: Arc<AtomicU64>,
    shutdown_requested: Arc<AtomicBool>,
    discard_requested: Arc<AtomicBool>,
    pending_jobs_by_generation: Arc<Mutex<HashMap<u64, usize>>>,
    job_events: Mutex<JobEventTracker>,
    synthetic_events: Mutex<VecDeque<WorkerEvent>>,
    transition_gate: Arc<Mutex<()>>,
    next_generation: u64,
    next_job: u64,
    load_cancellations: HashMap<u64, CancellationToken>,
}

impl Worker {
    pub fn spawn() -> Result<Self> {
        Self::spawn_with_loader(Box::new(RuntimeEngineLoader::new()?))
    }

    fn spawn_with_loader(loader: Box<dyn EngineLoader>) -> Result<Self> {
        let (command_tx, command_rx) = mpsc::channel();
        let (event_tx, event_rx) = mpsc::channel();
        let latest_generation = Arc::new(AtomicU64::new(0));
        let shutdown_requested = Arc::new(AtomicBool::new(false));
        let discard_requested = Arc::new(AtomicBool::new(false));
        let pending_jobs_by_generation = Arc::new(Mutex::new(HashMap::new()));
        let transition_gate = Arc::new(Mutex::new(()));
        let worker_generation = latest_generation.clone();
        let worker_shutdown = shutdown_requested.clone();
        let worker_discard = discard_requested.clone();
        let worker_pending_jobs = pending_jobs_by_generation.clone();
        let worker_transition_gate = transition_gate.clone();

        thread::Builder::new()
            .name("inference-worker".into())
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
                worker_main(
                    loader,
                    command_rx,
                    event_tx,
                    worker_generation,
                    worker_shutdown,
                    worker_discard,
                    worker_pending_jobs,
                    worker_transition_gate,
                )
            })
            .context("spawn inference worker")?;

        Ok(Self {
            command_tx,
            event_rx,
            latest_generation,
            shutdown_requested,
            discard_requested,
            pending_jobs_by_generation,
            job_events: Mutex::new(JobEventTracker::default()),
            synthetic_events: Mutex::new(VecDeque::new()),
            transition_gate,
            next_generation: 1,
            next_job: 1,
            load_cancellations: HashMap::new(),
        })
    }

    pub fn load(&mut self, engine: String) -> Result<u64> {
        if self.shutdown_requested.load(Ordering::Acquire) {
            return Err(anyhow!("inference worker is shutting down"));
        }
        let _transition = self.transition_gate.lock();
        let pending_jobs_by_generation = &self.pending_jobs_by_generation;
        self.load_cancellations.retain(|generation, cancellation| {
            let required = generation_has_pending_jobs(pending_jobs_by_generation, *generation);
            if !required {
                cancellation.cancel();
            }
            required
        });
        let generation = self.next_generation;
        self.next_generation = self.next_generation.saturating_add(1);
        self.latest_generation.store(generation, Ordering::Release);
        let cancellation = CancellationToken::default();
        self.load_cancellations
            .insert(generation, cancellation.clone());
        if let Err(error) = self.command_tx.send(Command::Load {
            generation,
            engine,
            cancellation,
        }) {
            if let Some(cancellation) = self.load_cancellations.remove(&generation) {
                cancellation.cancel();
            }
            return Err(error).context("send engine load command");
        }
        Ok(generation)
    }

    pub fn cancel_load(&mut self) -> u64 {
        let _transition = self.transition_gate.lock();
        let current_generation = self.latest_generation.load(Ordering::Acquire);
        if let Some(cancellation) = self.load_cancellations.get(&current_generation) {
            cancellation.cancel();
        }
        let invalid_generation = self.next_generation;
        self.next_generation = self.next_generation.saturating_add(1);
        self.latest_generation
            .store(invalid_generation, Ordering::Release);
        invalid_generation
    }

    pub fn enqueue(
        &mut self,
        samples: Vec<f32>,
        sample_rate: u32,
        config: JobConfig,
    ) -> Result<u64> {
        if self.shutdown_requested.load(Ordering::Acquire) {
            return Err(anyhow!("inference worker is shutting down"));
        }
        let id = self.next_job;
        self.next_job = self.next_job.saturating_add(1);
        let generation = self.latest_generation.load(Ordering::Acquire);
        register_pending_job(&self.pending_jobs_by_generation, generation);
        self.job_events.lock().register(id);
        if let Err(error) = self.command_tx.send(Command::Transcribe(Job {
            id,
            generation,
            samples,
            sample_rate,
            config,
        })) {
            unregister_pending_job(&self.pending_jobs_by_generation, generation);
            self.job_events.lock().forget(id);
            return Err(anyhow!("send transcription job: {error}"));
        }
        Ok(id)
    }

    pub fn shutdown(&self, mode: ShutdownMode) -> Result<()> {
        let _transition = self.transition_gate.lock();
        if mode == ShutdownMode::Discard {
            let discarded = self.job_events.lock().tombstone_all();
            self.synthetic_events.lock().extend(
                discarded
                    .into_iter()
                    .map(|id| WorkerEvent::JobDiscarded { id }),
            );
        }
        self.shutdown_requested.store(true, Ordering::Release);
        if mode == ShutdownMode::Discard {
            self.discard_requested.store(true, Ordering::Release);
        }
        for (generation, cancellation) in &self.load_cancellations {
            if mode == ShutdownMode::Discard
                || !generation_has_pending_jobs(&self.pending_jobs_by_generation, *generation)
            {
                cancellation.cancel();
            }
        }
        self.command_tx
            .send(Command::Shutdown(mode))
            .context("send worker shutdown command")
    }

    pub fn try_recv(&self) -> Option<WorkerEvent> {
        loop {
            if let Some(event) = self.synthetic_events.lock().pop_front() {
                return Some(event);
            }
            match self.event_rx.try_recv() {
                Ok(event) => {
                    if let Some(event) = self.normalize_event(event) {
                        return Some(event);
                    }
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => return None,
            }
        }
    }

    #[cfg(test)]
    fn recv_timeout(&self, timeout: std::time::Duration) -> Option<WorkerEvent> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if let Some(event) = self.synthetic_events.lock().pop_front() {
                return Some(event);
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            let event = self.event_rx.recv_timeout(remaining).ok()?;
            if let Some(event) = self.normalize_event(event) {
                return Some(event);
            }
        }
    }

    fn normalize_event(&self, event: WorkerEvent) -> Option<WorkerEvent> {
        self.job_events
            .lock()
            .normalize(event, self.discard_requested.load(Ordering::Acquire))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JobPhase {
    Queued,
    Transcribing,
    Enhancing,
}

#[derive(Default)]
struct JobEventTracker {
    phases: HashMap<u64, JobPhase>,
    tombstones: HashSet<u64>,
    tombstone_order: VecDeque<u64>,
}

impl JobEventTracker {
    fn register(&mut self, id: u64) {
        self.tombstones.remove(&id);
        self.phases.insert(id, JobPhase::Queued);
    }

    fn forget(&mut self, id: u64) {
        self.phases.remove(&id);
        self.tombstones.remove(&id);
    }

    fn finish(&mut self, id: u64) -> bool {
        if self.phases.remove(&id).is_some() {
            self.add_tombstone(id);
            true
        } else {
            false
        }
    }

    fn add_tombstone(&mut self, id: u64) {
        const TOMBSTONE_LIMIT: usize = 1024;
        if self.tombstones.insert(id) {
            self.tombstone_order.push_back(id);
        }
        while self.tombstone_order.len() > TOMBSTONE_LIMIT {
            if let Some(expired) = self.tombstone_order.pop_front() {
                self.tombstones.remove(&expired);
            }
        }
    }

    fn tombstone_all(&mut self) -> Vec<u64> {
        let mut ids = self.phases.drain().map(|(id, _)| id).collect::<Vec<_>>();
        ids.sort_unstable();
        for id in &ids {
            self.add_tombstone(*id);
        }
        ids
    }

    fn normalize(&mut self, event: WorkerEvent, discard_requested: bool) -> Option<WorkerEvent> {
        if Self::event_job_id(&event).is_some_and(|id| self.tombstones.contains(&id)) {
            return None;
        }
        let event = if discard_requested {
            match event {
                WorkerEvent::JobCompleted { id, .. } | WorkerEvent::JobFailed { id, .. } => {
                    WorkerEvent::JobDiscarded { id }
                }
                WorkerEvent::JobStarted { .. }
                | WorkerEvent::EnhancementStarted { .. }
                | WorkerEvent::JobProgress { .. } => {
                    return None;
                }
                other => other,
            }
        } else {
            event
        };

        match event {
            WorkerEvent::JobStarted { id } => {
                let phase = self.phases.get_mut(&id)?;
                if *phase != JobPhase::Queued {
                    return None;
                }
                *phase = JobPhase::Transcribing;
                Some(WorkerEvent::JobStarted { id })
            }
            WorkerEvent::EnhancementStarted { id } => {
                let phase = self.phases.get_mut(&id)?;
                if *phase != JobPhase::Transcribing {
                    return None;
                }
                *phase = JobPhase::Enhancing;
                Some(WorkerEvent::EnhancementStarted { id })
            }
            event @ WorkerEvent::JobProgress { id, .. } => {
                // Progress is only meaningful while the job is still live.
                matches!(
                    self.phases.get(&id),
                    Some(JobPhase::Transcribing | JobPhase::Enhancing)
                )
                .then_some(event)
            }
            event @ WorkerEvent::JobCompleted { id, .. } => {
                if !matches!(
                    self.phases.get(&id),
                    Some(JobPhase::Transcribing | JobPhase::Enhancing)
                ) {
                    return None;
                }
                self.finish(id).then_some(event)
            }
            event @ WorkerEvent::JobFailed { id, .. }
            | event @ WorkerEvent::JobDiscarded { id } => self.finish(id).then_some(event),
            WorkerEvent::ShutdownComplete => {
                let unfinished = self.phases.drain().map(|(id, _)| id).collect::<Vec<_>>();
                for id in unfinished {
                    self.add_tombstone(id);
                }
                Some(WorkerEvent::ShutdownComplete)
            }
            other => Some(other),
        }
    }

    fn event_job_id(event: &WorkerEvent) -> Option<u64> {
        match event {
            WorkerEvent::JobStarted { id }
            | WorkerEvent::EnhancementStarted { id }
            | WorkerEvent::JobProgress { id, .. }
            | WorkerEvent::JobCompleted { id, .. }
            | WorkerEvent::JobFailed { id, .. }
            | WorkerEvent::JobDiscarded { id } => Some(*id),
            _ => None,
        }
    }
}

fn worker_main(
    mut loader: Box<dyn EngineLoader>,
    command_rx: Receiver<Command>,
    event_tx: Sender<WorkerEvent>,
    latest_generation: Arc<AtomicU64>,
    shutdown_requested: Arc<AtomicBool>,
    discard_requested: Arc<AtomicBool>,
    pending_jobs_by_generation: Arc<Mutex<HashMap<u64, usize>>>,
    transition_gate: Arc<Mutex<()>>,
) {
    let mut engine: Option<Box<dyn Engine>> = None;
    let mut engine_generation: Option<u64> = None;
    let mut engine_cancellation: Option<CancellationToken> = None;
    while let Ok(command) = command_rx.recv() {
        match command {
            Command::Load {
                generation,
                engine: engine_name,
                cancellation,
            } => {
                {
                    let _transition = transition_gate.lock();
                    if cancellation.is_cancelled()
                        || !should_process_load(
                            generation,
                            latest_generation.load(Ordering::Acquire),
                            shutdown_requested.load(Ordering::Acquire),
                            discard_requested.load(Ordering::Acquire),
                            &pending_jobs_by_generation,
                        )
                    {
                        continue;
                    }
                }
                engine = None;
                engine_generation = None;
                engine_cancellation = None;
                let _ = event_tx.send(WorkerEvent::EngineLoading {
                    generation,
                    engine: engine_name.clone(),
                });
                let mut emit_model_state = |state| {
                    let _ = event_tx.send(WorkerEvent::ModelState {
                        generation,
                        engine: engine_name.clone(),
                        state,
                    });
                };
                let loaded = loader.load(&engine_name, &cancellation, &mut emit_model_state);
                let _transition = transition_gate.lock();
                if cancellation.is_cancelled()
                    || !should_process_load(
                        generation,
                        latest_generation.load(Ordering::Acquire),
                        shutdown_requested.load(Ordering::Acquire),
                        discard_requested.load(Ordering::Acquire),
                        &pending_jobs_by_generation,
                    )
                {
                    drop(loaded);
                    continue;
                }
                match loaded {
                    Ok(loaded_engine) => {
                        let label = loaded_engine.label().to_string();
                        engine = Some(loaded_engine);
                        engine_generation = Some(generation);
                        engine_cancellation = Some(cancellation.clone());
                        let _ = event_tx.send(WorkerEvent::EngineReady {
                            generation,
                            engine: engine_name,
                            label,
                        });
                    }
                    Err(error) => {
                        engine_generation = None;
                        engine_cancellation = None;
                        let _ = event_tx.send(WorkerEvent::EngineFailed {
                            generation,
                            engine: engine_name,
                            error: error.to_string(),
                        });
                    }
                }
            }
            Command::Transcribe(job) => {
                let _pending_job =
                    PendingJobGuard::new(&pending_jobs_by_generation, job.generation);
                {
                    let _transition = transition_gate.lock();
                    if discard_requested.load(Ordering::Acquire) {
                        let _ = event_tx.send(WorkerEvent::JobDiscarded { id: job.id });
                        continue;
                    }
                    if engine_generation != Some(job.generation)
                        || engine_cancellation
                            .as_ref()
                            .map(CancellationToken::is_cancelled)
                            .unwrap_or(true)
                    {
                        let _ = event_tx.send(WorkerEvent::JobFailed {
                            id: job.id,
                            error: "selected engine is not ready".into(),
                        });
                        continue;
                    }
                    let _ = event_tx.send(WorkerEvent::JobStarted { id: job.id });
                }
                let active_engine = engine
                    .as_mut()
                    .expect("validated engine generation is missing");
                let started = Instant::now();
                let mut diarization = DiarizationOutcome::NotRequested;
                let transcription = if job.config.long_form {
                    let interrupted = || {
                        interrupted_job_event(job.id, &discard_requested, &engine_cancellation)
                            .is_some()
                    };
                    let send_progress = |stage: JobStage, done: usize, total: usize| {
                        let _ = event_tx.send(WorkerEvent::JobProgress {
                            id: job.id,
                            stage,
                            done,
                            total,
                        });
                    };
                    let turns = if job.config.diarize {
                        send_progress(JobStage::Diarizing, 0, 0);
                        match asr::diarize_speakers(
                            &job.samples,
                            &interrupted,
                            &mut |done, total| send_progress(JobStage::Diarizing, done, total),
                        ) {
                            Ok(turns) => {
                                let speakers = asr::diarize::speaker_count(&turns);
                                if speakers >= 2 {
                                    diarization = DiarizationOutcome::Labelled { speakers };
                                    Some(turns)
                                } else {
                                    diarization = DiarizationOutcome::SingleSpeaker;
                                    None
                                }
                            }
                            Err(error) => {
                                // An interruption is handled right below by
                                // the transcription step's own checks.
                                warn!(error = %error, "speaker diarization failed; continuing without labels");
                                diarization =
                                    DiarizationOutcome::Failed(short_reason(&error.to_string()));
                                None
                            }
                        }
                    } else {
                        None
                    };
                    match turns {
                        Some(turns) => transcribe_turns(
                            &mut **active_engine,
                            &job.samples,
                            job.sample_rate,
                            &turns,
                            interrupted,
                            |done, total| send_progress(JobStage::Transcribing, done, total),
                        ),
                        None => transcribe_long_form(
                            &mut **active_engine,
                            &job.samples,
                            job.sample_rate,
                            interrupted,
                            |done, total| send_progress(JobStage::Transcribing, done, total),
                        ),
                    }
                } else {
                    active_engine.transcribe(&job.samples, job.sample_rate)
                };
                let text = match transcription {
                    Ok(text) => text,
                    Err(error) => {
                        let _transition = transition_gate.lock();
                        let event = terminal_job_event(
                            job.id,
                            &discard_requested,
                            &engine_cancellation,
                            || WorkerEvent::JobFailed {
                                id: job.id,
                                error: error.to_string(),
                            },
                        );
                        let _ = event_tx.send(event);
                        continue;
                    }
                };
                {
                    let _transition = transition_gate.lock();
                    if let Some(event) =
                        interrupted_job_event(job.id, &discard_requested, &engine_cancellation)
                    {
                        let _ = event_tx.send(event);
                        continue;
                    }
                }

                let (final_text, enhancement_warning) = if job.config.should_enhance() {
                    let _ = event_tx.send(WorkerEvent::EnhancementStarted { id: job.id });
                    info!("enhancement requested");
                    let still_wanted = || {
                        interrupted_job_event(job.id, &discard_requested, &engine_cancellation)
                            .is_none()
                    };
                    let outcome = if job.config.long_form {
                        enhance::enhance_long_form(
                            &text,
                            &job.config.settings,
                            || !still_wanted(),
                            |done, total| {
                                let _ = event_tx.send(WorkerEvent::JobProgress {
                                    id: job.id,
                                    stage: JobStage::CleaningUp,
                                    done,
                                    total,
                                });
                            },
                        )
                    } else {
                        enhance::enhance_outcome(&text, &job.config.settings, still_wanted)
                    };
                    let Some(outcome) = outcome else {
                        let _transition = transition_gate.lock();
                        if let Some(event) =
                            interrupted_job_event(job.id, &discard_requested, &engine_cancellation)
                        {
                            let _ = event_tx.send(event);
                        }
                        continue;
                    };
                    completed_cleanup(outcome)
                } else {
                    (text, None)
                };

                let (final_text, enhancement_warning, notes_text) = if job.config.notes {
                    // Phase events are idempotent: a repeat is ignored by the
                    // tracker when cleanup already announced the phase.
                    let _ = event_tx.send(WorkerEvent::EnhancementStarted { id: job.id });
                    info!("meeting notes requested");
                    let outcome =
                        enhance::generate_meeting_notes(&final_text, &job.config.settings, || {
                            interrupted_job_event(job.id, &discard_requested, &engine_cancellation)
                                .is_some()
                        });
                    match outcome {
                        enhance::NotesOutcome::Notes(notes) => (
                            final_text,
                            enhancement_warning,
                            Some(notes.trim().to_string()),
                        ),
                        enhance::NotesOutcome::Failed(reason) => {
                            info!(?reason, "meeting notes unavailable; saving transcript only");
                            (final_text, Some(reason), None)
                        }
                        enhance::NotesOutcome::Interrupted => {
                            let _transition = transition_gate.lock();
                            if let Some(event) = interrupted_job_event(
                                job.id,
                                &discard_requested,
                                &engine_cancellation,
                            ) {
                                let _ = event_tx.send(event);
                            }
                            continue;
                        }
                    }
                } else {
                    (final_text, enhancement_warning, None)
                };

                let chars = final_text.chars().count();
                let elapsed_ms = started.elapsed().as_millis() as u64;
                let _transition = transition_gate.lock();
                let event =
                    terminal_job_event(job.id, &discard_requested, &engine_cancellation, || {
                        WorkerEvent::JobCompleted {
                            id: job.id,
                            text: final_text,
                            diarization,
                            notes: notes_text,
                            auto_paste: job.config.settings.auto_paste,
                            enhancement_warning,
                            delivery_target: job.config.delivery_target,
                        }
                    });
                if matches!(&event, WorkerEvent::JobCompleted { .. }) {
                    info!(id = job.id, chars, elapsed_ms, "transcription job complete");
                }
                let _ = event_tx.send(event);
            }
            Command::Shutdown(mode) => {
                info!(?mode, "inference worker stopping");
                drop(engine.take());
                let _ = event_tx.send(WorkerEvent::ShutdownComplete);
                break;
            }
        }
    }
}

/// Longest window handed to the engine in one call for long-form audio.
const LONG_FORM_WINDOW_SECONDS: usize = 30;
/// Window cuts are placed at the quietest 100 ms inside this trailing span.
const LONG_FORM_SEARCH_SECONDS: usize = 4;

/// Split `samples` into windows of at most 30 s, cutting each at the quietest
/// point near the end so words are rarely bisected.
fn plan_long_form_windows(samples: &[f32], sample_rate: u32) -> Vec<std::ops::Range<usize>> {
    let rate = sample_rate.max(1) as usize;
    let max_len = rate * LONG_FORM_WINDOW_SECONDS;
    let search = rate * LONG_FORM_SEARCH_SECONDS;
    let frame = (rate / 10).max(1);
    let mut windows = Vec::new();
    let mut start = 0;
    while start < samples.len() {
        if samples.len() - start <= max_len {
            windows.push(start..samples.len());
            break;
        }
        let hard_end = start + max_len;
        let mut cut = hard_end;
        let mut quietest = f32::MAX;
        let mut position = hard_end - search;
        while position + frame <= hard_end {
            let energy: f32 = samples[position..position + frame]
                .iter()
                .map(|sample| sample * sample)
                .sum();
            if energy < quietest {
                quietest = energy;
                cut = position + frame / 2;
            }
            position += frame;
        }
        windows.push(start..cut);
        start = cut;
    }
    windows
}

fn transcribe_long_form(
    engine: &mut dyn Engine,
    samples: &[f32],
    sample_rate: u32,
    interrupted: impl Fn() -> bool,
    mut progress: impl FnMut(usize, usize),
) -> Result<String> {
    let windows = plan_long_form_windows(samples, sample_rate);
    let total = windows.len();
    let mut transcript = String::new();
    progress(0, total);
    for (index, window) in windows.into_iter().enumerate() {
        // The caller re-checks interruption and discards the partial text.
        if interrupted() {
            break;
        }
        let text = engine.transcribe(&samples[window], sample_rate)?;
        let text = text.trim();
        if !text.is_empty() {
            if !transcript.is_empty() {
                transcript.push(' ');
            }
            transcript.push_str(text);
        }
        progress(index + 1, total);
        info!(window = index + 1, total, "long-form window transcribed");
    }
    Ok(transcript)
}

/// One engine call: a stretch of audio spoken by one speaker.
#[derive(Debug, PartialEq, Eq)]
struct TurnWindow {
    range: std::ops::Range<usize>,
    speaker: usize,
}

/// Turn → engine windows: each turn is padded by 0.1 s (never into a
/// neighbouring turn) and long turns are split at quiet points to stay
/// within the 30 s engine window.
fn plan_turn_windows(turns: &[asr::diarize::Turn], samples: &[f32], rate: u32) -> Vec<TurnWindow> {
    // 0.1 s of context either side so word edges survive.
    let pad = rate as usize / 10;
    let mut windows = Vec::new();
    for (index, turn) in turns.iter().enumerate() {
        // Padding may only reach into the silence between turns, and each
        // neighbour gets half of it, so windows never overlap.
        let floor = match index.checked_sub(1).map(|i| &turns[i]) {
            Some(previous) => previous.end + turn.start.saturating_sub(previous.end) / 2,
            None => 0,
        };
        let ceiling = match turns.get(index + 1) {
            Some(next) => turn.end + next.start.saturating_sub(turn.end) / 2,
            None => samples.len(),
        };
        let start = turn.start.saturating_sub(pad).max(floor).min(turn.start);
        let end = (turn.end + pad)
            .min(ceiling)
            .max(turn.end)
            .min(samples.len());
        if start >= end {
            continue;
        }
        for sub in plan_long_form_windows(&samples[start..end], rate) {
            windows.push(TurnWindow {
                range: start + sub.start..start + sub.end,
                speaker: turn.speaker,
            });
        }
    }
    windows
}

/// Transcribe speaker by speaker and label the result:
/// `Speaker 1: …` blocks separated by blank lines; consecutive windows of
/// the same speaker run together.
fn transcribe_turns(
    engine: &mut dyn Engine,
    samples: &[f32],
    sample_rate: u32,
    turns: &[asr::diarize::Turn],
    interrupted: impl Fn() -> bool,
    mut progress: impl FnMut(usize, usize),
) -> Result<String> {
    let windows = plan_turn_windows(turns, samples, sample_rate);
    let total = windows.len();
    let mut transcript = String::new();
    let mut current: Option<usize> = None;
    progress(0, total);
    for (index, window) in windows.into_iter().enumerate() {
        // The caller re-checks interruption and discards the partial text.
        if interrupted() {
            break;
        }
        let text = engine.transcribe(&samples[window.range], sample_rate)?;
        let text = text.trim();
        if !text.is_empty() {
            if current == Some(window.speaker) {
                transcript.push(' ');
                transcript.push_str(text);
            } else {
                if !transcript.is_empty() {
                    transcript.push_str("\n\n");
                }
                transcript.push_str(&format!("Speaker {}: {text}", window.speaker + 1));
                current = Some(window.speaker);
            }
        }
        progress(index + 1, total);
    }
    Ok(transcript)
}

/// Short, single-line version of an error for user-facing notices.
fn short_reason(error: &str) -> String {
    let first_line = error.lines().next().unwrap_or_default();
    let mut text: String = first_line.chars().take(120).collect();
    if first_line.chars().count() > 120 {
        text.push('…');
    }
    text
}

fn completed_cleanup(outcome: EnhanceOutcome) -> (String, Option<FallbackReason>) {
    match outcome {
        EnhanceOutcome::Enhanced { text, .. } | EnhanceOutcome::Skipped { text, .. } => {
            (text, None)
        }
        EnhanceOutcome::RawFallback { text, reason } => (text, Some(reason)),
    }
}

fn interrupted_job_event(
    id: u64,
    discard_requested: &AtomicBool,
    engine_cancellation: &Option<CancellationToken>,
) -> Option<WorkerEvent> {
    if discard_requested.load(Ordering::Acquire) {
        Some(WorkerEvent::JobDiscarded { id })
    } else if engine_cancellation
        .as_ref()
        .map(CancellationToken::is_cancelled)
        .unwrap_or(true)
    {
        Some(WorkerEvent::JobFailed {
            id,
            error: "selected engine load was cancelled".into(),
        })
    } else {
        None
    }
}

fn terminal_job_event(
    id: u64,
    discard_requested: &AtomicBool,
    engine_cancellation: &Option<CancellationToken>,
    otherwise: impl FnOnce() -> WorkerEvent,
) -> WorkerEvent {
    interrupted_job_event(id, discard_requested, engine_cancellation).unwrap_or_else(otherwise)
}

fn should_process_load(
    generation: u64,
    latest_generation: u64,
    shutdown_requested: bool,
    discard_requested: bool,
    pending_jobs_by_generation: &Mutex<HashMap<u64, usize>>,
) -> bool {
    if discard_requested {
        return false;
    }
    let has_pending_jobs = generation_has_pending_jobs(pending_jobs_by_generation, generation);
    if shutdown_requested {
        has_pending_jobs
    } else {
        generation == latest_generation || has_pending_jobs
    }
}

fn register_pending_job(pending_jobs_by_generation: &Mutex<HashMap<u64, usize>>, generation: u64) {
    let mut pending = pending_jobs_by_generation.lock();
    *pending.entry(generation).or_default() += 1;
}

fn unregister_pending_job(
    pending_jobs_by_generation: &Mutex<HashMap<u64, usize>>,
    generation: u64,
) {
    let mut pending = pending_jobs_by_generation.lock();
    let Some(count) = pending.get_mut(&generation) else {
        return;
    };
    *count -= 1;
    if *count == 0 {
        pending.remove(&generation);
    }
}

fn generation_has_pending_jobs(
    pending_jobs_by_generation: &Mutex<HashMap<u64, usize>>,
    generation: u64,
) -> bool {
    pending_jobs_by_generation
        .lock()
        .get(&generation)
        .copied()
        .unwrap_or_default()
        > 0
}

struct PendingJobGuard<'a> {
    pending_jobs_by_generation: &'a Mutex<HashMap<u64, usize>>,
    generation: u64,
}

impl<'a> PendingJobGuard<'a> {
    fn new(pending_jobs_by_generation: &'a Mutex<HashMap<u64, usize>>, generation: u64) -> Self {
        Self {
            pending_jobs_by_generation,
            generation,
        }
    }
}

impl Drop for PendingJobGuard<'_> {
    fn drop(&mut self) {
        unregister_pending_job(self.pending_jobs_by_generation, self.generation);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    struct FakeLoader {
        failed: HashSet<String>,
        slow_engine: Option<String>,
        loads: Arc<Mutex<Vec<String>>>,
        calls: Arc<Mutex<Vec<i32>>>,
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
    }

    impl EngineLoader for FakeLoader {
        fn load(
            &mut self,
            engine: &str,
            _cancellation: &CancellationToken,
            emit_model_state: &mut dyn FnMut(ModelState),
        ) -> Result<Box<dyn Engine>> {
            self.loads.lock().push(engine.to_string());
            emit_model_state(ModelState::Ready);
            if self.slow_engine.as_deref() == Some(engine) {
                thread::sleep(Duration::from_millis(50));
            }
            if self.failed.contains(engine) {
                anyhow::bail!("injected load failure");
            }
            Ok(Box::new(FakeEngine {
                calls: self.calls.clone(),
                active: self.active.clone(),
                max_active: self.max_active.clone(),
                delay: Duration::from_millis(100),
            }))
        }
    }

    struct FakeEngine {
        calls: Arc<Mutex<Vec<i32>>>,
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
        delay: Duration,
    }

    impl Engine for FakeEngine {
        fn transcribe(&mut self, samples: &[f32], _sample_rate: u32) -> Result<String> {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(active, Ordering::SeqCst);
            thread::sleep(self.delay);
            let value = samples.first().copied().unwrap_or_default() as i32;
            self.calls.lock().push(value);
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(value.to_string())
        }

        fn label(&self) -> &'static str {
            "fake"
        }
    }

    fn worker(
        failed: &[&str],
        slow_engine: Option<&str>,
    ) -> (
        Worker,
        Arc<Mutex<Vec<i32>>>,
        Arc<AtomicUsize>,
        Arc<Mutex<Vec<String>>>,
    ) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let loads = Arc::new(Mutex::new(Vec::new()));
        let max_active = Arc::new(AtomicUsize::new(0));
        let loader = FakeLoader {
            failed: failed.iter().map(|value| value.to_string()).collect(),
            slow_engine: slow_engine.map(str::to_string),
            loads: loads.clone(),
            calls: calls.clone(),
            active: Arc::new(AtomicUsize::new(0)),
            max_active: max_active.clone(),
        };
        (
            Worker::spawn_with_loader(Box::new(loader)).unwrap(),
            calls,
            max_active,
            loads,
        )
    }

    fn wait_until(
        worker: &Worker,
        mut predicate: impl FnMut(&WorkerEvent) -> bool,
    ) -> Vec<WorkerEvent> {
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut events = Vec::new();
        while Instant::now() < deadline {
            if let Some(event) = worker.recv_timeout(Duration::from_millis(50)) {
                let done = predicate(&event);
                events.push(event);
                if done {
                    return events;
                }
            }
        }
        panic!("timed out waiting for worker event");
    }

    fn wait_ready(worker: &Worker, engine: &str) {
        wait_until(worker, |event| {
            matches!(
                event,
                WorkerEvent::EngineReady {
                    engine: ready,
                    ..
                } if ready == engine
            )
        });
    }

    fn job_config(settings: Settings, intent: RecordingIntent) -> JobConfig {
        JobConfig::new(settings, intent, None, false)
    }

    #[test]
    fn stale_engine_load_is_never_published() {
        let (mut worker, _, _, _) = worker(&[], Some("slow"));
        worker.load("slow".into()).unwrap();
        thread::sleep(Duration::from_millis(5));
        worker.load("current".into()).unwrap();

        let events = wait_until(&worker, |event| {
            matches!(
                event,
                WorkerEvent::EngineReady { engine, .. } if engine == "current"
            )
        });

        assert!(!events.iter().any(|event| {
            matches!(
                event,
                WorkerEvent::EngineReady { engine, .. } if engine == "slow"
            )
        }));
    }

    #[test]
    fn load_failure_never_substitutes_another_engine() {
        let (mut worker, _, _, _) = worker(&["broken"], None);
        worker.load("broken".into()).unwrap();

        let events = wait_until(
            &worker,
            |event| matches!(event, WorkerEvent::EngineFailed { engine, .. } if engine == "broken"),
        );

        assert!(!events
            .iter()
            .any(|event| matches!(event, WorkerEvent::EngineReady { .. })));
    }

    #[test]
    fn jobs_run_fifo_without_concurrent_engine_calls() {
        let (mut worker, calls, max_active, _) = worker(&[], None);
        worker.load("ready".into()).unwrap();
        wait_ready(&worker, "ready");
        let settings = Settings::default();
        let first = worker
            .enqueue(
                vec![1.0],
                16_000,
                job_config(settings.clone(), RecordingIntent::Raw),
            )
            .unwrap();
        let second = worker
            .enqueue(
                vec![2.0],
                16_000,
                job_config(settings, RecordingIntent::Raw),
            )
            .unwrap();

        let events = wait_until(
            &worker,
            |event| matches!(event, WorkerEvent::JobCompleted { id, .. } if *id == second),
        );
        let completed = events
            .iter()
            .filter_map(|event| match event {
                WorkerEvent::JobCompleted { id, .. } => Some(*id),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(completed, vec![first, second]);
        assert_eq!(&*calls.lock(), &[1, 2]);
        assert_eq!(max_active.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn delivery_target_metadata_is_carried_to_completion() {
        let (mut worker, _, _, _) = worker(&[], None);
        worker.load("ready".into()).unwrap();
        wait_ready(&worker, "ready");
        let target = DeliveryTarget {
            hwnd: 17,
            process_id: 23,
            process_creation_time_100ns: Some(42),
        };
        let job = worker
            .enqueue(
                vec![1.0],
                16_000,
                JobConfig::new(
                    Settings::default(),
                    RecordingIntent::Raw,
                    Some(target),
                    false,
                ),
            )
            .unwrap();

        let events = wait_until(
            &worker,
            |event| matches!(event, WorkerEvent::JobCompleted { id, .. } if *id == job),
        );
        let completed_target = events.into_iter().find_map(|event| match event {
            WorkerEvent::JobCompleted {
                delivery_target, ..
            } => Some(delivery_target),
            _ => None,
        });

        assert_eq!(completed_target, Some(Some(target)));
    }

    #[test]
    fn job_snapshot_drives_provider_model_endpoint_and_prompt_resolution() {
        let mut live = Settings::default();
        live.enhance.mode = crate::settings::EnhanceMode::Always;
        live.enhance.provider = "openai_compatible".into();
        live.enhance.base_url = "https://snapshot.example/v1/".into();
        live.enhance.model = "private-deployment-42".into();
        let target = live.prompt_target().unwrap();
        live.prompt_overrides
            .set(target, "Use the frozen prompt.".into());

        let config = JobConfig::new(live.clone(), RecordingIntent::Enhance, None, false);
        live.enhance.provider = "github_copilot".into();
        live.enhance.base_url = "https://changed.example/v1".into();
        live.enhance.model = "changed-model".into();

        let prepared =
            crate::enhance::prepare_cleanup_request("raw transcript", &config.settings).unwrap();
        assert_eq!(prepared.url, "https://snapshot.example/v1/chat/completions");
        assert_eq!(prepared.body["model"], "private-deployment-42");
        let messages = prepared.body["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(
            messages[0]["content"],
            format!(
                "Use the frozen prompt.\n\n{}",
                crate::cleanup::prompt::TRANSCRIPT_ONLY_POLICY
            )
        );
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "raw transcript");
    }

    #[test]
    fn enhancement_failure_returns_the_raw_transcript_with_a_warning() {
        let (mut worker, _, _, _) = worker(&[], None);
        worker.load("ready".into()).unwrap();
        wait_ready(&worker, "ready");
        let mut settings = Settings::default();
        settings.enhance.mode = crate::settings::EnhanceMode::Always;
        settings.enhance.provider = "invalid-provider".into();
        let job = worker
            .enqueue(
                vec![7.0],
                16_000,
                job_config(settings, RecordingIntent::Enhance),
            )
            .unwrap();

        let events = wait_until(
            &worker,
            |event| matches!(event, WorkerEvent::JobCompleted { id, .. } if *id == job),
        );
        let completed = events.into_iter().find_map(|event| match event {
            WorkerEvent::JobCompleted {
                text,
                enhancement_warning,
                ..
            } => Some((text, enhancement_warning)),
            _ => None,
        });

        let (text, warning) = completed.unwrap();
        assert_eq!(text, "7");
        assert_eq!(warning, Some(FallbackReason::UnknownProvider));
    }

    #[test]
    fn validator_rejection_is_a_typed_raw_fallback_for_the_worker() {
        let outcome =
            crate::cleanup::pipeline::finalize("transfer 42 dollars", "Transfer 24 dollars.");
        let (text, warning) = completed_cleanup(outcome);
        assert_eq!(text, "transfer 42 dollars");
        assert!(matches!(
            warning,
            Some(FallbackReason::IntegrityRejected(_))
        ));
    }

    #[test]
    fn enhancement_phase_is_published_before_completion() {
        let (mut worker, _, _, _) = worker(&[], None);
        worker.load("ready".into()).unwrap();
        wait_ready(&worker, "ready");
        let mut settings = Settings::default();
        settings.enhance.mode = crate::settings::EnhanceMode::Always;
        settings.enhance.provider = "invalid-provider".into();
        let job = worker
            .enqueue(
                vec![7.0],
                16_000,
                job_config(settings, RecordingIntent::Enhance),
            )
            .unwrap();

        let events = wait_until(
            &worker,
            |event| matches!(event, WorkerEvent::JobCompleted { id, .. } if *id == job),
        );
        let phases = events
            .iter()
            .filter_map(|event| match event {
                WorkerEvent::JobStarted { id } if *id == job => Some("transcribing"),
                WorkerEvent::EnhancementStarted { id } if *id == job => Some("enhancing"),
                WorkerEvent::JobCompleted { id, .. } if *id == job => Some("completed"),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(phases, ["transcribing", "enhancing", "completed"]);
    }

    #[test]
    fn job_tombstones_suppress_late_and_backward_events() {
        let mut tracker = JobEventTracker::default();
        tracker.register(41);

        assert!(matches!(
            tracker.normalize(WorkerEvent::JobStarted { id: 41 }, false),
            Some(WorkerEvent::JobStarted { id: 41 })
        ));
        assert!(matches!(
            tracker.normalize(WorkerEvent::EnhancementStarted { id: 41 }, false),
            Some(WorkerEvent::EnhancementStarted { id: 41 })
        ));
        assert!(matches!(
            tracker.normalize(
                WorkerEvent::JobCompleted {
                    id: 41,
                    text: "done".into(),
                    diarization: DiarizationOutcome::NotRequested,
                    notes: None,
                    auto_paste: true,
                    enhancement_warning: None,
                    delivery_target: None,
                },
                false,
            ),
            Some(WorkerEvent::JobCompleted { id: 41, .. })
        ));
        assert!(tracker
            .normalize(WorkerEvent::EnhancementStarted { id: 41 }, false)
            .is_none());
        assert!(tracker
            .normalize(
                WorkerEvent::JobCompleted {
                    id: 41,
                    text: "late".into(),
                    diarization: DiarizationOutcome::NotRequested,
                    notes: None,
                    auto_paste: true,
                    enhancement_warning: None,
                    delivery_target: None,
                },
                false,
            )
            .is_none());
    }

    fn turn(start: usize, end: usize, speaker: usize) -> asr::diarize::Turn {
        asr::diarize::Turn {
            start,
            end,
            speaker,
        }
    }

    #[test]
    fn turn_windows_are_padded_but_never_overlap_neighbours() {
        let samples = vec![0.5f32; 16_000 * 10];
        let windows = plan_turn_windows(
            &[turn(16_000, 32_000, 0), turn(32_500, 48_000, 1)],
            &samples,
            16_000,
        );
        assert_eq!(windows.len(), 2);
        // 0.1 s (1600 samples) of padding; the 500-sample gap between the
        // turns is split down the middle.
        assert_eq!(windows[0].range, 14_400..32_250);
        assert_eq!(windows[1].range, 32_250..49_600);
        assert!(windows[0].range.end <= windows[1].range.start);
        assert_eq!((windows[0].speaker, windows[1].speaker), (0, 1));
    }

    #[test]
    fn long_turns_are_split_for_the_engine() {
        let samples = vec![0.5f32; 16_000 * 90];
        let windows = plan_turn_windows(&[turn(0, 16_000 * 80, 2)], &samples, 16_000);
        assert!(windows.len() >= 3);
        assert!(windows.iter().all(|w| w.range.len() <= 16_000 * 30));
        assert!(windows.iter().all(|w| w.speaker == 2));
    }

    struct EchoEngine;

    impl Engine for EchoEngine {
        fn transcribe(&mut self, samples: &[f32], _sample_rate: u32) -> Result<String> {
            // The loudest sample encodes which turn this is (windows include
            // a little silence of padding).
            let peak = samples.iter().copied().fold(0.0f32, f32::max);
            Ok(format!("w{}", peak as i32))
        }

        fn label(&self) -> &'static str {
            "echo"
        }
    }

    #[test]
    fn turn_transcripts_are_labelled_and_merge_same_speaker_runs() {
        let mut samples = vec![0.0f32; 16_000 * 6];
        // Turns: spk0 [0,1s), spk0 [1.2s,2s), spk1 [3s,4s), spk0 [5s,6s)
        for (range, value) in [
            (0..16_000, 1.0),
            (19_200..32_000, 2.0),
            (48_000..64_000, 3.0),
            (80_000..96_000, 4.0),
        ] {
            for s in &mut samples[range] {
                *s = value;
            }
        }
        let turns = [
            turn(0, 16_000, 0),
            turn(19_200, 32_000, 0),
            turn(48_000, 64_000, 1),
            turn(80_000, 96_000, 0),
        ];
        let mut seen = Vec::new();
        let text = transcribe_turns(
            &mut EchoEngine,
            &samples,
            16_000,
            &turns,
            || false,
            |done, total| seen.push((done, total)),
        )
        .unwrap();
        assert_eq!(text, "Speaker 1: w1 w2\n\nSpeaker 2: w3\n\nSpeaker 1: w4");
        assert_eq!(seen.first(), Some(&(0, 4)));
        assert_eq!(seen.last(), Some(&(4, 4)));
    }

    #[test]
    fn long_form_jobs_report_ordered_progress_before_completion() {
        let (mut worker, _, _, _) = worker(&[], None);
        worker.load("ready".into()).unwrap();
        wait_ready(&worker, "ready");
        let mut settings = Settings::default();
        settings.enhance.mode = crate::settings::EnhanceMode::Never;
        let job = worker
            .enqueue(
                vec![1.0; 65 * 16_000],
                16_000,
                JobConfig::long_form(settings),
            )
            .unwrap();

        let events = wait_until(
            &worker,
            |event| matches!(event, WorkerEvent::JobCompleted { id, .. } if *id == job),
        );
        let progress = events
            .iter()
            .filter_map(|event| match event {
                WorkerEvent::JobProgress {
                    id,
                    stage: JobStage::Transcribing,
                    done,
                    total,
                } if *id == job => Some((*done, *total)),
                _ => None,
            })
            .collect::<Vec<_>>();

        let total = progress.first().expect("progress was reported").1;
        assert!(total >= 2);
        let expected = (0..=total).map(|done| (done, total)).collect::<Vec<_>>();
        assert_eq!(progress, expected);
    }

    #[test]
    fn dictation_jobs_never_report_progress() {
        let (mut worker, _, _, _) = worker(&[], None);
        worker.load("ready".into()).unwrap();
        wait_ready(&worker, "ready");
        let job = worker
            .enqueue(
                vec![1.0],
                16_000,
                job_config(Settings::default(), RecordingIntent::Raw),
            )
            .unwrap();
        let events = wait_until(
            &worker,
            |event| matches!(event, WorkerEvent::JobCompleted { id, .. } if *id == job),
        );
        assert!(!events
            .iter()
            .any(|event| matches!(event, WorkerEvent::JobProgress { .. })));
    }

    #[test]
    fn wait_shutdown_drains_jobs_before_stopping() {
        let (mut worker, calls, _, _) = worker(&[], None);
        worker.load("ready".into()).unwrap();
        wait_ready(&worker, "ready");
        let settings = Settings::default();
        worker
            .enqueue(
                vec![1.0],
                16_000,
                job_config(settings.clone(), RecordingIntent::Raw),
            )
            .unwrap();
        worker
            .enqueue(
                vec![2.0],
                16_000,
                job_config(settings, RecordingIntent::Raw),
            )
            .unwrap();
        worker.shutdown(ShutdownMode::Wait).unwrap();

        let events = wait_until(&worker, |event| {
            matches!(event, WorkerEvent::ShutdownComplete)
        });
        let completed = events
            .iter()
            .filter(|event| matches!(event, WorkerEvent::JobCompleted { .. }))
            .count();

        assert_eq!(completed, 2);
        assert_eq!(&*calls.lock(), &[1, 2]);
    }

    #[test]
    fn wait_shutdown_skips_a_queued_engine_load() {
        let (mut worker, calls, _, loads) = worker(&[], None);
        worker.load("ready".into()).unwrap();
        wait_ready(&worker, "ready");
        let job = worker
            .enqueue(
                vec![1.0],
                16_000,
                job_config(Settings::default(), RecordingIntent::Raw),
            )
            .unwrap();
        wait_until(
            &worker,
            |event| matches!(event, WorkerEvent::JobStarted { id } if *id == job),
        );
        worker.load("unused".into()).unwrap();
        worker.shutdown(ShutdownMode::Wait).unwrap();

        let events = wait_until(&worker, |event| {
            matches!(event, WorkerEvent::ShutdownComplete)
        });

        assert!(events
            .iter()
            .any(|event| matches!(event, WorkerEvent::JobCompleted { id, .. } if *id == job)));
        assert_eq!(&*calls.lock(), &[1]);
        assert_eq!(&*loads.lock(), &["ready"]);
    }

    #[test]
    fn wait_shutdown_finishes_a_job_that_depends_on_an_active_load() {
        let (mut worker, calls, _, loads) = worker(&[], Some("required"));
        worker.load("ready".into()).unwrap();
        wait_ready(&worker, "ready");
        worker.load("required".into()).unwrap();
        let job = worker
            .enqueue(
                vec![2.0],
                16_000,
                job_config(Settings::default(), RecordingIntent::Raw),
            )
            .unwrap();
        worker.shutdown(ShutdownMode::Wait).unwrap();

        let events = wait_until(&worker, |event| {
            matches!(event, WorkerEvent::ShutdownComplete)
        });

        assert!(events
            .iter()
            .any(|event| matches!(event, WorkerEvent::JobCompleted { id, .. } if *id == job)));
        assert_eq!(&*calls.lock(), &[2]);
        assert_eq!(&*loads.lock(), &["ready", "required"]);
    }

    #[test]
    fn newer_engine_request_preserves_an_older_load_needed_by_a_job() {
        let (mut worker, calls, _, loads) = worker(&[], Some("required"));
        worker.load("ready".into()).unwrap();
        wait_ready(&worker, "ready");
        worker.load("required".into()).unwrap();
        let job = worker
            .enqueue(
                vec![2.0],
                16_000,
                job_config(Settings::default(), RecordingIntent::Raw),
            )
            .unwrap();
        worker.load("latest".into()).unwrap();

        let events = wait_until(
            &worker,
            |event| matches!(event, WorkerEvent::EngineReady { engine, .. } if engine == "latest"),
        );

        assert!(events
            .iter()
            .any(|event| matches!(event, WorkerEvent::JobCompleted { id, .. } if *id == job)));
        assert_eq!(&*calls.lock(), &[2]);
        assert_eq!(&*loads.lock(), &["ready", "required", "latest"]);
    }

    #[test]
    fn cancelled_load_is_not_revived_by_a_pending_job() {
        let (mut worker, _, _, _) = worker(&[], Some("cancelled"));
        worker.load("ready".into()).unwrap();
        wait_ready(&worker, "ready");
        worker.load("cancelled".into()).unwrap();
        let job = worker
            .enqueue(
                vec![2.0],
                16_000,
                job_config(Settings::default(), RecordingIntent::Raw),
            )
            .unwrap();
        worker.cancel_load();

        let events = wait_until(
            &worker,
            |event| matches!(event, WorkerEvent::JobFailed { id, .. } if *id == job),
        );

        assert!(!events
            .iter()
            .any(|event| matches!(event, WorkerEvent::JobCompleted { id, .. } if *id == job)));
    }

    #[test]
    fn cancelling_an_active_engine_suppresses_job_completion() {
        let (mut worker, _, _, _) = worker(&[], None);
        worker.load("ready".into()).unwrap();
        wait_ready(&worker, "ready");
        let job = worker
            .enqueue(
                vec![2.0],
                16_000,
                job_config(Settings::default(), RecordingIntent::Raw),
            )
            .unwrap();
        wait_until(
            &worker,
            |event| matches!(event, WorkerEvent::JobStarted { id } if *id == job),
        );
        worker.cancel_load();

        let events = wait_until(
            &worker,
            |event| matches!(event, WorkerEvent::JobFailed { id, .. } if *id == job),
        );

        assert!(!events
            .iter()
            .any(|event| matches!(event, WorkerEvent::JobCompleted { id, .. } if *id == job)));
    }

    #[test]
    fn discard_suppresses_a_completion_already_waiting_in_the_event_queue() {
        let (mut worker, _, _, _) = worker(&[], None);
        worker.load("ready".into()).unwrap();
        wait_ready(&worker, "ready");
        let job = worker
            .enqueue(
                vec![2.0],
                16_000,
                job_config(Settings::default(), RecordingIntent::Raw),
            )
            .unwrap();
        thread::sleep(Duration::from_millis(150));
        worker.shutdown(ShutdownMode::Discard).unwrap();

        let events = wait_until(&worker, |event| {
            matches!(event, WorkerEvent::ShutdownComplete)
        });

        assert!(events
            .iter()
            .any(|event| matches!(event, WorkerEvent::JobDiscarded { id } if *id == job)));
        assert!(!events
            .iter()
            .any(|event| matches!(event, WorkerEvent::JobCompleted { id, .. } if *id == job)));
    }

    #[test]
    fn discard_shutdown_suppresses_results_and_skips_queued_jobs() {
        let (mut worker, _, _, _) = worker(&[], None);
        worker.load("ready".into()).unwrap();
        wait_ready(&worker, "ready");
        let settings = Settings::default();
        let first = worker
            .enqueue(
                vec![1.0],
                16_000,
                job_config(settings.clone(), RecordingIntent::Raw),
            )
            .unwrap();
        let second = worker
            .enqueue(
                vec![2.0],
                16_000,
                job_config(settings, RecordingIntent::Raw),
            )
            .unwrap();
        wait_until(
            &worker,
            |event| matches!(event, WorkerEvent::JobStarted { id } if *id == first),
        );
        worker.shutdown(ShutdownMode::Discard).unwrap();

        let events = wait_until(&worker, |event| {
            matches!(event, WorkerEvent::ShutdownComplete)
        });
        let discarded = events
            .iter()
            .filter_map(|event| match event {
                WorkerEvent::JobDiscarded { id } => Some(*id),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert!(discarded.contains(&first));
        assert!(discarded.contains(&second));
        assert!(!events
            .iter()
            .any(|event| matches!(event, WorkerEvent::JobCompleted { .. })));
    }

    #[test]
    fn long_form_windows_cover_the_audio_and_cut_at_quiet_points() {
        let rate = 16_000;
        // 70 s of loud audio with a silent gap at 28.5 s (inside the first
        // window's search span) and another at 57 s.
        let mut samples = vec![0.5f32; 70 * rate];
        for sample in &mut samples[28 * rate + 8_000..28 * rate + 12_000] {
            *sample = 0.0;
        }
        for sample in &mut samples[57 * rate..57 * rate + 4_000] {
            *sample = 0.0;
        }
        let windows = plan_long_form_windows(&samples, rate as u32);
        assert_eq!(windows.first().unwrap().start, 0);
        assert_eq!(windows.last().unwrap().end, samples.len());
        for pair in windows.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
        }
        assert!(windows.iter().all(|w| w.len() <= 30 * rate));
        let first_cut = windows[0].end;
        assert!((28 * rate + 8_000..28 * rate + 12_000).contains(&first_cut));
    }

    #[test]
    fn short_audio_is_a_single_long_form_window() {
        assert_eq!(
            plan_long_form_windows(&vec![0.1; 16_000 * 10], 16_000).len(),
            1
        );
        assert!(plan_long_form_windows(&[], 16_000).is_empty());
    }
}
