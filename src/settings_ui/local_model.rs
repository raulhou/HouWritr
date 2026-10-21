//! "Local model" choice for AI cleanup: a model that runs on this PC.
//!
//! Two sources:
//! * **Built-in** – HouWritr manages Qualcomm's GenieX tool and a model on
//!   the NPU or the CPU (see [`crate::local_llm`]): install the tool, pick
//!   and download a model, done.
//! * **Another local server** – any OpenAI-compatible server on this PC
//!   (GenieX's own, LM Studio, Ollama, Foundry Local), with model discovery,
//!   a connection test and a setup guide.
//!
//! Both end up as the OpenAI-compatible provider pointing at a loopback URL,
//! so prompts, validation and fallbacks are exactly the same as for a hosted
//! model. This file only holds the Settings-window state and widgets.

use super::*;
use crate::cleanup::EndpointScope;
use crate::local_llm::{self, DEFAULT_MODEL, MANAGED_BASE_URL};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Server addresses offered as presets for "Another local server".
const SERVER_PRESETS: &[(&str, &str)] = &[
    ("GenieX server (Qualcomm NPU)", "http://127.0.0.1:18181/v1"),
    ("LM Studio", "http://localhost:1234/v1"),
    ("Ollama (CPU / GPU only)", "http://localhost:11434/v1"),
];

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum LocalSource {
    BuiltIn,
    Server,
}

struct LocalStatus {
    cli: Option<PathBuf>,
    model_installed: Option<bool>,
    error: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum JobKind {
    Check,
    Install,
    Pull,
    Remove,
    Uninstall,
    Test,
    Discover,
}

impl JobKind {
    fn name(self) -> &'static str {
        match self {
            JobKind::Check => "check",
            JobKind::Install => "install GenieX",
            JobKind::Pull => "download model",
            JobKind::Remove => "remove model",
            JobKind::Uninstall => "uninstall GenieX",
            JobKind::Test => "test model",
            JobKind::Discover => "find models",
        }
    }
}

enum JobEvent {
    Progress(Option<f32>, String),
    Status(LocalStatus),
    Models(Vec<String>),
    Done(std::result::Result<String, String>),
}

struct LocalJob {
    kind: JobKind,
    events: Receiver<JobEvent>,
    cancel: Arc<AtomicBool>,
    fraction: Option<f32>,
    text: String,
}

pub(super) struct LocalModelUi {
    /// The "Local model" provider choice is selected (as opposed to a hosted
    /// OpenAI-compatible API). Kept as state so typing `localhost` into the
    /// hosted API's URL box does not flip the page.
    pub(super) local_selected: bool,
    source: LocalSource,
    /// What the hosted provider had before "Local model" was chosen.
    cloud_base_url: String,
    cloud_model: String,
    status: Option<LocalStatus>,
    job: Option<LocalJob>,
    outcome: Option<std::result::Result<String, String>>,
    discovered: Vec<String>,
    /// "Uninstall GenieX" was clicked once; the next click confirms.
    confirm_uninstall: bool,
}

impl LocalModelUi {
    pub(super) fn new(enhance: &Enhance) -> Self {
        let scope = EndpointScope::parse(&enhance.base_url).ok();
        let loopback = scope.as_ref().is_some_and(EndpointScope::is_loopback);
        let local_selected = enhance.provider == "openai_compatible" && loopback;
        let defaults = Enhance::default();
        Self {
            local_selected,
            source: if scope.as_ref().is_some_and(local_llm::is_managed) || !local_selected {
                LocalSource::BuiltIn
            } else {
                LocalSource::Server
            },
            cloud_base_url: if local_selected {
                defaults.base_url
            } else {
                enhance.base_url.clone()
            },
            cloud_model: if local_selected {
                defaults.model
            } else {
                enhance.model.clone()
            },
            status: None,
            job: None,
            outcome: None,
            discovered: Vec::new(),
            confirm_uninstall: false,
        }
    }
}

fn is_loopback_url(raw: &str) -> bool {
    EndpointScope::parse(raw).is_ok_and(|scope| scope.is_loopback())
}

impl SettingsApp {
    /// The provider radio group: Copilot, a hosted API, or a model on this PC.
    pub(super) fn render_provider_choice(&mut self, ui: &mut egui::Ui) {
        let is_copilot = self.settings.enhance.provider == "github_copilot";
        let local = self.local_model.local_selected && !is_copilot;
        ui.horizontal_wrapped(|ui| {
            if ui.radio(is_copilot, "GitHub Copilot").clicked() && !is_copilot {
                self.leave_local_model();
                self.settings.enhance.provider = "github_copilot".into();
                self.api_key_input.clear();
                self.remove_api_key = false;
                self.dirty = true;
            }
            if ui
                .radio(!is_copilot && !local, "OpenAI-compatible API")
                .clicked()
                && (is_copilot || local)
            {
                self.leave_local_model();
                self.settings.enhance.provider = "openai_compatible".into();
                self.dirty = true;
            }
            if ui
                .radio(local, "Local model (on this PC)")
                .on_hover_text("Runs on this PC, ideally on the NPU. Text never leaves the device.")
                .clicked()
                && !local
            {
                self.enter_local_model();
            }
        });
    }

    fn enter_local_model(&mut self) {
        let enhance = &mut self.settings.enhance;
        if !is_loopback_url(&enhance.base_url) {
            self.local_model.cloud_base_url = enhance.base_url.clone();
            self.local_model.cloud_model = enhance.model.clone();
            enhance.base_url = MANAGED_BASE_URL.into();
            enhance.model = DEFAULT_MODEL.into();
            self.local_model.source = LocalSource::BuiltIn;
        }
        enhance.provider = "openai_compatible".into();
        self.local_model.local_selected = true;
        self.api_key_input.clear();
        self.remove_api_key = false;
        self.dirty = true;
    }

    /// Put the hosted provider's address and model back.
    fn leave_local_model(&mut self) {
        if !self.local_model.local_selected {
            return;
        }
        self.local_model.local_selected = false;
        let enhance = &mut self.settings.enhance;
        if is_loopback_url(&enhance.base_url) {
            enhance.base_url = self.local_model.cloud_base_url.clone();
            enhance.model = self.local_model.cloud_model.clone();
        }
    }

    pub(super) fn render_local_model(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Source");
            let before = self.local_model.source;
            ui.radio_value(
                &mut self.local_model.source,
                LocalSource::BuiltIn,
                "Built-in model",
            );
            ui.radio_value(
                &mut self.local_model.source,
                LocalSource::Server,
                "Another local server",
            );
            if before != self.local_model.source {
                self.local_model.outcome = None;
                let enhance = &mut self.settings.enhance;
                match self.local_model.source {
                    LocalSource::BuiltIn => {
                        enhance.base_url = MANAGED_BASE_URL.into();
                        if local_llm::built_in(&enhance.model).is_none() {
                            enhance.model = DEFAULT_MODEL.into();
                        }
                    }
                    LocalSource::Server => {
                        if local_llm::is_managed_url(&enhance.base_url) {
                            enhance.base_url = SERVER_PRESETS[0].1.into();
                        }
                    }
                }
                self.dirty = true;
            }
        });
        ui.add_space(2.0);
        status_label(
            ui,
            "Transcript text is processed on this PC and never sent over the internet.",
            secondary_text(),
        );
        ui.add_space(6.0);
        match self.local_model.source {
            LocalSource::BuiltIn => self.render_builtin_model(ui),
            LocalSource::Server => self.render_local_server(ui),
        }
    }

    fn render_builtin_model(&mut self, ui: &mut egui::Ui) {
        if let Err(reason) = local_llm::platform_support() {
            message_frame(ui, &reason, warning_color());
            return;
        }
        if self.local_model.status.is_none() && self.local_model.job.is_none() {
            self.start_local_job(JobKind::Check);
        }
        let busy = self.local_model.job.is_some();

        // The tool.
        let cli = self
            .local_model
            .status
            .as_ref()
            .and_then(|status| status.cli.clone());
        ui.horizontal_wrapped(|ui| {
            ui.label("GenieX tool");
            match (&self.local_model.status, &cli) {
                (None, _) => status_label(ui, "Checking…", secondary_text()),
                (Some(_), Some(path)) => status_label(
                    ui,
                    &format!("Installed ({})", path.display()),
                    status_green(),
                ),
                (Some(_), None) => status_label(ui, "Not installed", warning_color()),
            }
        });
        if self.local_model.status.is_some() && cli.is_none() {
            status_label(
                ui,
                "Qualcomm's GenieX runs the model on the NPU or the CPU. Installing downloads the official v0.8.0 \
                 installer (about 62 MB) from GitHub, checks its SHA-256, and installs it for your Windows \
                 user only (no administrator rights; adds `geniex` to your PATH).",
                secondary_text(),
            );
            if ui
                .add_enabled(!busy, egui::Button::new("Install GenieX…"))
                .clicked()
            {
                self.start_local_job(JobKind::Install);
            }
        }

        // The model.
        if cli.is_some() {
            ui.add_space(4.0);
            let selected = match local_llm::built_in(&self.settings.enhance.model) {
                Some(model) => *model,
                None => {
                    // Settings from before the choice existed, or edited by hand.
                    self.settings.enhance.model = DEFAULT_MODEL.into();
                    self.dirty = true;
                    local_llm::BUILT_IN_MODELS[0]
                }
            };
            ui.horizontal_wrapped(|ui| {
                ui.label("Model");
                egui::ComboBox::from_id_salt("built_in_model")
                    .selected_text(selected.label)
                    .show_ui(ui, |ui| {
                        for model in local_llm::BUILT_IN_MODELS {
                            if ui
                                .add_enabled(
                                    !busy,
                                    egui::SelectableLabel::new(
                                        model.id == selected.id,
                                        model.label,
                                    ),
                                )
                                .clicked()
                                && model.id != selected.id
                            {
                                self.settings.enhance.model = model.id.into();
                                self.local_model.status = None;
                                self.local_model.outcome = None;
                                self.dirty = true;
                            }
                        }
                    });
            });
            status_label(ui, selected.detail, secondary_text());
            let installed = self
                .local_model
                .status
                .as_ref()
                .and_then(|status| status.model_installed);
            ui.horizontal_wrapped(|ui| {
                ui.label("Download");
                ui.monospace(selected.name);
                match installed {
                    Some(true) => status_label(ui, "Downloaded", status_green()),
                    Some(false) => status_label(ui, "Not downloaded", warning_color()),
                    None => status_label(ui, "Unknown", secondary_text()),
                }
            });
            if let Some(error) = self
                .local_model
                .status
                .as_ref()
                .and_then(|status| status.error.as_ref())
            {
                inline_error(ui, error.clone());
            }
            ui.horizontal_wrapped(|ui| {
                if installed != Some(true)
                    && ui
                        .add_enabled(!busy, egui::Button::new("Download model"))
                        .on_hover_text("Several GB. Cancelling keeps what was downloaded.")
                        .clicked()
                {
                    self.start_local_job(JobKind::Pull);
                }
                if installed == Some(true) {
                    if ui
                        .add_enabled(!busy, egui::Button::new("Test model"))
                        .on_hover_text(
                            "Loads the model and asks it one question; shows how fast it writes.",
                        )
                        .clicked()
                    {
                        self.start_local_job(JobKind::Test);
                    }
                    if ui
                        .add_enabled(!busy, egui::Button::new("Remove model"))
                        .clicked()
                    {
                        self.start_local_job(JobKind::Remove);
                    }
                }
                if ui
                    .add_enabled(!busy, egui::Button::new("Check again"))
                    .clicked()
                {
                    self.local_model.status = None;
                    self.local_model.outcome = None;
                }
            });
            ui.horizontal_wrapped(|ui| {
                if self.local_model.confirm_uninstall {
                    ui.label("Remove the model and uninstall GenieX?");
                    if ui
                        .add_enabled(!busy, egui::Button::new("Uninstall"))
                        .clicked()
                    {
                        self.local_model.confirm_uninstall = false;
                        self.start_local_job(JobKind::Uninstall);
                    }
                    if ui.button("Keep").clicked() {
                        self.local_model.confirm_uninstall = false;
                    }
                } else if ui
                    .add_enabled(!busy, egui::Button::new("Uninstall GenieX…"))
                    .on_hover_text(
                        "Removes the model and runs GenieX's own uninstaller for your user. \
                         Cleanup with the built-in model stops working until it is installed again.",
                    )
                    .clicked()
                {
                    self.local_model.confirm_uninstall = true;
                }
            });
        }
        self.render_local_job_state(ui);
        ui.add_space(4.0);
        status_label(
            ui,
            "Built-in models are smaller than hosted ones: cleanup of long recordings takes longer, \
             and a cleaned section that changes a number or a speaker label is discarded: the original text stays.",
            secondary_text(),
        );
    }

    fn render_local_server(&mut self, ui: &mut egui::Ui) {
        let busy = self.local_model.job.is_some();
        let current_label = SERVER_PRESETS
            .iter()
            .find(|(_, url)| *url == self.settings.enhance.base_url.trim_end_matches('/'))
            .map_or("Custom", |(label, _)| *label);
        ui.horizontal_wrapped(|ui| {
            ui.label("Server");
            egui::ComboBox::from_id_salt("local_server_preset")
                .selected_text(current_label)
                .show_ui(ui, |ui| {
                    for (label, url) in SERVER_PRESETS {
                        if ui
                            .selectable_label(current_label == *label, *label)
                            .clicked()
                        {
                            self.settings.enhance.base_url = (*url).into();
                            self.local_model.discovered.clear();
                            self.dirty = true;
                        }
                    }
                });
        });
        ui.add_space(4.0);
        ui.label("Base URL");
        self.dirty |= ui
            .add(
                egui::TextEdit::singleline(&mut self.settings.enhance.base_url)
                    .hint_text("http://127.0.0.1:18181/v1"),
            )
            .changed();
        if !self.settings.enhance.base_url.trim().is_empty()
            && !is_loopback_url(&self.settings.enhance.base_url)
        {
            inline_error(
                ui,
                "A local model needs an address on this PC (localhost or 127.0.0.1). \
                 Use \"OpenAI-compatible API\" for a server elsewhere.",
            );
        }
        ui.add_space(4.0);
        ui.label("Model ID");
        self.dirty |= ui
            .add(
                egui::TextEdit::singleline(&mut self.settings.enhance.model)
                    .hint_text("As the server names it, e.g. ai-hub-models/Qwen3-4B-Instruct-2507"),
            )
            .changed();
        ui.horizontal_wrapped(|ui| {
            let ready = is_loopback_url(&self.settings.enhance.base_url);
            if ui
                .add_enabled(ready && !busy, egui::Button::new("Find models"))
                .on_hover_text("Asks the server which models it offers.")
                .clicked()
            {
                self.start_local_job(JobKind::Discover);
            }
            if ui
                .add_enabled(ready && !busy, egui::Button::new("Test connection"))
                .on_hover_text("Sends one short question to the selected model.")
                .clicked()
            {
                self.start_local_job(JobKind::Test);
            }
        });
        let mut picked = None;
        for id in &self.local_model.discovered {
            if ui
                .selectable_label(*id == self.settings.enhance.model, id)
                .clicked()
            {
                picked = Some(id.clone());
            }
        }
        if let Some(id) = picked {
            self.settings.enhance.model = id;
            self.dirty = true;
        }
        self.render_local_job_state(ui);

        if !self.migration_blocked {
            ui.add_space(4.0);
            egui::CollapsingHeader::new("API key (only if the server asks for one)")
                .id_salt("local_server_key")
                .show(ui, |ui| self.render_api_key_controls(ui));
        }
        ui.add_space(4.0);
        egui::CollapsingHeader::new("Setup guide")
            .id_salt("local_server_guide")
            .show(ui, |ui| render_setup_guide(ui));
    }

    fn render_local_job_state(&mut self, ui: &mut egui::Ui) {
        let mut cancel = false;
        if let Some(job) = &self.local_model.job {
            ui.add_space(4.0);
            // One bar that says it all (percentage, size, speed, time left);
            // a spinner and the text while the amount is unknown.
            match job.fraction {
                Some(fraction) => {
                    let text = if job.text.is_empty() {
                        format!("{:.0}%", fraction * 100.0)
                    } else {
                        job.text.clone()
                    };
                    ui.add(egui::ProgressBar::new(fraction).text(text));
                }
                None => {
                    ui.horizontal_wrapped(|ui| {
                        ui.spinner();
                        if !job.text.is_empty() {
                            status_label(ui, &job.text, secondary_text());
                        }
                    });
                }
            }
            if matches!(job.kind, JobKind::Install | JobKind::Pull)
                && ui.small_button("Cancel").clicked()
            {
                cancel = true;
            }
        }
        if cancel {
            if let Some(job) = &self.local_model.job {
                job.cancel.store(true, Ordering::Relaxed);
            }
        }
        match &self.local_model.outcome {
            Some(Ok(message)) => status_label(ui, message, status_green()),
            Some(Err(message)) => inline_error(ui, message.clone()),
            None => {}
        }
    }

    fn start_local_job(&mut self, kind: JobKind) {
        if self.local_model.job.is_some() {
            return;
        }
        let (sender, events) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let token = cancel.clone();
        let base_url = self.settings.enhance.base_url.clone();
        let model = self.settings.enhance.model.clone();
        let spawned = std::thread::Builder::new()
            .name("local-model".into())
            .spawn(move || {
                let result = run_local_job(kind, &base_url, &model, &sender, &token);
                match &result {
                    Ok(message) => {
                        tracing::info!(job = kind.name(), result = %message, "local model action finished")
                    }
                    Err(message) => {
                        tracing::warn!(job = kind.name(), error = %message, "local model action failed")
                    }
                }
                let _ = sender.send(JobEvent::Done(result));
            });
        match spawned {
            Ok(_) => {
                // The automatic re-check after a download must not wipe the
                // message that download just produced.
                if kind != JobKind::Check {
                    self.local_model.outcome = None;
                }
                self.local_model.job = Some(LocalJob {
                    kind,
                    events,
                    cancel,
                    fraction: None,
                    text: match kind {
                        JobKind::Check => String::new(),
                        JobKind::Install => "Starting download…".into(),
                        JobKind::Pull => "Starting download…".into(),
                        JobKind::Remove => "Removing…".into(),
                        JobKind::Uninstall => "Uninstalling GenieX…".into(),
                        JobKind::Test => {
                            "Waiting for the model (the first answer loads it)…".into()
                        }
                        JobKind::Discover => "Asking the server…".into(),
                    },
                });
            }
            Err(error) => {
                self.local_model.outcome =
                    Some(Err(format!("Could not start the background task: {error}")));
            }
        }
    }

    pub(super) fn poll_local_model(&mut self, ctx: &egui::Context) {
        let Some(job) = self.local_model.job.as_mut() else {
            return;
        };
        let mut finished = None;
        loop {
            match job.events.try_recv() {
                Ok(JobEvent::Progress(fraction, text)) => {
                    job.fraction = fraction.or(job.fraction);
                    job.text = text;
                }
                Ok(JobEvent::Status(status)) => self.local_model.status = Some(status),
                Ok(JobEvent::Models(models)) => self.local_model.discovered = models,
                Ok(JobEvent::Done(result)) => {
                    finished = Some((job.kind, result));
                    break;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    finished = Some((
                        job.kind,
                        Err("the background task stopped unexpectedly".into()),
                    ));
                    break;
                }
            }
        }
        match finished {
            Some((kind, result)) => {
                self.local_model.job = None;
                self.local_model.outcome = match result {
                    Ok(message) if message.is_empty() => self.local_model.outcome.take(),
                    Ok(message) => Some(Ok(message)),
                    Err(message) if message.contains("cancelled") => Some(Err("Cancelled.".into())),
                    Err(message) => Some(Err(message)),
                };
                // Anything that changed what is installed is re-checked.
                if matches!(
                    kind,
                    JobKind::Install | JobKind::Pull | JobKind::Remove | JobKind::Uninstall
                ) {
                    self.local_model.status = None;
                }
            }
            None => ctx.request_repaint_after(std::time::Duration::from_millis(150)),
        }
    }
}

fn run_local_job(
    kind: JobKind,
    base_url: &str,
    model: &str,
    sender: &mpsc::Sender<JobEvent>,
    cancel: &AtomicBool,
) -> std::result::Result<String, String> {
    let fail = |error: anyhow::Error| format!("{error:#}");
    match kind {
        JobKind::Check => {
            let cli = local_llm::locate_cli();
            let (model_installed, error) = match &cli {
                Some(path) => match local_llm::model_installed(path, local_llm::geniex_name(model))
                {
                    Ok(installed) => (Some(installed), None),
                    Err(error) => (None, Some(format!("Could not list models: {error:#}"))),
                },
                None => (None, None),
            };
            let _ = sender.send(JobEvent::Status(LocalStatus {
                cli,
                model_installed,
                error,
            }));
            Ok(String::new())
        }
        JobKind::Install => {
            let path = local_llm::install_cli(
                &mut |fraction| {
                    let _ = sender.send(JobEvent::Progress(
                        Some(fraction),
                        "Downloading the GenieX installer…".into(),
                    ));
                },
                cancel,
            )
            .map_err(fail)?;
            Ok(format!("GenieX installed ({}).", path.display()))
        }
        JobKind::Pull => {
            let cli = local_llm::locate_cli().ok_or("The GenieX tool is not installed.")?;
            local_llm::pull_model(
                &cli,
                model,
                &mut |fraction, line| {
                    let _ = sender.send(JobEvent::Progress(fraction, line.to_string()));
                },
                cancel,
            )
            .map_err(fail)?;
            Ok("Model downloaded.".into())
        }
        JobKind::Remove => {
            let cli = local_llm::locate_cli().ok_or("The GenieX tool is not installed.")?;
            local_llm::shutdown();
            local_llm::remove_model(&cli, local_llm::geniex_name(model)).map_err(fail)?;
            Ok("Model removed.".into())
        }
        JobKind::Uninstall => {
            let cli = local_llm::locate_cli().ok_or("The GenieX tool is not installed.")?;
            local_llm::uninstall_cli(&cli).map_err(fail)?;
            Ok("GenieX and its model were removed.".into())
        }
        JobKind::Test => {
            let result = local_llm::test_chat(base_url, model);
            // A server HouWritr started for the test stops right away.
            local_llm::release("the test is finished");
            Ok(result.map_err(fail)?.summary())
        }
        JobKind::Discover => {
            let models = local_llm::discover_models(base_url);
            local_llm::release("the models were listed");
            let models = models.map_err(fail)?;
            let message = if models.is_empty() {
                "The server answered but lists no models.".to_string()
            } else {
                format!("{} model(s) found. Click one to use it.", models.len())
            };
            let _ = sender.send(JobEvent::Models(models));
            Ok(message)
        }
    }
}

fn render_setup_guide(ui: &mut egui::Ui) {
    fn command(ui: &mut egui::Ui, text: &str) {
        ui.horizontal_wrapped(|ui| {
            ui.monospace(text);
            if ui.small_button("Copy").clicked() {
                ui.ctx().copy_text(text.to_string());
            }
        });
    }
    ui.label(egui::RichText::new("GenieX server (Snapdragon NPU)").strong());
    status_label(
        ui,
        "1. Install the GenieX command-line tool from github.com/qualcomm/GenieX/releases, then open a new terminal.",
        secondary_text(),
    );
    command(ui, "geniex pull ai-hub-models/Qwen3-4B-Instruct-2507");
    command(ui, "geniex serve");
    status_label(
        ui,
        "2. Choose \"GenieX server\" above, press Find models, then Test connection.",
        secondary_text(),
    );
    ui.add_space(6.0);
    ui.label(egui::RichText::new("LM Studio").strong());
    status_label(
        ui,
        "Load a model, open the Developer tab and start the server (default port 1234). Pick \"LM Studio\" above.",
        secondary_text(),
    );
    ui.add_space(6.0);
    ui.label(egui::RichText::new("Ollama (no NPU support)").strong());
    command(ui, "ollama pull qwen3:4b-instruct");
    status_label(
        ui,
        "Ollama serves on port 11434 by itself. Pick \"Ollama\" above.",
        secondary_text(),
    );
    ui.add_space(6.0);
    ui.label(egui::RichText::new("Foundry Local").strong());
    command(ui, "foundry service status");
    status_label(
        ui,
        "Copy the address it prints into Base URL and add /v1 if it is missing.",
        secondary_text(),
    );
    ui.add_space(6.0);
    ui.horizontal_wrapped(|ui| {
        if ui.link("GenieX documentation").clicked() {
            let _ = about::open_url("https://geniex.aihub.qualcomm.com");
        }
    });
}
