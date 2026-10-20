//! egui settings dialog launched as a subprocess.

use crate::about;
use crate::autostart::{self, AutostartBackend, AutostartState};
use crate::cleanup::{catalog, PromptSource, PromptTarget};
use crate::credentials::{store_verified, CredentialStore, WindowsCredentialStore};
use crate::diagnostics;
use crate::enhance::{self, CopilotReadiness};
use crate::model_manager::{
    CancellationToken, ModelCacheState, ModelDetails, ModelInfo, ModelManager, ModelState,
};
use crate::paths::settings_path;
use crate::settings::{Enhance, EnhanceMode, MeetingNotesMode, Settings, SettingsRevision};
use crate::single_instance::SettingsTransaction;
use anyhow::{anyhow, Result};
use eframe::egui;
use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, TryRecvError};

mod local_model;
use local_model::LocalModelUi;

pub fn run_dialog(show_about: bool) -> Result<()> {
    let (settings, initial_error, migration_blocked, load_failed, settings_revision) =
        match Settings::load_runtime() {
            Ok(loaded) => {
                let blocked = loaded.credential_health.requires_user_resolution;
                let load_failed = loaded.settings_error.is_some();
                let initial_error = [loaded.settings_error, loaded.credential_health.message]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .join("\n");
                (
                    loaded.settings,
                    (!initial_error.is_empty()).then_some(initial_error),
                    blocked,
                    load_failed,
                    Some(loaded.revision),
                )
            }
            Err(error) => {
                let revision = Settings::revision().ok();
                (
                Settings::default(),
                Some(format!(
                    "Settings could not be loaded. Defaults are shown, but replacing the existing file requires explicit confirmation: {error}"
                )),
                false,
                true,
                revision,
            )
            }
        };
    let credential_status = read_credential_status(&WindowsCredentialStore);
    let engine_states = load_engine_states();
    let copilot_rx = start_copilot_probe();
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(scaled_size([640.0, 760.0]))
            .with_min_inner_size(scaled_size([600.0, 640.0]))
            .with_resizable(true)
            .with_title("HouWritr Settings"),
        ..Default::default()
    };
    eframe::run_native(
        "HouWritr Settings",
        opts,
        Box::new(move |cc| {
            apply_dark_theme(&cc.egui_ctx);
            Ok(Box::new(SettingsApp {
                local_model: LocalModelUi::new(&settings.enhance),
                settings,
                dirty: false,
                api_key_input: String::new(),
                remove_api_key: false,
                migration_blocked,
                load_failed,
                replace_load_failed: false,
                settings_revision,
                error: initial_error,
                credential_status,
                engine_states,
                copilot_readiness: None,
                copilot_rx: Some(copilot_rx),
                copilot_models: None,
                copilot_models_rx: None,
                confirm_discard: false,
                prompt_editor: PromptEditorState::default(),
                show_about,
                about_error: None,
                autostart: AutostartUi::new(),
                speaker_model: SpeakerModelUi::load(),
                confirm_remove: None,
                removal_notice: None,
            }))
        }),
    )
    .map_err(|error| anyhow!("egui run failed: {error}"))
}

/// Windows fonts that cover Chinese, best first. egui's built-in fonts have
/// no CJK glyphs, so without one of these Chinese text shows as boxes.
const CJK_FONT_FILES: &[&str] = &["msyh.ttc", "msjh.ttc", "simsun.ttc", "Deng.ttf"];

fn cjk_font_candidates(windows_dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    CJK_FONT_FILES
        .iter()
        .map(|file| windows_dir.join("Fonts").join(file))
        .collect()
}

/// Add the first installed Chinese-capable system font behind egui's own
/// fonts, so Latin text looks the same and Chinese falls through to it.
fn install_cjk_fallback_font(ctx: &egui::Context) {
    let windows_dir = std::env::var_os("WINDIR")
        .or_else(|| std::env::var_os("SystemRoot"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(r"C:\Windows"));
    let Some(bytes) = cjk_font_candidates(&windows_dir)
        .into_iter()
        .find_map(|path| std::fs::read(path).ok())
    else {
        return;
    };
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "system-cjk".into(),
        egui::FontData::from_owned(bytes).into(),
    );
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .push("system-cjk".into());
    }
    ctx.set_fonts(fonts);
}

/// Windows' own "Text size" setting (Settings → Accessibility → Text size)
/// as a factor, 1.0 to 2.25. The display scale (Settings → Display → Scale)
/// is applied by the window library already; this one is not.
pub(crate) fn system_text_scale() -> f32 {
    use windows::core::w;
    use windows::Win32::System::Registry::{RegGetValueW, HKEY_CURRENT_USER, RRF_RT_REG_DWORD};

    let mut value = 0_u32;
    let mut size = std::mem::size_of::<u32>() as u32;
    let read = unsafe {
        RegGetValueW(
            HKEY_CURRENT_USER,
            w!("Software\\Microsoft\\Accessibility"),
            w!("TextScaleFactor"),
            RRF_RT_REG_DWORD,
            None,
            Some((&mut value as *mut u32).cast()),
            Some(&mut size),
        )
    };
    text_scale_from(read.is_ok().then_some(value))
}

/// The percentage Windows stores (100 to 225), as a factor; 1.0 when unset
/// or out of range.
fn text_scale_from(percent: Option<u32>) -> f32 {
    match percent {
        Some(percent @ 100..=225) => percent as f32 / 100.0,
        _ => 1.0,
    }
}

/// A window size in points, grown with the Windows text size so the larger
/// text still fits.
pub(crate) fn scaled_size(size: [f32; 2]) -> [f32; 2] {
    let scale = system_text_scale();
    [size[0] * scale, size[1] * scale]
}

pub(crate) fn apply_dark_theme(ctx: &egui::Context) {
    // Follow Windows' text size: everything grows with it, as in Windows'
    // own windows.
    ctx.set_zoom_factor(system_text_scale());
    use egui::{Color32, FontFamily, FontId, Stroke, Style, Visuals};
    install_cjk_fallback_font(ctx);
    let mut style = Style::default();
    style.visuals = Visuals::dark();
    style.visuals.window_fill = Color32::from_rgb(20, 23, 31);
    style.visuals.panel_fill = Color32::from_rgb(20, 23, 31);
    style.visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(27, 31, 40);
    style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(42, 47, 58);
    style.visuals.widgets.active.bg_fill = Color32::from_rgb(79, 140, 255);
    style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(52, 58, 73);
    style.visuals.widgets.noninteractive.fg_stroke =
        Stroke::new(1.0, Color32::from_rgb(232, 236, 243));
    style.visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, Color32::from_rgb(232, 236, 243));
    style.visuals.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    style.visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::from_rgb(232, 236, 243));
    style.visuals.window_stroke = Stroke::new(1.0, Color32::from_rgb(54, 61, 76));
    style.visuals.window_rounding = 8.0.into();
    style.interaction.tooltip_delay = 0.0;
    style.interaction.show_tooltips_only_when_still = false;
    style.spacing.tooltip_width = 360.0;
    style.text_styles.insert(
        egui::TextStyle::Heading,
        FontId::new(22.0, FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Body,
        FontId::new(13.0, FontFamily::Proportional),
    );
    ctx.set_style(style);
}

struct SettingsApp {
    settings: Settings,
    dirty: bool,
    api_key_input: String,
    remove_api_key: bool,
    migration_blocked: bool,
    load_failed: bool,
    replace_load_failed: bool,
    settings_revision: Option<SettingsRevision>,
    error: Option<String>,
    credential_status: CredentialStatus,
    engine_states: HashMap<String, EngineUiState>,
    copilot_readiness: Option<CopilotReadiness>,
    copilot_rx: Option<Receiver<CopilotReadiness>>,
    /// The models Copilot offers this account, once asked (see
    /// [`enhance::copilot_offered_models`]).
    copilot_models: Option<Result<Vec<enhance::CopilotModel>, String>>,
    copilot_models_rx: Option<Receiver<Result<Vec<enhance::CopilotModel>, String>>>,
    confirm_discard: bool,
    prompt_editor: PromptEditorState,
    show_about: bool,
    about_error: Option<String>,
    autostart: AutostartUi,
    speaker_model: SpeakerModelUi,
    local_model: LocalModelUi,
    /// A model whose "Remove" was clicked once; the next click confirms.
    confirm_remove: Option<String>,
    /// What the last model removal did, and for which model.
    removal_notice: Option<(String, std::result::Result<String, String>)>,
}

/// Id of the speaker-label models in `models-manifest.json`.
const SPEAKER_MODEL_ID: &str = "diarization";

/// Progress messages from a background download / verification.
enum ModelJobEvent {
    State(ModelState),
    Done(std::result::Result<(), String>),
}

/// A running download or verification of the speaker-label models.
struct ModelJob {
    events: Receiver<ModelJobEvent>,
    cancel: CancellationToken,
    state: Option<ModelState>,
    verify_only: bool,
}

/// What the "Speaker labels" section shows about the local models.
struct SpeakerModelUi {
    details: std::result::Result<ModelDetails, String>,
    job: Option<ModelJob>,
    /// Result of the last download/verify: `Ok(message)` or `Err(message)`.
    outcome: Option<std::result::Result<String, String>>,
    link_error: Option<String>,
}

impl SpeakerModelUi {
    fn load() -> Self {
        Self {
            details: ModelManager::describe(SPEAKER_MODEL_ID).map_err(|error| error.to_string()),
            job: None,
            outcome: None,
            link_error: None,
        }
    }

    fn refresh(&mut self) {
        self.details = ModelManager::describe(SPEAKER_MODEL_ID).map_err(|error| error.to_string());
    }
}

/// OS autostart state, kept separate from `Settings`: it is machine/OS state,
/// not settings-file state. It never participates in `persist()`, the dirty
/// flag, or `SettingsRevision`; it writes through to the OS immediately on
/// click and is verified by a read-back. This also stops a copied
/// `settings.json` from silently arming autostart on another machine.
struct AutostartUi {
    backend: Box<dyn AutostartBackend>,
    state: Result<AutostartState, String>,
    error: Option<String>,
}

impl AutostartUi {
    fn new() -> Self {
        let backend = autostart::backend();
        let state = backend.state().map_err(|error| error.to_string());
        Self {
            backend,
            state,
            error: None,
        }
    }

    /// Re-read the real OS state so external changes (Task Manager, Settings →
    /// Apps → Startup) are shown truthfully.
    fn refresh(&mut self) {
        self.state = self.backend.state().map_err(|error| error.to_string());
    }

    fn set_enabled(&mut self, enabled: bool) {
        let result = if enabled {
            autostart::enable_verified(self.backend.as_ref())
        } else {
            autostart::disable_verified(self.backend.as_ref())
        };
        match result {
            Ok(()) => self.error = None,
            Err(error) => self.error = Some(error.to_string()),
        }
        // Always re-read the truth after a write, success or failure.
        self.refresh();
    }

    #[cfg(test)]
    fn from_backend(backend: Box<dyn AutostartBackend>) -> Self {
        let state = backend.state().map_err(|error| error.to_string());
        Self {
            backend,
            state,
            error: None,
        }
    }
}

/// Ephemeral editor state. Draft text is never written until the outer
/// Settings Save button commits the whole settings document.
#[derive(Default)]
struct PromptEditorState {
    draft: Option<PromptDraft>,
    pending_target_switch: bool,
    reset_target: Option<PromptTarget>,
}

struct PromptDraft {
    target: PromptTarget,
    text: String,
    original_target_fields: PromptTargetFields,
}

struct PromptTargetFields {
    provider: String,
    base_url: String,
    model: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PromptTargetSwitchDecision {
    Save,
    Discard,
    Cancel,
}

impl PromptEditorState {
    fn begin_edit(&mut self, target: PromptTarget, prompt: String, enhance: &Enhance) {
        self.draft = Some(PromptDraft {
            target,
            text: prompt,
            original_target_fields: PromptTargetFields {
                provider: enhance.provider.clone(),
                base_url: enhance.base_url.clone(),
                model: enhance.model.clone(),
            },
        });
        self.pending_target_switch = false;
    }

    fn cancel_edit(&mut self) {
        self.draft = None;
        self.pending_target_switch = false;
    }

    fn request_target_switch(&mut self, target: &PromptTarget) -> bool {
        let needs_decision = self
            .draft
            .as_ref()
            .is_some_and(|draft| draft.target != *target);
        self.pending_target_switch = needs_decision;
        needs_decision
    }

    fn resolve_target_switch(
        &mut self,
        settings: &mut Settings,
        decision: PromptTargetSwitchDecision,
    ) -> bool {
        let Some(draft) = self.draft.take() else {
            self.pending_target_switch = false;
            return false;
        };
        let staged = match decision {
            PromptTargetSwitchDecision::Save => {
                settings.prompt_overrides.set(draft.target, draft.text);
                true
            }
            PromptTargetSwitchDecision::Discard => false,
            PromptTargetSwitchDecision::Cancel => {
                restore_prompt_target_fields(settings, &draft.original_target_fields);
                self.draft = Some(draft);
                false
            }
        };
        self.pending_target_switch = false;
        staged
    }

    fn has_open_draft(&self) -> bool {
        self.draft.is_some()
    }
}

fn restore_prompt_target_fields(settings: &mut Settings, fields: &PromptTargetFields) {
    settings.enhance.provider = fields.provider.clone();
    settings.enhance.base_url = fields.base_url.clone();
    settings.enhance.model = fields.model.clone();
}

impl SettingsApp {
    fn persist(&mut self) -> Result<()> {
        let settings = self.settings.clone();
        settings.validate()?;
        if self.load_failed && !self.replace_load_failed {
            return Err(anyhow!(
                "Confirm replacement of the invalid or unreadable settings file before saving."
            ));
        }

        let replacement = self.api_key_input.trim();
        if self.migration_blocked && replacement.is_empty() && !self.remove_api_key {
            return Err(anyhow!(
                "Enter the API key again to secure it, or explicitly select Remove the legacy API key. The plaintext key has not been removed."
            ));
        }
        let change = requested_credential_change(
            &settings,
            self.migration_blocked,
            self.remove_api_key,
            replacement,
        );
        let path = settings_path();
        let expected_revision = self.settings_revision.as_ref().ok_or_else(|| {
            anyhow!(
                "The current settings file revision could not be captured; it was not replaced."
            )
        })?;
        let _transaction = SettingsTransaction::acquire(&path)?;
        ensure_revision_unchanged(&path, expected_revision)?;
        persist_with_credential_change(&WindowsCredentialStore, change, || {
            ensure_revision_unchanged(&path, expected_revision)?;
            settings.save_to(&path).map_err(Into::into)
        })?;
        self.api_key_input.clear();
        self.remove_api_key = false;
        self.migration_blocked = false;
        self.load_failed = false;
        self.replace_load_failed = false;
        self.error = None;
        self.dirty = false;
        Ok(())
    }

    fn poll_copilot_probe(&mut self) {
        let Some(receiver) = &self.copilot_rx else {
            return;
        };
        match receiver.try_recv() {
            Ok(status) => {
                let ready = matches!(status, CopilotReadiness::Ready(_));
                self.copilot_readiness = Some(status);
                self.copilot_rx = None;
                if ready && self.copilot_models.is_none() {
                    self.refresh_copilot_models();
                }
            }
            Err(TryRecvError::Disconnected) => {
                self.copilot_readiness = Some(CopilotReadiness::Failed(
                    "GitHub Copilot sign-in check stopped unexpectedly".into(),
                ));
                self.copilot_rx = None;
            }
            Err(TryRecvError::Empty) => {}
        }
    }

    /// A "Remove" button for downloaded model `model_id` that asks once more
    /// before deleting. `allowed` says why removing is not possible now.
    fn render_remove_model(
        &mut self,
        ui: &mut egui::Ui,
        model_id: &str,
        label: &str,
        allowed: std::result::Result<(), &str>,
    ) {
        ui.horizontal_wrapped(|ui| {
            if self.confirm_remove.as_deref() == Some(model_id) {
                ui.label("Delete these files? They are downloaded again when next needed.");
                if ui.button("Delete").clicked() {
                    self.confirm_remove = None;
                    self.remove_model_files(model_id);
                }
                if ui.button("Keep").clicked() {
                    self.confirm_remove = None;
                }
            } else {
                let button = ui.add_enabled(allowed.is_ok(), egui::Button::new(label).small());
                let button = match allowed {
                    Err(reason) => button.on_disabled_hover_text(reason),
                    Ok(()) => button,
                };
                if button.clicked() {
                    self.confirm_remove = Some(model_id.to_string());
                }
            }
        });
    }

    /// Delete a downloaded model, unless the running app uses it.
    fn remove_model_files(&mut self, model_id: &str) {
        // The app runs the engine saved in the settings file, which may differ
        // from an unsaved choice in this window.
        let saved_engine = Settings::load().map(|saved| saved.engine).ok();
        let outcome = if saved_engine.as_deref() == Some(model_id) {
            Err("HouWritr is using this engine. Choose and save another engine first.".into())
        } else {
            match ModelManager::remove_installed(model_id) {
                Ok(()) => Ok("Model files removed.".into()),
                Err(error) => Err(format!(
                    "Could not remove the model files: {error}. If HouWritr is using them, \
                     wait for the current transcription to finish and try again."
                )),
            }
        };
        self.removal_notice = Some((model_id.to_string(), outcome));
        self.refresh_engine_states();
        self.speaker_model.refresh();
    }

    /// The last removal's result, under the section it belongs to: speaker
    /// labels (`speaker == true`) or the speech engines.
    fn render_removal_notice(&self, ui: &mut egui::Ui, speaker: bool) {
        let Some((model_id, outcome)) = &self.removal_notice else {
            return;
        };
        if (model_id == SPEAKER_MODEL_ID) != speaker {
            return;
        }
        match outcome {
            Ok(message) => status_label(ui, message, status_green()),
            Err(message) => inline_error(ui, message.clone()),
        }
    }

    /// Ask Copilot, on a background thread, which models this account has.
    fn refresh_copilot_models(&mut self) {
        let (sender, receiver) = mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name("settings-copilot-models".into())
            .spawn(move || {
                let _ = sender.send(enhance::copilot_offered_models());
            });
        if spawned.is_ok() {
            self.copilot_models_rx = Some(receiver);
        }
    }

    fn poll_copilot_models(&mut self) {
        let Some(receiver) = &self.copilot_models_rx else {
            return;
        };
        match receiver.try_recv() {
            Ok(models) => {
                self.copilot_models = Some(models);
                self.copilot_models_rx = None;
            }
            Err(TryRecvError::Disconnected) => {
                self.copilot_models = Some(Err("the model list could not be loaded".into()));
                self.copilot_models_rx = None;
            }
            Err(TryRecvError::Empty) => {}
        }
    }

    /// The models to offer for Copilot: the account's own list once loaded,
    /// the built-in names until then (or if it could not be loaded).
    fn copilot_model_choices(&self) -> Vec<(String, String)> {
        match &self.copilot_models {
            Some(Ok(models)) if !models.is_empty() => models
                .iter()
                .map(|model| {
                    let label = if model.name == model.id {
                        model.id.clone()
                    } else {
                        format!("{} ({})", model.name, model.id)
                    };
                    (model.id.clone(), label)
                })
                .collect(),
            _ => catalog::all()
                .iter()
                .map(|model| (model.id.to_string(), model.display_name.to_string()))
                .collect(),
        }
    }

    fn restart_copilot_probe(&mut self) {
        self.copilot_readiness = None;
        self.copilot_rx = Some(start_copilot_probe());
    }

    /// Start downloading (`verify_only == false`) or just verifying the
    /// speaker-label models on a background thread.
    fn start_speaker_model_job(&mut self, verify_only: bool) {
        if self.speaker_model.job.is_some() {
            return;
        }
        let (sender, events) = mpsc::channel();
        let cancel = CancellationToken::default();
        let token = cancel.clone();
        let spawned = std::thread::Builder::new()
            .name("speaker-model".into())
            .spawn(move || {
                let result = (|| -> std::result::Result<(), String> {
                    let manager = ModelManager::new_preserving_work_dirs()
                        .map_err(|error| error.to_string())?;
                    if verify_only {
                        let _ = sender.send(ModelJobEvent::State(ModelState::Verifying));
                        manager
                            .verify_existing(SPEAKER_MODEL_ID, &token)
                            .map_err(|error| error.to_string())
                    } else {
                        manager
                            .ensure(SPEAKER_MODEL_ID, &token, |state| {
                                let _ = sender.send(ModelJobEvent::State(state));
                            })
                            .map(|_| ())
                            .map_err(|error| error.to_string())
                    }
                })();
                let _ = sender.send(ModelJobEvent::Done(result));
            });
        match spawned {
            Ok(_) => {
                self.speaker_model.outcome = None;
                self.speaker_model.link_error = None;
                self.speaker_model.job = Some(ModelJob {
                    events,
                    cancel,
                    state: None,
                    verify_only,
                });
            }
            Err(error) => {
                self.speaker_model.outcome =
                    Some(Err(format!("Could not start the background task: {error}")));
            }
        }
    }

    fn poll_speaker_model_job(&mut self, ctx: &egui::Context) {
        let Some(job) = self.speaker_model.job.as_mut() else {
            return;
        };
        let mut finished = None;
        loop {
            match job.events.try_recv() {
                Ok(ModelJobEvent::State(state)) => job.state = Some(state),
                Ok(ModelJobEvent::Done(result)) => {
                    finished = Some((result, job.verify_only));
                    break;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    finished = Some((
                        Err("the background task stopped unexpectedly".into()),
                        job.verify_only,
                    ));
                    break;
                }
            }
        }
        match finished {
            Some((result, verify_only)) => {
                self.speaker_model.job = None;
                self.speaker_model.outcome = Some(match result {
                    Ok(()) if verify_only => {
                        Ok("Files verified: sizes and SHA-256 hashes match.".into())
                    }
                    Ok(()) => Ok("Speaker label models are installed and verified.".into()),
                    Err(message) if message.contains("cancelled") => {
                        Err("Download cancelled. Nothing was installed.".into())
                    }
                    Err(message) => Err(message),
                });
                self.speaker_model.refresh();
            }
            None => ctx.request_repaint_after(std::time::Duration::from_millis(150)),
        }
    }

    fn refresh_engine_states(&mut self) {
        self.engine_states = load_engine_states();
    }

    fn selected_engine_error(&self) -> Option<String> {
        let state = self.engine_states.get(&self.settings.engine)?;
        match &state.support {
            Ok(support) if support.is_supported() => None,
            Ok(support) => Some(
                support
                    .reason()
                    .unwrap_or("The selected engine is unavailable.")
                    .to_string(),
            ),
            Err(error) => Some(format!("Could not determine engine support: {error}")),
        }
    }

    fn blocking_error(&self) -> Option<String> {
        if self.prompt_editor.has_open_draft() {
            return Some("Save or cancel the prompt editor draft before saving Settings.".into());
        }
        if let Err(error) = self.settings.validate() {
            return Some(error.to_string());
        }
        if let Some(error) = self.selected_engine_error() {
            return Some(error);
        }
        if self.load_failed && !self.replace_load_failed {
            return Some("Confirm replacement of the invalid or unreadable settings file.".into());
        }
        if self.migration_blocked && self.api_key_input.trim().is_empty() && !self.remove_api_key {
            return Some(
                "Re-enter the legacy API key to secure it, or choose to remove it.".into(),
            );
        }
        let changes_secure_credential = !self.api_key_input.trim().is_empty()
            || (self.remove_api_key && !self.migration_blocked);
        if changes_secure_credential && matches!(self.credential_status, CredentialStatus::Error(_))
        {
            return Some(
                "Windows Credential Manager must be available before changing the API key.".into(),
            );
        }
        if self.settings.enhance.mode.is_enabled() {
            match self.settings.enhance.provider.as_str() {
                "github_copilot" => match self.copilot_readiness.as_ref() {
                    Some(CopilotReadiness::Ready(_)) => {}
                    Some(CopilotReadiness::CliMissing) => {
                        return Some(enhance::CopilotAuthError::CliMissing.to_string())
                    }
                    Some(CopilotReadiness::NotAuthenticated) => {
                        return Some(enhance::CopilotAuthError::NotAuthenticated.to_string())
                    }
                    Some(CopilotReadiness::Failed(error)) => return Some(error.clone()),
                    None => return Some("Looking for your GitHub Copilot sign-in...".into()),
                },
                "openai_compatible" => {
                    if let CredentialStatus::Error(error) = &self.credential_status {
                        return Some(format!(
                            "Windows Credential Manager could not be read: {error}"
                        ));
                    }
                    if !has_openai_credential(
                        &self.credential_status,
                        self.migration_blocked,
                        self.remove_api_key,
                        &self.api_key_input,
                    ) {
                        return Some(
                            "Enter an API key or keep an existing saved credential.".into(),
                        );
                    }
                }
                _ => {}
            }
        }
        if self.settings_revision.is_none() {
            return Some("The settings file revision could not be captured.".into());
        }
        None
    }
}

#[derive(Clone, Debug)]
enum CredentialStatus {
    Saved,
    Missing,
    Error(String),
}

fn has_openai_credential(
    status: &CredentialStatus,
    migration_blocked: bool,
    remove_api_key: bool,
    replacement: &str,
) -> bool {
    let removes_secure_credential = remove_api_key && !migration_blocked;
    !removes_secure_credential
        && (!replacement.trim().is_empty() || matches!(status, CredentialStatus::Saved))
}

#[derive(Clone, Debug)]
struct EngineUiState {
    support: std::result::Result<crate::asr::EngineSupport, String>,
    model: std::result::Result<ModelInfo, String>,
}

fn read_credential_status(store: &dyn CredentialStore) -> CredentialStatus {
    match store.read() {
        Ok(Some(secret)) if !secret.trim().is_empty() => CredentialStatus::Saved,
        Ok(_) => CredentialStatus::Missing,
        Err(error) => CredentialStatus::Error(error.to_string()),
    }
}

fn load_engine_states() -> HashMap<String, EngineUiState> {
    ["parakeet_cpu", "parakeet_npu", "whisper_npu"]
        .into_iter()
        .map(|engine| {
            (
                engine.to_string(),
                EngineUiState {
                    support: crate::asr::engine_support(engine).map_err(|error| error.to_string()),
                    model: ModelManager::inspect(engine).map_err(|error| error.to_string()),
                },
            )
        })
        .collect()
}

fn start_copilot_probe() -> Receiver<CopilotReadiness> {
    let (sender, receiver) = mpsc::channel();
    let _ = std::thread::Builder::new()
        .name("settings-copilot-readiness".into())
        .spawn(move || {
            let _ = sender.send(enhance::github_copilot_readiness());
        });
    receiver
}

fn ensure_revision_unchanged(path: &std::path::Path, expected: &SettingsRevision) -> Result<()> {
    let current = Settings::revision_from(path)?;
    if &current != expected {
        return Err(anyhow!(
            "Settings changed after this window opened. Close and reopen Settings before saving."
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CredentialChange<'a> {
    Keep,
    Set(&'a str),
    Delete,
}

fn requested_credential_change<'a>(
    settings: &Settings,
    migration_blocked: bool,
    remove_api_key: bool,
    replacement: &'a str,
) -> CredentialChange<'a> {
    if remove_api_key {
        return if migration_blocked {
            CredentialChange::Keep
        } else {
            CredentialChange::Delete
        };
    }
    if !replacement.is_empty()
        && (migration_blocked
            || (settings.enhance.mode.is_enabled()
                && settings.enhance.provider == "openai_compatible"))
    {
        CredentialChange::Set(replacement)
    } else {
        CredentialChange::Keep
    }
}

fn persist_with_credential_change(
    store: &dyn CredentialStore,
    change: CredentialChange<'_>,
    commit_settings: impl FnOnce() -> Result<()>,
) -> Result<()> {
    if matches!(change, CredentialChange::Keep) {
        return commit_settings();
    }

    let previous = store.read()?;
    let apply_result = match change {
        CredentialChange::Keep => unreachable!(),
        CredentialChange::Set(secret) => store_verified(store, secret),
        CredentialChange::Delete => store.delete(),
    };
    if let Err(error) = apply_result {
        return Err(error_with_rollback(
            store,
            previous.as_deref(),
            format!("Credential Manager update failed: {error}"),
        ));
    }

    if let Err(error) = commit_settings() {
        return Err(error_with_rollback(
            store,
            previous.as_deref(),
            format!("Settings were not saved: {error}"),
        ));
    }
    Ok(())
}

fn error_with_rollback(
    store: &dyn CredentialStore,
    previous: Option<&str>,
    primary: String,
) -> anyhow::Error {
    let rollback = match previous {
        Some(secret) => store_verified(store, secret),
        None => store.delete(),
    };
    match rollback {
        Ok(()) => anyhow!("{primary}. The previous secure credential was restored."),
        Err(error) => {
            anyhow!("{primary}. Restoring the previous secure credential also failed: {error}")
        }
    }
}

impl SettingsApp {
    fn render_engines(&mut self, ui: &mut egui::Ui) {
        for descriptor in ENGINE_DESCRIPTORS {
            let state = self.engine_states.get(descriptor.id).cloned();
            let supported = state
                .as_ref()
                .and_then(|state| state.support.as_ref().ok())
                .map(|support| support.is_supported())
                .unwrap_or(false);
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.horizontal_wrapped(|ui| {
                    ui.add_enabled_ui(supported, |ui| {
                        if ui
                            .radio(self.settings.engine == descriptor.id, descriptor.title)
                            .clicked()
                        {
                            self.settings.engine = descriptor.id.to_string();
                            self.dirty = true;
                        }
                    });
                    info_button(ui, descriptor.description);
                });
                if let Some(state) = state {
                    match state.support {
                        Ok(support) if support.is_supported() => {
                            let detail = support
                                .detail()
                                .map(|detail| format!("Available: {detail}"))
                                .unwrap_or_else(|| "Available on this device".into());
                            status_label(ui, &detail, status_green());
                        }
                        Ok(support) => status_label(
                            ui,
                            &format!(
                                "Unavailable: {}",
                                support.reason().unwrap_or("unsupported hardware")
                            ),
                            warning_color(),
                        ),
                        Err(error) => inline_error(
                            ui,
                            format!("Could not determine hardware support: {error}"),
                        ),
                    }
                    match state.model {
                        Ok(info) => {
                            let cache = match info.cache_state {
                                ModelCacheState::Missing => "Not downloaded",
                                ModelCacheState::Installed => "Installed",
                                ModelCacheState::Incomplete => {
                                    "Incomplete; repaired on next engine load"
                                }
                            };
                            ui.label(
                                egui::RichText::new(format!(
                                    "{cache}. First-use download: {}.",
                                    format_bytes(info.download_bytes)
                                ))
                                .small()
                                .color(secondary_text()),
                            );
                            if info.cache_state != ModelCacheState::Missing {
                                let in_use = self.settings.engine == descriptor.id;
                                self.render_remove_model(
                                    ui,
                                    descriptor.id,
                                    "Remove model files",
                                    (!in_use).then_some(()).ok_or(
                                        "This engine is selected; choose and save another engine first.",
                                    ),
                                );
                            }
                        }
                        Err(error) => inline_error(
                            ui,
                            format!("Could not inspect the local model cache: {error}"),
                        ),
                    }
                }
            });
            ui.add_space(6.0);
        }
        if ui
            .small_button("Refresh availability and cache status")
            .clicked()
        {
            self.refresh_engine_states();
        }
        self.render_removal_notice(ui, false);
        if let Some(error) = self.selected_engine_error() {
            inline_error(ui, error);
        }
    }

    fn render_speaker_labels(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            self.dirty |= ui
                .checkbox(
                    &mut self.settings.diarize_speakers,
                    "Label speakers in file and meeting transcripts",
                )
                .changed();
            info_button(
                ui,
                "Adds \"Speaker 1:\", \"Speaker 2:\" … labels and a [m:ss] time marker per turn, using small on-device models that run on the CPU. Nothing is sent anywhere. Accuracy drops with overlapping talk or poor audio.",
            );
        });
        ui.add_space(6.0);

        let details = self.speaker_model.details.clone();
        let busy = self.speaker_model.job.is_some();
        let mut start_download = false;
        let mut start_verify = false;
        let mut refresh = false;
        let mut cancel = false;
        let mut open_folder = None;
        let mut open_link: Option<String> = None;

        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.label(egui::RichText::new("Local models").strong());
            match &details {
                Ok(details) => {
                    let (text, color) = match details.info.cache_state {
                        ModelCacheState::Installed => ("Installed", status_green()),
                        ModelCacheState::Missing => ("Not downloaded", warning_color()),
                        ModelCacheState::Incomplete => (
                            "Incomplete: files are missing or the wrong size",
                            warning_color(),
                        ),
                    };
                    status_label(ui, text, color);
                    ui.label(
                        egui::RichText::new(format!(
                            "pyannote segmentation 3.0 + WeSpeaker ResNet34 · {} · runs on the CPU",
                            format_bytes(details.info.download_bytes)
                        ))
                        .small()
                        .color(secondary_text()),
                    );
                    ui.label(
                        egui::RichText::new(format!(
                            "Location: {}",
                            details.directory.display()
                        ))
                        .small()
                        .color(secondary_text()),
                    );
                    if !self.settings.diarize_speakers
                        && details.info.cache_state != ModelCacheState::Installed
                    {
                        ui.label(
                            egui::RichText::new(
                                "Speaker labels are off, so nothing is downloaded until you turn them on or install the models here.",
                            )
                            .small()
                            .color(secondary_text()),
                        );
                    }
                }
                Err(error) => inline_error(
                    ui,
                    format!("Could not inspect the local models: {error}"),
                ),
            }

            if let Some(job) = &self.speaker_model.job {
                ui.add_space(6.0);
                match &job.state {
                    Some(ModelState::Downloading {
                        downloaded_bytes,
                        total_bytes,
                    }) => {
                        let fraction = if *total_bytes == 0 {
                            0.0
                        } else {
                            (*downloaded_bytes as f32 / *total_bytes as f32).clamp(0.0, 1.0)
                        };
                        ui.add(egui::ProgressBar::new(fraction).show_percentage());
                        status_label(
                            ui,
                            &format!(
                                "Downloading {} of {}",
                                format_bytes(*downloaded_bytes),
                                format_bytes(*total_bytes)
                            ),
                            secondary_text(),
                        );
                    }
                    Some(ModelState::Verifying) => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Verifying files…");
                        });
                    }
                    _ => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(if job.verify_only {
                                "Checking files…"
                            } else {
                                "Preparing download…"
                            });
                        });
                    }
                }
            }

            ui.add_space(6.0);
            let installed = matches!(
                &details,
                Ok(details) if details.info.cache_state == ModelCacheState::Installed
            );
            ui.horizontal_wrapped(|ui| {
                if ui
                    .add_enabled(!busy && !installed, egui::Button::new("Download and install"))
                    .clicked()
                {
                    start_download = true;
                }
                if busy && ui.button("Cancel").clicked() {
                    cancel = true;
                }
                if ui
                    .add_enabled(!busy, egui::Button::new("Verify files"))
                    .clicked()
                {
                    start_verify = true;
                }
                if ui.button("Open model folder").clicked() {
                    if let Ok(details) = &details {
                        open_folder = Some(details.directory.clone());
                    }
                }
                if ui
                    .add_enabled(!busy, egui::Button::new("Refresh status"))
                    .clicked()
                {
                    refresh = true;
                }
            });
            let present = matches!(
                &details,
                Ok(details) if details.info.cache_state != ModelCacheState::Missing
            );
            if present {
                self.render_remove_model(
                    ui,
                    SPEAKER_MODEL_ID,
                    "Remove models",
                    (!busy).then_some(()).ok_or("Wait for the download or check to finish."),
                );
            }
        });
        self.render_removal_notice(ui, true);

        match &self.speaker_model.outcome {
            Some(Ok(message)) => status_label(ui, message, status_green()),
            Some(Err(message)) => inline_error(ui, message.clone()),
            None => {}
        }
        if let Some(error) = &self.speaker_model.link_error {
            inline_error(ui, error.clone());
        }

        ui.add_space(6.0);
        egui::CollapsingHeader::new("Install manually (offline or restricted network)")
            .default_open(false)
            .show(ui, |ui| {
                ui.label(
                    "1. Download both files below in your browser.\n\
                     2. Save them with exactly these names into the model folder.\n\
                     3. Click \"Verify files\". HouWritr checks each file's size and SHA-256 and \
                        rejects anything that does not match.",
                );
                if let Ok(details) = &details {
                    ui.add_space(4.0);
                    for file in &details.files {
                        egui::Frame::group(ui.style()).show(ui, |ui| {
                            ui.set_min_width(ui.available_width());
                            ui.horizontal_wrapped(|ui| {
                                ui.label(egui::RichText::new(&file.path).monospace().strong());
                                if file.present {
                                    status_label(ui, "present", status_green());
                                } else {
                                    status_label(ui, "missing", warning_color());
                                }
                            });
                            ui.label(
                                egui::RichText::new(format!(
                                    "{} bytes · SHA-256 {}",
                                    file.bytes, file.sha256
                                ))
                                .small()
                                .color(secondary_text()),
                            );
                            if ui.small_button("Open download link").clicked() {
                                open_link = Some(file.url.clone());
                            }
                        });
                    }
                    ui.add_space(4.0);
                    if ui.button("Open model folder").clicked() {
                        open_folder = Some(details.directory.clone());
                    }
                }
            });

        if cancel {
            if let Some(job) = &self.speaker_model.job {
                job.cancel.cancel();
            }
        }
        if start_download {
            self.start_speaker_model_job(false);
        }
        if start_verify {
            self.start_speaker_model_job(true);
        }
        if refresh {
            self.speaker_model.refresh();
            self.speaker_model.outcome = None;
        }
        if let Some(directory) = open_folder {
            if let Err(error) = diagnostics::open_directory(&directory) {
                self.speaker_model.link_error =
                    Some(format!("Could not open the model folder: {error}"));
            }
        }
        if let Some(url) = open_link {
            if let Err(error) = about::open_url(&url) {
                self.speaker_model.link_error =
                    Some(format!("Could not open the download link: {error}"));
            }
        }
    }

    fn render_output(&mut self, ui: &mut egui::Ui) {
        self.dirty |= ui
            .checkbox(
                &mut self.settings.overlay,
                "Show transcription progress and status messages (progress is hidden while the recording indicator is hidden)",
            )
            .changed();
        self.dirty |= ui
            .checkbox(
                &mut self.settings.show_meeting_indicator,
                "Show a recording indicator during meeting recordings",
            )
            .changed();
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            let manual = !self.settings.power_saving_auto;
            self.dirty |= ui
                .add_enabled(
                    manual,
                    egui::Checkbox::new(
                        &mut self.settings.power_saving,
                        "Power saving: on battery only record; transcribe and clean up once plugged in",
                    ),
                )
                .changed();
            info_button(
                ui,
                "While this is on and the PC runs on its battery, meetings are only recorded and files are only queued. Everything waits, saved on disk, until the PC is plugged in, and is then done in order. Work that has already started finishes. Also in the tray menu.",
            );
        });
        self.dirty |= ui
            .checkbox(
                &mut self.settings.power_saving_auto,
                "Follow Windows: power saving is on exactly while Windows' power saving mode (battery saver) is on",
            )
            .changed();
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            ui.label("When a meeting recording stops, create AI meeting notes:");
            info_button(
                ui,
                "Notes (summary, decisions, action items) are written by your AI provider from the transcript text, so they need a provider set up under Text enhancement. The transcript itself is always saved.",
            );
        });
        ui.horizontal_wrapped(|ui| {
            for (mode, label) in [
                (MeetingNotesMode::Ask, "Ask every time"),
                (MeetingNotesMode::Always, "Always"),
                (MeetingNotesMode::Never, "Never"),
            ] {
                self.dirty |= ui
                    .radio_value(&mut self.settings.meeting_notes, mode, label)
                    .changed();
            }
        });
        ui.horizontal_wrapped(|ui| {
            self.dirty |= ui
                .checkbox(
                    &mut self.settings.markdown_transcripts,
                    "Save file and meeting transcripts as Markdown",
                )
                .changed();
            info_button(
                ui,
                "Imported-file and meeting transcripts are saved as .md with a title, details line and paragraphs instead of plain .txt. AI cleanup applies to both formats.",
            );
        });
    }

    fn render_enhancement(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Improve punctuation and wording of transcripts?");
            info_button(
                ui,
                "Enhancement sends recognized transcript text, never recorded audio, to the selected provider.",
            );
        });
        let before_mode = self.settings.enhance.mode;
        let mut enabled = before_mode.is_enabled();
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(&mut enabled, false, "Off");
            info_button(ui, "Transcript text never leaves this PC for cleanup.");
            ui.radio_value(&mut enabled, true, "On");
            info_button(
                ui,
                "Every file and meeting transcript is cleaned up after transcription.",
            );
        });
        if enabled != before_mode.is_enabled() {
            self.settings.enhance.mode = if enabled {
                EnhanceMode::Always
            } else {
                EnhanceMode::Never
            };
        }
        if before_mode != self.settings.enhance.mode {
            if self.settings.enhance.mode == EnhanceMode::Never {
                self.api_key_input.clear();
            }
            self.dirty = true;
        }
        if let Some(warning) = self.settings.prompt_override_warning() {
            ui.add_space(6.0);
            message_frame(ui, &warning, warning_color());
        }

        if self.migration_blocked {
            ui.add_space(8.0);
            self.render_legacy_credential_resolution(ui);
        }

        if self.settings.enhance.mode == EnhanceMode::Never {
            if !self.migration_blocked {
                self.render_disabled_credential_summary(ui);
            }
            self.render_prompt_editor(ui);
            return;
        }

        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            ui.label("Provider");
            info_button(
                ui,
                "Only recognized transcript text is sent to the provider. Recorded audio stays on this PC.",
            );
        });
        self.render_provider_choice(ui);

        ui.add_space(8.0);

        match self.settings.enhance.provider.as_str() {
            "github_copilot" => self.render_github_copilot(ui),
            "openai_compatible" if self.local_model.local_selected => self.render_local_model(ui),
            "openai_compatible" => self.render_openai_compatible(ui),
            _ => inline_error(ui, "Select a supported enhancement provider."),
        }
        self.render_prompt_editor(ui);

        if let Err(error) = self.settings.validate_enhancement() {
            inline_error(ui, error.to_string());
        }
    }

    fn render_github_copilot(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            match self.copilot_readiness.as_ref() {
                None => status_label(
                    ui,
                    "Looking for your GitHub Copilot sign-in...",
                    secondary_text(),
                ),
                Some(CopilotReadiness::Ready(source)) => status_label(
                    ui,
                    &format!("Signed in through {}.", source.describe()),
                    status_green(),
                ),
                Some(CopilotReadiness::CliMissing) => status_label(
                    ui,
                    &enhance::CopilotAuthError::CliMissing.to_string(),
                    error_color(),
                ),
                Some(CopilotReadiness::NotAuthenticated) => status_label(
                    ui,
                    &enhance::CopilotAuthError::NotAuthenticated.to_string(),
                    error_color(),
                ),
                Some(CopilotReadiness::Failed(error)) => status_label(ui, error, error_color()),
            }
            if !matches!(self.copilot_readiness, Some(CopilotReadiness::Ready(_)))
                && ui.small_button("Retry").clicked()
            {
                self.restart_copilot_probe();
            }
        });
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            ui.label("Model");
            let loading = self.copilot_models_rx.is_some();
            match &self.copilot_models {
                _ if loading => {
                    status_label(ui, "Loading your Copilot models...", secondary_text())
                }
                Some(Ok(models)) => status_label(
                    ui,
                    &format!("{} offered to your account", models.len()),
                    secondary_text(),
                ),
                Some(Err(error)) => status_label(ui, error, warning_color()),
                None => {}
            }
            let ready = matches!(self.copilot_readiness, Some(CopilotReadiness::Ready(_)));
            if ready && !loading && ui.small_button("Refresh").clicked() {
                self.refresh_copilot_models();
            }
        });
        let choices = self.copilot_model_choices();
        let mut selection = copilot_model_picker_value(&self.settings.enhance.model, &choices);
        let selected_label = choices
            .iter()
            .find(|(id, _)| *id == selection)
            .map(|(_, label)| label.clone())
            .unwrap_or_else(|| "Custom model ID".into());
        let before_selection = selection.clone();
        egui::ComboBox::from_id_salt("copilot_model")
            .selected_text(selected_label)
            .show_ui(ui, |ui| {
                for (id, label) in &choices {
                    ui.selectable_value(&mut selection, id.clone(), label);
                }
                ui.selectable_value(&mut selection, "custom".into(), "Custom model ID");
            });
        if selection != before_selection {
            if selection == "custom" {
                self.settings.enhance.model.clear();
            } else {
                self.settings.enhance.model = selection.clone();
            }
            self.dirty = true;
        }
        if selection == "custom" {
            self.dirty |= ui
                .add(
                    egui::TextEdit::singleline(&mut self.settings.enhance.model)
                        .hint_text("Model ID"),
                )
                .changed();
        }
        ui.horizontal_wrapped(|ui| {
            info_button(
                ui,
                "The list comes from GitHub Copilot for your account. On Copilot Free, GitHub offers models through automatic selection only, so a listed model can still be refused; HouWritr then uses one that answers and the log names it.",
            );
            ui.hyperlink_to(
                egui::RichText::new("View GitHub model documentation").small(),
                "https://docs.github.com/en/copilot/reference/ai-models/supported-models",
            );
        });
    }

    fn render_openai_compatible(&mut self, ui: &mut egui::Ui) {
        ui.label("Base URL");
        self.dirty |= ui
            .add(
                egui::TextEdit::singleline(&mut self.settings.enhance.base_url)
                    .hint_text("https://api.openai.com/v1"),
            )
            .changed();
        if insecure_remote_http(&self.settings.enhance.base_url) {
            status_label(
                ui,
                "This non-local endpoint uses unencrypted HTTP.",
                warning_color(),
            );
        }

        ui.add_space(4.0);
        ui.label("Model ID");
        self.dirty |= ui
            .add(
                egui::TextEdit::singleline(&mut self.settings.enhance.model)
                    .hint_text("Provider-specific model ID"),
            )
            .changed();

        if self.migration_blocked {
            return;
        }

        ui.add_space(6.0);
        self.render_api_key_controls(ui);
    }

    fn render_api_key_controls(&mut self, ui: &mut egui::Ui) {
        match &self.credential_status {
            CredentialStatus::Saved if !self.remove_api_key => status_label(
                ui,
                "API key saved in Windows Credential Manager.",
                status_green(),
            ),
            CredentialStatus::Saved => status_label(
                ui,
                "The saved API key will be removed when Settings are saved.",
                warning_color(),
            ),
            CredentialStatus::Missing => status_label(ui, "No API key is saved.", secondary_text()),
            CredentialStatus::Error(error) => inline_error(
                ui,
                format!("Windows Credential Manager could not be read: {error}"),
            ),
        }
        let hint = if matches!(self.credential_status, CredentialStatus::Saved) {
            "Enter a new key to replace the saved credential"
        } else {
            "Enter API key"
        };
        if ui
            .add(
                egui::TextEdit::singleline(&mut self.api_key_input)
                    .password(true)
                    .hint_text(hint),
            )
            .changed()
        {
            self.remove_api_key = false;
            self.dirty = true;
        }
        if matches!(self.credential_status, CredentialStatus::Saved) {
            let label = if self.remove_api_key {
                "Keep saved API key"
            } else {
                "Remove saved API key"
            };
            if ui.small_button(label).clicked() {
                self.remove_api_key = !self.remove_api_key;
                if self.remove_api_key {
                    self.api_key_input.clear();
                }
                self.dirty = true;
            }
        }
    }

    fn render_prompt_editor(&mut self, ui: &mut egui::Ui) {
        ui.add_space(10.0);
        ui.separator();
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("Cleanup prompt").strong());
            info_button(
                ui,
                "Prompts are tuned per provider, endpoint, and exact model or deployment ID.",
            );
        });

        if self.settings.prompt_overrides.preserves_unsupported_raw() {
            status_label(
                ui,
                "This prompt document is preserved for a newer format and cannot be edited here.",
                warning_color(),
            );
            return;
        }

        let target = match self.settings.prompt_target() {
            Ok(target) => target,
            Err(error) => {
                inline_error(ui, format!("Prompt target is incomplete: {error}"));
                return;
            }
        };
        self.prompt_editor.request_target_switch(&target);

        ui.add_space(4.0);
        ui.label(
            egui::RichText::new(format!("Target: {}", prompt_target_label(&target)))
                .small()
                .color(secondary_text()),
        );

        let resolved = self.settings.resolve_prompt(&target);
        let source = prompt_source_label(resolved.source);
        ui.label(
            egui::RichText::new(format!("Active prompt: {source}"))
                .small()
                .color(secondary_text()),
        );

        if self.prompt_editor.draft.is_some() {
            let pending_target_switch = self.prompt_editor.pending_target_switch;
            {
                let draft = self
                    .prompt_editor
                    .draft
                    .as_mut()
                    .expect("draft exists while editing");
                ui.add(
                    egui::TextEdit::multiline(&mut draft.text)
                        .desired_rows(8)
                        .hint_text("Write cleanup instructions"),
                );
            }
            let mut save_draft = false;
            let mut cancel_draft = false;
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(
                        !pending_target_switch,
                        egui::Button::new("Save prompt draft"),
                    )
                    .clicked()
                {
                    save_draft = true;
                }
                if ui
                    .add_enabled(
                        !pending_target_switch,
                        egui::Button::new("Cancel prompt draft"),
                    )
                    .clicked()
                {
                    cancel_draft = true;
                }
            });
            if save_draft {
                let draft = self
                    .prompt_editor
                    .draft
                    .take()
                    .expect("draft exists while saving");
                self.settings.prompt_overrides.set(draft.target, draft.text);
                self.prompt_editor.pending_target_switch = false;
                self.dirty = true;
            } else if cancel_draft {
                self.prompt_editor.cancel_edit();
            }
            if pending_target_switch {
                status_label(
                    ui,
                    "The target changed while this draft is open. Choose Save, Discard, or Cancel below.",
                    warning_color(),
                );
            }
        } else {
            let mut display = resolved.system;
            ui.add(
                egui::TextEdit::multiline(&mut display)
                    .desired_rows(8)
                    .interactive(false),
            );
            ui.horizontal(|ui| {
                if ui.button("Edit prompt").clicked() {
                    self.prompt_editor.begin_edit(
                        target.clone(),
                        display.clone(),
                        &self.settings.enhance,
                    );
                }
                if self.settings.prompt_overrides.get(&target).is_some()
                    && ui.small_button("Reset this target").clicked()
                {
                    self.prompt_editor.reset_target = Some(target.clone());
                }
            });
        }
    }

    fn render_legacy_credential_resolution(&mut self, ui: &mut egui::Ui) {
        message_frame(
            ui,
            "A legacy plaintext API key must be secured in Windows Credential Manager or removed before Settings can be saved.",
            warning_color(),
        );
        if ui
            .add(
                egui::TextEdit::singleline(&mut self.api_key_input)
                    .password(true)
                    .hint_text("Re-enter API key"),
            )
            .changed()
        {
            self.remove_api_key = false;
            self.dirty = true;
        }
        let label = if self.remove_api_key {
            "Keep the legacy API key"
        } else {
            "Remove the legacy API key"
        };
        if ui.small_button(label).clicked() {
            self.remove_api_key = !self.remove_api_key;
            if self.remove_api_key {
                self.api_key_input.clear();
            }
            self.dirty = true;
        }
    }

    fn render_disabled_credential_summary(&mut self, ui: &mut egui::Ui) {
        match &self.credential_status {
            CredentialStatus::Saved if self.remove_api_key => {
                status_label(
                    ui,
                    "The saved OpenAI-compatible API key will be removed when Settings are saved.",
                    warning_color(),
                );
                if ui.small_button("Keep saved API key").clicked() {
                    self.remove_api_key = false;
                    self.dirty = true;
                }
            }
            CredentialStatus::Saved => {
                status_label(
                    ui,
                    "An OpenAI-compatible API key is stored for later use.",
                    secondary_text(),
                );
                if ui.small_button("Remove saved API key").clicked() {
                    self.remove_api_key = true;
                    self.dirty = true;
                }
            }
            CredentialStatus::Missing => {}
            CredentialStatus::Error(error) => inline_error(
                ui,
                format!("Windows Credential Manager could not be read: {error}"),
            ),
        }
    }

    fn render_startup(&mut self, ui: &mut egui::Ui) {
        const LABEL: &str = "Start HouWritr automatically when I sign in to Windows";
        match self.autostart.state.clone() {
            Ok(AutostartState::DisabledByOs { reason }) => {
                let mut checked = false;
                ui.add_enabled_ui(false, |ui| {
                    ui.checkbox(&mut checked, LABEL);
                });
                inline_error(ui, format!("Windows overrides this setting. {reason}"));
            }
            Ok(state) => {
                let mut enabled = state.is_enabled();
                ui.horizontal_wrapped(|ui| {
                    let changed = ui.checkbox(&mut enabled, LABEL).changed();
                    info_button(
                        ui,
                        "HouWritr starts silently in the tray at sign-in — no window opens. This is OS state, not part of your settings file, so it is applied immediately.",
                    );
                    if changed {
                        self.autostart.set_enabled(enabled);
                    }
                });
            }
            Err(error) => {
                let mut checked = false;
                ui.add_enabled_ui(false, |ui| {
                    ui.checkbox(&mut checked, LABEL);
                });
                inline_error(ui, format!("Could not read the startup state: {error}"));
            }
        }
        if let Some(error) = &self.autostart.error {
            inline_error(ui, format!("Startup change failed: {error}"));
        }
    }

    fn render_about(&mut self, ui: &mut egui::Ui) {
        ui.heading("About HouWritr");
        ui.label("Meeting recording and audio/video transcription for Windows, on this PC.");
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new(format!(
                "Version {} · {} · Publisher: {}",
                env!("CARGO_PKG_VERSION"),
                architecture_label(),
                about::PUBLISHER
            ))
            .small()
            .color(secondary_text()),
        );
        ui.label(
            egui::RichText::new(about::COPYRIGHT)
                .small()
                .color(secondary_text()),
        );

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(8.0);
        ui.label(egui::RichText::new("Project and support").strong());
        ui.label(
            egui::RichText::new(
                "Official project links open in your default browser. To report a problem, export diagnostics from the tray menu and attach the resulting privacy-safe bundle.",
            )
            .small()
            .color(secondary_text()),
        );
        ui.add_space(4.0);

        let mut selected_link = None;
        ui.horizontal_wrapped(|ui| {
            for link in about::OFFICIAL_LINKS {
                if ui.link(link.label).clicked() {
                    selected_link = Some((link.label, link.url));
                }
            }
        });
        if let Some((label, url)) = selected_link {
            self.set_about_result(label, about::open_url(url));
        }

        ui.add_space(6.0);
        if ui.small_button("Open logs").clicked() {
            self.set_about_result("Open logs", diagnostics::open_logs_dir());
        }
        ui.label(
            egui::RichText::new(
                "Tray menu → Export diagnostics creates a bounded bundle without audio, transcript text or API keys.",
            )
            .small()
            .color(secondary_text()),
        );

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(8.0);
        ui.label(egui::RichText::new("Legal").strong());
        ui.label(about::LICENSE_SUMMARY);
        ui.horizontal_wrapped(|ui| {
            if ui.small_button("Open MIT License").clicked() {
                self.set_about_result("Open MIT License", about::open_license());
            }
            if ui.small_button("Open privacy policy").clicked() {
                self.set_about_result("Open privacy policy", about::open_privacy_policy());
            }
            if ui.small_button("Open third-party licenses").clicked() {
                self.set_about_result(
                    "Open third-party licenses",
                    about::open_third_party_licenses(),
                );
            }
            if ui.link("MIT License online").clicked() {
                self.set_about_result("MIT License online", about::open_url(about::LICENSE_URL));
            }
        });

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(8.0);
        ui.label(egui::RichText::new("Models, runtimes, and credits").strong());
        let mut selected_credit = None;
        for credit in about::CREDITS {
            ui.horizontal_wrapped(|ui| {
                if ui.link(credit.name).clicked() {
                    selected_credit = Some((credit.name, credit.url));
                }
                ui.label(
                    egui::RichText::new(credit.attribution)
                        .small()
                        .color(secondary_text()),
                );
            });
        }
        if let Some((label, url)) = selected_credit {
            self.set_about_result(label, about::open_url(url));
        }

        ui.add_space(10.0);
        ui.label(
            egui::RichText::new(about::DISCLAIMER)
                .small()
                .italics()
                .color(secondary_text()),
        );

        if let Some(error) = &self.about_error {
            ui.add_space(8.0);
            inline_error(ui, error);
        }
    }

    fn set_about_result(&mut self, action: &str, result: Result<()>) {
        self.about_error = result
            .err()
            .map(|error| format!("{action} failed: {error}"));
    }
}

impl eframe::App for SettingsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_copilot_probe();
        self.poll_copilot_models();
        self.poll_speaker_model_job(ctx);
        self.poll_local_model(ctx);

        if ctx.input(|input| input.viewport().close_requested()) && self.dirty {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.confirm_discard = true;
        }

        let mut save_requested = false;
        let mut cancel_requested = false;
        let blocking_error = self.blocking_error();

        egui::TopBottomPanel::top("settings_header").show(ctx, |ui| {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.heading("HouWritr");
                    ui.label(
                        egui::RichText::new("Local voice-to-text for Windows")
                            .color(secondary_text()),
                    );
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    status_label(ui, architecture_label(), secondary_text());
                });
            });
            ui.add_space(8.0);
        });

        egui::TopBottomPanel::bottom("settings_footer").show(ctx, |ui| {
            ui.add_space(6.0);
            if let Some(error) = &blocking_error {
                ui.label(egui::RichText::new(error).small().color(error_color()));
                ui.add_space(4.0);
            }
            ui.horizontal(|ui| {
                if ui.small_button("About, credits & support").clicked() {
                    self.about_error = None;
                    self.show_about = true;
                }
                ui.label(
                    egui::RichText::new(format!(
                        "v{} - {}",
                        env!("CARGO_PKG_VERSION"),
                        architecture_label()
                    ))
                    .small()
                    .color(secondary_text()),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add_enabled(
                            blocking_error.is_none(),
                            egui::Button::new(
                                egui::RichText::new("Save").color(egui::Color32::WHITE),
                            )
                            .fill(egui::Color32::from_rgb(79, 140, 255)),
                        )
                        .clicked()
                    {
                        save_requested = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel_requested = true;
                    }
                });
            });
            ui.add_space(6.0);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.set_min_width(ui.available_width());
                    ui.add_space(8.0);

                    if let Some(error) = &self.error {
                        message_frame(ui, error, error_color());
                        ui.add_space(8.0);
                    }
                    if self.load_failed {
                        let changed = ui
                            .checkbox(
                                &mut self.replace_load_failed,
                                "Replace the invalid or unreadable settings file when saving",
                            )
                            .changed();
                        self.dirty |= changed;
                        ui.add_space(8.0);
                    }

                    section(
                        ui,
                        "Transcription",
                        "Choose where local speech recognition runs.",
                        |ui| self.render_engines(ui),
                    );
                    section(
                        ui,
                        "Output and feedback",
                        "Choose how transcripts are saved and what HouWritr shows.",
                        |ui| self.render_output(ui),
                    );
                    section(
                        ui,
                        "Speaker labels",
                        "Label who is speaking in file and meeting transcripts.",
                        |ui| self.render_speaker_labels(ui),
                    );
                    section(
                        ui,
                        "Text enhancement",
                        "Optionally send transcript text for punctuation and cleanup.",
                        |ui| self.render_enhancement(ui),
                    );
                    section(
                        ui,
                        "Startup",
                        "Control whether HouWritr runs automatically when you sign in.",
                        |ui| self.render_startup(ui),
                    );
                    ui.add_space(12.0);
                });
        });

        if cancel_requested {
            if self.dirty {
                self.confirm_discard = true;
            } else {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }

        if save_requested {
            match self.persist() {
                Ok(()) => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
                Err(error) => self.error = Some(error.to_string()),
            }
        }

        if self.show_about {
            let mut close_about = false;
            egui::Modal::new(egui::Id::new("about_houwritr")).show(ctx, |ui| {
                ui.set_min_width(520.0);
                egui::ScrollArea::vertical()
                    .max_height(560.0)
                    .show(ui, |ui| self.render_about(ui));
                ui.add_space(8.0);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Close").clicked() {
                        close_about = true;
                    }
                });
            });
            if close_about {
                self.show_about = false;
            }
        }

        if self.prompt_editor.pending_target_switch {
            let mut decision = None;
            let response =
                egui::Modal::new(egui::Id::new("prompt_target_switch")).show(ctx, |ui| {
                    ui.heading("Save prompt draft before changing target?");
                    ui.label(
                        "The provider, endpoint, or model changed. This draft belongs only to its original target.",
                    );
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Save draft").clicked() {
                            decision = Some(PromptTargetSwitchDecision::Save);
                        }
                        if ui.button("Discard draft").clicked() {
                            decision = Some(PromptTargetSwitchDecision::Discard);
                        }
                        if ui.button("Cancel target change").clicked() {
                            decision = Some(PromptTargetSwitchDecision::Cancel);
                        }
                    });
                });
            if let Some(decision) = decision {
                self.dirty |= self
                    .prompt_editor
                    .resolve_target_switch(&mut self.settings, decision);
            } else if response.should_close() {
                self.dirty |= self
                    .prompt_editor
                    .resolve_target_switch(&mut self.settings, PromptTargetSwitchDecision::Cancel);
            }
        }

        if let Some(target) = self.prompt_editor.reset_target.clone() {
            let mut reset = false;
            let mut keep = false;
            let response = egui::Modal::new(egui::Id::new("reset_prompt_override")).show(
                ctx,
                |ui| {
                    ui.heading("Reset custom prompt?");
                    ui.label(
                        "This removes the custom prompt only for the current provider, endpoint, and model target.",
                    );
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Reset this target").clicked() {
                            reset = true;
                        }
                        if ui.button("Keep custom prompt").clicked() {
                            keep = true;
                        }
                    });
                },
            );
            if reset {
                self.dirty |= self.settings.prompt_overrides.remove(&target);
                self.prompt_editor.reset_target = None;
            } else if keep || response.should_close() {
                self.prompt_editor.reset_target = None;
            }
        }

        if self.confirm_discard {
            let mut keep_editing = false;
            let mut discard = false;
            let response = egui::Modal::new(egui::Id::new("discard_settings")).show(ctx, |ui| {
                ui.heading("Discard unsaved changes?");
                ui.label("Your changes have not been saved.");
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Keep editing").clicked() {
                        keep_editing = true;
                    }
                    if ui.button("Discard changes").clicked() {
                        discard = true;
                    }
                });
            });
            if keep_editing || response.should_close() {
                self.confirm_discard = false;
            }
            if discard {
                self.dirty = false;
                self.confirm_discard = false;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }

        if self.copilot_rx.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
    }
}

struct EngineDescriptor {
    id: &'static str,
    title: &'static str,
    description: &'static str,
}

const ENGINE_DESCRIPTORS: &[EngineDescriptor] = &[
    EngineDescriptor {
        id: "parakeet_cpu",
        title: "Parakeet TDT v3 - CPU",
        description: "Runs locally on the CPU and works on Intel, AMD, and ARM64 PCs.",
    },
    EngineDescriptor {
        id: "parakeet_npu",
        title: "Parakeet TDT v3 - Snapdragon NPU",
        description: "Runs the encoder on the Snapdragon X Elite Hexagon NPU.",
    },
    EngineDescriptor {
        id: "whisper_npu",
        title: "Whisper Large v3 Turbo - Snapdragon NPU",
        description: "Runs the multilingual Whisper encoder and decoder on Snapdragon X Elite.",
    },
];

fn section(ui: &mut egui::Ui, title: &str, description: &str, body: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new(title).size(16.0).strong());
        info_button(ui, description);
    });
    ui.add_space(4.0);
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        body(ui);
    });
    ui.add_space(14.0);
}

fn info_button(ui: &mut egui::Ui, help: &str) {
    let _ = ui
        .small_button("i")
        .on_hover_cursor(egui::CursorIcon::Help)
        .on_hover_text_at_pointer(help);
}

pub(crate) fn inline_error(ui: &mut egui::Ui, message: impl Into<String>) {
    ui.label(
        egui::RichText::new(message.into())
            .small()
            .color(error_color()),
    );
}

fn status_label(ui: &mut egui::Ui, message: &str, color: egui::Color32) {
    ui.label(egui::RichText::new(message).small().color(color));
}

fn message_frame(ui: &mut egui::Ui, message: &str, color: egui::Color32) {
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        ui.label(egui::RichText::new(message).color(color));
    });
}

fn secondary_text() -> egui::Color32 {
    egui::Color32::from_rgb(154, 163, 178)
}

fn status_green() -> egui::Color32 {
    egui::Color32::from_rgb(74, 222, 128)
}

fn warning_color() -> egui::Color32 {
    egui::Color32::from_rgb(250, 204, 21)
}

fn error_color() -> egui::Color32 {
    egui::Color32::from_rgb(248, 113, 113)
}

fn architecture_label() -> &'static str {
    match std::env::consts::ARCH {
        "aarch64" => "ARM64",
        "x86_64" => "x64",
        other => other,
    }
}

fn prompt_target_label(target: &PromptTarget) -> String {
    match target.endpoint() {
        Some(endpoint) => format!(
            "OpenAI-compatible · {} · {}",
            endpoint.base_url(),
            target.model_id()
        ),
        None => format!("GitHub Copilot · {}", target.model_id()),
    }
}

fn prompt_source_label(source: PromptSource) -> &'static str {
    match source {
        PromptSource::CustomOverride => "custom override",
        PromptSource::ModelDefault => "bundled model default",
        PromptSource::ProviderDefault => "bundled provider default",
        PromptSource::GlobalDefault => "bundled global default",
    }
}

fn copilot_model_picker_value(model_id: &str, choices: &[(String, String)]) -> String {
    if choices.iter().any(|(id, _)| id == model_id) {
        model_id.to_string()
    } else {
        "custom".to_string()
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else {
        format!("{:.0} MB", bytes as f64 / 1_000_000.0)
    }
}

fn insecure_remote_http(base_url: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(base_url.trim()) else {
        return false;
    };
    if url.scheme() != "http" {
        return false;
    }
    let Some(host) = url.host_str() else {
        return false;
    };
    if host.eq_ignore_ascii_case("localhost") {
        return false;
    }
    host.parse::<std::net::IpAddr>()
        .map(|address| !address.is_loopback())
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    #[test]
    fn windows_text_size_becomes_a_zoom_factor() {
        assert_eq!(text_scale_from(None), 1.0);
        assert_eq!(text_scale_from(Some(100)), 1.0);
        assert_eq!(text_scale_from(Some(150)), 1.5);
        assert_eq!(text_scale_from(Some(225)), 2.25);
        assert_eq!(text_scale_from(Some(40)), 1.0);
        assert_eq!(text_scale_from(Some(900)), 1.0);
    }

    #[test]
    fn cjk_font_candidates_live_in_the_windows_fonts_folder_best_first() {
        let found = cjk_font_candidates(std::path::Path::new("win"));
        assert_eq!(found.len(), CJK_FONT_FILES.len());
        assert_eq!(
            found[0],
            std::path::Path::new("win").join("Fonts").join("msyh.ttc")
        );
        assert!(found
            .iter()
            .all(|path| path.parent().unwrap().ends_with("Fonts")));
    }

    use super::*;
    use crate::credentials::CredentialError;
    use parking_lot::Mutex;

    #[derive(Default)]
    struct FakeStore {
        secret: Mutex<Option<String>>,
    }

    impl CredentialStore for FakeStore {
        fn read(&self) -> std::result::Result<Option<String>, CredentialError> {
            Ok(self.secret.lock().clone())
        }

        fn write(&self, secret: &str) -> std::result::Result<(), CredentialError> {
            *self.secret.lock() = Some(secret.to_string());
            Ok(())
        }

        fn delete(&self) -> std::result::Result<(), CredentialError> {
            *self.secret.lock() = None;
            Ok(())
        }
    }

    #[test]
    fn legacy_plaintext_removal_keeps_a_distinct_secure_credential() {
        let mut settings = Settings::default();
        assert_eq!(
            requested_credential_change(&settings, true, true, ""),
            CredentialChange::Keep
        );
        assert!(has_openai_credential(
            &CredentialStatus::Saved,
            true,
            true,
            ""
        ));
        assert_eq!(
            requested_credential_change(&settings, false, true, ""),
            CredentialChange::Delete
        );
        assert!(!has_openai_credential(
            &CredentialStatus::Saved,
            false,
            true,
            ""
        ));

        settings.enhance.mode = EnhanceMode::Always;
        settings.enhance.provider = "openai_compatible".into();
        assert_eq!(
            requested_credential_change(&settings, false, false, "replacement"),
            CredentialChange::Set("replacement")
        );
    }

    #[test]
    fn failed_settings_commit_restores_the_previous_credential() {
        let store = FakeStore::default();
        *store.secret.lock() = Some("previous".into());

        let result =
            persist_with_credential_change(&store, CredentialChange::Set("replacement"), || {
                Err(anyhow!("injected settings failure"))
            });

        assert!(result.is_err());
        assert_eq!(store.secret.lock().as_deref(), Some("previous"));
    }

    #[test]
    fn failed_settings_commit_removes_a_new_credential_when_none_existed() {
        let store = FakeStore::default();

        let result =
            persist_with_credential_change(&store, CredentialChange::Set("replacement"), || {
                Err(anyhow!("injected settings failure"))
            });

        assert!(result.is_err());
        assert!(store.secret.lock().is_none());
    }

    #[test]
    fn failed_settings_commit_restores_a_deleted_credential() {
        let store = FakeStore::default();
        *store.secret.lock() = Some("previous".into());

        let result = persist_with_credential_change(&store, CredentialChange::Delete, || {
            Err(anyhow!("injected settings failure"))
        });

        assert!(result.is_err());
        assert_eq!(store.secret.lock().as_deref(), Some("previous"));
    }

    #[test]
    fn stale_settings_revision_is_rejected_before_persistence() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("settings.json");
        std::fs::write(&path, b"first").unwrap();
        let revision = Settings::revision_from(&path).unwrap();
        std::fs::write(&path, b"second").unwrap();

        let error = ensure_revision_unchanged(&path, &revision).unwrap_err();

        assert!(error
            .to_string()
            .contains("changed after this window opened"));
    }

    #[test]
    fn credential_status_never_exposes_the_secret() {
        let store = FakeStore::default();
        *store.secret.lock() = Some("super-secret".into());

        assert!(matches!(
            read_credential_status(&store),
            CredentialStatus::Saved
        ));
    }

    #[test]
    fn prompt_editor_switch_save_stages_only_the_original_target() {
        let original = PromptTarget::github_copilot("custom-a").unwrap();
        let switched = PromptTarget::github_copilot("custom-b").unwrap();
        let mut editor = PromptEditorState::default();
        let mut settings = Settings::default();
        editor.begin_edit(original.clone(), "draft prompt".into(), &settings.enhance);

        assert!(editor.request_target_switch(&switched));
        assert!(editor.resolve_target_switch(&mut settings, PromptTargetSwitchDecision::Save));

        assert_eq!(
            settings.prompt_overrides.get(&original),
            Some("draft prompt")
        );
        assert!(settings.prompt_overrides.get(&switched).is_none());
        assert!(!editor.has_open_draft());
    }

    #[test]
    fn prompt_editor_switch_cancel_restores_target_and_keeps_draft() {
        let original = PromptTarget::openai_compatible(
            crate::cleanup::EndpointScope::parse("https://api.example.com/v1").unwrap(),
            "deployment-a",
        )
        .unwrap();
        let switched = PromptTarget::github_copilot("custom-b").unwrap();
        let mut editor = PromptEditorState::default();
        let mut settings = Settings::default();
        settings.enhance.provider = "openai_compatible".into();
        settings.enhance.base_url = "HTTPS://API.EXAMPLE.COM:443/v1/".into();
        settings.enhance.model = "deployment-a".into();
        editor.begin_edit(original.clone(), "draft prompt".into(), &settings.enhance);
        settings.enhance.provider = "github_copilot".into();
        settings.enhance.model = "custom-b".into();

        assert!(editor.request_target_switch(&switched));
        assert!(!editor.resolve_target_switch(&mut settings, PromptTargetSwitchDecision::Cancel));

        assert_eq!(settings.prompt_target().unwrap(), original);
        assert_eq!(settings.enhance.base_url, "HTTPS://API.EXAMPLE.COM:443/v1/");
        assert!(editor.has_open_draft());
        assert!(!editor.pending_target_switch);
    }

    #[test]
    fn advisory_picker_keeps_custom_model_ids_as_custom() {
        let choices = vec![
            ("gpt-4.1".to_string(), "GPT-4.1 (gpt-4.1)".to_string()),
            ("gpt-4o".to_string(), "GPT-4o (gpt-4o)".to_string()),
        ];
        assert_eq!(copilot_model_picker_value("gpt-4.1", &choices), "gpt-4.1");
        assert_eq!(
            copilot_model_picker_value("my-private/deployment-v42", &choices),
            "custom"
        );
        let mut settings = Settings::default();
        settings.enhance.model = "my-private/deployment-v42".into();
        assert_eq!(
            settings.enhance.model, "my-private/deployment-v42",
            "custom picker selection must not replace the configured ID"
        );
    }

    #[test]
    fn insecure_http_warning_excludes_loopback_endpoints() {
        assert!(!insecure_remote_http("http://localhost:11434/v1"));
        assert!(!insecure_remote_http("http://127.0.0.1:11434/v1"));
        assert!(insecure_remote_http("http://example.com/v1"));
        assert!(!insecure_remote_http("https://example.com/v1"));
    }

    struct FakeAutostart {
        state: Mutex<AutostartState>,
        fail_writes: bool,
    }

    impl AutostartBackend for FakeAutostart {
        fn kind(&self) -> &'static str {
            "fake"
        }
        fn state(&self) -> std::result::Result<AutostartState, autostart::AutostartError> {
            Ok(self.state.lock().clone())
        }
        fn enable(&self) -> std::result::Result<(), autostart::AutostartError> {
            if self.fail_writes {
                return Err(autostart::AutostartError::VerificationFailed("enable"));
            }
            *self.state.lock() = AutostartState::Enabled;
            Ok(())
        }
        fn disable(&self) -> std::result::Result<(), autostart::AutostartError> {
            if self.fail_writes {
                return Err(autostart::AutostartError::VerificationFailed("disable"));
            }
            *self.state.lock() = AutostartState::Disabled;
            Ok(())
        }
    }

    #[test]
    fn autostart_write_failure_surfaces_an_error_and_does_not_flip_state() {
        let backend = Box::new(FakeAutostart {
            state: Mutex::new(AutostartState::Disabled),
            fail_writes: true,
        });
        let mut ui = AutostartUi::from_backend(backend);

        ui.set_enabled(true);

        assert!(
            ui.error.is_some(),
            "a failed write must surface an error, not silently succeed"
        );
        assert_eq!(ui.state.as_ref().unwrap(), &AutostartState::Disabled);
    }

    #[test]
    fn autostart_successful_toggle_updates_state_without_error() {
        let backend = Box::new(FakeAutostart {
            state: Mutex::new(AutostartState::Disabled),
            fail_writes: false,
        });
        let mut ui = AutostartUi::from_backend(backend);

        ui.set_enabled(true);
        assert!(ui.error.is_none());
        assert_eq!(ui.state.as_ref().unwrap(), &AutostartState::Enabled);

        ui.set_enabled(false);
        assert!(ui.error.is_none());
        assert_eq!(ui.state.as_ref().unwrap(), &AutostartState::Disabled);
    }
}
