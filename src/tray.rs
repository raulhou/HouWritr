use crate::settings::Settings;
use anyhow::Result;
use image::{ImageBuffer, Rgba};
use tray_icon::menu::{Menu, MenuId, MenuItem, PredefinedMenuItem};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

pub struct Tray {
    pub _icon: TrayIcon,
    pub menu_quit_id: MenuId,
    pub menu_settings_id: MenuId,
    pub menu_about_id: MenuId,
    pub menu_open_logs_id: MenuId,
    pub menu_export_diagnostics_id: MenuId,
    pub menu_cancel_model_id: MenuId,
    pub menu_retry_engine_id: MenuId,
    pub menu_transcribe_file_id: MenuId,
    pub menu_meeting_id: MenuId,
    pub menu_meeting_pause_id: MenuId,
    pub menu_meeting_indicator_id: MenuId,
    meeting_item: MenuItem,
    meeting_pause_item: MenuItem,
    meeting_indicator_item: MenuItem,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeetingMenuState {
    Idle,
    Recording,
    Paused,
}

#[derive(Clone, Copy, Debug)]
pub enum IconColor {
    Idle,
    Recording,
    Transcribing,
    Error,
}

impl Tray {
    pub fn new(_s: &Settings) -> Result<Self> {
        let menu = Menu::new();
        let info = MenuItem::new("HouWritr — hold hotkey to dictate", false, None);
        let retry_engine = MenuItem::new("Retry model / engine", true, None);
        let cancel_model = MenuItem::new("Cancel model download", true, None);
        let model_sep = PredefinedMenuItem::separator();
        let transcribe_file = MenuItem::new("Transcribe audio or video file…", true, None);
        let meeting = MenuItem::new("Start meeting recording", true, None);
        let meeting_pause = MenuItem::new("Pause meeting recording", false, None);
        let meeting_indicator = MenuItem::new("Hide recording indicator", true, None);
        let transcribe_sep = PredefinedMenuItem::separator();
        let open_logs = MenuItem::new("Open logs", true, None);
        let export_diagnostics = MenuItem::new("Export diagnostics…", true, None);
        let settings = MenuItem::new("Settings…", true, None);
        let about = MenuItem::new("About, credits & support…", true, None);
        let sep = PredefinedMenuItem::separator();
        let sep2 = PredefinedMenuItem::separator();
        let quit = MenuItem::new("Quit", true, None);
        menu.append(&info)?;
        menu.append(&sep)?;
        menu.append(&retry_engine)?;
        menu.append(&cancel_model)?;
        menu.append(&model_sep)?;
        menu.append(&transcribe_file)?;
        menu.append(&meeting)?;
        menu.append(&meeting_pause)?;
        menu.append(&meeting_indicator)?;
        menu.append(&transcribe_sep)?;
        menu.append(&open_logs)?;
        menu.append(&export_diagnostics)?;
        menu.append(&settings)?;
        menu.append(&about)?;
        menu.append(&sep2)?;
        menu.append(&quit)?;

        let icon = make_icon(IconColor::Idle)?;
        let tray = TrayIconBuilder::new()
            .with_menu(Box::new(menu))
            .with_tooltip("HouWritr")
            .with_icon(icon)
            .build()?;

        Ok(Self {
            _icon: tray,
            menu_quit_id: quit.id().clone(),
            menu_settings_id: settings.id().clone(),
            menu_about_id: about.id().clone(),
            menu_open_logs_id: open_logs.id().clone(),
            menu_export_diagnostics_id: export_diagnostics.id().clone(),
            menu_cancel_model_id: cancel_model.id().clone(),
            menu_retry_engine_id: retry_engine.id().clone(),
            menu_transcribe_file_id: transcribe_file.id().clone(),
            menu_meeting_id: meeting.id().clone(),
            menu_meeting_pause_id: meeting_pause.id().clone(),
            menu_meeting_indicator_id: meeting_indicator.id().clone(),
            meeting_item: meeting,
            meeting_pause_item: meeting_pause,
            meeting_indicator_item: meeting_indicator,
        })
    }

    pub fn set_color(&self, color: IconColor) {
        if let Ok(icon) = make_icon(color) {
            let _ = self._icon.set_icon(Some(icon));
        }
    }

    pub fn set_status(&self, color: IconColor, tooltip: &str) {
        self.set_color(color);
        self.set_tooltip(tooltip);
    }

    /// Reflect the meeting recorder: the first item starts/stops, the second
    /// pauses/resumes and is only usable while a meeting is running.
    pub fn set_meeting_state(&self, state: MeetingMenuState) {
        let running = state != MeetingMenuState::Idle;
        self.meeting_item.set_text(if running {
            "Stop meeting recording"
        } else {
            "Start meeting recording"
        });
        self.meeting_pause_item
            .set_text(if state == MeetingMenuState::Paused {
                "Resume meeting recording"
            } else {
                "Pause meeting recording"
            });
        self.meeting_pause_item.set_enabled(running);
    }

    /// Menu text for the meeting recording indicator toggle.
    pub fn set_meeting_indicator_visible(&self, visible: bool) {
        self.meeting_indicator_item.set_text(if visible {
            "Hide recording indicator"
        } else {
            "Show recording indicator"
        });
    }

    pub fn set_tooltip(&self, tooltip: &str) {
        let _ = self._icon.set_tooltip(Some(tooltip));
    }
}

fn make_icon(color: IconColor) -> Result<Icon> {
    let (r, g, b) = match color {
        IconColor::Idle => (74, 144, 226),
        IconColor::Recording => (220, 38, 38),
        IconColor::Transcribing => (245, 158, 11),
        IconColor::Error => (107, 114, 128),
    };
    let mut img = ImageBuffer::from_pixel(64u32, 64u32, Rgba([0u8, 0, 0, 0]));
    let fg = Rgba([r, g, b, 255]);
    // Mic body
    for y in 12..38 {
        for x in 24..40 {
            img.put_pixel(x, y, fg);
        }
    }
    // Stem + base
    for y in 38..50 {
        img.put_pixel(31, y, fg);
        img.put_pixel(32, y, fg);
    }
    for x in 20..45 {
        img.put_pixel(x, 50, fg);
        img.put_pixel(x, 51, fg);
    }

    Ok(Icon::from_rgba(img.into_raw(), 64, 64)?)
}
