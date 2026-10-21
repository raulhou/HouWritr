use std::path::PathBuf;
use std::sync::OnceLock;
use tracing::{info, warn};

pub const APP_NAME: &str = "HouWritr";
/// What the app and its folders were called before the rename.
const LEGACY_APP_NAME: &str = "OpenWritr";

/// `parent\HouWritr`. A folder left by the old name is moved there first, so
/// models, settings and transcripts follow the rename. When the move fails
/// (a file in it is open) the old folder keeps being used rather than
/// starting from nothing.
fn migrated(parent: PathBuf) -> PathBuf {
    let new = parent.join(APP_NAME);
    let old = parent.join(LEGACY_APP_NAME);
    if new.exists() || !old.is_dir() {
        return new;
    }
    match std::fs::rename(&old, &new) {
        Ok(()) => {
            info!(from = %old.display(), to = %new.display(), "moved a folder to the new app name");
            new
        }
        Err(error) => {
            warn!(%error, folder = %old.display(), "could not move a folder to the new app name; keeping the old one");
            old
        }
    }
}

fn base() -> PathBuf {
    static BASE: OnceLock<PathBuf> = OnceLock::new();
    BASE.get_or_init(|| {
        if let Some(p) = std::env::var_os("LOCALAPPDATA") {
            return migrated(PathBuf::from(p));
        }
        directories::BaseDirs::new()
            .map(|d| migrated(d.data_local_dir().to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."))
    })
    .clone()
}

/// `Documents\HouWritr`, where meeting transcripts and notes go.
pub fn documents_dir() -> Option<PathBuf> {
    static DOCUMENTS: OnceLock<Option<PathBuf>> = OnceLock::new();
    DOCUMENTS
        .get_or_init(|| {
            let documents = directories::UserDirs::new()?.document_dir()?.to_path_buf();
            Some(migrated(documents))
        })
        .clone()
}

pub fn log_dir() -> PathBuf {
    base().join("logs")
}
pub fn diagnostics_dir() -> PathBuf {
    base().join("diagnostics")
}
pub fn models_dir() -> PathBuf {
    base().join("models")
}
pub fn settings_path() -> PathBuf {
    base().join("settings.json")
}
#[allow(dead_code)]
pub fn data_dir() -> PathBuf {
    base()
}
