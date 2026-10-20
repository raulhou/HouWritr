//! Jobs waiting on disk.
//!
//! With power saving on and the PC on its battery, a meeting is only
//! recorded and files are only queued: the work waits until the PC is plugged
//! in. Everything that waits is kept as files in `%LOCALAPPDATA%\HouWritr\pending`
//! so that nothing is lost when the battery runs out, HouWritr is closed, or
//! the setting is switched: every held job is a small JSON file (what to do
//! with the audio and where the transcript goes) and the audio itself as raw
//! 16 kHz mono 16 bit samples. A job's files are deleted when it has been
//! transcribed (or failed), and on the next start the jobs found here are
//! picked up again.
//!
//! The JSON file is written last, through a temporary name, so a job is
//! either complete or not there at all.

use crate::settings::Settings;
use crate::worker::SessionPart;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// What a held job needs besides its audio.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpooledJob {
    /// Assigned by [`Spool::save`]; also the order the jobs were held in.
    #[serde(default)]
    pub id: u64,
    /// "file", "meeting part" or "meeting", for the log.
    pub kind: String,
    pub output: PathBuf,
    pub markdown: bool,
    pub notes: bool,
    pub notes_output: Option<PathBuf>,
    pub title: String,
    pub details: String,
    pub background: bool,
    pub session: Option<(SessionPart, Settings)>,
    pub partial: Option<PathBuf>,
}

pub struct Spool {
    dir: PathBuf,
    next_id: u64,
}

impl Spool {
    /// The spool in `dir` (created when the first job is saved). Ids continue
    /// after the highest one already there.
    pub fn open(dir: PathBuf) -> Self {
        let next_id = ids_in(&dir)
            .into_iter()
            .max()
            .map_or(1, |highest| highest + 1);
        Self { dir, next_id }
    }

    pub fn default_dir() -> PathBuf {
        crate::paths::data_dir().join("pending")
    }

    fn audio_path(&self, id: u64) -> PathBuf {
        self.dir.join(format!("{id:08}.pcm"))
    }

    fn job_path(&self, id: u64) -> PathBuf {
        self.dir.join(format!("{id:08}.json"))
    }

    /// Keep `job` and its `samples` on disk; returns the id they were saved
    /// under. Nothing is left behind when it fails.
    pub fn save(&mut self, job: &SpooledJob, samples: &[f32]) -> io::Result<u64> {
        fs::create_dir_all(&self.dir)?;
        let id = self.next_id;
        let audio = self.audio_path(id);
        let result = write_audio(&audio, samples).and_then(|()| {
            let mut stored = job.clone();
            stored.id = id;
            let json = serde_json::to_vec_pretty(&stored).map_err(io::Error::other)?;
            write_replacing(&self.job_path(id), &json)
        });
        if let Err(error) = result {
            let _ = fs::remove_file(&audio);
            let _ = fs::remove_file(self.job_path(id).with_extension("json.tmp"));
            return Err(error);
        }
        self.next_id += 1;
        Ok(id)
    }

    /// The jobs on disk, oldest first. A file that cannot be read is left
    /// alone and reported in `unreadable`.
    pub fn list(&self) -> (Vec<SpooledJob>, Vec<PathBuf>) {
        let mut jobs = Vec::new();
        let mut unreadable = Vec::new();
        for id in ids_in(&self.dir) {
            let path = self.job_path(id);
            let job = fs::read(&path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<SpooledJob>(&bytes).ok())
                .filter(|job| job.id == id && self.audio_path(id).is_file());
            match job {
                Some(job) => jobs.push(job),
                None => unreadable.push(path),
            }
        }
        (jobs, unreadable)
    }

    /// The audio of job `id`.
    pub fn load_samples(&self, id: u64) -> io::Result<Vec<f32>> {
        let bytes = fs::read(self.audio_path(id))?;
        Ok(bytes
            .chunks_exact(2)
            .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32_768.0)
            .collect())
    }

    /// Delete job `id` (its JSON first: a job without audio is useless, a
    /// stray audio file is swept away on the next start).
    pub fn remove(&self, id: u64) {
        let _ = fs::remove_file(self.job_path(id));
        let _ = fs::remove_file(self.audio_path(id));
    }

    /// Keep the list of unfinished background retries (`repairs.json`).
    pub fn save_repairs(&self, bytes: &[u8]) -> io::Result<()> {
        fs::create_dir_all(&self.dir)?;
        write_replacing(&self.dir.join("repairs.json"), bytes)
    }

    pub fn load_repairs(&self) -> Option<Vec<u8>> {
        fs::read(self.dir.join("repairs.json")).ok()
    }

    pub fn clear_repairs(&self) {
        let _ = fs::remove_file(self.dir.join("repairs.json"));
    }

    /// Delete audio files that have no job file, and half written files.
    pub fn sweep(&self) {
        let Ok(entries) = fs::read_dir(&self.dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let orphan = if name.ends_with(".tmp") {
                true
            } else if let Some(id) = name
                .strip_suffix(".pcm")
                .and_then(|stem| stem.parse::<u64>().ok())
            {
                !self.job_path(id).is_file()
            } else {
                false
            };
            if orphan {
                let _ = fs::remove_file(path);
            }
        }
    }
}

/// Ids of the `NNNNNNNN.json` files in `dir`, ascending.
fn ids_in(dir: &Path) -> Vec<u64> {
    let mut ids: Vec<u64> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name();
            name.to_str()?.strip_suffix(".json")?.parse::<u64>().ok()
        })
        .collect();
    ids.sort_unstable();
    ids
}

fn write_audio(path: &Path, samples: &[f32]) -> io::Result<()> {
    let mut file = io::BufWriter::new(fs::File::create(path)?);
    let mut block = Vec::with_capacity(64 * 1024);
    for chunk in samples.chunks(32 * 1024) {
        block.clear();
        for sample in chunk {
            let value = (sample.clamp(-1.0, 1.0) * 32_767.0).round() as i16;
            block.extend_from_slice(&value.to_le_bytes());
        }
        file.write_all(&block)?;
    }
    file.flush()?;
    file.get_ref().sync_all()
}

/// Write `bytes` to `path` through a temporary file, so the file is either
/// the old one or the whole new one.
fn write_replacing(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let temporary = path.with_extension("json.tmp");
    {
        let mut file = fs::File::create(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    fs::rename(&temporary, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(kind: &str) -> SpooledJob {
        SpooledJob {
            id: 0,
            kind: kind.into(),
            output: PathBuf::from("C:\\Meetings\\Meeting.txt"),
            markdown: true,
            notes: false,
            notes_output: None,
            title: "Meeting".into(),
            details: "Recorded".into(),
            background: true,
            session: Some((
                SessionPart {
                    id: 7,
                    offset_samples: 4_800_000,
                    last: false,
                },
                Settings::default(),
            )),
            partial: Some(PathBuf::from("C:\\Meetings\\Meeting (in progress).txt")),
        }
    }

    #[test]
    fn a_held_job_comes_back_with_its_audio() {
        let dir = tempfile::tempdir().unwrap();
        let mut spool = Spool::open(dir.path().join("pending"));
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 / 1000.0) - 0.5).collect();
        let first = spool.save(&job("meeting part"), &samples).unwrap();
        let second = spool.save(&job("file"), &samples[..10]).unwrap();
        assert_eq!((first, second), (1, 2));

        let (jobs, unreadable) = spool.list();
        assert!(unreadable.is_empty());
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0].id, 1);
        assert_eq!(jobs[1].kind, "file");
        assert_eq!(
            jobs[0].session.as_ref().unwrap().0.offset_samples,
            4_800_000
        );
        let back = spool.load_samples(first).unwrap();
        assert_eq!(back.len(), samples.len());
        // 16 bit audio: close to the original, not equal.
        assert!(back
            .iter()
            .zip(&samples)
            .all(|(a, b)| (a - b).abs() < 1.0 / 16_000.0));
    }

    #[test]
    fn ids_continue_after_a_restart_and_removal_leaves_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let pending = dir.path().join("pending");
        let mut spool = Spool::open(pending.clone());
        let id = spool.save(&job("file"), &[0.1; 100]).unwrap();
        let mut again = Spool::open(pending.clone());
        assert_eq!(again.save(&job("file"), &[0.1; 100]).unwrap(), id + 1);
        again.remove(id);
        again.remove(id + 1);
        assert!(fs::read_dir(&pending).unwrap().next().is_none());
    }

    #[test]
    fn the_retry_list_is_kept_replaced_and_cleared_and_survives_a_sweep() {
        let dir = tempfile::tempdir().unwrap();
        let spool = Spool::open(dir.path().join("pending"));
        assert!(spool.load_repairs().is_none());
        spool.save_repairs(b"[1]").unwrap();
        spool.save_repairs(b"[1,2]").unwrap();
        spool.sweep();
        assert_eq!(spool.load_repairs().unwrap(), b"[1,2]");
        // It is not mistaken for a job.
        assert!(spool.list().0.is_empty());
        spool.clear_repairs();
        assert!(spool.load_repairs().is_none());
    }

    #[test]
    fn broken_or_half_written_files_are_never_used_as_jobs() {
        let dir = tempfile::tempdir().unwrap();
        let pending = dir.path().join("pending");
        let mut spool = Spool::open(pending.clone());
        let good = spool.save(&job("file"), &[0.2; 50]).unwrap();
        // A job file without audio, one that is not JSON, an audio file with
        // no job, and a leftover temporary file.
        fs::write(pending.join("00000005.json"), b"{}").unwrap();
        fs::write(pending.join("00000006.json"), b"not json").unwrap();
        fs::write(pending.join("00000007.pcm"), b"xx").unwrap();
        fs::write(pending.join("00000008.json.tmp"), b"{").unwrap();
        let (jobs, unreadable) = spool.list();
        assert_eq!(jobs.iter().map(|j| j.id).collect::<Vec<_>>(), vec![good]);
        assert_eq!(unreadable.len(), 2);
        spool.sweep();
        assert!(!pending.join("00000007.pcm").exists());
        assert!(!pending.join("00000008.json.tmp").exists());
        // The good job survived the sweep.
        assert!(spool.load_samples(good).is_ok());
    }
}
