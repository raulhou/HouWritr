//! Meeting capture: microphone + system output (WASAPI loopback) mixed into
//! one 16 kHz mono timeline.
//!
//! Both sources are downsampled inside their audio callbacks and summed into
//! a single shared `i16` buffer, so a two-hour meeting costs ~230 MB instead
//! of two full-rate f32 copies. Loopback devices deliver nothing while the
//! system is silent, so each track re-synchronises to the wall clock when it
//! falls behind; otherwise the two sources would drift apart.

use crate::media::{Downsampler, TARGET_RATE};
use anyhow::{anyhow, bail, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};

const MAX_MEETING_SECONDS: usize = 2 * 3600;
const MAX_SAMPLES: usize = MAX_MEETING_SECONDS * TARGET_RATE as usize;
/// Resync a track once it lags the wall clock by more than this.
const RESYNC_GAP_SAMPLES: usize = TARGET_RATE as usize / 2;

struct Track {
    downsampler: Downsampler,
    mono: Vec<f32>,
    resampled: Vec<f32>,
    position: usize,
}

struct Shared {
    mix: Mutex<Vec<i16>>,
    started: Instant,
    limit_reached: AtomicBool,
    failed: Mutex<Option<String>>,
}

pub struct MeetingRecorder {
    streams: Vec<cpal::Stream>,
    shared: Arc<Shared>,
    sources: Vec<&'static str>,
}

impl MeetingRecorder {
    /// Start capturing. Succeeds if at least one of microphone / system audio
    /// could be opened; `sources()` reports which.
    pub fn start() -> Result<Self> {
        let host = cpal::default_host();
        let shared = Arc::new(Shared {
            mix: Mutex::new(Vec::new()),
            started: Instant::now(),
            limit_reached: AtomicBool::new(false),
            failed: Mutex::new(None),
        });
        let mut streams = Vec::new();
        let mut sources = Vec::new();
        let mut errors = Vec::new();

        match host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default microphone"))
            .and_then(|device| {
                let config = device.default_input_config()?;
                open(&device, config, shared.clone())
            }) {
            Ok(stream) => {
                streams.push(stream);
                sources.push("microphone");
            }
            Err(error) => errors.push(format!("microphone: {error}")),
        }

        // Building an *input* stream on the default *output* device is how
        // cpal exposes WASAPI loopback.
        match host
            .default_output_device()
            .ok_or_else(|| anyhow!("no default output device"))
            .and_then(|device| {
                let config = device.default_output_config()?;
                open(&device, config, shared.clone())
            }) {
            Ok(stream) => {
                streams.push(stream);
                sources.push("system audio");
            }
            Err(error) => errors.push(format!("system audio: {error}")),
        }

        if streams.is_empty() {
            bail!("could not open any audio source ({})", errors.join("; "));
        }
        for error in &errors {
            warn!(
                error,
                "meeting source unavailable; continuing with the rest"
            );
        }
        info!(sources = ?sources, "meeting capture started");
        Ok(Self {
            streams,
            shared,
            sources,
        })
    }

    pub fn sources(&self) -> String {
        self.sources.join(" + ")
    }

    pub fn elapsed(&self) -> Duration {
        self.shared.started.elapsed()
    }

    pub fn limit_reached(&self) -> bool {
        self.shared.limit_reached.load(Ordering::Relaxed)
    }

    pub fn failure(&self) -> Option<String> {
        self.shared.failed.lock().clone()
    }

    /// Stop capture and return the mixed 16 kHz mono recording.
    pub fn finish(self) -> Vec<f32> {
        let Self {
            streams, shared, ..
        } = self;
        drop(streams);
        let mix = std::mem::take(&mut *shared.mix.lock());
        mix.into_iter()
            .map(|sample| sample as f32 / 32_768.0)
            .collect()
    }
}

fn open(
    device: &cpal::Device,
    config: cpal::SupportedStreamConfig,
    shared: Arc<Shared>,
) -> Result<cpal::Stream> {
    let format = config.sample_format();
    let stream_config: cpal::StreamConfig = config.into();
    let stream = match format {
        SampleFormat::F32 => build::<f32>(device, &stream_config, shared)?,
        SampleFormat::I16 => build::<i16>(device, &stream_config, shared)?,
        SampleFormat::I32 => build::<i32>(device, &stream_config, shared)?,
        SampleFormat::U16 => build::<u16>(device, &stream_config, shared)?,
        other => bail!("unsupported sample format {other:?}"),
    };
    stream.play()?;
    Ok(stream)
}

fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    shared: Arc<Shared>,
) -> Result<cpal::Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = config.channels.max(1) as usize;
    let rate = config.sample_rate.0;
    let mut track = Track {
        downsampler: Downsampler::new(rate, TARGET_RATE),
        mono: Vec::new(),
        resampled: Vec::new(),
        position: 0,
    };
    let error_shared = shared.clone();
    let stream = device.build_input_stream(
        config,
        move |data: &[T], _| {
            track.mono.clear();
            track.mono.extend(data.chunks_exact(channels).map(|frame| {
                frame.iter().map(|s| s.to_sample::<f32>()).sum::<f32>() / channels as f32
            }));
            track.resampled.clear();
            let Track {
                downsampler,
                mono,
                resampled,
                ..
            } = &mut track;
            downsampler.push(mono, resampled);
            write(&shared, &mut track);
        },
        move |error| {
            warn!(error = %error, "meeting capture stream error");
            *error_shared.failed.lock() = Some(error.to_string());
        },
        None,
    )?;
    Ok(stream)
}

fn write(shared: &Shared, track: &mut Track) {
    let incoming = track.resampled.len();
    if incoming == 0 {
        return;
    }
    // Loopback pauses during silence: jump forward to where the wall clock
    // says this audio belongs.
    let expected_end = (shared.started.elapsed().as_secs_f64() * TARGET_RATE as f64) as usize;
    let expected_start = expected_end.saturating_sub(incoming);
    if expected_start > track.position + RESYNC_GAP_SAMPLES {
        track.position = expected_start;
    }
    if track.position + incoming > MAX_SAMPLES {
        shared.limit_reached.store(true, Ordering::Relaxed);
        return;
    }
    let mut mix = shared.mix.lock();
    if mix.len() < track.position + incoming {
        mix.resize(track.position + incoming, 0);
    }
    for (slot, &sample) in mix[track.position..].iter_mut().zip(&track.resampled) {
        let value = (sample.clamp(-1.0, 1.0) * 32_767.0) as i16;
        *slot = slot.saturating_add(value);
    }
    track.position += incoming;
}

/// Transcript path for a meeting that started now:
/// `Documents\OpenWritr\Meetings\Meeting YYYY-MM-DD HH-MM.txt`.
pub fn default_output_path(local_time: LocalTime) -> std::path::PathBuf {
    let base = directories::UserDirs::new()
        .and_then(|dirs| dirs.document_dir().map(std::path::Path::to_path_buf))
        .unwrap_or_else(crate::paths::data_dir);
    base.join("OpenWritr").join("Meetings").join(format!(
        "Meeting {:04}-{:02}-{:02} {:02}-{:02}.txt",
        local_time.year, local_time.month, local_time.day, local_time.hour, local_time.minute
    ))
}

#[derive(Clone, Copy, Debug)]
pub struct LocalTime {
    pub year: u16,
    pub month: u16,
    pub day: u16,
    pub hour: u16,
    pub minute: u16,
}

impl LocalTime {
    pub fn now() -> Self {
        use windows::Win32::System::SystemInformation::GetLocalTime;
        // SAFETY: GetLocalTime has no preconditions.
        let time = unsafe { GetLocalTime() };
        Self {
            year: time.wYear,
            month: time.wMonth,
            day: time.wDay,
            hour: time.wHour,
            minute: time.wMinute,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shared() -> Shared {
        Shared {
            mix: Mutex::new(Vec::new()),
            started: Instant::now(),
            limit_reached: AtomicBool::new(false),
            failed: Mutex::new(None),
        }
    }

    fn track(samples: Vec<f32>) -> Track {
        Track {
            downsampler: Downsampler::new(16_000, 16_000),
            mono: Vec::new(),
            resampled: samples,
            position: 0,
        }
    }

    #[test]
    fn two_tracks_are_summed_into_one_timeline() {
        let shared = shared();
        write(&shared, &mut track(vec![0.25; 100]));
        write(&shared, &mut track(vec![0.25; 100]));
        let mix = shared.mix.lock();
        assert_eq!(mix.len(), 100);
        assert!(mix.iter().all(|&s| (s - 16_383).abs() <= 2));
    }

    #[test]
    fn summing_saturates_instead_of_wrapping() {
        let shared = shared();
        write(&shared, &mut track(vec![1.0; 10]));
        write(&shared, &mut track(vec![1.0; 10]));
        assert!(shared.mix.lock().iter().all(|&s| s == i16::MAX));
    }

    #[test]
    fn output_path_is_a_txt_under_documents() {
        let path = default_output_path(LocalTime {
            year: 2026,
            month: 3,
            day: 9,
            hour: 14,
            minute: 5,
        });
        assert!(path.ends_with("Meetings/Meeting 2026-03-09 14-05.txt"));
    }
}
