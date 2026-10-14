//! Meeting capture: microphone + system output (WASAPI loopback) mixed into
//! one 16 kHz mono timeline.
//!
//! Both sources are downsampled inside their audio callbacks and summed into
//! a single shared `i16` buffer, so a five-hour meeting costs ~576 MB instead
//! of two full-rate f32 copies. Loopback devices deliver nothing while the
//! system is silent, so each track re-synchronises to the wall clock when it
//! falls behind; otherwise the two sources would drift apart.
//!
//! Pausing closes the capture streams (so Windows' "microphone in use"
//! indicator turns off) and stops the recording clock; resuming reopens the
//! default devices and continues the same timeline with no gap.

use crate::media::{Downsampler, TARGET_RATE};
use anyhow::{anyhow, bail, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};

const MAX_MEETING_SECONDS: usize = 5 * 3600;
const MAX_SAMPLES: usize = MAX_MEETING_SECONDS * TARGET_RATE as usize;
/// Resync a track once it lags the wall clock by more than this.
const RESYNC_GAP_SAMPLES: usize = TARGET_RATE as usize / 2;

/// Recording buffer size step: ten minutes of 16 kHz samples (~19 MB).
const CHUNK_SAMPLES: usize = 10 * 60 * TARGET_RATE as usize;

/// The mixed recording, stored as fixed ten-minute chunks of `i16`.
///
/// One growing `Vec` would reallocate and copy hundreds of megabytes inside
/// the audio callback on a long meeting (stalling capture) and double its
/// footprint at the end. Chunks are allocated once, never moved, and are
/// freed one by one as the recording is converted for transcription.
struct Mix {
    chunks: Vec<Vec<i16>>,
    len: usize,
}

impl Mix {
    fn new() -> Self {
        Self {
            chunks: Vec::new(),
            len: 0,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    /// Sum `samples` (clamped to [-1, 1]) into the timeline at `position`,
    /// saturating instead of wrapping.
    fn add(&mut self, position: usize, samples: &[f32]) {
        let end = position + samples.len();
        while self.chunks.len() * CHUNK_SAMPLES < end {
            self.chunks.push(vec![0; CHUNK_SAMPLES]);
        }
        let (mut at, mut done) = (position, 0);
        while done < samples.len() {
            let offset = at % CHUNK_SAMPLES;
            let count = (CHUNK_SAMPLES - offset).min(samples.len() - done);
            let chunk = &mut self.chunks[at / CHUNK_SAMPLES];
            for (slot, &sample) in chunk[offset..offset + count]
                .iter_mut()
                .zip(&samples[done..done + count])
            {
                let value = (sample.clamp(-1.0, 1.0) * 32_767.0) as i16;
                *slot = slot.saturating_add(value);
            }
            at += count;
            done += count;
        }
        self.len = self.len.max(end);
    }

    /// The recording as f32, releasing each chunk as soon as it is read.
    fn into_f32(self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.len);
        let mut remaining = self.len;
        for chunk in self.chunks {
            if remaining == 0 {
                break;
            }
            let take = remaining.min(chunk.len());
            out.extend(chunk[..take].iter().map(|&s| s as f32 / 32_768.0));
            remaining -= take;
        }
        out
    }

    #[cfg(test)]
    fn to_vec(&self) -> Vec<i16> {
        self.chunks
            .iter()
            .flatten()
            .copied()
            .take(self.len)
            .collect()
    }
}

struct Track {
    downsampler: Downsampler,
    mono: Vec<f32>,
    resampled: Vec<f32>,
    position: usize,
}

/// Recording time: wall clock minus every pause.
struct Clock {
    started: Instant,
    paused_total: Duration,
    paused_at: Option<Instant>,
}

impl Clock {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            paused_total: Duration::ZERO,
            paused_at: None,
        }
    }

    fn elapsed(&self) -> Duration {
        let end = self.paused_at.unwrap_or_else(Instant::now);
        end.saturating_duration_since(self.started)
            .saturating_sub(self.paused_total)
    }
}

struct Shared {
    mix: Mutex<Mix>,
    clock: Mutex<Clock>,
    /// Live input level (RMS x 10 000, same scale as dictation) for the
    /// recording indicator; the louder of the microphone and system audio.
    level_x10000: Arc<AtomicU32>,
    limit_reached: AtomicBool,
    failed: Mutex<Option<String>>,
}

pub struct MeetingRecorder {
    streams: Vec<cpal::Stream>,
    shared: Arc<Shared>,
    sources: Vec<&'static str>,
    paused: bool,
}

impl MeetingRecorder {
    /// Start capturing. Succeeds if at least one of microphone / system audio
    /// could be opened; `sources()` reports which.
    pub fn start(level_x10000: Arc<AtomicU32>) -> Result<Self> {
        level_x10000.store(0, Ordering::Relaxed);
        let shared = Arc::new(Shared {
            mix: Mutex::new(Mix::new()),
            clock: Mutex::new(Clock::new()),
            level_x10000,
            limit_reached: AtomicBool::new(false),
            failed: Mutex::new(None),
        });
        let (streams, sources) = open_sources(&shared)?;
        info!(sources = ?sources, "meeting capture started");
        Ok(Self {
            streams,
            shared,
            sources,
            paused: false,
        })
    }

    /// Stop capturing without ending the meeting. Idempotent.
    pub fn pause(&mut self) {
        if self.paused {
            return;
        }
        // Dropping the streams stops the callbacks and releases the devices.
        self.streams.clear();
        self.shared.level_x10000.store(0, Ordering::Relaxed);
        self.shared.clock.lock().paused_at = Some(Instant::now());
        self.paused = true;
        info!("meeting capture paused");
    }

    /// Reopen the default devices and continue. On failure the recorder stays
    /// paused so the user can try again or stop.
    pub fn resume(&mut self) -> Result<()> {
        if !self.paused {
            return Ok(());
        }
        let (streams, sources) = open_sources(&self.shared)?;
        {
            let mut clock = self.shared.clock.lock();
            if let Some(paused_at) = clock.paused_at.take() {
                clock.paused_total += paused_at.elapsed();
            }
        }
        *self.shared.failed.lock() = None;
        self.streams = streams;
        self.sources = sources;
        self.paused = false;
        info!(sources = ?self.sources, "meeting capture resumed");
        Ok(())
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    pub fn sources(&self) -> String {
        self.sources.join(" + ")
    }

    /// Time actually recorded (pauses excluded).
    pub fn elapsed(&self) -> Duration {
        self.shared.clock.lock().elapsed()
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
        shared.level_x10000.store(0, Ordering::Relaxed);
        let mix = std::mem::replace(&mut *shared.mix.lock(), Mix::new());
        mix.into_f32()
    }
}

/// Open the default microphone and the default output device (loopback).
fn open_sources(shared: &Arc<Shared>) -> Result<(Vec<cpal::Stream>, Vec<&'static str>)> {
    let host = cpal::default_host();
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

    // Building an *input* stream on the default *output* device is how cpal
    // exposes WASAPI loopback.
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
    Ok((streams, sources))
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
            publish_level(&shared.level_x10000, &track.mono, channels);
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

/// RMS of one callback's mono samples on dictation's 0..10 000 scale.
fn scaled_level(mono: &[f32], channels: usize) -> u32 {
    if mono.is_empty() {
        return 0;
    }
    let rms = (mono.iter().map(|s| s * s).sum::<f32>() / mono.len() as f32).sqrt();
    // Averaging channels to mono lowers the level; compensate like audio.rs.
    let gain = if channels <= 1 {
        1.0
    } else {
        (channels as f32).sqrt()
    };
    ((rms * gain).min(1.0) * 10_000.0) as u32
}

/// Publish a level without letting the quieter of the two sources (which
/// report from separate callbacks) wipe out the louder one: new peaks win
/// at once, otherwise the displayed level decays by an eighth per callback.
fn next_level(current: u32, measured: u32) -> u32 {
    measured.max(current - current.div_ceil(8))
}

fn publish_level(level: &AtomicU32, mono: &[f32], channels: usize) {
    let measured = scaled_level(mono, channels);
    let current = level.load(Ordering::Relaxed);
    level.store(next_level(current, measured), Ordering::Relaxed);
}

fn write(shared: &Shared, track: &mut Track) {
    let incoming = track.resampled.len();
    if incoming == 0 {
        return;
    }
    // Loopback pauses during silence: jump forward to where the wall clock
    // says this audio belongs.
    let expected_end = (shared.clock.lock().elapsed().as_secs_f64() * TARGET_RATE as f64) as usize;
    let expected_start = expected_end.saturating_sub(incoming);
    if expected_start > track.position + RESYNC_GAP_SAMPLES {
        track.position = expected_start;
    }
    if track.position + incoming > MAX_SAMPLES {
        shared.limit_reached.store(true, Ordering::Relaxed);
        return;
    }
    shared.mix.lock().add(track.position, &track.resampled);
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

/// Notes path next to the transcript:
/// `Documents\OpenWritr\Meetings\Meeting notes YYYY-MM-DD HH-MM.md`.
pub fn default_notes_path(local_time: LocalTime) -> std::path::PathBuf {
    let mut path = default_output_path(local_time);
    let name = format!(
        "Meeting notes {:04}-{:02}-{:02} {:02}-{:02}.md",
        local_time.year, local_time.month, local_time.day, local_time.hour, local_time.minute
    );
    path.set_file_name(name);
    path
}

/// Ask whether to also create AI meeting notes. Blocks on a modal Windows
/// message box, so call it from a worker thread, never the tray event loop.
pub fn ask_create_notes(recorded: Duration) -> bool {
    use windows::core::PCWSTR;
    use windows::Win32::UI::WindowsAndMessaging::{
        MessageBoxW, IDYES, MB_ICONQUESTION, MB_SETFOREGROUND, MB_TOPMOST, MB_YESNO,
    };
    let minutes = recorded.as_secs() / 60;
    let seconds = recorded.as_secs() % 60;
    let text = format!(
        "Meeting recorded ({minutes}:{seconds:02}).\n\n\
         Also create meeting notes with AI?\n\n\
         Yes: save the transcript and a notes file (Summary, Decisions, Action items).\n\
         No: save the transcript only.\n\n\
         Notes send the transcript text to your configured AI provider."
    );
    let wide =
        |value: &str| -> Vec<u16> { value.encode_utf16().chain(std::iter::once(0)).collect() };
    let text = wide(&text);
    let caption = wide("OpenWritr — Meeting recording");
    // SAFETY: both buffers are NUL-terminated and outlive the call.
    let answer = unsafe {
        MessageBoxW(
            None,
            PCWSTR(text.as_ptr()),
            PCWSTR(caption.as_ptr()),
            MB_YESNO | MB_ICONQUESTION | MB_SETFOREGROUND | MB_TOPMOST,
        )
    };
    answer == IDYES
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
            mix: Mutex::new(Mix::new()),
            clock: Mutex::new(Clock::new()),
            level_x10000: Arc::new(AtomicU32::new(0)),
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
        assert!(mix.to_vec().iter().all(|&s| (s - 16_383).abs() <= 2));
    }

    #[test]
    fn mix_spans_chunk_boundaries_without_gaps() {
        let mut mix = Mix::new();
        // 100 samples straddling the end of the first chunk.
        let start = CHUNK_SAMPLES - 40;
        mix.add(start, &[0.5; 100]);
        assert_eq!(mix.len(), CHUNK_SAMPLES + 60);
        assert_eq!(mix.chunks.len(), 2);
        let all = mix.to_vec();
        assert_eq!(all[start - 1], 0);
        assert!(all[start..start + 100].iter().all(|&s| s == 16_383));
        assert_eq!(all[start + 100 - 1], 16_383);

        let floats = mix.into_f32();
        assert_eq!(floats.len(), CHUNK_SAMPLES + 60);
        assert!((floats[CHUNK_SAMPLES] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn the_cap_is_five_hours_of_audio() {
        assert_eq!(MAX_MEETING_SECONDS, 5 * 3600);
        assert_eq!(MAX_SAMPLES, 288_000_000);
    }

    #[test]
    fn summing_saturates_instead_of_wrapping() {
        let shared = shared();
        write(&shared, &mut track(vec![1.0; 10]));
        write(&shared, &mut track(vec![1.0; 10]));
        assert!(shared.mix.lock().to_vec().iter().all(|&s| s == i16::MAX));
    }

    #[test]
    fn level_matches_dictation_scale_and_compensates_channels() {
        assert_eq!(scaled_level(&[], 2), 0);
        assert_eq!(scaled_level(&[0.0; 64], 1), 0);
        let near = |actual: u32, expected: u32| actual.abs_diff(expected) <= 2;
        // Constant 0.1 -> RMS 0.1 -> 1000 (f32 truncation may cost a unit).
        assert!(near(scaled_level(&[0.1; 64], 1), 1_000));
        // Stereo gets sqrt(2) gain, and is clamped at full scale.
        assert!(near(scaled_level(&[0.1; 64], 2), 1_414));
        assert_eq!(scaled_level(&[0.9; 64], 4), 10_000);
    }

    #[test]
    fn level_decays_but_loud_peaks_win_immediately() {
        assert_eq!(next_level(0, 0), 0);
        assert_eq!(next_level(800, 100), 700); // decays by an eighth
        assert_eq!(next_level(5, 0), 4); // and always makes progress
        assert_eq!(next_level(800, 5_000), 5_000); // a new peak is immediate
        let mut level = 8_000;
        for _ in 0..200 {
            level = next_level(level, 0);
        }
        assert_eq!(level, 0, "the meter must settle at zero");
    }

    #[test]
    fn clock_excludes_paused_time() {
        let base = Instant::now();
        let running = Clock {
            started: base,
            paused_total: Duration::from_secs(3),
            paused_at: Some(base + Duration::from_secs(10)),
        };
        // 10 s on the wall clock, 3 s of it earlier pauses.
        assert_eq!(running.elapsed(), Duration::from_secs(7));

        // Still inside a pause: the clock is frozen at the pause instant.
        let frozen = Clock {
            started: base,
            paused_total: Duration::ZERO,
            paused_at: Some(base + Duration::from_secs(4)),
        };
        assert_eq!(frozen.elapsed(), Duration::from_secs(4));
    }

    #[test]
    fn notes_file_sits_next_to_the_transcript() {
        let time = LocalTime {
            year: 2026,
            month: 3,
            day: 9,
            hour: 14,
            minute: 5,
        };
        let transcript = default_output_path(time);
        let notes = default_notes_path(time);
        assert_eq!(notes.parent(), transcript.parent());
        assert!(notes.ends_with("Meetings/Meeting notes 2026-03-09 14-05.md"));
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
