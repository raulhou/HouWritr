//! Meeting capture: microphone + system output (WASAPI loopback) mixed into
//! one 16 kHz mono timeline.
//!
//! Both sources are downsampled inside their audio callbacks and summed into
//! a single shared `i16` buffer (about 115 MB per hour) instead of two
//! full-rate f32 copies. Loopback devices deliver nothing while the
//! system is silent, so each track re-synchronises to the wall clock when it
//! falls behind; otherwise the two sources would drift apart.
//!
//! Audio is handed out as it is recorded: every hour, and whenever the
//! recording is paused, the finished part is detached from the front of the
//! buffer for transcription while capture carries on. Positions are global
//! sample indices on the whole meeting's timeline, so cutting a part never
//! changes where later audio lands.
//!
//! Pausing closes the capture streams (so Windows' "microphone in use"
//! indicator turns off) and stops the recording clock; resuming reopens the
//! default devices and continues the same timeline with no gap.

use crate::media::{Downsampler, TARGET_RATE};
use anyhow::{anyhow, bail, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Hard limit on one meeting. Memory no longer grows with it: finished hours
/// are handed to the transcriber and released.
const MAX_MEETING_SECONDS: usize = 12 * 3600;
const MAX_SAMPLES: usize = MAX_MEETING_SECONDS * TARGET_RATE as usize;
/// Audio is cut off for transcription once this much is waiting. The
/// `OPENWRITR_ROLL_SECONDS` environment variable shortens it so the rolling
/// behaviour can be tried in minutes instead of hours.
const ROLL_SECONDS: usize = 3600;
/// A pause only hands out a part if at least this much was recorded since
/// the previous cut.
pub const MIN_PAUSE_SEGMENT_SAMPLES: usize = 5 * TARGET_RATE as usize;
/// A rolling cut looks for the quietest 100 ms in this trailing window …
const CUT_SEARCH_SAMPLES: usize = 10 * TARGET_RATE as usize;
/// … ending this far before the newest audio, so a track that lags the
/// others by a few hundred milliseconds never writes into released audio.
const CUT_GUARD_SAMPLES: usize = TARGET_RATE as usize;
const CUT_FRAME_SAMPLES: usize = TARGET_RATE as usize / 10;
/// Resync a track once it lags the wall clock by more than this.
const RESYNC_GAP_SAMPLES: usize = TARGET_RATE as usize / 2;

/// Recording buffer size step: ten minutes of 16 kHz samples (~19 MB).
const CHUNK_SAMPLES: usize = 10 * 60 * TARGET_RATE as usize;

/// The mixed recording, stored as fixed ten-minute chunks of `i16`.
///
/// One growing `Vec` would reallocate and copy hundreds of megabytes inside
/// the audio callback on a long meeting (stalling capture). Chunks are
/// allocated once and never moved; finished audio is detached from the front
/// (whole chunks are moved out, not copied) and released.
struct Mix {
    /// Chunk `i` holds global samples `(base + i) * CHUNK ..`.
    chunks: VecDeque<Vec<i16>>,
    /// Chunks already released from the front.
    base: usize,
    /// Global samples already handed out.
    taken: usize,
    /// Global high-water mark of written samples.
    len: usize,
}

impl Mix {
    fn new() -> Self {
        Self {
            chunks: VecDeque::new(),
            base: 0,
            taken: 0,
            len: 0,
        }
    }

    /// Samples recorded but not yet handed out.
    fn pending(&self) -> usize {
        self.len.saturating_sub(self.taken)
    }

    /// Sum `samples` (clamped to [-1, 1]) into the timeline at `position`,
    /// saturating instead of wrapping. Anything that falls before audio that
    /// was already handed out is dropped.
    fn add(&mut self, position: usize, samples: &[f32]) {
        let skip = self.taken.saturating_sub(position).min(samples.len());
        let position = position + skip;
        let samples = &samples[skip..];
        if samples.is_empty() {
            return;
        }
        let end = position + samples.len();
        while (self.base + self.chunks.len()) * CHUNK_SAMPLES < end {
            self.chunks.push_back(vec![0; CHUNK_SAMPLES]);
        }
        let (mut at, mut done) = (position, 0);
        while done < samples.len() {
            let offset = at % CHUNK_SAMPLES;
            let count = (CHUNK_SAMPLES - offset).min(samples.len() - done);
            let chunk = &mut self.chunks[at / CHUNK_SAMPLES - self.base];
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

    /// The quietest `frame`-sample stretch inside `from..to` (lowest energy,
    /// first one wins ties), as its centre sample.
    fn quietest_point(&self, from: usize, to: usize, frame: usize) -> Option<usize> {
        let from = from.max(self.taken);
        let to = to.min(self.len);
        let mut best = None;
        let mut quietest = u64::MAX;
        let mut position = from;
        while position + frame <= to {
            let mut energy = 0u64;
            let mut at = position;
            while at < position + frame {
                let offset = at % CHUNK_SAMPLES;
                let count = (CHUNK_SAMPLES - offset).min(position + frame - at);
                let chunk = &self.chunks[at / CHUNK_SAMPLES - self.base];
                energy += chunk[offset..offset + count]
                    .iter()
                    .map(|&s| (s as i64 * s as i64) as u64)
                    .sum::<u64>();
                at += count;
            }
            if energy < quietest {
                quietest = energy;
                best = Some(position + frame / 2);
            }
            position += frame;
        }
        best
    }

    /// Hand out `taken..until`. Whole chunks are *moved* out (no copy) and a
    /// partly used chunk is copied; chunks that are fully consumed are
    /// released. Cheap enough to do under the lock the audio callbacks use.
    fn detach(&mut self, until: usize) -> Vec<Vec<i16>> {
        let until = until.min(self.len);
        let mut parts = Vec::new();
        let mut at = self.taken;
        while at < until {
            let offset = at % CHUNK_SAMPLES;
            let count = (CHUNK_SAMPLES - offset).min(until - at);
            let index = at / CHUNK_SAMPLES - self.base;
            if offset == 0 && count == CHUNK_SAMPLES && index == 0 {
                if let Some(chunk) = self.chunks.pop_front() {
                    parts.push(chunk);
                    self.base += 1;
                }
            } else {
                parts.push(self.chunks[index][offset..offset + count].to_vec());
                if offset + count == CHUNK_SAMPLES && index == 0 {
                    self.chunks.pop_front();
                    self.base += 1;
                }
            }
            at += count;
        }
        self.taken = until.max(self.taken);
        parts
    }

    /// Everything not yet handed out, as f32.
    #[cfg(test)]
    fn into_f32(mut self) -> Vec<f32> {
        let until = self.len;
        pieces_to_f32(self.detach(until))
    }

    #[cfg(test)]
    fn to_vec(&self) -> Vec<i16> {
        (self.taken..self.len)
            .map(|at| self.chunks[at / CHUNK_SAMPLES - self.base][at % CHUNK_SAMPLES])
            .collect()
    }
}

/// Convert detached pieces to f32, releasing each piece as it is read.
fn pieces_to_f32(pieces: Vec<Vec<i16>>) -> Vec<f32> {
    let total = pieces.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(total);
    for piece in pieces {
        out.extend(piece.iter().map(|&s| s as f32 / 32_768.0));
    }
    out
}

fn roll_samples() -> usize {
    std::env::var("OPENWRITR_ROLL_SECONDS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|seconds| *seconds >= 10)
        .unwrap_or(ROLL_SECONDS)
        * TARGET_RATE as usize
}

/// A finished stretch of the meeting, ready for transcription.
pub struct Segment {
    /// 16 kHz mono.
    pub samples: Vec<f32>,
    /// Where this part starts on the meeting's timeline, in samples.
    pub start: usize,
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

    /// Whether enough audio is waiting to be handed out for transcription
    /// (one hour, or `OPENWRITR_ROLL_SECONDS`).
    pub fn roll_due(&self) -> bool {
        self.shared.mix.lock().pending() >= roll_samples()
    }

    /// Detach the finished part of the recording while capture continues.
    ///
    /// With `quiet_cut` the part ends at the quietest 100 ms shortly before
    /// the newest audio, so a word is rarely split between two parts. Without
    /// it (a pause: nothing more is being recorded) everything recorded so
    /// far is handed out. Returns `None` if less than `at_least` samples are
    /// waiting.
    pub fn take_segment(&mut self, quiet_cut: bool, at_least: usize) -> Option<Segment> {
        let (parts, start) = {
            let mut mix = self.shared.mix.lock();
            if mix.pending() < at_least.max(1) {
                return None;
            }
            let start = mix.taken;
            let until = if quiet_cut {
                let newest = mix.len.saturating_sub(CUT_GUARD_SAMPLES);
                let from = newest.saturating_sub(CUT_SEARCH_SAMPLES);
                mix.quietest_point(from, newest, CUT_FRAME_SAMPLES)
                    .unwrap_or(newest)
            } else {
                mix.len
            };
            if until <= start {
                return None;
            }
            (mix.detach(until), start)
        };
        // The conversion (tens of MB) happens outside the lock.
        Some(Segment {
            samples: pieces_to_f32(parts),
            start,
        })
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

    /// Stop capture and return the remaining (not yet handed out) 16 kHz
    /// mono audio with its start position on the meeting's timeline.
    pub fn finish(self) -> Segment {
        let Self {
            streams, shared, ..
        } = self;
        drop(streams);
        shared.level_x10000.store(0, Ordering::Relaxed);
        let (parts, start) = {
            let mut mix = shared.mix.lock();
            let start = mix.taken;
            let until = mix.len;
            (mix.detach(until), start)
        };
        Segment {
            samples: pieces_to_f32(parts),
            start,
        }
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
        assert_eq!(mix.len, 100);
        assert!(mix.to_vec().iter().all(|&s| (s - 16_383).abs() <= 2));
    }

    #[test]
    fn mix_spans_chunk_boundaries_without_gaps() {
        let mut mix = Mix::new();
        // 100 samples straddling the end of the first chunk.
        let start = CHUNK_SAMPLES - 40;
        mix.add(start, &[0.5; 100]);
        assert_eq!(mix.len, CHUNK_SAMPLES + 60);
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
    fn the_cap_is_twelve_hours_of_audio() {
        assert_eq!(MAX_MEETING_SECONDS, 12 * 3600);
        assert_eq!(MAX_SAMPLES, 691_200_000);
    }

    /// Value written at global sample `at`: distinct and recoverable, so a
    /// test can check exactly which samples came out where.
    fn marker(at: usize) -> f32 {
        ((at % 20_000) as f32 + 1.0) / 40_000.0
    }

    fn fill(mix: &mut Mix, from: usize, to: usize) {
        let samples: Vec<f32> = (from..to).map(marker).collect();
        mix.add(from, &samples);
    }

    fn expected_i16(at: usize) -> i16 {
        (marker(at).clamp(-1.0, 1.0) * 32_767.0) as i16
    }

    #[test]
    fn detach_hands_out_exactly_the_written_audio_and_releases_chunks() {
        let mut mix = Mix::new();
        // Two and a bit chunks.
        let total = 2 * CHUNK_SAMPLES + 5_000;
        fill(&mut mix, 0, total);
        assert_eq!(mix.chunks.len(), 3);

        // Cut in the middle of the second chunk.
        let cut = CHUNK_SAMPLES + 12_345;
        let first = pieces_flat(mix.detach(cut));
        assert_eq!(first.len(), cut);
        assert!((0..cut)
            .step_by(997)
            .all(|at| first[at] == expected_i16(at)));
        // The first chunk was moved out and released; the second is kept
        // because audio after the cut still lives in it.
        assert_eq!(mix.base, 1);
        assert_eq!(mix.chunks.len(), 2);
        assert_eq!(mix.taken, cut);
        assert_eq!(mix.pending(), total - cut);

        // The rest comes out with nothing lost or repeated.
        let rest = pieces_flat(mix.detach(total));
        assert_eq!(rest.len(), total - cut);
        assert!((0..rest.len())
            .step_by(1_009)
            .all(|i| rest[i] == expected_i16(cut + i)));
        assert_eq!(mix.base, 2);
        assert_eq!(mix.pending(), 0);
    }

    fn pieces_flat(pieces: Vec<Vec<i16>>) -> Vec<i16> {
        pieces.into_iter().flatten().collect()
    }

    #[test]
    fn writes_after_a_cut_keep_their_global_position() {
        let mut mix = Mix::new();
        fill(&mut mix, 0, 30_000);
        let _ = mix.detach(20_000);
        // A later write lands where the global timeline says, leaving the
        // samples in between silent.
        fill(&mut mix, 35_000, 50_000);
        let rest = mix.to_vec();
        assert_eq!(rest.len(), 30_000);
        assert_eq!(rest[0], expected_i16(20_000));
        assert_eq!(rest[9_999], expected_i16(29_999));
        assert_eq!(rest[12_000], 0, "the gap between writes stays silent");
        assert_eq!(rest[15_000], expected_i16(35_000));
        assert_eq!(rest[29_999], expected_i16(49_999));
    }

    #[test]
    fn late_samples_for_released_audio_are_dropped_not_misplaced() {
        let mut mix = Mix::new();
        fill(&mut mix, 0, 10_000);
        let _ = mix.detach(10_000);
        // A lagging track writes 500 samples that straddle the cut.
        let late: Vec<f32> = vec![0.5; 1_000];
        mix.add(9_500, &late);
        assert_eq!(mix.len, 10_500);
        let rest = mix.to_vec();
        assert_eq!(rest.len(), 500);
        assert!(rest.iter().all(|&s| s == 16_383));
    }

    #[test]
    fn quiet_point_is_found_in_the_silent_stretch() {
        let mut mix = Mix::new();
        let loud = vec![0.4f32; 60_000];
        mix.add(0, &loud);
        // 0.4 s of silence in the middle.
        let silence = vec![0.0f32; 6_400];
        let mut quiet = loud.clone();
        quiet[30_000..36_400].copy_from_slice(&silence);
        let mut mix2 = Mix::new();
        mix2.add(0, &quiet);
        let point = mix2
            .quietest_point(0, 60_000, CUT_FRAME_SAMPLES)
            .expect("a frame fits");
        assert!((30_000..36_400).contains(&point), "cut {point}");
        // Unchanged for the all-loud buffer: first frame wins.
        assert_eq!(
            mix.quietest_point(0, 60_000, CUT_FRAME_SAMPLES),
            Some(CUT_FRAME_SAMPLES / 2)
        );
    }

    #[test]
    fn the_roll_threshold_defaults_to_an_hour() {
        // (The override is read from the environment, which tests don't set.)
        if std::env::var_os("OPENWRITR_ROLL_SECONDS").is_none() {
            assert_eq!(roll_samples(), 3_600 * TARGET_RATE as usize);
        }
        assert_eq!(MIN_PAUSE_SEGMENT_SAMPLES, 5 * TARGET_RATE as usize);
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
