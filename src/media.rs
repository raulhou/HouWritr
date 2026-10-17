//! Offline media input: decode an audio/video file to 16 kHz mono f32, and
//! the streaming downsampler shared with meeting capture.
//!
//! Decoding is pure Rust (symphonia) so it works identically on x64 and
//! arm64 and needs no codec DLLs. Everything is streamed through
//! [`Downsampler`] packet by packet, so a multi-hour file never holds a
//! full-rate copy in memory.

use anyhow::{anyhow, bail, Context, Result};
use std::ffi::OsString;
use std::fs::File;
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub const TARGET_RATE: u32 = 16_000;

/// Hard cap so a mislabelled or enormous file cannot exhaust memory
/// (5 hours of 16 kHz f32 mono is ~1.15 GB). Matches the meeting recording
/// limit so anything you can record you can also import.
const MAX_FILE_SECONDS: usize = 5 * 3600;

/// Centre of the quietest `frame`-sample stretch fully inside `range`
/// (lowest energy, first one wins ties), or `None` if no frame fits.
pub fn quietest_point(
    samples: &[f32],
    range: std::ops::Range<usize>,
    frame: usize,
) -> Option<usize> {
    let mut best = None;
    let mut quietest = f32::MAX;
    let mut position = range.start;
    while position + frame <= range.end {
        let energy: f32 = samples[position..position + frame]
            .iter()
            .map(|sample| sample * sample)
            .sum();
        if energy < quietest {
            quietest = energy;
            best = Some(position + frame / 2);
        }
        position += frame;
    }
    best
}

/// Split `samples` into consecutive windows of at most `max_seconds`, each cut
/// at the quietest 100 ms in the trailing `search_seconds` so a word is rarely
/// bisected. The windows tile the input exactly.
pub fn plan_windows(
    samples: &[f32],
    sample_rate: u32,
    max_seconds: usize,
    search_seconds: usize,
) -> Vec<std::ops::Range<usize>> {
    let rate = sample_rate.max(1) as usize;
    let max_len = (rate * max_seconds).max(1);
    let search = (rate * search_seconds).min(max_len.saturating_sub(1));
    let frame = (rate / 10).max(1);
    let mut windows = Vec::new();
    let mut start = 0;
    while start < samples.len() {
        if samples.len() - start <= max_len {
            windows.push(start..samples.len());
            break;
        }
        let hard_end = start + max_len;
        let cut = quietest_point(samples, hard_end - search..hard_end, frame).unwrap_or(hard_end);
        windows.push(start..cut);
        start = cut;
    }
    windows
}

/// Streaming mono downsampler (box filter over fractional input spans).
///
/// Good enough for speech recognition, allocation-free per sample, and
/// stateful across calls so it can be fed from audio callbacks. Upsampling
/// (rare, e.g. 8 kHz telephony files) falls back to sample-and-hold.
pub struct Downsampler {
    step: f64,
    cursor: f64,
    boundary: f64,
    acc: f64,
    weight: f64,
}

impl Downsampler {
    pub fn new(input_rate: u32, output_rate: u32) -> Self {
        let step = input_rate.max(1) as f64 / output_rate.max(1) as f64;
        Self {
            step,
            cursor: 0.0,
            boundary: step,
            acc: 0.0,
            weight: 0.0,
        }
    }

    pub fn push(&mut self, input: &[f32], out: &mut Vec<f32>) {
        for &sample in input {
            let end = self.cursor + 1.0;
            let mut lo = self.cursor;
            while lo < end {
                let hi = end.min(self.boundary);
                let w = hi - lo;
                self.acc += sample as f64 * w;
                self.weight += w;
                lo = hi;
                if hi >= self.boundary {
                    if self.weight > 0.0 {
                        out.push((self.acc / self.weight) as f32);
                    }
                    self.acc = 0.0;
                    self.weight = 0.0;
                    self.boundary += self.step;
                }
            }
            self.cursor = end;
        }
    }
}

/// Decode `path` to 16 kHz mono f32. Video containers work as long as their
/// audio track uses a supported codec (AAC, MP3, FLAC, Vorbis, PCM, ...).
pub fn decode_file(path: &Path) -> Result<Vec<f32>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|error| anyhow!("unsupported or unreadable media file: {error}"))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|track| track.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("file has no decodable audio track"))?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|error| anyhow!("unsupported audio codec: {error}"))?;

    let mut output = Vec::new();
    let mut resampler: Option<(u32, Downsampler)> = None;
    let mut interleaved: Option<SampleBuffer<f32>> = None;
    let mut mono = Vec::new();
    let max_samples = MAX_FILE_SECONDS * TARGET_RATE as usize;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(error) => return Err(anyhow!("read media packet: {error}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            // A corrupt packet is skippable; the rest of the file is fine.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(error) => return Err(anyhow!("decode audio: {error}")),
        };
        let spec = *decoded.spec();
        let channels = spec.channels.count().max(1);
        let needed = decoded.capacity() as u64;
        if interleaved.as_ref().map_or(true, |buffer| {
            (buffer.capacity() as u64) < needed * channels as u64
        }) {
            interleaved = Some(SampleBuffer::<f32>::new(needed, spec));
        }
        let buffer = interleaved.as_mut().expect("buffer was just created");
        buffer.copy_interleaved_ref(decoded);

        mono.clear();
        mono.extend(
            buffer
                .samples()
                .chunks_exact(channels)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32),
        );
        if resampler
            .as_ref()
            .map_or(true, |(rate, _)| *rate != spec.rate)
        {
            resampler = Some((spec.rate, Downsampler::new(spec.rate, TARGET_RATE)));
        }
        if let Some((_, downsampler)) = resampler.as_mut() {
            downsampler.push(&mono, &mut output);
        }
        if output.len() > max_samples {
            bail!(
                "file is longer than the supported {} hours",
                MAX_FILE_SECONDS / 3600
            );
        }
    }

    if output.is_empty() {
        bail!("file contains no audio");
    }
    Ok(output)
}

/// First free path of the form `name.ext`, `name (1).ext`, `name (2).ext`...
pub fn unique_path(path: PathBuf) -> PathBuf {
    if !path.exists() {
        return path;
    }
    let stem = path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| "transcript".into());
    let extension = path
        .extension()
        .map(|extension| extension.to_string_lossy().into_owned());
    let parent = path.parent().map(Path::to_path_buf).unwrap_or_default();
    for index in 1..10_000 {
        let name = match &extension {
            Some(extension) => format!("{stem} ({index}).{extension}"),
            None => format!("{stem} ({index})"),
        };
        let candidate = parent.join(name);
        if !candidate.exists() {
            return candidate;
        }
    }
    path
}

/// Sentences per paragraph when a transcript has no paragraph structure yet
/// (raw, un-cleaned output is one run-on string).
const SENTENCES_PER_PARAGRAPH: usize = 4;

/// Render a transcript as a Markdown document: `# title`, an italic details
/// line, then paragraphs. Text that already contains paragraph breaks (AI
/// cleanup output) is kept as is; otherwise sentences are grouped so the file
/// is readable.
pub fn to_markdown(title: &str, details: &str, body: &str) -> String {
    let body = body.trim();
    let paragraphs = if body.contains("\n\n") {
        body.to_string()
    } else {
        paragraphize(body)
    };
    format!("# {title}\n\n*{details}*\n\n{paragraphs}\n")
}

fn paragraphize(text: &str) -> String {
    let mut paragraphs: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut sentences = 0;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        current.push(c);
        let ends_sentence = matches!(c, '.' | '!' | '?' | '。' | '！' | '？')
            && chars.peek().map_or(true, |next| next.is_whitespace());
        if ends_sentence {
            sentences += 1;
            if sentences >= SENTENCES_PER_PARAGRAPH {
                paragraphs.push(current.trim().to_string());
                current.clear();
                sentences = 0;
            }
        }
    }
    if !current.trim().is_empty() {
        paragraphs.push(current.trim().to_string());
    }
    paragraphs.join("\n\n")
}

/// Standard Windows "Open" dialog. Blocks until dismissed, so call it from a
/// worker thread, never from the tray event loop.
pub fn pick_audio_file() -> Option<PathBuf> {
    use windows::core::{PCWSTR, PWSTR};
    use windows::Win32::UI::Controls::Dialogs::{
        GetOpenFileNameW, OFN_EXPLORER, OFN_FILEMUSTEXIST, OFN_NOCHANGEDIR, OFN_PATHMUSTEXIST,
        OPENFILENAMEW,
    };

    let wide = |text: &str| -> Vec<u16> {
        std::ffi::OsStr::new(text)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect()
    };
    // Filter pairs are NUL-separated and the list ends with a double NUL;
    // `wide` supplies the final NUL.
    let filter = wide(
        "Audio and video files\0*.wav;*.mp3;*.m4a;*.aac;*.flac;*.ogg;*.mp4;*.m4v;*.mov;*.mkv\0All files\0*.*\0",
    );
    let title = wide("Transcribe an audio or video file");
    let mut file_buffer = vec![0u16; 32_768];

    let mut dialog = OPENFILENAMEW {
        lStructSize: std::mem::size_of::<OPENFILENAMEW>() as u32,
        lpstrFilter: PCWSTR(filter.as_ptr()),
        lpstrFile: PWSTR(file_buffer.as_mut_ptr()),
        nMaxFile: file_buffer.len() as u32,
        lpstrTitle: PCWSTR(title.as_ptr()),
        Flags: OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR,
        ..Default::default()
    };
    // SAFETY: `dialog` and the buffers it points at outlive the call.
    let chosen = unsafe { GetOpenFileNameW(&mut dialog) }.as_bool();
    if !chosen {
        return None;
    }
    let len = file_buffer
        .iter()
        .position(|&unit| unit == 0)
        .unwrap_or(file_buffer.len());
    Some(PathBuf::from(OsString::from_wide(&file_buffer[..len])))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downsampler_preserves_a_constant_signal_and_ratio() {
        let mut down = Downsampler::new(48_000, 16_000);
        let mut out = Vec::new();
        down.push(&vec![0.5; 48_000], &mut out);
        assert_eq!(out.len(), 16_000);
        assert!(out.iter().all(|sample| (sample - 0.5).abs() < 1e-5));
    }

    #[test]
    fn downsampler_is_stateful_across_arbitrary_block_sizes() {
        let input: Vec<f32> = (0..44_100).map(|n| ((n % 100) as f32) / 100.0).collect();
        let mut whole = Vec::new();
        Downsampler::new(44_100, 16_000).push(&input, &mut whole);
        let mut split = Vec::new();
        let mut down = Downsampler::new(44_100, 16_000);
        for block in input.chunks(333) {
            down.push(block, &mut split);
        }
        assert_eq!(whole.len(), split.len());
        for (a, b) in whole.iter().zip(&split) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn markdown_groups_runon_sentences_into_paragraphs() {
        let body = "One. Two. Three. Four. Five. Six.";
        let md = to_markdown("Talk", "Transcribed by HouWritr", body);
        assert!(md.starts_with("# Talk\n\n*Transcribed by HouWritr*\n\n"));
        assert!(md.contains("One. Two. Three. Four.\n\nFive. Six.\n"));
    }

    #[test]
    fn markdown_keeps_existing_paragraphs() {
        let body = "First para.\n\nSecond para.";
        let md = to_markdown("T", "d", body);
        assert!(md.ends_with("First para.\n\nSecond para.\n"));
    }

    #[test]
    fn plan_windows_tile_the_input_and_respect_the_limit() {
        // 100 "seconds" at 100 Hz with a limit of 30 s: every window <= 30 s.
        let samples: Vec<f32> = (0..10_000).map(|i| ((i % 17) as f32) / 17.0).collect();
        let windows = plan_windows(&samples, 100, 30, 4);
        assert_eq!(windows.first().unwrap().start, 0);
        assert_eq!(windows.last().unwrap().end, samples.len());
        for pair in windows.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
        }
        assert!(windows.iter().all(|w| w.len() <= 3_000 && !w.is_empty()));
        assert!(windows.len() >= 4);
        // Shorter than the limit stays whole; empty stays empty.
        assert_eq!(plan_windows(&samples[..500], 100, 30, 4), vec![0..500]);
        assert!(plan_windows(&[], 100, 30, 4).is_empty());
    }

    #[test]
    fn plan_windows_cut_in_the_silence_before_the_limit() {
        let rate = 1_000;
        let mut samples = vec![0.5f32; 70_000];
        // Silence at 28.0 s..28.4 s, inside the 4 s search before 30 s.
        for s in &mut samples[28_000..28_400] {
            *s = 0.0;
        }
        let windows = plan_windows(&samples, rate, 30, 4);
        assert!((28_000..28_400).contains(&windows[0].end));
        // A 5-minute limit works the same way.
        let long = plan_windows(&vec![0.1f32; 700_000], rate, 300, 10);
        assert!(long.iter().all(|w| w.len() <= 300_000));
    }

    #[test]
    fn file_cap_matches_the_meeting_recording_cap() {
        assert_eq!(MAX_FILE_SECONDS, 5 * 3600);
    }

    #[test]
    fn unique_path_appends_a_counter_for_existing_files() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("talk.txt");
        assert_eq!(unique_path(base.clone()), base);
        std::fs::write(&base, "x").unwrap();
        assert_eq!(unique_path(base.clone()), dir.path().join("talk (1).txt"));
    }
}
