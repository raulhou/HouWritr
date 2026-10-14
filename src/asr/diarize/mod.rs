//! Speaker diarization: who spoke when.
//!
//! Pipeline (the same shape as pyannote's, running on the CPU through ONNX
//! Runtime):
//!
//! 1. **Segmentation** – pyannote segmentation-3.0 looks at 10 s windows and
//!    marks, per frame, which of up to three *local* speakers is active.
//! 2. **Embedding** – a WeSpeaker ResNet34 network turns each local speaker's
//!    clean audio in a window into a 256-d voice print.
//! 3. **Clustering** – agglomerative clustering links local speakers across
//!    windows into global speakers.
//! 4. **Turns** – frame activity is mapped to global speakers and tidied.
//!
//! The result is a list of speaker turns in samples; transcription then runs
//! per turn, so no word-level timestamps are needed from the ASR engines.

mod cluster;
mod fbank;
mod segment;

use super::ort_helpers::{build_cpu_session, OrtResultExt};
use anyhow::{anyhow, bail, Context, Result};
use ndarray::Array3;
use ort::session::Session;
use ort::value::Value;
use segment::{FRAME_SHIFT, LOCAL_SPEAKERS, WINDOW_SAMPLES};
use std::path::Path;
use std::time::Instant;
use tracing::info;

pub const SEGMENTATION_MODEL: &str = "segmentation-3.0.onnx";
pub const EMBEDDING_MODEL: &str = "wespeaker-voxceleb-resnet34-LM.onnx";

/// Cosine-distance cut for average-linkage clustering. 0.6 recovered the
/// right speaker count on all the reference clips (0.7 merged speakers on
/// one of them, 0.5 split one speaker in two on another).
const CLUSTER_THRESHOLD: f32 = 0.6;
/// A local speaker needs this much clean speech in a window to be embedded.
const MIN_EMBED_SAMPLES: usize = 8_000;
/// Longer speech adds cost, not accuracy.
const MAX_EMBED_SAMPLES: usize = 6 * 16_000;
/// Turns shorter than this are dropped as blips.
const MIN_TURN_SAMPLES: usize = 4_000;
/// Same-speaker turns separated by less than this are joined.
const MAX_GAP_SAMPLES: usize = 8_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Turn {
    pub start: usize,
    pub end: usize,
    /// Zero-based, numbered in order of first appearance.
    pub speaker: usize,
}

pub struct Diarizer {
    segmentation: Session,
    embedding: Session,
    fbank: fbank::Fbank,
}

impl Diarizer {
    pub fn load(dir: &Path) -> Result<Self> {
        let started = Instant::now();
        let segmentation = build_cpu_session(&dir.join(SEGMENTATION_MODEL))?;
        let embedding = build_cpu_session(&dir.join(EMBEDDING_MODEL))?;
        info!(
            ms = started.elapsed().as_millis() as u64,
            "diarization models loaded"
        );
        Ok(Self {
            segmentation,
            embedding,
            fbank: fbank::Fbank::new(),
        })
    }

    /// Diarize 16 kHz mono `samples`.
    ///
    /// `progress(done, total)` counts 10 s windows. `cancelled` is polled
    /// between windows; a cancelled run returns what it has as an error.
    pub fn diarize(
        &mut self,
        samples: &[f32],
        cancelled: &dyn Fn() -> bool,
        progress: &mut dyn FnMut(usize, usize),
    ) -> Result<Vec<Turn>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let started = Instant::now();
        let windows = samples.len().div_ceil(WINDOW_SAMPLES);
        progress(0, windows);

        let mut activity: Vec<Vec<[bool; LOCAL_SPEAKERS]>> = Vec::with_capacity(windows);
        // (window, local speaker, embedding)
        let mut items: Vec<(usize, usize, Vec<f32>)> = Vec::new();

        for window in 0..windows {
            if cancelled() {
                bail!("diarization cancelled");
            }
            let start = window * WINDOW_SAMPLES;
            let end = (start + WINDOW_SAMPLES).min(samples.len());
            let mut chunk = vec![0.0f32; WINDOW_SAMPLES];
            chunk[..end - start].copy_from_slice(&samples[start..end]);

            let frames = self.segment(&chunk)?;
            for speaker in 0..LOCAL_SPEAKERS {
                if let Some(audio) = speaker_audio(&chunk, &frames, speaker, end - start) {
                    let embedding = self.embed(&audio)?;
                    items.push((window, speaker, embedding));
                }
            }
            activity.push(frames);
            progress(window + 1, windows);
        }

        let embeddings: Vec<Vec<f32>> = items.iter().map(|(_, _, e)| e.clone()).collect();
        let labels = cluster::cluster(&embeddings, CLUSTER_THRESHOLD);
        let mut local_to_global = vec![[None; LOCAL_SPEAKERS]; windows];
        for ((window, speaker, _), label) in items.iter().zip(&labels) {
            local_to_global[*window][*speaker] = Some(*label);
        }

        let turns = build_turns(&activity, &local_to_global, samples.len());
        info!(
            windows,
            embeddings = items.len(),
            speakers = speaker_count(&turns),
            turns = turns.len(),
            ms = started.elapsed().as_millis() as u64,
            "diarized"
        );
        Ok(turns)
    }

    fn segment(&mut self, chunk: &[f32]) -> Result<Vec<[bool; LOCAL_SPEAKERS]>> {
        let input = Array3::from_shape_vec((1, 1, WINDOW_SAMPLES), chunk.to_vec())?;
        let outputs = self
            .segmentation
            .run(ort::inputs!["x" => Value::from_array(input)?])
            .ortx()
            .context("run segmentation")?;
        let logits = outputs
            .get("y")
            .ok_or_else(|| anyhow!("segmentation output 'y' is missing"))?
            .try_extract_array::<f32>()
            .ortx()?;
        let flat: Vec<f32> = logits.iter().copied().collect();
        Ok(segment::decode_powerset(&flat))
    }

    fn embed(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        let mut features = self.fbank.compute(audio);
        fbank::mean_normalize(&mut features);
        let frames = features.len() / fbank::NUM_BINS;
        if frames == 0 {
            bail!("not enough audio to embed");
        }
        let input = Array3::from_shape_vec((1, frames, fbank::NUM_BINS), features)?;
        let outputs = self
            .embedding
            .run(ort::inputs!["feats" => Value::from_array(input)?])
            .ortx()
            .context("run speaker embedding")?;
        let embedding = outputs
            .get("embs")
            .ok_or_else(|| anyhow!("embedding output 'embs' is missing"))?
            .try_extract_array::<f32>()
            .ortx()?;
        let mut vector: Vec<f32> = embedding.iter().copied().collect();
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            vector.iter_mut().for_each(|x| *x /= norm);
        }
        Ok(vector)
    }
}

pub fn speaker_count(turns: &[Turn]) -> usize {
    turns.iter().map(|turn| turn.speaker + 1).max().unwrap_or(0)
}

/// The audio of one local speaker in a window: frames where they are the
/// only active speaker (falling back to all their frames when that is too
/// little), or `None` when there is not enough speech to identify them.
fn speaker_audio(
    chunk: &[f32],
    frames: &[[bool; LOCAL_SPEAKERS]],
    speaker: usize,
    valid_samples: usize,
) -> Option<Vec<f32>> {
    let collect = |only_alone: bool| -> Vec<f32> {
        let mut audio = Vec::new();
        for (index, active) in frames.iter().enumerate() {
            if !active[speaker] {
                continue;
            }
            if only_alone && active.iter().filter(|&&a| a).count() != 1 {
                continue;
            }
            let start = index * FRAME_SHIFT;
            let end = (start + FRAME_SHIFT).min(valid_samples).min(chunk.len());
            if start < end {
                audio.extend_from_slice(&chunk[start..end]);
            }
        }
        audio
    };
    let mut audio = collect(true);
    if audio.len() < MIN_EMBED_SAMPLES {
        audio = collect(false);
    }
    if audio.len() < MIN_EMBED_SAMPLES {
        return None;
    }
    audio.truncate(MAX_EMBED_SAMPLES);
    Some(audio)
}

/// Frame activity + local→global labels → tidy, ordered speaker turns.
fn build_turns(
    activity: &[Vec<[bool; LOCAL_SPEAKERS]>],
    local_to_global: &[[Option<usize>; LOCAL_SPEAKERS]],
    total_samples: usize,
) -> Vec<Turn> {
    let mut runs: Vec<Turn> = Vec::new();
    for (window, frames) in activity.iter().enumerate() {
        // Overlapped speech goes to whoever talks most in this window.
        let mut weight = [0usize; LOCAL_SPEAKERS];
        for active in frames {
            for s in 0..LOCAL_SPEAKERS {
                weight[s] += usize::from(active[s]);
            }
        }
        for (index, active) in frames.iter().enumerate() {
            let chosen = (0..LOCAL_SPEAKERS)
                .filter(|&s| active[s] && local_to_global[window][s].is_some())
                .max_by_key(|&s| weight[s]);
            let Some(local) = chosen else { continue };
            let Some(speaker) = local_to_global[window][local] else {
                continue;
            };
            let start = window * WINDOW_SAMPLES + index * FRAME_SHIFT;
            let end = (start + FRAME_SHIFT).min(total_samples);
            if start >= end {
                continue;
            }
            match runs.last_mut() {
                Some(last) if last.speaker == speaker && last.end == start => last.end = end,
                _ => runs.push(Turn {
                    start,
                    end,
                    speaker,
                }),
            }
        }
    }
    tidy_turns(runs)
}

/// Relabel by first appearance, drop blips, and join near-adjacent turns of
/// the same speaker.
fn tidy_turns(turns: Vec<Turn>) -> Vec<Turn> {
    let mut order: Vec<usize> = Vec::new();
    let mut tidy: Vec<Turn> = Vec::new();
    for turn in turns {
        if turn.end - turn.start < MIN_TURN_SAMPLES {
            continue;
        }
        let speaker = match order.iter().position(|&s| s == turn.speaker) {
            Some(position) => position,
            None => {
                order.push(turn.speaker);
                order.len() - 1
            }
        };
        match tidy.last_mut() {
            Some(last) if last.speaker == speaker && turn.start - last.end <= MAX_GAP_SAMPLES => {
                last.end = turn.end;
            }
            _ => tidy.push(Turn {
                start: turn.start,
                end: turn.end,
                speaker,
            }),
        }
    }
    tidy
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(start: usize, end: usize, speaker: usize) -> Turn {
        Turn {
            start,
            end,
            speaker,
        }
    }

    #[test]
    fn tidy_relabels_by_first_appearance_and_joins_gaps() {
        let tidy = tidy_turns(vec![
            turn(0, 16_000, 5),
            turn(20_000, 36_000, 5),   // 4000-sample gap: joined
            turn(80_000, 96_000, 2),   // new speaker -> 1
            turn(100_000, 101_000, 9), // blip: dropped
            turn(120_000, 136_000, 5), // back to speaker 0
        ]);
        assert_eq!(
            tidy,
            vec![
                turn(0, 36_000, 0),
                turn(80_000, 96_000, 1),
                turn(120_000, 136_000, 0),
            ]
        );
        assert_eq!(speaker_count(&tidy), 2);
    }

    #[test]
    fn large_gaps_keep_turns_apart() {
        let tidy = tidy_turns(vec![turn(0, 16_000, 0), turn(40_000, 56_000, 0)]);
        assert_eq!(tidy.len(), 2);
    }

    #[test]
    fn turns_follow_activity_and_pick_the_dominant_overlapped_speaker() {
        // One window: speaker 0 for 100 frames, then 0+1 overlapped for 20,
        // then speaker 1 for 100 frames. Local 0 -> global 7, local 1 -> 3.
        let mut frames = vec![[false; LOCAL_SPEAKERS]; 220];
        for f in frames.iter_mut().take(100) {
            f[0] = true;
        }
        for f in frames.iter_mut().skip(100).take(20) {
            f[0] = true;
            f[1] = true;
        }
        for f in frames.iter_mut().skip(120) {
            f[1] = true;
        }
        let mapping = vec![[Some(7), Some(3), None]];
        let turns = build_turns(&[frames], &mapping, WINDOW_SAMPLES);
        assert_eq!(speaker_count(&turns), 2);
        assert_eq!(turns[0].speaker, 0);
        assert_eq!(turns[1].speaker, 1);
        // Both locals talk 120 frames; the overlap goes to the later max.
        assert_eq!(turns[0].start, 0);
        assert_eq!(turns.last().unwrap().end, 220 * FRAME_SHIFT);
    }

    #[test]
    fn speakers_without_embeddings_produce_no_turns() {
        let frames = vec![[true, false, false]; 100];
        let turns = build_turns(&[frames], &[[None, None, None]], WINDOW_SAMPLES);
        assert!(turns.is_empty());
    }

    #[test]
    fn speaker_audio_prefers_clean_frames_and_needs_enough_speech() {
        let chunk = vec![0.1f32; WINDOW_SAMPLES];
        // 40 clean frames (10 800 samples) for speaker 0.
        let mut frames = vec![[false; LOCAL_SPEAKERS]; 100];
        for f in frames.iter_mut().take(40) {
            f[0] = true;
        }
        assert_eq!(
            speaker_audio(&chunk, &frames, 0, WINDOW_SAMPLES).map(|a| a.len()),
            Some(40 * FRAME_SHIFT)
        );
        // Too little speech.
        assert!(speaker_audio(&chunk, &frames, 1, WINDOW_SAMPLES).is_none());
    }
}
