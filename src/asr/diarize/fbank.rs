//! Kaldi-compatible 80-bin log-mel filterbank features.
//!
//! The WeSpeaker speaker-embedding network was trained on features produced
//! by Kaldi / `torchaudio.compliance.kaldi.fbank` with: 25 ms Povey window,
//! 10 ms shift, DC removal, 0.97 pre-emphasis, no dither, 20 Hz – Nyquist
//! mel range, natural-log energies, and per-utterance mean normalization.
//! Matching that exactly matters: embeddings from slightly different
//! features cluster measurably worse.

use realfft::{num_complex::Complex, RealFftPlanner, RealToComplex};
use std::f64::consts::PI;
use std::sync::Arc;

pub const NUM_BINS: usize = 80;
const SAMPLE_RATE: f64 = 16_000.0;
const FRAME_LENGTH: usize = 400; // 25 ms
const FRAME_SHIFT: usize = 160; // 10 ms
const FFT_SIZE: usize = 512;
const LOW_FREQ: f64 = 20.0;
const PREEMPHASIS: f32 = 0.97;
/// Kaldi expects 16-bit-range samples; ours are in [-1, 1].
const INPUT_SCALE: f32 = 32_768.0;
/// `f32::EPSILON`, the floor Kaldi applies before taking the log.
const ENERGY_FLOOR: f32 = 1.192_092_9e-7;

pub struct Fbank {
    window: Vec<f32>,
    /// `NUM_BINS` rows of `FFT_SIZE / 2` weights.
    mel: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl Default for Fbank {
    fn default() -> Self {
        Self::new()
    }
}

fn mel_scale(frequency: f64) -> f64 {
    1127.0 * (1.0 + frequency / 700.0).ln()
}

impl Fbank {
    pub fn new() -> Self {
        // Povey window: a Hann window raised to the power 0.85.
        let window = (0..FRAME_LENGTH)
            .map(|i| {
                let hann = 0.5 - 0.5 * (2.0 * PI * i as f64 / (FRAME_LENGTH - 1) as f64).cos();
                hann.powf(0.85) as f32
            })
            .collect();

        let bins = FFT_SIZE / 2;
        let high_freq = SAMPLE_RATE / 2.0;
        let mel_low = mel_scale(LOW_FREQ);
        let mel_high = mel_scale(high_freq);
        let delta = (mel_high - mel_low) / (NUM_BINS + 1) as f64;
        let bin_width = SAMPLE_RATE / FFT_SIZE as f64;
        let mut mel = vec![0.0f32; NUM_BINS * bins];
        for b in 0..NUM_BINS {
            let left = mel_low + b as f64 * delta;
            let center = left + delta;
            let right = center + delta;
            for i in 0..bins {
                let m = mel_scale(bin_width * i as f64);
                let weight = if m > left && m <= center {
                    (m - left) / (center - left)
                } else if m > center && m < right {
                    (right - m) / (right - center)
                } else {
                    0.0
                };
                mel[b * bins + i] = weight as f32;
            }
        }

        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(FFT_SIZE);
        Self { window, mel, fft }
    }

    /// Number of frames for `samples` samples (Kaldi `snip_edges`).
    pub fn num_frames(samples: usize) -> usize {
        if samples < FRAME_LENGTH {
            0
        } else {
            1 + (samples - FRAME_LENGTH) / FRAME_SHIFT
        }
    }

    /// Row-major `frames x NUM_BINS` log-mel features, *without* mean
    /// normalization (see [`mean_normalize`]).
    pub fn compute(&self, samples: &[f32]) -> Vec<f32> {
        let frames = Self::num_frames(samples.len());
        let mut out = Vec::with_capacity(frames * NUM_BINS);
        let mut frame = vec![0.0f32; FRAME_LENGTH];
        let mut padded = self.fft.make_input_vec();
        let mut spectrum: Vec<Complex<f32>> = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        let bins = FFT_SIZE / 2;

        for f in 0..frames {
            let start = f * FRAME_SHIFT;
            for (dst, &src) in frame.iter_mut().zip(&samples[start..start + FRAME_LENGTH]) {
                *dst = src * INPUT_SCALE;
            }
            // DC removal.
            let mean = frame.iter().sum::<f32>() / FRAME_LENGTH as f32;
            frame.iter_mut().for_each(|x| *x -= mean);
            // Pre-emphasis (the first sample is differenced against itself).
            for i in (1..FRAME_LENGTH).rev() {
                frame[i] -= PREEMPHASIS * frame[i - 1];
            }
            frame[0] -= PREEMPHASIS * frame[0];

            padded.iter_mut().for_each(|x| *x = 0.0);
            for i in 0..FRAME_LENGTH {
                padded[i] = frame[i] * self.window[i];
            }
            if self
                .fft
                .process_with_scratch(&mut padded, &mut spectrum, &mut scratch)
                .is_err()
            {
                out.extend(std::iter::repeat(0.0).take(NUM_BINS));
                continue;
            }

            for b in 0..NUM_BINS {
                let weights = &self.mel[b * bins..(b + 1) * bins];
                let energy: f32 = spectrum[..bins]
                    .iter()
                    .zip(weights)
                    .map(|(c, w)| (c.re * c.re + c.im * c.im) * w)
                    .sum();
                out.push(energy.max(ENERGY_FLOOR).ln());
            }
        }
        out
    }
}

/// Subtract each bin's mean over time, in place (per-utterance CMN).
pub fn mean_normalize(features: &mut [f32]) {
    let frames = features.len() / NUM_BINS;
    if frames == 0 {
        return;
    }
    for b in 0..NUM_BINS {
        let mean = (0..frames).map(|f| features[f * NUM_BINS + b]).sum::<f32>() / frames as f32;
        for f in 0..frames {
            features[f * NUM_BINS + b] -= mean;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_signal(n: usize) -> Vec<f32> {
        (0..n)
            .map(|t| {
                let t = t as f64;
                (0.5 * (2.0 * PI * 440.0 * t / 16_000.0).sin()
                    + 0.25 * (2.0 * PI * 1000.0 * t / 16_000.0).sin()) as f32
            })
            .collect()
    }

    #[test]
    fn frame_count_follows_snip_edges() {
        assert_eq!(Fbank::num_frames(399), 0);
        assert_eq!(Fbank::num_frames(400), 1);
        assert_eq!(Fbank::num_frames(559), 1);
        assert_eq!(Fbank::num_frames(560), 2);
        assert_eq!(Fbank::num_frames(8_000), 48);
    }

    /// Reference values from `kaldi_native_fbank` (dither 0, 80 bins) for the
    /// signal in `test_signal(8000)`.
    #[test]
    fn matches_kaldi_native_fbank_reference() {
        let features = Fbank::new().compute(&test_signal(8_000));
        assert_eq!(features.len(), 48 * NUM_BINS);
        let frame0 = &features[..NUM_BINS];
        let frame47 = &features[47 * NUM_BINS..];
        let expected0 = [
            (0, 9.4038),
            (6, 12.6935),
            (12, 21.2193),
            (13, 24.4516),
            (14, 25.2017),
            (26, 24.3990),
            (27, 25.6676),
            (40, 7.2613),
            (79, -1.5233),
        ];
        let expected47 = [
            (0, 10.2737),
            (12, 21.2198),
            (13, 24.4514),
            (27, 25.6676),
            (79, -2.6793),
        ];
        for (bin, value) in expected0 {
            assert!(
                (frame0[bin] - value as f32).abs() < 0.05,
                "frame 0 bin {bin}: {} vs {value}",
                frame0[bin]
            );
        }
        for (bin, value) in expected47 {
            assert!(
                (frame47[bin] - value as f32).abs() < 0.05,
                "frame 47 bin {bin}: {} vs {value}",
                frame47[bin]
            );
        }
    }

    #[test]
    fn silence_hits_the_log_floor() {
        let features = Fbank::new().compute(&vec![0.0; 1_600]);
        let floor = ENERGY_FLOOR.ln();
        assert!(features.iter().all(|v| (v - floor).abs() < 1e-3));
    }

    #[test]
    fn mean_normalization_centers_every_bin() {
        let mut features: Vec<f32> = (0..5 * NUM_BINS).map(|i| (i % 7) as f32).collect();
        mean_normalize(&mut features);
        for b in 0..NUM_BINS {
            let mean: f32 = (0..5).map(|f| features[f * NUM_BINS + b]).sum::<f32>() / 5.0;
            assert!(mean.abs() < 1e-5);
        }
    }
}
