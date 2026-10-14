//! Decoding of pyannote segmentation-3.0 output.
//!
//! The network sees 10 s of 16 kHz audio and emits, for each of 589 frames,
//! log-probabilities over 7 "powerset" classes: no speech, each of 3 local
//! speakers alone, and each of the 3 pairs.

pub const WINDOW_SAMPLES: usize = 160_000;
/// Samples between consecutive output frames (receptive-field shift).
pub const FRAME_SHIFT: usize = 270;
pub const CLASSES: usize = 7;
pub const LOCAL_SPEAKERS: usize = 3;

const POWERSET: [&[usize]; CLASSES] = [&[], &[0], &[1], &[2], &[0, 1], &[0, 2], &[1, 2]];

/// Per-frame activity of the three local speakers from `logits`
/// (`frames x CLASSES`, row-major).
pub fn decode_powerset(logits: &[f32]) -> Vec<[bool; LOCAL_SPEAKERS]> {
    logits
        .chunks_exact(CLASSES)
        .map(|row| {
            let class = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(index, _)| index)
                .unwrap_or(0);
            let mut active = [false; LOCAL_SPEAKERS];
            for &speaker in POWERSET[class] {
                active[speaker] = true;
            }
            active
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_every_powerset_class() {
        let mut logits = Vec::new();
        for class in 0..CLASSES {
            let mut row = vec![-5.0f32; CLASSES];
            row[class] = -0.01;
            logits.extend(row);
        }
        let decoded = decode_powerset(&logits);
        assert_eq!(decoded[0], [false, false, false]);
        assert_eq!(decoded[1], [true, false, false]);
        assert_eq!(decoded[2], [false, true, false]);
        assert_eq!(decoded[3], [false, false, true]);
        assert_eq!(decoded[4], [true, true, false]);
        assert_eq!(decoded[5], [true, false, true]);
        assert_eq!(decoded[6], [false, true, true]);
    }
}
