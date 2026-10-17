//! Agglomerative clustering of speaker embeddings.
//!
//! Average linkage on cosine distance, cut at a fixed threshold. Uses the
//! nearest-neighbour-chain algorithm, which is O(n^2) time and memory, so a
//! two-hour meeting (a couple of thousand embeddings) clusters instantly.

/// Cluster L2-normalized `embeddings`; returns one label per embedding,
/// numbered from 0 in order of first appearance.
pub fn cluster(embeddings: &[Vec<f32>], threshold: f32) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    let mut distance = vec![0.0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dot: f32 = embeddings[i]
                .iter()
                .zip(&embeddings[j])
                .map(|(a, b)| a * b)
                .sum();
            let d = (1.0 - dot).max(0.0);
            distance[i * n + j] = d;
            distance[j * n + i] = d;
        }
    }

    let mut size = vec![1.0f32; n];
    let mut active = vec![true; n];
    let mut parent: Vec<usize> = (0..n).collect();
    let mut chain: Vec<usize> = Vec::new();
    let mut remaining = n;

    while remaining > 1 {
        if chain.is_empty() {
            if let Some(first) = active.iter().position(|&a| a) {
                chain.push(first);
            }
        }
        loop {
            let a = *chain.last().expect("chain is never empty here");
            let previous = (chain.len() >= 2).then(|| chain[chain.len() - 2]);
            let mut best = usize::MAX;
            let mut best_distance = f32::INFINITY;
            for k in 0..n {
                if !active[k] || k == a {
                    continue;
                }
                let d = distance[a * n + k];
                if d < best_distance || (d == best_distance && Some(k) == previous) {
                    best = k;
                    best_distance = d;
                }
            }
            if Some(best) == previous {
                chain.pop();
                chain.pop();
                let b = best;
                if best_distance <= threshold {
                    let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                    parent[rb] = ra;
                }
                // Lance-Williams update for average linkage.
                let (sa, sb) = (size[a], size[b]);
                for k in 0..n {
                    if active[k] && k != a && k != b {
                        let merged =
                            (sa * distance[a * n + k] + sb * distance[b * n + k]) / (sa + sb);
                        distance[a * n + k] = merged;
                        distance[k * n + a] = merged;
                    }
                }
                size[a] = sa + sb;
                active[b] = false;
                remaining -= 1;
                break;
            }
            chain.push(best);
        }
    }

    let mut labels = vec![usize::MAX; n];
    let mut next = 0;
    let mut by_root = std::collections::HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        let label = *by_root.entry(root).or_insert_with(|| {
            let l = next;
            next += 1;
            l
        });
        labels[i] = label;
    }
    labels
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: &[f32]) -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn separates_two_directions() {
        let embeddings = vec![
            unit(&[1.0, 0.05, 0.0]),
            unit(&[0.0, 1.0, 0.05]),
            unit(&[1.0, 0.0, 0.1]),
            unit(&[0.05, 1.0, 0.0]),
            unit(&[1.0, 0.1, 0.0]),
        ];
        assert_eq!(cluster(&embeddings, 0.5), vec![0, 1, 0, 1, 0]);
    }

    #[test]
    fn threshold_controls_the_number_of_speakers() {
        let embeddings = vec![unit(&[1.0, 0.0]), unit(&[0.8, 0.6])];
        // Cosine distance between them is 0.2.
        assert_eq!(cluster(&embeddings, 0.1), vec![0, 1]);
        assert_eq!(cluster(&embeddings, 0.3), vec![0, 0]);
    }

    #[test]
    fn handles_trivial_inputs() {
        assert!(cluster(&[], 0.5).is_empty());
        assert_eq!(cluster(&[unit(&[1.0, 0.0])], 0.5), vec![0]);
    }

    #[test]
    fn scales_to_thousands_of_embeddings() {
        let embeddings: Vec<Vec<f32>> = (0..1_500)
            .map(|i| {
                let angle = if i % 3 == 0 {
                    0.0
                } else if i % 3 == 1 {
                    2.0
                } else {
                    4.0
                };
                let jitter = (i as f32 * 0.37).sin() * 0.02;
                unit(&[(angle + jitter as f32).cos(), (angle + jitter as f32).sin()])
            })
            .collect();
        let labels = cluster(&embeddings, 0.3);
        assert_eq!(labels.iter().copied().max(), Some(2));
    }
}
