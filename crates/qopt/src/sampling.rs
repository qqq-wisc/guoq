//! Weighted sampling used to prune the transformation set during search.

use rand::seq::SliceRandom;
use rand::Rng;

/// Softmax over `weights` at `temperature`.
///
/// The maximum is subtracted before exponentiating, so a large weight cannot overflow.
pub fn softmax(weights: &[f64], temperature: f64) -> Vec<f64> {
    if weights.is_empty() {
        return Vec::new();
    }
    let t = if temperature > 0.0 { temperature } else { 1.0 };
    let max = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probs: Vec<f64> = weights.iter().map(|w| ((w - max) / t).exp()).collect();
    let sum: f64 = probs.iter().sum();
    if sum > 0.0 && sum.is_finite() {
        for p in &mut probs {
            *p /= sum;
        }
    } else {
        let uniform = 1.0 / probs.len() as f64;
        probs.iter_mut().for_each(|p| *p = uniform);
    }
    probs
}

/// Draw one index from a probability distribution.
pub fn sample_index<R: Rng + ?Sized>(distribution: &[f64], rng: &mut R) -> Option<usize> {
    if distribution.is_empty() {
        return None;
    }
    let r: f64 = rng.gen();
    let mut acc = 0.0;
    for (i, p) in distribution.iter().enumerate() {
        acc += p;
        if r <= acc {
            return Some(i);
        }
    }
    Some(distribution.len() - 1)
}

/// Draw `n` distinct indices from `0..weights.len()`, favouring higher weights.
///
/// At `temperature == 0` this takes the `n` highest-weighted indices; above zero it
/// samples without replacement under a softmax.
///
/// # Why without replacement matters
///
/// Sampling *with* replacement quietly narrows the search: at temperature 0 it selects
/// the single highest-weighted rule `n` times over, and above zero it still repeats
/// draws, so the search explores far fewer distinct rules per iteration than it was
/// configured to.
/// Draw `k` distinct indices from `0..n`, uniformly.
///
/// Kept separate from [`sample_distinct`] because a uniform draw needs no weights, and
/// routing it through the weighted path was expensive out of all proportion to the work:
/// with the shipped defaults the search picks *one* transformation per iteration from a
/// set of about 23,000, and building a weight vector, an index vector, a live-weight
/// vector and a softmax over all of them cost four six-figure allocations and a pass over
/// the whole set — around 200 microseconds, against roughly 30 for everything the
/// iteration actually needed to do.
///
/// For small `k` this rejects duplicates, which is cheaper than materialising `0..n`; when
/// `k` approaches `n` it falls back to a partial shuffle.
pub fn sample_uniform_distinct<R: Rng + ?Sized>(n: usize, k: usize, rng: &mut R) -> Vec<usize> {
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }
    // Rejection sampling stops being the cheaper option once the draw covers much of the
    // set, because collisions dominate.
    if k * 2 >= n {
        let mut all: Vec<usize> = (0..n).collect();
        all.partial_shuffle(rng, k);
        all.truncate(k);
        return all;
    }
    let mut out: Vec<usize> = Vec::with_capacity(k);
    let mut seen: rustc_hash::FxHashSet<usize> = rustc_hash::FxHashSet::default();
    while out.len() < k {
        let i = rng.gen_range(0..n);
        if seen.insert(i) {
            out.push(i);
        }
    }
    out
}

pub fn sample_distinct<R: Rng + ?Sized>(
    weights: &[f64],
    n: usize,
    temperature: f64,
    rng: &mut R,
) -> Vec<usize> {
    let n = n.min(weights.len());
    if n == 0 {
        return Vec::new();
    }
    let mut remaining: Vec<usize> = (0..weights.len()).collect();

    if temperature <= 0.0 {
        // Greedy: highest weights first, ties broken by index for determinism.
        remaining.sort_by(|&a, &b| {
            weights[b]
                .partial_cmp(&weights[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        remaining.truncate(n);
        return remaining;
    }

    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if remaining.is_empty() {
            break;
        }
        let live: Vec<f64> = remaining.iter().map(|&i| weights[i]).collect();
        let probs = softmax(&live, temperature);
        let Some(pick) = sample_index(&probs, rng) else {
            break;
        };
        out.push(remaining.swap_remove(pick));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_draw_returns_distinct_indices_in_range() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for (n, k) in [
            (1, 1),
            (10, 3),
            (23_000, 1),
            (23_000, 5),
            (100, 100),
            (100, 150),
        ] {
            let out = sample_uniform_distinct(n, k, &mut rng);
            assert_eq!(out.len(), k.min(n), "n={n} k={k}");
            let unique: std::collections::HashSet<usize> = out.iter().copied().collect();
            assert_eq!(unique.len(), out.len(), "n={n} k={k}: duplicates");
            assert!(out.iter().all(|&i| i < n), "n={n} k={k}: out of range");
        }
        assert!(sample_uniform_distinct(10, 0, &mut rng).is_empty());
        assert!(sample_uniform_distinct(0, 5, &mut rng).is_empty());
    }

    #[test]
    fn uniform_draw_is_actually_uniform() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let n = 8;
        let mut counts = vec![0usize; n];
        let trials = 40_000;
        for _ in 0..trials {
            for i in sample_uniform_distinct(n, 1, &mut rng) {
                counts[i] += 1;
            }
        }
        let expected = trials as f64 / n as f64;
        for (i, &c) in counts.iter().enumerate() {
            let rel = (c as f64 - expected).abs() / expected;
            assert!(
                rel < 0.1,
                "index {i} drawn {c} times, expected about {expected}"
            );
        }
    }

    /// Drawing one index from a large set must not touch the whole set.
    ///
    /// The search does this every iteration with roughly 23,000 transformations; routing
    /// it through the weighted sampler cost four allocations of that size and a softmax
    /// pass, which was most of the per-iteration time.
    #[test]
    fn uniform_draw_of_one_is_cheap_at_scale() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let n = 200_000;

        let start = std::time::Instant::now();
        for _ in 0..2_000 {
            std::hint::black_box(sample_uniform_distinct(n, 1, &mut rng));
        }
        let uniform = start.elapsed();

        let weights = vec![0.0; n];
        let start = std::time::Instant::now();
        for _ in 0..20 {
            std::hint::black_box(sample_distinct(&weights, 1, 1.0, &mut rng));
        }
        let weighted = start.elapsed() * 100; // scale to the same number of draws

        assert!(
            uniform * 10 < weighted,
            "uniform draw should be far cheaper: {uniform:?} vs {weighted:?} per 2000"
        );
    }
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rustc_hash::FxHashSet;

    fn rng(seed: u64) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(seed)
    }

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[1.0, 2.0, 3.0], 1.0);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p[2] > p[1] && p[1] > p[0]);
    }

    #[test]
    fn softmax_handles_extremes_without_overflow() {
        let p = softmax(&[1e300, 0.0, -1e300], 1.0);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p.iter().all(|x| x.is_finite()));
        assert!((p[0] - 1.0).abs() < 1e-12);

        // Equal weights give a uniform distribution.
        let u = softmax(&[5.0, 5.0, 5.0], 1.0);
        assert!(u.iter().all(|x| (x - 1.0 / 3.0).abs() < 1e-12));
    }

    #[test]
    fn softmax_of_empty_is_empty() {
        assert!(softmax(&[], 1.0).is_empty());
        assert!(sample_index(&[], &mut rng(0)).is_none());
    }

    #[test]
    fn sample_index_respects_the_distribution() {
        let dist = [0.0, 1.0, 0.0];
        for seed in 0..20 {
            assert_eq!(sample_index(&dist, &mut rng(seed)), Some(1));
        }
    }

    #[test]
    fn sample_index_is_roughly_proportional() {
        let dist = [0.25, 0.75];
        let mut counts = [0usize; 2];
        let mut r = rng(42);
        for _ in 0..4000 {
            counts[sample_index(&dist, &mut r).unwrap()] += 1;
        }
        let ratio = counts[1] as f64 / 4000.0;
        assert!((ratio - 0.75).abs() < 0.05, "got {ratio}");
    }

    /// Greedy selection returns `n` *distinct* indices, not the best one `n` times.
    #[test]
    fn greedy_sampling_returns_distinct_indices() {
        let weights = [5.0, 1.0, 9.0, 3.0, 7.0];
        let picked = sample_distinct(&weights, 3, 0.0, &mut rng(1));
        assert_eq!(picked.len(), 3);
        let unique: FxHashSet<usize> = picked.iter().copied().collect();
        assert_eq!(unique.len(), 3, "expected distinct indices, got {picked:?}");
        // Highest weights first.
        assert_eq!(picked, vec![2, 4, 0]);
    }

    /// ...and above temperature 0 it sampled with replacement.
    #[test]
    fn stochastic_sampling_returns_distinct_indices() {
        let weights: Vec<f64> = (0..20).map(|i| i as f64).collect();
        for seed in 0..25 {
            let picked = sample_distinct(&weights, 8, 1.0, &mut rng(seed));
            assert_eq!(picked.len(), 8);
            let unique: FxHashSet<usize> = picked.iter().copied().collect();
            assert_eq!(unique.len(), 8, "seed {seed} repeated an index: {picked:?}");
        }
    }

    #[test]
    fn sampling_favours_higher_weights() {
        let weights: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let mut top_half = 0usize;
        for seed in 0..200 {
            for i in sample_distinct(&weights, 3, 1.0, &mut rng(seed)) {
                if i >= 5 {
                    top_half += 1;
                }
            }
        }
        assert!(
            top_half > 350,
            "expected a bias toward high weights, got {top_half}/600"
        );
    }

    #[test]
    fn asking_for_more_than_available_returns_everything() {
        let weights = [1.0, 2.0, 3.0];
        let picked = sample_distinct(&weights, 99, 1.0, &mut rng(3));
        assert_eq!(picked.len(), 3);
        let unique: FxHashSet<usize> = picked.iter().copied().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn asking_for_none_returns_none() {
        assert!(sample_distinct(&[1.0, 2.0], 0, 1.0, &mut rng(0)).is_empty());
        assert!(sample_distinct(&[], 5, 1.0, &mut rng(0)).is_empty());
    }

    #[test]
    fn sampling_is_deterministic_for_a_seed() {
        let weights: Vec<f64> = (0..30).map(|i| (i % 7) as f64).collect();
        let a = sample_distinct(&weights, 10, 1.0, &mut rng(7));
        let b = sample_distinct(&weights, 10, 1.0, &mut rng(7));
        assert_eq!(a, b);
    }
}
