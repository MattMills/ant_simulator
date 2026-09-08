//! A small, dependency-free pseudo-random number generator.
//!
//! The generator is xoshiro256** seeded through SplitMix64. Every piece of
//! randomness in the crate flows through [`Rng`], so a simulation, an arena,
//! or a learner is fully reproducible from a single `u64` seed.

/// Deterministic pseudo-random number generator (xoshiro256**).
#[derive(Clone, Debug)]
pub struct Rng {
    state: [u64; 4],
    spare_normal: Option<f64>,
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// Create a generator from a 64-bit seed.
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut s = seed;
        let state = [
            splitmix64(&mut s),
            splitmix64(&mut s),
            splitmix64(&mut s),
            splitmix64(&mut s),
        ];
        Rng {
            state,
            spare_normal: None,
        }
    }

    /// Next raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform `f64` in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform `f64` in `[lo, hi)`.
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Uniform integer in `[0, n)`. Panics if `n == 0`.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "Rng::below called with n == 0");
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Bernoulli trial with probability `p`.
    pub fn chance(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }

    /// Standard normal sample (Box–Muller, with the spare value cached).
    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare_normal.take() {
            return z;
        }
        let u1 = loop {
            let u = self.next_f64();
            if u > 0.0 {
                break u;
            }
        };
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        self.spare_normal = Some(r * theta.sin());
        r * theta.cos()
    }

    /// Normal sample with the given mean and standard deviation.
    pub fn normal_with(&mut self, mean: f64, std: f64) -> f64 {
        mean + std * self.normal()
    }

    /// In-place Fisher–Yates shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }

    /// A uniformly random permutation of `0..n`.
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        self.shuffle(&mut p);
        p
    }

    /// Choose an index with probability proportional to `weights`.
    ///
    /// Non-finite or negative weights are treated as zero. If every weight is
    /// zero the choice is uniform.
    pub fn choose_weighted(&mut self, weights: &[f64]) -> usize {
        assert!(!weights.is_empty(), "choose_weighted on empty slice");
        let total: f64 = weights
            .iter()
            .map(|w| if w.is_finite() && *w > 0.0 { *w } else { 0.0 })
            .sum();
        if total <= 0.0 {
            return self.below(weights.len());
        }
        let mut target = self.next_f64() * total;
        for (i, w) in weights.iter().enumerate() {
            let w = if w.is_finite() && *w > 0.0 { *w } else { 0.0 };
            if target < w {
                return i;
            }
            target -= w;
        }
        // Floating-point slack: return the last positive-weight index.
        weights
            .iter()
            .rposition(|w| w.is_finite() && *w > 0.0)
            .unwrap_or(weights.len() - 1)
    }

    /// Derive an independent generator from this one's stream.
    pub fn fork(&mut self) -> Rng {
        Rng::seed_from_u64(self.next_u64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_seed() {
        let mut a = Rng::seed_from_u64(7);
        let mut b = Rng::seed_from_u64(7);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn unit_interval_and_below() {
        let mut r = Rng::seed_from_u64(1);
        for _ in 0..10_000 {
            let f = r.next_f64();
            assert!((0.0..1.0).contains(&f));
            assert!(r.below(5) < 5);
        }
    }

    #[test]
    fn normal_moments() {
        let mut r = Rng::seed_from_u64(3);
        let n = 50_000;
        let xs: Vec<f64> = (0..n).map(|_| r.normal()).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.03, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "var {var}");
    }

    #[test]
    fn weighted_choice_respects_weights() {
        let mut r = Rng::seed_from_u64(9);
        let mut counts = [0usize; 3];
        for _ in 0..30_000 {
            counts[r.choose_weighted(&[0.0, 1.0, 3.0])] += 1;
        }
        assert_eq!(counts[0], 0);
        let ratio = counts[2] as f64 / counts[1] as f64;
        assert!((ratio - 3.0).abs() < 0.3, "ratio {ratio}");
    }

    #[test]
    fn permutation_is_bijective() {
        let mut r = Rng::seed_from_u64(11);
        let p = r.permutation(20);
        let mut sorted = p.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..20).collect::<Vec<_>>());
    }
}
