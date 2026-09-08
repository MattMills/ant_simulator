//! Discovering the period of a hidden, replayed rotation from what a lever
//! feels like (the parameters at its far end) and what it pays (rewards).

use std::collections::VecDeque;

#[derive(Clone, Debug)]
struct Sample {
    turn: u64,
    reward: f64,
    params: Option<Vec<f64>>,
}

/// Estimates the period of a periodic lever from its rewards and, when
/// supplied, the parameter vectors it observed.
///
/// For every candidate period `p` the samples are grouped by `turn mod p`
/// and scored with a BIC-style criterion, `n·ln(SSE/n) + penalty·k·ln(n)`,
/// summed over the reward and every varying parameter dimension (`k` counts
/// the per-phase means fitted). The candidate with the lowest score wins;
/// candidates whose phases have too few samples are skipped. Only the most
/// recent `window` samples are kept so the estimate can adapt when the
/// learner's own progress shifts the reward level.
#[derive(Clone, Debug)]
pub struct PeriodDetector {
    max_period: usize,
    window: usize,
    min_repeats: usize,
    penalty: f64,
    param_weight: f64,
    samples: VecDeque<Sample>,
}

impl PeriodDetector {
    /// Detector considering periods `1..=max_period`.
    pub fn new(max_period: usize) -> Self {
        PeriodDetector {
            max_period: max_period.max(1),
            window: 96,
            min_repeats: 3,
            penalty: 1.0,
            param_weight: 1.0,
            samples: VecDeque::new(),
        }
    }

    /// Keep only the last `window` samples.
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = window.max(2);
        self
    }

    /// Require at least this many samples per phase before a period is eligible.
    pub fn with_min_repeats(mut self, min_repeats: usize) -> Self {
        self.min_repeats = min_repeats.max(1);
        self
    }

    /// Scale the complexity penalty (larger favours shorter periods).
    pub fn with_penalty(mut self, penalty: f64) -> Self {
        self.penalty = penalty.max(0.0);
        self
    }

    /// Weight of the parameter-fingerprint evidence relative to rewards
    /// (`0` ignores parameters entirely).
    pub fn with_param_weight(mut self, weight: f64) -> Self {
        self.param_weight = weight.max(0.0);
        self
    }

    /// Largest period considered.
    pub fn max_period(&self) -> usize {
        self.max_period
    }

    /// Record the reward received on a turn.
    pub fn observe(&mut self, turn: u64, reward: f64) {
        self.push(turn, reward, None);
    }

    /// Record the reward received on a turn together with the parameters the
    /// lever showed at the start of that turn.
    pub fn observe_with_params(&mut self, turn: u64, reward: f64, params: &[f64]) {
        self.push(turn, reward, Some(params.to_vec()));
    }

    fn push(&mut self, turn: u64, reward: f64, params: Option<Vec<f64>>) {
        if !reward.is_finite() {
            return;
        }
        let params = params.filter(|p| p.iter().all(|x| x.is_finite()));
        self.samples.push_back(Sample {
            turn,
            reward,
            params,
        });
        while self.samples.len() > self.window {
            self.samples.pop_front();
        }
    }

    /// Number of stored samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether no samples are stored.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Dimension of the parameter fingerprints if every sample carries one.
    fn param_dim(&self) -> Option<usize> {
        let mut dim = None;
        for s in &self.samples {
            match (&s.params, dim) {
                (None, _) => return None,
                (Some(p), None) => dim = Some(p.len()),
                (Some(p), Some(d)) if p.len() != d => return None,
                _ => {}
            }
        }
        dim
    }

    /// BIC-style score of a candidate period (lower is better), or `None` if
    /// some phase has fewer than `min_repeats` samples.
    pub fn score(&self, period: usize) -> Option<f64> {
        if period == 0 {
            return None;
        }
        let n = self.samples.len();
        if n < self.min_repeats * period {
            return None;
        }
        let dim = self.param_dim().filter(|_| self.param_weight > 0.0);
        let width = 1 + dim.unwrap_or(0);
        let mut count = vec![0usize; period];
        let mut sum = vec![0.0f64; period * width];
        let mut sumsq = vec![0.0f64; period * width];
        let mut total = vec![0.0f64; width];
        let mut totalsq = vec![0.0f64; width];
        for s in &self.samples {
            let k = (s.turn % period as u64) as usize;
            count[k] += 1;
            let row = k * width;
            sum[row] += s.reward;
            sumsq[row] += s.reward * s.reward;
            total[0] += s.reward;
            totalsq[0] += s.reward * s.reward;
            if let (Some(p), Some(_)) = (&s.params, dim) {
                for (j, x) in p.iter().enumerate() {
                    sum[row + 1 + j] += x;
                    sumsq[row + 1 + j] += x * x;
                    total[1 + j] += x;
                    totalsq[1 + j] += x * x;
                }
            }
        }
        if count.iter().any(|&c| c < self.min_repeats) {
            return None;
        }
        let nf = n as f64;
        let mut score = 0.0;
        let mut fitted = 0usize;
        for j in 0..width {
            let overall_var = (totalsq[j] - total[j] * total[j] / nf) / nf;
            if j > 0 && overall_var <= 1e-12 {
                continue; // a dimension that never varies carries no evidence
            }
            let mut sse = 0.0;
            for (k, &c) in count.iter().enumerate() {
                let idx = k * width + j;
                sse += sumsq[idx] - sum[idx] * sum[idx] / c as f64;
            }
            let floor = 1e-9 + 1e-3 * overall_var.max(0.0);
            let weight = if j == 0 { 1.0 } else { self.param_weight };
            score += weight * nf * (sse / nf + floor).ln();
            fitted += 1;
        }
        Some(score + self.penalty * (period * fitted) as f64 * nf.ln())
    }

    /// The best-scoring period (1 when nothing else is supported by the data).
    pub fn period(&self) -> usize {
        let mut best = (1usize, f64::INFINITY);
        for p in 1..=self.max_period {
            if let Some(s) = self.score(p) {
                if s < best.1 - 1e-9 {
                    best = (p, s);
                }
            }
        }
        best.0
    }

    /// Scores of every candidate period, `None` where not enough data.
    pub fn scores(&self) -> Vec<Option<f64>> {
        (1..=self.max_period).map(|p| self.score(p)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn recovers_a_planted_period_from_rewards() {
        let mut rng = Rng::seed_from_u64(1);
        let levels = [10.0, 3.0, 7.0, 1.0, 12.0];
        let mut det = PeriodDetector::new(8);
        for turn in 0..80u64 {
            let r = levels[(turn % 5) as usize] + rng.normal_with(0.0, 0.5);
            det.observe(turn, r);
        }
        assert_eq!(det.period(), 5);
    }

    #[test]
    fn stationary_noise_gives_period_one() {
        let mut rng = Rng::seed_from_u64(2);
        let mut det = PeriodDetector::new(8);
        for turn in 0..120u64 {
            det.observe(turn, rng.normal_with(5.0, 1.0));
        }
        assert_eq!(det.period(), 1);
    }

    #[test]
    fn recovers_a_period_from_parameter_fingerprints_alone() {
        // Rewards are identical across phases; only the parameters differ.
        let mut rng = Rng::seed_from_u64(3);
        let fingerprints = [
            vec![0.0, 1.0, 2.0],
            vec![5.0, 1.0, -1.0],
            vec![0.0, 1.0, 9.0],
        ];
        let mut det = PeriodDetector::new(6);
        for turn in 0..60u64 {
            let p: Vec<f64> = fingerprints[(turn % 3) as usize]
                .iter()
                .map(|x| x + rng.normal_with(0.0, 0.05))
                .collect();
            det.observe_with_params(turn, rng.normal_with(4.0, 1.0), &p);
        }
        assert_eq!(det.period(), 3);
        // Ignoring parameters, nothing is visible.
        let blind = det.clone().with_param_weight(0.0);
        assert_eq!(blind.period(), 1);
    }

    #[test]
    fn mixed_samples_fall_back_to_rewards() {
        let mut det = PeriodDetector::new(4);
        for turn in 0..24u64 {
            let r = if turn % 2 == 0 { 10.0 } else { 0.0 };
            if turn % 5 == 0 {
                det.observe(turn, r);
            } else {
                det.observe_with_params(turn, r, &[turn as f64]);
            }
        }
        assert_eq!(det.period(), 2);
    }

    #[test]
    fn insufficient_data_is_not_scored() {
        let mut det = PeriodDetector::new(6).with_min_repeats(2);
        for turn in 0..5u64 {
            det.observe(turn, turn as f64);
        }
        assert!(det.score(3).is_none());
        assert!(det.score(2).is_some());
        assert_eq!(det.len(), 5);
        det.observe(9, f64::NAN);
        assert_eq!(det.len(), 5);
        assert_eq!(det.scores().len(), 6);
        let windowed = PeriodDetector::new(3).with_window(4);
        assert!(windowed.is_empty());
        assert_eq!(windowed.max_period(), 3);
    }
}
