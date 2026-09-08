//! A bandit over the entropy dial alone.

use super::{Learner, LeverView, Outcome, RunningStats};
use crate::rng::Rng;
use crate::surface::ENTROPY_PARAM;

/// UCB1 over a fixed grid of raw entropy settings.
///
/// This learner leaves the weights of whatever surface it holds untouched and
/// only turns the entropy dial. It is the purest form of the crate's
/// premise: an agent whose only lever is "how much disorder does the thing I
/// am connected to have", learning which setting pays.
#[derive(Clone, Debug)]
pub struct EntropyBandit {
    arms: Vec<f64>,
    stats: Vec<RunningStats>,
    global: RunningStats,
    exploration: f64,
    pending: Option<usize>,
}

impl EntropyBandit {
    /// Bandit over the given raw dial values (`sigmoid` for absolute dials,
    /// `exp` for relative ones).
    pub fn new(arms: Vec<f64>) -> Self {
        assert!(!arms.is_empty(), "EntropyBandit needs at least one arm");
        let n = arms.len();
        EntropyBandit {
            arms,
            stats: vec![RunningStats::default(); n],
            global: RunningStats::default(),
            exploration: 1.0,
            pending: None,
        }
    }

    /// Scale the exploration bonus.
    pub fn with_exploration(mut self, exploration: f64) -> Self {
        self.exploration = exploration.max(0.0);
        self
    }

    /// The arm values.
    pub fn arms(&self) -> &[f64] {
        &self.arms
    }

    /// Index of the arm with the highest mean reward (untried arms lose).
    pub fn best_arm(&self) -> usize {
        let mut best = 0;
        let mut best_mean = f64::NEG_INFINITY;
        for (i, s) in self.stats.iter().enumerate() {
            if s.count() > 0 && s.mean() > best_mean {
                best_mean = s.mean();
                best = i;
            }
        }
        best
    }

    fn choose(&self) -> usize {
        if let Some(i) = self.stats.iter().position(|s| s.count() == 0) {
            return i;
        }
        let total = self.global.count().max(1) as f64;
        let scale = self.global.std().max(1e-3);
        let mut best = 0;
        let mut best_ucb = f64::NEG_INFINITY;
        for (i, s) in self.stats.iter().enumerate() {
            let bonus = self.exploration * scale * (2.0 * total.ln() / s.count() as f64).sqrt();
            let ucb = s.mean() + bonus;
            if ucb > best_ucb {
                best_ucb = ucb;
                best = i;
            }
        }
        best
    }
}

impl Default for EntropyBandit {
    /// Seven arms spanning very ordered to very disordered.
    fn default() -> Self {
        EntropyBandit::new(vec![-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5])
    }
}

impl Learner for EntropyBandit {
    fn name(&self) -> &str {
        "entropy-bandit"
    }

    fn act(&mut self, view: &LeverView<'_>, _rng: &mut Rng) -> Vec<f64> {
        let arm = self.choose();
        self.pending = Some(arm);
        let mut params = view.params.to_vec();
        if let Some(slot) = params.get_mut(ENTROPY_PARAM.min(view.params.len().saturating_sub(1))) {
            *slot = self.arms[arm];
        }
        params
    }

    fn feedback(&mut self, outcome: &Outcome, _rng: &mut Rng) {
        if let Some(arm) = self.pending.take() {
            self.stats[arm].push(outcome.reward);
            self.global.push(outcome.reward);
        }
    }

    fn release(&mut self, view: &LeverView<'_>, _rng: &mut Rng) -> Option<Vec<f64>> {
        if self.global.count() == 0 || view.params.is_empty() {
            return None;
        }
        let mut params = view.params.to_vec();
        let idx = ENTROPY_PARAM.min(params.len() - 1);
        params[idx] = self.arms[self.best_arm()];
        Some(params)
    }

    fn summary(&self) -> String {
        let best = self.best_arm();
        let means: Vec<String> = self
            .stats
            .iter()
            .zip(&self.arms)
            .map(|(s, a)| format!("{a:+.1}:{:.1}", s.mean()))
            .collect();
        format!(
            "entropy-bandit best raw {:+.1} ({} pulls) [{}]",
            self.arms[best],
            self.stats[best].count(),
            means.join(" ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_paying_arm() {
        let mut rng = Rng::seed_from_u64(4);
        let mut bandit = EntropyBandit::default();
        let params = vec![0.0; 17];
        for turn in 0..300u64 {
            let view = LeverView {
                turn,
                lever: 0,
                params: &params,
                last_reward: None,
                revealed_node: None,
            };
            let p = bandit.act(&view, &mut rng);
            assert_eq!(p.len(), 17);
            // Reward peaks when the raw dial is 0.5.
            let reward = -(p[ENTROPY_PARAM] - 0.5).abs() + rng.normal_with(0.0, 0.2);
            bandit.feedback(
                &Outcome {
                    turn,
                    lever: 0,
                    reward,
                    score: None,
                    decisions: 0,
                    mean_entropy: 0.0,
                },
                &mut rng,
            );
        }
        assert_eq!(bandit.arms()[bandit.best_arm()], 0.5);
        let view = LeverView {
            turn: 300,
            lever: 0,
            params: &params,
            last_reward: None,
            revealed_node: None,
        };
        let released = bandit.release(&view, &mut rng).unwrap();
        assert_eq!(released[ENTROPY_PARAM], 0.5);
        assert_eq!(released[..ENTROPY_PARAM], params[..ENTROPY_PARAM]);
        assert!(bandit.summary().contains("best raw +0.5"));
    }
}
