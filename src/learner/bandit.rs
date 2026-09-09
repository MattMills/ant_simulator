//! Bandits over the dials of a surface: entropy alone, or the geometry.

use super::{Learner, LeverView, Outcome, RunningStats};
use crate::rng::Rng;
use crate::surface::{ENTROPY_PARAM, REACH_PARAM, ROUGH_PARAM, SMOOTH_PARAM};

/// One arm of a [`DialBandit`]: the parameter indices it sets and the
/// values it sets them to.
#[derive(Clone, Debug, PartialEq)]
pub struct Arm {
    /// `(parameter index, value)` pairs.
    pub settings: Vec<(usize, f64)>,
    /// Short label for reports.
    pub label: String,
}

impl Arm {
    /// An arm setting a single parameter.
    pub fn single(index: usize, value: f64) -> Self {
        Arm {
            settings: vec![(index, value)],
            label: format!("{value:+.1}"),
        }
    }

    fn apply(&self, params: &mut [f64]) {
        for &(index, value) in &self.settings {
            if let Some(slot) = params.get_mut(index) {
                *slot = value;
            }
        }
    }
}

/// UCB1 over a fixed set of dial settings.
///
/// The bandit leaves every parameter it does not own untouched and only sets
/// the dials of its arms. [`DialBandit::entropy`] is the purest form of the
/// crate's premise: an agent whose only lever is "how much disorder does the
/// thing I am connected to have", learning which setting pays.
/// [`DialBandit::geometry`] instead searches how that disorder is shaped
/// (smoothing, roughening, and the sucker's reach).
#[derive(Clone, Debug)]
pub struct DialBandit {
    arms: Vec<Arm>,
    stats: Vec<RunningStats>,
    global: RunningStats,
    exploration: f64,
    pending: Option<usize>,
    name: String,
}

/// The entropy-only bandit; see [`DialBandit::entropy`].
pub type EntropyBandit = DialBandit;

impl DialBandit {
    /// Bandit over arbitrary arms.
    pub fn new(arms: Vec<Arm>) -> Self {
        assert!(!arms.is_empty(), "DialBandit needs at least one arm");
        let n = arms.len();
        DialBandit {
            arms,
            stats: vec![RunningStats::default(); n],
            global: RunningStats::default(),
            exploration: 1.0,
            pending: None,
            name: "dial-bandit".to_string(),
        }
    }

    /// Seven arms on the entropy dial, from very ordered (`-2.5`) to very
    /// disordered (`+2.5`), leaving everything else alone.
    pub fn entropy() -> Self {
        let mut b = DialBandit::new(
            [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5]
                .into_iter()
                .map(|v| Arm::single(ENTROPY_PARAM, v))
                .collect(),
        );
        b.name = "entropy-bandit".to_string();
        b
    }

    /// Twelve arms over the geometry of disorder: smoothing off/on,
    /// roughening off/on, and the sucker's reach at a quarter, once, or four
    /// times its base value.
    pub fn geometry() -> Self {
        let mut arms = Vec::new();
        for (smooth, s_label) in [(0.0, "flat"), (1.2, "smooth")] {
            for (rough, r_label) in [(0.0, ""), (1.2, "+rough")] {
                for (reach, k_label) in [(-1.386, "¼reach"), (0.0, "reach"), (1.386, "4×reach")] {
                    arms.push(Arm {
                        settings: vec![
                            (SMOOTH_PARAM, smooth),
                            (ROUGH_PARAM, rough),
                            (REACH_PARAM, reach),
                        ],
                        label: format!("{s_label}{r_label}/{k_label}"),
                    });
                }
            }
        }
        let mut b = DialBandit::new(arms);
        b.name = "geometry-bandit".to_string();
        b
    }

    /// Scale the exploration bonus.
    pub fn with_exploration(mut self, exploration: f64) -> Self {
        self.exploration = exploration.max(0.0);
        self
    }

    /// The arms.
    pub fn arms(&self) -> &[Arm] {
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

impl Default for DialBandit {
    /// The entropy bandit.
    fn default() -> Self {
        DialBandit::entropy()
    }
}

impl Learner for DialBandit {
    fn name(&self) -> &str {
        &self.name
    }

    fn act(&mut self, view: &LeverView<'_>, _rng: &mut Rng) -> Vec<f64> {
        let arm = self.choose();
        self.pending = Some(arm);
        let mut params = view.params.to_vec();
        self.arms[arm].apply(&mut params);
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
        self.arms[self.best_arm()].apply(&mut params);
        Some(params)
    }

    fn summary(&self) -> String {
        let best = self.best_arm();
        let means: Vec<String> = self
            .stats
            .iter()
            .zip(&self.arms)
            .map(|(s, a)| format!("{}:{:.1}", a.label, s.mean()))
            .collect();
        format!(
            "{} best {} ({} pulls) [{}]",
            self.name,
            self.arms[best].label,
            self.stats[best].count(),
            means.join(" ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::PARAM_LEN;

    fn outcome(turn: u64, reward: f64) -> Outcome {
        Outcome {
            turn,
            lever: 0,
            reward,
            score: None,
            decisions: 0,
            mean_entropy: 0.0,
        }
    }

    #[test]
    fn finds_the_paying_arm() {
        let mut rng = Rng::seed_from_u64(4);
        let mut bandit = DialBandit::entropy();
        let params = vec![0.0; PARAM_LEN];
        for turn in 0..300u64 {
            let view = LeverView {
                turn,
                lever: 0,
                params: &params,
                last_reward: None,
                revealed_node: None,
            };
            let p = bandit.act(&view, &mut rng);
            assert_eq!(p.len(), PARAM_LEN);
            // Reward peaks when the raw dial is 0.5.
            let reward = -(p[ENTROPY_PARAM] - 0.5).abs() + rng.normal_with(0.0, 0.2);
            bandit.feedback(&outcome(turn, reward), &mut rng);
        }
        let best = &bandit.arms()[bandit.best_arm()];
        assert_eq!(best.settings, vec![(ENTROPY_PARAM, 0.5)]);
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
        assert!(bandit.summary().contains("best +0.5"));
        assert_eq!(bandit.name(), "entropy-bandit");
    }

    #[test]
    fn geometry_arms_set_only_geometry() {
        let mut rng = Rng::seed_from_u64(5);
        let mut bandit = DialBandit::geometry();
        assert_eq!(bandit.arms().len(), 12);
        let params: Vec<f64> = (0..PARAM_LEN).map(|i| i as f64).collect();
        let view = LeverView {
            turn: 0,
            lever: 0,
            params: &params,
            last_reward: None,
            revealed_node: None,
        };
        let p = bandit.act(&view, &mut rng);
        assert_eq!(p[..SMOOTH_PARAM], params[..SMOOTH_PARAM]);
        assert!(p[SMOOTH_PARAM] == 0.0 || p[SMOOTH_PARAM] == 1.2);
        assert!(p[REACH_PARAM].abs() < 1.5);
        bandit.feedback(&outcome(0, 1.0), &mut rng);
        assert!(bandit.summary().starts_with("geometry-bandit"));
    }
}
