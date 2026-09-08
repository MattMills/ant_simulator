//! Baseline learners: one that never changes anything and one that flails.

use super::{Learner, LeverView, Outcome};
use crate::rng::Rng;

/// Leaves every surface exactly as it found it.
#[derive(Clone, Debug, Default)]
pub struct StaticLearner;

impl Learner for StaticLearner {
    fn name(&self) -> &str {
        "static"
    }

    fn act(&mut self, view: &LeverView<'_>, _rng: &mut Rng) -> Vec<f64> {
        view.params.to_vec()
    }

    fn feedback(&mut self, _outcome: &Outcome, _rng: &mut Rng) {}
}

/// Adds fresh Gaussian noise to whatever it is handed, every turn, and never
/// learns from the outcome. Useful as a control condition.
#[derive(Clone, Debug)]
pub struct RandomLearner {
    /// Standard deviation of the noise.
    pub scale: f64,
    turns: u64,
    total: f64,
}

impl RandomLearner {
    /// Noise with the given standard deviation.
    pub fn new(scale: f64) -> Self {
        RandomLearner {
            scale,
            turns: 0,
            total: 0.0,
        }
    }
}

impl Default for RandomLearner {
    fn default() -> Self {
        RandomLearner::new(0.3)
    }
}

impl Learner for RandomLearner {
    fn name(&self) -> &str {
        "random"
    }

    fn act(&mut self, view: &LeverView<'_>, rng: &mut Rng) -> Vec<f64> {
        view.params
            .iter()
            .map(|p| p + rng.normal_with(0.0, self.scale))
            .collect()
    }

    fn feedback(&mut self, outcome: &Outcome, _rng: &mut Rng) {
        self.turns += 1;
        self.total += outcome.reward;
    }

    fn summary(&self) -> String {
        let mean = if self.turns == 0 {
            0.0
        } else {
            self.total / self.turns as f64
        };
        format!("random(σ={}) mean reward {:.2}", self.scale, mean)
    }
}
