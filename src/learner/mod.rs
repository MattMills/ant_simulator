//! Learners: agents that hold a lever without knowing where it connects.
//!
//! A learner never sees the hierarchy. Each turn it may be handed a
//! [`LeverView`]: the parameter vector of *some* node's entropic behavioral
//! surface, the turn number, and the reward it earned last time. It answers
//! with a new parameter vector, the colony runs, and it receives an
//! [`Outcome`]. Which node the lever reaches is decided by the arena's
//! [`crate::rotation::Rotation`] and is hidden unless the arena is configured
//! to reveal it.
//!
//! Because the rotation is drawn once and then replayed, the mapping is
//! random but static: a learner that tracks the turn number can discover the
//! period and keep separate knowledge per phase (see [`PhaseAware`]).

use crate::hierarchy::NodeId;
use crate::rng::Rng;

pub mod bandit;
pub mod cross_entropy;
pub mod hill_climb;
pub mod period;
pub mod phase_aware;
pub mod policy_gradient;
pub mod simple;

pub use bandit::EntropyBandit;
pub use cross_entropy::CrossEntropy;
pub use hill_climb::{HillClimber, Mutation};
pub use period::PeriodDetector;
pub use phase_aware::PhaseAware;
pub use policy_gradient::PolicyGradient;
pub use simple::{RandomLearner, StaticLearner};

/// What a learner sees when it is handed a lever.
#[derive(Clone, Copy, Debug)]
pub struct LeverView<'a> {
    /// Global turn number of the arena.
    pub turn: u64,
    /// Index of the learner's own lever (stable across turns).
    pub lever: usize,
    /// Current parameters of the surface behind the lever.
    pub params: &'a [f64],
    /// Reward this learner received at the end of its previous turn.
    pub last_reward: Option<f64>,
    /// The node behind the lever, if the arena reveals the mapping.
    pub revealed_node: Option<NodeId>,
}

/// What a learner is told after the colony has run with its parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    /// Global turn number.
    pub turn: u64,
    /// The learner's lever index.
    pub lever: usize,
    /// Reward for the turn (scope decided by the arena).
    pub reward: f64,
    /// Score-function sum for the controlled surface, if the arena traces.
    pub score: Option<Vec<f64>>,
    /// Decisions made beneath the controlled node during the turn.
    pub decisions: u64,
    /// Mean realised entropy per decision colony-wide.
    pub mean_entropy: f64,
}

/// An agent that learns to operate a lever of unknown connection.
pub trait Learner {
    /// Display name.
    fn name(&self) -> &str;

    /// Choose the parameters to run the colony with. Must return exactly
    /// `view.params.len()` values.
    fn act(&mut self, view: &LeverView<'_>, rng: &mut Rng) -> Vec<f64>;

    /// Receive the result of the last [`act`](Self::act).
    fn feedback(&mut self, outcome: &Outcome, rng: &mut Rng);

    /// Optionally leave different parameters on the surface before the lever
    /// rotates away (e.g. the best known setting rather than the last probe).
    fn release(&mut self, _view: &LeverView<'_>, _rng: &mut Rng) -> Option<Vec<f64>> {
        None
    }

    /// The rotation period this learner believes it is subject to, if it
    /// models one.
    fn inferred_period(&self) -> Option<usize> {
        None
    }

    /// One-line description of the learner's internal state.
    fn summary(&self) -> String {
        self.name().to_string()
    }
}

impl<L: Learner + ?Sized> Learner for Box<L> {
    fn name(&self) -> &str {
        (**self).name()
    }
    fn act(&mut self, view: &LeverView<'_>, rng: &mut Rng) -> Vec<f64> {
        (**self).act(view, rng)
    }
    fn feedback(&mut self, outcome: &Outcome, rng: &mut Rng) {
        (**self).feedback(outcome, rng)
    }
    fn release(&mut self, view: &LeverView<'_>, rng: &mut Rng) -> Option<Vec<f64>> {
        (**self).release(view, rng)
    }
    fn inferred_period(&self) -> Option<usize> {
        (**self).inferred_period()
    }
    fn summary(&self) -> String {
        (**self).summary()
    }
}

/// Running mean and variance (Welford).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunningStats {
    n: u64,
    mean: f64,
    m2: f64,
}

impl RunningStats {
    /// Add a sample.
    pub fn push(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        self.m2 += delta * (x - self.mean);
    }

    /// Number of samples.
    pub fn count(&self) -> u64 {
        self.n
    }

    /// Mean (0 if empty).
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Population variance (0 if fewer than two samples).
    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            self.m2 / self.n as f64
        }
    }

    /// Standard deviation.
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn running_stats() {
        let mut s = RunningStats::default();
        for x in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            s.push(x);
        }
        assert_eq!(s.count(), 8);
        assert!((s.mean() - 5.0).abs() < 1e-12);
        assert!((s.std() - 2.0).abs() < 1e-12);
    }
}
