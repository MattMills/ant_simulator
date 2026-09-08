//! Making any learner aware of the hidden rotation.

use super::{Learner, LeverView, Outcome, PeriodDetector};
use crate::rng::Rng;

/// Wraps a learner so it keeps one independent copy per phase of the
/// rotation it infers from its rewards.
///
/// The wrapper feeds every reward, together with the parameters the lever
/// showed that turn, to a [`PeriodDetector`]. Once an estimated
/// period `p` has been stable for `stability` consecutive turns it adopts
/// it: the inner learner is cloned `p` times and turn `t` is handled by copy
/// `t mod p`, which sees a compressed turn counter `t / p`. This is how an
/// agent with a lever of unknown connection discovers that "every fourth
/// turn I am holding the same thing" and learns it separately.
#[derive(Clone, Debug)]
pub struct PhaseAware<L: Learner + Clone> {
    prototype: L,
    slots: Vec<L>,
    detector: PeriodDetector,
    period: usize,
    candidate: usize,
    streak: usize,
    stability: usize,
    pending_slot: Option<usize>,
    pending_params: Vec<f64>,
    resets: u64,
    name: String,
}

impl<L: Learner + Clone> PhaseAware<L> {
    /// Wrap `inner`, considering periods up to `max_period`.
    pub fn new(inner: L, max_period: usize) -> Self {
        let name = format!("phase-aware({})", inner.name());
        PhaseAware {
            slots: vec![inner.clone()],
            prototype: inner,
            detector: PeriodDetector::new(max_period),
            period: 1,
            candidate: 1,
            streak: 0,
            stability: 3,
            pending_slot: None,
            pending_params: Vec::new(),
            resets: 0,
            name,
        }
    }

    /// Use a custom detector.
    pub fn with_detector(mut self, detector: PeriodDetector) -> Self {
        self.detector = detector;
        self
    }

    /// Number of consecutive agreeing estimates required before switching period.
    pub fn with_stability(mut self, stability: usize) -> Self {
        self.stability = stability.max(1);
        self
    }

    /// Period currently in use.
    pub fn period(&self) -> usize {
        self.period
    }

    /// How many times the period changed and the slots were rebuilt.
    pub fn resets(&self) -> u64 {
        self.resets
    }

    /// The per-phase learners.
    pub fn slots(&self) -> &[L] {
        &self.slots
    }

    fn inner_view<'a>(&self, view: &LeverView<'a>) -> LeverView<'a> {
        LeverView {
            turn: view.turn / self.period as u64,
            ..*view
        }
    }

    fn slot_for(&self, turn: u64) -> usize {
        (turn % self.period as u64) as usize
    }

    fn maybe_adopt(&mut self) {
        let estimate = self.detector.period();
        if estimate == self.candidate {
            self.streak += 1;
        } else {
            self.candidate = estimate;
            self.streak = 1;
        }
        if estimate != self.period && self.streak >= self.stability {
            self.period = estimate;
            self.slots = vec![self.prototype.clone(); estimate];
            self.pending_slot = None;
            self.resets += 1;
        }
    }
}

impl<L: Learner + Clone> Learner for PhaseAware<L> {
    fn name(&self) -> &str {
        &self.name
    }

    fn act(&mut self, view: &LeverView<'_>, rng: &mut Rng) -> Vec<f64> {
        let slot = self.slot_for(view.turn);
        self.pending_slot = Some(slot);
        self.pending_params = view.params.to_vec();
        let inner = self.inner_view(view);
        self.slots[slot].act(&inner, rng)
    }

    fn feedback(&mut self, outcome: &Outcome, rng: &mut Rng) {
        self.detector
            .observe_with_params(outcome.turn, outcome.reward, &self.pending_params);
        if let Some(slot) = self.pending_slot {
            let inner = Outcome {
                turn: outcome.turn / self.period as u64,
                ..outcome.clone()
            };
            self.slots[slot].feedback(&inner, rng);
        }
        self.maybe_adopt();
    }

    fn release(&mut self, view: &LeverView<'_>, rng: &mut Rng) -> Option<Vec<f64>> {
        let slot = self.pending_slot.take()?;
        let inner = self.inner_view(view);
        self.slots[slot].release(&inner, rng)
    }

    fn inferred_period(&self) -> Option<usize> {
        Some(self.period)
    }

    fn summary(&self) -> String {
        format!(
            "{} period {} ({} resets); slot 0: {}",
            self.name,
            self.period,
            self.resets,
            self.slots[0].summary()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::HillClimber;

    /// Two hidden surfaces alternate; each has its own optimum. A plain hill
    /// climber gets confused, a phase-aware one does not.
    fn hidden_pair(learner: &mut dyn Learner, turns: u64) -> f64 {
        let mut rng = Rng::seed_from_u64(9);
        let targets = [vec![2.0; 5], vec![-2.0; 5]];
        let mut surfaces = [vec![0.0; 5], vec![0.0; 5]];
        let mut total = 0.0;
        for turn in 0..turns {
            let which = (turn % 2) as usize;
            let current = surfaces[which].clone();
            let view = LeverView {
                turn,
                lever: 0,
                params: &current,
                last_reward: None,
                revealed_node: None,
            };
            let p = learner.act(&view, &mut rng);
            let reward = -p
                .iter()
                .zip(&targets[which])
                .map(|(x, t)| (x - t).powi(2))
                .sum::<f64>()
                + if which == 0 { 30.0 } else { 0.0 };
            learner.feedback(
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
            surfaces[which] = p;
            let view = LeverView {
                turn,
                lever: 0,
                params: &surfaces[which],
                last_reward: Some(reward),
                revealed_node: None,
            };
            if let Some(left) = learner.release(&view, &mut rng) {
                surfaces[which] = left;
            }
            if turn >= turns - 20 {
                total += reward;
            }
        }
        total / 20.0
    }

    #[test]
    fn phase_awareness_separates_the_two_surfaces() {
        let mut naive = HillClimber::new(0.3).with_forgetting(0.0);
        let naive_score = hidden_pair(&mut naive, 300);
        let mut aware = PhaseAware::new(HillClimber::new(0.3).with_forgetting(0.0), 6);
        let aware_score = hidden_pair(&mut aware, 300);
        assert_eq!(aware.period(), 2);
        assert!(aware.inferred_period() == Some(2));
        assert!(
            aware_score > naive_score + 5.0,
            "aware {aware_score} vs naive {naive_score}"
        );
        assert!(aware.summary().contains("period 2"));
    }
}
