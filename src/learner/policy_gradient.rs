//! REINFORCE on the surface weights, using the simulation's score trace.

use super::{Learner, LeverView, Outcome, RunningStats};
use crate::rng::Rng;

/// Policy-gradient learner.
///
/// Runs the colony with the parameters as handed over, then, in
/// [`release`](Learner::release), applies one REINFORCE step using the
/// score-function sum the arena traced for the controlled node. The step is
/// scale-free: `Δθ = lr · â · ĝ`, where `â` is the advantage normalised by
/// the running standard deviation of rewards (clamped to ±3) and `ĝ` is the
/// unit direction of `score / decisions`. Its norm is therefore about `lr`
/// per turn regardless of reward scale or of the temperature the entropy
/// dial implies, and it is additionally clipped to `max_step`. Because the
/// update is relative to whatever surface it currently holds, this learner
/// is naturally robust to the rotation: it always nudges the surface it
/// just evaluated.
///
/// Requires the arena to run with tracing enabled; without a score it
/// leaves the surface untouched.
#[derive(Clone, Debug)]
pub struct PolicyGradient {
    lr: f64,
    max_step: f64,
    baseline: RunningStats,
    pending: Option<(Vec<f64>, f64)>,
    updates: u64,
    last_norm: f64,
}

impl PolicyGradient {
    /// Learner with the given learning rate.
    pub fn new(lr: f64) -> Self {
        PolicyGradient {
            lr,
            max_step: 0.25,
            baseline: RunningStats::default(),
            pending: None,
            updates: 0,
            last_norm: 0.0,
        }
    }

    /// Clip the update's Euclidean norm.
    pub fn with_max_step(mut self, max_step: f64) -> Self {
        self.max_step = max_step.max(0.0);
        self
    }

    /// Number of gradient steps applied so far.
    pub fn updates(&self) -> u64 {
        self.updates
    }
}

impl Default for PolicyGradient {
    fn default() -> Self {
        PolicyGradient::new(0.05)
    }
}

impl Learner for PolicyGradient {
    fn name(&self) -> &str {
        "policy-gradient"
    }

    fn act(&mut self, view: &LeverView<'_>, _rng: &mut Rng) -> Vec<f64> {
        view.params.to_vec()
    }

    fn feedback(&mut self, outcome: &Outcome, _rng: &mut Rng) {
        self.pending = None;
        if let Some(score) = &outcome.score {
            if outcome.reward.is_finite() && self.baseline.count() >= 2 {
                let advantage =
                    (outcome.reward - self.baseline.mean()) / (self.baseline.std() + 1e-6);
                let advantage = advantage.clamp(-3.0, 3.0);
                let norm = score.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 0.0 && norm.is_finite() {
                    self.pending = Some((score.clone(), advantage / norm));
                }
            }
        }
        if outcome.reward.is_finite() {
            self.baseline.push(outcome.reward);
        }
    }

    fn release(&mut self, view: &LeverView<'_>, _rng: &mut Rng) -> Option<Vec<f64>> {
        let (score, scale) = self.pending.take()?;
        if score.len() != view.params.len() {
            return None;
        }
        let mut step: Vec<f64> = score.iter().map(|s| self.lr * scale * s).collect();
        let norm = step.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return None;
        }
        if norm > self.max_step {
            for x in step.iter_mut() {
                *x *= self.max_step / norm;
            }
        }
        self.last_norm = norm.min(self.max_step);
        self.updates += 1;
        Some(view.params.iter().zip(&step).map(|(p, d)| p + d).collect())
    }

    fn summary(&self) -> String {
        format!(
            "policy-gradient {} updates, baseline {:.2}, last step norm {:.3}",
            self.updates,
            self.baseline.mean(),
            self.last_norm
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(turn: u64, reward: f64, score: Option<Vec<f64>>) -> Outcome {
        Outcome {
            turn,
            lever: 0,
            reward,
            score,
            decisions: 10,
            mean_entropy: 0.0,
        }
    }

    #[test]
    fn steps_along_the_score_once_rewards_have_a_scale() {
        let mut rng = Rng::seed_from_u64(1);
        let mut pg = PolicyGradient::new(0.1).with_max_step(10.0);
        let params = vec![0.0; 3];
        let view = LeverView {
            turn: 0,
            lever: 0,
            params: &params,
            last_reward: None,
            revealed_node: None,
        };
        assert_eq!(pg.act(&view, &mut rng), params);
        // Two outcomes only establish the baseline's scale.
        pg.feedback(&outcome(0, 1.0, Some(vec![1.0, 0.0, 0.0])), &mut rng);
        assert!(pg.release(&view, &mut rng).is_none());
        pg.feedback(&outcome(1, 3.0, Some(vec![1.0, 0.0, 0.0])), &mut rng);
        assert!(pg.release(&view, &mut rng).is_none());
        // Reward well above the mean: a step of about `lr` along the score.
        pg.feedback(&outcome(2, 5.0, Some(vec![0.0, 3.0, 4.0])), &mut rng);
        let updated = pg.release(&view, &mut rng).unwrap();
        assert!(updated[0].abs() < 1e-12);
        assert!(updated[1] > 0.0 && updated[2] > 0.0);
        assert!(
            (updated[1] / updated[2] - 0.75).abs() < 1e-9,
            "unit direction of the score"
        );
        let norm = (updated[1].powi(2) + updated[2].powi(2)).sqrt();
        assert!(
            norm > 0.05 && norm <= 0.3 + 1e-9,
            "norm {norm} ≈ lr × normalised advantage"
        );
        assert_eq!(pg.updates(), 1);
        // Reward below the mean steps the other way.
        pg.feedback(&outcome(3, -20.0, Some(vec![0.0, 3.0, 4.0])), &mut rng);
        let down = pg.release(&view, &mut rng).unwrap();
        assert!(down[1] < 0.0 && down[2] < 0.0);
        // Without a score nothing happens.
        pg.feedback(&outcome(4, 5.0, None), &mut rng);
        assert!(pg.release(&view, &mut rng).is_none());
        assert!(pg.summary().contains("2 updates"));
    }

    #[test]
    fn clips_large_steps() {
        let mut rng = Rng::seed_from_u64(1);
        let mut pg = PolicyGradient::new(1.0).with_max_step(0.1);
        let params = vec![0.0; 2];
        let view = LeverView {
            turn: 0,
            lever: 0,
            params: &params,
            last_reward: None,
            revealed_node: None,
        };
        for (turn, reward) in [(0u64, 0.0), (1, 10.0), (2, 100.0)] {
            pg.feedback(&outcome(turn, reward, Some(vec![3.0, 4.0])), &mut rng);
        }
        let updated = pg.release(&view, &mut rng).unwrap();
        let norm = (updated[0].powi(2) + updated[1].powi(2)).sqrt();
        assert!((norm - 0.1).abs() < 1e-9);
    }
}
