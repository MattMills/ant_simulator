//! A (1+1) evolution strategy that nudges whatever surface it holds.
//!
//! The lever is a dial, not a memory: the climber never writes a remembered
//! configuration onto a surface. It proposes a displacement from the current
//! parameters, keeps it if the reward beat its baseline, and reverts it
//! otherwise. That bounds the damage a lever of unknown connection can do.

use super::{Learner, LeverView, Outcome};
use crate::rng::Rng;
use crate::surface::{random_rotation, ENTROPY_PARAM};

/// How a displacement is generated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mutation {
    /// Gaussian noise (standard deviation = step size) on every parameter.
    Gaussian,
    /// Rotate the weight vector by the step size (radians) in a random plane
    /// and nudge the entropy dial by Gaussian noise of the same size.
    Rotation,
    /// Flip a coin between the two above each turn.
    Mixed,
}

/// (1+1)-ES over displacements, with 1/5th-rule step adaptation.
///
/// The climber remembers the parameters it last left on a surface and the
/// reward they earned (its *baseline*). When it is handed a surface that
/// looks like what it left, it probes a displacement and compares the reward
/// with the baseline. When the surface does not match (someone else moved
/// it, or the lever now reaches something else) it spends one turn
/// evaluating the surface as found before probing again. The baseline decays
/// a little each turn (`forgetting`) so a lucky measurement cannot freeze the
/// search under noisy rewards.
#[derive(Clone, Debug)]
pub struct HillClimber {
    step: f64,
    step_min: f64,
    step_max: f64,
    mutation: Mutation,
    forgetting: f64,
    reanchor: bool,
    anchor: Option<Vec<f64>>,
    baseline: Option<f64>,
    pending: Option<Vec<f64>>,
    accepted: u64,
    rejected: u64,
    evaluations: u64,
    name: String,
}

impl HillClimber {
    /// Gaussian hill climber with the given initial step size.
    pub fn new(step: f64) -> Self {
        HillClimber {
            step,
            step_min: step * 0.05,
            step_max: step * 4.0,
            mutation: Mutation::Gaussian,
            forgetting: 0.02,
            reanchor: true,
            anchor: None,
            baseline: None,
            pending: None,
            accepted: 0,
            rejected: 0,
            evaluations: 0,
            name: "hill-climb".to_string(),
        }
    }

    /// Choose the mutation operator.
    pub fn with_mutation(mut self, mutation: Mutation) -> Self {
        self.mutation = mutation;
        self.name = match mutation {
            Mutation::Gaussian => "hill-climb",
            Mutation::Rotation => "rotation-search",
            Mutation::Mixed => "hill-climb(mixed)",
        }
        .to_string();
        self
    }

    /// Set the per-turn decay applied to the baseline reward.
    pub fn with_forgetting(mut self, forgetting: f64) -> Self {
        self.forgetting = forgetting.clamp(0.0, 1.0);
        self
    }

    /// Set the step-size bounds.
    pub fn with_step_bounds(mut self, min: f64, max: f64) -> Self {
        self.step_min = min;
        self.step_max = max;
        self.step = self.step.clamp(min, max);
        self
    }

    /// Whether to re-evaluate a surface that no longer matches what the
    /// climber left (default `true`). With `false` the stale baseline is used.
    pub fn with_reanchor(mut self, reanchor: bool) -> Self {
        self.reanchor = reanchor;
        self
    }

    /// Current step size.
    pub fn step(&self) -> f64 {
        self.step
    }

    /// The baseline reward, if established.
    pub fn baseline(&self) -> Option<f64> {
        self.baseline
    }

    /// Accepted, rejected, and evaluation-only turns so far.
    pub fn counts(&self) -> (u64, u64, u64) {
        (self.accepted, self.rejected, self.evaluations)
    }

    fn displacement(&self, len: usize, rng: &mut Rng) -> Vec<f64> {
        let mut delta = vec![0.0; len];
        let rotate = match self.mutation {
            Mutation::Gaussian => false,
            Mutation::Rotation => true,
            Mutation::Mixed => rng.chance(0.5),
        };
        if rotate && len > 2 {
            // Express the rotation as a displacement of the weight block so the
            // rest of the machinery only ever deals in displacements.
            let last = len - 1;
            let weights_end = ENTROPY_PARAM.min(last);
            let angle = rng.normal_with(0.0, self.step);
            let mut rotated = self.anchor.clone().unwrap_or_else(|| vec![0.0; len]);
            random_rotation(&mut rotated[..weights_end], angle, rng);
            let base = self.anchor.as_ref();
            for (i, d) in delta.iter_mut().enumerate().take(weights_end) {
                *d = rotated[i] - base.map(|b| b[i]).unwrap_or(0.0);
            }
            delta[last] = rng.normal_with(0.0, self.step);
        } else {
            for d in delta.iter_mut() {
                *d = rng.normal_with(0.0, self.step);
            }
        }
        delta
    }

    fn matches_anchor(&self, params: &[f64]) -> bool {
        match &self.anchor {
            Some(a) if a.len() == params.len() => {
                a.iter().zip(params).all(|(x, y)| (x - y).abs() <= 1e-9)
            }
            _ => false,
        }
    }
}

impl Default for HillClimber {
    fn default() -> Self {
        HillClimber::new(0.25)
    }
}

impl Learner for HillClimber {
    fn name(&self) -> &str {
        &self.name
    }

    fn act(&mut self, view: &LeverView<'_>, rng: &mut Rng) -> Vec<f64> {
        let current = view.params.to_vec();
        let known = self.baseline.is_some() && (!self.reanchor || self.matches_anchor(&current));
        if !known {
            // Feel the dial out before pushing it.
            self.anchor = Some(current.clone());
            self.baseline = None;
            self.pending = Some(vec![0.0; current.len()]);
            return current;
        }
        self.anchor = Some(current.clone());
        let delta = self.displacement(current.len(), rng);
        let proposal: Vec<f64> = current.iter().zip(&delta).map(|(c, d)| c + d).collect();
        self.pending = Some(delta);
        proposal
    }

    fn feedback(&mut self, outcome: &Outcome, _rng: &mut Rng) {
        let Some(delta) = self.pending.as_ref() else {
            return;
        };
        if !outcome.reward.is_finite() {
            self.pending = None;
            return;
        }
        let evaluating = delta.iter().all(|d| *d == 0.0);
        match self.baseline {
            None => {
                self.baseline = Some(outcome.reward);
                self.evaluations += 1;
                // The evaluated surface becomes the anchor as-is.
                self.pending = None;
            }
            Some(baseline) if outcome.reward >= baseline => {
                self.baseline = Some(outcome.reward);
                if !evaluating {
                    self.step = (self.step * 1.3).min(self.step_max);
                    self.accepted += 1;
                    if let Some(anchor) = self.anchor.as_mut() {
                        for (a, d) in anchor.iter_mut().zip(delta) {
                            *a += d;
                        }
                    }
                }
                self.pending = None;
            }
            Some(_) => {
                self.step = (self.step * 0.94).max(self.step_min);
                self.rejected += 1;
                // Keep `pending` so `release` can revert the displacement.
            }
        }
        if let Some(b) = self.baseline.as_mut() {
            *b -= self.forgetting * b.abs();
        }
    }

    fn release(&mut self, view: &LeverView<'_>, _rng: &mut Rng) -> Option<Vec<f64>> {
        let delta = self.pending.take()?;
        if delta.len() != view.params.len() {
            return None;
        }
        // Rejected probe: put the dial back where it was.
        let reverted: Vec<f64> = view.params.iter().zip(&delta).map(|(p, d)| p - d).collect();
        self.anchor = Some(reverted.clone());
        Some(reverted)
    }

    fn summary(&self) -> String {
        format!(
            "{} step {:.3}, accepted {}/{} probes, {} evaluations, baseline {}",
            self.name,
            self.step,
            self.accepted,
            self.accepted + self.rejected,
            self.evaluations,
            self.baseline
                .map(|b| format!("{b:.2}"))
                .unwrap_or_else(|| "none".to_string())
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(p: &[f64]) -> f64 {
        -p.iter().map(|x| x * x).sum::<f64>()
    }

    fn run(mut learner: HillClimber, turns: u64) -> f64 {
        let mut rng = Rng::seed_from_u64(3);
        let mut params = vec![2.0; 17];
        for turn in 0..turns {
            let view = LeverView {
                turn,
                lever: 0,
                params: &params,
                last_reward: None,
                revealed_node: None,
            };
            let proposal = learner.act(&view, &mut rng);
            let reward = sphere(&proposal);
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
            params = proposal;
            let view = LeverView {
                turn,
                lever: 0,
                params: &params,
                last_reward: Some(reward),
                revealed_node: None,
            };
            if let Some(left) = learner.release(&view, &mut rng) {
                params = left;
            }
        }
        sphere(&params)
    }

    #[test]
    fn climbs_a_sphere() {
        let start = sphere(&[2.0; 17]);
        let end = run(HillClimber::new(0.3).with_forgetting(0.0), 400);
        assert!(end > start * 0.25, "start {start}, end {end}");
    }

    #[test]
    fn rotation_mutation_also_improves() {
        let start = sphere(&[2.0; 17]);
        let end = run(HillClimber::new(0.3).with_mutation(Mutation::Mixed), 400);
        assert!(end > start * 0.5, "start {start}, end {end}");
    }

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
    fn evaluates_first_then_probes_and_reverts_rejections() {
        let mut rng = Rng::seed_from_u64(1);
        let mut hc = HillClimber::new(0.5);
        let params = vec![1.0; 17];
        let view = LeverView {
            turn: 0,
            lever: 0,
            params: &params,
            last_reward: None,
            revealed_node: None,
        };
        // Unknown surface: evaluated as found.
        assert_eq!(hc.act(&view, &mut rng), params);
        hc.feedback(&outcome(0, 10.0), &mut rng);
        assert_eq!(hc.baseline(), Some(10.0 - 0.2));
        assert!(hc.release(&view, &mut rng).is_none());

        // Known surface: a displaced probe.
        let probe = hc.act(&view, &mut rng);
        assert_ne!(probe, params);
        hc.feedback(&outcome(1, 20.0), &mut rng);
        let probe_view = LeverView {
            params: &probe,
            ..view
        };
        assert!(
            hc.release(&probe_view, &mut rng).is_none(),
            "accepted probes stay"
        );

        // A worse probe is reverted exactly.
        let probe2 = hc.act(&probe_view, &mut rng);
        hc.feedback(&outcome(2, -5.0), &mut rng);
        let probe2_view = LeverView {
            params: &probe2,
            ..view
        };
        let reverted = hc.release(&probe2_view, &mut rng).unwrap();
        for (r, p) in reverted.iter().zip(&probe) {
            assert!((r - p).abs() < 1e-12);
        }
        assert_eq!(hc.counts(), (1, 1, 1));
        assert!(hc.summary().contains("accepted 1/2"));
    }

    #[test]
    fn a_foreign_surface_is_never_overwritten() {
        let mut rng = Rng::seed_from_u64(2);
        let mut hc = HillClimber::new(0.5);
        let mine = vec![0.0; 17];
        let view = LeverView {
            turn: 0,
            lever: 0,
            params: &mine,
            last_reward: None,
            revealed_node: None,
        };
        hc.act(&view, &mut rng);
        hc.feedback(&outcome(0, 1.0), &mut rng);
        let foreign = vec![5.0; 17];
        let foreign_view = LeverView {
            params: &foreign,
            ..view
        };
        // Different surface: evaluated as found, not replaced by anything remembered.
        assert_eq!(hc.act(&foreign_view, &mut rng), foreign);
        hc.feedback(&outcome(1, 50.0), &mut rng);
        assert!(hc.release(&foreign_view, &mut rng).is_none());
        let probe = hc.act(&foreign_view, &mut rng);
        let max_move = probe
            .iter()
            .zip(&foreign)
            .map(|(p, f)| (p - f).abs())
            .fold(0.0, f64::max);
        assert!(max_move < 3.0, "moves are bounded by the step size");
    }
}
