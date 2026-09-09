//! The cross-entropy method over displacements of the surface it holds.

use super::{Learner, LeverView, Outcome};
use crate::rng::Rng;

/// Cross-entropy method (CEM) on displacements.
///
/// Each turn the learner probes `current + δ` with `δ ~ N(0, σ²)` (diagonal).
/// After every `population` probes it takes the `elite` best displacements,
/// applies their mean to the surface it is holding at that moment, and refits
/// `σ` to their spread. Between refits every probe is reverted, so the
/// surface only ever moves by one recombined displacement per generation.
///
/// The learner ignores the rotation entirely, which makes it a useful foil
/// for [`super::PhaseAware`].
#[derive(Clone, Debug)]
pub struct CrossEntropy {
    population: usize,
    elite: usize,
    init_std: f64,
    min_std: f64,
    smoothing: f64,
    inflation: f64,
    std: Vec<f64>,
    batch: Vec<(Vec<f64>, f64)>,
    pending: Option<Vec<f64>>,
    recombined: Option<Vec<f64>>,
    generations: u64,
}

impl CrossEntropy {
    /// CEM with the given population size and number of elites.
    pub fn new(population: usize, elite: usize) -> Self {
        let population = population.max(2);
        CrossEntropy {
            population,
            elite: elite.clamp(1, population),
            init_std: 0.3,
            min_std: 0.05,
            smoothing: 0.3,
            inflation: 1.1,
            std: Vec::new(),
            batch: Vec::new(),
            pending: None,
            recombined: None,
            generations: 0,
        }
    }

    /// Initial sampling standard deviation.
    pub fn with_init_std(mut self, std: f64) -> Self {
        self.init_std = std.max(0.0);
        self
    }

    /// Floor on the standard deviation after refitting.
    pub fn with_min_std(mut self, std: f64) -> Self {
        self.min_std = std.max(0.0);
        self
    }

    /// Fraction of the old standard deviation kept when refitting.
    pub fn with_smoothing(mut self, smoothing: f64) -> Self {
        self.smoothing = smoothing.clamp(0.0, 1.0);
        self
    }

    /// Factor applied to the measured elite spread when refitting.
    pub fn with_inflation(mut self, inflation: f64) -> Self {
        self.inflation = inflation.max(0.0);
        self
    }

    /// Current sampling standard deviations (empty before the first act).
    pub fn std(&self) -> &[f64] {
        &self.std
    }

    /// Completed refits.
    pub fn generations(&self) -> u64 {
        self.generations
    }

    fn refit(&mut self) {
        self.batch
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let dim = self.batch[0].0.len();
        let elite = &self.batch[..self.elite.min(self.batch.len())];
        let k = elite.len() as f64;
        let mut mean = vec![0.0; dim];
        let mut rms = vec![0.0; dim];
        for (d, _) in elite {
            for ((m, r), x) in mean.iter_mut().zip(rms.iter_mut()).zip(d) {
                *m += x / k;
                *r += x * x / k;
            }
        }
        // Spread measured around zero rather than around the elite mean: it
        // includes the recombined move, so the search cannot collapse while
        // it is still travelling. The inflation counters the downward bias
        // of a spread estimated from a handful of selected samples.
        let a = self.smoothing;
        for (s, r) in self.std.iter_mut().zip(&rms) {
            *s = (a * *s + (1.0 - a) * self.inflation * r.sqrt()).max(self.min_std);
        }
        self.recombined = Some(mean);
        self.batch.clear();
        self.generations += 1;
    }
}

impl Default for CrossEntropy {
    fn default() -> Self {
        CrossEntropy::new(8, 3)
    }
}

impl Learner for CrossEntropy {
    fn name(&self) -> &str {
        "cross-entropy"
    }

    fn act(&mut self, view: &LeverView<'_>, rng: &mut Rng) -> Vec<f64> {
        let dim = view.params.len();
        if self.std.len() != dim {
            self.std = vec![self.init_std; dim];
            self.batch.clear();
            self.recombined = None;
        }
        let delta: Vec<f64> = self.std.iter().map(|s| s * rng.normal()).collect();
        let proposal = view.params.iter().zip(&delta).map(|(p, d)| p + d).collect();
        self.pending = Some(delta);
        proposal
    }

    fn feedback(&mut self, outcome: &Outcome, _rng: &mut Rng) {
        let Some(delta) = self.pending.as_ref() else {
            return;
        };
        if outcome.reward.is_finite() {
            self.batch.push((delta.clone(), outcome.reward));
        }
        if self.batch.len() >= self.population {
            self.refit();
        }
    }

    fn release(&mut self, view: &LeverView<'_>, _rng: &mut Rng) -> Option<Vec<f64>> {
        let delta = self.pending.take()?;
        if delta.len() != view.params.len() {
            return None;
        }
        let mov = self
            .recombined
            .take()
            .unwrap_or_else(|| vec![0.0; delta.len()]);
        Some(
            view.params
                .iter()
                .zip(&delta)
                .zip(&mov)
                .map(|((p, d), m)| p - d + m)
                .collect(),
        )
    }

    fn summary(&self) -> String {
        let spread = if self.std.is_empty() {
            0.0
        } else {
            self.std.iter().sum::<f64>() / self.std.len() as f64
        };
        format!(
            "cross-entropy {} generations, mean σ {:.3}, batch {}/{}",
            self.generations,
            spread,
            self.batch.len(),
            self.population
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converges_on_a_quadratic() {
        let mut rng = Rng::seed_from_u64(6);
        let mut cem = CrossEntropy::new(10, 3).with_init_std(0.5);
        let target = vec![1.0; 17];
        let mut params = vec![0.0; 17];
        for turn in 0..400u64 {
            let view = LeverView {
                turn,
                lever: 0,
                params: &params,
                last_reward: None,
                revealed_node: None,
            };
            let p = cem.act(&view, &mut rng);
            let reward = -p
                .iter()
                .zip(&target)
                .map(|(x, t)| (x - t).powi(2))
                .sum::<f64>();
            cem.feedback(
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
            let probe_view = LeverView { params: &p, ..view };
            params = cem.release(&probe_view, &mut rng).unwrap();
        }
        let err: f64 = params.iter().map(|x| (x - 1.0).abs()).sum::<f64>() / 17.0;
        assert!(err < 0.3, "mean abs error {err}");
        assert_eq!(cem.generations(), 40);
        assert!(cem.summary().contains("generations"));
    }

    #[test]
    fn probes_are_reverted_between_generations() {
        let mut rng = Rng::seed_from_u64(1);
        let mut cem = CrossEntropy::new(4, 2);
        let params = vec![1.0; 5];
        let view = LeverView {
            turn: 0,
            lever: 0,
            params: &params,
            last_reward: None,
            revealed_node: None,
        };
        let p = cem.act(&view, &mut rng);
        cem.feedback(
            &Outcome {
                turn: 0,
                lever: 0,
                reward: 1.0,
                score: None,
                decisions: 0,
                mean_entropy: 0.0,
            },
            &mut rng,
        );
        let probe_view = LeverView { params: &p, ..view };
        let left = cem.release(&probe_view, &mut rng).unwrap();
        for (l, q) in left.iter().zip(&params) {
            assert!((l - q).abs() < 1e-12);
        }
    }
}
