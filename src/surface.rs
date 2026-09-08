//! The entropic behavioral surface.
//!
//! A [`BehavioralSurface`] is the general object that lives at every level of
//! the hierarchy. It has two parts:
//!
//! * a weight vector over the ant's sensory features, which scores each
//!   candidate action (the *surface*: a linear map from observations to
//!   preferences), and
//! * an [`EntropyControl`] that says how much disorder the resulting action
//!   distribution must carry (the *entropic* part).
//!
//! Surfaces compose additively along a root-to-leaf path, and their entropy
//! dials compose multiplicatively, so a controller high in the hierarchy
//! shapes every entity beneath it.
//!
//! Every surface is also a flat parameter vector (`FEATURES + 1` real
//! numbers), which is what learners read and write.

use crate::ant::FEATURES;
use crate::entropy::EntropyControl;
use crate::rng::Rng;
use std::fmt;

/// Number of free parameters in one surface: the weights plus the entropy dial.
pub const PARAM_LEN: usize = FEATURES + 1;

/// Index of the entropy parameter inside a flat parameter vector.
pub const ENTROPY_PARAM: usize = FEATURES;

/// Error returned when a parameter vector has the wrong shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SurfaceError {
    /// Expected length.
    pub expected: usize,
    /// Length that was supplied.
    pub got: usize,
}

impl fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "surface parameter vector has length {}, expected {}",
            self.got, self.expected
        )
    }
}

impl std::error::Error for SurfaceError {}

/// Weights over sensory features plus an entropy dial.
#[derive(Clone, Debug, PartialEq)]
pub struct BehavioralSurface {
    /// One weight per feature (see [`crate::ant::FEATURE_NAMES`]).
    pub weights: [f64; FEATURES],
    /// How much disorder the actions drawn through this surface carry.
    pub entropy: EntropyControl,
}

impl BehavioralSurface {
    /// A surface with the given weights and dial.
    pub fn new(weights: [f64; FEATURES], entropy: EntropyControl) -> Self {
        BehavioralSurface { weights, entropy }
    }

    /// A surface that adds nothing to its parent (zero weights, unit gain).
    pub fn neutral() -> Self {
        BehavioralSurface {
            weights: [0.0; FEATURES],
            entropy: EntropyControl::relative(1.0),
        }
    }

    /// A hand-tuned foraging instinct: follow food pheromone and pick up food
    /// when empty-handed, follow home pheromone and head for the nest when
    /// carrying, keep some momentum, avoid recently visited cells.
    pub fn instinct() -> Self {
        // Base (foraging) weights, indexed like `FEATURE_NAMES[0..8]`.
        let foraging = [2.0, -0.3, 4.0, -1.0, 1.2, -0.4, -2.0, -0.5];
        // Additive adjustments applied while carrying food.
        let carrying = [-2.0, 2.3, -4.0, 5.0, -0.2, 2.4, 1.0, 0.5];
        let mut weights = [0.0; FEATURES];
        weights[..8].copy_from_slice(&foraging);
        weights[8..].copy_from_slice(&carrying);
        BehavioralSurface {
            weights,
            entropy: EntropyControl::absolute(0.35),
        }
    }

    /// Score of one candidate action's feature vector.
    pub fn logit(&self, features: &[f64; FEATURES]) -> f64 {
        dot(&self.weights, features)
    }

    /// Flat parameter vector: the weights followed by the raw entropy value.
    pub fn to_params(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(PARAM_LEN);
        v.extend_from_slice(&self.weights);
        v.push(self.entropy.raw());
        v
    }

    /// Write the flat parameter vector into `out` (must have `PARAM_LEN` entries).
    pub fn write_params(&self, out: &mut [f64]) -> Result<(), SurfaceError> {
        if out.len() != PARAM_LEN {
            return Err(SurfaceError {
                expected: PARAM_LEN,
                got: out.len(),
            });
        }
        out[..FEATURES].copy_from_slice(&self.weights);
        out[ENTROPY_PARAM] = self.entropy.raw();
        Ok(())
    }

    /// Overwrite the surface from a flat parameter vector. Non-finite
    /// entries are ignored (the previous value is kept), so learners cannot
    /// poison a surface with NaNs.
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), SurfaceError> {
        if params.len() != PARAM_LEN {
            return Err(SurfaceError {
                expected: PARAM_LEN,
                got: params.len(),
            });
        }
        for (w, &p) in self.weights.iter_mut().zip(&params[..FEATURES]) {
            if p.is_finite() {
                *w = p;
            }
        }
        self.entropy.set_raw(params[ENTROPY_PARAM]);
        Ok(())
    }

    /// Add Gaussian noise with standard deviation `sigma` to every parameter.
    pub fn perturb(&mut self, sigma: f64, rng: &mut Rng) {
        for w in self.weights.iter_mut() {
            *w += rng.normal_with(0.0, sigma);
        }
        let raw = self.entropy.raw() + rng.normal_with(0.0, sigma);
        self.entropy.set_raw(raw);
    }

    /// Rotate the weight vector by `angle` radians in a random coordinate plane.
    pub fn rotate(&mut self, angle: f64, rng: &mut Rng) {
        random_rotation(&mut self.weights, angle, rng);
    }

    /// Euclidean norm of the weight vector.
    pub fn norm(&self) -> f64 {
        self.weights.iter().map(|w| w * w).sum::<f64>().sqrt()
    }
}

impl Default for BehavioralSurface {
    fn default() -> Self {
        BehavioralSurface::instinct()
    }
}

/// Dot product of two equal-length slices.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Apply a Givens rotation by `angle` in the `(i, j)` coordinate plane.
pub fn givens_rotation(v: &mut [f64], i: usize, j: usize, angle: f64) {
    if i == j || i >= v.len() || j >= v.len() {
        return;
    }
    let (s, c) = angle.sin_cos();
    let vi = v[i];
    let vj = v[j];
    v[i] = c * vi - s * vj;
    v[j] = s * vi + c * vj;
}

/// Rotate `v` by `angle` in a uniformly random coordinate plane.
/// Vectors shorter than two entries are left untouched.
pub fn random_rotation(v: &mut [f64], angle: f64, rng: &mut Rng) {
    let n = v.len();
    if n < 2 {
        return;
    }
    let i = rng.below(n);
    let mut j = rng.below(n - 1);
    if j >= i {
        j += 1;
    }
    givens_rotation(v, i, j, angle);
}

/// Euclidean distance between two parameter vectors.
pub fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_round_trip() {
        let s = BehavioralSurface::instinct();
        let p = s.to_params();
        assert_eq!(p.len(), PARAM_LEN);
        let mut t = BehavioralSurface::neutral();
        t.set_params(&p).unwrap();
        assert_eq!(t.weights, s.weights);
        assert!((t.entropy.raw() - s.entropy.raw()).abs() < 1e-12);
        assert!(t.set_params(&p[..3]).is_err());
    }

    #[test]
    fn nan_params_are_ignored() {
        let mut s = BehavioralSurface::instinct();
        let before = s.clone();
        let mut p = s.to_params();
        p[0] = f64::NAN;
        p[ENTROPY_PARAM] = f64::INFINITY;
        s.set_params(&p).unwrap();
        assert_eq!(s, before);
    }

    #[test]
    fn rotation_preserves_norm() {
        let mut rng = Rng::seed_from_u64(5);
        let mut s = BehavioralSurface::instinct();
        let n0 = s.norm();
        for _ in 0..50 {
            s.rotate(0.3, &mut rng);
        }
        assert!((s.norm() - n0).abs() < 1e-9);
        assert!(distance(&s.to_params(), &BehavioralSurface::instinct().to_params()) > 0.1);
    }

    #[test]
    fn givens_is_a_rotation() {
        let mut v = [1.0, 0.0, 5.0];
        givens_rotation(&mut v, 0, 1, std::f64::consts::FRAC_PI_2);
        assert!(v[0].abs() < 1e-12);
        assert!((v[1] - 1.0).abs() < 1e-12);
        assert_eq!(v[2], 5.0);
        givens_rotation(&mut v, 0, 0, 1.0);
        givens_rotation(&mut v, 0, 9, 1.0);
        assert!((v[1] - 1.0).abs() < 1e-12);
    }
}
