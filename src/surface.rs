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
//! A surface also carries a [`Deformation`]: how its entropy budget is spent
//! geometrically on the ring of candidate directions (blur along the ring,
//! random roughening, and the reach of the sucker that selects on it; see
//! [`crate::landscape`]).
//!
//! Every surface is also a flat parameter vector ([`PARAM_LEN`] real
//! numbers), which is what learners read and write.

use crate::ant::FEATURES;
use crate::entropy::EntropyControl;
use crate::rng::Rng;
use std::fmt;

/// Number of free parameters in one surface: the weights, the entropy dial,
/// and the three deformation parameters.
pub const PARAM_LEN: usize = FEATURES + 4;

/// Index of the entropy parameter inside a flat parameter vector.
pub const ENTROPY_PARAM: usize = FEATURES;

/// Index of the smoothing parameter.
pub const SMOOTH_PARAM: usize = FEATURES + 1;

/// Index of the roughening parameter.
pub const ROUGH_PARAM: usize = FEATURES + 2;

/// Index of the sucker-reach parameter.
pub const REACH_PARAM: usize = FEATURES + 3;

/// How a surface spends its entropy budget geometrically.
///
/// All three values are unconstrained reals that compose additively along
/// the hierarchy. `smooth` and `rough` map to shares in `[0, 1)` through
/// `1 - exp(-x²)`, so zero means "none" exactly and either sign turns the
/// channel on. `reach` scales the sucker's base reach by `exp(reach)`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Deformation {
    /// Blur along the ring (raw).
    pub smooth: f64,
    /// Random roughening of the ring (raw).
    pub rough: f64,
    /// Log-scale of the sucker's reach (raw).
    pub reach: f64,
}

impl Deformation {
    /// No deformation: pure tempering, base reach.
    pub fn none() -> Self {
        Deformation::default()
    }

    /// Share in `[0, 1)` implied by a raw smoothing or roughening value.
    pub fn share(raw: f64) -> f64 {
        1.0 - (-raw * raw).exp()
    }

    /// Smoothing share.
    pub fn smooth_share(&self) -> f64 {
        Self::share(self.smooth)
    }

    /// Roughening share.
    pub fn rough_share(&self) -> f64 {
        Self::share(self.rough)
    }

    /// Multiplier applied to the base reach.
    pub fn reach_scale(&self) -> f64 {
        self.reach.exp()
    }

    /// Whether every channel is off.
    pub fn is_none(&self) -> bool {
        self.smooth == 0.0 && self.rough == 0.0 && self.reach == 0.0
    }

    /// Human-readable description.
    pub fn describe(&self) -> String {
        format!(
            "smooth {:.2}, rough {:.2}, reach ×{:.2}",
            self.smooth_share(),
            self.rough_share(),
            self.reach_scale()
        )
    }
}

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

/// Weights over sensory features, an entropy dial, and a deformation.
#[derive(Clone, Debug, PartialEq)]
pub struct BehavioralSurface {
    /// One weight per feature (see [`crate::ant::FEATURE_NAMES`]).
    pub weights: [f64; FEATURES],
    /// How much disorder the actions drawn through this surface carry.
    pub entropy: EntropyControl,
    /// How that disorder is shaped on the ring.
    pub deformation: Deformation,
}

impl BehavioralSurface {
    /// A surface with the given weights and dial and no deformation.
    pub fn new(weights: [f64; FEATURES], entropy: EntropyControl) -> Self {
        BehavioralSurface {
            weights,
            entropy,
            deformation: Deformation::none(),
        }
    }

    /// A surface that adds nothing to its parent (zero weights, unit gain,
    /// no deformation).
    pub fn neutral() -> Self {
        BehavioralSurface {
            weights: [0.0; FEATURES],
            entropy: EntropyControl::relative(1.0),
            deformation: Deformation::none(),
        }
    }

    /// Set the deformation (builder style).
    pub fn with_deformation(mut self, deformation: Deformation) -> Self {
        self.deformation = deformation;
        self
    }

    /// The foraging instinct of the default species
    /// ([`crate::species::Species::lasius_niger`]): Deneubourg's choice
    /// exponent on the recruitment trail, path integration home, site
    /// fidelity out, at a fixed temperature of 1 so the pheromone response
    /// is the published choice function.
    pub fn instinct() -> Self {
        crate::species::Species::lasius_niger().instinct()
    }

    /// Score of one candidate action's feature vector.
    pub fn logit(&self, features: &[f64; FEATURES]) -> f64 {
        dot(&self.weights, features)
    }

    /// Flat parameter vector: the weights, the raw entropy value, then the
    /// raw smoothing, roughening and reach values.
    pub fn to_params(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(PARAM_LEN);
        v.extend_from_slice(&self.weights);
        v.push(self.entropy.raw());
        v.push(self.deformation.smooth);
        v.push(self.deformation.rough);
        v.push(self.deformation.reach);
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
        out[SMOOTH_PARAM] = self.deformation.smooth;
        out[ROUGH_PARAM] = self.deformation.rough;
        out[REACH_PARAM] = self.deformation.reach;
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
        for (slot, &value) in [
            (&mut self.deformation.smooth, &params[SMOOTH_PARAM]),
            (&mut self.deformation.rough, &params[ROUGH_PARAM]),
            (&mut self.deformation.reach, &params[REACH_PARAM]),
        ] {
            if value.is_finite() {
                *slot = value;
            }
        }
        Ok(())
    }

    /// Add Gaussian noise with standard deviation `sigma` to every parameter.
    pub fn perturb(&mut self, sigma: f64, rng: &mut Rng) {
        for w in self.weights.iter_mut() {
            *w += rng.normal_with(0.0, sigma);
        }
        let raw = self.entropy.raw() + rng.normal_with(0.0, sigma);
        self.entropy.set_raw(raw);
        self.deformation.smooth += rng.normal_with(0.0, sigma);
        self.deformation.rough += rng.normal_with(0.0, sigma);
        self.deformation.reach += rng.normal_with(0.0, sigma);
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
        p[REACH_PARAM] = f64::NAN;
        s.set_params(&p).unwrap();
        assert_eq!(s, before);
    }

    #[test]
    fn deformation_shares_and_params() {
        let d = Deformation::none();
        assert!(d.is_none());
        assert_eq!(d.smooth_share(), 0.0);
        assert_eq!(d.reach_scale(), 1.0);
        let d = Deformation {
            smooth: 1.0,
            rough: -1.0,
            reach: (2f64).ln(),
        };
        assert!((d.smooth_share() - (1.0 - (-1f64).exp())).abs() < 1e-12);
        assert_eq!(d.smooth_share(), d.rough_share());
        assert!((d.reach_scale() - 2.0).abs() < 1e-12);
        let s = BehavioralSurface::neutral().with_deformation(d);
        let p = s.to_params();
        assert_eq!(p.len(), PARAM_LEN);
        assert_eq!(p[SMOOTH_PARAM], 1.0);
        assert_eq!(p[ROUGH_PARAM], -1.0);
        let mut t = BehavioralSurface::instinct();
        t.set_params(&p).unwrap();
        assert_eq!(t.deformation, d);
        assert!(d.describe().contains("reach ×2.00"));
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
