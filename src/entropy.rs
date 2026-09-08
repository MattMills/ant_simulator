//! Entropy of action distributions and the entropy dial.
//!
//! The central control primitive of the crate is *entropic control*: instead
//! of choosing a temperature and accepting whatever randomness falls out, a
//! controller states how much disorder it wants (as a fraction of the maximum
//! attainable entropy) and the temperature is solved for so that the resulting
//! action distribution has exactly that entropy.
//!
//! Every level of the hierarchy carries an [`EntropyControl`]. The root sets an
//! absolute level; descendants scale what they inherit. The composition along
//! a root-to-leaf path yields the entropy fraction actually applied to an
//! ant's decision.

/// Logistic function.
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Inverse of [`sigmoid`]; the argument is clamped away from 0 and 1.
pub fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-9, 1.0 - 1e-9);
    (p / (1.0 - p)).ln()
}

/// Shannon entropy in nats of a probability vector (zeros contribute nothing).
pub fn entropy(probs: &[f64]) -> f64 {
    let mut h = 0.0;
    for &p in probs {
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h.max(0.0)
}

/// Maximum entropy of a distribution over `n` outcomes: `ln n`.
pub fn max_entropy(n: usize) -> f64 {
    if n == 0 {
        0.0
    } else {
        (n as f64).ln()
    }
}

/// Softmax of `logits / temperature` into `out`.
///
/// Logits of `-inf` denote masked (impossible) actions and receive
/// probability zero. If every logit is masked, `out` is filled with zeros.
pub fn softmax(logits: &[f64], temperature: f64, out: &mut [f64]) {
    debug_assert_eq!(logits.len(), out.len());
    let t = temperature.max(1e-12);
    let max = logits
        .iter()
        .copied()
        .filter(|l| l.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        return;
    }
    let mut sum = 0.0;
    for (o, &l) in out.iter_mut().zip(logits) {
        *o = if l.is_finite() {
            ((l - max) / t).exp()
        } else {
            0.0
        };
        sum += *o;
    }
    if sum > 0.0 {
        for o in out.iter_mut() {
            *o /= sum;
        }
    }
}

const LN_T_MIN: f64 = -9.0;
const LN_T_MAX: f64 = 9.0;

/// Find the temperature at which `softmax(logits / T)` has entropy `target`
/// (in nats). Uses bisection on `ln T`; `scratch` must have `logits.len()`
/// entries and is clobbered. The result is clamped to `[e^-9, e^9]`.
pub fn temperature_for_entropy(logits: &[f64], target: f64, scratch: &mut [f64]) -> f64 {
    let mut lo = LN_T_MIN;
    let mut hi = LN_T_MAX;
    let h_at = |ln_t: f64, scratch: &mut [f64]| {
        softmax(logits, ln_t.exp(), scratch);
        entropy(scratch)
    };
    if target <= h_at(lo, scratch) {
        return lo.exp();
    }
    if target >= h_at(hi, scratch) {
        return hi.exp();
    }
    for _ in 0..22 {
        let mid = 0.5 * (lo + hi);
        if h_at(mid, scratch) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (0.5 * (lo + hi)).exp()
}

/// Turn logits into a distribution whose entropy is `fraction` of the maximum
/// entropy over the unmasked actions. Returns `(temperature, entropy)`.
pub fn tempered_distribution(logits: &[f64], fraction: f64, out: &mut [f64]) -> (f64, f64) {
    let valid = logits.iter().filter(|l| l.is_finite()).count();
    if valid == 0 {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        return (1.0, 0.0);
    }
    let target = fraction.clamp(0.0, 1.0) * max_entropy(valid);
    let t = temperature_for_entropy(logits, target, out);
    softmax(logits, t, out);
    let h = entropy(out);
    (t, h)
}

/// The entropy dial carried by every node of the hierarchy.
///
/// Both variants store an unconstrained real `raw` so learners can treat
/// every parameter as a free real number:
///
/// * `Absolute`: the effective fraction is `sigmoid(raw)`, ignoring the parent.
/// * `Relative`: the effective fraction is `parent × exp(raw)` (clamped to
///   `[0, 1]`); with no parent it scales a neutral base of `0.5`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EntropyControl {
    /// Set the entropy fraction outright.
    Absolute {
        /// Logit of the fraction.
        raw: f64,
    },
    /// Scale the inherited entropy fraction.
    Relative {
        /// Natural log of the gain.
        raw: f64,
    },
}

impl EntropyControl {
    /// Neutral base used by a `Relative` control that has no parent.
    pub const ORPHAN_BASE: f64 = 0.5;

    /// Absolute control at the given fraction of maximum entropy.
    pub fn absolute(fraction: f64) -> Self {
        EntropyControl::Absolute {
            raw: logit(fraction),
        }
    }

    /// Relative control with the given multiplicative gain.
    pub fn relative(gain: f64) -> Self {
        EntropyControl::Relative {
            raw: gain.max(1e-12).ln(),
        }
    }

    /// The unconstrained parameter.
    pub fn raw(&self) -> f64 {
        match self {
            EntropyControl::Absolute { raw } | EntropyControl::Relative { raw } => *raw,
        }
    }

    /// Overwrite the unconstrained parameter (non-finite values are ignored).
    pub fn set_raw(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        match self {
            EntropyControl::Absolute { raw } | EntropyControl::Relative { raw } => *raw = value,
        }
    }

    /// Whether this is an `Absolute` control.
    pub fn is_absolute(&self) -> bool {
        matches!(self, EntropyControl::Absolute { .. })
    }

    /// Set the fraction (for `Absolute`) or gain (for `Relative`) directly.
    pub fn set_level(&mut self, level: f64) {
        match self {
            EntropyControl::Absolute { raw } => *raw = logit(level),
            EntropyControl::Relative { raw } => *raw = level.max(1e-12).ln(),
        }
    }

    /// Effective entropy fraction given the parent's effective fraction.
    pub fn effective(&self, parent: Option<f64>) -> f64 {
        match self {
            EntropyControl::Absolute { raw } => sigmoid(*raw),
            EntropyControl::Relative { raw } => {
                let base = parent.unwrap_or(Self::ORPHAN_BASE);
                (base * raw.exp()).clamp(0.0, 1.0)
            }
        }
    }

    /// Human-readable description.
    pub fn describe(&self) -> String {
        match self {
            EntropyControl::Absolute { raw } => format!("absolute {:.3}", sigmoid(*raw)),
            EntropyControl::Relative { raw } => format!("gain ×{:.3}", raw.exp()),
        }
    }
}

impl Default for EntropyControl {
    fn default() -> Self {
        EntropyControl::relative(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_basics() {
        assert_eq!(entropy(&[1.0, 0.0]), 0.0);
        let h = entropy(&[0.25; 4]);
        assert!((h - max_entropy(4)).abs() < 1e-12);
    }

    #[test]
    fn softmax_masks_and_limits() {
        let logits = [1.0, f64::NEG_INFINITY, 3.0];
        let mut out = [0.0; 3];
        softmax(&logits, 1.0, &mut out);
        assert_eq!(out[1], 0.0);
        assert!((out.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        softmax(&logits, 1e-6, &mut out);
        assert!((out[2] - 1.0).abs() < 1e-9);
        softmax(&logits, 1e6, &mut out);
        assert!((out[0] - 0.5).abs() < 1e-3);
        softmax(&[f64::NEG_INFINITY; 2], 1.0, &mut out[..2]);
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn tempering_hits_entropy_target() {
        let logits = [0.3, -1.2, 2.0, 0.0, f64::NEG_INFINITY, 1.1, -0.4, 0.9];
        let mut out = [0.0; 8];
        for fraction in [0.05, 0.2, 0.5, 0.8, 0.95] {
            let (_t, h) = tempered_distribution(&logits, fraction, &mut out);
            let target = fraction * max_entropy(7);
            assert!(
                (h - target).abs() < 1e-3,
                "fraction {fraction}: got {h}, want {target}"
            );
            assert_eq!(out[4], 0.0);
        }
        let (_, h0) = tempered_distribution(&logits, 0.0, &mut out);
        assert!(h0 < 1e-3);
        assert!((out[2] - 1.0).abs() < 1e-3);
        let (_, h1) = tempered_distribution(&logits, 1.0, &mut out);
        assert!((h1 - max_entropy(7)).abs() < 1e-3);
    }

    #[test]
    fn entropy_control_composition() {
        let root = EntropyControl::absolute(0.4);
        assert!((root.effective(None) - 0.4).abs() < 1e-9);
        let child = EntropyControl::relative(2.0);
        assert!((child.effective(Some(0.4)) - 0.8).abs() < 1e-9);
        assert_eq!(child.effective(Some(0.9)), 1.0);
        assert!((child.effective(None) - 1.0).abs() < 1e-9);
        let mut c = EntropyControl::default();
        assert!((c.effective(Some(0.3)) - 0.3).abs() < 1e-9);
        c.set_raw(f64::NAN);
        assert_eq!(c.raw(), 0.0);
        c.set_level(0.5);
        assert!((c.effective(Some(0.6)) - 0.3).abs() < 1e-9);
    }
}
