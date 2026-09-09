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

/// How a landscape is to be tempered: by an entropy target or by a fixed
/// temperature. This is what composing the dials along a path yields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tempering {
    /// Solve the temperature so the distribution has this fraction of the
    /// maximum entropy (entropic control).
    Entropy(f64),
    /// Divide the logits by this temperature and let the entropy fall where
    /// it may (the biological choice function; `1.0` is Deneubourg's).
    Temperature(f64),
}

impl Tempering {
    /// The entropy fraction, if this is an entropy target.
    pub fn fraction(&self) -> Option<f64> {
        match self {
            Tempering::Entropy(f) => Some(*f),
            Tempering::Temperature(_) => None,
        }
    }

    /// The temperature, if fixed.
    pub fn temperature(&self) -> Option<f64> {
        match self {
            Tempering::Entropy(_) => None,
            Tempering::Temperature(t) => Some(*t),
        }
    }

    /// Scale the temperature (or, for an entropy target, raise the target
    /// towards uniform) by a factor above one; used while searching.
    pub fn heated(&self, factor: f64) -> Tempering {
        match self {
            Tempering::Entropy(f) => {
                Tempering::Entropy((1.0 - (1.0 - f) / factor.max(1e-9)).clamp(0.0, 1.0))
            }
            Tempering::Temperature(t) => Tempering::Temperature(t * factor.max(1e-9)),
        }
    }

    /// Human-readable description.
    pub fn describe(&self) -> String {
        match self {
            Tempering::Entropy(f) => format!("entropy {f:.3}"),
            Tempering::Temperature(t) => format!("temperature {t:.3}"),
        }
    }
}

/// The entropy dial carried by every node of the hierarchy.
///
/// All variants store an unconstrained real `raw` so learners can treat
/// every parameter as a free real number:
///
/// * `Absolute`: the effective entropy fraction is `sigmoid(raw)`, ignoring
///   the parent.
/// * `Fixed`: the effective temperature is `exp(raw)`, ignoring the parent.
///   With `raw = 0` the logits are used as they are, which for the
///   pheromone features is Deneubourg's choice function.
/// * `Relative`: scales what it inherits by `exp(raw)`: an entropy fraction
///   (clamped to `[0, 1]`) or a temperature. With no parent it scales a
///   neutral entropy fraction of `0.5`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EntropyControl {
    /// Set the entropy fraction outright.
    Absolute {
        /// Logit of the fraction.
        raw: f64,
    },
    /// Scale the inherited setting.
    Relative {
        /// Natural log of the gain.
        raw: f64,
    },
    /// Set the temperature outright.
    Fixed {
        /// Natural log of the temperature.
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

    /// Fixed temperature.
    pub fn fixed(temperature: f64) -> Self {
        EntropyControl::Fixed {
            raw: temperature.max(1e-12).ln(),
        }
    }

    /// The unconstrained parameter.
    pub fn raw(&self) -> f64 {
        match self {
            EntropyControl::Absolute { raw }
            | EntropyControl::Relative { raw }
            | EntropyControl::Fixed { raw } => *raw,
        }
    }

    /// Overwrite the unconstrained parameter (non-finite values are ignored).
    pub fn set_raw(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        match self {
            EntropyControl::Absolute { raw }
            | EntropyControl::Relative { raw }
            | EntropyControl::Fixed { raw } => *raw = value,
        }
    }

    /// Whether this is an `Absolute` control.
    pub fn is_absolute(&self) -> bool {
        matches!(self, EntropyControl::Absolute { .. })
    }

    /// Set the fraction (`Absolute`), gain (`Relative`) or temperature
    /// (`Fixed`) directly.
    pub fn set_level(&mut self, level: f64) {
        match self {
            EntropyControl::Absolute { raw } => *raw = logit(level),
            EntropyControl::Relative { raw } | EntropyControl::Fixed { raw } => {
                *raw = level.max(1e-12).ln()
            }
        }
    }

    /// Effective tempering given the parent's.
    pub fn effective(&self, parent: Option<Tempering>) -> Tempering {
        match self {
            EntropyControl::Absolute { raw } => Tempering::Entropy(sigmoid(*raw)),
            EntropyControl::Fixed { raw } => Tempering::Temperature(raw.exp()),
            EntropyControl::Relative { raw } => match parent {
                Some(Tempering::Temperature(t)) => Tempering::Temperature(t * raw.exp()),
                Some(Tempering::Entropy(f)) => Tempering::Entropy((f * raw.exp()).clamp(0.0, 1.0)),
                None => Tempering::Entropy((Self::ORPHAN_BASE * raw.exp()).clamp(0.0, 1.0)),
            },
        }
    }

    /// Human-readable description.
    pub fn describe(&self) -> String {
        match self {
            EntropyControl::Absolute { raw } => format!("absolute {:.3}", sigmoid(*raw)),
            EntropyControl::Relative { raw } => format!("gain ×{:.3}", raw.exp()),
            EntropyControl::Fixed { raw } => format!("temperature {:.3}", raw.exp()),
        }
    }
}

/// Temper logits according to a [`Tempering`]. Returns `(temperature, entropy)`.
pub fn tempered_with(logits: &[f64], tempering: Tempering, out: &mut [f64]) -> (f64, f64) {
    match tempering {
        Tempering::Entropy(fraction) => tempered_distribution(logits, fraction, out),
        Tempering::Temperature(t) => {
            let t = t.max(1e-12);
            softmax(logits, t, out);
            (t, entropy(out))
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

    fn frac(t: Tempering) -> f64 {
        t.fraction().expect("entropy target")
    }

    #[test]
    fn entropy_control_composition() {
        let root = EntropyControl::absolute(0.4);
        assert!((frac(root.effective(None)) - 0.4).abs() < 1e-9);
        let child = EntropyControl::relative(2.0);
        assert!((frac(child.effective(Some(Tempering::Entropy(0.4)))) - 0.8).abs() < 1e-9);
        assert_eq!(frac(child.effective(Some(Tempering::Entropy(0.9)))), 1.0);
        assert!((frac(child.effective(None)) - 1.0).abs() < 1e-9);
        let mut c = EntropyControl::default();
        assert!((frac(c.effective(Some(Tempering::Entropy(0.3)))) - 0.3).abs() < 1e-9);
        c.set_raw(f64::NAN);
        assert_eq!(c.raw(), 0.0);
        c.set_level(0.5);
        assert!((frac(c.effective(Some(Tempering::Entropy(0.6)))) - 0.3).abs() < 1e-9);
    }

    #[test]
    fn fixed_temperature_composes_multiplicatively() {
        let root = EntropyControl::fixed(1.0);
        assert_eq!(root.effective(None), Tempering::Temperature(1.0));
        let child = EntropyControl::relative(0.5);
        let eff = child.effective(Some(Tempering::Temperature(2.0)));
        assert!((eff.temperature().unwrap() - 1.0).abs() < 1e-12);
        assert!(eff.fraction().is_none());
        let mut f = EntropyControl::fixed(3.0);
        f.set_level(0.25);
        assert!(
            (f.effective(Some(Tempering::Entropy(0.9)))
                .temperature()
                .unwrap()
                - 0.25)
                .abs()
                < 1e-12
        );
        assert!(f.describe().starts_with("temperature"));
        assert!(
            Tempering::Temperature(2.0)
                .heated(2.0)
                .temperature()
                .unwrap()
                > 3.99
        );
        let hot = Tempering::Entropy(0.4).heated(2.0);
        assert!((hot.fraction().unwrap() - 0.7).abs() < 1e-12);
    }

    #[test]
    fn tempered_with_fixed_temperature_uses_logits_as_they_are() {
        let logits = [2.0 * (1.0f64 + 40.0 / 20.0).ln(), 0.0];
        let mut out = [0.0; 2];
        let (t, h) = tempered_with(&logits, Tempering::Temperature(1.0), &mut out);
        assert_eq!(t, 1.0);
        assert!(
            (out[0] - 0.9).abs() < 1e-12,
            "Deneubourg's 0.9 for 40 vs 0 marks"
        );
        assert!(h > 0.0);
        let (_, h2) = tempered_with(&logits, Tempering::Entropy(0.5), &mut out);
        assert!((h2 - 0.5 * (2f64).ln()).abs() < 1e-3);
    }
}
