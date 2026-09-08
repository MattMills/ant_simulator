//! The surface flat in front of an ant, and the sucker that crawls on it.
//!
//! An ant's candidate headings form a ring around its current heading:
//! [`RING`] of them, evenly spaced, index 0 straight ahead, index
//! `RING / 2` straight back, index 1 one step clockwise (to the right on
//! screen). The scores the behavioral surface assigns to them are a
//! *landscape* on that ring: the deterministic information of the path,
//! laid out egocentrically. Stacked along a trajectory, the rings form the
//! path's surface.
//!
//! Entropy is then spent on that landscape through separable geometric
//! channels:
//!
//! * **tempering** rescales the landscape (to a fixed temperature, or so
//!   that the implied distribution has exactly the requested entropy);
//! * **smoothing** blurs the landscape along the ring, spreading preference
//!   to neighbouring headings (angular disorder, path-coherent);
//! * **roughening** adds a random low-mode field, carving random basins
//!   (landscape disorder, path-incoherent).
//!
//! Finally a heading is *selected*. The classic way is a global draw from
//! the tempered distribution. The geometric way is a [`Sucker`]: a walker
//! that starts straight ahead and crawls the ring by local Metropolis moves
//! for a bounded number of steps, settling in whatever basin it can reach.
//! Where it ends depends on where it started and on the basin structure, so
//! behaviour becomes a property of the path's geometry rather than of a
//! global lottery. Its realised entropy can fall either side of the target:
//! a walker parked on the peak ahead is narrower than the target, a walker
//! still travelling towards a peak off to the side is broader.

use crate::entropy::{entropy, softmax, tempered_distribution, tempered_with, Tempering};
use crate::rng::Rng;

/// Number of candidate headings: 22.5° apart.
pub const RING: usize = 16;

/// Angular step between neighbouring ring positions, radians.
pub const RING_STEP: f64 = std::f64::consts::TAU / RING as f64;

/// Ring positions in left-to-right display order: from the sharpest left
/// turn through straight ahead to the sharpest right turn, then reverse.
pub const DISPLAY_ORDER: [usize; RING] = [9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8];

/// Turn angle of a ring position relative to the heading, in degrees:
/// positive is clockwise (right on screen), `180` is reverse.
pub fn turn_degrees(ring: usize) -> f64 {
    let k = ring % RING;
    let signed = if k <= RING / 2 {
        k as f64
    } else {
        k as f64 - RING as f64
    };
    signed * 360.0 / RING as f64
}

/// Human-readable name of a ring position.
pub fn turn_label(ring: usize) -> String {
    let deg = turn_degrees(ring);
    if deg == 0.0 {
        "straight".to_string()
    } else if deg == 180.0 {
        "reverse".to_string()
    } else if deg > 0.0 {
        format!("right {deg:.1}°")
    } else {
        format!("left {:.1}°", -deg)
    }
}

/// Heading angle of a ring position given the current heading.
pub fn ring_heading(ring: usize, heading: f64) -> f64 {
    crate::geometry::wrap_angle(heading + (ring % RING) as f64 * RING_STEP)
}

/// Ring position nearest to a heading angle, relative to the current one.
pub fn ring_index_of(angle: f64, heading: f64) -> usize {
    let rel = crate::geometry::wrap_angle(angle - heading);
    let k = (rel / RING_STEP).round() as isize;
    k.rem_euclid(RING as isize) as usize
}

/// Shortest distance around the ring between two positions.
pub fn ring_distance(a: usize, b: usize) -> usize {
    let d = (a as isize - b as isize).unsigned_abs() % RING;
    d.min(RING - d)
}

/// Turn magnitude of a ring position in ring steps (0 straight, `RING/2`
/// reverse).
pub fn turn_magnitude(ring: usize) -> usize {
    ring_distance(ring, 0)
}

/// A scored ring: the surface flat in front of the ant.
#[derive(Clone, Debug, PartialEq)]
pub struct Landscape {
    /// Score of each ring position (`-inf` where masked).
    pub values: [f64; RING],
    /// Whether the heading can be taken.
    pub valid: [bool; RING],
}

impl Landscape {
    /// Landscape with the given values and validity mask; masked positions
    /// are forced to `-inf`.
    pub fn new(values: [f64; RING], valid: [bool; RING]) -> Self {
        let mut values = values;
        for (v, ok) in values.iter_mut().zip(&valid) {
            if !*ok {
                *v = f64::NEG_INFINITY;
            }
        }
        Landscape { values, valid }
    }

    /// Landscape with the given values, all positions valid.
    pub fn open(values: [f64; RING]) -> Self {
        Landscape {
            values,
            valid: [true; RING],
        }
    }

    /// Number of takeable headings.
    pub fn valid_count(&self) -> usize {
        self.valid.iter().filter(|v| **v).count()
    }

    /// Difference between the highest and lowest valid score (0 if fewer
    /// than two valid positions).
    pub fn range(&self) -> f64 {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for (v, ok) in self.values.iter().zip(&self.valid) {
            if *ok {
                lo = lo.min(*v);
                hi = hi.max(*v);
            }
        }
        if hi.is_finite() && lo.is_finite() {
            hi - lo
        } else {
            0.0
        }
    }

    /// The valid position nearest to `start` around the ring (ties go
    /// clockwise), or `None` if nothing is valid.
    pub fn nearest_valid(&self, start: usize) -> Option<usize> {
        for d in 0..=RING / 2 {
            let cw = (start + d) % RING;
            if self.valid[cw] {
                return Some(cw);
            }
            let ccw = (start + RING - d) % RING;
            if self.valid[ccw] {
                return Some(ccw);
            }
        }
        None
    }

    /// Blur the landscape along the ring with a circular Gaussian kernel of
    /// the given scale (in ring steps). Masked positions neither receive nor
    /// contribute. A scale near zero returns the landscape unchanged.
    pub fn smoothed(&self, scale: f64) -> Landscape {
        if scale < 1e-6 {
            return self.clone();
        }
        let mut out = self.clone();
        let kernel: Vec<f64> = (0..=RING / 2)
            .map(|d| (-(d as f64).powi(2) / (2.0 * scale * scale)).exp())
            .collect();
        for j in 0..RING {
            if !self.valid[j] {
                continue;
            }
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..RING {
                if self.valid[i] {
                    let k = kernel[ring_distance(i, j)];
                    num += k * self.values[i];
                    den += k;
                }
            }
            out.values[j] = num / den;
        }
        out
    }

    /// Add a random low-mode field of the given amplitude (see
    /// [`rough_field`]). An amplitude of zero returns the landscape unchanged
    /// without consuming randomness.
    pub fn roughened(&self, amplitude: f64, modes: usize, rng: &mut Rng) -> Landscape {
        if amplitude <= 0.0 {
            return self.clone();
        }
        let field = rough_field(modes, rng);
        let mut out = self.clone();
        for ((v, ok), f) in out.values.iter_mut().zip(&self.valid).zip(&field) {
            if *ok {
                *v += amplitude * f;
            }
        }
        out
    }

    /// Entropy of the softmax of this landscape at a temperature.
    pub fn entropy_at(&self, temperature: f64) -> f64 {
        let mut probs = [0.0; RING];
        softmax(&self.values, temperature, &mut probs);
        entropy(&probs)
    }

    /// Fraction of the maximum entropy the landscape's distribution carries
    /// at a temperature (0 if fewer than two positions are valid).
    pub fn entropy_fraction_at(&self, temperature: f64) -> f64 {
        let n = self.valid_count();
        if n < 2 {
            0.0
        } else {
            (self.entropy_at(temperature) / (n as f64).ln()).clamp(0.0, 1.0)
        }
    }

    /// Temper the landscape so its distribution carries `fraction` of the
    /// maximum entropy over the valid positions.
    pub fn temper(&self, fraction: f64) -> Tempered {
        self.temper_with(Tempering::Entropy(fraction))
    }

    /// Temper the landscape by an entropy target or a fixed temperature.
    pub fn temper_with(&self, tempering: Tempering) -> Tempered {
        let mut probs = [0.0; RING];
        let (temperature, h) = match tempering {
            Tempering::Entropy(fraction) => {
                tempered_distribution(&self.values, fraction, &mut probs)
            }
            Tempering::Temperature(_) => tempered_with(&self.values, tempering, &mut probs),
        };
        let max = self
            .values
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        let mut scaled = [f64::NEG_INFINITY; RING];
        for ((s, v), ok) in scaled.iter_mut().zip(&self.values).zip(&self.valid) {
            if *ok && v.is_finite() {
                *s = (v - max) / temperature;
            }
        }
        Tempered {
            probs,
            scaled,
            temperature,
            entropy: h,
        }
    }
}

/// A random field on the ring made of a few low Fourier modes with random
/// amplitudes and phases, normalised to roughly unit standard deviation.
pub fn rough_field(modes: usize, rng: &mut Rng) -> [f64; RING] {
    let modes = modes.clamp(1, RING / 2);
    let mut field = [0.0; RING];
    let mut variance = 0.0;
    for m in 1..=modes {
        let amp = rng.normal() / m as f64;
        let phase = rng.range(0.0, std::f64::consts::TAU);
        variance += 0.5 / (m * m) as f64;
        for (j, f) in field.iter_mut().enumerate() {
            let theta = std::f64::consts::TAU * j as f64 / RING as f64;
            *f += amp * (m as f64 * theta + phase).cos();
        }
    }
    let norm = variance.sqrt();
    for f in field.iter_mut() {
        *f /= norm;
    }
    field
}

/// A landscape after tempering.
#[derive(Clone, Debug, PartialEq)]
pub struct Tempered {
    /// The tempered distribution over ring positions.
    pub probs: [f64; RING],
    /// `(value - max) / temperature`, finite on valid positions: the
    /// log-landscape the sucker feels, immune to underflow.
    pub scaled: [f64; RING],
    /// The temperature that met the entropy target.
    pub temperature: f64,
    /// Entropy of `probs`, in nats.
    pub entropy: f64,
}

/// A local walker that selects a ring position by crawling.
///
/// Starting from a position, it proposes a move to a neighbour (clockwise or
/// counter-clockwise with equal probability) `reach` times, accepting uphill
/// moves always and downhill moves with the Metropolis probability
/// `exp(Δ)` on the scaled landscape. With infinite reach its position would
/// be distributed like the tempered distribution; with bounded reach it
/// settles wherever it can get to, which makes the choice depend on the
/// starting point and on the basins in between.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sucker {
    /// Number of proposal steps.
    pub reach: usize,
}

impl Sucker {
    fn accept(scaled: &[f64; RING], from: usize, to: usize) -> f64 {
        if !scaled[to].is_finite() {
            return 0.0;
        }
        if !scaled[from].is_finite() || scaled[to] >= scaled[from] {
            1.0
        } else {
            (scaled[to] - scaled[from]).exp()
        }
    }

    /// Crawl from `start`; returns the final position and every position
    /// visited (starting with `start`).
    pub fn walk(&self, scaled: &[f64; RING], start: usize, rng: &mut Rng) -> (usize, Vec<u8>) {
        let mut j = start;
        let mut trail = Vec::with_capacity(self.reach + 1);
        trail.push(j as u8);
        for _ in 0..self.reach {
            let next = if rng.chance(0.5) {
                (j + 1) % RING
            } else {
                (j + RING - 1) % RING
            };
            let p = Self::accept(scaled, j, next);
            if p >= 1.0 || (p > 0.0 && rng.chance(p)) {
                j = next;
            }
            trail.push(j as u8);
        }
        (j, trail)
    }

    /// Exact distribution of the walker's final position after `reach`
    /// steps from `start`.
    pub fn distribution(&self, scaled: &[f64; RING], start: usize) -> [f64; RING] {
        let mut p = [0.0; RING];
        p[start] = 1.0;
        for _ in 0..self.reach {
            let mut next = [0.0; RING];
            for j in 0..RING {
                if p[j] == 0.0 {
                    continue;
                }
                let cw = (j + 1) % RING;
                let ccw = (j + RING - 1) % RING;
                let a_cw = 0.5 * Self::accept(scaled, j, cw);
                let a_ccw = 0.5 * Self::accept(scaled, j, ccw);
                next[cw] += p[j] * a_cw;
                next[ccw] += p[j] * a_ccw;
                next[j] += p[j] * (1.0 - a_cw - a_ccw);
            }
            p = next;
        }
        p
    }
}

/// Where one decision's entropy came from, in nats.
///
/// The within-decision terms add up: `tempering + smoothing + roughening +
/// selection == selected`. `field` sits on top of them: disorder the random
/// field injects *between* decisions, invisible inside any single one.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Contributions {
    /// Entropy the deterministic information already had at the solved
    /// temperature.
    pub tempering: f64,
    /// Extra entropy from blurring along the ring.
    pub smoothing: f64,
    /// Change from the random field within the decision. Usually negative:
    /// a random spike makes the single decision sharper, displacing
    /// tempering, while the randomness it carries shows up in `field`.
    pub roughening: f64,
    /// Entropy of the choice marginalised over the random field, minus the
    /// target: how much the field randomises behaviour across decisions.
    /// Zero without roughening.
    pub field: f64,
    /// Change caused by the selection mechanism: zero for a global draw;
    /// for a sucker of bounded reach, negative when it sits on the peak it
    /// started on and positive while it is still in transit towards an
    /// off-axis peak.
    pub selection: f64,
    /// Entropy of the distribution the heading was actually drawn from.
    pub selected: f64,
}

/// Running sums of decision entropies at each stage of the pipeline.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EntropyLedger {
    /// Decisions accounted for.
    pub decisions: u64,
    /// Sum of entropies of the undeformed landscape at the solved temperature.
    pub base: f64,
    /// Sum after smoothing.
    pub smoothed: f64,
    /// Sum after roughening and tempering (the entropy target).
    pub deformed: f64,
    /// Sum of entropies of the tempered distribution marginalised over the
    /// random field (equals `deformed` without roughening).
    pub mixture: f64,
    /// Sum of entropies of the actual selection distribution.
    pub selected: f64,
}

impl EntropyLedger {
    /// Record one decision.
    pub fn record(&mut self, base: f64, smoothed: f64, deformed: f64, mixture: f64, selected: f64) {
        self.decisions += 1;
        self.base += base;
        self.smoothed += smoothed;
        self.deformed += deformed;
        self.mixture += mixture;
        self.selected += selected;
    }

    /// Add another ledger's sums.
    pub fn merge(&mut self, other: &EntropyLedger) {
        self.decisions += other.decisions;
        self.base += other.base;
        self.smoothed += other.smoothed;
        self.deformed += other.deformed;
        self.mixture += other.mixture;
        self.selected += other.selected;
    }

    /// Mean contributions per decision.
    pub fn contributions(&self) -> Contributions {
        if self.decisions == 0 {
            return Contributions::default();
        }
        let n = self.decisions as f64;
        let base = self.base / n;
        let smoothed = self.smoothed / n;
        let deformed = self.deformed / n;
        let mixture = self.mixture / n;
        let selected = self.selected / n;
        Contributions {
            tempering: base,
            smoothing: smoothed - base,
            roughening: deformed - smoothed,
            field: mixture - deformed,
            selection: selected - deformed,
            selected,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peaked(at: usize, height: f64) -> [f64; RING] {
        let mut v = [0.0; RING];
        v[at] = height;
        v
    }

    #[test]
    fn ring_mapping_round_trips() {
        for k in 0..RING {
            let h = 0.3;
            assert_eq!(ring_index_of(ring_heading(k, h), h), k);
        }
        assert_eq!(ring_index_of(0.0, 0.0), 0);
        assert_eq!(ring_index_of(std::f64::consts::PI, 0.0), RING / 2);
        assert_eq!(ring_index_of(RING_STEP, 0.0), 1);
        assert_eq!(ring_distance(1, RING - 1), 2);
        assert_eq!(turn_magnitude(RING - 3), 3);
        assert!((turn_degrees(1) - 22.5).abs() < 1e-12);
        assert!((turn_degrees(RING - 1) + 22.5).abs() < 1e-12);
        assert_eq!(turn_label(0), "straight");
        assert_eq!(turn_label(RING / 2), "reverse");
        assert!(turn_label(2).starts_with("right 45"));
        assert!(turn_label(RING - 4).starts_with("left 90"));
        let mut seen: Vec<usize> = DISPLAY_ORDER.to_vec();
        seen.sort_unstable();
        assert_eq!(seen, (0..RING).collect::<Vec<_>>());
    }

    #[test]
    fn new_masks_and_finds_neighbours() {
        let mut valid = [true; RING];
        valid[RING / 2] = false;
        let land = Landscape::new(peaked(0, 3.0), valid);
        assert!(!land.valid[RING / 2]);
        assert_eq!(land.values[RING / 2], f64::NEG_INFINITY);
        assert_eq!(land.valid_count(), RING - 1);
        assert_eq!(land.range(), 3.0);
        assert_eq!(land.nearest_valid(RING / 2), Some(RING / 2 + 1));
        let none = Landscape::new([0.0; RING], [false; RING]);
        assert_eq!(none.nearest_valid(0), None);
        assert_eq!(none.range(), 0.0);
    }

    #[test]
    fn smoothing_spreads_and_raises_entropy() {
        let land = Landscape::open(peaked(0, 5.0));
        let blurred = land.smoothed(1.0);
        assert!(blurred.values[0] < 5.0);
        assert!(blurred.values[1] > 0.0 && blurred.values[RING - 1] > 0.0);
        assert!((blurred.values[1] - blurred.values[RING - 1]).abs() < 1e-12);
        assert!(blurred.entropy_at(1.0) > land.entropy_at(1.0));
        assert_eq!(land.smoothed(0.0), land);
        let mut masked = land.clone();
        masked.valid[1] = false;
        masked.values[1] = f64::NEG_INFINITY;
        let b = masked.smoothed(1.0);
        assert!(!b.valid[1]);
        assert_eq!(b.values[1], f64::NEG_INFINITY);
        assert!(b.values[0].is_finite());
    }

    #[test]
    fn roughening_is_zero_mean_ish_and_scaled() {
        let mut rng = Rng::seed_from_u64(4);
        let land = Landscape::open([1.0; RING]);
        assert_eq!(land.roughened(0.0, 3, &mut rng), land);
        let mut total_var = 0.0;
        let trials = 2000;
        for _ in 0..trials {
            let r = land.roughened(2.0, 3, &mut rng);
            let mean = r.values.iter().sum::<f64>() / RING as f64;
            total_var += r.values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / RING as f64;
        }
        let std = (total_var / trials as f64).sqrt();
        assert!((std - 2.0).abs() < 0.4, "std {std}");
    }

    #[test]
    fn tempering_matches_entropy_target() {
        let mut v = [0.0; RING];
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((i as f64) * 0.7).sin() * 2.0;
        }
        let land = Landscape::open(v);
        let t = land.temper(0.4);
        assert!((t.entropy - 0.4 * (RING as f64).ln()).abs() < 1e-3);
        assert!((t.probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(t.scaled.iter().all(|s| s.is_finite() && *s <= 0.0));
        let cold = land.temper(0.0);
        let argmax = v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!((cold.probs[argmax] - 1.0).abs() < 1e-3);
        let fixed = land.temper_with(Tempering::Temperature(1.0));
        assert_eq!(fixed.temperature, 1.0);
        assert!((fixed.probs[1] / fixed.probs[0] - (v[1] - v[0]).exp()).abs() < 1e-9);
        assert!((land.entropy_fraction_at(1.0) - fixed.entropy / (RING as f64).ln()).abs() < 1e-12);
    }

    #[test]
    fn sucker_with_no_reach_stays_put() {
        let mut rng = Rng::seed_from_u64(1);
        let land = Landscape::open(peaked(RING / 2, 9.0));
        let t = land.temper(0.5);
        let (j, trail) = Sucker { reach: 0 }.walk(&t.scaled, 0, &mut rng);
        assert_eq!(j, 0);
        assert_eq!(trail, vec![0]);
        assert_eq!(Sucker { reach: 0 }.distribution(&t.scaled, 0)[0], 1.0);
    }

    #[test]
    fn sucker_walk_matches_exact_distribution() {
        let mut rng = Rng::seed_from_u64(2);
        let mut v = [0.0; RING];
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((i as f64) * 1.3).cos() * 3.0;
        }
        let t = Landscape::open(v).temper(0.6);
        let sucker = Sucker { reach: 5 };
        let exact = sucker.distribution(&t.scaled, 0);
        assert!((exact.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        let mut counts = [0usize; RING];
        let trials = 40_000;
        for _ in 0..trials {
            let (j, trail) = sucker.walk(&t.scaled, 0, &mut rng);
            assert_eq!(trail.len(), 6);
            counts[j] += 1;
        }
        for j in 0..RING {
            let freq = counts[j] as f64 / trials as f64;
            assert!(
                (freq - exact[j]).abs() < 0.012,
                "position {j}: {freq} vs {}",
                exact[j]
            );
        }
    }

    #[test]
    fn sucker_converges_to_the_target_with_long_reach() {
        let mut v = [0.0; RING];
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((i as f64) * 0.5).sin();
        }
        let t = Landscape::open(v).temper(0.9);
        let far = Sucker { reach: 1500 }.distribution(&t.scaled, 3);
        let tv: f64 = far
            .iter()
            .zip(&t.probs)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / 2.0;
        assert!(tv < 0.01, "total variation {tv}");
    }

    #[test]
    fn cold_sucker_climbs_to_the_nearest_peak_not_the_highest() {
        // A small peak two steps to the right, a huge one straight back.
        let mut v = [-3.0; RING];
        v[0] = 0.0;
        v[1] = 0.5;
        v[2] = 1.0;
        v[RING / 2] = 10.0;
        let t = Landscape::open(v).temper(0.0);
        let d = Sucker { reach: 24 }.distribution(&t.scaled, 0);
        assert!(d[2] > 0.99, "sucker settles on the nearby peak: {d:?}");
        assert!(d[RING / 2] < 1e-6);
        assert!(t.probs[RING / 2] > 0.99);
    }

    #[test]
    fn sucker_never_enters_masked_positions() {
        let mut rng = Rng::seed_from_u64(3);
        let mut valid = [true; RING];
        valid[1] = false;
        valid[RING - 1] = false;
        let t = Landscape::new([0.0; RING], valid).temper(1.0);
        for _ in 0..200 {
            let (j, trail) = Sucker { reach: 6 }.walk(&t.scaled, 0, &mut rng);
            assert_eq!(j, 0);
            assert!(trail.iter().all(|p| *p == 0));
        }
        assert_eq!(Sucker { reach: 6 }.distribution(&t.scaled, 0)[0], 1.0);
    }

    #[test]
    fn ledger_contributions_add_up() {
        let mut ledger = EntropyLedger::default();
        ledger.record(0.5, 0.8, 1.0, 1.1, 0.7);
        ledger.record(0.3, 0.4, 0.9, 1.0, 0.9);
        let c = ledger.contributions();
        let total = c.tempering + c.smoothing + c.roughening + c.selection;
        assert!((total - c.selected).abs() < 1e-12);
        assert!((c.selected - 0.8).abs() < 1e-12);
        assert!((c.field - 0.1).abs() < 1e-12);
        assert!(c.selection < 0.0);
        let mut other = EntropyLedger::default();
        other.merge(&ledger);
        assert_eq!(other, ledger);
        assert_eq!(
            EntropyLedger::default().contributions(),
            Contributions::default()
        );
    }
}
