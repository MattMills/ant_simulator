//! Pheromone channels and their kinetics.
//!
//! Ants communicate through several chemically distinct signals laid on the
//! substrate or released into the air. Each channel here decays by
//! first-order kinetics from a literature half-life, spreads a little to
//! neighbouring cells, saturates on the substrate, and is perceived through
//! a saturating response.
//!
//! The perception function is the one behind Deneubourg's choice function.
//! Ants offered branches carrying concentrations `C₁, C₂` choose branch 1
//! with probability `(k + C₁)ⁿ / ((k + C₁)ⁿ + (k + C₂)ⁿ)` (Deneubourg, Aron,
//! Goss & Pasteels 1990, *J. Insect Behav.* 3:159; fitted with `k ≈ 20`
//! marks and `n ≈ 2` for the Argentine ant). Feeding the behavioral surface
//! the feature `ln(1 + C/k)` with weight `n` reproduces that function exactly
//! through the softmax, and extends it to any number of directions.

/// A chemical channel on the substrate or in the air.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Pheromone {
    /// Recruitment trail laid by successful foragers (attractive).
    Trail = 0,
    /// Outbound trail laid by ants leaving the nest, followed home by
    /// loaded ants in species (or models) with bidirectional trails.
    Home = 1,
    /// Home-range marking: cuticular hydrocarbons deposited around the nest
    /// by all workers, which raise exploration and confidence within the
    /// colony's territory (Devigne & Detrain 2002, *Insectes Sociaux*
    /// 49:357).
    Territory = 2,
    /// Repellent "no entry" marking laid on unrewarding branches by
    /// Pharaoh's ants (Robinson, Jackson, Holcombe & Ratnieks 2005,
    /// *Nature* 438:442).
    NoEntry = 3,
    /// Volatile alarm pheromone released by injured or killed workers,
    /// which foragers far from the nest avoid (Hölldobler & Wilson 1990,
    /// *The Ants*, ch. 7).
    Alarm = 4,
    /// The smell of food: volatiles given off by sugar solution (weakly)
    /// and by dead insects (strongly), spreading through the air and gone
    /// within minutes. Searching ants climb its gradient and so find food
    /// from a distance (Buehlmann, Graham, Hansson & Knaden 2014, *Curr.
    /// Biol.* 24:960: desert ants locate food by its odour).
    Odour = 5,
}

impl Pheromone {
    /// Every channel, in index order.
    pub const ALL: [Pheromone; 6] = [
        Pheromone::Trail,
        Pheromone::Home,
        Pheromone::Territory,
        Pheromone::NoEntry,
        Pheromone::Alarm,
        Pheromone::Odour,
    ];

    /// Number of channels.
    pub const COUNT: usize = 6;

    /// Index into a [`PheromoneSet`] or a cell's pheromone array.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Short name.
    pub fn name(self) -> &'static str {
        match self {
            Pheromone::Trail => "trail",
            Pheromone::Home => "home",
            Pheromone::Territory => "territory",
            Pheromone::NoEntry => "no-entry",
            Pheromone::Alarm => "alarm",
            Pheromone::Odour => "odour",
        }
    }
}

/// Kinetic and perceptual parameters of one channel.
#[derive(Clone, Debug, PartialEq)]
pub struct PheromoneParams {
    /// Half-life of the deposit in seconds (first-order decay). Zero makes
    /// the channel inert (it vanishes every tick); infinity makes it
    /// permanent.
    pub half_life_s: f64,
    /// Fraction of a cell's content that spreads to its four orthogonal
    /// neighbours per second (0 for none).
    pub diffusion_per_s: f64,
    /// Saturation of the substrate: maximum concentration per cell.
    pub cap: f64,
    /// Deneubourg sensitivity constant `k`: the concentration at which the
    /// response starts to saturate.
    pub k: f64,
}

impl PheromoneParams {
    /// A channel that carries nothing.
    pub fn inert() -> Self {
        PheromoneParams {
            half_life_s: 0.0,
            diffusion_per_s: 0.0,
            cap: 0.0,
            k: 1.0,
        }
    }

    /// Whether the channel carries anything at all.
    pub fn is_active(&self) -> bool {
        self.half_life_s > 0.0 && self.cap > 0.0
    }

    /// Fraction retained after one tick of `tick_s` seconds.
    pub fn retention_per_tick(&self, tick_s: f64) -> f64 {
        if self.half_life_s <= 0.0 {
            0.0
        } else if self.half_life_s.is_infinite() {
            1.0
        } else {
            0.5f64.powf(tick_s / self.half_life_s)
        }
    }

    /// Fraction spread to neighbours in one tick (capped at 1).
    pub fn diffusion_per_tick(&self, tick_s: f64) -> f64 {
        (self.diffusion_per_s * tick_s).clamp(0.0, 1.0)
    }
}

/// Parameters for all channels, indexed by [`Pheromone::index`].
pub type PheromoneSet = [PheromoneParams; Pheromone::COUNT];

/// Perceived intensity of a concentration: `ln(1 + C/k)`.
///
/// A weight `n` on this feature yields Deneubourg's `(k + C)ⁿ` response
/// through the softmax (the `kⁿ` factor cancels).
pub fn perceived(concentration: f64, k: f64) -> f64 {
    if concentration <= 0.0 {
        return 0.0;
    }
    (1.0 + concentration / k.max(1e-9)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retention_follows_half_life() {
        let p = PheromoneParams {
            half_life_s: 100.0,
            diffusion_per_s: 0.0,
            cap: 10.0,
            k: 1.0,
        };
        assert!((p.retention_per_tick(100.0) - 0.5).abs() < 1e-12);
        assert!((p.retention_per_tick(1.0).powi(100) - 0.5).abs() < 1e-9);
        assert_eq!(PheromoneParams::inert().retention_per_tick(1.0), 0.0);
        assert!(!PheromoneParams::inert().is_active());
        let forever = PheromoneParams {
            half_life_s: f64::INFINITY,
            ..p
        };
        assert_eq!(forever.retention_per_tick(1e9), 1.0);
    }

    #[test]
    fn perception_reproduces_deneubourg_choice() {
        // Two branches with 0 and 40 marks, k = 20, n = 2:
        // P(40) = (60)² / (20² + 60²) = 3600 / 4000 = 0.9.
        let n = 2.0;
        let a = n * perceived(0.0, 20.0);
        let b = n * perceived(40.0, 20.0);
        let p = b.exp() / (a.exp() + b.exp());
        assert!((p - 0.9).abs() < 1e-12, "{p}");
        assert_eq!(perceived(-5.0, 20.0), 0.0);
        assert_eq!(Pheromone::ALL.len(), Pheromone::COUNT);
        assert_eq!(Pheromone::NoEntry.index(), 3);
        assert_eq!(Pheromone::Alarm.name(), "alarm");
    }
}
