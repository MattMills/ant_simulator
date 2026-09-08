//! Species profiles: literature-based parameters for the biological model.
//!
//! Every quantity is in physical units (seconds, centimetres, pheromone
//! "marks") and is converted to grid and tick units by the simulation. The
//! defaults are order-of-magnitude values from the cited studies; where the
//! literature gives ranges, a representative value is used and noted.
//!
//! Profiles:
//!
//! * [`Species::lasius_niger`] (black garden ant): mass-recruiting,
//!   inbound trail laying modulated by food quality, long-lived trail
//!   pheromone. The default.
//! * [`Species::argentine`] (*Linepithema humile*): lays trail in both
//!   directions, shorter-lived pheromone, the species of the double-bridge
//!   experiments.
//! * [`Species::pharaoh`] (*Monomorium pharaonis*): adds the repellent
//!   "no entry" marking.
//! * [`Species::cataglyphis`] (desert ant): no trail pheromone at all;
//!   solitary foraging by path integration and systematic search.

use crate::ant::{
    BASE_FEATURES, FEATURES, F_ALARM, F_CROWD, F_FOOD, F_HEADING, F_HOME, F_HOME_VECTOR, F_NEST,
    F_NO_ENTRY, F_RECENT, F_SITE, F_TERRITORY, F_TRAIL,
};
use crate::entropy::EntropyControl;
use crate::pheromone::{Pheromone, PheromoneParams, PheromoneSet};
use crate::surface::{BehavioralSurface, Deformation};

/// Parameters of one species.
#[derive(Clone, Debug, PartialEq)]
pub struct Species {
    /// Name.
    pub name: String,

    // ---- locomotion ----
    /// Walking speed of an unloaded forager, cm/s.
    pub speed_cm_s: f64,
    /// Speed multiplier when carrying a full crop.
    pub loaded_speed_factor: f64,
    /// Speed multiplier on a strong trail (ants on established trails walk
    /// straighter and faster).
    pub trail_speed_factor: f64,

    // ---- recruitment trail ----
    /// Kinetics of the recruitment trail.
    pub trail: PheromoneParams,
    /// Deneubourg's choice exponent `n`.
    pub choice_exponent: f64,
    /// Marks deposited per cell crossed while laying.
    pub trail_deposit: f64,
    /// Whether ants also lay trail on the way out to a known source.
    pub outbound_laying: bool,
    /// Food quality (0..1) at which the probability of laying trail on the
    /// way home is half its maximum.
    pub lay_quality_half: f64,
    /// Hill exponent of the quality modulation.
    pub lay_exponent: f64,
    /// Maximum probability that a returning forager lays trail.
    pub lay_max_probability: f64,
    /// Food quality at which a forager keeps returning to a remembered
    /// source with probability one half (site fidelity falls away at poor
    /// sources: Mailleux, Deneubourg & Detrain 2000, *Anim. Behav.* 59:1061;
    /// Detrain & Deneubourg 2008, *Adv. Insect Physiol.* 35:123).
    pub fidelity_quality_half: f64,
    /// Fraction of the crop filled at a source of zero quality; the load
    /// rises linearly to a full crop at quality one (crop load increases
    /// with concentration: Josens, Farina & Roces 1998).
    pub min_load_fraction: f64,

    // ---- other channels ----
    /// Outbound "home" trail kinetics (inert unless `uses_home_pheromone`).
    pub home: PheromoneParams,
    /// Whether the species lays a separate outbound trail followed home.
    pub uses_home_pheromone: bool,
    /// Home-range marking kinetics.
    pub territory: PheromoneParams,
    /// Home-range marks deposited per cell by every worker.
    pub territory_deposit: f64,
    /// Repellent "no entry" kinetics.
    pub no_entry: PheromoneParams,
    /// Whether unsuccessful returning foragers lay "no entry" marking.
    pub uses_no_entry: bool,
    /// No-entry marks deposited per cell.
    pub no_entry_deposit: f64,
    /// Alarm pheromone kinetics.
    pub alarm: PheromoneParams,
    /// Alarm released at the death of a worker.
    pub alarm_release: f64,

    // ---- sensing ----
    /// Cells ahead an antenna sweep integrates (1 or 2).
    pub sense_range: usize,

    // ---- navigation ----
    /// Standard deviation of the heading error per step, degrees.
    pub pi_heading_noise_deg: f64,
    /// Standard deviation of the relative odometric error per step.
    pub pi_distance_noise: f64,
    /// Distance (cells) from a remembered location at which the ant
    /// considers itself arrived and starts searching.
    pub arrival_radius: f64,
    /// How long an ant searches around a fictive location, seconds.
    pub search_time_s: f64,
    /// Time after which an unsuccessful outbound trip is abandoned, seconds.
    pub give_up_time_s: f64,
    /// Temperature multiplier applied while searching.
    pub search_temperature_factor: f64,

    // ---- feeding ----
    /// Crop capacity in food units (one unit is one full crop load).
    pub crop_capacity: f64,
    /// Time to fill the crop, seconds: `base + slope × quality`.
    pub feeding_time_s: (f64, f64),
    /// Time to unload by trophallaxis in a hungry nest, seconds.
    pub unloading_time_s: f64,
    /// Extra unloading time per unit of colony satiation, as a factor.
    pub unloading_satiation_factor: f64,

    // ---- survival ----
    /// Hazard rate outside the nest, per second.
    pub forager_hazard_per_s: f64,
    /// Time an ant survives away from food, seconds.
    pub starvation_s: f64,

    // ---- task allocation ----
    /// Median foraging response threshold of adult workers.
    pub threshold_median: f64,
    /// Log-normal spread (σ of ln θ) of thresholds across workers.
    pub threshold_spread: f64,
    /// Steepness of the response-threshold function.
    pub threshold_exponent: f64,
    /// Mean interval between task decisions inside the nest, seconds.
    pub decision_interval_s: f64,
    /// Age scale over which the foraging threshold falls to its adult
    /// value (temporal polyethism), seconds.
    pub maturation_s: f64,
    /// Factor by which a newborn's foraging threshold exceeds the adult one.
    pub youth_threshold_factor: f64,
    /// Median nursing threshold.
    pub nursing_threshold_median: f64,
    /// Duration of one nursing bout, seconds.
    pub nursing_bout_s: f64,
    /// Brood items one nurse can tend: the nursing stimulus is the brood
    /// load per nurse already at work, so recruitment of nurses damps it.
    pub brood_per_nurse: f64,
    /// Foraging stimulus bonus for a worker remembering a source of unit
    /// quality (site fidelity).
    pub reforage_bonus: f64,
    /// E-folding time over which a task's threshold falls while the worker
    /// performs it (Theraulaz, Bonabeau & Deneubourg 1998, *Proc. R. Soc.
    /// B* 265:327), seconds.
    pub threshold_learning_s: f64,
    /// E-folding time over which a task's threshold rises while the worker
    /// does not perform it, seconds.
    pub threshold_forgetting_s: f64,
    /// Bounds on a reinforced threshold as multiples of the species median.
    pub threshold_bounds: (f64, f64),

    // ---- colony ----
    /// Excitation added to the nest per unit of quality of a returned load.
    pub excitation_per_return: f64,
    /// Half-life of that excitation, seconds.
    pub excitation_half_life_s: f64,
    /// Foraging stimulus per unit of colony hunger.
    pub hunger_gain: f64,
    /// Food units consumed per worker inside the nest per second.
    pub consumption_per_ant_per_s: f64,
    /// Interval between eggs laid by the queen when the colony is fed,
    /// seconds.
    pub egg_interval_s: f64,
    /// Development time from egg to worker, seconds.
    pub development_s: f64,
    /// Food a brood item must receive to complete development.
    pub brood_food: f64,
    /// Food handed to brood per second of nursing.
    pub nursing_rate: f64,
}

fn lognormal_sigma_for_cv(cv: f64) -> f64 {
    (1.0 + cv * cv).ln().sqrt()
}

impl Species {
    /// *Lasius niger*, the black garden ant.
    ///
    /// Speed ≈ 1.5 cm/s at room temperature (Hurlbert et al. 2008,
    /// *Insectes Sociaux* 55:151 report 1–3 cm/s). Trail pheromone half-life
    /// ≈ 47 min and trail laying rising with sucrose concentration
    /// (Beckers, Deneubourg & Goss 1993, *J. Insect Behav.* 6:751). Choice
    /// parameters `k = 20`, `n = 2` (Deneubourg et al. 1990). Crop filling
    /// takes one to three minutes depending on concentration (Josens,
    /// Farina & Roces 1998, *J. Insect Physiol.* 44:579, for *Camponotus*).
    /// Colony hunger drives foraging (Mailleux, Deneubourg & Detrain 2003,
    /// *Anim. Behav.* 66:1093). Task allocation by response thresholds
    /// (Bonabeau, Theraulaz & Deneubourg 1996, *Proc. R. Soc. B* 263:1565)
    /// with age polyethism (Wilson 1976). Development egg→worker ≈ 45 days.
    pub fn lasius_niger() -> Self {
        Species {
            name: "Lasius niger".to_string(),
            speed_cm_s: 1.5,
            loaded_speed_factor: 0.8,
            trail_speed_factor: 1.15,
            trail: PheromoneParams {
                half_life_s: 47.0 * 60.0,
                diffusion_per_s: 0.002,
                cap: 4000.0,
                k: 20.0,
            },
            choice_exponent: 2.0,
            trail_deposit: 1.0,
            outbound_laying: false,
            lay_quality_half: 0.3,
            lay_exponent: 2.0,
            lay_max_probability: 0.9,
            fidelity_quality_half: 0.15,
            min_load_fraction: 0.3,
            home: PheromoneParams::inert(),
            uses_home_pheromone: false,
            territory: PheromoneParams {
                half_life_s: 6.0 * 3600.0,
                diffusion_per_s: 0.001,
                cap: 100.0,
                k: 5.0,
            },
            territory_deposit: 0.02,
            no_entry: PheromoneParams::inert(),
            uses_no_entry: false,
            no_entry_deposit: 0.0,
            alarm: PheromoneParams {
                half_life_s: 40.0,
                diffusion_per_s: 0.15,
                cap: 200.0,
                k: 5.0,
            },
            alarm_release: 60.0,
            sense_range: 2,
            pi_heading_noise_deg: 4.0,
            pi_distance_noise: 0.05,
            arrival_radius: 2.5,
            search_time_s: 90.0,
            give_up_time_s: 15.0 * 60.0,
            search_temperature_factor: 2.5,
            crop_capacity: 1.0,
            feeding_time_s: (45.0, 120.0),
            unloading_time_s: 30.0,
            unloading_satiation_factor: 3.0,
            forager_hazard_per_s: 1.0 / (24.0 * 3600.0),
            starvation_s: 8.0 * 3600.0,
            threshold_median: 0.5,
            threshold_spread: lognormal_sigma_for_cv(1.0),
            threshold_exponent: 2.0,
            decision_interval_s: 20.0,
            maturation_s: 3.0 * 24.0 * 3600.0,
            youth_threshold_factor: 4.0,
            nursing_threshold_median: 0.5,
            nursing_bout_s: 300.0,
            brood_per_nurse: 5.0,
            reforage_bonus: 1.0,
            threshold_learning_s: 10.0 * 60.0,
            threshold_forgetting_s: 3600.0,
            threshold_bounds: (0.1, 10.0),
            excitation_per_return: 0.15,
            excitation_half_life_s: 120.0,
            hunger_gain: 1.0,
            consumption_per_ant_per_s: 0.4 / 3600.0,
            egg_interval_s: 3600.0,
            development_s: 45.0 * 24.0 * 3600.0,
            brood_food: 3.0,
            nursing_rate: 0.5 / 60.0,
        }
    }

    /// *Linepithema humile*, the Argentine ant: the species of the
    /// double-bridge experiments (Goss, Aron, Deneubourg & Pasteels 1989,
    /// *Naturwissenschaften* 76:579). Lays trail in both directions; trail
    /// pheromone decays within tens of minutes; walks ≈ 2 cm/s.
    pub fn argentine() -> Self {
        let mut s = Species::lasius_niger();
        s.name = "Linepithema humile".to_string();
        s.speed_cm_s = 2.0;
        s.trail = PheromoneParams {
            half_life_s: 20.0 * 60.0,
            diffusion_per_s: 0.002,
            cap: 4000.0,
            k: 20.0,
        };
        s.outbound_laying = true;
        s.lay_quality_half = 0.2;
        s.reforage_bonus = 1.2;
        s
    }

    /// *Monomorium pharaonis*, Pharaoh's ant: as the Argentine profile plus
    /// the repellent "no entry" pheromone laid on unrewarding routes
    /// (Robinson et al. 2005), which decays faster than the trail.
    pub fn pharaoh() -> Self {
        let mut s = Species::argentine();
        s.name = "Monomorium pharaonis".to_string();
        s.speed_cm_s = 1.2;
        s.no_entry = PheromoneParams {
            half_life_s: 10.0 * 60.0,
            diffusion_per_s: 0.002,
            cap: 200.0,
            k: 10.0,
        };
        s.uses_no_entry = true;
        s.no_entry_deposit = 1.0;
        s
    }

    /// *Cataglyphis*, the desert ant: no trail pheromone; solitary foragers
    /// navigate by path integration and search systematically around the
    /// fictive nest (Wehner & Srinivasan 1981, *J. Comp. Physiol.* 142:315;
    /// Müller & Wehner 1988, *PNAS* 85:5287). Fast walkers, hot habitat,
    /// high forager mortality.
    pub fn cataglyphis() -> Self {
        let mut s = Species::lasius_niger();
        s.name = "Cataglyphis".to_string();
        s.speed_cm_s = 12.0;
        s.trail = PheromoneParams::inert();
        s.trail_deposit = 0.0;
        s.lay_max_probability = 0.0;
        s.territory_deposit = 0.0;
        s.territory = PheromoneParams::inert();
        s.pi_heading_noise_deg = 3.0;
        s.pi_distance_noise = 0.04;
        s.search_time_s = 300.0;
        s.forager_hazard_per_s = 1.0 / (6.0 * 3600.0);
        s.reforage_bonus = 1.5;
        s.sense_range = 1;
        s
    }

    /// Compress the life-history clocks by `factor` so that slow colony
    /// dynamics (worker maturation, brood development, egg laying, brood
    /// feeding) unfold within a short run. Behavioural clocks (walking,
    /// feeding, trail evaporation, task decisions, survival) are left alone,
    /// so foraging still looks like foraging. A modelling convenience for
    /// demonstrations, not a physical rescaling.
    pub fn compressed(mut self, factor: f64) -> Self {
        let f = factor.max(1e-9);
        self.maturation_s /= f;
        self.egg_interval_s /= f;
        self.development_s /= f;
        self.nursing_rate *= f;
        self
    }

    /// Kinetic parameters of every channel.
    pub fn pheromones(&self) -> PheromoneSet {
        let mut set = [
            self.trail.clone(),
            self.home.clone(),
            self.territory.clone(),
            self.no_entry.clone(),
            self.alarm.clone(),
        ];
        if !self.uses_home_pheromone {
            set[Pheromone::Home.index()] = PheromoneParams::inert();
        }
        if !self.uses_no_entry {
            set[Pheromone::NoEntry.index()] = PheromoneParams::inert();
        }
        set
    }

    /// Probability that a forager returning from food of `quality` (0..1)
    /// lays trail (Beckers et al. 1993: a Hill function of concentration).
    pub fn lay_probability(&self, quality: f64) -> f64 {
        let q = quality.clamp(0.0, 1.0).powf(self.lay_exponent);
        let h = self.lay_quality_half.powf(self.lay_exponent);
        if q + h <= 0.0 {
            0.0
        } else {
            self.lay_max_probability * q / (q + h)
        }
    }

    /// Probability that a forager keeps a remembered source of `quality`
    /// after unloading (a Hill function with the laying exponent).
    pub fn site_fidelity(&self, quality: f64) -> f64 {
        let q = quality.clamp(0.0, 1.0).powf(self.lay_exponent);
        let h = self.fidelity_quality_half.powf(self.lay_exponent);
        if q + h <= 0.0 {
            0.0
        } else {
            q / (q + h)
        }
    }

    /// Fraction of the crop filled at food of `quality`.
    pub fn load_fraction(&self, quality: f64) -> f64 {
        let f = self.min_load_fraction.clamp(0.0, 1.0);
        f + (1.0 - f) * quality.clamp(0.0, 1.0)
    }

    /// Time to fill the crop at food of `quality`, seconds.
    pub fn feeding_time(&self, quality: f64) -> f64 {
        self.feeding_time_s.0 + self.feeding_time_s.1 * quality.clamp(0.0, 1.0)
    }

    /// Time to unload by trophallaxis when the colony's satiation is
    /// `satiation` (0 hungry .. 1 replete), seconds.
    pub fn unloading_time(&self, satiation: f64) -> f64 {
        self.unloading_time_s * (1.0 + self.unloading_satiation_factor * satiation.clamp(0.0, 1.0))
    }

    /// Response-threshold engagement probability per decision:
    /// `sⁿ / (sⁿ + θⁿ)` (Bonabeau et al. 1996).
    pub fn response(&self, stimulus: f64, threshold: f64) -> f64 {
        let n = self.threshold_exponent;
        let s = stimulus.max(0.0).powf(n);
        let t = threshold.max(1e-9).powf(n);
        if s + t <= 0.0 {
            0.0
        } else {
            s / (s + t)
        }
    }

    /// Foraging threshold of a worker of the given age, given its adult
    /// threshold: young workers are reluctant to forage.
    pub fn threshold_at_age(&self, adult_threshold: f64, age_s: f64) -> f64 {
        let youth = (-age_s / self.maturation_s.max(1e-9)).exp();
        adult_threshold * (1.0 + (self.youth_threshold_factor - 1.0) * youth)
    }

    /// The species' movement instinct as a behavioral surface at a fixed
    /// temperature of 1: outbound, follow the recruitment trail with
    /// Deneubourg's exponent, head for a remembered site, avoid alarm and
    /// no-entry marks, keep momentum; inbound, follow the path-integration
    /// home vector, the nest, and (where laid) the outbound trail.
    pub fn instinct(&self) -> BehavioralSurface {
        let n = self.choice_exponent;
        let mut weights = [0.0; FEATURES];
        let out = &mut weights[..BASE_FEATURES];
        out[F_TRAIL] = n;
        out[F_HOME] = 0.0;
        out[F_TERRITORY] = 0.3;
        out[F_NO_ENTRY] = -2.0;
        out[F_ALARM] = -2.0;
        out[F_FOOD] = 4.0;
        out[F_NEST] = -1.0;
        out[F_HEADING] = 1.0;
        out[F_HOME_VECTOR] = -0.3;
        out[F_SITE] = 2.0;
        out[F_RECENT] = -0.5;
        out[F_CROWD] = -0.1;
        let inb = &mut weights[BASE_FEATURES..];
        inb[F_TRAIL] = 0.5 * n;
        inb[F_HOME] = n;
        inb[F_TERRITORY] = 0.5;
        inb[F_NO_ENTRY] = 0.0;
        inb[F_ALARM] = -2.0;
        inb[F_FOOD] = 0.0;
        inb[F_NEST] = 4.0;
        inb[F_HEADING] = 0.8;
        inb[F_HOME_VECTOR] = 3.0;
        inb[F_SITE] = 0.0;
        inb[F_RECENT] = -0.5;
        inb[F_CROWD] = -0.1;
        BehavioralSurface {
            weights,
            entropy: EntropyControl::fixed(1.0),
            deformation: Deformation::none(),
        }
    }
}

impl Default for Species {
    fn default() -> Self {
        Species::lasius_niger()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_are_internally_consistent() {
        for s in [
            Species::lasius_niger(),
            Species::argentine(),
            Species::pharaoh(),
            Species::cataglyphis(),
        ] {
            assert!(s.speed_cm_s > 0.0);
            let set = s.pheromones();
            assert_eq!(
                set[Pheromone::Home.index()].is_active(),
                s.uses_home_pheromone
            );
            assert_eq!(set[Pheromone::NoEntry.index()].is_active(), s.uses_no_entry);
            assert!(s.response(1.0, 1.0) - 0.5 < 1e-12);
            assert!(s.response(0.0, 1.0) == 0.0);
            assert!(s.lay_probability(1.0) <= s.lay_max_probability + 1e-12);
            assert!(s.lay_probability(0.0) == 0.0);
        }
        assert!(!Species::cataglyphis().pheromones()[Pheromone::Trail.index()].is_active());
        assert!(Species::pharaoh().pheromones()[Pheromone::NoEntry.index()].is_active());
        let instinct = Species::lasius_niger().instinct();
        assert_eq!(instinct.weights[F_TRAIL], 2.0);
        assert_eq!(instinct.weights[BASE_FEATURES + F_HOME_VECTOR], 3.0);
        assert_eq!(instinct.entropy, EntropyControl::fixed(1.0));
    }

    #[test]
    fn quality_modulates_laying_and_feeding() {
        let s = Species::lasius_niger();
        assert!(s.lay_probability(0.3) < s.lay_probability(1.0));
        assert!(
            (s.lay_probability(0.3) - 0.45).abs() < 1e-9,
            "half at the half quality"
        );
        assert!(s.feeding_time(1.0) > s.feeding_time(0.1));
        assert!(s.unloading_time(1.0) > s.unloading_time(0.0));
        assert!(s.site_fidelity(1.0) > 0.95 && s.site_fidelity(0.1) < 0.4);
        assert!((s.site_fidelity(0.15) - 0.5).abs() < 1e-9);
        assert!((s.load_fraction(0.0) - 0.3).abs() < 1e-12 && s.load_fraction(1.0) == 1.0);
    }

    #[test]
    fn age_lowers_the_foraging_threshold() {
        let s = Species::lasius_niger();
        let young = s.threshold_at_age(1.0, 0.0);
        let old = s.threshold_at_age(1.0, 100.0 * s.maturation_s);
        assert!((young - s.youth_threshold_factor).abs() < 1e-9);
        assert!((old - 1.0).abs() < 1e-6);
    }

    #[test]
    fn compression_scales_life_history_only() {
        let s = Species::lasius_niger();
        let c = s.clone().compressed(60.0);
        assert_eq!(c.trail.half_life_s, s.trail.half_life_s);
        assert_eq!(c.give_up_time_s, s.give_up_time_s);
        assert_eq!(c.consumption_per_ant_per_s, s.consumption_per_ant_per_s);
        assert!((c.maturation_s * 60.0 - s.maturation_s).abs() < 1e-3);
        assert!((c.development_s * 60.0 - s.development_s).abs() < 1e-3);
        assert!((c.nursing_rate / 60.0 - s.nursing_rate).abs() < 1e-12);
    }
}
