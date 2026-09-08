//! Species profiles: literature-based parameters for the biological model.
//!
//! Every quantity is in physical units (seconds, centimetres, microlitres,
//! moles per litre, milligrams, pheromone "marks", degrees Celsius) and is
//! converted to grid and tick units by the simulation. The defaults are
//! order-of-magnitude values from the cited studies; where the literature
//! gives ranges, a representative value is used and noted.
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
//!   solitary foraging by path integration, route memory and systematic
//!   search.

use crate::ant::{
    BASE_FEATURES, FEATURES, F_ALARM, F_CROWD, F_FOOD, F_HEADING, F_HOME, F_HOME_VECTOR, F_NEST,
    F_NO_ENTRY, F_ODOUR, F_RECENT, F_ROUTE, F_SITE, F_TERRITORY, F_TRAIL, F_WALL,
};
use crate::entropy::EntropyControl;
use crate::pheromone::{Pheromone, PheromoneParams, PheromoneSet};
use crate::surface::{BehavioralSurface, Deformation};

/// Milligrams of sucrose per microlitre of a one-molar solution
/// (342 g/mol).
pub const SUGAR_MG_PER_UL_PER_MOLAR: f64 = 0.342;

/// Parameters of one species.
#[derive(Clone, Debug, PartialEq)]
pub struct Species {
    /// Name.
    pub name: String,

    // ---- locomotion ----
    /// Walking speed of an unloaded forager at the reference temperature,
    /// cm/s.
    pub speed_cm_s: f64,
    /// Speed multiplier when carrying a full crop.
    pub loaded_speed_factor: f64,
    /// Speed multiplier on a strong trail (ants on established trails walk
    /// straighter and faster).
    pub trail_speed_factor: f64,
    /// Fraction of speed lost in a cell at capacity: ant traffic slows only
    /// mildly with density and does not jam (John, Schadschneider, Chowdhury
    /// & Nishinari 2009, *Phys. Rev. Lett.* 102:108001), but pushes and
    /// stalls at the densities of a crowded bridge (Dussutour et al. 2004).
    pub crowding_slowdown: f64,
    /// Trail deposition falls with crowding (occupancy over capacity):
    /// `1 / (1 + crowding_deposition × crowding)`, half at capacity
    /// (Czaczkes, Grüter & Ratnieks 2013, *Proc. R. Soc. B* 280:20122540:
    /// head-on encounters on a crowded trail reduce pheromone deposition).
    pub crowding_deposition: f64,
    /// Weight against entering a crowded patch, per unit of crowding
    /// (occupancy over capacity): at a jammed entrance it outweighs the
    /// trail, which is how a crowded colony comes to use both branches
    /// of a narrow bridge (Dussutour et al. 2004).
    pub crowding_avoidance: f64,
    /// Temperature below which walking stops, °C (speed rises linearly from
    /// here to the reference temperature: Hurlbert, Ballantyne & Powell
    /// 2008, *Ecol. Entomol.* 33:144).
    pub speed_t_min_c: f64,
    /// Reference temperature at which `speed_cm_s` applies, °C.
    pub speed_t_ref_c: f64,

    // ---- recruitment trail ----
    /// Kinetics of the recruitment trail at the reference temperature.
    pub trail: PheromoneParams,
    /// Deneubourg's choice exponent `n`.
    pub choice_exponent: f64,
    /// Marks deposited per cell length walked while laying at unit strength.
    pub trail_deposit: f64,
    /// Whether ants also lay trail on the way out to a known source.
    pub outbound_laying: bool,
    /// Whether naive ants lay (weakly) while exploring, as Argentine ants
    /// do (Aron, Pasteels & Deneubourg 1989, *Biol. Behav.* 14:207).
    pub exploratory_laying: bool,
    /// Trail weight on the way home as a fraction of the choice exponent.
    /// Trail-laying species read the trail in both directions (the
    /// bidirectional models of Beckers, Deneubourg & Goss 1992); path
    /// integration and route memory act alongside it inbound.
    pub inbound_trail_factor: f64,
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
    /// Fraction of the crop a forager fills at food of zero quality; the
    /// desired load rises linearly to a full crop at quality one (crop load
    /// increases with concentration: Josens, Farina & Roces 1998, *J. Insect
    /// Physiol.* 44:579).
    pub min_load_fraction: f64,
    /// Q10 of pheromone evaporation: the decay rate multiplies by this per
    /// 10 °C above the reference temperature.
    pub q10_evaporation: f64,

    // ---- other channels ----
    /// Outbound "home" trail kinetics (inert unless `uses_home_pheromone`).
    pub home: PheromoneParams,
    /// Whether the species lays a separate outbound trail followed home.
    pub uses_home_pheromone: bool,
    /// Home-range marking kinetics.
    pub territory: PheromoneParams,
    /// Home-range marks deposited per cell length by every worker.
    pub territory_deposit: f64,
    /// Repellent "no entry" kinetics.
    pub no_entry: PheromoneParams,
    /// Whether unsuccessful returning foragers lay "no entry" marking.
    pub uses_no_entry: bool,
    /// No-entry marks deposited per cell length.
    pub no_entry_deposit: f64,
    /// Alarm pheromone kinetics.
    pub alarm: PheromoneParams,
    /// Alarm released at the death of a worker.
    pub alarm_release: f64,
    /// Kinetics of the smell of food in the air.
    pub food_odour: PheromoneParams,
    /// Odour given off per microlitre of solution per second (a drop's
    /// surface saturates: only the first few microlitres count).
    pub odour_per_ul_s: f64,
    /// Odour given off per milligram of prey per second.
    pub odour_per_mg_s: f64,

    // ---- sensing ----
    /// Cells ahead an antennal sweep integrates (1 or 2).
    pub sense_range: usize,
    /// Distance at which a landmark can be seen, cm. Ants take a view of
    /// the landmarks around the nest and around a food site and fix their
    /// position from them when the same landmark comes into view again
    /// (Wehner & Räber 1979, *Experientia* 35:1569; Collett 1992, *J.
    /// Comp. Physiol. A* 170:435).
    pub sight_cm: f64,
    /// How far the path-integration estimate moves towards the position
    /// a familiar landmark gives, per sighting.
    pub landmark_correction: f64,

    // ---- navigation ----
    /// Persistence length of the direction of travel, cm: the heading an
    /// ant turns relative to is a running mean of its recent steps over
    /// this distance, so the wobble inside a corridor does not decide the
    /// next fork; turns of more than a right angle reorient it outright.
    pub heading_persistence_cm: f64,
    /// Standard deviation of the heading error per cell walked, degrees.
    pub pi_heading_noise_deg: f64,
    /// Standard deviation of the relative odometric error per cell walked.
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
    /// Number of familiar places a worker can remember local vectors for
    /// (route memory: Collett & Collett 2002, *Nat. Rev. Neurosci.* 3:542;
    /// Mangan & Webb 2012, *Behav. Ecol.* 23:944).
    pub route_capacity: usize,
    /// Weight of a remembered route against the other cues (1.5 lets a
    /// well-learned route override a trail in *Lasius niger*: Grüter,
    /// Czaczkes & Ratnieks 2011).
    pub route_weight: f64,
    /// Rate at which a remembered local vector moves towards the current
    /// path-integration estimate on each visit.
    pub route_learning_rate: f64,
    /// Fraction by which the path-integration estimate is pulled towards a
    /// well-known local vector on recognising a place.
    pub route_pi_correction: f64,

    // ---- feeding ----
    /// Molarity that counts as quality one for the behavioural responses.
    pub reference_molarity: f64,
    /// Crop capacity, microlitres (Lasius niger imbibes a fraction of a
    /// microlitre: Mailleux et al. 2000).
    pub crop_capacity_ul: f64,
    /// Intake rate of a dilute solution, µl/s.
    pub intake_max_ul_s: f64,
    /// Molarity at which the intake rate halves (viscosity: Josens et al.
    /// 1998).
    pub intake_half_molarity: f64,
    /// Longest a forager waits at a source that offers solution slower than
    /// it drinks before leaving with what it has, seconds (at a slow drip,
    /// ants ingest less and recruit less: Mailleux, Deneubourg & Detrain
    /// 2003, *Proc. R. Soc. B* 270:1609).
    pub feeding_patience_s: f64,
    /// Exponent on the crop's fill fraction in the probability of laying
    /// trail on the way home: foragers that ingested little do not recruit
    /// (Mailleux, Deneubourg & Detrain 2000, *Anim. Behav.* 59:1061).
    pub lay_load_exponent: f64,
    // ---- trophallaxis ----
    /// Interval between the contacts of a returning forager offering its
    /// load, seconds (Greenwald, Baltiansky & Feinerman 2018, *eLife*
    /// 7:e31730: interactions every few seconds).
    pub contact_interval_s: f64,
    /// Share of the receiver's empty crop space handed over in one contact
    /// (the amount given is proportional to the receiver's deficit:
    /// Greenwald et al. 2018).
    pub transfer_fraction: f64,
    /// A forager stops offering once its crop is below this fraction of
    /// its capacity, and keeps the rest for itself.
    pub unload_residual_fraction: f64,
    /// Contacts after which a forager that still cannot unload gives up.
    pub max_unloading_contacts: u32,
    /// Interval between the sharing contacts of a worker inside, seconds:
    /// at each, the fuller of two crops passes part of the difference to
    /// the emptier, so food spreads through the colony and crop loads even
    /// out (Buffin et al. 2009, *PLoS ONE* 4:e5919; Greenwald, Segre &
    /// Feinerman 2015, *Sci. Rep.* 5:12496). The nest's reserve takes part
    /// as one more nestmate.
    pub sharing_interval_s: f64,

    // ---- protein foraging ----
    /// Mass of prey a worker cuts off and carries, milligrams.
    pub prey_load_mg: f64,
    /// Time to cut a piece of prey, seconds.
    pub prey_handling_s: f64,
    /// Probability that a forager takes prey it finds when the colony has
    /// no protein demand: workers themselves avoid protein, which shortens
    /// their lives (Dussutour & Simpson 2012, *Proc. R. Soc. B* 279:2402);
    /// with larvae to feed the colony turns to protein (Dussutour &
    /// Simpson 2009, *Curr. Biol.* 19:740).
    pub protein_acceptance_base: f64,
    /// Foraging stimulus per unit of the colony's protein demand.
    pub protein_demand_gain: f64,

    // ---- necrophoresis ----
    /// Corpses near a carrier at which dropping is one quarter likely per
    /// cell entered: `p = (n / (k + n))²` (Deneubourg et al. 1991; Theraulaz
    /// et al. 2002, *PNAS* 99:9645: dropping grows with the pile, picking
    /// up shrinks with it, and cemeteries emerge).
    pub corpse_drop_k: f64,
    /// Corpses near a corpse at which picking it up is one quarter likely
    /// per encounter: `p = (k / (k + n))²`.
    pub corpse_pickup_k: f64,
    /// Chance per cell entered of dropping a corpse away from any pile.
    pub corpse_base_drop: f64,
    /// Distance from the nest beyond which an undertaker will drop its
    /// load, cm (refuse piles lie away from the nest).
    pub refuse_distance_cm: f64,
    /// Stimulus per corpse inside the nest for a worker to carry one out.
    pub undertaking_gain: f64,

    // ---- survival ----
    /// Hazard rate outside the nest, per second (predation and mishap).
    pub forager_hazard_per_s: f64,
    /// Critical thermal maximum, °C: the temperature at which a worker
    /// outside dies within minutes. Mortality rises steeply towards it,
    /// so foraging near the limit trades speed against losses (Cerdá,
    /// Retana & Cros 1998, *Funct. Ecol.* 12:45).
    pub critical_thermal_max_c: f64,
    /// Heat hazard rate at the critical thermal maximum, per second.
    pub heat_hazard_at_max_per_s: f64,
    /// Temperature interval over which the heat hazard falls by a factor
    /// e below the maximum, °C.
    pub heat_hazard_scale_c: f64,
    /// Time an ant survives away from food, seconds.
    pub starvation_s: f64,

    // ---- task allocation ----
    /// Coefficient of variation of worker body mass. Speed scales with
    /// mass to the 0.3 (longer legs: Hurlbert, Ballantyne & Powell 2008),
    /// crop and prey loads with mass, metabolism with mass to the 0.75.
    pub size_cv: f64,
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
    /// Larvae one nurse can tend: the nursing stimulus is the larval load
    /// per nurse already at work, so recruitment of nurses damps it.
    pub brood_per_nurse: f64,
    /// Walking speed inside the nest, centimetres per second: workers
    /// inside move in short bouts between rests, far slower than a
    /// forager on a trail.
    pub nest_speed_cm_s: f64,
    /// Width of a worker's spatial fidelity zone, as a fraction of the
    /// nest's depth: zones are fuzzy and overlap (Sendova-Franks & Franks
    /// 1995, *Behav. Ecol. Sociobiol.* 36:269).
    pub zone_spread: f64,
    /// Depth (0 the brood at the centre, 1 the entrance ring) that callow
    /// workers keep to. Zones drift outward with age, reaching the
    /// entrance after twice `maturation_s`, so that nurses sit with the
    /// brood and foragers by the entrance (Sendova-Franks & Franks 1995;
    /// Mersch, Crespi & Keller 2013, *Science* 340:1090).
    pub callow_depth: f64,
    /// Depth within which the brood lies: the brood chamber. Larvae are
    /// fed only by nurses standing in it.
    pub brood_depth: f64,
    /// Distance from the chamber, centimetres, over which the brood's
    /// demand fades for a worker standing outside it: contact range,
    /// about half a body length, so that workers do the tasks they meet
    /// where they stand (foraging for work, Tofts & Franks 1992, *Anim.
    /// Behav.* 43:1).
    pub brood_reach_cm: f64,
    /// Depth to which a returning forager goes in to hand over its load:
    /// unloading happens near the entrance, and the receivers carry the
    /// food inward (Greenwald, Segre & Feinerman 2015, *eLife* 4:e07128).
    pub unloading_depth: f64,
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
    /// Below this temperature foraging stops, °C.
    pub forage_min_c: f64,
    /// Above this temperature foraging stops, °C.
    pub forage_max_c: f64,

    // ---- colony ----
    /// Excitation a contact adds to a nestmate per unit of load quality.
    pub excitation_per_contact: f64,
    /// Half-life of a worker's excitation, seconds.
    pub excitation_half_life_s: f64,
    /// Foraging stimulus per unit of colony hunger. Kept low: hunger alone
    /// sends out a trickle of scouts, and it is the excitation spread by
    /// returning foragers that mobilises the colony, so the foraging force
    /// builds up sigmoidally as recruitment feeds on itself (Mailleux,
    /// Detrain & Deneubourg 2006, *J. Exp. Biol.* 209:4224).
    pub hunger_gain: f64,
    /// Sugar consumed per worker inside the nest per second at the
    /// reference temperature, milligrams.
    pub consumption_mg_per_ant_per_s: f64,
    /// Q10 of metabolism.
    pub q10_metabolism: f64,
    /// Interval between eggs laid by the queen when the colony is fed,
    /// seconds.
    pub egg_interval_s: f64,
    /// Egg stage duration at the reference temperature, seconds.
    pub egg_s: f64,
    /// Larval stage duration at the reference temperature, seconds.
    pub larva_s: f64,
    /// Pupal stage duration at the reference temperature, seconds.
    pub pupa_s: f64,
    /// Q10 of brood development (Kipyatkov & Lopatina 2015, *Adv. Insect
    /// Physiol.* 48:129).
    pub q10_development: f64,
    /// Protein a larva must receive to pupate, milligrams.
    pub larva_protein_mg: f64,
    /// Sugar a larva must receive to pupate, milligrams.
    pub larva_food_mg: f64,
    /// Time a larva survives without being fed, seconds.
    pub larva_starvation_s: f64,
    /// Sugar a nurse hands to larvae per second, milligrams.
    pub nursing_rate_mg_s: f64,
}

fn lognormal_sigma_for_cv(cv: f64) -> f64 {
    (1.0 + cv * cv).ln().sqrt()
}

fn q10_factor(q10: f64, temperature_c: f64, reference_c: f64) -> f64 {
    q10.max(1e-9).powf((temperature_c - reference_c) / 10.0)
}

impl Species {
    /// Reference temperature of every rate constant, °C.
    pub const REFERENCE_C: f64 = 22.0;

    /// *Lasius niger*, the black garden ant.
    ///
    /// Speed ≈ 1.5 cm/s at room temperature (Hurlbert et al. 2008 report
    /// 1–3 cm/s). Trail pheromone half-life ≈ 47 min and trail laying rising
    /// with sucrose concentration (Beckers, Deneubourg & Goss 1993, *J.
    /// Insect Behav.* 6:751). Choice parameters `k = 20`, `n = 2`
    /// (Deneubourg et al. 1990). Crop loads of a fraction of a microlitre
    /// (Mailleux et al. 2000), intake slowing with viscosity (Josens et al.
    /// 1998). Colony hunger drives foraging (Mailleux, Deneubourg & Detrain
    /// 2003, *Anim. Behav.* 66:1093). Task allocation by response thresholds
    /// (Bonabeau, Theraulaz & Deneubourg 1996, *Proc. R. Soc. B* 263:1565)
    /// with age polyethism (Wilson 1976) and reinforcement (Theraulaz et al.
    /// 1998). Egg, larval and pupal stages of roughly 10, 14 and 12 days at
    /// 22 °C.
    pub fn lasius_niger() -> Self {
        let day = 24.0 * 3600.0;
        Species {
            name: "Lasius niger".to_string(),
            speed_cm_s: 1.5,
            loaded_speed_factor: 0.8,
            trail_speed_factor: 1.15,
            crowding_slowdown: 0.5,
            crowding_deposition: 1.0,
            crowding_avoidance: 2.0,
            speed_t_min_c: 5.0,
            speed_t_ref_c: Self::REFERENCE_C,
            // A substrate deposit: it evaporates but hardly spreads sideways
            // (its vapour active space is within antennal reach, which the
            // patch-by-patch sensing covers), so lateral diffusion is a
            // small leak rather than a transport term.
            trail: PheromoneParams {
                half_life_s: 47.0 * 60.0,
                diffusion_per_s: 0.0002,
                cap: 4000.0,
                k: 20.0,
            },
            choice_exponent: 2.0,
            trail_deposit: 1.0,
            // Foragers lay on the return trip and, less intensely, on the
            // way out to a source they know (Beckers, Deneubourg & Goss
            // 1992, *Insectes Soc.* 39:59).
            outbound_laying: true,
            exploratory_laying: false,
            inbound_trail_factor: 1.0,
            lay_quality_half: 0.3,
            lay_exponent: 2.0,
            lay_max_probability: 0.9,
            fidelity_quality_half: 0.15,
            min_load_fraction: 0.3,
            q10_evaporation: 2.0,
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
            food_odour: PheromoneParams {
                half_life_s: 60.0,
                diffusion_per_s: 0.3,
                cap: 500.0,
                k: 5.0,
            },
            odour_per_ul_s: 0.5,
            odour_per_mg_s: 0.5,
            sense_range: 2,
            sight_cm: 20.0,
            landmark_correction: 0.8,
            heading_persistence_cm: 5.0,
            pi_heading_noise_deg: 4.0,
            pi_distance_noise: 0.05,
            arrival_radius: 2.5,
            search_time_s: 90.0,
            give_up_time_s: 15.0 * 60.0,
            search_temperature_factor: 2.5,
            route_capacity: 400,
            route_weight: 1.5,
            route_learning_rate: 0.3,
            route_pi_correction: 0.3,
            reference_molarity: 1.0,
            crop_capacity_ul: 0.5,
            intake_max_ul_s: 0.015,
            intake_half_molarity: 0.7,
            corpse_drop_k: 4.0,
            corpse_pickup_k: 2.0,
            corpse_base_drop: 0.002,
            refuse_distance_cm: 10.0,
            undertaking_gain: 0.5,
            prey_load_mg: 1.0,
            prey_handling_s: 60.0,
            protein_acceptance_base: 0.1,
            protein_demand_gain: 0.3,
            feeding_patience_s: 300.0,
            lay_load_exponent: 1.0,
            contact_interval_s: 5.0,
            transfer_fraction: 0.5,
            unload_residual_fraction: 0.2,
            max_unloading_contacts: 30,
            sharing_interval_s: 20.0,
            forager_hazard_per_s: 1.0 / day,
            critical_thermal_max_c: 44.0,
            heat_hazard_at_max_per_s: 1.0 / 300.0,
            heat_hazard_scale_c: 1.5,
            starvation_s: 8.0 * 3600.0,
            size_cv: 0.15,
            threshold_median: 0.5,
            threshold_spread: lognormal_sigma_for_cv(1.0),
            threshold_exponent: 2.0,
            decision_interval_s: 20.0,
            maturation_s: 3.0 * day,
            youth_threshold_factor: 4.0,
            nursing_threshold_median: 0.5,
            nursing_bout_s: 300.0,
            brood_per_nurse: 5.0,
            nest_speed_cm_s: 0.5,
            zone_spread: 0.15,
            callow_depth: 0.1,
            brood_depth: 0.5,
            brood_reach_cm: 1.0,
            unloading_depth: 0.75,
            reforage_bonus: 1.0,
            threshold_learning_s: 10.0 * 60.0,
            threshold_forgetting_s: 3600.0,
            threshold_bounds: (0.1, 10.0),
            forage_min_c: 10.0,
            forage_max_c: 42.0,
            excitation_per_contact: 0.3,
            excitation_half_life_s: 120.0,
            hunger_gain: 0.3,
            consumption_mg_per_ant_per_s: 0.05 / day,
            q10_metabolism: 2.0,
            egg_interval_s: 3600.0,
            egg_s: 10.0 * day,
            larva_s: 14.0 * day,
            pupa_s: 12.0 * day,
            q10_development: 2.5,
            larva_protein_mg: 0.3,
            larva_food_mg: 0.8,
            larva_starvation_s: 5.0 * day,
            nursing_rate_mg_s: 0.5 / 3600.0,
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
            diffusion_per_s: 0.0002,
            cap: 4000.0,
            k: 20.0,
        };
        s.outbound_laying = true;
        s.exploratory_laying = true;
        // A mass recruiter: orientation rests on the trail far more than on
        // individual memory (Aron, Beckers, Deneubourg & Pasteels 1993,
        // *Insectes Soc.* 40:369, comparing this species with L. niger).
        s.route_weight = 0.5;
        s.inbound_trail_factor = 1.0;
        s.lay_quality_half = 0.2;
        s.reforage_bonus = 1.2;
        s.crop_capacity_ul = 0.3;
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
            diffusion_per_s: 0.0002,
            cap: 200.0,
            k: 10.0,
        };
        s.uses_no_entry = true;
        s.no_entry_deposit = 1.0;
        s.crop_capacity_ul = 0.2;
        s
    }

    /// *Cataglyphis*, the desert ant: no trail pheromone; solitary foragers
    /// navigate by path integration, route memory and systematic search
    /// (Wehner & Srinivasan 1981, *J. Comp. Physiol.* 142:315; Müller &
    /// Wehner 1988, *PNAS* 85:5287). Fast walkers of a hot habitat, high
    /// forager mortality.
    pub fn cataglyphis() -> Self {
        let mut s = Species::lasius_niger();
        s.name = "Cataglyphis".to_string();
        s.speed_cm_s = 12.0;
        s.speed_t_min_c = 15.0;
        s.speed_t_ref_c = 35.0;
        s.forage_min_c = 25.0;
        s.forage_max_c = 55.0;
        // Forages within a degree or two of its thermal limit (Cerdá et al.
        // 1998).
        s.critical_thermal_max_c = 56.0;
        s.trail = PheromoneParams::inert();
        s.trail_deposit = 0.0;
        s.lay_max_probability = 0.0;
        s.territory_deposit = 0.0;
        s.territory = PheromoneParams::inert();
        s.heading_persistence_cm = 10.0;
        // Strongly size-polymorphic foragers (Cerdá & Retana 1997,
        // *Oecologia* 111:305).
        s.size_cv = 0.35;
        // Landmarks are read at a distance in open desert.
        s.sight_cm = 60.0;
        s.pi_heading_noise_deg = 3.0;
        s.pi_distance_noise = 0.04;
        s.search_time_s = 300.0;
        s.forager_hazard_per_s = 1.0 / (6.0 * 3600.0);
        s.reforage_bonus = 1.5;
        s.sense_range = 1;
        s.crop_capacity_ul = 2.0;
        s.route_capacity = 800;
        s.route_weight = 2.0;
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
        self.egg_s /= f;
        self.larva_s /= f;
        self.pupa_s /= f;
        self.larva_starvation_s /= f;
        self.nursing_rate_mg_s *= f;
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
            self.food_odour.clone(),
        ];
        if !self.uses_home_pheromone {
            set[Pheromone::Home.index()] = PheromoneParams::inert();
        }
        if !self.uses_no_entry {
            set[Pheromone::NoEntry.index()] = PheromoneParams::inert();
        }
        set
    }

    /// Food quality in `0..=1` of a solution: its molarity relative to the
    /// reference, clamped.
    pub fn quality(&self, molarity: f64) -> f64 {
        (molarity / self.reference_molarity.max(1e-9)).clamp(0.0, 1.0)
    }

    /// Intake rate at a given molarity, µl/s: `max / (1 + (M / M½)²)`.
    pub fn intake_rate(&self, molarity: f64) -> f64 {
        let r = molarity.max(0.0) / self.intake_half_molarity.max(1e-9);
        self.intake_max_ul_s / (1.0 + r * r)
    }

    /// Sugar content of a volume of solution, milligrams.
    pub fn sugar_mg(&self, volume_ul: f64, molarity: f64) -> f64 {
        volume_ul.max(0.0) * molarity.max(0.0) * SUGAR_MG_PER_UL_PER_MOLAR
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

    /// Fraction of the crop a forager wants to fill at food of `quality`.
    pub fn load_fraction(&self, quality: f64) -> f64 {
        let f = self.min_load_fraction.clamp(0.0, 1.0);
        f + (1.0 - f) * quality.clamp(0.0, 1.0)
    }

    /// Sugar a typical worker's crop holds, milligrams: the crop volume of
    /// solution at the reference molarity.
    pub fn crop_sugar_capacity_mg(&self) -> f64 {
        self.sugar_mg(self.crop_capacity_ul, self.reference_molarity)
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

    /// Walking-speed multiplier at a temperature: linear from zero at
    /// `speed_t_min_c` to one at the reference, continuing above it.
    pub fn speed_factor(&self, temperature_c: f64) -> f64 {
        let span = (self.speed_t_ref_c - self.speed_t_min_c).max(1e-9);
        ((temperature_c - self.speed_t_min_c) / span).clamp(0.0, 2.5)
    }

    /// Multiplier on pheromone decay rates at a temperature.
    pub fn evaporation_factor(&self, temperature_c: f64) -> f64 {
        q10_factor(self.q10_evaporation, temperature_c, Self::REFERENCE_C)
    }

    /// Multiplier on brood development rate at a temperature.
    pub fn development_factor(&self, temperature_c: f64) -> f64 {
        q10_factor(self.q10_development, temperature_c, Self::REFERENCE_C)
    }

    /// Multiplier on metabolic consumption at a temperature.
    pub fn metabolism_factor(&self, temperature_c: f64) -> f64 {
        q10_factor(self.q10_metabolism, temperature_c, Self::REFERENCE_C)
    }

    /// Heat hazard rate outside the nest at a temperature, per second:
    /// `rate_at_max × exp((T − CTmax) / scale)`, capped at ten times the
    /// rate at the maximum.
    pub fn heat_hazard_per_s(&self, temperature_c: f64) -> f64 {
        let x = (temperature_c - self.critical_thermal_max_c) / self.heat_hazard_scale_c.max(1e-9);
        (self.heat_hazard_at_max_per_s * x.exp()).min(10.0 * self.heat_hazard_at_max_per_s)
    }

    /// Foraging activity window at a temperature: one well inside
    /// `forage_min_c..forage_max_c`, fading smoothly to zero over two
    /// degrees at each edge.
    pub fn activity_factor(&self, temperature_c: f64) -> f64 {
        let ramp = |x: f64| -> f64 {
            let t = (x / 2.0).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        ramp(temperature_c - self.forage_min_c) * ramp(self.forage_max_c - temperature_c)
    }

    /// The species' movement instinct as a behavioral surface at a fixed
    /// temperature of 1: outbound, follow the recruitment trail with
    /// Deneubourg's exponent, head for a remembered site along a remembered
    /// route, avoid alarm, no-entry marks and walls, keep momentum;
    /// inbound, follow the path-integration home vector, the remembered
    /// route, the nest, and (where laid) the outbound trail.
    pub fn instinct(&self) -> BehavioralSurface {
        let n = self.choice_exponent;
        let mut weights = [0.0; FEATURES];
        let out = &mut weights[..BASE_FEATURES];
        out[F_TRAIL] = n;
        out[F_HOME] = 0.0;
        out[F_TERRITORY] = 0.3;
        out[F_NO_ENTRY] = -2.0;
        out[F_ALARM] = -2.0;
        out[F_ODOUR] = 1.5;
        out[F_FOOD] = 4.0;
        out[F_NEST] = -1.0;
        out[F_HEADING] = 1.0;
        out[F_HOME_VECTOR] = -0.3;
        out[F_SITE] = 2.0;
        out[F_ROUTE] = self.route_weight;
        out[F_RECENT] = -0.5;
        out[F_CROWD] = -self.crowding_avoidance;
        out[F_WALL] = -1.0;
        let inb = &mut weights[BASE_FEATURES..];
        inb[F_TRAIL] = self.inbound_trail_factor * n;
        inb[F_HOME] = n;
        inb[F_TERRITORY] = 0.5;
        inb[F_NO_ENTRY] = 0.0;
        inb[F_ALARM] = -2.0;
        inb[F_ODOUR] = 0.0;
        inb[F_FOOD] = 0.0;
        inb[F_NEST] = 4.0;
        inb[F_HEADING] = 0.8;
        inb[F_HOME_VECTOR] = 3.0;
        inb[F_SITE] = 0.0;
        inb[F_ROUTE] = self.route_weight;
        inb[F_RECENT] = -0.5;
        inb[F_CROWD] = -self.crowding_avoidance;
        inb[F_WALL] = -1.0;
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
            assert!((s.speed_factor(s.speed_t_ref_c) - 1.0).abs() < 1e-12);
            assert!(
                (s.activity_factor(0.5 * (s.forage_min_c + s.forage_max_c)) - 1.0).abs() < 1e-12
            );
        }
        assert!(!Species::cataglyphis().pheromones()[Pheromone::Trail.index()].is_active());
        assert!(Species::pharaoh().pheromones()[Pheromone::NoEntry.index()].is_active());
        let instinct = Species::lasius_niger().instinct();
        assert_eq!(instinct.weights[F_TRAIL], 2.0);
        assert_eq!(instinct.weights[BASE_FEATURES + F_HOME_VECTOR], 3.0);
        assert_eq!(instinct.entropy, EntropyControl::fixed(1.0));
    }

    #[test]
    fn quality_modulates_laying_feeding_and_loads() {
        let s = Species::lasius_niger();
        assert!(s.lay_probability(0.3) < s.lay_probability(1.0));
        assert!(
            (s.lay_probability(0.3) - 0.45).abs() < 1e-9,
            "half at the half quality"
        );
        assert!(s.crop_sugar_capacity_mg() > 0.0);
        assert!(
            s.heat_hazard_per_s(s.critical_thermal_max_c)
                > s.heat_hazard_per_s(s.critical_thermal_max_c - 5.0)
        );
        assert!(s.heat_hazard_per_s(s.critical_thermal_max_c - 15.0) < s.forager_hazard_per_s);
        assert!(s.site_fidelity(1.0) > 0.95 && s.site_fidelity(0.1) < 0.4);
        assert!((s.site_fidelity(0.15) - 0.5).abs() < 1e-9);
        assert!((s.load_fraction(0.0) - 0.3).abs() < 1e-12 && s.load_fraction(1.0) == 1.0);
        assert!(
            s.intake_rate(0.1) > s.intake_rate(1.0),
            "viscous solutions are drunk slowly"
        );
        assert!((s.intake_rate(0.7) - 0.5 * s.intake_max_ul_s).abs() < 1e-12);
        assert!((s.sugar_mg(1.0, 1.0) - SUGAR_MG_PER_UL_PER_MOLAR).abs() < 1e-12);
        assert_eq!(s.quality(2.0), 1.0);
        assert!((s.quality(0.25) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn temperature_responses() {
        let s = Species::lasius_niger();
        assert_eq!(s.speed_factor(2.0), 0.0);
        assert!((s.speed_factor(13.5) - 0.5).abs() < 1e-12);
        assert!((s.evaporation_factor(32.0) - 2.0).abs() < 1e-12);
        assert!((s.development_factor(12.0) - 1.0 / 2.5).abs() < 1e-12);
        assert!((s.metabolism_factor(22.0) - 1.0).abs() < 1e-12);
        assert_eq!(s.activity_factor(5.0), 0.0);
        assert_eq!(s.activity_factor(45.0), 0.0);
        assert!(s.activity_factor(11.0) > 0.0 && s.activity_factor(11.0) < 1.0);
        assert_eq!(s.activity_factor(20.0), 1.0);
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
        assert_eq!(
            c.consumption_mg_per_ant_per_s,
            s.consumption_mg_per_ant_per_s
        );
        assert!((c.maturation_s * 60.0 - s.maturation_s).abs() < 1e-3);
        assert!((c.larva_s * 60.0 - s.larva_s).abs() < 1e-3);
        assert!((c.nursing_rate_mg_s / 60.0 - s.nursing_rate_mg_s).abs() < 1e-12);
    }
}
