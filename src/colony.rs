//! The colony simulation: world, nest, ants, and the hierarchy that governs
//! their movement.
//!
//! Each tick every living ant acts according to its activity. Inside the
//! nest it rests, nurses larvae, or unloads by trophallaxis, and decides
//! whether to leave by a response threshold on the colony's hunger and its
//! own excitation from contacts with successful foragers. Outside, it walks
//! continuously: it scores sixteen candidate headings through the effective
//! policy of its category, lays that ring out as a landscape, deforms it
//! with the entropy budget, and selects a heading (a global draw or a
//! crawling sucker). It lays pheromone per centimetre walked, keeps a
//! path-integration home vector and a route memory of familiar places,
//! drinks at food at a viscosity-limited rate, returns, searches when its
//! estimate runs out, and may starve or be taken by a predator. The nest
//! consumes sugar, the queen lays eggs when fed, brood passes through egg,
//! larval and pupal stages, and temperature scales walking, evaporation,
//! development and metabolism.

use crate::ant::{observe, Activity, Ant, AntId, Mode, SearchTarget, Site, Traits, FEATURES};
use crate::entropy::{entropy, Tempering};
use crate::geometry::{angle_of, Point, Position};
use crate::hierarchy::{EffectivePolicy, Hierarchy, HierarchySpec, NodeId};
use crate::landscape::{ring_heading, turn_magnitude, EntropyLedger, Landscape, Sucker, RING};
use crate::pheromone::Pheromone;
use crate::rng::Rng;
use crate::species::Species;
use crate::surface::{BehavioralSurface, SurfaceError, PARAM_LEN};
use crate::world::{Nutrient, World, WorldConfig};

/// How events translate into the scalar reward learners optimise.
#[derive(Clone, Debug, PartialEq)]
pub struct RewardSpec {
    /// Reward per milligram of sugar delivered into the nest.
    pub sugar_mg: f64,
    /// Reward per milligram of protein delivered into the nest.
    pub protein_mg: f64,
    /// Reward per feeding visit at a source.
    pub food_picked: f64,
    /// Reward (normally negative) per dead worker.
    pub death: f64,
    /// Reward per worker that emerges from the brood.
    pub birth: f64,
}

impl Default for RewardSpec {
    fn default() -> Self {
        RewardSpec {
            sugar_mg: 5.0,
            protein_mg: 5.0,
            food_picked: 0.1,
            death: -0.5,
            birth: 0.0,
        }
    }
}

/// How a heading is drawn from the deformed landscape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    /// A global draw from the tempered distribution.
    Softmax,
    /// A [`Sucker`] crawling from straight ahead. `reach` is the base number
    /// of steps, scaled by the effective surface's reach parameter.
    Sucker {
        /// Base reach in proposal steps.
        reach: usize,
    },
}

/// Parameters of the geometric entropy channels.
#[derive(Clone, Debug, PartialEq)]
pub struct GeometryConfig {
    /// Smoothing scale, in ring steps, at full smoothing share and full
    /// entropy budget.
    pub smooth_max: f64,
    /// Number of Fourier modes in the roughening field.
    pub rough_modes: usize,
    /// Upper bound on the sucker's reach.
    pub max_reach: usize,
    /// Whether to keep the entropy ledger (a little extra work per decision).
    pub ledger: bool,
    /// Extra random fields drawn per decision, when roughening is active and
    /// the ledger is on, to estimate the entropy the field injects between
    /// decisions (`Contributions::field`). Consumes randomness, so a run with
    /// the ledger on differs from one with it off whenever roughening is on.
    pub field_samples: usize,
}

impl Default for GeometryConfig {
    fn default() -> Self {
        GeometryConfig {
            smooth_max: 3.0,
            rough_modes: 3,
            max_reach: 64,
            ledger: true,
            field_samples: 4,
        }
    }
}

/// Ambient conditions.
#[derive(Clone, Debug, PartialEq)]
pub struct Environment {
    /// Mean air temperature, °C.
    pub temperature_c: f64,
    /// Half the day–night temperature swing, °C (0 for constant weather).
    pub diurnal_amplitude_c: f64,
    /// Length of a day, seconds.
    pub day_length_s: f64,
    /// Time of the daily minimum, seconds after the start.
    pub coldest_at_s: f64,
}

impl Default for Environment {
    fn default() -> Self {
        Environment {
            temperature_c: Species::REFERENCE_C,
            diurnal_amplitude_c: 0.0,
            day_length_s: 24.0 * 3600.0,
            coldest_at_s: 0.0,
        }
    }
}

impl Environment {
    /// Temperature at a simulated time.
    pub fn temperature(&self, time_s: f64) -> f64 {
        if self.diurnal_amplitude_c == 0.0 {
            return self.temperature_c;
        }
        let phase =
            std::f64::consts::TAU * (time_s - self.coldest_at_s) / self.day_length_s.max(1e-9);
        self.temperature_c - self.diurnal_amplitude_c * phase.cos()
    }
}

/// Nest and colony-level parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct NestConfig {
    /// Sugar store capacity per worker, milligrams.
    pub store_capacity_mg_per_ant: f64,
    /// Initial fill of the store as a fraction of capacity.
    pub initial_satiation: f64,
    /// Initial brood items per worker, spread over the three stages.
    pub initial_brood_per_ant: f64,
    /// Maximum brood items per worker (the queen stops laying beyond it).
    pub max_brood_per_ant: f64,
    /// Whether a queen lays eggs.
    pub queen: bool,
    /// Population cap for emergence.
    pub max_ants: usize,
    /// Whether workers can starve or be taken by predators.
    pub mortality: bool,
    /// Age structure of the founding workers: ages are drawn uniformly up
    /// to this many seconds so the colony starts with nurses and foragers.
    pub initial_age_spread_s: f64,
    /// Interval between activity snapshots, seconds (`0` disables the log).
    pub log_every_s: f64,
}

impl Default for NestConfig {
    fn default() -> Self {
        NestConfig {
            store_capacity_mg_per_ant: 0.4,
            initial_satiation: 0.3,
            initial_brood_per_ant: 0.5,
            max_brood_per_ant: 1.0,
            queen: true,
            max_ants: 400,
            mortality: true,
            initial_age_spread_s: 6.0 * 24.0 * 3600.0,
            log_every_s: 60.0,
        }
    }
}

/// Complete simulation configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct SimConfig {
    /// World layout and physical scale.
    pub world: WorldConfig,
    /// Shape of the control hierarchy.
    pub hierarchy: HierarchySpec,
    /// Surface installed at the root (normally the species' instinct).
    pub instinct: BehavioralSurface,
    /// Species profile.
    pub species: Species,
    /// Ambient conditions.
    pub environment: Environment,
    /// Initial number of workers.
    pub ants: usize,
    /// Nest parameters.
    pub nest: NestConfig,
    /// Reward definition.
    pub reward: RewardSpec,
    /// Whether to accumulate policy-gradient score sums per node.
    pub trace: bool,
    /// How headings are selected from the landscape.
    pub selection: Selection,
    /// Geometric channel parameters.
    pub geometry: GeometryConfig,
    /// Record the path surface of this ant (see [`Simulation::surface_trace`]).
    pub record_surface: Option<AntId>,
    /// Maximum rows kept in the surface recording.
    pub surface_rows: usize,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig::for_species(Species::lasius_niger())
    }
}

impl SimConfig {
    /// A default configuration for a species, with that species' instinct
    /// at the root.
    pub fn for_species(species: Species) -> Self {
        // Hold the world at the species' reference temperature so that a
        // thermophile is not asked to forage in a temperate room.
        let environment = Environment {
            temperature_c: species.speed_t_ref_c,
            ..Environment::default()
        };
        SimConfig {
            world: WorldConfig::default(),
            hierarchy: HierarchySpec::default(),
            instinct: species.instinct(),
            species,
            environment,
            ants: 60,
            nest: NestConfig::default(),
            reward: RewardSpec::default(),
            trace: false,
            selection: Selection::Softmax,
            geometry: GeometryConfig::default(),
            record_surface: None,
            surface_rows: 256,
        }
    }
}

/// Developmental stage of a brood item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BroodStage {
    /// An egg: needs no food.
    Egg,
    /// A larva: must be fed by nurses to pupate; starves if neglected.
    Larva,
    /// A pupa: needs no food; emerges as a worker.
    Pupa,
}

/// One brood item.
#[derive(Clone, Debug, PartialEq)]
pub struct BroodItem {
    /// Current stage.
    pub stage: BroodStage,
    /// Developmental time accumulated in the current stage, seconds at the
    /// reference temperature.
    pub stage_age_s: f64,
    /// Sugar received as a larva, milligrams.
    pub fed_mg: f64,
    /// Protein received as a larva, milligrams.
    pub protein_mg: f64,
    /// Seconds since a nurse last fed it.
    pub unfed_s: f64,
}

/// The nest interior.
#[derive(Clone, Debug, PartialEq)]
pub struct Nest {
    /// Sugar in store, milligrams.
    pub store_mg: f64,
    /// Store capacity, milligrams.
    pub capacity_mg: f64,
    /// Protein (prey) in store, milligrams.
    pub protein_mg: f64,
    /// Developing brood.
    pub brood: Vec<BroodItem>,
    /// Mean recruitment excitation of the workers inside.
    pub excitation: f64,
    /// Eggs laid so far.
    pub eggs_laid: u64,
    /// Workers that emerged so far.
    pub emerged: u64,
    /// Larvae that starved so far.
    pub larvae_starved: u64,
    /// Tick of the last egg.
    pub last_egg_tick: u64,
}

impl Nest {
    /// Protein the current larvae still need to pupate, milligrams.
    pub fn protein_need(&self, larva_protein_mg: f64) -> f64 {
        self.brood
            .iter()
            .filter(|b| b.stage == BroodStage::Larva)
            .map(|b| (larva_protein_mg - b.protein_mg).max(0.0))
            .sum()
    }

    /// The colony's protein demand, 0..1: the share of what the larvae
    /// still need that the store does not cover.
    pub fn protein_demand(&self, larva_protein_mg: f64) -> f64 {
        let need = self.protein_need(larva_protein_mg);
        if need <= 0.0 {
            0.0
        } else {
            ((need - self.protein_mg) / need).clamp(0.0, 1.0)
        }
    }

    /// Fill of the store, 0 (empty) to 1 (full).
    pub fn satiation(&self) -> f64 {
        if self.capacity_mg <= 0.0 {
            0.0
        } else {
            (self.store_mg / self.capacity_mg).clamp(0.0, 1.0)
        }
    }

    /// `1 - satiation`.
    pub fn hunger(&self) -> f64 {
        1.0 - self.satiation()
    }

    /// Number of brood items in a stage.
    pub fn count(&self, stage: BroodStage) -> usize {
        self.brood.iter().filter(|b| b.stage == stage).count()
    }

    /// Number of larvae.
    pub fn larvae(&self) -> usize {
        self.count(BroodStage::Larva)
    }
}

/// Geometry of the paths ants walked, plus the entropy ledger.
#[derive(Clone, Debug, PartialEq)]
pub struct PathStats {
    /// Steps made.
    pub moves: u64,
    /// Path length walked, cells.
    pub length: f64,
    /// Steps by ring position relative to the previous heading
    /// (0 straight, `RING/2` reverse; see [`crate::landscape::turn_label`]).
    pub turns: [u64; RING],
    /// Entries into a cell the ant had visited recently.
    pub revisits: u64,
    /// Cells entered.
    pub cell_entries: u64,
    /// Completed food-to-nest trips.
    pub trips: u64,
    /// Path length walked on those trips, cells.
    pub trip_length: f64,
    /// Straight-line distance those trips needed, cells.
    pub trip_direct: f64,
    /// Where the decision entropy came from.
    pub ledger: EntropyLedger,
}

impl Default for PathStats {
    fn default() -> Self {
        PathStats {
            moves: 0,
            length: 0.0,
            turns: [0; RING],
            revisits: 0,
            cell_entries: 0,
            trips: 0,
            trip_length: 0.0,
            trip_direct: 0.0,
            ledger: EntropyLedger::default(),
        }
    }
}

impl PathStats {
    fn record_move(&mut self, ring: usize, length: f64) {
        self.moves += 1;
        self.length += length;
        self.turns[ring] += 1;
    }

    fn record_entry(&mut self, revisit: bool) {
        self.cell_entries += 1;
        if revisit {
            self.revisits += 1;
        }
    }

    fn record_trip(&mut self, length: f64, direct: f64) {
        self.trips += 1;
        self.trip_length += length;
        self.trip_direct += direct;
    }

    /// Add another record's counts.
    pub fn merge(&mut self, other: &PathStats) {
        self.moves += other.moves;
        self.length += other.length;
        for (a, b) in self.turns.iter_mut().zip(&other.turns) {
            *a += b;
        }
        self.revisits += other.revisits;
        self.cell_entries += other.cell_entries;
        self.trips += other.trips;
        self.trip_length += other.trip_length;
        self.trip_direct += other.trip_direct;
        self.ledger.merge(&other.ledger);
    }

    /// Entropy (nats) of the distribution of turns: the disorder of the path
    /// itself rather than of the decisions that produced it.
    pub fn turn_entropy(&self) -> f64 {
        if self.moves == 0 {
            return 0.0;
        }
        let probs: Vec<f64> = self
            .turns
            .iter()
            .map(|&t| t as f64 / self.moves as f64)
            .collect();
        entropy(&probs)
    }

    /// Fraction of steps that kept the heading (within half a ring step).
    pub fn straight_rate(&self) -> f64 {
        if self.moves == 0 {
            0.0
        } else {
            self.turns[0] as f64 / self.moves as f64
        }
    }

    /// Fraction of steps that turned by 90° or more.
    pub fn sharp_turn_rate(&self) -> f64 {
        if self.moves == 0 {
            return 0.0;
        }
        let sharp: u64 = (RING / 4..=3 * RING / 4).map(|k| self.turns[k]).sum();
        sharp as f64 / self.moves as f64
    }

    /// Fraction of steps that reversed the heading.
    pub fn reversal_rate(&self) -> f64 {
        if self.moves == 0 {
            0.0
        } else {
            self.turns[RING / 2] as f64 / self.moves as f64
        }
    }

    /// Fraction of cell entries into a recently visited cell.
    pub fn revisit_rate(&self) -> f64 {
        if self.cell_entries == 0 {
            0.0
        } else {
            self.revisits as f64 / self.cell_entries as f64
        }
    }

    /// Straight-line distance divided by path length, averaged over trips
    /// (1 is a perfectly direct return; 0 if there were no trips).
    pub fn trip_efficiency(&self) -> f64 {
        if self.trip_length <= 0.0 {
            0.0
        } else {
            (self.trip_direct / self.trip_length).min(1.0)
        }
    }
}

/// A periodic snapshot of colony state.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    /// Tick of the snapshot.
    pub tick: u64,
    /// Simulated seconds.
    pub time_s: f64,
    /// Air temperature, °C.
    pub temperature_c: f64,
    /// Living workers.
    pub alive: usize,
    /// Workers outside the nest.
    pub outside: usize,
    /// Workers nursing.
    pub nursing: usize,
    /// Workers resting.
    pub resting: usize,
    /// Sugar in store, milligrams.
    pub store_mg: f64,
    /// Store fill.
    pub satiation: f64,
    /// Mean recruitment excitation inside.
    pub excitation: f64,
    /// Eggs, larvae and pupae.
    pub brood: (usize, usize, usize),
    /// Loads delivered so far.
    pub delivered: u64,
    /// Total recruitment trail on the ground.
    pub trail_total: f64,
    /// Crossings of every counter so far.
    pub counters: Vec<u64>,
}

/// Running counters of a simulation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stats {
    /// Ticks simulated since the last reset.
    pub ticks: u64,
    /// Simulated seconds since the last reset.
    pub time_s: f64,
    /// Loads delivered into the nest.
    pub food_delivered: u64,
    /// Sugar delivered, milligrams.
    pub sugar_delivered_mg: f64,
    /// Feeding visits at sources.
    pub food_picked: u64,
    /// Returns from food on which the forager laid trail.
    pub recruiting_trips: u64,
    /// Solution drunk at sources, microlitres.
    pub food_collected_ul: f64,
    /// Prey pieces cut at sources.
    pub prey_picked: u64,
    /// Protein delivered, milligrams.
    pub protein_delivered_mg: f64,
    /// Outbound trips abandoned without food.
    pub failed_trips: u64,
    /// Workers that died.
    pub deaths: u64,
    /// Of which taken by predators.
    pub deaths_predation: u64,
    /// Of which starved.
    pub deaths_starvation: u64,
    /// Eggs laid.
    pub eggs: u64,
    /// Workers that emerged.
    pub births: u64,
    /// Larvae that starved.
    pub larvae_starved: u64,
    /// Total reward.
    pub reward: f64,
    /// Number of movement decisions made.
    pub decisions: u64,
    /// Sum of the entropies met by tempering (nats).
    pub entropy_sum: f64,
    /// Sum of entropies of the distributions headings were actually drawn
    /// from (equal to `entropy_sum` under [`Selection::Softmax`]).
    pub selected_entropy_sum: f64,
    /// Ant-ticks spent in each activity, indexed by [`Activity::index`].
    pub activity_ticks: [u64; Activity::COUNT],
    /// Reward attributed to each node (every event credits the whole path).
    pub reward_by_node: Vec<f64>,
    /// Deliveries attributed to each node.
    pub delivered_by_node: Vec<u64>,
    /// Path geometry and entropy ledger for the whole colony.
    pub path: PathStats,
    /// Path geometry and entropy ledger per node.
    pub path_by_node: Vec<PathStats>,
    /// Living workers at the end of the last tick.
    pub alive: usize,
    /// Workers outside at the end of the last tick.
    pub outside: usize,
    /// Sugar in store at the end of the last tick, milligrams.
    pub store_mg: f64,
    /// Periodic snapshots.
    pub log: Vec<Snapshot>,
}

impl Stats {
    fn new(nodes: usize) -> Self {
        Stats {
            reward_by_node: vec![0.0; nodes],
            delivered_by_node: vec![0; nodes],
            path_by_node: vec![PathStats::default(); nodes],
            ..Stats::default()
        }
    }

    /// Mean tempered entropy per decision.
    pub fn mean_entropy(&self) -> f64 {
        if self.decisions == 0 {
            0.0
        } else {
            self.entropy_sum / self.decisions as f64
        }
    }

    /// Mean entropy of the distributions headings were drawn from.
    pub fn mean_selected_entropy(&self) -> f64 {
        if self.decisions == 0 {
            0.0
        } else {
            self.selected_entropy_sum / self.decisions as f64
        }
    }

    /// Fraction of ant-ticks spent in an activity.
    pub fn activity_fraction(&self, activity: Activity) -> f64 {
        let total: u64 = self.activity_ticks.iter().sum();
        if total == 0 {
            0.0
        } else {
            self.activity_ticks[activity.index()] as f64 / total as f64
        }
    }

    /// Fraction of ant-ticks spent outside the nest.
    pub fn foraging_fraction(&self) -> f64 {
        Activity::ALL
            .iter()
            .filter(|a| a.is_foraging())
            .map(|a| self.activity_fraction(*a))
            .sum()
    }
}

/// Policy-gradient bookkeeping: for every node, the sum over decisions made
/// beneath it of `∇ log π(a)` with respect to that node's parameters.
///
/// The gradient is that of the tempered distribution, treating the solved
/// temperature as a constant; under [`Selection::Sucker`] it is an
/// approximation. Only the weight entries are non-zero.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Trace {
    /// Score-function sums, one `PARAM_LEN` vector per node.
    pub score_by_node: Vec<Vec<f64>>,
    /// Decisions made beneath each node.
    pub decisions_by_node: Vec<u64>,
}

impl Trace {
    fn new(nodes: usize) -> Self {
        Trace {
            score_by_node: vec![vec![0.0; PARAM_LEN]; nodes],
            decisions_by_node: vec![0; nodes],
        }
    }
}

/// One decision of the recorded ant: its landscape at every stage, laid out
/// egocentrically (index 0 straight ahead).
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceRow {
    /// Tick of the decision.
    pub tick: u64,
    /// Where the ant stood.
    pub position: Point,
    /// Its heading, radians.
    pub heading: f64,
    /// Whether it carried food.
    pub carrying: bool,
    /// Takeable ring positions.
    pub valid: [bool; RING],
    /// The deterministic information: scores from the effective surface.
    pub base: [f64; RING],
    /// Scores after smoothing and roughening.
    pub deformed: [f64; RING],
    /// The tempered distribution.
    pub probs: [f64; RING],
    /// The distribution the heading was actually drawn from.
    pub selected: [f64; RING],
    /// The chosen ring position.
    pub chosen: usize,
    /// The sucker's trail (empty under [`Selection::Softmax`]).
    pub walk: Vec<u8>,
    /// Solved temperature.
    pub temperature: f64,
}

/// Why a worker died.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cause {
    Predation,
    Starvation,
}

/// A running colony.
#[derive(Clone, Debug)]
pub struct Simulation {
    config: SimConfig,
    species: Species,
    world: World,
    ants: Vec<Ant>,
    hierarchy: Hierarchy,
    rng: Rng,
    policies: Vec<EffectivePolicy>,
    leaf_paths: Vec<Vec<NodeId>>,
    dirty: bool,
    stats: Stats,
    trace: Trace,
    surface: Vec<SurfaceRow>,
    nest: Nest,
    nest_cells: Vec<Position>,
    alive: usize,
    outside: usize,
    next_leaf: usize,
    tick: u64,
    tick_s: f64,
    /// Persistence length of the heading in cells.
    persistence_cells: f64,
    decision_prob: f64,
    hazard_per_tick: f64,
    excitation_retention: f64,
    log_every_ticks: u64,
    temperature_c: f64,
    speed_factor: f64,
    activity_factor: f64,
    metabolism_factor: f64,
    development_factor: f64,
}

impl Simulation {
    /// Smallest step an ant takes; movement credit accumulates below it.
    const MIN_STEP: f64 = 0.25;

    /// Upper bound on steps per tick, for very fast species or large ticks.
    const MAX_STEPS_PER_TICK: usize = 8;

    /// Build a colony from a config, with the hierarchy the config describes.
    pub fn new(config: SimConfig, seed: u64) -> Self {
        let hierarchy = Hierarchy::from_spec(&config.hierarchy, config.instinct.clone());
        Self::with_hierarchy(config, hierarchy, seed)
    }

    /// Build a colony governed by an explicit hierarchy (its spec in the
    /// config is ignored).
    pub fn with_hierarchy(config: SimConfig, hierarchy: Hierarchy, seed: u64) -> Self {
        let mut rng = Rng::seed_from_u64(seed);
        let species = config.species.clone();
        let mut world_config = config.world.clone();
        if world_config.pheromones.is_none() {
            world_config.pheromones = Some(species.pheromones());
        }
        let world = World::new(world_config, &mut rng);
        let tick_s = world.tick_s();
        let cell_cm = world.cell_cm();
        let nodes = hierarchy.len();
        let leaf_paths = hierarchy
            .leaves()
            .iter()
            .map(|&l| hierarchy.path(l).to_vec())
            .collect();
        let capacity = config.nest.store_capacity_mg_per_ant * config.ants.max(1) as f64;
        let nest = Nest {
            store_mg: capacity * config.nest.initial_satiation.clamp(0.0, 1.0),
            capacity_mg: capacity,
            protein_mg: 0.0,
            brood: Vec::new(),
            excitation: 0.0,
            eggs_laid: 0,
            emerged: 0,
            larvae_starved: 0,
            last_egg_tick: 0,
        };
        let nest_cells = world.nest_cells();
        let temperature_c = config.environment.temperature(0.0);
        let mut sim = Simulation {
            policies: hierarchy.compile(),
            leaf_paths,
            hierarchy,
            world,
            ants: Vec::new(),
            rng,
            dirty: false,
            stats: Stats::new(nodes),
            trace: Trace::new(nodes),
            surface: Vec::new(),
            nest,
            nest_cells,
            alive: 0,
            outside: 0,
            next_leaf: 0,
            tick: 0,
            tick_s,
            persistence_cells: species.heading_persistence_cm / cell_cm.max(1e-9),
            decision_prob: (tick_s / species.decision_interval_s.max(1e-9)).min(1.0),
            hazard_per_tick: 1.0 - (-species.forager_hazard_per_s * tick_s).exp(),
            excitation_retention: 0.5f64.powf(tick_s / species.excitation_half_life_s.max(1e-9)),
            log_every_ticks: if config.nest.log_every_s > 0.0 {
                (config.nest.log_every_s / tick_s).round().max(1.0) as u64
            } else {
                0
            },
            temperature_c,
            speed_factor: species.speed_factor(temperature_c),
            activity_factor: species.activity_factor(temperature_c),
            metabolism_factor: species.metabolism_factor(temperature_c),
            development_factor: species.development_factor(temperature_c),
            species,
            config,
        };
        sim.world
            .set_evaporation_factor(sim.species.evaporation_factor(temperature_c));
        for _ in 0..sim.config.ants {
            let age_s = sim
                .rng
                .range(0.0, sim.config.nest.initial_age_spread_s.max(0.0));
            let id = sim.spawn_ant();
            sim.ants[id].age = (age_s / tick_s) as u64;
        }
        let brood_items =
            (sim.config.nest.initial_brood_per_ant * sim.config.ants as f64).round() as usize;
        for _ in 0..brood_items {
            let item = sim.random_brood_item();
            sim.nest.brood.push(item);
        }
        sim.stats.alive = sim.alive;
        sim.stats.store_mg = sim.nest.store_mg;
        sim
    }

    /// A brood item at a uniformly random point of its development.
    fn random_brood_item(&mut self) -> BroodItem {
        let s = &self.species;
        let total = s.egg_s + s.larva_s + s.pupa_s;
        let t = self.rng.range(0.0, total.max(1e-9));
        if t < s.egg_s {
            BroodItem {
                stage: BroodStage::Egg,
                stage_age_s: t,
                fed_mg: 0.0,
                protein_mg: 0.0,
                unfed_s: 0.0,
            }
        } else if t < s.egg_s + s.larva_s {
            let age = t - s.egg_s;
            BroodItem {
                stage: BroodStage::Larva,
                stage_age_s: age,
                fed_mg: s.larva_food_mg * age / s.larva_s.max(1e-9),
                protein_mg: s.larva_protein_mg * age / s.larva_s.max(1e-9),
                unfed_s: 0.0,
            }
        } else {
            BroodItem {
                stage: BroodStage::Pupa,
                stage_age_s: t - s.egg_s - s.larva_s,
                fed_mg: s.larva_food_mg,
                protein_mg: s.larva_protein_mg,
                unfed_s: 0.0,
            }
        }
    }

    fn seconds_to_ticks(&self, seconds: f64) -> u32 {
        (seconds / self.tick_s).round().max(1.0) as u32
    }

    fn spawn_ant(&mut self) -> usize {
        let id = self.ants.len();
        let heading = self.rng.range(-std::f64::consts::PI, std::f64::consts::PI);
        let leaf = self.next_leaf;
        self.next_leaf = (self.next_leaf + 1) % self.leaf_paths.len();
        let traits = Traits::draw(
            &self.species,
            self.world.cell_cm(),
            self.tick_s,
            &mut self.rng,
        );
        let ant = Ant::new(
            id,
            self.world.nest(),
            heading,
            leaf,
            traits,
            self.species.starvation_s,
        );
        self.ants.push(ant);
        self.alive += 1;
        id
    }

    /// Configuration.
    pub fn config(&self) -> &SimConfig {
        &self.config
    }

    /// Species in use.
    pub fn species(&self) -> &Species {
        &self.species
    }

    /// The world.
    pub fn world(&self) -> &World {
        &self.world
    }

    /// All ants, living and dead.
    pub fn ants(&self) -> &[Ant] {
        &self.ants
    }

    /// Living ants.
    pub fn living(&self) -> impl Iterator<Item = &Ant> {
        self.ants.iter().filter(|a| a.alive)
    }

    /// The nest interior.
    pub fn nest(&self) -> &Nest {
        &self.nest
    }

    /// Mutable nest interior (to feed or starve the colony by hand).
    pub fn nest_mut(&mut self) -> &mut Nest {
        &mut self.nest
    }

    /// Ambient conditions.
    pub fn environment(&self) -> &Environment {
        &self.config.environment
    }

    /// Current air temperature, °C.
    pub fn temperature(&self) -> f64 {
        self.temperature_c
    }

    /// The control hierarchy.
    pub fn hierarchy(&self) -> &Hierarchy {
        &self.hierarchy
    }

    /// Mutable access to the hierarchy; policies are recompiled on the next tick.
    pub fn hierarchy_mut(&mut self) -> &mut Hierarchy {
        self.dirty = true;
        &mut self.hierarchy
    }

    /// Replace the whole hierarchy's parameters.
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), SurfaceError> {
        self.hierarchy.set_params(params)?;
        self.dirty = true;
        Ok(())
    }

    /// The hierarchy's flat parameter vector.
    pub fn params(&self) -> Vec<f64> {
        self.hierarchy.params()
    }

    /// Recompute the per-leaf effective policies from the hierarchy.
    pub fn recompile(&mut self) {
        self.policies = self.hierarchy.compile();
        self.dirty = false;
    }

    /// Effective policy of each leaf (as of the last compile).
    pub fn policies(&self) -> &[EffectivePolicy] {
        &self.policies
    }

    /// Counters since the last reset.
    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// Policy-gradient trace since the last reset (all zeros unless tracing).
    pub fn trace(&self) -> &Trace {
        &self.trace
    }

    /// The recorded path surface of the ant named by
    /// [`SimConfig::record_surface`], oldest row first.
    pub fn surface_trace(&self) -> &[SurfaceRow] {
        &self.surface
    }

    /// Choose which ant's path surfaces are recorded from now on (`None`
    /// stops recording); rows already recorded are kept.
    pub fn set_record_surface(&mut self, ant: Option<AntId>) {
        self.config.record_surface = ant;
    }

    /// Forget the recorded path surface.
    pub fn clear_surface_trace(&mut self) {
        self.surface.clear();
    }

    /// Zero the counters and trace without touching the colony state.
    pub fn reset_stats(&mut self) {
        let nodes = self.hierarchy.len();
        self.stats = Stats::new(nodes);
        self.stats.alive = self.alive;
        self.stats.outside = self.outside;
        self.stats.store_mg = self.nest.store_mg;
        self.trace = Trace::new(nodes);
    }

    /// Ticks simulated since construction.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Simulated seconds since construction.
    pub fn time_s(&self) -> f64 {
        self.tick as f64 * self.tick_s
    }

    /// Duration of a tick in seconds.
    pub fn tick_s(&self) -> f64 {
        self.tick_s
    }

    /// Number of living workers.
    pub fn alive(&self) -> usize {
        self.alive
    }

    /// Number of workers outside the nest.
    pub fn outside(&self) -> usize {
        self.outside
    }

    /// The sucker reach an effective policy implies under the current config
    /// (`None` under [`Selection::Softmax`]).
    pub fn effective_reach(&self, policy: &EffectivePolicy) -> Option<usize> {
        match self.config.selection {
            Selection::Softmax => None,
            Selection::Sucker { reach } => Some(
                (reach as f64 * policy.deformation.reach_scale())
                    .round()
                    .clamp(0.0, self.config.geometry.max_reach as f64) as usize,
            ),
        }
    }

    /// Division of labour among living workers: one minus the mean
    /// (time-weighted) normalised entropy of each worker's split between
    /// foraging and nursing. Zero when everybody does both equally, one when
    /// everybody specialises.
    pub fn division_of_labor(&self) -> f64 {
        let mut weighted = 0.0;
        let mut total = 0.0;
        for ant in self.living() {
            let f = ant.time_foraging as f64;
            let n = ant.time_nursing as f64;
            let t = f + n;
            if t <= 0.0 {
                continue;
            }
            let p = f / t;
            let h = entropy(&[p, 1.0 - p]) / std::f64::consts::LN_2;
            weighted += t * h;
            total += t;
        }
        if total <= 0.0 {
            0.0
        } else {
            1.0 - weighted / total
        }
    }

    /// Advance the colony by one tick.
    pub fn step(&mut self) {
        if self.dirty {
            self.recompile();
        }
        self.update_environment();
        for i in 0..self.ants.len() {
            if self.ants[i].alive {
                self.step_ant(i);
            }
        }
        self.world.step_pheromones();
        self.world.step_food();
        self.nest_step();
        self.tick += 1;
        self.stats.ticks += 1;
        self.stats.time_s += self.tick_s;
        self.stats.alive = self.alive;
        self.stats.outside = self.outside;
        self.stats.store_mg = self.nest.store_mg;
        if self.log_every_ticks > 0 && self.tick.is_multiple_of(self.log_every_ticks) {
            self.snapshot();
        }
    }

    /// Advance by `steps` ticks and return the counters.
    pub fn run(&mut self, steps: usize) -> &Stats {
        for _ in 0..steps {
            self.step();
        }
        &self.stats
    }

    /// Advance by the number of ticks closest to `seconds`.
    pub fn run_seconds(&mut self, seconds: f64) -> &Stats {
        let steps = (seconds / self.tick_s).round().max(0.0) as usize;
        self.run(steps)
    }

    fn update_environment(&mut self) {
        let t = self.config.environment.temperature(self.time_s());
        if (t - self.temperature_c).abs() < 1e-9 && self.tick > 0 {
            return;
        }
        self.temperature_c = t;
        self.speed_factor = self.species.speed_factor(t);
        self.activity_factor = self.species.activity_factor(t);
        self.metabolism_factor = self.species.metabolism_factor(t);
        self.development_factor = self.species.development_factor(t);
        self.world
            .set_evaporation_factor(self.species.evaporation_factor(t));
    }

    fn snapshot(&mut self) {
        let mut nursing = 0;
        let mut resting = 0;
        for a in self.living() {
            match a.activity {
                Activity::Nursing => nursing += 1,
                Activity::Resting => resting += 1,
                _ => {}
            }
        }
        self.stats.log.push(Snapshot {
            tick: self.tick,
            time_s: self.time_s(),
            temperature_c: self.temperature_c,
            alive: self.alive,
            outside: self.outside,
            nursing,
            resting,
            store_mg: self.nest.store_mg,
            satiation: self.nest.satiation(),
            excitation: self.nest.excitation,
            brood: (
                self.nest.count(BroodStage::Egg),
                self.nest.count(BroodStage::Larva),
                self.nest.count(BroodStage::Pupa),
            ),
            delivered: self.stats.food_delivered,
            trail_total: self.world.total_pheromone(Pheromone::Trail),
            counters: self.world.counters().iter().map(|c| c.crossings).collect(),
        });
    }

    // ---------------------------------------------------------------
    // Per-ant behaviour
    // ---------------------------------------------------------------

    fn step_ant(&mut self, i: usize) {
        let activity = self.ants[i].activity;
        self.stats.activity_ticks[activity.index()] += 1;
        self.ants[i].age += 1;
        self.ants[i].excitement *= self.excitation_retention;
        self.reinforce(i, activity);
        match activity {
            Activity::Resting => self.rest(i),
            Activity::Nursing => self.nurse(i),
            Activity::Unloading => self.unload(i),
            Activity::Feeding => self.feed(i),
            Activity::Outbound | Activity::Inbound | Activity::Searching => self.walk(i),
        }
        if self.ants[i].alive {
            self.metabolize(i);
        }
    }

    fn inside_count(&self) -> usize {
        self.alive.saturating_sub(self.outside)
    }

    fn nurses_now(&self) -> usize {
        self.ants
            .iter()
            .filter(|a| a.alive && a.activity == Activity::Nursing)
            .count()
    }

    /// Response-threshold reinforcement: the threshold of the task being
    /// performed falls, the others rise, within bounds.
    fn reinforce(&mut self, i: usize, activity: Activity) {
        let learn = (-self.tick_s / self.species.threshold_learning_s.max(1e-9)).exp();
        let forget = (self.tick_s / self.species.threshold_forgetting_s.max(1e-9)).exp();
        let (lo, hi) = self.species.threshold_bounds;
        let f_bounds = (
            lo * self.species.threshold_median,
            hi * self.species.threshold_median,
        );
        let n_bounds = (
            lo * self.species.nursing_threshold_median,
            hi * self.species.nursing_threshold_median,
        );
        let a = &mut self.ants[i];
        let (f_factor, n_factor) = match activity {
            Activity::Nursing => (forget, learn),
            Activity::Resting => (forget, forget),
            _ => (learn, forget),
        };
        a.traits.foraging_threshold =
            (a.traits.foraging_threshold * f_factor).clamp(f_bounds.0, f_bounds.1);
        a.traits.nursing_threshold =
            (a.traits.nursing_threshold * n_factor).clamp(n_bounds.0, n_bounds.1);
    }

    fn rest(&mut self, i: usize) {
        let hunger = self.nest.hunger();
        let (site_bonus, threshold, nursing_threshold, excitement) = {
            let a = &self.ants[i];
            let age_s = a.age as f64 * self.tick_s;
            (
                a.site
                    .map(|s| self.species.reforage_bonus * s.quality)
                    .unwrap_or(0.0),
                self.species
                    .threshold_at_age(a.traits.foraging_threshold, age_s),
                a.traits.nursing_threshold,
                a.excitement,
            )
        };
        // A known source is a reason to go again only while the colony
        // can take the food: satiated nestmates refuse to unload foragers,
        // which stops re-foraging (Mailleux, Detrain & Deneubourg 2006).
        let protein_demand = self.nest.protein_demand(self.species.larva_protein_mg);
        let stimulus = self.species.hunger_gain * hunger
            + self.species.protein_demand_gain * protein_demand
            + excitement
            + site_bonus * hunger.max(protein_demand);
        let p_forage =
            self.species.response(stimulus, threshold) * self.decision_prob * self.activity_factor;
        let larvae = self.nest.larvae() as f64;
        let demand = if larvae <= 0.0 {
            0.0
        } else {
            let nurses = self.nurses_now() as f64;
            (larvae / self.species.brood_per_nurse.max(1e-9)) / (nurses + 1.0)
        };
        let p_nurse = self.species.response(demand, nursing_threshold) * self.decision_prob;
        let u = self.rng.next_f64();
        if u < p_forage {
            self.depart(i);
        } else if u < p_forage + p_nurse {
            let bout = self.seconds_to_ticks(self.species.nursing_bout_s);
            let a = &mut self.ants[i];
            a.activity = Activity::Nursing;
            a.timer = bout;
        }
    }

    fn depart(&mut self, i: usize) {
        let exit = self.nest_cells[self.rng.below(self.nest_cells.len().max(1))];
        let heading = self.rng.range(-std::f64::consts::PI, std::f64::consts::PI);
        let jitter = (self.rng.range(-0.4, 0.4), self.rng.range(-0.4, 0.4));
        let outbound_laying = self.species.outbound_laying;
        let exploratory_laying = self.species.exploratory_laying;
        // A protein trip or a sugar trip, from the colony's demand.
        let demand = self.nest.protein_demand(self.species.larva_protein_mg);
        let base = self.species.protein_acceptance_base;
        let accepts_prey = self.rng.chance(base + (1.0 - base) * demand);
        let a = &mut self.ants[i];
        a.activity = Activity::Outbound;
        a.accepts_prey = accepts_prey;
        a.item_mg = 0.0;
        let c = Point::center_of(exit);
        a.position = Point::new(c.x + jitter.0, c.y + jitter.1);
        a.heading = heading;
        a.reset_home_vector();
        a.clear_memory();
        a.steps_since_nest = 0;
        a.search_steps = 0;
        a.move_credit = 0.0;
        a.laying = if (outbound_laying && a.site.is_some()) || exploratory_laying {
            a.lay_strength = if a.site.is_some() { 0.5 } else { 0.25 };
            Some(Pheromone::Trail)
        } else {
            None
        };
        if let Some(c) = self.world.cell_mut(exit) {
            c.occupancy = c.occupancy.saturating_add(1);
        }
        self.outside += 1;
    }

    fn nurse(&mut self, i: usize) {
        let rate = self.species.nursing_rate_mg_s * self.tick_s;
        let need_cap = self.species.larva_food_mg;
        let available = rate.min(self.nest.store_mg);
        if available > 0.0 {
            if let Some(hungriest) = self
                .nest
                .brood
                .iter_mut()
                .filter(|b| b.stage == BroodStage::Larva)
                .min_by(|a, b| {
                    a.fed_mg
                        .partial_cmp(&b.fed_mg)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            {
                let amount = available.min((need_cap - hungriest.fed_mg).max(0.0));
                hungriest.fed_mg += amount;
                hungriest.unfed_s = 0.0;
                self.nest.store_mg -= amount;
            }
        }
        // Protein goes to the larva that still needs the most of it.
        let protein_available = rate.min(self.nest.protein_mg);
        if protein_available > 0.0 {
            let need = self.species.larva_protein_mg;
            if let Some(neediest) = self
                .nest
                .brood
                .iter_mut()
                .filter(|b| b.stage == BroodStage::Larva && b.protein_mg < need)
                .min_by(|a, b| {
                    a.protein_mg
                        .partial_cmp(&b.protein_mg)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            {
                let amount = protein_available.min(need - neediest.protein_mg);
                neediest.protein_mg += amount;
                neediest.unfed_s = 0.0;
                self.nest.protein_mg -= amount;
            }
        }
        let no_larvae = self.nest.larvae() == 0;
        let a = &mut self.ants[i];
        a.time_nursing += 1;
        a.timer = a.timer.saturating_sub(1);
        if a.timer == 0 || no_larvae {
            a.activity = Activity::Resting;
        }
    }

    fn unload(&mut self, i: usize) {
        let a = &mut self.ants[i];
        a.timer = a.timer.saturating_sub(1);
        if a.timer > 0 {
            return;
        }
        let load_ul = a.crop_ul;
        let molarity = a.load_molarity;
        let quality = a.load_quality;
        let fill = a.load_fill;
        let item = a.item_mg;
        a.crop_ul = 0.0;
        a.item_mg = 0.0;
        a.activity = Activity::Resting;
        // Poor sources are abandoned: the memory survives with a
        // probability that depends on the quality of the food and on how
        // much of it there was to drink.
        let keep = self.species.site_fidelity(quality * fill);
        if !self.rng.chance(keep) {
            self.ants[i].site = None;
        }
        // The store gains the load's sugar content, or the prey.
        let sugar = self.species.sugar_mg(load_ul, molarity);
        self.nest.store_mg = (self.nest.store_mg + sugar).min(self.nest.capacity_mg);
        self.stats.sugar_delivered_mg += sugar;
        self.nest.protein_mg += item;
        self.stats.protein_delivered_mg += item;
        let leaf = self.ants[i].leaf;
        let reward = self.config.reward.sugar_mg * sugar + self.config.reward.protein_mg * item;
        self.credit(leaf, reward, false);
        // Recruitment by contact: the returning forager excites nestmates.
        let inside: Vec<usize> = (0..self.ants.len())
            .filter(|&j| {
                j != i
                    && self.ants[j].alive
                    && matches!(self.ants[j].activity, Activity::Resting | Activity::Nursing)
            })
            .collect();
        if !inside.is_empty() {
            for _ in 0..self.species.contacts_per_return {
                let j = inside[self.rng.below(inside.len())];
                self.ants[j].excitement += self.species.excitation_per_contact * quality;
            }
        }
    }

    fn feed(&mut self, i: usize) {
        if self.ants[i].load_kind == Nutrient::Protein {
            self.cut_prey(i);
            return;
        }
        let cell = self.ants[i].cell();
        let (molarity, quality, crop, want_total) = {
            let a = &self.ants[i];
            let desired =
                self.species.crop_capacity_ul * self.species.load_fraction(a.load_quality);
            (a.load_molarity, a.load_quality, a.crop_ul, desired)
        };
        let rate = self.species.intake_rate(molarity) * self.tick_s;
        let want = rate.min((want_total - crop).max(0.0));
        let (taken, _) = if want > 0.0 {
            self.world.take_food(cell, want)
        } else {
            (0.0, molarity)
        };
        self.ants[i].crop_ul += taken;
        self.stats.food_collected_ul += taken;
        let short = taken < want - 1e-12;
        let full = self.ants[i].crop_ul >= want_total - 1e-9;
        // A drop that refills is worth waiting at, for a while; a dry
        // patch is not.
        let renewing = self
            .world
            .cell(cell)
            .map(|c| c.renewal_ul_per_s > 0.0)
            .unwrap_or(false);
        let patience = self.seconds_to_ticks(self.species.feeding_patience_s);
        let waited = {
            let a = &mut self.ants[i];
            if short {
                a.feed_wait += 1;
            }
            a.feed_wait
        };
        let give_up_waiting = short && (!renewing || waited > patience);
        if full || give_up_waiting {
            let fill = (self.ants[i].crop_ul / want_total.max(1e-12)).clamp(0.0, 1.0);
            self.ants[i].feed_wait = 0;
            if self.ants[i].carrying() {
                self.finish_feeding(i, quality, fill);
            } else {
                self.give_up(i);
            }
        }
    }

    /// Cutting a piece of prey takes a handling time; then the piece is
    /// carried home like a crop load.
    fn cut_prey(&mut self, i: usize) {
        let cell = self.ants[i].cell();
        let a = &mut self.ants[i];
        a.timer = a.timer.saturating_sub(1);
        if a.timer > 0 {
            return;
        }
        let want = self.species.prey_load_mg;
        let taken = self.world.take_prey(cell, want);
        if taken > 0.0 {
            self.ants[i].item_mg = taken;
            let quality = self.ants[i].load_quality;
            let fill = (taken / want.max(1e-12)).clamp(0.0, 1.0);
            self.finish_feeding(i, quality, fill);
        } else {
            self.give_up(i);
        }
    }

    fn finish_feeding(&mut self, i: usize, quality: f64, fill: f64) {
        // Recruitment rises with quality and with the volume ingested.
        let lay = self.species.lay_probability(quality)
            * self.ants[i].traits.laying
            * fill.powf(self.species.lay_load_exponent);
        let lays = self.rng.chance(lay.clamp(0.0, 1.0));
        if lays {
            self.stats.recruiting_trips += 1;
        }
        self.ants[i].load_fill = fill;
        let a = &mut self.ants[i];
        a.steps_since_food = 0;
        a.trip_length = 0.0;
        a.site = Some(Site {
            vector: a.home_vector,
            quality,
            nutrient: a.load_kind,
        });
        a.heading = crate::geometry::wrap_angle(a.heading + std::f64::consts::PI);
        a.activity = Activity::Inbound;
        a.search_steps = 0;
        // Marks per trip also rise with concentration (Beckers et al. 1993).
        a.laying = if lays {
            a.lay_strength = 0.25 + 0.75 * quality;
            Some(Pheromone::Trail)
        } else {
            None
        };
    }

    fn give_up(&mut self, i: usize) {
        let uses_no_entry = self.species.uses_no_entry;
        let a = &mut self.ants[i];
        a.failed_trips += 1;
        a.site = None;
        a.activity = Activity::Inbound;
        a.search_steps = 0;
        a.heading = crate::geometry::wrap_angle(a.heading + std::f64::consts::PI);
        a.laying = if uses_no_entry {
            a.lay_strength = 1.0;
            Some(Pheromone::NoEntry)
        } else {
            None
        };
        self.stats.failed_trips += 1;
    }

    fn arrive(&mut self, i: usize) {
        let satiation = self.nest.satiation();
        let unloading = self.seconds_to_ticks(self.species.unloading_time(satiation));
        let nest = self.world.nest();
        let cell = self.ants[i].cell();
        if let Some(c) = self.world.cell_mut(cell) {
            c.occupancy = c.occupancy.saturating_sub(1);
        }
        self.outside = self.outside.saturating_sub(1);
        let leaf = self.ants[i].leaf;
        let nest_radius = self.world.nest_radius().max(0) as f64;
        let trip = {
            let a = &mut self.ants[i];
            a.position = Point::center_of(nest);
            a.reset_home_vector();
            a.steps_since_nest = 0;
            a.laying = None;
            a.search_steps = 0;
            if a.carrying() {
                a.activity = Activity::Unloading;
                a.timer = unloading;
                a.deliveries += 1;
                let length = a.trip_length;
                let direct = a
                    .pickup
                    .take()
                    .map(|p| (p.distance(Point::center_of(nest)) - nest_radius).max(0.0))
                    .unwrap_or(0.0);
                Some((length, direct))
            } else {
                a.activity = Activity::Resting;
                None
            }
        };
        if let Some((length, direct)) = trip {
            self.stats.path.record_trip(length, direct);
            for &node in &self.leaf_paths[leaf] {
                self.stats.path_by_node[node].record_trip(length, direct);
            }
            self.stats.food_delivered += 1;
            for &node in &self.leaf_paths[leaf] {
                self.stats.delivered_by_node[node] += 1;
            }
        }
    }

    fn walk(&mut self, i: usize) {
        // Speed: cells per tick at the current temperature, faster on a
        // strong trail, slower when loaded and in a crowd.
        let speed = {
            let a = &self.ants[i];
            let here = a.cell();
            let trail = self.world.level(here, Pheromone::Trail);
            let k = self.world.channel(Pheromone::Trail).k;
            let crowding = self
                .world
                .cell(here)
                .map(|c| c.crowding(true))
                .unwrap_or(0.0);
            let mut speed = a.traits.speed * self.speed_factor;
            if a.carrying() {
                speed *= self.species.loaded_speed_factor;
            }
            if trail > k {
                speed *= self.species.trail_speed_factor;
            }
            speed *= (1.0 - self.species.crowding_slowdown * crowding).max(0.2);
            speed
        };
        self.ants[i].move_credit += speed;
        let total = self.ants[i].move_credit;
        if total >= Self::MIN_STEP {
            let n = (total.ceil() as usize).clamp(1, Self::MAX_STEPS_PER_TICK);
            let step = total / n as f64;
            self.ants[i].move_credit = 0.0;
            for _ in 0..n {
                let mode = match (self.ants[i].activity, self.ants[i].search_target) {
                    (Activity::Inbound, _) | (Activity::Searching, SearchTarget::Nest) => {
                        Mode::Inbound
                    }
                    _ => Mode::Outbound,
                };
                if let Some(ring) = self.decide(i, mode, step) {
                    self.move_ant(i, ring, step);
                }
                if self.check_transitions(i) {
                    return;
                }
            }
        }
        self.tick_timers(i);
    }

    fn move_ant(&mut self, i: usize, ring: usize, step: f64) {
        let leaf = self.ants[i].leaf;
        let (from, heading) = {
            let a = &self.ants[i];
            (a.position, ring_heading(ring, a.heading))
        };
        let to = from.advanced(heading, step);
        let from_cell = from.cell();
        let to_cell = to.cell();
        self.stats.path.record_move(ring, step);
        for &node in &self.leaf_paths[leaf] {
            self.stats.path_by_node[node].record_move(ring, step);
        }
        let entered = to_cell != from_cell;
        if entered {
            let revisit = self.ants[i].recently_visited(to_cell);
            self.stats.path.record_entry(revisit);
            for &node in &self.leaf_paths[leaf] {
                self.stats.path_by_node[node].record_entry(revisit);
            }
            if let Some(c) = self.world.cell_mut(from_cell) {
                c.occupancy = c.occupancy.saturating_sub(1);
            }
            if let Some(c) = self.world.cell_mut(to_cell) {
                c.occupancy = c.occupancy.saturating_add(1);
            }
            self.world.record_crossing(to_cell);
        }
        let species = &self.species;
        let persistence = self.persistence_cells;
        let (laying, strength, home, activity) = {
            let a = &mut self.ants[i];
            a.position = to;
            // The direction of travel: a running mean of recent steps for
            // moderate turns, an outright reorientation for sharper ones.
            a.heading = if turn_magnitude(ring) > RING / 4 {
                heading
            } else {
                let alpha = (step / persistence.max(1e-9)).min(1.0);
                let (s0, c0) = a.heading.sin_cos();
                let (s1, c1) = heading.sin_cos();
                crate::geometry::angle_of(
                    (1.0 - alpha) * c0 + alpha * c1,
                    (1.0 - alpha) * s0 + alpha * s1,
                )
            };
            if entered {
                a.remember(from_cell);
            }
            a.trip_length += step;
            let (dx, dy) = from.to(to);
            a.integrate(dx, dy, species, &mut self.rng);
            let home = species.uses_home_pheromone && a.activity == Activity::Outbound;
            (a.laying, a.lay_strength, home, a.activity)
        };
        // Crowding on the patch reduces deposition (Czaczkes et al. 2013).
        let crowding = self
            .world
            .cell(to_cell)
            .map(|c| c.crowding(true))
            .unwrap_or(0.0);
        let crowd_factor = 1.0 / (1.0 + species.crowding_deposition * crowding);
        match laying {
            Some(Pheromone::Trail) => self.world.deposit(
                to_cell,
                Pheromone::Trail,
                species.trail_deposit * strength * step * crowd_factor,
            ),
            Some(Pheromone::NoEntry) => self.world.deposit(
                to_cell,
                Pheromone::NoEntry,
                species.no_entry_deposit * strength * step,
            ),
            Some(kind) => self.world.deposit(to_cell, kind, strength * step),
            None => {}
        }
        if home {
            self.world
                .deposit(to_cell, Pheromone::Home, species.trail_deposit * step);
        }
        if species.territory_deposit > 0.0 {
            self.world.deposit(
                to_cell,
                Pheromone::Territory,
                species.territory_deposit * step,
            );
        }
        // Route memory: the direction just walked from the place left is
        // its local vector for this leg; the place entered records the
        // current home estimate; familiar places recalibrate it.
        if entered {
            let capacity = species.route_capacity;
            let rate = species.route_learning_rate;
            let correction = species.route_pi_correction;
            let dir = {
                let (dx, dy) = from.to(to);
                let len = (dx * dx + dy * dy).sqrt().max(1e-9);
                (dx / len, dy / len)
            };
            let a = &mut self.ants[i];
            match activity {
                Activity::Inbound if !a.lost => {
                    let home_vec = (-a.home_vector.0, -a.home_vector.1);
                    a.learn_route_home(from_cell, dir, rate, capacity);
                    a.learn_home_estimate(to_cell, home_vec, rate, capacity);
                }
                Activity::Outbound if !a.lost && a.site.is_some() => {
                    a.learn_route_out(from_cell, dir, rate, capacity);
                }
                Activity::Searching | Activity::Inbound => {
                    // Recognising a familiar place corrects the home vector;
                    // an ant that had lost the nest resumes its way home.
                    let recognised = a.recalibrate(to_cell, correction);
                    if recognised
                        && a.activity == Activity::Searching
                        && a.search_target == SearchTarget::Nest
                    {
                        a.activity = Activity::Inbound;
                        a.search_steps = 0;
                    }
                }
                _ => {}
            }
        }
    }

    /// React to the current position. Returns `true` when the ant stopped
    /// walking (it started feeding or entered the nest).
    fn check_transitions(&mut self, i: usize) -> bool {
        let cell = self.ants[i].cell();
        let on_nest = self.world.is_nest(cell);
        let accepts_prey = self.ants[i].accepts_prey;
        let (solution_here, molarity, prey_here) = self
            .world
            .cell(cell)
            .map(|c| (c.has_solution(), c.molarity, accepts_prey && c.has_prey()))
            .unwrap_or((false, 0.0, false));
        let arrival = self.species.arrival_radius;
        let activity = self.ants[i].activity;
        let target = self.ants[i].search_target;
        match (activity, target) {
            (Activity::Outbound, _) | (Activity::Searching, SearchTarget::Food) => {
                if solution_here && !self.ants[i].carrying() {
                    self.start_feeding(i, molarity);
                    return true;
                }
                if prey_here && !self.ants[i].carrying() {
                    self.start_cutting(i);
                    return true;
                }
                let a = &mut self.ants[i];
                if activity == Activity::Outbound
                    && a.believed_distance_to_site()
                        .map(|d| d < arrival)
                        .unwrap_or(false)
                {
                    a.activity = Activity::Searching;
                    a.search_target = SearchTarget::Food;
                    a.search_steps = 0;
                }
                false
            }
            (Activity::Inbound, _) | (Activity::Searching, SearchTarget::Nest) => {
                if on_nest {
                    self.arrive(i);
                    return true;
                }
                let a = &mut self.ants[i];
                if activity == Activity::Inbound
                    && !a.lost
                    && a.believed_distance_home() < arrival
                    && a.believed_distance_home() > 0.0
                {
                    a.activity = Activity::Searching;
                    a.search_target = SearchTarget::Nest;
                    a.search_steps = 0;
                }
                false
            }
            _ => false,
        }
    }

    /// Per-tick clocks of a walking ant: trip duration, search and give-up
    /// timeouts.
    fn tick_timers(&mut self, i: usize) {
        let search_ticks = self.seconds_to_ticks(self.species.search_time_s);
        let give_up_ticks = self.seconds_to_ticks(self.species.give_up_time_s);
        let activity = self.ants[i].activity;
        let target = self.ants[i].search_target;
        {
            let a = &mut self.ants[i];
            a.steps_since_nest = a.steps_since_nest.saturating_add(1);
            a.steps_since_food = a.steps_since_food.saturating_add(1);
        }
        match (activity, target) {
            (Activity::Searching, SearchTarget::Food) => {
                self.ants[i].search_steps += 1;
                if self.ants[i].search_steps > search_ticks {
                    self.give_up(i);
                }
            }
            (Activity::Outbound, _) => {
                if self.ants[i].steps_since_nest > give_up_ticks {
                    self.give_up(i);
                }
            }
            (Activity::Searching, SearchTarget::Nest) => {
                let a = &mut self.ants[i];
                a.search_steps += 1;
                if a.search_steps > search_ticks {
                    // Lost: path integration has failed; rely on familiar
                    // places, the home-range marking and the trail from
                    // here on.
                    a.activity = Activity::Inbound;
                    a.lost = true;
                    a.search_steps = 0;
                }
            }
            _ => {}
        }
    }

    fn start_feeding(&mut self, i: usize, molarity: f64) {
        let quality = self.species.quality(molarity);
        let leaf = self.ants[i].leaf;
        let a = &mut self.ants[i];
        a.activity = Activity::Feeding;
        a.load_kind = Nutrient::Sugar;
        a.load_molarity = molarity;
        a.load_quality = quality;
        a.pickup = Some(a.position);
        a.laying = None;
        a.steps_since_food = 0;
        a.trip_length = 0.0;
        self.stats.food_picked += 1;
        let reward = self.config.reward.food_picked;
        self.credit(leaf, reward, false);
    }

    /// Start cutting a piece of prey: its value to the colony is its
    /// protein demand, which sets recruitment and site fidelity.
    fn start_cutting(&mut self, i: usize) {
        let demand = self.nest.protein_demand(self.species.larva_protein_mg);
        let handling = self.seconds_to_ticks(self.species.prey_handling_s);
        let leaf = self.ants[i].leaf;
        let a = &mut self.ants[i];
        a.activity = Activity::Feeding;
        a.load_kind = Nutrient::Protein;
        a.load_molarity = 0.0;
        a.load_quality = 0.5 + 0.5 * demand;
        a.timer = handling;
        a.pickup = Some(a.position);
        a.laying = None;
        a.steps_since_food = 0;
        a.trip_length = 0.0;
        self.stats.food_picked += 1;
        self.stats.prey_picked += 1;
        let reward = self.config.reward.food_picked;
        self.credit(leaf, reward, false);
    }

    fn metabolize(&mut self, i: usize) {
        let mortality = self.config.nest.mortality;
        let starvation = self.species.starvation_s;
        let tick_s = self.tick_s;
        let store_has_food = self.nest.store_mg > 0.0;
        let hazard = self.hazard_per_tick;
        let inside = self.ants[i].is_inside();
        if inside {
            let a = &mut self.ants[i];
            if store_has_food {
                a.energy = starvation;
            } else {
                a.energy -= tick_s;
            }
        } else {
            self.ants[i].time_foraging += 1;
            self.ants[i].energy -= tick_s;
            if self.ants[i].activity == Activity::Feeding {
                self.ants[i].energy = starvation;
            }
            if mortality && self.rng.chance(hazard) {
                self.die(i, Cause::Predation);
                return;
            }
        }
        if mortality && self.ants[i].energy <= 0.0 {
            self.die(i, Cause::Starvation);
        }
    }

    fn die(&mut self, i: usize, cause: Cause) {
        let leaf = self.ants[i].leaf;
        let outside = !self.ants[i].is_inside();
        let cell = self.ants[i].cell();
        self.ants[i].alive = false;
        if outside {
            if let Some(c) = self.world.cell_mut(cell) {
                c.occupancy = c.occupancy.saturating_sub(1);
            }
            self.outside = self.outside.saturating_sub(1);
            if cause == Cause::Predation {
                self.world
                    .deposit(cell, Pheromone::Alarm, self.species.alarm_release);
            }
        }
        self.alive = self.alive.saturating_sub(1);
        self.stats.deaths += 1;
        match cause {
            Cause::Predation => self.stats.deaths_predation += 1,
            Cause::Starvation => self.stats.deaths_starvation += 1,
        }
        let reward = self.config.reward.death;
        self.credit(leaf, reward, false);
    }

    fn credit(&mut self, leaf: usize, reward: f64, delivered: bool) {
        self.stats.reward += reward;
        for &node in &self.leaf_paths[leaf] {
            self.stats.reward_by_node[node] += reward;
            if delivered {
                self.stats.delivered_by_node[node] += 1;
            }
        }
    }

    // ---------------------------------------------------------------
    // Nest dynamics
    // ---------------------------------------------------------------

    fn nest_step(&mut self) {
        let inside = self.inside_count() as f64;
        let consumption = inside
            * self.species.consumption_mg_per_ant_per_s
            * self.metabolism_factor
            * self.tick_s;
        self.nest.store_mg = (self.nest.store_mg - consumption).max(0.0);

        // Mean excitation of the workers inside, for reporting.
        let (sum, n) = self
            .ants
            .iter()
            .filter(|a| a.alive && a.is_inside())
            .fold((0.0, 0usize), |(s, n), a| (s + a.excitement, n + 1));
        self.nest.excitation = if n == 0 { 0.0 } else { sum / n as f64 };

        // Development, larval starvation, and emergence.
        let dev = self.development_factor * self.tick_s;
        let s = &self.species;
        let (egg_s, larva_s, pupa_s, larva_food, larva_protein, larva_starvation) = (
            s.egg_s,
            s.larva_s,
            s.pupa_s,
            s.larva_food_mg,
            s.larva_protein_mg,
            s.larva_starvation_s,
        );
        let tick_s = self.tick_s;
        let mut emerging = 0usize;
        let mut starved = 0u64;
        self.nest.brood.retain_mut(|b| {
            b.stage_age_s += dev;
            match b.stage {
                BroodStage::Egg => {
                    if b.stage_age_s >= egg_s {
                        b.stage = BroodStage::Larva;
                        b.stage_age_s = 0.0;
                    }
                    true
                }
                BroodStage::Larva => {
                    b.unfed_s += tick_s;
                    if b.unfed_s > larva_starvation {
                        starved += 1;
                        return false;
                    }
                    if b.stage_age_s >= larva_s
                        && b.fed_mg >= larva_food
                        && b.protein_mg >= larva_protein
                    {
                        b.stage = BroodStage::Pupa;
                        b.stage_age_s = 0.0;
                    }
                    true
                }
                BroodStage::Pupa => {
                    if b.stage_age_s >= pupa_s {
                        emerging += 1;
                        false
                    } else {
                        true
                    }
                }
            }
        });
        self.nest.larvae_starved += starved;
        self.stats.larvae_starved += starved;
        for _ in 0..emerging {
            if self.alive >= self.config.nest.max_ants {
                break;
            }
            let id = self.spawn_ant();
            let leaf = self.ants[id].leaf;
            self.nest.emerged += 1;
            self.stats.births += 1;
            let reward = self.config.reward.birth;
            self.credit(leaf, reward, false);
        }

        // Egg laying.
        if self.config.nest.queen {
            let max_brood =
                (self.config.nest.max_brood_per_ant * self.alive as f64).ceil() as usize;
            let interval = (self.species.egg_interval_s / self.tick_s).round().max(1.0) as u64;
            if self.nest.satiation() > 0.2
                && self.nest.brood.len() < max_brood
                && self.tick.saturating_sub(self.nest.last_egg_tick) >= interval
            {
                self.nest.brood.push(BroodItem {
                    stage: BroodStage::Egg,
                    stage_age_s: 0.0,
                    fed_mg: 0.0,
                    protein_mg: 0.0,
                    unfed_s: 0.0,
                });
                self.nest.eggs_laid += 1;
                self.nest.last_egg_tick = self.tick;
                self.stats.eggs += 1;
            }
        }
    }

    // ---------------------------------------------------------------
    // Movement decision
    // ---------------------------------------------------------------

    fn decide(&mut self, i: usize, mode: Mode, step: f64) -> Option<usize> {
        let (heading, leaf, id, position, carrying, searching) = {
            let a = &self.ants[i];
            (
                a.heading,
                a.leaf,
                a.id,
                a.position,
                a.carrying(),
                a.activity == Activity::Searching,
            )
        };
        let obs = observe(&self.ants[i], &self.world, &self.species, mode, step);
        let policy = self.policies[leaf].clone();

        // The deterministic information of the path: scores per heading.
        let mut scores = [f64::NEG_INFINITY; RING];
        let mut any = false;
        for ((score, valid), features) in scores.iter_mut().zip(&obs.valid).zip(&obs.features) {
            if *valid {
                *score = crate::surface::dot(&policy.weights, features);
                any = true;
            }
        }
        if !any {
            return None;
        }
        let base = Landscape::new(scores, obs.valid);

        // Geometric deformation with the entropy budget.
        let tempering = if searching {
            policy
                .tempering
                .heated(self.species.search_temperature_factor)
        } else {
            policy.tempering
        };
        let h = match tempering {
            Tempering::Entropy(f) => f,
            Tempering::Temperature(t) => base.entropy_fraction_at(t),
        };
        let smooth_scale = policy.deformation.smooth_share() * h * self.config.geometry.smooth_max;
        let smoothed = base.smoothed(smooth_scale);
        let rough_amplitude = policy.deformation.rough_share() * h * base.range().max(1.0);
        let deformed = smoothed.roughened(
            rough_amplitude,
            self.config.geometry.rough_modes,
            &mut self.rng,
        );
        let tempered = deformed.temper_with(tempering);

        // Selection.
        let start = base.nearest_valid(0).expect("at least one valid heading");
        let recording = self.config.record_surface == Some(id);
        let keep = self.config.geometry.ledger || recording;
        let (chosen, walk, selected) = match self.config.selection {
            Selection::Softmax => (
                self.rng.choose_weighted(&tempered.probs),
                Vec::new(),
                tempered.probs,
            ),
            Selection::Sucker { .. } => {
                let reach = self.effective_reach(&policy).unwrap_or(0);
                let sucker = Sucker { reach };
                let (j, walk) = sucker.walk(&tempered.scaled, start, &mut self.rng);
                let dist = if keep {
                    sucker.distribution(&tempered.scaled, start)
                } else {
                    tempered.probs
                };
                (j, walk, dist)
            }
        };

        // Accounting.
        self.stats.decisions += 1;
        self.stats.entropy_sum += tempered.entropy;
        let selected_entropy = if keep {
            entropy(&selected)
        } else {
            tempered.entropy
        };
        self.stats.selected_entropy_sum += selected_entropy;
        if self.config.geometry.ledger {
            let t = tempered.temperature;
            let base_h = base.entropy_at(t);
            let smooth_h = smoothed.entropy_at(t);
            let samples = self.config.geometry.field_samples;
            let mixture_h = if rough_amplitude > 0.0 && samples > 0 {
                let mut mixture = tempered.probs;
                for _ in 0..samples {
                    let other = smoothed
                        .roughened(
                            rough_amplitude,
                            self.config.geometry.rough_modes,
                            &mut self.rng,
                        )
                        .temper_with(tempering);
                    for (m, p) in mixture.iter_mut().zip(&other.probs) {
                        *m += p;
                    }
                }
                for m in mixture.iter_mut() {
                    *m /= (samples + 1) as f64;
                }
                entropy(&mixture)
            } else {
                tempered.entropy
            };
            self.stats.path.ledger.record(
                base_h,
                smooth_h,
                tempered.entropy,
                mixture_h,
                selected_entropy,
            );
            for &node in &self.leaf_paths[leaf] {
                self.stats.path_by_node[node].ledger.record(
                    base_h,
                    smooth_h,
                    tempered.entropy,
                    mixture_h,
                    selected_entropy,
                );
            }
        }
        if self.config.trace {
            let mut score = [0.0; FEATURES];
            for (p, features) in tempered.probs.iter().zip(&obs.features) {
                if *p > 0.0 {
                    for (s, f) in score.iter_mut().zip(features) {
                        *s -= p * f;
                    }
                }
            }
            for (s, f) in score.iter_mut().zip(&obs.features[chosen]) {
                *s = (*s + f) / tempered.temperature;
            }
            for &node in &self.leaf_paths[leaf] {
                let acc = &mut self.trace.score_by_node[node];
                for (a, s) in acc.iter_mut().zip(&score) {
                    *a += s;
                }
                self.trace.decisions_by_node[node] += 1;
            }
        }
        if recording {
            self.surface.push(SurfaceRow {
                tick: self.tick,
                position,
                heading,
                carrying,
                valid: base.valid,
                base: base.values,
                deformed: deformed.values,
                probs: tempered.probs,
                selected,
                chosen,
                walk,
                temperature: tempered.temperature,
            });
            let cap = self.config.surface_rows.max(1);
            if self.surface.len() > cap {
                let excess = self.surface.len() - cap;
                self.surface.drain(..excess);
            }
        }
        Some(chosen)
    }
}

/// Heading angle of a vector, for callers that want to face something.
pub fn heading_towards(dx: f64, dy: f64) -> f64 {
    angle_of(dx, dy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::EntropyControl;
    use crate::surface::Deformation;
    use crate::world::FoodSource;

    fn quick_config() -> SimConfig {
        SimConfig {
            ants: 40,
            world: WorldConfig {
                width: 40,
                height: 30,
                nest: Position::new(20, 15),
                seed: Some(7),
                ..WorldConfig::default()
            },
            nest: NestConfig {
                initial_satiation: 0.1,
                ..NestConfig::default()
            },
            ..SimConfig::default()
        }
    }

    /// A hungry colony of the default species.
    fn hungry_fast() -> SimConfig {
        let mut cfg = quick_config();
        cfg.nest.initial_satiation = 0.05;
        cfg
    }

    /// A fed colony with life history compressed enough for brood to
    /// develop, pupate and emerge, and eggs to be laid, within a short run.
    fn life_history() -> SimConfig {
        let species = Species::lasius_niger().compressed(600.0);
        let mut cfg = SimConfig::for_species(species);
        cfg.ants = 30;
        cfg.world = WorldConfig {
            width: 40,
            height: 30,
            nest: Position::new(20, 15),
            seed: Some(7),
            ..WorldConfig::default()
        };
        cfg.nest.initial_satiation = 0.9;
        cfg.nest.initial_brood_per_ant = 0.5;
        cfg
    }

    #[test]
    fn colony_forages_and_delivers() {
        let mut sim = Simulation::new(hungry_fast(), 1);
        sim.run(1500);
        let s = sim.stats();
        assert!(s.food_picked > 0, "ants should find food: {s:?}");
        assert!(s.food_delivered > 0, "ants should bring food home: {s:?}");
        assert!(s.sugar_delivered_mg > 0.0 && s.food_collected_ul > 0.0);
        assert!(sim.nest().store_mg > 0.0);
        assert_eq!(s.path.trips, s.food_delivered);
        assert!(s.path.trip_efficiency() > 0.0 && s.path.trip_efficiency() <= 1.0);
        let total: u64 = sim
            .hierarchy()
            .leaves()
            .iter()
            .map(|&l| s.delivered_by_node[l])
            .sum();
        assert_eq!(total, s.food_delivered);
        assert_eq!(s.delivered_by_node[0], s.food_delivered);
        assert_eq!(s.path.moves, s.path.turns.iter().sum::<u64>());
        assert!(s.path.length > 0.0);
        assert!(s.mean_entropy() > 0.0);
        assert!(!s.log.is_empty());
        assert!(s.log.last().unwrap().delivered == s.food_delivered);
        assert!(
            sim.living().any(|a| a.familiar_places() > 0),
            "routes are learned"
        );
    }

    #[test]
    fn deterministic_for_seed() {
        let mut a = Simulation::new(hungry_fast(), 5);
        let mut b = Simulation::new(hungry_fast(), 5);
        a.run(300);
        b.run(300);
        assert_eq!(a.stats(), b.stats());
        let mut c = Simulation::new(hungry_fast(), 6);
        c.run(300);
        assert!(a.stats() != c.stats());
    }

    #[test]
    fn hunger_drives_foraging() {
        let mut hungry = Simulation::new(hungry_fast(), 3);
        hungry.run(600);
        let mut replete_cfg = hungry_fast();
        replete_cfg.nest.initial_satiation = 1.0;
        let mut replete = Simulation::new(replete_cfg, 3);
        replete.run(600);
        let out_h = hungry.stats().foraging_fraction();
        let out_r = replete.stats().foraging_fraction();
        assert!(out_h > out_r + 0.1, "hungry {out_h} vs replete {out_r}");
    }

    #[test]
    fn entropy_dial_still_controls_decisions() {
        let mut ordered = Simulation::new(hungry_fast(), 3);
        ordered.hierarchy_mut().node_mut(0).surface.entropy = EntropyControl::absolute(0.05);
        ordered.run(400);
        let mut chaotic = Simulation::new(hungry_fast(), 3);
        chaotic.hierarchy_mut().node_mut(0).surface.entropy = EntropyControl::absolute(0.98);
        chaotic.run(400);
        let h_low = ordered.stats().mean_entropy();
        let h_high = chaotic.stats().mean_entropy();
        let max = (RING as f64).ln();
        assert!(h_low < 0.15 * max, "low dial → low entropy, got {h_low}");
        assert!(
            h_high > 0.85 * max,
            "high dial → near-uniform, got {h_high}"
        );
        let mut natural = Simulation::new(hungry_fast(), 3);
        natural.run(400);
        let h = natural.stats().mean_entropy();
        assert!(h > 0.1 * max && h < 0.95 * max, "{h}");
    }

    #[test]
    fn relative_control_on_a_caste_scales_its_temperature() {
        let mut sim = Simulation::new(quick_config(), 3);
        sim.hierarchy_mut().node_mut(1).surface.entropy = EntropyControl::relative(2.5);
        sim.recompile();
        let policies = sim.policies();
        let leaves = sim.hierarchy().leaves().to_vec();
        for (idx, &leaf) in leaves.iter().enumerate() {
            let in_caste0 = sim.hierarchy().path(leaf).contains(&1);
            let t = policies[idx].temperature().unwrap();
            if in_caste0 {
                assert!((t - 2.5).abs() < 1e-9);
            } else {
                assert!((t - 1.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn trace_accumulates_only_when_enabled() {
        let mut off = Simulation::new(hungry_fast(), 2);
        off.run(100);
        assert!(off.trace().score_by_node[0].iter().all(|x| *x == 0.0));
        let cfg = SimConfig {
            trace: true,
            ..hungry_fast()
        };
        let mut on = Simulation::new(cfg, 2);
        on.run(100);
        let t = on.trace();
        assert_eq!(t.decisions_by_node[0], on.stats().decisions);
        assert!(t.score_by_node[0].iter().any(|x| x.abs() > 0.0));
        assert!(t.score_by_node[0][FEATURES..].iter().all(|x| *x == 0.0));
    }

    #[test]
    fn brood_develops_through_stages_and_emerges() {
        let mut sim = Simulation::new(life_history(), 4);
        let initial = sim.nest().brood.len();
        assert!(initial > 0);
        let stages_at_start = (
            sim.nest().count(BroodStage::Egg),
            sim.nest().count(BroodStage::Larva),
            sim.nest().count(BroodStage::Pupa),
        );
        assert!(stages_at_start.0 + stages_at_start.1 + stages_at_start.2 == initial);
        sim.run(2400);
        let s = sim.stats();
        assert!(s.births > 0, "pupae should emerge: {:?}", sim.nest());
        assert!(s.eggs > 0, "the queen should lay: {:?}", sim.nest());
        assert!(
            s.activity_ticks[Activity::Nursing.index()] > 0,
            "larvae get nursed"
        );
        assert_eq!(sim.alive(), sim.living().count());
        let occupancy: u32 = sim.world().cells().iter().map(|c| c.occupancy as u32).sum();
        assert_eq!(occupancy as usize, sim.outside());
        assert!(s.deaths_predation + s.deaths_starvation == s.deaths);
    }

    #[test]
    fn larvae_need_protein_to_pupate() {
        // Larvae only, sugar and prey within reach: foragers bring prey
        // because the larvae demand it, nurses feed it, larvae pupate.
        let mut cfg = life_history();
        cfg.nest.initial_brood_per_ant = 0.0;
        cfg.nest.queen = false;
        cfg.nest.initial_satiation = 0.5;
        cfg.world.random_food = None;
        cfg.world.food_sources = vec![
            FoodSource::pool(Position::new(28, 15), 1, 1.0e6, 1.0),
            FoodSource::prey(Position::new(12, 15), 1, 1.0e6),
        ];
        let larvae = |n: usize| {
            (0..n)
                .map(|_| BroodItem {
                    stage: BroodStage::Larva,
                    stage_age_s: 0.0,
                    fed_mg: 0.0,
                    protein_mg: 0.0,
                    unfed_s: 0.0,
                })
                .collect::<Vec<_>>()
        };
        let mut sim = Simulation::new(cfg.clone(), 4);
        sim.nest_mut().brood = larvae(15);
        assert!((sim.nest().protein_demand(sim.species().larva_protein_mg) - 1.0).abs() < 1e-12);
        sim.run(3000);
        let s = sim.stats();
        assert!(s.prey_picked > 0 && s.protein_delivered_mg > 0.0, "{s:?}");
        assert!(
            sim.nest().count(BroodStage::Pupa) > 0,
            "fed larvae should pupate: {:?}",
            sim.nest()
        );
        // Without prey the larvae stay larvae however much sugar they get.
        cfg.world.food_sources.pop();
        let mut starved = Simulation::new(cfg, 4);
        starved.nest_mut().brood = larvae(15);
        starved.run(3000);
        assert_eq!(starved.stats().prey_picked, 0);
        assert_eq!(starved.nest().count(BroodStage::Pupa), 0);
        assert!(starved.stats().food_delivered > 0);
    }

    #[test]
    fn unfed_larvae_starve() {
        let mut cfg = life_history();
        cfg.nest.initial_satiation = 0.0;
        cfg.world.random_food = None;
        cfg.species.larva_starvation_s = 200.0;
        let mut sim = Simulation::new(cfg, 4);
        let larvae = sim.nest().larvae();
        assert!(larvae > 0);
        sim.run(600);
        assert!(sim.stats().larvae_starved > 0, "{:?}", sim.nest());
    }

    #[test]
    fn no_mortality_keeps_everyone_alive() {
        let mut cfg = hungry_fast();
        cfg.nest.mortality = false;
        let mut sim = Simulation::new(cfg, 4);
        sim.run(800);
        assert_eq!(sim.stats().deaths, 0);
        assert_eq!(sim.alive(), 40);
    }

    #[test]
    fn temperature_changes_speed_and_activity() {
        let mut warm_cfg = hungry_fast();
        warm_cfg.environment.temperature_c = 30.0;
        let mut warm = Simulation::new(warm_cfg, 5);
        warm.run(600);
        let mut cold_cfg = hungry_fast();
        cold_cfg.environment.temperature_c = 12.0;
        let mut cold = Simulation::new(cold_cfg, 5);
        cold.run(600);
        assert!(warm.stats().path.length > 1.5 * cold.stats().path.length);
        let mut frozen_cfg = hungry_fast();
        frozen_cfg.environment.temperature_c = 4.0;
        let mut frozen = Simulation::new(frozen_cfg, 5);
        frozen.run(300);
        assert_eq!(
            frozen.stats().foraging_fraction(),
            0.0,
            "too cold to forage"
        );
        assert!(warm.world().evaporation_factor() > 1.5);
        assert!(cold.world().evaporation_factor() < 0.6);
    }

    #[test]
    fn diurnal_cycle_moves_the_temperature() {
        let env = Environment {
            temperature_c: 20.0,
            diurnal_amplitude_c: 8.0,
            day_length_s: 100.0,
            coldest_at_s: 0.0,
        };
        assert!((env.temperature(0.0) - 12.0).abs() < 1e-9);
        assert!((env.temperature(50.0) - 28.0).abs() < 1e-9);
        let mut cfg = hungry_fast();
        cfg.environment = env.clone();
        let mut sim = Simulation::new(cfg, 6);
        sim.run(100);
        let temps: Vec<f64> = sim.stats().log.iter().map(|s| s.temperature_c).collect();
        assert!(!temps.is_empty());
        // The temperature in force during a tick is the one at its start.
        let expected = env.temperature(sim.time_s() - sim.tick_s());
        assert!((sim.temperature() - expected).abs() < 1e-9);
        sim.step();
        assert!((sim.temperature() - 12.0).abs() < 1e-6);
    }

    #[test]
    fn set_params_marks_dirty_and_applies() {
        let mut sim = Simulation::new(quick_config(), 8);
        let mut p = sim.params();
        p[0] = 9.0;
        sim.set_params(&p).unwrap();
        sim.step();
        assert_eq!(sim.policies()[0].weights[0], 9.0);
        assert!(sim.set_params(&p[..5]).is_err());
    }

    #[test]
    fn ledger_decomposes_without_deformation() {
        let mut sim = Simulation::new(hungry_fast(), 9);
        sim.run(200);
        let c = sim.stats().path.ledger.contributions();
        assert_eq!(sim.stats().path.ledger.decisions, sim.stats().decisions);
        assert!(c.smoothing.abs() < 1e-9, "no smoothing configured: {c:?}");
        assert!(c.roughening.abs() < 1e-9, "no roughening configured: {c:?}");
        assert!(c.field.abs() < 1e-9, "no field without roughening: {c:?}");
        assert!(c.selection.abs() < 1e-9, "global draw loses nothing: {c:?}");
        assert!((c.tempering - sim.stats().mean_entropy()).abs() < 1e-6);
    }

    #[test]
    fn smoothing_and_roughening_show_up_in_the_ledger() {
        let mut smooth = Simulation::new(hungry_fast(), 10);
        smooth.hierarchy_mut().node_mut(0).surface.deformation = Deformation {
            smooth: 2.0,
            rough: 0.0,
            reach: 0.0,
        };
        smooth.run(200);
        let c = smooth.stats().path.ledger.contributions();
        assert!(c.smoothing > 0.02, "smoothing adds entropy: {c:?}");
        assert!(c.roughening.abs() < 1e-9);

        let mut rough = Simulation::new(hungry_fast(), 10);
        rough.hierarchy_mut().node_mut(0).surface.deformation = Deformation {
            smooth: 0.0,
            rough: 2.0,
            reach: 0.0,
        };
        rough.run(200);
        let c = rough.stats().path.ledger.contributions();
        assert!(
            c.roughening.abs() > 0.02,
            "roughening changes the landscape: {c:?}"
        );
        assert!(
            c.field > 0.02,
            "the field randomises across decisions: {c:?}"
        );
    }

    #[test]
    fn sucker_selection_runs() {
        let cfg = SimConfig {
            selection: Selection::Sucker { reach: 8 },
            ..hungry_fast()
        };
        let mut sim = Simulation::new(cfg, 11);
        sim.run(1200);
        let s = sim.stats();
        assert!(s.food_delivered > 0, "the sucker still forages: {s:?}");
        let gap = s.mean_selected_entropy() - s.mean_entropy();
        let c = s.path.ledger.contributions();
        assert!((c.selection - gap).abs() < 1e-6);
        assert_eq!(sim.effective_reach(&sim.policies()[0].clone()), Some(8));
    }

    #[test]
    fn surface_recording_keeps_the_last_rows() {
        let cfg = SimConfig {
            selection: Selection::Sucker { reach: 4 },
            record_surface: Some(3),
            surface_rows: 10,
            ..hungry_fast()
        };
        let mut sim = Simulation::new(cfg, 13);
        sim.run(600);
        let rows = sim.surface_trace();
        assert_eq!(rows.len(), 10);
        assert!(rows.windows(2).all(|w| w[0].tick <= w[1].tick));
        for row in rows {
            assert!(row.valid[row.chosen]);
            assert_eq!(row.walk.len(), 5);
            assert!((row.probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        sim.clear_surface_trace();
        assert!(sim.surface_trace().is_empty());
    }

    #[test]
    fn division_of_labor_emerges_with_brood() {
        let mut cfg = quick_config();
        cfg.ants = 60;
        cfg.nest.initial_satiation = 0.3;
        cfg.nest.initial_brood_per_ant = 1.0;
        let mut sim = Simulation::new(cfg, 2);
        sim.run(2000);
        let dol = sim.division_of_labor();
        let s = sim.stats();
        assert!(s.activity_ticks[Activity::Nursing.index()] > 0);
        assert!(s.foraging_fraction() > 0.0);
        assert!(dol > 0.3, "workers should specialise: {dol}");
    }

    #[test]
    fn contacts_excite_nestmates() {
        let mut sim = Simulation::new(hungry_fast(), 14);
        sim.run(900);
        assert!(sim.stats().food_delivered > 0);
        let excited = sim.living().filter(|a| a.excitement > 0.0).count();
        assert!(
            excited > 0,
            "returning foragers should have excited nestmates"
        );
        assert!(sim.stats().log.iter().any(|s| s.excitation > 0.0));
    }
}
