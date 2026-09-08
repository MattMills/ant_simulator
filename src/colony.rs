//! The colony simulation: world, nest, ants, and the hierarchy that governs
//! their movement.
//!
//! Each tick every living ant acts according to its activity. Inside the
//! nest it rests, nurses brood, or unloads by trophallaxis, and decides
//! whether to leave by a response threshold on the colony's hunger and
//! excitation. Outside, it walks by scoring the eight neighbouring cells
//! through the effective policy of its category, laying that ring out as a
//! landscape, deforming it with the entropy budget, and selecting a
//! direction (a global draw or a crawling sucker). It lays pheromone, keeps
//! a path-integration home vector, feeds at food, returns, searches when
//! its estimate runs out, and may starve or be taken by a predator. The
//! nest consumes food, the queen lays eggs when fed, brood develops into
//! workers, and excitation from returning foragers decays.

use crate::ant::{observe, Activity, Ant, AntId, Mode, SearchTarget, Site, Traits, FEATURES};
use crate::entropy::{entropy, Tempering};
use crate::geometry::{Direction, Position};
use crate::hierarchy::{EffectivePolicy, Hierarchy, HierarchySpec, NodeId};
use crate::landscape::{ring_index, world_direction, EntropyLedger, Landscape, Sucker, RING};
use crate::pheromone::Pheromone;
use crate::rng::Rng;
use crate::species::Species;
use crate::surface::{BehavioralSurface, SurfaceError, PARAM_LEN};
use crate::world::{World, WorldConfig};

/// How events translate into the scalar reward learners optimise.
#[derive(Clone, Debug, PartialEq)]
pub struct RewardSpec {
    /// Reward per crop load delivered into the nest.
    pub food_delivered: f64,
    /// Reward per crop load collected at a source.
    pub food_picked: f64,
    /// Reward (normally negative) per dead worker.
    pub death: f64,
    /// Reward per worker that emerges from the brood.
    pub birth: f64,
}

impl Default for RewardSpec {
    fn default() -> Self {
        RewardSpec {
            food_delivered: 1.0,
            food_picked: 0.1,
            death: -0.5,
            birth: 0.0,
        }
    }
}

/// How a direction is drawn from the deformed landscape.
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
            smooth_max: 2.0,
            rough_modes: 3,
            max_reach: 64,
            ledger: true,
            field_samples: 4,
        }
    }
}

/// Nest and colony-level parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct NestConfig {
    /// Food store capacity per worker, in crop loads.
    pub store_capacity_per_ant: f64,
    /// Initial fill of the store as a fraction of capacity.
    pub initial_satiation: f64,
    /// Initial brood items per worker.
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
            store_capacity_per_ant: 2.0,
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
    /// Initial number of workers.
    pub ants: usize,
    /// Nest parameters.
    pub nest: NestConfig,
    /// Reward definition.
    pub reward: RewardSpec,
    /// Whether to accumulate policy-gradient score sums per node.
    pub trace: bool,
    /// How directions are selected from the landscape.
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
        SimConfig {
            world: WorldConfig::default(),
            hierarchy: HierarchySpec::default(),
            instinct: species.instinct(),
            species,
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

/// One brood item (egg, larva, or pupa).
#[derive(Clone, Debug, PartialEq)]
pub struct BroodItem {
    /// Ticks since it was laid.
    pub age: u64,
    /// Food received so far.
    pub fed: f64,
}

/// The nest interior.
#[derive(Clone, Debug, PartialEq)]
pub struct Nest {
    /// Food in store, in crop loads.
    pub store: f64,
    /// Store capacity.
    pub capacity: f64,
    /// Developing brood.
    pub brood: Vec<BroodItem>,
    /// Recruitment excitation from recently returned foragers.
    pub excitation: f64,
    /// Eggs laid so far.
    pub eggs_laid: u64,
    /// Workers that emerged so far.
    pub emerged: u64,
    /// Tick of the last egg.
    pub last_egg_tick: u64,
}

impl Nest {
    /// Fill of the store, 0 (empty) to 1 (full).
    pub fn satiation(&self) -> f64 {
        if self.capacity <= 0.0 {
            0.0
        } else {
            (self.store / self.capacity).clamp(0.0, 1.0)
        }
    }

    /// `1 - satiation`.
    pub fn hunger(&self) -> f64 {
        1.0 - self.satiation()
    }
}

/// Geometry of the paths ants walked, plus the entropy ledger.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PathStats {
    /// Moves made.
    pub moves: u64,
    /// Moves by ring position relative to the previous heading
    /// (0 straight, 4 reverse; see [`crate::landscape::TURN_LABELS`]).
    pub turns: [u64; RING],
    /// Moves onto a cell the ant had visited recently.
    pub revisits: u64,
    /// Completed food-to-nest trips.
    pub trips: u64,
    /// Steps taken on those trips.
    pub trip_steps: u64,
    /// Straight-line (Chebyshev) distance those trips needed.
    pub trip_direct: u64,
    /// Where the decision entropy came from.
    pub ledger: EntropyLedger,
}

impl PathStats {
    fn record_move(&mut self, ring: usize, revisit: bool) {
        self.moves += 1;
        self.turns[ring] += 1;
        if revisit {
            self.revisits += 1;
        }
    }

    fn record_trip(&mut self, steps: u64, direct: u64) {
        self.trips += 1;
        self.trip_steps += steps;
        self.trip_direct += direct;
    }

    /// Add another record's counts.
    pub fn merge(&mut self, other: &PathStats) {
        self.moves += other.moves;
        for (a, b) in self.turns.iter_mut().zip(&other.turns) {
            *a += b;
        }
        self.revisits += other.revisits;
        self.trips += other.trips;
        self.trip_steps += other.trip_steps;
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

    /// Fraction of moves that kept the heading.
    pub fn straight_rate(&self) -> f64 {
        if self.moves == 0 {
            0.0
        } else {
            self.turns[0] as f64 / self.moves as f64
        }
    }

    /// Fraction of moves that reversed the heading.
    pub fn reversal_rate(&self) -> f64 {
        if self.moves == 0 {
            0.0
        } else {
            self.turns[RING / 2] as f64 / self.moves as f64
        }
    }

    /// Fraction of moves onto a recently visited cell.
    pub fn revisit_rate(&self) -> f64 {
        if self.moves == 0 {
            0.0
        } else {
            self.revisits as f64 / self.moves as f64
        }
    }

    /// Straight-line distance divided by steps taken, averaged over trips
    /// (1 is a perfectly direct return; 0 if there were no trips).
    pub fn trip_efficiency(&self) -> f64 {
        if self.trip_steps == 0 {
            0.0
        } else {
            self.trip_direct as f64 / self.trip_steps as f64
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
    /// Living workers.
    pub alive: usize,
    /// Workers outside the nest.
    pub outside: usize,
    /// Workers nursing.
    pub nursing: usize,
    /// Workers resting.
    pub resting: usize,
    /// Food in store.
    pub store: f64,
    /// Store fill.
    pub satiation: f64,
    /// Recruitment excitation.
    pub excitation: f64,
    /// Crop loads delivered so far.
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
    /// Crop loads delivered into the store.
    pub food_delivered: u64,
    /// Crop loads collected at sources.
    pub food_picked: u64,
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
    /// Total reward.
    pub reward: f64,
    /// Number of movement decisions made.
    pub decisions: u64,
    /// Sum of the entropies met by tempering (nats).
    pub entropy_sum: f64,
    /// Sum of entropies of the distributions directions were actually drawn
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
    /// Food in store at the end of the last tick.
    pub food_store: f64,
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

    /// Mean entropy of the distributions directions were drawn from.
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
    pub position: Position,
    /// Its heading (ring index 0 in world terms).
    pub heading: Direction,
    /// Whether it carried food.
    pub carrying: bool,
    /// Enterable ring positions.
    pub valid: [bool; RING],
    /// The deterministic information: scores from the effective surface.
    pub base: [f64; RING],
    /// Scores after smoothing and roughening.
    pub deformed: [f64; RING],
    /// The tempered distribution.
    pub probs: [f64; RING],
    /// The distribution the direction was actually drawn from.
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
    decision_prob: f64,
    hazard_per_tick: f64,
    excitation_retention: f64,
    log_every_ticks: u64,
}

impl Simulation {
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
        let nodes = hierarchy.len();
        let leaf_paths = hierarchy
            .leaves()
            .iter()
            .map(|&l| hierarchy.path(l).to_vec())
            .collect();
        let capacity = config.nest.store_capacity_per_ant * config.ants.max(1) as f64;
        let nest = Nest {
            store: capacity * config.nest.initial_satiation.clamp(0.0, 1.0),
            capacity,
            brood: Vec::new(),
            excitation: 0.0,
            eggs_laid: 0,
            emerged: 0,
            last_egg_tick: 0,
        };
        let nest_cells = world.nest_cells();
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
            decision_prob: (tick_s / species.decision_interval_s.max(1e-9)).min(1.0),
            hazard_per_tick: 1.0 - (-species.forager_hazard_per_s * tick_s).exp(),
            excitation_retention: 0.5f64.powf(tick_s / species.excitation_half_life_s.max(1e-9)),
            log_every_ticks: if config.nest.log_every_s > 0.0 {
                (config.nest.log_every_s / tick_s).round().max(1.0) as u64
            } else {
                0
            },
            species,
            config,
        };
        for _ in 0..sim.config.ants {
            let age_s = sim
                .rng
                .range(0.0, sim.config.nest.initial_age_spread_s.max(0.0));
            let id = sim.spawn_ant();
            sim.ants[id].age = (age_s / tick_s) as u64;
        }
        let brood_items =
            (sim.config.nest.initial_brood_per_ant * sim.config.ants as f64).round() as usize;
        let dev_ticks = sim.development_ticks();
        for _ in 0..brood_items {
            let age = sim.rng.below(dev_ticks.max(1) as usize) as u64;
            let fed = sim.species.brood_food * age as f64 / dev_ticks.max(1) as f64;
            sim.nest.brood.push(BroodItem { age, fed });
        }
        sim.stats.alive = sim.alive;
        sim.stats.food_store = sim.nest.store;
        sim
    }

    fn development_ticks(&self) -> u64 {
        (self.species.development_s / self.tick_s).round().max(1.0) as u64
    }

    fn seconds_to_ticks(&self, seconds: f64) -> u32 {
        (seconds / self.tick_s).round().max(1.0) as u32
    }

    fn spawn_ant(&mut self) -> usize {
        let id = self.ants.len();
        let heading = Direction::from_index(self.rng.below(Direction::COUNT));
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
        self.stats.food_store = self.nest.store;
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
        for i in 0..self.ants.len() {
            if self.ants[i].alive {
                self.step_ant(i);
            }
        }
        self.world.step_pheromones();
        self.nest_step();
        self.tick += 1;
        self.stats.ticks += 1;
        self.stats.time_s += self.tick_s;
        self.stats.alive = self.alive;
        self.stats.outside = self.outside;
        self.stats.food_store = self.nest.store;
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
            alive: self.alive,
            outside: self.outside,
            nursing,
            resting,
            store: self.nest.store,
            satiation: self.nest.satiation(),
            excitation: self.nest.excitation,
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
        let (site_bonus, threshold, nursing_threshold) = {
            let a = &self.ants[i];
            let age_s = a.age as f64 * self.tick_s;
            (
                a.site
                    .map(|s| self.species.reforage_bonus * s.quality)
                    .unwrap_or(0.0),
                self.species
                    .threshold_at_age(a.traits.foraging_threshold, age_s),
                a.traits.nursing_threshold,
            )
        };
        let stimulus = self.species.hunger_gain * hunger + self.nest.excitation + site_bonus;
        let p_forage = self.species.response(stimulus, threshold) * self.decision_prob;
        let demand = if self.nest.brood.is_empty() {
            0.0
        } else {
            let nurses = self.nurses_now() as f64;
            (self.nest.brood.len() as f64 / self.species.brood_per_nurse.max(1e-9)) / (nurses + 1.0)
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
        let heading = Direction::from_index(self.rng.below(Direction::COUNT));
        let outbound_laying = self.species.outbound_laying;
        let a = &mut self.ants[i];
        a.activity = Activity::Outbound;
        a.position = exit;
        a.heading = heading;
        a.reset_home_vector();
        a.clear_memory();
        a.steps_since_nest = 0;
        a.search_steps = 0;
        a.move_credit = 0.0;
        a.laying = if outbound_laying && a.site.is_some() {
            a.lay_strength = 0.5;
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
        let rate = self.species.nursing_rate * self.tick_s;
        let need_cap = self.species.brood_food;
        let available = rate.min(self.nest.store);
        if available > 0.0 {
            if let Some(hungriest) = self.nest.brood.iter_mut().min_by(|a, b| {
                a.fed
                    .partial_cmp(&b.fed)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                let amount = available.min((need_cap - hungriest.fed).max(0.0));
                hungriest.fed += amount;
                self.nest.store -= amount;
            }
        }
        let a = &mut self.ants[i];
        a.time_nursing += 1;
        a.timer = a.timer.saturating_sub(1);
        if a.timer == 0 || self.nest.brood.is_empty() {
            a.activity = Activity::Resting;
        }
    }

    fn unload(&mut self, i: usize) {
        let a = &mut self.ants[i];
        a.timer = a.timer.saturating_sub(1);
        if a.timer > 0 {
            return;
        }
        let load = a.crop;
        let quality = a.load_quality;
        a.crop = 0.0;
        a.activity = Activity::Resting;
        // Poor sources are abandoned: the memory survives with a
        // quality-dependent probability.
        let keep = self.species.site_fidelity(quality);
        if !self.rng.chance(keep) {
            self.ants[i].site = None;
        }
        // The store gains the load's sugar content.
        self.nest.store = (self.nest.store + load * quality).min(self.nest.capacity);
        self.nest.excitation += self.species.excitation_per_return * quality;
    }

    fn feed(&mut self, i: usize) {
        let a = &mut self.ants[i];
        a.timer = a.timer.saturating_sub(1);
        if a.timer > 0 {
            return;
        }
        let quality = a.load_quality;
        let lay = self.species.lay_probability(quality) * a.traits.laying;
        a.crop = self.species.crop_capacity * self.species.load_fraction(quality);
        a.steps_since_food = 0;
        a.trip_moves = 0;
        a.site = Some(Site {
            vector: a.home_vector,
            quality,
        });
        a.heading = a.heading.opposite();
        a.activity = Activity::Inbound;
        a.search_steps = 0;
        // Marks per trip also rise with concentration (Beckers et al. 1993).
        a.laying = if self.rng.chance(lay.clamp(0.0, 1.0)) {
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
        a.heading = a.heading.opposite();
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
        let pos = self.ants[i].position;
        if let Some(c) = self.world.cell_mut(pos) {
            c.occupancy = c.occupancy.saturating_sub(1);
        }
        self.outside = self.outside.saturating_sub(1);
        let leaf = self.ants[i].leaf;
        let trip = {
            let a = &mut self.ants[i];
            a.position = nest;
            a.reset_home_vector();
            a.steps_since_nest = 0;
            a.laying = None;
            a.search_steps = 0;
            if a.crop > 0.0 {
                a.activity = Activity::Unloading;
                a.timer = unloading;
                a.deliveries += 1;
                let moves = a.trip_moves as u64;
                let direct = a
                    .pickup
                    .take()
                    .map(|p| (p.chebyshev(nest) - self.world.nest_radius().max(0)).max(0))
                    .unwrap_or(0) as u64;
                Some((moves, direct))
            } else {
                a.activity = Activity::Resting;
                None
            }
        };
        if let Some((moves, direct)) = trip {
            self.stats.path.record_trip(moves, direct);
            for &node in &self.leaf_paths[leaf] {
                self.stats.path_by_node[node].record_trip(moves, direct);
            }
            self.stats.food_delivered += 1;
            let reward = self.config.reward.food_delivered;
            self.credit(leaf, reward, true);
        }
    }

    /// Upper bound on moves per tick, for large ticks or fast species.
    const MAX_MOVES_PER_TICK: usize = 6;

    fn walk(&mut self, i: usize) {
        // Speed: cells per tick, faster on a strong trail, slower when
        // loaded. Fractional speeds accumulate credit; speeds above one
        // cell per tick yield several moves per tick.
        let speed = {
            let a = &self.ants[i];
            let trail = self.world.level(a.position, Pheromone::Trail);
            let k = self.world.channel(Pheromone::Trail).k;
            let mut speed = a.traits.speed;
            if a.carrying() {
                speed *= self.species.loaded_speed_factor;
            }
            if trail > k {
                speed *= self.species.trail_speed_factor;
            }
            speed
        };
        let cap = speed.max(1.0) + 1.0;
        self.ants[i].move_credit = (self.ants[i].move_credit + speed).min(cap);
        let mut moves = 0;
        while self.ants[i].move_credit >= 1.0 && moves < Self::MAX_MOVES_PER_TICK {
            self.ants[i].move_credit -= 1.0;
            moves += 1;
            let mode = match (self.ants[i].activity, self.ants[i].search_target) {
                (Activity::Inbound, _) | (Activity::Searching, SearchTarget::Nest) => Mode::Inbound,
                _ => Mode::Outbound,
            };
            if let Some(dir) = self.decide(i, mode) {
                self.move_ant(i, dir);
            }
            if self.check_transitions(i) {
                return;
            }
        }
        self.tick_timers(i);
    }

    fn move_ant(&mut self, i: usize, dir: Direction) {
        let leaf = self.ants[i].leaf;
        let from = self.ants[i].position;
        let to = from.step(dir);
        let ring = ring_index(dir, self.ants[i].heading);
        let revisit = self.ants[i].recently_visited(to);
        self.stats.path.record_move(ring, revisit);
        for &node in &self.leaf_paths[leaf] {
            self.stats.path_by_node[node].record_move(ring, revisit);
        }
        if let Some(c) = self.world.cell_mut(from) {
            c.occupancy = c.occupancy.saturating_sub(1);
        }
        if let Some(c) = self.world.cell_mut(to) {
            c.occupancy = c.occupancy.saturating_add(1);
        }
        self.world.record_crossing(to);
        let species = &self.species;
        let (laying, strength, territory, home) = {
            let a = &mut self.ants[i];
            a.position = to;
            a.heading = dir;
            a.remember(from);
            a.trip_moves = a.trip_moves.saturating_add(1);
            a.integrate(dir, species, &mut self.rng);
            let home = species.uses_home_pheromone && a.activity == Activity::Outbound;
            (a.laying, a.lay_strength, species.territory_deposit, home)
        };
        match laying {
            Some(Pheromone::Trail) => {
                self.world
                    .deposit(to, Pheromone::Trail, species.trail_deposit * strength)
            }
            Some(Pheromone::NoEntry) => {
                self.world
                    .deposit(to, Pheromone::NoEntry, species.no_entry_deposit * strength)
            }
            Some(kind) => self.world.deposit(to, kind, strength),
            None => {}
        }
        if home {
            self.world
                .deposit(to, Pheromone::Home, species.trail_deposit);
        }
        if territory > 0.0 {
            self.world.deposit(to, Pheromone::Territory, territory);
        }
    }

    /// React to the cell just entered. Returns `true` when the ant stopped
    /// walking (it started feeding or entered the nest).
    fn check_transitions(&mut self, i: usize) -> bool {
        let pos = self.ants[i].position;
        let on_nest = self.world.is_nest(pos);
        let food_here = self
            .world
            .cell(pos)
            .map(|c| (c.food, c.quality))
            .unwrap_or((0, 0.0));
        let arrival = self.species.arrival_radius;
        let activity = self.ants[i].activity;
        let target = self.ants[i].search_target;
        match (activity, target) {
            (Activity::Outbound, _) | (Activity::Searching, SearchTarget::Food) => {
                if food_here.0 > 0 && !self.ants[i].carrying() {
                    self.start_feeding(i, pos, food_here.1 as f64);
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
                    // Lost: path integration has failed; rely on the
                    // home-range marking and the trail from here on.
                    a.activity = Activity::Inbound;
                    a.reset_home_vector();
                    a.search_steps = 0;
                }
            }
            _ => {}
        }
    }

    fn start_feeding(&mut self, i: usize, pos: Position, quality: f64) {
        if let Some(c) = self.world.cell_mut(pos) {
            c.food = c.food.saturating_sub(1);
        }
        let ticks = self.seconds_to_ticks(self.species.feeding_time(quality));
        let leaf = self.ants[i].leaf;
        let a = &mut self.ants[i];
        a.activity = Activity::Feeding;
        a.timer = ticks;
        a.load_quality = quality;
        a.pickup = Some(pos);
        a.laying = None;
        self.stats.food_picked += 1;
        let reward = self.config.reward.food_picked;
        self.credit(leaf, reward, false);
    }

    fn metabolize(&mut self, i: usize) {
        let mortality = self.config.nest.mortality;
        let starvation = self.species.starvation_s;
        let tick_s = self.tick_s;
        let store_has_food = self.nest.store > 0.0;
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
        let pos = self.ants[i].position;
        self.ants[i].alive = false;
        if outside {
            if let Some(c) = self.world.cell_mut(pos) {
                c.occupancy = c.occupancy.saturating_sub(1);
            }
            self.outside = self.outside.saturating_sub(1);
            if cause == Cause::Predation {
                self.world
                    .deposit(pos, Pheromone::Alarm, self.species.alarm_release);
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
        let consumption = inside * self.species.consumption_per_ant_per_s * self.tick_s;
        self.nest.store = (self.nest.store - consumption).max(0.0);
        self.nest.excitation *= self.excitation_retention;

        // Development and emergence.
        let dev_ticks = self.development_ticks();
        let brood_food = self.species.brood_food;
        let mut emerging = 0usize;
        self.nest.brood.retain_mut(|b| {
            b.age += 1;
            if b.age >= dev_ticks && b.fed >= brood_food {
                emerging += 1;
                false
            } else {
                true
            }
        });
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
                self.nest.brood.push(BroodItem { age: 0, fed: 0.0 });
                self.nest.eggs_laid += 1;
                self.nest.last_egg_tick = self.tick;
                self.stats.eggs += 1;
            }
        }
    }

    // ---------------------------------------------------------------
    // Movement decision
    // ---------------------------------------------------------------

    fn decide(&mut self, i: usize, mode: Mode) -> Option<Direction> {
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
        let obs = observe(&self.ants[i], &self.world, &self.species, mode);
        let policy = self.policies[leaf].clone();

        // The deterministic information of the path: scores per direction.
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
        let base = Landscape::from_world(&scores, &obs.valid, heading);

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
        let start = base.nearest_valid(0).expect("at least one valid direction");
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
            for (d, dir) in Direction::ALL.iter().enumerate() {
                let p = tempered.probs[ring_index(*dir, heading)];
                if p > 0.0 {
                    for (s, f) in score.iter_mut().zip(&obs.features[d]) {
                        *s -= p * f;
                    }
                }
            }
            let chosen_world = world_direction(chosen, heading).index();
            for (s, f) in score.iter_mut().zip(&obs.features[chosen_world]) {
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
        Some(world_direction(chosen, heading))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::EntropyControl;
    use crate::surface::Deformation;

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
    /// emerge and eggs to be laid within a short run.
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
        assert!(sim.nest().store > 0.0);
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
        assert!(s.mean_entropy() > 0.0);
        assert!(!s.log.is_empty());
        assert!(s.log.last().unwrap().delivered == s.food_delivered);
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
        assert!(h_low < 0.35, "low dial → low entropy, got {h_low}");
        assert!(h_high > 1.7, "high dial → near-uniform, got {h_high}");
        // The fixed-temperature default lands in between and reports the
        // entropy the choice function happens to have.
        let mut natural = Simulation::new(hungry_fast(), 3);
        natural.run(400);
        let h = natural.stats().mean_entropy();
        assert!(h > 0.2 && h < 1.9, "{h}");
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
    fn mortality_and_emergence() {
        let mut sim = Simulation::new(life_history(), 4);
        sim.run(1500);
        let s = sim.stats();
        assert!(s.births > 0, "brood should emerge: {s:?}");
        assert!(s.eggs > 0, "the queen should lay: {s:?}");
        assert_eq!(sim.alive(), sim.living().count());
        let occupancy: u32 = sim.world().cells().iter().map(|c| c.occupancy as u32).sum();
        assert_eq!(occupancy as usize, sim.outside());
        assert!(s.deaths_predation + s.deaths_starvation == s.deaths);
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
        assert!(rows.windows(2).all(|w| w[0].tick < w[1].tick));
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
}
