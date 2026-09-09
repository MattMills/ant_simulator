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

use crate::ant::{observe, Activity, Ant, AntId, Mode, SearchTarget, Site, Traits, View, FEATURES};
use crate::entropy::{entropy, Tempering};
use crate::geometry::{angle_of, Point, Position};
use crate::hierarchy::{EffectivePolicy, Hierarchy, HierarchySpec, NodeId};
use crate::hive::{HistoryConfig, MovementHistory, Queen, QueenConfig};
use crate::landscape::{
    ring_heading, turn_magnitude, EntropyLedger, Landscape, Sucker, RING, RING_STEP,
};
use crate::memo::{
    Leg, Memo, MemoConfig, Stop, Transit, TransitKey, TransitRecord, INSIDE, MAX_TRANSIT_LEVELS,
};
use crate::pheromone::Pheromone;
use crate::pipeline::{FrameStats, PipelineConfig};
use crate::rng::Rng;
use crate::scaling::{Phase, Profile};
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
    /// Capacity of the nest's sugar reserve per simulated worker,
    /// milligrams: the crops of the queen, the brood and any nestmates not
    /// simulated, filled by workers inside passing on their surplus. The
    /// simulated workers hold their own crops on top of it.
    pub store_capacity_mg_per_ant: f64,
    /// Initial fill of every crop and of the reserve, as a fraction.
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
            store_capacity_mg_per_ant: 0.2,
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
    /// Keep the colony's movement history (the hive's cognitive geometry);
    /// `None` keeps nothing.
    pub history: Option<HistoryConfig>,
    /// A queen who thinks through the movement history (needs `history`).
    pub mind: Option<QueenConfig>,
    /// Keep the behavioural memo (what is done where, over the quadtree;
    /// see [`crate::memo`]).
    pub memo: Option<MemoConfig>,
    /// The decision pipeline: decisions held for a horizon on invariant
    /// ground and scheduled against a frame budget (none: every step
    /// decided).
    pub pipeline: Option<PipelineConfig>,
    /// The frame the world's surfaces stand in, for falling: where a
    /// point lands is where the surface below it lies in space.
    pub frame: Option<crate::frame::Frame>,
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
            history: None,
            mind: None,
            memo: None,
            pipeline: None,
            frame: None,
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
    /// Sugar in the reserve (the crops of the queen, brood and nestmates
    /// not simulated), milligrams.
    pub store_mg: f64,
    /// Reserve capacity, milligrams.
    pub capacity_mg: f64,
    /// Nestmates the reserve stands for, in worker crops.
    pub virtual_nestmates: f64,
    /// Sugar in the crops of the workers inside, milligrams (updated each
    /// tick).
    pub crops_mg: f64,
    /// Crop capacity of the workers inside, milligrams (updated each tick).
    pub crops_capacity_mg: f64,
    /// Protein (prey) in store, milligrams.
    pub protein_mg: f64,
    /// Dead nestmates inside, waiting to be carried out.
    pub corpses: u32,
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
    /// Larvae at the last refresh.
    pub larvae_now: usize,
    /// Protein demand at the last refresh.
    pub protein_demand_now: f64,
    /// Where the nurses' line of larvae stands.
    pub feed_cursor: usize,
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

    /// Sugar held inside the nest, in workers' crops and in the reserve,
    /// milligrams.
    pub fn sugar_mg(&self) -> f64 {
        self.store_mg + self.crops_mg
    }

    /// Fill of the colony's crops and reserve together, 0 (empty) to 1
    /// (full).
    pub fn satiation(&self) -> f64 {
        let capacity = self.capacity_mg + self.crops_capacity_mg;
        if capacity <= 0.0 {
            0.0
        } else {
            (self.sugar_mg() / capacity).clamp(0.0, 1.0)
        }
    }

    /// Empty space in the reserve per nestmate it stands for, milligrams.
    pub fn reserve_deficit_per_nestmate(&self) -> f64 {
        if self.virtual_nestmates <= 0.0 {
            0.0
        } else {
            (self.capacity_mg - self.store_mg).max(0.0) / self.virtual_nestmates
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

    /// Bring the per-tick summaries of the brood up to date: the number
    /// of larvae and the protein demand (so that no worker's decision has
    /// to scan the brood).
    pub fn refresh(&mut self, larva_protein_mg: f64) {
        self.larvae_now = self.larvae();
        self.protein_demand_now = self.protein_demand(larva_protein_mg);
    }

    /// Sort the brood hungriest first, so that nurses feed from the
    /// front of the line.
    pub fn sort_by_hunger(&mut self) {
        self.brood.sort_by(|a, b| {
            a.fed_mg
                .partial_cmp(&b.fed_mg)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.feed_cursor = 0;
    }

    /// The next larva in line, if one is found within a short scan from
    /// the cursor: the line is walked round and round, so that every
    /// larva is tended in turn (fed what it still wants, its starvation
    /// clock reset), and it is re-formed hungriest first every minute.
    pub fn next_larva(&mut self) -> Option<usize> {
        let n = self.brood.len();
        if n == 0 {
            return None;
        }
        for k in 0..n.min(32) {
            let idx = (self.feed_cursor + k) % n;
            if self.brood[idx].stage == BroodStage::Larva {
                self.feed_cursor = (idx + 1) % n;
                return Some(idx);
            }
        }
        self.feed_cursor = (self.feed_cursor + 32) % n;
        None
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
    /// Outbound legs completed (departures that found food).
    pub outbound_trips: u64,
    /// Their summed path length, cells.
    pub outbound_length: f64,
    /// Their summed direct distance from the nest to the food, cells.
    pub outbound_direct: f64,
    /// Outbound legs no longer than one and a half times the direct
    /// distance: the ones a trail or a memory guided, rather than a
    /// search.
    pub ordered_trips: u64,
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
            outbound_trips: 0,
            outbound_length: 0.0,
            outbound_direct: 0.0,
            ordered_trips: 0,
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

    fn record_outbound(&mut self, length: f64, direct: f64) {
        self.outbound_trips += 1;
        self.outbound_length += length;
        self.outbound_direct += direct;
        if length <= 1.5 * direct.max(1e-9) {
            self.ordered_trips += 1;
        }
    }

    /// Share of the outbound legs that went to the food no more than one
    /// and a half times the direct distance.
    pub fn ordered_fraction(&self) -> f64 {
        if self.outbound_trips == 0 {
            0.0
        } else {
            self.ordered_trips as f64 / self.outbound_trips as f64
        }
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
        self.outbound_trips += other.outbound_trips;
        self.outbound_length += other.outbound_length;
        self.outbound_direct += other.outbound_direct;
        self.ordered_trips += other.ordered_trips;
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
    /// Mean crop fill of the workers inside.
    pub crop_fill: f64,
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
    /// Corpses picked up (in the nest or outside).
    pub corpses_moved: u64,
    /// Outbound trips abandoned without food.
    pub failed_trips: u64,
    /// Workers that died.
    pub deaths: u64,
    /// Of which taken by predators.
    pub deaths_predation: u64,
    /// Of which starved.
    pub deaths_starvation: u64,
    /// Of which killed by heat outside the nest.
    pub deaths_heat: u64,
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
    /// Sugar inside the nest (crops and reserve) at the end of the last
    /// tick, milligrams.
    pub store_mg: f64,
    /// Trophallactic contacts made by unloading foragers.
    pub unloading_contacts: u64,
    /// Sugar handed over by trophallaxis, milligrams.
    pub trophallaxis_mg: f64,
    /// Unloadings abandoned with a load still in the crop.
    pub failed_unloads: u64,
    /// Trophallaxis contacts inside the nest by the zones of the two
    /// workers (0 the brood chamber, 1 between, 2 the entrance ring),
    /// counted in both directions.
    pub nest_contacts: [[u64; 3]; 3],
    /// Corpses fetched from where they lay inside the nest by
    /// undertakers (`corpses_moved` counts every pick-up, outside too).
    pub corpses_fetched: u64,
    /// Of the decisions, those stood in for by replayed transits.
    pub decisions_replayed: u64,
    /// The decision pipeline's frames: decisions made, steps held and
    /// deferred, frames overrun.
    pub frames: FrameStats,
    /// Position fixes taken from familiar landmarks.
    pub landmark_fixes: u64,
    /// Falls from walls, rims and ceilings.
    pub falls: u64,
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

    /// Assortativity of the trophallaxis contacts by zone (Newman 2003):
    /// 1 when workers only exchange food within their own zone, 0 when
    /// zones mix at random, negative when exchanges cross zones more than
    /// chance would have them.
    pub fn contact_assortativity(&self) -> f64 {
        let total: u64 = self.nest_contacts.iter().flatten().sum();
        if total == 0 {
            return 0.0;
        }
        let e = |i: usize, j: usize| self.nest_contacts[i][j] as f64 / total as f64;
        let trace: f64 = (0..3).map(|i| e(i, i)).sum();
        let chance: f64 = (0..3)
            .map(|i| (0..3).map(|j| e(i, j)).sum::<f64>().powi(2))
            .sum();
        if (1.0 - chance).abs() < 1e-12 {
            0.0
        } else {
            (trace - chance) / (1.0 - chance)
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
    Heat,
}

/// The workers inside one zone of the nest.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ZoneProfile {
    /// Zone: 0 the brood chamber, 1 between, 2 the entrance ring.
    pub class: usize,
    /// Workers inside in this class.
    pub workers: usize,
    /// Their mean age, seconds.
    pub mean_age_s: f64,
    /// Their mean crop fill.
    pub mean_crop_fill: f64,
    /// How many of them are nursing.
    pub nursing: usize,
}

/// Pearson correlation of two series.
fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let mx = x[..n].iter().sum::<f64>() / n as f64;
    let my = y[..n].iter().sum::<f64>() / n as f64;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x[..n].iter().zip(&y[..n]) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx).powi(2);
        syy += (b - my).powi(2);
    }
    if sxx <= 1e-18 || syy <= 1e-18 {
        0.0
    } else {
        sxy / (sxx * syy).sqrt()
    }
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
    /// For each leaf, which distinct policy it acts through (transit
    /// kernels are keyed by it).
    policy_classes: Vec<u8>,
    /// Direct distances from the nest to pickup cells along the geodesic
    /// round the walls, remembered per cell.
    geodesics: std::collections::HashMap<Position, f64>,
    /// The decision pipeline, if decisions are coarse-grained in time.
    pipeline: Option<PipelineConfig>,
    /// The frame the surfaces stand in, if any.
    frame: Option<crate::frame::Frame>,
    /// Probability the last decision gave to holding the heading.
    straight_prob: f64,
    leaf_paths: Vec<Vec<NodeId>>,
    dirty: bool,
    stats: Stats,
    trace: Trace,
    surface: Vec<SurfaceRow>,
    history: Option<MovementHistory>,
    mind: Option<Queen>,
    memo: Option<Memo>,
    profile: Profile,
    decisions_time: std::time::Duration,
    nest: Nest,
    nest_cells: Vec<Position>,
    /// Living workers inside the nest, by world cell index.
    inside_by_cell: Vec<Vec<usize>>,
    /// Cells with an entry in `inside_by_cell` since the last indexing.
    inside_cells: Vec<usize>,
    /// Workers inside at the last indexing.
    inside_total: usize,
    /// Workers nursing right now.
    nurses: usize,
    alive: usize,
    outside: usize,
    next_leaf: usize,
    tick: u64,
    tick_s: f64,
    /// Persistence length of the heading in cells.
    persistence_cells: f64,
    decision_prob: f64,
    hazard_per_tick: f64,
    /// Heat hazard per tick outside at the current temperature.
    heat_hazard_per_tick: f64,
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
        let (world_width, world_height) = (world.width(), world.height());
        let pipeline = config.pipeline.clone();
        let frame = config.frame.clone();
        let memo = config
            .memo
            .clone()
            .map(|m| Memo::new(world_width, world_height, tick_s, m).with_transits(&world));
        let hierarchy_for_queen = hierarchy.clone();
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
            virtual_nestmates: capacity / species.crop_sugar_capacity_mg().max(1e-9),
            crops_mg: 0.0,
            crops_capacity_mg: 0.0,
            protein_mg: 0.0,
            corpses: 0,
            brood: Vec::new(),
            excitation: 0.0,
            eggs_laid: 0,
            emerged: 0,
            larvae_starved: 0,
            last_egg_tick: 0,
            larvae_now: 0,
            protein_demand_now: 0.0,
            feed_cursor: 0,
        };
        let nest_cells = world.nest_cells();
        let temperature_c = config.environment.temperature(0.0);
        let mut sim = Simulation {
            policy_classes: Self::policy_classes(&hierarchy.compile()),
            geodesics: std::collections::HashMap::new(),
            pipeline,
            frame,
            straight_prob: 0.0,
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
            history: config
                .history
                .clone()
                .map(|h| MovementHistory::new(world_width, world_height, tick_s, h)),
            mind: config
                .mind
                .clone()
                .map(|m| Queen::new(m, &hierarchy_for_queen)),
            profile: Profile::default(),
            memo,
            decisions_time: std::time::Duration::ZERO,
            nest,
            nest_cells,
            inside_by_cell: vec![Vec::new(); world_width * world_height],
            inside_cells: Vec::new(),
            inside_total: 0,
            nurses: 0,
            alive: 0,
            outside: 0,
            next_leaf: 0,
            tick: 0,
            tick_s,
            persistence_cells: species.heading_persistence_cm / cell_cm.max(1e-9),
            decision_prob: (tick_s / species.decision_interval_s.max(1e-9)).min(1.0),
            hazard_per_tick: 1.0 - (-species.forager_hazard_per_s * tick_s).exp(),
            heat_hazard_per_tick: 0.0,
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
            sim.place_by_zone(id);
        }
        let brood_items =
            (sim.config.nest.initial_brood_per_ant * sim.config.ants as f64).round() as usize;
        for _ in 0..brood_items {
            let item = sim.random_brood_item();
            sim.nest.brood.push(item);
        }
        sim.nest.sort_by_hunger();
        sim.nest.refresh(sim.species.larva_protein_mg);
        sim.stats.alive = sim.alive;
        sim.stats.store_mg = sim.nest.sugar_mg();
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

    // ---------------------------------------------------------------
    // The nest interior
    // ---------------------------------------------------------------

    /// Depth of a nest cell: 0 at the centre (the brood), 1 on the outer
    /// ring (the entrance), by Chebyshev distance over the nest radius.
    pub fn depth_of(&self, cell: Position) -> f64 {
        let radius = self.world.nest_radius();
        if radius <= 0 {
            return 0.0;
        }
        let nest = self.world.nest();
        let d = (cell.x - nest.x).abs().max((cell.y - nest.y).abs());
        (d as f64 / radius as f64).min(1.0)
    }

    /// Zone of a depth: 0 the brood chamber, 1 between, 2 the entrance
    /// ring.
    pub fn zone_class(&self, depth: f64) -> usize {
        if depth <= self.species.brood_depth {
            0
        } else if depth < 1.0 - 1e-9 {
            1
        } else {
            2
        }
    }

    /// How much of the brood's demand a worker at a cell perceives: all
    /// of it in the brood chamber, fading over the species' reach with
    /// the distance from it. Workers respond to the tasks they
    /// encounter, so where they stand decides what they do (foraging for
    /// work, Tofts & Franks 1992).
    pub fn brood_proximity(&self, cell: Position) -> f64 {
        let depth = (self.depth_of(cell) - self.species.brood_depth).max(0.0);
        let distance_cm = depth * self.world.nest_radius().max(0) as f64 * self.world.cell_cm();
        (-distance_cm / self.species.brood_reach_cm.max(1e-9)).exp()
    }

    /// The depth a worker keeps to: callow workers sit with the brood and
    /// drift outward with age, reaching the entrance ring after twice the
    /// maturation time, each with its own offset.
    pub fn zone_depth(&self, ant: &Ant) -> f64 {
        let s = &self.species;
        let age_s = ant.age as f64 * self.tick_s;
        let drift = (age_s / (2.0 * s.maturation_s).max(1e-9)).min(1.0);
        (s.callow_depth + (1.0 - s.callow_depth) * drift + ant.traits.zone_offset).clamp(0.0, 1.0)
    }

    /// Correlation of age with depth over the workers inside: positive
    /// when the old sit outward and the young with the brood.
    pub fn age_depth_correlation(&self) -> f64 {
        let (ages, depths): (Vec<f64>, Vec<f64>) = self
            .living()
            .filter(|a| a.is_inside())
            .map(|a| (a.age as f64 * self.tick_s, self.depth_of(a.cell())))
            .unzip();
        correlation(&ages, &depths)
    }

    /// The workers inside by zone.
    pub fn nest_profile(&self) -> [ZoneProfile; 3] {
        let mut out = [ZoneProfile::default(); 3];
        for (k, z) in out.iter_mut().enumerate() {
            z.class = k;
        }
        let mut ages = [0.0; 3];
        let mut fills = [0.0; 3];
        for a in self.living().filter(|a| a.is_inside()) {
            let k = self.zone_class(self.depth_of(a.cell()));
            out[k].workers += 1;
            ages[k] += a.age as f64 * self.tick_s;
            fills[k] += a.crop_fill();
            if a.activity == Activity::Nursing {
                out[k].nursing += 1;
            }
        }
        for k in 0..3 {
            if out[k].workers > 0 {
                out[k].mean_age_s = ages[k] / out[k].workers as f64;
                out[k].mean_crop_fill = fills[k] / out[k].workers as f64;
            }
        }
        out
    }

    /// The nest interior as text, one glyph per nest cell: `u` a forager
    /// unloading, `n` a nurse, `v` a worker fetching a corpse or leaving,
    /// `w` a worker at rest, `+` a corpse with nobody on it, `:` an empty
    /// cell of the brood chamber, `.` an empty cell elsewhere.
    pub fn render_nest(&self) -> String {
        let nest = self.world.nest();
        let r = self.world.nest_radius().max(0);
        let mut out = String::new();
        for y in nest.y - r..=nest.y + r {
            for x in nest.x - r..=nest.x + r {
                let cell = Position::new(x, y);
                let corpses = self.world.cell(cell).map(|c| c.corpses).unwrap_or(0);
                let rank = self
                    .world
                    .index(cell)
                    .map(|i| {
                        self.inside_by_cell[i]
                            .iter()
                            .map(|&j| match self.ants[j].activity {
                                Activity::Unloading => 4,
                                Activity::Nursing => 3,
                                Activity::Fetching | Activity::Leaving => 2,
                                _ => 1,
                            })
                            .max()
                            .unwrap_or(0)
                    })
                    .unwrap_or(0);
                let chamber = self.depth_of(cell) <= self.species.brood_depth;
                out.push(match rank {
                    4 => 'u',
                    3 => 'n',
                    2 => 'v',
                    1 => 'w',
                    _ if corpses > 0 => '+',
                    _ if chamber => ':',
                    _ => '.',
                });
            }
            out.push('\n');
        }
        out
    }

    /// Rebuild the index of workers inside by cell.
    fn index_inside(&mut self) {
        for &k in &self.inside_cells {
            self.inside_by_cell[k].clear();
        }
        self.inside_cells.clear();
        self.inside_total = 0;
        for i in 0..self.ants.len() {
            let a = &self.ants[i];
            if a.alive && a.is_inside() {
                if let Some(k) = self.world.index(a.cell()) {
                    if self.inside_by_cell[k].is_empty() {
                        self.inside_cells.push(k);
                    }
                    self.inside_by_cell[k].push(i);
                    self.inside_total += 1;
                }
            }
        }
    }

    /// Living workers inside the nest at the last indexing.
    fn inside_count(&self) -> usize {
        self.inside_total
    }

    /// One worker inside within reach of worker `i`, drawn uniformly from
    /// those on its cell and the eight around it that pass `ok` (a few
    /// draws, so that a crowded nest costs no more per contact than an
    /// empty one).
    fn random_neighbour_inside(&mut self, i: usize, ok: impl Fn(&Ant) -> bool) -> Option<usize> {
        let here = self.ants[i].cell();
        let mut cells = [(0usize, 0usize); 9];
        let mut n = 0;
        let mut total = 0usize;
        for dy in -1..=1 {
            for dx in -1..=1 {
                if let Some(k) = self.world.index(here.offset(dx, dy)) {
                    let count = self.inside_by_cell[k].len();
                    if count > 0 {
                        cells[n] = (k, count);
                        n += 1;
                        total += count;
                    }
                }
            }
        }
        if total <= 1 {
            return None;
        }
        for _ in 0..4 {
            let mut pick = self.rng.below(total);
            let mut chosen = None;
            for &(k, count) in &cells[..n] {
                if pick < count {
                    chosen = Some(self.inside_by_cell[k][pick]);
                    break;
                }
                pick -= count;
            }
            if let Some(j) = chosen {
                if j != i && self.ants[j].alive && ok(&self.ants[j]) {
                    return Some(j);
                }
            }
        }
        None
    }

    /// Count a trophallaxis contact between two workers inside.
    fn record_contact(&mut self, i: usize, j: usize) {
        let ci = self.zone_class(self.depth_of(self.ants[i].cell()));
        let cj = self.zone_class(self.depth_of(self.ants[j].cell()));
        self.stats.nest_contacts[ci][cj] += 1;
        self.stats.nest_contacts[cj][ci] += 1;
    }

    /// Corpses lying on nest cells.
    fn nest_corpses(&self) -> u32 {
        self.nest_cells
            .iter()
            .map(|&c| self.world.cell(c).map(|c| c.corpses as u32).unwrap_or(0))
            .sum()
    }

    /// The nest cell with a corpse nearest to `from`.
    fn nearest_corpse_cell(&self, from: Position) -> Option<Position> {
        self.nest_cells
            .iter()
            .copied()
            .filter(|&c| self.world.cell(c).map(|c| c.corpses > 0).unwrap_or(false))
            .min_by_key(|&c| (c.x - from.x).abs().max((c.y - from.y).abs()))
    }

    /// A cell of the entrance ring nearest to `from` (any of the nearest,
    /// at random).
    fn nearest_exit(&mut self, from: Position) -> Position {
        let mut best = i32::MAX;
        let mut candidates = Vec::new();
        for &c in &self.nest_cells {
            if self.depth_of(c) < 1.0 - 1e-9 {
                continue;
            }
            let d = (c.x - from.x).abs().max((c.y - from.y).abs());
            if d < best {
                best = d;
                candidates.clear();
            }
            if d == best {
                candidates.push(c);
            }
        }
        if candidates.is_empty() {
            from
        } else {
            candidates[self.rng.below(candidates.len())]
        }
    }

    /// Put a worker at a nest cell of its zone's depth.
    fn place_by_zone(&mut self, i: usize) {
        let zone = self.zone_depth(&self.ants[i]);
        let mut best = f64::INFINITY;
        let mut candidates = Vec::new();
        for &c in &self.nest_cells {
            let d = (self.depth_of(c) - zone).abs();
            if d < best - 1e-9 {
                best = d;
                candidates.clear();
            }
            if (d - best).abs() <= 1e-9 {
                candidates.push(c);
            }
        }
        if !candidates.is_empty() {
            let c = candidates[self.rng.below(candidates.len())];
            self.ants[i].position = Point::center_of(c);
        }
    }

    /// One tick of walking inside the nest: with the probability that the
    /// inside walking speed gives, step to one of the neighbouring nest
    /// cells (or stay), drawn by a soft preference for the target depth
    /// (the worker's zone; the brood for a nurse; near the entrance for a
    /// forager unloading) or for the goal cell of a worker fetching a
    /// corpse or leaving.
    fn walk_inside(&mut self, i: usize) {
        if self.world.nest_radius() <= 0 {
            return;
        }
        let speed_cells =
            self.species.nest_speed_cm_s * self.tick_s / self.world.cell_cm().max(1e-9);
        if !self
            .rng
            .chance((speed_cells * self.activity_factor).min(1.0))
        {
            return;
        }
        let from = self.ants[i].cell();
        let activity = self.ants[i].activity;
        let goal = self.ants[i].goal;
        let zone = self.zone_depth(&self.ants[i]);
        let spread = self.species.zone_spread.max(1e-3);
        let mut candidates: Vec<(Position, f64)> = Vec::with_capacity(9);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let c = from.offset(dx, dy);
                if !self.world.is_nest(c) {
                    continue;
                }
                let depth = self.depth_of(c);
                let score = match (activity, goal) {
                    // Anywhere in the brood chamber will do for a nurse.
                    (Activity::Nursing, _) => -(depth - self.species.brood_depth).max(0.0) / spread,
                    (Activity::Unloading, _) => {
                        -(depth - self.species.unloading_depth).abs() / spread
                    }
                    (Activity::Fetching | Activity::Leaving, Some(g)) => {
                        -((c.x - g.x).abs().max((c.y - g.y).abs()) as f64) / spread
                    }
                    _ => -(depth - zone).abs() / spread,
                };
                candidates.push((c, score));
            }
        }
        if candidates.is_empty() {
            return;
        }
        let top = candidates
            .iter()
            .map(|c| c.1)
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = candidates.iter().map(|c| (c.1 - top).exp()).collect();
        let total: f64 = weights.iter().sum();
        let mut u = self.rng.next_f64() * total;
        let mut chosen = candidates[0].0;
        for (c, w) in candidates.iter().zip(&weights) {
            if u < *w {
                chosen = c.0;
                break;
            }
            u -= w;
        }
        if chosen == from {
            return;
        }
        let from_point = self.ants[i].position;
        let to_point = Point::center_of(chosen);
        let (dx, dy) = from_point.to(to_point);
        let direction = crate::geometry::angle_of(dx, dy);
        let turn = crate::geometry::wrap_angle(direction - self.ants[i].heading);
        if let Some(h) = &mut self.history {
            h.record(from_point, to_point, turn);
        }
        if let (Some(a), Some(b)) = (self.world.index(from), self.world.index(chosen)) {
            self.inside_by_cell[a].retain(|&j| j != i);
            if self.inside_by_cell[b].is_empty() {
                self.inside_cells.push(b);
            }
            self.inside_by_cell[b].push(i);
        }
        let a = &mut self.ants[i];
        a.position = to_point;
        a.heading = direction;
    }

    /// The cell of the entrance ring that faces a remembered site: a
    /// forager that knows where it is going leaves on that side.
    fn exit_towards(&self, site: (f64, f64)) -> Option<Position> {
        let len = (site.0 * site.0 + site.1 * site.1).sqrt();
        if len <= 1e-9 {
            return None;
        }
        let nest = Point::center_of(self.world.nest());
        let radius = self.world.nest_radius() as f64;
        let aim = Point::new(
            nest.x + site.0 / len * radius,
            nest.y + site.1 / len * radius,
        );
        self.nest_cells
            .iter()
            .copied()
            .filter(|&c| self.depth_of(c) >= 1.0 - 1e-9)
            .min_by(|&a, &b| {
                let da = Point::center_of(a).distance(aim);
                let db = Point::center_of(b).distance(aim);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Set out for the entrance to go outside: by the ring cell facing a
    /// remembered site, or the nearest one; a worker in a nest without an
    /// interior leaves at once.
    fn start_leaving(&mut self, i: usize) {
        let here = self.ants[i].cell();
        if self.world.nest_radius() <= 0 {
            self.depart(i);
            return;
        }
        let exit = match self.ants[i].site.map(|s| s.vector) {
            Some(v) if !self.ants[i].corpse => self.exit_towards(v),
            _ => None,
        }
        .unwrap_or_else(|| self.nearest_exit(here));
        if exit == here {
            self.depart(i);
            return;
        }
        let a = &mut self.ants[i];
        a.activity = Activity::Leaving;
        a.goal = Some(exit);
    }

    /// A worker on its way to the entrance leaves once it stands on the
    /// ring cell it was making for (or on any ring cell, when it has no
    /// goal).
    fn leave(&mut self, i: usize) {
        let here = self.ants[i].cell();
        let arrived = match self.ants[i].goal {
            Some(goal) => here == goal,
            None => self.depth_of(here) >= 1.0 - 1e-9,
        };
        if self.world.nest_radius() <= 0 || arrived {
            self.depart(i);
        }
    }

    /// A worker on its way to a corpse picks it up where it lies and sets
    /// out for the entrance with it; if someone else took it, it looks
    /// for another or goes back to rest.
    fn fetch(&mut self, i: usize) {
        let Some(goal) = self.ants[i].goal else {
            self.ants[i].activity = Activity::Resting;
            return;
        };
        let here = self.ants[i].cell();
        if here != goal {
            return;
        }
        if self.world.take_corpse(here) {
            self.ants[i].corpse = true;
            self.stats.corpses_moved += 1;
            self.stats.corpses_fetched += 1;
            self.start_leaving(i);
        } else {
            match self.nearest_corpse_cell(here) {
                Some(c) => self.ants[i].goal = Some(c),
                None => {
                    let a = &mut self.ants[i];
                    a.goal = None;
                    a.activity = Activity::Resting;
                }
            }
        }
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
        let crop_capacity = self.species.crop_sugar_capacity_mg() * traits.size;
        let mut ant = Ant::new(
            id,
            self.world.nest(),
            heading,
            leaf,
            traits,
            self.species.starvation_s,
            crop_capacity,
        );
        ant.sugar_mg = crop_capacity * self.config.nest.initial_satiation.clamp(0.0, 1.0);
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

    /// The world, mutably (to place corpses or food during an experiment).
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
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
        self.policy_classes = Self::policy_classes(&self.policies);
        self.dirty = false;
    }

    /// Number each distinct policy among the leaves' (identical leaves
    /// share one).
    fn policy_classes(policies: &[EffectivePolicy]) -> Vec<u8> {
        let mut distinct: Vec<&EffectivePolicy> = Vec::new();
        policies
            .iter()
            .map(|p| {
                let class = match distinct.iter().position(|d| *d == p) {
                    Some(k) => k,
                    None => {
                        distinct.push(p);
                        distinct.len() - 1
                    }
                };
                class.min(u8::MAX as usize) as u8
            })
            .collect()
    }

    /// The colony's movement history, if kept.
    pub fn history(&self) -> Option<&MovementHistory> {
        self.history.as_ref()
    }

    /// The behavioural memo, if kept (its nodes above the leaves are
    /// composed at the queen's epochs and by [`Simulation::extract_memo`]).
    pub fn memo(&self) -> Option<&Memo> {
        self.memo.as_ref()
    }

    /// A composed copy of the behavioural memo, as an object of its own.
    pub fn extract_memo(&self) -> Option<Memo> {
        self.memo.as_ref().map(|m| m.extract())
    }

    /// Note a stop in the memo at the ant's cell, and close any transit
    /// being recorded as one that ended inside its node.
    fn note_stop(&mut self, i: usize, stop: Stop) {
        let cell = self.ants[i].cell();
        if let Some(m) = &mut self.memo {
            m.record_stop(cell, stop);
        }
        self.close_records(i, INSIDE, 0.0);
        self.ants[i].transit = None;
    }

    /// Lay pheromone on a cell and note it in the memo and in the
    /// transits being recorded.
    fn lay(
        world: &mut World,
        memo: Option<&mut Memo>,
        records: &mut [Option<TransitRecord>],
        cell: Position,
        kind: Pheromone,
        amount: f64,
    ) {
        world.deposit(cell, kind, amount);
        if let Some(m) = memo {
            m.record_deposit(cell, kind, amount);
        }
        for r in records.iter_mut().flatten() {
            r.deposits[kind.index()] += amount as f32;
        }
    }

    // ---------------------------------------------------------------
    // Memoized transits
    // ---------------------------------------------------------------

    /// The level of the memo's nodes, when transits are memoized.
    fn transit_level(&self) -> Option<u8> {
        self.memo
            .as_ref()
            .and_then(|m| m.transits.as_ref())
            .map(|t| t.level())
    }

    /// The leg an ant is on, as a transit key codes it.
    fn leg_code(&self, i: usize) -> u8 {
        match self.ants[i].activity {
            Activity::Searching => 2,
            Activity::Inbound => 1,
            _ => 0,
        }
    }

    /// The side of a node's rectangle that a cell outside it lies beyond.
    fn side_towards(rect: (usize, usize, usize, usize), cell: Position) -> u8 {
        let (x0, y0, x1, y1) = rect;
        let dn = y0 as i32 - cell.y;
        let de = cell.x - (x1 as i32 - 1);
        let ds = cell.y - (y1 as i32 - 1);
        let dw = x0 as i32 - cell.x;
        let mut best = (dn, 0u8);
        for (d, side) in [(de, 1u8), (ds, 2u8), (dw, 3u8)] {
            if d > best.0 {
                best = (d, side);
            }
        }
        best.1
    }

    /// Position along a side, 0 to 1, of a point.
    fn along_side(rect: (usize, usize, usize, usize), side: u8, p: Point) -> f32 {
        let (x0, y0, x1, y1) = rect;
        let t = match side {
            0 | 2 => (p.x - x0 as f64) / (x1 - x0).max(1) as f64,
            _ => (p.y - y0 as f64) / (y1 - y0).max(1) as f64,
        };
        t.clamp(0.0, 1.0) as f32
    }

    /// The point just across a side of a node, at a position along it.
    fn across_side(rect: (usize, usize, usize, usize), side: u8, along: f32) -> Point {
        let (x0, y0, x1, y1) = rect;
        let along = along as f64;
        match side {
            0 => Point::new(x0 as f64 + along * (x1 - x0) as f64, y0 as f64 - 0.5),
            1 => Point::new(x1 as f64 + 0.5, y0 as f64 + along * (y1 - y0) as f64),
            2 => Point::new(x0 as f64 + along * (x1 - x0) as f64, y1 as f64 + 0.5),
            _ => Point::new(x0 as f64 - 0.5, y0 as f64 + along * (y1 - y0) as f64),
        }
    }

    // ---------------------------------------------------------------
    // The decision pipeline

    /// Plan the frame: with a budget, count the decisions due and grant
    /// what is left of the budget to the pending ones, earliest deadline
    /// first and in the hierarchy's order among equals.
    fn plan_frame(&mut self) {
        let Some(cfg) = &self.pipeline else {
            return;
        };
        let budget = cfg.budget;
        self.stats.frames.frames += 1;
        if budget == 0 {
            return;
        }
        let tick = self.tick;
        let mut due = 0usize;
        let mut pending: Vec<(u64, usize, usize)> = Vec::new();
        for (i, a) in self.ants.iter_mut().enumerate() {
            a.granted = false;
            if !a.alive
                || a.is_inside()
                || !a.activity.is_moving()
                || a.transit.is_some()
                || tick < a.hold_until
            {
                continue;
            }
            if tick >= a.deadline || a.activity != a.hold_activity {
                due += 1;
            } else {
                pending.push((a.deadline, a.leaf, i));
            }
        }
        if due > budget {
            self.stats.frames.overrun += 1;
        }
        let left = budget.saturating_sub(due);
        if left == 0 {
            return;
        }
        if pending.len() > left {
            pending.select_nth_unstable(left - 1);
            pending.truncate(left);
        }
        for (_, _, i) in pending {
            self.ants[i].granted = true;
        }
    }

    /// What the pipeline says about an ant's next step: hold its heading,
    /// wait for the budget, or decide.
    fn pipeline_gate(&self, i: usize, step: f64) -> Gate {
        let Some(cfg) = &self.pipeline else {
            return Gate::Decide;
        };
        let a = &self.ants[i];
        if self.tick >= a.deadline || a.activity != a.hold_activity {
            return Gate::Decide;
        }
        // A step taken without looking must be clear.
        let ahead = a.position.advanced(a.heading, step);
        if !self.world.segment_passable(a.position, ahead) {
            return Gate::Decide;
        }
        if self.tick < a.hold_until {
            Gate::Hold
        } else if cfg.budget == 0 || a.granted {
            Gate::Decide
        } else {
            Gate::Defer
        }
    }

    /// After a decision: how long the ant may hold its heading, from
    /// the invariance of the flow through its node and the probability
    /// the decision itself gave to holding it, and the deadline of its
    /// next decision.
    fn set_horizon(&mut self, i: usize) {
        let Some(cfg) = &self.pipeline else {
            return;
        };
        let (cell, searching, activity) = {
            let a = &self.ants[i];
            (a.cell(), a.activity == Activity::Searching, a.activity)
        };
        let invariance = match &self.history {
            Some(h) => {
                let level = self.transit_level().unwrap_or(h.sector_level());
                match h.key_of(cell, level) {
                    Some(key) => {
                        let departure = h.node_variance(key);
                        if departure.is_finite() {
                            (1.0 - departure.min(1.0)).max(0.0)
                        } else {
                            0.0
                        }
                    }
                    None => 0.0,
                }
            }
            None => 0.0,
        };
        let h = crate::pipeline::horizon(cfg, invariance, self.straight_prob, searching);
        let slack = (h as f64 * cfg.slack.max(0.0)).floor() as u64;
        let a = &mut self.ants[i];
        a.hold_until = self.tick + h as u64;
        a.deadline = a.hold_until + slack;
        a.hold_activity = activity;
        self.stats.frames.served += 1;
        self.stats.frames.horizon_sum += h as u64;
        self.stats.frames.horizons += 1;
    }

    /// An ant loses its grip and falls to the surface below it in space,
    /// which on the net is a walk down to where it lands: its path
    /// integration takes the drop as such, it lands facing any way,
    /// stunned for a moment, and whatever transit it was in ends inside.
    fn fall(&mut self, i: usize) {
        let from = self.ants[i].position;
        let Some(target) = self.frame.as_ref().and_then(|f| f.fall_target(from)) else {
            return;
        };
        if !self.world.is_passable(target.cell()) {
            return;
        }
        let from_cell = from.cell();
        let to_cell = target.cell();
        if from_cell != to_cell {
            if let Some(c) = self.world.cell_mut(from_cell) {
                c.occupancy = c.occupancy.saturating_sub(1);
            }
            if let Some(c) = self.world.cell_mut(to_cell) {
                c.occupancy = c.occupancy.saturating_add(1);
            }
            self.world.record_crossing(to_cell);
        }
        let stun = self.seconds_to_ticks(self.species.fall_stun_s);
        let heading = self.rng.range(-std::f64::consts::PI, std::f64::consts::PI);
        let (dx, dy) = from.to(target);
        let species = &self.species;
        let a = &mut self.ants[i];
        a.position = target;
        a.integrate(dx, dy, species, &mut self.rng);
        a.heading = heading;
        a.stun = stun;
        a.remember(from_cell);
        self.close_records(i, INSIDE, 0.0);
        self.ants[i].transit = None;
        self.stats.falls += 1;
    }

    /// The decision pipeline's configuration, if any.
    pub fn pipeline(&self) -> Option<&PipelineConfig> {
        self.pipeline.as_ref()
    }

    // ---------------------------------------------------------------
    // Memoized transits, level by level

    /// The levels at which transits are memoized, finest first.
    fn transit_levels(&self) -> Vec<u8> {
        self.memo
            .as_ref()
            .and_then(|m| m.transits.as_ref())
            .map(|t| t.levels().to_vec())
            .unwrap_or_default()
    }

    /// Close the record of one slot, if any, as an outcome.
    fn close_slot(&mut self, i: usize, slot: usize, exit_side: u8, exit_along: f32) {
        let Some(record) = self.ants[i].records[slot].take() else {
            return;
        };
        let heading = self.ants[i].heading;
        let outcome = record.close(self.tick, exit_side, exit_along, heading);
        if let Some(t) = self.memo.as_mut().and_then(|m| m.transits.as_mut()) {
            t.record(record.key, outcome, &mut self.rng);
        }
    }

    /// Close every record being kept, with one exit for all (a stop
    /// inside).
    fn close_records(&mut self, i: usize, exit_side: u8, exit_along: f32) {
        for slot in 0..MAX_TRANSIT_LEVELS {
            self.close_slot(i, slot, exit_side, exit_along);
        }
    }

    /// An ant leaves the node of one slot for `to_cell`: close that
    /// record by the side it left through.
    fn leave_slot(&mut self, i: usize, slot: usize, to_cell: Position, to: Point) {
        let Some(record) = self.ants[i].records[slot] else {
            return;
        };
        let rect = self
            .memo
            .as_ref()
            .map(|m| m.rect(record.key.node))
            .unwrap_or((0, 0, 0, 0));
        let side = Self::side_towards(rect, to_cell);
        let along = Self::along_side(rect, side, to);
        self.close_slot(i, slot, side, along);
    }

    /// How many slots, finest first, a move from one cell to another
    /// crosses out of the nodes of: a prefix of the slots, since a
    /// coarser node's boundary is part of every finer one's.
    fn crossed_slots(&self, from_cell: Position, to_cell: Position) -> usize {
        let Some(m) = &self.memo else {
            return 0;
        };
        let mut crossed = 0;
        for (slot, &level) in self.transit_levels().iter().enumerate() {
            if m.key_of(from_cell, level) != m.key_of(to_cell, level) {
                crossed = slot + 1;
            }
        }
        crossed
    }

    /// A move from one cell into another that crosses out of the nodes
    /// of some slots: close their records by the sides left through,
    /// then enter the new nodes. Returns whether a replay began.
    fn cross(&mut self, i: usize, from_cell: Position, to_cell: Position, to: Point) -> bool {
        let crossed = self.crossed_slots(from_cell, to_cell);
        if crossed == 0 {
            return false;
        }
        for slot in 0..crossed.min(MAX_TRANSIT_LEVELS) {
            self.leave_slot(i, slot, to_cell, to);
        }
        self.enter_node(i, crossed, from_cell, to_cell, to)
    }

    /// An ant enters the nodes of the first `crossed` slots at `to_cell`
    /// from `from_cell`: replay a mature kernel's outcome at the coarsest
    /// of them that is plain and invariant, if the ant qualifies, or
    /// open records for those that are plain. Returns whether a replay
    /// began.
    fn enter_node(
        &mut self,
        i: usize,
        crossed: usize,
        from_cell: Position,
        to_cell: Position,
        to: Point,
    ) -> bool {
        let levels = self.transit_levels();
        let crossed = crossed.min(levels.len()).min(MAX_TRANSIT_LEVELS);
        if crossed == 0 || self.ants[i].corpse || !self.ants[i].activity.is_moving() {
            return false;
        }
        let (heading, leaf, laden, searching) = {
            let a = &self.ants[i];
            (
                a.heading,
                a.leaf,
                a.carrying(),
                a.activity == Activity::Searching,
            )
        };
        let tempering = if searching {
            self.policies[leaf]
                .tempering
                .heated(self.species.search_temperature_factor)
        } else {
            self.policies[leaf].tempering
        };
        let k = self.world.channel(Pheromone::Trail).k.max(1e-9);
        let trail = self.world.level(to_cell, Pheromone::Trail) / k;
        let crowded = self
            .world
            .cell(to_cell)
            .map(|c| c.crowding(false) > 0.5)
            .unwrap_or(false);
        let leg = self.leg_code(i);
        let policy = self.policy_classes.get(leaf).copied().unwrap_or(0);
        let dial = TransitKey::dial_class(tempering);
        let context = TransitKey::context_class(trail, crowded);
        let heading_class = TransitKey::heading_class(heading);
        let (explore, invariance, coarse_coherence) = self
            .memo
            .as_ref()
            .and_then(|m| m.transits.as_ref())
            .map(|t| {
                let c = t.config();
                (c.explore, c.invariance, c.coarse_coherence)
            })
            .unwrap_or((0.0, 0.0, 0.0));
        // The keys of the plain nodes entered, one per slot.
        let mut keys: Vec<Option<Entry>> = vec![None; crossed];
        for (slot, key) in keys.iter_mut().enumerate() {
            let level = levels[slot];
            let Some(node) = self.memo.as_ref().and_then(|m| m.key_of(to_cell, level)) else {
                continue;
            };
            let plain = self
                .memo
                .as_ref()
                .and_then(|m| m.transits.as_ref())
                .map(|t| t.is_plain(node))
                .unwrap_or(false);
            if !plain {
                continue;
            }
            let rect = self
                .memo
                .as_ref()
                .map(|m| m.rect(node))
                .unwrap_or((0, 0, 0, 0));
            let side = Self::side_towards(rect, from_cell);
            *key = Some((
                TransitKey {
                    node,
                    side,
                    heading: heading_class,
                    leg,
                    laden,
                    policy,
                    dial,
                    context,
                },
                rect,
            ));
        }
        // A share of entries is always simulated in full, so that the
        // kernels keep learning and a change in the ground shows.
        let exploring = explore > 0.0 && self.rng.chance(explore);
        if !exploring {
            // The coarsest plain node first: corpses on the ground change
            // what happens, and so does a field that is not what it
            // always was.
            for slot in (0..crossed).rev() {
                let Some((key, rect)) = keys[slot] else {
                    continue;
                };
                if self.world.corpses_in(rect) > 0 {
                    continue;
                }
                let invariant = match &self.history {
                    Some(h) => h
                        .key_of(to_cell, key.node.level)
                        .map(|hk| h.node_variance(hk) <= invariance)
                        .unwrap_or(false),
                    None => true,
                };
                if !invariant {
                    continue;
                }
                let coherence = if slot > 0 { coarse_coherence } else { 0.0 };
                let outcome = self
                    .memo
                    .as_ref()
                    .and_then(|m| m.transits.as_ref())
                    .and_then(|t| t.sample(&key, coherence, &mut self.rng));
                let Some(outcome) = outcome else {
                    continue;
                };
                let exit = Self::across_side(rect, outcome.exit_side, outcome.exit_along);
                if !self.world.has_clearance(exit) {
                    continue;
                }
                if let Some(t) = self.memo.as_mut().and_then(|m| m.transits.as_mut()) {
                    t.note_replay(&outcome, key.node.level);
                }
                // Finer nodes are not recorded while a coarser one is
                // replayed.
                for record in self.ants[i].records.iter_mut().take(slot) {
                    *record = None;
                }
                let until = self.tick + outcome.ticks.max(1) as u64;
                self.ants[i].transit = Some(Transit {
                    until,
                    entry: to,
                    exit,
                    outcome,
                    key,
                });
                return true;
            }
        }
        for (slot, key) in keys.iter().enumerate() {
            if let Some((key, rect)) = key {
                let side = (rect.2 - rect.0).max(rect.3 - rect.1) as f32;
                self.ants[i].records[slot] = Some(TransitRecord::open(
                    *key,
                    self.tick,
                    to,
                    (side / 4.0).max(1.0),
                ));
            }
        }
        false
    }

    /// A replayed transit ends: apply what the outcome carried along
    /// the path it recorded, put the ant at the exit, and enter the
    /// next node.
    fn complete_transit(&mut self, i: usize) {
        let Some(t) = self.ants[i].transit.take() else {
            return;
        };
        let o = t.outcome;
        let from_cell = self.ants[i].cell();
        // An exit beyond a portal's edge comes out on the joined edge,
        // and the body's frame turns with the fold.
        let (exit, turn) = if !self.world.is_passable(t.exit.cell()) {
            match self.world.warp(t.exit) {
                Some(warp) => (warp.point, warp.turn),
                None => (t.exit, 0.0),
            }
        } else {
            (t.exit, 0.0)
        };
        let to_cell = exit.cell();
        // The body moves from the entry cell to the exit cell.
        if let Some(c) = self.world.cell_mut(from_cell) {
            c.occupancy = c.occupancy.saturating_sub(1);
        }
        if let Some(c) = self.world.cell_mut(to_cell) {
            c.occupancy = c.occupancy.saturating_add(1);
        }
        self.world.record_crossing(to_cell);
        let leg = match t.key.leg {
            1 => Leg::Inbound,
            2 => Leg::Searching,
            _ => Leg::Outbound,
        };
        let laden = t.key.laden;
        let leaf = self.ants[i].leaf;
        // What was laid and walked, spread along the path from entry
        // through the waypoints to the exit.
        let mut path: Vec<Point> = Vec::with_capacity(o.via_len as usize + 2);
        path.push(t.entry);
        path.extend(o.waypoints());
        path.push(t.exit);
        let segments: Vec<f64> = path.windows(2).map(|w| w[0].distance(w[1])).collect();
        let total: f64 = segments.iter().sum();
        let samples = (o.length.ceil().max(1.0) as usize).clamp(1, 64);
        let per = 1.0 / samples as f64;
        let mut prev = t.entry;
        for s in 1..=samples {
            let p = point_along(&path, &segments, total * s as f64 / samples as f64);
            let cell = p.cell();
            for (k, &amount) in o.deposits.iter().enumerate() {
                if amount > 0.0 {
                    let kind = Pheromone::ALL[k];
                    self.world.deposit(cell, kind, amount as f64 * per);
                    if let Some(m) = &mut self.memo {
                        m.record_deposit(cell, kind, amount as f64 * per);
                    }
                }
            }
            if let Some(h) = &mut self.history {
                h.record(prev, p, 0.0);
            }
            if let Some(m) = &mut self.memo {
                m.record_move(cell, o.length as f64 * per);
                m.record_replay(
                    cell,
                    o.decisions as f64 * per,
                    o.entropy as f64 * per,
                    o.straight as f64 * per,
                    leg,
                    laden,
                );
            }
            prev = p;
        }
        // Accounting, as if the decisions had been made.
        self.stats.decisions += o.decisions as u64;
        self.stats.decisions_replayed += o.decisions as u64;
        self.stats.entropy_sum += o.entropy as f64;
        self.stats.selected_entropy_sum += o.entropy as f64;
        self.stats.path.moves += o.decisions as u64;
        self.stats.path.length += o.length as f64;
        for &node in &self.leaf_paths[leaf] {
            self.stats.path_by_node[node].moves += o.decisions as u64;
            self.stats.path_by_node[node].length += o.length as f64;
        }
        // Coarser nodes still being recorded take the outcome in.
        let level = t.key.node.level;
        let levels = self.transit_levels();
        for (slot, &l) in levels.iter().enumerate().take(MAX_TRANSIT_LEVELS) {
            if l < level {
                if let Some(r) = &mut self.ants[i].records[slot] {
                    r.absorb(&o, t.exit);
                }
            }
        }
        // The ant itself: where it is, where it faces, what it has walked
        // and integrated.
        let (dx, dy) = t.entry.to(t.exit);
        let species = &self.species;
        let a = &mut self.ants[i];
        a.position = exit;
        a.heading = crate::geometry::wrap_angle(o.heading as f64 + turn);
        a.trip_length += o.length as f64;
        a.integrate(dx, dy, species, &mut self.rng);
        a.rotate_frame(turn);
        a.remember(from_cell);
        // Out of the nodes crossed, into the next.
        self.cross(i, from_cell, to_cell, exit);
    }

    /// The queen's mind, if she has one.
    pub fn queen(&self) -> Option<&Queen> {
        self.mind.as_ref()
    }

    /// The queen's mind, mutably (to fit her readouts or change her
    /// thought).
    pub fn queen_mut(&mut self) -> Option<&mut Queen> {
        self.mind.as_mut()
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
        self.stats.store_mg = self.nest.sugar_mg();
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
        let t0 = std::time::Instant::now();
        if self.dirty {
            self.recompile();
        }
        self.update_environment();
        self.index_inside();
        self.plan_frame();
        for i in 0..self.ants.len() {
            if self.ants[i].alive {
                self.step_ant(i);
            }
        }
        let t1 = std::time::Instant::now();
        self.world.step_pheromones();
        let t2 = std::time::Instant::now();
        self.world
            .step_food(self.species.odour_per_ul_s, self.species.odour_per_mg_s);
        let t3 = std::time::Instant::now();
        self.nest_step();
        let t4 = std::time::Instant::now();
        if let Some(h) = &mut self.history {
            h.step();
        }
        if let Some(m) = &mut self.memo {
            m.step();
        }
        let t5 = std::time::Instant::now();
        self.tick += 1;
        if let (Some(queen), Some(history)) = (&mut self.mind, &self.history) {
            if queen.due(self.tick) {
                if let Some(m) = &mut self.memo {
                    m.compose();
                }
                queen.epoch(
                    self.tick,
                    history,
                    self.memo.as_ref(),
                    &mut self.hierarchy,
                    &mut self.rng,
                );
                self.dirty = true;
            }
        }
        let t6 = std::time::Instant::now();
        self.profile.ticks += 1;
        let decisions = std::mem::take(&mut self.decisions_time);
        self.profile.add(Phase::Decisions, decisions);
        self.profile
            .add(Phase::Ants, (t1 - t0).saturating_sub(decisions));
        self.profile.add(Phase::Pheromones, t2 - t1);
        self.profile.add(Phase::Food, t3 - t2);
        self.profile.add(Phase::Nest, t4 - t3);
        self.profile.add(Phase::History, t5 - t4);
        self.profile.add(Phase::Queen, t6 - t5);
        self.stats.ticks += 1;
        self.stats.time_s += self.tick_s;
        self.stats.alive = self.alive;
        self.stats.outside = self.outside;
        self.stats.store_mg = self.nest.sugar_mg();
        if self.log_every_ticks > 0 && self.tick.is_multiple_of(self.log_every_ticks) {
            self.snapshot();
        }
        self.profile.add(Phase::Other, t6.elapsed());
    }

    /// Wall time per phase of the tick since the start (or the last
    /// reset).
    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    /// Start the profile afresh.
    pub fn reset_profile(&mut self) {
        self.profile = Profile::default();
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
        self.heat_hazard_per_tick = 1.0 - (-self.species.heat_hazard_per_s(t) * self.tick_s).exp();
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
            store_mg: self.nest.sugar_mg(),
            satiation: self.nest.satiation(),
            crop_fill: {
                let (sum, n) = self
                    .living()
                    .filter(|a| a.is_inside())
                    .fold((0.0, 0usize), |(s, n), a| (s + a.crop_fill(), n + 1));
                if n == 0 {
                    0.0
                } else {
                    sum / n as f64
                }
            },
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

    /// Trophallaxis inside the nest: every sharing interval a worker meets
    /// a random nestmate, or the reserve standing in for the nestmates not
    /// simulated, and the fuller crop passes part of the difference in
    /// fill to the emptier one, so crop loads even out through the colony
    /// (Buffin et al. 2009; Greenwald et al. 2015). Then the nest's crop
    /// totals are refreshed.
    fn share_food(&mut self) {
        let fraction = self.species.transfer_fraction;
        let p_share = (self.tick_s / self.species.sharing_interval_s.max(1e-9)).min(1.0);
        let inside: Vec<usize> = (0..self.ants.len())
            .filter(|&j| self.ants[j].alive && self.ants[j].is_inside())
            .collect();
        let virtual_nestmates = self.nest.virtual_nestmates;
        for &i in &inside {
            if self.ants[i].activity == Activity::Unloading || !self.rng.chance(p_share) {
                continue;
            }
            // The reserve is met in proportion to the nestmates it stands
            // for among all those inside; a simulated nestmate has to be
            // within reach.
            let others = (inside.len() - 1) as f64;
            let meet_reserve = virtual_nestmates > 0.0
                && self
                    .rng
                    .chance(virtual_nestmates / (virtual_nestmates + others));
            let (si, ci) = (self.ants[i].sugar_mg, self.ants[i].crop_capacity_mg);
            if meet_reserve {
                // The reserve behaves as a nestmate of the same crop size
                // filled to the reserve's own level.
                let fill_r = if self.nest.capacity_mg > 0.0 {
                    (self.nest.store_mg / self.nest.capacity_mg).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let amount = fraction * (si - fill_r * ci) / 2.0;
                let amount = if amount > 0.0 {
                    amount.min((self.nest.capacity_mg - self.nest.store_mg).max(0.0))
                } else {
                    -((-amount).min(self.nest.store_mg))
                };
                self.ants[i].sugar_mg -= amount;
                self.nest.store_mg += amount;
                self.stats.trophallaxis_mg += amount.abs();
            } else if let Some(j) =
                self.random_neighbour_inside(i, |a| a.activity != Activity::Unloading)
            {
                {
                    // The transfer that would equalise the two fills.
                    let (sj, cj) = (self.ants[j].sugar_mg, self.ants[j].crop_capacity_mg);
                    let equalising = (si * cj - sj * ci) / (ci + cj).max(1e-12);
                    let amount = fraction * equalising;
                    if amount.abs() > 1e-9 {
                        self.ants[i].sugar_mg -= amount;
                        self.ants[j].sugar_mg += amount;
                        self.stats.trophallaxis_mg += amount.abs();
                        self.record_contact(i, j);
                    }
                }
            }
        }
        let (crops, capacity) = inside.iter().fold((0.0, 0.0), |(s, c), &i| {
            (s + self.ants[i].sugar_mg, c + self.ants[i].crop_capacity_mg)
        });
        self.nest.crops_mg = crops;
        self.nest.crops_capacity_mg = capacity;
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
        if activity.is_inside() {
            self.walk_inside(i);
        }
        match activity {
            Activity::Resting => self.rest(i),
            Activity::Nursing => self.nurse(i),
            Activity::Unloading => self.unload(i),
            Activity::Fetching => self.fetch(i),
            Activity::Leaving => self.leave(i),
            Activity::Feeding => self.feed(i),
            Activity::Outbound | Activity::Inbound | Activity::Searching => self.walk(i),
        }
        if self.ants[i].alive {
            self.metabolize(i);
        }
    }

    fn nurses_now(&self) -> usize {
        self.nurses
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
            Activity::Resting | Activity::Leaving | Activity::Fetching => (forget, forget),
            _ => (learn, forget),
        };
        a.traits.foraging_threshold =
            (a.traits.foraging_threshold * f_factor).clamp(f_bounds.0, f_bounds.1);
        a.traits.nursing_threshold =
            (a.traits.nursing_threshold * n_factor).clamp(n_bounds.0, n_bounds.1);
    }

    fn rest(&mut self, i: usize) {
        let colony_hunger = self.nest.hunger();
        let (site_bonus, threshold, nursing_threshold, excitement, own_hunger) = {
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
                1.0 - a.crop_fill(),
            )
        };
        // Hunger is felt in the worker's own crop, amplified by what it
        // reads off its nestmates: a full crop keeps a worker in, so a
        // forager that could not unload stays in, which is how a satiated
        // colony stops foraging (Greenwald et al. 2018: exits fall steeply
        // with the forager's crop load; Mailleux, Detrain & Deneubourg
        // 2006), while a starving individual goes out even when its
        // nestmates are replete (Mailleux et al. 2011).
        let hunger = own_hunger * (0.5 * own_hunger + 0.5 * colony_hunger);
        let protein_demand = self.nest.protein_demand_now;
        let stimulus = self.species.hunger_gain * hunger
            + self.species.protein_demand_gain * protein_demand
            + excitement
            + site_bonus * hunger.max(protein_demand);
        let p_forage =
            self.species.response(stimulus, threshold) * self.decision_prob * self.activity_factor;
        let larvae = self.nest.larvae_now as f64;
        let demand = if larvae <= 0.0 {
            0.0
        } else {
            let nurses = self.nurses_now() as f64;
            (larvae / self.species.brood_per_nurse.max(1e-9)) / (nurses + 1.0)
        };
        let p_nurse = self.species.response(demand, nursing_threshold)
            * self.decision_prob
            * self.brood_proximity(self.ants[i].cell());
        // Undertaking: corpses inside are carried out and dropped away
        // from the nest.
        let p_undertake = if self.nest.corpses > 0 {
            let stimulus = self.species.undertaking_gain * self.nest.corpses as f64;
            self.species.response(stimulus, threshold) * self.decision_prob
        } else {
            0.0
        };
        let u = self.rng.next_f64();
        if u < p_undertake {
            // Undertaking: go to the corpse where it lies.
            let here = self.ants[i].cell();
            if let Some(target) = self.nearest_corpse_cell(here) {
                let a = &mut self.ants[i];
                a.activity = Activity::Fetching;
                a.goal = Some(target);
            }
        } else if u < p_undertake + p_forage {
            self.start_leaving(i);
        } else if u < p_undertake + p_forage + p_nurse {
            let bout = self.seconds_to_ticks(self.species.nursing_bout_s);
            let a = &mut self.ants[i];
            a.activity = Activity::Nursing;
            a.timer = bout;
            self.nurses += 1;
        }
    }

    fn depart(&mut self, i: usize) {
        let here = self.ants[i].cell();
        let exit = if self.world.is_nest(here) {
            here
        } else {
            self.nest_cells[self.rng.below(self.nest_cells.len().max(1))]
        };
        let heading = self.rng.range(-std::f64::consts::PI, std::f64::consts::PI);
        let jitter = (self.rng.range(-0.4, 0.4), self.rng.range(-0.4, 0.4));
        let outbound_laying = self.species.outbound_laying;
        let exploratory_laying = self.species.exploratory_laying;
        // A protein trip or a sugar trip, from the colony's demand.
        let demand = self.nest.protein_demand_now;
        let base = self.species.protein_acceptance_base;
        let accepts_prey = self.rng.chance(base + (1.0 - base) * demand);
        let sight = self.species.sight_cm / self.world.cell_cm().max(1e-9);
        let exit_point = {
            let c = Point::center_of(exit);
            Point::new(c.x + jitter.0, c.y + jitter.1)
        };
        let nest_view = self
            .world
            .nearest_landmark(exit_point, sight)
            .map(|(landmark, l)| View {
                landmark,
                offset: exit_point.to(Point::center_of(l)),
            });
        let a = &mut self.ants[i];
        a.activity = Activity::Outbound;
        a.accepts_prey = accepts_prey;
        a.item_mg = 0.0;
        a.goal = None;
        a.records = [None; MAX_TRANSIT_LEVELS];
        a.transit = None;
        if nest_view.is_some() {
            a.nest_view = nest_view;
        }
        let c = Point::center_of(exit);
        a.position = Point::new(c.x + jitter.0, c.y + jitter.1);
        a.heading = heading;
        // Path integration aims at the nest as a whole: the vector starts
        // as the exit's offset from the centre, so that a returning
        // forager heads for the mound and not for the one cell it left by.
        a.reset_home_vector();
        a.home_vector = a.position.to(Point::center_of(self.world.nest()));
        a.home_vector = (-a.home_vector.0, -a.home_vector.1);
        a.clear_memory();
        a.steps_since_nest = 0;
        a.search_steps = 0;
        a.move_credit = 0.0;
        a.trip_length = 0.0;
        a.laying = if a.corpse {
            None
        } else if (outbound_laying && a.site.is_some()) || exploratory_laying {
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
        // Larvae are fed from the nurse's own crop, by a nurse standing in
        // the brood chamber.
        let in_chamber = self.depth_of(self.ants[i].cell()) <= self.species.brood_depth;
        let available = if in_chamber {
            rate.min(self.ants[i].sugar_mg)
        } else {
            0.0
        };
        let protein_available = if in_chamber {
            rate.min(self.nest.protein_mg)
        } else {
            0.0
        };
        // A nurse with something to give tends the next larva in line:
        // its starvation clock is reset, and it is fed what it still
        // wants of sugar and of protein.
        let tended = if available > 0.0 || protein_available > 0.0 {
            self.nest.next_larva()
        } else {
            None
        };
        if let Some(idx) = tended {
            let need = self.species.larva_protein_mg;
            let larva = &mut self.nest.brood[idx];
            larva.unfed_s = 0.0;
            let sugar = available.min((need_cap - larva.fed_mg).max(0.0));
            larva.fed_mg += sugar;
            let protein = protein_available.min((need - larva.protein_mg).max(0.0));
            larva.protein_mg += protein;
            self.ants[i].sugar_mg -= sugar;
            self.nest.protein_mg -= protein;
        }
        let no_larvae = self.nest.larvae_now == 0;
        let a = &mut self.ants[i];
        a.time_nursing += 1;
        a.timer = a.timer.saturating_sub(1);
        if a.timer == 0 || no_larvae {
            a.activity = Activity::Resting;
            self.nurses = self.nurses.saturating_sub(1);
        }
    }

    /// Unloading by trophallaxis: every contact interval the forager
    /// offers its load to a nestmate inside (or to the reserve, standing
    /// for nestmates not simulated), which takes a share of its own empty
    /// crop space (Greenwald et al. 2018). Each contact also excites the
    /// receiver in proportion to the food's quality. The forager stops
    /// once its crop is down to its residual, or gives up after too many
    /// contacts and keeps what is left.
    fn unload(&mut self, i: usize) {
        {
            let a = &mut self.ants[i];
            a.timer = a.timer.saturating_sub(1);
            if a.timer > 0 {
                return;
            }
        }
        let quality = self.ants[i].load_quality;
        let residual = self.species.unload_residual_fraction * self.ants[i].crop_capacity_mg;
        let offered = self.ants[i].sugar_mg;
        let mut done = offered <= residual + 1e-12;
        if !done {
            // Choose a receiver: the reserve (nestmates not simulated,
            // met in proportion to their number), or a nestmate within
            // reach; with nobody within reach the contact passes nothing.
            let inside_total = self.inside_count().saturating_sub(1) as f64;
            let virtual_share = self.nest.virtual_nestmates
                / (self.nest.virtual_nestmates + inside_total).max(1e-9);
            let fraction = self.species.transfer_fraction;
            let receiver = if self.rng.chance(virtual_share) {
                None
            } else {
                self.random_neighbour_inside(i, |a| {
                    matches!(a.activity, Activity::Resting | Activity::Nursing)
                })
            };
            let taken = if receiver.is_none() && self.rng.chance(virtual_share) {
                let deficit = self.nest.reserve_deficit_per_nestmate();
                let amount = (fraction * deficit).min(offered - residual).max(0.0);
                self.nest.store_mg += amount;
                amount
            } else if receiver.is_none() {
                0.0
            } else {
                let j = receiver.expect("a receiver");
                let amount = (fraction * self.ants[j].crop_deficit_mg())
                    .min(offered - residual)
                    .max(0.0);
                self.ants[j].sugar_mg += amount;
                // The food carries the message: a contact that passes
                // little excites little.
                let share =
                    (amount / (fraction * self.ants[j].crop_capacity_mg).max(1e-12)).min(1.0);
                self.ants[j].excitement += self.species.excitation_per_contact * quality * share;
                self.record_contact(i, j);
                amount
            };
            self.ants[i].sugar_mg -= taken;
            self.ants[i].contacts += 1;
            self.stats.unloading_contacts += 1;
            self.stats.trophallaxis_mg += taken;
            let contacts = self.ants[i].contacts;
            if self.ants[i].sugar_mg <= residual + 1e-12 {
                done = true;
            } else if contacts >= self.species.max_unloading_contacts {
                done = true;
                self.stats.failed_unloads += 1;
            }
        }
        if !done {
            self.ants[i].timer = self.seconds_to_ticks(self.species.contact_interval_s);
            return;
        }
        let fill = self.ants[i].load_fill;
        {
            let a = &mut self.ants[i];
            a.crop_ul = 0.0;
            a.activity = Activity::Resting;
        }
        // Poor sources are abandoned: the memory survives with a
        // probability that depends on the quality of the food and on how
        // much of it there was to drink.
        let keep = self.species.site_fidelity(quality * fill);
        if !self.rng.chance(keep) {
            self.ants[i].site = None;
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
            let desired = self.species.crop_capacity_ul
                * a.traits.size
                * self.species.load_fraction(a.load_quality);
            // What the crop can still take, given what it already holds.
            let per_ul = self.species.sugar_mg(1.0, a.load_molarity);
            let space_ul = if per_ul > 0.0 {
                a.crop_deficit_mg() / per_ul + a.crop_ul
            } else {
                desired
            };
            (
                a.load_molarity,
                a.load_quality,
                a.crop_ul,
                desired.min(space_ul),
            )
        };
        let rate = self.species.intake_rate(molarity) * self.tick_s;
        let want = rate.min((want_total - crop).max(0.0));
        let (taken, _) = if want > 0.0 {
            self.world.take_food(cell, want)
        } else {
            (0.0, molarity)
        };
        self.ants[i].crop_ul += taken;
        self.ants[i].sugar_mg += self.species.sugar_mg(taken, molarity);
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

    /// Necrophoresis on entering a cell: a carrier drops its corpse where
    /// corpses already lie (and, past the refuse distance, now and then
    /// anywhere); an unladen explorer picks one up, the less likely the
    /// bigger the pile it lies in (Deneubourg et al. 1991).
    fn handle_corpses(&mut self, i: usize, cell: Position) {
        if self.world.is_nest(cell) {
            // Corpses on the nest's cells are the undertakers' business.
            return;
        }
        let species = &self.species;
        let carrying_corpse = self.ants[i].corpse;
        if carrying_corpse {
            let near = self.world.corpses_near(cell) as f64;
            let far_enough = {
                let d = Point::center_of(self.world.nest()).distance(self.ants[i].position);
                d * self.world.cell_cm() >= species.refuse_distance_cm
            };
            let p_pile = (near / (species.corpse_drop_k + near)).powi(2);
            let p = if far_enough {
                p_pile.max(species.corpse_base_drop)
            } else {
                p_pile
            };
            if self.rng.chance(p) {
                self.world.add_corpse(cell);
                self.ants[i].corpse = false;
            }
            return;
        }
        let here = self.world.cell(cell).map(|c| c.corpses).unwrap_or(0);
        if here == 0 {
            return;
        }
        let a = &self.ants[i];
        let explorer = matches!(a.activity, Activity::Outbound | Activity::Searching)
            && a.site.is_none()
            && !a.carrying();
        if !explorer {
            return;
        }
        let near = self.world.corpses_near(cell) as f64;
        let p = (species.corpse_pickup_k / (species.corpse_pickup_k + near)).powi(2);
        if self.rng.chance(p) && self.world.take_corpse(cell) {
            self.ants[i].corpse = true;
            self.stats.corpses_moved += 1;
        }
    }

    /// Put a carried corpse down here (an ant does not bring one home).
    fn drop_corpse_here(&mut self, i: usize) {
        if self.ants[i].corpse {
            let cell = self.ants[i].cell();
            self.world.add_corpse(cell);
            self.ants[i].corpse = false;
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
        let want = self.species.prey_load_mg * self.ants[i].traits.size;
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
        let sight = self.species.sight_cm / self.world.cell_cm().max(1e-9);
        let here = self.ants[i].position;
        let site_view = self
            .world
            .nearest_landmark(here, sight)
            .map(|(landmark, l)| View {
                landmark,
                offset: here.to(Point::center_of(l)),
            });
        let a = &mut self.ants[i];
        a.steps_since_food = 0;
        a.trip_length = 0.0;
        a.site = Some(Site {
            vector: a.home_vector,
            quality,
            nutrient: a.load_kind,
        });
        a.site_view = site_view;
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
        self.drop_corpse_here(i);
        self.note_stop(i, Stop::GaveUp);
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
        // A worker that comes home still carrying a corpse turns round at
        // the entrance and carries it out again.
        if self.ants[i].corpse {
            let a = &mut self.ants[i];
            a.laying = None;
            a.search_steps = 0;
            a.steps_since_nest = 0;
            a.reset_home_vector();
            a.home_vector = a.position.to(Point::center_of(self.world.nest()));
            a.home_vector = (-a.home_vector.0, -a.home_vector.1);
            a.activity = Activity::Outbound;
            a.heading = self.rng.range(-std::f64::consts::PI, std::f64::consts::PI);
            return;
        }
        let contact = self.seconds_to_ticks(self.species.contact_interval_s);
        let nest = self.world.nest();
        let cell = self.ants[i].cell();
        self.note_stop(i, Stop::Nest);
        if let Some(c) = self.world.cell_mut(cell) {
            c.occupancy = c.occupancy.saturating_sub(1);
        }
        self.outside = self.outside.saturating_sub(1);
        let leaf = self.ants[i].leaf;
        let nest_radius = self.world.nest_radius().max(0) as f64;
        let trip = {
            let a = &mut self.ants[i];
            a.position = Point::center_of(cell);
            a.goal = None;
            a.reset_home_vector();
            a.steps_since_nest = 0;
            a.laying = None;
            a.search_steps = 0;
            if a.carrying() {
                a.activity = Activity::Unloading;
                a.timer = contact;
                a.contacts = 0;
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
            // The load is in the nest: prey goes to the protein store at
            // once, the sugar stays in the forager's crop until nestmates
            // take it.
            let (load_ul, molarity, item) = {
                let a = &mut self.ants[i];
                let out = (a.crop_ul, a.load_molarity, a.item_mg);
                a.item_mg = 0.0;
                out
            };
            let sugar = self.species.sugar_mg(load_ul, molarity);
            self.stats.sugar_delivered_mg += sugar;
            self.nest.protein_mg += item;
            self.stats.protein_delivered_mg += item;
            let reward = self.config.reward.sugar_mg * sugar + self.config.reward.protein_mg * item;
            self.credit(leaf, reward, false);
        }
    }

    fn walk(&mut self, i: usize) {
        // A replayed transit: nothing to decide until it ends.
        if let Some(until) = self.ants[i].transit.as_ref().map(|t| t.until) {
            if self.tick < until {
                self.tick_timers(i);
                return;
            }
            self.complete_transit(i);
            if !self.ants[i].alive || self.ants[i].transit.is_some() {
                self.tick_timers(i);
                return;
            }
            if self.check_transitions(i) {
                return;
            }
        }
        // A change of leg inside a node ends the transits being recorded
        // as ones that stopped inside.
        let leg = self.leg_code(i);
        if self.ants[i]
            .records
            .iter()
            .flatten()
            .any(|r| r.key.leg != leg)
        {
            self.close_records(i, INSIDE, 0.0);
        }
        // Stunned by a fall, or losing grip on a slippery slope and
        // falling to the surface below.
        if self.frame.is_some() {
            if self.ants[i].stun > 0 {
                self.ants[i].stun -= 1;
                self.tick_timers(i);
                return;
            }
            let here = self.ants[i].cell();
            let slip = self.world.slope_at(here).map(|s| s.slip).unwrap_or(0.0);
            if slip > 0.0 {
                let laden = self.ants[i].load_fill.clamp(0.0, 1.0);
                if self.rng.chance((slip * (1.0 + 2.0 * laden)).min(1.0)) {
                    self.fall(i);
                    self.tick_timers(i);
                    return;
                }
            }
        }
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
            if a.carrying() || a.corpse {
                speed *= self.species.loaded_speed_factor;
            }
            if trail > k {
                speed *= self.species.trail_speed_factor;
            }
            speed *= (1.0 - self.species.crowding_slowdown * crowding).max(0.2);
            speed *= self.world.climb_factor(here, a.heading);
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
                let decided = std::time::Instant::now();
                // The pipeline: hold the heading under a horizon, wait
                // for the budget within the slack, or decide.
                let ring = match self.pipeline_gate(i, step) {
                    Gate::Hold => {
                        self.stats.frames.held += 1;
                        Some(0)
                    }
                    Gate::Defer => {
                        self.stats.frames.deferred += 1;
                        Some(0)
                    }
                    Gate::Decide => {
                        let ring = self.decide(i, mode, step);
                        if ring.is_some() {
                            self.set_horizon(i);
                        }
                        ring
                    }
                };
                self.decisions_time += decided.elapsed();
                if let Some(ring) = ring {
                    if self.move_ant(i, ring, step) {
                        // A replayed transit began: the tick's walking is
                        // done.
                        return;
                    }
                }
                if self.check_transitions(i) {
                    return;
                }
            }
        }
        self.tick_timers(i);
    }

    fn move_ant(&mut self, i: usize, ring: usize, step: f64) -> bool {
        let leaf = self.ants[i].leaf;
        let (from, heading) = {
            let a = &self.ants[i];
            (a.position, ring_heading(ring, a.heading))
        };
        let to_raw = from.advanced(heading, step);
        // Through a portal the step comes out on the joined edge, and the
        // body's frame turns with the fold.
        let (to, turn) = if !self.world.is_passable(to_raw.cell()) {
            match self.world.warp(to_raw) {
                Some(warp) => (warp.point, warp.turn),
                None => (to_raw, 0.0),
            }
        } else {
            (to_raw, 0.0)
        };
        let from_cell = from.cell();
        let to_cell = to.cell();
        if let Some(h) = &mut self.history {
            h.record(from, to_raw, turn_magnitude(ring) as f64 * RING_STEP);
        }
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
            let (dx, dy) = from.to(to_raw);
            a.integrate(dx, dy, species, &mut self.rng);
            if turn != 0.0 {
                a.heading = crate::geometry::wrap_angle(a.heading + turn);
                a.rotate_frame(turn);
            }
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
        if let Some(m) = &mut self.memo {
            m.record_move(to_cell, step);
        }
        for r in self.ants[i].records.iter_mut().flatten() {
            r.walked(step, to);
        }
        match laying {
            Some(Pheromone::Trail) => Self::lay(
                &mut self.world,
                self.memo.as_mut(),
                &mut self.ants[i].records,
                to_cell,
                Pheromone::Trail,
                species.trail_deposit * strength * step * crowd_factor,
            ),
            Some(Pheromone::NoEntry) => Self::lay(
                &mut self.world,
                self.memo.as_mut(),
                &mut self.ants[i].records,
                to_cell,
                Pheromone::NoEntry,
                species.no_entry_deposit * strength * step,
            ),
            Some(kind) => Self::lay(
                &mut self.world,
                self.memo.as_mut(),
                &mut self.ants[i].records,
                to_cell,
                kind,
                strength * step,
            ),
            None => {}
        }
        if home {
            Self::lay(
                &mut self.world,
                self.memo.as_mut(),
                &mut self.ants[i].records,
                to_cell,
                Pheromone::Home,
                species.trail_deposit * step,
            );
        }
        if species.territory_deposit > 0.0 {
            Self::lay(
                &mut self.world,
                self.memo.as_mut(),
                &mut self.ants[i].records,
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
            self.fix_position_by_landmarks(i);
            self.handle_corpses(i, to_cell);
        }
        // Crossing into other memo nodes closes the transits being
        // recorded and opens the next, or replays one.
        let mut in_transit = false;
        if let Some(level) = self.transit_level() {
            let node_from = self.memo.as_ref().and_then(|m| m.key_of(from_cell, level));
            let node_to = self.memo.as_ref().and_then(|m| m.key_of(to_cell, level));
            if node_from != node_to {
                in_transit = self.cross(i, from_cell, to_cell, to);
            }
        }
        in_transit
    }

    /// View-based position fixing: when the landmark of a stored view is
    /// in sight, the ant knows where it stands relative to the place the
    /// view was taken at, and pulls its path-integration estimate towards
    /// that (Wehner & Räber 1979; Collett 1992).
    fn fix_position_by_landmarks(&mut self, i: usize) {
        if self.world.landmarks().is_empty() {
            return;
        }
        let sight = self.species.sight_cm / self.world.cell_cm().max(1e-9);
        let correction = self.species.landmark_correction.clamp(0.0, 1.0);
        let (position, activity, target, site, nest_view, site_view) = {
            let a = &self.ants[i];
            (
                a.position,
                a.activity,
                a.search_target,
                a.site,
                a.nest_view,
                a.site_view,
            )
        };
        let homeward = matches!(
            (activity, target),
            (Activity::Inbound, _) | (Activity::Searching, SearchTarget::Nest)
        );
        // Prefer the view that belongs to the current leg.
        let nest_candidate = nest_view.map(|v| (v, (0.0, 0.0)));
        let site_candidate = site.zip(site_view).map(|(s, v)| (v, s.vector));
        let candidates = if homeward {
            [nest_candidate, site_candidate]
        } else {
            [site_candidate, nest_candidate]
        };
        for (view, place_vector) in candidates.into_iter().flatten() {
            let Some(&landmark) = self.world.landmarks().get(view.landmark) else {
                continue;
            };
            let l = Point::center_of(landmark);
            if position.distance(l) > sight {
                continue;
            }
            // Where the ant stands relative to the place: the landmark's
            // offset from the place minus its offset from the ant.
            let (rx, ry) = position.to(l);
            let relative = (view.offset.0 - rx, view.offset.1 - ry);
            let fixed = (place_vector.0 + relative.0, place_vector.1 + relative.1);
            let a = &mut self.ants[i];
            a.home_vector.0 += correction * (fixed.0 - a.home_vector.0);
            a.home_vector.1 += correction * (fixed.1 - a.home_vector.1);
            a.lost = false;
            self.stats.landmark_fixes += 1;
            break;
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
                    self.drop_corpse_here(i);
                    self.start_feeding(i, molarity);
                    return true;
                }
                if prey_here && !self.ants[i].carrying() {
                    self.drop_corpse_here(i);
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
        let outbound = (a.trip_length, a.position);
        a.trip_length = 0.0;
        self.record_outbound(i, outbound);
        self.note_stop(i, Stop::Food);
        self.stats.food_picked += 1;
        let reward = self.config.reward.food_picked;
        self.credit(leaf, reward, false);
    }

    /// Record the outbound leg that just found food: its path length and
    /// the direct distance from the nest.
    fn record_outbound(&mut self, i: usize, (length, pickup): (f64, Point)) {
        let leaf = self.ants[i].leaf;
        let direct = self.direct_distance(pickup);
        self.stats.path.record_outbound(length, direct);
        for &node in &self.leaf_paths[leaf] {
            self.stats.path_by_node[node].record_outbound(length, direct);
        }
    }

    /// The direct distance from the nest's edge to a point: the geodesic
    /// round the walls (the lens's route) where there are walls, the
    /// straight line in the open; remembered per cell.
    fn direct_distance(&mut self, p: Point) -> f64 {
        let nest = Point::center_of(self.world.nest());
        let radius = self.world.nest_radius().max(0) as f64;
        if !self.world.has_walls() {
            return (p.distance(nest) - radius).max(0.0);
        }
        let cell = p.cell();
        if let Some(&d) = self.geodesics.get(&cell) {
            return d;
        }
        let along = crate::lens::geodesic(&self.world, nest, Point::center_of(cell), 3.0)
            .map(|g| g.length())
            .unwrap_or_else(|| p.distance(nest));
        let d = (along - radius).max(0.0);
        self.geodesics.insert(cell, d);
        d
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
        let outbound = (a.trip_length, a.position);
        a.trip_length = 0.0;
        self.record_outbound(i, outbound);
        self.note_stop(i, Stop::Food);
        self.stats.food_picked += 1;
        self.stats.prey_picked += 1;
        let reward = self.config.reward.food_picked;
        self.credit(leaf, reward, false);
    }

    /// Every worker lives off its own crop: metabolism (scaled by body
    /// mass to the three quarters and by temperature) drains it, and the
    /// starvation clock runs only while it is empty.
    fn metabolize(&mut self, i: usize) {
        let mortality = self.config.nest.mortality;
        let starvation = self.species.starvation_s;
        let tick_s = self.tick_s;
        let hazard = self.hazard_per_tick;
        let inside = self.ants[i].is_inside();
        let burn = self.species.consumption_mg_per_ant_per_s
            * self.metabolism_factor
            * tick_s
            * self.ants[i].traits.size.powf(0.75);
        {
            let a = &mut self.ants[i];
            if a.sugar_mg > 0.0 {
                a.sugar_mg = (a.sugar_mg - burn).max(0.0);
                a.energy = starvation;
            } else {
                a.energy -= tick_s;
            }
        }
        if !inside {
            self.ants[i].time_foraging += 1;
            if self.ants[i].activity == Activity::Feeding {
                self.ants[i].energy = starvation;
            }
            if mortality && self.rng.chance(hazard) {
                self.die(i, Cause::Predation);
                return;
            }
            // Heat: the risk of foraging near the thermal limit.
            if mortality && self.rng.chance(self.heat_hazard_per_tick) {
                self.die(i, Cause::Heat);
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
        if outside {
            self.note_stop(i, Stop::Death);
        }
        if self.ants[i].activity == Activity::Nursing {
            self.nurses = self.nurses.saturating_sub(1);
        }
        self.ants[i].alive = false;
        if self.ants[i].corpse {
            // A carried corpse is dropped where the carrier dies.
            self.ants[i].corpse = false;
            self.world.add_corpse(cell);
        }
        if outside {
            if let Some(c) = self.world.cell_mut(cell) {
                c.occupancy = c.occupancy.saturating_sub(1);
            }
            self.outside = self.outside.saturating_sub(1);
            if cause == Cause::Predation {
                self.world
                    .deposit(cell, Pheromone::Alarm, self.species.alarm_release);
            } else {
                self.world.add_corpse(cell);
            }
        } else {
            // The body lies where the worker stood inside.
            self.world.add_corpse(cell);
            self.nest.corpses += 1;
        }
        self.alive = self.alive.saturating_sub(1);
        self.stats.deaths += 1;
        match cause {
            Cause::Predation => self.stats.deaths_predation += 1,
            Cause::Starvation => self.stats.deaths_starvation += 1,
            Cause::Heat => self.stats.deaths_heat += 1,
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
        self.nest.corpses = self.nest_corpses();
        if self.tick.is_multiple_of(60) {
            self.nest.sort_by_hunger();
        }
        self.share_food();

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
        self.nest.refresh(self.species.larva_protein_mg);
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
        // The budget is only needed to size a deformation.
        let deforms =
            policy.deformation.smooth_share() > 0.0 || policy.deformation.rough_share() > 0.0;
        let h = match tempering {
            Tempering::Entropy(f) => f,
            Tempering::Temperature(t) if deforms => base.entropy_fraction_at(t),
            Tempering::Temperature(_) => 0.0,
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

        // Accounting: the probability of holding the heading, for the
        // pipeline.
        self.straight_prob = {
            let cone = self
                .pipeline
                .as_ref()
                .map(|p| p.cone)
                .unwrap_or(0)
                .min(RING / 4);
            let mut p = selected[0];
            for r in 1..=cone {
                p += selected[r] + selected[RING - r];
            }
            p
        };
        self.stats.decisions += 1;
        self.stats.entropy_sum += tempered.entropy;
        if let Some(m) = &mut self.memo {
            let leg = if searching {
                Leg::Searching
            } else if matches!(mode, Mode::Inbound) {
                Leg::Inbound
            } else {
                Leg::Outbound
            };
            m.record_decision(position.cell(), chosen, tempered.entropy, leg, carrying);
        }
        for r in self.ants[i].records.iter_mut().flatten() {
            r.decisions = r.decisions.saturating_add(1);
            r.entropy += tempered.entropy as f32;
            r.straight += (turn_magnitude(chosen) as f64 * RING_STEP).cos() as f32;
        }
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

/// A node entered, as a transit key and the node's cells.
type Entry = (TransitKey, (usize, usize, usize, usize));

/// What the pipeline says about an ant's next step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gate {
    /// Keep the heading: the hold has not ended.
    Hold,
    /// Keep the heading a little longer: the budget is spent.
    Defer,
    /// Decide.
    Decide,
}

/// The point at a distance along a polyline (its last point beyond the
/// end).
fn point_along(path: &[Point], segments: &[f64], distance: f64) -> Point {
    let mut acc = 0.0;
    for (i, &len) in segments.iter().enumerate() {
        if acc + len >= distance || i + 1 == segments.len() {
            let f = if len > 1e-12 {
                ((distance - acc) / len).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let (p, q) = (path[i], path[i + 1]);
            return Point::new(p.x + (q.x - p.x) * f, p.y + (q.y - p.y) * f);
        }
        acc += len;
    }
    *path.last().unwrap_or(&Point::new(0.0, 0.0))
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
        // No brood, so that sugar hunger is the only foraging stimulus
        // (larvae would send foragers out for protein regardless).
        let mut hungry_cfg = hungry_fast();
        hungry_cfg.nest.initial_brood_per_ant = 0.0;
        let mut hungry = Simulation::new(hungry_cfg.clone(), 3);
        hungry.run(600);
        let mut replete_cfg = hungry_cfg;
        replete_cfg.nest.initial_satiation = 1.0;
        let mut replete = Simulation::new(replete_cfg, 3);
        replete.run(600);
        let out_h = hungry.stats().foraging_fraction();
        let out_r = replete.stats().foraging_fraction();
        assert!(out_h > out_r + 0.1, "hungry {out_h} vs replete {out_r}");
    }

    #[test]
    fn entropy_dial_still_controls_decisions() {
        // Searching ants deliberately run hotter than the dial says; hold
        // that at one so the dial alone sets the decision entropy.
        let mut cfg = hungry_fast();
        cfg.species.search_temperature_factor = 1.0;
        let mut ordered = Simulation::new(cfg.clone(), 3);
        ordered.hierarchy_mut().node_mut(0).surface.entropy = EntropyControl::absolute(0.05);
        ordered.run(400);
        let mut chaotic = Simulation::new(cfg, 3);
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
        assert!(s.deaths_predation + s.deaths_starvation + s.deaths_heat == s.deaths);
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
    fn undertakers_carry_the_dead_out_of_the_nest() {
        let mut cfg = hungry_fast();
        cfg.nest.initial_brood_per_ant = 0.0;
        cfg.nest.mortality = false;
        cfg.world.random_food = None;
        let mut sim = Simulation::new(cfg, 5);
        let nest = sim.world().nest();
        for _ in 0..5 {
            sim.world_mut().add_corpse(nest);
        }
        sim.run(900);
        assert!(sim.stats().corpses_moved >= 5, "{:?}", sim.stats());
        assert_eq!(sim.nest().corpses, 0, "every corpse was taken out");
        let carried = sim.living().filter(|a| a.corpse).count() as u32;
        assert_eq!(sim.world().total_corpses() + carried, 5);
        // Refuse lies away from the nest.
        let nest = Point::center_of(sim.world().nest());
        for (i, c) in sim.world().cells().iter().enumerate() {
            if c.corpses > 0 {
                let w = sim.world().width() as i32;
                let p = Position::new(i as i32 % w, i as i32 / w);
                let d = nest.distance(Point::center_of(p)) * sim.world().cell_cm();
                assert!(
                    d >= sim.species().refuse_distance_cm - 2.0,
                    "corpse at {p:?}, {d} cm"
                );
            }
        }
    }

    #[test]
    fn unloading_slows_and_foraging_stops_as_the_colony_fills() {
        // Greenwald, Baltiansky & Feinerman 2018: receivers take a share of
        // their empty crop space, so as crops fill a forager needs more
        // contacts to unload, and eventually keeps its load and stays in.
        let mut cfg = hungry_fast();
        cfg.nest.store_capacity_mg_per_ant = 0.0;
        cfg.nest.initial_brood_per_ant = 0.0;
        cfg.nest.mortality = false;
        cfg.world.random_food = None;
        cfg.world.food_sources = vec![FoodSource::pool(Position::new(26, 15), 1, 1.0e6, 1.0)];
        let mut sim = Simulation::new(cfg, 9);
        sim.run(600);
        let early = sim.stats().clone();
        assert!(early.food_delivered > 5, "{early:?}");
        let contacts_early = early.unloading_contacts as f64 / early.food_delivered as f64;
        sim.reset_stats();
        sim.run(3000);
        let late = sim.stats().clone();
        let contacts_late = late.unloading_contacts as f64 / late.food_delivered.max(1) as f64;
        assert!(
            contacts_late > contacts_early,
            "unloading should take more contacts once crops fill: {contacts_early} then {contacts_late}"
        );
        assert!(sim.nest().satiation() > 0.85, "{:?}", sim.nest());
        assert!(late.failed_unloads > 0, "{late:?}");
        assert!(
            late.foraging_fraction() < 0.5 * early.foraging_fraction(),
            "foraging should wind down: {} then {}",
            early.foraging_fraction(),
            late.foraging_fraction()
        );
        // Everyone inside has been fed by trophallaxis.
        let hungry = sim
            .living()
            .filter(|a| a.is_inside() && a.crop_fill() < 0.3)
            .count();
        assert_eq!(hungry, 0);
    }

    #[test]
    fn crop_loads_converge_by_sharing() {
        // Food from a small pool reaches a few foragers first and is then
        // spread through the colony (Buffin et al. 2009; Greenwald et al.
        // 2015): the crop fills of the workers inside even out.
        let mut cfg = hungry_fast();
        cfg.nest.store_capacity_mg_per_ant = 0.0;
        cfg.nest.initial_satiation = 0.0;
        cfg.nest.initial_brood_per_ant = 0.0;
        cfg.nest.mortality = false;
        cfg.world.random_food = None;
        cfg.world.food_sources = vec![FoodSource::pool(Position::new(26, 15), 0, 3.0, 1.0)];
        let mut sim = Simulation::new(cfg, 11);
        let spread = |sim: &Simulation| {
            let fills: Vec<f64> = sim
                .living()
                .filter(|a| a.is_inside())
                .map(|a| a.crop_fill())
                .collect();
            let n = fills.len().max(1) as f64;
            let mean = fills.iter().sum::<f64>() / n;
            let var = fills.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / n;
            (mean, var.sqrt())
        };
        // The spread peaks as the first loads come in, then sharing
        // evens the crops out.
        let mut peak = 0.0f64;
        for _ in 0..40 {
            sim.run(30);
            peak = peak.max(spread(&sim).1);
        }
        assert!(sim.stats().food_delivered > 0);
        assert!(peak > 0.02, "loads should first make crops uneven: {peak}");
        sim.run(2400);
        assert!(
            !sim.world().cell(Position::new(26, 15)).unwrap().has_food(),
            "pool drunk up"
        );
        let (mean_now, sd_now) = spread(&sim);
        assert!(mean_now > 0.0);
        assert!(
            sd_now < 0.5 * peak,
            "crop fills should even out: peak sd {peak:.3} → {sd_now:.3}"
        );
        assert!(sim.stats().trophallaxis_mg > 0.0);
    }

    #[test]
    fn heat_kills_foragers_near_the_thermal_limit() {
        // Cerdá, Retana & Cros 1998: mortality rises steeply towards the
        // critical thermal maximum; well below it the heat takes nobody.
        let run = |temperature: f64| {
            let mut cfg = SimConfig::for_species(Species::cataglyphis());
            cfg.ants = 60;
            cfg.world = WorldConfig {
                seed: Some(3),
                ..WorldConfig::default()
            };
            cfg.nest.initial_satiation = 0.05;
            cfg.nest.initial_brood_per_ant = 0.0;
            cfg.environment.temperature_c = temperature;
            let mut sim = Simulation::new(cfg, 3);
            sim.run(1800);
            sim.stats().clone()
        };
        let cool = run(38.0);
        let hot = run(54.5);
        assert_eq!(cool.deaths_heat, 0, "{cool:?}");
        assert!(hot.deaths_heat > 3, "{hot:?}");
        assert!(hot.food_delivered > 0, "the colony still forages: {hot:?}");
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
