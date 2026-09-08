//! The colony simulation: world, ants, and the hierarchy that governs them.
//!
//! Each tick every living ant senses its surroundings, scores the eight
//! candidate directions through the effective policy of its category, lays
//! that ring out as a landscape in front of itself, deforms it with the
//! entropy budget (smoothing, roughening, tempering), selects a direction
//! (a global draw or a crawling sucker), moves, lays pheromone, and interacts
//! with food or the nest. Then pheromones evaporate and diffuse and the
//! colony spends stored food on new ants.

use crate::ant::{observe, Ant, AntId, FEATURES};
use crate::entropy::entropy;
use crate::geometry::{Direction, Position};
use crate::hierarchy::{EffectivePolicy, Hierarchy, HierarchySpec, NodeId};
use crate::landscape::{ring_index, world_direction, EntropyLedger, Landscape, Sucker, RING};
use crate::rng::Rng;
use crate::surface::{BehavioralSurface, SurfaceError, PARAM_LEN};
use crate::world::{Terrain, World, WorldConfig};

/// Energy, starvation, and reproduction parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct EnergyConfig {
    /// Energy of a fresh or refuelled ant.
    pub initial: f64,
    /// Energy burnt per tick.
    pub per_step: f64,
    /// Whether an ant dies when its energy reaches zero.
    pub starvation: bool,
    /// Food units the colony spends to spawn one ant (`0` disables spawning).
    pub spawn_cost: u64,
    /// Population cap for spawning.
    pub max_ants: usize,
}

impl Default for EnergyConfig {
    fn default() -> Self {
        EnergyConfig {
            initial: 400.0,
            per_step: 1.0,
            starvation: true,
            spawn_cost: 10,
            max_ants: 200,
        }
    }
}

/// Pheromone deposition parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct PheromoneConfig {
    /// Home pheromone laid per tick by an ant that just left the nest.
    pub home_deposit: f64,
    /// Food pheromone laid per tick by an ant that just picked up food.
    pub food_deposit: f64,
    /// Deposits shrink as `1 / (1 + decay × steps since the source)`.
    pub deposit_decay: f64,
}

impl Default for PheromoneConfig {
    fn default() -> Self {
        PheromoneConfig {
            home_deposit: 1.0,
            food_deposit: 1.0,
            deposit_decay: 0.03,
        }
    }
}

/// How events translate into the scalar reward learners optimise.
#[derive(Clone, Debug, PartialEq)]
pub struct RewardSpec {
    /// Reward per unit of food delivered to the nest.
    pub food_delivered: f64,
    /// Reward per unit of food picked up.
    pub food_picked: f64,
    /// Reward (normally negative) per starved ant.
    pub death: f64,
    /// Reward per spawned ant.
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

/// Complete simulation configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct SimConfig {
    /// World layout.
    pub world: WorldConfig,
    /// Shape of the control hierarchy.
    pub hierarchy: HierarchySpec,
    /// Surface installed at the root.
    pub instinct: BehavioralSurface,
    /// Initial number of ants.
    pub ants: usize,
    /// Energy parameters.
    pub energy: EnergyConfig,
    /// Pheromone parameters.
    pub pheromone: PheromoneConfig,
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
        SimConfig {
            world: WorldConfig::default(),
            hierarchy: HierarchySpec::default(),
            instinct: BehavioralSurface::instinct(),
            ants: 60,
            energy: EnergyConfig::default(),
            pheromone: PheromoneConfig::default(),
            reward: RewardSpec::default(),
            trace: false,
            selection: Selection::Softmax,
            geometry: GeometryConfig::default(),
            record_surface: None,
            surface_rows: 256,
        }
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

/// Running counters of a simulation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stats {
    /// Ticks simulated since the last reset.
    pub ticks: u64,
    /// Food units delivered to the nest.
    pub food_delivered: u64,
    /// Food units picked up.
    pub food_picked: u64,
    /// Ants that starved.
    pub deaths: u64,
    /// Ants spawned.
    pub births: u64,
    /// Total reward.
    pub reward: f64,
    /// Number of movement decisions made.
    pub decisions: u64,
    /// Sum of the entropy targets met by tempering (nats).
    pub entropy_sum: f64,
    /// Sum of entropies of the distributions directions were actually drawn
    /// from (equal to `entropy_sum` under [`Selection::Softmax`]).
    pub selected_entropy_sum: f64,
    /// Reward attributed to each node (every event credits the whole path).
    pub reward_by_node: Vec<f64>,
    /// Deliveries attributed to each node.
    pub delivered_by_node: Vec<u64>,
    /// Path geometry and entropy ledger for the whole colony.
    pub path: PathStats,
    /// Path geometry and entropy ledger per node.
    pub path_by_node: Vec<PathStats>,
    /// Living ants at the end of the last tick.
    pub alive: usize,
    /// Food currently stored in the nest.
    pub food_store: u64,
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

    /// Mean entropy target per decision (what the dial asked for).
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

/// A running colony.
#[derive(Clone, Debug)]
pub struct Simulation {
    config: SimConfig,
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
    food_store: u64,
    alive: usize,
    next_leaf: usize,
    tick: u64,
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
        let world = World::new(config.world.clone(), &mut rng);
        let nodes = hierarchy.len();
        let leaf_paths = hierarchy
            .leaves()
            .iter()
            .map(|&l| hierarchy.path(l).to_vec())
            .collect();
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
            food_store: 0,
            alive: 0,
            next_leaf: 0,
            tick: 0,
            config,
        };
        for _ in 0..sim.config.ants {
            sim.spawn_ant();
        }
        sim.stats.alive = sim.alive;
        sim
    }

    fn spawn_ant(&mut self) -> usize {
        let id = self.ants.len();
        let heading = Direction::from_index(self.rng.below(Direction::COUNT));
        let leaf = self.next_leaf;
        self.next_leaf = (self.next_leaf + 1) % self.leaf_paths.len();
        let nest = self.world.nest();
        let ant = Ant::new(id, nest, heading, leaf, self.config.energy.initial);
        if let Some(c) = self.world.cell_mut(nest) {
            c.occupancy = c.occupancy.saturating_add(1);
        }
        self.ants.push(ant);
        self.alive += 1;
        id
    }

    /// Configuration.
    pub fn config(&self) -> &SimConfig {
        &self.config
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
        self.stats.food_store = self.food_store;
        self.trace = Trace::new(nodes);
    }

    /// Ticks simulated since construction.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Food stored in the nest.
    pub fn food_store(&self) -> u64 {
        self.food_store
    }

    /// Number of living ants.
    pub fn alive(&self) -> usize {
        self.alive
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
        self.reproduce();
        self.tick += 1;
        self.stats.ticks += 1;
        self.stats.alive = self.alive;
        self.stats.food_store = self.food_store;
    }

    /// Advance by `steps` ticks and return the counters.
    pub fn run(&mut self, steps: usize) -> &Stats {
        for _ in 0..steps {
            self.step();
        }
        &self.stats
    }

    fn decide(&mut self, i: usize) -> Option<Direction> {
        let (heading, leaf, id, position, carrying) = {
            let a = &self.ants[i];
            (a.heading, a.leaf, a.id, a.position, a.carrying)
        };
        let obs = observe(&self.ants[i], &self.world);
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
        let h = policy.entropy_fraction;
        let smooth_scale = policy.deformation.smooth_share() * h * self.config.geometry.smooth_max;
        let smoothed = base.smoothed(smooth_scale);
        let rough_amplitude = policy.deformation.rough_share() * h * base.range().max(1.0);
        let deformed = smoothed.roughened(
            rough_amplitude,
            self.config.geometry.rough_modes,
            &mut self.rng,
        );
        let tempered = deformed.temper(h);

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
                // Every roughened landscape is tempered to the same target,
                // so the entropy of their mixture exceeds the target by
                // exactly the disorder the field carries between decisions.
                let mut mixture = tempered.probs;
                for _ in 0..samples {
                    let other = smoothed
                        .roughened(
                            rough_amplitude,
                            self.config.geometry.rough_modes,
                            &mut self.rng,
                        )
                        .temper(h);
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

    fn credit(&mut self, leaf: usize, reward: f64, delivered: bool) {
        self.stats.reward += reward;
        for &node in &self.leaf_paths[leaf] {
            self.stats.reward_by_node[node] += reward;
            if delivered {
                self.stats.delivered_by_node[node] += 1;
            }
        }
    }

    fn step_ant(&mut self, i: usize) {
        let leaf = self.ants[i].leaf;
        if let Some(dir) = self.decide(i) {
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
            let ant = &mut self.ants[i];
            ant.position = to;
            ant.heading = dir;
            ant.remember(from);
        }

        // Lay pheromone on the cell the ant now stands on.
        let (pos, home_amt, food_amt) = {
            let ant = &self.ants[i];
            let p = &self.config.pheromone;
            if ant.carrying {
                let strength =
                    p.food_deposit / (1.0 + p.deposit_decay * ant.steps_since_food as f64);
                (ant.position, 0.0, strength)
            } else {
                let strength =
                    p.home_deposit / (1.0 + p.deposit_decay * ant.steps_since_nest as f64);
                (ant.position, strength, 0.0)
            }
        };
        self.world.deposit(pos, home_amt, food_amt);

        // Interact with the cell.
        let reward = self.config.reward.clone();
        let nest = self.world.nest();
        let nest_radius = self.config.world.nest_radius.max(0);
        let mut delivered: Option<(u64, u64)> = None;
        let mut picked = false;
        let mut at_nest = false;
        if let Some(cell) = self.world.cell_mut(pos) {
            let ant = &mut self.ants[i];
            at_nest = cell.terrain == Terrain::Nest;
            if !ant.carrying && cell.food > 0 {
                cell.food -= 1;
                ant.carrying = true;
                ant.steps_since_food = 0;
                ant.pickup = Some(pos);
                ant.heading = ant.heading.opposite();
                picked = true;
            } else if ant.carrying && at_nest {
                ant.carrying = false;
                ant.deliveries += 1;
                ant.heading = ant.heading.opposite();
                // Straight-line distance to the nearest nest cell, so a
                // perfectly direct return scores exactly 1.
                let direct = ant
                    .pickup
                    .take()
                    .map(|p| (p.chebyshev(nest) - nest_radius).max(0))
                    .unwrap_or(0) as u64;
                delivered = Some((ant.steps_since_food as u64, direct));
            }
        }
        if picked {
            self.stats.food_picked += 1;
            self.credit(leaf, reward.food_picked, false);
        }
        if let Some((steps, direct)) = delivered {
            self.food_store += 1;
            self.stats.food_delivered += 1;
            self.stats.path.record_trip(steps, direct);
            for &node in &self.leaf_paths[leaf] {
                self.stats.path_by_node[node].record_trip(steps, direct);
            }
            self.credit(leaf, reward.food_delivered, true);
        }

        // Metabolism.
        let energy = self.config.energy.clone();
        let ant = &mut self.ants[i];
        if at_nest {
            ant.energy = energy.initial;
            ant.steps_since_nest = 0;
        }
        ant.energy -= energy.per_step;
        ant.age += 1;
        ant.steps_since_nest = ant.steps_since_nest.saturating_add(1);
        ant.steps_since_food = ant.steps_since_food.saturating_add(1);
        if energy.starvation && ant.energy <= 0.0 {
            ant.alive = false;
            let p = ant.position;
            if let Some(c) = self.world.cell_mut(p) {
                c.occupancy = c.occupancy.saturating_sub(1);
            }
            self.alive -= 1;
            self.stats.deaths += 1;
            self.credit(leaf, reward.death, false);
        }
    }

    fn reproduce(&mut self) {
        let cost = self.config.energy.spawn_cost;
        if cost == 0 {
            return;
        }
        while self.food_store >= cost && self.alive < self.config.energy.max_ants {
            self.food_store -= cost;
            let id = self.spawn_ant();
            let leaf = self.ants[id].leaf;
            self.stats.births += 1;
            let birth = self.config.reward.birth;
            self.credit(leaf, birth, false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::EntropyControl;
    use crate::surface::Deformation;

    fn quick_config() -> SimConfig {
        SimConfig {
            ants: 30,
            world: WorldConfig {
                width: 40,
                height: 30,
                nest: Position::new(20, 15),
                seed: Some(7),
                ..WorldConfig::default()
            },
            ..SimConfig::default()
        }
    }

    #[test]
    fn colony_delivers_food_with_instinct() {
        let mut sim = Simulation::new(quick_config(), 1);
        sim.run(400);
        let s = sim.stats();
        assert!(s.food_picked > 0, "ants should find food: {s:?}");
        assert!(s.food_delivered > 0, "ants should bring food home: {s:?}");
        assert!(s.decisions >= s.ticks * 30 - s.deaths * s.ticks);
        assert!(s.mean_entropy() > 0.0);
        assert!((s.mean_selected_entropy() - s.mean_entropy()).abs() < 1e-9);
        let total: u64 = sim
            .hierarchy()
            .leaves()
            .iter()
            .map(|&l| s.delivered_by_node[l])
            .sum();
        assert_eq!(total, s.food_delivered);
        assert_eq!(s.delivered_by_node[0], s.food_delivered);
        // Path statistics are consistent.
        assert_eq!(s.path.moves, s.path.turns.iter().sum::<u64>());
        assert_eq!(s.path.trips, s.food_delivered);
        assert!(s.path.trip_efficiency() > 0.0 && s.path.trip_efficiency() <= 1.0);
        assert!(s.path.turn_entropy() > 0.0);
        assert_eq!(s.path_by_node[0], s.path);
        let leaf_moves: u64 = sim
            .hierarchy()
            .leaves()
            .iter()
            .map(|&l| s.path_by_node[l].moves)
            .sum();
        assert_eq!(leaf_moves, s.path.moves);
    }

    #[test]
    fn deterministic_for_seed() {
        let mut a = Simulation::new(quick_config(), 5);
        let mut b = Simulation::new(quick_config(), 5);
        a.run(150);
        b.run(150);
        assert_eq!(a.stats(), b.stats());
        let mut c = Simulation::new(quick_config(), 6);
        c.run(150);
        assert_ne!(a.stats().decisions, 0);
        assert!(a.stats() != c.stats() || a.ants()[0].position != c.ants()[0].position);
    }

    #[test]
    fn entropy_dial_changes_behaviour() {
        let mut ordered = Simulation::new(quick_config(), 3);
        ordered.hierarchy_mut().node_mut(0).surface.entropy = EntropyControl::absolute(0.05);
        ordered.run(200);
        let mut chaotic = Simulation::new(quick_config(), 3);
        chaotic.hierarchy_mut().node_mut(0).surface.entropy = EntropyControl::absolute(0.98);
        chaotic.run(200);
        let h_low = ordered.stats().mean_entropy();
        let h_high = chaotic.stats().mean_entropy();
        assert!(h_low < 0.3, "low dial → low entropy, got {h_low}");
        assert!(h_high > 1.8, "high dial → near-uniform, got {h_high}");
    }

    #[test]
    fn relative_control_on_a_caste_only_affects_its_subtree() {
        let mut sim = Simulation::new(quick_config(), 3);
        sim.hierarchy_mut().node_mut(1).surface.entropy = EntropyControl::relative(2.5);
        sim.recompile();
        let policies = sim.policies();
        let leaves = sim.hierarchy().leaves().to_vec();
        for (idx, &leaf) in leaves.iter().enumerate() {
            let in_caste0 = sim.hierarchy().path(leaf).contains(&1);
            if in_caste0 {
                assert!((policies[idx].entropy_fraction - 0.875).abs() < 1e-9);
            } else {
                assert!((policies[idx].entropy_fraction - 0.35).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn trace_accumulates_only_when_enabled() {
        let mut off = Simulation::new(quick_config(), 2);
        off.run(20);
        assert!(off.trace().score_by_node[0].iter().all(|x| *x == 0.0));
        let cfg = SimConfig {
            trace: true,
            ..quick_config()
        };
        let mut on = Simulation::new(cfg, 2);
        on.run(20);
        let t = on.trace();
        assert_eq!(t.decisions_by_node[0], on.stats().decisions);
        assert!(t.score_by_node[0].iter().any(|x| x.abs() > 0.0));
        assert!(t.score_by_node[0][FEATURES..].iter().all(|x| *x == 0.0));
        let leaf_total: u64 = on
            .hierarchy()
            .leaves()
            .iter()
            .map(|&l| t.decisions_by_node[l])
            .sum();
        assert_eq!(leaf_total, t.decisions_by_node[0]);
    }

    #[test]
    fn starvation_and_reproduction() {
        let cfg = SimConfig {
            energy: EnergyConfig {
                initial: 5.0,
                per_step: 1.0,
                starvation: true,
                spawn_cost: 1,
                max_ants: 40,
            },
            ..quick_config()
        };
        let mut sim = Simulation::new(cfg, 4);
        sim.run(300);
        let s = sim.stats();
        assert!(s.deaths > 0);
        assert_eq!(sim.alive(), sim.living().count());
        assert_eq!(s.alive, sim.alive());
        let occupancy: u32 = sim.world().cells().iter().map(|c| c.occupancy as u32).sum();
        assert_eq!(occupancy as usize, sim.alive());
        if s.food_delivered > 0 {
            assert!(s.births > 0);
        }
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
        let mut sim = Simulation::new(quick_config(), 9);
        sim.run(60);
        let c = sim.stats().path.ledger.contributions();
        assert_eq!(sim.stats().path.ledger.decisions, sim.stats().decisions);
        assert!(c.smoothing.abs() < 1e-9, "no smoothing configured: {c:?}");
        assert!(c.roughening.abs() < 1e-9, "no roughening configured: {c:?}");
        assert!(c.field.abs() < 1e-9, "no field without roughening: {c:?}");
        assert!(c.selection.abs() < 1e-9, "global draw loses nothing: {c:?}");
        assert!((c.tempering - sim.stats().mean_entropy()).abs() < 1e-6);
        let off = SimConfig {
            geometry: GeometryConfig {
                ledger: false,
                ..GeometryConfig::default()
            },
            ..quick_config()
        };
        let mut sim = Simulation::new(off, 9);
        sim.run(10);
        assert_eq!(sim.stats().path.ledger.decisions, 0);
    }

    #[test]
    fn smoothing_and_roughening_show_up_in_the_ledger() {
        let mut smooth = Simulation::new(quick_config(), 10);
        smooth.hierarchy_mut().node_mut(0).surface.deformation = Deformation {
            smooth: 2.0,
            rough: 0.0,
            reach: 0.0,
        };
        smooth.run(60);
        let c = smooth.stats().path.ledger.contributions();
        assert!(c.smoothing > 0.05, "smoothing adds entropy: {c:?}");
        assert!(c.roughening.abs() < 1e-9);
        assert!((c.selected - smooth.stats().mean_entropy()).abs() < 1e-6);

        let mut rough = Simulation::new(quick_config(), 10);
        rough.hierarchy_mut().node_mut(0).surface.deformation = Deformation {
            smooth: 0.0,
            rough: 2.0,
            reach: 0.0,
        };
        rough.run(60);
        let c = rough.stats().path.ledger.contributions();
        assert!(
            c.roughening.abs() > 0.05,
            "roughening changes the landscape: {c:?}"
        );
        assert!(c.smoothing.abs() < 1e-9);
        assert!(
            c.field > 0.05,
            "the field randomises across decisions: {c:?}"
        );
        assert!((c.selected - rough.stats().mean_entropy()).abs() < 1e-6);
    }

    #[test]
    fn sucker_selection_runs_and_loses_entropy() {
        let cfg = SimConfig {
            selection: Selection::Sucker { reach: 8 },
            ..quick_config()
        };
        let mut sim = Simulation::new(cfg, 11);
        sim.run(300);
        let s = sim.stats();
        assert!(s.food_delivered > 0, "the sucker still forages: {s:?}");
        // A bounded walk realises a different entropy than the target it
        // crawls on (lower when the peak is ahead, higher while in transit
        // to an off-axis peak); the ledger accounts for the difference.
        let gap = s.mean_selected_entropy() - s.mean_entropy();
        assert!(
            gap.abs() > 1e-3,
            "sucker should change the realised entropy: {gap}"
        );
        let c = s.path.ledger.contributions();
        assert!((c.selection - gap).abs() < 1e-6);
        assert!((c.selected - s.mean_selected_entropy()).abs() < 1e-6);
        let policy = sim.policies()[0].clone();
        assert_eq!(sim.effective_reach(&policy), Some(8));
    }

    #[test]
    fn reach_parameter_scales_and_clamps() {
        let cfg = SimConfig {
            selection: Selection::Sucker { reach: 8 },
            ..quick_config()
        };
        let mut sim = Simulation::new(cfg, 12);
        sim.hierarchy_mut().node_mut(0).surface.deformation.reach = (4f64).ln();
        sim.recompile();
        let p = sim.policies()[0].clone();
        assert_eq!(sim.effective_reach(&p), Some(32));
        sim.hierarchy_mut().node_mut(0).surface.deformation.reach = 10.0;
        sim.recompile();
        let p = sim.policies()[0].clone();
        assert_eq!(sim.effective_reach(&p), Some(64));
        sim.hierarchy_mut().node_mut(0).surface.deformation.reach = -10.0;
        sim.recompile();
        let p = sim.policies()[0].clone();
        assert_eq!(sim.effective_reach(&p), Some(0));
        // Zero reach: always straight ahead when possible.
        sim.reset_stats();
        sim.run(40);
        let s = sim.stats();
        assert!(
            s.path.straight_rate() > 0.9,
            "reach 0 keeps heading: {}",
            s.path.straight_rate()
        );
        let plain = Simulation::new(quick_config(), 12);
        assert_eq!(plain.effective_reach(&plain.policies()[0].clone()), None);
    }

    #[test]
    fn surface_recording_keeps_the_last_rows() {
        let cfg = SimConfig {
            selection: Selection::Sucker { reach: 4 },
            record_surface: Some(3),
            surface_rows: 10,
            ..quick_config()
        };
        let mut sim = Simulation::new(cfg, 13);
        sim.run(30);
        let rows = sim.surface_trace();
        assert_eq!(rows.len(), 10);
        assert!(rows.windows(2).all(|w| w[0].tick < w[1].tick));
        for row in rows {
            assert!(row.valid[row.chosen]);
            assert_eq!(row.walk.len(), 5);
            assert_eq!(row.walk[row.walk.len() - 1] as usize, row.chosen);
            assert!((row.probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            assert!((row.selected.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            assert!(row.temperature > 0.0);
        }
        sim.clear_surface_trace();
        assert!(sim.surface_trace().is_empty());
        let mut plain = Simulation::new(quick_config(), 13);
        plain.run(5);
        assert!(plain.surface_trace().is_empty());
    }
}
