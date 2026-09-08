//! The colony simulation: world, ants, and the hierarchy that governs them.
//!
//! Each tick every living ant senses its surroundings, scores the eight
//! candidate directions through the effective policy of its category, draws a
//! direction from the entropy-controlled distribution, moves, lays pheromone,
//! and interacts with food or the nest. Then pheromones evaporate and diffuse
//! and the colony spends stored food on new ants.

use crate::ant::{observe, Ant, FEATURES};
use crate::entropy::tempered_distribution;
use crate::geometry::Direction;
use crate::hierarchy::{EffectivePolicy, Hierarchy, HierarchySpec, NodeId};
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
    /// Sum of realised decision entropies (nats).
    pub entropy_sum: f64,
    /// Reward attributed to each node (every event credits the whole path).
    pub reward_by_node: Vec<f64>,
    /// Deliveries attributed to each node.
    pub delivered_by_node: Vec<u64>,
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
            ..Stats::default()
        }
    }

    /// Mean realised entropy per decision.
    pub fn mean_entropy(&self) -> f64 {
        if self.decisions == 0 {
            0.0
        } else {
            self.entropy_sum / self.decisions as f64
        }
    }
}

/// Policy-gradient bookkeeping: for every node, the sum over decisions made
/// beneath it of `∇ log π(a)` with respect to that node's parameters.
///
/// The gradient treats the solved temperature as a constant. The entropy
/// parameter's entry is always zero.
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
        let ant = &self.ants[i];
        let policy = &self.policies[ant.leaf];
        let obs = observe(ant, &self.world);
        let mut logits = [f64::NEG_INFINITY; Direction::COUNT];
        let mut any = false;
        for ((logit, valid), features) in logits.iter_mut().zip(&obs.valid).zip(&obs.features) {
            if *valid {
                *logit = crate::surface::dot(&policy.weights, features);
                any = true;
            }
        }
        if !any {
            return None;
        }
        let mut probs = [0.0; Direction::COUNT];
        let (temperature, h) = tempered_distribution(&logits, policy.entropy_fraction, &mut probs);
        let choice = self.rng.choose_weighted(&probs);
        self.stats.decisions += 1;
        self.stats.entropy_sum += h;
        if self.config.trace {
            let mut score = [0.0; FEATURES];
            for (p, features) in probs.iter().zip(&obs.features) {
                if *p > 0.0 {
                    for (s, f) in score.iter_mut().zip(features) {
                        *s -= p * f;
                    }
                }
            }
            for (s, f) in score.iter_mut().zip(&obs.features[choice]) {
                *s = (*s + f) / temperature;
            }
            let leaf = ant.leaf;
            for &node in &self.leaf_paths[leaf] {
                let acc = &mut self.trace.score_by_node[node];
                for (a, s) in acc.iter_mut().zip(&score) {
                    *a += s;
                }
                self.trace.decisions_by_node[node] += 1;
            }
        }
        Some(Direction::from_index(choice))
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
        if let Some(dir) = self.decide(i) {
            let from = self.ants[i].position;
            let to = from.step(dir);
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
        let leaf = self.ants[i].leaf;
        let reward = self.config.reward.clone();
        let mut delivered = false;
        let mut picked = false;
        let mut at_nest = false;
        if let Some(cell) = self.world.cell_mut(pos) {
            let ant = &mut self.ants[i];
            at_nest = cell.terrain == Terrain::Nest;
            if !ant.carrying && cell.food > 0 {
                cell.food -= 1;
                ant.carrying = true;
                ant.steps_since_food = 0;
                ant.heading = ant.heading.opposite();
                picked = true;
            } else if ant.carrying && at_nest {
                ant.carrying = false;
                ant.deliveries += 1;
                ant.heading = ant.heading.opposite();
                delivered = true;
            }
        }
        if picked {
            self.stats.food_picked += 1;
            self.credit(leaf, reward.food_picked, false);
        }
        if delivered {
            self.food_store += 1;
            self.stats.food_delivered += 1;
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
    use crate::geometry::Position;

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
        let total: u64 = sim
            .hierarchy()
            .leaves()
            .iter()
            .map(|&l| s.delivered_by_node[l])
            .sum();
        assert_eq!(total, s.food_delivered);
        assert_eq!(s.delivered_by_node[0], s.food_delivered);
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
        assert_eq!(t.score_by_node[0][PARAM_LEN - 1], 0.0);
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
}
