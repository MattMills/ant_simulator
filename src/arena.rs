//! The arena: many learners, hidden levers, one colony.
//!
//! Each turn the arena
//!
//! 1. asks the [`Rotation`] which surface every lever reaches,
//! 2. lets each connected learner act on the parameters of that surface,
//! 3. runs the colony for `steps_per_turn` ticks,
//! 4. hands every connected learner its reward (colony-wide or for the
//!    subtree it controlled) and lets it leave parameters behind.
//!
//! The learners never learn which node they held unless
//! [`ArenaConfig::reveal_mapping`] is set.

use crate::colony::{SimConfig, Simulation, Stats, Trace};
use crate::hierarchy::{Hierarchy, NodeId};
use crate::learner::{Learner, LeverView, Outcome};
use crate::rng::Rng;
use crate::rotation::{Mapping, Rotation, RotationError, RotationSchedule};
use std::fmt;

/// Which nodes of the hierarchy learners may control.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ControlTargets {
    /// Every node, root included.
    AllNodes,
    /// Only the nodes at a given depth (`0` is the root).
    Depth(usize),
    /// An explicit list of nodes.
    Nodes(Vec<NodeId>),
}

/// What reward a learner is told.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeedbackScope {
    /// The colony's total reward for the turn.
    Colony,
    /// The reward credited to the subtree the learner controlled.
    Subtree,
}

/// How episodes are seeded when the arena runs a fresh colony each turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpisodeSeeding {
    /// Same seed every turn (common random numbers: deterministic evaluation).
    Fixed,
    /// A new seed every turn.
    Fresh,
}

/// Arena configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct ArenaConfig {
    /// Colony configuration used for every episode.
    pub sim: SimConfig,
    /// Ticks simulated per turn.
    pub steps_per_turn: usize,
    /// Controllable nodes.
    pub targets: ControlTargets,
    /// Lever rotation schedule.
    pub schedule: RotationSchedule,
    /// Reward scope reported to learners.
    pub feedback: FeedbackScope,
    /// Episode seeding (ignored when `persistent`).
    pub seeding: EpisodeSeeding,
    /// Keep one colony alive across turns instead of restarting each turn.
    pub persistent: bool,
    /// Tell learners which node they hold.
    pub reveal_mapping: bool,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        ArenaConfig {
            sim: SimConfig::default(),
            steps_per_turn: 300,
            targets: ControlTargets::AllNodes,
            schedule: RotationSchedule::RandomStatic { period: 5 },
            feedback: FeedbackScope::Subtree,
            seeding: EpisodeSeeding::Fresh,
            persistent: false,
            reveal_mapping: false,
        }
    }
}

/// Error building an arena.
#[derive(Clone, Debug, PartialEq)]
pub enum ArenaError {
    /// No learners were supplied.
    NoLearners,
    /// The control targets select no node.
    NoTargets,
    /// A target node id does not exist.
    InvalidTarget(NodeId),
    /// The rotation could not be built.
    Rotation(RotationError),
}

impl fmt::Display for ArenaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArenaError::NoLearners => write!(f, "arena needs at least one learner"),
            ArenaError::NoTargets => write!(f, "control targets select no node"),
            ArenaError::InvalidTarget(id) => write!(f, "target node {id} does not exist"),
            ArenaError::Rotation(e) => write!(f, "rotation: {e}"),
        }
    }
}

impl std::error::Error for ArenaError {}

impl From<RotationError> for ArenaError {
    fn from(e: RotationError) -> Self {
        ArenaError::Rotation(e)
    }
}

/// What happened on one turn.
#[derive(Clone, Debug, PartialEq)]
pub struct TurnRecord {
    /// Turn number.
    pub turn: u64,
    /// Node each lever controlled (`None` if disconnected).
    pub node_of_lever: Vec<Option<NodeId>>,
    /// Colony reward for the turn.
    pub colony_reward: f64,
    /// Food delivered during the turn.
    pub food_delivered: u64,
    /// Mean realised decision entropy.
    pub mean_entropy: f64,
    /// Reward reported to each lever.
    pub lever_reward: Vec<Option<f64>>,
    /// Living ants at the end of the turn.
    pub alive: usize,
}

/// Per-learner summary in an [`ArenaReport`].
#[derive(Clone, Debug, PartialEq)]
pub struct LearnerSummary {
    /// Learner name.
    pub name: String,
    /// Turns on which the learner held a lever.
    pub turns_connected: usize,
    /// Mean reward over those turns.
    pub mean_reward: f64,
    /// Mean reward over the last quarter of those turns.
    pub recent_reward: f64,
    /// Period the learner believes it is subject to.
    pub inferred_period: Option<usize>,
    /// The learner's own description of its state.
    pub summary: String,
}

/// Summary of an arena run.
#[derive(Clone, Debug, PartialEq)]
pub struct ArenaReport {
    /// Turns simulated.
    pub turns: usize,
    /// True rotation period (`None` for an unlearnable schedule).
    pub schedule_period: Option<usize>,
    /// Mean colony reward over the first quarter of turns.
    pub early_reward: f64,
    /// Mean colony reward over the last quarter of turns.
    pub late_reward: f64,
    /// Best colony reward seen on any turn.
    pub best_reward: f64,
    /// Mean realised entropy over the last quarter of turns.
    pub late_entropy: f64,
    /// Per-learner summaries.
    pub learners: Vec<LearnerSummary>,
}

impl fmt::Display for ArenaReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "arena: {} turns, rotation period {}",
            self.turns,
            self.schedule_period
                .map(|p| p.to_string())
                .unwrap_or_else(|| "none (fresh every turn)".to_string())
        )?;
        writeln!(
            f,
            "colony reward: early {:.2} → late {:.2} (best {:.2}), late entropy {:.3} nats",
            self.early_reward, self.late_reward, self.best_reward, self.late_entropy
        )?;
        for l in &self.learners {
            writeln!(
                f,
                "  {:<28} turns {:>4}  mean {:>8.2}  recent {:>8.2}  period {:<5} {}",
                l.name,
                l.turns_connected,
                l.mean_reward,
                l.recent_reward,
                l.inferred_period
                    .map(|p| p.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                l.summary
            )?;
        }
        Ok(())
    }
}

/// Result of evaluating one parameter vector over several fresh episodes.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    /// Episodes run.
    pub episodes: usize,
    /// Mean colony reward.
    pub mean: f64,
    /// Standard deviation of colony reward.
    pub std: f64,
    /// Mean food delivered.
    pub delivered: f64,
    /// Mean realised decision entropy.
    pub entropy: f64,
}

impl fmt::Display for Evaluation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "reward {:.1} ± {:.1}, delivered {:.0}, entropy {:.3} nats ({} episodes)",
            self.mean, self.std, self.delivered, self.entropy, self.episodes
        )
    }
}

/// Learners, levers, rotation, and colony, run turn by turn.
pub struct Arena {
    config: ArenaConfig,
    hierarchy: Hierarchy,
    targets: Vec<NodeId>,
    learners: Vec<Box<dyn Learner>>,
    rotation: Rotation,
    rng: Rng,
    base_seed: u64,
    turn: u64,
    history: Vec<TurnRecord>,
    sim: Option<Simulation>,
    best: Option<(f64, Vec<f64>)>,
    last_reward: Vec<Option<f64>>,
}

impl Arena {
    /// Build an arena over the hierarchy described by `config.sim`.
    ///
    /// Set `config.sim.trace` when using [`crate::learner::PolicyGradient`];
    /// without a trace that learner leaves surfaces untouched.
    pub fn new(
        config: ArenaConfig,
        learners: Vec<Box<dyn Learner>>,
        seed: u64,
    ) -> Result<Self, ArenaError> {
        if learners.is_empty() {
            return Err(ArenaError::NoLearners);
        }
        let hierarchy = Hierarchy::from_spec(&config.sim.hierarchy, config.sim.instinct.clone());
        let targets = match &config.targets {
            ControlTargets::AllNodes => (0..hierarchy.len()).collect(),
            ControlTargets::Depth(d) => hierarchy.nodes_at_depth(*d),
            ControlTargets::Nodes(list) => {
                for &id in list {
                    if id >= hierarchy.len() {
                        return Err(ArenaError::InvalidTarget(id));
                    }
                }
                list.clone()
            }
        };
        if targets.is_empty() {
            return Err(ArenaError::NoTargets);
        }
        let mut rng = Rng::seed_from_u64(seed ^ 0xA5A5_5A5A_1234_5678);
        let rotation = Rotation::new(
            config.schedule.clone(),
            learners.len(),
            targets.len(),
            &mut rng,
        )?;
        let n = learners.len();
        Ok(Arena {
            config,
            hierarchy,
            targets,
            learners,
            rotation,
            rng,
            base_seed: seed,
            turn: 0,
            history: Vec::new(),
            sim: None,
            best: None,
            last_reward: vec![None; n],
        })
    }

    /// Configuration.
    pub fn config(&self) -> &ArenaConfig {
        &self.config
    }

    /// The hierarchy as shaped by the learners so far.
    pub fn hierarchy(&self) -> &Hierarchy {
        &self.hierarchy
    }

    /// Nodes the levers can reach, in surface-index order.
    pub fn targets(&self) -> &[NodeId] {
        &self.targets
    }

    /// The learners.
    pub fn learners(&self) -> &[Box<dyn Learner>] {
        &self.learners
    }

    /// The rotation.
    pub fn rotation(&self) -> &Rotation {
        &self.rotation
    }

    /// Records of every turn so far.
    pub fn history(&self) -> &[TurnRecord] {
        &self.history
    }

    /// Turns completed.
    pub fn turn(&self) -> u64 {
        self.turn
    }

    /// Best colony reward seen and the hierarchy parameters that produced it.
    pub fn best(&self) -> Option<(f64, &[f64])> {
        self.best.as_ref().map(|(r, p)| (*r, p.as_slice()))
    }

    /// The persistent colony, if the arena runs one.
    pub fn simulation(&self) -> Option<&Simulation> {
        self.sim.as_ref()
    }

    /// The hierarchy's parameters as they were before any learner acted.
    pub fn initial_params(&self) -> Vec<f64> {
        Hierarchy::from_spec(&self.config.sim.hierarchy, self.config.sim.instinct.clone()).params()
    }

    /// Run `episodes` fresh episodes of `steps_per_turn` ticks with the given
    /// hierarchy parameters (seeded from `seed`) and summarise the colony
    /// reward. Useful for comparing the instinct, the learned hierarchy, and
    /// the best parameters on equal footing.
    pub fn evaluate(&self, params: &[f64], episodes: usize, seed: u64) -> Evaluation {
        let mut hierarchy = self.hierarchy.clone();
        hierarchy
            .set_params(params)
            .expect("parameter vector must match the hierarchy");
        let mut rng = Rng::seed_from_u64(seed);
        let mut rewards = Vec::with_capacity(episodes);
        let mut delivered = 0.0;
        let mut entropy = 0.0;
        for _ in 0..episodes {
            let mut sim = Simulation::with_hierarchy(
                self.config.sim.clone(),
                hierarchy.clone(),
                rng.next_u64(),
            );
            sim.run(self.config.steps_per_turn);
            let s = sim.stats();
            rewards.push(s.reward);
            delivered += s.food_delivered as f64;
            entropy += s.mean_entropy();
        }
        let n = episodes.max(1) as f64;
        let mean = rewards.iter().sum::<f64>() / n;
        let var = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        Evaluation {
            episodes,
            mean,
            std: var.sqrt(),
            delivered: delivered / n,
            entropy: entropy / n,
        }
    }

    /// The node behind each lever for a mapping.
    pub fn nodes_for(&self, mapping: &Mapping) -> Vec<Option<NodeId>> {
        mapping
            .surface_of_lever
            .iter()
            .map(|s| s.map(|s| self.targets[s]))
            .collect()
    }

    /// Human-readable rendering of a mapping using node names.
    pub fn describe_mapping(&self, mapping: &Mapping) -> String {
        self.nodes_for(mapping)
            .iter()
            .enumerate()
            .map(|(lever, node)| match node {
                Some(n) => format!(
                    "{}→{}",
                    self.learners[lever].name(),
                    self.hierarchy.node(*n).name
                ),
                None => format!("{}→∅", self.learners[lever].name()),
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn run_episode(&mut self) -> (Stats, Trace) {
        if self.config.persistent {
            let params = self.hierarchy.params();
            let sim = self.sim.get_or_insert_with(|| {
                Simulation::with_hierarchy(
                    self.config.sim.clone(),
                    self.hierarchy.clone(),
                    self.base_seed,
                )
            });
            sim.set_params(&params).expect("hierarchy shapes match");
            sim.reset_stats();
            sim.run(self.config.steps_per_turn);
            (sim.stats().clone(), sim.trace().clone())
        } else {
            let seed = match self.config.seeding {
                EpisodeSeeding::Fixed => self.base_seed,
                EpisodeSeeding::Fresh => self.rng.next_u64(),
            };
            let mut sim =
                Simulation::with_hierarchy(self.config.sim.clone(), self.hierarchy.clone(), seed);
            sim.run(self.config.steps_per_turn);
            (sim.stats().clone(), sim.trace().clone())
        }
    }

    /// Play one turn.
    pub fn turn_once(&mut self) -> &TurnRecord {
        let turn = self.turn;
        let mapping = self.rotation.next(&mut self.rng);
        let nodes = self.nodes_for(&mapping);
        let reveal = self.config.reveal_mapping;

        for (lever, node) in nodes.iter().enumerate() {
            let Some(node) = *node else { continue };
            let params = self.hierarchy.node(node).surface.to_params();
            let view = LeverView {
                turn,
                lever,
                params: &params,
                last_reward: self.last_reward[lever],
                revealed_node: if reveal { Some(node) } else { None },
            };
            let proposal = self.learners[lever].act(&view, &mut self.rng);
            self.hierarchy
                .node_mut(node)
                .surface
                .set_params(&proposal)
                .unwrap_or_else(|e| {
                    panic!(
                        "learner '{}' returned a bad parameter vector: {e}",
                        self.learners[lever].name()
                    )
                });
        }

        let (stats, trace) = self.run_episode();

        let mut lever_reward = vec![None; self.learners.len()];
        for (lever, node) in nodes.iter().enumerate() {
            let Some(node) = *node else { continue };
            let reward = match self.config.feedback {
                FeedbackScope::Colony => stats.reward,
                FeedbackScope::Subtree => stats.reward_by_node[node],
            };
            let score = if self.config.sim.trace {
                Some(trace.score_by_node[node].clone())
            } else {
                None
            };
            let outcome = Outcome {
                turn,
                lever,
                reward,
                score,
                decisions: trace.decisions_by_node[node],
                mean_entropy: stats.mean_entropy(),
            };
            self.learners[lever].feedback(&outcome, &mut self.rng);
            self.last_reward[lever] = Some(reward);
            lever_reward[lever] = Some(reward);

            let params = self.hierarchy.node(node).surface.to_params();
            let view = LeverView {
                turn,
                lever,
                params: &params,
                last_reward: Some(reward),
                revealed_node: if reveal { Some(node) } else { None },
            };
            if let Some(left) = self.learners[lever].release(&view, &mut self.rng) {
                self.hierarchy
                    .node_mut(node)
                    .surface
                    .set_params(&left)
                    .unwrap_or_else(|e| {
                        panic!(
                            "learner '{}' released a bad parameter vector: {e}",
                            self.learners[lever].name()
                        )
                    });
            }
        }

        if self
            .best
            .as_ref()
            .map(|(r, _)| stats.reward > *r)
            .unwrap_or(true)
        {
            self.best = Some((stats.reward, self.hierarchy.params()));
        }

        self.history.push(TurnRecord {
            turn,
            node_of_lever: nodes,
            colony_reward: stats.reward,
            food_delivered: stats.food_delivered,
            mean_entropy: stats.mean_entropy(),
            lever_reward,
            alive: stats.alive,
        });
        self.turn += 1;
        self.history.last().expect("just pushed")
    }

    /// Play `turns` turns and summarise.
    pub fn run(&mut self, turns: usize) -> ArenaReport {
        for _ in 0..turns {
            self.turn_once();
        }
        self.report()
    }

    /// Summarise everything played so far.
    pub fn report(&self) -> ArenaReport {
        let n = self.history.len();
        let quarter = (n / 4).max(1).min(n.max(1));
        let mean = |records: &[TurnRecord], f: &dyn Fn(&TurnRecord) -> f64| -> f64 {
            if records.is_empty() {
                0.0
            } else {
                records.iter().map(f).sum::<f64>() / records.len() as f64
            }
        };
        let early = &self.history[..quarter.min(n)];
        let late = &self.history[n.saturating_sub(quarter)..];
        let learners = self
            .learners
            .iter()
            .enumerate()
            .map(|(lever, l)| {
                let rewards: Vec<f64> = self
                    .history
                    .iter()
                    .filter_map(|r| r.lever_reward[lever])
                    .collect();
                let k = rewards.len();
                let q = (k / 4).max(1);
                let mean_reward = if k == 0 {
                    0.0
                } else {
                    rewards.iter().sum::<f64>() / k as f64
                };
                let recent = &rewards[k.saturating_sub(q)..];
                let recent_reward = if recent.is_empty() {
                    0.0
                } else {
                    recent.iter().sum::<f64>() / recent.len() as f64
                };
                LearnerSummary {
                    name: l.name().to_string(),
                    turns_connected: k,
                    mean_reward,
                    recent_reward,
                    inferred_period: l.inferred_period(),
                    summary: l.summary(),
                }
            })
            .collect();
        ArenaReport {
            turns: n,
            schedule_period: self.rotation.period(),
            early_reward: mean(early, &|r| r.colony_reward),
            late_reward: mean(late, &|r| r.colony_reward),
            best_reward: self.best.as_ref().map(|(r, _)| *r).unwrap_or(0.0),
            late_entropy: mean(late, &|r| r.mean_entropy),
            learners,
        }
    }
}

impl fmt::Debug for Arena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Arena")
            .field("turn", &self.turn)
            .field("targets", &self.targets)
            .field(
                "learners",
                &self.learners.iter().map(|l| l.name()).collect::<Vec<_>>(),
            )
            .field("schedule", self.rotation.schedule())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Position;
    use crate::learner::{EntropyBandit, HillClimber, PhaseAware, StaticLearner};
    use crate::world::WorldConfig;

    fn small_config() -> ArenaConfig {
        ArenaConfig {
            sim: SimConfig {
                ants: 20,
                world: WorldConfig {
                    width: 32,
                    height: 24,
                    nest: Position::new(16, 12),
                    seed: Some(3),
                    ..WorldConfig::default()
                },
                ..SimConfig::default()
            },
            steps_per_turn: 60,
            ..ArenaConfig::default()
        }
    }

    #[test]
    fn turns_record_mappings_and_rewards() {
        let learners: Vec<Box<dyn Learner>> = vec![
            Box::new(StaticLearner),
            Box::new(HillClimber::new(0.1)),
            Box::new(EntropyBandit::default()),
        ];
        let mut arena = Arena::new(small_config(), learners, 1).unwrap();
        let record = arena.turn_once().clone();
        assert_eq!(record.turn, 0);
        assert_eq!(record.node_of_lever.len(), 3);
        assert_eq!(record.node_of_lever.iter().flatten().count(), 3);
        assert!(record.lever_reward.iter().all(|r| r.is_some()));
        let report = arena.run(9);
        assert_eq!(report.turns, 10);
        assert_eq!(report.schedule_period, Some(5));
        assert_eq!(report.learners.len(), 3);
        assert_eq!(report.learners[0].turns_connected, 10);
        assert!(format!("{report}").contains("hill-climb"));
        // The rotation replays: turn 0 and turn 5 use the same mapping.
        assert_eq!(
            arena.history()[0].node_of_lever,
            arena.history()[5].node_of_lever
        );
        assert!(arena.best().is_some());
    }

    #[test]
    fn static_learners_leave_the_hierarchy_alone() {
        let learners: Vec<Box<dyn Learner>> =
            vec![Box::new(StaticLearner), Box::new(StaticLearner)];
        let mut arena = Arena::new(small_config(), learners, 2).unwrap();
        let before = arena.hierarchy().params();
        arena.run(6);
        assert_eq!(arena.hierarchy().params(), before);
    }

    #[test]
    fn persistent_colony_keeps_state_between_turns() {
        let config = ArenaConfig {
            persistent: true,
            ..small_config()
        };
        let learners: Vec<Box<dyn Learner>> = vec![Box::new(StaticLearner)];
        let mut arena = Arena::new(config, learners, 2).unwrap();
        arena.run(3);
        let sim = arena.simulation().unwrap();
        assert_eq!(sim.tick(), 180);
        assert_eq!(sim.stats().ticks, 60);
    }

    #[test]
    fn fixed_seeding_is_deterministic_per_turn() {
        let config = ArenaConfig {
            seeding: EpisodeSeeding::Fixed,
            schedule: RotationSchedule::Fixed,
            ..small_config()
        };
        let learners: Vec<Box<dyn Learner>> = vec![Box::new(StaticLearner)];
        let mut arena = Arena::new(config, learners, 2).unwrap();
        arena.run(3);
        let h = arena.history();
        assert_eq!(h[0].colony_reward, h[1].colony_reward);
        assert_eq!(h[1].colony_reward, h[2].colony_reward);
    }

    #[test]
    fn target_selection_and_errors() {
        let config = ArenaConfig {
            targets: ControlTargets::Depth(1),
            ..small_config()
        };
        let arena = Arena::new(config, vec![Box::new(StaticLearner)], 1).unwrap();
        assert_eq!(arena.targets(), &[1, 4, 7]);

        let bad = ArenaConfig {
            targets: ControlTargets::Nodes(vec![99]),
            ..small_config()
        };
        assert_eq!(
            Arena::new(bad, vec![Box::new(StaticLearner)], 1).err(),
            Some(ArenaError::InvalidTarget(99))
        );
        let none = ArenaConfig {
            targets: ControlTargets::Depth(9),
            ..small_config()
        };
        assert_eq!(
            Arena::new(none, vec![Box::new(StaticLearner)], 1).err(),
            Some(ArenaError::NoTargets)
        );
        assert_eq!(
            Arena::new(small_config(), vec![], 1).err(),
            Some(ArenaError::NoLearners)
        );
        let zero = ArenaConfig {
            schedule: RotationSchedule::RandomStatic { period: 0 },
            ..small_config()
        };
        assert!(matches!(
            Arena::new(zero, vec![Box::new(StaticLearner)], 1).err(),
            Some(ArenaError::Rotation(RotationError::ZeroPeriod))
        ));
    }

    #[test]
    fn evaluate_compares_parameter_vectors() {
        let arena = Arena::new(small_config(), vec![Box::new(StaticLearner)], 3).unwrap();
        let instinct = arena.initial_params();
        assert_eq!(instinct, arena.hierarchy().params());
        let a = arena.evaluate(&instinct, 3, 1);
        let b = arena.evaluate(&instinct, 3, 1);
        assert_eq!(a, b, "evaluation is deterministic for a seed");
        assert_eq!(a.episodes, 3);
        let mut chaotic = instinct.clone();
        chaotic[crate::surface::ENTROPY_PARAM] = 6.0;
        let c = arena.evaluate(&chaotic, 3, 1);
        assert!(c.entropy > a.entropy);
        assert!(format!("{a}").contains("episodes"));
    }

    #[test]
    fn reveal_and_describe() {
        let config = ArenaConfig {
            reveal_mapping: true,
            schedule: RotationSchedule::Fixed,
            ..small_config()
        };
        let learners: Vec<Box<dyn Learner>> = vec![
            Box::new(PhaseAware::new(HillClimber::new(0.1), 4)),
            Box::new(StaticLearner),
        ];
        let mut arena = Arena::new(config, learners, 5).unwrap();
        arena.turn_once();
        let mapping = arena.rotation().mapping_at(0).unwrap().clone();
        let text = arena.describe_mapping(&mapping);
        assert!(text.contains("phase-aware(hill-climb)→colony"));
        assert!(text.contains("static→caste-0"));
        assert!(format!("{arena:?}").contains("Arena"));
    }
}
