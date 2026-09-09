//! Practice: learners operate the mind's surfaces through levers whose
//! connections are hidden and rotate, as in the ants' arena
//! ([`ant_simulator::arena`]).
//!
//! Each turn the rotation connects every lever to a node of the mind's
//! hierarchy; the learner behind the lever proposes that node's
//! parameters (its surface weights, its entropy dial, its deformation)
//! seeing only the parameters, the turn and its last reward; the mind
//! runs a turn's worth of ticks on the problem; and the learner is told
//! the reward: the quality brought home per thought. The rotation is a
//! static random sequence, so which node a lever moves is learnable
//! from the rewards alone, and a phase-aware learner learns it.

use crate::mind::{Mind, MindConfig};
use crate::problem::Problem;
use ant_simulator::hierarchy::{Hierarchy, NodeId};
use ant_simulator::learner::{Learner, LeverView, Outcome};
use ant_simulator::rng::Rng;
use ant_simulator::rotation::{Mapping, Rotation, RotationError, RotationSchedule};
use std::fmt;

/// Which nodes of the hierarchy learners may control.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Targets {
    /// Every node, root included.
    AllNodes,
    /// Only the nodes at a depth (0 is the root, 1 the castes).
    Depth(usize),
    /// An explicit list.
    Nodes(Vec<NodeId>),
}

/// Configuration of a practice.
#[derive(Clone, Debug)]
pub struct PracticeConfig {
    /// The mind run each turn.
    pub mind: MindConfig,
    /// Ticks per turn.
    pub ticks_per_turn: u64,
    /// Controllable nodes.
    pub targets: Targets,
    /// The rotation of the levers.
    pub schedule: RotationSchedule,
    /// Keep one mind across turns instead of a fresh one each turn.
    pub persistent: bool,
    /// A fresh seed each turn (otherwise common random numbers).
    pub fresh_seeds: bool,
    /// Tell the learners which node they hold.
    pub reveal_mapping: bool,
    /// A survival threshold: with one, a turn's reward is 1 where the
    /// quality brought home per thought reaches it and 0 where not, so
    /// that the learners are conditioned on the colony's survival (the
    /// Q-process) rather than on its mean yield.
    pub survival: Option<f64>,
}

impl Default for PracticeConfig {
    fn default() -> Self {
        PracticeConfig {
            mind: MindConfig::default(),
            ticks_per_turn: 2000,
            targets: Targets::Depth(1),
            schedule: RotationSchedule::RandomStatic { period: 3 },
            persistent: false,
            fresh_seeds: false,
            reveal_mapping: false,
            survival: None,
        }
    }
}

/// Why a practice could not be set up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PracticeError {
    /// No learners.
    NoLearners,
    /// No controllable nodes.
    NoTargets,
    /// A target beyond the hierarchy.
    InvalidTarget(NodeId),
    /// The rotation could not be built.
    Rotation(RotationError),
}

impl fmt::Display for PracticeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PracticeError::NoLearners => write!(f, "no learners"),
            PracticeError::NoTargets => write!(f, "no controllable nodes"),
            PracticeError::InvalidTarget(id) => write!(f, "node {id} is not in the hierarchy"),
            PracticeError::Rotation(e) => write!(f, "rotation: {e}"),
        }
    }
}

impl std::error::Error for PracticeError {}

impl From<RotationError> for PracticeError {
    fn from(e: RotationError) -> Self {
        PracticeError::Rotation(e)
    }
}

/// What one turn did.
#[derive(Clone, Debug, PartialEq)]
pub struct Turn {
    /// The turn.
    pub turn: u64,
    /// The node behind each lever, if connected.
    pub node_of_lever: Vec<Option<NodeId>>,
    /// The reward: quality brought home per thought.
    pub reward: f64,
    /// Solutions brought home.
    pub deliveries: u64,
    /// The best quality brought home this turn.
    pub best: f64,
    /// Mean realised entropy per decision.
    pub mean_entropy: f64,
    /// The quality brought home per thought, before any survival
    /// threshold.
    pub yield_: f64,
    /// Whether the colony survived the turn (always, without a
    /// threshold).
    pub survived: bool,
}

/// A summary of a practice.
#[derive(Clone, Debug, PartialEq)]
pub struct PracticeReport {
    /// Turns played.
    pub turns: usize,
    /// Mean reward over the first quarter of the turns.
    pub early_reward: f64,
    /// Mean reward over the last quarter.
    pub late_reward: f64,
    /// The best turn's reward.
    pub best_reward: f64,
    /// Turns the colony survived.
    pub survived: usize,
    /// Each learner's name and state.
    pub learners: Vec<(String, String, Option<usize>)>,
}

impl fmt::Display for PracticeReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{} turns: reward {:.3} early, {:.3} late, {:.3} best, {} survived",
            self.turns, self.early_reward, self.late_reward, self.best_reward, self.survived
        )?;
        for (name, summary, period) in &self.learners {
            match period {
                Some(p) => writeln!(f, "  {name} (period {p}): {summary}")?,
                None => writeln!(f, "  {name}: {summary}")?,
            }
        }
        Ok(())
    }
}

/// Learners practising on a mind through rotating levers.
pub struct Practice<P: Problem + Clone> {
    problem: P,
    cfg: PracticeConfig,
    hierarchy: Hierarchy,
    targets: Vec<NodeId>,
    learners: Vec<Box<dyn Learner>>,
    rotation: Rotation,
    rng: Rng,
    seed: u64,
    turn: u64,
    history: Vec<Turn>,
    best: Option<(f64, Vec<f64>)>,
    last_reward: Vec<Option<f64>>,
    mind: Option<Mind<P>>,
    delivered_before: f64,
}

impl<P: Problem + Clone> Practice<P> {
    /// A practice of the learners on a problem.
    pub fn new(
        problem: P,
        cfg: PracticeConfig,
        learners: Vec<Box<dyn Learner>>,
        seed: u64,
    ) -> Result<Practice<P>, PracticeError> {
        if learners.is_empty() {
            return Err(PracticeError::NoLearners);
        }
        let template = Mind::new(problem.clone(), cfg.mind.clone());
        let hierarchy = template.hierarchy().clone();
        let targets = match &cfg.targets {
            Targets::AllNodes => (0..hierarchy.len()).collect(),
            Targets::Depth(d) => hierarchy.nodes_at_depth(*d),
            Targets::Nodes(list) => {
                for &id in list {
                    if id >= hierarchy.len() {
                        return Err(PracticeError::InvalidTarget(id));
                    }
                }
                list.clone()
            }
        };
        if targets.is_empty() {
            return Err(PracticeError::NoTargets);
        }
        let mut rng = Rng::seed_from_u64(seed ^ 0x5A5A_A5A5_8765_4321);
        let rotation = Rotation::new(
            cfg.schedule.clone(),
            learners.len(),
            targets.len(),
            &mut rng,
        )?;
        let n = learners.len();
        Ok(Practice {
            problem,
            cfg,
            hierarchy,
            targets,
            learners,
            rotation,
            rng,
            seed,
            turn: 0,
            history: Vec::new(),
            best: None,
            last_reward: vec![None; n],
            mind: None,
            delivered_before: 0.0,
        })
    }

    /// The hierarchy as the learners have left it.
    pub fn hierarchy(&self) -> &Hierarchy {
        &self.hierarchy
    }

    /// The controllable nodes.
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

    /// The turns played.
    pub fn history(&self) -> &[Turn] {
        &self.history
    }

    /// The best reward seen and the parameters that earned it.
    pub fn best(&self) -> Option<(f64, &[f64])> {
        self.best.as_ref().map(|(r, p)| (*r, p.as_slice()))
    }

    /// The mind kept across turns, if persistent.
    pub fn mind(&self) -> Option<&Mind<P>> {
        self.mind.as_ref()
    }

    /// The node behind each lever under a mapping.
    pub fn nodes_for(&self, mapping: &Mapping) -> Vec<Option<NodeId>> {
        mapping
            .surface_of_lever
            .iter()
            .map(|s| s.map(|s| self.targets[s]))
            .collect()
    }

    /// Play one turn.
    pub fn turn_once(&mut self) -> &Turn {
        let turn = self.turn;
        let mapping = self.rotation.next(&mut self.rng);
        let nodes = self.nodes_for(&mapping);
        let reveal = self.cfg.reveal_mapping;
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

        // The mind runs with the surfaces as they stand.
        let (reward, deliveries, best, mean_entropy) = {
            let ticks = self.cfg.ticks_per_turn;
            if self.cfg.persistent {
                if self.mind.is_none() {
                    let mut cfg = self.cfg.mind.clone();
                    cfg.seed = self.seed;
                    self.mind = Some(Mind::new(self.problem.clone(), cfg));
                }
                let mind = self.mind.as_mut().expect("a mind");
                mind.set_hierarchy(self.hierarchy.clone());
                let before = mind.stats().quality_sum;
                let deliveries_before = mind.stats().deliveries;
                let decisions_before = mind.stats().decisions;
                let entropy_before = mind.stats().entropy_sum;
                let best_before = mind.best().map(|b| b.quality).unwrap_or(0.0);
                mind.run(ticks);
                let s = mind.stats();
                let delivered = s.quality_sum - before;
                let n = mind.thoughts().len().max(1) as f64;
                let decisions = s.decisions - decisions_before;
                let mean_entropy = if decisions > 0 {
                    (s.entropy_sum - entropy_before) / decisions as f64
                } else {
                    0.0
                };
                self.delivered_before = s.quality_sum;
                (
                    delivered / n,
                    s.deliveries - deliveries_before,
                    mind.best()
                        .map(|b| b.quality)
                        .unwrap_or(0.0)
                        .max(best_before),
                    mean_entropy,
                )
            } else {
                let mut cfg = self.cfg.mind.clone();
                cfg.seed = if self.cfg.fresh_seeds {
                    self.seed
                        .wrapping_add(turn.wrapping_mul(0x9E37_79B9_7F4A_7C15))
                } else {
                    self.seed
                };
                let mut mind = Mind::new(self.problem.clone(), cfg);
                mind.set_hierarchy(self.hierarchy.clone());
                mind.run(ticks);
                let s = mind.stats();
                let n = mind.thoughts().len().max(1) as f64;
                (
                    s.quality_sum / n,
                    s.deliveries,
                    mind.best().map(|b| b.quality).unwrap_or(0.0),
                    s.mean_entropy(),
                )
            }
        };

        let yield_ = reward;
        let survived = self.cfg.survival.map(|t| yield_ >= t).unwrap_or(true);
        let reward = match self.cfg.survival {
            Some(_) => {
                if survived {
                    1.0
                } else {
                    0.0
                }
            }
            None => reward,
        };
        for (lever, node) in nodes.iter().enumerate() {
            let Some(node) = *node else { continue };
            let outcome = Outcome {
                turn,
                lever,
                reward,
                score: None,
                decisions: 0,
                mean_entropy,
            };
            self.learners[lever].feedback(&outcome, &mut self.rng);
            self.last_reward[lever] = Some(reward);
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
        if self.best.as_ref().map(|(r, _)| reward > *r).unwrap_or(true) {
            self.best = Some((reward, self.hierarchy.params()));
        }
        self.history.push(Turn {
            turn,
            node_of_lever: nodes,
            reward,
            deliveries,
            best,
            mean_entropy,
            yield_,
            survived,
        });
        self.turn += 1;
        self.history.last().expect("just pushed")
    }

    /// Play `turns` turns and summarise.
    pub fn run(&mut self, turns: usize) -> PracticeReport {
        for _ in 0..turns {
            self.turn_once();
        }
        self.report()
    }

    /// A summary of the turns played.
    pub fn report(&self) -> PracticeReport {
        let n = self.history.len();
        let quarter = (n / 4).max(1).min(n.max(1));
        let mean = |turns: &[Turn]| {
            if turns.is_empty() {
                0.0
            } else {
                turns.iter().map(|t| t.reward).sum::<f64>() / turns.len() as f64
            }
        };
        let early = mean(&self.history[..quarter.min(n)]);
        let late = mean(&self.history[n.saturating_sub(quarter)..]);
        PracticeReport {
            turns: n,
            early_reward: early,
            late_reward: late,
            best_reward: self.best.as_ref().map(|(r, _)| *r).unwrap_or(0.0),
            survived: self.history.iter().filter(|t| t.survived).count(),
            learners: self
                .learners
                .iter()
                .map(|l| (l.name().to_string(), l.summary(), l.inferred_period()))
                .collect(),
        }
    }
}

impl<P: Problem + Clone> fmt::Debug for Practice<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Practice")
            .field("turn", &self.turn)
            .field("targets", &self.targets)
            .field("learners", &self.learners.len())
            .finish()
    }
}
