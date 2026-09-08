//! # ant_simulator
//!
//! An ant colony simulator built around one idea: **behaviour is governed
//! through entropy, by a hierarchy of general objects, operated through
//! levers whose connections are hidden and rotate.**
//!
//! ## The pieces
//!
//! * **Colony simulation** ([`colony`], [`world`], [`ant`]): a grid world
//!   with a nest, food clusters, and two evaporating, diffusing pheromone
//!   fields. Every ant senses its eight neighbouring cells as feature
//!   vectors, scores them, draws a direction, moves, lays pheromone, picks up
//!   and delivers food, burns energy, and may starve. The colony spends
//!   delivered food on new ants.
//!
//! * **Entropic behavioral surface** ([`surface`], [`entropy`]): the general
//!   object that controls behaviour. It is a weight vector over the sensory
//!   features (the *surface*: preferences over what to do) plus an
//!   [`entropy::EntropyControl`] (the dial: how much disorder the action
//!   distribution must carry). Actions are drawn from the distribution whose
//!   entropy is exactly the requested fraction of the maximum, solved for on
//!   every decision. Turning the dial down makes the ants follow the surface
//!   deterministically; turning it up dissolves them into a random walk.
//!
//! * **Hierarchical class system** ([`hierarchy`]): a tree of those objects
//!   (`colony → castes → squads → ants` by default, any shape you like).
//!   Every level is the same object. Surfaces add up along the path from the
//!   root to a leaf, and entropy dials compose, so each level controls the
//!   behaviour of every entity beneath it.
//!
//! * **Path surfaces and geometric selection** ([`landscape`]): the eight
//!   candidate directions form a ring around the ant's heading, and the
//!   scores on that ring are the deterministic information of the path.
//!   The entropy budget deforms it through separable channels (tempering,
//!   smoothing along the ring, a random roughening field), and a direction
//!   is selected either by a global draw or by a *sucker* that crawls the
//!   ring for a bounded reach. An entropy ledger says where each decision's
//!   disorder came from, and path statistics say what it did to the paths.
//!
//! * **Learners and rotating levers** ([`learner`], [`rotation`], [`arena`]):
//!   several learners each hold a lever. Each turn the arena connects the
//!   levers to nodes of the hierarchy according to a [`rotation::Rotation`],
//!   the default being a random sequence of permutations drawn once and
//!   replayed forever: *random, sequential, static*. A learner never sees
//!   where its lever goes; it only sees the parameter vector at the far end,
//!   the turn number, and the reward that came back. Because the sequence is
//!   static it is learnable, and [`learner::PhaseAware`] shows how: it infers
//!   the period from its rewards and keeps a separate copy of an inner
//!   learner per phase.
//!
//! ## Quick start
//!
//! ```
//! use ant_simulator::prelude::*;
//!
//! // Run a colony on instinct alone.
//! let mut sim = Simulation::new(SimConfig::default(), 42);
//! sim.run(200);
//! println!("{}", render(&sim));
//!
//! // Turn the colony's entropy dial and watch behaviour change.
//! sim.hierarchy_mut().node_mut(0).surface.entropy = EntropyControl::absolute(0.9);
//! sim.reset_stats();
//! sim.run(50);
//! assert!(sim.stats().mean_entropy() > 1.5);
//! ```
//!
//! ```
//! use ant_simulator::prelude::*;
//!
//! // Spend the budget geometrically and select with a sucker of reach 4.
//! let config = SimConfig {
//!     selection: Selection::Sucker { reach: 4 },
//!     instinct: BehavioralSurface::instinct()
//!         .with_deformation(Deformation { smooth: 1.0, rough: 0.0, reach: 0.0 }),
//!     record_surface: Some(0),
//!     ..SimConfig::default()
//! };
//! let mut sim = Simulation::new(config, 3);
//! sim.run(60);
//! let c = sim.stats().path.ledger.contributions();
//! assert!(c.smoothing > 0.0);
//! println!("{}", render_surface(sim.surface_trace(), 8));
//! ```
//!
//! ```
//! use ant_simulator::prelude::*;
//!
//! // Several learners with hidden, rotating levers over the whole hierarchy.
//! let learners: Vec<Box<dyn Learner>> = vec![
//!     Box::new(PhaseAware::new(HillClimber::new(0.2), 8)),
//!     Box::new(HillClimber::new(0.2)),
//!     Box::new(PhaseAware::new(EntropyBandit::default(), 8)),
//! ];
//! let config = ArenaConfig {
//!     steps_per_turn: 40,
//!     schedule: RotationSchedule::RandomStatic { period: 3 },
//!     ..ArenaConfig::default()
//! };
//! let mut arena = Arena::new(config, learners, 7).unwrap();
//! let report = arena.run(6);
//! println!("{report}");
//! ```
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod ant;
pub mod arena;
pub mod colony;
pub mod entropy;
pub mod geometry;
pub mod hierarchy;
pub mod landscape;
pub mod learner;
pub mod render;
pub mod rng;
pub mod rotation;
pub mod surface;
pub mod world;

/// The commonly used types, re-exported.
pub mod prelude {
    pub use crate::ant::{Ant, Observation, FEATURES, FEATURE_NAMES};
    pub use crate::arena::{
        Arena, ArenaConfig, ArenaReport, ControlTargets, EpisodeSeeding, Evaluation, FeedbackScope,
        TurnRecord,
    };
    pub use crate::colony::{
        EnergyConfig, GeometryConfig, PathStats, PheromoneConfig, RewardSpec, Selection, SimConfig,
        Simulation, Stats, SurfaceRow, Trace,
    };
    pub use crate::entropy::EntropyControl;
    pub use crate::geometry::{Direction, Position};
    pub use crate::hierarchy::{
        EffectivePolicy, Hierarchy, HierarchySpec, LevelSpec, Node, NodeId,
    };
    pub use crate::landscape::{
        ring_index, world_direction, Contributions, EntropyLedger, Landscape, Sucker, Tempered,
        DISPLAY_ORDER, RING, TURN_LABELS,
    };
    pub use crate::learner::{
        Arm, CrossEntropy, DialBandit, EntropyBandit, HillClimber, Learner, LeverView, Mutation,
        Outcome, PeriodDetector, PhaseAware, PolicyGradient, RandomLearner, StaticLearner,
    };
    pub use crate::render::{render, render_surface};
    pub use crate::rng::Rng;
    pub use crate::rotation::{Mapping, Rotation, RotationSchedule};
    pub use crate::surface::{
        BehavioralSurface, Deformation, ENTROPY_PARAM, PARAM_LEN, REACH_PARAM, ROUGH_PARAM,
        SMOOTH_PARAM,
    };
    pub use crate::world::{FoodSource, RandomFood, Rect, Terrain, World, WorldConfig};
}
