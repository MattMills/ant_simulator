//! # ant_simulator
//!
//! An ant colony simulator built around one idea: **behaviour is governed
//! through entropy, by a hierarchy of general objects, operated through
//! levers whose connections are hidden and rotate.**
//!
//! ## The pieces
//!
//! * **Colony simulation** ([`colony`], [`world`], [`ant`], [`pheromone`],
//!   [`species`]): a world in physical units, with ants moving continuously
//!   over a substrate grid that carries a nest, sucrose solutions of varying
//!   molarity, and five pheromone channels with literature kinetics under a
//!   temperature that may cycle through the day. Workers rest, nurse,
//!   forage, feed, return, search and unload through an activity state
//!   machine; they navigate by path integration and route memory, follow
//!   trails through Deneubourg's choice function sensed by a forward
//!   antennal probe, lay trail in proportion to food quality, recruit by
//!   contact, and engage in tasks by reinforced response thresholds. The
//!   nest stores sugar, gets hungry, lays eggs and raises brood through egg,
//!   larva and pupa. The [`experiments`] module reproduces the classic
//!   double-bridge, equal-bridge, two-source, hunger and division-of-labour
//!   setups.
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
//! * **Path surfaces and geometric selection** ([`landscape`]): sixteen
//!   candidate headings form a ring around the ant's direction of travel, and the
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
//! // Run a hungry colony of the default species for ten simulated minutes.
//! let mut cfg = SimConfig::default();
//! cfg.nest.initial_satiation = 0.1;
//! let mut sim = Simulation::new(cfg, 42);
//! sim.run_seconds(600.0);
//! println!("{}", render(&sim));
//! assert!(sim.stats().food_delivered > 0);
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
pub mod experiments;
pub mod geometry;
pub mod hierarchy;
pub mod landscape;
pub mod learner;
pub mod pheromone;
pub mod render;
pub mod rng;
pub mod rotation;
pub mod species;
pub mod surface;
pub mod world;

/// The commonly used types, re-exported.
pub mod prelude {
    pub use crate::ant::{
        Activity, Ant, Mode, Observation, Route, SearchTarget, Site, Traits, BASE_FEATURES,
        FEATURES, FEATURE_NAMES,
    };
    pub use crate::arena::{
        Arena, ArenaConfig, ArenaReport, ControlTargets, EpisodeSeeding, Evaluation, FeedbackScope,
        TurnRecord,
    };
    pub use crate::colony::{
        BroodItem, BroodStage, Environment, GeometryConfig, Nest, NestConfig, PathStats,
        RewardSpec, Selection, SimConfig, Simulation, Snapshot, Stats, SurfaceRow, Trace,
    };
    pub use crate::entropy::{EntropyControl, Tempering};
    pub use crate::experiments::{
        crowded_bridge, double_bridge, dripping_source, experiment_config, pure_pheromone_feedback,
        run_crowded_bridge, run_division_of_labor, run_double_bridge, run_double_bridge_configured,
        run_double_bridge_once, run_hunger_response, run_productivity_response,
        run_two_sources_once, summarize, two_sources, BridgeOutcome, BridgeSpec, HungerOutcome,
        LaborOutcome, ProductivityOutcome, SourcesOutcome, Summary,
    };
    pub use crate::geometry::{Direction, Point, Position};
    pub use crate::hierarchy::{
        EffectivePolicy, Hierarchy, HierarchySpec, LevelSpec, Node, NodeId,
    };
    pub use crate::landscape::{
        ring_heading, ring_index_of, turn_degrees, turn_label, Contributions, EntropyLedger,
        Landscape, Sucker, Tempered, DISPLAY_ORDER, RING, RING_STEP,
    };
    pub use crate::learner::{
        Arm, CrossEntropy, DialBandit, EntropyBandit, HillClimber, Learner, LeverView, Mutation,
        Outcome, PeriodDetector, PhaseAware, PolicyGradient, RandomLearner, StaticLearner,
    };
    pub use crate::pheromone::{perceived, Pheromone, PheromoneParams, PheromoneSet};
    pub use crate::render::{render, render_surface};
    pub use crate::rng::Rng;
    pub use crate::rotation::{Mapping, Rotation, RotationSchedule};
    pub use crate::species::Species;
    pub use crate::surface::{
        BehavioralSurface, Deformation, ENTROPY_PARAM, PARAM_LEN, REACH_PARAM, ROUGH_PARAM,
        SMOOTH_PARAM,
    };
    pub use crate::world::{
        CapacityZone, Cell, Counter, CounterState, FoodSource, RandomFood, Rect, Terrain, World,
        WorldConfig,
    };
}
