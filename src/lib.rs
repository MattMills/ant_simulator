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
//!   molarity (some refilling), heaps of prey, corpses, landmarks, and six
//!   chemical channels (five signals and the smell of food) with literature
//!   kinetics under a temperature that may cycle through the day. Cells
//!   have a capacity, so traffic crowds, slows and spreads. Workers rest,
//!   nurse, forage, feed, return, search, unload and carry out the dead
//!   through an activity state machine; they navigate by path integration,
//!   one-way route memory and views of landmarks, follow trails through
//!   Deneubourg's choice function sensed by a forward antennal probe, find
//!   food by its smell, lay trail in proportion to food quality and crop
//!   load, collect sugar or protein as the larvae demand, and engage in
//!   tasks by reinforced response thresholds. Every worker carries its own
//!   crop; food spreads by trophallaxis, and a forager whose nestmates will
//!   not take its load stays in. The nest has an interior: workers inside
//!   stand on its cells and keep to zones that drift from the brood chamber
//!   to the entrance with age, nurse the brood where it lies, hand food on
//!   to neighbours only, fetch corpses from where they fall, and leave by
//!   the entrance ring. The nest keeps a protein store, lays eggs and
//!   raises brood through egg, larva and pupa. The [`experiments`]
//!   module reproduces the classic double-bridge, equal-bridge,
//!   crowded-bridge, two-source, dripping-source, odour, hunger,
//!   communal-nutrition, thermal, cemetery and division-of-labour setups.
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
//! * **Hive cognitive geometry** ([`hive`]): the colony's movement history
//!   collapsed over time, at two grains and two time constants, into an
//!   invariant skeleton (channels and loops) and a non-invariant residual;
//!   and a queen who thinks on a coarse clock, expresses her thought in the
//!   entropy dials, and reads it back from the field through fitted
//!   readouts, so that what the colony's paths remember of her past
//!   thoughts feeds her next one. [`experiments::run_memory_probe`]
//!   measures how much the field carries, lag by lag, and
//!   [`experiments::run_closed_loop`] closes the loop.
//!
//! * **Quadkeys, the behavioural memo and memoized transits** ([`quad`],
//!   [`memo`]): a quadtree over the world whose nodes compose from their
//!   children, so that the movement history and the memo of what ants do
//!   where (decisions, their entropy and turns, legs, loads, what was
//!   laid, what happened) exist at every grain, in an invariant and a
//!   variant record. The memo can be extracted, classified into kinds of
//!   ground and learned against. Transits through plain, invariant nodes
//!   are memoized in kernels keyed by how an ant entered, and mature
//!   kernels stand in for the simulation, as memoized neighbourhoods do
//!   in a cellular automaton, so that colonies of thousands run. The
//!   kernels are kept at several levels at once, and a coarser node
//!   stands in for a finer one's worth of transits where its transition
//!   is coherent.
//!
//! * **The field on two structures, the lens and the decision pipeline**
//!   ([`world`], [`lens`], [`pipeline`]): a substrate mark is kept at
//!   cell resolution where it has structure and as one mean per node of
//!   the quadtree where it is faint, a volatile lives on the grain
//!   throughout as volumetric information read as planes, and flat
//!   coarse nodes merge into blocks, so a large or shaped arena costs its
//!   structured ground rather than its bounding box; the lens
//!   tessellates the tree between two points, fine at both ends and
//!   coarse between, for a geodesic that is the same from either end;
//!   and the decision pipeline holds an ant's heading where its decision
//!   would have been the same and schedules the rest against a frame
//!   budget. `cargo bench -- shapes` measures larger and shaped arenas.
//!
//! * **The frame** ([`frame`]): surfaces joined along their edges on one
//!   grid, as the net of a box unfolds: an outworld box with its corners
//!   joined by portals and its walls sloping, a slab nest, tubes round
//!   whose backs the two sides meet, a lid over the box or a fluon band
//!   round its rim; each region standing somewhere in space, so a slip
//!   on a slope drops the ant straight down onto whatever lies below and
//!   stuns it. An ant walking over a fold turns its frame and the vectors
//!   it carries, the field flows through, the lens and the memoized
//!   transits see through the fold, walls block the sight of landmarks,
//!   and a trail is followed round a corner, so a colony forages over a
//!   formicarium's surfaces.
//!
//! * **The extras crate** (`ant_extras`, in the workspace): every
//!   component above re-leveraged as a general-purpose architecture
//!   that thinks in the style of ants, where a problem is embedded on
//!   a surface and walked by a colony of thoughts whose trails become
//!   the solution: a maze, a tour of cities, a graph colouring; and a
//!   path-topological network that makes the pheromone surface
//!   variable, registering the classes of routes walked (their words
//!   round the embedding's punctures) as learned symbols that label
//!   routes and generate them, and a holonomy embedding in which the
//!   thoughts integrate a high-dimensional state through the moves
//!   they make and a knowledge graph is learned by walking its
//!   queries; and a bridge that makes the colony's forward and
//!   backward filters explicit.
//!
//! * **Benchmarks and scale analysis** ([`scaling`], [`experiments`]):
//!   every simulation profiles its tick phase by phase; `cargo bench`
//!   sweeps colony size and world area and fits the cost's exponents; and
//!   the experiments module scans how foraging organises with colony size
//!   and how the hive's memory depends on colony size and grain.
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
pub mod frame;
pub mod geometry;
pub mod hierarchy;
pub mod hive;
pub mod landscape;
pub mod learner;
pub mod lens;
pub mod memo;
pub mod pheromone;
pub mod pipeline;
pub mod quad;
pub mod render;
pub mod rng;
pub mod rotation;
pub mod scaling;
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
    pub use crate::experiments::{cemetery_arena, run_cemetery, CemeteryOutcome};
    pub use crate::experiments::{
        crowded_bridge, double_bridge, dripping_source, experiment_config, pure_pheromone_feedback,
        run_crowded_bridge, run_division_of_labor, run_double_bridge, run_double_bridge_configured,
        run_double_bridge_once, run_hunger_response, run_productivity_response,
        run_two_sources_once, summarize, two_sources, BridgeOutcome, BridgeSpec, HungerOutcome,
        LaborOutcome, ProductivityOutcome, SourcesOutcome, Summary,
    };
    pub use crate::experiments::{hidden_source, run_discovery, DiscoveryOutcome};
    pub use crate::experiments::{
        run_closed_loop, run_memory_probe, sustained_colony, ClosedLoop, MemoryProbe,
        MemoryProbeConfig,
    };
    pub use crate::experiments::{
        run_colony_size_scan, run_memory_scaling, single_feeder, trail_only, MemoryScaleOutcome,
        SizeOutcome, SizeScan,
    };
    pub use crate::experiments::{run_communal_nutrition, sugar_and_prey, NutritionOutcome};
    pub use crate::experiments::{run_thermal_tradeoff, ThermalOutcome};
    pub use crate::frame::{Frame, Outworld, Region};
    pub use crate::geometry::{Direction, Point, Position};
    pub use crate::hierarchy::{
        EffectivePolicy, Hierarchy, HierarchySpec, LevelSpec, Node, NodeId,
    };
    pub use crate::hive::{
        memory_capacity, Capacity, Component, Epoch, FieldSummary, Flow, HistoryConfig,
        MovementHistory, Queen, QueenConfig, Readout, Recursion, Topology,
    };
    pub use crate::landscape::{
        ring_heading, ring_index_of, turn_degrees, turn_label, Contributions, EntropyLedger,
        Landscape, Sucker, Tempered, DISPLAY_ORDER, RING, RING_STEP,
    };
    pub use crate::learner::{
        Arm, CrossEntropy, DialBandit, EntropyBandit, HillClimber, Learner, LeverView, Mutation,
        Outcome, PeriodDetector, PhaseAware, PolicyGradient, RandomLearner, StaticLearner,
    };
    pub use crate::lens::{geodesic, Geodesic, Ground, Leaf, Lens};
    pub use crate::memo::{
        Classification, Kernel, Layer, Leg, Memo, MemoConfig, Signature, Stop, TransitConfig,
        TransitKey, TransitOutcome, Transits,
    };
    pub use crate::pheromone::{perceived, Pheromone, PheromoneParams, PheromoneSet};
    pub use crate::pipeline::{horizon, FrameStats, PipelineConfig};
    pub use crate::quad::{QuadKey, QuadTree};
    pub use crate::render::{render, render_surface};
    pub use crate::rng::Rng;
    pub use crate::rotation::{Mapping, Rotation, RotationSchedule};
    pub use crate::scaling::{
        ant_scan, area_scan, exponent, measure, Measurement, Phase, Profile, Scan, Shapes, Workload,
    };
    pub use crate::species::Species;
    pub use crate::surface::{
        BehavioralSurface, Deformation, ENTROPY_PARAM, PARAM_LEN, REACH_PARAM, ROUGH_PARAM,
        SMOOTH_PARAM,
    };
    pub use crate::world::{
        CapacityZone, Cell, Counter, CounterState, Edge, FoodSource, Nutrient, Portal, RandomFood,
        Rect, Side, Slope, Terrain, Warp, World, WorldConfig,
    };
}
