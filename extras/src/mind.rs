//! The mind: a colony of thoughts walking a problem over a medium.
//!
//! A mind is built from the components of the ant simulator, each in
//! the role it plays for the ants:
//!
//! * the **medium** is the ants' world ([`World`]): its chemical field
//!   on two structures (substrate marks at cell resolution where they
//!   have structure, volatiles on the grain) carries the trail the
//!   thoughts lay in proportion to the quality of what they bring back,
//!   the no-entry marks they leave at dead ends, the smell of solutions
//!   found, and the crowding of thoughts on their way;
//! * the **surface and the hierarchy** ([`Hierarchy`]) govern the
//!   thoughts: a root instinct (the ant's own), castes of scouts and
//!   followers whose entropy dials compose with it, and a leaf per
//!   thought;
//! * the **path surfaces** ([`Landscape`]) deform each decision's ring
//!   of moves with the entropy budget and select from it;
//! * the **movement history** ([`MovementHistory`]) collapses the
//!   thoughts' walks into an invariant skeleton and a residual, which
//!   the pipeline reads for its horizons and the queen for her thought;
//! * the **memo and the transits** ([`Memo`]) record what is done where
//!   at every grain and, where the ground is plain and invariant, stand
//!   in for the walk with a memoized transit: a habit;
//! * the **lens** ([`geodesic`]) foresees the route between where a
//!   thought is and where it means to go, fine at both ends and coarse
//!   between, which it follows as a plan;
//! * the **decision pipeline** ([`horizon`]) holds a thought's heading
//!   where its decision would have been the same and schedules the rest
//!   against a frame budget;
//! * the **queen** ([`Queen`]) thinks on a coarse clock from what the
//!   history shows her (how organised the walks have become, how long
//!   since anything better was found) and expresses her thought in the
//!   castes' dials.
//!
//! The thoughts themselves behave as foragers do: they leave the origin,
//! search or follow what they know, bring a solution home laying trail,
//! rest, and go out again with site fidelity and a route memory.

use crate::problem::{Moves, Problem};
use crate::sense::{clear_ahead, features, Body, Candidate, Senses, Sight};
use crate::thought::{unit, Activity, Site, Target, Thought};
use crate::topos::{move_letter, Label, Letter, PathNet, Route, SymbolConfig, Word};
use ant_simulator::ant::{Mode, FEATURES};
use ant_simulator::entropy::{entropy, EntropyControl, Tempering};
use ant_simulator::geometry::{angle_of, wrap_angle, Point, Position};
use ant_simulator::hierarchy::{EffectivePolicy, Hierarchy, HierarchySpec, LevelSpec, NodeId};
use ant_simulator::hive::{
    Component, HistoryConfig, MovementHistory, Queen, QueenConfig, Recursion,
};
use ant_simulator::landscape::{
    ring_heading, ring_index_of, turn_magnitude, EntropyLedger, Landscape, Sucker, RING, RING_STEP,
};
use ant_simulator::lens::geodesic;
use ant_simulator::memo::{
    Leg, Memo, MemoConfig, Stop, Transit, TransitKey, TransitRecord, INSIDE, MAX_TRANSIT_LEVELS,
};
use ant_simulator::pheromone::{Pheromone, PheromoneParams, PheromoneSet};
use ant_simulator::pipeline::{horizon, FrameStats, PipelineConfig};
use ant_simulator::rng::Rng;
use ant_simulator::species::Species;
use ant_simulator::surface::{dot, BehavioralSurface};
use ant_simulator::world::World;
use std::fmt::Write as _;
use std::hash::Hash;

/// How decisions are deformed and selected (see
/// [`ant_simulator::colony::GeometryConfig`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    /// Smoothing scale at a full entropy budget, in ring positions.
    pub smooth_max: f64,
    /// Fourier modes of the roughening field.
    pub rough_modes: usize,
    /// Longest reach of the sucker.
    pub max_reach: usize,
}

impl Default for Geometry {
    fn default() -> Self {
        Geometry {
            smooth_max: 3.0,
            rough_modes: 3,
            max_reach: 64,
        }
    }
}

/// How a move is selected from the tempered ring.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Choice {
    /// What fits the geometry: the sucker for free movement, where
    /// straight ahead means something, and a global draw for moves
    /// between states, where the ring is only where the states lie.
    #[default]
    Natural,
    /// A global draw.
    Softmax,
    /// A sucker crawling the ring from straight ahead for a bounded reach.
    Sucker {
        /// Base reach in proposal steps.
        reach: usize,
    },
}

impl Choice {
    /// The reach of the sucker used for a problem walked freely or not,
    /// if a sucker is used.
    pub fn reach(self, free: bool) -> Option<usize> {
        match self {
            Choice::Natural => free.then_some(8),
            Choice::Softmax => None,
            Choice::Sucker { reach } => Some(reach),
        }
    }
}

/// How the queen thinks: on a coarse clock, from how long it has been
/// since the mind found anything better (stagnation, which heats the
/// scouts) and how organised the thoughts' walks have become
/// (organisation, which cools the followers).
#[derive(Clone, Debug, PartialEq)]
pub struct QueenPolicy {
    /// Ticks per epoch.
    pub epoch_ticks: u64,
    /// Ticks before her first epoch.
    pub warmup_ticks: u64,
    /// Size of the expression: a thought component `x` sets a relative
    /// gain of `exp(expression × x)` on a caste's dial.
    pub expression: f64,
    /// Epochs without improvement that count as full stagnation.
    pub patience: u64,
    /// Recall of past thought from the field, once readouts are fitted.
    pub recursion: Option<Recursion>,
}

impl Default for QueenPolicy {
    fn default() -> Self {
        QueenPolicy {
            epoch_ticks: 200,
            warmup_ticks: 400,
            expression: 1.0,
            patience: 5,
            recursion: None,
        }
    }
}

/// Configuration of a mind.
#[derive(Clone, Debug)]
pub struct MindConfig {
    /// Number of thoughts.
    pub thoughts: usize,
    /// Share of the thoughts in the scout caste.
    pub scouts: f64,
    /// Leaves of the hierarchy per caste (thoughts share leaves).
    pub leaves_per_caste: usize,
    /// The root surface: the disposition every thought inherits.
    pub instinct: BehavioralSurface,
    /// Relative entropy gain of the scouts' dial.
    pub scout_heat: f64,
    /// Relative entropy gain of the followers' dial.
    pub follower_cool: f64,
    /// Selection from the ring.
    pub choice: Choice,
    /// Deformation geometry.
    pub geometry: Geometry,
    /// Temperature factor of a searching thought (one with nothing to go
    /// on).
    pub search_heat: f64,
    /// Cells walked per tick.
    pub speed: f64,
    /// Moves a trip may make before it is given up.
    pub trip_budget: usize,
    /// Ticks rested at the origin between trips.
    pub rest_ticks: u64,
    /// Trail laid per cell on the way home with a solution of quality 1.
    pub trail_deposit: f64,
    /// No-entry marking laid at a dead end (a quarter of it per cell on
    /// the way back from it).
    pub no_entry_deposit: f64,
    /// Cells of the way back from a dead end that are marked.
    pub no_entry_reach: f64,
    /// Odour released where a solution of quality 1 is found.
    pub odour_release: f64,
    /// Home marking laid per cell on the way out (none by default).
    pub home_deposit: f64,
    /// Whether a thought goes back to the site of its best finding.
    pub site_fidelity: bool,
    /// Empty-handed trips to a site before it is forgotten.
    pub site_patience: u32,
    /// Sharpness of recruitment: a solution recruits in proportion to
    /// `(quality / best the thought knows)` to this power, so a thought
    /// that knows better lays little for worse.
    pub recruitment: f64,
    /// Retreats a trip may make from dead ends, one state back each,
    /// before it is given up.
    pub retreats: usize,
    /// The movement history, if kept.
    pub history: Option<HistoryConfig>,
    /// The behavioural memo (with memoized transits when configured).
    pub memo: Option<MemoConfig>,
    /// The decision pipeline.
    pub pipeline: Option<PipelineConfig>,
    /// Foresight: the radius of the lens the plans are made with, if
    /// thoughts plan their routes.
    pub foresight: Option<f64>,
    /// The queen.
    pub queen: Option<QueenPolicy>,
    /// The path-topological network: the variable surface of learned
    /// symbols (see [`crate::topos`]).
    pub symbols: Option<SymbolConfig>,
    /// Kinetics of the medium's channels.
    pub pheromones: PheromoneSet,
    /// Seconds per tick.
    pub tick_s: f64,
    /// Whether the entropy ledger is kept.
    pub ledger: bool,
    /// Random seed.
    pub seed: u64,
}

impl Default for MindConfig {
    fn default() -> Self {
        MindConfig {
            thoughts: 64,
            scouts: 0.1,
            leaves_per_caste: 4,
            instinct: MindConfig::instinct(),
            scout_heat: 2.0,
            follower_cool: 0.7,
            choice: Choice::default(),
            geometry: Geometry::default(),
            search_heat: 1.5,
            speed: 1.0,
            trip_budget: 400,
            rest_ticks: 4,
            trail_deposit: 20.0,
            no_entry_deposit: 40.0,
            no_entry_reach: 8.0,
            odour_release: 200.0,
            home_deposit: 0.0,
            site_fidelity: true,
            site_patience: 3,
            recruitment: 4.0,
            retreats: 4,
            history: Some(HistoryConfig::default()),
            memo: None,
            pipeline: None,
            foresight: None,
            queen: None,
            symbols: None,
            pheromones: MindConfig::pheromones(),
            tick_s: 1.0,
            ledger: true,
            seed: 1,
        }
    }
}

impl MindConfig {
    /// The ant's own instinct ([`Species::lasius_niger`]): Deneubourg's
    /// choice exponent on the trail, path integration home, site
    /// fidelity out, at a temperature of one.
    pub fn instinct() -> BehavioralSurface {
        Species::lasius_niger().instinct()
    }

    /// The medium's kinetics: a trail that forgets in ten minutes, a
    /// no-entry mark that lasts a quarter of an hour, and the smell of a
    /// solution that spreads and fades in a minute.
    pub fn pheromones() -> PheromoneSet {
        let substrate = |half_life_s: f64, cap: f64| PheromoneParams {
            half_life_s,
            diffusion_per_s: 0.0002,
            cap,
            k: 20.0,
        };
        [
            substrate(600.0, 4000.0),
            substrate(600.0, 4000.0),
            PheromoneParams::inert(),
            PheromoneParams {
                half_life_s: 900.0,
                diffusion_per_s: 0.0005,
                cap: 2000.0,
                k: 20.0,
            },
            PheromoneParams {
                half_life_s: 40.0,
                diffusion_per_s: 0.15,
                cap: 200.0,
                k: 5.0,
            },
            PheromoneParams {
                half_life_s: 60.0,
                diffusion_per_s: 0.3,
                cap: 500.0,
                k: 5.0,
            },
        ]
    }

    /// A mind that keeps the memo with memoized transits (habits) at
    /// the given grain.
    pub fn with_habits(mut self, grain: usize) -> MindConfig {
        self.memo = Some(MemoConfig {
            grain,
            transits: Some(Default::default()),
            ..MemoConfig::default()
        });
        self
    }

    /// A mind with the decision pipeline.
    pub fn with_pipeline(mut self, pipeline: PipelineConfig) -> MindConfig {
        self.pipeline = Some(pipeline);
        self
    }

    /// A mind whose thoughts plan their routes through a lens of the
    /// given radius.
    pub fn with_foresight(mut self, radius: f64) -> MindConfig {
        self.foresight = Some(radius);
        self
    }

    /// A mind with a queen.
    pub fn with_queen(mut self, queen: QueenPolicy) -> MindConfig {
        self.queen = Some(queen);
        self
    }

    /// A mind with a path-topological network: routes are classed by
    /// their words, classes are registered as symbols with channels of
    /// their own, and the thoughts sense them with learned weights.
    pub fn with_symbols(mut self, symbols: SymbolConfig) -> MindConfig {
        self.symbols = Some(symbols);
        self
    }

    /// A mind whose instinct weighs a feature (an index from
    /// [`ant_simulator::ant`], `F_TRAIL` and the rest) as given on the
    /// way out.
    pub fn with_weight(mut self, feature: usize, weight: f64) -> MindConfig {
        self.instinct.weights[feature] = weight;
        self
    }

    /// A mind whose instinct weighs a feature as given on the way back.
    pub fn with_inbound_weight(mut self, feature: usize, weight: f64) -> MindConfig {
        self.instinct.weights[ant_simulator::ant::BASE_FEATURES + feature] = weight;
        self
    }

    /// A mind whose trail forgets with the given half-life, in seconds.
    pub fn with_trail_half_life(mut self, half_life_s: f64) -> MindConfig {
        self.pheromones[Pheromone::Trail.index()].half_life_s = half_life_s;
        self
    }
}

/// A solution brought home.
#[derive(Clone, Debug)]
pub struct Finding<S> {
    /// Its quality.
    pub quality: f64,
    /// The states walked to it, the origin first.
    pub route: Vec<S>,
    /// Their places.
    pub places: Vec<Point>,
    /// When it came home.
    pub tick: u64,
    /// Which thought brought it.
    pub thought: usize,
    /// The letters of its moves (see [`crate::topos`]).
    pub letters: Vec<Letter>,
    /// The letters its word begins with.
    pub prefix: Vec<Letter>,
    /// The integrated state it arrived with.
    pub x: Vec<f64>,
}

/// Counters of a mind.
#[derive(Clone, Debug, Default)]
pub struct MindStats {
    /// Ticks run.
    pub ticks: u64,
    /// Decisions made.
    pub decisions: u64,
    /// Realised entropy summed over decisions.
    pub entropy_sum: f64,
    /// Selected entropy summed over decisions.
    pub selected_entropy_sum: f64,
    /// Trips begun.
    pub trips: u64,
    /// Solutions found.
    pub findings: u64,
    /// Solutions brought home.
    pub deliveries: u64,
    /// Their qualities summed.
    pub quality_sum: f64,
    /// Dead ends met.
    pub dead_ends: u64,
    /// Retreats from them, one state back.
    pub retreats: u64,
    /// Trips given up on their budget.
    pub given_up: u64,
    /// Thoughts lost on the way home and recycled.
    pub lost: u64,
    /// Steps walked.
    pub moves: u64,
    /// Cells walked.
    pub length: f64,
    /// Transits replayed from habits.
    pub replayed: u64,
    /// Decisions those stood in for.
    pub decisions_replayed: u64,
    /// Moves made on a held heading.
    pub held: u64,
    /// Decisions deferred by the frame budget.
    pub deferred: u64,
    /// The queen's epochs.
    pub epochs: u64,
    /// Improvements of the best finding.
    pub improvements: u64,
    /// The pipeline's frames.
    pub frames: FrameStats,
    /// Where each decision's disorder came from.
    pub ledger: EntropyLedger,
}

impl MindStats {
    /// Mean realised entropy per decision.
    pub fn mean_entropy(&self) -> f64 {
        if self.decisions == 0 {
            0.0
        } else {
            self.entropy_sum / self.decisions as f64
        }
    }

    /// Mean quality of the solutions brought home.
    pub fn mean_quality(&self) -> f64 {
        if self.deliveries == 0 {
            0.0
        } else {
            self.quality_sum / self.deliveries as f64
        }
    }
}

/// What the pipeline says about a thought's next move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gate {
    Hold,
    Defer,
    Decide,
}

type Entry = (TransitKey, (usize, usize, usize, usize));

/// A colony of thoughts walking a problem.
pub struct Mind<P: Problem> {
    problem: P,
    cfg: MindConfig,
    world: World,
    hierarchy: Hierarchy,
    policies: Vec<EffectivePolicy>,
    castes: Vec<NodeId>,
    thoughts: Vec<Thought<P::State>>,
    origin: P::State,
    origin_place: Point,
    free: Option<f64>,
    history: Option<MovementHistory>,
    memo: Option<Memo>,
    queen: Option<Queen>,
    net: Option<PathNet>,
    rng: Rng,
    tick: u64,
    stats: MindStats,
    best: Option<Finding<P::State>>,
    last_improvement_epoch: u64,
    epoch_quality_mark: f64,
    peak_yield: f64,
    epochs_since_peak: u64,
    straight_prob: f64,
}

impl<P: Problem> Mind<P> {
    /// A mind for a problem.
    pub fn new(problem: P, cfg: MindConfig) -> Mind<P> {
        let mut rng = Rng::seed_from_u64(cfg.seed);
        let mut world_cfg = problem.embedding();
        world_cfg.pheromones = Some(cfg.pheromones.clone());
        world_cfg.tick_s = cfg.tick_s;
        world_cfg.seed = Some(cfg.seed);
        let world = World::new(world_cfg, &mut rng);

        // mind → castes (their standing dials) → moods (the queen's
        // expression composes with them) → thoughts.
        let spec = HierarchySpec {
            root_name: "mind".to_string(),
            levels: vec![
                LevelSpec::new("caste", 2),
                LevelSpec::new("mood", 1),
                LevelSpec::new("thought", cfg.leaves_per_caste.max(1)),
            ],
        };
        let mut hierarchy = Hierarchy::from_spec(&spec, cfg.instinct.clone());
        let castes = hierarchy.nodes_at_depth(1);
        let moods = hierarchy.nodes_at_depth(2);
        hierarchy.node_mut(castes[0]).name = "scouts".to_string();
        hierarchy.node_mut(castes[0]).surface.entropy = EntropyControl::relative(cfg.scout_heat);
        hierarchy.node_mut(castes[1]).name = "followers".to_string();
        hierarchy.node_mut(castes[1]).surface.entropy = EntropyControl::relative(cfg.follower_cool);
        let policies = hierarchy.compile();

        let origin = problem.origin();
        let origin_place = problem.place(&origin);
        let free = match problem.moves(&origin) {
            Moves::Free { step } => Some(step.max(1e-3)),
            Moves::States(_) => None,
        };
        let n = cfg.thoughts.max(1);
        let scouts = ((n as f64) * cfg.scouts.clamp(0.0, 1.0)).round() as usize;
        let mut thoughts = Vec::with_capacity(n);
        for i in 0..n {
            let caste = if i < scouts { 0 } else { 1 };
            let leaves = &hierarchy.node(moods[caste]).children;
            let leaf_node = leaves[i % leaves.len()];
            let leaf = hierarchy.leaf_index(leaf_node).expect("a leaf");
            let mut t = Thought::new(i, leaf, caste, origin.clone(), origin_place);
            t.rest_until = (i % 8) as u64;
            thoughts.push(t);
        }

        let (w, h, tick_s) = (world.width(), world.height(), cfg.tick_s);
        let history_cfg = match (&cfg.history, &cfg.queen) {
            (Some(h), _) => Some(h.clone()),
            (None, Some(_)) => Some(HistoryConfig::default()),
            (None, None) => None,
        };
        let history = history_cfg.map(|c| MovementHistory::new(w, h, tick_s, c));
        let memo = cfg
            .memo
            .clone()
            .map(|m| Memo::new(w, h, tick_s, m).with_transits(&world));
        let queen = cfg.queen.as_ref().map(|q| {
            Queen::new(
                QueenConfig {
                    epoch_ticks: q.epoch_ticks,
                    warmup_ticks: q.warmup_ticks,
                    targets: moods.clone(),
                    expression: q.expression,
                    script: Vec::new(),
                    random_input: false,
                    recursion: q.recursion.clone(),
                    component: Component::Residual,
                    ridge: 3.0,
                },
                &hierarchy,
            )
        });

        let net = cfg.symbols.clone().map(|c| {
            let mut net = PathNet::new(w, h, tick_s, problem.punctures(), c);
            for (letter, name) in problem.letter_names() {
                net.set_name(letter, &name);
            }
            net
        });

        Mind {
            problem,
            cfg,
            world,
            hierarchy,
            policies,
            castes,
            thoughts,
            origin,
            origin_place,
            free,
            history,
            memo,
            queen,
            net,
            rng,
            tick: 0,
            stats: MindStats::default(),
            best: None,
            last_improvement_epoch: 0,
            epoch_quality_mark: 0.0,
            peak_yield: 0.0,
            epochs_since_peak: 0,
            straight_prob: 0.0,
        }
    }

    // ---------------------------------------------------------------
    // Access

    /// The problem.
    pub fn problem(&self) -> &P {
        &self.problem
    }

    /// The configuration.
    pub fn config(&self) -> &MindConfig {
        &self.cfg
    }

    /// The medium.
    pub fn world(&self) -> &World {
        &self.world
    }

    /// The hierarchy of surfaces.
    pub fn hierarchy(&self) -> &Hierarchy {
        &self.hierarchy
    }

    /// The hierarchy, to change (call [`recompile`](Self::recompile)
    /// after).
    pub fn hierarchy_mut(&mut self) -> &mut Hierarchy {
        &mut self.hierarchy
    }

    /// Replace the hierarchy (it must have the shape a mind builds:
    /// two castes of leaves).
    pub fn set_hierarchy(&mut self, hierarchy: Hierarchy) {
        self.hierarchy = hierarchy;
        self.recompile();
    }

    /// Recompute the effective policies after the hierarchy changed.
    pub fn recompile(&mut self) {
        self.policies = self.hierarchy.compile();
    }

    /// The caste nodes: scouts, then followers.
    pub fn castes(&self) -> &[NodeId] {
        &self.castes
    }

    /// The mood nodes the queen expresses herself through, one under
    /// each caste.
    pub fn moods(&self) -> Vec<NodeId> {
        self.hierarchy.nodes_at_depth(2)
    }

    /// The thoughts.
    pub fn thoughts(&self) -> &[Thought<P::State>] {
        &self.thoughts
    }

    /// The counters.
    pub fn stats(&self) -> &MindStats {
        &self.stats
    }

    /// The best solution brought home so far.
    pub fn best(&self) -> Option<&Finding<P::State>> {
        self.best.as_ref()
    }

    /// The tick.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// The movement history, if kept.
    pub fn history(&self) -> Option<&MovementHistory> {
        self.history.as_ref()
    }

    /// The memo, if kept.
    pub fn memo(&self) -> Option<&Memo> {
        self.memo.as_ref()
    }

    /// The queen, if there is one.
    pub fn queen(&self) -> Option<&Queen> {
        self.queen.as_ref()
    }

    /// The path-topological network, if the mind keeps one.
    pub fn net(&self) -> Option<&PathNet> {
        self.net.as_ref()
    }

    /// The network, to name, attend to or express its symbols.
    pub fn net_mut(&mut self) -> Option<&mut PathNet> {
        self.net.as_mut()
    }

    /// What class a route of places is, by the network (its moves no
    /// letters, its word without a prefix).
    pub fn label(&self, route: &[Point]) -> Option<Label> {
        self.net
            .as_ref()
            .map(|n| n.label(&Route::of(route.to_vec()), &[]))
    }

    /// What class a route with the letters of its moves and a prefix
    /// is, by the network.
    pub fn label_route(&self, route: &Route, prefix: &[Letter]) -> Option<Label> {
        self.net.as_ref().map(|n| n.label(route, prefix))
    }

    /// A route of a class from the origin to a cell, by the network's
    /// search over the medium's passable ground, through its portals
    /// (a portal crossed is a letter of the move).
    pub fn generate(&self, word: &Word, to: Position) -> Option<Vec<Point>> {
        let net = self.net.as_ref()?;
        let world = &self.world;
        let neighbours = |cell: Position| -> Vec<(Position, Letter)> {
            let mut out = Vec::with_capacity(8);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let next = Position::new(cell.x + dx, cell.y + dy);
                    if world.is_passable(next) {
                        if dx != 0
                            && dy != 0
                            && !(world.is_passable(Position::new(cell.x + dx, cell.y))
                                && world.is_passable(Position::new(cell.x, cell.y + dy)))
                        {
                            continue;
                        }
                        out.push((next, 0));
                    } else if dx == 0 || dy == 0 {
                        if let Some(w) = world.warp(Point::center_of(next)) {
                            let landing = w.point.cell();
                            if world.is_passable(landing) {
                                out.push((landing, move_letter(w.portal, w.forward)));
                            }
                        }
                    }
                }
            }
            out
        };
        if !world.is_passable(to) {
            return None;
        }
        net.generate_with(word, self.origin_place.cell(), to, &neighbours)
    }

    /// A route of a class over the problem's states from a state, by
    /// search over states and the words of the ways to them, ending at
    /// a state `done` accepts; none within `max_moves` moves or the
    /// longest word. The word begins with the state's prefix.
    pub fn generate_states(
        &self,
        word: &Word,
        from: P::State,
        done: &dyn Fn(&P::State) -> bool,
        max_moves: usize,
    ) -> Option<Vec<P::State>>
    where
        P::State: Hash + Eq,
    {
        use std::collections::{HashMap, HashSet, VecDeque};
        let net = self.net.as_ref()?;
        let max_word = net.config().max_word.max(word.len());
        let start_word = Word::from_letters(&self.problem.prefix(&from));
        let start = (from, start_word);
        let mut parent: HashMap<(P::State, Word), (P::State, Word)> = HashMap::new();
        let mut depth: HashMap<(P::State, Word), usize> = HashMap::new();
        let mut seen: HashSet<(P::State, Word)> = HashSet::new();
        let mut queue = VecDeque::new();
        seen.insert(start.clone());
        depth.insert(start.clone(), 0);
        queue.push_back(start);
        while let Some(state) = queue.pop_front() {
            if state.1 == *word && done(&state.0) {
                let mut path = vec![state.0.clone()];
                let mut cur = state;
                while let Some(p) = parent.get(&cur) {
                    path.push(p.0.clone());
                    cur = p.clone();
                }
                path.reverse();
                return Some(path);
            }
            let d = depth.get(&state).copied().unwrap_or(0);
            if d >= max_moves {
                continue;
            }
            let (here, w) = state;
            let Moves::States(successors) = self.problem.moves(&here) else {
                continue;
            };
            for next in successors {
                let mut nw = w.clone();
                net.rays().step(
                    self.problem.place(&here),
                    self.problem.place(&next),
                    &mut nw,
                );
                if let Some(l) = self.problem.letter(&here, &next) {
                    nw.push(l);
                }
                if nw.len() > max_word {
                    continue;
                }
                let key = (next, nw);
                if seen.insert(key.clone()) {
                    parent.insert(key.clone(), (here.clone(), w.clone()));
                    depth.insert(key.clone(), d + 1);
                    queue.push_back(key);
                }
            }
        }
        None
    }

    /// The route of a trip as the network sees it: its places with the
    /// letters of its moves.
    fn route_of(&self, i: usize) -> Route {
        let t = &self.thoughts[i];
        Route::with_letters(t.places.clone(), t.letters.clone())
    }

    /// Express a symbol: its glyph is laid into its channel and the
    /// thoughts attend to it with the given gain until released, so
    /// that they walk its class again.
    pub fn express(&mut self, k: usize, gain: f64) {
        let deposit = self
            .cfg
            .symbols
            .as_ref()
            .map(|c| c.deposit * c.expression)
            .unwrap_or(0.0);
        if let Some(net) = &mut self.net {
            net.express(k, deposit);
            net.attend(k, gain);
        }
    }

    /// Let a symbol go.
    pub fn release(&mut self, k: usize) {
        if let Some(net) = &mut self.net {
            net.release(k);
        }
    }

    /// The origin's place.
    pub fn origin_place(&self) -> Point {
        self.origin_place
    }

    /// Whether the problem is walked freely (rather than state to state).
    pub fn is_free(&self) -> bool {
        self.free.is_some()
    }

    /// The effective policy of a thought.
    pub fn policy_of(&self, thought: usize) -> &EffectivePolicy {
        &self.policies[self.thoughts[thought].leaf]
    }

    // ---------------------------------------------------------------
    // Time

    /// Run for `ticks` ticks.
    pub fn run(&mut self, ticks: u64) {
        for _ in 0..ticks {
            self.step();
        }
    }

    /// One tick: the frame is planned, every thought thinks, the medium
    /// moves, the history and the memo forget, the queen may think.
    pub fn step(&mut self) {
        self.plan_frame();
        for i in 0..self.thoughts.len() {
            self.think(i);
        }
        self.world.step_pheromones();
        if let Some(h) = &mut self.history {
            h.step();
        }
        if let Some(m) = &mut self.memo {
            m.step();
        }
        self.queen_epoch();
        if let Some(net) = &mut self.net {
            net.step();
            let epoch = net.config().epoch_ticks.max(1);
            if self.tick > 0 && self.tick.is_multiple_of(epoch) {
                if let Some(h) = &self.history {
                    let walked = h.flow_mask(net.config().hole_rate, 0.0);
                    if net.learn_punctures(&walked) > 0 {
                        // The alphabet grew: the classes of trips under
                        // way are named anew when they come home.
                        for t in self.thoughts.iter_mut() {
                            t.symbol = None;
                        }
                    }
                }
            }
        }
        self.tick += 1;
        self.stats.ticks += 1;
    }

    fn think(&mut self, i: usize) {
        if let Some(t) = self.thoughts[i].transit {
            if self.tick >= t.until {
                self.complete_transit(i);
            }
            if self.thoughts[i].transit.is_some() {
                return;
            }
        }
        if self.thoughts[i].activity == Activity::Resting {
            if self.tick >= self.thoughts[i].rest_until {
                self.depart(i);
            } else {
                return;
            }
        }
        if self.thoughts[i].target.is_none() && !self.arrive(i) {
            return;
        }
        self.walk(i);
    }

    // ---------------------------------------------------------------
    // The trip

    fn depart(&mut self, i: usize) {
        let origin = self.problem.depart(self.stats.trips);
        let place = self.problem.place(&origin);
        let prefix = self.problem.prefix(&origin);
        let x = self.problem.departure(&origin);
        let random = self.rng.range(0.0, std::f64::consts::TAU);
        let site_fidelity = self.cfg.site_fidelity;
        let free = self.free.is_some();
        let foresight = self.cfg.foresight;
        // A thought sets out any way at all, unless a symbol is being
        // expressed: then it sets out with the symbol in mind, the way
        // its glyph goes (the sucker that selects its moves climbs from
        // straight ahead, so where it starts is where it looks).
        let attended = self.net.as_ref().and_then(|n| n.attended());
        let t = &mut self.thoughts[i];
        t.reset_trip(origin, place);
        t.prefix = prefix;
        t.x = x;
        t.heading = match attended {
            Some((_, heading)) => heading,
            None => random,
        };
        t.activity = if site_fidelity && t.site.is_some() {
            Activity::Outbound
        } else {
            Activity::Searching
        };
        t.hold_activity = t.activity;
        let site = t.site;
        self.occupy(place.cell(), 1);
        self.stats.trips += 1;
        if let (true, Some(radius), Some(site), Activity::Outbound) =
            (free, foresight, site, self.thoughts[i].activity)
        {
            self.plan(i, site.place, radius);
        }
    }

    /// A thought stands in a state: what next. Returns whether a move
    /// was begun.
    fn arrive(&mut self, i: usize) -> bool {
        let activity = self.thoughts[i].activity;
        if activity == Activity::Inbound {
            let home = match self.free {
                Some(_) => self.world.is_nest(self.thoughts[i].cell()),
                None => {
                    let t = &self.thoughts[i];
                    t.route.first().map(|o| t.state == *o).unwrap_or(false)
                }
            };
            if home {
                self.deliver(i);
                return false;
            }
            if self.free.is_none() {
                let t = &mut self.thoughts[i];
                if t.back == 0 {
                    self.deliver(i);
                    return false;
                }
                t.back -= 1;
                let state = t.route[t.back].clone();
                let point = t.places[t.back];
                let (dx, dy) = t.place.to(point);
                if dx * dx + dy * dy > 1e-18 {
                    t.heading = angle_of(dx, dy);
                }
                t.target = Some(Target { state, point });
                return true;
            }
            if self.thoughts[i].steps >= 2 * self.cfg.trip_budget.max(1) {
                self.lose(i);
                return false;
            }
            return self.choose_free(i, Mode::Inbound);
        }
        let state = self.thoughts[i].state.clone();
        let at_origin = self.thoughts[i]
            .route
            .first()
            .map(|o| state == *o)
            .unwrap_or(true);
        if !at_origin {
            if let Some(q) = self.problem.quality_at(&state, &self.thoughts[i].x) {
                self.found(i, q);
                return self.arrive(i);
            }
        }
        if self.thoughts[i].steps >= self.cfg.trip_budget.max(1) {
            self.give_up(i, false);
            return self.arrive(i);
        }
        match self.problem.moves(&state) {
            Moves::States(successors) if successors.is_empty() => {
                self.give_up(i, true);
                if self.thoughts[i].retreat {
                    return true;
                }
                self.arrive(i)
            }
            Moves::States(mut successors) => {
                if let Some(avoid) = self.thoughts[i].avoid.take() {
                    successors.retain(|s| *s != avoid);
                    if successors.is_empty() {
                        self.give_up(i, true);
                        if self.thoughts[i].retreat {
                            return true;
                        }
                        return self.arrive(i);
                    }
                }
                self.choose_states(i, successors)
            }
            Moves::Free { .. } => self.choose_free(i, Mode::Outbound),
        }
    }

    fn found(&mut self, i: usize, quality: f64) {
        self.stats.findings += 1;
        let (cell, place) = {
            let t = &self.thoughts[i];
            (t.cell(), t.place)
        };
        let release = self.cfg.odour_release * quality;
        if release > 0.0 {
            self.world.deposit(cell, Pheromone::Odour, release);
            if let Some(m) = &mut self.memo {
                m.record_deposit(cell, Pheromone::Odour, release);
            }
        }
        if let Some(m) = &mut self.memo {
            m.record_stop(cell, Stop::Food);
        }
        self.close_records(i, INSIDE, 0.0);
        let free = self.free.is_some();
        let foresight = self.cfg.foresight;
        let origin_place = self.origin_place;
        let recruitment = self.cfg.recruitment.max(0.0);
        let colony_best = self.best.as_ref().map(|b| b.quality).unwrap_or(0.0);
        let route = self.route_of(i);
        let symbol = self
            .net
            .as_ref()
            .and_then(|n| n.find(&n.word(&route, &self.thoughts[i].prefix)));
        let t = &mut self.thoughts[i];
        t.symbol = symbol;
        t.load = Some(quality);
        // Recruitment is judged against the best the thought knows and
        // the best the mind has brought home.
        let known = t
            .site
            .map(|s| s.quality)
            .unwrap_or(0.0)
            .max(colony_best)
            .max(quality);
        t.recruit = if known > 0.0 {
            (quality / known).clamp(0.0, 1.0).powf(recruitment)
        } else {
            1.0
        };
        let better = t.site.map(|s| quality >= s.quality).unwrap_or(true);
        if better {
            t.site = Some(Site {
                place,
                quality,
                failures: 0,
            });
        }
        t.activity = Activity::Inbound;
        t.hold_activity = Activity::Inbound;
        t.hold_until = 0;
        t.deadline = 0;
        t.back = t.route.len().saturating_sub(1);
        t.heading = wrap_angle(t.heading + std::f64::consts::PI);
        if let (true, Some(radius)) = (free, foresight) {
            self.plan(i, origin_place, radius);
        }
    }

    /// A dead end or an exhausted budget: mark it, and either retreat a
    /// state to try another way, or turn for home empty-handed.
    fn give_up(&mut self, i: usize, dead_end: bool) {
        let cell = self.thoughts[i].cell();
        {
            let route = self.route_of(i);
            if let Some(net) = &mut self.net {
                net.observe_miss(&route, &self.thoughts[i].prefix);
            }
            let t = &self.thoughts[i];
            if let Some(origin) = t.route.first() {
                self.problem.learn(origin, &t.route, &t.x, 0.0);
            }
        }
        if dead_end {
            self.stats.dead_ends += 1;
            let amount = self.cfg.no_entry_deposit;
            if amount > 0.0 {
                self.world.deposit(cell, Pheromone::NoEntry, amount);
                if let Some(m) = &mut self.memo {
                    m.record_deposit(cell, Pheromone::NoEntry, amount);
                }
            }
            self.thoughts[i].no_entry_left = self.cfg.no_entry_reach;
            let can_retreat = self.free.is_none()
                && self.thoughts[i].retreats < self.cfg.retreats
                && self.thoughts[i].route.len() >= 2;
            if can_retreat {
                self.stats.retreats += 1;
                let t = &mut self.thoughts[i];
                let n = t.route.len();
                let state = t.route[n - 2].clone();
                let point = t.places[n - 2];
                t.avoid = Some(t.state.clone());
                t.retreats += 1;
                t.retreat = true;
                t.no_entry_left = t.place.distance(point).max(1.0);
                let (dx, dy) = t.place.to(point);
                if dx * dx + dy * dy > 1e-18 {
                    t.heading = angle_of(dx, dy);
                }
                t.target = Some(Target { state, point });
                return;
            }
        } else {
            self.stats.given_up += 1;
        }
        if let Some(m) = &mut self.memo {
            m.record_stop(cell, Stop::GaveUp);
        }
        self.close_records(i, INSIDE, 0.0);
        let free = self.free.is_some();
        let foresight = self.cfg.foresight;
        let origin_place = self.origin_place;
        let t = &mut self.thoughts[i];
        t.load = None;
        t.activity = Activity::Inbound;
        t.hold_activity = Activity::Inbound;
        t.hold_until = 0;
        t.deadline = 0;
        t.back = t.route.len().saturating_sub(1);
        t.heading = wrap_angle(t.heading + std::f64::consts::PI);
        if let (true, Some(radius)) = (free, foresight) {
            self.plan(i, origin_place, radius);
        }
    }

    fn deliver(&mut self, i: usize) {
        let cell = self.thoughts[i].cell();
        self.occupy(cell, -1);
        if let Some(m) = &mut self.memo {
            m.record_stop(cell, Stop::Nest);
        }
        self.close_records(i, INSIDE, 0.0);
        let tick = self.tick;
        let patience = self.cfg.site_patience;
        let rest = self.cfg.rest_ticks;
        let t = &mut self.thoughts[i];
        if let Some(q) = t.load.take() {
            self.stats.deliveries += 1;
            self.stats.quality_sum += q;
            t.findings += 1;
            let route = Route::with_letters(t.places.clone(), t.letters.clone());
            if let Some(net) = &mut self.net {
                // What the trip integrated: where it arrived less where
                // it set out.
                let departure = t
                    .route
                    .first()
                    .map(|o| self.problem.departure(o))
                    .unwrap_or_default();
                let transported: Vec<f64> =
                    t.x.iter()
                        .enumerate()
                        .map(|(d, v)| v - departure.get(d).copied().unwrap_or(0.0))
                        .collect();
                net.observe(&route, &t.prefix, &transported, q, tick);
            }
            if let Some(origin) = t.route.first() {
                self.problem.learn(origin, &t.route, &t.x, q);
            }
            // The route is remembered when it led to the best the
            // thought knows (its site was set on finding it).
            if t.site.map(|s| q >= s.quality).unwrap_or(true) {
                t.remembered = t.places.clone();
            }
            let better = self.best.as_ref().map(|b| q > b.quality).unwrap_or(true);
            if better {
                self.best = Some(Finding {
                    quality: q,
                    route: t.route.clone(),
                    places: t.places.clone(),
                    tick,
                    thought: i,
                    letters: t.letters.clone(),
                    prefix: t.prefix.clone(),
                    x: t.x.clone(),
                });
                self.stats.improvements += 1;
                self.last_improvement_epoch = self.stats.epochs;
            }
        } else if let Some(site) = t.site.as_mut() {
            site.failures += 1;
            if site.failures >= patience.max(1) {
                t.site = None;
            }
        }
        t.activity = Activity::Resting;
        t.hold_activity = Activity::Resting;
        t.target = None;
        t.rest_until = tick + rest;
        t.trips += 1;
    }

    /// A thought that cannot find its way home is recycled at the origin.
    fn lose(&mut self, i: usize) {
        self.stats.lost += 1;
        let cell = self.thoughts[i].cell();
        self.occupy(cell, -1);
        self.close_records(i, INSIDE, 0.0);
        let tick = self.tick;
        let rest = self.cfg.rest_ticks;
        let origin = self.thoughts[i]
            .route
            .first()
            .cloned()
            .unwrap_or_else(|| self.origin.clone());
        let place = self.problem.place(&origin);
        let t = &mut self.thoughts[i];
        t.reset_trip(origin, place);
        if let Some(site) = t.site.as_mut() {
            site.failures += 1;
        }
        t.activity = Activity::Resting;
        t.hold_activity = Activity::Resting;
        t.rest_until = tick + rest;
        t.trips += 1;
    }

    fn occupy(&mut self, cell: Position, delta: i32) {
        if let Some(c) = self.world.cell_mut(cell) {
            if delta > 0 {
                c.occupancy = c.occupancy.saturating_add(delta as u16);
            } else {
                c.occupancy = c.occupancy.saturating_sub((-delta) as u16);
            }
        }
    }

    /// Foresee the route from where a thought stands to a place.
    fn plan(&mut self, i: usize, to: Point, radius: f64) {
        let from = self.thoughts[i].place;
        let plan = geodesic(&self.world, from, to, radius)
            .map(|g| g.points)
            .unwrap_or_default();
        let t = &mut self.thoughts[i];
        t.plan = plan;
        t.plan_index = 0;
        t.planned_at = self.tick;
    }

    // ---------------------------------------------------------------
    // Choosing

    fn choose_states(&mut self, i: usize, successors: Vec<P::State>) -> bool {
        let (place, heading) = {
            let t = &self.thoughts[i];
            (t.place, t.heading)
        };
        let mut cands: Vec<Candidate<P::State>> = Vec::with_capacity(successors.len());
        for state in successors {
            let point = self.problem.place(&state);
            let (dx, dy) = place.to(point);
            let len = (dx * dx + dy * dy).sqrt();
            let ring = if len < 1e-9 {
                0
            } else {
                ring_index_of(angle_of(dx, dy), heading)
            };
            let valid = len < 1e-9 || self.world.segment_passable(place, point);
            cands.push(Candidate {
                state,
                point,
                turn: 0.0,
                len,
                ring,
                valid,
            });
        }
        self.choose(i, Mode::Outbound, cands)
    }

    fn choose_free(&mut self, i: usize, mode: Mode) -> bool {
        let step = self.free.unwrap_or(1.0);
        let (place, heading) = {
            let t = &self.thoughts[i];
            (t.place, t.heading)
        };
        let mut cands: Vec<Candidate<P::State>> = Vec::with_capacity(RING);
        for k in 0..RING {
            let h = ring_heading(k, heading);
            let raw = place.advanced(h, step);
            let (landing, turn) = if self.world.is_passable_point(raw) {
                if !self.world.segment_passable(place, raw) {
                    continue;
                }
                (raw, 0.0)
            } else if let Some(w) = self.world.warp(raw) {
                (w.point, w.turn)
            } else {
                continue;
            };
            let Some(state) = self.problem.locate(landing) else {
                continue;
            };
            cands.push(Candidate {
                state,
                point: raw,
                turn,
                len: step,
                ring: k,
                valid: true,
            });
        }
        self.choose(i, mode, cands)
    }

    /// Choose among candidates under the pipeline's gate. Returns
    /// whether a move was begun.
    fn choose(&mut self, i: usize, mode: Mode, cands: Vec<Candidate<P::State>>) -> bool {
        if !cands.iter().any(|c| c.valid) {
            return false;
        }
        match self.gate(i) {
            Gate::Defer => {
                self.stats.deferred += 1;
                return false;
            }
            Gate::Hold => {
                // Keep going straight: the candidate nearest straight
                // ahead, within a ring position of it.
                let straight = cands
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.valid && turn_magnitude(c.ring) <= 1)
                    .min_by_key(|(_, c)| turn_magnitude(c.ring));
                if let Some((j, _)) = straight {
                    self.stats.held += 1;
                    self.stats.frames.held += 1;
                    self.begin(i, &cands[j]);
                    return true;
                }
            }
            Gate::Decide => {}
        }
        let senses = self.sense(i, mode, &cands);
        let Some(j) = self.decide(i, mode, &senses) else {
            return false;
        };
        self.begin(i, &cands[j]);
        self.set_horizon(i);
        true
    }

    fn begin(&mut self, i: usize, cand: &Candidate<P::State>) {
        let t = &mut self.thoughts[i];
        let (dx, dy) = t.place.to(cand.point);
        if dx * dx + dy * dy > 1e-18 {
            t.heading = angle_of(dx, dy);
        }
        t.last_turn = turn_magnitude(cand.ring) as f64 * RING_STEP;
        t.target = Some(Target {
            state: cand.state.clone(),
            point: cand.point,
        });
    }

    /// What a thought perceives of its candidates.
    fn sense(&mut self, i: usize, mode: Mode, cands: &[Candidate<P::State>]) -> Senses {
        let offset = mode.offset();
        let leaf = self.thoughts[i].leaf;
        let weights = self.policies[leaf].weights;
        let route_dir = {
            let t = &mut self.thoughts[i];
            if t.plan.is_empty() {
                t.route_direction()
            } else {
                t.plan_direction()
            }
        };
        let t = &self.thoughts[i];
        let (place, heading, state) = (t.place, t.heading, t.state.clone());
        let home = t.route.first().cloned();
        let x = t.x.clone();
        let body = Body {
            place,
            home_dir: t.home_direction(),
            site_dir: if t.activity.is_out() && self.free.is_some() {
                t.site_direction()
            } else {
                None
            },
            route_dir,
            recent: t.recent(),
            sensitivity: 1.0,
        };
        let free = self.free.is_some();
        let mut senses = Senses::default();
        let mut best = [f64::NEG_INFINITY; RING];
        for (j, c) in cands.iter().enumerate() {
            if !c.valid {
                continue;
            }
            let u = unit(place.to(c.point)).unwrap_or_else(|| {
                let h = ring_heading(c.ring, heading);
                (h.cos(), h.sin())
            });
            let is_home = home.as_ref().map(|h| c.state == *h).unwrap_or(false);
            let food = !is_home && {
                if x.is_empty() || free {
                    self.problem.quality_at(&c.state, &x).is_some()
                } else {
                    let mut ahead = x.clone();
                    self.problem.integrate(&state, &c.state, &mut ahead);
                    self.problem.quality_at(&c.state, &ahead).is_some()
                }
            };
            let nest = if free {
                self.world.is_nest(c.point.cell())
            } else {
                is_home
            };
            let sight = Sight {
                food,
                nest,
                scent: if free {
                    0.0
                } else {
                    self.problem.scent_at(&state, &c.state, &x)
                },
                clear: if free {
                    clear_ahead(&self.world, place, u)
                } else {
                    2.0
                },
            };
            let mut feat = [0.0; FEATURES];
            features(
                &self.world,
                &body,
                u,
                c.len,
                c.ring,
                sight,
                offset,
                &mut feat,
            );
            let extra = self
                .net
                .as_ref()
                .map(|n| n.extra(place, u, c.len))
                .unwrap_or(0.0);
            let logit = dot(&weights, &feat) + extra;
            if !senses.valid[c.ring] || logit > best[c.ring] {
                senses.valid[c.ring] = true;
                senses.slot[c.ring] = j;
                senses.features[c.ring] = feat;
                senses.extra[c.ring] = extra;
                best[c.ring] = logit;
            }
        }
        senses
    }

    /// The decision: the ring of scores deformed with the entropy budget,
    /// tempered and selected. Returns the chosen candidate.
    fn decide(&mut self, i: usize, mode: Mode, senses: &Senses) -> Option<usize> {
        let (leaf, cell, laden, searching) = {
            let t = &self.thoughts[i];
            (
                t.leaf,
                t.cell(),
                t.laden(),
                t.activity == Activity::Searching,
            )
        };
        let policy = self.policies[leaf].clone();
        let mut scores = [f64::NEG_INFINITY; RING];
        let mut any = false;
        for (((score, &valid), features), &extra) in scores
            .iter_mut()
            .zip(&senses.valid)
            .zip(&senses.features)
            .zip(&senses.extra)
        {
            if valid {
                *score = dot(&policy.weights, features) + extra;
                any = true;
            }
        }
        if !any {
            return None;
        }
        let base = Landscape::new(scores, senses.valid);
        let tempering = if searching {
            policy.tempering.heated(self.cfg.search_heat)
        } else {
            policy.tempering
        };
        let deforms =
            policy.deformation.smooth_share() > 0.0 || policy.deformation.rough_share() > 0.0;
        let h = match tempering {
            Tempering::Entropy(f) => f,
            Tempering::Temperature(t) if deforms => base.entropy_fraction_at(t),
            Tempering::Temperature(_) => 0.0,
        };
        let smooth_scale = policy.deformation.smooth_share() * h * self.cfg.geometry.smooth_max;
        let smoothed = base.smoothed(smooth_scale);
        let rough_amplitude = policy.deformation.rough_share() * h * base.range().max(1.0);
        let deformed = smoothed.roughened(
            rough_amplitude,
            self.cfg.geometry.rough_modes,
            &mut self.rng,
        );
        let tempered = deformed.temper_with(tempering);
        let start = base.nearest_valid(0).expect("a valid ring position");
        let keep = self.cfg.ledger;
        let (chosen, selected) = match self.cfg.choice.reach(self.free.is_some()) {
            None => (self.rng.choose_weighted(&tempered.probs), tempered.probs),
            Some(reach) => {
                let reach = (reach as f64 * policy.deformation.reach_scale())
                    .round()
                    .clamp(0.0, self.cfg.geometry.max_reach as f64)
                    as usize;
                let sucker = Sucker { reach };
                let (j, _) = sucker.walk(&tempered.scaled, start, &mut self.rng);
                let dist = if keep {
                    sucker.distribution(&tempered.scaled, start)
                } else {
                    tempered.probs
                };
                (j, dist)
            }
        };
        let cone = self
            .cfg
            .pipeline
            .as_ref()
            .map(|p| p.cone)
            .unwrap_or(0)
            .min(RING / 4);
        let mut p = selected[0];
        for r in 1..=cone {
            p += selected[r] + selected[RING - r];
        }
        self.straight_prob = p;
        self.stats.decisions += 1;
        self.stats.entropy_sum += tempered.entropy;
        let selected_entropy = if keep {
            entropy(&selected)
        } else {
            tempered.entropy
        };
        self.stats.selected_entropy_sum += selected_entropy;
        if keep {
            let t = tempered.temperature;
            self.stats.ledger.record(
                base.entropy_at(t),
                smoothed.entropy_at(t),
                tempered.entropy,
                tempered.entropy,
                selected_entropy,
            );
        }
        let leg = if searching {
            Leg::Searching
        } else if matches!(mode, Mode::Inbound) {
            Leg::Inbound
        } else {
            Leg::Outbound
        };
        if let Some(m) = &mut self.memo {
            m.record_decision(cell, chosen, tempered.entropy, leg, laden);
        }
        for r in self.thoughts[i].records.iter_mut().flatten() {
            r.decisions = r.decisions.saturating_add(1);
            r.entropy += tempered.entropy as f32;
            r.straight += (turn_magnitude(chosen) as f64 * RING_STEP).cos() as f32;
        }
        Some(senses.slot[chosen])
    }

    // ---------------------------------------------------------------
    // Walking

    fn walk(&mut self, i: usize) {
        let Some(target) = self.thoughts[i].target.clone() else {
            return;
        };
        let (place, heading, cell) = {
            let t = &self.thoughts[i];
            (t.place, t.heading, t.cell())
        };
        let (dx, dy) = place.to(target.point);
        let dist = (dx * dx + dy * dy).sqrt();
        let climb = self.world.climb_factor(cell, heading);
        let step = (self.cfg.speed * climb).max(1e-6);
        let (pre, reached) = if dist <= step + 1e-9 {
            (target.point, true)
        } else {
            let u = (dx / dist, dy / dist);
            (
                Point::new(place.x + u.0 * step, place.y + u.1 * step),
                false,
            )
        };
        let walked = place.distance(pre);
        let free_walk = self.free.is_some();
        let (mut next, mut turn) = (pre, 0.0);
        let mut portal: Letter = 0;
        if !self.world.is_passable_point(pre) {
            match self.world.warp(pre) {
                Some(w) => {
                    next = w.point;
                    turn = w.turn;
                    portal = move_letter(w.portal, w.forward);
                }
                None => {
                    // Blocked: the move is given up and decided again.
                    self.thoughts[i].target = None;
                    return;
                }
            }
        }
        let from_cell = cell;
        let to_cell = next.cell();
        if from_cell != to_cell {
            self.occupy(from_cell, -1);
            self.occupy(to_cell, 1);
        }
        let last_turn = std::mem::replace(&mut self.thoughts[i].last_turn, 0.0);
        if let Some(h) = &mut self.history {
            h.record(place, pre, last_turn);
        }
        if let Some(m) = &mut self.memo {
            m.record_move(to_cell, walked);
        }
        // What is laid along the way.
        let (activity, load, no_entry_left, recruit, symbol) = {
            let t = &self.thoughts[i];
            (t.activity, t.load, t.no_entry_left, t.recruit, t.symbol)
        };
        if let Some(q) = load {
            let per_cell = self.cfg.trail_deposit * q * recruit;
            self.lay_along(i, place, pre, Pheromone::Trail, per_cell);
            if let (Some(net), Some(k)) = (&mut self.net, symbol) {
                let per_cell = net.config().deposit * q * recruit;
                net.lay(k, place, pre, per_cell);
            }
        }
        if no_entry_left > 0.0 && self.cfg.no_entry_deposit > 0.0 {
            let per_cell = 0.25 * self.cfg.no_entry_deposit;
            self.lay_along(i, place, pre, Pheromone::NoEntry, per_cell);
            self.thoughts[i].no_entry_left = (no_entry_left - walked).max(0.0);
        }
        if self.cfg.home_deposit > 0.0 && activity.is_out() {
            let per_cell = self.cfg.home_deposit;
            self.lay_along(i, place, pre, Pheromone::Home, per_cell);
        }
        for r in self.thoughts[i].records.iter_mut().flatten() {
            r.walked(walked, next);
        }
        {
            let t = &mut self.thoughts[i];
            t.home_vector.0 += pre.x - place.x;
            t.home_vector.1 += pre.y - place.y;
            if free_walk && t.activity.is_out() {
                t.develop(pre.x - place.x, pre.y - place.y);
            }
            t.rotate_frame(turn);
            t.place = next;
            t.trip_length += walked;
            if turn != 0.0 {
                t.heading = wrap_angle(t.heading + turn);
            }
        }
        self.stats.moves += 1;
        self.stats.length += walked;
        if self.free.is_some() && from_cell != to_cell && self.cross(i, from_cell, to_cell, next) {
            self.thoughts[i].target = None;
            return;
        }
        if reached || turn != 0.0 {
            let state = if turn != 0.0 {
                self.problem
                    .locate(next)
                    .unwrap_or_else(|| target.state.clone())
            } else {
                target.state.clone()
            };
            let free = self.free.is_some();
            let from_state = self.thoughts[i].state.clone();
            let t = &mut self.thoughts[i];
            t.state = state;
            t.target = None;
            t.steps += 1;
            t.remember(to_cell);
            if t.retreat {
                // Back at the state before the dead end: the way it
                // went is dropped from the route, and what it
                // integrated is undone by integrating the rest again.
                t.retreat = false;
                t.route.pop();
                t.places.pop();
                t.letters.pop();
                t.no_entry_left = 0.0;
                if !free && !t.x.is_empty() {
                    let route = t.route.clone();
                    let mut x = self.problem.departure(&route[0]);
                    for w in route.windows(2) {
                        self.problem.integrate(&w[0], &w[1], &mut x);
                    }
                    self.thoughts[i].x = x;
                }
            } else if t.activity.is_out() {
                let new_cell =
                    !free || t.places.last().map(|p| p.cell() != to_cell).unwrap_or(true);
                if new_cell && t.route.len() < 4 * self.cfg.trip_budget.max(1) {
                    let letter = if free {
                        portal
                    } else {
                        self.problem.letter(&from_state, &t.state).unwrap_or(0)
                    };
                    if !free && !t.x.is_empty() {
                        let mut x = std::mem::take(&mut t.x);
                        self.problem.integrate(&from_state, &t.state, &mut x);
                        t.x = x;
                    }
                    t.route.push(t.state.clone());
                    t.places.push(next);
                    t.letters.push(letter);
                }
            }
        }
    }

    /// Lay a channel along a step, patch by patch.
    fn lay_along(&mut self, _i: usize, from: Point, to: Point, kind: Pheromone, per_cell: f64) {
        if per_cell <= 0.0 {
            return;
        }
        let len = from.distance(to);
        let n = (len.ceil() as usize).clamp(1, 64);
        let amount = per_cell * len / n as f64;
        for s in 1..=n {
            let f = s as f64 / n as f64;
            let p = Point::new(from.x + (to.x - from.x) * f, from.y + (to.y - from.y) * f);
            let cell = p.cell();
            self.world.deposit(cell, kind, amount);
            if let Some(m) = &mut self.memo {
                m.record_deposit(cell, kind, amount);
            }
        }
    }

    // ---------------------------------------------------------------
    // The decision pipeline

    fn plan_frame(&mut self) {
        let Some(cfg) = &self.cfg.pipeline else {
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
        for (i, t) in self.thoughts.iter_mut().enumerate() {
            t.granted = false;
            if !t.activity.is_moving()
                || t.transit.is_some()
                || t.target.is_some()
                || tick < t.hold_until
            {
                continue;
            }
            if tick >= t.deadline || t.activity != t.hold_activity {
                due += 1;
            } else {
                pending.push((t.deadline, t.leaf, i));
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
            self.thoughts[i].granted = true;
        }
    }

    fn gate(&self, i: usize) -> Gate {
        let Some(cfg) = &self.cfg.pipeline else {
            return Gate::Decide;
        };
        let t = &self.thoughts[i];
        if self.tick >= t.deadline || t.activity != t.hold_activity {
            return Gate::Decide;
        }
        if self.tick < t.hold_until {
            Gate::Hold
        } else if cfg.budget == 0 || t.granted {
            Gate::Decide
        } else {
            Gate::Defer
        }
    }

    fn set_horizon(&mut self, i: usize) {
        let Some(cfg) = &self.cfg.pipeline else {
            return;
        };
        let (cell, searching, activity) = {
            let t = &self.thoughts[i];
            (t.cell(), t.activity == Activity::Searching, t.activity)
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
        let h = horizon(cfg, invariance, self.straight_prob, searching);
        let slack = (h as f64 * cfg.slack.max(0.0)).floor() as u64;
        let t = &mut self.thoughts[i];
        t.hold_until = self.tick + h as u64;
        t.deadline = t.hold_until + slack;
        t.hold_activity = activity;
        self.stats.frames.served += 1;
        self.stats.frames.horizon_sum += h as u64;
        self.stats.frames.horizons += 1;
    }

    // ---------------------------------------------------------------
    // Habits: memoized transits through plain, invariant ground

    fn transit_levels(&self) -> Vec<u8> {
        self.memo
            .as_ref()
            .and_then(|m| m.transits.as_ref())
            .map(|t| t.levels().to_vec())
            .unwrap_or_default()
    }

    fn transit_level(&self) -> Option<u8> {
        self.memo
            .as_ref()
            .and_then(|m| m.transits.as_ref())
            .map(|t| t.level())
    }

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

    fn along_side(rect: (usize, usize, usize, usize), side: u8, p: Point) -> f32 {
        let (x0, y0, x1, y1) = rect;
        let t = match side {
            0 | 2 => (p.x - x0 as f64) / (x1 - x0).max(1) as f64,
            _ => (p.y - y0 as f64) / (y1 - y0).max(1) as f64,
        };
        t.clamp(0.0, 1.0) as f32
    }

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

    fn close_slot(&mut self, i: usize, slot: usize, exit_side: u8, exit_along: f32) {
        let Some(record) = self.thoughts[i].records[slot].take() else {
            return;
        };
        let heading = self.thoughts[i].heading;
        let outcome = record.close(self.tick, exit_side, exit_along, heading);
        if let Some(t) = self.memo.as_mut().and_then(|m| m.transits.as_mut()) {
            t.record(record.key, outcome, &mut self.rng);
        }
    }

    fn close_records(&mut self, i: usize, exit_side: u8, exit_along: f32) {
        for slot in 0..MAX_TRANSIT_LEVELS {
            self.close_slot(i, slot, exit_side, exit_along);
        }
    }

    fn leave_slot(&mut self, i: usize, slot: usize, to_cell: Position, to: Point) {
        let Some(record) = self.thoughts[i].records[slot] else {
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

    fn leg_code(&self, i: usize) -> u8 {
        match self.thoughts[i].activity {
            Activity::Inbound => 1,
            Activity::Searching => 2,
            _ => 0,
        }
    }

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
        if crossed == 0 || !self.thoughts[i].activity.is_moving() {
            return false;
        }
        let (heading, leaf, laden, searching, caste) = {
            let t = &self.thoughts[i];
            (
                t.heading,
                t.leaf,
                t.laden(),
                t.activity == Activity::Searching,
                t.caste,
            )
        };
        let tempering = if searching {
            self.policies[leaf].tempering.heated(self.cfg.search_heat)
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
                    policy: caste as u8,
                    dial,
                    context,
                },
                rect,
            ));
        }
        let exploring = explore > 0.0 && self.rng.chance(explore);
        if !exploring {
            for slot in (0..crossed).rev() {
                let Some((key, rect)) = keys[slot] else {
                    continue;
                };
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
                for record in self.thoughts[i].records.iter_mut().take(slot) {
                    *record = None;
                }
                let until = self.tick + outcome.ticks.max(1) as u64;
                self.thoughts[i].transit = Some(Transit {
                    until,
                    entry: to,
                    exit,
                    outcome,
                    key,
                });
                self.stats.replayed += 1;
                return true;
            }
        }
        for (slot, key) in keys.iter().enumerate() {
            if let Some((key, rect)) = key {
                let side = (rect.2 - rect.0).max(rect.3 - rect.1) as f32;
                self.thoughts[i].records[slot] = Some(TransitRecord::open(
                    *key,
                    self.tick,
                    to,
                    (side / 4.0).max(1.0),
                ));
            }
        }
        false
    }

    fn complete_transit(&mut self, i: usize) {
        let Some(t) = self.thoughts[i].transit.take() else {
            return;
        };
        let o = t.outcome;
        let from_cell = self.thoughts[i].cell();
        let (exit, turn) = if !self.world.is_passable(t.exit.cell()) {
            match self.world.warp(t.exit) {
                Some(warp) => (warp.point, warp.turn),
                None => (t.exit, 0.0),
            }
        } else {
            (t.exit, 0.0)
        };
        let to_cell = exit.cell();
        self.occupy(from_cell, -1);
        self.occupy(to_cell, 1);
        let leg = match t.key.leg {
            1 => Leg::Inbound,
            2 => Leg::Searching,
            _ => Leg::Outbound,
        };
        let laden = t.key.laden;
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
        self.stats.decisions += o.decisions as u64;
        self.stats.decisions_replayed += o.decisions as u64;
        self.stats.entropy_sum += o.entropy as f64;
        self.stats.selected_entropy_sum += o.entropy as f64;
        self.stats.moves += o.decisions as u64;
        self.stats.length += o.length as f64;
        let level = t.key.node.level;
        let levels = self.transit_levels();
        for (slot, &l) in levels.iter().enumerate().take(MAX_TRANSIT_LEVELS) {
            if l < level {
                if let Some(r) = &mut self.thoughts[i].records[slot] {
                    r.absorb(&o, t.exit);
                }
            }
        }
        let state = self.problem.locate(exit);
        let (dx, dy) = t.entry.to(t.exit);
        let th = &mut self.thoughts[i];
        th.place = exit;
        th.heading = wrap_angle(o.heading as f64 + turn);
        th.trip_length += o.length as f64;
        th.home_vector.0 += dx;
        th.home_vector.1 += dy;
        th.rotate_frame(turn);
        th.remember(from_cell);
        th.steps += o.decisions as usize;
        if let Some(s) = state {
            th.state = s;
        }
        th.target = None;
        if th.activity.is_out() && th.route.len() < 4 * self.cfg.trip_budget.max(1) {
            th.route.push(th.state.clone());
            th.places.push(exit);
        }
        self.cross(i, from_cell, to_cell, exit);
    }

    // ---------------------------------------------------------------
    // The queen

    fn queen_epoch(&mut self) {
        let due = self
            .queen
            .as_ref()
            .map(|q| q.due(self.tick))
            .unwrap_or(false);
        if !due {
            return;
        }
        let (Some(queen), Some(history)) = (&mut self.queen, &self.history) else {
            return;
        };
        let patience = self
            .cfg
            .queen
            .as_ref()
            .map(|q| q.patience.max(1))
            .unwrap_or(1);
        // Stagnation: epochs since the yield (quality brought home per
        // epoch) last peaked or the best finding last improved.
        let yield_now = self.stats.quality_sum - self.epoch_quality_mark;
        self.epoch_quality_mark = self.stats.quality_sum;
        let improved =
            self.last_improvement_epoch == self.stats.epochs && self.stats.improvements > 0;
        // The peak is forgotten slowly, so a steady yield peaks again.
        self.peak_yield *= 0.97;
        if yield_now > self.peak_yield * 1.02 || improved {
            self.peak_yield = self.peak_yield.max(yield_now);
            self.epochs_since_peak = 0;
        } else {
            self.epochs_since_peak += 1;
        }
        let stagnation = (self.epochs_since_peak as f64 / patience as f64).min(1.0);
        // Organisation: the axial order of the invariant flow, weighted
        // by how much flows where (two-way traffic on a trail scores
        // high, wandering low).
        let (cols, rows) = history.sectors();
        let (mut ordered, mut weight) = (0.0, 0.0);
        for sector in 0..cols * rows {
            let f = history.invariant(sector);
            ordered += f.weight * f.nematic();
            weight += f.weight;
        }
        let organisation = if weight > 0.0 {
            (ordered / weight).clamp(0.0, 1.0)
        } else {
            0.0
        };
        queen.set_input(vec![vec![stagnation, -organisation]], false);
        queen.epoch(
            self.tick,
            history,
            self.memo.as_ref(),
            &mut self.hierarchy,
            &mut self.rng,
        );
        self.stats.epochs += 1;
        self.policies = self.hierarchy.compile();
    }

    /// The gains on the dials of each caste's mood (scouts, then
    /// followers): the queen's expression, one without a queen.
    pub fn dials(&self) -> Vec<f64> {
        self.moods()
            .iter()
            .map(|&m| self.hierarchy.node(m).surface.entropy.raw().exp())
            .collect()
    }

    /// The temperature each caste's thoughts decide at (scouts, then
    /// followers), with everything composed.
    pub fn temperatures(&self) -> Vec<f64> {
        self.moods()
            .iter()
            .map(|&m| {
                let leaf = self.hierarchy.node(m).children[0];
                match self.hierarchy.effective(leaf).tempering {
                    Tempering::Temperature(t) => t,
                    Tempering::Entropy(f) => f,
                }
            })
            .collect()
    }

    // ---------------------------------------------------------------
    // Reports

    /// A summary of the mind's state.
    pub fn report(&self) -> String {
        let s = &self.stats;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "{}: {} thoughts, tick {}, {} trips, {} findings, {} brought home (mean quality {:.3})",
            self.problem.name(),
            self.thoughts.len(),
            self.tick,
            s.trips,
            s.findings,
            s.deliveries,
            s.mean_quality()
        );
        match &self.best {
            Some(b) => {
                let last = b.route.last().unwrap_or(&self.origin);
                let _ = writeln!(
                    out,
                    "  best: quality {:.3} at tick {} by thought {}: {}",
                    b.quality,
                    b.tick,
                    b.thought,
                    self.problem.describe(last)
                );
            }
            None => {
                let _ = writeln!(out, "  best: nothing brought home yet");
            }
        }
        let _ = writeln!(
            out,
            "  {} dead ends ({} retreats), {} given up, {} lost; {} decisions at mean entropy {:.3}, {} moves over {:.0} cells",
            s.dead_ends, s.retreats, s.given_up, s.lost, s.decisions, s.mean_entropy(), s.moves, s.length
        );
        if let Some(m) = &self.memo {
            if let Some(t) = &m.transits {
                let (kernels, mature) = t.maturity();
                let _ = writeln!(
                    out,
                    "  habits: {} transits replayed standing in for {} decisions; {} kernels, {} mature",
                    s.replayed, s.decisions_replayed, kernels, mature
                );
            }
        }
        if self.cfg.pipeline.is_some() {
            let _ = writeln!(
                out,
                "  pipeline: {} moves on a held heading, {} deferred, mean horizon {:.2}",
                s.held,
                s.deferred,
                s.frames.mean_horizon()
            );
        }
        if let Some(h) = &self.history {
            let topo = h.topology();
            let summary = h.summary();
            let _ = writeln!(
                out,
                "  history: {} channels, {} loops, alignment {:.3}, spatial entropy {:.3}",
                topo.channels, topo.loops, summary.alignment, summary.spatial_entropy
            );
        }
        if let Some(n) = &self.net {
            out.push_str(&n.report());
        }
        if let Some(q) = &self.queen {
            let dials = self.dials();
            let temps = self.temperatures();
            let _ = writeln!(
                out,
                "  queen: {} epochs, thought {:?}, dials scouts ×{:.2} followers ×{:.2} (temperatures {:.2} and {:.2})",
                s.epochs,
                q.thought
                    .iter()
                    .map(|x| (x * 100.0).round() / 100.0)
                    .collect::<Vec<_>>(),
                dials[0],
                dials[1],
                temps[0],
                temps[1]
            );
        }
        out
    }

    /// The medium drawn: walls `#`, the origin `@`, thoughts `o`, and the
    /// trail's perceived strength as shades.
    pub fn render(&self) -> String {
        let (w, h) = (self.world.width(), self.world.height());
        let shades: &[u8] = b" .:-=+*#%";
        let k = self.world.channel(Pheromone::Trail).k.max(1e-9);
        let mut grid: Vec<Vec<char>> = (0..h)
            .map(|y| {
                (0..w)
                    .map(|x| {
                        let p = Position::new(x as i32, y as i32);
                        if !self.world.is_passable(p) {
                            return '#';
                        }
                        let level = self.world.level(p, Pheromone::Trail);
                        let v = (level / k).ln_1p();
                        let idx = ((v / 4.0) * (shades.len() - 1) as f64).round() as usize;
                        shades[idx.min(shades.len() - 1)] as char
                    })
                    .collect()
            })
            .collect();
        for t in &self.thoughts {
            if !t.activity.is_moving() {
                continue;
            }
            let c = t.cell();
            if c.x >= 0 && c.y >= 0 && (c.x as usize) < w && (c.y as usize) < h {
                grid[c.y as usize][c.x as usize] = 'o';
            }
        }
        let o = self.origin_place.cell();
        if o.x >= 0 && o.y >= 0 && (o.x as usize) < w && (o.y as usize) < h {
            grid[o.y as usize][o.x as usize] = '@';
        }
        let mut out = String::with_capacity((w + 1) * h);
        for row in grid {
            out.extend(row);
            out.push('\n');
        }
        out
    }
}

impl<P: Problem> Mind<P> {
    /// A symbol's channel drawn like [`render`](Self::render), with the
    /// symbol's glyph marked `+` and the punctures `x`.
    pub fn render_symbol(&self, k: usize) -> Option<String> {
        let net = self.net.as_ref()?;
        let s = net.symbol(k)?;
        let (w, h) = (self.world.width(), self.world.height());
        let shades: &[u8] = b" .:-=+*#%";
        let key = net.field().k;
        let mut grid: Vec<Vec<char>> = (0..h)
            .map(|y| {
                (0..w)
                    .map(|x| {
                        let p = Position::new(x as i32, y as i32);
                        if !self.world.is_passable(p) {
                            return '#';
                        }
                        let v = (net.field().level(p, s.channel) / key).ln_1p();
                        let idx = ((v / 4.0) * (shades.len() - 1) as f64).round() as usize;
                        shades[idx.min(shades.len() - 1)] as char
                    })
                    .collect()
            })
            .collect();
        let mark = |grid: &mut Vec<Vec<char>>, p: Point, ch: char| {
            let c = p.cell();
            if c.x >= 0 && c.y >= 0 && (c.x as usize) < w && (c.y as usize) < h {
                grid[c.y as usize][c.x as usize] = ch;
            }
        };
        for p in &s.glyph.points {
            mark(&mut grid, *p, '+');
        }
        for p in net.punctures() {
            mark(&mut grid, *p, 'x');
        }
        let mut out = String::with_capacity((w + 1) * h);
        for row in grid {
            out.extend(row);
            out.push('\n');
        }
        Some(out)
    }
}

/// The point a distance along a polyline.
fn point_along(path: &[Point], segments: &[f64], distance: f64) -> Point {
    let mut left = distance.max(0.0);
    for (w, &len) in path.windows(2).zip(segments) {
        if left <= len || len <= 0.0 {
            let f = if len > 0.0 {
                (left / len).min(1.0)
            } else {
                0.0
            };
            return Point::new(
                w[0].x + (w[1].x - w[0].x) * f,
                w[0].y + (w[1].y - w[0].y) * f,
            );
        }
        left -= len;
    }
    *path.last().expect("a path")
}
