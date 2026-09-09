//! Canonical experiments from the ant-foraging literature: world builders,
//! replicate runners, and summary statistics.
//!
//! * [`double_bridge`]: the nest and a food source joined by two branches of
//!   different lengths (Goss, Aron, Deneubourg & Pasteels 1989; Beckers,
//!   Deneubourg & Goss 1992). Trail reinforcement makes the colony converge
//!   on the shorter branch; with equal branches it still converges on one
//!   (symmetry breaking, Deneubourg et al. 1990).
//! * [`two_sources`]: two sources at equal distance but different quality
//!   (Beckers, Deneubourg, Goss & Pasteels 1990, *Insectes Sociaux* 37:258).
//!   Quality-modulated trail laying focuses the colony on the richer one.
//! * [`run_hunger_response`]: foraging activity as a function of colony
//!   satiation (Mailleux, Deneubourg & Detrain 2003).
//! * [`run_division_of_labor`]: specialisation with and without response
//!   threshold reinforcement (Theraulaz, Bonabeau & Deneubourg 1998).
//! * [`run_colony_size_scan`]: foraging organisation against colony size
//!   (Beekman, Sumpter & Ratnieks 2001): below a critical size a colony
//!   cannot keep a trail against evaporation and forages by individual
//!   search; above it, foraging is organised along the trail.
//! * [`run_memory_scaling`]: the hive's memory capacity against colony
//!   size and against the grain of the reading.
//! * [`run_memory_probe`] and [`run_closed_loop`]: the hive's cognitive
//!   geometry. A queen writes random thoughts into an entropy dial of a
//!   colony kept foraging for hours; readouts of the movement history
//!   retrodict them, lag by lag, from the invariant and the non-invariant
//!   component of the field; and with recall feeding her next thought,
//!   an alternation is sustained through the colony's movement alone.

use crate::ant::{BASE_FEATURES, F_CROWD, F_ODOUR, F_RECENT, F_ROUTE};
use crate::colony::{BroodItem, BroodStage, SimConfig, Simulation};
use crate::geometry::Position;
use crate::hierarchy::NodeId;
use crate::hive::{
    memory_capacity, Capacity, Component, FieldSummary, HistoryConfig, QueenConfig, Recursion,
};
use crate::memo::MemoConfig;
use crate::pheromone::Pheromone;
use crate::species::Species;
use crate::world::{CapacityZone, Counter, FoodSource, Rect, WorldConfig};

/// Geometry of a double bridge, in cells.
///
/// Both branches leave each junction perpendicular to the nest–food axis,
/// one looping above it and one below, so the choice at a junction is
/// geometrically symmetric and only pheromone (and chance) decides. A
/// branch's length is `span + 1 + 2 × excursion`.
#[derive(Clone, Debug, PartialEq)]
pub struct BridgeSpec {
    /// Horizontal distance between the two junctions.
    pub span: usize,
    /// Excursion of the lower (short) branch below the axis.
    pub short_excursion: usize,
    /// Excursion of the upper (long) branch above the axis.
    pub long_excursion: usize,
    /// Length of the corridors from the nest to the first junction and from
    /// the second junction to the food.
    pub stem: usize,
}

impl BridgeSpec {
    /// A bridge whose long branch is twice the short one (8 against 16
    /// cells).
    pub fn ratio_two() -> Self {
        BridgeSpec {
            span: 5,
            short_excursion: 1,
            long_excursion: 5,
            stem: 3,
        }
    }

    /// Two branches of equal length (the symmetry-breaking setup).
    pub fn equal() -> Self {
        BridgeSpec {
            span: 5,
            short_excursion: 2,
            long_excursion: 2,
            stem: 3,
        }
    }

    /// Length of the short (lower) branch.
    pub fn short(&self) -> usize {
        self.span + 1 + 2 * self.short_excursion
    }

    /// Length of the long (upper) branch.
    pub fn long(&self) -> usize {
        self.span + 1 + 2 * self.long_excursion
    }
}

/// Build the double-bridge world. Counters named `"short"` (lower branch)
/// and `"long"` (upper branch) sit in the middle of each branch.
pub fn double_bridge(spec: &BridgeSpec) -> WorldConfig {
    let up = spec.long_excursion.max(1) as i32;
    let down = spec.short_excursion.max(1) as i32;
    let y_mid = up + 3;
    let height = (up + down + 7) as usize;
    let nest = Position::new(2, y_mid);
    let x_j1 = 4 + spec.stem as i32;
    let x_j2 = x_j1 + spec.span as i32 + 1;
    let x_food = x_j2 + spec.stem as i32 + 1;
    let width = (x_food + 4) as usize;
    let row = |x0: i32, x1: i32, y: i32| Rect::new(Position::new(x0, y), Position::new(x1, y));
    let col = |x: i32, y0: i32, y1: i32| Rect::new(Position::new(x, y0), Position::new(x, y1));
    let x_mid = (x_j1 + x_j2) / 2;
    let open = vec![
        // nest chamber, stems (which end at the junction cells)
        Rect::new(Position::new(1, y_mid - 1), Position::new(3, y_mid + 1)),
        row(4, x_j1, y_mid),
        row(x_j2, x_food - 1, y_mid),
        // upper (long) branch
        col(x_j1, y_mid - up, y_mid - 1),
        row(x_j1, x_j2, y_mid - up),
        col(x_j2, y_mid - up, y_mid - 1),
        // lower (short) branch
        col(x_j1, y_mid + 1, y_mid + down),
        row(x_j1, x_j2, y_mid + down),
        col(x_j2, y_mid + 1, y_mid + down),
        // food chamber
        Rect::new(
            Position::new(x_food, y_mid - 1),
            Position::new(x_food + 2, y_mid + 1),
        ),
    ];
    WorldConfig {
        width,
        height,
        nest,
        nest_radius: 1,
        food_sources: vec![FoodSource::pool(
            Position::new(x_food + 1, y_mid),
            1,
            1.0e6,
            1.0,
        )],
        random_food: None,
        walls: Vec::new(),
        open,
        counters: vec![
            Counter {
                name: "short".to_string(),
                rect: row(x_mid, x_mid, y_mid + down),
            },
            Counter {
                name: "long".to_string(),
                rect: row(x_mid, x_mid, y_mid - up),
            },
        ],
        ..WorldConfig::default()
    }
}

/// The equal double bridge with branches of a given width, expressed as
/// the number of ants a branch cell holds; the nest and food chambers hold
/// any number (Dussutour, Fourcassié, Helbing & Deneubourg 2004: bridges
/// of 10 and 6 mm).
pub fn crowded_bridge(spec: &BridgeSpec, branch_capacity: u16) -> WorldConfig {
    let mut world = double_bridge(spec);
    let up = spec.long_excursion.max(1) as i32;
    let down = spec.short_excursion.max(1) as i32;
    let y_mid = up + 3;
    let x_j1 = 4 + spec.stem as i32;
    let x_j2 = x_j1 + spec.span as i32 + 1;
    let x_food = x_j2 + spec.stem as i32 + 1;
    world.capacity_zones = vec![
        // stems and junctions
        CapacityZone {
            rect: Rect::new(Position::new(4, y_mid), Position::new(x_food - 1, y_mid)),
            capacity: branch_capacity,
        },
        // branches
        CapacityZone {
            rect: Rect::new(
                Position::new(x_j1, y_mid - up),
                Position::new(x_j2, y_mid - 1),
            ),
            capacity: branch_capacity,
        },
        CapacityZone {
            rect: Rect::new(
                Position::new(x_j1, y_mid + 1),
                Position::new(x_j2, y_mid + down),
            ),
            capacity: branch_capacity,
        },
        CapacityZone {
            rect: Rect::new(
                Position::new(x_food, y_mid - 1),
                Position::new(x_food + 2, y_mid + 1),
            ),
            capacity: u16::MAX,
        },
    ];
    world
}

/// Run one crowded equal-bridge replicate: `ants` foragers on branches
/// holding `branch_capacity` ants per cell. Returns the lower branch as
/// "short".
pub fn run_crowded_bridge(
    species: Species,
    ants: usize,
    branch_capacity: u16,
    seconds: f64,
    window_s: f64,
    seed: u64,
) -> BridgeOutcome {
    let world = crowded_bridge(&BridgeSpec::equal(), branch_capacity);
    let cfg = experiment_config(species, world, ants);
    run_double_bridge_configured(cfg, seconds, window_s, seed)
}

/// Outcome of one double-bridge replicate.
#[derive(Clone, Debug, PartialEq)]
pub struct BridgeOutcome {
    /// Crossings of the short branch's counter during the final window.
    pub short_crossings: u64,
    /// Crossings of the long branch's counter during the final window.
    pub long_crossings: u64,
    /// Share of final-window traffic on the short branch (0.5 if none).
    pub short_fraction: f64,
    /// Recruitment trail on the short branch's counter cell at the end.
    pub trail_short: f64,
    /// Recruitment trail on the long branch's counter cell at the end.
    pub trail_long: f64,
    /// Crop loads delivered.
    pub delivered: u64,
}

/// Base simulation configuration used by the experiment runners: a hungry
/// colony of `ants` workers of `species` in `world`, with logging every
/// minute.
pub fn experiment_config(species: Species, world: WorldConfig, ants: usize) -> SimConfig {
    let mut cfg = SimConfig::for_species(species);
    cfg.world = world;
    cfg.ants = ants;
    // The simulated ants are the foraging force of a starved colony many
    // times their number, whose appetite does not saturate within the
    // experiment: the store stands for that colony.
    cfg.nest.store_capacity_mg_per_ant = 5.0;
    cfg.nest.initial_satiation = 0.05;
    cfg.nest.initial_brood_per_ant = 0.0;
    cfg.nest.queen = false;
    cfg.nest.log_every_s = 60.0;
    cfg
}

/// Reduce a configuration to the assumptions of the classic
/// double-bridge model (Deneubourg et al. 1990): a trail that does not
/// evaporate on the experiment's timescale, no behavioural negative
/// feedback at junctions (no crowding term, no memory of recently visited
/// cells), and no private route memory. Symmetry breaking on equal
/// branches is a marginal instability that any negative feedback
/// suppresses, and private memory splits the colony into individuals each
/// faithful to their own first choice (Grüter, Czaczkes & Ratnieks 2011
/// found route memory overriding the trail in *Lasius niger*), which
/// starves the collective feedback of the difference it needs to amplify.
pub fn pure_pheromone_feedback(cfg: &mut SimConfig) {
    cfg.species.trail.half_life_s = f64::INFINITY;
    let mut world_pheromones = cfg.species.pheromones();
    world_pheromones[Pheromone::Trail.index()].half_life_s = f64::INFINITY;
    cfg.world.pheromones = Some(world_pheromones);
    cfg.species.route_capacity = 0;
    cfg.world.cell_capacity = u16::MAX;
    cfg.species.crowding_slowdown = 0.0;
    cfg.species.crowding_deposition = 0.0;
    cfg.species.food_odour = crate::pheromone::PheromoneParams::inert();
    for f in [F_CROWD, F_RECENT, F_ROUTE, F_ODOUR] {
        cfg.instinct.weights[f] = 0.0;
        cfg.instinct.weights[BASE_FEATURES + f] = 0.0;
    }
}

fn crossings_in_window(sim: &Simulation, name: &str, window_s: f64) -> u64 {
    let idx = sim
        .world()
        .counters()
        .iter()
        .position(|c| c.counter.name == name)
        .expect("counter exists");
    let now = sim.world().counters()[idx].crossings;
    let start_time = sim.time_s() - window_s;
    let before = sim
        .stats()
        .log
        .iter()
        .rev()
        .find(|s| s.time_s <= start_time)
        .map(|s| s.counters[idx])
        .unwrap_or(0);
    now.saturating_sub(before)
}

/// Run one double-bridge replicate for `seconds` and measure the traffic
/// split over the final `window_s` seconds.
pub fn run_double_bridge_once(
    spec: &BridgeSpec,
    species: Species,
    ants: usize,
    seconds: f64,
    window_s: f64,
    seed: u64,
) -> BridgeOutcome {
    let world = double_bridge(spec);
    let cfg = experiment_config(species, world, ants);
    run_double_bridge_configured(cfg, seconds, window_s, seed)
}

/// Run one double-bridge replicate from a prepared configuration (whose
/// world must come from [`double_bridge`]).
pub fn run_double_bridge_configured(
    cfg: SimConfig,
    seconds: f64,
    window_s: f64,
    seed: u64,
) -> BridgeOutcome {
    let mut sim = Simulation::new(cfg, seed);
    sim.run_seconds(seconds);
    let short = crossings_in_window(&sim, "short", window_s);
    let long = crossings_in_window(&sim, "long", window_s);
    let total = short + long;
    let counters = sim.world().counters();
    let trail = |name: &str| {
        let c = counters
            .iter()
            .find(|c| c.counter.name == name)
            .expect("counter exists");
        sim.world().pheromone_in(&c.counter.rect, Pheromone::Trail)
    };
    BridgeOutcome {
        short_crossings: short,
        long_crossings: long,
        short_fraction: if total == 0 {
            0.5
        } else {
            short as f64 / total as f64
        },
        trail_short: trail("short"),
        trail_long: trail("long"),
        delivered: sim.stats().food_delivered,
    }
}

/// Run several double-bridge replicates.
pub fn run_double_bridge(
    spec: &BridgeSpec,
    species: &Species,
    ants: usize,
    seconds: f64,
    window_s: f64,
    seeds: &[u64],
) -> Vec<BridgeOutcome> {
    seeds
        .iter()
        .map(|&seed| run_double_bridge_once(spec, species.clone(), ants, seconds, window_s, seed))
        .collect()
}

/// Summary of a set of fractions.
#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    /// Number of replicates.
    pub n: usize,
    /// Mean.
    pub mean: f64,
    /// Standard deviation.
    pub std: f64,
    /// Fraction of replicates above one half.
    pub majority: f64,
    /// Fraction of replicates outside `0.2..0.8` (a decision either way).
    pub decided: f64,
}

/// Summarise fractions.
pub fn summarize(fractions: &[f64]) -> Summary {
    let n = fractions.len();
    if n == 0 {
        return Summary {
            n: 0,
            mean: 0.0,
            std: 0.0,
            majority: 0.0,
            decided: 0.0,
        };
    }
    let mean = fractions.iter().sum::<f64>() / n as f64;
    let var = fractions.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / n as f64;
    Summary {
        n,
        mean,
        std: var.sqrt(),
        majority: fractions.iter().filter(|f| **f > 0.5).count() as f64 / n as f64,
        decided: fractions.iter().filter(|f| **f < 0.2 || **f > 0.8).count() as f64 / n as f64,
    }
}

/// Build an open arena with the nest in the middle and two food sources at
/// the same distance east (`molarity_a`) and west (`molarity_b`).
pub fn two_sources(distance: i32, molarity_a: f64, molarity_b: f64) -> WorldConfig {
    let margin = 4;
    let width = (2 * distance + 2 * margin + 1) as usize;
    let height = (2 * margin + 9) as usize;
    let nest = Position::new(distance + margin, margin + 4);
    WorldConfig {
        width,
        height,
        nest,
        nest_radius: 1,
        food_sources: vec![
            FoodSource::pool(
                Position::new(nest.x + distance, nest.y),
                1,
                1.0e6,
                molarity_a,
            ),
            FoodSource::pool(
                Position::new(nest.x - distance, nest.y),
                1,
                1.0e6,
                molarity_b,
            ),
        ],
        random_food: None,
        counters: vec![
            Counter {
                name: "a".to_string(),
                rect: Rect::new(
                    Position::new(nest.x + distance - 1, nest.y - 1),
                    Position::new(nest.x + distance + 1, nest.y + 1),
                ),
            },
            Counter {
                name: "b".to_string(),
                rect: Rect::new(
                    Position::new(nest.x - distance - 1, nest.y - 1),
                    Position::new(nest.x - distance + 1, nest.y + 1),
                ),
            },
        ],
        ..WorldConfig::default()
    }
}

/// Outcome of one two-source replicate.
#[derive(Clone, Debug, PartialEq)]
pub struct SourcesOutcome {
    /// Solution taken from source A, microlitres.
    pub volume_a: f64,
    /// Solution taken from source B, microlitres.
    pub volume_b: f64,
    /// Share of the volume taken from A (0.5 if none).
    pub fraction_a: f64,
    /// Loads delivered.
    pub delivered: u64,
}

/// Run one two-source replicate.
pub fn run_two_sources_once(
    distance: i32,
    molarity_a: f64,
    molarity_b: f64,
    species: Species,
    ants: usize,
    seconds: f64,
    seed: u64,
) -> SourcesOutcome {
    let world = two_sources(distance, molarity_a, molarity_b);
    let cfg = experiment_config(species, world, ants);
    let mut sim = Simulation::new(cfg, seed);
    let rect_of = |name: &str| {
        sim.world()
            .counters()
            .iter()
            .find(|c| c.counter.name == name)
            .expect("counter exists")
            .counter
            .rect
    };
    let (ra, rb) = (rect_of("a"), rect_of("b"));
    let (fa0, fb0) = (sim.world().food_in(&ra), sim.world().food_in(&rb));
    sim.run_seconds(seconds);
    let volume_a = fa0 - sim.world().food_in(&ra);
    let volume_b = fb0 - sim.world().food_in(&rb);
    let total = volume_a + volume_b;
    SourcesOutcome {
        volume_a,
        volume_b,
        fraction_a: if total <= 0.0 { 0.5 } else { volume_a / total },
        delivered: sim.stats().food_delivered,
    }
}

/// Foraging activity at one satiation level.
#[derive(Clone, Debug, PartialEq)]
pub struct HungerOutcome {
    /// Initial store fill.
    pub satiation: f64,
    /// Fraction of ant-time spent outside the nest.
    pub foraging_fraction: f64,
    /// Crop loads delivered.
    pub delivered: u64,
}

/// Foraging activity of otherwise identical colonies started at several
/// satiation levels.
pub fn run_hunger_response(
    species: &Species,
    ants: usize,
    satiations: &[f64],
    seconds: f64,
    seed: u64,
) -> Vec<HungerOutcome> {
    satiations
        .iter()
        .map(|&satiation| {
            let world = WorldConfig {
                seed: Some(seed),
                ..WorldConfig::default()
            };
            let mut cfg = experiment_config(species.clone(), world, ants);
            cfg.nest.initial_satiation = satiation;
            let mut sim = Simulation::new(cfg, seed);
            sim.run_seconds(seconds);
            HungerOutcome {
                satiation,
                foraging_fraction: sim.stats().foraging_fraction(),
                delivered: sim.stats().food_delivered,
            }
        })
        .collect()
}

/// A single drop of solution at `distance` cells east of the nest, fed at
/// `flow_ul_per_min` (Mailleux, Deneubourg & Detrain 2003: a source whose
/// productivity the colony's foraging effort comes to match).
pub fn dripping_source(distance: i32, flow_ul_per_min: f64, molarity: f64) -> WorldConfig {
    let margin = 4;
    let width = (distance + 2 * margin + 1) as usize;
    let height = (2 * margin + 9) as usize;
    let nest = Position::new(margin, margin + 4);
    WorldConfig {
        width,
        height,
        nest,
        nest_radius: 1,
        food_sources: vec![FoodSource::pool(
            Position::new(nest.x + distance, nest.y),
            0,
            1.0,
            molarity,
        )
        .renewing(flow_ul_per_min / 60.0)],
        random_food: None,
        ..WorldConfig::default()
    }
}

/// Outcome of one dripping-source replicate.
#[derive(Clone, Debug, PartialEq)]
pub struct ProductivityOutcome {
    /// Flow of the source, microlitres per minute.
    pub flow_ul_per_min: f64,
    /// Share of ant-time spent outside.
    pub foraging_fraction: f64,
    /// Loads delivered.
    pub delivered: u64,
    /// Solution drunk, microlitres.
    pub collected_ul: f64,
    /// Mean crop load per feeding visit, microlitres.
    pub mean_load_ul: f64,
    /// Share of returns on which the forager laid trail.
    pub recruiting_fraction: f64,
    /// Mean number of ants at the source (its cell and neighbours),
    /// sampled every minute.
    pub at_source: f64,
}

/// Foraging effort and recruitment against the productivity of a source:
/// one replicate per flow, 1 M sucrose 12 cells from the nest.
pub fn run_productivity_response(
    species: &Species,
    ants: usize,
    flows_ul_per_min: &[f64],
    seconds: f64,
    seed: u64,
) -> Vec<ProductivityOutcome> {
    flows_ul_per_min
        .iter()
        .map(|&flow| {
            let world = dripping_source(12, flow, 1.0);
            let source = world.food_sources[0].center;
            let around = Rect::new(source.offset(-1, -1), source.offset(1, 1));
            let cfg = experiment_config(species.clone(), world, ants);
            let mut sim = Simulation::new(cfg, seed);
            let minutes = (seconds / 60.0).ceil().max(1.0) as usize;
            let mut at_source = 0.0;
            for _ in 0..minutes {
                sim.run_seconds(60.0);
                at_source += sim.world().occupancy_in(&around) as f64;
            }
            let s = sim.stats();
            ProductivityOutcome {
                flow_ul_per_min: flow,
                foraging_fraction: s.foraging_fraction(),
                delivered: s.food_delivered,
                collected_ul: s.food_collected_ul,
                mean_load_ul: s.food_collected_ul / s.food_picked.max(1) as f64,
                recruiting_fraction: s.recruiting_trips as f64 / s.food_picked.max(1) as f64,
                at_source: at_source / minutes as f64,
            }
        })
        .collect()
}

/// A sugar source east of the nest and a heap of prey west of it, at the
/// same distance (Dussutour & Simpson 2009: a choice between the two
/// macronutrients).
pub fn sugar_and_prey(distance: i32) -> WorldConfig {
    let margin = 4;
    let width = (2 * distance + 2 * margin + 1) as usize;
    let height = (2 * margin + 9) as usize;
    let nest = Position::new(distance + margin, margin + 4);
    WorldConfig {
        width,
        height,
        nest,
        nest_radius: 1,
        food_sources: vec![
            FoodSource::pool(Position::new(nest.x + distance, nest.y), 1, 1.0e6, 1.0),
            FoodSource::prey(Position::new(nest.x - distance, nest.y), 1, 1.0e6),
        ],
        random_food: None,
        ..WorldConfig::default()
    }
}

/// Outcome of one communal-nutrition replicate.
#[derive(Clone, Debug, PartialEq)]
pub struct NutritionOutcome {
    /// Whether the colony had larvae to feed.
    pub with_larvae: bool,
    /// Sugar delivered, milligrams.
    pub sugar_mg: f64,
    /// Protein delivered, milligrams.
    pub protein_mg: f64,
    /// Protein as a share of everything delivered by mass (0.5 if nothing).
    pub protein_share: f64,
    /// Loads delivered.
    pub delivered: u64,
}

/// Run one communal-nutrition replicate: a colony of `ants` workers, with
/// one larva per worker or none, choosing between sugar and prey.
pub fn run_communal_nutrition(
    species: Species,
    ants: usize,
    with_larvae: bool,
    seconds: f64,
    seed: u64,
) -> NutritionOutcome {
    let larva_protein_mg = species.larva_protein_mg;
    let _ = larva_protein_mg;
    let cfg = experiment_config(species, sugar_and_prey(12), ants);
    let mut sim = Simulation::new(cfg, seed);
    if with_larvae {
        let larvae = (0..ants)
            .map(|_| BroodItem {
                stage: BroodStage::Larva,
                stage_age_s: 0.0,
                fed_mg: 0.0,
                protein_mg: 0.0,
                unfed_s: 0.0,
            })
            .collect();
        sim.nest_mut().brood = larvae;
    }
    sim.run_seconds(seconds);
    let s = sim.stats();
    let total = s.sugar_delivered_mg + s.protein_delivered_mg;
    NutritionOutcome {
        with_larvae,
        sugar_mg: s.sugar_delivered_mg,
        protein_mg: s.protein_delivered_mg,
        protein_share: if total > 0.0 {
            s.protein_delivered_mg / total
        } else {
            0.5
        },
        delivered: s.food_delivered,
    }
}

/// An open arena of `side` cells with `corpses` dead ants scattered over
/// it and a nest in the middle (Theraulaz et al. 2002: corpses in a
/// circular arena are gathered into a few piles).
pub fn cemetery_arena(side: usize, corpses: usize, seed: u64) -> WorldConfig {
    let mid = (side / 2) as i32;
    WorldConfig {
        width: side,
        height: side,
        nest: Position::new(mid, mid),
        nest_radius: 1,
        food_sources: Vec::new(),
        random_food: None,
        scattered_corpses: corpses,
        seed: Some(seed),
        ..WorldConfig::default()
    }
}

/// Outcome of one cemetery replicate.
#[derive(Clone, Debug, PartialEq)]
pub struct CemeteryOutcome {
    /// Piles at the start (eight-connected groups of cells holding
    /// corpses).
    pub clusters_start: usize,
    /// Piles at the end.
    pub clusters_end: usize,
    /// Corpses in the largest pile at the start.
    pub largest_start: u32,
    /// Corpses in the largest pile at the end.
    pub largest_end: u32,
    /// Corpses picked up over the run.
    pub corpses_moved: u64,
}

/// Run one cemetery replicate: `ants` hungry workers of `species` with no
/// food to find, roaming an arena with `corpses` dead nestmates.
pub fn run_cemetery(
    species: Species,
    ants: usize,
    corpses: usize,
    seconds: f64,
    seed: u64,
) -> CemeteryOutcome {
    let world = cemetery_arena(40, corpses, seed);
    let mut cfg = experiment_config(species, world, ants);
    cfg.nest.mortality = false;
    let mut sim = Simulation::new(cfg, seed);
    let start = sim.world().corpse_clusters();
    sim.run_seconds(seconds);
    let end = sim.world().corpse_clusters();
    CemeteryOutcome {
        clusters_start: start.len(),
        clusters_end: end.len(),
        largest_start: start.first().copied().unwrap_or(0),
        largest_end: end.first().copied().unwrap_or(0),
        corpses_moved: sim.stats().corpses_moved,
    }
}

/// A world with one small pool of sugar solution off to one side of the
/// nest, for timing its discovery.
pub fn hidden_source(distance: i32) -> WorldConfig {
    let margin = 6;
    let side = (2 * distance + 2 * margin + 1) as usize;
    let nest = Position::new(distance + margin, distance + margin);
    let off = ((distance as f64) / std::f64::consts::SQRT_2).round() as i32;
    WorldConfig {
        width: side,
        height: side,
        nest,
        nest_radius: 1,
        food_sources: vec![FoodSource::pool(
            Position::new(nest.x + off, nest.y - off),
            1,
            1.0e6,
            1.0,
        )],
        random_food: None,
        ..WorldConfig::default()
    }
}

/// Outcome of one discovery replicate.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscoveryOutcome {
    /// Whether the food gave off an odour.
    pub odour: bool,
    /// Time of the first feeding visit, seconds, if any.
    pub first_find_s: Option<f64>,
    /// Loads delivered by the end.
    pub delivered: u64,
}

/// Time the first discovery of a hidden source by `ants` workers of
/// `species`, with the food's odour switched on or off.
pub fn run_discovery(
    species: Species,
    ants: usize,
    distance: i32,
    odour: bool,
    seconds: f64,
    seed: u64,
) -> DiscoveryOutcome {
    let mut species = species;
    if !odour {
        species.food_odour = crate::pheromone::PheromoneParams::inert();
    }
    let cfg = experiment_config(species, hidden_source(distance), ants);
    let mut sim = Simulation::new(cfg, seed);
    let steps = (seconds / sim.tick_s()).round().max(1.0) as usize;
    let mut first_find_s = None;
    for _ in 0..steps {
        sim.step();
        if first_find_s.is_none() && sim.stats().food_picked > 0 {
            first_find_s = Some(sim.time_s());
        }
    }
    DiscoveryOutcome {
        odour,
        first_find_s,
        delivered: sim.stats().food_delivered,
    }
}

/// Outcome of one temperature in a thermal trade-off scan.
#[derive(Clone, Debug, PartialEq)]
pub struct ThermalOutcome {
    /// Surface temperature, °C.
    pub temperature_c: f64,
    /// Loads delivered.
    pub delivered: u64,
    /// Workers killed by heat.
    pub deaths_heat: u64,
    /// Walking-speed multiplier at that temperature.
    pub speed_factor: f64,
}

/// Foraging return against heat losses across temperatures (Cerdá, Retana
/// & Cros 1998): a hungry colony of `ants` workers on the default map for
/// `seconds` at each temperature.
pub fn run_thermal_tradeoff(
    species: &Species,
    ants: usize,
    temperatures_c: &[f64],
    seconds: f64,
    seed: u64,
) -> Vec<ThermalOutcome> {
    temperatures_c
        .iter()
        .map(|&t| {
            let world = WorldConfig {
                seed: Some(seed),
                ..WorldConfig::default()
            };
            let mut cfg = experiment_config(species.clone(), world, ants);
            cfg.nest.mortality = true;
            cfg.environment.temperature_c = t;
            let mut sim = Simulation::new(cfg, seed);
            sim.run_seconds(seconds);
            ThermalOutcome {
                temperature_c: t,
                delivered: sim.stats().food_delivered,
                deaths_heat: sim.stats().deaths_heat,
                speed_factor: species.speed_factor(t),
            }
        })
        .collect()
}

/// Division of labour with and without threshold reinforcement.
#[derive(Clone, Debug, PartialEq)]
pub struct LaborOutcome {
    /// Index with reinforcement (the species as given).
    pub with_reinforcement: f64,
    /// Index with reinforcement switched off (infinite time constants).
    pub without_reinforcement: f64,
}

/// Run a colony with brood twice, with and without threshold reinforcement.
pub fn run_division_of_labor(
    species: &Species,
    ants: usize,
    seconds: f64,
    seed: u64,
) -> LaborOutcome {
    let run = |species: Species| {
        let world = WorldConfig {
            seed: Some(seed),
            ..WorldConfig::default()
        };
        let mut cfg = experiment_config(species, world, ants);
        cfg.nest.initial_satiation = 0.3;
        cfg.nest.initial_brood_per_ant = 1.0;
        let mut sim = Simulation::new(cfg, seed);
        sim.run_seconds(seconds);
        sim.division_of_labor()
    };
    let mut frozen = species.clone();
    frozen.threshold_learning_s = f64::INFINITY;
    frozen.threshold_forgetting_s = f64::INFINITY;
    LaborOutcome {
        with_reinforcement: run(species.clone()),
        without_reinforcement: run(frozen),
    }
}

/// The single feeder of Beekman, Sumpter & Ratnieks 2001: a nest whose
/// one entrance is a corridor `stem` cells long opening into an arena,
/// with an inexhaustible pool of 1 M solution `distance` cells beyond the
/// corridor's mouth. Every forager leaves and returns through the
/// corridor, where the trail, if there is one, lies.
pub fn single_feeder(stem: i32, distance: i32) -> WorldConfig {
    let margin = 6;
    let height = (2 * margin + distance + 3) as usize;
    let nest = Position::new(2, height as i32 / 2);
    let mouth = nest.x + 2 + stem;
    let width = (mouth + distance + margin + 1) as usize;
    WorldConfig {
        width,
        height,
        nest,
        nest_radius: 1,
        open: vec![
            Rect::new(
                Position::new(nest.x - 1, nest.y - 1),
                Position::new(nest.x + 1, nest.y + 1),
            ),
            Rect::new(
                Position::new(nest.x + 2, nest.y),
                Position::new(mouth - 1, nest.y),
            ),
            Rect::new(
                Position::new(mouth, 0),
                Position::new(width as i32 - 1, height as i32 - 1),
            ),
        ],
        food_sources: vec![FoodSource::pool(
            Position::new(mouth + distance, nest.y),
            1,
            1.0e6,
            1.0,
        )],
        random_food: None,
        ..WorldConfig::default()
    }
}

/// Take a colony's individual navigation away, so that the trail is the
/// only way to a source (the world of Beekman, Sumpter & Ratnieks 2001):
/// no route memory, no memory of a site between trips, no smell of
/// food. Path integration still brings a forager home, and the trail
/// still evaporates at the species' rate.
pub fn trail_only(cfg: &mut SimConfig) {
    cfg.species.route_capacity = 0;
    cfg.species.fidelity_quality_half = f64::INFINITY;
    cfg.species.food_odour = crate::pheromone::PheromoneParams::inert();
    for f in [F_ROUTE, F_ODOUR] {
        cfg.instinct.weights[f] = 0.0;
        cfg.instinct.weights[BASE_FEATURES + f] = 0.0;
    }
}

/// Foraging at one colony size, measured over a window after the colony
/// has settled.
#[derive(Clone, Debug, PartialEq)]
pub struct SizeOutcome {
    /// Simulated workers.
    pub ants: usize,
    /// Loads delivered in the window.
    pub delivered: u64,
    /// Trips that came home empty in the window.
    pub failed_trips: u64,
    /// Share of trips that brought food home: the order of the foraging
    /// (individual search fails often; a trail hardly ever).
    pub success: f64,
    /// Loads per worker per hour.
    pub per_capita_per_hour: f64,
    /// Share of ant-time spent outside.
    pub outside_fraction: f64,
    /// Trail concentration halfway between the corridor's mouth and the
    /// feeder at the end, in units of the perception constant.
    pub trail_mid: f64,
    /// Mean direct distance over path length of the trips: one for
    /// straight trips, small for wandering ones.
    pub directness: f64,
    /// Share of the outbound legs that reached the food no more than one
    /// and a half times the direct distance: the ordered foraging, guided
    /// by a trail or a memory rather than by search.
    pub ordered: f64,
}

/// The colony-size scan: colonies of several sizes at a single feeder,
/// each settled and then measured over a window.
#[derive(Clone, Debug, PartialEq)]
pub struct SizeScan {
    /// The species.
    pub species: Species,
    /// Colony sizes to run.
    pub sizes: Vec<usize>,
    /// Length of the entrance corridor, cells.
    pub stem: i32,
    /// Distance from the corridor's mouth to the feeder, cells.
    pub distance: i32,
    /// Time the colony is given to settle before measuring, seconds.
    pub settle_s: f64,
    /// The measurement window, seconds.
    pub window_s: f64,
    /// Seed of the runs.
    pub seed: u64,
    /// Whether the colony has no individual navigation to the feeder
    /// (see [`trail_only`]).
    pub trail_only: bool,
}

impl Default for SizeScan {
    fn default() -> Self {
        SizeScan {
            species: Species::pharaoh(),
            sizes: vec![10, 20, 40, 80, 160, 320, 640],
            stem: 6,
            distance: 20,
            settle_s: 20.0 * 60.0,
            window_s: 20.0 * 60.0,
            seed: 3,
            trail_only: true,
        }
    }
}

/// Run a colony-size scan.
pub fn run_colony_size_scan(scan: &SizeScan) -> Vec<SizeOutcome> {
    scan.sizes
        .iter()
        .map(|&ants| {
            let world = single_feeder(scan.stem, scan.distance);
            let nest = world.nest;
            let mid = Position::new(nest.x + 2 + scan.stem + scan.distance / 2, nest.y);
            let mut cfg = experiment_config(scan.species.clone(), world, ants);
            cfg.nest.max_ants = cfg.nest.max_ants.max(ants);
            if scan.trail_only {
                trail_only(&mut cfg);
            }
            let mut sim = Simulation::new(cfg, scan.seed);
            sim.run_seconds(scan.settle_s);
            sim.reset_stats();
            sim.run_seconds(scan.window_s);
            let s = sim.stats();
            let trips = s.food_delivered + s.failed_trips;
            let k = sim.world().channel(Pheromone::Trail).k.max(1e-9);
            let hours = scan.window_s / 3600.0;
            SizeOutcome {
                ants,
                delivered: s.food_delivered,
                failed_trips: s.failed_trips,
                success: if trips == 0 {
                    0.0
                } else {
                    s.food_delivered as f64 / trips as f64
                },
                per_capita_per_hour: s.food_delivered as f64 / (ants as f64 * hours).max(1e-9),
                outside_fraction: s.foraging_fraction(),
                trail_mid: sim.world().level(mid, Pheromone::Trail) / k,
                directness: if s.path.trip_length > 0.0 {
                    s.path.trip_direct / s.path.trip_length
                } else {
                    0.0
                },
                ordered: s.path.ordered_fraction(),
            }
        })
        .collect()
}

/// The hive's memory at one colony size and grain of reading.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryScaleOutcome {
    /// Simulated workers.
    pub ants: usize,
    /// Cells per sector side of the reading.
    pub sector: usize,
    /// Whether the reading was multi-scale.
    pub multiscale: bool,
    /// Memory capacity of the residual component.
    pub residual: Capacity,
    /// Held-out R² of the residual readout at lag 1.
    pub lag1: f64,
    /// Correlation of the thought with the residual straightness of the
    /// whole field.
    pub straightness_correlation: f64,
}

/// Run the memory probe at each colony size (at the base sector, multi-
/// scale) and at each sector size (at the base colony, multi-scale and
/// sector-only).
pub fn run_memory_scaling(
    base: &MemoryProbeConfig,
    sizes: &[usize],
    sectors: &[usize],
) -> Vec<MemoryScaleOutcome> {
    let mut out = Vec::new();
    let one = |cfg: MemoryProbeConfig| {
        let probe = run_memory_probe(&cfg);
        let residual = probe.capacities[1].clone();
        MemoryScaleOutcome {
            ants: cfg.ants,
            sector: cfg.history.sector,
            multiscale: cfg.history.multiscale,
            lag1: residual.by_lag.first().copied().unwrap_or(0.0),
            residual,
            straightness_correlation: probe.straightness_correlation,
        }
    };
    for &ants in sizes {
        out.push(one(MemoryProbeConfig {
            ants,
            ..base.clone()
        }));
    }
    for &sector in sectors {
        for multiscale in [true, false] {
            out.push(one(MemoryProbeConfig {
                history: HistoryConfig {
                    sector,
                    multiscale,
                    ..base.history.clone()
                },
                ..base.clone()
            }));
        }
    }
    out
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

/// The hive memory probe: a colony kept foraging for hours, whose queen
/// writes a fresh random thought into an entropy dial each epoch.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryProbeConfig {
    /// The species.
    pub species: Species,
    /// Simulated workers.
    pub ants: usize,
    /// Epochs recorded after the warm-up.
    pub epochs: usize,
    /// Length of an epoch, seconds.
    pub epoch_s: f64,
    /// Time the colony is given to settle into its foraging before the
    /// queen's clock starts, seconds.
    pub warmup_s: f64,
    /// Size of the queen's expression: a thought `x` sets a temperature of
    /// `exp(expression × x)` at the root.
    pub expression: f64,
    /// Nodes whose dials she writes; empty means the castes.
    pub targets: Vec<NodeId>,
    /// How the movement history is kept (its fast half-life is set to an
    /// epoch when zero).
    pub history: HistoryConfig,
    /// Lags the readouts are fitted for.
    pub lags: usize,
    /// Ridge penalty of the readouts, relative to the number of epochs.
    pub ridge: f64,
    /// Fraction of the epochs the readouts are fitted on; the rest score
    /// them.
    pub train_fraction: f64,
    /// Steady drain on the nest's reserve per simulated worker,
    /// milligrams of sugar per hour: the growth of a large brood and the
    /// metabolism of the nestmates a simulated worker stands for, which
    /// keeps the foragers trafficking.
    pub drain_mg_per_ant_per_h: f64,
    /// Renewal of every food cell, microlitres per second (honeydew that
    /// keeps flowing).
    pub renewal_ul_per_s: f64,
    /// Seed of the map and of the run.
    pub seed: u64,
    /// Whether the colony keeps the behavioural memo too, read at the
    /// history's sector grain, so that its capacity is measured as well.
    pub memo: bool,
}

impl Default for MemoryProbeConfig {
    fn default() -> Self {
        MemoryProbeConfig {
            species: Species::lasius_niger(),
            ants: 100,
            epochs: 180,
            epoch_s: 60.0,
            warmup_s: 45.0 * 60.0,
            expression: 2.0,
            targets: vec![0],
            history: HistoryConfig {
                sector: 16,
                fast_half_life_s: 0.0,
                ..HistoryConfig::default()
            },
            lags: 6,
            ridge: 3.0,
            train_fraction: 2.0 / 3.0,
            drain_mg_per_ant_per_h: 0.5,
            renewal_ul_per_s: 0.02,
            seed: 7,
            memo: true,
        }
    }
}

/// A colony that keeps foraging for hours: renewing sources, a steady
/// drain on the reserve, no mortality, and the movement history kept.
pub fn sustained_colony(cfg: &MemoryProbeConfig) -> SimConfig {
    let mut species = cfg.species.clone();
    species.consumption_mg_per_ant_per_s = cfg.drain_mg_per_ant_per_h / 3600.0;
    let world = WorldConfig {
        seed: Some(cfg.seed),
        ..WorldConfig::default()
    };
    let mut sim = experiment_config(species, world, cfg.ants);
    if let Some(food) = sim.world.random_food.as_mut() {
        food.renewal_ul_per_s = cfg.renewal_ul_per_s;
    }
    for source in sim.world.food_sources.iter_mut() {
        source.renewal_ul_per_s = cfg.renewal_ul_per_s;
    }
    sim.nest.initial_satiation = 0.1;
    sim.nest.mortality = false;
    let mut history = cfg.history.clone();
    if history.fast_half_life_s <= 0.0 {
        history.fast_half_life_s = cfg.epoch_s;
    }
    sim.memo = if cfg.memo {
        Some(MemoConfig {
            slow_half_life_s: history.slow_half_life_s,
            fast_half_life_s: history.fast_half_life_s,
            grain: history.sector,
            transits: None,
        })
    } else {
        None
    };
    sim.history = Some(history);
    sim
}

/// What the memory probe found.
pub struct MemoryProbe {
    /// Epochs recorded.
    pub epochs: usize,
    /// Memory capacity of the invariant, the residual and both
    /// components, then of the behavioural memo when the colony keeps one.
    pub capacities: Vec<Capacity>,
    /// Correlation, over epochs, of the thought with the colony's decision
    /// entropy: whether the dial took effect.
    pub entropy_correlation: f64,
    /// Correlation of the thought with the straightness of the current
    /// flow over the whole field, minus the invariant one: whether the
    /// thought reached the non-invariant movement.
    pub straightness_correlation: f64,
    /// Mean current density over the epochs, moves per tick.
    pub mean_density: f64,
    /// The field at the end.
    pub summary: FieldSummary,
    /// The colony as it stands, its queen's epochs recorded, for the
    /// closed loop.
    pub simulation: Simulation,
}

/// Run the memory probe: warm the colony up, let the queen think a random
/// ±1 per target per epoch, and measure how much of her past thought the
/// field carries.
pub fn run_memory_probe(cfg: &MemoryProbeConfig) -> MemoryProbe {
    let mut sim_cfg = sustained_colony(cfg);
    let tick_s = sim_cfg.world.tick_s.max(1e-9);
    sim_cfg.mind = Some(QueenConfig {
        epoch_ticks: (cfg.epoch_s / tick_s).round().max(1.0) as u64,
        warmup_ticks: (cfg.warmup_s / tick_s).round().max(0.0) as u64,
        targets: cfg.targets.clone(),
        expression: cfg.expression,
        random_input: true,
        ridge: cfg.ridge,
        ..QueenConfig::default()
    });
    let mut sim = Simulation::new(sim_cfg, cfg.seed);
    sim.run_seconds(cfg.warmup_s);
    let mut thoughts = Vec::with_capacity(cfg.epochs);
    let mut entropies = Vec::with_capacity(cfg.epochs);
    let mut straightness = Vec::with_capacity(cfg.epochs);
    let mut density = 0.0;
    for _ in 0..cfg.epochs {
        sim.reset_stats();
        sim.run_seconds(cfg.epoch_s);
        let (Some(queen), Some(history)) = (sim.queen(), sim.history()) else {
            break;
        };
        let Some(epoch) = queen.epochs.last() else {
            break;
        };
        thoughts.push(epoch.thought.first().copied().unwrap_or(0.0));
        entropies.push(sim.stats().mean_entropy());
        let current = history.current_whole();
        straightness.push(current.straightness() - history.invariant_whole().straightness());
        density += current.weight;
    }
    let epochs = thoughts.len();
    let queen = sim.queen().expect("the probe has a queen");
    let mut components = vec![Component::Invariant, Component::Residual, Component::Both];
    if cfg.memo {
        components.push(Component::Memo);
    }
    let capacities = components
        .into_iter()
        .map(|c| memory_capacity(&queen.epochs, c, cfg.lags, cfg.ridge, cfg.train_fraction))
        .collect();
    let summary = sim.history().expect("the probe keeps a history").summary();
    MemoryProbe {
        epochs,
        capacities,
        entropy_correlation: correlation(&thoughts, &entropies),
        straightness_correlation: correlation(&thoughts, &straightness),
        mean_density: if epochs > 0 {
            density / epochs as f64
        } else {
            0.0
        },
        summary,
        simulation: sim,
    }
}

/// The closed loop: the queen recalls her last thought from the field and
/// thinks its opposite, so an alternation is sustained only through the
/// colony's movement.
#[derive(Clone, Debug, PartialEq)]
pub struct ClosedLoop {
    /// Held-out R² of the lag-1 readout the recall uses.
    pub lag1_r2: f64,
    /// The thought of the first target after each epoch.
    pub thoughts: Vec<f64>,
    /// Sign changes between successive thoughts.
    pub alternations: usize,
    /// Mean absolute thought: how decided the queen stayed.
    pub conviction: f64,
}

/// Fit the queen's readouts on her recorded epochs, switch her external
/// input off and recall-driven recursion on, and run `epochs` more
/// epochs. With `expression` zero the dial is disconnected: she still
/// recalls and thinks, but nothing she thinks reaches the colony, so the
/// alternation has nothing to carry it.
pub fn run_closed_loop(
    sim: &mut Simulation,
    epochs: usize,
    recursion: Recursion,
    expression: f64,
    train_fraction: f64,
) -> ClosedLoop {
    let epoch_s = {
        let queen = sim.queen().expect("a queen");
        queen.config().epoch_ticks as f64 * sim.config().world.tick_s
    };
    let lag1_r2 = {
        let queen = sim.queen_mut().expect("a queen");
        let cap = queen.fit(recursion.lag.max(1), train_fraction);
        queen.set_input(Vec::new(), false);
        queen.set_recursion(Some(recursion));
        queen.set_expression(expression);
        cap.by_lag.first().copied().unwrap_or(0.0)
    };
    let mut thoughts = Vec::with_capacity(epochs);
    for _ in 0..epochs {
        sim.run_seconds(epoch_s);
        let queen = sim.queen().expect("a queen");
        thoughts.push(queen.thought.first().copied().unwrap_or(0.0));
    }
    let alternations = thoughts
        .windows(2)
        .filter(|w| (w[0] > 0.0) != (w[1] > 0.0))
        .count();
    let conviction = if thoughts.is_empty() {
        0.0
    } else {
        thoughts.iter().map(|t| t.abs()).sum::<f64>() / thoughts.len() as f64
    };
    ClosedLoop {
        lag1_r2,
        thoughts,
        alternations,
        conviction,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::world::World;

    #[test]
    fn bridge_geometry_is_connected_and_counted() {
        let spec = BridgeSpec::ratio_two();
        assert_eq!(spec.short(), 8);
        assert_eq!(spec.long(), 16);
        assert_eq!(BridgeSpec::equal().short(), BridgeSpec::equal().long());
        let cfg = double_bridge(&spec);
        let world = World::new(cfg.clone(), &mut Rng::seed_from_u64(1));
        assert_eq!(world.counters().len(), 2);
        assert!(world.total_food() > 0.0);
        // Flood fill from the nest reaches the food and both counters.
        let mut seen = vec![false; world.width() * world.height()];
        let mut stack = vec![world.nest()];
        while let Some(p) = stack.pop() {
            let Some(idx) = world.index(p) else { continue };
            if seen[idx] || !world.is_passable(p) {
                continue;
            }
            seen[idx] = true;
            for (_, q) in world.passable_neighbors(p) {
                stack.push(q);
            }
        }
        let reached = |p: Position| seen[world.index(p).unwrap()];
        assert!(reached(cfg.food_sources[0].center));
        for c in &cfg.counters {
            assert!(reached(c.rect.min), "{} unreachable", c.name);
        }
        let equal = double_bridge(&BridgeSpec::equal());
        let world = World::new(equal.clone(), &mut Rng::seed_from_u64(1));
        assert!(world.is_passable(equal.counters[0].rect.min));
        assert!(world.is_passable(equal.counters[1].rect.min));
    }

    #[test]
    fn summary_statistics() {
        let s = summarize(&[0.9, 0.85, 0.3, 0.95]);
        assert_eq!(s.n, 4);
        assert!((s.mean - 0.75).abs() < 1e-12);
        assert!((s.majority - 0.75).abs() < 1e-12);
        assert!((s.decided - 0.75).abs() < 1e-12);
        assert_eq!(summarize(&[]).n, 0);
    }

    #[test]
    fn two_source_world_has_both_sources() {
        let cfg = two_sources(10, 1.0, 0.3);
        let world = World::new(cfg.clone(), &mut Rng::seed_from_u64(1));
        assert_eq!(world.counters().len(), 2);
        let a = &cfg.counters[0].rect;
        let b = &cfg.counters[1].rect;
        assert!(world.food_in(a) > 0.0 && world.food_in(b) > 0.0);
        assert_eq!(
            world.cell(cfg.food_sources[1].center).unwrap().molarity,
            0.3
        );
    }
}
