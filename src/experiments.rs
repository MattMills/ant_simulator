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

use crate::ant::{BASE_FEATURES, F_CROWD, F_ODOUR, F_RECENT, F_ROUTE};
use crate::colony::{BroodItem, BroodStage, SimConfig, Simulation};
use crate::geometry::Position;
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
            .clone()
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
    odour: bool,
    seconds: f64,
    seed: u64,
) -> DiscoveryOutcome {
    let mut species = species;
    if !odour {
        species.food_odour = crate::pheromone::PheromoneParams::inert();
    }
    let cfg = experiment_config(species, hidden_source(14), ants);
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
