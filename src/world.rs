//! The grid world: terrain, food, nest, pheromone fields, and counters.
//!
//! The grid discretises the substrate. Cells are `cell_cm` centimetres
//! across and ticks are `tick_s` seconds; ants move continuously across
//! the grid (see [`crate::geometry::Point`]) while pheromone, food and
//! occupancy live in cells. Each cell carries every [`Pheromone`] channel;
//! the channels evaporate by first-order kinetics from their half-lives
//! (scaled by temperature through [`World::set_evaporation_factor`]) and
//! diffuse conservatively to orthogonal neighbours each tick. Food is a
//! volume of sugar solution of a given molarity.
//!
//! The field is kept at two grains. The grid is tiled by the nodes of
//! its quadtree at one level (`kinetics_grain` cells across), and for
//! every channel a node is either *active*, its cells carrying the
//! field, or *coarse*, one mean concentration standing for all of them.
//! Mass moves between cells, between a cell and a coarse node, and
//! between coarse nodes by the same conservative shares, so the field is
//! exact where it has structure (on and around the trails, where marks
//! land) and cheap where it is faint and flat. A deposit or an inflow
//! that lifts a coarse node above a threshold refines it into cells
//! (coarse to fine); a node whose cells have all faded below half of it
//! is composed into its mean (fine to coarse). [`World::field_tree`]
//! reads the mass at every level above the grain.

use crate::geometry::{Direction, Point, Position};
use crate::pheromone::{Pheromone, PheromoneParams, PheromoneSet};
use crate::quad::{QuadKey, QuadTree};
use crate::rng::Rng;
use crate::species::Species;

/// Half-width of an ant's body in cells (0.2 cm on the default 2-cm grid):
/// paths must keep this clearance from walls.
pub const BODY_RADIUS: f64 = 0.1;

/// The two macronutrients a colony forages for: sugar solutions
/// (honeydew, nectar), which fuel the workers, and protein prey, which the
/// larvae need to grow (Dussutour & Simpson 2009, *Curr. Biol.* 19:740).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Nutrient {
    /// Sucrose solution, drunk into the crop.
    #[default]
    Sugar,
    /// Solid prey, cut and carried in the mandibles.
    Protein,
}

impl Nutrient {
    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Nutrient::Sugar => "sugar",
            Nutrient::Protein => "protein",
        }
    }
}

/// What a cell fundamentally is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Terrain {
    /// Walkable ground.
    #[default]
    Open,
    /// Impassable.
    Wall,
    /// Part of the nest; food delivered here counts.
    Nest,
}

/// One grid cell.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Cell {
    /// Terrain type.
    pub terrain: Terrain,
    /// Volume of food solution lying here, microlitres.
    pub food_ul: f64,
    /// Sucrose molarity of that solution.
    pub molarity: f64,
    /// Standing volume a renewing source refills to, microlitres.
    pub food_capacity_ul: f64,
    /// Renewal rate of the solution here (a drop fed by a syringe, an
    /// aphid colony), microlitres per second; zero for a fixed pool.
    pub renewal_ul_per_s: f64,
    /// Mass of protein prey lying here (a dead insect), milligrams.
    pub prey_mg: f64,
    /// Dead nestmates lying here.
    pub corpses: u16,
    /// Concentration of every pheromone channel, indexed by
    /// [`Pheromone::index`].
    pub pheromone: [f64; Pheromone::COUNT],
    /// Number of living ants currently in the cell (ants inside the nest
    /// are not on the grid).
    pub occupancy: u16,
    /// How many ants the cell comfortably holds: beyond it ants slow down,
    /// push, and steer away.
    pub capacity: u16,
}

impl Cell {
    /// Concentration of one channel.
    pub fn level(&self, kind: Pheromone) -> f64 {
        self.pheromone[kind.index()]
    }

    /// Whether the cell is below capacity.
    pub fn has_room(&self) -> bool {
        self.occupancy < self.capacity
    }

    /// Crowding: occupancy over capacity (1 at capacity, above it when
    /// overfilled), not counting one ant (the observer) when
    /// `excluding_self`.
    pub fn crowding(&self, excluding_self: bool) -> f64 {
        let others = if excluding_self {
            self.occupancy.saturating_sub(1)
        } else {
            self.occupancy
        };
        others as f64 / self.capacity.max(1) as f64
    }

    /// Whether any food remains, solution or prey.
    pub fn has_food(&self) -> bool {
        self.has_solution() || self.has_prey()
    }

    /// Whether any solution remains.
    pub fn has_solution(&self) -> bool {
        self.food_ul > 1e-9
    }

    /// Whether any prey remains.
    pub fn has_prey(&self) -> bool {
        self.prey_mg > 1e-9
    }
}

/// A hand-placed cluster of food.
#[derive(Clone, Debug, PartialEq)]
pub struct FoodSource {
    /// Cluster centre.
    pub center: Position,
    /// Chebyshev radius of the cluster.
    pub radius: i32,
    /// Microlitres of solution placed on every cell of the cluster.
    pub volume_ul_per_cell: f64,
    /// Sucrose molarity of the solution.
    pub molarity: f64,
    /// Rate at which each cell refills towards its placed volume,
    /// microlitres per second (zero: a fixed pool).
    pub renewal_ul_per_s: f64,
    /// Milligrams of protein prey placed on every cell of the cluster.
    pub prey_mg_per_cell: f64,
}

impl FoodSource {
    /// A fixed pool of solution.
    pub fn pool(center: Position, radius: i32, volume_ul_per_cell: f64, molarity: f64) -> Self {
        FoodSource {
            center,
            radius,
            volume_ul_per_cell,
            molarity,
            renewal_ul_per_s: 0.0,
            prey_mg_per_cell: 0.0,
        }
    }

    /// Prey (dead insects) of `mass_mg_per_cell` on every cell.
    pub fn prey(center: Position, radius: i32, mass_mg_per_cell: f64) -> Self {
        FoodSource {
            center,
            radius,
            volume_ul_per_cell: 0.0,
            molarity: 0.0,
            renewal_ul_per_s: 0.0,
            prey_mg_per_cell: mass_mg_per_cell,
        }
    }

    /// The same source refilling at `renewal_ul_per_s` per cell.
    pub fn renewing(mut self, renewal_ul_per_s: f64) -> Self {
        self.renewal_ul_per_s = renewal_ul_per_s.max(0.0);
        self
    }
}

/// An inclusive axis-aligned rectangle of cells.
#[derive(Clone, Debug, PartialEq)]
pub struct Rect {
    /// Top-left corner.
    pub min: Position,
    /// Bottom-right corner (inclusive).
    pub max: Position,
}

impl Rect {
    /// Build a rectangle from two corners in any order.
    pub fn new(a: Position, b: Position) -> Self {
        Rect {
            min: Position::new(a.x.min(b.x), a.y.min(b.y)),
            max: Position::new(a.x.max(b.x), a.y.max(b.y)),
        }
    }

    /// Whether `p` lies inside the rectangle.
    pub fn contains(&self, p: Position) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }
}

/// A region whose cells hold a different number of ants than the rest of
/// the world: a narrow bridge, or a chamber with room for everyone.
#[derive(Clone, Debug, PartialEq)]
pub struct CapacityZone {
    /// The cells concerned.
    pub rect: Rect,
    /// Ants per cell.
    pub capacity: u16,
}

/// Randomly placed food clusters, generated when the world is built.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomFood {
    /// Number of clusters.
    pub clusters: usize,
    /// Chebyshev radius of each cluster.
    pub radius: i32,
    /// Microlitres per cell.
    pub volume_ul_per_cell: f64,
    /// Minimum Chebyshev distance between a cluster centre and the nest.
    pub min_distance_from_nest: i32,
    /// Molarity range the clusters are drawn from.
    pub molarity: (f64, f64),
    /// Renewal rate of every cluster cell, microlitres per second.
    pub renewal_ul_per_s: f64,
    /// Number of prey clusters (dead insects) placed as well.
    pub prey_clusters: usize,
    /// Milligrams of prey on each prey cluster cell.
    pub prey_mg_per_cell: f64,
}

/// A region whose crossings are counted (the "bridge counters" of the
/// double-bridge experiments).
#[derive(Clone, Debug, PartialEq)]
pub struct Counter {
    /// Name used in reports.
    pub name: String,
    /// Cells that count.
    pub rect: Rect,
}

/// World construction parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldConfig {
    /// Grid width in cells.
    pub width: usize,
    /// Grid height in cells.
    pub height: usize,
    /// Edge length of a cell, centimetres.
    pub cell_cm: f64,
    /// Duration of a tick, seconds.
    pub tick_s: f64,
    /// Nest centre.
    pub nest: Position,
    /// Chebyshev radius of the nest.
    pub nest_radius: i32,
    /// Explicit food clusters.
    pub food_sources: Vec<FoodSource>,
    /// Randomly generated food clusters, in addition to `food_sources`.
    pub random_food: Option<RandomFood>,
    /// Wall rectangles.
    pub walls: Vec<Rect>,
    /// Whether everything outside `open` rectangles is wall (for mazes and
    /// bridges). Ignored when empty.
    pub open: Vec<Rect>,
    /// Crossing counters.
    pub counters: Vec<Counter>,
    /// Ants that fit in an ordinary cell (about two per square centimetre
    /// on the default grid: the density at which trail traffic on a
    /// 6-mm bridge begins to push, Dussutour, Fourcassié, Helbing &
    /// Deneubourg 2004, *Nature* 428:70). Nest cells hold any number.
    pub cell_capacity: u16,
    /// Regions with their own capacity.
    pub capacity_zones: Vec<CapacityZone>,
    /// Step the chemical kinetics every this many ticks, with the
    /// evaporation and diffusion of the ticks skipped applied at once
    /// (deposits accumulate meanwhile). One is exact; a few is
    /// indistinguishable for half-lives of minutes and cheaper by as
    /// much on large grids.
    pub kinetics_stride: u32,
    /// Side, in cells, of the nodes at which the chemical field is kept
    /// coarse where it is faint: the grid is tiled by square nodes of
    /// this many cells (rounded up to a power of two, the nodes of the
    /// world's quadtree at one level), and for every channel a node is
    /// either active, its cells carrying the field, or coarse, one mean
    /// concentration standing for all of them. Zero keeps every cell
    /// active.
    pub kinetics_grain: usize,
    /// Concentration, as a fraction of a channel's sensitivity constant
    /// `k`, below which a node of that channel may be coarse: a node
    /// whose cells are all below half of it is composed into its mean,
    /// and a coarse node whose mean reaches it is refined into cells
    /// again.
    pub coarse_below: f64,
    /// The same for the volatile channels (the smell of food, alarm),
    /// which spread through the air over the whole field and are read
    /// at the grain beyond their near field, a coarse node reading as
    /// the interpolation of the means around it so that a smell keeps
    /// its gradient there.
    pub coarse_below_volatile: f64,
    /// Corpses scattered at random over open cells when the world is built
    /// (the arenas of the cemetery-formation experiments).
    pub scattered_corpses: usize,
    /// Conspicuous objects (stones, tufts) that ants can see from a
    /// distance and take views of.
    pub landmarks: Vec<Position>,
    /// Landmarks placed at random over open cells when the world is built,
    /// in addition to `landmarks`.
    pub random_landmarks: usize,
    /// Pheromone kinetics; `None` takes them from the species.
    pub pheromones: Option<PheromoneSet>,
    /// Seed for random food placement. `None` uses the simulation's generator,
    /// `Some` fixes the map independently of everything else.
    pub seed: Option<u64>,
}

impl Default for WorldConfig {
    fn default() -> Self {
        WorldConfig {
            width: 64,
            height: 40,
            cell_cm: 2.0,
            tick_s: 1.0,
            nest: Position::new(32, 20),
            nest_radius: 3,
            food_sources: Vec::new(),
            random_food: Some(RandomFood {
                clusters: 3,
                radius: 2,
                volume_ul_per_cell: 20.0,
                min_distance_from_nest: 12,
                molarity: (0.4, 1.0),
                renewal_ul_per_s: 0.0,
                prey_clusters: 1,
                prey_mg_per_cell: 30.0,
            }),
            walls: Vec::new(),
            open: Vec::new(),
            counters: Vec::new(),
            cell_capacity: 8,
            kinetics_stride: 1,
            kinetics_grain: 8,
            coarse_below: 0.01,
            coarse_below_volatile: 2.0,
            capacity_zones: Vec::new(),
            scattered_corpses: 0,
            landmarks: Vec::new(),
            random_landmarks: 0,
            pheromones: None,
            seed: None,
        }
    }
}

/// A counter's running total.
#[derive(Clone, Debug, PartialEq)]
pub struct CounterState {
    /// The counter's definition.
    pub counter: Counter,
    /// Ant cell entries inside the region.
    pub crossings: u64,
}

/// How often, in sweeps of the kinetics, active nodes are checked for
/// coarsening.
const COARSEN_EVERY: u32 = 16;

/// The chemical field's two grains: the grid tiled by square nodes, and
/// for every channel each node either active (its cells carry the
/// field) or coarse (one mean stands for all of them).
#[derive(Clone, Debug)]
struct Grain {
    /// Log2 of the node side.
    shift: u32,
    cols: usize,
    rows: usize,
    width: usize,
    height: usize,
    /// Open cells per node.
    open: Vec<u32>,
    /// Nodes with a wall in them, which are always active.
    walled: Vec<bool>,
    /// Per node and channel, whether the node is active.
    active: Vec<[bool; Pheromone::COUNT]>,
    /// Per node and channel, the mean concentration of a coarse node.
    mean: Vec<[f64; Pheromone::COUNT]>,
    /// Per node and channel, the mean concentration of every node as of
    /// the last sweep, active or coarse.
    avg: Vec<[f64; Pheromone::COUNT]>,
    /// Per node and channel, the gradient of the mean across a coarse
    /// node's neighbours, per cell, as of the last sweep (volatiles
    /// only): a coarse node reads as a plane, so a smell keeps its
    /// gradient at the grain.
    grad: Vec<[(f64, f64); Pheromone::COUNT]>,
    /// Sweep scratch: the mass flowing into a coarse node, its own mass
    /// after evaporation and outflow, and whether the front of the field
    /// has reached it (an active cell at its door above the threshold).
    inflow: Vec<f64>,
    pool: Vec<f64>,
    touched: Vec<bool>,
    /// Per channel, the concentration below which a node may be coarse.
    threshold: [f64; Pheromone::COUNT],
    /// Per channel, whether it spreads through the air and is read at
    /// the grain by interpolation where coarse.
    volatile: [bool; Pheromone::COUNT],
    since_coarsen: u32,
    /// Corpses per node, for counts over regions.
    corpses: Vec<u32>,
}

impl Grain {
    fn build(config: &WorldConfig, cells: &[Cell], params: &PheromoneSet) -> Option<Grain> {
        if config.kinetics_grain == 0 {
            return None;
        }
        let root = config.width.max(config.height).max(1);
        let mut shift = 0u32;
        while (1usize << shift) < config.kinetics_grain && (1usize << shift) < root {
            shift += 1;
        }
        let side = 1usize << shift;
        let cols = config.width.div_ceil(side);
        let rows = config.height.div_ceil(side);
        let n = cols * rows;
        let mut g = Grain {
            shift,
            cols,
            rows,
            width: config.width,
            height: config.height,
            open: vec![0; n],
            walled: vec![false; n],
            active: vec![[false; Pheromone::COUNT]; n],
            mean: vec![[0.0; Pheromone::COUNT]; n],
            avg: vec![[0.0; Pheromone::COUNT]; n],
            grad: vec![[(0.0, 0.0); Pheromone::COUNT]; n],
            inflow: vec![0.0; n],
            pool: vec![0.0; n],
            touched: vec![false; n],
            threshold: [0.0; Pheromone::COUNT],
            volatile: [false; Pheromone::COUNT],
            since_coarsen: 0,
            corpses: vec![0; n],
        };
        for (k, p) in params.iter().enumerate() {
            let volatile = Pheromone::ALL[k].is_volatile();
            let below = if volatile {
                config.coarse_below_volatile
            } else {
                config.coarse_below
            };
            g.threshold[k] = below.max(0.0) * p.k.max(1e-9);
            g.volatile[k] = volatile;
        }
        for node in 0..n {
            let (x0, y0, x1, y1) = g.rect(node);
            let mut open = 0;
            let mut walled = false;
            let mut corpses = 0;
            for y in y0..y1 {
                for x in x0..x1 {
                    let c = &cells[y * config.width + x];
                    if c.terrain == Terrain::Wall {
                        walled = true;
                    } else {
                        open += 1;
                    }
                    corpses += c.corpses as u32;
                }
            }
            g.open[node] = open;
            g.walled[node] = walled;
            g.corpses[node] = corpses;
            if walled {
                g.active[node] = [true; Pheromone::COUNT];
            }
        }
        Some(g)
    }

    #[inline]
    fn node_of(&self, x: usize, y: usize) -> usize {
        (y >> self.shift) * self.cols + (x >> self.shift)
    }

    /// The cells of a node, `(x0, y0, x1, y1)` with exclusive far
    /// corners, clipped to the grid.
    fn rect(&self, node: usize) -> (usize, usize, usize, usize) {
        let side = 1usize << self.shift;
        let x0 = (node % self.cols) * side;
        let y0 = (node / self.cols) * side;
        (
            x0,
            y0,
            (x0 + side).min(self.width),
            (y0 + side).min(self.height),
        )
    }

    /// The node across a side (0 north, 1 east, 2 south, 3 west).
    fn neighbour(&self, node: usize, side: u8) -> Option<usize> {
        let col = node % self.cols;
        let row = node / self.cols;
        match side {
            0 => (row > 0).then(|| node - self.cols),
            1 => (col + 1 < self.cols).then_some(node + 1),
            2 => (row + 1 < self.rows).then(|| node + self.cols),
            _ => (col > 0).then(|| node - 1),
        }
    }

    /// Coarse to fine: the node's cells take its mean and carry the
    /// field from now on.
    fn refine(&mut self, k: usize, node: usize, cells: &mut [Cell], width: usize) {
        if self.active[node][k] {
            return;
        }
        let m = self.mean[node][k];
        let (x0, y0, x1, y1) = self.rect(node);
        for y in y0..y1 {
            for x in x0..x1 {
                let c = &mut cells[y * width + x];
                if c.terrain != Terrain::Wall {
                    c.pheromone[k] = m;
                }
            }
        }
        self.active[node][k] = true;
        self.mean[node][k] = 0.0;
    }

    /// Fine to coarse: the node's cells are composed into one mean.
    fn coarsen(&mut self, k: usize, node: usize, cells: &mut [Cell], width: usize) {
        let (x0, y0, x1, y1) = self.rect(node);
        let mut sum = 0.0;
        for y in y0..y1 {
            for x in x0..x1 {
                let c = &mut cells[y * width + x];
                sum += c.pheromone[k];
                c.pheromone[k] = 0.0;
            }
        }
        let open = self.open[node] as f64;
        let mean = if open > 0.0 { sum / open } else { 0.0 };
        self.mean[node][k] = mean;
        self.avg[node][k] = mean;
        self.active[node][k] = false;
    }

    /// The gradients of a volatile channel across the coarse nodes,
    /// from the means of their neighbours (one-sided at the grid's
    /// edge), per cell.
    fn gradients(&mut self, k: usize) {
        let side = (1usize << self.shift) as f64;
        for node in 0..self.cols * self.rows {
            if self.active[node][k] {
                self.grad[node][k] = (0.0, 0.0);
                continue;
            }
            let c = self.avg[node][k];
            let along = |a: Option<usize>, b: Option<usize>, g: &Grain| -> f64 {
                match (a, b) {
                    (Some(a), Some(b)) => (g.avg[b][k] - g.avg[a][k]) / (2.0 * side),
                    (None, Some(b)) => (g.avg[b][k] - c) / side,
                    (Some(a), None) => (c - g.avg[a][k]) / side,
                    (None, None) => 0.0,
                }
            };
            let gx = along(self.neighbour(node, 3), self.neighbour(node, 1), self);
            let gy = along(self.neighbour(node, 0), self.neighbour(node, 2), self);
            self.grad[node][k] = (gx, gy);
        }
    }
}

/// The simulated environment.
#[derive(Clone, Debug)]
pub struct World {
    config: WorldConfig,
    cells: Vec<Cell>,
    scratch: Vec<[f64; Pheromone::COUNT]>,
    /// The field's two grains, when the kinetics grain is set.
    grain: Option<Grain>,
    /// Wall cells per node of the world's quadtree, when there are any.
    wall_tree: Option<QuadTree<u32>>,
    /// Whether any cell carries the channel (empty channels are skipped
    /// by the kinetics and by perception).
    present: [bool; Pheromone::COUNT],
    /// Whether the grid has any wall (open worlds skip clearance sweeps).
    has_walls: bool,
    /// Calls to the kinetics so far, for the stride.
    kinetics_calls: u64,
    /// Indices of the cells that carry or renew food, the only ones the
    /// food kinetics visit.
    food_cells: Vec<usize>,
    landmarks: Vec<Position>,
    params: PheromoneSet,
    base_retention: [f64; Pheromone::COUNT],
    retention: [f64; Pheromone::COUNT],
    diffusion: [f64; Pheromone::COUNT],
    evaporation_factor: f64,
    counters: Vec<CounterState>,
}

impl World {
    /// Build a world. Random food clusters are drawn from `rng` unless the
    /// config carries its own seed. Pheromone kinetics default to those of
    /// [`Species::lasius_niger`] when the config has none.
    pub fn new(config: WorldConfig, rng: &mut Rng) -> World {
        assert!(
            config.width > 0 && config.height > 0,
            "world must be non-empty"
        );
        assert!(
            config.tick_s > 0.0 && config.cell_cm > 0.0,
            "units must be positive"
        );
        let params = config
            .pheromones
            .clone()
            .unwrap_or_else(|| Species::lasius_niger().pheromones());
        let mut retention = [0.0; Pheromone::COUNT];
        let mut diffusion = [0.0; Pheromone::COUNT];
        for (i, p) in params.iter().enumerate() {
            retention[i] = p.retention_per_tick(config.tick_s);
            diffusion[i] = p.diffusion_per_tick(config.tick_s);
        }
        let n = config.width * config.height;
        let counters = config
            .counters
            .iter()
            .map(|c| CounterState {
                counter: c.clone(),
                crossings: 0,
            })
            .collect();
        let mut world = World {
            cells: vec![Cell::default(); n],
            scratch: vec![[0.0; Pheromone::COUNT]; n],
            grain: None,
            wall_tree: None,
            present: [false; Pheromone::COUNT],
            has_walls: false,
            kinetics_calls: 0,
            food_cells: Vec::new(),
            landmarks: config.landmarks.clone(),
            params,
            base_retention: retention,
            retention,
            diffusion,
            evaporation_factor: 1.0,
            counters,
            config,
        };
        world.lay_terrain();
        world.has_walls = world.cells.iter().any(|c| c.terrain == Terrain::Wall);
        if world.has_walls {
            let mut tree = QuadTree::<u32>::new(world.config.width, world.config.height);
            let w = world.config.width;
            for (idx, c) in world.cells.iter().enumerate() {
                if c.terrain == Terrain::Wall {
                    let p = Position::new((idx % w) as i32, (idx / w) as i32);
                    if let Some(key) = tree.leaf(p) {
                        *tree.get_mut(key) = 1;
                    }
                }
            }
            tree.compose(|a, b| *a += *b);
            world.wall_tree = Some(tree);
        }
        let sources = world.config.food_sources.clone();
        for src in &sources {
            world.place_food(src);
        }
        if let Some(random) = world.config.random_food.clone() {
            let mut local;
            let r: &mut Rng = match world.config.seed {
                Some(seed) => {
                    local = Rng::seed_from_u64(seed);
                    &mut local
                }
                None => rng,
            };
            world.place_random_food(&random, r);
        }
        if world.config.scattered_corpses > 0 {
            let mut local;
            let r: &mut Rng = match world.config.seed {
                Some(seed) => {
                    local = Rng::seed_from_u64(seed ^ 0x9e37_79b9_7f4a_7c15);
                    &mut local
                }
                None => rng,
            };
            world.scatter_corpses(world.config.scattered_corpses, r);
        }
        if world.config.random_landmarks > 0 {
            let mut local;
            let r: &mut Rng = match world.config.seed {
                Some(seed) => {
                    local = Rng::seed_from_u64(seed ^ 0x2545_f491_4f6c_dd1d);
                    &mut local
                }
                None => rng,
            };
            let w = world.config.width as i32;
            let h = world.config.height as i32;
            let mut placed = 0;
            for _attempt in 0..world.config.random_landmarks * 50 {
                if placed >= world.config.random_landmarks {
                    break;
                }
                let p = Position::new(r.below(w as usize) as i32, r.below(h as usize) as i32);
                let open = world
                    .cell(p)
                    .map(|c| c.terrain == Terrain::Open)
                    .unwrap_or(false);
                if open && !world.landmarks.contains(&p) {
                    world.landmarks.push(p);
                    placed += 1;
                }
            }
        }
        world.grain = Grain::build(&world.config, &world.cells, &world.params);
        world
    }

    /// The landmarks of this world.
    pub fn landmarks(&self) -> &[Position] {
        &self.landmarks
    }

    /// The nearest landmark within `sight` cells of a point, with its
    /// index.
    pub fn nearest_landmark(&self, from: Point, sight: f64) -> Option<(usize, Position)> {
        let mut best: Option<(usize, Position, f64)> = None;
        for (i, &l) in self.landmarks.iter().enumerate() {
            let d = from.distance(Point::center_of(l));
            if d <= sight && best.map(|b| d < b.2).unwrap_or(true) {
                best = Some((i, l, d));
            }
        }
        best.map(|(i, l, _)| (i, l))
    }

    fn scatter_corpses(&mut self, count: usize, rng: &mut Rng) {
        let w = self.config.width as i32;
        let h = self.config.height as i32;
        let mut placed = 0;
        for _attempt in 0..count * 50 {
            if placed >= count {
                break;
            }
            let p = Position::new(rng.below(w as usize) as i32, rng.below(h as usize) as i32);
            if self
                .cell(p)
                .map(|c| c.terrain == Terrain::Open)
                .unwrap_or(false)
            {
                self.add_corpse(p);
                placed += 1;
            }
        }
    }

    fn lay_terrain(&mut self) {
        let walls = self.config.walls.clone();
        let open = self.config.open.clone();
        for y in 0..self.config.height as i32 {
            for x in 0..self.config.width as i32 {
                let p = Position::new(x, y);
                let closed = (!open.is_empty() && !open.iter().any(|o| o.contains(p)))
                    || walls.iter().any(|w| w.contains(p));
                if closed {
                    self.cell_mut(p).unwrap().terrain = Terrain::Wall;
                }
            }
        }
        let nest = self.config.nest;
        let r = self.config.nest_radius;
        for dy in -r..=r {
            for dx in -r..=r {
                if let Some(c) = self.cell_mut(nest.offset(dx, dy)) {
                    c.terrain = Terrain::Nest;
                }
            }
        }
        let default_capacity = self.config.cell_capacity;
        let zones = self.config.capacity_zones.clone();
        for y in 0..self.config.height as i32 {
            for x in 0..self.config.width as i32 {
                let p = Position::new(x, y);
                let capacity = if self.is_nest(p) {
                    u16::MAX
                } else {
                    zones
                        .iter()
                        .rev()
                        .find(|z| z.rect.contains(p))
                        .map(|z| z.capacity)
                        .unwrap_or(default_capacity)
                };
                self.cell_mut(p).unwrap().capacity = capacity;
            }
        }
    }

    fn place_food(&mut self, src: &FoodSource) {
        for dy in -src.radius..=src.radius {
            for dx in -src.radius..=src.radius {
                if let Some(c) = self.cell_mut(src.center.offset(dx, dy)) {
                    if c.terrain == Terrain::Open {
                        if src.volume_ul_per_cell > 0.0 {
                            // Mixing: the molarity becomes the volume-weighted mean.
                            let total = c.food_ul + src.volume_ul_per_cell;
                            c.molarity = (c.molarity * c.food_ul
                                + src.molarity.max(0.0) * src.volume_ul_per_cell)
                                / total;
                            c.food_ul = total;
                            c.food_capacity_ul = c.food_capacity_ul.max(total);
                            c.renewal_ul_per_s = c.renewal_ul_per_s.max(src.renewal_ul_per_s);
                        }
                        c.prey_mg += src.prey_mg_per_cell.max(0.0);
                    }
                }
            }
        }
        self.refresh_food_cells();
    }

    /// Recollect the cells that carry or renew food.
    fn refresh_food_cells(&mut self) {
        self.food_cells = self
            .cells
            .iter()
            .enumerate()
            .filter(|(_, c)| c.has_food() || c.renewal_ul_per_s > 0.0 || c.food_capacity_ul > 0.0)
            .map(|(i, _)| i)
            .collect();
    }

    /// Whether any cell carries the channel.
    pub fn channel_present(&self, kind: Pheromone) -> bool {
        self.present[kind.index()]
    }

    /// Whether the grid has any wall.
    pub fn has_walls(&self) -> bool {
        self.has_walls
    }

    /// Refill renewing sources by one tick and let food give off its
    /// smell: `odour_per_ul_s` per microlitre of solution (the first five
    /// count, a drop's surface being what evaporates) and `odour_per_mg_s`
    /// per milligram of prey (the first thirty), into the odour channel.
    pub fn step_food(&mut self, odour_per_ul_s: f64, odour_per_mg_s: f64) {
        let tick_s = self.config.tick_s;
        let cap = self.params[Pheromone::Odour.index()].cap;
        let active = self.retention[Pheromone::Odour.index()] > 0.0;
        let mut emitted = false;
        let width = self.config.width;
        let food_cells = std::mem::take(&mut self.food_cells);
        for &i in &food_cells {
            let c = &mut self.cells[i];
            if c.renewal_ul_per_s > 0.0 && c.food_ul < c.food_capacity_ul {
                c.food_ul = (c.food_ul + c.renewal_ul_per_s * tick_s).min(c.food_capacity_ul);
            }
            if active && c.has_food() {
                let emission =
                    odour_per_ul_s * c.food_ul.min(5.0) + odour_per_mg_s * c.prey_mg.min(30.0);
                if emission > 0.0 {
                    let p = Position::new((i % width) as i32, (i / width) as i32);
                    self.refine_at(Pheromone::Odour.index(), p);
                    let slot = &mut self.cells[i].pheromone[Pheromone::Odour.index()];
                    *slot = (*slot + emission * tick_s).min(cap);
                    emitted = true;
                }
            }
        }
        self.food_cells = food_cells;
        if emitted {
            self.present[Pheromone::Odour.index()] = true;
        }
    }

    fn place_random_food(&mut self, random: &RandomFood, rng: &mut Rng) {
        let w = self.config.width as i32;
        let h = self.config.height as i32;
        let r = random.radius;
        for _ in 0..random.clusters {
            for _attempt in 0..200 {
                let x = r + rng.below((w - 2 * r).max(1) as usize) as i32;
                let y = r + rng.below((h - 2 * r).max(1) as usize) as i32;
                let center = Position::new(x, y);
                if center.chebyshev(self.config.nest) < random.min_distance_from_nest {
                    continue;
                }
                if self
                    .cell(center)
                    .map(|c| c.terrain != Terrain::Open || c.has_food())
                    .unwrap_or(true)
                {
                    continue;
                }
                let molarity = rng.range(random.molarity.0, random.molarity.1);
                self.place_food(&FoodSource {
                    center,
                    radius: r,
                    volume_ul_per_cell: random.volume_ul_per_cell,
                    molarity,
                    renewal_ul_per_s: random.renewal_ul_per_s,
                    prey_mg_per_cell: 0.0,
                });
                break;
            }
        }
        for _ in 0..random.prey_clusters {
            for _attempt in 0..200 {
                let x = rng.below(w.max(1) as usize) as i32;
                let y = rng.below(h.max(1) as usize) as i32;
                let center = Position::new(x, y);
                if center.chebyshev(self.config.nest) < random.min_distance_from_nest {
                    continue;
                }
                if self
                    .cell(center)
                    .map(|c| c.terrain != Terrain::Open || c.has_food())
                    .unwrap_or(true)
                {
                    continue;
                }
                self.place_food(&FoodSource::prey(center, 0, random.prey_mg_per_cell));
                break;
            }
        }
    }

    /// Construction parameters.
    pub fn config(&self) -> &WorldConfig {
        &self.config
    }

    /// Pheromone kinetics in use.
    pub fn pheromone_params(&self) -> &PheromoneSet {
        &self.params
    }

    /// Parameters of one channel.
    pub fn channel(&self, kind: Pheromone) -> &PheromoneParams {
        &self.params[kind.index()]
    }

    /// Scale every channel's decay rate by `factor` (temperature: warmer
    /// substrates lose pheromone faster). `1.0` restores the nominal
    /// half-lives.
    pub fn set_evaporation_factor(&mut self, factor: f64) {
        let factor = factor.max(0.0);
        if (factor - self.evaporation_factor).abs() < 1e-12 {
            return;
        }
        self.evaporation_factor = factor;
        for (r, base) in self.retention.iter_mut().zip(&self.base_retention) {
            *r = if *base <= 0.0 { 0.0 } else { base.powf(factor) };
        }
    }

    /// Current evaporation factor.
    pub fn evaporation_factor(&self) -> f64 {
        self.evaporation_factor
    }

    /// Grid width.
    pub fn width(&self) -> usize {
        self.config.width
    }

    /// Grid height.
    pub fn height(&self) -> usize {
        self.config.height
    }

    /// Nest centre.
    pub fn nest(&self) -> Position {
        self.config.nest
    }

    /// Chebyshev radius of the nest.
    pub fn nest_radius(&self) -> i32 {
        self.config.nest_radius
    }

    /// Duration of a tick in seconds.
    pub fn tick_s(&self) -> f64 {
        self.config.tick_s
    }

    /// Edge of a cell in centimetres.
    pub fn cell_cm(&self) -> f64 {
        self.config.cell_cm
    }

    /// Linear index of a position, if in bounds.
    pub fn index(&self, p: Position) -> Option<usize> {
        if p.x < 0 || p.y < 0 || p.x >= self.config.width as i32 || p.y >= self.config.height as i32
        {
            None
        } else {
            Some(p.y as usize * self.config.width + p.x as usize)
        }
    }

    /// Whether a position is inside the grid.
    pub fn in_bounds(&self, p: Position) -> bool {
        self.index(p).is_some()
    }

    /// Cell at a position, if in bounds.
    pub fn cell(&self, p: Position) -> Option<&Cell> {
        self.index(p).map(|i| &self.cells[i])
    }

    /// Mutable cell at a position, if in bounds.
    pub fn cell_mut(&mut self, p: Position) -> Option<&mut Cell> {
        match self.index(p) {
            Some(i) => Some(&mut self.cells[i]),
            None => None,
        }
    }

    /// All cells in row-major order.
    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    /// Whether an ant may stand on `p`.
    pub fn is_passable(&self, p: Position) -> bool {
        self.cell(p)
            .map(|c| c.terrain != Terrain::Wall)
            .unwrap_or(false)
    }

    /// Whether an ant may stand at a continuous point.
    pub fn is_passable_point(&self, p: Point) -> bool {
        self.is_passable(p.cell())
    }

    /// Probe along a ray from `from` in direction `heading` for up to
    /// `reach` cells, as an antenna does: the probe stops at the first
    /// wall. Returns the farthest passable point reached and its distance
    /// (zero, at `from`, when the very first quarter cell is blocked).
    pub fn probe(&self, from: Point, heading: f64, reach: f64) -> (Point, f64) {
        let samples = (reach / 0.25).ceil().max(1.0) as usize;
        let mut last = (from, 0.0);
        for i in 1..=samples {
            let d = reach * i as f64 / samples as f64;
            let p = from.advanced(heading, d);
            if !self.is_passable_point(p) {
                break;
            }
            last = (p, d);
        }
        last
    }

    /// Whether a body centred on `p` fits on passable ground: the point and
    /// the corners of a square of half-width [`BODY_RADIUS`] around it.
    pub fn has_clearance(&self, p: Point) -> bool {
        const R: f64 = BODY_RADIUS;
        self.is_passable_point(p)
            && self.is_passable_point(Point::new(p.x - R, p.y - R))
            && self.is_passable_point(Point::new(p.x + R, p.y - R))
            && self.is_passable_point(Point::new(p.x - R, p.y + R))
            && self.is_passable_point(Point::new(p.x + R, p.y + R))
    }

    /// Whether a body can travel the straight segment from `a` to `b`:
    /// clearance is checked every quarter cell along it, so walls cannot be
    /// clipped and corners cannot be cut.
    pub fn segment_passable(&self, a: Point, b: Point) -> bool {
        if !self.has_walls {
            // Only the edge of the grid can stop a body in an open world,
            // and the grid is convex: the far end decides.
            return self.has_clearance(b);
        }
        let len = a.distance(b);
        let samples = (len / 0.25).ceil().max(1.0) as usize;
        for i in 1..=samples {
            let t = i as f64 / samples as f64;
            let p = Point::new(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y));
            if !self.has_clearance(p) {
                return false;
            }
        }
        true
    }

    /// Whether `p` is a nest cell.
    pub fn is_nest(&self, p: Position) -> bool {
        self.cell(p)
            .map(|c| c.terrain == Terrain::Nest)
            .unwrap_or(false)
    }

    /// All nest cells.
    pub fn nest_cells(&self) -> Vec<Position> {
        let mut out = Vec::new();
        for y in 0..self.config.height as i32 {
            for x in 0..self.config.width as i32 {
                let p = Position::new(x, y);
                if self.is_nest(p) {
                    out.push(p);
                }
            }
        }
        out
    }

    /// The field of a channel at a cell: the cell's own value in an
    /// active node; in a coarse one, the node's mean for a substrate
    /// mark (faint there by construction), and for a volatile the means
    /// of the nodes around it interpolated to the cell (an active node
    /// in the stencil counting with its cell nearest the reader), so
    /// that a smell keeps its gradient at the grain.
    #[inline]
    fn read(&self, idx: usize, p: Position, k: usize) -> f64 {
        if let Some(g) = &self.grain {
            let node = g.node_of(p.x as usize, p.y as usize);
            if !g.active[node][k] {
                return if g.volatile[k] {
                    self.read_coarse(g, node, p, k)
                } else {
                    g.mean[node][k]
                };
            }
        }
        self.cells[idx].pheromone[k]
    }

    /// The reading of a coarse node at a cell for a volatile: the plane
    /// through the node's mean with the gradient of its neighbours'
    /// means, never below zero.
    #[inline]
    fn read_coarse(&self, g: &Grain, node: usize, p: Position, k: usize) -> f64 {
        let mask = (1usize << g.shift) - 1;
        let half = (1usize << g.shift) as f64 / 2.0;
        let fx = ((p.x as usize) & mask) as f64 + 0.5 - half;
        let fy = ((p.y as usize) & mask) as f64 + 0.5 - half;
        let (gx, gy) = g.grad[node][k];
        (g.mean[node][k] + gx * fx + gy * fy).max(0.0)
    }

    /// Refine the node holding a cell for a channel, if it is coarse.
    fn refine_at(&mut self, k: usize, p: Position) {
        let width = self.config.width;
        if let Some(g) = self.grain.as_mut() {
            let node = g.node_of(p.x as usize, p.y as usize);
            if !g.active[node][k] {
                g.refine(k, node, &mut self.cells, width);
            }
        }
    }

    /// Concentration of a channel at a cell (0 outside the grid).
    pub fn level(&self, p: Position, kind: Pheromone) -> f64 {
        match self.index(p) {
            Some(idx) => self.read(idx, p, kind.index()),
            None => 0.0,
        }
    }

    /// Concentration of a channel on the substrate patch (cell) under a
    /// continuous point; walls and points off the grid read zero. Marks
    /// are laid on the substrate cell by cell, and an antenna touching a
    /// patch senses that patch: at a trail fork this keeps the two branch
    /// trails distinct instead of blending them with the heavily marked
    /// junction (see [`World::sample`] for the smooth field).
    pub fn level_at(&self, p: Point, kind: Pheromone) -> f64 {
        let cell = p.cell();
        match self.index(cell) {
            Some(idx) if self.cells[idx].terrain != Terrain::Wall => {
                self.read(idx, cell, kind.index())
            }
            _ => 0.0,
        }
    }

    /// Concentration of a channel at a continuous point, bilinearly
    /// interpolated between the four nearest cell centres. Walls and cells
    /// off the grid contribute nothing.
    pub fn sample(&self, p: Point, kind: Pheromone) -> f64 {
        let fx = p.x - 0.5;
        let fy = p.y - 0.5;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = fx - x0;
        let ty = fy - y0;
        let mut total = 0.0;
        for (dx, dy, w) in [
            (0, 0, (1.0 - tx) * (1.0 - ty)),
            (1, 0, tx * (1.0 - ty)),
            (0, 1, (1.0 - tx) * ty),
            (1, 1, tx * ty),
        ] {
            if w <= 0.0 {
                continue;
            }
            let cell = Position::new(x0 as i32 + dx, y0 as i32 + dy);
            if let Some(idx) = self.index(cell) {
                if self.cells[idx].terrain != Terrain::Wall {
                    total += w * self.read(idx, cell, kind.index());
                }
            }
        }
        total
    }

    /// Add pheromone of one channel to a cell (clamped to the channel's cap;
    /// inert channels take nothing).
    pub fn deposit(&mut self, p: Position, kind: Pheromone, amount: f64) {
        let k = kind.index();
        let cap = self.params[k].cap;
        if amount <= 0.0 || cap <= 0.0 {
            return;
        }
        let Some(idx) = self.index(p) else {
            return;
        };
        self.refine_at(k, p);
        let v = &mut self.cells[idx].pheromone[k];
        *v = (*v + amount).min(cap);
        self.present[k] = true;
    }

    /// Take up to `volume_ul` of solution from a cell; returns the volume
    /// actually taken and its molarity.
    pub fn take_food(&mut self, p: Position, volume_ul: f64) -> (f64, f64) {
        match self.cell_mut(p) {
            Some(c) if c.has_food() => {
                let taken = volume_ul.max(0.0).min(c.food_ul);
                c.food_ul -= taken;
                let molarity = c.molarity;
                if c.food_ul < 1e-9 {
                    c.food_ul = 0.0;
                }
                (taken, molarity)
            }
            _ => (0.0, 0.0),
        }
    }

    /// Cut up to `mass_mg` of prey from cell `p`; returns the mass taken.
    pub fn take_prey(&mut self, p: Position, mass_mg: f64) -> f64 {
        match self.cell_mut(p) {
            Some(c) if c.has_prey() => {
                let taken = mass_mg.max(0.0).min(c.prey_mg);
                c.prey_mg -= taken;
                if c.prey_mg < 1e-9 {
                    c.prey_mg = 0.0;
                }
                taken
            }
            _ => 0.0,
        }
    }

    /// Record an ant entering cell `p` in every counter covering it.
    pub fn record_crossing(&mut self, p: Position) {
        for c in self.counters.iter_mut() {
            if c.counter.rect.contains(p) {
                c.crossings += 1;
            }
        }
    }

    /// Crossing counters.
    pub fn counters(&self) -> &[CounterState] {
        &self.counters
    }

    /// Crossings of the counter with the given name.
    pub fn crossings(&self, name: &str) -> Option<u64> {
        self.counters
            .iter()
            .find(|c| c.counter.name == name)
            .map(|c| c.crossings)
    }

    /// The passable neighbours of `p` with the direction leading to each.
    pub fn passable_neighbors(
        &self,
        p: Position,
    ) -> impl Iterator<Item = (Direction, Position)> + '_ {
        Direction::ALL.into_iter().filter_map(move |d| {
            let q = p.step(d);
            if self.is_passable(q) {
                Some((d, q))
            } else {
                None
            }
        })
    }

    /// Evaporate and diffuse every pheromone channel by one tick.
    pub fn step_pheromones(&mut self) {
        let w = self.config.width as i32;
        let h = self.config.height as i32;
        self.kinetics_calls += 1;
        let stride = self.config.kinetics_stride.max(1);
        if !self.kinetics_calls.is_multiple_of(stride as u64) {
            return;
        }
        // The ticks skipped are applied at once.
        let retention: Vec<f64> = self
            .retention
            .iter()
            .map(|r| r.powi(stride as i32))
            .collect();
        let diffusion: Vec<f64> = self
            .diffusion
            .iter()
            .map(|d| (d * stride as f64).min(0.9))
            .collect();
        // Only channels that carry something somewhere are stepped; a
        // cell counts as empty below a millionth of a unit.
        let active: Vec<usize> = (0..Pheromone::COUNT)
            .filter(|&k| retention[k] > 0.0 && self.present[k])
            .collect();
        if active.is_empty() {
            return;
        }
        if self.grain.is_some() {
            self.step_multiscale(&active, &retention, &diffusion);
            return;
        }
        for s in self.scratch.iter_mut() {
            for &k in &active {
                s[k] = 0.0;
            }
        }
        for y in 0..h {
            for x in 0..w {
                let idx = y as usize * w as usize + x as usize;
                let cell = &self.cells[idx];
                if cell.terrain == Terrain::Wall {
                    continue;
                }
                for &k in &active {
                    let amount = cell.pheromone[k] * retention[k];
                    if amount <= 0.0 {
                        continue;
                    }
                    let diffusion = diffusion[k];
                    if diffusion > 0.0 {
                        // Conservative diffusion: shares that would cross
                        // into a wall or off the grid stay in the cell.
                        let share = amount * diffusion / 4.0;
                        let mut moved = 0.0;
                        for (dx, dy) in [(0, -1), (1, 0), (0, 1), (-1, 0)] {
                            let nx = x + dx;
                            let ny = y + dy;
                            if nx < 0 || ny < 0 || nx >= w || ny >= h {
                                continue;
                            }
                            let nidx = ny as usize * w as usize + nx as usize;
                            if self.cells[nidx].terrain == Terrain::Wall {
                                continue;
                            }
                            self.scratch[nidx][k] += share;
                            moved += share;
                        }
                        self.scratch[idx][k] += amount - moved;
                    } else {
                        self.scratch[idx][k] += amount;
                    }
                }
            }
        }
        let mut present = [false; Pheromone::COUNT];
        for (cell, s) in self.cells.iter_mut().zip(self.scratch.iter()) {
            for &k in &active {
                let v = s[k];
                let value = if v < 1e-6 {
                    0.0
                } else {
                    v.min(self.params[k].cap)
                };
                cell.pheromone[k] = value;
                if value > 0.0 {
                    present[k] = true;
                }
            }
        }
        for &k in &active {
            self.present[k] = present[k];
        }
    }

    /// The sweep at two grains: active nodes cell by cell, coarse nodes
    /// as one pool each, with every share of mass leaving one place and
    /// arriving in another.
    fn step_multiscale(&mut self, channels: &[usize], retention: &[f64], diffusion: &[f64]) {
        let Some(mut g) = self.grain.take() else {
            return;
        };
        let w = self.config.width;
        let h = self.config.height;
        let nodes = g.cols * g.rows;
        let mut present = [false; Pheromone::COUNT];
        for &k in channels {
            let r = retention[k];
            let d = diffusion[k];
            let cap = self.params[k].cap;
            let threshold = g.threshold[k];
            for node in 0..nodes {
                g.inflow[node] = 0.0;
                g.pool[node] = 0.0;
                g.touched[node] = false;
                if g.active[node][k] {
                    let (x0, y0, x1, y1) = g.rect(node);
                    for y in y0..y1 {
                        for x in x0..x1 {
                            self.scratch[y * w + x][k] = 0.0;
                        }
                    }
                }
            }
            for node in 0..nodes {
                let (x0, y0, x1, y1) = g.rect(node);
                if g.active[node][k] {
                    let walled = g.walled[node];
                    for y in y0..y1 {
                        for x in x0..x1 {
                            let idx = y * w + x;
                            if walled && self.cells[idx].terrain == Terrain::Wall {
                                continue;
                            }
                            let value = self.cells[idx].pheromone[k];
                            let amount = value * r;
                            if amount <= 0.0 {
                                continue;
                            }
                            if d > 0.0 {
                                let share = amount * d / 4.0;
                                // A cell inside an open node has its four
                                // neighbours in the node, all open.
                                if !walled && x > x0 && x + 1 < x1 && y > y0 && y + 1 < y1 {
                                    self.scratch[idx - w][k] += share;
                                    self.scratch[idx + 1][k] += share;
                                    self.scratch[idx + w][k] += share;
                                    self.scratch[idx - 1][k] += share;
                                    self.scratch[idx][k] += amount - 4.0 * share;
                                    continue;
                                }
                                // Conservative diffusion: shares that
                                // would cross into a wall or off the grid
                                // stay in the cell; a share crossing into
                                // a coarse node joins its pool, and a cell
                                // above the threshold at a coarse node's
                                // door brings the front to it.
                                let mut moved = 0.0;
                                for (dx, dy) in [(0i64, -1i64), (1, 0), (0, 1), (-1, 0)] {
                                    let nx = x as i64 + dx;
                                    let ny = y as i64 + dy;
                                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                                        continue;
                                    }
                                    let (nx, ny) = (nx as usize, ny as usize);
                                    let nidx = ny * w + nx;
                                    let nnode = g.node_of(nx, ny);
                                    if g.active[nnode][k] {
                                        if g.walled[nnode]
                                            && self.cells[nidx].terrain == Terrain::Wall
                                        {
                                            continue;
                                        }
                                        self.scratch[nidx][k] += share;
                                    } else {
                                        g.inflow[nnode] += share;
                                        if value >= threshold {
                                            g.touched[nnode] = true;
                                        }
                                    }
                                    moved += share;
                                }
                                self.scratch[idx][k] += amount - moved;
                            } else {
                                self.scratch[idx][k] += amount;
                            }
                        }
                    }
                } else {
                    let mean = g.mean[node][k];
                    if mean <= 0.0 {
                        continue;
                    }
                    let mut pool = mean * g.open[node] as f64 * r;
                    if d > 0.0 {
                        // Each cell along a side sends its share across:
                        // to the neighbouring cells of an active node,
                        // to the pool of a coarse one.
                        let per = mean * r * d / 4.0;
                        for side in 0..4u8 {
                            let Some(nb) = g.neighbour(node, side) else {
                                continue;
                            };
                            if g.active[nb][k] {
                                let (bx0, by0, bx1, by1) = g.rect(nb);
                                let (ax0, ay0, ax1, ay1) = match side {
                                    0 => (x0.max(bx0), by1 - 1, x1.min(bx1), by1),
                                    1 => (bx0, y0.max(by0), bx0 + 1, y1.min(by1)),
                                    2 => (x0.max(bx0), by0, x1.min(bx1), by0 + 1),
                                    _ => (bx1 - 1, y0.max(by0), bx1, y1.min(by1)),
                                };
                                for y in ay0..ay1 {
                                    for x in ax0..ax1 {
                                        let nidx = y * w + x;
                                        if self.cells[nidx].terrain == Terrain::Wall {
                                            continue;
                                        }
                                        self.scratch[nidx][k] += per;
                                        pool -= per;
                                    }
                                }
                            } else {
                                let pairs = match side {
                                    0 | 2 => x1 - x0,
                                    _ => y1 - y0,
                                } as f64;
                                g.inflow[nb] += per * pairs;
                                pool -= per * pairs;
                            }
                        }
                    }
                    g.pool[node] = pool;
                }
            }
            for node in 0..nodes {
                if g.active[node][k] {
                    let (x0, y0, x1, y1) = g.rect(node);
                    let mut sum = 0.0;
                    for y in y0..y1 {
                        for x in x0..x1 {
                            let idx = y * w + x;
                            // No cut below a millionth here: a faint
                            // node is composed into its mean instead.
                            let value = self.scratch[idx][k].min(cap);
                            self.cells[idx].pheromone[k] = value;
                            sum += value;
                            if value > 0.0 {
                                present[k] = true;
                            }
                        }
                    }
                    let open = g.open[node] as f64;
                    g.avg[node][k] = if open > 0.0 { sum / open } else { 0.0 };
                } else {
                    let open = g.open[node] as f64;
                    let m = if open > 0.0 {
                        (g.pool[node] + g.inflow[node]) / open
                    } else {
                        0.0
                    };
                    let m = m.min(cap);
                    g.mean[node][k] = m;
                    g.avg[node][k] = m;
                    if m > 0.0 {
                        present[k] = true;
                    }
                    if g.touched[node] || m >= threshold {
                        g.refine(k, node, &mut self.cells, w);
                    }
                }
            }
            if g.volatile[k] {
                g.gradients(k);
            }
        }
        g.since_coarsen += 1;
        if g.since_coarsen >= COARSEN_EVERY {
            g.since_coarsen = 0;
            for &k in channels {
                let limit = 0.5 * g.threshold[k];
                for node in 0..nodes {
                    if !g.active[node][k] || g.walled[node] {
                        continue;
                    }
                    let (x0, y0, x1, y1) = g.rect(node);
                    let mut flat = true;
                    'scan: for y in y0..y1 {
                        for x in x0..x1 {
                            if self.cells[y * w + x].pheromone[k] >= limit {
                                flat = false;
                                break 'scan;
                            }
                        }
                    }
                    if flat {
                        g.coarsen(k, node, &mut self.cells, w);
                    }
                }
            }
        }
        for &k in channels {
            self.present[k] = present[k];
        }
        self.grain = Some(g);
    }

    /// What a node of the world's quadtree is made of, for the lens:
    /// open ground crossed at unit cost, walled through, or both.
    pub fn ground(&self, key: QuadKey, rect: (usize, usize, usize, usize)) -> crate::lens::Ground {
        let Some(tree) = &self.wall_tree else {
            return crate::lens::Ground::Open(1.0);
        };
        let (x0, y0, x1, y1) = rect;
        let area = (x1.saturating_sub(x0)) * (y1.saturating_sub(y0));
        let walls = *tree.get(key) as usize;
        if walls == 0 {
            crate::lens::Ground::Open(1.0)
        } else if walls >= area {
            crate::lens::Ground::Blocked
        } else {
            crate::lens::Ground::Mixed
        }
    }

    /// Side of the kinetics grain's nodes, cells (0 when every cell is
    /// always active).
    pub fn kinetics_grain(&self) -> usize {
        self.grain.as_ref().map(|g| 1usize << g.shift).unwrap_or(0)
    }

    /// The level of the world's quadtree whose nodes are the kinetics
    /// grain's, if there is one.
    pub fn grain_level(&self) -> Option<u8> {
        let g = self.grain.as_ref()?;
        let mut levels = 0u8;
        while (1usize << levels) < self.config.width.max(self.config.height).max(1) {
            levels += 1;
        }
        Some(levels - g.shift as u8)
    }

    /// Share of the grain's nodes at which a channel is at cell
    /// resolution (1 when every cell is always active).
    pub fn active_share(&self, kind: Pheromone) -> f64 {
        match &self.grain {
            None => 1.0,
            Some(g) => {
                let n = g.cols * g.rows;
                g.active.iter().filter(|a| a[kind.index()]).count() as f64 / n.max(1) as f64
            }
        }
    }

    /// The mass of a channel on the world's quadtree, from the kinetics
    /// grain (the cells, when every cell is active) up to the root: the
    /// field read at every coarser grain.
    pub fn field_tree(&self, kind: Pheromone) -> QuadTree<f64> {
        let k = kind.index();
        let w = self.config.width;
        let mut tree = QuadTree::<f64>::new(w, self.config.height);
        match &self.grain {
            None => {
                let levels = tree.levels();
                for (idx, c) in self.cells.iter().enumerate() {
                    if c.pheromone[k] > 0.0 {
                        let p = Position::new((idx % w) as i32, (idx / w) as i32);
                        if let Some(key) = tree.key_of(p, levels) {
                            *tree.get_mut(key) = c.pheromone[k];
                        }
                    }
                }
                tree.compose(|a, b| *a += *b);
            }
            Some(g) => {
                let level = tree.levels() - g.shift as u8;
                for node in 0..g.cols * g.rows {
                    let mass = if g.active[node][k] {
                        let (x0, y0, x1, y1) = g.rect(node);
                        let mut sum = 0.0;
                        for y in y0..y1 {
                            for x in x0..x1 {
                                sum += self.cells[y * w + x].pheromone[k];
                            }
                        }
                        sum
                    } else {
                        g.mean[node][k] * g.open[node] as f64
                    };
                    let key = QuadKey::new(level, (node % g.cols) as u32, (node / g.cols) as u32);
                    *tree.get_mut(key) = mass;
                }
                tree.compose_from(level, |a, b| *a += *b);
            }
        }
        tree
    }

    /// Corpses lying in a rectangle of cells, `(x0, y0, x1, y1)` with
    /// exclusive far corners (counted node by node where the rectangle
    /// covers whole nodes of the grain).
    pub fn corpses_in(&self, rect: (usize, usize, usize, usize)) -> u32 {
        let (x0, y0, x1, y1) = rect;
        let x1 = x1.min(self.config.width);
        let y1 = y1.min(self.config.height);
        if x0 >= x1 || y0 >= y1 {
            return 0;
        }
        let w = self.config.width;
        let scan = |ax0: usize, ay0: usize, ax1: usize, ay1: usize| -> u32 {
            let mut n = 0;
            for y in ay0..ay1 {
                for x in ax0..ax1 {
                    n += self.cells[y * w + x].corpses as u32;
                }
            }
            n
        };
        let Some(g) = &self.grain else {
            return scan(x0, y0, x1, y1);
        };
        let mut n = 0;
        for row in (y0 >> g.shift)..=((y1 - 1) >> g.shift) {
            for col in (x0 >> g.shift)..=((x1 - 1) >> g.shift) {
                let node = row * g.cols + col;
                let (nx0, ny0, nx1, ny1) = g.rect(node);
                if nx0 >= x0 && ny0 >= y0 && nx1 <= x1 && ny1 <= y1 {
                    n += g.corpses[node];
                } else {
                    n += scan(nx0.max(x0), ny0.max(y0), nx1.min(x1), ny1.min(y1));
                }
            }
        }
        n
    }

    /// Total volume of food remaining on the ground, microlitres.
    pub fn total_food(&self) -> f64 {
        self.cells.iter().map(|c| c.food_ul).sum()
    }

    /// Food volume remaining inside a region, microlitres.
    pub fn food_in(&self, rect: &Rect) -> f64 {
        let mut total = 0.0;
        for y in rect.min.y..=rect.max.y {
            for x in rect.min.x..=rect.max.x {
                total += self
                    .cell(Position::new(x, y))
                    .map(|c| c.food_ul)
                    .unwrap_or(0.0);
            }
        }
        total
    }

    /// Lay a corpse down on cell `p`.
    pub fn add_corpse(&mut self, p: Position) {
        if let Some(idx) = self.index(p) {
            self.cells[idx].corpses = self.cells[idx].corpses.saturating_add(1);
            if let Some(g) = self.grain.as_mut() {
                let node = g.node_of(p.x as usize, p.y as usize);
                g.corpses[node] += 1;
            }
        }
    }

    /// Pick a corpse up from cell `p`; false if there was none.
    pub fn take_corpse(&mut self, p: Position) -> bool {
        match self.index(p) {
            Some(idx) if self.cells[idx].corpses > 0 => {
                self.cells[idx].corpses -= 1;
                if let Some(g) = self.grain.as_mut() {
                    let node = g.node_of(p.x as usize, p.y as usize);
                    g.corpses[node] = g.corpses[node].saturating_sub(1);
                }
                true
            }
            _ => false,
        }
    }

    /// Corpses on cell `p` and its eight neighbours.
    pub fn corpses_near(&self, p: Position) -> u32 {
        let mut n = 0u32;
        for dy in -1..=1 {
            for dx in -1..=1 {
                n += self
                    .cell(p.offset(dx, dy))
                    .map(|c| c.corpses as u32)
                    .unwrap_or(0);
            }
        }
        n
    }

    /// Total corpses lying on the grid.
    pub fn total_corpses(&self) -> u32 {
        self.cells.iter().map(|c| c.corpses as u32).sum()
    }

    /// Sizes (in corpses) of the piles on the grid: eight-connected groups
    /// of cells holding at least one corpse, largest first.
    pub fn corpse_clusters(&self) -> Vec<u32> {
        let w = self.config.width;
        let h = self.config.height;
        let mut seen = vec![false; w * h];
        let mut sizes = Vec::new();
        for start in 0..w * h {
            if seen[start] || self.cells[start].corpses == 0 {
                continue;
            }
            let mut size = 0u32;
            let mut stack = vec![start];
            seen[start] = true;
            while let Some(i) = stack.pop() {
                size += self.cells[i].corpses as u32;
                let (x, y) = ((i % w) as i32, (i / w) as i32);
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let p = Position::new(x + dx, y + dy);
                        if let Some(j) = self.index(p) {
                            if !seen[j] && self.cells[j].corpses > 0 {
                                seen[j] = true;
                                stack.push(j);
                            }
                        }
                    }
                }
            }
            sizes.push(size);
        }
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        sizes
    }

    /// Total prey over the grid, milligrams.
    pub fn total_prey(&self) -> f64 {
        self.cells.iter().map(|c| c.prey_mg).sum()
    }

    /// Number of ants on the grid inside a rectangle.
    pub fn occupancy_in(&self, rect: &Rect) -> u32 {
        let mut total = 0u32;
        for y in rect.min.y..=rect.max.y {
            for x in rect.min.x..=rect.max.x {
                total += self
                    .cell(Position::new(x, y))
                    .map(|c| c.occupancy as u32)
                    .unwrap_or(0);
            }
        }
        total
    }

    /// Total amount of one channel over the grid.
    pub fn total_pheromone(&self, kind: Pheromone) -> f64 {
        let k = kind.index();
        match &self.grain {
            None => self.cells.iter().map(|c| c.pheromone[k]).sum(),
            Some(g) => {
                let w = self.config.width;
                let mut total = 0.0;
                for node in 0..g.cols * g.rows {
                    if g.active[node][k] {
                        let (x0, y0, x1, y1) = g.rect(node);
                        for y in y0..y1 {
                            for x in x0..x1 {
                                total += self.cells[y * w + x].pheromone[k];
                            }
                        }
                    } else {
                        total += g.mean[node][k] * g.open[node] as f64;
                    }
                }
                total
            }
        }
    }

    /// Total of one channel inside a region.
    pub fn pheromone_in(&self, rect: &Rect, kind: Pheromone) -> f64 {
        let mut total = 0.0;
        for y in rect.min.y..=rect.max.y {
            for x in rect.min.x..=rect.max.x {
                total += self.level(Position::new(x, y), kind);
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> WorldConfig {
        WorldConfig {
            width: 12,
            height: 10,
            nest: Position::new(6, 5),
            nest_radius: 1,
            food_sources: vec![FoodSource::pool(Position::new(1, 1), 1, 5.0, 0.7)],
            random_food: None,
            walls: vec![Rect::new(Position::new(9, 0), Position::new(9, 9))],
            ..WorldConfig::default()
        }
    }

    #[test]
    fn terrain_and_food_layout() {
        let mut rng = Rng::seed_from_u64(1);
        let world = World::new(small_config(), &mut rng);
        assert!(world.is_nest(Position::new(6, 5)));
        assert!(world.is_nest(Position::new(7, 6)));
        assert!(!world.is_nest(Position::new(8, 5)));
        assert!(!world.is_passable(Position::new(9, 4)));
        assert!(!world.is_passable(Position::new(-1, 0)));
        assert!((world.total_food() - 45.0).abs() < 1e-9);
        assert_eq!(world.cell(Position::new(1, 1)).unwrap().molarity, 0.7);
        assert_eq!(world.nest_cells().len(), 9);
    }

    #[test]
    fn food_is_taken_and_mixed() {
        let mut world = World::new(small_config(), &mut Rng::seed_from_u64(1));
        let (taken, molarity) = world.take_food(Position::new(1, 1), 2.0);
        assert!((taken - 2.0).abs() < 1e-12 && (molarity - 0.7).abs() < 1e-12);
        let (rest, _) = world.take_food(Position::new(1, 1), 10.0);
        assert!((rest - 3.0).abs() < 1e-12);
        assert_eq!(world.take_food(Position::new(1, 1), 1.0), (0.0, 0.0));
        assert!(!world.cell(Position::new(1, 1)).unwrap().has_food());
        world.place_food(&FoodSource::pool(Position::new(1, 1), 0, 2.0, 0.1));
        world.place_food(&FoodSource::pool(Position::new(1, 1), 0, 2.0, 0.5));
        let c = world.cell(Position::new(1, 1)).unwrap();
        assert!((c.food_ul - 4.0).abs() < 1e-12 && (c.molarity - 0.3).abs() < 1e-12);
    }

    #[test]
    fn open_regions_make_everything_else_wall() {
        let cfg = WorldConfig {
            open: vec![Rect::new(Position::new(0, 5), Position::new(11, 5))],
            walls: Vec::new(),
            ..small_config()
        };
        let world = World::new(cfg, &mut Rng::seed_from_u64(1));
        assert!(world.is_passable(Position::new(3, 5)));
        assert!(!world.is_passable(Position::new(3, 4)));
        assert!(
            world.is_nest(Position::new(6, 4)),
            "the nest is carved regardless"
        );
    }

    #[test]
    fn segments_cannot_cut_corners() {
        let cfg = WorldConfig {
            open: vec![
                Rect::new(Position::new(0, 5), Position::new(5, 5)),
                Rect::new(Position::new(5, 0), Position::new(5, 5)),
            ],
            walls: Vec::new(),
            nest_radius: 0,
            nest: Position::new(0, 5),
            ..small_config()
        };
        let world = World::new(cfg, &mut Rng::seed_from_u64(1));
        let corner = Point::center_of(Position::new(4, 5));
        assert!(world.segment_passable(corner, Point::center_of(Position::new(5, 5))));
        assert!(!world.segment_passable(corner, Point::center_of(Position::new(5, 4))));
        assert!(!world.segment_passable(corner, Point::new(4.5, 3.0)));
        assert!(world.is_passable_point(Point::new(5.9, 0.1)));
        assert!(!world.is_passable_point(Point::new(6.1, 0.1)));
    }

    #[test]
    fn random_food_is_reproducible_from_config_seed() {
        let cfg = WorldConfig {
            seed: Some(99),
            ..WorldConfig::default()
        };
        let mut r1 = Rng::seed_from_u64(1);
        let mut r2 = Rng::seed_from_u64(2);
        let a = World::new(cfg.clone(), &mut r1);
        let b = World::new(cfg, &mut r2);
        let fa: Vec<(f64, f64)> = a.cells().iter().map(|c| (c.food_ul, c.molarity)).collect();
        let fb: Vec<(f64, f64)> = b.cells().iter().map(|c| (c.food_ul, c.molarity)).collect();
        assert_eq!(fa, fb);
        assert!(a.total_food() > 0.0);
        assert!(a.cells().iter().any(|c| c.has_food() && c.molarity >= 0.4));
    }

    #[test]
    fn pheromone_evaporates_by_half_life_and_diffuses() {
        let mut rng = Rng::seed_from_u64(1);
        let cfg = WorldConfig {
            tick_s: 60.0,
            ..small_config()
        };
        let mut world = World::new(cfg, &mut rng);
        let p = Position::new(3, 3);
        world.deposit(p, Pheromone::Trail, 10.0);
        let before = world.total_pheromone(Pheromone::Trail);
        world.step_pheromones();
        let after = world.total_pheromone(Pheromone::Trail);
        let expected = before * world.channel(Pheromone::Trail).retention_per_tick(60.0);
        assert!((after - expected).abs() < 1e-9, "{after} vs {expected}");
        assert!(
            world
                .cell(Position::new(3, 2))
                .unwrap()
                .level(Pheromone::Trail)
                > 0.0
        );
        assert!(world.cell(p).unwrap().level(Pheromone::Trail) > 5.0);
        world.deposit(p, Pheromone::Trail, 1e9);
        assert_eq!(
            world.level(p, Pheromone::Trail),
            world.channel(Pheromone::Trail).cap
        );
    }

    #[test]
    fn evaporation_factor_scales_decay() {
        let cfg = WorldConfig {
            tick_s: 60.0,
            ..small_config()
        };
        let mut warm = World::new(cfg.clone(), &mut Rng::seed_from_u64(1));
        let mut cool = World::new(cfg, &mut Rng::seed_from_u64(1));
        warm.set_evaporation_factor(2.0);
        assert_eq!(warm.evaporation_factor(), 2.0);
        let p = Position::new(3, 3);
        warm.deposit(p, Pheromone::Trail, 10.0);
        cool.deposit(p, Pheromone::Trail, 10.0);
        warm.step_pheromones();
        cool.step_pheromones();
        let r = cool.channel(Pheromone::Trail).retention_per_tick(60.0);
        assert!((warm.total_pheromone(Pheromone::Trail) - 10.0 * r * r).abs() < 1e-9);
        warm.set_evaporation_factor(1.0);
        assert_eq!(warm.evaporation_factor(), 1.0);
    }

    #[test]
    fn sampling_interpolates_between_cells() {
        let mut world = World::new(small_config(), &mut Rng::seed_from_u64(1));
        world.deposit(Position::new(3, 3), Pheromone::Trail, 8.0);
        let centre = Point::center_of(Position::new(3, 3));
        assert!((world.sample(centre, Pheromone::Trail) - 8.0).abs() < 1e-12);
        let half = Point::new(4.0, 3.5);
        assert!((world.sample(half, Pheromone::Trail) - 4.0).abs() < 1e-12);
        let far = Point::center_of(Position::new(5, 5));
        assert_eq!(world.sample(far, Pheromone::Trail), 0.0);
        // Walls contribute nothing and points off the grid are safe.
        world.deposit(Position::new(8, 4), Pheromone::Trail, 8.0);
        assert!((world.sample(Point::new(9.0, 4.5), Pheromone::Trail) - 4.0).abs() < 1e-12);
        assert_eq!(world.sample(Point::new(-3.0, -3.0), Pheromone::Trail), 0.0);
    }

    #[test]
    fn inert_channels_take_nothing() {
        let mut rng = Rng::seed_from_u64(1);
        let mut world = World::new(small_config(), &mut rng);
        world.deposit(Position::new(3, 3), Pheromone::NoEntry, 10.0);
        assert_eq!(world.total_pheromone(Pheromone::NoEntry), 0.0);
        world.deposit(Position::new(3, 3), Pheromone::Alarm, 10.0);
        world.step_pheromones();
        assert!(world.total_pheromone(Pheromone::Alarm) > 0.0);
        assert!(world.total_pheromone(Pheromone::Alarm) < 10.0);
    }

    #[test]
    fn diffusion_conserves_mass_in_corridors() {
        let cfg = WorldConfig {
            open: vec![Rect::new(Position::new(0, 5), Position::new(11, 5))],
            walls: Vec::new(),
            pheromones: Some({
                let mut set = Species::lasius_niger().pheromones();
                set[Pheromone::Trail.index()].half_life_s = f64::INFINITY;
                set[Pheromone::Trail.index()].diffusion_per_s = 0.05;
                set
            }),
            ..small_config()
        };
        let mut world = World::new(cfg, &mut Rng::seed_from_u64(1));
        world.deposit(Position::new(5, 5), Pheromone::Trail, 50.0);
        for _ in 0..200 {
            world.step_pheromones();
        }
        let total = world.total_pheromone(Pheromone::Trail);
        assert!(
            (total - 50.0).abs() < 1e-3,
            "no evaporation, no leak: {total}"
        );
        assert!(world.level(Position::new(8, 5), Pheromone::Trail) > 0.0);
    }

    #[test]
    fn walls_block_diffusion() {
        // A diffusive trail, so that one tick moves more than the cut
        // below which a cell counts as empty.
        let cfg = WorldConfig {
            pheromones: Some({
                let mut set = Species::lasius_niger().pheromones();
                set[Pheromone::Trail.index()].diffusion_per_s = 0.05;
                set
            }),
            ..small_config()
        };
        let mut world = World::new(cfg, &mut Rng::seed_from_u64(1));
        world.deposit(Position::new(8, 4), Pheromone::Trail, 10.0);
        world.step_pheromones();
        assert_eq!(world.level(Position::new(9, 4), Pheromone::Trail), 0.0);
        assert!(world.level(Position::new(7, 4), Pheromone::Trail) > 0.0);
    }

    #[test]
    fn the_grain_conserves_mass_and_coarsens_the_far_field() {
        // An open field, no evaporation: whatever diffuses out of the
        // active nodes is carried by the coarse ones, and nothing leaks.
        let cfg = WorldConfig {
            width: 64,
            height: 64,
            nest: Position::new(32, 32),
            random_food: None,
            kinetics_grain: 8,
            pheromones: Some({
                let mut set = Species::lasius_niger().pheromones();
                set[Pheromone::Trail.index()].half_life_s = f64::INFINITY;
                set[Pheromone::Trail.index()].diffusion_per_s = 0.2;
                set
            }),
            ..WorldConfig::default()
        };
        let mut world = World::new(cfg, &mut Rng::seed_from_u64(1));
        assert_eq!(world.kinetics_grain(), 8);
        world.deposit(Position::new(20, 20), Pheromone::Trail, 40.0);
        assert!(
            world.active_share(Pheromone::Trail) < 0.1,
            "one node refined"
        );
        for _ in 0..300 {
            world.step_pheromones();
        }
        let total = world.total_pheromone(Pheromone::Trail);
        assert!((total - 40.0).abs() < 1e-6, "mass conserved: {total}");
        // The far field is carried as means, and reads as such.
        let share = world.active_share(Pheromone::Trail);
        assert!(share < 1.0, "the far field stays coarse: {share}");
        assert!(world.level(Position::new(60, 60), Pheromone::Trail) > 0.0);
        let tree = world.field_tree(Pheromone::Trail);
        assert!((tree.get(QuadKey::ROOT) - 40.0).abs() < 1e-6);
        let level = world.grain_level().unwrap();
        let sum: f64 = tree.nodes(level).map(|(_, v)| *v).sum();
        assert!((sum - 40.0).abs() < 1e-6);
    }

    #[test]
    fn the_grain_matches_the_dense_sweep_where_the_field_has_structure() {
        // A trail laid every tick across an evaporating field: the two
        // grains agree on and around the trail, and on the total, and the
        // coarse one keeps most nodes coarse.
        let set = {
            let mut set = Species::lasius_niger().pheromones();
            set[Pheromone::Trail.index()].diffusion_per_s = 0.05;
            set
        };
        let make = |grain: usize| {
            World::new(
                WorldConfig {
                    width: 96,
                    height: 64,
                    nest: Position::new(48, 32),
                    random_food: None,
                    kinetics_grain: grain,
                    pheromones: Some(set.clone()),
                    ..WorldConfig::default()
                },
                &mut Rng::seed_from_u64(1),
            )
        };
        let mut dense = make(0);
        let mut coarse = make(8);
        for tick in 0..1200 {
            for x in 20..76 {
                let y = 32 + ((x as f64 - 20.0) * 0.15).sin().round() as i32;
                let amount = 0.5 + 0.5 * ((tick % 7) as f64 / 7.0);
                dense.deposit(Position::new(x, y), Pheromone::Trail, amount);
                coarse.deposit(Position::new(x, y), Pheromone::Trail, amount);
            }
            dense.step_pheromones();
            coarse.step_pheromones();
        }
        let k = dense.channel(Pheromone::Trail).k;
        let (td, tc) = (
            dense.total_pheromone(Pheromone::Trail),
            coarse.total_pheromone(Pheromone::Trail),
        );
        assert!((td - tc).abs() < 0.01 * td, "totals agree: {td} vs {tc}");
        // On the trail and its halo the two agree; at the halo's edge
        // (a twentieth of the perception constant, where the coarse
        // nodes begin) the coarse mean stands in for the last of the
        // dense front, within a few percent.
        let (mut core, mut halo) = (0.0f64, 0.0f64);
        for y in 0..64 {
            for x in 0..96 {
                let p = Position::new(x, y);
                let a = dense.level(p, Pheromone::Trail);
                let b = coarse.level(p, Pheromone::Trail);
                if a >= k {
                    core = core.max((a - b).abs() / a);
                } else if a >= 0.05 * k {
                    halo = halo.max((a - b).abs() / a);
                }
            }
        }
        assert!(core < 1e-3, "the trail agrees: {core}");
        assert!(halo < 0.05, "the halo agrees within a few percent: {halo}");
        let share = coarse.active_share(Pheromone::Trail);
        assert!(share < 0.6, "most of the grid is coarse: {share}");
    }

    #[test]
    fn renewing_sources_refill_to_their_standing_volume() {
        let cfg = WorldConfig {
            food_sources: vec![
                FoodSource::pool(Position::new(1, 1), 0, 0.5, 1.0).renewing(0.1),
                FoodSource::pool(Position::new(0, 0), 0, 5.0, 1.0),
            ],
            ..small_config()
        };
        let mut world = World::new(cfg, &mut Rng::seed_from_u64(1));
        let c = Position::new(1, 1);
        assert_eq!(world.cell(c).unwrap().food_capacity_ul, 0.5);
        let (taken, _) = world.take_food(c, 0.4);
        assert!((taken - 0.4).abs() < 1e-12);
        world.step_food(0.0, 0.0);
        assert!((world.cell(c).unwrap().food_ul - 0.2).abs() < 1e-12);
        for _ in 0..10 {
            world.step_food(0.0, 0.0);
        }
        assert!(
            (world.cell(c).unwrap().food_ul - 0.5).abs() < 1e-12,
            "capped at the standing volume"
        );
        // A plain pool does not refill.
        let (taken, _) = world.take_food(Position::new(0, 0), 5.0);
        assert!((taken - 5.0).abs() < 1e-12);
        world.step_food(0.0, 0.0);
        assert!(!world.cell(Position::new(0, 0)).unwrap().has_food());
    }

    #[test]
    fn prey_is_placed_and_cut() {
        let cfg = WorldConfig {
            food_sources: vec![FoodSource::prey(Position::new(2, 7), 0, 3.0)],
            ..small_config()
        };
        let mut world = World::new(cfg, &mut Rng::seed_from_u64(1));
        let p = Position::new(2, 7);
        let c = world.cell(p).unwrap();
        assert!(c.has_prey() && c.has_food() && !c.has_solution());
        assert!((world.total_prey() - 3.0).abs() < 1e-12);
        assert!((world.take_prey(p, 1.0) - 1.0).abs() < 1e-12);
        assert!((world.take_prey(p, 5.0) - 2.0).abs() < 1e-12);
        assert_eq!(world.take_prey(p, 1.0), 0.0);
        assert!(!world.cell(p).unwrap().has_food());
        assert_eq!(
            world.take_prey(Position::new(0, 0), 1.0),
            0.0,
            "a solution cell has no prey"
        );
    }

    #[test]
    fn corpses_pile_and_cluster() {
        let mut world = World::new(small_config(), &mut Rng::seed_from_u64(1));
        let a = Position::new(2, 7);
        let b = Position::new(3, 8);
        let far = Position::new(7, 1);
        world.add_corpse(a);
        world.add_corpse(a);
        world.add_corpse(b);
        world.add_corpse(far);
        assert_eq!(world.total_corpses(), 4);
        assert_eq!(world.corpses_near(a), 3, "a diagonal neighbour counts");
        assert_eq!(world.corpses_near(far), 1);
        assert_eq!(world.corpse_clusters(), vec![3, 1]);
        assert!(world.take_corpse(far));
        assert!(!world.take_corpse(far));
        assert_eq!(world.corpse_clusters(), vec![3]);
        // Scattering at construction is reproducible from the config seed.
        let cfg = WorldConfig {
            scattered_corpses: 20,
            seed: Some(3),
            ..small_config()
        };
        let w1 = World::new(cfg.clone(), &mut Rng::seed_from_u64(9));
        let w2 = World::new(cfg, &mut Rng::seed_from_u64(10));
        assert_eq!(w1.total_corpses(), 20);
        assert!(w1
            .cells()
            .iter()
            .all(|c| c.corpses == 0 || c.terrain == Terrain::Open));
        let lay = |w: &World| w.cells().iter().map(|c| c.corpses).collect::<Vec<_>>();
        assert_eq!(lay(&w1), lay(&w2));
    }

    #[test]
    fn food_gives_off_an_odour_that_spreads_and_fades() {
        let cfg = WorldConfig {
            food_sources: vec![
                FoodSource::pool(Position::new(2, 7), 0, 100.0, 1.0),
                FoodSource::prey(Position::new(7, 7), 0, 100.0),
            ],
            ..small_config()
        };
        let mut world = World::new(cfg, &mut Rng::seed_from_u64(1));
        world.step_food(0.5, 0.5);
        let sugar = world.level(Position::new(2, 7), Pheromone::Odour);
        let prey = world.level(Position::new(7, 7), Pheromone::Odour);
        assert!(
            (sugar - 2.5).abs() < 1e-9,
            "five microlitres count: {sugar}"
        );
        assert!(
            (prey - 15.0).abs() < 1e-9,
            "thirty milligrams count: {prey}"
        );
        for _ in 0..20 {
            world.step_pheromones();
            world.step_food(0.5, 0.5);
        }
        assert!(
            world.level(Position::new(6, 7), Pheromone::Odour) > 0.0,
            "the smell spreads"
        );
        assert!(world.level(Position::new(7, 7), Pheromone::Odour) < 500.0 + 1e-9);
        // Without food the smell fades within minutes.
        let mut bare = World::new(small_config(), &mut Rng::seed_from_u64(1));
        bare.deposit(Position::new(4, 4), Pheromone::Odour, 100.0);
        for _ in 0..300 {
            bare.step_pheromones();
        }
        assert!(bare.total_pheromone(Pheromone::Odour) < 5.0);
        // Inert odour: nothing is given off.
        let mut quiet = Species::lasius_niger();
        quiet.food_odour = PheromoneParams::inert();
        let cfg = WorldConfig {
            pheromones: Some(quiet.pheromones()),
            food_sources: vec![FoodSource::prey(Position::new(7, 7), 0, 100.0)],
            ..small_config()
        };
        let mut silent = World::new(cfg, &mut Rng::seed_from_u64(1));
        silent.step_food(0.5, 0.5);
        assert_eq!(silent.total_pheromone(Pheromone::Odour), 0.0);
    }

    #[test]
    fn landmarks_are_placed_and_seen_within_sight() {
        let cfg = WorldConfig {
            landmarks: vec![Position::new(2, 2), Position::new(8, 8)],
            random_landmarks: 3,
            seed: Some(4),
            ..small_config()
        };
        let world = World::new(cfg.clone(), &mut Rng::seed_from_u64(1));
        assert_eq!(world.landmarks().len(), 5);
        assert!(world
            .landmarks()
            .iter()
            .all(|&l| world.cell(l).unwrap().terrain == Terrain::Open));
        let again = World::new(cfg, &mut Rng::seed_from_u64(2));
        assert_eq!(
            world.landmarks(),
            again.landmarks(),
            "placement follows the config seed"
        );
        let from = Point::new(3.5, 3.5);
        assert_eq!(
            world.nearest_landmark(from, 2.0),
            Some((0, Position::new(2, 2)))
        );
        assert_eq!(world.nearest_landmark(from, 0.5), None);
        let near_far = world.nearest_landmark(Point::new(7.5, 7.5), 2.0);
        assert_eq!(near_far, Some((1, Position::new(8, 8))));
    }

    #[test]
    fn capacity_zones_and_crowding() {
        let cfg = WorldConfig {
            cell_capacity: 8,
            kinetics_stride: 1,
            capacity_zones: vec![CapacityZone {
                rect: Rect::new(Position::new(0, 0), Position::new(3, 3)),
                capacity: 2,
            }],
            ..small_config()
        };
        let mut world = World::new(cfg, &mut Rng::seed_from_u64(1));
        assert_eq!(world.cell(Position::new(1, 1)).unwrap().capacity, 2);
        assert_eq!(world.cell(Position::new(5, 0)).unwrap().capacity, 8);
        assert_eq!(
            world.cell(Position::new(6, 5)).unwrap().capacity,
            u16::MAX,
            "nest cells hold anyone"
        );
        let c = world.cell_mut(Position::new(1, 1)).unwrap();
        c.occupancy = 3;
        assert!(!c.has_room());
        assert!((c.crowding(false) - 1.5).abs() < 1e-12);
        assert!((c.crowding(true) - 1.0).abs() < 1e-12);
        let empty = world.cell(Position::new(5, 0)).unwrap();
        assert!(empty.has_room());
        assert_eq!(empty.crowding(true), 0.0);
    }

    #[test]
    fn counters_count_crossings() {
        let cfg = WorldConfig {
            counters: vec![Counter {
                name: "gate".to_string(),
                rect: Rect::new(Position::new(4, 0), Position::new(4, 9)),
            }],
            ..small_config()
        };
        let mut world = World::new(cfg, &mut Rng::seed_from_u64(1));
        world.record_crossing(Position::new(4, 2));
        world.record_crossing(Position::new(5, 2));
        world.record_crossing(Position::new(4, 7));
        assert_eq!(world.crossings("gate"), Some(2));
        assert_eq!(world.crossings("nope"), None);
        let rect = Rect::new(Position::new(0, 0), Position::new(2, 2));
        world.deposit(Position::new(1, 1), Pheromone::Trail, 3.0);
        assert!((world.pheromone_in(&rect, Pheromone::Trail) - 3.0).abs() < 1e-12);
        assert!((world.food_in(&rect) - 45.0).abs() < 1e-9);
    }
}
