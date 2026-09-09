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
//! The field is kept on two structures. The grid is tiled by the nodes
//! of its quadtree at one level (`kinetics_grain` cells across). A
//! *substrate mark* (trail, home, territory, no entry) lies where ants
//! walked and is read within antennal reach, so it is kept at cell
//! resolution where it has structure and as one mean per node where it
//! is faint: a node is *active*, its cells carrying the field, or
//! *coarse*. Mass moves between cells, between a cell and a coarse
//! node, and between coarse nodes by the same conservative shares; a
//! deposit or an arriving front refines a coarse node into cells
//! (coarse to fine), and a node whose cells have all faded below half a
//! threshold is composed into its mean (fine to coarse). A *volatile*
//! (the smell of food, alarm) is volumetric information that flows
//! through the air, and lives on the grain throughout: its sources emit
//! into the pools of their nodes, its mass moves between nodes, and a
//! reader sees the plane through a node's mean with the gradient of its
//! neighbours', never a cell. On both structures flat coarse nodes merge
//! into blocks of up to eight nodes across and split again when the
//! field reaches them, so a flat far field costs a few blocks. Walls
//! take part cell by cell: two coarse nodes exchange through the open
//! cells facing each other across their side, and a node with no open
//! cell is never visited, so a shaped arena costs what its open ground
//! costs, not its bounding box. [`World::field_tree`] reads the mass at
//! every level above the grain. A node whose open ground is in several
//! pieces keeps a substrate mark at cell resolution, one mean being no
//! account of two corridors.

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    /// Edges joined in pairs: faces of a box meeting at a corner, the two
    /// sides of a tube. See [`Portal`].
    pub portals: Vec<Portal>,
    /// Regions that stand at an angle in space, where walking one way is
    /// climbing. See [`Slope`].
    pub slopes: Vec<Slope>,
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
    /// `k`, below which a node of a substrate mark may be coarse: a node
    /// whose cells are all below half of it is composed into its mean,
    /// and a coarse node whose mean reaches it is refined into cells
    /// again. The volatile channels (the smell of food, alarm) are
    /// volumetric and live on the grain throughout, whatever this is.
    pub coarse_below: f64,
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
            portals: Vec::new(),
            slopes: Vec::new(),
            counters: Vec::new(),
            cell_capacity: 8,
            kinetics_stride: 1,
            kinetics_grain: 8,
            coarse_below: 0.01,
            capacity_zones: Vec::new(),
            scattered_corpses: 0,
            landmarks: Vec::new(),
            random_landmarks: 0,
            pheromones: None,
            seed: None,
        }
    }
}

/// One of the four sides of a cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    /// The top side (towards smaller `y`).
    North,
    /// The right side.
    East,
    /// The bottom side.
    South,
    /// The left side.
    West,
}

impl Side {
    /// Heading of the outward normal (`x` east, `y` south, clockwise
    /// positive).
    pub fn outward(self) -> f64 {
        match self {
            Side::North => -std::f64::consts::FRAC_PI_2,
            Side::East => 0.0,
            Side::South => std::f64::consts::FRAC_PI_2,
            Side::West => std::f64::consts::PI,
        }
    }

    /// The cell across this side, as an offset.
    pub fn offset(self) -> (i32, i32) {
        match self {
            Side::North => (0, -1),
            Side::East => (1, 0),
            Side::South => (0, 1),
            Side::West => (-1, 0),
        }
    }
}

/// A straight run of cells with one side open: the boundary a portal
/// joins. The run goes from `start` to `end` (inclusive) along the side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Edge {
    /// The first cell of the run.
    pub start: Position,
    /// The last cell of the run.
    pub end: Position,
    /// The side of the cells that opens.
    pub side: Side,
}

impl Edge {
    /// The run's cells, from `start` to `end`.
    pub fn cells(&self) -> Vec<Position> {
        let n = self.len() as i32;
        let (dx, dy) = match self.side {
            Side::North | Side::South => ((self.end.x - self.start.x).signum(), 0),
            Side::East | Side::West => (0, (self.end.y - self.start.y).signum()),
        };
        (0..n)
            .map(|i| Position::new(self.start.x + dx * i, self.start.y + dy * i))
            .collect()
    }

    /// Cells in the run.
    pub fn len(&self) -> usize {
        match self.side {
            Side::North | Side::South => (self.end.x - self.start.x).unsigned_abs() as usize + 1,
            Side::East | Side::West => (self.end.y - self.start.y).unsigned_abs() as usize + 1,
        }
    }

    /// Whether the run is empty (it never is).
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Where a point lies relative to the edge: `along` the run from the
    /// start cell's near corner, and `depth` beyond the open side
    /// (negative inside the cells).
    fn coords(&self, p: Point) -> (f64, f64) {
        let (sx, sy) = (self.start.x as f64, self.start.y as f64);
        match self.side {
            Side::North | Side::South => {
                let along = if self.end.x >= self.start.x {
                    p.x - sx
                } else {
                    sx + 1.0 - p.x
                };
                let depth = if self.side == Side::North {
                    sy - p.y
                } else {
                    p.y - (sy + 1.0)
                };
                (along, depth)
            }
            Side::East | Side::West => {
                let along = if self.end.y >= self.start.y {
                    p.y - sy
                } else {
                    sy + 1.0 - p.y
                };
                let depth = if self.side == Side::East {
                    p.x - (sx + 1.0)
                } else {
                    sx - p.x
                };
                (along, depth)
            }
        }
    }

    /// The point `along` the run and `depth` into the cells from the
    /// open side.
    fn point_inside(&self, along: f64, depth: f64) -> Point {
        let (sx, sy) = (self.start.x as f64, self.start.y as f64);
        match self.side {
            Side::North | Side::South => {
                let x = if self.end.x >= self.start.x {
                    sx + along
                } else {
                    sx + 1.0 - along
                };
                let y = if self.side == Side::North {
                    sy + depth
                } else {
                    sy + 1.0 - depth
                };
                Point::new(x, y)
            }
            Side::East | Side::West => {
                let y = if self.end.y >= self.start.y {
                    sy + along
                } else {
                    sy + 1.0 - along
                };
                let x = if self.side == Side::East {
                    sx + 1.0 - depth
                } else {
                    sx + depth
                };
                Point::new(x, y)
            }
        }
    }
}

/// Two edges of the same length joined: what lies beyond the open side
/// of one is the cells of the other, as two faces of a box meet at a
/// corner or a tube's two sides meet round its back. A body crossing
/// turns by the angle between the sides, the field flows through, and
/// the cells beyond either edge must be wall or off the grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Portal {
    /// One edge.
    pub a: Edge,
    /// The other, its cells matched to `a`'s in order.
    pub b: Edge,
}

impl Portal {
    /// The turn a body makes crossing from `a` into `b`.
    pub fn turn(&self) -> f64 {
        crate::geometry::wrap_angle(
            self.b.side.outward() + std::f64::consts::PI - self.a.side.outward(),
        )
    }
}

/// A region of the net that stands at an angle in space: walking
/// towards `up` is climbing, and is slower by `factor` (1 for no
/// slope, 0.5 for half speed straight up); a body on it may lose its
/// grip and fall with chance `slip` per tick (1 on a fluon barrier, a
/// little on a ceiling, none on a rough wall), more when laden.
#[derive(Clone, Debug, PartialEq)]
pub struct Slope {
    /// The region.
    pub rect: Rect,
    /// The direction of the net that points up in space.
    pub up: Side,
    /// Speed factor straight up.
    pub factor: f64,
    /// Chance per tick of losing grip.
    pub slip: f64,
}

/// Where a point beyond a portal's edge comes out, and the turn made.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Warp {
    /// The point on the other side.
    pub point: Point,
    /// The turn a heading makes, radians.
    pub turn: f64,
}

/// How far beyond an edge a point may lie and still come through its
/// portal, cells.
const PORTAL_DEPTH: i32 = 4;

/// A counter's running total.
#[derive(Clone, Debug, PartialEq)]
pub struct CounterState {
    /// The counter's definition.
    pub counter: Counter,
    /// Ant cell entries inside the region.
    pub crossings: u64,
}

/// How often, in sweeps of the kinetics, active nodes are checked for
/// coarsening and flat coarse nodes for merging.
const COARSEN_EVERY: u32 = 16;

/// The largest block of coarse nodes, in nodes per side: flat coarse
/// nodes merge into blocks of 2, 4 and 8 nodes across (up to 64 cells
/// at the default grain), so a flat far field costs a few blocks.
const BLOCK_MAX: usize = 8;

/// Relative spread of four blocks' means within which they are flat
/// enough to merge.
const FLAT: f64 = 0.1;

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
    /// Nodes with a wall in them (their cells are checked one by one).
    walled: Vec<bool>,
    /// Nodes whose open ground is in several pieces (two corridors, say),
    /// which keep a substrate mark at cell resolution: one mean would
    /// mix the pieces.
    split: Vec<bool>,
    /// Nodes with no open cell at all, which the kinetics never visit.
    void: Vec<bool>,
    /// Per node and side (north, east, south, west), the neighbouring
    /// node, or `u32::MAX` at the grid's edge.
    nb: Vec<[u32; 4]>,
    /// Cells with food or prey per node: the sources of the smell of
    /// food, whose nodes stay at cell resolution.
    sources: Vec<u32>,
    /// Per node and side (north, east, south, west), the pairs of open
    /// cells facing each other across the side: what two coarse nodes
    /// exchange through.
    pairs: Vec<[u32; 4]>,
    /// Per node and channel, whether the node is active.
    active: Vec<[bool; Pheromone::COUNT]>,
    /// Per node and channel, the anchor (top-left node) of the block of
    /// coarse nodes it belongs to: itself unless merged.
    anchor: Vec<[u32; Pheromone::COUNT]>,
    /// Per anchor and channel, the block's side in nodes (1, 2, 4, 8).
    bsize: Vec<[u8; Pheromone::COUNT]>,
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
    /// Per channel, whether it is volumetric: kept on the grain
    /// throughout, its sources emitting into node pools, read as planes.
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
            split: vec![false; n],
            void: vec![false; n],
            nb: vec![[u32::MAX; 4]; n],
            sources: vec![0; n],
            pairs: vec![[0; 4]; n],
            active: vec![[false; Pheromone::COUNT]; n],
            anchor: (0..n).map(|i| [i as u32; Pheromone::COUNT]).collect(),
            bsize: vec![[1; Pheromone::COUNT]; n],
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
            // A volatile never leaves the grain: no threshold lifts it
            // into cells.
            g.threshold[k] = if volatile {
                f64::INFINITY
            } else {
                config.coarse_below.max(0.0) * p.k.max(1e-9)
            };
            g.volatile[k] = volatile;
        }
        for node in 0..n {
            let (x0, y0, x1, y1) = g.rect(node);
            let mut open = 0;
            let mut walled = false;
            let mut corpses = 0;
            let mut sources = 0;
            for y in y0..y1 {
                for x in x0..x1 {
                    let c = &cells[y * config.width + x];
                    if c.terrain == Terrain::Wall {
                        walled = true;
                    } else {
                        open += 1;
                    }
                    corpses += c.corpses as u32;
                    if c.has_food() || c.renewal_ul_per_s > 0.0 || c.food_capacity_ul > 0.0 {
                        sources += 1;
                    }
                }
            }
            g.open[node] = open;
            g.walled[node] = walled;
            g.void[node] = open == 0;
            g.corpses[node] = corpses;
            g.sources[node] = sources;
            // Is the open ground one piece? Flood it from its first
            // open cell, within the node.
            if walled && open > 0 {
                let side = x1 - x0;
                let mut seen = vec![false; side * (y1 - y0)];
                let mut stack = Vec::new();
                'first: for y in y0..y1 {
                    for x in x0..x1 {
                        if cells[y * config.width + x].terrain != Terrain::Wall {
                            stack.push((x, y));
                            seen[(y - y0) * side + (x - x0)] = true;
                            break 'first;
                        }
                    }
                }
                let mut reached = 0u32;
                while let Some((x, y)) = stack.pop() {
                    reached += 1;
                    for (dx, dy) in [(0i64, -1i64), (1, 0), (0, 1), (-1, 0)] {
                        let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                        if nx < x0 as i64 || ny < y0 as i64 || nx >= x1 as i64 || ny >= y1 as i64 {
                            continue;
                        }
                        let (nx, ny) = (nx as usize, ny as usize);
                        let s = (ny - y0) * side + (nx - x0);
                        if !seen[s] && cells[ny * config.width + nx].terrain != Terrain::Wall {
                            seen[s] = true;
                            stack.push((nx, ny));
                        }
                    }
                }
                if reached < open {
                    g.split[node] = true;
                    for k in 0..Pheromone::COUNT {
                        if !g.volatile[k] {
                            g.active[node][k] = true;
                        }
                    }
                }
            }
            for side in 0..4u8 {
                g.nb[node][side as usize] = g
                    .neighbour(node, side)
                    .map(|n| n as u32)
                    .unwrap_or(u32::MAX);
            }
        }
        // The open pairs across every side.
        let is_open = |x: i64, y: i64| -> bool {
            x >= 0
                && y >= 0
                && (x as usize) < config.width
                && (y as usize) < config.height
                && cells[y as usize * config.width + x as usize].terrain != Terrain::Wall
        };
        for node in 0..n {
            let (x0, y0, x1, y1) = g.rect(node);
            let mut pairs = [0u32; 4];
            for x in x0..x1 {
                if is_open(x as i64, y0 as i64) && is_open(x as i64, y0 as i64 - 1) {
                    pairs[0] += 1;
                }
                if is_open(x as i64, y1 as i64 - 1) && is_open(x as i64, y1 as i64) {
                    pairs[2] += 1;
                }
            }
            for y in y0..y1 {
                if is_open(x1 as i64 - 1, y as i64) && is_open(x1 as i64, y as i64) {
                    pairs[1] += 1;
                }
                if is_open(x0 as i64, y as i64) && is_open(x0 as i64 - 1, y as i64) {
                    pairs[3] += 1;
                }
            }
            g.pairs[node] = pairs;
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

    /// Whether a node holds sources of a channel: food or prey for the
    /// smell of food, corpses for alarm (nothing for the substrate
    /// marks).
    fn has_sources(&self, node: usize, k: usize) -> bool {
        match Pheromone::ALL[k] {
            Pheromone::Odour => self.sources[node] > 0,
            Pheromone::Alarm => self.corpses[node] > 0,
            _ => false,
        }
    }

    /// The open cells of a block (whole nodes beyond a single one).
    fn block_open(&self, anchor: usize, k: usize) -> u32 {
        let s = self.bsize[anchor][k] as u32;
        if s <= 1 {
            self.open[anchor]
        } else {
            s * s * (1u32 << (2 * self.shift))
        }
    }

    /// Break a block into its nodes, each keeping the block's mean.
    fn split(&mut self, k: usize, anchor: usize) {
        let s = self.bsize[anchor][k] as usize;
        if s <= 1 {
            return;
        }
        let (col, row) = (anchor % self.cols, anchor / self.cols);
        let (m, a) = (self.mean[anchor][k], self.avg[anchor][k]);
        for dy in 0..s {
            for dx in 0..s {
                let n = (row + dy) * self.cols + col + dx;
                self.anchor[n][k] = n as u32;
                self.bsize[n][k] = 1;
                self.mean[n][k] = m;
                self.avg[n][k] = a;
            }
        }
    }

    /// Merge flat coarse blocks into larger ones, level by level: four
    /// blocks of a size, whole and open, whose means lie within a tenth
    /// of one another (or all below `cut`) become one of twice the side.
    fn merge(&mut self, k: usize, cut: f64) {
        let full = 1u32 << (2 * self.shift);
        let mut s = 1usize;
        while s * 2 <= BLOCK_MAX {
            let s2 = s * 2;
            let mut row = 0;
            while row + s2 <= self.rows {
                let mut col = 0;
                while col + s2 <= self.cols {
                    let a = row * self.cols + col;
                    let quads = [a, a + s, a + s * self.cols, a + s * self.cols + s];
                    let ok = quads.iter().all(|&q| {
                        self.anchor[q][k] as usize == q
                            && self.bsize[q][k] as usize == s
                            && !self.active[q][k]
                            && !self.void[q]
                            && !self.walled[q]
                            && self.open[q] == full
                    });
                    if ok {
                        let means = quads.map(|q| self.mean[q][k]);
                        let hi = means.iter().cloned().fold(f64::MIN, f64::max);
                        let lo = means.iter().cloned().fold(f64::MAX, f64::min);
                        if hi <= cut || hi - lo <= FLAT * hi {
                            let m = means.iter().sum::<f64>() / 4.0;
                            for dy in 0..s2 {
                                for dx in 0..s2 {
                                    let n = (row + dy) * self.cols + col + dx;
                                    self.anchor[n][k] = a as u32;
                                    self.bsize[n][k] = 1;
                                    self.mean[n][k] = m;
                                    self.avg[n][k] = m;
                                }
                            }
                            self.bsize[a][k] = s2 as u8;
                        }
                    }
                    col += s2;
                }
                row += s2;
            }
            s = s2;
        }
    }

    /// Coarse to fine: the node's cells take its mean and carry the
    /// field from now on (its block, if it was in one, breaks up first).
    fn refine(&mut self, k: usize, node: usize, cells: &mut [Cell], width: usize) {
        if self.active[node][k] {
            return;
        }
        let a = self.anchor[node][k] as usize;
        if self.bsize[a][k] > 1 {
            self.split(k, a);
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
    /// from the means of the neighbours they exchange with (one-sided
    /// where there is none), per cell; none in a node with walls, whose
    /// open ground is no plane.
    fn gradients(&mut self, k: usize) {
        let side = (1usize << self.shift) as f64;
        for node in 0..self.cols * self.rows {
            if self.active[node][k] || self.void[node] || self.walled[node] {
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
            let at = |side: usize| -> Option<usize> {
                let n = self.nb[node][side];
                (n != u32::MAX && self.pairs[node][side] > 0 && !self.void[n as usize])
                    .then_some(n as usize)
            };
            let gx = along(at(3), at(1), self);
            let gy = along(at(0), at(2), self);
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
    /// The cell beyond each portal edge, at every depth, to the portals
    /// and edges (0 for `a`, 1 for `b`) it lies beyond; at a corner two
    /// edges' reaches overlap, and the nearer edge takes a point.
    portal_index: std::collections::HashMap<Position, Vec<(u16, u8)>>,
    /// The cells paired through portals, by index.
    portal_pairs: Vec<(usize, usize)>,
    /// The cells that have a portal on a side.
    portal_cells: std::collections::HashSet<Position>,
    /// Per cell, the slope it lies on (`u8::MAX` for none).
    slope_of: Vec<u8>,
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
            portal_index: std::collections::HashMap::new(),
            portal_pairs: Vec::new(),
            portal_cells: std::collections::HashSet::new(),
            slope_of: Vec::new(),
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
        world.lay_portals_and_slopes();
        world.grain = Grain::build(&world.config, &world.cells, &world.params);
        world
    }

    /// Index the portals' edges and the slopes' cells.
    fn lay_portals_and_slopes(&mut self) {
        let portals = self.config.portals.clone();
        for (i, portal) in portals.iter().enumerate() {
            assert_eq!(
                portal.a.len(),
                portal.b.len(),
                "a portal joins two edges of the same length"
            );
            let (ca, cb) = (portal.a.cells(), portal.b.cells());
            for (edge, cells, which) in [(&portal.a, &ca, 0u8), (&portal.b, &cb, 1u8)] {
                let (dx, dy) = edge.side.offset();
                for &c in cells {
                    self.portal_cells.insert(c);
                    for depth in 1..=PORTAL_DEPTH {
                        let beyond = Position::new(c.x + dx * depth, c.y + dy * depth);
                        let entry = self.portal_index.entry(beyond).or_default();
                        if !entry.contains(&(i as u16, which)) {
                            entry.push((i as u16, which));
                        }
                    }
                }
            }
            for (&pa, &pb) in ca.iter().zip(&cb) {
                if let (Some(ia), Some(ib)) = (self.index(pa), self.index(pb)) {
                    self.portal_pairs.push((ia, ib));
                }
            }
        }
        let n = self.cells.len();
        self.slope_of = vec![u8::MAX; n];
        let slopes = self.config.slopes.clone();
        for (i, s) in slopes.iter().enumerate().take(u8::MAX as usize) {
            let (w, h) = (self.config.width as i32, self.config.height as i32);
            for y in s.rect.min.y.max(0)..=s.rect.max.y.min(h - 1) {
                for x in s.rect.min.x.max(0)..=s.rect.max.x.min(w - 1) {
                    self.slope_of[y as usize * self.config.width + x as usize] = i as u8;
                }
            }
        }
    }

    /// Where a point beyond a portal's edge comes out on the other side,
    /// and the turn made, if it lies within reach of one.
    pub fn warp(&self, p: Point) -> Option<Warp> {
        let candidates = self.portal_index.get(&p.cell())?;
        let mut best: Option<(f64, Warp)> = None;
        for &(i, which) in candidates {
            let portal = &self.config.portals[i as usize];
            let (from, to) = if which == 0 {
                (&portal.a, &portal.b)
            } else {
                (&portal.b, &portal.a)
            };
            let (along, depth) = from.coords(p);
            if along < 0.0
                || along >= from.len() as f64
                || depth <= 0.0
                || depth > PORTAL_DEPTH as f64
            {
                continue;
            }
            if best.map(|(d, _)| depth < d).unwrap_or(true) {
                let turn = crate::geometry::wrap_angle(
                    to.side.outward() + std::f64::consts::PI - from.side.outward(),
                );
                best = Some((
                    depth,
                    Warp {
                        point: to.point_inside(along, depth),
                        turn,
                    },
                ));
            }
        }
        best.map(|(_, w)| w)
    }

    /// Whether a cell has a portal on one of its sides.
    pub fn has_portal(&self, p: Position) -> bool {
        self.portal_cells.contains(&p)
    }

    /// The cells joined through portals, in pairs.
    pub fn portal_cell_pairs(&self) -> Vec<(Position, Position)> {
        let w = self.config.width;
        self.portal_pairs
            .iter()
            .map(|&(a, b)| {
                (
                    Position::new((a % w) as i32, (a / w) as i32),
                    Position::new((b % w) as i32, (b / w) as i32),
                )
            })
            .collect()
    }

    /// Whether one point is in sight of another: nothing but open ground
    /// on the line between them, sampled every half cell (a sight, not a
    /// body: no clearance is needed).
    pub fn line_of_sight(&self, a: Point, b: Point) -> bool {
        if !self.has_walls {
            return self.is_passable_point(b);
        }
        let len = a.distance(b);
        let samples = (len / 0.5).ceil().max(1.0) as usize;
        for i in 1..=samples {
            let t = i as f64 / samples as f64;
            let p = Point::new(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y));
            if !self.is_passable(p.cell()) {
                return false;
            }
        }
        true
    }

    /// Speed factor for walking from a cell on a heading: less than one
    /// when climbing a slope, one on the flat or downhill.
    pub fn climb_factor(&self, p: Position, heading: f64) -> f64 {
        let Some(idx) = self.index(p) else {
            return 1.0;
        };
        let s = self.slope_of[idx];
        if s == u8::MAX {
            return 1.0;
        }
        let slope = &self.config.slopes[s as usize];
        let up = (heading - slope.up.outward()).cos();
        if up > 0.0 {
            1.0 - (1.0 - slope.factor.clamp(0.0, 1.0)) * up
        } else {
            1.0
        }
    }

    /// The slope a cell lies on, if any.
    pub fn slope_at(&self, p: Position) -> Option<&Slope> {
        let idx = self.index(p)?;
        let s = *self.slope_of.get(idx)?;
        if s == u8::MAX {
            None
        } else {
            self.config.slopes.get(s as usize)
        }
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
            let centre = Point::center_of(l);
            let d = from.distance(centre);
            if d <= sight
                && best.map(|b| d < b.2).unwrap_or(true)
                && self.line_of_sight(from, centre)
            {
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
        let (w, h) = (self.config.width as i32, self.config.height as i32);
        // Rectangles are laid one by one rather than looked up cell by
        // cell, so a shape drawn from many of them costs their area.
        let fill = |cells: &mut Vec<Cell>, r: &Rect, terrain: Terrain| {
            for y in r.min.y.max(0)..=r.max.y.min(h - 1) {
                for x in r.min.x.max(0)..=r.max.x.min(w - 1) {
                    cells[y as usize * w as usize + x as usize].terrain = terrain;
                }
            }
        };
        if !open.is_empty() {
            for c in self.cells.iter_mut() {
                c.terrain = Terrain::Wall;
            }
            for o in &open {
                fill(&mut self.cells, o, Terrain::Open);
            }
        }
        for wall in &walls {
            fill(&mut self.cells, wall, Terrain::Wall);
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
        if let Some(g) = self.grain.as_mut() {
            for s in g.sources.iter_mut() {
                *s = 0;
            }
            let w = self.config.width;
            for &i in &self.food_cells {
                let node = g.node_of(i % w, i / w);
                g.sources[node] += 1;
            }
        }
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
                    self.deposit(p, Pheromone::Odour, emission * tick_s);
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
            || (!self.portal_index.is_empty()
                && self
                    .warp(p)
                    .map(|w| self.is_passable(w.point.cell()))
                    .unwrap_or(false))
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

    /// Concentration of a channel at a cell (0 outside the grid and in
    /// a wall).
    pub fn level(&self, p: Position, kind: Pheromone) -> f64 {
        match self.index(p) {
            Some(idx) if self.cells[idx].terrain != Terrain::Wall => {
                self.read(idx, p, kind.index())
            }
            _ => 0.0,
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
            _ if !self.portal_index.is_empty() => match self.warp(p) {
                Some(w) => {
                    let cell = w.point.cell();
                    match self.index(cell) {
                        Some(idx) if self.cells[idx].terrain != Terrain::Wall => {
                            self.read(idx, cell, kind.index())
                        }
                        _ => 0.0,
                    }
                }
                None => 0.0,
            },
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
        if self.cells[idx].terrain == Terrain::Wall {
            return;
        }
        if let Some(g) = self.grain.as_mut() {
            if g.volatile[k] {
                // Into the pool of the node (its block breaking up
                // first): volumetric information has no cell.
                let node = g.node_of(p.x as usize, p.y as usize);
                let a = g.anchor[node][k] as usize;
                if g.bsize[a][k] > 1 {
                    g.split(k, a);
                }
                let open = g.open[node].max(1) as f64;
                let m = (g.mean[node][k] + amount / open).min(cap);
                g.mean[node][k] = m;
                g.avg[node][k] = m;
                self.present[k] = true;
                return;
            }
        }
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
        // Through the portals: the share that stayed at a blocked side
        // crosses to the cell joined to it.
        for &(a, b) in &self.portal_pairs {
            for &k in &active {
                let d = diffusion[k];
                if d <= 0.0 {
                    continue;
                }
                let sa = self.cells[a].pheromone[k] * retention[k] * d / 4.0;
                let sb = self.cells[b].pheromone[k] * retention[k] * d / 4.0;
                self.scratch[a][k] += sb - sa;
                self.scratch[b][k] += sa - sb;
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
        let mut cut = [0.0; Pheromone::COUNT];
        for (c, p) in cut.iter_mut().zip(self.params.iter()) {
            *c = 1e-9 * p.k.max(1e-9);
        }
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
                if g.active[node][k] {
                    let (x0, y0, x1, y1) = g.rect(node);
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
                                    if g.walled[nnode] && self.cells[nidx].terrain == Terrain::Wall
                                    {
                                        continue;
                                    }
                                    if g.active[nnode][k] {
                                        self.scratch[nidx][k] += share;
                                    } else {
                                        g.inflow[g.anchor[nnode][k] as usize] += share;
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
                    // A coarse block, swept at its anchor: one pool for
                    // all its nodes.
                    if g.void[node] || g.anchor[node][k] as usize != node {
                        continue;
                    }
                    let mean = g.mean[node][k];
                    if mean <= 0.0 {
                        continue;
                    }
                    let s = g.bsize[node][k] as usize;
                    let mut pool = mean * g.block_open(node, k) as f64 * r;
                    if d > 0.0 {
                        // Each open cell along a side sends its share
                        // across to an open cell facing it: to the cells
                        // of an active node, to the pool of a coarse one.
                        let per = mean * r * d / 4.0;
                        let (col, row) = (node % g.cols, node / g.cols);
                        for side in 0..4u8 {
                            for i in 0..s {
                                let m = match side {
                                    0 => row * g.cols + col + i,
                                    1 => (row + i) * g.cols + col + s - 1,
                                    2 => (row + s - 1) * g.cols + col + i,
                                    _ => (row + i) * g.cols + col,
                                };
                                let nb = g.nb[m][side as usize];
                                if nb == u32::MAX || g.pairs[m][side as usize] == 0 {
                                    continue;
                                }
                                let nb = nb as usize;
                                if g.active[nb][k] {
                                    let (x0, y0, x1, y1) = g.rect(m);
                                    let (bx0, by0, bx1, by1) = g.rect(nb);
                                    let (ax0, ay0, ax1, ay1) = match side {
                                        0 => (x0.max(bx0), by1 - 1, x1.min(bx1), by1),
                                        1 => (bx0, y0.max(by0), bx0 + 1, y1.min(by1)),
                                        2 => (x0.max(bx0), by0, x1.min(bx1), by0 + 1),
                                        _ => (bx1 - 1, y0.max(by0), bx1, y1.min(by1)),
                                    };
                                    let walled = g.walled[m];
                                    for y in ay0..ay1 {
                                        for x in ax0..ax1 {
                                            let nidx = y * w + x;
                                            if self.cells[nidx].terrain == Terrain::Wall {
                                                continue;
                                            }
                                            if walled {
                                                // The cell of this node facing
                                                // it must be open too.
                                                let (mx, my) = match side {
                                                    0 => (x, y0),
                                                    1 => (x1 - 1, y),
                                                    2 => (x, y1 - 1),
                                                    _ => (x0, y),
                                                };
                                                if self.cells[my * w + mx].terrain == Terrain::Wall
                                                {
                                                    continue;
                                                }
                                            }
                                            self.scratch[nidx][k] += per;
                                            pool -= per;
                                        }
                                    }
                                } else {
                                    let pairs = g.pairs[m][side as usize] as f64;
                                    g.inflow[g.anchor[nb][k] as usize] += per * pairs;
                                    pool -= per * pairs;
                                }
                            }
                        }
                    }
                    g.pool[node] = pool;
                }
            }
            // Through the portals: the share that stayed at a blocked
            // side crosses to the cell joined to it, cell to cell, cell
            // to pool or pool to cell as the nodes are.
            if d > 0.0 {
                for &(a, b) in &self.portal_pairs {
                    let (na, nb) = (g.node_of(a % w, a / w), g.node_of(b % w, b / w));
                    for (from, to, nf, nt) in [(a, b, na, nb), (b, a, nb, na)] {
                        let value = if g.active[nf][k] {
                            self.cells[from].pheromone[k]
                        } else {
                            g.mean[nf][k]
                        };
                        let share = value * r * d / 4.0;
                        if share <= 0.0 {
                            continue;
                        }
                        if g.active[nf][k] {
                            self.scratch[from][k] -= share;
                        } else {
                            let anchor = g.anchor[nf][k] as usize;
                            g.pool[anchor] -= share;
                        }
                        if g.active[nt][k] {
                            self.scratch[to][k] += share;
                        } else {
                            let anchor = g.anchor[nt][k] as usize;
                            g.inflow[anchor] += share;
                            if value >= threshold {
                                g.touched[nt] = true;
                            }
                        }
                    }
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
                    if g.void[node] || g.anchor[node][k] as usize != node {
                        continue;
                    }
                    let open = g.block_open(node, k) as f64;
                    let m = if open > 0.0 {
                        (g.pool[node] + g.inflow[node]) / open
                    } else {
                        0.0
                    };
                    // A coarse block below a billionth of the perception
                    // constant is emptied, so the sweep skips it.
                    let m = if m < cut[k] { 0.0 } else { m.min(cap) };
                    if m > 0.0 {
                        present[k] = true;
                    }
                    // Every node of the block takes the mean; a node the
                    // front has reached is refined, and a block lifted
                    // to the threshold breaks up.
                    let s = g.bsize[node][k] as usize;
                    let (col, row) = (node % g.cols, node / g.cols);
                    let mut touched = false;
                    for dy in 0..s {
                        for dx in 0..s {
                            let n = (row + dy) * g.cols + col + dx;
                            g.mean[n][k] = m;
                            g.avg[n][k] = m;
                            touched |= g.touched[n];
                        }
                    }
                    if touched || m >= threshold {
                        if s > 1 {
                            g.split(k, node);
                        }
                        for dy in 0..s {
                            for dx in 0..s {
                                let n = (row + dy) * g.cols + col + dx;
                                if g.touched[n] || m >= threshold {
                                    g.refine(k, n, &mut self.cells, w);
                                }
                            }
                        }
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
                    // A node whose open ground is in pieces is left at
                    // cell resolution: one mean would mix the pieces.
                    if !g.active[node][k] || g.has_sources(node, k) || g.split[node] {
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
                g.merge(k, cut[k]);
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
        let (x0, y0, x1, y1) = rect;
        let area = (x1.saturating_sub(x0)) * (y1.saturating_sub(y0));
        // A node with a portal's edge in it is refined to cells, so that
        // the hop through the portal is a cell's.
        if area > 1
            && self.portal_cells.iter().any(|c| {
                c.x >= x0 as i32 && c.y >= y0 as i32 && (c.x as usize) < x1 && (c.y as usize) < y1
            })
        {
            return crate::lens::Ground::Mixed;
        }
        let Some(tree) = &self.wall_tree else {
            return crate::lens::Ground::Open(1.0);
        };
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
                let n = g.void.iter().filter(|v| !**v).count();
                g.active.iter().filter(|a| a[kind.index()]).count() as f64 / n.max(1) as f64
            }
        }
    }

    /// What the sweep of a channel visits: its active nodes and its
    /// coarse blocks (every cell when every cell is always active).
    pub fn kinetics_leaves(&self, kind: Pheromone) -> usize {
        let k = kind.index();
        match &self.grain {
            None => self.cells.len(),
            Some(g) => (0..g.cols * g.rows)
                .filter(|&n| !g.void[n] && (g.active[n][k] || g.anchor[n][k] as usize == n))
                .count(),
        }
    }

    /// Share of the grain's nodes with no open cell at all, which the
    /// kinetics never visit (0 when every cell is always active).
    pub fn void_share(&self) -> f64 {
        match &self.grain {
            None => 0.0,
            Some(g) => {
                let n = g.cols * g.rows;
                g.void.iter().filter(|v| **v).count() as f64 / n.max(1) as f64
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
        // Within a millionth: a coarse node below a billionth of the
        // perception constant is emptied.
        let total = world.total_pheromone(Pheromone::Trail);
        assert!((total - 40.0).abs() < 1e-3, "mass conserved: {total}");
        // The far field is carried as means, and reads as such.
        let share = world.active_share(Pheromone::Trail);
        assert!(share < 1.0, "the far field stays coarse: {share}");
        assert!(world.level(Position::new(60, 60), Pheromone::Trail) > 0.0);
        let tree = world.field_tree(Pheromone::Trail);
        assert!((tree.get(QuadKey::ROOT) - total).abs() < 1e-9);
        let level = world.grain_level().unwrap();
        let sum: f64 = tree.nodes(level).map(|(_, v)| *v).sum();
        assert!((sum - total).abs() < 1e-9);
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
    fn a_shaped_arena_costs_its_open_ground() {
        // A ring of open ground in a walled square: the field spreads
        // round the ring without leaking into the walls, nodes with
        // walls in them coarsen like any other once faint, and nodes
        // with no open cell are never visited.
        let mut open = Vec::new();
        let (cx, cy) = (32i32, 32i32);
        for y in 0..64i32 {
            let dy = (y - cy) as f64;
            let ho = (28.0f64.powi(2) - dy * dy).max(0.0).sqrt() as i32;
            if dy.abs() < 14.0 {
                let hi = (14.0f64.powi(2) - dy * dy).max(0.0).sqrt() as i32;
                open.push(Rect::new(
                    Position::new(cx - ho, y),
                    Position::new(cx - hi - 1, y),
                ));
                open.push(Rect::new(
                    Position::new(cx + hi + 1, y),
                    Position::new(cx + ho, y),
                ));
            } else if ho > 0 {
                open.push(Rect::new(
                    Position::new(cx - ho, y),
                    Position::new(cx + ho, y),
                ));
            }
        }
        let cfg = WorldConfig {
            width: 64,
            height: 64,
            nest: Position::new(32, 10),
            nest_radius: 1,
            random_food: None,
            open,
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
        assert!(
            world.void_share() > 0.1,
            "the hole and the corners are void"
        );
        world.deposit(Position::new(32, 12), Pheromone::Trail, 40.0);
        for _ in 0..400 {
            world.step_pheromones();
        }
        let total = world.total_pheromone(Pheromone::Trail);
        assert!(
            (total - 40.0).abs() < 1e-3,
            "mass conserved round the ring: {total}"
        );
        for (idx, c) in world.cells().iter().enumerate() {
            if c.terrain == Terrain::Wall {
                let p = Position::new((idx % 64) as i32, (idx / 64) as i32);
                assert_eq!(
                    world.level(p, Pheromone::Trail),
                    0.0,
                    "nothing in a wall at {p:?}"
                );
                assert_eq!(c.pheromone[Pheromone::Trail.index()], 0.0);
            }
        }
        // It went round the bend: a quarter of the way round the ring
        // carries something (the nodes with walls in them keep cell
        // resolution, so the field spreads through them no faster than
        // diffusion).
        assert!(world.level(Position::new(52, 32), Pheromone::Trail) > 0.0);
        assert!(world.level(Position::new(12, 32), Pheromone::Trail) > 0.0);
        assert!(world.active_share(Pheromone::Trail) < 1.0);
    }

    #[test]
    fn portals_join_edges_turn_bodies_and_carry_the_field() {
        use std::f64::consts::FRAC_PI_2;
        let make = |open: Vec<Rect>, portal: Portal, grain: usize| {
            World::new(
                WorldConfig {
                    width: 40,
                    height: 40,
                    nest: Position::new(4, 8),
                    nest_radius: 1,
                    random_food: None,
                    open,
                    portals: vec![portal],
                    kinetics_grain: grain,
                    pheromones: Some({
                        let mut set = Species::lasius_niger().pheromones();
                        set[Pheromone::Trail.index()].half_life_s = f64::INFINITY;
                        set[Pheromone::Trail.index()].diffusion_per_s = 0.2;
                        set
                    }),
                    ..WorldConfig::default()
                },
                &mut Rng::seed_from_u64(1),
            )
        };
        // Two regions with twelve cells of wall between, joined straight
        // across.
        let straight = Portal {
            a: Edge {
                start: Position::new(15, 0),
                end: Position::new(15, 39),
                side: Side::East,
            },
            b: Edge {
                start: Position::new(28, 0),
                end: Position::new(28, 39),
                side: Side::West,
            },
        };
        let two = vec![
            Rect::new(Position::new(0, 0), Position::new(15, 39)),
            Rect::new(Position::new(28, 0), Position::new(39, 39)),
        ];
        let world = make(two.clone(), straight, 0);
        let w = world.warp(Point::new(16.3, 5.5)).expect("through");
        assert!((w.point.x - 28.3).abs() < 1e-9 && (w.point.y - 5.5).abs() < 1e-9);
        assert!(w.turn.abs() < 1e-9, "no turn straight across");
        let back = world.warp(Point::new(27.7, 5.5)).expect("and back");
        assert!((back.point.x - 15.7).abs() < 1e-9 && back.turn.abs() < 1e-9);
        assert!(
            world.is_passable_point(Point::new(16.3, 5.5)),
            "the way through"
        );
        assert!(
            !world.is_passable_point(Point::new(22.0, 5.5)),
            "not the wall beyond the portal's reach"
        );
        assert!(world.warp(Point::new(16.3, 45.0)).is_none(), "off the run");
        assert!(world.has_portal(Position::new(15, 7)) && !world.has_portal(Position::new(14, 7)));
        // The field flows through, on either sweep, and never into the wall.
        for grain in [0, 8] {
            let mut world = make(two.clone(), straight, grain);
            world.deposit(Position::new(12, 20), Pheromone::Trail, 40.0);
            for _ in 0..300 {
                world.step_pheromones();
            }
            let total = world.total_pheromone(Pheromone::Trail);
            // Within a hundredth: the dense sweep drops a trace below a
            // millionth of a unit, and the front spreads over many cells.
            assert!(
                (total - 40.0).abs() < 1e-2,
                "conserved through the portal: {total}"
            );
            let far = world.pheromone_in(
                &Rect::new(Position::new(28, 0), Position::new(39, 39)),
                Pheromone::Trail,
            );
            assert!(far > 1.0, "the far region receives (grain {grain}): {far}");
            assert_eq!(world.level(Position::new(22, 20), Pheromone::Trail), 0.0);
        }
        // Round a corner: the first region's east edge meets the second's
        // north edge, a quarter turn clockwise on the way through.
        let corner = Portal {
            a: Edge {
                start: Position::new(15, 0),
                end: Position::new(15, 15),
                side: Side::East,
            },
            b: Edge {
                start: Position::new(24, 30),
                end: Position::new(39, 30),
                side: Side::North,
            },
        };
        assert!((corner.turn() - FRAC_PI_2).abs() < 1e-9);
        let world = make(
            vec![
                Rect::new(Position::new(0, 0), Position::new(15, 15)),
                Rect::new(Position::new(24, 30), Position::new(39, 39)),
            ],
            corner,
            0,
        );
        let w = world.warp(Point::new(16.5, 3.0)).expect("round the corner");
        assert!((w.point.x - 27.0).abs() < 1e-9 && (w.point.y - 30.5).abs() < 1e-9);
        assert!(
            (w.turn - FRAC_PI_2).abs() < 1e-9,
            "a quarter turn: {}",
            w.turn
        );
        let back = world.warp(Point::new(27.0, 29.5)).expect("and back");
        assert!((back.point.x - 15.5).abs() < 1e-9 && (back.point.y - 3.0).abs() < 1e-9);
        assert!((back.turn + FRAC_PI_2).abs() < 1e-9);
    }

    #[test]
    fn walls_block_sight_of_landmarks() {
        // A landmark behind the wall of the small world is not seen from
        // across it, and is from the same side.
        let cfg = WorldConfig {
            landmarks: vec![Position::new(11, 4)],
            ..small_config()
        };
        let world = World::new(cfg, &mut Rng::seed_from_u64(1));
        assert!(world.nearest_landmark(Point::new(7.5, 4.5), 20.0).is_none());
        assert!(world
            .nearest_landmark(Point::new(10.5, 2.5), 20.0)
            .is_some());
        assert!(!world.line_of_sight(Point::new(7.5, 4.5), Point::new(11.5, 4.5)));
        assert!(world.line_of_sight(Point::new(2.5, 2.5), Point::new(7.5, 7.5)));
    }

    #[test]
    fn slopes_slow_climbing() {
        use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
        let cfg = WorldConfig {
            slopes: vec![Slope {
                rect: Rect::new(Position::new(0, 0), Position::new(11, 4)),
                up: Side::North,
                factor: 0.5,
                slip: 0.0,
            }],
            ..small_config()
        };
        let world = World::new(cfg, &mut Rng::seed_from_u64(1));
        let p = Position::new(3, 2);
        assert!(
            (world.climb_factor(p, -FRAC_PI_2) - 0.5).abs() < 1e-9,
            "straight up"
        );
        assert!(
            (world.climb_factor(p, FRAC_PI_2) - 1.0).abs() < 1e-9,
            "down is free"
        );
        assert!(
            (world.climb_factor(p, 0.0) - 1.0).abs() < 1e-9,
            "along the level"
        );
        let diagonal = 1.0 - 0.5 * FRAC_PI_4.cos();
        assert!((world.climb_factor(p, -FRAC_PI_4) - diagonal).abs() < 1e-9);
        assert!((world.climb_factor(Position::new(3, 8), -FRAC_PI_2) - 1.0).abs() < 1e-9);
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
        // Five microlitres and thirty milligrams count, into the pools of
        // the sources' nodes (the smell is volumetric: it has no cell).
        let total = world.total_pheromone(Pheromone::Odour);
        assert!((total - 17.5).abs() < 1e-9, "what was given off: {total}");
        assert!(world.level(Position::new(2, 7), Pheromone::Odour) > 0.0);
        assert!(world.level(Position::new(7, 7), Pheromone::Odour) > 0.0);
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
