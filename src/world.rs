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

use crate::geometry::{Direction, Point, Position};
use crate::pheromone::{Pheromone, PheromoneParams, PheromoneSet};
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
            nest_radius: 2,
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
            capacity_zones: Vec::new(),
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

/// The simulated environment.
#[derive(Clone, Debug)]
pub struct World {
    config: WorldConfig,
    cells: Vec<Cell>,
    scratch: Vec<[f64; Pheromone::COUNT]>,
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
            params,
            base_retention: retention,
            retention,
            diffusion,
            evaporation_factor: 1.0,
            counters,
            config,
        };
        world.lay_terrain();
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
        world
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
    }

    /// Refill renewing sources by one tick.
    pub fn step_food(&mut self) {
        let tick_s = self.config.tick_s;
        for c in self.cells.iter_mut() {
            if c.renewal_ul_per_s > 0.0 && c.food_ul < c.food_capacity_ul {
                c.food_ul = (c.food_ul + c.renewal_ul_per_s * tick_s).min(c.food_capacity_ul);
            }
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

    /// Concentration of a channel at a cell (0 outside the grid).
    pub fn level(&self, p: Position, kind: Pheromone) -> f64 {
        self.cell(p).map(|c| c.level(kind)).unwrap_or(0.0)
    }

    /// Concentration of a channel on the substrate patch (cell) under a
    /// continuous point; walls and points off the grid read zero. Marks
    /// are laid on the substrate cell by cell, and an antenna touching a
    /// patch senses that patch: at a trail fork this keeps the two branch
    /// trails distinct instead of blending them with the heavily marked
    /// junction (see [`World::sample`] for the smooth field).
    pub fn level_at(&self, p: Point, kind: Pheromone) -> f64 {
        self.cell(p.cell())
            .filter(|c| c.terrain != Terrain::Wall)
            .map(|c| c.level(kind))
            .unwrap_or(0.0)
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
            if let Some(c) = self.cell(cell) {
                if c.terrain != Terrain::Wall {
                    total += w * c.level(kind);
                }
            }
        }
        total
    }

    /// Add pheromone of one channel to a cell (clamped to the channel's cap;
    /// inert channels take nothing).
    pub fn deposit(&mut self, p: Position, kind: Pheromone, amount: f64) {
        let cap = self.params[kind.index()].cap;
        if amount <= 0.0 || cap <= 0.0 {
            return;
        }
        if let Some(c) = self.cell_mut(p) {
            let v = &mut c.pheromone[kind.index()];
            *v = (*v + amount).min(cap);
        }
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
        for s in self.scratch.iter_mut() {
            *s = [0.0; Pheromone::COUNT];
        }
        let active: Vec<usize> = (0..Pheromone::COUNT)
            .filter(|&k| self.retention[k] > 0.0)
            .collect();
        for y in 0..h {
            for x in 0..w {
                let idx = y as usize * w as usize + x as usize;
                let cell = &self.cells[idx];
                if cell.terrain == Terrain::Wall {
                    continue;
                }
                for &k in &active {
                    let amount = cell.pheromone[k] * self.retention[k];
                    if amount <= 0.0 {
                        continue;
                    }
                    let diffusion = self.diffusion[k];
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
        for (cell, s) in self.cells.iter_mut().zip(self.scratch.iter()) {
            for ((slot, &v), params) in cell.pheromone.iter_mut().zip(s).zip(self.params.iter()) {
                *slot = if v < 1e-6 { 0.0 } else { v.min(params.cap) };
            }
        }
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
        self.cells.iter().map(|c| c.level(kind)).sum()
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
        let mut rng = Rng::seed_from_u64(1);
        let mut world = World::new(small_config(), &mut rng);
        world.deposit(Position::new(8, 4), Pheromone::Trail, 10.0);
        world.step_pheromones();
        assert_eq!(world.level(Position::new(9, 4), Pheromone::Trail), 0.0);
        assert!(world.level(Position::new(7, 4), Pheromone::Trail) > 0.0);
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
        world.step_food();
        assert!((world.cell(c).unwrap().food_ul - 0.2).abs() < 1e-12);
        for _ in 0..10 {
            world.step_food();
        }
        assert!(
            (world.cell(c).unwrap().food_ul - 0.5).abs() < 1e-12,
            "capped at the standing volume"
        );
        // A plain pool does not refill.
        let (taken, _) = world.take_food(Position::new(0, 0), 5.0);
        assert!((taken - 5.0).abs() < 1e-12);
        world.step_food();
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
    fn capacity_zones_and_crowding() {
        let cfg = WorldConfig {
            cell_capacity: 8,
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
