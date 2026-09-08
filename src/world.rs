//! The grid world: terrain, food, nest, pheromone fields, and counters.
//!
//! The grid has a physical scale (`cell_cm`) and the simulation a time step
//! (`tick_s`), so species parameters in centimetres and seconds convert to
//! cells and ticks. Each cell carries every [`Pheromone`] channel; the
//! channels evaporate by first-order kinetics from their half-lives and
//! diffuse a little to orthogonal neighbours each tick.

use crate::geometry::{Direction, Position};
use crate::pheromone::{Pheromone, PheromoneParams, PheromoneSet};
use crate::rng::Rng;
use crate::species::Species;

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
    /// Crop loads of food lying here.
    pub food: u32,
    /// Quality of that food in `0..=1` (sucrose concentration relative to
    /// the most attractive solution).
    pub quality: f32,
    /// Concentration of every pheromone channel, indexed by
    /// [`Pheromone::index`].
    pub pheromone: [f64; Pheromone::COUNT],
    /// Number of living ants currently on the cell (ants inside the nest are
    /// not on the grid).
    pub occupancy: u16,
}

impl Cell {
    /// Concentration of one channel.
    pub fn level(&self, kind: Pheromone) -> f64 {
        self.pheromone[kind.index()]
    }
}

/// A hand-placed cluster of food.
#[derive(Clone, Debug, PartialEq)]
pub struct FoodSource {
    /// Cluster centre.
    pub center: Position,
    /// Chebyshev radius of the cluster.
    pub radius: i32,
    /// Crop loads placed on every cell of the cluster.
    pub amount_per_cell: u32,
    /// Quality in `0..=1`.
    pub quality: f64,
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

/// Randomly placed food clusters, generated when the world is built.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomFood {
    /// Number of clusters.
    pub clusters: usize,
    /// Chebyshev radius of each cluster.
    pub radius: i32,
    /// Crop loads per cell.
    pub amount_per_cell: u32,
    /// Minimum Chebyshev distance between a cluster centre and the nest.
    pub min_distance_from_nest: i32,
    /// Quality range the clusters are drawn from.
    pub quality: (f64, f64),
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
                amount_per_cell: 25,
                min_distance_from_nest: 12,
                quality: (0.4, 1.0),
            }),
            walls: Vec::new(),
            open: Vec::new(),
            counters: Vec::new(),
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
    /// Ant moves that ended inside the region.
    pub crossings: u64,
}

/// The simulated environment.
#[derive(Clone, Debug)]
pub struct World {
    config: WorldConfig,
    cells: Vec<Cell>,
    scratch: Vec<[f64; Pheromone::COUNT]>,
    params: PheromoneSet,
    retention: [f64; Pheromone::COUNT],
    diffusion: [f64; Pheromone::COUNT],
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
            retention,
            diffusion,
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
    }

    fn place_food(&mut self, src: &FoodSource) {
        for dy in -src.radius..=src.radius {
            for dx in -src.radius..=src.radius {
                if let Some(c) = self.cell_mut(src.center.offset(dx, dy)) {
                    if c.terrain == Terrain::Open {
                        c.food = c.food.saturating_add(src.amount_per_cell);
                        c.quality = src.quality.clamp(0.0, 1.0) as f32;
                    }
                }
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
                    .map(|c| c.terrain != Terrain::Open || c.food > 0)
                    .unwrap_or(true)
                {
                    continue;
                }
                let quality = rng.range(random.quality.0, random.quality.1);
                self.place_food(&FoodSource {
                    center,
                    radius: r,
                    amount_per_cell: random.amount_per_cell,
                    quality,
                });
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

    /// Concentration of a channel at `p` (0 outside the grid).
    pub fn level(&self, p: Position, kind: Pheromone) -> f64 {
        self.cell(p).map(|c| c.level(kind)).unwrap_or(0.0)
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

    /// Record an ant move ending at `p` in every counter covering it.
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

    /// Total food remaining on the ground.
    pub fn total_food(&self) -> u64 {
        self.cells.iter().map(|c| c.food as u64).sum()
    }

    /// Total amount of one channel over the grid.
    pub fn total_pheromone(&self, kind: Pheromone) -> f64 {
        self.cells.iter().map(|c| c.level(kind)).sum()
    }

    /// Food remaining inside a region.
    pub fn food_in(&self, rect: &Rect) -> u64 {
        let mut total = 0;
        for y in rect.min.y..=rect.max.y {
            for x in rect.min.x..=rect.max.x {
                total += self
                    .cell(Position::new(x, y))
                    .map(|c| c.food as u64)
                    .unwrap_or(0);
            }
        }
        total
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
            food_sources: vec![FoodSource {
                center: Position::new(1, 1),
                radius: 1,
                amount_per_cell: 5,
                quality: 0.7,
            }],
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
        assert_eq!(world.total_food(), 45);
        assert_eq!(world.cell(Position::new(1, 1)).unwrap().quality, 0.7);
        assert_eq!(world.nest_cells().len(), 9);
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
    fn random_food_is_reproducible_from_config_seed() {
        let cfg = WorldConfig {
            seed: Some(99),
            ..WorldConfig::default()
        };
        let mut r1 = Rng::seed_from_u64(1);
        let mut r2 = Rng::seed_from_u64(2);
        let a = World::new(cfg.clone(), &mut r1);
        let b = World::new(cfg, &mut r2);
        let fa: Vec<(u32, f32)> = a.cells().iter().map(|c| (c.food, c.quality)).collect();
        let fb: Vec<(u32, f32)> = b.cells().iter().map(|c| (c.food, c.quality)).collect();
        assert_eq!(fa, fb);
        assert!(a.total_food() > 0);
        assert!(a.cells().iter().any(|c| c.food > 0 && c.quality >= 0.4));
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
    }
}
