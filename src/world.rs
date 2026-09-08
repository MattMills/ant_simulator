//! The grid world: terrain, food, nest, and the two pheromone fields.
//!
//! Ants communicate stigmergically. Foraging ants lay *home* pheromone
//! (pointing back to the nest), ants carrying food lay *food* pheromone
//! (pointing to a food source). Both fields evaporate and diffuse each tick.

use crate::geometry::{Direction, Position};
use crate::rng::Rng;

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
#[derive(Clone, Debug, Default)]
pub struct Cell {
    /// Terrain type.
    pub terrain: Terrain,
    /// Units of food lying here.
    pub food: u32,
    /// Pheromone laid by foraging ants ("this way home").
    pub home_pheromone: f64,
    /// Pheromone laid by ants carrying food ("this way to food").
    pub food_pheromone: f64,
    /// Number of living ants currently on the cell.
    pub occupancy: u16,
}

/// A hand-placed cluster of food.
#[derive(Clone, Debug, PartialEq)]
pub struct FoodSource {
    /// Cluster centre.
    pub center: Position,
    /// Chebyshev radius of the cluster.
    pub radius: i32,
    /// Food units placed on every cell of the cluster.
    pub amount_per_cell: u32,
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
    /// Food units per cell.
    pub amount_per_cell: u32,
    /// Minimum Chebyshev distance between a cluster centre and the nest.
    pub min_distance_from_nest: i32,
}

/// World construction parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldConfig {
    /// Grid width in cells.
    pub width: usize,
    /// Grid height in cells.
    pub height: usize,
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
    /// Fraction of pheromone lost per tick.
    pub evaporation: f64,
    /// Fraction of remaining pheromone spread to orthogonal neighbours per tick.
    pub diffusion: f64,
    /// Upper bound on pheromone per cell.
    pub pheromone_cap: f64,
    /// Seed for random food placement. `None` uses the simulation's generator,
    /// `Some` fixes the map independently of everything else.
    pub seed: Option<u64>,
}

impl Default for WorldConfig {
    fn default() -> Self {
        WorldConfig {
            width: 64,
            height: 40,
            nest: Position::new(32, 20),
            nest_radius: 2,
            food_sources: Vec::new(),
            random_food: Some(RandomFood {
                clusters: 3,
                radius: 2,
                amount_per_cell: 25,
                min_distance_from_nest: 12,
            }),
            walls: Vec::new(),
            evaporation: 0.02,
            diffusion: 0.05,
            pheromone_cap: 40.0,
            seed: None,
        }
    }
}

/// The simulated environment.
#[derive(Clone, Debug)]
pub struct World {
    config: WorldConfig,
    cells: Vec<Cell>,
    scratch: Vec<(f64, f64)>,
}

impl World {
    /// Build a world. Random food clusters are drawn from `rng` unless the
    /// config carries its own seed.
    pub fn new(config: WorldConfig, rng: &mut Rng) -> World {
        assert!(
            config.width > 0 && config.height > 0,
            "world must be non-empty"
        );
        let n = config.width * config.height;
        let mut world = World {
            cells: vec![Cell::default(); n],
            scratch: vec![(0.0, 0.0); n],
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
        for y in 0..self.config.height as i32 {
            for x in 0..self.config.width as i32 {
                let p = Position::new(x, y);
                if walls.iter().any(|w| w.contains(p)) {
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
                    .map(|c| c.terrain != Terrain::Open)
                    .unwrap_or(true)
                {
                    continue;
                }
                self.place_food(&FoodSource {
                    center,
                    radius: r,
                    amount_per_cell: random.amount_per_cell,
                });
                break;
            }
        }
    }

    /// Construction parameters.
    pub fn config(&self) -> &WorldConfig {
        &self.config
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

    /// Add pheromone to a cell (clamped to the cap).
    pub fn deposit(&mut self, p: Position, home: f64, food: f64) {
        let cap = self.config.pheromone_cap;
        if let Some(c) = self.cell_mut(p) {
            c.home_pheromone = (c.home_pheromone + home).min(cap);
            c.food_pheromone = (c.food_pheromone + food).min(cap);
        }
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

    /// Evaporate and diffuse both pheromone fields by one tick.
    pub fn step_pheromones(&mut self) {
        let w = self.config.width as i32;
        let h = self.config.height as i32;
        let retain = 1.0 - self.config.evaporation.clamp(0.0, 1.0);
        let diffusion = self.config.diffusion.clamp(0.0, 1.0);
        let keep = 1.0 - diffusion;
        let share = diffusion / 4.0;
        let cap = self.config.pheromone_cap;

        for s in self.scratch.iter_mut() {
            *s = (0.0, 0.0);
        }
        for y in 0..h {
            for x in 0..w {
                let idx = y as usize * w as usize + x as usize;
                let cell = &self.cells[idx];
                if cell.terrain == Terrain::Wall {
                    continue;
                }
                let home = cell.home_pheromone * retain;
                let food = cell.food_pheromone * retain;
                if home <= 0.0 && food <= 0.0 {
                    continue;
                }
                self.scratch[idx].0 += home * keep;
                self.scratch[idx].1 += food * keep;
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
                    self.scratch[nidx].0 += home * share;
                    self.scratch[nidx].1 += food * share;
                }
            }
        }
        for (cell, s) in self.cells.iter_mut().zip(self.scratch.iter()) {
            cell.home_pheromone = if s.0 < 1e-6 { 0.0 } else { s.0.min(cap) };
            cell.food_pheromone = if s.1 < 1e-6 { 0.0 } else { s.1.min(cap) };
        }
    }

    /// Total food remaining on the ground.
    pub fn total_food(&self) -> u64 {
        self.cells.iter().map(|c| c.food as u64).sum()
    }

    /// Total pheromone of both kinds, `(home, food)`.
    pub fn total_pheromone(&self) -> (f64, f64) {
        self.cells.iter().fold((0.0, 0.0), |acc, c| {
            (acc.0 + c.home_pheromone, acc.1 + c.food_pheromone)
        })
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
        // 3x3 cluster at (1,1) → 9 cells × 5 food, but (0,0)…(2,2) are all open.
        assert_eq!(world.total_food(), 45);
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
        let fa: Vec<u32> = a.cells().iter().map(|c| c.food).collect();
        let fb: Vec<u32> = b.cells().iter().map(|c| c.food).collect();
        assert_eq!(fa, fb);
        assert!(a.total_food() > 0);
    }

    #[test]
    fn pheromone_evaporates_and_diffuses() {
        let mut rng = Rng::seed_from_u64(1);
        let mut world = World::new(small_config(), &mut rng);
        let p = Position::new(3, 3);
        world.deposit(p, 10.0, 0.0);
        let before = world.total_pheromone().0;
        world.step_pheromones();
        let after = world.total_pheromone().0;
        assert!(after < before, "evaporation must lower the total");
        assert!(after > before * 0.9, "only a little is lost per tick");
        assert!(world.cell(Position::new(3, 2)).unwrap().home_pheromone > 0.0);
        assert!(world.cell(Position::new(3, 3)).unwrap().home_pheromone > 5.0);
        // Cap is respected.
        world.deposit(p, 1e9, 1e9);
        assert_eq!(
            world.cell(p).unwrap().home_pheromone,
            world.config().pheromone_cap
        );
    }

    #[test]
    fn walls_block_diffusion() {
        let mut rng = Rng::seed_from_u64(1);
        let mut world = World::new(small_config(), &mut rng);
        world.deposit(Position::new(8, 4), 10.0, 10.0);
        world.step_pheromones();
        assert_eq!(world.cell(Position::new(9, 4)).unwrap().home_pheromone, 0.0);
        assert!(world.cell(Position::new(7, 4)).unwrap().home_pheromone > 0.0);
    }
}
