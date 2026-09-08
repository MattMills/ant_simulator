//! Individual ants and what they perceive.
//!
//! An ant is a leaf entity of the hierarchy: it belongs to exactly one leaf
//! category and acts through the effective policy composed along that
//! category's path to the root. Its senses are summarised as a feature vector
//! per candidate direction; the behavioral surface scores those vectors.

use crate::geometry::{Direction, Position};
use crate::world::{Terrain, World};

/// Identifier of an ant within a simulation.
pub type AntId = usize;

/// How many recent positions an ant remembers (to avoid dithering).
pub const MEMORY_LEN: usize = 8;

/// Number of base sensory features per candidate direction.
pub const BASE_FEATURES: usize = 8;

/// Total features per candidate direction: base features, then the same
/// features gated by "carrying food" so behaviour can differ by mode.
pub const FEATURES: usize = 2 * BASE_FEATURES;

/// Human-readable feature names, indexed like a surface's weight vector.
pub const FEATURE_NAMES: [&str; FEATURES] = [
    "food_pheromone",
    "home_pheromone",
    "food_here",
    "nest_here",
    "heading_alignment",
    "nest_alignment",
    "recently_visited",
    "crowding",
    "carrying×food_pheromone",
    "carrying×home_pheromone",
    "carrying×food_here",
    "carrying×nest_here",
    "carrying×heading_alignment",
    "carrying×nest_alignment",
    "carrying×recently_visited",
    "carrying×crowding",
];

/// A single ant.
#[derive(Clone, Debug)]
pub struct Ant {
    /// Identifier.
    pub id: AntId,
    /// Current cell.
    pub position: Position,
    /// Direction of the last move.
    pub heading: Direction,
    /// Whether the ant carries a unit of food.
    pub carrying: bool,
    /// Remaining energy; refilled at the nest.
    pub energy: f64,
    /// Ticks lived.
    pub age: u64,
    /// Dead ants stay in the roster but do nothing.
    pub alive: bool,
    /// Index of the hierarchy leaf (category) the ant belongs to.
    pub leaf: usize,
    /// Ticks since the ant last stood in the nest.
    pub steps_since_nest: u32,
    /// Ticks since the ant last picked up food.
    pub steps_since_food: u32,
    /// Completed food deliveries.
    pub deliveries: u32,
    memory: [Position; MEMORY_LEN],
    memory_cursor: usize,
}

impl Ant {
    /// Create a living ant at `position`.
    pub fn new(
        id: AntId,
        position: Position,
        heading: Direction,
        leaf: usize,
        energy: f64,
    ) -> Self {
        Ant {
            id,
            position,
            heading,
            carrying: false,
            energy,
            age: 0,
            alive: true,
            leaf,
            steps_since_nest: 0,
            steps_since_food: u32::MAX / 2,
            deliveries: 0,
            memory: [position; MEMORY_LEN],
            memory_cursor: 0,
        }
    }

    /// Record a position in the short-term memory ring.
    pub fn remember(&mut self, p: Position) {
        self.memory[self.memory_cursor] = p;
        self.memory_cursor = (self.memory_cursor + 1) % MEMORY_LEN;
    }

    /// Whether `p` is among the recently visited cells.
    pub fn recently_visited(&self, p: Position) -> bool {
        self.memory.contains(&p)
    }

    /// Forget every remembered position (used when the ant is reset).
    pub fn clear_memory(&mut self) {
        self.memory = [self.position; MEMORY_LEN];
    }
}

/// What an ant perceives: one feature vector per candidate direction, and
/// whether that direction is walkable at all.
#[derive(Clone, Debug, Default)]
pub struct Observation {
    /// Feature vectors, indexed by [`Direction::index`].
    pub features: [[f64; FEATURES]; Direction::COUNT],
    /// Whether the cell in that direction can be entered.
    pub valid: [bool; Direction::COUNT],
}

impl Observation {
    /// Number of walkable directions.
    pub fn valid_count(&self) -> usize {
        self.valid.iter().filter(|v| **v).count()
    }
}

/// Sense the world from an ant's point of view.
pub fn observe(ant: &Ant, world: &World) -> Observation {
    let nest = world.nest();
    let to_nest = (
        (nest.x - ant.position.x) as f64,
        (nest.y - ant.position.y) as f64,
    );
    let mut obs = Observation::default();
    for (d, dir) in Direction::ALL.iter().enumerate() {
        let target = ant.position.step(*dir);
        let Some(cell) = world.cell(target) else {
            continue;
        };
        if cell.terrain == Terrain::Wall {
            continue;
        }
        obs.valid[d] = true;
        let f = &mut obs.features[d];
        f[0] = (1.0 + cell.food_pheromone).ln();
        f[1] = (1.0 + cell.home_pheromone).ln();
        f[2] = if cell.food > 0 { 1.0 } else { 0.0 };
        f[3] = if cell.terrain == Terrain::Nest {
            1.0
        } else {
            0.0
        };
        f[4] = ant.heading.cosine(*dir);
        f[5] = dir.cosine_to(to_nest.0, to_nest.1);
        f[6] = if ant.recently_visited(target) {
            1.0
        } else {
            0.0
        };
        f[7] = cell.occupancy.min(4) as f64 / 4.0;
        if ant.carrying {
            let (base, gated) = f.split_at_mut(BASE_FEATURES);
            gated.copy_from_slice(base);
        }
    }
    obs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::world::{FoodSource, WorldConfig};

    fn world() -> World {
        let cfg = WorldConfig {
            width: 10,
            height: 10,
            nest: Position::new(5, 5),
            nest_radius: 0,
            food_sources: vec![FoodSource {
                center: Position::new(2, 2),
                radius: 0,
                amount_per_cell: 3,
            }],
            random_food: None,
            ..WorldConfig::default()
        };
        World::new(cfg, &mut Rng::seed_from_u64(0))
    }

    #[test]
    fn observation_masks_edges_and_gates_on_carrying() {
        let w = world();
        let mut ant = Ant::new(0, Position::new(0, 0), Direction::East, 0, 100.0);
        let obs = observe(&ant, &w);
        assert_eq!(obs.valid_count(), 3);
        assert!(!obs.valid[Direction::North.index()]);
        assert!(obs.valid[Direction::SouthEast.index()]);
        let se = &obs.features[Direction::SouthEast.index()];
        assert_eq!(se[8..], [0.0; BASE_FEATURES]);
        assert!(se[5] > 0.9, "south-east points at the nest");
        ant.carrying = true;
        let obs = observe(&ant, &w);
        let se = &obs.features[Direction::SouthEast.index()];
        assert_eq!(se[..8], se[8..]);
    }

    #[test]
    fn food_and_memory_features() {
        let w = world();
        let mut ant = Ant::new(0, Position::new(2, 3), Direction::North, 0, 100.0);
        ant.remember(Position::new(3, 3));
        let obs = observe(&ant, &w);
        assert_eq!(obs.features[Direction::North.index()][2], 1.0);
        assert_eq!(obs.features[Direction::East.index()][6], 1.0);
        assert_eq!(obs.features[Direction::West.index()][6], 0.0);
        assert!((obs.features[Direction::North.index()][4] - 1.0).abs() < 1e-12);
    }
}
