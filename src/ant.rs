//! Individual ants: activity, traits, path integration, and what they
//! perceive.
//!
//! An ant is a leaf entity of the hierarchy: it belongs to exactly one leaf
//! category and moves through the effective policy composed along that
//! category's path to the root. Inside the nest it rests, nurses, or
//! unloads; outside it heads out, feeds, heads home, or searches. It keeps
//! a path-integration home vector with odometric noise (Müller & Wehner
//! 1988), remembers the last rewarding site as a vector from the nest (site
//! fidelity), and carries individual response thresholds (Bonabeau et al.
//! 1996).

use crate::geometry::{Direction, Position};
use crate::pheromone::{perceived, Pheromone};
use crate::rng::Rng;
use crate::species::Species;
use crate::world::{Terrain, World};

/// Identifier of an ant within a simulation.
pub type AntId = usize;

/// How many recent positions an ant remembers (to avoid dithering).
pub const MEMORY_LEN: usize = 8;

/// What an ant is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Activity {
    /// Inside the nest, not engaged in a task.
    Resting,
    /// Inside the nest, feeding brood.
    Nursing,
    /// Outside, heading away from the nest in search of food.
    Outbound,
    /// At a food source, filling the crop.
    Feeding,
    /// Outside, heading home (loaded or not).
    Inbound,
    /// Outside, searching systematically around a fictive location.
    Searching,
    /// Inside the nest, handing over the load by trophallaxis.
    Unloading,
}

impl Activity {
    /// Every activity, in index order.
    pub const ALL: [Activity; 7] = [
        Activity::Resting,
        Activity::Nursing,
        Activity::Outbound,
        Activity::Feeding,
        Activity::Inbound,
        Activity::Searching,
        Activity::Unloading,
    ];

    /// Number of activities.
    pub const COUNT: usize = 7;

    /// Index in `0..7`.
    pub fn index(self) -> usize {
        Activity::ALL
            .iter()
            .position(|a| *a == self)
            .expect("listed")
    }

    /// Short name.
    pub fn name(self) -> &'static str {
        match self {
            Activity::Resting => "resting",
            Activity::Nursing => "nursing",
            Activity::Outbound => "outbound",
            Activity::Feeding => "feeding",
            Activity::Inbound => "inbound",
            Activity::Searching => "searching",
            Activity::Unloading => "unloading",
        }
    }

    /// Whether the ant is inside the nest (off the grid).
    pub fn is_inside(self) -> bool {
        matches!(
            self,
            Activity::Resting | Activity::Nursing | Activity::Unloading
        )
    }

    /// Whether the ant walks during this activity.
    pub fn is_moving(self) -> bool {
        matches!(
            self,
            Activity::Outbound | Activity::Inbound | Activity::Searching
        )
    }

    /// Whether the ant counts as foraging (outside on colony business).
    pub fn is_foraging(self) -> bool {
        !self.is_inside()
    }
}

/// What a searching ant is trying to find.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchTarget {
    /// The nest entrance, after path integration ran out.
    Nest,
    /// A remembered food site that was not where expected.
    Food,
}

/// Individual differences, drawn from the species' distributions.
#[derive(Clone, Debug, PartialEq)]
pub struct Traits {
    /// Adult foraging response threshold.
    pub foraging_threshold: f64,
    /// Nursing response threshold.
    pub nursing_threshold: f64,
    /// Walking speed in cells per tick.
    pub speed: f64,
    /// Multiplier on the probability of laying trail.
    pub laying: f64,
    /// Multiplier on the pheromone sensitivity constant (`< 1` is keener).
    pub sensitivity: f64,
}

impl Traits {
    /// Draw traits for a worker of `species` in a world of the given scale.
    pub fn draw(species: &Species, cell_cm: f64, tick_s: f64, rng: &mut Rng) -> Self {
        let spread = species.threshold_spread;
        let lognormal = |rng: &mut Rng, median: f64| median * (spread * rng.normal()).exp();
        Traits {
            foraging_threshold: lognormal(rng, species.threshold_median),
            nursing_threshold: lognormal(rng, species.nursing_threshold_median),
            speed: species.speed_cm_s * tick_s / cell_cm
                * (1.0 + 0.15 * rng.normal()).clamp(0.5, 1.5),
            laying: (1.0 + 0.2 * rng.normal()).clamp(0.3, 1.7),
            sensitivity: (0.25 * rng.normal()).exp().clamp(0.5, 2.0),
        }
    }
}

/// A remembered food location.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Site {
    /// Vector from the nest to the site, in cells (as path-integrated).
    pub vector: (f64, f64),
    /// Quality of the food found there.
    pub quality: f64,
}

/// A single ant.
#[derive(Clone, Debug)]
pub struct Ant {
    /// Identifier.
    pub id: AntId,
    /// Current cell (the nest centre while inside).
    pub position: Position,
    /// Direction of the last move.
    pub heading: Direction,
    /// What the ant is doing.
    pub activity: Activity,
    /// What a searching ant looks for.
    pub search_target: SearchTarget,
    /// Food carried, in crop loads (0 or up to the species' capacity).
    pub crop: f64,
    /// Quality of the food carried or last collected.
    pub load_quality: f64,
    /// Seconds of reserve before starvation.
    pub energy: f64,
    /// Ticks lived.
    pub age: u64,
    /// Dead ants stay in the roster but do nothing.
    pub alive: bool,
    /// Index of the hierarchy leaf (category) the ant belongs to.
    pub leaf: usize,
    /// Individual traits.
    pub traits: Traits,
    /// Path-integrated displacement from the nest, in cells.
    pub home_vector: (f64, f64),
    /// Remembered food site.
    pub site: Option<Site>,
    /// Ticks remaining in a timed activity (feeding, unloading, nursing).
    pub timer: u32,
    /// Channel currently being laid while walking, if any.
    pub laying: Option<Pheromone>,
    /// Strength multiplier of the deposit while laying.
    pub lay_strength: f64,
    /// Ticks since the ant last stood in the nest.
    pub steps_since_nest: u32,
    /// Ticks since the ant last picked up food.
    pub steps_since_food: u32,
    /// Moves made since the ant last picked up food.
    pub trip_moves: u32,
    /// Ticks spent in the current search.
    pub search_steps: u32,
    /// Completed food deliveries.
    pub deliveries: u32,
    /// Outbound trips that ended without food.
    pub failed_trips: u32,
    /// Where the food currently carried was picked up.
    pub pickup: Option<Position>,
    /// Ticks spent outside the nest.
    pub time_foraging: u64,
    /// Ticks spent nursing.
    pub time_nursing: u64,
    /// Fractional movement credit (speed below one cell per tick).
    pub move_credit: f64,
    memory: [Position; MEMORY_LEN],
    memory_cursor: usize,
}

impl Ant {
    /// Create a living ant resting inside the nest.
    pub fn new(
        id: AntId,
        nest: Position,
        heading: Direction,
        leaf: usize,
        traits: Traits,
        energy: f64,
    ) -> Self {
        Ant {
            id,
            position: nest,
            heading,
            activity: Activity::Resting,
            search_target: SearchTarget::Nest,
            crop: 0.0,
            load_quality: 0.0,
            energy,
            age: 0,
            alive: true,
            leaf,
            traits,
            home_vector: (0.0, 0.0),
            site: None,
            timer: 0,
            laying: None,
            lay_strength: 1.0,
            steps_since_nest: 0,
            steps_since_food: u32::MAX / 2,
            trip_moves: 0,
            search_steps: 0,
            deliveries: 0,
            failed_trips: 0,
            pickup: None,
            time_foraging: 0,
            time_nursing: 0,
            move_credit: 0.0,
            memory: [nest; MEMORY_LEN],
            memory_cursor: 0,
        }
    }

    /// Whether the ant is inside the nest.
    pub fn is_inside(&self) -> bool {
        self.activity.is_inside()
    }

    /// Whether the ant carries food.
    pub fn carrying(&self) -> bool {
        self.crop > 0.0
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

    /// Forget every remembered position.
    pub fn clear_memory(&mut self) {
        self.memory = [self.position; MEMORY_LEN];
    }

    /// Believed direction to the nest (the negated home vector).
    pub fn believed_nest_direction(&self) -> (f64, f64) {
        (-self.home_vector.0, -self.home_vector.1)
    }

    /// Believed distance to the nest, in cells.
    pub fn believed_distance_home(&self) -> f64 {
        (self.home_vector.0.powi(2) + self.home_vector.1.powi(2)).sqrt()
    }

    /// Believed vector from the current position to the remembered site.
    pub fn site_direction(&self) -> Option<(f64, f64)> {
        self.site.map(|s| {
            (
                s.vector.0 - self.home_vector.0,
                s.vector.1 - self.home_vector.1,
            )
        })
    }

    /// Believed distance to the remembered site, in cells.
    pub fn believed_distance_to_site(&self) -> Option<f64> {
        self.site_direction().map(|(x, y)| (x * x + y * y).sqrt())
    }

    /// Update the home vector for a move in `dir`, with the species'
    /// heading and odometric noise.
    pub fn integrate(&mut self, dir: Direction, species: &Species, rng: &mut Rng) {
        let (dx, dy) = dir.delta();
        let (dx, dy) = (dx as f64, dy as f64);
        let theta = species.pi_heading_noise_deg.to_radians() * rng.normal();
        let scale = 1.0 + species.pi_distance_noise * rng.normal();
        let (s, c) = theta.sin_cos();
        let bx = (c * dx - s * dy) * scale;
        let by = (s * dx + c * dy) * scale;
        self.home_vector.0 += bx;
        self.home_vector.1 += by;
    }

    /// Reset path integration at the nest.
    pub fn reset_home_vector(&mut self) {
        self.home_vector = (0.0, 0.0);
    }
}

/// Number of sensory features per candidate direction in one mode.
pub const BASE_FEATURES: usize = 12;

/// Total features per candidate direction: one block for outbound
/// movement and one for inbound movement, so behaviour differs by mode.
pub const FEATURES: usize = 2 * BASE_FEATURES;

/// Index of the recruitment-trail feature within a block.
pub const F_TRAIL: usize = 0;
/// Index of the outbound "home" trail feature.
pub const F_HOME: usize = 1;
/// Index of the home-range marking feature.
pub const F_TERRITORY: usize = 2;
/// Index of the "no entry" marking feature.
pub const F_NO_ENTRY: usize = 3;
/// Index of the alarm feature.
pub const F_ALARM: usize = 4;
/// Index of the food-present feature.
pub const F_FOOD: usize = 5;
/// Index of the nest-cell feature.
pub const F_NEST: usize = 6;
/// Index of the heading-alignment feature.
pub const F_HEADING: usize = 7;
/// Index of the path-integration home-vector alignment feature.
pub const F_HOME_VECTOR: usize = 8;
/// Index of the remembered-site alignment feature.
pub const F_SITE: usize = 9;
/// Index of the recently-visited feature.
pub const F_RECENT: usize = 10;
/// Index of the crowding feature.
pub const F_CROWD: usize = 11;

/// Human-readable feature names, indexed like a surface's weight vector.
pub const FEATURE_NAMES: [&str; FEATURES] = [
    "out:trail",
    "out:home_trail",
    "out:territory",
    "out:no_entry",
    "out:alarm",
    "out:food_here",
    "out:nest_here",
    "out:heading_alignment",
    "out:home_vector_alignment",
    "out:site_alignment",
    "out:recently_visited",
    "out:crowding",
    "in:trail",
    "in:home_trail",
    "in:territory",
    "in:no_entry",
    "in:alarm",
    "in:food_here",
    "in:nest_here",
    "in:heading_alignment",
    "in:home_vector_alignment",
    "in:site_alignment",
    "in:recently_visited",
    "in:crowding",
];

/// Which block of the surface a movement decision uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Away from the nest.
    Outbound,
    /// Towards the nest.
    Inbound,
}

impl Mode {
    /// Offset of the block within the feature vector.
    pub fn offset(self) -> usize {
        match self {
            Mode::Outbound => 0,
            Mode::Inbound => BASE_FEATURES,
        }
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

fn cosine(dir: Direction, v: Option<(f64, f64)>) -> f64 {
    match v {
        Some((x, y)) => dir.cosine_to(x, y),
        None => 0.0,
    }
}

/// Sense the world from an ant's point of view in the given mode.
///
/// Pheromone features integrate an antennal sweep of `species.sense_range`
/// cells ahead (the second cell at half weight) and pass through the
/// saturating perception `ln(1 + C/k)` with the ant's own sensitivity.
pub fn observe(ant: &Ant, world: &World, species: &Species, mode: Mode) -> Observation {
    let mut obs = Observation::default();
    let nest_dir = ant.believed_nest_direction();
    let nest_dir = if ant.believed_distance_home() < 1e-9 {
        None
    } else {
        Some(nest_dir)
    };
    let site_dir = ant.site_direction();
    let offset = mode.offset();
    for (d, dir) in Direction::ALL.iter().enumerate() {
        let target = ant.position.step(*dir);
        let Some(cell) = world.cell(target) else {
            continue;
        };
        if cell.terrain == Terrain::Wall {
            continue;
        }
        obs.valid[d] = true;
        let beyond = if species.sense_range >= 2 {
            world
                .cell(target.step(*dir))
                .filter(|c| c.terrain != Terrain::Wall)
        } else {
            None
        };
        let f = &mut obs.features[d][offset..offset + BASE_FEATURES];
        for (slot, kind) in [
            (F_TRAIL, Pheromone::Trail),
            (F_HOME, Pheromone::Home),
            (F_TERRITORY, Pheromone::Territory),
            (F_NO_ENTRY, Pheromone::NoEntry),
            (F_ALARM, Pheromone::Alarm),
        ] {
            let mut c = cell.level(kind);
            if let Some(b) = beyond {
                c += 0.5 * b.level(kind);
            }
            let k = world.channel(kind).k * ant.traits.sensitivity;
            f[slot] = perceived(c, k);
        }
        f[F_FOOD] = if cell.food > 0 { 1.0 } else { 0.0 };
        f[F_NEST] = if cell.terrain == Terrain::Nest {
            1.0
        } else {
            0.0
        };
        f[F_HEADING] = ant.heading.cosine(*dir);
        f[F_HOME_VECTOR] = cosine(*dir, nest_dir);
        f[F_SITE] = cosine(*dir, site_dir);
        f[F_RECENT] = if ant.recently_visited(target) {
            1.0
        } else {
            0.0
        };
        f[F_CROWD] = cell.occupancy.min(4) as f64 / 4.0;
    }
    obs
}

#[cfg(test)]
mod tests {
    use super::*;
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
                quality: 1.0,
            }],
            random_food: None,
            ..WorldConfig::default()
        };
        World::new(cfg, &mut Rng::seed_from_u64(0))
    }

    fn ant_at(p: Position, heading: Direction) -> Ant {
        let species = Species::lasius_niger();
        let traits = Traits::draw(&species, 2.0, 1.0, &mut Rng::seed_from_u64(1));
        let mut a = Ant::new(0, Position::new(5, 5), heading, 0, traits, 100.0);
        a.position = p;
        a.activity = Activity::Outbound;
        a
    }

    #[test]
    fn observation_masks_edges_and_uses_the_mode_block() {
        let w = world();
        let species = Species::lasius_niger();
        let mut ant = ant_at(Position::new(0, 0), Direction::East);
        ant.home_vector = (-5.0, -5.0);
        let obs = observe(&ant, &w, &species, Mode::Outbound);
        assert_eq!(obs.valid_count(), 3);
        assert!(!obs.valid[Direction::North.index()]);
        let se = &obs.features[Direction::SouthEast.index()];
        assert!(se[BASE_FEATURES..].iter().all(|x| *x == 0.0));
        assert!(se[F_HOME_VECTOR] > 0.9, "south-east points home");
        let obs = observe(&ant, &w, &species, Mode::Inbound);
        let se = &obs.features[Direction::SouthEast.index()];
        assert!(se[..BASE_FEATURES].iter().all(|x| *x == 0.0));
        assert!(se[BASE_FEATURES + F_HOME_VECTOR] > 0.9);
    }

    #[test]
    fn pheromone_features_are_perceived_and_swept() {
        let mut w = world();
        let species = Species::lasius_niger();
        w.deposit(Position::new(6, 5), Pheromone::Trail, 40.0);
        w.deposit(Position::new(7, 5), Pheromone::Trail, 40.0);
        let mut ant = ant_at(Position::new(5, 5), Direction::East);
        ant.traits.sensitivity = 1.0;
        let obs = observe(&ant, &w, &species, Mode::Outbound);
        let east = obs.features[Direction::East.index()][F_TRAIL];
        assert!(
            (east - perceived(60.0, 20.0)).abs() < 1e-12,
            "sweep adds half the second cell"
        );
        assert_eq!(obs.features[Direction::West.index()][F_TRAIL], 0.0);
        let mut near = Species::lasius_niger();
        near.sense_range = 1;
        let obs = observe(&ant, &w, &near, Mode::Outbound);
        assert!(
            (obs.features[Direction::East.index()][F_TRAIL] - perceived(40.0, 20.0)).abs() < 1e-12
        );
    }

    #[test]
    fn food_memory_and_site_features() {
        let w = world();
        let species = Species::lasius_niger();
        let mut ant = ant_at(Position::new(2, 3), Direction::North);
        ant.remember(Position::new(3, 3));
        ant.home_vector = (-3.0, -2.0);
        ant.site = Some(Site {
            vector: (-3.0, -12.0),
            quality: 1.0,
        });
        let obs = observe(&ant, &w, &species, Mode::Outbound);
        assert_eq!(obs.features[Direction::North.index()][F_FOOD], 1.0);
        assert_eq!(obs.features[Direction::East.index()][F_RECENT], 1.0);
        assert_eq!(obs.features[Direction::West.index()][F_RECENT], 0.0);
        assert!((obs.features[Direction::North.index()][F_HEADING] - 1.0).abs() < 1e-12);
        assert!((obs.features[Direction::North.index()][F_SITE] - 1.0).abs() < 1e-12);
        assert!((ant.believed_distance_to_site().unwrap() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn path_integration_accumulates_with_noise() {
        let species = Species::lasius_niger();
        let mut rng = Rng::seed_from_u64(7);
        let mut ant = ant_at(Position::new(5, 5), Direction::East);
        for _ in 0..100 {
            ant.integrate(Direction::East, &species, &mut rng);
        }
        let d = ant.believed_distance_home();
        assert!(
            (d - 100.0).abs() < 5.0,
            "odometry within a few percent: {d}"
        );
        assert!(
            ant.home_vector.1.abs() < 12.0,
            "heading noise stays modest: {:?}",
            ant.home_vector
        );
        let (nx, _) = ant.believed_nest_direction();
        assert!(nx < 0.0);
        ant.reset_home_vector();
        assert_eq!(ant.believed_distance_home(), 0.0);
        let exact = Species {
            pi_heading_noise_deg: 0.0,
            pi_distance_noise: 0.0,
            ..species
        };
        ant.integrate(Direction::NorthEast, &exact, &mut rng);
        assert!((ant.home_vector.0 - 1.0).abs() < 1e-12 && (ant.home_vector.1 + 1.0).abs() < 1e-12);
    }

    #[test]
    fn traits_are_positive_and_varied() {
        let species = Species::lasius_niger();
        let mut rng = Rng::seed_from_u64(3);
        let a = Traits::draw(&species, 2.0, 1.0, &mut rng);
        let b = Traits::draw(&species, 2.0, 1.0, &mut rng);
        assert!(a.foraging_threshold > 0.0 && a.speed > 0.0);
        assert_ne!(a.foraging_threshold, b.foraging_threshold);
        assert!(
            (a.speed - 0.75).abs() < 0.4,
            "1.5 cm/s over 2 cm cells ≈ 0.75 cells/tick"
        );
        assert_eq!(Activity::Unloading.index(), 6);
        assert!(Activity::Nursing.is_inside());
        assert!(Activity::Searching.is_moving() && Activity::Searching.is_foraging());
        assert_eq!(Activity::Feeding.name(), "feeding");
    }
}
