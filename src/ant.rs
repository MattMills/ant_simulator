//! Individual ants: activity, traits, path integration, route memory, and
//! what they perceive.
//!
//! An ant is a leaf entity of the hierarchy: it belongs to exactly one leaf
//! category and moves through the effective policy composed along that
//! category's path to the root. It has a continuous position and heading.
//! Inside the nest it rests, nurses, or unloads; outside it heads out,
//! feeds, heads home, or searches. It keeps a path-integration home vector
//! with odometric noise (Müller & Wehner 1988), remembers the last rewarding
//! site as a vector from the nest (site fidelity), learns local vectors at
//! familiar places (route memory: Collett & Collett 2002), and carries
//! individual response thresholds (Bonabeau et al. 1996).

use crate::geometry::{Point, Position};
use crate::landscape::{ring_heading, turn_magnitude, RING, RING_STEP};
use crate::memo::{Transit, TransitRecord, MAX_TRANSIT_LEVELS};
use crate::pheromone::{perceived, Pheromone};
use crate::rng::Rng;
use crate::species::Species;
use crate::world::{Nutrient, World};
use std::collections::HashMap;

/// Identifier of an ant within a simulation.
pub type AntId = usize;

/// How many recently visited cells an ant remembers (to avoid dithering).
pub const MEMORY_LEN: usize = 8;

/// What an ant is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Activity {
    /// Inside the nest, not engaged in a task.
    Resting,
    /// Inside the nest, feeding larvae.
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
    /// Inside the nest, walking to a corpse to carry it out.
    Fetching,
    /// Inside the nest, walking to the entrance to go out.
    Leaving,
}

impl Activity {
    /// Every activity, in index order.
    pub const ALL: [Activity; 9] = [
        Activity::Resting,
        Activity::Nursing,
        Activity::Outbound,
        Activity::Feeding,
        Activity::Inbound,
        Activity::Searching,
        Activity::Unloading,
        Activity::Fetching,
        Activity::Leaving,
    ];

    /// Number of activities.
    pub const COUNT: usize = 9;

    /// Index in `0..9`.
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
            Activity::Fetching => "fetching",
            Activity::Leaving => "leaving",
        }
    }

    /// Whether the ant is inside the nest.
    pub fn is_inside(self) -> bool {
        matches!(
            self,
            Activity::Resting
                | Activity::Nursing
                | Activity::Unloading
                | Activity::Fetching
                | Activity::Leaving
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
    /// Body mass relative to the species' typical worker.
    pub size: f64,
    /// Walking speed in cells per tick at the reference temperature.
    pub speed: f64,
    /// Multiplier on the probability of laying trail.
    pub laying: f64,
    /// Multiplier on the pheromone sensitivity constant (`< 1` is keener).
    pub sensitivity: f64,
    /// Individual offset of the spatial fidelity zone inside the nest, as
    /// a fraction of the nest's depth.
    pub zone_offset: f64,
}

impl Traits {
    /// Draw traits for a worker of `species` in a world of the given scale.
    pub fn draw(species: &Species, cell_cm: f64, tick_s: f64, rng: &mut Rng) -> Self {
        let spread = species.threshold_spread;
        let lognormal = |rng: &mut Rng, median: f64| median * (spread * rng.normal()).exp();
        let size_sigma = (1.0 + species.size_cv.max(0.0).powi(2)).ln().sqrt();
        let size = (size_sigma * rng.normal()).exp().clamp(0.4, 2.5);
        Traits {
            foraging_threshold: lognormal(rng, species.threshold_median),
            nursing_threshold: lognormal(rng, species.nursing_threshold_median),
            size,
            speed: species.speed_cm_s * tick_s / cell_cm
                * size.powf(0.3)
                * (1.0 + 0.1 * rng.normal()).clamp(0.6, 1.4),
            laying: (1.0 + 0.2 * rng.normal()).clamp(0.3, 1.7),
            sensitivity: (0.25 * rng.normal()).exp().clamp(0.5, 2.0),
            zone_offset: (0.15 * rng.normal()).clamp(-0.3, 0.3),
        }
    }
}

/// A view of a landmark taken at a place worth returning to: which
/// landmark, and where it stood relative to the place. Seeing the landmark
/// again tells the ant where it is relative to that place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// Index of the landmark in the world's list.
    pub landmark: usize,
    /// Vector from the place to the landmark, in cells.
    pub offset: (f64, f64),
}

/// A remembered food location.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Site {
    /// Vector from the nest to the site, in cells (as path-integrated).
    pub vector: (f64, f64),
    /// Quality of the food found there.
    pub quality: f64,
    /// What was found there.
    pub nutrient: Nutrient,
}

/// What an ant remembers at a familiar place: the local vectors of a
/// route (the direction it walked from here on the way home and on the way
/// out, learned on successful trips, one-way as ant routes are: Collett,
/// Collett, Bisch & Wehner 1998, *Nature* 394:269; Wehner, Boyer, Loertscher,
/// Sommer & Menzi 2006, *Curr. Biol.* 16:75) and the path-integration
/// estimate it had here, against which a later estimate is recalibrated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Route {
    /// Remembered path-integration vector from this place to the nest, in
    /// cells.
    pub home_estimate: (f64, f64),
    /// Unit direction walked from this place on the way home.
    pub home_dir: (f64, f64),
    /// Homeward passes on which this place was learned.
    pub home_strength: f32,
    /// Unit direction walked from this place on the way out to a known
    /// source.
    pub out_dir: (f64, f64),
    /// Outward passes on which this place was learned.
    pub out_strength: f32,
    last_used: u64,
}

/// A single ant.
#[derive(Clone, Debug)]
pub struct Ant {
    /// Identifier.
    pub id: AntId,
    /// Current position in cell units (the nest centre while inside).
    pub position: Point,
    /// Heading in radians (0 = east, clockwise positive on screen).
    pub heading: f64,
    /// What the ant is doing.
    pub activity: Activity,
    /// What a searching ant looks for.
    pub search_target: SearchTarget,
    /// Food carried, microlitres.
    pub crop_ul: f64,
    /// Molarity of the food carried or last collected.
    pub load_molarity: f64,
    /// Quality (0..1) of the food carried or last collected.
    pub load_quality: f64,
    /// How full the crop was on leaving the source (0..1).
    pub load_fill: f64,
    /// What the ant is collecting on this trip.
    pub load_kind: Nutrient,
    /// Prey carried in the mandibles, milligrams.
    pub item_mg: f64,
    /// Whether the ant will take prey this trip (decided on leaving the
    /// nest from the colony's protein demand).
    pub accepts_prey: bool,
    /// Whether the ant is carrying a dead nestmate.
    pub corpse: bool,
    /// A view taken at the nest: the landmark seen from the nest exit and
    /// its position relative to the exit, in cells.
    pub nest_view: Option<View>,
    /// A view taken at the food site: the landmark seen there and its
    /// position relative to the site.
    pub site_view: Option<View>,
    /// Seconds of reserve before starvation once the crop is empty.
    pub energy: f64,
    /// Sugar in the crop, milligrams (the ant's own reserve, and on the
    /// way home the load it will hand over).
    pub sugar_mg: f64,
    /// Sugar the crop can hold, milligrams.
    pub crop_capacity_mg: f64,
    /// Contacts made while unloading the current load.
    pub contacts: u32,
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
    /// Whether path integration has failed and been reset outside the nest.
    pub lost: bool,
    /// Remembered food site.
    pub site: Option<Site>,
    /// Ticks remaining in a timed activity (unloading, nursing).
    pub timer: u32,
    /// Channel currently being laid while walking, if any.
    pub laying: Option<Pheromone>,
    /// Strength multiplier of the deposit while laying.
    pub lay_strength: f64,
    /// Ticks since the ant last stood in the nest.
    pub steps_since_nest: u32,
    /// Ticks since the ant last picked up food.
    pub steps_since_food: u32,
    /// Path length walked since the ant last picked up food, cells.
    pub trip_length: f64,
    /// Ticks spent in the current search.
    pub search_steps: u32,
    /// Ticks spent waiting at a source that offered nothing.
    pub feed_wait: u32,
    /// Completed food deliveries.
    pub deliveries: u32,
    /// Outbound trips that ended without food.
    pub failed_trips: u32,
    /// Where the food currently carried was picked up.
    pub pickup: Option<Point>,
    /// Ticks spent outside the nest.
    pub time_foraging: u64,
    /// Ticks spent nursing.
    pub time_nursing: u64,
    /// Where the ant is going inside the nest (the corpse it fetches, the
    /// exit it leaves by); none while it keeps to its zone.
    pub goal: Option<Position>,
    /// Recruitment excitation from contacts with successful foragers.
    pub excitement: f64,
    /// The transits being recorded through the memo's nodes, one slot
    /// per level memoized, finest first.
    pub records: [Option<TransitRecord>; MAX_TRANSIT_LEVELS],
    /// The transit being replayed, if any: the ant is inside a node and
    /// appears at its exit when the transit ends.
    pub transit: Option<Transit>,
    /// The pipeline's hold on the heading: no decision before this tick.
    pub hold_until: u64,
    /// The pipeline's deadline: a decision by this tick at the latest.
    pub deadline: u64,
    /// Whether the frame's budget has been granted to this ant.
    pub granted: bool,
    /// The activity the hold was set under; a change of leg ends it.
    pub hold_activity: Activity,
    /// Fractional movement credit (unused sub-cell movement).
    pub move_credit: f64,
    /// Ticks left lying stunned after a fall.
    pub stun: u32,
    routes: HashMap<Position, Route>,
    memory: [Position; MEMORY_LEN],
    memory_cursor: usize,
}

impl Ant {
    /// Create a living ant resting inside the nest.
    pub fn new(
        id: AntId,
        nest: Position,
        heading: f64,
        leaf: usize,
        traits: Traits,
        energy: f64,
        crop_capacity_mg: f64,
    ) -> Self {
        Ant {
            id,
            position: Point::center_of(nest),
            heading,
            activity: Activity::Resting,
            search_target: SearchTarget::Nest,
            crop_ul: 0.0,
            load_molarity: 0.0,
            load_quality: 0.0,
            load_fill: 0.0,
            load_kind: Nutrient::Sugar,
            item_mg: 0.0,
            accepts_prey: false,
            corpse: false,
            nest_view: None,
            site_view: None,
            energy,
            sugar_mg: 0.0,
            crop_capacity_mg,
            contacts: 0,
            age: 0,
            alive: true,
            leaf,
            traits,
            home_vector: (0.0, 0.0),
            lost: false,
            site: None,
            timer: 0,
            laying: None,
            lay_strength: 1.0,
            steps_since_nest: 0,
            steps_since_food: u32::MAX / 2,
            trip_length: 0.0,
            search_steps: 0,
            feed_wait: 0,
            deliveries: 0,
            failed_trips: 0,
            pickup: None,
            time_foraging: 0,
            time_nursing: 0,
            goal: None,
            excitement: 0.0,
            records: [None; MAX_TRANSIT_LEVELS],
            hold_until: 0,
            deadline: 0,
            granted: false,
            hold_activity: Activity::Resting,
            transit: None,
            move_credit: 0.0,
            stun: 0,
            routes: HashMap::new(),
            memory: [nest; MEMORY_LEN],
            memory_cursor: 0,
        }
    }

    /// The cell the ant stands in.
    pub fn cell(&self) -> Position {
        self.position.cell()
    }

    /// Whether the ant is inside the nest.
    pub fn is_inside(&self) -> bool {
        self.activity.is_inside()
    }

    /// Whether the ant carries food.
    pub fn carrying(&self) -> bool {
        self.crop_ul > 1e-9 || self.item_mg > 1e-9
    }

    /// Fill of the crop, 0 (empty) to 1 (full).
    pub fn crop_fill(&self) -> f64 {
        if self.crop_capacity_mg <= 0.0 {
            0.0
        } else {
            (self.sugar_mg / self.crop_capacity_mg).clamp(0.0, 1.0)
        }
    }

    /// Empty space in the crop, milligrams of sugar.
    pub fn crop_deficit_mg(&self) -> f64 {
        (self.crop_capacity_mg - self.sugar_mg).max(0.0)
    }

    /// Record a cell in the short-term memory ring.
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
        self.memory = [self.cell(); MEMORY_LEN];
    }

    /// Believed direction to the nest (the negated home vector), `None`
    /// when the ant is at the nest or lost.
    pub fn believed_nest_direction(&self) -> Option<(f64, f64)> {
        if self.lost || self.believed_distance_home() < 1e-9 {
            None
        } else {
            Some((-self.home_vector.0, -self.home_vector.1))
        }
    }

    /// Believed distance to the nest, in cells.
    pub fn believed_distance_home(&self) -> f64 {
        (self.home_vector.0.powi(2) + self.home_vector.1.powi(2)).sqrt()
    }

    /// Believed vector from the current position to the remembered site.
    pub fn site_direction(&self) -> Option<(f64, f64)> {
        if self.lost {
            return None;
        }
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

    /// Update the home vector for a displacement `(dx, dy)` in cells, with
    /// the species' heading and odometric noise (scaled by the distance).
    pub fn integrate(&mut self, dx: f64, dy: f64, species: &Species, rng: &mut Rng) {
        let dist = (dx * dx + dy * dy).sqrt();
        if dist <= 0.0 {
            return;
        }
        let theta = species.pi_heading_noise_deg.to_radians() * dist.sqrt() * rng.normal();
        let scale = 1.0 + species.pi_distance_noise * dist.sqrt() * rng.normal();
        let (s, c) = theta.sin_cos();
        self.home_vector.0 += (c * dx - s * dy) * scale;
        self.home_vector.1 += (s * dx + c * dy) * scale;
    }

    /// Turn the body's frame, as crossing a fold of the ground does (a
    /// floor onto a wall, one face of a box onto the next): the vectors
    /// the ant carries turn with it.
    pub fn rotate_frame(&mut self, turn: f64) {
        if turn == 0.0 {
            return;
        }
        let (s, c) = turn.sin_cos();
        let rotate = |v: (f64, f64)| (c * v.0 - s * v.1, s * v.0 + c * v.1);
        self.home_vector = rotate(self.home_vector);
        if let Some(site) = self.site.as_mut() {
            site.vector = rotate(site.vector);
        }
    }

    /// Reset path integration at the nest.
    pub fn reset_home_vector(&mut self) {
        self.home_vector = (0.0, 0.0);
        self.lost = false;
    }

    /// The route memory at a cell, if any.
    pub fn route(&self, cell: Position) -> Option<&Route> {
        self.routes.get(&cell)
    }

    /// Number of familiar places remembered.
    pub fn familiar_places(&self) -> usize {
        self.routes.len()
    }

    fn route_entry(&mut self, cell: Position, capacity: usize, now: u64) -> &mut Route {
        if !self.routes.contains_key(&cell) && self.routes.len() >= capacity.max(1) {
            if let Some((&oldest, _)) = self.routes.iter().min_by_key(|(_, r)| r.last_used) {
                self.routes.remove(&oldest);
            }
        }
        let entry = self.routes.entry(cell).or_insert(Route {
            home_estimate: (0.0, 0.0),
            home_dir: (0.0, 0.0),
            home_strength: 0.0,
            out_dir: (0.0, 0.0),
            out_strength: 0.0,
            last_used: now,
        });
        entry.last_used = now;
        entry
    }

    /// Remember the path-integration estimate of the nest's position as
    /// seen from a cell (moved towards the current estimate by `rate`).
    pub fn learn_home_estimate(
        &mut self,
        cell: Position,
        estimate: (f64, f64),
        rate: f64,
        capacity: usize,
    ) {
        let now = self.age;
        let r = self.route_entry(cell, capacity, now);
        if r.home_estimate == (0.0, 0.0) {
            r.home_estimate = estimate;
        } else {
            r.home_estimate.0 += rate * (estimate.0 - r.home_estimate.0);
            r.home_estimate.1 += rate * (estimate.1 - r.home_estimate.1);
        }
    }

    /// Learn the homeward local vector at a cell: the direction just walked
    /// from it on the way home.
    pub fn learn_route_home(
        &mut self,
        cell: Position,
        dir: (f64, f64),
        rate: f64,
        capacity: usize,
    ) {
        let now = self.age;
        let r = self.route_entry(cell, capacity, now);
        r.home_dir = blend_direction(
            r.home_dir,
            dir,
            if r.home_strength <= 0.0 { 1.0 } else { rate },
        );
        r.home_strength += 1.0;
    }

    /// Learn the outward local vector at a cell: the direction just walked
    /// from it on the way out to a known source.
    pub fn learn_route_out(&mut self, cell: Position, dir: (f64, f64), rate: f64, capacity: usize) {
        let now = self.age;
        let r = self.route_entry(cell, capacity, now);
        r.out_dir = blend_direction(
            r.out_dir,
            dir,
            if r.out_strength <= 0.0 { 1.0 } else { rate },
        );
        r.out_strength += 1.0;
    }

    /// On recognising a familiar place, pull the path-integration estimate
    /// towards the remembered homeward vector. Returns whether a correction
    /// was applied.
    pub fn recalibrate(&mut self, cell: Position, correction: f64) -> bool {
        let now = self.age;
        let Some(r) = self.routes.get_mut(&cell) else {
            return false;
        };
        if r.home_strength < 2.0 {
            return false;
        }
        r.last_used = now;
        let w =
            correction.clamp(0.0, 1.0) * (r.home_strength as f64 / (r.home_strength as f64 + 2.0));
        let target = (-r.home_estimate.0, -r.home_estimate.1);
        self.home_vector.0 += w * (target.0 - self.home_vector.0);
        self.home_vector.1 += w * (target.1 - self.home_vector.1);
        self.lost = false;
        true
    }
}

/// Move a remembered unit direction towards a new one by `rate` and
/// renormalise (a zero result keeps the new direction).
fn blend_direction(old: (f64, f64), new: (f64, f64), rate: f64) -> (f64, f64) {
    let x = old.0 + rate * (new.0 - old.0);
    let y = old.1 + rate * (new.1 - old.1);
    let len = (x * x + y * y).sqrt();
    if len < 1e-9 {
        new
    } else {
        (x / len, y / len)
    }
}

/// Number of sensory features per candidate heading in one mode.
pub const BASE_FEATURES: usize = 15;

/// Total features per candidate heading: one block for outbound movement
/// and one for inbound movement, so behaviour differs by mode.
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
/// Index of the food-ahead feature.
pub const F_FOOD: usize = 5;
/// Index of the nest-ahead feature.
pub const F_NEST: usize = 6;
/// Index of the heading-persistence feature.
pub const F_HEADING: usize = 7;
/// Index of the path-integration home-vector alignment feature.
pub const F_HOME_VECTOR: usize = 8;
/// Index of the remembered-site alignment feature.
pub const F_SITE: usize = 9;
/// Index of the remembered-route alignment feature.
pub const F_ROUTE: usize = 10;
/// Index of the recently-visited feature.
pub const F_RECENT: usize = 11;
/// Index of the crowding feature.
pub const F_CROWD: usize = 12;
/// Index of the wall-ahead feature.
pub const F_WALL: usize = 13;
/// Index of the food-odour feature.
pub const F_ODOUR: usize = 14;

/// Human-readable feature names, indexed like a surface's weight vector.
pub const FEATURE_NAMES: [&str; FEATURES] = [
    "out:trail",
    "out:home_trail",
    "out:territory",
    "out:no_entry",
    "out:alarm",
    "out:food_ahead",
    "out:nest_ahead",
    "out:heading_persistence",
    "out:home_vector_alignment",
    "out:site_alignment",
    "out:route_alignment",
    "out:recently_visited",
    "out:crowding",
    "out:wall_ahead",
    "out:food_odour",
    "in:trail",
    "in:home_trail",
    "in:territory",
    "in:no_entry",
    "in:alarm",
    "in:food_ahead",
    "in:nest_ahead",
    "in:heading_persistence",
    "in:home_vector_alignment",
    "in:site_alignment",
    "in:route_alignment",
    "in:recently_visited",
    "in:crowding",
    "in:wall_ahead",
    "in:food_odour",
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

/// What an ant perceives: one feature vector per candidate heading, and
/// whether a step in that heading is possible.
#[derive(Clone, Debug)]
pub struct Observation {
    /// Feature vectors, indexed by ring position (0 straight ahead).
    pub features: [[f64; FEATURES]; RING],
    /// Whether a step of the requested length can be taken that way.
    pub valid: [bool; RING],
}

impl Default for Observation {
    fn default() -> Self {
        Observation {
            features: [[0.0; FEATURES]; RING],
            valid: [false; RING],
        }
    }
}

impl Observation {
    /// Number of takeable headings.
    pub fn valid_count(&self) -> usize {
        self.valid.iter().filter(|v| **v).count()
    }
}

/// Heading persistence of a ring position: a quadratic turning cost,
/// `1 - (turn / 90°)²`, which agrees with the cosine for moderate turns
/// (1 straight ahead, 0 at a right angle) but charges a reversal three
/// units rather than one. Ants on a trail keep their direction; U-turns
/// come from losing the trail, not from the trail being stronger behind.
pub fn turn_persistence(ring: usize) -> f64 {
    let turn = turn_magnitude(ring) as f64 * RING_STEP;
    let x = turn / std::f64::consts::FRAC_PI_2;
    1.0 - x * x
}

/// Whether a ring position lies within the antennal sweep (a turn of at
/// most 90°): pheromone ahead is sensed, pheromone behind is not.
pub fn within_antennal_sweep(ring: usize) -> bool {
    turn_magnitude(ring) <= RING / 4
}

/// Sense the world from an ant's point of view in the given mode, for a
/// step of `step` cells.
///
/// Pheromone features integrate an antennal sweep of `species.sense_range`
/// cells ahead (the second cell at half weight), read patch by patch from
/// the substrate along a probe that stops at walls, and pass through the
/// saturating perception `ln(1 + C/k)` with the ant's own sensitivity.
/// The sweep covers the forward half-plane only: headings that turn by
/// more than 90° sense nothing, so a strong trail behind does not pull an
/// ant round and U-turns arise from losing the trail ahead.
pub fn observe(ant: &Ant, world: &World, species: &Species, mode: Mode, step: f64) -> Observation {
    let mut obs = Observation::default();
    let nest_dir = ant.believed_nest_direction();
    let site_dir = ant.site_direction();
    let here = ant.cell();
    let route_dir = ant.route(here).and_then(|r| match mode {
        Mode::Inbound if r.home_strength > 0.0 => {
            Some((r.home_dir, (r.home_strength as f64 / 3.0).min(1.0)))
        }
        Mode::Outbound if r.out_strength > 0.0 && ant.site.is_some() => {
            Some((r.out_dir, (r.out_strength as f64 / 3.0).min(1.0)))
        }
        _ => None,
    });
    let offset = mode.offset();
    let step = step.max(1e-6);
    // The believed directions with their lengths, taken once for the ring.
    let with_len = |v: Option<(f64, f64)>| {
        v.and_then(|(dx, dy)| {
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-12 {
                None
            } else {
                Some((dx, dy, len))
            }
        })
    };
    let nest_vec = with_len(nest_dir);
    let site_vec = with_len(site_dir);
    let route_vec = route_dir.and_then(|(v, w)| with_len(Some(v)).map(|u| (u, w)));
    let channels = [
        (F_TRAIL, Pheromone::Trail),
        (F_HOME, Pheromone::Home),
        (F_TERRITORY, Pheromone::Territory),
        (F_NO_ENTRY, Pheromone::NoEntry),
        (F_ALARM, Pheromone::Alarm),
        (F_ODOUR, Pheromone::Odour),
    ];
    for k in 0..RING {
        let heading = ring_heading(k, ant.heading);
        let (sin, cos) = heading.sin_cos();
        let along = |d: f64| Point::new(ant.position.x + d * cos, ant.position.y + d * sin);
        let target = along(step);
        if !world.segment_passable(ant.position, target) {
            continue;
        }
        let target_cell = target.cell();
        obs.valid[k] = true;
        // The antennal probe: one cell ahead, then a second, stopping at
        // the first wall so nothing is sensed through or around a corner
        // (one sweep of quarter-cell samples serves both).
        let (one, two, reach) = if !world.has_walls() && world.is_passable_point(along(2.0)) {
            // Nothing stops the probe in an open world within the grid.
            (along(1.0), along(2.0), 2.0)
        } else {
            let mut one = ant.position;
            let mut last = (ant.position, 0.0);
            for i in 1..=8 {
                let d = 0.25 * i as f64;
                let p = along(d);
                if !world.is_passable_point(p) {
                    break;
                }
                last = (p, d);
                if i <= 4 {
                    one = p;
                }
            }
            (one, last.0, last.1)
        };
        let second = species.sense_range >= 2 && reach > 1.0 + 1e-9;
        let ahead = within_antennal_sweep(k);
        let f = &mut obs.features[k][offset..offset + BASE_FEATURES];
        for (slot, kind) in channels {
            // Nothing is sensed behind: headings outside the sweep carry
            // no pheromone information, so turning back is governed by the
            // turning cost alone. U-turns then happen where the trail
            // ahead has faded and hardly ever on a strong trail (Beckers,
            // Deneubourg & Goss 1992, *J. Theor. Biol.* 159:397). A
            // channel nothing carries anywhere is not looked at.
            let c = if ahead && world.channel_present(kind) {
                let mut c = world.level_at(one, kind);
                if second {
                    c += 0.5 * world.level_at(two, kind);
                }
                c
            } else {
                0.0
            };
            let k_half = world.channel(kind).k * ant.traits.sensitivity;
            f[slot] = perceived(c, k_half);
        }
        // Food this ant is after: solution always, prey only when the trip
        // is a protein trip.
        let food_at = |p: Point| {
            world
                .cell(p.cell())
                .map(|c| c.has_solution() || (ant.accepts_prey && c.has_prey()))
                .unwrap_or(false)
        };
        f[F_FOOD] = if food_at(target) || food_at(one) {
            1.0
        } else if second && food_at(two) {
            0.5
        } else {
            0.0
        };
        f[F_NEST] = if world.is_nest(target_cell) { 1.0 } else { 0.0 };
        f[F_HEADING] = turn_persistence(k);
        let towards = |v: Option<(f64, f64, f64)>| {
            v.map(|(dx, dy, len)| (cos * dx + sin * dy) / len)
                .unwrap_or(0.0)
        };
        f[F_HOME_VECTOR] = towards(nest_vec);
        f[F_SITE] = towards(site_vec);
        f[F_ROUTE] = route_vec.map(|(u, w)| w * towards(Some(u))).unwrap_or(0.0);
        f[F_RECENT] = if target_cell != here && ant.recently_visited(target_cell) {
            1.0
        } else {
            0.0
        };
        // Crowding ahead: the patch to be entered and the one the probe
        // reaches, whichever is denser (a jam at a bridge entrance pushes
        // arriving ants to the other branch: Dussutour et al. 2004).
        let crowd_at = |p: Position| world.cell(p).map(|c| c.crowding(p == here)).unwrap_or(0.0);
        f[F_CROWD] = crowd_at(target_cell).max(crowd_at(one.cell()));
        f[F_WALL] = if reach < 2.0 - 1e-9 { 1.0 } else { 0.0 };
    }
    obs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::landscape::ring_index_of;
    use crate::world::{FoodSource, WorldConfig};

    fn world() -> World {
        let cfg = WorldConfig {
            width: 10,
            height: 10,
            nest: Position::new(5, 5),
            nest_radius: 0,
            food_sources: vec![FoodSource::pool(Position::new(2, 2), 0, 3.0, 1.0)],
            random_food: None,
            ..WorldConfig::default()
        };
        World::new(cfg, &mut Rng::seed_from_u64(0))
    }

    fn ant_at(cell: Position, heading: f64) -> Ant {
        let species = Species::lasius_niger();
        let traits = Traits::draw(&species, 2.0, 1.0, &mut Rng::seed_from_u64(1));
        let mut a = Ant::new(0, Position::new(5, 5), heading, 0, traits, 100.0, 0.17);
        a.position = Point::center_of(cell);
        a.activity = Activity::Outbound;
        a
    }

    #[test]
    fn observation_masks_edges_and_uses_the_mode_block() {
        let w = world();
        let species = Species::lasius_niger();
        let east = 0.0;
        let mut ant = ant_at(Position::new(0, 0), east);
        ant.home_vector = (-5.0, -5.0);
        let obs = observe(&ant, &w, &species, Mode::Outbound, 0.75);
        // Headings pointing up or left leave the grid.
        assert!(obs.valid[0], "east is open");
        assert!(!obs.valid[RING / 2], "west leaves the grid");
        assert!(!obs.valid[RING * 3 / 4], "north leaves the grid");
        assert!(obs.valid[RING / 4], "south is open");
        let se = ring_index_of(std::f64::consts::FRAC_PI_4, east);
        let f = &obs.features[se];
        assert!(f[BASE_FEATURES..].iter().all(|x| *x == 0.0));
        assert!(f[F_HOME_VECTOR] > 0.99, "south-east points home");
        let obs = observe(&ant, &w, &species, Mode::Inbound, 0.75);
        let f = &obs.features[se];
        assert!(f[..BASE_FEATURES].iter().all(|x| *x == 0.0));
        assert!(f[BASE_FEATURES + F_HOME_VECTOR] > 0.99);
        assert!((obs.features[0][BASE_FEATURES + F_HEADING] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pheromone_features_are_sampled_and_swept() {
        let mut w = world();
        let species = Species::lasius_niger();
        w.deposit(Position::new(6, 5), Pheromone::Trail, 40.0);
        w.deposit(Position::new(7, 5), Pheromone::Trail, 40.0);
        let mut ant = ant_at(Position::new(5, 5), 0.0);
        ant.traits.sensitivity = 1.0;
        let obs = observe(&ant, &w, &species, Mode::Outbound, 0.75);
        let east = obs.features[0][F_TRAIL];
        assert!(
            (east - perceived(60.0, 20.0)).abs() < 1e-9,
            "sweep adds half the second cell"
        );
        assert_eq!(obs.features[RING / 2][F_TRAIL], 0.0);
        let mut near = Species::lasius_niger();
        near.sense_range = 1;
        let obs = observe(&ant, &w, &near, Mode::Outbound, 0.75);
        assert!((obs.features[0][F_TRAIL] - perceived(40.0, 20.0)).abs() < 1e-9);
    }

    #[test]
    fn crowding_is_seen_ahead_relative_to_capacity() {
        let mut w = world();
        let species = Species::lasius_niger();
        let ant = ant_at(Position::new(5, 5), 0.0);
        // Four ants in the cell straight ahead, capacity eight: half crowded.
        w.cell_mut(Position::new(6, 5)).unwrap().occupancy = 4;
        let obs = observe(&ant, &w, &species, Mode::Outbound, 0.75);
        assert!((obs.features[0][F_CROWD] - 0.5).abs() < 1e-12);
        // The probe also looks a cell on: a jam in the next cell but one
        // counts even when the step itself stays in the current cell.
        w.cell_mut(Position::new(6, 5)).unwrap().occupancy = 0;
        let mut far = ant_at(Position::new(6, 5), 0.0);
        far.position = Point::new(6.1, 5.5);
        w.cell_mut(Position::new(7, 5)).unwrap().occupancy = 8;
        let obs = observe(&far, &w, &species, Mode::Outbound, 0.5);
        assert!(
            (obs.features[0][F_CROWD] - 1.0).abs() < 1e-12,
            "{}",
            obs.features[0][F_CROWD]
        );
        // The ant's own presence does not count.
        w.cell_mut(Position::new(5, 5)).unwrap().occupancy = 1;
        let obs = observe(&ant, &w, &species, Mode::Outbound, 0.2);
        assert_eq!(obs.features[0][F_CROWD], 0.0);
    }

    #[test]
    fn food_memory_route_and_wall_features() {
        let w = world();
        let species = Species::lasius_niger();
        let north = -std::f64::consts::FRAC_PI_2;
        let mut ant = ant_at(Position::new(2, 3), north);
        ant.remember(Position::new(3, 3));
        ant.home_vector = (-3.0, -2.0);
        ant.site = Some(Site {
            vector: (-3.0, -12.0),
            quality: 1.0,
            nutrient: Nutrient::Sugar,
        });
        ant.learn_route_out(Position::new(2, 3), (0.0, -1.0), 0.3, 10);
        let obs = observe(&ant, &w, &species, Mode::Outbound, 0.75);
        assert_eq!(obs.features[0][F_FOOD], 1.0, "food straight ahead");
        assert_eq!(
            obs.features[RING / 4][F_RECENT],
            1.0,
            "east was just visited"
        );
        assert_eq!(obs.features[RING / 2][F_RECENT], 0.0);
        assert!((obs.features[0][F_SITE] - 1.0).abs() < 1e-12);
        assert!(
            (obs.features[0][F_ROUTE] - 1.0 / 3.0).abs() < 1e-12,
            "one visit gives a third"
        );
        assert!((ant.believed_distance_to_site().unwrap() - 10.0).abs() < 1e-12);
        // Two cells north of (2,1) is off the grid: wall ahead.
        let mut edge = ant_at(Position::new(2, 1), north);
        edge.site = None;
        let obs = observe(&edge, &w, &species, Mode::Outbound, 0.5);
        assert_eq!(obs.features[0][F_WALL], 1.0);
        assert_eq!(obs.features[RING / 4][F_WALL], 0.0);
    }

    #[test]
    fn path_integration_accumulates_with_noise() {
        let species = Species::lasius_niger();
        let mut rng = Rng::seed_from_u64(7);
        let mut ant = ant_at(Position::new(5, 5), 0.0);
        for _ in 0..100 {
            ant.integrate(1.0, 0.0, &species, &mut rng);
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
        let (nx, _) = ant.believed_nest_direction().unwrap();
        assert!(nx < 0.0);
        ant.lost = true;
        assert!(ant.believed_nest_direction().is_none());
        ant.reset_home_vector();
        assert!(!ant.lost && ant.believed_distance_home() == 0.0);
        let exact = Species {
            pi_heading_noise_deg: 0.0,
            pi_distance_noise: 0.0,
            ..species
        };
        ant.integrate(1.0, -1.0, &exact, &mut rng);
        assert!((ant.home_vector.0 - 1.0).abs() < 1e-12 && (ant.home_vector.1 + 1.0).abs() < 1e-12);
    }

    #[test]
    fn route_memory_learns_and_recalibrates() {
        let mut ant = ant_at(Position::new(4, 4), 0.0);
        let c = Position::new(4, 4);
        ant.learn_route_home(c, (-1.0, 0.0), 0.5, 3);
        ant.learn_home_estimate(c, (-4.0, 0.0), 0.5, 3);
        assert_eq!(ant.route(c).unwrap().home_dir, (-1.0, 0.0));
        assert_eq!(ant.route(c).unwrap().home_estimate, (-4.0, 0.0));
        ant.learn_route_home(c, (0.0, -1.0), 0.5, 3);
        ant.learn_home_estimate(c, (-2.0, 0.0), 0.5, 3);
        let r = ant.route(c).unwrap();
        assert!((r.home_estimate.0 + 3.0).abs() < 1e-12);
        let norm = (r.home_dir.0.powi(2) + r.home_dir.1.powi(2)).sqrt();
        assert!((norm - 1.0).abs() < 1e-12, "directions stay unit length");
        assert!(
            r.home_dir.0 < 0.0 && r.home_dir.1 < 0.0,
            "blended between west and north"
        );
        assert_eq!(r.home_strength, 2.0);
        // Capacity evicts the least recently used place.
        ant.age = 10;
        ant.learn_route_home(Position::new(1, 1), (0.0, 0.0), 0.5, 3);
        ant.age = 20;
        ant.learn_route_home(Position::new(2, 2), (0.0, 0.0), 0.5, 3);
        ant.age = 30;
        ant.learn_route_home(Position::new(3, 3), (0.0, 0.0), 0.5, 3);
        assert_eq!(ant.familiar_places(), 3);
        assert!(ant.route(c).is_none(), "the oldest place was forgotten");
        // Recalibration pulls the estimate towards the remembered vector.
        let d = Position::new(7, 7);
        ant.learn_home_estimate(d, (-7.0, -7.0), 0.5, 10);
        ant.learn_route_home(d, (-0.7, -0.7), 0.5, 10);
        ant.home_vector = (10.0, 10.0);
        assert!(!ant.recalibrate(d, 0.5), "one visit is not enough");
        ant.learn_route_home(d, (-0.7, -0.7), 0.5, 10);
        ant.lost = true;
        assert!(ant.recalibrate(d, 0.5));
        assert!(!ant.lost);
        assert!(ant.home_vector.0 < 10.0 && ant.home_vector.0 > 7.0);
        assert!(!ant.recalibrate(Position::new(9, 9), 0.5));
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
        // Body size: a log-normal spread around one, faster when larger.
        let sizes: Vec<f64> = (0..2000)
            .map(|_| Traits::draw(&species, 2.0, 1.0, &mut rng).size)
            .collect();
        let mean = sizes.iter().sum::<f64>() / sizes.len() as f64;
        let var = sizes.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / sizes.len() as f64;
        assert!((mean - 1.0).abs() < 0.05, "mean size {mean}");
        assert!(
            (var.sqrt() / mean - species.size_cv).abs() < 0.05,
            "cv {}",
            var.sqrt() / mean
        );
        let mut big = a.clone();
        big.size = 2.0;
        assert!(big.size > b.size || big.size > a.size);
        let mut desert = Species::cataglyphis();
        desert.size_cv = 0.0;
        let uniform = Traits::draw(&desert, 2.0, 1.0, &mut rng);
        assert!(
            (uniform.size - 1.0).abs() < 1e-12,
            "no spread, no variation"
        );
        assert_eq!(Activity::Unloading.index(), 6);
        assert!(Activity::Nursing.is_inside());
        assert!(Activity::Searching.is_moving() && Activity::Searching.is_foraging());
        assert_eq!(Activity::Feeding.name(), "feeding");
        let ant = ant_at(Position::new(3, 3), 0.0);
        assert_eq!(ant.cell(), Position::new(3, 3));
        assert!(!ant.carrying());
    }
}
