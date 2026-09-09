//! The sensorium: what a thought perceives of a move, in the ant's own
//! sensory slots.
//!
//! An ant's surface weighs fifteen features of each candidate heading,
//! in an outbound and an inbound block (see
//! [`ant_simulator::ant::FEATURE_NAMES`]). A thought perceives a move
//! into the same slots, so that the ant's instinct, its hierarchy of
//! surfaces and its queen apply to thinking unchanged: the trail, home,
//! territory, no-entry and alarm marks read along the move as an antenna
//! reads them; a solution one move ahead as food ahead; the origin ahead
//! as the nest; the turn's persistence; the alignments with the home
//! vector, the remembered site and the plan or route; whether the cell
//! was just crossed; the crowding there; a wall ahead; and the smell of
//! the goal, which is the odour in the medium and the problem's own
//! scent of the move.

use ant_simulator::ant::{turn_persistence, within_antennal_sweep};
pub use ant_simulator::ant::{
    BASE_FEATURES, FEATURES, FEATURE_NAMES, F_ALARM, F_CROWD, F_FOOD, F_HEADING, F_HOME,
    F_HOME_VECTOR, F_NEST, F_NO_ENTRY, F_ODOUR, F_RECENT, F_ROUTE, F_SITE, F_TERRITORY, F_TRAIL,
    F_WALL,
};
use ant_simulator::geometry::{Point, Position};
use ant_simulator::landscape::RING;
use ant_simulator::pheromone::{perceived, Pheromone};
use ant_simulator::world::World;

/// A move a thought could make.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate<S> {
    /// The state at the end of the move.
    pub state: S,
    /// Its place, in the thought's own chart (before any fold).
    pub point: Point,
    /// The turn of the body's frame if the move crosses a fold.
    pub turn: f64,
    /// Length of the move, in cells.
    pub len: f64,
    /// The ring position of the move's direction.
    pub ring: usize,
    /// Whether the move can be made.
    pub valid: bool,
}

/// What a thought perceives, one feature vector per ring position, with
/// the candidate each position stands for.
#[derive(Clone, Debug)]
pub struct Senses {
    /// Feature vectors indexed by ring position (0 straight ahead).
    pub features: [[f64; FEATURES]; RING],
    /// Whether a move lies that way.
    pub valid: [bool; RING],
    /// The candidate at each ring position (meaningless where invalid).
    pub slot: [usize; RING],
    /// What the learned symbols add to the score at each ring position
    /// (see [`crate::topos`]).
    pub extra: [f64; RING],
}

impl Default for Senses {
    fn default() -> Self {
        Senses {
            features: [[0.0; FEATURES]; RING],
            valid: [false; RING],
            slot: [usize::MAX; RING],
            extra: [0.0; RING],
        }
    }
}

/// The thought's own state of body, as far as sensing needs it.
#[derive(Clone, Debug)]
pub struct Body<'a> {
    /// Where it stands.
    pub place: Point,
    /// Direction home by path integration.
    pub home_dir: Option<(f64, f64)>,
    /// Direction to the remembered site.
    pub site_dir: Option<(f64, f64)>,
    /// Direction along the plan or the remembered route.
    pub route_dir: Option<(f64, f64)>,
    /// Cells just crossed.
    pub recent: &'a [Position],
    /// Sensitivity to the marks (1 is the species' own).
    pub sensitivity: f64,
}

/// What is known of a candidate beyond the medium.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sight {
    /// A solution lies at the end of the move.
    pub food: bool,
    /// The origin lies at the end of the move.
    pub nest: bool,
    /// The problem's own scent of the move.
    pub scent: f64,
    /// How far ahead the ground is clear, in cells, up to 2.
    pub clear: f64,
}

/// Read a channel along a move. A step (up to two cells) is read as an
/// antenna reads it: one cell ahead at full weight, the second at half.
/// A longer move, which the thought commits to as a whole, is read the
/// whole way: the mean of a sample per cell along it (up to 24), so
/// that what was laid along that very way is what is sensed, not what
/// crosses its first cells.
pub fn read_along(world: &World, from: Point, unit: (f64, f64), len: f64, kind: Pheromone) -> f64 {
    let at = |d: f64| Point::new(from.x + d * unit.0, from.y + d * unit.1);
    if len <= 2.0 + 1e-9 {
        let mut c = world.level_at(at(len.min(1.0)), kind);
        if len > 1.0 {
            c += 0.5 * world.level_at(at(len.min(2.0)), kind);
        }
        return c;
    }
    let n = (len.ceil() as usize).clamp(2, 24);
    let mut total = 0.0;
    for i in 1..=n {
        total += world.level_at(at(len * i as f64 / n as f64), kind);
    }
    1.5 * total / n as f64
}

/// The features of one candidate move into one block of the vector.
#[allow(clippy::too_many_arguments)]
pub fn features(
    world: &World,
    body: &Body<'_>,
    unit: (f64, f64),
    len: f64,
    ring: usize,
    sight: Sight,
    offset: usize,
    out: &mut [f64; FEATURES],
) {
    let f = &mut out[offset..offset + BASE_FEATURES];
    let ahead = within_antennal_sweep(ring);
    let channels = [
        (F_TRAIL, Pheromone::Trail),
        (F_HOME, Pheromone::Home),
        (F_TERRITORY, Pheromone::Territory),
        (F_NO_ENTRY, Pheromone::NoEntry),
        (F_ALARM, Pheromone::Alarm),
        (F_ODOUR, Pheromone::Odour),
    ];
    for (slot, kind) in channels {
        let c = if ahead && world.channel_present(kind) {
            read_along(world, body.place, unit, len, kind)
        } else {
            0.0
        };
        let k = world.channel(kind).k * body.sensitivity;
        f[slot] = perceived(c, k);
    }
    f[F_ODOUR] += sight.scent;
    f[F_FOOD] = if sight.food { 1.0 } else { 0.0 };
    f[F_NEST] = if sight.nest { 1.0 } else { 0.0 };
    f[F_HEADING] = turn_persistence(ring);
    let cos = |d: Option<(f64, f64)>| d.map(|(x, y)| x * unit.0 + y * unit.1).unwrap_or(0.0);
    f[F_HOME_VECTOR] = cos(body.home_dir);
    f[F_SITE] = cos(body.site_dir);
    f[F_ROUTE] = cos(body.route_dir);
    let end = Point::new(
        body.place.x + unit.0 * len.max(1.0),
        body.place.y + unit.1 * len.max(1.0),
    );
    let end_cell = end.cell();
    f[F_RECENT] = if body.recent.contains(&end_cell) {
        1.0
    } else {
        0.0
    };
    f[F_CROWD] = world
        .cell(end_cell)
        .map(|c| c.crowding(false))
        .unwrap_or(0.0);
    f[F_WALL] = (1.0 - sight.clear / 2.0).clamp(0.0, 1.0);
}

/// How far the ground ahead is clear along a direction, in quarter-cell
/// samples up to two cells, stopping at the first wall.
pub fn clear_ahead(world: &World, from: Point, unit: (f64, f64)) -> f64 {
    let mut clear = 0.0;
    for i in 1..=8 {
        let d = 0.25 * i as f64;
        let p = Point::new(from.x + d * unit.0, from.y + d * unit.1);
        if !world.is_passable_point(p) {
            break;
        }
        clear = d;
    }
    clear
}
