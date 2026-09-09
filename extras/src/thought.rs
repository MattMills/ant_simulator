//! A thought: one walker of the problem, with the body of an ant.
//!
//! A thought stands in a state at its place on the embedding, faces a
//! heading, and is either resting at the origin, out on a trip, or on
//! its way back. It integrates its path (the home vector), remembers the
//! cells it just crossed, the site of the best solution it found and the
//! route it took there, keeps the transit records the habits need, and
//! carries the horizon and deadline the decision pipeline gives it.

use crate::topos::Letter;
use ant_simulator::geometry::{Point, Position};
use ant_simulator::memo::{Transit, TransitRecord, MAX_TRANSIT_LEVELS};

/// Cells a thought remembers having just crossed.
pub const RECENT: usize = 8;

/// What a thought is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activity {
    /// At the origin between trips.
    Resting,
    /// Out towards a remembered site, along what it knows.
    Outbound,
    /// Out with nothing to go on: a scout.
    Searching,
    /// Back to the origin, laden with a solution or not.
    Inbound,
}

impl Activity {
    /// Whether the thought is away from the origin.
    pub fn is_moving(self) -> bool {
        !matches!(self, Activity::Resting)
    }

    /// Whether the thought is on its way out.
    pub fn is_out(self) -> bool {
        matches!(self, Activity::Outbound | Activity::Searching)
    }

    /// Name for reports.
    pub fn name(self) -> &'static str {
        match self {
            Activity::Resting => "resting",
            Activity::Outbound => "outbound",
            Activity::Searching => "searching",
            Activity::Inbound => "inbound",
        }
    }
}

/// A remembered site: where a solution was found, how good it was, and
/// how often going back there came to nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Site {
    /// The place of the finding.
    pub place: Point,
    /// Its quality.
    pub quality: f64,
    /// Trips to it since that came home empty-handed.
    pub failures: u32,
}

/// The move in progress: the state walked to and its place.
#[derive(Clone, Debug, PartialEq)]
pub struct Target<S> {
    /// The state at the end of the move.
    pub state: S,
    /// Its place.
    pub point: Point,
}

/// One walker of the problem.
#[derive(Clone, Debug)]
pub struct Thought<S> {
    /// Index among the mind's thoughts.
    pub id: usize,
    /// Index of its leaf in the hierarchy's leaves.
    pub leaf: usize,
    /// Index of its caste among the castes.
    pub caste: usize,
    /// The state it stands in.
    pub state: S,
    /// Where it stands on the embedding.
    pub place: Point,
    /// Heading in radians (0 east, clockwise positive on screen).
    pub heading: f64,
    /// What it is doing.
    pub activity: Activity,
    /// The move in progress, if any.
    pub target: Option<Target<S>>,
    /// The quality of the solution carried home, if any.
    pub load: Option<f64>,
    /// The states of the trip out, the origin first.
    pub route: Vec<S>,
    /// Their places.
    pub places: Vec<Point>,
    /// Where along the route a retrace stands.
    pub back: usize,
    /// The places of the last successful trip out: the route memory.
    pub remembered: Vec<Point>,
    /// The best site found.
    pub site: Option<Site>,
    /// The path integrated since the origin, in cells.
    pub home_vector: (f64, f64),
    recent: [Position; RECENT],
    recent_len: usize,
    recent_next: usize,
    /// Moves made this trip.
    pub steps: usize,
    /// Cells walked this trip.
    pub trip_length: f64,
    /// Cells of no-entry marking left to lay on the way back.
    pub no_entry_left: f64,
    /// How strongly the solution carried recruits: the trail laid per
    /// cell is the configured deposit times the quality times this.
    pub recruit: f64,
    /// Retreats from dead ends this trip.
    pub retreats: usize,
    /// Whether the move in progress is a retreat from a dead end.
    pub retreat: bool,
    /// The state just retreated from, not to be tried again at once.
    pub avoid: Option<S>,
    /// The class of the trip's route, once a solution was found (see
    /// [`crate::topos`]).
    pub symbol: Option<usize>,
    /// The sign the thought set out with in mind (see
    /// [`crate::lexicon`]).
    pub intent: Option<usize>,
    /// Whether the thought is an echo (see [`crate::echo`]).
    pub echo: bool,
    /// The sign an echo holds across its trips.
    pub held: Option<usize>,
    /// Tick the held sign was taken up.
    pub held_since: u64,
    /// Walks of the held sign that found nothing at the glyph's end.
    pub empty_walks: u32,
    /// Whether the intent was taken from an echo met on the way.
    pub heard: bool,
    /// Where the trip's solution was found.
    pub finding: Option<Position>,
    /// The letters of the moves of the trip out, one per move after
    /// the origin (0 where a move is no letter).
    pub letters: Vec<Letter>,
    /// The letters the trip's word begins with.
    pub prefix: Vec<Letter>,
    /// The letters the trip's word ends with, once a solution is found.
    pub suffix: Vec<Letter>,
    /// The integrated state: what the trip has transported, or the
    /// developed displacement of a free walk.
    pub x: Vec<f64>,
    /// The turn accumulated through folds, so that a free walk develops
    /// in the frame it set out in.
    pub phi: f64,
    /// The turn of the last decision, for the movement history.
    pub last_turn: f64,
    /// Tick at which a rest ends.
    pub rest_until: u64,
    /// Tick until which the heading is held without a decision.
    pub hold_until: u64,
    /// Tick by which the next decision is due.
    pub deadline: u64,
    /// The activity the horizon was set in.
    pub hold_activity: Activity,
    /// Whether the frame's budget serves this thought's decision.
    pub granted: bool,
    /// Transit records being kept, one per memoized level.
    pub records: [Option<TransitRecord>; MAX_TRANSIT_LEVELS],
    /// A transit being replayed.
    pub transit: Option<Transit>,
    /// The foreseen route (a geodesic), followed point by point.
    pub plan: Vec<Point>,
    /// The next point of the plan.
    pub plan_index: usize,
    /// Tick the plan was made.
    pub planned_at: u64,
    /// Trips completed.
    pub trips: u64,
    /// Solutions brought home.
    pub findings: u64,
}

impl<S: Clone> Thought<S> {
    /// A thought resting at the origin.
    pub fn new(id: usize, leaf: usize, caste: usize, state: S, place: Point) -> Thought<S> {
        Thought {
            id,
            leaf,
            caste,
            state,
            place,
            heading: 0.0,
            activity: Activity::Resting,
            target: None,
            load: None,
            route: Vec::new(),
            places: Vec::new(),
            back: 0,
            remembered: Vec::new(),
            site: None,
            home_vector: (0.0, 0.0),
            recent: [Position::new(i32::MIN, i32::MIN); RECENT],
            recent_len: 0,
            recent_next: 0,
            steps: 0,
            trip_length: 0.0,
            no_entry_left: 0.0,
            recruit: 1.0,
            retreats: 0,
            retreat: false,
            avoid: None,
            symbol: None,
            intent: None,
            echo: false,
            held: None,
            held_since: 0,
            empty_walks: 0,
            heard: false,
            finding: None,
            letters: Vec::new(),
            prefix: Vec::new(),
            suffix: Vec::new(),
            x: Vec::new(),
            phi: 0.0,
            last_turn: 0.0,
            rest_until: 0,
            hold_until: 0,
            deadline: 0,
            hold_activity: Activity::Resting,
            granted: false,
            records: [None; MAX_TRANSIT_LEVELS],
            transit: None,
            plan: Vec::new(),
            plan_index: 0,
            planned_at: 0,
            trips: 0,
            findings: 0,
        }
    }

    /// The cell stood in.
    pub fn cell(&self) -> Position {
        self.place.cell()
    }

    /// Whether a solution is carried.
    pub fn laden(&self) -> bool {
        self.load.is_some()
    }

    /// Note a cell crossed.
    pub fn remember(&mut self, p: Position) {
        if self.recently_visited(p) {
            return;
        }
        self.recent[self.recent_next] = p;
        self.recent_next = (self.recent_next + 1) % RECENT;
        self.recent_len = (self.recent_len + 1).min(RECENT);
    }

    /// Whether a cell was crossed lately.
    pub fn recently_visited(&self, p: Position) -> bool {
        self.recent[..self.recent_len].contains(&p)
    }

    /// The cells crossed lately.
    pub fn recent(&self) -> &[Position] {
        &self.recent[..self.recent_len]
    }

    /// Forget the cells crossed.
    pub fn clear_recent(&mut self) {
        self.recent_len = 0;
        self.recent_next = 0;
    }

    /// The body's frame turns with a fold of the surface: the vector
    /// integrated turns with it.
    pub fn rotate_frame(&mut self, turn: f64) {
        if turn == 0.0 {
            return;
        }
        let (s, c) = turn.sin_cos();
        let v = self.home_vector;
        self.home_vector = (c * v.0 - s * v.1, s * v.0 + c * v.1);
        self.phi += turn;
    }

    /// Develop a step of a free walk into the frame the trip set out
    /// in: the step turned back by the turns accumulated, added to the
    /// first two components of the integrated state.
    pub fn develop(&mut self, dx: f64, dy: f64) {
        if self.x.len() < 2 {
            return;
        }
        let (s, c) = (-self.phi).sin_cos();
        self.x[0] += c * dx - s * dy;
        self.x[1] += s * dx + c * dy;
    }

    /// The direction home by path integration, if any way has been
    /// walked.
    pub fn home_direction(&self) -> Option<(f64, f64)> {
        unit((-self.home_vector.0, -self.home_vector.1))
    }

    /// The direction to the remembered site, if any.
    pub fn site_direction(&self) -> Option<(f64, f64)> {
        let site = self.site?;
        unit(self.place.to(site.place))
    }

    /// The direction along the plan: to the point after the nearest
    /// point within reach among the next few from where the plan
    /// stands, which then stands there; or, off the plan, back to the
    /// point it stands at.
    pub fn plan_direction(&mut self) -> Option<(f64, f64)> {
        if self.plan.is_empty() {
            return None;
        }
        let end = (self.plan_index + 16).min(self.plan.len());
        let mut best: Option<(usize, f64)> = None;
        for j in self.plan_index..end {
            let d = self.place.distance(self.plan[j]);
            if d <= 1.5 && best.map(|(_, bd)| d < bd).unwrap_or(true) {
                best = Some((j, d));
            }
        }
        if let Some((j, _)) = best {
            self.plan_index = j + 1;
        }
        let next = *self.plan.get(self.plan_index)?;
        unit(self.place.to(next))
    }

    /// Whether the plan has been walked to its end.
    pub fn plan_done(&self) -> bool {
        !self.plan.is_empty() && self.plan_index >= self.plan.len()
    }

    /// How far the thought stands from the plan's next point.
    pub fn plan_distance(&self) -> Option<f64> {
        let next = *self.plan.get(self.plan_index)?;
        Some(self.place.distance(next))
    }

    /// The direction along the remembered route: to the point after the
    /// nearest remembered point within reach, if any.
    pub fn route_direction(&self) -> Option<(f64, f64)> {
        if self.remembered.len() < 2 {
            return None;
        }
        let mut best: Option<(usize, f64)> = None;
        for (j, p) in self.remembered.iter().enumerate() {
            let d = self.place.distance(*p);
            if d <= 1.5 && best.map(|(_, bd)| d < bd).unwrap_or(true) {
                best = Some((j, d));
            }
        }
        let (j, _) = best?;
        let next = self.remembered.get(j + 1).copied()?;
        unit(self.place.to(next))
    }

    /// Reset everything of a trip at the origin.
    pub fn reset_trip(&mut self, origin: S, place: Point) {
        self.state = origin.clone();
        self.place = place;
        self.target = None;
        self.load = None;
        self.route = vec![origin];
        self.places = vec![place];
        self.back = 0;
        self.home_vector = (0.0, 0.0);
        self.clear_recent();
        self.steps = 0;
        self.trip_length = 0.0;
        self.no_entry_left = 0.0;
        self.recruit = 1.0;
        self.retreats = 0;
        self.retreat = false;
        self.avoid = None;
        self.symbol = None;
        self.intent = None;
        self.heard = false;
        self.finding = None;
        self.letters.clear();
        self.prefix.clear();
        self.suffix.clear();
        self.x.clear();
        self.phi = 0.0;
        self.last_turn = 0.0;
        self.hold_until = 0;
        self.deadline = 0;
        self.granted = false;
        self.records = [None; MAX_TRANSIT_LEVELS];
        self.transit = None;
        self.plan.clear();
        self.plan_index = 0;
    }
}

/// A vector as a unit direction, unless it is (nearly) zero.
pub fn unit(v: (f64, f64)) -> Option<(f64, f64)> {
    let len = (v.0 * v.0 + v.1 * v.1).sqrt();
    if len < 1e-9 {
        None
    } else {
        Some((v.0 / len, v.1 / len))
    }
}
