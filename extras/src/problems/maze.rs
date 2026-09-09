//! A maze: free movement over a grid with walls, or over the folded-out
//! surfaces of a box, from a start to a goal. The thoughts walk it as
//! ants walk their world, and the trail they lay between the two is the
//! path found: with time it straightens into the geodesic round the
//! walls, as the ants' trails do (Goss et al. 1989).

use crate::problem::{embedding, Moves, Problem};
use crate::topos::{destination_letter, Letter};
use ant_simulator::frame::Frame;
use ant_simulator::geometry::{Point, Position};
use ant_simulator::world::{Rect, WorldConfig};

/// A maze between a start and a goal.
#[derive(Clone, Debug)]
pub struct Maze {
    config: WorldConfig,
    start: Position,
    goals: Vec<(Position, f64)>,
    goal_radius: i32,
    step: f64,
}

impl Maze {
    /// An open grid of the given size.
    pub fn new(width: usize, height: usize, start: Position, goal: Position) -> Maze {
        let mut config = embedding(width, height, start, 1);
        config.cell_capacity = 8;
        Maze {
            config,
            start,
            goals: vec![(goal, 1.0)],
            goal_radius: 1,
            step: 1.0,
        }
    }

    /// A maze over a frame's surfaces (see [`ant_simulator::frame`]):
    /// its regions, portals and slopes shape the walk.
    pub fn on_frame(frame: &Frame, start: Position, goal: Position) -> Maze {
        let mut config = frame.config();
        config.nest = start;
        config.nest_radius = 1;
        config.random_food = None;
        config.cell_capacity = 8;
        Maze {
            config,
            start,
            goals: vec![(goal, 1.0)],
            goal_radius: 1,
            step: 1.0,
        }
    }

    /// Change the goals (the world outside the thoughts changes).
    pub fn set_goals(&mut self, goals: Vec<(Position, f64)>) {
        if !goals.is_empty() {
            self.goals = goals;
        }
    }

    /// Several goals, each with its quality (the first is the goal
    /// [`goal`](Self::goal) names).
    pub fn with_goals(mut self, goals: Vec<(Position, f64)>) -> Maze {
        if !goals.is_empty() {
            self.goals = goals;
        }
        self
    }

    /// The goals and their qualities.
    pub fn goals(&self) -> &[(Position, f64)] {
        &self.goals
    }

    /// Two ways round a wall, one shorter than the other: the double
    /// bridge. The start and the goal sit in the upper third of the
    /// grid on either side of a wall that leaves a gap at the top and
    /// another at the bottom.
    pub fn around_a_wall(width: usize, height: usize) -> Maze {
        let (w, h) = (width as i32, height as i32);
        let start = Position::new(4, h / 3);
        let goal = Position::new(w - 5, h / 3);
        let wall = Rect::new(Position::new(w / 2 - 1, 3), Position::new(w / 2 + 1, h - 4));
        Maze::new(width, height, start, goal).with_walls(vec![wall])
    }

    /// Walls as rectangles of cells.
    pub fn with_walls(mut self, walls: Vec<Rect>) -> Maze {
        self.config.walls = walls;
        self
    }

    /// How close to the goal cell counts as reaching it (Chebyshev).
    pub fn with_goal_radius(mut self, radius: i32) -> Maze {
        self.goal_radius = radius.max(0);
        self
    }

    /// The start cell.
    pub fn start(&self) -> Position {
        self.start
    }

    /// The goal cell (the first goal).
    pub fn goal(&self) -> Position {
        self.goals[0].0
    }

    /// The world the maze is drawn on.
    pub fn config(&self) -> &WorldConfig {
        &self.config
    }

    /// Whether a cell is ground a thought can stand on.
    pub fn passable(&self, p: Position) -> bool {
        let c = &self.config;
        if p.x < 0 || p.y < 0 || p.x as usize >= c.width || p.y as usize >= c.height {
            return false;
        }
        if c.walls.iter().any(|r| r.contains(p)) {
            return false;
        }
        c.open.is_empty() || c.open.iter().any(|r| r.contains(p))
    }

    /// The straight-line distance from start to goal, in cells.
    pub fn crow_flight(&self) -> f64 {
        self.start.euclid(self.goals[0].0)
    }
}

impl Problem for Maze {
    type State = Position;

    fn embedding(&self) -> WorldConfig {
        self.config.clone()
    }

    fn origin(&self) -> Position {
        self.start
    }

    fn place(&self, state: &Position) -> Point {
        Point::center_of(*state)
    }

    fn moves(&self, _state: &Position) -> Moves<Position> {
        Moves::Free { step: self.step }
    }

    fn quality(&self, state: &Position) -> Option<f64> {
        self.goals
            .iter()
            .filter(|(g, _)| state.chebyshev(*g) <= self.goal_radius)
            .map(|(_, q)| *q)
            .fold(None, |best: Option<f64>, q| {
                Some(best.map_or(q, |b| b.max(q)))
            })
    }

    fn locate(&self, place: Point) -> Option<Position> {
        let cell = place.cell();
        self.passable(cell).then_some(cell)
    }

    /// Which goal a solution is: the nearest goal within reach.
    fn suffix(&self, state: &Position) -> Vec<Letter> {
        self.goals
            .iter()
            .enumerate()
            .filter(|(_, (g, _))| state.chebyshev(*g) <= self.goal_radius)
            .min_by_key(|(_, (g, _))| state.chebyshev(*g))
            .map(|(i, _)| vec![destination_letter(i)])
            .unwrap_or_default()
    }

    fn letter_names(&self) -> Vec<(Letter, String)> {
        self.goals
            .iter()
            .enumerate()
            .map(|(i, (g, q))| {
                (
                    destination_letter(i),
                    format!("@({},{}) {:.1}", g.x, g.y, q),
                )
            })
            .collect()
    }

    /// The centre of each wall.
    fn punctures(&self) -> Vec<Point> {
        self.config
            .walls
            .iter()
            .map(|r| {
                Point::new(
                    (r.min.x + r.max.x + 1) as f64 / 2.0,
                    (r.min.y + r.max.y + 1) as f64 / 2.0,
                )
            })
            .collect()
    }

    fn describe(&self, state: &Position) -> String {
        format!("({}, {})", state.x, state.y)
    }

    fn name(&self) -> String {
        "maze".to_string()
    }
}
