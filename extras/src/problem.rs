//! What the mind can think about: a problem is a set of states placed on a
//! surface, with moves between them.
//!
//! The ants' world is a grid with a nest, food somewhere on it, and walls;
//! an ant's moves are steps in sixteen headings. A problem for the mind
//! is the same thing seen generally: its states have *places* on a grid
//! (the embedding), its origin is the nest, its solutions are the food,
//! and its moves are either free steps over the grid (a spatial problem)
//! or a set of successor states, each somewhere on the grid (a
//! combinatorial one). The embedding is where knowledge of the problem
//! goes in: states that are alike should lie close, so that what the
//! thoughts lay down about one is read about the others.

use crate::topos::Letter;
use ant_simulator::geometry::{Point, Position};
use ant_simulator::world::WorldConfig;

/// The moves open from a state.
#[derive(Clone, Debug, PartialEq)]
pub enum Moves<S> {
    /// Discrete successors: the walk is a sequence of states, and the
    /// solution is the sequence.
    States(Vec<S>),
    /// Free movement over the embedding in steps of `step` cells, in any
    /// of the sixteen headings: the walk is a path over the surface.
    Free {
        /// Step length in cells.
        step: f64,
    },
}

/// A problem the mind can walk.
pub trait Problem {
    /// A state of the problem.
    type State: Clone + PartialEq;

    /// The embedding: the grid the thoughts walk, with whatever walls,
    /// portals and slopes shape it. Its nest should be the origin's cell.
    fn embedding(&self) -> WorldConfig;

    /// Where every walk begins.
    fn origin(&self) -> Self::State;

    /// Where a state lies on the embedding.
    fn place(&self, state: &Self::State) -> Point;

    /// The moves open from a state. An empty list of states is a dead end.
    fn moves(&self, state: &Self::State) -> Moves<Self::State>;

    /// The quality of a state as a solution, if it is one, on a scale
    /// where 1 is a good solution: the food a thought carries home.
    fn quality(&self, state: &Self::State) -> Option<f64>;

    /// How promising a move smells before it is taken: a heuristic in
    /// about `0..=1`, sensed as the smell of food is.
    fn scent(&self, _from: &Self::State, _to: &Self::State) -> f64 {
        0.0
    }

    /// For free movement: the state at a place, if a thought can stand
    /// there. Problems with discrete moves need not answer.
    fn locate(&self, _place: Point) -> Option<Self::State> {
        None
    }

    /// The punctures of the embedding a path-topological network starts
    /// with (see [`crate::topos`]): points routes wind around, such as
    /// the walls of a maze or the cities of a tour. More are learned
    /// from the holes the thoughts' walks enclose.
    fn punctures(&self) -> Vec<Point> {
        Vec::new()
    }

    // -----------------------------------------------------------------
    // The holonomy embedding: letters, integration, targets, learning
    // (see `extras/docs/holonomy.md`). Everything here has a default
    // that leaves a problem as it was.

    /// Where the trip numbered `trip` begins: the origin by default, or
    /// a different state per trip (a query of a relational problem).
    fn depart(&self, _trip: u64) -> Self::State {
        self.origin()
    }

    /// The letters a trip from this state begins its word with (a
    /// query relation, say), so that its class is a class of that
    /// query's routes. Use [`crate::topos::prefix_letter`].
    fn prefix(&self, _origin: &Self::State) -> Vec<Letter> {
        Vec::new()
    }

    /// The letter of a move between states, if the move is a letter of
    /// the route's word (a relation of a graph, signed by direction).
    /// Use [`crate::topos::move_letter`].
    fn letter(&self, _from: &Self::State, _to: &Self::State) -> Option<Letter> {
        None
    }

    /// Names for the letters of moves and prefixes, for showing words.
    fn letter_names(&self) -> Vec<(Letter, String)> {
        Vec::new()
    }

    /// The dimension of the integrated state a thought carries on a
    /// trip, or 0 for none. On a walk of states it is integrated move
    /// by move; on a free walk it is the developed displacement.
    fn capacity(&self) -> usize {
        0
    }

    /// The integrated state a trip from this state starts with.
    fn departure(&self, _origin: &Self::State) -> Vec<f64> {
        vec![0.0; self.capacity()]
    }

    /// Integrate a move into the state: the move's transport.
    fn integrate(&self, _from: &Self::State, _to: &Self::State, _x: &mut [f64]) {}

    /// The vector a trip from this state should land on, if there is
    /// one.
    fn target(&self, _origin: &Self::State) -> Option<Vec<f64>> {
        None
    }

    /// The quality of a state reached with an integrated state: by
    /// default the state's own quality.
    fn quality_at(&self, state: &Self::State, _x: &[f64]) -> Option<f64> {
        self.quality(state)
    }

    /// The scent of a move given the integrated state: by default the
    /// move's scent.
    fn scent_at(&self, from: &Self::State, to: &Self::State, _x: &[f64]) -> f64 {
        self.scent(from, to)
    }

    /// Learn from a trip: its origin, its route out (the origin first),
    /// the integrated state at its end, and the quality it brought
    /// home (0 for a trip that came home with nothing).
    fn learn(&mut self, _origin: &Self::State, _route: &[Self::State], _x: &[f64], _quality: f64) {}

    /// A short description of a state.
    fn describe(&self, state: &Self::State) -> String;

    /// The problem's name.
    fn name(&self) -> String;
}

/// A grid embedding of a layered problem: variables along the width,
/// their values along the height, the origin at the left edge.
#[derive(Clone, Debug, PartialEq)]
pub struct Layered {
    /// Grid width in cells.
    pub width: usize,
    /// Grid height in cells.
    pub height: usize,
    /// Number of layers (variables) after the origin.
    pub layers: usize,
    /// Number of values per layer.
    pub values: usize,
    /// Cells kept clear round the edge.
    pub margin: usize,
}

impl Layered {
    /// An embedding of `layers` variables with `values` values each on a
    /// grid of the given size.
    pub fn new(width: usize, height: usize, layers: usize, values: usize) -> Layered {
        Layered {
            width,
            height,
            layers: layers.max(1),
            values: values.max(1),
            margin: 2,
        }
    }

    /// The place of the origin: the left margin, halfway up.
    pub fn origin(&self) -> Point {
        Point::new(self.margin as f64 + 0.5, self.height as f64 / 2.0)
    }

    /// The place of a value of a layer (both from zero).
    pub fn place(&self, layer: usize, value: usize) -> Point {
        let inner_w = (self.width - 2 * self.margin).max(1) as f64;
        let inner_h = (self.height - 2 * self.margin).max(1) as f64;
        let x = self.margin as f64 + (layer as f64 + 1.0) * inner_w / (self.layers as f64 + 1.0);
        let y = self.margin as f64 + (value as f64 + 0.5) * inner_h / self.values as f64;
        Point::new(x.floor() + 0.5, y.floor() + 0.5)
    }

    /// A world for the embedding: open ground, the nest at the origin.
    pub fn config(&self) -> WorldConfig {
        embedding(self.width, self.height, self.origin().cell(), 0)
    }
}

/// A plain world for an embedding: open ground of the given size with the
/// nest at a cell, no food of its own, and room for traffic.
pub fn embedding(width: usize, height: usize, nest: Position, nest_radius: i32) -> WorldConfig {
    WorldConfig {
        width,
        height,
        nest,
        nest_radius,
        random_food: None,
        food_sources: Vec::new(),
        cell_capacity: 16,
        ..WorldConfig::default()
    }
}
