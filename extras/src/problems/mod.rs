//! Three problems embedded for the mind: a maze walked freely (on a plain
//! grid or over the surfaces of a box), a tour of cities in the plane, and
//! a graph colouring laid out layer by layer.

pub mod colouring;
pub mod maze;
pub mod tour;

pub use colouring::{ColourState, Colouring};
pub use maze::Maze;
pub use tour::{Tour, TourState};
