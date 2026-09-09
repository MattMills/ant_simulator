//! # ant_extras
//!
//! A general-purpose architecture that **thinks in the style of ants**,
//! built by re-leveraging every component of [`ant_simulator`]: a
//! problem is embedded on a surface and walked by a colony of thoughts,
//! and what the thoughts lay down on their way home is the solution.
//!
//! The ants' way of solving their problem (where the food is, and how
//! to get it home) has a fixed shape: an individual senses a little of
//! its surroundings, scores a ring of moves with a surface of
//! preferences, draws from that ring with a set amount of disorder,
//! walks, and on the way back marks the ground in proportion to what it
//! found; the marks evaporate, the colony's history collapses into an
//! invariant skeleton, and a queen on a slow clock turns the colony's
//! disorder up or down. Nothing in that shape is about food. This crate
//! keeps the shape and changes the problem.
//!
//! ## The pieces, and what they were
//!
//! | Here | In the ants | What it does for thinking |
//! |---|---|---|
//! | [`problem::Problem`] | the world's geometry, nest and food | states with places on a grid, an origin, moves (free steps or successor states), solutions with a quality, a scent |
//! | the medium ([`ant_simulator::world::World`]) | the chemical field on two structures | trail laid home in proportion to quality, no-entry at dead ends, the smell of solutions on the grain, crowding |
//! | [`sense`] | the antennal sweep and the thirty sensory features | the same thirty slots, read along a move, so the ant's surfaces apply unchanged |
//! | [`thought::Thought`] | the ant's body | state, place, heading, path integration, site fidelity, route memory, transit records, a horizon |
//! | surface and hierarchy ([`ant_simulator::hierarchy`]) | instinct, castes, ants | the ant's instinct at the root, scouts and followers with their dials, the queen's moods, a leaf per thought |
//! | path surfaces ([`ant_simulator::landscape`]) | the ring of headings deformed and selected | the same, over a ring the moves are placed on by their direction in the embedding |
//! | history ([`ant_simulator::hive::MovementHistory`]) | the hive's cognitive geometry | invariant skeleton and residual of the thoughts' walks: the pipeline's ground, the queen's eyes |
//! | memo and transits ([`ant_simulator::memo`]) | memoized transits | habits: the walk through plain, invariant ground stood in for by a kernel |
//! | the lens ([`ant_simulator::lens`]) | the two-position geodesic | foresight: a plan followed point by point |
//! | pipeline ([`ant_simulator::pipeline`]) | horizons and the frame budget | a heading held where the decision would have been the same |
//! | the queen ([`ant_simulator::hive::Queen`]) | the queen's mind | stagnation heats the scouts, organisation cools the followers |
//! | [`practice`] | learners, levers, the arena | learners tune the surfaces through a hidden, rotating connection, rewarded by what comes home |
//! | [`topos`] | the six pheromone channels | a variable surface: a channel per learned symbol, each a class of routes (its word round the punctures), registered, weighed, labelled and generated, with a transport as its vector |
//! | [`problems::Relations`] | path integration | the holonomy embedding: thoughts integrate a D-dimensional state through the relations they follow, the words of their trips are rules, and the geometry learns to close along them |
//! | [`bridge`] | the outbound walk and the trail laid home | the forward and backward filters solved explicitly on the grain: the predicted corridor, the surprise of the flows against it, the drift of the conditioned walk |
//! | [`lexicon`] | recruitment at the nest, the honeybee's dance floor | signs: the classes danced by the thoughts that come home, drawn by those setting out, meaning where they lead, understood as far as the listeners get there |
//! | [`echo`] | tandem running | signs persisted in walk form: followers that hold a sign for an epoch and walk its glyph again and again, for the others to hear on the way, the route to enter the invariant skeleton, and the glyph to be refined |
//! | the frame ([`ant_simulator::frame`]) | surfaces joined along edges | a maze over the folded-out surfaces of a box |
//!
//! ## Three problems
//!
//! * [`problems::Maze`]: free movement over a grid with walls, or over
//!   a frame's box. This is the ants' own problem stated generally, and
//!   the trail found round the wall is the path.
//! * [`problems::Tour`]: a tour of cities in the plane. A state is the
//!   city stood in and the cities seen; the moves are the cities not yet
//!   seen; the trail laid home runs along the tour's own edges through
//!   the medium.
//! * [`problems::Colouring`]: a graph colouring laid out layer by layer,
//!   nodes along the width and colours along the height, with dead ends
//!   where no colour is left, marked no-entry and retreated from.
//!
//! ## Symbols: the surface that grows
//!
//! The ants' surface has six channels with fixed meanings. The
//! path-topological network ([`topos`]) lets it grow: a route's
//! **word** is the sequence of rays it crosses, one ray running up from
//! each **puncture** of the embedding (a wall, a city, or a hole the
//! walks turn out to enclose), freely reduced; two routes have the same
//! word exactly when one can be deformed into the other without
//! crossing a puncture. A word walked home often enough is registered
//! as a **symbol** with a channel of its own, a glyph, and a weight
//! learned from how its class yields, which the thoughts sense as they
//! sense a pheromone. The label is blind: nobody gave it. A name can be
//! attached after the fact, a new route is labelled by its word
//! (inference), and a route of a class is produced again, by a search
//! over cells and words or by expressing the symbol into its channel
//! for the colony to walk (generation). Punctures themselves are
//! learned where the walks go round something both ways.
//!
//! ## The holonomy embedding
//!
//! A thought's home vector is the integral of its displacement; the
//! holonomy embedding generalises that integral to D dimensions and
//! makes it the representation (`extras/docs/holonomy.md`). A problem
//! may give its thoughts an integrative capacity: on a free walk the
//! state develops the steps through the turns of the folds crossed,
//! on a walk of states each move adds its transport. A route's word is
//! its class, and on a flat connection what a route integrates depends
//! only on its word, so a symbol's transport, the mean of what its
//! routes integrate, is the class's vector and the spread the
//! curvature. Portals and moves are decidable gates, admitting a
//! thought by its state and its word, so a walk is a typed computation
//! and generation over places, words and gates terminates. Learning is
//! local and gradient-free: closure credit on the way home, arrival
//! near a target vector as food. Round a cylinder the classes are the
//! windings and their transports circumferences; on a knowledge graph
//! of families the words of the trips are rules, the relations'
//! vectors learn to close along them, and held-out facts are predicted
//! by the geometry, by the rules and by both.
//!
//! ## The bridge
//!
//! The walk conditioned on where its trips end is a Schrödinger
//! bridge: forward filter times backward filter, with the log-gradient
//! of the backward one as drift. The colony computes the pair
//! implicitly, the outbound walks sampling the forward filter and the
//! trail laid home accumulating the backward one; [`bridge`] computes
//! both explicitly on a coarse graph of the medium at a temperature
//! that is the forcing pressure, measures the surprise of the flows
//! against their product, lets the queen read it and the thoughts
//! follow the drift, weighs the symbols by their sector partition
//! functions, and conditions a practice on survival
//! (`extras/docs/bridge.md`).
//!
//! ## The lexicon
//!
//! A symbol becomes a sign when it is used to tell another thought
//! something. [`lexicon`] borrows the use from the honeybee: a thought
//! home with a solution dances the sign of its route's class in
//! proportion to what it found, a follower setting out may draw a sign
//! in proportion to its dance and set out with it in mind (the way its
//! glyph goes, the glyph as its plan, the class's channel as its
//! scent, its own memory set aside), a sign means where the trips of
//! its class ended, synonyms are signs that lead to the same place,
//! and a sign is understood as far as the thoughts that heard it end
//! where it means. The lexicon measures itself: the mutual information
//! between sign and outcome, the entropy of the floor, each sign's
//! agreement, and what talking is worth (`extras/docs/lexicon.md`).
//!
//! ## The echo
//!
//! The floor speaks fast and at the nest; an [`echo`] speaks the same
//! language at a slower epoch and along the route. It holds a sign,
//! walks its glyph out and back again and again, lays the sign's
//! channel both ways, and is heard by the thoughts it meets on the way,
//! who take the sign from where they met it. The walks put the route
//! into the movement history's invariant skeleton an epoch before the
//! foragers do, refine the glyph, and re-seed the floor whenever the
//! source is still there; they do not lengthen the memory, which lives
//! in the ground already (`extras/docs/echo.md`).
//!
//! ## Where the ants' way stops
//!
//! Habits and foresight belong to the walk between decisions, so they
//! serve the problems walked freely; where every step is a choice the
//! solution is made of, no ground is plain and nothing is foreseen but
//! the next state. The embedding decides what the medium can carry: two
//! moves that share cells share marks, which is generalisation when
//! like states lie close and noise when they do not. And the ant's
//! instinct is a good start, not a fit: `cargo run --release -p
//! ant_extras --example think` runs the three problems, and the
//! practice module is how a surface is tuned to one.

pub mod bridge;
pub mod echo;
pub mod lexicon;
pub mod mind;
pub mod practice;
pub mod problem;
pub mod problems;
pub mod sense;
pub mod thought;
pub mod topos;

/// Everything commonly needed.
pub mod prelude {
    pub use crate::bridge::{Bridge, BridgeConfig};
    pub use crate::echo::{Echo, EchoConfig};
    pub use crate::lexicon::{Lexicon, LexiconConfig, Sign};
    pub use crate::mind::{Choice, Finding, Geometry, Mind, MindConfig, MindStats, QueenPolicy};
    pub use crate::practice::{Practice, PracticeConfig, PracticeReport, Targets, Turn};
    pub use crate::problem::{embedding, Layered, Moves, Problem};
    pub use crate::problems::{
        ColourState, Colouring, Evaluation, Maze, Node, Relations, Tour, TourState, Triple,
    };
    pub use crate::sense::{Body, Candidate, Senses, Sight};
    pub use crate::thought::{Activity, Site, Target, Thought};
    pub use crate::topos::{
        link_letter, link_of, move_letter, prefix_letter, Label, Letter, PathNet, Rays, Route,
        Symbol, SymbolConfig, SymbolField, Word,
    };
}
