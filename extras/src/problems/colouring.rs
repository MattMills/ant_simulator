//! A graph colouring, laid out layer by layer: the nodes of the graph
//! along the width of the grid, the colours along its height. A thought
//! colours the nodes in order, one move per node, choosing a colour no
//! coloured neighbour has; where none is left it stands at a dead end and
//! marks it, as Pharaoh's ants mark an unrewarding branch. A full
//! colouring is a solution, the better the fewer colours it used.

use crate::problem::{Layered, Moves, Problem};
use ant_simulator::geometry::Point;
use ant_simulator::rng::Rng;
use ant_simulator::world::WorldConfig;

/// The colours given so far, node by node.
#[derive(Clone, Debug, PartialEq)]
pub struct ColourState {
    /// Colour of each node coloured so far.
    pub colours: Vec<u8>,
}

/// A graph and a palette. The nodes are coloured in order of falling
/// degree, the most constrained first, as the greedy heuristics do.
#[derive(Clone, Debug)]
pub struct Colouring {
    n: usize,
    /// Adjacency in colouring order.
    adjacency: Vec<Vec<usize>>,
    /// The original node at each position of the colouring order.
    order: Vec<usize>,
    edges: usize,
    palette: usize,
    layout: Layered,
}

impl Colouring {
    /// A graph of `n` nodes with the given edges, to be coloured from a
    /// palette of `palette` colours, on a grid of the given size.
    pub fn new(
        n: usize,
        edges: &[(usize, usize)],
        palette: usize,
        width: usize,
        height: usize,
    ) -> Colouring {
        assert!(n > 0 && n <= 255, "1 to 255 nodes");
        assert!(palette > 0 && palette <= 255, "1 to 255 colours");
        let mut original = vec![Vec::new(); n];
        let mut count = 0;
        for &(a, b) in edges {
            if a == b || a >= n || b >= n || original[a].contains(&b) {
                continue;
            }
            original[a].push(b);
            original[b].push(a);
            count += 1;
        }
        // Colouring order: by falling degree, ties by index.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&v| (std::cmp::Reverse(original[v].len()), v));
        let mut position = vec![0usize; n];
        for (i, &v) in order.iter().enumerate() {
            position[v] = i;
        }
        let adjacency: Vec<Vec<usize>> = order
            .iter()
            .map(|&v| original[v].iter().map(|&u| position[u]).collect())
            .collect();
        Colouring {
            n,
            adjacency,
            order,
            edges: count,
            palette,
            layout: Layered::new(width, height, n, palette),
        }
    }

    /// The original node coloured at each step of the order.
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    /// A state's colours by original node (`None` where not yet
    /// coloured).
    pub fn colours_by_node(&self, state: &ColourState) -> Vec<Option<u8>> {
        let mut out = vec![None; self.n];
        for (i, &c) in state.colours.iter().enumerate() {
            out[self.order[i]] = Some(c);
        }
        out
    }

    /// A random graph with a planted colouring: the nodes fall into
    /// `classes` classes and edges join nodes of different classes with
    /// probability `density`, so a colouring with `classes` colours
    /// exists; the palette offers `palette` colours.
    pub fn planted(
        n: usize,
        classes: usize,
        density: f64,
        palette: usize,
        width: usize,
        height: usize,
        seed: u64,
    ) -> Colouring {
        let mut rng = Rng::seed_from_u64(seed);
        let class: Vec<usize> = (0..n).map(|_| rng.below(classes.max(1))).collect();
        let mut edges = Vec::new();
        for a in 0..n {
            for b in a + 1..n {
                if class[a] != class[b] && rng.chance(density) {
                    edges.push((a, b));
                }
            }
        }
        Colouring::new(n, &edges, palette, width, height)
    }

    /// Number of nodes.
    pub fn nodes(&self) -> usize {
        self.n
    }

    /// Number of edges.
    pub fn edges(&self) -> usize {
        self.edges
    }

    /// Colours in the palette.
    pub fn palette(&self) -> usize {
        self.palette
    }

    /// The layout of the embedding.
    pub fn layout(&self) -> &Layered {
        &self.layout
    }

    /// Distinct colours used by a state.
    pub fn colours_used(&self, state: &ColourState) -> usize {
        let mut seen = vec![false; self.palette];
        for &c in &state.colours {
            if let Some(s) = seen.get_mut(c as usize) {
                *s = true;
            }
        }
        seen.iter().filter(|s| **s).count()
    }

    /// Whether every edge among the coloured nodes joins different colours.
    pub fn is_proper(&self, state: &ColourState) -> bool {
        self.conflicts(state) == 0
    }

    /// Edges among the coloured nodes joining nodes of one colour.
    pub fn conflicts(&self, state: &ColourState) -> usize {
        let k = state.colours.len();
        let mut n = 0;
        for a in 0..k {
            for &b in &self.adjacency[a] {
                if b > a && b < k && state.colours[a] == state.colours[b] {
                    n += 1;
                }
            }
        }
        n
    }

    fn allowed(&self, state: &ColourState, node: usize, colour: u8) -> bool {
        !self.adjacency[node]
            .iter()
            .any(|&j| j < state.colours.len() && state.colours[j] == colour)
    }
}

impl Problem for Colouring {
    type State = ColourState;

    fn embedding(&self) -> WorldConfig {
        self.layout.config()
    }

    fn origin(&self) -> ColourState {
        ColourState {
            colours: Vec::new(),
        }
    }

    fn place(&self, state: &ColourState) -> Point {
        match state.colours.last() {
            None => self.layout.origin(),
            Some(&c) => self.layout.place(state.colours.len() - 1, c as usize),
        }
    }

    fn moves(&self, state: &ColourState) -> Moves<ColourState> {
        let node = state.colours.len();
        if node >= self.n {
            return Moves::States(Vec::new());
        }
        let out = (0..self.palette as u8)
            .filter(|&c| self.allowed(state, node, c))
            .map(|c| {
                let mut colours = state.colours.clone();
                colours.push(c);
                ColourState { colours }
            })
            .collect();
        Moves::States(out)
    }

    fn quality(&self, state: &ColourState) -> Option<f64> {
        (state.colours.len() == self.n).then(|| {
            let used = self.colours_used(state);
            (self.palette + 1 - used) as f64 / self.palette as f64
        })
    }

    /// A colour already in use smells of 1, a new one of nothing: the
    /// fewer colours the better.
    fn scent(&self, from: &ColourState, to: &ColourState) -> f64 {
        match to.colours.last() {
            Some(c) if from.colours.contains(c) => 1.0,
            _ => 0.0,
        }
    }

    fn describe(&self, state: &ColourState) -> String {
        let s: String = self
            .colours_by_node(state)
            .iter()
            .map(|c| match c {
                Some(c) => char::from_digit((c % 36) as u32, 36).unwrap_or('?'),
                None => '.',
            })
            .collect();
        format!(
            "{} of {} nodes coloured with {} colours: {}",
            state.colours.len(),
            self.n,
            self.colours_used(state),
            s
        )
    }

    fn name(&self) -> String {
        "colouring".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_triangle_needs_three_colours_and_a_square_two() {
        let triangle = Colouring::new(3, &[(0, 1), (1, 2), (2, 0)], 3, 32, 24);
        let mut state = triangle.origin();
        for expected in [3usize, 2, 1] {
            let Moves::States(next) = triangle.moves(&state) else {
                panic!("states")
            };
            assert_eq!(next.len(), expected, "colours left for the next node");
            state = next[0].clone();
            assert!(triangle.is_proper(&state));
        }
        assert_eq!(triangle.colours_used(&state), 3);
        assert!((triangle.quality(&state).unwrap() - 1.0 / 3.0).abs() < 1e-9);
        let square = Colouring::new(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], 2, 32, 24);
        let mut state = square.origin();
        while state.colours.len() < 4 {
            let Moves::States(next) = square.moves(&state) else {
                panic!("states")
            };
            state = next[0].clone();
        }
        assert!(square.is_proper(&state));
        assert_eq!(square.colours_used(&state), 2);
        assert!((square.quality(&state).unwrap() - 0.5).abs() < 1e-9);
        // Two colours cannot do a triangle: a dead end.
        let stuck = Colouring::new(3, &[(0, 1), (1, 2), (2, 0)], 2, 32, 24);
        let mut state = stuck.origin();
        for _ in 0..2 {
            let Moves::States(next) = stuck.moves(&state) else {
                panic!("states")
            };
            state = next[0].clone();
        }
        assert!(matches!(stuck.moves(&state), Moves::States(v) if v.is_empty()));
        assert!(stuck.quality(&state).is_none());
    }

    #[test]
    fn the_order_is_by_falling_degree_and_describes_by_node() {
        let star = Colouring::new(4, &[(3, 0), (3, 1), (3, 2)], 3, 32, 24);
        assert_eq!(star.order()[0], 3, "the hub is coloured first");
        let mut state = star.origin();
        let Moves::States(next) = star.moves(&state) else {
            panic!("states")
        };
        state = next[0].clone();
        let by_node = star.colours_by_node(&state);
        assert_eq!(by_node, vec![None, None, None, Some(0)]);
        assert!(star.describe(&state).starts_with("1 of 4 nodes"));
    }
}
