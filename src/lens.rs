//! The lens: a two-position structure on the quadtree, and the geodesic
//! over it.
//!
//! Between two points the tree is tessellated so that its resolution
//! follows the distance to the nearer of them: cells within `radius` of
//! either end, and beyond it nodes that double in size with their
//! distance from the nearer end. The tessellation is the same whichever
//! end is named first, so anything computed on it is invariant under
//! exchanging the ends, and it has a number of leaves that grows with the
//! logarithm of the span rather than with the span, so a route between
//! the ends is found on a graph of a few hundred nodes on any grid. A
//! node that holds both walls and open ground is refined wherever it
//! lies, so the geodesic respects walls exactly; a node walled through
//! is an impassable leaf. The route is the least-cost path between the
//! leaves' centres, pulled straight afterwards wherever the line of
//! sight is clear.

use crate::geometry::{Point, Position};
use crate::quad::{QuadKey, QuadTree};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// What a node of the tree is made of.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ground {
    /// Open throughout, crossed at this cost per cell.
    Open(f64),
    /// Walls and open ground both: to be refined until its parts are one
    /// or the other.
    Mixed,
    /// Walled through: impassable.
    Blocked,
}

/// A leaf of the lens: one node of the tree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Leaf {
    /// The node.
    pub key: QuadKey,
    /// Its cells, `(x0, y0, x1, y1)` with exclusive far corners, clipped
    /// to the grid.
    pub rect: (usize, usize, usize, usize),
    /// Cost per cell of crossing it, infinite when it is walled through.
    pub cost: f64,
}

impl Leaf {
    /// The centre of the leaf.
    pub fn centre(&self) -> Point {
        let (x0, y0, x1, y1) = self.rect;
        Point::new((x0 + x1) as f64 / 2.0, (y0 + y1) as f64 / 2.0)
    }

    /// Cells across the leaf (its longer extent, where the grid clips it).
    pub fn side(&self) -> usize {
        let (x0, y0, x1, y1) = self.rect;
        (x1 - x0).max(y1 - y0)
    }

    /// Whether the leaf can be crossed.
    pub fn passable(&self) -> bool {
        self.cost.is_finite()
    }

    /// Whether a cell lies in the leaf.
    pub fn contains(&self, cell: Position) -> bool {
        let (x0, y0, x1, y1) = self.rect;
        cell.x >= x0 as i32
            && cell.y >= y0 as i32
            && (cell.x as usize) < x1
            && (cell.y as usize) < y1
    }
}

/// Distance from a point to a rectangle of cells (zero inside it).
fn distance_to_rect(p: Point, rect: (usize, usize, usize, usize)) -> f64 {
    let (x0, y0, x1, y1) = rect;
    let dx = (x0 as f64 - p.x).max(0.0).max(p.x - x1 as f64);
    let dy = (y0 as f64 - p.y).max(0.0).max(p.y - y1 as f64);
    (dx * dx + dy * dy).sqrt()
}

/// Length of the side two rectangles share (zero when they only touch
/// at a corner or not at all).
fn shared_side(a: (usize, usize, usize, usize), b: (usize, usize, usize, usize)) -> f64 {
    let (ax0, ay0, ax1, ay1) = a;
    let (bx0, by0, bx1, by1) = b;
    if ax1 == bx0 || bx1 == ax0 {
        (ay1.min(by1) as f64 - ay0.max(by0) as f64).max(0.0)
    } else if ay1 == by0 || by1 == ay0 {
        (ax1.min(bx1) as f64 - ax0.max(bx0) as f64).max(0.0)
    } else {
        0.0
    }
}

/// A route between the two ends of a lens.
#[derive(Clone, Debug, PartialEq)]
pub struct Geodesic {
    /// The points passed through, from one end to the other.
    pub points: Vec<Point>,
    /// The cost of the path over the lens's leaves.
    pub cost: f64,
    /// Leaves the path crossed.
    pub leaves: usize,
}

impl Geodesic {
    /// Length of the route, cells.
    pub fn length(&self) -> f64 {
        self.points.windows(2).map(|w| w[0].distance(w[1])).sum()
    }

    /// Pull the route straight: skip every point that a clear line of
    /// sight makes unnecessary.
    pub fn pull(&mut self, passable: impl Fn(Point, Point) -> bool) {
        if self.points.len() < 3 {
            return;
        }
        let mut kept = vec![self.points[0]];
        let mut i = 0;
        while i + 1 < self.points.len() {
            let mut j = self.points.len() - 1;
            while j > i + 1 && !passable(self.points[i], self.points[j]) {
                j -= 1;
            }
            kept.push(self.points[j]);
            i = j;
        }
        self.points = kept;
    }
}

/// The tessellation between two points.
#[derive(Clone, Debug)]
pub struct Lens {
    a: Point,
    b: Point,
    radius: f64,
    levels: u8,
    leaves: Vec<Leaf>,
    index: HashMap<QuadKey, usize>,
    /// Per leaf, its neighbours across a side and the length of the side
    /// shared.
    adjacency: Vec<Vec<(usize, f64)>>,
}

impl Lens {
    /// The lens between `a` and `b` over a tree: cells within `radius`
    /// of either, nodes doubling with the distance beyond, with `ground`
    /// saying what every node is made of.
    pub fn new<T: Clone + Default>(
        tree: &QuadTree<T>,
        a: Point,
        b: Point,
        radius: f64,
        ground: impl Fn(QuadKey, (usize, usize, usize, usize)) -> Ground,
    ) -> Lens {
        let levels = tree.levels();
        let mut leaves = Vec::new();
        let mut stack = vec![QuadKey::ROOT];
        while let Some(key) = stack.pop() {
            if !tree.on_grid(key) {
                continue;
            }
            let rect = tree.rect(key);
            let what = ground(key, rect);
            let side = tree.node_side(key.level) as f64;
            let near = distance_to_rect(a, rect).min(distance_to_rect(b, rect));
            let refine =
                key.level < levels && (matches!(what, Ground::Mixed) || near < radius + side);
            if refine {
                for child in key.children() {
                    stack.push(child);
                }
            } else {
                let cost = match what {
                    Ground::Open(c) => c.max(0.0),
                    Ground::Blocked => f64::INFINITY,
                    Ground::Mixed => 1.0,
                };
                leaves.push(Leaf { key, rect, cost });
            }
        }
        leaves.sort_by_key(|l| l.key);
        let index: HashMap<QuadKey, usize> =
            leaves.iter().enumerate().map(|(i, l)| (l.key, i)).collect();
        let mut adjacency: Vec<Vec<(usize, f64)>> = vec![Vec::new(); leaves.len()];
        for i in 0..leaves.len() {
            let key = leaves[i].key;
            let (col, row) = (key.col() as i64, key.row() as i64);
            for side in 0..4u8 {
                let (ncol, nrow) = match side {
                    0 => (col, row - 1),
                    1 => (col + 1, row),
                    2 => (col, row + 1),
                    _ => (col - 1, row),
                };
                if ncol < 0 || nrow < 0 {
                    continue;
                }
                let nkey = QuadKey::new(key.level, ncol as u32, nrow as u32);
                if !tree.on_grid(nkey) {
                    continue;
                }
                // The neighbour is that node, an ancestor of it, or the
                // leaves among its descendants along the facing side.
                let mut found = Vec::new();
                let mut k = nkey;
                loop {
                    if let Some(&j) = index.get(&k) {
                        found.push(j);
                        break;
                    }
                    match k.parent() {
                        Some(p) => k = p,
                        None => break,
                    }
                }
                if found.is_empty() {
                    let facing: [u8; 2] = match side {
                        0 => [2, 3],
                        1 => [0, 2],
                        2 => [0, 1],
                        _ => [1, 3],
                    };
                    let mut down = vec![nkey];
                    while let Some(k) = down.pop() {
                        if !tree.on_grid(k) {
                            continue;
                        }
                        if let Some(&j) = index.get(&k) {
                            found.push(j);
                        } else if k.level < levels {
                            for q in facing {
                                down.push(k.child(q));
                            }
                        }
                    }
                }
                for j in found {
                    let shared = shared_side(leaves[i].rect, leaves[j].rect);
                    if shared <= 0.0 || adjacency[i].iter().any(|&(n, _)| n == j) {
                        continue;
                    }
                    adjacency[i].push((j, shared));
                    adjacency[j].push((i, shared));
                }
            }
        }
        Lens {
            a,
            b,
            radius,
            levels,
            leaves,
            index,
            adjacency,
        }
    }

    /// The two ends.
    pub fn ends(&self) -> (Point, Point) {
        (self.a, self.b)
    }

    /// The radius within which the lens is made of cells.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// The leaves, in key order.
    pub fn leaves(&self) -> &[Leaf] {
        &self.leaves
    }

    /// The neighbours of a leaf and the length of the side shared with
    /// each.
    pub fn neighbours(&self, leaf: usize) -> &[(usize, f64)] {
        &self.adjacency[leaf]
    }

    /// The leaf holding a cell.
    pub fn leaf_of(&self, cell: Position) -> Option<usize> {
        if cell.x < 0 || cell.y < 0 {
            return None;
        }
        for level in (0..=self.levels).rev() {
            let shift = (self.levels - level) as u32;
            let key = QuadKey::new(level, (cell.x as u32) >> shift, (cell.y as u32) >> shift);
            if let Some(&i) = self.index.get(&key) {
                return Some(i);
            }
        }
        None
    }

    /// The least-cost path from one end to the other over the leaves:
    /// the ends, then the centres of the leaves crossed between them.
    pub fn geodesic(&self) -> Option<Geodesic> {
        let start = self.leaf_of(self.a.cell())?;
        let goal = self.leaf_of(self.b.cell())?;
        if !self.leaves[start].passable() || !self.leaves[goal].passable() {
            return None;
        }
        #[derive(PartialEq)]
        struct Item(f64, usize);
        impl Eq for Item {}
        impl PartialOrd for Item {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Item {
            fn cmp(&self, other: &Self) -> Ordering {
                // A min-heap on cost, ties by leaf index.
                other
                    .0
                    .partial_cmp(&self.0)
                    .unwrap_or(Ordering::Equal)
                    .then(other.1.cmp(&self.1))
            }
        }
        let n = self.leaves.len();
        let mut dist = vec![f64::INFINITY; n];
        let mut prev = vec![usize::MAX; n];
        let mut done = vec![false; n];
        let mut heap = BinaryHeap::new();
        dist[start] = 0.0;
        heap.push(Item(0.0, start));
        while let Some(Item(d, i)) = heap.pop() {
            if done[i] {
                continue;
            }
            done[i] = true;
            if i == goal {
                break;
            }
            let ci = self.leaves[i].centre();
            for &(j, _) in &self.adjacency[i] {
                if done[j] || !self.leaves[j].passable() {
                    continue;
                }
                let cj = self.leaves[j].centre();
                let step = ci.distance(cj) * 0.5 * (self.leaves[i].cost + self.leaves[j].cost);
                let nd = d + step;
                if nd < dist[j] {
                    dist[j] = nd;
                    prev[j] = i;
                    heap.push(Item(nd, j));
                }
            }
        }
        if !dist[goal].is_finite() {
            return None;
        }
        let mut path = vec![goal];
        let mut k = goal;
        while k != start {
            k = prev[k];
            path.push(k);
        }
        path.reverse();
        let mut points = Vec::with_capacity(path.len() + 2);
        points.push(self.a);
        for &i in &path[1..path.len().saturating_sub(1)] {
            points.push(self.leaves[i].centre());
        }
        if path.len() == 1 {
            // Both ends in one leaf.
        }
        points.push(self.b);
        Some(Geodesic {
            points,
            cost: dist[goal],
            leaves: path.len(),
        })
    }

    /// The lens drawn cell by cell: `·` for cells, a digit for the
    /// doubling of a coarser leaf's side (`1` two cells, `2` four, …),
    /// `#` for walled-through leaves, `A` and `B` for the ends and `*`
    /// along a route.
    pub fn render(&self, route: Option<&Geodesic>) -> String {
        let (mut w, mut h) = (0usize, 0usize);
        for l in &self.leaves {
            w = w.max(l.rect.2);
            h = h.max(l.rect.3);
        }
        let mut grid = vec![vec![' '; w]; h];
        for l in &self.leaves {
            let (x0, y0, x1, y1) = l.rect;
            let ch = if !l.passable() {
                '#'
            } else {
                let mut side = l.side();
                let mut n = 0u32;
                while side > 1 {
                    side >>= 1;
                    n += 1;
                }
                if n == 0 {
                    '·'
                } else {
                    char::from_digit(n.min(9), 10).unwrap_or('9')
                }
            };
            for row in grid.iter_mut().take(y1).skip(y0) {
                for slot in row.iter_mut().take(x1).skip(x0) {
                    *slot = ch;
                }
            }
        }
        if let Some(r) = route {
            for pair in r.points.windows(2) {
                let (p, q) = (pair[0], pair[1]);
                let steps = (p.distance(q) * 2.0).ceil().max(1.0) as usize;
                for s in 0..=steps {
                    let f = s as f64 / steps as f64;
                    let c = Point::new(p.x + (q.x - p.x) * f, p.y + (q.y - p.y) * f).cell();
                    if c.x >= 0 && c.y >= 0 && (c.x as usize) < w && (c.y as usize) < h {
                        grid[c.y as usize][c.x as usize] = '*';
                    }
                }
            }
        }
        for (mark, p) in [('A', self.a), ('B', self.b)] {
            let c = p.cell();
            if c.x >= 0 && c.y >= 0 && (c.x as usize) < w && (c.y as usize) < h {
                grid[c.y as usize][c.x as usize] = mark;
            }
        }
        grid.into_iter()
            .map(|row| row.into_iter().collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// The geodesic between two points of a world, round its walls, pulled
/// straight where the line of sight is clear.
pub fn geodesic(world: &crate::world::World, a: Point, b: Point, radius: f64) -> Option<Geodesic> {
    let tree = QuadTree::<()>::new(world.width(), world.height());
    let lens = Lens::new(&tree, a, b, radius, |key, rect| world.ground(key, rect));
    let mut r = lens.geodesic()?;
    r.pull(|p, q| world.segment_passable(p, q));
    Some(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::world::{Rect, World, WorldConfig};

    fn open(w: usize, h: usize) -> QuadTree<()> {
        QuadTree::<()>::new(w, h)
    }

    #[test]
    fn the_lens_is_fine_at_both_ends_coarse_between_and_the_same_from_either_end() {
        let tree = open(128, 128);
        let a = Point::new(10.5, 10.5);
        let b = Point::new(117.5, 117.5);
        let lens = Lens::new(&tree, a, b, 2.0, |_, _| Ground::Open(1.0));
        let n = lens.leaves().len();
        assert!(n < 400, "a few hundred leaves, not 16384 cells: {n}");
        let at = |p: Point| lens.leaves()[lens.leaf_of(p.cell()).unwrap()];
        assert_eq!(at(a).side(), 1);
        assert_eq!(at(b).side(), 1);
        assert!(at(Point::new(64.0, 64.0)).side() >= 8, "coarse between");
        let back = Lens::new(&tree, b, a, 2.0, |_, _| Ground::Open(1.0));
        assert_eq!(
            lens.leaves(),
            back.leaves(),
            "the same tessellation from either end"
        );
        // Every leaf has neighbours, and adjacency is symmetric.
        for i in 0..n {
            assert!(!lens.neighbours(i).is_empty());
            for &(j, s) in lens.neighbours(i) {
                assert!(lens
                    .neighbours(j)
                    .iter()
                    .any(|&(k, t)| k == i && (t - s).abs() < 1e-9));
            }
        }
        // The leaves tile the grid.
        let area: usize = lens
            .leaves()
            .iter()
            .map(|l| (l.rect.2 - l.rect.0) * (l.rect.3 - l.rect.1))
            .sum();
        assert_eq!(area, 128 * 128);
    }

    #[test]
    fn the_geodesic_is_symmetric_and_the_lens_grows_with_the_log_of_the_span() {
        let tree = open(256, 256);
        let count = |span: f64| {
            let a = Point::new(8.5, 128.5);
            let b = Point::new(8.5 + span, 128.5);
            Lens::new(&tree, a, b, 2.0, |_, _| Ground::Open(1.0))
                .leaves()
                .len()
        };
        let (n32, n128, n240) = (count(32.0), count(128.0), count(240.0));
        assert!(n128 < 2 * n32 && n240 < 2 * n32, "{n32} {n128} {n240}");
        let a = Point::new(8.5, 128.5);
        let b = Point::new(200.5, 40.5);
        let there = Lens::new(&tree, a, b, 2.0, |_, _| Ground::Open(1.0))
            .geodesic()
            .unwrap();
        let back = Lens::new(&tree, b, a, 2.0, |_, _| Ground::Open(1.0))
            .geodesic()
            .unwrap();
        assert!((there.cost - back.cost).abs() < 1e-9);
        let mut pulled = there.clone();
        pulled.pull(|_, _| true);
        assert!(
            (pulled.length() - a.distance(b)).abs() < 1e-9,
            "pulled straight in the open"
        );
    }

    #[test]
    fn the_geodesic_goes_round_walls() {
        // A wall down the middle with a gap at the bottom.
        let cfg = WorldConfig {
            width: 40,
            height: 40,
            nest: Position::new(5, 5),
            random_food: None,
            walls: vec![Rect::new(Position::new(20, 0), Position::new(20, 31))],
            ..WorldConfig::default()
        };
        let world = World::new(cfg, &mut Rng::seed_from_u64(1));
        let a = Point::new(5.5, 20.5);
        let b = Point::new(35.5, 20.5);
        let there = geodesic(&world, a, b, 2.0).expect("a way round");
        let back = geodesic(&world, b, a, 2.0).expect("a way round");
        let straight = a.distance(b);
        assert!(
            there.length() > straight + 10.0,
            "round the wall: {}",
            there.length()
        );
        assert!((there.length() - back.length()).abs() < 0.05 * there.length());
        for pair in there.points.windows(2) {
            assert!(world.segment_passable(pair[0], pair[1]));
        }
        // Through the gap: some point of the route lies below the wall's end.
        assert!(there.points.iter().any(|p| p.y > 31.0));
        // The lens refined the wall to cells and left the open corners coarse.
        let tree = QuadTree::<()>::new(40, 40);
        let lens = Lens::new(&tree, a, b, 2.0, |key, rect| world.ground(key, rect));
        assert!(lens.leaves().iter().any(|l| !l.passable() && l.side() == 1));
        assert!(lens.leaves().iter().any(|l| l.passable() && l.side() >= 4));
        let picture = lens.render(Some(&there));
        assert!(picture.contains('#') && picture.contains('*') && picture.contains('A'));
    }
}
