//! Quadkeys and quadtrees: hierarchical addresses over the grid.
//!
//! A [`QuadKey`] names a square node of the quadtree that covers the
//! grid: the root is the whole square, and each level splits every node
//! into four. The key is the node's level and its Z-order (Morton) index
//! at that level, the interleaved bits of its column and row, so that a
//! parent's code is its child's code shifted right by two and the four
//! children of a node are contiguous. Keys print as digit strings from
//! the root down (`0` north-west, `1` north-east, `2` south-west, `3`
//! south-east), the convention of map tiles.
//!
//! A [`QuadTree`] stores one value per node at every level. Anything
//! whose value at a node is the sum of its children's (counts, moments,
//! flows) composes bottom-up through [`QuadTree::update_path`], which
//! applies a change to a leaf and every node above it, so that a reading
//! at any grain is a direct lookup.

use crate::geometry::Position;
use std::fmt;

/// Address of a square node of the quadtree over the grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QuadKey {
    /// Depth below the root (0 the whole square, each level halves the
    /// side).
    pub level: u8,
    /// Z-order index of the node among the nodes of its level.
    pub code: u64,
}

/// Spread the low 32 bits of `v` over the even bit positions.
fn spread(v: u32) -> u64 {
    let mut x = v as u64;
    x = (x | (x << 16)) & 0x0000_ffff_0000_ffff;
    x = (x | (x << 8)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555;
    x
}

/// Gather the even bit positions of `x` into the low 32 bits.
fn gather(mut x: u64) -> u32 {
    x &= 0x5555_5555_5555_5555;
    x = (x | (x >> 1)) & 0x3333_3333_3333_3333;
    x = (x | (x >> 2)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x >> 4)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x >> 8)) & 0x0000_ffff_0000_ffff;
    x = (x | (x >> 16)) & 0x0000_0000_ffff_ffff;
    x as u32
}

impl QuadKey {
    /// The whole square.
    pub const ROOT: QuadKey = QuadKey { level: 0, code: 0 };

    /// The node at `level` in column `col` and row `row` of that level.
    pub fn new(level: u8, col: u32, row: u32) -> QuadKey {
        QuadKey {
            level,
            code: spread(col) | (spread(row) << 1),
        }
    }

    /// Column of the node among the nodes of its level.
    pub fn col(self) -> u32 {
        gather(self.code)
    }

    /// Row of the node among the nodes of its level.
    pub fn row(self) -> u32 {
        gather(self.code >> 1)
    }

    /// The node containing this one one level up (none for the root).
    pub fn parent(self) -> Option<QuadKey> {
        if self.level == 0 {
            None
        } else {
            Some(QuadKey {
                level: self.level - 1,
                code: self.code >> 2,
            })
        }
    }

    /// The node containing this one at a coarser (or equal) level.
    pub fn ancestor(self, level: u8) -> QuadKey {
        let level = level.min(self.level);
        QuadKey {
            level,
            code: self.code >> (2 * (self.level - level) as u32),
        }
    }

    /// The containing nodes from the parent up to the root.
    pub fn ancestors(self) -> impl Iterator<Item = QuadKey> {
        let mut key = self;
        std::iter::from_fn(move || {
            key = key.parent()?;
            Some(key)
        })
    }

    /// One of the four nodes this one splits into.
    pub fn child(self, quadrant: u8) -> QuadKey {
        QuadKey {
            level: self.level + 1,
            code: (self.code << 2) | (quadrant as u64 & 3),
        }
    }

    /// The four nodes this one splits into, north-west first.
    pub fn children(self) -> [QuadKey; 4] {
        [self.child(0), self.child(1), self.child(2), self.child(3)]
    }

    /// Which child of its parent this node is (0 for the root).
    pub fn quadrant(self) -> u8 {
        if self.level == 0 {
            0
        } else {
            (self.code & 3) as u8
        }
    }

    /// Whether `other` lies within this node (a node contains itself).
    pub fn contains(self, other: QuadKey) -> bool {
        other.level >= self.level && other.ancestor(self.level) == self
    }

    /// The key as digits from the root down; the root is the empty
    /// string.
    pub fn digits(self) -> String {
        (0..self.level)
            .map(|i| {
                let shift = 2 * (self.level - 1 - i) as u32;
                char::from(b'0' + ((self.code >> shift) & 3) as u8)
            })
            .collect()
    }

    /// A key from its digits.
    pub fn parse(digits: &str) -> Option<QuadKey> {
        let mut key = QuadKey::ROOT;
        for c in digits.chars() {
            let q = c.to_digit(4)? as u8;
            key = key.child(q);
        }
        Some(key)
    }
}

impl fmt::Display for QuadKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.level == 0 {
            write!(f, "root")
        } else {
            write!(f, "{}", self.digits())
        }
    }
}

/// One value per node of the quadtree over a grid.
#[derive(Clone, Debug)]
pub struct QuadTree<T> {
    width: usize,
    height: usize,
    levels: u8,
    nodes: Vec<Vec<T>>,
}

impl<T: Clone + Default> QuadTree<T> {
    /// A tree over a `width` × `height` grid: the root is the smallest
    /// power-of-two square covering it, the leaves are its cells.
    pub fn new(width: usize, height: usize) -> QuadTree<T> {
        let mut levels = 0u8;
        while (1usize << levels) < width.max(height).max(1) {
            levels += 1;
        }
        let nodes = (0..=levels)
            .map(|l| vec![T::default(); 1usize << (2 * l as usize)])
            .collect();
        QuadTree {
            width,
            height,
            levels,
            nodes,
        }
    }

    /// Levels below the root: the leaf level.
    pub fn levels(&self) -> u8 {
        self.levels
    }

    /// Side of the root square, cells.
    pub fn side(&self) -> usize {
        1 << self.levels
    }

    /// Width of the grid, cells.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Height of the grid, cells.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Cells per side of a node at a level.
    pub fn node_side(&self, level: u8) -> usize {
        1 << (self.levels - level.min(self.levels))
    }

    /// The finest level whose nodes are at least `side` cells across
    /// (the leaf level for anything smaller than a cell).
    pub fn level_for_side(&self, side: usize) -> u8 {
        let mut level = self.levels;
        while level > 0 && self.node_side(level) < side.max(1) {
            level -= 1;
        }
        level
    }

    /// Columns and rows of nodes at a level that meet the grid.
    pub fn extent(&self, level: u8) -> (usize, usize) {
        let side = self.node_side(level);
        (self.width.div_ceil(side), self.height.div_ceil(side))
    }

    /// The node at `level` containing a cell, if the cell is on the grid.
    pub fn key_of(&self, cell: Position, level: u8) -> Option<QuadKey> {
        if cell.x < 0
            || cell.y < 0
            || cell.x as usize >= self.width
            || cell.y as usize >= self.height
        {
            return None;
        }
        let shift = (self.levels - level.min(self.levels)) as u32;
        Some(QuadKey::new(
            level.min(self.levels),
            (cell.x as u32) >> shift,
            (cell.y as u32) >> shift,
        ))
    }

    /// The leaf holding a cell.
    pub fn leaf(&self, cell: Position) -> Option<QuadKey> {
        self.key_of(cell, self.levels)
    }

    /// The cells of a node, as `(x0, y0, x1, y1)` with exclusive far
    /// corners, clipped to the grid.
    pub fn rect(&self, key: QuadKey) -> (usize, usize, usize, usize) {
        let side = self.node_side(key.level);
        let x0 = key.col() as usize * side;
        let y0 = key.row() as usize * side;
        (
            x0.min(self.width),
            y0.min(self.height),
            (x0 + side).min(self.width),
            (y0 + side).min(self.height),
        )
    }

    /// Whether a node meets the grid.
    pub fn on_grid(&self, key: QuadKey) -> bool {
        let side = self.node_side(key.level);
        key.level <= self.levels
            && (key.col() as usize) * side < self.width
            && (key.row() as usize) * side < self.height
    }

    /// The value at a node.
    pub fn get(&self, key: QuadKey) -> &T {
        &self.nodes[key.level as usize][key.code as usize]
    }

    /// The value at a node, mutably.
    pub fn get_mut(&mut self, key: QuadKey) -> &mut T {
        &mut self.nodes[key.level as usize][key.code as usize]
    }

    /// The nodes of a level that meet the grid, in Z-order.
    pub fn keys(&self, level: u8) -> Vec<QuadKey> {
        let level = level.min(self.levels);
        (0..self.nodes[level as usize].len() as u64)
            .map(|code| QuadKey { level, code })
            .filter(|&k| self.on_grid(k))
            .collect()
    }

    /// The nodes of a level that meet the grid, with their values.
    pub fn nodes(&self, level: u8) -> impl Iterator<Item = (QuadKey, &T)> {
        self.keys(level).into_iter().map(move |k| (k, self.get(k)))
    }

    /// Apply a change to the leaf holding a cell and to every node above
    /// it, leaf first.
    pub fn update_path(&mut self, cell: Position, mut f: impl FnMut(&mut T)) {
        let Some(leaf) = self.leaf(cell) else {
            return;
        };
        f(self.get_mut(leaf));
        for key in leaf.ancestors() {
            f(self.get_mut(key));
        }
    }

    /// Every value, at every level.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.nodes.iter_mut().flat_map(|level| level.iter_mut())
    }

    /// Rebuild every node above the leaves as the composition of its
    /// children (for values that are sums of their parts).
    pub fn compose(&mut self, mut add: impl FnMut(&mut T, &T)) {
        for level in (0..self.levels as usize).rev() {
            for code in 0..self.nodes[level].len() {
                let mut acc = T::default();
                for q in 0..4u64 {
                    let child = &self.nodes[level + 1][(code as u64 * 4 + q) as usize];
                    add(&mut acc, child);
                }
                self.nodes[level][code] = acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_interleave_parse_and_print() {
        let k = QuadKey::new(3, 5, 2);
        assert_eq!((k.col(), k.row()), (5, 2));
        assert_eq!(k.digits(), "121");
        assert_eq!(QuadKey::parse("121"), Some(k));
        assert_eq!(k.to_string(), "121");
        assert_eq!(QuadKey::ROOT.to_string(), "root");
        assert_eq!(k.parent(), Some(QuadKey::new(2, 2, 1)));
        assert_eq!(k.parent().unwrap().child(k.quadrant()), k);
        assert_eq!(k.ancestor(1), QuadKey::new(1, 1, 0));
        assert_eq!(k.ancestors().count(), 3);
        assert!(QuadKey::new(1, 1, 0).contains(k) && !QuadKey::new(1, 0, 0).contains(k));
        assert_eq!(QuadKey::ROOT.children()[3], QuadKey::new(1, 1, 1));
        assert_eq!(QuadKey::parse("4"), None);
        // Z-order keeps siblings contiguous.
        let codes: Vec<u64> = QuadKey::new(2, 1, 1)
            .children()
            .iter()
            .map(|c| c.code)
            .collect();
        assert_eq!(
            codes,
            vec![codes[0], codes[0] + 1, codes[0] + 2, codes[0] + 3]
        );
    }

    #[test]
    fn tree_covers_the_grid_and_composes() {
        let mut t: QuadTree<u64> = QuadTree::new(20, 12);
        assert_eq!(t.levels(), 5);
        assert_eq!(t.side(), 32);
        assert_eq!(t.node_side(3), 4);
        assert_eq!(t.level_for_side(8), 2);
        assert_eq!(
            t.level_for_side(5),
            2,
            "the finest level at least five across"
        );
        assert_eq!(t.level_for_side(1), 5);
        assert_eq!(t.extent(3), (5, 3));
        assert_eq!(t.keys(3).len(), 15);
        assert_eq!(t.keys(0), vec![QuadKey::ROOT]);
        let leaf = t.leaf(Position::new(19, 11)).unwrap();
        assert_eq!(t.rect(leaf), (19, 11, 20, 12));
        assert_eq!(t.rect(leaf.ancestor(3)), (16, 8, 20, 12));
        assert!(t.leaf(Position::new(20, 0)).is_none());
        t.update_path(Position::new(19, 11), |v| *v += 1);
        t.update_path(Position::new(0, 0), |v| *v += 2);
        assert_eq!(*t.get(QuadKey::ROOT), 3);
        assert_eq!(*t.get(leaf), 1);
        assert_eq!(*t.get(leaf.ancestor(1)), 1);
        assert_eq!(*t.get(QuadKey::new(1, 0, 0)), 2);
        // Composition from the leaves rebuilds the same sums.
        for level in 0..5u8 {
            for k in t.keys(level) {
                *t.get_mut(k) = 0;
            }
        }
        t.compose(|acc, child| *acc += child);
        assert_eq!(*t.get(QuadKey::ROOT), 3);
        assert_eq!(*t.get(QuadKey::new(1, 1, 0)), 1);
    }
}
