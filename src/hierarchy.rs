//! The hierarchical class system.
//!
//! A [`Hierarchy`] is a tree of [`Node`]s. Every node is the same general
//! object: it carries a [`BehavioralSurface`] (weights plus an entropy dial)
//! and it controls the behaviour of everything beneath it. Ants hang off the
//! leaves. The effective policy an ant acts through is the composition of the
//! surfaces along the path from the root to its leaf:
//!
//! * weights are summed, so a parent biases all descendants and a child adds
//!   its own specialisation on top;
//! * entropy dials compose, so the root sets a colony-wide level of disorder
//!   and each level below scales what it inherits.
//!
//! The whole hierarchy is also a single flat parameter vector (one
//! [`PARAM_LEN`] block per node, in preorder), which is what the learning
//! machinery reads and writes.

use crate::ant::FEATURES;
use crate::surface::{BehavioralSurface, SurfaceError, PARAM_LEN};
use std::fmt::Write as _;
use std::ops::Range;

/// Identifier of a node (its preorder index).
pub type NodeId = usize;

/// One level of the tree beneath the root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LevelSpec {
    /// Name of the category at this level (e.g. "caste").
    pub name: String,
    /// Number of children each node of the previous level gets.
    pub branching: usize,
}

impl LevelSpec {
    /// Convenience constructor.
    pub fn new(name: impl Into<String>, branching: usize) -> Self {
        LevelSpec {
            name: name.into(),
            branching,
        }
    }
}

/// Shape of a hierarchy: a root plus a list of levels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HierarchySpec {
    /// Name of the root category.
    pub root_name: String,
    /// Levels beneath the root, top-down.
    pub levels: Vec<LevelSpec>,
}

impl Default for HierarchySpec {
    /// `colony → 3 castes → 2 squads each` (10 nodes, 6 leaves).
    fn default() -> Self {
        HierarchySpec {
            root_name: "colony".to_string(),
            levels: vec![LevelSpec::new("caste", 3), LevelSpec::new("squad", 2)],
        }
    }
}

impl HierarchySpec {
    /// A hierarchy consisting of the root alone.
    pub fn flat() -> Self {
        HierarchySpec {
            root_name: "colony".to_string(),
            levels: Vec::new(),
        }
    }

    /// Total number of nodes the spec produces.
    pub fn node_count(&self) -> usize {
        let mut total = 1;
        let mut width = 1;
        for level in &self.levels {
            width *= level.branching;
            total += width;
        }
        total
    }
}

/// The general, entropically controllable object at every level of the tree.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// Preorder index.
    pub id: NodeId,
    /// Parent node, `None` for the root.
    pub parent: Option<NodeId>,
    /// Distance from the root.
    pub depth: usize,
    /// Category name of the level ("colony", "caste", ...).
    pub level: String,
    /// Unique name, e.g. `caste-1`.
    pub name: String,
    /// The node's entropic behavioral surface.
    pub surface: BehavioralSurface,
    /// Child nodes.
    pub children: Vec<NodeId>,
}

impl Node {
    /// Whether the node has no children.
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// The policy an ant actually acts through, after composing its path.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectivePolicy {
    /// Sum of weights along the path.
    pub weights: [f64; FEATURES],
    /// Composed entropy fraction in `[0, 1]`.
    pub entropy_fraction: f64,
}

/// A tree of controllable nodes.
#[derive(Clone, Debug, PartialEq)]
pub struct Hierarchy {
    nodes: Vec<Node>,
    leaves: Vec<NodeId>,
    paths: Vec<Vec<NodeId>>,
}

impl Hierarchy {
    /// Build a hierarchy from a spec. The root receives `root_surface`; every
    /// other node starts [`BehavioralSurface::neutral`], so initially all
    /// ants behave exactly like the root says.
    pub fn from_spec(spec: &HierarchySpec, root_surface: BehavioralSurface) -> Self {
        let mut nodes = vec![Node {
            id: 0,
            parent: None,
            depth: 0,
            level: spec.root_name.clone(),
            name: spec.root_name.clone(),
            surface: root_surface,
            children: Vec::new(),
        }];
        let mut frontier = vec![0usize];
        for (depth, level) in spec.levels.iter().enumerate() {
            let mut next = Vec::new();
            let mut counter = 0usize;
            for parent in frontier {
                for _ in 0..level.branching {
                    let id = nodes.len();
                    nodes.push(Node {
                        id,
                        parent: Some(parent),
                        depth: depth + 1,
                        level: level.name.clone(),
                        name: format!("{}-{}", level.name, counter),
                        surface: BehavioralSurface::neutral(),
                        children: Vec::new(),
                    });
                    nodes[parent].children.push(id);
                    next.push(id);
                    counter += 1;
                }
            }
            frontier = next;
        }
        // Children were appended level by level, which is BFS order rather
        // than preorder; renumber into preorder so ids are stable and
        // parameter blocks follow the tree.
        Self::from_nodes(nodes)
    }

    /// A single-node hierarchy.
    pub fn single(surface: BehavioralSurface) -> Self {
        Self::from_spec(&HierarchySpec::flat(), surface)
    }

    fn from_nodes(bfs: Vec<Node>) -> Self {
        let n = bfs.len();
        let mut order = Vec::with_capacity(n);
        let mut stack = vec![0usize];
        while let Some(id) = stack.pop() {
            order.push(id);
            for &c in bfs[id].children.iter().rev() {
                stack.push(c);
            }
        }
        let mut remap = vec![0usize; n];
        for (new_id, &old_id) in order.iter().enumerate() {
            remap[old_id] = new_id;
        }
        let mut nodes: Vec<Node> = Vec::with_capacity(n);
        for &old_id in &order {
            let old = &bfs[old_id];
            nodes.push(Node {
                id: remap[old_id],
                parent: old.parent.map(|p| remap[p]),
                depth: old.depth,
                level: old.level.clone(),
                name: old.name.clone(),
                surface: old.surface.clone(),
                children: old.children.iter().map(|&c| remap[c]).collect(),
            });
        }
        let leaves = nodes.iter().filter(|n| n.is_leaf()).map(|n| n.id).collect();
        let paths = (0..n)
            .map(|id| {
                let mut path = Vec::new();
                let mut cur = Some(id);
                while let Some(c) = cur {
                    path.push(c);
                    cur = nodes[c].parent;
                }
                path.reverse();
                path
            })
            .collect();
        Hierarchy {
            nodes,
            leaves,
            paths,
        }
    }

    /// All nodes in preorder.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Always false: a hierarchy has at least a root.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The root node.
    pub fn root(&self) -> &Node {
        &self.nodes[0]
    }

    /// A node by id. Panics on an invalid id.
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id]
    }

    /// A mutable node by id. Panics on an invalid id.
    pub fn node_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id]
    }

    /// Find a node by name.
    pub fn find(&self, name: &str) -> Option<NodeId> {
        self.nodes.iter().find(|n| n.name == name).map(|n| n.id)
    }

    /// Leaf node ids, in preorder.
    pub fn leaves(&self) -> &[NodeId] {
        &self.leaves
    }

    /// Index of a leaf within [`leaves`](Self::leaves).
    pub fn leaf_index(&self, id: NodeId) -> Option<usize> {
        self.leaves.iter().position(|&l| l == id)
    }

    /// Root-to-node path.
    pub fn path(&self, id: NodeId) -> &[NodeId] {
        &self.paths[id]
    }

    /// Ids of all nodes at a given depth.
    pub fn nodes_at_depth(&self, depth: usize) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|n| n.depth == depth)
            .map(|n| n.id)
            .collect()
    }

    /// Maximum depth in the tree.
    pub fn max_depth(&self) -> usize {
        self.nodes.iter().map(|n| n.depth).max().unwrap_or(0)
    }

    /// Ids of a node and all its descendants, in preorder.
    pub fn subtree(&self, id: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut stack = vec![id];
        while let Some(n) = stack.pop() {
            out.push(n);
            for &c in self.nodes[n].children.iter().rev() {
                stack.push(c);
            }
        }
        out
    }

    /// Compose the surfaces along the path to `id`.
    pub fn effective(&self, id: NodeId) -> EffectivePolicy {
        let mut weights = [0.0; FEATURES];
        let mut fraction: Option<f64> = None;
        for &n in &self.paths[id] {
            let s = &self.nodes[n].surface;
            for (w, x) in weights.iter_mut().zip(&s.weights) {
                *w += x;
            }
            fraction = Some(s.entropy.effective(fraction));
        }
        EffectivePolicy {
            weights,
            entropy_fraction: fraction.unwrap_or(0.0),
        }
    }

    /// Effective policy for every leaf, indexed like [`leaves`](Self::leaves).
    pub fn compile(&self) -> Vec<EffectivePolicy> {
        self.leaves.iter().map(|&l| self.effective(l)).collect()
    }

    /// Length of the flat parameter vector.
    pub fn param_len(&self) -> usize {
        self.nodes.len() * PARAM_LEN
    }

    /// Where a node's parameters live inside the flat vector.
    pub fn param_range(&self, id: NodeId) -> Range<usize> {
        id * PARAM_LEN..(id + 1) * PARAM_LEN
    }

    /// The flat parameter vector of the whole tree.
    pub fn params(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.param_len());
        for n in &self.nodes {
            v.extend(n.surface.to_params());
        }
        v
    }

    /// Overwrite every node from a flat parameter vector.
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), SurfaceError> {
        if params.len() != self.param_len() {
            return Err(SurfaceError {
                expected: self.param_len(),
                got: params.len(),
            });
        }
        for (n, chunk) in self.nodes.iter_mut().zip(params.chunks(PARAM_LEN)) {
            n.surface.set_params(chunk)?;
        }
        Ok(())
    }

    /// Multi-line description of the tree with each node's entropy setting.
    pub fn describe(&self) -> String {
        let mut out = String::new();
        for n in &self.nodes {
            let eff = self.effective(n.id);
            let _ = writeln!(
                out,
                "{}{} [{}] entropy: {} → effective {:.3}, |w| = {:.2}",
                "  ".repeat(n.depth),
                n.name,
                n.level,
                n.surface.entropy.describe(),
                eff.entropy_fraction,
                eff.weights.iter().map(|w| w * w).sum::<f64>().sqrt()
            );
        }
        out
    }
}

impl Default for Hierarchy {
    fn default() -> Self {
        Hierarchy::from_spec(&HierarchySpec::default(), BehavioralSurface::instinct())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::EntropyControl;

    #[test]
    fn default_shape() {
        let h = Hierarchy::default();
        assert_eq!(h.len(), 10);
        assert_eq!(h.leaves().len(), 6);
        assert_eq!(h.root().children.len(), 3);
        assert_eq!(h.max_depth(), 2);
        assert_eq!(HierarchySpec::default().node_count(), 10);
        // Preorder: root, caste-0, its two squads, caste-1, ...
        assert_eq!(h.node(1).name, "caste-0");
        assert_eq!(h.node(2).level, "squad");
        assert_eq!(h.node(2).parent, Some(1));
        assert_eq!(h.node(4).name, "caste-1");
        assert_eq!(h.path(3), &[0, 1, 3]);
        assert_eq!(h.subtree(1), vec![1, 2, 3]);
        assert_eq!(h.leaf_index(3), Some(1));
        assert_eq!(h.find("squad-5"), Some(9));
        for n in h.nodes() {
            for &c in &n.children {
                assert_eq!(h.node(c).parent, Some(n.id));
            }
        }
    }

    #[test]
    fn composition_along_path() {
        let mut h = Hierarchy::default();
        let root_eff = h.effective(0);
        assert!((root_eff.entropy_fraction - 0.35).abs() < 1e-9);
        h.node_mut(1).surface.weights[0] = 1.5;
        h.node_mut(1).surface.entropy = EntropyControl::relative(2.0);
        h.node_mut(2).surface.weights[0] = -0.5;
        let leaf = h.effective(2);
        let expected_w0 = BehavioralSurface::instinct().weights[0] + 1.5 - 0.5;
        assert!((leaf.weights[0] - expected_w0).abs() < 1e-12);
        assert!((leaf.entropy_fraction - 0.7).abs() < 1e-9);
        let other = h.effective(5);
        assert!((other.entropy_fraction - 0.35).abs() < 1e-9);
        let compiled = h.compile();
        assert_eq!(compiled.len(), 6);
        assert_eq!(compiled[0], leaf);
    }

    #[test]
    fn flat_params_round_trip() {
        let mut h = Hierarchy::default();
        let p = h.params();
        assert_eq!(p.len(), 10 * PARAM_LEN);
        let mut q = p.clone();
        let range = h.param_range(4);
        q[range.start] = 42.0;
        h.set_params(&q).unwrap();
        assert_eq!(h.node(4).surface.weights[0], 42.0);
        assert_eq!(h.params(), q);
        assert!(h.set_params(&q[1..]).is_err());
    }

    #[test]
    fn single_and_describe() {
        let h = Hierarchy::single(BehavioralSurface::instinct());
        assert_eq!(h.len(), 1);
        assert_eq!(h.leaves(), &[0]);
        assert!(h.describe().contains("colony"));
    }
}
