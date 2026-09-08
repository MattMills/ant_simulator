//! The behavioural memo: what the ants do where, as compositional
//! signatures over the quadtree, at two time constants.
//!
//! A [`Signature`] is the record of the operative behaviour in a region:
//! how many decisions were made there and how disordered they were, which
//! turns were taken, how far ants walked, in which modes and laden or
//! not, what they laid, and what happened to them (food found, nest
//! reached, search given up, death). Every field is a sum, so a node's
//! signature is the sum of its children's and the whole tree composes
//! from its leaves. A [`Memo`] keeps two trees of them with lazy
//! forgetting, a slow one whose signatures are the invariant character of
//! the landscape and a fast one whose departure from it is the variant,
//! so the memo of a region is a characterisation of the behavioural
//! surface over it that can be extracted as an object of its own
//! ([`Memo::extract`]), classified into categories ([`Memo::classify`]),
//! rendered as a map, and read as features to learn against
//! ([`Memo::features`]).

use crate::landscape::{turn_magnitude, RING, RING_STEP};
use crate::pheromone::Pheromone;
use crate::quad::{QuadKey, QuadTree};
use crate::rng::Rng;
use std::fmt::Write as _;

/// What ended an ant's movement in a region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// Food was found there.
    Food,
    /// The nest was reached there.
    Nest,
    /// A search was given up there.
    GaveUp,
    /// The ant died there.
    Death,
}

impl Stop {
    /// Index in the signature's counts.
    pub fn index(self) -> usize {
        match self {
            Stop::Food => 0,
            Stop::Nest => 1,
            Stop::GaveUp => 2,
            Stop::Death => 3,
        }
    }
}

/// Mode of a movement decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Leg {
    /// Away from the nest.
    Outbound,
    /// Towards the nest.
    Inbound,
    /// A systematic search.
    Searching,
}

impl Leg {
    fn index(self) -> usize {
        match self {
            Leg::Outbound => 0,
            Leg::Inbound => 1,
            Leg::Searching => 2,
        }
    }
}

/// The compositional record of behaviour in a region (weighted counts,
/// so that they can be forgotten and composed).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Signature {
    /// Movement decisions.
    pub decisions: f64,
    /// Summed decision entropy, nats.
    pub entropy: f64,
    /// Summed cosine of the turns taken.
    pub straight: f64,
    /// Ring positions chosen (the turn histogram).
    pub turns: [f64; RING],
    /// Decisions by leg: outbound, inbound, searching.
    pub legs: [f64; 3],
    /// Decisions made while carrying something.
    pub laden: f64,
    /// Moves made.
    pub moves: f64,
    /// Cells walked.
    pub length: f64,
    /// Pheromone laid, per channel.
    pub deposits: [f64; Pheromone::COUNT],
    /// Stops, by kind.
    pub stops: [f64; 4],
}

/// Numbers in a signature's feature vector.
pub const FEATURES: usize = 10;

/// Names of the features, in order.
pub const FEATURE_NAMES: [&str; FEATURES] = [
    "decisions",
    "entropy",
    "straightness",
    "turn entropy",
    "outbound",
    "homing",
    "searching",
    "laden",
    "marking",
    "stopping",
];

impl Signature {
    /// Add another signature (composition).
    pub fn add(&mut self, other: &Signature) {
        self.decisions += other.decisions;
        self.entropy += other.entropy;
        self.straight += other.straight;
        for (a, b) in self.turns.iter_mut().zip(&other.turns) {
            *a += b;
        }
        for (a, b) in self.legs.iter_mut().zip(&other.legs) {
            *a += b;
        }
        self.laden += other.laden;
        self.moves += other.moves;
        self.length += other.length;
        for (a, b) in self.deposits.iter_mut().zip(&other.deposits) {
            *a += b;
        }
        for (a, b) in self.stops.iter_mut().zip(&other.stops) {
            *a += b;
        }
    }

    /// Subtract another signature (the variant part of a current record
    /// over an invariant one).
    pub fn minus(&self, other: &Signature) -> Signature {
        let mut out = self.clone();
        out.decisions -= other.decisions;
        out.entropy -= other.entropy;
        out.straight -= other.straight;
        for (a, b) in out.turns.iter_mut().zip(&other.turns) {
            *a -= b;
        }
        for (a, b) in out.legs.iter_mut().zip(&other.legs) {
            *a -= b;
        }
        out.laden -= other.laden;
        out.moves -= other.moves;
        out.length -= other.length;
        for (a, b) in out.deposits.iter_mut().zip(&other.deposits) {
            *a -= b;
        }
        for (a, b) in out.stops.iter_mut().zip(&other.stops) {
            *a -= b;
        }
        out
    }

    fn scale_in_place(&mut self, k: f64) {
        self.decisions *= k;
        self.entropy *= k;
        self.straight *= k;
        self.turns.iter_mut().for_each(|t| *t *= k);
        self.legs.iter_mut().for_each(|t| *t *= k);
        self.laden *= k;
        self.moves *= k;
        self.length *= k;
        self.deposits.iter_mut().for_each(|t| *t *= k);
        self.stops.iter_mut().for_each(|t| *t *= k);
    }

    /// Record one movement decision.
    pub fn decide(&mut self, ring: usize, entropy: f64, leg: Leg, laden: bool, w: f64) {
        self.decisions += w;
        self.entropy += entropy * w;
        self.straight += (turn_magnitude(ring) as f64 * RING_STEP).cos() * w;
        self.turns[ring % RING] += w;
        self.legs[leg.index()] += w;
        if laden {
            self.laden += w;
        }
    }

    /// Mean decision entropy, nats.
    pub fn mean_entropy(&self) -> f64 {
        if self.decisions > 0.0 {
            self.entropy / self.decisions
        } else {
            0.0
        }
    }

    /// Mean cosine of the turns: 1 for straight travel, 0 for random
    /// turning.
    pub fn straightness(&self) -> f64 {
        if self.decisions > 0.0 {
            self.straight / self.decisions
        } else {
            0.0
        }
    }

    /// Entropy of the turn histogram, nats.
    pub fn turn_entropy(&self) -> f64 {
        let total: f64 = self.turns.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }
        -self
            .turns
            .iter()
            .filter(|&&t| t > 0.0)
            .map(|&t| {
                let p = t / total;
                p * p.ln()
            })
            .sum::<f64>()
    }

    /// Share of decisions in a leg.
    pub fn leg_share(&self, leg: Leg) -> f64 {
        if self.decisions > 0.0 {
            self.legs[leg.index()] / self.decisions
        } else {
            0.0
        }
    }

    /// The feature vector: decisions (density), mean entropy,
    /// straightness, turn entropy, the three leg shares, the laden share,
    /// trail laid per decision, stops per decision.
    pub fn features(&self) -> [f64; FEATURES] {
        let d = self.decisions.max(1e-12);
        [
            self.decisions,
            self.mean_entropy(),
            self.straightness(),
            self.turn_entropy(),
            self.leg_share(Leg::Outbound),
            self.leg_share(Leg::Inbound),
            self.leg_share(Leg::Searching),
            self.laden / d,
            self.deposits[Pheromone::Trail.index()] / d,
            self.stops.iter().sum::<f64>() / d,
        ]
    }
}

/// Which record of the memo a reading uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    /// The slow record: the invariant character of the region.
    Invariant,
    /// The fast record: the behaviour of late.
    Current,
    /// The current record minus the invariant one: the variant part.
    Variant,
}

/// How the memo is kept.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoConfig {
    /// Half-life of the slow, invariant record, seconds.
    pub slow_half_life_s: f64,
    /// Half-life of the fast record, seconds.
    pub fast_half_life_s: f64,
    /// Cells per side of the nodes the memo is read and classified at.
    pub grain: usize,
    /// Memoize transits through the nodes (see [`Transits`]); `None`
    /// keeps the memo without standing in for the simulation.
    pub transits: Option<TransitConfig>,
}

impl Default for MemoConfig {
    fn default() -> Self {
        MemoConfig {
            slow_half_life_s: 3600.0,
            fast_half_life_s: 60.0,
            grain: 4,
            transits: None,
        }
    }
}

/// The colony's behavioural memo over the quadtree.
#[derive(Clone, Debug)]
pub struct Memo {
    slow: QuadTree<Signature>,
    fast: QuadTree<Signature>,
    slow_r: f64,
    fast_r: f64,
    slow_g: f64,
    fast_g: f64,
    level: u8,
    composed: bool,
    cfg: MemoConfig,
    /// Elapsed ticks, for unbiased rate readings before the records have
    /// filled.
    ticks: u64,
    /// The transit kernels, when transits are memoized.
    pub transits: Option<Transits>,
}

impl Memo {
    /// An empty memo for a world of the given size and tick.
    pub fn new(width: usize, height: usize, tick_s: f64, cfg: MemoConfig) -> Memo {
        let retention = |half_life: f64| -> f64 {
            if half_life <= 0.0 {
                0.0
            } else {
                0.5f64.powf(tick_s / half_life)
            }
        };
        let slow: QuadTree<Signature> = QuadTree::new(width, height);
        let level = slow.level_for_side(cfg.grain.max(1));
        Memo {
            fast: slow.clone(),
            slow,
            slow_r: retention(cfg.slow_half_life_s),
            fast_r: retention(cfg.fast_half_life_s),
            slow_g: 1.0,
            fast_g: 1.0,
            level,
            composed: true,
            cfg,
            ticks: 0,
            transits: None,
        }
    }

    /// Attach transit kernels over the memo's level, with the plain nodes
    /// taken from the world.
    pub fn with_transits(mut self, world: &crate::world::World) -> Memo {
        if let Some(t) = self.cfg.transits.clone() {
            self.transits = Some(Transits::new(&self.slow, self.level, world, t));
        }
        self
    }

    /// The level the memo is read at.
    pub fn level(&self) -> u8 {
        self.level
    }

    /// The configuration.
    pub fn config(&self) -> &MemoConfig {
        &self.cfg
    }

    /// The nodes of a level that meet the grid, in Z-order.
    pub fn keys(&self, level: u8) -> Vec<QuadKey> {
        self.slow.keys(level)
    }

    /// The node at a level holding a cell.
    pub fn key_of(&self, cell: crate::geometry::Position, level: u8) -> Option<QuadKey> {
        self.slow.key_of(cell, level)
    }

    /// The cells of a node, `(x0, y0, x1, y1)` with exclusive far corners.
    pub fn rect(&self, key: QuadKey) -> (usize, usize, usize, usize) {
        self.slow.rect(key)
    }

    /// Columns and rows of nodes at the memo's level.
    pub fn extent(&self) -> (usize, usize) {
        self.slow.extent(self.level)
    }

    /// Change the leaf holding a cell in both records (parents go stale
    /// until the next composition).
    fn touch(&mut self, cell: crate::geometry::Position, f: impl Fn(&mut Signature, f64)) {
        let Some(leaf) = self.slow.leaf(cell) else {
            return;
        };
        if self.slow_r > 0.0 {
            f(self.slow.get_mut(leaf), 1.0 / self.slow_g);
        }
        if self.fast_r > 0.0 {
            f(self.fast.get_mut(leaf), 1.0 / self.fast_g);
        }
        self.composed = false;
    }

    /// Record a movement decision made in a cell.
    pub fn record_decision(
        &mut self,
        cell: crate::geometry::Position,
        ring: usize,
        entropy: f64,
        leg: Leg,
        laden: bool,
    ) {
        self.touch(cell, |s, w| s.decide(ring, entropy, leg, laden, w));
    }

    /// Record decisions stood in for by a replayed transit, at a cell.
    pub fn record_replay(
        &mut self,
        cell: crate::geometry::Position,
        decisions: f64,
        entropy: f64,
        straight: f64,
        leg: Leg,
        laden: bool,
    ) {
        if decisions <= 0.0 {
            return;
        }
        self.touch(cell, |s, w| {
            s.decisions += decisions * w;
            s.entropy += entropy * w;
            s.straight += straight * w;
            s.turns[0] += decisions * w;
            s.legs[leg.index()] += decisions * w;
            if laden {
                s.laden += decisions * w;
            }
        });
    }

    /// Record a move ending in a cell.
    pub fn record_move(&mut self, cell: crate::geometry::Position, length: f64) {
        self.touch(cell, |s, w| {
            s.moves += w;
            s.length += length * w;
        });
    }

    /// Record pheromone laid in a cell.
    pub fn record_deposit(
        &mut self,
        cell: crate::geometry::Position,
        kind: Pheromone,
        amount: f64,
    ) {
        if amount <= 0.0 {
            return;
        }
        self.touch(cell, |s, w| s.deposits[kind.index()] += amount * w);
    }

    /// Record a stop in a cell.
    pub fn record_stop(&mut self, cell: crate::geometry::Position, stop: Stop) {
        self.touch(cell, |s, w| s.stops[stop.index()] += w);
    }

    /// Let one tick pass: both records forget lazily.
    pub fn step(&mut self) {
        self.ticks += 1;
        self.slow_g *= self.slow_r;
        self.fast_g *= self.fast_r;
        if self.slow_g < 1e-150 || self.fast_g < 1e-150 {
            let (sg, fg) = (self.slow_g, self.fast_g);
            for s in self.slow.values_mut() {
                s.scale_in_place(sg);
            }
            for s in self.fast.values_mut() {
                s.scale_in_place(fg);
            }
            self.slow_g = 1.0;
            self.fast_g = 1.0;
        }
    }

    /// Rebuild every node above the leaves from its children.
    pub fn compose(&mut self) {
        if self.composed {
            return;
        }
        self.slow.compose(|acc, child| acc.add(child));
        self.fast.compose(|acc, child| acc.add(child));
        self.composed = true;
    }

    /// Whether the nodes above the leaves are up to date.
    pub fn is_composed(&self) -> bool {
        self.composed
    }

    /// A composed copy: the memo as an object of its own.
    pub fn extract(&self) -> Memo {
        let mut m = self.clone();
        m.compose();
        m
    }

    /// Normalisation that turns a record into a rate per tick after
    /// `ticks` ticks of accumulation (a constant rate `λ` added before
    /// each tick's forgetting stands at `λ r (1 − rᵗ) / (1 − r)`).
    fn norm(r: f64, ticks: u64) -> f64 {
        if r <= 0.0 || ticks == 0 {
            0.0
        } else {
            let filled = 1.0 - r.powi(ticks.min(1 << 30) as i32);
            (1.0 - r) / (r * filled.max(1e-300))
        }
    }

    /// The signature of a node as rates per tick (the memo must be
    /// composed for nodes above the leaves).
    pub fn signature(&self, key: QuadKey, layer: Layer) -> Signature {
        let ticks = self.ticks;
        let read = |tree: &QuadTree<Signature>, g: f64, r: f64| {
            let mut s = tree.get(key).clone();
            s.scale_in_place(g * Self::norm(r, ticks));
            s
        };
        match layer {
            Layer::Invariant => read(&self.slow, self.slow_g, self.slow_r),
            Layer::Current => read(&self.fast, self.fast_g, self.fast_r),
            Layer::Variant => read(&self.fast, self.fast_g, self.fast_r).minus(&read(
                &self.slow,
                self.slow_g,
                self.slow_r,
            )),
        }
    }

    /// Feature vectors of the nodes at a level (nodes with no decisions
    /// read as zeros).
    pub fn features(&self, level: u8, layer: Layer) -> Vec<f64> {
        self.keys(level)
            .into_iter()
            .flat_map(|k| self.signature(k, layer).features())
            .collect()
    }

    /// Classify the nodes of a level into `k` categories by their
    /// invariant signatures (k-means on standardised features, nodes
    /// without decisions left out).
    pub fn classify(&self, level: u8, k: usize, seed: u64) -> Classification {
        let keys = self.keys(level);
        let rows: Vec<(QuadKey, [f64; FEATURES])> = keys
            .iter()
            .map(|&key| (key, self.signature(key, Layer::Invariant)))
            .filter(|(_, s)| s.decisions > 0.0)
            .map(|(key, s)| (key, s.features()))
            .collect();
        Classification::fit(level, rows, k, seed)
    }

    /// The memo's level as text, one glyph per node: the category's
    /// digit, or a space where nothing happened.
    pub fn render(&self, classes: &Classification) -> String {
        let (cols, rows) = self.slow.extent(classes.level);
        let mut out = String::with_capacity((cols + 1) * rows);
        for r in 0..rows {
            for c in 0..cols {
                let key = QuadKey::new(classes.level, c as u32, r as u32);
                out.push(match classes.label(key) {
                    Some(l) => char::from(b'0' + (l % 10) as u8),
                    None => ' ',
                });
            }
            out.push('\n');
        }
        out
    }
}

/// A categorisation of the nodes of a level by their signatures.
#[derive(Clone, Debug, PartialEq)]
pub struct Classification {
    /// The level classified.
    pub level: u8,
    /// Category of each node that had decisions.
    pub labels: Vec<(QuadKey, usize)>,
    /// Category centres in feature space (unstandardised).
    pub centroids: Vec<[f64; FEATURES]>,
    /// Nodes per category.
    pub sizes: Vec<usize>,
    /// A name for each category from what marks it out.
    pub names: Vec<String>,
}

impl Classification {
    /// k-means with k-means++ seeding on standardised features.
    fn fit(
        level: u8,
        rows: Vec<(QuadKey, [f64; FEATURES])>,
        k: usize,
        seed: u64,
    ) -> Classification {
        let n = rows.len();
        let k = k.max(1).min(n.max(1));
        if n == 0 {
            return Classification {
                level,
                labels: Vec::new(),
                centroids: Vec::new(),
                sizes: Vec::new(),
                names: Vec::new(),
            };
        }
        // Standardise.
        let mut mean = [0.0; FEATURES];
        let mut sd = [0.0; FEATURES];
        for (_, f) in &rows {
            for j in 0..FEATURES {
                mean[j] += f[j] / n as f64;
            }
        }
        for (_, f) in &rows {
            for j in 0..FEATURES {
                sd[j] += (f[j] - mean[j]).powi(2) / n as f64;
            }
        }
        for s in sd.iter_mut() {
            *s = if *s > 1e-18 { s.sqrt() } else { 1.0 };
        }
        let z: Vec<[f64; FEATURES]> = rows
            .iter()
            .map(|(_, f)| {
                let mut out = [0.0; FEATURES];
                for j in 0..FEATURES {
                    out[j] = (f[j] - mean[j]) / sd[j];
                }
                out
            })
            .collect();
        let dist2 = |a: &[f64; FEATURES], b: &[f64; FEATURES]| {
            a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>()
        };
        // k-means++ seeding.
        let mut rng = Rng::seed_from_u64(seed);
        let mut centres: Vec<[f64; FEATURES]> = vec![z[rng.below(n)]];
        while centres.len() < k {
            let weights: Vec<f64> = z
                .iter()
                .map(|p| {
                    centres
                        .iter()
                        .map(|c| dist2(p, c))
                        .fold(f64::INFINITY, f64::min)
                })
                .collect();
            let total: f64 = weights.iter().sum();
            let next = if total <= 0.0 {
                rng.below(n)
            } else {
                rng.choose_weighted(&weights)
            };
            centres.push(z[next]);
        }
        let mut labels = vec![0usize; n];
        for _ in 0..50 {
            let mut changed = false;
            for (i, p) in z.iter().enumerate() {
                let best = (0..k)
                    .min_by(|&a, &b| {
                        dist2(p, &centres[a])
                            .partial_cmp(&dist2(p, &centres[b]))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0);
                if labels[i] != best {
                    labels[i] = best;
                    changed = true;
                }
            }
            let mut sums = vec![[0.0; FEATURES]; k];
            let mut counts = vec![0usize; k];
            for (i, p) in z.iter().enumerate() {
                counts[labels[i]] += 1;
                for j in 0..FEATURES {
                    sums[labels[i]][j] += p[j];
                }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..FEATURES {
                        centres[c][j] = sums[c][j] / counts[c] as f64;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let mut sizes = vec![0usize; k];
        for &l in &labels {
            sizes[l] += 1;
        }
        // Order categories by their density, busiest first.
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|&a, &b| {
            centres[b][0]
                .partial_cmp(&centres[a][0])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut rank = vec![0usize; k];
        for (r, &c) in order.iter().enumerate() {
            rank[c] = r;
        }
        let centroids: Vec<[f64; FEATURES]> = order
            .iter()
            .map(|&c| {
                let mut out = [0.0; FEATURES];
                for j in 0..FEATURES {
                    out[j] = centres[c][j] * sd[j] + mean[j];
                }
                out
            })
            .collect();
        let names = order.iter().map(|&c| Self::name(&centres[c])).collect();
        Classification {
            level,
            labels: rows
                .iter()
                .zip(&labels)
                .map(|((key, _), &l)| (*key, rank[l]))
                .collect(),
            centroids,
            sizes: order.iter().map(|&c| sizes[c]).collect(),
            names,
        }
    }

    /// Name a category by the two standardised features that mark it out
    /// most.
    fn name(z: &[f64; FEATURES]) -> String {
        let words = |j: usize, high: bool| -> &'static str {
            match (j, high) {
                (0, true) => "busy",
                (0, false) => "quiet",
                (1, true) => "undecided",
                (1, false) => "decided",
                (2, true) => "straight",
                (2, false) => "winding",
                (3, true) => "turning",
                (3, false) => "steady",
                (4, true) => "outbound",
                (4, false) => "rarely outbound",
                (5, true) => "homing",
                (5, false) => "rarely homing",
                (6, true) => "searching",
                (6, false) => "rarely searching",
                (7, true) => "laden",
                (7, false) => "unladen",
                (8, true) => "marking",
                (8, false) => "unmarked",
                (9, true) => "stopping",
                _ => "passing",
            }
        };
        let mut order: Vec<usize> = (0..FEATURES).collect();
        order.sort_by(|&a, &b| {
            z[b].abs()
                .partial_cmp(&z[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let picks: Vec<&str> = order
            .iter()
            .take(2)
            .filter(|&&j| z[j].abs() > 0.3)
            .map(|&j| words(j, z[j] > 0.0))
            .collect();
        if picks.is_empty() {
            "ordinary".to_string()
        } else {
            picks.join(", ")
        }
    }

    /// The category of a node, if it was classified.
    pub fn label(&self, key: QuadKey) -> Option<usize> {
        self.labels.iter().find(|(k, _)| *k == key).map(|(_, l)| *l)
    }

    /// Number of categories.
    pub fn categories(&self) -> usize {
        self.centroids.len()
    }

    /// The categories as a table: size, name, and the centre's features.
    pub fn table(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "{:>3} {:>5}  {:<28} {}",
            "cat",
            "nodes",
            "character",
            FEATURE_NAMES
                .iter()
                .map(|n| format!("{n:>12}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        for (c, centre) in self.centroids.iter().enumerate() {
            let _ = writeln!(
                out,
                "{:>3} {:>5}  {:<28} {}",
                c,
                self.sizes[c],
                self.names[c],
                centre
                    .iter()
                    .map(|v| format!("{v:>12.3}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Position;

    #[test]
    fn signatures_compose_and_summarise() {
        let mut a = Signature::default();
        a.decide(0, 0.5, Leg::Outbound, false, 1.0);
        a.decide(0, 0.5, Leg::Outbound, true, 1.0);
        let mut b = Signature::default();
        b.decide(RING / 2, 2.0, Leg::Searching, false, 1.0);
        b.stops[Stop::Food.index()] += 1.0;
        let mut whole = a.clone();
        whole.add(&b);
        assert_eq!(whole.decisions, 3.0);
        assert!((whole.mean_entropy() - 1.0).abs() < 1e-12);
        assert!((whole.straightness() - (1.0 + 1.0 - 1.0) / 3.0).abs() < 1e-12);
        assert!((a.straightness() - 1.0).abs() < 1e-12);
        assert!(whole.turn_entropy() > 0.0 && a.turn_entropy() == 0.0);
        assert!((whole.leg_share(Leg::Searching) - 1.0 / 3.0).abs() < 1e-12);
        let f = whole.features();
        assert_eq!(f.len(), FEATURES);
        assert!((f[7] - 1.0 / 3.0).abs() < 1e-12 && (f[9] - 1.0 / 3.0).abs() < 1e-12);
        let back = whole.minus(&b);
        assert!((back.decisions - a.decisions).abs() < 1e-12);
    }

    #[test]
    fn memo_records_forgets_and_composes() {
        let mut m = Memo::new(
            16,
            16,
            1.0,
            MemoConfig {
                slow_half_life_s: 100.0,
                fast_half_life_s: 5.0,
                grain: 8,
                ..MemoConfig::default()
            },
        );
        assert_eq!(m.level(), 1);
        assert_eq!(m.extent(), (2, 2));
        for _ in 0..2000 {
            m.record_decision(Position::new(3, 2), 0, 0.4, Leg::Inbound, true);
            m.record_move(Position::new(3, 2), 1.0);
            m.record_deposit(Position::new(3, 2), Pheromone::Trail, 2.0);
            m.step();
        }
        m.record_stop(Position::new(12, 12), Stop::Nest);
        assert!(!m.is_composed());
        let x = m.extract();
        assert!(x.is_composed());
        let leaf = x.key_of(Position::new(3, 2), x.slow.levels()).unwrap();
        let s = x.signature(leaf, Layer::Invariant);
        assert!(
            (s.decisions - 1.0).abs() < 0.02,
            "steady decisions per tick: {}",
            s.decisions
        );
        assert!((s.mean_entropy() - 0.4).abs() < 1e-9);
        assert!((s.deposits[Pheromone::Trail.index()] - 2.0).abs() < 0.05);
        let sector = x.key_of(Position::new(3, 2), x.level()).unwrap();
        let cs = x.signature(sector, Layer::Invariant);
        assert!(
            (cs.decisions - s.decisions).abs() < 1e-9,
            "the node composes its cells"
        );
        let root = x.signature(QuadKey::ROOT, Layer::Current);
        assert!((root.decisions - 1.0).abs() < 0.02);
        let variant = x.signature(QuadKey::ROOT, Layer::Variant);
        assert!(variant.decisions.abs() < 0.05);
        let far = x.key_of(Position::new(12, 12), x.level()).unwrap();
        assert!(x.signature(far, Layer::Invariant).stops[Stop::Nest.index()] > 0.0);
        // Stop moving: the current record drops, the variant goes negative.
        let mut m2 = x.clone();
        for _ in 0..30 {
            m2.step();
        }
        assert!(m2.signature(QuadKey::ROOT, Layer::Variant).decisions < -0.5);
        assert_eq!(x.features(1, Layer::Invariant).len(), 4 * FEATURES);
    }

    #[test]
    fn classification_separates_kinds_of_ground() {
        let mut m = Memo::new(
            32,
            32,
            1.0,
            MemoConfig {
                grain: 8,
                ..MemoConfig::default()
            },
        );
        // A straight, busy, decided trail in the north; wandering search
        // in the south; nothing in between.
        for t in 0..400 {
            for x in 0..32 {
                m.record_decision(Position::new(x, 2), 0, 0.3, Leg::Inbound, true);
            }
            for x in (0..32).step_by(4) {
                m.record_decision(
                    Position::new(x, 28),
                    (t + x) as usize % RING,
                    2.4,
                    Leg::Searching,
                    false,
                );
            }
            m.step();
        }
        let x = m.extract();
        let classes = x.classify(x.level(), 2, 1);
        assert_eq!(classes.categories(), 2);
        assert_eq!(
            classes.labels.len(),
            8,
            "eight nodes had decisions: {classes:?}"
        );
        let north = x.key_of(Position::new(2, 2), x.level()).unwrap();
        let south = x.key_of(Position::new(2, 28), x.level()).unwrap();
        assert_eq!(
            classes.label(north),
            Some(0),
            "the busy trail is category 0"
        );
        assert_eq!(classes.label(south), Some(1));
        assert!(
            classes.names[0].contains("busy")
                || classes.names[0].contains("straight")
                || classes.names[0].contains("decided"),
            "{:?}",
            classes.names
        );
        let picture = x.render(&classes);
        assert_eq!(picture.lines().count(), 4);
        assert!(picture.contains('0') && picture.contains('1'));
        assert!(classes.table().contains("character"));
        let quiet = x.key_of(Position::new(2, 14), x.level()).unwrap();
        assert_eq!(classes.label(quiet), None);
    }
}

// ---------------------------------------------------------------
// Memoized transits
// ---------------------------------------------------------------

/// Sides of a node: north (entered from above), east, south, west; the
/// fifth value marks a transit that ended inside the node.
pub const INSIDE: u8 = 4;

/// Classes the entry heading is quantised into.
pub const HEADING_CLASSES: u8 = 8;

/// Waypoints an outcome keeps of the path through the node.
pub const VIA: usize = 6;

/// Levels at which transits can be memoized at once: the memo's grain
/// and the levels above it.
pub const MAX_TRANSIT_LEVELS: usize = 4;

/// Where and how a transit through a node began: the key of a transit
/// kernel. Two ants entering the same node from the same side with the
/// same heading, on the same leg, laden alike, under the same dial and
/// into the same local field are expected to fare alike, as two
/// identical neighbourhoods of a cellular automaton do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransitKey {
    /// The node.
    pub node: QuadKey,
    /// The side entered by.
    pub side: u8,
    /// The entry heading's class.
    pub heading: u8,
    /// The leg: 0 outbound, 1 inbound, 2 searching.
    pub leg: u8,
    /// Whether the ant carried something.
    pub laden: bool,
    /// Which distinct policy the ant acts through.
    pub policy: u8,
    /// The entropy dial's class for that leaf.
    pub dial: u8,
    /// The local field's class at entry: the trail's strength and the
    /// crowding.
    pub context: u8,
}

impl TransitKey {
    /// Class of a heading, `HEADING_CLASSES` slices of the circle.
    pub fn heading_class(heading: f64) -> u8 {
        let turns = heading.rem_euclid(std::f64::consts::TAU) / std::f64::consts::TAU;
        ((turns * HEADING_CLASSES as f64).floor() as u8).min(HEADING_CLASSES - 1)
    }

    /// Class of a local field: the trail concentration in perception
    /// constants (none, faint, moderate, strong) and whether the patch
    /// is crowded.
    pub fn context_class(trail_in_k: f64, crowded: bool) -> u8 {
        let trail = if trail_in_k <= 0.0 {
            0
        } else if trail_in_k < 1.0 {
            1
        } else if trail_in_k < 4.0 {
            2
        } else {
            3
        };
        trail | if crowded { 4 } else { 0 }
    }

    /// Class of a dial: a fixed temperature by its octave, an entropy
    /// target by quarters.
    pub fn dial_class(tempering: crate::entropy::Tempering) -> u8 {
        match tempering {
            crate::entropy::Tempering::Temperature(t) => {
                (t.max(1e-9).log2().round().clamp(-4.0, 4.0) + 4.0) as u8
            }
            crate::entropy::Tempering::Entropy(f) => 16 + (f.clamp(0.0, 1.0) * 4.0).floor() as u8,
        }
    }
}

/// What came of a transit.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TransitOutcome {
    /// The side left by, or [`INSIDE`] when the transit ended in the
    /// node (food found, nest reached, search begun or given up, death).
    pub exit_side: u8,
    /// Position along the exit side, 0 to 1 (west to east, north to
    /// south).
    pub exit_along: f32,
    /// Heading on leaving, radians.
    pub heading: f32,
    /// Ticks spent in the node.
    pub ticks: u16,
    /// Cells walked in the node.
    pub length: f32,
    /// Movement decisions made in the node.
    pub decisions: u16,
    /// Their summed entropy, nats.
    pub entropy: f32,
    /// Their summed turn cosines.
    pub straight: f32,
    /// Pheromone laid in the node, per channel.
    pub deposits: [f32; Pheromone::COUNT],
    /// Points passed on the way through the node, one every quarter of
    /// the node's side, so that what a replay lays follows the path
    /// rather than the chord.
    pub via: [(f32, f32); VIA],
    /// How many of them are set.
    pub via_len: u8,
}

impl TransitOutcome {
    /// The waypoints, as points.
    pub fn waypoints(&self) -> impl Iterator<Item = crate::geometry::Point> + '_ {
        self.via[..self.via_len as usize]
            .iter()
            .map(|&(x, y)| crate::geometry::Point::new(x as f64, y as f64))
    }
}

/// A transit being recorded: where it began and what has happened in it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransitRecord {
    /// The key it will be filed under.
    pub key: TransitKey,
    /// Tick of entry.
    pub start_tick: u64,
    /// Point of entry.
    pub entry: crate::geometry::Point,
    /// Cells walked so far.
    pub length: f32,
    /// Decisions so far.
    pub decisions: u16,
    /// Their summed entropy.
    pub entropy: f32,
    /// Their summed turn cosines.
    pub straight: f32,
    /// Pheromone laid so far.
    pub deposits: [f32; Pheromone::COUNT],
    /// Waypoints so far.
    pub via: [(f32, f32); VIA],
    /// How many of them are set.
    pub via_len: u8,
    /// Cells walked between waypoints.
    pub spacing: f32,
    /// Cells walked since the last waypoint.
    pub since_via: f32,
}

impl TransitRecord {
    /// A record just opened, keeping a waypoint every `spacing` cells.
    pub fn open(
        key: TransitKey,
        tick: u64,
        entry: crate::geometry::Point,
        spacing: f32,
    ) -> TransitRecord {
        TransitRecord {
            key,
            start_tick: tick,
            entry,
            length: 0.0,
            decisions: 0,
            entropy: 0.0,
            straight: 0.0,
            deposits: [0.0; Pheromone::COUNT],
            via: [(0.0, 0.0); VIA],
            via_len: 0,
            spacing: spacing.max(0.5),
            since_via: 0.0,
        }
    }

    /// A step of `step` cells ending at `at`.
    pub fn walked(&mut self, step: f64, at: crate::geometry::Point) {
        self.length += step as f32;
        self.since_via += step as f32;
        if self.since_via >= self.spacing && (self.via_len as usize) < VIA {
            self.via[self.via_len as usize] = (at.x as f32, at.y as f32);
            self.via_len += 1;
            self.since_via = 0.0;
        }
    }

    /// Take in a replayed outcome inside this record's node (a finer
    /// node crossed by kernel), ending at `at`.
    pub fn absorb(&mut self, o: &TransitOutcome, at: crate::geometry::Point) {
        self.decisions = self.decisions.saturating_add(o.decisions);
        self.entropy += o.entropy;
        self.straight += o.straight;
        for (d, a) in self.deposits.iter_mut().zip(&o.deposits) {
            *d += a;
        }
        self.walked(o.length as f64, at);
    }

    /// Close the record as an outcome.
    pub fn close(&self, tick: u64, exit_side: u8, exit_along: f32, heading: f64) -> TransitOutcome {
        TransitOutcome {
            exit_side,
            exit_along,
            heading: heading as f32,
            ticks: tick.saturating_sub(self.start_tick).min(u16::MAX as u64) as u16,
            length: self.length,
            decisions: self.decisions,
            entropy: self.entropy,
            straight: self.straight,
            deposits: self.deposits,
            via: self.via,
            via_len: self.via_len,
        }
    }
}

/// A transit being replayed: the ant is inside the node until `until`,
/// when it appears at `exit` with everything the outcome carries applied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transit {
    /// Tick at which the transit ends.
    pub until: u64,
    /// Point of entry.
    pub entry: crate::geometry::Point,
    /// Point of exit, just across the side left by.
    pub exit: crate::geometry::Point,
    /// The outcome replayed.
    pub outcome: TransitOutcome,
    /// The key replayed (for the ledger).
    pub key: TransitKey,
}

/// The outcomes seen for one key: a reservoir that keeps a bounded,
/// unbiased sample of them, and counts of what it saw.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Kernel {
    /// The retained outcomes.
    pub outcomes: Vec<TransitOutcome>,
    /// Transits seen.
    pub seen: u64,
    /// Of them, transits that ended inside the node.
    pub stopped: u64,
}

impl Kernel {
    fn push(&mut self, outcome: TransitOutcome, capacity: usize, rng: &mut Rng) {
        self.seen += 1;
        if outcome.exit_side == INSIDE {
            self.stopped += 1;
        }
        if self.outcomes.len() < capacity.max(1) {
            self.outcomes.push(outcome);
        } else {
            // A forgetting reservoir: each new outcome replaces one
            // retained at random, so that what is replayed follows the
            // field as it changes rather than the field as it was.
            let j = rng.below(self.outcomes.len());
            self.outcomes[j] = outcome;
        }
    }

    /// Whether the kernel can stand in for the simulation: enough
    /// outcomes, and next to none of them ended inside the node.
    pub fn mature(&self, min_samples: usize, max_stop_share: f64) -> bool {
        self.outcomes.len() >= min_samples.max(1)
            && self.seen > 0
            && (self.stopped as f64) <= max_stop_share * self.seen as f64
    }

    /// How definite the kernel's transition is: the share of its
    /// retained outcomes that left by the commonest side, discounted by
    /// the spread of where along that side they left (1 when every
    /// outcome leaves at the same point, 0 when the sides are split or
    /// the exits are spread over the whole side).
    pub fn coherence(&self) -> f64 {
        let leaving: Vec<&TransitOutcome> = self
            .outcomes
            .iter()
            .filter(|o| o.exit_side != INSIDE)
            .collect();
        if leaving.is_empty() {
            return 0.0;
        }
        let mut counts = [0usize; 4];
        for o in &leaving {
            counts[(o.exit_side as usize).min(3)] += 1;
        }
        let side = (0..4).max_by_key(|&s| counts[s]).unwrap_or(0);
        let share = counts[side] as f64 / leaving.len() as f64;
        let along: Vec<f64> = leaving
            .iter()
            .filter(|o| o.exit_side as usize == side)
            .map(|o| o.exit_along as f64)
            .collect();
        let n = along.len() as f64;
        let mean = along.iter().sum::<f64>() / n;
        let var = along.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / n;
        (share * (1.0 - 2.0 * var.sqrt())).clamp(0.0, 1.0)
    }

    /// One retained outcome that left the node, at random.
    pub fn sample(&self, rng: &mut Rng) -> Option<TransitOutcome> {
        let leaving: Vec<&TransitOutcome> = self
            .outcomes
            .iter()
            .filter(|o| o.exit_side != INSIDE)
            .collect();
        if leaving.is_empty() {
            None
        } else {
            Some(*leaving[rng.below(leaving.len())])
        }
    }
}

/// How transits are memoized.
#[derive(Clone, Debug, PartialEq)]
pub struct TransitConfig {
    /// Outcomes a key must have seen before it stands in for the
    /// simulation.
    pub min_samples: usize,
    /// Outcomes retained per key.
    pub capacity: usize,
    /// Share of eligible entries still simulated in full, so that the
    /// kernels keep learning and a change shows.
    pub explore: f64,
    /// Largest relative departure of a node's current flow from its
    /// invariant one at which its kernels are trusted.
    pub invariance: f64,
    /// Largest share of a key's transits that may have ended inside the
    /// node.
    pub max_stop_share: f64,
    /// Levels above the memo's grain at which transits are memoized
    /// too: an ant entering a coarser node whose kernel is mature and
    /// whose flow is invariant is advanced across the whole of it.
    pub depth: usize,
    /// Coherence ([`Kernel::coherence`]) a kernel at a level above the
    /// grain must have before it stands in for the simulation: the
    /// coarser the node, the more definite its transition has to be.
    pub coarse_coherence: f64,
}

impl Default for TransitConfig {
    fn default() -> Self {
        TransitConfig {
            min_samples: 12,
            capacity: 16,
            explore: 0.1,
            invariance: 0.5,
            max_stop_share: 0.02,
            depth: 2,
            coarse_coherence: 0.7,
        }
    }
}

/// The transit kernels of the memo's nodes at every level memoized, and
/// the ledger of what they stood in for.
#[derive(Clone, Debug)]
pub struct Transits {
    /// The levels, finest first.
    levels: Vec<u8>,
    kernels: std::collections::HashMap<TransitKey, Kernel>,
    /// Per level and node, whether it is plain ground: no food, nest,
    /// prey, wall or landmark, nothing that changes an ant's state.
    plain: Vec<Vec<bool>>,
    cfg: TransitConfig,
    /// Transits recorded in full.
    pub recorded: u64,
    /// Transits replayed from a kernel.
    pub replayed: u64,
    /// Ticks the replayed transits stood in for.
    pub ticks_replayed: u64,
    /// Decisions the replayed transits stood in for.
    pub decisions_replayed: u64,
    /// Transits replayed per level, finest first.
    pub replayed_by_level: Vec<u64>,
    /// Decisions stood in for per level, finest first.
    pub decisions_by_level: Vec<u64>,
}

impl Transits {
    /// Kernels over the nodes of a level and the `depth` levels above
    /// it, with the plain nodes taken from the world.
    pub fn new(
        tree: &QuadTree<Signature>,
        level: u8,
        world: &crate::world::World,
        cfg: TransitConfig,
    ) -> Transits {
        let levels: Vec<u8> = (0..=cfg.depth)
            .filter_map(|d| {
                let l = level as i32 - d as i32;
                (l >= 1).then_some(l as u8)
            })
            .take(MAX_TRANSIT_LEVELS)
            .collect();
        let mut plain: Vec<Vec<bool>> = levels
            .iter()
            .map(|&l| vec![false; 1usize << (2 * l as usize)])
            .collect();
        for key in tree.keys(level) {
            let (x0, y0, x1, y1) = tree.rect(key);
            let mut ok = true;
            'cells: for y in y0..y1 {
                for x in x0..x1 {
                    let p = crate::geometry::Position::new(x as i32, y as i32);
                    let Some(c) = world.cell(p) else {
                        ok = false;
                        break 'cells;
                    };
                    if c.terrain != crate::world::Terrain::Open
                        || c.food_capacity_ul > 0.0
                        || c.has_food()
                        || c.renewal_ul_per_s > 0.0
                        || world.has_portal(p)
                    {
                        ok = false;
                        break 'cells;
                    }
                }
            }
            if ok
                && world.landmarks().iter().any(|l| {
                    l.x >= x0 as i32
                        && (l.x as usize) < x1
                        && l.y >= y0 as i32
                        && (l.y as usize) < y1
                })
            {
                ok = false;
            }
            plain[0][key.code as usize] = ok;
        }
        // A coarser node is plain when every child on the grid is.
        for slot in 1..levels.len() {
            let l = levels[slot];
            for key in tree.keys(l) {
                let all = key
                    .children()
                    .iter()
                    .filter(|c| tree.on_grid(**c))
                    .all(|c| plain[slot - 1][c.code as usize]);
                plain[slot][key.code as usize] = all;
            }
        }
        let n = levels.len();
        Transits {
            levels,
            kernels: std::collections::HashMap::new(),
            plain,
            cfg,
            recorded: 0,
            replayed: 0,
            ticks_replayed: 0,
            decisions_replayed: 0,
            replayed_by_level: vec![0; n],
            decisions_by_level: vec![0; n],
        }
    }

    /// The finest level memoized (the memo's grain).
    pub fn level(&self) -> u8 {
        self.levels[0]
    }

    /// The levels memoized, finest first.
    pub fn levels(&self) -> &[u8] {
        &self.levels
    }

    /// The slot of a level among those memoized.
    pub fn slot_of(&self, level: u8) -> Option<usize> {
        self.levels.iter().position(|&l| l == level)
    }

    /// The configuration.
    pub fn config(&self) -> &TransitConfig {
        &self.cfg
    }

    /// Whether a node (at any level memoized) is plain ground.
    pub fn is_plain(&self, node: QuadKey) -> bool {
        self.slot_of(node.level)
            .and_then(|s| self.plain[s].get(node.code as usize).copied())
            .unwrap_or(false)
    }

    /// Record a transit's outcome under its key.
    pub fn record(&mut self, key: TransitKey, outcome: TransitOutcome, rng: &mut Rng) {
        let capacity = self.cfg.capacity;
        self.kernels
            .entry(key)
            .or_default()
            .push(outcome, capacity, rng);
        self.recorded += 1;
    }

    /// The kernel of a key.
    pub fn kernel(&self, key: &TransitKey) -> Option<&Kernel> {
        self.kernels.get(key)
    }

    /// Whether a key's kernel is mature.
    pub fn mature(&self, key: &TransitKey) -> bool {
        self.kernels
            .get(key)
            .map(|k| k.mature(self.cfg.min_samples, self.cfg.max_stop_share))
            .unwrap_or(false)
    }

    /// An outcome of a mature kernel of at least `coherence`, at random
    /// (not yet counted as replayed: see [`Transits::note_replay`]).
    pub fn sample(
        &self,
        key: &TransitKey,
        coherence: f64,
        rng: &mut Rng,
    ) -> Option<TransitOutcome> {
        let kernel = self.kernels.get(key)?;
        if !kernel.mature(self.cfg.min_samples, self.cfg.max_stop_share) {
            return None;
        }
        if coherence > 0.0 && kernel.coherence() < coherence {
            return None;
        }
        kernel.sample(rng)
    }

    /// Count an outcome replayed at a level.
    pub fn note_replay(&mut self, outcome: &TransitOutcome, level: u8) {
        self.replayed += 1;
        self.ticks_replayed += outcome.ticks as u64;
        self.decisions_replayed += outcome.decisions as u64;
        if let Some(slot) = self.slot_of(level) {
            self.replayed_by_level[slot] += 1;
            self.decisions_by_level[slot] += outcome.decisions as u64;
        }
    }

    /// An outcome to replay for a key, if its kernel is mature (and the
    /// exploration draw does not ask for the full simulation), counted.
    pub fn replay(&mut self, key: &TransitKey, rng: &mut Rng) -> Option<TransitOutcome> {
        if self.cfg.explore > 0.0 && rng.chance(self.cfg.explore) {
            return None;
        }
        let coherence = if key.node.level < self.level() {
            self.cfg.coarse_coherence
        } else {
            0.0
        };
        let outcome = self.sample(key, coherence, rng)?;
        self.note_replay(&outcome, key.node.level);
        Some(outcome)
    }

    /// Keys with kernels, and how many are mature.
    pub fn maturity(&self) -> (usize, usize) {
        let mature = self
            .kernels
            .values()
            .filter(|k| k.mature(self.cfg.min_samples, self.cfg.max_stop_share))
            .count();
        (self.kernels.len(), mature)
    }

    /// Nodes (at any level) with at least one mature kernel.
    pub fn memoized_nodes(&self) -> usize {
        let mut nodes: Vec<QuadKey> = self
            .kernels
            .iter()
            .filter(|(_, k)| k.mature(self.cfg.min_samples, self.cfg.max_stop_share))
            .map(|(key, _)| key.node)
            .collect();
        nodes.sort();
        nodes.dedup();
        nodes.len()
    }

    /// A short account of the kernels, level by level.
    pub fn report(&self) -> String {
        let (keys, mature) = self.maturity();
        let mut out = format!(
            "{} keys, {} mature over {} nodes; {} transits recorded, {} replayed standing in for {} ticks and {} decisions",
            keys,
            mature,
            self.memoized_nodes(),
            self.recorded,
            self.replayed,
            self.ticks_replayed,
            self.decisions_replayed
        );
        if self.levels.len() > 1 {
            let by_level: Vec<String> = self
                .levels
                .iter()
                .enumerate()
                .map(|(s, &l)| {
                    format!(
                        "level {}: {} replayed for {} decisions",
                        l, self.replayed_by_level[s], self.decisions_by_level[s]
                    )
                })
                .collect();
            out.push_str("; ");
            out.push_str(&by_level.join(", "));
        }
        out
    }
}

#[cfg(test)]
mod transit_tests {
    use super::*;

    #[test]
    fn keys_quantise_and_kernels_mature() {
        assert_eq!(TransitKey::heading_class(0.0), 0);
        assert_eq!(TransitKey::heading_class(-0.01), HEADING_CLASSES - 1);
        assert_eq!(
            TransitKey::heading_class(std::f64::consts::PI),
            HEADING_CLASSES / 2
        );
        assert_eq!(TransitKey::context_class(0.0, false), 0);
        assert_eq!(TransitKey::context_class(2.0, true), 6);
        assert_eq!(
            TransitKey::dial_class(crate::entropy::Tempering::Temperature(1.0)),
            4
        );
        assert_eq!(
            TransitKey::dial_class(crate::entropy::Tempering::Temperature(0.5)),
            3
        );
        assert_eq!(
            TransitKey::dial_class(crate::entropy::Tempering::Entropy(0.5)),
            18
        );
        let mut k = Kernel::default();
        let mut rng = Rng::seed_from_u64(1);
        let leaving = TransitOutcome {
            exit_side: 1,
            exit_along: 0.5,
            heading: 0.0,
            ticks: 5,
            length: 4.0,
            decisions: 5,
            entropy: 2.0,
            straight: 4.0,
            deposits: [0.0; Pheromone::COUNT],
            via: [(0.0, 0.0); VIA],
            via_len: 0,
        };
        for _ in 0..40 {
            k.push(leaving, 8, &mut rng);
        }
        assert_eq!(k.outcomes.len(), 8);
        assert_eq!(k.seen, 40);
        assert!(k.mature(8, 0.02));
        k.push(
            TransitOutcome {
                exit_side: INSIDE,
                ..leaving
            },
            8,
            &mut rng,
        );
        k.push(
            TransitOutcome {
                exit_side: INSIDE,
                ..leaving
            },
            8,
            &mut rng,
        );
        assert!(!k.mature(8, 0.02), "two stops in forty-two taint the key");
        assert!(k.mature(8, 0.1));
        assert!(k
            .sample(&mut rng)
            .map(|o| o.exit_side != INSIDE)
            .unwrap_or(true));
    }
}
