//! The bridge: the colony's forward and backward filters made explicit.
//!
//! A thought's walk is a diffusion with a drift. Condition that
//! diffusion on where trips end, at food and back home, and the law of
//! the conditioned paths is a Schrödinger bridge: its density is the
//! product of a function running forward from the source and one
//! running backward from the outcome, and its drift is the gradient of
//! the log of the backward one, Doob's h-transform. The colony computes
//! this implicitly: the outbound walks sample the forward filter, and
//! the trail laid on the way home accumulates the backward filter, the
//! desirability of every place as a start for a successful trip. This
//! module computes both explicitly on a coarse graph of the medium, so
//! that the colony holds a model of its own flows, can measure how far
//! its actual flows depart from it (the surprise), can see how well its
//! trail approximates the desirability, and can follow the model's
//! drift as well as the trail.
//!
//! The passive kernel is the walk without any trail: from a block, to a
//! neighbouring block in proportion to the width of the passage between
//! them (portals included). A step costs `1 / temperature` in the
//! exponent, so the forward function is the discounted mass reaching a
//! block from the nest, the backward function the discounted mass of
//! the colony's own findings reachable from it, and the temperature is
//! the forcing pressure: as it falls, the product concentrates on the
//! shortest successful corridor, the known answer.

use ant_simulator::geometry::{Point, Position};
use ant_simulator::pheromone::Pheromone;
use ant_simulator::world::World;
use std::fmt::Write as _;

/// Configuration of the bridge.
#[derive(Clone, Debug, PartialEq)]
pub struct BridgeConfig {
    /// Cells per side of a block of the coarse graph.
    pub block: usize,
    /// The temperature: a step costs `1 / temperature`, so the discount
    /// per step is `exp(-1 / temperature)`.
    pub temperature: f64,
    /// Most relaxation sweeps per epoch.
    pub sweeps: usize,
    /// Relative change below which a relaxation has converged.
    pub tolerance: f64,
    /// Logits per unit alignment of a move with the desirability's
    /// gradient; 0 leaves the bridge an observer.
    pub gain: f64,
    /// Rate of the exponentially weighted findings and flows, per epoch.
    pub rate: f64,
    /// Ticks between relaxations.
    pub epoch_ticks: u64,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        BridgeConfig {
            block: 4,
            temperature: 8.0,
            sweeps: 400,
            tolerance: 1e-9,
            gain: 0.0,
            rate: 0.2,
            epoch_ticks: 100,
        }
    }
}

/// The bridge over a medium.
#[derive(Clone, Debug)]
pub struct Bridge {
    cfg: BridgeConfig,
    width: usize,
    height: usize,
    cols: usize,
    rows: usize,
    open: Vec<u32>,
    adjacency: Vec<Vec<(usize, f64)>>,
    directions: Vec<Vec<(f64, f64)>>,
    out_weight: Vec<f64>,
    source: usize,
    findings: Vec<f64>,
    findings_epoch: Vec<f64>,
    observed: Vec<f64>,
    observed_epoch: Vec<f64>,
    forward: Vec<f64>,
    backward: Vec<f64>,
    density: Vec<f64>,
    surprise: f64,
    correlation: f64,
    correlation_desirability: f64,
    entropy: f64,
    mass: f64,
    epochs: u64,
    /// Relaxations run.
    pub relaxations: u64,
    /// Sweeps the last relaxation took.
    pub last_sweeps: usize,
}

impl Bridge {
    /// A bridge over a world, with its source at the nest's cell.
    pub fn new(world: &World, nest: Position, cfg: BridgeConfig) -> Bridge {
        let block = cfg.block.max(1);
        let (width, height) = (world.width(), world.height());
        let cols = width.div_ceil(block);
        let rows = height.div_ceil(block);
        let n = cols * rows;
        let index = |x: usize, y: usize| (y / block) * cols + x / block;
        let mut open = vec![0u32; n];
        for y in 0..height {
            for x in 0..width {
                if world.is_passable(Position::new(x as i32, y as i32)) {
                    open[index(x, y)] += 1;
                }
            }
        }
        // Widths: passable cell pairs across the sides between blocks,
        // and portal cell pairs.
        let mut widths: std::collections::HashMap<(usize, usize), f64> =
            std::collections::HashMap::new();
        // The direction of each passage from each end: for a side, the
        // way to the neighbouring block; for a portal, the way to the
        // portal's cell on this side.
        let mut ways: std::collections::HashMap<(usize, usize), (f64, f64)> =
            std::collections::HashMap::new();
        let mut join =
            |a: usize, b: usize, from_a: Option<(f64, f64)>, from_b: Option<(f64, f64)>| {
                if a != b {
                    *widths.entry((a.min(b), a.max(b))).or_insert(0.0) += 1.0;
                    if let Some(d) = from_a {
                        ways.entry((a, b)).or_insert(d);
                    }
                    if let Some(d) = from_b {
                        ways.entry((b, a)).or_insert(d);
                    }
                }
            };
        for y in 0..height {
            for x in 0..width {
                let p = Position::new(x as i32, y as i32);
                if !world.is_passable(p) {
                    continue;
                }
                if x + 1 < width && world.is_passable(Position::new(x as i32 + 1, y as i32)) {
                    join(
                        index(x, y),
                        index(x + 1, y),
                        Some((1.0, 0.0)),
                        Some((-1.0, 0.0)),
                    );
                }
                if y + 1 < height && world.is_passable(Position::new(x as i32, y as i32 + 1)) {
                    join(
                        index(x, y),
                        index(x, y + 1),
                        Some((0.0, 1.0)),
                        Some((0.0, -1.0)),
                    );
                }
            }
        }
        let block = cfg.block.max(1) as f64;
        let centre = |blk: usize| -> Point {
            let (x, y) = ((blk % cols) as f64, (blk / cols) as f64);
            Point::new(
                ((x + 0.5) * block).min(width as f64 - 0.5),
                ((y + 0.5) * block).min(height as f64 - 0.5),
            )
        };
        for (a, b) in world.portal_cell_pairs() {
            if world.is_passable(a) && world.is_passable(b) {
                let (ba, bb) = (
                    index(a.x as usize, a.y as usize),
                    index(b.x as usize, b.y as usize),
                );
                let towards = |blk: usize, cell: Position| -> Option<(f64, f64)> {
                    let (dx, dy) = centre(blk).to(Point::center_of(cell));
                    let len = (dx * dx + dy * dy).sqrt();
                    (len > 1e-9).then(|| (dx / len, dy / len))
                };
                join(ba, bb, towards(ba, a), towards(bb, b));
            }
        }
        let mut adjacency: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut directions: Vec<Vec<(f64, f64)>> = vec![Vec::new(); n];
        for (&(a, b), &w) in &widths {
            adjacency[a].push((b, w));
            directions[a].push(ways.get(&(a, b)).copied().unwrap_or((0.0, 0.0)));
            adjacency[b].push((a, w));
            directions[b].push(ways.get(&(b, a)).copied().unwrap_or((0.0, 0.0)));
        }
        let out_weight: Vec<f64> = adjacency
            .iter()
            .map(|list| list.iter().map(|(_, w)| w).sum())
            .collect();
        let source = index(
            (nest.x.max(0) as usize).min(width - 1),
            (nest.y.max(0) as usize).min(height - 1),
        );
        Bridge {
            cfg,
            width,
            height,
            cols,
            rows,
            open,
            adjacency,
            directions,
            out_weight,
            source,
            findings: vec![0.0; n],
            findings_epoch: vec![0.0; n],
            observed: vec![0.0; n],
            observed_epoch: vec![0.0; n],
            forward: vec![0.0; n],
            backward: vec![0.0; n],
            density: vec![0.0; n],
            surprise: 0.0,
            correlation: 0.0,
            correlation_desirability: 0.0,
            entropy: 0.0,
            mass: 0.0,
            epochs: 0,
            relaxations: 0,
            last_sweeps: 0,
        }
    }

    /// The configuration.
    pub fn config(&self) -> &BridgeConfig {
        &self.cfg
    }

    /// The temperature.
    pub fn temperature(&self) -> f64 {
        self.cfg.temperature
    }

    /// Set the temperature (the next relaxation uses it).
    pub fn set_temperature(&mut self, temperature: f64) {
        self.cfg.temperature = temperature.max(1e-6);
    }

    /// Blocks across and down.
    pub fn blocks(&self) -> (usize, usize) {
        (self.cols, self.rows)
    }

    /// The block of a cell.
    pub fn block_of(&self, cell: Position) -> Option<usize> {
        if cell.x < 0
            || cell.y < 0
            || cell.x as usize >= self.width
            || cell.y as usize >= self.height
        {
            return None;
        }
        let b = self.cfg.block.max(1);
        Some((cell.y as usize / b) * self.cols + cell.x as usize / b)
    }

    /// The centre of a block.
    pub fn centre(&self, block: usize) -> Point {
        let b = self.cfg.block.max(1) as f64;
        let (x, y) = ((block % self.cols) as f64, (block / self.cols) as f64);
        Point::new(
            ((x + 0.5) * b).min(self.width as f64 - 0.5),
            ((y + 0.5) * b).min(self.height as f64 - 0.5),
        )
    }

    /// A finding at a cell: a sink of the backward filter.
    pub fn record_finding(&mut self, cell: Position) {
        if let Some(b) = self.block_of(cell) {
            self.findings_epoch[b] += 1.0;
        }
    }

    /// A thought out at a cell this tick: the observed flow.
    pub fn count(&mut self, cell: Position) {
        if let Some(b) = self.block_of(cell) {
            self.observed_epoch[b] += 1.0;
        }
    }

    /// The findings, exponentially weighted, per block.
    pub fn findings(&self) -> &[f64] {
        &self.findings
    }

    /// The observed outbound flow, exponentially weighted, per block.
    pub fn observed(&self) -> &[f64] {
        &self.observed
    }

    /// The forward function: discounted mass from the nest, per block.
    pub fn forward(&self) -> &[f64] {
        &self.forward
    }

    /// The backward function: the desirability, per block.
    pub fn backward(&self) -> &[f64] {
        &self.backward
    }

    /// The bridge density: the predicted share of successful trips
    /// through each block.
    pub fn density(&self) -> &[f64] {
        &self.density
    }

    /// How far the observed flow departs from the predicted density:
    /// the relative entropy of the one against the other, as a number
    /// in `0..1` (`kl / (1 + kl)`).
    pub fn surprise(&self) -> f64 {
        self.surprise
    }

    /// The correlation, over the blocks reached, of the log of the
    /// trail's mean level with the log of the bridge density: the trail
    /// is laid on the way home along successful routes, so it is the
    /// colony's implicit estimate of the product of the two filters.
    pub fn correlation(&self) -> f64 {
        self.correlation
    }

    /// The correlation of the log of the trail with the log of the
    /// desirability alone.
    pub fn correlation_desirability(&self) -> f64 {
        self.correlation_desirability
    }

    /// The entropy of the density, in nats.
    pub fn entropy(&self) -> f64 {
        self.entropy
    }

    /// The blocks the density is spread over, effectively
    /// (`exp(entropy)`).
    pub fn effective_blocks(&self) -> f64 {
        self.entropy.exp()
    }

    /// The mass of the bridge (the product's sum before normalising):
    /// how much discounted success the nest reaches.
    pub fn mass(&self) -> f64 {
        self.mass
    }

    /// Epochs relaxed.
    pub fn epochs(&self) -> u64 {
        self.epochs
    }

    /// Whether the backward filter has anything to run from.
    pub fn has_findings(&self) -> bool {
        self.findings.iter().any(|&f| f > 0.0)
    }

    /// An epoch ends: fold the epoch's findings and flows into their
    /// running means, solve the two filters, and measure.
    pub fn relax(&mut self, world: &World) {
        let rate = self.cfg.rate.clamp(0.0, 1.0);
        let n = self.findings.len();
        for b in 0..n {
            self.findings[b] += rate * (self.findings_epoch[b] - self.findings[b]);
            self.observed[b] += rate * (self.observed_epoch[b] - self.observed[b]);
            self.findings_epoch[b] = 0.0;
            self.observed_epoch[b] = 0.0;
        }
        self.epochs += 1;
        self.relaxations += 1;
        let gamma = (-1.0 / self.cfg.temperature.max(1e-6)).exp();
        // Forward: mass from the nest.
        let mut source = vec![0.0; n];
        source[self.source] = 1.0;
        self.forward = self.solve(&source, gamma, true);
        // Backward: the desirability, from the findings.
        let total: f64 = self.findings.iter().sum();
        let sink: Vec<f64> = if total > 0.0 {
            self.findings.iter().map(|f| f / total).collect()
        } else {
            vec![0.0; n]
        };
        self.backward = self.solve(&sink, gamma, false);
        // The product.
        let mut product: Vec<f64> = self
            .forward
            .iter()
            .zip(&self.backward)
            .map(|(f, b)| f * b)
            .collect();
        self.mass = product.iter().sum();
        if self.mass > 0.0 {
            product.iter_mut().for_each(|p| *p /= self.mass);
        }
        self.density = product;
        self.entropy = -self
            .density
            .iter()
            .filter(|&&d| d > 0.0)
            .map(|&d| d * d.ln())
            .sum::<f64>();
        // Surprise: the observed flow against the predicted density.
        let observed_total: f64 = self.observed.iter().sum();
        self.surprise = if observed_total > 0.0 && self.mass > 0.0 {
            let eps = 1e-6;
            let kl: f64 = self
                .observed
                .iter()
                .zip(&self.density)
                .filter(|(o, _)| **o > 0.0)
                .map(|(o, d)| {
                    let p = o / observed_total;
                    p * (p / (d + eps)).ln()
                })
                .sum::<f64>()
                .max(0.0);
            kl / (1.0 + kl)
        } else {
            0.0
        };
        // Correlation of the trail with the density and the desirability.
        let (density, desirability) = self.trail_correlations(world);
        self.correlation = density;
        self.correlation_desirability = desirability;
    }

    /// Solve `x = source + gamma * K x`, with `K` the passive kernel
    /// applied forward (mass flows out of a block along its widths) or
    /// backward (a block takes the mean of its neighbours' values).
    fn solve(&mut self, source: &[f64], gamma: f64, forward: bool) -> Vec<f64> {
        let n = source.len();
        let mut x = source.to_vec();
        let mut next = vec![0.0; n];
        let mut sweeps = 0;
        for _ in 0..self.cfg.sweeps.max(1) {
            sweeps += 1;
            next.copy_from_slice(source);
            for a in 0..n {
                if self.out_weight[a] <= 0.0 || x[a] == 0.0 && forward {
                    continue;
                }
                if forward {
                    // Mass at `a` flows to its neighbours.
                    let share = gamma * x[a] / self.out_weight[a];
                    for &(b, w) in &self.adjacency[a] {
                        next[b] += share * w;
                    }
                } else {
                    // `a` takes the mean of its neighbours.
                    let mut acc = 0.0;
                    for &(b, w) in &self.adjacency[a] {
                        acc += w * x[b];
                    }
                    next[a] += gamma * acc / self.out_weight[a];
                }
            }
            let scale = next.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
            let change = next
                .iter()
                .zip(&x)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            std::mem::swap(&mut x, &mut next);
            if change <= self.cfg.tolerance * scale {
                break;
            }
        }
        self.last_sweeps = sweeps;
        x
    }

    fn trail_correlations(&self, world: &World) -> (f64, f64) {
        let b = self.cfg.block.max(1);
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut zs = Vec::new();
        for block in 0..self.density.len() {
            if self.open[block] == 0 || self.backward[block] <= 0.0 || self.forward[block] <= 0.0 {
                continue;
            }
            let (bx, by) = ((block % self.cols) * b, (block / self.cols) * b);
            let mut trail = 0.0;
            let mut cells = 0;
            for y in by..(by + b).min(self.height) {
                for x in bx..(bx + b).min(self.width) {
                    let p = Position::new(x as i32, y as i32);
                    if world.is_passable(p) {
                        trail += world.level(p, Pheromone::Trail);
                        cells += 1;
                    }
                }
            }
            if cells == 0 {
                continue;
            }
            xs.push((trail / cells as f64 + 1.0).ln());
            ys.push((self.density[block] + 1e-12).ln());
            zs.push(self.backward[block].ln());
        }
        (pearson(&xs, &ys), pearson(&xs, &zs))
    }

    fn interpolated(&self, field: &[f64], p: Point) -> f64 {
        let b = self.cfg.block.max(1) as f64;
        let fx = (p.x / b - 0.5).clamp(0.0, (self.cols - 1) as f64);
        let fy = (p.y / b - 0.5).clamp(0.0, (self.rows - 1) as f64);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.cols - 1), (y0 + 1).min(self.rows - 1));
        let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
        let at = |x: usize, y: usize| field[y * self.cols + x];
        (1.0 - tx) * (1.0 - ty) * at(x0, y0)
            + tx * (1.0 - ty) * at(x1, y0)
            + (1.0 - tx) * ty * at(x0, y1)
            + tx * ty * at(x1, y1)
    }

    /// The desirability at a point, interpolated between block centres.
    pub fn desirability_at(&self, p: Point) -> f64 {
        self.interpolated(&self.backward, p)
    }

    /// The predicted density at a point, interpolated.
    pub fn density_at(&self, p: Point) -> f64 {
        self.interpolated(&self.density, p)
    }

    /// The drift of the conditioned walk at a point: the mean
    /// displacement out of the point's block under the passive kernel
    /// reweighted by the desirability of where each passage leads
    /// (Doob's transform of the kernel), as a unit vector; none where
    /// no passage leads anywhere desirable. It runs only through
    /// passages, never into a wall.
    pub fn drift(&self, p: Point) -> Option<(f64, f64)> {
        let block = self.block_of(p.cell())?;
        let (mut dx, mut dy, mut total) = (0.0, 0.0, 0.0);
        for (&(next, w), &(ux, uy)) in self.adjacency[block].iter().zip(&self.directions[block]) {
            let weight = w * self.backward[next];
            dx += weight * ux;
            dy += weight * uy;
            total += weight;
        }
        if total <= 0.0 {
            return None;
        }
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-9 || !len.is_finite() {
            None
        } else {
            Some((dx / len, dy / len))
        }
    }

    /// The density drawn block by block, expanded to cells: walls `#`,
    /// the nest `@`, shades for the predicted share.
    pub fn render(&self, world: &World) -> String {
        let shades: &[u8] = b" .:-=+*#%";
        let peak = self
            .density
            .iter()
            .cloned()
            .fold(0.0f64, f64::max)
            .max(1e-12);
        let mut out = String::with_capacity((self.width + 1) * self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let p = Position::new(x as i32, y as i32);
                if !world.is_passable(p) {
                    out.push('#');
                    continue;
                }
                let block = self.block_of(p).unwrap_or(0);
                if block == self.source
                    && (x % self.cfg.block.max(1) == 0)
                    && (y % self.cfg.block.max(1) == 0)
                {
                    out.push('@');
                    continue;
                }
                let v = (self.density[block] / peak).sqrt();
                let idx = (v * (shades.len() - 1) as f64).round() as usize;
                out.push(shades[idx.min(shades.len() - 1)] as char);
            }
            out.push('\n');
        }
        out
    }

    /// A summary line.
    pub fn report(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "  bridge: {} epochs at temperature {:.1}, surprise {:.3}, trail~density r {:.3} (trail~desirability r {:.3}), density over {:.1} effective blocks of {}, mass {:.4}",
            self.epochs,
            self.cfg.temperature,
            self.surprise,
            self.correlation,
            self.correlation_desirability,
            self.effective_blocks(),
            self.open.iter().filter(|&&o| o > 0).count(),
            self.mass
        );
        out
    }
}

/// Pearson's correlation.
pub fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 3 {
        return 0.0;
    }
    let mx = xs.iter().sum::<f64>() / n as f64;
    let my = ys.iter().sum::<f64>() / n as f64;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (x, y) in xs.iter().zip(ys) {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx) * (x - mx);
        syy += (y - my) * (y - my);
    }
    if sxx <= 0.0 || syy <= 0.0 {
        0.0
    } else {
        sxy / (sxx * syy).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ant_simulator::rng::Rng;
    use ant_simulator::world::{Rect, WorldConfig};

    fn world_with_wall() -> World {
        let cfg = WorldConfig {
            width: 32,
            height: 24,
            nest: Position::new(3, 12),
            nest_radius: 1,
            random_food: None,
            walls: vec![Rect::new(Position::new(12, 3), Position::new(19, 23))],
            ..WorldConfig::default()
        };
        World::new(cfg, &mut Rng::seed_from_u64(1))
    }

    #[test]
    fn the_product_of_the_filters_runs_through_the_gap_and_narrows_as_the_temperature_falls() {
        let world = world_with_wall();
        let mut bridge = Bridge::new(&world, Position::new(3, 12), BridgeConfig::default());
        assert!(!bridge.has_findings());
        bridge.record_finding(Position::new(28, 12));
        bridge.relax(&world);
        assert!(bridge.has_findings());
        assert!(bridge.mass() > 0.0);
        let gap = bridge.block_of(Position::new(13, 1)).unwrap();
        let wall_side = bridge.block_of(Position::new(13, 13)).unwrap();
        assert!(
            bridge.density()[gap] > 0.0,
            "the way round runs through the gap"
        );
        assert_eq!(bridge.density()[wall_side], 0.0, "not through the wall");
        // The desirability rises towards the finding.
        let near = bridge.desirability_at(Point::new(26.5, 12.5));
        let far = bridge.desirability_at(Point::new(4.5, 20.5));
        assert!(near > far);
        let (dx, _) = bridge.drift(Point::new(20.5, 12.5)).expect("a drift");
        assert!(dx > 0.0, "the drift points at the finding");
        let warm = bridge.entropy();
        bridge.set_temperature(5.0);
        bridge.relax(&world);
        let cold = bridge.entropy();
        assert!(cold < warm, "colder is narrower: {cold} < {warm}");
        // Surprise: a flow along the corridor is unsurprising, one
        // piled against the wall is.
        let along = bridge.block_of(Position::new(10, 12)).unwrap();
        for _ in 0..50 {
            bridge.count(Position::new(10, 12));
        }
        bridge.relax(&world);
        let corridor = bridge.surprise();
        let _ = along;
        for _ in 0..500 {
            bridge.count(Position::new(3, 22));
        }
        bridge.relax(&world);
        assert!(
            bridge.surprise() > corridor,
            "{} > {corridor}",
            bridge.surprise()
        );
    }

    #[test]
    fn pearson_is_one_on_a_line_and_zero_on_noise_it_cannot_see() {
        assert!((pearson(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]) - 1.0).abs() < 1e-12);
        assert_eq!(pearson(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]), 0.0);
        assert_eq!(pearson(&[1.0, 2.0], &[1.0, 2.0]), 0.0, "too few");
    }
}
