//! Hive cognitive geometry: the colony's movement history as a
//! path-topological structure collapsed over time, and a queen who thinks
//! through it.
//!
//! The aggregate of all the colony's movements, kept with forgetting, is a
//! flow field: where ants persistently go (the *invariant* skeleton, which
//! is the trail network and the geometry of the ground) and how the current
//! flow departs from it (the *non-invariant* residual, where the colony's
//! disorder lives). A [`MovementHistory`] keeps both at two grains (cells
//! and sectors) and two time constants (slow and fast), and summarises the
//! skeleton's topology as channels and loops.
//!
//! A [`Queen`] acts within time on a coarse clock. Each epoch she reads the
//! retrodictive coarse-grained field, updates a recurrent thought, and
//! expresses it in the entropy dials of the hierarchy, so her thoughts are
//! written into the colony's non-invariant movement over the epoch that
//! follows. A [`Readout`] fitted on the field retrodicts her past thoughts
//! from the present field alone: the colony's movements are her memory,
//! and [`memory_capacity`] measures how much of it the field carries, lag
//! by lag, in the invariant and the non-invariant component. With
//! recursion on, what she recalls from the field feeds her next thought,
//! so cognitive material laid down in the colony's paths is included
//! self-recursively in future cognition.

use crate::entropy::EntropyControl;
use crate::geometry::{Point, Position};
use crate::hierarchy::{Hierarchy, NodeId};
use crate::rng::Rng;

/// How the movement history is kept.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoryConfig {
    /// Cells per sector side (the coarse grain).
    pub sector: usize,
    /// Half-life of the slow, invariant accumulator, seconds.
    pub slow_half_life_s: f64,
    /// Half-life of the fast accumulator whose departure from the slow one
    /// is the non-invariant residual, seconds (an epoch of the queen's
    /// clock, by default).
    pub fast_half_life_s: f64,
    /// Steady move rate (moves per tick through a cell) above which a cell
    /// counts as part of a channel.
    pub channel_rate: f64,
    /// Axial order of the flow (0 wandering, 1 straight two-way traffic)
    /// above which a cell counts as part of a channel.
    pub channel_alignment: f64,
    /// Whether a reading is multi-scale: the pyramid of grains from the
    /// whole field down to the sector, or the sector grain alone.
    pub multiscale: bool,
}

impl Default for HistoryConfig {
    fn default() -> Self {
        HistoryConfig {
            sector: 8,
            slow_half_life_s: 3600.0,
            fast_half_life_s: 60.0,
            channel_rate: 0.02,
            channel_alignment: 0.4,
            multiscale: true,
        }
    }
}

/// Exponentially weighted flow through a place: how many moves, and their
/// summed unit directions.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Flow {
    /// Weighted count of moves.
    pub weight: f64,
    /// Weighted sum of the moves' unit directions, x.
    pub vx: f64,
    /// Weighted sum of the moves' unit directions, y.
    pub vy: f64,
    /// Weighted sum of `cos 2θ` over the moves' directions, for the
    /// axial (nematic) order that bidirectional traffic keeps.
    pub cx: f64,
    /// Weighted sum of `sin 2θ`.
    pub sx: f64,
    /// Weighted sum of the cosine of each move's turn from its mover's
    /// heading: how straight the paths through here run.
    pub straight: f64,
}

impl Flow {
    fn add(&mut self, ux: f64, uy: f64, turn_cos: f64, w: f64) {
        self.weight += w;
        self.vx += ux * w;
        self.vy += uy * w;
        // cos 2θ = ux² − uy², sin 2θ = 2 ux uy for a unit vector.
        self.cx += (ux * ux - uy * uy) * w;
        self.sx += 2.0 * ux * uy * w;
        self.straight += turn_cos * w;
    }

    fn scale_in_place(&mut self, k: f64) {
        self.weight *= k;
        self.vx *= k;
        self.vy *= k;
        self.cx *= k;
        self.sx *= k;
        self.straight *= k;
    }

    fn scaled(self, k: f64) -> Flow {
        Flow {
            weight: self.weight * k,
            vx: self.vx * k,
            vy: self.vy * k,
            cx: self.cx * k,
            sx: self.sx * k,
            straight: self.straight * k,
        }
    }

    /// Straightness of the paths: the mean cosine of the moves' turns
    /// from their movers' headings, 1 when every move holds its course,
    /// 0 when turns are uniformly random, negative for doubling back.
    /// This is the spatial trace of the decision entropy.
    pub fn straightness(&self) -> f64 {
        if self.weight <= 0.0 {
            0.0
        } else {
            self.straight / self.weight
        }
    }

    /// Axial order of the flow: 1 when every move lies along one line
    /// (either way along it), 0 when moves point every way. Straight
    /// two-way traffic on a trail scores high; wandering scores low.
    pub fn nematic(&self) -> f64 {
        if self.weight <= 0.0 {
            0.0
        } else {
            (self.cx * self.cx + self.sx * self.sx).sqrt() / self.weight
        }
    }

    /// How one-way the flow is: the length of the mean unit direction, 0
    /// for movement in all directions equally, 1 for a single direction.
    pub fn alignment(&self) -> f64 {
        if self.weight <= 0.0 {
            0.0
        } else {
            (self.vx * self.vx + self.vy * self.vy).sqrt() / self.weight
        }
    }

    /// Mean unit direction of the flow (zero when there is none).
    pub fn mean(&self) -> (f64, f64) {
        if self.weight <= 0.0 {
            (0.0, 0.0)
        } else {
            (self.vx / self.weight, self.vy / self.weight)
        }
    }
}

/// Which part of the movement history a reading uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Component {
    /// The slow, invariant skeleton.
    Invariant,
    /// The non-invariant residual of the current flow over the skeleton.
    Residual,
    /// Both together.
    Both,
}

/// Topology of the invariant skeleton: its channels (eight-connected groups
/// of channel cells) and the loops they close (regions of the ground the
/// channels enclose: the first Betti number of the skeleton).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Topology {
    /// Channel cells.
    pub cells: usize,
    /// Connected channels.
    pub channels: usize,
    /// Loops: regions enclosed by channels.
    pub loops: usize,
}

/// Global summary of the movement history.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldSummary {
    /// Steady moves per tick over the whole world.
    pub density: f64,
    /// Weight-averaged alignment of the invariant flow over sectors.
    pub alignment: f64,
    /// Entropy of the invariant flow's distribution over sectors, nats.
    pub spatial_entropy: f64,
    /// Root-mean-square residual density over sectors, relative to the
    /// invariant density.
    pub residual_energy: f64,
    /// Topology of the skeleton.
    pub topology: Topology,
}

/// The colony's movement history, collapsed over time at two grains and
/// two time constants.
#[derive(Clone, Debug)]
pub struct MovementHistory {
    width: usize,
    height: usize,
    sector: usize,
    cols: usize,
    rows: usize,
    fine_slow: Vec<Flow>,
    fine_fast: Vec<Flow>,
    coarse_slow: Vec<Flow>,
    coarse_fast: Vec<Flow>,
    slow_r: f64,
    fast_r: f64,
    /// Forgetting is lazy: the accumulators store values divided by the
    /// running decay factor, so that a tick costs nothing and a value is
    /// read back as stored × factor. The factor is renormalised into the
    /// stores before it underflows.
    slow_g: f64,
    fast_g: f64,
    cfg: HistoryConfig,
    /// Moves recorded so far.
    pub moves: u64,
}

impl MovementHistory {
    /// An empty history for a world of the given size and tick.
    pub fn new(width: usize, height: usize, tick_s: f64, cfg: HistoryConfig) -> Self {
        let sector = cfg.sector.max(1);
        let cols = width.div_ceil(sector);
        let rows = height.div_ceil(sector);
        let retention = |half_life: f64| -> f64 {
            if half_life <= 0.0 {
                0.0
            } else {
                0.5f64.powf(tick_s / half_life)
            }
        };
        MovementHistory {
            width,
            height,
            sector,
            cols,
            rows,
            fine_slow: vec![Flow::default(); width * height],
            fine_fast: vec![Flow::default(); width * height],
            coarse_slow: vec![Flow::default(); cols * rows],
            coarse_fast: vec![Flow::default(); cols * rows],
            slow_r: retention(cfg.slow_half_life_s),
            fast_r: retention(cfg.fast_half_life_s),
            slow_g: 1.0,
            fast_g: 1.0,
            cfg,
            moves: 0,
        }
    }

    /// Fold the running decay factors into the stores.
    fn renormalise(&mut self) {
        let (sg, fg) = (self.slow_g, self.fast_g);
        for f in self.fine_slow.iter_mut().chain(self.coarse_slow.iter_mut()) {
            f.scale_in_place(sg);
        }
        for f in self.fine_fast.iter_mut().chain(self.coarse_fast.iter_mut()) {
            f.scale_in_place(fg);
        }
        self.slow_g = 1.0;
        self.fast_g = 1.0;
    }

    /// Sectors across and down.
    pub fn sectors(&self) -> (usize, usize) {
        (self.cols, self.rows)
    }

    /// Cells per sector side.
    pub fn sector_size(&self) -> usize {
        self.sector
    }

    /// The sector a cell belongs to.
    pub fn sector_of(&self, cell: Position) -> Option<usize> {
        if cell.x < 0
            || cell.y < 0
            || cell.x as usize >= self.width
            || cell.y as usize >= self.height
        {
            return None;
        }
        Some((cell.y as usize / self.sector) * self.cols + cell.x as usize / self.sector)
    }

    fn cell_index(&self, cell: Position) -> Option<usize> {
        if cell.x < 0
            || cell.y < 0
            || cell.x as usize >= self.width
            || cell.y as usize >= self.height
        {
            return None;
        }
        Some(cell.y as usize * self.width + cell.x as usize)
    }

    /// Record one move, weighted by its length, at the cell it ends in;
    /// `turn` is the angle (radians) by which the move departs from its
    /// mover's heading.
    pub fn record(&mut self, from: Point, to: Point, turn: f64) {
        let (dx, dy) = from.to(to);
        let len = (dx * dx + dy * dy).sqrt();
        if len <= 1e-12 {
            return;
        }
        let (ux, uy) = (dx / len, dy / len);
        let turn_cos = turn.cos();
        let cell = to.cell();
        // Stored at the current scale of each accumulator; a dead
        // accumulator (no half-life) keeps nothing.
        let ws = if self.slow_r > 0.0 {
            len / self.slow_g
        } else {
            0.0
        };
        let wf = if self.fast_r > 0.0 {
            len / self.fast_g
        } else {
            0.0
        };
        if let Some(i) = self.cell_index(cell) {
            self.fine_slow[i].add(ux, uy, turn_cos, ws);
            self.fine_fast[i].add(ux, uy, turn_cos, wf);
        }
        if let Some(s) = self.sector_of(cell) {
            self.coarse_slow[s].add(ux, uy, turn_cos, ws);
            self.coarse_fast[s].add(ux, uy, turn_cos, wf);
        }
        self.moves += 1;
    }

    /// Let one tick pass: both accumulators forget (lazily, by advancing
    /// their decay factors).
    pub fn step(&mut self) {
        self.slow_g *= self.slow_r;
        self.fast_g *= self.fast_r;
        if self.slow_g < 1e-150 || self.fast_g < 1e-150 {
            self.renormalise();
        }
    }

    /// Normalisation that turns an accumulator into a steady rate per tick
    /// (moves are added before the tick's forgetting, so a constant rate
    /// `λ` settles at `λ r / (1 − r)`).
    fn slow_norm(&self) -> f64 {
        if self.slow_r <= 0.0 {
            0.0
        } else {
            (1.0 - self.slow_r) / self.slow_r
        }
    }

    fn fast_norm(&self) -> f64 {
        if self.fast_r <= 0.0 {
            0.0
        } else {
            (1.0 - self.fast_r) / self.fast_r
        }
    }

    /// The invariant flow through a sector, as a steady rate per tick.
    pub fn invariant(&self, sector: usize) -> Flow {
        self.coarse_slow[sector].scaled(self.slow_g * self.slow_norm())
    }

    /// The current flow through a sector, as a rate per tick.
    pub fn current(&self, sector: usize) -> Flow {
        self.coarse_fast[sector].scaled(self.fast_g * self.fast_norm())
    }

    /// The invariant flow over the whole field, as a steady rate per tick.
    pub fn invariant_whole(&self) -> Flow {
        let mut whole = Flow::default();
        for f in &self.coarse_slow {
            whole.weight += f.weight;
            whole.vx += f.vx;
            whole.vy += f.vy;
            whole.cx += f.cx;
            whole.sx += f.sx;
            whole.straight += f.straight;
        }
        whole.scaled(self.slow_g * self.slow_norm())
    }

    /// The current flow over the whole field, as a rate per tick.
    pub fn current_whole(&self) -> Flow {
        let mut whole = Flow::default();
        for f in &self.coarse_fast {
            whole.weight += f.weight;
            whole.vx += f.vx;
            whole.vy += f.vy;
            whole.cx += f.cx;
            whole.sx += f.sx;
            whole.straight += f.straight;
        }
        whole.scaled(self.fast_g * self.fast_norm())
    }

    /// The non-invariant residual of a sector: the current density minus
    /// the invariant one, and the current mean direction minus the
    /// invariant one.
    pub fn residual(&self, sector: usize) -> (f64, (f64, f64)) {
        let inv = self.invariant(sector);
        let cur = self.current(sector);
        let (ix, iy) = inv.mean();
        let (cx, cy) = cur.mean();
        (cur.weight - inv.weight, (cx - ix, cy - iy))
    }

    /// The invariant flow through a cell, as a steady rate per tick.
    pub fn fine_invariant(&self, cell: Position) -> Flow {
        self.cell_index(cell)
            .map(|i| self.fine_slow[i].scaled(self.slow_g * self.slow_norm()))
            .unwrap_or_default()
    }

    /// The current flow through a cell, as a rate per tick.
    pub fn fine_current(&self, cell: Position) -> Flow {
        self.cell_index(cell)
            .map(|i| self.fine_fast[i].scaled(self.fast_g * self.fast_norm()))
            .unwrap_or_default()
    }

    /// Numbers per block in a reading.
    pub const READING: usize = 6;

    /// The blocks a reading is taken over, as cell rectangles
    /// `(x0, y0, x1, y1)` with exclusive far corners: the pyramid of
    /// grains from the whole field down to the sector (each level halves
    /// the grain), or the sectors alone when the reading is not
    /// multi-scale.
    pub fn blocks(&self) -> Vec<(usize, usize, usize, usize)> {
        let mut grain = self.sector;
        if self.cfg.multiscale {
            while grain < self.width.max(self.height) {
                grain *= 2;
            }
        }
        let mut out = Vec::new();
        loop {
            let cols = self.width.div_ceil(grain);
            let rows = self.height.div_ceil(grain);
            for r in 0..rows {
                for c in 0..cols {
                    out.push((
                        c * grain,
                        r * grain,
                        ((c + 1) * grain).min(self.width),
                        ((r + 1) * grain).min(self.height),
                    ));
                }
            }
            if grain <= self.sector {
                break;
            }
            grain /= 2;
        }
        out
    }

    /// Reading of one accumulator over a block of cells, built from the
    /// cells so that local coherence survives the coarse grain: density
    /// (moves per tick), the weighted mean polar alignment of the cells,
    /// their weighted mean axial order, the straightness of the paths,
    /// and the block's mean direction.
    fn block_reading(
        &self,
        block: (usize, usize, usize, usize),
        slow: bool,
    ) -> [f64; Self::READING] {
        let (norm, fine) = if slow {
            (self.slow_g * self.slow_norm(), &self.fine_slow)
        } else {
            (self.fast_g * self.fast_norm(), &self.fine_fast)
        };
        let (x0, y0, x1, y1) = block;
        let (mut align, mut nematic) = (0.0, 0.0);
        let mut whole = Flow::default();
        for y in y0..y1.min(self.height) {
            for x in x0..x1.min(self.width) {
                let f = &fine[y * self.width + x];
                align += f.weight * f.alignment();
                nematic += f.weight * f.nematic();
                whole.weight += f.weight;
                whole.vx += f.vx;
                whole.vy += f.vy;
                whole.straight += f.straight;
            }
        }
        let (mx, my) = whole.mean();
        if whole.weight > 0.0 {
            [
                whole.weight * norm,
                align / whole.weight,
                nematic / whole.weight,
                whole.straightness(),
                mx,
                my,
            ]
        } else {
            [0.0; Self::READING]
        }
    }

    /// The coarse-grained reading of one component: six numbers per
    /// block of the pyramid (see [`MovementHistory::blocks`]). Invariant:
    /// steady density, local polar alignment, local axial order, path
    /// straightness, mean direction (x, y). Residual: the same six for
    /// the current flow minus the invariant ones.
    pub fn features(&self, component: Component) -> Vec<f64> {
        let blocks = self.blocks();
        let mut out = Vec::with_capacity(2 * blocks.len() * Self::READING);
        if matches!(component, Component::Invariant | Component::Both) {
            for &b in &blocks {
                out.extend_from_slice(&self.block_reading(b, true));
            }
        }
        if matches!(component, Component::Residual | Component::Both) {
            for &b in &blocks {
                let inv = self.block_reading(b, true);
                let cur = self.block_reading(b, false);
                for k in 0..Self::READING {
                    out.push(cur[k] - inv[k]);
                }
            }
        }
        out
    }

    /// Topology of the invariant skeleton: channel cells are those with
    /// enough steady flow and enough axial order; channels are their
    /// eight-connected groups; loops are the four-connected regions of
    /// other cells that channels enclose (regions not reaching the edge
    /// of the world).
    pub fn topology(&self) -> Topology {
        let norm = self.slow_g * self.slow_norm();
        let channel: Vec<bool> = self
            .fine_slow
            .iter()
            .map(|f| {
                f.weight * norm >= self.cfg.channel_rate
                    && f.nematic() >= self.cfg.channel_alignment
            })
            .collect();
        let w = self.width as i32;
        let h = self.height as i32;
        let idx = |x: i32, y: i32| (y as usize) * self.width + x as usize;
        let cells = channel.iter().filter(|&&c| c).count();
        // Channels: eight-connected components of channel cells.
        let mut seen = vec![false; channel.len()];
        let mut channels = 0usize;
        for start in 0..channel.len() {
            if !channel[start] || seen[start] {
                continue;
            }
            channels += 1;
            let mut stack = vec![start];
            seen[start] = true;
            while let Some(i) = stack.pop() {
                let (x, y) = ((i % self.width) as i32, (i / self.width) as i32);
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (nx, ny) = (x + dx, y + dy);
                        if (dx == 0 && dy == 0) || nx < 0 || ny < 0 || nx >= w || ny >= h {
                            continue;
                        }
                        let j = idx(nx, ny);
                        if channel[j] && !seen[j] {
                            seen[j] = true;
                            stack.push(j);
                        }
                    }
                }
            }
        }
        // Loops: four-connected components of the other cells that do not
        // touch the edge of the world.
        let mut seen = vec![false; channel.len()];
        let mut loops = 0usize;
        for start in 0..channel.len() {
            if channel[start] || seen[start] {
                continue;
            }
            let mut enclosed = true;
            let mut stack = vec![start];
            seen[start] = true;
            while let Some(i) = stack.pop() {
                let (x, y) = ((i % self.width) as i32, (i / self.width) as i32);
                if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                    enclosed = false;
                }
                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w || ny >= h {
                        continue;
                    }
                    let j = idx(nx, ny);
                    if !channel[j] && !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
            if enclosed {
                loops += 1;
            }
        }
        Topology {
            cells,
            channels,
            loops,
        }
    }

    /// Global summary.
    pub fn summary(&self) -> FieldSummary {
        let n = self.cols * self.rows;
        let mut density = 0.0;
        let mut aligned = 0.0;
        let mut residual_sq = 0.0;
        let mut weights = Vec::with_capacity(n);
        for s in 0..n {
            let f = self.invariant(s);
            density += f.weight;
            aligned += f.weight * f.alignment();
            weights.push(f.weight);
            let (dd, _) = self.residual(s);
            residual_sq += dd * dd;
        }
        let spatial_entropy = if density > 0.0 {
            -weights
                .iter()
                .filter(|&&w| w > 0.0)
                .map(|&w| {
                    let p = w / density;
                    p * p.ln()
                })
                .sum::<f64>()
        } else {
            0.0
        };
        FieldSummary {
            density,
            alignment: if density > 0.0 {
                aligned / density
            } else {
                0.0
            },
            spatial_entropy,
            residual_energy: if density > 0.0 {
                (residual_sq / n as f64).sqrt() / (density / n as f64)
            } else {
                0.0
            },
            topology: self.topology(),
        }
    }

    /// Glyph for a flow: an arrow for one-way flow, a line for two-way
    /// traffic along an axis, a dot for flow without either, a space for
    /// none.
    fn glyph(flow: &Flow, floor: f64) -> char {
        if flow.weight < floor {
            ' '
        } else if flow.alignment() >= 0.15 {
            let (mx, my) = flow.mean();
            let angle = my.atan2(mx);
            let k = ((angle / std::f64::consts::FRAC_PI_4).round() as i32).rem_euclid(8);
            ['→', '↘', '↓', '↙', '←', '↖', '↑', '↗'][k as usize]
        } else if flow.nematic() >= 0.3 {
            let axis = 0.5 * flow.sx.atan2(flow.cx);
            let k = ((axis / std::f64::consts::FRAC_PI_4).round() as i32).rem_euclid(4);
            ['─', '╲', '│', '╱'][k as usize]
        } else {
            '·'
        }
    }

    /// The coarse field as text, one glyph per sector: an arrow for the
    /// mean direction of one-way flow, a line for the axis of two-way
    /// traffic, a dot for flow without either; for the residual, an
    /// arrow for the direction the current flow departs in.
    pub fn render(&self, component: Component) -> String {
        let mut out = String::new();
        let max_density = (0..self.cols * self.rows)
            .map(|s| self.invariant(s).weight)
            .fold(0.0, f64::max)
            .max(1e-12);
        for r in 0..self.rows {
            for c in 0..self.cols {
                let s = r * self.cols + c;
                let ch = match component {
                    Component::Residual => {
                        let (dd, (mx, my)) = self.residual(s);
                        let strength = (mx * mx + my * my).sqrt();
                        if dd.abs() < 0.05 * max_density {
                            ' '
                        } else if strength < 0.15 {
                            '·'
                        } else {
                            let angle = my.atan2(mx);
                            let k = ((angle / std::f64::consts::FRAC_PI_4).round() as i32)
                                .rem_euclid(8);
                            ['→', '↘', '↓', '↙', '←', '↖', '↑', '↗'][k as usize]
                        }
                    }
                    _ => Self::glyph(&self.invariant(s), 0.05 * max_density),
                };
                out.push(ch);
            }
            out.push('\n');
        }
        out
    }

    /// The invariant skeleton as text, one glyph per cell: channel cells
    /// by the glyph of their flow, other cells with some steady flow as a
    /// dot, the rest blank.
    pub fn render_skeleton(&self) -> String {
        let norm = self.slow_g * self.slow_norm();
        let mut out = String::with_capacity((self.width + 1) * self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let f = self.fine_slow[y * self.width + x].scaled(norm);
                let ch = if f.weight >= self.cfg.channel_rate
                    && f.nematic() >= self.cfg.channel_alignment
                {
                    Self::glyph(&f, 0.0)
                } else if f.weight >= 0.25 * self.cfg.channel_rate {
                    '·'
                } else {
                    ' '
                };
                out.push(ch);
            }
            out.push('\n');
        }
        out
    }
}

/// Solve `a · x = b` by Gaussian elimination with partial pivoting.
#[allow(clippy::needless_range_loop)]
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap())
            .unwrap_or(col);
        a.swap(col, pivot);
        b.swap(col, pivot);
        let p = a[col][col];
        if p.abs() < 1e-15 {
            continue;
        }
        for row in col + 1..n {
            let f = a[row][col] / p;
            if f == 0.0 {
                continue;
            }
            for k in col..n {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= a[i][k] * x[k];
        }
        x[i] = if a[i][i].abs() < 1e-15 {
            0.0
        } else {
            s / a[i][i]
        };
    }
    x
}

/// A linear readout of the movement field, fitted by ridge regression on
/// standardised features, that predicts one number from a feature vector.
/// The ridge penalty is relative to the number of samples, so that a
/// penalty of one shrinks every unit-variance direction by half whatever
/// the record's length; penalties well above one make the readout a
/// correlation-weighted sum of the features, which is what many noisy
/// features and few epochs call for.
#[derive(Clone, Debug, PartialEq)]
pub struct Readout {
    mean: Vec<f64>,
    scale: Vec<f64>,
    weights: Vec<f64>,
    bias: f64,
}

impl Readout {
    /// Fit `targets ≈ w · features + b` with ridge penalty `ridge` (times
    /// the number of samples) on the standardised features.
    #[allow(clippy::needless_range_loop)]
    pub fn fit(features: &[Vec<f64>], targets: &[f64], ridge: f64) -> Readout {
        let n = features.len().min(targets.len());
        let p = features.first().map(|f| f.len()).unwrap_or(0);
        let mut mean = vec![0.0; p];
        let mut scale = vec![1.0; p];
        for j in 0..p {
            let m = features[..n].iter().map(|f| f[j]).sum::<f64>() / n.max(1) as f64;
            let v = features[..n]
                .iter()
                .map(|f| (f[j] - m).powi(2))
                .sum::<f64>()
                / n.max(1) as f64;
            mean[j] = m;
            scale[j] = if v > 1e-18 { v.sqrt() } else { 1.0 };
        }
        let target_mean = targets[..n].iter().sum::<f64>() / n.max(1) as f64;
        let z = |f: &Vec<f64>, j: usize| (f[j] - mean[j]) / scale[j];
        let mut gram = vec![vec![0.0; p]; p];
        let mut rhs = vec![0.0; p];
        for (f, &t) in features[..n].iter().zip(targets) {
            for j in 0..p {
                let zj = z(f, j);
                rhs[j] += zj * (t - target_mean);
                for k in j..p {
                    gram[j][k] += zj * z(f, k);
                }
            }
        }
        for j in 0..p {
            for k in 0..j {
                gram[j][k] = gram[k][j];
            }
            gram[j][j] += ridge.max(0.0) * n as f64;
        }
        let weights = if p == 0 { Vec::new() } else { solve(gram, rhs) };
        Readout {
            mean,
            scale,
            weights,
            bias: target_mean,
        }
    }

    /// Predict from a feature vector.
    pub fn predict(&self, features: &[f64]) -> f64 {
        self.bias
            + self
                .weights
                .iter()
                .zip(features)
                .zip(self.mean.iter().zip(&self.scale))
                .map(|((w, f), (m, s))| w * (f - m) / s)
                .sum::<f64>()
    }

    /// Coefficient of determination on a set (1 perfect, 0 no better than
    /// the mean, negative worse).
    pub fn r2(&self, features: &[Vec<f64>], targets: &[f64]) -> f64 {
        let n = features.len().min(targets.len());
        if n == 0 {
            return 0.0;
        }
        let mean = targets[..n].iter().sum::<f64>() / n as f64;
        let ss_tot: f64 = targets[..n].iter().map(|t| (t - mean).powi(2)).sum();
        let ss_res: f64 = features[..n]
            .iter()
            .zip(targets)
            .map(|(f, t)| (t - self.predict(f)).powi(2))
            .sum();
        if ss_tot <= 1e-18 {
            0.0
        } else {
            1.0 - ss_res / ss_tot
        }
    }
}

/// What the queen recorded at one epoch boundary: the thought that was in
/// force over the epoch, the input that made it, and the field as it stood
/// when the epoch ended.
#[derive(Clone, Debug, PartialEq)]
pub struct Epoch {
    /// Tick at which the epoch ended.
    pub tick: u64,
    /// External input to the thought.
    pub input: Vec<f64>,
    /// The thought in force.
    pub thought: Vec<f64>,
    /// Coarse invariant features at the end of the epoch.
    pub invariant: Vec<f64>,
    /// Coarse residual features at the end of the epoch.
    pub residual: Vec<f64>,
}

impl Epoch {
    /// The features of one component.
    pub fn features(&self, component: Component) -> Vec<f64> {
        match component {
            Component::Invariant => self.invariant.clone(),
            Component::Residual => self.residual.clone(),
            Component::Both => {
                let mut v = self.invariant.clone();
                v.extend_from_slice(&self.residual);
                v
            }
        }
    }
}

/// How much of the queen's past thought the field carries: the held-out
/// R² of a readout per lag (1 is the epoch just ended), averaged over the
/// thought's components, and their sum.
#[derive(Clone, Debug, PartialEq)]
pub struct Capacity {
    /// Which component of the field was read.
    pub component: Component,
    /// Held-out R² by lag, starting at lag 1.
    pub by_lag: Vec<f64>,
    /// Sum of the positive parts: the memory capacity.
    pub total: f64,
}

/// Fit readouts from the field to the thought `lag` epochs back, for lags
/// `1..=lags`, on the first `train_fraction` of the epochs, and score them
/// on the rest.
pub fn memory_capacity(
    epochs: &[Epoch],
    component: Component,
    lags: usize,
    ridge: f64,
    train_fraction: f64,
) -> Capacity {
    let dim = epochs.first().map(|e| e.thought.len()).unwrap_or(0);
    let mut by_lag = Vec::with_capacity(lags);
    for lag in 1..=lags {
        let mut r2_sum = 0.0;
        for j in 0..dim {
            let pairs: Vec<(Vec<f64>, f64)> = (lag - 1..epochs.len())
                .map(|t| {
                    (
                        epochs[t].features(component),
                        epochs[t + 1 - lag].thought[j],
                    )
                })
                .collect();
            let split = ((pairs.len() as f64) * train_fraction.clamp(0.1, 0.9)).round() as usize;
            let split = split.clamp(1, pairs.len().saturating_sub(1).max(1));
            let (train, test) = pairs.split_at(split.min(pairs.len()));
            if train.is_empty() || test.is_empty() {
                continue;
            }
            let (xf, yf): (Vec<Vec<f64>>, Vec<f64>) = train.iter().cloned().unzip();
            let (xt, yt): (Vec<Vec<f64>>, Vec<f64>) = test.iter().cloned().unzip();
            let readout = Readout::fit(&xf, &yf, ridge);
            r2_sum += readout.r2(&xt, &yt);
        }
        by_lag.push(if dim == 0 { 0.0 } else { r2_sum / dim as f64 });
    }
    let total = by_lag.iter().map(|r| r.max(0.0)).sum();
    Capacity {
        component,
        by_lag,
        total,
    }
}

/// Self-recursive inclusion: what the queen recalls from the field feeds
/// her next thought.
#[derive(Clone, Debug, PartialEq)]
pub struct Recursion {
    /// Which lag she recalls (1 is the thought of the epoch just ended).
    pub lag: usize,
    /// Weight of the recalled thought in the next one (negative for
    /// alternation).
    pub gain: f64,
}

/// How the queen thinks.
#[derive(Clone, Debug, PartialEq)]
pub struct QueenConfig {
    /// Ticks per epoch: her clock.
    pub epoch_ticks: u64,
    /// Ticks before her clock starts: she is silent while the colony
    /// settles into its foraging.
    pub warmup_ticks: u64,
    /// Nodes whose entropy dials express her thought, one component each.
    /// Empty means every node at depth 1 (the castes).
    pub targets: Vec<NodeId>,
    /// Size of the expression: a component `x` sets a relative gain of
    /// `exp(expression × x)` on its node's dial.
    pub expression: f64,
    /// External inputs per epoch, cycled; empty means none.
    pub script: Vec<Vec<f64>>,
    /// Whether, absent a script, each epoch's input is a random ±1 per
    /// component (the memory-capacity probe).
    pub random_input: bool,
    /// Recall from the field feeding the next thought, once readouts are
    /// fitted.
    pub recursion: Option<Recursion>,
    /// Which component of the field the recall reads: the non-invariant
    /// residual by default, which the slow drift of the skeleton cancels
    /// out of by construction.
    pub component: Component,
    /// Ridge penalty of the readouts, relative to the number of epochs.
    pub ridge: f64,
}

impl Default for QueenConfig {
    fn default() -> Self {
        QueenConfig {
            epoch_ticks: 60,
            warmup_ticks: 0,
            targets: Vec::new(),
            expression: 1.0,
            script: Vec::new(),
            random_input: true,
            recursion: None,
            component: Component::Residual,
            ridge: 3.0,
        }
    }
}

/// The queen as a cognitive agent: a slow recurrent state expressed in the
/// hierarchy's dials and read back from the colony's movement.
#[derive(Clone, Debug)]
pub struct Queen {
    cfg: QueenConfig,
    targets: Vec<NodeId>,
    /// The thought in force, one component per target, in `-1..=1`.
    pub thought: Vec<f64>,
    /// The epochs recorded so far.
    pub epochs: Vec<Epoch>,
    /// Fitted readouts, per thought component, per lag (index 0 is lag 1).
    readouts: Vec<Vec<Readout>>,
    next_epoch: u64,
    epoch_count: u64,
}

impl Queen {
    /// A queen for a hierarchy; targets default to the nodes at depth 1.
    pub fn new(cfg: QueenConfig, hierarchy: &Hierarchy) -> Queen {
        let targets = if cfg.targets.is_empty() {
            hierarchy.nodes_at_depth(1)
        } else {
            cfg.targets.clone()
        };
        let dim = targets.len();
        let next_epoch = cfg.warmup_ticks + cfg.epoch_ticks.max(1);
        Queen {
            cfg,
            targets,
            thought: vec![0.0; dim],
            epochs: Vec::new(),
            readouts: Vec::new(),
            next_epoch,
            epoch_count: 0,
        }
    }

    /// The nodes she expresses herself through.
    pub fn targets(&self) -> &[NodeId] {
        &self.targets
    }

    /// Her configuration.
    pub fn config(&self) -> &QueenConfig {
        &self.cfg
    }

    /// Whether an epoch ends at this tick.
    pub fn due(&self, tick: u64) -> bool {
        tick >= self.next_epoch
    }

    /// Epochs completed.
    pub fn epoch_count(&self) -> u64 {
        self.epoch_count
    }

    /// Whether readouts have been fitted.
    pub fn can_recall(&self) -> bool {
        !self.readouts.is_empty()
    }

    /// Switch recall-driven recursion on or off.
    pub fn set_recursion(&mut self, recursion: Option<Recursion>) {
        self.cfg.recursion = recursion;
    }

    /// Change the size of the expression (zero disconnects the dials: the
    /// thought goes on but is no longer written into the colony).
    pub fn set_expression(&mut self, expression: f64) {
        self.cfg.expression = expression;
    }

    /// Replace the external input: a script (cycled) and whether, absent
    /// one, inputs are random.
    pub fn set_input(&mut self, script: Vec<Vec<f64>>, random: bool) {
        self.cfg.script = script;
        self.cfg.random_input = random;
    }

    /// Recall the thought `lag` epochs back from the field as it stands.
    pub fn recall(&self, history: &MovementHistory, lag: usize) -> Option<Vec<f64>> {
        if lag == 0 || self.readouts.is_empty() {
            return None;
        }
        let features = history.features(self.cfg.component);
        let mut out = Vec::with_capacity(self.readouts.len());
        for per_lag in &self.readouts {
            let r = per_lag.get(lag - 1)?;
            out.push(r.predict(&features));
        }
        Some(out)
    }

    /// Fit the readouts on the recorded epochs for lags `1..=lags`, so that
    /// recall becomes possible; returns the memory capacity of the
    /// configured component on held-out epochs.
    pub fn fit(&mut self, lags: usize, train_fraction: f64) -> Capacity {
        let dim = self.thought.len();
        let mut readouts = Vec::with_capacity(dim);
        for j in 0..dim {
            let mut per_lag = Vec::with_capacity(lags);
            for lag in 1..=lags {
                let pairs: Vec<(Vec<f64>, f64)> = (lag - 1..self.epochs.len())
                    .map(|t| {
                        (
                            self.epochs[t].features(self.cfg.component),
                            self.epochs[t + 1 - lag].thought[j],
                        )
                    })
                    .collect();
                let (x, y): (Vec<Vec<f64>>, Vec<f64>) = pairs.into_iter().unzip();
                per_lag.push(Readout::fit(&x, &y, self.cfg.ridge));
            }
            readouts.push(per_lag);
        }
        self.readouts = readouts;
        memory_capacity(
            &self.epochs,
            self.cfg.component,
            lags,
            self.cfg.ridge,
            train_fraction,
        )
    }

    /// Close an epoch: record the field with the thought that was in
    /// force, form the next thought from the input and (with recursion)
    /// what the field recalls, clamped to `-1..=1`, and express it in the
    /// dials.
    pub fn epoch(
        &mut self,
        tick: u64,
        history: &MovementHistory,
        hierarchy: &mut Hierarchy,
        rng: &mut Rng,
    ) {
        let dim = self.thought.len();
        let input: Vec<f64> = if !self.cfg.script.is_empty() {
            let row = &self.cfg.script[(self.epoch_count as usize) % self.cfg.script.len()];
            (0..dim)
                .map(|j| row.get(j).copied().unwrap_or(0.0))
                .collect()
        } else if self.cfg.random_input {
            (0..dim)
                .map(|_| if rng.chance(0.5) { 1.0 } else { -1.0 })
                .collect()
        } else {
            vec![0.0; dim]
        };
        self.epochs.push(Epoch {
            tick,
            input: input.clone(),
            thought: self.thought.clone(),
            invariant: history.features(Component::Invariant),
            residual: history.features(Component::Residual),
        });
        let recalled = match &self.cfg.recursion {
            Some(r) => self
                .recall(history, r.lag)
                .map(|v| v.into_iter().map(|x| r.gain * x).collect::<Vec<f64>>()),
            None => None,
        };
        for j in 0..dim {
            let drive = input[j] + recalled.as_ref().map(|v| v[j]).unwrap_or(0.0);
            self.thought[j] = drive.clamp(-1.0, 1.0);
        }
        self.express(hierarchy);
        self.epoch_count += 1;
        self.next_epoch = tick + self.cfg.epoch_ticks.max(1);
    }

    /// Write the thought into the targets' dials: a temperature of
    /// `exp(expression × x)` at the root (whose biological default is a
    /// temperature of one), a relative gain of the same size below it.
    pub fn express(&self, hierarchy: &mut Hierarchy) {
        for (j, &node) in self.targets.iter().enumerate() {
            let factor = (self.cfg.expression * self.thought[j]).exp();
            let control = if hierarchy.node(node).depth == 0 {
                EntropyControl::fixed(factor)
            } else {
                EntropyControl::relative(factor)
            };
            hierarchy.node_mut(node).surface.entropy = control;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hierarchy::HierarchySpec;
    use std::f64::consts::PI;

    fn history() -> MovementHistory {
        MovementHistory::new(
            16,
            16,
            1.0,
            HistoryConfig {
                sector: 8,
                slow_half_life_s: 100.0,
                fast_half_life_s: 5.0,
                channel_rate: 0.01,
                channel_alignment: 0.3,
                multiscale: true,
            },
        )
    }

    #[test]
    fn flow_accumulates_and_forgets() {
        let mut f = Flow::default();
        f.add(1.0, 0.0, 1.0, 1.0);
        f.add(0.0, 1.0, 0.0, 1.0);
        assert!((f.straightness() - 0.5).abs() < 1e-12);
        assert!((f.alignment() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        let (mx, my) = f.mean();
        assert!((mx - 0.5).abs() < 1e-12 && (my - 0.5).abs() < 1e-12);
        f.scale_in_place(0.5);
        assert!((f.weight - 1.0).abs() < 1e-12);
        assert_eq!(Flow::default().alignment(), 0.0);
        // Two-way traffic along one line: no polar alignment, full axial order.
        let mut g = Flow::default();
        g.add(1.0, 0.0, 1.0, 1.0);
        g.add(-1.0, 0.0, 1.0, 1.0);
        assert!(g.alignment() < 1e-12 && (g.nematic() - 1.0).abs() < 1e-12);
        let mut every_way = Flow::default();
        for k in 0..8 {
            let a = k as f64 * std::f64::consts::FRAC_PI_4;
            every_way.add(a.cos(), a.sin(), a.cos(), 1.0);
        }
        assert!(every_way.nematic() < 1e-9);
    }

    #[test]
    fn history_maps_cells_to_sectors_and_normalises_to_rates() {
        let mut h = history();
        assert_eq!(h.sectors(), (2, 2));
        assert_eq!(h.sector_of(Position::new(9, 3)), Some(1));
        assert_eq!(h.sector_of(Position::new(3, 9)), Some(2));
        assert_eq!(h.sector_of(Position::new(16, 0)), None);
        // One eastward move per tick through the same cell settles to a
        // rate of one per tick in both accumulators.
        for _ in 0..2000 {
            h.record(Point::new(2.0, 2.5), Point::new(3.0, 2.5), 0.0);
            h.step();
        }
        let inv = h.fine_invariant(Position::new(3, 2));
        assert!(
            (inv.weight - 1.0).abs() < 0.02,
            "steady rate {}",
            inv.weight
        );
        assert!((inv.alignment() - 1.0).abs() < 1e-9);
        let cur = h.fine_current(Position::new(3, 2));
        assert!((cur.weight - 1.0).abs() < 0.02, "{}", cur.weight);
        let sector = h.invariant(0);
        assert!((sector.weight - 1.0).abs() < 0.02);
        let (dd, (rx, ry)) = h.residual(0);
        assert!(dd.abs() < 0.05 && rx.abs() < 1e-6 && ry.abs() < 1e-6);
        // Stop moving: the fast accumulator drops first.
        for _ in 0..20 {
            h.step();
        }
        let (dd, _) = h.residual(0);
        assert!(dd < -0.5, "the residual shows the flow has stopped: {dd}");
        // The pyramid: the whole field, then its four sectors.
        assert_eq!(h.blocks().len(), 5);
        assert_eq!(h.blocks()[0], (0, 0, 16, 16));
        assert_eq!(
            h.features(Component::Both).len(),
            2 * 5 * MovementHistory::READING
        );
        let reading = h.features(Component::Invariant);
        assert!(
            (reading[2] - 1.0).abs() < 1e-9,
            "one-way traffic is axially ordered"
        );
        assert!(
            (reading[3] - 1.0).abs() < 1e-9,
            "moves that hold their course are straight"
        );
        let whole = &reading[..MovementHistory::READING];
        let sector0 = &reading[MovementHistory::READING..2 * MovementHistory::READING];
        assert_eq!(whole, sector0, "all the flow is in sector 0");
        let mut flat = history();
        flat.cfg.multiscale = false;
        assert_eq!(flat.blocks().len(), 4);
        assert_eq!(h.moves, 2000);
    }

    #[test]
    fn topology_counts_channels_and_loops() {
        let mut h = history();
        // A square loop of eastward/southward/westward/northward moves.
        let corners = [(2.5, 2.5), (10.5, 2.5), (10.5, 10.5), (2.5, 10.5)];
        for _ in 0..300 {
            for k in 0..4 {
                let (ax, ay) = corners[k];
                let (bx, by) = corners[(k + 1) % 4];
                let steps = 8;
                for s in 0..steps {
                    let t0 = s as f64 / steps as f64;
                    let t1 = (s + 1) as f64 / steps as f64;
                    let from = Point::new(ax + (bx - ax) * t0, ay + (by - ay) * t0);
                    let to = Point::new(ax + (bx - ax) * t1, ay + (by - ay) * t1);
                    h.record(from, to, 0.0);
                }
            }
            h.step();
        }
        let t = h.topology();
        assert_eq!(t.channels, 1, "{t:?}");
        assert_eq!(t.loops, 1, "the square encloses one region: {t:?}");
        assert!(t.cells >= 28, "{t:?}");
        let s = h.summary();
        assert!(s.density > 0.0 && s.alignment > 0.5 && s.spatial_entropy > 0.0);
        let picture = h.render(Component::Invariant);
        assert_eq!(picture.lines().count(), 2);
        let skeleton = h.render_skeleton();
        assert_eq!(skeleton.lines().count(), 16);
        assert!(
            skeleton.contains('→') && skeleton.contains('↓'),
            "{skeleton}"
        );
        // Two-way traffic along one row renders as a line.
        let mut two_way = history();
        for _ in 0..300 {
            for x in 2..12 {
                let (a, b) = (
                    Point::new(x as f64 + 0.5, 4.5),
                    Point::new(x as f64 + 1.5, 4.5),
                );
                two_way.record(a, b, 0.0);
                two_way.record(b, a, 0.0);
            }
            two_way.step();
        }
        assert!(two_way
            .render_skeleton()
            .lines()
            .nth(4)
            .unwrap()
            .contains('─'));
        // A random walk builds no channel.
        let mut r = history();
        let mut rng = Rng::seed_from_u64(1);
        for _ in 0..3000 {
            let from = Point::new(rng.range(1.0, 15.0), rng.range(1.0, 15.0));
            let to = from.advanced(rng.range(-PI, PI), 0.5);
            r.record(from, to, rng.range(-PI, PI));
            r.step();
        }
        assert_eq!(r.topology().loops, 0);
    }

    #[test]
    fn readout_recovers_a_linear_map() {
        let mut rng = Rng::seed_from_u64(5);
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..200 {
            let f: Vec<f64> = (0..5).map(|_| rng.normal()).collect();
            let target = 2.0 * f[0] - f[3] + 0.5 + 0.01 * rng.normal();
            x.push(f);
            y.push(target);
        }
        let r = Readout::fit(&x[..150], &y[..150], 1e-3);
        assert!(r.r2(&x[150..], &y[150..]) > 0.99);
        assert!((r.predict(&[1.0, 0.0, 0.0, 0.0, 0.0]) - 2.5).abs() < 0.1);
        // Pure noise targets read out nothing on held-out data.
        let noise: Vec<f64> = (0..200).map(|_| rng.normal()).collect();
        let n = Readout::fit(&x[..150], &noise[..150], 1.0);
        assert!(n.r2(&x[150..], &noise[150..]) < 0.2);
    }

    #[test]
    fn memory_capacity_of_synthetic_epochs() {
        // A field that carries the last two thoughts exactly.
        let mut rng = Rng::seed_from_u64(9);
        let mut thoughts: Vec<f64> = Vec::new();
        let mut epochs = Vec::new();
        for t in 0..120u64 {
            let thought = if rng.chance(0.5) { 1.0 } else { -1.0 };
            thoughts.push(thought);
            let prev = if t >= 1 {
                thoughts[t as usize - 1]
            } else {
                0.0
            };
            epochs.push(Epoch {
                tick: t,
                input: vec![thought],
                thought: vec![thought],
                invariant: vec![0.3 * thought + 0.1 * rng.normal(), prev],
                residual: vec![rng.normal()],
            });
        }
        let cap = memory_capacity(&epochs, Component::Invariant, 4, 0.1, 0.7);
        assert!(cap.by_lag[0] > 0.8, "{cap:?}");
        assert!(cap.by_lag[1] > 0.8, "{cap:?}");
        assert!(cap.by_lag[3] < 0.3, "{cap:?}");
        assert!(cap.total > 1.6 && cap.total < 2.6, "{cap:?}");
        let noise = memory_capacity(&epochs, Component::Residual, 2, 0.1, 0.7);
        assert!(noise.total < 0.3, "{noise:?}");
    }

    #[test]
    fn queen_expresses_her_thought_in_the_caste_dials() {
        let mut hierarchy = Hierarchy::from_spec(
            &HierarchySpec::default(),
            crate::surface::BehavioralSurface::instinct(),
        );
        let h = history();
        let mut rng = Rng::seed_from_u64(2);
        let mut queen = Queen::new(
            QueenConfig {
                epoch_ticks: 10,
                script: vec![vec![1.0, -1.0, 0.0]],
                random_input: false,
                expression: 2.0,
                ..QueenConfig::default()
            },
            &hierarchy,
        );
        assert_eq!(queen.targets().len(), 3);
        assert!(!queen.due(5) && queen.due(10));
        queen.epoch(10, &h, &mut hierarchy, &mut rng);
        assert_eq!(queen.epoch_count(), 1);
        assert!(queen.due(20) && !queen.due(19));
        let t = &queen.thought;
        assert!((t[0] - 2.0f64.tanh() / 2.0f64.tanh() * 1.0f64.tanh()).abs() < 1e-12 || t[0] > 0.7);
        assert!(t[1] < -0.7 && t[2].abs() < 1e-12);
        let policies = hierarchy.compile();
        let castes = hierarchy.nodes_at_depth(1);
        let gain_of = |node: NodeId| hierarchy.node(node).surface.entropy.raw().exp();
        assert!(gain_of(castes[0]) > 1.0 && gain_of(castes[1]) < 1.0);
        assert!((gain_of(castes[2]) - 1.0).abs() < 1e-12);
        assert!(!policies.is_empty());
        assert_eq!(queen.epochs.len(), 1);
        assert!(queen.recall(&h, 1).is_none(), "nothing fitted yet");
    }
}
