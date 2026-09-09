//! The path-topological network: a variable pheromone surface whose
//! channels are learned symbols, each a class of routes.
//!
//! The ants' surface has six channels with fixed meanings. Here the
//! surface grows: a channel is registered for every *kind of route*
//! the thoughts keep walking, and the kind is a topological invariant
//! of the route. Given a set of **punctures** in the embedding (the
//! walls of a maze, the cities of a tour, and the holes the thoughts'
//! own walks turn out to enclose), a ray runs up from each; a route's
//! **word** is the sequence of rays it crosses, each crossing a letter
//! signed by its direction, freely reduced (a crossing followed by its
//! reverse cancels). Two routes with the same endpoints have the same
//! word if and only if one can be deformed into the other without
//! passing through a puncture: the word is the route's homotopy class,
//! its *h-signature* (Bhattacharya, Likhachev & Kumar 2012). It is a
//! label nobody gave: the pattern names itself.
//!
//! A word seen on enough successful trips is **registered** as a
//! symbol: a channel of its own in the symbol field, where the trips of
//! that class lay their trail; a representative route (its *glyph*); a
//! weight learned from how the class yields (quality per length,
//! against the other classes), which the thoughts sense as they sense a
//! pheromone. The symbols are the units of a network whose wiring is
//! the topology of the routes: an input (a route) activates the unit
//! whose class it is (**inference**, [`PathNet::label`]), and a unit
//! can be driven to produce a route of its class, either by a search
//! in the space of cells and words ([`PathNet::generate`]) or by
//! expressing its glyph into its channel and attending to it, so that
//! the thoughts walk the class again (**generation**,
//! [`PathNet::express`]). Punctures are learned as well: a hole the
//! invariant skeleton of the movement history encloses for several
//! epochs becomes a puncture, and every symbol's word is refined under
//! the new alphabet.

use ant_simulator::geometry::{Point, Position};
use ant_simulator::pheromone::perceived;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::fmt::Write as _;

/// A crossing of a ray: the puncture's number from one, negative when
/// the crossing runs from the ray's right to its left (falling `x`).
pub type Letter = i16;

/// A word: the freely reduced sequence of ray crossings of a route.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Word(Vec<Letter>);

impl Word {
    /// The empty word.
    pub fn new() -> Word {
        Word(Vec::new())
    }

    /// A word from letters, reduced.
    pub fn from_letters(letters: &[Letter]) -> Word {
        let mut w = Word::new();
        for &l in letters {
            w.push(l);
        }
        w
    }

    /// Append a letter, cancelling it against an inverse last letter.
    pub fn push(&mut self, letter: Letter) {
        if letter == 0 {
            return;
        }
        if self.0.last() == Some(&-letter) {
            self.0.pop();
        } else {
            self.0.push(letter);
        }
    }

    /// The letters.
    pub fn letters(&self) -> &[Letter] {
        &self.0
    }

    /// Number of letters.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the word is empty (the class of routes crossing nothing).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The word of the route walked backwards.
    pub fn inverse(&self) -> Word {
        Word(self.0.iter().rev().map(|l| -l).collect())
    }

    /// The word of one route followed by another.
    pub fn then(&self, other: &Word) -> Word {
        let mut w = self.clone();
        for &l in &other.0 {
            w.push(l);
        }
        w
    }

    /// Edit distance to another word, in letters.
    pub fn distance(&self, other: &Word) -> usize {
        let (a, b) = (&self.0, &other.0);
        let mut prev: Vec<usize> = (0..=b.len()).collect();
        let mut cur = vec![0; b.len() + 1];
        for (i, &x) in a.iter().enumerate() {
            cur[0] = i + 1;
            for (j, &y) in b.iter().enumerate() {
                let cost = if x == y { 0 } else { 1 };
                cur[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(cur[j] + 1);
            }
            std::mem::swap(&mut prev, &mut cur);
        }
        prev[b.len()]
    }
}

impl fmt::Display for Word {
    /// Letters `a`, `b`, `c`… for the punctures, primed when inverse;
    /// `1` for the empty word.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return write!(f, "1");
        }
        for (i, &l) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            let n = (l.unsigned_abs() - 1) as u32;
            if n < 26 {
                write!(f, "{}", char::from_u32('a' as u32 + n).unwrap_or('?'))?;
            } else {
                write!(f, "p{n}")?;
            }
            if l < 0 {
                write!(f, "'")?;
            }
        }
        Ok(())
    }
}

/// Rays running straight up (towards falling `y`) from punctures.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Rays {
    /// The punctures.
    pub punctures: Vec<Point>,
}

impl Rays {
    /// Rays from the given punctures.
    pub fn new(punctures: Vec<Point>) -> Rays {
        Rays { punctures }
    }

    /// Number of rays.
    pub fn len(&self) -> usize {
        self.punctures.len()
    }

    /// Whether there are no rays.
    pub fn is_empty(&self) -> bool {
        self.punctures.is_empty()
    }

    /// The letters of one step from `p` to `q`, in the order crossed,
    /// appended to a word.
    pub fn step(&self, p: Point, q: Point, into: &mut Word) {
        if (q.x - p.x).abs() < 1e-12 {
            return;
        }
        let mut crossings: Vec<(f64, Letter)> = Vec::new();
        for (i, c) in self.punctures.iter().enumerate() {
            let before = p.x < c.x;
            let after = q.x < c.x;
            if before == after {
                continue;
            }
            let t = (c.x - p.x) / (q.x - p.x);
            if !(0.0..=1.0).contains(&t) {
                continue;
            }
            let y = p.y + t * (q.y - p.y);
            if y > c.y {
                continue;
            }
            let sign = if q.x > p.x { 1 } else { -1 };
            crossings.push((t, sign * (i as Letter + 1)));
        }
        crossings.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (_, l) in crossings {
            into.push(l);
        }
    }

    /// The word of a route.
    pub fn word(&self, route: &[Point]) -> Word {
        let mut w = Word::new();
        for pair in route.windows(2) {
            self.step(pair[0], pair[1], &mut w);
        }
        w
    }
}

/// The variable surface: a field with as many channels as symbols.
#[derive(Clone, Debug)]
pub struct SymbolField {
    width: usize,
    height: usize,
    channels: Vec<Vec<f64>>,
    retention: f64,
    diffusion: f64,
    cap: f64,
    /// Half-saturation of perception.
    pub k: f64,
}

impl SymbolField {
    /// An empty field over a grid with the given kinetics.
    pub fn new(
        width: usize,
        height: usize,
        tick_s: f64,
        half_life_s: f64,
        diffusion_per_s: f64,
        cap: f64,
        k: f64,
    ) -> SymbolField {
        SymbolField {
            width,
            height,
            channels: Vec::new(),
            retention: if half_life_s > 0.0 {
                0.5f64.powf(tick_s / half_life_s)
            } else {
                0.0
            },
            diffusion: (diffusion_per_s * tick_s).clamp(0.0, 0.2),
            cap,
            k: k.max(1e-9),
        }
    }

    /// Number of channels.
    pub fn channels(&self) -> usize {
        self.channels.len()
    }

    /// Add an empty channel.
    pub fn add_channel(&mut self) -> usize {
        self.channels.push(vec![0.0; self.width * self.height]);
        self.channels.len() - 1
    }

    /// Empty a channel.
    pub fn clear(&mut self, k: usize) {
        if let Some(c) = self.channels.get_mut(k) {
            c.iter_mut().for_each(|v| *v = 0.0);
        }
    }

    fn index(&self, cell: Position) -> Option<usize> {
        if cell.x < 0
            || cell.y < 0
            || cell.x as usize >= self.width
            || cell.y as usize >= self.height
        {
            None
        } else {
            Some(cell.y as usize * self.width + cell.x as usize)
        }
    }

    /// The level of a channel in a cell.
    pub fn level(&self, cell: Position, k: usize) -> f64 {
        match (self.channels.get(k), self.index(cell)) {
            (Some(c), Some(i)) => c[i],
            _ => 0.0,
        }
    }

    /// The level of a channel at a point.
    pub fn level_at(&self, p: Point, k: usize) -> f64 {
        self.level(p.cell(), k)
    }

    /// Add to a channel in a cell, up to the cap.
    pub fn deposit(&mut self, cell: Position, k: usize, amount: f64) {
        let cap = self.cap;
        if let Some(i) = self.index(cell) {
            if let Some(c) = self.channels.get_mut(k) {
                c[i] = (c[i] + amount.max(0.0)).min(cap);
            }
        }
    }

    /// Lay a channel along a step, a patch per cell.
    pub fn lay(&mut self, k: usize, from: Point, to: Point, per_cell: f64) {
        if per_cell <= 0.0 {
            return;
        }
        let len = from.distance(to);
        let n = (len.ceil() as usize).clamp(1, 64);
        let amount = per_cell * len / n as f64;
        for s in 1..=n {
            let f = s as f64 / n as f64;
            let p = Point::new(from.x + (to.x - from.x) * f, from.y + (to.y - from.y) * f);
            self.deposit(p.cell(), k, amount);
        }
    }

    /// Lay a channel along a route.
    pub fn stamp(&mut self, k: usize, route: &[Point], per_cell: f64) {
        for pair in route.windows(2) {
            self.lay(k, pair[0], pair[1], per_cell);
        }
        if route.len() == 1 {
            self.deposit(route[0].cell(), k, per_cell);
        }
    }

    /// Read a channel along a move as the sensorium reads a pheromone
    /// (see [`crate::sense::read_along`]).
    pub fn read_along(&self, from: Point, unit: (f64, f64), len: f64, k: usize) -> f64 {
        let at = |d: f64| Point::new(from.x + d * unit.0, from.y + d * unit.1);
        if len <= 2.0 + 1e-9 {
            let mut c = self.level_at(at(len.min(1.0)), k);
            if len > 1.0 {
                c += 0.5 * self.level_at(at(len.min(2.0)), k);
            }
            return c;
        }
        let n = (len.ceil() as usize).clamp(2, 24);
        let mut total = 0.0;
        for i in 1..=n {
            total += self.level_at(at(len * i as f64 / n as f64), k);
        }
        1.5 * total / n as f64
    }

    /// The mass of a channel.
    pub fn total(&self, k: usize) -> f64 {
        self.channels.get(k).map(|c| c.iter().sum()).unwrap_or(0.0)
    }

    /// One tick: every channel evaporates and spreads a little.
    pub fn step(&mut self) {
        let (w, h) = (self.width, self.height);
        let (retention, diffusion) = (self.retention, self.diffusion);
        let mut scratch = vec![0.0; w * h];
        for c in self.channels.iter_mut() {
            if c.iter().all(|&v| v <= 1e-12) {
                continue;
            }
            if diffusion > 0.0 {
                scratch.iter_mut().for_each(|v| *v = 0.0);
                for y in 0..h {
                    for x in 0..w {
                        let i = y * w + x;
                        let v = c[i];
                        if v <= 0.0 {
                            continue;
                        }
                        let mut neighbours = 0;
                        let mut list = [0usize; 4];
                        if x > 0 {
                            list[neighbours] = i - 1;
                            neighbours += 1;
                        }
                        if x + 1 < w {
                            list[neighbours] = i + 1;
                            neighbours += 1;
                        }
                        if y > 0 {
                            list[neighbours] = i - w;
                            neighbours += 1;
                        }
                        if y + 1 < h {
                            list[neighbours] = i + w;
                            neighbours += 1;
                        }
                        let share = v * diffusion / 4.0;
                        scratch[i] += v - share * neighbours as f64;
                        for &j in &list[..neighbours] {
                            scratch[j] += share;
                        }
                    }
                }
                c.copy_from_slice(&scratch);
            }
            for v in c.iter_mut() {
                *v *= retention;
                if *v < 1e-9 {
                    *v = 0.0;
                }
            }
        }
    }
}

/// A symbol: a class of routes, with its channel, its statistics, its
/// weight, its glyph and, if someone gave it one, its name.
#[derive(Clone, Debug, PartialEq)]
pub struct Symbol {
    /// The class: the word of its routes.
    pub word: Word,
    /// Its channel in the symbol field.
    pub channel: usize,
    /// Trips of the class brought home.
    pub support: u32,
    /// Mean quality of those trips (exponentially weighted).
    pub quality: f64,
    /// Mean length of their routes, in cells (exponentially weighted).
    pub length: f64,
    /// Quality per length against the shortest class: how the class
    /// yields.
    pub yield_: f64,
    /// The learned weight the thoughts sense the channel with: logits
    /// per unit of perceived level, from the yield against the others.
    pub weight: f64,
    /// A representative route: the best brought home.
    pub glyph: Vec<Point>,
    /// The glyph's quality.
    pub glyph_quality: f64,
    /// The last few routes of the class brought home, which a candidate
    /// puncture is tested against: it is learned only where routes of
    /// one class go round it both ways.
    pub exemplars: Vec<Vec<Point>>,
    /// A name given after the fact.
    pub name: Option<String>,
    /// Tick registered.
    pub born: u64,
    /// Tick last seen.
    pub last_seen: u64,
    /// Whether the symbol still stands (false once merged or evicted).
    pub alive: bool,
    /// The symbol this one was merged into when the alphabet grew.
    pub merged_into: Option<usize>,
}

/// What the network says of a route.
#[derive(Clone, Debug, PartialEq)]
pub struct Label {
    /// The route's word.
    pub word: Word,
    /// The symbol of exactly that class, if registered.
    pub symbol: Option<usize>,
    /// The nearest symbol by edit distance, and the distance.
    pub nearest: Option<(usize, usize)>,
    /// The name of the symbol (or of the nearest), if any.
    pub name: Option<String>,
}

/// Configuration of the network.
#[derive(Clone, Debug, PartialEq)]
pub struct SymbolConfig {
    /// Trips of a class brought home before it is registered.
    pub support: u32,
    /// Most symbols kept; beyond it the least supported unnamed symbol
    /// gives way.
    pub capacity: usize,
    /// Half-life of the symbol channels, seconds.
    pub half_life_s: f64,
    /// Diffusion of the symbol channels per second.
    pub diffusion_per_s: f64,
    /// Cap of a channel in a cell.
    pub cap: f64,
    /// Half-saturation of perception.
    pub k: f64,
    /// Laid per cell into a class's channel by a trip of quality 1.
    pub deposit: f64,
    /// Logits per unit of relative yield: a class yielding twice the
    /// mean is sensed with this weight.
    pub gain: f64,
    /// Rate of the exponentially weighted means.
    pub rate: f64,
    /// Longest word kept; a route with a longer word is not a class.
    pub max_word: usize,
    /// Routes kept per symbol as exemplars.
    pub exemplars: usize,
    /// How much stronger than a trip's deposit a symbol is laid when it
    /// is expressed.
    pub expression: f64,
    /// Whether punctures are learned from the holes the walks enclose.
    pub learn_punctures: bool,
    /// The steady flow, in moves per tick, that counts as walked when
    /// the holes are looked for: low, so that a way walked once and
    /// left is still remembered to have enclosed what it went round.
    pub hole_rate: f64,
    /// Smallest hole, in cells, that counts: the braids of a busy trail
    /// enclose small holes that are not the shape of the ground.
    pub min_hole_area: usize,
    /// Epochs a hole must persist to become a puncture.
    pub hole_persistence: u32,
    /// Ticks between looks at the skeleton.
    pub epoch_ticks: u64,
}

impl Default for SymbolConfig {
    fn default() -> Self {
        SymbolConfig {
            support: 3,
            capacity: 32,
            half_life_s: 600.0,
            diffusion_per_s: 0.0005,
            cap: 4000.0,
            k: 20.0,
            deposit: 20.0,
            gain: 3.0,
            rate: 0.1,
            max_word: 12,
            exemplars: 8,
            expression: 10.0,
            learn_punctures: true,
            hole_rate: 0.005,
            min_hole_area: 16,
            hole_persistence: 3,
            epoch_ticks: 500,
        }
    }
}

#[derive(Clone, Debug)]
struct Pending {
    count: u32,
    best: Vec<Point>,
    quality: f64,
    length: f64,
}

/// The network.
#[derive(Clone, Debug)]
pub struct PathNet {
    cfg: SymbolConfig,
    rays: Rays,
    given: usize,
    symbols: Vec<Symbol>,
    field: SymbolField,
    free_channels: Vec<usize>,
    pending: HashMap<Word, Pending>,
    holes: HashMap<(i32, i32), (u32, Point)>,
    attention: Vec<f64>,
    mean_yield: f64,
    width: usize,
    height: usize,
    lcg: u64,
    /// Routes observed.
    pub observations: u64,
    /// Symbols registered over time.
    pub registered: u64,
    /// Symbols merged when the alphabet grew.
    pub merged: u64,
    /// Routes whose word was too long to be a class.
    pub dropped: u64,
    /// Symbols evicted for room.
    pub evicted: u64,
}

impl PathNet {
    /// A network over a grid, starting from the given punctures.
    pub fn new(
        width: usize,
        height: usize,
        tick_s: f64,
        punctures: Vec<Point>,
        cfg: SymbolConfig,
    ) -> PathNet {
        let field = SymbolField::new(
            width,
            height,
            tick_s,
            cfg.half_life_s,
            cfg.diffusion_per_s,
            cfg.cap,
            cfg.k,
        );
        PathNet {
            cfg,
            given: punctures.len(),
            rays: Rays::new(punctures),
            symbols: Vec::new(),
            field,
            free_channels: Vec::new(),
            pending: HashMap::new(),
            holes: HashMap::new(),
            attention: Vec::new(),
            mean_yield: 0.0,
            width,
            height,
            lcg: 0x9E37_79B9_7F4A_7C15,
            observations: 0,
            registered: 0,
            merged: 0,
            dropped: 0,
            evicted: 0,
        }
    }

    /// The configuration.
    pub fn config(&self) -> &SymbolConfig {
        &self.cfg
    }

    /// The rays.
    pub fn rays(&self) -> &Rays {
        &self.rays
    }

    /// The punctures: the given ones first, then those learned.
    pub fn punctures(&self) -> &[Point] {
        &self.rays.punctures
    }

    /// How many punctures were learned.
    pub fn learned_punctures(&self) -> usize {
        self.rays.punctures.len() - self.given
    }

    /// The symbols, dead ones included (indices are stable).
    pub fn symbols(&self) -> &[Symbol] {
        &self.symbols
    }

    /// The living symbols' indices.
    pub fn living(&self) -> Vec<usize> {
        (0..self.symbols.len())
            .filter(|&k| self.symbols[k].alive)
            .collect()
    }

    /// A symbol.
    pub fn symbol(&self, k: usize) -> Option<&Symbol> {
        self.symbols.get(k)
    }

    /// The field.
    pub fn field(&self) -> &SymbolField {
        &self.field
    }

    /// The word of a route.
    pub fn word(&self, route: &[Point]) -> Word {
        self.rays.word(route)
    }

    /// The living symbol of a word.
    pub fn find(&self, word: &Word) -> Option<usize> {
        self.symbols.iter().position(|s| s.alive && s.word == *word)
    }

    /// The mean yield over the classes.
    pub fn mean_yield(&self) -> f64 {
        self.mean_yield
    }

    /// The holes the skeleton encloses at present, each with the epochs
    /// it has persisted.
    pub fn holes(&self) -> Vec<(Point, u32)> {
        let mut out: Vec<(Point, u32)> = self.holes.values().map(|(n, p)| (*p, *n)).collect();
        out.sort_by(|a, b| b.1.cmp(&a.1));
        out
    }

    /// A route brought home with a quality: its class is found or,
    /// with enough support, registered; the class's statistics, glyph
    /// and every class's weight are updated. Returns the class.
    pub fn observe(&mut self, route: &[Point], quality: f64, tick: u64) -> Option<usize> {
        self.observations += 1;
        if route.len() < 2 {
            return None;
        }
        let word = self.rays.word(route);
        if word.len() > self.cfg.max_word {
            self.dropped += 1;
            return None;
        }
        let length: f64 = route.windows(2).map(|w| w[0].distance(w[1])).sum();
        let rate = self.cfg.rate.clamp(0.0, 1.0);
        let k = match self.find(&word) {
            Some(k) => {
                let s = &mut self.symbols[k];
                s.support += 1;
                s.quality += rate * (quality - s.quality);
                s.length += rate * (length - s.length);
                s.last_seen = tick;
                let better = quality > s.glyph_quality + 1e-9
                    || ((quality - s.glyph_quality).abs() <= 1e-9
                        && length < s.glyph.windows(2).map(|w| w[0].distance(w[1])).sum::<f64>());
                if better {
                    s.glyph = route.to_vec();
                    s.glyph_quality = quality;
                }
                // The exemplars are a uniform sample of the class's routes
                // over its life (a reservoir), so that a way it went early
                // is still on record when a puncture is tested.
                let keep = self.cfg.exemplars.max(1);
                if s.exemplars.len() < keep {
                    s.exemplars.push(route.to_vec());
                } else {
                    self.lcg = self
                        .lcg
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let slot = (self.lcg >> 33) as usize % s.support.max(1) as usize;
                    if slot < keep {
                        s.exemplars[slot] = route.to_vec();
                    }
                }
                k
            }
            None => {
                let entry = self.pending.entry(word.clone()).or_insert(Pending {
                    count: 0,
                    best: route.to_vec(),
                    quality,
                    length,
                });
                entry.count += 1;
                if quality > entry.quality || (quality == entry.quality && length < entry.length) {
                    entry.best = route.to_vec();
                    entry.quality = quality;
                    entry.length = length;
                }
                if entry.count < self.cfg.support {
                    return None;
                }
                let pending = self.pending.remove(&word).expect("just seen");
                self.register(word, pending, tick)?
            }
        };
        self.reweigh();
        Some(k)
    }

    fn register(&mut self, word: Word, pending: Pending, tick: u64) -> Option<usize> {
        if self.living().len() >= self.cfg.capacity.max(1) && !self.evict(tick) {
            return None;
        }
        let channel = match self.free_channels.pop() {
            Some(c) => {
                self.field.clear(c);
                c
            }
            None => self.field.add_channel(),
        };
        let deposit = self.cfg.deposit * pending.quality;
        self.field.stamp(channel, &pending.best, deposit);
        self.symbols.push(Symbol {
            word,
            channel,
            support: pending.count,
            quality: pending.quality,
            length: pending.length,
            yield_: 0.0,
            weight: 0.0,
            glyph: pending.best.clone(),
            glyph_quality: pending.quality,
            exemplars: vec![pending.best],
            name: None,
            born: tick,
            last_seen: tick,
            alive: true,
            merged_into: None,
        });
        self.attention.push(0.0);
        self.registered += 1;
        Some(self.symbols.len() - 1)
    }

    /// Make room: the least supported unnamed symbol, oldest seen among
    /// equals, gives way. Returns whether room was made.
    fn evict(&mut self, _tick: u64) -> bool {
        let victim = self
            .living()
            .into_iter()
            .filter(|&k| self.symbols[k].name.is_none())
            .min_by_key(|&k| (self.symbols[k].support, self.symbols[k].last_seen));
        let Some(k) = victim else {
            return false;
        };
        self.symbols[k].alive = false;
        self.field.clear(self.symbols[k].channel);
        self.free_channels.push(self.symbols[k].channel);
        self.attention[k] = 0.0;
        self.evicted += 1;
        true
    }

    /// Every class's yield against the shortest class, and its weight
    /// against the mean yield.
    fn reweigh(&mut self) {
        let living = self.living();
        let reference = living
            .iter()
            .map(|&k| self.symbols[k].length)
            .fold(f64::INFINITY, f64::min)
            .max(1e-9);
        if !reference.is_finite() {
            return;
        }
        let mut total = 0.0;
        for &k in &living {
            let s = &mut self.symbols[k];
            s.yield_ = s.quality * reference / s.length.max(1e-9);
            total += s.yield_;
        }
        let mean = total / living.len().max(1) as f64;
        self.mean_yield = mean;
        let gain = self.cfg.gain;
        for &k in &living {
            let s = &mut self.symbols[k];
            s.weight = if mean > 1e-12 {
                (gain * (s.yield_ / mean - 1.0)).clamp(-gain, gain)
            } else {
                0.0
            };
        }
    }

    /// Lay a class's channel along a step.
    pub fn lay(&mut self, k: usize, from: Point, to: Point, per_cell: f64) {
        if let Some(s) = self.symbols.get(k) {
            if s.alive {
                self.field.lay(s.channel, from, to, per_cell);
            }
        }
    }

    /// What the symbols say of a move: every living class's channel
    /// read along the move, perceived, times its weight and the
    /// attention on it, summed.
    pub fn extra(&self, from: Point, unit: (f64, f64), len: f64) -> f64 {
        let mut total = 0.0;
        for (k, s) in self.symbols.iter().enumerate() {
            if !s.alive {
                continue;
            }
            let w = s.weight + self.attention[k];
            if w.abs() < 1e-9 {
                continue;
            }
            let c = self.field.read_along(from, unit, len, s.channel);
            if c > 0.0 {
                total += w * perceived(c, self.field.k);
            }
        }
        total
    }

    /// What class a route is.
    pub fn label(&self, route: &[Point]) -> Label {
        let word = self.rays.word(route);
        let symbol = self.find(&word);
        let nearest = self
            .living()
            .into_iter()
            .map(|k| (k, self.symbols[k].word.distance(&word)))
            .min_by_key(|&(k, d)| (d, std::cmp::Reverse(self.symbols[k].support)));
        let name = symbol
            .or(nearest.map(|(k, _)| k))
            .and_then(|k| self.symbols[k].name.clone());
        Label {
            word,
            symbol,
            nearest,
            name,
        }
    }

    /// Give a symbol a name.
    pub fn name(&mut self, k: usize, name: &str) {
        if let Some(s) = self.symbols.get_mut(k) {
            s.name = Some(name.to_string());
        }
    }

    /// Attend to a symbol: an extra weight on its channel until
    /// released.
    pub fn attend(&mut self, k: usize, gain: f64) {
        if let Some(a) = self.attention.get_mut(k) {
            *a = gain;
        }
    }

    /// Let a symbol go.
    pub fn release(&mut self, k: usize) {
        self.attend(k, 0.0);
    }

    /// The symbol most attended to, if any, and the heading its glyph
    /// sets out on: a thought that departs with a symbol in mind sets
    /// out that way.
    pub fn attended(&self) -> Option<(usize, f64)> {
        let (k, _) = self
            .attention
            .iter()
            .enumerate()
            .filter(|(k, a)| **a > 1e-9 && self.symbols[*k].alive)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;
        let glyph = &self.symbols[k].glyph;
        let first = glyph.iter().find(|p| p.distance(glyph[0]) > 0.5)?;
        let (dx, dy) = glyph[0].to(*first);
        Some((k, dy.atan2(dx)))
    }

    /// Express a symbol: lay its glyph into its channel, `per_cell` per
    /// cell (a mind lays a trip's deposit times the configured
    /// expression), so that the thoughts find its class laid out before
    /// them.
    pub fn express(&mut self, k: usize, per_cell: f64) {
        if let Some(s) = self.symbols.get(k) {
            if s.alive {
                let (channel, glyph) = (s.channel, s.glyph.clone());
                self.field.stamp(channel, &glyph, per_cell);
            }
        }
    }

    /// A route of a class from one cell to another, or none within the
    /// longest word: a breadth-first search over cells and the words
    /// of the way to them, so that the route found is a shortest one
    /// of that class in steps.
    pub fn generate(
        &self,
        word: &Word,
        from: Position,
        to: Position,
        passable: &dyn Fn(Position) -> bool,
    ) -> Option<Vec<Point>> {
        if !passable(from) || !passable(to) {
            return None;
        }
        let max_word = self.cfg.max_word.max(word.len());
        let start = (from, Word::new());
        let mut parent: HashMap<(Position, Word), (Position, Word)> = HashMap::new();
        let mut seen: HashSet<(Position, Word)> = HashSet::new();
        let mut queue = VecDeque::new();
        seen.insert(start.clone());
        queue.push_back(start.clone());
        let limit = self.width * self.height * 64;
        let mut visited = 0usize;
        while let Some(state) = queue.pop_front() {
            visited += 1;
            if visited > limit {
                return None;
            }
            if state.0 == to && state.1 == *word {
                let mut path = vec![Point::center_of(state.0)];
                let mut cur = state;
                while let Some(p) = parent.get(&cur) {
                    path.push(Point::center_of(p.0));
                    cur = p.clone();
                }
                path.reverse();
                return Some(path);
            }
            let (cell, w) = state;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let next = Position::new(cell.x + dx, cell.y + dy);
                    if !passable(next) {
                        continue;
                    }
                    if dx != 0
                        && dy != 0
                        && !(passable(Position::new(cell.x + dx, cell.y))
                            && passable(Position::new(cell.x, cell.y + dy)))
                    {
                        continue;
                    }
                    let mut nw = w.clone();
                    self.rays
                        .step(Point::center_of(cell), Point::center_of(next), &mut nw);
                    if nw.len() > max_word {
                        continue;
                    }
                    let key = (next, nw);
                    if seen.insert(key.clone()) {
                        parent.insert(key.clone(), (cell, w.clone()));
                        queue.push_back(key);
                    }
                }
            }
        }
        None
    }

    /// One tick: the field evaporates and spreads.
    pub fn step(&mut self) {
        self.field.step();
    }

    /// Look at where the walks steadily go (a mask of cells, row by
    /// row, such as the history's [`flow_mask`] at the configured
    /// `hole_rate`): a hole they enclose, seen for enough epochs and
    /// telling apart routes now of one class, becomes a puncture, and
    /// every symbol's word is refined under the new alphabet. Returns
    /// how many punctures were learned.
    ///
    /// [`flow_mask`]: ant_simulator::hive::MovementHistory::flow_mask
    pub fn learn_punctures(&mut self, channel: &[bool]) -> usize {
        if !self.cfg.learn_punctures || channel.len() != self.width * self.height {
            return 0;
        }
        let (w, h) = (self.width as i32, self.height as i32);
        let idx = |x: i32, y: i32| (y as usize) * self.width + x as usize;
        let mut seen = vec![false; channel.len()];
        let mut found: Vec<Point> = Vec::new();
        for start in 0..channel.len() {
            if channel[start] || seen[start] {
                continue;
            }
            let mut enclosed = true;
            let mut cells: Vec<usize> = Vec::new();
            let mut stack = vec![start];
            seen[start] = true;
            while let Some(i) = stack.pop() {
                cells.push(i);
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
            if enclosed && cells.len() >= self.cfg.min_hole_area.max(1) {
                let n = cells.len() as f64;
                let (sx, sy) = cells.iter().fold((0.0, 0.0), |(sx, sy), &i| {
                    (
                        sx + (i % self.width) as f64 + 0.5,
                        sy + (i / self.width) as f64 + 0.5,
                    )
                });
                found.push(Point::new(sx / n, sy / n));
            }
        }
        // Persistence: a hole counts where it keeps being. A skeleton
        // flickers, so a hole missed once is not forgotten but counted
        // down, and dropped when it has been missed as often as seen.
        let mut current: HashSet<(i32, i32)> = HashSet::new();
        for p in &found {
            let key = ((p.x / 4.0).floor() as i32, (p.y / 4.0).floor() as i32);
            current.insert(key);
            let e = self.holes.entry(key).or_insert((0, *p));
            e.0 += 1;
            e.1 = *p;
        }
        for (key, e) in self.holes.iter_mut() {
            if !current.contains(key) {
                e.0 = e.0.saturating_sub(1);
            }
        }
        self.holes.retain(|_, e| e.0 > 0);
        let mut added = 0;
        let candidates: Vec<Point> = self
            .holes
            .values()
            .filter(|(count, _)| *count >= self.cfg.hole_persistence.max(1))
            .map(|(_, p)| *p)
            .collect();
        for p in candidates {
            let near = self.rays.punctures.iter().any(|q| q.distance(p) < 3.0);
            if near {
                continue;
            }
            // Off any cell centre, so no route point lies on the ray.
            let puncture = Point::new(p.x.floor() + 0.25, p.y.floor() + 0.25);
            if self.discriminates(puncture) {
                self.rays.punctures.push(puncture);
                added += 1;
            }
        }
        if added > 0 {
            self.refine();
        }
        added
    }

    /// Whether a puncture would tell apart routes now of one class: the
    /// exemplars of some living symbol fall into at least two words
    /// under the rays with it, each word with at least two exemplars.
    fn discriminates(&self, puncture: Point) -> bool {
        let mut trial = self.rays.clone();
        trial.punctures.push(puncture);
        for s in self.symbols.iter().filter(|s| s.alive) {
            if s.exemplars.len() < 4 {
                continue;
            }
            // Exemplars of one word now, by their word with the puncture.
            let mut groups: HashMap<Word, HashMap<Word, usize>> = HashMap::new();
            for route in &s.exemplars {
                let now = self.rays.word(route);
                let then = trial.word(route);
                *groups.entry(now).or_default().entry(then).or_insert(0) += 1;
            }
            if groups
                .values()
                .any(|split| split.values().filter(|&&n| n >= 2).count() >= 2)
            {
                return true;
            }
        }
        false
    }

    /// Recompute every symbol's word under the current alphabet,
    /// merging those that coincide (the earlier keeps the channel).
    pub fn refine(&mut self) {
        self.pending.clear();
        let mut by_word: HashMap<Word, usize> = HashMap::new();
        for k in 0..self.symbols.len() {
            if !self.symbols[k].alive {
                continue;
            }
            let word = self.rays.word(&self.symbols[k].glyph);
            match by_word.get(&word) {
                Some(&into) => {
                    let (support, quality, length, glyph, glyph_quality, name, channel) = {
                        let s = &self.symbols[k];
                        (
                            s.support,
                            s.quality,
                            s.length,
                            s.glyph.clone(),
                            s.glyph_quality,
                            s.name.clone(),
                            s.channel,
                        )
                    };
                    let target = &mut self.symbols[into];
                    let total = (target.support + support).max(1) as f64;
                    target.quality =
                        (target.quality * target.support as f64 + quality * support as f64) / total;
                    target.length =
                        (target.length * target.support as f64 + length * support as f64) / total;
                    target.support += support;
                    if glyph_quality > target.glyph_quality {
                        target.glyph = glyph;
                        target.glyph_quality = glyph_quality;
                    }
                    if target.name.is_none() {
                        target.name = name;
                    }
                    let exemplars = self.symbols[k].exemplars.clone();
                    let target = &mut self.symbols[into];
                    target.exemplars.extend(exemplars);
                    let keep = self.cfg.exemplars.max(1);
                    if target.exemplars.len() > keep {
                        let drop = target.exemplars.len() - keep;
                        target.exemplars.drain(..drop);
                    }
                    let s = &mut self.symbols[k];
                    s.alive = false;
                    s.merged_into = Some(into);
                    self.field.clear(channel);
                    self.free_channels.push(channel);
                    self.attention[k] = 0.0;
                    self.merged += 1;
                }
                None => {
                    self.symbols[k].word = word.clone();
                    by_word.insert(word, k);
                }
            }
        }
        // An exemplar whose word is no longer its symbol's belongs to
        // another class now.
        for k in 0..self.symbols.len() {
            if !self.symbols[k].alive {
                continue;
            }
            let word = self.symbols[k].word.clone();
            let rays = &self.rays;
            self.symbols[k]
                .exemplars
                .retain(|route| rays.word(route) == word);
        }
        self.reweigh();
    }

    /// A summary of the network.
    pub fn report(&self) -> String {
        let mut out = String::new();
        let living = self.living();
        let _ = writeln!(
            out,
            "  symbols: {} living of {} registered ({} merged, {} evicted), {} punctures ({} learned), {} routes seen ({} with words too long), mean yield {:.3}",
            living.len(),
            self.registered,
            self.merged,
            self.evicted,
            self.rays.len(),
            self.learned_punctures(),
            self.observations,
            self.dropped,
            self.mean_yield
        );
        let mut order = living;
        order.sort_by(|&a, &b| self.symbols[b].support.cmp(&self.symbols[a].support));
        for k in order.into_iter().take(12) {
            let s = &self.symbols[k];
            let _ = writeln!(
                out,
                "    #{k} [{}]{} support {} quality {:.3} length {:.0} yield {:.3} weight {:+.2}{}",
                s.word,
                s.name
                    .as_ref()
                    .map(|n| format!(" \"{n}\""))
                    .unwrap_or_default(),
                s.support,
                s.quality,
                s.length,
                s.yield_,
                s.weight,
                if self.attention[k].abs() > 1e-9 {
                    format!(" attended {:+.1}", self.attention[k])
                } else {
                    String::new()
                }
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_reduce_invert_and_measure() {
        let mut w = Word::new();
        w.push(1);
        w.push(2);
        w.push(-2);
        assert_eq!(w.letters(), &[1]);
        assert_eq!(w.to_string(), "a");
        let v = Word::from_letters(&[1, -3, 2]);
        assert_eq!(v.to_string(), "a c' b");
        assert_eq!(v.inverse().to_string(), "b' c a'");
        assert!(v.then(&v.inverse()).is_empty());
        assert_eq!(Word::new().to_string(), "1");
        assert_eq!(v.distance(&Word::from_letters(&[1, 2])), 1);
        assert_eq!(v.distance(&v), 0);
    }

    #[test]
    fn a_route_over_a_puncture_crosses_its_ray_and_one_under_does_not() {
        let rays = Rays::new(vec![Point::new(10.5, 10.0)]);
        let over = vec![Point::new(2.5, 4.5), Point::new(18.5, 4.5)];
        assert_eq!(rays.word(&over).to_string(), "a");
        let back = vec![Point::new(18.5, 4.5), Point::new(2.5, 4.5)];
        assert_eq!(rays.word(&back).to_string(), "a'");
        let under = vec![Point::new(2.5, 14.5), Point::new(18.5, 14.5)];
        assert!(rays.word(&under).is_empty());
        let there_and_back = vec![
            Point::new(2.5, 4.5),
            Point::new(18.5, 4.5),
            Point::new(2.5, 4.5),
        ];
        assert!(rays.word(&there_and_back).is_empty(), "cancels");
        let two = Rays::new(vec![Point::new(5.5, 10.0), Point::new(15.5, 10.0)]);
        let across = vec![Point::new(0.5, 2.5), Point::new(20.5, 2.5)];
        assert_eq!(two.word(&across).to_string(), "a b");
    }

    #[test]
    fn the_field_keeps_channels_apart_and_forgets() {
        let mut f = SymbolField::new(8, 8, 1.0, 1.0, 0.0, 100.0, 1.0);
        let a = f.add_channel();
        let b = f.add_channel();
        f.deposit(Position::new(2, 2), a, 10.0);
        assert_eq!(f.level(Position::new(2, 2), a), 10.0);
        assert_eq!(f.level(Position::new(2, 2), b), 0.0);
        f.step();
        assert!(
            (f.level(Position::new(2, 2), a) - 5.0).abs() < 1e-9,
            "half-life one tick"
        );
        f.stamp(b, &[Point::new(0.5, 0.5), Point::new(4.5, 0.5)], 1.0);
        assert!(f.total(b) > 3.0);
    }

    #[test]
    fn symbols_register_with_support_and_are_labelled_and_generated() {
        let cfg = SymbolConfig {
            support: 2,
            ..SymbolConfig::default()
        };
        let mut net = PathNet::new(24, 16, 1.0, vec![Point::new(12.0, 8.0)], cfg);
        let over = vec![
            Point::new(2.5, 2.5),
            Point::new(12.5, 1.5),
            Point::new(21.5, 2.5),
        ];
        let under = vec![
            Point::new(2.5, 13.5),
            Point::new(12.5, 14.5),
            Point::new(21.5, 13.5),
        ];
        assert_eq!(net.observe(&over, 1.0, 1), None, "one trip is not a class");
        assert_eq!(net.observe(&over, 1.0, 2), Some(0));
        assert_eq!(net.observe(&under, 1.0, 3), None);
        assert_eq!(net.observe(&under, 1.0, 4), Some(1));
        assert_eq!(net.symbols()[0].word.to_string(), "a");
        assert!(net.symbols()[1].word.is_empty());
        net.name(0, "over");
        let label = net.label(&[Point::new(1.5, 3.5), Point::new(22.5, 3.5)]);
        assert_eq!(label.symbol, Some(0));
        assert_eq!(label.name.as_deref(), Some("over"));
        let label = net.label(&[Point::new(1.5, 12.5), Point::new(22.5, 12.5)]);
        assert_eq!(label.symbol, Some(1));
        // The over class is the shorter one here: it weighs more.
        assert!(net.symbols()[0].weight >= net.symbols()[1].weight);
        assert!(
            net.field().total(net.symbols()[0].channel) > 0.0,
            "the glyph is laid"
        );
        // Generation: a route of each class, checked by its word.
        let passable = |p: Position| {
            p.x >= 0 && p.y >= 0 && p.x < 24 && p.y < 16 && !(p.x == 12 && (5..=10).contains(&p.y))
        };
        let from = Position::new(1, 8);
        let to = Position::new(22, 8);
        let a = net
            .generate(&Word::from_letters(&[1]), from, to, &passable)
            .expect("a route over");
        assert_eq!(net.word(&a).to_string(), "a");
        assert!(a.iter().all(|p| passable(p.cell())));
        let b = net
            .generate(&Word::new(), from, to, &passable)
            .expect("a route under");
        assert!(net.word(&b).is_empty());
        assert_eq!(a[0].cell(), from);
        assert_eq!(b.last().unwrap().cell(), to);
    }

    #[test]
    fn holes_the_skeleton_encloses_become_punctures_and_words_are_refined() {
        let cfg = SymbolConfig {
            support: 1,
            hole_persistence: 1,
            min_hole_area: 1,
            ..SymbolConfig::default()
        };
        let mut net = PathNet::new(16, 16, 1.0, Vec::new(), cfg);
        let over = vec![Point::new(2.5, 2.5), Point::new(13.5, 2.5)];
        let under = vec![Point::new(2.5, 12.5), Point::new(13.5, 12.5)];
        assert_eq!(net.observe(&over, 1.0, 1), Some(0));
        assert_eq!(
            net.observe(&under, 1.0, 2),
            Some(0),
            "the same empty word: one class"
        );
        assert_eq!(net.observe(&over, 1.0, 3), Some(0));
        assert_eq!(net.observe(&under, 1.0, 4), Some(0));
        assert_eq!(net.symbols()[0].support, 4);
        assert_eq!(net.symbols()[0].exemplars.len(), 4);
        // A skeleton: a ring of channel cells round the middle.
        let mut channel = vec![false; 256];
        for x in 4..=11 {
            channel[3 * 16 + x] = true;
            channel[11 * 16 + x] = true;
        }
        for y in 3..=11 {
            channel[y * 16 + 4] = true;
            channel[y * 16 + 11] = true;
        }
        assert_eq!(net.learn_punctures(&channel), 1);
        assert_eq!(net.learned_punctures(), 1);
        let p = net.punctures()[0];
        assert!(
            (p.x - 8.25).abs() < 1.0 && (p.y - 7.25).abs() < 1.0,
            "the hole's middle: {p:?}"
        );
        // Now the two routes differ: the glyph (over) keeps the class,
        // and a route under is a new word.
        assert_eq!(net.symbols()[0].word.to_string(), "a");
        assert!(net.word(&under).is_empty());
        assert_eq!(net.observe(&under, 1.0, 5), Some(1));
        // A hole all the routes pass the same side of teaches nothing.
        let mut aside = vec![false; 256];
        for x in 1..=3 {
            aside[6 * 16 + x] = true;
            aside[9 * 16 + x] = true;
        }
        for y in 6..=9 {
            aside[y * 16 + 1] = true;
            aside[y * 16 + 3] = true;
        }
        assert_eq!(
            net.learn_punctures(&aside),
            0,
            "no class goes round it both ways"
        );
    }
}
