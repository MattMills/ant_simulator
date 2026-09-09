//! The path-topological network: a variable pheromone surface whose
//! channels are learned symbols, each a class of routes, and the
//! holonomy embedding those classes carry.
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
//! label nobody gave: the pattern names itself. Moves can be letters
//! too, a portal crossed on a free walk or a relation followed on a
//! walk of states, and a trip's word may begin with letters of its
//! own (a query), so that the classes are classes of that query's
//! routes.
//!
//! A word seen on enough successful trips is **registered** as a
//! symbol: a channel of its own in the symbol field, where the trips of
//! that class lay their trail; a representative route (its *glyph*); a
//! weight learned from how the class yields (quality per length,
//! against the other classes), which the thoughts sense as they sense a
//! pheromone; and a **transport**, the mean of the integrated state the
//! routes of the class arrive with, the class's vector in the holonomy
//! embedding, with the spread of those states as the curvature signal.
//! The symbols are the units of a network whose wiring is the topology
//! of the routes: an input (a route) activates the unit whose class it
//! is (**inference**, [`PathNet::label`]), and a unit can be driven to
//! produce a route of its class, either by a search in the space of
//! places and words ([`PathNet::generate`]) or by expressing its glyph
//! into its channel and attending to it, so that the thoughts walk the
//! class again (**generation**, [`PathNet::express`]). Punctures are
//! learned as well: a hole the walks enclose for several epochs, that
//! would tell apart routes now of one class, becomes a puncture, and
//! every symbol's word is refined under the new alphabet.

use ant_simulator::geometry::{Point, Position};
use ant_simulator::pheromone::perceived;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::fmt::Write as _;

/// A letter of a word: a crossing of a ray (the puncture's number from
/// one), a move (from [`MOVE_LETTERS`]) or a prefix (from
/// [`PREFIX_LETTERS`]), negative when inverse (a crossing from the
/// ray's right to its left, a move walked backwards).
pub type Letter = i16;

/// Letters from here are moves over the surface (a portal crossed),
/// numbered by the problem; below are ray crossings. A move followed by
/// its reverse cancels, as a crossing does: the way back through a
/// portal undoes the way there.
pub const MOVE_LETTERS: Letter = 1000;

/// Letters from here are links (a relation of a graph followed), which
/// never cancel: a relation followed and then followed backwards leads
/// somewhere else (a parent's other child), so the word keeps both.
pub const LINK_LETTERS: Letter = 10000;

/// Letters from here begin a trip's word (a query).
pub const PREFIX_LETTERS: Letter = 20000;

/// The letter of a move over the surface, by its number and direction.
pub fn move_letter(id: usize, forward: bool) -> Letter {
    let l = MOVE_LETTERS + (id as Letter).min(LINK_LETTERS - MOVE_LETTERS - 1);
    if forward {
        l
    } else {
        -l
    }
}

/// The letter of a link, by its number and direction.
pub fn link_letter(id: usize, forward: bool) -> Letter {
    let l = LINK_LETTERS + (id as Letter).min(PREFIX_LETTERS - LINK_LETTERS - 1);
    if forward {
        l
    } else {
        -l
    }
}

/// The number of a link letter, if it is one.
pub fn link_of(letter: Letter) -> Option<(usize, bool)> {
    let a = letter.abs();
    (LINK_LETTERS..PREFIX_LETTERS)
        .contains(&a)
        .then(|| ((a - LINK_LETTERS) as usize, letter > 0))
}

/// The letter a trip's word begins with, by its number.
pub fn prefix_letter(id: usize) -> Letter {
    PREFIX_LETTERS + (id as Letter).min(Letter::MAX - PREFIX_LETTERS - 1)
}

/// A word: the freely reduced sequence of letters of a route.
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

    /// Append a letter, cancelling it against an inverse last letter
    /// (a link never cancels).
    pub fn push(&mut self, letter: Letter) {
        if letter == 0 {
            return;
        }
        let cancels = letter.abs() < LINK_LETTERS && self.0.last() == Some(&-letter);
        if cancels {
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

    /// The word shown with names for its letters where given.
    pub fn show(&self, name: &dyn Fn(Letter) -> Option<String>) -> String {
        if self.0.is_empty() {
            return "1".to_string();
        }
        let mut out = String::new();
        for (i, &l) in self.0.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            match name(l.abs()) {
                Some(n) => out.push_str(&n),
                None => out.push_str(&raw_letter(l.abs())),
            }
            if l < 0 {
                out.push('\'');
            }
        }
        out
    }
}

fn raw_letter(l: Letter) -> String {
    if l >= PREFIX_LETTERS {
        format!("q{}", l - PREFIX_LETTERS)
    } else if l >= LINK_LETTERS {
        format!("r{}", l - LINK_LETTERS)
    } else if l >= MOVE_LETTERS {
        format!("m{}", l - MOVE_LETTERS)
    } else {
        let n = (l - 1) as u32;
        if n < 26 {
            char::from_u32('a' as u32 + n).unwrap_or('?').to_string()
        } else {
            format!("p{n}")
        }
    }
}

impl fmt::Display for Word {
    /// Letters `a`, `b`, `c`… for the punctures, `m0`, `m1`… for moves,
    /// `q0`… for prefixes, primed when inverse; `1` for the empty word.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.show(&|_| None))
    }
}

/// A route: its places, and the letter of each move between them (0
/// where a move is no letter; an empty list where none is).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Route {
    /// The places, the origin first.
    pub points: Vec<Point>,
    /// One letter per move, or none.
    pub letters: Vec<Letter>,
}

impl Route {
    /// A route of places whose moves are no letters.
    pub fn of(points: Vec<Point>) -> Route {
        Route {
            points,
            letters: Vec::new(),
        }
    }

    /// A route with the letters of its moves.
    pub fn with_letters(points: Vec<Point>, letters: Vec<Letter>) -> Route {
        Route { points, letters }
    }

    /// Length in cells.
    pub fn length(&self) -> f64 {
        self.points.windows(2).map(|w| w[0].distance(w[1])).sum()
    }

    /// The letter of the move to the point at `i` (from 1).
    pub fn letter(&self, i: usize) -> Letter {
        if i == 0 {
            0
        } else {
            self.letters.get(i - 1).copied().unwrap_or(0)
        }
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

    /// The word of a route: its prefix, then for every move the rays it
    /// crosses and its own letter.
    pub fn word(&self, route: &Route, prefix: &[Letter]) -> Word {
        let mut w = Word::new();
        for &l in prefix {
            w.push(l);
        }
        for (i, pair) in route.points.windows(2).enumerate() {
            self.step(pair[0], pair[1], &mut w);
            w.push(route.letter(i + 1));
        }
        w
    }

    /// The word of a route of places whose moves are no letters.
    pub fn word_of(&self, points: &[Point]) -> Word {
        let mut w = Word::new();
        for pair in points.windows(2) {
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
    pub fn stamp(&mut self, k: usize, points: &[Point], per_cell: f64) {
        for pair in points.windows(2) {
            self.lay(k, pair[0], pair[1], per_cell);
        }
        if points.len() == 1 {
            self.deposit(points[0].cell(), k, per_cell);
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
/// weight, its transport, its glyph and, if someone gave it one, its
/// name.
#[derive(Clone, Debug, PartialEq)]
pub struct Symbol {
    /// The class: the word of its routes.
    pub word: Word,
    /// The letters the class's routes begin their word with.
    pub prefix: Vec<Letter>,
    /// Its channel in the symbol field.
    pub channel: usize,
    /// Trips of the class brought home.
    pub support: u32,
    /// Trips of the class given up.
    pub misses: u32,
    /// Mean quality of the trips brought home (exponentially weighted).
    pub quality: f64,
    /// Mean length of their routes, in cells (exponentially weighted).
    pub length: f64,
    /// Quality per length against the shortest class: how the class
    /// yields.
    pub yield_: f64,
    /// The mean of quality times `exp(-length / temperature)` over the
    /// class's trips (exponentially weighted), when a temperature is
    /// set: with the support, the class's sector partition function.
    pub partition: f64,
    /// The class's share of the exponentiated-cost measure over the
    /// living classes, when a temperature is set.
    pub share: f64,
    /// The learned weight the thoughts sense the channel with: logits
    /// per unit of perceived level, from the yield against the others.
    pub weight: f64,
    /// The transport: the mean of what the class's routes integrate
    /// between departure and arrival, its vector in the holonomy
    /// embedding (empty where the problem integrates nothing).
    pub vector: Vec<f64>,
    /// The spread of those states about the mean (mean squared
    /// distance): the curvature signal, zero on a flat connection.
    pub spread: f64,
    /// A representative route: the best brought home.
    pub glyph: Route,
    /// The glyph's quality.
    pub glyph_quality: f64,
    /// The last few routes of the class brought home, which a candidate
    /// puncture is tested against: it is learned only where routes of
    /// one class go round it both ways.
    pub exemplars: Vec<Route>,
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

impl Symbol {
    /// Of the trips of the class, the share brought home.
    pub fn precision(&self) -> f64 {
        let n = self.support + self.misses;
        if n == 0 {
            0.0
        } else {
            self.support as f64 / n as f64
        }
    }
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
    /// A temperature: with one, a class's weight is the log of its
    /// share of the exponentiated-cost measure over the classes, its
    /// sector partition function (trips times the mean of quality
    /// times `exp(-length / temperature)`), rather than its yield.
    pub temperature: Option<f64>,
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
            temperature: None,
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
    best: Route,
    quality: f64,
    length: f64,
    prefix: Vec<Letter>,
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
    names: HashMap<Letter, String>,
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
            names: HashMap::new(),
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

    /// Name a letter (a move or a prefix), for showing words.
    pub fn set_name(&mut self, letter: Letter, name: &str) {
        self.names.insert(letter.abs(), name.to_string());
    }

    /// A word shown with the letters' names.
    pub fn show(&self, word: &Word) -> String {
        word.show(&|l| self.names.get(&l).cloned())
    }

    /// The word of a route with a prefix.
    pub fn word(&self, route: &Route, prefix: &[Letter]) -> Word {
        self.rays.word(route, prefix)
    }

    /// The word of a route of places whose moves are no letters.
    pub fn word_of(&self, points: &[Point]) -> Word {
        self.rays.word_of(points)
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

    /// A route brought home with a quality and what it integrated (the
    /// state it arrived with less the state it set out with): its
    /// class is found or, with enough support, registered; the class's
    /// statistics, transport, glyph and every class's weight are
    /// updated. Returns the class.
    pub fn observe(
        &mut self,
        route: &Route,
        prefix: &[Letter],
        x: &[f64],
        quality: f64,
        tick: u64,
    ) -> Option<usize> {
        self.observations += 1;
        if route.points.len() < 2 {
            return None;
        }
        let word = self.rays.word(route, prefix);
        if word.len() > self.cfg.max_word {
            self.dropped += 1;
            return None;
        }
        let length = route.length();
        let rate = self.cfg.rate.clamp(0.0, 1.0);
        let k = match self.find(&word) {
            Some(k) => {
                let s = &mut self.symbols[k];
                s.support += 1;
                s.quality += rate * (quality - s.quality);
                s.length += rate * (length - s.length);
                if let Some(t) = self.cfg.temperature {
                    let boltzmann = quality * (-length / t.max(1e-9)).exp();
                    s.partition += rate * (boltzmann - s.partition);
                }
                s.last_seen = tick;
                if s.vector.len() == x.len() && !x.is_empty() {
                    let dist2: f64 = s.vector.iter().zip(x).map(|(m, v)| (v - m) * (v - m)).sum();
                    s.spread += rate * (dist2 - s.spread);
                    for (m, v) in s.vector.iter_mut().zip(x) {
                        *m += rate * (v - *m);
                    }
                } else if !x.is_empty() {
                    s.vector = x.to_vec();
                    s.spread = 0.0;
                }
                let better = quality > s.glyph_quality + 1e-9
                    || ((quality - s.glyph_quality).abs() <= 1e-9 && length < s.glyph.length());
                if better {
                    s.glyph = route.clone();
                    s.glyph_quality = quality;
                }
                // The exemplars are a uniform sample of the class's routes
                // over its life (a reservoir), so that a way it went early
                // is still on record when a puncture is tested.
                let keep = self.cfg.exemplars.max(1);
                if s.exemplars.len() < keep {
                    s.exemplars.push(route.clone());
                } else {
                    self.lcg = self
                        .lcg
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let slot = (self.lcg >> 33) as usize % s.support.max(1) as usize;
                    if slot < keep {
                        s.exemplars[slot] = route.clone();
                    }
                }
                k
            }
            None => {
                let entry = self.pending.entry(word.clone()).or_insert(Pending {
                    count: 0,
                    best: route.clone(),
                    quality,
                    length,
                    prefix: prefix.to_vec(),
                });
                entry.count += 1;
                if quality > entry.quality || (quality == entry.quality && length < entry.length) {
                    entry.best = route.clone();
                    entry.quality = quality;
                    entry.length = length;
                }
                if entry.count < self.cfg.support {
                    return None;
                }
                let pending = self.pending.remove(&word).expect("just seen");
                self.register(word, pending, x, tick)?
            }
        };
        self.reweigh();
        Some(k)
    }

    /// A trip of a class given up: counted against the class's
    /// precision, if the class is registered.
    pub fn observe_miss(&mut self, route: &Route, prefix: &[Letter]) -> Option<usize> {
        let word = self.rays.word(route, prefix);
        let k = self.find(&word)?;
        self.symbols[k].misses += 1;
        Some(k)
    }

    fn register(&mut self, word: Word, pending: Pending, x: &[f64], tick: u64) -> Option<usize> {
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
        self.field.stamp(channel, &pending.best.points, deposit);
        self.symbols.push(Symbol {
            word,
            prefix: pending.prefix,
            channel,
            support: pending.count,
            misses: 0,
            quality: pending.quality,
            length: pending.length,
            yield_: 0.0,
            partition: self
                .cfg
                .temperature
                .map(|t| pending.quality * (-pending.length / t.max(1e-9)).exp())
                .unwrap_or(0.0),
            share: 0.0,
            weight: 0.0,
            vector: x.to_vec(),
            spread: 0.0,
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
        if self.cfg.temperature.is_some() {
            // Sector partition functions: a class's weight is the log of
            // its share against an even share.
            let z: f64 = living
                .iter()
                .map(|&k| self.symbols[k].support as f64 * self.symbols[k].partition)
                .sum();
            let n = living.len().max(1) as f64;
            for &k in &living {
                let s = &mut self.symbols[k];
                s.share = if z > 0.0 {
                    s.support as f64 * s.partition / z
                } else {
                    0.0
                };
                s.weight = if s.share > 0.0 {
                    (gain * (s.share * n).ln()).clamp(-gain, gain)
                } else {
                    -gain
                };
            }
            return;
        }
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
    pub fn label(&self, route: &Route, prefix: &[Letter]) -> Label {
        let word = self.rays.word(route, prefix);
        self.label_word(word)
    }

    /// What class a word is.
    pub fn label_word(&self, word: Word) -> Label {
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
        let glyph = &self.symbols[k].glyph.points;
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
                let (channel, glyph) = (s.channel, s.glyph.points.clone());
                self.field.stamp(channel, &glyph, per_cell);
            }
        }
    }

    /// A route of a class from one cell to another over a grid, or none
    /// within the longest word: the eight neighbours of a cell are its
    /// moves (no cutting of corners), and only the rays give letters.
    pub fn generate(
        &self,
        word: &Word,
        from: Position,
        to: Position,
        passable: &dyn Fn(Position) -> bool,
    ) -> Option<Vec<Point>> {
        let neighbours = |cell: Position| -> Vec<(Position, Letter)> {
            let mut out = Vec::with_capacity(8);
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
                    out.push((next, 0));
                }
            }
            out
        };
        if !passable(from) || !passable(to) {
            return None;
        }
        self.generate_with(word, from, to, &neighbours)
    }

    /// A route of a class from one cell to another, the moves of a cell
    /// given by `neighbours` with the letter of each (0 for none; a
    /// portal's letter, say): a breadth-first search over cells and the
    /// words of the ways to them, so that the route found is a shortest
    /// one of that class in moves. The rays' letters are added on every
    /// move between cells of one chart; a move that is a letter itself
    /// (a jump through a portal) contributes only its own.
    pub fn generate_with(
        &self,
        word: &Word,
        from: Position,
        to: Position,
        neighbours: &dyn Fn(Position) -> Vec<(Position, Letter)>,
    ) -> Option<Vec<Point>> {
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
            for (next, letter) in neighbours(cell) {
                let mut nw = w.clone();
                if letter == 0 {
                    self.rays
                        .step(Point::center_of(cell), Point::center_of(next), &mut nw);
                } else {
                    nw.push(letter);
                }
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
                let now = self.rays.word(route, &s.prefix);
                let then = trial.word(route, &s.prefix);
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
            let word = self
                .rays
                .word(&self.symbols[k].glyph, &self.symbols[k].prefix);
            match by_word.get(&word) {
                Some(&into) => {
                    let (support, misses, quality, length, glyph, glyph_quality, name, channel) = {
                        let s = &self.symbols[k];
                        (
                            s.support,
                            s.misses,
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
                    target.misses += misses;
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
            let prefix = self.symbols[k].prefix.clone();
            let rays = &self.rays;
            self.symbols[k]
                .exemplars
                .retain(|route| rays.word(route, &prefix) == word);
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
            let vector = if s.vector.is_empty() {
                String::new()
            } else {
                let norm: f64 = s.vector.iter().map(|v| v * v).sum::<f64>().sqrt();
                format!(" |vector| {:.2} spread {:.3}", norm, s.spread)
            };
            let share = if self.cfg.temperature.is_some() {
                format!(" share {:.3}", s.share)
            } else {
                String::new()
            };
            let _ = writeln!(
                out,
                "    #{k} [{}]{} support {}{} quality {:.3} length {:.0} yield {:.3}{} weight {:+.2}{}{}",
                self.show(&s.word),
                s.name
                    .as_ref()
                    .map(|n| format!(" \"{n}\""))
                    .unwrap_or_default(),
                s.support,
                if s.misses > 0 {
                    format!(" (precision {:.2})", s.precision())
                } else {
                    String::new()
                },
                s.quality,
                s.length,
                s.yield_,
                share,
                s.weight,
                vector,
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
        let m = Word::from_letters(&[
            prefix_letter(2),
            move_letter(0, true),
            move_letter(1, false),
        ]);
        assert_eq!(m.to_string(), "q2 m0 m1'");
        let back = Word::from_letters(&[move_letter(0, true), move_letter(0, false)]);
        assert!(back.is_empty(), "a move and its reverse cancel");
        let link = Word::from_letters(&[link_letter(0, true), link_letter(0, false)]);
        assert_eq!(link.to_string(), "r0 r0'", "a link and its reverse do not");
        assert_eq!(link_of(link_letter(3, false)), Some((3, false)));
        assert_eq!(link_of(move_letter(3, true)), None);
        let names = |l: Letter| match l {
            l if l == prefix_letter(2) => Some("cousin".to_string()),
            l if l == move_letter(0, true) => Some("parent".to_string()),
            _ => None,
        };
        assert_eq!(m.show(&names), "cousin parent m1'");
    }

    #[test]
    fn a_route_over_a_puncture_crosses_its_ray_and_one_under_does_not() {
        let rays = Rays::new(vec![Point::new(10.5, 10.0)]);
        let over = vec![Point::new(2.5, 4.5), Point::new(18.5, 4.5)];
        assert_eq!(rays.word_of(&over).to_string(), "a");
        let back = vec![Point::new(18.5, 4.5), Point::new(2.5, 4.5)];
        assert_eq!(rays.word_of(&back).to_string(), "a'");
        let under = vec![Point::new(2.5, 14.5), Point::new(18.5, 14.5)];
        assert!(rays.word_of(&under).is_empty());
        let there_and_back = vec![
            Point::new(2.5, 4.5),
            Point::new(18.5, 4.5),
            Point::new(2.5, 4.5),
        ];
        assert!(rays.word_of(&there_and_back).is_empty(), "cancels");
        let two = Rays::new(vec![Point::new(5.5, 10.0), Point::new(15.5, 10.0)]);
        let across = vec![Point::new(0.5, 2.5), Point::new(20.5, 2.5)];
        assert_eq!(two.word_of(&across).to_string(), "a b");
        // Move letters and a prefix join the word in order.
        let route = Route::with_letters(across.clone(), vec![move_letter(3, true)]);
        let word = two.word(&route, &[prefix_letter(0)]);
        assert_eq!(word.to_string(), "q0 a b m3");
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
        let over = Route::of(vec![
            Point::new(2.5, 2.5),
            Point::new(12.5, 1.5),
            Point::new(21.5, 2.5),
        ]);
        let under = Route::of(vec![
            Point::new(2.5, 13.5),
            Point::new(12.5, 14.5),
            Point::new(21.5, 13.5),
        ]);
        assert_eq!(
            net.observe(&over, &[], &[], 1.0, 1),
            None,
            "one trip is not a class"
        );
        assert_eq!(net.observe(&over, &[], &[], 1.0, 2), Some(0));
        assert_eq!(net.observe(&under, &[], &[], 1.0, 3), None);
        assert_eq!(net.observe(&under, &[], &[], 1.0, 4), Some(1));
        assert_eq!(net.symbols()[0].word.to_string(), "a");
        assert!(net.symbols()[1].word.is_empty());
        net.name(0, "over");
        let label = net.label(
            &Route::of(vec![Point::new(1.5, 3.5), Point::new(22.5, 3.5)]),
            &[],
        );
        assert_eq!(label.symbol, Some(0));
        assert_eq!(label.name.as_deref(), Some("over"));
        let label = net.label(
            &Route::of(vec![Point::new(1.5, 12.5), Point::new(22.5, 12.5)]),
            &[],
        );
        assert_eq!(label.symbol, Some(1));
        assert!(net.symbols()[0].weight >= net.symbols()[1].weight);
        assert!(
            net.field().total(net.symbols()[0].channel) > 0.0,
            "the glyph is laid"
        );
        assert_eq!(net.observe_miss(&over, &[]), Some(0));
        assert!((net.symbols()[0].precision() - 2.0 / 3.0).abs() < 1e-9);
        let passable = |p: Position| {
            p.x >= 0 && p.y >= 0 && p.x < 24 && p.y < 16 && !(p.x == 12 && (5..=10).contains(&p.y))
        };
        let from = Position::new(1, 8);
        let to = Position::new(22, 8);
        let a = net
            .generate(&Word::from_letters(&[1]), from, to, &passable)
            .expect("a route over");
        assert_eq!(net.word_of(&a).to_string(), "a");
        assert!(a.iter().all(|p| passable(p.cell())));
        let b = net
            .generate(&Word::new(), from, to, &passable)
            .expect("a route under");
        assert!(net.word_of(&b).is_empty());
        assert_eq!(a[0].cell(), from);
        assert_eq!(b.last().unwrap().cell(), to);
    }

    #[test]
    fn transports_are_the_mean_state_of_a_class_and_their_spread_its_curvature() {
        let cfg = SymbolConfig {
            support: 1,
            rate: 0.5,
            ..SymbolConfig::default()
        };
        let mut net = PathNet::new(16, 16, 1.0, Vec::new(), cfg);
        let route = Route::with_letters(
            vec![Point::new(2.5, 2.5), Point::new(8.5, 2.5)],
            vec![move_letter(0, true)],
        );
        assert_eq!(net.observe(&route, &[], &[1.0, 0.0], 1.0, 1), Some(0));
        assert_eq!(net.symbols()[0].vector, vec![1.0, 0.0]);
        net.observe(&route, &[], &[3.0, 0.0], 1.0, 2);
        let s = &net.symbols()[0];
        assert!((s.vector[0] - 2.0).abs() < 1e-9, "{:?}", s.vector);
        assert!(s.spread > 0.0, "a class arriving in two places has spread");
        assert_eq!(s.word.to_string(), "m0");
    }

    #[test]
    fn generation_follows_letters_of_moves_such_as_portals() {
        // A strip of 12 cells whose right end joins its left end (a
        // tube): jumping across is the letter of move 0.
        let cfg = SymbolConfig {
            max_word: 4,
            ..SymbolConfig::default()
        };
        let net = PathNet::new(12, 3, 1.0, Vec::new(), cfg);
        let neighbours = |c: Position| -> Vec<(Position, Letter)> {
            let mut out = Vec::new();
            if c.x > 0 {
                out.push((Position::new(c.x - 1, c.y), 0));
            } else {
                out.push((Position::new(11, c.y), move_letter(0, false)));
            }
            if c.x < 11 {
                out.push((Position::new(c.x + 1, c.y), 0));
            } else {
                out.push((Position::new(0, c.y), move_letter(0, true)));
            }
            out
        };
        let from = Position::new(2, 1);
        let once = net
            .generate_with(
                &Word::from_letters(&[move_letter(0, true)]),
                from,
                from,
                &neighbours,
            )
            .expect("once round");
        assert_eq!(once.len(), 13, "twelve moves round the tube");
        let twice = net
            .generate_with(
                &Word::from_letters(&[move_letter(0, true), move_letter(0, true)]),
                from,
                from,
                &neighbours,
            )
            .expect("twice round");
        assert_eq!(twice.len(), 25);
        let back = net
            .generate_with(
                &Word::from_letters(&[move_letter(0, false)]),
                from,
                from,
                &neighbours,
            )
            .expect("once round the other way");
        assert_eq!(back.len(), 13);
        assert!(net
            .generate_with(&Word::new(), from, Position::new(5, 1), &neighbours)
            .is_some());
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
        let over = Route::of(vec![Point::new(2.5, 2.5), Point::new(13.5, 2.5)]);
        let under = Route::of(vec![Point::new(2.5, 12.5), Point::new(13.5, 12.5)]);
        assert_eq!(net.observe(&over, &[], &[], 1.0, 1), Some(0));
        assert_eq!(
            net.observe(&under, &[], &[], 1.0, 2),
            Some(0),
            "the same empty word: one class"
        );
        assert_eq!(net.observe(&over, &[], &[], 1.0, 3), Some(0));
        assert_eq!(net.observe(&under, &[], &[], 1.0, 4), Some(0));
        assert_eq!(net.symbols()[0].support, 4);
        assert_eq!(net.symbols()[0].exemplars.len(), 4);
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
        assert_eq!(net.symbols()[0].word.to_string(), "a");
        assert!(net.word(&under, &[]).is_empty());
        assert_eq!(net.observe(&under, &[], &[], 1.0, 5), Some(1));
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
