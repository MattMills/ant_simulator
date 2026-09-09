//! The interior: a higher-order layer of thought that walks the
//! colony's own abstraction.
//!
//! The thoughts that walk a problem leave an invariant structure behind
//! them: the words registered as symbols, what each yields, the glyph
//! of each, the skeleton of their traffic. That structure is a space,
//! and the same dynamics can walk it. An [`Interior`] is a problem
//! whose states are words: a move appends a letter of the vocabulary,
//! or a whole known word at once (a *chunk*); a solution is a word the
//! outer thoughts have brought home, worth what it yields them; the
//! scent of a move is the yield of the known words it leads toward. A
//! [`Colony`] is an outer mind on its problem and an inner mind on the
//! interior, stepped at a slower rate, with marks that last longer.
//! The inner thoughts lay trail over words: the colony's belief about
//! its own vocabulary. Their history keeps the invariant traffic over
//! words. Their symbols, if they keep any, are classes of derivations:
//! which chunks a word was composed from, the parse of a thought.
//!
//! The two layers interact. Upward, every epoch the interior's
//! landscape is rewritten from the outer symbols: the interior's food
//! is the colony's success, and a class that stops yielding stops
//! being food. Downward, every epoch each outer symbol is attended to
//! in proportion to the interior's trail on its word, so that what the
//! colony thinks about is what its thoughts sense and set out toward:
//! the imprinting of the new thoughts by the colony's interior
//! structure. And a word the inner thoughts arrive at that no outer
//! thought has walked is a hypothesis: a route of that class is
//! generated over the outer embedding and proposed as a symbol with no
//! support, laid out for the outer thoughts to test; confirmed if they
//! bring it home, evicted if they never do. The interior thinks in the
//! foragers' language, one level up, and the architecture is the same
//! at both levels, so a third could walk the interior's symbols in
//! turn.

use crate::lexicon::LexiconConfig;
use crate::mind::{Mind, MindConfig};
use crate::problem::{Layered, Moves, Problem};
use crate::topos::{
    move_letter, Letter, PathNet, Route, SymbolConfig, Word, DESTINATION_LETTERS, MOVE_LETTERS,
};
use ant_simulator::geometry::Point;
use ant_simulator::pheromone::Pheromone;
use ant_simulator::world::WorldConfig;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Write as _;

/// Configuration of an interior and its coupling to the outer mind.
#[derive(Clone, Debug, PartialEq)]
pub struct InteriorConfig {
    /// Inner thoughts.
    pub thoughts: usize,
    /// The interior's grid: words are laid out layer by layer of their
    /// length across the width, in slots down the height.
    pub width: usize,
    /// Height of the interior's grid.
    pub height: usize,
    /// The longest word the inner thoughts walk.
    pub depth: usize,
    /// Whether known words are moves in themselves (chunks), so that a
    /// word can be composed of known words as well as spelt.
    pub chunks: bool,
    /// Outer ticks per inner tick: the interior's slower epoch.
    pub slowness: u64,
    /// Inner ticks an inner thought rests between trips: long, so that
    /// a long thought costs little more than a short one and the
    /// interior's trail measures worth rather than brevity.
    pub rest_ticks: u64,
    /// Trips a word needs before it is worth its full yield to the
    /// interior: its worth is shrunk by `support / (support + prior)`.
    pub support_prior: f64,
    /// Outer ticks between couplings of the two layers.
    pub epoch_ticks: u64,
    /// Half-life of the interior's trail, seconds: its marks last.
    pub half_life_s: f64,
    /// Gain of the attention laid on an outer symbol whose word carries
    /// the interior's strongest trail (the others in proportion).
    pub attention: f64,
    /// The standing dance laid on the outer lexicon's sign of the word
    /// that carries the interior's strongest trail (the others in
    /// proportion), as a fraction of the loudest dance on the floor
    /// (of one dance, on a silent floor): the interior speaking through
    /// the floor, so that recruits enact what it believes, and test
    /// what it proposes.
    pub standing: f64,
    /// How strongly the interior's best word is expressed for the
    /// outer thoughts, as a factor of a trip's deposit per cell; none
    /// at zero.
    pub expression: f64,
    /// Whether words the inner thoughts arrive at that no outer thought
    /// has walked are proposed to the outer mind as hypotheses.
    pub propose: bool,
    /// The interior's own symbols (classes of derivations), if kept.
    pub symbols: Option<SymbolConfig>,
    /// The interior's own dance floor, if kept: with symbols and a
    /// floor of its own, an interior is an outer mind for a further
    /// interior.
    pub lexicon: Option<LexiconConfig>,
}

impl Default for InteriorConfig {
    fn default() -> Self {
        InteriorConfig {
            thoughts: 12,
            width: 64,
            height: 48,
            depth: 8,
            chunks: true,
            slowness: 4,
            rest_ticks: 16,
            support_prior: 10.0,
            epoch_ticks: 200,
            half_life_s: 3000.0,
            attention: 0.0,
            standing: 0.5,
            expression: 0.0,
            propose: true,
            symbols: Some(SymbolConfig {
                support: 2,
                max_word: 8,
                epoch_ticks: 250,
                ..SymbolConfig::default()
            }),
            lexicon: None,
        }
    }
}

/// A word the outer thoughts have brought home, as the interior knows
/// it.
#[derive(Clone, Debug, PartialEq)]
struct Known {
    word: Word,
    quality: f64,
    symbol: usize,
}

/// The colony's vocabulary as a problem: states are words, moves
/// append letters or known words, solutions are the words that yield.
#[derive(Clone, Debug)]
pub struct Interior {
    layout: Layered,
    depth: usize,
    chunks: bool,
    alphabet: Vec<Letter>,
    known: Vec<Known>,
    index: HashMap<Word, usize>,
    chunk_words: Vec<Word>,
    names: HashMap<Letter, String>,
    support_prior: f64,
    slots: RefCell<HashMap<Word, usize>>,
    taken: RefCell<Vec<Vec<bool>>>,
    hypotheses: RefCell<Vec<Word>>,
    refreshed: u64,
}

impl Interior {
    /// An empty interior: nothing is known yet.
    pub fn new(cfg: &InteriorConfig) -> Interior {
        let depth = cfg.depth.max(1);
        let values = cfg.height.saturating_sub(4).max(1);
        Interior {
            layout: Layered::new(cfg.width.max(8), cfg.height.max(8), depth, values),
            depth,
            chunks: cfg.chunks,
            alphabet: Vec::new(),
            known: Vec::new(),
            index: HashMap::new(),
            chunk_words: Vec::new(),
            names: HashMap::new(),
            support_prior: cfg.support_prior.max(0.0),
            slots: RefCell::new(HashMap::new()),
            taken: RefCell::new(vec![vec![false; values]; depth]),
            hypotheses: RefCell::new(Vec::new()),
            refreshed: 0,
        }
    }

    /// The vocabulary, read from the outer network: the living symbols
    /// with support, their words and what they are worth (their yield
    /// against the best yield, times their precision), and the letters
    /// they use.
    pub fn refresh(&mut self, net: &PathNet) {
        self.refreshed += 1;
        self.known.clear();
        self.index.clear();
        self.chunk_words.clear();
        self.names.clear();
        let best_yield = net
            .living()
            .into_iter()
            .map(|k| net.symbols()[k].yield_)
            .fold(0.0, f64::max)
            .max(1e-9);
        let mut letters: Vec<Letter> = Vec::new();
        for k in net.living() {
            let s = &net.symbols()[k];
            if s.support == 0 || s.word.is_empty() {
                continue;
            }
            for &l in s.word.letters() {
                letters.push(l);
                if l.abs() < MOVE_LETTERS {
                    letters.push(-l);
                }
            }
            let evidence = s.support as f64 / (s.support as f64 + self.support_prior);
            let quality = (s.yield_ / best_yield * s.precision() * evidence).clamp(0.0, 1.0);
            if quality > 0.0 {
                self.index.insert(s.word.clone(), self.known.len());
                self.known.push(Known {
                    word: s.word.clone(),
                    quality,
                    symbol: k,
                });
            }
            let route = s.word.without_destinations();
            if route.len() >= 2 && !self.chunk_words.contains(&route) {
                self.chunk_words.push(route);
            }
        }
        letters.sort_by_key(|l| (l.abs() >= DESTINATION_LETTERS, *l));
        letters.dedup();
        self.alphabet = letters;
        for &l in &self.alphabet {
            self.names.insert(l, net.show(&Word::from_letters(&[l])));
        }
        for (j, w) in self.chunk_words.iter().enumerate() {
            self.names
                .insert(move_letter(j, true), format!("[{}]", net.show(w)));
        }
    }

    /// Words the inner thoughts arrived at that are not known, since
    /// last asked.
    pub fn take_hypotheses(&mut self) -> Vec<Word> {
        std::mem::take(&mut *self.hypotheses.borrow_mut())
    }

    /// The known words and what they yield.
    pub fn known(&self) -> Vec<(Word, f64, usize)> {
        self.known
            .iter()
            .map(|k| (k.word.clone(), k.quality, k.symbol))
            .collect()
    }

    /// The letters in use.
    pub fn alphabet(&self) -> &[Letter] {
        &self.alphabet
    }

    /// The known words that are moves in themselves.
    pub fn chunk_words(&self) -> &[Word] {
        &self.chunk_words
    }

    fn is_arrival(word: &Word) -> bool {
        word.letters()
            .last()
            .map(|l| l.abs() >= DESTINATION_LETTERS)
            .unwrap_or(false)
    }

    fn layer_of(&self, word: &Word) -> usize {
        word.len().clamp(1, self.depth) - 1
    }

    fn hash(word: &Word) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &l in word.letters() {
            for b in (l as i64).to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }

    /// The slot of a word in its layer: taken on first sight, the
    /// nearest free one to where the word hashes, so that a word always
    /// lies where it lay.
    fn slot_of(&self, word: &Word) -> usize {
        if let Some(&s) = self.slots.borrow().get(word) {
            return s;
        }
        let layer = self.layer_of(word);
        let values = self.layout.values.max(1);
        let start = (Self::hash(word) % values as u64) as usize;
        let mut taken = self.taken.borrow_mut();
        let row = &mut taken[layer];
        let slot = (0..values)
            .map(|i| (start + i) % values)
            .find(|&s| !row[s])
            .unwrap_or(start);
        row[slot] = true;
        self.slots.borrow_mut().insert(word.clone(), slot);
        slot
    }
}

impl Problem for Interior {
    type State = Word;

    fn embedding(&self) -> WorldConfig {
        self.layout.config()
    }

    fn origin(&self) -> Word {
        Word::new()
    }

    fn place(&self, word: &Word) -> Point {
        if word.is_empty() {
            self.layout.origin()
        } else {
            self.layout.place(self.layer_of(word), self.slot_of(word))
        }
    }

    fn moves(&self, word: &Word) -> Moves<Word> {
        if Self::is_arrival(word) {
            // An arrival that is not known: a hypothesis, and a dead end
            // until the outer thoughts confirm it.
            if !self.index.contains_key(word) {
                let mut h = self.hypotheses.borrow_mut();
                if !h.contains(word) {
                    h.push(word.clone());
                }
            }
            return Moves::States(Vec::new());
        }
        if word.len() >= self.depth {
            return Moves::States(Vec::new());
        }
        let last = word.letters().last().copied();
        let mut out = Vec::new();
        for &l in &self.alphabet {
            if last == Some(-l) {
                continue;
            }
            let mut w = word.clone();
            w.push(l);
            if w.len() <= self.depth && w.len() > word.len() {
                out.push(w);
            }
        }
        if self.chunks {
            for v in &self.chunk_words {
                if word.len() + v.len() > self.depth || last == v.letters().first().map(|f| -f) {
                    continue;
                }
                let w = word.then(v);
                if w.len() == word.len() + v.len() && !out.contains(&w) {
                    out.push(w);
                }
            }
        }
        Moves::States(out)
    }

    fn quality(&self, word: &Word) -> Option<f64> {
        self.index.get(word).map(|&i| self.known[i].quality)
    }

    /// The yield of the known words a move leads toward, discounted by
    /// how many letters remain; or the word's own, if it is known.
    fn scent(&self, _from: &Word, to: &Word) -> f64 {
        let mut best: f64 = self.quality(to).unwrap_or(0.0);
        for k in &self.known {
            if k.word.len() > to.len() && k.word.letters()[..to.len()] == *to.letters() {
                best = best.max(k.quality / (1.0 + (k.word.len() - to.len()) as f64));
            }
        }
        best.min(1.0)
    }

    /// A chunk is a letter of the interior's own words: the parse.
    fn letter(&self, from: &Word, to: &Word) -> Option<Letter> {
        if to.len() < from.len() + 2 {
            return None;
        }
        self.chunk_words
            .iter()
            .position(|v| from.then(v) == *to)
            .map(|j| move_letter(j, true))
    }

    fn letter_names(&self) -> Vec<(Letter, String)> {
        self.names.iter().map(|(l, n)| (*l, n.clone())).collect()
    }

    fn describe(&self, word: &Word) -> String {
        if word.is_empty() {
            "the empty word".to_string()
        } else {
            word.show(&|l| self.names.get(&l).cloned())
        }
    }

    fn name(&self) -> String {
        "interior".to_string()
    }
}

/// An outer mind on its problem and an inner mind on the interior,
/// coupled.
pub struct Colony<P: Problem> {
    outer: Mind<P>,
    inner: Mind<Interior>,
    cfg: InteriorConfig,
    tick: u64,
    proposals: Vec<usize>,
    /// Couplings made.
    pub couplings: u64,
    /// Outer symbols attended to at the last coupling.
    pub attended: Vec<(usize, f64)>,
    /// Expressions of the interior's best word.
    pub expressed: u64,
    /// Hypotheses proposed to the outer mind.
    pub proposed: u64,
    /// Hypotheses the outer thoughts confirmed by bringing them home.
    pub confirmed: u64,
    /// Hypotheses the inner thoughts arrived at that could not be
    /// generated as routes.
    pub ungenerable: u64,
    /// The confirmed hypotheses, shown.
    pub confirmed_words: Vec<String>,
}

impl<P: Problem> Colony<P> {
    /// A colony: the outer mind under its configuration (which needs
    /// symbols for the interior to have anything to walk), the inner
    /// one under the interior's.
    pub fn new(problem: P, outer: MindConfig, cfg: InteriorConfig) -> Colony<P> {
        let outer = Mind::new(problem, outer);
        let interior = Interior::new(&cfg);
        let mut inner_cfg = MindConfig {
            thoughts: cfg.thoughts.max(1),
            trip_budget: cfg.depth + 2,
            rest_ticks: cfg.rest_ticks,
            ..MindConfig::default()
        }
        .with_trail_half_life(cfg.half_life_s);
        if let Some(s) = &cfg.symbols {
            inner_cfg = inner_cfg.with_symbols(s.clone());
        }
        if let Some(l) = &cfg.lexicon {
            inner_cfg = inner_cfg.with_lexicon(l.clone());
        }
        let inner = Mind::new(interior, inner_cfg);
        Colony {
            outer,
            inner,
            cfg,
            tick: 0,
            proposals: Vec::new(),
            couplings: 0,
            attended: Vec::new(),
            expressed: 0,
            proposed: 0,
            confirmed: 0,
            ungenerable: 0,
            confirmed_words: Vec::new(),
        }
    }

    /// The outer mind.
    pub fn outer(&self) -> &Mind<P> {
        &self.outer
    }

    /// The outer mind, to change its world.
    pub fn outer_mut(&mut self) -> &mut Mind<P> {
        &mut self.outer
    }

    /// The inner mind.
    pub fn inner(&self) -> &Mind<Interior> {
        &self.inner
    }

    /// The configuration.
    pub fn config(&self) -> &InteriorConfig {
        &self.cfg
    }

    /// Outer ticks run.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// One outer tick; an inner tick every `slowness` of them; a
    /// coupling every epoch.
    pub fn step(&mut self) {
        self.outer.step();
        self.tick += 1;
        if self.tick.is_multiple_of(self.cfg.slowness.max(1)) {
            self.inner.step();
        }
        if self.tick.is_multiple_of(self.cfg.epoch_ticks.max(1)) {
            self.couple();
        }
    }

    /// Run for a number of outer ticks.
    pub fn run(&mut self, ticks: u64) {
        for _ in 0..ticks {
            self.step();
        }
    }

    /// The interior's trail on each known word and each word it has
    /// proposed, normalised to the strongest: the colony's belief about
    /// its vocabulary.
    pub fn beliefs(&self) -> Vec<(Word, f64, usize)> {
        let mut words: Vec<(Word, usize)> = self
            .inner
            .problem()
            .known()
            .into_iter()
            .map(|(w, _, k)| (w, k))
            .collect();
        if let Some(net) = self.outer.net() {
            for &k in &self.proposals {
                if let Some(s) = net.symbols().get(k).filter(|s| s.alive) {
                    words.push((s.word.clone(), k));
                }
            }
        }
        let levels: Vec<f64> = words
            .iter()
            .map(|(w, _)| {
                let p = self.inner.problem().place(w);
                self.inner.world().level(p.cell(), Pheromone::Trail)
            })
            .collect();
        let peak = levels.iter().cloned().fold(0.0, f64::max).max(1e-12);
        words
            .into_iter()
            .zip(levels)
            .map(|((w, k), l)| (w, l / peak, k))
            .collect()
    }

    /// The two layers meet: the interior reads the vocabulary, the
    /// outer symbols are attended to by the interior's trail, the
    /// interior's best word is expressed, and its hypotheses are
    /// proposed.
    fn couple(&mut self) {
        self.couplings += 1;
        let Some(net) = self.outer.net() else {
            return;
        };
        // Confirmations of earlier hypotheses.
        let mut still = Vec::new();
        for &k in &self.proposals {
            match net.symbols().get(k) {
                Some(s) if s.alive && s.support > 0 => {
                    self.confirmed += 1;
                    self.confirmed_words.push(net.show(&s.word));
                }
                Some(s) if s.alive => still.push(k),
                _ => {}
            }
        }
        self.proposals = still;
        // Up: the vocabulary.
        self.inner.problem_mut().refresh(net);
        // Down: attention in proportion to the interior's trail, and a
        // standing dance on the floor in the same proportion.
        let beliefs = self.beliefs();
        let gain = self.cfg.attention;
        let standing = self.cfg.standing;
        let mut attended = Vec::new();
        if let Some(net) = self.outer.net_mut() {
            for (_, level, k) in &beliefs {
                let a = gain * level;
                if a > 1e-3 {
                    net.attend(*k, a);
                    attended.push((*k, a));
                } else {
                    net.release(*k);
                }
            }
        }
        if let Some(lex) = self.outer.lexicon_mut() {
            let loudest = lex
                .signs()
                .iter()
                .map(|s| s.dance)
                .fold(0.0, f64::max)
                .max(1.0);
            for (_, level, k) in &beliefs {
                lex.set_standing(*k, standing * loudest * level);
            }
        }
        attended.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        self.attended = attended;
        // Down: the interior's best word, expressed.
        if self.cfg.expression > 0.0 {
            let best = self
                .inner
                .best()
                .and_then(|b| b.route.last().cloned())
                .and_then(|w| self.outer.net().and_then(|n| n.find(&w)));
            if let Some(k) = best {
                let per_cell = self
                    .outer
                    .config()
                    .symbols
                    .as_ref()
                    .map(|c| c.deposit * self.cfg.expression)
                    .unwrap_or(0.0);
                if let Some(net) = self.outer.net_mut() {
                    net.express(k, per_cell);
                    self.expressed += 1;
                }
            }
        }
        // Down: hypotheses.
        if self.cfg.propose {
            let hypotheses = self.inner.problem_mut().take_hypotheses();
            for word in hypotheses {
                if self
                    .outer
                    .net()
                    .map(|n| n.find(&word).is_some())
                    .unwrap_or(true)
                {
                    continue;
                }
                let Some(&suffix) = word
                    .letters()
                    .last()
                    .filter(|l| l.abs() >= DESTINATION_LETTERS)
                else {
                    continue;
                };
                let Some(state) = self.outer.problem().destination(suffix) else {
                    self.ungenerable += 1;
                    continue;
                };
                let to = self.outer.problem().place(&state).cell();
                let Some(points) = self.outer.generate(&word, to) else {
                    self.ungenerable += 1;
                    continue;
                };
                let route = Route::of(points).ending(vec![suffix]);
                let tick = self.outer.tick();
                let deposit = self
                    .outer
                    .config()
                    .symbols
                    .as_ref()
                    .map(|c| c.deposit * c.expression)
                    .unwrap_or(0.0);
                if let Some(net) = self.outer.net_mut() {
                    if let Some(k) = net.propose(word, route, tick) {
                        net.express(k, deposit);
                        if gain > 0.0 {
                            net.attend(k, gain);
                        }
                        self.proposals.push(k);
                        self.proposed += 1;
                    }
                }
            }
        }
    }

    /// A summary of both layers and their coupling.
    pub fn report(&self) -> String {
        let mut out = self.outer.report();
        let s = self.inner.stats();
        let _ = writeln!(
            out,
            "  interior: tick {}, {} thoughts, {} trips, {} brought home (mean quality {:.3}), {} couplings; {} known words, {} chunks; {} attended, {} expressed, {} hypotheses proposed, {} confirmed, {} not generable",
            self.inner.tick(),
            self.inner.thoughts().len(),
            s.trips,
            s.deliveries,
            s.mean_quality(),
            self.couplings,
            self.inner.problem().known().len(),
            self.inner.problem().chunk_words().len(),
            self.attended.len(),
            self.expressed,
            self.proposed,
            self.confirmed,
            self.ungenerable
        );
        let mut beliefs = self.beliefs();
        beliefs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (w, level, k) in beliefs.iter().take(6) {
            let shown = self
                .outer
                .net()
                .map(|n| n.show(w))
                .unwrap_or_else(|| w.to_string());
            let _ = writeln!(
                out,
                "    belief {:.2} in #{k} [{}]{}",
                level,
                shown,
                self.attended
                    .iter()
                    .find(|(j, _)| j == k)
                    .map(|(_, a)| format!(", attended at {a:.2}"))
                    .unwrap_or_default()
            );
        }
        if !self.confirmed_words.is_empty() {
            let _ = writeln!(
                out,
                "    confirmed hypotheses: {}",
                self.confirmed_words.join("; ")
            );
        }
        if let Some(net) = self.inner.net() {
            let living = net.living();
            if !living.is_empty() {
                let parses: Vec<String> = living
                    .iter()
                    .take(6)
                    .map(|&k| {
                        let s = &net.symbols()[k];
                        format!("[{}] ×{}", net.show(&s.word), s.support)
                    })
                    .collect();
                let _ = writeln!(
                    out,
                    "    parses (the interior's own classes): {}",
                    parses.join(", ")
                );
            }
        }
        out
    }
}
