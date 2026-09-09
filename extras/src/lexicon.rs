//! The lexicon: the symbols in use between the thoughts.
//!
//! A registered symbol is a class of routes with a label nobody gave.
//! It becomes a *sign* when it is used to tell another thought
//! something. Here the telling is a dance at the nest, as honeybees
//! dance a source's direction and distance: a thought that brings a
//! solution home dances the sign of the route it walked, in proportion
//! to what it found; the dance fades; a thought about to set out may
//! listen, draw a sign from the floor in proportion to how strongly it
//! is danced (the bees' draw: a dance is followed in proportion to how
//! much of it there is, so that a source found late can still take the
//! floor if it is worth more per trip), and depart with that sign in
//! mind: it sets out the way the sign's glyph goes, takes the glyph,
//! the route the class is known by, as the plan of its trip, follows
//! the ridge of the class's channel whatever its level, and disregards
//! the common trail, its own site and its habits for the trip, as a
//! recruit flies the vector it was danced rather than after the
//! foragers it sees leave. The sign then *refers*: its meaning is the
//! distribution of where trips that walked it ended, which the lexicon
//! keeps, block by block, as the sign's vector; what it is *taken* to
//! mean is where the trips that set out with it in mind ended, and the
//! agreement of the two is how well the sign is understood. Two signs
//! that lead to the same place are synonyms; the mutual information
//! between the sign danced and the outcome reached says how much the
//! lexicon carries. Nothing here is designed in: the signs are the
//! classes the walks registered, the meanings are the outcomes, and
//! the lexicon is whatever the dance floor keeps alive.

use crate::topos::PathNet;
use ant_simulator::geometry::Position;
use ant_simulator::rng::Rng;
use std::collections::HashMap;
use std::fmt::Write as _;

/// Configuration of the lexicon.
#[derive(Clone, Debug, PartialEq)]
pub struct LexiconConfig {
    /// The chance a departing follower listens to the dance floor.
    pub listen: f64,
    /// The temperature of the draw over danced signs: a sign is drawn
    /// in proportion to its dance intensity to the power of one over
    /// this. At one the draw is the bees' (a dance is followed in
    /// proportion to how much of it there is); toward zero only the
    /// loudest dance is followed.
    pub temperature: f64,
    /// Dance laid per delivery per unit of quality.
    pub dance: f64,
    /// Half-life of a dance, seconds.
    pub half_life_s: f64,
    /// Logits on a move for a thought that set out with a sign in
    /// mind, per unit of the sign's channel along the move relative to
    /// the most of it along any move open to the thought: the ridge of
    /// the channel is followed whatever its level, faint or strong.
    pub gain: f64,
    /// Whether a thought with a sign in mind takes the sign's glyph,
    /// the route its class is known by, as the plan of its trip, and
    /// walks it again; without, it follows the class's channel alone.
    pub enact: bool,
    /// How much of the common trail, and of what the classes say in
    /// general, a thought with a sign in mind disregards: at one it
    /// follows its sign alone, as a recruit flies the vector it was
    /// danced rather than after the foragers it sees leave.
    pub focus: f64,
    /// Rate of the exponentially weighted outcome distributions.
    pub rate: f64,
    /// Cells per side of the blocks outcomes are counted in.
    pub block: usize,
    /// Whether scouts listen too (followers always may).
    pub scouts_listen: bool,
}

impl Default for LexiconConfig {
    fn default() -> Self {
        LexiconConfig {
            listen: 0.7,
            temperature: 1.0,
            dance: 1.0,
            half_life_s: 300.0,
            gain: 3.0,
            enact: true,
            focus: 1.0,
            rate: 0.1,
            block: 8,
            scouts_listen: false,
        }
    }
}

/// A sign: a symbol as it is used.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sign {
    /// How strongly the sign is danced at present.
    pub dance: f64,
    /// Times danced.
    pub dances: u64,
    /// Departures with the sign in mind.
    pub recruits: u64,
    /// Recruited trips that came home in the sign's class.
    pub successes: u64,
    /// Recruited trips that came home in another class.
    pub strayed: u64,
    /// Recruited trips that came home with nothing.
    pub lost: u64,
    /// Where trips of the sign ended, by block, exponentially weighted:
    /// the sign's meaning.
    pub outcomes: HashMap<usize, f64>,
    /// Outcomes recorded.
    pub outcome_count: u64,
    /// Where trips that set out with the sign in mind ended, by block,
    /// exponentially weighted: what the sign is taken to mean.
    pub taken: HashMap<usize, f64>,
    /// Takings recorded.
    pub taken_count: u64,
}

impl Sign {
    /// Of the recruited trips, the share that came home in the sign's
    /// class.
    pub fn success_rate(&self) -> f64 {
        let n = self.successes + self.strayed + self.lost;
        if n == 0 {
            0.0
        } else {
            self.successes as f64 / n as f64
        }
    }
}

/// The lexicon over a mind's symbols.
#[derive(Clone, Debug)]
pub struct Lexicon {
    cfg: LexiconConfig,
    retention: f64,
    cols: usize,
    rows: usize,
    signs: Vec<Sign>,
    /// Dances danced.
    pub utterances: u64,
    /// Departures that listened.
    pub recruitments: u64,
}

impl Lexicon {
    /// A lexicon over a grid of the given size.
    pub fn new(width: usize, height: usize, tick_s: f64, cfg: LexiconConfig) -> Lexicon {
        let block = cfg.block.max(1);
        Lexicon {
            retention: if cfg.half_life_s > 0.0 {
                0.5f64.powf(tick_s / cfg.half_life_s)
            } else {
                0.0
            },
            cols: width.div_ceil(block),
            rows: height.div_ceil(block),
            cfg,
            signs: Vec::new(),
            utterances: 0,
            recruitments: 0,
        }
    }

    /// The configuration.
    pub fn config(&self) -> &LexiconConfig {
        &self.cfg
    }

    /// Room for `n` signs (one per symbol, by index).
    pub fn ensure(&mut self, n: usize) {
        if self.signs.len() < n {
            self.signs.resize(n, Sign::default());
        }
    }

    /// The signs, by symbol index.
    pub fn signs(&self) -> &[Sign] {
        &self.signs
    }

    /// A sign.
    pub fn sign(&self, k: usize) -> Option<&Sign> {
        self.signs.get(k)
    }

    /// One tick: the dances fade.
    pub fn step(&mut self) {
        for s in self.signs.iter_mut() {
            if s.dance > 0.0 {
                s.dance *= self.retention;
                if s.dance < 1e-9 {
                    s.dance = 0.0;
                }
            }
        }
    }

    /// Silence: every dance stops (the floor is cleared).
    pub fn silence(&mut self) {
        for s in self.signs.iter_mut() {
            s.dance = 0.0;
        }
    }

    /// A thought home with a solution dances its sign.
    pub fn dance(&mut self, k: usize, quality: f64) {
        self.ensure(k + 1);
        let s = &mut self.signs[k];
        s.dance += self.cfg.dance * quality.max(0.0);
        s.dances += 1;
        self.utterances += 1;
    }

    /// The block of a cell.
    pub fn block_of(&self, cell: Position) -> usize {
        let b = self.cfg.block.max(1) as i32;
        let x = (cell.x.max(0) / b) as usize;
        let y = (cell.y.max(0) / b) as usize;
        y.min(self.rows - 1) * self.cols + x.min(self.cols - 1)
    }

    /// The centre cell of a block.
    pub fn block_centre(&self, block: usize) -> Position {
        let b = self.cfg.block.max(1) as i32;
        Position::new(
            (block % self.cols) as i32 * b + b / 2,
            (block / self.cols) as i32 * b + b / 2,
        )
    }

    /// Where a trip of a sign ended: the sign's meaning takes it in.
    pub fn outcome(&mut self, k: usize, cell: Position) {
        self.ensure(k + 1);
        let block = self.block_of(cell);
        let rate = self.cfg.rate.clamp(0.0, 1.0);
        let s = &mut self.signs[k];
        Self::take_in(&mut s.outcomes, s.outcome_count, block, rate);
        s.outcome_count += 1;
    }

    /// Where a trip that set out with a sign in mind ended: what the
    /// sign was taken to mean.
    pub fn taken(&mut self, k: usize, cell: Position) {
        self.ensure(k + 1);
        let block = self.block_of(cell);
        let rate = self.cfg.rate.clamp(0.0, 1.0);
        let s = &mut self.signs[k];
        Self::take_in(&mut s.taken, s.taken_count, block, rate);
        s.taken_count += 1;
    }

    fn take_in(dist: &mut HashMap<usize, f64>, count: u64, block: usize, rate: f64) {
        if count == 0 {
            dist.insert(block, 1.0);
        } else {
            for v in dist.values_mut() {
                *v *= 1.0 - rate;
            }
            *dist.entry(block).or_insert(0.0) += rate;
            dist.retain(|_, v| *v > 1e-6);
        }
    }

    /// A sign drawn from the dance floor, in proportion to
    /// `dance ^ (1 / temperature)` over the living signs danced at all,
    /// or none if nothing is danced.
    pub fn draw(&self, living: &[usize], rng: &mut Rng) -> Option<usize> {
        let danced: Vec<usize> = living
            .iter()
            .copied()
            .filter(|&k| self.signs.get(k).map(|s| s.dance > 1e-9).unwrap_or(false))
            .collect();
        if danced.is_empty() {
            return None;
        }
        let t = self.cfg.temperature.max(1e-9);
        let peak = danced
            .iter()
            .map(|&k| self.signs[k].dance)
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = danced
            .iter()
            .map(|&k| (self.signs[k].dance / peak).powf(1.0 / t))
            .collect();
        let i = rng.choose_weighted(&weights);
        Some(danced[i])
    }

    /// A thought set out with a sign in mind.
    pub fn recruited(&mut self, k: usize) {
        self.ensure(k + 1);
        self.signs[k].recruits += 1;
        self.recruitments += 1;
    }

    /// A recruited trip came home in the sign's class.
    pub fn success(&mut self, k: usize) {
        self.ensure(k + 1);
        self.signs[k].successes += 1;
    }

    /// A recruited trip came home in another class.
    pub fn strayed(&mut self, k: usize) {
        self.ensure(k + 1);
        self.signs[k].strayed += 1;
    }

    /// A recruited trip came home with nothing.
    pub fn lost(&mut self, k: usize) {
        self.ensure(k + 1);
        self.signs[k].lost += 1;
    }

    /// Signs danced at present.
    pub fn in_use(&self, living: &[usize]) -> usize {
        living
            .iter()
            .filter(|&&k| self.signs.get(k).map(|s| s.dance > 1e-3).unwrap_or(false))
            .count()
    }

    /// The entropy of the dance floor, in nats: how many signs are in
    /// use, effectively.
    pub fn usage_entropy(&self, living: &[usize]) -> f64 {
        let total: f64 = living
            .iter()
            .map(|&k| self.signs.get(k).map(|s| s.dance).unwrap_or(0.0))
            .sum();
        if total <= 0.0 {
            return 0.0;
        }
        -living
            .iter()
            .filter_map(|&k| self.signs.get(k).map(|s| s.dance / total))
            .filter(|&p| p > 0.0)
            .map(|p| p * p.ln())
            .sum::<f64>()
    }

    /// The mutual information between the sign and the outcome block,
    /// and the entropy of the outcome, in nats, over the living signs
    /// weighted by their outcomes recorded: how much of where a trip
    /// ends its sign tells.
    pub fn mutual_information(&self, living: &[usize]) -> (f64, f64) {
        let mut joint: HashMap<(usize, usize), f64> = HashMap::new();
        let mut by_sign: HashMap<usize, f64> = HashMap::new();
        let mut by_block: HashMap<usize, f64> = HashMap::new();
        let mut total = 0.0;
        for &k in living {
            let Some(s) = self.signs.get(k) else { continue };
            if s.outcome_count == 0 {
                continue;
            }
            let mass: f64 = s.outcomes.values().sum();
            if mass <= 0.0 {
                continue;
            }
            let weight = s.outcome_count as f64;
            for (&b, &v) in &s.outcomes {
                let p = weight * v / mass;
                *joint.entry((k, b)).or_insert(0.0) += p;
                *by_sign.entry(k).or_insert(0.0) += p;
                *by_block.entry(b).or_insert(0.0) += p;
                total += p;
            }
        }
        if total <= 0.0 {
            return (0.0, 0.0);
        }
        let mut mi = 0.0;
        for (&(k, b), &p) in &joint {
            let pk = by_sign[&k] / total;
            let pb = by_block[&b] / total;
            let pkb = p / total;
            if pkb > 0.0 {
                mi += pkb * (pkb / (pk * pb)).ln();
            }
        }
        let h = -by_block
            .values()
            .map(|&v| v / total)
            .filter(|&p| p > 0.0)
            .map(|p| p * p.ln())
            .sum::<f64>();
        (mi.max(0.0), h)
    }

    /// A sign's meaning: the blocks its trips ended in, with their
    /// shares, largest first.
    pub fn meaning(&self, k: usize) -> Vec<(usize, f64)> {
        let Some(s) = self.signs.get(k) else {
            return Vec::new();
        };
        let mass: f64 = s.outcomes.values().sum();
        let mut out: Vec<(usize, f64)> = s
            .outcomes
            .iter()
            .map(|(&b, &v)| (b, if mass > 0.0 { v / mass } else { 0.0 }))
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// The cosine similarity of two signs' meanings.
    pub fn similarity(&self, a: usize, b: usize) -> f64 {
        let (Some(sa), Some(sb)) = (self.signs.get(a), self.signs.get(b)) else {
            return 0.0;
        };
        cosine(&sa.outcomes, &sb.outcomes)
    }

    /// How well a sign is understood: the cosine similarity of what it
    /// means (where its trips ended) and what it is taken to mean
    /// (where the trips that set out with it in mind ended); zero
    /// before anyone set out with it.
    pub fn understanding(&self, k: usize) -> f64 {
        let Some(s) = self.signs.get(k) else {
            return 0.0;
        };
        cosine(&s.outcomes, &s.taken)
    }

    /// What a sign is taken to mean: the blocks the trips that set out
    /// with it in mind ended in, with their shares, largest first.
    pub fn taken_meaning(&self, k: usize) -> Vec<(usize, f64)> {
        let Some(s) = self.signs.get(k) else {
            return Vec::new();
        };
        let mass: f64 = s.taken.values().sum();
        let mut out: Vec<(usize, f64)> = s
            .taken
            .iter()
            .map(|(&b, &v)| (b, if mass > 0.0 { v / mass } else { 0.0 }))
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// The living sign nearest in meaning to a sign, and the similarity.
    pub fn nearest(&self, k: usize, living: &[usize]) -> Option<(usize, f64)> {
        living
            .iter()
            .copied()
            .filter(|&j| j != k)
            .map(|j| (j, self.similarity(k, j)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// A summary of the lexicon.
    pub fn report(&self, net: &PathNet) -> String {
        let living = net.living();
        let (mi, h) = self.mutual_information(&living);
        let recruited: u64 = living
            .iter()
            .filter_map(|&k| self.signs.get(k))
            .map(|s| s.successes + s.strayed + s.lost)
            .sum();
        let successes: u64 = living
            .iter()
            .filter_map(|&k| self.signs.get(k))
            .map(|s| s.successes)
            .sum();
        let mut out = String::new();
        let _ = writeln!(
            out,
            "  lexicon: {} signs in use of {} living, usage entropy {:.2} nats, {} dances, {} departures listened, understood {:.0}%, sign tells {:.2} of {:.2} nats about the outcome",
            self.in_use(&living),
            living.len(),
            self.usage_entropy(&living),
            self.utterances,
            self.recruitments,
            if recruited > 0 {
                100.0 * successes as f64 / recruited as f64
            } else {
                0.0
            },
            mi,
            h
        );
        let mut order = living.clone();
        order.sort_by(|&a, &b| {
            let da = self.signs.get(a).map(|s| s.dance).unwrap_or(0.0);
            let db = self.signs.get(b).map(|s| s.dance).unwrap_or(0.0);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
        for k in order.into_iter().take(8) {
            let Some(s) = self.signs.get(k) else { continue };
            let symbol = &net.symbols()[k];
            let meaning: Vec<String> = self
                .meaning(k)
                .into_iter()
                .take(3)
                .map(|(b, p)| {
                    let c = self.block_centre(b);
                    format!("near ({},{}) {:.0}%", c.x, c.y, 100.0 * p)
                })
                .collect();
            let taken = if s.taken_count == 0 {
                String::new()
            } else {
                let c = self
                    .taken_meaning(k)
                    .first()
                    .map(|(b, _)| self.block_centre(*b))
                    .unwrap_or_default();
                format!(
                    ", taken to mean near ({},{}) at agreement {:.2}",
                    c.x,
                    c.y,
                    self.understanding(k)
                )
            };
            let _ = writeln!(
                out,
                "    #{k} [{}]{} dance {:.2}, danced {}, recruited {}, understood {:.0}%: {}{}",
                net.show(&symbol.word),
                symbol
                    .name
                    .as_ref()
                    .map(|n| format!(" \"{n}\""))
                    .unwrap_or_default(),
                s.dance,
                s.dances,
                s.recruits,
                100.0 * s.success_rate(),
                if meaning.is_empty() {
                    "no outcome yet".to_string()
                } else {
                    meaning.join(", ")
                },
                taken
            );
        }
        out
    }
}

/// The cosine similarity of two sparse vectors.
fn cosine(a: &HashMap<usize, f64>, b: &HashMap<usize, f64>) -> f64 {
    let dot: f64 = a
        .iter()
        .map(|(k, v)| v * b.get(k).copied().unwrap_or(0.0))
        .sum();
    let na: f64 = a.values().map(|v| v * v).sum::<f64>().sqrt();
    let nb: f64 = b.values().map(|v| v * v).sum::<f64>().sqrt();
    if na <= 0.0 || nb <= 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signs_are_danced_drawn_and_mean_where_they_lead() {
        let mut lex = Lexicon::new(32, 32, 1.0, LexiconConfig::default());
        let living = vec![0, 1];
        let mut rng = Rng::seed_from_u64(1);
        assert!(lex.draw(&living, &mut rng).is_none(), "nothing danced");
        lex.dance(0, 1.0);
        lex.dance(1, 0.2);
        for _ in 0..20 {
            lex.outcome(0, Position::new(28, 4));
            lex.outcome(1, Position::new(4, 28));
        }
        let draws: Vec<usize> = (0..200)
            .map(|_| lex.draw(&living, &mut rng).unwrap())
            .collect();
        let zeros = draws.iter().filter(|&&k| k == 0).count();
        assert!(
            zeros > 150,
            "the stronger dance recruits more: {zeros} of 200"
        );
        let (mi, h) = lex.mutual_information(&living);
        assert!(
            h > 0.6 && (mi - h).abs() < 1e-9,
            "each sign means one place: I {mi} = H {h}"
        );
        assert!(lex.similarity(0, 1) < 1e-9, "different places");
        lex.outcome(2, Position::new(29, 5));
        assert!(lex.similarity(0, 2) > 0.99, "the same block: synonyms");
        assert_eq!(lex.nearest(0, &[0, 1, 2]).unwrap().0, 2);
        lex.recruited(0);
        lex.success(0);
        lex.recruited(0);
        lex.strayed(0);
        assert!((lex.sign(0).unwrap().success_rate() - 0.5).abs() < 1e-9);
        assert_eq!(
            lex.understanding(0),
            0.0,
            "nobody has arrived with it in mind"
        );
        lex.taken(0, Position::new(27, 5));
        assert!(lex.understanding(0) > 0.99, "taken to mean where it means");
        for _ in 0..5 {
            lex.taken(0, Position::new(4, 28));
        }
        assert!(
            lex.understanding(0) < 0.9,
            "and less so when taken elsewhere"
        );
        for _ in 0..3000 {
            lex.step();
        }
        assert!(lex.sign(0).unwrap().dance < 0.01, "dances fade");
        assert_eq!(lex.in_use(&living), 0);
    }
}
