//! The echo: signs persisted in walk form.
//!
//! The dance floor speaks fast and at the nest: a sign is danced when
//! a thought comes home with something, and fades within a few
//! hundred ticks. An *echo* is a thought that speaks the same language
//! at a slower epoch and everywhere along the route. It draws a sign
//! from the floor, holds it for an epoch, and walks the sign's glyph
//! out and back, again and again, laying the sign's channel both ways
//! whether or not anything is at the end. The other thoughts receive
//! it as traffic: the walks enter the movement history, whose
//! invariant skeleton is the colony's cognitive geometry, and a
//! thought out searching that meets an echo on its way may take the
//! sign it walks as its own, from the point where they met, as a
//! follower takes the route from a tandem leader. The echoes are the
//! colony's replay: the memory of a sign is kept by being re-spoken,
//! refined by every walk of it that finds a shorter route of the
//! class, re-seeded into the floor whenever a walk finds the source
//! still there, and let go after enough walks that find nothing at the
//! end, or when the epoch ends and the floor has moved on.

use crate::lexicon::Lexicon;
use crate::topos::PathNet;
use std::collections::HashMap;
use std::fmt::Write as _;

/// Configuration of the echoes.
#[derive(Clone, Debug, PartialEq)]
pub struct EchoConfig {
    /// Thoughts that echo (taken from the followers).
    pub echoes: usize,
    /// Ticks an echo holds a sign before drawing from the floor again.
    pub epoch_ticks: u64,
    /// Walks that find nothing at the glyph's end before the sign is
    /// let go.
    pub patience: u32,
    /// Laid per cell into the sign's channel by an echo on its way,
    /// out and back, when it carries nothing.
    pub deposit: f64,
    /// The chance per tick that a thought out searching within reach
    /// of an echo takes the echo's sign as its own.
    pub hear: f64,
    /// Reach of hearing, in cells (Chebyshev).
    pub radius: i32,
}

impl Default for EchoConfig {
    fn default() -> Self {
        EchoConfig {
            echoes: 4,
            epoch_ticks: 1000,
            patience: 6,
            deposit: 10.0,
            hear: 0.5,
            radius: 1,
        }
    }
}

/// The echoes' bookkeeping.
#[derive(Clone, Debug)]
pub struct Echo {
    cfg: EchoConfig,
    /// Walks set out on with a sign held.
    pub walks: u64,
    /// Walks that found the source at the glyph's end and brought it
    /// home (and danced it: the floor re-seeded from memory).
    pub verified: u64,
    /// Walks that found nothing at the glyph's end.
    pub empty: u64,
    /// Signs let go after too many empty walks or the death of the
    /// class.
    pub dropped: u64,
    /// Signs taken up anew from the floor.
    pub drawn: u64,
    /// Epochs a sign was held through in silence (nothing danced).
    pub kept_in_silence: u64,
    /// Walks of an echo that shortened the class's glyph.
    pub refined: u64,
    /// Thoughts that took a sign from an echo they met on the way.
    pub heard: u64,
    /// Of those, trips that came home at all.
    pub heard_home: u64,
    /// Of those, trips that came home in the sign's class.
    pub heard_understood: u64,
    /// Walks per sign.
    pub echoed: HashMap<usize, u64>,
}

impl Echo {
    /// Echoes under a configuration.
    pub fn new(cfg: EchoConfig) -> Echo {
        Echo {
            cfg,
            walks: 0,
            verified: 0,
            empty: 0,
            dropped: 0,
            drawn: 0,
            kept_in_silence: 0,
            refined: 0,
            heard: 0,
            heard_home: 0,
            heard_understood: 0,
            echoed: HashMap::new(),
        }
    }

    /// The configuration.
    pub fn config(&self) -> &EchoConfig {
        &self.cfg
    }

    /// Of the thoughts that took a sign from an echo and came home, the
    /// share that came home in the sign's class.
    pub fn heard_success_rate(&self) -> f64 {
        if self.heard_home == 0 {
            0.0
        } else {
            self.heard_understood as f64 / self.heard_home as f64
        }
    }

    /// A summary.
    pub fn report(&self, net: &PathNet, lex: &Lexicon) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "  echoes: {} walks ({} found the source, {} found nothing), {} signs drawn, {} let go, {} epochs kept in silence, {} walks refined a glyph; {} thoughts heard a sign on the way, {} came home, {:.0}% in its class",
            self.walks,
            self.verified,
            self.empty,
            self.drawn,
            self.dropped,
            self.kept_in_silence,
            self.refined,
            self.heard,
            self.heard_home,
            100.0 * self.heard_success_rate()
        );
        let mut signs: Vec<(usize, u64)> = self.echoed.iter().map(|(&k, &n)| (k, n)).collect();
        signs.sort_by(|a, b| b.1.cmp(&a.1));
        for (k, n) in signs.into_iter().take(6) {
            let Some(symbol) = net.symbols().get(k) else {
                continue;
            };
            let meaning = lex
                .meaning(k)
                .first()
                .map(|(b, p)| {
                    let c = lex.block_centre(*b);
                    format!("near ({},{}) {:.0}%", c.x, c.y, 100.0 * p)
                })
                .unwrap_or_else(|| "no outcome yet".to_string());
            let _ = writeln!(
                out,
                "    #{k} [{}] echoed {} times, glyph {:.0} cells: {}",
                net.show(&symbol.word),
                n,
                symbol.glyph.length(),
                meaning
            );
        }
        out
    }
}
