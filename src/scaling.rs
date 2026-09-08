//! Benchmarks and computational scale analysis: how the cost of a tick is
//! divided among the simulation's phases, and how it grows with the
//! colony's size and the world's area.
//!
//! Every [`Simulation`] keeps a [`Profile`] of wall time per phase. A
//! [`Workload`] describes a colony to time (size, world, duration, what is
//! switched on), [`measure`] runs it a few times and keeps the median, and
//! [`ant_scan`] and [`area_scan`] sweep one dimension and fit the
//! empirical exponent of the time per tick against it ([`exponent`]):
//! one for a linear cost, two for a quadratic one.
//!
//! ```text
//! cargo bench            # the scan at full size
//! cargo bench -- quick   # a shorter one
//! cargo bench -- phases  # one colony's phase breakdown only
//! ```

use crate::colony::{SimConfig, Simulation};
use crate::geometry::Position;
use crate::hive::{HistoryConfig, QueenConfig};
use crate::memo::{MemoConfig, TransitConfig};
use crate::pipeline::PipelineConfig;
use crate::species::Species;
use std::fmt::Write as _;
use std::time::{Duration, Instant};

/// A phase of the tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// The movement decisions: perceiving the ring of headings, scoring
    /// it, deforming it with the entropy budget, and selecting.
    Decisions,
    /// The rest of the ants' tick: recompiling policies when dials
    /// changed, indexing the workers inside, moves, contacts, tasks.
    Ants,
    /// Evaporation and diffusion of every chemical channel over the grid.
    Pheromones,
    /// Renewal of food and emission of its odour over the grid.
    Food,
    /// Sharing inside the nest, brood development, egg laying.
    Nest,
    /// Forgetting in the movement history.
    History,
    /// The queen's epoch: reading the field, fitting nothing, expressing.
    Queen,
    /// Counters, snapshots and the rest of the tick.
    Other,
}

impl Phase {
    /// Every phase, in table order.
    pub const ALL: [Phase; 8] = [
        Phase::Decisions,
        Phase::Ants,
        Phase::Pheromones,
        Phase::Food,
        Phase::Nest,
        Phase::History,
        Phase::Queen,
        Phase::Other,
    ];

    /// Number of phases.
    pub const COUNT: usize = 8;

    /// Index in table order.
    pub fn index(self) -> usize {
        Phase::ALL.iter().position(|p| *p == self).expect("listed")
    }

    /// Short name.
    pub fn name(self) -> &'static str {
        match self {
            Phase::Decisions => "decisions",
            Phase::Ants => "ants",
            Phase::Pheromones => "pheromones",
            Phase::Food => "food",
            Phase::Nest => "nest",
            Phase::History => "history",
            Phase::Queen => "queen",
            Phase::Other => "other",
        }
    }
}

/// Wall time spent in each phase of the tick, accumulated over a run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Profile {
    /// Ticks profiled.
    pub ticks: u64,
    /// Time per phase.
    pub totals: [Duration; Phase::COUNT],
}

impl Profile {
    /// Add one phase's time for the current tick.
    pub fn add(&mut self, phase: Phase, elapsed: Duration) {
        self.totals[phase.index()] += elapsed;
    }

    /// Time in one phase.
    pub fn time(&self, phase: Phase) -> Duration {
        self.totals[phase.index()]
    }

    /// Time over all phases.
    pub fn total(&self) -> Duration {
        self.totals.iter().sum()
    }

    /// Share of the total spent in one phase.
    pub fn fraction(&self, phase: Phase) -> f64 {
        let total = self.total().as_secs_f64();
        if total <= 0.0 {
            0.0
        } else {
            self.time(phase).as_secs_f64() / total
        }
    }

    /// Mean time per tick in one phase, microseconds.
    pub fn micros_per_tick(&self, phase: Phase) -> f64 {
        if self.ticks == 0 {
            0.0
        } else {
            self.time(phase).as_secs_f64() * 1e6 / self.ticks as f64
        }
    }

    /// The breakdown as a table.
    pub fn table(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "{:<12} {:>10} {:>7}", "phase", "µs/tick", "share");
        for phase in Phase::ALL {
            let _ = writeln!(
                out,
                "{:<12} {:>10.1} {:>6.1}%",
                phase.name(),
                self.micros_per_tick(phase),
                100.0 * self.fraction(phase)
            );
        }
        let _ = writeln!(
            out,
            "{:<12} {:>10.1}",
            "total",
            Phase::ALL
                .iter()
                .map(|&p| self.micros_per_tick(p))
                .sum::<f64>()
        );
        out
    }
}

/// A colony to time.
#[derive(Clone, Debug, PartialEq)]
pub struct Workload {
    /// Simulated workers.
    pub ants: usize,
    /// Grid width, cells.
    pub width: usize,
    /// Grid height, cells.
    pub height: usize,
    /// Simulated time, seconds.
    pub seconds: f64,
    /// Whether the movement history is kept.
    pub history: bool,
    /// Whether the queen thinks (needs the history).
    pub mind: bool,
    /// Whether transits through the memo's nodes are memoized (keeps the
    /// history too, for the invariance check).
    pub memoize: bool,
    /// Ticks between steps of the chemical kinetics (one is exact).
    pub kinetics_stride: u32,
    /// Whether the decision pipeline holds decisions for a horizon on
    /// invariant ground (keeps the history too).
    pub pipeline: bool,
    /// The pipeline's frame budget of decisions beyond those due (zero
    /// for no limit).
    pub frame_budget: usize,
    /// The species.
    pub species: Species,
    /// Seed of the map and the run.
    pub seed: u64,
}

impl Default for Workload {
    fn default() -> Self {
        Workload {
            ants: 100,
            width: 64,
            height: 40,
            seconds: 600.0,
            history: false,
            mind: false,
            memoize: false,
            kinetics_stride: 1,
            pipeline: false,
            frame_budget: 0,
            species: Species::lasius_niger(),
            seed: 7,
        }
    }
}

impl Workload {
    /// Cells in the grid.
    pub fn cells(&self) -> usize {
        self.width * self.height
    }

    /// The configuration: a hungry colony kept foraging by renewing
    /// sources (as many clusters as the area warrants) and a reserve
    /// drain, without mortality, so that the activity mix is
    /// representative of a working colony whatever the size.
    pub fn config(&self) -> SimConfig {
        let mut species = self.species.clone();
        species.consumption_mg_per_ant_per_s = 0.5 / 3600.0;
        let mut cfg = SimConfig::for_species(species);
        cfg.ants = self.ants;
        cfg.world.width = self.width;
        cfg.world.height = self.height;
        cfg.world.nest = Position::new(self.width as i32 / 2, self.height as i32 / 2);
        // A nest that grows with the colony: about twenty workers per cell
        // when everyone is in, three cells of radius at least.
        cfg.world.nest_radius = ((self.ants as f64 / 20.0).sqrt() / 2.0).ceil().max(3.0) as i32;
        cfg.world.seed = Some(self.seed);
        cfg.world.kinetics_stride = self.kinetics_stride.max(1);
        if let Some(food) = cfg.world.random_food.as_mut() {
            food.renewal_ul_per_s = 0.02;
            food.clusters = ((3 * self.cells()) as f64 / 2560.0).round().max(3.0) as usize;
        }
        cfg.nest.initial_satiation = 0.1;
        cfg.nest.mortality = false;
        cfg.nest.max_ants = cfg.nest.max_ants.max(self.ants);
        cfg.nest.log_every_s = 0.0;
        cfg.history = if self.history || self.mind || self.memoize || self.pipeline {
            Some(HistoryConfig::default())
        } else {
            None
        };
        cfg.pipeline = if self.pipeline {
            Some(PipelineConfig {
                budget: self.frame_budget,
                ..PipelineConfig::default()
            })
        } else {
            None
        };
        cfg.memo = if self.memoize {
            Some(MemoConfig {
                transits: Some(TransitConfig::default()),
                ..MemoConfig::default()
            })
        } else {
            None
        };
        cfg.mind = if self.mind {
            Some(QueenConfig::default())
        } else {
            None
        };
        cfg
    }
}

/// What timing a workload found.
#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    /// The workload.
    pub workload: Workload,
    /// Ticks run.
    pub ticks: u64,
    /// Wall time of the median run, seconds.
    pub wall_s: f64,
    /// Ticks per second of wall time.
    pub ticks_per_s: f64,
    /// Ant-ticks per second of wall time.
    pub ant_ticks_per_s: f64,
    /// Share of ant-time spent outside the nest (walking is the costly
    /// part of an ant's tick).
    pub outside_fraction: f64,
    /// Share of the movement decisions stood in for by memoized transits.
    pub replayed_fraction: f64,
    /// Share of the steps taken without a decision, held by the pipeline.
    pub held_fraction: f64,
    /// Loads delivered into the nest: what the colony achieved, to set a
    /// memoized or strided run against the full one.
    pub delivered: u64,
    /// Mean entropy of the movement decisions, nats: how the colony
    /// behaved, for the same comparison.
    pub entropy: f64,
    /// Share of the kinetics grain's nodes at which the trail channel
    /// is kept at cell resolution at the end.
    pub active: f64,
    /// The phase breakdown of the median run.
    pub profile: Profile,
}

impl Measurement {
    /// Wall time per tick, microseconds.
    pub fn micros_per_tick(&self) -> f64 {
        if self.ticks == 0 {
            0.0
        } else {
            self.wall_s * 1e6 / self.ticks as f64
        }
    }

    /// Wall time per ant-tick, nanoseconds.
    pub fn nanos_per_ant_tick(&self) -> f64 {
        let ant_ticks = self.ticks as f64 * self.workload.ants as f64;
        if ant_ticks <= 0.0 {
            0.0
        } else {
            self.wall_s * 1e9 / ant_ticks
        }
    }
}

/// Time a workload `repeats` times and keep the median run.
pub fn measure(workload: &Workload, repeats: usize) -> Measurement {
    let mut runs: Vec<(f64, Simulation)> = Vec::with_capacity(repeats.max(1));
    for _ in 0..repeats.max(1) {
        let mut sim = Simulation::new(workload.config(), workload.seed);
        let start = Instant::now();
        sim.run_seconds(workload.seconds);
        let wall = start.elapsed().as_secs_f64();
        runs.push((wall, sim));
    }
    runs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let (wall_s, sim) = runs.swap_remove(runs.len() / 2);
    let ticks = sim.tick();
    Measurement {
        workload: workload.clone(),
        ticks,
        wall_s,
        ticks_per_s: if wall_s > 0.0 {
            ticks as f64 / wall_s
        } else {
            0.0
        },
        ant_ticks_per_s: if wall_s > 0.0 {
            ticks as f64 * workload.ants as f64 / wall_s
        } else {
            0.0
        },
        outside_fraction: sim.stats().foraging_fraction(),
        replayed_fraction: {
            let s = sim.stats();
            if s.decisions == 0 {
                0.0
            } else {
                s.decisions_replayed as f64 / s.decisions as f64
            }
        },
        held_fraction: sim.stats().frames.held_share(),
        delivered: sim.stats().food_delivered,
        entropy: sim.stats().mean_entropy(),
        active: sim.world().active_share(crate::pheromone::Pheromone::Trail),
        profile: sim.profile().clone(),
    }
}

/// Slope of `log y` against `log x` by least squares: the empirical
/// exponent of a power law (zero when there is nothing to fit).
pub fn exponent(xs: &[f64], ys: &[f64]) -> f64 {
    let pairs: Vec<(f64, f64)> = xs
        .iter()
        .zip(ys)
        .filter(|(x, y)| **x > 0.0 && **y > 0.0)
        .map(|(x, y)| (x.ln(), y.ln()))
        .collect();
    let n = pairs.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mx = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx: f64 = pairs.iter().map(|p| (p.0 - mx).powi(2)).sum();
    let sxy: f64 = pairs.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    if sxx <= 1e-18 {
        0.0
    } else {
        sxy / sxx
    }
}

/// A sweep of one dimension of a workload.
#[derive(Clone, Debug, PartialEq)]
pub struct Scan {
    /// The dimension swept.
    pub dimension: String,
    /// Its values.
    pub x: Vec<f64>,
    /// The timing at each.
    pub measurements: Vec<Measurement>,
}

impl Scan {
    /// Empirical exponent of the wall time per tick against the
    /// dimension.
    pub fn exponent(&self) -> f64 {
        let y: Vec<f64> = self
            .measurements
            .iter()
            .map(|m| m.micros_per_tick())
            .collect();
        exponent(&self.x, &y)
    }

    /// Empirical exponent of one phase's time per tick against the
    /// dimension.
    pub fn phase_exponent(&self, phase: Phase) -> f64 {
        let y: Vec<f64> = self
            .measurements
            .iter()
            .map(|m| m.profile.micros_per_tick(phase))
            .collect();
        exponent(&self.x, &y)
    }

    /// The scan as a table: throughput and the phase breakdown per row.
    pub fn table(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "{:>8} {:>8} {:>9} {:>11} {:>8} {:>8} {:>8} {:>5} {:>9} {:>7} {:>6} | {}",
            self.dimension,
            "cells",
            "ticks/s",
            "ant-ticks/s",
            "ns/ant",
            "outside",
            "replayed",
            "held",
            "delivered",
            "entropy",
            "active",
            Phase::ALL
                .iter()
                .map(|p| format!("{:>7}", p.name()))
                .collect::<Vec<_>>()
                .join(" ")
        );
        for (x, m) in self.x.iter().zip(&self.measurements) {
            let _ = writeln!(
                out,
                "{:>8} {:>8} {:>9.0} {:>11.0} {:>8.0} {:>7.0}% {:>7.0}% {:>4.0}% {:>9} {:>7.3} {:>5.0}% | {}",
                x,
                m.workload.cells(),
                m.ticks_per_s,
                m.ant_ticks_per_s,
                m.nanos_per_ant_tick(),
                100.0 * m.outside_fraction,
                100.0 * m.replayed_fraction,
                100.0 * m.held_fraction,
                m.delivered,
                m.entropy,
                100.0 * m.active,
                Phase::ALL
                    .iter()
                    .map(|&p| format!("{:>6.0}%", 100.0 * m.profile.fraction(p)))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        let _ = writeln!(
            out,
            "exponent of time per tick against {}: {:.2} (phases: {})",
            self.dimension,
            self.exponent(),
            Phase::ALL
                .iter()
                .filter(|&&p| p != Phase::Other)
                .map(|&p| format!("{} {:.2}", p.name(), self.phase_exponent(p)))
                .collect::<Vec<_>>()
                .join(", ")
        );
        out
    }
}

/// Sweep the colony's size at a fixed world.
pub fn ant_scan(ants: &[usize], base: &Workload, repeats: usize) -> Scan {
    let measurements = ants
        .iter()
        .map(|&n| {
            measure(
                &Workload {
                    ants: n,
                    ..base.clone()
                },
                repeats,
            )
        })
        .collect();
    Scan {
        dimension: "ants".into(),
        x: ants.iter().map(|&n| n as f64).collect(),
        measurements,
    }
}

/// Sweep the world's area (square grids of the given sides) at a fixed
/// colony.
pub fn area_scan(sides: &[usize], base: &Workload, repeats: usize) -> Scan {
    let measurements = sides
        .iter()
        .map(|&s| {
            measure(
                &Workload {
                    width: s,
                    height: s,
                    ..base.clone()
                },
                repeats,
            )
        })
        .collect();
    Scan {
        dimension: "cells".into(),
        x: sides.iter().map(|&s| (s * s) as f64).collect(),
        measurements,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponent_recovers_power_laws() {
        let xs = [10.0, 20.0, 40.0, 80.0];
        let linear: Vec<f64> = xs.iter().map(|x| 3.0 * x).collect();
        let quadratic: Vec<f64> = xs.iter().map(|x| 0.1 * x * x).collect();
        assert!((exponent(&xs, &linear) - 1.0).abs() < 1e-9);
        assert!((exponent(&xs, &quadratic) - 2.0).abs() < 1e-9);
        assert_eq!(exponent(&[1.0], &[1.0]), 0.0);
        assert_eq!(exponent(&xs, &[1.0, 1.0, 1.0, 1.0]), 0.0);
    }

    #[test]
    fn profile_divides_the_tick() {
        let mut p = Profile {
            ticks: 4,
            ..Profile::default()
        };
        p.add(Phase::Ants, Duration::from_micros(300));
        p.add(Phase::Pheromones, Duration::from_micros(100));
        assert_eq!(p.total(), Duration::from_micros(400));
        assert!((p.fraction(Phase::Ants) - 0.75).abs() < 1e-12);
        assert!((p.micros_per_tick(Phase::Pheromones) - 25.0).abs() < 1e-9);
        assert!(p.table().contains("ants"));
        assert_eq!(Phase::Queen.index(), 6);
    }

    #[test]
    fn a_small_workload_is_measured_and_profiled() {
        let w = Workload {
            ants: 20,
            width: 32,
            height: 24,
            seconds: 20.0,
            history: true,
            mind: true,
            ..Workload::default()
        };
        let cfg = w.config();
        assert_eq!(cfg.ants, 20);
        assert!(cfg.history.is_some() && cfg.mind.is_some());
        assert!(!cfg.nest.mortality);
        let m = measure(&w, 2);
        assert_eq!(m.ticks, 20);
        assert!(m.wall_s > 0.0 && m.ticks_per_s > 0.0);
        assert!(m.nanos_per_ant_tick() > 0.0);
        assert_eq!(m.profile.ticks, 20);
        // The phases account for most of the wall time.
        let profiled = m.profile.total().as_secs_f64();
        assert!(
            profiled > 0.0 && profiled <= m.wall_s * 1.5,
            "{profiled} of {}",
            m.wall_s
        );
        assert!(m.profile.time(Phase::History) > Duration::ZERO);
        let scan = ant_scan(&[10, 20], &w, 1);
        assert_eq!(scan.measurements.len(), 2);
        assert!(scan.table().contains("exponent"));
        let area = area_scan(&[16, 32], &w, 1);
        assert_eq!(area.x, vec![256.0, 1024.0]);
    }
}
