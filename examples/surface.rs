//! The surface flat in front of an ant, the separable entropy channels that
//! deform it, and the sucker that selects on it.
//!
//!     cargo run --release --example surface [ticks] [seeds]

use ant_simulator::prelude::*;

fn base_config() -> SimConfig {
    let mut cfg = SimConfig::default();
    cfg.world.seed = Some(2024);
    cfg.nest.initial_satiation = 0.1;
    cfg
}

/// Derived measurements of one run, so several seeds can be averaged.
#[derive(Clone, Copy, Default)]
struct Metrics {
    target: f64,
    drawn: f64,
    temper: f64,
    smooth: f64,
    rough: f64,
    field: f64,
    select: f64,
    turn_entropy: f64,
    straight: f64,
    reversal: f64,
    revisit: f64,
    efficiency: f64,
    delivered: f64,
    deaths: f64,
}

impl Metrics {
    fn of(s: &Stats) -> Self {
        let c = s.path.ledger.contributions();
        Metrics {
            target: s.mean_entropy(),
            drawn: s.mean_selected_entropy(),
            temper: c.tempering,
            smooth: c.smoothing,
            rough: c.roughening,
            field: c.field,
            select: c.selection,
            turn_entropy: s.path.turn_entropy(),
            straight: s.path.straight_rate(),
            reversal: s.path.reversal_rate(),
            revisit: s.path.revisit_rate(),
            efficiency: s.path.trip_efficiency(),
            delivered: s.food_delivered as f64,
            deaths: s.deaths as f64,
        }
    }

    fn mean(all: &[Metrics]) -> Self {
        let n = all.len().max(1) as f64;
        let mut m = Metrics::default();
        for x in all {
            m.target += x.target / n;
            m.drawn += x.drawn / n;
            m.temper += x.temper / n;
            m.smooth += x.smooth / n;
            m.rough += x.rough / n;
            m.field += x.field / n;
            m.select += x.select / n;
            m.turn_entropy += x.turn_entropy / n;
            m.straight += x.straight / n;
            m.reversal += x.reversal / n;
            m.revisit += x.revisit / n;
            m.efficiency += x.efficiency / n;
            m.delivered += x.delivered / n;
            m.deaths += x.deaths / n;
        }
        m
    }
}

fn run(
    deformation: Deformation,
    selection: Selection,
    entropy: f64,
    ticks: usize,
    seeds: u64,
) -> Metrics {
    let runs: Vec<Metrics> = (1..=seeds)
        .map(|seed| {
            let config = SimConfig {
                selection,
                instinct: BehavioralSurface::instinct().with_deformation(deformation),
                ..base_config()
            };
            let mut sim = Simulation::new(config, seed);
            sim.hierarchy_mut().node_mut(0).surface.entropy = EntropyControl::absolute(entropy);
            sim.run(ticks);
            Metrics::of(sim.stats())
        })
        .collect();
    Metrics::mean(&runs)
}

fn header() {
    println!(
        "{:<24} {:>6} {:>6} | {:>6} {:>6} {:>6} {:>6} {:>6} | {:>6} {:>5} {:>5} {:>5} {:>5} | {:>6} {:>4}",
        "setting", "target", "drawn", "temper", "smooth", "rough", "field", "select", "turnH", "str%",
        "rev%", "revis", "eff", "deliv", "died"
    );
}

fn print_row(label: &str, m: &Metrics) {
    println!(
        "{:<24} {:>6.3} {:>6.3} | {:>6.3} {:>+6.3} {:>+6.3} {:>+6.3} {:>+6.3} | {:>6.3} {:>5.1} {:>5.1} {:>5.2} {:>5.2} | {:>6.0} {:>4.1}",
        label,
        m.target,
        m.drawn,
        m.temper,
        m.smooth,
        m.rough,
        m.field,
        m.select,
        m.turn_entropy,
        100.0 * m.straight,
        100.0 * m.reversal,
        m.revisit,
        m.efficiency,
        m.delivered,
        m.deaths
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let ticks: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1200);
    let seeds: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    let sucker = Selection::Sucker { reach: 8 };
    let none = Deformation::none();
    let smooth = Deformation {
        smooth: 1.5,
        ..none
    };
    let rough = Deformation { rough: 1.5, ..none };
    let both = Deformation {
        smooth: 1.5,
        rough: 1.5,
        ..none
    };

    // 1. Watch one ant's path surface scroll past under the sucker.
    println!("== one ant's path surface (sucker, reach 8, smoothing on) ==");
    println!("rows: the ring in front of the ant, shaded by the probability each");
    println!("direction was drawn with; [..] marks the choice; trail = the sucker's walk\n");
    let config = SimConfig {
        selection: sucker,
        record_surface: Some(0),
        instinct: BehavioralSurface::instinct().with_deformation(smooth),
        ..base_config()
    };
    let mut sim = Simulation::new(config, 5);
    sim.run(900);
    println!("{}", render_surface(sim.surface_trace(), 24));

    // 2. Same entropy budget, different geometry.
    println!(
        "== same budget (0.35 of max entropy), different shape of disorder ({ticks} ticks, mean of {seeds} seeds) =="
    );
    println!("entropy target vs entropy drawn from; the ledger (within-decision terms sum to");
    println!("'drawn'; 'field' is disorder between decisions); then the geometry of the paths\n");
    header();
    for (name, deformation) in [
        ("flat", none),
        ("smooth", smooth),
        ("rough", rough),
        ("smooth+rough", both),
    ] {
        for (mode, selection) in [("softmax", Selection::Softmax), ("sucker", sucker)] {
            let m = run(deformation, selection, 0.35, ticks, seeds);
            print_row(&format!("{name} / {mode}"), &m);
        }
    }

    // 3. How far the sucker may crawl.
    println!("\n== sucker reach (budget 0.35, no deformation) ==");
    header();
    for reach in [0usize, 1, 2, 4, 8, 16, 32] {
        let m = run(none, Selection::Sucker { reach }, 0.35, ticks, seeds);
        print_row(&format!("reach {reach}"), &m);
    }

    // 4. Budget sweep under both selection mechanisms.
    println!("\n== entropy budget × selection (no deformation) ==");
    header();
    for fraction in [0.05, 0.15, 0.35, 0.6, 0.85] {
        for (mode, selection) in [("softmax", Selection::Softmax), ("sucker", sucker)] {
            let m = run(none, selection, fraction, ticks, seeds);
            print_row(&format!("budget {fraction:.2} / {mode}"), &m);
        }
    }
}
