//! The classic experiments, replicated: double bridge, equal bridge, two
//! sources of different quality, hunger response, division of labour.
//!
//!     cargo run --release --example experiments [replicates] [minutes]

use ant_simulator::prelude::*;

fn main() {
    let mut args = std::env::args().skip(1);
    let replicates: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let minutes: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(30.0);
    let seconds = minutes * 60.0;
    let window = (seconds / 3.0).min(600.0);
    let seeds: Vec<u64> = (1..=replicates).collect();
    let ants = 80;

    println!("== double bridge, long branch twice the short one (Goss et al. 1989) ==");
    println!("{ants} ants, {minutes:.0} min, traffic measured over the last {:.0} min, {replicates} replicates\n", window / 60.0);
    for species in [Species::argentine(), Species::lasius_niger()] {
        let name = species.name.clone();
        let outcomes = run_double_bridge(
            &BridgeSpec::ratio_two(),
            &species,
            ants,
            seconds,
            window,
            &seeds,
        );
        let fractions: Vec<f64> = outcomes.iter().map(|o| o.short_fraction).collect();
        let s = summarize(&fractions);
        println!(
            "{name:<22} short-branch share {:.2} ± {:.2}; majority short in {:.0}% of runs; decided in {:.0}%",
            s.mean, s.std, 100.0 * s.majority, 100.0 * s.decided
        );
        for (seed, o) in seeds.iter().zip(&outcomes) {
            println!(
                "   seed {seed:>2}: short {:>4} long {:>4} → {:.2}; trail short {:>6.0} long {:>6.0}; delivered {}",
                o.short_crossings, o.long_crossings, o.short_fraction, o.trail_short, o.trail_long, o.delivered
            );
        }
    }

    println!("\n== equal branches: symmetry breaking (Deneubourg et al. 1990) ==");
    let long_run = seconds.max(45.0 * 60.0);
    for (label, pure) in [
        ("full behavioural model", false),
        (
            "pure pheromone feedback (the 1990 model's assumptions)",
            true,
        ),
    ] {
        let fractions: Vec<f64> = seeds
            .iter()
            .map(|&seed| {
                let mut cfg = experiment_config(
                    Species::argentine(),
                    double_bridge(&BridgeSpec::equal()),
                    ants,
                );
                if pure {
                    pure_pheromone_feedback(&mut cfg);
                }
                run_double_bridge_configured(cfg, long_run, window, seed).short_fraction
            })
            .collect();
        let s = summarize(&fractions);
        println!(
            "{label}: after {:.0} min, lower-branch share {:.2} ± {:.2}; decided (one branch > 80%) in {:.0}% of runs",
            long_run / 60.0, s.mean, s.std, 100.0 * s.decided
        );
    }

    println!("\n== two sources at equal distance, quality 1.0 vs 0.1 (Beckers et al. 1990) ==");
    let mut fractions = Vec::new();
    for &seed in &seeds {
        let o = run_two_sources_once(12, 1.0, 0.1, Species::lasius_niger(), ants, seconds, seed);
        fractions.push(o.fraction_a);
        println!(
            "   seed {seed:>2}: rich {:>6.1} µl poor {:>6.1} µl → share from rich {:.2} ({} loads)",
            o.volume_a, o.volume_b, o.fraction_a, o.delivered
        );
    }
    let s = summarize(&fractions);
    println!(
        "share of solution taken from the richer source {:.2} ± {:.2}; majority rich in {:.0}% of runs",
        s.mean,
        s.std,
        100.0 * s.majority
    );

    println!("\n== foraging activity against colony satiation (Mailleux et al. 2003) ==");
    for o in run_hunger_response(
        &Species::lasius_niger(),
        60,
        &[0.05, 0.3, 0.6, 0.9, 1.0],
        seconds.min(1200.0),
        1,
    ) {
        println!(
            "   satiation {:.2}: {:.0}% of ant-time outside, {} loads delivered",
            o.satiation,
            100.0 * o.foraging_fraction,
            o.delivered
        );
    }

    println!("\n== division of labour with and without threshold reinforcement (Theraulaz et al. 1998) ==");
    let o = run_division_of_labor(&Species::lasius_niger(), 60, seconds, 1);
    println!(
        "   index with reinforcement {:.2}, without {:.2}",
        o.with_reinforcement, o.without_reinforcement
    );
}
