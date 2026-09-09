//! Scale analysis: how foraging organises with colony size, how the
//! colony's output scales, and how the hive's memory depends on colony
//! size and on the grain of the reading.
//!
//!     cargo run --release --example scale [minutes]

use ant_simulator::prelude::*;

fn main() {
    let minutes: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20.0);
    let stem = 6;
    let distance = 20;

    for (species, only_trail) in [
        (Species::pharaoh(), true),
        (Species::pharaoh(), false),
        (Species::lasius_niger(), false),
    ] {
        println!(
            "== foraging organisation against colony size: {} at a single feeder {} cm beyond a {} cm entrance corridor, {}; {:.0} min to settle, {:.0} min measured ==",
            species.name,
            distance * 2,
            stem * 2,
            if only_trail {
                "the trail the only way to it (no route or site memory, no smell)"
            } else {
                "with individual navigation (path integration, site and route memory, smell)"
            },
            minutes,
            minutes
        );
        let outcomes = run_colony_size_scan(&SizeScan {
            species: species.clone(),
            stem,
            distance,
            settle_s: minutes * 60.0,
            window_s: minutes * 60.0,
            trail_only: only_trail,
            ..SizeScan::default()
        });
        println!(
            "{:>6} {:>8} {:>8} {:>10} {:>8} {:>9} {:>10} {:>8}",
            "ants", "loads", "failed", "per ant/h", "outside", "trail mid", "directness", "ordered"
        );
        for o in &outcomes {
            println!(
                "{:>6} {:>8} {:>8} {:>10.2} {:>7.0}% {:>9.2} {:>10.2} {:>7.0}%",
                o.ants,
                o.delivered,
                o.failed_trips,
                o.per_capita_per_hour,
                100.0 * o.outside_fraction,
                o.trail_mid,
                o.directness,
                100.0 * o.ordered
            );
        }
        let n: Vec<f64> = outcomes.iter().map(|o| o.ants as f64).collect();
        let loads: Vec<f64> = outcomes.iter().map(|o| o.delivered as f64).collect();
        println!(
            "colony output scales as N^{:.2} over this range (1 would be proportional; below 1, per-capita output falls with size)\n",
            exponent(&n, &loads)
        );
    }

    println!("== the hive's memory against colony size (sector 16, multi-scale) and against the grain (100 workers) ==");
    let base = MemoryProbeConfig {
        epochs: 120,
        ..MemoryProbeConfig::default()
    };
    println!(
        "{} epochs of {:.0} s after a {:.0} min warm-up; held-out R² of the residual readout by lag",
        base.epochs, base.epoch_s, base.warmup_s / 60.0
    );
    let outcomes = run_memory_scaling(&base, &[25, 50, 100, 200], &[8, 16, 32]);
    println!(
        "{:>6} {:>7} {:>11} {:>7} {:>7} {:>7} {:>9} {:>12}",
        "ants", "sector", "multiscale", "lag 1", "lag 2", "lag 3", "capacity", "corr(straight)"
    );
    for o in &outcomes {
        let lag = |l: usize| o.residual.by_lag.get(l).copied().unwrap_or(0.0);
        println!(
            "{:>6} {:>7} {:>11} {:>7.2} {:>7.2} {:>7.2} {:>9.2} {:>12.2}",
            o.ants,
            o.sector,
            if o.multiscale { "yes" } else { "no" },
            lag(0),
            lag(1),
            lag(2),
            o.residual.total,
            o.straightness_correlation
        );
    }
}
