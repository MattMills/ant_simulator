//! Throughput and scaling of the simulation.
//!
//!     cargo bench            # the full scan (a few minutes)
//!     cargo bench -- quick   # a shorter one
//!     cargo bench -- phases  # one colony's phase breakdown only
//!
//! Each row times a hungry colony kept foraging for ten simulated minutes
//! (the median of several runs) and divides the tick among its phases.

use ant_simulator::prelude::*;

fn main() {
    let quick = std::env::args().any(|a| a == "quick");
    let phases_only = std::env::args().any(|a| a == "phases");
    let repeats = if quick { 3 } else { 5 };
    let base = Workload::default();

    println!(
        "== one colony, phase by phase: {} ants on {}×{} cells, history and queen on ==",
        base.ants, base.width, base.height
    );
    let m = measure(
        &Workload {
            history: true,
            mind: true,
            ..base.clone()
        },
        repeats,
    );
    println!(
        "{:.0} ticks/s, {:.0} ant-ticks/s, {:.0} ns per ant-tick, {:.0}% of ant-time outside",
        m.ticks_per_s,
        m.ant_ticks_per_s,
        m.nanos_per_ant_tick(),
        100.0 * m.outside_fraction
    );
    println!("{}", m.profile.table());
    if phases_only {
        return;
    }

    println!(
        "== colony size at a fixed world ({}×{} cells) ==",
        base.width, base.height
    );
    let ants: &[usize] = if quick {
        &[25, 100, 400]
    } else {
        &[25, 50, 100, 200, 400, 800, 1600]
    };
    println!("{}", ant_scan(ants, &base, repeats).table());

    println!("== world area at a fixed colony ({} ants) ==", base.ants);
    let sides: &[usize] = if quick {
        &[32, 128]
    } else {
        &[32, 64, 128, 256]
    };
    println!("{}", area_scan(sides, &base, repeats).table());

    println!("== the same with the movement history and the queen on ==");
    let with_mind = Workload {
        history: true,
        mind: true,
        ..base.clone()
    };
    println!("{}", area_scan(sides, &with_mind, repeats).table());
}
