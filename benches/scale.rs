//! Throughput and scaling of the simulation.
//!
//!     cargo bench            # the full scan (the better part of an hour)
//!     cargo bench -- quick   # a shorter one
//!     cargo bench -- phases  # one colony's phase breakdown only
//!     cargo bench -- colony  # the colony-scale rows only, approximated against full
//!
//! Each row times a hungry colony kept foraging for ten simulated minutes
//! (the median of several runs) and divides the tick among its phases;
//! the colony-scale rows run an hour and set the deliveries and decision
//! entropy of the colony with memoized transits and the decision
//! pipeline against the full simulation's, with the field stepped every
//! tick and every fourth.

use ant_simulator::prelude::*;

fn main() {
    let quick = std::env::args().any(|a| a == "quick");
    let phases_only = std::env::args().any(|a| a == "phases");
    let colony_only = std::env::args().any(|a| a == "colony");
    let repeats = if quick { 3 } else { 5 };
    let base = Workload::default();

    if colony_only {
        colony_scale(&base, quick);
        return;
    }

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

    colony_scale(&base, quick);
}

/// The colony-scale rows: an hour on a 128×128 world, the full
/// simulation against memoized transits and the decision pipeline, with
/// the field stepped every tick and every fourth.
fn colony_scale(base: &Workload, quick: bool) {
    let big: &[usize] = if quick {
        &[400, 1600]
    } else {
        &[400, 1600, 6400]
    };
    let repeats = if quick { 1 } else { 2 };
    let large = Workload {
        width: 128,
        height: 128,
        seconds: 3600.0,
        history: true,
        ..base.clone()
    };
    println!("== colony scale: a 128×128 world, an hour, in full ==");
    println!("{}", ant_scan(big, &large, repeats).table());
    println!("== the same with memoized transits (kernels learning as they go) and the decision pipeline ==");
    let approximated = Workload {
        memoize: true,
        pipeline: true,
        ..large.clone()
    };
    println!("{}", ant_scan(big, &approximated, repeats).table());
    println!("== the same with the field stepped every 4 ticks as well ==");
    println!(
        "{}",
        ant_scan(
            big,
            &Workload {
                kinetics_stride: 4,
                ..approximated
            },
            repeats
        )
        .table()
    );
}
