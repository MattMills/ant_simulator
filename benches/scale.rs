//! Throughput and scaling of the simulation.
//!
//!     cargo bench            # the full scan (the better part of an hour)
//!     cargo bench -- quick   # a shorter one
//!     cargo bench -- phases  # one colony's phase breakdown only
//!     cargo bench -- colony  # the colony-scale rows only, approximated against full
//!     cargo bench -- shapes  # larger and shaped arenas, the field's grain against the dense sweep
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
    let shapes_only = std::env::args().any(|a| a == "shapes");
    let repeats = if quick { 3 } else { 5 };
    let base = Workload::default();

    if colony_only {
        colony_scale(&base, quick);
        return;
    }
    if shapes_only {
        shapes(&base, quick);
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
    shapes(&base, quick);
}

/// A shaped arena to time: name, width, height, open ground, nest.
type Shaped = (&'static str, usize, usize, Vec<Rect>, Option<Position>);

/// Larger and shaped arenas: 200 workers for ten minutes, the field at
/// its grain against the dense sweep of every cell.
fn shapes(base: &Workload, quick: bool) {
    println!("== larger and shaped arenas: 200 workers, ten minutes, the field's grain (8 cells) against the dense sweep of every cell ==");
    let side = 512;
    let centre = Position::new(side / 2, side / 2);
    let mut arenas: Vec<Shaped> = vec![
        ("square", 256, 256, Vec::new(), None),
        ("square", 512, 512, Vec::new(), None),
        ("disc", 512, 512, Shapes::disc(centre, 240), None),
        ("cross", 512, 512, Shapes::cross(side, 128), None),
    ];
    if !quick {
        arenas.push(("square", 1024, 1024, Vec::new(), None));
        arenas.push(("strip", 1024, 64, Vec::new(), None));
        arenas.push((
            "ring",
            512,
            512,
            Shapes::ring(centre, 250, 130),
            Some(Position::new(side / 2, side / 2 - 190)),
        ));
        arenas.push((
            "L",
            512,
            512,
            Shapes::l_shape(side, 128),
            Some(Position::new(64, side - 64)),
        ));
        arenas.push((
            "rooms",
            512,
            512,
            Shapes::rooms(side, 4),
            Some(Position::new(192, 192)),
        ));
    }
    println!(
        "{:>7} {:>10} {:>5} | {:>8} {:>9} {:>6} | {:>8} {:>9} {:>6} {:>6} | {:>5}",
        "arena",
        "cells",
        "open",
        "dense/s",
        "kinetics",
        "decide",
        "grain/s",
        "kinetics",
        "trail",
        "decide",
        "gain"
    );
    for (name, width, height, open, nest) in arenas {
        let workload = Workload {
            ants: 200,
            width,
            height,
            seconds: 600.0,
            open,
            nest,
            ..base.clone()
        };
        let dense = measure(
            &Workload {
                kinetics_grain: 0,
                ..workload.clone()
            },
            1,
        );
        let grain = measure(
            &Workload {
                kinetics_grain: 8,
                ..workload.clone()
            },
            1,
        );
        println!(
            "{:>7} {:>10} {:>4.0}% | {:>8.0} {:>9.0} {:>6.0} | {:>8.0} {:>9.0} {:>5.0}% {:>6.0} | {:>5.2}",
            name,
            format!("{width}x{height}"),
            100.0 * dense.open_cells as f64 / (width * height) as f64,
            dense.ticks_per_s,
            dense.profile.micros_per_tick(Phase::Pheromones),
            dense.profile.micros_per_tick(Phase::Decisions),
            grain.ticks_per_s,
            grain.profile.micros_per_tick(Phase::Pheromones),
            100.0 * grain.active,
            grain.profile.micros_per_tick(Phase::Decisions),
            grain.ticks_per_s / dense.ticks_per_s.max(1e-9)
        );
    }
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
