//! Hive cognitive geometry: the colony's movement history as a
//! path-topological structure collapsed over time, and a queen who thinks
//! through it.
//!
//! A colony is kept foraging for hours. Each epoch its queen thinks a
//! random ±1 and writes it into the root entropy dial, so her thought is
//! embedded in the colony's non-invariant movement. Readouts of the field
//! then retrodict her past thoughts, lag by lag, from the invariant
//! skeleton, from the residual over it, and from both; and with recall
//! feeding her next thought, an alternation is sustained through the
//! colony alone.
//!
//!     cargo run --release --example hive [epochs] [epoch_seconds]

use ant_simulator::prelude::*;

fn main() {
    let mut args = std::env::args().skip(1);
    let epochs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(180);
    let epoch_s: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(60.0);
    let cfg = MemoryProbeConfig {
        epochs,
        epoch_s,
        ..MemoryProbeConfig::default()
    };

    println!("== the memory probe ==");
    println!(
        "{} workers kept foraging by renewing sources ({} µl/s per cell) and a reserve drain of {} mg per worker per hour;",
        cfg.ants, cfg.renewal_ul_per_s, cfg.drain_mg_per_ant_per_h
    );
    println!(
        "{:.0} min warm-up, then {} epochs of {:.0} s in which the queen thinks a random ±1 and sets the root temperature to e^({} × thought);",
        cfg.warmup_s / 60.0, cfg.epochs, cfg.epoch_s, cfg.expression
    );
    println!(
        "the history is read at {}-cell sectors and the coarser grains above them, fast half-life one epoch, slow half-life {:.0} min.\n",
        cfg.history.sector, cfg.history.slow_half_life_s / 60.0
    );
    let mut probe = run_memory_probe(&cfg);
    let h = probe.simulation.history().expect("history on");
    println!("invariant flow at the end (arrows: mean direction; ·: flow without one):");
    println!("{}", h.render(Component::Invariant));
    println!("non-invariant residual (the direction the current flow departs in):");
    println!("{}", h.render(Component::Residual));
    let s = &probe.summary;
    println!(
        "moves {}; steady density {:.1} moves/tick; alignment {:.2}; spatial entropy {:.2} nats; residual energy {:.2}; skeleton: {} channel cells in {} channels closing {} loops",
        h.moves, s.density, s.alignment, s.spatial_entropy, s.residual_energy,
        s.topology.cells, s.topology.channels, s.topology.loops
    );
    println!(
        "the dial in force: corr(thought, decision entropy) = {:.2}; reaching the paths: corr(thought, residual straightness of the whole field) = {:.2}; mean current density {:.1} moves/tick\n",
        probe.entropy_correlation, probe.straightness_correlation, probe.mean_density
    );

    println!("== memory capacity: how much of the queen's past thought the field carries ==");
    println!(
        "{} epochs recorded; readouts fitted on the first {:.0}%, held-out R² on the rest (lag 1 is the epoch just ended)",
        probe.epochs,
        100.0 * cfg.train_fraction
    );
    println!(
        "{:<12} {}",
        "component",
        (1..=cfg.lags)
            .map(|l| format!("lag {l:>2}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    for cap in &probe.capacities {
        println!(
            "{:<12} {}   capacity {:.2}",
            format!("{:?}", cap.component).to_lowercase(),
            cap.by_lag
                .iter()
                .map(|r| format!("{r:>6.2}"))
                .collect::<Vec<_>>()
                .join(" "),
            cap.total
        );
    }
    println!("(negative: the readout extrapolates the slow drift of that component and does worse than the mean)\n");

    println!("== self-recursive inclusion: alternation carried by the colony ==");
    println!("the queen recalls her last thought from the field and thinks its opposite, decisively (gain -10), with no other input");
    let recursion = Recursion {
        lag: 1,
        gain: -10.0,
    };
    let loops = 24;
    let mut connected_sim = probe.simulation.clone();
    let connected = run_closed_loop(
        &mut connected_sim,
        loops,
        recursion.clone(),
        cfg.expression,
        cfg.train_fraction,
    );
    let disconnected = run_closed_loop(
        &mut probe.simulation,
        loops,
        recursion,
        0.0,
        cfg.train_fraction,
    );
    let signs = |t: &[f64]| {
        t.iter()
            .map(|s| if *s > 0.0 { '+' } else { '-' })
            .collect::<String>()
    };
    println!("lag-1 readout held-out R² {:.2}", connected.lag1_r2);
    println!(
        "dial connected:    {}  alternations {:>2} of {}, conviction {:.2}",
        signs(&connected.thoughts),
        connected.alternations,
        loops - 1,
        connected.conviction
    );
    println!(
        "dial disconnected: {}  alternations {:>2} of {}, conviction {:.2}",
        signs(&disconnected.thoughts),
        disconnected.alternations,
        loops - 1,
        disconnected.conviction
    );
    println!("(a coin gives about half; with the dial disconnected nothing she thinks reaches the colony, so the field recalls only noise)");
}
