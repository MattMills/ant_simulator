//! Run a colony on instinct and show what the entropy dial does.
//!
//!     cargo run --release --example colony [ticks]

use ant_simulator::prelude::*;

fn main() {
    let ticks: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(600);

    let config = SimConfig {
        world: WorldConfig {
            seed: Some(2024),
            ..WorldConfig::default()
        },
        ..SimConfig::default()
    };

    println!("hierarchy:\n{}", Hierarchy::default().describe());

    let mut sim = Simulation::new(config.clone(), 1);
    let checkpoints = [ticks / 3, 2 * ticks / 3, ticks];
    let mut done = 0;
    for &cp in &checkpoints {
        sim.run(cp - done);
        done = cp;
        println!("{}", render(&sim));
    }

    // The same colony under three settings of the colony-wide entropy dial.
    println!(
        "entropy dial sweep ({} ticks each, same world and seed):",
        ticks
    );
    println!(
        "{:>10} {:>10} {:>10} {:>10} {:>12}",
        "fraction", "delivered", "picked", "deaths", "mean entropy"
    );
    for fraction in [0.02, 0.15, 0.35, 0.6, 0.85, 0.99] {
        let mut sim = Simulation::new(config.clone(), 1);
        sim.hierarchy_mut().node_mut(0).surface.entropy = EntropyControl::absolute(fraction);
        sim.run(ticks);
        let s = sim.stats();
        println!(
            "{:>10.2} {:>10} {:>10} {:>10} {:>12.3}",
            fraction,
            s.food_delivered,
            s.food_picked,
            s.deaths,
            s.mean_entropy()
        );
    }

    // A caste can be made "hotter" than the rest of the colony on its own.
    let mut sim = Simulation::new(config, 1);
    let scouts = sim.hierarchy().find("caste-0").unwrap();
    sim.hierarchy_mut().node_mut(scouts).surface.entropy = EntropyControl::relative(2.5);
    sim.run(ticks);
    println!("\nwith caste-0 at 2.5× the colony's entropy:");
    println!("{}", sim.hierarchy().describe());
    let s = sim.stats();
    for (i, &leaf) in sim.hierarchy().leaves().iter().enumerate() {
        println!(
            "  leaf {:<8} entropy {:.3}  deliveries {}",
            sim.hierarchy().node(leaf).name,
            sim.policies()[i].entropy_fraction,
            s.delivered_by_node[leaf]
        );
    }
}
