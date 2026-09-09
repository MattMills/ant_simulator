//! The nest interior: workers inside have positions, keep to spatial
//! fidelity zones that drift outward with age, hand food on to neighbours
//! only, nurse in the brood chamber, and carry corpses out from where
//! they lie.
//!
//!     cargo run --release --example nest [minutes]

use ant_simulator::prelude::*;

fn main() {
    let minutes: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30.0);
    let mut cfg = SimConfig::default();
    cfg.world.seed = Some(11);
    // A 7 × 7 nest: four depths from the brood chamber to the entrance ring.
    cfg.world.nest_radius = 3;
    cfg.ants = 80;
    cfg.nest.initial_satiation = 0.2;
    cfg.nest.initial_brood_per_ant = 1.0;
    // Nobody dies, so the corpses carried out are the ones placed.
    cfg.nest.mortality = false;
    let mut sim = Simulation::new(cfg, 5);
    // Three corpses in the brood chamber, for the undertakers.
    let nest = sim.world().nest();
    for _ in 0..3 {
        sim.world_mut().add_corpse(nest);
    }
    sim.run_seconds(minutes * 60.0);

    println!("== the nest after {minutes:.0} minutes (: the brood chamber) ==");
    println!("u a forager unloading, n a nurse, v a worker fetching a corpse or leaving, w a worker at rest, + a corpse, : an empty chamber cell, . nobody\n");
    println!("{}", sim.render_nest());
    println!(
        "{:<16} {:>8} {:>9} {:>10} {:>8}",
        "zone", "workers", "age (d)", "crop fill", "nursing"
    );
    let names = ["brood chamber", "between", "entrance ring"];
    for z in sim.nest_profile() {
        println!(
            "{:<16} {:>8} {:>9.2} {:>10.2} {:>8}",
            names[z.class],
            z.workers,
            z.mean_age_s / 86_400.0,
            z.mean_crop_fill,
            z.nursing
        );
    }
    println!(
        "\nage–depth correlation over the workers inside: {:.2} (the young with the brood, the old by the entrance)",
        sim.age_depth_correlation()
    );
    let s = sim.stats();
    println!(
        "trophallaxis contacts by zone pair (rows and columns: brood chamber, between, entrance ring):"
    );
    for row in s.nest_contacts {
        println!("  {:>6} {:>6} {:>6}", row[0], row[1], row[2]);
    }
    println!(
        "contact assortativity {:.2} (0: zones mix at random; 1: food only changes hands within a zone)",
        s.contact_assortativity()
    );
    println!(
        "sugar delivered {:.2} mg; passed on inside by trophallaxis {:.2} mg ({:.1} milligrams handed on per milligram delivered)",
        s.sugar_delivered_mg,
        s.trophallaxis_mg,
        s.trophallaxis_mg / s.sugar_delivered_mg.max(1e-9)
    );
    println!(
        "unloading contacts {} ({} foragers gave up); larvae fed by nurses in the chamber: {} nursing ticks",
        s.unloading_contacts,
        s.failed_unloads,
        s.activity_ticks[Activity::Nursing.index()]
    );
    println!(
        "corpses fetched from inside the nest and carried out: {} (still inside: {}); pick-ups outside since, as the refuse is shifted about: {}\n",
        s.corpses_fetched,
        sim.nest().corpses,
        s.corpses_moved - s.corpses_fetched
    );
    println!("{}", render(&sim));
}
