//! Run a colony of the default species, compare species, and sweep the
//! colony-wide temperature dial.
//!
//!     cargo run --release --example colony [minutes]

use ant_simulator::prelude::*;

fn world() -> WorldConfig {
    WorldConfig {
        seed: Some(2024),
        ..WorldConfig::default()
    }
}

fn hungry(species: Species) -> SimConfig {
    let mut cfg = SimConfig::for_species(species);
    cfg.world = world();
    cfg.nest.initial_satiation = 0.1;
    cfg.nest.initial_brood_per_ant = 0.5;
    cfg
}

fn main() {
    let minutes: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30.0);
    let seconds = minutes * 60.0;

    let species = Species::lasius_niger();
    println!(
        "species: {} ({:.1} cm/s, trail half-life {:.0} min, k = {}, n = {})",
        species.name,
        species.speed_cm_s,
        species.trail.half_life_s / 60.0,
        species.trail.k,
        species.choice_exponent
    );
    println!("hierarchy:\n{}", Hierarchy::default().describe());

    let mut sim = Simulation::new(hungry(species), 1);
    for third in 1..=3 {
        sim.run_seconds(seconds / 3.0);
        println!("after {:.0} minutes:", minutes * third as f64 / 3.0);
        println!("{}", render(&sim));
    }
    let s = sim.stats();
    println!(
        "activity: outside {:.0}%, nursing {:.0}%, resting {:.0}%, unloading {:.0}%",
        100.0 * s.foraging_fraction(),
        100.0 * s.activity_fraction(Activity::Nursing),
        100.0 * s.activity_fraction(Activity::Resting),
        100.0 * s.activity_fraction(Activity::Unloading)
    );
    println!(
        "delivered {} loads, {} failed trips, {} deaths ({} predation, {} starvation, {} heat), {} eggs, {} emerged, division of labour {:.2}",
        s.food_delivered, s.failed_trips, s.deaths, s.deaths_predation, s.deaths_starvation, s.deaths_heat, s.eggs, s.births,
        sim.division_of_labor()
    );
    println!(
        "paths: turn entropy {:.2} nats, straight {:.0}%, trip efficiency {:.2}\n",
        s.path.turn_entropy(),
        100.0 * s.path.straight_rate(),
        s.path.trip_efficiency()
    );

    // Species comparison on the same map.
    println!("== species on the same map ({minutes:.0} minutes) ==");
    println!(
        "{:<22} {:>9} {:>7} {:>7} {:>8} {:>7} {:>7}",
        "species", "delivered", "failed", "deaths", "trail", "turnH", "eff"
    );
    for species in [
        Species::lasius_niger(),
        Species::argentine(),
        Species::pharaoh(),
        Species::cataglyphis(),
    ] {
        let name = species.name.clone();
        let mut sim = Simulation::new(hungry(species), 1);
        sim.run_seconds(seconds);
        let s = sim.stats();
        println!(
            "{:<22} {:>9} {:>7} {:>7} {:>8.0} {:>7.2} {:>7.2}",
            name,
            s.food_delivered,
            s.failed_trips,
            s.deaths,
            sim.world().total_pheromone(Pheromone::Trail),
            s.path.turn_entropy(),
            s.path.trip_efficiency()
        );
    }

    // The colony-wide dial in temperature mode: 1 is the published choice
    // function; below it ants commit harder, above it they wander.
    println!("\n== colony-wide temperature ({minutes:.0} minutes) ==");
    println!(
        "{:>12} {:>9} {:>7} {:>8} {:>7} {:>7}",
        "temperature", "delivered", "failed", "entropy", "turnH", "eff"
    );
    for temperature in [0.25, 0.5, 1.0, 2.0, 4.0] {
        let mut sim = Simulation::new(hungry(Species::lasius_niger()), 1);
        sim.hierarchy_mut().node_mut(0).surface.entropy = EntropyControl::fixed(temperature);
        sim.run_seconds(seconds);
        let s = sim.stats();
        println!(
            "{:>12.2} {:>9} {:>7} {:>8.3} {:>7.2} {:>7.2}",
            temperature,
            s.food_delivered,
            s.failed_trips,
            s.mean_entropy(),
            s.path.turn_entropy(),
            s.path.trip_efficiency()
        );
    }

    // A caste can be made hotter than the rest of the colony on its own.
    let mut sim = Simulation::new(hungry(Species::lasius_niger()), 1);
    let scouts = sim.hierarchy().find("caste-0").unwrap();
    sim.hierarchy_mut().node_mut(scouts).surface.entropy = EntropyControl::relative(3.0);
    sim.run_seconds(seconds);
    println!("\nwith caste-0 at three times the colony's temperature:");
    println!("{}", sim.hierarchy().describe());
    let s = sim.stats();
    for (i, &leaf) in sim.hierarchy().leaves().iter().enumerate() {
        println!(
            "  leaf {:<8} {}  deliveries {:>4}  trip efficiency {:.2}",
            sim.hierarchy().node(leaf).name,
            sim.policies()[i].tempering.describe(),
            s.delivered_by_node[leaf],
            s.path_by_node[leaf].trip_efficiency()
        );
    }
}
