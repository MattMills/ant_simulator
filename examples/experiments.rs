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
    let long_run = seconds.max(60.0 * 60.0);
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

    println!("\n== crowding on a narrow bridge (Dussutour et al. 2004) ==");
    println!("equal branches; a wide bridge holds 64 ants per cell, a narrow one 8\n");
    for (n, cap) in [(80usize, 64u16), (80, 8), (400, 64), (400, 8)] {
        let outcomes: Vec<BridgeOutcome> = seeds
            .iter()
            .map(|&seed| run_crowded_bridge(Species::lasius_niger(), n, cap, seconds, window, seed))
            .collect();
        let dev = outcomes
            .iter()
            .map(|o| (o.short_fraction - 0.5).abs())
            .sum::<f64>()
            / outcomes.len() as f64;
        let traffic = outcomes
            .iter()
            .map(|o| (o.short_crossings + o.long_crossings) as f64 / (window / 60.0))
            .sum::<f64>()
            / outcomes.len() as f64;
        println!(
            "   {n:>3} ants, {:<6} bridge: mean |share - ½| {dev:.2}, traffic {traffic:.0} crossings/min",
            if cap > 16 { "wide" } else { "narrow" }
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

    println!("\n== foraging effort against the productivity of a dripping source (Mailleux et al. 2003) ==");
    for o in run_productivity_response(
        &Species::lasius_niger(),
        60,
        &[0.02, 0.1, 0.5, 2.0, 10.0],
        seconds,
        1,
    ) {
        println!(
            "   flow {:>5.2} µl/min: {:>4.1} ants at the source, {:.0}% of ant-time outside, {:>3} loads, mean load {:.2} µl, {:.0}% of returns recruit",
            o.flow_ul_per_min,
            o.at_source,
            100.0 * o.foraging_fraction,
            o.delivered,
            o.mean_load_ul,
            100.0 * o.recruiting_fraction
        );
    }

    println!(
        "\n== finding a hidden pool by its smell (Buehlmann et al. 2014): a party of 20 scouts, the pool 20 cm off =="
    );
    for odour in [false, true] {
        let times: Vec<f64> = seeds
            .iter()
            .map(|&seed| {
                run_discovery(
                    Species::lasius_niger(),
                    20,
                    10,
                    odour,
                    seconds.min(1200.0),
                    seed,
                )
                .first_find_s
                .unwrap_or(seconds.min(1200.0))
            })
            .collect();
        println!(
            "   {:<14} first find after {:.0} s on average (seeds: {})",
            if odour { "with odour:" } else { "no odour:" },
            times.iter().sum::<f64>() / times.len() as f64,
            times
                .iter()
                .map(|t| format!("{t:.0}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }

    println!("\n== desert ants against the heat (Cerdá, Retana & Cros 1998) ==");
    for o in run_thermal_tradeoff(
        &Species::cataglyphis(),
        60,
        &[30.0, 40.0, 48.0, 52.0, 54.0, 55.0],
        seconds.min(1800.0),
        1,
    ) {
        println!(
            "   {:>4.0} °C: speed ×{:.2}, {:>4} loads, {:>3} killed by heat",
            o.temperature_c, o.speed_factor, o.delivered, o.deaths_heat
        );
    }

    println!("\n== foraging activity against colony satiation (Mailleux et al. 2006) ==");
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

    println!("\n== communal nutrition: sugar or prey, with and without larvae (Dussutour & Simpson 2009) ==");
    for larvae in [false, true] {
        let outcomes: Vec<NutritionOutcome> = seeds
            .iter()
            .map(|&seed| run_communal_nutrition(Species::lasius_niger(), 60, larvae, seconds, seed))
            .collect();
        let n = outcomes.len() as f64;
        println!(
            "   {:<14} protein share {:.2}, sugar {:.1} mg, protein {:.1} mg, {:.0} loads",
            if larvae { "with larvae:" } else { "no larvae:" },
            outcomes.iter().map(|o| o.protein_share).sum::<f64>() / n,
            outcomes.iter().map(|o| o.sugar_mg).sum::<f64>() / n,
            outcomes.iter().map(|o| o.protein_mg).sum::<f64>() / n,
            outcomes.iter().map(|o| o.delivered as f64).sum::<f64>() / n
        );
    }

    println!("\n== cemetery formation: 300 corpses in an 80 cm arena (Theraulaz et al. 2002) ==");
    for &seed in seeds.iter().take(3) {
        let o = run_cemetery(
            Species::lasius_niger(),
            60,
            300,
            seconds.max(40.0 * 60.0),
            seed,
        );
        println!(
            "   seed {seed}: piles {} → {}, largest pile {} → {} corpses, {} pick-ups",
            o.clusters_start, o.clusters_end, o.largest_start, o.largest_end, o.corpses_moved
        );
    }

    println!("\n== division of labour with and without threshold reinforcement (Theraulaz et al. 1998) ==");
    let o = run_division_of_labor(&Species::lasius_niger(), 60, seconds, 1);
    println!(
        "   index with reinforcement {:.2}, without {:.2}",
        o.with_reinforcement, o.without_reinforcement
    );
}
