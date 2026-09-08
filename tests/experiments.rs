//! Validation against the classic experiments. Each test reproduces the
//! qualitative outcome reported in the literature with a small number of
//! replicates; the `experiments` example runs the same setups at full size.

use ant_simulator::prelude::*;

#[test]
fn double_bridge_selects_the_shorter_branch() {
    // Goss, Aron, Deneubourg & Pasteels 1989: with a long branch twice the
    // short one, traffic concentrates on the short branch.
    let outcomes = run_double_bridge(
        &BridgeSpec::ratio_two(),
        &Species::argentine(),
        60,
        20.0 * 60.0,
        8.0 * 60.0,
        &[1, 2],
    );
    for o in &outcomes {
        assert!(
            o.delivered > 20,
            "the colony should forage on the bridge: {o:?}"
        );
        assert!(
            o.short_fraction > 0.7,
            "short branch should carry most traffic: {o:?}"
        );
        assert!(o.trail_short > o.trail_long, "and most trail: {o:?}");
    }
    let summary = summarize(
        &outcomes
            .iter()
            .map(|o| o.short_fraction)
            .collect::<Vec<_>>(),
    );
    assert_eq!(summary.majority, 1.0);
}

#[test]
fn lasius_also_selects_the_shorter_branch() {
    // Beckers, Deneubourg & Goss 1992: Lasius niger selects the shorter
    // branch with a large majority of foragers. A minority that learned
    // the long branch on its first trips may keep to it (route memory),
    // so the majority is not always overwhelming.
    let outcomes = run_double_bridge(
        &BridgeSpec::ratio_two(),
        &Species::lasius_niger(),
        60,
        20.0 * 60.0,
        8.0 * 60.0,
        &[1, 3],
    );
    for o in &outcomes {
        assert!(o.short_fraction > 0.55, "{o:?}");
    }
    let mean = outcomes.iter().map(|o| o.short_fraction).sum::<f64>() / outcomes.len() as f64;
    assert!(mean > 0.75, "{outcomes:?}");
}

#[test]
fn equal_branches_break_symmetry_under_pure_pheromone_feedback() {
    // Deneubourg, Aron, Goss & Pasteels 1990: with two equal branches and a
    // trail that does not evaporate, the colony ends up on one of them.
    let fractions: Vec<f64> = [1u64, 2, 3, 4]
        .iter()
        .map(|&seed| {
            let mut cfg = experiment_config(
                Species::argentine(),
                double_bridge(&BridgeSpec::equal()),
                60,
            );
            pure_pheromone_feedback(&mut cfg);
            run_double_bridge_configured(cfg, 45.0 * 60.0, 10.0 * 60.0, seed).short_fraction
        })
        .collect();
    let decided = fractions
        .iter()
        .filter(|f| (**f - 0.5).abs() > 0.25)
        .count();
    assert!(
        decided >= 2,
        "expected most runs to settle on one branch: {fractions:?}"
    );
}

#[test]
fn crowding_spreads_traffic_over_a_narrow_bridge() {
    // Dussutour, Fourcassié, Helbing & Deneubourg 2004: at high traffic a
    // narrow bridge is used symmetrically, and throughput is not lost.
    let ants = 240;
    let run = |cap: u16, seed: u64| {
        run_crowded_bridge(
            Species::lasius_niger(),
            ants,
            cap,
            20.0 * 60.0,
            5.0 * 60.0,
            seed,
        )
    };
    let narrow: Vec<BridgeOutcome> = [1, 2].iter().map(|&s| run(6, s)).collect();
    let wide: Vec<BridgeOutcome> = [1, 2].iter().map(|&s| run(64, s)).collect();
    let dev = |o: &[BridgeOutcome]| {
        o.iter()
            .map(|o| (o.short_fraction - 0.5).abs())
            .sum::<f64>()
            / o.len() as f64
    };
    let traffic = |o: &[BridgeOutcome]| {
        o.iter()
            .map(|o| (o.short_crossings + o.long_crossings) as f64)
            .sum::<f64>()
            / o.len() as f64
    };
    assert!(
        dev(&narrow) < 0.1,
        "narrow bridge should split evenly: {narrow:?}"
    );
    assert!(
        traffic(&narrow) > 0.7 * traffic(&wide),
        "throughput should not collapse: narrow {} wide {}",
        traffic(&narrow),
        traffic(&wide)
    );
    // Crowding also thins the trail laid per passage (Czaczkes et al. 2013).
    let marks_per_crossing = |o: &[BridgeOutcome]| {
        o.iter().map(|o| o.trail_short + o.trail_long).sum::<f64>()
            / o.iter()
                .map(|o| (o.short_crossings + o.long_crossings) as f64)
                .sum::<f64>()
                .max(1.0)
    };
    assert!(
        marks_per_crossing(&narrow) < 0.9 * marks_per_crossing(&wide),
        "narrow {} vs wide {} marks per crossing",
        marks_per_crossing(&narrow),
        marks_per_crossing(&wide)
    );
    // The full-size contrast (one branch on the wide bridge, both on the
    // narrow one at high traffic) is shown by the `experiments` example.
}

#[test]
fn foraging_effort_tracks_source_productivity() {
    // Mailleux, Deneubourg & Detrain 2003: at a slow drip foragers leave
    // with partial loads and recruit little; effort scales with the flow.
    let outcomes = run_productivity_response(
        &Species::lasius_niger(),
        60,
        &[0.05, 0.5, 5.0],
        30.0 * 60.0,
        3,
    );
    assert_eq!(outcomes.len(), 3);
    let (slow, mid, fast) = (&outcomes[0], &outcomes[1], &outcomes[2]);
    assert!(
        slow.delivered < mid.delivered && mid.delivered < fast.delivered,
        "{outcomes:?}"
    );
    assert!(slow.mean_load_ul < fast.mean_load_ul, "{outcomes:?}");
    assert!(
        slow.recruiting_fraction < fast.recruiting_fraction,
        "{outcomes:?}"
    );
    assert!(fast.delivered > 3 * slow.delivered.max(1), "{outcomes:?}");
}

#[test]
fn larvae_turn_foraging_towards_protein() {
    // Dussutour & Simpson 2009: colonies with larvae collect protein;
    // colonies of workers alone take carbohydrate and leave prey.
    let run =
        |larvae: bool| run_communal_nutrition(Species::lasius_niger(), 60, larvae, 30.0 * 60.0, 2);
    let with = run(true);
    let without = run(false);
    assert!(
        with.delivered > 20 && without.delivered > 20,
        "{with:?} {without:?}"
    );
    assert!(
        with.protein_share > without.protein_share + 0.2,
        "larvae should raise the protein share: {with:?} vs {without:?}"
    );
    assert!(without.protein_share < 0.3, "{without:?}");
}

#[test]
fn corpses_are_gathered_into_piles() {
    // Deneubourg et al. 1991; Theraulaz et al. 2002: workers pick up lone
    // corpses and drop them where corpses lie, so scattered corpses end
    // up in a few piles.
    let o = run_cemetery(Species::lasius_niger(), 60, 300, 40.0 * 60.0, 1);
    assert!(o.corpses_moved > 100, "{o:?}");
    assert!(
        o.clusters_end * 2 < o.clusters_start,
        "piles should merge: {o:?}"
    );
    assert!(o.largest_end > o.largest_start, "{o:?}");
}

#[test]
fn food_odour_speeds_up_discovery() {
    // Buehlmann et al. 2014: ants locate food by its smell; here a hidden
    // pool is found sooner by a party of scouts when it gives off an
    // odour (with many more scouts leaving at once, one of them walks
    // straight into it either way).
    let seeds = [1u64, 2, 3];
    let time = |odour: bool| -> f64 {
        seeds
            .iter()
            .map(|&s| {
                run_discovery(Species::lasius_niger(), 20, odour, 20.0 * 60.0, s)
                    .first_find_s
                    .unwrap_or(20.0 * 60.0)
            })
            .sum::<f64>()
            / seeds.len() as f64
    };
    let with = time(true);
    let without = time(false);
    assert!(
        with < 0.75 * without,
        "odour should shorten discovery: {with:.0} s with, {without:.0} s without"
    );
}

#[test]
fn landmarks_shorten_the_search_for_the_nest() {
    // Wehner & Räber 1979; Collett 1992: a desert ant that has taken a view
    // of the landmarks around its nest fixes its position from them and
    // needs less searching to find the entrance. Path integration is made
    // noisier than the species default so that the search phase matters.
    let mut species = Species::cataglyphis();
    species.pi_heading_noise_deg = 8.0;
    species.pi_distance_noise = 0.1;
    let searching_per_trip = |landmarks: usize| -> (f64, u64) {
        let mut ticks = 0.0;
        let mut fixes = 0;
        for seed in 1..=3u64 {
            let world = WorldConfig {
                seed: Some(seed),
                random_landmarks: landmarks,
                ..WorldConfig::default()
            };
            let cfg = experiment_config(species.clone(), world, 40);
            let mut sim = Simulation::new(cfg, seed);
            sim.run_seconds(20.0 * 60.0);
            let s = sim.stats();
            assert!(s.food_delivered > 20, "{s:?}");
            ticks += s.activity_ticks[Activity::Searching.index()] as f64 / s.food_delivered as f64;
            fixes += s.landmark_fixes;
        }
        (ticks / 3.0, fixes)
    };
    let (bare, no_fixes) = searching_per_trip(0);
    let (marked, fixes) = searching_per_trip(12);
    assert_eq!(no_fixes, 0);
    assert!(fixes > 0);
    assert!(
        marked < 0.8 * bare,
        "landmarks should cut the search: {marked:.2} vs {bare:.2} searching ticks per trip"
    );
}

#[test]
fn richer_source_wins_the_colony() {
    // Beckers, Deneubourg, Goss & Pasteels 1990: two sources at equal
    // distance; quality-modulated trail laying and site fidelity focus the
    // colony on the richer one.
    for seed in [1, 2] {
        let o = run_two_sources_once(12, 1.0, 0.1, Species::lasius_niger(), 60, 20.0 * 60.0, seed);
        assert!(o.volume_a + o.volume_b > 10.0, "{o:?}");
        assert!(o.fraction_a > 0.6, "rich source should dominate: {o:?}");
    }
}

#[test]
fn foraging_activity_tracks_colony_hunger() {
    // Mailleux, Deneubourg & Detrain 2003: hungrier colonies send more
    // foragers out.
    let outcomes = run_hunger_response(
        &Species::lasius_niger(),
        40,
        &[0.05, 0.6, 1.0],
        15.0 * 60.0,
        4,
    );
    assert_eq!(outcomes.len(), 3);
    let (hungry, middle, replete) = (
        outcomes[0].foraging_fraction,
        outcomes[1].foraging_fraction,
        outcomes[2].foraging_fraction,
    );
    assert!(hungry > middle && middle > replete, "{outcomes:?}");
    assert!(hungry > replete + 0.3, "{outcomes:?}");
    assert!(outcomes[0].delivered > outcomes[2].delivered);
}

#[test]
fn threshold_reinforcement_increases_specialisation() {
    // Theraulaz, Bonabeau & Deneubourg 1998: reinforcing response thresholds
    // makes workers specialise.
    let outcomes: Vec<LaborOutcome> = [1, 2]
        .iter()
        .map(|&seed| run_division_of_labor(&Species::lasius_niger(), 45, 30.0 * 60.0, seed))
        .collect();
    let mean = |f: &dyn Fn(&LaborOutcome) -> f64| outcomes.iter().map(f).sum::<f64>() / 2.0;
    let with = mean(&|o| o.with_reinforcement);
    let without = mean(&|o| o.without_reinforcement);
    assert!(with > 0.0 && without > 0.0, "{outcomes:?}");
    assert!(
        with > without + 0.03,
        "reinforcement should raise the index: with {with:.3}, without {without:.3} ({outcomes:?})"
    );
}

#[test]
fn desert_ants_forage_without_any_trail() {
    // Cataglyphis: no recruitment trail, yet path integration brings
    // foragers home and site fidelity brings them back to food.
    let world = WorldConfig {
        seed: Some(5),
        ..WorldConfig::default()
    };
    let cfg = experiment_config(Species::cataglyphis(), world, 30);
    let mut sim = Simulation::new(cfg, 5);
    sim.run_seconds(15.0 * 60.0);
    let s = sim.stats();
    assert!(s.food_delivered > 20, "{s:?}");
    assert_eq!(sim.world().total_pheromone(Pheromone::Trail), 0.0);
    assert!(
        s.path.trip_efficiency() > 0.5,
        "{}",
        s.path.trip_efficiency()
    );
}

#[test]
fn pharaoh_ants_mark_exhausted_routes() {
    // Robinson et al. 2005: unsuccessful foragers lay a repellent marking.
    let world = WorldConfig {
        seed: Some(6),
        random_food: None,
        food_sources: vec![FoodSource::pool(Position::new(50, 20), 0, 1.0, 1.0)],
        ..WorldConfig::default()
    };
    let cfg = experiment_config(Species::pharaoh(), world, 40);
    let mut sim = Simulation::new(cfg, 6);
    sim.run_seconds(20.0 * 60.0);
    assert!(sim.stats().failed_trips > 0);
    assert!(
        sim.world().total_pheromone(Pheromone::NoEntry) > 0.0,
        "no-entry marking should appear once the source is exhausted"
    );
}

#[test]
fn queen_thoughts_are_written_into_the_non_invariant_flow() {
    // The hive's cognitive geometry: a thought expressed in the root
    // temperature reaches the colony's decisions and the straightness of
    // its paths, and the readouts and the closed loop run on the record.
    let cfg = MemoryProbeConfig {
        ants: 60,
        epochs: 40,
        epoch_s: 30.0,
        warmup_s: 10.0 * 60.0,
        lags: 2,
        ..MemoryProbeConfig::default()
    };
    let mut probe = run_memory_probe(&cfg);
    assert_eq!(probe.epochs, 40);
    assert!(
        probe.mean_density > 1.0,
        "the colony keeps moving: {}",
        probe.mean_density
    );
    assert!(
        probe.entropy_correlation > 0.5,
        "the dial reaches the decisions: {}",
        probe.entropy_correlation
    );
    assert!(
        probe.straightness_correlation < -0.3,
        "a hotter colony walks more tortuously: {}",
        probe.straightness_correlation
    );
    assert_eq!(probe.capacities.len(), 3);
    assert_eq!(probe.capacities[1].component, Component::Residual);
    let closed = run_closed_loop(
        &mut probe.simulation,
        6,
        Recursion {
            lag: 1,
            gain: -10.0,
        },
        cfg.expression,
        cfg.train_fraction,
    );
    assert_eq!(closed.thoughts.len(), 6);
    assert!(closed
        .thoughts
        .iter()
        .all(|t| t.is_finite() && t.abs() <= 1.0));
    assert!(
        closed.conviction > 0.0,
        "recall drives a thought: {closed:?}"
    );
}

#[test]
fn trail_foraging_is_cooperative_while_memory_foraging_is_proportional() {
    // Beekman, Sumpter & Ratnieks 2001: a colony that reaches its feeder
    // by trail alone forages poorly when small and well when large, as
    // the trail's reinforcement outgrows its evaporation; a colony whose
    // foragers navigate by memory forages as well at any size.
    let scan = SizeScan {
        sizes: vec![10, 80],
        ..SizeScan::default()
    };
    let by_trail = run_colony_size_scan(&scan);
    let by_memory = run_colony_size_scan(&SizeScan {
        trail_only: false,
        ..scan
    });
    assert!(
        by_trail[1].per_capita_per_hour > 1.5 * by_trail[0].per_capita_per_hour,
        "trail foraging gains with size: {:?}",
        by_trail
    );
    assert!(
        by_trail[1].trail_mid > 5.0 * by_trail[0].trail_mid,
        "the large colony keeps a trail: {:?}",
        by_trail
    );
    assert!(
        by_trail[1].ordered > by_trail[0].ordered,
        "and its trips are ordered: {:?}",
        by_trail
    );
    assert!(
        by_memory[1].per_capita_per_hour < 1.5 * by_memory[0].per_capita_per_hour,
        "memory foraging does not: {:?}",
        by_memory
    );
    assert!(
        by_memory[0].per_capita_per_hour > by_trail[0].per_capita_per_hour,
        "a small colony forages far better by memory than by trail: {:?} vs {:?}",
        by_memory[0],
        by_trail[0]
    );
}
