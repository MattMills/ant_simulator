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
fn double_bridge_works_for_inbound_only_layers() {
    // Beckers, Deneubourg & Goss 1992: Lasius niger, which lays trail on
    // the way home, also selects the shorter branch.
    let o = run_double_bridge_once(
        &BridgeSpec::ratio_two(),
        Species::lasius_niger(),
        60,
        20.0 * 60.0,
        8.0 * 60.0,
        3,
    );
    assert!(o.short_fraction > 0.7, "{o:?}");
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
        dev(&narrow) < 0.08,
        "narrow bridge should split evenly: {narrow:?}"
    );
    assert!(
        dev(&wide) > dev(&narrow),
        "the wide bridge should keep more of its asymmetry: wide {wide:?} narrow {narrow:?}"
    );
    assert!(
        traffic(&narrow) > 0.7 * traffic(&wide),
        "throughput should not collapse: narrow {} wide {}",
        traffic(&narrow),
        traffic(&wide)
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
        food_sources: vec![FoodSource {
            center: Position::new(50, 20),
            radius: 0,
            volume_ul_per_cell: 1.0,
            molarity: 1.0,
        }],
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
