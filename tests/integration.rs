use ant_simulator::prelude::*;

fn world() -> WorldConfig {
    WorldConfig {
        width: 36,
        height: 28,
        nest: Position::new(18, 14),
        seed: Some(11),
        ..WorldConfig::default()
    }
}

#[test]
fn colony_with_default_hierarchy_forages() {
    let config = SimConfig {
        world: world(),
        ants: 40,
        ..SimConfig::default()
    };
    let mut sim = Simulation::new(config.clone(), 3);
    sim.run(1500);
    let s = sim.stats();
    assert!(s.food_delivered >= 5, "expected foraging to work: {s:?}");
    // Under the biological default the temperature is fixed and entropy
    // simply reports what the choice function does; an entropy dial pins it.
    let mut dialed = Simulation::new(config, 3);
    dialed.hierarchy_mut().node_mut(0).surface.entropy = EntropyControl::absolute(0.35);
    dialed.run(600);
    let target = 0.35 * (RING as f64).ln();
    let h = dialed.stats().mean_entropy();
    assert!(
        (h - target).abs() < 0.25,
        "mean entropy {h} should track the dial ({target})"
    );
}

#[test]
fn custom_hierarchy_shape_and_targets() {
    let spec = HierarchySpec {
        root_name: "nest".into(),
        levels: vec![
            LevelSpec::new("guild", 2),
            LevelSpec::new("team", 2),
            LevelSpec::new("pair", 2),
        ],
    };
    assert_eq!(spec.node_count(), 15);
    let config = ArenaConfig {
        sim: SimConfig {
            world: world(),
            hierarchy: spec,
            ants: 24,
            ..SimConfig::default()
        },
        steps_per_turn: 40,
        targets: ControlTargets::Depth(3),
        schedule: RotationSchedule::Cyclic,
        ..ArenaConfig::default()
    };
    let learners: Vec<Box<dyn Learner>> = vec![
        Box::new(HillClimber::new(0.1)),
        Box::new(RandomLearner::new(0.1)),
    ];
    let mut arena = Arena::new(config, learners, 4).unwrap();
    assert_eq!(arena.targets().len(), 8);
    let report = arena.run(8);
    assert_eq!(report.schedule_period, Some(8));
    assert_eq!(report.learners[1].turns_connected, 8);
    // Only the pair nodes may have changed; everything above them is intact.
    let h = arena.hierarchy();
    for node in h.nodes() {
        if node.depth < 3 {
            let expected = if node.depth == 0 {
                BehavioralSurface::instinct()
            } else {
                BehavioralSurface::neutral()
            };
            assert_eq!(node.surface, expected, "{} was touched", node.name);
        }
    }
}

#[test]
fn phase_aware_learner_recovers_the_rotation_period() {
    // Two levers rotate over the colony, one caste, and one squad: subtrees of
    // very different size, so the subtree reward a lever sees is periodic
    // with the rotation period of 3.
    let config = ArenaConfig {
        sim: SimConfig {
            world: world(),
            ants: 30,
            ..SimConfig::default()
        },
        steps_per_turn: 80,
        targets: ControlTargets::Nodes(vec![0, 1, 2]),
        schedule: RotationSchedule::RandomStatic { period: 3 },
        feedback: FeedbackScope::Subtree,
        seeding: EpisodeSeeding::Fixed,
        ..ArenaConfig::default()
    };
    let learners: Vec<Box<dyn Learner>> = vec![
        Box::new(PhaseAware::new(EntropyBandit::default(), 6)),
        Box::new(StaticLearner),
    ];
    let mut arena = Arena::new(config, learners, 21).unwrap();
    let report = arena.run(60);
    let inferred = report.learners[0].inferred_period;
    assert!(
        inferred == Some(3) || inferred == Some(6),
        "phase-aware learner should find the period (or a multiple): {inferred:?}\n{report}"
    );
}

#[test]
fn policy_gradient_needs_trace_and_moves_weights() {
    let config = ArenaConfig {
        sim: SimConfig {
            world: world(),
            ants: 30,
            trace: true,
            ..SimConfig::default()
        },
        steps_per_turn: 60,
        targets: ControlTargets::Nodes(vec![0]),
        schedule: RotationSchedule::Fixed,
        // Fresh seeds: with identical rewards every turn the advantage against
        // the baseline would be exactly zero and nothing would move.
        seeding: EpisodeSeeding::Fresh,
        ..ArenaConfig::default()
    };
    let learners: Vec<Box<dyn Learner>> = vec![Box::new(PolicyGradient::new(0.05))];
    let mut arena = Arena::new(config, learners, 8).unwrap();
    let before = arena.hierarchy().root().surface.clone();
    arena.run(4);
    let after = arena.hierarchy().root().surface.clone();
    assert_ne!(before.weights, after.weights);
    assert_eq!(
        before.entropy, after.entropy,
        "policy gradient never touches the dial"
    );
}

#[test]
fn fresh_rotation_is_unlearnable_but_runs() {
    let config = ArenaConfig {
        sim: SimConfig {
            world: world(),
            ants: 16,
            ..SimConfig::default()
        },
        steps_per_turn: 30,
        schedule: RotationSchedule::Fresh,
        feedback: FeedbackScope::Colony,
        ..ArenaConfig::default()
    };
    let learners: Vec<Box<dyn Learner>> = vec![
        Box::new(CrossEntropy::new(4, 2)),
        Box::new(HillClimber::new(0.1).with_mutation(Mutation::Rotation)),
    ];
    let mut arena = Arena::new(config, learners, 1).unwrap();
    let report = arena.run(6);
    assert_eq!(report.schedule_period, None);
    assert_eq!(report.turns, 6);
    let text = format!("{report}");
    assert!(text.contains("rotation-search"));
    assert!(text.contains("none (fresh every turn)"));
}

fn geometry_run(deformation: Deformation, selection: Selection, seed: u64) -> Stats {
    // A fixed entropy budget so that only its shape differs between runs.
    let mut instinct = BehavioralSurface::instinct().with_deformation(deformation);
    instinct.entropy = EntropyControl::absolute(0.35);
    let mut config = SimConfig {
        world: world(),
        ants: 40,
        selection,
        instinct,
        ..SimConfig::default()
    };
    config.nest.initial_satiation = 0.05;
    // Searching ants run hotter than the dial; keep every decision on the
    // same budget so that only the geometry differs between runs.
    config.species.search_temperature_factor = 1.0;
    let mut sim = Simulation::new(config, seed);
    sim.run(900);
    sim.stats().clone()
}

#[test]
fn same_entropy_budget_different_path_topology() {
    let flat = geometry_run(Deformation::none(), Selection::Softmax, 4);
    let rough = geometry_run(
        Deformation {
            rough: 1.5,
            ..Deformation::none()
        },
        Selection::Softmax,
        4,
    );
    // Both decision streams carry the same per-decision entropy target...
    assert!((flat.mean_entropy() - rough.mean_entropy()).abs() < 0.08);
    // ...but the paths look nothing alike.
    assert!(
        rough.path.turn_entropy() > flat.path.turn_entropy() + 0.3,
        "rough {} vs flat {}",
        rough.path.turn_entropy(),
        flat.path.turn_entropy()
    );
    assert!(rough.path.straight_rate() < flat.path.straight_rate() - 0.15);
    // The ledger shows where the disorder went: displaced tempering within
    // the decision, and randomness between decisions.
    let c = rough.path.ledger.contributions();
    assert!(c.roughening < -0.05, "{c:?}");
    assert!(c.field > 0.1, "{c:?}");
    let f = flat.path.ledger.contributions();
    assert_eq!(f.field, 0.0);
}

#[test]
fn smoothing_keeps_paths_coherent() {
    let flat = geometry_run(Deformation::none(), Selection::Softmax, 5);
    let smooth = geometry_run(
        Deformation {
            smooth: 1.5,
            ..Deformation::none()
        },
        Selection::Softmax,
        5,
    );
    let c = smooth.path.ledger.contributions();
    assert!(c.smoothing > 0.05, "{c:?}");
    assert_eq!(c.field, 0.0);
    assert!((smooth.path.turn_entropy() - flat.path.turn_entropy()).abs() < 0.25);
    assert!(smooth.path.trip_efficiency() > 0.8);
}

#[test]
fn sucker_with_short_reach_is_path_bound() {
    let free = geometry_run(Deformation::none(), Selection::Softmax, 6);
    let bound = geometry_run(Deformation::none(), Selection::Sucker { reach: 1 }, 6);
    // One proposal step can never turn by more than one ring step unless
    // straight ahead is blocked, so sharp turns (90° or more) all but vanish.
    let sharp = |s: &Stats| s.path.sharp_turn_rate();
    assert!(sharp(&bound) < 0.02, "bound sharp turns {}", sharp(&bound));
    assert!(
        sharp(&free) > 2.0 * sharp(&bound),
        "free {} vs bound {}",
        sharp(&free),
        sharp(&bound)
    );
    assert!(bound.mean_selected_entropy() < bound.mean_entropy() - 0.05);
    let c = bound.path.ledger.contributions();
    assert!(c.selection < -0.05, "{c:?}");
}

#[test]
fn geometry_bandit_learns_in_a_sucker_arena() {
    let config = ArenaConfig {
        sim: SimConfig {
            world: world(),
            ants: 24,
            selection: Selection::Sucker { reach: 8 },
            ..SimConfig::default()
        },
        steps_per_turn: 40,
        targets: ControlTargets::Nodes(vec![0]),
        schedule: RotationSchedule::Fixed,
        ..ArenaConfig::default()
    };
    let learners: Vec<Box<dyn Learner>> = vec![Box::new(DialBandit::geometry())];
    let mut arena = Arena::new(config, learners, 3).unwrap();
    let report = arena.run(14);
    assert_eq!(report.learners[0].turns_connected, 14);
    let root = &arena.hierarchy().root().surface;
    assert_eq!(root.weights, BehavioralSurface::instinct().weights);
    assert!(report.learners[0].summary.starts_with("geometry-bandit"));
    assert_eq!(arena.hierarchy().param_len(), 10 * PARAM_LEN);
}

#[test]
fn nest_interior_is_structured_by_age() {
    // Sendova-Franks & Franks 1995; Mersch, Crespi & Keller 2013: workers
    // keep to zones that drift outward with age, so the young sit with
    // the brood and the old by the entrance, and food changes hands
    // mostly within a zone.
    let mut cfg = SimConfig::default();
    cfg.world.seed = Some(11);
    cfg.world.nest_radius = 3;
    cfg.ants = 60;
    cfg.nest.initial_satiation = 0.3;
    cfg.nest.initial_brood_per_ant = 1.0;
    let mut sim = Simulation::new(cfg, 5);
    sim.run_seconds(20.0 * 60.0);
    let r = sim.age_depth_correlation();
    assert!(r > 0.3, "age–depth correlation {r}");
    let profile = sim.nest_profile();
    assert!(
        profile[0].mean_age_s < profile[2].mean_age_s,
        "the brood chamber holds the younger workers: {profile:?}"
    );
    let contacts = sim.stats().nest_contacts;
    let assortativity = sim.stats().contact_assortativity();
    assert!(
        assortativity > 0.0,
        "food changes hands within a zone more than chance: {assortativity} {contacts:?}"
    );
    assert_eq!(
        contacts[0][2], 0,
        "food from the entrance reaches the brood only through the workers between: {contacts:?}"
    );
}

#[test]
fn food_is_handed_inward_and_corpses_carried_out() {
    // Greenwald, Segre & Feinerman 2015: foragers unload near the
    // entrance and the food percolates inward by trophallaxis; corpses
    // inside are fetched from where they lie and carried out.
    let mut cfg = SimConfig::default();
    cfg.world.seed = Some(11);
    cfg.world.nest_radius = 3;
    cfg.ants = 60;
    cfg.nest.initial_satiation = 0.05;
    cfg.nest.initial_brood_per_ant = 1.0;
    let mut sim = Simulation::new(cfg, 5);
    let nest = sim.world().nest();
    for _ in 0..2 {
        sim.world_mut().add_corpse(nest);
    }
    sim.run_seconds(30.0 * 60.0);
    let s = sim.stats();
    assert!(s.food_delivered > 0, "{s:?}");
    assert!(
        s.trophallaxis_mg > 0.5 * s.sugar_delivered_mg,
        "the food is handed on: {} mg passed on of {} delivered",
        s.trophallaxis_mg,
        s.sugar_delivered_mg
    );
    let profile = sim.nest_profile();
    assert!(
        profile[0].mean_crop_fill > 0.0,
        "the chamber is fed: {profile:?}"
    );
    assert!(s.activity_ticks[Activity::Nursing.index()] > 0);
    assert!(
        s.corpses_fetched >= 2,
        "undertakers fetch corpses where they lie: {}",
        s.corpses_fetched
    );
    assert_eq!(sim.nest().corpses, 0);
}

fn memo_colony(memoize: bool) -> SimConfig {
    let mut cfg = SimConfig::default();
    cfg.world.seed = Some(2024);
    cfg.ants = 200;
    cfg.nest.initial_satiation = 0.1;
    cfg.nest.mortality = false;
    if let Some(food) = cfg.world.random_food.as_mut() {
        food.renewal_ul_per_s = 0.02;
    }
    cfg.species.consumption_mg_per_ant_per_s = 0.5 / 3600.0;
    cfg.history = Some(HistoryConfig::default());
    cfg.memo = Some(MemoConfig {
        transits: if memoize {
            Some(TransitConfig::default())
        } else {
            None
        },
        ..MemoConfig::default()
    });
    cfg
}

#[test]
fn memo_classifies_trails_apart_from_search_ground() {
    let mut sim = Simulation::new(memo_colony(false), 7);
    sim.run_seconds(20.0 * 60.0);
    let memo = sim.extract_memo().expect("memo on");
    assert!(memo.is_composed());
    let root = memo.signature(QuadKey::ROOT, Layer::Invariant);
    let s = sim.stats();
    let rate = s.decisions as f64 / sim.tick() as f64;
    assert!(
        (root.decisions - rate).abs() < 0.25 * rate,
        "the root composes the colony's decision rate: {} against {rate}",
        root.decisions
    );
    assert!(
        (root.mean_entropy() - s.mean_entropy()).abs() < 0.1,
        "and its mean entropy: {} against {}",
        root.mean_entropy(),
        s.mean_entropy()
    );
    let classes = memo.classify(memo.level(), 3, 1);
    assert_eq!(classes.categories(), 3);
    assert!(classes.labels.len() > 20, "{classes:?}");
    // The busiest category is the trail: straighter and more laden than
    // the most populous one, the search ground.
    let most = (0..3)
        .max_by_key(|&c| classes.sizes[c])
        .expect("three categories");
    assert_ne!(most, 0, "the trail is not the commonest ground");
    assert!(
        classes.centroids[0][2] > classes.centroids[most][2],
        "straighter: {:?}",
        classes.centroids
    );
    assert!(
        classes.centroids[0][7] > classes.centroids[most][7],
        "more laden: {:?}",
        classes.centroids
    );
    assert!(
        memo.render(&classes).contains('0'),
        "{}",
        memo.render(&classes)
    );
    assert_eq!(
        memo.features(memo.level(), Layer::Current).len(),
        memo.keys(memo.level()).len() * ant_simulator::memo::FEATURES
    );
}

#[test]
fn memoized_transits_stand_in_for_the_simulation() {
    // A colony with memoized transits replays a good share of its
    // movement decisions from the kernels and still forages, keeps its
    // decision entropy and its trails, as the full simulation does.
    let mut full = Simulation::new(memo_colony(false), 7);
    full.run_seconds(30.0 * 60.0);
    let mut memoized = Simulation::new(memo_colony(true), 7);
    memoized.run_seconds(30.0 * 60.0);
    let (f, m) = (full.stats(), memoized.stats());
    let transits = memoized
        .memo()
        .and_then(|memo| memo.transits.as_ref())
        .expect("transits on");
    assert!(transits.replayed > 0, "{}", transits.report());
    let share = m.decisions_replayed as f64 / m.decisions as f64;
    assert!(share > 0.1, "replayed share {share}: {}", transits.report());
    let within = |a: f64, b: f64, tol: f64| (a - b).abs() <= tol * a.abs().max(b.abs()).max(1e-9);
    assert!(
        within(f.food_delivered as f64, m.food_delivered as f64, 0.2),
        "deliveries {} full, {} memoized",
        f.food_delivered,
        m.food_delivered
    );
    assert!(
        within(f.mean_entropy(), m.mean_entropy(), 0.15),
        "entropy {} full, {} memoized",
        f.mean_entropy(),
        m.mean_entropy()
    );
    assert!(
        (f.foraging_fraction() - m.foraging_fraction()).abs() < 0.08,
        "outside {} full, {} memoized",
        f.foraging_fraction(),
        m.foraging_fraction()
    );
    let (tf, tm) = (
        full.world().total_pheromone(Pheromone::Trail),
        memoized.world().total_pheromone(Pheromone::Trail),
    );
    assert!(within(tf, tm, 0.3), "trail {tf} full, {tm} memoized");
    // Replayed ants are accounted for like the rest.
    assert_eq!(memoized.alive(), 200);
}
