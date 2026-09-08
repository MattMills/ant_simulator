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
    let mut sim = Simulation::new(config, 3);
    sim.run(500);
    let s = sim.stats();
    assert!(s.food_delivered >= 5, "expected foraging to work: {s:?}");
    let target = 0.35 * (8f64).ln();
    assert!(
        (s.mean_entropy() - target).abs() < 0.25,
        "mean entropy {} should track the dial ({target})",
        s.mean_entropy()
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
