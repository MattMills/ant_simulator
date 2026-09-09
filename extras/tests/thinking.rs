//! The mind at work: the three problems, the habits, the queen, the
//! pipeline, a maze over a box, and a practice.

use ant_extras::prelude::*;
use ant_extras::sense::{F_HEADING, F_ODOUR, F_ROUTE};
use ant_simulator::frame::Frame;
use ant_simulator::geometry::Position;
use ant_simulator::learner::{HillClimber, Learner, StaticLearner};
use ant_simulator::pheromone::Pheromone;
use ant_simulator::pipeline::PipelineConfig;
use ant_simulator::rotation::RotationSchedule;

fn small_mind() -> MindConfig {
    MindConfig {
        thoughts: 32,
        trip_budget: 200,
        ..MindConfig::default()
    }
}

#[test]
fn the_maze_is_solved_and_the_trail_runs_round_the_wall() {
    let problem = Maze::around_a_wall(48, 32);
    let mut mind = Mind::new(problem.clone(), small_mind());
    mind.run(3000);
    let s = mind.stats();
    assert!(s.deliveries >= 50, "solutions come home: {}", s.deliveries);
    let best = mind.best().expect("a finding");
    assert!((best.quality - 1.0).abs() < 1e-9);
    assert!(
        problem.quality(best.route.last().unwrap()).is_some(),
        "the route ends at the goal"
    );
    // Trail lies on the short way round (the gap above the wall) and
    // the goal is reached from the start's side: the field near the
    // goal is stronger than a far corner.
    let world = mind.world();
    let goal = problem.goal();
    let near_goal = world.level(Position::new(goal.x - 2, goal.y), Pheromone::Trail);
    let far_corner = world.level(Position::new(2, 30), Pheromone::Trail);
    assert!(
        near_goal > far_corner,
        "trail near the goal {near_goal} beats a far corner {far_corner}"
    );
    // The wall carries nothing.
    let wall = Position::new(24, 16);
    assert!(!world.is_passable(wall));
    assert_eq!(world.level(wall, Pheromone::Trail), 0.0);
    // Every thought resting or moving stands on passable ground.
    for t in mind.thoughts() {
        assert!(world.is_passable(t.cell()), "thought {} in a wall", t.id);
    }
}

#[test]
fn foresight_shortens_the_trips() {
    let problem = Maze::around_a_wall(48, 32);
    let mut plain = Mind::new(problem.clone(), small_mind());
    let mut seeing = Mind::new(problem, small_mind().with_foresight(3.0));
    plain.run(4000);
    seeing.run(4000);
    let trip = |m: &Mind<Maze>| m.stats().length / m.stats().trips.max(1) as f64;
    assert!(
        seeing.stats().deliveries as f64 >= 0.9 * plain.stats().deliveries as f64,
        "foresight does not cost deliveries: {} vs {}",
        seeing.stats().deliveries,
        plain.stats().deliveries
    );
    assert!(
        trip(&seeing) < trip(&plain) * 1.05,
        "planned trips are no longer: {:.0} vs {:.0}",
        trip(&seeing),
        trip(&plain)
    );
}

#[test]
fn habits_replay_transits_through_plain_ground() {
    let problem = Maze::new(48, 32, Position::new(6, 16), Position::new(41, 16));
    let mut mind = Mind::new(problem, small_mind().with_habits(4));
    mind.run(3000);
    let s = mind.stats();
    assert!(s.deliveries > 20, "{} delivered", s.deliveries);
    assert!(
        s.replayed > 0 && s.decisions_replayed > 0,
        "transits replayed: {} for {} decisions",
        s.replayed,
        s.decisions_replayed
    );
    let (kernels, mature) = mind.memo().unwrap().transits.as_ref().unwrap().maturity();
    assert!(
        kernels > 0 && mature > 0,
        "{kernels} kernels, {mature} mature"
    );
}

#[test]
fn the_pipeline_holds_headings_and_the_queen_moves_the_dials() {
    let problem = Maze::around_a_wall(48, 32);
    let cfg = small_mind()
        .with_pipeline(PipelineConfig {
            cone: 0,
            ..PipelineConfig::default()
        })
        .with_queen(QueenPolicy {
            epoch_ticks: 150,
            warmup_ticks: 300,
            ..QueenPolicy::default()
        });
    let mut mind = Mind::new(problem, cfg);
    let before = mind.dials();
    assert!(
        before.iter().all(|d| (d - 1.0).abs() < 1e-9),
        "moods start neutral"
    );
    mind.run(3000);
    let s = mind.stats();
    assert!(s.held > 0, "some moves were made on a held heading");
    assert!(s.frames.mean_horizon() >= 1.0);
    assert!(s.epochs >= 10, "{} epochs", s.epochs);
    let queen = mind.queen().expect("a queen");
    assert_eq!(queen.thought.len(), 2, "one component per mood");
    let temps = mind.temperatures();
    assert!(
        temps[0] > temps[1],
        "scouts decide hotter than followers: {temps:?}"
    );
    assert!(s.deliveries > 20, "{} delivered", s.deliveries);
}

#[test]
fn a_maze_over_a_box_is_walked_through_its_folds() {
    // A box folded out: the floor with its four walls round it, joined
    // at the corners by portals. The start is on the floor, the goal up
    // the east wall.
    let mut frame = Frame::new(64, 64, 1.0);
    let out = frame.outworld(20, 20, 24, 24, 12, [0.0, 0.0, 0.0], 0.8);
    let east = frame.regions()[out.east].rect;
    let start = Position::new(24, 32);
    let goal = Position::new(east.max.x - 2, (east.min.y + east.max.y) / 2);
    let problem = Maze::on_frame(&frame, start, goal);
    assert!(problem.passable(start) && problem.passable(goal));
    let mut mind = Mind::new(
        problem.clone(),
        MindConfig {
            thoughts: 32,
            trip_budget: 300,
            ..MindConfig::default()
        },
    );
    mind.run(4000);
    let s = mind.stats();
    assert!(s.deliveries > 0, "solutions come home over the fold");
    for t in mind.thoughts() {
        assert!(problem.passable(t.cell()), "thought {} off the net", t.id);
    }
}

#[test]
fn a_tour_beats_the_random_tours_and_its_route_closes() {
    let problem = Tour::random(12, 48, 40, 11);
    let cfg = MindConfig {
        thoughts: 32,
        speed: 2.0,
        trip_budget: 30,
        odour_release: 0.0,
        recruitment: 8.0,
        ..MindConfig::default()
    }
    .with_trail_half_life(150.0)
    .with_weight(F_HEADING, 0.0)
    .with_weight(F_ODOUR, 8.0)
    .with_weight(F_ROUTE, 3.0);
    let mut mind = Mind::new(problem.clone(), cfg);
    mind.run(4000);
    let best = mind.best().expect("a tour");
    let order = problem.order(&best.route);
    assert_eq!(order.len(), problem.len(), "every city once: {order:?}");
    let mut seen = vec![false; problem.len()];
    for &c in &order {
        assert!(!seen[c], "city {c} twice");
        seen[c] = true;
    }
    assert!(
        (problem.reference() / problem.length_of(&order) - best.quality).abs() < 1e-6,
        "the quality is the reference over the length"
    );
    let random = problem.best_random_quality(500, 1);
    assert!(
        best.quality > random,
        "best {:.3} beats the best of 500 random tours {:.3}",
        best.quality,
        random
    );
    assert!(
        mind.stats().mean_quality() > 0.8,
        "tours are mostly near the nearest-neighbour tour: {:.3}",
        mind.stats().mean_quality()
    );
}

#[test]
fn a_colouring_is_found_with_dead_ends_marked_and_retreated_from() {
    let problem = Colouring::planted(16, 3, 0.3, 4, 48, 32, 3);
    let cfg = MindConfig {
        thoughts: 32,
        speed: 2.0,
        trip_budget: 50,
        ..MindConfig::default()
    };
    let mut mind = Mind::new(problem.clone(), cfg);
    mind.run(3000);
    let s = mind.stats();
    let best = mind.best().expect("a colouring");
    let state = best.route.last().unwrap();
    assert_eq!(state.colours.len(), problem.nodes());
    assert!(problem.is_proper(state), "no edge joins one colour");
    assert!(best.quality > 0.0);
    if s.dead_ends > 0 {
        assert!(
            mind.world().total_pheromone(Pheromone::NoEntry) > 0.0,
            "dead ends are marked"
        );
        assert!(s.retreats > 0, "and retreated from");
    }
    assert!(s.deliveries > 10, "{} delivered", s.deliveries);
    // Every state along the best route is proper.
    for st in &best.route {
        assert!(problem.is_proper(st));
    }
}

#[test]
fn a_practice_runs_its_levers_through_a_rotation() {
    let problem = Maze::around_a_wall(40, 28);
    let cfg = PracticeConfig {
        mind: MindConfig {
            thoughts: 24,
            trip_budget: 150,
            ..MindConfig::default()
        },
        ticks_per_turn: 800,
        targets: Targets::Depth(1),
        schedule: RotationSchedule::RandomStatic { period: 2 },
        ..PracticeConfig::default()
    };
    let learners: Vec<Box<dyn Learner>> =
        vec![Box::new(HillClimber::new(0.2)), Box::new(StaticLearner)];
    let mut practice = Practice::new(problem, cfg, learners, 3).expect("a practice");
    let report = practice.run(4);
    assert_eq!(report.turns, 4);
    assert_eq!(practice.targets().len(), 2, "the two castes");
    let nodes: Vec<_> = practice
        .history()
        .iter()
        .map(|t| t.node_of_lever.clone())
        .collect();
    assert_eq!(nodes[0], nodes[2], "the rotation has period 2");
    assert_ne!(nodes[0], nodes[1], "and it rotates");
    assert!(
        practice.history().iter().all(|t| t.reward >= 0.0),
        "rewards are qualities brought home"
    );
    assert!(practice.best().is_some());
}

#[test]
fn the_layered_embedding_places_layers_along_the_width() {
    let layout = Layered::new(64, 40, 5, 3);
    let o = layout.origin();
    let a = layout.place(0, 0);
    let b = layout.place(4, 2);
    assert!(o.x < a.x && a.x < b.x, "layers run left to right");
    assert!(a.y < b.y, "values run down");
    assert!(b.x < 64.0 && b.y < 40.0);
    let cfg = layout.config();
    assert_eq!(cfg.nest, o.cell());
    assert!(cfg.random_food.is_none());
}
