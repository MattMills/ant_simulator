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

// ---------------------------------------------------------------------
// The path-topological network

use ant_extras::topos::{SymbolConfig, Word};
use ant_simulator::geometry::Point;

#[test]
fn the_classes_of_routes_round_a_wall_register_themselves_and_are_named_inferred_and_generated() {
    let problem = Maze::around_a_wall(48, 32);
    let cfg = MindConfig {
        thoughts: 32,
        trip_budget: 200,
        ..MindConfig::default()
    }
    .with_symbols(SymbolConfig::default());
    let mut mind = Mind::new(problem.clone(), cfg);
    mind.run(3000);
    let net = mind.net().expect("a network");
    assert!(
        net.punctures()[0].x > 20.0 && net.punctures()[0].x < 28.0,
        "the wall's centre"
    );
    let living = net.living();
    assert!(
        living.len() >= 2,
        "both ways round the wall: {}",
        net.report()
    );
    // The two classes most walked: one crosses the wall's ray (over),
    // the other does not (under).
    let mut top = living.clone();
    top.sort_by_key(|&k| std::cmp::Reverse(net.symbols()[k].support));
    let crosses = |k: usize| net.symbols()[k].word.letters().iter().any(|l| l.abs() == 1);
    let (over, under) = if crosses(top[0]) {
        (top[0], top[1])
    } else {
        (top[1], top[0])
    };
    assert!(crosses(over) && !crosses(under), "{}", net.report());
    // The over class is the short way (the start and goal sit in the
    // upper third): it yields more and weighs more.
    let (so, su) = (&net.symbols()[over], &net.symbols()[under]);
    assert!(
        so.length < su.length,
        "over {:.0} under {:.0}",
        so.length,
        su.length
    );
    assert!(
        so.weight > su.weight,
        "over {:+.2} under {:+.2}",
        so.weight,
        su.weight
    );
    assert!(so.support >= 3 && su.support >= 3);
    assert!(!so.glyph.points.is_empty() && net.field().total(so.channel) > 0.0);
    let (over_word, under_word) = (so.word.clone(), su.word.clone());

    // Names after the fact, and inference on routes nobody walked.
    mind.net_mut().unwrap().name(over, "over the wall");
    mind.net_mut().unwrap().name(under, "under the wall");
    let high = vec![
        Point::new(4.5, 10.5),
        Point::new(24.5, 1.5),
        Point::new(43.5, 10.5),
    ];
    let low = vec![
        Point::new(4.5, 10.5),
        Point::new(24.5, 30.5),
        Point::new(43.5, 10.5),
    ];
    let label = mind.label(&high).unwrap();
    assert_eq!(label.name.as_deref(), Some("over the wall"), "{label:?}");
    let label = mind.label(&low).unwrap();
    assert_eq!(label.name.as_deref(), Some("under the wall"), "{label:?}");
    let twice: Vec<Point> = high
        .iter()
        .chain(low.iter().rev())
        .chain(high.iter())
        .copied()
        .collect();
    let label = mind.label(&twice).unwrap();
    assert!(label.symbol.is_none(), "not a class walked: {label:?}");
    assert!(label.nearest.is_some());

    // Generation by search: a route of each class from the origin to
    // the goal, on passable ground, of the class asked for.
    for (word, name) in [(over_word, "over the wall"), (under_word, "under the wall")] {
        let route = mind
            .generate(&word, problem.goal())
            .expect("a route of the class");
        assert_eq!(route[0].cell(), problem.start());
        assert_eq!(route.last().unwrap().cell(), problem.goal());
        assert!(route.iter().all(|p| problem.passable(p.cell())));
        let label = mind.label(&route).unwrap();
        assert_eq!(
            label.name.as_deref(),
            Some(name),
            "generated {word}: {label:?}"
        );
    }
    // A word nobody could walk within the longest word: none.
    let long = Word::from_letters(&[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]);
    assert!(mind.generate(&long, problem.goal()).is_none());
}

#[test]
fn expressing_a_symbol_makes_the_thoughts_walk_its_class() {
    let problem = Maze::around_a_wall(48, 32);
    let cfg = MindConfig {
        thoughts: 32,
        trip_budget: 200,
        ..MindConfig::default()
    }
    .with_symbols(SymbolConfig::default());
    let mut mind = Mind::new(problem, cfg);
    mind.run(3000);
    let net = mind.net().expect("a network");
    let living = net.living();
    assert!(living.len() >= 2);
    // Of the two classes most walked, the one walked less.
    let mut top = living.clone();
    top.sort_by_key(|&k| std::cmp::Reverse(net.symbols()[k].support));
    let weaker = top[1];
    let share = |mind: &Mind<Maze>, before: &[u32]| {
        let net = mind.net().unwrap();
        let after: Vec<u32> = living.iter().map(|&k| net.symbols()[k].support).collect();
        let total: u32 = after.iter().zip(before).map(|(a, b)| a - b).sum();
        let idx = living.iter().position(|&k| k == weaker).unwrap();
        (
            (after[idx] - before[idx]) as f64 / total.max(1) as f64,
            after,
        )
    };
    let start: Vec<u32> = living.iter().map(|&k| net.symbols()[k].support).collect();
    mind.run(1500);
    let (before, mid) = share(&mind, &start);
    mind.express(weaker, 6.0);
    mind.run(1500);
    let (during, _) = share(&mind, &mid);
    assert!(
        during > before + 0.05,
        "the class's share of trips rises when expressed: {before:.2} -> {during:.2}\n{}",
        mind.net().unwrap().report()
    );
    mind.release(weaker);
}

#[test]
fn tours_fall_into_classes_by_how_they_wind_round_the_cities() {
    let problem = Tour::random(8, 40, 32, 5);
    let cfg = MindConfig {
        thoughts: 24,
        speed: 2.0,
        trip_budget: 24,
        odour_release: 0.0,
        recruitment: 8.0,
        ..MindConfig::default()
    }
    .with_trail_half_life(150.0)
    .with_weight(F_HEADING, 0.0)
    .with_weight(F_ODOUR, 8.0)
    .with_weight(F_ROUTE, 3.0)
    .with_symbols(SymbolConfig {
        max_word: 24,
        learn_punctures: false,
        ..SymbolConfig::default()
    });
    let mut mind = Mind::new(problem.clone(), cfg);
    mind.run(3000);
    let net = mind.net().expect("a network");
    assert_eq!(
        net.punctures().len(),
        problem.len(),
        "the cities are the punctures"
    );
    let living = net.living();
    assert!(!living.is_empty(), "{}", net.report());
    // Every class's glyph is a closed tour of its word, and the best
    // finding's route is of a registered class.
    for &k in &living {
        let s = &net.symbols()[k];
        assert_eq!(net.word(&s.glyph, &s.prefix), s.word);
        assert!(s.support >= 3);
    }
    let best = mind.best().unwrap();
    let label = mind.label(&best.places).unwrap();
    assert!(label.symbol.is_some(), "{label:?}");
    // The classes are weighed by their yield: the best-yielding class
    // has the largest weight.
    let top = living
        .iter()
        .max_by(|&&a, &&b| {
            net.symbols()[a]
                .yield_
                .partial_cmp(&net.symbols()[b].yield_)
                .unwrap()
        })
        .unwrap();
    assert!(living
        .iter()
        .all(|&k| net.symbols()[k].weight <= net.symbols()[*top].weight + 1e-9));
}

// ---------------------------------------------------------------------
// The holonomy embedding

use ant_extras::topos::{move_letter, Route};
use ant_simulator::world::{Edge, Rect, Side};

/// A cylinder: a region whose north edge joins its south edge.
fn cylinder(width: usize, height: usize) -> Frame {
    let mut frame = Frame::new(width, height, 1.0);
    let rect = Rect::new(
        Position::new(1, 1),
        Position::new(width as i32 - 2, height as i32 - 2),
    );
    frame.region(
        "ring",
        rect,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    );
    frame.portal(
        Edge {
            start: Position::new(1, 1),
            end: Position::new(width as i32 - 2, 1),
            side: Side::North,
        },
        Edge {
            start: Position::new(1, height as i32 - 2),
            end: Position::new(width as i32 - 2, height as i32 - 2),
            side: Side::South,
        },
    );
    frame
}

#[test]
fn windings_round_a_cylinder_are_classes_with_transports_of_a_circumference() {
    // The ring is 16 cells tall between its joined edges; the goal is
    // four cells south of the start, so the short way is four cells
    // and the way round through the fold is twelve.
    let frame = cylinder(14, 18);
    let start = Position::new(6, 3);
    let goal = Position::new(6, 7);
    let problem = Maze::on_frame(&frame, start, goal);
    struct Developing(Maze);
    impl Problem for Developing {
        type State = Position;
        fn embedding(&self) -> ant_simulator::world::WorldConfig {
            self.0.embedding()
        }
        fn origin(&self) -> Position {
            self.0.origin()
        }
        fn place(&self, s: &Position) -> Point {
            self.0.place(s)
        }
        fn moves(&self, s: &Position) -> Moves<Position> {
            self.0.moves(s)
        }
        fn quality(&self, s: &Position) -> Option<f64> {
            self.0.quality(s)
        }
        fn locate(&self, p: Point) -> Option<Position> {
            self.0.locate(p)
        }
        fn describe(&self, s: &Position) -> String {
            self.0.describe(s)
        }
        fn name(&self) -> String {
            "cylinder".to_string()
        }
        fn capacity(&self) -> usize {
            2
        }
    }
    let cfg = MindConfig {
        thoughts: 32,
        trip_budget: 120,
        ..MindConfig::default()
    }
    .with_symbols(SymbolConfig {
        learn_punctures: false,
        ..SymbolConfig::default()
    });
    let mut mind = Mind::new(Developing(problem), cfg);
    mind.run(3000);
    let net = mind.net().expect("a network");
    let living = net.living();
    assert!(!living.is_empty(), "{}", net.report());
    // The direct class crosses nothing and develops four cells south;
    // a class round the fold carries the portal's letter and develops
    // a circumference less: its transport is twelve cells north.
    let direct = net.find(&Word::new()).expect("the direct class");
    let d = &net.symbols()[direct];
    assert_eq!(d.vector.len(), 2);
    assert!(
        (d.vector[1] - 4.0).abs() < 1.5,
        "direct transport {:?}",
        d.vector
    );
    assert!(d.vector[0].abs() < 1.5, "direct transport {:?}", d.vector);
    // Express the way round, if the colony found it, or generate it.
    let round = net
        .find(&Word::from_letters(&[move_letter(0, true)]))
        .or_else(|| net.find(&Word::from_letters(&[move_letter(0, false)])));
    if let Some(k) = round {
        let s = &net.symbols()[k];
        assert!(
            (s.vector[1] + 12.0).abs() < 2.0,
            "a winding develops a circumference: {:?}",
            s.vector
        );
        assert!(s.spread < 4.0, "a flat fold: spread {}", s.spread);
    }
    // Generation through the fold: once round, and twice.
    let once = mind
        .generate(&Word::from_letters(&[move_letter(0, true)]), goal)
        .expect("a route once round");
    let label = mind.label_route(&Route::of(once.clone()), &[]).unwrap();
    assert!(
        label.word.is_empty(),
        "the rays see no crossing on a cylinder"
    );
    assert!(
        once.len() >= 12,
        "round the fold is the long way: {} points",
        once.len()
    );
    let twice = mind
        .generate(
            &Word::from_letters(&[move_letter(0, true), move_letter(0, true)]),
            goal,
        )
        .expect("a route twice round");
    assert!(twice.len() > once.len() + 12);
    let none = mind.generate(&Word::new(), goal).expect("the direct route");
    assert!(
        none.len() <= 6,
        "the direct route is four moves: {}",
        none.len()
    );
}

#[test]
fn walking_queries_learns_rules_transports_and_a_geometry_that_predicts() {
    use ant_extras::problems::{Node, Relations};
    use ant_extras::sense::F_SITE;
    use ant_extras::topos::{link_letter, prefix_letter};
    const PARENT: u16 = 0;
    const SIBLING: u16 = 1;
    const GRANDPARENT: u16 = 3;
    let (problem, tests) = Relations::family(8, 64, 48, 12, 0.3, 3);
    assert!(problem.entities() > 50 && tests.len() > 10);
    let before = problem.evaluate(&tests, None);
    let cfg = MindConfig {
        thoughts: 48,
        speed: 16.0,
        trip_budget: 12,
        rest_ticks: 1,
        odour_release: 0.0,
        recruitment: 4.0,
        site_fidelity: false,
        history: None,
        ..MindConfig::default()
    }
    .with_weight(F_HEADING, 0.0)
    .with_weight(F_ODOUR, 8.0)
    .with_weight(F_ROUTE, 0.0)
    .with_weight(F_SITE, 0.0)
    .with_symbols(SymbolConfig {
        support: 3,
        capacity: 128,
        max_word: 5,
        learn_punctures: false,
        ..SymbolConfig::default()
    });
    let mut mind = Mind::new(problem, cfg);
    mind.run(4000);
    let net = mind.net().expect("a network");
    let problem = mind.problem();
    assert!(mind.stats().deliveries > 100, "{}", mind.report());
    assert!(problem.updates > 100, "the geometry learned");
    // The planted grandparent rule is a registered class whose
    // transport is exact: two parent vectors, with no spread.
    let word = Word::from_letters(&[
        prefix_letter(GRANDPARENT as usize),
        link_letter(PARENT as usize, true),
        link_letter(PARENT as usize, true),
    ]);
    let k = net
        .find(&word)
        .unwrap_or_else(|| panic!("grandparent: parent parent\n{}", net.report()));
    let s = &net.symbols()[k];
    assert!(s.support >= 3);
    let twice: Vec<f64> = problem
        .relation_vector(PARENT)
        .iter()
        .map(|v| 2.0 * v)
        .collect();
    let err: f64 = s
        .vector
        .iter()
        .zip(&twice)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    assert!(
        err < 0.5,
        "transport {:?} vs twice parent {:?}",
        s.vector,
        twice
    );
    assert!(s.spread < 0.5, "a flat connection: spread {}", s.spread);
    assert_eq!(net.show(&s.word), "grandparent: parent parent");
    // Prediction of the held-out facts improves greatly on chance.
    let after = problem.evaluate(&tests, Some(net));
    assert!(
        after.mrr > 5.0 * before.mrr.max(0.02),
        "before {before}after {after}"
    );
    assert!(after.hits10 > 0.5, "{after}");
    // The sibling rule closes: parent then parent backwards.
    let sibling = problem.closure(SIBLING, &[(PARENT, true), (PARENT, false)]);
    assert!(sibling < 0.8, "sibling closes to {sibling}");
    // Generation over states: a route of the grandparent rule from a
    // held-out grandparent's head reaches the held-out grandparent.
    let t = tests
        .iter()
        .find(|t| t.relation == GRANDPARENT)
        .expect("a held-out grandparent");
    let from = Node {
        at: t.head,
        head: t.head,
        query: GRANDPARENT,
        hops: 0,
        last: None,
        prev: u32::MAX,
    };
    let tail = t.tail;
    let path = mind
        .generate_states(&word, from, &|n| n.hops >= 2 && n.at == tail, 3)
        .expect("a route of the rule to the held-out fact");
    assert_eq!(path.len(), 3);
    assert_eq!(path.last().unwrap().at, tail);
}

// ---------------------------------------------------------------------
// The bridge

use ant_extras::bridge::BridgeConfig;

#[test]
fn the_bridge_predicts_the_corridor_the_trail_approximates_its_desirability_and_the_queen_reads_the_surprise(
) {
    let problem = Maze::around_a_wall(48, 32);
    let cfg = MindConfig {
        thoughts: 32,
        trip_budget: 200,
        ..MindConfig::default()
    }
    .with_bridge(BridgeConfig::default())
    .with_queen(QueenPolicy {
        epoch_ticks: 150,
        warmup_ticks: 300,
        ..QueenPolicy::default()
    });
    let mut mind = Mind::new(problem.clone(), cfg);
    mind.run(3000);
    let b = mind.bridge().expect("a bridge");
    assert!(b.has_findings(), "the colony's findings are the sinks");
    assert!(b.epochs() >= 20);
    assert!(b.mass() > 0.0);
    // The predicted density runs through the gap above the wall (the
    // short way) more than below it.
    let above = b.block_of(Position::new(24, 1)).unwrap();
    let below = b.block_of(Position::new(24, 30)).unwrap();
    assert!(
        b.density()[above] > b.density()[below],
        "above {} > below {}",
        b.density()[above],
        b.density()[below]
    );
    let wall = b.block_of(Position::new(24, 16)).unwrap();
    assert!(b.density()[wall] <= b.density()[above]);
    // The trail is the colony's implicit backward filter: it
    // correlates with the explicit desirability.
    assert!(b.correlation() > 0.3, "r = {}", b.correlation());
    // The surprise is a number in (0, 1), and the queen read it.
    let surprise = b.surprise();
    assert!(surprise > 0.0 && surprise < 1.0, "surprise {surprise}");
    assert!(mind.queen().is_some());
    assert!(mind.stats().deliveries > 20);
    // The drift points along the corridor at the start: eastwards.
    let (dx, _) = b
        .drift(Point::center_of(problem.start()))
        .expect("a drift at the start");
    assert!(dx > 0.0, "drift {dx}");
    // Colder is narrower.
    let warm = b.effective_blocks();
    let world = mind.world().clone();
    let b = mind.bridge_mut().unwrap();
    b.set_temperature(5.0);
    b.relax(&world);
    assert!(
        b.effective_blocks() < warm,
        "{} < {warm}",
        b.effective_blocks()
    );
}

/// The drift is a bias, not a controller: at a small gain it shortens
/// the trips, at a large one the block-level field traps the thoughts
/// (the example shows the curve).
#[test]
fn following_the_drift_does_not_cost_deliveries() {
    let problem = Maze::around_a_wall(48, 32);
    let plain = MindConfig {
        thoughts: 32,
        trip_budget: 200,
        ..MindConfig::default()
    };
    let mut without = Mind::new(problem.clone(), plain.clone());
    let mut with = Mind::new(
        problem,
        plain.with_bridge(BridgeConfig {
            gain: 0.5,
            ..BridgeConfig::default()
        }),
    );
    without.run(3000);
    with.run(3000);
    assert!(
        with.stats().deliveries as f64 >= 0.85 * without.stats().deliveries as f64,
        "with the drift {} vs without {}",
        with.stats().deliveries,
        without.stats().deliveries
    );
}

#[test]
fn sector_weights_at_a_temperature_favour_the_short_class_and_sharpen_as_it_falls() {
    let problem = Maze::around_a_wall(48, 32);
    let gap = |temperature: f64| -> (f64, f64) {
        let cfg = MindConfig {
            thoughts: 32,
            trip_budget: 200,
            ..MindConfig::default()
        }
        .with_symbols(SymbolConfig {
            temperature: Some(temperature),
            ..SymbolConfig::default()
        });
        let mut mind = Mind::new(problem.clone(), cfg);
        mind.run(3000);
        let net = mind.net().unwrap();
        let mut top = net.living();
        top.sort_by_key(|&k| std::cmp::Reverse(net.symbols()[k].support));
        assert!(top.len() >= 2, "{}", net.report());
        let (a, b) = (&net.symbols()[top[0]], &net.symbols()[top[1]]);
        let (short, long) = if a.length < b.length { (a, b) } else { (b, a) };
        assert!(short.share > long.share, "{}", net.report());
        assert!(short.weight > long.weight);
        let total: f64 = net.living().iter().map(|&k| net.symbols()[k].share).sum();
        assert!((total - 1.0).abs() < 1e-6, "shares sum to one: {total}");
        (short.share, long.share)
    };
    let (short_warm, long_warm) = gap(100.0);
    let (short_cold, long_cold) = gap(10.0);
    assert!(
        short_cold / long_cold.max(1e-9) > short_warm / long_warm.max(1e-9),
        "colder sharpens: {short_cold}/{long_cold} vs {short_warm}/{long_warm}"
    );
}

#[test]
fn a_practice_conditioned_on_survival_rewards_only_survival() {
    let cfg = PracticeConfig {
        mind: MindConfig {
            thoughts: 24,
            trip_budget: 150,
            ..MindConfig::default()
        },
        ticks_per_turn: 800,
        targets: Targets::Depth(1),
        schedule: RotationSchedule::RandomStatic { period: 2 },
        survival: Some(1.0),
        ..PracticeConfig::default()
    };
    let learners: Vec<Box<dyn Learner>> =
        vec![Box::new(HillClimber::new(0.2)), Box::new(StaticLearner)];
    let mut practice = Practice::new(Maze::around_a_wall(40, 28), cfg, learners, 3).unwrap();
    let report = practice.run(3);
    for t in practice.history() {
        assert!(
            t.reward == 0.0 || t.reward == 1.0,
            "an indicator: {}",
            t.reward
        );
        assert_eq!(t.survived, t.yield_ >= 1.0);
        assert_eq!(t.reward, if t.survived { 1.0 } else { 0.0 });
    }
    assert_eq!(
        report.survived,
        practice.history().iter().filter(|t| t.survived).count()
    );
}

// ---------------------------------------------------------------------
// The lexicon

use ant_extras::lexicon::LexiconConfig;

/// A maze with a rich source beyond the wall (reached over it or under
/// it) and a poor one on the near side.
fn two_sources() -> Maze {
    let maze = Maze::around_a_wall(48, 32);
    let rich = maze.goal();
    let poor = Position::new(6, 26);
    maze.with_goals(vec![(rich, 1.0), (poor, 0.2)])
}

#[test]
fn signs_emerge_carry_their_outcomes_recruit_and_form_synonyms() {
    let problem = two_sources();
    let cfg = MindConfig {
        thoughts: 32,
        trip_budget: 200,
        ..MindConfig::default()
    }
    .with_symbols(SymbolConfig::default())
    .with_lexicon(LexiconConfig::default());
    let mut mind = Mind::new(problem.clone(), cfg);
    mind.run(4000);
    let net = mind.net().expect("a network");
    let lex = mind.lexicon().expect("a lexicon");
    let living = net.living();
    assert!(lex.utterances > 50, "dances: {}", lex.utterances);
    assert!(
        lex.recruitments > 20,
        "departures listened: {}",
        lex.recruitments
    );
    assert!(lex.in_use(&living) >= 2, "{}", lex.report(net));
    // The signs tell where a trip ends: the mutual information between
    // sign and outcome is most of the outcome's entropy.
    let (mi, h) = lex.mutual_information(&living);
    assert!(
        h > 0.1,
        "two sources are two outcomes: H {h}\n{}",
        lex.report(net)
    );
    assert!(
        mi > 0.6 * h,
        "signs mean places: I {mi} of H {h}\n{}",
        lex.report(net)
    );
    // Recruitment is understood more often than not.
    let (successes, tried) =
        living
            .iter()
            .filter_map(|&k| lex.sign(k))
            .fold((0u64, 0u64), |(s, t), sign| {
                (
                    s + sign.successes,
                    t + sign.successes + sign.strayed + sign.lost,
                )
            });
    assert!(tried > 10);
    assert!(
        successes as f64 > 0.5 * tried as f64,
        "understood {successes} of {tried}\n{}",
        lex.report(net)
    );
    // The dance floor favours the rich source (it is worth more per
    // trip): the signs meaning its block are the ones danced.
    let rich_block = lex.block_of(problem.goal());
    let poor_block = lex.block_of(Position::new(6, 26));
    let means = |k: usize, block: usize| {
        lex.meaning(k)
            .first()
            .map(|(b, _)| *b == block)
            .unwrap_or(false)
    };
    let rich_signs: Vec<usize> = living
        .iter()
        .copied()
        .filter(|&k| means(k, rich_block))
        .collect();
    let poor_signs: Vec<usize> = living
        .iter()
        .copied()
        .filter(|&k| means(k, poor_block))
        .collect();
    assert!(
        !rich_signs.is_empty(),
        "a sign for the rich source\n{}",
        lex.report(net)
    );
    assert!(
        !poor_signs.is_empty(),
        "a sign for the poor source\n{}",
        lex.report(net)
    );
    let dance =
        |signs: &[usize]| -> f64 { signs.iter().map(|&k| lex.sign(k).unwrap().dance).sum() };
    let (rich_dance, poor_dance) = (dance(&rich_signs), dance(&poor_signs));
    assert!(
        rich_dance > 4.0 * poor_dance,
        "the rich source danced {rich_dance:.1} against the poor {poor_dance:.1}\n{}",
        lex.report(net)
    );
    // Recruits to the rich source's signs got there: what the signs
    // are taken to mean is what they mean.
    for &k in &rich_signs {
        if lex.sign(k).unwrap().taken_count >= 10 {
            assert!(
                lex.understanding(k) > 0.9,
                "understood: {}\n{}",
                lex.understanding(k),
                lex.report(net)
            );
        }
    }
    // Two ways to the rich source are synonyms; a sign for the poor
    // source is not.
    if rich_signs.len() >= 2 {
        assert!(
            lex.similarity(rich_signs[0], rich_signs[1]) > 0.9,
            "synonyms: {}",
            lex.similarity(rich_signs[0], rich_signs[1])
        );
    }
    if let (Some(&r), Some(&p)) = (rich_signs.first(), poor_signs.first()) {
        assert!(
            lex.similarity(r, p) < 0.5,
            "different places: {}",
            lex.similarity(r, p)
        );
    }
}

#[test]
fn a_lexicon_raises_the_yield_by_recruiting_to_the_rich_source() {
    let problem = two_sources();
    let base = MindConfig {
        thoughts: 32,
        trip_budget: 200,
        ..MindConfig::default()
    }
    .with_symbols(SymbolConfig::default());
    let mut silent = Mind::new(problem.clone(), base.clone());
    let mut talking = Mind::new(problem, base.with_lexicon(LexiconConfig::default()));
    silent.run(4000);
    talking.run(4000);
    let (qs, qt) = (silent.stats().quality_sum, talking.stats().quality_sum);
    assert!(
        qt > 1.1 * qs,
        "quality brought home with a lexicon {qt:.0} against without {qs:.0}"
    );
}

// ---------------------------------------------------------------------
// The echo

use ant_extras::echo::EchoConfig;

/// A talking mind on the two-source maze, with or without echoes, and
/// a floor listened to with the given chance.
fn talking(echoes: usize, listen: f64) -> MindConfig {
    let mut cfg = MindConfig {
        thoughts: 32,
        trip_budget: 200,
        ..MindConfig::default()
    }
    .with_symbols(SymbolConfig::default())
    .with_lexicon(LexiconConfig {
        listen,
        ..LexiconConfig::default()
    });
    if echoes > 0 {
        cfg = cfg.with_echo(EchoConfig {
            echoes,
            ..EchoConfig::default()
        });
    }
    cfg
}

/// The living signs that mean the rich source.
fn rich_signs(mind: &Mind<Maze>, rich: Position) -> Vec<usize> {
    let (net, lex) = (mind.net().unwrap(), mind.lexicon().unwrap());
    let block = lex.block_of(rich);
    net.living()
        .into_iter()
        .filter(|&k| {
            lex.meaning(k)
                .first()
                .map(|(b, _)| *b == block)
                .unwrap_or(false)
        })
        .collect()
}

/// The shortest glyph among the rich source's signs, and the share of
/// its cells that lie in the movement history's invariant skeleton.
fn rich_glyph(mind: &Mind<Maze>, rich: Position) -> Option<(f64, f64)> {
    let net = mind.net().unwrap();
    let (k, length) = rich_signs(mind, rich)
        .into_iter()
        .map(|k| (k, net.symbols()[k].glyph.length()))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())?;
    let mask = mind.history()?.channel_mask();
    let w = mind.problem().config().width;
    let mut cells: Vec<Position> = net.symbols()[k]
        .glyph
        .points
        .iter()
        .map(|p| p.cell())
        .collect();
    cells.dedup();
    let inside = cells
        .iter()
        .filter(|c| {
            mask.get(c.y as usize * w + c.x as usize)
                .copied()
                .unwrap_or(false)
        })
        .count();
    Some((length, inside as f64 / cells.len().max(1) as f64))
}

#[test]
fn echoes_carry_the_sign_where_the_floor_does_not_reach() {
    // A floor few listen to: the sign has to travel some other way.
    let problem = two_sources();
    let mut quiet = Mind::new(problem.clone(), talking(0, 0.1));
    let mut echoing = Mind::new(problem, talking(4, 0.1));
    quiet.run(4000);
    echoing.run(4000);
    let (q0, q1) = (quiet.stats().quality_sum, echoing.stats().quality_sum);
    assert!(
        q1 > 1.15 * q0,
        "quality with echoes {q1:.0} against {q0:.0} without, on a floor few listen to"
    );
    let e = echoing.echo().expect("echoes");
    assert!(e.walks > 100, "walks: {}", e.walks);
    assert!(e.heard > 200, "heard on the way: {}", e.heard);
    assert!(
        e.heard_success_rate() > 0.8,
        "understood on the way: {:.2}",
        e.heard_success_rate()
    );
    assert!(
        e.verified > e.empty,
        "the echoes' walks found the source {} times and nothing {} times",
        e.verified,
        e.empty
    );
}

#[test]
fn echo_traffic_makes_the_route_invariant_early_and_refines_the_glyph() {
    let problem = two_sources();
    let rich = problem.goal();
    let mut plain = Mind::new(problem.clone(), talking(0, 0.7));
    let mut echoing = Mind::new(problem, talking(4, 0.7));
    plain.run(1000);
    echoing.run(1000);
    let (_, c0) = rich_glyph(&plain, rich).expect("a rich sign without echoes");
    let (_, c1) = rich_glyph(&echoing, rich).expect("a rich sign with echoes");
    assert!(
        c1 >= 0.4 && c1 > c0 + 0.2,
        "of the rich source's glyph, {:.0}% lies in the invariant skeleton at tick 1000 with echoes against {:.0}% without",
        100.0 * c1,
        100.0 * c0
    );
    plain.run(5000);
    echoing.run(5000);
    let (g0, _) = rich_glyph(&plain, rich).unwrap();
    let (g1, _) = rich_glyph(&echoing, rich).unwrap();
    assert!(
        g1 <= g0 + 1e-9,
        "the glyph walked again and again is no longer: {g1:.0} against {g0:.0}"
    );
    let e = echoing.echo().unwrap();
    assert!(e.refined > 0, "walks that refined a glyph: {}", e.refined);
    // Four thoughts of thirty-two echoing cost the colony nothing.
    let (q0, q1) = (plain.stats().quality_sum, echoing.stats().quality_sum);
    assert!(
        q1 > 0.95 * q0,
        "quality with echoes {q1:.0} against {q0:.0} without"
    );
}

#[test]
fn new_thoughts_are_imprinted_from_the_ground() {
    let problem = two_sources();
    let rich = problem.goal();
    let mut mind = Mind::new(problem, talking(4, 0.7));
    mind.run(2500);
    // Every thought but the echoes forgets what it knew, and the floor
    // falls silent: what the colony knows is in the ground and in the
    // echoes' walks.
    mind.lexicon_mut().unwrap().silence();
    let renewed = mind.renew(1.0);
    assert!(renewed >= 27, "renewed: {renewed}");
    let q0 = mind.stats().quality_sum;
    mind.run(200);
    let first = mind.stats().quality_sum - q0;
    assert!(
        first > 25.0,
        "the first window after the renewal brought home {first:.0}"
    );
    mind.run(1400);
    let lex = mind.lexicon().unwrap();
    let dance: f64 = rich_signs(&mind, rich)
        .iter()
        .map(|&k| lex.sign(k).unwrap().dance)
        .sum();
    assert!(
        dance > 50.0,
        "the floor re-seeded from memory: rich dance {dance:.1}"
    );
    let e = mind.echo().unwrap();
    assert!(e.heard > 100, "heard on the way: {}", e.heard);
    assert!(
        e.verified > 50,
        "walks that found the source: {}",
        e.verified
    );
}
