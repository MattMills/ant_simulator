//! The path-topological network at work: the classes of routes round a
//! wall register themselves as symbols, are named after the fact,
//! label new routes (inference), and are generated as routes again
//! (generation), by search and by the colony; and round a pillar nobody
//! declared, the punctures are learned from the holes the walks
//! enclose. Run with `--release`.
//!
//! `cargo run --release -p ant_extras --example symbols`

use ant_extras::mind::{Mind, MindConfig};
use ant_extras::problem::Problem;
use ant_extras::problems::{Maze, Tour};
use ant_extras::sense::{F_HEADING, F_ODOUR, F_ROUTE};
use ant_extras::topos::SymbolConfig;
use ant_simulator::geometry::{Point, Position};
use ant_simulator::world::Rect;

fn main() {
    round_a_wall();
    round_a_pillar();
    tours();
}

fn round_a_wall() {
    let problem = Maze::around_a_wall(64, 40);
    println!(
        "== round a wall: {} to {}, puncture at the wall's centre {:?}",
        problem.describe(&problem.start()),
        problem.describe(&problem.goal()),
        problem.punctures()[0]
    );
    let cfg = MindConfig {
        thoughts: 48,
        trip_budget: 300,
        ..MindConfig::default()
    }
    .with_symbols(SymbolConfig::default());
    let mut mind = Mind::new(problem.clone(), cfg);
    mind.run(4000);
    print!("{}", mind.report());

    // Naming after the fact: the two classes most walked; the one whose
    // word crosses the wall's ray (letter `a`, the first puncture) went
    // over it, the other under.
    let net = mind.net().expect("a network");
    let mut top = net.living();
    top.sort_by_key(|&k| std::cmp::Reverse(net.symbols()[k].support));
    let (first, second) = (top[0], top[1]);
    let crosses_wall = |k: usize| net.symbols()[k].word.letters().iter().any(|l| l.abs() == 1);
    let (over, under) = if crosses_wall(first) {
        (first, second)
    } else {
        (second, first)
    };
    let (over_word, under_word) = (
        net.symbols()[over].word.clone(),
        net.symbols()[under].word.clone(),
    );
    if let Some(net) = mind.net_mut() {
        net.name(over, "over the wall");
        net.name(under, "under the wall");
    }

    // Inference: routes nobody walked, labelled by their class.
    let h = 40.0;
    let high = vec![
        Point::new(4.5, 13.5),
        Point::new(32.5, 1.5),
        Point::new(59.5, 13.5),
    ];
    let low = vec![
        Point::new(4.5, 13.5),
        Point::new(32.5, h - 1.5),
        Point::new(59.5, 13.5),
    ];
    let twice = vec![
        Point::new(4.5, 13.5),
        Point::new(32.5, 1.5),
        Point::new(59.5, 13.5),
        Point::new(32.5, h - 1.5),
        Point::new(4.5, 13.5),
        Point::new(32.5, 1.5),
        Point::new(59.5, 13.5),
    ];
    for (what, route) in [
        ("a high route", &high),
        ("a low route", &low),
        ("over, back under, over again", &twice),
    ] {
        let label = mind.label(route).expect("a label");
        println!(
            "  {what}: word [{}] -> symbol {:?} {}{}",
            label.word,
            label.symbol,
            label.name.as_deref().unwrap_or("(unnamed)"),
            match label.nearest {
                Some((k, d)) if label.symbol.is_none() => format!(", nearest #{k} at {d} letters"),
                _ => String::new(),
            }
        );
    }

    // Generation by search: a route of each class, checked by its word.
    for (name, word) in [("over the wall", over_word), ("under the wall", under_word)] {
        match mind.generate(&word, problem.goal()) {
            Some(route) => {
                let label = mind.label(&route).expect("a label");
                let length: f64 = route.windows(2).map(|p| p[0].distance(p[1])).sum();
                println!(
                    "  generated \"{name}\" from [{word}]: {} points, {length:.0} cells, labelled {}",
                    route.len(),
                    label.name.as_deref().unwrap_or("?")
                );
            }
            None => println!("  generated \"{name}\" from [{word}]: none within the longest word"),
        }
    }

    // Generation by the colony: express the class the mind walks less
    // (under the wall), and count what comes home.
    let net = mind.net().expect("a network");
    let living = net.living();
    let support_before: Vec<u32> = living.iter().map(|&k| net.symbols()[k].support).collect();
    let share = |before: &[u32], after: &[u32], k: usize| {
        let total: u32 = after.iter().zip(before).map(|(a, b)| a - b).sum();
        let mine = after[k] - before[k];
        if total == 0 {
            0.0
        } else {
            mine as f64 / total as f64
        }
    };
    let idx = living.iter().position(|&k| k == under).expect("in living");
    mind.run(1500);
    let net = mind.net().expect("a network");
    let support_mid: Vec<u32> = living.iter().map(|&k| net.symbols()[k].support).collect();
    let before_share = share(&support_before, &support_mid, idx);
    mind.express(under, 6.0);
    mind.run(1500);
    let net = mind.net().expect("a network");
    let support_after: Vec<u32> = living.iter().map(|&k| net.symbols()[k].support).collect();
    let while_share = share(&support_mid, &support_after, idx);
    mind.release(under);
    mind.run(1500);
    let net = mind.net().expect("a network");
    let support_later: Vec<u32> = living.iter().map(|&k| net.symbols()[k].support).collect();
    let after_share = share(&support_after, &support_later, idx);
    println!(
        "  expressing \"under the wall\": its share of the trips brought home {:.0}% before, {:.0}% while expressed, {:.0}% after release",
        100.0 * before_share,
        100.0 * while_share,
        100.0 * after_share
    );
    if let Some(picture) = mind.render_symbol(under) {
        println!("{picture}");
    }
}

fn round_a_pillar() {
    // No puncture is given: the pillar is a wall the maze does not
    // declare, and the walks round it enclose a hole the network learns.
    let start = Position::new(4, 20);
    let goal = Position::new(59, 20);
    let pillar = Rect::new(Position::new(29, 16), Position::new(34, 23));
    let mut problem = Maze::new(64, 40, start, goal).with_walls(vec![pillar]);
    println!("== round a pillar nobody declared");
    let cfg = MindConfig {
        thoughts: 48,
        trip_budget: 300,
        ..MindConfig::default()
    }
    .with_symbols(SymbolConfig::default());
    // The maze's own punctures (its walls) are withheld: an open
    // problem with a pillar in it.
    struct Undeclared(Maze);
    impl Problem for Undeclared {
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
        fn moves(&self, s: &Position) -> ant_extras::problem::Moves<Position> {
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
            "maze with an undeclared pillar".to_string()
        }
    }
    let _ = &mut problem;
    let mut mind = Mind::new(Undeclared(problem), cfg);
    for quarter in 1..=4 {
        mind.run(1500);
        let net = mind.net().expect("a network");
        println!(
            "  tick {}: holes {:?}; {} punctures learned, {} symbols living: {}",
            mind.tick(),
            net.holes()
                .iter()
                .map(|(p, n)| format!("({:.0},{:.0})×{n}", p.x, p.y))
                .collect::<Vec<_>>(),
            net.learned_punctures(),
            net.living().len(),
            net.living()
                .iter()
                .map(|&k| format!("[{}]×{}", net.symbols()[k].word, net.symbols()[k].support))
                .collect::<Vec<_>>()
                .join(" ")
        );
        if quarter == 4 {
            print!("{}", mind.report());
        }
    }
}

fn tours() {
    let problem = Tour::random(10, 48, 40, 5);
    println!(
        "== tours of {} cities: the cities are the punctures, a tour's class is how it winds round them",
        problem.len()
    );
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
    .with_weight(F_ROUTE, 3.0)
    .with_symbols(SymbolConfig {
        max_word: 24,
        learn_punctures: false,
        ..SymbolConfig::default()
    });
    let mut mind = Mind::new(problem, cfg);
    mind.run(6000);
    print!("{}", mind.report());
}
