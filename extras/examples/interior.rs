//! The interior: a higher-order layer of thought that walks the
//! colony's own vocabulary, coupled both ways to the foragers. Run with
//! `--release`.
//!
//! `cargo run --release -p ant_extras --example interior`

use ant_extras::interior::{Colony, Interior, InteriorConfig};
use ant_extras::lexicon::LexiconConfig;
use ant_extras::mind::{Mind, MindConfig};
use ant_extras::problem::Problem;
use ant_extras::problems::Maze;
use ant_extras::topos::{PathNet, SymbolConfig};
use ant_simulator::geometry::Position;
use std::time::Instant;

fn talking(listen: f64, budget: usize) -> MindConfig {
    MindConfig {
        thoughts: 32,
        trip_budget: budget,
        ..MindConfig::default()
    }
    .with_symbols(SymbolConfig::default())
    .with_lexicon(LexiconConfig {
        listen,
        ..LexiconConfig::default()
    })
}

fn classes(net: &PathNet) -> usize {
    net.living()
        .into_iter()
        .filter(|&k| net.symbols()[k].support > 0)
        .count()
}

fn main() {
    let start = Instant::now();

    // A hall with pillars: words of up to five letters, room to compose.
    println!("== a hall with a wall and four pillars, 32 thoughts talking; with and without an interior of 12 thoughts");
    let hall = Maze::hall(64, 40);
    let mut alone = Mind::new(hall.clone(), talking(0.7, 300));
    let mut colony = Colony::new(hall, talking(0.7, 300), InteriorConfig::default());
    for _ in 0..6 {
        alone.run(1000);
        colony.run(1000);
        println!(
            "  tick {:4}: alone quality {:4.0}, {:2} classes; with the interior quality {:4.0}, {:2} classes, {} hypotheses proposed, {} confirmed; the interior {} home of {} trips",
            alone.tick(),
            alone.stats().quality_sum,
            classes(alone.net().unwrap()),
            colony.outer().stats().quality_sum,
            classes(colony.outer().net().unwrap()),
            colony.proposed,
            colony.confirmed,
            colony.inner().stats().deliveries,
            colony.inner().stats().trips
        );
    }
    print!(
        "{}",
        colony
            .report()
            .lines()
            .filter(|l| l.contains("interior")
                || l.contains("belief")
                || l.contains("confirmed")
                || l.contains("parses"))
            .map(|l| format!("{l}\n"))
            .collect::<String>()
    );

    // A third layer over the interior's own classes.
    let parses = colony.inner().net().unwrap();
    let mut third = Interior::new(&InteriorConfig::default());
    third.refresh(parses);
    println!(
        "== a third layer: an interior over the interior's {} classes (its parses)",
        third.known().len()
    );
    for (w, worth, _) in third.known().iter().take(5) {
        println!("  knows [{}] worth {worth:.2}", third.describe(w));
    }
    let mut mind = Mind::new(
        third,
        MindConfig {
            thoughts: 8,
            trip_budget: 10,
            rest_ticks: 16,
            ..MindConfig::default()
        },
    );
    mind.run(2000);
    println!(
        "  after 2000 ticks it brought home {} of {} trips; its best thought is [{}]",
        mind.stats().deliveries,
        mind.stats().trips,
        mind.best()
            .map(|b| mind.problem().describe(b.route.last().unwrap()))
            .unwrap_or_default()
    );

    // The two-source maze on a quiet floor.
    println!("== the maze round a wall, rich source beyond it and poor near the nest, a floor listened to one departure in ten");
    let maze = Maze::around_a_wall(48, 32);
    let rich = maze.goal();
    let problem = maze.with_goals(vec![(rich, 1.0), (Position::new(6, 26), 0.2)]);
    let mut alone = Mind::new(problem.clone(), talking(0.1, 200));
    let mut colony = Colony::new(problem, talking(0.1, 200), InteriorConfig::default());
    alone.run(4000);
    colony.run(4000);
    println!(
        "  alone: quality {:.0}; with the interior: quality {:.0}",
        alone.stats().quality_sum,
        colony.outer().stats().quality_sum
    );
    let mut beliefs = colony.beliefs();
    beliefs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let net = colony.outer().net().unwrap();
    let lex = colony.outer().lexicon().unwrap();
    for (w, level, k) in beliefs.iter().take(3) {
        println!(
            "  belief {level:.2} in [{}], standing dance {:.1} against its own {:.1}",
            net.show(w),
            lex.sign(*k).map(|s| s.standing).unwrap_or(0.0),
            lex.sign(*k).map(|s| s.dance).unwrap_or(0.0)
        );
    }
    println!("  ({:.1} s)", start.elapsed().as_secs_f64());
}
