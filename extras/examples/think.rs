//! The mind at work on three problems: a maze walked round a wall, a
//! tour of cities, and a graph colouring. Run with `--release`.
//!
//! `cargo run --release -p ant_extras --example think [maze|tour|colouring|all]`

use ant_extras::mind::{Mind, MindConfig, QueenPolicy};
use ant_extras::problem::Problem;
use ant_extras::problems::{Colouring, Maze, Tour};
use ant_extras::sense::{F_HEADING, F_ODOUR, F_ROUTE};
use ant_simulator::pipeline::PipelineConfig;
use std::time::Instant;

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    if which == "maze" || which == "all" {
        maze();
    }
    if which == "tour" || which == "all" {
        tour();
    }
    if which == "colouring" || which == "all" {
        colouring();
    }
}

fn maze() {
    let problem = Maze::around_a_wall(64, 40);
    println!(
        "== maze: {} to {} round a wall, {:.1} cells as the crow flies",
        problem.describe(&problem.start()),
        problem.describe(&problem.goal()),
        problem.crow_flight()
    );
    let cfg = MindConfig {
        thoughts: 48,
        trip_budget: 300,
        ..MindConfig::default()
    };
    for (label, cfg) in [
        ("plain", cfg.clone()),
        ("with habits", cfg.clone().with_habits(4)),
        (
            "with habits, pipeline, foresight and a queen",
            cfg.clone()
                .with_habits(4)
                .with_pipeline(PipelineConfig {
                    cone: 0,
                    ..PipelineConfig::default()
                })
                .with_foresight(3.0)
                .with_queen(QueenPolicy::default()),
        ),
    ] {
        let start = Instant::now();
        let mut mind = Mind::new(problem.clone(), cfg);
        for quarter in 1..=4 {
            mind.run(1000);
            let s = mind.stats();
            println!(
                "  [{label}] tick {}: {} brought home, {} findings, mean trip {:.0} cells",
                mind.tick(),
                s.deliveries,
                s.findings,
                if s.trips > 0 {
                    s.length / s.trips as f64
                } else {
                    0.0
                }
            );
            if quarter == 4 {
                print!("{}", mind.report());
                println!("{}", mind.render());
            }
        }
        println!("  ({:.2} s)", start.elapsed().as_secs_f64());
    }
}

fn tour() {
    let problem = Tour::random(16, 64, 48, 7);
    println!(
        "== tour of {} cities: nearest neighbour {:.1} cells (quality 1), best of 2000 random {:.3}",
        problem.len(),
        problem.reference(),
        problem.best_random_quality(2000, 1)
    );
    // A tour is not walked: its ring has no straight ahead, so the
    // instinct's persistence is dropped, the scent (the nearest city
    // smells strongest) weighs more, the route memory weighs more, the
    // trail forgets faster and only tours near the best recruit.
    let cfg = MindConfig {
        thoughts: 48,
        speed: 2.0,
        trip_budget: 40,
        odour_release: 0.0,
        recruitment: 8.0,
        ..MindConfig::default()
    }
    .with_trail_half_life(150.0)
    .with_weight(F_HEADING, 0.0)
    .with_weight(F_ODOUR, 8.0)
    .with_weight(F_ROUTE, 3.0);
    let start = Instant::now();
    let mut mind = Mind::new(problem.clone(), cfg);
    for _ in 0..8 {
        mind.run(1000);
        let s = mind.stats();
        println!(
            "  tick {}: {} tours home, mean quality {:.3}, best {:.3}",
            mind.tick(),
            s.deliveries,
            s.mean_quality(),
            mind.best().map(|b| b.quality).unwrap_or(0.0)
        );
    }
    print!("{}", mind.report());
    if let Some(b) = mind.best() {
        let order = problem.order(&b.route);
        println!(
            "  order {:?}, length {:.1}",
            order,
            problem.length_of(&order)
        );
    }
    println!("{}", mind.render());
    println!("  ({:.2} s)", start.elapsed().as_secs_f64());
}

fn colouring() {
    // An easy graph, where every walk colours it and the trail selects
    // the colourings with fewer colours, and a harder one, where walks
    // meet dead ends, mark them and retreat.
    for (label, n, density) in [("easy", 20, 0.25), ("hard", 24, 0.3)] {
        let problem = Colouring::planted(n, 3, density, 4, 64, 40, 3);
        println!(
            "== {label} colouring: {} nodes, {} edges, palette {}",
            problem.nodes(),
            problem.edges(),
            problem.palette()
        );
        let cfg = MindConfig {
            thoughts: 48,
            speed: 2.0,
            trip_budget: 60,
            ..MindConfig::default()
        };
        let start = Instant::now();
        let mut mind = Mind::new(problem.clone(), cfg);
        for _ in 0..6 {
            mind.run(1000);
            let s = mind.stats();
            println!(
                "  tick {}: {} colourings home, mean quality {:.3}, {} dead ends ({} retreats), best {:.3}",
                mind.tick(),
                s.deliveries,
                s.mean_quality(),
                s.dead_ends,
                s.retreats,
                mind.best().map(|b| b.quality).unwrap_or(0.0)
            );
        }
        print!("{}", mind.report());
        if label == "hard" {
            println!("{}", mind.render());
        }
        println!("  ({:.2} s)", start.elapsed().as_secs_f64());
    }
}
