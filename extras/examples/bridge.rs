//! The bridge made explicit: the colony's forward and backward filters
//! solved on a coarse graph of the maze, the surprise of its flows
//! against them, the trail's agreement with the desirability, the
//! narrowing of the predicted corridor as the temperature falls, the
//! desirability's gradient followed as a drift, sector weights on the
//! symbols at a temperature, and a practice conditioned on survival.
//! Run with `--release`.
//!
//! `cargo run --release -p ant_extras --example bridge`

use ant_extras::bridge::BridgeConfig;
use ant_extras::mind::{Mind, MindConfig, QueenPolicy};
use ant_extras::practice::{Practice, PracticeConfig, Targets};
use ant_extras::problems::Maze;
use ant_extras::topos::SymbolConfig;
use ant_simulator::learner::{HillClimber, Learner, RandomLearner};
use ant_simulator::rotation::RotationSchedule;
use std::time::Instant;

fn main() {
    let problem = Maze::around_a_wall(64, 40);
    let base = MindConfig {
        thoughts: 48,
        trip_budget: 300,
        ..MindConfig::default()
    };

    // A. The bridge as an observer, with the queen reading the surprise.
    println!("== the bridge observing the colony round the wall (temperature 8)");
    let start = Instant::now();
    let mut mind = Mind::new(
        problem.clone(),
        base.clone()
            .with_bridge(BridgeConfig::default())
            .with_queen(QueenPolicy::default()),
    );
    for _ in 0..8 {
        mind.run(500);
        let b = mind.bridge().expect("a bridge");
        println!(
            "  tick {:5}: {:4} brought home, surprise {:.3}, trail~density r {:+.3} (trail~desirability {:+.3}), corridor {:.1} effective blocks, mass {:.4}, dials {:?}",
            mind.tick(),
            mind.stats().deliveries,
            b.surprise(),
            b.correlation(),
            b.correlation_desirability(),
            b.effective_blocks(),
            b.mass(),
            mind.dials().iter().map(|d| (d * 100.0).round() / 100.0).collect::<Vec<_>>()
        );
    }
    print!("{}", mind.report());
    println!("{}", mind.render_bridge().expect("a picture"));

    // C. The forcing pressure: the corridor narrows as the temperature falls.
    println!("== the same bridge relaxed at falling temperatures");
    for t in [64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0] {
        let world_snapshot = mind.world().clone();
        let b = mind.bridge_mut().expect("a bridge");
        b.set_temperature(t);
        b.relax(&world_snapshot);
        println!(
            "  temperature {:6.1}: corridor {:6.1} effective blocks, entropy {:.2} nats, mass {:.5}",
            t,
            b.effective_blocks(),
            b.entropy(),
            b.mass()
        );
    }
    println!("  ({:.1} s)", start.elapsed().as_secs_f64());

    // B. Following the model's drift.
    println!("== following the desirability's gradient, 4000 ticks");
    for (label, gain) in [
        ("trail alone", 0.0),
        ("drift gain 0.5", 0.5),
        ("drift gain 1", 1.0),
        ("drift gain 2", 2.0),
        ("drift gain 4", 4.0),
    ] {
        let cfg = base.clone().with_bridge(BridgeConfig {
            gain,
            ..BridgeConfig::default()
        });
        let mut mind = Mind::new(problem.clone(), cfg);
        mind.run(4000);
        let s = mind.stats();
        println!(
            "  {label:14}: {} brought home, mean trip {:.0} cells, surprise {:.3}, r {:+.3}",
            s.deliveries,
            s.length / s.trips.max(1) as f64,
            mind.bridge().unwrap().surprise(),
            mind.bridge().unwrap().correlation()
        );
    }

    // D. Sector weights at a temperature.
    println!("== sector partition functions on the symbols, 4000 ticks");
    for temperature in [None, Some(100.0), Some(30.0), Some(10.0)] {
        let cfg = base.clone().with_symbols(SymbolConfig {
            temperature,
            ..SymbolConfig::default()
        });
        let mut mind = Mind::new(problem.clone(), cfg);
        mind.run(4000);
        let net = mind.net().unwrap();
        let mut top = net.living();
        top.sort_by_key(|&k| std::cmp::Reverse(net.symbols()[k].support));
        let line: Vec<String> = top
            .iter()
            .take(3)
            .map(|&k| {
                let s = &net.symbols()[k];
                format!(
                    "[{}] ×{} length {:.0} share {:.3} weight {:+.2}",
                    net.show(&s.word),
                    s.support,
                    s.length,
                    s.share,
                    s.weight
                )
            })
            .collect();
        println!(
            "  temperature {:>5}: {} brought home; {}",
            temperature
                .map(|t| format!("{t:.0}"))
                .unwrap_or_else(|| "none".into()),
            mind.stats().deliveries,
            line.join("; ")
        );
    }

    // E. A practice conditioned on survival.
    println!("== a practice conditioned on survival (yield per thought of 6 per turn)");
    let cfg = PracticeConfig {
        mind: MindConfig {
            thoughts: 32,
            trip_budget: 200,
            ..MindConfig::default()
        },
        ticks_per_turn: 1500,
        targets: Targets::Depth(1),
        schedule: RotationSchedule::RandomStatic { period: 2 },
        survival: Some(6.0),
        ..PracticeConfig::default()
    };
    let learners: Vec<Box<dyn Learner>> = vec![
        Box::new(HillClimber::new(0.3)),
        Box::new(RandomLearner::new(0.3)),
    ];
    let mut practice = Practice::new(Maze::around_a_wall(48, 32), cfg, learners, 5).unwrap();
    for _ in 0..8 {
        let t = practice.turn_once();
        println!(
            "  turn {} yield {:.2} survived {} reward {:.0}",
            t.turn, t.yield_, t.survived, t.reward
        );
    }
    print!("  {}", practice.report());
}
