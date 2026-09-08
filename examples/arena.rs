//! Several learners hold levers into the hierarchy; the levers are rotated
//! through a random sequence they never see but can learn.
//!
//!     cargo run --release --example arena [turns] [period] [sucker]
//!
//! Pass `sucker` as the third argument to select directions with a crawling
//! sucker instead of a global draw.

use ant_simulator::prelude::*;

fn main() {
    let mut args = std::env::args().skip(1);
    let turns: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(150);
    let period: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let selection = match args.next().as_deref() {
        Some("sucker") => Selection::Sucker { reach: 8 },
        _ => Selection::Softmax,
    };

    let config = ArenaConfig {
        sim: SimConfig {
            world: WorldConfig {
                seed: Some(2024),
                ..WorldConfig::default()
            },
            trace: true,
            selection,
            ..SimConfig::default()
        },
        steps_per_turn: 300,
        targets: ControlTargets::AllNodes,
        schedule: RotationSchedule::RandomStatic { period },
        feedback: FeedbackScope::Subtree,
        seeding: EpisodeSeeding::Fresh,
        persistent: false,
        reveal_mapping: false,
    };

    let learners: Vec<Box<dyn Learner>> = vec![
        Box::new(PhaseAware::new(HillClimber::new(0.2), 8)),
        Box::new(HillClimber::new(0.2)),
        Box::new(PhaseAware::new(DialBandit::entropy(), 8)),
        Box::new(PhaseAware::new(DialBandit::geometry(), 8)),
        Box::new(PolicyGradient::new(0.02)),
        Box::new(CrossEntropy::new(8, 3).with_init_std(0.2)),
        Box::new(StaticLearner),
    ];

    let mut arena = Arena::new(config, learners, 7).expect("valid arena");
    println!(
        "{} learners, {} controllable surfaces, rotation period {:?}, selection {:?}",
        arena.learners().len(),
        arena.targets().len(),
        arena.rotation().period(),
        selection
    );
    for t in 0..period as u64 {
        let m = arena.rotation().mapping_at(t).unwrap().clone();
        println!("  phase {t}: {}", arena.describe_mapping(&m));
    }
    println!();

    let mut window = Vec::new();
    for turn in 0..turns {
        let record = arena.turn_once().clone();
        window.push(record.colony_reward);
        if (turn + 1) % 10 == 0 {
            let mean = window.iter().sum::<f64>() / window.len() as f64;
            window.clear();
            let periods: Vec<String> = arena
                .learners()
                .iter()
                .map(|l| {
                    l.inferred_period()
                        .map(|p| p.to_string())
                        .unwrap_or_else(|| "-".into())
                })
                .collect();
            println!(
                "turn {:>4}  colony reward (last 10) {:>8.2}  delivered {:>4}  entropy {:.3}  inferred periods [{}]",
                turn + 1,
                mean,
                record.food_delivered,
                record.mean_entropy,
                periods.join(" ")
            );
        }
    }

    println!("\n{}", arena.report());
    println!("final hierarchy:\n{}", arena.hierarchy().describe());

    // Compare on equal footing: fresh episodes, same seeds, no learners.
    let episodes = 12;
    println!("evaluation over {episodes} fresh episodes:");
    println!(
        "  instinct (untouched): {}",
        arena.evaluate(&arena.initial_params(), episodes, 99)
    );
    println!(
        "  learned hierarchy:    {}",
        arena.evaluate(&arena.hierarchy().params(), episodes, 99)
    );
    if let Some((_, best)) = arena.best() {
        println!(
            "  best turn's params:   {}",
            arena.evaluate(best, episodes, 99)
        );
    }
}
