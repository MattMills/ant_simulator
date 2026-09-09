//! An emergent lexicon: the classes of routes the colony registers
//! become signs when returning thoughts dance them at the nest and
//! departing thoughts listen; a sign's meaning is where its trips end;
//! synonyms are signs that lead to the same place; and the embedding
//! is that meaning beside the transport. Run with `--release`.
//!
//! `cargo run --release -p ant_extras --example lexicon`

use ant_extras::lexicon::LexiconConfig;
use ant_extras::mind::{Mind, MindConfig};
use ant_extras::problems::Maze;
use ant_extras::topos::SymbolConfig;
use ant_simulator::geometry::Position;
use std::time::Instant;

fn main() {
    // A rich source beyond the wall, reachable over it or under it, and
    // a poor one on the near side.
    let maze = Maze::around_a_wall(64, 40);
    let rich = maze.goal();
    let poor = Position::new(11, 35);
    let problem = maze.with_goals(vec![(rich, 1.0), (poor, 0.2)]);
    println!(
        "== two sources: rich at ({},{}) beyond the wall, poor at ({},{}) on the near side",
        rich.x, rich.y, poor.x, poor.y
    );
    let base = MindConfig {
        thoughts: 48,
        trip_budget: 300,
        ..MindConfig::default()
    }
    .with_symbols(SymbolConfig::default());

    let start = Instant::now();
    let mut mind = Mind::new(
        problem.clone(),
        base.clone().with_lexicon(LexiconConfig::default()),
    );
    for _ in 0..6 {
        mind.run(1000);
        let net = mind.net().unwrap();
        let lex = mind.lexicon().unwrap();
        let living = net.living();
        let (mi, h) = lex.mutual_information(&living);
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
        println!(
            "  tick {:5}: {:4} brought home (quality {:6.1}), {} signs in use, usage entropy {:.2} nats, sign tells {:.2} of {:.2} nats, understood {:.0}% of {}",
            mind.tick(),
            mind.stats().deliveries,
            mind.stats().quality_sum,
            lex.in_use(&living),
            lex.usage_entropy(&living),
            mi,
            h,
            if tried > 0 { 100.0 * successes as f64 / tried as f64 } else { 0.0 },
            tried
        );
    }
    print!("{}", mind.report());

    // Naming after the fact, and synonyms by meaning.
    let net = mind.net().unwrap();
    let lex = mind.lexicon().unwrap();
    let living = net.living();
    let rich_block = lex.block_of(rich);
    let poor_block = lex.block_of(poor);
    let leads_to = |k: usize, block: usize| {
        lex.meaning(k)
            .first()
            .map(|(b, _)| *b == block)
            .unwrap_or(false)
    };
    let mut rich_signs: Vec<usize> = living
        .iter()
        .copied()
        .filter(|&k| leads_to(k, rich_block))
        .collect();
    rich_signs.sort_by_key(|&k| std::cmp::Reverse(net.symbols()[k].support));
    let poor_signs: Vec<usize> = living
        .iter()
        .copied()
        .filter(|&k| leads_to(k, poor_block))
        .collect();
    println!(
        "  signs meaning the rich source: {:?}; the poor source: {:?}",
        rich_signs
            .iter()
            .map(|&k| format!("#{k} [{}]", net.show(&net.symbols()[k].word)))
            .collect::<Vec<_>>(),
        poor_signs
            .iter()
            .map(|&k| format!("#{k} [{}]", net.show(&net.symbols()[k].word)))
            .collect::<Vec<_>>()
    );
    if let Some(&first) = rich_signs.first() {
        println!(
            "  sign #{first} is taken to mean what it means at agreement {:.3} ({} set out with it in mind)",
            lex.understanding(first),
            lex.sign(first).map(|s| s.taken_count).unwrap_or(0)
        );
        if let Some((nearest, similarity)) = lex.nearest(first, &living) {
            println!(
                "  naming #{first} \"the rich source\": its nearest sign in meaning is #{nearest} [{}] at similarity {similarity:.3}{}",
                net.show(&net.symbols()[nearest].word),
                if similarity > 0.9 { ", a synonym: another way there" } else { "" }
            );
        }
        if let Some(&p) = poor_signs.first() {
            println!(
                "  and the poor source's sign #{p} is at similarity {:.3} to it: not a synonym",
                lex.similarity(first, p)
            );
        }
    }
    println!("  ({:.1} s)", start.elapsed().as_secs_f64());

    // Does talking help? Quality brought home with and without the lexicon.
    println!("== with and without the lexicon, 6000 ticks");
    for (label, cfg) in [
        ("silent", base.clone()),
        (
            "talking",
            base.clone().with_lexicon(LexiconConfig::default()),
        ),
    ] {
        let mut mind = Mind::new(problem.clone(), cfg);
        mind.run(6000);
        let s = mind.stats();
        println!(
            "  {label:8}: {} brought home, quality {:.1}, mean quality {:.3}",
            s.deliveries,
            s.quality_sum,
            s.mean_quality()
        );
    }
}
