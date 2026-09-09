//! The echo: signs persisted in walk form. Followers that hold a sign
//! for an epoch and walk its glyph again and again, for the others to
//! hear on the way; the route enters the invariant skeleton early, the
//! glyph is refined by the walks, and thoughts newly born into the
//! colony are imprinted from the ground. Run with `--release`.
//!
//! `cargo run --release -p ant_extras --example echo`

use ant_extras::echo::EchoConfig;
use ant_extras::lexicon::LexiconConfig;
use ant_extras::mind::{Mind, MindConfig};
use ant_extras::problems::Maze;
use ant_extras::topos::SymbolConfig;
use ant_simulator::geometry::Position;
use std::time::Instant;

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
/// its cells in the invariant skeleton.
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

fn main() {
    let start = Instant::now();
    let maze = Maze::around_a_wall(48, 32);
    let rich = maze.goal();
    let poor = Position::new(6, 26);
    let problem = maze.with_goals(vec![(rich, 1.0), (poor, 0.2)]);
    println!(
        "== the maze round a wall: rich source at ({},{}) beyond it, poor at ({},{}) near the nest; 32 thoughts",
        rich.x, rich.y, poor.x, poor.y
    );

    // Reach: a floor few listen to.
    println!("== a floor listened to one departure in ten, 4000 ticks: the sign travels by echo or not at all");
    for echoes in [0usize, 4, 8] {
        let mut mind = Mind::new(problem.clone(), talking(echoes, 0.1));
        mind.run(4000);
        let s = mind.stats();
        let heard = mind
            .echo()
            .map(|e| {
                format!(
                    ", {} thoughts heard a sign on the way, {:.0}% came home in its class",
                    e.heard,
                    100.0 * e.heard_success_rate()
                )
            })
            .unwrap_or_default();
        println!(
            "  {echoes} echoes: {} brought home, quality {:.0}{heard}",
            s.deliveries, s.quality_sum
        );
    }

    // Invariance and refinement.
    println!("== the route in the invariant skeleton, and the glyph, over 6000 ticks");
    let mut plain = Mind::new(problem.clone(), talking(0, 0.7));
    let mut echoing = Mind::new(problem.clone(), talking(4, 0.7));
    for _ in 0..6 {
        plain.run(1000);
        echoing.run(1000);
        let (g0, c0) = rich_glyph(&plain, rich).unwrap_or((0.0, 0.0));
        let (g1, c1) = rich_glyph(&echoing, rich).unwrap_or((0.0, 0.0));
        println!(
            "  tick {:4}: without echoes the rich glyph is {g0:.0} cells, {:.0}% of it invariant; with echoes {g1:.0} cells, {:.0}% invariant; quality {:.0} against {:.0}",
            plain.tick(),
            100.0 * c0,
            100.0 * c1,
            plain.stats().quality_sum,
            echoing.stats().quality_sum
        );
    }
    print!(
        "{}",
        echoing
            .report()
            .lines()
            .filter(|l| l.contains("echoes") || l.contains("echoed") || l.contains("lexicon:"))
            .map(|l| format!("{l}\n"))
            .collect::<String>()
    );

    // Imprinting: new thoughts, a silent floor.
    println!("== at tick 2500 every thought but the echoes forgets what it knew and the floor falls silent");
    for echoes in [0usize, 4] {
        let mut mind = Mind::new(problem.clone(), talking(echoes, 0.7));
        mind.run(2500);
        mind.lexicon_mut().unwrap().silence();
        let renewed = mind.renew(1.0);
        let mut windows = Vec::new();
        for _ in 0..5 {
            let q0 = mind.stats().quality_sum;
            mind.run(200);
            windows.push(format!("{:.0}", mind.stats().quality_sum - q0));
        }
        let lex = mind.lexicon().unwrap();
        let dance: f64 = rich_signs(&mind, rich)
            .iter()
            .map(|&k| lex.sign(k).unwrap().dance)
            .sum();
        println!(
            "  {echoes} echoes: {renewed} thoughts renewed; quality per 200 ticks after: {}; the rich source's dance {dance:.0} again",
            windows.join(", ")
        );
    }
    println!("  ({:.1} s)", start.elapsed().as_secs_f64());
}
