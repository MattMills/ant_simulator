//! The behavioural memo: what the colony does where, composed over the
//! quadtree, extracted as an object of its own, classified into kinds of
//! ground, and split into its invariant and variant parts.
//!
//!     cargo run --release --example memo [minutes] [categories]

use ant_simulator::prelude::*;

fn main() {
    let mut args = std::env::args().skip(1);
    let minutes: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(30.0);
    let categories: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let mut cfg = SimConfig::default();
    cfg.world.seed = Some(2024);
    cfg.ants = 100;
    cfg.nest.initial_satiation = 0.1;
    cfg.nest.mortality = false;
    if let Some(food) = cfg.world.random_food.as_mut() {
        food.renewal_ul_per_s = 0.02;
    }
    cfg.species.consumption_mg_per_ant_per_s = 0.5 / 3600.0;
    cfg.history = Some(HistoryConfig::default());
    cfg.memo = Some(MemoConfig {
        grain: 4,
        ..MemoConfig::default()
    });
    let mut sim = Simulation::new(cfg, 7);
    sim.run_seconds(minutes * 60.0);
    let memo = sim.extract_memo().expect("memo on");
    let level = memo.level();
    let (cols, rows) = memo.extent();
    println!(
        "== the memo after {minutes:.0} min: {}×{} nodes of {} cells, quadtree level {level} ==\n",
        cols,
        rows,
        4usize.pow(0) * memo.rect(memo.keys(level)[0]).2
    );
    println!("{}", render(&sim));

    println!("== {categories} kinds of ground, by the invariant signatures (k-means on standardised features) ==");
    let classes = memo.classify(level, categories, 1);
    println!("{}", classes.table());
    println!("the map (category digit per node; blank where nothing was decided):");
    println!("{}", memo.render(&classes));

    println!("== invariant and variant signatures of the busiest nodes ==");
    let mut busiest: Vec<(QuadKey, Signature)> = memo
        .keys(level)
        .into_iter()
        .map(|k| (k, memo.signature(k, Layer::Invariant)))
        .filter(|(_, s)| s.decisions > 0.0)
        .collect();
    busiest.sort_by(|a, b| {
        b.1.decisions
            .partial_cmp(&a.1.decisions)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    println!(
        "{:>8} {:>5} {:>10} {:>8} {:>9} {:>9} {:>9} {:>8} | {:>10} {:>8}",
        "quadkey",
        "cat",
        "decisions",
        "entropy",
        "straight",
        "outbound",
        "homing",
        "trail/d",
        "variant d",
        "var. H"
    );
    for (key, s) in busiest.iter().take(8) {
        let v = memo.signature(*key, Layer::Variant);
        println!(
            "{:>8} {:>5} {:>10.2} {:>8.2} {:>9.2} {:>8.0}% {:>8.0}% {:>8.2} | {:>+10.2} {:>+8.2}",
            key.to_string(),
            classes
                .label(*key)
                .map(|l| l.to_string())
                .unwrap_or_default(),
            s.decisions,
            s.mean_entropy(),
            s.straightness(),
            100.0 * s.leg_share(Leg::Outbound),
            100.0 * s.leg_share(Leg::Inbound),
            s.deposits[Pheromone::Trail.index()] / s.decisions.max(1e-9),
            v.decisions,
            v.mean_entropy()
        );
    }
    println!("\n(decisions and trail per tick; the variant columns are the current record minus the invariant one)");
    let root = memo.signature(QuadKey::ROOT, Layer::Invariant);
    println!(
        "\nthe whole field composes to {:.1} decisions per tick at mean entropy {:.2}, straightness {:.2}, {:.0}% outbound, {:.0}% homing, {:.0}% searching",
        root.decisions,
        root.mean_entropy(),
        root.straightness(),
        100.0 * root.leg_share(Leg::Outbound),
        100.0 * root.leg_share(Leg::Inbound),
        100.0 * root.leg_share(Leg::Searching)
    );
}
