//! A formicarium in a plastic frame: a slab nest, a tube through a hole
//! in the wall of an open outworld box, and food on the floor of the
//! box and on a shelf up its far wall. The ants find their way
//! pheromonally over the folds and through the tube.
//!
//!     cargo run --release --example frame [minutes]

use ant_simulator::prelude::*;

fn main() {
    let minutes: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(40.0);
    // The net: a 96 × 64 grid of 2-cm cells. The box's floor is 56 × 40 cm
    // with 16-cm walls; the nest slab lies west of it, joined by a tube
    // through a hole at the foot of the box's west wall.
    let mut frame = Frame::new(96, 64, 2.0);
    let out = frame.outworld(48, 20, 28, 20, 8, [56.0, 0.0, 0.0], 0.6);
    let slab = frame.slab("slab", 4, 22, 20, 12, [0.0, 4.0, 0.0]);
    frame.nest(Position::new(10, 28), 2);
    frame.wall(Rect::new(Position::new(44, 30), Position::new(47, 33)));
    let tube = frame.tube(
        "tube",
        28,
        46,
        16,
        4,
        Edge {
            start: Position::new(23, 30),
            end: Position::new(23, 33),
            side: Side::East,
        },
        Edge {
            start: Position::new(48, 30),
            end: Position::new(48, 33),
            side: Side::West,
        },
        [40.0, 60.0, 0.0],
        [1.0, 0.0, 0.0],
    );
    let mut world = frame.config();
    // A pool on the floor, and one on a shelf nine centimetres up the
    // east wall.
    let on_floor = Position::new(70, 30);
    let on_wall = Position::new(80, 30);
    world.food_sources = vec![
        FoodSource::pool(on_floor, 1, 40.0, 1.0),
        FoodSource::pool(on_wall, 1, 40.0, 1.0),
    ];
    let mut cfg = SimConfig {
        world,
        ants: 150,
        ..SimConfig::default()
    };
    cfg.nest.initial_satiation = 0.05;
    cfg.history = Some(HistoryConfig::default());
    cfg.memo = Some(MemoConfig {
        transits: Some(TransitConfig::default()),
        ..MemoConfig::default()
    });
    cfg.pipeline = Some(PipelineConfig::default());
    let mut sim = Simulation::new(cfg, 7);
    println!(
        "== the frame: f floor, n/e/s/w walls, t tube, s slab, N nest; | and - where portals open ==\n{}\n",
        frame.render()
    );
    println!(
        "the shelf is {:.0} cm up the east wall; the floor pool is {:.0} cm from the tube's mouth",
        frame.height(Point::center_of(on_wall)).unwrap_or(0.0),
        Point::center_of(on_floor).distance(Point::new(48.5, 31.5)) * 2.0
    );
    let regions: Vec<(String, Rect)> = frame
        .regions()
        .iter()
        .map(|r| (r.name.clone(), r.rect))
        .collect();
    let quarters = 4;
    for q in 1..=quarters {
        sim.run_seconds(minutes * 60.0 / quarters as f64);
        let s = sim.stats();
        let world = sim.world();
        let (mut heights, mut n) = (0.0, 0usize);
        let mut by_region = vec![0usize; regions.len()];
        for a in sim.living().filter(|a| !a.is_inside()) {
            if let Some(h) = frame.height(a.position) {
                heights += h;
                n += 1;
            }
            if let Some(r) = frame.region_of(a.cell()) {
                by_region[r] += 1;
            }
        }
        let taken =
            |at: Position| 360.0 - world.food_in(&Rect::new(at.offset(-1, -1), at.offset(1, 1)));
        println!(
            "after {:.0} min: delivered {}, taken {:.0} µl from the floor pool and {:.0} µl from the shelf, {} outside, mean height {:.1} cm; ants per region {}",
            minutes * q as f64 / quarters as f64,
            s.food_delivered,
            taken(on_floor),
            taken(on_wall),
            n,
            if n > 0 { heights / n as f64 } else { 0.0 },
            regions
                .iter()
                .zip(&by_region)
                .map(|((name, _), c)| format!("{name} {c}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let world = sim.world();
    println!("\ntrail on each surface at the end (units of the perception constant, summed):");
    let k = world.channel(Pheromone::Trail).k.max(1e-9);
    for (name, rect) in &regions {
        println!(
            "   {:<10} {:>8.1}",
            name,
            world.pheromone_in(rect, Pheromone::Trail) / k
        );
    }
    println!(
        "\nfood left: floor pool {:.1} µl, shelf pool {:.1} µl; transits {}",
        world.food_in(&Rect::new(on_floor.offset(-1, -1), on_floor.offset(1, 1))),
        world.food_in(&Rect::new(on_wall.offset(-1, -1), on_wall.offset(1, 1))),
        sim.memo()
            .and_then(|m| m.transits.as_ref())
            .map(|t| t.report())
            .unwrap_or_default()
    );
    println!("{}", render(&sim));
    let _ = (out, slab, tube);
}
