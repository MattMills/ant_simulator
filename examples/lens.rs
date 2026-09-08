//! The lens: the two-position tessellation of the quadtree and the
//! geodesic over it, drawn on a walled world.
//!
//!     cargo run --release --example lens

use ant_simulator::prelude::*;

fn main() {
    // A wall across the middle of an open field with a gap at its
    // southern end, the nest west of it and a pool east of it.
    let cfg = WorldConfig {
        width: 64,
        height: 40,
        nest: Position::new(8, 20),
        random_food: None,
        food_sources: vec![FoodSource::pool(Position::new(54, 12), 1, 20.0, 1.0)],
        walls: vec![
            Rect::new(Position::new(31, 0), Position::new(32, 30)),
            Rect::new(Position::new(40, 24), Position::new(48, 25)),
        ],
        ..WorldConfig::default()
    };
    let world = World::new(cfg, &mut Rng::seed_from_u64(1));
    let a = Point::center_of(world.nest());
    let b = Point::new(54.5, 12.5);
    let tree = QuadTree::<()>::new(world.width(), world.height());
    let lens = Lens::new(&tree, a, b, 2.0, |key, rect| world.ground(key, rect));
    let route = lens.geodesic().expect("a way round the wall");
    let mut pulled = route.clone();
    pulled.pull(|p, q| world.segment_passable(p, q));
    println!(
        "== the lens between the nest (A) and the pool (B): {} leaves over {} cells, cells within 2 of either end, nodes doubling with the distance beyond; # walled leaves, * the geodesic ==",
        lens.leaves().len(),
        world.width() * world.height()
    );
    println!("{}\n", lens.render(Some(&pulled)));
    println!(
        "geodesic over the leaves: cost {:.1} through {} leaves; pulled straight where the line of sight is clear: {:.1} cells through {} points, against {:.1} as the crow flies",
        route.cost,
        route.leaves,
        pulled.length(),
        pulled.points.len(),
        a.distance(b)
    );
    let back = Lens::new(&tree, b, a, 2.0, |key, rect| world.ground(key, rect))
        .geodesic()
        .expect("a way back");
    println!(
        "from the other end: {} leaves, cost {:.1} (the same tessellation and cost from either end)\n",
        lens.leaves().len(),
        back.cost
    );

    println!("== the lens grows with the logarithm of the span: an open 512 × 512 world ==");
    let big = QuadTree::<()>::new(512, 512);
    println!("{:>6} {:>7} {:>10}", "span", "leaves", "cells");
    for span in [8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 500.0] {
        let a = Point::new(5.5, 256.5);
        let b = Point::new(5.5 + span, 256.5);
        let lens = Lens::new(&big, a, b, 2.0, |_, _| Ground::Open(1.0));
        println!("{:>6.0} {:>7} {:>10}", span, lens.leaves().len(), 512 * 512);
    }
}
