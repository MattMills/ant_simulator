//! Plain-text rendering of a colony.

use crate::colony::Simulation;
use crate::geometry::Position;
use crate::world::Terrain;
use std::fmt::Write as _;

/// Render the world as ASCII art, one character per cell.
///
/// * `#` wall, `N` nest, `F` food, `A` ant carrying food, `a` ant,
/// * `:` and `.` strong / faint food pheromone, `,` home pheromone,
/// * space for nothing.
pub fn render(sim: &Simulation) -> String {
    let world = sim.world();
    let w = world.width();
    let h = world.height();
    let mut grid = vec![vec![' '; w]; h];
    for (y, row) in grid.iter_mut().enumerate() {
        for (x, slot) in row.iter_mut().enumerate() {
            let cell = world
                .cell(Position::new(x as i32, y as i32))
                .expect("in bounds");
            *slot = match cell.terrain {
                Terrain::Wall => '#',
                Terrain::Nest => 'N',
                Terrain::Open => {
                    if cell.food > 0 {
                        'F'
                    } else if cell.food_pheromone > 2.0 {
                        ':'
                    } else if cell.food_pheromone > 0.3 {
                        '.'
                    } else if cell.home_pheromone > 1.0 {
                        ','
                    } else {
                        ' '
                    }
                }
            };
        }
    }
    for ant in sim.living() {
        let (x, y) = (ant.position.x as usize, ant.position.y as usize);
        if y < h && x < w {
            grid[y][x] = if ant.carrying { 'A' } else { 'a' };
        }
    }
    let mut out = String::with_capacity((w + 3) * (h + 2));
    let border: String = "-".repeat(w);
    let _ = writeln!(out, "+{border}+");
    for row in grid {
        out.push('|');
        out.extend(row);
        out.push_str("|\n");
    }
    let _ = writeln!(out, "+{border}+");
    let s = sim.stats();
    let _ = writeln!(
        out,
        "tick {} | alive {} | delivered {} | picked {} | store {} | food left {} | mean entropy {:.3}",
        sim.tick(),
        sim.alive(),
        s.food_delivered,
        s.food_picked,
        sim.food_store(),
        world.total_food(),
        s.mean_entropy()
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colony::SimConfig;

    #[test]
    fn renders_a_frame() {
        let mut sim = Simulation::new(SimConfig::default(), 1);
        sim.run(5);
        let frame = render(&sim);
        assert!(frame.contains('N'));
        assert!(frame.contains('F'));
        assert!(frame.contains('a'));
        assert!(frame.contains("tick 5"));
        assert_eq!(frame.lines().count(), sim.world().height() + 3);
    }
}
