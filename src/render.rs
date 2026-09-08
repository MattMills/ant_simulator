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

/// Render recorded path-surface rows as text, newest row last.
///
/// Each row shows the ant's tick, position and load, then the eight ring
/// positions from 135° left through straight ahead to 135° right and finally
/// reverse, shaded by the probability the direction was drawn with
/// (` .:-=+*#@` from nothing to certainty). The chosen position is bracketed,
/// masked positions show `xx`, and the sucker's trail (if any) follows.
pub fn render_surface(rows: &[crate::colony::SurfaceRow], last: usize) -> String {
    use crate::landscape::DISPLAY_ORDER;
    const SHADES: [char; 9] = [' ', '.', ':', '-', '=', '+', '*', '#', '@'];
    let shade = |p: f64| -> char {
        let idx = ((p * 8.0).round() as usize).min(8);
        SHADES[idx]
    };
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{:>5} {:>9} {:>1}  {:^4}{:^4}{:^4}{:^4}{:^4}{:^4}{:^4}{:^4}  trail",
        "tick", "position", "", "L135", "L90", "L45", "^", "R45", "R90", "R135", "rev"
    );
    let start = rows.len().saturating_sub(last);
    for row in &rows[start..] {
        let _ = write!(
            out,
            "{:>5} ({:>3},{:>3}) {}  ",
            row.tick,
            row.position.x,
            row.position.y,
            if row.carrying { 'F' } else { ' ' }
        );
        for &j in &DISPLAY_ORDER {
            let cell = if !row.valid[j] {
                "xx".to_string()
            } else {
                let c = shade(row.selected[j]);
                format!("{c}{c}")
            };
            if j == row.chosen {
                let _ = write!(out, "[{cell}]");
            } else {
                let _ = write!(out, " {cell} ");
            }
        }
        if !row.walk.is_empty() {
            let trail: Vec<String> = row.walk.iter().map(|p| p.to_string()).collect();
            let _ = write!(out, "  {}", trail.join("→"));
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod surface_tests {
    use super::*;
    use crate::colony::{Selection, SimConfig};

    #[test]
    fn renders_surface_rows() {
        let cfg = SimConfig {
            selection: Selection::Sucker { reach: 3 },
            record_surface: Some(0),
            ..SimConfig::default()
        };
        let mut sim = Simulation::new(cfg, 2);
        sim.run(12);
        let text = render_surface(sim.surface_trace(), 5);
        assert_eq!(text.lines().count(), 6);
        assert!(text.contains('['));
        assert!(text.contains('→'));
        assert!(render_surface(&[], 5).lines().count() == 1);
    }

    #[test]
    fn chosen_bracket_sits_under_its_column() {
        use crate::colony::SurfaceRow;
        use crate::geometry::{Direction, Position};
        use crate::landscape::DISPLAY_ORDER;
        for chosen in 0..8 {
            let mut valid = [true; 8];
            valid[(chosen + 3) % 8] = false;
            let mut selected = [0.0; 8];
            selected[chosen] = 1.0;
            let row = SurfaceRow {
                tick: 1,
                position: Position::new(2, 3),
                heading: Direction::North,
                carrying: false,
                valid,
                base: [0.0; 8],
                deformed: [0.0; 8],
                probs: selected,
                selected,
                chosen,
                walk: vec![],
                temperature: 1.0,
            };
            let text = render_surface(&[row], 1);
            let line = text.lines().nth(1).unwrap();
            let column = DISPLAY_ORDER.iter().position(|&j| j == chosen).unwrap();
            let cells = &line[19..19 + 32];
            let cell = &cells[column * 4..column * 4 + 4];
            assert_eq!(cell, "[@@]", "chosen {chosen}: {line:?}");
            let masked_column = DISPLAY_ORDER
                .iter()
                .position(|&j| j == (chosen + 3) % 8)
                .unwrap();
            assert_eq!(&cells[masked_column * 4..masked_column * 4 + 4], " xx ");
        }
    }
}
