//! Plain-text rendering of a colony and of recorded path surfaces.

use crate::ant::Activity;
use crate::colony::{Simulation, SurfaceRow};
use crate::geometry::Position;
use crate::landscape::DISPLAY_ORDER;
use crate::pheromone::Pheromone;
use crate::world::Terrain;
use std::fmt::Write as _;

/// Render the world as ASCII art, one character per cell.
///
/// * `#` wall, `N` nest, `F` food, `x` an alarm cloud,
/// * `o` an outbound ant, `<` an inbound ant carrying food, `-` an inbound
///   ant returning empty, `?` a searching ant, `f` an ant feeding,
/// * `@`, `:` and `.` strong, medium and faint recruitment trail,
///   `,` home-range marking, `!` no-entry marking,
/// * space for nothing.
///
/// Ants inside the nest are not drawn.
pub fn render(sim: &Simulation) -> String {
    let world = sim.world();
    let w = world.width();
    let h = world.height();
    let k = world.channel(Pheromone::Trail).k.max(1e-9);
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
                    let trail = cell.level(Pheromone::Trail);
                    if cell.food > 0 {
                        'F'
                    } else if cell.level(Pheromone::Alarm) > 1.0 {
                        'x'
                    } else if trail > 4.0 * k {
                        '@'
                    } else if trail > k {
                        ':'
                    } else if trail > 0.1 * k {
                        '.'
                    } else if cell.level(Pheromone::NoEntry) > 0.5 {
                        '!'
                    } else if cell.level(Pheromone::Territory) > 0.5 {
                        ','
                    } else {
                        ' '
                    }
                }
            };
        }
    }
    for ant in sim.living() {
        if ant.is_inside() {
            continue;
        }
        let (x, y) = (ant.position.x as usize, ant.position.y as usize);
        if y < h && x < w {
            grid[y][x] = match ant.activity {
                Activity::Outbound => 'o',
                Activity::Inbound if ant.carrying() => '<',
                Activity::Inbound => '-',
                Activity::Searching => '?',
                Activity::Feeding => 'f',
                _ => 'o',
            };
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
        "t = {:.0} s | alive {} | outside {} | delivered {} | store {:.1} ({:.0}% full) | food left {} | trail {:.0} | mean entropy {:.3}",
        sim.time_s(),
        sim.alive(),
        sim.outside(),
        s.food_delivered,
        sim.nest().store,
        100.0 * sim.nest().satiation(),
        world.total_food(),
        world.total_pheromone(Pheromone::Trail),
        s.mean_entropy()
    );
    out
}

/// Render recorded path-surface rows as text, newest row last.
///
/// Each row shows the ant's tick, position and load, then the eight ring
/// positions from 135° left through straight ahead to 135° right and finally
/// reverse, shaded by the probability the direction was drawn with
/// (` .:-=+*#@` from nothing to certainty). The chosen position is bracketed,
/// masked positions show `xx`, and the sucker's trail (if any) follows.
pub fn render_surface(rows: &[SurfaceRow], last: usize) -> String {
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
mod tests {
    use super::*;
    use crate::colony::{Selection, SimConfig};
    use crate::geometry::Direction;

    #[test]
    fn renders_a_frame() {
        let mut cfg = SimConfig::default();
        cfg.nest.initial_satiation = 0.0;
        let mut sim = Simulation::new(cfg, 1);
        sim.run(300);
        let frame = render(&sim);
        assert!(frame.contains('N'));
        assert!(frame.contains('F'));
        assert!(frame.contains('o') || frame.contains('<') || frame.contains('-'));
        assert!(frame.contains("t = 300 s"));
        assert_eq!(frame.lines().count(), sim.world().height() + 3);
    }

    #[test]
    fn renders_surface_rows() {
        let mut cfg = SimConfig {
            selection: Selection::Sucker { reach: 3 },
            record_surface: Some(0),
            ..SimConfig::default()
        };
        cfg.nest.initial_satiation = 0.0;
        let mut sim = Simulation::new(cfg, 2);
        sim.run(400);
        let text = render_surface(sim.surface_trace(), 5);
        assert!(text.lines().count() >= 2, "{text}");
        assert!(text.contains('['));
        assert!(text.contains('→'));
        assert!(render_surface(&[], 5).lines().count() == 1);
    }

    #[test]
    fn chosen_bracket_sits_under_its_column() {
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
