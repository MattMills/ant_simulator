//! Plain-text rendering of a colony and of recorded path surfaces.

use crate::ant::Activity;
use crate::colony::{Simulation, SurfaceRow};
use crate::geometry::Position;
use crate::landscape::{turn_degrees, DISPLAY_ORDER};
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
                    if cell.has_food() {
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
        let cell = ant.cell();
        if cell.x < 0 || cell.y < 0 {
            continue;
        }
        let (x, y) = (cell.x as usize, cell.y as usize);
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
        "t = {:.0} s | {:.1} °C | alive {} | outside {} | delivered {} ({:.2} mg sugar) | store {:.2} mg ({:.0}% full) | food left {:.0} µl | trail {:.0} | mean entropy {:.3}",
        sim.time_s(),
        sim.temperature(),
        sim.alive(),
        sim.outside(),
        s.food_delivered,
        s.sugar_delivered_mg,
        sim.nest().store_mg,
        100.0 * sim.nest().satiation(),
        world.total_food(),
        world.total_pheromone(Pheromone::Trail),
        s.mean_entropy()
    );
    out
}

/// Width of one ring column in the surface rendering.
const COLUMN: usize = 5;

/// Render recorded path-surface rows as text, newest row last.
///
/// Each row shows the ant's tick, position and load, then the ring
/// positions from the sharpest left turn through straight ahead to the
/// sharpest right turn and finally reverse, shaded by the probability the
/// heading was drawn with (` .:-=+*#@` from nothing to certainty). The
/// chosen position is bracketed, blocked positions show `xx`, and the
/// sucker's trail (if any) follows.
pub fn render_surface(rows: &[SurfaceRow], last: usize) -> String {
    const SHADES: [char; 9] = [' ', '.', ':', '-', '=', '+', '*', '#', '@'];
    let shade = |p: f64| -> char {
        let idx = ((p * 8.0).round() as usize).min(8);
        SHADES[idx]
    };
    let mut out = String::new();
    let _ = write!(out, "{:>5} {:>11} {:>1}  ", "tick", "position", "");
    for &j in &DISPLAY_ORDER {
        let deg = turn_degrees(j);
        let label = if deg == 0.0 {
            "^".to_string()
        } else if deg == 180.0 {
            "rev".to_string()
        } else if deg > 0.0 {
            format!("R{}", deg.round() as i32)
        } else {
            format!("L{}", (-deg).round() as i32)
        };
        let _ = write!(out, "{label:^COLUMN$}");
    }
    out.push_str("  trail\n");
    let start = rows.len().saturating_sub(last);
    for row in &rows[start..] {
        let _ = write!(
            out,
            "{:>5} ({:>4.1},{:>4.1}) {}  ",
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
                let _ = write!(out, "{:^COLUMN$}", format!("[{cell}]"));
            } else {
                let _ = write!(out, "{:^COLUMN$}", cell);
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
    use crate::geometry::Point;
    use crate::landscape::RING;

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
        // Ant 0 leaves the nest when its own threshold lets it; wait for it.
        for _ in 0..40 {
            sim.run(100);
            if sim.surface_trace().len() >= 5 {
                break;
            }
        }
        let text = render_surface(sim.surface_trace(), 5);
        assert!(text.lines().count() >= 2, "{text}");
        assert!(text.contains('['));
        assert!(text.contains('→'));
        assert!(render_surface(&[], 5).lines().count() == 1);
    }

    #[test]
    fn chosen_bracket_sits_under_its_column() {
        for chosen in 0..RING {
            let mut valid = [true; RING];
            valid[(chosen + 3) % RING] = false;
            let mut selected = [0.0; RING];
            selected[chosen] = 1.0;
            let row = SurfaceRow {
                tick: 1,
                position: Point::new(2.5, 3.5),
                heading: 0.0,
                carrying: false,
                valid,
                base: [0.0; RING],
                deformed: [0.0; RING],
                probs: selected,
                selected,
                chosen,
                walk: vec![],
                temperature: 1.0,
            };
            let text = render_surface(&[row], 1);
            let header = text.lines().next().unwrap();
            let line = text.lines().nth(1).unwrap();
            let prefix = header.find("L158").unwrap();
            let column = DISPLAY_ORDER.iter().position(|&j| j == chosen).unwrap();
            let cell = &line[prefix + column * COLUMN..prefix + (column + 1) * COLUMN];
            assert_eq!(cell.trim(), "[@@]", "chosen {chosen}: {line:?}");
            let masked_column = DISPLAY_ORDER
                .iter()
                .position(|&j| j == (chosen + 3) % RING)
                .unwrap();
            let masked =
                &line[prefix + masked_column * COLUMN..prefix + (masked_column + 1) * COLUMN];
            assert_eq!(masked.trim(), "xx");
        }
    }
}
