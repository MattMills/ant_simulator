//! The frame: surfaces joined along their edges, as an ant colony lives
//! in a plastic frame.
//!
//! An ant walks on surfaces, and its degrees of freedom are a position
//! and a heading on the surface it is on, with the ability to walk over
//! a fold onto the next surface (a floor onto a wall, a wall onto the
//! next wall round a corner) and along a tube. A formicarium is a set of
//! such surfaces: the floor and walls of an outworld box, the slab of a
//! nest, the tubes between them. The frame lays those surfaces out flat
//! on one grid, as the net of a box unfolds, so that the whole model
//! (the chemical field, the memo, the transits, the pipeline) runs on
//! them unchanged; where two surfaces meet in space but not in the net,
//! a [`Portal`] joins their edges and turns the body crossing it, and a
//! [`Slope`] makes walking up a wall slower than along it. Every region
//! carries its place in space, so a point of the net has a position and
//! a height in three dimensions.

use crate::geometry::{Point, Position};
use crate::world::{Edge, Portal, Rect, Side, Slope, WorldConfig};

/// A rectangle of the net standing somewhere in space.
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    /// What the region is.
    pub name: String,
    /// Its cells on the net.
    pub rect: Rect,
    /// Where its top-left corner is in space, centimetres (`x` east,
    /// `y` south, `z` up).
    pub origin: [f64; 3],
    /// The direction in space of the net's `+x` across it.
    pub u: [f64; 3],
    /// The direction in space of the net's `+y` across it.
    pub v: [f64; 3],
}

impl Region {
    /// The position in space of a point of the net within the region.
    pub fn position_3d(&self, p: Point, cell_cm: f64) -> [f64; 3] {
        let dx = (p.x - self.rect.min.x as f64) * cell_cm;
        let dy = (p.y - self.rect.min.y as f64) * cell_cm;
        [
            self.origin[0] + dx * self.u[0] + dy * self.v[0],
            self.origin[1] + dx * self.u[1] + dy * self.v[1],
            self.origin[2] + dx * self.u[2] + dy * self.v[2],
        ]
    }
}

/// The regions of an outworld box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Outworld {
    /// The floor.
    pub floor: usize,
    /// The wall along the floor's north edge (above it on the net).
    pub north: usize,
    /// The wall along the floor's east edge.
    pub east: usize,
    /// The wall along the floor's south edge.
    pub south: usize,
    /// The wall along the floor's west edge.
    pub west: usize,
}

/// Surfaces laid out on one grid, joined by portals, standing in space.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    width: usize,
    height: usize,
    cell_cm: f64,
    regions: Vec<Region>,
    portals: Vec<Portal>,
    slopes: Vec<Slope>,
    walls: Vec<Rect>,
    nest: Option<(Position, i32)>,
}

impl Frame {
    /// An empty frame over a `width` × `height` net of `cell_cm` cells.
    pub fn new(width: usize, height: usize, cell_cm: f64) -> Frame {
        Frame {
            width,
            height,
            cell_cm,
            regions: Vec::new(),
            portals: Vec::new(),
            slopes: Vec::new(),
            walls: Vec::new(),
            nest: None,
        }
    }

    /// Add a region: a rectangle of the net standing at `origin` in
    /// space with the net's axes along `u` and `v`. Returns its index.
    pub fn region(
        &mut self,
        name: &str,
        rect: Rect,
        origin: [f64; 3],
        u: [f64; 3],
        v: [f64; 3],
    ) -> usize {
        self.regions.push(Region {
            name: name.to_string(),
            rect,
            origin,
            u,
            v,
        });
        self.regions.len() - 1
    }

    /// Join two edges.
    pub fn portal(&mut self, a: Edge, b: Edge) {
        self.portals.push(Portal { a, b });
    }

    /// Make a region a slope: walking towards `up` on it is climbing.
    pub fn slope(&mut self, region: usize, up: Side, factor: f64) {
        self.slopes.push(Slope {
            rect: self.regions[region].rect,
            up,
            factor,
        });
    }

    /// Wall off cells of a region (a hole in a wall is made by walling
    /// the wall's cells and opening a portal at the floor's edge).
    pub fn wall(&mut self, rect: Rect) {
        self.walls.push(rect);
    }

    /// Put the nest at a cell, with a Chebyshev radius.
    pub fn nest(&mut self, at: Position, radius: i32) {
        self.nest = Some((at, radius));
    }

    /// An open box: a floor `w` × `d` cells at `(x0, y0)` of the net,
    /// with walls `h` cells high folded out on its four sides and joined
    /// at the corners, its floor's north-west corner at `origin` in
    /// space; walking up a wall goes at `climb` of the speed along it.
    #[allow(clippy::too_many_arguments)]
    pub fn outworld(
        &mut self,
        x0: i32,
        y0: i32,
        w: i32,
        d: i32,
        h: i32,
        origin: [f64; 3],
        climb: f64,
    ) -> Outworld {
        let c = self.cell_cm;
        let (ox, oy, oz) = (origin[0], origin[1], origin[2]);
        let floor = self.region(
            "floor",
            Rect::new(Position::new(x0, y0), Position::new(x0 + w - 1, y0 + d - 1)),
            origin,
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        );
        // The north wall stands above the floor on the net; up the net is
        // up in space.
        let north = self.region(
            "north wall",
            Rect::new(Position::new(x0, y0 - h), Position::new(x0 + w - 1, y0 - 1)),
            [ox, oy, oz + h as f64 * c],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        );
        let east = self.region(
            "east wall",
            Rect::new(
                Position::new(x0 + w, y0),
                Position::new(x0 + w + h - 1, y0 + d - 1),
            ),
            [ox + w as f64 * c, oy, oz],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        );
        let south = self.region(
            "south wall",
            Rect::new(
                Position::new(x0, y0 + d),
                Position::new(x0 + w - 1, y0 + d + h - 1),
            ),
            [ox, oy + d as f64 * c, oz],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        );
        let west = self.region(
            "west wall",
            Rect::new(Position::new(x0 - h, y0), Position::new(x0 - 1, y0 + d - 1)),
            [ox, oy, oz + h as f64 * c],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        );
        // The corners, each edge listed from the floor end.
        let edge = |start: (i32, i32), end: (i32, i32), side: Side| Edge {
            start: Position::new(start.0, start.1),
            end: Position::new(end.0, end.1),
            side,
        };
        self.portal(
            edge((x0 + w - 1, y0 - 1), (x0 + w - 1, y0 - h), Side::East),
            edge((x0 + w, y0), (x0 + w + h - 1, y0), Side::North),
        );
        self.portal(
            edge(
                (x0 + w, y0 + d - 1),
                (x0 + w + h - 1, y0 + d - 1),
                Side::South,
            ),
            edge(
                (x0 + w - 1, y0 + d),
                (x0 + w - 1, y0 + d + h - 1),
                Side::East,
            ),
        );
        self.portal(
            edge((x0, y0 + d), (x0, y0 + d + h - 1), Side::West),
            edge((x0 - 1, y0 + d - 1), (x0 - h, y0 + d - 1), Side::South),
        );
        self.portal(
            edge((x0 - 1, y0), (x0 - h, y0), Side::North),
            edge((x0, y0 - 1), (x0, y0 - h), Side::West),
        );
        self.slope(north, Side::North, climb);
        self.slope(east, Side::East, climb);
        self.slope(south, Side::South, climb);
        self.slope(west, Side::West, climb);
        Outworld {
            floor,
            north,
            east,
            south,
            west,
        }
    }

    /// A flat slab `w` × `h` cells at `(x0, y0)` of the net, lying at
    /// `origin` in space with the net's axes east and south: the nest
    /// module of a formicarium, or a table.
    pub fn slab(
        &mut self,
        name: &str,
        x0: i32,
        y0: i32,
        w: i32,
        h: i32,
        origin: [f64; 3],
    ) -> usize {
        self.region(
            name,
            Rect::new(Position::new(x0, y0), Position::new(x0 + w - 1, y0 + h - 1)),
            origin,
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        )
    }

    /// A tube: a strip `length` cells long and `girth` cells round at
    /// `(x0, y0)` of the net, running east, its two long sides joined
    /// round the back, its west end joined to `from` and its east end to
    /// `to` (edges of `girth` cells each, listed in the same sense as the
    /// tube's rows, top to bottom). In space it runs straight between
    /// the two ends.
    #[allow(clippy::too_many_arguments)]
    pub fn tube(
        &mut self,
        name: &str,
        x0: i32,
        y0: i32,
        length: i32,
        girth: i32,
        from: Edge,
        to: Edge,
        origin: [f64; 3],
        axis: [f64; 3],
    ) -> usize {
        assert_eq!(from.len(), girth as usize, "a tube's end is its girth wide");
        assert_eq!(to.len(), girth as usize, "a tube's end is its girth wide");
        let across = [-axis[1], axis[0], 0.0];
        let region = self.region(
            name,
            Rect::new(
                Position::new(x0, y0),
                Position::new(x0 + length - 1, y0 + girth - 1),
            ),
            origin,
            axis,
            across,
        );
        // Round the back.
        self.portal(
            Edge {
                start: Position::new(x0, y0),
                end: Position::new(x0 + length - 1, y0),
                side: Side::North,
            },
            Edge {
                start: Position::new(x0, y0 + girth - 1),
                end: Position::new(x0 + length - 1, y0 + girth - 1),
                side: Side::South,
            },
        );
        // The ends.
        self.portal(
            from,
            Edge {
                start: Position::new(x0, y0),
                end: Position::new(x0, y0 + girth - 1),
                side: Side::West,
            },
        );
        self.portal(
            Edge {
                start: Position::new(x0 + length - 1, y0),
                end: Position::new(x0 + length - 1, y0 + girth - 1),
                side: Side::East,
            },
            to,
        );
        region
    }

    /// The regions.
    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    /// The portals.
    pub fn portals(&self) -> &[Portal] {
        &self.portals
    }

    /// The region holding a cell.
    pub fn region_of(&self, cell: Position) -> Option<usize> {
        self.regions.iter().position(|r| r.rect.contains(cell))
    }

    /// The position in space of a point of the net, centimetres.
    pub fn position_3d(&self, p: Point) -> Option<[f64; 3]> {
        let r = self.region_of(p.cell())?;
        Some(self.regions[r].position_3d(p, self.cell_cm))
    }

    /// The height in space of a point of the net, centimetres.
    pub fn height(&self, p: Point) -> Option<f64> {
        self.position_3d(p).map(|q| q[2])
    }

    /// The world configuration of the frame: the regions open, all else
    /// wall, with the portals, slopes and nest.
    pub fn config(&self) -> WorldConfig {
        let (nest, nest_radius) = self.nest.unwrap_or((
            Position::new(self.width as i32 / 2, self.height as i32 / 2),
            1,
        ));
        WorldConfig {
            width: self.width,
            height: self.height,
            cell_cm: self.cell_cm,
            nest,
            nest_radius,
            open: self.regions.iter().map(|r| r.rect).collect(),
            walls: self.walls.clone(),
            portals: self.portals.clone(),
            slopes: self.slopes.clone(),
            random_food: None,
            ..WorldConfig::default()
        }
    }

    /// The net drawn cell by cell: each region by the first letter of its
    /// name, `#` elsewhere, with `|` and `-` where a portal's edge opens.
    pub fn render(&self) -> String {
        let mut grid = vec![vec!['#'; self.width]; self.height];
        for r in &self.regions {
            let ch = r.name.chars().next().unwrap_or('?');
            for y in r.rect.min.y.max(0)..=r.rect.max.y.min(self.height as i32 - 1) {
                for x in r.rect.min.x.max(0)..=r.rect.max.x.min(self.width as i32 - 1) {
                    grid[y as usize][x as usize] = ch;
                }
            }
        }
        for p in &self.portals {
            for e in [&p.a, &p.b] {
                let mark = match e.side {
                    Side::North | Side::South => '-',
                    Side::East | Side::West => '|',
                };
                for c in e.cells() {
                    if c.x >= 0
                        && c.y >= 0
                        && (c.x as usize) < self.width
                        && (c.y as usize) < self.height
                    {
                        grid[c.y as usize][c.x as usize] = mark;
                    }
                }
            }
        }
        if let Some((n, r)) = self.nest {
            for dy in -r..=r {
                for dx in -r..=r {
                    let (x, y) = (n.x + dx, n.y + dy);
                    if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
                        grid[y as usize][x as usize] = 'N';
                    }
                }
            }
        }
        grid.into_iter()
            .map(|row| row.into_iter().collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::world::{Terrain, World};

    #[test]
    fn a_box_folds_out_with_its_corners_joined_and_its_walls_climbing() {
        let mut frame = Frame::new(60, 60, 2.0);
        let out = frame.outworld(20, 20, 20, 16, 6, [0.0, 0.0, 0.0], 0.6);
        frame.nest(Position::new(30, 28), 1);
        let cfg = frame.config();
        let world = World::new(cfg.clone(), &mut Rng::seed_from_u64(1));
        assert_eq!(frame.regions().len(), 5);
        assert_eq!(frame.portals().len(), 4);
        // Every portal joins edges of one length, and what lies beyond
        // either edge is wall or off the grid.
        for p in frame.portals() {
            assert_eq!(p.a.len(), p.b.len());
            for e in [&p.a, &p.b] {
                let (dx, dy) = e.side.offset();
                for c in e.cells() {
                    assert!(world
                        .cell(c)
                        .map(|c| c.terrain != Terrain::Wall)
                        .unwrap_or(false));
                    let beyond = Position::new(c.x + dx, c.y + dy);
                    assert!(!world.is_passable(beyond), "beyond {c:?} is open");
                }
            }
        }
        // The floor is flat, the walls stand: the top of the north wall is
        // six cells (12 cm) up, and the floor's far corner is where it
        // should be.
        let floor = &frame.regions()[out.floor];
        assert!((frame.height(Point::new(30.5, 28.5)).unwrap()).abs() < 1e-9);
        let top = Point::new(30.5, 14.5);
        assert!(
            (frame.height(top).unwrap() - 11.0).abs() < 1e-9,
            "{:?}",
            frame.position_3d(top)
        );
        let far = frame.position_3d(Point::new(39.5, 35.5)).unwrap();
        assert!((far[0] - 39.0).abs() < 1e-9 && (far[1] - 31.0).abs() < 1e-9);
        assert_eq!(
            floor.rect,
            Rect::new(Position::new(20, 20), Position::new(39, 35))
        );
        // Round the north-east corner: walking east along the north wall
        // comes out on the east wall walking south, at the same height.
        let w = world.warp(Point::new(40.5, 17.5)).expect("the corner");
        assert!((w.turn - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
        let h_before = frame.height(Point::new(39.5, 17.5)).unwrap();
        let h_after = frame.height(w.point).unwrap();
        assert!((h_before - h_after).abs() < 1e-9, "{h_before} vs {h_after}");
        // Climbing the east wall (east on the net) is slower than walking
        // along it.
        assert!(world.climb_factor(Position::new(42, 28), 0.0) < 1.0);
        assert!(
            (world.climb_factor(Position::new(42, 28), std::f64::consts::FRAC_PI_2) - 1.0).abs()
                < 1e-9
        );
        let picture = frame.render();
        assert!(picture.contains('f') && picture.contains('N') && picture.contains('|'));
    }

    #[test]
    fn a_tube_joins_a_slab_to_a_box() {
        let mut frame = Frame::new(80, 60, 2.0);
        let out = frame.outworld(40, 20, 20, 16, 6, [40.0, 0.0, 0.0], 0.6);
        let slab = frame.slab("slab", 4, 24, 16, 8, [0.0, 8.0, 0.0]);
        frame.nest(Position::new(8, 28), 1);
        // A hole in the west wall at floor level, four cells wide, and a
        // tube from the slab's east edge to the floor's west edge there.
        frame.wall(Rect::new(Position::new(39, 26), Position::new(39, 29)));
        let from = Edge {
            start: Position::new(19, 26),
            end: Position::new(19, 29),
            side: Side::East,
        };
        let to = Edge {
            start: Position::new(40, 26),
            end: Position::new(40, 29),
            side: Side::West,
        };
        let tube = frame.tube(
            "tube",
            24,
            40,
            12,
            4,
            from,
            to,
            [32.0, 16.0, 0.0],
            [1.0, 0.0, 0.0],
        );
        assert_eq!(frame.regions().len(), 7);
        assert_eq!(frame.portals().len(), 7);
        let world = World::new(frame.config(), &mut Rng::seed_from_u64(1));
        // Off the slab's east edge into the tube, and out of the tube onto
        // the floor.
        let into = world.warp(Point::new(20.5, 27.5)).expect("into the tube");
        assert!(frame.regions()[tube].rect.contains(into.point.cell()));
        let out_of = world.warp(Point::new(36.5, 41.5)).expect("out of the tube");
        assert!(frame.regions()[out.floor]
            .rect
            .contains(out_of.point.cell()));
        // Round the tube's back: off its top side comes in at its bottom.
        let round = world.warp(Point::new(30.5, 39.5)).expect("round the back");
        assert!((round.point.y - 43.5).abs() < 1e-9 && round.turn.abs() < 1e-9);
        let _ = slab;
    }
}
