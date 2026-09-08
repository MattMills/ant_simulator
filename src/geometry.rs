//! Grid positions and the eight movement directions.

/// An integer grid position. `y` grows downward (row index).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Position {
    /// Column.
    pub x: i32,
    /// Row.
    pub y: i32,
}

impl Position {
    /// Construct a position.
    pub const fn new(x: i32, y: i32) -> Self {
        Position { x, y }
    }

    /// Position shifted by `(dx, dy)`.
    pub const fn offset(self, dx: i32, dy: i32) -> Self {
        Position {
            x: self.x + dx,
            y: self.y + dy,
        }
    }

    /// Position one cell away in `dir`.
    pub fn step(self, dir: Direction) -> Self {
        let (dx, dy) = dir.delta();
        self.offset(dx, dy)
    }

    /// Chebyshev (king-move) distance.
    pub fn chebyshev(self, other: Position) -> i32 {
        (self.x - other.x).abs().max((self.y - other.y).abs())
    }

    /// Euclidean distance.
    pub fn euclid(self, other: Position) -> f64 {
        let dx = (self.x - other.x) as f64;
        let dy = (self.y - other.y) as f64;
        (dx * dx + dy * dy).sqrt()
    }
}

/// One of the eight compass directions an ant can move in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Direction {
    /// Up (negative y).
    North = 0,
    /// Up-right.
    NorthEast = 1,
    /// Right (positive x).
    East = 2,
    /// Down-right.
    SouthEast = 3,
    /// Down (positive y).
    South = 4,
    /// Down-left.
    SouthWest = 5,
    /// Left (negative x).
    West = 6,
    /// Up-left.
    NorthWest = 7,
}

const COSINES: [f64; 8] = [
    1.0,
    std::f64::consts::FRAC_1_SQRT_2,
    0.0,
    -std::f64::consts::FRAC_1_SQRT_2,
    -1.0,
    -std::f64::consts::FRAC_1_SQRT_2,
    0.0,
    std::f64::consts::FRAC_1_SQRT_2,
];

impl Direction {
    /// All directions in index order (clockwise from north).
    pub const ALL: [Direction; 8] = [
        Direction::North,
        Direction::NorthEast,
        Direction::East,
        Direction::SouthEast,
        Direction::South,
        Direction::SouthWest,
        Direction::West,
        Direction::NorthWest,
    ];

    /// Number of directions.
    pub const COUNT: usize = 8;

    /// Index in `0..8`.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Direction from an index (taken modulo 8).
    pub fn from_index(i: usize) -> Direction {
        Direction::ALL[i % 8]
    }

    /// Grid delta `(dx, dy)`.
    pub const fn delta(self) -> (i32, i32) {
        match self {
            Direction::North => (0, -1),
            Direction::NorthEast => (1, -1),
            Direction::East => (1, 0),
            Direction::SouthEast => (1, 1),
            Direction::South => (0, 1),
            Direction::SouthWest => (-1, 1),
            Direction::West => (-1, 0),
            Direction::NorthWest => (-1, -1),
        }
    }

    /// Unit vector of the direction.
    pub fn unit(self) -> (f64, f64) {
        let (dx, dy) = self.delta();
        let len = ((dx * dx + dy * dy) as f64).sqrt();
        (dx as f64 / len, dy as f64 / len)
    }

    /// The direction pointing the other way.
    pub fn opposite(self) -> Direction {
        Direction::from_index(self.index() + 4)
    }

    /// Direction rotated clockwise by `steps` eighth-turns (may be negative).
    pub fn rotated(self, steps: i32) -> Direction {
        Direction::from_index((self.index() as i32 + steps).rem_euclid(8) as usize)
    }

    /// Cosine of the angle between two directions.
    pub fn cosine(self, other: Direction) -> f64 {
        COSINES[(self.index() + 8 - other.index()) % 8]
    }

    /// Cosine of the angle between this direction and the vector `(dx, dy)`.
    /// Returns `0` for the zero vector.
    pub fn cosine_to(self, dx: f64, dy: f64) -> f64 {
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-12 {
            return 0.0;
        }
        let (ux, uy) = self.unit();
        (ux * dx + uy * dy) / len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opposite_and_cosines() {
        assert_eq!(Direction::North.opposite(), Direction::South);
        assert_eq!(Direction::NorthEast.opposite(), Direction::SouthWest);
        assert!((Direction::North.cosine(Direction::North) - 1.0).abs() < 1e-12);
        assert!((Direction::North.cosine(Direction::South) + 1.0).abs() < 1e-12);
        assert!(Direction::East.cosine(Direction::North).abs() < 1e-12);
        assert!((Direction::East.cosine(Direction::NorthEast) - COSINES[1]).abs() < 1e-12);
    }

    #[test]
    fn cosine_to_vector() {
        assert!((Direction::East.cosine_to(5.0, 0.0) - 1.0).abs() < 1e-12);
        assert!((Direction::West.cosine_to(5.0, 0.0) + 1.0).abs() < 1e-12);
        assert_eq!(Direction::North.cosine_to(0.0, 0.0), 0.0);
    }

    #[test]
    fn stepping_and_distance() {
        let p = Position::new(3, 3);
        assert_eq!(p.step(Direction::North), Position::new(3, 2));
        assert_eq!(p.step(Direction::SouthWest), Position::new(2, 4));
        assert_eq!(p.chebyshev(Position::new(0, 1)), 3);
        assert!((p.euclid(Position::new(0, 3)) - 3.0).abs() < 1e-12);
        assert_eq!(Direction::North.rotated(-1), Direction::NorthWest);
    }
}

/// A continuous position in cell units: cell `(i, j)` spans
/// `[i, i + 1) × [j, j + 1)` and has its centre at `(i + 0.5, j + 0.5)`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    /// Horizontal coordinate.
    pub x: f64,
    /// Vertical coordinate (grows downward).
    pub y: f64,
}

impl Point {
    /// Construct a point.
    pub const fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    /// The centre of a cell.
    pub fn center_of(cell: Position) -> Self {
        Point {
            x: cell.x as f64 + 0.5,
            y: cell.y as f64 + 0.5,
        }
    }

    /// The cell containing the point.
    pub fn cell(self) -> Position {
        Position::new(self.x.floor() as i32, self.y.floor() as i32)
    }

    /// Point displaced by `distance` along `heading` (radians, 0 = east,
    /// increasing clockwise on screen).
    pub fn advanced(self, heading: f64, distance: f64) -> Self {
        let (s, c) = heading.sin_cos();
        Point {
            x: self.x + distance * c,
            y: self.y + distance * s,
        }
    }

    /// Vector from this point to another.
    pub fn to(self, other: Point) -> (f64, f64) {
        (other.x - self.x, other.y - self.y)
    }

    /// Euclidean distance to another point.
    pub fn distance(self, other: Point) -> f64 {
        let (dx, dy) = self.to(other);
        (dx * dx + dy * dy).sqrt()
    }
}

/// Wrap an angle into `(-π, π]`.
pub fn wrap_angle(theta: f64) -> f64 {
    let mut t = theta % std::f64::consts::TAU;
    if t > std::f64::consts::PI {
        t -= std::f64::consts::TAU;
    } else if t <= -std::f64::consts::PI {
        t += std::f64::consts::TAU;
    }
    t
}

/// Angle of a vector (radians, 0 = east, clockwise positive on screen).
pub fn angle_of(dx: f64, dy: f64) -> f64 {
    dy.atan2(dx)
}

/// Cosine of the angle between a heading and a vector (0 for the zero
/// vector).
pub fn cosine_heading_to(heading: f64, dx: f64, dy: f64) -> f64 {
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-12 {
        return 0.0;
    }
    let (s, c) = heading.sin_cos();
    (c * dx + s * dy) / len
}

impl Direction {
    /// The heading angle of a grid direction.
    pub fn angle(self) -> f64 {
        let (dx, dy) = self.delta();
        angle_of(dx as f64, dy as f64)
    }

    /// The grid direction nearest to a heading.
    pub fn nearest(heading: f64) -> Direction {
        let eighth = std::f64::consts::FRAC_PI_4;
        // East is index 2 in the enum; angles grow clockwise like indices.
        let steps = (wrap_angle(heading) / eighth).round() as i32;
        Direction::from_index((2 + steps).rem_euclid(8) as usize)
    }
}

#[cfg(test)]
mod point_tests {
    use super::*;

    #[test]
    fn cells_centres_and_advance() {
        let c = Point::center_of(Position::new(3, 4));
        assert_eq!(c, Point::new(3.5, 4.5));
        assert_eq!(c.cell(), Position::new(3, 4));
        assert_eq!(Point::new(3.999, 4.0).cell(), Position::new(3, 4));
        let east = c.advanced(0.0, 1.0);
        assert!((east.x - 4.5).abs() < 1e-12 && (east.y - 4.5).abs() < 1e-12);
        let south = c.advanced(std::f64::consts::FRAC_PI_2, 2.0);
        assert!((south.y - 6.5).abs() < 1e-12);
        assert!((c.distance(south) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn angles_round_trip_with_directions() {
        for d in Direction::ALL {
            assert_eq!(Direction::nearest(d.angle()), d);
        }
        assert!((Direction::East.angle()).abs() < 1e-12);
        assert!((Direction::South.angle() - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!((Direction::North.angle() + std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!((wrap_angle(3.0 * std::f64::consts::PI) - std::f64::consts::PI).abs() < 1e-12);
        assert!(
            (wrap_angle(-3.5 * std::f64::consts::PI) - 0.5 * std::f64::consts::PI).abs() < 1e-12
        );
        assert!((cosine_heading_to(0.0, 5.0, 0.0) - 1.0).abs() < 1e-12);
        assert!((cosine_heading_to(0.0, 0.0, 5.0)).abs() < 1e-12);
        assert_eq!(cosine_heading_to(1.0, 0.0, 0.0), 0.0);
    }
}
