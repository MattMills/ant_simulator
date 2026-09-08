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
