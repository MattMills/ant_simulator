//! A tour: visit every city in the plane once and come home. A state is
//! the city a thought stands in, the cities it has seen and the length
//! walked; the moves are the cities not yet seen; the tour closes with
//! the move home, and its quality is the nearest-neighbour tour's length
//! over its own. The cities lie where they lie, so the trail laid on the
//! way home runs along the tour's own edges through the medium, and a
//! thought choosing its next city reads the trail in that direction as
//! an ant reads a trail ahead.

use crate::problem::{embedding, Moves, Problem};
use ant_simulator::geometry::{Point, Position};
use ant_simulator::rng::Rng;
use ant_simulator::world::WorldConfig;

/// Where a thought stands in a tour.
#[derive(Clone, Debug, PartialEq)]
pub struct TourState {
    /// The city stood in.
    pub at: u8,
    /// The cities seen, one bit each.
    pub seen: u64,
    /// The length walked so far, in cells.
    pub length: f32,
}

/// Cities in the plane, scaled onto a grid.
#[derive(Clone, Debug)]
pub struct Tour {
    cities: Vec<Point>,
    width: usize,
    height: usize,
    mean_edge: f64,
    reference: f64,
}

impl Tour {
    /// Cities at the given coordinates, scaled to fit a grid of the given
    /// size (at most 64 cities).
    pub fn new(points: &[(f64, f64)], width: usize, height: usize) -> Tour {
        assert!(!points.is_empty() && points.len() <= 64, "1 to 64 cities");
        let margin = 2.0;
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for &(x, y) in points {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        let span_x = (x1 - x0).max(1e-9);
        let span_y = (y1 - y0).max(1e-9);
        let scale = ((width as f64 - 2.0 * margin - 1.0) / span_x)
            .min((height as f64 - 2.0 * margin - 1.0) / span_y);
        let cities: Vec<Point> = points
            .iter()
            .map(|&(x, y)| {
                let gx = margin + (x - x0) * scale;
                let gy = margin + (y - y0) * scale;
                Point::new(gx.floor() + 0.5, gy.floor() + 0.5)
            })
            .collect();
        let mut tour = Tour {
            cities,
            width,
            height,
            mean_edge: 1.0,
            reference: 1.0,
        };
        let n = tour.cities.len();
        let mut total = 0.0;
        let mut pairs = 0usize;
        for a in 0..n {
            for b in a + 1..n {
                total += tour.distance(a, b);
                pairs += 1;
            }
        }
        tour.mean_edge = if pairs > 0 { total / pairs as f64 } else { 1.0 }.max(1e-9);
        tour.reference = tour.length_of(&tour.nearest_neighbour()).max(1e-9);
        tour
    }

    /// `n` cities scattered at random over a grid of the given size.
    pub fn random(n: usize, width: usize, height: usize, seed: u64) -> Tour {
        let mut rng = Rng::seed_from_u64(seed);
        let points: Vec<(f64, f64)> = (0..n)
            .map(|_| {
                (
                    rng.range(0.0, width as f64 - 5.0),
                    rng.range(0.0, height as f64 - 5.0),
                )
            })
            .collect();
        Tour::new(&points, width, height)
    }

    /// The cities' places on the grid.
    pub fn cities(&self) -> &[Point] {
        &self.cities
    }

    /// Number of cities.
    pub fn len(&self) -> usize {
        self.cities.len()
    }

    /// Whether there are no cities.
    pub fn is_empty(&self) -> bool {
        self.cities.is_empty()
    }

    /// Distance between two cities, in cells.
    pub fn distance(&self, a: usize, b: usize) -> f64 {
        self.cities[a].distance(self.cities[b])
    }

    /// The length of a closed tour visiting the cities in this order.
    pub fn length_of(&self, order: &[usize]) -> f64 {
        if order.len() < 2 {
            return 0.0;
        }
        let mut total = 0.0;
        for w in order.windows(2) {
            total += self.distance(w[0], w[1]);
        }
        total + self.distance(order[order.len() - 1], order[0])
    }

    /// The nearest-neighbour tour from city 0: the reference a quality
    /// of 1 means.
    pub fn nearest_neighbour(&self) -> Vec<usize> {
        let n = self.cities.len();
        let mut order = vec![0usize];
        let mut seen = vec![false; n];
        seen[0] = true;
        while order.len() < n {
            let last = *order.last().expect("non-empty");
            let next = (0..n)
                .filter(|&c| !seen[c])
                .min_by(|&a, &b| {
                    self.distance(last, a)
                        .partial_cmp(&self.distance(last, b))
                        .expect("finite")
                })
                .expect("a city left");
            seen[next] = true;
            order.push(next);
        }
        order
    }

    /// The nearest-neighbour tour's length.
    pub fn reference(&self) -> f64 {
        self.reference
    }

    /// The order of cities a walk of states visited.
    pub fn order(&self, route: &[TourState]) -> Vec<usize> {
        let mut order: Vec<usize> = route.iter().map(|s| s.at as usize).collect();
        if order.len() > 1 && order.last() == order.first() {
            order.pop();
        }
        order
    }

    /// The best of `tries` random tours, as a quality (the reference
    /// length over the tour's).
    pub fn best_random_quality(&self, tries: usize, seed: u64) -> f64 {
        let mut rng = Rng::seed_from_u64(seed);
        let mut best = 0.0f64;
        for _ in 0..tries {
            let order = rng.permutation(self.cities.len());
            best = best.max(self.reference / self.length_of(&order).max(1e-9));
        }
        best
    }

    fn all(&self) -> u64 {
        let n = self.cities.len();
        if n >= 64 {
            u64::MAX
        } else {
            (1u64 << n) - 1
        }
    }
}

impl Problem for Tour {
    type State = TourState;

    fn embedding(&self) -> WorldConfig {
        let mut cfg = embedding(self.width, self.height, self.cities[0].cell(), 0);
        cfg.cell_capacity = 32;
        cfg
    }

    fn origin(&self) -> TourState {
        TourState {
            at: 0,
            seen: 1,
            length: 0.0,
        }
    }

    fn place(&self, state: &TourState) -> Point {
        self.cities[state.at as usize]
    }

    fn moves(&self, state: &TourState) -> Moves<TourState> {
        let all = self.all();
        if state.seen == all {
            if state.at != 0 {
                let d = self.distance(state.at as usize, 0);
                return Moves::States(vec![TourState {
                    at: 0,
                    seen: all,
                    length: state.length + d as f32,
                }]);
            }
            return Moves::States(Vec::new());
        }
        let n = self.cities.len();
        let out = (0..n)
            .filter(|&c| state.seen & (1u64 << c) == 0)
            .map(|c| TourState {
                at: c as u8,
                seen: state.seen | (1u64 << c),
                length: state.length + self.distance(state.at as usize, c) as f32,
            })
            .collect();
        Moves::States(out)
    }

    fn quality(&self, state: &TourState) -> Option<f64> {
        (state.seen == self.all() && state.at == 0 && state.length > 0.0)
            .then(|| self.reference / (state.length as f64).max(1e-9))
    }

    /// The smell of a move: how near the city is compared with the
    /// nearest city not yet seen, squared, so that the nearest smells
    /// of 1 and one twice as far of a quarter (the visibility of ant
    /// colony optimisation, with its usual exponent).
    fn scent(&self, from: &TourState, to: &TourState) -> f64 {
        let n = self.cities.len();
        let here = from.at as usize;
        let nearest = (0..n)
            .filter(|&c| c != here && from.seen & (1u64 << c) == 0)
            .map(|c| self.distance(here, c))
            .fold(f64::INFINITY, f64::min);
        let d = self.distance(here, to.at as usize).max(1e-9);
        if nearest.is_finite() && nearest > 0.0 {
            (nearest / d).min(1.0).powi(2)
        } else {
            self.mean_edge / (self.mean_edge + d)
        }
    }

    /// The cities themselves, each a little off its centre so that no
    /// route point lies on a ray: a tour's class is how it winds round
    /// them.
    fn punctures(&self) -> Vec<Point> {
        self.cities
            .iter()
            .map(|c| Point::new(c.x + 0.3, c.y + 0.3))
            .collect()
    }

    fn describe(&self, state: &TourState) -> String {
        format!(
            "{} of {} cities seen, {:.1} cells walked",
            state.seen.count_ones(),
            self.cities.len(),
            state.length
        )
    }

    fn name(&self) -> String {
        "tour".to_string()
    }
}

/// The cell of a city (for placing things on the grid).
pub fn city_cell(p: Point) -> Position {
    p.cell()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tour_closes_at_the_origin_with_the_reference_quality() {
        let tour = Tour::new(
            &[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            32,
            32,
        );
        assert_eq!(tour.len(), 4);
        let nn = tour.nearest_neighbour();
        assert_eq!(nn, vec![0, 1, 2, 3]);
        assert!((tour.length_of(&nn) - tour.reference()).abs() < 1e-9);
        let mut state = tour.origin();
        assert!(tour.quality(&state).is_none());
        let mut route = vec![state.clone()];
        for city in [1u8, 2, 3] {
            let Moves::States(next) = tour.moves(&state) else {
                panic!("states")
            };
            state = next.into_iter().find(|s| s.at == city).expect("open");
            route.push(state.clone());
        }
        let Moves::States(home) = tour.moves(&state) else {
            panic!("states")
        };
        assert_eq!(home.len(), 1, "only the way home is left");
        state = home[0].clone();
        route.push(state.clone());
        assert!((tour.quality(&state).unwrap() - 1.0).abs() < 1e-9);
        assert_eq!(tour.order(&route), vec![0, 1, 2, 3]);
        assert!(matches!(tour.moves(&state), Moves::States(v) if v.is_empty()));
        // The nearest city smells of 1, a farther one of less.
        let origin = tour.origin();
        let Moves::States(next) = tour.moves(&origin) else {
            panic!("states")
        };
        let near = next.iter().find(|s| s.at == 1).unwrap();
        let far = next.iter().find(|s| s.at == 2).unwrap();
        assert!((tour.scent(&origin, near) - 1.0).abs() < 1e-9);
        assert!(tour.scent(&origin, far) < 0.6);
    }
}
