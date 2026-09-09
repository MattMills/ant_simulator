//! Rotating which learner holds which surface.
//!
//! A [`Rotation`] produces, for every turn, a [`Mapping`] from levers
//! (learners) to controllable surfaces (hierarchy nodes chosen by the arena).
//! The interesting schedule is [`RotationSchedule::RandomStatic`]: a
//! sequence of `period` random permutations is drawn once and then replayed
//! forever. The mapping is random, so no learner can know a priori where its
//! lever connects; it is sequential and static, so the connection can be
//! learned from the pattern of feedback.

use crate::rng::Rng;
use std::fmt;

/// How the lever→surface mapping evolves over turns.
#[derive(Clone, Debug, PartialEq)]
pub enum RotationSchedule {
    /// Lever `i` always controls surface `i`.
    Fixed,
    /// Every turn each lever moves on to the next surface (a cyclic shift).
    Cyclic,
    /// `period` random permutations drawn once, then replayed in order.
    RandomStatic {
        /// Length of the replayed sequence.
        period: usize,
    },
    /// A fresh random permutation every turn; unlearnable by construction.
    Fresh,
    /// An explicit sequence of mappings (each entry is `surface_of_lever`),
    /// replayed in order.
    Explicit(Vec<Vec<usize>>),
}

/// Which surface each lever reaches on a given turn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mapping {
    /// `surface_of_lever[i]` is the surface index lever `i` controls, or
    /// `None` if the lever is disconnected this turn.
    pub surface_of_lever: Vec<Option<usize>>,
}

impl Mapping {
    /// The lever holding `surface`, if any.
    pub fn lever_of_surface(&self, surface: usize) -> Option<usize> {
        self.surface_of_lever
            .iter()
            .position(|s| *s == Some(surface))
    }

    /// Number of connected levers.
    pub fn connected(&self) -> usize {
        self.surface_of_lever.iter().filter(|s| s.is_some()).count()
    }

    fn from_permutation(perm: &[usize], levers: usize, surfaces: usize) -> Self {
        Mapping {
            surface_of_lever: (0..levers)
                .map(|i| perm.get(i).copied().filter(|&s| s < surfaces))
                .collect(),
        }
    }
}

impl fmt::Display for Mapping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self
            .surface_of_lever
            .iter()
            .enumerate()
            .map(|(i, s)| match s {
                Some(s) => format!("L{i}→S{s}"),
                None => format!("L{i}→∅"),
            })
            .collect();
        write!(f, "{}", parts.join(" "))
    }
}

/// Error building a rotation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RotationError {
    /// A `RandomStatic` period of zero.
    ZeroPeriod,
    /// An `Explicit` schedule with no mappings.
    EmptyExplicit,
    /// An `Explicit` mapping is not injective over its surfaces.
    NotInjective {
        /// Index of the offending mapping.
        index: usize,
    },
    /// No levers or no surfaces.
    Empty,
}

impl fmt::Display for RotationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RotationError::ZeroPeriod => write!(f, "rotation period must be at least 1"),
            RotationError::EmptyExplicit => write!(f, "explicit schedule has no mappings"),
            RotationError::NotInjective { index } => {
                write!(
                    f,
                    "explicit mapping {index} assigns one surface to two levers"
                )
            }
            RotationError::Empty => write!(f, "rotation needs at least one lever and one surface"),
        }
    }
}

impl std::error::Error for RotationError {}

/// The turn-by-turn lever assignment.
#[derive(Clone, Debug)]
pub struct Rotation {
    schedule: RotationSchedule,
    levers: usize,
    surfaces: usize,
    sequence: Vec<Mapping>,
    turn: u64,
}

impl Rotation {
    /// Build a rotation for `levers` learners over `surfaces` surfaces.
    pub fn new(
        schedule: RotationSchedule,
        levers: usize,
        surfaces: usize,
        rng: &mut Rng,
    ) -> Result<Self, RotationError> {
        if levers == 0 || surfaces == 0 {
            return Err(RotationError::Empty);
        }
        let m = levers.max(surfaces);
        let sequence = match &schedule {
            RotationSchedule::Fixed => {
                let perm: Vec<usize> = (0..m).collect();
                vec![Mapping::from_permutation(&perm, levers, surfaces)]
            }
            RotationSchedule::Cyclic => (0..m)
                .map(|t| {
                    let perm: Vec<usize> = (0..m).map(|i| (i + t) % m).collect();
                    Mapping::from_permutation(&perm, levers, surfaces)
                })
                .collect(),
            RotationSchedule::RandomStatic { period } => {
                if *period == 0 {
                    return Err(RotationError::ZeroPeriod);
                }
                (0..*period)
                    .map(|_| Mapping::from_permutation(&rng.permutation(m), levers, surfaces))
                    .collect()
            }
            RotationSchedule::Fresh => Vec::new(),
            RotationSchedule::Explicit(list) => {
                if list.is_empty() {
                    return Err(RotationError::EmptyExplicit);
                }
                let mut seq = Vec::with_capacity(list.len());
                for (index, entry) in list.iter().enumerate() {
                    let mapping = Mapping::from_permutation(entry, levers, surfaces);
                    let mut seen = vec![false; surfaces];
                    for s in mapping.surface_of_lever.iter().flatten() {
                        if seen[*s] {
                            return Err(RotationError::NotInjective { index });
                        }
                        seen[*s] = true;
                    }
                    seq.push(mapping);
                }
                seq
            }
        };
        Ok(Rotation {
            schedule,
            levers,
            surfaces,
            sequence,
            turn: 0,
        })
    }

    /// The schedule in use.
    pub fn schedule(&self) -> &RotationSchedule {
        &self.schedule
    }

    /// Number of levers.
    pub fn levers(&self) -> usize {
        self.levers
    }

    /// Number of surfaces.
    pub fn surfaces(&self) -> usize {
        self.surfaces
    }

    /// Period of the replayed sequence (`None` for `Fresh`).
    pub fn period(&self) -> Option<usize> {
        if self.sequence.is_empty() {
            None
        } else {
            Some(self.sequence.len())
        }
    }

    /// The mapping used at a given turn for replayed schedules.
    pub fn mapping_at(&self, turn: u64) -> Option<&Mapping> {
        if self.sequence.is_empty() {
            None
        } else {
            Some(&self.sequence[(turn % self.sequence.len() as u64) as usize])
        }
    }

    /// Turns issued so far.
    pub fn turn(&self) -> u64 {
        self.turn
    }

    /// Produce the mapping for the next turn.
    pub fn next(&mut self, rng: &mut Rng) -> Mapping {
        let turn = self.turn;
        self.turn += 1;
        match self.mapping_at(turn) {
            Some(m) => m.clone(),
            None => {
                let m = self.levers.max(self.surfaces);
                Mapping::from_permutation(&rng.permutation(m), self.levers, self.surfaces)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_injective(m: &Mapping) -> bool {
        let mut seen = std::collections::HashSet::new();
        m.surface_of_lever.iter().flatten().all(|s| seen.insert(*s))
    }

    #[test]
    fn fixed_and_cyclic() {
        let mut rng = Rng::seed_from_u64(1);
        let mut fixed = Rotation::new(RotationSchedule::Fixed, 3, 5, &mut rng).unwrap();
        let m = fixed.next(&mut rng);
        assert_eq!(m.surface_of_lever, vec![Some(0), Some(1), Some(2)]);
        assert_eq!(fixed.period(), Some(1));

        let mut cyclic = Rotation::new(RotationSchedule::Cyclic, 2, 3, &mut rng).unwrap();
        assert_eq!(cyclic.period(), Some(3));
        let a = cyclic.next(&mut rng);
        let b = cyclic.next(&mut rng);
        let c = cyclic.next(&mut rng);
        let d = cyclic.next(&mut rng);
        assert_eq!(a.surface_of_lever, vec![Some(0), Some(1)]);
        assert_eq!(b.surface_of_lever, vec![Some(1), Some(2)]);
        assert_eq!(c.surface_of_lever, vec![Some(2), Some(0)]);
        assert_eq!(a, d);
        assert_eq!(a.lever_of_surface(1), Some(1));
        assert_eq!(a.lever_of_surface(2), None);
    }

    #[test]
    fn random_static_replays_and_is_injective() {
        let mut rng = Rng::seed_from_u64(2);
        let mut rot =
            Rotation::new(RotationSchedule::RandomStatic { period: 4 }, 5, 3, &mut rng).unwrap();
        let first: Vec<Mapping> = (0..4).map(|_| rot.next(&mut rng)).collect();
        let second: Vec<Mapping> = (0..4).map(|_| rot.next(&mut rng)).collect();
        assert_eq!(first, second);
        for m in &first {
            assert!(is_injective(m));
            assert_eq!(m.connected(), 3, "3 surfaces, 5 levers → 3 connected");
        }
        assert_eq!(rot.turn(), 8);
        assert!(format!("{}", first[0]).contains("L0→"));
    }

    #[test]
    fn fresh_has_no_period() {
        let mut rng = Rng::seed_from_u64(3);
        let mut rot = Rotation::new(RotationSchedule::Fresh, 4, 4, &mut rng).unwrap();
        assert_eq!(rot.period(), None);
        assert!(rot.mapping_at(0).is_none());
        let ms: Vec<Mapping> = (0..20).map(|_| rot.next(&mut rng)).collect();
        assert!(ms.iter().all(is_injective));
        assert!(ms.iter().any(|m| *m != ms[0]));
    }

    #[test]
    fn explicit_validation() {
        let mut rng = Rng::seed_from_u64(4);
        assert_eq!(
            Rotation::new(RotationSchedule::Explicit(vec![]), 2, 2, &mut rng).err(),
            Some(RotationError::EmptyExplicit)
        );
        assert_eq!(
            Rotation::new(RotationSchedule::Explicit(vec![vec![0, 0]]), 2, 2, &mut rng).err(),
            Some(RotationError::NotInjective { index: 0 })
        );
        assert_eq!(
            Rotation::new(RotationSchedule::RandomStatic { period: 0 }, 2, 2, &mut rng).err(),
            Some(RotationError::ZeroPeriod)
        );
        assert_eq!(
            Rotation::new(RotationSchedule::Fixed, 0, 2, &mut rng).err(),
            Some(RotationError::Empty)
        );
        let mut ok = Rotation::new(
            RotationSchedule::Explicit(vec![vec![1, 0], vec![0, 7]]),
            2,
            2,
            &mut rng,
        )
        .unwrap();
        assert_eq!(ok.next(&mut rng).surface_of_lever, vec![Some(1), Some(0)]);
        assert_eq!(ok.next(&mut rng).surface_of_lever, vec![Some(0), None]);
    }
}
