//! Relations: a knowledge graph walked as a problem, the holonomy
//! embedding's flagship instantiation (see `extras/docs/holonomy.md`).
//!
//! Entities have places on a chart; a relation is a move between two
//! entities, with a translation as its transport; a query `(head,
//! relation, ?)` is a trip that sets out from the head with the query
//! relation as the first letter of its word and its integrated state at
//! the head's vector, walks relations (not the query relation itself
//! from the head, either way; not straight back; no more than a few
//! hops), and
//! finds food at a known answer, the more the nearer its integrated
//! state has come to the head's vector plus the query's. The word of a
//! trip brought home is a *rule*: `grandparent: parent parent`. The
//! symbols registered are the rules found, their precision the share
//! of trips of that word that came home, their transports the mean
//! state they arrived with.
//!
//! Learning is local and needs no gradient: a trip brought home credits
//! the residual between its integrated state and its target back to the
//! relations of its path (closure), and pulls the head's vector plus
//! the query's toward the answer's vector, away from a random entity's
//! (the translational embedding's own rule, learned by arrival).
//! Prediction of a held-out triple ranks every entity by the geometry,
//! by the rules, and by both.

use crate::problem::{embedding, Moves, Problem};
use crate::topos::{link_letter, link_of, prefix_letter, Letter, PathNet};
use ant_simulator::geometry::{Point, Position};
use ant_simulator::rng::Rng;
use ant_simulator::world::WorldConfig;
use std::collections::{HashMap, HashSet};

/// A fact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Triple {
    /// The head entity.
    pub head: u32,
    /// The relation.
    pub relation: u16,
    /// The tail entity.
    pub tail: u32,
}

/// Where a thought stands on a query's trip.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Node {
    /// The entity stood at.
    pub at: u32,
    /// The entity the trip set out from.
    pub head: u32,
    /// The relation asked about.
    pub query: u16,
    /// Moves made.
    pub hops: u8,
    /// The relation last followed and its direction.
    pub last: Option<(u16, bool)>,
    /// The entity before this one (not to be gone straight back to).
    pub prev: u32,
}

/// How the prediction of held-out triples fared.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Evaluation {
    /// Triples evaluated.
    pub triples: usize,
    /// Mean reciprocal rank by the geometry alone.
    pub mrr_geometry: f64,
    /// Share ranked first by the geometry.
    pub hits1_geometry: f64,
    /// Share ranked in the first ten by the geometry.
    pub hits10_geometry: f64,
    /// Mean reciprocal rank by the rules alone.
    pub mrr_rules: f64,
    /// Share ranked first by the rules.
    pub hits1_rules: f64,
    /// Share ranked in the first ten by the rules.
    pub hits10_rules: f64,
    /// Mean reciprocal rank by both.
    pub mrr: f64,
    /// Share ranked first by both.
    pub hits1: f64,
    /// Share ranked in the first ten by both.
    pub hits10: f64,
}

impl std::fmt::Display for Evaluation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "{} held-out triples: MRR {:.3} hits@1 {:.3} hits@10 {:.3} (geometry {:.3}/{:.3}/{:.3}, rules {:.3}/{:.3}/{:.3})",
            self.triples,
            self.mrr,
            self.hits1,
            self.hits10,
            self.mrr_geometry,
            self.hits1_geometry,
            self.hits10_geometry,
            self.mrr_rules,
            self.hits1_rules,
            self.hits10_rules
        )
    }
}

/// A knowledge graph with entity and relation vectors.
#[derive(Clone, Debug)]
pub struct Relations {
    entities: usize,
    relation_names: Vec<String>,
    adjacency: Vec<Vec<(u16, u32, bool)>>,
    known: HashSet<(u32, u16, u32)>,
    answers: HashMap<(u32, u16), Vec<u32>>,
    queries: Vec<(u32, u16)>,
    places: Vec<Point>,
    width: usize,
    height: usize,
    dim: usize,
    rel: Vec<Vec<f64>>,
    ent: Vec<Vec<f64>>,
    sigma: f64,
    eta: f64,
    margin: f64,
    max_hops: u8,
    lcg: u64,
    /// Learning steps taken.
    pub updates: u64,
}

impl Relations {
    /// A graph of `entities` entities and the given relations, from
    /// its facts (the training triples), on a chart of the given size
    /// with vectors of `dim` dimensions.
    pub fn new(
        entities: usize,
        relation_names: &[&str],
        facts: &[Triple],
        width: usize,
        height: usize,
        dim: usize,
        seed: u64,
    ) -> Relations {
        let mut rng = Rng::seed_from_u64(seed);
        let relations = relation_names.len();
        let mut adjacency = vec![Vec::new(); entities];
        let mut known = HashSet::new();
        let mut answers: HashMap<(u32, u16), Vec<u32>> = HashMap::new();
        for t in facts {
            if t.head as usize >= entities
                || t.tail as usize >= entities
                || t.relation as usize >= relations
            {
                continue;
            }
            if !known.insert((t.head, t.relation, t.tail)) {
                continue;
            }
            adjacency[t.head as usize].push((t.relation, t.tail, true));
            adjacency[t.tail as usize].push((t.relation, t.head, false));
            answers
                .entry((t.head, t.relation))
                .or_default()
                .push(t.tail);
        }
        let mut queries: Vec<(u32, u16)> = answers.keys().copied().collect();
        queries.sort();
        let places: Vec<Point> = (0..entities)
            .map(|_| {
                Point::new(
                    rng.range(2.0, width as f64 - 2.0).floor() + 0.5,
                    rng.range(2.0, height as f64 - 2.0).floor() + 0.5,
                )
            })
            .collect();
        let unit = |rng: &mut Rng| -> Vec<f64> {
            let v: Vec<f64> = (0..dim).map(|_| rng.normal()).collect();
            let n = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-9);
            v.into_iter().map(|x| x / n).collect()
        };
        let rel = (0..relations).map(|_| unit(&mut rng)).collect();
        let ent = (0..entities).map(|_| unit(&mut rng)).collect();
        Relations {
            entities,
            relation_names: relation_names.iter().map(|s| s.to_string()).collect(),
            adjacency,
            known,
            answers,
            queries,
            places,
            width,
            height,
            dim,
            rel,
            ent,
            sigma: 1.0,
            eta: 0.05,
            margin: 1.0,
            max_hops: 3,
            lcg: seed | 1,
            updates: 0,
        }
    }

    /// Place the entities (one point each).
    pub fn with_places(mut self, places: Vec<Point>) -> Relations {
        if places.len() == self.entities {
            self.places = places;
        }
        self
    }

    /// Ask only about these relations (all with answers by default).
    pub fn with_queries_about(mut self, relations: &[u16]) -> Relations {
        self.queries.retain(|(_, r)| relations.contains(r));
        self
    }

    /// The longest rule body, in hops.
    pub fn with_max_hops(mut self, hops: u8) -> Relations {
        self.max_hops = hops.max(1);
        self
    }

    /// The learning rate, the width of the arrival's match, and the
    /// margin negatives are pushed beyond.
    pub fn with_learning(mut self, eta: f64, sigma: f64, margin: f64) -> Relations {
        self.eta = eta.max(0.0);
        self.sigma = sigma.max(1e-6);
        self.margin = margin.max(0.0);
        self
    }

    /// Number of entities.
    pub fn entities(&self) -> usize {
        self.entities
    }

    /// The relations' names.
    pub fn relations(&self) -> &[String] {
        &self.relation_names
    }

    /// Number of facts.
    pub fn facts(&self) -> usize {
        self.known.len()
    }

    /// The queries asked, `(head, relation)`.
    pub fn queries(&self) -> &[(u32, u16)] {
        &self.queries
    }

    /// An entity's vector.
    pub fn entity_vector(&self, e: u32) -> &[f64] {
        &self.ent[e as usize]
    }

    /// A relation's vector.
    pub fn relation_vector(&self, r: u16) -> &[f64] {
        &self.rel[r as usize]
    }

    /// The letter of a relation followed forwards or backwards.
    pub fn letter_of(&self, relation: u16, forward: bool) -> Letter {
        link_letter(relation as usize, forward)
    }

    /// The letter a query's word begins with.
    pub fn query_letter(&self, relation: u16) -> Letter {
        prefix_letter(relation as usize)
    }

    fn target_of(&self, head: u32, query: u16) -> Vec<f64> {
        self.ent[head as usize]
            .iter()
            .zip(&self.rel[query as usize])
            .map(|(e, r)| e + r)
            .collect()
    }

    fn dist2(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    fn matching(&self, x: &[f64], target: &[f64]) -> f64 {
        (-Self::dist2(x, target) / (2.0 * self.sigma * self.sigma)).exp()
    }

    fn normalise(v: &mut [f64]) {
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n > 1e-9 {
            v.iter_mut().for_each(|x| *x /= n);
        }
    }

    fn random_entity(&mut self) -> u32 {
        self.lcg = self
            .lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.lcg >> 33) as usize % self.entities.max(1)) as u32
    }

    /// The entities reached from `head` by following a rule body: the
    /// letters of a word after its prefix, each a relation and a
    /// direction, never straight back.
    pub fn follow(&self, head: u32, body: &[Letter]) -> HashSet<u32> {
        let mut frontier: Vec<(u32, u32)> = vec![(head, u32::MAX)];
        for &l in body {
            let Some((relation, forward)) = link_of(l) else {
                continue;
            };
            let relation = relation as u16;
            let mut next: Vec<(u32, u32)> = Vec::new();
            let mut seen: HashSet<u32> = HashSet::new();
            for (at, prev) in frontier {
                for &(r, other, fwd) in &self.adjacency[at as usize] {
                    if r == relation && fwd == forward && other != prev && seen.insert(other) {
                        next.push((other, at));
                    }
                }
            }
            frontier = next;
            if frontier.is_empty() {
                break;
            }
        }
        frontier.into_iter().map(|(e, _)| e).collect()
    }

    /// The rank of the tail among all entities for a query, by the
    /// geometry, by the rules the network holds, and by both (each
    /// filtered: other known answers of the query do not count).
    pub fn rank(&self, triple: Triple, net: Option<&PathNet>) -> (usize, usize, usize) {
        let target = self.target_of(triple.head, triple.relation);
        let known: HashSet<u32> = self
            .answers
            .get(&(triple.head, triple.relation))
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();
        let geometry: Vec<f64> = (0..self.entities)
            .map(|e| self.matching(&self.ent[e], &target))
            .collect();
        let mut rules = vec![0.0; self.entities];
        if let Some(net) = net {
            let prefix = prefix_letter(triple.relation as usize);
            for k in net.living() {
                let s = &net.symbols()[k];
                let letters = s.word.letters();
                if letters.first() != Some(&prefix) {
                    continue;
                }
                let weight = s.precision() * (1.0 + s.support as f64).ln();
                for e in self.follow(triple.head, &letters[1..]) {
                    rules[e as usize] += weight;
                }
            }
        }
        let rank_by = |score: &dyn Fn(usize) -> f64| -> usize {
            let mine = score(triple.tail as usize);
            1 + (0..self.entities)
                .filter(|&e| {
                    e as u32 != triple.tail
                        && !known.contains(&(e as u32))
                        && score(e) > mine + 1e-12
                })
                .count()
        };
        let by_geometry = rank_by(&|e| geometry[e]);
        let by_rules = rank_by(&|e| rules[e]);
        let by_both = rank_by(&|e| rules[e] + geometry[e]);
        (by_geometry, by_rules, by_both)
    }

    /// Evaluate the prediction of held-out triples.
    pub fn evaluate(&self, tests: &[Triple], net: Option<&PathNet>) -> Evaluation {
        let mut ev = Evaluation {
            triples: tests.len(),
            ..Evaluation::default()
        };
        if tests.is_empty() {
            return ev;
        }
        for t in tests {
            let (g, r, b) = self.rank(*t, net);
            ev.mrr_geometry += 1.0 / g as f64;
            ev.hits1_geometry += (g == 1) as u8 as f64;
            ev.hits10_geometry += (g <= 10) as u8 as f64;
            ev.mrr_rules += 1.0 / r as f64;
            ev.hits1_rules += (r == 1) as u8 as f64;
            ev.hits10_rules += (r <= 10) as u8 as f64;
            ev.mrr += 1.0 / b as f64;
            ev.hits1 += (b == 1) as u8 as f64;
            ev.hits10 += (b <= 10) as u8 as f64;
        }
        let n = tests.len() as f64;
        for v in [
            &mut ev.mrr_geometry,
            &mut ev.hits1_geometry,
            &mut ev.hits10_geometry,
            &mut ev.mrr_rules,
            &mut ev.hits1_rules,
            &mut ev.hits10_rules,
            &mut ev.mrr,
            &mut ev.hits1,
            &mut ev.hits10,
        ] {
            *v /= n;
        }
        ev
    }

    /// How well a rule body closes: the mean distance between the sum
    /// of the body's relation vectors and the query relation's, over
    /// the unit sphere the entities live on.
    pub fn closure(&self, query: u16, body: &[(u16, bool)]) -> f64 {
        let mut sum = vec![0.0; self.dim];
        for &(r, fwd) in body {
            for (s, v) in sum.iter_mut().zip(&self.rel[r as usize]) {
                *s += if fwd { *v } else { -*v };
            }
        }
        Self::dist2(&sum, &self.rel[query as usize]).sqrt()
    }

    /// Families: `families` of them, each two grandparents, two or
    /// three children with a spouse each, and one to three
    /// grandchildren per couple. Base facts: `parent` (child to
    /// parent), `sibling` (both ways) and `spouse` (both ways).
    /// Derived: `grandparent`, `uncle` (a parent's sibling) and
    /// `cousin` (an uncle's child); a share `holdout` of the derived
    /// facts is held out as tests and returned. The families sit in
    /// blobs on the chart.
    pub fn family(
        families: usize,
        width: usize,
        height: usize,
        dim: usize,
        holdout: f64,
        seed: u64,
    ) -> (Relations, Vec<Triple>) {
        const PARENT: u16 = 0;
        const SIBLING: u16 = 1;
        const SPOUSE: u16 = 2;
        const GRANDPARENT: u16 = 3;
        const UNCLE: u16 = 4;
        const COUSIN: u16 = 5;
        let names = [
            "parent",
            "sibling",
            "spouse",
            "grandparent",
            "uncle",
            "cousin",
        ];
        let mut rng = Rng::seed_from_u64(seed);
        let mut places: Vec<Point> = Vec::new();
        let mut base: Vec<Triple> = Vec::new();
        let mut derived: Vec<Triple> = Vec::new();
        let cols = (families as f64).sqrt().ceil().max(1.0) as usize;
        let rows = families.div_ceil(cols).max(1);
        let cell_w = (width as f64 - 4.0) / cols as f64;
        let cell_h = (height as f64 - 4.0) / rows as f64;
        let fact = |v: &mut Vec<Triple>, h: u32, r: u16, t: u32| {
            v.push(Triple {
                head: h,
                relation: r,
                tail: t,
            })
        };
        for f in 0..families {
            let (cx, cy) = (
                2.0 + (f % cols) as f64 * cell_w + cell_w / 2.0,
                2.0 + (f / cols) as f64 * cell_h + cell_h / 2.0,
            );
            let person = |places: &mut Vec<Point>, rng: &mut Rng| -> u32 {
                let r = (cell_w.min(cell_h) / 2.0 - 1.0).max(1.0);
                let p = Point::new(
                    (cx + rng.range(-r, r))
                        .clamp(1.0, width as f64 - 2.0)
                        .floor()
                        + 0.5,
                    (cy + rng.range(-r, r))
                        .clamp(1.0, height as f64 - 2.0)
                        .floor()
                        + 0.5,
                );
                places.push(p);
                (places.len() - 1) as u32
            };
            let g1 = person(&mut places, &mut rng);
            let g2 = person(&mut places, &mut rng);
            fact(&mut base, g1, SPOUSE, g2);
            fact(&mut base, g2, SPOUSE, g1);
            let k = 2 + rng.below(2);
            let mut children: Vec<(u32, u32, Vec<u32>)> = Vec::new();
            for _ in 0..k {
                let c = person(&mut places, &mut rng);
                let s = person(&mut places, &mut rng);
                fact(&mut base, c, PARENT, g1);
                fact(&mut base, c, PARENT, g2);
                fact(&mut base, c, SPOUSE, s);
                fact(&mut base, s, SPOUSE, c);
                let m = 1 + rng.below(3);
                let kids: Vec<u32> = (0..m).map(|_| person(&mut places, &mut rng)).collect();
                for &kid in &kids {
                    fact(&mut base, kid, PARENT, c);
                    fact(&mut base, kid, PARENT, s);
                    fact(&mut derived, kid, GRANDPARENT, g1);
                    fact(&mut derived, kid, GRANDPARENT, g2);
                }
                for &a in &kids {
                    for &b in &kids {
                        if a != b {
                            fact(&mut base, a, SIBLING, b);
                        }
                    }
                }
                children.push((c, s, kids));
            }
            for (i, (ci, _, _)) in children.iter().enumerate() {
                for (j, (cj, _, _)) in children.iter().enumerate() {
                    if i != j {
                        fact(&mut base, *ci, SIBLING, *cj);
                    }
                }
            }
            for (i, (_, _, kids_i)) in children.iter().enumerate() {
                for (j, (cj, _, kids_j)) in children.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    for &kid in kids_i {
                        fact(&mut derived, kid, UNCLE, *cj);
                        for &cousin in kids_j {
                            fact(&mut derived, kid, COUSIN, cousin);
                        }
                    }
                }
            }
        }
        rng.shuffle(&mut derived);
        let held = ((derived.len() as f64) * holdout.clamp(0.0, 0.9)).round() as usize;
        let tests: Vec<Triple> = derived[..held].to_vec();
        let mut facts = base;
        facts.extend_from_slice(&derived[held..]);
        let entities = places.len();
        let relations = Relations::new(entities, &names, &facts, width, height, dim, seed)
            .with_places(places)
            .with_queries_about(&[SIBLING, GRANDPARENT, UNCLE, COUSIN]);
        (relations, tests)
    }
}

impl Problem for Relations {
    type State = Node;

    fn embedding(&self) -> WorldConfig {
        let nest = self
            .queries
            .first()
            .map(|(h, _)| self.places[*h as usize].cell())
            .unwrap_or(Position::new(1, 1));
        let mut cfg = embedding(self.width, self.height, nest, 0);
        cfg.cell_capacity = 64;
        cfg
    }

    fn origin(&self) -> Node {
        self.depart(0)
    }

    fn depart(&self, trip: u64) -> Node {
        let (head, query) = self
            .queries
            .get((trip % self.queries.len().max(1) as u64) as usize)
            .copied()
            .unwrap_or((0, 0));
        Node {
            at: head,
            head,
            query,
            hops: 0,
            last: None,
            prev: u32::MAX,
        }
    }

    fn place(&self, state: &Node) -> Point {
        self.places[state.at as usize]
    }

    fn moves(&self, state: &Node) -> Moves<Node> {
        if state.hops >= self.max_hops {
            return Moves::States(Vec::new());
        }
        let out = self.adjacency[state.at as usize]
            .iter()
            .filter(|&&(r, other, _)| other != state.prev && !(state.hops == 0 && r == state.query))
            .map(|&(r, other, fwd)| Node {
                at: other,
                head: state.head,
                query: state.query,
                hops: state.hops + 1,
                last: Some((r, fwd)),
                prev: state.at,
            })
            .collect();
        Moves::States(out)
    }

    fn quality(&self, state: &Node) -> Option<f64> {
        (state.hops >= 1
            && self
                .answers
                .get(&(state.head, state.query))
                .map(|a| a.contains(&state.at))
                .unwrap_or(false))
        .then_some(1.0)
    }

    fn quality_at(&self, state: &Node, x: &[f64]) -> Option<f64> {
        let q = self.quality(state)?;
        if x.len() != self.dim {
            return Some(q);
        }
        let target = self.target_of(state.head, state.query);
        Some(0.5 * q + 0.5 * self.matching(x, &target))
    }

    fn scent(&self, _from: &Node, _to: &Node) -> f64 {
        0.0
    }

    fn scent_at(&self, from: &Node, to: &Node, x: &[f64]) -> f64 {
        if x.len() != self.dim {
            return 0.0;
        }
        let Some((r, fwd)) = to.last else {
            return 0.0;
        };
        let target = self.target_of(from.head, from.query);
        let mut ahead = x.to_vec();
        for (a, v) in ahead.iter_mut().zip(&self.rel[r as usize]) {
            *a += if fwd { *v } else { -*v };
        }
        self.matching(&ahead, &target)
    }

    fn describe(&self, state: &Node) -> String {
        format!(
            "entity {} after {} hops asking {} of {}",
            state.at, state.hops, self.relation_names[state.query as usize], state.head
        )
    }

    fn name(&self) -> String {
        "relations".to_string()
    }

    fn prefix(&self, origin: &Node) -> Vec<Letter> {
        vec![prefix_letter(origin.query as usize)]
    }

    fn letter(&self, _from: &Node, to: &Node) -> Option<Letter> {
        to.last.map(|(r, fwd)| link_letter(r as usize, fwd))
    }

    fn letter_names(&self) -> Vec<(Letter, String)> {
        let mut names = Vec::new();
        for (i, n) in self.relation_names.iter().enumerate() {
            names.push((link_letter(i, true), n.clone()));
            names.push((prefix_letter(i), format!("{n}:")));
        }
        names
    }

    fn capacity(&self) -> usize {
        self.dim
    }

    fn departure(&self, origin: &Node) -> Vec<f64> {
        self.ent[origin.head as usize].clone()
    }

    fn integrate(&self, _from: &Node, to: &Node, x: &mut [f64]) {
        let Some((r, fwd)) = to.last else {
            return;
        };
        for (a, v) in x.iter_mut().zip(&self.rel[r as usize]) {
            *a += if fwd { *v } else { -*v };
        }
    }

    fn target(&self, origin: &Node) -> Option<Vec<f64>> {
        Some(self.target_of(origin.head, origin.query))
    }

    fn learn(&mut self, origin: &Node, route: &[Node], x: &[f64], quality: f64) {
        if quality <= 0.0 || route.len() < 2 || x.len() != self.dim {
            return;
        }
        self.updates += 1;
        let eta = self.eta;
        let (head, query) = (origin.head, origin.query);
        let tail = route.last().map(|n| n.at).unwrap_or(head);
        // Closure: the path's relations meet the query's halfway.
        let target = self.target_of(head, query);
        let residual: Vec<f64> = x.iter().zip(&target).map(|(a, b)| a - b).collect();
        let moves: Vec<(u16, bool)> = route[1..].iter().filter_map(|n| n.last).collect();
        let n = moves.len().max(1) as f64;
        for &(r, fwd) in &moves {
            for (v, e) in self.rel[r as usize].iter_mut().zip(&residual) {
                *v -= if fwd { eta * e / n } else { -eta * e / n };
            }
        }
        for (v, e) in self.rel[query as usize].iter_mut().zip(&residual) {
            *v += 0.5 * eta * e;
        }
        // Arrival: head + query ≈ tail, and not ≈ some other entity.
        let delta: Vec<f64> = (0..self.dim)
            .map(|d| {
                self.ent[head as usize][d] + self.rel[query as usize][d]
                    - self.ent[tail as usize][d]
            })
            .collect();
        for (d, &delta_d) in delta.iter().enumerate() {
            self.ent[tail as usize][d] += eta * delta_d;
            self.ent[head as usize][d] -= eta * delta_d;
            self.rel[query as usize][d] -= 0.5 * eta * delta_d;
        }
        let other = self.random_entity();
        if other != tail && other != head {
            let away: Vec<f64> = (0..self.dim)
                .map(|d| {
                    self.ent[head as usize][d] + self.rel[query as usize][d]
                        - self.ent[other as usize][d]
                })
                .collect();
            let dist = away.iter().map(|v| v * v).sum::<f64>().sqrt();
            if dist < self.margin {
                for (d, &away_d) in away.iter().enumerate() {
                    self.ent[other as usize][d] -= eta * away_d;
                    self.ent[head as usize][d] += 0.5 * eta * away_d;
                }
            }
        }
        for e in [head, tail, other] {
            Self::normalise(&mut self.ent[e as usize]);
        }
    }
}
