//! The holonomy embedding on a knowledge graph of families: the mind
//! walks queries as trips, the words of the trips brought home are
//! rules, the relations' vectors learn to close along them, and
//! held-out facts are predicted by the geometry, by the rules and by
//! both. Run with `--release`.
//!
//! `cargo run --release -p ant_extras --example relations`

use ant_extras::mind::{Mind, MindConfig};
use ant_extras::problems::{Relations, Triple};
use ant_extras::sense::{F_HEADING, F_ODOUR, F_ROUTE, F_SITE};
use ant_extras::topos::{link_letter, prefix_letter, SymbolConfig, Word};
use std::time::Instant;

const PARENT: u16 = 0;
const SIBLING: u16 = 1;
const GRANDPARENT: u16 = 3;
const UNCLE: u16 = 4;
const COUSIN: u16 = 5;

fn main() {
    let families: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(30);
    let ticks: u64 = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(20000);
    let eta: f64 = std::env::args()
        .nth(3)
        .and_then(|a| a.parse().ok())
        .unwrap_or(0.01);
    let (problem, tests) = Relations::family(families, 96, 64, 16, 0.3, 7);
    let problem = problem.with_learning(eta, 1.0, 1.0);
    println!(
        "== {} families: {} entities, {} facts, {} queries, {} held-out facts",
        families,
        problem.entities(),
        problem.facts(),
        problem.queries().len(),
        tests.len()
    );
    type Rule = (&'static str, u16, Vec<(u16, bool)>);
    let planted: [Rule; 4] = [
        (
            "sibling: parent parent'",
            SIBLING,
            vec![(PARENT, true), (PARENT, false)],
        ),
        (
            "grandparent: parent parent",
            GRANDPARENT,
            vec![(PARENT, true), (PARENT, true)],
        ),
        (
            "uncle: parent sibling",
            UNCLE,
            vec![(PARENT, true), (SIBLING, true)],
        ),
        (
            "cousin: parent sibling parent'",
            COUSIN,
            vec![(PARENT, true), (SIBLING, true), (PARENT, false)],
        ),
    ];
    let closure = |p: &Relations| -> String {
        planted
            .iter()
            .map(|(n, q, body)| format!("{n} {:.2}", p.closure(*q, body)))
            .collect::<Vec<_>>()
            .join(", ")
    };
    println!(
        "  closure of the planted rules before: {}",
        closure(&problem)
    );
    println!("  before: {}", problem.evaluate(&tests, None));

    let cfg = MindConfig {
        thoughts: 64,
        speed: 16.0,
        trip_budget: 12,
        rest_ticks: 1,
        odour_release: 0.0,
        recruitment: 4.0,
        retreats: 2,
        site_fidelity: false,
        history: None,
        ..MindConfig::default()
    }
    .with_weight(F_HEADING, 0.0)
    .with_weight(F_ODOUR, 8.0)
    .with_weight(F_ROUTE, 0.0)
    .with_weight(F_SITE, 0.0)
    .with_symbols(SymbolConfig {
        support: 3,
        capacity: 256,
        max_word: 5,
        learn_punctures: false,
        ..SymbolConfig::default()
    });
    let start = Instant::now();
    let mut mind = Mind::new(problem, cfg);
    let quarters = 4;
    for q in 1..=quarters {
        mind.run(ticks / quarters);
        let s = mind.stats();
        println!(
            "  tick {}: {} trips, {} brought home, {} dead ends ({} retreats), {} given up, {} learning steps",
            mind.tick(),
            s.trips,
            s.deliveries,
            s.dead_ends,
            s.retreats,
            s.given_up,
            mind.problem().updates
        );
        if q == quarters {
            print!("{}", mind.report());
        }
    }
    println!(
        "  closure of the planted rules after: {}",
        closure(mind.problem())
    );
    let eval = mind.problem().evaluate(&tests, mind.net());
    println!("  after: {eval}");
    println!("  ({:.1} s)", start.elapsed().as_secs_f64());

    // Generation: a route of the cousin rule from a held-out cousin's
    // head, and whether it lands on the held-out tail.
    if let Some(t) = tests.iter().find(|t| t.relation == COUSIN) {
        let word = Word::from_letters(&[
            prefix_letter(COUSIN as usize),
            link_letter(PARENT as usize, true),
            link_letter(SIBLING as usize, true),
            link_letter(PARENT as usize, false),
        ]);
        let from = ant_extras::problems::Node {
            at: t.head,
            head: t.head,
            query: COUSIN,
            hops: 0,
            last: None,
            prev: u32::MAX,
        };
        let tail = t.tail;
        match mind.generate_states(&word, from, &|n| n.hops >= 3 && n.at == tail, 4) {
            Some(path) => println!(
                "  generated a route of the cousin rule from {} to held-out cousin {}: {:?}",
                t.head,
                tail,
                path.iter().map(|n| n.at).collect::<Vec<_>>()
            ),
            None => println!(
                "  no route of the cousin rule from {} reaches held-out cousin {}",
                t.head, tail
            ),
        }
    }
    let _ = Triple {
        head: 0,
        relation: 0,
        tail: 0,
    };
}
