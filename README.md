# ant_simulator

An ant colony simulator, written in Rust with no dependencies, built around one
idea:

> Behaviour is governed through **entropy**, by a **hierarchy of general
> objects**, operated through **levers whose connections are hidden and rotate**
> in a sequence that is random, but static, and therefore learnable.

The colony is a real simulation (stigmergy, foraging, starvation,
reproduction). The control system on top of it is the point.

## Concepts

| Phrase | In the crate |
| --- | --- |
| System of categorization | [`Hierarchy`](src/hierarchy.rs): a tree such as `colony → castes → squads`, with the ants hanging off the leaves. Any shape can be specified. |
| General object at every level | [`Node`](src/hierarchy.rs): every level is the same object. It owns a `BehavioralSurface` and controls everything beneath it. |
| Entropic behavioral surface | [`BehavioralSurface`](src/surface.rs): a weight vector over the ant's senses (the surface: preferences over what to do) plus an [`EntropyControl`](src/entropy.rs) (the dial: how much disorder those preferences are executed with). |
| Entropic control | On every decision the action distribution is *solved* to have exactly the requested fraction of maximum entropy. Dial at 0: the ant follows the surface deterministically. Dial at 1: a random walk. |
| Control flows down the hierarchy | Surfaces add up along the root-to-leaf path; entropy dials compose (the root sets an absolute level, every level below scales what it inherits). |
| Multiple learners | [`Learner`](src/learner/mod.rs) implementations: hill climbing, rotation search, cross-entropy method, policy gradient, an entropy-only bandit, plus static and random baselines. |
| Randomly rotated in a sequence | [`Rotation`](src/rotation.rs): each turn maps levers to nodes. `RandomStatic { period }` draws `period` random permutations once and replays them forever. |
| Learning to interact with an unknown connection | [`PhaseAware`](src/learner/phase_aware.rs) wraps any learner, infers the rotation period from the pattern of its own rewards, and keeps a separate copy of the learner per phase. |
| The path as a surface flat in front of you | [`Landscape`](src/landscape.rs): the eight candidate directions as a ring around the ant's heading, scored by the effective surface; stacked along a trajectory they are the path's surface (`Simulation::surface_trace`, `render_surface`). |
| Deformation over the deterministic information | The scores are the deterministic information. The entropy budget is spent on them through separable geometric channels: tempering, smoothing along the ring, and a random roughening field ([`Deformation`](src/surface.rs)). |
| Geometric selection, like a sucker | [`Sucker`](src/landscape.rs): a walker that starts straight ahead and crawls the ring by local Metropolis moves for a bounded reach, settling in the basin it can get to (`Selection::Sucker`). |
| Behaviour as separable entropy | The [`EntropyLedger`](src/landscape.rs) decomposes every decision's entropy into tempering, smoothing, roughening, selection, and the disorder the field injects between decisions; [`PathStats`](src/colony.rs) measures what that does to the paths. |

A learner never sees the hierarchy. It gets a [`LeverView`](src/learner/mod.rs):
the parameter vector at the far end of its lever, the turn number, and the
reward from last time. Everything it knows about *what* it controls it has to
infer from the regularity of the feedback: how the lever feels (the
parameters it shows) and what it pays (the reward).

A lever is a dial, not a memory. Every learner in the crate moves the surface
*from where it finds it*: a bounded displacement that is kept if it paid and
reverted if it did not. A learner that wrote a remembered configuration onto
whatever it happened to be holding would, under a hidden rotation, overwrite
the colony's instinct with a squad's blank slate. (That is exactly what the
first version of the hill climber did, and the colony collapsed.)

## Quick start

```rust
use ant_simulator::prelude::*;

// A colony on instinct alone.
let mut sim = Simulation::new(SimConfig::default(), 42);
sim.run(500);
println!("{}", render(&sim));
println!("{:?}", sim.stats());

// Turn the colony-wide entropy dial down: ants follow their preferences
// more strictly. Turn a single caste up: that caste alone becomes erratic.
sim.hierarchy_mut().node_mut(0).surface.entropy = EntropyControl::absolute(0.15);
let scouts = sim.hierarchy().find("caste-0").unwrap();
sim.hierarchy_mut().node_mut(scouts).surface.entropy = EntropyControl::relative(3.0);
sim.reset_stats();
sim.run(500);

// Spend the budget geometrically instead, and select with a sucker.
let config = SimConfig {
    selection: Selection::Sucker { reach: 8 },
    instinct: BehavioralSurface::instinct().with_deformation(Deformation {
        smooth: 1.0,
        rough: 0.5,
        reach: 0.0,
    }),
    record_surface: Some(0),
    ..SimConfig::default()
};
let mut sim = Simulation::new(config, 7);
sim.run(200);
println!("{}", render_surface(sim.surface_trace(), 16));
println!("{:?}", sim.stats().path.ledger.contributions());
println!("turn entropy {:.3}", sim.stats().path.turn_entropy());
```

```rust
use ant_simulator::prelude::*;

// Several learners, hidden levers over every node, rotated with period 4.
let learners: Vec<Box<dyn Learner>> = vec![
    Box::new(PhaseAware::new(HillClimber::new(0.2), 8)),
    Box::new(HillClimber::new(0.2)),
    Box::new(PhaseAware::new(EntropyBandit::default(), 8)),
    Box::new(PolicyGradient::new(0.02)),
];
let config = ArenaConfig {
    sim: SimConfig { trace: true, ..SimConfig::default() },
    schedule: RotationSchedule::RandomStatic { period: 4 },
    targets: ControlTargets::AllNodes,
    feedback: FeedbackScope::Subtree,
    ..ArenaConfig::default()
};
let mut arena = Arena::new(config, learners, 7).unwrap();
let report = arena.run(100);
println!("{report}");
println!("{}", arena.hierarchy().describe());
```

## Path surfaces: separable entropy, deformation, and geometric selection

Every decision runs through a short geometric pipeline:

1. **Deterministic information.** The effective surface scores the eight
   walkable directions. Laid out as a ring around the heading (straight
   ahead in the middle) this is the landscape in front of the ant.
2. **Deformation with the entropy budget** `h` (the node's dial):
   * *smoothing* blurs the landscape along the ring with a circular
     Gaussian of scale `h × share × 2` ring steps: preference leaks to
     neighbouring directions (angular disorder, path-coherent);
   * *roughening* adds a random field of a few low Fourier modes with
     amplitude `h × share × range`: random basins (landscape disorder,
     path-incoherent);
   * *tempering* solves the temperature so the deformed landscape's
     distribution has exactly `h` of the maximum entropy.
   The shares are per node (`Deformation { smooth, rough, reach }`, three
   more parameters learners can move) and compose down the hierarchy like
   the weights.
3. **Selection.** `Selection::Softmax` draws globally from the tempered
   distribution. `Selection::Sucker { reach }` starts a walker straight
   ahead and lets it crawl the ring for `reach × exp(node reach)` local
   Metropolis steps; it goes where it can get to. Reach 0 never turns, reach
   1 turns at most 45°, and by reach 8 the walker is nearly a global draw
   unless the landscape has a valley between it and the peak.

`cargo run --release --example surface` shows one ant's surface scrolling
past under the sucker (`[..]` is the choice, the trail is the walk):

```
 tick  position    L135L90 L45  ^  R45 R90 R135rev   trail
  146 ( 52, 23)                 ** [::]              0→0→0→0→0→0→0→1→1
  147 ( 53, 23)             .. [**] ..               0→0→0→1→0→0→0→0→0
  157 ( 63, 23)                 xx  xx  xx [@@]      2→2→2→2→2→2→2→2→2
```

and then holds the budget fixed at 0.35 of maximum entropy while changing
its shape (400 ticks, mean of 3 seeds):

```
setting          target  drawn | temper smooth  rough  field select |  turnH  str%   eff | deliv
flat / softmax    0.703  0.703 |  0.703 +0.000 +0.000 +0.000 +0.000 |  1.197  54.4  0.96 |   421
smooth / softmax  0.705  0.705 |  0.579 +0.126 +0.000 +0.000 +0.000 |  1.163  55.5  0.97 |   415
rough / softmax   0.707  0.707 |  1.025 +0.000 -0.319 +0.629 +0.000 |  1.775  29.4  0.69 |   102
reach 1 / sucker  0.684  0.542 |  0.684 +0.000 +0.000 +0.000 -0.142 |  0.839  72.8  0.87 |   208
```

Same per-decision entropy, different paths. Smoothing spends a fifth of the
budget on angular blur and the paths stay as straight and efficient as
before. Roughening looks *sharper* inside each decision (a random spike is a
confident choice, so it displaces 0.32 nats of tempering) but the ledger's
`field` column shows the 0.63 nats it injects *between* decisions, and the
paths turn twice as much and deliver a quarter of the food. The entropy of
behaviour is separable, and the split between within-decision and
between-decision disorder is what the path's topology responds to. A short
sucker reach cuts the realised entropy below the target and binds the ant
to its heading; a long one converges back to the global draw.

## Examples

```
cargo run --release --example colony  [ticks]
cargo run --release --example arena   [turns] [period] [sucker]
cargo run --release --example surface [ticks] [seeds]
```

`colony` renders the world as ASCII, sweeps the colony-wide entropy dial, and
heats a single caste. `arena` runs six learners over the ten nodes of the
default hierarchy with a hidden rotation of period 4, prints which learners
recover the period, and finally evaluates the untouched instinct, the learned
hierarchy, and the best turn's parameters on the same fresh episodes:

```
  phase-aware(hill-climb)      ... period 4
  hill-climb                   ... period -     112 evaluations (never sure what it holds)
  phase-aware(entropy-bandit)  ... period 4
  policy-gradient              ... period -
  cross-entropy                ... period -
  static                       ... period -

evaluation over 12 fresh episodes:
  instinct (untouched): reward 297.9 ± 47.2, delivered 268, entropy 0.702 nats
  learned hierarchy:    reward 407.1 ± 56.1, delivered 366, entropy 0.346 nats
```

The colony-wide dial ends up near 0.18 of maximum entropy, which is also
where a brute-force sweep of the dial in the `colony` example puts the
optimum. Nobody told the learners that; two of them worked out the rotation
and one of them only ever touched entropy.

## How the simulation works

* **World**: a grid with a nest, food clusters (random or placed), optional
  walls, and two pheromone fields. Foraging ants lay *home* pheromone; ants
  carrying food lay *food* pheromone. Both evaporate and diffuse each tick.
* **Senses**: for each of the eight neighbouring cells an ant computes eight
  features (food and home pheromone, food present, nest present, alignment
  with its heading, alignment with the direction of the nest, recently
  visited, crowding). The same eight are repeated gated by "carrying food",
  so the surface can prefer different things in the two modes. Sixteen
  weights plus one dial value make a surface: 17 parameters per node.
* **Decision**: the effective surface of the ant's leaf scores each walkable
  direction; the scores are turned into a distribution whose entropy is
  exactly the effective dial's fraction of `ln(walkable directions)`; a
  direction is sampled.
* **Life**: ants pick food up automatically, deliver it at the nest, burn
  energy and refuel at the nest, and starve if they run out. The colony spends
  delivered food on new ants.
* **Reward**: configurable per delivery, pickup, death and birth. Every event
  is credited to the whole root-to-leaf path, so each node has a subtree
  reward that learners can be fed instead of the colony total.

## The arena, turn by turn

1. The rotation says which node each lever reaches this turn.
2. Each connected learner sees that node's parameters and writes new ones.
3. The colony runs for `steps_per_turn` ticks (a fresh episode, or one
   persistent colony, as configured).
4. Each connected learner receives its reward (colony-wide or subtree) and,
   if tracing is on, the REINFORCE score for the node it held. It may then
   leave different parameters behind (for instance its best known setting
   rather than the last probe) before the lever rotates away.

Schedules: `Fixed`, `Cyclic`, `RandomStatic { period }`, `Fresh` (a new
permutation every turn, unlearnable by construction), and `Explicit`.

## Learners

| Learner | What it does with the lever |
| --- | --- |
| `HillClimber` | (1+1) evolution strategy on displacements with adaptive step size. When the surface it is handed is not the one it left, it spends a turn feeling it out before probing. `Mutation::Rotation` rotates the weight vector in a random plane instead of adding noise. |
| `CrossEntropy` | Probes Gaussian displacements, applies the mean of the elite displacements once per generation, refits the spread. |
| `PolicyGradient` | REINFORCE using the traced score, with a normalised advantage and a unit gradient direction so the step size is independent of reward scale and temperature. |
| `DialBandit` | UCB1 over dial settings only, leaving the weights alone. `DialBandit::entropy()` (alias `EntropyBandit`) searches the entropy dial; `DialBandit::geometry()` searches smoothing, roughening and reach. |
| `PhaseAware<L>` | Infers the rotation period (BIC over candidate periods, from rewards and parameter fingerprints) and runs one `L` per phase. |
| `StaticLearner`, `RandomLearner` | Controls. |

Whether the period is discoverable depends on whether the categories differ.
Levers rotating over the root, a caste and a squad produce a strongly periodic
reward. Levers rotating over nodes with similar rewards can still be told
apart by the parameter vectors they show, as long as those have diverged.

## Evaluating what was learned

`Arena::evaluate(params, episodes, seed)` runs fresh episodes with any
parameter vector, so the instinct (`Arena::initial_params`), the current
hierarchy, and the best turn (`Arena::best`) can be compared on identical
seeds. Turn-by-turn rewards in the arena are noisy and are shaped by every
learner's exploration; the evaluation is the honest number.

## Reproducibility

Everything is driven by a single `u64` seed through the crate's own xoshiro
generator, so simulations, arenas and learners replay exactly. The world's
food layout can be pinned separately with `WorldConfig::seed`.

## Tests

```
cargo test
cargo clippy --all-targets
```
