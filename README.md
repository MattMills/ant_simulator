# ant_simulator

An ant colony simulator, written in Rust with no dependencies. Its foraging
biology follows the literature (pheromone kinetics, Deneubourg's choice
function, quality-modulated recruitment, path integration, response-threshold
task allocation) and is validated against the classic experiments. On top of
that biology sits one idea:

> Behaviour is governed through **entropy**, by a **hierarchy of general
> objects**, operated through **levers whose connections are hidden and rotate**
> in a sequence that is random, but static, and therefore learnable.

## Concepts

| Phrase | In the crate |
| --- | --- |
| System of categorization | [`Hierarchy`](src/hierarchy.rs): a tree such as `colony → castes → squads`, with the ants hanging off the leaves. Any shape can be specified. |
| General object at every level | [`Node`](src/hierarchy.rs): every level is the same object. It owns a `BehavioralSurface` and controls everything beneath it. |
| Entropic behavioral surface | [`BehavioralSurface`](src/surface.rs): a weight vector over the ant's senses plus an [`EntropyControl`](src/entropy.rs) dial. The dial either fixes a temperature (the biological default: 1 gives Deneubourg's choice function) or pins the entropy of every decision to a fraction of its maximum. |
| Control flows down the hierarchy | Surfaces add up along the root-to-leaf path; dials compose (an absolute setting at the root, gains below). |
| Multiple learners | [`Learner`](src/learner/mod.rs) implementations: hill climbing, rotation search, cross-entropy, policy gradient, dial bandits, plus static and random baselines. |
| Randomly rotated in a sequence | [`Rotation`](src/rotation.rs): each turn maps levers to nodes. `RandomStatic { period }` draws `period` random permutations once and replays them forever. |
| Learning to interact with an unknown connection | [`PhaseAware`](src/learner/phase_aware.rs) wraps any learner, infers the rotation period from its own rewards and parameter fingerprints, and keeps one learner per phase. |
| The path as a surface flat in front of you | [`Landscape`](src/landscape.rs): the eight candidate directions as a ring around the heading, scored by the effective surface; recorded and rendered by `render_surface`. |
| Deformation over the deterministic information | The scores are the deterministic information; the entropy budget is spent through separable geometric channels: tempering, smoothing along the ring, a random roughening field ([`Deformation`](src/surface.rs)). |
| Geometric selection, like a sucker | [`Sucker`](src/landscape.rs): a walker that starts straight ahead and crawls the ring by local Metropolis moves for a bounded reach (`Selection::Sucker`). |
| Behaviour as separable entropy | The [`EntropyLedger`](src/landscape.rs) decomposes each decision's entropy into tempering, smoothing, roughening, selection and between-decision field terms; [`PathStats`](src/colony.rs) measures what that does to the paths. |
| Pheromonally styled | Five [`Pheromone`](src/pheromone.rs) channels with literature half-lives, diffusion, saturation and a saturating perception; four [`Species`](src/species.rs) profiles. |
| Emergent behaviour | Trail formation, shortest-path selection, choice of the richer source, hunger-driven foraging and division of labour all arise from individual rules; the [`experiments`](src/experiments.rs) module reproduces the published setups. |

A learner never sees the hierarchy. It gets a [`LeverView`](src/learner/mod.rs):
the parameter vector at the far end of its lever, the turn number, and the
reward from last time. A lever is a dial, not a memory: every learner moves the
surface from where it finds it with bounded, revertible displacements.

## Quick start

```rust
use ant_simulator::prelude::*;

// A hungry colony of Lasius niger on a random map, one simulated hour.
let mut cfg = SimConfig::default();
cfg.nest.initial_satiation = 0.1;
let mut sim = Simulation::new(cfg, 42);
sim.run_seconds(3600.0);
println!("{}", render(&sim));
let s = sim.stats();
println!("delivered {} loads; {:.0}% of ant-time outside; division of labour {:.2}",
    s.food_delivered, 100.0 * s.foraging_fraction(), sim.division_of_labor());

// Another species, and the colony-wide temperature dial.
let mut cfg = SimConfig::for_species(Species::argentine());
cfg.nest.initial_satiation = 0.1;
let mut sim = Simulation::new(cfg, 7);
sim.hierarchy_mut().node_mut(0).surface.entropy = EntropyControl::fixed(0.5);
sim.run_seconds(1800.0);

// The double bridge, six replicates.
let outcomes = run_double_bridge(&BridgeSpec::ratio_two(), &Species::argentine(),
    80, 1800.0, 600.0, &[1, 2, 3, 4, 5, 6]);
let summary = summarize(&outcomes.iter().map(|o| o.short_fraction).collect::<Vec<_>>());
println!("short branch carries {:.0}% of traffic", 100.0 * summary.mean);
```

```rust
use ant_simulator::prelude::*;

// Several learners, hidden levers over every node, rotated with period 4.
let learners: Vec<Box<dyn Learner>> = vec![
    Box::new(PhaseAware::new(HillClimber::new(0.2), 8)),
    Box::new(HillClimber::new(0.2)),
    Box::new(PhaseAware::new(DialBandit::entropy(), 8)),
    Box::new(PhaseAware::new(DialBandit::geometry(), 8)),
    Box::new(PolicyGradient::new(0.02)),
];
let mut sim = SimConfig::default();
sim.trace = true;
sim.nest.initial_satiation = 0.1;
let config = ArenaConfig {
    sim,
    steps_per_turn: 600,
    schedule: RotationSchedule::RandomStatic { period: 4 },
    targets: ControlTargets::AllNodes,
    feedback: FeedbackScope::Subtree,
    ..ArenaConfig::default()
};
let mut arena = Arena::new(config, learners, 7).unwrap();
let report = arena.run(100);
println!("{report}");
```

## The biological model

Units are physical: cells of `cell_cm` centimetres (2 by default), ticks of
`tick_s` seconds (1 by default); species parameters are in centimetres,
seconds and pheromone "marks", and are converted on the way in.

**Pheromones.** Every cell carries five channels: the recruitment *trail*,
an optional outbound *home* trail (for bidirectional-trail models), colony
*territory* marking (Devigne & Detrain 2002), the Pharaoh ant's repellent
*no entry* marking (Robinson et al. 2005), and volatile *alarm* released at a
worker's death. Each channel decays by first-order kinetics from its
half-life, diffuses conservatively to orthogonal neighbours, and saturates
on the substrate. Perception is `ln(1 + C/k)`; a weight `n` on that feature
makes the movement softmax exactly Deneubourg's choice function
`(k + C₁)ⁿ / Σ (k + Cⱼ)ⁿ` with `k = 20` marks and `n = 2` (Deneubourg, Aron,
Goss & Pasteels 1990), extended from two branches to eight directions.

**Senses and movement.** An ant scores its eight neighbouring cells on
twelve features (the five channels through an antennal sweep two cells
ahead, food, nest, heading persistence, alignment with its path-integrated
home vector, alignment with a remembered site, recent visits, crowding),
with a separate weight block for outbound and inbound movement. Speed is a
species value in cm/s, lower when loaded and higher on a strong trail.

**Foraging.** Feeding time and crop load rise with sugar concentration
(Josens, Farina & Roces 1998). On the way home a forager lays trail with a
probability and an intensity that rise with quality (Beckers, Deneubourg &
Goss 1993). It navigates by path integration with odometric and heading
noise (Müller & Wehner 1988), searches around the fictive location when its
estimate runs out (Wehner & Srinivasan 1981), remembers a rewarding site and
returns to it, but abandons poor sites with a quality-dependent probability
(Mailleux, Deneubourg & Detrain 2000). Unsuccessful trips are given up after
a species-specific time; Pharaoh's ants then mark the route as unrewarding.

**Colony.** Loads are handed over by trophallaxis, slower in a satiated
nest. The store's hunger and the excitation left by returning foragers make
up the foraging stimulus; every worker engages by a response threshold
`sⁿ / (sⁿ + θⁿ)` (Bonabeau, Theraulaz & Deneubourg 1996) drawn from a broad
log-normal distribution, high in young workers (temporal polyethism), and
reinforced while a task is performed (Theraulaz, Bonabeau & Deneubourg 1998).
Nursing competes for the same workers, with a stimulus that falls as nurses
are recruited. Inside workers consume the store; a queen lays eggs when the
colony is fed; brood must be fed by nurses and emerges after its development
time. Foragers face a predation hazard and starve without food.

**Species** ([`Species`](src/species.rs)): *Lasius niger* (default; 1.5 cm/s,
47-minute trail half-life, inbound trail laying), *Linepithema humile*
(lays trail both ways, faster-decaying trail, the double-bridge species),
*Monomorium pharaonis* (adds the no-entry marking), and *Cataglyphis* (no
trail at all; fast, path-integrating solitary foragers). `Species::compressed`
shortens life-history clocks (maturation, development, egg laying) for
demonstrations without touching behavioural clocks.

## Validation against the classic experiments

`cargo run --release --example experiments` runs each setup in replicate;
`tests/experiments.rs` asserts the same outcomes at smaller size.

| Experiment | Published finding | Here (80 ants, 30 min, 6 replicates) |
| --- | --- | --- |
| Double bridge, long branch 2× short (Goss et al. 1989; Beckers et al. 1992) | traffic concentrates on the short branch | short branch carries 99–100% of late traffic in every run, for both the Argentine profile and *Lasius* |
| Two sources at equal distance, 1.0 vs 0.1 quality (Beckers et al. 1990) | the colony focuses on the richer source | 81% ± 1% of loads from the rich source, majority in every run |
| Colony satiation 0.05 → 1.0 (Mailleux et al. 2003) | hungrier colonies forage more | ant-time outside falls monotonically from 78% to 3% |
| Threshold reinforcement (Theraulaz et al. 1998) | specialisation | division-of-labour index 0.44 with reinforcement, 0.27 without |
| Equal branches (Deneubourg et al. 1990) | symmetry breaking onto one branch | see below |

Symmetry breaking on equal branches is a marginal instability. With the
model's assumptions matched (`pure_pheromone_feedback`: a trail that does
not evaporate on the experiment's timescale and no behavioural negative
feedback at junctions) one branch wins in a third to three quarters of
45-minute runs, depending on colony size. With the full behavioural model the
split stays near one half: a modest crowding term and the memory that stops
ants dithering both act as negative feedback at the junction, and either is
enough to hold the symmetric state. The crowding effect is itself documented
(Dussutour, Fourcassié, Helbing & Deneubourg 2004). The crate exposes both
regimes rather than hiding the sensitivity.

## Path surfaces: separable entropy, deformation, and geometric selection

Every decision runs through a short geometric pipeline:

1. **Deterministic information.** The effective surface scores the
   walkable directions. Laid out as a ring around the heading this is the
   landscape in front of the ant.
2. **Deformation with the entropy budget** `h`: *smoothing* blurs the
   landscape along the ring (angular disorder, path-coherent); *roughening*
   adds a random field of a few low Fourier modes (landscape disorder,
   path-incoherent); *tempering* then either divides by a fixed temperature
   or solves the temperature for an exact entropy target. The shares are per
   node (`Deformation { smooth, rough, reach }`) and compose down the
   hierarchy like the weights.
3. **Selection.** `Selection::Softmax` draws globally. `Selection::Sucker`
   crawls the ring from straight ahead for a bounded reach and goes where it
   can get to.

`cargo run --release --example surface` shows one ant's surface scrolling
past and then holds an entropy budget of 0.35 while changing its shape
(20 simulated minutes, mean of 2 seeds):

```
setting           target  drawn | temper smooth  rough  field select | turnH  str%  eff | deliv
flat / softmax     0.704  0.704 |  0.704 +0.000 +0.000 +0.000 +0.000 | 1.101  63.6 0.96 |   105
smooth / softmax   0.716  0.716 |  0.634 +0.082 +0.000 +0.000 +0.000 | 1.123  62.3 0.97 |   137
rough / softmax    0.723  0.723 |  1.014 +0.000 -0.291 +0.604 +0.000 | 1.789  29.4 0.69 |    71
reach 1 / sucker   0.744  0.575 |  0.744 +0.000 +0.000 +0.000 -0.168 | 0.874  70.6 0.83 |   134
```

Same per-decision entropy, different paths: roughening looks sharper inside
each decision but injects 0.6 nats *between* decisions (the `field` column),
and the paths turn twice as much and deliver a third less. A short sucker
reach binds the ant to its heading; a long one converges to the global draw.

## Examples

```
cargo run --release --example colony      [minutes]
cargo run --release --example experiments [replicates] [minutes]
cargo run --release --example arena       [turns] [period] [sucker]
cargo run --release --example surface     [ticks] [seeds]
```

`colony` renders the world, compares the four species on one map, sweeps
the colony-wide temperature, and heats one caste. `experiments` replicates
the classic setups. `arena` runs learners over the hierarchy with a hidden
rotation and evaluates the untouched instinct against the learned hierarchy
on fresh episodes. `surface` explores the geometric entropy channels.

## The arena, turn by turn

1. The rotation says which node each lever reaches this turn.
2. Each connected learner sees that node's parameters and writes new ones.
3. The colony runs for `steps_per_turn` ticks (a fresh episode, or one
   persistent colony).
4. Each connected learner receives its reward (colony-wide or subtree) and,
   with tracing on, the REINFORCE score for the node it held, and may leave
   different parameters behind before the lever rotates away.

Schedules: `Fixed`, `Cyclic`, `RandomStatic { period }`, `Fresh` (a new
permutation every turn, unlearnable by construction), and `Explicit`.

## Learners

| Learner | What it does with the lever |
| --- | --- |
| `HillClimber` | (1+1) evolution strategy on displacements with adaptive step size; feels out a surface it does not recognise before probing. `Mutation::Rotation` rotates the weight vector in a random plane. |
| `CrossEntropy` | Probes Gaussian displacements, applies the mean of the elite displacements once per generation. |
| `PolicyGradient` | REINFORCE with a normalised advantage and unit gradient direction. |
| `DialBandit` | UCB1 over dial settings only. `DialBandit::entropy()` (alias `EntropyBandit`) searches the dial; `DialBandit::geometry()` searches smoothing, roughening and reach. |
| `PhaseAware<L>` | Infers the rotation period (BIC over candidate periods, from rewards and parameter fingerprints) and runs one `L` per phase. |
| `StaticLearner`, `RandomLearner` | Controls. |

## Evaluating what was learned

`Arena::evaluate(params, episodes, seed)` runs fresh episodes with any
parameter vector, so the instinct (`Arena::initial_params`), the current
hierarchy, and the best turn (`Arena::best`) can be compared on identical
seeds.

## What is simplified

A two-dimensional grid with eight-neighbour moves; one crop load as the unit
of food; no vision or landmarks (navigation is path integration, marking and
trails); brood as a single developing stage; no queen pheromone, nest
architecture or temperature; and parameters that are representative values
from the cited studies rather than fits to any one dataset.

## Reproducibility and tests

Everything is driven by a single `u64` seed through the crate's own xoshiro
generator, so simulations, experiments, arenas and learners replay exactly.

```
cargo test
cargo clippy --all-targets
```
