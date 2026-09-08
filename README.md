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
| The path as a surface flat in front of you | [`Landscape`](src/landscape.rs): sixteen candidate headings, 22.5° apart, as a ring around the direction of travel, scored by the effective surface; recorded and rendered by `render_surface`. |
| Deformation over the deterministic information | The scores are the deterministic information; the entropy budget is spent through separable geometric channels: tempering, smoothing along the ring, a random roughening field ([`Deformation`](src/surface.rs)). |
| Geometric selection, like a sucker | [`Sucker`](src/landscape.rs): a walker that starts straight ahead and crawls the ring by local Metropolis moves for a bounded reach (`Selection::Sucker`). |
| Behaviour as separable entropy | The [`EntropyLedger`](src/landscape.rs) decomposes each decision's entropy into tempering, smoothing, roughening, selection and between-decision field terms; [`PathStats`](src/colony.rs) measures what that does to the paths. |
| Pheromonally styled | Five [`Pheromone`](src/pheromone.rs) channels with literature half-lives, temperature-dependent evaporation, saturation and a saturating perception through a forward antennal probe; four [`Species`](src/species.rs) profiles. |
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
seconds, microlitres, milligrams, degrees Celsius and pheromone "marks", and
are converted on the way in. Ants have a continuous position and heading;
the grid only carries the substrate (terrain, food, marks).

**Pheromones.** Every cell carries five channels: the recruitment *trail*,
an optional outbound *home* trail (for bidirectional-trail models), colony
*territory* marking (Devigne & Detrain 2002), the Pharaoh ant's repellent
*no entry* marking (Robinson et al. 2005), and volatile *alarm* released at a
worker's death. Each channel decays by first-order kinetics from its
half-life, scaled by a Q10 with temperature, spreads conservatively to
orthogonal neighbours (the trail hardly at all: it is a substrate deposit),
and saturates on the substrate. Marks are laid per centimetre walked.
Perception is `ln(1 + C/k)`; a weight `n` on that feature makes the movement
softmax exactly Deneubourg's choice function `(k + C₁)ⁿ / Σ (k + Cⱼ)ⁿ` with
`k = 20` marks and `n = 2` (Deneubourg, Aron, Goss & Pasteels 1990), extended
from two branches to sixteen headings.

**Senses and movement.** An ant scores sixteen headings on fourteen
features, with a separate weight block for outbound and inbound movement:
the five channels read patch by patch along an antennal probe one and two
cells ahead that stops at walls (nothing is sensed behind, so a strong trail
behind never pulls an ant round and U-turns arise from losing the trail
ahead, as in Beckers, Deneubourg & Goss 1992); food and nest ahead; a
quadratic turning cost relative to the direction of travel, which is a
running mean of recent steps over a species persistence length; alignment
with the path-integrated home vector, with a remembered site and with a
remembered route; recent visits; crowding; a wall ahead. Speed is a species
value in cm/s, linear in temperature above a species minimum, lower when
loaded and higher on a strong trail. Movement is by sub-steps of at most one
cell; a body of finite width keeps a clearance from walls and cannot cut
corners.

**Foraging.** Food is a volume of sucrose solution of some molarity; intake
rate falls with concentration (viscosity) and crop load rises with quality
(Josens, Farina & Roces 1998), and a load delivers its sugar in milligrams.
On the way home a forager lays trail with a probability and an intensity
that rise with quality (Beckers, Deneubourg & Goss 1993). It navigates by
path integration with odometric and heading noise (Müller & Wehner 1988),
searches around the fictive location when its estimate runs out (Wehner &
Srinivasan 1981), and learns local vectors at familiar places (route memory:
Collett & Collett 2002), which recalibrate the path integrator on
recognition and guide the next trip. It remembers a rewarding site and
returns to it, but abandons poor sites with a quality-dependent probability
(Mailleux, Deneubourg & Detrain 2000). Unsuccessful trips are given up after
a species-specific time; Pharaoh's ants then mark the route as unrewarding.

**Colony.** Loads are handed over by trophallaxis into a sugar store,
slower in a satiated nest. The store's hunger, the excitation spread by
returning foragers through nestmate contacts, and a known source (while the
colony can still take food) make up the foraging stimulus; every worker
engages by a response threshold `sⁿ / (sⁿ + θⁿ)` (Bonabeau, Theraulaz &
Deneubourg 1996) drawn from a broad log-normal distribution, high in young
workers (temporal polyethism), and reinforced while a task is performed
(Theraulaz, Bonabeau & Deneubourg 1998). Nursing competes for the same
workers, with a stimulus that falls as nurses are recruited. Inside workers
consume the store at a Q10-scaled metabolic rate; a queen lays eggs when the
colony is fed; brood passes through egg, larva and pupa with Q10-scaled
development, larvae must be fed by nurses and starve if they are not, and
workers emerge from pupae. Foragers face a predation hazard and starve
without food. The environment has a temperature, optionally with a diurnal
cycle, and each species forages only within its thermal window.

**Species** ([`Species`](src/species.rs)): *Lasius niger* (default; 1.5 cm/s,
47-minute trail half-life, inbound trail laying, route memory that
overrides the trail: Grüter, Czaczkes & Ratnieks 2011), *Linepithema
humile* (lays trail both ways and while exploring, reads it in both
directions, relies on it far more than on memory: Aron et al. 1993; the
double-bridge species), *Monomorium pharaonis* (adds the no-entry marking),
and *Cataglyphis* (no trail at all; fast, hot-habitat, path-integrating
solitary foragers with strong route memory). `SimConfig::for_species` holds
the world at the species' reference temperature. `Species::compressed`
shortens life-history clocks (maturation, development, egg laying) for
demonstrations without touching behavioural clocks.

## Validation against the classic experiments

`cargo run --release --example experiments` runs each setup in replicate;
`tests/experiments.rs` asserts the same outcomes at smaller size.

| Experiment | Published finding | Here (80 ants, 30 min, 8 replicates) |
| --- | --- | --- |
| Double bridge, long branch 2× short (Goss et al. 1989; Beckers et al. 1992) | traffic concentrates on the short branch | short branch carries 96% ± 1% of late traffic (Argentine) and 91% ± 2% (*Lasius*), majority in every run |
| Equal branches (Deneubourg et al. 1990) | most colonies end up with one branch carrying over 80% | one branch exceeds 80% within 45 min in half the runs with the full model, three quarters under the 1990 model's assumptions |
| Two sources at equal distance, 1.0 vs 0.1 M (Beckers et al. 1990) | the colony focuses on the richer source | 95% ± 1% of the solution taken from the rich source, majority in every run |
| Colony satiation 0.05 → 1.0 (Mailleux et al. 2003) | hungrier colonies forage more | ant-time outside falls monotonically from 73% to 0% |
| Threshold reinforcement (Theraulaz et al. 1998) | specialisation | division-of-labour index 0.54 with reinforcement, 0.32 without |

Symmetry breaking on equal branches is a marginal instability, and getting
it out of an individual-based model says a good deal about what real ants
must be doing. Any symmetric leak into the two branch trails weakens it:
lateral diffusion of the trail, ants dithering at the fork and marking both
entrances, a fork choice biased by the wobble of the last step, or trail
sensed round a corner. Private route memory kills it outright, by splitting
the colony into individuals each faithful to their own first choice; Aron,
Beckers, Deneubourg & Pasteels (1993) found precisely that Argentine ants
orient by the trail while *Lasius niger* relies on memory, and the two
profiles differ accordingly. `pure_pheromone_feedback` reduces a
configuration to the assumptions of the 1990 model (no evaporation on the
experiment's timescale, no crowding or recency terms, no route memory) for
comparison.

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
(20 simulated minutes, mean of 3 seeds):

```
setting           target  drawn | temper smooth  rough  field select | turnH  str%  eff | deliv
flat / softmax     1.025  1.025 |  1.025 +0.000 +0.000 +0.000 +0.000 | 1.578  44.8 0.97 |   199
smooth / softmax   1.025  1.025 |  0.930 +0.095 +0.000 +0.000 +0.000 | 1.505  46.3 0.97 |   202
rough / softmax    1.002  1.002 |  1.520 +0.000 -0.517 +0.847 +0.000 | 2.443  14.1 0.67 |   107
reach 1 / sucker   0.983  0.726 |  0.983 +0.000 +0.000 +0.000 -0.257 | 1.162  51.4 0.19 |    83
```

Same per-decision entropy, different paths: roughening looks sharper inside
each decision but injects 0.85 nats *between* decisions (the `field`
column), and the paths turn far more and deliver half as much. A short
sucker reach binds the ant to its heading (one ring step of 22.5° per
proposal, so it can barely turn for home); a long one converges to the
global draw.

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

Space is a plane with a 2-cm substrate grid: there is no vision, no
landmark recognition beyond place memory keyed to substrate cells, and no
nest architecture (the nest is a chamber). Trails are laid on the substrate
and sensed as patches within antennal reach; there is no airborne plume
model. Brood has three stages but no cohort structure; there is no queen
pheromone, no male or gyne production, and a single worker caste per
species. Temperature acts through Q10 factors, speed and an activity window,
without humidity or microclimate. Parameters are representative values from
the cited studies rather than fits to any one dataset.

## Reproducibility and tests

Everything is driven by a single `u64` seed through the crate's own xoshiro
generator, so simulations, experiments, arenas and learners replay exactly.

```
cargo test
cargo clippy --all-targets
```
