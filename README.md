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
| Pheromonally styled | Six [`Pheromone`](src/pheromone.rs) channels (five signals and the smell of food) with literature half-lives, temperature-dependent evaporation, saturation and a saturating perception through a forward antennal probe; four [`Species`](src/species.rs) profiles. |
| Emergent behaviour | Trail formation, shortest-path selection, symmetry breaking, traffic sharing under crowding, choice of the richer or the more productive source, hunger-driven and larva-driven foraging, foraging that winds down as crops fill, division of labour and cemetery formation all arise from individual rules; the [`experiments`](src/experiments.rs) module reproduces the published setups. |

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
the grid only carries the substrate (terrain, food, marks, corpses,
landmarks).

**Pheromones and smells.** Every cell carries six channels: the
recruitment *trail*, an optional outbound *home* trail (for
bidirectional-trail models), colony *territory* marking (Devigne & Detrain
2002), the Pharaoh ant's repellent *no entry* marking (Robinson et al.
2005), volatile *alarm* released at a worker's death, and the *smell of
food*, given off weakly by sugar solution and strongly by prey, spreading
through the air and gone within minutes (Buehlmann, Graham, Hansson &
Knaden 2014: ants locate food by its odour). Each channel decays by
first-order kinetics from its half-life, scaled by a Q10 with temperature,
spreads conservatively to orthogonal neighbours (the trail hardly at all: it
is a substrate deposit), and saturates on the substrate. Marks are laid per
centimetre walked, fewer on a crowded patch (Czaczkes, Grüter & Ratnieks
2013). Perception is `ln(1 + C/k)`; a weight `n` on that feature makes the
movement softmax exactly Deneubourg's choice function `(k + C₁)ⁿ / Σ (k +
Cⱼ)ⁿ` with `k = 20` marks and `n = 2` (Deneubourg, Aron, Goss & Pasteels
1990), extended from two branches to sixteen headings.

**Senses and movement.** An ant scores sixteen headings on fifteen
features, with a separate weight block for outbound and inbound movement:
the six channels read patch by patch along an antennal probe one and two
cells ahead that stops at walls (nothing is sensed behind, so a strong trail
behind never pulls an ant round and U-turns arise from losing the trail
ahead, as in Beckers, Deneubourg & Goss 1992); food and nest ahead; a
quadratic turning cost relative to the direction of travel, which is a
running mean of recent steps over a species persistence length; alignment
with the path-integrated home vector, with a remembered site and with a
remembered route; recent visits; crowding ahead; a wall ahead. Speed is a
species value in cm/s, scaled by body mass to the 0.3, linear in temperature
above a species minimum, lower when loaded and in a crowd, and higher on a
strong trail. Movement is by sub-steps of at most one cell; a body of finite
width keeps a clearance from walls and cannot cut corners. Cells have a
capacity (a narrow bridge holds fewer ants): beyond it ants slow down, lay
less, and steer away, which is what pushes a crowded colony onto both
branches of a bridge (Dussutour, Fourcassié, Helbing & Deneubourg 2004).

**Navigation.** An ant navigates by path integration with odometric and
heading noise (Müller & Wehner 1988) and searches around the fictive
location when its estimate runs out (Wehner & Srinivasan 1981). It learns
one-way routes: at each familiar place, the direction it walked from there
on the way home and on the way out (Collett, Collett, Bisch & Wehner 1998;
Wehner et al. 2006), together with the path-integration estimate it had
there, which recalibrates the integrator on recognition. Worlds can carry
landmarks that ants see from a species distance: on leaving the nest and on
finding food an ant takes a view (which landmark, and where it stands
relative to the place), and when that landmark comes into sight again it
fixes its position from it (Wehner & Räber 1979; Collett 1992).

**Foraging.** Food is a volume of sucrose solution of some molarity, or a
heap of protein prey. Intake rate falls with concentration (viscosity) and
crop load rises with quality (Josens, Farina & Roces 1998); prey is cut
over a handling time and carried in the mandibles. A source may refill at
a rate (a drop fed by a syringe, an aphid colony); a forager waits at a
slow drip for a species patience and leaves with what it has. On the way
home it lays trail with a probability that rises with quality (Beckers,
Deneubourg & Goss 1993) and with how full its crop is (Mailleux, Deneubourg
& Detrain 2000), at an intensity that rises with quality. It remembers a
rewarding site and returns to it, but abandons poor or meagre sites with a
quality-dependent probability (Mailleux, Deneubourg & Detrain 2000).
Unsuccessful trips are given up after a species-specific time; Pharaoh's
ants then mark the route as unrewarding.

**Crops and trophallaxis.** Every worker carries its own crop of sugar and
lives off it; metabolism, scaled by body mass to the 0.75 and by
temperature, drains it, and the starvation clock runs only while it is
empty. A returning forager unloads by contacts every few seconds: each
receiver takes a share of its own empty crop space (Greenwald, Baltiansky &
Feinerman 2018), and the food carries the recruitment signal, so a contact
that passes little excites little. The forager stops at a residual or gives
up after too many contacts and keeps its load. Inside the nest, crops even
out by pairwise sharing between nestmates and with a reserve standing for
the queen, brood and nestmates not simulated (Buffin et al. 2009; Greenwald,
Segre & Feinerman 2015). Prey goes to a protein store.

**Colony.** Departure is gated by the worker's own crop, amplified by the
hunger it reads off its nestmates: a forager that could not unload stays
in, which is how foraging winds down as the colony fills (Greenwald et al.
2018; Mailleux, Detrain & Deneubourg 2006), while a starving individual
goes out even among replete nestmates (Mailleux et al. 2011). Hunger alone
sends out a trickle of scouts; the foraging force builds up sigmoidally
through the excitation spread by returning foragers, and a known source
counts only while the colony can still take food. Each trip is a sugar trip
or a protein trip, decided on leaving the nest from the colony's protein
demand: workers alone rarely take prey, a colony with larvae turns to it
(Dussutour & Simpson 2009, 2012). Every worker engages by a response
threshold `sⁿ / (sⁿ + θⁿ)` (Bonabeau, Theraulaz & Deneubourg 1996) drawn
from a broad log-normal distribution, high in young workers (temporal
polyethism), and reinforced while a task is performed (Theraulaz, Bonabeau
& Deneubourg 1998); nutritional state adds to the specialisation, as the
fuller workers stay in (Robinson et al. 2009). Nursing competes for the
same workers, with a stimulus that falls as nurses are recruited; nurses
feed larvae from their own crops, larvae need both sugar and protein to
pupate and starve if unfed. A queen lays eggs when the colony is fed; brood
passes through egg, larva and pupa with Q10-scaled development, and
workers emerge from pupae. Foragers face a predation hazard, a heat hazard
that rises exponentially towards the species' critical thermal maximum
(Cerdá, Retana & Cros 1998), and starve without food. The dead leave
corpses: undertakers carry those inside the nest out past a refuse
distance, and any explorer drops a corpse where corpses lie and picks one
up the less readily the bigger the pile it lies in (Deneubourg et al. 1991),
so cemeteries form (Theraulaz et al. 2002). The environment has a
temperature, optionally with a diurnal cycle, and each species forages
only within its thermal window.

**Species** ([`Species`](src/species.rs)): *Lasius niger* (default; 1.5 cm/s,
47-minute trail half-life, lays on the way home and on the way out to a
known source, reads the trail both ways, and learns routes that can
override it: Grüter, Czaczkes & Ratnieks 2011), *Linepithema humile* (lays
while exploring too and relies on the trail far more than on memory: Aron
et al. 1993; the double-bridge species), *Monomorium pharaonis* (adds the
no-entry marking), and *Cataglyphis* (no trail at all; fast, hot-habitat,
size-polymorphic, path-integrating solitary foragers with strong route
memory, long sight and a thermal limit near 56 °C that they forage right
up to). Workers vary in body mass around the species' typical worker.
`SimConfig::for_species` holds the world at the species' reference
temperature. `Species::compressed` shortens life-history clocks
(maturation, development, egg laying) for demonstrations without touching
behavioural clocks.

## Validation against the classic experiments

`cargo run --release --example experiments` runs each setup in replicate;
`tests/experiments.rs` asserts the same outcomes at smaller size, and the
unit tests in `src/colony.rs` cover the trophallaxis and heat mechanisms.

| Experiment | Published finding | Here (80 ants unless stated, 30 min, 4 replicates) |
| --- | --- | --- |
| Double bridge, long branch 2× short (Goss et al. 1989; Beckers et al. 1992) | traffic concentrates on the short branch | short branch carries 97% ± 1% of late traffic (Argentine) and 84% ± 17% (*Lasius*), majority in every run |
| Equal branches (Deneubourg et al. 1990) | most colonies end up with one branch carrying over 80% | after 60 min the favoured branch carries 69% ± 31% with the full model and 70% ± 30% under the 1990 model's assumptions; one branch exceeds 80% in all and half of the runs |
| Crowding on a narrow bridge (Dussutour et al. 2004) | one trail at low density; at high density on a narrow bridge both branches, without loss of throughput | 400 ants: mean deviation from an even split 0.22 on the wide bridge, 0.04 on the narrow one, at 219 and 183 crossings/min; 80 ants: 0.27 and 0.09 |
| Two sources at equal distance, 1.0 vs 0.1 M (Beckers et al. 1990) | the colony focuses on the richer source | 94% ± 1% of the solution taken from the rich source, majority in every run |
| A dripping source, 0.02 to 10 µl/min (Mailleux et al. 2003) | foraging effort and recruitment match the source's productivity | ants at the source 6.0 → 20.9, mean load 0.06 → 0.41 µl, recruiting returns 11% → 79% |
| A hidden pool, with and without its smell (Buehlmann et al. 2014) | ants find food by its odour | first find after 101 s without odour, 42 s with |
| Colony satiation 0.05 → 1.0 (Mailleux et al. 2006) | starved colonies forage and recruit more | ant-time outside falls monotonically from 47% to 0% |
| Unloading as the colony fills (Greenwald et al. 2018) | receivers take less as their crops fill; foragers that cannot unload stop | unit test: contacts per return rise, foraging falls by more than half, no hungry worker left inside; crop loads even out by sharing |
| Sugar or prey, with and without larvae (Dussutour & Simpson 2009) | larvae turn the colony to protein | protein share of what is collected 0.08 without larvae, 0.47 with |
| Desert ants against the heat (Cerdá et al. 1998) | mortality rises steeply towards the thermal limit while speed still rises | 60 *Cataglyphis*, 30 min: at 40 °C speed ×1.25, 267 loads, nobody killed; at 52 °C ×1.85, 254 loads, 14 killed; at 54 °C ×1.95, 98 loads, 26 killed; at 55 °C the colony stays in |
| 300 scattered corpses, 60 workers (Theraulaz et al. 2002) | corpses are gathered into a few piles | piles fall from about 120 to 25 to 33 within the hour; the largest grows two- to threefold |
| Threshold reinforcement (Theraulaz et al. 1998) | specialisation | division-of-labour index 0.77 with reinforcement, 0.53 without |

Symmetry breaking on equal branches is a marginal instability, and getting
it out of an individual-based model says a good deal about what real ants
must be doing. Any symmetric leak into the two branch trails weakens it:
lateral diffusion of the trail, ants dithering at the fork and marking both
entrances, a fork choice biased by the wobble of the last step, or trail
sensed round a corner. Private route memory can kill it outright, by
splitting the colony into individuals each faithful to their own first
choice; Aron, Beckers, Deneubourg & Pasteels (1993) found precisely that
Argentine ants orient by the trail while *Lasius niger* relies on memory,
and the two profiles differ accordingly. Since the foraging force builds up
by recruitment and every forager's departures are gated by its own crop,
the decision is made by the first cohorts and takes longer to show in the
traffic. `pure_pheromone_feedback` reduces a configuration to the
assumptions of the 1990 model (no evaporation on the experiment's
timescale, no crowding, recency or odour terms, no route memory) for
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

## Hive cognitive geometry: the movement history as the queen's memory

The colony's movements, collapsed over time, are a structure the queen
can think through. `MovementHistory` keeps every move at two grains
(cells and sectors) and two time constants: a slow accumulator (one hour
half-life) that holds the *invariant* skeleton, where ants persistently
go, and a fast one (one epoch) whose departure from the slow one is the
*non-invariant* residual, where the colony's disorder lives. Each
accumulator is a `Flow`: how many moves, their mean direction (polar
alignment, high on one-way flow), their axial order (high on straight
two-way traffic), and their straightness (the mean cosine of each move's
turn from its mover's heading). A reading of the field is multi-scale:
the same six numbers over the whole field, over quadrants, and over
sectors. The skeleton's topology is counted as channels (eight-connected
groups of cells with enough steady, axially ordered flow) and loops (the
regions they enclose), and `render_skeleton` draws it cell by cell.

The `Queen` acts within time on a coarse clock. Each epoch she reads the
retrodictive field, forms a thought vector, and expresses it in the
hierarchy's entropy dials: a component `x` sets the root temperature to
`exp(expression × x)` (or a relative gain below the root). A `Readout`
(ridge regression on standardised features) retrodicts her past thoughts
from the present field, and `memory_capacity` scores it lag by lag on
held-out epochs for the invariant component, the residual, and both. With
`Recursion` on, what she recalls from the field feeds her next thought.

`cargo run --release --example hive` runs the memory probe: 100 workers
kept foraging by renewing sources and a steady drain on the reserve (the
brood and nestmates a simulated worker stands for), a 45-minute warm-up,
then 180 one-minute epochs in which the queen thinks a random ±1 and sets
the root temperature to `e^(2 × thought)`:

```
the dial in force: corr(thought, decision entropy) = 1.00
reaching the paths: corr(thought, residual straightness of the whole field) = -0.76

component    lag  1 lag  2 lag  3 lag  4 lag  5 lag  6   capacity
invariant      0.04   0.03  -0.02  -0.12  -0.17  -0.19   0.07
residual       0.70   0.38   0.08   0.00  -0.00  -0.02   1.17
both           0.73   0.42   0.09  -0.08  -0.12  -0.10   1.24
```

Her thoughts are embedded in the non-invariant output and nowhere else:
the residual retrodicts the thought of the epoch just ended with R² 0.70,
the one before with 0.38, the one before that with 0.08, while the
skeleton carries none of it. What carries the thought is the
tortuosity of the paths: a hotter colony turns more at every decision.
The loop then closes. Readouts fitted, external input off, the queen
recalls her last thought from the field and thinks its opposite (gain
−10) for 24 more epochs:

```
dial connected:    -+-+-+-+-+-+-+-+-+-+-+-+  alternations 23 of 23
dial disconnected: -+-++++++++-++++++++++++  alternations  5 of 23
```

With the dial connected the alternation is sustained through the colony
alone: nothing in her state remembers the last thought, only the ants'
paths do. With the dial disconnected (expression zero) she still recalls
and thinks, but nothing she thinks reaches the colony, and the field
recalls only its noise. Cognitive material laid down in the colony's
movement is thus included self-recursively in future cognition, at the
epoch scale and over two to three epochs back.

## Examples

```
cargo run --release --example colony      [minutes]
cargo run --release --example experiments [replicates] [minutes]
cargo run --release --example arena       [turns] [period] [sucker]
cargo run --release --example surface     [ticks] [seeds]
cargo run --release --example hive        [epochs] [epoch_seconds]
```

`colony` renders the world, compares the four species on one map, sweeps
the colony-wide temperature, and heats one caste. `experiments` replicates
the classic setups. `arena` runs learners over the hierarchy with a hidden
rotation and evaluates the untouched instinct against the learned hierarchy
on fresh episodes. `surface` explores the geometric entropy channels.
`hive` runs the memory probe and the closed loop of the queen's mind.

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

Space is a plane with a 2-cm substrate grid. Vision is limited to point
landmarks seen within a species distance; there are no panoramic views,
no walls that block sight, and no sun compass. The nest interior is a
well-mixed chamber: workers, brood, stores and corpses inside have no
positions, and trophallaxis meets nestmates at random. Trails are laid on
the substrate and sensed as patches within antennal reach; smells spread
by diffusion without wind, and the death cue of a corpse is not modelled
chemically. Prey is dead insects, not hunted. There is one colony: no
neighbours, fights or territory contests. Brood has three stages but no
cohort structure; there is no queen pheromone, no male or gyne production,
and a single worker caste per species (body mass varies continuously).
Temperature acts through Q10 factors, speed, an activity window and a
thermal limit, without humidity or microclimate. Parameters are
representative values from the cited studies rather than fits to any one
dataset. The queen's mind is a probe, not a model of a real queen: a
thought is a vector on a fixed clock, expressed through the entropy dials
and read back by linear readouts, and the colony it thinks through is
kept foraging by a reserve drain standing for nestmates that are not
simulated.

## Reproducibility and tests

Everything is driven by a single `u64` seed through the crate's own xoshiro
generator, so simulations, experiments, arenas and learners replay exactly.

```
cargo test
cargo clippy --all-targets
```
