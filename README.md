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

**Nest interior.** The nest is a patch of cells whose depth runs from the
brood chamber at the centre to the entrance ring at the edge, and every
worker inside stands somewhere on it and walks between neighbouring cells
in short bouts. Each keeps to a spatial fidelity zone: a depth that is
near the brood in callow workers and drifts outward with age over the
maturation time, with an individual offset, so that nurses sit with the
brood and foragers by the entrance (Sendova-Franks & Franks 1995; Mersch,
Crespi & Keller 2013). Workers respond to the tasks they meet: the brood's
demand is felt fully in the chamber and fades with distance from it
(foraging for work, Tofts & Franks 1992), nurses feed larvae only while
standing in the chamber, and a worker that sets out to forage walks to the
entrance ring first. Trophallaxis needs contact: a returning forager
unloads to nestmates within reach near the entrance, and the receivers
hand the food inward by pairwise sharing with their own neighbours, so
food percolates from the entrance to the brood through the workers
between (Greenwald, Segre & Feinerman 2015). Corpses lie where workers
die; an undertaker walks to one, picks it up, and carries it out by the
entrance. The inside walks are part of the colony's movement history.

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

| Experiment | Published finding | Here (80 ants unless stated, 30 min, 8 replicates) |
| --- | --- | --- |
| Double bridge, long branch 2× short (Goss et al. 1989; Beckers et al. 1992) | traffic concentrates on the short branch | short branch carries 97% ± 2% of late traffic (Argentine) and 89% ± 6% (*Lasius*), majority in every run |
| Equal branches (Deneubourg et al. 1990) | most colonies end up with one branch carrying over 80% | after 60 min one branch carries over 80% in 4 of 8 runs with the full model and in 5 of 8 under the 1990 model's assumptions; the instability is marginal (see below) |
| Crowding on a narrow bridge (Dussutour et al. 2004) | one trail at low density; at high density on a narrow bridge both branches, without loss of throughput | 400 ants: mean deviation from an even split 0.15 on the wide bridge, 0.05 on the narrow one, at 212 and 178 crossings/min; 80 ants: 0.25 and 0.07 |
| Two sources at equal distance, 1.0 vs 0.1 M (Beckers et al. 1990) | the colony focuses on the richer source | 94% ± 1% of the solution taken from the rich source, majority in every run |
| A dripping source, 0.02 to 10 µl/min (Mailleux et al. 2003) | foraging effort and recruitment match the source's productivity | ants at the source 6.5 → 20.6, mean load 0.06 → 0.40 µl, recruiting returns 19% → 81% |
| A hidden pool, with and without its smell (Buehlmann et al. 2014) | ants find food by its odour | a party of 20 scouts, the pool 20 cm off: first find after 86 s without odour, 44 s with, over 8 seeds (far off, or with 40 scouts leaving at once, one walks into it by luck either way) |
| Colony satiation 0.05 → 1.0 (Mailleux et al. 2006) | starved colonies forage and recruit more | ant-time outside falls monotonically from 50% to 0% |
| Unloading as the colony fills (Greenwald et al. 2018) | receivers take less as their crops fill; foragers that cannot unload stop | unit test: contacts per return rise, foraging falls by more than half, no hungry worker left inside; crop loads even out by sharing |
| Sugar or prey, with and without larvae (Dussutour & Simpson 2009) | larvae turn the colony to protein | protein share of what is collected 0.04 without larvae, 0.46 with |
| Desert ants against the heat (Cerdá et al. 1998) | mortality rises steeply towards the thermal limit while speed still rises | 60 *Cataglyphis*, 30 min: at 40 °C speed ×1.25, 303 loads, nobody killed; at 52 °C ×1.85, 262 loads, 9 killed; at 54 °C ×1.95, 132 loads, 26 killed; at 55 °C the colony stays in |
| 300 scattered corpses, 60 workers (Theraulaz et al. 2002) | corpses are gathered into a few piles | piles fall from about 120 to 30 within the hour; the largest grows two- to threefold |
| Threshold reinforcement (Theraulaz et al. 1998) | specialisation | division-of-labour index 0.72 with reinforcement, 0.43 without |
| Spatial fidelity zones and social groups (Sendova-Franks & Franks 1995; Mersch et al. 2013) | young workers with the brood, old by the entrance; contacts mostly within a group | `nest` example, 80 workers in a 7 × 7 nest, 30 min: brood chamber 31 workers of mean age 2.2 d (24 nursing), between 29 of 2.8 d, entrance ring 8 of 4.4 d; age–depth correlation 0.35; no food changes hands directly between the chamber and the ring |
| Food dissemination (Greenwald et al. 2015) | foragers unload near the entrance and the food percolates inward | 2.0 mg handed on inside per milligram delivered; the chamber's crops as full as the entrance's (0.92 against 0.84) |
| Undertaking | corpses inside are carried out | 3 corpses placed in the brood chamber are fetched from where they lie and carried out within 30 min |
| Colony size at a single feeder (Beekman, Sumpter & Ratnieks 2001) | below a critical size a trail cannot be kept and foraging is disordered | trail alone: 12% of trips direct at 10 workers, 92% at 80, per-capita output ×1.6; with memory, 77–92% at any size (see the scale analysis) |

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
held-out epochs for the invariant component, the residual, both, and the
behavioural memo (below). With `Recursion` on, what she recalls from the
field feeds her next thought.

`cargo run --release --example hive` runs the memory probe: 100 workers
kept foraging by renewing sources and a steady drain on the reserve (the
brood and nestmates a simulated worker stands for), a 45-minute warm-up,
then 180 one-minute epochs in which the queen thinks a random ±1 and sets
the root temperature to `e^(2 × thought)`:

```
the dial in force: corr(thought, decision entropy) = 1.00
reaching the paths: corr(thought, residual straightness of the whole field) = -0.78

component    lag  1 lag  2 lag  3 lag  4 lag  5 lag  6   capacity
invariant      0.02  -0.02  -0.22  -0.18  -0.13  -0.07   0.02
residual       0.75   0.50   0.21   0.01  -0.02  -0.03   1.46
both           0.76   0.50   0.12  -0.04  -0.09  -0.03   1.38
memo           0.81   0.38   0.11  -0.10  -0.09  -0.07   1.30
```

Her thoughts are embedded in the non-invariant output: the residual
retrodicts the thought of the epoch just ended with R² 0.75, the one
before with 0.50 and the one before that with 0.21, while the invariant
skeleton, which is where the colony persistently goes, carries none of it.
What carries the thought is the tortuosity of the paths: a hotter colony
turns more at every decision. The behavioural memo's current signatures
(the turns taken, the legs, what was laid, node by node) read the last
epoch better still. The loop then closes. Readouts fitted, external input
off, the queen recalls her last thought from the field and thinks its
opposite (gain −10) for 24 more epochs:

```
dial connected:    -++-+-+-+-+-+-+-+-+--+-+  alternations 21 of 23, conviction 0.91
dial disconnected: -+++++++++++++++++++++++  alternations  1 of 23, conviction 0.91
```

With the dial connected the alternation is sustained through the colony
alone: nothing in her state remembers the last thought, only the ants'
paths do (the two misses are the epochs in which the field's noise
outweighed a thought one minute old). With the dial disconnected
(expression zero) she still recalls and thinks, but nothing she thinks
reaches the colony, and the field recalls only its noise. Cognitive
material laid down in the colony's movement is thus included
self-recursively in future cognition, at the epoch scale and over three
epochs back.

## Examples

```
cargo run --release --example colony      [minutes]
cargo run --release --example experiments [replicates] [minutes]
cargo run --release --example arena       [turns] [period] [sucker]
cargo run --release --example surface     [ticks] [seeds]
cargo run --release --example hive        [epochs] [epoch_seconds]
cargo run --release --example nest        [minutes]
cargo run --release --example scale       [minutes]
cargo run --release --example memo        [minutes] [categories]
cargo run --release --example lens
cargo run --release --example frame       [minutes]
cargo bench                               [-- quick | phases | colony | shapes]
```

`colony` renders the world, compares the four species on one map, sweeps
the colony-wide temperature, and heats one caste. `experiments` replicates
the classic setups. `arena` runs learners over the hierarchy with a hidden
rotation and evaluates the untouched instinct against the learned hierarchy
on fresh episodes. `surface` explores the geometric entropy channels.
`hive` runs the memory probe and the closed loop of the queen's mind.
`nest` shows the nest interior: zones by age, food handed inward, and
undertakers at work. `scale` runs the scale analysis: foraging
organisation against colony size and the hive's memory against colony
size and grain. `memo` extracts the behavioural memo and classifies the
ground. `lens` draws the two-position tessellation of the quadtree and
the geodesic over it. `frame` builds a formicarium (a slab nest, a tube,
an open box whose walls the ants climb to a shelf of food) and lets a
colony forage over its surfaces. `cargo bench` runs the throughput scan
(see below); `colony` selects its colony-scale rows, the memoized and
pipelined colony against the full simulation, and `shapes` its larger
and shaped arenas, the field's grain against the dense sweep.

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
flat patch whose depth is the distance from the centre, without chambers
or tunnels; the brood, the stores and the queen have no positions within
the brood chamber, and the reserve that stands for nestmates not simulated
is met anywhere inside. Trails are laid on
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
simulated. Memoized transits replay recorded outcomes: a replayed ant
crosses its node along the waypoints of another ant of its kind, laying
and moving for the history along them, learns no route there, and takes
that ant's outcome; the chemical field is kept at cell resolution only
where it has structure, and as one mean per node where it is faint; the
kinetics can be stepped every few ticks with the evaporation and
diffusion of the ticks skipped applied at once; and under the decision
pipeline an ant on invariant, straight ground holds its heading for a
few steps between decisions.

## Scale analysis

`cargo run --release --example scale` asks how the colony's behaviour
changes with its size, and how the hive's memory depends on size and
grain. Colonies of 10 to 640 workers forage at a single inexhaustible
feeder 40 cm beyond a 12 cm entrance corridor (the feeder of Beekman,
Sumpter & Ratnieks 2001), settle for 20 minutes and are measured over the
next 20. *Ordered* is the share of outbound legs that reached the food no
more than one and a half times the direct distance: guided rather than
searched. With the trail the only way to the feeder (no route or site
memory, no smell; *Monomorium pharaonis*):

```
  ants    loads  per ant/h  trail mid  ordered
    10       25       7.50       0.28      12%
    20       47       7.05       2.11      41%
    40      128       9.60       5.14      75%
    80      329      12.34      13.73      92%
   160      602      11.29      22.42      95%
   320      785       7.36      28.78      24%
   640      802       3.76      33.30       7%
```

This is Beekman's transition: below a few dozen workers the returning
foragers cannot lay trail faster than it evaporates (the mid-trail
concentration sits below the perception constant), each forager searches
for the feeder on its own, and one trip in eight goes there directly;
between 20 and 160 workers the trail takes hold and organises the
traffic, so that per-capita output rises by three quarters and nearly
every trip is direct. Beyond 320 workers the entrance corridor jams
(Dussutour et al. 2004), trips detour and per-capita output falls again.
With individual navigation on, the same species forages as well at any
size until the jam, and its output is proportional to its size rather
than cooperative:

```
  ants    loads  per ant/h  trail mid  ordered      (Lasius niger: loads  per ant/h  ordered)
    10       55      16.50       2.89      77%                       45      13.50      91%
    40      208      15.60      12.01      88%                      195      14.63      95%
    80      398      14.93      21.77      92%                      361      13.54      97%
   160      690      12.94      33.02      90%                      679      12.73      97%
   320     1016       9.53      38.76      86%                     1052       9.86      91%
   640     1272       5.96      45.90      11%                     1731       8.11      81%
```

Memory removes the size dependence that the trail alone has: a colony of
ten forages by memory more than twice as well as by trail, and gains
nothing from growing except a jam. Over 10 to 640 workers the colony's
output scales as N^0.91 by trail alone and as N^0.78 to N^0.88 by
memory, sublinear because of the corridor.

The hive's memory (the residual readout of the queen's thought, held-out
R² by lag; 120 epochs of a minute after a 45-minute warm-up) grows with
the colony and shrinks with the grain of the reading:

```
  ants  sector  multiscale   lag 1   lag 2   lag 3  capacity
    25      16         yes    0.72    0.36   -0.10      1.08
    50      16         yes    0.71    0.55    0.09      1.46
   100      16         yes    0.84    0.50   -0.04      1.34
   200      16         yes    0.85    0.66    0.47      1.98
   100       8         yes    0.72    0.60    0.05      1.37
   100       8          no    0.60    0.53    0.09      1.22
   100      16          no    0.82    0.46   -0.10      1.29
   100      32         yes    0.68    0.25    0.11      1.03
   100      32          no    0.66    0.24    0.09      0.99
```

More workers write the thought into more movement, so the field carries
it more faithfully and for longer (capacity 1.08 at 25 workers, 1.98 at
200). The grain cuts both ways: 32-cell sectors are too coarse to see the
paths turn (R² 0.68 at lag 1), 8-cell sectors hand the readout more
numbers than 80 epochs can fit (0.72), and 16-cell sectors read best
(0.84); the coarser grains above the sectors, the quadtree's upper
levels, add a little at every grain (0.04 to 0.12).

## Quadkeys, the behavioural memo and memoized transits

The world carries a quadtree: the root is the smallest power-of-two
square covering the grid, each level splits every node in four, and a
node's address is its *quadkey*, the level and the Z-order index of its
column and row (printed as digits from the root down, `0` north-west to
`3` south-east, as map tiles are). Anything whose value at a node is the
sum of its children's composes bottom-up, so a reading at any grain is a
lookup. The movement history lives on it, with the invariant and the
variant flow at every node from the cells to the whole field.

The **behavioural memo** (`Memo`) records, on the same tree, what the
ants do where: decisions and their entropy, the turns taken, the legs
(outbound, homing, searching), how often laden, cells walked, what was
laid, and what happened (food found, nest reached, search given up,
death). Every field is a sum, kept in a slow record (the invariant
character of the ground, an hour's half-life) and a fast one whose
departure from it is the variant part. `Simulation::extract_memo` hands
the memo out as an object of its own, composed to every grain, which can
be classified into kinds of ground (`Memo::classify`, k-means on the
standardised signatures, categories named by what marks them out) and
read as feature vectors (`Memo::features`). `cargo run --release
--example memo` does this for a hundred workers after half an hour, at
4-cell nodes:

```
cat nodes  character              decisions  entropy  straightness  outbound  homing  laden  marking
  0     8  busy, marking              1.323    1.468         0.847     0.272   0.560  0.573    0.357
  1    26  steady, searching          0.197    2.035         0.666     0.863   0.071  0.053    0.062
  2    22  homing, laden              0.190    1.789         0.703     0.448   0.521  0.477    0.205
  3   103  undecided, outbound        0.082    2.292         0.580     0.974   0.025  0.005    0.010
```

The eight busiest nodes are the two trails (straight, laden, marking,
decided); the homing approaches and the search ground fall out as their
own categories, and the map of category digits draws the trails through
the search ground. The memo is also learned against: the queen's epochs
carry its current features as a fourth component, and the memory probe
retrodicts her thought from it as well as from the residual flow (see the
hive table above).

**Memoized transits** are the cellular-automaton idea applied to
behaviour. Every transit of an ant through a node of *plain* ground (no
food, nest, prey or landmark in it) is filed under a key: the node, the
side entered by, the entry heading's class, the leg, whether laden, which
distinct policy the ant acts through, the entropy dial's class, and the
local field's class (the trail's strength, the crowding). What came of it
is the outcome: the side and point left by, the heading, the ticks, the
cells walked, the decisions and their entropy, what was laid, and the
waypoints passed (one every quarter of the node's side). A key's kernel
keeps a forgetting reservoir of outcomes. Once a kernel is mature (a
dozen outcomes, next to none of which ended inside the node) and the
node's current flow is within half of its invariant one, an ant entering
under that key is advanced in one step: it sits inside the node for the
outcome's ticks and appears at its exit with its path integrated, its
deposits laid along the waypoints it followed, and its decisions
counted, at a tick's cost of a comparison instead of one decision per
cell. A tenth of eligible entries are still simulated in full, so that
the kernels keep learning and a change in the ground shows; a change of
leg, a corpse on the ground or a node whose flow has departed from its
invariant turns memoization off there. Transits chain: an ant leaving
one memoized node into another is advanced again.

The kernels are kept at several levels of the tree at once, the memo's
grain and the levels above it (`TransitConfig::depth`, two by default:
4-, 8- and 16-cell nodes), and a transit is recorded at every level the
ant is crossing a node of. An entry is replayed at the coarsest level
whose node is plain and invariant and whose kernel is mature and
*coherent*: above the grain a kernel must leave by one side at one place
(`Kernel::coherence`, the share of outcomes leaving by the commonest
side discounted by the spread of where along it, 0.85 by default), so
that the bigger the step, the more definite the transition it stands in
for, as a macro-cell of a cellular automaton is memoized only where its
transition is a function. A coarser node still being recorded takes a
finer replay into its record, so the coarse kernels keep learning while
the fine ones are replayed. Over eight seeds of a 400-worker hour on a
128 × 128 world the hierarchy replayed a sixth more decisions through a
fifth fewer replays than the grain alone (92 thousand through 12.5
thousand, against 79 thousand through 15.5 thousand), delivering 4043
loads against 4064 in full and 4027 with the grain alone, at a decision
entropy of 1.64 against 1.54 and 1.61: the coarser the kernel, the more
its outcomes lag the trail as it strengthens.

`cargo bench -- colony` sets the approximated colony against the full
one at colony scale: 400, 1600 and 6400 workers for a simulated hour on
a 128 × 128 world, in full, then with the transits memoized (the
kernels learning as they go, so the replayed share is that of the whole
hour and higher by its end) and the decision pipeline (below) holding a
quarter of the steps, then with the field's kinetics stepped every four
ticks as well:

```
                       full simulation      |   memoized transits and the pipeline     |  and the field every 4 ticks
  ants   ticks/s  ns/ant  delivered  entropy | ticks/s  ns/ant  replayed  held  delivered  entropy | ticks/s  ns/ant  delivered  entropy
   400       821    3044       3603    1.520 |     924    2705       30%   24%       3967    1.778 |    1637    1528       3697    1.781
  1600       269    2323      15069    1.414 |     426    1468       37%   26%      15617    1.697 |     533    1173      15672    1.678
  6400        61    2546      52833    1.483 |      94    1656       36%   20%      56399    1.632 |      99    1575      56726    1.646
```

The approximated colony runs at 1.5 to 2 times the throughput of the
full one and forages as it does: over eight seeds of the 400-worker
hour the transits alone deliver within 1% of the full simulation and
the pipeline within 2% (single runs, as above, scatter by ±5% and lean
a few percent above the full one, since the kernels lag the trail as it
strengthens and a strided field evaporates in steps). The entropy per
decision rises because the decisions no longer made are the easy ones,
on straight invariant ground, while those still made are the doubtful
ones. On one core, 6400 workers live an hour in 36 seconds. What is
approximated: a replayed ant walks the waypoints of another ant of its
kind for the purposes of laying and of the history, learns no route
inside the node, and carries that ant's outcome; and a held ant walks
straight where it would most probably have walked straight. What
remains is the share of the steps still decided, three fifths at 6400
workers, and the bookkeeping of every step walked, which together are
nine tenths of the tick at that size.

## The lens: a two-position structure on the quadtree

Between two points the tree can be tessellated so that its resolution
follows the distance to the nearer of them: cells within a radius of
either end, and beyond it nodes that double in size with their distance
from the nearer end (`Lens`). The tessellation is the same whichever end
is named first, so anything computed on it is invariant under exchanging
the ends, and it has a number of leaves that grows with the logarithm of
the span rather than with the span, so that the two ends are joined
through a graph of a few hundred nodes on any grid. A node that holds
both walls and open ground is refined wherever it lies, so the geodesic
over the lens (`Lens::geodesic`, the least-cost path between the leaves'
centres, pulled straight where the line of sight is clear) respects the
walls exactly. `cargo run --release --example lens` draws it on a walled
field: the nest and a pool at cell resolution, the wall refined to cells
along its length with the open ground beside it in 2- and 4-cell leaves,
the corners of the field in 8- and 16-cell leaves, and the geodesic
through the wall's gap; on an open 512 × 512 world the lens between two
points 8 cells apart has 313 leaves and between points 256 cells apart
469, over 262144 cells. The simulation measures the directness of
outbound legs against this geodesic, so that in a walled arena an ant
that walked round the wall is not counted as having wandered.

## The decision pipeline: horizons, deadlines and a frame budget

Decisions can be coarse-grained in time as transits are in space. Under
the pipeline (`PipelineConfig`) an ant's decision sets its next: a hold
on the heading for a horizon of steps, and a deadline, up to a horizon's
slack later, by which the next decision must be made. The horizon is
one step where the flow through the node has departed from its
invariant or where the ant is searching; on invariant ground it is the
expected run of straight choices the decision itself would make,
`p / (1 − p)` for the probability `p` it gave to keeping the heading
(within one ring position, 22.5°, which the heading's persistence
smooths into the direction of travel), up to `max_horizon` steps
(four), so that a decision is skipped only where it would have come out
the same. A dial the queen runs hot flattens the distribution and
shortens the hold of itself: she orders the colony's decisions by where
she spends its entropy. A hold ends early when the ant's leg changes or
the step ahead is not clear. Every tick is a frame
with a budget of decisions: those at their deadline are made whatever
the budget, the rest of the budget goes to the pending ones, earliest
deadline first and, among equal deadlines, in the hierarchy's order (the
castes the queen put first are served first), and an ant not served
holds its heading a little longer, within its slack. The frames' ledger
(`Stats::frames`) counts the decisions made, the steps held and
deferred, the frames overrun and the mean horizon. In the colony-scale
bench above the pipeline holds a fifth to a quarter of the steps on top
of the transits' replays, and over eight seeds of a 400-worker hour it
costs 2% of the deliveries; a budget of fifty decisions a frame for
four hundred workers defers a further fifth of the steps within their
slack for the same cost.

## The frame: surfaces joined along their edges

An ant walks on surfaces. Its degrees of freedom are a position and a
heading on the surface it is on, and it walks over a fold onto the next
surface (a floor onto a wall, a wall onto the next round a corner) and
along a tube; an ant colony in a plastic frame lives on a set of such
surfaces. The `Frame` lays them out flat on one grid, as the net of a
box unfolds, so that everything above (the field on its two structures,
the memo, the transits, the pipeline) runs on them unchanged. Where two
surfaces meet in space but not on the net, a `Portal` joins their
edges: what lies beyond one edge is the other's cells, a body crossing
turns by the angle between the sides and turns the vectors it carries
with it (its path integration and its remembered site, as an ant
integrates in its body's frame over a fold), the field flows through in
both sweeps, and perception's probes read through it, so a trail is
followed round a corner. A `Slope` makes walking up a wall slower than
along it. Every region carries its place in space, so a point of the
net has a position and a height in three dimensions.

Gravity acts through the slopes. A body on a slope loses its grip with
the slope's `slip` chance per tick, more when laden, and falls: straight
down in space onto the highest level surface below it (the floor at the
foot of a wall, the floor beneath a lid), which on the net is a walk
down to where it lands, so its path integration takes the drop as such;
it lands facing any way and lies stunned for a few seconds, and
whatever transit it was in ends inside. A rough wall has no slip; a
band of fluon on the rim of a wall (`Frame::barrier`) has a slip of one,
so an ant that reaches it falls at once; a ceiling has a little.

`Frame::outworld` folds out an open box: a floor with four walls
attached along its sides on the net and joined at the corners by
portals, each wall a slope; `Frame::lid` closes it with a ceiling
folded out above the north wall and joined to the other three walls'
tops; `Frame::slab` lays a flat nest module; `Frame::tube` runs a strip
whose two long sides are joined round the back and whose ends are
portals to the edges it connects; `Frame::wall` cuts a hole. Walls
block sight: a landmark is seen only along a clear line, and the lens
sees through portals, so the geodesic directness of a trip through a
tube is measured along the tube. `cargo run --release --example frame`
builds a slab nest joined by a 32-cm tube through a hole at the foot of
the west wall of a 56 × 40 cm box with 16-cm walls, with a pool on the
floor and one on a shelf 9 cm up the east wall, and lets 150 workers
forage for forty minutes with the memo, memoized transits and the
pipeline on, with fluon on the top two rows of the walls, or with a lid
(`frame lid`):

```
fluon rim:  after 40 min: delivered 100, taken 69 µl from the floor pool and 17 µl from the shelf, 4677 falls, 118 outside, mean height 2.3 cm
            ants per region: floor 15, north wall 16, east wall 1, south wall 9, west wall 16, slab 32, tube 29
lid:        after 40 min: delivered 78, taken 53 µl from the floor pool and 27 µl from the shelf, 69 falls, 116 outside, mean height 7.3 cm
            ants per region: floor 26, north wall 8, east wall 11, south wall 14, west wall 8, slab 20, lid 22, tube 7
neither:    after 40 min: delivered 43, taken 45 µl from the floor pool and 30 µl from the shelf, 125 outside, mean height 13.2 cm
            ants per region: floor 3, north wall 32, east wall 17, south wall 34, west wall 26, slab 11, tube 2
```

The ants find the tube, come out on the floor, climb every wall and
reach the shelf; the trail runs through the tube, across the floor and
up the east wall. With neither fluon nor lid, searching ants turned
back by the rim of a wall walk along it, and the colony spends its
foragers up there; the fluon returns them to the floor (an ant a minute
or so falls off the rim, for they keep climbing to it, as ants do at
the barrier of a real box) and the colony delivers more than twice as
much; a lid gives them a ceiling to walk over, from which a laden ant
falls now and then. What is simplified: a surface is flat within a
region and folds only at the edges the builder joins (a cylinder is a
tube, a sphere has no net); a fall costs a stun and not an injury; and
path integration runs along the surface, where desert ants on hills are
known to integrate the ground distance instead (Wohlgemuth, Ronacher &
Wehner 2001), which on the walls of a box is the distance that matters
to a walker anyway.

## Performance and scaling

Every simulation profiles its tick phase by phase (`Simulation::profile`),
and `cargo bench` (a plain binary, so it runs on stable) sweeps colony
size and world area with a hungry colony kept foraging, keeping the
median of several runs and fitting the empirical exponent of the time per
tick against each dimension. On one core of the development machine, 100
*Lasius* workers on a 64 × 40 grid with the movement history and the queen
on run at 4.2 thousand ticks per second, 2.4 µs per ant-tick, divided as:

```
phase           µs/tick   share
decisions         131.3   55.3%
ants               33.4   14.1%
pheromones         69.8   29.4%
food                0.6    0.2%
nest                1.9    0.8%
history             0.0    0.0%
queen               0.3    0.1%
```

A movement decision (perceiving the ring of sixteen headings, scoring it,
tempering it, selecting) costs about 3 µs, and a walking ant makes one per
tick; the chemical kinetics cost under 30 ns per cell per tick where the
field is at cell resolution. Sweeping colony size on the same grid:

```
    ants   ticks/s  ant-ticks/s   ns/ant  decisions  pheromones
      25     11918       297945     3356        29%         62%
     100      4287       428681     2333        57%         30%
     400      1378       551123     1814        72%         11%
    1600       384       614108     1628        77%          3%
```

The decision phase is linear in the number of ants (exponent 1.03), the
kinetics do not depend on it (0.10), and the cost per ant-tick falls by
a third from 100 to 1600 workers as the kinetics are shared out.
Sweeping the area at 100 workers:

```
   cells   ticks/s  ant-ticks/s   ns/ant  decisions  pheromones  active
    1024      5761       576146     1736        61%         22%     88%
    4096      3582       358161     2792        51%         37%     52%
   16384      1815       181475     5510        39%         53%     34%
   65536       817        81733    12235        21%         73%     17%
```

The field is kept on two structures, over the grid tiled by the
quadtree's nodes at one level (8 cells across). A *substrate mark*
(trail, home, territory, no entry) lies where ants walked and is read
within antennal reach, so it is kept at cell resolution where it has
structure and as one mean per node where it is faint: a node is either
*active*, its cells carrying the field, or *coarse*. Mass moves between
cells, between a cell and a coarse node, and between coarse nodes by
the same conservative shares, so the field is exact on and around the
trails, where marks land and the front of the field reaches, and cheap
where it is faint: a node whose cells have all faded below a hundredth
of the perception constant is composed into its mean (fine to coarse),
and a coarse node that a deposit or the front reaches is refined into
cells again (coarse to fine). A *volatile* (the smell of food, alarm)
is volumetric information that flows through the air, not a mark on
the ground, and lives on the grain throughout: its sources emit into
the pools of their nodes, its mass moves between nodes, and a reader
sees the plane through a node's mean with the gradient of its
neighbours', never a cell; the scouts of the discovery experiment find
the hidden pool by its smell in 54 s against 86 s without, 153 against
250 s from twice as far and 297 against 536 s from three times as far,
as they did with the smell at cell resolution. On both structures, flat
coarse nodes merge into blocks of two, four and eight nodes across and
split again when the field reaches them, so a flat far field costs a
few blocks. *Active* above is the share of the nodes at which the trail
is at cell resolution: a sixth of a 65536-cell world, on which the
kinetics scale as area^0.76 and the tick as area^0.47, where the dense
sweep of the previous round ran this world at 527 ticks a second and
scaled as area^0.91. The food kinetics visit only the cells that carry
food, the movement history forgets lazily (a tick costs nothing), and
the nest keeps its counts, so none of those grows with the grid or the
colony. What remains is the decisions, which are the model, and the
bookkeeping of every step walked; at colony scale the memoized transits
and the pipeline stand in for half of the steps (the colony-scale table
in the memo section above).

`cargo bench -- shapes` asks what larger and shaped environments cost:
200 workers for ten minutes, the field on its grain against the dense
sweep of every cell, on squares up to 1024 × 1024, a strip, and arenas
cut into a 512 × 512 grid by walls (a disc, a ring, an L, a cross, and
sixteen rooms joined by corridors); *open* is the share of the grid
that is not wall, *kinetics* and *decide* are microseconds per tick,
*trail* the share of the nodes at which the trail is at cell
resolution:

```
  arena      cells  open |  dense/s  kinetics decide |  grain/s  kinetics  trail decide |  gain
 square    256x256  100% |      436      1727    444 |      889       547    23%    455 |  2.04
 square    512x512  100% |       93      9930    499 |      708       721     4%    442 |  7.64
 square  1024x1024  100% |       21     45120    381 |      386      1435     1%    376 | 18.24
  strip    1024x64  100% |      445      1616    516 |     1169       307    12%    439 |  2.63
   disc    512x512   69% |      101      8557    588 |      486       639     6%    622 |  4.82
   ring    512x512   55% |      103      8226    522 |      453       580     7%    560 |  4.42
      L    512x512   44% |      117      7749    481 |      934       392     6%    417 |  8.01
  cross    512x512   44% |      120      7596    408 |     1051       321     5%    376 |  8.77
  rooms    512x512   60% |      108      8126    562 |      699       455     4%    488 |  6.47
```

The gain grows with the world, because the colony's trails occupy the
same ground whatever the grid: the dense sweep costs the area, the
grain costs the structured ground and a few blocks for the rest, so a
1024 × 1024 world runs eighteen times faster and its kinetics fall from
45 to 1.4 ms a tick, with the trail at cell resolution on 1% of the
nodes. A shape costs its open ground and not its bounding box: the
nodes with no open cell are never visited, the nodes along a wall
coarsen like any other once faint, and the L and the cross, open on 44%
of their grid, run as a square of that area would. What the walls do
cost is perception, since an antennal probe stops at a wall and is
swept cell by cell wherever there is one, so the decisions of the disc
and the ring run a third dearer than in the open. The smell of food is
what made the sweep scale with the area before this: the bench scales
food clusters with the grid, so the smell spans the whole of it, and on
the substrate structure a third of the nodes were at cell resolution
for it; on its own structure it costs the nodes of its sources and the
blocks between them. Beyond 512 × 512 the movement history and the
behavioural memo are the next cost, since their quadtrees reach down
to the cells (the memo's two trees would take some 800 MB at
1024 × 1024); they are off in these rows, and keeping them from their
grain up is the natural next step.

## The extras crate: an architecture that thinks in the style of ants

The workspace carries a second crate, [`ant_extras`](extras/README.md),
that re-leverages every component above as a general-purpose
architecture. A *problem* is a set of states with places on a grid, an
origin, solutions with a quality and moves (free steps, or successor
states); a *mind* is a colony of thoughts walking it over the ants'
medium, governed by the ant's instinct through the hierarchy, deformed
and selected on the ring, collapsed into the movement history, stood
in for by memoized transits where the ground is plain, foreseen through
the lens, paced by the pipeline, and dialled by a queen who reads
stagnation and organisation; learners tune its surfaces through
rotating levers. The thoughts perceive a move into the ant's thirty
sensory slots, so the surfaces apply unchanged.

Three problems are embedded: a maze walked freely (on a grid or over a
frame's box), where the trail found round the wall is the path and
foresight shortens the trips by a quarter; a tour of sixteen cities,
where the trail laid home along the tour's edges finds tours 5 % better
than the nearest-neighbour tour (a mean of 0.98 and a best of 1.05 of
it, against 0.63 for the best of two thousand random tours); and a
graph colouring laid out layer by layer, where dead ends are marked
no-entry and retreated from and the trail selects the colourings with
fewer colours. The pheromone surface is made variable as well: a
path-topological network classes every route by its word (the rays it
crosses from the embedding's punctures, freely reduced, which is its
homotopy class), registers the classes walked home as symbols with
channels of their own and weights learned from their yields, labels any
route by its class (with a name given after the fact), generates a
route of any class by search or by expressing the symbol for the colony
to walk, and learns its punctures from the holes the walks enclose.
On that network stands a *holonomy embedding*: thoughts integrate a
high-dimensional state through the moves and portals they cross, the
transport of a class of routes is its vector, and a knowledge graph
walked as queries yields rules as words, relation vectors that close
along them, and predictions of held-out facts by geometry, rules and
both (`extras/docs/holonomy.md`). And the colony's forward and
backward filters are made explicit as a bridge on the grain, the
predicted corridor of successful trips, the surprise of the flows
against it, and a drift the thoughts can follow
(`extras/docs/bridge.md`). And the symbols become signs: a thought home
with a solution dances the sign of its route's class at the nest, a
thought setting out may draw a sign from the floor and enact its glyph,
a sign means where its trips end, synonyms are signs that lead to the
same place, and a sign is understood as far as those who heard it get
there; on a maze with a rich source and a poor one the sign tells all
there is about the outcome, every recruit is understood, and the colony
that talks brings home 46 % more (`extras/docs/lexicon.md`). And the
signs are persisted in walk form: echoes hold a sign for an epoch and
walk its glyph again and again, for the others to hear on the way, so
that the route enters the colony's invariant geometry early, the glyph
is refined by the walks, and thoughts born into the colony are
imprinted from the ground (`extras/docs/echo.md`). And the abstraction
is a ground: an interior is a second mind whose states are the words
the foragers registered, coupled to them both ways, believing through
its trail, speaking through the floor, proposing words nobody walked
for the foragers to test, and registering classes of its own that a
third layer can walk (`extras/docs/interior.md`). What the exploration
found, and where the ants' way stops, is in the crate's README.

```text
cargo run --release -p ant_extras --example think
cargo run --release -p ant_extras --example symbols
cargo run --release -p ant_extras --example relations
cargo run --release -p ant_extras --example bridge
cargo run --release -p ant_extras --example lexicon
cargo run --release -p ant_extras --example echo
cargo run --release -p ant_extras --example interior
cargo test --release -p ant_extras
```

## Reproducibility and tests

Everything is driven by a single `u64` seed through the crate's own xoshiro
generator, so simulations, experiments, arenas and learners replay exactly.

```
cargo test
cargo clippy --all-targets
cargo bench -- quick
```
