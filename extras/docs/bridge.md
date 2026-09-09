# The bridge

*The colony's forward and backward filters made explicit, with the
surprise, the drift, the sector weights and the survival rung.*

This is the design document for the third extension of `ant_extras`
(after the path-topological network and the holonomy embedding). It
states the claim, what was built, what was measured, and where it
stops.

## The claim

A thought's walk is a diffusion with a drift. The forward evolution of
the pheromone field alone is dissipative: diffusion is a contraction
semigroup. But the colony is not the unconditioned field. Condition
the walk on where its trips end, at food and back home, and the law of
the conditioned paths is a Schrödinger bridge. Its density is the
product of a function running forward from the source and a function
running backward from the outcome; its drift is the gradient of the
log of the backward function, Doob's h-transform; and it is symmetric
under time reversal. Schrödinger found it in 1931 looking for the
classical structure nearest to his equation.

The colony computes the pair implicitly. The outbound walks sample the
forward filter. The trail laid on the way home, in proportion to what
was found, accumulates the backward filter along the route just
walked: each place's trail is the colony's running estimate of how
much success starts there. In path-integral control, Kappen's form,
that backward function is the expected exponentiated cost to go, it
satisfies a linear backward equation, and the optimal drift is its
log-gradient; trail following is that drift. Each generation of trips
is a sweep of the iteration that solves the bridge, and the
temperature at which costs are exponentiated is the forcing pressure:
as it falls, the path measure collapses onto the shortest successful
corridor, the known answer.

"A thought closes autonomously" has a technical meaning in this
picture. An h-transformed process is Markov again: the future has been
folded into the drift, so the walker never reasons about it. The ant
does not plan; the trail has already done the planning backward.

The path measure also decomposes by topology. On a punctured surface
the partition function is a sum over homotopy classes, so the registry
of symbols is the sector decomposition of the colony's path measure,
and a symbol's share of the exponentiated-cost measure is its sector
partition function. In the quantum path integral the sectors carry a
representation of the fundamental group and interfere; here they carry
positive weights and add, which is the exact boundary between what a
pheromone can do and what "superposition" would add.

## What was built

* **The bridge** (`bridge.rs`). A coarse graph of blocks over the
  medium: the passable cells of each block, the width of every passage
  between neighbouring blocks (passable cell pairs across the side, and
  portal cell pairs, with the direction of each passage from each end).
  The passive kernel walks a block's passages in proportion to their
  widths. Every epoch the bridge solves the forward function (the
  discounted mass reaching each block from the nest) and the backward
  function (the discounted mass of the colony's own findings reachable
  from each block, the desirability) by relaxation to a tolerance, with
  a step costing `1 / temperature` in the exponent; their product,
  normalised, is the predicted density of successful trips. It keeps
  an exponentially weighted record of where thoughts out on trips are,
  and measures the **surprise**, the relative entropy of the observed
  flow against the predicted density (as `kl / (1 + kl)`); the
  **correlation** of the log of the trail's mean level with the log of
  the density and with the log of the desirability; the **entropy** of
  the density and the effective number of blocks it covers; and its
  **mass**. The **drift** at a point is the mean displacement out of
  its block under the kernel reweighted by the desirability of where
  each passage leads, so it runs only through passages, never into a
  wall.
* **In the mind.** `MindConfig::with_bridge`: thoughts out are counted
  every tick, findings are recorded as sinks, the bridge relaxes every
  epoch, the drift is added to a move's score with the configured gain
  (0 leaves the bridge an observer), the report carries the bridge's
  line, and the queen's stagnation is raised to the surprise where the
  surprise is larger, so the flows' departure from the colony's own
  model drives the scouts as stagnation does.
* **Sector weights** (`topos.rs`). With `SymbolConfig::temperature`, a
  symbol keeps the exponentially weighted mean of quality times
  `exp(-length / temperature)` over its trips; its share of the measure
  over the living classes is its support times that mean, normalised;
  its weight is the log of the share against an even share, clamped to
  the gain. Without a temperature the yield heuristic stands.
* **Survival** (`practice.rs`). With `PracticeConfig::survival`, a
  turn's reward is 1 where the quality brought home per thought reaches
  the threshold and 0 where not, so the learners are conditioned on the
  colony's survival, the Q-process, rather than on its mean yield; the
  report counts the turns survived.

## What was measured

`cargo run --release -p ant_extras --example bridge`, the maze round a
wall, 48 thoughts.

**The bridge as an observer, with a queen** (temperature 8, 4 000
ticks): the surprise settles near 0.38; the trail's correlation with
the predicted density rises from 0.21 to 0.37 as the trail organises,
while its correlation with the desirability alone stays near 0.1. The
trail is the colony's estimate of the *product* of the filters, since it
is laid along successful routes, not of the backward function alone.
The density covers 86 effective blocks of 160.

**The forcing pressure**: the same bridge relaxed at falling
temperatures.

| temperature | effective blocks | entropy (nats) |
|---|---|---|
| 64 | 146.7 | 4.99 |
| 16 | 110.4 | 4.70 |
| 8 | 85.9 | 4.45 |
| 4 | 66.2 | 4.19 |
| 1 | 43.5 | 3.77 |

**Following the drift** (4 000 ticks):

| drift gain | brought home | mean trip (cells) |
|---|---|---|
| 0 (trail alone) | 704 | 209 |
| 0.5 | 880 | 194 |
| 1 | 844 | 206 |
| 2 | 327 | 394 |
| 4 | 0 | 568 |

A small gain is worth a quarter more deliveries: the model's drift
breaks the ties the trail leaves. A large gain traps the thoughts
between blocks whose desirabilities point at each other, since the
field is piecewise constant at the block's resolution. The drift is a
bias, not a controller.

**Sector weights** (symbols with a temperature, 4 000 ticks): the
over-the-wall class's share rises from 0.883 at temperature 100 to
0.977 at 10, and the under class's falls from 0.117 to 0.023; the
weights follow, +1.7 to +2.0 against −3.0. Colder sharpens, as the
large-deviation principle says it must.

**Survival**: with a threshold of 6 per thought per turn, the rewards
are 0 and 1, and the colony survives 6 turns of 8.

## Where it stops

* The bridge is spatial: it stands over the medium, so it serves the
  problems walked freely. The relational build would want the same two
  filters over its own graph, forward from the head and backward from
  the answers, which is the rule reachability computed both ways.
* The passive kernel is isotropic over passages. The thoughts' own
  passive walk has persistence and reads walls ahead; a kernel fitted
  to it would predict better and surprise less.
* The stationary picture pins one source and one sink distribution.
  The full Schrödinger bridge pins marginals at every time and is
  solved by alternating rescalings; the epoch-by-epoch relaxation with
  running means is its stationary shadow.
* Positive weights add. Interference between sectors would need
  phases, and nothing in a pheromone carries one.
