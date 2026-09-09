# ant_extras: an architecture that thinks in the style of ants

An exploration: take every component of [`ant_simulator`](../README.md)
and re-leverage it as a general-purpose architecture. The ants' way of
solving their one problem (where the food is, and how to get it home)
has a fixed shape, and nothing in that shape is about food:

* an individual senses a little of its surroundings and scores a **ring
  of moves** with a **surface** of preferences;
* it draws from that ring with a set amount of **disorder** (the entropy
  dial, composed down a **hierarchy**);
* it walks, and on the way back **marks the ground** in proportion to
  what it found; the marks **evaporate**;
* the colony's movement collapses into an **invariant skeleton** and a
  residual, which a **queen** reads on a slow clock to turn the disorder
  up or down;
* where the ground is plain and the flow through it invariant, the walk
  is stood in for by a **memoized transit**; between two positions the
  route is foreseen through a **lens**; a decision that would have been
  the same is not made again (the **pipeline**).

This crate keeps the shape and changes the problem. A *problem* is a
set of states with *places* on a grid (the embedding), an origin (the
nest), solutions with a quality (the food), and moves: free steps over
the grid, or a set of successor states. A *mind* is a colony of
*thoughts* walking the problem over a *medium*, and what the thoughts
lay down on their way home is the solution.

```text
cargo run --release -p ant_extras --example think        # the three problems
cargo test --release -p ant_extras
```

## The pieces, and what they were

| Here | In the ants | What it does for thinking |
|---|---|---|
| `Problem` | the world's geometry, nest and food | states with places, an origin, moves, solutions with a quality, a scent |
| the medium (`World`) | the chemical field on two structures | trail laid home in proportion to quality, no-entry at dead ends, the smell of solutions on the grain, crowding |
| `sense` | the antennal sweep and the thirty sensory features | the same thirty slots, read along a move, so the ant's surfaces apply unchanged |
| `Thought` | the ant's body | state, place, heading, path integration, site fidelity, route memory, transit records, a horizon |
| surface and hierarchy | instinct, castes, ants | the ant's instinct at the root, scouts and followers with their dials, the queen's moods under them, a leaf per thought |
| path surfaces (`Landscape`, `Sucker`) | the ring of headings deformed and selected | the same ring, the moves placed on it by their direction in the embedding |
| `MovementHistory` | the hive's cognitive geometry | the invariant skeleton and residual of the thoughts' walks: the pipeline's ground, the queen's eyes |
| memo and transits | memoized transits | habits: the walk through plain, invariant ground stood in for by a kernel |
| the lens (`geodesic`) | the two-position geodesic | foresight: a plan followed point by point |
| pipeline (`horizon`) | horizons and the frame budget | a heading held where the decision would have been the same |
| `Queen` | the queen's mind | stagnation heats the scouts, organisation cools the followers |
| `Practice` | learners, levers, rotation, the arena | learners tune the surfaces through hidden, rotating levers, rewarded by what comes home |
| the frame | surfaces joined along edges | a maze over the folded-out surfaces of a box |

### The sensorium

The ant's surface weighs fifteen features of each candidate heading in
an outbound and an inbound block. A thought perceives a move into the
same slots (`sense::features`): the trail, home, territory, no-entry
and alarm marks read along the move as an antenna reads them (one cell
ahead at full weight, the second at half; a longer move, which the
thought commits to as a whole, is read the whole way); a solution one
move ahead as *food ahead*; the origin as *nest ahead*; the turn's
persistence; the alignments with the home vector, the remembered site
and the plan or remembered route; whether the cell was just crossed;
the crowding there; a wall ahead; and the smell of the goal, which is
the odour in the medium plus the problem's own *scent* of the move (a
heuristic in `0..=1`). So the ant's instinct, its hierarchy and its
queen govern thinking without a change of type.

### The trip

A thought leaves the origin *searching* (nothing to go on) or
*outbound* (to the site of its best finding, along the route it
remembers), decides at every state among the moves the problem offers,
placed on the ring by their direction, and at a solution turns
*inbound*. On free ground it walks home by path integration and the
inbound block of the surface; on a walk of states it retraces its
route. Either way it lays trail in proportion to the quality it
carries, sharpened by how the solution compares with the best it knows
and the best the mind has brought home (`recruitment`), so that a mind
that knows better lays little for worse. At a dead end it marks the
place no-entry, retreats a state (up to `retreats` times a trip) and
tries another way; over budget, it turns for home empty-handed. Home,
it rests, then goes out again. A finding releases the smell of a
solution into the medium, a volatile that spreads on the grain.

### The queen

Under each caste's dial sits a *mood* node, and the queen expresses
herself there so that her thought composes with the castes' standing
dials rather than replacing them. She thinks every epoch from two
things the history and the ledger show her: **stagnation**, how many
epochs since the yield (quality brought home per epoch) last peaked or
the best finding improved, which heats the scouts; and
**organisation**, the axial order of the invariant flow weighted by how
much flows where (two-way traffic on a trail scores high, wandering
low), which cools the followers.

## Three problems

### A maze, walked freely

`Maze::around_a_wall(64, 40)`: a start and a goal 55 cells apart in the
upper third of the grid, a wall between them with a gap above and a
gap below. This is the ants' own problem stated generally, and the
trail that forms is the path round the wall. 48 thoughts, 4 000 ticks:

| Mind | Brought home | Mean trip (cells), first and last thousand ticks |
|---|---|---|
| the instinct alone | 704 | 267 → 209 |
| with habits | 639 | 267 → 227 |
| with habits, the pipeline, foresight and a queen | 806 | 262 → 194 |

Foresight (the lens between the origin and the site, followed as a
plan) is what shortens the trips most: the site is behind the wall,
and a thought pulled straight at it piles against the wall until it
finds a gap, whereas a thought that foresees the geodesic walks round.
The habits replay thousands of transits standing in for a quarter of
the decisions (12 194 replays for 48 743 decisions in the last row),
and the queen ends with the scouts hot (×1.49) and the followers cool
(×0.70).

Two defaults differ from the ants'. A searching ant heats its
temperature 2.5-fold, and a quarter of a colony's thoughts as scouts at
that heat found the goal so rarely that half the trips were given up;
a mind searches at 1.5-fold with a tenth of its thoughts as scouts
(1 211 brought home in 6 000 ticks against 722). And the pipeline's
cone is best set to nothing for thinking: holding a heading that was
chosen within a ring position of straight ahead costs a quarter of the
deliveries on a trail that bends round a wall, holding only one that
was straight ahead costs nothing.

The same maze runs over a frame: `Maze::on_frame` takes a box folded
out with its corners joined by portals and its walls sloping, and the
thoughts walk from the floor up a wall through the fold (a test does).

### A tour of cities

`Tour::random(16, 64, 48, 7)`: sixteen cities in the plane. A state is
the city stood in, the cities seen and the length walked; the moves
are the cities not seen, and the last move home closes the tour, whose
quality is the nearest-neighbour tour's length over its own. The trail
laid on the way home runs along the tour's own edges through the
medium, and a thought choosing its next city reads the trail the whole
way to each. The scent of a move is how near the city is compared with
the nearest not yet seen, squared (the visibility of ant colony
optimisation).

A tour is not walked, so its surface is not the ant's: the ring has no
straight ahead, so the sucker that crawls from it is replaced by a
global draw (`Choice::Natural` does that for any problem of states)
and the persistence is dropped; the scent weighs 8, the route memory
3; the trail forgets in 150 s and only tours near the best recruit
(`recruitment: 8`). 48 thoughts, 8 000 ticks, 1 730 tours brought home:

| | Quality |
|---|---|
| nearest-neighbour tour (the reference) | 1.000 |
| best of 2 000 random tours | 0.625 |
| the mind's mean tour | 0.978 |
| the mind's best tour | 1.050 |

The trail is what finds the tours better than the reference, and it is
also what pulls the mean down: two moves that share cells share marks,
and in a plane of sixteen cities every edge crosses others. With the
trail weighed 4 instead of 2 the mean falls to 0.85 and the best stays;
with the sucker instead of the draw the mean falls to 0.68, because
the sucker's crawl from straight ahead is the ant's prior on a ring of
headings and is noise on a ring of cities.

### A graph colouring

`Colouring::planted(n, 3, density, 4, 64, 40, 3)`: a graph whose nodes
fall into three classes with edges between classes, so a colouring
with three colours exists, to be coloured from a palette of four. The
nodes are coloured in order of falling degree, one move each, from a
palette of the colours no coloured neighbour has; the embedding lays
the nodes along the width and the colours along the height. A full
colouring's quality is `(palette + 1 − colours used) / palette`: a
half with three colours, a quarter with four. A colour already in use
smells of 1, a new one of nothing.

| Graph | Brought home in 6 000 ticks | Mean quality, first → last thousand | Dead ends | Best |
|---|---|---|---|---|
| 20 nodes, 24 edges | 1 403 | 0.309 → 0.421 | 0 | 0.500 (three colours, at tick 141) |
| 24 nodes, 45 edges | 1 338 | 0.250 | 613 (514 retreated from) | 0.500 (at tick 3 849) |

On the easy graph every walk colours it and the trail selects: the
share of three-colourings rises from a quarter to two thirds as the
paths through the layered embedding organise. On the harder one the
walks meet dead ends where no colour is left, mark them no-entry, and
retreat a state to try another colour; nearly every colouring that
comes home uses four colours, and a three-colouring is a rare find
(one in 1 338).
With a palette of three (no slack) and a denser graph the walks find
nothing at all: a greedy walk with a few retreats is not a
backtracking search, and the no-entry marks are context-free (a place
is a node and a colour, not a partial colouring), which is what a
Pharaoh's ant's marks are too.

### Practice

`Practice` is the ants' arena over a mind: learners hold levers that a
hidden, rotating connection attaches to nodes of the hierarchy, each
turn they propose that node's parameters (the surface's weights, its
entropy dial, its deformation) seeing only the parameters, the turn
and their last reward, the mind runs a turn's ticks, and the reward is
the quality brought home per thought. On the maze with a hill-climber
and a random learner over the two castes, with a static rotation of
period two, the reward rises from 7.2 over the first quarter of twelve
turns to 8.4 over the last, against 4.9 for the untouched instinct: the
learners find surfaces that think better than the ant's, mostly by
turning the followers' disorder down.

## What the exploration found

* **The sensorium transfers.** Putting a problem's moves into the ant's
  thirty sensory slots is enough for the surfaces, the hierarchy, the
  entropy control, the deformation and the queen to apply unchanged.
  The problem enters through the embedding (where states lie), the
  scent (what a move smells of) and the quality (what a solution is
  worth).
* **The medium's geometry is the architecture's prior.** Marks are on
  cells, so moves that share cells share marks. For the maze that is
  the point (trails generalise across nearby paths); for the tour it is
  a source of noise the recruitment sharpness and a fast-forgetting
  trail have to fight; for the colouring it makes the no-entry marks
  context-free. An embedding should put like states close.
* **Habits and foresight belong to the walk between decisions.** They
  serve the problems walked freely; where every step is a choice the
  solution is made of, no ground is plain and nothing is foreseen but
  the next state.
* **The ant's instinct is a start, not a fit.** The maze thinks well on
  it with two of its constants cooled; the tour needs the persistence
  dropped and the scent raised, and the sucker, which is the ant's
  geometry of choice, replaced by a draw; the colouring runs on it as
  it is. The practice module is the way from a start to a fit.
* **The queen is a policy over two numbers.** Given stagnation and
  organisation she has a sensible thing to do with the castes' dials;
  what she cannot do is tell a saturating quality from a stagnant one,
  which is why stagnation is measured on the yield with a forgetting
  peak rather than on the best.

## Layout

```text
extras/src/problem.rs     Problem, Moves, Layered, embedding
extras/src/problems/      Maze, Tour, Colouring
extras/src/sense.rs       Candidate, Senses, Body, Sight, features, read_along
extras/src/thought.rs     Thought, Activity, Site, Target
extras/src/mind.rs        Mind, MindConfig, MindStats, Finding, Choice, Geometry, QueenPolicy
extras/src/practice.rs    Practice, PracticeConfig, PracticeReport, Targets, Turn
extras/examples/think.rs  the three problems
extras/tests/thinking.rs  what the tests check
```
