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

## Symbols: the pheromone surface made variable

The ants' surface has six channels with fixed meanings. The
path-topological network (`topos`) lets the surface grow: a channel is
registered for every *kind of route* the thoughts keep walking home,
and the kind is a topological invariant of the route.

**Words.** Given a set of *punctures* in the embedding, a ray runs up
from each. A route's *word* is the sequence of rays it crosses, each
crossing a letter signed by its direction, freely reduced (a crossing
followed by its reverse cancels): `b a` crosses the second ray, then
the first; `1` crosses nothing. Two routes with the same endpoints have
the same word if and only if one can be deformed into the other without
passing through a puncture: the word is the route's homotopy class, its
h-signature (Bhattacharya, Likhachev & Kumar 2012). It is a label
nobody gave; the pattern names itself.

**Registration.** A word walked home on enough trips (`support`, three
by default) is registered as a *symbol*: a channel of its own in the
symbol field, where the trips of that class lay their trail on the way
home; a *glyph*, the best route of the class; statistics (support, mean
quality, mean length); and a *weight*, learned from how the class
yields (quality per length, against the shortest class, relative to the
mean over classes), which the thoughts sense along a move as they sense
a pheromone and add to their scores. The symbols are the units of a
network whose wiring is the topology of the routes.

**Inference and generation.** A route, walked or not, is labelled by
its word: the symbol of that class if there is one, else the nearest by
edit distance, and its name if a name was attached after the fact
(`Mind::label`, `PathNet::name`). A route of a class is produced again
two ways: by a breadth-first search over cells and the words of the
ways to them, which returns a shortest route of exactly that class
(`Mind::generate`); or by expressing the symbol, laying its glyph into
its channel strongly and attending to it, so that the thoughts walk the
class again and bring it home (`Mind::express`), which the labels of
what comes home then confirm.

**Learned punctures.** The alphabet grows too. Where the walks steadily
go (the history's `flow_mask` at a low rate) encloses holes; a hole
large enough, persisting for enough epochs, that would tell apart the
exemplars of one class (routes of one word now falling into two words
with it, both walked) becomes a puncture, and every symbol's word is
refined under the new alphabet, coinciding classes merged. A hole every
route passes the same side of teaches nothing.

`cargo run --release -p ant_extras --example symbols`:

* **Round a wall** (the wall's centre the given puncture): after 4 000
  ticks, 739 routes home fall into two classes, `a` over the wall (637,
  mean length 82, weight +0.54) and `1` under it (102, length 119,
  weight −0.54). Named after the fact, a route nobody walked high over
  the wall is labelled "over the wall", one low "under the wall", and
  one that goes over, comes back under and goes over again (`a a`) is
  no class walked, nearest "over the wall" at one letter. Generation by
  search gives a 66-cell route over and a 76-cell route under, each
  labelled as asked. Expressing "under the wall" raises its share of
  the trips brought home from 21 % to 47 % while expressed, back to
  20 % after release: a thought that departs while a symbol is
  expressed sets out the way its glyph goes, since the sucker that
  selects its moves climbs from straight ahead and looks where it
  starts.
* **Round a pillar nobody declared**: with no puncture given, all
  routes are one class (`1`) until the walks round the pillar enclose
  it; the hole persists (seen at every look from tick 1 500 on), the
  puncture is learned at tick 3 000 once it tells the class's exemplars
  apart, and the routes fall into `a` (1 055) and `1` (663), the two
  ways round it, of equal yield.
* **Tours of ten cities** (the cities the punctures): 1 437 tours home
  fall into seven classes by how they wind round the cities; the best
  class (`g d c' d' j' e' g' a'`, quality 1.248, 1 197 tours) weighs
  +0.87 and the worst (`a g d c' d' j' e j c d c' d' j' g' a'`, quality
  0.957) −0.73.

Where it stops: the invariant is exact only for routes with the same
endpoints, so a trip's class is its class from the origin; a word is
kept to twelve letters (a route winding more is noise, not a class);
and a puncture is a point, so the class of a route round a long wall is
one letter whatever its shape, which is what a homotopy class is.

## The holonomy embedding

The symbols carry vectors as well. A problem may give its thoughts an
integrative capacity, a state in D dimensions that a free walk develops
step by step through the turns of the folds it crosses and a walk of
states integrates move by move (a relation followed adds its vector).
On a flat connection what a route integrates depends only on its word,
so a symbol's *transport*, the mean of what its routes integrate, is
the class's vector and the spread of those integrals its curvature.
Moves and portals are decidable gates, admitting a thought by its state
and its word so far. Learning is local: a trip brought home credits the
residual between what it integrated and its target back to the moves
it made, and arrival near a target vector is food.

Round a cylinder (`tests/thinking.rs`) the classes are the windings and
their transports circumferences. On a knowledge graph of families
(`cargo run --release -p ant_extras --example relations`) queries are
trips, the words of the trips brought home are rules such as
`grandparent: parent parent`, the relations' vectors learn to close
along them (the planted rules close to 0.01), and 293 held-out facts
are predicted with an MRR of 0.895 and hits@1 of 0.80 by rules and
geometry together, against 0.011 before. The design, the numbers and
where it stops are in [`docs/holonomy.md`](docs/holonomy.md).

## The bridge

The colony's forward and backward filters, made explicit. A thought's
walk conditioned on where its trips end is a Schrödinger bridge, the
product of a function running forward from the nest and one running
backward from the outcome, and its drift is the log-gradient of the
backward one, Doob's h-transform. The colony computes that pair
implicitly, the outbound walks sampling the forward filter and the
trail laid home accumulating the backward one. `bridge.rs` solves both
on a coarse graph of the medium every epoch, at a temperature that is
the forcing pressure, and measures the surprise of the colony's flows
against their product, the trail's agreement with it, and the corridor
the product predicts; the queen reads the surprise, and the thoughts
may follow the model's drift. On the maze round a wall a small drift
gain is worth a quarter more deliveries, the corridor narrows from 147
to 44 effective blocks as the temperature falls from 64 to 1, and the
trail correlates with the predicted density at 0.37. Symbols can be
weighed by their sector partition functions at a temperature, and a
practice can be conditioned on survival. The design, the numbers and
where it stops are in [`docs/bridge.md`](docs/bridge.md).

## The lexicon

The symbols become signs the way a honeybee's source becomes a dance.
A thought that brings a solution home dances the sign of the class it
walked, in proportion to what it found, and the dance fades; a thought
about to set out may listen, draw a sign in proportion to how much of
it is danced, and set out with it in mind: the way its glyph goes,
with the glyph as the plan of its trip, following the ridge of the
class's channel, and with its own site, route, habits and the common
trail set aside for the trip. A sign means where the trips of its
class ended, a distribution over blocks of the medium that is its
vector; signs that lead to the same place are synonyms whatever their
words; and a sign is understood as far as the thoughts that heard it
end where it means. `lexicon.rs` keeps the floor and measures it: the
mutual information between the sign danced and the outcome reached,
the entropy of the floor, each sign's agreement and success rate, and
what talking is worth. On the maze round a wall with a rich source
beyond it and a poor one near the nest, nine signs form, the sign
tells all 0.68 nats there are about the outcome, every one of 1263
recruited trips is understood, eight signs meaning the rich source are
synonyms at similarity 1.000, and the colony that talks brings home
46 % more quality than the one that does not. The three lexicons
built before the one that works, and what each failure taught, are in
[`docs/lexicon.md`](docs/lexicon.md).

## The echo

The dance floor speaks fast and at the nest. An *echo* is a follower
that speaks the same language at a slower epoch and everywhere along
the route: it draws a sign from the floor, holds it for an epoch, and
walks the sign's glyph out and back, again and again, laying the
sign's channel both ways whether or not anything is at the end. The
other thoughts receive it as traffic. The walks enter the movement
history, whose invariant skeleton is the colony's geometry; a thought
out searching that meets an echo takes the sign it walks as its own,
from the point where they met, as a follower takes the route from a
tandem leader; and a walk that finds the source still there brings it
home, dances it, and refines the glyph if it was shorter. On the
two-source maze with a floor few listen to, four echoes among
thirty-two thoughts raise the quality brought home from 695 to 890 and
carry the sign to 627 thoughts on the way, 96 % of whom come home in
its class; the route lies 68 % in the invariant skeleton at tick 1000
against 3 % without them; the glyph shortens from 44 to 42 cells; and
thoughts newly born into the colony, with the floor silenced, are
imprinted from the ground within 200 ticks. What the echo does not add
is persistence: the memory of a route already lives in the glyph and
the floor's linear draw, and patient echoes of a sign the world has
left cost yield. The design, the numbers and where it stops are in
[`docs/echo.md`](docs/echo.md).

## The interior

The reading of the same idea that generalises: the invariant structure
the foragers produce, the words registered as symbols with their
yields and glyphs, is itself a space, and the same dynamics can walk
it. An *interior* is a problem whose states are words, a move appends
a letter of the vocabulary or a whole known word, and a solution is a
word the foragers have brought home, worth what it yields them; a
*colony* is an outer mind on its problem and an inner mind on the
interior, stepped at a quarter of the rate with marks that last ten
times longer. Every epoch the layers couple: the interior reads the
vocabulary; each sign on the floor is given a standing dance in
proportion to the interior's trail on its word, so that recruits enact
what the interior believes; and a word the inner thoughts arrived at
that nobody has walked is a hypothesis, generated as a route, proposed
as a symbol with no support, and confirmed or evicted by what the
foragers do with it. The interior's own traffic registers classes of
derivations, parses, and a third layer can walk those in turn. In a
hall with a wall and four pillars the interior proposes the class that
crosses no ray before any forager walks it and the foragers confirm
it; on a quiet floor it raises the quality brought home by 6 %; on a
loud one it changes nothing; and its first coupling, attention on a
faint channel, collapsed discovery, which is why it speaks only through
the floor. The design, the numbers, what failed on the way and where
it stops are in [`docs/interior.md`](docs/interior.md).

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
* **A route's class is a label nobody gave.** The word of a route
  round the punctures is exact, cheap, and grows with the alphabet;
  registering words as channels makes the pheromone surface variable
  without touching the fixed sensorium, since the symbols add to the
  score beside it. What is learned is which classes yield, which
  punctures matter, and the glyph of each class; what is inferred is
  the class of any route; what is generated is a route of any class.
* **The trail is the product, not the backward filter.** Laid along
  successful routes on the way home, the trail estimates the bridge
  density, forward times backward, and correlates with it at 0.37
  where it correlates with the desirability alone at 0.1. Following
  the explicit desirability's drift at a small gain helps because it
  breaks the ties the product leaves; at a large gain it traps.
* **A translation is a coarse transport.** The holonomy embedding's
  transports are exact (zero spread) and the planted rules close, yet
  the geometry alone predicts poorly, because a symmetric relation can
  only close at zero and then tells nothing apart; the words carry
  what the translation cannot. The affine transport the design allows
  is the next step.
* **A sign is a class with a use.** Nothing had to be added to the
  symbol for it to become a sign: a dance floor at the nest, a draw in
  proportion to the dance (the bees' linear feedback, which picks the
  source worth more per trip rather than the one found first), and a
  recruit that sets its own memory aside and enacts the sign's glyph.
  Each of those was found by a lexicon that failed without it: a
  softmax draw locks in on the first source, a recruit with its own
  site in mind strays to it, and a sign followed as a scent is followed
  too slowly to pay. Meaning is measured by use: where the thoughts
  that heard the sign went.
* **A sign walked is a sign heard everywhere, but not a sign kept
  longer.** Echoes carry the lexicon to thoughts the floor never
  reaches, put the route into the colony's invariant geometry an epoch
  early, and refine the glyph by walking it; they do not lengthen the
  memory, which is in the ground already, and an echo faithful to a
  sign the world has left is the design's failure mode.
* **The abstraction is a ground.** The words the foragers register are
  a space the same machinery walks; an interior's trail over them is a
  belief, its dead ends at unknown words are hypotheses, and its own
  classes are parses. What it cannot do is act directly: attention on
  a channel breaks discovery, and the only downward path that works is
  the colony's own language, a standing dance on the floor.
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
extras/src/topos.rs       Word, Route, Rays, SymbolField, Symbol, PathNet, Label, SymbolConfig
extras/src/problems/relations.rs  Relations, Node, Triple, Evaluation (the knowledge graph)
extras/docs/holonomy.md   the holonomy embedding: proposal, build, numbers, limits
extras/src/bridge.rs      Bridge, BridgeConfig (the forward and backward filters on the grain)
extras/docs/bridge.md     the bridge: claim, build, numbers, limits
extras/examples/bridge.rs the bridge observing, forcing, drifting; sector weights; survival
extras/src/lexicon.rs     Lexicon, LexiconConfig, Sign (the dance floor: signs danced, drawn, meant, taken)
extras/docs/lexicon.md    the lexicon: claim, build, the three that failed and the one that works, limits
extras/examples/lexicon.rs signs forming over two sources, naming, synonyms, talking against silence
extras/src/echo.rs        Echo, EchoConfig (signs persisted in walk form: held, walked, heard on the way)
extras/docs/echo.md       the echo: claim, build, reach, invariance, refinement, imprinting, what it does not add
extras/examples/echo.rs   echoes on a quiet floor, the route in the skeleton, new thoughts imprinted
extras/src/interior.rs    Interior (words as a problem), Colony (two layers coupled), InteriorConfig
extras/docs/interior.md   the interior: claim, build, the hall, a third layer, what failed, limits
extras/examples/interior.rs  a hall with pillars, hypotheses, parses, a third layer, a quiet floor
extras/examples/relations.rs  the family graph learned by walking
extras/examples/symbols.rs the network round a wall, round a pillar, over tours
extras/examples/think.rs  the three problems
extras/tests/thinking.rs  what the tests check
```
