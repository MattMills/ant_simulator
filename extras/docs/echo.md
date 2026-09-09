# The echo

*Signs persisted in walk form: a caste that speaks the lexicon's
language at a slower epoch and everywhere along the route, so that the
sign becomes traffic, the traffic becomes the colony's invariant
geometry, and the geometry imprints the thoughts born into it.*

This is the design document for the fifth extension of `ant_extras`
(after the path-topological network, the holonomy embedding, the
bridge and the lexicon). It states the claim, what was built, what was
measured, including what the echo does not do, and where it stops.

## The claim

The lexicon's dance floor speaks fast and at the nest. A sign is
danced when a thought comes home with something and fades within a
few hundred ticks; a thought hears it only when it sets out. The claim
is that the same language can be spoken in another register: a set of
thoughts that *persist the dance in walk form*, holding a sign for an
epoch and walking its glyph out and back, again and again, whether or
not anything is at the end. The other thoughts receive it not as a
dance but as traffic. The walks lay the sign's channel both ways, so
the sign's trace in the ground is maintained; they enter the movement
history, whose invariant skeleton is the colony's cognitive geometry
and the ground the punctures are learned from; and a thought out
searching that meets an echo takes the sign it walks as its own, from
the point where they met, as a follower takes the route from a tandem
leader.

That closes a loop the lexicon left open. Past foraging success is
danced (fast, at the nest); the echoes re-walk it (slow, along the
route); the re-walking writes it into the colony's internal structure,
the channels and the invariant skeleton, as the literal shape in the
ground of how good food is found in this area; and that structure
imprints the thoughts that come after, who follow the ridge, hear the
echo on the way, and take the transits keyed on the invariant ground.
The echo is the colony's replay: memory kept by being re-spoken,
refined by every walk that finds a shorter route of the class, and
re-seeded into the floor whenever a walk finds the source still there.
In the ants this is tandem running and group recruitment; in a brain
it is the replay of a trajectory at rest, at a slower timescale,
consolidating the episode into structure.

## What was built

`extras/src/echo.rs`:

* `EchoConfig`: how many followers echo (`echoes`, 4), the epoch a
  sign is held before the floor is consulted again (`epoch_ticks`,
  1000), the empty walks tolerated before a sign is let go
  (`patience`, 6), what an echo lays per cell into the sign's channel
  on its way (`deposit`, 10), the chance per tick that a thought out
  searching within reach takes the echo's sign (`hear`, 0.5), and the
  reach (`radius`, 1 cell).
* `Echo`: the bookkeeping (walks, walks that found the source, walks
  that found nothing, signs drawn and let go, epochs kept in silence,
  walks that refined a glyph, thoughts that heard a sign on the way and
  how they came home, walks per sign) and a report.

`extras/src/mind.rs`, `thought.rs`:

* `MindConfig::with_echo`; the last `echoes` followers are echoes.
* At departure an echo walks the sign it holds. The sign is drawn from
  the floor (the lexicon's draw) when the echo has none, when its epoch
  is over, when it has found nothing at the glyph's end `patience`
  times, or when its class has died; when nothing is danced at all the
  held sign is kept in silence. The glyph is the plan of the walk out;
  at the end the echo either finds the source and brings it home like
  any thought (dancing it, re-seeding the floor from memory, and
  refining the glyph if its walk was shorter), or finds nothing, has
  the miss observed against the class, and turns back along the glyph.
  Out and back, carrying nothing, it lays the sign's channel.
* Every tick, a thought out searching with no sign in mind, no load
  and no habit under way, within reach of an echo and away from the
  nest, takes the echo's sign with chance `hear`: the glyph becomes its
  plan from the nearest point, and it counts as recruited.
* `Mind::renew(share)`: new thoughts, for the experiments: a share of
  the thoughts that are not echoes forget their site, their remembered
  route and any sign held. `Lexicon::silence` clears the floor.
  `Mind::problem_mut` and `Maze::set_goals` change the world.

## What was measured

The setting is the lexicon's test maze: `Maze::around_a_wall(48, 32)`
with a rich source of quality 1 beyond the wall and a poor one of
quality 0.2 near the nest, 32 thoughts, a trip budget of 200 ticks,
four of the thoughts echoing where echoes are on. Numbers are from
`cargo run --release -p ant_extras --example echo`.

**Reach.** With a floor listened to on one departure in ten, the sign
has to travel by echo or not at all.

| | brought home | quality | heard a sign on the way | came home in its class |
|---|---|---|---|---|
| no echoes | 1634 | 695 | | |
| 4 echoes | 1197 | 890 | 627 | 96 % |
| 8 echoes | 1407 | 873 | 776 | 97 % |

The echoes' own walks found the source 206 times and nothing 27 times.

**Invariance.** At tick 1000 the rich source's glyph lies 68 % in the
movement history's invariant skeleton with echoes and 3 % without; by
tick 2000 both are at 95 %. The echo traffic makes the route part of
the colony's geometry an epoch before the foragers' own traffic does,
which is the epoch in which the punctures are learned from that
geometry.

**Refinement.** Over 6000 ticks the walks shortened the rich glyph
from 44 to 42 cells (seven walks refined it), and the colony with four
of its thirty-two thoughts echoing brought home 1430 against 1346:
the echoes cost nothing, because their walks deliver.

**Imprinting.** At tick 2500 every thought but the echoes forgets its
site, its route and any sign it held, and the floor falls silent. In
the next 200 ticks the colony brings home 47 with echoes and 41
without, against a steady 50, and the rich source's dance is back
above 100 within 1600 ticks: new thoughts are imprinted from the
ground, the classes' channels weighed by their yields and the glyphs
in the network, in both colonies, and from the echoes' walks a little
faster. The echoes re-seed the silent floor: 288 thoughts heard a sign
on the way, and 153 echo walks found the source and danced it.

**Persistence is not where the echo adds.** The rich source was taken
away for 800, 2000 and 3000 ticks and put back. In every case the
colony without echoes recovered as fast as the one with them (27 to 33
brought home in the first 200 ticks, 41 to 55 in each window after,
against 46 to 55 in steady state), because the memory of the route
already lives in the ground: the glyph is in the network and the
lexicon's linear draw revives a sign from the smallest residue of
dance. Patient echoes (30 to 40 empty walks tolerated, epochs of 4000
to 6000 ticks) held the poor source's sign through the gap, walked it
464 and 695 times, and cost yield. An echo that is faithful to a sign
the world has left is the failure mode of this design, and the
lexicon's patience of six empty walks is the guard.

## Where it stops

* **The echo is the literal reading.** A caste walking glyphs on the
  same ground is one way to persist a sign; the reading that
  generalises, a second mind whose ground is the first's vocabulary,
  is the interior (`docs/interior.md`).
* **The echoes choose from the floor.** They amplify what is danced;
  they cannot speak a sign nobody dances. A colony whose floor is
  wrong is echoed wrong, faster.
* **Hearing is by contact.** A thought takes the sign from an echo
  within a cell; there is no dialect, no distance at which the echo is
  heard as something else, and no way for a hearer to tell an echo
  from a forager.
* **One register.** The echo repeats a glyph; it does not compose
  glyphs, abbreviate them, or vary the epoch by how much the sign is
  worth. The next step is the same as the lexicon's: a walk that
  concatenates two glyphs is the first sentence.
* **Memory has a lifetime set by patience, not by evidence.** An echo
  lets a sign go after a fixed number of empty walks. The evidence
  it has, the miss observed against the class and the class's falling
  precision, is what a Bayesian echo would use; here it is counted, not
  weighed.
