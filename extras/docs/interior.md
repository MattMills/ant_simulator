# The interior

*A higher-order layer of thought that walks the colony's own
vocabulary, coupled both ways to the thoughts that walk the world.*

This is the design document for the sixth extension of `ant_extras`
(after the path-topological network, the holonomy embedding, the
bridge, the lexicon and the echo). It states the claim, what was
built, what was measured, including what failed on the way, and where
it stops.

## The claim

Everything in the architecture so far is one layer: thoughts walk an
embedding, leave marks, and the colony extracts invariants from their
traffic, the words registered as symbols, what each yields, the glyph
of each, the skeleton of the walks. The echo of the previous round
read the idea of a slower, self-interacting layer literally, as a
caste walking glyphs on the same ground. The reading that generalises
is this: the invariant structure the foragers produce is itself a
space, and the same dynamics can walk it. A second colony of thoughts
whose ground is the first colony's abstraction, at a slower epoch,
with longer-lived marks, whose food is the foragers' success and whose
trails are the colony's beliefs about its own vocabulary. It
self-interacts downward by speaking through the floor, so that the
new thoughts are imprinted with what the interior believes, and by
proposing words nobody has walked for the foragers to test; upward,
because the foragers' outcomes rewrite its landscape. Its own traffic
has invariants, so it registers symbols of its own, classes of
derivations, and a third layer can walk those in turn. The
architecture is the same at every level; only the ground changes.

## What was built

`extras/src/interior.rs`:

* `Interior`, a `Problem` whose states are words. The origin is the
  empty word. A move appends a letter of the vocabulary (the letters
  of every word the foragers have registered, both ways round a ray)
  or, when `chunks` is on, a whole known route-word at once. A word
  ending in a destination letter is an arrival; if the outer thoughts
  have brought that word home it is a solution worth the class's yield
  against the best yield, times its precision, shrunk by its support
  (`support / (support + prior)`); if not, it is a dead end and a
  hypothesis. The scent of a move is the worth of the known words it
  is a prefix of, discounted by the letters remaining. Words are laid
  out on a grid layer by layer of their length, each in the slot it
  first took. A chunk is a letter of the interior's own words, so the
  interior's classes are parses: which known words a thought composed.
* `Colony<P>`: an outer `Mind<P>` and an inner `Mind<Interior>`,
  the inner stepped once per `slowness` outer ticks, resting
  `rest_ticks` between trips so that a long thought costs little more
  than a short one, its trail with a half-life of `half_life_s`. Every
  `epoch_ticks`, the layers couple. Up: the interior reads the outer
  network's living symbols. Down: each outer sign is given a standing
  dance on the floor in proportion to the interior's trail on its
  word, as a fraction `standing` of the loudest dance, so that recruits
  enact what the interior believes; optionally attention on the
  symbol's channel (`attention`, off by default) and expression of the
  best word (`expression`, off). Hypotheses: for each word the inner
  thoughts arrived at that no forager has walked, a route of that class
  to the destination the word names is generated over the outer
  embedding and proposed as a symbol with no support, laid out and
  given a standing dance; it is confirmed when a forager brings it
  home, and evicted first when room is needed if nobody does.
* `beliefs()`: the interior's trail on every known and proposed word,
  normalised to the strongest; a `report`.

`topos.rs`: `PathNet::propose` (a symbol with no support), the
`proposed` flag, a proposed class taking its first trip's quality and
glyph as they are, and no weight for or against a class nobody has
walked. `problem.rs`: `Problem::destination`, the state a destination
letter names. `lexicon.rs`: `Sign::standing`, a dance that does not
fade, counted with the dance in the draw. `Maze::hall`: a wall with a
way round it above and below and four pillars, words of up to five
letters.

## What was measured

**The hall** (`Maze::hall(64, 40)`, 32 thoughts talking, a trip
budget of 300, 6000 ticks; the interior of 12 thoughts at a quarter
of the rate; `cargo run --release -p ant_extras --example interior`,
13 s in all):

| | alone | with the interior |
|---|---|---|
| quality brought home | 961 | 972 |
| classes registered | 9 | 9 |
| hypotheses proposed, confirmed | | 1, 1 |
| the interior's trips, brought home | | 321, 179 |
| the interior's own classes | | 5 |

The hypothesis was the class that crosses no ray at all, the way
round the bottom of everything, proposed before any forager had walked
it and confirmed by the foragers within the next thousand ticks. The
interior's strongest belief is in `[c @goal]`, the shortest word
that yields the best; its own classes are parses over a puncture it
learned in its own ground and the chunks its thoughts composed:
`[a m1]` 85 times, `[1]` 74, `[a]` 11, `[a m0]` 6, `[a m4]` 2.

**A third layer.** An interior refreshed from the interior's own
network knows its four living parses at worths from 0.05 to 0.84, and
a mind of eight thoughts on it brings home 300 of 309 trips in 2000
ticks; its best thought is the parse `[a [a m1]]`. The construction
is the same one level up.

**The two-source maze** (rich beyond the wall, poor near the nest):

| floor listened to | alone | with the interior |
|---|---|---|
| on seven departures in ten | 848 | 831 |
| on one departure in ten | 695 | 740 |

On a loud floor the interior changes nothing, because the floor
already finds the rich source. On a quiet floor the interior's
standing dance (20.5 on the class it believes in, against that class's
own dance of 14.9) is what the few who listen hear, and the colony
brings home 6 % more. Without a lexicon the interior has no voice, and
the colony is unchanged.

**What failed on the way**, which is the measurement that matters:

1. *Attention on a channel does not move a forager.* The first
   coupling laid attention on the symbol the interior believed in.
   The channel was faint (a class walked once), the attention made
   every departing thought set out along its glyph, and discovery of
   the rich source collapsed: 633 against 848 on the maze, 861 against
   965 in the hall. The lexicon's recruit machinery, the glyph as the
   plan, the ridge as the scent, the common trail set aside, is the
   only downward path that works, and the interior now speaks only
   through the floor.
2. *A belief without evidence.* The interior believed in a class one
   trip had walked, because that trip's quality was 1. Worth is now
   shrunk by support.
3. *A hypothesis weighed by its yield is a hypothesis avoided.* A
   proposed class has quality 0, so the network's reweighing gave it
   the lowest weight and the foragers avoided its channel. A class
   nobody has walked now weighs nothing, for or against, and is
   recruited to by the standing dance instead.
4. *The interior prefers the shortest good word.* A one-letter word
   is thought more often than a five-letter one, so the belief goes to
   `[@goal]` over `[a @goal]` at equal yield. Long rests between
   thoughts temper this but do not remove it.

## Where it stops

* **Beliefs are shallow.** A trail over words tracks yield and
  brevity; the interior does not model why a class yields, and it
  carries no transports yet (its capacity is zero).
* **Hypotheses are bounded by geometry.** Six of the seven unknown
  words the inner thoughts arrived at had no route of that class
  within the longest word; the interior learns which orders of letters
  are impossible only through the no-entry marks its own dead ends
  leave.
* **The interior speaks only through the floor.** Without a lexicon
  it thinks and cannot act, and the one direct lever it has,
  attention, breaks discovery.
* **The recursion is shown, not run.** The third layer walks a
  snapshot of the second's parses. A live third layer needs the
  coupling written over a layer trait that a mind and a colony both
  satisfy, so that a colony can be the outer mind of another.
* **One interior, one floor, one queen.** The interior has no queen
  reading it and does not read the outer queen; stagnation in the
  interior's beliefs is not yet a signal to anyone.
