# The lexicon

*An emergent linguistic-symbol embedding: the classes the colony
registers become signs when they are danced at the nest and listened
to on departure, mean where they lead, and are understood as far as
the listeners get there.*

This is the design document for the fourth extension of `ant_extras`
(after the path-topological network, the holonomy embedding and the
bridge). It states the claim, what was built, what was measured on the
way to a lexicon that works, and where it stops.

## The claim

The path-topological network registers classes of routes as symbols:
a symbol is a word (the rays a route crosses, freely reduced), a
channel of its own in the pheromone surface, a glyph (the best route
of the class), a transport, and a weight learned from its yield. That
is a symbol in the sense of a pattern with a label nobody gave. It is
not yet a *sign*: nothing in the colony uses it to tell another
thought anything. The claim of this round is that the step from a
symbol to a sign needs no new representation, only a use, and that
the use can be borrowed whole from the honeybee's dance floor:

* A thought that brings a solution home *dances* the sign of the
  class it walked, in proportion to what it found. The dance fades.
* A thought about to set out may *listen*: it draws a sign from the
  floor in proportion to how much of it is danced, and sets out with
  the sign in mind.
* A sign *means* where the trips that walked its class ended: a
  distribution over blocks of the medium, kept by the lexicon as the
  sign's vector. Signs that lead to the same place are synonyms, by
  the cosine of their meanings, whatever their words.
* A sign is *understood* as far as the thoughts that set out with it
  in mind end where it means: the agreement of what it is taken to
  mean with what it means, and the share of recruited trips that came
  home in its class.

Everything a linguist would ask of the result is then measurable
without anyone labelling anything: the mutual information between the
sign danced and the outcome reached (how much the lexicon carries),
the entropy of the dance floor (how many signs are in use), the
agreement and the success rate (how well each sign is understood),
and the quality brought home with and without the floor (whether
talking is worth anything). The embedding is the sign's meaning vector
beside the symbol's transport: one says where the class leads, the
other what walking it integrates.

## What was built

`extras/src/lexicon.rs`:

* `LexiconConfig`: the chance a departing follower listens (`listen`,
  0.7), the temperature of the draw (`temperature`, 1: a sign is drawn
  in proportion to its dance to the power of one over it, the bees'
  draw at one), dance laid per delivery per unit of quality (`dance`,
  1), the dance's half-life (`half_life_s`, 300), the gain on the
  intended sign's channel and glyph (`gain`, 3), whether the sign's
  glyph is enacted as the plan of the trip (`enact`, true), how much
  of the common trail and the classes' general say a recruit
  disregards (`focus`, 1), the rate of the outcome distributions
  (`rate`, 0.1), the block size outcomes are counted in (`block`, 8),
  and whether scouts listen too (`scouts_listen`, false).
* `Sign`: dance intensity, dances, recruits, successes, strayed, lost,
  the outcome distribution (its meaning) and the distribution of where
  its recruits ended (what it is taken to mean).
* `Lexicon`: `dance`, `draw`, `outcome`, `taken`, `recruited`,
  `success`, `strayed`, `lost`; `meaning`, `taken_meaning`,
  `similarity`, `nearest`, `understanding`; `in_use`, `usage_entropy`,
  `mutual_information`; a `report`.

`extras/src/mind.rs`, `thought.rs`, `topos.rs`:

* `MindConfig::with_lexicon`; `Thought::intent`.
* At departure, a follower may listen and draw a sign. It sets out the
  way the sign's glyph goes, takes the glyph as the plan of its trip,
  and for the trip sets aside its own site, its remembered route, its
  habits (no transit carries it) and the searching heat.
* In the sensorium, a thought with a sign in mind disregards the
  common trail and the classes' general say by its focus, and adds
  the lexicon's gain times the ridge of its sign's channel (the level
  along a move relative to the most of it along any move open to it,
  `PathNet::channel_along`) and times the plan's direction.
* On finding, the outcome is the cell; on delivery the sign of the
  trip's class is danced in proportion to the quality, the outcome is
  taken into its meaning, and if the trip set out with a sign in mind
  it counts as understood (home in that class), strayed (home in
  another) or lost (home with nothing), and where it ended is what
  the sign was taken to mean.
* A trip's word ends in a *destination letter*, the index of the goal
  it reached (`Problem::suffix`, `DESTINATION_LETTERS`), so that two
  sources on the same side of every puncture are two classes; the
  route-only word (`Word::without_destinations`) still labels and
  generates.
* `Maze::with_goals` places several goals of different qualities;
  `Thought::plan_direction` now advances past the plan's points
  already passed rather than returning to each.

`extras/examples/lexicon.rs` runs the maze round a wall with a rich
source beyond the wall and a poor one on the near side, prints the
lexicon as it forms, names the rich source's sign after the fact, finds
its synonyms, and compares the quality brought home with and without
the floor.

## What was measured

The setting is `Maze::around_a_wall(64, 40)`: nest at (4, 13), a rich
source of quality 1 at (59, 13) beyond a wall reachable over it or
under it, a poor source of quality 0.2 at (11, 35) on the near side;
48 thoughts, a trip budget of 300 ticks, 6000 ticks. Numbers are from
`cargo run --release -p ant_extras --example lexicon` (2.7 s).

| | |
|---|---|
| signs in use | 9 of 9 living, usage entropy 1.64 nats |
| what the sign tells about the outcome | 0.68 of 0.68 nats: all of it |
| recruited trips understood | 100 % of 1263 |
| agreement of what the rich source's sign is taken to mean with what it means | 1.00 (551 set out with it) |
| the dance floor at the end | the rich source's main sign at 38.6, the poor source's at 5.0 |
| synonyms | eight signs mean the rich source (over the wall, under it, and round the learned punctures), all at similarity 1.000; the poor source's sign at 0.000 to them |
| quality brought home, silent | 973.6 (3168 deliveries, mean quality 0.31) |
| quality brought home, talking | 1419.6 (2114 deliveries, mean quality 0.67) |

The same on the test's smaller maze (`Maze::around_a_wall(48, 32)`, 32
thoughts, budget 200, 4000 ticks): 848 against 596, the rich source's
signs danced at 105 against the poor one's 1.4.

The lexicon that works is the fourth one built. The three that did not
are the measurement that matters, because each failure named a way the
colony's own machinery fights a sign:

1. **A softmax draw at a fixed temperature locks in.** The poor source
   is near, so it is found first and danced first; a draw by
   `exp(dance / T)` over intensities of order ten then never draws
   anything else. The rich source was danced 82 times against 2663
   and recruited never. The bees' draw, in proportion to the dance
   itself, is a linear positive feedback whose winner is the source
   with the larger quality per trip, not the one found first: with it
   the rich source took the floor.
2. **A recruit that keeps its own mind strays.** With the proportional
   draw, recruits set out with the rich sign in mind and 92 % of them
   ended at the poor source. The thought's own site (the sensorium's
   site feature, weight 2), its remembered route, its habits (memoized
   transits keyed by leg and heading, not by intent) and the common
   trail all pointed at the source it knew. A recruit has to set its
   own memory aside for the trip: that is `focus`, the site and route
   set aside, and no transit taken with a sign in mind. With it, 86 to
   96 % of the recruits reached the rich source.
3. **A sign followed as a scent is followed slowly.** Attending to the
   sign's channel with a sensitised perception (the half-saturation
   divided by 20 so that a faint channel is still felt) got the
   recruits there, but along routes of 120 to 175 cells against a
   glyph of 44 to 63, and a third ran out of budget; the yield stayed
   below the silent colony's (435 against 596). The log perception
   flattens the channel's ridge, so the recruit random-walks in its
   band. Following the ridge relative to its peak instead shortened
   the routes (88) but a third strayed again.
4. **A sign enacted is a sign understood.** Taking the sign's glyph as
   the plan of the trip, with the plan followed by skipping the points
   already passed, the recruits' routes ran at 49 cells against a
   glyph of 44, none strayed, none was lost, and the talking colony
   out-yielded the silent one by 44 %.

The reading of 4 is the point of the round. The symbol's generative
use, the glyph walked again, is what makes the sign *understood*; the
stigmergic reading, the channel's ridge, keeps it *grounded* where the
glyph is stale; the dance floor is what makes it *used*; and the
outcomes are what make it *mean*. A sign is a class with all four.
Meaning here is use, in the plain sense that the lexicon measures a
sign's meaning by where the thoughts that heard it went.

## Where it stops

* **Signs are one-place.** Each refers to a place, a block of the
  medium. The words have structure (`d a b c` is four rays crossed in
  order) but the floor does not compose them: nobody dances "this and
  then that". The free product of two words is a word, and a listener
  could enact the concatenated glyphs; that is the next step, and the
  one that would make the lexicon a language rather than a vocabulary.
* **Meaning is spatial.** The outcome is a cell of the medium, so the
  lexicon applies to any problem whose states have places, but for
  the tour or the knowledge graph a block of the embedding is a coarse
  and arbitrary referent. A meaning over the problem's own states
  (the tour's last city, the query's answer) is a small change to
  `outcome` and a large one to what synonyms would be.
* **The floor is one and global.** Every returning thought dances at
  the same nest and every departing follower hears everything; there
  is no locality of listening, no dialect, no drift between groups.
  The echo (`docs/echo.md`) is a first answer to the reach of the
  floor, not to its unity: a sign walked is heard along the way.
* **The decoding is shared memory.** A bee's dance carries a vector;
  our sign carries a class, and the recruit decodes it through the
  network's glyph and channel. The lexicon is public only because the
  network is. A private lexicon, in which each thought kept its own
  glyphs, would have to *learn* to decode, and that is where the
  question of a language proper begins.
* **Only success is danced.** A dead end is marked no-entry on the
  ground but never told at the nest; the lexicon has no sign for a
  danger, a direction to avoid, or a source exhausted.
