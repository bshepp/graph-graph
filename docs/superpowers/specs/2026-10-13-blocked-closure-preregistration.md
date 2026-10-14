# Pre-registration — is a blocked closure the trigger of the sheet's crossover?

**Date:** 2026-10-13, night. **Status:** PRE-REGISTRATION for the fresh-seed test below,
committed before any fresh seed was grown. The observation that motivates it is post hoc
and is stated as such.

## Origin (post hoc, exploratory; not part of the test)

While running the pre-registered onset scan of seed 308 (`edge_spectrum.py --onset`), the
deepest fjord of the edge turned out to sit at a *fixed node* (24862) from N = 32,768 on,
and the blocked-closure census around it read "blocked closures here, none anywhere else".
An ad-hoc scan (`blocked_scan`, scratch) then showed: seed 308 has **zero** boundary nodes of
degree >= 6 up to N = 25,500, two adjacent ones (24862, 25107) from N = 26,000 -- both
permanently un-closable -- and from then on node 24862 is the oldest node on the boundary
and, from 32,768, the bottom of the deepest fjord, with the blocked set growing 2 -> 24 by
524 k along the fjord wall. `B / sqrt(N)` starts rising at ~27-30 k. Seed 305 (compact to
4.19 M) has zero blocked nodes at 32 k, 131 k, 524 k. Seed 300 has zero up to 900 k and six
at 1,048,576, all within 3 hops of node 916282, the oldest boundary node and the fjord
bottom; its N*_B was 886 k.

Mechanism in the rule's own terms (`sheet_growth.grow_sheet`): a boundary edge whose
endpoint has degree >= c fires only closure events, and only if that endpoint is closable
(two boundary neighbours a, b, not adjacent, no other common neighbour, both of degree
< c). Degrees never fall and adjacency never shrinks, so an un-closable degree-c boundary
node is **dead forever**: growth stalls there while the disc around it advances, the dead
spot becomes the bottom of a fjord, and the boundary length grows with N. This is item 1's
DEFECT hypothesis (the degree guard leaving a crack no later event repairs), found at the
bottom of a fjord instead of the base of an arm.

## Frozen definitions (`blocked_closure.py`)

- **Blocked node**: a boundary node (has a boundary edge) of degree >= c = 6 for which
  `sheet_growth._closable` returns None. Blocked is permanent, so "first N with a blocked
  node" is well defined; on saved graphs it is found by octave scan plus bisection on the
  prefix size; on fresh growths it is the first checkpoint (2^(1/8) spacing from 4096) at
  which the count is positive.
- **N*_B**: the step-2 boundary-exponent crossover (`sheet_transition.crossover_size` on
  `boundary_edges` at the step-2 checkpoints `checkpoints(1024, n_max)`).

## The test: fresh seeds

`python blocked_closure.py --grow --beta 1.5 --seeds 400 407 --n-max 524288 --jobs 8`.
Eight fresh seeds at beta = 1.5 (the Part B law gives N* ~ 1e5 there, so most seeds cross
well inside 524,288 nodes and the run is cheap). Everything -- seeds, beta, n_max, the
checkpoints -- is fixed here.

**Frozen readings** (over the seeds whose N*_B is finite):
- **TRIGGER CONFIRMED**: in >= 3/4 of them a blocked node exists by N*_B and
  `N*_B / N_first <= 4` (the dead spot appears at most two octaves before the exponent
  crosses 0.75).
- **TRIGGER REFUTED**: in >= 1/2 of them either no blocked node exists by the end, or the
  first appears after N*_B, or more than two octaves before it.
- **UNDECIDED**: otherwise.
Control, reported with the verdict: the seeds that have not crossed by 524,288 should have
no blocked node (count disclosed, no rule).

**Prediction.** TRIGGER CONFIRMED (70%), with `N*_B / N_first` typically 1.5-3 (the
exponent needs a few doublings after the dead spot to reach 0.75), and 0 blocked nodes on
every uncrossed seed.

## Also reported (post hoc, descriptive): the twelve saved beta = 2 graphs

`python blocked_closure.py results/sheet_cross_b2_s3*_N4194304.npz --crossover` prints, per
seed, N_first and its node ids, N*_B, the ratio, the blocked count at 4.19 M, the number of
blocked clusters (groups within 10 hops), and whether the oldest boundary node is blocked.
No verdict is attached to this table; the three seeds already looked at (300, 305, 308) are
in it.
