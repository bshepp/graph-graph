# Pre-registration — literature paths, wave 1: the dead corner in curvature and rate units

**Date:** 2026-10-17. **Status:** PRE-REGISTRATION, committed before any of the numbers
below were computed. Plan: `2026-10-17-literature-paths-plan.md`. Source of the paths:
`reports/Local graph growth literature survey.md` (A1, A2, A3, A6, A7).

What was already known at registration, and is therefore not a test: the per-draw onset
counts `A_star` and core fractions of the throat runs exist in CSVs (only their medians and
widths were read); the twelve saved beta = 2 graphs' first dead corner sizes, ids, blocked
counts and cluster counts (FINDINGS, dead-corner follow-up); that seed 308's first dead
corner was an adjacent pair; the eight beta = 1.5 first dead corner sizes. Those numbers
enter only the descriptive parts below. Every verdict-bearing reading uses a quantity not
computed before this commit, or fresh seeds.

## Definitions used throughout

- **Boundary edge**: an edge in exactly one triangle. **Boundary node**: has a boundary
  edge. `sheet` is a triangulated disc by construction (attachment hangs a triangle on a
  boundary edge; closure is refused when it would pinch), so its boundary should be one
  simple cycle: boundary node count B_v equals boundary edge count B_e, and every
  boundary node of degree d lies in d - 1 triangles.
- **Combinatorial curvature** (Keller-Peyerimhoff; Wu et al.), outer face as an
  infinigon: interior node kappa = (6 - d)/6, boundary node kappa = (4 - d)/6.
  Interior degree is 6 everywhere in `sheet`, so all curvature sits on the boundary.
  Gauss-Bonnet for a disc: sum over boundary nodes of (4 - d) = 6.
- **Dead corner** (= blocked node of the 2026-10-13 follow-up): boundary node of degree
  >= c = 6 for which `sheet_growth._closable` is None. Its **failure mode** is the set of
  `_closable` conditions that fail: (i) not exactly two boundary edges, (ii) the link ends
  a, b adjacent, (iii) a and b share a neighbour other than x (pinch), (iv) deg a >= c or
  deg b >= c (degree guard). Every condition is permanent once failed (degrees and
  adjacency never shrink, and a node at the cap never gains an edge), so a dead corner
  stays dead and on the boundary.

## W1-a. Throat onset as an integer mixture (A7)

Inputs: the nine 2000-draw FSS CSVs (grown r = 2, 3, 4 of 2026-08-18; triangular and sheet
r = 2, 3, 4 of 2026-10-13). Per finite draw, the core size at onset is
m = round(A_star * core_frac) (an integer: `core_frac = len(core) / A_star`).

Instrument check: m >= 2 in every draw (a single strand has no common neighbour because the
two balls are far apart, so the smallest mutually protecting set is two strands). A draw
with m < 2 is a bug and voids W1-a.

**Frozen readings.**
- **INTEGER MIXTURE** if P(m = 2) + P(m = 3) >= 0.80 in every one of the nine cells;
  **BROAD** if it is below 0.80 in at least three cells; otherwise UNDECIDED.
- **FRACTION UNIVERSAL** if max - min of P(m = 2) over the nine cells is <= 0.15;
  otherwise **FRACTION VARIES** (then the table by substrate and capacity is the result).

**Predictions.** INTEGER MIXTURE (75%); mode m = 2 in every cell (60%); FRACTION UNIVERSAL
(50%). The survey's guess was a 40/60 mixture of 2 and 3.

## W1-b. Curvature and failure-mode census (A1)

Inputs: the twelve saved beta = 2 graphs at 4,194,304 nodes (seeds 300-311) at prefix sizes
2^12 ... 2^22 and at each seed's first dead corner size; descriptively also the 512,000-node
c = 6 graphs at beta = 0, 1, 2, 3 (three seeds each). Tool: `curvature_census.py` (new).

Per checkpoint: B_v, B_e, the boundary degree histogram (d = 2 ... 6), the Gauss-Bonnet sum,
interior degree check, the dead corner count split by failure mode, and the negative
curvature mass M_- = sum of max(0, d - 4)/6 over boundary nodes.

Instrument gate: B_v = B_e, every interior node has degree 6, and the Gauss-Bonnet sum is
exactly 6 at every checkpoint of every beta >= 1 graph. If it fails anywhere, the sheet is
not a disc there and the curvature reading of that graph is void (reported, not repaired).

**Frozen readings.**
- **DEGREE GUARD** if, in at least 8 of the 9 crossed seeds, the first dead corner is a
  pair of adjacent boundary nodes, both of degree 6, each failing (iv) and nothing else.
  **PINCH** if (iii) is among the failures of the first dead corner in at least 2 of the
  9. Otherwise MIXED.
- At 4.19 M, the fraction of all dead corners failing (iv) only: reported against the
  prediction >= 0.80.

**Predictions.** DEGREE GUARD (75%); (iv)-only fraction >= 0.80 at 4.19 M (65%); boundary
degree composition (fraction at d = 5, 6) shifts toward high degree after the first dead
corner (descriptive, no rule).

## W1-c. Is the sheet past N* still a disc? (A6)

The banked text says `sheet` "grows tree-like arms" past N* (FINDINGS, 2026-09-29). Round 2
item 1 found no arm (the edge roughens and the core stays intact), the window audit of seed
300 at 4 M read d = 1.84-1.99 out to radius 400, and the survey notes that with every
interior degree equal to 6 the bulk cannot grow faster than r^2 (Hua-Jost-Liu, abstract
only). The boundary carries B_e / N ~ 0.6% of the nodes at 4 M, so the label cannot be about
mass. This tests whether anything at large scale is not two-dimensional.

Inputs: the nine crossed beta = 2 graphs at 4.19 M (the three compact seeds 305, 309, 310
as controls). Tool: `curvature_census.py --disc`.

1. **Large-radius window audit**: 30 uniformly sampled sources, ball curves to radius 480,
   the window-stability rule of `window_stability.py` (drift <= 0.15 and power law beats
   exponential) over windows 10, 20, 40, 80, 160, 240, 320, 480, each used only by
   sources unsaturated (< 0.1 N) at that radius.
2. **Radius growth**: eccentricity of node 0 (the seed triangle, the disc's centre) and
   the median hop distance from node 0 to the boundary, at the prefix octaves; per-octave
   exponent against N.

**Frozen readings.**
- **STILL A DISC** if the audit is DIMENSION DEFINED or MIXED (not EXPONENTIAL) with
  median d at the largest usable window in [1.8, 2.1] on at least 8 of the 9 crossed
  graphs, and the eccentricity exponent over the last octave (2.1 M -> 4.2 M) is in
  [0.40, 0.65] on at least 7 of the 9.
- **DISC WITH A STALLED RADIUS** if the audit condition holds but the eccentricity
  condition fails.
- **TREE-LIKE CONFIRMED** if the audit reads EXPONENTIAL on at least 5 of the 9.
- Otherwise UNDECIDED.

**Prediction.** STILL A DISC (60%). If it holds, the banked "tree-like beyond N*" label is
revised to "a disc whose boundary stops scaling as sqrt(N)". The 2026-09-29 diameter
exponent of 0.25 in the last octave would then need its own explanation (the double-sweep
lower bound is the first suspect); the eccentricity exponent is the check.

## W1-d. The dead-corner hazard, measured directly (A2 + A3)

**Instrument.** An optional `on_event` hook in `sheet_growth.grow_sheet`, called after
every event with the event's kind and the nodes whose degree or adjacency changed; the
default is None, the hook does not touch the random stream, and growth with the hook is
bit-identical to growth without it. The detector checks every changed node and all their
neighbours and records each new dead corner the moment it appears: node count N, node id,
failure mode, the event that created it (attach or close, its endpoints and their degrees),
and the degrees of the node's boundary neighbours. A hook return of True stops the growth
(used to stop at the first dead corner).

Validation (`--validate`, before the scan): with and without the hook, the same seed gives
the same graph; on small growths at beta = 1.0 the incremental dead set equals
`blocked_closure.blocked_nodes` on the final graph, and the first dead corner size equals
`blocked_closure.first_blocked` bisection on the saved prefix.

**Scan (fixed here).** beta in {1.0, 1.25, 1.5, 1.75}, seeds 500-539 (40 per beta), stop at
the first dead corner or at n_max = 131,072 / 524,288 / 2,097,152 / 4,194,304 respectively
(right-censored there). Ten worker processes on the desktop. The twelve beta = 2 seeds (all
seen) are shown beside the fit as a fifth, non-confirmatory point.

**Estimators.** Per beta: exposure MLE of a constant per-node hazard
lambda_hat = d / sum_i min(N_first_i, n_max), chi-square 95% interval; censored Weibull MLE
of the shape k with a profile-likelihood 95% interval; Nelson-Aalen cumulative hazard
plotted against N (straight line = constant hazard). Pooled: common shape with per-beta
scale.

**Frozen readings.**
- **P1, the clock.** **PER-NODE HAZARD** if the shape interval covers 1 at >= 3 of the 4
  betas and the pooled shape interval covers 1. **BOUNDARY-SITE HAZARD** if the pooled
  interval excludes 1 and covers 1.5 (a per-boundary-site hazard gives cumulative hazard
  ~ N^1.5 because B ~ sqrt(N)). Otherwise UNDECIDED.
- **P2, the Arrhenius law.** Weighted least squares of ln lambda_hat on beta (weights from
  the Poisson variance 1/d). **ARRHENIUS** if a quadratic term's 95% interval covers 0.
  The slope a (lambda ~ exp(-a beta)) is then compared with the N*_B law's 5.38:
  **SAME LAW** if |a - 5.38| <= 1.0, otherwise **RATE LAW DIFFERS** (the law is then
  restated in rate units).
- **P3, the mechanism.** **LOST RACE** if, in >= 90% of the seeds with a dead corner, the
  first one is an adjacent pair of degree-6 boundary nodes failing (iv) only, created by an
  attachment event. Otherwise the table of creating events and modes is the result.

**Predictions.** PER-NODE HAZARD (65%); ARRHENIUS (70%) with SAME LAW (55%); LOST RACE
(75%). The mechanism behind LOST RACE, stated in advance: a degree-6 boundary node a waits
to close (rate exp(12 beta)) while its degree-5 boundary neighbour x receives an attachment
on its other boundary edge (x, q) (rate exp(beta (5 + deg q))); if the attachment wins, x
and a are both capped and each blocks the other forever. The single race has odds about
exp(-beta (7 - deg q)), so a slope a near 5.4 would need the precursor configuration itself
to be suppressed by about exp(-3 beta). The creating events' degrees are reported so this
decomposition can be checked.

## Reporting

Results go into a FINDINGS section dated when they land, with every verdict read as frozen
here, misses stated as misses, and the tables behind them. Tools: `curvature_census.py`
(new, with `--validate`) and `dead_corner_hazard.py` (new, with `--validate`).
