# Pre-registration — literature paths, wave 4: the fixed-N rules against their literature

**Date:** 2026-10-17. **Status:** PRE-REGISTRATION, committed before any number below was
computed. Plan: `2026-10-17-literature-paths-plan.md`. Paths: D5, C8, C9, C7, C10, B2.

**D4 dropped, disclosed.** The survey's D4 ("barrier-law discrimination for `activation`")
assumed the banked bootstrapping barrier is a bootstrap-percolation threshold of the
activation dynamics. It is not: `activation` is an SIS epidemic, and the banked barrier is
the failure of fixed-N rewiring (`triadic`, `geometrize`, `ricci`) to grow extent from a
random graph (extent ~ N^0.13-0.17 against 0.515 on a lattice; FINDINGS, step 2 and step 5).
The droplet law has nothing to test there. C10 below is the path that addresses the actual
barrier.

All variants live in `fixed_rules.py` (new, with `--validate`) as labelled experiment
dynamics. None is registered in `rules.py`, so the two backends and every banked result are
unchanged. Locality holds in every variant (a node reads itself and its neighbours; the
Ollivier curvature of an edge reads the two endpoints' neighbourhoods and distances among
them).

## W4-a. Majority: the tie rule and the schedule (D5)

`majority` reads neighbours only and breaks ties by argmax, i.e. toward state 0 (a known
asymmetry; memory note "majority rule tie-break bias"). Here on a 2D torus 100 x 100 (degree
4, so 2-2 ties occur): initial state 1 with probability p1 in {0.5, 0.6, 0.7, 0.8, 0.9}; tie
rules **keep** (keep own state), **bias0** (the project's), **random**; schedules
**synchronous** and **sequential** (uniform random order each sweep); noise 0 and 0.01; up
to 200 sweeps; 4 seeds per cell.

**Frozen readings.**
- **Gate (Goles-Olivos):** every synchronous noise-0 run ends in a fixed point or a 2-cycle,
  and every sequential noise-0 run in a fixed point. A failure is a bug.
- **BIAS TAKES OVER** if, with bias0, synchronous, noise 0, the final fraction of state 0
  exceeds 0.9 at p1 = 0.7 (30% zeros at the start) in all 4 seeds, while with keep the
  final fraction stays within 0.1 of its initial value. Otherwise the table is the result.
- Descriptive: at p1 = 0.5, noise 0.01 (the banked setting), the final mean of state 0 under
  bias0 against keep, which quantifies how far the banked majority runs were pushed by the
  tie rule.

**Predictions.** Gate (95%); BIAS TAKES OVER (55%).

## W4-b. `triadic` in the long run, and closure with a loss term (C8)

(1) The registered `triadic` rule from a random graph (mean degree 6), N in {1000, 4000},
3 seeds, 5000 sweeps; double-sweep diameter of the largest component, its size, and
clustering every 100 sweeps. BRANCHES records extent peaking near 300 sweeps and falling
below its start (8.5 -> 11.5 -> 4.0 at N = 1000). **COLLAPSE** if in every run the diameter
at 5000 sweeps is below the starting diameter and clustering exceeds 0.5. Prediction:
COLLAPSE (70%; closure without a loss term ends in dense clusters, Jin-Girvan-Newman).

(2) A Marsili-Vega-Redondo-Slanina variant: per sweep each node adds one triadic edge (closes
a random open triad at itself) with probability xi, adds one uniformly random edge with
probability eta = 0.0005, and every edge is deleted with probability lambda = 0.01. N = 1000,
xi swept 0 -> 1 -> 0 in steps of 0.05, 300 sweeps per step, the state carried over.
**HYSTERESIS** if at some xi the mean degree on the up-sweep and the down-sweep differ by a
factor of 2 or more. Prediction: HYSTERESIS (55%).

## W4-c. Anti-Hebbian rewiring self-organizes (C9)

Bornholdt-Rohlf: a random threshold network, N = 500, directed links with weights +/-1,
sigma_i(t+1) = sign(sum_j w_ij sigma_j(t)) (sign(0) = +1); per epoch: random initial state,
200 steps, then one random node i: if sigma_i did not change over the last 100 steps (frozen)
it gains an incoming link from a random node, else it loses a random incoming link. 5000
epochs from mean in-degree K0 = 1 and K0 = 4, two seeds each. **SELF-ORGANIZED** if the mean
K over the last 1000 epochs agrees between the two starts within 0.5 and lies in [1.5, 3.0].
Prediction: SELF-ORGANIZED (60%).

## W4-d. Ricci flow with surgery as a shortcut detector, and on a random graph (C7)

Ni-Lin-Luo-Gao flow: Ollivier curvature with idleness alpha = 0.5 on weighted shortest-path
distances; edge weights are lengths, w <- w - kappa w per iteration, then rescaled to keep
the total weight; 20 iterations; surgery cuts every edge whose final weight exceeds twice the
median. Graphs: `triangular` N = 2000 plus 40 shortcuts (endpoints at least 2 r0 apart, the
banked injection), 3 seeds; a random graph N = 1000, mean degree 6, 3 seeds.

**Frozen readings.**
- **SHORTCUT SURGERY** if, on the lattice-plus-shortcuts graphs, at least 90% of the
  shortcuts are cut and at most 2% of the lattice edges are, in every seed.
- Random graph: **FRAGMENTS, NO DIMENSION** if after surgery the graph has at least two
  components of 50 or more nodes, or its largest component is not window-stable 2D-like
  (window audit not DIMENSION DEFINED).

**Predictions.** SHORTCUT SURGERY (65%); FRAGMENTS, NO DIMENSION (90%).

## W4-e. Can a thermal ingredient cross the barrier? Hard-core squares (C10)

Kelly-Trugenberger-Biancalana's flatness motif is the square: on a triangle-free 4-regular
graph, Z^2 has exactly two squares on every edge. Energy E = sum over edges of |s_e - 2|
(s_e = number of 4-cycles through e); moves are degree-preserving double-edge swaps that keep
the graph simple and triangle-free; N = 900 (a 30 x 30 torus is the target's size), start from
a random 4-regular triangle-free graph. **Annealed**: Metropolis with T falling geometrically
from 2.0 to 0.05 over 1000 sweeps (sweep = 2N proposals). **Quench**: T = 0, same number of
proposals. 4 seeds each. Measured: final E / edges, phi = fraction of edges with s_e = 2,
mean ball volume at r = 2 ... 8 against the 30 x 30 torus.

**Frozen readings.**
- **THERMAL ROUTE WORKS** if the annealed runs reach phi >= 0.9 and mean ball volume within
  10% of the torus's at r = 6 in at least 2 of 4 seeds, while the quench runs stay at
  phi <= 0.6. Then the barrier is about the missing thermal ingredient, not locality.
- **NO ROUTE** if the annealed runs reach phi < 0.6 in all seeds.
- Otherwise UNDECIDED.

**Prediction.** UNDECIDED (40%), THERMAL ROUTE WORKS (35%), NO ROUTE (25%).

## W4-f. Ollivier curvature of the sheet's dead corners (B2)

Ollivier curvature with idleness 0 (uniform measures on neighbours; hop distances) on every
edge of two 50,000-node sheets: beta = 1.0 (seed 0, crosses early; has dead corners) and
beta = 3.0 (seed 0, compact). Instrument check: edges with both ends interior take one common
value (the triangular lattice's), to within 1e-9. **DEAD CORNERS ARE THE MOST NEGATIVE** if
the edge joining the two nodes of a dead-corner pair is below the 1st percentile of all
edges' curvature in at least 90% of pairs. Descriptive: the curvature distribution's negative
tail, crossed vs compact, against Kelly-Trugenberger-Biancalana's claim that defects are
suppressed macroscopically.

**Prediction.** DEAD CORNERS ARE THE MOST NEGATIVE (70%).

## Reporting

A FINDINGS section dated when the results land; each verdict as frozen here, misses stated as
misses.
