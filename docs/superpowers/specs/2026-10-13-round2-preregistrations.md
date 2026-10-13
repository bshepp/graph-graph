# Pre-registrations — round of 2026-10-13 (four items, run together and compared)

**Date:** 2026-10-13. **Status:** PRE-REGISTRATION, committed before any of the four runs.
Owner's standing instruction: implement everything that can be implemented, then compare.
Owner is remote; everything here runs on the desktop (jaga is not required for any item).

---

## Item 1 — What nucleates the arm? (`arm_birth.py`)

**Question.** `sheet` at beta = 2 stays exactly compact and then, within one octave, grows a
tree-like arm, at a size that varied 0.47-1.09 M across four seeds. What is at the base of
the first arm?

**Data.** The four saved 4,194,304-node graphs (`results/sheet_cross_b2_s30{0-3}_N4194304.npz`)
plus the eight new seeds from item 4 as they land. Node index is growth order, so the graph
at size N is approximated by the subgraph induced on nodes < N (closure edges added later
between early nodes are included; this biases the boundary low by a few percent and is
disclosed).

**Frozen definitions.**
- Root distance `rho(v)`: BFS distance from node 0 (the seed triangle).
- Boundary at size N: edges of the prefix graph lying in exactly one triangle; boundary
  nodes are their endpoints.
- Disc radius `R(N)`: median root distance of boundary nodes.
- **Arm**: a connected component of the set of boundary nodes with `rho > 1.5 R(N)`. The
  **first arm** is the earliest prefix size (checked at 2^(1/8) spacing from 2^17) at which
  an arm of >= 50 boundary nodes exists; its **base** is the arm's node of smallest `rho`.
- **Blocked closure**: a boundary vertex of degree >= 6 whose wheel cannot be closed under
  the generator's own rule (`sheet_growth._closable` returns None).

**Measurements at the first-arm size, per seed:** (a) root distance of the base vs R(N);
(b) density of blocked closures within 10 hops of the base, against the density on the
rest of the boundary; (c) degree histogram of the base's 10-hop boundary neighbourhood
against the rest of the boundary; (d) the arm's width: nodes in the arm per unit of its
root-distance extent.

**Instrument amendment (before any real graph was analysed):** validation on a planted
strip showed the "base = lowest node above 1.5 R" lands at the detection threshold, not at
the arm's root. The base is now the lowest node of the arm's component among boundary
nodes above 1.2 R, and the census radius is 20 hops instead of 10, so the root is inside it.

**Frozen readings.**
- **DEFECT**: blocked-closure density at the base >= 3x the rest of the boundary in at least
  3 of 4 seeds (and >= 8 of 12 once item 4 lands).
- **NOT A DEFECT**: the ratio is < 1.5 in at least 3 of 4 seeds.
- **UNDECIDED**: otherwise.

**Prediction.** DEFECT, at about 60%: the degree guard (`len(adj[a]) >= c` blocks closure)
is the one place where the rule can leave a crack that no later event repairs.

---

## Item 2 — Throat criticality on substrates with a dimension (`throat_criticality.py --topology`)

**Question.** The stage-1 verdict (threshold = a local motif onset; onset core ~2.6 strands
at every capacity; relative width rises) was measured on `grown` balls. Does it hold on
balls of a real plane?

**Protocol.** `throat_criticality.py --topology {triangular, sheet}` for `--pilot
--draws 300`, `--fss --draws 2000` (r = 2 / 3 / 4 at N = 2000 / 5000 / 10000, the stage-1
geometries) and `--anchor` at the stage-1 frozen a-values with `--seeds 8 --rider`.
Everything else as in stage 1 (seeds, rules, sweeps). Substrate regeneration never triggers on
these substrates and is reported as 0.

**Frozen anchor a-values** (pilot quantiles 10/30/50/70/90, fixed before the FSS and anchor
runs): triangular 0.0111 / 0.0187 / 0.0249 / 0.0307 / 0.0439; sheet 0.0111 / 0.0194 / 0.0249 /
0.0307 / 0.0443 (frozen 2026-10-13 from the sheet pilot before its FSS and anchor runs).

**Predictions.**
- Capacities are larger than on `grown` (a 2D ball of radius 2 has ~19 nodes against ~12),
  so A* in strands is larger too.
- **The motif-onset reading holds**: `core_frac @ A* x A*50` (the onset core in strands) is
  between 2 and 3.5 at every capacity; relative width `w / a50` does not fall with capacity
  (rises or flat within 10%).
- Gate 1 (peeling == dynamics) holds exactly (40/40) and Gate 2 (triadic demolishes
  cores) holds, as both are properties of the rules.

**Readings, per substrate.** MOTIF ONSET HOLDS if both bullets of the second prediction
hold; SUBSTRATE-DEPENDENT if the onset core differs from `grown`'s 2.6 by more than 1.0
strand or the relative width falls by more than 10% across capacities; otherwise UNDECIDED.

---

## Item 3 — Banked numbers re-run under the window-stability gate

**Protocol.** `python coherence.py` (defaults: N = 5000, caps 6, 3 seeds, 999 permutations)
and `python prune_dimension.py` (defaults: N = 2000 / 8000 / 32000, ten p values, 3 seeds),
with `prune_dimension` now reporting `median_drift` and `window_stable` per cell.

**Predictions.**
- `grown` reads "field fragmented (N/A)" in coherence, defined_frac ~0.27; lattice unchanged.
- d(p): defined_frac ~1 and window-stable for p <= 0.1; defined_frac falling and
  window-unstable for p >= 0.3 at every N; the median `d_eff` of the still-defined nodes
  no longer rises smoothly to ~2 but is dominated by survivors and is reported as such.

**Reading.** The log's two tables are replaced by the gated ones with the old values kept
under the supersession notice; no verdict beyond PASS/FAIL per prediction.

---

## Item 4 — Distribution of the crossover size at beta = 2 (`sheet_crossover.py --seeds 304..311`)

**Protocol.** Eight fresh seeds 304-311, beta = 2, N to 4,194,304, same checkpoints and
estimators as step 2. Combined with seeds 300-303: twelve values of N*_B.

**Frozen analysis.** On the twelve per-seed N*_B: coefficient of variation CV = sd / mean;
and the fraction of seeds censored (no crossover by 4.19 M).

**Readings.**
- **MEMORYLESS**: CV >= 0.7 (an exponential has CV = 1; a hazard rising with boundary
  length gives CV between ~0.5 and 1).
- **SHARP THRESHOLD**: CV <= 0.3.
- **IN BETWEEN**: otherwise, reported with the CV.
The four known values (0.47, 0.89, 0.90, 1.09 M) have CV 0.31; the reading waits for twelve.

**Prediction.** IN BETWEEN with CV 0.3-0.6: a nucleation whose rate per unit boundary
rises as the boundary grows is neither memoryless in N nor a threshold.

---

## Comparison at the end

One table: for each item, prediction, outcome, and what it changes in the description of
`sheet` and of the portal program.
