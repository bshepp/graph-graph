# Pre-registration — literature paths, wave 2: the ruffle, the fjords and the edge

**Date:** 2026-10-17. **Status:** PRE-REGISTRATION, committed before any number below was
computed. Plan: `2026-10-17-literature-paths-plan.md`. Wave 1 results (FINDINGS,
"Literature paths, wave 1") set the questions: past its first dead corner the sheet is
two-dimensional in every audit window, but its centre's eccentricity grows as N^0.23-0.30
and the outer annulus packs up to 4x a flat disc's mass; every dead corner is an adjacent
capped pair failing the degree guard, and it can never become interior.

A topological consequence, stated before measuring: a dead corner stays on the boundary
forever and the boundary is one simple cycle, so the disc can never close over it. A
channel from the dead corner to the outside must persist, and its depth must grow as the
disc grows around it. A fjord at every dead corner is therefore forced by the rule, not an
instability. Wave 2 tests that reading, measures what the ruffle is, and supplies the null
model.

Inputs unless stated: the twelve saved beta = 2 graphs at 4,194,304 nodes (nine crossed,
three compact), the wave 1 census CSV for dead-corner ids. Tool: `edge_geometry.py` (new,
with `--validate`) and `eden_pin.py` (new, with `--validate`). All runs serial or at most
three workers (wave 1's hazard scan was killed for low memory).

## W2-a. The ruffle as cone excess (new; follows W1-c)

Each dead corner is a boundary node of degree 6 (five triangles, interior angle 5 pi/3 in
units where a triangle corner is pi/3), so its boundary turns by -2 pi/3 instead of 0.
Model: grown around, a dead corner acts like a cone point of excess angle 2 pi/3, so every
shell beyond it is longer than a flat shell by one third of a full turn per dead corner
enclosed.

Measured: shells S(r) = |B(0, r)| - |B(0, r - 1)| around node 0 (the seed triangle); the
reference S_ref(r) is the mean over the three compact seeds; X(r) = S(r) / S_ref(r) - 1;
D(r) = number of dead corners within hop distance r of node 0. Radii r from 50 to
0.7 x ecc(node 0) of the graph (beyond that the shells are cut by the edge).

**Frozen readings.**
- Per crossed graph, least-squares slope g of X(r) against D(r) / 3 (through the origin).
  **CONE EXCESS** if g > 0 on all nine and the median g is in [0.3, 1.5] (each dead corner
  adds roughly its own one-third turn). **NO CONE EXCESS** if the median g is below 0.1 or
  X(r) rises before any dead corner is enclosed (D(r) = 0) on at least three graphs.
  Otherwise UNDECIDED.
- Control: X(r) stays within +/- 0.05 of 0 on the compact seeds by construction of S_ref
  (leave-one-out reference, reported).

**Prediction.** CONE EXCESS (50%; the factor is the uncertain part, the sign is not).

## W2-b. Fjords are forced at dead corners (A5)

A **neck** is a pair of boundary nodes at hop distance <= 3 whose distance along the
boundary cycle is >= 100 boundary nodes; necks within 10 hops of one another are merged
into one **mouth**, and the boundary arc they cut off (the shorter side) is its **fjord**.
Per fjord: arc length, depth (largest hop distance from the mouth to a node on the arc),
and the number of dead corners on the arc. The census is taken on every final graph and at
the prefix octaves.

**Frozen readings.**
- **FORCED FJORDS** if (1) at least 90% of all fjords in the crossed final graphs contain
  at least one dead corner, and (2) the compact final graphs (305, 309, 310) and every
  crossed graph at the last octave before its first dead corner have no fjord at all.
  **FJORDS NOT TIED TO DEAD CORNERS** if fewer than 60% contain one. Otherwise UNDECIDED.
- Descriptive: the Barabasi-Vicsek neck-to-depth distribution n(x) (x = mouth width /
  depth) and the depth / R ratio of the deepest fjord against wave 1's 0.75.

**Prediction.** FORCED FJORDS (70%).

## W2-c. Local versus spectral roughness (A4)

Height profile h(s) along the boundary cycle = hop distance from node 0 minus its median.
Local width w(l) = root-mean-square of h within windows of l consecutive boundary nodes,
averaged over windows; local roughness exponent alpha_loc from w(l) ~ l^alpha over
l = 8 ... B/20. Spectral exponent alpha_s from the existing `edge_spectrum` power-law fit
with k = 6, 12, 18 removed (S(k) ~ k^-(1 + 2 alpha_s)).

**Frozen readings** (compact edges = the three compact seeds at 4.19 M and each crossed
seed at the last octave before its first dead corner; rough edges = crossed seeds at 4.19
M):
- **KPZ-LOCAL COMPACT EDGE** if alpha_loc is in [0.4, 0.6] on at least 80% of compact
  edges. **ANOMALOUS** additionally if alpha_s - alpha_loc >= 0.5 on those edges (local
  KPZ roughness under a shape-dominated spectrum, the survey's reading of the -3.4 to -4.7
  slopes).
- Rough edges: alpha_loc reported against the band [0.5, 0.9] (free KPZ to the
  Eden-with-obstacles value), no verdict.

**Prediction.** KPZ-LOCAL COMPACT EDGE (55%) and ANOMALOUS (50%).

## W2-d. The boundary as a process (B3)

Descriptive only: B / sqrt(N) at the octaves across the twelve seeds was already printed in
wave 1 (3.73-3.81 at 16 k; 3.83-3.86 compact at 4.19 M), so it cannot be a test. Reported:
the across-seed spread of B / sqrt(N) by octave before the first dead corner, against
Sisko-Yambartsev-Zohren's diffusive boundary for layered causal growth (which predicts a
spread growing like N^(1/4) in relative terms of B / sqrt(N), not shrinking).

## W2-e. The null model: Eden with one frozen site (D1)

Eden growth (occupy a uniformly chosen perimeter site) on a triangular lattice graph built
from adjacency only (hop distances; no positions are read). When the cluster's hop radius
reaches R0 = 20, one perimeter site at radius R0 is frozen (never occupied). Two variants,
20 runs each to radius 200 (about 1.2e5 sites):

- **Plain**: nothing else changes. The frozen site may be enclosed (become a hole).
- **No enclosure**: an occupation that would cut the frozen site off from the lattice's
  outer frame is refused (the frozen site, like a dead corner, must stay on the edge).

Measured per run when the cluster's hop radius R (largest seed distance of an occupied
site) reaches 40, 80, 120, 160, 200: **exposed** = the frozen site is connected to the
lattice's outer frame through unoccupied sites; **channel width at half depth** = the
number of unoccupied sites at seed distance rho = (R0 + R) / 2 that are connected to the
outer frame (inside a compact Eden cluster every site at rho is occupied except channels
and enclosed holes, and holes are not connected to the frame). Channel depth is R - R0
while the site is exposed.

**Frozen readings.**
- **PLAIN EDEN HEALS** if the frozen site is enclosed by radius 40 in at least 95% of plain
  runs.
- **CONSTRAINED EDEN CRACKS** if in the no-enclosure variant the channel survives to radius
  200 in every run with width at half depth <= 3 sites in at least 90% of runs (a thin crack,
  not a wide fjord).
- The sheet's fjords are compared descriptively (W2-b mouth widths are <= 3 by definition;
  their arc length per unit depth is the comparison).

**Predictions.** PLAIN EDEN HEALS (85%); CONSTRAINED EDEN CRACKS (65%). If both hold, the
sheet's fjords are the topological channel a permanent boundary point forces, and their
extra width (arc per depth) is what the sheet's own growth adds.

## Reporting

A FINDINGS section dated when the results land, each verdict read as frozen here, misses
stated as misses.
