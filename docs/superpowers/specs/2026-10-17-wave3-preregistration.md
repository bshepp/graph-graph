# Pre-registration — literature paths, wave 3: instruments and their known-answer controls

**Date:** 2026-10-17. **Status:** PRE-REGISTRATION, committed before any number below was
computed. Plan: `2026-10-17-literature-paths-plan.md`. Paths: B1 + D2 (spectral dimension
with critical-tree controls), A8 (spectral gap count), A9 (effective resistance for the
walker contrast), D3 (Newman-Watts collapse of the portal threshold), D6 (a random
triangulation as a d_H = 4 null), B4 (box covering).

**A8 correction, disclosed.** The survey's A8 also proposed decomposing "spikes" in the
Laplacian spectrum, citing `LORENTZIAN_SPIKE.md`. That document is a scoping note on the
causal graph of update events, not a spectral feature; the brief given to the literature
researchers described it wrongly (the coordinator's error). The spike half of A8 is
dropped; the gap count stays.

Tools: `instruments.py` (new, with `--validate`) for W3-a, b, c, e, f; W3-d uses
`shortcut_tolerance.measure_tolerance` with a new optional `max_radius` argument (default
None keeps the banked automatic radius, so every banked run is unchanged). Serial or at
most three workers.

## W3-a. Spectral dimension next to the window audit, with critical trees (B1 + D2)

Instrument: `spectral_flow` running dimension D(t) (lazy walk, Rademacher probes, frozen
2026-09-27 definitions), plus a saturation fit D(t) = a - b / (c + t) over
t_min = 16 to the last t with P~(t) > 10 / N; a is the plateau. Beside it, the window audit
(`window_stability.audit_adjacency`, 60 sources) reads the ball-growth d.

Graphs (one each): 2D torus 300 x 300; 3D torus 45^3; uniform random labelled tree,
N = 200,000 (Pruefer code; the critical Galton-Watson tree conditioned on size); uniform
spanning tree of a 300 x 300 torus (Wilson's algorithm); `sheet` c = 6, beta = 3, 512,000
(saved, seed 100); `sheet` beta = 2 seed 308 at 4.19 M (saved, crossed, ruffled);
`grown` cap 6, N = 200,000.

**Gate (instrument):** torus 2D a in [1.85, 2.15], 3D a in [2.7, 3.3], and both audits
DIMENSION DEFINED at 2 and 3 within 0.15. A failed gate voids W3-a.

**Frozen readings.**
- **AUDIT SEPARATES CRITICAL TREES** if the random tree's audit is DIMENSION DEFINED or
  MIXED with d in [1.75, 2.25] (its Hausdorff dimension is 2) while its spectral plateau a
  is in [1.15, 1.5] (theory 4/3). If the audit reads EXPONENTIAL on the tree, the audit
  rejects all trees and the reading is **AUDIT REJECTS TREES**.
- UST: audit d against [1.45, 1.8] (theory 8/5) and a against [1.05, 1.45] (theory 16/13);
  reported, no verdict.
- `sheet` beta = 3: a in [1.85, 2.1] (prediction). `grown`: no plateau (already banked;
  descriptive). Seed 308 at 4.19 M: descriptive; the stated expectation is that D(t) at
  long times rises above its short-time value, because a negatively curved annulus spreads a
  walk faster (40%).

**Predictions.** Gate passes (90%); AUDIT SEPARATES CRITICAL TREES (65%).

## W3-b. Spectral gap count (A8, gap half)

Full normalized-Laplacian spectrum (dense) at N = 5000 on: random 4-regular graph (control),
`triangular`, `sheet` (c = 6, beta = 3), `grown` cap 6, uniform random tree; seed 0 each.
f = fraction of eigenvalues in (0, lambda_edge) with lambda_edge = 1 - 2 sqrt(kbar - 1) / kbar
(the Kesten-McKay lower edge for the mean degree kbar).

**Frozen readings.** Control: f <= 0.005 on the random regular graph (gate). **GAP COUNT
SEPARATES** if f >= 0.05 on both `triangular` and `sheet` and f <= 0.02 on `grown`.
Otherwise the table is the result.

**Predictions.** Gate (85%); GAP COUNT SEPARATES (55%; `grown` is a 2-tree, not a regular
tree, so Kesten-McKay is only a guide for it).

## W3-c. Effective resistance behind the walker contrast (A9)

The banked E3 configuration exactly (`shortcut_walkers`: N = 1500, seeds 0-9, source random,
target the farthest node, offset portal between nodes 2 hops from each). Per seed and
substrate (`triangular`, `sheet`, `grown`): R_eff(source, target) with unit conductances,
without and with the offset portal, from the Laplacian pseudo-inverse; ratio
rho = R_none / R_offset (the commute-time ratio, since 2m changes by one edge).

**Frozen readings.** **RESISTANCE EXPLAINS THE CONTRAST** if the median rho is >= 3 on
`grown` and <= 1.3 on both 2D substrates, and on each substrate the median rho is within a
factor 2 of the banked classical hitting-time gain (6.7x, 1.4x, 1.3x).

**Prediction.** RESISTANCE EXPLAINS THE CONTRAST (70%).

## W3-d. Newman-Watts collapse of the portal threshold (D3)

`measure_tolerance` on `triangular`, N in {2000, 5000, 20000} x max_radius r0 in
{automatic, 8, 14}, seeds 0-2, counts {0, 2, 5, 10, 15, 20, 30, 40, 60, 80, 120, 160, 240,
320}. k* = first count whose seed-median `median_drift` exceeds 0.10 (the banked E2
definition). Newman-Watts (xi = (N/S)^(1/2) up to a constant) and the file's own damage
model (k* ~ N / B_r0, B_r0 ~ r0^2) both give k* proportional to N / r0^2.

**Frozen readings.** Least-squares slope s of ln k* on ln(N / r0^2) over the cells with a
finite k*: **NEWMAN-WATTS COLLAPSE** if s is in [0.7, 1.3] and the residual scatter is
below 0.35 in ln k*. **NO COLLAPSE** if s < 0.4 or > 1.6. Otherwise UNDECIDED. Also
reported: the banked N = 5000 automatic-radius cell must reproduce k* = 20 (a check of the
re-run, not a verdict).

**Prediction.** NEWMAN-WATTS COLLAPSE (60%).

## W3-e. A random triangulation as the d_H = 4 null (D6)

A triangulation of the sphere with N = 20,000 vertices, equilibrated by uniformly proposed
edge flips (a flip of edge ab with opposite vertices c, d is accepted when cd is not already
an edge and a, b keep degree >= 3; the proposal is symmetric, so the chain samples the
uniform measure on triangulations with N vertices), 150 sweeps of 3N proposals from an
Apollonian start. Seed 0. Compared with `sheet` beta = 3 at 512,000 (saved).

Measured on both, from 40 sources: ball volume V(r) and the hull boundary H(r) (the
ball's complement split into connected components; all but the largest are added to the
ball; H = nodes of the hulled ball adjacent to the largest component); exponents
d_V = d ln V / d ln r and s_HV = d ln H / d ln V over the middle third of the usable radii.

**Frozen readings.** **PERIMETER CRITERION BLIND** if s_HV is in [0.35, 0.65] on both graphs
while d_V differs by at least 0.8 between them (the random triangulation's finite-size d_V
is expected well above 2 but below 4). If s_HV differs by more than 0.2 between the two, the
reading is **PERIMETER DISCRIMINATES**.

**Prediction.** PERIMETER CRITERION BLIND (60%).

## W3-f. Box covering (B4)

Random sequential ball covering (pick a uncovered node uniformly, cover its ball of radius
r_B; N_B = number of balls) at r_B in {1, 2, 3, 4, 6, 8, 12, 16, 24}, five repetitions, on
`sheet` beta = 3 at 512,000 and `grown` cap 6 at 200,000. Model comparison on ln N_B
against r_B: power law (d_B) versus exponential.

**Frozen readings.** `sheet`: d_B in [1.8, 2.2] (prediction). `grown`: **EXPONENTIAL** if
the exponential RSS is below the power-law RSS, **FINITE d_B** otherwise.

**Prediction.** `grown` EXPONENTIAL (60%; with no hubs, box counting should follow ball
volume, unlike the scale-free networks of Song-Havlin-Makse).

## Reporting

A FINDINGS section dated when the results land; each verdict as frozen here, misses stated
as misses.
