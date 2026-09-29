# Pre-registration — the portal program on substrates that have a dimension

**Date:** 2026-09-28
**Status:** PRE-REGISTRATION, committed before any run. Step 3 of the owner-approved
plan. Owner's standing instruction: run every applicable method and compare.
Driver `portals_on_geometry.py`; walkers via `shortcut_walkers.py`.

## Why

Every banked portal result (tolerance, censorship, walkers, and the async censorship
checkpoint) was measured on `grown`, which the audit showed has no dimension: it is
tree-like, its distances are logarithmic, and "advantage = distance at injection" meant
something different from what was intended. Two substrates with a window-stable dimension
of 2 now exist: the triangular lattice and `sheet` at beta = 3 (crossover size ~5e8, far
above anything here). This step re-runs the three portal experiments on both, with
`grown` re-run under the identical code as the paired reference.

## What is already known (disclosed)

The banked `grown` numbers (FINDINGS "Portal experiments") and the substrate smoke test:
at N = 2000, triangular and sheet read d = 1.82, drift -0.02 / -0.05, Moran's I 0.86 /
0.92; `grown` reads defined 0.35, drift +0.38, unstable. Distances: diameter 65 / 57 / 31.

## Frozen protocol

Substrates: `triangular`, `sheet` (c = 6, beta = 3), `grown` (cap 6). Seeds 0-4 for E1,
0-2 for E2, 0-9 for E3, matched across substrates.

- **E1 censorship** (`shortcut_censorship.run_condition`, verbatim): N = 2000, 40 long
  portals (advantage >= 6) + 20 detour-2, conditions prune / ricci / triadic+prune,
  120 steps, prob 0.05. Observables per condition: long survival, detour-2 survival,
  mean removal step, rank(advantage, removal time), woven-in count, collateral.
- **E2 tolerance** (`shortcut_tolerance.measure_tolerance`, verbatim, now reporting
  `median_drift` and `window_stable`): N = 5000, k = 0, 5, 10, 20, 50, 100, 200, 400
  portals, 199 permutations.
- **E3 walkers** (`shortcut_walkers.py --nodes 1500 --seeds 10`, both generators):
  stationary occupancy, hitting time, horizon-free Pbar, for none / offset / direct.

## Predictions, frozen

- **P1 (threshold, advantage-blind) holds on every substrate.** It is a property of the
  rule, not the substrate: on triangular and sheet every fabric edge lies in a triangle,
  so collateral is 0; detour-2 survival ~1; long survival ~ e^-6; |rank corr| < 0.2.
- **P2 (triadic self-stabilization) survives on the 2D substrates**, with woven > 0 and
  long survival above prune-only. No prediction on its size relative to `grown`; the
  comparison is the result.
- **E2: portals do not inflate a real dimension, they destroy it.** On triangular and
  sheet, `d_eff` of the still-defined nodes rises with k as on `grown`, but
  `window_stable` turns False at some k* — reported per substrate — and `defined_frac`
  falls. The banked "dimension inflation 2.21 → 3.19, coherent throughout" is expected
  not to reproduce as a statement about a dimension. k* is predicted in the range
  10-50 at N = 5000, r0 = 10 (portal endpoints then sit within one measurement radius of
  most nodes).
- **E3: the qualitative walker result holds** (classical: kinetic, occupancy unchanged;
  quantum: Pbar gain > 10x) on every substrate. Classical hitting-time gain is
  predicted *larger* on the 2D substrates than on `grown`, because the portal's
  advantage is larger when distances grow like sqrt(N) instead of log N.

## Readings

Each prediction is read PASS / FAIL per substrate from the table; there is no composite
verdict. The deliverable is the comparison table with `grown` alongside.

## Known limits

N = 2000-5000 only, the banked sizes; the sheet at these sizes is a triangular-lattice
patch with a rough boundary, so triangular and sheet are expected to agree closely and
their agreement is a consistency check rather than two independent substrates.
