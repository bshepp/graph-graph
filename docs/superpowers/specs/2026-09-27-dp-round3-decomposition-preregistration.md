# Pre-registration — d(p) fine structure, round 3: exact per-radius decomposition

**Date:** 2026-09-27
**Status:** PRE-REGISTRATION, committed before the ball counts are computed. Owner
delegated the choice of round-3 formulation (2026-09-26).

## Why a decomposition instead of a third mechanism guess

Rounds 1 and 2 each proposed a mechanism and were killed. Round 3 narrows the space
first. The 2026-09-26 audit showed that above p ~ 0.2 the pruned graph has no dimension,
so the banked d(p) is a radius-10 reading of ball counts. That reading is an **exactly
linear** functional of the log ball counts: with design matrix X (columns `log r`, `1`,
`1/r`, r = 1..10), the fitted slope is

    d = sum_r w_r * ln|B(r)|,     w = first row of pinv(X).

So the waveform in d(p) splits, with no modelling, into one contribution per radius.

## Frozen protocol

- Graphs: exactly the banked dense grid — N = 32000, k = 6, p on
  `geomspace(0.02, 0.5, 30)`, seeds 0..5, `prune_to_convergence`, 400 sampled nodes,
  `max_radius = 10` — rebuilt through the same code path as
  `prune_dimension.measure_pruned` so the sampled nodes are the same.
- `d_med(p)`: seed-mean of the per-seed median `d_eff` (the banked object).
- `d_lin(p)`: `sum_r w_r * L_r(p)`, with `L_r(p)` the mean over nodes and seeds of
  `ln|B(r)|`.
- Residuals: each curve minus its cubic least-squares fit in `ln p`. Detrending is
  linear, so `resid(d_lin) = sum_r c_r` exactly, with `c_r = w_r * resid(L_r)`.
- Variance share of a radius set S: `cov(sum_{r in S} c_r, resid d_lin) / var(resid
  d_lin)`. Shares over a partition sum to 1.

## Gates and frozen readings

- **Gate R (reproduction):** `d_med` per (p, seed) must equal the banked
  `results/prune_dimension_dense32k_20260811.csv` to 1e-9. If not, the run is measuring
  different graphs and stops.
- **Gate M1 (validity):** Pearson correlation of `resid(d_med)` with `resid(d_lin)` must
  be >= 0.8. If not, the median-of-fits and the fit-of-means are different objects, the
  decomposition does not describe the banked waveform, and no reading is made.
- **Readings** (exactly one applies):
  - **LOCAL**: share of r <= 3 is >= 0.7 — the waveform lives in the immediate
    neighbourhood (degree and second-shell counts): a micro-motif statistic.
  - **MESOSCALE**: share of r >= 4 is >= 0.7 — it lives in how surviving shortcuts are
    reached.
  - **DISTRIBUTED**: neither.
- **Amplification check (frozen threshold):** let `A = max_r |w_r|` and let `s` be the
  largest over r of the peak-to-peak of `resid(L_r)`. If the peak-to-peak of
  `resid(d_lin)` exceeds `3 * s`, the waveform is reported as **AMPLIFIED** — a small
  structure in the counts magnified by the fit's ill-conditioned weights — alongside
  whichever reading applies. Seed-to-seed scatter of `resid(L_r)` is reported so the
  per-radius signal can be compared with its own noise.

## What this round cannot do

It locates the waveform in radius; it does not name a mechanism. A mechanism hypothesis
for the radii that carry it is round 4, with its own pre-registration.
