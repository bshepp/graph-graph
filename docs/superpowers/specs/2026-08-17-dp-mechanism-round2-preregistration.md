# Pre-registration: d(p) multi-bump mechanism, round 2 — integer crossings of the ring→mesh crossover radius

**Date:** 2026-08-17 (committed BEFORE the data below existed)
**Context:** FINDINGS "Overnight pre-registered tests (2026-08-12)": the pruned-WS d(p)
fine structure is real, deterministic, density-intrinsic, N-invariant — and NOT
log-periodic (P1 refuted) — with no surviving mechanism. This registers round 2's
candidate before testing it, per the BRANCHES discipline (formulate → pre-register →
test). Run overnight on owner's blanket authorization ("anything you need to do").

## The candidate mechanism

The pruned graph interpolates ring (local slope of ln B(r) ≈ 1) and mesh (slope ≈ 2)
regimes, with a crossover radius r_c(p) that shrinks as p grows (more surviving
shortcuts → mesh behavior sets in earlier). The d_eff estimator fits ln B over
*integer* radii; as r_c sweeps down through successive integers, the fit's regime
mixture changes non-smoothly, modulating d_eff. Why this fits everything known:

- **Window-robust:** all fit windows (max_radius 8/10/12) see the same two-regime
  curve — the modulation source is r_c itself, not the window edge.
- **N-invariant:** r_c is set by surviving-shortcut density (a p-property), not N.
- **Not log-periodic:** integer crossings of a power-law-ish r_c(p) give feature
  spacings ln(p_{k+1}/p_k) ∝ ln((k+1)/k) — SHRINKING gaps at larger p, not constant
  ones. Observed: ln-gaps 1.61 → 1.28 (ratio 1.26); consecutive-integer prediction
  for k=3→4→5: ln(4/3)/ln(5/4) = 1.29.

## Measurement (frozen before data)

Ensemble: identical construction + convergence to `prune_dimension`
(`create_initial_graph(..., 'small_world', p=p)` then `prune_to_convergence`),
N=32000, the 30-point geomspace [0.02, 0.5] grid, 3 seeds. Per (p, seed): mean
|B(r)| over 400 sampled sources for r = 1..15; convergence-round count recorded on
the side.

**r_c extraction (operational, fixed now):** local log-log slope
s(r) = ln(B(r+1)/B(r−1)) / ln((r+1)/(r−1)); r_c = the first radius where s crosses
1.5, linearly interpolated between integer r. Seeds averaged after extraction.

## Pre-committed predictions

- **M1 (monotonicity, necessary):** r_c(p) is monotone decreasing across the grid.
  If not, the mechanism is dead immediately.
- **M2 (the quantitative test):** the d(p) residual waveform (cubic detrend of the
  banked dense-grid curve, frozen pipeline) is a single-valued function of the
  fractional part of r_c: fit residual(p) ≈ A·cos(2π·frac(r_c(p)) + φ) with A, φ
  free. Supported if R² ≥ 0.5; unsupported if R² < 0.2; 0.2–0.5 = inconclusive,
  reported as such. (Phase φ is NOT pre-committed — only single-valuedness in
  frac(r_c) is.)
- **M3 (feature alignment, descriptive):** waveform extrema (peaks ~0.02/0.10/0.36,
  troughs ~0.033/0.23) sit within grid resolution of integer or half-integer r_c
  values, consistently (same offset class for like extrema).
- **Side-measurement (convergence bands, candidate (b)):** convergence-round count
  vs p reported descriptively; if it bands, band edges compared to features. No gate.

## Falsifiers / honesty

M1 false or M2 R² < 0.2 kills the candidate; it goes to BRANCHES CLOSED with the
numbers, and round 3 starts from the remaining candidate (WS shortcut-overlap
statistics). Analysis exactly as specified; deviations flagged post-hoc. Amplitudes
quoted with 3-seed spread.
