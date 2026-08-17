# Pre-registration: overnight jaga runs on the d(p) log-periodic fine structure

**Date:** 2026-08-12 (committed BEFORE any of the data below existed)
**Context:** FINDINGS "Dense-grid follow-up (2026-08-11)" established the d(p) fine
structure as real (window/seed/N-robust, period ~1.6 ln p, peaks ~0.02/0.10/0.36).
These runs test its predictions. Owner authorized overnight jaga use ("anything you
might want to run on jaga for the next 8+ hours").

## P1 — the periodicity prediction (the sharp one)

If the structure is genuinely log-periodic with period Δ ≈ 1.6 ln p, the next feature
pair below the current grid edge must exist: **a peak near p ≈ 0.004 (±30%) and a
trough near p ≈ 0.007**, i.e. one more full cycle. Run: 45 log-spaced p in
[0.004, 0.5], N=32000, 12 seeds.

- Confirmed if: the extended waveform's peak/trough sit within ±30% (in p) of the
  extrapolated positions and the oscillation amplitude there is resolved above the
  12-seed noise floor.
- Refuted/downgraded if: no feature below p = 0.02 — the reading honestly downgrades
  from "log-periodic" to "two-bump structure in the crossover region."

## P2 — N-invariance (mechanism-class discriminator)

The protection-hierarchy hypothesis is density-intrinsic: waveform phase and period
should not depend on N. Runs: the 30-point [0.02, 0.5] grid at **N=8000** and
**N=128000** (16× span), 6 seeds each.

- Consistent if: residual-waveform correlation with the N=32000 reference ≥ 0.8 and
  best-period shift < 20% at both N.
- Violated if: phase or period drifts with N — pointing at a finite-size mechanism
  instead, and the hierarchy hypothesis is weakened or dead.

## P3 — cluster-composition alignment (exploratory, descriptive only)

Surviving-shortcut cluster composition vs p (`shortcut_clusters` probe: singles /
pairs / triples / ≥4, same construction + convergence as `prune_dimension`), 30-point
grid, N=32000, 6 seeds. The hierarchy hypothesis suggests composition shifts
(generation onsets) near the d(p) peaks. No hard gate — this is a mechanism *lead*,
not a test; reported descriptively either way.

## Secondary — tighter DSS-null bound on `grown` ball growth

N=100000, 2000 sources, r ≤ 40, 3 seeds. Expectation: null (consistent with the
banked scan); deliverable is a tighter a90 exclusion bound (currently ~10-20%).

## Analysis commitments

Same pipeline as the banked scan (`logperiodic_scan.py` machinery: cubic detrend for
d(p), permutation p-values, the discriminator ladder), no retuning of thresholds on
this data. Amplitude/period estimates quoted with the seed-noise floor. Any deviation
from this plan gets flagged as post-hoc in the write-up.
