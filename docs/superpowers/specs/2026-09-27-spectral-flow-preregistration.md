# Pre-registration — spectral-dimension flow `d_s(t)` on `grown`

**Date:** 2026-09-27
**Status:** PRE-REGISTRATION. Committed before any random-walk return probability has
been computed on any graph in this project. Owner delegated the forks (2026-09-26); they
are taken at the readiness note's recommendations and listed so they can be overruled.
Builds on `2026-08-17-spectral-flow-readiness.md`.

## What is already known when this is written (disclosure)

The 2026-09-26 review ran a **different** observable — ball growth — to large radius on
`grown` at N = 5.12e5 and found it is **exponential, not polynomial**: the ratio
`|B(r+1)|/|B(r)|` is constant (~1.27-1.31 at cap 6) from r ~ 9 to r ~ 32, the diameter
grows like log N, and the project's own `d_eff` rises from 1.74 to 3.95 as the fit window
grows from radius 6 to 32, while 2D/3D tori are window-stable to 0.05. So the banked
"`grown` has d ~ 2.2" is a fixed-window reading (see `window_stability.py`).

That changes the prior. The readiness note's null (`d_s = d_H`, flat, no flow) assumed
`grown` has a dimension. **The prediction committed here is instead RUNAWAY** (defined
below): `d_s(t)` on `grown` rises with `t` and never plateaus. No walk data has been
looked at; the prediction is an inference from ball growth and can fail.

## Forks taken

1. Walk convention: **lazy** (stay probability 1/2).
2. Estimator: **deterministic sparse matvec**, node-averaged return probability by a
   Hutchinson trace, `P(2k) = E ||S^k z||^2 / N` with `S = (I + D^-1/2 A D^-1/2)/2`
   (symmetric, positive semidefinite, so every term is a squared norm) and Rademacher
   `z`. Cross-checked in `--validate` against the exact spectrum on small graphs.
3. Scale: **local**, N = 5e4 and 2e5.
4. Definition of flow: frozen below.
5. Gates: anchors must pass before `grown` is read.

## Frozen definitions

- Graph: largest connected component. `P~(t) = P(t) - 1/N` (stationary floor removed).
- Grid: `t_j = 16 * 2^(j/2)`, j = 0, 1, 2, ... (rounded to even integers).
- `D(t_j)`: minus twice the least-squares slope of `ln P~` vs `ln t` over all computed
  even `t` in `[t_j / sqrt 2, t_j * sqrt 2]`.
- Admissible range: `t_j >= 16`, `P~(t) >= 20/N` over the whole fit window, and
  `t_j * sqrt 2 <= t_max` (t_max = 20000).
- **Plateau**: a run of >= 5 consecutive grid points (a factor 4 in t) with
  max - min of `D` <= 0.2. Its value is the median of `D` over the run.
- Verdicts, evaluated in this order, on the admissible range:
  - **INSUFFICIENT RANGE**: fewer than 9 admissible grid points (a factor 16).
  - **FLAT**: the whole admissible range is one plateau.
  - **PLATEAU-TO-PLATEAU FLOW** (the CDT-like outcome): two disjoint plateaus whose
    values differ by > 0.3, the later one reaching the end of the admissible range.
  - **RUNAWAY**: total change `D(last) - D(first) > 0.5`, Spearman rank correlation of
    `D` with `t` > 0.9, and no plateau inside the last factor 4.
  - **UNCLASSIFIED**: anything else. Reported as such; no re-binning after the fact.

## Gate A — instrument (must pass before anything is interpreted)

- Exactness: on graphs with N <= 600 the Hutchinson estimate with 64 probes must match
  `sum_i lambda_i^t / N` from the full spectrum to within 5% at every `t` with
  `P~ >= 20/N`.
- Known answers, each must read **FLAT** at the stated value +- 0.1:
  ring (1), 2D torus (2), 3D torus (3), Sierpinski gasket (2 ln 3 / ln 5 = 1.365).
  The gasket is the control that the instrument can read a spectral dimension that
  differs from the Hausdorff one (1.585).
- Negative control: a random 3-regular graph must **not** read FLAT.

If the gasket's known log-periodic ripple makes it fail the 0.2 plateau tolerance, that is
reported as a Gate-A failure of the *plateau definition* and the definition is not
silently widened; an amendment would be written before `grown` is read.

## Gate B — the question

`grown` at caps 6, 7, 8, N = 5e4 and 2e5, 3 seeds each.

- **P1 (predicted): RUNAWAY** at every cap, both N, all seeds.
- **P2 (N-robustness):** on the common admissible range, the seed-mean `D(t)` curves at
  N = 5e4 and 2e5 agree to within 0.15 at every grid point.
- **P3 (ordering):** at every admissible `t`, `D` is ordered cap 6 < cap 7 < cap 8.

Outcomes that would refute the prediction, and what each would mean:

- FLAT: `grown` has a spectral dimension even though it has no Hausdorff one — a
  genuine `d_s`/`d_H` split, and the most interesting possible result.
- PLATEAU-TO-PLATEAU FLOW: a CDT-like flow from a minimal growth rule — the readiness
  note's flagship positive.
- UNCLASSIFIED / INSUFFICIENT RANGE: the walk mixes before a verdict is possible;
  reported as "no verdict", which is itself evidence of expander-like mixing but is
  not claimed as RUNAWAY.

## Descriptive riders (no verdict attached)

- Pruned Watts-Strogatz at p = 0.1, 0.3, 0.5 (the project's other dimension knob).
- The short-time value `D(16..64)` against the banked fixed-window `d_eff`.
