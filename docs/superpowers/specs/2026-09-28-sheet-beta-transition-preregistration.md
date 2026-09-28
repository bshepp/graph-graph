# Pre-registration — is there a transition in the `sheet` rate strength?

**Date:** 2026-09-28
**Status:** PRE-REGISTRATION, committed before the scan starts. Step 1 of the
owner-approved plan of 2026-09-28. Driver `sheet_transition.py`.

## Question

`sheet` (c = 6) is tree-like at rate strength beta = 1 and two-dimensional at beta = 2 and
3, where its numbers are indistinguishable. Between them there is either

- a **transition**: a finite beta_c above which the sheet is compact at every size, or
- a **crossover**: every beta is tree-like at large enough size, and what grows with beta
  is only the size N*(beta) up to which the sheet looks two-dimensional.

A finite scan cannot prove either. It can measure N*(beta) wherever N* is in reach and ask
which law it follows.

## What is already known (disclosed)

From the confirmatory run of 2026-09-27 (seeds 100-102): beta = 1 has per-octave boundary
exponents 0.58, 0.89, 1.07, 1.04 at N = 2e3 → 5.12e5 (factor-4 octaves), so N*(1) is
near 1e4; beta = 2 and 3 stay at 0.50 throughout. Nothing between 1 and 2 has been run.

## Frozen protocol

- `python sheet_transition.py --jobs 12`: c = 6, beta = 1.0, 1.1, ..., 2.0 (11 values),
  seeds 200-203 (never used), growth to N = 512000 with checkpoints at every factor 2
  from N = 1000 (10 checkpoints).
- `B(N)`: seed-mean number of boundary edges. Local exponent
  `e(N_i) = ln(B(N_{i+1}) / B(N_i)) / ln 2`, assigned to the geometric midpoint.
  Compact growth has e = 1/2, tree-like growth e = 1.
- **Crossover size** `N*(beta)`: the size at which `e` crosses 0.75 going up, by linear
  interpolation in ln N between the two bracketing midpoints, provided `e` stays above
  0.75 at every later midpoint. If `e` never reaches 0.75, N* is **censored** (> 512000).
  If `e` is above 0.75 from the first midpoint, N* is **below range**.

## Frozen analysis

Let F be the set of beta with a finite, in-range N*. Fit to ln N* over F, by least squares:

- **E (crossover, exponential):** `ln N* = a + b * beta` (2 parameters).
- **P (transition, power law):** `ln N* = a - nu * ln(beta_c - beta)` (3 parameters;
  beta_c scanned on a grid from max(F) + 0.01 to 4.0).

Compared by AICc. Verdicts, in this order:

- **INSUFFICIENT**: fewer than 5 values of beta in F. No verdict.
- **TRANSITION SUPPORTED** requires all of: AICc(P) < AICc(E) - 4; the fitted beta_c is
  <= 2.2; and every scanned beta > beta_c is censored with `e <= 0.6` at its last
  midpoint.
- **CROSSOVER SUPPORTED** requires: AICc(E) <= AICc(P) + 2, **and** the exponential
  fit's prediction for the smallest censored beta is consistent with it being censored
  (predicted N* > 512000). If E fits but predicts an N* inside the range for a beta that
  is in fact censored, E is contradicted by the data it did not see.
- **UNDECIDED**: anything else.

The censored points carry real information and are used only through the two
consistency clauses above, never as fitted values.

## Prediction

I do not have a strong one. Growth models with a rate favouring high coordination
usually show a roughening crossover rather than a sharp morphological transition, which
argues for CROSSOVER; but without an embedding nothing forces two arms to collide, which
is what normally smooths such transitions out, and beta = 2 and 3 being numerically
identical argues for a genuine compact phase. I put it at 55 / 45 for CROSSOVER or
UNDECIDED against TRANSITION.

## What each outcome means for the plan

- TRANSITION: the project's first critical point. Step 2 becomes its finite-size scaling
  study (order parameter, exponent nu, behaviour of the seed-to-seed variance at beta_c).
- CROSSOVER: `sheet` is "two-dimensional up to a size that grows exponentially with
  beta". Still usable as a substrate at beta >= 2 for any N in reach, which is what
  step 3 needs, but it must be described that way.
- UNDECIDED / INSUFFICIENT: says which beta and N a second scan would need.

## Known limits

- Four seeds; N* is read from a seed-mean curve, with seed scatter reported.
- One decade and a half of N* at best. A power law with a large exponent and an
  exponential are hard to tell apart over that range; the AICc margin is set so that a
  near-tie is reported as UNDECIDED rather than forced.
- c = 6 only.
