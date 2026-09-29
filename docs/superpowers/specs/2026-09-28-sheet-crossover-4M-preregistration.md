# Pre-registration — does `sheet` at beta = 2 turn tree-like near 1.6 million nodes?

**Date:** 2026-09-28
**Status:** PRE-REGISTRATION, committed before the growth starts. Step 2 of the
owner-approved plan. Owner's standing instruction for this step: run every applicable
method and compare, rather than choose one. Driver `sheet_crossover.py`.

## The prediction under test

The transition scan (step 1) supported a crossover, thinly: `ln N* = 2.68 + 5.82 beta`
fitted at beta = 1.0-1.7, which puts the crossover of beta = 2 at **N* = 1.6 million**.
The residual scatter of that fit is 0.47 in ln N*, so the law's own uncertainty is about
a factor 1.6 either way: roughly 1.0 - 2.6 million. The rival, a transition at
beta_c = 2.33, predicts no crossover at beta = 2 at any size.

## Frozen protocol

**Part A — the test.** beta = 2.0, c = 6, seeds 300-303, growth to N = 4,194,304 (2^22),
checkpoints at every factor 2 from N = 1024 (13 checkpoints). At each checkpoint:
boundary edge count B, double-sweep diameter D, and the **arm fraction** A = fraction
of nodes within 10 hops of a boundary edge. A single-seed trial runs first to measure
memory; if it does not fit, N_max is halved and that is disclosed.

**Part B — firming the law.** Re-scan beta = 1.0-1.7 with 8 fresh seeds (304-311) to
N = 512000, so that N*(beta) rests on 12 seeds instead of 4, and refit both laws.

## Methods, each with its own frozen reading (seed-mean curves, factor-2 steps)

- **M1, boundary exponent** `e_B = dln B / dln N`: compact 0.5, tree-like 1.0.
  Crossover size N*_B: where e_B passes 0.75 and stays above it (the step-1 definition).
- **M2, diameter exponent** `e_D = dln D / dln N`: compact 0.5, tree-like ~0.1
  (logarithmic). N*_D: where e_D falls below 0.3 and stays below it.
- **M3, arm fraction** A(N): in a compact disc A ~ 10 * perimeter / N ~ N^(-1/2); once
  arms of width <~ 20 hold a finite fraction of the mass, A stops falling. N*_A: where
  `dln A / dln N` rises above -0.25 and stays there.
- **M4, window audit** (`window_stability.audit_adjacency`, radii to 400) and
  **M5, spectral flow** on the final 4M-node graphs: supplementary only. Both see scales
  well below a million nodes, so they are expected to read "2D" whatever happens at
  the crossover; they are run because the owner wants all methods reported, and because
  a *failure* to read 2D would itself be news.

## Frozen verdicts

Using M1-M3 on the seed-mean curves of Part A:

- **CROSSOVER CONFIRMED**: at least two of N*_B, N*_D, N*_A are finite and inside
  [262,144, 4,194,304], and every finite one lies within a factor 4 of every other.
  *Quantitatively* confirmed if, in addition, N*_B lies inside the 95% prediction
  interval of the exponential law refitted in Part B.
- **TRANSITION BACK ON THE TABLE**: all three are censored at 4,194,304, with
  e_B <= 0.6, e_D >= 0.4 and dln A / dln N <= -0.35 at the last step.
- **UNDECIDED**: anything else, including methods that disagree by more than a factor 4.

Part B's refit is reported with both laws' AICc as in step 1, and with the 12-seed N*
values against the 4-seed ones. If the 12-seed refit changes the predicted N*(2.0) by
more than a factor 3, that is reported before the Part A result is read against it.

## Predictions

Crossover, i.e. CROSSOVER CONFIRMED, at about 60%; N*_B between 1 and 3 million. The
three methods should agree within a factor 2 if the crossover is a single event; a
factor-4 spread would itself say the arms form gradually.

## What each outcome means for the plan

- CROSSOVER CONFIRMED: `sheet` is two-dimensional up to exp(2.7 + 5.8 beta) nodes and
  must always be described with that limit. Step 3 uses beta >= 3 at N <= 1e6, or the
  plain triangular lattice.
- TRANSITION BACK ON THE TABLE: step 1 was wrong, the FSS study of beta_c becomes step 2'.
- UNDECIDED: report the disagreement; the arm fraction (M3) is the most direct of the
  three and would be the one to trust for a follow-up design.
