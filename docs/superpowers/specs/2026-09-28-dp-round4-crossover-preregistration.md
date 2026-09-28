# Pre-registration — d(p) fine structure, round 4: is it just the crossover?

**Date:** 2026-09-28
**Status:** PRE-REGISTRATION, committed before any crossover form has been fitted to any
data and before the fresh-seed grid is computed. Step 4 of the owner-approved plan.
Driver `dp_round4.py`.

## Hypothesis

Round 3 showed the waveform is carried by log ball counts at radius >= 4 and is smooth in
p. **H:** there is no mechanism to find. `ln|B(r)|` is pinned at the ring's value at low p
and rises once shortcuts come within reach; a polynomial in `ln p` cannot follow a curve
that is flat and then rises, and the oscillating remainder is what has been called
"multi-bump fine structure".

## What has been seen (disclosed)

The round-3 table of cubic-detrended residuals (seeds 0-5), including its sign pattern.
No crossover form has been fitted to anything. The test below uses seeds 6-11.

## Frozen protocol

- Graphs: N = 32000, k = 6, the 30 banked p values, **seeds 6-11**, pruned to
  convergence, 400 sampled nodes, max_radius 10 (same path as round 3).
- `L_r(p)`: mean over nodes and seeds of `ln|B(r)|`; `SE_r(p)`: standard error over the
  six seeds; `SE_r`: its mean over p.
- **Primary form (4 parameters, as many as a cubic):**
  `F(p) = a + h * ln(1 + (p / p0)^g)`.
- **Comparator form (7 parameters, as many as a sextic):** two such crossovers added,
  `a + h1 ln(1 + (p/p1)^g1) + h2 ln(1 + (p/p2)^g2)`.
- Fits by least squares in `L`, 40 random restarts, at each r = 4..10 separately.
- Polynomial baselines on the same data: cubic and sextic in `ln p`.
- Waveform in d: `sum_r w_r * resid_r(p)` with the round-3 fit weights, using for each r
  the residual from the named detrend.

## Frozen verdict rules (on the primary form)

Let `RMS_r` be the root-mean-square residual of the primary form at radius r, and
`W_F`, `W_3` the peak-to-peak of the d-waveform under the primary form and under the
cubic. Let `sigma_w` be the noise level of the waveform itself: the standard error over
seeds of `sum_r w_r L_r(p, seed)`, averaged over p.

- **H SUPPORTED** requires both: `RMS_r <= 2 * SE_r` at each of r = 8, 9, 10; and
  `W_F <= max(0.25 * W_3, 5 * sigma_w)`.

(The noise term was added before this document was committed, when the instrument's own
validation showed that pure noise produces a waveform of about `4 sigma_w` peak-to-peak
over 30 points, so that a bare `0.25 * W_3` could fail a perfect fit.)
- **H REFUTED** requires both: `RMS_r > 4 * SE_r` at r = 10; and the primary-form
  residual at r = 10 is reproducible, i.e. the residuals computed separately from seeds
  {6, 7, 8} and {9, 10, 11} correlate at > 0.7.
- **PARTIAL** otherwise, reported with the fraction of the cubic waveform's variance the
  primary form removes.

The comparator is reported alongside and carries no verdict. It answers the one thing
the banked record holds against H — that the structure survived a sextic detrend: if a
7-parameter crossover form leaves residuals at seed noise where a 7-parameter polynomial
does not, the survival was a property of polynomials.

## What each outcome means

- SUPPORTED: the d(p) fine-structure branch closes. There was a crossover and a detrend.
- REFUTED: there is reproducible structure beyond a single smooth crossover, now located
  (radius >= 4) and quantified against noise; round 5 would need a mechanism.
- PARTIAL: most likely reading is a crossover with more than one scale (the ring
  backbone itself thins at high p); the comparator result says whether two scales suffice.

## Known limits

The primary form assumes one crossover scale and a power law above it. If the high-p end
bends for a separate reason, the primary form fails for a reason that is not a
"mechanism" in the sense of rounds 1-3; that is why PARTIAL exists and why the comparator
is reported.
