# Pre-registration — wave 6: is `relief` a hyperbolic disc, and does a strict veto stay flat at every beta?

**Date:** 2026-10-17 (evening). **Status:** PRE-REGISTRATION, committed before any wave 6
growth on the registered seeds. Follows wave 5 (FINDINGS "Literature paths, wave 5"), whose
two open questions these are. The owner approved continued research overnight ("sidequests
approved, research approved"); this wave is my choice of what to run.

## What was already seen (disclosed)

1. **The motivation for W6-a is post hoc.** After writing wave 5 I compared relief's
   measured local slopes of the ball volume around node 0 (wave 5 CSV,
   `results/sheet_variants_20261017_034005.csv`) with the slope of a hyperbolic disc whose
   curvature is set by the measured density of interior degree-7 nodes, with no fitted
   parameter. 38 of the 40 (growth, r) pairs agreed within 10% (largest misses -11% and
   -9%), at beta 1.0 (slopes 2.0 -> 3.9 over r = 50 -> 200) and beta 1.25 (2.0 -> 3.0 over
   r = 50 -> 300). Those numbers motivated the law; they are not evidence for it. The test
   below uses fresh seeds, a larger N (so larger r, where the law and a power law differ
   most), and a third beta.
2. **Feasibility pilot for W6-b** (not a registered seed): one `strict` growth at beta 0.5,
   seed 1, reached 65,536 nodes without stalling, in 62 s (the default: 45 s). Nothing else
   about it was looked at.
3. Validation (`python sheet_variants.py --validate`) passed: the default is bit-identical,
   no dead corner ever appears under `strict` (beta 0.75, 40,000 nodes), Gauss-Bonnet holds
   on all three variants.

## W6-a. The hyperbolic-disc law for `relief`

**Law.** Each interior degree-7 node is a cone point of angle excess pi/3; a node covers
area sqrt(3)/2 of the unit triangular lattice. A uniform density lambda of cone points per
node is then a mean Gaussian curvature K = -(pi/3) lambda / (sqrt(3)/2) = -2 pi lambda /
(3 sqrt 3). (W1-d found the defect-forming event to be a constant-rate-per-node process, so
lambda is uniform in birth order.) A hyperbolic disc of curvature K has area
A(r) = (2 pi / |K|)(cosh(r sqrt|K|) - 1), whose local log-log slope is
**s(r) = x sinh x / (cosh x - 1), x = r sqrt|K|** (2 in the flat limit). The prediction for
the instrument (V(1.25 r) / V(r / 1.25) around node 0, the wave 5 estimator) is
s(r) x FLAT(r) / 2, where FLAT(r) is that estimator's own slope on a flat disc, fixed from
the wave 5 veto growths with no dead corner: 1.94 at r = 50, 1.99 at r = 100-200, 2.00 at
r >= 250. lambda is measured on each growth (interior degree-7 count / N at the end).

**Growths.** `relief`, beta in {1.0, 1.25, 1.5}, seeds 700-707, to 2^21 = 2,097,152 nodes.
Radii r in {50, 100, 150, 200, 250, 300, 400, 500}; a radius is **eligible** if the ball of
radius 1.25 r holds at most half the graph (keeps the boundary out).

**Frozen reading.** **HYPERBOLIC DISC LAW** if measured / predicted is within 10% in at least
90% of all eligible (growth, r) pairs AND, at each growth's largest eligible radius, within
10% in at least 20 of 24 growths. Otherwise **LAW MISSED** (the table is then the result).

Note on power: at beta 1.5 lambda is small (wave 5 extrapolation: ~1e-5), so the predicted
slope stays near 2 at every eligible radius and that cell barely separates the law from
flatness; beta 1.0 at r >= 250 carries the test (predicted slopes 4.5-6).

**Prediction.** HYPERBOLIC DISC LAW (65%). The most likely failure: the slope falls behind
the law at large r, because cone points near the boundary are not yet in the hop-ball and
the mean-curvature continuum ignores the discreteness of a few hundred cone points.

## W6-b. A strict veto: no dead corner can ever form

**Variant `strict`** (labelled, default-off, `sheet_growth.grow_sheet(variant='strict')`):
every event, attachment or closure, is applied tentatively; if any node among the event's
nodes and their neighbours is then a boundary node at the cap that `_closable` refuses (a
dead corner, any mode), the event is reverted and its entry dropped (re-offered when a
degree nearby changes, as in `veto`). The rule reads at most two hops: it is local. Wave 5's
`veto` refused only the attachment route and leaked through closures (6/16 growths); this
closes both routes by construction.

**Growths.** `strict`, beta in {0.5, 0.75, 1.0, 1.25}, seeds 700-707, to 2^20. Reference:
default `sheet` at beta 0.5 and 0.75, seeds 700-707, to 2^20 (wave 5 has the sheet reference
at 1.0 and 1.25). Recorded per checkpoint as wave 5, plus the number of capped boundary nodes
(degree 6, on the boundary, closable or not: a strict refusal can leave a node waiting).

**Frozen readings, per (variant, beta), 8 growths.**
- **Gate** (strict only): no dead corner at any time in any growth. A failure is a bug and
  voids the strict cells.
- **STALLED** if at least 4 of 8 stop before 2^20.
- else **FLAT** if, in at least 7 of 8, B / sqrt(N) changes by a factor within [0.85, 1.15]
  between 2^16 and 2^20 AND the eccentricity exponent over the last octave is in [0.40, 0.60].
  (A ratio, not wave 5's distance from 3.85: a flat disc grown at low beta may have a rougher
  edge and a larger constant.)
- else **CROSSOVER** if, in at least 6 of 8, that factor exceeds 1.3 or the eccentricity
  exponent falls below 0.35.
- else **UNDECIDED**.

**Predictions.** Gate PASS (95%). `strict` FLAT at beta 1.25 and 1.0 (75%), at 0.75 (60%),
at 0.5 (45%; the alternative I expect is UNDECIDED, from waiting capped nodes acting like
slow defects). Default `sheet` at 0.5 and 0.75: CROSSOVER (95%). The question behind the
cells: wave 5 showed dead corners are necessary for the crossover at beta 1.0-1.25 in the
growths where `veto` happened not to leak; `strict` asks whether that holds at every beta.

## Run

`python sheet_variants.py --scan6 --jobs 24` on jaga (72 growths; the 24 relief growths to 2^21
are the long ones), with the usual usage-log CHECK-IN/CHECK-OUT. `--read6 CSV` re-reads a
saved CSV with the same frozen code. Results go into a FINDINGS section, each verdict as
frozen here.
