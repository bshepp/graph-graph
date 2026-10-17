# Pre-registration — literature paths, wave 5: rule variants of `sheet`

**Date:** 2026-10-17. **Status:** PRE-REGISTRATION, committed before any variant was grown.
Plan: `2026-10-17-literature-paths-plan.md`. Paths: C1, C3, C6; C2, C4 and C5 are not run
(reasons below).

**What a variant is.** Each variant is a labelled, default-off option of
`sheet_growth.grow_sheet` (`variant=...`). The default `sheet` is unchanged and
bit-identical (checked by `--validate`), every banked result and saved graph stands, and
no variant is ever reported as `sheet`. The survey marked variants as the owner's decision;
the plan of 2026-10-17 implements them as labelled options so the owner can keep, rename or
discard any of them.

## Why two variants, and the contrast they test

Wave 1 fixed the trigger: every dead corner is born when a degree-6 boundary node a waits to
close (its link end x has degree 5) and an attachment on x's other boundary edge wins the
race, capping x; a and x then block each other forever (the degree guard, 474 of 474). There
are two ways to remove that event, and the curvature theorems say they should differ:

- **`relief` (C1)**: the degree guard admits a link end at the cap and raises it to c + 1.
  The waiting closure then still fires, and x, now degree 7, closes in turn: each would-be
  dead pair becomes **one interior degree-7 node**, a cone point of curvature -1/6. These
  appear at a constant rate per added node, i.e. at a constant areal density. Constant
  negative curvature density makes a surface hyperbolic beyond the radius where the enclosed
  deficit reaches a full turn (Keller-Peyerimhoff; Angel-Benjamini-Horesh Prop. 6: a degree-7
  vertex in every ball of radius R forces non-amenability). With density lambda per node,
  that radius is about sqrt(2 / lambda) hops, reached near N ~ 6 / lambda. **Prediction from
  the theorems: the crossover returns, as interior hyperbolic growth with no dead corners and
  no fjords, within about a decade of the default rule's crossover.** BRANCHES (2026-10-14)
  predicted the opposite ("a variant without dead corners should be a plane at every beta").
- **`veto` (C3)**: an attachment is refused when it would cap a node whose other boundary
  neighbour is already capped (the lost race; one-hop degree read). No defect of any kind
  forms, every degree stays <= 6, so by Hua-Jost-Liu growth stays quadratic. **Prediction:
  no dead corners and no crossover.**

## The scan (fixed here)

`python sheet_variants.py --scan`: variants `sheet` (default, reference), `relief`, `veto`;
beta in {1.0, 1.25}; seeds 600-607; grown to 2^20 = 1,048,576 nodes (at least 30 times the
default rule's mean first-dead-corner size at these betas); exact checkpoints at every
octave from 2^12 via the growth callback. Per checkpoint: B / sqrt(N), dead corners (with the
variant's own closability), interior degree histogram (count of degree-7 nodes), and the
first dead corner / first degree-7 node sizes. On each final graph: eccentricity of node 0
and its last-octave exponent; local slope of the ball volume around node 0 at r = 50, 100,
200, 300, 400; the window audit (30 sources, windows to 320, as wave 1).

**Frozen readings, per variant (over its 16 growths):**
- **NO CROSSOVER** if B / sqrt(N) at 2^20 is within 10% of 3.85 in at least 14 of 16 growths
  and the eccentricity exponent over the last octave is in [0.40, 0.60] in at least 14.
- **CROSSOVER** if B / sqrt(N) at 2^20 exceeds 1.2 x 3.85 or the eccentricity exponent falls
  below 0.35 in at least 12 of 16.
- Otherwise UNDECIDED.
For `relief`, also **HYPERBOLIC INTERIOR** if, on the crossed growths, the ball volume around
node 0 has local slope above 2.5 at some r while the graph has no dead corner, and the
degree-7 count grows in proportion to N (ratio of counts at 2^20 and 2^19 in [1.6, 2.4]).

**Predictions.** `sheet`: CROSSOVER (95%; the reference). `veto`: NO CROSSOVER (65%), with
zero dead corners in at least 14 of 16 (75%). `relief`: CROSSOVER (60%) and HYPERBOLIC
INTERIOR (50%); zero dead corners in at least 14 of 16 (80%).

## C6. Cap 7 against the Keller-Peyerimhoff bound

On the three saved c = 7, beta = 2, 512,000-node sheets: interior degree histogram, and the
exponential growth rate mu = d ln V / d r of balls around 30 interior sources over the radii
where V < 0.1 N. With every interior node of degree 7 (curvature -1/6 with triangular faces),
Keller-Peyerimhoff give mu >= ln(1 + 2 q C / (q - 1)) = ln 1.5 = 0.405 for q = 3, C = 1/6,
for a tessellation without boundary. **BOUND RESPECTED** if the measured mu >= 0.405 on all
three; otherwise **BELOW THE BOUND** (the boundary and finite size would then be the reason
to state). Prediction: BOUND RESPECTED (65%).

## Not run, with reasons

- **C2 fjord-mouth bridging**: it needs two boundary nodes far apart along the edge but within
  two or three hops across a fjord mouth. Wave 2 (W2-b) found no such pair on any of the
  twelve graphs or fourteen prefixes, so the variant would never fire.
- **C5 motif-size programming** ("require an additional capped second neighbour before a node
  is declared un-closable"): in this rule nothing is declared; a dead corner is a mechanical
  consequence of the degree guard. The proposal has no counterpart.
- **C4 reversible attachment**: deferred, not dropped. Detaching nodes breaks the invariant
  that node ids are birth order, which every prefix tool relies on, so it needs its own
  bookkeeping and its own pre-registration. Flagged for the owner.

## Reporting

A FINDINGS section dated when the results land; each verdict as frozen here; the BRANCHES
row "A `sheet` rule without dead corners" updated with whichever way it falls.
