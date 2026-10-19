# Pre-registration — wave 8: the hyperbolic-disc law with the graph metric read correctly, at higher curvature

**Date:** 2026-10-18. **Status:** PRE-REGISTRATION, committed before any registered growth.
Owner's standing approval for overnight research (2026-10-18, second night: "research or
recreate ... side quests approved").

## What happened before this file (disclosed in full)

1. **Calibration side quest (`cone_lattice.py`).** Ring-built triangulated discs with prescribed
   degrees reproduce the exact triangular lattice (V = 3r^2 + 3r + 1) and the exact {3,7}
   sphere sizes 7, 21, 56, 147, 385, ... with growth (3 + sqrt 5)/2 (Keller-Peyerimhoff Prop.
   1.1, credited to Cannon-Wagreich and Floyd-Plotnick). On discs with random degree-7 cone
   points (density p = 1e-4 ... 0.1, 32 discs per density, seed 0), the wave 6 law as
   registered (x = r sqrt K) **overshoots** the ensemble-mean local slope by 2-9%, growing
   with x (ratio 0.92 at x = 4-5 at every density). Reading the graph ball as what it is on the
   triangular lattice, a hexagon of circumradius r, and replacing r by the radius of the disc
   of equal area, HEX_AREA r with HEX_AREA = sqrt(3 sqrt 3 / (2 pi)) = 0.9094, brings every
   cell within 0.987-1.033 (x = 0.5 ... 5). No parameter is fitted: the factor is fixed by the
   lattice. The geometry verification note (research_notes/Supporting literature verification/
   geometry_theorems.md) independently flags that the law needs a metric conversion constant.
2. **Post-hoc re-read of wave 6 relief with the correction** (seen; this is why the test below
   must be on fresh growths): mean measured / predicted 1.000 (beta 1.0), 0.997 (1.25), 1.004
   (1.5); 0 of 134 pairs outside 10% (registered law: 4 of 134, all low, the shortfall
   growing with curvature). Wave 6 reached x <= 3.8 only.
3. **Smoke test** of the new reader on an unregistered seed: relief beta 0.75, seed 1, to 2^15:
   no dead corner, max degree 7, lambda 1.25e-3, two eligible pairs (0.957, 1.008).

The wave 6 verdict stands as registered (HYPERBOLIC DISC LAW); what this wave tests is the
corrected law, where it differs most from the registered one.

## Growths

`relief`, beta in {0.75, 0.875, 1.0}, seeds 900-907 (24 growths), to 2^21; jaga, 24 workers.
Lower beta means more cone points (W1-d / W6-a: lambda ~ 1.2-1.3 x 0.15 exp(-6.44 beta),
about 1.5e-3 at 0.75 and 7e-4 at 0.875), so the half-filled ball reaches x_hex ~ 5.5-6, the
steep part wave 6 could not test. Radii 25, 75, 125, 175 are added to the ball-slope set
(read6 is unchanged: it reads only its registered radii); the flat slope is now the exact
lattice estimator `lattice_slope(r)`, which reproduces wave 6's measured FLAT_SLOPE table to
0.01 at every registered radius.

## Frozen readings (`python sheet_variants.py --read8 CSV`)

- **Gate (per growth):** not stalled, no dead corner at 2^21, max degree <= 7 (relief's
  premise: the only defects are interior degree-7 cone points). Failing growths are excluded
  from W8-a and listed.
- **W8-a** on gated growths, eligible pairs = radii whose ball V(1.25 r) is at most half the
  graph; ratio = measured slope / (hyperbolic_slope(HEX_AREA r, lambda) x lattice_slope(r)/2),
  lambda = interior degree-7 nodes per node at 2^21 (as in wave 6).
  - **HEX LAW HOLDS** if (i) at least 90% of eligible pairs are within 10%, (ii) the mean
    ratio per beta lies in [0.97, 1.03], and (iii) the steep part, pairs with
    x_hex = HEX_AREA r sqrt K >= 4, has at least 10 pairs with mean ratio in [0.95, 1.05].
  - **HEX LAW HOLDS (steep part untested)** if (i) and (ii) hold and (iii) has fewer than 10
    pairs.
  - **HEX LAW MISSED** otherwise.
  - Reported, no verdict: the registered (uncorrected) law's mean ratio on the steep pairs.
- **W8-b (local, cheap):** `python cone_lattice.py --calibrate --seed 1` (fresh discs, 32 per
  density): every cell's hex ratio in [0.97, 1.04] -> **CALIBRATION REPLICATES**, else
  **CALIBRATION MISSED** (cells listed).

## Predictions

- Gate passes in all 8 growths at beta 1.0 (90%), 0.875 (75%), 0.75 (55%). The risk at low
  beta: relief's guard is c + 1 = 7, so a lost race at a degree-7 boundary node could leave a
  dead corner (relief's own analogue of the default rule's defect).
- W8-a: HEX LAW HOLDS (60%); HEX LAW HOLDS (steep part untested) (15%); HEX LAW MISSED (25%,
  most likely by (iii): the {3,7} lattice says the continuum law overshoots again at very
  high curvature, 0.962 asymptotically even with the correction, but that is at lambda = 1).
- Registered law on the steep pairs: mean ratio below 0.95 (75%).
- W8-b: CALIBRATION REPLICATES (80%).

## Run

`python sheet_variants.py --scan8 --jobs 24` on jaga under a usage-log CHECK-IN/OUT, working
dataset /scratch/graphgraph destroyed after an md5-verified pull; `--read8 CSV` locally.
