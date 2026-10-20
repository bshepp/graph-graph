# Pre-registration — wave 9: four new rule variants, each removing or reshaping the lost race's curvature

**Date:** 2026-10-19. **Status:** PRE-REGISTRATION, committed before any registered growth.
Owner's approval (2026-10-19): "yes make however many variants you can think of" — this
answers the variant-design decision flagged at the end of waves 7 and 8 (a rule that removes
the lost-race configuration instead of relabelling it; C4, reversible attachment).

## Why these four

Waves 5-8 read the `sheet` crossover as **the deposition of permanent negative curvature by
the lost race**: a dead pair on the boundary (`sheet`), a frozen corner (`strict`), or an
interior degree-7 cone point (`relief`), and for `relief` quantitatively (the hex-corrected
hyperbolic-disc law, W6-a, W8-a). Gauss-Bonnet for a triangulated disc,
sum over boundary (4 - d) + sum over interior (6 - d) = 6, constrains what a local repair can
do. The lost race fixes a -1 of turning at the boundary. A local move can only **move** that
curvature (an edge flip conserves the total), **keep** it (dead/frozen corner, cone point),
**cancel** it with a +1 (an interior degree-5 point), or **remove the material** that made it.
The earlier suggestion of "a closure that flips a neighbouring triangle" therefore ends as
either a frozen corner or a 5-7 pair; the second is built here directly. The four variants:

| variant | on top of | rule (labelled, default-off option of `grow_sheet`) | question |
|---|---|---|---|
| `disloc` | `relief` | every interior degree-7 cone point leaves a credit on the two link ends of the closure that made it; a credit holder closes at degree 5 at the closure rate, consuming it; a holder that closes at 6 or 7 passes its credits to its link ends | **cancel**: is it the net curvature, not the defect, that ends the plane? |
| `pent` | `relief` | any closable boundary node at degree 5 closes at rate `rate` (closure rate 1) | **net law**: with degree-5 points placed independently of the 7s, does the disc law follow lambda_net = (n7 - n5)/N, or the 7s alone? |
| `eager` | `relief` | the closure rate is `rate` (rho) instead of 1 | **race law**: is the cone density set by one race between a waiting closure and an attachment, lambda ~ 1/rho? |
| `detach` | `sheet` | C4: a boundary node of degree b whose removal leaves a disc detaches at rate `rate` exp(-beta (b + c)); `rate` = exp(beta mu) | **remove**: does reversibility heal the lost race? |

Considered and not built: per-node random caps (curvature set by hand; the ring-disc
calibration, W8-b, already tests the law on random cone discs); `pent` without `relief` (a
sheet with dead corners and positive points mixes two defect kinds, not a clean contrast); a
flip-repair closure (reduces to a frozen corner or to `disloc`, above); a probabilistic
`relief` (a mixture of two tested rules).

Node ids under `detach`: a detached node keeps its id with an empty neighbour set, so ids stay
birth order; `n_nodes` and checkpoints count live nodes; analysis compacts the graph
(`sheet_growth.compact`, order-preserving). Nodes 0-2 never detach.

## What happened before this file (disclosed in full)

1. **Code checks.** `python sheet_variants.py --validate` passes: the four old variants are
   bit-identical to the committed generator (checked separately against `git show
   HEAD:sheet_growth.py` on eight (variant, beta, seed) cases); `eager` at rate 1 is
   bit-identical to `relief`; every new variant keeps a manifold disc (Euler characteristic 1,
   every edge in one or two triangles) with sum_bdry (4 - d) = 6 + n7 - n5; under `disloc`
   every degree-5 point has a degree-7 point within two hops. A brute-force audit (after every
   event, every valid event has a live entry in the sampler) found a bug in `detach`: a closure
   did not re-offer the closed node's neighbours, whose removability reads it (a boundary node
   touching the closed node across a neck). Fixed; the audit then passed on all variants
   (`sheet_sampler_audit.py`). The `detach` pilot below was re-run after the fix.
2. **Pilots (seeds 3 and 5, never registered seeds).**
   - `disloc`: beta 1.0 to 2^18: B/sqrtN 3.88-4.02 at every octave, 78 fives / 79 sevens;
     beta 0.75: 4.09 -> 4.52 (437 / 437); beta 0.5 (seed 5): 4.69 -> 6.41 (1884 / 1897),
     ball-volume slope around node 0 2.02 / 2.10 / 2.20 at r = 50 / 75 / 100 against the
     lattice's 1.94 / 1.95 / 1.99, the centre's eccentricity exponent 0.41, boundary nodes at
     distance 215-255 from node 0. `relief` on the same footing (beta 0.75, seed 3): 22.9.
     The nearest degree-7 point of every degree-5 point is at distance 1 (214 of 218) or 2:
     tight 5-7 pairs, elementary dislocations.
   - `pent`, beta 1.0, rate 1e-6 / 3e-6 / 1e-5 to 2^18: n5 / n7 = 3 / 77, 21 / 87, 70 / 86;
     rate 3e-5 to 40,000: 34 / 13; rate 1e-4 (seed 5) and 3e-4 (seed 3) stall at 3,800 and 773
     nodes with a 4-edge hole left open (n5 - n7 = 9 and 8).
   - `eager`, beta 0.75, seed 5, to 2^18: cones 334 / 86 / 29 / 3 at rho 1 / 4 / 16 / 64
     (rho 64 stays flat, B/sqrtN 4.23); beta 0.5: 1387 / 138 at rho 1 / 16. Seed 3, beta 0.75,
     to 2^17: 33 / 23 at rho 4 / 16.
   - `detach` (after the fix), seed 3 to 2^16 unless stated: beta 1.0 mu 3: B/sqrtN 10.45, 49
     dead corners (2^18: 21.6, 238; `sheet` same seed 28.3, 415); beta 1.0 mu 3.8: 4.94, 10;
     beta 1.25 mu 3 and 3.8: flat to 2^16, 0 dead; beta 1.25 mu 3.8 to 2^18: first dead
     corners by 2^17, B/sqrtN 8.07 (`sheet` 16.7). Detached fraction of all nodes grown 3-13%.
   The pilots chose the grids below (`pent` rates on the hyperbolic side of the sign change,
   mu values at which `detach` acts but growth still proceeds) and are why several
   predictions are not 50/50.
3. **Dry runs of the reader (pilot seeds 3, 5, 6; to 2^17-2^18).** These changed W9-b before
   registration. (a) `pent` past the sign change (rate 3e-5, 1e-4) does not grow a spherical
   cap: positive curvature closes a bulb early, its last boundary nodes are left capped and
   pinched (dead corners under the relief guard), and growth continues through a neck; seed 3
   at 1e-4 reached 131,072 nodes with B/sqrtN 0.99 and the centre's eccentricity frozen
   (exponent 0.01), n5 - n7 = 168, a multiply-wrapped surface; seed 5 stalled at 3,800 with a
   4-edge hole. A stop at n5 - n7 = 10 did not help (seed 3 pinched at 4,880 before it). The
   "spherical side" is therefore not a test of a disc law and is kept descriptive (rates 3e-5,
   1e-4, 4 seeds, to 2^18). (b) Eligibility now also requires a **complete ball**: no boundary
   node within 1.25 r of node 0 (`r_complete`), the condition the disc law actually needs;
   "at most half the graph" alone admitted radii in the bulb-and-neck graphs. It changes
   nothing for disc-like growths. (c) Near balance (rate 1e-5) the net law held on one seed to
   r = 100 (0.98-1.07; 1.11 at 125) and failed on the other beyond r = 60 (0.89 at 75, 0.42 at 150), with
   n5 / n7 = 72 / 83: independent degree-5 and degree-7 points cancel on average but leave a
   curvature excess fluctuating like sqrt(n) over a ball, about 2 pi in total by r ~ 100-150
   at these densities (estimate: sqrt(3 r^2 (lambda5 + lambda7)) pi / 3 = 2 pi). `disloc`'s
   bound pairs cancel at the lattice scale and do not fluctuate this way. At 6e-6 the net law
   held on both seeds (0.94-1.07).

## Growths (seeds 1000-1007, to 2^20 = 1,048,576 nodes; `W9_SPECS`)

- `detach`: beta 1.0 and 1.25 x mu 3.0 and 3.8 x 8 seeds (32).
- `eager`: beta 0.5 and 0.75 x rho 1, 4, 16, 64 x seeds 1000-1003 (32). rho = 1 is `relief`
  (bit-identical), so these are also the `relief` controls for `disloc` on the same seeds.
- `disloc`: beta 0.5, 0.75, 1.0 x 8 seeds (24); `relief` beta 1.0 x seeds 1000-1003 (4).
- `pent`: beta 1.0 x rate 3e-6, 6e-6, 1e-5 x 8 seeds (24, registered); rate 3e-5 and 1e-4 x
  seeds 1000-1003 to 2^18 (8, descriptive).

124 growths, jaga, 24 workers, `python sheet_variants.py --scan9 --jobs 24`, under a
usage-log CHECK-IN/OUT, working dataset /scratch/graph-graph destroyed after an md5-verified
pull; `--read9 CSV` locally.

## Frozen readings (`python sheet_variants.py --read9 CSV`)

Per growth, the wave 6 reading: **flat** if B/sqrtN at 2^20 over B/sqrtN at 2^16 is in
[0.85, 1.15] and the centre's eccentricity exponent (2^19 -> 2^20) is in [0.40, 0.60];
**crossed** if the ratio exceeds 1.3 or the exponent is below 0.35. Per cell: STALLED if at
least half the growths stalled; else FLAT if at least 7/8 of the cell is flat; CROSSOVER if
at least 6/8 crossed; else UNDECIDED. The "law" ratio for a radius r is the measured local
slope of V(r) around node 0 divided by signed_slope_hex(r, lambda_net) x lattice_slope(r) / 2,
lambda_net = (interior 7s - interior 5s) / N, radii 10-500 (`RADII9`) whose ball V(1.25 r) is
at most half the graph and contains no boundary node.

- **W9-a `disloc`.** Gate per growth: not stalled, no dead corner at 2^20, max degree <= 7,
  interior degrees in {5, 6, 7}, at least 95% of degree-5 points with a degree-7 point within
  two hops. Cell verdicts at beta 0.5, 0.75, 1.0. **Net-curvature law HOLDS** if at least 90%
  of the pooled `disloc` pairs are within 10%. Controls: the `relief` growths (beta 1.0
  `relief`, beta 0.5 / 0.75 `eager` rho 1), crossed count per beta.
- **W9-b `pent`** (rates 3e-6, 6e-6, 1e-5). Discriminating pairs: those where the net-law and
  the gross-law (degree-7 points alone) predictions differ by more than 10%. **NET LAW HOLDS**
  if at least 90% of pooled pairs are within 10% of the net law, the discriminating pairs'
  mean net ratio is in [0.95, 1.05] and their mean gross ratio is below 0.90; **UNDERPOWERED**
  if fewer than 10 discriminating pairs; else **NET LAW MISSED**. Reported, no verdict: cell
  verdicts; the descriptive spherical cells (morphology, dead corners, stalls, complete-ball
  ratios).
- **W9-c `eager`.** Per beta, the least-squares slope of ln(cones / N) against ln rho over the
  16 growths (a growth with no cone counts as 0.5): **RACE LAW** if the slope lies in
  [-1.25, -0.75] at both betas; else RACE LAW MISSED.
- **W9-d `detach`.** Cell verdicts; reported, no verdict: the detached fraction, and dead
  corners present at one checkpoint and gone at the next (healed / total).

## Predictions

- W9-a: gate passes in all 24 (75%). beta 1.0 **FLAT** (85%); beta 0.75 **FLAT** (65%);
  beta 0.5: CROSSOVER by the boundary reading 45%, UNDECIDED 35%, FLAT 20% (the pilot's
  boundary excess grew by a factor of 1.2 over two octaves while the ball slope stayed within
  about 10% of flat). Net-curvature law **HOLDS** (65%). `relief` controls crossed in all 12
  (95%). The reading I will give if beta 0.5 crosses by the boundary while the law holds: the
  boundary excess there is roughness from a high dislocation density (about one per 140 nodes),
  not curvature.
- W9-b: **NET LAW MISSED** (55%), HOLDS (35%), UNDERPOWERED (10%). The expected way to miss:
  the 90% clause, from the rate-1e-5 growths, where curvature fluctuations decide the ball
  beyond r ~ 100 (dry run 3c). If it misses that way while `disloc` holds, the reading is that
  cancelling curvature on average is not enough; it must cancel locally, as bound 5-7 pairs
  do. The gross law fails on the discriminating pairs (mean below 0.90; 80%). Cells: 3e-6
  CROSSOVER (75%), 6e-6 CROSSOVER (60%), 1e-5 UNDECIDED (50%).
- W9-c: **RACE LAW** (65%; the pilot's single-seed slopes were -1.13 and -0.83).
- W9-d: CROSSOVER at beta 1.0, mu 3 (90%) and mu 3.8 (75%); at beta 1.25, mu 3 (70%) and
  mu 3.8 (60%). Reversibility delays the dead corners but does not remove them: a dead corner
  heals only when one of its degree-6 nodes detaches, at rate exp(-beta (12 - mu)).

## Reporting

A FINDINGS section dated when the results land, each verdict as frozen here; BRANCHES' row
"A `sheet` rule without dead corners" updated. Variants are never reported as `sheet`.
