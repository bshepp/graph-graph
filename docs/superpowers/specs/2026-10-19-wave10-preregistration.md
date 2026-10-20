# Pre-registration — wave 10: does `disloc`'s leftover curvature settle or run away?

**Date:** 2026-10-19 (late). **Status:** PRE-REGISTRATION, committed before any registered growth.
Authority: the owner's standing approvals of 2026-10-19 ("make however many variants you can
think of"; "continue as long as possible"; commit and push whenever needed) and of the earlier
overnight sessions (research and side quests approved; jaga used under usage-log
CHECK-IN/OUT). This wave adds no new rule: it grows the wave 9 variant `disloc` four times
larger to answer the question wave 9 left open.

## The question

Wave 9 (FINDINGS "Wave 9"): `disloc` binds every degree-7 cone point to a later degree-5
point. At beta 1.0 it is flat to 2^20; at 0.75 the frozen reading calls it flat only
marginally (2^16 -> 2^20 ratios 1.12-1.15 against the 1.15 limit); at 0.5 it crosses. The
leftover curvature is the unpaired remainder n7 - n5 (credits not yet spent, riding the
front). A descriptive follow-up (`disloc_remainder.py`, seeds 20-23 to 2^19) found that
remainder roughly proportional to the boundary length (about 0.003 per boundary edge at beta
0.5, 0.001 at 0.75), while the boundary itself lengthens faster than sqrt(N) at beta 0.5. A
remainder proportional to a boundary that its own curvature roughens is a feedback. Does it
settle (local cancellation is enough at large N, given a beta) or run away (every beta
eventually crosses)?

## What happened before this file (disclosed in full)

1. All of wave 9's data, read (FINDINGS "Wave 9"); the follow-up above (seeds 20-23).
2. The reader below, applied post hoc to wave 9's `disloc` growths (base 2^16, end 2^20):
   **feedback ACCELERATING 8/8 at both betas** (B/sqrtN octave factors at beta 0.5 rising from
   1.06-1.13 to 1.17-1.21; at 0.75 from 1.00-1.03 to 1.04-1.06); cells CROSSOVER (0.5) and
   FLAT (0.75); net disc law 103/116 (MISSED at 90%) and 128/128 (HOLDS). These numbers are
   why the feedback predictions below are not 50/50.
3. Code: `checkpoint_stats` now also counts interior degree-5 nodes; `grow_one9` writes a
   `profile5` column (N:interior 5s/interior 7s/boundary edges per octave). Neither touches
   growth (no random draws; `--validate` passes). Smoke test: four `disloc` growths to 2^17
   (seeds 30-31, never registered) through `read10`. The smoke run changed one reading before
   registration: a least-squares slope of ln(n7 - n5) on ln B read 4.04 from remainders of
   1-9 (noise), so the remainder reading is now the pooled ratio below.

## Growths

`disloc`, beta 0.5 and 0.75, seeds 1100-1107, to 2^22 = 4,194,304 nodes (16 growths,
`W10_SPECS`). jaga, ONE `python sheet_variants.py --scan10 --jobs 16` process, 16 long-lived
nice'd workers, under a usage-log CHECK-IN/OUT, working dataset /scratch/graph-graph
destroyed after an md5-verified pull; `--read10 CSV` locally. Estimated ~1.5 h (a 2^20
`disloc` growth took ~1,050 s); RAM watched during the run, and the run is stopped if free
memory falls below 30 GB (a stop would be reported, not re-run silently).

## Frozen readings (`python sheet_variants.py --read10 CSV`)

Per growth (base = 2^18, end = 2^22):
- **Crossing:** the wave 6 reading over the last four octaves: flat if B/sqrtN(2^22) /
  B/sqrtN(2^18) is in [0.85, 1.15] and the centre's eccentricity exponent (2^21 -> 2^22) is in
  [0.40, 0.60]; crossed if the ratio exceeds 1.3 or the exponent is below 0.35; else neither.
- **Octave factors** g_k = B/sqrtN(2^(k+1)) / B/sqrtN(2^k), k = 18..21.

Per cell (8 growths):
- **Cell verdict:** STALLED if at least half stalled; FLAT if at least 7/8 flat; CROSSOVER if
  at least 6/8 crossed; else UNDECIDED.
- **Feedback:** ACCELERATING if g_21 > g_18 + 0.01 in at least 6/8 growths; SETTLING if
  g_21 < g_18 - 0.01 in at least 6/8; else STEADY.
- **Remainder:** q(N) = sum over the cell of (interior 7s - interior 5s) / sum of boundary
  edges, at N = 2^20 and 2^22; R = q(2^22) / q(2^20). SHRINKING if R < 0.7;
  BOUNDARY-PROPORTIONAL if 0.7 <= R <= 1.4; BETWEEN if 1.4 < R < 1.6; AREA-LIKE if R >= 1.6
  (a remainder proportional to area against a boundary growing ~2.1x in two octaves gives
  R ~ 1.9). UNREAD if q(2^20) has no remainder.
- **Net disc law:** wave 9's pairs (radii 10-500 with a complete ball at most half the
  graph, lambda_net = (n7 - n5)/N): HOLDS if at least 90% within 10%.

## Predictions

- beta 0.5: **CROSSOVER** (95%). Feedback **ACCELERATING** (75%; STEADY 20%: the factor
  may approach the hyperbolic limit sqrt 2 and level off). Remainder BOUNDARY-PROPORTIONAL
  (60%), BETWEEN or AREA-LIKE (35%). Net disc law **MISSED** (65%).
- beta 0.75: **UNDECIDED** (45%), CROSSOVER (40%), FLAT (15%): continuing wave 9's factors
  (~1.05 rising ~0.01 per octave) gives a four-octave ratio near 1.3, at the crossed
  threshold. Feedback **ACCELERATING** (70%). Remainder BOUNDARY-PROPORTIONAL (55%). Net disc
  law **HOLDS** (70%).
- The reading I will give: ACCELERATING at both betas means local cancellation with a
  boundary-carried remainder only delays the crossover; `disloc` would then be flat at large
  N only where the remainder is essentially zero (beta >= 1.0 at these sizes). SETTLING at
  0.75 would mean a flat phase with a finite beta threshold between 0.5 and 0.75.

## Reporting

A FINDINGS subsection under Wave 9 ("Wave 10"), each verdict as frozen here, misses named;
BRANCHES' `disloc` remainder row updated. Limits stated: radii above 500 are not read by
the disc law; 8 seeds per cell.
