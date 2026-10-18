# Pre-registration — wave 7: do frozen corners decide the crossover under `strict`, growth by growth?

**Date:** 2026-10-18. **Status:** PRE-REGISTRATION, committed before any registered growth.
Follows wave 6 (FINDINGS "Wave 6"), owner's standing approval for overnight research.

## The claim under test

Wave 6 revised the trigger to "a permanently unclosed capped boundary node": under `strict`
no dead corner ever forms, but **frozen corners** do (a capped boundary node whose closure
`_closable` allows and the strict veto refuses, because closing would leave a dead corner
next door), and the sheet crosses at beta <= 1.0. That rests on group verdicts plus one
unregistered diagnostic growth (beta 0.5, seed 1: every frozen corner persisted, 20/20,
45/45, 103/103). If the revision is right it must hold **per growth**: a strict growth without
frozen corners stays flat, one that acquires them early crosses.

## Instrument

`sheet_variants.frozen_set(adj)` at every octave checkpoint of a `strict` growth (2^12 ...
2^20): the set of frozen corners (closure tentatively applied, `_makes_dead` checked,
reverted; the growth is untouched). Recorded per growth: the frozen count per checkpoint, the
first checkpoint with a frozen corner (octave resolution), and the persistence count (frozen
at one checkpoint and still frozen at the next, pooled). Smoke test on the diagnostic seed
(beta 0.5, seed 1, to 2^16, already seen): frozen 0, 0, 2, 20, 45, persistence 22/22, no dead
corner.

## Growths

`strict`, beta in {1.0, 1.125}, seeds 800-815 (32 growths), to 2^20. These betas are where
wave 6's strict growths split (beta 1.0: 6 crossed, 2 flat; beta 1.25: 7 flat, 1 crossed),
so both outcomes should occur.

## Frozen readings (flat and crossed as in wave 6: B/sqrtN factor 2^16 -> 2^20 in
[0.85, 1.15] and eccentricity exponent in [0.40, 0.60] = flat; factor > 1.3 or exponent
< 0.35 = crossed)

- **Gate:** no dead corner in any growth (else the strict cells are void).
- **P1:** among growths with **no frozen corner at 2^19**, at least 90% are flat at 2^20.
- **P2:** among growths whose **first frozen corner appears by 2^18**, at least 90% are crossed
  at 2^20.
- **FROZEN CORNERS DECIDE** if P1 and P2 both hold with at least 4 growths in each group;
  **UNDERPOWERED** if either group has fewer than 4; otherwise **NOT DECISIVE** (the per-growth
  table is then the result).
- Reported, no verdict: persistence (prediction: 100%), and the first-frozen octave against
  the octave in which B/sqrtN leaves its 2^16 value.

## Predictions

Gate PASS (95%). P1 holds (75%). P2 holds (70%). FROZEN CORNERS DECIDE (55%); UNDERPOWERED
(20%, if nearly every beta-1.0 growth freezes early and nearly every 1.125 growth never does,
one group could still be large enough; the risk is the other way, too few flat growths).
Persistence 100% (85%).

## Run

`python sheet_variants.py --scan7 --jobs 24` on jaga (usage-log CHECK-IN/OUT); `--read7 CSV`
re-reads with the same frozen code. Results go into FINDINGS, each verdict as frozen here.
