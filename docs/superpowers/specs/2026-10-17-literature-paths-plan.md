# Plan — working through the literature survey's new paths

**Date:** 2026-10-17. **Status:** PLAN (not a pre-registration). Each wave below gets its own
pre-registration, committed before its data are computed; each wave's results go into
FINDINGS before the next wave's pre-registration is written.

Source: `reports/Local graph growth literature survey.md` (2026-10-16), section "New paths,
deduplicated and prioritised", 29 paths in four groups: A re-analyses of saved data, B new
instruments, C rule variants, D controls and null models. The owner asked (2026-10-17) to
"get started on the new paths systematically".

## Waves

| wave | paths | why together | compute |
|---|---|---|---|
| 1 | A7 throat integer histogram; A1 curvature and failure-mode census; A6 "still a disc?" large-radius audit; A2 + A3 direct dead-corner hazard scan with an exact detector | test the dead-corner mechanism and the banked "tree-like beyond N*" label in curvature and rate units, mostly on saved graphs | ~2 h desktop |
| 2 | A4 local vs spectral roughness; A5 fjord-loop and neck census; B3 boundary process vs layered causal growth; D1 Eden with one frozen site | the edge: what the rough boundary is, with the null model for one permanent pin | small |
| 3 | A8 spectral gap count and spike decomposition; A9 effective resistance for the walker contrast; B1 running spectral dimension; D2 critical-tree controls; D3 Newman-Watts collapse of the portal threshold; D6 random-triangulation null; B4 box covering | instruments and their known-answer controls | moderate |
| 4 | D4 barrier-law discrimination; D5 majority tie-rule and schedule controls; C8 triadic plus decay; C9 anti-Hebbian reinforcement; C7 `ricci` as the Ni et al. flow; C10 hard-core square `geometrize`; B2 Ollivier curvature | the fixed-N rules | moderate |
| 5 | C1 dead-corner-free sheet; C3 proofreading veto; C5 motif-size programming; C4 reversible attachment; C6 cap scan with bounds; C2 fjord-mouth bridging | rule variants of `sheet` | moderate |

## Rule variants (wave 5)

The survey marks group C as the owner's decision because a variant changes what `sheet` is.
They will be implemented as **labelled, default-off options** of `sheet_growth` (and of
`rules.py` for C7-C10), so the default generator, every banked result and every saved graph
stay exactly as they are; each variant is reported under its own name and never as `sheet`.
If the owner prefers a different variant from the ones listed, that substitutes in wave 5
without affecting waves 1-4.

## Order inside a wave

Cheapest first; anything that grows fresh graphs is pre-registered with seeds, sizes and
readings fixed, and its detector validated on known cases before the run.
