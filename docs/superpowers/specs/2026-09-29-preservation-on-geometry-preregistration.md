# Pre-registration — preservation of a real dimension under the local rules

**Date:** 2026-09-29
**Status:** PRE-REGISTRATION, committed before the runs. Sidequest (owner: sidequests
approved), the substrate swap for the banked "Preservation" table, which was measured on
`grown` and is stale under the window-stability gate.

## Question

The banked table (FINDINGS "Preservation") said `grown`'s dimension is a stable fixed
point under `prune` and the state rules and is eroded by `triadic` and `rewire`. `grown`
has no dimension. Does a real one behave the same way?

## Frozen protocol

`python track_dimension.py --topology {triangular, sheet} --nodes 10000 --rules {prune,
majority, triadic, rewire} --steps 200 --track-interval 20 --max-radius 10 --seed {0,1,2}`
(fast backend). Read at t = 0 and t = 200: `defined_frac`, median `d_eff`, `median_drift`,
`window_stable`. 24 runs.

## Predictions, frozen

| rule | prediction | reason |
|---|---|---|
| `majority` | PRESERVED: stable at both ends, defined_frac and d_eff unchanged within 0.05 | topology untouched |
| `prune` | PRESERVED, and *exactly* inert (edge count unchanged) | every edge of a triangulation lies in a triangle, so nothing has zero overlap |
| `triadic` | ERODED: window-stable at t = 0, UNSTABLE by t = 200, defined_frac down by > 0.3 | friend-of-friend rewiring puts edges across the plane; E1 showed triadic churns 85% of a 2D fabric in 120 steps |
| `rewire` | DESTROYED: UNSTABLE within the first 40 steps, defined_frac < 0.1 at the end | random shortcuts at prob 0.01 per edge per step |

Readings are PASS / FAIL per row against these thresholds; no composite verdict.

## What it adds

The banked reading "the dimension verdict tracks extent" was made on a graph with no
dimension. If the same pattern holds on the lattice and the sheet, the *reading* was
right for the wrong graph; if `triadic` preserves a real dimension, the banked
"crumpling" was a tree artifact like the walker result.
