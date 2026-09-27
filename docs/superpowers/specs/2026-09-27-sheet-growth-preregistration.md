# Pre-registration — `sheet`: can strictly local growth produce a real dimension?

**Date:** 2026-09-27
**Status:** PRE-REGISTRATION of a confirmatory run, committed before it starts. This is a
sidequest (owner: "sidequests are approved", 2026-09-26). It adds a generator; it changes
no existing rule or result.

## Origin

The 2026-09-26 audit found that `grown` has no dimension: its frontier is a constant 37%
of N, so ball growth is exponential. The structural reason — nothing makes growing arms
meet — suggests what a local rule would need. `sheet_growth.py` implements three local
ingredients (manifold attachment, wheel closure at degree `c`, coordination-dependent
rates with strength `beta`); see its docstring.

## What has already been seen (exploratory, disclosed)

Scratch prototypes, seeds 0-2, N up to 32000 (beta = 0 and 2 to 128000):

| beta | diameter exponent | boundary exponent | reading |
|---|---|---|---|
| 0 | 0.18 → 0.10 | 1.00 | tree-like, like `grown` |
| 0.5 | 0.28 → 0.17 | 1.00 | tree-like |
| 1 | 0.35 → 0.26 | 0.88 → 0.98 | 2D-like at small N, tree-like beyond |
| 2 | 0.52, 0.50, 0.50 | 0.51, 0.51, 0.51 | 2D-like through N = 128000 |

Also seen: closure threshold c = 5 closes into a 12-node graph in 5 of 5 seeds (the
icosahedron), and c = 7 at beta = 0 is tree-like.

The beta = 1 row is the warning: it looked two-dimensional until N grew. So beta = 2 may
be the same crossover with a longer persistence length. **The confirmatory run exists to
find out**, at 4 times the largest exploratory size, on seeds never used, and judged by the
window-stability and spectral instruments rather than by two exponents.

## Frozen protocol

- `python sheet_growth.py --betas 0 1 2 3 --caps 6 --nodes 512000 --seeds 3 --seed 100`
  and a curvature control `--betas 2 --caps 7 --nodes 512000 --seeds 3 --seed 100`.
- Checkpoints at N = 2000, 8000, 32000, 128000, 512000 within each growth.
- Per-octave exponents from seed-mean diameter (double sweep) and boundary edge count.
- At N = 512000, for beta = 2 and 3: `window_stability.audit` and
  `spectral_flow` verdicts on each seed's graph.

## Frozen verdict rules, per (c, beta)

- **GEOMETRY** requires all of:
  1. diameter exponent in [0.42, 0.58] on **both** of the last two octaves
     (32000 → 128000 and 128000 → 512000);
  2. boundary exponent <= 0.65 on both of those octaves;
  3. `window_stability` reads DIMENSION DEFINED with final-window `d_eff` in [1.8, 2.2]
     on every seed;
  4. `spectral_flow` reads FLAT with `d_s` in [1.8, 2.2] on every seed.
- **CROSSOVER** (a finite persistence length, not a dimension): the boundary exponent
  rises by more than 0.1, or the diameter exponent falls by more than 0.1, between the
  last two octaves.
- **TREE-LIKE**: boundary exponent >= 0.9 on the last octave.
- **STALLED**: growth stops short of 512000 in any seed; reported with the sizes.
- Anything else: **UNCLASSIFIED**, reported as such.

## Predictions

- beta = 0: TREE-LIKE. beta = 1: TREE-LIKE or CROSSOVER.
- beta = 2: GEOMETRY is the hoped-for outcome; CROSSOVER is the honest rival, and I put
  it at roughly even odds given what beta = 1 did.
- beta = 3: GEOMETRY if beta = 2 is; possibly STALLED (rates this skewed can deadlock).
- c = 7, beta = 2: not GEOMETRY — negative curvature should give exponential growth
  whatever the rates do. If c = 7 reads GEOMETRY, the curvature interpretation is wrong.

## What a positive would and would not mean

It would be the project's first case of a strictly local rule **producing** a
window-stable dimension, as opposed to revealing a latent one or reading a fixed window.
It would not be a spontaneously selected dimension: `c = 6` selects flatness and `beta`
selects compactness, so it is emergence with two knobs. If interior vertices all have
degree 6 the result is a patch of the triangular lattice assembled without coordinates;
the interior degree census is reported so that can be said plainly.
