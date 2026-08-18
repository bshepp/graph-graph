# Stage 1 — wormhole-throat critical collapse (`throat_criticality.py`)

**Date:** 2026-08-17
**Status:** design approved (owner: program structure "looks great" 2026-08-11; dynamics
fork + FSS slicing locked same session; go-ahead 2026-08-17), pre-implementation.
**Program position:** stage 1 of the critical-collapse program (the Choptuik protocol
ported to the sandbox; trigger: Ecker/Ecker/Grumiller PRL 2026). Stage 2 (universality
across seed families) and stage 3 (driven injection) are gated on this stage's verdict —
see BRANCHES.md.

## Question and pre-registered readings

Does the censor-only sandbox exhibit **critical collapse** — a sharp threshold `A*` in
wormhole-throat thickness separating *evaporation* (all strands censored) from a
*permanent mutually-protected core* — or another crossover?

Pre-registered before any number is measured:

- **Sharp** (transition width shrinks as throat capacity grows): the sandbox's **first
  genuine critical point**; program proceeds to stage 2 (universality).
- **Crossover** (width flat in capacity): banked as *"no critical collapse in the
  censor-only rule family"* — consistent with every boundary probed so far (`prune`
  d(p), the barrier); stages 2–3 mostly dissolve. Fully publishable.
- Secondary predictions, registered now, reported either way:
  1. **Hybrid-transition signature**: the surviving-core size at threshold is bounded
     away from zero (discontinuous jump *with* critical scaling — the k-core/bootstrap
     universality expectation), rather than growing continuously from zero.
  2. **Critical slowing down**: mean evaporation time diverges as `A -> A*` from below
     (cascade lengthening), consistent with a logarithmic divergence.

## The physics core: protection and peeling

`prune` removes an edge iff its endpoints share no neighbor. A throat strand's only
defense is a **triangle across the throat**: strand `(u1, v1)` is protected when another
strand shares one endpoint and lands adjacent at the other (e.g. `(u2, v1)` with
`u2 ~ u1` in the ball). Two consequences drive the whole design:

1. **A spread bundle is provably trivial** under prune-only (no edge creation → no new
   protection → every isolated strand dies w.p. 1). Concentration at *both* ends is
   what makes survival possible — hence the throat geometry.
2. **The prune-only outcome is a deterministic fixed point.** An unprotected strand
   eventually dies (geometric rate); a strand protected by *survivors* never dies. The
   final state is the **bootstrap-percolation peeling fixed point** of the initial
   throat: iteratively strip strands with fewer than `min_overlap` common neighbors in
   the current graph (base + surviving strands); what remains is the core. Randomness
   sets *timing only*. The threshold therefore lives in the random throat-geometry
   ensemble — `P(core nonempty | A)` — and the dynamics reveal it and time it.
   (Known caveat: `prune`'s degree floor (`min_degree`) can block removals and break
   exact determinism — Gate 1 measures this rather than assuming it away.)

## Seed ensemble

- Base: `grown`, cap 6 (the banked portal-experiment substrate). Reference N=2000.
- Anchor balls: pick `c1` uniformly; pick `c2` uniformly among nodes with
  `dist(c1, c2) >= 2r + 6` (boundary-to-boundary >= 6, the long-portal regime; balls
  disjoint). Balls `B(c1, r)`, `B(c2, r)` by BFS on the base; reference `r = 2`.
- Throat at amplitude A: draw A distinct pairs uniformly from `B1 x B2` (all such pairs
  are non-adjacent by construction), added with the standard weight 0.5. **Capacity**
  `= |B1| * |B2|`; cross-geometry comparisons use strand **density** `a = A / capacity`.
- FSS geometries: `r = 2, 3, 4` with `N = 2000 / 5000 / 10000` (N grows only to keep
  `2r + 6`-separated balls plentiful; the FSS variable is throat capacity, not N).
- All distances are BFS hop counts; no node positions anywhere (project invariant).

## Slices (sequenced, gated — not alternatives)

- **1a — pilot (existence gate):** reference geometry, peeling `P(core | A)` over an A
  scan. Gate: both outcomes occur with an S-curve between them. If P is 0 or 1
  everywhere reachable (`A <= capacity`), STOP and redesign the ensemble before
  anything else.
- **1b — the measurement:**
  * **Peeling-FSS**: `P(core | a)` at all three geometries, ~2000 ensemble draws per
    a-point (peeling is milliseconds — no dynamics). Transition width `w(r)` = a-span
    where P goes 0.1 -> 0.9 (direct interpolation on the monotone curve; a logistic fit
    reported as cross-check). **Sharpness verdict**: `w` decreasing with capacity
    (monotone across the three), vs flat-within-errors = crossover.
  * **Jump test**: distribution of core size (as fraction of A) just above threshold,
    per capacity — bounded away from 0 = hybrid signature.
  * **Dynamics anchor** (reference geometry only): stochastic prune-only runs (banked
    params: `prune_prob 0.05, min_overlap 1, min_degree 2`; `T = 400` sweeps ~ 20
    geometric lifetimes), seed-by-seed comparison of the surviving-strand set against
    the peeling fixed point — **Gate 1**. Plus the timing observable: evaporation time
    (last strand death, sweep-equivalents) vs `a` below threshold — the
    slowing-down test.
  * **Triadic rider** (descriptive — **Gate 2**, same seeds, `triadic`+`prune` both at
    the banked rates): can weaving *rescue* a throat the peeling fixed point condemns
    (sub-`A*` survival above the prune-only expectation), and does churn *demolish*
    super-`A*` cores? Pre-registered both ways; never gates the exit code.
- **1c — contingency:** full-dynamics FSS **only if** Gate 1 fails beyond the
  degree-floor accounting (i.e. peeling provably does not predict the dynamics — the
  cheap path is then untrusted and every capacity pays for dynamics).

## Gates (exit-code semantics, step-4 style)

- **Gate 1 (instrument + mechanism, gates the exit):** at the reference geometry the
  dynamics' final surviving-strand set equals the peeling fixed point, seed by seed;
  mismatches are counted and must be attributable to the degree floor (checked
  explicitly per mismatch: the blocked removal's endpoint sat at `<= min_degree`).
  Threshold frozen after the first run.
- **Gate 2 (the race, descriptive):** printed comparison + pre-registered verdict text,
  both directions written before data; never affects the exit code.
- Slice-1a's existence gate governs whether 1b runs at all.

## Driver and validation

**New module `throat_criticality.py`** (parallels `async_censorship.py` in structure):
`build_throat(...)` (base + balls + strands), `peel(...)` (the fixed-point computation),
`run_dynamics(...)` (prune-only / triadic+prune with per-strand death-time tracking via
`run_sequential` / `run_sequential_multi` — reusing the step-4 engine), the ensemble
sweeps, and `_validate()`:

1. **Hand-built known-answer throats** (positive + negative): a constructed
   mutually-protecting 2-strand configuration whose peeling core is exactly those 2
   strands, and a constructed unprotected configuration whose core is empty. Exact
   equality required.
2. **Peeling == dynamics** at small scale (the Gate-1 check, N=500-1000, several
   seeds), with the degree-floor accounting printed.
3. **A miniature 1a pilot** (small ensemble) demonstrating both outcomes occur.
`--validate` must run in a few minutes; the production sweeps are CLI runs (local for
the pilot; jaga for the 2000-draw FSS ensembles if convenient — power button required).

## Scope / non-goals

**In:** the driver, slices 1a-1b (peeling-FSS + dynamics anchor + triadic rider), the
two gates, FINDINGS/BRANCHES updates either way.

**Out (deferred):** stage 2 (universality across families — gated on a sharp verdict);
stage 3 (driven injection); any async-schedule variation beyond the step-4 engine's
sequential runners; anything causal-DAG; the d(p) mechanism hunt (separate thread —
the shared protection mechanism is noted as cross-pollination, not a dependency).

## Explicit pre-commitments

1. The sharp-vs-crossover verdict comes from `w(r)` monotone shrinkage across three
   capacities — not from any single geometry's S-curve steepness.
2. Gate-1 mismatch accounting: only degree-floor-attributable mismatches are tolerated;
   the tolerance is frozen after the first run, not retuned.
3. The jump and slowing-down tests are secondary — reported either way, never used to
   rescue a failed sharpness verdict.
4. Ensemble sizes, grids, and thresholds are frozen once the pilot fixes the
   interesting `a`-range; widening is allowed only for coverage, never re-centering on
   a preferred answer.
5. A crossover verdict is banked with the same prominence as a critical point would be.
