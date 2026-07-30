# Step 3 — static-graph causal calibration (`causal_dag.py`)

**Date:** 2026-07-29
**Status:** design approved, pre-implementation
**Ladder position:** Rung 1 of the de-toying ladder, stage 3 of `LORENTZIAN_SPIKE.md` §6.
Last instrument stage before first physics (step 4, censorship checkpoint).

## Goal

Build the missing bridge between the two instruments already in hand and use it to
run the free calibration the spike identified: on a **static** graph the causal set
of update events must have dimension `d_causal = d_H + 1`. If it does not, the
instrument is broken, not the physics (`LORENTZIAN_SPIKE.md` §1, §5).

Prerequisites are all green and gated:

- `causal_sets.py --validate` — MM + midpoint estimators, calibrated on flat-space
  Minkowski sprinklings (2.000 / 2.988 / 3.976).
- `async_engine.py --validate` — Poisson-clock async engine; `run_sequential` already
  runs the Harris construction and tracks event times, but discards the parent
  structure.
- `dimension.py --validate` — ball-growth Hausdorff `d_H`.
- the `grown` generator and `lattice` topology.

Neither existing module bridges the two: `async_engine.py` never records the event
DAG, and `causal_sets.py`'s estimators consume a boolean relation matrix `R` built
from *sprinkled coordinates*, not from a run. This spec is that bridge.

## Pre-committed null and gate structure

The pre-committed null is `d_causal = d_H + 1`. Stated now, before any number is
measured, so no story can be fitted later. A null (agreement) result is the
*expected* and publishable outcome — it certifies the instrument.

Two-tier gate (owner's choice, 2026-07-29):

| tier | graph | `d_H` | required `d_causal` | role |
|------|-------|-------|--------------------|------|
| **Gate 1 — known answer** | 2D lattice | 2 (exact) | ≈ 3 | clean integer target; exposes any MM systematic on graph×time causal sets |
| **Gate 1 — known answer** | 3D lattice | 3 (exact) | ≈ 4 | second integer anchor |
| **Gate 2 — tracking** | `grown` cap 6 | ≈ 2.2 | `d_H + 1` | application on the real (non-integer) target |
| **Gate 2 — tracking** | `grown` cap 8 | ≈ 3.6 | `d_H + 1` | second point — require the *offset* ≈ +1 across the pair, not one point |
| **Cross-check (all)** | — | — | — | MM and midpoint must agree on the event DAG, as they do in flat space |

The lattice anchors carry the falsifiability: two clean integer known-answers. `grown`
tests that the offset holds where `d_H` is non-integer. Requiring the offset to hold
*across* two caps is stronger than a single-point tolerance and matches the project's
cross-validation style (`dimension.py` vs `lattice`, `coherence.py` vs a permutation
null, `ctqw_time_average` vs Krylov).

## Architecture

New module **`causal_dag.py`** — the seam between three existing modules, following the
repo's one-purpose-per-module convention:

- imports the scheduling primitive idea from `async_engine.py` (Harris construction;
  `apply_event`, `EVENTS`);
- imports the estimators from `causal_sets.py` (`ordering_fraction`, `mm_dimension`,
  `midpoint_dimension`) verbatim;
- imports `fast_dimension_field` / `dimension_stats` from `dimension.py` for the `d_H`
  baseline.

`async_engine.py` stays pure-dynamics; `causal_sets.py` stays pure-flat-space; the new
file holds all the DAG-specific logic.

Rejected alternatives: extending `async_engine.run_sequential` with a `record_dag`
flag (couples two modules' concerns, bloats both); and an analytic shortcut that
approximates the static causal relation as `graph_distance ≤ Δtime` without
materialising the DAG (bakes in an approximation and, fatally, does not reuse for
step 4, where the DAG must encode *dynamic* topology history — the same measurement
code has to serve both steps).

## Components

### A. Event-DAG recorder

`record_event_dag(G, rule, n_events, seed, rate=1.0) -> EventDAG`

Runs the Poisson-clock loop (identical order semantics to `run_sequential`: each node
holds an independent exponential clock, `argmin` selects the next event). At each event
`(v, event_id)`:

1. snapshot the **current** closed neighbourhood `R(e) = {v} ∪ N(v)`;
2. record parents `= [last_event[u] for u in R(e) if last_event[u] >= 0]` — this
   includes `v`'s own previous event, so each node's worldline is a timelike chain;
3. `apply_event(G, v, rule, seed, event_id)`;
4. `last_event[v] = event_id`; advance `v`'s clock.

Returned `EventDAG` holds: `parents` (list per event), `event_nodes` (node per event),
`times`, and lazily-built `children` (reversed `parents`).

**Rule-independence on a static graph:** parents come from the read-set *footprint*,
not the state *values*, so on a topology-preserving rule (`activation`) every state
rule yields the same DAG. Step 3 uses `activation`; a fixed point does not distort the
DAG because events keep firing and keep reading the neighbourhood regardless of whether
a value changed. The identical recorder serves step 4, where a rewiring rule mutates
`N(v)` between events and the per-event snapshot captures that history — that is where
the causal structure stops being trivial.

### B. Reachability and interval sampling

- `future(p)` / `past(q)`: **bounded-depth** directed BFS (children / parents), stopping
  once depth exceeds the max sampled interval height. Cost scales with interval size,
  not total event count — this is what keeps the sampler affordable at 10^5–10^6 events.
- Alexandrov interval `I(p,q) = future(p) ∩ past(q)` (inclusive of `p`, `q`).
- `sample_intervals(dag, rng, n_pairs≈200, min_cardinality=64, ...)`: sample `p`, pick
  `q` from `p`'s bounded future across a range of DAG-depths, keep intervals with
  cardinality in `[min_cardinality, max_cardinality]`.
- **min-interval-count gate**: refuse a verdict (`nan`) if fewer than `min_intervals`
  (~20) qualifying intervals, exactly as `midpoint_dimension` does. The lesson from
  step 1 — raising the cardinality floor starves the sample and injects noise — applies
  here: the gate that matters is the sample *count*, diagnosable without knowing the
  answer.

### C. Estimator wiring

Per sampled interval:

1. build the local transitive-closure relation matrix `R` — `R[a,b] = 1` iff
   `b ∈ future(a)` restricted to interval elements (small, ≤ few hundred, so a dense
   boolean matrix is fine);
2. `mm_dimension(ordering_fraction(R))` — **primary**;
3. `midpoint_dimension(R, rng)` — **cross-check**.

Aggregate across intervals: report median ± spread for each estimator. MM's non-manifold
signal (`ordering_fraction > r(0.5)` → `mm_dimension` returns `nan`) is counted and
reported as a `defined_frac`-style honesty signal, never silently dropped.

### D. `d_H` baseline, apples-to-apples

`d_H` from `dimension.py`'s `fast_dimension_field`, measured at the **same local radius**
the causal intervals probe. Pre-commitment: the causal intervals are *local* (bounded
height), so the comparison is against the **local ball-growth Hausdorff** dimension, not
`grown`'s global extent dimension — the two differ on `grown` (local `d_eff` ≈ 2.2 vs
diameter ~ N^0.19). Matching the radius scale makes the `d_H + 1` comparison honest.

## The scientific risk (why this stage has content)

The graph light cone is **first-passage percolation** on `G`: information spreads one hop
per update-generation, Poisson-timed. Its limiting shape is the FPP shape (Cox–Durrett) —
**anisotropic, with a preferred frame** — *not* the Lorentz-symmetric Minkowski cone MM
was derived for. So `d_causal = d_H + 1` can fail specifically because the ordering
fraction is light-cone-shape-sensitive.

That failure is itself the finding: it would mean emergent-causal-set work needs a
shape/frame correction to MM (indicts the estimator, per §5, not the model). The lattice
anchor is exactly the test — a clean 2D-lattice manifold *should* return 3; if it does,
the anisotropy is benign for the ordering fraction and we proceed to `grown`; if it does
not, we have learned something real and cheap before spending any physics budget.

Note the Poisson clocks are load-bearing here (`LORENTZIAN_SPIKE.md` §1): the *time*
direction is a genuine sprinkling, so only the *spatial* light-cone shape is at issue,
not a doubly-regular lattice (which would be non-manifold-like in both directions and
break MM outright).

## Validation (`python causal_dag.py --validate`)

A single self-check, in the style of the other two rungs, seeded and reproducible:

1. **Gate 1:** 2D lattice → `d_causal` within tolerance of 3; 3D lattice → within
   tolerance of 4. Tolerance to be fixed during implementation against the observed
   spread, then frozen (the flat-space precedent: MM to ~1%, midpoint +4% at d=4).
2. **Gate 2:** `grown` cap 6 and cap 8 → `(d_causal − d_H)` ≈ 1 for both, offset
   consistent across the pair.
3. **Cross-check:** MM and midpoint agree on every graph tested.
4. Report `defined_frac` (fraction of intervals returning a finite MM dimension) and the
   qualifying-interval counts for each graph.

Run scale (tune in implementation to keep `--validate` to a few minutes, like the other
gates): lattices ~400–1000 nodes; `grown` ~1000–2000 nodes; `n_events` chosen so sampled
intervals reach ≥64 elements with ≥20 qualifying intervals. Causal measurement is
memory-capped ~10^4 nodes (`LORENTZIAN_SPIKE.md` §4); calibration runs well under that.

## Scope / non-goals

**In:** the recorder, bounded-BFS reachability + interval sampling, the estimator bridge,
the `d_H` baseline, the two-tier `--validate` gate.

**Out (deferred):**

- Vectorised batch *application* in the recorder — the Python loop is fine at calibration
  scale, and causal measurement is memory-capped anyway. This validates scheduling, not
  vectorised-application speed (the same honest caveat carried since step 2).
- Any dynamic-topology rule (`prune`, `triadic`) — that is step 4+. Step 3 is static by
  definition.
- Curved-graph / curved-spacetime tests — rung 2 remainder, separate work.
- The causal-future-growth (`|J⁺|`) barrier restatement — step 5.

## Explicit pre-commitments (stated before measuring)

1. `d_causal = d_H + 1` is the null; agreement certifies the instrument and is the
   expected outcome.
2. `d_H` is the **local ball-growth** dimension at the causal-interval radius scale, not
   the global extent dimension.
3. The offset must hold across **both** `grown` caps, not at a single point.
4. A lattice failure (≠ integer+1) is reported as an instrument limitation (light-cone
   anisotropy), not hidden by widening tolerance — and it is a legitimate, publishable
   negative.
5. MM is primary and gated on the known answer; midpoint is the cross-check and gated on
   agreeing with MM (different jobs, different gates — the step-1 lesson).
6. Tolerances are frozen after the first calibration run and not retuned to force a pass.
