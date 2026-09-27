# Step 5 pre-registration — the bootstrapping barrier under asynchronous updates

**Date:** 2026-09-27
**Status:** PRE-REGISTRATION. Committed before any step-5 data exists. Owner delegated the
forks ("do the three candidates", sidequests approved, 2026-09-26); the forks below are
taken at the readiness note's recommended values and are listed so they can be overruled.
**Supersedes** the gate sketch in `2026-08-03-step5-barrier-async-readiness.md`, which
contained three errors found in the 2026-09-26 review (see "Corrections" below).

## Question

Does the banked flagship negative — fixed-N local rewiring cannot grow extent from an
expander (`extent ~ N^alpha`, rewiring `alpha ~ 0.13` vs 2D lattice `0.515`,
`barrier_scaling.py`) — survive when the global synchronous sweep is replaced by
event-driven Poisson-clock updates?

## Corrections to the readiness note (made before design, not after data)

1. It called `grown` the alpha ~ 0.5 positive control. Banked `grown` alpha is **0.194**;
   the alpha ~ 0.5 control is the **lattice**.
2. Its Gate 1 ("the control recovers alpha ~ 0.5 *under async*") is vacuous: lattice,
   `grown` and `none` are **static** series in `barrier_scaling.py` — no rule is applied,
   so they cannot depend on the schedule. A schedule gate needs a *dynamic* control.
3. It said the banked fit reached N ~ 1e5. The banked grid is N = 1000..16000.
4. It assumed sequential async is too slow and batched is the vehicle. Measured:
   sequential is ~70 us/event flat in N, so N = 32000 x 200 sweeps is ~7.5 min per run;
   batched (conflict matrix rebuilt every round, ~58 rounds/sweep at radius 3) is slower.
   The **sequential reference path** is used — it is also the path that needs no
   batching-validity argument.

## Forks taken

- Rule coverage: **triadic only** (the one rewiring rule that exists as an async event).
- Scale: **local**, N = 1000, 2000, 4000, 8000, 16000, 32000 (banked grid + one octave).
- Budget: **200 sweep-equivalents** (absolute Poisson time 200 at rate 1), matching the
  banked 200 synchronous steps; per-edge opportunity is matched by edge ownership.
- Causal-future growth: **deferred** (rides on the retired causal-DAG observable).
- Seeds: **8 paired seeds** (0..7); both schedules start from the identical ER graph.

## Frozen protocol

- Start graph: `create_initial_graph(N, 'random', k=6, seed=s)` after seeding `random`
  and `numpy.random` with `s` (exactly `barrier_scaling.run_trajectory`).
- Sync arm: `barrier_scaling.run_trajectory('triadic', ...)` verbatim, interval 50.
- Async arm: `async_engine.run_sequential_multi(['triadic'], [1.0], max_time=200)`,
  extent recorded at Poisson times 50/100/150/200.
- Observable: `barrier_scaling.estimate_extent` (double-sweep diameter on the largest
  component, plus `lcc_frac`), global numpy RNG re-seeded immediately before each
  measurement so the estimator's source draws are schedule-independent.
- Fit: per-seed log-log slope alpha_s over the six N, for each arm; also the pooled
  fit of seed-mean extents (the banked statistic).

## Gates and frozen verdict rules

**Gate 0 (reproduction).** The sync arm for seeds 0..2 at N = 1000..16000 must reproduce
the banked `results/barrier_scaling_20260530_094041.csv` triadic rows **exactly**
(step-0 diameters; final diameters up to the estimator's RNG state, which the banked
driver did not re-seed — so step-0 equality is required and final-step agreement within
+-2 is required). Failure stops the run.

**Gate 1 (dynamic positive control).** `prune` on `small_world` (k=6, p=0.1) *does* grow
extent (it reveals the latent ring). Both schedules, same 200-sweep budget, N = 1000..8000,
4 seeds. PASS requires: both arms give pooled alpha > 0.5, and the two arms' pooled
alphas differ by < 0.1. This certifies that the async driver can produce and measure
polynomial extent growth when it exists — so a low async triadic alpha is not an
instrument floor.

**Gate 2 (the question).** With D_s = alpha_async,s - alpha_sync,s over the 8 paired
seeds, mean D and its standard error SE (t-interval, 7 dof, t_crit = 2.365):

- **BARRIER SURVIVES** iff the 95% upper bound on mean alpha_async is **< 0.25** (half
  the lattice exponent). This is the physics verdict.
- **SCHEDULE-INVARIANT** iff the 95% CI of D lies inside **[-0.05, +0.05]**.
- **SHIFT** iff the 95% CI of D excludes 0 **and** |mean D| >= 0.05.
- otherwise **UNDERPOWERED** for the invariance question (the survival verdict may
  still be decided). No seeds are added after the fact to rescue a verdict; a follow-up
  with more seeds would be a new pre-registration.

**Pre-committed null:** barrier survives and alpha is schedule-invariant.
**Interesting outcome:** SHIFT, or async alpha CI reaching 0.25 — the flagship negative
would be partly a synchronous artifact.

## Descriptive riders (no verdict attached)

- `lcc_frac` and extent-vs-time trajectories per arm (does extent plateau inside 200?).
- A long-budget check: 800 sweep-equivalents at N = 1000, 4000 (both arms, 4 seeds), to
  show whether the 200-sweep extent is converged. The review noted the banked budget was
  never tested for convergence; if extent is still growing at 200, alpha at fixed budget
  is a statement about that budget, and that will be said.
- Static references (lattice / none / grown) are reported for context, labelled static.

## Known limits, stated up front

- Diameters here are integers in the range ~8-16, so per-seed alpha is coarse; this is why
  the verdict uses 8 paired seeds and why the banked triadic fit had R^2 = 0.53.
- One rule. `geometrize`/`ricci` are not async events; the three-rule sync claim is
  tested under async for `triadic` only.
- Extent is a state-graph observable; nothing here uses the causal DAG.
