# Step 5 readiness assessment — bootstrapping barrier under async updates

**Date:** 2026-08-03
**Status:** PRE-BRAINSTORM readiness note / proposal. **Not an approved spec.** Nothing here
is decided; it exists so the next session (or the owner) can start the step-5 brainstorm
from a grounded position. Written unsupervised after step 4 landed; needs owner approval on
the forks below before any implementation. (Mirrors `LORENTZIAN_SPIKE.md`'s "decision
document — nothing is committed" stance.)

## The question

Step 5 of the Lorentzian ladder (LORENTZIAN_SPIKE.md §5-6): **does the bootstrapping
barrier survive when time is emergent?** The banked flagship-negative (FINDINGS.md,
`barrier_scaling.py`) is `extent ~ N^α` with **α ≈ 0.13 for local rewiring rules**
(triadic/geometrize/ricci — extent stays ~log N, expander-like) vs **α ≈ 0.5 for the
`grown`/lattice positive control** (a true ~2D graph). The gap *is* the result. Step 5
re-runs it under asynchronous (event-driven) updates and asks whether α is unchanged.

This parallels step 4: a banked state-graph result, re-tested under emergent time, with a
pre-committed null of "schedule-invariant." Step 4's outcome (both P1 and P2 schedule-
invariant) makes invariance the leading expectation — extent is a geometric obstruction
like the schedule-invariant P1 — but α is a quantitative exponent, so async could still
shift it and it is worth measuring.

## What ports cleanly (already in hand)

- **The observable.** `barrier_scaling.estimate_extent` (double-sweep BFS diameter +
  lcc-fraction on the largest component) is graph-agnostic — it runs on the async-evolved
  **state graph** unchanged. The spike (LORENTZIAN §3a) already settled that no "extent on
  a DAG" definition is needed: the state graph still exists at every instant; measure its
  diameter as now, with time = events elapsed / mean updates per node (sweep-equivalents,
  exactly the step-4 unit).
- **The rewiring rule as an async event.** `triadic` is a validated async event
  (`async_engine.EVENTS`, conflict radius 3). The barrier held for triadic synchronously.
- **The fast path for scale.** `async_engine.run_batched` (conflict-free batches with
  rate-correcting thinning) exists and is validated for triadic. Sequential async is too
  slow at the N the α-fit wants; batched is the vehicle.
- **The α-fit + controls.** `barrier_scaling.py` already fits `extent ~ N^α` across an
  N-sweep with the `grown` positive control. Reuse its fitting and `estimate_extent`
  verbatim; only the *evolution* (sync sweep → async batched) changes.

## Real gaps / forks (for the brainstorm)

1. **Rule coverage.** The barrier was confirmed on **three** rewiring rules
   (triadic/geometrize/ricci), but only `triadic` is an async event today; geometrize and
   ricci are not implemented in `EVENTS` (both radius 3 — friend-of-friend writes).
   - *Fork:* (a) **triadic-only** async barrier (recommended first cut — representative
     rewiring rule, zero new event code, gets the α answer); vs (b) also implement + validate
     `_event_geometrize`/`_event_ricci` for full parity with the banked three-rule claim.
2. **Scale / AWS.** The α-fit needs a lever arm in N; the banked flagship reached ~10⁵
   (dynamics are not memory-capped). At 10⁵, batched triadic at radius 3 is the expensive
   checkpoint the spike explicitly flagged (~58 rounds/sweep from the step-2 cost model).
   - *Fork:* (a) **local ~2×10⁴** first (cheaper, enough lever arm for a first α estimate);
     escalate to (b) **10⁵ on AWS** only if the α CI needs the longer arm. The owner's "you
     have access to AWS" note points squarely at this run — but start local to de-risk the
     driver before spending cloud budget.
3. **Convergence budget.** Sync runs "to convergence"/fixed steps; async needs a matched
   sweep-equivalent budget (step-4 unit). Pre-commit it (e.g. run to the same
   mean-updates-per-node the sync sweep used, or to extent-plateau) before measuring.
4. **Causal-future-growth (the second half of step 5) — DEFER or exploratory-only.** The
   spike's alternative barrier restatement — "do light cones form?" via causal-future
   `|J⁺(e)|` growth (polynomial = manifold-like, exponential = expander) — rides on the
   causal DAG, which **step 3 retired as an absolute observable**. It survives only as a
   *relative* comparator (async-rewiring vs async-lattice, or async vs sync), never an
   absolute dimension.
   - *Recommendation:* ship the **state-graph extent** barrier first (clean, ports
     directly); treat causal-future-growth as an optional exploratory addendum with the
     non-manifold caveat front-and-centre, not a headline.

## Pre-committed null (to state before measuring)

α unchanged under async: rewiring α ≈ 0.13, control α ≈ 0.5, gap preserved. The interesting
outcome is any α *shift* — that would mean the flagship-negative was partly a synchronous
artifact (a banked result overturned — the kind of outcome step 4 pre-registered for but
did not find). Either way it is publishable; interpretation fixed before the run.

## Rough shape (if approved)

A thin `async_barrier.py` (parallels `async_censorship.py`): build initial graph → run
async **batched** triadic across an N-sweep to a matched sweep-equivalent budget → call the
existing `estimate_extent` → fit α, with the `grown`/lattice control on the same axes. Two
gates in the step-4 style: **Gate 1 (control)** the positive control (`grown`) recovers
α ≈ 0.5 under async (certifies the async driver measures extent-growth correctly); **Gate 2
(the question)** async triadic α vs the banked sync α ≈ 0.13. Reuses `estimate_extent`,
`create_initial_graph`, `run_batched` — small surface, like step 4.

**Estimated effort:** comparable to step 4 (one thin driver + a gate), *plus* whatever scale
decision (fork 2) costs. The batched-at-scale run is the only genuinely new expense.

## Recommendation

Ready to brainstorm. Suggested first cut: **triadic-only, local ~2×10⁴, state-graph extent
only, causal-future-growth deferred** — the cheapest path to the α answer. Escalate scale
(AWS) and rule coverage only if the first α estimate warrants it. Owner picks the forks.
