# Step 4 — shortcut censorship under async updates (`async_censorship.py`)

**Date:** 2026-08-02
**Status:** design approved, pre-implementation
**Ladder position:** Rung 1 of the de-toying ladder, stage 4 of `LORENTZIAN_SPIKE.md` §6.
**First physics stage** — the first checkpoint that tests a banked physics result rather
than calibrating an instrument.

## Goal

Re-run the banked **shortcut censorship** result under **asynchronous (event-driven)**
updates and ask whether it survives when time is emergent rather than a global sweep
clock. The synchronous result (FINDINGS.md §"Censorship: threshold and advantage-blind",
`shortcut_censorship.py`) has two parts:

- **P1 — threshold + advantage-blind.** Pure `prune` deletes long portals at the base
  geometric rate (≈ `1/prune_prob` steps), detour-2 portals are immune, removal time is
  *uncorrelated* with a portal's transport advantage, and collateral on the `grown`
  fabric is zero. (Sync: long survival 0.03, detour-2 1.00, ⟨t⟩ 18.1, rank-corr −0.07,
  collateral 0.000.)
- **P2 — self-stabilization.** With `triadic` running alongside `prune`, ~2/40 long
  portals get *woven in* — triadic lays a parallel edge, forming a triangle *on* the
  portal, after which `prune` cannot see it — raising long survival 0.03 → 0.12. (Sync:
  long survival 0.12, detour-2 survival collapses to 0.22, collateral 0.584, woven ≈ 2.)

This step is unblocked by the step-3 negative. Censorship acts entirely on the **state
graph** (`prune`'s locality), so it needs neither the causal DAG nor the causal-set
dimension the step-3 result retired. The retired instrument stays out of this module
completely.

Prerequisites are all green and gated:

- `async_engine.py --validate` — Poisson-clock async engine. `prune` is a validated event
  (conflict radius 2); `triadic` too (radius 3). `run_sequential` runs one rule at a time.
- `shortcut_censorship.py` — the synchronous experiment and the banked P1/P2 numbers.
- `inject_shortcuts` (`shortcuts.py`) and `create_initial_graph` (`simulation.py`) —
  the identical portal-injection and base-graph construction both schedules will share.

## Why the checkpoint has content

The pre-committed null (LORENTZIAN_SPIKE.md §5) is *"threshold-type and advantage-blind,
as synchronously"* — censorship is a property of `prune`'s locality, not of the update
schedule. The stated interesting outcome is any **disagreement** between a synchronous
result and its async counterpart: that would mean a banked result was a synchronous
artifact.

The two parts sit very differently on that axis:

- **P1 is nearly schedule-invariant by construction.** `prune` removes an edge iff its
  endpoints share no neighbour; a long portal has zero common neighbours regardless of
  firing order, so advantage-blindness cannot depend on the schedule. P1's *one* place
  to bite is the degree-floor coupling (`min_degree`): a removal at `v` lowers a shared
  neighbour's degree, which a later event reads. Step 2 showed this makes async `prune`
  differ from sync *trajectory-by-trajectory* while agreeing *in distribution* on generic
  observables — but never checked the portal-survival observables specifically. Gate 1
  closes that gap and, in doing so, becomes the control for Gate 2.

- **P2 is exactly where async can bite.** Synchronously, every step runs
  `triadic`-*then*-`prune` in lockstep: triadic can lay the protective triangle and prune
  gets its shot at the now-protected portal *within the same step*. Async has no fixed
  intra-step ordering — a prune event can fire on a portal before triadic ever weaves it.
  So the "woven-in" protection could be a **synchronous-ordering artifact** that async
  dissolves. That is the real physics content of this checkpoint.

## Approach — merged Poisson clocks, sequential

The concurrent `triadic`+`prune` race is realized as **two independent Poisson clocks per
node** (a prune-clock and a triadic-clock), merged into one event stream by `argmin` over
all `2N` clocks; each event applies its own rule via the existing per-event RNG. This is
the honest continuous-time analog of two processes running concurrently, and it dissolves
the lockstep ordering precisely the way P2 needs to be tested.

Sequential (one event at a time) is deliberate: it is correct by construction and sidesteps
the mixed conflict-radius problem that batching two different-radius rules would introduce.
At N=2000 × ~120 sweep-equivalents (~2.4×10⁵ events, each O(degree)) it runs in
seconds/seed — batching buys nothing at calibration scale.

Rejected alternatives:

- **Single clock, per-event rule coin-flip** — distributionally close but couples the two
  rates awkwardly and is less faithful to "two independent processes."
- **Batched fast path with a mixed radius-3 conflict rule** — needed only for large-N
  throughput (the barrier's N-sweep, step 5) and adds a mixed-radius correctness argument
  we do not need here. YAGNI for a sampled N=2000 checkpoint.

## Architecture

**New module `async_censorship.py`** (parallels `shortcut_censorship.py` +
`async_engine.py`; named `async_*`, not `causal_*`, because it is a pure state-graph
checkpoint — the causal DAG is not involved). It:

- imports `create_initial_graph` (`simulation.py`) and `inject_shortcuts` (`shortcuts.py`)
  to build the identical base + portals both schedules share;
- imports the async runners (`run_sequential`, the new `run_sequential_multi`) and
  `EVENTS` from `async_engine.py`;
- imports `run_condition` from `shortcut_censorship.py` as the **single source of truth**
  for the synchronous side — the banked semantics are not re-implemented.

**One small engine extension** — add to `async_engine.py`:

`run_sequential_multi(G, rules, rates, n_events=None, max_time=None, seed=0)
-> (nx.Graph, np.ndarray, np.ndarray)`

The merged-clock runner: `len(rules)` independent exponential clocks per node, `argmin`
over the full `len(nodes)*len(rules)` clock array, each event dispatched to its rule with
`apply_event`. Stops at whichever of `n_events` / `max_time` is given (the censorship
conditions pass `max_time=sweeps`). Returns the mutated graph, the per-event absolute
times, and a per-event rule-index array (needed to attribute events by rule). It belongs in
`async_engine.py` — the module docstring already names non-uniform / multi-rate clocks as
"the natural next generalisation," and `run_sequential` becomes the `len(rules)==1` special
case.

This extension earns a **teeth-y equivalence check** folded into `async_engine --validate`.
With a single clock (`len(rules)==1`) the clock array is the same shape as
`run_sequential`'s, so `run_sequential_multi` must draw its Poisson clocks in the same order
(initial `rng.exponential` per node, then `argmin`, then the winner's re-draw) — under that
constraint it must return a graph **bit-identical** to
`run_sequential(G, rule, n_events, seed, rate)` for a matched seed. Bit-identity (not just
distributional agreement) is the bar because the shared clock layout makes it achievable,
and it catches any drift of the generalisation from the validated single-rule path.

## Components (`async_censorship.py`)

### A. Shared setup

`build_base_and_portals(n, cap, n_long, n_detour2, seed) -> (base, long_portals, detour2)`
— one `grown` base + `n_long` long portals (base distance ≥ 6, "advantage" = injection
distance) + `n_detour2` detour-2 controls (base distance exactly 2), via the existing
`inject_shortcuts`. The **same** returned base+portals feed both schedules for a seed, so
the comparison is paired.

### B. Async conditions

- `async_prune(base, portals, sweeps, prune_prob, seed) -> result` — `run_sequential`
  with the `prune` event, run until absolute Poisson time reaches `sweeps`.
- `async_triadic_prune(base, portals, sweeps, prune_prob, rewire_prob, seed) -> result` —
  `run_sequential_multi([triadic, prune], rates=[1, 1], ...)`, run until absolute Poisson
  time reaches `sweeps`; `rewire_prob` / `prune_prob` are passed as the events' own
  parameters, not as clock rates.

Portals are tracked *during* the run: a portal `(u,v)` is stamped removed at the first
event after which `not G.has_edge(u,v)`, its removal time recorded as the **absolute
continuous Poisson time** of that event (see Time-matching — with the prune-clock at rate
1, one time unit = one prune-sweep-equivalent). (Checking only after events incident to
`u` or `v` is sufficient and keeps tracking O(events), not O(events·portals).)

### C. Observables — identical to the synchronous script

Per condition, aggregated over seeds: long survival, detour-2 survival, mean removal time
(sweep-equivalents), Spearman rank-corr(advantage, removal-time) among removed long
portals, woven-in count (surviving long portal that acquired a common neighbour),
collateral (fraction of base-fabric edges lost). Computed by the *same* reductions the
synchronous script uses, so sync and async rows are directly comparable.

### D. Sync reference

`shortcut_censorship.run_condition(base, portals, condition, steps, prune_prob)` on the
identical base+portals, `steps = sweeps`. Single source of truth for the banked numbers.

## Time-matching (making sync and async comparable)

- The sweep-equivalent unit is the **absolute continuous Poisson time** the runner already
  tracks, with every clock at **rate 1**. At rate 1 a node accrues Poisson(t) events of a
  given rule by absolute time `t`, so one time unit = one sweep-equivalent of *each* rule
  independently — this holds identically whether one clock or two share the timeline, which
  is why it avoids the `event_id / N` double-count when two rules run concurrently.
- **Equal clock rates, not equal success:** sync runs exactly one triadic sweep and one
  prune sweep per step (each edge gets one attempt of each), so the async triadic- and
  prune-clocks fire at the *same* opportunity rate (both 1). The success probabilities
  `rewire_prob` / `prune_prob` live *inside* the events (their defaults 0.05/0.05 match
  sync), never in the clock rate.
- The `min(a,b)` edge-ownership convention in `_event_prune` / `_event_triadic` gives each
  edge one attempt per sweep-equivalent, matching the synchronous per-edge rate. Both
  schedules are compared at a common budget of `sweeps` sweep-equivalents (= sync `steps`):
  portals still present at absolute time `sweeps` count as survived.
- **Asserted, not assumed:** the harness logs mean prune-events-per-node and
  triadic-events-per-node for both schedules and asserts they match within a small
  tolerance before Gate 2 interprets any difference — a weaving change from a rate
  mismatch would be a boring confound, not a schedule effect.

## Two gates, both pre-committed

### Gate 1 — anchor (P1); null: async `prune` matches sync `prune`

Distributional agreement (paired over seeds) on long survival, detour-2 survival, mean
removal (sweep-equiv), and collateral, at the step-2 Test-B bar (**z < 3**); plus
**|rank-corr(advantage, t)| < 0.2** under async. Sync reference: long 0.03 / detour-2 1.00
/ ⟨t⟩ 18 / corr −0.07 / collateral 0.

Gate 1 is also the **control**: passing it certifies that async's concurrency-granularity
difference *alone* (one-at-a-time vs simultaneous prunes, incl. the degree-floor coupling)
does not move the P1 observables — so any Gate-2 difference is attributable specifically to
the `triadic`/`prune` interleaving, not to async per se.

### Gate 2 — the race (P2); pre-registered both ways

Run `triadic`+`prune` under both schedules on identical base+portals.

- **Null (schedule-invariant):** async also weaves (woven > 0) and async long survival is
  materially above async prune-only → self-stabilization is genuine dynamics, banked more
  strongly under emergent time.
- **Interesting (the catch):** async weaving **collapses** (woven ≈ 0 and long survival ≈
  prune-only) → P2's protection was an artifact of synchronous `triadic`-then-`prune`
  lockstep, and a banked result falls under emergent time.

Both outcomes are reported and interpreted; the interpretation is fixed here, before any
async number is measured. Gate 2 is **descriptive** (it reports and interprets), not
pass/fail — the checkpoint's `--validate` exit status is governed by Gate 1 (the
instrument/control) plus the confound assertions, so an "interesting" P2 result is a
finding, never a test failure.

## Validation (`python async_censorship.py --validate`)

Seeded and reproducible, in the style of the other rungs:

1. **Engine equivalence** (in `async_engine --validate`): `run_sequential_multi` with one
   rule reproduces `run_sequential` bit-for-bit.
2. **Gate 1:** async prune-only vs sync prune-only agree in distribution (z < 3) on the P1
   observables, and |rank-corr| < 0.2 async.
3. **Confound assertions:** per-node prune-event and triadic-event counts match across
   schedules within tolerance; detour-2 survival reported under async (triadic churn
   control).
4. **Gate 2 (descriptive):** print the sync-vs-async `triadic`+`prune` comparison — woven
   count, long survival, detour-2 survival, collateral — and the pre-registered verdict
   (null vs interesting) for the observed numbers.

Run scale (tuned in implementation to keep `--validate` to a few minutes): N ≈ 1000–2000,
40 long + 20 detour-2 portals, ~120 sweeps, 3 seeds — matching the synchronous experiment
so the comparison is like-for-like. A `--quick` path (small N, few seeds) mirrors
`shortcut_censorship.py --quick` for smoke tests.

## Scope / non-goals

**In:** `run_sequential_multi` in `async_engine.py` (+ its equivalence check); the shared
base+portal setup; async prune-only and async triadic+prune conditions with in-run portal
tracking; the sync reference via `shortcut_censorship.run_condition`; the identical
observable reductions; the two-gate `--validate`.

**Out (deferred):**

- **The `ricci`-only condition** — `ricci` is not an async event, and it is redundant with
  `prune` for the P1 claim (same zero-overlap criterion). Porting it means implementing and
  validating a new `_event_ricci`; not worth the surface for this checkpoint.
- **Batched / large-N throughput** — sequential is sufficient at N=2000 sampled. The
  batched fast path (and mixed conflict-radius reasoning) is reserved for the barrier's
  N-sweep, step 5.
- **Any causal-DAG / causal-set measurement** — this is a state-graph checkpoint by design;
  the step-3-retired causal dimension stays out.
- **The bootstrapping-barrier checkpoint and causal-future-growth** — step 5.

## Explicit pre-commitments (stated before measuring)

1. The null is *"censorship is schedule-invariant"*: async reproduces sync P1
   (advantage-blind threshold) and P2 (self-stabilization). Agreement banks the result more
   strongly; disagreement means a banked result was a synchronous artifact — both are
   legitimate, publishable outcomes.
2. Gate 1 (P1) is the pass/fail instrument gate **and** the control that isolates Gate 2's
   interpretation. Gate 2 (P2) is descriptive with a pre-registered two-way reading.
3. Time is matched in **sweep-equivalents** (`event_id / N`); per-node event counts are
   asserted matched across schedules before any Gate-2 difference is interpreted.
4. Detour-2 survival is reported under async so a weaving change is never conflated with a
   triadic-churn change.
5. Tolerances (the z < 3 bar, |rank-corr| < 0.2, the event-count match) are frozen after
   the first run and not retuned to force a pass.
6. The synchronous side is `shortcut_censorship.run_condition` verbatim — the banked
   semantics are the reference, not a re-implementation that could drift.
