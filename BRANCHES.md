# BRANCHES — the living registry of research branches

Every design fork produces un-chosen options. This file keeps them as **dormant
branches with revival conditions** instead of dead leaves scattered across spec
out-lists, memory notes, and FINDINGS asides. Practice, adopted 2026-08-11:

- When a fork is decided, the un-chosen options land here with: origin, what was
  chosen instead, status, revival condition, rough cost.
- Statuses: **OPEN** (ready to start now), **CONDITIONAL** (gated on a stated
  outcome), **PARKED** (deliberately shelved, no gate), **ENGINEERING** (enabler,
  not science), **CLOSED** (superseded or killed by a result — kept for
  auditability, with the reason).
- A branch leaves this file only by being promoted to work or explicitly CLOSED
  with a reason. Rot is the failure mode this file exists to prevent.

## OPEN — revival condition already met (or none needed)

| branch | origin | chosen instead | why it's alive | cost |
|---|---|---|---|---|
| **Spectral-dimension flow `d_s(scale)` on `grown`** | FINDINGS "Scaling directions" #1 (2026-05-30), gated behind the cap→d check | cheaper items first | **its gate was satisfied when `cap_dimension_scaling` landed (2026-05-30) and nobody noticed** — the flagship open question: does a minimal local growth rule reproduce a CDT-like `d_s` flow, and does `d_s` split from `d_H`? Pre-committed null (`d_s = d_H`, no flow) already written | high (large-N walks; `traverse.py`/`braket_walks.py` are the seed) |
| **Triangulated-base `prune` transition** | step-3 fork, 2026-06-27 ("bank the crossover honestly") | banking `prune` as a continuum knob | the un-taken alternative was a genuine-transition hunt on a 2D-triangulated substrate; never invalidated — and now doubly relevant as a stage-2 substrate family for the critical-collapse program | medium (`prune_dimension.py` variant) |
| **`d(p)` multi-bump fine structure — mechanism hunt (round 3)** | round 1 (protection-hierarchy onsets) unsupported 2026-08-12; round 2 (integer-r_c crossings, spec 4d4044c) KILLED 2026-08-17 — coverage failure (no s=1.5 crossing at 22/30 points incl. 3 of 5 features) + M2 R²=0.099 on the valid points; convergence-depth bands (candidate b) descriptively unsupported (smooth 7→10 rounds, no banding) | two candidates burned cheaply via pre-registration | a real, deterministic, density-intrinsic multi-bump structure with NO surviving mechanism hypothesis after two rounds. Remaining named candidate: (c) WS-construction shortcut-overlap statistics — plus fresh formulation needed. Formulate → pre-register → test | low-medium |
| **Causal ordering-fraction as a *relative* comparator** | step-3 aftermath, 2026-07-29 | retiring the absolute observable | the monotone family r_graph(D) is intact for async-vs-sync or rule-vs-rule *comparisons*; offered post-step-3, never picked up | medium |
| **Quasi-1D fragmentation scaling** | step-3 fork, 2026-06-27 | (same fork as triangulated base) | never invalidated; lowest-value survivor of that fork | low-medium |

## CONDITIONAL — gated on a stated outcome

| branch | gate | origin |
|---|---|---|
| **Step 5: barrier under async (+ causal-future-growth as relative comparator)** | queued behind the collapse program by owner ordering; readiness note exists (`docs/superpowers/specs/2026-08-03-step5-barrier-async-readiness.md`) | ladder |
| **`ricci` under async** (`_event_ricci` + validation) | rises from parked only if any async-vs-sync discrepancy appears anywhere | step-4 scope fork 2026-08-03 |
| **Ladder rung 2 remainder: curved-spacetime sprinkling** | causal-set instruments regain a consumer (e.g. the relative comparator gets used) | LORENTZIAN_SPIKE |
| **Ladder rung 3: entanglement edges (stabilizer states, ER=EPR toys)** | owner prioritization | de-toying ladder 2026-07-16 |
| **Ladder rung 4: action principle + universality (Metropolis at T, FSS across microrules)** | owner prioritization; carries the continuum-limit requirement the whole ladder points at | de-toying ladder 2026-07-16 |

## OPEN (continued) — observations from stage 1 (2026-08-17)

| branch | origin | why it's alive | cost |
|---|---|---|---|
| **Grown-generator persistent expander phase** | stage-1 substrate regeneration: 19/2000 draws fail geometry at ALL THREE nested-seed geometries (N=2000/5000/10000); first seen seed 111 N=600 (diameter 4) | if the same growth seeds fail at every N, the compact phase is decided early and persists — generator bistability is not a small-N artifact (~1% of seeds). Next: confirm the 19 failing seeds coincide, then phase statistics vs N and what the early-growth discriminator is | low |
| **Throat-motif ↔ d(p) mechanism cross-pollination** | stage-1 verdict: threshold = first ~2.6-strand mutually-protecting motif | the d(p) fine-structure mechanism hunt (round 3) and the throat onset concern the same object — protection motifs in random shortcut ensembles — in different ensembles; the exact peeling machinery now exists to count motifs directly in the pruned-WS ensemble | low-medium |

## PARKED — deliberate, no gate

| branch | origin | note |
|---|---|---|
| **Density-healing rule design** | stage-1 brainstorm 2026-08-11 (dense-ball seed is censor-blind) | a strictly-local density-regulating rule is the missing "restoring force" for lump-type seeds — value beyond the collapse program (it is the closest thing to gravity in the rule set); must honor the locality invariant |
| **Large-cap (1/cap) analytic limit** | paper-directions #4, 2026-08-11 | theory note; expander limit is the tractable end — inverse of the paper's large-D trick |
| **Horizon-free quantum hitting-time analogue** | walkers 2026-07-18 | needs a measurement model first |

## ENGINEERING — enablers, not science

| item | blocks | origin |
|---|---|---|
| **Vectorised batch *application*** (still a Python loop) | async at ≥10⁵ nodes — step 5's AWS-scale fork | step-2 known gap |
| **`rewire` under async** (collision detection + deferral) | any async experiment involving `rewire` | async_engine design |
| **`adv_corr` guard backport to `shortcut_censorship.py`** | nothing (latent-bug hygiene; the banked numbers are unaffected) | step-4 review 2026-08-03 |
| **`throat_criticality.py --fss` width label prints the unguarded diagnostic value** (parked Important from final review; harmless while ensembles are 100%-finite — guarded and unguarded values identical — but a <90%-finite geometry would print a plausible width where nan is owed) | nothing for stage-1 results (verdict computed from persisted CSVs) | stage-1 final review 2026-08-17 |

## CLOSED — kept for auditability

| branch | killed by |
|---|---|
| Critical-collapse stages 2–3 (universality; driven injection) + stage-1 full-dynamics-FSS contingency | stage-1 verdict (2026-08-17): no critical collapse — the throat threshold is a local-motif onset (onset core constant ~2.6 strands across capacities 134–1043; relative width RISES 1.47→1.66; frozen sharpness rule satisfied only vacuously via 1/capacity rescaling). Gate 1 (peeling≡dynamics) passed 40/40 exactly, so the contingency never triggered |
| P2 attenuation micro-effect | 40-seed paired run (2026-08-11): woven ratio 1.01, t=-0.08; long-survival t=-0.28 — fully schedule-invariant, the 12-seed t=1.77 hint was noise |
| Interval-scaling as primary causal estimator | step-1 result (biased low at R²>0.99) |
| Absolute causal-set dimension on the event DAG | step-3 key negative (not manifold-like; estimators disagree) |
| Rewiring-from-disorder at scale | structural argument (log vs polynomial extent — scale *worsens* it) |
| cap→d "sharpens to integers at scale?" | `cap_dimension_scaling` result: plateaus at non-integer values (continuum knob) |
| Event-counter time for merged clocks / single-clock rule coin-flip / `record_dag` flag / analytic causal-relation shortcut | design-level: double-counts / inferior construction / module coupling / can't serve dynamic topology |
| Ising exponents as `prune` defaults | methodology guard (extract, don't assume) — never was a branch |
