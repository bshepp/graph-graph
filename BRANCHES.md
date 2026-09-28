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
| **Triangulated-base `prune` transition** | step-3 fork, 2026-06-27 ("bank the crossover honestly") | banking `prune` as a continuum knob | the un-taken alternative was a genuine-transition hunt on a 2D-triangulated substrate; never invalidated. **Note 2026-09-27:** `grown` is not such a substrate (it has no dimension); the triangular lattice is, and `sheet` may be | medium (`prune_dimension.py` variant) |
| **`d(p)` multi-bump fine structure — mechanism hunt (round 3)** | round 1 (protection-hierarchy onsets) unsupported 2026-08-12; round 2 (integer-r_c crossings, spec 4d4044c) KILLED 2026-08-17 — coverage failure (no s=1.5 crossing at 22/30 points incl. 3 of 5 features) + M2 R²=0.099 on the valid points; convergence-depth bands (candidate b) descriptively unsupported (smooth 7→10 rounds, no banding) | two candidates burned cheaply via pre-registration | **round 4 done 2026-09-28** (`dp_round4.py`): a single crossover is REFUTED (fits worse than a cubic), but two added crossovers leave residuals at seed noise (0.7-0.9 SE) where a sextic leaves 1.2-1.7. Remaining question is only what the two scales are (candidates: shortcuts within reach; backbone thinning) — a low-priority identification, no longer a mystery. **round 3 done 2026-09-27** (`dp_decomposition.py`): the waveform is carried entirely by ball counts at radius ≥ 4 (share 1.09; r ≤ 3 share −0.09) and is not a fit-weight amplification. Every local mechanism is out. **Round 4 candidate:** there is no mechanism — it is the ring → small-world crossover in `|B(10)|` seen through a polynomial detrend. Test on fresh seeds: fit a crossover form, ask whether the residual falls to seed noise; must explain the banked survival under a sextic detrend | low |
| **Causal ordering-fraction as a *relative* comparator** | step-3 aftermath, 2026-07-29 | retiring the absolute observable | the monotone family r_graph(D) is intact for async-vs-sync or rule-vs-rule *comparisons*; offered post-step-3, never picked up | medium |
| **Quasi-1D fragmentation scaling** | step-3 fork, 2026-06-27 | (same fork as triangulated base) | never invalidated; lowest-value survivor of that fork | low-medium |

## CONDITIONAL — gated on a stated outcome

| branch | gate | origin |
|---|---|---|
| **`ricci` under async** (`_event_ricci` + validation) | rises from parked only if any async-vs-sync discrepancy appears anywhere | step-4 scope fork 2026-08-03 |
| **Ladder rung 2 remainder: curved-spacetime sprinkling** | causal-set instruments regain a consumer (e.g. the relative comparator gets used) | LORENTZIAN_SPIKE |
| **Ladder rung 3: entanglement edges (stabilizer states, ER=EPR toys)** | owner prioritization | de-toying ladder 2026-07-16 |
| **Ladder rung 4: action principle + universality (Metropolis at T, FSS across microrules)** | owner prioritization; carries the continuum-limit requirement the whole ladder points at | de-toying ladder 2026-07-16 |

## OPEN (continued) — observations from stage 1 (2026-08-17)

| branch | origin | why it's alive | cost |
|---|---|---|---|
| **Throat-motif ↔ d(p) mechanism cross-pollination** | stage-1 verdict: threshold = first ~2.6-strand mutually-protecting motif | the d(p) fine-structure mechanism hunt (round 3) and the throat onset concern the same object — protection motifs in random shortcut ensembles — in different ensembles; the exact peeling machinery now exists to count motifs directly in the pruned-WS ensemble | low-medium |

## OPEN (continued) — raised by the 2026-09-26/27 audit and step 5

| branch | origin | why it's alive | cost |
|---|---|---|---|
| **Re-run the portal / censorship / throat program on a substrate that has a dimension** | audit: `grown` is tree-like, so every "portal into a geometry" result was measured on a fabric with no geometry | the dynamics results stand, but "advantage = distance at injection" means something different when distances are logarithmic. Substrates with a real dimension now exist: the triangular lattice, and `sheet` if its confirmatory verdict holds | medium (drivers take a topology; the substrate swap is the work) |
| **Does `sheet` at beta = 2 turn tree-like near 1.6 million nodes?** (plan step 2) | transition scan 2026-09-28: CROSSOVER SUPPORTED, thinly (exponential law N* = exp(2.68 + 5.82 beta); a transition at beta_c = 2.33 not excluded; consistency clause passed by 0.6%) | the exponential law makes a prediction that can fail: N*(2.0) = 1.6e6. Grow beta = 2 to ~4e6 on several seeds. Tree-like by then = crossover confirmed; still compact = transition below 2.33 back on the table. Also re-scan 1.0-1.7 with more seeds: N* was non-monotone at 4 seeds | medium-high (memory and hours per run; jaga-scale, or local overnight) |
| **`sheet` as a registered topology, and the rules run on it** | `sheet_growth.py` is a standalone module | wiring it into `create_initial_graph` lets preservation, portals, censorship and `prune` run on a substrate with a real dimension. Generation is slow (~10 min per 512000 nodes); whether it belongs in the core is an owner call | low-medium |
| **Schedule-invariance of the barrier exponent, at power** | step 5: difference +0.010 ± 0.032, CI wider than the ±0.05 band | integer diameters of 10-18 are too coarse. Needs a continuous extent observable (mean eccentricity or mean pair distance) or ~4x the seeds; a new pre-registration either way | low-medium |
| **Long-run collapse of extent under `triadic`** | step-5 rider: extent peaks near 300 sweeps then falls below its starting value (N=1000: 8.5 → 11.5 → 4.0) with the largest component intact | the banked alpha is a 200-step transient. What is the long-run state — a hub? does the peak time scale with N? Both schedules agree, so the sync driver suffices | low |
| **Async events for `geometrize` and `ricci`** | step-5 scope: only `triadic` exists as an async event | the three-rule barrier claim is tested under async for one rule | low-medium |
| **Re-run the banked `defined_frac` results under the window-stability gate** | gate added to `dimension.py` 2026-09-28 (owner-approved) | preservation, coherence, portal tolerance and the `prune` d(p) curve were all recorded under the old estimator and are now stale as numbers. Most will simply read "not window-stable"; worth one pass so the log and the code agree | low |

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
| Spectral-dimension flow on `grown` | **done 2026-09-27** (`spectral_flow.py`) — slow monotone rise 1.2 → 1.9, N-robust; predicted RUNAWAY refuted. `grown` has exponential volume but bottlenecked, tree-like walks: a `d_s`/`d_H` split of the tree kind, not a CDT-like flow |
| An honest dimension gate inside `dimension.py` | **done 2026-09-28** — gate 3 (window drift) in `local_dimension`, field verdict `window_stable` in `dimension_stats`. Known blind spot: exponential growth slower than base ~1.2 is invisible inside a radius-10 window |
| Step 5: barrier under async | **done 2026-09-27** — barrier SURVIVES (alpha_async 0.168, 95% CI [0.136, 0.200], bound 0.25); schedule invariance underpowered. Gates 0 and 1 failed as frozen (tolerances), recorded as such. Causal-future-growth stays deferred with the retired causal DAG |
| Grown-generator "persistent expander phase" | **resolved 2026-09-26** — it is growth extinction: the 19 failing seeds coincide and are 7-17 node graphs whose frontier died. Extinction probability 0.0082 at cap 6, decided within the first ~20 nodes |
| "`grown` has a tunable emergent dimension" and "`prune` is a dimension knob 1 → 2" | **audit 2026-09-26** — exponential ball growth; banked values are radius-10 readings. Pruned WS has a real dimension (1) only for p ≲ 0.1 |
| Critical-collapse stages 2–3 (universality; driven injection) + stage-1 full-dynamics-FSS contingency | stage-1 verdict (2026-08-17): no critical collapse — the throat threshold is a local-motif onset (onset core constant ~2.6 strands across capacities 134–1043; relative width RISES 1.47→1.66; frozen sharpness rule satisfied only vacuously via 1/capacity rescaling). Gate 1 (peeling≡dynamics) passed 40/40 exactly, so the contingency never triggered |
| P2 attenuation micro-effect | 40-seed paired run (2026-08-11): woven ratio 1.01, t=-0.08; long-survival t=-0.28 — fully schedule-invariant, the 12-seed t=1.77 hint was noise |
| Interval-scaling as primary causal estimator | step-1 result (biased low at R²>0.99) |
| Absolute causal-set dimension on the event DAG | step-3 key negative (not manifold-like; estimators disagree) |
| Rewiring-from-disorder at scale | structural argument (log vs polynomial extent — scale *worsens* it) |
| cap→d "sharpens to integers at scale?" | `cap_dimension_scaling` result: plateaus at non-integer values (continuum knob) |
| Event-counter time for merged clocks / single-clock rule coin-flip / `record_dag` flag / analytic causal-relation shortcut | design-level: double-counts / inferior construction / module coupling / can't serve dynamic topology |
| Ising exponents as `prune` defaults | methodology guard (extract, don't assume) — never was a branch |
