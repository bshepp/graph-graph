# Lorentzian upgrade: scoping spike

**Status: a scoping document, not a commitment.** Rung 1 of the de-toying ladder
(FINDINGS.md) proposes replacing synchronous sweeps with asynchronous event
updates and measuring the *causal DAG of update events* rather than the state
graph. The ladder claims it is the cheapest rung because "nearly all
instrumentation here ports over." This spike tests that claim before any compute
is committed, by answering four questions:

1. What exactly is an event, and what is the DAG?
2. What replaces `dimension.py`, and how does it get validated?
3. What does "extent" mean on a DAG, so the barrier question stays well-posed?
4. Can the fast backend be saved, or is the work capped at ~10K nodes?

**Headline: the claim is half right, and the cost is better than feared in one
place and worse in another.** The dynamics backend survives with a constant-factor
slowdown (§4) and the *fitting and gating scaffolding* of `dimension.py` ports
almost literally (§3). But the dimension *estimator* itself does not port, and the
replacement is only trustworthy with rung 2's calibration built alongside it (§2).
The honest unit of work is **rungs 1+2 together**, not rung 1 alone.

---

## 1. The event and the DAG

### Definitions

An **event** `e = (v, n)` is the *n*-th update of node `v`. Each event reads a
**read set** `R(e)` — under the locality invariant, `v` itself plus its neighbours
*at the moment of the update*. The **causal parents** of `e` are, for each
`u ∈ R(e)`, the latest event at `u` preceding `e`. Those parent→child edges are
the DAG's **links**; the causal order is their transitive closure.

Never materialise the closure. Interval cardinality `|I(p,q)|` is computed on
demand as `|future(p) ∩ past(q)|` from two directed BFS passes — the same
sample-a-few-hundred-pairs strategy `dimension.py` already uses.

### How events get ordered — and why the choice is not cosmetic

Three options, in increasing fidelity to "emergent time":

| scheme | how order arises | verdict |
|---|---|---|
| random sequential | a global RNG picks who goes next | a global clock in disguise — rejects the premise |
| **local Poisson clocks** | each node holds an independent exponential clock (Harris construction) | **recommended** |
| fully relational (Lamport) | an event fires when its dependencies are satisfied; concurrent events genuinely unordered | purest, but no natural rate parameter |

Poisson clocks are recommended for a reason that turns out to be load-bearing in
§2: **Myrheim–Meyer and its relatives assume a Poisson sprinkling into a
continuum.** Regular, lattice-like causal sets are famously *not* manifold-like
and can return meaningless dimensions. Deterministic round-robin updating would
build exactly such a lattice. Poisson update times *are* a sprinkling in the time
direction, which is what keeps the estimator applicable at all. The update-time
distribution is therefore an instrument-validity decision, not a modelling
detail.

### The triviality risk — and where the content actually is

If the graph is static and every node updates at the same rate reading all
neighbours, the event DAG is essentially `G × Z`: a spacetime lattice whose
causal dimension is just `d_H + 1`. Measuring it would be an expensive
re-derivation of the spatial dimension we already have.

**The content comes from the topology being dynamic.** These rules rewire the
graph, so an event at `v` can only influence `u` later if an edge existed *at that
time*. The causal structure therefore encodes the history of the topology — and
that is precisely where the bootstrapping barrier lives (rewiring crumples
extent). A secondary source of structure: if update rates vary with local state,
the "lapse" varies spatially, which is a nontrivial local time.

This gives a free calibration, exploited in §5: **on a static graph the answer
must be `d_H + 1`.** If it is not, the instrument is broken, not the physics.

---

## 2. What replaces `dimension.py`

Ball growth `|B(v,r)| ~ r^d` is an undirected-*metric* estimator. A DAG has no
such metric, so it does not port. Two candidate replacements:

### (a) Interval-cardinality scaling — the closest port

`|I(p,q)| ~ ℓ(p,q)^d`, where `ℓ` is the **longest chain** between `p` and `q` (the
causal-set analogue of proper time). This is the direct analogue of ball growth
with "geodesic radius" swapped for "causal height", and it reuses the existing
scaffolding almost verbatim: fit `log|I|` against `log ℓ`, apply the same
finite-size correction, the same R² and saturation gates, and the same
`defined_frac` honesty signal. Most of `dimension.py`'s *machinery* survives even
though its *metric* does not.

### (b) Myrheim–Meyer ordering fraction — the literature standard

Within an interval, compute the **ordering fraction** `r = R / C(N,2)`, where `R`
counts causally related pairs and `N = |I|`. For Poisson sprinklings into
*d*-dimensional Minkowski space, `r` is a known decreasing function of `d`;
inverting it gives the dimension.

> **Resolved 2026-07-19 (step 1).** This section originally declined to state
> the constant, on the grounds that a scoping doc with a wrong formula is worse
> than one naming the gap. It has since been reconstructed from the 2-chain
> abundance, pinned against both anchors below, and confirmed numerically to
> ~1% at d = 2, 3, 4:
>
>     r(d) = Γ(d+1) Γ(d/2) / ( 2 Γ(3d/2) )
>
> Measured against predicted: 0.4993/0.5000 (d=2), 0.2261/0.2286 (d=3),
> 0.1012/0.1000 (d=4). Implemented in `causal_sets.py`.

Two anchors, both derivable directly and both used as unit tests:

- **d = 1** → `r = 1`. A 1D causal set is a total order; every pair is related.
- **d = 2** → `r = 1/2`. In light-cone coordinates `(u,v)` an Alexandrov interval
  is a square, and two uniform points are related iff both coordinates agree in
  order: `P = 2 × (1/2)² = 1/2`. **Confirmed numerically** by sprinkling into the
  unit square (20 reps each): `r = 0.5014 ± 0.0288` at n=200, `0.4991 ± 0.0144`
  at n=800, `0.4999 ± 0.0047` at n=3000. This is step 1 of §6 in miniature, and
  it already works.

### Recommendation — and how step 1 revised it

The plan was: implement both, require agreement, in this project's established
cross-validation style (`dimension.py` against the `lattice` control,
`coherence.py` against a permutation null, `ctqw_time_average` against Krylov
propagation with a degeneracy-blind control).

> **Step 1 outcome (2026-07-19): estimator (a) failed and was replaced.**
> Interval scaling is systematically biased low — **1.92 / 2.79 / 3.53** against
> a true 2 / 3 / 4 — while reporting **R² > 0.99**. Fixing a regression-dilution
> error (the longest chain is the noisy variable, so cardinality belongs on the
> x-axis) improved it from 1.75/2.21/1.86 but did not rescue it. The residual
> bias is *not* finite-size: it is flat in N (3.70, 3.71, 3.59 at
> N = 1500/3000/4500). A large-`|I|` regime gate converges at d=2 but is
> non-monotonic and never reaches 3.0 at d=3, so it would need tuning per
> dimension — and an estimator whose gate must be tuned against the known answer
> is not an instrument, because on an event DAG there is no known answer to tune
> against.
>
> **This retracts the spike's claim that `dimension.py`'s scaffolding ports.**
> The machinery ports; the estimator built from it does not work. The log-log-fit
> port produces a confident wrong number, which is worse than no port at all.
>
> Replaced by **midpoint scaling** (Bombelli): find the element maximising
> `min(|I(p,r)|, |I(r,q)|)`; it halves proper time, so `N_half ≈ N/2^d`. This is
> genuinely independent of MM — a one-point volume extremum against a two-point
> pair count — which is what makes agreement evidence rather than tautology.
> Both now recover the truth and each other, so the gate passes:

| d | MM (primary) | midpoint (cross-check) | interval (demoted to diagnostic) |
|---|--------------|------------------------|----------------------------------|
| 2 | 2.000 | 2.010 ± 0.105 | 1.907 |
| 3 | 2.988 | 3.078 ± 0.242 | 2.745 |
| 4 | 3.976 | 4.151 ± 0.400 | 3.626 |

`N = 4000`, 3 reps; run with `python causal_sets.py --validate`.

**The two estimators get different gates, because they have different jobs.** MM
is the primary and is gated on recovering the truth (it does, to ~1%). Midpoint
is the cross-check — its job is to catch a gross error in MM — so it is gated on
*agreeing with MM*, and its own deviation is characterised rather than gated. It
carries a **documented +4% bias at d=4** (4.151 at N=4000, 4.160 at N=2500 —
stable in N, mechanism understood as the discreteness of an extremum taken over
few elements). Gating it at MM's tolerance would fail a working instrument over
a known systematic; loosening MM's tolerance to cover it would hide a real bias.

One tuning trap avoided along the way: raising the minimum interval size for the
midpoint estimator *looks* like it should buy accuracy and does the opposite —
at `|I| ≥ 200`, d=4, N=2500 so few intervals qualified that the estimate went to
4.47 on sampling noise, against 3.94 at `|I| ≥ 64` with a full sample. The gate
that matters is therefore the **sample count**, which is diagnosable without
knowing the true dimension; a cardinality threshold would have had to be tuned
per-dimension against the answer.

**This is why rung 1 cannot ship alone.** Rung 2 — Poisson sprinkling into flat
and one curved spacetime — *is* the known-answer calibration for both estimators.
Without it, rung 1 delivers a number with no way to tell whether it is right.

---

## 3. "Extent" on a DAG, and the barrier restated

The barrier is `extent ~ N^α` with extent = state-graph diameter, α ≈ 0.13 for
rewiring rules against 0.515 for a 2D lattice control. The DAG has no spatial
metric — its natural invariant is *time* (longest chain), not space. So the
observable does not port directly. Two resolutions, and both are worth having:

### (a) Keep the original observable — it never needed to change

The state graph still exists at every moment; asynchrony changes the *dynamics*,
not the observable. Measure diameter exactly as now, with "time" defined as
number of events elapsed (or mean updates per node). **The barrier question
survives unmodified**, which means the checkpoint is a genuine like-for-like
comparison rather than a redefinition — exactly what a checkpoint needs to be.

### (b) The causal restatement — the new thing the DAG buys

Measure **causal future growth**: `|J⁺(e, n)|` against DAG-depth `n`. This is the
light cone, measured directly.

- Polynomial growth → light cones exist → causal locality, manifold-like.
- Exponential growth / fast saturation → expander-like, no causal locality.

This restates the barrier as: **the model fails to build light-cone structure —
its causal futures are expander-like rather than manifold-like.** That is the same
obstruction seen through the causal structure, and it is arguably the more
fundamental statement of it. Mechanically it is a directed BFS with the existing
ball-growth fitting code behind it, so instrumentation reuse here is real.

---

## 4. Can the fast backend be saved?

**Yes, with a constant-factor slowdown — this is the biggest correction to the
ladder's cost estimate, in the favourable direction.**

The concern was that `FastGraph`'s vectorisation (`A @ active`, one-hot majority)
is inherently synchronous, while async updates are inherently sequential. That is
false: **asynchronous is not sequential when events are causally independent.**
Two events at non-adjacent nodes commute — any order gives the same result — so
they can be applied in one vectorised batch.

Concretely, per round: draw random priorities, and let `v` update iff its priority
exceeds all its neighbours'. Neighbour-max is a sparse segmented max
(`A.multiply(prio[None,:]).max(axis=1)`), so the batch selection is itself
vectorised. Any linear extension of a batch is a valid asynchronous ordering, so
this is exact, not an approximation.

**Verified, not assumed** (`grown`, seed 0, 5 draws each): the selected set is
provably independent at every size tested, and the batch fraction slightly beats
the `1/(deg+1)` estimate.

| N | cap | mean deg | batch fraction | `1/(deg+1)` | independent? |
|---|-----|----------|----------------|-------------|--------------|
| 2,000 | 6 | 4.00 | 0.228 | 0.200 | yes |
| 20,000 | 6 | 4.00 | 0.229 | 0.200 | yes |
| 20,000 | 8 | 4.00 | 0.240 | 0.200 | yes |

So a full sweep costs ~4–5 rounds — a **constant factor in degree, not an
asymptotic loss**. (Note that `grown` holds mean degree ≈ 4 regardless of cap;
the cap bounds the tail, not the mean.)

> **Implementation footgun, found while verifying this.** Build the adjacency with
> `nx.to_scipy_sparse_array(G, weight=None, ...)`. This project sets edge
> `weight=0.5`, and the *default* uses that attribute — silently yielding a
> weighted adjacency that halves every neighbour max. The first run of this check
> selected 62% of nodes in a non-independent set for exactly that reason. All
> seven existing sparse call-sites pass `weight=None`; the async engine must too.

**Fidelity caveat.** Random-independent-set batching is *a* valid async ordering,
not a uniformly sampled one; it biases toward simultaneous updates of distant
nodes. For Poisson-clock fidelity, discretise time finely enough that conflicts
are rare and defer the rare collisions. Required validation: batched and strictly
sequential runs must agree statistically at small `N`.

### The real cap is memory, and it is on measurement only

The DAG has `N × T` events with `~(k+1)` links each. At `N = 2×10⁵`, `T = 2000`
that is `4×10⁸` events — not storable. So the scaling story splits:

| what | cap | why |
|---|---|---|
| **dynamics** (async evolution, state-graph observables incl. the barrier) | ~`10⁵`, as today | batching preserves vectorisation |
| **causal measurement** (interval sampling, MM, light cones) | ~`10⁴` nodes × `10³` steps | `10⁷` events fits in int32 arrays |

This is materially better than "everything is capped at 10K". The barrier
checkpoint — the expensive one, needing an `N`-sweep to fit `α` — runs at full
scale. Only the causal observables are capped, and those are sampled anyway.

---

## 5. Checkpoint experiments, in order

**Start with censorship, not the barrier.** It ports without needing a new
definition (`prune` still acts on the state graph), so it tests emergent time
against a settled result with no confounding redefinition. The barrier needs
§3(a) fixed first and is better run once the async machinery is trusted.

Pre-committed nulls, in the project's established style — a null result is
acceptable and publishable, and stating it now prevents fitting a story later:

| experiment | pre-committed null | why it would still be worth knowing |
|---|---|---|
| static-graph async run | causal dimension = `d_H + 1` | it is the instrument calibration; failure indicts the estimator, not the model |
| shortcut censorship under async | threshold-type and advantage-blind, as synchronously | censorship is a property of `prune`'s locality, not of the update schedule |
| bootstrapping barrier under async | `α` unchanged (≈0.13) | the obstruction is geometric, not an artifact of synchronous updating |
| causal future growth | exponential / fast-saturating | the causal restatement of the same barrier |

The interesting outcome is any *disagreement* between a synchronous result and
its async counterpart — that would mean a result the project has banked was an
artifact of the global clock. That is the real risk this rung is buying down.

## 6. Staging and kill criteria

1. ~~**Sprinkling + estimators (rung 2 work, no model).**~~ **DONE 2026-07-19 —
   `causal_sets.py`, gate PASSES.** Poisson sprinkling into d=2,3,4 Alexandrov
   intervals; Myrheim-Meyer and midpoint scaling both recover the truth to
   within ~0.1 and agree with each other (table in §2). Anchors `r(1)=1` and
   `r(2)=1/2` hold exactly. Cost the plan did not anticipate: the originally
   proposed estimator failed calibration and had to be replaced, which is
   precisely what this stage exists to find out — and it means the first
   instrument now in hand is *not* the one the spike designed.
2. **Async engine + batching.** Poisson clocks, independent-set batching,
   equivalence test against strictly sequential at small `N`. *Kill criterion:
   batched and sequential disagree statistically.*
3. **Static-graph calibration.** Async on a fixed `grown` graph; require
   `d_causal = d_H + 1`.
4. **Censorship checkpoint.**
5. **Barrier checkpoint** (state-graph extent) **and causal future growth.**

Steps 1–3 are all instrument work and produce no physics. That is the correct
shape for this project, and it is what the cost estimate should reflect: the
first physics result arrives at step 4.

## 7. What this rung does not buy

The ladder's own standing requirement is *"a continuum-limit / universality story
— without it the model stays a toy regardless of ingredients."* That is rung 4's
job. Rung 1 buys causal structure and a connection to a literature that already
carries the model-to-nature argument — real value, and it makes the barrier and
censorship results statements about a model class rather than about one
hand-rolled sandbox. But it does not answer "does anything converge as `N` grows,
independently of the microrule."

Cost ordering points at rung 1; the actual toy-ness gate points at rung 4. Worth
deciding deliberately which is being bought, rather than letting the ordering
decide.
