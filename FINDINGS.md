# Findings: Emergent Dimension from Local Graph Rules

> **Read first (2026-09-27).** An audit found that the `grown` generator and pruned
> small-world graphs above p ~ 0.2 have **no dimension**: their ball growth is
> exponential, and the `d_eff` values quoted for them below (2.2 / 3.0 / 3.6, and the
> pruned 1 → 2 curve) are readings of one fit window. Passages that depend on this are
> marked *[superseded]* and left in place. See "Audit of 2026-09-26/27".

A running record of what the emergence experiments have actually shown.
Theory and framing live in [DIMENSIONAL_COHERENCE.md](DIMENSIONAL_COHERENCE.md);
this file is the empirical log. All results are reproducible with the seeds
shown via `dimension.py`, `track_dimension.py`, and the rules in `rules.py`.

## The instrument

Before measuring emergence we had to trust the ruler. The local effective
dimension estimator (`dimension.py`) was rebuilt to:

- use a **finite-size-corrected** log-log fit (`log|B| = d·log r + c + a/r`),
  which recovers known dimensions to within ~0.05 (a plain fit reads ~0.4
  low and would mis-bin 3D as 2D);
- **gate on regime existence** -- it returns *undefined* (nan) unless there
  are enough unsaturated radii (ball kept under ~10% of the graph) and the
  power-law fit is clean. Small-world / expander graphs have no polynomial
  ball-growth regime, so their dimension is genuinely undefined, not a
  fabricated number.

Validated against graphs of known dimension: `python dimension.py --validate`
(1D/2D/3D recovery, backend agreement, and undefined-on-expander all pass).

The key derived signal is **`defined_frac`**: the fraction of nodes that have
a well-defined effective dimension. On a geometric graph it is ~1; on an
expander it is ~0. Its change over time (via `track_dimension.py`) is the
preservation / emergence signal.

## What changes dimension, and what doesn't

| Rule | Mechanism (local only) | Result | Why |
|------|------------------------|--------|-----|
| `majority`, `activation`, `reinforcement` | update state / weights, not topology | dimension invariant | topology unchanged |
| `rewire` | rewire edges to **random** targets | **destroys** structure (lattice d≈2 → undefined in ~30 steps) | adds long-range shortcuts → collapses diameter |
| `prune` | remove low-overlap "shortcut" edges | small-world → **d≈1** (`defined_frac` 0→100%) | strips shortcuts → diameter grows → latent ring revealed |
| `triadic` | rewire edges to friends-of-friends | clustering 0.002 → 0.625 but **dimension stays undefined** | clustering rises, diameter stays short |
| `grown` (generator) | degree-capped frontier growth | *[superseded]* ~~tunable emergent d~~ — **no dimension**; radius-10 readings 2.2 / 3.0 / 3.6 of exponential growth | frontier stays ∝ N; diameter ~ log N |

> The cap→d numbers above are the **scale-converged** values (`N` up to 2e5;
> see "cap→d scaling" below). The earlier single-`N` estimates (6→2.1, 7→2.7,
> 8→2.9) were biased *low* by ball saturation at small `N` -- a finite-size
> artifact that the scaling check corrected.

**Unifying insight:** the active ingredient for emergent dimension is
**extent (large diameter)**, not local density. `prune` and `triadic` succeed
/ fail for the *same* reason -- one grows the diameter, the other doesn't.
`grown` makes it cleanest: bound the degree, force growth to the frontier,
and dimension falls out, tunable by one local scalar.

### A spectrum of "emergence"

- `prune` is **weak emergence**: d≈1 was *latent* in the Watts-Strogatz
  construction (ring + shortcuts); pruning revealed it.
- `triadic` is an **honest negative**: proves clustering ≠ geometry.
- *[superseded 2026-09-26: `grown` has no dimension.]*
  `grown` is **strong emergence**: dimension that was never latent (grown
  from a triangle), from a simple local rule, tunable by the degree cap.
  Caveat: the cap *selects* d, so it is "emergence with a knob," not a
  spontaneously preferred dimension -- and the knob is a **continuum**, not an
  integer quantizer: at scale cap 8 plateaus at ~3.6, not a clean 4 (see
  cap→d scaling). The dimension is stable and tunable, but not integer-valued.

## The attractor question: is there a *preferred* dimension?

The strongest version of the hypothesis: a local rule whose emergent
dimension is reached from **any** initial condition (selected by the
dynamics, not inherited or set by a knob).

Tested with `geometrize` -- homeostatic local rewiring toward a target
degree (shed the most shortcut-like edge when over-connected, add a triadic
edge when under-connected): the two ingredients that make dimension, run as
a feedback loop.

**Result: no global attractor. The dynamics are bistable.**

| Initial graph | Under `geometrize(target=6)` |
|---------------|------------------------------|
| 2D lattice (d≈2) | **stable fixed point** -- stays d≈2, stays connected (recovers after a dip) |
| random, scale-free (undefined) | **stuck at `defined_frac` ≈ 0** -- never geometrizes |
| small-world | geometrizes (has a latent ring) but d **drifts**, doesn't settle |
| `grown` (cap 10) | **fragments** (largest component 100% → ~13%) |

`target_degree` does not cleanly tune the stable dimension either (on a
lattice: 4→2.0, 6→2.2, 8→ mostly destroyed). So `geometrize` *preserves*
geometry within its basin but neither nucleates nor tunes it.

### The bootstrapping barrier

The reason random / scale-free graphs cannot be geometrized is sharp:

> "Local" attachment (friend-of-friend) is only meaningful once locality
> already exists. In a zero-clustering expander a node's 2-hop neighborhood
> is a *random* part of the graph, so "rewire locally" = "rewire randomly."
> There is no seed of geometry to amplify.

This is why `prune` worked on small-world (latent ring = seed) and `grown`
worked (built geometrically, never passing through an expander), but
`geometrize` cannot geometrize a random graph. **The expander/dimensionless
state is itself a stable phase that local rewiring cannot escape.**

### Curvature flow hits the same barrier

The strongest principled candidate for nucleating geometry is a discrete
**Forman-Ricci curvature flow** (`ricci`): rewire negative-curvature
shortcuts toward triangle-closing (positive-curvature) positions, so
curvature only ever increases. Run on a random graph:

| step | defined | clustering | mean Forman curvature | eccentricity |
|------|---------|------------|------------------------|--------------|
| 0    | 0%      | 0.003      | -9.8                   | 6            |
| 80   | 0%      | 0.42       | -6.7                   | 7            |
| 480  | 0%      | 0.43       | -6.6                   | 8            |

It raises clustering and curvature but **plateaus at a clumped, negatively
curved, short-diameter fixed point** -- the eccentricity never grows, so
dimension never appears. The flat eccentricity is the key: local rewiring
builds clustering by *crumpling*, not by *unfolding* into extent.

### Interpretation

Three distinct local rewiring mechanisms -- `triadic` (clustering),
`geometrize` (degree homeostasis), and `ricci` (curvature flow) -- **all**
fail to nucleate geometry from a random graph, and all reach the same
clumped small-world fixed point:

> **Fixed-N local rewiring cannot grow *extent* (diameter) from an expander.**
> It can create local triangles, but those crumple into the existing
> short-diameter structure instead of unfolding into an extended manifold.
> Growing the diameter would require removing shortcuts faster than the graph
> re-localizes, which either fragments it or stalls.

So, under fixed-N local rules: **dimension is bistable, not attracting.
Geometry must be *seeded* -- grown outward (the `grown` generator, which
never passes through an expander) or already latent (a small-world ring) --
it does not spontaneously condense from maximal disorder.** This maps onto
the "dimensionally incoherent" phase in DIMENSIONAL_COHERENCE.md (the
dark-matter analog): a stable, non-geometric phase that local dynamics
cannot escape.

**Scope -- what is and isn't claimed.** The defensible statement is narrower
and therefore stronger than "mechanism-independent": **fixed-N local rewiring
cannot nucleate extent.** All three rules tested share more than locality --
they conserve node count and operate by local edge moves, which is *exactly*
the regime where the diameter argument bites. The obstruction is pinned to a
conserved quantity (roughly, extent): on a fixed node set, local moves can
redistribute edges but cannot manufacture the long geodesics a manifold needs.
This is why `grown` -- which is *not* fixed-N; it adds nodes at a frontier --
escapes. Two honesty notes:

1. The three-rule agreement is suggestive induction; the **structural
   diameter-growth argument is what actually carries the result** (see Scaling
   directions: expander diameter `~log N` vs manifold `~N^(1/d)`). The next
   move that adds real weight is therefore *not* a fourth rule (Ollivier-Ricci
   is correctly predicted to hit the same wall and would add little) but
   **measuring the obstruction**: extent-growth-rate vs N across the three
   rules, showing the gap sharpens with N as the argument predicts. The flat
   eccentricity column in the `ricci` table (6→7→8) is the best single piece
   of evidence and is currently one N -- that is the thing to scale.
2. Not a formal proof. The clean way to *get* a chosen dimension remains the
   `grown` generator (build it geometrically; the degree cap tunes d).

## Preservation: is emergent geometry stable under the local rules?

> *[superseded 2026-09-29]* Measured on `grown`, which has no dimension; the
> `defined_frac` numbers below are stale under the window-stability gate. The same
> experiment on the triangular lattice and on `sheet` gives the same qualitative
> pattern -- see "Preservation of a real dimension under the rules".

The barrier says local rewiring cannot *build* extent from disorder. The dual
question is whether the extent we *do* build (the `grown` generator) *survives*
the same local dynamics, or erodes. Tested with `track_dimension.py` on a
pristine cap-6 `grown` graph (`N = 1e4`, `d_eff ≈ 2.2`, `defined_frac ≈ 1.0`
at `t=0`), self-controlled against that `t=0` baseline, 200 steps, seeds 0/1/2:

| rule (200 steps) | defined_frac t=0 → end | diameter 42 → | lcc_frac → | verdict |
|------------------|------------------------|---------------|-----------|---------|
| `prune`               | 99.6% → **99.3%** | **42** (unchanged) | 1.00 | **preserved** |
| `majority` (state-only) | 99.6% → **99.4%** | 42 (unchanged)   | 1.00 | preserved (trivial) |
| `triadic`             | 99.6% → **~46%**  | **22**             | **0.48** | **eroded** (crumple + fragment) |
| `rewire`              | 99.6% → **~2%**   | **14**             | 0.95 | **eroded** (diameter collapse) |

(End `defined_frac` are seed means; ranges over seeds 0/1/2: prune 99.2-99.4,
triadic 41-55, rewire 1.6-2.2. Diameter / `lcc_frac` are a double-sweep estimate
at seed 0, same estimator as `barrier_scaling.py`.)

The dimension verdict tracks **extent**, exactly as the barrier predicts on the
build side:

- **`rewire` destroys it fast.** Random shortcuts collapse the diameter (42 →
  14) while the graph stays ~95% connected -- so this is *diameter collapse, not
  fragmentation*. Balls saturate and `d_eff` goes undefined (the few survivors
  read a spurious `d_eff` 6-8, the saturation artifact). Same failure mode as
  lattice-under-`rewire`.
- **`triadic` erodes it slowly by *crumpling*:** it halves the diameter (42 →
  22) and sheds nodes (lcc 1.0 → 0.48), and the surviving component's median
  `d_eff` drifts toward 0. This is the **same crumpling signature** `triadic`
  shows when it *fails to build* geometry from a random graph -- the rule
  crumples whether it starts from disorder or from geometry.
- **`prune` preserves it exactly** (diameter 42 → 42, fully connected). `prune`
  removes low-overlap *shortcut* edges, and `grown` -- built entirely from
  triangle closures -- has essentially none, so the rule is a near-no-op. A
  satisfying consistency check: the same rule that *reveals* latent geometry in
  small-world (by stripping shortcuts) is inert on a graph with no shortcuts to
  strip.
- **State-only rules** (`majority`; `activation` / `reinforcement` by
  construction) leave the topology untouched, so dimension is invariant. The
  flat `majority` trajectory also confirms the estimator itself does not drift.

**Reading.** Emergent geometry is **not** trivially fragile: it is a stable
fixed point under the structure-respecting rules (`prune`, all state rules).
What destroys it is precisely what the extent argument flags -- injecting
long-range shortcuts (`rewire`) or over-densifying locally until the structure
crumples (`triadic`). The picture is consistent from both sides: **extent is the
load-bearing quantity** -- hard to build (the barrier), and erodable only by the
moves that attack extent directly. This is the `grown` analog of the lattice
being a stable fixed point under `geometrize` (bistability table above), and of
`geometrize` *fragmenting* `grown` there. Reproduce: `python track_dimension.py
--topology grown --nodes 10000 --rules <rule> --steps 200 --seed 0`.

## Spatial coherence: the dimension field forms contiguous phases, not noise

> *[superseded 2026-10-13]* Measured on `grown` with the pre-gate estimator; under the
> window-stability gate `grown` reads "field fragmented (N/A)". The lattice row stands.
> See "Round of 2026-10-13, item 3".

A per-node `d_eff` could be spatially *coherent* (graph-neighbouring nodes share
a dimension, so the field forms contiguous phases) or just per-node estimation
*noise*. `coherence.py` settles it with **Moran's I** of the `d_eff` field under
graph-adjacency weights (I > 0 = neighbours similar; I ≈ E[I] = -1/(n-1) = no
structure; I < 0 = checkerboard), significance from a permutation null. The
statistic is validated on known fields (`coherence.py --validate`: smooth
gradient → I = +0.97, checkerboard → -1.00, random field → ≈0, z ≈ 0).

Result (`N = 5000`, 3 seeds, 999 permutations):

| topology | defined_frac | edge_frac | d_eff | Moran's I | z | verdict |
|----------|--------------|-----------|-------|-----------|---|---------|
| `grown` (cap 6)     | 0.99 | 0.99 | 2.21 ± 0.53 | **0.881** | 86.6 | **coherent** |
| `lattice` (control) | 1.00 | 1.00 | 1.88 ± 0.13 | **0.958** | 93.3 | coherent |
| `small_world`       | 0.12 | 0.07 | 4.69 ± 0.38 | (0.910)   | 29.9 | field fragmented -- N/A |

On `grown` the field is defined almost everywhere (`defined_frac ≈ 1`) and the
defined nodes form **one connected fabric** (`edge_frac ≈ 1`: a single component
of ~4919/4963 nodes), so Moran's I ≈ 0.88 (z ≈ 87, p = 0.001) is a genuine
*whole-field* measurement: **emergent dimension is spatially smooth -- it forms
contiguous geometric phases, not scattered per-node noise.** The `lattice`
positive control reads the same (I ≈ 0.96).

**Honesty caveat on `small_world`.** Its raw I (0.91) *looks* coherent but is
**not** a whole-field statement: only 12% of nodes have a defined `d_eff`, and
those scatter into ~120 disconnected fragments (`edge_frac` 0.07) with an
artifactual `d_eff ≈ 4.6` (ball saturation at the regime gate's minimum radius).
Moran's I over a sparse, fragmented subset measures coherence *within tiny
patches*, not of the field -- so `coherence.py` refuses the "coherent" verdict
whenever `defined_frac` or `edge_frac` is low. Coherence is only meaningful where
the field is whole.

**Cross-check against preservation (an independent confirmation).** Coherence is
a different instrument from `defined_frac` (spatial autocorrelation vs ball-growth
resolvability), yet it tells the *same* story under the rules -- coherence tracks
extent:

| `grown` after 200 steps | defined_frac | Moran's I | reading |
|-------------------------|--------------|-----------|---------|
| `prune`   | 0.99 → 0.99 | 0.867 → 0.867 | coherence preserved |
| `triadic` | 0.99 → 0.59 | 0.867 → 0.593 | coherence degraded |
| `rewire`  | 0.99 → 0.00 | 0.867 → (gone) | coherence destroyed |

So two independent statistics agree: the rules that preserve extent preserve the
coherent dimension field, and the rules that erode extent erode its coherence.
Reproduce: `python coherence.py --validate` then `python coherence.py`.

## Open threads

- **Quantify the bootstrapping barrier (flagship negative):** *done (step 2)* --
  the obstruction now has a measured exponent. See "Quantified barrier" below.
- **Curvature flow** vs the bootstrapping barrier: tested (`ricci`) -- hits
  the same barrier (above). Ollivier-Ricci (optimal-transport) is the one
  untested variant but is expected to behave the same and would add little.
- **Spatial coherence:** *done* -- Moran's I confirms `grown`'s `d_eff` field is
  spatially coherent (I ≈ 0.88, z ≈ 87) where the field is whole, and the
  coherence tracks extent under the rules. See "Spatial coherence" above.
- **Robustness / scale:** *done (step 1), then superseded (2026-09-26)* -- the
  fixed-radius reading plateaus in N, but it is not a dimension.
- **Preservation of emergent structure:** *done* -- `grown`'s dimension is a
  stable fixed point under the structure-respecting rules (`prune`, state-only)
  and eroded only by extent-attacking moves (`rewire`, `triadic`). See
  "Preservation" above.
- **Portal experiments:** *done (2026-07)* -- tolerance / censorship / walkers;
  see "Portal experiments" below.
- **Laplacian-generator CTQW cross-check:** *done (2026-07-18)* -- the quantum
  portal gain is **not** a degree artifact of H = adjacency; it survives (and
  grows) under H = Laplacian. But the headline magnitude was retracted: the gain
  depends on the arbitrary CTQW time horizon, and the original run's portal
  placement was RNG-coupled to the solver. See "3b. Laplacian cross-check".
- **Horizon-free transport observable:** *done (2026-07-18)* -- infinite-time
  average `Pbar` (exact, degeneracy-grouped, `--validate`d against Krylov
  propagation) vs the matched classical stationary occupancy. The portal gain
  survives at 48x (adjacency) / 131x (Laplacian) and sharpens into a qualitative
  result: a portal is a **kinetic** device classically (changes arrival speed,
  provably not long-run occupancy) and a **structural** one quantum-mechanically.
  See "3c. Horizon-free observable". Open: a horizon-free quantum analogue of
  hitting time (needs a measurement model).

> **On "emergence may require 100K+ nodes"** (DIMENSIONAL_COHERENCE.md, Phase
> 5): that is a *hope by analogy to thermodynamics, not a derived crossover
> scale.* No calculation predicts where (or whether) any of these transitions
> sharpens. The cap→d and Ising-pipeline runs on local hardware are exactly
> what would let us *extrapolate* a real crossover estimate -- and that
> extrapolation is the gate to clear before reserving any large compute.

## De-toying ladder: upgrade paths out of the toy model class (action items)

The project's measurements are internally rigorous but externally capped as a
*toy*: time is a global synchronous for-loop, the geometry is undirected
(Riemannian-flavored, no causal structure), the rules are chosen rather than
derived, and edges are classical. Each gap has a concrete upgrade, ordered by
cost. None of them makes the model "about nature" -- they connect its
measurements to established quantum-gravity programs (causal sets, quantum
graphity, tensor networks) whose model-to-nature arguments the literature
already carries. Decided with the owner 2026-07-16; unscheduled.

1. **Lorentzian upgrade (causal event DAG)** -- **scoped 2026-07-19, see
   [LORENTZIAN_SPIKE.md](LORENTZIAN_SPIKE.md); steps 1-4 built and passing
   (`causal_sets.py`, `async_engine.py`, `causal_dag.py`, `async_censorship.py`
   -- all `--validate`).
   Step 3 (2026-07-29) is a KEY NEGATIVE: the static-graph causal calibration
   fails -- the async event DAG is not manifold-like and `d_causal != d_H+1`
   (see "Causal calibration (step 3)" below). This retires the causal-set
   dimension as an absolute observable but leaves the state-graph checkpoints
   (censorship, barrier) intact. Step 4 (2026-08-03) is the first physics
   checkpoint and it **passes**: shortcut censorship re-run under async updates
   reproduces P1 (threshold, advantage-blind) as a schedule-invariant, and P2
   (self-stabilization) survives emergent time essentially unchanged (40-seed
   paired: no attenuation -- the preliminary "~2x" was small-sample
   noise) (`async_censorship.py`; see "censorship under async (step 4)"
   below).** Step 2 revised
   the cost model: the conflict radius is *rule-dependent* (state rules 1,
   `prune` 2, `triadic`/`ricci`/`geometrize` 3, because triadic closure writes
   at distance 2), and naive independent-set batching silently under-samples
   high-degree nodes -- a node wins its priority contest with probability
   1/(c+1) -- which shifted steady-state activation 11% (z=5.9) until corrected
   by degree thinning. Cost is still a constant factor flat in N, but 7/24/58
   rounds per sweep by radius rather than the ~4-5 originally claimed. The spike's verdict: "nearly all
   instrumentation ports over" is half right. The fast backend *survives*
   (asynchronous != sequential; causally independent events batch into vectorised
   independent sets -- verified; 7 / 24 / 58 rounds per sweep by radius, as above) and `dimension.py`'s
   fitting/gating scaffolding ports to causal-future growth. But the dimension
   *estimator* does not port, and its replacement is only trustworthy with rung 2's
   calibration built alongside -- so **the honest unit of work is rungs 1+2
   together**, with the first physics result arriving only after three stages of
   instrument work. Causal measurement is capped near 1e4 nodes by DAG size; the
   dynamics (and so the barrier checkpoint) still run at full scale.
   Replace synchronous sweeps with asynchronous event-based updates and measure
   the **causal DAG of update events** instead of the state graph: that object
   has light cones by construction, and causal-set dimension estimators
   (Myrheim-Meyer) exist for it. Nearly all instrumentation here ports over.
   **Checkpoint experiment: do the bootstrapping barrier and the shortcut
   censorship survive when time is emergent?** If yes, those results become
   statements about a model class the QG literature owns -- the point where the
   toy stops being only a toy.
2. **Causal-set calibration anchor.** -- **flat-space stage DONE 2026-07-19
   (`causal_sets.py --validate`, gate passes).** Poisson sprinkling into d=2,3,4
   Alexandrov intervals; two independent estimators recover the truth and agree:
   Myrheim-Meyer (2.000 / 2.988 / 3.976) and midpoint scaling (2.010 / 3.078 /
   4.151, carrying a documented +4% bias at d=4 that is stable in N).
   The ordering-fraction constant `r(d) = G(d+1)G(d/2) / (2 G(3d/2))`,
   which LORENTZIAN_SPIKE.md had declined to state from memory, is reconstructed
   and confirmed to ~1%. **Negative result worth keeping:** the spike's proposed
   primary estimator -- interval scaling `|I| ~ l^d`, the direct port of
   `dimension.py`'s ball growth -- *failed* calibration, reading 1.92 / 2.79 /
   3.53 at R^2 > 0.99, with a bias that is flat in N and a regime gate that
   would have to be tuned per-dimension. It is demoted to a diagnostic. So the
   claim that `dimension.py`'s scaffolding ports is retracted: the machinery
   ports, the estimator built from it returns a confident wrong number. Still
   open on this rung: the curved-spacetime sprinkling.
3. **Change the nature of connection: entanglement edges.** Nodes carry qubits;
   geometry is read from mutual information (it-from-qubit / quantum graphity).
   The tractable route is **stabilizer/graph states under local Clifford
   dynamics** -- efficiently simulable at thousands of qubits, with the
   entanglement structure literally being the graph. The portal experiments
   restate as ER=EPR toys: a Bell pair between far regions *is* the portal;
   rerun tolerance / censorship / walkers in that representation.
4. **Action principle + universality.** Replace hand-chosen rules with a graph
   action (Forman or Ollivier curvature functional; `ricci` is already a crude
   gradient step of one) evolved by Metropolis at temperature T, then test
   **universality across microrules** with the validated FSS machinery --
   sameness across rules is what converts "this rule does X" into "this class
   does X".

Standing requirement across all rungs: a **continuum-limit / universality
story** (does anything converge as N grows, and is it rule-independent?) --
without it the model stays a toy regardless of ingredients. The cap→d plateau
work is the existing foothold.

### Causal calibration (step 3, 2026-07-29): the event DAG is NOT manifold-like -- d_causal != d_H+1

**Result: the pre-committed null `d_causal = d_H + 1` is REJECTED.** `causal_dag.py`
(new; `--validate` passes) records the causal DAG of async update events on a
*static* graph and estimates its causal-set dimension. On a static graph the DAG
is the product `G x Poisson-time`, so the free calibration says its dimension
must be the spatial Hausdorff dimension plus one. It is not: the emergent time
dimension is systematically **under-counted**, and the shortfall grows with
dimension until the "+1" vanishes entirely.

Known-answer lattices (integer targets), Myrheim-Meyer under the flat-space
calibration:

| true d = d_H+1 | graph        | MM   | midpoint | deficit |
|----------------|--------------|------|----------|---------|
| 2              | 1D path      | 1.9  | 1.6      | 0.1     |
| 3              | 2D lattice   | 2.5  | 2.0      | 0.5     |
| 4              | 3D lattice   | 3.15 | 2.37     | 0.85    |

And on the graph the physics actually uses (n=4000): `grown` cap6 reads
d_mm=2.32 vs target 2.85 (offset over d_H = +0.48); `grown` cap8 reads d_mm=2.65
vs target 3.67 (**offset +0.0 -- time contributes nothing**). Identical deficit
on an isotropic 2D random-geometric graph as on the cubic lattice, so it is
**not lattice anisotropy** -- it is fundamental to graph x Poisson-time.

**Why it is a genuine negative, not a bug** (verified three ways, and by an
independent adversarial audit):

- *Machinery is correct.* The transitive-closure relation matrix matches a
  brute-force full-DAG reachability computation exactly (0 mismatches); the DAG
  is bit-identical across rules (it depends only on the schedule and topology,
  the read set being the closed neighbourhood by construction); parent recording
  is complete.
- *Positive control isolates the cause.* On the **same** lattice, **same** Poisson
  event times, **same** sampler and estimators, swapping the async causal
  relation for an artificial fixed-speed light cone `dist <= c*(t)` recovers
  d ~ 3 (midpoint robustly ~2.8 across cone speeds). Only the async causal
  structure reads low. Both live in `causal_dag.py --validate`.
- *The estimators disagree.* On true Minkowski sprinklings MM and midpoint agree
  to within 0.09; on the event DAG they diverge (0.5 at 2D, 0.78 at 3D). That
  growing disagreement is the operational definition of a **non-manifold-like**
  causal set -- there is no single dimension the estimators concur on.

**Mechanism.** The async model has **no fixed light-cone speed**: a quick
succession of neighbour firings lets influence propagate many hops in near-zero
time (a last-passage-percolation effect), so the longest chain between two
events runs ~6-7x their worldline separation. The diamond is a genuine 3D
*volume* (`|I| ~ height^2.8`) but its causal *ordering* is denser than a 3D
Minkowski interval (ordering fraction 0.34 vs 0.229), which reads low and
distorts the volume bisection midpoint measures. This is exactly the
FPP-light-cone-shape sensitivity the spike pre-committed to as a legitimate
negative (spec §"scientific risk", pre-commitment #4).

**What it blocks and what it does not.** It BLOCKS using the event DAG to measure
*absolute* emergent spacetime dimension via Myrheim-Meyer -- the instrument does
not calibrate, so a causal-dimension number off it cannot be trusted. It does
**not** block the physics checkpoints: the spike's primary step-4 experiment
(shortcut censorship under async, LORENTZIAN_SPIKE.md §5) acts on the **state
graph** and needs no causal-set dimension, and the barrier checkpoint likewise
uses state-graph extent. So step 4 remains viable; only the causal-dimension
sub-observable is retired unless recalibrated. A graph-specific recalibration is
*mathematically* available -- the ordering fraction r_graph(D) is a clean
monotonic family (0.54 / 0.31 / 0.20 at true D = 2 / 3 / 4) -- but it cannot
rescue an absolute dimension while the estimators still disagree among
themselves, and calibrating against a known answer to apply where there is none
is the same anti-pattern that retired interval-scaling in step 1. **Methodology
lesson (reused): a clean monotonic calibration certifies invertibility, not
manifold-likeness.**

## Log-periodicity scan (2026-08-11): no discrete scale invariance -- bounded null

*Trigger: Ecker/Ecker/Grumiller (PRL 2026, arXiv:2601.14358) construct the Choptuik
critical solution analytically at large D -- a "spacetime crystal" periodic in log-scale
(discrete self-similarity). The graph analogue would be a log-periodic modulation riding
on one of this sandbox's scaling relations. Cheap to check, so it was checked.*
(`logperiodic_scan.py`; method pre-registered before residuals were examined.)

Tested: prune d(p) at three N (10 pts each), a dense 24-point `grown` extent~N grid and
per-seed ball-growth curves ln|B(r)| (generated by `--generate`); the historical barrier
(5 N-values) and cap->d (3 caps) grids are auto-declared INSUFFICIENT rather than
over-fitted. Lomb-Scargle on detrended residuals vs the log axis, permutation-null
p-values, Bonferroni across 7 curves, plus injection-recovery so every null is bounded
(a90 = smallest detectable modulation amplitude).

**Result: global null.** Both families that showed raw p<0.05 were resolved as
smooth-misfit artifacts by a discriminator ladder, not periodicity:

- **prune d(p)**: raw p 0.037-0.057 at all three N, but cross-N residual correlations
  +0.89..+0.98 (the same smooth shape at every N = the cubic detrend misfitting d(p),
  not noise) and the signal dies under quartic/quintic detrends at 2 of 3 N. Residual
  curiosity: d(p) has reproducible N-independent fine structure beyond a quintic --
  **follow-up below: the dense grid shows it is REAL and N-invariant to 16x; the
  pre-registered overnight tests then REFUTED the log-periodic extrapolation -- it is
  deterministic multi-bump structure in the crossover window, not a periodic law.**
- **ball growth**: raw p 0.001-0.003 at all three seeds under a quadratic detrend --
  but only ~1.5 "cycles" in range (one smooth bow plus the r=2 discreteness point),
  killed by quartic/quintic, and cross-seed residual correlation +0.95..+1.00: the
  deterministic non-polynomial shape of ln B(ln r) (local regime -> bulk power law ->
  saturation approach), shared by every seed. Not a crystal.

**Bounds:** no log-periodic modulation above ~10% amplitude on `grown` extent scaling,
~10-20% on ball growth after shape removal. prune d(p) sensitivity is insufficient at
n=10 (a90=inf) -- no claim either way there.

**Methodology lesson (banked):** a permutation test on max periodogram power detects
*any* autocorrelated residual structure -- a smooth misfit bow triggers it as readily as
genuine periodicity. Before calling anything DSS, demand (a) >= 2-3 cycles in range,
(b) survival under a detrend-order ladder (a polynomial absorbs a bow, not several
cycles), (c) an integer-staircase false-positive calibration for integer-derived
observables. `--validate` plants a 3-cycle crystal (must be found, ladder green) and
runs pure/staircase negative controls.

Reproduce: `python logperiodic_scan.py --generate` then `python logperiodic_scan.py`;
instrument check `python logperiodic_scan.py --validate`.

### Dense-grid follow-up (2026-08-11, run on jaga): the d(p) fine structure is REAL -- a log-periodic modulation of the pruned-WS dimension

The "residual curiosity" above was given a dense grid: 30 log-spaced p in [0.02, 0.5],
N=32000, 6 fresh seeds per point (30-way parallel on jaga, ~2 min wall clock). The
pre-registered scan now reads **perm-p = 0.0005 at n=30** with ~2.0 cycles in range --
and the signal survives quintic and sextic detrending (p = 0.001-0.002), which a smooth
misfit bow cannot do. (The discriminator ladder's mechanical all-orders rule printed
"misfit artifact" off one marginal deg4 = 0.022; the direct artifact tests below
override that label -- the ladder is a smooth-misfit screen, not the final word.)

Three discriminators, all pointing the same way:

- **Deterministic, not noise:** disjoint seed triples {0,1,2} vs {3,4,5} reproduce the
  same residual waveform at r = +0.83; every single seed carries it (mean pairwise
  r = +0.70, min +0.34).
- **Not a degree/pruning-generation staircase:** d-residuals track neither mean-degree
  residuals (r = -0.27) nor clustering residuals (r = -0.03).
- **Not the estimator's fixed fit window (the decisive test):** re-measuring the entire
  sweep at `--max-radius` 8 and 12 leaves the waveform intact -- cross-window residual
  correlations **+0.96 to +0.99**, period stable (1.64 / 1.58 / 1.55 ln-units). A
  fixed-integer-window artifact would shift with the window; this does not.

**The structure:** period ~1.6 in ln p (features recurring every ~5x in p), peaks near
p ~ 0.02 / 0.10 / 0.36, troughs near ~0.033 / 0.23, amplitude ~ +-0.05-0.08 in d_eff on
top of the smooth 1->2 crossover. Window-invariant, N-invariant (the cross-N r ~ +0.9
that first flagged it), seed-robust.

**Mechanism: OPEN.** Leading hypothesis -- a *protection hierarchy*: prune's survivors
are mutually-protecting shortcut clusters (triangles across shortcuts, the same
mechanism as the wormhole-throat protection cores), and the shortcut densities at which
2-fold, 3-fold, k-fold mutual protection first percolates should be spaced roughly
geometrically in p, which is precisely how log-periodic features arise. Concrete test,
not yet run: histogram surviving-shortcut cluster sizes vs p and check whether
cluster-generation onsets align with the d(p) peaks.

Irony worth recording: the pre-registered DSS hunt returned a clean global null on the
*grown* geometry -- and then the discipline it imposed (dense grid + artifact
discriminators) promoted its own throwaway curiosity into the sandbox's first genuine
log-periodic structure, in the *pruned-WS* ensemble instead.

Reproduce: sweep `python prune_dimension.py --nodes 32000 --ps <geomspace(0.02,0.5,30)>
--seeds 6 --seed 0 [--max-radius 8|10|12]` (one process per p; results merge by
concatenation), then `python logperiodic_scan.py --prune-csv <merged.csv>`.

### Overnight pre-registered tests (2026-08-12, jaga): N-invariance CONFIRMED at 16x; the log-periodic reading REFUTED

Predictions were committed before the data existed
(`docs/superpowers/specs/2026-08-12-dp-overnight-preregistration.md`, commit a134c15);
the analysis pipeline was frozen. Outcomes:

- **P1 (the sharp periodicity prediction) -- REFUTED.** A genuine log-periodic law
  (period ~1.6) demanded a third-cycle peak near p ~ 0.004 and trough near ~ 0.007.
  The 45-point grid down to p=0.004 (12 seeds) shows neither robustly: the apparent
  low-p "peak" is a grid-endpoint feature that inverts under quartic/quintic detrends
  (+0.126 -> -0.042 -> -0.014), the cubic trough lands at p=0.0107 (outside the
  pre-registered window), and the fitted period is range-unstable (1.58 on
  [0.02,0.5] -> 2.71 on [0.004,0.5]). Per the pre-registered falsifier, the reading
  **downgrades from "log-periodic" to "deterministic multi-bump fine structure
  confined to the crossover window [0.02, 0.5]"** (features near p ~ 0.02 / 0.10 /
  0.36, troughs ~ 0.033 / 0.23; below p ~ 0.02 the pruned graph is essentially the
  bare ring and the structure is gone). This retracts the previous subsection's
  "log-periodic" language; the structure itself stands.
- **P2 (N-invariance) -- PASS, decisively.** Waveform correlation with the N=32000
  reference: **+0.963 at N=8000, +0.945 at N=128000** (a 16x span), period shift
  0% / 2%. The structure is density-intrinsic -- a property of p and the ensemble,
  not of graph scale. (d(p) itself is also N-converged: mean d 1.173 / 1.174 / 1.173
  across the three N -- consistent with the banked crossover claim.)
- **P3 (cluster-composition mechanism probe) -- hierarchy story unsupported.**
  Surviving-shortcut cluster onsets (pairs from the lowest densities, triples ~
  p=0.165, 4+ clusters ~ 0.21) do NOT align with the d(p) peaks; they fall if
  anywhere in the 0.15-0.23 trough. Survival fraction and composition evolve
  smoothly. The simple "geometrically-spaced protection-generation onsets" mechanism
  is not supported; the mechanism is OPEN again.
- **Secondary (tighter grown ball-growth DSS bound) -- NOT achieved.** At N=100000
  the quadratic-detrend pipeline got *worse* (a90 0.2-0.4 vs 0.1-0.2 at N=20000):
  the smooth crossover-shape residual grows with the fitted r-range faster than
  noise shrinks. Methodological note: tightening this bound requires modeling the
  saturating ball-growth shape explicitly, not more N. The banked ~10-20% bound
  stands.

Net position: the pruned-WS d(p) crossover carries **real, deterministic,
density-intrinsic, N-invariant multi-bump fine structure** (window-invariant,
seed-robust, 16x-N-robust) whose mechanism is unknown -- and it is *not* discrete
scale invariance. Both "it is real" and "it is not a crystal" are now
well-supported, each by its own pre-registered test.

### Mechanism round 2 (2026-08-17, pre-registered): the integer-r_c-crossing candidate is KILLED

Candidate (spec `2026-08-17-dp-mechanism-round2-preregistration.md`, commit 4d4044c,
predictions before data): the pruned graph's ring->mesh crossover radius r_c(p) sweeps
down through integer fit radii, modulating d_eff. Killed on two independent grounds:

- **Coverage failure.** The operational r_c (local ln-ln slope crossing 1.5 within
  r <= 14) does not exist for 22 of 30 grid points -- including the features at
  p ~ 0.02, ~ 0.033, and ~ 0.10. In hindsight this was partly a formulation blunder:
  the local slope *is* the dimension estimate, so demanding s = 1.5 crossings in the
  region where d_eff ~ 1.0-1.3 was self-contradictory. The candidate could only ever
  have addressed the high-p features.
- **M2 killed by the frozen rule even where r_c exists.** On the 8 valid high-p
  points, residual vs frac(r_c) fits at **R^2 = 0.099** (< 0.2 = pre-registered kill),
  and M3 shows no consistent phase (the 0.23 trough sits at frac 0.235, the 0.36 peak
  at frac 0.179 -- same fractional class for opposite extrema). M1 (r_c monotone
  decreasing, rho = -1.00) held, but it was only the necessary condition.

Side-measurement: convergence-round counts drift smoothly 7 -> 10 across the grid with
no banding and no feature alignment -- candidate (b) (convergence-depth bands) gets no
descriptive support either. **Round 3 starts from the remaining candidate** (WS
shortcut-overlap statistics) **plus fresh formulation.** Cost accounting: the ill-posed
candidate cost one cheap overnight run precisely because it was pre-registered with a
kill rule -- the discipline is doing its job in both directions.

Reproduce: `rc_measure`/`rc_verdict` scratch pair per the spec's frozen definitions
(construction + convergence imported from `prune_dimension`; N=32000, 30-point grid,
3 seeds, 400 sources, r <= 15).

## Wormhole-throat critical collapse (stage 1, 2026-08-17): a local-motif onset, not a critical point -- and throat cores are censor-proof but not churn-proof

*Stage 1 of the critical-collapse program (the Choptuik protocol ported to the sandbox;
design spec `2026-08-17-throat-criticality-design.md`; driver `throat_criticality.py`).
All verdict rules and ensembles frozen before measurement; anchor a-values frozen from
the pilot (0.0109/0.0333/0.0473/0.0681/0.1408); FSS = 2000 draws per geometry.*

**The deterministic-core mechanism is EXACT (Gate 1, strongest possible form).** Under
prune-only dynamics the surviving-strand set equals the bootstrap-peeling fixed point of
the initial throat in *every one of 40 production runs* (5 frozen a-values x 8 seeds,
N=2000, T=400 sweeps) -- not merely "attributable mismatches": zero mismatches of any
kind, `survivors == core` exactly. Stochastic censorship dynamics on a throat is a
deterministic geometry computation plus timing noise. (The degree-floor escape hatch the
design pre-committed to measuring never fired at production scale.)

**The threshold exists per draw but is NOT a critical point (the sharpness verdict).**
Every throat has a finite critical thickness A*_j (300/300 pilot draws; 2000/2000 at
every FSS geometry). The FROZEN rule -- transition width w = q90-q10 of a* = A*/capacity
shrinking across capacities -- fires "sharp": w = 0.0716+-0.0023 -> 0.0397+-0.0008 ->
0.0256+-0.0007 (r=2/3/4, N=2000/5000/10000, capacities ~134/425/1043; logistic
cross-check agrees: 0.0707/0.0400/0.0252). But the final whole-branch code review caught
the confound *before the rule was read against data*, and the disclosure diagnostics
(added pre-verdict; frozen rule untouched) show the shrinkage is a trivial 1/capacity
rescaling: the location-relative width w/a50 RISES (1.473 -> 1.574 -> 1.655), the
ABSOLUTE strand-count width GROWS (7 -> 13 -> 22 strands), and A*'s median grows
(6 -> 9 -> 13). *[Revised 2026-10-17, wave 1 W1-a: the 2.6 is an estimator artifact (mean(core_frac) x median(A*)); the per-draw core at onset is exactly 2 strands in 91-98% of draws on every substrate. The motif-onset verdict stands; the number is withdrawn.]* The clincher: **the core at onset is ~2.6 strands at every capacity**
(core_frac@A* 0.435/0.288/0.204 x A*50 = 2.61/2.59/2.65) -- the threshold is literally
the first appearance of a single mutually-protecting motif. **Honest verdict: the frozen
rule is satisfied vacuously; the intended reading (a genuine collective critical point)
is NOT supported. A* is a local-motif onset, and the transition does not sharpen in any
capacity-honest normalization. Banked per the pre-registered crossover reading: no
critical collapse in the censor-only family -- consistent with every boundary this
sandbox has probed (prune d(p), the barrier).**

**Secondary predictions.**
- *Hybrid-jump signature:* reinterpreted by the motif finding. The onset core is
  "macroscopic" as a fraction of A* only because A* itself is small; the
  capacity-invariant statement is a constant ~2.6-strand absolute core -- a motif, not a
  jump to a macroscopic phase.
- *Critical slowing down:* qualitatively present -- deep-sub-threshold throats evaporate
  in ~7-39 sweeps (geometric, mean ~ 1/prune_prob) while near-threshold sub-critical
  throats stretch to 99-105 sweeps (longer peeling cascades). With the motif-onset
  reading this is cascade-depth growth, not a diverging correlation time.

**Gate 2 (descriptive, pre-registered both ways): triadic DEMOLISHES throat cores.** No
weaving rescue below threshold (0-1 strands kept across all empty-core runs), and
wholesale demolition above it: prune-only cores of 19-28 strands are reduced to 0-4
under triadic+prune (mean core retention ~0.1). A wormhole core is *permanently immune
to the censor* yet destroyed by the stabilizer's fabric churn -- the starkest form yet
of the banked portal/step-4 finding that the stabilizer is a worse threat than the
censor. Pre-registered reading (b) fires.

**Substrate observation (disclosed ensemble refinement + a finding in its own right):**
the grown generator occasionally lands in a compact expander-like phase (first seen at
seed 111, N=600: diameter 4, radius 2) -- the banked bistability appearing in the
GENERATOR itself. Such substrates cannot host a throat and are regenerated
deterministically with counts disclosed (1/300 pilot; **19/2000 at ALL THREE FSS
geometries**). The identical count across N=2000/5000/10000 is striking: the FSS seed
blocks are nested (methodology note ii), so the same growth seeds appear to fail at
every N -- i.e. **the expander phase is decided early in growth and persists to
N=10000** (~1% of seeds), not a small-N artifact. In BRANCHES as an open observation
(needs direct confirmation the 19 failing seeds coincide, then phase statistics vs N).

**Methodology notes.** (i) The width-statistic confound (normalizing an intensive motif
count by capacity manufactures sharpness) was caught by review BEFORE the verdict was
read -- the frozen rule is reported as written alongside the disclosure diagnostics, and
the verdict states which reading survives. (ii) The FSS geometries share nested growth
seeds (the N=2000 substrate is exactly the first 2000 nodes of the N=5000 one), so
cross-capacity comparisons are PAIRED -- disclosed rather than redrawn
post-registration. (iii) P(core|a) was computed as the CDF of per-draw bisected a*
(exact, ~11 peels/draw) -- valid because core existence is monotone in nested strand
sets, verified by test and by independent full linear scans in review.

**Consequences for the program:** stage 2 (universality) and stage 3 (driven injection)
were gated on a SHARP verdict -- the gate does not open; both close in BRANCHES. The
standing positives are the exact deterministic-core mechanism and the churn-demolition
result, and the motif-onset structure connects directly to the d(p) fine-structure
mechanism hunt (protection motifs in random shortcut ensembles -- the same object in a
different ensemble).

Reproduce: `python throat_criticality.py --validate`; `--pilot --draws 300`;
`--fss --draws 2000`; `--anchor --a-values 0.0109 0.0333 0.0473 0.0681 0.1408 --seeds 8
--rider`. Rows persist to `results/throat_*.csv`.

## Audit of 2026-09-26/27: `grown` and high-`p` pruned graphs have NO dimension

*A review of past work before starting new candidates. All thirteen `--validate` gates
and both smoke tests re-pass, and `barrier_scaling` reproduces its banked CSV bit-for-bit.
The instruments are sound at what they test. The problem found is in what one of them was
never asked.*

### The finding

`dimension.local_dimension` fits `log|B(r)| = d log r + c + a/r` over radii 1..`max_radius`
and gates on R². **A high R² certifies that a curve was fitted, not that the growth is a
power law** — the lesson this log already records for interval scaling (step 1), now
applying to the project's primary ruler. Slow *exponential* growth, `|B| ~ b^r` with `b`
near 1, is fitted at R² > 0.97 over any one window and returns a confident,
window-dependent number.

`window_stability.py` (new; `--validate` passes) asks the two questions the R² gate
cannot: does `d_eff` move when the fit window moves, and does a power law beat an
exponential on the same radii with the same number of parameters? Known answers first:

| control | window drift R6→R40 | model that wins | verdict |
|---|---|---|---|
| 2D torus | 1.95 → 1.99 (0.04) | power law, d = 2.00 | DIMENSION DEFINED |
| 3D torus | 2.94 → 2.99 (0.05) | power law, d = 3.01 | DIMENSION DEFINED |
| subdivided random 3-regular graph (exponential by construction, base 2^(1/6) = 1.1225) | 1.54 → 2.55 (1.07) | exponential, base 1.121 | EXPONENTIAL |

Then the project's own graphs (N = 2e5, 3 seeds each; `grown` also at N = 5.12e5):

| graph | `d_eff` by fit window | exponential base | verdict |
|---|---|---|---|
| `grown` cap 6 | R6 1.74 · R8 2.04 · **R10 2.20** · R12 2.34 · R16 2.73 · R20 3.11 · R24 3.46 · R32 3.95 | 1.30 | EXPONENTIAL, 3/3 seeds |
| `grown` cap 7 | R6 2.34 · R8 2.65 · **R10 3.05** · R12 3.32 · R16 3.96 · R20 4.52 | 1.47 | EXPONENTIAL, 3/3 |
| `grown` cap 8 | R6 2.61 · R8 2.99 · **R10 3.48** · R12 3.96 · R16 4.61 | 1.60 | EXPONENTIAL, 3/3 |
| pruned WS p = 0.05 | 1.00 at every window | — | DIMENSION DEFINED (d = 1), 3/3 |
| pruned WS p = 0.1 | 0.92 → 1.01 | — | DIMENSION DEFINED (d = 1), 3/3 |
| pruned WS p = 0.2 | ~1.0 → ~1.7 | tie | MIXED (2 seeds), EXPONENTIAL (1) |
| pruned WS p = 0.3 | 1.3 → 3.1 | 1.15 | EXPONENTIAL, 3/3 |
| pruned WS p = 0.4 | 1.6 → 3.7 | 1.24 | EXPONENTIAL, 3/3 |
| pruned WS p = 0.5 | 1.8 → 4.0 | 1.30 | EXPONENTIAL, 3/3 |

The bold R10 column reproduces the banked cap→d table (2.2 / 3.0 / 3.6) to two digits:
those numbers are what a radius-10 window reads off an exponential. Independent
confirmations on `grown` cap 6: the ratio `|B(r+1)|/|B(r)|` is flat at 1.26-1.31 from
r = 9 to r = 32; the diameter is 31 / 43 / 52 / 61 / 71 at N = 2e3 / 8e3 / 3.2e4 /
1.28e5 / 5.12e5 — about +10 per factor 4, i.e. **logarithmic**; and the live frontier is
a constant 37% of N at every size.

**Why.** Every `grown` step attaches a new node to an existing edge, so the graph is a
partial 2-tree (treewidth ≤ 2): tree-like at every scale above a few hops, however
triangle-rich it is locally. Nothing in the rule makes two growing arms meet, so the
frontier stays proportional to the volume, and a frontier proportional to volume *is*
exponential growth. A d-dimensional object needs a frontier ~ N^((d-1)/d).

### What is retracted, and what stands

Retracted as stated:

- **"`grown` has emergent dimension, tunable by the cap" (strong emergence).** `grown`
  has no dimension. The cap tunes an exponential growth *rate*; 2.2 / 3.0 / 3.6 are
  radius-10 readings of it.
- **"cap→d plateaus at scale."** The plateau in N is real, but it is convergence of a
  fixed-radius local measurement, which any graph with N-independent local structure
  shows. It was never evidence of a power-law regime.
- **"`prune` is a third continuum dimension knob, 1 → 2."** Pruning reveals a genuine
  d = 1 ring for p ≲ 0.1. Above p ≈ 0.2 the surviving shortcuts make the pruned graph a
  small world again, with no dimension; the smooth d(p) curve is a fixed-window reading
  of a ring → small-world crossover. "Crossover, not a transition" stands and is
  strengthened.
- **"`grown` is locally 2D but globally compressed (diameter ~ N^0.19)."** The diameter is
  logarithmic; 0.19 was a power law fitted to a logarithm over one decade.
- **Step 3's `grown` rows** used `d_H + 1` targets (2.85, 3.67) built on these readings.
  The lattice and random-geometric rows, which carry the step-3 negative, are unaffected.

Stands unchanged:

- **The bootstrapping barrier** — its positive control is the hand-built lattice, which
  is window-stable. The audit *extends* it: frontier growth fails to make extent too,
  for the same structural reason. There is now no example in this project of a local rule
  that **produces** polynomial extent; the only real geometries are the lattice (built by
  hand) and the low-p pruned ring (latent in the construction).
- **Everything measured *on* `grown` as a substrate** — portal tolerance, censorship
  (sync and async), throat peeling, walkers, preservation, coherence. These are correct
  statements about dynamics on a triangle-rich, tree-like fabric. What changes is the
  description of the substrate: it is not "a coherent ~2D geometry". In particular
  "dimension inflation 2.21 → 3.19 under portals" and Moran's I of the `d_eff` field are
  statements about the radius-10 reading.
- **The `d(p)` fine structure** is real and N-invariant as measured, but it is structure
  in a fixed-window ball count, not in a dimension. That reframes the mechanism hunt.

### Growth extinction (the "persistent expander phase" is not a phase)

BRANCHES carried an open observation: 19 of 2000 `grown` draws fail at all three FSS
geometries, read as a compact expander phase persisting to N = 10000. Checked directly:
the 19 failing draws are **the same 19 seeds** at all three geometries (the FSS blocks are
offset by exactly 17), and each is a graph of **7 to 17 nodes** whatever N was requested.
Growth stops when every frontier edge has an endpoint at the cap; `_grow_dimensional`
then returns the small graph silently. "Seed 111, N = 600, diameter 4" is a 15-node graph.

| cap | extinction probability | sizes at extinction |
|---|---|---|
| 4 | 0.474 ± 0.009 | 5-6 |
| 5 | 0.279 ± 0.008 (survival plateaus at 0.718 beyond ~600 nodes) | 6 - 537, heavy tail |
| 6 | 0.0082 ± 0.0006 (164 / 20000) | 7-20 |
| 7, 8 | 0.0002 | 8, 12 |

Extinction is decided in the first few dozen attachments (a few hundred at cap 5) and
never later. No banked result is contaminated: the lowest extinct cap-6 seed is 75, the
drivers use seeds 0-39 or 3000-3039, and `throat_criticality` regenerated its 19 (and
seed 5006 in the anchor block) with disclosure. `create_initial_graph` now warns when
`grown` returns fewer nodes than requested.

Reproduce: `python window_stability.py --validate`; `python window_stability.py --nodes
200000 --seeds 3`.

### The estimator now gates on window stability (2026-09-28, owner-approved)

`dimension.local_dimension` has a third gate. After the fit passes the scale-separation
and R² gates, it re-fits the inner half of the window and requires the slope not to move:
`|d(radii 1..n) - d(radii 1..k)| <= 0.25`, `k = max(5, ceil(n/2))`. `dimension_stats`
adds the field-level verdict, which is the decisive one: `median_drift` over every
testable node, and `window_stable` (true when `|median_drift| <= 0.10`, `None` when fewer
than half the sampled nodes have the 8 unsaturated radii the test needs).
`drift_tol=None` restores the old behaviour. `dimension.py --validate` now has 15 checks.

What it does to the numbers quoted in this log (max_radius 10, 400 samples):

| graph | `defined_frac` before | after | median drift | `window_stable` |
|---|---|---|---|---|
| 2D lattice, N = 40000 | 1.00 | 1.00 | +0.02 | yes |
| pruned WS p = 0.05 | 1.00 | 1.00 | +0.00 | yes |
| pruned WS p = 0.1 | ~1.00 | 0.89 | +0.03 | yes |
| pruned WS p = 0.2 | ~1.00 | 0.58 | +0.10 | borderline |
| pruned WS p = 0.3 | ~1.00 | 0.37 | +0.22 | no |
| pruned WS p = 0.5 | 0.97 | 0.23 | +0.53 | no |
| `grown` cap 6, N = 50000 | 0.99 | 0.21 | +0.51 | no |
| `grown` cap 7 | ~1.00 | 0.14 | +0.75 | no |
| `grown` cap 8 | ~1.00 | 0.08 | +1.06 | no |

Two things to know before relying on it.

- **The per-node gate thins exponential graphs; it does not empty them.** Ball counts
  from one node are noisy, so a fifth of `grown` cap-6 nodes still pass. Read
  `window_stable`, not `defined_frac` alone. `coherence.py` already refuses its verdict
  on a fragmented field, so on `grown` it now reports "field fragmented (N/A)" where it
  used to report COHERENT, I = 0.88.
- **It certifies only the radii it is given.** Exponential growth of base 1.26 is caught
  at radius 10 (0% defined). Growth of base 1.12 is not: it looks like d ~ 1.5 out to
  radius 10, drifts only beyond, and passes. `--validate` prints that case as a known
  blind spot. `window_stability.py`, which audits to radius 40, remains the check to run
  on any new graph family.

**Every banked number that depends on `defined_frac` of `grown` or of pruned graphs above
p ~ 0.2 is no longer what the code returns.** That covers the preservation table, the
coherence table, portal tolerance, and the `prune` d(p) curve. They are left in this log
as recorded, under the supersession notice; none has been re-run under the new gate.

## Step 5 (2026-09-27): the bootstrapping barrier survives asynchronous updates

*Pre-registration `docs/superpowers/specs/2026-09-27-step5-async-barrier-preregistration.md`
(committed before data, with one disclosed post-hoc amendment); driver `async_barrier.py`
(`--validate` passes). `triadic` from a random graph, N = 1000..32000, 8 paired seeds, 200
sweep-equivalents, sequential Poisson-clock engine against the synchronous rule on the
identical start graph.*

| | sync | async |
|---|---|---|
| pooled alpha (extent ~ N^alpha) | 0.155 (R² 0.95) | 0.167 (R² 0.97) |
| per-seed alpha, mean ± SE | 0.158 ± 0.022 | 0.168 ± 0.013 |
| 95% CI | [0.106, 0.211] | [0.136, 0.200] |
| mean diameter, N = 1000 → 32000 | 10.5 → 18.0 | 10.5 → 18.8 |
| largest-component fraction | 0.984 | 0.984 |

Paired difference async − sync: **+0.010 ± 0.032**, 95% CI [−0.067, +0.086].

**Verdicts under the frozen rules.**

- **BARRIER SURVIVES.** The async exponent's upper bound is 0.200, under the 0.25
  threshold and far from the lattice's 0.515. The flagship negative is not an artifact of
  the global sweep clock.
- **Schedule invariance: UNDERPOWERED.** The difference is consistent with zero, but its
  interval is wider than the ±0.05 band, so invariance is not established. Integer
  diameters of 10-18 make per-seed exponents coarse. No seeds were added to rescue it.

**Two gates failed as frozen, both through tolerances set badly in the pre-registration,
and both are recorded as failures.**

- *Gate 0* required my sync arm's final diameters to match the banked CSV within ±2. The
  banked driver itself reproduces that CSV **exactly**; my arm, which draws a different
  random realization, differs by up to 4 with no bias (mean 15.0 vs 14.9). Realization
  scatter is larger than I assumed.
- *Gate 1* (`prune` on small-world as a control that does grow extent) required the two
  schedules' exponents to agree within 0.1; they read 0.825 and 0.576. Pruning fragments
  the ring (largest component 31-100%), so that diameter hinges on a handful of edges.
  The amended Gate 1A, written after seeing this and before reading Gate 2, compares the
  final edge sets instead: worst Jaccard distance 0.0052 over 16 runs, diameter growth
  ×9 to ×97 in every run. It passes — and is post hoc.

**The 200-sweep budget is a transient, not a converged state** (rider, 800 sweeps,
N = 1000 and 4000, 4 seeds): extent rises to a peak near 300 sweeps and then *falls*, in
both schedules alike — N = 1000: 8.5 → 11.5 → 4.0 (sync), 8.5 → 13.0 → 4.2 (async) —
while the largest component stays at 96-97%. So it is crumpling, not fragmentation.
The banked alpha ≈ 0.13 is therefore a statement about a 200-step budget; run longer,
`triadic` ends with *less* extent than the expander it started from. This strengthens
the barrier and means the exponent should not be quoted as an asymptotic quantity.

Scope: one rule (`triadic`); `geometrize` and `ricci` are not async events.

Reproduce: `python async_barrier.py --validate`, then `--gate0`, `--gate1`, `--gate1a`,
`--gate2 --jobs 12`, `--long --jobs 8`.

## d(p) fine structure, round 3 (2026-09-27): the waveform lives at large radius

*Pre-registration `docs/superpowers/specs/2026-09-27-dp-round3-decomposition-preregistration.md`
(one disclosed post-hoc amendment); driver `dp_decomposition.py` (`--validate` passes).*

Rounds 1 and 2 each guessed a mechanism and were killed. Round 3 locates the waveform
instead. The fitted slope is exactly linear in the log ball counts,
`d = sum_r w_r ln|B(r)|`, so the waveform splits into one contribution per radius with no
modelling.

| gate | result |
|---|---|
| R (bit-reproduce the banked grid) | **FAIL as frozen** — graphs rebuild identically, the 400-node sample does not (55 of 180 rows agree to 1e-9; worst difference 0.067) |
| R' (amended, post hoc: waveform correlation with the banked grid) | +0.991, PASS |
| M1 (median-of-fits vs fit-of-means) | +0.873, PASS |

| radius | fit weight | peak-to-peak of detrended log ball count | seed SE | variance share |
|---|---|---|---|---|
| 1 | +0.52 | 0.015 | 0.003 | +0.01 |
| 2 | −0.60 | 0.023 | 0.003 | −0.02 |
| 3 | −0.62 | 0.054 | 0.003 | −0.07 |
| 4-6 | −0.46 … −0.07 | 0.07 - 0.10 | 0.004 | −0.25 |
| 7 | +0.12 | 0.111 | 0.005 | +0.07 |
| 8 | +0.30 | 0.127 | 0.006 | +0.22 |
| 9 | +0.46 | 0.145 | 0.007 | +0.41 |
| 10 | +0.62 | 0.162 | 0.008 | +0.64 |

**Reading: MESOSCALE, not amplified.** Radii ≥ 4 carry 109% of the variance and radii
≤ 3 carry −9%; the waveform is not a product of the fit's weights (its peak-to-peak,
0.125, is below the raw per-radius one). It is far above seed noise — 0.16 against 0.008
at r = 10 — so it is real structure in how many nodes lie within ten hops.

What this rules out: every local explanation. Degree, triangles and the second shell
(r ≤ 3) contribute nothing, which is why the earlier residuals tracked neither mean
degree nor clustering.

**A hypothesis for round 4, stated after seeing the data and therefore not tested
here.** The detrended `ln|B(10)|` is smooth and slowly varying in p: positive, negative,
positive, negative, positive across the grid, with no sharp features. That is the
signature of a polynomial failing to follow a crossover. `|B(10)|` is pinned at the
ring's value at low p and rises once shortcuts come within reach; a cubic in `ln p`
cannot follow a curve that is flat and then rises, and what it leaves behind oscillates.
On this reading there is no mechanism to find: the "multi-bump fine structure" is the
ring → small-world crossover seen through a polynomial detrend, and it is N-invariant,
seed-deterministic and window-invariant because the crossover is. The banked survival
under a sextic detrend argues against it and has to be faced by the test. Round 4 would
fit a crossover form to fresh seeds and ask whether the residual falls to seed noise.

Reproduce: `python dp_decomposition.py --validate`, then `python dp_decomposition.py
--jobs 6`.

## Spectral-dimension flow (2026-09-27): `grown` flows slowly from ~1.2 to ~1.9 -- my prediction was wrong

*Pre-registration `docs/superpowers/specs/2026-09-27-spectral-flow-preregistration.md`
(committed before any walk data); driver `spectral_flow.py` (`--validate` passes).
Lazy-walk return probability by a Hutchinson trace; `D(t) = -2 dln P / dln t`.*

**Gate A (instrument) passes.** The trace matches the exact spectrum to 2-5% on small
graphs, and every known answer reads FLAT at the right value:

| anchor | truth | measured |
|---|---|---|
| ring | 1 | 0.998 |
| 2D torus | 2 | 2.027 |
| 3D torus | 3 | 3.041 |
| Sierpinski gasket | 2 ln 3 / ln 5 = 1.365 | 1.368 |
| random 3-regular (negative control) | none | not FLAT (2.8 → 6.1 in two octaves) |

The gasket is the control that matters: its Hausdorff dimension is 1.585, so the
instrument can read a spectral dimension that differs from the Hausdorff one.

**Gate B.** `grown` caps 6 / 7 / 8, N = 5e4 and 2e5, 3 seeds each (18 runs):

| cap | D at t = 16 | D at mid-range | D at the end (t ~ 1e4) |
|---|---|---|---|
| 6 | 1.15 | 1.50 - 1.56 | 1.78 - 2.05 |
| 7 | 1.23 - 1.25 | 1.63 - 1.71 | 1.88 - 2.02 |
| 8 | 1.30 - 1.32 | 1.67 - 1.79 | 1.89 - 2.05 |

- **P1 (predicted RUNAWAY at every run) -- REFUTED.** 1 run of 18 reads RUNAWAY; 9 read
  PLATEAU-TO-PLATEAU FLOW and 8 UNCLASSIFIED. I inferred from exponential ball growth
  that the walk would mix like an expander's. It does not.
- **P2 (N-robustness) -- PASS.** The seed-mean curves at N = 5e4 and 2e5 agree within
  0.045 at every common grid point (tolerance 0.15).
- **P3 (ordering cap 6 < 7 < 8) -- holds at 17 of 17 points at N = 5e4 and 19 of 20 at
  N = 2e5**; it fails at the last point, where the three curves have converged to within
  their seed scatter. As frozen ("at every admissible t") that is a FAIL.

**What the curve is.** A slow, monotone, decelerating rise from about 1.2 to about 1.9
over three decades of walk time, the same at both sizes. The classifier splits between
two labels because the curve sits on the boundary of its plateau tolerance; the honest
description is the curve, not either label. Whether it levels at 2 cannot be decided
inside the admissible range.

**Reading.** `grown` has exponential volume growth but does *not* behave like an
expander for a walker. The pruned small world is the contrast: at p = 0.3 and 0.5 it reads
RUNAWAY in 4 of 4 runs (D climbing past 4), so the instrument does report expander-like
mixing when it is there. `grown` is tree-like -- treewidth <= 2 -- and a tree is full of
bottlenecks: a walker spends its early time inside one arm, which looks nearly
one-dimensional, and only slowly discovers the branching. So the volume and the walk
disagree completely: ball growth says "no dimension, exponential", the walk says "between
1 and 2". That is a genuine `d_s` / `d_H` split, of the kind trees are known for, and it
is not a CDT-like flow between two dimensions of a manifold.

Two more tree-like graphs read the same way (one seed each, N = 512000): `sheet` at
beta = 0 flows 1.40 → 1.92, and the negatively curved `sheet` c = 7 flows 1.57 → 2.03.

Scope: pruned p = 0.1 was not measured -- its largest component fell under the driver's
size threshold.

Reproduce: `python spectral_flow.py --validate`; `python spectral_flow.py --graphs grown6
grown7 grown8 --nodes 50000 200000 --seeds 3`.

## `sheet` (2026-09-27): a strictly local growth rule that produces a real dimension

*Sidequest. Pre-registration
`docs/superpowers/specs/2026-09-27-sheet-growth-preregistration.md` (confirmatory run on
fresh seeds, exploratory numbers disclosed in it); generator and driver `sheet_growth.py`
(`--validate` passes).*

The audit says why `grown` fails: nothing makes growing arms meet, so the frontier stays
proportional to the volume. `sheet` adds three local ingredients, none using a
coordinate:

- **manifold** -- attach only at edges that lie in exactly one triangle;
- **closure** -- a boundary vertex whose degree reaches `c` joins the two ends of its
  link, completing its wheel (reads the same radius as `triadic`);
- **tension** -- events fire on Poisson clocks with rate `exp(beta * (deg u + deg v))`,
  so nearly complete neighbourhoods fill in first.

**Confirmatory run**, N = 512000, seeds 100-102, per-octave exponents of seed-mean
diameter and boundary length:

| c | beta | diameter exponent, last two octaves | boundary exponent, last two octaves | window audit | spectral | verdict (frozen rules) |
|---|---|---|---|---|---|---|
| 6 | 0 | 0.110, 0.106 | 1.002, 1.000 | EXPONENTIAL | flow 1.40 → 1.92 | TREE-LIKE (predicted) |
| 6 | 1 | 0.248, 0.174 | 1.074, 1.037 | defined, d = 1.87 | flow 1.89 → 1.46 | TREE-LIKE (predicted) |
| 6 | 2 | **0.496, 0.500** | **0.499, 0.503** | DEFINED, d = 1.99, 3/3 | FLAT 1.977 / 1.991 / 1.985 | **GEOMETRY** |
| 6 | 3 | **0.501, 0.505** | **0.509, 0.502** | DEFINED, d = 1.99, 3/3 | FLAT 1.985 / 1.999 / 1.941 | **GEOMETRY** |
| 7 | 2 | 0.093, 0.095 | 1.000, 1.000 | EXPONENTIAL | unclassified | not GEOMETRY (predicted) |

(Window audit and spectral columns: three seeds each for beta = 2 and 3, one seed for the
other rows.)

At beta = 2 the diameter is 56 / 111 / 226 / 450 / 899 at N = 2e3 ... 5.12e5 -- it doubles
with every factor 4 -- and the boundary is 0.5% of the volume at the largest size, within
about 15% of the perimeter of a perfect disc.

**The curvature knob behaves as geometry says it should.** `c = 5` closes into a graph of
exactly 12 nodes in 5 of 5 seeds: the icosahedron. `c = 7` is negatively curved and
grows exponentially whatever the rates do. `c = 6` is flat.

**What this is, stated plainly.**

- It is the project's first case of a strictly local rule **producing** a window-stable
  dimension -- not revealing one latent in the construction, and not a fixed-window
  reading. Four independent observables agree on 2.
- It is **not** a spontaneously selected dimension. `c = 6` selects flatness and
  `beta >= 2` selects compactness. Two knobs.
- Every interior vertex has degree exactly 6 (fraction 1.000 in every run). The object is
  a **patch of the triangular lattice, assembled without coordinates**. That is less
  than "a geometry emerged" and more than "a lattice was written down": no step of the
  rule knows where anything is.
- beta = 1 is the cautionary case. It reads two-dimensional up to N ~ 8000 and then
  turns tree-like. beta = 2 and 3 show no such turn through N = 512000, and their
  numbers are indistinguishable from each other, which suggests a transition between
  beta = 1 and 2 rather than a persistence length that merely grows -- but nothing here
  tests that, and a crossover beyond 512000 is not excluded.
  *[Tested 2026-09-28/29: a crossover, confirmed at 4 M nodes -- beta = 2 turns tree-like
  at ~800,000 nodes, by a sudden per-seed event. See "`sheet` at beta = 2 turns tree-like".]*

**A limit of the window audit, found here.** It labels beta = 1 "DIMENSION DEFINED"
(drift 0.09). It probes radii up to 40; the beta = 1 graph is two-dimensional at that
scale and tree-like above it. The audit certifies the scales it looks at and no more.
At beta = 2 the large-scale evidence is the diameter and boundary exponents and the
walk, not the audit.

Reproduce: `python sheet_growth.py --validate`; `python sheet_growth.py --betas 0 1 2 3
--nodes 512000 --seeds 3 --seed 100 --jobs 7 --save-graphs`; then `python
sheet_growth.py --judge results/sheet_c6_b2_s100_N512000.npz`.

## `sheet` rate strength (2026-09-28): a crossover, thinly -- the two-dimensional range grows like exp(5.8 beta)

*Step 1 of the plan of 2026-09-28. Pre-registration
`docs/superpowers/specs/2026-09-28-sheet-beta-transition-preregistration.md` (committed
before the scan); driver `sheet_transition.py` (`--validate` recovers both planted laws).
c = 6, seeds 200-203, N to 512000, boundary length at every factor 2.*

Local boundary exponent `e` by size (compact 0.50, tree-like 1.00), and the crossover
size N* where it passes 0.75 for good:

| beta | N* | e at N ~ 1.4e3 ... 3.6e5 |
|---|---|---|
| 1.0 | 5,600 | 0.48 0.65 0.75 0.90 1.00 1.00 0.97 0.98 1.00 |
| 1.1 | 16,200 | 0.55 0.58 0.62 0.66 0.84 1.08 1.06 1.02 1.03 |
| 1.2 | 9,300 | 0.55 0.52 0.62 0.80 0.85 0.91 0.94 0.99 1.00 |
| 1.3 | 17,900 | 0.51 0.52 0.51 0.59 0.83 0.93 0.93 0.99 0.94 |
| 1.4 | 32,700 | 0.51 0.51 0.51 0.55 0.68 0.81 0.89 0.98 1.01 |
| 1.5 | 151,000 | 0.50 0.52 0.52 0.57 0.59 0.63 0.69 0.77 1.15 |
| 1.6 | 164,000 | 0.53 0.50 0.50 0.51 0.50 0.55 0.64 0.77 1.10 |
| 1.7 | 320,000 | 0.49 0.52 0.50 0.51 0.51 0.51 0.50 0.51 0.80 |
| 1.8 | > 512,000 | 0.51 0.50 0.51 0.51 0.51 0.50 0.50 0.50 **0.66** |
| 1.9 | > 512,000 | 0.49 0.53 0.50 0.51 0.49 0.51 0.55 **0.66 0.68** |
| 2.0 | > 512,000 | 0.53 0.50 0.52 0.50 0.50 0.50 0.51 0.51 0.54 |

| law | fit | RSS | AICc |
|---|---|---|---|
| exponential (crossover) | ln N* = 2.68 + 5.82 beta | 1.34 | -7.88 |
| power law (transition) | ln N* = 10.23 - 5.49 ln(2.325 - beta) | 1.13 | -3.68 |

**Verdict under the frozen rules: CROSSOVER SUPPORTED.** The two laws fit about equally
well; the exponential wins because it needs one parameter fewer. Its consistency clause
passes: it predicts N*(1.8) = 515,000, and beta = 1.8 shows no crossover inside the range.

**How thin this is.**

- The consistency clause passed by 0.6% (515,000 against a range ending at 512,000).
- N* is not monotone (beta = 1.1 reads higher than 1.2). With four seeds the seed scatter
  of the last exponent is 0.1 - 0.4 wherever the crossover is in progress.
- A transition at beta_c = 2.33 is not excluded. It fits the finite points slightly
  better and is penalised only for its third parameter.

**What speaks for a crossover beyond the fit.** beta = 1.8 and 1.9 are censored by the
rule but their last exponents have left 0.50 (0.66, 0.68): they are turning, at the size
the exponential law says they should. If there were a compact phase beginning near 2.3,
nothing requires 1.8 and 1.9 to turn inside this range; if the law is exponential, they
must.

**A prediction that can fail.** The exponential law puts N*(2.0) at 1.6 million. The
2026-09-27 result "beta = 2 is two-dimensional through N = 512,000" is then a statement
about sizes below its crossover. Growing beta = 2 to four million nodes decides it: tree-
like by then if this is a crossover, still compact if there is a transition below 2.33.

**What `sheet` is, restated.** A strictly local rule that produces a two-dimensional
sheet up to a size set by the rate strength, about exp(2.7 + 5.8 beta) nodes: 5e8 at
beta = 3, 2e11 at beta = 4. Beyond that size it is tree-like, like `grown`. As a substrate
for experiments at N <= 1e6 it is two-dimensional at beta >= 3 on this reading, and that
qualifier has to travel with it.

Reproduce: `python sheet_transition.py --validate`; `python sheet_transition.py --jobs 14`.

## d(p) fine structure, round 4 (2026-09-28): not one crossover -- two, to within seed noise

*Step 4 of the plan. Pre-registration
`docs/superpowers/specs/2026-09-28-dp-round4-crossover-preregistration.md`; driver
`dp_round4.py` (`--validate` passes). Fresh seeds 6-11.*

Hypothesis H: the waveform is a single ring-to-small-world crossover seen through a
polynomial detrend. RMS residual of each detrend, in units of the seed standard error:

| radius | cubic (4 par.) | single crossover (4 par.) | sextic (7 par.) | two crossovers (7 par.) |
|---|---|---|---|---|
| 4 | 3.5 | 2.1 | 1.2 | 0.8 |
| 6 | 3.7 | 4.9 | 1.3 | 0.7 |
| 8 | 4.2 | 8.7 | 1.6 | 0.8 |
| 10 | 4.8 | 5.8 | 1.7 | 0.9 |

**Verdict under the frozen rules: H REFUTED.** A single crossover fits *worse* than the
cubic (5.8 against 4.8 seed errors at radius 10), and its residual is reproducible
(split-half correlation +0.92). The log ball counts are not one flat-then-rising curve.

**The comparator, which carries no verdict, is the informative part.** Two crossovers
added together leave residuals of 0.7 - 0.9 seed errors at every radius from 4 to 10.
That is what a correct 7-parameter model of 30 points should leave (about 0.88). A
sextic polynomial, with the same number of parameters, leaves 1.2 - 1.7. So the structure
that "survived a sextic detrend" in August survived because polynomials are the wrong
family, not because there is a ripple.

**Reading, with its limits.** The ball counts are described to within noise by a smooth
curve with two characteristic shortcut densities and nothing else. That is a description,
not yet a mechanism: this round did not identify what the two scales are. The natural
candidates are the density at which surviving shortcuts come within ten hops, and the
density at which the ring backbone itself starts to thin; neither was tested. The
waveform left in d under the two-crossover form has peak-to-peak 0.050 against a noise
expectation near 0.03, so a small remainder is not excluded.

Combined with the audit, the d(p) thread now reads: a fixed-window reading (audit) of ball
counts at radius >= 4 (round 3) that follow a smooth two-scale crossover (round 4). No
periodicity, no hierarchy, no integer crossings.

Reproduce: `python dp_round4.py --validate`; `python dp_round4.py --jobs 5`.

## Portals on substrates with a dimension (2026-09-28/29): censorship holds, tolerance is a threshold, and the walker result was the tree

*Step 3 of the plan. Pre-registration
`docs/superpowers/specs/2026-09-28-portals-on-geometry-preregistration.md` (committed
before any run); driver `portals_on_geometry.py` plus `shortcut_walkers.py --topology`.
Substrates: the triangular lattice, `sheet` (c = 6, beta = 3), and `grown` re-run under
the identical code as the paired reference. `triangular` and `sheet` are now topologies
in `create_initial_graph`.*

Every banked portal result was measured on `grown`, which the audit showed to be
tree-like. This step asks which of them were about portals in a geometry and which were
about portals in a tree. Two of four predictions held; the two about walkers failed, and
their failure is the finding.

### E1 censorship (N = 2000, 40 long + 20 detour-2 portals, 120 steps, 5 seeds)

| substrate | condition | long survival | detour-2 survival | rank(adv, t) | woven | collateral |
|---|---|---|---|---|---|---|
| triangular | prune | 0.010 | 1.00 | -0.08 | -- | 0.000 |
| sheet | prune | 0.015 | 1.00 | -0.03 | -- | 0.000 |
| grown | prune | 0.020 | 1.00 | -0.03 | -- | 0.000 |
| triangular | triadic+prune | 0.065 | 0.06 | -0.08 | 2.2 | 0.855 |
| sheet | triadic+prune | 0.040 | 0.07 | +0.01 | 0.8 | 0.853 |
| grown | triadic+prune | 0.140 | 0.21 | -0.01 | 2.2 | 0.583 |

**P1 (threshold, advantage-blind) holds on every substrate**, as predicted: it is a
property of the rule. **P2 (triadic self-stabilization) holds on every substrate but is
weaker on a real dimension** -- long survival 0.04-0.065 against 0.14 on `grown` -- while
`triadic` churns *more* of a 2D fabric (85% of edges against 58%). The banked reading
"the stabilizer is a worse threat than the censor" is stronger on a geometry than it was
on the tree.

### E2 tolerance (N = 5000, fixed r0 = 10, 3 seeds)

| substrate | k | placed | defined | median drift | window-stable | d_eff | Moran I | mean distance |
|---|---|---|---|---|---|---|---|---|
| triangular | 0 | 0 | 0.96 | +0.02 | 3/3 | 1.86 | 0.89 | 37.5 |
| triangular | 10 | 10 | 0.84 | +0.05 | 3/3 | 1.95 | 0.85 | 23.4 |
| triangular | 20 | 20 | 0.69 | +0.14 | **0/3** | 2.04 | 0.86 | 20.4 |
| triangular | 50 | 50 | 0.41 | +0.30 | 0/3 | 2.32 | 0.92 | 16.0 |
| triangular | 100 | 100 | 0.29 | +0.43 | 0/3 | 2.73 | 0.70 | 13.3 |
| triangular | 400 | 125 | 0.33 | +0.46 | 0/3 | 2.81 | 0.65 | 12.8 |
| sheet | 0 | 0 | 0.97 | +0.02 | 3/3 | 1.87 | 0.91 | 37.9 |
| sheet | 20 | 20 | 0.71 | +0.12 | **0/3** | 2.04 | 0.83 | 20.5 |
| sheet | 400 | 128 | 0.31 | +0.46 | 0/3 | 2.83 | 0.65 | 12.9 |
| grown | 0 | 0 | 0.27 | +0.43 | 0/3 | 2.15 | 1.10 | 19.8 |
| grown | 400 | 400 | 0.17 | +0.68 | 0/3 | 3.13 | 0.64 | 13.0 |

**Predicted and found: portals do not inflate a real dimension, they destroy it.** The
`d_eff` of the still-defined nodes does climb with k (1.86 → 2.8), which is the banked
"inflation" -- but the window audit turns unstable at **k* = 20** *(revised 2026-10-17, wave 3 W3-d: k* is the Newman-Watts crossover, proportional to N / r0^2 -- 20 at N = 5000 with radius 10, 80 at N = 20000 -- not a fixed tolerance of the planes)* on both 2D substrates
(predicted range 10-50), and `defined_frac` falls from 0.96 to 0.3. Above k = 20 the
number is a reading of one fit window, not a dimension. A 2D substrate of 5000 nodes also
exhausts its far pairs at ~125 portals, where `grown` accepted 400: distances that grow
like sqrt(N) run out sooner. `grown` is unstable already at k = 0, as the audit said.

### E3 walkers (N = 1500, 10 seeds, offset portal ~2 hops from source and target)

| substrate | source-target distance | classical t_hit gain | classical pi gain | quantum Pbar gain (adjacency) | (Laplacian) | baseline Pbar x N |
|---|---|---|---|---|---|---|
| triangular | 41 | **1.4x** | 1.00x | **1.0x** | 1.0x | 0.98 |
| sheet | 38 | **1.3x** | 1.00x | **1.0x** | 1.0x | 0.97 |
| grown | 24 | 6.7x | 1.00x | 122x | 166x | 0.03 |

**Both walker predictions FAILED, in the same direction.** I predicted a larger classical
gain on the 2D substrates (longer distances, larger advantage) and a quantum gain above
10x everywhere. Instead, on a real geometry an offset portal is almost invisible to both
walkers in the long run: 1.3-1.4x classically, **1.0x quantum-mechanically**. Only the
direct source-target edge does anything (2.4x quantum, 3-4x classical).

The last column says why. On the 2D substrates the no-portal quantum occupancy already sits
at equipartition (`Pbar x N` ~ 1): the eigenstates of a lattice are spread over the
whole graph, so there is nothing for a portal to improve. On `grown` the baseline is
30-100 times *below* equipartition -- the walker is localized in the tree's arms -- and
the portal lifts it to equipartition. **The banked headline "a portal is a structural
device quantum-mechanically" (3c, x48-131) was a statement about localization on a
tree-like graph, not about portals in a geometry.** The same holds classically: the
6.7x hitting-time gain on `grown` came from a walker confined to an arm that finds the
portal mouth at once; on a plane it wanders off before finding a node two hops away.

What survives of the walker section: the *method* (horizon-free Pbar, degeneracy-grouped,
validated) and the analytic point that classical occupancy is degree-determined. What
does not: any claim that portals matter to walkers on a geometry.

**Follow-up, same night: the localization is in the eigenvectors.** Inverse
participation ratio of the walk Hamiltonian's eigenvectors, N = 1500, seed 0
(`N x IPR` = 1 for a state spread over the whole graph, N for a state on one node;
"fraction localized" = eigenvectors with IPR > 0.05, i.e. living on fewer than 20 nodes):

| graph | H = A: N x median IPR | fraction localized | H = L: N x median IPR | fraction localized |
|---|---|---|---|---|
| triangular | 2.6 | 0.000 | 3.0 | 0.000 |
| sheet | 2.9 | 0.000 | 3.2 | 0.000 |
| random k = 6 | 3.4 | 0.000 | 5.4 | 0.043 |
| pruned WS p = 0.05 | 26.8 | 0.106 | 34.1 | 0.173 |
| grown cap 6 | 39.7 | 0.278 | 68.0 | 0.473 |
| grown cap 7 | 42.1 | 0.283 | 102.0 | 0.589 |
| grown cap 8 | 45.0 | 0.307 | 154.3 | 0.700 |

On the lattice, the sheet and even the random graph every eigenvector is spread over the
graph. On `grown` a quarter to two thirds of them live on fewer than 20 nodes, more so
under the Laplacian and more so at higher cap -- which is the order of the banked portal
gains (x48 adjacency, x131 Laplacian). A tree-like graph with dangling structure
localizes its walk eigenstates; that, and not the portal, was the physics.

### Summary against the pre-registration

| prediction | triangular | sheet | grown (reference) |
|---|---|---|---|
| P1 threshold, advantage-blind | PASS | PASS | PASS |
| P2 self-stabilization survives | PASS (weaker) | PASS (weaker) | PASS |
| portals destroy, not inflate; k* in 10-50 | PASS, k* = 20 | PASS, k* = 20 | unstable at k = 0 |
| quantum Pbar gain > 10x | **FAIL** (1.0x) | **FAIL** (1.0x) | 122-166x |
| classical gain larger on 2D | **FAIL** (1.4x vs 6.7x) | **FAIL** (1.3x) | -- |

`triangular` and `sheet` agree to two decimals on everything, which at these sizes is a
consistency check (the sheet is a lattice patch with a rough edge), not two substrates.

Reproduce: `python portals_on_geometry.py --e1 --jobs 6`, `--e2 --jobs 6`;
`python shortcut_walkers.py --topology triangular --nodes 1500 --seeds 10 --generators
adjacency laplacian` (and `sheet`, `grown`).

## `sheet` at beta = 2 turns tree-like near 800,000 nodes (2026-09-29): the crossover is confirmed, and it is a sudden event

*Step 2 of the plan. Pre-registration
`docs/superpowers/specs/2026-09-28-sheet-crossover-4M-preregistration.md` (committed
before the growth); driver `sheet_crossover.py` (`--validate` passes). Three crossover
estimators on the same growths, per the owner's instruction to run every method and
compare.*

**Part B first (the law, on 12 seeds).** Re-scanning beta = 1.0-1.7 with 8 fresh seeds
makes N*(beta) monotone (7.3k, 11k, 14k, 25k, 34k, 117k, 122k, 324k) and gives

| law | fit | AICc |
|---|---|---|
| exponential (crossover) | ln N* = 3.30 + 5.38 beta | -16.3 |
| power law (transition) | ln N* = 10.00 - 4.76 ln(2.270 - beta) | -14.4 |

still a near tie on the finite points, and a prediction for beta = 2 of **N* = 1.26
million, 95% prediction interval [0.47 M, 3.4 M]** (the step-1 figure of 1.6 M moved by
less than a factor 3, as the pre-registration required before reading Part A).

**Part A (the test): beta = 2, four seeds, grown to 4,194,304 nodes.** Seed-mean local
exponents by size (compact: 0.50 / 0.50 / -0.50; tree-like: 1 / ~0.1 / ~0):

| N (midpoint) | boundary e_B | diameter e_D | arm fraction e_A |
|---|---|---|---|
| 1.4e3 ... 3.7e5 | 0.49 - 0.52 | 0.49 - 0.53 | -0.32 → -0.48 |
| 741,455 | **0.71** | 0.49 | **-0.28** |
| 1,482,910 | **1.18** | 0.40 | **+0.18** |
| 2,965,821 | 1.24 | **0.25** | 0.24 |

| method | crossover size N* |
|---|---|
| M1 boundary exponent | 782,000 |
| M3 arm fraction | 777,000 |
| M2 diameter exponent | 2,340,000 |

**Verdict under the frozen rules: CROSSOVER CONFIRMED, quantitatively.** All three
estimators are finite and inside the window; the boundary and arm-fraction readings agree
to 1% and the diameter reading lags them by a factor 3 (within the frozen factor 4); and
N*_B = 782,000 lies inside the 12-seed law's prediction interval. The rival transition at
beta_c ~ 2.3 is out: beta = 2 is not compact at every size.

**Per seed it is an event, not a drift.** Boundary over sqrt(N), which is 3.75-3.85 for a
compact disc at every size up to 512,000, then does this:

| seed | 512 k | 1 M | 2 M | 4 M | N*_B |
|---|---|---|---|---|---|
| 300 | 3.85 | 3.99 | 7.05 | 11.66 | 886,000 |
| 301 | 3.86 | 6.17 | 9.16 | 12.16 | 475,000 |
| 302 | 3.84 | 3.85 | 7.09 | 12.75 | 900,000 |
| 303 | 3.83 | 3.84 | 5.22 | 11.09 | 1,094,000 |

Each seed stays exactly compact and then, within one octave, its boundary exponent jumps
from 0.5 to above 1 (0.55 → 1.32 for seed 300). The crossover size varies by a factor 2.3
across four seeds. That is the signature of a nucleation: one protrusion escapes the
filling-in and, once it is an arm, grows like a tree; the diameter reacts a factor 3
later because an arm has to become long before it dominates the diameter. Why a
protrusion escapes at a size that grows like exp(5.4 beta) is not explained here.

**What `sheet` is, final form.** *[Revised 2026-10-17, wave 1 W1-c: there are no tree-like arms. Past its first dead corner the sheet stays two-dimensional wherever a ball fits inside it and its outer annulus ruffles (excess area, up to 4x a flat disc's mass within a given distance of the centre; the centre's eccentricity grows as N^0.23-0.30). Read 'tree-like' below as 'ruffled'.]* *[Revised 2026-10-13, item 4: twelve seeds put the beta = 2 crossover anywhere from 4e4 to beyond 4.2e6 nodes, consistent with a constant rate per node; the 0.8 M figure is a four-seed mean curve. Quote the distribution, not a size.]* A strictly local growth rule that produces a
window-stable two-dimensional sheet up to a size of about exp(3.3 + 5.4 beta) nodes --
0.8 M at beta = 2, 0.2 G at beta = 3 -- and grows tree-like arms beyond it. Below that
size four observables (ball growth, walk, diameter, boundary) agree it is a plane; above
it, `grown`'s fate. Every use of `sheet` as a substrate carries that size. At beta = 3 and
N <= 1e6 (everything in this log) the margin is a factor 200.

**Supplementary M4 / M5 on the 4 M-node seed-300 graph**, which see scales far below the
crossover: the window audit reads d = 1.94-1.99 at every window from radius 6 to 160 and
1.94, 1.84 at radii 250 and 400 (drift 0.15, just over its 0.15 tolerance: MIXED), and the
spectral flow reads FLAT at d_s = 1.96 with a slow decline from 2.00 to 1.85 over t = 16
to 11,585. Both say what they should: a plane wherever they look, with the arms just
becoming visible at the largest radii and longest times.

Reproduce: `python sheet_crossover.py --rescan --jobs 12 --old-scan
results/sheet_transition_20260928_133040.csv`; `python sheet_crossover.py --seeds 300 301
302 303 --n-max 4194304 --jobs 4` (~1.5 h per seed, ~5 GB each); `--analyze` on the CSVs.

## Preservation of a real dimension under the rules (2026-09-29)

*Sidequest; pre-registration
`docs/superpowers/specs/2026-09-29-preservation-on-geometry-preregistration.md`
(committed before the runs). `track_dimension.py` on `triangular` and `sheet`,
N = 10000, 200 steps, max_radius 10, seeds 0-2. `track_dimension` now reports
`median_drift` and `window_stable` at every checkpoint.*

The banked "Preservation" table was measured on `grown`, which has no dimension. On a
substrate that has one, all four predictions hold on both substrates:

| rule | defined_frac, t = 0 → 200 | median d_eff | window-stable | prediction | reading |
|---|---|---|---|---|---|
| `majority` | 0.98 → 0.98 | 1.96 → 1.96 | stable → stable | preserved | PASS |
| `prune` | 0.98 → 0.98 | 1.96 → 1.96 | stable → stable; edge count unchanged (29601 → 29601, 29625 → 29625) | preserved, exactly inert | PASS |
| `triadic` | 0.98 → 0.02 | 1.96 → 6.8 (triangular), 5.5 (sheet) | unstable from step 20 (all 6 runs) | eroded, unstable by 200, defined down > 0.3 | PASS |
| `rewire` | 0.98 → 0.00 | 1.96 → -- | untestable from step 20: no node keeps eight unsaturated radii | destroyed within 40 steps | PASS on destruction; the "reads UNSTABLE" clause cannot be read, because there is nothing left to test |

(Triangular and sheet numbers are identical to two decimals; one column shown.)

`triadic` is worth a sentence. It does not merely thin the field: within 20 steps the
median drift turns *negative* (-0.13, then -0.25, -0.33 by step 60 -- balls growing
slower than a power law at large radius, the fabric crumpling locally) and by step 200 it
is +0.5 with a median `d_eff` of 6-7 among the 2% of nodes still passing: the plane has
become a small world. The banked "crumpling" reading was right for the wrong graph; the
same rule crumples a real plane faster than it crumpled the tree (grown: 0.99 → ~0.46 in
200 steps).

So the banked reading stands, restated on a substrate where it means something: a real
dimension is a fixed point under the rules that respect triangles (`prune` is exactly
inert on a triangulation; state rules never touch topology) and is destroyed by the two
rules that move edges across the plane -- `rewire` within 20 steps, `triadic` within 20
steps as well, just less completely.

Reproduce: `python track_dimension.py --topology triangular --nodes 10000 --rules triadic
--steps 200 --track-interval 20 --max-radius 10 --seed 0` (and `sheet`; rules `prune`,
`majority`, `rewire`).

## Round of 2026-10-13, item 3: the banked tables under the window-stability gate

*Pre-registration `docs/superpowers/specs/2026-10-13-round2-preregistrations.md`. Same
drivers and defaults as the banked runs; only the estimator changed.*

**Coherence** (N = 5000, cap 6, 3 seeds, 999 permutations):

| topology | defined_frac | edge_frac | d_eff | Moran I | z | verdict |
|---|---|---|---|---|---|---|
| `grown` | 0.27 (was 0.99) | 0.14 | 2.15 | (1.10) | 42 | **field fragmented (N/A)** (was COHERENT, I = 0.88) |
| `lattice` | 0.99 | 0.99 | 1.89 | 0.932 | 93 | COHERENT (unchanged) |
| `small_world` | 0.12 | 0.07 | 4.69 | (0.91) | 30 | field fragmented (unchanged) |

Predicted and found. The banked "`grown`'s dimension field is spatially coherent" was a
statement about a fixed-window reading; the refusal logic in `coherence.py` now declines
the verdict on its own.

**Pruned Watts-Strogatz d(p)** (N = 2000 / 8000 / 32000, 3 seeds, max_radius 10; last
column at N = 32000):

| p | d_eff (N = 2000 / 8000 / 32000) | clustering | defined_frac | median drift | window-stable |
|---|---|---|---|---|---|
| 0.05 | 0.99 / 0.99 / 0.99 | 0.58 | 0.99 | +0.00 | 3/3 |
| 0.10 | 0.98 / 0.97 / 0.98 | 0.57 | 0.89 | +0.02 | 3/3 |
| 0.15 | 0.98 / 0.96 / 0.97 | 0.56 | 0.72 | +0.02 | 3/3 |
| 0.20 | 1.03 / 0.97 / 0.98 | 0.57 | 0.56 | +0.06 | 2/3 |
| 0.30 | 1.44 / 1.32 / 1.35 | 0.55 | 0.38 | +0.25 | 0/3 |
| 0.40 | 1.94 / 1.67 / 1.71 | 0.47 | 0.25 | +0.41 | 0/3 |
| 0.50 | 2.12 / 2.12 / 2.01 | 0.34 | 0.20 | +0.50 | 0/3 |
| 0.60-0.80 | 2.0 - 2.2 | 0.21 → 0.03 | 0.22 - 0.24 | +0.47 - 0.51 | 0/3 |

Predicted and found: window-stable with d = 1 for p <= 0.15, unstable from p = 0.3 on. Two
things the gated table says that the banked one could not. First, the "1 → 2 knob" is now
visibly a survivor statistic: at p >= 0.4 only a fifth of nodes pass, and their median of
~2 is the reading of a fit window on a small world. Second, `defined_frac` starts falling
(0.99 → 0.56) across p = 0.05-0.2 while the field is still window-stable at d = 1: the
first shortcuts to survive pruning cost individual nodes their power law one at a time
before the whole field turns over near p = 0.25. The banked "crossover, not a transition"
numbers (N-spread 0.12, peak slope 5.0 / 4.5 / 3.7 not growing with N) still hold.

Reproduce: `python coherence.py`; `python prune_dimension.py`.

## Round of 2026-10-13, item 1: there is no arm -- the sheet's edge roughens

*Pre-registration `docs/superpowers/specs/2026-10-13-round2-preregistrations.md` (item 1);
tool `arm_birth.py` (`--validate` passes, including a planted strip).*

The step-2 picture -- "one protrusion escapes and grows like a tree" -- was wrong. By the
frozen definition (a connected run of >= 50 boundary nodes farther than 1.5 R from the
root) **none of the four 4 M-node sheets has an arm at any size**, so the frozen DEFECT /
NOT A DEFECT readings cannot be applied; the item's pre-registered verdict is UNDECIDED
by construction, and the exploratory measurements below are what replaced it.

What the boundary does instead (seeds 300 and 301; R = median root distance of the
boundary):

| N | boundary edges | 2N / B ("finger width") | root-distance of boundary, q5 / q95 in units of R | share beyond 1.2 R | share inside 0.8 R | max depth from the edge |
|---|---|---|---|---|---|---|
| 524,288 | 2,660 | 394 | 0.88 / 1.08 | 0.000 | 0.000 | 365 |
| 1,048,576 | 3,883 | 540 | 0.88 / 1.08 | 0.000 | 0.000 | 512 |
| 2,097,152 | 9,717 | 432 | 0.81 / 1.20 | 0.052 | 0.035 | 602 |
| 4,194,304 | 22,684 | 370 | 0.75 / 1.22 | 0.078 | 0.098 | 675 |

(seed 301 reads the same: q5 / q95 0.73 / 1.17 at 4 M, 11% of the boundary inside 0.8 R.)

Three things follow. The disc keeps a deep core -- the maximum distance from the edge
keeps growing, 365 → 675 -- so no thin arms are draining the growth. The extra boundary is
**roughness**: fjords cutting inward to 0.75 R and bumps reaching 1.22 R, on a tenth of the
edge each. And boundary length proportional to area with a persistent length scale
2N / B ~ 350-400 nodes says the edge is becoming a fractal with a fixed smallest feature
size -- a **shape instability** of the growing disc, not a nucleation. Growth that favours
high coordination acts like surface tension; a circle growing under a surface tension
goes unstable once its radius exceeds a critical radius set by that tension (the
Mullins-Sekerka picture), which is exactly a crossover size N* = pi R_c^2 that grows with
beta, and which would make N* ~ exp(5.4 beta) a statement about R_c ~ exp(2.7 beta).

*[Revised 2026-10-14 (follow-up section "the crossover's trigger is a dead corner"): the
trigger is a boundary node the growth rule can never close, not a surface-tension
instability, and the roughness spectrum is near scale-free. The measured facts below stand;
the Mullins-Sekerka reading is withdrawn.]* That is an interpretation, not a result. The measured facts are: no arm; roughness
amplitude ~0.25 R; feature size ~370 nodes; core intact. A test would be the roughness
spectrum of the edge against R (the instability should appear at a fixed wavelength, the
feature size, and R_c should scale with the same width at other beta). The twelve-seed
crossover-size distribution (item 4) is also read against this picture: an instability at
a critical radius predicts a *narrow* N* distribution, a nucleation a broad one.

Reproduce: `python arm_birth.py results/sheet_cross_b2_s30{0,1,2,3}_N4194304.npz`.

## Round of 2026-10-13, item 4: the crossover size is broad -- two decades -- and consistent with a constant rate per node

*Pre-registration `docs/superpowers/specs/2026-10-13-round2-preregistrations.md` (item 4);
driver `sheet_crossover.py --seeds 304..311 --n-max 4194304` (run on jaga, 8 workers, ~50 min
per seed); reducer `crossover_cv.py` (`--validate` passes; reproduces the four step-2 values).*

Twelve seeds at beta = 2 grown to 4,194,304 nodes. Boundary over sqrt(N) (3.8 for a compact
disc) and the boundary-exponent crossover N*_B by the step-2 estimator (e_B crosses 0.75
going up and stays above):

| seed | 512 k | 1 M | 2 M | 4 M | e_B last octave | N*_B |
|---|---|---|---|---|---|---|
| 301 | 3.86 | 6.17 | 9.16 | 12.16 | 0.91 | 475,000 |
| 300 | 3.85 | 3.99 | 7.05 | 11.66 | 1.23 | 886,000 |
| 302 | 3.84 | 3.85 | 7.09 | 12.75 | 1.35 | 900,000 |
| 303 | 3.83 | 3.84 | 5.22 | 11.09 | 1.59 | 1,094,000 |
| 304 | 3.83 | 3.84 | 3.86 | 7.86 | 1.53 | 1,751,000 |
| 307 | 3.82 | 3.83 | 3.86 | 7.55 | 1.47 | 1,765,000 |
| 306 | 3.82 | 3.83 | 3.84 | 6.20 | 1.19 | 1,901,000 |
| 311 | 3.84 | 3.84 | 3.86 | 4.57 | 0.75 | censored (turning at 4 M) |
| 305 | 3.82 | 3.84 | 3.85 | 3.86 | 0.50 | censored (compact) |
| 309 | 3.82 | 3.84 | 3.84 | 3.85 | 0.50 | censored (compact) |
| 310 | 3.83 | 3.85 | 3.85 | 3.86 | 0.51 | censored (compact) |
| **308** | **9.00** | **10.47** | **12.02** | **13.74** | 0.69 | censored *by the rule* -- left at ~41,000 |

**Frozen reading: IN BETWEEN** (7 finite values, CV = 0.44, bootstrap SE 0.10; censored
fraction 0.42). The label matches the pre-registered prediction (IN BETWEEN, CV 0.3-0.6),
but it should not be read as support for the prediction's *mechanism* (a hazard rising with
boundary length), because the frozen rule did not anticipate censoring on this scale and the
CV of the uncensored values is biased low by it. Two things the table shows directly:

- **Seed 308 is the earliest crossover of all, and the frozen estimator misses it.** Its
  boundary leaves the compact curve between 33 k and 66 k nodes (B / sqrt(N) 3.9 → 4.8 → 6.1
  → 7.5 → 9.0 by 512 k) and its exponent then settles at 0.69-0.72, just *under* the 0.75
  line for the last three octaves, so the "stays above" clause fails. Read by eye it crossed
  at ~41,000 nodes; its graph is as manifold as the others (interior degree-6 fraction 1.0,
  no non-manifold edges), and its diameter at 4 M (1,209) is the smallest of the twelve.
- **Three seeds are exactly compact at 4.19 M** (3.85-3.86, exponents 0.50-0.51 to the last
  step) and a fourth is just turning. So the crossover size spans from 4e4 to beyond 4e6:
  **two decades across twelve seeds.** A sharp threshold is excluded outright.

**Censored-data analysis (exploratory, not pre-registered).** Right-censored Weibull maximum
likelihood on the twelve seeds (shape k = 1 is memoryless in N; k > 1 a rising hazard):

| treatment of seed 308 | shape k [95% profile CI] | scale | median N* | P(no crossover by 4.19 M) | LR test vs k = 1 |
|---|---|---|---|---|---|
| censored, as the frozen rule says | 1.02 [0.49, 1.78] | 4.2 M | 3.0 M | 0.37 | p = 0.96 |
| interval-censored in (0, 512 k] | 0.91 [0.43, 1.56] | 3.3 M | 2.2 M | 0.29 | p = 0.75 |
| at its visible crossing, 41 k | 0.80 [0.40, 1.37] | 3.3 M | 2.1 M | 0.30 | p = 0.45 |

A constant hazard per added node -- one crossover event per ~3-4 M nodes grown at beta = 2 --
fits as well as anything; a rising hazard (k > 1) is not favoured under any treatment, and
the observed 4-5 censored of 12 is what k = 1 predicts (0.3-0.37). The twelve-seed *mean*
curves read N*_B = 1.27 M and N*_A = 1.25 M (M2 censored), numerically on top of the Part B
prediction of 1.26 M; the eight new seeds alone read 2.18 / 2.17 M. Mean curves are dominated
by the earliest seeds and should not be quoted as "the" crossover size any more.

**What this changes.** The step-2 statement "`sheet` is 2D up to ~exp(3.3 + 5.4 beta) nodes"
described the mean curve of four seeds; the honest statement is distributional: at beta = 2
the crossover is a **rare event with a roughly constant rate**, so a given sheet has about a
30% chance of still being a plane at 4 M nodes and a few percent chance of leaving before
100 k. The item-1 reading of the *shape* (roughening at a fixed feature size, no arms) stands;
what item 4 adds is that the onset of that roughening is triggered stochastically, not at a
deterministic critical radius -- a critical radius would give a narrow N*. Whatever the
trigger is, it is a rare local event *[identified 2026-10-14: a dead corner of the rule -- see
the follow-up section]*. The exp(5.4 beta) law for N*(beta) was fitted on means
of 4-12 seeds per beta and now reads as the beta-dependence of a *rate*; its form is not
re-tested here.

**No arm on any of the eight new graphs either** (`arm_birth.py`, exploratory, run on the same
saved graphs): by the item-1 definition no boundary run of >= 50 nodes lies beyond 1.5 R at any
prefix size, including seed 308, whose boundary is 3.6x the compact value at 4 M. Whatever
leaves at 41 k and whatever leaves at 2 M produce the same thing -- a rough edge, not an arm.

Reproduce: `python crossover_cv.py results/sheet_crossover_partA_N4194304_*.csv`;
`python sheet_crossover.py --analyze` on the same files for the mean-curve estimators.
Graphs saved as `results/sheet_cross_b2_s30{4..9}_N4194304.npz`, `s310`, `s311` (42 MB each).

## Round of 2026-10-13, item 2: the throat threshold is the same ~2.6-strand motif on a real plane -- bigger throats, no sharper

*[Revised 2026-10-17, wave 1 W1-a: the 2.6 is an estimator artifact (mean(core_frac) x median(A*)); the per-draw core at onset is exactly 2 strands in 91-98% of draws on every substrate. The motif-onset verdict stands; the number is withdrawn.]*

*Pre-registration `docs/superpowers/specs/2026-10-13-round2-preregistrations.md` (item 2);
driver `throat_criticality.py --topology {triangular, sheet}` (`--fss-r` added this round so
the three FSS geometries could run as separate processes); reducer `throat_compare.py`
(`--validate` passes; reproduces the stage-1 `grown` numbers from the saved CSVs). Pilots
and the triangular r = 2 / 3 FSS ran on the desktop; triangular r = 4, the whole `sheet`
FSS and both anchor runs ran on jaga after the desktop runs were killed for memory (the
first attempt is disclosed in the pre-registration's status note). Anchor a-values were
frozen from the pilots before the FSS and anchor runs.*

Pilots (300 draws, N = 2000, r = 2): onset core 2.62 strands on `triangular`, 2.40 on
`sheet`, against 2.6 banked on `grown`. Then the stage-1 geometries, 2000 draws each:

| substrate | r | capacity (strands) | A*50 | a50 | relative width w / a50 (+/- boot) | core_frac @ A* | onset core (strands) | finite | regens |
|---|---|---|---|---|---|---|---|---|---|
| `grown` (banked 2026-08-17) | 2 | 134 | 6 | 0.0486 | 1.473 (0.050) | 0.435 | **2.61** | 2000/2000 | 19 |
| | 3 | 425 | 9 | 0.0253 | 1.574 (0.038) | 0.288 | **2.59** | 2000/2000 | 19 |
| | 4 | 1043 | 13 | 0.0155 | 1.655 (0.050) | 0.204 | **2.65** | 2000/2000 | 19 |
| `triangular` | 2 | 329 | 8 | 0.0249 | 1.315 (0.034) | 0.310 | **2.48** | 2000/2000 | 0 |
| | 3 | 1264 | 14 | 0.0117 | 1.312 (0.034) | 0.185 | **2.68** | 2000/2000 | 0 |
| | 4 | 3448 | 22 | 0.0064 | 1.416 (0.035) | 0.121 | **2.66** | 2000/2000 | 0 |
| `sheet` (beta = 3) | 2 | 330 | 8 | 0.0249 | 1.315 (0.030) | 0.313 | **2.50** | 2000/2000 | 0 |
| | 3 | 1265 | 14 | 0.0110 | 1.339 (0.043) | 0.186 | **2.61** | 2000/2000 | 0 |
| | 4 | 3458 | 23 | 0.0067 | 1.354 (0.037) | 0.121 | **2.77** | 2000/2000 | 0 |

**Frozen readings: `triangular` MOTIF ONSET HOLDS; `sheet` MOTIF ONSET HOLDS** (`grown`
re-reads MOTIF ONSET HOLDS under the same rule). Both pre-registered bullets held on both
planes: the onset core is inside [2, 3.5] at every capacity, and the relative width does
not fall with capacity (every step is a rise or flat within 10%). Predicted and found.

What the plane changes, and what it does not:

- **Capacity, not the motif.** A ball of radius r on a plane holds 2.5-3.3x more strands than
  the same-radius ball of a `grown` tree (329 / 1264 / 3448 against 134 / 425 / 1043), and the
  median threshold in strands grows with it (8 / 14 / 22 against 6 / 9 / 13; on both
  substrates A*50 ~ capacity^0.4, so the critical *density* a50 falls). The core at onset
  does not move: 2.48-2.68 strands on `triangular`, 2.50-2.77 on `sheet`,
  2.59-2.65 on `grown`. The threshold is the first appearance of one mutually-protecting
  motif of ~2.6 strands whatever the ball looks like -- the stage-1 reading was a statement
  about the peeling rule, not about `grown`, and now it is shown to be.
- **The plane's threshold is better defined but still does not sharpen.** The relative width
  is 1.31-1.42 on the planes against 1.47-1.66 on `grown`, and it is flat then rising with
  capacity (1.315 / 1.312 / 1.416 on `triangular`, 1.315 / 1.339 / 1.354 on `sheet`) exactly as on `grown` -- a 20% tighter spread, not a width that
  shrinks toward a critical point. In absolute strands the 10-90 span grows with capacity
  on all three substrates.
- **`triangular` and `sheet` agree to two decimals** at r = 2 (330 vs 329 strands, a50
  0.0249 on both, relative width 1.315 on both) and to a few percent at r = 3 and 4, as they
  did throughout step 3: at N <= 10,000 a beta = 3 sheet is a triangular lattice with a
  rough edge, far below its own crossover size (~exp(3.3 + 5.4 x 3) ~ 2e8 nodes).
- **No substrate regenerations** on either plane (0 / 2000 at every r, 0 / 40 anchor runs);
  the 19 / 2000 regenerations banked on `grown` were its expander phase, absent here.

**Gates (anchors: 5 frozen a-values x 8 seeds, N = 2000, 400 sweeps, with `--rider`).**
Gate 1 (peeling == dynamics) PASSES exactly on both planes: 0 missing / unattributed strands
across 40 runs each, `survivors == core` in every run, as on `grown`. Gate 2 reads (b) on
both, as on `grown`: under triadic + prune the cores are demolished (mean core retention 0.08
on `triangular`, 0.02 on `sheet`; `grown` ~0.1) and the sub-threshold "rescue" is 1-2
strands surviving in 5 of 24 (`triangular`) and 7 of 26 (`sheet`) empty-core runs, the same
0-1-strand scale banked on `grown`. Mean prune-only cores at the frozen a-values
(pilot q10 / q30 / q50 / q70 / q90): 0 / 0.25 / 1.38 / 2.00 / 4.88 on `triangular`,
0 / 0 / 0.88 / 1.75 / 4.75 on `sheet`. Throat cores are censor-proof but not churn-proof on a
plane too.

**What this changes.** This was the last portal-program result that rested on `grown` balls
alone. With it, every banked portal statement has now been re-run on a substrate with a
real dimension, and the throat one survives unchanged: no critical collapse in the
censor-only family, a local motif onset, Gate 1 exact, cores demolished by the stabilizer.
The quantities that *are* substrate-dependent -- capacity, A* in strands, the relative width
-- all moved in the direction of "a plane is a bigger, more regular container for the same
motif". BRANCHES row "Throat criticality on a substrate with a dimension" closes.

Reproduce: `python throat_compare.py` (reads the newest `results/throat_fss_*.csv`);
`python throat_criticality.py --topology triangular --fss --fss-r 4` etc. for the rows; the
anchor tables are in the run logs (`results/jaga_logs_20261013/throat_{tri,sheet}_anchor.txt`),
not in CSVs.

### The round, compared

Per the pre-registration's last section: for each item, prediction, outcome, and what it
changes.

| item | pre-registered prediction | outcome | what it changes |
|---|---|---|---|
| 1 -- what nucleates the arm | DEFECT (60%): a blocked closure at the arm's base | **UNDECIDED by construction**: no arm exists on any of 12 graphs; the edge roughens instead (fjords to 0.75 R, bumps to 1.22 R, feature size ~370 nodes, core intact) | the step-2 "arm" picture is withdrawn and `arm_birth.py` is its instrument; the "shape instability" reading given here was itself superseded the same night by the dead-corner trigger (follow-up section) |
| 2 -- throat criticality on a plane | MOTIF ONSET HOLDS on both planes; larger capacities; gates hold | **HELD on `triangular`; HELD on `sheet`**: onset core 2.5-2.8 strands at every capacity (*revised 2026-10-17: 2 strands per draw; the 2.5-2.8 was an artifact*), relative width 1.31-1.42 flat-to-rising, capacities 2.5-3.3x grown, Gate 1 40/40 exact, Gate 2 (b) | the throat verdict is substrate-independent; the whole portal program now stands on a real geometry |
| 3 -- banked tables under the gate | `grown` fragmented (~0.27); d(p) stable at d = 1 for p <= 0.1, unstable for p >= 0.3 | **all held**: defined_frac 0.27; window-stable to p = 0.15, unstable from 0.3; the "1 -> 2 knob" is a survivor statistic | two banked tables replaced under supersession notices |
| 4 -- crossover-size distribution | IN BETWEEN, CV 0.3-0.6 (rising hazard) | **IN BETWEEN by the frozen rule (CV 0.44)** but 5 / 12 censored and N* spanning 4e4 to > 4e6; right-censored Weibull k = 0.8-1.0, CI covers 1: constant rate per node, sharp threshold excluded | `sheet`'s 2D range at a given beta is a distribution, not a size: at beta = 2 about one event per 3-4 M nodes grown; the mean-curve "0.8 M" is retired as a headline |

Two of the four predictions held as written (items 2 and 3), one held by label but not by
mechanism (item 4), and one could not be read because its object does not exist (item 1).
The two negative-leaning outcomes are the informative ones: together they say the sheet's
crossover is a stochastic roughening with a constant rate, which is neither the nucleated
arm of step 2 nor the deterministic critical radius that item 1's shape reading suggested.
The round also moved most of its compute to jaga mid-way after the desktop was out of memory;
the pre-registration's "jaga is not required" line was wrong, and the first-attempt kills
are disclosed there.

## Round of 2026-10-13, follow-up: the crossover's trigger is a dead corner -- a boundary node the rule can never close

*Two pre-registrations the same night: `docs/superpowers/specs/2026-10-13-edge-spectrum-preregistration.md`
(tool `edge_spectrum.py`, `--validate` passes: planted wavelength recovered, 0 / 10 false peaks on
periodic random walks) and `docs/superpowers/specs/2026-10-13-blocked-closure-preregistration.md`
(tool `blocked_closure.py`, `--validate` passes: a planted dead corner is found at the exact
prefix size). The second grew out of the first's onset scan; its spec says which parts are post
hoc and which are the test. Everything ran on the desktop on the twelve saved beta = 2 graphs
plus eight fresh beta = 1.5 seeds.*

**1. The roughness spectrum (pre-registered; frozen reading UNDECIDED, and the instrument has a
fault).** Height profile h(s) = rho(s) - R along the boundary cycle (covered fraction 1.00 at
every size on every seed), periodogram in log bins of >= 8 modes, power-law fit, peak = bin
excess >= 3.0 over the fit. Compact edges (B / sqrt(N) < 4.5) are smooth: slopes -3.4 to -4.7.
Rough edges have slopes -2.0 to -2.8 and peaks. **Frozen verdict: UNDECIDED (fixed 2,
scale-free 0 of 5 counting seeds).** The fault: the three seeds that never cross (305, 309,
310) also show "peaks", at lambda = 389 / 551 / 781 / 1110 boundary nodes at 512 k / 1 M / 2 M /
4 M -- i.e. at k = L / lambda ~ 7 at every size, excess 3.7-8.6. That is the disc's six-fold
anisotropy on a triangular lattice (a hexagon has power at k = 6 and its neighbours), a global
shape mode, not roughness; the frozen rule excluded k = 1-3 (offset, ellipticity,
triangularity) and should have excluded the hexagon too. Post-hoc re-read with k >= 16
(flagged as such, `--k-min 16`): the compact seeds lose their "peaks" entirely (slopes -3.4 to
-4.1, no bin above 2.1x), which confirms the diagnosis; the rough edges keep slopes -2.1 to -3.1
and only modest peaks (excess 3.2-5.9 at lambda = 183-685 boundary nodes, no wavelength shared
across sizes within a seed except in 308), and the verdict is again UNDECIDED (fixed 1 of 5).
So the rough edge is closer to a scale-free random surface than to an instability at one
wavelength; the shape-instability reading of item 1 is not supported by the spectrum, and
what the next part finds makes it unnecessary. What survives the fault: **seed 308, the earliest
crossover, has a genuine roughness peak** -- lambda = 247-301 boundary nodes from 262 k to 2 M
under the frozen settings (k = 15-50, excess 4.8-11.6; 487 at 4 M, excess 17.5), and with
k >= 16 at 217 / 353 / 428 from 1 M to 4 M (excess 4.1 / 7.4 / 16.7) -- with slope -2.0 to -2.5
and sigma_h / R rising 0.19 -> 0.26. Its edge is a random-walk-like surface with one preferred
wavelength of a few hundred nodes that drifts slowly upward with N, of the order of item 1's
feature size 2N / B (lambda / (2N / B) = 1.1-1.8).

**2. The onset scans (pre-registered, descriptive).** Seed 308 at 2^(1/4) spacing from 16 k:
the largest excursion of the edge is always inward (a fjord, never a bump); its depth sits at
the compact baseline (-0.14 to -0.16 R) to 27.5 k, then deepens monotonically (-0.17 R at 33 k,
-0.24 at 39 k, -0.38 at 66 k, -0.50 at 131 k) while B / sqrt(N) rises 3.8 -> 6.1. **From 32,768
on the deepest point is the same node, 24862.** The census around it reads blocked closures
within 20 hops (density 0.04 -> 0.19 of boundary nodes) and **exactly zero on the rest of the
boundary** at every size. Seed 300 (N*_B 886 k) at 2^(1/8) spacing from 512 k: fjord depth
-0.135 to -0.147 R and zero blocked closures through 961,548; at 1,048,576 the deepest point is
the oldest boundary node (916282) with blocked closures around it and nowhere else.

**3. What a blocked closure is, in the rule's own terms.** In `sheet_growth` a boundary edge
whose endpoint has degree >= c can only fire a closure, and only if that endpoint is
*closable*: its two boundary neighbours a, b are not adjacent, share no neighbour but it, and
both still have degree < c. Degrees never fall and adjacency never shrinks, so a degree-c
boundary node that is not closable is **dead forever**: no event will ever fire on its edges,
it stays on the boundary while the disc grows past it, and it becomes the bottom of a fjord
whose walls keep growing. The ad-hoc scan that found it (seed 308, prefix sizes 24 k-524 k):

| N | B / sqrt(N) | boundary nodes of degree >= 6 | blocked | oldest boundary node | deepest fjord (h / R) at node |
|---|---|---|---|---|---|
| 24,000-25,500 | 3.77-3.78 | 0 | 0 | 22267-24156 | -0.14 to -0.16 (wandering) |
| 26,000 | 3.79 | 2 | 2 (24862, 25107, adjacent) | 24284 | -0.151 at 24658 |
| 27,000-30,000 | 3.81-3.85 | 2 | 2 | **24862** | -0.14 to -0.16 (wandering) |
| 32,768 | 3.92 | 3 | 3 | 24862 | **-0.171 at 24862** |
| 40,000 | 4.17 | 5 | 5 | 24862 | -0.243 at 24862 |
| 65,536 | 4.80 | 7 | 7 | 24862 | -0.379 at 24862 |
| 131,072 | 6.09 | 12 | 12 | 24862 | -0.503 at 24862 |
| 524,288 | 8.99 | 24 | 24 | 24862 | -0.653 at 24862 |

Every degree->= 6 boundary node in the graph is blocked, all of them are strung up the fjord
wall from node 24862 (hops 0, 1, 2, 3, ... from the bottom), and the two first ones appear
*before* the fjord deepens: dead corner at 26 k, oldest-boundary-node status at 27 k, deepest
point at 33 k, boundary exponent visibly up at 30-40 k. Seed 305 (compact to 4 M): zero at 32 k,
131 k, 524 k. Seed 300: zero at every size to 900 k, six at 1,048,576, all within 3 hops of node
916282.

**The twelve saved graphs (post hoc, no verdict):**

| seed | first dead corner at N (node ids) | N*_B (step-2 estimator) | octave in which B / sqrt(N) leaves 3.85 | dead corners at 4.19 M | clusters (10-hop) | oldest boundary node is a dead corner |
|---|---|---|---|---|---|---|
| 308 | 25,885 (24862, 25107) | censored by the rule; ~41 k by eye | 33 k - 66 k | 57 | 2 | yes |
| 301 | 495,183 (490066, 491027) | 474,877 | 512 k - 1 M | 58 | 3 | yes |
| 300 | 921,272 (916282, 917875) | 886,232 | 1 M - 2 M | 70 | 3 | yes |
| 302 | 1,056,569 (1052853, 1052862) | 899,866 | 1 M - 2 M | 81 | 3 | yes |
| 303 | 1,499,592 (1490903, 1494234) | 1,094,243 | 1 M - 2 M | 71 | 4 | yes |
| 307 | 2,062,098 (2055514, 2056616) | 1,764,590 | 2 M - 4 M | 39 | 2 | yes |
| 304 | 2,485,497 (2477931, 2478637) | 1,751,102 | 2 M - 4 M | 48 | 2 | yes |
| 306 | 2,846,999 (2842027, 2842398) | 1,900,546 | 2 M - 4 M | 36 | 2 | yes |
| 311 | 3,338,158 (3332769, 3332968) | censored (turning: 4.57 at 4 M) | 2 M - 4 M | 14 | 1 | yes |
| 305 | none to 4,194,304 | censored (compact, 3.86) | -- | 0 | 0 | no |
| 309 | none to 4,194,304 | censored (compact, 3.85) | -- | 0 | 0 | no |
| 310 | none to 4,194,304 | censored (compact, 3.86) | -- | 0 | 0 | no |

Twelve for twelve. Every seed whose boundary left the compact curve has a dead corner, and
its first dead corner lies inside the octave in which the boundary leaves the curve -- never
before it; the three seeds still compact at 4.19 M have **no dead corner anywhere**; seed 311,
which is just turning at 4 M, acquired its first at 3.34 M. In every crossed seed the first
dead corner is a pair of adjacent nodes born within a few thousand nodes of each other and
within ~1% of N_first (the two link ends reach the cap together), and that node is the
*oldest node still on the boundary* at 4.19 M -- the one spot where the disc stopped. By 4 M
there are 14-81 dead corners in 1-4 clusters: the first pit seeds further ones along its
walls, and a second or third pit forms elsewhere later. The step-2 N*_B sits 4-50% *before*
N_first in all nine: the estimator interpolates the 0.75-crossing between octave midpoints,
so inside a one-octave jump it lands early; the dead corner, not the interpolation, is where
the boundary actually leaves sqrt(N).

**4. The test on fresh seeds (pre-registered).** Eight new seeds at beta = 1.5 grown to
524,288 nodes with the blocked count and boundary length at 2^(1/8) spacing; N*_B by the
step-2 estimator on the octave checkpoints; frozen readings TRIGGER CONFIRMED (>= 3/4 of the
crossed seeds have a blocked node by N*_B and N*_B / N_first <= 4) / REFUTED / UNDECIDED, with
the uncrossed seeds as a control.

| seed | B / sqrt(N) at 524 k | N*_B | N_first (first blocked node) | N*_B / N_first | blocked nodes at 524 k |
|---|---|---|---|---|---|
| 407 | 15.48 | 12,865 | 5,312 | 2.42 | 85 |
| 400 | 10.86 | 52,705 | 38,968 | 1.35 | 62 |
| 403 | 13.41 | 56,369 | 55,109 | 1.02 | 82 |
| 405 | 14.14 | 62,091 | 55,109 | 1.13 | 100 |
| 401 | 10.03 | 96,026 | 84,990 | 1.13 | 51 |
| 402 | 8.04 | 149,595 | 155,872 | 0.96 | 44 |
| 404 | 8.20 | 157,409 | 155,872 | 1.01 | 39 |
| 406 | 7.46 | 219,324 | 240,387 | 0.91 | 36 |

**Frozen verdict: TRIGGER CONFIRMED (6 / 8 crossed seeds; no uncrossed seed to serve as
control -- all eight crossed, as the Part B law said they would at beta = 1.5).** Predicted
and found, and tighter than predicted: the first dead corner appears within a factor 2.5 of
N*_B in every seed, within 13% in six. The two "misses" under the frozen rule (402, 406) have
N_first 4-10% *after* the interpolated N*_B; the step-2 estimator interpolates the exponent's
0.75-crossing between octave midpoints, so its N*_B carries about half an octave of
resolution and can land just before the event itself (seed 300 at beta = 2 does the same:
N_first 921 k against N*_B 886 k). Read without the frozen inequality, 8 / 8 seeds have the
dead corner and the exponent jump at the same size. The number of dead corners at 524 k
(36-100) tracks the roughness: Spearman-wise, the seeds that crossed earliest have the most
and the largest B / sqrt(N).

**What this changes.** Items 1 and 4 of the round, and step 2 before them, were describing one
mechanism from three sides. The sheet's crossover is triggered by a **local, permanent defect
of the growth rule**: a boundary node reaches the degree cap while its two boundary neighbours
are already at the cap (or would pinch), the rule's own degree guard then forbids the closure
that would bury it, and growth stalls at that one spot forever. The disc grows around the
stalled point, which becomes the bottom of a deepening fjord; more corners die along the
fjord's walls; the boundary length stops scaling like sqrt(N) and the "two-dimensional range"
ends. That is item 1's DEFECT hypothesis, found at the bottom of a fjord rather than at the
base of an arm (there is no arm). It also says why item 4 saw a constant rate: the crossover
happens at the first dead corner, and the first occurrence of a rare local configuration in a
growing boundary is a Poisson-like event -- one per 3-4 M nodes grown at beta = 2. And it
reframes the exp(5.4 beta) law as the beta-dependence of the probability per boundary site of
producing a dead corner (higher beta favours high-coordination events, which close wheels
before they can die); that is a hypothesis, not yet a measurement.

A sharp prediction follows and goes to BRANCHES: a variant of the rule that cannot leave a dead
corner -- for instance allowing the closure when a link end is already at the cap (raising its
degree to c + 1), or choosing the cap so that a closure is always available -- should have **no
crossover at any beta**. If it does, the mechanism here is wrong.

Reproduce: `python edge_spectrum.py results/sheet_cross_b2_s3*_N4194304.npz` (frozen) and
`--k-min 16` (post hoc); `python edge_spectrum.py --onset results/sheet_cross_b2_s308_N4194304.npz
--n-min 16384 --n-max 131072 --per-octave 4`; `python blocked_closure.py
results/sheet_cross_b2_s3*_N4194304.npz --crossover`; `python blocked_closure.py --grow --beta
1.5 --seeds 400 407 --n-max 524288 --jobs 8`. CSVs under `results/edge_*.csv`,
`results/blocked_*.csv`; logs under `results/edge_logs/`.

## Literature paths, wave 1 (2026-10-17): the throat motif is two strands, every dead corner is the degree guard, and past N* the sheet stays flat locally but ruffles globally

*Pre-registration `docs/superpowers/specs/2026-10-17-wave1-preregistration.md` (committed
before any number below was computed); plan `2026-10-17-literature-paths-plan.md`; the
paths come from `reports/Local graph growth literature survey.md` (A1, A2, A3, A6, A7).
All four parts are complete. The fourth, the direct hazard scan, was killed by the
harness for low system memory after 14 of its 160 growths and was resumed on jaga on the
owner's go the same afternoon (see W1-d for the deviation).*

**Summary.**

| part | frozen reading | prediction | consequence |
|---|---|---|---|
| W1-a throat onset as integers | **INTEGER MIXTURE, FRACTION UNIVERSAL**; m = 2 in 91-98% of draws | held (75%, 50%); survey's 40/60 guess wrong | the banked "~2.6-strand onset core" was an estimator artifact; the onset is the two-strand triangle |
| W1-b curvature and failure modes | **DEGREE GUARD**, 9/9; all 474 dead corners fail (iv) only | held (75%, 65%) | the dead corner is the cap and nothing else; the edge's local make-up never changes |
| W1-c still a disc? | **DISC WITH A STALLED RADIUS** (audit 2D on 8/9, eccentricity exponent 0.23-0.30 after the defect) | missed (STILL A DISC, 60%) | past N* the sheet is locally flat and globally ruffled: excess area in the outer annulus, not a tree |
| W1-d direct hazard scan | **PER-NODE HAZARD** (pooled k 1.07 [0.95, 1.21]); **ARRHENIUS, RATE LAW DIFFERS** (slope 6.44 vs 5.38); **LOST RACE** 160/160 | held (65%), held / missed (70% / 55%), held (75%) | the first dead corner is a constant-rate-per-node event, lambda ~ 0.15 exp(-6.44 beta); the lost race is the only birth route in the default rule |

### W1-a. The throat onset is the two-strand triangle, not a 2.6-strand motif

*Reducer: `python throat_compare.py --integer` (validated). Inputs: the nine 2000-draw FSS
CSVs already on disk.*

The per-draw core size at onset, m = A_star x core_frac, is an integer, and it is almost
always the smallest possible core:

| substrate | r | P(m = 2) | P(m = 3) | P(m = 4) | P(m >= 5) | mean m |
|---|---|---|---|---|---|---|
| grown | 2 | 0.907 | 0.086 | 0.006 | 0 | 2.10 |
| grown | 3 | 0.937 | 0.059 | 0.004 | 0 | 2.07 |
| grown | 4 | 0.956 | 0.043 | 0.001 | 0 | 2.05 |
| triangular | 2 | 0.929 | 0.069 | 0.003 | 0 | 2.07 |
| triangular | 3 | 0.958 | 0.042 | 0.001 | 0 | 2.04 |
| triangular | 4 | 0.979 | 0.021 | 0 | 0 | 2.02 |
| sheet | 2 | 0.925 | 0.071 | 0.003 | 0 | 2.08 |
| sheet | 3 | 0.963 | 0.037 | 0 | 0 | 2.04 |
| sheet | 4 | 0.977 | 0.024 | 0 | 0 | 2.02 |

**Frozen readings: INTEGER MIXTURE** (P(2) + P(3) >= 0.994 in every cell) and **FRACTION
UNIVERSAL** (P(2) spans 0.907-0.979, inside the 0.15 band); mode 2 in 9 of 9 cells. The
instrument check passed (m >= 2 in every one of the 18,000 draws). Predictions: the
mixture and the universal fraction were predicted (75%, 50%) and mode 2 was predicted
(60%); the survey's guess of a 40/60 mixture of 2 and 3 was wrong, because it was built on
the banked mean of 2.6.

**The banked "~2.6-strand onset core" was an estimator artifact.** It was computed as
mean(core_frac) x median(A_star). Since core_frac = m / A_star and m is nearly always 2,
that product is about 2 x mean(1/A_star) x median(A_star), which exceeds 2 whenever
A_star is right-skewed, as it is (10/50/90% of A_star are 3/6/10 on grown r = 2 and 9/23/40
on sheet r = 4):

| cell | banked mean(cf) x median(A*) | 2 x mean(1/A*) x median(A*) | true mean m |
|---|---|---|---|
| grown r = 2 | 2.61 | 2.51 | 2.10 |
| triangular r = 4 | 2.66 | 2.64 | 2.02 |
| sheet r = 4 | 2.77 | 2.75 | 2.02 |

What the throat threshold is, corrected: the first moment a random prefix of strands
contains **two strands sharing an endpoint whose far ends are adjacent** (a triangle with one
base edge, the smallest set in which every strand has a common neighbour). That fixed point
is reached exactly at onset in 91-98% of draws, and the fraction rises toward 1 with
capacity on every substrate. The qualitative verdict (a local-motif onset, not a critical
point) stands and is sharper. The number 2.6 is withdrawn everywhere it was quoted.

### W1-b. Every dead corner is the degree guard, and the rough edge is locally identical to the compact one

*Tool: `python curvature_census.py results/sheet_cross_b2_s3*_N4194304.npz` (validated:
disc identities and dead-corner census agree with `blocked_closure` on three test sheets).
CSV `results/curvature_census_20261017_010558.csv`.*

**Instrument gate: PASS.** On all twelve final graphs B_v = B_e (one simple boundary
cycle), every interior node has degree 6, no edge is in zero or three triangles, and the
Gauss-Bonnet sum over boundary nodes of (4 - d) is exactly 6. The prefix artifact the
pre-registration worried about (closure edges among old nodes appearing early) did not
show at all: 0 of 138 prefix rows have an edge in no triangle.

**Frozen reading: DEGREE GUARD**, 9 of 9 crossed seeds. In every one, the first dead
corner is a pair of adjacent boundary nodes, both of degree 6, each failing condition (iv)
and nothing else:

| seed | N_first | first dead pair | dead corners at 4.19 M |
|---|---|---|---|
| 308 | 25,885 | 24862, 25107 | 57 |
| 301 | 495,183 | 490066, 491027 | 58 |
| 300 | 921,272 | 916282, 917875 | 70 |
| 302 | 1,056,569 | 1052853, 1052862 | 81 |
| 303 | 1,499,592 | 1490903, 1494234 | 71 |
| 307 | 2,062,098 | 2055514, 2056616 | 39 |
| 304 | 2,485,497 | 2477931, 2478637 | 48 |
| 306 | 2,846,999 | 2842027, 2842398 | 36 |
| 311 | 3,338,158 | 3332769, 3332968 | 14 |
| 305, 309, 310 | none | | 0 |

All **474 of 474** dead corners in the final graphs fail (iv) only (prediction >= 0.80).
No pinch, no other mode, ever. In curvature terms each dead corner is a boundary node of
kappa = -1/3 whose link end also has kappa = -1/3; the fill that would neutralise it needs
a degree-7 node, which the cap forbids.

**What did not change: the boundary's local make-up** (predicted to shift toward high
degree; it does not). The fraction of boundary nodes at degree 5 or 6 is 0.31-0.34 in
every seed at every size, compact or crossed, and the degree histogram per boundary node
is the same before and after the first dead corner (seed 300: 0.313 at 16 k, 0.333 at
N_first, 0.329 at 4.19 M with B 6.5 times longer). The negative-curvature mass M_- grows
in proportion to B (about 0.055 per boundary node throughout). So the roughened edge is,
node by node, the same edge as the compact one; only its global length and shape differ.
The crossover is not a change in what the boundary is made of, it is a change in how much
of it there is, driven by the permanent defects.

### W1-c. Past N* the sheet is flat everywhere locally but ruffled globally: DISC WITH A STALLED RADIUS

*Tool: `python curvature_census.py results/sheet_cross_b2_s3*_N4194304.npz --disc`
(validated: a compact 400 k beta = 3 sheet reads DIMENSION DEFINED at 1.96-1.99). CSV
`results/disc_audit_20261017_010926.csv`.*

**Large-radius window audit** (30 uniformly sampled sources per graph, windows to radius
480, used only by sources whose ball is below 0.1 N):

| seed | state at 4.19 M | median d by window (R10 ... last usable) | drift | power vs exponential RSS | audit |
|---|---|---|---|---|---|
| 300 | crossed | 1.96 1.98 1.99 1.99 1.94 1.93 1.90 (R320) | 0.09 | 2.08 vs 18.0 | DEFINED |
| 301 | crossed | 1.96 ... 1.99 1.95 1.89 (R320) | 0.10 | 1.30 vs 19.3 | DEFINED |
| 302 | crossed | 1.96 ... 1.99 1.96 1.91 (R320) | 0.08 | 0.12 vs 26.8 | DEFINED |
| 303 | crossed | 1.96 ... 1.99 1.99 1.96 (R320) | 0.03 | 1.17 vs 20.0 | DEFINED |
| 304 | crossed | 1.96 ... 2.00 2.00 (R320) | 0.04 | 0.11 vs 35.5 | DEFINED |
| 306 | crossed | 1.96 ... 2.00 2.00, 1.79 at R480 (7 sources) | 0.21 | 0.06 vs 36.5 | MIXED, d_last 1.79 |
| 307 | crossed | 1.96 ... 1.98 1.97 (R320) | 0.03 | 0.13 vs 31.8 | DEFINED |
| 308 | crossed | 1.96 ... 1.95 1.86 1.82 (R320) | 0.17 | 1.25 vs 16.6 | MIXED, d_last 1.82 |
| 311 | crossed | 1.96 ... 2.00 2.00 (R320) | 0.04 | 0.02 vs 34.0 | DEFINED |
| 305 | compact | 1.96 ... 2.00 2.00 (R320) | 0.04 | 0.06 vs 34.7 | DEFINED |
| 309 | compact | 1.96 ... 1.99, 1.77 at R480 (8 sources) | 0.22 | 0.06 vs 38.0 | MIXED, d_last 1.77 |
| 310 | compact | 1.96 ... 2.00 2.00 (R320) | 0.04 | 0.07 vs 35.6 | DEFINED |

Disc-like by the frozen rule on 8 of 9 crossed graphs (seed 306 misses on d_last = 1.79 at a
radius-480 window read by 7 sources). EXPONENTIAL on 0 of 9; the power law beats the
exponential by a factor 8-1400 in RSS on every graph. The radius-480 dip is an edge effect
of the instrument, not of the crossover: compact seed 309 shows the same dip (1.77 from 8
sources).

**Radius growth from the centre** (eccentricity of node 0; exponent over 2.1 M -> 4.2 M):

| seed | N_first | ecc at 1 M / 2 M / 4.2 M | last-octave exponent |
|---|---|---|---|
| 308 | 26 k | 402 / 497 / 611 | 0.30 |
| 301 | 495 k | 614 / 750 / 891 | 0.25 |
| 300 | 921 k | 650 / 853 / 1017 | 0.25 |
| 302 | 1.06 M | 649 / 865 / 1016 | 0.23 |
| 303 | 1.50 M | 652 / 907 / 1096 | 0.27 |
| 307 | 2.06 M | 649 / 920 / 1204 | 0.39 |
| 304 | 2.49 M | 650 / 921 / 1226 | 0.41 |
| 306 | 2.85 M | 649 / 920 / 1305 | 0.50 |
| 311 | 3.34 M | 647 / 917 / 1294 | 0.50 |
| 305, 309, 310 (compact) | none | 648 / 917 / 1303 | 0.50 |

Eccentricity exponent in [0.40, 0.65] on 3 of 9 crossed graphs, so **the frozen reading is
DISC WITH A STALLED RADIUS** (prediction: STILL A DISC, 60%; missed). Every seed follows
the compact curve exactly (ecc 648 at 1 M, 917 at 2 M) until its first dead corner and then
falls below it; the three seeds that crossed in the last octave have not yet moved.

**What the two readings mean together** (post hoc, descriptive; ball volume around node 0
on five graphs). Around the centre the volume grows exactly as r^2 out to the radius of
the first dead corners; beyond it there is a band of super-quadratic growth, local slope
d ln V / d ln r up to 3.2-3.5, before the ball saturates:

| seed | local slope of V(r) from node 0 at r = 100, 200, 300, 400, 500, 600, 700 | V(600) / flat disc |
|---|---|---|
| 308 | 3.25 3.27 2.26 1.54 (saturated) | 3.88 |
| 301 | 1.99 1.99 2.00 2.39 3.43 3.54 2.37 | 1.77 |
| 300 | 1.99 1.99 2.00 2.00 2.67 3.22 2.98 | 1.25 |
| 305, 309 (compact) | 2.00 at every r to 800, then the edge | 1.00 |

So past N* the sheet is two-dimensional wherever a ball fits inside it (every audit window,
every source), and the outer annulus carries **excess area**: it packs up to four times a
flat disc's mass within a given distance of the centre. That is what a surface of negative
curvature does (a hyperbolic disc has excess circumference), and it is the global
expression of the -1/3 boundary charges of W1-b: each dead corner forces extra boundary,
the boundary ruffles, and the ruffled outer region is neither flat nor a tree. The banked
phrase "grows tree-like arms" is withdrawn (item 1 already found no arm); "turns tree-like"
is replaced by **"stays locally flat and becomes ruffled (negatively curved) at large
scale, starting at the first dead corner"**. The 2026-09-29 diameter exponent of 0.25 is
explained: the centre's eccentricity itself grows as N^0.23-0.30 after the defect, so the
double-sweep reading was right, not an artifact.

### W1-d. The direct hazard scan: a constant per-node clock, Arrhenius at slope 6.4, and the lost race every time

*Tool: `dead_corner_hazard.py` (validated: the `on_event` hook leaves growth bit-identical;
the incremental dead set equals the static census; prefix bisection never fires after the
exact history, gap 0 on three seeds; interval coverage 0.93-0.95 on synthetic data). The
frozen scan (beta 1.0, 1.25, 1.5, 1.75 x seeds 500-539) started on the desktop 00:49 with
ten workers and was **killed by the harness for low system memory** after 14 growths (beta
1.75, seeds 500-513). **Deviation from the pre-registration, disclosed:** on the owner's go
it was resumed on jaga (`--resume`, 24 workers, 15:07-15:33 PDT) instead of the desktop;
`--resume` skips the finished (beta, seed) pairs and appends the rest, every growth is
independently seeded, and a seeded test growth was checked bit-identical on both machines
before the launch, so the CSV is the one the desktop would have written. The 14 desktop rows
are unchanged in the merged file. Log: `results/wave1_logs/w1d_jaga.log`. The twelve beta = 2
seeds the pre-registration promised as a fifth, non-confirmatory point are added below by
hand (the analysis does not produce them).*

All 160 growths reached a dead corner; none was censored.

| beta | n | first dead corner seen | hazard per node (95% CI) | mean N_first | Weibull shape k (95% CI) | Nelson-Aalen late/early slope |
|---|---|---|---|---|---|---|
| 1.00 | 40 | 40 | 2.18e-4 [1.55e-4, 2.96e-4] | 4,596 | 1.31 [1.01, 1.65] | 1.74 |
| 1.25 | 40 | 40 | 5.47e-5 [3.90e-5, 7.44e-5] | 18,300 | 1.03 [0.81, 1.27] | 0.60 |
| 1.50 | 40 | 40 | 9.65e-6 [6.89e-6, 1.31e-5] | 103,647 | 0.85 [0.65, 1.08] | 1.34 |
| 1.75 | 40 | 40 | 1.80e-6 [1.29e-6, 2.46e-6] | 554,467 | 1.25 [0.97, 1.57] | 1.44 |

Pooled common shape k = 1.07 [0.95, 1.21].

- **P1: PER-NODE HAZARD** (held, 65%). The shape interval covers 1 at three of the four
  betas (beta 1.0 misses it by 0.01) and the pooled interval covers 1. The first dead corner
  arrives on a constant per-node clock: one chance per node grown, not one per boundary site
  (that would give k near 1.5, which the pooled interval excludes). The interim shape of 2.4
  on the 14 desktop seeds was the truncation bias disclosed at the time: with all 40 seeds,
  beta 1.75 reads 1.25.
- **P2: ARRHENIUS** (held, 70%), **RATE LAW DIFFERS** (SAME LAW missed, 55%). ln lambda =
  -1.89 - 6.44 beta (slope se 0.28); the quadratic term's interval [-17, 15] covers 0. The
  slope 6.44 is 1.06 from the N*_B law's 5.38, just over the frozen 1.0 allowance. In rate
  units the law is **lambda(beta) ~ 0.15 exp(-6.44 beta) per node**. The miss is narrow and
  the two laws measure different things (the first dead corner, against the boundary
  exponent's departure on other seeds); mean N_first sits close to N*_B at beta 1.5 (104 k vs
  117 k) and below it at beta 1.0 (4.6 k vs 7.3 k).
- **Fifth point, beta = 2 (non-confirmatory, all twelve seeds seen before):** the twelve saved
  4.19 M-node graphs of the trigger follow-up have their first dead corner at the sizes in
  that table (nine seeds; three compact seeds censored at 4,194,304). Exposure MLE: 9 events
  in 27.3 M node-steps, lambda = 3.3e-7 per node [1.5e-7, 6.3e-7]. The Arrhenius fit above
  extrapolates to 3.85e-7 at beta 2 (ratio 0.86); a slope of 5.38 through the beta 1.75 point
  would give 4.7e-7. Both lie inside the interval: the fifth point is consistent with the law
  and too wide to choose between the two slopes.
- **P3: LOST RACE** (held, 75%), **160/160**: every first dead corner is an adjacent pair of
  degree-6 boundary nodes failing (iv) only, created by a single attachment.

Decomposition of the slope, as the pre-registration asked (descriptive). The attachment
endpoints' degrees after the creating event were (4, 6) in 91 seeds, (5, 6) in 40, (3, 6) in
25 and (6, 6) in 4. The capped node x went 5 -> 6 and its partner q went from deg q - 1, so
the attachment beat the waiting closure by a rate gap of 12 - (5 + deg q before) = 4, 3, 5 and
2 units of beta; the mean is 3.9. The single race therefore costs about exp(-3.9 beta), and the
remaining ~2.6 units of the measured 6.44 belong to how often the precursor (a capped boundary
node waiting next to a degree-5 one) is present. The pre-registration guessed ~3 for that
factor against the old slope 5.4.

### What wave 1 revises in the banked text

- **"~2.6-strand onset core"** (throat stage 1, 2026-08-17; round 2 item 2): withdrawn. The
  onset is the two-strand triangle in 91-98% of draws on every substrate; the verdict
  "local-motif onset, not a critical point" stands and is sharper. Revision flags added at
  each place the number was quoted (FINDINGS, BRANCHES, README).
- **"`sheet` ... grows tree-like arms beyond N*"** (2026-09-29): replaced by "stays locally
  flat and ruffles at large scale from its first dead corner on" (W1-c). The diameter
  exponent of 0.25 measured then is confirmed by the centre's eccentricity, not an
  artifact.
- **The dead corner** (follow-up of 2026-10-13/14): its mechanism is now fixed in the rule's
  own terms and in curvature terms. It is always the degree guard (474 of 474), always born
  as an adjacent capped pair (9 of 9 saved graphs, 160 of 160 fresh seeds), created by
  an attachment that beats a waiting closure.

Closed later the same day (W1-d, resumed on jaga): the hazard is constant per node, and its
rate follows exp(-a beta) with a = 6.44, not 5.4 (a narrow miss of the frozen allowance).

Reproduce: `python throat_compare.py --integer`; `python curvature_census.py
results/sheet_cross_b2_s3*_N4194304.npz`; the same with `--disc`; `python
dead_corner_hazard.py --resume results/dead_corner_hazard_20261017_004912.csv --jobs 6`
(owner's call; ten workers plus concurrent 4 M-node analyses exhausted memory).

## Literature paths, wave 2 (2026-10-17): a pin alone makes only a hairline crack; the sheet's open fjords and its ruffle are its own

*Pre-registration `docs/superpowers/specs/2026-10-17-wave2-preregistration.md` (committed
before any number below). Tools `edge_geometry.py` and `eden_pin.py` (both validated:
boundary cycle, zero fjords on a compact sheet, exact fjord on a synthetic chord graph,
alpha = 0.5 on a random-walk bridge; Eden clusters compact, the local enclosure test never
misses a disconnection, the constrained variant keeps its pin exposed). Inputs: the twelve
saved beta = 2 graphs at 4.19 M; 40 Eden runs to radius 200.*

**Summary.**

| part | frozen reading | prediction | what it says |
|---|---|---|---|
| W2-a ruffle as cone excess | **NO CONE EXCESS** (median slope g = 0.094, cut 0.1) | missed (CONE EXCESS, 50%) | the excess area starts only beyond the dead corners, but each adds about a tenth of a one-third turn, not a full one |
| W2-b fjords forced at dead corners | **UNDECIDED**: no fjord by the frozen neck definition on any graph | missed (FORCED FJORDS, 70%) | the sheet has no pinched mouth anywhere: its fjords are open channels, not baby-universe necks |
| W2-c local vs spectral roughness | **NOT KPZ-LOCAL** (alpha_loc 0.92-0.96 on 11/11 compact edges) | missed (55%) | instrument-limited: hop-distance height is quantised at one hop, so short-scale roughness is not resolved |
| W2-d boundary as a process | descriptive | | across-seed spread of B / sqrt(N) shrinks from 0.036 to 0.005 between 4 k and 2 M: self-averaging, not diffusive |
| W2-e Eden with one frozen site | **PLAIN EDEN HEALS** (20/20); **CONSTRAINED EDEN CRACKS** (20/20, width 1) | held (85%, 65%) | a permanent pin forces only a one-site crack and no excess area; the sheet's wide fjords and ruffle are not explained by the pin alone |

### W2-a. The ruffle begins at the dead corners but is not a sum of cones

Shells around node 0 against the compact seeds' mean shell, X(r) = S(r)/S_ref(r) - 1, and
D(r) = dead corners within distance r (radii 50 to 0.7 ecc):

| seed | nearest dead corner to the centre (hops) | X max | D max in window | slope g of X on D/3 |
|---|---|---|---|---|
| 308 | 87 | +6.83 | 52 | 0.309 |
| 301 | 431 | +2.35 | 47 | 0.111 |
| 300 | 502 | +1.70 | 42 | 0.094 |
| 302 | 554 | +1.50 | 65 | 0.068 |
| 303 | 712 | +0.53 | 34 | 0.027 |
| 304, 306, 307, 311 | 879-1156 (beyond the window) | 0.00 | 0 | (none) |

Leave-one-out control on the compact seeds: X = 0.000 at every radius. **No graph shows any
excess before its first dead corner is enclosed** (the early-rise clause fired on 0 of 9),
and every graph with a dead corner inside the window has X > 0 beyond it. But the slope is
0.03-0.31 (median 0.094), so the frozen reading is NO CONE EXCESS: a dead corner adds about
a tenth of the one-third turn the cone model assumed. In hindsight the model ignored
Gauss-Bonnet: the boundary's total turning is fixed at one full turn, so each dead corner's
-2 pi/3 is paid back by convex corners elsewhere on the same boundary, and a shell enclosing
both gains much less than a cone. What the ruffle does show is that it is **caused** by the
dead corners: it begins exactly where they are, on every graph.

### W2-b. No pinched mouths: the sheet's fjords are open channels

Not one neck (two boundary nodes within 3 hops, at least 100 apart along the boundary) on
any of the twelve final graphs or the 14 prefix states, crossed or compact. The frozen
reading is UNDECIDED (its fraction is 0/0). The finding behind it is definite: the sheet
grows no minimal-neck "baby universes" (Ambjorn-Jain-Thorleifsson). *[Corrected 2026-10-18:
that paper uses minimal necks to measure gamma_string, not as the defect that destroys a
dimension; pure 2D random triangulations have d_H = 4 anyway. See the survey's verification
addendum.]* Its fjords, rooted at dead corners 430-1160
hops from the centre, have walls more than 3 hops apart wherever they are separated by
more than 100 boundary nodes. A neck census with a wider mouth would be a new,
post hoc instrument; not run.

### W2-c. The compact edge looks smooth at the resolution of hop distance

| edges | alpha_loc | alpha_s (k = 6, 12, 18 removed) | spectral slope |
|---|---|---|---|
| 11 compact (3 compact seeds at 4.19 M, 8 crossed seeds before their first dead corner) | 0.92-0.96 | 0.96-1.23 | -2.93 to -3.47 |
| 9 rough (crossed seeds at 4.19 M) | 0.68-0.93 | 0.70-1.09 | -2.39 to -3.18 |

Frozen reading NOT KPZ-LOCAL (0/11 compact edges in [0.4, 0.6]; not anomalous either).
Post hoc check, labelled as such: removing each window's linear trend as well as its mean
gives 0.77-0.83 on compact edges and 0.96-0.98 on rough ones (0.51 on a random-walk
control), and the width at the smallest window is 0.31 hops on every edge, which is the
floor set by integer hop distances. The radial height in hops is too coarse to see KPZ
roughness at short scales; this reading is instrument-limited and does not refute KPZ.
The rough edges' alpha_loc (0.68-0.93) sits in the band [0.5, 0.9] for 8 of 9.

### W2-d. The compact boundary self-averages

| N | seeds | mean B / sqrt(N) | across-seed sd |
|---|---|---|---|
| 4,096 | 12 | 3.737 | 0.036 |
| 65,536 | 11 | 3.790 | 0.018 |
| 1,048,576 | 9 | 3.838 | 0.008 |
| 2,097,152 | 6 | 3.847 | 0.005 |

The spread shrinks roughly as N^(-1/4) and the mean creeps up by 3% over three decades.
That is a deterministic limit shape with shrinking relative fluctuations (Richardson /
Cox-Durrett), not the diffusing boundary of layered causal growth (Sisko-Yambartsev-Zohren).

### W2-e. A permanent pin alone forces a hairline crack and nothing more

Eden on a triangular lattice graph, one perimeter site frozen when the cluster reaches
radius 20; 20 runs per variant to radius 200 (about 9e4 sites):

| variant | frozen site exposed at R = 40 / 80 / 120 / 160 / 200 | channel width at half depth (median, R >= 80) | sites at R = 200 |
|---|---|---|---|
| plain | 0 / 0 / 0 / 0 / 0 of 20 | (enclosed) | 90,737 +/- 4,387 |
| no enclosure | 20 / 20 / 20 / 20 / 20 of 20 | 1 site (10-90%: 1-7 at R = 80, 1-1 at R = 200) | 90,557 +/- 2,521 |

**PLAIN EDEN HEALS** (enclosed by R = 40 in 20/20) and **CONSTRAINED EDEN CRACKS** (alive
at R = 200 in 20/20, width <= 3 in 20/20); both predicted. A plain Eden front simply grows
over a frozen site and leaves a one-site hole. If the pin must stay on the edge, as a
sheet's dead corner must, the front leaves a crack exactly one site wide down to it and
otherwise fills everything: no wide fjord and no excess area (the cluster holds the same
number of sites at the same radius with or without the constraint). The width at R = 40
(median 100) is the unfilled front, not the channel; by R = 80 it is 1.

**What wave 2 adds.** The dead corner is the cause of everything that happens past N*: the
ruffle starts at the dead corners on every graph and nowhere before them (W2-a), and a
pin that must stay on the edge necessarily leaves a channel (W2-e). But a pin by itself
leaves only a hairline crack in a compact front, while the sheet's channels are open (no
two boundary nodes 100 or more apart along the edge come within 3 hops, W2-b) and its outer annulus carries up to 4x a flat disc's
mass (wave 1). That extra is the sheet's own growth rule acting around the defect. A hypothesis, not yet
tested: the dead
pair blocks attachment on its own boundary edges, and the degree-capped closure cannot fill
in next to a capped node, so the stalled zone is wider than one site and the boundary near
it grows extra length. Which of those two local effects carries the excess is the next
question (it separates cleanly in a rule variant, wave 5's C1).

Reproduce: `python edge_geometry.py results/sheet_cross_b2_s3*_N4194304.npz`; `python
eden_pin.py --runs 20 --r-max 200`.


## Literature paths, wave 3 (2026-10-17): the portal threshold is Newman-Watts, the perimeter criterion is blind, and resistance explains the walkers

*Pre-registration `docs/superpowers/specs/2026-10-17-wave3-preregistration.md` (committed
before any number below); tool `instruments.py` (validated: Pruefer tree and Wilson UST are
spanning trees; effective resistance exact on a path and a cycle; a random 4-regular graph
has nothing below the Kesten-McKay edge; flipped sphere triangulations keep E = 3N - 6 and
F = 2N - 4; hull volumes exact on a torus; the saturation fit recovers a known plateau).
`shortcut_tolerance.measure_tolerance` gained an optional `max_radius` (default unchanged).
The pre-registration also records that the survey's "spike decomposition" (half of A8)
rested on a misdescribed brief and was dropped.*

**Summary.**

| part | frozen reading | prediction | what it says |
|---|---|---|---|
| W3-a spectral plateau + critical trees | **VOID** (gate failed: the saturation fit hit its bound on the 3D torus) | gate 90%, missed | the walk instrument itself reads every known answer (3D torus 3.0-3.2, tree 1.31 vs 4/3, UST 1.22 vs 16/13), so the void is the fit, not the instrument |
| W3-b gap count | not separated (`grown` has the most modes below the edge) | missed (55%) | `grown` is not an expander at N = 5000; the gap count is not a dimension test |
| W3-c effective resistance | missed on one clause (`triangular` 1.37 > 1.3) | missed (70%) | resistance ratios match the banked hitting gains (1.37 vs 1.4, 1.29 vs 1.3, 3.4 vs 6.7); the rule was self-inconsistent |
| W3-d portal threshold vs N and radius | **NEWMAN-WATTS COLLAPSE** (slope 0.74, scatter 0.19; banked k* = 20 reproduced) | held (60%) | "20 shortcuts" is the Newman-Watts crossover for N = 5000 and radius 10, not a property of the planes |
| W3-e perimeter criterion vs a random surface | **PERIMETER CRITERION BLIND** | held (60%) | B ~ sqrt(V) on a d_H = 4 surface too; the sheet's flatness rests on its volume exponent |
| W3-f box covering | `sheet` d_B 1.67 (missed [1.8, 2.2]); `grown` FINITE d_B by 3% | missed (60%) | box covering does not separate these graphs |

### W3-a. The spectral instrument reads every known answer; the frozen fit did not

Running spectral dimension D(t) (lazy walk) and the window audit, one graph each:

| graph | D(t) at t = 16 ... 11,585 | saturation-fit plateau a | window audit | theory (d_s; d_H) |
|---|---|---|---|---|
| 2D torus 300^2 | 2.03 ... 1.97-2.02 | 1.98 | DEFINED, 1.99 | 2; 2 |
| 3D torus 45^3 | 3.13 3.07 3.04 3.04 3.07 3.10 3.16 (to t = 128) | **10.0 (fit at its bound)** | DEFINED, 2.99 | 3; 3 |
| uniform random tree, 200 k | 1.15 ... 1.30 1.31 1.31 1.33 | 1.31 | MIXED, 1.87 | 4/3; 2 |
| UST of a 300^2 torus | 1.11 ... 1.22 1.22 1.24 1.25 | 1.23 | DEFINED, 1.56 | 16/13 = 1.23; 8/5 = 1.6 |
| `sheet` beta = 3, 512 k | 2.00 ... 1.90 1.88 1.85 | 1.85 | DEFINED, 1.99 | 2; 2 |
| `grown` cap 6, 200 k | 1.15 rising to 1.94, no plateau | 1.93 (not a plateau) | EXPONENTIAL | none |
| `sheet` beta = 2 seed 308, 4.19 M (ruffled) | 2.00 ... 1.82 1.77 1.73 1.70 | 1.52 | (not run) | |

**Frozen reading: VOID.** The gate required the saturation fit to return a in [2.7, 3.3] on
the 3D torus; on that short, slightly rising curve (seven points before the 45^3 torus
equilibrates) the fit ran to its parameter bound. The raw D(t) there is 3.04-3.16, and the
audit reads 2.99, so the walk instrument is sound and the fit is not robust on short curves.
Per the pre-registration the critical-tree verdict is void; it is not rescued here.

Descriptively, and labelled as such: the random tree is exactly the case the survey asked
for. Its ball growth reads d = 1.87 (MIXED, not EXPONENTIAL; Hausdorff dimension 2) while its
walk reads 1.31 (4/3), so the two instruments disagree in the way theory says they should,
and the window audit does not reject every tree. The uniform spanning tree reads 1.56 and
1.22 against 8/5 and 16/13. `grown` is the supercritical class: no plateau and an exponential
audit. On the ruffled 4.19 M-node seed 308 the stated expectation (D rising at long times,
40%) failed: D falls to 1.70, faster than on the compact 512 k sheet (1.85), so the ruffled
annulus slows the walk's spreading rather than speeding it. A robust plateau estimator
(for example the median of the last five D values) would replace the saturation fit in any
re-run; that is a new instrument and needs its own registration.

### W3-b. Gap count: `grown` is not an expander at N = 5000

Full normalized-Laplacian spectra, N = 5000 (`triangular` 4900), seed 0:

| graph | mean degree | Kesten-McKay edge | fraction f below it | lambda_2 |
|---|---|---|---|---|
| random 4-regular (control) | 4.00 | 0.134 | 0.0002 | 0.134 |
| `triangular` | 5.89 | 0.249 | 0.0755 | 0.00052 |
| `sheet` | 5.89 | 0.249 | 0.0756 | 0.00058 |
| `grown` cap 6 | 4.00 | 0.134 | **0.0850** | **0.00011** |
| random tree | 2.00 | 0 (degenerate) | 0 | 0.00001 |

Gate PASS (the random regular graph sits exactly at the edge). **Frozen reading: not
separated** (prediction GAP COUNT SEPARATES, 55%, missed): `grown` has *more* eigenvalues
below the edge than the planes and the smallest spectral gap of all. Ball growth on
`grown` is exponential, but at this size it is not an expander: its long degree-capped
arms carry many slow modes. The gap count does not discriminate "has a dimension" here;
the window audit and the spectral plateau remain the instruments. (The random tree's row
is uninformative: with mean degree 2 the edge is at 0.)

### W3-c. Effective resistance tracks the walker contrast

The banked E3 configuration exactly (N = 1500, seeds 0-9, offset portal 2 hops from source
and target), unit conductances:

| substrate | source-target distance | R_eff without portal | with portal | median ratio (range) | banked hitting-time gain |
|---|---|---|---|---|---|
| `triangular` | 30-51 | 1.8-2.5 | 1.3-1.7 | **1.37** (1.34-1.43) | 1.4x |
| `sheet` | 30-45 | 1.4-1.7 | 1.1-1.3 | **1.29** (1.25-1.42) | 1.3x |
| `grown` | 19-28 | 7.0-10.2 | 2.2-2.6 | **3.42** (2.67-3.86) | 6.7x |

**Frozen reading: missed** (RESISTANCE EXPLAINS THE CONTRAST, 70%), on one clause:
`triangular`'s 1.37 exceeds the 1.3 cap. The rule was inconsistent with itself (it asked the
triangular ratio to stay below 1.3 and to sit within a factor 2 of a banked gain of 1.4).
The other clauses hold: `grown` 3.42 >= 3, `sheet` 1.29 <= 1.3, and every substrate's ratio
is within a factor 2 of its banked gain (1.37/1.4, 1.29/1.3, 3.42/6.7 = 0.51). The mechanism
reads directly off the raw numbers: on `grown` the resistance is 7-10 over ~24 hops (close
to a single path), on the planes about 2 over ~40 hops (two-dimensional resistance grows
only logarithmically), so one extra edge relieves `grown` by a factor 3-4 and a plane by a
third.

### W3-d. The portal threshold is Newman-Watts

`measure_tolerance` on `triangular`, 3 seeds per cell; k* = first count whose seed-median
window drift exceeds 0.10:

| N | r0 | k* | N / r0^2 |
|---|---|---|---|
| 2,000 | 14 / 10 / 8 | 10 / 15 / 15 | 10 / 20 / 31 |
| 5,000 | 14 / 10 / 8 | 15 / **20** / 30 | 26 / 50 / 78 |
| 20,000 | 14 / 10 / 8 | 40 / 80 / 120 | 102 / 200 / 313 |

Slope of ln k* on ln(N/r0^2) = 0.74, scatter 0.19 over 9 cells: **NEWMAN-WATTS COLLAPSE**
(predicted, 60%; the slope is at the low end of the band). The banked cell (N = 5000,
automatic radius 10) reproduces k* = 20 exactly. The banked "a 2D substrate tolerates about
20 shortcuts" is therefore not a property of the planes: it is where the Newman-Watts
crossover length (N/S)^(1/2) falls inside a radius-10 window at N = 5000, and it moves with
N and with the window as the formula says. (Moran's I used 19 permutations instead of 199;
the drift that defines k* does not use them.)

### W3-e. The perimeter criterion cannot tell a plane from a random surface

A uniform random sphere triangulation (N = 20,000, 150 sweeps of edge flips, 6.7 M accepted;
mean degree 5.999, maximum 33) against `sheet` beta = 3 at 512,000; hulled balls below 0.1 N:

| graph | radii (middle third) | ball volume exponent d_V | hull boundary vs hull volume s_HV |
|---|---|---|---|
| random triangulation | 4-6 | 2.84 | 0.41 |
| `sheet` | 9-30 | 1.91 | 0.50 |

**PERIMETER CRITERION BLIND** (predicted, 60%): both boundaries grow like the square root of
the enclosed volume while the volume exponents differ by 0.93. A boundary that scales as
sqrt(N) certifies nothing by itself; the sheet's flatness rests on its ball-volume exponent
and the window audit, as the survey said. The triangulation's usable range is short (radii
4-6 at 20,000 nodes), so its d_V of 2.84 is a finite-size reading of a d_H = 4 surface.

### W3-f. Box covering does not separate either

Random sequential ball covering, five repetitions per radius:

| graph | N_B at r_B = 1, 4, 8, 16, 24 | power-law d_B (RSS) | exponential (RSS) |
|---|---|---|---|
| `sheet` beta = 3, 512 k | 118,681 / 16,681 / 4,961 / 1,377 / 635 | 1.67 (0.12) | (2.86) |
| `grown` cap 6, 200 k | 72,537 / 14,099 / 5,241 / 1,153 / 301 | 1.64 (1.02) | (1.05) |

`sheet`'s d_B = 1.67 misses the predicted [1.8, 2.2]: random sequential covering wastes
overlap at small radii, and over r_B = 8-24 alone the slope is 1.87. `grown` reads FINITE d_B
by a 3% RSS margin, which is no discrimination at all: its curve is power-like at small
radii and falls steeply at large ones. Prediction (EXPONENTIAL, 60%) missed on the frozen
rule; in substance box covering cannot tell these two graphs apart, which is the survey's
point that d_B is a different exponent from a dimension.

**What wave 3 adds.** Two banked statements are now explained by textbook results: the
portal threshold is the Newman-Watts crossover (it scales as N / r0^2 and reproduces the
banked 20), and the walker contrast is effective resistance (a near-single path on `grown`,
logarithmic resistance on the planes). Two cheap discriminators the survey proposed fail on
these graphs (the spectral gap count and box covering), and the perimeter criterion is shown
blind on a known d_H = 4 surface. The spectral dimension reads every known answer, but the
frozen fit voided its verdict.

Reproduce: `python instruments.py --gap | --reff | --nw | --triang | --boxes | --spectral`.


## Literature paths, wave 4 (2026-10-17): the fixed-N rules -- the tie rule decides `majority`, `triadic` collapses, curvature finds every dead corner

*Pre-registration `docs/superpowers/specs/2026-10-17-wave4-preregistration.md` (committed
before any number below; it drops the survey's D4 because the banked barrier is not a
bootstrap threshold). Tool `fixed_rules.py` (validated: majority conventions; Ollivier
curvature exact on K5, a cycle and a star; the square graph stays 4-regular and
triangle-free; a torus has two squares on every edge; the dynamic graph's bookkeeping matches
a set model). No variant is registered in `rules.py`; both backends are unchanged.*

**Summary.**

| part | frozen reading | prediction | what it says |
|---|---|---|---|
| W4-a majority tie rule | gate FAIL as frozen (only the stochastic `random` tie rule); BIAS TAKES OVER not met (its `keep` clause) | missed (55%) | with the project's argmax tie rule, state 0 takes the whole torus from any start (with 1% noise, even from 10% zeros); banked majority runs on even-degree graphs measured the tie rule |
| W4-b triadic long run; closure with decay | **COLLAPSE** (6/6, diameter -> 3, clustering 0.98); **HYSTERESIS** | held (70%, 55%) | fixed-N triadic closure ends in a dense clump; with a loss term it is bistable |
| W4-c anti-Hebbian rewiring | **SELF-ORGANIZED** (K -> 2.6-2.8 from K0 = 1 and 4) | held (60%) | a local rewiring rule can tune itself to criticality |
| W4-d Ricci flow with surgery | SHORTCUT SURGERY not met (32/32 shortcuts cut, but 7% of lattice edges too); random graph: nothing cut | missed (65%) | a sharp shortcut detector on a plane; inert on an expander |
| W4-e hard-core squares, anneal vs quench | **UNDECIDED** (both stop at phi 0.73, still expanders) | modal (40%) | a thermal ingredient does not cross the barrier at this size and schedule |
| W4-f Ollivier curvature of dead corners | **DEAD CORNERS ARE THE MOST NEGATIVE** (69/69 at -2/3, unique) | held (70%) | a local curvature identifies every dead corner and nothing else |

### W4-a. The majority tie rule decides the outcome on even-degree graphs

2D torus 100 x 100 (degree 4), 4 seeds per cell. Final fraction of state 0:

| start (zeros) | keep, sync, noise 0 | bias0 (the project's), sync, noise 0 | random, sync, noise 0 | keep, noise 0.01 | bias0, noise 0.01 |
|---|---|---|---|---|---|
| 0.50 | 0.48-0.52 | **1.000** | 0.44-0.60 | 0.46-0.54 | **0.994-0.996** |
| 0.40 | 0.20-0.23 | **1.000** | 0.000-0.002 | 0.08-0.12 | **0.994-0.996** |
| 0.30 | 0.05-0.07 | **1.000** | 0.000-0.002 | 0.01-0.02 | **0.994-0.996** |
| 0.20 | 0.006-0.013 | **1.000** | 0 | 0.004-0.006 | **0.994-0.996** |
| 0.10 | 0 | 0.05-0.07 | 0 | 0.004-0.006 | **0.994-0.996** |

(Sequential updating gives the same picture.) **Gate as frozen: FAIL**, because the gate
required every noise-0 run to end in a fixed point (sequential) or a 2-cycle (synchronous),
and the `random` tie rule is stochastic, outside the Goles-Olivos theorem; restricted to the
deterministic rules (`keep`, `bias0`) the gate passes in every run. The gate was
mis-specified for `random`. **Frozen reading: not met** (BIAS TAKES OVER required `keep` to
stay within 0.1 of its start; under `keep` the initial majority wins instead, 0.30 -> 0.06),
so the prediction (55%) missed on its second clause. The first clause held completely and is
the finding: with the argmax tie rule, **state 0 takes the whole torus from any start up to
80% ones without noise, and from every start, even 10% zeros, with 1% noise** (final
0.994-0.996). The banked setting (half-and-half start, noise 0.01) gives 0.995 zeros with
`bias0` against 0.501 with `keep`. Every banked `majority` run on an even-degree graph was
driven to state-0 consensus by the tie rule, not by the vote. (The memory note on this bias
said it existed; this is its size.)

### W4-b. `triadic` in the long run; closure with a loss term

(1) The registered rule from a random graph (mean degree 6), 5000 sweeps:

| N | seed | diameter: start -> peak (sweep) -> 5000 | clustering start -> 5000 | largest component at 5000 |
|---|---|---|---|---|
| 4000 | 0 / 1 / 2 | 9 -> 21 (400) -> 3; 8 -> 13 (200) -> 3; 8 -> 16 (400) -> 3 | 0.001-0.002 -> 0.979-0.983 | 0.958-0.968 |
| 1000 | 0 / 1 / 2 | 7 -> 9 (200) -> 3; 8 -> 10 (200) -> 3; 9 -> 11 (100) -> 3 | 0.006-0.007 -> 0.977-0.980 | 0.971-0.979 |

**COLLAPSE** (predicted, 70%), in all six runs: the extent rises for a few hundred sweeps,
then the graph folds into a clump of diameter 3 with clustering 0.98 while keeping 96-98% of
its nodes connected. The rule conserves edges but hands each rewired edge to the node it
closes on, so degree concentrates and the end state is a dense, hub-dominated core. The
banked 200-step alpha was a transient on the way up; the long-run fate of fixed-N triadic
closure is collapse (Jin-Girvan-Newman: closure without a loss term). This closes the
BRANCHES row "long-run collapse of extent under `triadic`".

(2) Closure with decay (Marsili-Vega-Redondo-Slanina variant), N = 1000, lambda = 0.01,
eta = 0.0005, 300 sweeps per step:

| xi | up-sweep mean degree (seeds 0 / 1) | down-sweep mean degree |
|---|---|---|
| 0.05 | 0.16 / 0.16 | 9.26 / 9.30 |
| 0.25 | 2.91 / 1.76 | 46.9 / 46.5 |
| 0.45 | 39.2 / 32.1 | 81.7 / 81.6 |
| 0.50 | 50.9 / 46.4 | 90.1 / 89.4 |
| 0.95 | 154.5 / 150.4 | 156.6 / 154.2 |

**HYSTERESIS** (predicted, 55%): for xi <= 0.45 the two branches differ by a factor 2 to 60.
The dense branch sits at its own balance (per sweep each node closes a triad with probability
xi, so the mean degree gains 2 xi while decay removes lambda k: k ~ 2 xi / lambda = 10 at
xi = 0.05, measured 9.3), and the sparse branch
survives because closure needs an existing triad: a bistable, first-order picture, as
Marsili et al. found.

### W4-c. Anti-Hebbian rewiring self-organizes

Bornholdt-Rohlf threshold network, N = 500, 5000 epochs:

| start K0 | seed | mean in-degree over the last 1000 epochs |
|---|---|---|
| 1 | 0, 1 | (mean of both) **2.62** |
| 4 | 0, 1 | (mean of both) **2.79** |

**SELF-ORGANIZED** (predicted, 60%): from either side the network walks to the same
connectivity, about 2.7 at N = 500 (Bornholdt-Rohlf report convergence toward K_c = 2 as N
grows). A local rule with an anti-Hebbian half tunes itself to a critical point instead of
the bistability the project's Hebbian `reinforcement` shows. No dimension is involved (this
is a directed random network); the result is a positive control for self-organization by a
local rewiring rule.

### W4-d. Ricci flow with surgery cuts every shortcut, and some lattice too

Ni-Lin-Luo-Gao flow (Ollivier, idleness 0.5, 20 iterations, cut at twice the median weight):

| graph | seed | shortcuts cut | lattice edges cut |
|---|---|---|---|
| `triangular` 2000 + shortcuts (32 placed) | 0 / 1 / 2 | **32/32** each | 383 / 394 / 397 of 5633 (6.8-7.0%) |
| random, N = 1000 | 0 / 1 / 2 | (no shortcuts) | 0 of ~3000 cut |

**Frozen reading: SHORTCUT SURGERY not met** (prediction 65%): every shortcut is cut in every
seed, but 7% of lattice edges go with them, over the 2% allowance. Which lattice edges the
flow stretches was not measured (the patch's own perimeter, whose edges have one triangle,
is the obvious candidate; untested). On the random graph the flow evens out the weights so
completely that nothing exceeds twice the median: **surgery cuts nothing and the graph does
not fragment**. The frozen label FRAGMENTS, NO DIMENSION is formally met only through its
"no dimension" clause (the audit reads INSUFFICIENT RANGE on a 1000-node expander), and should
not be read as fragmentation. Curvature flow is a sharp shortcut detector on a plane and does
nothing at all to an expander: no geometry is created, consistent with the banked barrier.

### W4-e. A thermal ingredient does not cross the barrier at this size

Hard-core square energy on 4-regular triangle-free graphs, N = 900, 1000 sweeps:

| run | E per edge | phi (edges with exactly 2 squares) | ball volume at r = 4 / 6 / 8 (torus: 41 / 85 / 145) |
|---|---|---|---|
| annealed, 4 seeds | 0.42-0.44 | 0.72-0.74 | 75 / 315-325 / 779-799 |
| quench, 4 seeds | 0.41-0.43 | 0.72-0.74 | 73-74 / 301-308 / 765-773 |

**UNDECIDED** (the modal prediction, 40%): annealing reaches no better than the zero-temperature
quench, both stop near phi = 0.73, and globally both remain expanders (balls 4 times the
torus's at r = 6). The square motif forms locally on three quarters of the edges without any
global geometry: the same crumpling the banked barrier describes for `triadic`, `geometrize`
and `ricci`, now with a thermal ingredient and the flatness motif of Kelly-Trugenberger-
Biancalana. Their equilibrium result is not reproduced by this anneal (1000 sweeps,
N = 900); a much slower schedule is the open variable.

### W4-f. Ollivier curvature finds the dead corners exactly

Ollivier curvature (idleness 0, hop distances) on every edge of two 50,000-node sheets:

| sheet | edges | interior-edge value | most negative edge | dead pairs | their curvature |
|---|---|---|---|---|---|
| beta = 1.0 (crossed) | 147,208 | 0 (all 138,980 interior edges) | -2/3 | 69 | **-2/3, every one** |
| beta = 3.0 (compact) | 149,157 | 0 (all 146,647) | -0.40 | 0 | |

Instrument check PASS (one interior value, exactly 0: the flat triangular lattice).
**DEAD CORNERS ARE THE MOST NEGATIVE** (predicted, 70%): all 69 dead-pair edges sit at -2/3,
the minimum, and no other edge in either sheet takes that value (the next values on the
crossed sheet's boundary are -0.57 and -0.42; the compact sheet never goes below -0.40). A
coordinate-free, strictly local curvature identifies every dead corner and nothing else.
Against Kelly-Trugenberger-Biancalana's claim that defects are suppressed macroscopically:
in this growth rule the defect is not suppressed, it is permanent, and it is the most
curved object in the graph.

**What wave 4 adds.** Two corrections to banked material (the `majority` tie bias is total,
not slight; the banked `triadic` exponent was a transient before collapse), one strong
positive identification (Ollivier curvature singles out exactly the dead corners), and the
barrier stands against two literature remedies (curvature flow does nothing to an expander;
a square-motif anneal crumples like the zero-temperature rules).

Reproduce: `python fixed_rules.py --majority | --triadic | --br | --ricci | --squares |
--ollivier`.


## Literature paths, wave 5 (2026-10-17): rule variants of `sheet` -- relief brings the crossover back without dead corners; veto stays a plane wherever no dead corner forms

*Pre-registration `docs/superpowers/specs/2026-10-17-wave5-preregistration.md` (committed
before any variant was grown). Variants are labelled, default-off options of
`sheet_growth.grow_sheet(variant=...)`; the default `sheet` is bit-identical (checked by
`python sheet_variants.py --validate`, which also checks Gauss-Bonnet with interior degree-7
nodes on both variants), and no variant is reported as `sheet`. Tool `sheet_variants.py`.
48 growths to 2^20 = 1,048,576 nodes, on 4 workers, about 3.5 h. Interim rows were looked
at while the scan ran; none changed anything, and all numbers below come from that one
frozen run.*

**Summary.**

| part | frozen reading | prediction | what it says |
|---|---|---|---|
| `sheet` (reference) | **CROSSOVER** 16/16 | held (95%) | the reference reproduces |
| `relief` (C1) | **CROSSOVER** 16/16; **HYPERBOLIC INTERIOR** 15/16; zero dead corners 16/16 | held (60%, 50%, 80%) | remove the dead corners by putting their curvature in the interior and the crossover comes back, milder: BRANCHES' "a plane at every beta" was wrong, the curvature theorems were right |
| `veto` (C3) | **UNDECIDED** (flat in 11-12/16, crossed in 4/16); zero dead corners only 10/16 | missed (65%, 75%) | the veto cuts the dead-corner rate by one and a half to three decades but not to zero; **every veto growth with no dead corner is a flat disc (10/10), every one that crossed had dead corners** |
| C6 cap 7 | **BOUND RESPECTED** (mu 0.48-0.49 >= 0.405 on 3/3) | held (65%) | the cap-7 sheets grow exponentially, above the Keller-Peyerimhoff bound |

### W5-a. `relief`: the crossover returns as interior negative curvature

| beta | B / sqrt(N) at 2^20 (sheet) | ecc exponent, last octave (sheet) | dead corners | interior degree-7 nodes (2^19 -> 2^20) | V(r) local slope around node 0, max | audit |
|---|---|---|---|---|---|---|
| 1.0 | 19.8-21.0 (54.7-58.0) | 0.21-0.24 (0.11-0.16) | 0 (all 8) | 142-164 -> 300-320 | 3.60-3.93 (at r = 200) | MIXED 7, DIMENSION DEFINED 1; d 1.78-1.88 |
| 1.25 | 8.9-10.1 (28.7-33.6) | 0.31-0.32 (0.16-0.19) | 0 (all 8) | 24-39 -> 56-66 | 2.62-2.99 (at r = 300) | MIXED 7, DIMENSION DEFINED 1; d 1.72-1.99 |

(The flat-disc reference is B / sqrt(N) = 3.85 and ecc exponent 0.5.) **CROSSOVER** (held,
60%): all 16 relief growths leave the plane, by both clauses. **Zero dead corners 16/16**
(held, 80%). **HYPERBOLIC INTERIOR in 15/16** crossed growths (held, 50%); the one miss
(beta 1.25, seed 600) meets every clause but the degree-7 ratio, 61/25 = 2.44 against the
[1.6, 2.4] window. Elsewhere the degree-7 count doubles per octave (ratios 1.67-2.33): cone
points arrive at a constant areal density, as the pre-registration argued.

Same-seed check of the mechanism (not registered, read off the CSV): `sheet` and `relief`
share their random stream until the first event at which they differ, and relief's first
interior degree-7 node appears within 0-3 nodes of the size at which `sheet` makes its first
dead corner on the same seed, in all 16 pairs (e.g. beta 1.0 seed 601: 11,449 vs 11,447).
Each would-be dead pair is converted into a cone point, as designed.

The crossover is milder: B / sqrt(N) is 2.7-3.3 times smaller than `sheet`'s, and the
eccentricity exponent lies between `sheet`'s and a disc's. A rough post hoc density check:
about 300 cone points per 10^6 nodes at beta 1.0 gives lambda ~ 3e-4, so the
pre-registration's onset estimate is N ~ 6 / lambda ~ 2e4; relief's B / sqrt(N) leaves 4.0
between 16k and 64k on all eight beta-1.0 seeds (profiles in the CSV). Consistent, not a test.

Caveat on "hyperbolic": the reading as frozen is met, but a local slope of 3-4 in the ball
volume around node 0 does not by itself separate exponential growth from a steeper power or
from a ruffled annulus (W1-c's reading of `sheet`); the window audit reads MIXED, d ~ 1.8, on
14 of 16. What is established is the theorem's qualitative prediction: a constant density of
negative cone points, with no defect on the boundary, is enough to end the plane. Whether
relief has fjords was not measured.

### W5-b. `veto`: no dead corner, no crossover -- but the veto leaks

| beta | growths with zero dead corners at 2^20 | their B / sqrt(N), ecc exponent, V(r) slope r = 50-400 | growths with dead corners (first at N) | their B / sqrt(N), ecc exponent |
|---|---|---|---|---|
| 1.0 | 3 (600, 605, 606) | 3.98-4.01, 0.49-0.50, 1.94-2.00 | 5: 607 (36,730), 603 (367,516), 602 (369,738), 604 (388,423), 601 (1,030,566) | 607: 16.0, 0.23; 602-604: 8.3-10.2, 0.29-0.34; 601: 4.01, 0.49 |
| 1.25 | 7 | 3.89-3.92, 0.50-0.51, 1.94-2.00 | 1: 605 (838,427) | 4.56, 0.50 |

**UNDECIDED** as frozen (prediction NO CROSSOVER, 65%: missed): 11/16 within 10% of 3.85 and
12/16 with a disc's eccentricity exponent, short of 14; 4/16 crossed. **Zero dead corners in
10/16** (prediction at least 14/16, 75%: missed). The veto removes the race it was written
for but not every birth route. On the same seeds `sheet` makes its first dead corner between
1,002 and 24,344 nodes; under `veto` the first comes at 36,730 at the earliest, and not at
all by 2^20 in ten growths: a rate reduction of one and a half to three decades.

The useful content is the conditional split, which the frozen reading pools: **all ten veto
growths that never form a dead corner are flat discs** (B / sqrt(N) 3.89-4.01, ecc exponent
0.49-0.51, ball-volume slope 2.00 out to r = 400, against `sheet`'s 29-58 on the same seeds),
and **every veto growth that crossed had dead corners, by an amount that follows how early
they came**: seed 607 (first at 36,730, 78 by 2^20) is the most crossed, the three born near
370,000 are about twice the disc's B / sqrt(N), and the two born after 800,000 have barely
begun (4.0 and 4.6). This is the cleanest evidence yet that, in this rule, a dead corner is
necessary for the crossover as well as sufficient. *[Revised by wave 6 (2026-10-18): too
narrow. Under `strict` no dead corner ever forms and the sheet still crosses at beta <= 1.0,
through frozen corners (capped boundary nodes whose closure the veto refuses); what is
necessary is a permanently unclosed capped boundary node, of which a dead corner is one kind.]*

How the leak happens (diagnostic, not registered; one growth): on seed 607 at beta 1.0 the
first dead pair is born by a **closure**, not an attachment. Closing the wheel of node 35779
adds the edge 36717-35414, raising 35414 to degree 6 next to its boundary neighbour 36448,
already at 6; the two block each other by the degree guard (mode iv). The veto reads only
attachments, so this second route stays open. Whether it is the only leak was not checked on
the other five growths.

### W5-c. C6: cap 7 sits above the Keller-Peyerimhoff bound

On the three saved c = 7, beta = 2, 512,000-node sheets (34.3% of nodes at degree 7, max 7),
the growth rate of balls around 30 core sources is mu = 0.482 / 0.490 / 0.491 per hop (13-16
radii with V < 0.1 N), all above ln 1.5 = 0.405: **BOUND RESPECTED** (held, 65%). The bound is
proved for a tessellation with every vertex at degree 7 and no boundary; these sheets have
a third of their nodes at 7, so the agreement is consistent with the theorem's constant, not
a test of it.

### Not run (as pre-registered)

C2 (fjord-mouth bridging) would never fire: W2-b found no fjord mouth on any graph. C5
(motif-size programming) has no counterpart in this rule. **C4 (reversible attachment) is
deferred, not dropped**: detaching nodes breaks the birth-order node ids every prefix tool
relies on, so it needs its own bookkeeping and pre-registration; flagged for the owner.

**What wave 5 adds.** *[Qualified by wave 6: read "dead corner" below as "permanently unclosed capped boundary node"; see wave 6.]* The dead-corner reading survives its sharpest test in both directions.
With no dead corners and nothing in their place (`veto`, ten growths), the sheet is a plane to
2^20 at beta 1.0 and 1.25, where `sheet` crosses by 10^4-10^5. With the dead corners replaced
by the same events turned into interior negative curvature (`relief`), it crosses again. What
ends the plane is the negative curvature the lost race deposits, wherever it sits; the dead
corner is how `sheet` deposits it on the boundary. BRANCHES' prediction that a variant without
dead corners would be a plane at every beta was half wrong; the pre-registration's
curvature-theorem prediction held. The veto's leak (a closure can also make a dead pair) is a
measured miss.

Reproduce: `python sheet_variants.py --validate`; `--scan --jobs 4` (writes
results/sheet_variants_*.csv); `--cap7`.

## Wave 6 (2026-10-18): relief is a hyperbolic disc to the curvature its cone points carry; a strict veto removes every dead corner and the sheet still crosses, through frozen corners

*Pre-registration `docs/superpowers/specs/2026-10-17-wave6-preregistration.md` (committed
before any registered growth; it discloses that the W6-a law was found post hoc on wave 5's
relief data, 38 of 40 pairs within 10%, and one unregistered feasibility growth of `strict`).
Owner's standing approval for overnight research (2026-10-17). New labelled, default-off
variant `strict` in `sheet_growth.grow_sheet`; default bit-identical (`python
sheet_variants.py --validate`). 72 growths on jaga, 24 workers, 00:02-01:07 PDT; CSV
`results/sheet_variants_w6_20261018_000226.csv` (md5-verified copy), re-read locally with
`--read6` to the same verdicts.*

**Summary.**

| part | frozen reading | prediction | what it says |
|---|---|---|---|
| W6-a relief vs the hyperbolic-disc law | **HYPERBOLIC DISC LAW** (129/134 pairs within 10%; largest radius within 10% in 21/24) | held (65%) | relief's ball growth is that of a hyperbolic disc whose curvature is the cone points' density times pi/3, with no fitted parameter |
| W6-b strict, gate | **PASS** (no dead corner in any of 32 growths) | held (95%) | |
| W6-b strict, beta 1.25 | **FLAT** (7/8) | held (75%) | |
| W6-b strict, beta 1.0 | **CROSSOVER** (6/8) | missed (FLAT, 75%) | |
| W6-b strict, beta 0.75 | **CROSSOVER** (8/8) | missed (FLAT, 60%) | |
| W6-b strict, beta 0.5 | **CROSSOVER** (8/8) | missed (FLAT 45%; my named alternative was UNDECIDED) | |
| W6-b sheet reference, beta 0.5 / 0.75 | **CROSSOVER** 8/8, 8/8 | held (95%) | |

### W6-a. relief follows the hyperbolic-disc law

*[Sharpened by wave 8: the low-side drift below is the graph metric; reading the ball as an equal-area hexagon (x = 0.9094 r sqrt K) removes it, and the corrected law holds on fresh growths into x = 4.5.]*

Prediction for the local slope of V(r) around node 0: s(r) = x sinh x / (cosh x - 1),
x = r sqrt(2 pi lambda / (3 sqrt 3)), times the flat estimator's slope / 2; lambda = interior
degree-7 nodes per node, measured on each graph.

| beta | lambda (8 seeds) | eligible radii | measured / predicted, mean (range) | pairs outside 10% |
|---|---|---|---|---|
| 1.0 | 2.70-3.05e-4 | 50-200 | 0.956 (0.880-1.031) | 4 of 32 (all low, 0.88-0.89) |
| 1.25 | 5.0-6.7e-5 | 50-300 | 0.971 (0.898-1.040) | 1 of 48 (0.898) |
| 1.5 | 0.67-1.24e-5 | 50-400 | 0.996 (0.946-1.055) | 0 of 54 |

Example, beta 1.0 seed 704: measured 2.14 / 2.61 / 3.18 / 3.87 at r = 50 / 100 / 150 / 200
against 2.08 / 2.54 / 3.15 / 3.90. **HYPERBOLIC DISC LAW** (held, 65%). The misses all lie on
the low side, and the mean ratio falls below 1 as the curvature grows (0.996, 0.971, 0.956):
the failure direction the pre-registration named, a ball that lags the continuum law where it
is most curved, by 3-4% on average. As registered, beta 1.5 is a weak cell (predicted slopes
2.0-2.4). The radii above 200 at beta 1.0 were not eligible (the ball reaches half the graph),
so the steepest part of the law (slopes 4.5-6) remains untested.

A consequence, descriptive: relief's cone density is close to the default rule's
dead-corner hazard per node from W1-d (2.9e-4 vs 2.2e-4 at beta 1.0; 5.6e-5 vs 5.5e-5 at
1.25; 1.0e-5 vs 9.6e-6 at 1.5), as it should be if each relief cone point is one would-be lost
race. Combined with W1-d's Arrhenius law, relief's curvature is predicted in closed form,
|K(beta)| ~ 1.21 x 0.15 exp(-6.44 beta) per unit lattice area, from two local measurements.

### W6-b. strict: no dead corner, but frozen corners, and the crossover returns

`strict` refuses any attachment or closure that would leave a dead corner within one hop.
The gate held in all 32 growths: **no dead corner ever formed**. The sheet still crossed at
every beta below 1.25:

| beta | strict: B/sqrtN at 2^20 (factor since 2^16) | ecc exponent | capped boundary nodes at 2^20 | default `sheet`, same seeds: B/sqrtN, dead corners |
|---|---|---|---|---|
| 0.5 | 56.5-60.7 (x3.4-5.7) | 0.12-0.16 | 4,122-4,404 | 194-197, ~15,600 |
| 0.75 | 16.5-25.2 (x2.3-5.8) | 0.17-0.26 | 574-877 | 102-105, ~5,000 |
| 1.0 | 4.03-13.29 (x1.02-3.31) | 0.26-0.50 | 69-195 | (wave 5, other seeds: 55-58, ~1,550) |
| 1.25 | 3.91-4.05 in 7, 8.46 in one (x1.00-1.03; x2.15) | 0.49-0.50; 0.32 | 16-23; 68 | (wave 5: 29-34, ~470) |

So the frozen readings: FLAT at 1.25 (held), CROSSOVER at 1.0, 0.75 and 0.5 (three misses;
I predicted FLAT, with UNDECIDED as the named alternative at 0.5). `strict` delays and weakens
the crossover (3-4 times less excess boundary than `sheet` at 0.5 and 0.75) but does not
remove it.

**What crosses instead (diagnostic, not registered; one unregistered growth, beta 0.5,
seed 1, to 2^18).** At each checkpoint I classified every capped boundary node: either its
closure is allowed and simply has not fired yet, or `strict` refuses it because closing would
make a dead corner next door. The refused ones are **frozen corners**: a degree-6 boundary node
whose wheel can never close. Every frozen corner persisted to the next checkpoint (20 of 20,
45 of 45, 103 of 103; 230 at 2^18), and every persisting capped boundary node was one; the
waiting ones were all gone a checkpoint later. A frozen corner has the dead corner's
boundary curvature, (4 - 6) / 6 = -1/3. The veto relabels the defect instead of removing it:
the lost-race configuration still forms, and `strict` freezes the closure that would have
turned it into a dead pair. (At beta 0.75, seed 1, to 2^17, no frozen corner had formed and
every capped node was transient.)

**What waves 5 and 6 together say.** The crossover is triggered by permanent negative
curvature deposited by the lost race, in whatever form the rule leaves it: a dead pair on the
boundary (`sheet`), a frozen corner on the boundary (`strict`), an interior cone point
(`relief`), and quantitatively so for the last one (W6-a). Wave 5's `veto` growths that stayed
flat were the ones in which neither a dead nor (untested) a frozen configuration had formed by
2^20. "A dead corner is necessary" (wave 5) was too narrow and is revised: **a permanently
unclosed capped boundary node is what is needed**, and a dead corner is one kind. The curvature
statement is unchanged and stronger. Untested: a frozen-corner count on the registered
strict growths (the CSV has only the total capped boundary count, which mixes in waiting
nodes; it does rank the beta 1.0 growths: the two flat ones have 69-74, the crossed ones
86-195).

Reproduce: `python sheet_variants.py --validate`; `--scan6 --jobs 24`; `--read6 CSV`.

## Wave 7 (2026-10-18): frozen corners decide the crossover growth by growth

*Pre-registration `docs/superpowers/specs/2026-10-18-wave7-preregistration.md` (committed
before any registered growth). Instrument `sheet_variants.frozen_set` at every octave
checkpoint (closure tentatively applied, checked, reverted; growth untouched). 32 `strict`
growths (beta 1.0 and 1.125, seeds 800-815, to 2^20) on jaga, 01:15-01:45 PDT; CSV
`results/sheet_variants_w7_20261018_011531.csv` (md5-verified copy), re-read locally with
`--read7` to the same verdicts.*

| reading | result | prediction |
|---|---|---|
| gate: no dead corner | **PASS** (32/32) | held (95%) |
| P1: no frozen corner at 2^19 -> flat at 2^20 | **18/19** | held (75%) |
| P2: first frozen corner by 2^18 -> crossed at 2^20 | **8/8** | held (70%) |
| verdict | **FROZEN CORNERS DECIDE** | held (55%) |
| persistence (frozen at one checkpoint, still frozen at the next) | **264/264** | held (85%) |

The one P1 exception (beta 1.0, seed 809) had no frozen corner at 2^19, acquired 16 in the
last octave, and crossed (B/sqrtN x1.72, ecc exponent 0.42), which the reading itself allows
for (its first frozen corner came inside the octave it was judged on).

Grouped by the frozen count at 2^20 (descriptive):

| frozen corners at 2^20 | growths | flat | crossed | B/sqrtN at 2^20 |
|---|---|---|---|---|
| 0 | 17 (beta 1.0: 7; 1.125: 10) | 17 | 0 | 3.93-4.00 |
| 2 (first in the last octave) | 1 (1.125 seed 811) | 1 | 0 | 3.98 |
| 16-36 | 14 (beta 1.0: 9; 1.125: 5) | 0 | 14 | 6.80-12.47 |

The earlier the first frozen corner, the larger the excess: first seen by 2^17 (four
growths) gives B/sqrtN 11.7-12.5, at 2^18 (four) 9.4-11.2, at 2^19 (five) 6.9-8.7, at 2^20
(two) 6.8 and 4.0. No growth with
frozen corners in more than one checkpoint stayed flat, and no flat growth had more than two.

**What this settles.** Wave 6's revision holds per growth: under `strict`, which never makes a
dead corner, whether a sheet stays a disc to a million nodes is decided by whether it has
acquired permanently unclosed capped boundary nodes, and how far it departs follows how early
they came. With wave 5 (`relief`: the same events deposited as interior cone points, and the
crossover follows a hyperbolic-disc law, W6-a) the trigger now reads in one line: **the
`sheet` crossover is the deposition of permanent negative curvature by the lost race --
on the boundary as a dead or frozen corner, or in the interior as a cone point.** A rule that
removes the configuration rather than relabelling it remains untested; that would be the
next variant (e.g. a closure that is allowed to proceed and flip a neighbouring triangle). *(Wave 9 built four such rules; a flip-repair closure reduces by
Gauss-Bonnet to a frozen corner or a 5-7 pair, so the 5-7 pair (`disloc`) was built directly.
Only local cancellation keeps the plane.)*

Reproduce: `python sheet_variants.py --scan7 --jobs 24`; `--read7 CSV`.

## Wave 8 (2026-10-18): the hyperbolic-disc law holds with the graph metric read correctly, into the steep part

*Pre-registration `docs/superpowers/specs/2026-10-18-wave8-preregistration.md` (committed
before any registered growth; it discloses the ring-disc calibration that motivated the
correction, a post-hoc re-read of wave 6 with it, and one smoke growth). 24 `relief`
growths (beta 0.75 / 0.875 / 1.0, seeds 900-907, to 2^21) on jaga, 22:37-23:12 PDT; CSV
`results/sheet_variants_w8_20261018_223657.csv` (md5-verified copy), re-read locally with
`--read8` to the same verdict. W8-b ran locally (`results/w8b_calibration_seed1.txt`).*

**The correction.** A graph ball on the triangular lattice is a hexagon of circumradius r,
not a disc. Read as the disc of equal area its radius is HEX_AREA r, HEX_AREA =
sqrt(3 sqrt 3 / (2 pi)) = 0.9094, a lattice constant with nothing fitted. The wave 6 law
becomes s(r) = x sinh x / (cosh x - 1) with x = 0.9094 r sqrt(2 pi lambda / (3 sqrt 3)),
times the exact lattice slope / 2.

| reading | result | prediction |
|---|---|---|
| gate (not stalled, no dead corner, max degree 7), per beta | **24/24** (8/8 at each beta) | held (90% / 75% / 55%) |
| W8-a: pairs within 10% | **144/154 = 0.935** (>= 0.90) | |
| W8-a: mean ratio per beta | **1.015 / 1.019 / 1.019** (beta 0.75 / 0.875 / 1.0; all in [0.97, 1.03]) | |
| W8-a: steep part, x_hex >= 4 | **10 pairs (the registered minimum), mean 1.030** | |
| W8-a verdict | **HEX LAW HOLDS** | held (60%) |
| registered (uncorrected) law on the steep pairs | **0.945** | held (below 0.95, 75%) |
| W8-b: ring-disc calibration, fresh seed | **CALIBRATION REPLICATES** (58/58 cells in [0.97, 1.04]; range 0.984-1.038) | held (80%) |

**What it says.** `relief`'s ball growth is that of a hyperbolic disc whose curvature is
fixed by the density of its cone points, with the graph metric converted by a lattice
constant. That holds from x = 0.4 to 4.5, for beta from 0.75 to 1.5 (waves 6 and 8), and on
random cone-point discs built without any growth rule (W8-b). The systematic low side of
wave 6 (means 0.996 / 0.971 / 0.956) was the metric, not the sheet. Re-read with the
correction (post hoc, disclosed), wave 6 gives 1.000 / 0.997 / 1.004 and 0 of 134 pairs
outside 10%. Now the residual leans the other way, about +1.5-2%, and 9 of the 10 pairs outside
10% are high (1.109-1.144; one low, 0.893): on relief the correction slightly overshoots.
How far it overshoots varies from ball to ball, with the number of cone points a radius-r
ball happens to hold (the beta 1.0 seeds wander between 0.95 and 1.14 at the same radius).
The truth may sit between the two readings (the ring discs centre on 1.00-1.01). Limits: the steep part was reached only to x = 4.5, not the hoped-for 6,
because at beta 0.75 the ball passes half the graph by r ~ 150. Ten pairs is the bare
minimum, all but two from one beta.

**A by-product, descriptive.** Over 48 relief growths, beta 0.75-1.5, the cone density is
lambda = 0.193 exp(-6.55 beta). The default rule's per-node dead-corner hazard (W1-d) is
0.15 exp(-6.44 beta). Per beta the ratio is 1.03-1.22 (against the W1-d fit) across a factor 130 in rate: each
relief cone point is a would-be lost race, and relief's curvature radius
1 / sqrt|K| = 1 / sqrt(1.209 lambda) ~ 2.07 exp(3.28 beta) lattice spacings follows from one
local Arrhenius measurement. This is the Henderson-Taimina hyperbolic-crochet relation ("the
ratio determines the radius") with the increase rate set by a kinetic barrier.

Reproduce: `python sheet_variants.py --scan8 --jobs 24`; `--read8 CSV`;
`python cone_lattice.py --calibrate --seed 1`.

## Wave 9 (2026-10-19): cancelling the lost race's curvature locally keeps the plane; removing it or diluting it does not

*Pre-registration `docs/superpowers/specs/2026-10-19-wave9-preregistration.md` (committed
before any registered growth; it discloses the code checks, the sampler bug found and fixed
in `detach`, the pilots and the reader dry runs that made the `pent` spherical side
descriptive). Four labelled, default-off variants of `grow_sheet` (the default `sheet` is
bit-identical): `disloc` (every degree-7 cone point leaves a credit that a later degree-5
closure consumes: bound 5-7 pairs), `pent` (degree-5 closures at a fixed small rate,
independent of the 7s), `eager` (closure rate rho instead of 1), `detach` (C4: reversible
attachment on `sheet`). 124 growths on jaga, 21:11-22:57 PDT, 24 workers; CSV
`results/sheet_variants_w9_20261019_211135.csv` (md5-verified copy; log
`results/wave1_logs/wave9_jaga.log`), read locally with `--read9`.*

| reading | result | prediction |
|---|---|---|
| W9-a `disloc` gate (no stall, no dead corner, max degree 7, interior degrees 5-7, >= 95% of 5s with a 7 within two hops) | **PASS** (24/24; 5s paired 99.7-100%) | held (75%) |
| W9-a cells: beta 1.0 / 0.75 / 0.5 | **FLAT / FLAT / CROSSOVER** (beta 0.75: 7/8 flat, the eighth "neither") | held (85% / 65% / 45%) |
| W9-a net-curvature law | **HOLDS** (359/372 pairs within 10%, mean ratio 1.023) | held (65%) |
| W9-a `relief` controls crossed | **12/12** (B/sqrtN 19-21 / 43-45 / 92-94 at beta 1.0 / 0.75 / 0.5) | held (95%) |
| W9-b `pent` net law | **NET LAW MISSED** (288/339 = 85% within 10%; discriminating pairs 93: net mean 0.974, gross mean 0.760) | held (55%) |
| W9-b gross law on discriminating pairs below 0.90 | **0.760** | held (80%) |
| W9-b cells: rate 3e-6 / 6e-6 / 1e-5 | **CROSSOVER / CROSSOVER / CROSSOVER** | held, held, **missed** (I gave UNDECIDED 50%) |
| W9-c `eager` race law: slope of ln lambda vs ln rho | **RACE LAW** (-0.934 at beta 0.5, -0.927 at 0.75) | held (65%) |
| W9-d `detach` cells: beta 1.0 mu 3 / 3.8, beta 1.25 mu 3 / 3.8 | **CROSSOVER x 4** (32/32 crossed) | held (90% / 75% / 70% / 60%) |

Fifteen of sixteen registered calls held; the miss is the near-balance `pent` cell, which
crossed where I expected the fluctuations to leave it undecided.

**W9-a, `disloc`: the first variant with no dead or frozen corner that reads flat at
beta 0.75 -- marginally.** At beta 1.0 every growth is flat to 2^20 (B/sqrtN 4.11-4.13,
level over eight octaves; flat `strict` growths read 3.9-4.0, and `relief` on the same footing
is 19-21), with 306-350 degree-5 and 308-351 degree-7 points, every 5 within two hops of a 7.
At beta 0.75 (1,582-1,719 pairs) the frozen reading calls 7/8 flat, but only just: B/sqrtN
rises 4.1 -> 4.9 over eight octaves, the 2^16 -> 2^20 ratio is 1.12-1.15 in every growth
against the window's 1.15 (seed 1000, 1.152, is the "neither"), it is still rising 4-6% in
the last octave, and the eccentricity exponent is 0.45-0.46 against 0.48-0.50 at beta 1.0.
The ball around node 0 stays close to the lattice's slope (2.03 at r = 100, 2.06 at r = 250).
Flat at 2^20 is what the reading says; whether beta 0.75 stays flat beyond it is open (see
the beta 0.5 remainder below). `strict` crossed at both
betas; `veto` leaked. The disc law with lambda_net = (n7 - n5)/N predicts both cells to
within 10% at every eligible radius (256/256 pairs).

At beta 0.5 it crosses, as I gave most weight to, but not quite for the reason I registered.
I wrote that a crossing there would be roughness from the dense dislocations (one pair per
~135 nodes), not curvature. The ball around node 0 says otherwise: its local slope climbs
2.17 / 2.36 / 2.58 at r = 100 / 200 / 300 against the lattice's ~2.0. The source is the
unpaired remainder: at 2^20 each beta 0.5 growth has 18-34 more 7s than 5s, credits not yet
spent that ride outward on the boundary (beta 0.75: 2-10; beta 1.0: 0-2). The net law with
that remainder accounts for most of the excess (103/116 pairs within 10%); all 13 misses are
here, all high (1.10-1.13 at r = 150-250). So the beta 0.5 crossing is mostly real curvature
from cancellation that lags behind the growth front, with something extra on top. Whether
the remainder grows with the boundary, and so whether every beta eventually crosses, is not
answered by this wave.

**W9-b, `pent`: on average the net curvature sets the disc, but independent defects do not
cancel ball by ball.** Where the two laws differ by more than 10%, the net law sits at 0.974
and the 7s-alone law at 0.760: degree-5 points do subtract their curvature. But the
registered 90% clause failed, the way I expected: within 10% at 91% / 87% / 77% of pairs at
rates 3e-6 / 6e-6 / 1e-5, the misses at r >= 50 and growing with the rate and the radius
(rate 1e-5 beyond r = 200: 4/14). Individual balls wander between 0.67 and 1.28, the sqrt(n)
excess the dry run estimated. Against `disloc`, where the same law holds at 96.5%: **the
curvature must cancel locally, as bound pairs do, for the disc law to hold ball by ball.**
The descriptive spherical side (rates 3e-5, 1e-4, to 2^18) behaved as the dry runs warned:
pinched bulbs (28-128 dead corners from capped boundary nodes under the relief guard), one
stall at 7,443 nodes, one growth whose node 0 sits in a closed bulb (ball volume saturates,
ratios -> 0); complete-ball pairs' mean 0.778. That side does not test a disc law.

**W9-c, `eager`: the cone density is one race.** Across rho = 1-64 the cone density falls as
rho^-0.93 at both betas (beta 0.75: 1,323-1,414 cones at rho 1 down to 25-34 at rho 64). So
the curvature radius grows only as sqrt(rho): no finite closure speed gives a plane. Every
`eager` growth crossed at 2^20, including rho 64 (B/sqrtN 6.5-7.8 at beta 0.75), whose pilot
had stayed flat to 2^18; at lambda = 2.8e-5 the curvature radius 1/sqrt(1.209 lambda) ~ 170
spacings sits well inside a 2^20 disc (hex radius ~590). *Post hoc, descriptive:* the
hex-corrected disc law of wave 8 holds on all 32 `eager` growths, 376/385 pairs within 10%,
cell means 0.988-1.022, over a factor 190 in cone density (5.3e-3 down to 2.8e-5).

**W9-d, `detach`: reversibility heals some dead corners but not enough.** Detachment
removed 3-13% of all nodes grown and healed (a dead corner present at one checkpoint, gone at
the next) 16% / 40% / 6% / 21% of dead corners in the four cells. Dead corners at 2^20 fall
from `sheet`'s ~1,550 (beta 1.0) and ~470 (beta 1.25; wave 5, other seeds) to 900-1,070 / 360-590 / 220-360 /
120-190, and B/sqrtN from 55-58 / 29-34 to 42-47 / 27-36 / 21-29 / 15-22: a weaker crossover,
not none. A dead corner heals only when one of its degree-6 nodes detaches, at rate
exp(-beta (12 - mu)), slower than the attachments that bury it.

**What wave 9 settles.** Gauss-Bonnet leaves a local rule four things to do with the -1 of
turning the lost race fixes: keep it (`sheet`, `strict`, `relief`: all cross), dilute it
(`eager`: the density falls as rho^-0.93 but never reaches zero; still crosses), remove the
material (`detach`: heals a fraction; still crosses at mu <= 3.8), or cancel it with a +1. Only
cancellation keeps the plane, and only when the cancellation is local: bound 5-7 pairs
(`disloc`) give a flat disc to a million nodes at beta 1.0 and, marginally and still
drifting, at 0.75; independent 5s (`pent`) cancel on average but leave the ball-to-ball excess
of a random walk. **What ends the plane is the net curvature, not the defect**: the disc law
with the net density predicts the ball within 10% for `relief` (waves 6, 8), `disloc` at
beta >= 0.75 and `eager` (post hoc); at `disloc` beta 0.5 it misses high at 13 of 116 radii,
and for `pent` it holds only on average (85%, the registered clause missed). It was not
tested on `detach` or `strict`. One caveat about cost:
`disloc`'s credit is a node-state token carried to neighbours, so it is local, but it is a
conservation law written into the rule by hand. It shows what a rule must do to stay flat;
it is not evidence that a rule without such bookkeeping would find it.

Reproduce: `python sheet_variants.py --scan9 --jobs 24`; `--read9 CSV`;
`python sheet_variants.py --validate`; `python sheet_sampler_audit.py`.

## Supporting-literature reproductions (2026-10-18, side quests, not pre-registered)

*The owner asked for "supporting papers we rely on" to be checked or recreated. Citations
were re-read in full; the corrections are in the survey's verification addendum
(`reports/Local graph growth literature survey.md`, notes under `research_notes/Supporting
literature verification/`). Where a published result could be recomputed, it was.*

| published result | reproduction | outcome |
|---|---|---|
| {3,7} triangulation: sphere sizes 7, 21, 56, 147, 385, ..., growth (3 + sqrt 5)/2 (Keller-Peyerimhoff Prop. 1.1, after Cannon-Wagreich, Floyd-Plotnick) | `cone_lattice.py --validate` (ring-built disc, target degree 7) | exact: 7, 21, 56, 147, 385, 1008, 2639, 6909; ratio 2.6180339888 |
| triangular lattice ball V = 3r^2 + 3r + 1 | same, target degree 6 | exact to r = 50; the exact lattice slope estimator reproduces wave 6's measured FLAT_SLOPE to 0.01 at every radius |
| Angel-Benjamini-Horesh Thm 2: V <= floor((n + 3)^2 / 12) for a disc with interior degrees >= 6 | default `sheet`, beta 1.5 and 0.75, seed 5, four checkpoints to 2^17 | hypothesis holds (interior min degree 6 at every checkpoint), bound holds; V / bound = 1/E is 0.79-0.82 while compact and falls to 0.32 (beta 1.5, after its first dead corner) and 0.009 (beta 0.75): E is bounded below by 1 by theorem |
| Wu-Menichetti-Rahmede-Bianconi 2015, m = 2, p = 0: P(k) = (1/2)(2/3)^(k - 1), distance ~ log N | `wu_network.py --m 2 --p 0` (N to 256,000, seed 0) | P(k) to the third decimal for k = 2-13 (0.3333 / 0.2224 / 0.1474 vs 0.3333 / 0.2222 / 0.1481 ...); mean distance 1.07-1.18 ln N |
| same, m = 2, p = 0.9: exponential degree distribution, exponential growth (infinite d_H) | `wu_network.py --m 2 --p 0.9` | exponential degree tail (ratio ~0.72 per degree) and exponential balls around node 0 (x1.7-2.0 per hop to 35,000 nodes); mean distance grows faster than ln N at these sizes (1.35 -> 2.31 ln N from 10^3 to 2.6 x 10^5, one seed): log N not yet visible |

**Two reading notes from the Wu et al. reproduction.** (1) Their process (b), read literally
("add every triangle the new link closes"), folds the first few triangles into a closed
tetrahedron at p = 0.9 within a handful of steps (no unsaturated link left); the paper's own
statement that for m = 2 process (b) adds "only one new link and one new triangle" fixes the
reading, and that is what `wu_network.py` does for m = 2. (2) With m = 2 their rule is
`sheet`'s move set without the degree cap and without the bond energetics: attachments on a
uniformly chosen boundary link, closures of a boundary vertex's two boundary links. Without
the cap, interior vertices reach any degree (negative curvature everywhere), and the result
is hyperbolic from the start; the cap is what makes `sheet` a plane, and the lost race is
what lets curvature in anyway.

**The hyperbolic-disc law, calibrated (`cone_lattice.py --calibrate`).** Ring-built discs with
random degree-7 cone points at density p = 1e-4 ... 0.1 (32 discs per density) test wave 6's
law on graphs whose curvature is set by construction. The law as registered in wave 6
(x = r sqrt K) overshoots the measured slope by up to 8% at x = 4-5 at every density. Reading
the graph ball as a hexagon of equal area (x = 0.9094 r sqrt K, a lattice constant, nothing
fitted) brings every cell to 0.987-1.033 (seed 0) and 0.984-1.038 (seed 1, registered as
W8-b). The {3,7} lattice is the far limit: growth per layer 0.962 against the corrected law's
1.000. This is what wave 8 tests on `relief`.

## Scaling directions: what more compute could (and couldn't) unlock

A standing question is whether *scaling up* -- to the largest graphs a private
budget on cloud compute can reach -- would reveal emergence the small-scale
runs miss. Our own results already partition the question. The split is sharp,
so it is worth stating before spending the compute.

### The rewiring-from-disorder direction is a dead end that scale *worsens*

The bootstrapping barrier is **structural, not finite-size**. An expander on
`N` nodes has diameter `~ log N`; a `d`-dimensional graph on `N` nodes needs
diameter `~ N^(1/d)`. As `N` grows the target extent grows *polynomially* while
the disordered graph's extent grows only *logarithmically* -- so a larger
random graph is *further* from geometric, not closer. No amount of compute
rescues `triadic` / `geometrize` / `ricci` nucleating geometry from disorder;
scaling them only buys a more expensive confirmation of the same wall. (That
confirmation -- "local rewiring provably cannot grow extent, verified to 1e8
nodes" -- is a legitimate negative, but it is a negative.)

### The growth + walk-probe direction is where scale is the right lever

Three questions only resolve at large `N`:

1. **Spectral dimension via walks** -- measure `d_s` from random-walk
   return-probability scaling `P(t) ~ t^(-d_s/2)` on `grown` graphs, and ask
   whether it matches the ball-growth (Hausdorff) `d_eff` or diverges from it
   (a fractal signature). On a 32-node subgraph the quantum-vs-classical TVD is
   noise; over decades of `t` on a 1e7-node graph it is a precise instrument.
   The `traverse.py` walk machinery + `braket_walks.py` CTQW are the seed of
   this. **This is the "including random walks" path.** *Pre-committed null:*
   the most likely outcome is `d_s = d_H` (no flow) -- CDT's reduction comes
   from causal/geometric content that `grown` deliberately lacks, so a flat
   `d_s` is the *expected* result and is itself clean ("minimal local growth
   gives a manifold-like graph with no anomalous spectral flow, isolating what
   extra ingredient CDT's reduction requires"). It is a negative dressed as a
   positive -- worth doing only if that outcome is acceptable upfront. Highest
   cost, highest risk; gate it behind the cheap `cap -> d` check.
2. **`cap -> d` finite-size scaling** -- does the `grown` generator's
   cap→dimension law sharpen to clean integers as `N` climbs, or drift?
   Cheapest of the three; settles an existing open thread and is the
   prerequisite for trusting anything at 1e8.
3. **Hunt a phase transition** -- the canonical "more is different" test. Sweep
   a rule parameter, and use finite-size scaling (Binder-cumulant crossing,
   diverging susceptibility, data collapse) to find a critical point where a
   correlation length diverges and long-range structure appears spontaneously.
   A critical point is, by construction, emergence invisible at small `N`. The
   first concrete step is a **`majority`/Ising-on-a-lattice validation** (see
   `ising_sweep.py` / below) -- recover known critical behavior to prove the
   FSS machinery works -- then turn it on `prune` (shortcut-density → dimensional
   onset), the novel case tied to our one positive emergence result.
   **Universality-class caveat:** `prune`'s transition has no reason to be in
   the Ising class -- it may be percolation-like / a connectivity transition
   with different exponents and possibly no `M = <|m|>`-style order parameter.
   The Ising run validates the *pipeline*; it does **not** license importing
   Ising exponents (`β/ν=1/8`, `1/ν=1`) as `prune` defaults. For `prune` the
   exponents must be **extracted, not assumed**, and the order parameter argued
   from the actual symmetry/structure of the transition -- otherwise the data
   collapse becomes a fit-anything trap.

### Where the literature already stands on the walk path (is a big negative result waiting?)

Short answer: **no -- the random-walk-on-emergent-geometry path is a mature
*positive*-result literature, not a negative-result one.** The largest published
work is the opposite of trivial:

- **Causal dynamical triangulations (CDT)** measure the spectral dimension by
  random-walk return probability on Monte-Carlo-sampled emergent geometries and
  find a famous *scale-dependent dimensional reduction* -- `d_s` flows toward
  ~2 at short scales and matches the topological dimension at intermediate
  scales. This is a flagship quantum-gravity result, run at large simplex
  counts. (Coumbe & collaborators, *Scaling analyses of the spectral dimension
  in 3D CDT*, arXiv:1711.02685.)
- **Tunable-spectral-dimension networks** are built precisely as a
  "universality playground" for critical phenomena -- a direct cousin of our
  `grown` cap→d generator. (Millán et al., *Complex networks with tuneable
  spectral dimension as a universality playground*, Phys. Rev. Research 3,
  023015, 2021.)
- **Quantum walks on networks** are established probes of community structure,
  faults, and clustering-induced localization -- i.e. quantum-vs-classical
  divergence reliably *does* reveal structure.

**Implication.** "Do walks reveal emergent structure at scale?" is settled
*yes*; merely scaling that up would re-confirm known physics, so the bar for
novelty is specific. The compute-worthy, genuinely-open question sits *between*
the engineered tunable-`d_s` networks and the geometry-baked-in CDT result:

> Does a **minimal local growth rule** (our `grown` generator -- no baked-in
> causal or geometric structure) spontaneously reproduce a CDT-like spectral-
> dimension *flow* `d_s(scale)`, and does `d_s` agree with or split from the
> ball-growth `d_eff`?

That is the thing that takes real compute and would be new -- emergence of
spectral dimension from a dead-simple rule, rather than from a construction
designed to have it. The closest thing to a "negative" in the literature is the
known fact that `d_s != d_H` (Hausdorff) on fractals -- which is a feature to
measure, not a null result to overturn.

### Recommendation / sequencing (budget-honest)

1. **`cap -> d` finite-size scaling on `grown`** -- cheapest, highest-certainty
   win; a small *positive* that proves the instrument scales and lets us
   extrapolate a crossover estimate. Do it first regardless.
2. **Quantified bootstrapping barrier** -- extent-growth-rate vs N for the three
   rewiring rules. This is the *flagship negative* and the one place where
   "verified to large N" genuinely adds evidence (the gap is supposed to widen
   with N). Local hardware reaches the N where the trend is clear.
3. **`prune` phase-transition FSS** -- *done (step 3): there is no transition.*
   The dimensional onset is a **crossover, not a critical point** -- `p` tunes
   the pruned dimension continuously (1→2), N-independently, with a peak slope
   that does not grow with N. So no exponents to extract; the honesty was in
   *not* forcing a collapse. See "prune dimensional onset" below.
4. **Spectral-dimension flow on `grown`** -- the only potential positive
   flagship, but most expensive and most likely a (still-publishable) null;
   gate it behind step 1 confirming `grown` behaves at scale.

The **`majority`/Ising FSS validation** (below) is already done and is the
prerequisite that licenses the *pipeline* for steps 3-4 (not the exponents).

### cap → d scaling: done (step 1) -- the instrument scales; the law is a continuum

> *[superseded 2026-09-26]* The plateau in N below is real, but it is the convergence
> of a radius-10 reading; `grown` has no dimension. See the audit.

`cap_dimension_scaling.py` sweeps the `grown` generator over cap x N x seeds and
measures the dimension field at a *fixed* radius (so the regime gate, not a
varying radius, decides resolvability), then checks for a plateau in `d_eff(N)`.
Result (caps 6/7/8, `N` to 2e5, 3 seeds, radius 10):

| cap | d_eff at small N (old table) | converged d_eff (this run) | plateau by |
|-----|------------------------------|----------------------------|------------|
| 6   | 2.1                          | **~2.2**                   | N ~ 1e4    |
| 7   | 2.7                          | **~3.0**                   | N ~ 2e4    |
| 8   | 2.9                          | **~3.6** (not 4)           | N ~ 1e5    |

Three takeaways, all useful:

1. **The instrument and the generator behave at scale.** `d_eff(N)` plateaus
   cleanly (Δ < 0.03 between the last two `N`), and `defined_frac ~ 1` for all
   but the smallest `N`. This is the small *positive* that licenses trusting
   larger runs.
2. **The small-N cap→d numbers were biased low** -- a fixed radius reads the
   ball-growth slope low until `N` is large enough that all radii clear the
   saturation gate. Higher-d caps resolve only at larger `N` (cap 8 needs
   ~1e5), exactly as the saturation argument predicts -- a nice internal
   consistency check.
3. **The cap is a continuum knob, not an integer quantizer.** Converged values
   are non-integer (cap 8 settles at ~3.6, not 4). The dimension is stable and
   tunable but not quantized -- which sharpens the "emergence with a knob"
   caveat rather than softening it.

This also lets us begin to *extrapolate* a crossover: the "resolves only above
N ~ X" scaling per cap is the first real number to replace the 100K hope.

### Quantified barrier: done (step 2) -- a measured obstruction exponent

`barrier_scaling.py` measures achieved extent (double-sweep diameter on the
largest component) vs rewiring steps and vs N, starting each rewiring rule from
a random (expander) graph, with a 2D `lattice` as positive control and the
unrewired random graph as baseline. Result (N to 1.6e4, 3 seeds, 200 steps,
mean degree 6), fitting extent ~ N^alpha:

| series | alpha (extent~N^α) | reading |
|--------|--------------------|---------|
| `lattice` (positive control) | **0.515** (R²=1.00) | true 2D: α = 1/2, as it must |
| `none` (expander baseline) | **0.083** | diameter ~ log N |
| `triadic` | 0.127 | barrier (noisy fit, R²=0.53) |
| `geometrize` | 0.136 (R²=0.95) | barrier |
| `ricci` | 0.147 (R²=0.99) | barrier |
| `grown` (growth, for contrast) | 0.194 | polynomial but globally compressed |

**The barrier is now a number.** All three fixed-N rewiring rules sit at
**alpha ~ 0.13**, right next to the expander baseline (0.08) and nowhere near
the 2D control (0.51). Two supporting observations:

- **vs steps:** at N=1.6e4 the rewiring rules move the diameter only from ~10 to
  ~16 over 200 steps while the lattice reference sits at 250 -- they *crumple*,
  they do not *unfold*. The one-time **growth factor is ~1.3-1.5x** (a constant
  stretch), and it does not scale: alpha stays ~0.13 as N grows.
- **honest caveat:** the rewiring alpha (~0.13) sits slightly *above* the pure
  expander baseline (0.08), but this is partly a *thinning* artifact --
  `geometrize` sheds edges, and lowering mean degree alone raises diameter. So
  even the small excess is not geometry-building. The conclusion stands and
  *strengthens*: none of the modest stretch is manifold formation.

So "fixed-N local rewiring cannot grow extent" is no longer an assertion over
three samples -- it is a measured exponent (alpha ~ 0.13 vs the required 0.5),
and the gap to a true manifold widens with N exactly as the log-N-vs-N^(1/d)
argument predicts. This is the flagship negative, with the obstruction rate
measured rather than asserted.

> *[superseded 2026-09-26: the diameter is logarithmic, and there is no local 2D.]*
> **Aside (a real surprise worth chasing):** `grown` is locally ~2D by
> ball-growth (`d_eff` ≈ 2.2) yet its *global* diameter scales only as
> ~N^0.19, far below the N^0.5 a flat 2D sheet would give. So `grown` is
> locally low-dimensional but globally compressed (small-world-like) -- its
> ball-growth (Hausdorff) dimension and its diameter (extent) dimension
> disagree. That split is itself a clean, unplanned finding and is exactly the
> kind of local-vs-global dimension mismatch the spectral-dimension study (step
> 4) is built to probe.

### FSS machinery: validated on the majority-vote / Ising transition

`ising_sweep.py` implements the full FSS pipeline (order parameter `M = <|m|>`,
susceptibility `chi = N Var(m)`, Binder cumulant `U`, and a 2D-Ising data
collapse) and validates it on a 2D lattice. Result (sides 16/24/32, seeds 4):

- the **Binder curves cross at a single point** and the **susceptibility peak
  grows with `L`**, locating `q_c ~= 0.08` -- consistent with the known
  square-lattice majority-vote value (~0.075; the small offset is open
  boundaries + modest `L`);
- the **data collapse** with 2D-Ising exponents (`beta/nu = 1/8`, `1/nu = 1`)
  pulls all three `L` onto one master curve.

So the machinery (Binder crossing, susceptibility scaling, collapse optimizer)
is correctly implemented and recovers known physics. **What this licenses is
the pipeline, not the exponents:** the 2D-Ising values were *recovered* here on
a system known to be in that class -- they must not be carried over as defaults
to `prune`, whose transition may be in a different class entirely (see the
universality-class caveat above). Assuming them on `prune` would be the
fit-anything trap.

**A non-obvious finding it surfaced:** the project's actual `majority` rule
(`rules.py`) is **not** `Z2`-symmetric -- it breaks ties with `argmax`,
deterministically toward state 0. On an even-degree lattice (grid degree 4)
2-2 ties are constant, so this acts as a *strong symmetry-breaking field*: under
the real rule the lattice stays ordered across the whole noise range and shows
no clean transition. The clean Ising validation therefore uses a textbook
`Z2`-symmetric majority-vote update (random tie-breaking, checkerboard sweep to
avoid synchronous period-2 blinking on the bipartite lattice); `ising_sweep.py
--model project` reproduces the biased, transition-free behavior of the real
rule for comparison. Worth remembering when interpreting any `majority` result:
state 0 is weakly favored. See [[majority-rule-tie-break-bias]].

### prune dimensional onset: done (step 3) -- a continuum knob, not a transition

> *[partly superseded 2026-09-26]* "Not a transition" stands. "A dimension knob 1 → 2"
> does not: only p ≲ 0.1 gives a real dimension (d = 1). See the audit.

Step 3 was meant to turn the validated FSS pipeline on `prune` (shortcut-density
→ dimensional onset) and extract the critical exponents. A scout overturned the
premise before any collapse was attempted, and *that is the result*.
`prune_dimension.py` prunes a Watts-Strogatz ring (k=6) to convergence across
rewire probability `p` × N × seeds and measures the pruned graph's emergent
dimension.

**There is no sharp dimensional onset.** `defined_frac` stays ~1 for every `p`
and N -- pruning always yields a *defined* dimension, so the "switches on at a
`p_c`" picture is simply wrong. What `p` does instead is tune the dimension
*continuously* (N = 3.2e4, seed-averaged):

| p | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 |
|---|------|------|------|------|------|------|------|------|------|
| median `d_eff` | 1.00 | 0.98 | 1.04 | 1.52 | 1.91 | 2.22 | 2.26 | 2.16 | 2.07 |
| clustering     | 0.58 | 0.57 | 0.57 | 0.55 | 0.47 | 0.34 | 0.21 | 0.10 | 0.03 |

The pruned graph slides from a near-pure **1D ring** (`d≈1`) toward a **2D mesh**
(`d≈2`, saturating ~2.1) as `p` rises. Mechanism: `prune` strips zero-triangle
shortcuts to convergence, peeling the graph back to the high-overlap backbone
that survived rewiring; at low `p` that backbone is a clean ring, and as `p`
rises the rewired-in edges that happen to land in triangles cross-link it into a
more 2D fabric. So `prune` + shortcut-density is a **third continuum dimension
knob**, alongside the `grown` degree cap -- and like that one it is a continuum,
not an integer quantizer, and it tops out near 2.

**It is a crossover, not a critical point -- quantified, not asserted.** Two
numbers kill the transition reading (`prune_dimension.py` reports both):

- **N-independence.** The d(p) curve is essentially identical at N = 2e3, 8e3,
  3.2e4 (max over `p` of the across-N spread = **0.063**). A genuine transition
  would keep shifting/sharpening with N; this one has already converged.
- **Non-sharpening.** The peak slope `max_p |dd/dp|` is **flat across 16× in N**
  (4.80, 4.97, 4.79 at N = 2e3 / 8e3 / 3.2e4). This is the dual of a diverging
  susceptibility: at a real critical point the steepest response grows without
  bound as N → ∞; here it does not move. Nothing for finite-size scaling to
  latch onto -- a forced data collapse would have been exactly the fit-anything
  trap the universality caveat warned about.

**The dimension is real geometry, not a low-degree artifact** (`--validate-real`).
At high `p` the pruned graph is sparse (mean degree → ~2.5), so the `d≈2` reading
needs a control: an Erdős–Rényi graph at the *same* mean degree.

| p | pruned-WS | ER at matched degree |
|---|-----------|----------------------|
| 0.1 | deg 5.3, clustering **0.56**, defined **1.00**, d≈1.0 | clustering 0.001, defined **0.00** (expander) |
| 0.5 | deg 2.8, clustering **0.35**, defined **0.97**, d≈2.1 | clustering 0.000, defined 0.53, d≈5.6 (garbage) |

At identical degree the pruned graph has high clustering and clean power-law ball
growth while the ER control is an undefined expander -- so the dimension is a
property of the *pruned structure* (which `p` controls), not of the degree.

**Honesty caveat on the high-`p` end.** The clean tunable regime is `p ≲ 0.5`,
where clustering stays ≥ 0.34. Beyond `p ≈ 0.6` clustering collapses toward
ER-like (0.03 at p=0.8) even as `d_eff` plateaus at ~2; that plateau sits on
thinning, weakly-clustered structure and should be read as "saturates near 2,"
not as clean 2D geometry. The knob is sharpest and most clearly geometric in the
ring→mesh band.

So step 3 is a paired result: a **positive** (prune is a tunable-dimension
generator, the third in the program) and an honest **negative** on the original
question (the dimensional onset is a crossover, not a phase transition -- no
critical exponents to extract, established by a *measured* non-divergence rather
than a failed fit). Reproduce: `python prune_dimension.py --validate-real` then
`python prune_dimension.py`.

## Portal experiments: shortcuts vs. geometry

*Run 2026-07-16 for the exotic-transport program (umbrella: `../exotic-transport/00-fence/`, lattice row Q3), but standing alone as graph-graph findings. A "portal" is an injected long-range edge -- the graph skeleton of a wormhole. Program grading: internally A (seeded, controlled, N-swept), externally C (toy model class).*

### 1. Tolerance: geometry doesn't shatter under portals -- it inflates

Inject k random shortcuts (endpoints >= 2*r0+1 apart at injection) into a coherent
`grown` base (cap 6, d ~ 2.2); measure the d_eff field at the **fixed** radius r0
calibrated on the k=0 base (auto-recalibrating would shrink the radius as the
diameter collapses and confound the measurement). N = 2000/5000/10000, 3 seeds,
k = 0..400. (`shortcut_tolerance.py`)

The a-priori damage model -- each portal endpoint corrupts the balls within r0 of
it, so `defined_frac` should fall like 1 - c*(2k*B_r0/N) -- is **refuted**:
`defined_frac` stays 0.97-1.00 across the entire sweep, at every N, even at k=400.
What actually happens (N=5000, 3-seed means):

| k | d_eff | Moran I | z | mean pair dist |
|---|-------|---------|---|----------------|
| 0 | 2.21±0.53 | 0.881 | 89 | 20.3 |
| 20 | 2.30±0.52 | 0.872 | 89 | 19.1 |
| 100 | 2.65±0.45 | 0.809 | 80 | 16.5 |
| 400 | 3.19±0.32 | 0.668 | 66 | 12.9 |

- **Dimension inflation.** Ball growth through portals reads as *extra
  dimensions*, not as noise: d_eff drifts 2.21 -> 3.19 while the field stays
  defined nearly everywhere and stays whole-field COHERENT by the coherence.py
  criteria even at k=400. A wormhole-riddled geometry measures as a
  higher-dimensional coherent geometry -- up until (at `rewire`-regime densities)
  growth saturates and d goes undefined. This closes the gap between two earlier
  findings: `rewire` destroys dimension not by locally breaking the power law at
  low densities, but by inflating d until the regime gate fails.
- **Ordering of damage.** The metric collapses first (pair distance 20.3 -> 12.9,
  the small-world effect), dimension inflates second, coherence erodes third
  (I 0.881 -> 0.668), definedness essentially never at these densities. The
  portal damage is *anisotropy* of the field (portal neighborhoods read high-d
  against a low-d background), not undefinedness.
- **Portal capacity.** At N=2000 placement itself saturates near k ~ 200: every
  portal shrinks the metric, until NO pair of nodes is far enough apart to host
  another far portal (see `k_placed` in the CSV). A small geometry has a bounded
  budget of genuinely-far portals.

### 2. Censorship: threshold and advantage-blind -- and weak self-stabilization is real

40 long portals (advantage = distance at injection, >= 6) + 20 detour-2 controls
into grown N=2000; run `prune` / `ricci` / `triadic`+`prune` for 120 steps at
prob 0.05; track per-portal removal step. 3 seeds. (`shortcut_censorship.py`)
Two a-priori predictions were recorded in the module docstring before running:

**P1 (threshold, not graded) -- CONFIRMED.** Any portal between nodes at distance
>= 3 has zero common neighbors *by definition*, so `prune` cannot see how much
advantage an edge carries, only that it is unembedded:

| condition | long survival | detour-2 survival | mean removal step | rank corr(adv, t) | collateral |
|-----------|---------------|-------------------|-------------------|--------------------|------------|
| prune | 0.03 | 1.00 | 18.1 (geom. ~20) | -0.07 | 0.000 |
| ricci | 0.03 | 1.00 | 23.4 | +0.11 | 0.000 |
| triadic+prune | 0.12 | 0.22 | 19.4 | -0.08 | 0.584 |

Detour-2 portals are immune, long portals die at the base geometric rate
*uncorrelated with advantage*, and the censorship is surgical (zero collateral on
the grown fabric, whose edges all sit in triangles).

**P2 (self-stabilization) -- WEAKLY CONFIRMED, with a twist.** With `triadic`
running alongside `prune`, long-portal survival rises 3% -> 12% and ~2 of 40
portals per run end *woven in* (triangles formed ON the portal edge -> permanently
prune-immune): triadic closure operating *through* the portal lays parallel edges
and legitimizes it, exactly as predicted. The twist: triadic is a double agent --
it also displaces portals wholesale (detour-2 survival collapses 100% -> 22%) and
churns 58% of the base fabric. A portal CAN be stabilized against the censor by
local dynamics, but the stabilizer is a worse threat to any individual edge than
the censor is.

### 2b. The same censorship under async updates (step 4, 2026-08-03): P1 schedule-invariant, P2 survives essentially unchanged

The first *physics* checkpoint of the Lorentzian ladder (LORENTZIAN_SPIKE.md §5-6):
re-run §2 with **event-driven Poisson-clock** updates instead of synchronous sweeps and
ask whether emergent time changes the verdict. Identical injected base+portals per seed
fed through **both** schedules; one shared reduction (`summarize_portals`); the sync side
is `shortcut_censorship.run_condition` verbatim. Sweep-equivalent time = absolute Poisson
time, both clocks at rate 1 (equal opportunity; `prune_prob`/`rewire_prob`=0.05 live
inside the events). (`async_censorship.py --validate`: grown N=2000, 40 long + 20
detour-2, 120 sweeps, 3 seeds; frozen gate at N=1200/5 seeds agrees.)

**P1 (threshold + advantage-blind) -- SCHEDULE-INVARIANT.** Async prune reproduces the
synchronous P1 within noise at both scales (N=1200/5-seed shown):

| observable | sync | async | z |
|---|---|---|---|
| long survival | 0.020 | 0.025 | 0.34 |
| detour-2 survival | 1.00 | 1.00 | 0.00 |
| mean removal (steps / sweep-equiv) | 20.6 | 21.3 | 0.44 |
| collateral | 0.000 | 0.000 | 0.00 |
| rank(advantage, t) | -- | +0.013 | -- |

Long portals still die at the base geometric rate (≈1/prune_prob=20), detour-2 still
immune, removal still uncorrelated with advantage (|rank|<0.02), collateral still zero.
This is the anchor gate **and** the control: async's one-at-a-time concurrency -- including
the degree-floor coupling that made async ≠ sync *trajectory-by-trajectory* in step 2 --
does not move the P1 observables, so any Gate-2 difference is attributable specifically to
the triadic/prune interleaving, not to asynchrony itself.

**P2 (self-stabilization) -- SURVIVES async, and is schedule-invariant in magnitude.**
Pre-registered both ways before the run: weaving persists (genuine dynamics) vs weaving
collapses (a synchronous lockstep artifact). It **persists**. A **40-seed paired** estimate
(N=1200, triadic+prune, independent seed set via `async_censorship_paired.py --seeds 40`,
2026-08-11) is the authoritative comparison -- the 3-5 seed `--validate` gate is too noisy
to quantify the difference:

| observable | sync | async | paired diff (sync-async), t |
|---|---|---|---|
| long survival | 0.126 ± 0.009 | 0.129 ± 0.008 | -0.003 ± 0.011, t=-0.28 |
| woven-in (of 40) | 2.98 ± 0.23 | 3.00 ± 0.23 | -0.03 ± 0.29, t=-0.08 |

Async long-portal survival (0.129) sits well above the async prune-only baseline (0.03) and
portals still get woven in (3.0/run), so triadic-through-a-portal self-stabilization is
**genuine emergent-time dynamics, not an artifact of the synchronous triadic-then-prune
lockstep** -- and its magnitude is **schedule-invariant**: long survival ratio 1.02
(t=-0.28) and woven ratio 1.01 (t=-0.08), with async below sync in only 16-17 of 40 seeds.
(A 12-seed first pass had hinted at a weak woven dip, ratio 0.81 at t=1.77; the 40-seed run
shows that hint was itself noise.)

**Correction of the preliminary read (kept in the open, not quietly fixed).** The first-pass
`--validate` gate runs (5 seeds at N=1200, 3 at N=2000) showed woven 2.6 vs 4.4 and 1.33 vs
3.67 -- a spurious "~2x attenuation" that did **not** survive the 12-seed paired analysis. It
was small-sample noise; the honest conclusion is schedule-invariance, not attenuation. A
*possible* weak mechanism was mooted -- async interleaving can prune a fresh weaving edge
before its protective triangle closes, whereas a synchronous step lays and closes it before
that step's prune pass -- but the 40-seed run settles it: no detectable effect (woven
t=-0.08); even the 12-seed t=1.77 hint of it was noise. (Methodology lesson, reused: don't
quantify a sync-vs-async delta off a 3-5 seed gate; the gate certifies P1, a paired sweep
quantifies P2.)

**Checkpoint verdict.** Unlike the step-3 causal-calibration negative, this first physics
result is a clean **positive**: the *entire* banked censorship result -- threshold,
advantage-blindness, **and** the magnitude of the weak self-stabilization -- survives when
time is made emergent, essentially unchanged. Reproduce: `python async_censorship.py
--validate` (gate); `python async_censorship_paired.py` (the paired P2 estimate).

### 3. Walkers: the quantum walker is the portal's best customer

> *[superseded 2026-09-29: on a substrate with a dimension neither walker cares much
> about an offset portal -- see "Portals on substrates with a dimension".]*

grown N=1500, source-target distance 24.4±2.6; conditions: none / offset portal
(ends ~2 hops from source and target) / direct source-target edge. Classical:
exact absorbing-walk median hitting time. Quantum: CTQW (H = adjacency) peak
transfer probability and its time. 5 seeds. (`shortcut_walkers.py`)

The a-priori expectation -- a lone portal is a weak link whose quantum benefit is
interference-limited -- is **refuted in the interesting direction**:

| condition | classical median t | quantum peak P | quantum peak t |
|-----------|--------------------|----------------|----------------|
| none | 21,199 | 1.03e-05 | 61.8 |
| offset | 3,226 (x6.6) | 2.61e-02 (x2,534) | 26.9 |
| direct | 78 (x272) | 1.84e-01 | 0.8 |

The offset portal buys the classical walker x6.6 in transit time but buys the
quantum walker **two to three orders of magnitude** in peak transfer probability
(and x2.3 in arrival time). Baseline coherent transfer to one specific far node
of an irregular graph is essentially nil (amplitude dilutes over the whole
graph); the portal creates a channel that the ballistic walker exploits far
better than diffusion does. In this model class, if you build a wormhole, the
thing that most wants to go through it is a quantum excitation.

Caveats: absolute quantum transfer stays small (~1% peak through the offset
portal); the direct-edge condition is a degenerate control (adjacency, not
transport).

### 3b. Laplacian cross-check: the gain is real, the number was not (2026-07-18)

The open follow-up was that an adjacency-generated CTQW on an irregular graph
conflates degree with interference. Done: `shortcut_walkers.py` now takes
`--generators adjacency laplacian` (H = A or H = D - A) and runs both on the
**same** graphs and source/target pairs. On a regular graph the two are the same
experiment -- L = kI - A differ by a global phase and a time reversal, both of
which drop out of |<target|psi(t)>|^2 -- so on irregular `grown` any difference
between them is exactly the degree effect in question. N=1500, 20 seeds:

| t_max | generator | none | offset | offset gain | direct gain |
|-------|-----------|------|--------|-------------|-------------|
| 3d | adjacency | 1.78e-05 | 1.06e-02 | x596 | x7,153 |
| 3d | laplacian | 2.48e-06 | 6.29e-03 | **x2,534** | x41,950 |
| 8d | adjacency | 5.47e-05 | 9.57e-03 | x175 | x1,530 |
| 8d | laplacian | 9.47e-06 | 6.01e-03 | x635 | x9,560 |

(Classical is generator- and t_max-independent: offset x6.9, direct x294.)

**The cross-check passes.** The quantum portal advantage is not a degree
artifact: switching to the Laplacian generator makes it *larger*, not smaller,
at every horizon. The finding -- quantum gain exceeds classical gain (x6.9) by
two to three orders of magnitude -- survives.

**But the specific number x2,534 is retracted as a quantity.** Two problems, both
found by instrumenting rather than by re-deriving:

1. **The gain is t_max-dependent, because the baseline is.** The no-portal peak
   is a running *maximum* of a diffuse, wandering amplitude over the window
   (0, t_max], so it grows as the window grows (adjacency 1.78e-05 -> 5.47e-05
   when t_max goes 3d -> 8d) while the offset peak stays flat. The ratio
   therefore shrinks ~3x for a 2.7x longer window, and the baseline still has not
   converged (3/20 and 7/20 seeds peak at the grid edge even at 8d, now flagged
   by the driver). Any single gain figure is a statement about an arbitrary
   horizon. *The x2,534 reappearing in the 3d-Laplacian row is a coincidence,
   not a reproduction.*
2. **The original run's offset placement was RNG-coupled to the solver.**
   scipy's `expm_multiply` estimates operator norms with `onenormest`, which
   draws from the global numpy RNG -- so a quantum solve in the `none` condition
   shifted which offset portal got placed later in the same seed. Placement is
   now decided before any solver call; verified by `--generators adjacency`
   alone reproducing the paired run bit-for-bit. Across placements the offset
   peak moves by ~2x, which at 5 seeds is most of the original headline.

The durable statement is the *ordering and separation* -- classical O(1), quantum
O(10^2-10^3), robust across both generators and both horizons -- not any single
ratio. A better-defined observable is the honest follow-up; done below.

### 3c. Horizon-free observable: a portal is kinetic classically, structural quantum-mechanically (2026-07-18)

> *[superseded 2026-09-29]* The quantum gain below is a property of localization on the
> tree-like `grown` substrate: on the triangular lattice and on `sheet` the same offset
> portal gives a Pbar gain of 1.0x, because the no-portal walker is already at
> equipartition. See "Portals on substrates with a dimension".

Replaced the max-over-a-window with the **infinite-time average**, the standard
CTQW observable with no free time parameter:

    Pbar = lim_{T->inf} (1/T) \int_0^T |<target|exp(-iHt)|source>|^2 dt
         = sum over DISTINCT eigenvalues l of ( sum_{k: l_k = l} <t|phi_k><phi_k|s> )^2

Computed exactly from the eigendecomposition -- no cutoff, no integration.
Degenerate eigenvalues must be grouped (states sharing an eigenvalue never
dephase relative to each other), which is the easy thing to get wrong, so
`shortcut_walkers.py --validate` checks it against independent Krylov
propagation with a **degeneracy-blind control**: on the star K_1,5 the blind sum
gives 0.180 against the true 0.060, and on the Laplacian C_6 it gives 0.194
against 0.278, so the grouping is genuinely under test rather than merely
tolerated. K_2 reproduces the analytic 1/2 to 2e-16.

The matched classical counterpart is the stationary occupancy
`pi(target) = deg(target)/2|E|` -- same units, same question (what fraction of
the long run is spent at the target), also horizon-free. N=1500, 20 seeds:

| condition | pi(target) | pi gain | median t_hit | t_hit gain | Pbar (adj) | Pbar*N | gain | Pbar (lap) | gain |
|-----------|-----------|---------|--------------|------------|------------|--------|------|------------|------|
| none | 3.337e-04 | 1.00x | 21,482 | 1.0x | 4.39e-05 | 0.07 | 1.0x | 1.06e-05 | 1.0x |
| offset | 3.336e-04 | **1.00x** | 3,124 | 6.9x | 2.10e-03 | 3.15 | **47.9x** | 1.39e-03 | **131.2x** |
| direct | 5.003e-04 | 1.50x | 73 | 294.3x | 1.22e-02 | 18.29 | 277.8x | 1.83e-02 | 1731.6x |

**The finding survives the better observable, and sharpens into a qualitative
statement.** A portal is two different kinds of object depending on who uses it:

- **Classically it is a *kinetic* device.** It changes how fast you arrive
  (hitting time x6.9) and *nothing* about where you end up: long-run occupancy
  is unchanged to four digits. Be explicit that this is analytic, not empirical
  -- `pi` is degree-determined, so any edge not incident to the target leaves
  `pi(target)` fixed by construction (the offset row's 1.00x is really 0.9997,
  the |E| -> |E|+1 dilution). The direct row's 1.50x is deg(target) 2 -> 3,
  bookkeeping rather than transport.
- **Quantum-mechanically it is a *structural* device.** It changes long-run
  transfer by ~50x (adjacency) to ~130x (Laplacian), because the time average is
  set by eigenvector overlaps rather than by degree. Read in equipartition units
  (`Pbar*N`, where 1.0 = amplitude spread uniformly), the portal moves the far
  target from **15x below equipartition (0.07) to 3x above it (3.15)**.

That asymmetry *is* the physics: classical diffusion equilibrates to a purely
local statistic and therefore cannot see a distant shortcut in its long-run
distribution, while unitary evolution never equilibrates to a local statistic at
all and retains the portal in its spectral structure forever.

Two corrections to the record that this observable forces:

- **The peak-based gains were inflated ~10x.** Offset gain is 47.9x horizon-free
  vs 596x by peak-at-t_max=3d (adjacency); 131x vs 2,534x (Laplacian). The
  earlier retraction was right, and the direction of the error is now measured.
- **The original headline compared different quantities.** "x6.6 classical vs
  x2,534 quantum" set a ratio of *times* against a ratio of *probabilities*.
  Matched now: on occupancy it is x1.00 vs x48-131; on speed it is x6.9
  classical (the quantum side has no horizon-free analogue of hitting time
  without a measurement model -- left open).

The Laplacian cross-check verdict is unchanged and now rests on a well-defined
quantity: the gain is larger under H = L (131x vs 48x), so it is not a degree
artifact of H = A.

Reproduce: `python shortcut_tolerance.py --nodes 2000 5000 10000 --seeds 3`,
`python shortcut_censorship.py --nodes 2000 --seeds 3`,
`python shortcut_walkers.py --validate` then
`python shortcut_walkers.py --nodes 1500 --seeds 20`.
(Pbar costs a dense eigendecomposition, O(N^3) -- N is capped at a few thousand.)
