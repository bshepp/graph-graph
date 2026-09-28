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
(6 -> 9 -> 13). The clincher: **the core at onset is ~2.6 strands at every capacity**
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

**A limit of the window audit, found here.** It labels beta = 1 "DIMENSION DEFINED"
(drift 0.09). It probes radii up to 40; the beta = 1 graph is two-dimensional at that
scale and tree-like above it. The audit certifies the scales it looks at and no more.
At beta = 2 the large-scale evidence is the diameter and boundary exponents and the
walk, not the audit.

Reproduce: `python sheet_growth.py --validate`; `python sheet_growth.py --betas 0 1 2 3
--nodes 512000 --seeds 3 --seed 100 --jobs 7 --save-graphs`; then `python
sheet_growth.py --judge results/sheet_c6_b2_s100_N512000.npz`.

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
