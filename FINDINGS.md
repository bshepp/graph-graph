# Findings: Emergent Dimension from Local Graph Rules

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
| `grown` (generator) | degree-capped frontier growth | **tunable emergent d** (converged at scale): cap 6→~2.2, 7→~3.0, 8→~3.6 | cap forces outward growth → large diameter |

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
- `grown` is **strong emergence**: dimension that was never latent (grown
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
- **Robustness / scale:** does `grown`'s cap→dimension law hold across N and
  seeds, and converge to clean integers? (Cheapest win; prerequisite for
  trusting anything at large N -- see Scaling directions.)
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
   (self-stabilization) survives emergent time essentially unchanged (12-seed
   paired: no significant attenuation -- the preliminary "~2x" was small-sample
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
   independent sets -- verified, ~4-5 rounds per sweep) and `dimension.py`'s
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
