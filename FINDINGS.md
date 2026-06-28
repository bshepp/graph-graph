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

> **On "emergence may require 100K+ nodes"** (DIMENSIONAL_COHERENCE.md, Phase
> 5): that is a *hope by analogy to thermodynamics, not a derived crossover
> scale.* No calculation predicts where (or whether) any of these transitions
> sharpens. The cap→d and Ising-pipeline runs on local hardware are exactly
> what would let us *extrapolate* a real crossover estimate -- and that
> extrapolation is the gate to clear before reserving any large compute.

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
