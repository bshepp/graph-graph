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
| `grown` (generator) | degree-capped frontier growth | **tunable emergent d**: cap 6→2.1, 7→2.7, 8→2.9 | cap forces outward growth → large diameter |

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
  spontaneously preferred dimension.

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
clumped small-world fixed point. The obstruction is mechanism-independent:

> **Local rewiring cannot grow *extent* (diameter) from an expander.** It can
> create local triangles, but those crumple into the existing short-diameter
> structure instead of unfolding into an extended manifold. Growing the
> diameter would require removing shortcuts faster than the graph
> re-localizes, which either fragments it or stalls.

So, under fixed-N local rules: **dimension is bistable, not attracting.
Geometry must be *seeded* -- grown outward (the `grown` generator, which
never passes through an expander) or already latent (a small-world ring) --
it does not spontaneously condense from maximal disorder.** This maps onto
the "dimensionally incoherent" phase in DIMENSIONAL_COHERENCE.md (the
dark-matter analog): a stable, non-geometric phase that local dynamics
cannot escape.

**Scope:** strong, mechanism-independent evidence across four local rewiring
rules plus a growth generator -- not a formal proof. An Ollivier-Ricci flow
(optimal-transport curvature) is the one untested variant, but the obstruction
observed is about *extent*, not the curvature measure, so the same barrier is
expected. The clean way to *get* a chosen dimension remains the `grown`
generator (build it geometrically; the degree cap tunes d).

## Open threads

- **Curvature flow** vs the bootstrapping barrier: tested (`ricci`) -- hits
  the same barrier (above). Ollivier-Ricci (optimal-transport) is the one
  untested variant but is expected to behave the same (the obstruction is
  extent, not the curvature measure).
- **Spatial coherence:** Moran's I on the `d_eff` field, to confirm emergent
  structure forms coherent phases rather than scattered noise.
- **Robustness / scale:** does `grown`'s cap→dimension law hold across N and
  seeds, and at the 100K+ scale the theory says emergence may require?
- **Preservation of emergent structure:** do the dynamic rules preserve or
  erode a `grown` graph's dimension (as they do for the lattice)?

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
   this. **This is the "including random walks" path.**
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

### Recommendation / sequencing

Start with the **`majority`/Ising FSS validation** (proves the scaling
machinery on known physics, cheap), then the two compute-justified pushes in
parallel as budget allows: **`prune` phase-transition FSS** and **spectral-
dimension flow on `grown` graphs**. The `cap -> d` scaling check is a cheap
prerequisite that can ride along with either.

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

So the machinery recovers known physics and is trustworthy to point at `prune`.

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
