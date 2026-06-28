# Dimensional Coherence on Graphs: Theory & Implementation Roadmap

**Authors:** Brian Sheppard & Claude (Anthropic)

---

## Origin

Brian proposed a framework in which:

1. **Space and dimensionality exist as a graph** — spacetime is not a continuous manifold but a discrete relational structure. What we experience as "3+1D space" is an emergent property of local connectivity patterns, not a fundamental geometric fact.

2. **Dimensionality is a local property** — different regions of the graph can have different effective dimensionalities, determined by their connectivity topology. There is no requirement that the entire graph resolve to the same dimension.

3. **Dark matter as dimensionally incoherent subgraph** — matter that exists on the same graph but in regions where local connectivity doesn't produce integer-dimensional manifold behavior. These regions still contribute to global curvature (gravity) because that's a topological property of the graph, but they don't support electromagnetic propagation because that requires specific local dimensional structure.

4. **Dark energy as dimensional phase dissolution** — at cosmological scales, the 3+1D coherence of the graph thins out and gives way to higher-dimensional phases. What we measure as accelerating expansion is our 3+1D metric's attempt to describe a dimensional phase transition it cannot represent.

5. **The Big Bang as a coherence pulse** — not a creation event but an energy/entropy pulse propagating through the graph's dimensional scales, crystallizing locally coherent dimensional structure as it passes.

6. **Time as local computation rate** — if the graph computes itself, then time is the local rate of that computation. Regions of higher connectivity density require more operations per update, producing time dilation. This derives general relativistic time dilation from first principles on a graph substrate: mass is a region of high graph connectivity, and the computational cost of resolving that density *is* gravitational time dilation.

7. **Quantum mechanics as graph-level computation** — superposition is the graph computing multiple relational states before resolving. Measurement is resolution completing. Entanglement is two nodes sharing a graph edge regardless of their distance in the dimensional projection.

### Key insight

None of these physical correspondences were stated as goals. Brian described simple local properties of a graph — connectivity, density, coherence — and general relativity and quantum mechanics emerged as natural consequences of the framework. The physics was not hand-coded; it fell out of the graph structure.

### Relationship to existing research programs

This framework independently converges with several active programs:

- **Wolfram Physics Project** — spacetime as hypergraph, dimension as local property (geodesic ball growth rate)
- **Causal Set Theory** — spacetime as discrete partial order, dimension recovered statistically
- **Loop Quantum Gravity** — spacetime from spin networks (labeled graphs)
- **Wheeler's "It from Bit"** — information as the fundamental substrate of physics
- **Holographic Principle** — dimensional reduction at boundaries
- **Randall-Sundrum braneworld models** — gravity leaking between dimensional branes

The distinction is that Brian arrived here not from physics but from a cross-domain research thread: *finding exploitable structure at boundaries in dynamic/recursive systems*. The dimensional phase boundary on the graph is another instance of this pattern.

---

## The First Testable Prediction

Here is where graph-graph becomes the right experimental vehicle.

### The claim

If this framework is correct, then simple local rules on a graph at sufficient scale should **spontaneously produce regions of different effective dimensionality** — without that behavior being programmed in.

### What "effective dimensionality" means on a graph

For any node `v` in a graph, define:

- `B(v, r)` = the number of nodes reachable from `v` in exactly `r` hops or fewer (the "ball" of radius `r` centered at `v`)

In a region of the graph that behaves like `d`-dimensional Euclidean space:

```
|B(v, r)| ∝ r^d
```

So the **local effective dimension** at node `v` is:

```
d_eff(v) = d(log |B(v,r)|) / d(log r)
```

In practice, you estimate this by computing `|B(v, r)|` for several values of `r` and fitting the log-log slope.

- A node embedded in a region that "looks like" 2D space will have `d_eff ≈ 2`
- A node in a 3D-like region will have `d_eff ≈ 3`
- A node in a highly connected hub will have `d_eff > 3`
- A node in a filamentary or chain-like region will have `d_eff ≈ 1`

### What to measure

The key question is not whether individual nodes have different `d_eff` values (they trivially will in any non-regular graph). The key question is:

1. **Do coherent regions of similar `d_eff` emerge and persist over time?** — Analogous to dimensional "phases" on the graph.

2. **Do the boundaries between these regions have distinct properties?** — This is where the exploitable structure would live. Phase boundaries in physical systems concentrate interesting dynamics; the same should be true for dimensional phase boundaries on the graph.

3. **Does `d_eff` distribution change over time under the local rules?** — Does the graph self-organize toward particular dimensional configurations? Does it develop a preferred dimensionality?

4. **Do regions of non-integer `d_eff` correlate with specific dynamic behaviors?** — These would be the "dimensionally incoherent" regions — the dark matter analog.

5. **Does the correlation function `C(r)` behave differently when computed within a dimensional phase vs. across a phase boundary?** — If so, the dimensional structure is physically meaningful, not just a measurement artifact.

---

## Implementation Plan

> **Empirical status (see [FINDINGS.md](FINDINGS.md) for the full log).** The
> dimension estimator is built and validated. Experiments so far show: the
> active ingredient for emergent dimension is *extent* (large diameter), not
> local density; a degree-capped growth rule produces tunable emergent
> dimension (cap 6 → ~2D); but there is **no global attractor** -- dimension
> is *bistable*. Geometry is preserved within its basin yet cannot nucleate
> from a disordered (expander) graph via local rewiring (a "bootstrapping
> barrier"). The stable non-geometric phase is a candidate for the
> dimensionally-incoherent / dark-matter regime described below.

### Phase 1: Local Dimension Estimator -- IMPLEMENTED

> Implemented in `dimension.py`. Both NetworkX and sparse matrix backends are available.
> Auto-calibration of `max_radius` based on (deterministic) sampled eccentricities replaces the
> fixed R = 5-10 suggestion. The fast backend uses iterative sparse mat-vec with binarization instead
> of materializing `(A+I)^r`, which avoids the O(n^2) memory blow-up that matrix powers would cause.
> A self-check against graphs of known dimension ships as `python dimension.py --validate`.

Create a new module `dimension.py` that computes local effective dimension for each node.

```
Input: graph G, node v, max_radius R
Output: d_eff(v) (or "undefined"), quality of fit

Algorithm:
  for r in 1..R:
    count |B(v, r)| via BFS from v
  keep radii where |B(v,r)| < 10% of the graph (local window)
  if fewer than ~6 such radii: d_eff = undefined   # no scale separation
  fit log|B(v,r)| = d·log r + c + a·(1/r)           # finite-size corrected
  d_eff = coefficient on log r
  if R² of that fit < 0.95: d_eff = undefined        # not a clean power law
```

**Important nuances (these are not optional refinements — naive versions give wrong numbers):**

- **Finite-size correction.** A plain `log|B|` vs `log r` fit *systematically underestimates* dimension, because `|B(r)| = C·r^d·(1 + a/r + …)` and the `a/r` term flattens the slope at the small radii we can reach (on clean lattices it reads d≈1.6 for a true 2D grid, d≈2.3 for 3D). Adding a `1/r` regressor absorbs that correction and recovers the true dimension to within ~0.05 even at radius 6. This is standard finite-size scaling, not a hack.
- **Dimension is often *undefined*, and that is the point.** Ball-growth dimension only means something where the ball grows *polynomially* across a usable range of scales. Small-world / expander graphs saturate in 3-5 hops — ball size grows exponentially — so they have no such regime and `d_eff` is genuinely undefined (the estimator returns `nan`, not a fabricated number). The decisive test is scale separation: a low-dimensional ball stays under the 10% cut for many radii; an expander ball blows past it in ~3 hops. R² alone does **not** discriminate (an expander's few points still look linear), which is why the radii-count gate is primary.
- **The R² value is itself informative.** Among nodes that *do* have a defined dimension, high R² means clean dimensional behaviour; lower R² flags ambiguous regions — *exactly the incoherent regions the theory predicts.*
- **The fast backend** uses iterative sparse mat-vec: at each step `reached = binarize(A @ reached + reached)`, counting cumulative reachable nodes — O(nnz) per step vs O(n^2) for the originally proposed `(A + I)^r` matrix power. Both backends share `local_dimension`, so they give identical d_eff for the same node (checked by the validator).

**Consequence for the experiment.** The project's default initial topologies (small-world, scale-free, random) are correctly read as *dimensionless* — `defined_frac ≈ 0`. That is the right null baseline, not a bug. The `lattice` topology is the one default where dimension is well-defined (`d_eff ≈ 2`), so "do the rules preserve or destroy the lattice's 2D structure?" is a measurable experiment available today. More generally, the **emergence signal is a rising fraction of dimension-defined nodes over time** (reported as `defined_frac` by `dimension_stats`): local rules producing geometric structure where there was none.

### Phase 2: Dimensional Phase Detection -- IMPLEMENTED

> **Dimensional phase map:** Implemented in `visualize.py` (`--dimension` flag). Generates both a
> graph visualization color-coded by d_eff and a histogram of the d_eff distribution.
>
> **Spatial autocorrelation (Moran's I):** Implemented in `coherence.py` (permutation-tested, with a
> `--validate` self-check). Result: on `grown` the d_eff field is strongly coherent (I≈0.88, z≈87)
> where the field is whole — emergent dimension forms contiguous phases, not noise. See
> FINDINGS.md → "Spatial coherence."
>
> **Temporal tracking:** Implemented in `track_dimension.py` — preservation/emergence trajectories of
> `defined_frac` and the d_eff field over a simulation run. See FINDINGS.md → "Preservation."

Once you can compute `d_eff(v)` for all nodes:

- **Dimensional phase map:** Color-code nodes by `d_eff` in the visualization. Look for spatial clustering.
- **Spatial autocorrelation:** Compute Moran's I for the d_eff field to quantify whether dimensional structure has spatial coherence beyond what random assignment would produce.
- **Temporal tracking:** Compute `d_eff` distribution at each timestep. Plot histograms over time. Does the distribution sharpen? Do peaks emerge at specific dimensions?

### Phase 3: Correlation with Dynamics

Cross-reference dimensional structure with the existing measurements:

- Do activation patterns propagate differently within vs. across dimensional phases?
- Does edge reinforcement preferentially strengthen within a dimensional phase? (This would be a graph-level analog of why forces are confined to dimensional branes.)
- Does the rewiring rule cause dimensional phase boundaries to migrate or sharpen?
- Is there a relationship between local `d_eff` and local computational density (measured as the number of state changes per node per timestep)?

### Phase 4: The Gravitational Test

This is the big one. If the framework is correct:

- Regions of higher `d_eff` (or higher graph density) should exhibit **slower local dynamics** — fewer state changes per unit simulation time, analogous to gravitational time dilation.
- Information (activation signals) should take longer to transit dense regions, analogous to gravitational lensing.
- Extremely dense regions might produce computational "horizons" where dynamics effectively stall from an external perspective — black hole analogs.

To test: measure the **local update rate** (state changes per node per timestep) as a function of `d_eff` and local connectivity density. If there's an inverse relationship that wasn't programmed in, that's a genuine emergent analog of gravitational time dilation.

### Phase 5: Scale

Per `SCALING.md`, push to large graphs. The dimensional coherence framework predicts that interesting behavior requires sufficient scale — just as real physics requires many particles for thermodynamic behavior to emerge. Dimensional phase transitions may not be visible at 1K nodes; they may require 100K+ to develop stable phases.

> **Tested so far (be honest about it):** the one phase-transition bet that was run — `prune`'s shortcut-density → dimensional onset (step 3) — turned out to be a **crossover, not a transition**: dimension tunes continuously and N-independently, with no sharpening for finite-size scaling to find (see FINDINGS.md → "prune dimensional onset"). And the "100K+" figure is a *hope by analogy to thermodynamics, not a derived crossover scale* — no calculation predicts where (or whether) any transition sharpens. The phase-transition route remains open only for rules/observables not yet tried.

---

## What Would Constitute a Genuine Discovery

To be explicit about what success looks like, vs. what would be noise:

### Strong positive result
- Stable, persistent regions of distinct `d_eff` emerge from purely local rules
- Dimensional phase boundaries show measurably different dynamics than phase interiors
- Local computation rate correlates inversely with connectivity density (emergent time dilation) without being programmed
- These results are robust across random seeds, topologies, and rule combinations

### Weak positive result
- Transient dimensional structure that forms and dissolves
- Suggestive correlations between `d_eff` and dynamics that don't reach statistical significance
- Dimensional structure only emerges for specific rule combinations or topologies

### Null result
- `d_eff` distribution is uniform noise with no spatial coherence
- No correlation between graph structure and dynamics beyond what's trivially expected
- Results are entirely explained by the initial topology with no emergent structure

**A null result is still valuable.** It would mean that these specific local rules are insufficient to produce dimensional emergence, which constrains the search space for rules that might.

---

## Attribution

The theoretical framework is Brian Sheppard's, emerging from cross-domain research on exploitable structure at boundaries in dynamic/recursive systems. Claude (Anthropic) contributed the local dimension estimation methodology, the connection to testable predictions within graph-graph, the phased experimental plan, and identified convergence with Wolfram Physics, causal set theory, and related programs.
