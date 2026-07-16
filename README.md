# graph-graph

**The Honest Experiment:** Do simple local graph rules at large scale produce interesting emergent behavior?

## Philosophy

Unlike other projects that hand-code physics and then "discover" it, this project starts with **only simple local rules** and observes what emerges -- Game of Life style.

- No hand-coded physics (classical or quantum)
- No spatial coordinates baked in
- No pre-programmed correlations
- Just nodes, edges, and simple update rules

If interesting structure, correlations, or dynamics emerge, that's a genuine discovery.
If nothing emerges, that's also a valid result.

The goal is emergence first, physics (maybe) later.

## Core Question

> Can simple local rules on a graph produce long-range correlations or structure that wasn't explicitly programmed?

## Program context

Since Jul 2026 this project also serves as the **emergent-locality branch** of the owner's
exotic-transport research program (umbrella: `../exotic-transport/00-fence/`; private GitHub:
`bshepp/the-fence`). In that program's terms it attacks lattice row Q3 ("are light cones
fundamental or emergent?"), and its measurements are graded **internally A, externally C**:
rigorous within this toy model class, speculative in their bearing on actual spacetime.
Nothing about the project's own emergence-first question changes.

## Status

**Core simulation working.** Rules, measurement, visualization, and a sparse fast path are all implemented. Ready for experiments.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. Core dependencies: `numpy`, `networkx`, `scipy`, `matplotlib`, `tqdm`.

## Project Structure

| File | Purpose |
|------|---------|
| `simulation.py` | Main simulation loop (NetworkX backend) |
| `simulation_fast.py` | Sparse-matrix simulation (10-50x faster) |
| `rules.py` | Local update rules (activation, reinforcement, majority, rewire) |
| `measure.py` | Analysis: correlation functions, agreement fraction, domain detection |
| `visualize.py` | Metric plots and graph-state visualization |
| `animate.py` | Animated dashboard: watch rules evolve the graph in real time |
| `traverse.py` | Animated graph *traversals*: walk diffusion (quantum vs classical) and geodesic ball growth (the dimension estimator's BFS) |
| `showcase.py` | Generate curated demo animations (one per rule + combos + traversals) |
| `sweep.py` | Parameter sweep with parallel execution and CSV export |
| `dimension.py` | Local effective dimension estimator (d_eff via geodesic ball growth) |
| `track_dimension.py` | Temporal dimension tracking: how dimensional structure evolves under the rules |
| `ising_sweep.py` | Finite-size-scaling driver: validates the FSS machinery (Binder cumulant, susceptibility, data collapse) on the majority-vote/Ising transition |
| `cap_dimension_scaling.py` | cap→dimension finite-size scaling for the `grown` generator: does `d_eff` plateau on a stable value as N grows? (it does — at non-integer values) |
| `barrier_scaling.py` | Quantifies the bootstrapping barrier: measures extent (diameter) vs rewiring steps and vs N, fitting extent ~ N^α — rewiring rules stay at α≈0.13 (log N) vs a 2D lattice's α=0.5 |
| `coherence.py` | Spatial coherence of the `d_eff` field via Moran's I (permutation-tested): is emergent dimension contiguous phases or per-node noise? (`grown` is coherent, I≈0.88) |
| `prune_dimension.py` | `prune` + shortcut-density as a tunable-dimension knob: sweeps rewire prob `p` × N, showing d slides 1→2 as a **crossover, not a phase transition** (N-independent; peak slope flat in N) |
| `FINDINGS.md` | Empirical log of the emergent-dimension experiments |
| `braket_walks.py` | Quantum walk analysis: matrix-based CTQW vs. classical walks (experimental; runs on core deps) |
| `SCALING.md` | Roadmap from 1K to 100M+ nodes |
| `DIMENSIONAL_COHERENCE.md` | Theory and roadmap for dimensional coherence measurements |

## Rules

All rules are **local** -- each node only sees its immediate neighbors. Rules can be combined freely.

| Rule | Key | What it does |
|------|-----|--------------|
| Activation spreading | `activation` | Active nodes spread activation to neighbors; active nodes may decay (SIS epidemic model) |
| Edge reinforcement | `reinforcement` | Edges between co-active nodes strengthen; all edges slowly decay (Hebbian learning) |
| Majority vote | `majority` | Nodes adopt the majority state of their neighbors; small noise prevents freezing |
| Random rewiring | `rewire` | Small probability of rewiring edges to random targets, creating small-world structure (destroys geometric/dimensional structure) |
| Shortcut pruning | `prune` | Remove "shortcut" edges (endpoints share no neighbors); grows diameter and can recover latent low-dimensional structure |
| Triadic closure | `triadic` | Rewire edges toward friends-of-friends; raises clustering (but clustering alone does not create dimension -- see `FINDINGS.md`) |
| Geometrize | `geometrize` | Homeostatic local rewiring toward a target degree; preserves geometry within its basin but does not nucleate it from disorder |
| Ricci flow | `ricci` | Forman-Ricci curvature flow (rewire shortcuts toward triangles); confirms the bootstrapping barrier -- raises clustering but cannot grow extent from an expander |

See [FINDINGS.md](FINDINGS.md) for what these rules do to *dimensional* structure (emergence experiments).

## Measurements

- **Correlation function** C(r): state correlation between nodes at graph distance r
- **Agreement fraction**: how often distant node pairs share the same state vs. random baseline
- **Domain detection**: coherent regions of same state, largest domain size
- **Clustering coefficient** over time
- **Largest connected component** fraction
- **Local effective dimension** d_eff(v): estimated from geodesic ball growth |B(v,r)| ~ r^d (see `dimension.py`)
- **Dimensional coherence** R^2: how cleanly each node fits a power-law ball growth

## Usage

### Single run

```bash
# NetworkX backend (up to ~10K nodes)
python simulation.py --nodes 1000 --steps 1000 --rules activation majority --seed 42

# Fast sparse backend (up to ~500K nodes)
python simulation_fast.py --nodes 50000 --steps 2000 --rules activation reinforcement --seed 42
```

### Analyze and visualize

```bash
python measure.py results/run_TIMESTAMP.pkl
python visualize.py results/run_TIMESTAMP.pkl
```

### Animate

```bash
# Watch activation + majority vote evolve live
python animate.py --nodes 300 --steps 400 --rules activation majority --seed 42

# Save as GIF
python animate.py --nodes 500 --steps 600 --rules activation reinforcement --save anim.gif

# All four rules, scale-free topology, 20 fps
python animate.py --nodes 400 --steps 500 --topology scale_free --rules activation reinforcement majority rewire --fps 20 --save emergence.gif
```

The animation shows a dark-themed dashboard with the graph on the left (active nodes glow orange, edges thicken between co-active pairs) and metric sparklines on the right.

### Showcase animations

```bash
python showcase.py             # generate all 7 showcase GIFs → showcase/
python showcase.py --pick 1 3  # just epidemic + majority-vote
python showcase.py --list      # describe all available showcases
```

Seven pre-tuned animations, one for each rule (epidemic spreading, Hebbian reinforcement, majority-vote domains, small-world rewiring), one per notable topology (scale-free hubs, random baseline), and one combining all four rules to show full emergence. Three more (#8-10) are *traversal* showcases driven by `traverse.py` (see below).

### Graph traversals

Where `animate.py` shows the *rules* evolving a graph's state, `traverse.py`
animates a *traversal* of a fixed graph -- either a walk spreading from a seed
node, or the geodesic ball the dimension estimator grows. It takes a
`results/*.pkl` (traverse a real run's final graph) or builds one from
`--topology`.

```bash
# Walk diffusion: classical (diffusive) vs quantum (ballistic) side-by-side
python traverse.py --mode walk --topology grown --nodes 600 --seed 0 --save walk.gif

# Ball growth: the estimator's BFS, with a live log|B| vs log r panel that
# calls the real local_dimension() -- watch d_eff lock in at ~2 on a lattice
python traverse.py --mode ball --topology lattice --nodes 2500 --seed 0 --max-radius 12 --save ball.gif

# ...or never define on an expander (ball engulfs the graph in a few hops)
python traverse.py --mode ball --topology random --nodes 2000 --seed 0 --max-radius 8

# Traverse a completed simulation instead of a fresh topology
python traverse.py results/run_TIMESTAMP.pkl --mode walk
```

The spring layout is for display only -- no node coordinates feed any rule, so
the no-baked-in-geometry invariant holds. The ball-growth panel reports
**undefined** wherever there is no genuine power-law regime (too small a graph,
or an expander), so what you watch is the honest estimator verdict, not a
forced number.

### Dimension analysis

```bash
# Validate the estimator against graphs of known dimension (no run needed)
python dimension.py --validate

# Analyze local effective dimension from a completed run
python dimension.py results/run_TIMESTAMP.pkl

# Use fast sparse backend with custom radius
python dimension.py results/run_TIMESTAMP.pkl --fast --max-radius 6 --samples 300

# Visualize dimension map and histogram
python visualize.py results/run_TIMESTAMP.pkl --dimension
```

The estimator uses a finite-size-corrected log-log fit and only reports `d_eff`
where ball growth is genuinely polynomial; small-world / expander graphs have no
such regime, so their dimension is reported as **undefined** rather than a
fabricated number. The fraction of dimension-defined nodes (`defined_frac`) is
itself a signal -- on the small-world default topologies it should be ~0, and any
rise over time is evidence of emergent geometric structure. The `lattice`
topology is the one default where dimension is well-defined (`d_eff ≈ 2`).

### Temporal dimension tracking

```bash
# Does a 2D lattice keep its dimension under a rule, or lose it?
python track_dimension.py --topology lattice --rules majority --steps 200   # preserved
python track_dimension.py --topology lattice --rules rewire   --steps 200   # destroyed

# Emergence: shortcut-pruning recovers the latent ~1D ring of a small-world graph
python track_dimension.py --topology small_world --rules prune --steps 360 --max-radius 10

# Tunable emergent dimension from a grown graph
python track_dimension.py --topology grown --rules majority --steps 100 --max-radius 10
```

Measures the dimension field at t=0 and every `--track-interval` steps, plotting
`defined_frac`, the `d_eff` distribution, and dimension composition over time.
Use `--max-radius` to set a measurement radius large enough to resolve emergent
(high-diameter) structure. See [FINDINGS.md](FINDINGS.md) for results.

### cap → dimension scaling (does the grown law hold at scale?)

```bash
# Does grown's cap->dimension law plateau as N grows? (caps 6/7/8)
python cap_dimension_scaling.py --caps 6 7 8 --nodes 2000 5000 10000 20000 50000 --seeds 3

# Push a single cap to large N to test convergence
python cap_dimension_scaling.py --caps 8 --nodes 50000 100000 200000 --seeds 3
```

Measures the dimension field at a fixed radius across N and reports whether
`d_eff(N)` plateaus. Finding: it does, at **non-integer** values (cap 6→~2.2,
7→~3.0, 8→~3.6), and higher-d caps resolve only at larger N. The small-N
single-point estimates were biased low by ball saturation. See
[FINDINGS.md](FINDINGS.md) → "cap → d scaling."

### Quantified bootstrapping barrier

```bash
# Measure extent (diameter) vs rewiring steps and vs N; fit extent ~ N^alpha
python barrier_scaling.py --nodes 1000 2000 4000 8000 16000 --seeds 3 --steps 200
```

Starts each rewiring rule (`triadic`, `geometrize`, `ricci`) from a random
expander, with a 2D `lattice` positive control and the unrewired random graph
as baseline. Finding: the rewiring rules stay at **α ≈ 0.13** (extent ~ log N,
like the expander) while the lattice control gives **α = 0.5** (true 2D) — local
rewiring crumples rather than unfolds, and the gap widens with N. This turns the
bootstrapping barrier from an asserted claim into a measured obstruction
exponent. See [FINDINGS.md](FINDINGS.md) → "Quantified barrier."

### Finite-size scaling (phase transitions)

```bash
# Validate the FSS machinery on the majority-vote/Ising transition (q_c ~ 0.08)
python ising_sweep.py --sides 16 24 32 --seeds 4 --collapse

# Quick smoke run
python ising_sweep.py --quick

# The actual (Z2-biased) majority rule, for comparison -- shows no clean transition
python ising_sweep.py --model project --noise-max 0.6
```

Sweeps a noise grid across lattice sizes and computes the order parameter,
susceptibility, and Binder cumulant; the Binder-curve crossing locates the
critical point and `--collapse` overlays a 2D-Ising data collapse. This is the
validation step before applying finite-size scaling to hunt for *novel*
transitions (e.g. shortcut-density → dimensional onset under `prune`). See
[FINDINGS.md](FINDINGS.md) → "Scaling directions."

### prune as a tunable-dimension knob

```bash
# Sweep shortcut density p x N; map d(p) and test crossover-vs-transition
python prune_dimension.py --nodes 2000 8000 32000 --seeds 3
python prune_dimension.py --validate-real        # ER-control reality check
```

The `prune` phase-transition hunt (step 3) came back negative — and that *is*
the finding. Pruning a Watts-Strogatz ring to convergence does not switch
dimension on at a critical `p`; instead `p` tunes the pruned dimension
**continuously** from ~1 (ring) to ~2 (mesh), N-independently. It is a
**crossover, not a critical point**: the peak slope `max|dd/dp|` stays flat
across 16× in N (no diverging response for finite-size scaling to latch onto),
so there are no exponents to extract. An Erdős–Rényi control at matched mean
degree confirms the dimension is real geometry, not a low-degree artifact. So
`prune` is a third continuum dimension knob alongside the `grown` cap. See
[FINDINGS.md](FINDINGS.md) → "prune dimensional onset."

### Spatial coherence of the dimension field

```bash
# Validate Moran's I on known fields, then measure field coherence
python coherence.py --validate
python coherence.py --nodes 5000 --cap 6 --seeds 3
```

Moran's I (permutation-tested) asks whether the per-node `d_eff` field is
spatially smooth or just noise. Finding: on `grown` the field is defined almost
everywhere and **strongly coherent** (I≈0.88, z≈87) — emergent dimension forms
contiguous phases, not scattered noise — and the coherence tracks extent under
the rules (preserved by `prune`, destroyed by `rewire`). On an expander the
field is mostly undefined, so coherence is reported N/A rather than over-claimed.
See [FINDINGS.md](FINDINGS.md) → "Spatial coherence."

### Quantum walk analysis

```bash
# Self-check the matrix walks against known graphs (no results file needed)
python braket_walks.py --validate

# Compare continuous-time quantum walks to classical walks on sampled subgraphs
python braket_walks.py results/run_TIMESTAMP.pkl --samples 5 --subgraph-size 32

# Show plots interactively instead of saving to plots/
python braket_walks.py results/run_TIMESTAMP.pkl --show
```

Matrix-based CTQW (via `scipy.sparse.linalg.expm_multiply`) compared against classical
random walks; high total-variation distance flags structure (bottlenecks, symmetries)
that classical walks miss. Runs on the core dependencies -- no Amazon Braket SDK needed.

### Parameter sweep

```bash
python sweep.py \
    --nodes 1000 5000 \
    --topologies small_world scale_free random \
    --rules "activation" "activation majority" "activation reinforcement" \
    --steps 2000 --seeds 5 --jobs 4
```

Results are saved to `results/sweep_TIMESTAMP.csv` with a summary of which configurations showed interesting emergent behavior.

## Graph Topologies

Four initial topologies are available via `--topology`:

- `small_world` -- Watts-Strogatz (default, k=6, p=0.1)
- `scale_free` -- Barabasi-Albert preferential attachment
- `lattice` -- 2D grid
- `random` -- Erdos-Renyi
- `grown` -- degree-capped frontier growth (`k` = degree cap); produces tunable *emergent* dimension (cap 6 → ~2D, 8 → ~3D)

## Goals

1. **Small scale (1K nodes)** -- Verify rules work, debug
2. **Medium scale (100K nodes)** -- Look for emergent patterns (use `simulation_fast.py`)
3. **Large scale (1M+ nodes)** -- GPU acceleration, serious exploration (see `SCALING.md`)


