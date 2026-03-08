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
| `showcase.py` | Generate curated demo animations (one per rule + combos) |
| `sweep.py` | Parameter sweep with parallel execution and CSV export |
| `dimension.py` | Local effective dimension estimator (d_eff via geodesic ball growth) |
| `braket_walks.py` | Quantum walk analysis (optional, experimental) |
| `SCALING.md` | Roadmap from 1K to 100M+ nodes |
| `DIMENSIONAL_COHERENCE.md` | Theory and roadmap for dimensional coherence measurements |

## Rules

All rules are **local** -- each node only sees its immediate neighbors. Rules can be combined freely.

| Rule | Key | What it does |
|------|-----|--------------|
| Activation spreading | `activation` | Active nodes spread activation to neighbors; active nodes may decay (SIS epidemic model) |
| Edge reinforcement | `reinforcement` | Edges between co-active nodes strengthen; all edges slowly decay (Hebbian learning) |
| Majority vote | `majority` | Nodes adopt the majority state of their neighbors; small noise prevents freezing |
| Random rewiring | `rewire` | Small probability of rewiring edges, creating small-world structure over time |

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

Seven pre-tuned animations, one for each rule (epidemic spreading, Hebbian reinforcement, majority-vote domains, small-world rewiring), one per notable topology (scale-free hubs, random baseline), and one combining all four rules to show full emergence.

### Dimension analysis

```bash
# Analyze local effective dimension from a completed run
python dimension.py results/run_TIMESTAMP.pkl

# Use fast sparse backend with custom radius
python dimension.py results/run_TIMESTAMP.pkl --fast --max-radius 6 --samples 300

# Visualize dimension map and histogram
python visualize.py results/run_TIMESTAMP.pkl --dimension
```

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

## Goals

1. **Small scale (1K nodes)** -- Verify rules work, debug
2. **Medium scale (100K nodes)** -- Look for emergent patterns (use `simulation_fast.py`)
3. **Large scale (1M+ nodes)** -- GPU acceleration, serious exploration (see `SCALING.md`)


