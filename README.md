# graph-graph

**The Honest Experiment:** Do simple local graph rules at large scale produce interesting emergent behavior?

## Philosophy

Unlike other projects that hand-code physics and then "discover" it, this project starts with **only simple local rules** and observes what emerges—Game of Life style.

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

🚧 **Just Starting** - Scaffolding phase

## Rules to Explore

1. **Activation spreading** - Active nodes activate neighbors
2. **Edge dynamics** - Edges strengthen/weaken based on use
3. **Birth/death** - Nodes spawn or die based on connectivity
4. **Rewiring** - Random edge rewiring (small-world dynamics)
5. **State voting** - Nodes adopt majority state of neighbors

## Measurements

- Correlation functions C(r) between nodes at distance r
- Clustering coefficient over time
- Information propagation speed
- Spontaneous symmetry breaking
- Any long-range order

## Goals

1. **Small scale (1K nodes)** - Verify rules work, debug
2. **Medium scale (100K nodes)** - Look for emergent patterns
3. **Large scale (1M+ nodes)** - GPU acceleration, serious exploration

## Relationship to photon_web_reality

This is what `photon_web_reality` *should* have been - an honest experiment rather than circular verification of hand-coded QM.

If graph-graph produces interesting results, they could inform a V3 of the photon web concept.

## Usage

```bash
python simulation.py --nodes 1000 --steps 1000 --rules activation
python simulation.py --nodes 1000 --steps 1000 --rules activation majority  # combine rules
python measure.py results/run_TIMESTAMP.pkl
python visualize.py results/run_TIMESTAMP.pkl
```

