# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Orientation

This is a research sandbox testing one question: **do simple, strictly-local graph
update rules produce emergent structure at scale?** It deliberately has no hand-coded
physics, no baked-in spatial coordinates, and no pre-programmed correlations — emergence
first, physics maybe later. Do not add features that violate this (e.g. anything computed
from "node positions" — there are none).

`AGENTS.md` holds the full agent-facing guide (architectural invariants, editing
conventions, pitfalls). Read it before making non-trivial changes — it is authoritative
where this file and it overlap. `README.md` is the project tour and CLI reference.

## Commands

No build step, no linter config, and **no test suite**. It is a script-first repo of
standalone CLIs. Validate changes by running a small, seeded invocation and inspecting
the printed metrics or the generated pickle:

```powershell
# Shell is PowerShell 5.1 — chain with ';', never '&&'.
pip install -r requirements.txt

# Smoke-test a change (small + seeded = reproducible)
python simulation.py --nodes 200 --steps 100 --seed 0

# Quantum-walk module ships a self-check
python braket_walks.py --validate
```

### Running a simulation (pick backend by scale)

| Nodes      | Command                                                                 |
|------------|-------------------------------------------------------------------------|
| ≤ 10K      | `python simulation.py --nodes 1000 --steps 1000 --rules activation majority --seed 42` |
| 10K – 500K | `python simulation_fast.py --nodes 50000 --steps 2000 --rules activation reinforcement --seed 42` |
| > 500K     | Not supported yet — see `SCALING.md`                                    |

Both write a pickle to `results/run_*.pkl`. Rules combine freely; available keys:
`activation`, `reinforcement`, `majority`, `rewire`. Topologies (`--topology`):
`small_world` (default), `scale_free`, `lattice`, `random`.

### Analysis / visualization (all take a `results/*.pkl` path as first positional arg)

```powershell
python measure.py   results/run_TIMESTAMP.pkl          # correlations, domains, verdict
python visualize.py results/run_TIMESTAMP.pkl           # metric + graph-state plots
python visualize.py results/run_TIMESTAMP.pkl --dimension
python dimension.py results/run_TIMESTAMP.pkl --fast --max-radius 6 --samples 300
python braket_walks.py results/run_TIMESTAMP.pkl --samples 5 --subgraph-size 32
```

### Animations and parameter sweeps

```powershell
python animate.py --nodes 300 --steps 400 --rules activation majority --seed 42 --save anim.gif
python showcase.py            # generate all curated demo GIFs -> showcase/
python sweep.py --nodes 1000 5000 --topologies small_world random \
    --rules "activation" "activation majority" --steps 2000 --seeds 5 --jobs 4
```

## Architecture

The pipeline is **simulate → pickle → analyze**, with the pickle as the contract between
stages.

- **Rules (`rules.py`)** are pure `nx.Graph -> nx.Graph` functions, each reading only a
  node's own state plus immediate neighbors. They are registered in the `RULES` dict and
  resolved by name via `get_rule()`. Node state lives in graph attributes:
  `active` (bool, used by `activation`/`reinforcement`), `state` (int, used by `majority`),
  and edge `weight` (float, used by `reinforcement`). This locality is an invariant — no
  global reductions or distance-`r>1` reads inside a rule.

- **Two backends, identical semantics.** `simulation.py` is the NetworkX reference
  implementation. `simulation_fast.py` reimplements every rule as vectorized sparse-matrix
  ops in the `FastGraph` class (CSR adjacency + dense state arrays; e.g. `A @ active`
  counts active neighbors, a one-hot multiply does the majority vote). When you change a
  rule's behavior, **change it in both files** and keep them equivalent for the same
  seed/params. `FAST_RULES` maps rule keys to `FastGraph` method names.

- **Shared output contract.** `run_simulation` and `run_fast_simulation` return the same
  dict shape — `{'snapshots', 'metrics', 'final_graph', 'params'}` — so `measure.py`,
  `visualize.py`, `dimension.py`, and `braket_walks.py` work on either backend's pickle.
  `final_graph` is always a NetworkX graph (`FastGraph.to_networkx()` converts back).
  `metrics` tracks `step, n_active, mean_weight, clustering, largest_component` timeseries.

- **`create_initial_graph` lives in `simulation.py`** and is imported by the fast backend
  and `sweep.py` — it is the single source of topology construction and initial state.
  Don't duplicate it. (Note: `lattice` rounds `n_nodes` down to the nearest perfect square.)

- **`dimension.py`** estimates per-node effective dimension `d_eff` by fitting
  `log|B(v,r)|` vs `log r` over geodesic balls; it has a NetworkX path (`dimension_field`)
  and a sparse `--fast` path (`fast_dimension_field`, iterative `A @ v` BFS) mirroring the
  two simulation backends. See `DIMENSIONAL_COHERENCE.md` for the theory.

- **`sweep.py`** fans `create_initial_graph` + run + `measure.py` analysis across a
  topology × rules × scale × seed grid via `ProcessPoolExecutor`, prefers the fast backend
  when available (falls back to NetworkX), and writes `results/sweep_*.csv`.

## Conventions

- Python 3.10+ (`int | None` unions, etc.). Type hints on public functions, NumPy-style
  docstrings, `tqdm` for long loops — match `simulation.py` / `rules.py`.
- Every entry-point CLI takes `--seed` and seeds both `random` and `numpy.random`.
  New scripts must do the same.
- Snapshots are opt-in (`snapshot_interval=0` default) to bound memory — don't flip this.
- Keep `simulation.py` free of scipy/sparse-only imports so the reference path stays
  minimal; sparse deps belong in `simulation_fast.py`.
- `results/`, `plots/`, `showcase/` are generated outputs, not source.
