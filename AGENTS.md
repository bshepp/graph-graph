# AGENTS.md

Guidance for AI coding agents working in `graph-graph`. Read [README.md](README.md) first
for the full project tour, rule catalog, CLI examples, and topology list.

## What this project is

A research sandbox for the question: *do simple local graph rules produce emergent
structure at scale?* See [README.md](README.md#philosophy) and
[DIMENSIONAL_COHERENCE.md](DIMENSIONAL_COHERENCE.md) for the framing.

The orientation matters when proposing changes: this repo deliberately avoids
hand-coded physics, baked-in spatial coordinates, or pre-programmed correlations.
**Emergence first, physics (maybe) later.** Do not add features that contradict this
(e.g. computing things from "node positions" — there are none).

## Architectural invariants

- **Rules are strictly local.** Every function in [rules.py](rules.py) may only read a
  node's own state and its immediate neighbors (and incident edge weights). No global
  reductions, no distance-`r>1` lookups inside a rule. New rules must follow this.
- **Two simulation backends, identical semantics.** [simulation.py](simulation.py) is
  the NetworkX reference (≤~10K nodes). [simulation_fast.py](simulation_fast.py) is the
  sparse-matrix vectorized path (≤~500K nodes) — see [SCALING.md](SCALING.md). When you
  change a rule, update **both** backends and keep their behavior equivalent for the
  same seed/parameters.
- **Determinism via `--seed`.** Every entry-point CLI accepts `--seed`; reproducibility
  matters for this project. New scripts should do the same and seed both `random` and
  `numpy.random`.
- **Snapshots are opt-in.** `run_simulation(..., snapshot_interval=0)` is the default
  to keep memory bounded. Don't flip this default; large runs will OOM.

## File map (delta from README)

The [README project structure table](README.md#project-structure) is authoritative.
Beyond that:

- No `tests/`, no `pyproject.toml`, no linter config — this is a script-first repo.
- Run artifacts go to `results/` (`run_TIMESTAMP.pkl`, `sweep_TIMESTAMP.csv`); plots
  go to `plots/`; curated demo GIFs go to `showcase/`. Treat these as outputs, not
  source.
- GPU / quantum extras in [requirements.txt](requirements.txt) are commented out
  intentionally. Don't enable them without an explicit ask.

## Running things (Windows / PowerShell)

The shell here is PowerShell 5.1. Chain commands with `;`, never `&&`. All entry
points are plain `python <script>.py --help`-style CLIs documented in
[README.md § Usage](README.md#usage). Pick the backend by scale:

| Nodes | Use |
|-------|-----|
| ≤ 10K | `python simulation.py …` |
| 10K – 500K | `python simulation_fast.py …` |
| > 500K | Not supported yet — see [SCALING.md](SCALING.md) before adding code |

Analysis scripts (`measure.py`, `visualize.py`, `dimension.py`) all take a
`results/run_TIMESTAMP.pkl` path as their first positional arg.

## Conventions when editing

- Python 3.10+ syntax is fine (`int | None` unions are already used).
- Type hints on public functions, NumPy-style docstrings, `tqdm` for long loops —
  match the existing style in [simulation.py](simulation.py) and [rules.py](rules.py).
- New rules: add to [rules.py](rules.py), register in the `RULES` dict, implement the
  vectorized version in [simulation_fast.py](simulation_fast.py), and expose via the
  `--rules` CLI flag. Add a one-line entry to the rules table in
  [README.md](README.md#rules).
- New measurements live in [measure.py](measure.py); new plots in
  [visualize.py](visualize.py). Don't inline analysis into the simulation loop.
- No `pytest` suite — validate changes by running a small reproducible CLI
  invocation (e.g. `python simulation.py --nodes 200 --steps 100 --seed 0`) and
  inspecting the printed metrics / generated pickle. Two modules ship a built-in
  self-check against known-answer cases: `python dimension.py --validate` (estimator
  vs. graphs of known dimension) and `python braket_walks.py --validate` (walks vs.
  analytic expectations). Run the relevant one after touching those modules.

## Pitfalls

- Importing `simulation_fast` pulls in `scipy.sparse`; keep `simulation.py` free of
  sparse-only deps so the slow path stays minimal.
- `create_initial_graph` lives in [simulation.py](simulation.py) and is reused by the
  fast backend — don't duplicate it.
- The `lattice` topology silently rounds `n_nodes` down to the nearest perfect square
  (`int(sqrt(n))**2`). Mention this if a user reports "wrong node count".
- `braket_walks.py` runs entirely on the core deps (scipy/networkx/matplotlib) — its
  CTQW is matrix-based (`scipy.sparse.linalg.expm_multiply`), **not** circuit-based. The
  Amazon Braket SDK is *not* required; it appears only in deferred comments for a future
  hardware path. Still treat the module as experimental and don't import it from core
  modules.
