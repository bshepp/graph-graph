# Async Censorship Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run the banked synchronous shortcut-censorship result under asynchronous (event-driven) Poisson-clock updates and test whether it survives when time is emergent — a two-gate checkpoint (`async_censorship.py`) plus one reusable engine primitive.

**Architecture:** Add a merged-clock multi-rule runner `run_sequential_multi` to `async_engine.py` (the natural multi-rate generalisation of `run_sequential`, gated by a bit-identical single-rule equivalence check). Then `async_censorship.py` builds an identical `grown` base + injected portals, runs it through **both** schedules (synchronous via `shortcut_censorship.run_condition` verbatim, async via `run_sequential_multi` with a per-event portal-tracking callback), reduces both with one shared observable function, and reports Gate 1 (pure-prune P1 reproduction, the instrument gate + control) and Gate 2 (the triadic+prune race, descriptive, pre-registered both ways).

**Tech Stack:** Python 3.10+, NetworkX, NumPy, tqdm. No pytest — this repo validates via `--validate` self-check functions and seeded smoke runs (CLAUDE.md / AGENTS.md). RED→GREEN is done with throwaway scratch scripts, then the checks are folded into the module's `_validate()`.

## Global Constraints

- **Python 3.10+**; type hints on public functions; NumPy-style docstrings; `tqdm` for long loops — match `simulation.py` / `async_engine.py`.
- **Every entry-point CLI takes `--seed` and seeds both `random` and `numpy.random`.**
- **State-graph only.** Do NOT import or use `causal_dag.py` / causal-set estimators — the step-3-retired causal dimension is out of scope by design.
- **Sync side is `shortcut_censorship.run_condition` verbatim** — the banked semantics are the reference, never re-implemented.
- **`rewire_prob` must stay `0.05`** in the async triadic+prune condition: `run_condition` hardcodes `triadic_closure(G, rewire_prob=0.05)`, so any other value silently breaks sync/async parity. It is NOT a CLI knob.
- **No node positions / no baked geometry / rules stay local** (project invariant) — this module adds no rules, so it inherits this for free.
- **`nx.to_scipy_sparse_array` footgun:** if any sparse matrix is ever built here, pass `weight=None` (edges carry weight 0.5). This plan builds none — flagged only so it stays true if extended.
- **Tolerances are frozen after the first run** (the `z < 3` bar, `|rank-corr| < 0.2`, the event-count match) and not retuned to force a pass.
- **Scratchpad dir** for throwaway test scripts: `C:\Users\Snarf\AppData\Local\Temp\claude\F--science-projects-graph-graph\9784062b-5b9c-402d-94ac-0fcf88b4fad4\scratchpad`.
- Commits go directly to `main` (project norm); end each commit message with the standard `Co-Authored-By: Claude Opus 4.8 …` / `Claude-Session: …` trailer.

## File Structure

- **Modify `async_engine.py`** — add `run_sequential_multi(...)` (merged-clock, multi-rule, optional `on_event` callback and per-rule `params`); add a `Test D` block to `_validate()` asserting single-rule equivalence with `run_sequential`. Responsibility: the async scheduling engine. Stays pure-dynamics (no experiment/observable logic).
- **Create `async_censorship.py`** — the checkpoint experiment: shared base+portal setup, one shared observable reduction, the async tracked conditions, the sync reference wiring, the two-gate `_validate()`, and `main()` (`--validate` / `--quick` / production-scale run). Responsibility: the censorship checkpoint, both schedules.
- **Modify `FINDINGS.md`, `LORENTZIAN_SPIKE.md`, `README.md`** (Task 5) — record the result and mark step 4 done.
- **Scratch tests** (scratchpad, not committed) — RED→GREEN drivers for each code task.

---

### Task 1: `run_sequential_multi` in `async_engine.py` + single-rule equivalence gate

**Files:**
- Modify: `async_engine.py` (add function after `run_sequential`, ~line 263; add `Test D` inside `_validate`, before the final `PASS/FAIL` print ~line 527)
- Test: scratchpad `test_multi_equiv.py`

**Interfaces:**
- Consumes: `apply_event`, `EVENTS`, `fingerprint`, `create_initial_graph`, `run_sequential` (all already in `async_engine.py`).
- Produces:
  `run_sequential_multi(G: nx.Graph, rules: Sequence[str], rates: Sequence[float], n_events: int | None = None, max_time: float | None = None, seed: int = 0, params: Sequence[dict | None] | None = None, on_event: Callable[[int, int, int, float, nx.Graph], None] | None = None) -> tuple[nx.Graph, np.ndarray, np.ndarray]`
  Returns `(final_graph, times, rule_ids)` — `times` = per-event absolute Poisson times, `rule_ids` = per-event index into `rules`. With one rule it reproduces `run_sequential`'s graph bit-for-bit at a matched seed.

- [ ] **Step 1: Write the failing test** (scratchpad `test_multi_equiv.py`)

```python
import numpy as np
from async_engine import (run_sequential, run_sequential_multi, fingerprint)
from simulation import create_initial_graph

def test_single_rule_equivalence():
    for rule, topo in (('activation', 'grown'), ('prune', 'small_world')):
        G = create_initial_graph(200, topology=topo, k=6, seed=3)
        if rule == 'activation':
            for i, node in enumerate(G.nodes()):
                G.nodes[node]['active'] = (i % 3 == 0)
        g1, _ = run_sequential(G, rule, 500, seed=7)
        g2, times, rule_ids = run_sequential_multi(G, [rule], [1.0],
                                                   n_events=500, seed=7)
        assert fingerprint(g1) == fingerprint(g2), f"{rule}: multi != single"
        assert len(times) == 500 and set(rule_ids.tolist()) == {0}
    print("PASS single-rule equivalence")

def test_two_rule_runs_and_counts():
    G = create_initial_graph(300, topology='small_world', k=6, seed=1)
    g, times, rule_ids = run_sequential_multi(
        G, ['triadic', 'prune'], [1.0, 1.0], max_time=10.0, seed=2,
        params=[{'rewire_prob': 0.05}, {'prune_prob': 0.05}])
    assert len(times) == len(rule_ids) > 0
    assert times.max() <= 10.0
    # both rules fire; times are non-decreasing (absolute Poisson time)
    assert set(rule_ids.tolist()) == {0, 1}
    assert np.all(np.diff(times) >= -1e-12)
    print("PASS two-rule run")

if __name__ == '__main__':
    test_single_rule_equivalence()
    test_two_rule_runs_and_counts()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_multi_equiv.py` (from the repo root, with the scratchpad script copied in or run by absolute path)
Expected: FAIL — `ImportError: cannot import name 'run_sequential_multi'`.

- [ ] **Step 3: Write minimal implementation** (add to `async_engine.py` after `run_sequential`)

```python
def run_sequential_multi(G: nx.Graph, rules: Sequence[str],
                         rates: Sequence[float],
                         n_events: Optional[int] = None,
                         max_time: Optional[float] = None,
                         seed: int = 0,
                         params: Optional[Sequence[Optional[Dict]]] = None,
                         on_event: Optional[Callable] = None
                         ) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Merged-clock async engine for several rules running concurrently.

    Each (node, rule) pair carries an independent exponential clock; the next
    event is the global argmin, applied via `apply_event` with the rule's own
    per-event RNG. This is the multi-rate generalisation `run_sequential`'s
    docstring anticipates. With ONE rule it draws its clocks in the same order
    as `run_sequential` (one size-N exponential, then one scalar re-draw per
    event), so it returns a bit-identical graph for a matched seed -- checked
    in `_validate` Test D.

    Stops at whichever of `n_events` (event count) or `max_time` (absolute
    Poisson time) is reached first; at least one must be given. With every
    rate 1.0, absolute time equals sweep-equivalents of each rule
    independently, so two concurrent rate-1 clocks do NOT double-count time
    the way an event counter would.

    Parameters
    ----------
    rules, rates : one entry per concurrent rule.
    params : per-rule event kwargs (``params[i]`` -> rule ``i``); None -> the
        event's own defaults, which reproduces `run_sequential`.
    on_event : optional ``(event_id, node, rule_idx, time, G) -> None`` hook
        called after each event, so a caller can observe the trajectory (e.g.
        portal removal times) without the engine knowing the observable.

    Returns
    -------
    (final_graph, times, rule_ids) : times are per-event absolute Poisson
        times (non-decreasing); rule_ids index into `rules`.
    """
    if n_events is None and max_time is None:
        raise ValueError("run_sequential_multi needs n_events or max_time")
    if len(rates) != len(rules) or (params is not None
                                    and len(params) != len(rules)):
        raise ValueError("rules, rates, params must be the same length")

    G = G.copy()
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    N = len(nodes)
    R = len(rules)

    clocks = np.empty((R, N))
    for r in range(R):
        clocks[r] = rng.exponential(1.0 / rates[r], size=N)

    times: List[float] = []
    rule_ids: List[int] = []
    event_id = 0
    while n_events is None or event_id < n_events:
        idx = int(clocks.argmin())
        r, k = divmod(idx, N)
        t = float(clocks[r, k])
        if max_time is not None and t > max_time:
            break
        p = params[r] if params is not None else None
        apply_event(G, nodes[k], rules[r], seed, event_id, p)
        if on_event is not None:
            on_event(event_id, nodes[k], r, t, G)
        times.append(t)
        rule_ids.append(r)
        clocks[r, k] += rng.exponential(1.0 / rates[r])
        event_id += 1

    return G, np.asarray(times), np.asarray(rule_ids, dtype=int)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_multi_equiv.py`
Expected: `PASS single-rule equivalence` and `PASS two-rule run`. If the equivalence assert fails, the clock draw order drifted from `run_sequential` — recheck that the initial draw is one `rng.exponential(1.0/rates[r], size=N)` per rule and the re-draw is a single scalar.

- [ ] **Step 5: Fold the equivalence check into `_validate`** (add before the final `PASS/FAIL` print in `async_engine.py::_validate`)

```python
    print("\nTest D -- run_sequential_multi single-rule equivalence "
          "(the multi-rate\n         generalisation must not drift from the "
          "validated single-rule path)")
    for rule, topo in (('activation', 'grown'), ('prune', 'small_world')):
        G = create_initial_graph(200, topology=topo, k=6, seed=3)
        if rule == 'activation':
            for i, node in enumerate(G.nodes()):
                G.nodes[node]['active'] = (i % 3 == 0)
        g1, _ = run_sequential(G, rule, 500, seed=7)
        g2, _, _ = run_sequential_multi(G, [rule], [1.0], n_events=500, seed=7)
        same = fingerprint(g1) == fingerprint(g2)
        ok &= same
        print(f"  {rule:<12s} on {topo:<12s}: "
              f"{'bit-identical OK' if same else 'DIVERGED -- FAIL'}")
```

- [ ] **Step 6: Run the full engine gate**

Run: `python async_engine.py --validate`
Expected: Tests A, A', B, C unchanged and passing; new `Test D` prints `bit-identical OK` for both rules; final line `PASS: asynchronous engine`.

- [ ] **Step 7: Commit**

```bash
git add async_engine.py
git commit -m "async_engine: run_sequential_multi (merged-clock multi-rule) + equivalence gate"
# end with the standard Co-Authored-By / Claude-Session trailer
```

---

### Task 2: `async_censorship.py` — base+portal setup and the shared observable reduction

**Files:**
- Create: `async_censorship.py` (module docstring, imports, `build_base_and_portals`, `fabric_edges`, `summarize_portals`)
- Test: scratchpad `test_censor_reduce.py`

**Interfaces:**
- Consumes: `create_initial_graph` (`simulation.py`), `inject_shortcuts` (`shortcuts.py`).
- Produces:
  - `build_base_and_portals(n_nodes: int, cap: int, n_long: int, n_detour2: int, seed: int) -> tuple[nx.Graph, list[tuple[int,int,int]], list[tuple[int,int,int]]]` → `(base, long_portals, detour2)`; portal tuples are `(u, v, advantage)`.
  - `fabric_edges(base, long_portals, detour2) -> set[frozenset]` → base edges minus the portal edges.
  - `summarize_portals(removal: dict[tuple[int,int], float | None], long_portals, detour2, final_graph: nx.Graph, base_edges: set[frozenset]) -> dict` → keys `long_survival, detour2_survival, mean_removal, adv_corr, woven, collateral`. Used for BOTH schedules so rows are directly comparable.

- [ ] **Step 1: Write the failing test** (scratchpad `test_censor_reduce.py`)

```python
import networkx as nx
import numpy as np
from async_censorship import (build_base_and_portals, fabric_edges,
                              summarize_portals)

def test_build_reproducible_and_shaped():
    b1, l1, d1 = build_base_and_portals(500, 6, 12, 6, seed=0)
    b2, l2, d2 = build_base_and_portals(500, 6, 12, 6, seed=0)
    assert [t[:2] for t in l1] == [t[:2] for t in l2]  # reproducible
    assert len(l1) <= 12 and len(d1) <= 6
    assert all(adv >= 6 for *_uv, adv in l1)           # long = advantage >=6
    assert all(adv == 2 for *_uv, adv in d1)           # detour-2 controls

def test_summarize_hand_graph():
    # Square 0-1-2-3-0 (fabric) + portal (0,2) long + detour (1,3).
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)], weight=0.5)
    long_portals = [(0, 2, 7)]
    detour2 = [(1, 3, 2)]
    base = G.copy()
    base.add_edge(0, 2, weight=0.5)  # portals present in base
    base.add_edge(1, 3, weight=0.5)
    fabric = fabric_edges(base, long_portals, detour2)
    assert fabric == {frozenset((0, 1)), frozenset((1, 2)),
                      frozenset((2, 3)), frozenset((3, 0))}
    # final graph: long portal removed (fabric intact), detour survived
    final = base.copy(); final.remove_edge(0, 2)
    removal = {(0, 2): 3.5, (1, 3): None}
    s = summarize_portals(removal, long_portals, detour2, final, fabric)
    assert s['long_survival'] == 0.0
    assert s['detour2_survival'] == 1.0
    assert s['mean_removal'] == 3.5
    assert s['collateral'] == 0.0
    assert s['woven'] == 0.0
    print("PASS summarize")

if __name__ == '__main__':
    test_build_reproducible_and_shaped()
    test_summarize_hand_graph()
    print("PASS all")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_censor_reduce.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'async_censorship'`.

- [ ] **Step 3: Write minimal implementation** (create `async_censorship.py`)

```python
"""
Shortcut censorship under ASYNCHRONOUS updates -- step 4, the first physics
checkpoint (LORENTZIAN_SPIKE.md sec 6; design spec
docs/superpowers/specs/2026-08-02-async-censorship-checkpoint-design.md).

Re-runs the banked synchronous censorship result (FINDINGS.md;
shortcut_censorship.py) with event-driven Poisson-clock dynamics and asks
whether it survives when time is emergent rather than a global sweep clock.

  Gate 1 (anchor + control): pure `prune` under async must reproduce P1 --
    threshold, advantage-blind, zero collateral -- IN DISTRIBUTION vs the
    synchronous result. Passing certifies the harness and isolates Gate 2.
  Gate 2 (the race, descriptive): `triadic`+`prune` under async. The
    synchronous "woven-in" self-stabilization (P2) may be an artifact of the
    triadic-then-prune lockstep WITHIN a synchronous step; async has no such
    fixed intra-step ordering. Pre-registered both ways.

State-graph only: the step-3-retired causal DAG is not involved.

Usage:
    python async_censorship.py --validate
    python async_censorship.py --validate --quick
    python async_censorship.py --nodes 2000 --seeds 3
"""

import argparse
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx
from tqdm import tqdm

from simulation import create_initial_graph
from shortcuts import inject_shortcuts
from shortcut_censorship import run_condition
from async_engine import run_sequential_multi

Portal = Tuple[int, int, int]


def build_base_and_portals(n_nodes: int, cap: int, n_long: int,
                           n_detour2: int, seed: int
                           ) -> Tuple[nx.Graph, List[Portal], List[Portal]]:
    """
    Grown base + long (advantage >= 6) and detour-2 (advantage == 2) portals.

    Seeds numpy before injection (`inject_shortcuts` draws from the global
    RNG), so the returned base+portals are reproducible and can be shared
    unchanged by both schedules for a seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    base = create_initial_graph(n_nodes, topology='grown', k=cap, seed=seed)
    long_portals = inject_shortcuts(base, n_long, min_distance=6)
    detour2 = inject_shortcuts(base, n_detour2, min_distance=2, max_distance=2)
    return base, long_portals, detour2


def fabric_edges(base: nx.Graph, long_portals: List[Portal],
                 detour2: List[Portal]) -> set:
    """Base edges minus the injected portal edges (the 'fabric' to protect)."""
    portal_set = set(frozenset((u, v)) for u, v, _ in long_portals + detour2)
    return set(frozenset(e) for e in base.edges()) - portal_set


def summarize_portals(removal: Dict[Tuple[int, int], Optional[float]],
                      long_portals: List[Portal], detour2: List[Portal],
                      final_graph: nx.Graph, base_edges: set) -> Dict[str, float]:
    """
    The censorship observables from a per-portal removal-time dict.

    `removal[(u, v)]` is the removal time (float, any monotone unit) or None
    if the portal survived; keys are the raw (u, v) from the portal tuples.
    The SAME reduction is applied to both schedules, so their rows compare
    directly. Removal-time UNITS differ (sync: integer steps; async:
    continuous sweep-equivalents) but both mean ~1/prune_prob, and the rank
    correlation is unit-free.
    """
    def alive(u: int, v: int) -> bool:
        return removal[(u, v)] is None

    long_survival = float(np.mean([alive(u, v) for u, v, _ in long_portals]))
    detour2_survival = (float(np.mean([alive(u, v) for u, v, _ in detour2]))
                        if detour2 else float('nan'))

    removed = [(float(adv), float(removal[(u, v)]))
               for u, v, adv in long_portals if removal[(u, v)] is not None]
    mean_removal = float(np.mean([t for _, t in removed])) if removed \
        else float('nan')

    if len(removed) >= 8:
        advs = np.array([a for a, _ in removed])
        ts = np.array([t for _, t in removed])
        ra = np.argsort(np.argsort(advs)).astype(float)
        rt = np.argsort(np.argsort(ts)).astype(float)
        adv_corr = float(np.corrcoef(ra, rt)[0, 1]) \
            if ra.std() > 0 and rt.std() > 0 else float('nan')
    else:
        adv_corr = float('nan')

    woven = sum(1 for u, v, _ in long_portals
                if removal[(u, v)] is None
                and len(set(final_graph[u]) & set(final_graph[v])) >= 1)

    remaining = set(frozenset(e) for e in final_graph.edges())
    collateral = 1.0 - len(base_edges & remaining) / max(len(base_edges), 1)

    return {'long_survival': long_survival,
            'detour2_survival': detour2_survival,
            'mean_removal': mean_removal, 'adv_corr': adv_corr,
            'woven': float(woven), 'collateral': collateral}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_censor_reduce.py`
Expected: `PASS summarize` then `PASS all`. (If `test_build_reproducible_and_shaped` finds fewer than expected long portals at N=500, that is acceptable — the assert uses `<=`; the count is graph-limited by `inject_shortcuts`.)

- [ ] **Step 5: Commit**

```bash
git add async_censorship.py
git commit -m "async_censorship: base+portal setup and shared observable reduction"
# end with the standard Co-Authored-By / Claude-Session trailer
```

---

### Task 3: async tracked conditions + sync wiring

**Files:**
- Modify: `async_censorship.py` (add `async_condition`, `sync_condition`)
- Test: scratchpad `test_censor_conditions.py`

**Interfaces:**
- Consumes: `run_sequential_multi` (Task 1), `run_condition` (`shortcut_censorship.py`), `build_base_and_portals`/`summarize_portals` (Task 2).
- Produces:
  - `async_condition(base, long_portals, detour2, rules: Sequence[str], rates: Sequence[float], params: Sequence[dict | None], sweeps: float, seed: int) -> tuple[dict, nx.Graph, dict[str,float]]` → `(removal, final_graph, events_per_node)`; `removal` keyed by raw `(u,v)` with float time or None; `events_per_node[rule]` = mean events per node for each rule.
  - `sync_condition(base, long_portals, detour2, condition: str, steps: int, prune_prob: float) -> tuple[dict, nx.Graph]` → `(removal_step, final_graph)` from `run_condition`.

- [ ] **Step 1: Write the failing test** (scratchpad `test_censor_conditions.py`)

```python
import numpy as np
from async_censorship import (build_base_and_portals, fabric_edges,
                              summarize_portals, async_condition,
                              sync_condition)

def test_async_prune_removes_long_keeps_detour2():
    base, longp, d2 = build_base_and_portals(800, 6, 30, 15, seed=1)
    fabric = fabric_edges(base, longp, d2)
    removal, final, counts = async_condition(
        base, longp, d2, ['prune'], [1.0], [{'prune_prob': 0.05}],
        sweeps=120.0, seed=1001)
    s = summarize_portals(removal, longp, d2, final, fabric)
    # P1 shape: long portals mostly die, detour-2 immune, ~zero collateral
    assert s['long_survival'] < 0.3
    assert s['detour2_survival'] > 0.9
    assert s['collateral'] < 0.02
    # ~one prune event per node per sweep-equivalent -> ~sweeps total
    assert 90 < counts['prune'] < 150
    # base is untouched (run copies internally) -> reusable
    assert base.has_edge(*longp[0][:2])
    print("PASS async prune")

def test_sync_condition_shapes():
    base, longp, d2 = build_base_and_portals(800, 6, 30, 15, seed=1)
    fabric = fabric_edges(base, longp, d2)
    removal, final = sync_condition(base, longp, d2, 'prune', 120, 0.05)
    s = summarize_portals(removal, longp, d2, final, fabric)
    assert s['long_survival'] < 0.3 and s['detour2_survival'] > 0.9
    print("PASS sync prune")

if __name__ == '__main__':
    test_async_prune_removes_long_keeps_detour2()
    test_sync_condition_shapes()
    print("PASS all")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python test_censor_conditions.py`
Expected: FAIL — `ImportError: cannot import name 'async_condition'`.

- [ ] **Step 3: Write minimal implementation** (add to `async_censorship.py`)

```python
def async_condition(base: nx.Graph, long_portals: List[Portal],
                    detour2: List[Portal], rules: Sequence[str],
                    rates: Sequence[float],
                    params: Sequence[Optional[Dict]], sweeps: float,
                    seed: int) -> Tuple[Dict, nx.Graph, Dict[str, float]]:
    """
    Run one async condition on a fresh copy of `base`, tracking per-portal
    removal time in absolute Poisson time (= sweep-equivalents, clocks at
    rate 1). A portal is stamped removed the first time an event at one of its
    endpoints leaves the edge absent. `run_sequential_multi` copies `base`, so
    the caller's graph is untouched and reusable across conditions.
    """
    portals = long_portals + detour2
    removal: Dict[Tuple[int, int], Optional[float]] = {
        (u, v): None for u, v, _ in portals}
    by_node: Dict[int, List[Tuple[int, int]]] = {}
    for u, v, _ in portals:
        by_node.setdefault(u, []).append((u, v))
        by_node.setdefault(v, []).append((u, v))

    def on_event(event_id: int, node: int, rule_idx: int, t: float,
                 G: nx.Graph) -> None:
        for (u, v) in by_node.get(node, ()):
            if removal[(u, v)] is None and not G.has_edge(u, v):
                removal[(u, v)] = t

    final, times, rule_ids = run_sequential_multi(
        base, list(rules), list(rates), max_time=float(sweeps), seed=seed,
        params=list(params), on_event=on_event)

    n = base.number_of_nodes()
    events_per_node = {rules[i]: float(np.sum(rule_ids == i)) / max(n, 1)
                       for i in range(len(rules))}
    return removal, final, events_per_node


def sync_condition(base: nx.Graph, long_portals: List[Portal],
                   detour2: List[Portal], condition: str, steps: int,
                   prune_prob: float) -> Tuple[Dict, nx.Graph]:
    """
    The banked synchronous reference, via `shortcut_censorship.run_condition`
    verbatim. Returns (removal_step, final_graph); `run_condition` copies
    `base` internally. NOTE it hardcodes triadic `rewire_prob=0.05`.
    """
    res = run_condition(base, long_portals + detour2, condition, steps,
                        prune_prob)
    return res['removal_step'], res['final_graph']
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python test_censor_conditions.py`
Expected: `PASS async prune`, `PASS sync prune`, `PASS all`. If `counts['prune']` lands outside `(90, 150)`, the `max_time`/rate wiring is off — with rate 1 each node should fire ~`sweeps` times (Poisson(120), std ≈ 11).

- [ ] **Step 5: Commit**

```bash
git add async_censorship.py
git commit -m "async_censorship: async tracked conditions + sync reference wiring"
# end with the standard Co-Authored-By / Claude-Session trailer
```

---

### Task 4: two-gate `_validate()` + `main()`

**Files:**
- Modify: `async_censorship.py` (add `_z`, `_validate`, `main`, `if __name__ == '__main__'`)
- Test: run `python async_censorship.py --validate --quick`

**Interfaces:**
- Consumes: everything from Tasks 2–3.
- Produces: `python async_censorship.py --validate [--quick]` — exit 0 iff Gate 1 passes and the confound assertions hold; Gate 2 is printed descriptively with a pre-registered verdict and never sets the exit code by itself.

- [ ] **Step 1: Add the z-helper, `_validate`, and `main`**

```python
def _z(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sample z on seed-means (NaNs dropped); the async_engine Test-B bar."""
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return 0.0 if se == 0 else abs(a.mean() - b.mean()) / se


# rewire_prob is FIXED at 0.05 to match shortcut_censorship.run_condition's
# hardcoded triadic rate -- any other value silently breaks sync/async parity.
_REWIRE_PROB = 0.05


def _validate(n_nodes: int = 1200, cap: int = 6, n_long: int = 40,
              n_detour2: int = 20, sweeps: int = 120, seeds: int = 5,
              prune_prob: float = 0.05) -> bool:
    ok = True
    prune_params = [{'prune_prob': prune_prob}]
    tp_rules = ['triadic', 'prune']
    tp_params = [{'rewire_prob': _REWIRE_PROB}, {'prune_prob': prune_prob}]

    # ---- Gate 1: async prune vs sync prune (P1) ----
    keys = ['long_survival', 'detour2_survival', 'mean_removal', 'collateral']
    sync_g1 = {k: [] for k in keys}
    async_g1 = {k: [] for k in keys}
    async_corr: List[float] = []
    prune_pn: List[float] = []

    print(f"Gate 1 -- async prune vs sync prune (P1): N={n_nodes}, "
          f"{n_long} long + {n_detour2} detour-2, {sweeps} sweeps, "
          f"{seeds} seeds")
    for s in tqdm(range(seeds), desc='gate1'):
        base, longp, d2 = build_base_and_portals(n_nodes, cap, n_long,
                                                 n_detour2, s)
        fabric = fabric_edges(base, longp, d2)
        sr, sg = sync_condition(base, longp, d2, 'prune', sweeps, prune_prob)
        ar, ag, ac = async_condition(base, longp, d2, ['prune'], [1.0],
                                     prune_params, float(sweeps), seed=1000 + s)
        ssum = summarize_portals(sr, longp, d2, sg, fabric)
        asum = summarize_portals(ar, longp, d2, ag, fabric)
        for k in keys:
            sync_g1[k].append(ssum[k])
            async_g1[k].append(asum[k])
        async_corr.append(asum['adv_corr'])
        prune_pn.append(ac['prune'])

    print(f"  {'observable':>17s} {'sync':>8s} {'async':>8s} {'z':>6s}")
    for k in keys:
        z = _z(sync_g1[k], async_g1[k])
        passed = np.isnan(z) or z < 3.0
        ok &= passed
        print(f"  {k:>17s} {np.nanmean(sync_g1[k]):8.3f} "
              f"{np.nanmean(async_g1[k]):8.3f} {z:6.2f}  "
              f"{'' if passed else 'FAIL'}")
    corr = float(np.nanmean(async_corr))
    blind = abs(corr) < 0.2
    ok &= blind
    print(f"  async rank(adv, t) = {corr:+.3f}  "
          f"{'advantage-blind OK' if blind else 'CORRELATED -- FAIL'}")
    print(f"  async prune events/node = {np.mean(prune_pn):.1f} "
          f"(target ~{sweeps})")

    # ---- Gate 2: triadic+prune race (P2), descriptive ----
    print("\nGate 2 -- triadic+prune race (P2): DESCRIPTIVE, "
          "pre-registered both ways")
    g2k = ['long_survival', 'detour2_survival', 'woven', 'collateral']
    g2 = {'sync': {k: [] for k in g2k}, 'async': {k: [] for k in g2k}}
    tri_pn: List[float] = []
    prune_pn2: List[float] = []
    for s in tqdm(range(seeds), desc='gate2'):
        base, longp, d2 = build_base_and_portals(n_nodes, cap, n_long,
                                                 n_detour2, s)
        fabric = fabric_edges(base, longp, d2)
        sr, sg = sync_condition(base, longp, d2, 'triadic+prune', sweeps,
                                prune_prob)
        ar, ag, ac = async_condition(base, longp, d2, tp_rules, [1.0, 1.0],
                                     tp_params, float(sweeps), seed=2000 + s)
        ssum = summarize_portals(sr, longp, d2, sg, fabric)
        asum = summarize_portals(ar, longp, d2, ag, fabric)
        for k in g2k:
            g2['sync'][k].append(ssum[k])
            g2['async'][k].append(asum[k])
        tri_pn.append(ac['triadic'])
        prune_pn2.append(ac['prune'])

    print(f"  {'observable':>17s} {'sync':>8s} {'async':>8s}")
    for k in g2k:
        print(f"  {k:>17s} {np.nanmean(g2['sync'][k]):8.3f} "
              f"{np.nanmean(g2['async'][k]):8.3f}")

    tri, prn = float(np.mean(tri_pn)), float(np.mean(prune_pn2))
    matched = abs(tri - prn) / max(prn, 1e-9) < 0.1
    ok &= matched
    print(f"  async events/node: triadic {tri:.1f}, prune {prn:.1f}  "
          f"{'matched OK' if matched else 'MISMATCH -- confound, FAIL'}")

    a_long = float(np.nanmean(g2['async']['long_survival']))
    a_woven = float(np.nanmean(g2['async']['woven']))
    base_long = float(np.nanmean(async_g1['long_survival']))
    if a_woven > 0.0 and a_long > base_long + 0.02:
        verdict = ("P2 SURVIVES async -- weaving persists (woven "
                   f"{a_woven:.2f}, long survival {a_long:.2f} vs prune-only "
                   f"{base_long:.2f}); self-stabilization is genuine dynamics")
    else:
        verdict = ("P2 COLLAPSES under async -- woven ~0 / long survival "
                   f"{a_long:.2f} ~ prune-only {base_long:.2f}; the synchronous "
                   "self-stabilization was a triadic-then-prune lockstep "
                   "artifact")
    print(f"  VERDICT: {verdict}")

    print(f"\n{'PASS' if ok else 'FAIL'}: async censorship checkpoint "
          f"(Gate 1 instrument + confounds gate the exit; Gate 2 descriptive)")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Shortcut censorship under asynchronous updates.")
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--nodes', type=int, default=2000)
    ap.add_argument('--cap', type=int, default=6)
    ap.add_argument('--long', type=int, default=40)
    ap.add_argument('--detour2', type=int, default=20)
    ap.add_argument('--sweeps', type=int, default=120)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--prune-prob', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.validate:
        kw = dict(n_nodes=1200, cap=6, n_long=40, n_detour2=20, sweeps=120,
                  seeds=5, prune_prob=args.prune_prob)
        if args.quick:
            kw.update(n_nodes=400, n_long=10, n_detour2=5, sweeps=40, seeds=2)
        raise SystemExit(0 if _validate(**kw) else 1)

    # A production-scale run IS the checkpoint at the requested scale.
    raise SystemExit(0 if _validate(
        n_nodes=args.nodes, cap=args.cap, n_long=args.long,
        n_detour2=args.detour2, sweeps=args.sweeps, seeds=args.seeds,
        prune_prob=args.prune_prob) else 1)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the quick gate**

Run: `python async_censorship.py --validate --quick`
Expected: Gate 1 table prints with small z values and `advantage-blind OK`; Gate 2 table + a `VERDICT:` line; final `PASS`. (At `--quick` N=400 the numbers are noisy — this step only checks the harness runs end-to-end and Gate 1 does not blow up; the real numbers come from the full run in Task 5.)

- [ ] **Step 3: Commit**

```bash
git add async_censorship.py
git commit -m "async_censorship: two-gate --validate (P1 instrument gate + P2 descriptive race)"
# end with the standard Co-Authored-By / Claude-Session trailer
```

---

### Task 5: run the checkpoint at scale, record the result, update docs

**Files:**
- Run: `async_censorship.py` (full scale)
- Modify: `FINDINGS.md` (new subsection under the censorship material), `LORENTZIAN_SPIKE.md` (§6 step 4 → done + outcome), `README.md` (driver-table row for `async_censorship.py`)
- Modify (outside git): the emergence-research-state memory file

**Interfaces:** none produced; this task records outcomes.

- [ ] **Step 1: Run the full checkpoint and capture output**

Run: `python async_censorship.py --validate` (N=1200, 5 seeds — the frozen gate scale), then a larger confirmation `python async_censorship.py --nodes 2000 --seeds 3` (matches the banked sync scale).
Save both console outputs. Record: Gate 1 z-values + advantage-blindness, and Gate 2's sync-vs-async `long_survival` / `woven` / `detour2_survival` / `collateral` and the printed VERDICT.

- [ ] **Step 2: Interpret honestly against the pre-registration**

Decide which pre-registered Gate-2 outcome occurred (weaving survives vs collapses). **Do not** retune tolerances to force Gate 1; if Gate 1 *fails*, that is itself a finding (async P1 differs from sync) — investigate it (most likely the degree-floor coupling) and report it, do not paper over it. The exit-code PASS means "Gate 1 + confounds held"; the VERDICT line is the physics result either way.

- [ ] **Step 3: Write the FINDINGS.md entry**

Add a subsection near the censorship material (after the §"Censorship: threshold and advantage-blind" block, ~FINDINGS.md:818) titled with the actual result, e.g.:
`### Censorship under async (step 4, 2026-08-03): P1 schedule-invariant; P2 <survives|collapses>`
Include: the two-gate table (sync vs async), the pre-committed nulls, the VERDICT, and one paragraph of mechanism. Mirror the honest, numbers-first style of the step-3 entry. Update the ladder line (`FINDINGS.md` §checkpoint list, ~line 343) to mark step 4 done with its one-line outcome.

- [ ] **Step 4: Update LORENTZIAN_SPIKE.md §6 and README.md**

`LORENTZIAN_SPIKE.md`: mark step 4 DONE in the §6 ladder with the outcome (P1 reproduced; P2 verdict), and note step 5 (barrier + causal-future-growth) is the remaining rung-1 work. `README.md`: add an `async_censorship.py` row to the analysis/driver table alongside `causal_dag.py`, with the `--validate` invocation.

- [ ] **Step 5: Commit the code+docs milestone**

```bash
git add async_censorship.py async_engine.py FINDINGS.md LORENTZIAN_SPIKE.md README.md
git commit -m "Step 4: shortcut censorship under async updates -- P1 reproduced, P2 <survives|collapses>"
# end with the standard Co-Authored-By / Claude-Session trailer
git push
```

- [ ] **Step 6: Update the research memory** (outside git)

Update `C:\Users\Snarf\.claude\projects\F--science-projects-graph-graph\memory\emergence-research-state.md` with the step-4 outcome and set "Next per plan" to step 5.

---

## Self-Review

**1. Spec coverage** (checked each spec section against a task):
- `run_sequential_multi` + `max_time` + `params` + `on_event` + bit-identity equivalence → Task 1. ✓
- New module `async_censorship.py`, state-graph only, no causal import → Tasks 2–4 (constraint stated). ✓
- Shared base+portals; `inject_shortcuts` long (d≥6) + detour-2 → Task 2 `build_base_and_portals`. ✓
- Identical observable reduction for both schedules → Task 2 `summarize_portals`. ✓
- Async tracked conditions (per-event removal, sweep-equivalent = absolute Poisson time) → Task 3 `async_condition`. ✓
- Sync reference reused verbatim → Task 3 `sync_condition` (wraps `run_condition`). ✓
- Time-matching (rate-1 clocks; equal opportunity; compare at `sweeps`) → Task 3 (`max_time=sweeps`, rates `[1,1]`) + Task 4 event-count confound assertion. ✓
- Gate 1 distributional agreement (z<3) + |rank-corr|<0.2 → Task 4. ✓
- Gate 2 descriptive, pre-registered both ways, doesn't gate exit → Task 4 (`VERDICT`, `ok` unaffected by verdict). ✓
- Confound: detour-2 survival reported under async; per-node event counts matched/asserted → Task 4. ✓
- Run scale N≈1000–2000, ~120 sweeps, matched to sync; `--quick` smoke → Task 4 `main`, Task 5 run. ✓
- Deferred: ricci condition, batched/large-N, causal measurement, barrier → not implemented (correctly out of scope). ✓

**2. Placeholder scan:** No TBD/TODO; every code step is complete runnable code; the `<survives|collapses>` markers in Task 5 are deliberate — the actual word is filled from the measured VERDICT, not a plan gap.

**3. Type consistency:** `run_sequential_multi` signature identical in Task 1 interface, implementation, and both call sites (Task 3). `summarize_portals` keys (`long_survival, detour2_survival, mean_removal, adv_corr, woven, collateral`) defined in Task 2 and consumed unchanged in Task 4. `async_condition` returns `(removal, final, events_per_node)` and Task 4 unpacks exactly that. `removal` keyed by raw `(u,v)` in both `async_condition` (Task 3) and `run_condition`'s `removal_step` (sync), matching `summarize_portals`'s key convention. `_REWIRE_PROB = 0.05` matches `run_condition`'s hardcoded value (Global Constraints). ✓
