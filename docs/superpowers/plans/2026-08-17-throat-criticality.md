# Throat Criticality (Stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run `throat_criticality.py` — the stage-1 critical-collapse experiment: does censor-only dynamics show a sharp threshold `A*` in wormhole-throat thickness (evaporation vs a permanent mutually-protected core), or another crossover?

**Architecture:** One new driver module. Deterministic core: `build_arena` (grown base + two separated BFS balls) → nested-strand throats → `peel` (bootstrap fixed point). Because the peeling core is **monotone in nested strand sets** (adding a strand can only add protection), each ensemble draw bisects its critical thickness `A*_j` in ~log2(capacity) peels, and `P(core|a)` is exactly the CDF of the `a*_j` distribution — the spec's pre-committed observable, computed ~100× cheaper than an A-scan. Stochastic side reuses the step-4 engine (`run_sequential_multi`) for the Gate-1 peeling≡dynamics anchor (with explicit degree-floor mismatch attribution) and the Gate-2 triadic rider.

**Tech Stack:** Python 3.10+, NetworkX, NumPy, tqdm. No pytest — `--validate` self-checks + scratch RED→GREEN scripts, per repo convention.

**Spec:** `docs/superpowers/specs/2026-08-17-throat-criticality-design.md`

## Global Constraints

- Python 3.10+; type hints on public functions; NumPy-style docstrings; `tqdm` for long loops; CLI seeds both `random` and `numpy.random` from `--seed`.
- **Banked dynamics params, frozen:** `prune_prob=0.05, min_overlap=1, min_degree=2`; triadic rider `rewire_prob=0.05`; both clocks rate 1.0. Production anchor `T=400` sweeps; `_validate` uses `T=200` (10 geometric lifetimes — documented).
- **Reference geometry** `r=2, N=2000, cap=6`; FSS geometries `(r,N) = (2,2000), (3,5000), (4,10000)`; ball-center separation `dist(c1,c2) >= 2r+6`.
- **Verdict rules frozen (spec):** sharpness = `width(r)` (a-span, P 0.1→0.9 = 10th→90th percentile of `a*_j`) monotonically decreasing across the three capacities; jump and slowing-down are secondary, reported either way, never rescue a verdict; a crossover is banked with equal prominence.
- **Gate 1 gates the exit; Gate 2 is descriptive** and must never touch the exit code.
- State-graph only: no `causal_dag` import. No node positions anywhere (BFS distances only). Strand edges get the standard `weight=0.5`.
- Commits go directly to `main`; end every commit message with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01376hyaQjsoNiLdEGArP8fo`
- Scratch tests live in `C:\Users\Snarf\AppData\Local\Temp\claude\F--science-projects-graph-graph\9784062b-5b9c-402d-94ac-0fcf88b4fad4\scratchpad` and are not committed. Run python from the repo root (`F:\science-projects\graph-graph`); if imports fail set `PYTHONPATH` to the repo root.

## File Structure

- **Create `throat_criticality.py`** — everything: arena/throat construction, peeling, dynamics wrapper, mismatch attribution, ensemble bisection, anchor harness, `_validate`, `main`. (Repo convention: one standalone CLI per experiment.)
- **Task 5 modifies** `FINDINGS.md`, `BRANCHES.md`, `README.md` (+ the memory file, outside git).

---

### Task 1: arena, nested throats, and the peeling fixed point

**Files:**
- Create: `throat_criticality.py` (module header, imports, `Strand`, `build_arena`, `strand_pairs`, `throat_with_strands`, `peel`)
- Test: scratchpad `test_throat_core.py`

**Interfaces:**
- Consumes: `create_initial_graph` (`simulation.py`).
- Produces (later tasks rely on these exact signatures):
  - `Strand = Tuple[int, int]`
  - `build_arena(n_nodes: int, cap: int, r: int, seed: int, max_tries: int = 50) -> Tuple[nx.Graph, List[int], List[int]]` → `(base, ball1, ball2)`; balls disjoint, centers `>= 2r+6` apart; raises `RuntimeError` if no far pair found.
  - `strand_pairs(ball1: List[int], ball2: List[int]) -> List[Strand]` → all `|B1|*|B2|` pairs (capacity = its length).
  - `throat_with_strands(base: nx.Graph, strands: Sequence[Strand]) -> nx.Graph` → copy of base + strand edges (weight 0.5).
  - `peel(G: nx.Graph, strands: Sequence[Strand], min_overlap: int = 1) -> Tuple[Set[Strand], int]` → `(core, n_rounds)`; G not mutated.

- [ ] **Step 1: Write the failing test** (scratchpad `test_throat_core.py`)

```python
import networkx as nx
import numpy as np
from throat_criticality import (build_arena, strand_pairs,
                                throat_with_strands, peel)

def _hand_base():
    # u1-u2 adjacent (near side), v1, v2 far side; pendant triangles keep
    # every endpoint's degree >= 3 so the degree floor never binds here.
    G = nx.Graph()
    G.add_edge('u1', 'u2', weight=0.5)
    for x in ('u1', 'u2', 'v1', 'v2'):
        G.add_edge(x, f'{x}a', weight=0.5)
        G.add_edge(x, f'{x}b', weight=0.5)
        G.add_edge(f'{x}a', f'{x}b', weight=0.5)
    return G

def test_peel_positive_pair():
    # (u1,v1) and (u2,v1): each protects the other via the shared far
    # endpoint + adjacent anchors -> core = both, exactly.
    G = _hand_base()
    strands = [('u1', 'v1'), ('u2', 'v1')]
    H = throat_with_strands(G, strands)
    core, rounds = peel(H, strands)
    assert core == set(strands), core
    print("PASS peel positive")

def test_peel_negative_isolated():
    # (u1,v1) and (u2,v2): no shared endpoints, v1 !~ v2 -> zero common
    # neighbors each -> core empty; peeling takes exactly 1 round.
    G = _hand_base()
    strands = [('u1', 'v1'), ('u2', 'v2')]
    H = throat_with_strands(G, strands)
    core, rounds = peel(H, strands)
    assert core == set() and rounds == 1, (core, rounds)
    print("PASS peel negative")

def test_peel_cascade():
    # (u1,v1),(u2,v1),(u2,v2): the pair protects itself; (u2,v2) hangs off
    # it unprotected (common nbr of u2,v2? none: v1 !~ v2) -> cascades away.
    G = _hand_base()
    strands = [('u1', 'v1'), ('u2', 'v1'), ('u2', 'v2')]
    H = throat_with_strands(G, strands)
    core, _ = peel(H, strands)
    assert core == {('u1', 'v1'), ('u2', 'v1')}, core
    print("PASS peel cascade")

def test_arena_invariants():
    base, b1, b2 = build_arena(1000, 6, 2, seed=0)
    assert not set(b1) & set(b2)
    d = nx.shortest_path_length(base, b1[0], b2[0])
    # centers are b1[0], b2[0] (documented); separation >= 2r+6 = 10
    assert d >= 10, d
    pairs = strand_pairs(b1, b2)
    assert len(pairs) == len(b1) * len(b2)
    assert all(not base.has_edge(u, v) for u, v in pairs)
    b1b, b2b = build_arena(1000, 6, 2, seed=0)[1:]
    assert (b1, b2) == (b1b, b2b)  # deterministic in seed
    print("PASS arena invariants")

def test_peel_monotone_nested():
    # core existence is monotone in nested strand sets (the bisection's
    # correctness condition): if core nonempty at the first A, it stays
    # nonempty for every larger prefix of the same permutation.
    base, b1, b2 = build_arena(800, 6, 2, seed=3)
    pairs = strand_pairs(b1, b2)
    rng = np.random.default_rng(3)
    perm = [pairs[i] for i in rng.permutation(len(pairs))]
    seen_core = False
    for A in range(1, len(perm) + 1, max(1, len(perm) // 40)):
        H = throat_with_strands(base, perm[:A])
        core, _ = peel(H, perm[:A])
        if seen_core:
            assert core, f"monotonicity broken at A={A}"
        seen_core = seen_core or bool(core)
    print("PASS monotone nesting")

if __name__ == '__main__':
    test_peel_positive_pair()
    test_peel_negative_isolated()
    test_peel_cascade()
    test_arena_invariants()
    test_peel_monotone_nested()
    print("PASS all")
```

- [ ] **Step 2: Run it, verify it FAILS**

Run (repo root): `python "<scratchpad>\test_throat_core.py"` with `PYTHONPATH` set to the repo root.
Expected: FAIL — `ModuleNotFoundError: No module named 'throat_criticality'`.

- [ ] **Step 3: Implement** (create `throat_criticality.py`)

```python
"""
Wormhole-throat critical collapse -- stage 1 of the critical-collapse program
(design spec docs/superpowers/specs/2026-08-17-throat-criticality-design.md).

Does censor-only dynamics (`prune`) show a sharp threshold A* in throat
thickness -- evaporation vs a permanent mutually-protected strand core -- or
another crossover? Physics core: under prune-only the outcome is the
bootstrap-peeling fixed point of the initial throat geometry (randomness sets
timing only, modulo the degree floor, which Gate 1 measures). The threshold
therefore lives in the random-ensemble P(core | a), computed here as the CDF
of per-draw critical densities a*_j found by bisection over nested strands
(core existence is monotone in nested strand sets).

  Gate 1 (instrument + mechanism, gates the exit): stochastic prune-only
    dynamics land on the peeling fixed point seed-by-seed; every mismatch
    must be degree-floor-attributable.
  Gate 2 (the race, descriptive): can triadic weaving rescue a throat the
    peeling fixed point condemns, or demolish a core it promises?

Usage:
    python throat_criticality.py --validate [--quick]
    python throat_criticality.py --pilot   [--draws 300]
    python throat_criticality.py --fss     [--draws 2000]
    python throat_criticality.py --anchor --a-values 0.05 0.1 0.2 [--seeds 8]
"""

import argparse
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import networkx as nx
from tqdm import tqdm

from simulation import create_initial_graph
from async_engine import run_sequential_multi

Strand = Tuple[int, int]


def build_arena(n_nodes: int, cap: int, r: int, seed: int,
                max_tries: int = 50) -> Tuple[nx.Graph, List[int], List[int]]:
    """
    Grown base + two disjoint BFS balls whose centers are >= 2r+6 apart.

    Returns (base, ball1, ball2); ball lists start with their center node.
    Deterministic in `seed`. Raises RuntimeError if no sufficiently far
    center pair exists after `max_tries` candidate centers.
    """
    random.seed(seed)
    np.random.seed(seed)
    base = create_initial_graph(n_nodes, topology='grown', k=cap, seed=seed)
    nodes = list(base.nodes())
    rng = np.random.default_rng(seed)
    min_sep = 2 * r + 6

    for _ in range(max_tries):
        c1 = nodes[int(rng.integers(len(nodes)))]
        dist = nx.single_source_shortest_path_length(base, c1)
        far = [w for w, d in dist.items() if d >= min_sep]
        if not far:
            continue
        c2 = far[int(rng.integers(len(far)))]
        d1 = nx.single_source_shortest_path_length(base, c1, cutoff=r)
        d2 = nx.single_source_shortest_path_length(base, c2, cutoff=r)
        ball1 = [c1] + sorted(w for w in d1 if w != c1)
        ball2 = [c2] + sorted(w for w in d2 if w != c2)
        return base, ball1, ball2
    raise RuntimeError(
        f"no center pair at distance >= {min_sep} in {max_tries} tries "
        f"(N={n_nodes} too small for r={r}?)")


def strand_pairs(ball1: List[int], ball2: List[int]) -> List[Strand]:
    """All |B1|*|B2| candidate strands; len(...) is the throat capacity."""
    return [(u, v) for u in ball1 for v in ball2]


def throat_with_strands(base: nx.Graph,
                        strands: Sequence[Strand]) -> nx.Graph:
    """Copy of `base` with the given strand edges added (standard weight)."""
    G = base.copy()
    for u, v in strands:
        G.add_edge(u, v, weight=0.5)
    return G


def peel(G: nx.Graph, strands: Sequence[Strand],
         min_overlap: int = 1) -> Tuple[Set[Strand], int]:
    """
    Bootstrap-peeling fixed point: iteratively remove strands with fewer
    than `min_overlap` common neighbors in the current graph (base +
    surviving strands). Returns (core, n_rounds). `G` is not mutated.

    Core existence is MONOTONE in nested strand sets (adding a strand only
    ever adds protection) -- the correctness condition for the bisection in
    `critical_density_draws`.
    """
    H = G.copy()
    alive: Set[Strand] = set(strands)
    rounds = 0
    while True:
        doomed = [s for s in alive
                  if len(set(H[s[0]]) & set(H[s[1]])) < min_overlap]
        if not doomed:
            return alive, rounds
        for u, v in doomed:
            H.remove_edge(u, v)
        alive -= set(doomed)
        rounds += 1
```

- [ ] **Step 4: Run the test, verify it PASSES**

Same command as Step 2. Expected: all five `PASS` lines. If `test_arena_invariants` raises RuntimeError, N=1000 lacks far pairs — that would contradict grown's banked diameter (~40 at N=2000); investigate before proceeding, do not weaken the assert.

- [ ] **Step 5: Commit**

```bash
git add throat_criticality.py
git commit -m "throat_criticality: arena, nested throats, peeling fixed point"
# + the standard two trailer lines from Global Constraints
```

---

### Task 2: stochastic dynamics wrapper + degree-floor mismatch attribution

**Files:**
- Modify: `throat_criticality.py` (add `run_dynamics`, `classify_mismatches`)
- Test: scratchpad `test_throat_dynamics.py`

**Interfaces:**
- Consumes: `run_sequential_multi` (`async_engine.py`, step-4 engine: `(G, rules, rates, max_time=..., seed=..., params=..., on_event=...) -> (finalG, times, rule_ids)`; copies G internally); Task-1 helpers.
- Produces:
  - `run_dynamics(G: nx.Graph, strands: Sequence[Strand], rules: Sequence[str], params: Sequence[Optional[Dict]], sweeps: float, seed: int) -> Tuple[Set[Strand], Dict[Strand, Optional[float]], nx.Graph]` → `(survivors, death_time, final_graph)`; survivors judged from final-graph edge membership; `death_time` = first-absence stamp in absolute Poisson time (None = never absent).
  - `classify_mismatches(final_graph: nx.Graph, peel_core: Set[Strand], survivors: Set[Strand], min_overlap: int = 1, min_degree: int = 2) -> Dict[str, List[Strand]]` with keys `'missing'` (peel-core strands the dynamics lost — must be empty), `'excess_floor'` (excess survivor with a final-graph endpoint degree <= min_degree), `'excess_protected'` (excess survivor with >= min_overlap common neighbors in the final graph — legitimately protected downstream of a floor event), `'excess_unattributed'` (neither — Gate-1 failures).

- [ ] **Step 1: Write the failing test** (scratchpad `test_throat_dynamics.py`)

```python
import numpy as np
from throat_criticality import (build_arena, strand_pairs,
                                throat_with_strands, peel, run_dynamics,
                                classify_mismatches)

PRUNE = (['prune'], [{'prune_prob': 0.05}])

def test_gate1_small():
    # Peeling must predict prune-only dynamics: core subset of survivors,
    # every excess attributable. T=200 sweeps = 10 lifetimes.
    bad = 0
    for seed in range(4):
        base, b1, b2 = build_arena(600, 6, 2, seed=seed)
        pairs = strand_pairs(b1, b2)
        rng = np.random.default_rng(seed)
        perm = [pairs[i] for i in rng.permutation(len(pairs))]
        for A in (max(2, len(pairs) // 10), max(4, len(pairs) // 3)):
            strands = perm[:A]
            H = throat_with_strands(base, strands)
            core, _ = peel(H, strands)
            surv, death, final = run_dynamics(
                H, strands, PRUNE[0], PRUNE[1], sweeps=200.0,
                seed=1000 + seed)
            cls = classify_mismatches(final, core, surv)
            assert not cls['missing'], cls['missing']
            bad += len(cls['excess_unattributed'])
            print(f"seed={seed} A={A}: |core|={len(core)} "
                  f"|surv|={len(surv)} floor={len(cls['excess_floor'])} "
                  f"prot={len(cls['excess_protected'])} "
                  f"unattr={len(cls['excess_unattributed'])}")
    assert bad == 0, f"{bad} unattributed mismatches"
    print("PASS gate1 small")

def test_death_times_sane():
    base, b1, b2 = build_arena(600, 6, 2, seed=9)
    pairs = strand_pairs(b1, b2)
    strands = pairs[:3]  # sparse -> all unprotected -> all die
    H = throat_with_strands(base, strands)
    core, _ = peel(H, strands)
    assert core == set()
    surv, death, _ = run_dynamics(H, strands, PRUNE[0], PRUNE[1],
                                  sweeps=200.0, seed=7)
    times = [t for t in death.values() if t is not None]
    # geometric mean lifetime ~ 1/0.05 = 20 sweeps; all should be stamped
    # well before 200 unless floor-stuck
    assert len(times) >= 2 and all(0 < t <= 200 for t in times)
    print("PASS death times")

if __name__ == '__main__':
    test_gate1_small()
    test_death_times_sane()
    print("PASS all")
```

- [ ] **Step 2: Run it, verify it FAILS**

Expected: `ImportError: cannot import name 'run_dynamics'`.

- [ ] **Step 3: Implement** (add to `throat_criticality.py`)

```python
def run_dynamics(G: nx.Graph, strands: Sequence[Strand],
                 rules: Sequence[str], params: Sequence[Optional[Dict]],
                 sweeps: float, seed: int
                 ) -> Tuple[Set[Strand], Dict[Strand, Optional[float]],
                            nx.Graph]:
    """
    Evolve a throat under the step-4 engine, tracking per-strand death.

    Survivors are judged from FINAL-GRAPH edge membership (robust to a rule
    re-creating an edge); `death_time` records the first absence stamp in
    absolute Poisson time (= sweep-equivalents at rate 1), None if never
    absent. `run_sequential_multi` copies G, so the caller's graph is
    untouched.
    """
    death: Dict[Strand, Optional[float]] = {s: None for s in strands}
    by_node: Dict[int, List[Strand]] = {}
    for u, v in strands:
        by_node.setdefault(u, []).append((u, v))
        by_node.setdefault(v, []).append((u, v))

    def on_event(event_id: int, node: int, rule_idx: int, t: float,
                 H: nx.Graph) -> None:
        for (u, v) in by_node.get(node, ()):
            if death[(u, v)] is None and not H.has_edge(u, v):
                death[(u, v)] = t

    final, _, _ = run_sequential_multi(
        G, list(rules), [1.0] * len(rules), max_time=float(sweeps),
        seed=seed, params=list(params), on_event=on_event)
    survivors = {s for s in strands if final.has_edge(*s)}
    return survivors, death, final


def classify_mismatches(final_graph: nx.Graph, peel_core: Set[Strand],
                        survivors: Set[Strand], min_overlap: int = 1,
                        min_degree: int = 2) -> Dict[str, List[Strand]]:
    """
    Gate-1 accounting. Peel-core strands must all survive ('missing' empty).
    Excess survivors are attributed: 'excess_floor' (an endpoint sits at or
    below prune's degree floor in the final graph), 'excess_protected'
    (>= min_overlap common neighbors in the final graph -- legitimately
    protected in the final configuration, downstream of a floor event), or
    'excess_unattributed' (a genuine Gate-1 failure).
    """
    out: Dict[str, List[Strand]] = {'missing': [], 'excess_floor': [],
                                    'excess_protected': [],
                                    'excess_unattributed': []}
    out['missing'] = sorted(peel_core - survivors)
    for s in sorted(survivors - peel_core):
        u, v = s
        if min(final_graph.degree(u), final_graph.degree(v)) <= min_degree:
            out['excess_floor'].append(s)
        elif len(set(final_graph[u]) & set(final_graph[v])) >= min_overlap:
            out['excess_protected'].append(s)
        else:
            out['excess_unattributed'].append(s)
    return out
```

- [ ] **Step 4: Run the test, verify it PASSES**

Expected: per-seed lines then `PASS gate1 small`, `PASS death times`, `PASS all`. Runtime ~3–6 min (8 sequential runs at N=600×200 sweeps). If `excess_unattributed` is nonzero, that is a REAL finding about the mechanism — do NOT loosen `classify_mismatches`; report BLOCKED with the counts.

- [ ] **Step 5: Commit**

```bash
git add throat_criticality.py
git commit -m "throat_criticality: dynamics wrapper + degree-floor mismatch attribution"
# + trailers
```

---

### Task 3: ensemble bisection, transition stats, anchor harness

**Files:**
- Modify: `throat_criticality.py` (add `critical_density_draws`, `transition_stats`, `anchor_runs`)
- Test: scratchpad `test_throat_ensemble.py`

**Interfaces:**
- Consumes: Task-1/2 functions, exact signatures above.
- Produces:
  - `critical_density_draws(n_nodes: int, cap: int, r: int, n_draws: int, seed0: int, progress: bool = False) -> List[Dict]` — per draw: `{'draw', 'capacity', 'A_star', 'a_star', 'core_frac'}`; `A_star` = smallest A (nested permutation) with nonempty core, found by bisection; `a_star = A_star/capacity`; `core_frac` = |core|/A at `A_star`; a draw whose FULL throat has empty core gets `A_star = -1, a_star = np.inf` (counted, excluded from quantiles).
  - `transition_stats(a_stars: Sequence[float]) -> Dict[str, float]` — `{'a10','a50','a90','width','n_finite','n_total'}` (quantiles over finite values; `width = a90 - a10`).
  - `anchor_runs(n_nodes: int, cap: int, r: int, a_values: Sequence[float], seeds: int, sweeps: float, rider: bool, seed0: int = 5000) -> List[Dict]` — per (a, seed): build arena+throat at `A = max(1, round(a*capacity))`, peel, prune-only `run_dynamics`, `classify_mismatches`; row keys `'a','seed','capacity','A','core','surv','n_missing','n_floor','n_protected','n_unattributed','evap_time'` (`evap_time` = max death stamp when the peel core is empty AND all strands died, else None); if `rider`, additional keys `'rider_surv','rider_core_kept'` from a triadic+prune run on the SAME throat (seed offset +50000): `rider_surv` = surviving strand count, `rider_core_kept` = |peel_core ∩ rider survivors|.

- [ ] **Step 1: Write the failing test** (scratchpad `test_throat_ensemble.py`)

```python
import numpy as np
from throat_criticality import (critical_density_draws, transition_stats,
                                anchor_runs, build_arena, strand_pairs,
                                throat_with_strands, peel)

def test_bisection_matches_linear_scan():
    # The bisection must find exactly the smallest nested-A with a core.
    rows = critical_density_draws(800, 6, 2, n_draws=3, seed0=42)
    for row in rows:
        if row['A_star'] < 0:
            continue
        base, b1, b2 = build_arena(800, 6, 2, seed=42 + row['draw'])
        pairs = strand_pairs(b1, b2)
        rng = np.random.default_rng(42 + row['draw'])
        perm = [pairs[i] for i in rng.permutation(len(pairs))]
        A = row['A_star']
        core_at, _ = peel(throat_with_strands(base, perm[:A]), perm[:A])
        assert core_at, "core must exist at A_star"
        if A > 1:
            core_below, _ = peel(throat_with_strands(base, perm[:A - 1]),
                                 perm[:A - 1])
            assert not core_below, "core must NOT exist at A_star - 1"
    print("PASS bisection exact")

def test_transition_stats():
    s = transition_stats([0.1, 0.2, 0.3, 0.4, np.inf])
    assert s['n_finite'] == 4 and s['n_total'] == 5
    assert abs(s['width'] - (s['a90'] - s['a10'])) < 1e-12
    print("PASS stats")

def test_anchor_rows_shape():
    rows = anchor_runs(600, 6, 2, a_values=[0.05], seeds=2, sweeps=200.0,
                       rider=True)
    assert len(rows) == 2
    for row in rows:
        assert row['n_missing'] == 0 and row['n_unattributed'] == 0
        assert 'rider_surv' in row and 'rider_core_kept' in row
    print("PASS anchor shape")

if __name__ == '__main__':
    test_bisection_matches_linear_scan()
    test_transition_stats()
    test_anchor_rows_shape()
    print("PASS all")
```

- [ ] **Step 2: Run it, verify it FAILS**

Expected: `ImportError: cannot import name 'critical_density_draws'`.

- [ ] **Step 3: Implement** (add to `throat_criticality.py`)

```python
def critical_density_draws(n_nodes: int, cap: int, r: int, n_draws: int,
                           seed0: int, progress: bool = False) -> List[Dict]:
    """
    Per-draw critical throat thickness by bisection over nested strands.

    Draw j: one arena (seed0+j) + one uniform permutation of all capacity
    pairs; A*_j = smallest prefix length whose peeling core is nonempty
    (valid because core existence is monotone in nested prefixes). P(core|a)
    is the CDF of a*_j = A*_j/capacity -- the spec's pre-committed ensemble
    observable at ~log2(capacity) peels per draw.
    """
    rows: List[Dict] = []
    it = tqdm(range(n_draws), desc=f"draws r={r}") if progress \
        else range(n_draws)
    for j in it:
        seed = seed0 + j
        base, b1, b2 = build_arena(n_nodes, cap, r, seed=seed)
        pairs = strand_pairs(b1, b2)
        cap_j = len(pairs)
        rng = np.random.default_rng(seed)
        perm = [pairs[i] for i in rng.permutation(cap_j)]

        def core_at(A: int):
            return peel(throat_with_strands(base, perm[:A]), perm[:A])[0]

        top = core_at(cap_j)
        if not top:
            rows.append({'draw': j, 'capacity': cap_j, 'A_star': -1,
                         'a_star': float('inf'), 'core_frac': 0.0})
            continue
        lo, hi = 0, cap_j          # invariant: core(lo) empty, core(hi) not
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if core_at(mid):
                hi = mid
            else:
                lo = mid
        rows.append({'draw': j, 'capacity': cap_j, 'A_star': hi,
                     'a_star': hi / cap_j,
                     'core_frac': len(core_at(hi)) / hi})
    return rows


def transition_stats(a_stars: Sequence[float]) -> Dict[str, float]:
    """10/50/90 percentiles and width of the finite a* distribution."""
    a = np.asarray(list(a_stars), dtype=float)
    finite = a[np.isfinite(a)]
    if len(finite) == 0:
        return {'a10': float('nan'), 'a50': float('nan'),
                'a90': float('nan'), 'width': float('nan'),
                'n_finite': 0, 'n_total': len(a)}
    q10, q50, q90 = np.quantile(finite, [0.1, 0.5, 0.9])
    return {'a10': float(q10), 'a50': float(q50), 'a90': float(q90),
            'width': float(q90 - q10), 'n_finite': int(len(finite)),
            'n_total': int(len(a))}


def anchor_runs(n_nodes: int, cap: int, r: int, a_values: Sequence[float],
                seeds: int, sweeps: float, rider: bool,
                seed0: int = 5000) -> List[Dict]:
    """
    Gate-1 (peeling == prune-only dynamics, floor-attributed) and the
    timing observable; optional Gate-2 triadic rider on the same throats.
    """
    prune_params = [{'prune_prob': 0.05}]
    tp_params = [{'rewire_prob': 0.05}, {'prune_prob': 0.05}]
    rows: List[Dict] = []
    for a in a_values:
        for s in tqdm(range(seeds), desc=f"anchor a={a}"):
            seed = seed0 + s
            base, b1, b2 = build_arena(n_nodes, cap, r, seed=seed)
            pairs = strand_pairs(b1, b2)
            rng = np.random.default_rng(seed)
            perm = [pairs[i] for i in rng.permutation(len(pairs))]
            A = max(1, round(a * len(pairs)))
            strands = perm[:A]
            H = throat_with_strands(base, strands)
            core, _ = peel(H, strands)
            surv, death, final = run_dynamics(
                H, strands, ['prune'], prune_params, sweeps,
                seed=seed + 10000)
            cls = classify_mismatches(final, core, surv)
            evap = None
            if not core and all(t is not None for t in death.values()):
                evap = max(death.values())
            row = {'a': a, 'seed': s, 'capacity': len(pairs), 'A': A,
                   'core': len(core), 'surv': len(surv),
                   'n_missing': len(cls['missing']),
                   'n_floor': len(cls['excess_floor']),
                   'n_protected': len(cls['excess_protected']),
                   'n_unattributed': len(cls['excess_unattributed']),
                   'evap_time': evap}
            if rider:
                r_surv, _, _ = run_dynamics(
                    H, strands, ['triadic', 'prune'], tp_params, sweeps,
                    seed=seed + 50000)
                row['rider_surv'] = len(r_surv)
                row['rider_core_kept'] = len(core & r_surv)
            rows.append(row)
    return rows
```

- [ ] **Step 4: Run the test, verify it PASSES**

Expected: three `PASS` lines + `PASS all`. Runtime ~4–8 min (bisection is fast; the anchor rows run 4 dynamics at N=600). The bisection-vs-linear-scan check is the load-bearing one — if it fails, the monotonicity assumption or the bisection invariant is wrong; stop and investigate, do not paper over.

- [ ] **Step 5: Commit**

```bash
git add throat_criticality.py
git commit -m "throat_criticality: ensemble bisection, transition stats, anchor harness"
# + trailers
```

---

### Task 4: `_validate()` + `main()` CLI

**Files:**
- Modify: `throat_criticality.py` (add `_validate`, `main`, `__main__` guard)

**Interfaces:**
- Consumes: everything above.
- Produces: `python throat_criticality.py --validate [--quick]` → exit 0 iff hand-built known-answers exact AND Gate-1 small-scale clean (no missing, no unattributed) AND mini-pilot machinery sane; `--pilot/--fss/--anchor` production modes print tables (they are research runs — their exit code is 0 unless the machinery itself fails).

- [ ] **Step 1: Implement** (append to `throat_criticality.py`)

```python
def _hand_graph() -> nx.Graph:
    """Known-answer base: u1~u2 near side, v1/v2 far side, pendant triangles
    keep all endpoint degrees >= 3 so the degree floor never binds."""
    G = nx.Graph()
    G.add_edge('u1', 'u2', weight=0.5)
    for x in ('u1', 'u2', 'v1', 'v2'):
        G.add_edge(x, f'{x}a', weight=0.5)
        G.add_edge(x, f'{x}b', weight=0.5)
        G.add_edge(f'{x}a', f'{x}b', weight=0.5)
    return G


def _validate(quick: bool = False) -> bool:
    ok = True
    n_gate = 400 if quick else 600
    seeds_gate = 2 if quick else 4
    n_pilot = 600 if quick else 1000
    draws_pilot = 30 if quick else 60

    print("[1] hand-built known-answer throats (exact equality required)")
    G = _hand_graph()
    pos = [('u1', 'v1'), ('u2', 'v1')]
    core, _ = peel(throat_with_strands(G, pos), pos)
    good = core == set(pos)
    ok &= good
    print(f"  mutually-protecting pair -> core == both: "
          f"{'OK' if good else 'FAIL ' + str(core)}")
    neg = [('u1', 'v1'), ('u2', 'v2')]
    core, rounds = peel(throat_with_strands(G, neg), neg)
    good = core == set() and rounds == 1
    ok &= good
    print(f"  isolated strands -> empty core in 1 round: "
          f"{'OK' if good else 'FAIL ' + str((core, rounds))}")

    print(f"\n[2] Gate 1 small-scale: peeling == prune-only dynamics "
          f"(N={n_gate}, {seeds_gate} seeds, T=200)")
    n_floor = n_prot = 0
    for s in range(seeds_gate):
        base, b1, b2 = build_arena(n_gate, 6, 2, seed=s)
        pairs = strand_pairs(b1, b2)
        rng = np.random.default_rng(s)
        perm = [pairs[i] for i in rng.permutation(len(pairs))]
        for A in (max(2, len(pairs) // 10), max(4, len(pairs) // 3)):
            strands = perm[:A]
            H = throat_with_strands(base, strands)
            core, _ = peel(H, strands)
            surv, _, final = run_dynamics(H, strands, ['prune'],
                                          [{'prune_prob': 0.05}], 200.0,
                                          seed=1000 + s)
            cls = classify_mismatches(final, core, surv)
            bad = bool(cls['missing']) or bool(cls['excess_unattributed'])
            ok &= not bad
            n_floor += len(cls['excess_floor'])
            n_prot += len(cls['excess_protected'])
            if bad:
                print(f"  seed {s} A={A}: FAIL missing={cls['missing']} "
                      f"unattributed={cls['excess_unattributed']}")
    print(f"  all runs: core subset of survivors, every excess attributed "
          f"(floor {n_floor}, downstream-protected {n_prot})")

    print(f"\n[3] mini-pilot machinery (N={n_pilot}, {draws_pilot} draws)")
    rows = critical_density_draws(n_pilot, 6, 2, draws_pilot, seed0=100)
    stats = transition_stats([r['a_star'] for r in rows])
    sane = (stats['n_finite'] > 0
            and 0.0 < stats['a10'] <= stats['a90'] <= 1.0)
    ok &= sane
    print(f"  a* quantiles: a10={stats['a10']:.3f} a50={stats['a50']:.3f} "
          f"a90={stats['a90']:.3f} width={stats['width']:.3f} "
          f"(finite {stats['n_finite']}/{stats['n_total']}) "
          f"{'OK' if sane else 'FAIL'}")
    both = stats['a10'] > 0.02 and stats['a90'] < 0.98
    print(f"  both outcomes reachable inside (0,1): "
          f"{'yes' if both else 'WARN -- check ensemble design'}")

    print(f"\n{'PASS' if ok else 'FAIL'}: throat criticality instrument")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 1: wormhole-throat critical collapse.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validate', action='store_true')
    mode.add_argument('--pilot', action='store_true')
    mode.add_argument('--fss', action='store_true')
    mode.add_argument('--anchor', action='store_true')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--draws', type=int, default=None)
    ap.add_argument('--a-values', type=float, nargs='+', default=None)
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--sweeps', type=float, default=400.0)
    ap.add_argument('--rider', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.validate:
        raise SystemExit(0 if _validate(args.quick) else 1)

    if args.pilot:
        draws = args.draws or 300
        rows = critical_density_draws(2000, 6, 2, draws, seed0=200,
                                      progress=True)
        s = transition_stats([r['a_star'] for r in rows])
        cf = np.mean([r['core_frac'] for r in rows if r['A_star'] > 0])
        print(f"\nPILOT r=2 N=2000 ({draws} draws): "
              f"a10={s['a10']:.4f} a50={s['a50']:.4f} a90={s['a90']:.4f} "
              f"width={s['width']:.4f} finite={s['n_finite']}/{s['n_total']}"
              f" mean core_frac at A*={cf:.3f}")
        return

    if args.fss:
        draws = args.draws or 2000
        print(f"FSS: {draws} draws per geometry; width must shrink with "
              f"capacity for a SHARP verdict (frozen rule).")
        for (r, n) in ((2, 2000), (3, 5000), (4, 10000)):
            rows = critical_density_draws(n, 6, r, draws,
                                          seed0=300 + 17 * r, progress=True)
            s = transition_stats([row['a_star'] for row in rows])
            capm = np.mean([row['capacity'] for row in rows])
            cf = np.mean([row['core_frac'] for row in rows
                          if row['A_star'] > 0])
            print(f"  r={r} N={n}: capacity~{capm:.0f}  "
                  f"a50={s['a50']:.4f}  width={s['width']:.4f}  "
                  f"finite={s['n_finite']}/{s['n_total']}  "
                  f"core_frac@A*={cf:.3f}")
        return

    if args.anchor:
        if not args.a_values:
            raise SystemExit("--anchor needs --a-values (frozen from the "
                             "pilot quantiles)")
        rows = anchor_runs(2000, 6, 2, args.a_values, args.seeds,
                           args.sweeps, rider=args.rider)
        bad = sum(r['n_missing'] + r['n_unattributed'] for r in rows)
        print(f"\n{'a':>7s} {'seed':>4s} {'A':>4s} {'core':>4s} "
              f"{'surv':>4s} {'floor':>5s} {'prot':>4s} {'unattr':>6s} "
              f"{'evap_t':>7s}" + ("  rider(surv/kept)" if args.rider
                                   else ""))
        for r in rows:
            ev = f"{r['evap_time']:.1f}" if r['evap_time'] is not None \
                else '--'
            line = (f"{r['a']:7.4f} {r['seed']:4d} {r['A']:4d} "
                    f"{r['core']:4d} {r['surv']:4d} {r['n_floor']:5d} "
                    f"{r['n_protected']:4d} {r['n_unattributed']:6d} "
                    f"{ev:>7s}")
            if args.rider:
                line += f"  {r['rider_surv']}/{r['rider_core_kept']}"
            print(line)
        print(f"\nGATE 1: {'PASS' if bad == 0 else 'FAIL'} "
              f"({bad} missing/unattributed across {len(rows)} runs)")
        raise SystemExit(0 if bad == 0 else 1)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the validate gate**

Run: `python throat_criticality.py --validate --quick` then (if green) `python throat_criticality.py --validate`.
Expected: `[1]` both exact-equality OKs; `[2]` all excess attributed; `[3]` sane quantiles inside (0,1) and final `PASS`. Full validate ~5–10 min. A `WARN -- check ensemble design` in [3] does not fail the gate but MUST be quoted in your report (it is the slice-1a early signal).

- [ ] **Step 3: Commit**

```bash
git add throat_criticality.py
git commit -m "throat_criticality: --validate gate + pilot/fss/anchor CLI"
# + trailers
```

---

### Task 5: production runs, verdict, documentation (controller task)

**Files:**
- Run: `throat_criticality.py` (pilot → fss → anchor)
- Modify: `FINDINGS.md` (new stage-1 section), `BRANCHES.md` (stage-2/3 gates updated by the verdict), `README.md` (driver-table row)
- Modify (outside git): the emergence-research-state memory file

**Interfaces:** none produced; this task records outcomes.

- [ ] **Step 1: Slice 1a — pilot.** `python throat_criticality.py --pilot --draws 300`. Existence gate: both outcomes occur with an S-curve between (a10>0, a90<1, width nonzero). If P(core) is degenerate across the reachable range: STOP — the ensemble needs redesign; report to the owner rather than proceeding.
- [ ] **Step 2: Freeze production parameters.** From the pilot: record `a10/a50/a90`; freeze the anchor `--a-values` as `(a10-below, a25ish, a50, a75ish, a90-above)` — concretely: one value ~half of a10, plus the pilot's 0.25/0.5/0.75 quantiles, plus one ~1.5× a90. Write the frozen values into the SDD ledger before running.
- [ ] **Step 3: Slice 1b compute.** `--fss --draws 2000` (roughly 30–60 min total, local; jaga optional — power button required) and `--anchor --a-values <frozen> --seeds 8 --rider` (~2–5 h local; run in background). Capture both outputs.
- [ ] **Step 4: Verdict, per the frozen rules.** Sharpness: `width(r)` monotone decreasing across r=2/3/4 → SHARP; flat-within-spread → CROSSOVER. Jump: `core_frac@A*` per capacity (bounded away from 0 = hybrid signature). Slowing down: `evap_time` vs `a` approaching a* from below. Gate 1 must PASS (0 missing/unattributed) for any of it to be interpreted; Gate-2 rider is reported descriptively (sub-a* rescue = rider_surv >> prune-only surv where core=0; core demolition = rider_core_kept << core).
- [ ] **Step 5: Document.** FINDINGS: new section "Wormhole-throat critical collapse (stage 1, 2026-08-17)" with the verdict table, both gates, and the pre-registered reading that fired; BRANCHES: stage-2/3 gate rows updated per the verdict; README: driver-table row for `throat_criticality.py`. Commit with the standard trailers; push.
- [ ] **Step 6: Memory.** Update the emergence-research-state memory file with the stage-1 outcome and next gates.

---

## Self-Review

**1. Spec coverage:** ensemble/arena/separation/capacity/density → Task 1; peeling fixed point + monotonicity → Task 1 (tested explicitly); dynamics via step-4 engine + per-strand death + floor attribution taxonomy → Task 2; P(core|a)-as-CDF via bisection (same observable, cheaper — documented in docstring), width 0.1→0.9, jump stat, evap timing, rider → Task 3; hand-built known answers + Gate-1 validate + mini-pilot + CLI modes → Task 4; slices 1a/1b sequencing, frozen-after-pilot params, verdict rules, docs → Task 5. Slice 1c (full-dynamics FSS) is a contingency, correctly not planned unless Gate 1 fails. ✓

**2. Placeholder scan:** none; all code complete; production a-values are deliberately "frozen from pilot" per spec pre-commitment 4, with the freezing procedure spelled out (Task 5 Step 2). ✓

**3. Type consistency:** `Strand` tuple flows through `strand_pairs → throat_with_strands → peel → run_dynamics → classify_mismatches`; `critical_density_draws` row keys match `transition_stats` input and Task-5 usage; `anchor_runs` row keys match the `--anchor` printer exactly; validate [2] reuses Task-2 semantics unchanged. ✓
