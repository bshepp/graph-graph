"""
Step 3 of the Lorentzian upgrade (LORENTZIAN_SPIKE.md 6): the bridge between the
async event engine and the causal-set dimension estimators.

`async_engine.py` evolves a graph by local Poisson-clock events but throws the
parent structure away; `causal_sets.py` estimates dimension from a relation
matrix but only ever built one from a flat-space sprinkling. This module records
the *causal DAG of update events* during an async run, samples Alexandrov
intervals from it, and feeds each interval's relation matrix to the calibrated
Myrheim-Meyer and midpoint estimators.

The free calibration it exists to run: on a STATIC graph the event DAG is the
clean product `G x Poisson-time`, whose causal dimension must be `d_H + 1`
(d_H = spatial Hausdorff dimension from dimension.py). If it is not, the
instrument is broken, not the physics. See the design spec at
docs/superpowers/specs/2026-07-29-causal-dag-static-calibration-design.md.

The recorder is rule-agnostic on a static graph: causal parents come from the
read-set *footprint* (a node's closed neighbourhood), not the state *values*, so
every topology-preserving rule yields the same DAG. The identical recorder serves
step 4, where a rewiring rule mutates the neighbourhood between events and the
per-event snapshot captures that history.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Sequence, Set

import numpy as np
import networkx as nx

from async_engine import apply_event, EVENTS
from causal_sets import ordering_fraction, mm_dimension, midpoint_dimension
from dimension import fast_dimension_field, dimension_stats
from simulation import create_initial_graph


# --------------------------------------------------------------------------
# The event DAG
# --------------------------------------------------------------------------

class EventDAG:
    """
    A recorded causal DAG of update events.

    Events are indexed 0..n-1 in the order they fired, so an event's id exceeds
    every id it depends on: ascending id is always a valid topological order,
    and `parents[e]` contains only ids < e.
    """

    def __init__(self, n: int, parents: List[List[int]],
                 event_nodes: np.ndarray, times: np.ndarray):
        self.n = n
        self.parents = parents
        self.event_nodes = event_nodes
        self.times = times
        self._children: List[List[int]] | None = None

    @property
    def children(self) -> List[List[int]]:
        """Reverse adjacency, built lazily."""
        if self._children is None:
            ch: List[List[int]] = [[] for _ in range(self.n)]
            for e, ps in enumerate(self.parents):
                for p in ps:
                    ch[p].append(e)
            self._children = ch
        return self._children


def _build_parents(event_nodes: Sequence[int],
                   read_sets: Sequence[Set[int]]) -> List[List[int]]:
    """
    Pure core of the recorder: given the node that fired at each event and the
    set of nodes it read, return each event's causal parents.

    A parent of event `e` is, for every node `u` in its read set, the latest
    event at `u` strictly before `e`. The firing node reads itself, so an event
    is always a child of its own previous event -- each node's worldline is a
    timelike chain. Separated out so it can be checked on hand-built sequences.
    """
    last_event: Dict[int, int] = {}
    parents: List[List[int]] = []
    for e, (v, read) in enumerate(zip(event_nodes, read_sets)):
        ps = sorted(last_event[u] for u in read if u in last_event)
        parents.append(ps)
        last_event[int(v)] = e
    return parents


def record_event_dag(G: nx.Graph, rule: str, n_events: int, seed: int,
                     rate: float = 1.0) -> EventDAG:
    """
    Run `n_events` Poisson-clock events on `G` and record the causal DAG.

    The event ORDER is the Harris construction used by
    `async_engine.run_sequential`: each node holds an independent exponential
    clock, and the minimum clock fires next. At each event the *current* closed
    neighbourhood is snapshotted as the read set before the rule is applied, so
    a topology-changing rule's history is captured correctly.

    `rule` is applied for fidelity with the dynamic case, but on a static
    (topology-preserving) rule it does not affect the DAG. Missing node/edge
    attributes are initialised to the project defaults so any rule can run.
    """
    if rule not in EVENTS:
        raise ValueError(f"unsupported rule {rule!r}; have {sorted(EVENTS)}")

    G = G.copy()
    for node in G.nodes():
        G.nodes[node].setdefault('active', False)
        G.nodes[node].setdefault('state', 0)
    for u, v in G.edges():
        if 'weight' not in G[u][v]:
            G[u][v]['weight'] = 0.5

    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    clocks = rng.exponential(1.0 / rate, size=len(nodes))

    event_nodes = np.empty(n_events, dtype=np.int64)
    times = np.empty(n_events)
    parents: List[List[int]] = []
    last_event: Dict[int, int] = {}

    for event_id in range(n_events):
        k = int(np.argmin(clocks))
        v = nodes[k]
        read = {v}
        read.update(G.neighbors(v))
        parents.append(sorted(last_event[u] for u in read if u in last_event))

        event_nodes[event_id] = v
        times[event_id] = clocks[k]
        last_event[v] = event_id

        apply_event(G, v, rule, seed, event_id)
        clocks[k] += rng.exponential(1.0 / rate)

    return EventDAG(n_events, parents, event_nodes, times)


# --------------------------------------------------------------------------
# Reachability and intervals
# --------------------------------------------------------------------------

def reach(dag: EventDAG, src: int, forward: bool = True,
          max_depth: int | None = None) -> Set[int]:
    """
    Set of events causally reachable from `src` (inclusive of `src`).

    `forward=True` follows children (the causal future); `forward=False`
    follows parents (the causal past). `max_depth`, if given, bounds the BFS in
    DAG links (unused by the interval sampler, which confines by `past(q)`
    instead, but handy for depth-limited probes).
    """
    adj = dag.children if forward else dag.parents
    seen = {src}
    frontier = deque([(src, 0)])
    while frontier:
        node, depth = frontier.popleft()
        if max_depth is not None and depth >= max_depth:
            continue
        for nxt in adj[node]:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append((nxt, depth + 1))
    return seen


def alexandrov_interval(dag: EventDAG, p: int, q: int) -> np.ndarray:
    """
    The Alexandrov interval I(p, q) = future(p) ∩ past(q), inclusive of the
    endpoints, as a sorted int array. Empty (beyond possibly p, q) when q is
    not in the causal future of p.

    Built by growing `past(q)` first, then forward-BFS from `p` confined to it.
    Crucially there is NO depth cap. The async causal structure has no fixed
    light-cone speed: a quick succession of neighbour firings lets influence
    propagate many hops in near-zero time (a last-passage-percolation effect), so
    the longest chain between p and q runs several times the worldline separation
    (measured ~6-7x). A depth cap would truncate those long chains, drop pairs,
    and distort the ordering fraction. Confinement to `past(q)` keeps the cost at
    |I|, not |DAG|. (This chain-length excess over the worldline is exactly why
    the FPP DAG reads a dimension below d_H+1 -- see `_validate` and FINDINGS.)
    """
    past_q = reach(dag, q, forward=False)
    children = dag.children
    seen = {p} if p in past_q else set()
    frontier = deque(seen)
    while frontier:
        node = frontier.popleft()
        for c in children[node]:
            if c in past_q and c not in seen:
                seen.add(c)
                frontier.append(c)
    return np.array(sorted(seen), dtype=np.int64)


def interval_relation_matrix(dag: EventDAG, ids: np.ndarray) -> np.ndarray:
    """
    Transitive-closure relation matrix over the events in `ids`.

    `R[i, j]` is True iff `ids[j]` lies in the causal future of `ids[i]`.
    Alexandrov intervals are causally convex -- if a < c < b and a, b are in the
    interval then so is c -- so reachability within the induced sub-DAG equals
    reachability in the full DAG restricted to the interval, and the closure can
    be built from local links alone. `ids` must be ascending (topological).
    """
    m = len(ids)
    local = {int(e): i for i, e in enumerate(ids)}
    # direct child links that stay inside the interval
    child_local: List[List[int]] = [[] for _ in range(m)]
    for j, e in enumerate(ids):
        for p in dag.parents[int(e)]:
            pi = local.get(p)
            if pi is not None:
                child_local[pi].append(j)

    R = np.zeros((m, m), dtype=bool)
    # descending id (= descending local index) is reverse-topological
    for i in range(m - 1, -1, -1):
        for c in child_local[i]:
            R[i, c] = True
            R[i] |= R[c]
    return R


# --------------------------------------------------------------------------
# Interval sampling
# --------------------------------------------------------------------------

def _node_worldlines(dag: EventDAG) -> Dict[int, List[int]]:
    """Per node, its event ids in firing order (its timelike worldline)."""
    wl: Dict[int, List[int]] = {}
    for e in range(dag.n):
        wl.setdefault(int(dag.event_nodes[e]), []).append(e)
    return wl


def sample_intervals(dag: EventDAG, rng: np.random.Generator,
                     n_intervals: int, min_card: int, max_card: int,
                     height: int) -> List[np.ndarray]:
    """
    Sample REST-FRAME Alexandrov intervals of controlled cardinality.

    Endpoints `p`, `q` are two events at the *same node*, `height` worldline
    steps apart -- a purely timelike-separated pair, so the diamond
    I(p, q) = future(p) ∩ past(q) is spatially symmetric about that node's
    worldline. This matches how `causal_sets.py` calibrates MM (it sprinkles
    the interval between the origin and a purely timelike point). The frame
    matters because this causal set is NOT Lorentz-invariant: it has a
    preferred frame (the graph at rest under Poisson time), so a boosted
    (near-null) diamond has a different ordering fraction than a rest-frame one,
    and only the rest-frame value is comparable to the MM calibration.

    A rest-frame diamond scales as |I| ~ height^(d+1), so `height` is the
    interval-size knob, tuned against the cardinality band -- never against the
    dimension, keeping the instrument answer-agnostic (the step-1 lesson).
    """
    wl = _node_worldlines(dag)
    live = [a for a, es in wl.items() if len(es) > height + 1]
    intervals: List[np.ndarray] = []
    if not live:
        return intervals

    max_tries = n_intervals * 80
    for _ in range(max_tries):
        if len(intervals) >= n_intervals:
            break
        a = live[rng.integers(len(live))]
        es = wl[a]
        i = int(rng.integers(len(es) - height))
        p, q = es[i], es[i + height]
        ids = alexandrov_interval(dag, p, q)
        if min_card <= len(ids) <= max_card:
            intervals.append(ids)
    return intervals


# --------------------------------------------------------------------------
# Dimension estimation
# --------------------------------------------------------------------------

def causal_dimension(G: nx.Graph, rule: str, n_events: int, seed: int,
                     height: int, n_intervals: int = 6,
                     min_card: int = 200, max_card: int = 2000) -> Dict:
    """
    Record the event DAG on `G`, sample intervals, and estimate causal
    dimension by Myrheim-Meyer (primary) and midpoint scaling (cross-check).

    Each sampled interval is one causal set, the direct analogue of one
    flat-space sprinkle in `causal_sets.py`: MM inverts its ordering fraction,
    midpoint bisects its volume. Reported values are medians over intervals,
    with the fraction of intervals returning a finite MM dimension surfaced as
    an honesty signal rather than silently dropped.
    """
    dag = record_event_dag(G, rule, n_events, seed)
    rng = np.random.default_rng(seed + 1)
    intervals = sample_intervals(dag, rng, n_intervals, min_card, max_card,
                                 height)

    d_mm: List[float] = []
    d_mid: List[float] = []
    rs: List[float] = []
    cards: List[int] = []
    for ids in intervals:
        R = interval_relation_matrix(dag, ids)
        r = ordering_fraction(R)
        rs.append(r)
        cards.append(len(ids))
        dm = mm_dimension(r)
        if np.isfinite(dm):
            d_mm.append(dm)
        mid = midpoint_dimension(R, rng)
        if np.isfinite(mid['d_eff']):
            d_mid.append(mid['d_eff'])

    def _med(xs):
        return float(np.median(xs)) if xs else float('nan')

    def _std(xs):
        return float(np.std(xs)) if xs else float('nan')

    return {
        'd_mm': _med(d_mm), 'd_mm_std': _std(d_mm), 'n_mm_defined': len(d_mm),
        'd_midpoint': _med(d_mid), 'd_midpoint_std': _std(d_mid),
        'n_midpoint_defined': len(d_mid),
        'n_intervals': len(intervals),
        'defined_frac': (len(d_mm) / len(intervals)) if intervals else 0.0,
        'mean_ordering_fraction': _med(rs),
        'mean_cardinality': float(np.mean(cards)) if cards else float('nan'),
        'd_mm_all': d_mm,
    }


def hausdorff_dimension(G: nx.Graph, max_radius: int,
                        n_samples: int | None = None,
                        seed: int = 0) -> float:
    """
    Spatial (ball-growth) Hausdorff dimension of `G`, the calibration target:
    the causal dimension should be this + 1. Measured at `max_radius` so the
    scale matches the *local* extent the causal intervals probe -- not the
    global extent dimension, which differs on `grown`.

    Builds the adjacency with `weight=None`: the project sets edge weight 0.5,
    and the sparse default would silently use it (AGENTS.md footgun).
    """
    nodes = sorted(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight=None,
                                 dtype=np.float32, format='csr')
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    if n_samples is None:
        n_samples = min(n, 400)
    idx = (np.arange(n) if n_samples >= n
           else rng.choice(n, n_samples, replace=False))
    field = fast_dimension_field(A, max_radius=max_radius, sample_indices=idx)
    stats = dimension_stats(field, n)
    return float(stats['d_eff_median'])


# --------------------------------------------------------------------------
# Analytic fixed-speed-cone control
# --------------------------------------------------------------------------

def cone_dimension(dag: EventDAG, coords: np.ndarray, c: float,
                   rng: np.random.Generator, n_intervals: int = 6,
                   height: int = 12, min_card: int = 300,
                   max_card: int = 2500) -> Dict:
    """
    Positive control: dimension of the SAME event cloud (same nodes, same Poisson
    times, same rest-frame diamonds) but under an ARTIFICIAL fixed-speed light
    cone `dist(u,w) <= c * (t_w - t_u)` instead of the recorded async causal DAG.

    `coords` maps event index -> spatial coordinates (Manhattan metric). If this
    clean, speed-limited cone recovers d_H + 1 while the async DAG does not, the
    deficit is isolated to the async causal structure, not the estimators or the
    sampler. This is the audit's controlled experiment, wired into the gate.
    """
    wl = _node_worldlines(dag)
    live = [a for a, es in wl.items() if len(es) > height + 1]
    t = dag.times
    d_mm: List[float] = []
    d_mid: List[float] = []
    tries = 0
    while len(d_mm) < n_intervals and tries < n_intervals * 80:
        tries += 1
        a = live[rng.integers(len(live))]
        es = wl[a]
        i = int(rng.integers(len(es) - height))
        p, q = es[i], es[i + height]
        tp, tq = t[p], t[q]
        # candidate events in the time slab, then inside the double cone from a
        cand = np.nonzero((t >= tp) & (t <= tq))[0]
        cp = coords[dag.event_nodes[cand]]        # event -> its node's coords
        dist_a = np.abs(cp[:, 0] - coords[a][0]) + np.abs(cp[:, 1] - coords[a][1])
        reach_lim = c * np.minimum(t[cand] - tp, tq - t[cand])
        sel = cand[dist_a <= reach_lim]
        if not (min_card <= len(sel) <= max_card):
            continue
        cs = coords[dag.event_nodes[sel]]
        dt = t[sel][None, :] - t[sel][:, None]
        dx = (np.abs(cs[:, 0][None, :] - cs[:, 0][:, None])
              + np.abs(cs[:, 1][None, :] - cs[:, 1][:, None]))
        R = (dt > 0) & (dx <= c * dt)
        dm = mm_dimension(ordering_fraction(R))
        md = midpoint_dimension(R, rng)['d_eff']
        if np.isfinite(dm):
            d_mm.append(dm)
        if np.isfinite(md):
            d_mid.append(md)
    return {'d_mm': float(np.median(d_mm)) if d_mm else float('nan'),
            'd_midpoint': float(np.median(d_mid)) if d_mid else float('nan'),
            'n': len(d_mm)}


# --------------------------------------------------------------------------
# Validation gate
# --------------------------------------------------------------------------

def _prep(G: nx.Graph) -> nx.Graph:
    G = nx.convert_node_labels_to_integers(G)
    for x in G.nodes():
        G.nodes[x]['active'] = False
        G.nodes[x]['state'] = 0
    for u, v in G.edges():
        G[u][v]['weight'] = 0.5
    return G


def _validate() -> bool:
    """
    Gate for the static-graph causal calibration (step 3 of LORENTZIAN_SPIKE.md).

    Unusually, this gate documents a NEGATIVE scientific result: the pre-committed
    null `d_causal = d_H + 1` is REJECTED for the async event DAG. So it asserts
    the three things that are actually true and stable:

      (A) the DAG machinery is correct (transitive closure matches brute-force
          reachability; the DAG is rule-independent on a static graph);
      (B) POSITIVE CONTROL -- a clean fixed-speed light cone on the same substrate
          recovers d_H + 1, proving the sampler + estimators are sound;
      (C) the FINDING -- the async FPP DAG reads well below d_H + 1 AND its two
          estimators disagree (the non-manifold-like signature), while the 1D case
          (where the +1 is cleanly present) is recovered.

    Returns True when all three hold: the instrument code is correct and the
    negative is faithfully reproduced. See FINDINGS.md for the write-up.
    """
    ok = True
    seed = 0

    print("=" * 70)
    print("STEP 3: static-graph causal calibration -- known-answer gate")
    print("=" * 70)

    # (A1) machinery: hand-verified parent recording -------------------------
    parents = _build_parents([0, 1, 0, 2],
                             [{0, 1}, {0, 1, 2}, {0, 1}, {1, 2}])
    parents = [sorted(p) for p in parents]
    a1 = parents == [[], [0], [0, 1], [1]]
    print(f"[A1] parent recording (hand case)            {'OK' if a1 else 'FAIL'}")
    ok &= a1

    # (A2) rule-independence of the static-graph DAG -------------------------
    Gsmall = _prep(create_initial_graph(400, 'grown', k=6, seed=2))
    da = record_event_dag(Gsmall, 'activation', 6000, seed)
    dm = record_event_dag(Gsmall, 'majority', 6000, seed)
    a2 = da.parents == dm.parents and np.array_equal(da.event_nodes,
                                                     dm.event_nodes)
    print(f"[A2] DAG identical across rules              {'OK' if a2 else 'FAIL'}")
    ok &= a2

    # build one 2D-lattice DAG; reused for the machinery, control, and finding
    side = 40
    G = _prep(nx.grid_2d_graph(side, side))
    coords = np.array([[i // side, i % side] for i in range(side * side)])
    dag = record_event_dag(G, 'activation', 110000, seed)
    rng = np.random.default_rng(seed + 1)

    # (A3) transitive closure vs independent brute-force reachability --------
    wl = _node_worldlines(dag)
    live = [a for a, es in wl.items() if len(es) > 8]
    small = None
    for _ in range(200):
        a = live[rng.integers(len(live))]
        es = wl[a]
        i = int(rng.integers(len(es) - 6))
        ids = alexandrov_interval(dag, es[i], es[i + 6])
        if 120 <= len(ids) <= 500:
            small = ids
            break
    R = interval_relation_matrix(dag, small)
    idset = set(int(x) for x in small)
    localid = {int(e): k for k, e in enumerate(small)}
    Rb = np.zeros_like(R)
    for k, e in enumerate(small):
        fut = reach(dag, int(e), forward=True) & idset
        for f in fut:
            if f != int(e):
                Rb[k, localid[f]] = True
    a3 = np.array_equal(R, Rb)
    print(f"[A3] transitive closure vs brute force       "
          f"{'OK' if a3 else 'FAIL'}  ({int(R.sum())} relations, card {len(small)})")
    ok &= a3

    # (B) POSITIVE CONTROL: fixed-speed cone on the same substrate -----------
    dHl = 2.0  # a 2D lattice is exactly 2-dimensional by construction
    cone = cone_dimension(dag, coords, c=1.5, rng=np.random.default_rng(seed + 2),
                          height=12)
    b = 2.55 <= cone['d_midpoint'] <= 3.45
    print(f"[B ] control: fixed-speed cone -> d_mid={cone['d_midpoint']:.2f}  "
          f"(target {dHl+1:.0f})   {'OK' if b else 'FAIL'}")
    ok &= b

    # (C) THE FINDING: async FPP DAG on the same substrate -------------------
    fpp = causal_dimension(G, 'activation', 110000, seed, height=12,
                           n_intervals=6, min_card=400, max_card=8000)
    below = fpp['d_mm'] < 2.8 and fpp['d_midpoint'] < 2.5
    disagree = abs(fpp['d_mm'] - fpp['d_midpoint']) > 0.3
    print(f"[C1] async FPP DAG -> d_mm={fpp['d_mm']:.2f}, "
          f"d_mid={fpp['d_midpoint']:.2f}  (target {dHl+1:.0f})")
    print(f"     below d_H+1: {below},  estimators disagree (non-manifold): "
          f"{disagree}")
    ok &= (below and disagree)

    G1 = _prep(nx.path_graph(160))
    one = causal_dimension(G1, 'activation', 9000, seed, height=24,
                           n_intervals=6, min_card=200, max_card=3000)
    c1d = 1.6 <= one['d_mm'] <= 2.2
    print(f"[C2] 1D path (clean +1) -> d_mm={one['d_mm']:.2f}  (target 2)   "
          f"{'OK' if c1d else 'FAIL'}")
    ok &= c1d

    print("-" * 70)
    print("VERDICT: pre-committed null  d_causal = d_H + 1  is REJECTED for the")
    print("async event DAG. Clean fixed-speed cone recovers it (control passes),")
    print("so the deficit is the async FPP causal structure, not the instrument.")
    print("This is the FPP-shape negative pre-committed in the design spec (#4).")
    print("=" * 70)
    print(f"{'PASS' if ok else 'FAIL'}: machinery correct, control recovers "
          f"d_H+1, negative reproduced.")
    return ok


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--validate', action='store_true',
                    help='run the known-answer calibration gate')
    args = ap.parse_args()
    if args.validate:
        raise SystemExit(0 if _validate() else 1)
    ap.print_help()


if __name__ == '__main__':
    main()
