"""
Asynchronous event-driven update engine, and the proof that batching it is safe.

This is **step 2** of the staged plan in LORENTZIAN_SPIKE.md. The existing
backends sweep every node synchronously from a global clock; the Lorentzian
upgrade needs updates to happen as *local events*, so that a causal DAG of
those events exists at all. This module provides that, in two implementations:

  * `run_sequential` -- one event at a time, in Poisson-clock order. Obviously
    correct, and the reference the fast path is checked against.
  * `run_batched`    -- events grouped into conflict-free batches so they can
    be applied together, which is what preserves vectorisation.

**Kill criterion (from the spike): if batched and sequential disagree, stop.**

CONFLICT RADIUS IS RULE-DEPENDENT -- and the spike got this wrong.
------------------------------------------------------------------
Two events commute when neither writes what the other reads. The spike verified
independent sets in G (radius 1), which is correct ONLY for pure state rules:

    activation, majority   read N(v) states, write v's state       -> radius 1
    prune                  reads |N(u) & N(v)|, i.e. a NEIGHBOUR'S
                           neighbourhood                            -> radius 2
    triadic, ricci,        rewire toward friends-of-friends         -> radius 2
      geometrize
    rewire                 picks a target uniformly from ALL nodes  -> non-local

So every topology rule in this project -- including `prune`, which the
censorship checkpoint depends on -- needs an independent set in G^2 (nodes
pairwise at least 3 hops apart), not in G. Using radius 1 for those rules
produces silently order-dependent results; `--validate` demonstrates exactly
that as a negative control, so the test is known to have teeth.

`rewire` is deliberately not supported here. Its target is drawn uniformly over
the whole graph, so no graph-distance criterion can make it conflict-free; it
would need collision detection and deferral instead.

Determinism
-----------
Each event draws from its own generator seeded by (run seed, event id), never
from a shared stream. Order of application therefore cannot change what an
event decides -- which is what makes exact batched-vs-sequential comparison
meaningful rather than a coincidence of RNG bookkeeping.

Usage:
    python async_engine.py --validate
    python async_engine.py --validate --nodes 400 --seeds 16
"""

import argparse
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx
import scipy.sparse as sp
from tqdm import tqdm

from simulation import create_initial_graph


# Conflict radius r means: no two events in a batch may be within r hops.
# Derived from read/write footprints, where "reads X" means "reads the
# adjacency or state of X", and two events conflict when one WRITES what the
# other READS:
#
#   rule        reads              writes                 disjoint when
#   activation  {v} u N(v) state   v's state              dist >= 2  -> r=1
#   majority    N(v) state         v's state              dist >= 2  -> r=1
#   prune       {v} u N(v) adj     {v} u N(v) adj         dist >= 3  -> r=2
#   triadic     {v} u N(v) adj     {v} u N(v) u {w}, and  dist >= 4  -> r=3
#                                  w is a FRIEND-OF-FRIEND, i.e. distance 2
#
# The spike assumed radius 1 throughout, which is right only for state rules.
# `triadic` is the expensive case and it is expensive for a specific reason:
# it is the only rule that WRITES at distance 2, since triadic closure attaches
# v to a node two hops away. The same argument applies to `ricci` and
# `geometrize`, which also rewire toward friends-of-friends.
CONFLICT_RADIUS: Dict[str, int] = {
    'activation': 1,
    'majority': 1,
    'prune': 2,
    'triadic': 3,
}


# --------------------------------------------------------------------------
# Node events -- the local action a single update performs
# --------------------------------------------------------------------------
#
# Each takes (G, v, rng) and mutates G in place. Edge-touching rules use
# OWNERSHIP -- edge (a, b) belongs to min(a, b) -- so that a full sweep gives
# each edge exactly one chance, matching the per-edge rate of the synchronous
# rule rather than doubling it.

def _event_activation(G: nx.Graph, v: int, rng: np.random.Generator,
                      spread_prob: float = 0.3,
                      decay_prob: float = 0.1) -> None:
    if G.nodes[v].get('active', False):
        G.nodes[v]['active'] = bool(rng.random() > decay_prob)
        return
    k = sum(1 for u in G[v] if G.nodes[u].get('active', False))
    if k == 0:
        G.nodes[v]['active'] = False
    else:
        G.nodes[v]['active'] = bool(rng.random() < 1.0 - (1.0 - spread_prob) ** k)


def _event_majority(G: nx.Graph, v: int, rng: np.random.Generator,
                    num_states: int = 2, noise: float = 0.01) -> None:
    neighbours = list(G[v])
    if not neighbours:
        return
    if rng.random() < noise:
        G.nodes[v]['state'] = int(rng.integers(num_states))
        return
    counts = [0] * num_states
    for u in neighbours:
        counts[G.nodes[u].get('state', 0)] += 1
    G.nodes[v]['state'] = int(np.argmax(counts))


def _event_prune(G: nx.Graph, v: int, rng: np.random.Generator,
                 prune_prob: float = 0.05, min_overlap: int = 1,
                 min_degree: int = 2) -> None:
    for u in [w for w in G[v] if v < w]:
        if rng.random() >= prune_prob:
            continue
        if G.degree(v) <= min_degree or G.degree(u) <= min_degree:
            continue
        if len(set(G[u]) & set(G[v])) < min_overlap:
            G.remove_edge(v, u)


def _event_triadic(G: nx.Graph, v: int, rng: np.random.Generator,
                   rewire_prob: float = 0.05, min_degree: int = 2) -> None:
    for p in [w for w in G[v] if v < w]:
        if rng.random() >= rewire_prob:
            continue
        if G.degree(p) <= min_degree:
            continue
        nbrs = list(G[v])
        if not nbrs:
            continue
        x = nbrs[int(rng.integers(len(nbrs)))]
        xn = list(G[x])
        if not xn:
            continue
        w = xn[int(rng.integers(len(xn)))]
        if w == v or G.has_edge(v, w):
            continue
        weight = G[v][p].get('weight', 0.5)
        G.remove_edge(v, p)
        G.add_edge(v, w, weight=weight)


EVENTS: Dict[str, Callable] = {
    'activation': _event_activation,
    'majority': _event_majority,
    'prune': _event_prune,
    'triadic': _event_triadic,
}


def _event_rng(seed: int, event_id: int) -> np.random.Generator:
    """Per-event generator, so application order cannot change any decision."""
    return np.random.default_rng((seed, event_id))


def apply_event(G: nx.Graph, v: int, rule: str, seed: int, event_id: int,
                params: Optional[Dict] = None) -> None:
    EVENTS[rule](G, v, _event_rng(seed, event_id), **(params or {}))


# --------------------------------------------------------------------------
# Conflict-free batch selection
# --------------------------------------------------------------------------

def conflict_matrix(G: nx.Graph, radius: int,
                    nodelist: Optional[Sequence] = None) -> sp.csr_array:
    """
    Sparse pattern of "these two events may not run together".

    radius 1 -> adjacency; radius 2 -> adjacency plus paths of length 2.
    NOTE weight=None: edges here carry weight 0.5, and the default would build
    a WEIGHTED matrix, silently halving every neighbour max downstream.
    """
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=None,
                                 format='csr', dtype=np.float64)
    if radius < 1:
        raise ValueError(f"unsupported conflict radius {radius}")
    C, power = A, A
    for _ in range(radius - 1):
        power = power @ A
        C = C + power
    C = sp.csr_array((C > 0).astype(np.float64))
    C.setdiag(0)
    C.eliminate_zeros()
    return C


def select_batch(C: sp.csr_array, rng: np.random.Generator,
                 thin: bool = True) -> np.ndarray:
    """
    A conflict-free batch: give every node a random priority and keep those
    that beat all conflicting nodes.

    Independence is exact, not heuristic -- if i and j conflict they cannot
    both win, since that would need prio[i] > prio[j] and prio[j] > prio[i].

    DEGREE THINNING (`thin`) IS NOT OPTIONAL FOR FAITHFUL DYNAMICS.
    A node wins the priority contest with probability exactly 1/(c(v)+1),
    where c(v) is its number of conflicting nodes -- all orderings of a node
    and its conflict neighbourhood being equally likely. So naive batching
    samples high-degree nodes LESS often, silently replacing uniform Poisson
    clocks with degree-dependent ones. On `grown` (degrees 2..6) that shifted
    steady-state activation by 11% at z=5.9 against the sequential reference.

    The fix: accept a winner with probability (c(v)+1)/(c_max+1), making the
    net rate 1/(c_max+1) for every node regardless of degree. Independence is
    preserved because thinning only removes nodes. The cost is a smaller
    batch -- uniformity is paid for in throughput.
    """
    n = C.shape[0]
    prio = rng.random(n)
    nbr_max = np.asarray(C.multiply(prio[None, :]).max(axis=1).todense()).ravel()
    selected = prio > nbr_max
    if thin:
        conflict_degree = np.diff(C.indptr)
        c_max = int(conflict_degree.max()) if n else 0
        accept = rng.random(n) < (conflict_degree + 1.0) / (c_max + 1.0)
        selected &= accept
    return np.nonzero(selected)[0]


# --------------------------------------------------------------------------
# The two execution paths
# --------------------------------------------------------------------------

def run_sequential(G: nx.Graph, rule: str, n_events: int, seed: int,
                   rate: float = 1.0) -> Tuple[nx.Graph, np.ndarray]:
    """
    Reference path: one event at a time, in Poisson-clock order.

    Each node carries an independent exponential clock of the given rate; the
    resulting event ORDER is a uniformly random node sequence (the Harris
    construction). Times are tracked explicitly because the causal DAG will
    need them later, and because non-uniform rates -- a spatially varying
    lapse -- are the natural next generalisation.
    """
    G = G.copy()
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    clocks = rng.exponential(1.0 / rate, size=len(nodes))
    times = np.empty(n_events)

    for event_id in range(n_events):
        k = int(np.argmin(clocks))
        times[event_id] = clocks[k]
        apply_event(G, nodes[k], rule, seed, event_id)
        clocks[k] += rng.exponential(1.0 / rate)
    return G, times


def run_sequential_multi(G: nx.Graph, rules: Sequence[str],
                         rates: Sequence[float],
                         n_events: Optional[int] = None,
                         max_time: Optional[float] = None,
                         seed: int = 0,
                         params: Optional[Sequence[Optional[Dict]]] = None,
                         on_event: Optional[Callable[[int, int, int, float, nx.Graph], None]] = None
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


def run_batched(G: nx.Graph, rule: str, n_events: int, seed: int
                ) -> Tuple[nx.Graph, List[int]]:
    """
    Fast path: repeatedly select a conflict-free batch and apply it whole.

    Every event in a batch commutes with every other, so the batch is
    equivalent to *some* valid sequential ordering. Returns the graph and the
    per-round batch sizes.
    """
    G = G.copy()
    rng = np.random.default_rng(seed)
    radius = CONFLICT_RADIUS[rule]
    sizes: List[int] = []
    event_id = 0

    while event_id < n_events:
        nodes = list(G.nodes())
        C = conflict_matrix(G, radius, nodelist=nodes)
        batch = select_batch(C, rng)
        if len(batch) == 0:
            break
        sizes.append(len(batch))
        for idx in batch:
            if event_id >= n_events:
                break
            apply_event(G, nodes[int(idx)], rule, seed, event_id)
            event_id += 1
    return G, sizes


# --------------------------------------------------------------------------
# Graph fingerprint, for exact comparison
# --------------------------------------------------------------------------

def fingerprint(G: nx.Graph) -> Tuple:
    """Order-independent exact summary: node states and weighted edge set."""
    states = tuple(sorted(
        (n, bool(G.nodes[n].get('active', False)), int(G.nodes[n].get('state', 0)))
        for n in G.nodes()))
    edges = tuple(sorted(
        (min(u, v), max(u, v), round(float(d.get('weight', 0.5)), 12))
        for u, v, d in G.edges(data=True)))
    return states, edges


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _order_dependence_found(rule: str, radius: int, n_nodes: int,
                            params: Dict, topology: str = 'grown',
                            n_trials: int = 8, n_orders: int = 6) -> bool:
    """
    Apply a batch in several different orders and report whether ANY ordering
    changed the result. A correctly-chosen conflict radius must give False;
    a too-small radius must eventually give True, or the test proves nothing.

    Searches several independent batches, because a single batch may happen to
    contain no interacting pair even when the radius is wrong.
    """
    for trial in range(n_trials):
        G = create_initial_graph(n_nodes, topology=topology, k=6, seed=trial)
        if rule == 'majority':
            for i, node in enumerate(G.nodes()):
                G.nodes[node]['state'] = i % 2
        if rule == 'activation':
            for i, node in enumerate(G.nodes()):
                G.nodes[node]['active'] = (i % 3 == 0)

        nodes = list(G.nodes())
        C = conflict_matrix(G, radius, nodelist=nodes)
        batch = select_batch(C, np.random.default_rng(trial), thin=False)
        if len(batch) < 2:
            continue

        prints = set()
        for order_i in range(n_orders):
            order = np.arange(len(batch))
            np.random.default_rng(1000 + order_i).shuffle(order)
            H = G.copy()
            for pos in order:
                # The event id is tied to the node's slot in the batch, never
                # to its position in the application order, so reordering
                # cannot change any event's random draws. Any difference that
                # survives is a genuine read/write conflict.
                apply_event(H, nodes[int(batch[pos])], rule, trial,
                            int(pos), params)
            prints.add(fingerprint(H))
        if len(prints) > 1:
            return True
    return False


def prune_radius1_counterexample() -> Tuple[bool, str]:
    """
    Constructed proof that radius 1 is insufficient for `prune`.

    Random search does not find this reliably, because prune's only distance-2
    coupling runs through the degree floor and that conjunction is rare: two
    selected nodes must both own a zero-overlap edge into the same neighbour,
    both fire, and find that neighbour sitting exactly on the floor. Rare is
    not safe, so here is the minimal configuration where it is forced.

        0 -- 5 -- 1        deg(5) = 3, min_degree = 2
        |         |        0 and 1 are at distance 2, so a radius-1
       2-3       7-8       independent set may legally contain both.

    Edges (0,5) and (1,5) both have zero overlap, so both are prunable; the
    2-3 and 7-8 triangles exist only to make 0's and 1's other edges
    non-prunable, isolating the effect. Whichever of the two events runs first
    removes its edge and drops deg(5) to the floor, which then blocks the
    other. Both orders remove exactly one edge -- but a DIFFERENT one, so the
    resulting graphs differ.
    """
    G = nx.Graph()
    G.add_edges_from([(0, 5), (1, 5), (5, 6),
                      (0, 2), (0, 3), (2, 3),
                      (1, 7), (1, 8), (7, 8)], weight=0.5)
    params = {'prune_prob': 1.0, 'min_overlap': 1, 'min_degree': 2}

    prints = {}
    for order in ((0, 1), (1, 0)):
        H = G.copy()
        for pos, v in enumerate(order):
            # Event id keyed to the NODE, so the draws cannot depend on order.
            apply_event(H, v, 'prune', seed=0, event_id=v, params=params)
        prints[order] = fingerprint(H)

    differ = prints[(0, 1)] != prints[(1, 0)]
    edges_a = sorted(e[:2] for e in prints[(0, 1)][1])
    edges_b = sorted(e[:2] for e in prints[(1, 0)][1])
    detail = (f"0-then-1 keeps {sorted(set(edges_b) - set(edges_a))}, "
              f"1-then-0 keeps {sorted(set(edges_a) - set(edges_b))}")
    return differ, detail if differ else "orders agree -- no conflict"


def _observables(G: nx.Graph) -> Dict[str, float]:
    n = G.number_of_nodes()
    return {
        'n_edges': float(G.number_of_edges()),
        'n_active': float(sum(1 for v in G if G.nodes[v].get('active', False))),
        'mean_degree': 2.0 * G.number_of_edges() / max(n, 1),
        'clustering': float(nx.average_clustering(G)),
    }


def _test_distribution(rule: str, n_nodes: int, n_events: int, seeds: int
                       ) -> Tuple[bool, str]:
    """
    Batched and sequential runs must agree in DISTRIBUTION. They are not
    expected to agree trajectory-by-trajectory: batching imposes its own
    ordering, which is a valid asynchronous ordering but not the same sample
    as the Poisson one.
    """
    seq, bat = [], []
    for s in range(seeds):
        G = create_initial_graph(n_nodes, topology='grown', k=6, seed=s)
        if rule == 'majority':
            for i, n in enumerate(G.nodes()):
                G.nodes[n]['state'] = i % 2
        if rule == 'activation':
            for i, n in enumerate(G.nodes()):
                G.nodes[n]['active'] = (i % 5 == 0)
        gs, _ = run_sequential(G, rule, n_events, seed=1000 + s)
        gb, _ = run_batched(G, rule, n_events, seed=2000 + s)
        seq.append(_observables(gs))
        bat.append(_observables(gb))

    msgs, ok = [], True
    for key in seq[0]:
        a = np.array([r[key] for r in seq])
        b = np.array([r[key] for r in bat])
        if a.std() == 0 and b.std() == 0:
            continue
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        if se == 0:
            continue
        z = abs(a.mean() - b.mean()) / se
        if z > 3.0:
            ok = False
            msgs.append(f"{key}: seq {a.mean():.2f} vs bat {b.mean():.2f} "
                        f"(z={z:.1f})")
    return ok, '; '.join(msgs) if msgs else 'all observables agree'


def _validate(n_nodes: int = 400, n_events: int = 1200,
              seeds: int = 12) -> bool:
    ok = True

    # Aggressive rule parameters: at the production rates most events are
    # no-ops, so conflicts almost never get a chance to show. Turning the
    # rates up is what gives both the test and its control teeth.
    # `prune` needs min_degree high enough that the degree FLOOR actually
    # binds. Its only distance-2 coupling runs through that floor: a removal
    # at v' lowers deg(m) for a shared neighbour m, which v reads. With
    # min_degree=1 the floor never triggers on a mean-degree-4 graph, the
    # coupling never fires, and the negative control silently passes for the
    # wrong reason.
    hot = {'activation': {}, 'majority': {'noise': 0.0},
           'prune': {'prune_prob': 0.9, 'min_degree': 3},
           'triadic': {'rewire_prob': 0.9, 'min_degree': 1}}

    # `prune` must be exercised on `small_world`, not `grown`. `grown` is built
    # from triangle closures, so essentially every edge has overlap >= 1 and
    # prune removes nothing (the inertness reported in FINDINGS). A rule that
    # never fires can neither conflict nor prove anything, so testing it there
    # would pass vacuously. small_world's random shortcuts are what it bites.
    topo = {'activation': 'grown', 'majority': 'grown',
            'prune': 'small_world', 'triadic': 'grown'}

    print("Test A -- batch commutation (a conflict-free batch must be "
          "order-independent)")
    for rule in EVENTS:
        radius = CONFLICT_RADIUS[rule]
        bad = _order_dependence_found(rule, radius, n_nodes, hot[rule],
                                      topo[rule])
        ok &= not bad
        print(f"  {rule:<12s} radius {radius} on {topo[rule]:<12s}: "
              f"{'ORDER-DEPENDENT FAIL' if bad else 'order-independent OK'}")

    print("\nTest A' -- negative control (one radius BELOW the derived value "
          "must fail,\n           otherwise Test A proves nothing)")
    broke = _order_dependence_found('triadic', 2, n_nodes, hot['triadic'],
                                    topo['triadic'])
    ok &= broke
    print(f"  {'triadic':<12s} radius 2 on {topo['triadic']:<12s}: "
          + ('order-dependent as expected OK' if broke
             else 'no conflict detected -- TEST HAS NO TEETH'))

    # prune needs a CONSTRUCTED counterexample: random search does not find its
    # rare degree-floor coupling, and "rare" is not "safe".
    broke, detail = prune_radius1_counterexample()
    ok &= broke
    print(f"  {'prune':<12s} radius 1 constructed: "
          + ('order-dependent as expected OK' if broke
             else 'NO CONFLICT -- radius 2 may be unnecessary'))
    print(f"  {'':<12s}   {detail}")

    print(f"\nTest B -- batched vs sequential distributions "
          f"(N={n_nodes}, {n_events} events, {seeds} seeds)")
    for rule in EVENTS:
        good, msg = _test_distribution(rule, n_nodes, n_events, seeds)
        ok &= good
        print(f"  {rule:<12s}: {'OK' if good else 'FAIL'} -- {msg}")

    print("\nTest C -- batch fraction by conflict radius (the spike's cost "
          "model)")
    print("  (thinned batches -- the rate-correcting acceptance step is "
          "included, since\n   without it the dynamics are wrong)")
    for n in (2000, 10000):
        G = create_initial_graph(n, topology='grown', k=6, seed=0)
        rng = np.random.default_rng(0)
        cells = []
        for radius in (1, 2, 3):
            C = conflict_matrix(G, radius)
            frac = float(np.mean([len(select_batch(C, rng)) / n
                                  for _ in range(5)]))
            cells.append(f"r{radius}: {frac:.4f} ({1 / frac:5.1f} rounds)")
        print(f"  N={n:6d}  " + "   ".join(cells))
    print("  Rules by radius: activation/majority r1, prune r2, "
          "triadic/ricci/geometrize r3.")

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

    print(f"\n{'PASS' if ok else 'FAIL'}: asynchronous engine "
          f"(kill criterion: batched must match sequential)")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Asynchronous event-driven update engine.")
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--nodes', type=int, default=400)
    parser.add_argument('--events', type=int, default=1200)
    parser.add_argument('--seeds', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.validate:
        raise SystemExit(0 if _validate(args.nodes, args.events,
                                        args.seeds) else 1)
    parser.print_help()


if __name__ == '__main__':
    main()
