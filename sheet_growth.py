"""
`sheet`: frontier growth that closes wheels, with coordination-dependent rates.

Motivation (FINDINGS.md, audit of 2026-09-26): the `grown` generator has no
dimension. Every step hangs a new triangle on an existing edge, nothing makes
two growing arms meet, so the frontier stays proportional to the volume and
ball growth is exponential. A d-dimensional object needs a frontier that grows
like N^((d-1)/d). This module tests whether strictly local rules can do that.

Three local ingredients, none of which uses a coordinate:

  manifold   attach only at BOUNDARY edges -- edges in exactly one triangle
             (one common neighbour). `grown` also attaches at interior edges,
             which puts three triangles on an edge and branches the surface.
  closure    a boundary vertex x whose degree has reached `c` joins the two
             ends a, b of its link path, completing its wheel; x becomes
             interior. Reads x's neighbours and their adjacency, the same
             radius as `triadic`. `c` sets the curvature: 6 is flat, 7 is
             negatively curved, 5 closes up.
  tension    events fire on independent Poisson clocks whose rate depends only
             on the degrees of the edge's own endpoints,
                 attach at (u, v):  exp(beta * (deg u + deg v))
                 close at x:        exp(beta * 2c)
             so nearly-complete neighbourhoods fill in first. beta = 0 is
             uniform growth; large beta is compact growth.

Pre-registration (confirmatory run):
docs/superpowers/specs/2026-09-27-sheet-growth-preregistration.md

Usage:
    python sheet_growth.py --validate
    python sheet_growth.py --betas 0 1 2 3 --nodes 512000 --seeds 3 --jobs 12
"""

import argparse
import csv
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm

Adj = List[Set[int]]


def _is_boundary(adj: Adj, u: int, v: int) -> bool:
    return len(adj[u] & adj[v]) == 1


def _closable(adj: Adj, x: int, c: int) -> Optional[Tuple[int, int]]:
    """The link ends (a, b) of x if its wheel can be closed, else None."""
    ends = [y for y in adj[x] if _is_boundary(adj, x, y)]
    if len(ends) != 2:
        return None
    a, b = ends
    if b in adj[a] or (adj[a] & adj[b]) != {x}:
        return None                       # would pinch the surface
    if len(adj[a]) >= c or len(adj[b]) >= c:
        return None                       # keeps every degree <= c
    return a, b


def grow_sheet(n_nodes: int, c: int = 6, beta: float = 2.0,
               seed: int | None = None,
               checkpoints: Sequence[int] = (),
               on_checkpoint: Optional[Callable[[int, Adj], None]] = None
               ) -> Adj:
    """
    Grow a sheet from a seed triangle; returns adjacency as a list of sets.

    Rejection-free: boundary edges are bucketed by rate class (deg u + deg v,
    or the closure class), a class is drawn with probability proportional to
    its rate times its population, then an entry uniformly within it. Entries
    are invalidated lazily by a per-edge token, which keeps the draw exact:
    conditional on hitting a live entry, every live event is chosen in
    proportion to its rate.

    Growth can stall (no live event); the returned graph is then smaller than
    `n_nodes`, and the caller must check.
    """
    rng = np.random.default_rng(seed)
    adj: Adj = [{1, 2}, {0, 2}, {0, 1}]
    kmax = 2 * c
    weight = np.exp(beta * (np.arange(kmax + 1) - kmax))
    buckets: List[List[Tuple[Tuple[int, int], int]]] = \
        [[] for _ in range(kmax + 1)]
    token: Dict[Tuple[int, int], int] = {}

    def push(u: int, v: int) -> None:
        e = (u, v) if u < v else (v, u)
        t = token.get(e, 0) + 1
        token[e] = t
        if not _is_boundary(adj, u, v):
            return
        du, dv = len(adj[u]), len(adj[v])
        if du >= c or dv >= c:
            if any(len(adj[x]) >= c and _closable(adj, x, c) for x in e):
                buckets[kmax].append((e, t))
        else:
            buckets[du + dv].append((e, t))

    def refresh(x: int) -> None:
        # x's degree changed: re-rate its edges, and its neighbours' boundary
        # edges, whose closure eligibility reads x's degree and adjacency.
        for y in list(adj[x]):
            push(x, y)
            for z in adj[y]:
                if z != x and _is_boundary(adj, y, z):
                    push(y, z)

    for e in ((0, 1), (0, 2), (1, 2)):
        push(*e)
    marks = sorted(m for m in checkpoints if m <= n_nodes)
    sizes = np.zeros(kmax + 1)
    while len(adj) < n_nodes:
        for k in range(kmax + 1):
            sizes[k] = len(buckets[k])
        tot = sizes * weight
        s = tot.sum()
        if s <= 0:
            break                         # stalled
        k = int(rng.choice(kmax + 1, p=tot / s))
        b = buckets[k]
        i = int(rng.integers(len(b)))
        e, t = b[i]
        b[i] = b[-1]
        b.pop()
        if token.get(e) != t:
            continue                      # stale entry
        u, v = e
        if k == kmax:
            cand = [x for x in e
                    if len(adj[x]) >= c and _closable(adj, x, c)]
            if not cand:
                continue
            x = cand[int(rng.integers(len(cand)))]
            a, bb = _closable(adj, x, c)
            adj[a].add(bb)
            adj[bb].add(a)
            refresh(a)
            refresh(bb)
        else:
            w = len(adj)
            adj.append({u, v})
            adj[u].add(w)
            adj[v].add(w)
            refresh(u)
            refresh(v)
            if marks and len(adj) == marks[0]:
                marks.pop(0)
                if on_checkpoint is not None:
                    on_checkpoint(len(adj), adj)
    return adj


def grow_sheet_reference(n_nodes: int, c: int, beta: float,
                         seed: int | None = None) -> Adj:
    """
    Obviously-correct reference: recompute every live event's rate at every
    step and draw from the full list. O(E) per event; small graphs only.
    """
    rng = np.random.default_rng(seed)
    adj: Adj = [{1, 2}, {0, 2}, {0, 1}]
    while len(adj) < n_nodes:
        events, rates = [], []
        for u in range(len(adj)):
            for v in adj[u]:
                if u < v and _is_boundary(adj, u, v):
                    du, dv = len(adj[u]), len(adj[v])
                    if du >= c or dv >= c:
                        if any(len(adj[x]) >= c and _closable(adj, x, c)
                               for x in (u, v)):
                            events.append((u, v, True))
                            rates.append(1.0)
                    else:
                        events.append((u, v, False))
                        rates.append(np.exp(beta * (du + dv - 2 * c)))
        if not events:
            break
        r = np.array(rates)
        u, v, closing = events[int(rng.choice(len(events), p=r / r.sum()))]
        if closing:
            cand = [x for x in (u, v)
                    if len(adj[x]) >= c and _closable(adj, x, c)]
            x = cand[int(rng.integers(len(cand)))]
            a, b = _closable(adj, x, c)
            adj[a].add(b)
            adj[b].add(a)
        else:
            w = len(adj)
            adj.append({u, v})
            adj[u].add(w)
            adj[v].add(w)
    return adj


def to_csr(adj: Adj) -> sp.csr_array:
    n = len(adj)
    rows = [i for i, s in enumerate(adj) for _ in s]
    cols = [j for s in adj for j in s]
    return sp.csr_array((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def to_networkx(adj: Adj) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(len(adj)))
    G.add_edges_from((u, v) for u in range(len(adj)) for v in adj[u] if u < v)
    return G


def diameter(adj: Adj, seed: int) -> float:
    """Double-sweep diameter lower bound."""
    A = to_csr(adj)
    rng = np.random.default_rng(seed)
    best = 0.0
    for _ in range(3):
        d = dijkstra(A, indices=int(rng.integers(len(adj))), unweighted=True)
        d2 = dijkstra(A, indices=int(np.argmax(d)), unweighted=True)
        best = max(best, float(d2.max()))
    return best


def census(adj: Adj, c: int) -> Dict[str, float]:
    """Boundary length, interior degree defects, manifold violations."""
    n = len(adj)
    n_bnd_edges, n_bad = 0, 0
    on_boundary = np.zeros(n, dtype=bool)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                t = len(adj[u] & adj[v])
                if t == 1:
                    n_bnd_edges += 1
                    on_boundary[u] = on_boundary[v] = True
                elif t != 2:
                    n_bad += 1
    deg = np.array([len(s) for s in adj])
    interior = ~on_boundary
    n_int = int(interior.sum())
    return {'N': n, 'boundary_edges': n_bnd_edges,
            'boundary_frac': n_bnd_edges / n,
            'interior_frac': n_int / n,
            'interior_deg_c': float(np.mean(deg[interior] == c))
            if n_int else float('nan'),
            'interior_mean_deg': float(deg[interior].mean())
            if n_int else float('nan'),
            'max_deg': int(deg.max()), 'nonmanifold_edges': n_bad}


def save_edges(adj: Adj, path: str) -> None:
    e = np.array([(u, v) for u in range(len(adj)) for v in adj[u] if u < v],
                 dtype=np.int32)
    np.savez_compressed(path, edges=e, n=len(adj))


def load_csr(path: str) -> sp.csr_array:
    z = np.load(path)
    e, n = z['edges'], int(z['n'])
    A = sp.coo_array((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return (A + A.T).tocsr()


def run_one(spec: Tuple) -> List[Dict]:
    c, beta, seed, n_nodes, marks, save = spec
    rows: List[Dict] = []

    def snap(n: int, adj: Adj) -> None:
        rows.append({'c': c, 'beta': beta, 'seed': seed,
                     **census(adj, c), 'diameter': diameter(adj, seed)})

    t0 = time.time()
    adj = grow_sheet(n_nodes, c, beta, seed, checkpoints=marks,
                     on_checkpoint=snap)
    stalled = len(adj) < n_nodes
    if save and not stalled:
        Path("results").mkdir(exist_ok=True)
        save_edges(adj, f"results/sheet_c{c}_b{beta:g}_s{seed}_N{n_nodes}.npz")
    for r in rows:
        r['stalled_at'] = len(adj) if stalled else 0
        r['seconds'] = time.time() - t0
    if stalled and not rows:
        rows.append({'c': c, 'beta': beta, 'seed': seed, **census(adj, c),
                     'diameter': diameter(adj, seed),
                     'stalled_at': len(adj), 'seconds': time.time() - t0})
    return rows


def local_exponents(rows: List[Dict], c: int, beta: float) -> List[Dict]:
    """Per-octave exponents of seed-mean diameter and boundary length."""
    sel = [r for r in rows if r['c'] == c and r['beta'] == beta]
    Ns = sorted({r['N'] for r in sel})
    mean = {n: (np.mean([r['diameter'] for r in sel if r['N'] == n]),
                np.mean([r['boundary_edges'] for r in sel if r['N'] == n]))
            for n in Ns}
    out = []
    for a, b in zip(Ns[:-1], Ns[1:]):
        out.append({'N_lo': a, 'N_hi': b,
                    'diam_exp': float(np.log(mean[b][0] / mean[a][0])
                                      / np.log(b / a)),
                    'bnd_exp': float(np.log(mean[b][1] / mean[a][1])
                                     / np.log(b / a))})
    return out


def judge(paths: Sequence[str], n_sources: int = 60, n_probes: int = 8
          ) -> List[Dict]:
    """Window-stability and spectral verdicts on saved sheets."""
    from window_stability import audit_adjacency
    from spectral_flow import measure_adjacency, show
    out = []
    for path in paths:
        A = load_csr(path)
        name = Path(path).stem
        seed = int(name.split('_s')[1].split('_')[0])
        ws = audit_adjacency(name, A, n_sources, seed)
        sf = measure_adjacency(name, A, seed, n_probes)
        show(sf)
        out.append({'graph': name, 'ws_verdict': ws['verdict'],
                    'ws_drift': ws['drift'],
                    'ws_d_last': ws['table'][-1][1] if ws['table']
                    else float('nan'),
                    'sf_verdict': sf['verdict'],
                    'sf_value': sf['value'] if sf['verdict'] == 'FLAT'
                    else float('nan'),
                    'sf_first': float(sf['D'][0]) if len(sf['D'])
                    else float('nan'),
                    'sf_last': float(sf['D'][-1]) if len(sf['D'])
                    else float('nan')})
    print("\nJudged sheets")
    for r in out:
        print(f"  {r['graph']:>28}: window {r['ws_verdict']} (drift "
              f"{r['ws_drift']:.2f}, d {r['ws_d_last']:.2f});  spectral "
              f"{r['sf_verdict']} (D {r['sf_first']:.2f} -> "
              f"{r['sf_last']:.2f})")
    return out


def _validate() -> bool:
    ok = True
    print("[1] invariants: manifold, degree bound, determinism")
    for beta in (0.0, 2.0):
        a1 = grow_sheet(3000, 6, beta, seed=1)
        a2 = grow_sheet(3000, 6, beta, seed=1)
        cs = census(a1, 6)
        good = (a1 == a2 and cs['nonmanifold_edges'] == 0
                and cs['max_deg'] <= 6 and len(a1) == 3000)
        ok &= good
        print(f"  beta={beta}: N={cs['N']} max degree {cs['max_deg']}, "
              f"non-manifold edges {cs['nonmanifold_edges']}, "
              f"reproducible {a1 == a2}  {'OK' if good else 'FAIL'}")

    print("[2] bucketed sampler vs recompute-everything reference "
          "(N=300, beta=1, 40 seeds each)")
    fast = np.array([census(grow_sheet(300, 6, 1.0, seed=s), 6)
                     ['boundary_frac'] for s in range(40)])
    ref = np.array([census(grow_sheet_reference(300, 6, 1.0, seed=1000 + s),
                           6)['boundary_frac'] for s in range(40)])
    se = np.sqrt(fast.var(ddof=1) / 40 + ref.var(ddof=1) / 40)
    z = (fast.mean() - ref.mean()) / se
    good = abs(z) < 3.0
    ok &= good
    print(f"  boundary fraction: bucketed {fast.mean():.4f}, reference "
          f"{ref.mean():.4f}, z = {z:+.2f}  {'OK' if good else 'FAIL'}")

    print("[3] the curvature knob: c = 5 must close up (stall at finite size)")
    sizes = [len(grow_sheet(5000, 5, 2.0, seed=s)) for s in range(5)]
    good = max(sizes) < 5000
    ok &= good
    print(f"  c=5 sizes reached: {sizes}  {'OK' if good else 'FAIL'}")

    print("\nPASS: sheet growth generator" if ok
          else "\nFAIL: sheet growth generator")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Wheel-closing frontier growth with local rates",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--betas', type=float, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--caps', type=int, nargs='+', default=[6],
                    help='closure thresholds c')
    ap.add_argument('--nodes', type=int, default=512000)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--seed', type=int, default=100, help='base seed')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--judge', nargs='+', default=None,
                    help='saved results/sheet_*.npz files to run the '
                         'window-stability and spectral instruments on')
    ap.add_argument('--save-graphs', action='store_true',
                    help='save each final edge list to results/sheet_*.npz')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.validate:
        raise SystemExit(0 if _validate() else 1)
    if args.judge:
        judge(args.judge)
        return

    marks = []
    n = args.nodes
    while n >= 2000:
        marks.append(n)
        n //= 4
    marks = tuple(sorted(marks))
    specs = [(c, b, args.seed + s, args.nodes, marks, args.save_graphs)
             for c in args.caps for b in args.betas
             for s in range(args.seeds)]
    rows: List[Dict] = []
    if args.jobs <= 1:
        for sp_ in tqdm(specs, desc='growth'):
            rows.extend(run_one(sp_))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for out in tqdm(ex.map(run_one, specs), total=len(specs),
                            desc='growth'):
                rows.extend(out)

    Path("results").mkdir(exist_ok=True)
    path = f"results/sheet_growth_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path}")

    for c in args.caps:
        for b in args.betas:
            sel = [r for r in rows if r['c'] == c and r['beta'] == b]
            stalled = sorted({r['stalled_at'] for r in sel if r['stalled_at']})
            print(f"\nc={c} beta={b}"
                  + (f"   STALLED at {stalled}" if stalled else ""))
            for n in sorted({r['N'] for r in sel}):
                s_ = [r for r in sel if r['N'] == n]
                print(f"  N={n:7d} ({len(s_)} seeds): diameter "
                      f"{np.mean([r['diameter'] for r in s_]):7.1f}  "
                      f"boundary/N "
                      f"{np.mean([r['boundary_frac'] for r in s_]):.4f}  "
                      f"interior deg=c "
                      f"{np.mean([r['interior_deg_c'] for r in s_]):.3f}")
            for e in local_exponents(rows, c, b):
                print(f"  {e['N_lo']:7d} -> {e['N_hi']:7d}: diameter "
                      f"exponent {e['diam_exp']:.3f}   boundary exponent "
                      f"{e['bnd_exp']:.3f}")


if __name__ == '__main__':
    main()
