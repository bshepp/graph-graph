"""
Wave 4: the fixed-N rules against their literature.

Pre-registration: docs/superpowers/specs/2026-10-17-wave4-preregistration.md.

Labelled experiment dynamics, none registered in rules.py (both backends and every
banked result are unchanged). Locality holds throughout.

  --majority  W4-a  tie rule x schedule on a 2D torus (Goles-Olivos gate; biased-majority
                    takeover, Gaertner-Zehmakan)
  --triadic   W4-b  registered `triadic` to 5000 sweeps; Marsili closure-with-decay
                    hysteresis scan
  --br        W4-c  Bornholdt-Rohlf anti-Hebbian rewiring of a threshold network
  --ricci     W4-d  Ni-Lin-Luo-Gao Ollivier-Ricci flow with surgery
  --squares   W4-e  hard-core square energy, annealed Metropolis vs quench
  --ollivier  W4-f  Ollivier curvature (idleness 0) on sheets, dead-corner edges

Usage:
    python fixed_rules.py --validate
    python fixed_rules.py --majority | --triadic | --br | --ricci | --squares | --ollivier
"""

import argparse
import csv
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse.csgraph import connected_components, dijkstra

from simulation import create_initial_graph

JOBS = 3


def save(rows: List[Dict], tag: str) -> str:
    out = f"results/fixed_rules_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"saved: {out}")
    return out


def double_sweep_diameter(G: nx.Graph, rng: np.random.Generator) -> Tuple[int, int]:
    cc = max(nx.connected_components(G), key=len)
    H = G.subgraph(cc)
    nodes = list(H)
    s = nodes[int(rng.integers(len(nodes)))]
    d = nx.single_source_shortest_path_length(H, s)
    far = max(d, key=d.get)
    d2 = nx.single_source_shortest_path_length(H, far)
    return int(max(d2.values())), len(cc)


# --------------------------------------------------------------------------
# W4-a majority
# --------------------------------------------------------------------------

def torus_neighbours(L: int) -> np.ndarray:
    G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(L, L, periodic=True))
    return np.array([sorted(G[u]) for u in range(L * L)], dtype=np.int64)


def majority_run(nb: np.ndarray, p1: float, tie: str, sched: str, noise: float,
                 seed: int, max_sweeps: int = 200) -> Dict:
    rng = np.random.default_rng(seed)
    n = len(nb)
    s = (rng.random(n) < p1).astype(np.int8)
    zero0 = float(np.mean(s == 0))
    k = nb.shape[1]
    hist = [s.copy()]
    end = 'max_sweeps'
    for sweep in range(1, max_sweeps + 1):
        if sched == 'sync':
            ones = s[nb].sum(axis=1)
            new = np.where(2 * ones > k, 1, np.where(2 * ones < k, 0, -1)).astype(np.int8)
            tied = new < 0
            if tie == 'keep':
                new[tied] = s[tied]
            elif tie == 'bias0':
                new[tied] = 0
            else:
                new[tied] = rng.integers(0, 2, tied.sum())
            if noise > 0:
                flip = rng.random(n) < noise
                new[flip] = rng.integers(0, 2, flip.sum())
            s = new
        else:
            for u in rng.permutation(n).tolist():
                if noise > 0 and rng.random() < noise:
                    s[u] = rng.integers(0, 2)
                    continue
                ones = int(s[nb[u]].sum())
                if 2 * ones > k:
                    s[u] = 1
                elif 2 * ones < k:
                    s[u] = 0
                elif tie == 'bias0':
                    s[u] = 0
                elif tie == 'random':
                    s[u] = rng.integers(0, 2)
        if noise == 0:
            if np.array_equal(s, hist[-1]):
                end = 'fixed'
                break
            if sched == 'sync' and len(hist) >= 2 and np.array_equal(s, hist[-2]):
                end = 'cycle2'
                break
        hist = (hist + [s.copy()])[-2:]
    return {'p1': p1, 'tie': tie, 'sched': sched, 'noise': noise, 'seed': seed,
            'zero_start': zero0, 'zero_end': float(np.mean(s == 0)), 'end': end,
            'sweeps': sweep}


def _maj_job(spec):
    nb = torus_neighbours(100)
    return majority_run(nb, *spec)


def run_majority() -> List[Dict]:
    specs = [(p1, tie, sched, noise, seed)
             for p1 in (0.5, 0.6, 0.7, 0.8, 0.9) for tie in ('keep', 'bias0', 'random')
             for sched in ('sync', 'seq') for noise in (0.0, 0.01) for seed in range(4)]
    with ProcessPoolExecutor(JOBS) as ex:
        rows = list(ex.map(_maj_job, specs, chunksize=4))
    print(f"{'p1':>4} {'tie':>6} {'sched':>5} {'noise':>5} {'zero start':>10} "
          f"{'zero end (4 seeds)':>28} {'ends':>24}")
    for p1 in (0.5, 0.6, 0.7, 0.8, 0.9):
        for tie in ('keep', 'bias0', 'random'):
            for sched in ('sync', 'seq'):
                for noise in (0.0, 0.01):
                    rs = [r for r in rows if (r['p1'], r['tie'], r['sched'], r['noise'])
                          == (p1, tie, sched, noise)]
                    print(f"{p1:4.1f} {tie:>6} {sched:>5} {noise:5.2f} "
                          f"{np.mean([r['zero_start'] for r in rs]):10.3f} "
                          f"{' '.join(f'{r['zero_end']:.3f}' for r in rs):>28} "
                          f"{','.join(sorted({r['end'] for r in rs})):>24}")
    det = [r for r in rows if r['noise'] == 0]
    gate_sync = all(r['end'] in ('fixed', 'cycle2') for r in det if r['sched'] == 'sync')
    gate_seq = all(r['end'] == 'fixed' for r in det if r['sched'] == 'seq')
    gate_det = all(r['end'] in ('fixed', 'cycle2') for r in det
                   if r['sched'] == 'sync' and r['tie'] != 'random') and \
        all(r['end'] == 'fixed' for r in det if r['sched'] == 'seq' and r['tie'] != 'random')
    print(f"\nGate as frozen (all noise-0 runs): sync {'PASS' if gate_sync else 'FAIL'}, "
          f"seq {'PASS' if gate_seq else 'FAIL'}; restricted to the deterministic tie rules "
          f"(keep, bias0): {'PASS' if gate_det else 'FAIL'}")
    b = [r for r in rows if (r['p1'], r['tie'], r['sched'], r['noise']) == (0.7, 'bias0', 'sync', 0.0)]
    kp = [r for r in rows if (r['p1'], r['tie'], r['sched'], r['noise']) == (0.7, 'keep', 'sync', 0.0)]
    take = all(r['zero_end'] > 0.9 for r in b) and \
        all(abs(r['zero_end'] - r['zero_start']) <= 0.1 for r in kp)
    print(f"W4-a: {'BIAS TAKES OVER' if take else 'see table'} (bias0 p1 0.7 final zeros "
          f"{[round(r['zero_end'], 3) for r in b]}; keep {[round(r['zero_end'], 3) for r in kp]})")
    for tie in ('keep', 'bias0'):
        rs = [r for r in rows if (r['p1'], r['tie'], r['noise']) == (0.5, tie, 0.01)]
        print(f"  banked setting p1 0.5 noise 0.01, {tie}: final zeros mean "
              f"{np.mean([r['zero_end'] for r in rs]):.3f}")
    return rows


# --------------------------------------------------------------------------
# W4-b triadic long run and the Marsili variant
# --------------------------------------------------------------------------

def _triadic_job(spec):
    from rules import triadic_closure
    n, seed, sweeps = spec
    random.seed(seed)
    np.random.seed(seed)
    G = create_initial_graph(n, topology='random', k=6, seed=seed)
    rng = np.random.default_rng(seed + 1)
    rows = []
    for t in range(0, sweeps + 1):
        if t % 100 == 0:
            d, lc = double_sweep_diameter(G, rng)
            rows.append({'n': n, 'seed': seed, 'sweep': t, 'diameter': d,
                         'largest': lc / n, 'clustering': nx.average_clustering(G)})
        if t < sweeps:
            triadic_closure(G)
    return rows


class DynGraph:
    """Undirected simple graph with O(1) random edge, random neighbour,
    insertion and deletion."""

    def __init__(self, n: int):
        self.nb: List[List[int]] = [[] for _ in range(n)]
        self.pos: List[Dict[int, int]] = [{} for _ in range(n)]
        self.edges: List[Tuple[int, int]] = []
        self.epos: Dict[Tuple[int, int], int] = {}

    def has(self, u: int, v: int) -> bool:
        return v in self.pos[u]

    def add(self, u: int, v: int) -> None:
        if u == v or v in self.pos[u]:
            return
        for a, b in ((u, v), (v, u)):
            self.pos[a][b] = len(self.nb[a])
            self.nb[a].append(b)
        e = (u, v) if u < v else (v, u)
        self.epos[e] = len(self.edges)
        self.edges.append(e)

    def remove_edge_at(self, i: int) -> None:
        e = self.edges[i]
        last = self.edges[-1]
        self.edges[i] = last
        self.epos[last] = i
        self.edges.pop()
        del self.epos[e]
        u, v = e
        for a, b in ((u, v), (v, u)):
            j = self.pos[a].pop(b)
            lastn = self.nb[a][-1]
            self.nb[a][j] = lastn
            if lastn != b:
                self.pos[a][lastn] = j
            self.nb[a].pop()


def marsili(n: int, seed: int, lam: float = 0.01, eta: float = 0.0005,
            sweeps_per: int = 300) -> List[Dict]:
    rng = np.random.default_rng(seed)
    g = DynGraph(n)
    xis = [round(x, 2) for x in np.arange(0, 1.0001, 0.05)]
    path = [('up', x) for x in xis] + [('down', x) for x in xis[-2::-1]]
    rows = []
    for direction, xi in path:
        degs = []
        for t in range(sweeps_per):
            m = rng.binomial(len(g.edges), lam) if g.edges else 0
            for _ in range(m):
                g.remove_edge_at(int(rng.integers(len(g.edges))))
            for u in rng.permutation(n).tolist():
                if g.nb[u] and rng.random() < xi:
                    x = g.nb[u][int(rng.integers(len(g.nb[u])))]
                    if g.nb[x]:
                        w = g.nb[x][int(rng.integers(len(g.nb[x])))]
                        g.add(u, w)
                if rng.random() < eta:
                    g.add(u, int(rng.integers(n)))
            if t >= sweeps_per - 100:
                degs.append(2 * len(g.edges) / n)
        rows.append({'seed': seed, 'dir': direction, 'xi': xi,
                     'mean_degree': float(np.mean(degs))})
    return rows


def _marsili_job(seed):
    return marsili(1000, seed)


def run_triadic() -> List[Dict]:
    specs = [(n, s, 5000) for n in (4000, 1000) for s in range(3)]
    with ProcessPoolExecutor(JOBS) as ex:
        res = list(ex.map(_triadic_job, specs))
        mres = list(ex.map(_marsili_job, [0, 1]))
    rows = [r for rs in res for r in rs]
    ok = True
    for rs in res:
        a, z = rs[0], rs[-1]
        peak = max(rs, key=lambda r: r['diameter'])
        coll = z['diameter'] < a['diameter'] and z['clustering'] > 0.5
        ok &= coll
        print(f"triadic N {a['n']} seed {a['seed']}: diameter {a['diameter']} -> peak "
              f"{peak['diameter']} at {peak['sweep']} -> {z['diameter']} at 5000; clustering "
              f"{a['clustering']:.3f} -> {z['clustering']:.3f}; largest component "
              f"{z['largest']:.3f}")
    print(f"W4-b(1): {'COLLAPSE' if ok else 'see table'}")
    mrows = [r for rs in mres for r in rs]
    hyst = False
    for seed in (0, 1):
        up = {r['xi']: r['mean_degree'] for r in mrows if r['seed'] == seed and r['dir'] == 'up'}
        dn = {r['xi']: r['mean_degree'] for r in mrows if r['seed'] == seed and r['dir'] == 'down'}
        print(f"Marsili seed {seed}: xi: up / down mean degree")
        for xi in sorted(dn):
            ratio = max(up[xi], dn[xi]) / max(min(up[xi], dn[xi]), 1e-9)
            hyst |= ratio >= 2
            print(f"  {xi:.2f}: {up[xi]:7.2f} / {dn[xi]:7.2f}{'   <- x2' if ratio >= 2 else ''}")
    print(f"W4-b(2): {'HYSTERESIS' if hyst else 'no hysteresis'}")
    return rows + [{'marsili': 1, **r} for r in mrows]


# --------------------------------------------------------------------------
# W4-c Bornholdt-Rohlf
# --------------------------------------------------------------------------

def bornholdt_rohlf(n: int, k0: float, seed: int, epochs: int = 5000, steps: int = 200,
                    window: int = 100) -> List[float]:
    rng = np.random.default_rng(seed)
    W = np.zeros((n, n), dtype=np.int8)
    m = int(k0 * n)
    for _ in range(m):
        i, j = rng.integers(n), rng.integers(n)
        if i != j:
            W[i, j] = rng.choice([-1, 1])
    Ks = []
    for ep in range(epochs):
        s = rng.choice([-1, 1], n).astype(np.int32)
        hist = np.zeros((window, n), dtype=np.int8)
        for t in range(steps):
            h = W @ s
            s = np.where(h >= 0, 1, -1).astype(np.int32)
            if t >= steps - window:
                hist[t - (steps - window)] = s
        i = int(rng.integers(n))
        frozen = bool(np.all(hist[:, i] == hist[0, i]))
        if frozen:
            j = int(rng.integers(n))
            if j != i and W[i, j] == 0:
                W[i, j] = rng.choice([-1, 1])
        else:
            nz = np.nonzero(W[i])[0]
            if len(nz):
                W[i, nz[int(rng.integers(len(nz)))]] = 0
        Ks.append(np.count_nonzero(W) / n)
    return Ks


def _br_job(spec):
    k0, seed = spec
    return {'k0': k0, 'seed': seed, 'K': bornholdt_rohlf(500, k0, seed)}


def run_br() -> List[Dict]:
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(_br_job, [(1.0, 0), (1.0, 1), (4.0, 0), (4.0, 1)]))
    rows = []
    late = {}
    for r in res:
        K = np.array(r['K'])
        late[(r['k0'], r['seed'])] = float(K[-1000:].mean())
        print(f"K0 {r['k0']} seed {r['seed']}: K at epochs 0/1000/2000/3000/4000/5000 = "
              + ' '.join(f"{K[min(e, len(K) - 1)]:.2f}" for e in (0, 999, 1999, 2999, 3999, 4999))
              + f"; last-1000 mean {late[(r['k0'], r['seed'])]:.2f}")
        rows += [{'k0': r['k0'], 'seed': r['seed'], 'epoch': e, 'K': float(K[e])}
                 for e in range(0, len(K), 50)]
    lo = np.mean([v for (k, s), v in late.items() if k == 1.0])
    hi = np.mean([v for (k, s), v in late.items() if k == 4.0])
    ok = abs(lo - hi) <= 0.5 and 1.5 <= lo <= 3.0 and 1.5 <= hi <= 3.0
    print(f"W4-c: {'SELF-ORGANIZED' if ok else 'see table'} (late K from K0 = 1: {lo:.2f}, "
          f"from K0 = 4: {hi:.2f})")
    return rows


# --------------------------------------------------------------------------
# Ollivier curvature
# --------------------------------------------------------------------------

def w1(mu_nodes, mu_mass, nu_nodes, nu_mass, D: np.ndarray) -> float:
    """Earth mover's distance between two finite measures with cost matrix D
    (rows mu, columns nu)."""
    a, b = len(mu_nodes), len(nu_nodes)
    if a == b and np.allclose(mu_mass, mu_mass[0]) and np.allclose(nu_mass, nu_mass[0]) \
            and np.isclose(mu_mass[0], nu_mass[0]):
        r, c = linear_sum_assignment(D)
        return float(D[r, c].sum() * mu_mass[0])
    A_eq = []
    for i in range(a):
        row = np.zeros(a * b)
        row[i * b:(i + 1) * b] = 1
        A_eq.append(row)
    for j in range(b):
        row = np.zeros(a * b)
        row[j::b] = 1
        A_eq.append(row)
    res = linprog(D.ravel(), A_eq=np.array(A_eq), b_eq=np.concatenate([mu_mass, nu_mass]),
                  bounds=(0, None), method='highs')
    return float(res.fun)


def ollivier_edge(nb: Sequence[Sequence[int]], x: int, y: int, alpha: float,
                  dist) -> float:
    """kappa(x, y) = 1 - W1(m_x, m_y) / d(x, y); m_v = alpha at v plus (1 - alpha)
    spread uniformly on v's neighbours. `dist(u, v)` gives the ground distance."""
    def measure(v):
        nodes = list(nb[v])
        mass = [(1 - alpha) / len(nodes)] * len(nodes)
        if alpha > 0:
            nodes = [v] + nodes
            mass = [alpha] + mass
        return nodes, np.array(mass)
    mx, px = measure(x)
    my, py = measure(y)
    D = np.array([[dist(u, v) for v in my] for u in mx], dtype=float)
    return 1.0 - w1(mx, px, my, py, D) / dist(x, y)


# --------------------------------------------------------------------------
# W4-d Ricci flow with surgery
# --------------------------------------------------------------------------

def ricci_flow_surgery(G: nx.Graph, iterations: int = 20, alpha: float = 0.5
                       ) -> Tuple[Dict[Tuple[int, int], float], List[Tuple[int, int]]]:
    nodes = list(G)
    idx = {u: i for i, u in enumerate(nodes)}
    nb = [[idx[v] for v in G[u]] for u in nodes]
    edges = [(idx[u], idx[v]) for u, v in G.edges()]
    w = np.ones(len(edges))
    total = w.sum()
    n = len(nodes)
    for it in range(iterations):
        A = sp.coo_array((np.concatenate([w, w]),
                          (np.concatenate([[e[0] for e in edges], [e[1] for e in edges]]),
                           np.concatenate([[e[1] for e in edges], [e[0] for e in edges]]))),
                         shape=(n, n)).tocsr()
        Dm = dijkstra(A, directed=False)
        kap = np.array([ollivier_edge(nb, x, y, alpha, lambda u, v: Dm[u, v])
                        for x, y in edges])
        w = w - kap * w
        w = np.maximum(w, 1e-9)
        w *= total / w.sum()
    med = np.median(w)
    cut = [edges[i] for i in range(len(edges)) if w[i] > 2 * med]
    weights = {edges[i]: float(w[i]) for i in range(len(edges))}
    return weights, cut


def _ricci_job(spec):
    from shortcuts import inject_shortcuts
    from dimension import _estimate_max_radius_sparse
    from window_stability import audit_adjacency
    kind, seed = spec
    random.seed(seed)
    np.random.seed(seed)
    if kind == 'lattice+shortcuts':
        G = create_initial_graph(2000, topology='triangular', k=6, seed=seed)
        G = nx.convert_node_labels_to_integers(G)
        base_edges = {tuple(sorted(e)) for e in G.edges()}
        A0 = nx.to_scipy_sparse_array(G, weight=None, format='csr', dtype=np.float32)
        r0 = _estimate_max_radius_sparse(A0)
        inject_shortcuts(G, 40, min_distance=2 * r0 + 1)
        sc = {tuple(sorted(e)) for e in G.edges()} - base_edges
        _, cut = ricci_flow_surgery(G)
        nodes = list(G)
        cut_set = {tuple(sorted((nodes[a], nodes[b]))) for a, b in cut}
        return {'kind': kind, 'seed': seed, 'shortcuts': len(sc),
                'shortcuts_cut': len(sc & cut_set),
                'lattice_edges': len(base_edges),
                'lattice_cut': len(cut_set - sc)}
    G = create_initial_graph(1000, topology='random', k=6, seed=seed)
    G = nx.convert_node_labels_to_integers(G)
    _, cut = ricci_flow_surgery(G)
    nodes = list(G)
    H = G.copy()
    H.remove_edges_from([(nodes[a], nodes[b]) for a, b in cut])
    comps = sorted((len(c) for c in nx.connected_components(H)), reverse=True)
    big = max(nx.connected_components(H), key=len)
    Hs = nx.convert_node_labels_to_integers(H.subgraph(big).copy())
    au = audit_adjacency('random after surgery', nx.to_scipy_sparse_array(
        Hs, weight=None, format='csr', dtype=float), 30, seed)
    return {'kind': kind, 'seed': seed, 'cut': len(cut), 'edges': G.number_of_edges(),
            'components_50plus': sum(1 for c in comps if c >= 50),
            'largest': comps[0], 'audit': au['verdict']}


def run_ricci() -> List[Dict]:
    specs = [('lattice+shortcuts', s) for s in range(3)] + [('random', s) for s in range(3)]
    with ProcessPoolExecutor(JOBS) as ex:
        rows = list(ex.map(_ricci_job, specs))
    lat = [r for r in rows if r['kind'] == 'lattice+shortcuts']
    rnd = [r for r in rows if r['kind'] == 'random']
    for r in lat:
        print(f"lattice+shortcuts seed {r['seed']}: shortcuts cut {r['shortcuts_cut']}/"
              f"{r['shortcuts']}, lattice edges cut {r['lattice_cut']}/{r['lattice_edges']}")
    for r in rnd:
        print(f"random seed {r['seed']}: cut {r['cut']}/{r['edges']} edges; components >= 50: "
              f"{r['components_50plus']}; largest {r['largest']}; audit {r['audit']}")
    ok1 = all(r['shortcuts_cut'] >= 0.9 * r['shortcuts'] and
              r['lattice_cut'] <= 0.02 * r['lattice_edges'] for r in lat)
    ok2 = all(r['components_50plus'] >= 2 or r['audit'] != 'DIMENSION DEFINED' for r in rnd)
    print(f"W4-d: {'SHORTCUT SURGERY' if ok1 else 'no clean shortcut surgery'}; random graph: "
          f"{'FRAGMENTS, NO DIMENSION' if ok2 else 'see table'}")
    return rows


# --------------------------------------------------------------------------
# W4-e hard-core squares
# --------------------------------------------------------------------------

class SquareGraph:
    def __init__(self, n: int, rng: np.random.Generator):
        self.rng = rng
        G = nx.random_regular_graph(4, n, seed=int(rng.integers(1 << 30)))
        self.adj = [set(G[u]) for u in range(n)]
        self.edges = [tuple(sorted(e)) for e in G.edges()]
        self.eidx = {e: i for i, e in enumerate(self.edges)}
        # remove triangles by swaps that reduce the triangle count
        while self.triangles() > 0:
            self._swap()

    def triangles(self) -> int:
        return sum(len(self.adj[u] & self.adj[v]) for u, v in self.edges) // 3

    def squares(self, u: int, v: int) -> int:
        return sum(len(self.adj[x] & self.adj[v] - {u}) for x in self.adj[u] if x != v)

    def _propose(self):
        E = len(self.edges)
        i, j = self.rng.integers(E), self.rng.integers(E)
        if i == j:
            return None
        (a, b), (c, d) = self.edges[i], self.edges[j]
        if self.rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4 or c in self.adj[a] or d in self.adj[b]:
            return None
        return i, j, a, b, c, d

    def _apply(self, i, j, a, b, c, d):
        for (u, v) in ((a, b), (c, d)):
            self.adj[u].discard(v)
            self.adj[v].discard(u)
        for (u, v) in ((a, c), (b, d)):
            self.adj[u].add(v)
            self.adj[v].add(u)
        self.edges[i] = tuple(sorted((a, c)))
        self.edges[j] = tuple(sorted((b, d)))

    def _swap(self) -> None:
        """A swap kept only if it lowers the triangle count (start-up only)."""
        p = self._propose()
        if p is None:
            return
        i, j, a, b, c, d = p
        before = self.triangles()
        self._apply(i, j, a, b, c, d)
        if self.triangles() >= before:
            self._apply(i, j, a, c, b, d)       # undo: (a,c),(b,d) -> (a,b),(c,d)

    def near_nodes(self, nodes) -> Set[int]:
        near = set(nodes)
        for u in nodes:
            near |= self.adj[u]
        return near

    def edges_at(self, near) -> Set[Tuple[int, int]]:
        out = set()
        for u in near:
            for v in self.adj[u]:
                out.add((u, v) if u < v else (v, u))
        return out

    def energy_of(self, es) -> int:
        return sum(abs(self.squares(*e) - 2) for e in es)

    def step(self, T: float) -> bool:
        """One Metropolis double-edge swap; every 4-cycle whose square count
        can change lies on edges touching N[a, b, c, d] (taken before the
        swap; after it the neighbourhoods are a subset of the same set)."""
        p = self._propose()
        if p is None:
            return False
        i, j, a, b, c, d = p
        near = self.near_nodes((a, b, c, d))
        e0 = self.energy_of(self.edges_at(near))
        self._apply(i, j, a, b, c, d)
        if (self.adj[a] & self.adj[c]) or (self.adj[b] & self.adj[d]):
            ok = False
        else:
            dE = self.energy_of(self.edges_at(near)) - e0
            ok = dE <= 0 or (T > 0 and self.rng.random() < np.exp(-dE / T))
        if not ok:
            self._apply(i, j, a, c, b, d)        # (a,c),(b,d) -> (a,b),(c,d)
        return ok

    def stats(self) -> Dict[str, float]:
        s = np.array([self.squares(*e) for e in self.edges])
        n = len(self.adj)
        A = sp.coo_array((np.ones(len(self.edges)), ([e[0] for e in self.edges],
                                                     [e[1] for e in self.edges])),
                         shape=(n, n))
        A = (A + A.T).tocsr()
        D = dijkstra(A, unweighted=True, limit=8.5)
        vol = {r: float(np.mean((D <= r).sum(axis=1))) for r in range(2, 9)}
        return {'E_per_edge': float(np.abs(s - 2).mean()), 'phi': float(np.mean(s == 2)),
                **{f'V{r}': vol[r] for r in vol}}


def _squares_job(spec):
    mode, seed, n, sweeps = spec
    rng = np.random.default_rng(seed)
    g = SquareGraph(n, rng)
    props = 2 * n
    T0, T1 = 2.0, 0.05
    for sw in range(sweeps):
        T = 0.0 if mode == 'quench' else T0 * (T1 / T0) ** (sw / max(sweeps - 1, 1))
        for _ in range(props):
            g.step(T)
    return {'mode': mode, 'seed': seed, **g.stats()}


def run_squares(n: int = 900, sweeps: int = 1000) -> List[Dict]:
    specs = [(m, s, n, sweeps) for m in ('anneal', 'quench') for s in range(4)]
    with ProcessPoolExecutor(4) as ex:
        rows = list(ex.map(_squares_job, specs))
    tor = nx.convert_node_labels_to_integers(nx.grid_2d_graph(30, 30, periodic=True))
    At = nx.to_scipy_sparse_array(tor, weight=None, format='csr', dtype=float)
    Dt = dijkstra(At, unweighted=True, limit=8.5)
    vt = {r: float(np.mean((Dt <= r).sum(axis=1))) for r in range(2, 9)}
    print(f"torus 30x30 ball volumes: " + ' '.join(f"r{r}:{vt[r]:.0f}" for r in vt))
    for r in rows:
        print(f"{r['mode']:>6} seed {r['seed']}: E/edge {r['E_per_edge']:.3f}  phi {r['phi']:.3f}  "
              + ' '.join(f"r{k}:{r[f'V{k}']:.0f}" for k in range(2, 9)))
    an = [r for r in rows if r['mode'] == 'anneal']
    qu = [r for r in rows if r['mode'] == 'quench']
    good = sum(1 for r in an if r['phi'] >= 0.9 and abs(r['V6'] / vt[6] - 1) <= 0.10)
    if good >= 2 and all(r['phi'] <= 0.6 for r in qu):
        v = 'THERMAL ROUTE WORKS'
    elif all(r['phi'] < 0.6 for r in an):
        v = 'NO ROUTE'
    else:
        v = 'UNDECIDED'
    print(f"W4-e: {v} (annealed seeds meeting both conditions: {good}/4; quench phi "
          f"{[round(r['phi'], 3) for r in qu]})")
    return rows


# --------------------------------------------------------------------------
# W4-f Ollivier curvature on sheets
# --------------------------------------------------------------------------

def _sheet_ollivier_job(spec):
    from sheet_growth import grow_sheet
    from blocked_closure import blocked_nodes
    beta, seed, n = spec
    adj = grow_sheet(n, 6, beta, seed)
    nb = [sorted(s) for s in adj]
    cache: Dict[int, Dict[int, int]] = {}

    def ball3(u):
        if u not in cache:
            d = {u: 0}
            fr = [u]
            for h in (1, 2, 3):
                nxt = []
                for x in fr:
                    for y in adj[x]:
                        if y not in d:
                            d[y] = h
                            nxt.append(y)
                fr = nxt
            cache[u] = d
        return cache[u]

    def dist(u, v):
        return ball3(u).get(v, 4)

    edges = [(u, v) for u in range(len(adj)) for v in adj[u] if u < v]
    kap = {}
    for k, (u, v) in enumerate(edges):
        kap[(u, v)] = ollivier_edge(nb, u, v, 0.0, dist)
        if len(cache) > 200000:
            cache.clear()
    on_b = set()
    for u, v in edges:
        if len(adj[u] & adj[v]) == 1:
            on_b |= {u, v}
    interior = [kap[e] for e in edges if e[0] not in on_b and e[1] not in on_b]
    dead = set(blocked_nodes(adj, 6))
    pairs = [e for e in edges if e[0] in dead and e[1] in dead]
    vals = np.array(list(kap.values()))
    p1 = float(np.percentile(vals, 1))
    return {'beta': beta, 'seed': seed, 'N': len(adj), 'edges': len(edges),
            'interior_values': sorted({round(x, 9) for x in interior})[:5],
            'n_interior': len(interior),
            'p1': p1, 'min': float(vals.min()), 'median': float(np.median(vals)),
            'frac_negative': float(np.mean(vals < -1e-9)),
            'dead': len(dead), 'pairs': len(pairs),
            'pair_kappa': [round(kap[e], 4) for e in pairs],
            'pairs_below_p1': sum(1 for e in pairs if kap[e] < p1),
            'boundary_kappa_hist': np.unique(np.round([kap[e] for e in edges
                                                        if len(adj[e[0]] & adj[e[1]]) == 1], 4),
                                              return_counts=True)}


def run_ollivier() -> List[Dict]:
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(_sheet_ollivier_job, [(1.0, 0, 50000), (3.0, 0, 50000)]))
    rows = []
    verdict = 'see table'
    for r in res:
        vals, cnts = r['boundary_kappa_hist']
        print(f"sheet beta {r['beta']}: N {r['N']}, edges {r['edges']}; interior-edge curvature "
              f"value(s) {r['interior_values']} over {r['n_interior']} edges; median {r['median']:.4f}"
              f", 1st percentile {r['p1']:.4f}, min {r['min']:.4f}, negative fraction "
              f"{r['frac_negative']:.4f}; dead corners {r['dead']}, dead pairs {r['pairs']}, "
              f"pair curvatures {r['pair_kappa'][:12]}, below p1: {r['pairs_below_p1']}")
        print("   boundary-edge curvature histogram: " +
              ' '.join(f"{v}:{c}" for v, c in zip(vals, cnts)))
        if r['pairs']:
            verdict = ('DEAD CORNERS ARE THE MOST NEGATIVE'
                       if r['pairs_below_p1'] >= 0.9 * r['pairs'] else 'not the most negative')
        rows.append({k: (str(v) if isinstance(v, (list, tuple)) else v) for k, v in r.items()
                     if k != 'boundary_kappa_hist'})
    check = all(len(r['interior_values']) == 1 for r in res)
    print(f"instrument check (one interior value per sheet): {'PASS' if check else 'FAIL'}")
    print(f"W4-f: {verdict}")
    return rows


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _validate() -> bool:
    ok = True
    nb = torus_neighbours(20)
    r = majority_run(nb, 0.5, 'keep', 'sync', 0.0, 0)
    good = r['end'] in ('fixed', 'cycle2')
    print(f"[{'PASS' if good else 'FAIL'}] majority sync keep ends {r['end']}")
    ok &= good
    r = majority_run(nb, 1.0, 'bias0', 'seq', 0.0, 0)
    good = r['zero_end'] == 0.0 and r['end'] == 'fixed'
    print(f"[{'PASS' if good else 'FAIL'}] all-ones state is fixed under bias0 (no ties)")
    ok &= good
    # Ollivier curvature known values (idleness 0, hop distance): complete graph K_n
    # edge: 1 - W1/1 with W1 = 1/(n-1)... computed via the instrument on K5 and on a cycle.
    K5 = nx.complete_graph(5)
    nbk = [sorted(K5[u]) for u in range(5)]
    Dk = dict(nx.all_pairs_shortest_path_length(K5))
    kk = ollivier_edge(nbk, 0, 1, 0.0, lambda u, v: Dk[u][v])
    good = abs(kk - (1 - 1 / 4)) < 1e-9 or abs(kk - 0.5) < 1e-9
    C = nx.cycle_graph(10)
    nbc = [sorted(C[u]) for u in range(10)]
    Dc = dict(nx.all_pairs_shortest_path_length(C))
    kc = ollivier_edge(nbc, 0, 1, 0.0, lambda u, v: Dc[u][v])
    good &= abs(kc) < 1e-9
    print(f"[{'PASS' if good else 'FAIL'}] Ollivier (idleness 0): K5 edge {kk:.4f}, cycle edge "
          f"{kc:.4f} (0)")
    ok &= good
    # Unequal supports go through the LP: a star leaf-centre edge.
    S = nx.star_graph(4)
    nbs = [sorted(S[u]) for u in range(5)]
    Ds = dict(nx.all_pairs_shortest_path_length(S))
    ks = ollivier_edge(nbs, 0, 1, 0.0, lambda u, v: Ds[u][v])
    # m_0 uniform on leaves 1..4; m_1 = delta_0; W1 = 1 -> kappa = 0.
    good = abs(ks) < 1e-9
    print(f"[{'PASS' if good else 'FAIL'}] Ollivier on a star edge (LP path): {ks:.4f} (0)")
    ok &= good
    rng = np.random.default_rng(0)
    g = SquareGraph(200, rng)
    tri = g.triangles()
    degs = {len(a) for a in g.adj}
    for _ in range(2000):
        g.step(0.5)
    tri2 = g.triangles()
    good = tri == 0 and tri2 == 0 and degs == {4} and {len(a) for a in g.adj} == {4}
    print(f"[{'PASS' if good else 'FAIL'}] square graph stays 4-regular and triangle-free "
          f"(triangles {tri} -> {tri2})")
    ok &= good
    tor = nx.convert_node_labels_to_integers(nx.grid_2d_graph(10, 10, periodic=True))
    g.adj = [set(tor[u]) for u in range(100)]
    g.edges = [tuple(sorted(e)) for e in tor.edges()]
    st = g.stats()
    good = abs(st['phi'] - 1.0) < 1e-12 and st['E_per_edge'] == 0
    print(f"[{'PASS' if good else 'FAIL'}] 10x10 torus: every edge in exactly 2 squares "
          f"(phi {st['phi']}, E {st['E_per_edge']})")
    ok &= good
    # DynGraph bookkeeping against a plain set model.
    dg = DynGraph(50)
    model = set()
    for _ in range(5000):
        if dg.edges and rng.random() < 0.4:
            i = int(rng.integers(len(dg.edges)))
            model.discard(dg.edges[i])
            dg.remove_edge_at(i)
        else:
            u, v = int(rng.integers(50)), int(rng.integers(50))
            if u != v:
                dg.add(u, v)
                model.add((min(u, v), max(u, v)))
    nbok = all(set(dg.nb[u]) == {v for e in model for v in e if u in e and v != u}
               for u in range(50))
    good = set(dg.edges) == model and nbok
    print(f"[{'PASS' if good else 'FAIL'}] dynamic graph bookkeeping matches a set model "
          f"({len(model)} edges)")
    ok &= good
    print('\nVALIDATION', 'PASSED' if ok else 'FAILED')
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_mutually_exclusive_group(required=True)
    for f in ('validate', 'majority', 'triadic', 'br', 'ricci', 'squares', 'ollivier'):
        m.add_argument(f'--{f}', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.validate:
        raise SystemExit(0 if _validate() else 1)
    fn = {'majority': run_majority, 'triadic': run_triadic, 'br': run_br,
          'ricci': run_ricci, 'squares': run_squares, 'ollivier': run_ollivier}
    for k, f in fn.items():
        if getattr(args, k):
            save(f(), k)


if __name__ == '__main__':
    main()
