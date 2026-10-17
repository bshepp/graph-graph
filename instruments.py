"""
Wave 3 instruments and their known-answer controls.

Pre-registration: docs/superpowers/specs/2026-10-17-wave3-preregistration.md.

  --spectral   W3-a  spectral plateau (saturation fit) next to the window audit, on
                     tori, a uniform random tree, a uniform spanning tree, sheet, grown
  --gap        W3-b  fraction of normalized-Laplacian eigenvalues below the
                     Kesten-McKay edge, N = 5000
  --reff       W3-c  effective resistance source-target with and without the offset
                     portal, in the banked E3 configuration
  --triang     W3-e  random sphere triangulation (uniform edge flips) vs sheet:
                     ball volume and hull-boundary exponents
  --boxes      W3-f  random sequential ball covering, sheet vs grown

Control graphs are built from adjacency only (lattice index arithmetic exists only
inside networkx's generators); every measurement uses hop distance.

Usage:
    python instruments.py --validate
    python instruments.py --spectral | --gap | --reff | --triang | --boxes
"""

import argparse
import csv
import heapq
import random
import time
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.optimize import curve_fit
from scipy.sparse.csgraph import connected_components, dijkstra

from simulation import create_initial_graph
from sheet_growth import load_csr
import spectral_flow as sf
from window_stability import audit_adjacency

SHEET_512K = 'results/sheet_c6_b3_s100_N512000.npz'
SHEET_308 = 'results/sheet_cross_b2_s308_N4194304.npz'


def nx_csr(G: nx.Graph) -> sp.csr_array:
    G = nx.convert_node_labels_to_integers(G)
    return sp.csr_array(nx.to_scipy_sparse_array(G, format='csr', dtype=float,
                                                 weight=None))


def neighbour_lists(A: sp.csr_array) -> List[List[int]]:
    return [A.indices[A.indptr[i]:A.indptr[i + 1]].tolist() for i in range(A.shape[0])]


# --------------------------------------------------------------------------
# Control graph builders
# --------------------------------------------------------------------------

def torus(L: int, dim: int) -> sp.csr_array:
    return nx_csr(nx.grid_graph(dim=[L] * dim, periodic=True))


def prufer_tree(n: int, rng: np.random.Generator) -> sp.csr_array:
    """Uniform random labelled tree on n nodes (decoded Pruefer sequence)."""
    seq = rng.integers(0, n, n - 2)
    degree = np.ones(n, dtype=np.int64)
    np.add.at(degree, seq, 1)
    leaves = [i for i in range(n) if degree[i] == 1]
    heapq.heapify(leaves)
    rows, cols = [], []
    for x in seq.tolist():
        leaf = heapq.heappop(leaves)
        rows.append(leaf)
        cols.append(x)
        degree[x] -= 1
        if degree[x] == 1:
            heapq.heappush(leaves, x)
    u, v = heapq.heappop(leaves), heapq.heappop(leaves)
    rows.append(u)
    cols.append(v)
    A = sp.coo_array((np.ones(n - 1), (rows, cols)), shape=(n, n))
    return (A + A.T).tocsr()


def wilson_ust(A: sp.csr_array, rng: np.random.Generator) -> sp.csr_array:
    """Uniform spanning tree of a connected graph (Wilson's algorithm)."""
    nb = neighbour_lists(A)
    n = len(nb)
    in_tree = np.zeros(n, dtype=bool)
    nxt = -np.ones(n, dtype=np.int64)
    root = int(rng.integers(n))
    in_tree[root] = True
    order = rng.permutation(n)
    for s in order.tolist():
        u = s
        while not in_tree[u]:
            nbs = nb[u]
            nxt[u] = nbs[int(rng.integers(len(nbs)))]
            u = int(nxt[u])
        u = s
        while not in_tree[u]:
            in_tree[u] = True
            u = int(nxt[u])
    rows = [i for i in range(n) if nxt[i] >= 0]
    cols = [int(nxt[i]) for i in rows]
    T = sp.coo_array((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return (T + T.T).tocsr()


def random_regular(n: int, d: int, seed: int) -> sp.csr_array:
    return nx_csr(nx.random_regular_graph(d, n, seed=seed))


def topology(n: int, name: str, seed: int) -> sp.csr_array:
    random.seed(seed)
    np.random.seed(seed)
    G = create_initial_graph(n, topology=name, k=6, seed=seed)
    return nx_csr(G)


# --------------------------------------------------------------------------
# W3-a spectral plateau
# --------------------------------------------------------------------------

def saturation_fit(grid: np.ndarray, D: np.ndarray) -> Dict[str, float]:
    m = grid >= 16
    g, d = grid[m], D[m]
    if len(g) < 4:
        return {'a': float('nan'), 'b': float('nan'), 'c': float('nan')}
    f = lambda t, a, b, c: a - b / (c + t)
    try:
        p, _ = curve_fit(f, g, d, p0=[d[-1], 0.0, 10.0],
                         bounds=([0, -1e6, 0], [10, 1e6, 1e6]), maxfev=20000)
        return {'a': float(p[0]), 'b': float(p[1]), 'c': float(p[2])}
    except Exception:
        return {'a': float('nan'), 'b': float('nan'), 'c': float('nan')}


def spectral_and_audit(name: str, A: sp.csr_array, seed: int, n_probes: int = 8,
                       audit: bool = True) -> Dict[str, object]:
    t0 = time.time()
    r = sf.measure_adjacency(name, A, seed, n_probes)
    fit = saturation_fit(r['grid'], r['D'])
    out = {'graph': name, 'N': A.shape[0], 'a': fit['a'],
           'D_first': float(r['D'][0]) if len(r['D']) else float('nan'),
           'D_last': float(r['D'][-1]) if len(r['D']) else float('nan'),
           't_last': float(r['grid'][-1]) if len(r['grid']) else float('nan'),
           'spectral_class': r.get('verdict', r.get('class', '')),
           'D_curve': ' '.join(f"{t:.0f}:{d:.2f}" for t, d in zip(r['grid'], r['D']))}
    if audit:
        au = audit_adjacency(name, A, 60, seed)
        out.update({'audit': au['verdict'],
                    'audit_d': au['table'][-1][1] if au['table'] else float('nan'),
                    'audit_drift': au['drift']})
    print(f"{name}: plateau a = {out['a']:.3f}; D(t) {out['D_curve']}; "
          f"audit {out.get('audit', '-')} d {out.get('audit_d', float('nan')):.2f} "
          f"[{time.time() - t0:.0f} s]", flush=True)
    return out


def run_spectral(seed: int = 0) -> List[Dict]:
    rng = np.random.default_rng(seed)
    rows = []
    rows.append(spectral_and_audit('torus2 300^2', torus(300, 2), seed))
    rows.append(spectral_and_audit('torus3 45^3', torus(45, 3), seed))
    gate = (1.85 <= rows[0]['a'] <= 2.15 and 2.7 <= rows[1]['a'] <= 3.3 and
            rows[0]['audit'] == 'DIMENSION DEFINED' and abs(rows[0]['audit_d'] - 2) <= 0.15
            and rows[1]['audit'] == 'DIMENSION DEFINED' and abs(rows[1]['audit_d'] - 3) <= 0.15)
    print(f"GATE: {'PASS' if gate else 'FAIL'}", flush=True)
    tree = spectral_and_audit('random tree 200k', prufer_tree(200_000, rng), seed)
    rows.append(tree)
    rows.append(spectral_and_audit('UST torus 300^2', wilson_ust(torus(300, 2), rng), seed))
    rows.append(spectral_and_audit('sheet b3 512k', load_csr(SHEET_512K), seed))
    rows.append(spectral_and_audit('grown cap6 200k', topology(200_000, 'grown', seed), seed))
    rows.append(spectral_and_audit('sheet b2 s308 4.19M', load_csr(SHEET_308), seed,
                                   n_probes=4, audit=False))
    if not gate:
        verdict = 'VOID (gate failed)'
    elif tree['audit'].startswith('EXPONENTIAL'):
        verdict = 'AUDIT REJECTS TREES'
    elif tree['audit'] in ('DIMENSION DEFINED', 'MIXED') and \
            1.75 <= tree['audit_d'] <= 2.25 and 1.15 <= tree['a'] <= 1.5:
        verdict = 'AUDIT SEPARATES CRITICAL TREES'
    else:
        verdict = 'see table'
    print(f"\nW3-a: {verdict}")
    return rows


# --------------------------------------------------------------------------
# W3-b gap count
# --------------------------------------------------------------------------

def gap_fraction(A: sp.csr_array) -> Dict[str, float]:
    deg = np.asarray(A.sum(axis=1)).ravel()
    kbar = float(deg.mean())
    inv = 1.0 / np.sqrt(deg)
    M = (A.toarray() * inv[:, None]) * inv[None, :]
    lam = 1.0 - np.linalg.eigvalsh(M)
    lam.sort()
    edge = 1.0 - 2.0 * np.sqrt(max(kbar - 1.0, 0.0)) / kbar
    f = float(np.mean((lam > 1e-9) & (lam < edge)))
    return {'kbar': kbar, 'edge': edge, 'f': f, 'lam2': float(lam[1]),
            'n_zero': int((lam < 1e-9).sum())}


def run_gap(seed: int = 0) -> List[Dict]:
    rng = np.random.default_rng(seed)
    graphs = [('random 4-regular', random_regular(5000, 4, seed)),
              ('triangular', topology(5000, 'triangular', seed)),
              ('sheet', topology(5000, 'sheet', seed)),
              ('grown cap6', topology(5000, 'grown', seed)),
              ('random tree', prufer_tree(5000, rng))]
    rows = []
    for name, A in graphs:
        t0 = time.time()
        g = gap_fraction(A)
        rows.append({'graph': name, 'N': A.shape[0], **g})
        print(f"{name:>17}: N {A.shape[0]}  kbar {g['kbar']:.2f}  KM edge {g['edge']:.3f}  "
              f"f = {g['f']:.4f}  lambda_2 {g['lam2']:.5f}  [{time.time() - t0:.0f} s]",
              flush=True)
    f = {r['graph']: r['f'] for r in rows}
    gate = f['random 4-regular'] <= 0.005
    sep = f['triangular'] >= 0.05 and f['sheet'] >= 0.05 and f['grown cap6'] <= 0.02
    print(f"\nW3-b: gate {'PASS' if gate else 'FAIL'}; "
          f"{'GAP COUNT SEPARATES' if gate and sep else 'see table'}")
    return rows


# --------------------------------------------------------------------------
# W3-c effective resistance
# --------------------------------------------------------------------------

def effective_resistance(A: sp.csr_array, s: int, t: int) -> float:
    L = sp.diags_array(np.asarray(A.sum(axis=1)).ravel()) - A
    Lp = np.linalg.pinv(L.toarray(), hermitian=True)
    return float(Lp[s, s] + Lp[t, t] - 2 * Lp[s, t])


def run_reff(seeds: Sequence[int] = range(10), n: int = 1500) -> List[Dict]:
    from shortcuts import far_node
    from shortcut_walkers import node_near
    banked = {'triangular': 1.4, 'sheet': 1.3, 'grown': 6.7}
    rows = []
    for topo in ('triangular', 'sheet', 'grown'):
        for seed in seeds:
            # Exactly the E3 selection sequence of shortcut_walkers.main.
            random.seed(seed)
            np.random.seed(seed)
            base = create_initial_graph(n, topology=topo, k=6, seed=seed)
            nodes = list(base.nodes())
            source = nodes[np.random.randint(len(nodes))]
            target, dist_st = far_node(base, source)
            a = node_near(base, source, 2)
            b = node_near(base, target, 2)
            if a is None or b is None or base.has_edge(a, b):
                continue
            idx = {u: i for i, u in enumerate(nodes)}
            A = sp.csr_array(nx.to_scipy_sparse_array(base, nodelist=nodes, weight=None,
                                                      format='csr', dtype=float))
            r0 = effective_resistance(A, idx[source], idx[target])
            G2 = base.copy()
            G2.add_edge(a, b)
            A2 = sp.csr_array(nx.to_scipy_sparse_array(G2, nodelist=nodes, weight=None,
                                                       format='csr', dtype=float))
            r1 = effective_resistance(A2, idx[source], idx[target])
            rows.append({'substrate': topo, 'seed': seed, 'dist': dist_st,
                         'R_none': r0, 'R_offset': r1, 'rho': r0 / r1})
        rs = [r['rho'] for r in rows if r['substrate'] == topo]
        print(f"{topo:>10}: median R_none/R_offset = {np.median(rs):.2f} "
              f"(range {min(rs):.2f}-{max(rs):.2f}, n {len(rs)}); banked t_hit gain "
              f"{banked[topo]}x", flush=True)
    med = {t: np.median([r['rho'] for r in rows if r['substrate'] == t]) for t in banked}
    ok = (med['grown'] >= 3 and med['triangular'] <= 1.3 and med['sheet'] <= 1.3 and
          all(0.5 <= med[t] / banked[t] <= 2.0 for t in banked))
    print(f"\nW3-c: {'RESISTANCE EXPLAINS THE CONTRAST' if ok else 'see table'}")
    return rows


# --------------------------------------------------------------------------
# W3-e random triangulation
# --------------------------------------------------------------------------

class SphereTriangulation:
    def __init__(self, n: int, rng: np.random.Generator):
        self.rng = rng
        self.adj: List[set] = [set() for _ in range(4)]
        self.faces: set = set()
        self.face_list: List[Tuple[int, int, int]] = []
        for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            self._add_face(f)
        for a in range(4):
            for b in range(4):
                if a != b:
                    self.adj[a].add(b)
        while len(self.adj) < n:                      # Apollonian start
            i = int(rng.integers(len(self.face_list)))
            a, b, c = self.face_list[i]
            if (a, b, c) not in self.faces:
                self.face_list[i] = self.face_list[-1]
                self.face_list.pop()
                continue
            v = len(self.adj)
            self.adj.append({a, b, c})
            for x in (a, b, c):
                self.adj[x].add(v)
            self.faces.discard((a, b, c))
            for f in ((a, b, v), (a, c, v), (b, c, v)):
                self._add_face(f)
        self.edges = [(u, v) for u in range(n) for v in self.adj[u] if u < v]

    def _add_face(self, f) -> None:
        t = tuple(sorted(f))
        self.faces.add(t)
        self.face_list.append(t)

    def flip_sweeps(self, sweeps: float) -> int:
        E = len(self.edges)
        acc = 0
        rint = self.rng.integers
        for _ in range(int(sweeps * E)):
            k = int(rint(E))
            a, b = self.edges[k]
            if len(self.adj[a]) <= 3 or len(self.adj[b]) <= 3:
                continue
            opp = [c for c in self.adj[a] & self.adj[b]
                   if tuple(sorted((a, b, c))) in self.faces]
            if len(opp) != 2:
                continue
            c, d = opp
            if d in self.adj[c]:
                continue
            for f in ((a, b, c), (a, b, d)):
                self.faces.discard(tuple(sorted(f)))
            for f in ((a, c, d), (b, c, d)):
                self.faces.add(tuple(sorted(f)))
            self.adj[a].discard(b)
            self.adj[b].discard(a)
            self.adj[c].add(d)
            self.adj[d].add(c)
            self.edges[k] = (min(c, d), max(c, d))
            acc += 1
        return acc

    def csr(self) -> sp.csr_array:
        e = np.array(self.edges)
        n = len(self.adj)
        A = sp.coo_array((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        return (A + A.T).tocsr()


def hull_profile(A: sp.csr_array, src: int, radii: Sequence[int]
                 ) -> List[Tuple[int, int, int]]:
    """(r, hull volume, hull boundary) for each r: the ball's complement is
    split into components; all but the largest join the ball; the boundary
    is the hulled ball's nodes adjacent to the largest component."""
    n = A.shape[0]
    d = dijkstra(A, indices=src, unweighted=True, limit=max(radii) + 1.5)
    out = []
    for r in radii:
        ball = d <= r
        comp_nodes = np.nonzero(~ball)[0]
        if len(comp_nodes) == 0:
            break
        sub = A[comp_nodes][:, comp_nodes]
        k, lab = connected_components(sub, directed=False)
        big = np.bincount(lab).argmax()
        outside = np.zeros(n, dtype=bool)
        outside[comp_nodes[lab == big]] = True
        hull = ~outside
        touch = (A @ outside.astype(float)) > 0
        H = int((hull & touch).sum())
        out.append((r, int(hull.sum()), H))
    return out


def exponents(A: sp.csr_array, n_src: int, rng: np.random.Generator
              ) -> Dict[str, float]:
    n = A.shape[0]
    src = rng.choice(n, n_src, replace=False)
    # Usable radii: hulled balls below 0.1 N (the estimator's saturation fraction).
    d0 = dijkstra(A, indices=int(src[0]), unweighted=True)
    rmax = 1
    while (d0 <= rmax + 1).sum() < 0.1 * n:
        rmax += 1
    radii = list(range(2, rmax + 1))
    if len(radii) > 18:
        radii = sorted(set(np.round(np.geomspace(2, rmax, 18)).astype(int).tolist()))
    Vs, Hs = [], []
    for s in src:
        p = hull_profile(A, int(s), radii)
        if len(p) == len(radii):
            Vs.append([x[1] for x in p])
            Hs.append([x[2] for x in p])
    V = np.exp(np.log(np.array(Vs, float)).mean(axis=0))
    H = np.exp(np.log(np.maximum(np.array(Hs, float), 1)).mean(axis=0))
    r = np.array(radii, float)
    k = len(r)
    mid = slice(k // 3, max(2 * k // 3, k // 3 + 3))
    dV = float(np.polyfit(np.log(r[mid]), np.log(V[mid]), 1)[0])
    sHV = float(np.polyfit(np.log(V[mid]), np.log(H[mid]), 1)[0])
    return {'r_range': f"{int(r[mid][0])}-{int(r[mid][-1])}", 'd_V': dV, 's_HV': sHV,
            'sources': len(Vs)}


def run_triang(seed: int = 0, n: int = 20000, sweeps: float = 150) -> List[Dict]:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    T = SphereTriangulation(n, rng)
    acc = T.flip_sweeps(sweeps)
    A = T.csr()
    degs = np.diff(A.indptr)
    print(f"triangulation: N {n}, {acc} accepted flips in {sweeps} sweeps, degree "
          f"mean {degs.mean():.3f} max {degs.max()} [{time.time() - t0:.0f} s]", flush=True)
    rows = []
    for name, M, ns in (('random triangulation 20k', A, 40),
                        ('sheet b3 512k', load_csr(SHEET_512K), 20)):
        t1 = time.time()
        e = exponents(M, ns, rng)
        rows.append({'graph': name, **e})
        print(f"{name}: radii {e['r_range']}  d_V {e['d_V']:.2f}  s_HV {e['s_HV']:.2f}  "
              f"({e['sources']} sources) [{time.time() - t1:.0f} s]", flush=True)
    s = [r['s_HV'] for r in rows]
    dv = [r['d_V'] for r in rows]
    if all(0.35 <= x <= 0.65 for x in s) and abs(dv[0] - dv[1]) >= 0.8:
        v = 'PERIMETER CRITERION BLIND'
    elif abs(s[0] - s[1]) > 0.2:
        v = 'PERIMETER DISCRIMINATES'
    else:
        v = 'see table'
    print(f"\nW3-e: {v}")
    return rows


# --------------------------------------------------------------------------
# W3-f box covering
# --------------------------------------------------------------------------

def ball_cover(nb: List[List[int]], rB: int, rng: np.random.Generator) -> int:
    n = len(nb)
    covered = np.zeros(n, dtype=bool)
    count = 0
    for s in rng.permutation(n).tolist():
        if covered[s]:
            continue
        count += 1
        covered[s] = True
        frontier = [s]
        seen = {s}
        for _ in range(rB):
            nxt = []
            for x in frontier:
                for y in nb[x]:
                    if y not in seen:
                        seen.add(y)
                        nxt.append(y)
                        covered[y] = True
            frontier = nxt
    return count


def run_boxes(seed: int = 0) -> List[Dict]:
    rng = np.random.default_rng(seed)
    radii = [1, 2, 3, 4, 6, 8, 12, 16, 24]
    rows = []
    for name, A in (('sheet b3 512k', load_csr(SHEET_512K)),
                    ('grown cap6 200k', topology(200_000, 'grown', seed))):
        nb = neighbour_lists(A)
        t0 = time.time()
        NB = np.array([[ball_cover(nb, r, rng) for _ in range(5)] for r in radii], float)
        m = NB.mean(axis=1)
        lr, lN, rr = np.log(radii), np.log(m), np.array(radii, float)
        cp = np.polyfit(lr, lN, 1)
        rss_p = float(((lN - np.polyval(cp, lr)) ** 2).sum())
        ce = np.polyfit(rr, lN, 1)
        rss_e = float(((lN - np.polyval(ce, rr)) ** 2).sum())
        rows.append({'graph': name, 'd_B': -cp[0], 'rss_power': rss_p, 'rss_exp': rss_e,
                     'N_B': ' '.join(f"{r}:{x:.0f}" for r, x in zip(radii, m))})
        print(f"{name}: N_B {rows[-1]['N_B']}; power-law d_B {-cp[0]:.2f} (RSS {rss_p:.3f})"
              f" vs exponential (RSS {rss_e:.3f}) [{time.time() - t0:.0f} s]", flush=True)
    sh, gr = rows
    print(f"\nW3-f: sheet d_B {sh['d_B']:.2f} (prediction [1.8, 2.2]); grown "
          f"{'EXPONENTIAL' if gr['rss_exp'] < gr['rss_power'] else 'FINITE d_B'}")
    return rows


# --------------------------------------------------------------------------
# W3-d Newman-Watts collapse of the portal threshold
# --------------------------------------------------------------------------

NW_N = (2000, 5000, 20000)
NW_R = (None, 8, 14)            # None = automatic (10 at every N here)
NW_COUNTS = [0, 2, 5, 10, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320]


def _nw_job(spec):
    from shortcut_tolerance import measure_tolerance
    n, r0, seed = spec
    # n_perm only feeds Moran's I; the drift that defines k* does not use it.
    rows = measure_tolerance(n, 'triangular', 6, NW_COUNTS, seed, n_perm=19,
                             max_radius=r0)
    for r in rows:
        r['r0_setting'] = 'auto' if r0 is None else r0
    return rows


def run_nw(seed: int = 0, jobs: int = 3) -> List[Dict]:
    from concurrent.futures import ProcessPoolExecutor
    specs = [(n, r0, s) for n in sorted(NW_N, reverse=True) for r0 in NW_R
             for s in (0, 1, 2)]
    rows: List[Dict] = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for res in ex.map(_nw_job, specs):
            rows += res
            r = res[0]
            print(f"done N {r['n_nodes']} r0 {r['max_radius']} seed {r['seed']}: drift by k "
                  + ' '.join(f"{x['k']}:{x['median_drift']:.2f}" for x in res), flush=True)
    cells = []
    for n in NW_N:
        for r0 in NW_R:
            sel = [r for r in rows if r['n_nodes'] == n and
                   r['r0_setting'] == ('auto' if r0 is None else r0)]
            R = sel[0]['max_radius']
            kstar = None
            for k in NW_COUNTS:
                d = [r['median_drift'] for r in sel if r['k'] == k]
                if np.isfinite(np.median(d)) and np.median(d) > 0.10:
                    kstar = k
                    break
            cells.append((n, R, kstar))
            print(f"N {n:>6} r0 {R:>3}: k* = {kstar}   N/r0^2 = {n / R ** 2:.1f}")
    fin = [(n, R, k) for n, R, k in cells if k]
    if len(fin) >= 3:
        x = np.log([n / R ** 2 for n, R, k in fin])
        y = np.log([k for n, R, k in fin])
        sl, ic = np.polyfit(x, y, 1)
        scat = float(np.std(y - (sl * x + ic), ddof=2)) if len(fin) > 2 else float('nan')
        if 0.7 <= sl <= 1.3 and scat < 0.35:
            v = 'NEWMAN-WATTS COLLAPSE'
        elif sl < 0.4 or sl > 1.6:
            v = 'NO COLLAPSE'
        else:
            v = 'UNDECIDED'
        print(f"
slope of ln k* on ln(N/r0^2) = {sl:.2f}, scatter {scat:.2f} over "
              f"{len(fin)} cells")
        print(f"W3-d: {v}")
    banked = [k for n, R, k in cells if n == 5000 and R == 10]
    print(f"banked-cell check (N 5000, automatic r0 = 10): k* = {banked} (banked 20)")
    return rows


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _validate() -> bool:
    ok = True
    rng = np.random.default_rng(1)
    T = prufer_tree(2000, rng)
    good = T.nnz == 2 * 1999 and connected_components(T)[0] == 1
    print(f"[{'PASS' if good else 'FAIL'}] Pruefer tree: 1999 edges, connected")
    ok &= good
    U = wilson_ust(torus(30, 2), rng)
    good = U.nnz == 2 * 899 and connected_components(U)[0] == 1
    print(f"[{'PASS' if good else 'FAIL'}] Wilson UST of a 30x30 torus: 899 edges, connected")
    ok &= good
    # Effective resistance on a path of 5 nodes between ends = 4; on a cycle of
    # 6 between opposite nodes = 3*3/6 = 1.5.
    P = nx_csr(nx.path_graph(5))
    C = nx_csr(nx.cycle_graph(6))
    r1, r2 = effective_resistance(P, 0, 4), effective_resistance(C, 0, 3)
    good = abs(r1 - 4) < 1e-9 and abs(r2 - 1.5) < 1e-9
    print(f"[{'PASS' if good else 'FAIL'}] effective resistance: path {r1:.3f} (4), "
          f"cycle {r2:.3f} (1.5)")
    ok &= good
    g = gap_fraction(random_regular(1000, 4, 0))
    good = g['f'] <= 0.01
    print(f"[{'PASS' if good else 'FAIL'}] random 4-regular N 1000: fraction below the "
          f"Kesten-McKay edge {g['f']:.4f}")
    ok &= good
    S = SphereTriangulation(500, rng)
    S.flip_sweeps(20)
    A = S.csr()
    good = (A.nnz // 2 == 3 * 500 - 6 and len(S.faces) == 2 * 500 - 4 and
            np.diff(A.indptr).min() >= 3)
    print(f"[{'PASS' if good else 'FAIL'}] sphere triangulation after flips: E = 3N - 6 "
          f"({A.nnz // 2}), F = 2N - 4 ({len(S.faces)}), min degree "
          f"{np.diff(A.indptr).min()}")
    ok &= good
    tor = torus(60, 2)
    p = hull_profile(tor, 0, [5, 10])
    good = p[0][1] == 61 and p[1][1] == 221     # L1 balls on Z^2: 2r^2 + 2r + 1
    print(f"[{'PASS' if good else 'FAIL'}] hull volume on a torus: {p[0][1]} (61), "
          f"{p[1][1]} (221)")
    ok &= good
    nbl = neighbour_lists(tor)
    c1 = ball_cover(nbl, 0, rng)
    good = c1 == 3600
    print(f"[{'PASS' if good else 'FAIL'}] ball cover at radius 0 counts every node "
          f"({c1})")
    ok &= good
    fit = saturation_fit(np.array([16, 32, 64, 128, 256.]),
                         2 - 3 / (5 + np.array([16, 32, 64, 128, 256.])))
    good = abs(fit['a'] - 2) < 1e-3
    print(f"[{'PASS' if good else 'FAIL'}] saturation fit recovers a = {fit['a']:.4f} (2)")
    ok &= good
    print('\nVALIDATION', 'PASSED' if ok else 'FAILED')
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_mutually_exclusive_group(required=True)
    for f in ('validate', 'spectral', 'gap', 'reff', 'triang', 'boxes', 'nw'):
        m.add_argument(f'--{f}', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.validate:
        raise SystemExit(0 if _validate() else 1)
    fn = {'spectral': run_spectral, 'gap': run_gap, 'reff': run_reff,
          'triang': run_triang, 'boxes': run_boxes, 'nw': run_nw}
    for k, f in fn.items():
        if getattr(args, k):
            rows = f() if k == 'reff' else f(args.seed)
            stamp = time.strftime('%Y%m%d_%H%M%S')
            out = f"results/instruments_{k}_{stamp}.csv"
            with open(out, 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"saved: {out}")


if __name__ == '__main__':
    main()
