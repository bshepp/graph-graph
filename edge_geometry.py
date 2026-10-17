"""
Edge geometry of saved `sheet` graphs: the ruffle, the fjords, the roughness.

Pre-registration: docs/superpowers/specs/2026-10-17-wave2-preregistration.md
(W2-a, W2-b, W2-c, W2-d).

  W2-a  cone excess: shells S(r) around node 0 against the compact seeds'
        mean shell, X(r) = S/S_ref - 1, regressed on D(r)/3, D(r) = dead
        corners within hop distance r of node 0.
  W2-b  fjords: a neck is a pair of boundary nodes at hop distance <= 3 that
        are >= 100 apart along the boundary cycle; necks whose two ends are
        each within 10 boundary positions of another neck's ends are one
        mouth (positions along the boundary bound hop distance from above, so
        this merges no more than "within 10 hops" would); the shorter arc
        is the fjord. Depth = largest hop distance from the mouth to the arc.
  W2-c  local width w(l) of h(s) = d(0, s) - median along the boundary, and
        the binned spectral slope with k = 6, 12, 18 removed.
  W2-d  descriptive: spread of B / sqrt(N) across seeds by octave.

Usage:
    python edge_geometry.py --validate
    python edge_geometry.py results/sheet_cross_b2_s3*_N4194304.npz
"""

import argparse
import csv
import glob
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from sheet_growth import load_csr, grow_sheet, to_csr
from curvature_census import edge_triangles, census, seed_of, first_sizes

NECK_HOPS = 3
NECK_ARC = 100
MERGE = 10
R_MIN = 50
R_FRAC = 0.7
SPEC_DROP = (6, 12, 18)
K_MIN = 4
LAMBDA_MIN = 40
BINS_PER_DECADE = 8
MIN_MODES = 8
COMPACT = (305, 309, 310)


# --------------------------------------------------------------------------
# Boundary cycle
# --------------------------------------------------------------------------

def boundary_cycle(A: sp.csr_array, T: Optional[sp.csr_array] = None
                   ) -> np.ndarray:
    """The boundary nodes in cyclic order (the boundary is one simple cycle)."""
    if T is None:
        T = edge_triangles(A)
    coo = T.tocoo()
    m = (coo.data == 1) & (coo.row < coo.col)
    u, v = coo.row[m], coo.col[m]
    nb: Dict[int, List[int]] = {}
    for a, b in zip(u.tolist(), v.tolist()):
        nb.setdefault(a, []).append(b)
        nb.setdefault(b, []).append(a)
    if any(len(x) != 2 for x in nb.values()):
        raise ValueError('boundary is not a simple cycle')
    start = min(nb)
    cyc = [start]
    prev, cur = start, nb[start][0]
    while cur != start:
        cyc.append(cur)
        a, b = nb[cur]
        prev, cur = cur, (b if a == prev else a)
    if len(cyc) != len(nb):
        raise ValueError('boundary has more than one cycle')
    return np.array(cyc, dtype=np.int64)


# --------------------------------------------------------------------------
# W2-b fjords
# --------------------------------------------------------------------------

def _ball(A: sp.csr_array, s: int, hops: int) -> Dict[int, int]:
    dist = {s: 0}
    frontier = [s]
    for h in range(1, hops + 1):
        nxt = []
        for x in frontier:
            for y in A.indices[A.indptr[x]:A.indptr[x + 1]].tolist():
                if y not in dist:
                    dist[y] = h
                    nxt.append(y)
        frontier = nxt
    return dist


def find_fjords(A: sp.csr_array, cyc: np.ndarray, dead: Sequence[int]
                ) -> List[Dict[str, object]]:
    B = len(cyc)
    pos = {int(x): i for i, x in enumerate(cyc.tolist())}
    necks = []
    for i, u in enumerate(cyc.tolist()):
        for v, h in _ball(A, u, NECK_HOPS).items():
            j = pos.get(v)
            if j is None or j <= i:
                continue
            arc = min(j - i, B - (j - i))
            if arc >= NECK_ARC:
                necks.append((i, j, h))
    if not necks:
        return []
    # Union-find on necks whose ends are each within MERGE positions.
    parent = list(range(len(necks)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def cd(a, b):
        d = abs(a - b)
        return min(d, B - d)

    # Two necks are one mouth when their ends match within MERGE positions,
    # in either order and across the cycle's wrap point. Necks are bucketed
    # by (position // MERGE) under both orderings so only nearby buckets are
    # compared.
    nb = (B + MERGE - 1) // MERGE
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for k, (i, j, _) in enumerate(necks):
        buckets.setdefault((i // MERGE, j // MERGE), []).append(k)
        buckets.setdefault((j // MERGE, i // MERGE), []).append(k)
    for a, (i, j, _) in enumerate(necks):
        bi, bj = i // MERGE, j // MERGE
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for b in buckets.get(((bi + di) % nb, (bj + dj) % nb), ()):
                    if b <= a:
                        continue
                    i2, j2, _ = necks[b]
                    if (cd(i, i2) <= MERGE and cd(j, j2) <= MERGE) or \
                            (cd(i, j2) <= MERGE and cd(j, i2) <= MERGE):
                        parent[find(a)] = find(b)
    groups: Dict[int, List[int]] = {}
    for k in range(len(necks)):
        groups.setdefault(find(k), []).append(k)
    dead_pos = np.array(sorted(pos[d] for d in dead if d in pos), dtype=np.int64)
    out = []
    for ks in groups.values():
        # Representative: the neck cutting off the longest (shorter-side) arc.
        def arc_of(k):
            i, j, _ = necks[k]
            return min(j - i, B - (j - i))
        k = max(ks, key=arc_of)
        i, j, h = necks[k]
        if j - i <= B - (j - i):
            arc_idx = np.arange(i, j + 1)
        else:
            arc_idx = np.concatenate([np.arange(j, B), np.arange(0, i + 1)])
        arc_nodes = cyc[arc_idx]
        d = dijkstra(A, indices=[int(cyc[i]), int(cyc[j])], unweighted=True,
                     min_only=True)
        depth = float(d[arc_nodes].max())
        nd = int(np.isin(arc_idx, dead_pos).sum())
        out.append({'i': int(i), 'j': int(j),
                    'mouth_hops': int(min(necks[k2][2] for k2 in ks)),
                    'arc': int(len(arc_idx)), 'depth': depth, 'dead': nd})
    return out


# --------------------------------------------------------------------------
# W2-c roughness
# --------------------------------------------------------------------------

def local_width(h: np.ndarray, ls: Sequence[int]) -> np.ndarray:
    """Mean over non-overlapping windows of the within-window std of h."""
    out = []
    for l in ls:
        n = len(h) // l
        w = h[:n * l].reshape(n, l)
        out.append(float(w.std(axis=1).mean()))
    return np.array(out)


def alpha_loc(h: np.ndarray) -> Tuple[float, int]:
    B = len(h)
    hi = max(B // 20, 16)
    ls = np.unique(np.round(np.geomspace(8, hi, 12)).astype(int))
    ls = ls[ls <= hi]
    if len(ls) < 4:
        return float('nan'), len(ls)
    w = local_width(h, ls)
    ok = w > 0
    slope = np.polyfit(np.log(ls[ok]), np.log(w[ok]), 1)[0]
    return float(slope), len(ls)


def spectral_slope(h: np.ndarray) -> Tuple[float, float]:
    """Two-pass binned power-law slope with k = 6, 12, 18 removed; returns
    (slope, alpha_s) with S(k) ~ k^-(1 + 2 alpha_s)."""
    L = len(h)
    P = np.abs(np.fft.rfft(h - h.mean())) ** 2 / L
    k = np.arange(len(P))
    sel = (k >= K_MIN) & (k <= L // LAMBDA_MIN) & ~np.isin(k, SPEC_DROP)
    ks, Ps = k[sel], P[sel]
    if len(ks) < 2 * MIN_MODES:
        return float('nan'), float('nan')
    step = 10 ** (1 / BINS_PER_DECADE)
    edges = [ks[0]]
    while edges[-1] <= ks[-1]:
        edges.append(max(edges[-1] * step, edges[-1] + MIN_MODES))
    bk, bP = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ks >= lo) & (ks < hi)
        if m.sum() >= MIN_MODES:
            bk.append(np.exp(np.mean(np.log(ks[m]))))
            bP.append(Ps[m].mean())
    if len(bk) < 4:
        return float('nan'), float('nan')
    lk, lP = np.log10(bk), np.log10(np.array(bP) + 1e-300)
    slope, icpt = np.polyfit(lk, lP, 1)
    keep = np.ones(len(lk), dtype=bool)
    if len(lk) >= 6:
        keep[int(np.argmax(lP - (slope * lk + icpt)))] = False
    slope = float(np.polyfit(lk[keep], lP[keep], 1)[0])
    return slope, (-slope - 1) / 2


# --------------------------------------------------------------------------
# One graph (or prefix)
# --------------------------------------------------------------------------

def edge_state(A: sp.csr_array) -> Dict[str, object]:
    T = edge_triangles(A)
    cz = census(A)
    cyc = boundary_cycle(A, T)
    d0 = dijkstra(A, indices=0, unweighted=True)
    h = d0[cyc] - np.median(d0[cyc])
    a_loc, nl = alpha_loc(h)
    s_slope, a_s = spectral_slope(h)
    fj = find_fjords(A, cyc, cz['dead_ids'])
    return {'N': A.shape[0], 'B': len(cyc), 'dead_ids': cz['dead_ids'],
            'd0': d0, 'alpha_loc': a_loc, 'spec_slope': s_slope, 'alpha_s': a_s,
            'fjords': fj}


def shells(d0: np.ndarray) -> np.ndarray:
    return np.bincount(d0[np.isfinite(d0)].astype(int))


def run(paths: Sequence[str], first_csv: str, out_prefix: str) -> None:
    firsts = first_sizes(first_csv)
    per: Dict[int, Dict] = {}
    rows_fj, rows_r = [], []
    for path in paths:
        t0 = time.time()
        A = load_csr(path)
        s = seed_of(path)
        st = edge_state(A)
        ecc = int(np.nanmax(st['d0'][np.isfinite(st['d0'])]))
        D = np.sort(st['d0'][st['dead_ids']]) if st['dead_ids'] else np.array([])
        per[s] = {'S': shells(st['d0']), 'ecc': ecc, 'dead_d0': D,
                  'crossed': firsts.get(s, -1) > 0}
        rows_r.append({'seed': s, 'state': 'final', 'N': st['N'], 'B': st['B'],
                       'alpha_loc': st['alpha_loc'], 'alpha_s': st['alpha_s'],
                       'spec_slope': st['spec_slope'],
                       'compact_edge': int(s in COMPACT)})
        for f in st['fjords']:
            rows_fj.append({'seed': s, 'state': 'final', 'N': st['N'], **f})
        nfj = len(st['fjords'])
        nwith = sum(1 for f in st['fjords'] if f['dead'] > 0)
        print(f"seed {s}: B {st['B']}  fjords {nfj} (with a dead corner {nwith})  "
              f"alpha_loc {st['alpha_loc']:.2f}  alpha_s {st['alpha_s']:.2f}  "
              f"[{time.time() - t0:.0f} s]", flush=True)
        # Prefix octaves before the first dead corner (compact edges), and the
        # octave just after it.
        nf = firsts.get(s, -1)
        if nf > 0:
            pre = [2 ** k for k in range(15, 23) if 2 ** k < nf]
            marks = pre[-1:] + [m for m in (2 ** k for k in range(15, 23))
                                if nf <= m < A.shape[0]][:1]
            for m in marks:
                P = A[:m, :m].tocsr()
                sp_ = edge_state(P)
                tag = 'pre' if m < nf else 'post'
                rows_r.append({'seed': s, 'state': tag, 'N': m, 'B': sp_['B'],
                               'alpha_loc': sp_['alpha_loc'],
                               'alpha_s': sp_['alpha_s'],
                               'spec_slope': sp_['spec_slope'],
                               'compact_edge': int(tag == 'pre')})
                for f in sp_['fjords']:
                    rows_fj.append({'seed': s, 'state': tag, 'N': m, **f})
                print(f"    prefix {m} ({tag}): B {sp_['B']} fjords "
                      f"{len(sp_['fjords'])}  alpha_loc {sp_['alpha_loc']:.2f}  "
                      f"alpha_s {sp_['alpha_s']:.2f}", flush=True)
        del A, st
    for name, rows in (('fjords', rows_fj), ('rough', rows_r)):
        if rows:
            with open(f"{out_prefix}_{name}.csv", 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    report(per, rows_fj, rows_r)


def report(per: Dict[int, Dict], rows_fj: List[Dict], rows_r: List[Dict]) -> None:
    # ---- W2-a cone excess
    comp = [s for s in COMPACT if s in per]
    print("\nW2-a cone excess")
    def sref(excl=None):
        use = [per[c]['S'] for c in comp if c != excl]
        L = min(len(x) for x in use)
        return np.mean([x[:L] for x in use], axis=0)
    for c in comp:
        ref = sref(excl=c)
        S = per[c]['S']
        rmax = int(R_FRAC * per[c]['ecc'])
        r = np.arange(R_MIN, min(rmax, len(ref), len(S)))
        X = S[r] / ref[r] - 1
        print(f"  compact {c} (leave-one-out): X in [{X.min():+.3f}, {X.max():+.3f}]")
    ref = sref()
    gs, early = [], 0
    for s in sorted(per):
        if not per[s]['crossed']:
            continue
        S = per[s]['S']
        rmax = int(R_FRAC * per[s]['ecc'])
        r = np.arange(R_MIN, min(rmax, len(ref), len(S)))
        X = S[r] / ref[r] - 1
        Dr = np.searchsorted(per[s]['dead_d0'], r, side='right')
        x = Dr / 3.0
        g = float((X * x).sum() / (x * x).sum()) if (x > 0).any() else float('nan')
        rises = bool(((Dr == 0) & (X > 0.05)).any())
        early += rises
        gs.append(g)
        r_dc = int(per[s]['dead_d0'][0]) if len(per[s]['dead_d0']) else -1
        print(f"  seed {s}: nearest dead corner at r = {r_dc}, radii {r[0]}-{r[-1]}, "
              f"X max {X.max():+.2f}, D max {Dr.max()}, slope g = {g:.3f}"
              f"{'  (X rises before any dead corner)' if rises else ''}")
    gs = np.array(gs)
    med = float(np.nanmedian(gs))
    if (gs > 0).all() and 0.3 <= med <= 1.5:
        v = 'CONE EXCESS'
    elif med < 0.1 or early >= 3:
        v = 'NO CONE EXCESS'
    else:
        v = 'UNDECIDED'
    print(f"W2-a: {v}  (g > 0 on {(gs > 0).sum()}/{len(gs)}, median g {med:.3f}, "
          f"early rise on {early})")

    # ---- W2-b fjords
    print("\nW2-b fjords")
    fin = [f for f in rows_fj if f['state'] == 'final' and f['seed'] not in COMPACT]
    with_dead = sum(1 for f in fin if f['dead'] > 0)
    frac = with_dead / len(fin) if fin else float('nan')
    comp_fj = sum(1 for f in rows_fj if f['state'] == 'final' and f['seed'] in COMPACT)
    pre_fj = sum(1 for f in rows_fj if f['state'] == 'pre')
    if fin and frac >= 0.9 and comp_fj == 0 and pre_fj == 0:
        v = 'FORCED FJORDS'
    elif fin and frac < 0.6:
        v = 'FJORDS NOT TIED TO DEAD CORNERS'
    else:
        v = 'UNDECIDED'
    print(f"  crossed final graphs: {len(fin)} fjords, {with_dead} with a dead corner "
          f"({frac:.2f}); compact final graphs: {comp_fj} fjords; last octave before "
          f"the first dead corner: {pre_fj} fjords")
    if fin:
        x = np.array([f['mouth_hops'] / max(f['depth'], 1) for f in fin])
        dep = np.array([f['depth'] for f in fin])
        print(f"  mouth/depth x: 10/50/90% = {np.percentile(x, [10, 50, 90]).round(3)}; "
              f"depth 10/50/90% = {np.percentile(dep, [10, 50, 90]).round(0)}")
        for s in sorted({f['seed'] for f in fin}):
            fs = [f for f in fin if f['seed'] == s]
            deepest = max(fs, key=lambda f: f['depth'])
            R = per[s]['ecc']
            print(f"    seed {s}: {len(fs)} fjords, deepest depth {deepest['depth']:.0f} "
                  f"(= {deepest['depth'] / R:.2f} of ecc(0)), arc {deepest['arc']}, "
                  f"dead corners in it {deepest['dead']}")
    print(f"W2-b: {v}")

    # ---- W2-c roughness
    print("\nW2-c roughness")
    ce = [r for r in rows_r if r['compact_edge']]
    re_ = [r for r in rows_r if r['state'] == 'final' and not r['compact_edge']]
    for r in ce + re_:
        kind = 'compact' if r['compact_edge'] else 'rough'
        print(f"  seed {r['seed']} {r['state']:>5} N {r['N']:>8} {kind:>7}: alpha_loc "
              f"{r['alpha_loc']:.2f}  alpha_s {r['alpha_s']:.2f}  (slope "
              f"{r['spec_slope']:.2f})")
    ok = [0.4 <= r['alpha_loc'] <= 0.6 for r in ce]
    an = [r['alpha_s'] - r['alpha_loc'] >= 0.5 for r in ce]
    v = 'KPZ-LOCAL COMPACT EDGE' if np.mean(ok) >= 0.8 else 'NOT KPZ-LOCAL'
    if v.startswith('KPZ') and np.mean(an) >= 0.8:
        v += ', ANOMALOUS'
    print(f"W2-c: {v}  (alpha_loc in [0.4, 0.6] on {sum(ok)}/{len(ce)} compact edges; "
          f"alpha_s - alpha_loc >= 0.5 on {sum(an)}/{len(ce)}); rough edges alpha_loc "
          f"{', '.join(f'{r['alpha_loc']:.2f}' for r in re_)} (band [0.5, 0.9])")


def w2d(census_csv: str) -> None:
    rows = list(csv.DictReader(open(census_csv)))
    firsts = {int(r['seed']): int(r['N_first']) for r in rows}
    print("\nW2-d B/sqrt(N) across seeds by octave (only octaves before each seed's "
          "first dead corner):")
    for k in range(12, 23):
        m = 2 ** k
        vals = [int(r['B_v']) / m ** 0.5 for r in rows
                if int(r['N']) == m and (firsts[int(r['seed'])] < 0 or
                                         m < firsts[int(r['seed'])])]
        if len(vals) >= 3:
            print(f"  N {m:>8}: n {len(vals):2d}  mean {np.mean(vals):.3f}  "
                  f"sd {np.std(vals, ddof=1):.4f}")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _chord_cycle(n: int, chords) -> sp.csr_array:
    """A cycle 0..n-1 plus, per chord (a, b), one extra node joined to a and b."""
    G = nx.cycle_graph(n)
    for k, (a, b) in enumerate(chords):
        z = n + k
        G.add_edge(a, z)
        G.add_edge(z, b)
    return sp.csr_array(nx.to_scipy_sparse_array(G, format='csr', dtype=float))


def _validate() -> bool:
    ok = True
    adj = grow_sheet(100000, 6, 3.0, 2)
    A = to_csr(adj)
    T = edge_triangles(A)
    cyc = boundary_cycle(A, T)
    cz = census(A)
    closed = all(A[int(cyc[i]), int(cyc[(i + 1) % len(cyc)])] for i in range(len(cyc)))
    good = len(cyc) == cz['B_e'] and closed
    print(f"[{'PASS' if good else 'FAIL'}] boundary cycle of a compact sheet: "
          f"{len(cyc)} nodes = B_e {cz['B_e']}, consecutive nodes adjacent and closed")
    ok &= good
    fj = find_fjords(A, cyc, cz['dead_ids'])
    good = len(fj) == 0
    print(f"[{'PASS' if good else 'FAIL'}] compact beta = 3 sheet: {len(fj)} fjords (want 0)")
    ok &= good
    # Synthetic necks: a 400-cycle with two 2-hop chords across positions
    # (0, 150) and (2, 148) -> one mouth, arc 151, depth ~75; a chord across
    # (200, 280) cuts off only 81 < 100 positions -> no fjord.
    cyc2 = np.arange(400)
    A2 = _chord_cycle(400, [(0, 150), (2, 148), (200, 280)])
    fj2 = find_fjords(A2, cyc2, [75])
    good = (len(fj2) == 1 and 150 <= fj2[0]['arc'] <= 153
            and fj2[0]['mouth_hops'] <= 3
            and 70 <= fj2[0]['depth'] <= 76 and fj2[0]['dead'] == 1)
    print(f"[{'PASS' if good else 'FAIL'}] synthetic chords: {len(fj2)} fjord(s) "
          f"{fj2} (want one: arc ~151, mouth <= 3 hops, depth ~75, 1 dead corner)")
    ok &= good
    rng = np.random.default_rng(0)
    rw = np.cumsum(rng.standard_normal(20000))
    rw -= np.linspace(0, rw[-1], len(rw))        # periodic bridge
    a, _ = alpha_loc(rw)
    sl, a_s = spectral_slope(rw)
    good = 0.42 <= a <= 0.58 and 0.35 <= a_s <= 0.65
    print(f"[{'PASS' if good else 'FAIL'}] random-walk bridge: alpha_loc {a:.2f} (0.5), "
          f"spectral slope {sl:.2f} -> alpha_s {a_s:.2f} (0.5)")
    ok &= good
    sm = np.sin(np.linspace(0, 2 * np.pi * 3, 20000, endpoint=False)) * 100
    a, _ = alpha_loc(sm)
    good = a >= 0.9
    print(f"[{'PASS' if good else 'FAIL'}] smooth profile: alpha_loc {a:.2f} (~1)")
    ok &= good
    print('\nVALIDATION', 'PASSED' if ok else 'FAILED')
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('graphs', nargs='*')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--first', type=str, default='results/blocked_scan_b2_20261013.csv')
    ap.add_argument('--census', type=str,
                    default='results/curvature_census_20261017_010558.csv')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.validate:
        raise SystemExit(0 if _validate() else 1)
    paths = sorted(p for g in args.graphs for p in glob.glob(g))
    stamp = time.strftime('%Y%m%d_%H%M%S')
    run(paths, args.first, f"results/edge_geometry_{stamp}")
    w2d(args.census)


if __name__ == '__main__':
    main()
