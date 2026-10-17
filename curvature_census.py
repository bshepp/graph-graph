"""
Curvature census of saved `sheet` graphs, and the "is it still a disc?" audit.

Pre-registration: docs/superpowers/specs/2026-10-17-wave1-preregistration.md
(W1-b and W1-c).

Combinatorial curvature (Keller-Peyerimhoff; Wu et al.) with the outer face as
an infinigon: interior node kappa = (6 - d)/6, boundary node kappa = (4 - d)/6.
`sheet` is a triangulated disc, so Gauss-Bonnet reads
    sum over boundary nodes of (4 - d) = 6        (interior degree is c = 6)
exactly, B_v = B_e (one simple boundary cycle) and every boundary node of
degree d lies in d - 1 triangles. A dead corner is a boundary node of degree
>= c that `sheet_growth._closable` refuses; its failure modes are
(i) not two boundary edges, (ii) link ends adjacent, (iii) link ends share
another neighbour (pinch), (iv) a link end at the cap (degree guard).

Prefix caveat: node ids are birth order, so the prefix A[:m, :m] is the graph
when node m was born PLUS any closure edges among older nodes that fired
later. The final graph is exact; prefixes are approximate, and the number of
edges in no triangle (the visible artifact) is reported with every prefix row.

Usage:
    python curvature_census.py --validate
    python curvature_census.py results/sheet_cross_b2_s3*_N4194304.npz \
        --first results/blocked_scan_b2_20261013.csv              # W1-b
    python curvature_census.py results/sheet_cross_b2_s3*_N4194304.npz --disc   # W1-c
"""

import argparse
import csv
import glob
import os
import random
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from sheet_growth import load_csr, grow_sheet, to_csr, _closable
from dimension import local_dimension
from window_stability import model_comparison

CAP = 6
DISC_WINDOWS = (10, 20, 40, 80, 160, 240, 320, 480)
DISC_SOURCES = 30
DRIFT_TOL = 0.15
SATURATION = 0.1


# --------------------------------------------------------------------------
# Census of one graph
# --------------------------------------------------------------------------

def edge_triangles(A: sp.csr_array) -> sp.csr_array:
    """Number of triangles on every edge, as a sparse matrix on A's pattern."""
    A = A.astype(np.float32)
    A.sort_indices()
    return (A @ A).multiply(A).tocsr()


def failure_modes_csr(A: sp.csr_array, T: sp.csr_array, x: int, c: int = CAP
                      ) -> str:
    row = T.indices[T.indptr[x]:T.indptr[x + 1]]
    val = T.data[T.indptr[x]:T.indptr[x + 1]]
    ends = row[val == 1]
    if len(ends) != 2:
        return 'i'
    a, b = int(ends[0]), int(ends[1])
    na = set(A.indices[A.indptr[a]:A.indptr[a + 1]].tolist())
    nbb = set(A.indices[A.indptr[b]:A.indptr[b + 1]].tolist())
    out = []
    if b in na:
        out.append('ii')
    if (na & nbb) != {x}:
        out.append('iii')
    if len(na) >= c or len(nbb) >= c:
        out.append('iv')
    return '+'.join(out)


def census(A: sp.csr_array, c: int = CAP) -> Dict[str, object]:
    n = A.shape[0]
    T = edge_triangles(A)
    deg = np.diff(A.indptr)
    coo = T.tocoo()
    upper = coo.row < coo.col
    t_up = coo.data[upper]
    b_mask = t_up == 1
    B_e = int(b_mask.sum())
    nonmanifold = int(((t_up != 1) & (t_up != 2)).sum())
    on_b = np.zeros(n, dtype=bool)
    on_b[coo.row[upper][b_mask]] = True
    on_b[coo.col[upper][b_mask]] = True
    B_v = int(on_b.sum())
    db = deg[on_b]
    hist = {d: int((db == d).sum()) for d in range(2, c + 1)}
    gb = int((4 - db).sum())
    interior_ok = bool(np.all(deg[~on_b] == c)) if (~on_b).any() else True
    # Dead corners: boundary nodes at the cap that cannot close.
    modes: Dict[str, int] = {}
    dead: List[int] = []
    for x in np.nonzero(on_b & (deg >= c))[0]:
        m = failure_modes_csr(A, T, int(x), c)
        if m:
            dead.append(int(x))
            modes[m] = modes.get(m, 0) + 1
    m_minus = float(np.maximum(0, db - 4).sum() / 6)
    return {'N': n, 'B_v': B_v, 'B_e': B_e, 'gb_sum': gb,
            'interior_deg_ok': interior_ok, 'nonmanifold_edges': nonmanifold,
            **{f'bd{d}': hist[d] for d in hist},
            'dead': len(dead), 'dead_ids': dead,
            'modes': ' '.join(f"{k}:{v}" for k, v in sorted(modes.items())),
            'iv_only_frac': modes.get('iv', 0) / len(dead) if dead else float('nan'),
            'M_minus': m_minus}


def seed_of(path: str) -> int:
    base = os.path.basename(path)
    i = base.index('_s') + 2
    j = i
    while j < len(base) and base[j].isdigit():
        j += 1
    return int(base[i:j])


def first_sizes(path: Optional[str]) -> Dict[int, int]:
    if not path:
        return {}
    with open(path) as f:
        return {int(r['seed']): int(r['N_first']) for r in csv.DictReader(f)}


def run_census(paths: Sequence[str], first_csv: Optional[str], out_csv: str
               ) -> List[Dict]:
    firsts = first_sizes(first_csv)
    rows: List[Dict] = []
    for path in paths:
        t0 = time.time()
        A = load_csr(path)
        n = A.shape[0]
        s = seed_of(path)
        marks = [2 ** k for k in range(12, 23) if 2 ** k < n] + [n]
        nf = firsts.get(s, -1)
        if nf > 0:
            marks += [nf - 1, nf]
        for m in sorted(set(marks)):
            P = A[:m, :m].tocsr() if m < n else A
            cz = census(P)
            row = {'graph': os.path.basename(path), 'seed': s, 'exact': int(m == n),
                   'N_first': nf, **{k: v for k, v in cz.items() if k != 'dead_ids'}}
            if m == nf:
                # Modes read on the final graph: modes only accumulate, so
                # 'iv' alone at the end means 'iv' alone at creation.
                row['first_dead'] = ' '.join(map(str, cz['dead_ids']))
                TA = edge_triangles(A)
                row['first_modes'] = ' '.join(failure_modes_csr(A, TA, x)
                                              for x in cz['dead_ids'])
                del TA
                adjp = [int(y) for y in cz['dead_ids']]
                row['first_adjacent_pair'] = int(
                    len(adjp) == 2 and adjp[1] in
                    set(P.indices[P.indptr[adjp[0]]:P.indptr[adjp[0] + 1]].tolist()))
                row['first_degs'] = ' '.join(str(int(P.indptr[x + 1] - P.indptr[x]))
                                             for x in adjp)
            rows.append(row)
        last = rows[-1]
        print(f"seed {s}: N {n}  B_v {last['B_v']}  B_e {last['B_e']}  GB {last['gb_sum']}"
              f"  interior ok {last['interior_deg_ok']}  nonmanifold "
              f"{last['nonmanifold_edges']}  dead {last['dead']} ({last['modes']})"
              f"   [{time.time() - t0:.0f} s]", flush=True)
        del A
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"saved: {out_csv}")
    return rows


def report_census(rows: List[Dict]) -> Dict[str, object]:
    exact = [r for r in rows if r['exact']]
    gate = all(r['B_v'] == r['B_e'] and r['gb_sum'] == 6 and r['interior_deg_ok']
               and r['nonmanifold_edges'] == 0 for r in exact)
    print(f"\nInstrument gate on the exact (final) graphs: "
          f"{'PASS' if gate else 'FAIL'}  "
          f"({sum(1 for r in exact)} graphs)")
    prefix_art = [r['nonmanifold_edges'] for r in rows if not r['exact']]
    if prefix_art:
        print(f"prefix artifact: edges in no triangle, max {max(prefix_art)}, "
              f"rows with any {sum(1 for a in prefix_art if a)} of {len(prefix_art)}")
    print(f"\n{'seed':>4} {'N_first':>9} {'first dead (final-graph modes)':>34} "
          f"{'pair':>4} {'degs':>5} {'dead@end':>8} {'iv-only':>7}")
    crossed, guard = 0, 0
    for s in sorted({r['seed'] for r in rows}):
        rs = [r for r in rows if r['seed'] == s]
        end = [r for r in rs if r['exact']][0]
        fr = [r for r in rs if r.get('first_dead')]
        if fr:
            f = fr[0]
            crossed += 1
            ok = (f['first_adjacent_pair'] == 1 and
                  all(m == 'iv' for m in f['first_modes'].split()) and
                  all(d == str(CAP) for d in f['first_degs'].split()))
            guard += ok
            print(f"{s:4d} {f['N_first']:9d} {f['first_dead'] + ' (' + f['first_modes'] + ')':>34} "
                  f"{f['first_adjacent_pair']:4d} {f['first_degs']:>5} "
                  f"{end['dead']:8d} {end['iv_only_frac']:7.2f}")
        else:
            print(f"{s:4d} {'--':>9} {'':>34} {'':>4} {'':>5} {end['dead']:8d}")
    if crossed:
        if guard >= max(crossed - 1, 1) and crossed >= 2:
            verdict = 'DEGREE GUARD'
        else:
            verdict = 'see table'
        pinch = sum(1 for r in rows if r.get('first_modes') and
                    any('iii' in m for m in r['first_modes'].split()))
        if pinch >= 2:
            verdict = 'PINCH'
        print(f"\nW1-b: {verdict}  (adjacent capped (iv)-only pair in {guard}/{crossed})")
    ivs = [r['iv_only_frac'] for r in rows if r['exact'] and r['dead']]
    if ivs:
        allm: Dict[str, int] = {}
        for r in rows:
            if r['exact'] and r['modes']:
                for kv in r['modes'].split():
                    k, v = kv.split(':')
                    allm[k] = allm.get(k, 0) + int(v)
        tot = sum(allm.values())
        print(f"all dead corners in the final graphs: {tot}; by mode {allm}; "
              f"(iv)-only fraction {allm.get('iv', 0) / tot:.3f} (prediction >= 0.80)")
    return {'gate': gate}


# --------------------------------------------------------------------------
# W1-c: is the sheet past N* still a disc?
# --------------------------------------------------------------------------

def disc_audit(A: sp.csr_array, seed: int) -> Dict[str, object]:
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    src = rng.choice(n, DISC_SOURCES, replace=False)
    r_max = DISC_WINDOWS[-1]
    B = np.zeros((len(src), r_max))
    for i in range(0, len(src), 5):
        D = dijkstra(A, indices=src[i:i + 5], unweighted=True, limit=r_max + 0.5)
        for j, row in enumerate(D):
            h = np.bincount(row[np.isfinite(row)].astype(int), minlength=r_max + 1)
            B[i + j] = np.cumsum(h)[1:r_max + 1]
    tab = []
    for mr in DISC_WINDOWS:
        ds = []
        for row in B:
            if row[mr - 1] >= SATURATION * n:
                continue
            d, _ = local_dimension([(r + 1, int(c)) for r, c in enumerate(row[:mr])],
                                   n, drift_tol=None)
            if np.isfinite(d):
                ds.append(d)
        if len(ds) >= max(5, len(src) // 4):
            tab.append((mr, float(np.median(ds)), len(ds)))
    mc = model_comparison(B, n)
    verdict, drift = 'INSUFFICIENT RANGE', float('nan')
    if len(tab) >= 2 and tab[-1][0] >= 2 * tab[0][0] and mc.get('n_pts', 0) >= 5:
        ds = [t[1] for t in tab]
        drift = max(ds) - min(ds)
        power = mc['rss_power'] < mc['rss_exp']
        if drift <= DRIFT_TOL and power:
            verdict = 'DIMENSION DEFINED'
        elif drift > DRIFT_TOL and not power:
            verdict = 'EXPONENTIAL (no dimension)'
        else:
            verdict = 'MIXED'
    return {'table': tab, 'drift': drift, 'verdict': verdict,
            'd_last': tab[-1][1] if tab else float('nan'),
            'r_last': tab[-1][0] if tab else 0, **mc}


def radius_profile(A: sp.csr_array) -> List[Dict[str, float]]:
    n = A.shape[0]
    out = []
    for m in [2 ** k for k in range(12, 23) if 2 ** k <= n]:
        P = A[:m, :m].tocsr()
        d = dijkstra(P, indices=0, unweighted=True)
        T = edge_triangles(P)
        coo = T.tocoo()
        bm = (coo.data == 1)
        bnodes = np.unique(coo.row[bm])
        db = d[bnodes]
        out.append({'N': m, 'ecc0': float(d[np.isfinite(d)].max()),
                    'bnd_median': float(np.median(db)),
                    'bnd_min': float(db.min())})
    return out


def run_disc(paths: Sequence[str], first_csv: Optional[str], out_csv: str) -> None:
    firsts = first_sizes(first_csv)
    rows = []
    disc_ok = ecc_ok = n_cross = 0
    exp_count = 0
    for path in paths:
        t0 = time.time()
        A = load_csr(path)
        s = seed_of(path)
        crossed = firsts.get(s, -1) > 0
        au = disc_audit(A, s)
        prof = radius_profile(A)
        e_last = np.log2(prof[-1]['ecc0'] / prof[-2]['ecc0']) / \
            np.log2(prof[-1]['N'] / prof[-2]['N'])
        m_last = np.log2(prof[-1]['bnd_median'] / prof[-2]['bnd_median']) / \
            np.log2(prof[-1]['N'] / prof[-2]['N'])
        okd = au['verdict'] in ('DIMENSION DEFINED', 'MIXED') and \
            1.8 <= au['d_last'] <= 2.1
        oke = 0.40 <= e_last <= 0.65
        if crossed:
            n_cross += 1
            disc_ok += okd
            ecc_ok += oke
            exp_count += au['verdict'].startswith('EXPONENTIAL')
        win = '  '.join(f"R{r}={d:.2f}({k})" for r, d, k in au['table'])
        print(f"seed {s} {'crossed' if crossed else 'compact'}: {win}  drift "
              f"{au['drift']:.2f}  power RSS {au.get('rss_power', float('nan')):.4f} vs exp "
              f"{au.get('rss_exp', float('nan')):.4f} -> {au['verdict']}", flush=True)
        print("    ecc(node 0): " + "  ".join(f"{int(p['N'] / 1024)}k:{p['ecc0']:.0f}"
                                         for p in prof))
        print("    median boundary distance: " + "  ".join(
            f"{int(p['N'] / 1024)}k:{p['bnd_median']:.0f}" for p in prof))
        print(f"    last-octave exponents: ecc {e_last:.2f}, median boundary "
              f"distance {m_last:.2f}   [{time.time() - t0:.0f} s]", flush=True)
        rows.append({'seed': s, 'crossed': int(crossed), 'verdict': au['verdict'],
                     'drift': au['drift'], 'd_last': au['d_last'], 'r_last': au['r_last'],
                     'rss_power': au.get('rss_power'), 'rss_exp': au.get('rss_exp'),
                     'windows': win, 'ecc_exp_last': e_last, 'bndmed_exp_last': m_last,
                     'ecc_profile': ' '.join(f"{p['N']}:{p['ecc0']:.0f}" for p in prof),
                     'bndmed_profile': ' '.join(f"{p['N']}:{p['bnd_median']:.0f}"
                                                for p in prof)})
        del A
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved: {out_csv}")
    if n_cross:
        if exp_count >= 5:
            v = 'TREE-LIKE CONFIRMED'
        elif disc_ok >= 8 and ecc_ok >= 7:
            v = 'STILL A DISC'
        elif disc_ok >= 8:
            v = 'DISC WITH A STALLED RADIUS'
        else:
            v = 'UNDECIDED'
        print(f"\nW1-c: {v}  (audit disc-like on {disc_ok}/{n_cross} crossed graphs, "
              f"eccentricity exponent in [0.40, 0.65] on {ecc_ok}/{n_cross}, "
              f"EXPONENTIAL on {exp_count}/{n_cross})")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _validate() -> bool:
    from blocked_closure import blocked_nodes
    ok = True
    for beta, seed in ((1.0, 11), (2.0, 4), (0.0, 5)):
        adj = grow_sheet(30000, CAP, beta, seed)
        A = to_csr(adj)
        cz = census(A)
        static = set(blocked_nodes(adj, CAP))
        disc = (cz['B_v'] == cz['B_e'] and cz['gb_sum'] == 6 and cz['interior_deg_ok']
                and cz['nonmanifold_edges'] == 0)
        same = set(cz['dead_ids']) == static
        print(f"[{'PASS' if disc else 'FAIL'}] beta {beta} seed {seed}: disc identities "
              f"(B_v {cz['B_v']} = B_e {cz['B_e']}, GB {cz['gb_sum']}, interior ok "
              f"{cz['interior_deg_ok']}, nonmanifold {cz['nonmanifold_edges']})")
        print(f"[{'PASS' if same else 'FAIL'}]   dead corners == blocked_closure census "
              f"({len(static)}; modes {cz['modes'] or 'none'})")
        ok &= disc and same
    # Hand graph: one triangle -> B = 3, GB = 3 * (4 - 2) = 6.
    A = to_csr([{1, 2}, {0, 2}, {0, 1}])
    cz = census(A)
    good = cz['B_v'] == 3 and cz['gb_sum'] == 6
    print(f"[{'PASS' if good else 'FAIL'}] seed triangle: B 3, GB {cz['gb_sum']}")
    ok &= good
    # Disc audit on a big compact sheet reads a dimension near 2.
    adj = grow_sheet(400000, CAP, 3.0, 1)
    au = disc_audit(to_csr(adj), 1)
    good = au['verdict'] in ('DIMENSION DEFINED', 'MIXED') and 1.8 <= au['d_last'] <= 2.1
    print(f"[{'PASS' if good else 'FAIL'}] disc audit on a compact 400k sheet (beta 3): "
          f"{au['verdict']}, d at R{au['r_last']} = {au['d_last']:.2f}, windows "
          + ' '.join(f"R{r}={d:.2f}" for r, d, _ in au['table']))
    ok &= good
    print('\nVALIDATION', 'PASSED' if ok else 'FAILED')
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('graphs', nargs='*')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--disc', action='store_true')
    ap.add_argument('--first', type=str, default='results/blocked_scan_b2_20261013.csv')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.validate:
        raise SystemExit(0 if _validate() else 1)
    paths = sorted(p for g in args.graphs for p in glob.glob(g))
    stamp = time.strftime('%Y%m%d_%H%M%S')
    if args.disc:
        run_disc(paths, args.first, f"results/disc_audit_{stamp}.csv")
    else:
        rows = run_census(paths, args.first, f"results/curvature_census_{stamp}.csv")
        report_census(rows)


if __name__ == '__main__':
    main()
