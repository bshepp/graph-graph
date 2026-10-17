"""
Wave 5: labelled rule variants of `sheet` (relief, veto) against the default.

Pre-registration: docs/superpowers/specs/2026-10-17-wave5-preregistration.md.

  relief  a closure is admitted when a link end is at the cap (it rises to c + 1):
          each would-be dead pair becomes one interior degree-7 cone point
  veto    an attachment that would cap a node whose other boundary neighbour is
          already capped is refused: no defect forms

Both are default-off options of `sheet_growth.grow_sheet(variant=...)`; the default
generator is unchanged (checked by --validate). Neither is ever reported as `sheet`.

Usage:
    python sheet_variants.py --validate
    python sheet_variants.py --scan [--jobs 6]
    python sheet_variants.py --cap7
"""

import argparse
import csv
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from sheet_growth import grow_sheet, to_csr, _closable, _is_boundary, load_csr
from dead_corner_hazard import DeadDetector

CAP = 6
SCAN_VARIANTS = ('sheet', 'relief', 'veto')
SCAN_BETAS = (1.0, 1.25)
SCAN_SEEDS = tuple(range(600, 608))
N_MAX = 2 ** 20
MARKS = [2 ** k for k in range(12, 21)]
COMPACT_B = 3.85


def checkpoint_stats(adj, guard: int) -> Dict[str, float]:
    n = len(adj)
    b_edges = 0
    on_b = np.zeros(n, dtype=bool)
    for u in range(n):
        for v in adj[u]:
            if u < v and len(adj[u] & adj[v]) == 1:
                b_edges += 1
                on_b[u] = on_b[v] = True
    deg = np.fromiter((len(s) for s in adj), dtype=np.int64, count=n)
    interior = ~on_b
    dead = sum(1 for x in np.nonzero(on_b & (deg >= CAP))[0].tolist()
               if _closable(adj, x, CAP, guard) is None)
    A = to_csr(adj)
    d0 = dijkstra(A, indices=0, unweighted=True)
    return {'N': n, 'B': b_edges, 'B_sqrtN': b_edges / np.sqrt(n),
            'deg7_interior': int((interior & (deg == CAP + 1)).sum()),
            'deg7_all': int((deg == CAP + 1).sum()), 'max_deg': int(deg.max()),
            'dead': dead, 'ecc0': float(d0[np.isfinite(d0)].max())}


def grow_one(spec) -> Dict:
    variant, beta, seed = spec
    random.seed(seed)
    np.random.seed(seed)
    guard = CAP + 1 if variant == 'relief' else CAP
    det = DeadDetector(CAP, guard=guard)
    first7 = {'N': -1}

    def on_event(kind, nodes, adj):
        det(kind, nodes, adj)
        if first7['N'] < 0 and any(len(adj[x]) > CAP for x in nodes):
            first7['N'] = len(adj)
        return False

    rows: List[Dict] = []

    def on_ck(n, adj):
        rows.append(checkpoint_stats(adj, guard))

    t0 = time.time()
    adj = grow_sheet(N_MAX, CAP, beta, seed, checkpoints=MARKS, on_checkpoint=on_ck,
                     on_event=on_event, variant=variant)
    A = to_csr(adj)
    d0 = dijkstra(A, indices=0, unweighted=True)
    h = np.bincount(d0[np.isfinite(d0)].astype(int))
    V = np.cumsum(h)
    slopes = {}
    for r in (50, 100, 200, 300, 400):
        if int(r * 1.25) < len(V):
            slopes[r] = float(np.log(V[int(r * 1.25)] / V[int(r / 1.25)]) / np.log(1.5625))
    from curvature_census import disc_audit
    au = disc_audit(A, seed)
    last, prev = rows[-1], rows[-2]
    ecc_exp = float(np.log(last['ecc0'] / prev['ecc0']) / np.log(last['N'] / prev['N']))
    return {'variant': variant, 'beta': beta, 'seed': seed, 'n_end': len(adj),
            'stalled': int(len(adj) < N_MAX),
            'first_dead': det.records[0]['N'] if det.records else -1,
            'first_dead_mode': det.records[0]['mode'] if det.records else '',
            'first_deg7': first7['N'], 'ecc_exp_last': ecc_exp,
            'B_sqrtN_end': last['B_sqrtN'], 'dead_end': last['dead'],
            'deg7_end': last['deg7_interior'], 'deg7_prev': prev['deg7_interior'],
            'max_deg': last['max_deg'],
            'slopes0': ' '.join(f"{r}:{x:.2f}" for r, x in slopes.items()),
            'slope_max': max(slopes.values()) if slopes else float('nan'),
            'audit': au['verdict'], 'audit_d': au['d_last'],
            'profile': ' '.join(f"{r['N']}:{r['B_sqrtN']:.2f}/{r['dead']}/{r['deg7_interior']}"
                                f"/{r['ecc0']:.0f}" for r in rows),
            'seconds': round(time.time() - t0)}


def read(rows: List[Dict]) -> None:
    for v in SCAN_VARIANTS:
        rs = [r for r in rows if r['variant'] == v]
        if not rs:
            continue
        near = sum(1 for r in rs if abs(r['B_sqrtN_end'] / COMPACT_B - 1) <= 0.10)
        ecc_ok = sum(1 for r in rs if 0.40 <= r['ecc_exp_last'] <= 0.60)
        cross = sum(1 for r in rs if r['B_sqrtN_end'] > 1.2 * COMPACT_B or r['ecc_exp_last'] < 0.35)
        nodead = sum(1 for r in rs if r['dead_end'] == 0)
        if near >= 14 and ecc_ok >= 14:
            verdict = 'NO CROSSOVER'
        elif cross >= 12:
            verdict = 'CROSSOVER'
        else:
            verdict = 'UNDECIDED'
        line = (f"{v:>6}: {verdict}  (B/sqrtN within 10% of 3.85 in {near}/{len(rs)}, ecc exponent "
                f"in [0.40, 0.60] in {ecc_ok}/{len(rs)}, crossed by the CROSSOVER clause in "
                f"{cross}/{len(rs)}; zero dead corners at 2^20 in {nodead}/{len(rs)})")
        if v == 'relief':
            crossed = [r for r in rs if r['B_sqrtN_end'] > 1.2 * COMPACT_B or r['ecc_exp_last'] < 0.35]
            hyp = [r for r in crossed if r['slope_max'] > 2.5 and r['dead_end'] == 0 and
                   r['deg7_prev'] > 0 and 1.6 <= r['deg7_end'] / r['deg7_prev'] <= 2.4]
            line += (f"; HYPERBOLIC INTERIOR in {len(hyp)}/{len(crossed)} crossed growths")
        print(line)
        for r in rs:
            print(f"    beta {r['beta']} seed {r['seed']}: B/sqrtN {r['B_sqrtN_end']:.2f}, ecc exp "
                  f"{r['ecc_exp_last']:.2f}, dead {r['dead_end']} (first {r['first_dead']} "
                  f"{r['first_dead_mode']}), deg-7 interior {r['deg7_end']} (first at "
                  f"{r['first_deg7']}), V(r) slopes {r['slopes0']}, audit {r['audit']} "
                  f"{r['audit_d']:.2f}")


def scan(jobs: int) -> None:
    specs = [(v, b, s) for b in SCAN_BETAS for v in SCAN_VARIANTS for s in SCAN_SEEDS]
    out = f"results/sheet_variants_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    rows: List[Dict] = []
    with ProcessPoolExecutor(jobs) as ex, open(out, 'w', newline='') as f:
        w = None
        for r in ex.map(grow_one, specs):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(r.keys()))
                w.writeheader()
            w.writerow(r)
            f.flush()
            rows.append(r)
            print(f"{r['variant']:>6} beta {r['beta']} seed {r['seed']}: B/sqrtN "
                  f"{r['B_sqrtN_end']:.2f} ecc exp {r['ecc_exp_last']:.2f} dead {r['dead_end']} "
                  f"deg7 {r['deg7_end']} [{r['seconds']} s]", flush=True)
    print(f"saved: {out}")
    read(rows)


def cap7() -> None:
    """C6: growth rate of cap-7 sheets against ln 1.5 (Keller-Peyerimhoff)."""
    bound = float(np.log(1.5))
    ok = True
    for s in (100, 101, 102):
        A = load_csr(f'results/sheet_c7_b2_s{s}_N512000.npz')
        n = A.shape[0]
        deg = np.diff(A.indptr)
        rng = np.random.default_rng(s)
        # interior sources: nodes among the oldest tenth (the core)
        src = rng.choice(np.arange(n // 10), 30, replace=False)
        D = dijkstra(A, indices=src, unweighted=True)
        mus = []
        for row in D:
            h = np.bincount(row[np.isfinite(row)].astype(int))
            V = np.cumsum(h)
            r = np.arange(len(V))
            m = (V < 0.1 * n) & (r >= 2)
            if m.sum() >= 4:
                mus.append(np.polyfit(r[m], np.log(V[m]), 1)[0])
        mu = float(np.median(mus))
        frac7 = float(np.mean(deg == 7))
        ok &= mu >= bound
        print(f"cap 7 seed {s}: N {n}, fraction of degree-7 nodes {frac7:.3f}, max degree "
              f"{deg.max()}, growth rate mu = {mu:.3f} per hop (bound ln 1.5 = {bound:.3f}), "
              f"radii used {int(m.sum())}")
    print(f"C6: {'BOUND RESPECTED' if ok else 'BELOW THE BOUND'}")


def _validate() -> bool:
    ok = True
    a = grow_sheet(20000, CAP, 2.0, 3)
    b = grow_sheet(20000, CAP, 2.0, 3, variant='sheet')
    good = len(a) == len(b) and all(x == y for x, y in zip(a, b))
    print(f"[{'PASS' if good else 'FAIL'}] default variant is bit-identical to the generator")
    ok &= good
    for v, cap_max in (('relief', CAP + 1), ('veto', CAP)):
        adj = grow_sheet(40000, CAP, 1.0, 11, variant=v)
        st = checkpoint_stats(adj, CAP + 1 if v == 'relief' else CAP)
        # Gauss-Bonnet with interior degree-7 cone points:
        # sum_bdry (4 - d) = 6 + (number of interior degree-7 nodes).
        n = len(adj)
        bdry = [u for u in range(n) if any(len(adj[u] & adj[w]) == 1 for w in adj[u])]
        bset = set(bdry)
        gb = sum(4 - len(adj[u]) for u in bdry)
        int7 = sum(1 for u in range(n) if u not in bset and len(adj[u]) == 7)
        int_other = sum(1 for u in range(n) if u not in bset and len(adj[u]) not in (6, 7))
        good = st['max_deg'] <= cap_max and gb == 6 + int7 and int_other == 0
        print(f"[{'PASS' if good else 'FAIL'}] {v}: max degree {st['max_deg']} (<= {cap_max}); "
              f"Gauss-Bonnet sum_bdry(4 - d) = {gb} = 6 + {int7} interior degree-7 nodes; "
              f"dead corners {st['dead']}")
        ok &= good
    print('\nVALIDATION', 'PASSED' if ok else 'FAILED')
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument('--validate', action='store_true')
    m.add_argument('--scan', action='store_true')
    m.add_argument('--cap7', action='store_true')
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.validate:
        raise SystemExit(0 if _validate() else 1)
    if args.scan:
        scan(args.jobs)
    else:
        cap7()


if __name__ == '__main__':
    main()
