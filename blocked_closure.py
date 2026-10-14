"""
Blocked closures: the permanent local defect at the bottom of a `sheet`'s fjords.

Pre-registration: docs/superpowers/specs/2026-10-13-blocked-closure-preregistration.md.

In `sheet_growth`, a boundary edge whose endpoint has reached degree c can only
fire a closure event, and only if that endpoint is closable (its two boundary
neighbours can be linked without pinching and both still have degree < c). If
neither endpoint is closable the edge is dead: degrees never fall and adjacency
never shrinks, so nothing ever fires on it again and it stays on the boundary
forever. A **blocked node** is a boundary node of degree >= c that is not
closable. Growth stalls there while the rest of the disc advances, so the
blocked node ends up at the bottom of a fjord.

Modes:
    python blocked_closure.py --validate
    python blocked_closure.py results/sheet_cross_b2_s3*_N4194304.npz \
        --crossover results/sheet_crossover_partA_N4194304_*.csv      # saved graphs (post hoc)
    python blocked_closure.py --grow --beta 1.5 --seeds 400 407 --n-max 524288 --jobs 8
                                                                       # fresh seeds (the test)
"""

import argparse
import csv
import glob
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from sheet_growth import grow_sheet, load_csr, census, _closable
from sheet_transition import checkpoints, local_exponents, crossover_size
from arm_birth import adjacency_sets, sets_to_csr, boundary_nodes

CAP = 6
N_LO = 2 ** 12
STEP = 2 ** 0.125
RATIO_MAX = 4.0         # N*_B / N_first within this factor counts as "the trigger"
CLUSTER_HOPS = 10


def blocked_nodes(adj: List[set], c: int = CAP) -> List[int]:
    """Boundary nodes of degree >= c whose wheel can never be closed."""
    out = []
    for x, nb in enumerate(adj):
        if len(nb) < c:
            continue
        if not any(len(nb & adj[y]) == 1 for y in nb):
            continue                      # interior
        if _closable(adj, x, c) is None:
            out.append(x)
    return out


def _blocked_at(A: sp.csr_array, m: int, c: int = CAP) -> List[int]:
    P = A[:m, :m].tocsr()
    return blocked_nodes(adjacency_sets(P), c)


def first_blocked(A: sp.csr_array, n_lo: int = N_LO, c: int = CAP
                  ) -> Optional[Dict[str, object]]:
    """Smallest prefix size at which a blocked node exists (octave scan, then
    bisection -- blocked is permanent, so the predicate is monotone in N)."""
    n_max = A.shape[0]
    prev, m = 0, n_lo
    while True:
        m = min(m, n_max)
        if _blocked_at(A, m, c):
            break
        if m >= n_max:
            return None
        prev, m = m, m * 2
    lo, hi = prev, m
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _blocked_at(A, mid, c):
            hi = mid
        else:
            lo = mid
    return {'n_first': hi, 'ids': _blocked_at(A, hi, c)}


def clusters(A: sp.csr_array, nodes: Sequence[int], hops: int = CLUSTER_HOPS
             ) -> int:
    """Number of groups of nodes linked by chains of <= `hops` steps."""
    left = set(int(v) for v in nodes)
    k = 0
    while left:
        k += 1
        frontier = [left.pop()]
        while frontier:
            v = frontier.pop()
            d = dijkstra(A, indices=v, unweighted=True, limit=hops + 0.5)
            near = [u for u in list(left) if np.isfinite(d[u])]
            for u in near:
                left.discard(u)
            frontier.extend(near)
    return k


def seed_of(path: str) -> int:
    base = os.path.basename(path)
    i = base.index('_s') + 2
    return int(base[i:i + 3])


def nstar_from_csvs(paths: Sequence[str]) -> Dict[int, Tuple[str, float]]:
    from crossover_cv import per_seed_nstar
    rows: List[Dict] = []
    for p in paths:
        with open(p) as f:
            rows += list(csv.DictReader(f))
    rows = [r for r in rows if float(r['beta']) == 2.0]
    ps = per_seed_nstar(rows)
    return {s: (v['status'], v['N_star']) for s, v in ps.items()}


def scan(paths: Sequence[str], cross_csvs: Sequence[str], out_csv: str) -> List[Dict]:
    nstar = nstar_from_csvs(cross_csvs) if cross_csvs else {}
    rows = []
    print(f"{'seed':>4} {'N_first':>9} {'first ids':>22} {'N*_B':>11} {'N*/N_first':>10} "
          f"{'blocked@Nmax':>12} {'clusters':>8} {'oldest bnd':>10} {'oldest blocked':>14}")
    for path in paths:
        t0 = time.time()
        A = load_csr(path)
        s = seed_of(path)
        fb = first_blocked(A)
        adj = adjacency_sets(A)
        blk = blocked_nodes(adj)
        bnd = boundary_nodes(adj)
        oldest = int(bnd.min()) if bnd.size else -1
        ncl = clusters(A, blk) if blk else 0
        st, ns = nstar.get(s, ('n/a', float('nan')))
        n_first = fb['n_first'] if fb else None
        ratio = (ns / n_first) if (fb and st == 'finite') else float('nan')
        rows.append({'seed': s, 'n_max': A.shape[0], 'N_first': n_first or -1,
                     'first_ids': ' '.join(map(str, fb['ids'][:6])) if fb else '',
                     'nstar_status': st, 'N_star_B': ns, 'ratio': ratio,
                     'blocked_at_nmax': len(blk), 'clusters': ncl,
                     'oldest_boundary': oldest,
                     'oldest_is_blocked': bool(oldest in set(blk))})
        ns_txt = f"{ns:,.0f}" if st == 'finite' else st
        print(f"{s:4d} {n_first if n_first else '--':>9} "
              f"{rows[-1]['first_ids']:>22} {ns_txt:>11} "
              f"{ratio:10.2f} {len(blk):12d} {ncl:8d} {oldest:10d} "
              f"{str(rows[-1]['oldest_is_blocked']):>14}   [{time.time() - t0:.0f} s]",
              flush=True)
        del A, adj
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved: {out_csv}")
    return rows


# --------------------------------------------------------------------------
# Fresh seeds: grow, record boundary length and blocked count at checkpoints
# --------------------------------------------------------------------------

def fine_marks(n_max: int) -> List[int]:
    k = int(np.floor(np.log(n_max / N_LO) / np.log(STEP))) + 1
    fine = {int(round(N_LO * STEP ** i)) for i in range(k)}
    return sorted((fine | set(checkpoints(1024, n_max))) - {n for n in fine if n > n_max})


def grow_one(spec: Tuple[float, int, int]) -> List[Dict]:
    beta, seed, n_max = spec
    rows: List[Dict] = []

    def snap(n: int, adj) -> None:
        cs = census(adj, CAP)
        blk = blocked_nodes(adj, CAP)
        rows.append({'beta': beta, 'seed': seed, 'N': n,
                     'boundary_edges': cs['boundary_edges'],
                     'B_over_sqrtN': cs['boundary_edges'] / np.sqrt(n),
                     'blocked': len(blk),
                     'blocked_min_id': min(blk) if blk else -1})

    adj = grow_sheet(n_max, CAP, beta, seed, checkpoints=fine_marks(n_max),
                     on_checkpoint=snap)
    for r in rows:
        r['reached'] = len(adj)
    return rows


def summarize(rows: List[Dict], n_max: int) -> List[Dict]:
    marks = set(checkpoints(1024, n_max))
    out = []
    for seed in sorted({int(r['seed']) for r in rows}):
        rs = sorted((r for r in rows if int(r['seed']) == seed), key=lambda r: int(r['N']))
        coarse = [r for r in rs if int(r['N']) in marks]
        Ns = [int(r['N']) for r in coarse]
        B = [float(r['boundary_edges']) for r in coarse]
        if len(Ns) >= 3:
            mid, e = local_exponents(Ns, B)
            st, ns = crossover_size(mid, e)
        else:
            st, ns = 'censored', float('nan')
        firsts = [int(r['N']) for r in rs if int(r['blocked']) > 0]
        n_first = firsts[0] if firsts else None
        ratio = ns / n_first if (n_first and st == 'finite') else float('nan')
        out.append({'seed': seed, 'reached': int(rs[-1]['reached']),
                    'B_over_sqrtN_last': float(rs[-1]['B_over_sqrtN']),
                    'nstar_status': st, 'N_star_B': ns, 'N_first': n_first or -1,
                    'blocked_last': int(rs[-1]['blocked']), 'ratio': ratio})
    return out


def reading(summary: List[Dict]) -> str:
    fin = [s for s in summary if s['nstar_status'] == 'finite']
    cens = [s for s in summary if s['nstar_status'] != 'finite']
    hit = sum(1 for s in fin if s['N_first'] > 0 and s['N_first'] <= s['N_star_B']
              and s['ratio'] <= RATIO_MAX)
    miss = sum(1 for s in fin if s['N_first'] < 0 or s['N_first'] > s['N_star_B']
               or s['ratio'] > RATIO_MAX)
    clean = sum(1 for s in cens if s['blocked_last'] == 0)
    n = len(fin)
    ctrl = f"; control: {clean}/{len(cens)} uncrossed seeds have no blocked node"
    if n == 0:
        return 'UNDECIDED (no seed crossed)' + ctrl
    if hit >= np.ceil(0.75 * n):
        return f'TRIGGER CONFIRMED ({hit}/{n} crossed seeds)' + ctrl
    if miss >= np.ceil(0.5 * n):
        return f'TRIGGER REFUTED ({miss}/{n} crossed seeds miss)' + ctrl
    return f'UNDECIDED ({hit} hit, {miss} miss of {n} crossed seeds)' + ctrl


def grow(beta: float, seeds: Sequence[int], n_max: int, jobs: int, out_csv: str) -> None:
    specs = [(beta, s, n_max) for s in seeds]
    rows: List[Dict] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for rs in ex.map(grow_one, specs):
            rows += rs
            print(f"  seed {rs[0]['seed']} done, reached {rs[0]['reached']:,} "
                  f"[{time.time() - t0:.0f} s]", flush=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved: {out_csv}")
    report(rows, n_max)


def report(rows: List[Dict], n_max: int) -> None:
    summ = summarize(rows, n_max)
    print(f"\n{'seed':>4} {'reached':>8} {'B/sqrtN':>7} {'N*_B':>9} {'N_first':>9} "
          f"{'N*/N_first':>10} {'blocked@end':>11}")
    for s in summ:
        ns = f"{s['N_star_B']:,.0f}" if s['nstar_status'] == 'finite' else s['nstar_status']
        nf = f"{s['N_first']:,}" if s['N_first'] > 0 else '--'
        print(f"{s['seed']:4d} {s['reached']:8,d} {s['B_over_sqrtN_last']:7.2f} {ns:>9} "
              f"{nf:>9} {s['ratio']:10.2f} {s['blocked_last']:11d}")
    print(f"\nVERDICT (frozen rules): {reading(summ)}")


def _validate() -> bool:
    ok = True
    print("[1] a compact 40 k-node sheet has no blocked node")
    adj = grow_sheet(40000, CAP, 3.0, seed=11)
    blk = blocked_nodes(adj)
    good = len(blk) == 0
    ok &= good
    print(f"  blocked: {len(blk)}  {'OK' if good else 'FAIL'}")
    print("[2] a planted dead corner is found, at the right prefix size")
    adj2 = [set(s) for s in adj]
    bnd = boundary_nodes(adj2)
    x = next(int(v) for v in bnd if _closable(adj2, int(v), CAP) is not None)
    a, b = _closable(adj2, x, CAP)
    n0 = len(adj2)

    def pendant(v: int) -> None:
        adj2.append({v})
        adj2[v].add(len(adj2) - 1)

    while len(adj2[x]) < CAP:            # x reaches the cap first (still closable)
        pendant(x)
    n_before = len(adj2)
    while len(adj2[a]) < CAP:            # then a: the corner is now dead
        pendant(a)
    n_dead = len(adj2)
    for _ in range(50):                  # padding after the event
        pendant(b) if len(adj2[b]) < CAP - 1 else pendant(int(bnd[-1]))
    A2 = sets_to_csr(adj2)
    blk2 = blocked_nodes(adj2)
    fb = first_blocked(A2, n_lo=1024)
    good = (x in blk2 and fb is not None and fb['n_first'] == n_dead and x in fb['ids']
            and blocked_nodes(adjacency_sets(A2[:n_before, :n_before].tocsr())) == [])
    ok &= good
    print(f"  planted dead corner {x}: blocked {blk2}, first at N = "
          f"{fb['n_first'] if fb else None} (expected {n_dead}; sheet had {n0})  "
          f"{'OK' if good else 'FAIL'}")
    print("[3] readings")
    mk = lambda st, ns, nf, bl: {'nstar_status': st, 'N_star_B': ns, 'N_first': nf,
                                 'blocked_last': bl,
                                 'ratio': ns / nf if (nf > 0 and st == 'finite') else float('nan')}
    conf = [mk('finite', 100e3, 60e3, 5)] * 6 + [mk('finite', 100e3, 10e3, 5),
                                                  mk('censored', float('nan'), -1, 0)]
    refu = [mk('finite', 100e3, 10e3, 5)] * 3 + [mk('finite', 100e3, -1, 0)] * 2 \
        + [mk('finite', 100e3, 60e3, 5)] * 3
    und = [mk('finite', 100e3, 60e3, 5)] * 5 + [mk('finite', 100e3, 10e3, 5)] * 2 \
        + [mk('finite', 100e3, 150e3, 5)]
    for name, s, want in (('confirmed', conf, 'TRIGGER CONFIRMED'),
                          ('refuted', refu, 'TRIGGER REFUTED'),
                          ('undecided', und, 'UNDECIDED')):
        got = reading(s)
        good = got.startswith(want)
        ok &= good
        print(f"  {name:>9}: {got}  {'OK' if good else 'FAIL'}")
    print("\nPASS: blocked closure" if ok else "\nFAIL: blocked closure")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Blocked closures on a sheet's edge")
    ap.add_argument('paths', nargs='*', help='saved sheet graphs (.npz)')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--crossover', nargs='*', default=None,
                    help='Part A CSVs giving N*_B per seed')
    ap.add_argument('--grow', action='store_true')
    ap.add_argument('--beta', type=float, default=1.5)
    ap.add_argument('--seeds', type=int, nargs=2, default=(400, 407))
    ap.add_argument('--n-max', type=int, default=2 ** 19)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--report', type=str, default=None,
                    help='re-summarize a saved --grow CSV')
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.validate:
        raise SystemExit(0 if _validate() else 1)
    Path('results').mkdir(exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    if args.report:
        with open(args.report) as f:
            rows = list(csv.DictReader(f))
        report(rows, args.n_max)
        return
    if args.grow:
        out = args.out or f'results/blocked_grow_b{args.beta:g}_{stamp}.csv'
        grow(args.beta, list(range(args.seeds[0], args.seeds[1] + 1)), args.n_max,
             args.jobs, out)
        return
    if not args.paths:
        ap.error('give saved sheet graphs, or --grow')
    cross = args.crossover
    if cross is not None and not cross:
        cross = sorted(glob.glob('results/sheet_crossover_partA_N4194304_*.csv'))
    out = args.out or f'results/blocked_scan_{stamp}.csv'
    scan(args.paths, cross or [], out)


if __name__ == '__main__':
    main()
