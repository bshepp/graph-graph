"""Descriptive follow-up to wave 9 (not registered): does `disloc`'s unpaired
remainder n7 - n5 grow like the boundary (~sqrt N) or like the area (~N)?
Seeds 20-23 (never registered), beta 0.5 and 0.75, to 2^19.

Usage:
    python disloc_remainder.py [--seed 20] [--nmax 524288]
"""
import random, argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sheet_growth import grow_sheet

CAP = 6


def stats(adj):
    n = len(adj)
    onb = np.zeros(n, bool)
    be = 0
    for u in range(n):
        for v in adj[u]:
            if u < v and len(adj[u] & adj[v]) == 1:
                onb[u] = onb[v] = True
                be += 1
    deg = np.fromiter((len(s) for s in adj), np.int64, n)
    return n, be, int((~onb & (deg == 5)).sum()), int((~onb & (deg == 7)).sum())


def one(spec):
    beta, seed, nmax = spec
    random.seed(seed)
    np.random.seed(seed)
    rows = []
    marks = [2 ** k for k in range(12, 30) if 2 ** k <= nmax]
    grow_sheet(nmax, CAP, beta, seed, checkpoints=marks,
               on_checkpoint=lambda n, adj: rows.append(stats(adj)), variant='disloc')
    return beta, seed, rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=20)
    ap.add_argument('--nmax', type=int, default=2 ** 19)
    a = ap.parse_args()
    specs = [(b, s, a.nmax) for b in (0.5, 0.75) for s in range(a.seed, a.seed + 4)]
    with ProcessPoolExecutor(8) as ex:
        for beta, seed, rows in ex.map(one, specs):
            print(f"beta {beta} seed {seed}")
            for n, be, i5, i7 in rows:
                r = i7 - i5
                print(f"  N {n:8d}  boundary edges {be:5d}  i5 {i5:6d} i7 {i7:6d}  "
                      f"remainder {r:4d}  rem/boundary {r / be:.4f}  rem/N {r / n:.2e}",
                      flush=True)
