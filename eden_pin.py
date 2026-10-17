"""
Eden growth with one frozen site: the null model for a permanent pin (D1).

Pre-registration: docs/superpowers/specs/2026-10-17-wave2-preregistration.md (W2-e).

Eden (occupy a uniformly chosen perimeter site) on a triangular lattice graph,
using adjacency only: distances are hop distances from the seed, no node
positions are read. When the cluster's radius R (largest seed distance of an
occupied site) reaches R0, one perimeter site at distance R0 is frozen: it is
never occupied. Two variants:

  plain         nothing else changes; the frozen site may be enclosed.
  noenclose     an occupation that would cut the frozen site off from the
                outside is refused (the frozen site, like a sheet's dead
                corner, must stay on the edge). A refused site stays refused:
                occupying more sites can only make enclosure easier.

"Outside" = any unoccupied site at seed distance > R (beyond the cluster all
sites are unoccupied and connected to the lattice frame).

Usage:
    python eden_pin.py --validate
    python eden_pin.py --runs 20 --r-max 200
"""

import argparse
import csv
import random
import time
from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

R0 = 20
MARKS = (40, 80, 120, 160, 200)


def build_lattice(r_max: int) -> Tuple[List[np.ndarray], int, np.ndarray]:
    """Triangular lattice big enough for hop radius r_max + 15 around a
    central node; returns neighbour arrays, the seed, seed distances."""
    side = 2 * (r_max + 20)
    G = nx.triangular_lattice_graph(side, 2 * side)
    G = nx.convert_node_labels_to_integers(G, ordering='sorted')
    A = sp.csr_array(nx.to_scipy_sparse_array(G, format='csr', dtype=float))
    deg = np.diff(A.indptr)
    frame = np.nonzero(deg < 6)[0]
    dfr = dijkstra(A, indices=frame, unweighted=True, min_only=True)
    seed = int(np.argmax(dfr))                     # the most central node
    if dfr[seed] < r_max + 10:
        raise ValueError('lattice too small')
    dist = dijkstra(A, indices=seed, unweighted=True).astype(np.int32)
    nbrs = [A.indices[A.indptr[i]:A.indptr[i + 1]] for i in range(A.shape[0])]
    return nbrs, seed, dist


class Eden:
    def __init__(self, nbrs, seed: int, dist: np.ndarray, rng: np.random.Generator,
                 variant: str):
        self.nbrs, self.dist, self.rng, self.variant = nbrs, dist, rng, variant
        n = len(nbrs)
        self.occ = np.zeros(n, dtype=bool)
        self.per: List[int] = []
        self.idx = -np.ones(n, dtype=np.int64)
        self.forbidden: Set[int] = set()
        self.frozen: Optional[int] = None
        self.R = 0
        self.refused = 0
        self._occupy(seed)

    # perimeter bookkeeping ------------------------------------------------
    def _add_per(self, x: int) -> None:
        if self.idx[x] < 0 and not self.occ[x] and x != self.frozen \
                and x not in self.forbidden:
            self.idx[x] = len(self.per)
            self.per.append(x)

    def _drop_per(self, x: int) -> None:
        i = self.idx[x]
        if i < 0:
            return
        last = self.per[-1]
        self.per[i] = last
        self.idx[last] = i
        self.per.pop()
        self.idx[x] = -1

    def _occupy(self, x: int) -> None:
        self._drop_per(x)
        self.occ[x] = True
        self.R = max(self.R, int(self.dist[x]))
        for y in self.nbrs[x]:
            self._add_per(int(y))

    # enclosure ------------------------------------------------------------
    def reaches_outside(self, start: int, blocked: int = -1, limit: int = 10 ** 9
                        ) -> bool:
        """BFS through unoccupied sites (excluding `blocked`) from `start`;
        True on reaching a site beyond the cluster radius."""
        if start == blocked:
            return False
        seen = {start}
        q = deque([start])
        R = self.R
        while q:
            x = q.popleft()
            if self.dist[x] > R:
                return True
            for y in self.nbrs[x]:
                y = int(y)
                if y != blocked and not self.occ[y] and y not in seen:
                    seen.add(y)
                    q.append(y)
            if len(seen) > limit:
                return True
        return False

    def _splits_locally(self, s: int) -> bool:
        """Do s's unoccupied neighbours fall into >= 2 groups within its
        link? (If not, occupying s cannot disconnect the unoccupied region.)"""
        un = [int(y) for y in self.nbrs[s] if not self.occ[y]]
        if len(un) <= 1:
            return False
        unset = set(un)
        seen = {un[0]}
        stack = [un[0]]
        while stack:
            x = stack.pop()
            for y in self.nbrs[x]:
                y = int(y)
                if y in unset and y not in seen:
                    seen.add(y)
                    stack.append(y)
        return len(seen) < len(un)

    # one step -------------------------------------------------------------
    def step(self) -> bool:
        while self.per:
            s = self.per[int(self.rng.integers(len(self.per)))]
            if self.variant == 'noenclose' and self.frozen is not None \
                    and self._splits_locally(s) \
                    and not self.reaches_outside(self.frozen, blocked=s):
                self._drop_per(s)
                self.forbidden.add(s)
                self.refused += 1
                continue
            self._occupy(s)
            if self.frozen is None and self.R >= R0:
                cand = [x for x in self.per if self.dist[x] == R0]
                if not cand:
                    cand = list(self.per)
                self.frozen = int(cand[int(self.rng.integers(len(cand)))])
                self._drop_per(self.frozen)
            return True
        return False

    # measurements -------------------------------------------------------------
    def exposed(self) -> bool:
        return self.frozen is not None and self.reaches_outside(self.frozen)

    def channel_width(self) -> int:
        rho = (R0 + self.R) // 2
        cnt = 0
        cand = np.nonzero((self.dist == rho) & ~self.occ)[0]
        for x in cand.tolist():
            if x == self.frozen or self.reaches_outside(x, limit=200000):
                cnt += 1
        return cnt


def run_one(nbrs, seed, dist, variant: str, run_seed: int, r_max: int,
            marks: Sequence[int] = MARKS) -> List[Dict]:
    rng = np.random.default_rng(run_seed)
    e = Eden(nbrs, seed, dist, rng, variant)
    rows = []
    todo = [m for m in marks if m <= r_max]
    while todo and e.step():
        if e.R >= todo[0]:
            m = todo.pop(0)
            ex = e.exposed()
            rows.append({'variant': variant, 'run': run_seed, 'R': m,
                         'sites': int(e.occ.sum()), 'exposed': int(ex),
                         'width_half_depth': e.channel_width() if ex else 0,
                         'refused': e.refused})
    return rows


def report(rows: List[Dict]) -> Dict[str, str]:
    out = {}
    for variant in ('plain', 'noenclose'):
        rs = [r for r in rows if r['variant'] == variant]
        runs = sorted({r['run'] for r in rs})
        print(f"\n{variant}: {len(runs)} runs")
        for m in MARKS:
            at = [r for r in rs if r['R'] == m]
            if not at:
                continue
            ex = [r['exposed'] for r in at]
            w = [r['width_half_depth'] for r in at if r['exposed']]
            print(f"  R {m:3d}: exposed {sum(ex)}/{len(at)}"
                  + (f", width at half depth median {np.median(w):.0f} "
                     f"(10/90%: {np.percentile(w, 10):.0f}/{np.percentile(w, 90):.0f})"
                     if w else ''))
        if variant == 'plain':
            at40 = [r for r in rs if r['R'] == 40]
            healed = sum(1 for r in at40 if not r['exposed'])
            v = 'PLAIN EDEN HEALS' if at40 and healed >= 0.95 * len(at40) else \
                'PLAIN EDEN DOES NOT HEAL'
            print(f"  -> {v} ({healed}/{len(at40)} enclosed by R = 40)")
        else:
            last = [r for r in rs if r['R'] == max(MARKS)]
            alive = all(r['exposed'] for r in last) and len(last) == len(runs)
            thin = sum(1 for r in last if r['width_half_depth'] <= 3)
            v = 'CONSTRAINED EDEN CRACKS' if alive and thin >= 0.9 * len(last) else \
                'CONSTRAINED EDEN: NOT A THIN CRACK'
            print(f"  -> {v} (channel alive at R = 200 in {sum(r['exposed'] for r in last)}"
                  f"/{len(runs)}; width <= 3 in {thin}/{len(last)})")
        out[variant] = v
    return out


def _validate() -> bool:
    ok = True
    nbrs, seed, dist = build_lattice(60)
    rng0 = np.random.default_rng(1)
    e0 = Eden(nbrs, seed, dist, rng0, 'plain')
    while e0.R < 60:
        e0.step()
    inner = dist <= 30
    fill = float(e0.occ[inner].mean())
    good = fill >= 0.99
    print(f"[{'PASS' if good else 'FAIL'}] Eden cluster is compact: at R = 60 "
          f"{fill:.4f} of sites within seed distance 30 are occupied "
          f"({int(e0.occ.sum())} sites = {e0.occ.sum() / 60 ** 2:.2f} R^2)")
    ok &= good
    # Local split test against the global BFS on random cluster states.
    rng = np.random.default_rng(3)
    e = Eden(nbrs, seed, dist, rng, 'plain')
    for _ in range(3000):
        e.step()
    bad = 0
    for s in list(e.per)[:400]:
        if not e._splits_locally(s):
            # Occupying s must not cut any unoccupied neighbour from outside.
            un = [int(y) for y in e.nbrs[s] if not e.occ[y]]
            reach_before = [e.reaches_outside(u) for u in un]
            reach_after = [e.reaches_outside(u, blocked=s) for u in un]
            bad += sum(1 for a, b in zip(reach_before, reach_after) if a and not b)
    good = bad == 0
    print(f"[{'PASS' if good else 'FAIL'}] local split test never misses a "
          f"disconnection (400 perimeter sites, {bad} misses)")
    ok &= good
    rows = run_one(nbrs, seed, dist, 'noenclose', 2, 60, marks=(30, 45, 60))
    good = all(r['exposed'] for r in rows)
    print(f"[{'PASS' if good else 'FAIL'}] no-enclosure variant keeps the frozen site "
          f"exposed at every mark ({[r['exposed'] for r in rows]}, refused "
          f"{rows[-1]['refused']})")
    ok &= good
    print('\nVALIDATION', 'PASSED' if ok else 'FAILED')
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--runs', type=int, default=20)
    ap.add_argument('--r-max', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0, help='first run seed')
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.validate:
        raise SystemExit(0 if _validate() else 1)
    t0 = time.time()
    nbrs, seed, dist = build_lattice(args.r_max)
    print(f"lattice: {len(nbrs)} nodes, seed {seed}  [{time.time() - t0:.0f} s]",
          flush=True)
    rows: List[Dict] = []
    for variant in ('plain', 'noenclose'):
        for k in range(args.runs):
            t1 = time.time()
            rs = run_one(nbrs, seed, dist, variant, args.seed + k, args.r_max)
            rows += rs
            print(f"{variant} run {k}: " + "  ".join(
                f"R{r['R']}:{'open' if r['exposed'] else 'shut'}/{r['width_half_depth']}"
                for r in rs) + f"  refused {rs[-1]['refused'] if rs else 0}"
                f"  [{time.time() - t1:.0f} s]", flush=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    out = f"results/eden_pin_{stamp}.csv"
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved: {out}")
    report(rows)


if __name__ == '__main__':
    main()
