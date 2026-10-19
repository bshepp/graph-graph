"""Reproduction: the growing geometrical network of Wu, Menichetti, Rahmede and
Bianconi (2015, arXiv 1412.3405), the nearest published relative of `sheet`.

Their rule (verified from the full text, research_notes/Supporting literature
verification/geometry_theorems.md): start from a triangle. Every step
  (a) pick an unsaturated link (in fewer than m triangles) uniformly and glue a
      triangle to it with a new node;
  (b) with probability p, pick an unsaturated link (i, j) uniformly, then an
      unsaturated link (j, r) with r not already in a triangle with i and j; add
      link (i, r) with every triangle it closes, provided no link then exceeds m
      triangles (otherwise nothing happens; the text leaves the all-or-nothing
      reading slightly open). For m = 2 the paper says (b) adds one link and one
      triangle, so only (i, j, r) is added; the literal all-closing reading folds the
      first few triangles into a closed tetrahedron at p = 0.9.
The cap is on triangles per link, not on degree, which is the one difference from
`sheet`'s move set that the survey identifies.

Published results checked here: for m = 2 the degree distribution is exponential,
P(k) = (1/2)(2/3)^(k - 1) (p = 0); mean distance grows as log N (infinite
Hausdorff dimension). Also measured with the project's own instruments: ball
growth around the first node and the boundary/size ratio.

Usage:
    python wu_network.py [--m 2] [--p 0.0] [--seed 0]
"""
from __future__ import annotations

import argparse
import math
import random
from collections import Counter, deque

import numpy as np


def grow(n_max: int, m: int, p: float, rng: random.Random):
    adj = [set([1, 2]), set([0, 2]), set([0, 1])]
    tri = {}            # edge (a<b) -> triangle count
    unsat = []          # unsaturated edges
    pos = {}

    def key(a, b):
        return (a, b) if a < b else (b, a)

    def register(e):
        if e not in tri:
            tri[e] = 0
            pos[e] = len(unsat)
            unsat.append(e)

    def bump(e):
        register(e)
        tri[e] += 1
        if tri[e] >= m and e in pos:
            i = pos.pop(e)
            last = unsat.pop()
            if i < len(unsat):
                unsat[i] = last
                pos[last] = i

    for e in ((0, 1), (0, 2), (1, 2)):
        bump(e)
    while len(adj) < n_max:
        a, b = unsat[rng.randrange(len(unsat))]
        w = len(adj)
        adj.append({a, b})
        adj[a].add(w)
        adj[b].add(w)
        bump(key(a, b))
        bump(key(a, w))
        bump(key(b, w))
        if p > 0 and rng.random() < p:
            i, j = unsat[rng.randrange(len(unsat))]
            if rng.random() < 0.5:
                i, j = j, i
            common = adj[i] & adj[j]
            cand = [r for r in adj[j] if r != i and r not in common
                    and key(j, r) in pos]
            if cand:
                r = cand[rng.randrange(len(cand))]
                # m = 2: the paper states (b) adds "only one new link and one new
                # triangle" (Delta chi = 0), i.e. the triangle (i, j, r); the literal
                # all-closing reading folds the early disc into a tetrahedron.
                closes = {j} if m == 2 else adj[i] & adj[r]
                new = {key(i, r): len(closes)}
                for c in closes:
                    for e in (key(i, c), key(r, c)):
                        new[e] = new.get(e, 0) + 1
                if all(tri.get(e, 0) + k <= m for e, k in new.items()):
                    adj[i].add(r)
                    adj[r].add(i)
                    register(key(i, r))
                    for e, k in new.items():
                        for _ in range(k):
                            bump(e)
    return adj


def bfs(adj, s):
    d = [-1] * len(adj)
    d[s] = 0
    q = deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if d[v] < 0:
                d[v] = d[u] + 1
                q.append(v)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--m', type=int, default=2)
    ap.add_argument('--p', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    rng = random.Random(a.seed)
    print(f'Wu et al. growing network, m = {a.m}, p = {a.p}, seed {a.seed}')
    print('     N   mean dist (20 sources)   ln N   ratio')
    for n in (1000, 4000, 16000, 64000, 256000):
        adj = grow(n, a.m, a.p, random.Random(a.seed * 1000 + n))
        src = rng.sample(range(n), 20)
        md = np.mean([np.mean(bfs(adj, s)) for s in src])
        print(f'{n:7d}   {md:8.2f}               {math.log(n):5.2f}  {md / math.log(n):.3f}', flush=True)
    deg = Counter(len(s) for s in adj)
    print('degree distribution at N = 256000 (measured vs (1/2)(2/3)^(k-1) for m = 2):')
    for k in range(2, 14):
        th = 0.5 * (2 / 3) ** (k - 1) if a.m == 2 else float('nan')
        print(f'  k {k:2d}: {deg.get(k, 0) / len(adj):.5f}   {th:.5f}')
    d = bfs(adj, 0)
    h = np.bincount([x for x in d if x >= 0])
    V = np.cumsum(h)
    print('ball volume around node 0, V(r):', V[:16].tolist())
    print('successive ratios V(r+1)/V(r):', [round(V[r + 1] / V[r], 2) for r in range(1, 15)])


if __name__ == '__main__':
    main()
