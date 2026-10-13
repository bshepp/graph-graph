"""
What nucleates the arm? -- the first tree-like protrusion of a `sheet`.

Pre-registration: docs/superpowers/specs/2026-10-13-round2-preregistrations.md (item 1).

A `sheet` at beta = 2 stays compact and then grows an arm within one octave of
size. Node index is growth order, so the graph at size N is approximated by
the subgraph induced on nodes < N (closure edges added later are included;
boundary biased low by a few percent -- disclosed). This tool finds the first
arm and characterises its base.

Usage:
    python arm_birth.py --validate
    python arm_birth.py results/sheet_cross_b2_s300_N4194304.npz [...]
"""

import argparse
import os
import random
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra

from sheet_growth import grow_sheet, load_csr, _closable

ARM_FACTOR = 1.5      # detection: boundary nodes beyond 1.5 R
BASE_FACTOR = 1.2     # base: lowest node of the arm's component above 1.2 R
ARM_MIN = 50
N_START = 2 ** 17
STEP = 2 ** 0.125
HOPS = 20


def sets_to_csr(adj: List[set]) -> sp.csr_array:
    n = len(adj)
    rows = [u for u, s in enumerate(adj) for _ in s]
    cols = [v for s in adj for v in s]
    return sp.csr_array((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def adjacency_sets(A: sp.csr_array) -> List[set]:
    A = A.tocsr()
    return [set(A.indices[A.indptr[i]:A.indptr[i + 1]].tolist())
            for i in range(A.shape[0])]


def boundary_nodes(adj: List[set]) -> np.ndarray:
    out = []
    for u, nb in enumerate(adj):
        for v in nb:
            if u < v and len(nb & adj[v]) == 1:
                out.append(u)
                out.append(v)
                break
    return np.unique(np.array(out, dtype=int))


def find_arm(A: sp.csr_array, n: int) -> Optional[Dict]:
    """The largest arm at prefix size n, or None."""
    P = A[:n, :n].tocsr()
    adj = adjacency_sets(P)
    bnd = boundary_nodes(adj)
    rho = dijkstra(P, indices=0, unweighted=True)
    R = float(np.median(rho[bnd]))
    far = bnd[rho[bnd] > ARM_FACTOR * R]
    if far.size < ARM_MIN:
        return None
    sub = P[far][:, far]
    _, lab = connected_components(sub, directed=False)
    sizes = np.bincount(lab)
    big = int(np.argmax(sizes))
    if sizes[big] < ARM_MIN:
        return None
    arm = far[lab == big]
    # walk the arm back toward the disc: its component among boundary nodes
    # above BASE_FACTOR * R, whose lowest node is the base
    lo = bnd[rho[bnd] > BASE_FACTOR * R]
    sub2 = P[lo][:, lo]
    _, lab2 = connected_components(sub2, directed=False)
    pos = {int(v): i for i, v in enumerate(lo)}
    comp = lab2[pos[int(arm[0])]]
    arm = lo[lab2 == comp]
    base = int(arm[np.argmin(rho[arm])])
    extent = float(rho[arm].max() - rho[arm].min() + 1)
    return {'n': n, 'R': R, 'arm_nodes': arm, 'base': base,
            'base_rho': float(rho[base]), 'arm_size': int(arm.size),
            'width': float(arm.size / extent), 'adj': adj, 'bnd': bnd,
            'rho': rho, 'P': P}


def first_arm(A: sp.csr_array, n_start: int = N_START) -> Optional[Dict]:
    n = float(n_start)
    n_max = A.shape[0]
    while n <= n_max:
        found = find_arm(A, int(n))
        if found is not None:
            return found
        n *= STEP
    return None


def base_census(found: Dict, c: int = 6) -> Dict[str, object]:
    """Blocked-closure density and degrees near the base vs the rest."""
    adj, bnd, base, P = found['adj'], found['bnd'], found['base'], found['P']
    d_base = dijkstra(P, indices=base, unweighted=True, limit=HOPS + 0.5)
    near = bnd[d_base[bnd] <= HOPS]
    rest = bnd[d_base[bnd] > HOPS]

    def blocked(nodes):
        hi = [v for v in nodes if len(adj[v]) >= c]
        blk = [v for v in hi if _closable(adj, v, c) is None]
        return len(blk), len(hi), len(nodes)

    bn, hn, nn = blocked(near)
    br, hr, nr = blocked(rest)
    deg_near = np.array([len(adj[v]) for v in near], dtype=int)
    deg_rest = np.array([len(adj[v]) for v in rest], dtype=int)
    dn, dr = bn / max(nn, 1), br / max(nr, 1)
    return {'near_boundary': nn, 'near_blocked': bn, 'near_deg6': hn,
            'rest_boundary': nr, 'rest_blocked': br, 'rest_deg6': hr,
            'blocked_density_near': dn, 'blocked_density_rest': dr,
            'ratio': dn / max(dr, 1e-12),
            'mean_deg_near': float(deg_near.mean()) if nn else float('nan'),
            'mean_deg_rest': float(deg_rest.mean()) if nr else float('nan'),
            'deg_hist_near': np.bincount(deg_near, minlength=7)[:7].tolist(),
            'deg_hist_rest': np.bincount(deg_rest, minlength=7)[:7].tolist()}


def analyze(paths: Sequence[str]) -> List[Dict]:
    out = []
    for path in paths:
        A = load_csr(path)
        name = os.path.basename(path)
        f = first_arm(A)
        if f is None:
            print(f"{name}: no arm up to N = {A.shape[0]}")
            out.append({'graph': name, 'arm': False})
            continue
        cs = base_census(f)
        print(f"\n{name}: first arm at N = {f['n']:,} (R = {f['R']:.0f}, "
              f"base at rho = {f['base_rho']:.0f}, arm {f['arm_size']} "
              f"boundary nodes, width {f['width']:.1f})")
        print(f"  blocked closures within {HOPS} hops of the base: "
              f"{cs['near_blocked']}/{cs['near_boundary']} boundary nodes "
              f"(density {cs['blocked_density_near']:.4f}); elsewhere "
              f"{cs['rest_blocked']}/{cs['rest_boundary']} "
              f"(density {cs['blocked_density_rest']:.4f}); ratio "
              f"{cs['ratio']:.2f}")
        print(f"  boundary degree histogram 0..6 near base "
              f"{cs['deg_hist_near']}  rest {cs['deg_hist_rest']}  "
              f"(mean {cs['mean_deg_near']:.2f} vs {cs['mean_deg_rest']:.2f})")
        out.append({'graph': name, 'arm': True, 'n_first': f['n'],
                    'R': f['R'], 'base_rho': f['base_rho'],
                    'width': f['width'], **cs})
    arms = [o for o in out if o['arm']]
    if arms:
        ratios = [o['ratio'] for o in arms]
        k = len(arms)
        need = 3 if k <= 4 else int(np.ceil(2 * k / 3))
        n_def = sum(r >= 3.0 for r in ratios)
        n_not = sum(r < 1.5 for r in ratios)
        if n_def >= need:
            v = 'DEFECT'
        elif n_not >= need:
            v = 'NOT A DEFECT'
        else:
            v = 'UNDECIDED'
        print("\n  blocked-closure ratios at the base: "
              + " ".join(f"{r:.2f}" for r in ratios))
        print(f"  VERDICT (frozen rules, {k} arms): {v}")
    return out


def _validate() -> bool:
    ok = True
    print("[1] a compact sheet has no arm; a planted strip is found at its base")
    adj = grow_sheet(40000, 6, 3.0, seed=11)
    n0 = len(adj)
    A = sets_to_csr(adj)
    f = find_arm(A, n0)
    good = f is None
    ok &= good
    print(f"  compact N=40000: arm found = {f is not None}  "
          f"{'OK' if good else 'FAIL'}")
    rho = dijkstra(A, indices=0, unweighted=True)
    bset = boundary_nodes(adj)
    start = int(bset[np.argmax(rho[bset])])
    nb = next(iter(adj[start]))
    adj2 = [set(s) for s in adj]
    prev2, prev1 = nb, start
    for i in range(300):
        w = n0 + i
        adj2.append({prev1, prev2})
        adj2[prev1].add(w)
        adj2[prev2].add(w)
        prev2, prev1 = prev1, w
    A2 = sets_to_csr(adj2)
    f = find_arm(A2, len(adj2))
    good = (f is not None and f['arm_size'] >= ARM_MIN
            and f['base_rho'] <= max(rho[start], BASE_FACTOR * f['R']) + 3)
    ok &= good
    print(f"  planted strip: arm found = {f is not None}"
          + (f", base rho {f['base_rho']:.0f} vs planted {rho[start]:.0f}"
             if f else '') + f"  {'OK' if good else 'FAIL'}")
    print("\nPASS: arm-birth instrument" if ok
          else "\nFAIL: arm-birth instrument")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="First arm of a sheet: where it starts and what is there")
    ap.add_argument('paths', nargs='*')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.validate:
        raise SystemExit(0 if _validate() else 1)
    if not args.paths:
        ap.error('give saved sheet .npz files')
    analyze(args.paths)


if __name__ == '__main__':
    main()
