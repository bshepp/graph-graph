"""
Window-stability audit of the ball-growth dimension: is `d_eff` a dimension,
or a reading of one fit window?

`dimension.local_dimension` fits  log|B(r)| = d*log r + c + a/r  over radii
1..max_radius and gates on R^2. A high R^2 certifies that a curve was fitted,
not that the growth is a power law: slow EXPONENTIAL growth (|B| ~ b^r with b
close to 1) is fitted at R^2 > 0.97 over any one window, and returns a
confident, window-dependent "dimension". This driver asks the two questions
the R^2 gate cannot:

  1. WINDOW DRIFT -- re-run the project's own estimator at several
     max_radius values. A real dimension does not move; an exponential
     reads higher with every larger window.
  2. MODEL COMPARISON -- on the mean ball-growth curve, residual sum of
     squares of a power law vs an exponential over the same radii, each
     with the same 1/r correction term (equal parameter counts).

Verdict per graph (both must hold for DIMENSION DEFINED):
  drift  = max - min of median d_eff over windows   <= DRIFT_TOL (0.15)
  across windows spanning at least a factor 2 in radius, and
  RSS(power law + 1/r) < RSS(exponential).

`--validate` checks the instrument on known answers: 2D and 3D tori must read
DEFINED at the right value, and a subdivided random 3-regular graph
(exponential growth by construction, base 2^(1/6)) must read EXPONENTIAL.

Usage:
    python window_stability.py --validate
    python window_stability.py --nodes 200000 --seed 0
    python window_stability.py --nodes 200000 --graphs grown6 pruned0.3
"""

import argparse
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from simulation import create_initial_graph
from dimension import local_dimension

WINDOWS = (6, 8, 10, 12, 16, 20, 24, 28, 32, 40)
DRIFT_TOL = 0.15
SATURATION = 0.1          # the estimator's own saturation fraction


def adjacency(G: nx.Graph) -> sp.csr_array:
    return nx.to_scipy_sparse_array(G, weight=None, format='csr')


def ball_curves(A: sp.csr_array, n_sources: int, r_max: int,
                rng: np.random.Generator) -> np.ndarray:
    """|B(v, r)| for r = 1..r_max, one row per sampled source."""
    n = A.shape[0]
    src = rng.choice(n, min(n_sources, n), replace=False)
    D = dijkstra(A, indices=src, unweighted=True)
    return np.array([[np.count_nonzero(row <= r) for r in range(1, r_max + 1)]
                     for row in D], dtype=float)


def window_table(B: np.ndarray, n: int) -> List[Tuple[int, float, float, int]]:
    """(max_radius, median d_eff, median R^2, n_defined) per usable window.

    A window is used only by sources whose ball is unsaturated at every one
    of its radii, so every row of the table is fitted over exactly the radii
    it is labelled with.
    """
    out = []
    for mr in WINDOWS:
        if mr > B.shape[1]:
            break
        ds, r2s = [], []
        for row in B:
            if row[mr - 1] >= SATURATION * n:
                continue
            d, r2 = local_dimension(
                [(r + 1, int(c)) for r, c in enumerate(row[:mr])], n)
            if np.isfinite(d):
                ds.append(d)
                r2s.append(r2)
        if len(ds) >= max(5, B.shape[0] // 4):
            out.append((mr, float(np.median(ds)), float(np.median(r2s)),
                        len(ds)))
    return out


def model_comparison(B: np.ndarray, n: int, r_min: int = 6) -> Dict[str, float]:
    """RSS of power-law(+1/r) vs exponential on the geometric-mean curve."""
    mB = np.exp(np.log(B).mean(axis=0))
    r = np.arange(1, B.shape[1] + 1, dtype=float)
    m = (r >= r_min) & (mB < SATURATION * n)
    if m.sum() < 5:
        return {'n_pts': int(m.sum())}
    lr, lB, rr = np.log(r[m]), np.log(mB[m]), r[m]

    def fit(X):
        c = np.linalg.lstsq(X, lB, rcond=None)[0]
        return float(np.sum((lB - X @ c) ** 2)), c

    one = np.ones_like(rr)
    rss_p, cp = fit(np.column_stack([lr, one, 1.0 / rr]))
    # Same parameter count on both sides: each model gets a 1/r correction.
    rss_e, ce = fit(np.column_stack([rr, one, 1.0 / rr]))
    return {'n_pts': int(m.sum()), 'r_lo': int(rr[0]), 'r_hi': int(rr[-1]),
            'rss_power': rss_p, 'd_power': float(cp[0]),
            'rss_exp': rss_e, 'base_exp': float(np.exp(ce[0]))}


def audit(name: str, G: nx.Graph, n_sources: int, seed: int) -> Dict:
    return audit_adjacency(name, adjacency(G), n_sources, seed)


def audit_adjacency(name: str, A: sp.csr_array, n_sources: int,
                    seed: int) -> Dict:
    """`audit` on a sparse adjacency matrix (connected graph assumed)."""
    n = A.shape[0]
    B = ball_curves(A, n_sources, WINDOWS[-1], np.random.default_rng(seed))
    tab = window_table(B, n)
    mc = model_comparison(B, n)
    verdict = 'INSUFFICIENT RANGE'
    drift = float('nan')
    if len(tab) >= 2 and tab[-1][0] >= 2 * tab[0][0] and mc['n_pts'] >= 5:
        ds = [t[1] for t in tab]
        drift = max(ds) - min(ds)
        power_wins = mc['rss_power'] < mc['rss_exp']
        if drift <= DRIFT_TOL and power_wins:
            verdict = 'DIMENSION DEFINED'
        elif drift > DRIFT_TOL and not power_wins:
            verdict = 'EXPONENTIAL (no dimension)'
        else:
            verdict = 'MIXED'
    print(f"\n{name}  (N={n})")
    print("  window:  " + "  ".join(f"R{t[0]}={t[1]:.2f}" for t in tab))
    print("  R^2:     " + "  ".join(f"R{t[0]}={t[2]:.3f}" for t in tab))
    if mc['n_pts'] >= 5:
        print(f"  radii {mc['r_lo']}..{mc['r_hi']}: RSS power+1/r "
              f"{mc['rss_power']:.4f} (d={mc['d_power']:.2f})   RSS "
              f"exponential+1/r {mc['rss_exp']:.4f} (base {mc['base_exp']:.3f})")
    print(f"  drift = {drift:.2f}   ->  {verdict}")
    return {'name': name, 'N': n, 'table': tab, 'drift': drift,
            'verdict': verdict, **mc}


# --------------------------------------------------------------------------
# Graph builders
# --------------------------------------------------------------------------

def subdivided_regular(n0: int, seg: int, seed: int) -> nx.Graph:
    """Random 3-regular graph with every edge replaced by a path of `seg` edges.

    Locally a tree with branching 2 and no boundary, so ball growth is
    exponential with base 2^(1/seg) by construction -- the slow-exponential
    known answer the R^2 gate cannot reject. (A finite balanced tree is the
    wrong control: most of its nodes sit next to the leaves.)
    """
    T = nx.random_regular_graph(3, n0, seed=seed)
    G = nx.Graph()
    nxt = T.number_of_nodes()
    for u, v in T.edges():
        prev = u
        for _ in range(seg - 1):
            G.add_edge(prev, nxt)
            prev = nxt
            nxt += 1
        G.add_edge(prev, v)
    return G


def build(spec: str, n: int, seed: int) -> nx.Graph:
    random.seed(seed)
    np.random.seed(seed)
    if spec.startswith('grown'):
        return create_initial_graph(n, topology='grown', k=int(spec[5:]),
                                    seed=seed)
    if spec.startswith('pruned'):
        from prune_dimension import prune_to_convergence
        G = create_initial_graph(n, topology='small_world', k=6,
                                 p=float(spec[6:]), seed=seed)
        G = prune_to_convergence(G)
        cc = max(nx.connected_components(G), key=len)
        return nx.convert_node_labels_to_integers(G.subgraph(cc).copy())
    if spec == 'torus2':
        side = int(round(n ** 0.5))
        return nx.convert_node_labels_to_integers(
            nx.grid_2d_graph(side, side, periodic=True))
    if spec == 'torus3':
        side = int(round(n ** (1 / 3)))
        return nx.convert_node_labels_to_integers(
            nx.grid_graph(dim=[side] * 3, periodic=True))
    if spec == 'tree':
        n0 = 2 * int(n / (1 + 1.5 * 5) / 2)
        return subdivided_regular(n0, 6, seed)
    raise ValueError(f"unknown graph spec: {spec}")


def _validate(seed: int) -> bool:
    ok = True
    cases = (('torus2', 160000, 'DIMENSION DEFINED', 2.0),
             ('torus3', 125000, 'DIMENSION DEFINED', 3.0),
             ('tree', 200000, 'EXPONENTIAL (no dimension)', None))
    for spec, n, want, d_true in cases:
        r = audit(spec, build(spec, n, seed), 40, seed)
        good = r['verdict'] == want
        if good and d_true is not None:
            good = abs(r['table'][-1][1] - d_true) < 0.1
        if good and spec == 'tree':
            good = abs(r['base_exp'] - 2 ** (1 / 6)) < 0.02
        ok &= good
        print(f"  -> expected {want}: {'OK' if good else 'FAIL'}")
    print("\nPASS: window-stability instrument" if ok
          else "\nFAIL: window-stability instrument")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Window-stability audit of the ball-growth dimension",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--nodes', type=int, default=200000)
    ap.add_argument('--graphs', nargs='+',
                    default=['torus2', 'grown6', 'grown7', 'grown8',
                             'pruned0.1', 'pruned0.3', 'pruned0.5'],
                    help='torus2 | torus3 | tree | grown<cap> | pruned<p>')
    ap.add_argument('--sources', type=int, default=60)
    ap.add_argument('--seeds', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.validate:
        raise SystemExit(0 if _validate(args.seed) else 1)

    out = []
    for spec in args.graphs:
        for s in range(args.seeds):
            seed = args.seed + s
            G = build(spec, args.nodes, seed)
            if G.number_of_nodes() < 0.05 * args.nodes:
                print(f"\n{spec} seed {seed}: only {G.number_of_nodes()} "
                      f"nodes (growth extinct) -- skipped")
                continue
            out.append(audit(f"{spec} seed {seed}", G, args.sources, seed))

    print("\nSummary")
    print(f"  {'graph':>20} {'N':>8} {'drift':>6}  verdict")
    for r in out:
        print(f"  {r['name']:>20} {r['N']:>8} {r['drift']:>6.2f}  "
              f"{r['verdict']}")


if __name__ == '__main__':
    main()
