"""
Spectral-dimension flow d_s(t) from lazy-random-walk return probability.

Pre-registration: docs/superpowers/specs/2026-09-27-spectral-flow-preregistration.md
(all definitions and verdict rules below are frozen there).

The node-averaged return probability of the lazy walk is a trace,

    P(t) = Tr(W^t) / N,     W = (I + D^-1 A) / 2,

and W is similar to the symmetric positive-semidefinite
S = (I + D^-1/2 A D^-1/2) / 2, so for Rademacher probes z

    P(2k) = E ||S^k z||^2 / N.

Every term is a squared norm (no sign cancellation), and the estimate
self-averages over the N nodes, so a handful of probes suffices at large N.
The spectral dimension is the running log-slope

    D(t) = -2 dln P~ / dln t,     P~(t) = P(t) - 1/N,

which is flat at d_s on a graph that has a spectral dimension.

Usage:
    python spectral_flow.py --validate
    python spectral_flow.py --graphs grown6 grown7 grown8 --nodes 50000 200000 --seeds 3
    python spectral_flow.py --graphs pruned0.1 pruned0.3 pruned0.5 --nodes 200000
"""

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx
import scipy.sparse as sp
from tqdm import tqdm

from simulation import create_initial_graph

# Frozen by the pre-registration.
T_MAX = 20000
T_FIRST = 16
FLOOR_FACTOR = 20.0
PLATEAU_TOL = 0.2
PLATEAU_LEN = 5
FLOW_GAP = 0.3
RUNAWAY_RISE = 0.5
RUNAWAY_RANK = 0.9
MIN_POINTS = 9


def largest_component(G: nx.Graph) -> nx.Graph:
    cc = max(nx.connected_components(G), key=len)
    if len(cc) == G.number_of_nodes():
        return G
    return nx.convert_node_labels_to_integers(G.subgraph(cc).copy())


def lazy_operator(G: nx.Graph) -> sp.csr_array:
    """S = (I + D^-1/2 A D^-1/2) / 2 on a connected graph."""
    return lazy_operator_adjacency(nx.to_scipy_sparse_array(
        G, weight=None, format='csr', dtype=np.float64))


def lazy_operator_adjacency(A: sp.csr_array) -> sp.csr_array:
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = sp.diags_array(1.0 / np.sqrt(deg))
    n = A.shape[0]
    return (0.5 * (sp.eye_array(n, format='csr') + inv @ A @ inv)).tocsr()


def return_probability(S: sp.csr_array, t_max: int, n_probes: int,
                       rng: np.random.Generator, progress: bool = False
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """(t, P(t)) at even t = 2, 4, ..., t_max by the Hutchinson trace."""
    n = S.shape[0]
    Z = rng.choice([-1.0, 1.0], size=(n, n_probes))
    K = t_max // 2
    P = np.empty(K)
    it = range(K)
    if progress:
        it = tqdm(it, desc='walk', leave=False)
    for k in it:
        Z = S @ Z
        P[k] = float(np.mean(np.sum(Z * Z, axis=0))) / n
        # Past the admissibility floor nothing further is used: stop early.
        if P[k] - 1.0 / n < 0.25 * FLOOR_FACTOR / n:
            K = k + 1
            break
    return 2 * np.arange(1, K + 1), P[:K]


def exact_return_probability(S: sp.csr_array, ts: np.ndarray) -> np.ndarray:
    lam = np.linalg.eigvalsh(S.toarray())
    lam = np.clip(lam, 0.0, 1.0)
    return np.array([np.sum(lam ** t) for t in ts]) / S.shape[0]


def running_dimension(ts: np.ndarray, P: np.ndarray, n: int
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    (grid t_j, D(t_j)) on the admissible range.

    D(t_j) is minus twice the least-squares slope of ln P~ vs ln t over the
    computed t in [t_j/sqrt2, t_j*sqrt2]; a grid point is admissible when
    its whole window has P~ >= FLOOR_FACTOR/N and lies inside [.., t_max].
    """
    Pt = P - 1.0 / n
    grid, D = [], []
    j = 0
    while True:
        tj = T_FIRST * 2.0 ** (j / 2.0)
        lo, hi = tj / np.sqrt(2.0), tj * np.sqrt(2.0)
        if hi > ts[-1]:
            break
        m = (ts >= lo) & (ts <= hi)
        if m.sum() >= 3:
            if np.min(Pt[m]) < FLOOR_FACTOR / n:
                break
            slope = np.polyfit(np.log(ts[m]), np.log(Pt[m]), 1)[0]
            grid.append(tj)
            D.append(-2.0 * slope)
        j += 1
    return np.array(grid), np.array(D)


def plateaus(D: np.ndarray) -> List[Tuple[int, int, float]]:
    """Maximal runs (start, stop_exclusive, median) with spread <= tol."""
    out = []
    i, n = 0, len(D)
    while i < n:
        j = i + 1
        while j < n and (np.max(D[i:j + 1]) - np.min(D[i:j + 1])
                         <= PLATEAU_TOL):
            j += 1
        if j - i >= PLATEAU_LEN:
            out.append((i, j, float(np.median(D[i:j]))))
            i = j
        else:
            i += 1
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def classify(grid: np.ndarray, D: np.ndarray) -> Dict[str, object]:
    """The frozen verdict rules, in the frozen order."""
    if len(D) < MIN_POINTS:
        return {'verdict': 'INSUFFICIENT RANGE', 'value': float('nan'),
                'plateaus': []}
    pl = plateaus(D)
    spread = float(np.max(D) - np.min(D))
    if spread <= PLATEAU_TOL:
        return {'verdict': 'FLAT', 'value': float(np.median(D)),
                'plateaus': pl}
    if len(pl) >= 2 and pl[-1][1] == len(D):
        for a in pl[:-1]:
            if abs(pl[-1][2] - a[2]) > FLOW_GAP:
                return {'verdict': 'PLATEAU-TO-PLATEAU FLOW',
                        'value': (a[2], pl[-1][2]), 'plateaus': pl}
    tail = D[-PLATEAU_LEN:]
    tail_plateau = float(np.max(tail) - np.min(tail)) <= PLATEAU_TOL
    if (D[-1] - D[0] > RUNAWAY_RISE
            and _spearman(grid, D) > RUNAWAY_RANK and not tail_plateau):
        return {'verdict': 'RUNAWAY', 'value': float('nan'), 'plateaus': pl}
    return {'verdict': 'UNCLASSIFIED', 'value': float('nan'), 'plateaus': pl}


# --------------------------------------------------------------------------
# Graph builders
# --------------------------------------------------------------------------

def sierpinski_gasket(level: int) -> nx.Graph:
    """Sierpinski gasket graph; d_s = 2 ln3/ln5, d_H = ln3/ln2."""
    side = 2 ** level
    tri = [((0, 0), (side, 0), (0, side))]
    for _ in range(level):
        nxt = []
        for a, b, c in tri:
            ab = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            bc = ((b[0] + c[0]) // 2, (b[1] + c[1]) // 2)
            ca = ((c[0] + a[0]) // 2, (c[1] + a[1]) // 2)
            nxt += [(a, ab, ca), (ab, b, bc), (ca, bc, c)]
        tri = nxt
    G = nx.Graph()
    for a, b, c in tri:
        G.add_edges_from([(a, b), (b, c), (c, a)])
    return nx.convert_node_labels_to_integers(G)


def build(spec: str, n: int, seed: int) -> nx.Graph:
    random.seed(seed)
    np.random.seed(seed)
    if spec.startswith('grown'):
        G = create_initial_graph(n, topology='grown', k=int(spec[5:]),
                                 seed=seed)
    elif spec.startswith('pruned'):
        from prune_dimension import prune_to_convergence
        G = prune_to_convergence(create_initial_graph(
            n, topology='small_world', k=6, p=float(spec[6:]), seed=seed))
    elif spec == 'ring':
        G = nx.cycle_graph(n)
    elif spec == 'torus2':
        side = int(round(n ** 0.5))
        G = nx.grid_2d_graph(side, side, periodic=True)
    elif spec == 'torus3':
        side = int(round(n ** (1 / 3)))
        G = nx.grid_graph(dim=[side] * 3, periodic=True)
    elif spec == 'gasket':
        level = max(3, int(round(np.log(2 * n / 3) / np.log(3))) - 1)
        G = sierpinski_gasket(level)
    elif spec == 'regular3':
        G = nx.random_regular_graph(3, 2 * (n // 2), seed=seed)
    else:
        raise ValueError(f"unknown graph spec: {spec}")
    return largest_component(nx.convert_node_labels_to_integers(G))


def measure(spec: str, n: int, seed: int, n_probes: int, t_max: int = T_MAX,
            progress: bool = False) -> Dict[str, object]:
    G = build(spec, n, seed)
    S = lazy_operator(G)
    ts, P = return_probability(S, t_max, n_probes,
                               np.random.default_rng(seed + 7919), progress)
    grid, D = running_dimension(ts, P, S.shape[0])
    return {'spec': spec, 'N': S.shape[0], 'seed': seed, 'grid': grid,
            'D': D, **classify(grid, D)}


def measure_adjacency(name: str, A: sp.csr_array, seed: int, n_probes: int,
                      t_max: int = T_MAX, progress: bool = False
                      ) -> Dict[str, object]:
    """`measure` on a sparse adjacency matrix (connected graph assumed)."""
    S = lazy_operator_adjacency(sp.csr_array(A, dtype=np.float64))
    ts, P = return_probability(S, t_max, n_probes,
                               np.random.default_rng(seed + 7919), progress)
    grid, D = running_dimension(ts, P, S.shape[0])
    return {'spec': name, 'N': S.shape[0], 'seed': seed, 'grid': grid,
            'D': D, **classify(grid, D)}


def show(r: Dict[str, object]) -> None:
    grid, D = r['grid'], r['D']
    print(f"\n{r['spec']} seed {r['seed']}  (N={r['N']}, "
          f"{len(D)} admissible grid points"
          + (f", t = {grid[0]:.0f}..{grid[-1]:.0f})" if len(D) else ")"))
    if len(D):
        step = max(1, len(D) // 12)
        print("  D(t): " + "  ".join(f"{t:.0f}:{d:.2f}" for t, d in
                                     list(zip(grid, D))[::step]))
    for a, b, v in r['plateaus']:
        print(f"  plateau t = {grid[a]:.0f}..{grid[b - 1]:.0f}: {v:.3f}")
    val = r['value']
    extra = ''
    if r['verdict'] == 'FLAT':
        extra = f" at d_s = {val:.3f}"
    elif r['verdict'] == 'PLATEAU-TO-PLATEAU FLOW':
        extra = f" {val[0]:.2f} -> {val[1]:.2f}"
    print(f"  VERDICT: {r['verdict']}{extra}")


def _validate(seed: int, quick: bool) -> bool:
    ok = True
    print("[A1] Hutchinson trace vs exact spectrum (N <= 600)")
    for spec, n in (('torus2', 576), ('grown6', 600), ('ring', 400)):
        G = build(spec, n, seed)
        S = lazy_operator(G)
        ts, P = return_probability(S, 400, 64, np.random.default_rng(seed))
        Pe = exact_return_probability(S, ts)
        m = (Pe - 1.0 / S.shape[0]) >= FLOOR_FACTOR / S.shape[0]
        err = float(np.max(np.abs(P[m] / Pe[m] - 1.0)))
        good = err < 0.05
        ok &= good
        print(f"  {spec:>8} N={S.shape[0]}: max relative error {err:.4f} "
              f"over {m.sum()} times  {'OK' if good else 'FAIL'}")

    print("[A2] classifier on constructed curves")
    g = T_FIRST * 2.0 ** (np.arange(16) / 2.0)
    cases = (('flat', np.full(16, 2.0) + 0.05 * np.sin(np.arange(16)),
              'FLAT'),
             ('two plateaus', np.r_[np.full(7, 2.0), 2.5, 3.0,
                                    np.full(7, 3.5)],
              'PLATEAU-TO-PLATEAU FLOW'),
             ('runaway', 1.5 + 0.25 * np.arange(16), 'RUNAWAY'),
             ('short', np.full(6, 2.0), 'INSUFFICIENT RANGE'))
    for name, D, want in cases:
        v = classify(g[:len(D)], np.asarray(D, float))['verdict']
        good = v == want
        ok &= good
        print(f"  {name:>13}: {v}  {'OK' if good else 'FAIL'}")

    print("[A3] known answers")
    n = 20000 if quick else 100000
    # The 3D torus mixes fastest (P ~ t^-1.5), so it needs the most nodes
    # to keep a factor-16 admissible range.
    anchors = (('ring', n, 1.0), ('torus2', n, 2.0), ('torus3', 10 * n, 3.0),
               ('gasket', n, 2 * np.log(3) / np.log(5)))
    for spec, nn, truth in anchors:
        r = measure(spec, nn, seed, 8)
        show(r)
        good = r['verdict'] == 'FLAT' and abs(r['value'] - truth) <= 0.1
        ok &= good
        print(f"  -> expected FLAT at {truth:.3f}: "
              f"{'OK' if good else 'FAIL'}")
    r = measure('regular3', n, seed, 8)
    show(r)
    good = r['verdict'] != 'FLAT'
    ok &= good
    print(f"  -> negative control must not read FLAT: "
          f"{'OK' if good else 'FAIL'}")

    print("\nPASS: spectral-flow instrument" if ok
          else "\nFAIL: spectral-flow instrument")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Spectral-dimension flow from walk return probability",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--graphs', nargs='+',
                    default=['grown6', 'grown7', 'grown8'])
    ap.add_argument('--nodes', type=int, nargs='+', default=[50000, 200000])
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--probes', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.validate:
        raise SystemExit(0 if _validate(args.seed, args.quick) else 1)

    rows, results = [], []
    for spec in args.graphs:
        for n in args.nodes:
            for s in range(args.seeds):
                r = measure(spec, n, args.seed + s, args.probes,
                            progress=True)
                if r['N'] < 0.5 * n:
                    print(f"\n{spec} N={n} seed {args.seed + s}: only "
                          f"{r['N']} nodes -- skipped")
                    continue
                show(r)
                results.append((n, r))
                for t, d in zip(r['grid'], r['D']):
                    rows.append({'spec': spec, 'N_target': n, 'N': r['N'],
                                 'seed': r['seed'], 't': t, 'D': d,
                                 'verdict': r['verdict']})

    print("\nSummary")
    for n, r in results:
        print(f"  {r['spec']:>10} N={n:>7} seed {r['seed']}: {r['verdict']}"
              f"  ({len(r['D'])} pts)")
    if rows:
        Path("results").mkdir(exist_ok=True)
        path = f"results/spectral_flow_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Saved {path}")


if __name__ == '__main__':
    main()
