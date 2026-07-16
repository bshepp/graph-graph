"""
Shortcut tolerance of coherent emergent geometry.

Question: how many long-range shortcuts ("wormholes") can a geometric graph
host before it stops being a geometry? FINDINGS.md established that `rewire`
(which keeps adding shortcuts forever) destroys dimensional structure and
`prune` (which deletes them) recovers it -- this experiment measures the
transition quantitatively by injecting a CONTROLLED number k of shortcuts
into a coherent base graph and measuring the dimension field's survival:

  * defined_frac  -- fraction of nodes where d_eff is still defined
  * Moran's I     -- spatial coherence of the surviving field
  * mean pair distance -- the small-world collapse, for contrast

Methodological note: max_radius is calibrated ONCE on the k=0 base graph and
held fixed for all k. Auto-calibrating per graph would shrink the measurement
radius as shortcuts collapse the diameter, confounding "geometry damaged"
with "we measured at a different scale".

The damage model tested: each shortcut endpoint corrupts the balls of nodes
within max_radius of it (their ball growth jumps through the portal), so the
early decay should follow

    defined_frac(k) ~ 1 - c * (2k * B_r0 / N),   B_r0 = mean ball size at r0

i.e. tolerance is set by the ratio of "damage zones" to graph volume, and
k* ~ N / B_r0 shortcuts should destroy coherence. Both the curve and the
fitted c are reported; the N-sweep checks the law is about density, not size.

Usage:
    python shortcut_tolerance.py --quick                      # smoke test
    python shortcut_tolerance.py --nodes 2000 --seeds 3
    python shortcut_tolerance.py --nodes 2000 8000 --topology grown --cap 6
"""

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import networkx as nx
from tqdm import tqdm

from simulation import create_initial_graph
from coherence import field_coherence
from dimension import _estimate_max_radius_sparse
from shortcuts import inject_shortcuts, mean_pair_distance


def measure_tolerance(n_nodes: int, topology: str, cap: int,
                      counts: List[int], seed: int,
                      n_perm: int = 199,
                      min_distance_factor: float = 2.0) -> List[Dict]:
    """
    Survival curve for one (N, topology, seed): inject k shortcuts for each
    k in `counts` (fresh base graph each time) and measure the field.
    """
    rows = []

    # Base graph and the FIXED measurement radius.
    random.seed(seed)
    np.random.seed(seed)
    base = create_initial_graph(n_nodes, topology=topology, k=cap, seed=seed)
    A0 = nx.to_scipy_sparse_array(base, weight=None, format='csr',
                                  dtype=np.float32)
    r0 = _estimate_max_radius_sparse(A0)
    min_dist = int(np.ceil(min_distance_factor * r0)) + 1

    # Mean ball volume at r0 on the base graph (for the damage model),
    # from a node sample.
    sample = np.random.choice(list(base.nodes()),
                              min(100, n_nodes), replace=False)
    ball_sizes = []
    for u in sample:
        dist = nx.single_source_shortest_path_length(base, u, cutoff=r0)
        ball_sizes.append(len(dist))
    b_r0 = float(np.mean(ball_sizes))

    for k in tqdm(counts, desc=f"N={n_nodes} seed={seed}", leave=False):
        random.seed(seed * 100_003 + k)
        np.random.seed(seed * 100_003 + k)
        G = base.copy()
        placed = inject_shortcuts(G, k, min_distance=min_dist)
        res = field_coherence(G, max_radius=r0, n_perm=n_perm)
        rows.append({
            'n_nodes': n_nodes, 'topology': topology, 'cap': cap,
            'seed': seed, 'k': k, 'k_placed': len(placed),
            'max_radius': r0, 'min_distance': min_dist, 'ball_r0': b_r0,
            'damage_load': 2 * len(placed) * b_r0 / n_nodes,
            'defined_frac': res['defined_frac'],
            'edge_frac': res.get('edge_frac', 0.0),
            'd_eff_mean': res['d_eff_mean'], 'd_eff_std': res['d_eff_std'],
            'moran_I': res.get('I', float('nan')),
            'moran_z': res.get('z', float('nan')),
            'mean_pair_dist': mean_pair_distance(G),
        })

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="How many shortcuts can coherent geometry tolerate?")
    parser.add_argument('--nodes', type=int, nargs='+', default=[2000])
    parser.add_argument('--topology', type=str, default='grown')
    parser.add_argument('--cap', type=int, default=6,
                        help='Degree cap for the grown generator.')
    parser.add_argument('--counts', type=int, nargs='+',
                        default=[0, 1, 2, 5, 10, 20, 50, 100, 200, 400])
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0,
                        help='Base RNG seed (run i uses seed + i).')
    parser.add_argument('--permutations', type=int, default=199)
    parser.add_argument('--quick', action='store_true',
                        help='Tiny smoke-test configuration.')
    parser.add_argument('--out', type=str, default=None,
                        help='CSV output path (default results/shortcut_tolerance_<ts>.csv).')
    args = parser.parse_args()

    if args.quick:
        args.nodes, args.counts, args.seeds = [500], [0, 5, 20], 1
        args.permutations = 99

    all_rows: List[Dict] = []
    for n in args.nodes:
        for s in range(args.seeds):
            all_rows.extend(measure_tolerance(
                n, args.topology, args.cap, args.counts,
                seed=args.seed + s, n_perm=args.permutations))

    # Aggregate over seeds for the console table.
    print(f"\nShortcut tolerance -- topology={args.topology} cap={args.cap} "
          f"(fixed max_radius from k=0 base; {args.seeds} seed(s))")
    header = (f"{'N':>6s} {'k':>5s} {'load':>6s} {'def_frac':>8s} "
              f"{'edge_fr':>7s} {'d_eff':>11s} {'Moran I':>8s} {'z':>7s} "
              f"{'<pair d>':>8s}")
    print(header)
    print('-' * len(header))
    for n in args.nodes:
        for k in args.counts:
            sub = [r for r in all_rows if r['n_nodes'] == n and r['k'] == k]
            if not sub:
                continue
            df = np.mean([r['defined_frac'] for r in sub])
            ef = np.mean([r['edge_frac'] for r in sub])
            dm = np.nanmean([r['d_eff_mean'] for r in sub])
            ds = np.nanmean([r['d_eff_std'] for r in sub])
            mi = np.nanmean([r['moran_I'] for r in sub])
            mz = np.nanmean([r['moran_z'] for r in sub])
            pd_ = np.mean([r['mean_pair_dist'] for r in sub])
            load = np.mean([r['damage_load'] for r in sub])
            dstr = f"{dm:4.2f}+-{ds:4.2f}" if np.isfinite(dm) else "   --  "
            print(f"{n:6d} {k:5d} {load:6.2f} {df:8.3f} {ef:7.2f} "
                  f"{dstr:>11s} {mi:8.3f} {mz:7.1f} {pd_:8.2f}")

    # Damage-model fit: early-regime slope of (1 - defined_frac) vs load.
    early = [r for r in all_rows
             if 0 < r['damage_load'] <= 0.5 and r['k'] > 0]
    if early:
        x = np.array([r['damage_load'] for r in early])
        y = np.array([1.0 - r['defined_frac'] for r in early])
        c = float(x @ y / (x @ x))          # least squares through origin
        resid = float(np.sqrt(np.mean((y - c * x) ** 2)))
        print(f"\nDamage-model fit (load <= 0.5): 1 - defined_frac ~= c * load, "
              f"c = {c:.2f} (rms {resid:.3f}, {len(early)} points)")

    out = args.out or (Path('results') /
                       f"shortcut_tolerance_{args.topology}.csv")
    Path(out).parent.mkdir(exist_ok=True)
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Rows written to {out}")


if __name__ == '__main__':
    main()
