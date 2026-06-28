"""
Spatial coherence of the effective-dimension field via Moran's I.

Open thread from FINDINGS.md: a per-node ``d_eff`` field could be *spatially
coherent* (graph-neighbouring nodes share a dimension -> the field forms
contiguous geometric phases) or it could be *scattered estimation noise* (a
node's d_eff is unrelated to its neighbours'). Moran's I is the standard test:

    I = (n / S0) * (x'^T W x') / (x'^T x'),   x' = x - mean(x),  S0 = sum(W)

with W the graph adjacency (w_ij = 1 iff i~j). I > 0 means neighbouring nodes
have *similar* d_eff (coherent); I ~ 0 means no spatial structure (noise);
I < 0 means anti-correlation (checkerboard). Significance is established by a
permutation null -- shuffle the field over the nodes and recompute I many times
-- which makes no distributional assumption. The analytic null expectation is
E[I] = -1/(n-1).

This is coordinate-free: "spatial" means *graph-adjacent*, never node positions
(there are none). It reuses the validated dimension estimator in dimension.py.

Usage:
    python coherence.py --validate                  # self-check on known fields
    python coherence.py                             # grown / lattice / small_world
    python coherence.py --nodes 5000 --cap 7 --seeds 3 --permutations 999
"""

import argparse
import random
from typing import Any, Dict

import numpy as np
import networkx as nx
import scipy.sparse as sp

from simulation import create_initial_graph
from dimension import fast_dimension_field, _estimate_max_radius_sparse


# ----------------------------------------------------------------------
# Moran's I
# ----------------------------------------------------------------------

def morans_i(W: sp.spmatrix, x: np.ndarray) -> float:
    """Moran's I of field ``x`` under symmetric weight matrix ``W``.

    W is binary graph adjacency (n x n); x is length n. Returns nan if the
    field has no variance (Moran's I is undefined for a constant field).
    """
    x_dev = x - x.mean()
    denom = float(x_dev @ x_dev)
    if denom == 0.0:
        return float('nan')
    s0 = float(W.sum())
    if s0 == 0.0:
        return float('nan')
    num = float(x_dev @ (W @ x_dev))
    return (len(x) / s0) * (num / denom)


def morans_i_test(W: sp.spmatrix, x: np.ndarray,
                  n_perm: int = 999) -> Dict[str, float]:
    """Moran's I plus a permutation-null significance test.

    Returns observed I, the analytic null E[I] = -1/(n-1), the permutation
    null mean/std, a z-score, and a one-sided pseudo-p for positive spatial
    autocorrelation ((#perm >= obs) + 1) / (n_perm + 1).
    """
    n = len(x)
    i_obs = morans_i(W, x)
    if not np.isfinite(i_obs):
        return {'I': float('nan'), 'E_I': -1.0 / (n - 1), 'z': float('nan'),
                'p': float('nan'), 'null_mean': float('nan'),
                'null_std': float('nan'), 'n': n}

    # The denominator and S0 are permutation-invariant; only the quadratic
    # form x' W x' changes, so recompute just that.
    x_dev = x - x.mean()
    denom = float(x_dev @ x_dev)
    s0 = float(W.sum())
    scale = n / s0

    null = np.empty(n_perm)
    for k in range(n_perm):
        xp = np.random.permutation(x_dev)
        null[k] = scale * float(xp @ (W @ xp)) / denom

    null_mean = float(null.mean())
    null_std = float(null.std())
    z = (i_obs - null_mean) / null_std if null_std > 0 else float('nan')
    p = (np.sum(null >= i_obs) + 1) / (n_perm + 1)
    return {'I': i_obs, 'E_I': -1.0 / (n - 1), 'z': z, 'p': float(p),
            'null_mean': null_mean, 'null_std': null_std, 'n': n}


# ----------------------------------------------------------------------
# Coherence of a graph's dimension field
# ----------------------------------------------------------------------

def field_coherence(G: nx.Graph, max_radius: int | None = None,
                    n_perm: int = 999,
                    min_defined: int = 30) -> Dict[str, Any]:
    """Compute the d_eff field over *all* nodes, then Moran's I over the
    subgraph of nodes with a defined dimension."""
    A = nx.to_scipy_sparse_array(G, weight=None, format='csr', dtype=np.float32)
    n = A.shape[0]
    if max_radius is None:
        max_radius = _estimate_max_radius_sparse(A)

    field = fast_dimension_field(A, max_radius=max_radius,
                                 sample_indices=np.arange(n))
    d_eff = np.array([field[i][0] for i in range(n)], dtype=float)
    defined = np.where(np.isfinite(d_eff))[0]

    out: Dict[str, Any] = {
        'n_nodes': n,
        'n_defined': int(defined.size),
        'defined_frac': float(defined.size / n),
        'max_radius': max_radius,
        'd_eff_mean': float(np.nanmean(d_eff)) if defined.size else float('nan'),
        'd_eff_std': float(np.nanstd(d_eff)) if defined.size else float('nan'),
    }
    if defined.size < min_defined:
        out.update({'I': float('nan'), 'E_I': float('nan'), 'z': float('nan'),
                    'p': float('nan'), 'edge_frac': 0.0,
                    'note': 'too few defined nodes'})
        return out

    W = A[np.ix_(defined, defined)]          # adjacency among defined nodes
    x = d_eff[defined]
    out.update(morans_i_test(W, x, n_perm=n_perm))
    # Fraction of the graph's edges that survive among defined nodes. When this
    # is low the defined set is a scattered, fragmented subsample, so Moran's I
    # measures coherence within tiny patches, NOT whole-field coherence -- the
    # claim is only meaningful when the field is defined almost everywhere.
    out['edge_frac'] = float(W.nnz / A.nnz) if A.nnz else 0.0
    return out


# ----------------------------------------------------------------------
# Validation on fields with known spatial structure
# ----------------------------------------------------------------------

def _validate() -> bool:
    """Moran's I must read +1 on a smooth gradient, ~0 on a random field,
    and -1 on a checkerboard -- on a lattice with known structure."""
    side = 40
    G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
    A = nx.to_scipy_sparse_array(G, weight=None, format='csr', dtype=np.float32)
    rows = np.array([i // side for i in range(side * side)], dtype=float)
    cols = np.array([i % side for i in range(side * side)], dtype=float)

    gradient = rows                               # smooth linear ramp
    checker = (rows + cols) % 2                    # maximal anti-correlation
    np.random.seed(0)
    noise = np.random.rand(side * side)            # no spatial structure

    i_grad = morans_i(A, gradient)
    i_check = morans_i(A, checker)
    res_noise = morans_i_test(A, noise, n_perm=499)

    print("Moran's I validation (40x40 lattice, binary adjacency):")
    print(f"  smooth gradient : I = {i_grad:+.3f}   (expect -> +1)")
    print(f"  checkerboard    : I = {i_check:+.3f}   (expect -> -1)")
    print(f"  random field    : I = {res_noise['I']:+.3f}, "
          f"E[I] = {res_noise['E_I']:+.3f}, z = {res_noise['z']:+.2f}, "
          f"p = {res_noise['p']:.3f}   (expect z~0, p not significant)")

    ok = (i_grad > 0.85 and i_check < -0.85 and abs(res_noise['z']) < 3.0)
    print("  -> PASS" if ok else "  -> FAIL")
    return ok


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Spatial coherence of the d_eff field (Moran's I).")
    parser.add_argument('--validate', action='store_true',
                        help='Self-check Moran\'s I on known fields and exit.')
    parser.add_argument('--nodes', type=int, default=5000)
    parser.add_argument('--cap', type=int, default=6,
                        help='Degree cap for the grown generator.')
    parser.add_argument('--topologies', type=str, nargs='+',
                        default=['grown', 'lattice', 'small_world'])
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--permutations', type=int, default=999)
    parser.add_argument('--max-radius', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0,
                        help='Base RNG seed (seed i = base + i).')
    args = parser.parse_args()

    if args.validate:
        ok = _validate()
        raise SystemExit(0 if ok else 1)

    print(f"Moran's I coherence of the d_eff field  "
          f"(N={args.nodes}, cap={args.cap}, "
          f"{args.seeds} seeds, {args.permutations} perms)\n")
    header = (f"{'topology':12s} {'def_frac':>8s} {'edge_fr':>8s} {'d_eff':>12s} "
              f"{'Moran I':>9s} {'z':>8s} {'p':>7s}  verdict")
    print(header)
    print('-' * len(header))

    for topo in args.topologies:
        Is, zs, dfs, efs, dmeans, dstds, ps, notes = [], [], [], [], [], [], [], []
        for s in range(args.seeds):
            seed = args.seed + s
            random.seed(seed)
            np.random.seed(seed)
            k = args.cap if topo == 'grown' else 6
            G = create_initial_graph(args.nodes, topology=topo, k=k, seed=seed)
            res = field_coherence(G, max_radius=args.max_radius,
                                  n_perm=args.permutations)
            dfs.append(res['defined_frac'])
            efs.append(res.get('edge_frac', 0.0))
            dmeans.append(res['d_eff_mean'])
            dstds.append(res['d_eff_std'])
            Is.append(res['I'])
            zs.append(res.get('z', float('nan')))
            ps.append(res.get('p', float('nan')))
            notes.append(res.get('note', ''))

        df = float(np.mean(dfs))
        ef = float(np.mean(efs))
        dmean = float(np.nanmean(dmeans))
        dstd = float(np.nanmean(dstds))
        I = float(np.nanmean(Is))
        z = float(np.nanmean(zs))
        p = float(np.nanmean(ps))

        # Coherence is a whole-field claim: it only holds when d_eff is defined
        # almost everywhere AND the defined nodes stay one connected fabric.
        # A high I over a sparse, fragmented defined subset (low edge_frac) is
        # not whole-field coherence, so refuse to call it COHERENT.
        whole_field = df > 0.8 and ef > 0.7
        if not np.isfinite(I):
            verdict = notes[0] or 'no defined field'
        elif not whole_field:
            verdict = 'field fragmented (N/A)'
        elif z > 3 and I > 0.1:
            verdict = 'COHERENT'
        elif abs(z) <= 3:
            verdict = 'noise (no structure)'
        else:
            verdict = 'anti-correlated'

        dstr = f"{dmean:5.2f}±{dstd:4.2f}" if np.isfinite(dmean) else "  --  "
        print(f"{topo:12s} {df:8.2f} {ef:8.2f} {dstr:>12s} "
              f"{I:9.3f} {z:8.1f} {p:7.3f}  {verdict}")


if __name__ == '__main__':
    main()
