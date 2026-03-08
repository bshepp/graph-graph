"""
Local effective dimension estimation for graph-graph.

Measures how "d-dimensional" the local neighborhood of each node is
by fitting geodesic ball growth |B(v,r)| ~ r^d.

Implements the first testable prediction from DIMENSIONAL_COHERENCE.md:
do simple local rules produce regions of coherent effective dimensionality?

Usage:
    python dimension.py results/run_TIMESTAMP.pkl
    python dimension.py results/run_TIMESTAMP.pkl --max-radius 6 --samples 300 --fast
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import networkx as nx


# ======================================================================
# Core: ball sizes and dimension fitting
# ======================================================================

def ball_sizes(G: nx.Graph, node, max_radius: int) -> List[Tuple[int, int]]:
    """
    Compute cumulative ball sizes |B(v, r)| for r = 1..max_radius.

    B(v, r) is the set of nodes reachable from v in <= r hops (including v).
    Returns list of (radius, cumulative_count) pairs.
    """
    distances = nx.single_source_shortest_path_length(G, node, cutoff=max_radius)
    counts_at_distance = {}
    for _target, dist in distances.items():
        counts_at_distance[dist] = counts_at_distance.get(dist, 0) + 1

    # Cumulative: |B(v, r)| = sum of nodes at distance <= r
    cumulative = 0
    result = []
    for r in range(0, max_radius + 1):
        cumulative += counts_at_distance.get(r, 0)
        if r >= 1:
            result.append((r, cumulative))

    return result


def local_dimension(ball_counts: List[Tuple[int, int]], n_total: int,
                    saturation_frac: float = 0.5) -> Tuple[float, float]:
    """
    Estimate effective dimension from ball growth data.

    Fits log|B(v,r)| vs log(r) and returns (d_eff, r_squared).
    Trims radii where the ball has reached saturation_frac of the graph
    to prevent finite-size effects from flattening the slope.

    Returns (0.0, 0.0) for isolated nodes.
    """
    if not ball_counts:
        return 0.0, 0.0

    # Trim saturated radii
    saturation_threshold = int(n_total * saturation_frac)
    trimmed = [(r, c) for r, c in ball_counts if c < saturation_threshold]

    # Need at least 2 points for a linear fit
    if len(trimmed) < 2:
        # If even the first radius saturates, use untrimmed but mark poor fit
        if len(ball_counts) >= 2:
            trimmed = ball_counts[:2]
        else:
            return 0.0, 0.0

    radii = np.array([r for r, _ in trimmed], dtype=np.float64)
    counts = np.array([c for _, c in trimmed], dtype=np.float64)

    # Guard against log(0)
    counts = np.maximum(counts, 1.0)

    log_r = np.log(radii)
    log_c = np.log(counts)

    # Linear regression: log|B| = d_eff * log(r) + intercept
    coeffs = np.polyfit(log_r, log_c, 1)
    d_eff = coeffs[0]

    # R-squared
    predicted = np.polyval(coeffs, log_r)
    ss_res = np.sum((log_c - predicted) ** 2)
    ss_tot = np.sum((log_c - np.mean(log_c)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(d_eff), float(r_squared)


# ======================================================================
# Auto-calibration of max_radius
# ======================================================================

def estimate_max_radius(G: nx.Graph, n_probes: int = 5) -> int:
    """
    Estimate a good max_radius by sampling node eccentricities.

    Uses R = max(3, min(8, estimated_diameter // 2)) to stay within
    the range where ball growth is informative without saturating.
    """
    nodes = list(G.nodes())
    n = len(nodes)

    if n <= 10:
        return 3

    probes = np.random.choice(nodes, min(n_probes, n), replace=False)
    max_ecc = 0
    for node in probes:
        lengths = nx.single_source_shortest_path_length(G, node)
        ecc = max(lengths.values()) if lengths else 0
        max_ecc = max(max_ecc, ecc)

    return max(3, min(8, max_ecc // 2))


def _estimate_max_radius_sparse(A, n_probes: int = 5) -> int:
    """Estimate max_radius from sparse adjacency matrix via iterative BFS."""
    n = A.shape[0]
    if n <= 10:
        return 3

    probes = np.random.choice(n, min(n_probes, n), replace=False)
    max_ecc = 0

    for idx in probes:
        reached = np.zeros(n, dtype=np.float32)
        reached[idx] = 1.0
        prev_count = 1
        for r in range(1, n):
            new_reached = A @ reached + reached
            reached = (new_reached > 0).astype(np.float32)
            count = int(reached.sum())
            if count == prev_count or count == n:
                max_ecc = max(max_ecc, r)
                break
            prev_count = count

    return max(3, min(8, max_ecc // 2))


# ======================================================================
# Dimension field computation -- NetworkX backend
# ======================================================================

def dimension_field(G: nx.Graph, max_radius: int | None = None,
                    n_samples: int | None = None
                    ) -> Dict[Any, Tuple[float, float, List[Tuple[int, int]]]]:
    """
    Compute local effective dimension for nodes in the graph.

    Args:
        max_radius: Maximum BFS radius. Auto-calibrated if None.
        n_samples:  Number of nodes to sample. None = min(len(G), 500).

    Returns:
        {node: (d_eff, r_squared, ball_sizes_list)}
    """
    n = len(G)
    nodes = list(G.nodes())

    if max_radius is None:
        max_radius = estimate_max_radius(G)

    if n_samples is None:
        n_samples = min(n, 500)

    if n_samples >= n:
        sample = nodes
    else:
        sample = list(np.random.choice(nodes, n_samples, replace=False))

    result = {}
    for node in sample:
        balls = ball_sizes(G, node, max_radius)
        d_eff, r_sq = local_dimension(balls, n)
        result[node] = (d_eff, r_sq, balls)

    return result


# ======================================================================
# Dimension field computation -- sparse fast path
# ======================================================================

def fast_ball_sizes(A, node_idx: int,
                    max_radius: int) -> List[Tuple[int, int]]:
    """
    Compute ball sizes using iterative sparse mat-vec with binarization.

    Uses A @ v + v (adjacency times reachability + identity) at each step,
    then binarizes. Avoids materializing (A+I)^r which fills memory.
    """
    n = A.shape[0]
    reached = np.zeros(n, dtype=np.float32)
    reached[node_idx] = 1.0

    result = []
    for r in range(1, max_radius + 1):
        new_reached = A @ reached + reached
        reached = (new_reached > 0).astype(np.float32)
        result.append((r, int(reached.sum())))

    return result


def fast_dimension_field(A, max_radius: int | None = None,
                         sample_indices: np.ndarray | None = None,
                         n_samples: int | None = None
                         ) -> Dict[int, Tuple[float, float, List[Tuple[int, int]]]]:
    """
    Compute dimension field using sparse adjacency matrix.

    Args:
        A:              Sparse CSR adjacency matrix (n x n).
        max_radius:     Max BFS radius. Auto-calibrated if None.
        sample_indices: Specific node indices to sample. Overrides n_samples.
        n_samples:      Number of nodes to sample. None = min(n, 500).

    Returns:
        {node_index: (d_eff, r_squared, ball_sizes_list)}
    """
    n = A.shape[0]

    if max_radius is None:
        max_radius = _estimate_max_radius_sparse(A)

    if sample_indices is None:
        if n_samples is None:
            n_samples = min(n, 500)
        sample_indices = np.random.choice(n, min(n_samples, n), replace=False)

    result = {}
    for idx in sample_indices:
        balls = fast_ball_sizes(A, int(idx), max_radius)
        d_eff, r_sq = local_dimension(balls, n)
        result[int(idx)] = (d_eff, r_sq, balls)

    return result


# ======================================================================
# Statistics
# ======================================================================

def dimension_stats(dim_field: dict, n_nodes: int) -> Dict[str, Any]:
    """
    Compute summary statistics over a dimension field.

    Args:
        dim_field: Output of dimension_field or fast_dimension_field.
        n_nodes:   Total node count in the graph.

    Returns dict with keys:
        d_eff_mean, d_eff_std, d_eff_median, d_eff_min, d_eff_max,
        r_squared_mean, n_sampled, n_nodes,
        hist_bins (bin edges), hist_counts,
        coherent_frac (fraction with R^2 > 0.9)
    """
    if not dim_field:
        return {
            'd_eff_mean': 0.0, 'd_eff_std': 0.0, 'd_eff_median': 0.0,
            'd_eff_min': 0.0, 'd_eff_max': 0.0,
            'r_squared_mean': 0.0, 'n_sampled': 0, 'n_nodes': n_nodes,
            'hist_bins': [], 'hist_counts': [],
            'coherent_frac': 0.0,
        }

    d_effs = np.array([v[0] for v in dim_field.values()])
    r_squareds = np.array([v[1] for v in dim_field.values()])

    # Histogram with fixed bins for interpretability
    bin_edges = [0.0, 1.5, 2.5, 3.5, float('inf')]
    bin_labels = ['d < 1.5', '1.5 <= d < 2.5', '2.5 <= d < 3.5', 'd >= 3.5']
    hist_counts = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        count = int(np.sum((d_effs >= lo) & (d_effs < hi)))
        hist_counts.append(count)

    return {
        'd_eff_mean': float(np.mean(d_effs)),
        'd_eff_std': float(np.std(d_effs)),
        'd_eff_median': float(np.median(d_effs)),
        'd_eff_min': float(np.min(d_effs)),
        'd_eff_max': float(np.max(d_effs)),
        'r_squared_mean': float(np.mean(r_squareds)),
        'n_sampled': len(dim_field),
        'n_nodes': n_nodes,
        'hist_bins': bin_labels,
        'hist_counts': hist_counts,
        'coherent_frac': float(np.mean(r_squareds > 0.9)),
    }


# ======================================================================
# Pretty printing
# ======================================================================

def print_dimension_analysis(stats: Dict[str, Any], max_radius: int):
    """Print formatted dimension analysis summary."""
    print("\n" + "=" * 60)
    print("DIMENSION ANALYSIS")
    print("=" * 60)

    print(f"\n  Nodes sampled:     {stats['n_sampled']} / {stats['n_nodes']}")
    print(f"  Max radius used:   {max_radius}")
    print(f"  Mean d_eff:        {stats['d_eff_mean']:.2f} +/- {stats['d_eff_std']:.2f}")
    print(f"  Median d_eff:      {stats['d_eff_median']:.2f}")
    print(f"  Range:             [{stats['d_eff_min']:.2f}, {stats['d_eff_max']:.2f}]")
    print(f"  Mean R-squared:    {stats['r_squared_mean']:.3f}")

    print(f"\n  Dimension distribution:")
    labels = stats['hist_bins']
    counts = stats['hist_counts']
    n = stats['n_sampled']
    annotations = ['filamentary', '', '', 'high-dimensional hubs']
    for label, count, ann in zip(labels, counts, annotations):
        pct = 100.0 * count / n if n > 0 else 0.0
        line = f"    {label:20s} {count:5d} ({pct:5.1f}%)"
        if ann and count > 0:
            line += f"  -- {ann}"
        print(line)

    coherent_pct = stats['coherent_frac'] * 100
    print(f"\n  Coherence: {coherent_pct:.1f}% of sampled nodes have R^2 > 0.9")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute local effective dimension for graph-graph results'
    )
    parser.add_argument('input', type=str, help='Results pickle file')
    parser.add_argument('--max-radius', type=int, default=None,
                        help='Max BFS radius (auto-calibrated if omitted)')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of nodes to sample (default: min(N, 500))')
    parser.add_argument('--fast', action='store_true',
                        help='Use sparse matrix backend (faster for large graphs)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save analysis dict to pickle file')

    args = parser.parse_args()

    print(f"Loading {args.input}...")
    with open(args.input, 'rb') as f:
        results = pickle.load(f)

    print(f"Loaded: {results['params']}")
    G = results['final_graph']
    n_nodes = len(G)

    # Auto-calibrate radius
    if args.max_radius is None:
        max_radius = estimate_max_radius(G)
        print(f"Auto-calibrated max_radius = {max_radius}")
    else:
        max_radius = args.max_radius

    # Compute dimension field
    if args.fast:
        import scipy.sparse as sp
        print(f"Computing dimension field (sparse backend, "
              f"{args.samples or min(n_nodes, 500)} samples)...")
        A = nx.to_scipy_sparse_array(G, weight=None, format='csr',
                                     dtype=np.float32)
        dim = fast_dimension_field(A, max_radius=max_radius,
                                   n_samples=args.samples)
    else:
        print(f"Computing dimension field (NetworkX backend, "
              f"{args.samples or min(n_nodes, 500)} samples)...")
        dim = dimension_field(G, max_radius=max_radius,
                              n_samples=args.samples)

    stats = dimension_stats(dim, n_nodes)
    stats['max_radius'] = max_radius
    print_dimension_analysis(stats, max_radius)

    if args.save:
        analysis = {
            'dimension_field': dim,
            'stats': stats,
            'params': results['params'],
        }
        with open(args.save, 'wb') as f:
            pickle.dump(analysis, f)
        print(f"\nAnalysis saved to {args.save}")


if __name__ == '__main__':
    main()
