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
                    saturation_frac: float = 0.1,
                    min_radii: int = 6,
                    r2_threshold: float = 0.95) -> Tuple[float, float]:
    """
    Estimate local effective dimension from ball-growth data, with a
    finite-size correction and a gate for whether dimension is even
    well-defined.

    Returns (d_eff, r_squared):
      * ``d_eff`` is the corrected log-log slope where a polynomial
        ball-growth regime exists, otherwise ``nan`` -- meaning the region
        has no power-law regime and effective dimension is *undefined*
        there (e.g. small-world / expander structure).
      * ``r_squared`` is the fit quality (0.0 when undefined for lack of
        data).

    Method -- finite-size-corrected slope
    -------------------------------------
    Ball growth obeys ``|B(r)| = C * r^d * (1 + a/r + ...)``. A plain fit of
    ``log|B|`` vs ``log r`` underestimates ``d`` because the ``a/r`` term
    flattens the slope at the small radii we can actually reach. We instead
    regress

        log|B(r)| = d*log r + c + a*(1/r)

    so the ``1/r`` column absorbs the leading correction and the ``log r``
    coefficient is a (near-)unbiased estimate of ``d``. (On clean lattices
    this recovers d=2 and d=3 to within ~0.05 at radius 8, vs ~0.4 low for
    the plain fit.)

    Gating -- when is dimension defined?
    ------------------------------------
    Ball-growth dimension only means something when the ball grows
    polynomially across a usable range of scales. Small-world / expander
    graphs saturate in a few hops and have no such regime -- forcing a
    slope there yields a meaningless number. We require:
      1. at least ``min_radii`` unsaturated radii (genuine scale
         separation -- the decisive test; expander graphs fail this), and
      2. corrected-fit ``R^2 >= r2_threshold`` (the growth is actually a
         clean power law).
    If either fails, ``d_eff`` is ``nan``.
    """
    nan = float('nan')

    if not ball_counts:
        return nan, 0.0

    # Keep only radii where the ball is still a small *local* fraction of
    # the graph. Effective dimension is a local property; once the ball
    # covers a large share of the graph the slope reflects global finite
    # size, not local geometry. This low fraction is also what makes the
    # regime gate work: expander / small-world balls blow past it in ~3
    # hops (few radii -> undefined), while genuinely low-dimensional balls
    # stay under it across many radii.
    saturation_threshold = int(n_total * saturation_frac)
    trimmed = [(r, c) for r, c in ball_counts if c < saturation_threshold]

    # Gate 1: not enough scale separation to establish a power law.
    if len(trimmed) < min_radii:
        return nan, 0.0

    radii = np.array([r for r, _ in trimmed], dtype=np.float64)
    counts = np.maximum(
        np.array([c for _, c in trimmed], dtype=np.float64), 1.0
    )
    log_r = np.log(radii)
    log_c = np.log(counts)

    # Corrected fit: columns [log r, 1, 1/r]. min_radii >= 4 guarantees we
    # keep spare degrees of freedom, so the correction term can't overfit.
    design = np.column_stack([log_r, np.ones_like(log_r), 1.0 / radii])
    coeffs, _, _, _ = np.linalg.lstsq(design, log_c, rcond=None)
    d_eff = coeffs[0]

    predicted = design @ coeffs
    ss_res = np.sum((log_c - predicted) ** 2)
    ss_tot = np.sum((log_c - np.mean(log_c)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Gate 2: growth isn't a clean power law -> dimension undefined.
    if r_squared < r2_threshold:
        return nan, float(r_squared)

    return float(d_eff), float(r_squared)


# ======================================================================
# Auto-calibration of max_radius
# ======================================================================

def estimate_max_radius(G: nx.Graph, n_probes: int = 5) -> int:
    """
    Estimate a good max_radius by sampling node eccentricities.

    Uses R = max(3, min(10, estimated_diameter // 2)) to stay within
    the range where ball growth is informative without saturating. The
    upper cap gives geometric graphs enough radii for the corrected fit
    and the regime gate in local_dimension; small-world graphs are limited
    by their own tiny diameter well below it.
    """
    nodes = list(G.nodes())
    n = len(nodes)

    if n <= 10:
        return 3

    # Deterministic, evenly-spaced probes: the radius must not depend on
    # RNG state, or the same graph would be judged differently run to run.
    probe_idx = np.linspace(0, n - 1, min(n_probes, n)).astype(int)
    max_ecc = 0
    for i in probe_idx:
        lengths = nx.single_source_shortest_path_length(G, nodes[i])
        ecc = max(lengths.values()) if lengths else 0
        max_ecc = max(max_ecc, ecc)

    return max(3, min(10, max_ecc // 2))


def _estimate_max_radius_sparse(A, n_probes: int = 5) -> int:
    """Estimate max_radius from sparse adjacency matrix via iterative BFS."""
    n = A.shape[0]
    if n <= 10:
        return 3

    # Deterministic, evenly-spaced probes (see estimate_max_radius).
    probes = np.linspace(0, n - 1, min(n_probes, n)).astype(int)
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

    return max(3, min(10, max_ecc // 2))


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

    Nodes whose dimension is undefined (no power-law regime -> d_eff is
    nan; see local_dimension) are reported as a separate count and
    excluded from the d_eff statistics and histogram. The fraction of
    nodes with a *defined* dimension is itself a key signal: on the
    project's small-world initial graphs it should be ~0, and any rise
    over time is evidence that geometric structure is emerging.

    Args:
        dim_field: Output of dimension_field or fast_dimension_field.
        n_nodes:   Total node count in the graph.

    Returns dict with keys:
        d_eff_mean, d_eff_std, d_eff_median, d_eff_min, d_eff_max,
        r_squared_mean, n_sampled, n_defined, n_undefined, defined_frac,
        n_nodes, hist_bins, hist_counts,
        coherent_frac (fraction of defined nodes with R^2 > 0.9)
    """
    empty = {
        'd_eff_mean': 0.0, 'd_eff_std': 0.0, 'd_eff_median': 0.0,
        'd_eff_min': 0.0, 'd_eff_max': 0.0,
        'r_squared_mean': 0.0, 'n_sampled': 0,
        'n_defined': 0, 'n_undefined': 0, 'defined_frac': 0.0,
        'n_nodes': n_nodes,
        'hist_bins': ['d < 1.5', '1.5 <= d < 2.5',
                      '2.5 <= d < 3.5', 'd >= 3.5'],
        'hist_counts': [0, 0, 0, 0],
        'coherent_frac': 0.0,
    }
    if not dim_field:
        return empty

    d_all = np.array([v[0] for v in dim_field.values()], dtype=np.float64)
    r2_all = np.array([v[1] for v in dim_field.values()], dtype=np.float64)

    n_sampled = len(d_all)
    defined_mask = np.isfinite(d_all)
    n_defined = int(defined_mask.sum())
    n_undefined = n_sampled - n_defined

    if n_defined == 0:
        stats = dict(empty)
        stats.update(n_sampled=n_sampled, n_undefined=n_undefined)
        return stats

    d_effs = d_all[defined_mask]
    r_squareds = r2_all[defined_mask]

    # Histogram with fixed bins for interpretability (defined nodes only)
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
        'n_sampled': n_sampled,
        'n_defined': n_defined,
        'n_undefined': n_undefined,
        'defined_frac': float(n_defined / n_sampled),
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

    n_defined = stats.get('n_defined', stats['n_sampled'])
    n_undef = stats.get('n_undefined', 0)
    defined_pct = 100.0 * stats.get('defined_frac', 0.0)

    print(f"\n  Nodes sampled:     {stats['n_sampled']} / {stats['n_nodes']}")
    print(f"  Max radius used:   {max_radius}")
    print(f"  Dimension defined: {n_defined} ({defined_pct:.1f}%)  "
          f"-- undefined (no power-law regime): {n_undef}")

    if n_defined == 0:
        print("\n  No nodes have a well-defined effective dimension.")
        print("  Ball growth is non-polynomial everywhere sampled "
              "(small-world / expander structure).")
        return

    print(f"\n  Over the {n_defined} nodes with a defined dimension:")
    print(f"    Mean d_eff:      {stats['d_eff_mean']:.2f} +/- {stats['d_eff_std']:.2f}")
    print(f"    Median d_eff:    {stats['d_eff_median']:.2f}")
    print(f"    Range:           [{stats['d_eff_min']:.2f}, {stats['d_eff_max']:.2f}]")
    print(f"    Mean R-squared:  {stats['r_squared_mean']:.3f}")

    print(f"\n  Dimension distribution (defined nodes):")
    labels = stats['hist_bins']
    counts = stats['hist_counts']
    annotations = ['filamentary', '', '', 'high-dimensional hubs']
    for label, count, ann in zip(labels, counts, annotations):
        pct = 100.0 * count / n_defined if n_defined > 0 else 0.0
        line = f"    {label:20s} {count:5d} ({pct:5.1f}%)"
        if ann and count > 0:
            line += f"  -- {ann}"
        print(line)

    coherent_pct = stats['coherent_frac'] * 100
    print(f"\n  Coherence: {coherent_pct:.1f}% of defined nodes have R^2 > 0.9")


# ======================================================================
# Validation against known-dimension graphs
# ======================================================================
#
# The estimator has two jobs, and we validate both:
#
#  (A) On graphs with a genuine polynomial ball-growth regime, recover the
#      known effective dimension:
#        path / cycle ...... d = 1   (|B(r)| ~ 2r)
#        2D square lattice . d = 2   (|B(r)| ~ 2r^2)
#        3D cubic lattice .. d = 3   (|B(r)| ~ (4/3)r^3)
#      The 1/r-corrected fit recovers these to within ~0.05 even at small
#      radius, where a plain log-log fit reads ~0.4 low.
#
#  (B) On small-world / expander graphs (the project's default initial
#      topologies) there is NO polynomial regime -- the ball saturates in
#      a few hops -- so effective dimension is *undefined*. The estimator
#      must report nan there, not invent a number.
#
# Lattice centers are chosen deep in the interior so boundary truncation
# doesn't bias the fit, and the lattices are sized so the ball stays below
# the saturation threshold across the full radius range.

def _known_dimension_graphs(max_radius: int):
    """
    Build graphs with analytically known effective dimension.

    Returns list of (name, G, center_node, expected_dim). Graphs are sized
    so that, at `max_radius`, the center stays interior AND the ball stays
    below the 10%-of-graph saturation cut across the whole radius window
    (so the regime gate keeps all radii). Slow-growing 1D balls need the
    most headroom: |B(r)| = 2r+1 must stay under 0.1 n out to max_radius.
    """
    R = max_radius

    # 1D path/cycle: need 2R+1 < 0.1 n  ->  n > ~10(2R+1). Use generous n.
    len_1d = 40 * R + 1
    path = nx.path_graph(len_1d)
    cycle = nx.cycle_graph(len_1d)

    # 2D: need 2R^2 < 0.1 n  ->  side > ~sqrt(20) R ~ 4.5 R.
    side2 = 7 * R + 1
    grid2 = nx.grid_2d_graph(side2, side2)

    # 3D: balls ~ (4/3)R^3; side of ~3R keeps it well under 0.1 n and the
    # center > R from any face.
    side3 = 3 * R + 1
    grid3 = nx.grid_graph(dim=(side3, side3, side3))

    return [
        (f"path-1D ({len_1d} nodes)", path, len_1d // 2, 1.0),
        (f"cycle-1D ({len_1d} nodes)", cycle, 0, 1.0),
        (f"grid-2D ({side2}x{side2})", grid2, (side2 // 2, side2 // 2), 2.0),
        (f"grid-3D ({side3}^3)", grid3,
         (side3 // 2, side3 // 2, side3 // 2), 3.0),
    ]


def _d_eff_both_backends(G: nx.Graph, center, max_radius: int):
    """
    Compute d_eff for a single node via both the NetworkX and the sparse
    backends. Returns (d_nx, r2_nx, d_fast, r2_fast).
    """
    n = len(G)

    # NetworkX reference
    balls_nx = ball_sizes(G, center, max_radius)
    d_nx, r2_nx = local_dimension(balls_nx, n)

    # Sparse fast path -- resolve the center's matrix index
    node_list = list(G.nodes())
    center_idx = node_list.index(center)
    A = nx.to_scipy_sparse_array(G, weight=None, format='csr', dtype=np.float32)
    balls_fast = fast_ball_sizes(A, center_idx, max_radius)
    d_fast, r2_fast = local_dimension(balls_fast, n)

    return d_nx, r2_nx, d_fast, r2_fast


def _agree(a: float, b: float, atol: float = 1e-4) -> bool:
    """Backend agreement, treating nan==nan as agreement."""
    if not np.isfinite(a) and not np.isfinite(b):
        return True
    if np.isfinite(a) != np.isfinite(b):
        return False
    return abs(a - b) <= atol


def validate_estimator(max_radius: int = 12, tol: float = 0.2) -> bool:
    """
    Validate the dimension estimator. Returns True iff every check passes.

    Checks:
      1. Recovery   -- on 1D/2D/3D lattices, d_eff is within `tol` of the
                       true integer dimension.
      2. Ordering   -- d_eff(1D) < d_eff(2D) < d_eff(3D).
      3. Backends   -- NetworkX and sparse paths agree for the same node.
      4. Undefined  -- small-world / scale-free / random graphs have no
                       polynomial regime, so almost no node gets a defined
                       dimension (d_eff = nan).
    """
    np.random.seed(0)  # deterministic probe/sample choices

    print("=" * 64)
    print("DIMENSION ESTIMATOR VALIDATION")
    print("=" * 64)
    print(f"max_radius = {max_radius}, recovery tolerance = +/-{tol}")

    passed = 0
    total = 0

    # ----- (A) Recovery + backend agreement on known-dimension graphs -----
    print("\n(A) Recovery on graphs with a known dimension")
    print(f"\n{'graph':22s} {'expect':>6s} {'d_nx':>7s} {'d_fast':>7s} "
          f"{'R^2':>6s}  result")
    print("-" * 64)

    d_by_dim: Dict[float, float] = {}
    for name, G, center, expected in _known_dimension_graphs(max_radius):
        d_nx, r2_nx, d_fast, r2_fast = _d_eff_both_backends(
            G, center, max_radius
        )
        d_by_dim[expected] = d_nx

        total += 1  # recovery
        recovered = np.isfinite(d_nx) and abs(d_nx - expected) <= tol
        if recovered:
            passed += 1

        total += 1  # backend agreement
        backends_agree = _agree(d_nx, d_fast)
        if backends_agree:
            passed += 1

        tag = "PASS" if (recovered and backends_agree) else "FAIL"
        print(f"{name:22s} {expected:6.1f} {d_nx:7.3f} {d_fast:7.3f} "
              f"{r2_nx:6.3f}  {tag}")
        if not recovered:
            print(f"    -> recovery off: d_nx={d_nx:.3f}, "
                  f"expected {expected:.1f} +/-{tol}")
        if not backends_agree:
            print(f"    -> backend mismatch: nx={d_nx:.4f} fast={d_fast:.4f}")

    # ----- (2) Ordering across dimensions -----
    print("-" * 64)
    total += 1
    ordered = (np.isfinite(d_by_dim[1.0]) and np.isfinite(d_by_dim[2.0])
               and np.isfinite(d_by_dim[3.0])
               and d_by_dim[1.0] < d_by_dim[2.0] < d_by_dim[3.0])
    if ordered:
        passed += 1
        print(f"Ordering: {d_by_dim[1.0]:.3f} (1D) < {d_by_dim[2.0]:.3f} "
              f"(2D) < {d_by_dim[3.0]:.3f} (3D)  PASS")
    else:
        print(f"Ordering FAIL: 1D={d_by_dim[1.0]:.3f} "
              f"2D={d_by_dim[2.0]:.3f} 3D={d_by_dim[3.0]:.3f}")

    # ----- (4) Undefined on small-world / expander graphs -----
    print("\n(B) Dimension must be UNDEFINED on small-world / expander graphs")
    print(f"\n{'graph':22s} {'R':>3s} {'defined %':>10s}  result")
    print("-" * 64)

    undefined_cases = [
        ("small_world (2000)", nx.watts_strogatz_graph(2000, 6, 0.1, seed=1)),
        ("scale_free (2000)", nx.barabasi_albert_graph(2000, 3, seed=1)),
        ("random (2000)", nx.erdos_renyi_graph(2000, 6 / 2000, seed=1)),
    ]
    for name, G in undefined_cases:
        R = estimate_max_radius(G)
        dim = dimension_field(G, max_radius=R, n_samples=100)
        stats = dimension_stats(dim, len(G))
        defined_frac = stats['defined_frac']

        total += 1
        # Expander graphs should yield essentially no defined dimensions.
        is_undefined = defined_frac <= 0.05
        if is_undefined:
            passed += 1
        tag = "PASS" if is_undefined else "FAIL"
        print(f"{name:22s} {R:3d} {defined_frac * 100:9.1f}%  {tag}")
        if not is_undefined:
            print(f"    -> expected ~0% defined, got "
                  f"{defined_frac * 100:.1f}% (regime gate too weak?)")

    # ----- Diagnostic: corrected fit is now radius-stable near the truth -----
    print("\nRadius stability (2D lattice, corrected fit, expect ~2.0):")
    big_side = 2 * 30 + 1
    big_grid = nx.grid_2d_graph(big_side, big_side)
    big_center = (big_side // 2, big_side // 2)
    n_big = len(big_grid)
    for r in (6, 8, 10, 12):
        balls = ball_sizes(big_grid, big_center, r)
        d_r, r2_r = local_dimension(balls, n_big)
        print(f"    max_radius={r:2d}:  d_eff = {d_r:.3f}  (R^2={r2_r:.3f})")

    print("\n" + "=" * 64)
    print(f"VALIDATION: {passed}/{total} checks passed")
    print("=" * 64)
    return passed == total


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute local effective dimension for graph-graph results'
    )
    parser.add_argument('input', type=str, nargs='?', default=None,
                        help='Results pickle file')
    parser.add_argument('--max-radius', type=int, default=None,
                        help='Max BFS radius (auto-calibrated if omitted)')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of nodes to sample (default: min(N, 500))')
    parser.add_argument('--fast', action='store_true',
                        help='Use sparse matrix backend (faster for large graphs)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save analysis dict to pickle file')
    parser.add_argument('--validate', action='store_true',
                        help='Validate the estimator against known-dimension '
                             'graphs (no results file needed)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible node sampling')

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.validate:
        ok = validate_estimator(
            max_radius=args.max_radius if args.max_radius is not None else 12
        )
        print("\nAll checks passed." if ok else "\nSome checks FAILED.")
        return

    if args.input is None:
        parser.print_help()
        print("\nExamples:")
        print("  python dimension.py --validate")
        print("  python dimension.py results/run_TIMESTAMP.pkl")
        print("  python dimension.py results/run_TIMESTAMP.pkl --fast "
              "--max-radius 6 --samples 300")
        return

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
