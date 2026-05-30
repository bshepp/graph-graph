"""
Temporal dimension tracking for graph-graph.

Runs a simulation while measuring the local effective dimension field at
intervals, producing a *trajectory* of how dimensional structure evolves:

  - defined_frac : fraction of nodes with a well-defined effective dimension
                   (the preservation / emergence signal -- see
                   DIMENSIONAL_COHERENCE.md and dimension.py)
  - d_eff distribution over the nodes that do have a defined dimension

This is the experiment the dimension estimator was built for. It answers a
sharp, falsifiable question:

    Do local rules PRESERVE, DESTROY, or CREATE geometric (low-dimensional)
    structure over time?

The pristine initial graph is measured as the t=0 baseline, so the
trajectory is self-controlled: every later point is compared against the
graph before any rule ran.

Usage:
    # Preservation: does a 2D lattice keep its dimension under the rules?
    python track_dimension.py --topology lattice --rules majority --steps 200
    python track_dimension.py --topology lattice --rules rewire   --steps 200

    # Emergence: can structure arise from a dimensionless small-world start?
    python track_dimension.py --topology small_world --rules rewire --steps 500

Notes:
  * max_radius is calibrated ONCE on the t=0 graph and reused for every
    measurement, so points are directly comparable even as the topology
    changes.
  * Dimension sampling uses a saved/restored RNG state, so measuring the
    field does not perturb the simulation's random stream -- the chunked
    run is identical to a single continuous run with the same seed.
"""

import argparse
import random
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import networkx as nx

from simulation import create_initial_graph, run_simulation
from dimension import (
    dimension_field,
    fast_dimension_field,
    dimension_stats,
    estimate_max_radius,
)

try:
    from simulation_fast import run_fast_simulation
    _HAS_FAST = True
except ImportError:  # pragma: no cover - scipy missing
    _HAS_FAST = False


# ----------------------------------------------------------------------
# Measurement (RNG-isolated from the simulation stream)
# ----------------------------------------------------------------------

def _measure_dimension(G: nx.Graph, max_radius: int, n_samples: int | None,
                       use_fast: bool) -> Dict[str, Any]:
    """
    Compute dimension stats for G at a fixed max_radius, without disturbing
    the global RNG (so the simulation downstream is unaffected).
    """
    rng_state = np.random.get_state()
    try:
        if use_fast:
            A = nx.to_scipy_sparse_array(
                G, weight=None, format='csr', dtype=np.float32
            )
            dim = fast_dimension_field(A, max_radius=max_radius,
                                       n_samples=n_samples)
        else:
            dim = dimension_field(G, max_radius=max_radius,
                                  n_samples=n_samples)
        stats = dimension_stats(dim, len(G))
    finally:
        np.random.set_state(rng_state)
    return stats


# ----------------------------------------------------------------------
# Trajectory driver
# ----------------------------------------------------------------------

def run_dimension_trajectory(
    nodes: int,
    topology: str,
    rules: List[str],
    total_steps: int,
    track_interval: int,
    seed: int | None = None,
    use_fast: bool = True,
    max_radius: int | None = None,
    n_samples: int | None = None,
) -> Dict[str, Any]:
    """
    Run a simulation in chunks, measuring the dimension field at t=0 and
    after every `track_interval` steps. Returns a trajectory dict.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    use_fast = use_fast and _HAS_FAST

    # Pristine t=0 graph and the fixed measurement radius.
    G = create_initial_graph(nodes, topology, seed=seed)
    if max_radius is None:
        max_radius = estimate_max_radius(G)

    traj: Dict[str, Any] = {
        'steps': [],
        'defined_frac': [],
        'd_eff_mean': [],
        'd_eff_median': [],
        'd_eff_std': [],
        'coherent_frac': [],
        'hist_bins': None,
        'hist_counts': [],
        'params': {
            'nodes': nodes,
            'topology': topology,
            'rules': rules,
            'total_steps': total_steps,
            'track_interval': track_interval,
            'seed': seed,
            'max_radius': max_radius,
            'n_samples': n_samples,
            'backend': 'fast' if use_fast else 'networkx',
        },
    }

    def record(step: int, G_now: nx.Graph):
        stats = _measure_dimension(G_now, max_radius, n_samples, use_fast)
        traj['steps'].append(step)
        traj['defined_frac'].append(stats['defined_frac'])
        traj['d_eff_mean'].append(stats['d_eff_mean'])
        traj['d_eff_median'].append(stats['d_eff_median'])
        traj['d_eff_std'].append(stats['d_eff_std'])
        traj['coherent_frac'].append(stats['coherent_frac'])
        traj['hist_counts'].append(stats['hist_counts'])
        if traj['hist_bins'] is None:
            traj['hist_bins'] = stats['hist_bins']
        print(f"  step {step:5d}:  defined={stats['defined_frac']*100:5.1f}%"
              f"   median d_eff={stats['d_eff_median']:.2f}"
              f"   (n_defined={stats['n_defined']})")

    print(f"Tracking dimension: {topology}, rules={rules}, "
          f"{total_steps} steps, interval {track_interval}, "
          f"max_radius={max_radius}, backend="
          f"{'fast' if use_fast else 'networkx'}")
    record(0, G)

    steps_done = 0
    while steps_done < total_steps:
        chunk = min(track_interval, total_steps - steps_done)
        if use_fast:
            res = run_fast_simulation(G, rules, chunk)
        else:
            res = run_simulation(G, rules, chunk)
        G = res['final_graph']
        steps_done += chunk
        record(steps_done, G)

    return traj


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Track local effective dimension over a simulation run',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n'
               '  python track_dimension.py --topology lattice --rules rewire '
               '--steps 200\n'
               '  python track_dimension.py --topology small_world '
               '--rules rewire --steps 500',
    )
    parser.add_argument('--nodes', type=int, default=2500)
    parser.add_argument('--steps', type=int, default=200,
                        help='Total simulation steps')
    parser.add_argument('--track-interval', type=int, default=20,
                        help='Measure dimension every N steps')
    parser.add_argument('--topology', type=str, default='lattice',
                        choices=['small_world', 'scale_free', 'lattice',
                                 'random', 'grown'])
    parser.add_argument('--rules', type=str, nargs='+', default=['rewire'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-radius', type=int, default=None,
                        help='Fixed measurement radius (auto-calibrated on '
                             'the t=0 graph if omitted)')
    parser.add_argument('--samples', type=int, default=None,
                        help='Nodes to sample per measurement '
                             '(default: min(N, 500))')
    parser.add_argument('--no-fast', action='store_true',
                        help='Force the NetworkX backend')
    parser.add_argument('--output', type=str, default=None,
                        help='Trajectory pickle path '
                             '(default: results/traj_TIMESTAMP.pkl)')
    parser.add_argument('--show', action='store_true',
                        help='Show the plot instead of saving it')

    args = parser.parse_args()

    use_fast = _HAS_FAST and not args.no_fast
    if not _HAS_FAST and not args.no_fast:
        print("Note: fast path unavailable (scipy missing?), using NetworkX.")

    traj = run_dimension_trajectory(
        nodes=args.nodes,
        topology=args.topology,
        rules=args.rules,
        total_steps=args.steps,
        track_interval=args.track_interval,
        seed=args.seed,
        use_fast=use_fast,
        max_radius=args.max_radius,
        n_samples=args.samples,
    )

    # Save trajectory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path('results')
        out_dir.mkdir(exist_ok=True)
        args.output = str(out_dir / f"traj_{timestamp}.pkl")
    with open(args.output, 'wb') as f:
        pickle.dump(traj, f)
    print(f"\nTrajectory saved to {args.output}")

    # Verdict
    d0 = traj['defined_frac'][0]
    d1 = traj['defined_frac'][-1]
    print(f"\nDefined fraction: {d0*100:.1f}% (t=0) -> {d1*100:.1f}% (end)")
    delta = d1 - d0
    if delta < -0.1:
        print("  -> dimensional structure ERODED by the rules")
    elif delta > 0.1:
        print("  -> dimensional structure EMERGED under the rules")
    else:
        print("  -> dimensional structure roughly PRESERVED")

    # Plot
    from visualize import plot_dimension_trajectory
    rules_str = '+'.join(args.rules)
    title = f"{args.topology} | rules: {rules_str}"
    if args.show:
        plot_dimension_trajectory(traj, title=title)
    else:
        stem = Path(args.output).stem
        save = str(Path('plots') / f"{stem}.png")
        Path('plots').mkdir(exist_ok=True)
        plot_dimension_trajectory(traj, save_path=save, title=title)


if __name__ == '__main__':
    main()
