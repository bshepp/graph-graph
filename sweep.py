"""
Parameter sweep for graph-graph.

Systematically explores combinations of topology, rules, and scale,
running simulations in parallel and aggregating results.

Usage:
    python sweep.py --nodes 1000 5000 --topologies small_world random \\
        --rules "activation" "activation majority" "activation reinforcement" \\
        --steps 2000 --seeds 5 --jobs 4
"""

import argparse
import random as _random
import csv
import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np

from simulation import create_initial_graph
from measure import correlation_function, agreement_fraction, detect_domains

# Try fast path first, fall back to NetworkX simulation
try:
    from simulation_fast import run_fast_simulation
    _HAS_FAST = True
except ImportError:
    _HAS_FAST = False

from simulation import run_simulation


# ======================================================================
# Single experiment runner (executed in worker processes)
# ======================================================================

def _run_single(
    n_nodes: int,
    topology: str,
    rules: List[str],
    n_steps: int,
    seed: int,
    use_fast: bool,
) -> Dict[str, Any]:
    """Run one simulation and return summary metrics."""
    # Seed everything for reproducibility
    _random.seed(seed)
    np.random.seed(seed)

    G = create_initial_graph(n_nodes, topology, seed=seed)

    if use_fast:
        results = run_fast_simulation(G, rules, n_steps)
    else:
        results = run_simulation(G, rules, n_steps)

    # Analyze
    final_graph = results['final_graph']
    metrics = results['metrics']

    corr = correlation_function(final_graph, 'active', max_distance=5)
    agree = agreement_fraction(final_graph, 'active')
    domains = detect_domains(final_graph, 'state')

    return {
        # Config
        'n_nodes': n_nodes,
        'topology': topology,
        'rules': ' '.join(rules),
        'n_steps': n_steps,
        'seed': seed,
        # Final metrics
        'final_active': metrics['n_active'][-1],
        'final_active_frac': metrics['n_active'][-1] / n_nodes,
        'final_mean_weight': metrics['mean_weight'][-1],
        'final_clustering': metrics['clustering'][-1],
        'clustering_change': metrics['clustering'][-1] - metrics['clustering'][0],
        'final_largest_component': metrics['largest_component'][-1],
        # Analysis
        'corr_r1': corr.get(1, 0.0),
        'corr_r2': corr.get(2, 0.0),
        'corr_r3': corr.get(3, 0.0),
        'corr_r5': corr.get(5, 0.0),
        'agreement_fraction': agree,
        'n_domains': domains['n_domains'],
        'largest_domain_frac': (
            domains['largest_domain']['fraction']
            if domains['largest_domain'] else 0.0
        ),
    }


# ======================================================================
# Verdict helpers
# ======================================================================

def _is_interesting(row: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check if a result row shows interesting emergent behavior."""
    reasons = []

    if row['agreement_fraction'] > 0.1:
        reasons.append("high distant-node agreement")
    if any(abs(row[f'corr_r{r}']) > 0.1 for r in [1, 2, 3, 5]):
        reasons.append("strong correlations")
    if row['largest_domain_frac'] > 0.3:
        reasons.append("large coherent domain")
    if abs(row['clustering_change']) > 0.05:
        reasons.append("significant clustering shift")

    return bool(reasons), reasons


# ======================================================================
# Main sweep
# ======================================================================

def run_sweep(
    nodes_list: List[int],
    topologies: List[str],
    rules_combos: List[List[str]],
    n_steps: int,
    n_seeds: int,
    n_jobs: int,
    use_fast: bool,
) -> List[Dict[str, Any]]:
    """Run a full parameter sweep and return aggregated results."""

    # Build experiment grid
    experiments = []
    for n_nodes, topology, rules in itertools.product(
        nodes_list, topologies, rules_combos
    ):
        for seed_idx in range(n_seeds):
            experiments.append({
                'n_nodes': n_nodes,
                'topology': topology,
                'rules': rules,
                'n_steps': n_steps,
                'seed': seed_idx,
                'use_fast': use_fast,
            })

    total = len(experiments)
    print(f"\nSweep: {total} experiments "
          f"({len(nodes_list)} scales x {len(topologies)} topologies "
          f"x {len(rules_combos)} rule combos x {n_seeds} seeds)")
    print(f"Backend: {'fast (sparse)' if use_fast else 'NetworkX'}")
    print(f"Workers: {n_jobs}\n")

    results = []
    t0 = time.time()

    if n_jobs == 1:
        # Sequential (easier to debug)
        for i, exp in enumerate(experiments, 1):
            print(f"  [{i}/{total}] nodes={exp['n_nodes']} "
                  f"topo={exp['topology']} rules={exp['rules']} "
                  f"seed={exp['seed']}")
            row = _run_single(**exp)
            results.append(row)
    else:
        # Parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {
                pool.submit(_run_single, **exp): exp
                for exp in experiments
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                exp = futures[future]
                try:
                    row = future.result()
                    results.append(row)
                    tag = f"nodes={exp['n_nodes']} topo={exp['topology']} " \
                          f"rules={exp['rules']}"
                    print(f"  [{done}/{total}] {tag} ... done")
                except Exception as e:
                    print(f"  [{done}/{total}] FAILED: {e}")

    elapsed = time.time() - t0
    print(f"\nCompleted {len(results)}/{total} experiments in {elapsed:.1f}s")

    return results


def print_summary(results: List[Dict[str, Any]]):
    """Print a text summary highlighting interesting results."""
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)

    interesting_rows = []
    for row in results:
        is_int, reasons = _is_interesting(row)
        if is_int:
            interesting_rows.append((row, reasons))

    if interesting_rows:
        print(f"\nInteresting results ({len(interesting_rows)}/{len(results)}):\n")
        for row, reasons in interesting_rows:
            print(f"  {row['topology']:12s} | {row['rules']:30s} | "
                  f"n={row['n_nodes']:>6d} | seed={row['seed']}")
            for r in reasons:
                print(f"    -> {r}")
            print(f"    corr(r=1)={row['corr_r1']:+.4f}  "
                  f"agree={row['agreement_fraction']:+.4f}  "
                  f"domain={row['largest_domain_frac']:.2%}  "
                  f"clust_chg={row['clustering_change']:+.4f}")
            print()
    else:
        print("\nNo strongly interesting results detected.")
        print("Try: more steps, larger scale, or different rule combinations.\n")

    # Aggregate stats per rule combo
    print("-" * 70)
    print("Per-configuration averages (across seeds):\n")
    print(f"  {'Topology':12s} | {'Rules':30s} | {'Nodes':>6s} | "
          f"{'Agree':>7s} | {'Corr1':>7s} | {'Domain':>7s} | {'ClustChg':>8s}")
    print(f"  {'-'*12} | {'-'*30} | {'-'*6} | "
          f"{'-'*7} | {'-'*7} | {'-'*7} | {'-'*8}")

    # Group by (topology, rules, n_nodes)
    groups: Dict[tuple, list] = {}
    for row in results:
        key = (row['topology'], row['rules'], row['n_nodes'])
        groups.setdefault(key, []).append(row)

    for (topo, rules, n_nodes), rows in sorted(groups.items()):
        avg_agree = np.mean([r['agreement_fraction'] for r in rows])
        avg_corr1 = np.mean([r['corr_r1'] for r in rows])
        avg_domain = np.mean([r['largest_domain_frac'] for r in rows])
        avg_clust = np.mean([r['clustering_change'] for r in rows])
        print(f"  {topo:12s} | {rules:30s} | {n_nodes:>6d} | "
              f"{avg_agree:>+7.4f} | {avg_corr1:>+7.4f} | "
              f"{avg_domain:>7.2%} | {avg_clust:>+8.4f}")

    print()


def save_csv(results: List[Dict[str, Any]], path: str):
    """Save results to CSV."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {path}")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Parameter sweep for graph-graph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example:\n'
               '  python sweep.py --nodes 1000 5000 '
               '--topologies small_world random '
               '--rules "activation" "activation majority" '
               '--steps 2000 --seeds 5 --jobs 4',
    )
    parser.add_argument(
        '--nodes', type=int, nargs='+', default=[1000],
        help='Node counts to test (default: 1000)',
    )
    parser.add_argument(
        '--topologies', type=str, nargs='+', default=['small_world'],
        choices=['small_world', 'scale_free', 'lattice', 'random'],
        help='Graph topologies (default: small_world)',
    )
    parser.add_argument(
        '--rules', type=str, nargs='+',
        default=['activation'],
        help='Rule combos as space-separated strings, e.g. '
             '"activation" "activation majority"',
    )
    parser.add_argument(
        '--steps', type=int, default=1000,
        help='Steps per simulation (default: 1000)',
    )
    parser.add_argument(
        '--seeds', type=int, default=3,
        help='Number of random seeds per configuration (default: 3)',
    )
    parser.add_argument(
        '--jobs', type=int, default=1,
        help='Parallel workers (default: 1)',
    )
    parser.add_argument(
        '--no-fast', action='store_true',
        help='Force NetworkX backend instead of sparse fast path',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output CSV path (default: results/sweep_TIMESTAMP.csv)',
    )

    args = parser.parse_args()

    # Parse rule combos: each CLI argument is a space-separated string
    rules_combos = [r.split() for r in args.rules]

    use_fast = _HAS_FAST and not args.no_fast
    if not use_fast and not args.no_fast:
        print("Note: fast path not available (scipy missing?), using NetworkX.")

    results = run_sweep(
        nodes_list=args.nodes,
        topologies=args.topologies,
        rules_combos=rules_combos,
        n_steps=args.steps,
        n_seeds=args.seeds,
        n_jobs=args.jobs,
        use_fast=use_fast,
    )

    print_summary(results)

    # Save CSV
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        args.output = str(output_dir / f"sweep_{timestamp}.csv")

    save_csv(results, args.output)


if __name__ == '__main__':
    main()
