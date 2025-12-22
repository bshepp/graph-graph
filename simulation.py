"""
Main simulation loop for graph-graph.

Creates a graph, applies rules, records history.
"""

import argparse
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import networkx as nx
from tqdm import tqdm

from rules import get_rule, RULES


def create_initial_graph(n_nodes: int, topology: str = 'small_world',
                         k: int = 6, p: float = 0.1) -> nx.Graph:
    """Create initial graph topology."""
    
    if topology == 'small_world':
        G = nx.watts_strogatz_graph(n_nodes, k, p)
    elif topology == 'scale_free':
        G = nx.barabasi_albert_graph(n_nodes, k // 2)
    elif topology == 'lattice':
        side = int(np.sqrt(n_nodes))
        G = nx.grid_2d_graph(side, side)
        G = nx.convert_node_labels_to_integers(G)
    elif topology == 'random':
        G = nx.erdos_renyi_graph(n_nodes, k / n_nodes)
    else:
        raise ValueError(f"Unknown topology: {topology}")
    
    # Initialize node states
    for node in G.nodes():
        G.nodes[node]['active'] = np.random.random() < 0.1  # 10% initially active
        G.nodes[node]['state'] = np.random.randint(2)  # Binary state
    
    # Initialize edge weights
    for u, v in G.edges():
        G[u][v]['weight'] = 0.5
    
    return G


def run_simulation(G: nx.Graph, rules: List[str], n_steps: int,
                   record_interval: int = 10) -> Dict[str, Any]:
    """
    Run simulation and record history.
    
    Returns dict with:
        - 'snapshots': List of graph copies at intervals
        - 'metrics': Dict of metric timeseries
        - 'params': Simulation parameters
    """
    
    rule_funcs = [get_rule(r) for r in rules]
    
    # Metrics to track
    metrics = {
        'step': [],
        'n_active': [],
        'mean_weight': [],
        'clustering': [],
        'largest_component': [],
    }
    
    snapshots = []
    
    for step in tqdm(range(n_steps), desc="Simulating"):
        # Apply each rule
        for rule_func in rule_funcs:
            G = rule_func(G)
        
        # Record metrics
        if step % record_interval == 0:
            metrics['step'].append(step)
            metrics['n_active'].append(
                sum(1 for n in G.nodes() if G.nodes[n].get('active', False))
            )
            weights = [G[u][v].get('weight', 0.5) for u, v in G.edges()]
            metrics['mean_weight'].append(np.mean(weights) if weights else 0)
            metrics['clustering'].append(nx.average_clustering(G))
            
            if len(G) > 0:
                largest_cc = max(nx.connected_components(G), key=len)
                metrics['largest_component'].append(len(largest_cc) / len(G))
            else:
                metrics['largest_component'].append(0)
        
        # Save snapshots less frequently
        if step % (record_interval * 10) == 0:
            snapshots.append(G.copy())
    
    return {
        'snapshots': snapshots,
        'metrics': metrics,
        'final_graph': G,
        'params': {
            'n_nodes': len(G),
            'rules': rules,
            'n_steps': n_steps,
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Run graph-graph simulation')
    parser.add_argument('--nodes', type=int, default=1000, help='Number of nodes')
    parser.add_argument('--steps', type=int, default=1000, help='Simulation steps')
    parser.add_argument('--topology', type=str, default='small_world',
                        choices=['small_world', 'scale_free', 'lattice', 'random'])
    parser.add_argument('--rules', type=str, nargs='+', default=['activation'],
                        choices=list(RULES.keys()))
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--output', type=str, default=None, help='Output file')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
    
    print(f"Creating {args.topology} graph with {args.nodes} nodes...")
    G = create_initial_graph(args.nodes, args.topology)
    
    print(f"Running {args.steps} steps with rules: {args.rules}")
    results = run_simulation(G, args.rules, args.steps)
    
    # Save results
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        args.output = output_dir / f"run_{timestamp}.pkl"
    
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {args.output}")
    
    # Print summary
    m = results['metrics']
    print(f"\nSummary:")
    print(f"  Final active nodes: {m['n_active'][-1]} / {args.nodes}")
    print(f"  Final mean edge weight: {m['mean_weight'][-1]:.3f}")
    print(f"  Final clustering: {m['clustering'][-1]:.3f}")


if __name__ == '__main__':
    main()

