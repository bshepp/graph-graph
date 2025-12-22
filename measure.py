"""
Measurement functions for graph-graph.

The key question: Do we see any interesting correlations that weren't programmed in?
"""

import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import networkx as nx
from collections import defaultdict


def correlation_function(G: nx.Graph, state_key: str = 'active', 
                         max_distance: int = 10,
                         normalize_by_variance: bool = False) -> Dict[int, float]:
    """
    Compute correlation C(r) = <s_i * s_j> for nodes at graph distance r.
    
    This measures if nodes at distance r have correlated states.
    For random states, C(r) ≈ <s>² for all r.
    Interesting if C(r) decays slowly or has structure.
    """
    # Get states as +1/-1
    states = {}
    for node in G.nodes():
        s = G.nodes[node].get(state_key, False)
        states[node] = 1 if s else -1
    
    mean_state = np.mean(list(states.values()))
    
    # Sample node pairs at each distance
    correlations = defaultdict(list)
    nodes = list(G.nodes())
    
    # Sample for efficiency
    n_samples = min(1000, len(nodes))
    sample_nodes = np.random.choice(nodes, n_samples, replace=False)
    
    for source in sample_nodes:
        distances = nx.single_source_shortest_path_length(G, source, cutoff=max_distance)
        for target, dist in distances.items():
            if dist > 0:  # Skip self
                corr = states[source] * states[target]
                correlations[dist].append(corr)
    
    # Average
    result = {}
    variance = np.var(list(states.values()))
    
    for dist in range(1, max_distance + 1):
        if correlations[dist]:
            raw_corr = np.mean(correlations[dist]) - mean_state**2
            if normalize_by_variance and variance > 0:
                # Normalized correlation: C(r) / Var(s) 
                # Ranges from -1 to 1, more interpretable for asymmetric states
                result[dist] = raw_corr / variance
            else:
                result[dist] = raw_corr
        else:
            result[dist] = 0.0
    
    return result


def mutual_information_estimate(G: nx.Graph, state_key: str = 'active',
                                n_bins: int = 2) -> float:
    """
    Estimate mutual information between distant node pairs.
    
    High MI suggests non-trivial correlations.
    """
    # Get states
    states = [G.nodes[n].get(state_key, False) for n in G.nodes()]
    
    # For binary states, just compute correlation
    states = np.array([1 if s else 0 for s in states])
    
    # Get pairs at distance > diameter/2
    nodes = list(G.nodes())
    n = len(nodes)
    
    if n < 10:
        return 0.0
    
    # Sample distant pairs
    mi_sum = 0
    n_pairs = 0
    
    for _ in range(min(500, n * 10)):
        i, j = np.random.choice(n, 2, replace=False)
        try:
            dist = nx.shortest_path_length(G, nodes[i], nodes[j])
            if dist > 3:  # "Distant" pairs
                # Joint and marginal probabilities
                si, sj = states[i], states[j]
                mi_sum += si == sj  # Simple correlation proxy
                n_pairs += 1
        except nx.NetworkXNoPath:
            continue
    
    if n_pairs == 0:
        return 0.0
    
    return mi_sum / n_pairs - 0.5  # Subtract random baseline


def detect_domains(G: nx.Graph, state_key: str = 'state') -> Dict[str, Any]:
    """
    Detect if coherent domains have formed (regions of same state).
    
    Returns stats about domain structure.
    """
    # Group nodes by state
    state_groups = defaultdict(set)
    for node in G.nodes():
        s = G.nodes[node].get(state_key, 0)
        state_groups[s].add(node)
    
    # Find connected components within each state group
    domains = []
    for state, nodes in state_groups.items():
        subgraph = G.subgraph(nodes)
        for component in nx.connected_components(subgraph):
            domains.append({
                'state': state,
                'size': len(component),
                'fraction': len(component) / len(G),
            })
    
    # Sort by size
    domains.sort(key=lambda x: x['size'], reverse=True)
    
    return {
        'n_domains': len(domains),
        'largest_domain': domains[0] if domains else None,
        'domain_sizes': [d['size'] for d in domains],
        'domains': domains[:10],  # Top 10
    }


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full analysis of simulation results.
    """
    G = results['final_graph']
    metrics = results['metrics']
    
    print("Analyzing final graph state...")
    
    # Correlation function
    print("  Computing correlation function...")
    corr = correlation_function(G, 'active')
    
    # Mutual information
    print("  Estimating mutual information...")
    mi = mutual_information_estimate(G, 'active')
    
    # Domain detection
    print("  Detecting domains...")
    domains = detect_domains(G, 'state')
    
    analysis = {
        'correlation_function': corr,
        'mutual_information': mi,
        'domains': domains,
        'metrics_summary': {
            'final_active_fraction': metrics['n_active'][-1] / results['params']['n_nodes'],
            'active_variance': np.var(metrics['n_active']),
            'clustering_final': metrics['clustering'][-1],
            'clustering_change': metrics['clustering'][-1] - metrics['clustering'][0],
        }
    }
    
    return analysis


def print_analysis(analysis: Dict[str, Any]):
    """Pretty print analysis results."""
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    print("\n📊 Correlation Function C(r):")
    corr = analysis['correlation_function']
    for r in sorted(corr.keys()):
        bar = '█' * int(abs(corr[r]) * 50)
        sign = '+' if corr[r] >= 0 else '-'
        print(f"  r={r:2d}: {sign}{abs(corr[r]):.4f} {bar}")
    
    print(f"\n🔗 Mutual Information Estimate: {analysis['mutual_information']:.4f}")
    print("   (0 = random, >0.1 = interesting correlations)")
    
    print(f"\n🗺️ Domain Structure:")
    d = analysis['domains']
    print(f"   Number of domains: {d['n_domains']}")
    if d['largest_domain']:
        print(f"   Largest domain: {d['largest_domain']['fraction']*100:.1f}% of graph")
    
    print(f"\n📈 Metrics Summary:")
    m = analysis['metrics_summary']
    print(f"   Final active fraction: {m['final_active_fraction']:.3f}")
    print(f"   Active count variance: {m['active_variance']:.1f}")
    print(f"   Clustering change: {m['clustering_change']:+.4f}")
    
    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    
    interesting = False
    reasons = []
    
    if analysis['mutual_information'] > 0.1:
        interesting = True
        reasons.append("Non-trivial mutual information detected")
    
    if any(abs(c) > 0.1 for c in corr.values() if corr):
        interesting = True
        reasons.append("Long-range correlations present")
    
    if d['largest_domain'] and d['largest_domain']['fraction'] > 0.3:
        interesting = True
        reasons.append("Large coherent domains formed")
    
    if interesting:
        print("🌟 INTERESTING: Potential emergent structure detected!")
        for r in reasons:
            print(f"   - {r}")
    else:
        print("😐 BASELINE: No strong emergent correlations (yet)")
        print("   Try: more steps, different rules, larger scale")


def main():
    parser = argparse.ArgumentParser(description='Analyze graph-graph results')
    parser.add_argument('input', type=str, help='Results pickle file')
    parser.add_argument('--save', type=str, default=None, help='Save analysis to file')
    
    args = parser.parse_args()
    
    print(f"Loading {args.input}...")
    with open(args.input, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Loaded: {results['params']}")
    
    analysis = analyze_results(results)
    print_analysis(analysis)
    
    if args.save:
        with open(args.save, 'wb') as f:
            pickle.dump(analysis, f)
        print(f"\nAnalysis saved to {args.save}")


if __name__ == '__main__':
    main()

