"""
Visualization for graph-graph simulations.
"""

import pickle
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_metrics(results: dict, save_path: str | None = None):
    """Plot metric timeseries."""
    m = results['metrics']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Active nodes
    ax = axes[0, 0]
    ax.plot(m['step'], m['n_active'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Active Nodes')
    ax.set_title('Activation Dynamics')
    ax.grid(True, alpha=0.3)
    
    # Mean weight
    ax = axes[0, 1]
    ax.plot(m['step'], m['mean_weight'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Edge Weight')
    ax.set_title('Edge Weight Evolution')
    ax.grid(True, alpha=0.3)
    
    # Clustering
    ax = axes[1, 0]
    ax.plot(m['step'], m['clustering'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Clustering Coefficient')
    ax.set_title('Network Clustering')
    ax.grid(True, alpha=0.3)
    
    # Largest component
    ax = axes[1, 1]
    ax.plot(m['step'], m['largest_component'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Fraction in Largest Component')
    ax.set_title('Network Connectivity')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics plot to {save_path}")
    else:
        plt.show()


def plot_graph_state(G: nx.Graph, title: str = "Graph State", 
                     save_path: str | None = None, max_nodes: int = 500):
    """Visualize graph with node states as colors."""
    
    # Subsample if too large
    total_nodes = len(G)
    if total_nodes > max_nodes:
        nodes = list(G.nodes())[:max_nodes]
        G = G.subgraph(nodes)
        title += f" (showing {max_nodes}/{total_nodes} nodes)"
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Node colors based on state
    colors = ['red' if G.nodes[n].get('active', False) else 'lightblue' 
              for n in G.nodes()]
    
    # Edge widths based on weight
    weights = [G[u][v].get('weight', 0.5) * 2 for u, v in G.edges()]
    
    # Layout
    pos = nx.spring_layout(G, k=1/np.sqrt(len(G)), iterations=50)
    
    nx.draw_networkx(G, pos, ax=ax,
                     node_color=colors,
                     node_size=50,
                     edge_color='gray',
                     width=weights,
                     alpha=0.7,
                     with_labels=False)
    
    ax.set_title(title)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved graph plot to {save_path}")
    else:
        plt.show()


def plot_correlation_function(corr: dict, save_path: str | None = None):
    """Plot correlation function C(r) vs distance r."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    distances = sorted(corr.keys())
    values = [corr[d] for d in distances]
    
    ax.bar(distances, values, color='steelblue', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Graph Distance r')
    ax.set_ylabel('Correlation C(r)')
    ax.set_title('Two-Point Correlation Function')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved correlation plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize graph-graph results')
    parser.add_argument('input', type=str, help='Results pickle file')
    parser.add_argument('--output-dir', type=str, default='plots', 
                        help='Output directory for plots')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    
    args = parser.parse_args()
    
    print(f"Loading {args.input}...")
    with open(args.input, 'rb') as f:
        results = pickle.load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    stem = Path(args.input).stem
    
    # Plot metrics
    save = None if args.show else str(output_dir / f"{stem}_metrics.png")
    plot_metrics(results, save)
    
    # Plot final graph
    save = None if args.show else str(output_dir / f"{stem}_graph.png")
    plot_graph_state(results['final_graph'], "Final Graph State", save)
    
    print("Done!")


if __name__ == '__main__':
    main()

