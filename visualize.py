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
    
    # Subsample if too large (random sample for unbiased view)
    total_nodes = len(G)
    if total_nodes > max_nodes:
        nodes = np.random.choice(list(G.nodes()), max_nodes, replace=False)
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


def plot_dimension_map(G: nx.Graph, dim_field: dict,
                       title: str = "Effective Dimension Map",
                       save_path: str | None = None, max_nodes: int = 500):
    """
    Visualize graph with nodes colored by local effective dimension d_eff.

    Args:
        G:         NetworkX graph.
        dim_field: Output of dimension.dimension_field() --
                   {node: (d_eff, r_squared, ball_sizes)}.
        title:     Plot title.
        save_path: If set, save to file instead of showing.
        max_nodes: Subsample threshold for large graphs.
    """
    total_nodes = len(G)
    if total_nodes > max_nodes:
        nodes = np.random.choice(list(G.nodes()), max_nodes, replace=False)
        G = G.subgraph(nodes)
        title += f" (showing {max_nodes}/{total_nodes} nodes)"

    fig, ax = plt.subplots(figsize=(12, 12))

    # Build color array: d_eff for sampled nodes, NaN for unsampled
    node_list = list(G.nodes())
    d_effs = np.array([
        dim_field[n][0] if n in dim_field else np.nan
        for n in node_list
    ])

    # Separate sampled vs unsampled nodes for drawing
    sampled_mask = ~np.isnan(d_effs)
    sampled_nodes = [n for n, m in zip(node_list, sampled_mask) if m]
    unsampled_nodes = [n for n, m in zip(node_list, sampled_mask) if not m]
    sampled_d_effs = d_effs[sampled_mask]

    pos = nx.spring_layout(G, k=1 / np.sqrt(len(G)), iterations=50)

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray',
                           alpha=0.3, width=0.5)

    # Draw unsampled nodes in gray
    if unsampled_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=unsampled_nodes, ax=ax,
                               node_color='lightgray', node_size=30,
                               alpha=0.4)

    # Draw sampled nodes with d_eff colormap
    if len(sampled_nodes) > 0:
        sc = nx.draw_networkx_nodes(
            G, pos, nodelist=sampled_nodes, ax=ax,
            node_color=sampled_d_effs.tolist(), cmap=plt.get_cmap('viridis'),
            node_size=50, alpha=0.85,
        )
        plt.colorbar(sc, ax=ax, label='$d_{eff}$', shrink=0.7)

    ax.set_title(title)
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved dimension map to {save_path}")
    else:
        plt.show()


def plot_dimension_histogram(dim_field: dict,
                             save_path: str | None = None):
    """
    Histogram of local effective dimension values.

    Args:
        dim_field: Output of dimension.dimension_field().
        save_path: If set, save to file instead of showing.
    """
    d_effs = np.array([v[0] for v in dim_field.values()])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(d_effs, bins=30, color='steelblue', alpha=0.7, edgecolor='white')

    # Reference lines at integer dimensions
    for d in [1, 2, 3, 4]:
        ax.axvline(x=d, color='black', linestyle='--', alpha=0.4, linewidth=1)

    mean_d = float(np.mean(d_effs))
    std_d = float(np.std(d_effs))
    ax.axvline(x=mean_d, color='red', linestyle='-', alpha=0.6, linewidth=1.5,
               label=f'mean = {mean_d:.2f} $\\pm$ {std_d:.2f}')

    ax.set_xlabel('Effective Dimension $d_{eff}$')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Local Effective Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved dimension histogram to {save_path}")
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
    parser.add_argument('--dimension', action='store_true',
                        help='Compute and plot local effective dimension')
    
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
    
    # Dimension analysis (opt-in)
    if args.dimension:
        from dimension import dimension_field, dimension_stats, print_dimension_analysis

        G = results['final_graph']
        print("Computing dimension field...")
        dim = dimension_field(G)
        max_radius = dim[next(iter(dim))][2][-1][0] if dim else 0

        stats = dimension_stats(dim, len(G))
        print_dimension_analysis(stats, max_radius)

        save = None if args.show else str(output_dir / f"{stem}_dimension_map.png")
        plot_dimension_map(G, dim, save_path=save)

        save = None if args.show else str(output_dir / f"{stem}_dimension_hist.png")
        plot_dimension_histogram(dim, save_path=save)
    
    print("Done!")


if __name__ == '__main__':
    main()

