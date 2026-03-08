"""
Fast simulation using sparse matrices.
NetworkX only for I/O, NumPy/SciPy for computation.

Provides 10-50x speedup over simulation.py for large graphs.

Usage:
    python simulation_fast.py --nodes 10000 --steps 2000 --rules activation reinforcement
"""

import argparse
import random
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import networkx as nx
from tqdm import tqdm

from simulation import create_initial_graph
from rules import RULES


class FastGraph:
    """
    Sparse matrix representation for fast rule updates.

    Stores graph structure as CSR adjacency matrix and node states
    as dense NumPy arrays.  Rule updates are vectorized using
    sparse mat-vec multiplies instead of Python-level node iteration.
    """

    def __init__(self, G: nx.Graph):
        self.n_nodes = len(G)
        self.node_list = list(G.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.node_list)}

        # Binary adjacency (structure only, all 1s)
        self.A = nx.to_scipy_sparse_array(
            G, weight=None, format='csr', dtype=np.float32
        )

        # Dense state vectors
        self.active = np.array(
            [G.nodes[n].get('active', False) for n in self.node_list],
            dtype=np.float32,
        )
        self.state = np.array(
            [G.nodes[n].get('state', 0) for n in self.node_list],
            dtype=np.int32,
        )

        # Edge weights (same sparsity pattern as A)
        self.weights = nx.to_scipy_sparse_array(
            G, weight='weight', format='csr', dtype=np.float32
        )

        # Precompute degrees
        self.degrees = np.array(self.A.sum(axis=1)).flatten()

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    def activation_spread(self, spread_prob: float = 0.3,
                          decay_prob: float = 0.1):
        """Vectorized activation spreading (SIS model)."""
        # Count active neighbors via sparse mat-vec multiply
        active_neighbor_count = self.A @ self.active

        # Inactive nodes: probability of activation from neighbors
        activation_prob = 1.0 - (1.0 - spread_prob) ** active_neighbor_count
        activate_mask = np.random.random(self.n_nodes) < activation_prob

        # Active nodes: probability of decay
        decay_mask = np.random.random(self.n_nodes) < decay_prob

        was_active = self.active.astype(bool)
        self.active = np.where(
            was_active, ~decay_mask, activate_mask
        ).astype(np.float32)

    def edge_reinforcement(self, reinforce_amount: float = 0.1,
                           decay_amount: float = 0.01,
                           min_weight: float = 0.01,
                           max_weight: float = 1.0):
        """Vectorized Hebbian edge-weight update."""
        coo = self.weights.tocoo()

        # Decay all weights
        coo.data -= decay_amount

        # Reinforce co-active edges
        active_bool = self.active.astype(bool)
        mask = active_bool[coo.row] & active_bool[coo.col]
        coo.data[mask] += reinforce_amount

        # Clamp
        np.clip(coo.data, min_weight, max_weight, out=coo.data)

        self.weights = coo.tocsr()

    def majority_vote(self, num_states: int = 2, noise: float = 0.01):
        """Vectorized majority vote via sparse one-hot multiply."""
        # One-hot state matrix (n_nodes x num_states)
        one_hot = np.zeros((self.n_nodes, num_states), dtype=np.float32)
        one_hot[np.arange(self.n_nodes), self.state] = 1.0

        # neighbor_counts[i, s] = number of neighbors of i in state s
        neighbor_counts = (self.A @ one_hot)  # dense (n x num_states)

        new_states = np.argmax(neighbor_counts, axis=1).astype(np.int32)

        # Isolated nodes keep current state
        isolated = self.degrees == 0
        new_states[isolated] = self.state[isolated]

        # Apply noise
        noise_mask = np.random.random(self.n_nodes) < noise
        noise_states = np.random.randint(
            0, num_states, self.n_nodes, dtype=np.int32
        )
        new_states[noise_mask] = noise_states[noise_mask]

        self.state = new_states

    def random_rewire(self, rewire_prob: float = 0.01):
        """Vectorized edge rewiring via COO manipulation."""
        coo_a = self.A.tocoo()
        coo_w = self.weights.tocoo()

        upper = coo_a.row < coo_a.col
        upper_idx = np.where(upper)[0]
        n_upper = len(upper_idx)

        if n_upper == 0:
            return

        rewire_mask = np.random.random(n_upper) < rewire_prob
        if not rewire_mask.any():
            return

        sel = np.where(rewire_mask)[0]
        coo_sel = upper_idx[sel]
        us = coo_a.row[coo_sel]
        old_vs = coo_a.col[coo_sel]
        old_ws = coo_w.data[coo_sel]

        new_vs = np.random.randint(0, self.n_nodes, size=len(us))

        valid = new_vs != us
        if valid.any():
            existing = np.asarray(
                self.A[us[valid], new_vs[valid]]
            ).flatten()
            sub_valid = existing == 0
            valid_indices = np.where(valid)[0]
            valid[valid_indices[~sub_valid]] = False

        if not valid.any():
            return

        us_v = us[valid]
        old_vs_v = old_vs[valid]
        new_vs_v = new_vs[valid]
        ws_v = old_ws[valid]
        n = self.n_nodes

        remove_fwd = us_v.astype(np.int64) * n + old_vs_v.astype(np.int64)
        remove_rev = old_vs_v.astype(np.int64) * n + us_v.astype(np.int64)
        remove_ids = np.concatenate([remove_fwd, remove_rev])

        all_ids = coo_a.row.astype(np.int64) * n + coo_a.col.astype(np.int64)
        keep = ~np.isin(all_ids, remove_ids)

        new_row = np.concatenate([coo_a.row[keep], us_v, new_vs_v])
        new_col = np.concatenate([coo_a.col[keep], new_vs_v, us_v])
        new_da = np.ones(len(new_row), dtype=np.float32)
        new_dw = np.concatenate([coo_w.data[keep], ws_v, ws_v])

        shape = (n, n)
        self.A = sp.csr_matrix(
            (new_da, (new_row, new_col)), shape=shape,
        )
        self.weights = sp.csr_matrix(
            (new_dw, (new_row, new_col)), shape=shape,
        )
        self.degrees = np.asarray(self.A.sum(axis=1)).flatten()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(self) -> Dict[str, float]:
        """Compute simulation metrics using sparse operations."""
        n_active = int(self.active.sum())

        # Mean edge weight
        w_coo = self.weights.tocoo()
        mean_weight = float(w_coo.data.mean()) if len(w_coo.data) > 0 else 0.0

        # Clustering coefficient via A^3 diagonal trick:
        #   triangles_i = (A^3)_{ii} / 2   (undirected)
        #   C_i = triangles_i / (d_i choose 2) = (A^3)_{ii} / (d_i*(d_i-1))
        # We avoid computing full A^3 by using:
        #   diag(A^3) = row-sums of (A^2 ∘ A)   (Hadamard product)
        A2 = self.A @ self.A
        A3_diag = np.array(A2.multiply(self.A).sum(axis=1)).flatten()
        denom = self.degrees * (self.degrees - 1)
        mask = denom > 0
        clustering = np.zeros(self.n_nodes)
        clustering[mask] = A3_diag[mask] / denom[mask]
        avg_clustering = float(clustering.mean())

        # Largest connected component via scipy
        if self.n_nodes > 0:
            _n_comp, labels = connected_components(self.A, directed=False)
            comp_sizes = np.bincount(labels)
            largest_frac = float(comp_sizes.max()) / self.n_nodes
        else:
            largest_frac = 0.0

        return {
            'n_active': n_active,
            'mean_weight': mean_weight,
            'clustering': avg_clustering,
            'largest_component': largest_frac,
        }

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_networkx(self) -> nx.Graph:
        """Convert back to NetworkX for measurement/visualization."""
        # Build graph from weighted adjacency
        G = nx.from_scipy_sparse_array(self.weights)

        # Relabel nodes back to original labels when needed
        if self.node_list != list(range(self.n_nodes)):
            mapping = {i: self.node_list[i] for i in range(self.n_nodes)}
            G = nx.relabel_nodes(G, mapping)

        # Attach node attributes
        for i, node in enumerate(self.node_list):
            G.nodes[node]['active'] = bool(self.active[i])
            G.nodes[node]['state'] = int(self.state[i])

        return G


# ======================================================================
# Fast rule dispatch
# ======================================================================

FAST_RULES = {
    'activation': 'activation_spread',
    'reinforcement': 'edge_reinforcement',
    'majority': 'majority_vote',
    'rewire': 'random_rewire',
}


def run_fast_simulation(G: nx.Graph, rules: List[str], n_steps: int,
                        record_interval: int = 10,
                        snapshot_interval: int = 0) -> Dict[str, Any]:
    """
    Fast simulation using sparse matrices.

    Same interface and output format as simulation.run_simulation()
    for compatibility with measure.py and visualize.py.
    """
    n_nodes_initial = len(G)
    fg = FastGraph(G)

    # Validate rules and resolve methods
    rule_methods = []
    for r in rules:
        if r not in FAST_RULES:
            raise ValueError(
                f"Unknown rule: {r}. Available: {list(FAST_RULES.keys())}"
            )
        rule_methods.append(getattr(fg, FAST_RULES[r]))

    metrics: Dict[str, list] = {
        'step': [],
        'n_active': [],
        'mean_weight': [],
        'clustering': [],
        'largest_component': [],
    }

    snapshots = []

    for step in tqdm(range(n_steps), desc="Simulating (fast)"):
        # Apply rules
        for rule_method in rule_methods:
            rule_method()

        # Record metrics
        if step % record_interval == 0:
            m = fg.compute_metrics()
            metrics['step'].append(step)
            metrics['n_active'].append(m['n_active'])
            metrics['mean_weight'].append(m['mean_weight'])
            metrics['clustering'].append(m['clustering'])
            metrics['largest_component'].append(m['largest_component'])

        # Snapshots (opt-in)
        if snapshot_interval > 0 and step % snapshot_interval == 0:
            snapshots.append(fg.to_networkx())

    return {
        'snapshots': snapshots,
        'metrics': metrics,
        'final_graph': fg.to_networkx(),
        'params': {
            'n_nodes': n_nodes_initial,
            'rules': rules,
            'n_steps': n_steps,
        },
    }


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run graph-graph simulation (fast sparse path)'
    )
    parser.add_argument('--nodes', type=int, default=1000,
                        help='Number of nodes')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Simulation steps')
    parser.add_argument('--topology', type=str, default='small_world',
                        choices=['small_world', 'scale_free', 'lattice', 'random'])
    parser.add_argument('--rules', type=str, nargs='+', default=['activation'],
                        choices=list(FAST_RULES.keys()))
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--snapshot-interval', type=int, default=0,
                        help='Save graph snapshots every N steps (0=disabled)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output pickle file')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    print(f"Creating {args.topology} graph with {args.nodes} nodes...")
    G = create_initial_graph(args.nodes, args.topology, seed=args.seed)

    print(f"Running {args.steps} steps with rules: {args.rules} [FAST]")
    results = run_fast_simulation(
        G, args.rules, args.steps,
        snapshot_interval=args.snapshot_interval,
    )

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        args.output = output_dir / f"run_fast_{timestamp}.pkl"

    with open(args.output, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to {args.output}")

    m = results['metrics']
    print(f"\nSummary:")
    print(f"  Final active nodes: {m['n_active'][-1]} / {args.nodes}")
    print(f"  Final mean edge weight: {m['mean_weight'][-1]:.3f}")
    print(f"  Final clustering: {m['clustering'][-1]:.3f}")


if __name__ == '__main__':
    main()
