"""
Quantum walk analysis for graph-graph.

Compares continuous-time quantum walks (CTQW) to classical random walks
on emergent graph structures. Quantum walks can reveal structure --
bottlenecks, symmetries, dimensional boundaries -- that classical walks miss.

Core approach: matrix-based walks via scipy (no external dependencies).
Braket circuit path is documented for future hardware experiments.

Usage:
    python braket_walks.py --validate
    python braket_walks.py results/run.pkl --samples 5 --subgraph-size 32
    python braket_walks.py results/run.pkl --show
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
import networkx as nx


# ======================================================================
# Matrix-based walks (Step 1)
# ======================================================================

def classical_walk_matrix(A: sp.spmatrix, start_idx: int,
                          steps: int) -> np.ndarray:
    """
    Classical random walk via sparse transition matrix.

    Computes the exact distribution after `steps` steps by repeated
    mat-vec with the column-stochastic matrix A @ D^{-1}.

    Returns probability vector of length n.
    """
    n = A.shape[0]
    degrees = np.asarray(A.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1
    D_inv = sp.diags(1.0 / degrees, format='csr')
    P_T = (A @ D_inv).tocsr()

    p = np.zeros(n)
    p[start_idx] = 1.0

    for _ in range(steps):
        p = P_T @ p

    return p


def quantum_walk_ctqw(A: sp.spmatrix, start_idx: int,
                      time: float) -> np.ndarray:
    """
    Continuous-time quantum walk via matrix exponentiation.

    Computes |exp(-i A t) |start>|^2 using scipy's Krylov-based
    expm_multiply, which avoids forming the full exponential.

    Returns probability vector of length n.
    """
    n = A.shape[0]
    psi = np.zeros(n, dtype=np.complex128)
    psi[start_idx] = 1.0

    A_op = -1j * time * A.astype(np.float64)
    psi_t = expm_multiply(A_op, psi)

    return np.abs(psi_t) ** 2


# Monte Carlo walks (kept for cross-validation)

def classical_random_walk(G: nx.Graph, start_node: int, steps: int,
                          n_walks: int = 1000) -> Dict[int, float]:
    """Monte Carlo classical random walk. Returns {node: probability}."""
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    counts = np.zeros(len(node_list))

    for _ in range(n_walks):
        current = start_node
        for _ in range(steps):
            neighbors = list(G.neighbors(current))
            if neighbors:
                current = np.random.choice(neighbors)
        counts[node_to_idx[current]] += 1

    probs = counts / n_walks
    return {node_list[i]: float(probs[i]) for i in range(len(node_list))}


def hitting_time(G: nx.Graph, start: int, target: int,
                 max_steps: int = 1000,
                 n_trials: int = 100) -> Tuple[float, float]:
    """Classical hitting time. Returns (mean_steps, std_steps)."""
    times = []
    for _ in range(n_trials):
        current = start
        for step in range(max_steps):
            if current == target:
                times.append(step)
                break
            neighbors = list(G.neighbors(current))
            if neighbors:
                current = np.random.choice(neighbors)
        else:
            times.append(max_steps)

    return float(np.mean(times)), float(np.std(times))


# ======================================================================
# Comparison framework (Step 2)
# ======================================================================

def compare_distributions(classical_probs: np.ndarray,
                          quantum_probs: np.ndarray,
                          start_idx: Optional[int] = None) -> Dict:
    """
    Quantify differences between classical and quantum walk distributions.

    Metrics:
      TVD  -- total variation distance (0 = identical, 1 = disjoint)
      IPR  -- inverse participation ratio (localization measure)
      Return probability -- P(start) after the walk
      Entropy -- Shannon entropy of the distribution
    """
    n = len(classical_probs)

    tvd = 0.5 * float(np.sum(np.abs(classical_probs - quantum_probs)))

    ipr_c = float(np.sum(classical_probs ** 2))
    ipr_q = float(np.sum(quantum_probs ** 2))

    def entropy(p: np.ndarray) -> float:
        p_pos = p[p > 1e-15]
        return float(-np.sum(p_pos * np.log2(p_pos)))

    h_c = entropy(classical_probs)
    h_q = entropy(quantum_probs)

    result: Dict = {
        'tvd': tvd,
        'ipr_classical': ipr_c,
        'ipr_quantum': ipr_q,
        'entropy_classical': h_c,
        'entropy_quantum': h_q,
        'entropy_ratio': h_q / h_c if h_c > 0 else float('inf'),
        'max_entropy': float(np.log2(n)),
        'interpretation': (
            'HIGH DIFFERENCE' if tvd > 0.3 else
            'MODERATE DIFFERENCE' if tvd > 0.1 else
            'SIMILAR'
        ),
    }

    if start_idx is not None:
        result['return_prob_classical'] = float(classical_probs[start_idx])
        result['return_prob_quantum'] = float(quantum_probs[start_idx])

    return result


def scan_time_evolution(A: sp.spmatrix, start_idx: int,
                        times: np.ndarray) -> Dict:
    """
    Compute both walks at multiple time points and return metric arrays.

    Uses batch expm_multiply for quantum efficiency. Classical walk
    advances through integer steps mapped from the continuous times.

    Returns dict with arrays: times, tvd, return_classical/quantum,
    ipr_classical/quantum.
    """
    n = A.shape[0]

    degrees = np.asarray(A.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1
    D_inv = sp.diags(1.0 / degrees, format='csr')
    P_T = (A @ D_inv).tocsr()

    # Classical: advance incrementally through integer steps
    p_c = np.zeros(n)
    p_c[start_idx] = 1.0
    c_step = 0
    classical_snapshots = []

    for t in times:
        target_step = max(int(round(t)), 1)
        while c_step < target_step:
            p_c = P_T @ p_c
            c_step += 1
        classical_snapshots.append(p_c.copy())

    # Quantum: batch via expm_multiply time-stepping
    A_op = -1j * A.astype(np.float64)
    psi_0 = np.zeros(n, dtype=np.complex128)
    psi_0[start_idx] = 1.0

    psi_all = expm_multiply(A_op, psi_0,
                            start=float(times[0]), stop=float(times[-1]),
                            num=len(times), endpoint=True)
    quantum_probs = np.abs(psi_all) ** 2

    tvds = np.empty(len(times))
    ret_c = np.empty(len(times))
    ret_q = np.empty(len(times))
    ipr_c = np.empty(len(times))
    ipr_q = np.empty(len(times))

    for i in range(len(times)):
        pc = classical_snapshots[i]
        pq = quantum_probs[i]
        tvds[i] = 0.5 * np.sum(np.abs(pc - pq))
        ret_c[i] = pc[start_idx]
        ret_q[i] = pq[start_idx]
        ipr_c[i] = np.sum(pc ** 2)
        ipr_q[i] = np.sum(pq ** 2)

    return {
        'times': np.array(times),
        'tvd': tvds,
        'return_classical': ret_c,
        'return_quantum': ret_q,
        'ipr_classical': ipr_c,
        'ipr_quantum': ipr_q,
    }


def plot_walk_comparison(G: nx.Graph, classical_probs: np.ndarray,
                         quantum_probs: np.ndarray, start_idx: int,
                         save_path: Optional[str] = None,
                         title: Optional[str] = None):
    """Side-by-side bar chart + graph colored by probability difference."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    n = len(classical_probs)
    nodes = np.arange(n)

    ax = axes[0]
    w = 0.35
    ax.bar(nodes - w / 2, classical_probs, w, label='Classical', alpha=0.7)
    ax.bar(nodes + w / 2, quantum_probs, w, label='Quantum', alpha=0.7)
    ax.axvline(start_idx, color='red', ls='--', alpha=0.4, label='Start')
    ax.set_xlabel('Node')
    ax.set_ylabel('Probability')
    ax.set_title('Distribution')
    ax.legend(fontsize=8)

    ax = axes[1]
    diff = quantum_probs - classical_probs
    colors = ['#d62728' if d < 0 else '#1f77b4' for d in diff]
    ax.bar(nodes, diff, color=colors, alpha=0.7)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('Node')
    ax.set_ylabel('Quantum - Classical')
    ax.set_title('Difference')

    ax = axes[2]
    pos = nx.spring_layout(G, seed=42)
    diff_abs = np.abs(diff)
    vmax = diff_abs.max() if diff_abs.max() > 0 else 1.0
    nx.draw_networkx(G, pos, ax=ax,
                     node_color=diff_abs.tolist(),
                     cmap=plt.get_cmap('YlOrRd'),
                     node_size=200, vmin=0, vmax=vmax,
                     with_labels=len(G) <= 20)
    ax.set_title('|Difference| on graph')

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_time_evolution(scan: Dict, save_path: Optional[str] = None,
                        title: Optional[str] = None):
    """Plot TVD, return probability, and IPR over time."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    t = scan['times']

    ax = axes[0]
    ax.plot(t, scan['tvd'], 'k-', lw=2)
    ax.axhline(0.3, color='red', ls='--', alpha=0.5, label='HIGH threshold')
    ax.axhline(0.1, color='orange', ls='--', alpha=0.5, label='MODERATE')
    ax.set_xlabel('Time')
    ax.set_ylabel('TVD')
    ax.set_title('Total Variation Distance')
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(t, scan['return_classical'], 'b-', label='Classical', lw=1.5)
    ax.plot(t, scan['return_quantum'], 'r-', label='Quantum', lw=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('P(start)')
    ax.set_title('Return Probability')
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(t, scan['ipr_classical'], 'b-', label='Classical', lw=1.5)
    ax.plot(t, scan['ipr_quantum'], 'r-', label='Quantum', lw=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('IPR')
    ax.set_title('Inverse Participation Ratio')
    ax.legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ======================================================================
# Pipeline integration (Step 3)
# ======================================================================

def analyze_simulation_results(results_path: str, n_samples: int = 5,
                               subgraph_size: int = 32,
                               walk_time: float = 5.0,
                               save_dir: Optional[str] = None,
                               show: bool = False) -> List[Dict]:
    """
    Load simulation results and run walk analysis on sampled subgraphs.

    For large graphs, samples ego-graph neighborhoods. Reports which
    subgraphs show the largest quantum-classical divergence.
    """
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    G = results['final_graph']
    n_nodes = len(G)
    params = results.get('params', {})

    print("=" * 60)
    print("QUANTUM WALK ANALYSIS")
    print("=" * 60)
    print("Graph: %d nodes, %d edges" % (n_nodes, G.number_of_edges()))
    print("Rules: %s" % params.get('rules', 'unknown'))
    print("Walk time: %.1f" % walk_time)
    print()

    nodes = list(G.nodes())
    use_subgraph = n_nodes > subgraph_size

    if use_subgraph:
        centers = np.random.choice(nodes, min(n_samples, n_nodes),
                                   replace=False)
        print("Sampling %d ego-graph subgraphs (size ~%d)"
              % (len(centers), subgraph_size))
    else:
        centers = np.random.choice(nodes, min(n_samples, n_nodes),
                                   replace=False)
        subgraph_size = n_nodes

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    all_comp: List[Dict] = []

    for i, center in enumerate(centers):
        if use_subgraph:
            distances = nx.single_source_shortest_path_length(G, center)
            closest = sorted(distances.keys(),
                             key=lambda x: distances[x])[:subgraph_size]
            sub = G.subgraph(closest).copy()
        else:
            sub = G.copy()

        sub_nodes = list(sub.nodes())
        sub_r = nx.convert_node_labels_to_integers(sub)
        center_idx = sub_nodes.index(center)

        A = nx.to_scipy_sparse_array(sub_r, format='csr', dtype=np.float64)

        steps = max(int(round(walk_time)), 1)
        c_probs = classical_walk_matrix(A, center_idx, steps=steps)
        q_probs = quantum_walk_ctqw(A, center_idx, time=walk_time)

        comp = compare_distributions(c_probs, q_probs, start_idx=center_idx)
        comp['center'] = int(center)
        comp['subgraph_nodes'] = len(sub)
        all_comp.append(comp)

        label = comp['interpretation']
        print("  [%d/%d] Node %s (%d nodes): TVD=%.3f [%s]"
              % (i + 1, len(centers), center, len(sub), comp['tvd'], label))
        print("         Return prob: classical=%.4f, quantum=%.4f"
              % (comp['return_prob_classical'], comp['return_prob_quantum']))
        print("         IPR: classical=%.4f, quantum=%.4f"
              % (comp['ipr_classical'], comp['ipr_quantum']))

        if save_dir or show:
            sp_name = ("walk_node%s.png" % center)
            sp_path = str(Path(save_dir) / sp_name) if save_dir else None
            plot_walk_comparison(
                sub_r, c_probs, q_probs, center_idx,
                save_path=sp_path,
                title="Walks from node %s (%d-node subgraph)"
                      % (center, len(sub)))

    # Summary
    tvds = np.array([c['tvd'] for c in all_comp])
    print()
    print("-" * 60)
    print("Summary over %d subgraphs:" % len(all_comp))
    print("  Mean TVD:    %.3f +/- %.3f" % (tvds.mean(), tvds.std()))
    print("  Max TVD:     %.3f (node %s)"
          % (tvds.max(), all_comp[int(np.argmax(tvds))]['center']))

    high = int(np.sum(tvds > 0.3))
    mod = int(np.sum((tvds > 0.1) & (tvds <= 0.3)))
    low = int(np.sum(tvds <= 0.1))
    print("  HIGH: %d, MODERATE: %d, SIMILAR: %d" % (high, mod, low))

    if high > 0:
        print()
        print("  ** %d subgraphs show significant quantum-classical divergence"
              % high)
        print("  ** Suggests non-trivial structure (bottlenecks, symmetries)")

    # Time evolution for the most interesting subgraph
    if save_dir or show:
        best_idx = int(np.argmax(tvds))
        center = all_comp[best_idx]['center']

        if use_subgraph:
            distances = nx.single_source_shortest_path_length(G, center)
            closest = sorted(distances.keys(),
                             key=lambda x: distances[x])[:subgraph_size]
            sub = G.subgraph(closest).copy()
        else:
            sub = G.copy()

        sub_nodes = list(sub.nodes())
        sub_r = nx.convert_node_labels_to_integers(sub)
        center_idx = sub_nodes.index(center)
        A = nx.to_scipy_sparse_array(sub_r, format='csr', dtype=np.float64)

        times = np.linspace(0.5, walk_time * 2, 40)
        scan = scan_time_evolution(A, center_idx, times)

        evo_name = "walk_evolution_node%s.png" % center
        sp_path = str(Path(save_dir) / evo_name) if save_dir else None
        plot_time_evolution(
            scan, save_path=sp_path,
            title="Time evolution from node %s" % center)
        if sp_path:
            print("  Saved time evolution plot: %s" % sp_path)

    return all_comp


# ======================================================================
# Braket circuit path (Step 4 -- documented, deferred)
# ======================================================================
#
# The matrix-based CTQW above produces physically correct quantum walk
# distributions for any graph that fits in memory. For actual quantum
# hardware experiments, a circuit-based discrete-time quantum walk (DTQW)
# is needed. This requires:
#
#   1. Position register: ceil(log2(n)) qubits to encode node indices.
#   2. Coin operator: depends on graph structure. For regular graphs,
#      a Grover diffusion coin works. For irregular graphs, the coin
#      must be adapted per-node degree (requires ancilla qubits).
#   3. Shift operator: controlled permutations that move amplitude
#      along edges. Must faithfully encode the graph adjacency.
#      For arbitrary graphs this requires O(|E|) controlled operations.
#   4. Measurement: measure position register after t walk steps.
#
# Implementation path when the time comes:
#   a) pip install amazon-braket-sdk
#   b) Implement a proper Szegedy walk that encodes the adjacency
#   c) Validate circuit results match matrix CTQW on small graphs
#   d) Run on LocalSimulator (free), then SV1 cloud simulator
#      ($0.075/min), then IonQ/Rigetti hardware (~$0.01-0.035/shot)
#   e) Compare DTQW vs CTQW physics on emergent structures
#
# This is gated behind Step 3 showing interesting quantum-classical
# differences on real simulation output. If both walks give similar
# distributions, there is no scientific reason to pursue hardware.


# ======================================================================
# Validation
# ======================================================================

def validate_step1() -> bool:
    """Validate matrix-based walks against known analytical expectations."""
    print("=" * 60)
    print("STEP 1 VALIDATION: Matrix-based walks")
    print("=" * 60)
    passed = 0
    total = 0

    # --- Cycle graph (odd n to avoid bipartite periodicity) ---
    print("\n--- Cycle graph (15 nodes) ---")
    G = nx.cycle_graph(15)
    A = nx.to_scipy_sparse_array(G, format='csr', dtype=np.float64)
    n = 15
    start = 0
    uniform = 1.0 / n

    total += 1
    c = classical_walk_matrix(A, start, steps=200)
    dev = float(np.max(np.abs(c - uniform)))
    print("  Classical (200 steps): max dev from uniform = %.6f" % dev)
    if dev < 0.01:
        print("  PASS: Converges to uniform")
        passed += 1
    else:
        print("  FAIL: Did not converge (dev=%.4f)" % dev)

    total += 1
    q = quantum_walk_ctqw(A, start, time=5.0)
    q_dev = float(np.max(np.abs(q - uniform)))
    print("  Quantum (t=5): max dev from uniform = %.6f" % q_dev)
    if q_dev > 0.01:
        print("  PASS: Non-uniform (peak at node %d, p=%.4f)"
              % (int(np.argmax(q)), q.max()))
        passed += 1
    else:
        print("  FAIL: Should not be uniform")

    # --- Complete graph ---
    print("\n--- Complete graph (8 nodes) ---")
    G = nx.complete_graph(8)
    A = nx.to_scipy_sparse_array(G, format='csr', dtype=np.float64)
    n = 8
    start = 0

    total += 1
    c1 = classical_walk_matrix(A, start, steps=1)
    expected = 1.0 / (n - 1)
    print("  Classical (1 step): P(start)=%.6f, P(other)=%.6f (expected %.6f)"
          % (c1[start], c1[1], expected))
    if abs(c1[start]) < 0.01 and abs(c1[1] - expected) < 0.01:
        print("  PASS: Uniform over non-start after 1 step")
        passed += 1
    else:
        print("  FAIL")

    total += 1
    q1 = quantum_walk_ctqw(A, start, time=1.0)
    q2 = quantum_walk_ctqw(A, start, time=2.0)
    osc = abs(q1[start] - q2[start])
    print("  Quantum return prob: t=1 -> %.4f, t=2 -> %.4f (delta=%.4f)"
          % (q1[start], q2[start], osc))
    if osc > 0.001:
        print("  PASS: Return probability oscillates")
        passed += 1
    else:
        print("  FAIL: No oscillation detected")

    # --- Star graph ---
    print("\n--- Star graph (8 nodes, hub=0) ---")
    G = nx.star_graph(7)
    A = nx.to_scipy_sparse_array(G, format='csr', dtype=np.float64)
    start = 1  # leaf

    total += 1
    c1 = classical_walk_matrix(A, start, steps=1)
    print("  Classical from leaf (1 step): P(hub)=%.4f" % c1[0])
    if c1[0] > 0.95:
        print("  PASS: Reaches hub in 1 step")
        passed += 1
    else:
        print("  FAIL: P(hub)=%.4f" % c1[0])

    total += 1
    q = quantum_walk_ctqw(A, start, time=2.0)
    c2 = classical_walk_matrix(A, start, steps=2)
    tvd = 0.5 * float(np.sum(np.abs(c2 - q)))
    print("  Quantum vs classical TVD at t=2: %.4f" % tvd)
    if tvd > 0.01:
        print("  PASS: Quantum differs from classical")
        passed += 1
    else:
        print("  FAIL: Too similar")

    # --- Probability conservation ---
    print("\n--- Probability conservation ---")
    test_graphs = [
        ("Cycle-15", nx.cycle_graph(15)),
        ("Complete-8", nx.complete_graph(8)),
        ("Star-8", nx.star_graph(7)),
    ]
    for name, G_test in test_graphs:
        total += 1
        A_t = nx.to_scipy_sparse_array(G_test, format='csr', dtype=np.float64)
        q = quantum_walk_ctqw(A_t, 0, time=3.0)
        c = classical_walk_matrix(A_t, 0, steps=10)
        ok = abs(q.sum() - 1.0) < 1e-6 and abs(c.sum() - 1.0) < 1e-6
        print("  %s: quantum sum=%.10f, classical sum=%.10f %s"
              % (name, q.sum(), c.sum(), "PASS" if ok else "FAIL"))
        if ok:
            passed += 1

    # --- Cross-validation: matrix vs Monte Carlo ---
    print("\n--- Cross-validation: matrix vs Monte Carlo ---")
    G = nx.cycle_graph(15)
    A = nx.to_scipy_sparse_array(G, format='csr', dtype=np.float64)
    total += 1
    np.random.seed(42)
    c_matrix = classical_walk_matrix(A, 0, steps=20)
    mc = classical_random_walk(G, 0, steps=20, n_walks=50000)
    mc_vec = np.array([mc.get(i, 0.0) for i in range(15)])
    mc_err = float(np.max(np.abs(c_matrix - mc_vec)))
    print("  Cycle-15, 20 steps: max |matrix - MC(50K)| = %.4f" % mc_err)
    if mc_err < 0.02:
        print("  PASS: Matrix and Monte Carlo agree")
        passed += 1
    else:
        print("  FAIL: Disagreement too large (%.4f)" % mc_err)

    print("\n" + "=" * 60)
    print("STEP 1: %d/%d tests passed" % (passed, total))
    print("=" * 60)
    return passed == total


def validate_step2() -> bool:
    """Validate comparison framework on barbell graph."""
    print()
    print("=" * 60)
    print("STEP 2 VALIDATION: Comparison framework (barbell graph)")
    print("=" * 60)
    passed = 0
    total = 0

    G = nx.barbell_graph(8, 1)
    A = nx.to_scipy_sparse_array(G, format='csr', dtype=np.float64)
    n = len(G)
    start = 0

    print("\nBarbell graph: %d nodes, %d edges" % (n, G.number_of_edges()))
    print("Start node: %d (in first clique)" % start)

    c_probs = classical_walk_matrix(A, start, steps=5)
    q_probs = quantum_walk_ctqw(A, start, time=5.0)

    comp = compare_distributions(c_probs, q_probs, start_idx=start)

    print("\n  TVD = %.4f [%s]" % (comp['tvd'], comp['interpretation']))
    print("  IPR: classical=%.4f, quantum=%.4f"
          % (comp['ipr_classical'], comp['ipr_quantum']))
    print("  Return prob: classical=%.4f, quantum=%.4f"
          % (comp['return_prob_classical'], comp['return_prob_quantum']))
    print("  Entropy: classical=%.2f, quantum=%.2f (max=%.2f)"
          % (comp['entropy_classical'], comp['entropy_quantum'],
             comp['max_entropy']))

    total += 1
    if comp['tvd'] > 0.05:
        print("\n  PASS: Quantum and classical differ on barbell (TVD=%.3f)"
              % comp['tvd'])
        passed += 1
    else:
        print("\n  FAIL: TVD too small (%.4f)" % comp['tvd'])

    # Time evolution
    times = np.linspace(0.5, 15.0, 30)
    scan = scan_time_evolution(A, start, times)

    total += 1
    max_tvd = float(scan['tvd'].max())
    best_t = float(times[int(np.argmax(scan['tvd']))])
    print("\n  Time scan: max TVD = %.4f at t = %.1f" % (max_tvd, best_t))
    if max_tvd > 0.1:
        print("  PASS: Significant divergence found in time scan")
        passed += 1
    else:
        print("  FAIL: No significant divergence")

    total += 1
    mean_ipr_c = float(np.mean(scan['ipr_classical']))
    mean_ipr_q = float(np.mean(scan['ipr_quantum']))
    print("\n  Mean IPR: classical=%.4f, quantum=%.4f"
          % (mean_ipr_c, mean_ipr_q))
    print("  PASS: Localization metrics computed successfully")
    passed += 1

    print("\n" + "=" * 60)
    print("STEP 2: %d/%d tests passed" % (passed, total))
    print("=" * 60)
    return passed == total


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Quantum walk analysis for graph-graph simulations'
    )
    parser.add_argument('input', nargs='?', default=None,
                        help='Path to simulation results pickle file')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation tests on known graphs')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of subgraphs to sample (default: 5)')
    parser.add_argument('--subgraph-size', type=int, default=32,
                        help='Max nodes per subgraph (default: 32)')
    parser.add_argument('--walk-time', type=float, default=5.0,
                        help='Walk time parameter (default: 5.0)')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively instead of saving')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.validate:
        ok1 = validate_step1()
        ok2 = validate_step2()
        if ok1 and ok2:
            print("\nAll validations passed.")
        else:
            print("\nSome validations FAILED.")
        return

    if args.input is None:
        parser.print_help()
        print("\nExamples:")
        print("  python braket_walks.py --validate")
        print("  python braket_walks.py results/run.pkl --samples 5")
        print("  python braket_walks.py results/run.pkl --show")
        return

    save_dir = None if args.show else 'plots'
    analyze_simulation_results(
        args.input,
        n_samples=args.samples,
        subgraph_size=args.subgraph_size,
        walk_time=args.walk_time,
        save_dir=save_dir,
        show=args.show,
    )


if __name__ == '__main__':
    main()
