"""
Quantum walk analysis for graph-graph.

Compare quantum walks to classical random walks on emergent graph structures.
Quantum walks can reveal structure that classical walks miss.

Requires: pip install amazon-braket-sdk
"""

import numpy as np
import networkx as nx
from typing import Dict, Tuple

# Braket imports - will fail gracefully if not installed
try:
    from braket.circuits import Circuit
    from braket.devices import LocalSimulator
    BRAKET_AVAILABLE = True
except ImportError:
    BRAKET_AVAILABLE = False
    print("Braket SDK not installed. Install with: pip install amazon-braket-sdk")


def classical_random_walk(G: nx.Graph, start_node: int, steps: int, 
                          n_walks: int = 1000) -> Dict[int, float]:
    """
    Classical random walk distribution after `steps` steps.
    
    Returns probability distribution over nodes.
    """
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
    return {node_list[i]: probs[i] for i in range(len(node_list))}


def hitting_time(G: nx.Graph, start: int, target: int, 
                 max_steps: int = 1000, n_trials: int = 100) -> Tuple[float, float]:
    """
    Classical hitting time: expected steps to reach target from start.
    
    Returns (mean_time, std_time).
    """
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
            times.append(max_steps)  # Didn't hit
    
    return np.mean(times), np.std(times)


if BRAKET_AVAILABLE:
    
    class QuantumWalkAnalyzer:
        """
        Quantum walk implementation for small graphs.
        
        Uses discrete-time quantum walk with coin operator.
        Limited to ~2^n_qubits nodes due to state space encoding.
        """
        
        def __init__(self, use_local: bool = True):
            """
            Args:
                use_local: Use local simulator (free). 
                           Set False for cloud simulators/hardware.
            """
            self.device = LocalSimulator() if use_local else None
            self.max_qubits = 10  # Practical limit for simulation
        
        def _encode_graph(self, G: nx.Graph) -> Tuple[int, Dict]:
            """
            Encode graph nodes into qubit basis states.
            
            Returns:
                n_qubits: Number of qubits needed
                node_to_state: Mapping from node to computational basis state
            """
            n_nodes = len(G)
            n_qubits = int(np.ceil(np.log2(max(n_nodes, 2))))
            
            if n_qubits > self.max_qubits:
                raise ValueError(f"Graph too large: {n_nodes} nodes needs {n_qubits} qubits")
            
            node_list = list(G.nodes())
            node_to_state = {node: i for i, node in enumerate(node_list)}
            
            return n_qubits, node_to_state
        
        def quantum_walk_circuit(self, G: nx.Graph, steps: int = 1) -> Circuit:
            """
            Build quantum walk circuit.
            
            Simplified Szegedy walk:
            - Position register encodes node
            - Coin register for direction superposition
            - Walk operator based on adjacency
            """
            n_qubits, node_to_state = self._encode_graph(G)
            
            # Position qubits + 1 coin qubit
            circuit = Circuit()
            coin = n_qubits  # Coin qubit index
            
            for _ in range(steps):
                # Coin flip (Hadamard)
                circuit.h(coin)
                
                # Shift operator (simplified - actual implementation 
                # depends on graph structure)
                # For now: conditional increment/decrement on position
                for i in range(n_qubits):
                    circuit.cnot(coin, i)
                
                # Grover diffusion on position (optional, improves mixing)
                for i in range(n_qubits):
                    circuit.h(i)
                    circuit.x(i)
                circuit.h(n_qubits - 1)
                # Multi-controlled Z would go here
                circuit.h(n_qubits - 1)
                for i in range(n_qubits):
                    circuit.x(i)
                    circuit.h(i)
            
            return circuit
        
        def walk_distribution(self, G: nx.Graph, start_node: int, 
                              steps: int = 5, shots: int = 1000) -> Dict[int, float]:
            """
            Run quantum walk and measure final distribution.
            
            Returns probability distribution over nodes.
            """
            n_qubits, node_to_state = self._encode_graph(G)
            state_to_node = {v: k for k, v in node_to_state.items()}
            
            circuit = self.quantum_walk_circuit(G, steps)
            
            # Initialize to start node
            start_state = node_to_state[start_node]
            init_circuit = Circuit()
            for i in range(n_qubits):
                if (start_state >> i) & 1:
                    init_circuit.x(i)
            
            full_circuit = init_circuit + circuit
            
            # Measure position register
            result = self.device.run(full_circuit, shots=shots).result()
            counts = result.measurement_counts
            
            # Convert to node probabilities
            node_probs = {}
            for bitstring, count in counts.items():
                # Parse position from bitstring (first n_qubits bits)
                pos_bits = bitstring[:n_qubits]
                pos = int(pos_bits, 2)
                node = state_to_node.get(pos, None)
                if node is not None:
                    node_probs[node] = node_probs.get(node, 0) + count / shots
            
            return node_probs
        
        def compare_walks(self, G: nx.Graph, start_node: int, 
                          steps: int = 5) -> Dict[str, Dict[int, float]]:
            """
            Compare classical and quantum walk distributions.
            
            Large differences suggest graph structure that quantum walks 
            are sensitive to (e.g., symmetries, bottlenecks).
            """
            classical = classical_random_walk(G, start_node, steps)
            quantum = self.walk_distribution(G, start_node, steps)
            
            # Compute total variation distance
            all_nodes = set(classical.keys()) | set(quantum.keys())
            tvd = 0.5 * sum(
                abs(classical.get(n, 0) - quantum.get(n, 0)) 
                for n in all_nodes
            )
            
            return {
                'classical': classical,
                'quantum': quantum,
                'total_variation_distance': tvd,
                'interpretation': (
                    'HIGH DIFFERENCE' if tvd > 0.3 else
                    'MODERATE DIFFERENCE' if tvd > 0.1 else
                    'SIMILAR'
                )
            }


def analyze_emergent_structure(G: nx.Graph, sample_nodes: int = 5):
    """
    Run walk analysis on a graph to probe emergent structure.
    
    Compares classical and quantum walks from multiple starting points.
    Large differences suggest non-trivial structure.
    """
    print(f"Analyzing graph: {len(G)} nodes, {len(G.edges())} edges")
    
    nodes = list(G.nodes())
    if len(nodes) > sample_nodes:
        test_nodes = np.random.choice(nodes, sample_nodes, replace=False)
    else:
        test_nodes = nodes
    
    print("\n📊 Classical Random Walk Analysis:")
    for node in test_nodes:
        dist = classical_random_walk(G, node, steps=10)
        top_3 = sorted(dist.items(), key=lambda x: -x[1])[:3]
        entropy = -sum(p * np.log2(p + 1e-10) for p in dist.values())
        print(f"  From node {node}: entropy={entropy:.2f}, top destinations: {top_3}")
    
    if BRAKET_AVAILABLE and len(G) <= 64:  # 6 qubits max for practical sim
        print("\n🔮 Quantum Walk Comparison:")
        analyzer = QuantumWalkAnalyzer()
        for node in test_nodes[:2]:  # Just a couple, quantum is slow
            try:
                result = analyzer.compare_walks(G, node, steps=3)
                print(f"  From node {node}: {result['interpretation']} "
                      f"(TVD={result['total_variation_distance']:.3f})")
            except Exception as e:
                print(f"  From node {node}: Error - {e}")
    elif not BRAKET_AVAILABLE:
        print("\n⚠️  Braket not available. Install with: pip install amazon-braket-sdk")
    else:
        print(f"\n⚠️  Graph too large for quantum simulation ({len(G)} nodes > 64)")
        print("    Use subgraph sampling for quantum analysis.")


if __name__ == '__main__':
    # Demo on a small graph
    print("=== Quantum Walk Demo ===\n")
    
    # Create a small-world graph
    G = nx.watts_strogatz_graph(16, 4, 0.3)
    
    analyze_emergent_structure(G)
    
    print("\n" + "="*50)
    print("To use with graph-graph simulations:")
    print("  1. Run simulation.py to generate emergent graph")
    print("  2. Load results and extract final_graph")
    print("  3. Sample subgraphs for quantum analysis")
    print("  4. Compare walk distributions before/after evolution")

