# Scaling Roadmap for graph-graph

## Current Limits (NetworkX on CPU)

| Scale | Nodes | Edges (~6 per node) | RAM | Time per step |
|-------|-------|---------------------|-----|---------------|
| Small | 1K | 6K | ~50 MB | ~10 ms |
| Medium | 10K | 60K | ~500 MB | ~200 ms |
| Large | 100K | 600K | ~5 GB | ~5 sec |
| **Limit** | ~500K | ~3M | ~25 GB | ~30 sec |

Beyond 500K nodes, NetworkX becomes impractical due to:
- Python object overhead (each node/edge is a dict)
- Single-threaded execution
- No vectorization

---

## Upgrade Path

### Phase 1: Optimized CPU (up to 1M nodes)

**NumPy/SciPy sparse matrices**

Replace NetworkX with sparse adjacency matrices for the hot path:

```python
import scipy.sparse as sp
import numpy as np

# Adjacency matrix (CSR format)
A = nx.to_scipy_sparse_array(G, format='csr')

# Node states as dense array
states = np.array([G.nodes[n]['active'] for n in G.nodes()])

# Activation spread in ONE vectorized operation
neighbor_counts = A @ states  # Matrix-vector multiply
activation_prob = 1 - (1 - spread_prob) ** neighbor_counts
new_active = np.random.random(len(states)) < activation_prob
```

**Benefits:**
- 10-50x speedup on CPU
- 1M nodes feasible
- Keep NetworkX for initialization and measurement only

**Implementation:** Create `rules_fast.py` with NumPy equivalents.

---

### Phase 2: GPU Acceleration (1M - 100M nodes)

**Option A: PyTorch Geometric**

```python
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# Convert once
data = from_networkx(G)
data.x = torch.tensor(states, dtype=torch.float32).cuda()
edge_index = data.edge_index.cuda()

# Scatter operations for neighbor aggregation
from torch_scatter import scatter_add
neighbor_sums = scatter_add(data.x[edge_index[0]], edge_index[1])
```

**Option B: RAPIDS cuGraph**

```python
import cudf
import cugraph

# Create GPU graph
edges_df = cudf.DataFrame({'src': sources, 'dst': targets})
G_gpu = cugraph.Graph()
G_gpu.from_cudf_edgelist(edges_df, source='src', destination='dst')

# GPU-native algorithms
clustering = cugraph.clustering_coefficient(G_gpu)
```

**Performance:**
| Library | 1M nodes | 10M nodes | 100M nodes |
|---------|----------|-----------|------------|
| PyTorch Geometric | ~100ms/step | ~1s/step | ~10s/step |
| cuGraph | ~50ms/step | ~500ms/step | ~5s/step |

**Hardware:** NVIDIA GPU with 24GB+ VRAM (RTX 4090, A100)

---

### Phase 3: Batch Processing (Parameter Sweeps)

**Local: Joblib parallelization**

```python
from joblib import Parallel, delayed

def run_experiment(seed, nodes, rules):
    np.random.seed(seed)
    G = create_initial_graph(nodes)
    return run_simulation(G, rules, n_steps=1000)

# Run 100 experiments in parallel
results = Parallel(n_jobs=-1)(
    delayed(run_experiment)(seed, 10000, ['activation', 'majority'])
    for seed in range(100)
)
```

**AWS Batch (recommended for large sweeps)**

```yaml
# job-definition.json
{
  "jobDefinitionName": "graph-graph-sweep",
  "type": "container",
  "containerProperties": {
    "image": "your-ecr-repo/graph-graph:latest",
    "vcpus": 4,
    "memory": 16000,
    "command": [
      "python", "simulation.py",
      "--nodes", "Ref::nodes",
      "--rules", "Ref::rules", 
      "--seed", "Ref::seed",
      "--output", "s3://your-bucket/results/run_${AWS_BATCH_JOB_ID}.pkl"
    ]
  }
}
```

**Workflow:**
1. Build Docker image with dependencies
2. Push to ECR
3. Submit array job with 1000 variants
4. Results land in S3
5. Aggregate locally

**Cost estimate:** 1000 runs × 10K nodes × 10 min = ~$20 (spot instances)

---

### Phase 4: GPU Clusters (100M+ nodes, serious research)

**AWS ParallelCluster with GPU nodes**

For graphs that don't fit on a single GPU:

```python
# Distributed graph partitioning
import torch.distributed as dist
from torch_geometric.distributed import DistNeighborLoader

# Each GPU handles a partition
# Message passing crosses partition boundaries
```

**Alternative: GraphScope (Alibaba)**
- Handles trillion-edge graphs
- Kubernetes-native
- Good for production-scale analysis

---

## AWS Braket Integration

Braket enables quantum algorithms on graphs that behave fundamentally differently 
from classical algorithms. This is valuable for:

1. **Comparison** — Do quantum walks find different structure than classical walks?
2. **Probing** — Quantum algorithms as measurement tools on emergent structures
3. **Hybrid** — Classical rule evolution + quantum analysis passes

---

### Quantum Walks on Graphs

Quantum walks spread amplitude differently than classical random walks. On certain 
graph structures, they exhibit:
- Faster hitting times
- Different stationary distributions  
- Interference patterns that reveal graph structure

```python
from braket.aws import AwsDevice
from braket.circuits import Circuit
import numpy as np

def quantum_walk_step(n_nodes: int, adjacency: np.ndarray) -> Circuit:
    """
    One step of a discrete-time quantum walk.
    Encodes graph structure in the coin operator.
    """
    circuit = Circuit()
    
    # Position register: log2(n_nodes) qubits
    n_position_qubits = int(np.ceil(np.log2(n_nodes)))
    
    # Coin qubit for superposition of directions
    coin_qubit = n_position_qubits
    
    # Hadamard coin flip
    circuit.h(coin_qubit)
    
    # Conditional shift based on adjacency
    # (simplified - real implementation uses controlled swaps)
    for i in range(n_position_qubits):
        circuit.cnot(coin_qubit, i)
    
    return circuit

# Run on simulator first
device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")

# Then on real hardware
# device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/ionq/Harmony")
```

### Graph Structure via Quantum Algorithms

**Grover search for subgraph patterns:**
```python
def grover_subgraph_search(G, pattern):
    """
    Use Grover's algorithm to find nodes matching a pattern.
    Quadratic speedup over classical search.
    """
    # Oracle marks nodes where local structure matches pattern
    # Diffusion operator amplifies marked states
    # O(√N) vs O(N) for classical
    pass
```

**Quantum approximate optimization (QAOA) for graph partitioning:**
```python
from braket.circuits import Circuit

def qaoa_layer(G, gamma, beta):
    """
    One QAOA layer for MaxCut-style problems.
    Can find graph partitions/communities.
    """
    circuit = Circuit()
    
    # Cost unitary (based on edges)
    for u, v in G.edges():
        circuit.zz(u, v, gamma)
    
    # Mixer unitary
    for node in G.nodes():
        circuit.rx(node, 2 * beta)
    
    return circuit
```

### Practical Braket Workflow

```python
# braket_analysis.py

from braket.aws import AwsDevice
from braket.devices import LocalSimulator
import networkx as nx

class BraketGraphAnalyzer:
    """Quantum analysis tools for graph-graph."""
    
    def __init__(self, use_hardware: bool = False):
        if use_hardware:
            # IonQ for small circuits, Rigetti for larger
            self.device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/ionq/Harmony")
        else:
            # Local simulator (free, fast)
            self.device = LocalSimulator()
    
    def quantum_walk_distribution(self, G: nx.Graph, steps: int = 10,
                                   start_node: int = 0) -> dict:
        """
        Run quantum walk and return probability distribution over nodes.
        Compare to classical random walk distribution.
        """
        # Build circuit for `steps` walk iterations
        # Measure position register
        # Return {node: probability}
        pass
    
    def find_communities_qaoa(self, G: nx.Graph, p: int = 2) -> list:
        """
        Use QAOA to find graph communities/partitions.
        Compare to classical Louvain algorithm.
        """
        pass
    
    def interference_signature(self, G: nx.Graph) -> np.ndarray:
        """
        Quantum walks create interference patterns unique to graph structure.
        This could be a 'fingerprint' of emergent structure.
        """
        pass
```

### Cost & Scale Considerations

| Backend | Max Qubits | Cost | Use Case |
|---------|------------|------|----------|
| LocalSimulator | ~25 | Free | Development, small graphs |
| SV1 (state vector) | 34 | $0.075/min | Medium graphs, exact |
| TN1 (tensor network) | 50 | $0.075/min | Sparse circuits |
| DM1 (density matrix) | 17 | $0.075/min | Noisy simulation |
| IonQ Harmony | 11 | ~$0.01/shot | Small real experiments |
| IonQ Aria | 25 | ~$0.03/shot | Larger experiments |
| Rigetti Ankaa | 84 | ~$0.035/shot | Biggest circuits |

**Qubit requirements for graph problems:**
- N nodes → log₂(N) qubits for position + ancilla
- 1000 nodes → ~10 position qubits + overhead ≈ 15-20 qubits total
- Current hardware limit: ~100 nodes for direct encoding

**Hybrid approach for larger graphs:**
1. Run classical simulation to generate graph
2. Sample subgraphs for quantum analysis
3. Use quantum results to inform measurements on full graph

---

## Recommended Progression

| Phase | When | Investment |
|-------|------|------------|
| 1. NumPy sparse | Now | 1-2 days coding |
| 2. PyTorch Geometric | When 100K isn't enough | 3-5 days + GPU |
| 3. AWS Batch | For parameter sweeps | ~$50-200/sweep |
| 4. Braket (simulator) | When probing emergent structure | Free locally, ~$5-20/experiment |
| 5. Distributed | If results are exciting | Significant |
| 6. Braket (hardware) | Comparing quantum vs classical walks | ~$50-200/experiment |

### The Emergence Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  GENERATE                          PROBE                        │
│  ────────                          ─────                        │
│  Simple local rules                Walks & measurements          │
│  on massive graph                  on emergent structure         │
│                                                                  │
│  ┌─────────┐    evolve    ┌─────────┐    analyze    ┌─────────┐ │
│  │ Initial │ ──────────▶  │Emergent │ ───────────▶  │ Metrics │ │
│  │  Graph  │   (rules)    │Structure│   (walks)     │ & Plots │ │
│  └─────────┘              └─────────┘               └─────────┘ │
│       │                        │                         │      │
│       ▼                        ▼                         ▼      │
│  NetworkX/NumPy           Classical walks           Correlation │
│  PyTorch Geometric        Quantum walks (Braket)    MI, Domains │
│  cuGraph                  Subgraph sampling         Phase plots │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start: NumPy Acceleration

Create `simulation_fast.py`:

```python
"""
Fast simulation using sparse matrices.
NetworkX only for I/O, NumPy for computation.
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx

class FastGraph:
    """Sparse representation for fast updates."""
    
    def __init__(self, G: nx.Graph):
        self.n_nodes = len(G)
        self.node_list = list(G.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.node_list)}
        
        # Sparse adjacency
        self.A = nx.to_scipy_sparse_array(G, format='csr', dtype=np.float32)
        
        # Dense state vectors
        self.active = np.array([G.nodes[n].get('active', False) 
                                for n in self.node_list], dtype=np.float32)
        self.state = np.array([G.nodes[n].get('state', 0) 
                               for n in self.node_list], dtype=np.int32)
        
        # Edge weights (sparse)
        self.weights = self.A.copy()
        self.weights.data[:] = 0.5
    
    def activation_spread(self, spread_prob=0.3, decay_prob=0.1):
        # Decay active nodes
        decay_mask = np.random.random(self.n_nodes) < decay_prob
        
        # Count active neighbors (sparse matrix-vector multiply)
        active_neighbor_count = self.A @ self.active
        
        # Activation probability
        activation_prob = 1 - (1 - spread_prob) ** active_neighbor_count
        activate_mask = np.random.random(self.n_nodes) < activation_prob
        
        # Update: decay existing, activate new
        self.active = np.where(self.active, ~decay_mask, activate_mask).astype(np.float32)
    
    def to_networkx(self) -> nx.Graph:
        """Convert back to NetworkX for measurement/visualization."""
        G = nx.from_scipy_sparse_array(self.A)
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['active'] = bool(self.active[i])
            G.nodes[node]['state'] = int(self.state[i])
        return G
```

This gives you 10-50x speedup immediately with minimal code changes.

