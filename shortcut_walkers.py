"""
Who can use the portal? Classical vs quantum walkers through a shortcut.

A shortcut edge reshapes graph distance without breaking the local update
bound (1 edge/step) -- metric engineering, not speed violation. But a
shorter metric only matters if a dynamical process actually EXPLOITS it.
This experiment measures the transport advantage a single portal gives to:

  * a classical random walker -- exact absorbing-walk arrival CDF at the
    target (median hitting time), diffusive baseline;
  * a continuous-time quantum walker (CTQW, H = adjacency) -- peak transfer
    probability at the target and its arrival time, ballistic baseline.

Conditions per seed: no portal / portal offset near both endpoints
(ends at distance ~2 from source and target) / portal directly source->target.

A-priori expectation to test: the classical walker benefits monotonically
(any extra path helps diffusion reach the far region), while the quantum
walker's benefit is interference-limited -- a lone portal is a weak link
between two large regions and may transfer little amplitude, so its speedup
can be much smaller or absent. Either outcome is a finding.

Usage:
    python shortcut_walkers.py --quick
    python shortcut_walkers.py --nodes 1500 --seeds 5
"""

import argparse
import random
from typing import Dict, Optional, Tuple

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from tqdm import tqdm

from simulation import create_initial_graph
from shortcuts import far_node


def absorbing_arrival(G: nx.Graph, source: int, target: int,
                      max_steps: int) -> Tuple[float, float]:
    """
    Exact classical arrival at an absorbing target.

    Random walk with the target made absorbing; returns (median_step,
    absorbed_mass) where median_step is the first step at which >= 50% of
    the walk's probability mass has been absorbed (inf if never within
    max_steps).
    """
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight=None,
                                 format='csr', dtype=np.float64)
    deg = np.asarray(A.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    P_T = (A @ sp.diags(1.0 / deg)).tocsr()   # column-stochastic

    t_idx, s_idx = idx[target], idx[source]
    p = np.zeros(len(nodes))
    p[s_idx] = 1.0
    absorbed = 0.0
    median = float('inf')

    for step in range(1, max_steps + 1):
        p = P_T @ p
        absorbed += p[t_idx]
        p[t_idx] = 0.0                        # absorb
        if absorbed >= 0.5 and median == float('inf'):
            median = step
            break

    return median, absorbed


def ctqw_peak(G: nx.Graph, source: int, target: int,
              t_max: float, n_times: int = 160) -> Tuple[float, float]:
    """
    CTQW transfer to the target: returns (peak probability, time of peak)
    of |<target| exp(-iAt) |source>|^2 over t in (0, t_max].
    """
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight=None,
                                 format='csr', dtype=np.float64)
    psi0 = np.zeros(len(nodes), dtype=np.complex128)
    psi0[idx[source]] = 1.0

    traj = expm_multiply(-1j * A, psi0, start=0.0, stop=t_max,
                         num=n_times, endpoint=True)
    probs = np.abs(traj[:, idx[target]]) ** 2
    k = int(np.argmax(probs[1:]) + 1)         # exclude t=0
    times = np.linspace(0.0, t_max, n_times)
    return float(probs[k]), float(times[k])


def node_near(G: nx.Graph, anchor: int, distance: int) -> Optional[int]:
    """A random node at exactly `distance` hops from anchor (None if none)."""
    dist = nx.single_source_shortest_path_length(G, anchor, cutoff=distance)
    ring = [u for u, d in dist.items() if d == distance]
    if not ring:
        return None
    return ring[np.random.randint(len(ring))]


def main():
    parser = argparse.ArgumentParser(
        description="Classical vs quantum walker advantage through a portal.")
    parser.add_argument('--nodes', type=int, default=1500)
    parser.add_argument('--topology', type=str, default='grown')
    parser.add_argument('--cap', type=int, default=6)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=200_000,
                        help='Classical absorbing-walk step budget.')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if args.quick:
        args.nodes, args.seeds, args.max_steps = 400, 2, 50_000

    conditions = ['none', 'offset', 'direct']
    agg: Dict[str, Dict[str, list]] = {
        c: {'t_cls': [], 'absorbed': [], 'q_peak': [], 'q_time': []}
        for c in conditions}
    distances = []

    for s in tqdm(range(args.seeds), desc='seeds'):
        seed = args.seed + s
        random.seed(seed)
        np.random.seed(seed)
        base = create_initial_graph(args.nodes, topology=args.topology,
                                    k=args.cap, seed=seed)
        nodes = list(base.nodes())
        source = nodes[np.random.randint(len(nodes))]
        target, dist_st = far_node(base, source)
        distances.append(dist_st)
        t_max = 3.0 * dist_st                 # ballistic scale ~ distance

        for cond in conditions:
            G = base.copy()
            if cond == 'direct':
                G.add_edge(source, target, weight=0.5, shortcut=True)
            elif cond == 'offset':
                a = node_near(G, source, 2)
                b = node_near(G, target, 2)
                if a is None or b is None or G.has_edge(a, b):
                    continue
                G.add_edge(a, b, weight=0.5, shortcut=True)

            t_cls, mass = absorbing_arrival(G, source, target, args.max_steps)
            q_peak, q_time = ctqw_peak(G, source, target, t_max)
            agg[cond]['t_cls'].append(t_cls)
            agg[cond]['absorbed'].append(mass)
            agg[cond]['q_peak'].append(q_peak)
            agg[cond]['q_time'].append(q_time)

    print(f"\nPortal walkers -- {args.topology} N={args.nodes}, "
          f"{args.seeds} seed(s), source-target distance "
          f"{np.mean(distances):.1f}±{np.std(distances):.1f}")
    header = (f"{'condition':>9s} {'cls median t':>13s} {'q peak P':>10s} "
              f"{'q peak t':>9s}")
    print(header)
    print('-' * len(header))
    base_cls = np.median(agg['none']['t_cls']) if agg['none']['t_cls'] else 1.0
    base_qp = np.median(agg['none']['q_peak']) if agg['none']['q_peak'] else 1.0
    for cond in conditions:
        a = agg[cond]
        if not a['t_cls']:
            continue
        tc = np.median(a['t_cls'])
        qp = np.median(a['q_peak'])
        qt = np.median(a['q_time'])
        print(f"{cond:>9s} {tc:13.0f} {qp:10.2e} {qt:9.1f}")
    print(f"\nSpeedups vs no-portal (medians): "
          f"classical direct x{base_cls / np.median(agg['direct']['t_cls']):.1f}, "
          f"offset x{base_cls / np.median(agg['offset']['t_cls']):.1f}; "
          f"quantum peak-P gain direct "
          f"x{np.median(agg['direct']['q_peak']) / base_qp:.1f}, "
          f"offset x{np.median(agg['offset']['q_peak']) / base_qp:.1f}")


if __name__ == '__main__':
    main()
