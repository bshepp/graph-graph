"""
Who can use the portal? Classical vs quantum walkers through a shortcut.

A shortcut edge reshapes graph distance without breaking the local update
bound (1 edge/step) -- metric engineering, not speed violation. But a
shorter metric only matters if a dynamical process actually EXPLOITS it.
This experiment measures the transport advantage a single portal gives to:

  * a classical random walker -- exact absorbing-walk arrival CDF at the
    target (median hitting time), diffusive baseline;
  * a continuous-time quantum walker (CTQW) -- peak transfer probability at
    the target and its arrival time, ballistic baseline.

Conditions per seed: no portal / portal offset near both endpoints
(ends at distance ~2 from source and target) / portal directly source->target.

A-priori expectation to test: the classical walker benefits monotonically
(any extra path helps diffusion reach the far region), while the quantum
walker's benefit is interference-limited -- a lone portal is a weak link
between two large regions and may transfer little amplitude, so its speedup
can be much smaller or absent. Either outcome is a finding.

CTQW generator (`--generators`): H = adjacency A, or H = graph Laplacian
L = D - A. On a REGULAR graph these are the same experiment -- L = kI - A, so
the two evolutions differ by a global phase and a time reversal, both of which
drop out of |<target|psi(t)>|^2. Any difference between them is therefore a
pure degree-heterogeneity effect, which is exactly the confound the Laplacian
cross-check exists to expose: on irregular graphs (like `grown`) an
adjacency-generated CTQW couples to degree, so a measured portal advantage
could be a degree artifact rather than interference. Both generators run on
the SAME graphs and the same source/target pairs, so the comparison is paired.

Usage:
    python shortcut_walkers.py --quick
    python shortcut_walkers.py --nodes 1500 --seeds 5
    python shortcut_walkers.py --nodes 1500 --seeds 5 --generators laplacian
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


def ctqw_peak(G: nx.Graph, source: int, target: int, t_max: float,
              n_times: int = 160,
              generator: str = 'adjacency') -> Tuple[float, float, bool]:
    """
    CTQW transfer to the target: returns (peak probability, time of peak,
    peak_at_edge) of |<target| exp(-iHt) |source>|^2 over t in (0, t_max].

    `generator` selects H: 'adjacency' (H = A) or 'laplacian' (H = D - A).
    `peak_at_edge` is True when the maximum lands in the last 5% of the time
    grid -- a flag that t_max was too short to contain the peak, so the
    reported value is a lower bound rather than a peak.
    """
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight=None,
                                 format='csr', dtype=np.float64)
    if generator == 'adjacency':
        H = A
    elif generator == 'laplacian':
        deg = np.asarray(A.sum(axis=1)).flatten()
        H = (sp.diags(deg) - A).tocsr()
    else:
        raise ValueError(f"unknown generator {generator!r}")

    psi0 = np.zeros(len(nodes), dtype=np.complex128)
    psi0[idx[source]] = 1.0

    traj = expm_multiply(-1j * H, psi0, start=0.0, stop=t_max,
                         num=n_times, endpoint=True)
    probs = np.abs(traj[:, idx[target]]) ** 2
    k = int(np.argmax(probs[1:]) + 1)         # exclude t=0
    times = np.linspace(0.0, t_max, n_times)
    return float(probs[k]), float(times[k]), k >= int(0.95 * (n_times - 1))


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
    parser.add_argument('--t-max-factor', type=float, default=3.0,
                        help='CTQW time horizon as a multiple of the '
                             'source-target distance. The no-portal baseline '
                             'peak is a max over this window, so gains are '
                             'only comparable at matched t_max -- sweep it.')
    parser.add_argument('--generators', type=str, nargs='+',
                        default=['adjacency', 'laplacian'],
                        choices=['adjacency', 'laplacian'],
                        help='CTQW generator(s); all run on the same graphs.')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if args.quick:
        args.nodes, args.seeds, args.max_steps = 400, 2, 50_000

    conditions = ['none', 'offset', 'direct']
    agg: Dict[str, Dict[str, list]] = {
        c: {'t_cls': [], 'absorbed': []} for c in conditions}
    qagg: Dict[tuple, Dict[str, list]] = {
        (c, g): {'peak': [], 'time': [], 'edge': []}
        for c in conditions for g in args.generators}
    distances = []
    skipped = {c: 0 for c in conditions}

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
        t_max = args.t_max_factor * dist_st   # ballistic scale ~ distance

        # Choose the offset endpoints BEFORE any solver call. scipy's
        # expm_multiply estimates operator norms with onenormest, which draws
        # from the GLOBAL numpy RNG -- so selecting the portal inside the
        # condition loop would make which portal gets placed depend on how
        # many generators are being run, and on whether a quantum solve
        # happened earlier in the loop.
        a = node_near(base, source, 2)
        b = node_near(base, target, 2)
        offset_ends = (None if (a is None or b is None or base.has_edge(a, b))
                       else (a, b))

        for cond in conditions:
            G = base.copy()
            if cond == 'direct':
                G.add_edge(source, target, weight=0.5, shortcut=True)
            elif cond == 'offset':
                if offset_ends is None:
                    skipped[cond] += 1
                    continue
                G.add_edge(*offset_ends, weight=0.5, shortcut=True)

            t_cls, mass = absorbing_arrival(G, source, target, args.max_steps)
            agg[cond]['t_cls'].append(t_cls)
            agg[cond]['absorbed'].append(mass)
            for gen in args.generators:
                q_peak, q_time, at_edge = ctqw_peak(G, source, target, t_max,
                                                    generator=gen)
                qagg[(cond, gen)]['peak'].append(q_peak)
                qagg[(cond, gen)]['time'].append(q_time)
                qagg[(cond, gen)]['edge'].append(at_edge)

    print(f"\nPortal walkers -- {args.topology} N={args.nodes}, "
          f"{args.seeds} seed(s), source-target distance "
          f"{np.mean(distances):.1f}±{np.std(distances):.1f}")
    if any(skipped.values()):
        print("NOTE: portal placement failed for "
              + ", ".join(f"{n} {c} seed(s)" for c, n in skipped.items() if n)
              + " -- those conditions have fewer samples than 'none'.")

    base_cls = np.median(agg['none']['t_cls']) if agg['none']['t_cls'] else 1.0
    header = (f"{'condition':>9s} {'n':>3s} {'cls median t':>13s} "
              f"{'cls gain':>9s}")
    print('\n' + header)
    print('-' * len(header))
    for cond in conditions:
        a = agg[cond]
        if not a['t_cls']:
            continue
        tc = np.median(a['t_cls'])
        print(f"{cond:>9s} {len(a['t_cls']):3d} {tc:13.0f} "
              f"{base_cls / tc:8.1f}x")

    for gen in args.generators:
        base_qp = (np.median(qagg[('none', gen)]['peak'])
                   if qagg[('none', gen)]['peak'] else 1.0)
        header = (f"{'condition':>9s} {'n':>3s} {'q peak P':>10s} "
                  f"{'q peak t':>9s} {'q gain':>9s} {'at t_max':>9s}")
        print(f"\nCTQW generator: H = {gen}")
        print(header)
        print('-' * len(header))
        for cond in conditions:
            q = qagg[(cond, gen)]
            if not q['peak']:
                continue
            qp = np.median(q['peak'])
            qt = np.median(q['time'])
            print(f"{cond:>9s} {len(q['peak']):3d} {qp:10.2e} {qt:9.1f} "
                  f"{qp / base_qp:8.1f}x {sum(q['edge']):5d}/"
                  f"{len(q['edge']):<3d}")
        if any(sum(qagg[(c, gen)]['edge']) for c in conditions):
            print("  WARNING: peak(s) landed at the end of the time grid -- "
                  "those values are lower bounds, not peaks (raise t_max).")

    if len(args.generators) > 1:
        print("\nCross-check: a portal gain that holds for BOTH generators is "
              "interference, not a degree artifact of H = A.")


if __name__ == '__main__':
    main()
