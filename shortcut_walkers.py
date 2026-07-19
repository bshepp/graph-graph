"""
Who can use the portal? Classical vs quantum walkers through a shortcut.

A shortcut edge reshapes graph distance without breaking the local update
bound (1 edge/step) -- metric engineering, not speed violation. But a
shorter metric only matters if a dynamical process actually EXPLOITS it.
This experiment measures the transport advantage a single portal gives to:

  * a classical random walker -- exact absorbing-walk arrival CDF at the
    target (median hitting time), plus the stationary occupancy
    pi(target) = deg(target)/2|E|;
  * a continuous-time quantum walker (CTQW) -- the exact infinite-time
    average transfer probability Pbar, plus peak transfer over a window.

The PRIMARY observables are the horizon-free pair, pi(target) and Pbar: same
units, same question (what fraction of the long run is spent at the target),
neither depending on a time cutoff. `peak transfer` is retained only as a
transient diagnostic and is explicitly horizon-dependent -- it is a maximum
over (0, t_max], and the no-portal baseline is a running max of a wandering
amplitude, so it grows with the window and any gain ratio built from it is a
statement about an arbitrary horizon rather than about the graph. An earlier
version of this experiment reported peak-based gains as the headline; see
FINDINGS.md "3b" for the retraction and "3c" for the replacement.

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
    python shortcut_walkers.py --validate
    python shortcut_walkers.py --quick
    python shortcut_walkers.py --nodes 1500 --seeds 20
    python shortcut_walkers.py --nodes 1500 --seeds 20 --generators laplacian
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


def _hamiltonian(G: nx.Graph, nodes: list, generator: str) -> sp.spmatrix:
    """Sparse CTQW generator: adjacency A, or graph Laplacian D - A."""
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight=None,
                                 format='csr', dtype=np.float64)
    if generator == 'adjacency':
        return A
    if generator == 'laplacian':
        deg = np.asarray(A.sum(axis=1)).flatten()
        return (sp.diags(deg) - A).tocsr()
    raise ValueError(f"unknown generator {generator!r}")


def ctqw_time_average(G: nx.Graph, source: int, target: int,
                      generator: str = 'adjacency',
                      rel_tol: float = 1e-8) -> Tuple[float, int]:
    """
    Horizon-free CTQW transfer: the infinite-time average

        Pbar = lim_{T->inf} (1/T) \\int_0^T |<target| exp(-iHt) |source>|^2 dt

    computed EXACTLY from the eigendecomposition rather than by integrating
    to some cutoff. Writing a_k = <target|phi_k><phi_k|source> (real, since H
    is real symmetric), the cross terms oscillate as exp(-i(l_k - l_j)t) and
    time-average to zero unless l_k == l_j, leaving

        Pbar = sum over distinct eigenvalues l of ( sum_{k: l_k = l} a_k )^2.

    Degenerate eigenvalues MUST be grouped -- states sharing an eigenvalue
    never dephase relative to each other, so their amplitudes add coherently
    before squaring. Treating every k separately silently under-counts
    transfer on any symmetric graph.

    Unlike a peak-over-a-window, this has no free time parameter. Cost is a
    dense symmetric eigendecomposition, O(N^3), so N is limited to a few
    thousand.

    Returns (Pbar, n_distinct_eigenvalues).
    """
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    H = _hamiltonian(G, nodes, generator).toarray()

    w, V = np.linalg.eigh(H)
    a = V[idx[target], :] * V[idx[source], :]

    # Group consecutive (eigh returns sorted) eigenvalues within tolerance.
    scale = max(float(np.abs(w).max()), 1.0)
    tol = rel_tol * scale
    boundaries = np.nonzero(np.diff(w) > tol)[0] + 1
    groups = np.split(a, boundaries)

    pbar = float(sum(float(g.sum()) ** 2 for g in groups))
    return pbar, len(groups)


def stationary_occupancy(G: nx.Graph, target: int) -> float:
    """
    Horizon-free CLASSICAL analogue: long-run fraction of time a random
    walker spends at the target, pi(v) = deg(v) / 2|E|. Same units as the
    CTQW time average, so the two are directly comparable -- and it is set
    entirely by degree, which is the point.
    """
    return G.degree(target) / (2.0 * G.number_of_edges())


def node_near(G: nx.Graph, anchor: int, distance: int) -> Optional[int]:
    """A random node at exactly `distance` hops from anchor (None if none)."""
    dist = nx.single_source_shortest_path_length(G, anchor, cutoff=distance)
    ring = [u for u, d in dist.items() if d == distance]
    if not ring:
        return None
    return ring[np.random.randint(len(ring))]


def _numerical_time_average(G: nx.Graph, source: int, target: int,
                            generator: str, t_max: float,
                            n_times: int) -> float:
    """
    Independent estimate of the CTQW time average: propagate with
    `expm_multiply` (Krylov / scaling-and-squaring) and average the target
    probability over a long uniform grid. Deliberately does NOT use eigh, so
    it is a genuine cross-check of `ctqw_time_average` rather than a
    restatement of it. Converges as ~1/(T * min eigenvalue gap).
    """
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    H = _hamiltonian(G, nodes, generator)
    psi0 = np.zeros(len(nodes), dtype=np.complex128)
    psi0[idx[source]] = 1.0
    traj = expm_multiply(-1j * H, psi0, start=0.0, stop=t_max,
                         num=n_times, endpoint=True)
    return float((np.abs(traj[:, idx[target]]) ** 2).mean())


def _validate() -> bool:
    """
    Self-check for the horizon-free observable. Cases are chosen so that the
    degeneracy grouping is load-bearing: on the star and the cycle, the
    degeneracy-blind sum (treating every eigenvector separately) gives a
    materially different answer, so a passing grouped result is evidence the
    grouping works rather than evidence the tolerance is loose.
    """
    print("Validating ctqw_time_average ...\n")
    ok = True

    # 1. K_2, analytic. |<1|exp(-iAt)|0>|^2 = sin^2(t), time average = 1/2.
    pbar, ngroups = ctqw_time_average(nx.complete_graph(2), 0, 1)
    err = abs(pbar - 0.5)
    good = err < 1e-12
    ok &= good
    print(f"  K_2 vs analytic 1/2            : Pbar={pbar:.6f} "
          f"err={err:.1e} groups={ngroups} "
          f"{'OK' if good else 'FAIL'}")

    # 2-4. Grouped closed form vs independent long-time propagation.
    cases = [
        ('star K_1,5 (leaf->leaf)', nx.star_graph(5), 1, 2, 'adjacency'),
        ('cycle C_6 (opposite)', nx.cycle_graph(6), 0, 3, 'adjacency'),
        ('cycle C_6 laplacian', nx.cycle_graph(6), 0, 3, 'laplacian'),
        ('grown N=60', create_initial_graph(60, topology='grown', k=6,
                                            seed=0), None, None, 'adjacency'),
    ]
    for name, G, s, t, gen in cases:
        if s is None:                       # pick a far pair for grown
            s = list(G.nodes())[0]
            t, _ = far_node(G, s)
        pbar, ngroups = ctqw_time_average(G, s, t, generator=gen)
        num = _numerical_time_average(G, s, t, gen, t_max=4000.0,
                                      n_times=40_000)
        rel = abs(pbar - num) / max(num, 1e-12)
        good = rel < 0.05
        ok &= good
        # Degeneracy-blind control: does grouping actually change anything?
        nodes = list(G.nodes())
        idx = {u: i for i, u in enumerate(nodes)}
        H = _hamiltonian(G, nodes, gen).toarray()
        _w, V = np.linalg.eigh(H)
        a = V[idx[t], :] * V[idx[s], :]
        blind = float((a ** 2).sum())
        blind_rel = abs(blind - num) / max(num, 1e-12)
        print(f"  {name:<30s}: Pbar={pbar:.6f} numeric={num:.6f} "
              f"rel={rel:.3f} groups={ngroups}/{len(a)} "
              f"{'OK' if good else 'FAIL'}")
        print(f"  {'':<30s}  degeneracy-blind={blind:.6f} "
              f"(rel={blind_rel:.3f} -- grouping "
              f"{'matters here' if blind_rel > 0.05 else 'is a no-op here'})")

    print(f"\n{'PASS' if ok else 'FAIL'}: horizon-free transfer observable")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Classical vs quantum walker advantage through a portal.")
    parser.add_argument('--validate', action='store_true',
                        help='Self-check the horizon-free observable and exit.')
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

    if args.validate:
        raise SystemExit(0 if _validate() else 1)

    if args.quick:
        args.nodes, args.seeds, args.max_steps = 400, 2, 50_000

    conditions = ['none', 'offset', 'direct']
    agg: Dict[str, Dict[str, list]] = {
        c: {'t_cls': [], 'absorbed': [], 'pi': []} for c in conditions}
    qagg: Dict[tuple, Dict[str, list]] = {
        (c, g): {'peak': [], 'time': [], 'edge': [], 'pbar': []}
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
            agg[cond]['pi'].append(stationary_occupancy(G, target))
            for gen in args.generators:
                q_peak, q_time, at_edge = ctqw_peak(G, source, target, t_max,
                                                    generator=gen)
                pbar, _ngroups = ctqw_time_average(G, source, target,
                                                   generator=gen)
                qagg[(cond, gen)]['peak'].append(q_peak)
                qagg[(cond, gen)]['time'].append(q_time)
                qagg[(cond, gen)]['edge'].append(at_edge)
                qagg[(cond, gen)]['pbar'].append(pbar)

    print(f"\nPortal walkers -- {args.topology} N={args.nodes}, "
          f"{args.seeds} seed(s), source-target distance "
          f"{np.mean(distances):.1f}±{np.std(distances):.1f}")
    if any(skipped.values()):
        print("NOTE: portal placement failed for "
              + ", ".join(f"{n} {c} seed(s)" for c, n in skipped.items() if n)
              + " -- those conditions have fewer samples than 'none'.")

    base_cls = np.median(agg['none']['t_cls']) if agg['none']['t_cls'] else 1.0
    base_pi = np.median(agg['none']['pi']) if agg['none']['pi'] else 1.0

    print("\n== CLASSICAL ==  (both quantities horizon-free)")
    header = (f"{'condition':>9s} {'n':>3s} {'pi(target)':>11s} "
              f"{'pi gain':>8s} {'median t_hit':>13s} {'t_hit gain':>11s}")
    print(header)
    print('-' * len(header))
    for cond in conditions:
        a = agg[cond]
        if not a['t_cls']:
            continue
        tc = np.median(a['t_cls'])
        pi = np.median(a['pi'])
        print(f"{cond:>9s} {len(a['t_cls']):3d} {pi:11.3e} "
              f"{pi / base_pi:7.2f}x {tc:13.0f} {base_cls / tc:10.1f}x")

    for gen in args.generators:
        qn = qagg[('none', gen)]
        base_pb = np.median(qn['pbar']) if qn['pbar'] else 1.0
        base_qp = np.median(qn['peak']) if qn['peak'] else 1.0
        print(f"\n== QUANTUM, H = {gen} ==")
        header = (f"{'condition':>9s} {'n':>3s} {'Pbar':>11s} "
                  f"{'Pbar*N':>8s} {'Pbar gain':>10s} | {'peak P':>10s} "
                  f"{'peak gain':>10s} {'at t_max':>9s}")
        print(header)
        print('-' * len(header))
        for cond in conditions:
            q = qagg[(cond, gen)]
            if not q['peak']:
                continue
            pb = np.median(q['pbar'])
            qp = np.median(q['peak'])
            print(f"{cond:>9s} {len(q['peak']):3d} {pb:11.3e} "
                  f"{pb * args.nodes:8.2f} {pb / base_pb:9.1f}x | "
                  f"{qp:10.2e} {qp / base_qp:9.1f}x {sum(q['edge']):5d}/"
                  f"{len(q['edge']):<3d}")
        if any(sum(qagg[(c, gen)]['edge']) for c in conditions):
            print("  NOTE: peak(s) landed at the end of the time grid -- those "
                  "peak values are lower bounds. Pbar is unaffected.")

    print("\nPbar is the infinite-time average -- no time window, exact. "
          "'peak P' is kept only as a\ntransient diagnostic and IS "
          "horizon-dependent: it is a max over (0, "
          f"{args.t_max_factor:g}d], so its\ngain shrinks as the window grows. "
          "Compare Pbar against classical pi(target): same\nunits, same "
          "question (long-run occupancy of the target), both horizon-free.")
    if len(args.generators) > 1:
        print("A portal gain holding for BOTH generators is interference, not "
              "a degree artifact of H = A.")


if __name__ == '__main__':
    main()
