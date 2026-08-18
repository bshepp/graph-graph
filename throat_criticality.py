"""
Wormhole-throat critical collapse -- stage 1 of the critical-collapse program
(design spec docs/superpowers/specs/2026-08-17-throat-criticality-design.md).

Does censor-only dynamics (`prune`) show a sharp threshold A* in throat
thickness -- evaporation vs a permanent mutually-protected strand core -- or
another crossover? Physics core: under prune-only the outcome is the
bootstrap-peeling fixed point of the initial throat geometry (randomness sets
timing only, modulo the degree floor, which Gate 1 measures). The threshold
therefore lives in the random-ensemble P(core | a), computed here as the CDF
of per-draw critical densities a*_j found by bisection over nested strands
(core existence is monotone in nested strand sets).

  Gate 1 (instrument + mechanism, gates the exit): stochastic prune-only
    dynamics land on the peeling fixed point seed-by-seed; every mismatch
    must be degree-floor-attributable.
  Gate 2 (the race, descriptive): can triadic weaving rescue a throat the
    peeling fixed point condemns, or demolish a core it promises?

Usage:
    python throat_criticality.py --validate [--quick]
    python throat_criticality.py --pilot   [--draws 300]
    python throat_criticality.py --fss     [--draws 2000]
    python throat_criticality.py --anchor --a-values 0.05 0.1 0.2 [--seeds 8]
"""

import argparse
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import networkx as nx
from tqdm import tqdm

from simulation import create_initial_graph
from async_engine import run_sequential_multi

Strand = Tuple[int, int]


def build_arena(n_nodes: int, cap: int, r: int, seed: int,
                max_tries: int = 50) -> Tuple[nx.Graph, List[int], List[int]]:
    """
    Grown base + two disjoint BFS balls whose centers are >= 2r+6 apart.

    Returns (base, ball1, ball2); ball lists start with their center node.
    Deterministic in `seed`. Raises RuntimeError if no sufficiently far
    center pair exists after `max_tries` candidate centers.
    """
    random.seed(seed)
    np.random.seed(seed)
    base = create_initial_graph(n_nodes, topology='grown', k=cap, seed=seed)
    nodes = list(base.nodes())
    rng = np.random.default_rng(seed)
    min_sep = 2 * r + 6

    for _ in range(max_tries):
        c1 = nodes[int(rng.integers(len(nodes)))]
        dist = nx.single_source_shortest_path_length(base, c1)
        far = [w for w, d in dist.items() if d >= min_sep]
        if not far:
            continue
        c2 = far[int(rng.integers(len(far)))]
        d1 = nx.single_source_shortest_path_length(base, c1, cutoff=r)
        d2 = nx.single_source_shortest_path_length(base, c2, cutoff=r)
        ball1 = [c1] + sorted(w for w in d1 if w != c1)
        ball2 = [c2] + sorted(w for w in d2 if w != c2)
        return base, ball1, ball2
    raise RuntimeError(
        f"no center pair at distance >= {min_sep} in {max_tries} tries "
        f"(N={n_nodes} too small for r={r}?)")


def build_arena_geometric(n_nodes: int, cap: int, r: int, seed: int,
                          max_regen: int = 20
                          ) -> Tuple[nx.Graph, List[int], List[int], int]:
    """
    `build_arena` with substrate-failure regeneration.

    The grown generator occasionally lands in a compact, expander-like phase
    at small N (e.g. seed 111 at N=600: diameter 4) -- the banked
    bistability showing up in the generator itself. Such substrates have no
    geometry to host a throat, so the draw's base is REGENERATED
    deterministically (seed + 1000003*k) and the regeneration count is
    returned and reported downstream -- a disclosed refinement of the
    ensemble to geometric substrates.
    """
    for k in range(max_regen):
        try:
            base, b1, b2 = build_arena(n_nodes, cap, r, seed + 1000003 * k)
            return base, b1, b2, k
        except RuntimeError:
            continue
    raise RuntimeError(
        f"substrate persistently non-geometric after {max_regen} "
        f"regenerations (N={n_nodes}, r={r}, seed={seed})")


def strand_pairs(ball1: List[int], ball2: List[int]) -> List[Strand]:
    """All |B1|*|B2| candidate strands; len(...) is the throat capacity."""
    return [(u, v) for u in ball1 for v in ball2]


def throat_with_strands(base: nx.Graph,
                        strands: Sequence[Strand]) -> nx.Graph:
    """Copy of `base` with the given strand edges added (standard weight)."""
    G = base.copy()
    for u, v in strands:
        G.add_edge(u, v, weight=0.5)
    return G


def peel(G: nx.Graph, strands: Sequence[Strand],
         min_overlap: int = 1) -> Tuple[Set[Strand], int]:
    """
    Bootstrap-peeling fixed point: iteratively remove strands with fewer
    than `min_overlap` common neighbors in the current graph (base +
    surviving strands). Returns (core, n_rounds). `G` is not mutated.

    Core existence is MONOTONE in nested strand sets (adding a strand only
    ever adds protection) -- the correctness condition for the bisection in
    `critical_density_draws`.
    """
    H = G.copy()
    alive: Set[Strand] = set(strands)
    rounds = 0
    while True:
        doomed = [s for s in alive
                  if len(set(H[s[0]]) & set(H[s[1]])) < min_overlap]
        if not doomed:
            return alive, rounds
        for u, v in doomed:
            H.remove_edge(u, v)
        alive -= set(doomed)
        rounds += 1


def run_dynamics(G: nx.Graph, strands: Sequence[Strand],
                 rules: Sequence[str], params: Sequence[Optional[Dict]],
                 sweeps: float, seed: int
                 ) -> Tuple[Set[Strand], Dict[Strand, Optional[float]],
                            nx.Graph]:
    """
    Evolve a throat under the step-4 engine, tracking per-strand death.

    Survivors are judged from FINAL-GRAPH edge membership (robust to a rule
    re-creating an edge); `death_time` records the first absence stamp in
    absolute Poisson time (= sweep-equivalents at rate 1), None if never
    absent. `run_sequential_multi` copies G, so the caller's graph is
    untouched.
    """
    death: Dict[Strand, Optional[float]] = {s: None for s in strands}
    by_node: Dict[int, List[Strand]] = {}
    for u, v in strands:
        by_node.setdefault(u, []).append((u, v))
        by_node.setdefault(v, []).append((u, v))

    def on_event(event_id: int, node: int, rule_idx: int, t: float,
                 H: nx.Graph) -> None:
        for (u, v) in by_node.get(node, ()):
            if death[(u, v)] is None and not H.has_edge(u, v):
                death[(u, v)] = t

    final, _, _ = run_sequential_multi(
        G, list(rules), [1.0] * len(rules), max_time=float(sweeps),
        seed=seed, params=list(params), on_event=on_event)
    survivors = {s for s in strands if final.has_edge(*s)}
    return survivors, death, final


def classify_mismatches(final_graph: nx.Graph, peel_core: Set[Strand],
                        survivors: Set[Strand], min_overlap: int = 1,
                        min_degree: int = 2) -> Dict[str, List[Strand]]:
    """
    Gate-1 accounting. Peel-core strands must all survive ('missing' empty).
    Excess survivors are attributed: 'excess_floor' (an endpoint sits at or
    below prune's degree floor in the final graph), 'excess_protected'
    (>= min_overlap common neighbors in the final graph -- legitimately
    protected in the final configuration, downstream of a floor event), or
    'excess_unattributed' (a genuine Gate-1 failure).
    """
    out: Dict[str, List[Strand]] = {'missing': [], 'excess_floor': [],
                                    'excess_protected': [],
                                    'excess_unattributed': []}
    out['missing'] = sorted(peel_core - survivors)
    for s in sorted(survivors - peel_core):
        u, v = s
        if min(final_graph.degree(u), final_graph.degree(v)) <= min_degree:
            out['excess_floor'].append(s)
        elif len(set(final_graph[u]) & set(final_graph[v])) >= min_overlap:
            out['excess_protected'].append(s)
        else:
            out['excess_unattributed'].append(s)
    return out


def critical_density_draws(n_nodes: int, cap: int, r: int, n_draws: int,
                           seed0: int, progress: bool = False) -> List[Dict]:
    """
    Per-draw critical throat thickness by bisection over nested strands.

    Draw j: one arena (seed0+j) + one uniform permutation of all capacity
    pairs; A*_j = smallest prefix length whose peeling core is nonempty
    (valid because core existence is monotone in nested prefixes). P(core|a)
    is the CDF of a*_j = A*_j/capacity -- the spec's pre-committed ensemble
    observable at ~log2(capacity) peels per draw.
    """
    rows: List[Dict] = []
    it = tqdm(range(n_draws), desc=f"draws r={r}") if progress \
        else range(n_draws)
    for j in it:
        seed = seed0 + j
        base, b1, b2, regen = build_arena_geometric(n_nodes, cap, r, seed)
        pairs = strand_pairs(b1, b2)
        cap_j = len(pairs)
        rng = np.random.default_rng(seed)
        perm = [pairs[i] for i in rng.permutation(cap_j)]

        def core_at(A: int):
            return peel(throat_with_strands(base, perm[:A]), perm[:A])[0]

        top = core_at(cap_j)
        if not top:
            rows.append({'draw': j, 'capacity': cap_j, 'A_star': -1,
                         'a_star': float('inf'), 'core_frac': 0.0,
                         'substrate_regens': regen})
            continue
        lo, hi = 0, cap_j          # invariant: core(lo) empty, core(hi) not
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if core_at(mid):
                hi = mid
            else:
                lo = mid
        rows.append({'draw': j, 'capacity': cap_j, 'A_star': hi,
                     'a_star': hi / cap_j,
                     'core_frac': len(core_at(hi)) / hi,
                     'substrate_regens': regen})
    return rows


def transition_stats(a_stars: Sequence[float]) -> Dict[str, float]:
    """10/50/90 percentiles and width of the finite a* distribution."""
    a = np.asarray(list(a_stars), dtype=float)
    finite = a[np.isfinite(a)]
    if len(finite) == 0:
        return {'a10': float('nan'), 'a50': float('nan'),
                'a90': float('nan'), 'width': float('nan'),
                'n_finite': 0, 'n_total': len(a)}
    q10, q50, q90 = np.quantile(finite, [0.1, 0.5, 0.9])
    return {'a10': float(q10), 'a50': float(q50), 'a90': float(q90),
            'width': float(q90 - q10), 'n_finite': int(len(finite)),
            'n_total': int(len(a))}


def anchor_runs(n_nodes: int, cap: int, r: int, a_values: Sequence[float],
                seeds: int, sweeps: float, rider: bool,
                seed0: int = 5000) -> List[Dict]:
    """
    Gate-1 (peeling == prune-only dynamics, floor-attributed) and the
    timing observable; optional Gate-2 triadic rider on the same throats.
    """
    prune_params = [{'prune_prob': 0.05}]
    tp_params = [{'rewire_prob': 0.05}, {'prune_prob': 0.05}]
    rows: List[Dict] = []
    for a in a_values:
        for s in tqdm(range(seeds), desc=f"anchor a={a}"):
            seed = seed0 + s
            base, b1, b2, regen = build_arena_geometric(n_nodes, cap, r, seed)
            pairs = strand_pairs(b1, b2)
            rng = np.random.default_rng(seed)
            perm = [pairs[i] for i in rng.permutation(len(pairs))]
            A = max(1, round(a * len(pairs)))
            strands = perm[:A]
            H = throat_with_strands(base, strands)
            core, _ = peel(H, strands)
            surv, death, final = run_dynamics(
                H, strands, ['prune'], prune_params, sweeps,
                seed=seed + 10000)
            cls = classify_mismatches(final, core, surv)
            evap = None
            if not core and all(t is not None for t in death.values()):
                evap = max(death.values())
            row = {'a': a, 'seed': s, 'capacity': len(pairs), 'A': A,
                   'core': len(core), 'surv': len(surv),
                   'n_missing': len(cls['missing']),
                   'n_floor': len(cls['excess_floor']),
                   'n_protected': len(cls['excess_protected']),
                   'n_unattributed': len(cls['excess_unattributed']),
                   'evap_time': evap, 'substrate_regens': regen}
            if rider:
                r_surv, _, _ = run_dynamics(
                    H, strands, ['triadic', 'prune'], tp_params, sweeps,
                    seed=seed + 50000)
                row['rider_surv'] = len(r_surv)
                row['rider_core_kept'] = len(core & r_surv)
            rows.append(row)
    return rows


def _hand_graph() -> nx.Graph:
    """Known-answer base: u1~u2 near side, v1/v2 far side, pendant triangles
    keep all endpoint degrees >= 3 so the degree floor never binds."""
    G = nx.Graph()
    G.add_edge('u1', 'u2', weight=0.5)
    for x in ('u1', 'u2', 'v1', 'v2'):
        G.add_edge(x, f'{x}a', weight=0.5)
        G.add_edge(x, f'{x}b', weight=0.5)
        G.add_edge(f'{x}a', f'{x}b', weight=0.5)
    return G


def _validate(quick: bool = False) -> bool:
    ok = True
    n_gate = 400 if quick else 600
    seeds_gate = 2 if quick else 4
    n_pilot = 600 if quick else 1000
    draws_pilot = 30 if quick else 60

    print("[1] hand-built known-answer throats (exact equality required)")
    G = _hand_graph()
    pos = [('u1', 'v1'), ('u2', 'v1')]
    core, _ = peel(throat_with_strands(G, pos), pos)
    good = core == set(pos)
    ok &= good
    print(f"  mutually-protecting pair -> core == both: "
          f"{'OK' if good else 'FAIL ' + str(core)}")
    neg = [('u1', 'v1'), ('u2', 'v2')]
    core, rounds = peel(throat_with_strands(G, neg), neg)
    good = core == set() and rounds == 1
    ok &= good
    print(f"  isolated strands -> empty core in 1 round: "
          f"{'OK' if good else 'FAIL ' + str((core, rounds))}")

    print(f"\n[2] Gate 1 small-scale: peeling == prune-only dynamics "
          f"(N={n_gate}, {seeds_gate} seeds, T=200)")
    n_floor = n_prot = 0
    for s in range(seeds_gate):
        base, b1, b2 = build_arena(n_gate, 6, 2, seed=s)
        pairs = strand_pairs(b1, b2)
        rng = np.random.default_rng(s)
        perm = [pairs[i] for i in rng.permutation(len(pairs))]
        for A in (max(2, len(pairs) // 10), max(4, len(pairs) // 3)):
            strands = perm[:A]
            H = throat_with_strands(base, strands)
            core, _ = peel(H, strands)
            surv, _, final = run_dynamics(H, strands, ['prune'],
                                          [{'prune_prob': 0.05}], 200.0,
                                          seed=1000 + s)
            cls = classify_mismatches(final, core, surv)
            bad = bool(cls['missing']) or bool(cls['excess_unattributed'])
            ok &= not bad
            n_floor += len(cls['excess_floor'])
            n_prot += len(cls['excess_protected'])
            if bad:
                print(f"  seed {s} A={A}: FAIL missing={cls['missing']} "
                      f"unattributed={cls['excess_unattributed']}")
    print(f"  all runs: core subset of survivors, every excess attributed "
          f"(floor {n_floor}, downstream-protected {n_prot})")

    print(f"\n[3] mini-pilot machinery (N={n_pilot}, {draws_pilot} draws)")
    rows = critical_density_draws(n_pilot, 6, 2, draws_pilot, seed0=100)
    stats = transition_stats([r['a_star'] for r in rows])
    regens = sum(r['substrate_regens'] for r in rows)
    print(f"  substrate regenerations: {regens} across {len(rows)} draws")
    sane = (stats['n_finite'] > 0
            and 0.0 < stats['a10'] <= stats['a90'] <= 1.0)
    ok &= sane
    print(f"  a* quantiles: a10={stats['a10']:.3f} a50={stats['a50']:.3f} "
          f"a90={stats['a90']:.3f} width={stats['width']:.3f} "
          f"(finite {stats['n_finite']}/{stats['n_total']}) "
          f"{'OK' if sane else 'FAIL'}")
    both = stats['a10'] > 0.02 and stats['a90'] < 0.98
    print(f"  both outcomes reachable inside (0,1): "
          f"{'yes' if both else 'WARN -- check ensemble design'}")

    print(f"\n{'PASS' if ok else 'FAIL'}: throat criticality instrument")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 1: wormhole-throat critical collapse.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validate', action='store_true')
    mode.add_argument('--pilot', action='store_true')
    mode.add_argument('--fss', action='store_true')
    mode.add_argument('--anchor', action='store_true')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--draws', type=int, default=None)
    ap.add_argument('--a-values', type=float, nargs='+', default=None)
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--sweeps', type=float, default=400.0)
    ap.add_argument('--rider', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.validate:
        raise SystemExit(0 if _validate(args.quick) else 1)

    if args.pilot:
        draws = args.draws or 300
        rows = critical_density_draws(2000, 6, 2, draws, seed0=200,
                                      progress=True)
        s = transition_stats([r['a_star'] for r in rows])
        regens = sum(r['substrate_regens'] for r in rows)
        print(f"  substrate regenerations: {regens} across {len(rows)} draws")
        cf = np.mean([r['core_frac'] for r in rows if r['A_star'] > 0])
        print(f"\nPILOT r=2 N=2000 ({draws} draws): "
              f"a10={s['a10']:.4f} a50={s['a50']:.4f} a90={s['a90']:.4f} "
              f"width={s['width']:.4f} finite={s['n_finite']}/{s['n_total']}"
              f" mean core_frac at A*={cf:.3f}")
        return

    if args.fss:
        draws = args.draws or 2000
        print(f"FSS: {draws} draws per geometry; width must shrink with "
              f"capacity for a SHARP verdict (frozen rule).")
        for (r, n) in ((2, 2000), (3, 5000), (4, 10000)):
            rows = critical_density_draws(n, 6, r, draws,
                                          seed0=300 + 17 * r, progress=True)
            s = transition_stats([row['a_star'] for row in rows])
            regens = sum(row['substrate_regens'] for row in rows)
            capm = np.mean([row['capacity'] for row in rows])
            cf = np.mean([row['core_frac'] for row in rows
                          if row['A_star'] > 0])
            print(f"  r={r} N={n}: capacity~{capm:.0f}  "
                  f"a50={s['a50']:.4f}  width={s['width']:.4f}  "
                  f"finite={s['n_finite']}/{s['n_total']}  "
                  f"core_frac@A*={cf:.3f}  regens={regens}")
        return

    if args.anchor:
        if not args.a_values:
            raise SystemExit("--anchor needs --a-values (frozen from the "
                             "pilot quantiles)")
        rows = anchor_runs(2000, 6, 2, args.a_values, args.seeds,
                           args.sweeps, rider=args.rider)
        bad = sum(r['n_missing'] + r['n_unattributed'] for r in rows)
        print(f"\n{'a':>7s} {'seed':>4s} {'A':>4s} {'core':>4s} "
              f"{'surv':>4s} {'floor':>5s} {'prot':>4s} {'unattr':>6s} "
              f"{'evap_t':>7s}" + ("  rider(surv/kept)" if args.rider
                                   else ""))
        for r in rows:
            ev = f"{r['evap_time']:.1f}" if r['evap_time'] is not None \
                else '--'
            line = (f"{r['a']:7.4f} {r['seed']:4d} {r['A']:4d} "
                    f"{r['core']:4d} {r['surv']:4d} {r['n_floor']:5d} "
                    f"{r['n_protected']:4d} {r['n_unattributed']:6d} "
                    f"{ev:>7s}")
            if args.rider:
                line += f"  {r['rider_surv']}/{r['rider_core_kept']}"
            print(line)
        print(f"\nGATE 1: {'PASS' if bad == 0 else 'FAIL'} "
              f"({bad} missing/unattributed across {len(rows)} runs)")
        raise SystemExit(0 if bad == 0 else 1)


if __name__ == '__main__':
    main()
