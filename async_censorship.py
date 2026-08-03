"""
Shortcut censorship under ASYNCHRONOUS updates -- step 4, the first physics
checkpoint (LORENTZIAN_SPIKE.md sec 6; design spec
docs/superpowers/specs/2026-08-02-async-censorship-checkpoint-design.md).

Re-runs the banked synchronous censorship result (FINDINGS.md;
shortcut_censorship.py) with event-driven Poisson-clock dynamics and asks
whether it survives when time is emergent rather than a global sweep clock.

  Gate 1 (anchor + control): pure `prune` under async must reproduce P1 --
    threshold, advantage-blind, zero collateral -- IN DISTRIBUTION vs the
    synchronous result. Passing certifies the harness and isolates Gate 2.
  Gate 2 (the race, descriptive): `triadic`+`prune` under async. The
    synchronous "woven-in" self-stabilization (P2) may be an artifact of the
    triadic-then-prune lockstep WITHIN a synchronous step; async has no such
    fixed intra-step ordering. Pre-registered both ways.

State-graph only: the step-3-retired causal DAG is not involved.

Usage:
    python async_censorship.py --validate
    python async_censorship.py --validate --quick
    python async_censorship.py --nodes 2000 --seeds 3
"""

import argparse
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx
from tqdm import tqdm

from simulation import create_initial_graph
from shortcuts import inject_shortcuts
from shortcut_censorship import run_condition
from async_engine import run_sequential_multi

Portal = Tuple[int, int, int]


def build_base_and_portals(n_nodes: int, cap: int, n_long: int,
                           n_detour2: int, seed: int
                           ) -> Tuple[nx.Graph, List[Portal], List[Portal]]:
    """
    Grown base + long (advantage >= 6) and detour-2 (advantage == 2) portals.

    Seeds numpy before injection (`inject_shortcuts` draws from the global
    RNG), so the returned base+portals are reproducible and can be shared
    unchanged by both schedules for a seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    base = create_initial_graph(n_nodes, topology='grown', k=cap, seed=seed)
    long_portals = inject_shortcuts(base, n_long, min_distance=6)
    detour2 = inject_shortcuts(base, n_detour2, min_distance=2, max_distance=2)
    return base, long_portals, detour2


def fabric_edges(base: nx.Graph, long_portals: List[Portal],
                 detour2: List[Portal]) -> set:
    """Base edges minus the injected portal edges (the 'fabric' to protect)."""
    portal_set = set(frozenset((u, v)) for u, v, _ in long_portals + detour2)
    return set(frozenset(e) for e in base.edges()) - portal_set


def summarize_portals(removal: Dict[Tuple[int, int], Optional[float]],
                      long_portals: List[Portal], detour2: List[Portal],
                      final_graph: nx.Graph, base_edges: set) -> Dict[str, float]:
    """
    The censorship observables from a per-portal removal-time dict.

    `removal[(u, v)]` is the removal time (float, any monotone unit) or None
    if the portal survived; keys are the raw (u, v) from the portal tuples.
    The SAME reduction is applied to both schedules, so their rows compare
    directly. Removal-time UNITS differ (sync: integer steps; async:
    continuous sweep-equivalents) but both mean ~1/prune_prob, and the rank
    correlation is unit-free.
    """
    def alive(u: int, v: int) -> bool:
        return removal[(u, v)] is None

    long_survival = (float(np.mean([alive(u, v) for u, v, _ in long_portals]))
                     if long_portals else float('nan'))
    detour2_survival = (float(np.mean([alive(u, v) for u, v, _ in detour2]))
                        if detour2 else float('nan'))

    removed = [(float(adv), float(removal[(u, v)]))
               for u, v, adv in long_portals if removal[(u, v)] is not None]
    mean_removal = float(np.mean([t for _, t in removed])) if removed \
        else float('nan')

    if len(removed) >= 8:
        advs = np.array([a for a, _ in removed])
        ts = np.array([t for _, t in removed])
        ra = np.argsort(np.argsort(advs)).astype(float)
        rt = np.argsort(np.argsort(ts)).astype(float)
        adv_corr = float(np.corrcoef(ra, rt)[0, 1]) \
            if advs.std() > 0 and ts.std() > 0 else float('nan')
    else:
        adv_corr = float('nan')

    woven = sum(1 for u, v, _ in long_portals
                if removal[(u, v)] is None
                and len(set(final_graph[u]) & set(final_graph[v])) >= 1)

    remaining = set(frozenset(e) for e in final_graph.edges())
    collateral = 1.0 - len(base_edges & remaining) / max(len(base_edges), 1)

    return {'long_survival': long_survival,
            'detour2_survival': detour2_survival,
            'mean_removal': mean_removal, 'adv_corr': adv_corr,
            'woven': float(woven), 'collateral': collateral}


def async_condition(base: nx.Graph, long_portals: List[Portal],
                    detour2: List[Portal], rules: Sequence[str],
                    rates: Sequence[float],
                    params: Sequence[Optional[Dict]], sweeps: float,
                    seed: int) -> Tuple[Dict, nx.Graph, Dict[str, float]]:
    """
    Run one async condition on a fresh copy of `base`, tracking per-portal
    removal time in absolute Poisson time (= sweep-equivalents, clocks at
    rate 1). A portal is stamped removed the first time an event at one of its
    endpoints leaves the edge absent. `run_sequential_multi` copies `base`, so
    the caller's graph is untouched and reusable across conditions.
    """
    portals = long_portals + detour2
    removal: Dict[Tuple[int, int], Optional[float]] = {
        (u, v): None for u, v, _ in portals}
    by_node: Dict[int, List[Tuple[int, int]]] = {}
    for u, v, _ in portals:
        by_node.setdefault(u, []).append((u, v))
        by_node.setdefault(v, []).append((u, v))

    def on_event(event_id: int, node: int, rule_idx: int, t: float,
                 G: nx.Graph) -> None:
        for (u, v) in by_node.get(node, ()):
            if removal[(u, v)] is None and not G.has_edge(u, v):
                removal[(u, v)] = t

    final, times, rule_ids = run_sequential_multi(
        base, list(rules), list(rates), max_time=float(sweeps), seed=seed,
        params=list(params), on_event=on_event)

    n = base.number_of_nodes()
    events_per_node = {rules[i]: float(np.sum(rule_ids == i)) / max(n, 1)
                       for i in range(len(rules))}
    return removal, final, events_per_node


def sync_condition(base: nx.Graph, long_portals: List[Portal],
                   detour2: List[Portal], condition: str, steps: int,
                   prune_prob: float) -> Tuple[Dict, nx.Graph]:
    """
    The banked synchronous reference, via `shortcut_censorship.run_condition`
    verbatim. Returns (removal_step, final_graph); `run_condition` copies
    `base` internally. NOTE it hardcodes triadic `rewire_prob=0.05`.
    """
    res = run_condition(base, long_portals + detour2, condition, steps,
                        prune_prob)
    return res['removal_step'], res['final_graph']


def _z(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sample z on seed-means (NaNs dropped); the async_engine Test-B bar."""
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return 0.0 if se == 0 else abs(a.mean() - b.mean()) / se


# rewire_prob is FIXED at 0.05 to match shortcut_censorship.run_condition's
# hardcoded triadic rate -- any other value silently breaks sync/async parity.
_REWIRE_PROB = 0.05


def _validate(n_nodes: int = 1200, cap: int = 6, n_long: int = 40,
              n_detour2: int = 20, sweeps: int = 120, seeds: int = 5,
              prune_prob: float = 0.05) -> bool:
    ok = True
    prune_params = [{'prune_prob': prune_prob}]
    tp_rules = ['triadic', 'prune']
    tp_params = [{'rewire_prob': _REWIRE_PROB}, {'prune_prob': prune_prob}]

    # ---- Gate 1: async prune vs sync prune (P1) ----
    keys = ['long_survival', 'detour2_survival', 'mean_removal', 'collateral']
    sync_g1 = {k: [] for k in keys}
    async_g1 = {k: [] for k in keys}
    async_corr: List[float] = []
    prune_pn: List[float] = []

    print(f"Gate 1 -- async prune vs sync prune (P1): N={n_nodes}, "
          f"{n_long} long + {n_detour2} detour-2, {sweeps} sweeps, "
          f"{seeds} seeds")
    for s in tqdm(range(seeds), desc='gate1'):
        base, longp, d2 = build_base_and_portals(n_nodes, cap, n_long,
                                                 n_detour2, s)
        fabric = fabric_edges(base, longp, d2)
        sr, sg = sync_condition(base, longp, d2, 'prune', sweeps, prune_prob)
        ar, ag, ac = async_condition(base, longp, d2, ['prune'], [1.0],
                                     prune_params, float(sweeps), seed=1000 + s)
        ssum = summarize_portals(sr, longp, d2, sg, fabric)
        asum = summarize_portals(ar, longp, d2, ag, fabric)
        for k in keys:
            sync_g1[k].append(ssum[k])
            async_g1[k].append(asum[k])
        async_corr.append(asum['adv_corr'])
        prune_pn.append(ac['prune'])

    print(f"  {'observable':>17s} {'sync':>8s} {'async':>8s} {'z':>6s}")
    for k in keys:
        z = _z(sync_g1[k], async_g1[k])
        passed = np.isnan(z) or z < 3.0
        ok &= passed
        print(f"  {k:>17s} {np.nanmean(sync_g1[k]):8.3f} "
              f"{np.nanmean(async_g1[k]):8.3f} {z:6.2f}  "
              f"{'' if passed else 'FAIL'}")
    corr = float(np.nanmean(async_corr))
    blind = abs(corr) < 0.2
    ok &= blind
    print(f"  async rank(adv, t) = {corr:+.3f}  "
          f"{'advantage-blind OK' if blind else 'CORRELATED -- FAIL'}")
    print(f"  async prune events/node = {np.mean(prune_pn):.1f} "
          f"(target ~{sweeps})")

    # ---- Gate 2: triadic+prune race (P2), descriptive ----
    print("\nGate 2 -- triadic+prune race (P2): DESCRIPTIVE, "
          "pre-registered both ways")
    g2k = ['long_survival', 'detour2_survival', 'woven', 'collateral']
    g2 = {'sync': {k: [] for k in g2k}, 'async': {k: [] for k in g2k}}
    tri_pn: List[float] = []
    prune_pn2: List[float] = []
    for s in tqdm(range(seeds), desc='gate2'):
        base, longp, d2 = build_base_and_portals(n_nodes, cap, n_long,
                                                 n_detour2, s)
        fabric = fabric_edges(base, longp, d2)
        sr, sg = sync_condition(base, longp, d2, 'triadic+prune', sweeps,
                                prune_prob)
        ar, ag, ac = async_condition(base, longp, d2, tp_rules, [1.0, 1.0],
                                     tp_params, float(sweeps), seed=2000 + s)
        ssum = summarize_portals(sr, longp, d2, sg, fabric)
        asum = summarize_portals(ar, longp, d2, ag, fabric)
        for k in g2k:
            g2['sync'][k].append(ssum[k])
            g2['async'][k].append(asum[k])
        tri_pn.append(ac['triadic'])
        prune_pn2.append(ac['prune'])

    print(f"  {'observable':>17s} {'sync':>8s} {'async':>8s}")
    for k in g2k:
        print(f"  {k:>17s} {np.nanmean(g2['sync'][k]):8.3f} "
              f"{np.nanmean(g2['async'][k]):8.3f}")

    tri, prn = float(np.mean(tri_pn)), float(np.mean(prune_pn2))
    matched = abs(tri - prn) / max(prn, 1e-9) < 0.1
    ok &= matched
    print(f"  async events/node: triadic {tri:.1f}, prune {prn:.1f}  "
          f"{'matched OK' if matched else 'MISMATCH -- confound, FAIL'}")

    a_long = float(np.nanmean(g2['async']['long_survival']))
    a_woven = float(np.nanmean(g2['async']['woven']))
    base_long = float(np.nanmean(async_g1['long_survival']))
    if a_woven > 0.0 and a_long > base_long + 0.02:
        verdict = ("P2 SURVIVES async -- weaving persists (woven "
                   f"{a_woven:.2f}, long survival {a_long:.2f} vs prune-only "
                   f"{base_long:.2f}); self-stabilization is genuine dynamics")
    else:
        verdict = ("P2 COLLAPSES under async -- woven ~0 / long survival "
                   f"{a_long:.2f} ~ prune-only {base_long:.2f}; the synchronous "
                   "self-stabilization was a triadic-then-prune lockstep "
                   "artifact")
    print(f"  VERDICT: {verdict}")

    print(f"\n{'PASS' if ok else 'FAIL'}: async censorship checkpoint "
          f"(Gate 1 instrument + confounds gate the exit; Gate 2 descriptive)")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Shortcut censorship under asynchronous updates.")
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--nodes', type=int, default=2000)
    ap.add_argument('--cap', type=int, default=6)
    ap.add_argument('--long', type=int, default=40)
    ap.add_argument('--detour2', type=int, default=20)
    ap.add_argument('--sweeps', type=int, default=120)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--prune-prob', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.validate:
        kw = dict(n_nodes=1200, cap=6, n_long=40, n_detour2=20, sweeps=120,
                  seeds=5, prune_prob=args.prune_prob)
        if args.quick:
            kw.update(n_nodes=400, n_long=10, n_detour2=5, sweeps=40, seeds=2)
        raise SystemExit(0 if _validate(**kw) else 1)

    # A production-scale run IS the checkpoint at the requested scale.
    raise SystemExit(0 if _validate(
        n_nodes=args.nodes, cap=args.cap, n_long=args.long,
        n_detour2=args.detour2, sweeps=args.sweeps, seeds=args.seeds,
        prune_prob=args.prune_prob) else 1)


if __name__ == '__main__':
    main()
