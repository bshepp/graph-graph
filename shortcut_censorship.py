"""
Does geometry-maintaining dynamics censor shortcuts by their transport
advantage -- and can a shortcut be stabilized against the censorship?

FINDINGS.md established qualitatively that `prune` recovers latent geometry
by deleting shortcut edges. This experiment makes the censorship
quantitative, and records two a-priori predictions to kill-test:

  P1 (threshold, not graded): `prune` removes edges whose endpoints share no
     neighbor. Any shortcut between nodes at base distance >= 3 has zero
     common neighbors BY DEFINITION (a common neighbor would make the
     distance <= 2). So censorship should be a step function of advantage:
     detour-2 shortcuts are immune, everything longer is uniformly eligible
     -- and among the eligible, removal TIME should be geometric with rate
     prune_prob, UNCORRELATED with advantage.

  P2 (self-stabilization): running `triadic` closure alongside `prune`
     gives shortcuts a defense: triadic rewires toward friends-of-friends,
     and a portal makes the far side's nodes into friends-of-friends -- so
     triadic can lay a parallel edge next to the portal, forming a triangle
     ON it, after which prune cannot touch it. Prediction: with triadic
     active, a nonzero fraction of portals survives indefinitely ("woven
     in"), and survivors show triangles on the portal edge.

Conditions: prune-only, ricci-only (same zero-overlap criterion, rewires
instead of deleting), triadic+prune (the race).

Usage:
    python shortcut_censorship.py --quick
    python shortcut_censorship.py --nodes 2000 --shortcuts 40 --steps 120 --seeds 3
"""

import argparse
import random
from typing import Dict, List

import numpy as np
import networkx as nx
from tqdm import tqdm

from simulation import create_initial_graph
from rules import shortcut_prune, ricci_rewire, triadic_closure
from shortcuts import inject_shortcuts


def run_condition(base: nx.Graph, portals: List, condition: str,
                  steps: int, prune_prob: float) -> Dict:
    """
    Run one dynamics condition on a copy of `base` and track each portal.

    Returns per-portal removal step (None = survived all steps), plus
    final triangle counts on survivors and collateral damage to the base
    graph's own edges.
    """
    G = base.copy()
    base_edges = set(frozenset(e) for e in base.edges()) - \
        set(frozenset((u, v)) for u, v, _ in portals)
    removal_step: Dict = {(u, v): None for u, v, _ in portals}

    for step in range(1, steps + 1):
        if condition in ('triadic+prune',):
            G = triadic_closure(G, rewire_prob=0.05)
        if condition in ('prune', 'triadic+prune'):
            G = shortcut_prune(G, prune_prob=prune_prob)
        elif condition == 'ricci':
            G = ricci_rewire(G, rewire_prob=prune_prob)

        for (u, v, _d) in portals:
            if removal_step[(u, v)] is None and not G.has_edge(u, v):
                removal_step[(u, v)] = step

    remaining = set(frozenset(e) for e in G.edges())
    collateral = 1.0 - len(base_edges & remaining) / max(len(base_edges), 1)

    return {
        'removal_step': removal_step,
        'collateral': collateral,
        'final_graph': G,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Censorship of shortcuts by geometry-maintaining rules.")
    parser.add_argument('--nodes', type=int, default=2000)
    parser.add_argument('--topology', type=str, default='grown')
    parser.add_argument('--cap', type=int, default=6)
    parser.add_argument('--shortcuts', type=int, default=40,
                        help='Long portals injected (base distance >= 6).')
    parser.add_argument('--short-shortcuts', type=int, default=20,
                        help='Detour-2 controls (base distance exactly 2).')
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--prune-prob', type=float, default=0.05)
    parser.add_argument('--conditions', type=str, nargs='+',
                        default=['prune', 'ricci', 'triadic+prune'])
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if args.quick:
        args.nodes, args.shortcuts, args.short_shortcuts = 500, 10, 5
        args.steps, args.seeds = 40, 1

    agg: Dict[str, Dict[str, List]] = {
        c: {'long_survival': [], 'short_survival': [], 'mean_removal': [],
            'adv_corr': [], 'woven': [], 'collateral': []}
        for c in args.conditions}

    for s in tqdm(range(args.seeds), desc='seeds'):
        seed = args.seed + s
        random.seed(seed)
        np.random.seed(seed)
        base = create_initial_graph(args.nodes, topology=args.topology,
                                    k=args.cap, seed=seed)
        long_portals = inject_shortcuts(base, args.shortcuts, min_distance=6)
        short_portals = inject_shortcuts(base, args.short_shortcuts,
                                         min_distance=2, max_distance=2)
        portals = long_portals + short_portals

        for cond in args.conditions:
            random.seed(seed * 7 + 1)
            np.random.seed(seed * 7 + 1)
            res = run_condition(base, portals, cond,
                                args.steps, args.prune_prob)
            rs = res['removal_step']

            long_alive = sum(1 for (u, v, _d) in long_portals
                             if rs[(u, v)] is None)
            short_alive = sum(1 for (u, v, _d) in short_portals
                              if rs[(u, v)] is None)
            agg[cond]['long_survival'].append(long_alive / len(long_portals))
            agg[cond]['short_survival'].append(
                short_alive / max(len(short_portals), 1))
            agg[cond]['collateral'].append(res['collateral'])

            removed = [(d, rs[(u, v)]) for (u, v, d) in long_portals
                       if rs[(u, v)] is not None]
            if removed:
                agg[cond]['mean_removal'].append(
                    float(np.mean([t for _, t in removed])))
                if len(removed) >= 8:
                    advs = np.array([d for d, _ in removed], dtype=float)
                    times = np.array([t for _, t in removed], dtype=float)
                    # Spearman rank correlation without scipy.stats ceremony
                    ra = np.argsort(np.argsort(advs)).astype(float)
                    rt = np.argsort(np.argsort(times)).astype(float)
                    if ra.std() > 0 and rt.std() > 0:
                        agg[cond]['adv_corr'].append(
                            float(np.corrcoef(ra, rt)[0, 1]))
            # "Woven in" = LONG portal that survived AND acquired a triangle
            # (a common neighbor), i.e. prune can no longer touch it.
            Gf = res['final_graph']
            woven = sum(1 for (u, v, _d) in long_portals
                        if rs[(u, v)] is None
                        and len(set(Gf[u]) & set(Gf[v])) >= 1)
            agg[cond]['woven'].append(woven)

    exp_lifetime = 1.0 / args.prune_prob
    print(f"\nShortcut censorship -- {args.topology} N={args.nodes}, "
          f"{args.shortcuts} long (d>=6) + {args.short_shortcuts} detour-2 "
          f"portals, {args.steps} steps, prob={args.prune_prob} "
          f"(geometric expectation: mean removal ~{exp_lifetime:.0f} steps)")
    header = (f"{'condition':>14s} {'long surv':>10s} {'d2 surv':>8s} "
              f"{'<t_remove>':>10s} {'rank(adv,t)':>11s} {'woven':>6s} "
              f"{'collateral':>10s}")
    print(header)
    print('-' * len(header))
    for cond in args.conditions:
        a = agg[cond]
        ls = np.mean(a['long_survival'])
        ss = np.mean(a['short_survival'])
        mr = np.mean(a['mean_removal']) if a['mean_removal'] else float('nan')
        cr = np.mean(a['adv_corr']) if a['adv_corr'] else float('nan')
        wv = np.mean(a['woven'])
        co = np.mean(a['collateral'])
        print(f"{cond:>14s} {ls:10.2f} {ss:8.2f} {mr:10.1f} {cr:11.2f} "
              f"{wv:6.1f} {co:10.3f}")

    print("\nP1 (threshold censorship) is supported if: d2 survival >> long "
          "survival, and rank(adv, t_remove) ~ 0 among removed portals.")
    print("P2 (self-stabilization) is supported if: triadic+prune long "
          "survival > prune-only, with woven > 0 (triangles on survivors).")


if __name__ == '__main__':
    main()
