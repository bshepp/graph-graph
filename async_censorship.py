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
