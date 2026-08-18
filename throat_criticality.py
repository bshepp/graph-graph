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
