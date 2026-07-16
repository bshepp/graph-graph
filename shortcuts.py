"""
Shared helpers for the shortcut ("portal") experiments.

A *shortcut* here is an injected edge between nodes that are far apart in
the base graph -- the graph-theoretic skeleton of a wormhole. These helpers
inject shortcuts with a recorded transport advantage and measure distances,
so the three experiment CLIs (`shortcut_tolerance.py`,
`shortcut_censorship.py`, `shortcut_walkers.py`) share one implementation.

No node positions are used anywhere -- distances are BFS hop counts, so the
no-baked-in-geometry invariant holds.
"""

from typing import List, Optional, Tuple

import numpy as np
import networkx as nx


def inject_shortcuts(G: nx.Graph, count: int,
                     min_distance: int = 8,
                     max_distance: Optional[int] = None,
                     max_tries: int = 200) -> List[Tuple[int, int, int]]:
    """
    Add `count` random long-range edges to G in place.

    Each shortcut connects a uniformly random node u to a uniformly random
    node v whose CURRENT graph distance from u lies in
    [min_distance, max_distance]. The distance is measured in the graph as
    it stands when the shortcut is injected (so later shortcuts see earlier
    ones), and is recorded as that shortcut's *transport advantage*: the
    detour a signal would need if the edge vanished.

    Returns a list of (u, v, distance_at_injection). Edges are tagged with
    ``shortcut=True`` and get the standard weight 0.5.
    """
    nodes = list(G.nodes())
    placed: List[Tuple[int, int, int]] = []

    for _ in range(count):
        for _attempt in range(max_tries):
            u = nodes[np.random.randint(len(nodes))]
            dist = nx.single_source_shortest_path_length(G, u)
            candidates = [w for w, d in dist.items()
                          if d >= min_distance
                          and (max_distance is None or d <= max_distance)
                          and not G.has_edge(u, w)]
            if candidates:
                v = candidates[np.random.randint(len(candidates))]
                G.add_edge(u, v, weight=0.5, shortcut=True)
                placed.append((u, v, dist[v]))
                break
        else:
            break  # graph has no eligible far pairs left

    return placed


def mean_pair_distance(G: nx.Graph, n_pairs: int = 200) -> float:
    """
    Mean BFS distance over random connected node pairs -- a cheap proxy for
    characteristic path length, to watch the small-world collapse.
    """
    nodes = list(G.nodes())
    total, got = 0, 0
    while got < n_pairs:
        u = nodes[np.random.randint(len(nodes))]
        v = nodes[np.random.randint(len(nodes))]
        if u == v:
            continue
        try:
            total += nx.shortest_path_length(G, u, v)
            got += 1
        except nx.NetworkXNoPath:
            continue
    return total / got


def far_node(G: nx.Graph, source: int) -> Tuple[int, int]:
    """Return (node, distance) for a node at maximal BFS distance from source."""
    dist = nx.single_source_shortest_path_length(G, source)
    target = max(dist, key=dist.get)
    return target, dist[target]
