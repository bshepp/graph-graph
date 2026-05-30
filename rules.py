"""
Simple Graph Update Rules

Each rule takes a graph and returns a modified graph.
Rules should be LOCAL - each node only sees its neighbors.
"""

import numpy as np
import networkx as nx
from typing import Callable


def activation_spread(G: nx.Graph, activation_key: str = 'active', 
                      spread_prob: float = 0.3, decay_prob: float = 0.1) -> nx.Graph:
    """
    Active nodes spread activation to neighbors.
    Active nodes may decay to inactive.
    
    This is basically a simple epidemic/SIS model.
    """
    new_states = {}
    
    for node in G.nodes():
        is_active = G.nodes[node].get(activation_key, False)
        
        if is_active:
            # Active nodes may decay
            new_states[node] = np.random.random() > decay_prob
        else:
            # Inactive nodes may become active from neighbors
            active_neighbors = sum(
                1 for n in G.neighbors(node) 
                if G.nodes[n].get(activation_key, False)
            )
            if active_neighbors > 0:
                # More active neighbors = higher chance of activation
                prob = 1 - (1 - spread_prob) ** active_neighbors
                new_states[node] = np.random.random() < prob
            else:
                new_states[node] = False
    
    # Apply new states
    for node, state in new_states.items():
        G.nodes[node][activation_key] = state
    
    return G


def edge_reinforcement(G: nx.Graph, weight_key: str = 'weight',
                       reinforce_amount: float = 0.1,
                       decay_amount: float = 0.01,
                       min_weight: float = 0.01,
                       max_weight: float = 1.0) -> nx.Graph:
    """
    Edges between co-active nodes strengthen.
    All edges slowly decay.
    
    Hebbian-style learning: "neurons that fire together wire together"
    """
    new_weights = {}
    
    for u, v in G.edges():
        current_weight = G[u][v].get(weight_key, 0.5)
        
        # Decay
        current_weight -= decay_amount
        
        # Reinforce if both active
        u_active = G.nodes[u].get('active', False)
        v_active = G.nodes[v].get('active', False)
        
        if u_active and v_active:
            current_weight += reinforce_amount
        
        # Clamp
        current_weight = max(min_weight, min(max_weight, current_weight))
        new_weights[(u, v)] = current_weight
    
    # Apply batched updates
    for (u, v), w in new_weights.items():
        G[u][v][weight_key] = w
    
    return G


def majority_vote(G: nx.Graph, state_key: str = 'state', 
                  num_states: int = 2, noise: float = 0.01) -> nx.Graph:
    """
    Each node adopts the majority state of its neighbors.
    Small noise prevents frozen states.
    
    This can produce domain formation and phase transitions.
    """
    new_states = {}
    
    for node in G.nodes():
        # Count neighbor states
        neighbors = list(G.neighbors(node))
        
        if not neighbors:
            # Isolated nodes keep their current state
            new_states[node] = G.nodes[node].get(state_key, 0)
            continue
        
        state_counts = [0] * num_states
        for neighbor in neighbors:
            s = G.nodes[neighbor].get(state_key, 0)
            state_counts[s] += 1
        
        # Majority vote with noise
        if np.random.random() < noise:
            new_states[node] = np.random.randint(num_states)
        else:
            new_states[node] = np.argmax(state_counts)
    
    # Apply
    for node, state in new_states.items():
        G.nodes[node][state_key] = state
    
    return G


def random_rewire(G: nx.Graph, rewire_prob: float = 0.01,
                  preserve_connectivity: bool = False) -> nx.Graph:
    """
    Small probability of rewiring edges.
    Creates small-world structure over time.
    
    Args:
        preserve_connectivity: If True, reject rewirings that would disconnect 
                               the graph. More expensive but maintains single component.
                               Default False lets fragmentation happen naturally.
    """
    nodes = list(G.nodes())
    
    # Snapshot edge list so we don't iterate over a mutating collection
    edges = list(G.edges())
    
    if preserve_connectivity:
        # Apply rewirings one at a time so connectivity checks account
        # for all prior changes within this step.
        for u, v in edges:
            if np.random.random() >= rewire_prob:
                continue
            
            new_v = np.random.choice(nodes)
            attempts = 0
            while (new_v == u or G.has_edge(u, new_v)) and attempts < 10:
                new_v = np.random.choice(nodes)
                attempts += 1
            
            if attempts >= 10:
                continue
            
            # Check connectivity *after* tentative removal
            old_weight = G[u][v].get('weight', 0.5)
            G.remove_edge(u, v)
            if not nx.has_path(G, u, v):
                # Would disconnect -- roll back
                G.add_edge(u, v, weight=old_weight)
                continue
            
            # Safe to rewire; carry over the edge weight
            G.add_edge(u, new_v, weight=old_weight)
    else:
        # Batch mode (faster, no connectivity guarantee)
        edges_to_remove = []
        edges_to_add = []
        
        for u, v in edges:
            if np.random.random() >= rewire_prob:
                continue
            
            new_v = np.random.choice(nodes)
            attempts = 0
            while (new_v == u or G.has_edge(u, new_v)) and attempts < 10:
                new_v = np.random.choice(nodes)
                attempts += 1
            
            if attempts >= 10:
                continue
            
            old_weight = G[u][v].get('weight', 0.5)
            edges_to_remove.append((u, v))
            edges_to_add.append((u, new_v, {'weight': old_weight}))
        
        G.remove_edges_from(edges_to_remove)
        G.add_edges_from(edges_to_add)
    
    return G


def shortcut_prune(G: nx.Graph, prune_prob: float = 0.05,
                   min_overlap: int = 1, min_degree: int = 2) -> nx.Graph:
    """
    Remove "shortcut" edges -- those whose endpoints share fewer than
    `min_overlap` common neighbors (i.e. the edge is not embedded in any
    local triangle).

    Local information only: each edge is judged from its two endpoints'
    neighbor sets. This is the inverse of random rewiring -- instead of
    adding long-range shortcuts it strips them.

    On a Watts-Strogatz small-world graph (a ring lattice + random
    shortcuts) the shortcuts connect ring-distant nodes that share no
    neighbors, while the local ring edges sit in triangles. Pruning the
    zero-overlap edges should therefore peel the graph back toward its
    underlying ~1D ring. The rule self-terminates: once every surviving
    edge sits in a local cluster, nothing is removable.

    Connectivity is protected loosely by a minimum-degree floor: an edge is
    not removed if either endpoint would drop to/below `min_degree`. The
    degree bookkeeping is updated as we go so the floor accounts for earlier
    removals in the same step.
    """
    deg = dict(G.degree())
    to_remove = []

    for u, v in list(G.edges()):
        if np.random.random() >= prune_prob:
            continue
        if deg[u] <= min_degree or deg[v] <= min_degree:
            continue
        # Common neighbors (excludes u, v themselves -- no self loops).
        overlap = len(set(G[u]) & set(G[v]))
        if overlap < min_overlap:
            to_remove.append((u, v))
            deg[u] -= 1
            deg[v] -= 1

    G.remove_edges_from(to_remove)
    return G


def triadic_closure(G: nx.Graph, rewire_prob: float = 0.05,
                    min_degree: int = 2) -> nx.Graph:
    """
    Rewire edges toward friends-of-friends (triadic closure).

    For each edge (u, v) selected with probability `rewire_prob`, detach the
    v-end and reattach u to a 2-hop neighbor w (a neighbor of one of u's
    neighbors) that u is not already adjacent to. This closes a triangle
    u-x-w and biases connectivity toward local, clustered structure -- the
    exact opposite of random rewiring's long-range shortcuts. Edge count is
    conserved and the rewired edge keeps its weight.

    Local information only (u's neighbors and their neighbors). Open
    question this tests: can repeated local triadic closure *build* extended
    low-dimensional structure from a dimensionless start, or does it just
    collapse into dense clumps?
    """
    deg = dict(G.degree())

    for u, v in list(G.edges()):
        if np.random.random() >= rewire_prob:
            continue
        if deg[v] <= min_degree:
            continue

        u_neighbors = list(G[u])
        if not u_neighbors:
            continue
        x = u_neighbors[np.random.randint(len(u_neighbors))]
        x_neighbors = list(G[x])
        if not x_neighbors:
            continue
        w = x_neighbors[np.random.randint(len(x_neighbors))]

        if w == u or G.has_edge(u, w):
            continue

        # Rewire (u, v) -> (u, w): u's degree is unchanged, v loses one, w gains one.
        wt = G[u][v].get('weight', 0.5)
        G.remove_edge(u, v)
        G.add_edge(u, w, weight=wt)
        deg[v] -= 1
        deg[w] = deg.get(w, 0) + 1

    return G


def geometrize(G: nx.Graph, target_degree: int = 6, rate: float = 0.2,
               min_degree: int = 3) -> nx.Graph:
    """
    Homeostatic local rewiring toward a target degree, keeping edges local.

    For each node (with probability `rate`):
      - over-connected (degree > target_degree): shed its most shortcut-like
        edge -- the neighbor with the fewest shared neighbors;
      - under-connected (degree < target_degree): gain a triadic
        (friend-of-friend) edge.

    Bounded degree + locality are the ingredients that produce dimension
    (cf. the 'grown' generator). This rule was built to test whether they
    yield an ATTRACTOR -- a dimension reached from any initial topology.

    They do NOT (see FINDINGS.md). The dynamics are bistable:
      - a 2D lattice is a stable fixed point (d stays ~2, stays connected);
      - random / scale-free graphs are stuck at d "undefined" and cannot be
        pulled into geometry -- a bootstrapping barrier, because
        friend-of-friend is meaningless without pre-existing locality.
    So this rule PRESERVES geometry within its basin but does not nucleate
    it from disorder, and target_degree does not cleanly tune the result.

    Local information only (a node's neighbors and their neighbors).

    NetworkX backend only (not vectorized in simulation_fast.py); run the
    driver with --no-fast to use it.
    """
    deg = dict(G.degree())

    for u in list(G.nodes()):
        if np.random.random() >= rate:
            continue
        du = deg[u]

        if du > target_degree and du > min_degree:
            # Shed the most shortcut-like edge (neighbor with least overlap).
            uset = set(G[u])
            cand = [(len(uset & set(G[v])), v) for v in G[u]
                    if deg[v] > min_degree]
            if cand:
                _, v = min(cand, key=lambda t: t[0])
                G.remove_edge(u, v)
                deg[u] -= 1
                deg[v] -= 1

        elif du < target_degree:
            # Gain a triadic (friend-of-friend) edge.
            nbrs = list(G[u])
            if not nbrs:
                continue
            x = nbrs[np.random.randint(len(nbrs))]
            xn = list(G[x])
            if not xn:
                continue
            w = xn[np.random.randint(len(xn))]
            if w != u and not G.has_edge(u, w):
                G.add_edge(u, w, weight=0.5)
                deg[u] += 1
                deg[w] += 1

    return G


# Registry of available rules
RULES = {
    'activation': activation_spread,
    'reinforcement': edge_reinforcement,
    'majority': majority_vote,
    'rewire': random_rewire,
    'prune': shortcut_prune,
    'triadic': triadic_closure,
    'geometrize': geometrize,
}


def get_rule(name: str) -> Callable:
    """Get a rule function by name."""
    if name not in RULES:
        raise ValueError(f"Unknown rule: {name}. Available: {list(RULES.keys())}")
    return RULES[name]

