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
        G[u][v][weight_key] = current_weight
    
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
        state_counts = [0] * num_states
        for neighbor in G.neighbors(node):
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
    edges_to_remove = []
    edges_to_add = []
    nodes = list(G.nodes())
    
    for u, v in G.edges():
        if np.random.random() < rewire_prob:
            # Pick new endpoint
            new_v = np.random.choice(nodes)
            attempts = 0
            while (new_v == u or G.has_edge(u, new_v)) and attempts < 10:
                new_v = np.random.choice(nodes)
                attempts += 1
            
            if attempts >= 10:
                continue  # Skip if can't find valid rewire
            
            if preserve_connectivity:
                # Check if this rewiring would disconnect
                # (expensive, only do if requested)
                G.remove_edge(u, v)
                would_disconnect = not nx.has_path(G, u, v)
                G.add_edge(u, v)
                if would_disconnect:
                    continue
            
            edges_to_remove.append((u, v))
            edges_to_add.append((u, new_v))
    
    G.remove_edges_from(edges_to_remove)
    G.add_edges_from(edges_to_add)
    
    return G


# Registry of available rules
RULES = {
    'activation': activation_spread,
    'reinforcement': edge_reinforcement,
    'majority': majority_vote,
    'rewire': random_rewire,
}


def get_rule(name: str) -> Callable:
    """Get a rule function by name."""
    if name not in RULES:
        raise ValueError(f"Unknown rule: {name}. Available: {list(RULES.keys())}")
    return RULES[name]

