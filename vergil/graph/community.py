# graph/community.py
import networkx as nx
from cdlib import algorithms


def detect_communities(G: nx.Graph, resolution: float = 1.0) -> dict:
    """
    Run Leiden community detection on the product graph.

    Leiden is used (not Louvain) because:
    1. It guarantees connected communities (Louvain can produce disconnected ones)
    2. It's faster on large graphs
    3. It's what Microsoft GraphRAG uses

    Returns:
        {
            "level_0": [community_0, community_1, ...],  # finest granularity
            "level_1": [...],  # coarser (communities of communities)
        }
        where each community is a set of node IDs
    """
    # Install: pip install cdlib leidenalg

    # Level 0: fine-grained communities (weight-aware + reproducible)
    communities_l0 = algorithms.leiden(G, weights="weight", seed=42,
                                       resolution_parameter=resolution)

    # Anti-blob guard: if one community swallows >40% of the graph, the resolution
    # is too coarse — re-run L0 once at 2x resolution (same weights/seed).
    n_nodes = G.number_of_nodes()
    if communities_l0.communities and n_nodes:
        largest = max(len(c) for c in communities_l0.communities)
        if largest > 0.40 * n_nodes:
            communities_l0 = algorithms.leiden(G, weights="weight", seed=42,
                                               resolution_parameter=resolution * 2.0)

    # Level 1: coarser communities (run Leiden on the community graph)
    # Build a community graph where nodes = L0 communities, edges = inter-community connections
    community_graph = _build_community_graph(G, communities_l0)
    if community_graph.number_of_nodes() > 1:
        communities_l1 = algorithms.leiden(community_graph, weights="weight", seed=42,
                                           resolution_parameter=resolution * 0.5)
        level_1 = communities_l1.communities
    else:
        # Only one L0 community — nothing to coarsen into.
        level_1 = [list(community_graph.nodes())] if community_graph.number_of_nodes() else []

    return {
        "level_0": communities_l0.communities,  # list of lists of node IDs
        "level_1": level_1,
    }


def _build_community_graph(G, communities):
    """Contract the original graph to a community-level graph."""
    # Each L0 community becomes a node
    # Edge weight = number of inter-community edges in the original graph
    CG = nx.Graph()
    node_to_community = {}
    for i, comm in enumerate(communities.communities):
        CG.add_node(i, size=len(comm))
        for node in comm:
            node_to_community[node] = i

    for u, v in G.edges():
        cu, cv = node_to_community.get(u), node_to_community.get(v)
        if cu is not None and cv is not None and cu != cv:
            if CG.has_edge(cu, cv):
                CG[cu][cv]["weight"] += 1
            else:
                CG.add_edge(cu, cv, weight=1)
    return CG
