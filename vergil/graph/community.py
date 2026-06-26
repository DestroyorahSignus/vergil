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

    # Level 0: fine-grained communities
    communities_l0 = algorithms.leiden(G, resolution_parameter=resolution)

    # Level 1: coarser communities (run Leiden on the community graph)
    # Build a community graph where nodes = L0 communities, edges = inter-community connections
    community_graph = _build_community_graph(G, communities_l0)
    communities_l1 = algorithms.leiden(community_graph, resolution_parameter=resolution * 0.5)

    return {
        "level_0": communities_l0.communities,  # list of lists of node IDs
        "level_1": communities_l1.communities,
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
