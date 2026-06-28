# graph/community.py
import igraph as ig
import leidenalg
import networkx as nx


def _nx_to_igraph(G: nx.Graph):
    """Convert a weighted networkx graph to igraph; return (igraph, node_list)."""
    nodes = list(G.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    edges, weights = [], []
    for u, v, d in G.edges(data=True):
        edges.append((index[u], index[v]))
        weights.append(float(d.get("weight", 1.0)))
    g = ig.Graph(n=len(nodes), edges=edges)
    if weights:
        g.es["weight"] = weights
    return g, nodes


def _leiden_partition(g: "ig.Graph", resolution: float, seed: int = 42):
    """Weighted, resolution-controlled, reproducible Leiden via leidenalg DIRECTLY.

    We call leidenalg rather than cdlib's `algorithms.leiden` because cdlib's
    wrapper signature is version-fragile (rejects `seed` / `resolution_parameter`
    on some releases). RBConfigurationVertexPartition is the resolution-aware
    objective; find_partition accepts weights + resolution_parameter + seed.
    """
    weights = g.es["weight"] if "weight" in g.es.attributes() else None
    return leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        seed=seed,
    )


def detect_communities(G: nx.Graph, resolution: float = 1.0) -> dict:
    """Run weighted Leiden community detection (L0 + L1) on the product graph.

    leidenalg directly (not cdlib) for version-stable weights/resolution/seed control.

    Returns:
        {
            "level_0": [[node_id, ...], ...],   # finest granularity, node-id lists
            "level_1": [[l0_community_index, ...], ...],   # coarser: groups of L0 communities
        }
    """
    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return {"level_0": [], "level_1": []}

    g, nodes = _nx_to_igraph(G)
    part0 = _leiden_partition(g, resolution)
    level_0 = [[nodes[i] for i in comm] for comm in part0]

    # Anti-blob guard: if one community swallows >40% of the graph, the resolution
    # is too coarse — re-run L0 once at 2x resolution.
    if level_0 and max(len(c) for c in level_0) > 0.40 * n_nodes:
        part0 = _leiden_partition(g, resolution * 2.0)
        level_0 = [[nodes[i] for i in comm] for comm in part0]

    # Level 1: contract to a community graph and re-cluster (guard single community).
    if len(level_0) > 1:
        community_graph = _build_community_graph(G, level_0)
        cg, cg_nodes = _nx_to_igraph(community_graph)
        part1 = _leiden_partition(cg, resolution * 0.5)
        level_1 = [[cg_nodes[i] for i in comm] for comm in part1]
    else:
        level_1 = [list(range(len(level_0)))] if level_0 else []

    return {"level_0": level_0, "level_1": level_1}


def _build_community_graph(G: nx.Graph, communities: list) -> nx.Graph:
    """Contract G to a community-level graph. `communities` = list of node-id lists.

    Each L0 community becomes a node; edge weight = #inter-community edges in G.
    """
    CG = nx.Graph()
    node_to_community = {}
    for i, comm in enumerate(communities):
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
