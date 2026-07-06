# graph/traversal.py
"""Graph query functions: neighbors, paths, subgraph extraction."""
import networkx as nx

# HUB GUARD — category/brand hub nodes can have thousands of neighbors
# ("cat:electronics" alone touches most of the catalog). Any traversal that
# hops THROUGH such a node caps how many of its neighbors it will visit, so
# a 2-hop BFS stays bounded instead of exploding to the whole graph.
HUB_NEIGHBOR_CAP = 30


def product_neighbors(
    G: nx.Graph, node: str, edge_types: list[str] | None = None, limit: int = 50
) -> list[tuple[str, str]]:
    """Return up to `limit` (neighbor_id, edge_type) pairs for a node.

    Args:
        G: the product knowledge graph.
        node: source node id (product/brand/category/feature).
        edge_types: if given, only edges whose `type` attribute is in this
            list are followed (e.g. ["similar_to", "bought_together"]).
        limit: cap on returned neighbors (hub guard for brand/category nodes).

    Returns [] for unknown nodes — callers never need to pre-check membership.
    """
    if node not in G:
        return []
    results: list[tuple[str, str]] = []
    for neighbor in G.neighbors(node):
        edge_type = G.edges[node, neighbor].get("type", "related")
        if edge_types is not None and edge_type not in edge_types:
            continue
        results.append((neighbor, edge_type))
        if len(results) >= limit:
            break
    return results


def describe_path(G: nx.Graph, path: list[str]) -> str:
    """Convert a node path to a human-readable citation string.

    Example: "WH-1000XM5 --[has_brand]--> Sony --[has_brand]--> Sony Case"
    """
    parts = []
    for i, node in enumerate(path):
        node_data = G.nodes[node]
        name = str(node_data.get("name", node))[:40]
        if i < len(path) - 1:
            edge_data = G.edges[path[i], path[i + 1]]
            edge_type = edge_data.get("type", "related")
            parts.append(f"{name} --[{edge_type}]-->")
        else:
            parts.append(name)
    return " ".join(parts)


def shortest_path_desc(G: nx.Graph, src: str, dst: str) -> str | None:
    """Return the human-readable shortest path between two nodes, or None.

    Wraps nx.shortest_path + describe_path; returns None when either node is
    missing or no path exists (callers treat that as "no citation available").
    """
    try:
        path = nx.shortest_path(G, src, dst)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    return describe_path(G, path)


def extract_subgraph(
    G: nx.Graph, seeds: list[str], hops: int = 1, max_nodes: int = 1000
) -> nx.Graph:
    """Extract the hub-guarded `hops`-neighborhood subgraph around seed nodes.

    BFS from every seed, but visit at most HUB_NEIGHBOR_CAP neighbors per
    node, so hopping through a category/brand hub pulls a bounded sample of
    its products instead of the entire catalog. Stops early at `max_nodes`.

    Returns an independent copy (nodes + induced edges, attributes preserved),
    safe to mutate without touching G.
    """
    visited = {s for s in seeds if s in G}
    frontier = list(visited)
    for _ in range(hops):
        next_frontier = []
        for node in frontier:
            for i, neighbor in enumerate(G.neighbors(node)):
                if i >= HUB_NEIGHBOR_CAP:  # hub guard
                    break
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                next_frontier.append(neighbor)
                if len(visited) >= max_nodes:
                    return G.subgraph(visited).copy()
        frontier = next_frontier
    return G.subgraph(visited).copy()


# --- Back-compat thin wrappers (pre-existing stub names) ---------------------

def get_neighbors(G: nx.Graph, node_id: str, edge_types: list[str] | None = None) -> list[str]:
    """Return neighbors of a node, optionally filtered by edge type."""
    return [n for n, _ in product_neighbors(G, node_id, edge_types=edge_types, limit=10**9)]


def shortest_paths(G: nx.Graph, source: str, target: str) -> list[list[str]]:
    """Return shortest path(s) between two nodes for citation/reasoning chains."""
    try:
        return [list(p) for p in nx.all_shortest_paths(G, source, target)]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
