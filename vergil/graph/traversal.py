# graph/traversal.py
"""Graph query functions: neighbors, paths, subgraph extraction."""
import networkx as nx


def get_neighbors(G: nx.Graph, node_id: str, edge_types: list[str] | None = None) -> list[str]:
    """Return neighbors of a node, optionally filtered by edge type."""
    raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §6")


def shortest_paths(G: nx.Graph, source: str, target: str) -> list[list[str]]:
    """Return shortest path(s) between two nodes for citation/reasoning chains."""
    raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §6")


def extract_subgraph(G: nx.Graph, nodes: list[str], radius: int = 1) -> nx.Graph:
    """Extract the subgraph induced by `nodes` plus their `radius`-hop neighborhood."""
    raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §6")
