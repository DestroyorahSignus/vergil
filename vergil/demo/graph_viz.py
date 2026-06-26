# demo/graph_viz.py
"""Pyvis interactive visualization of a relevant subgraph."""
import networkx as nx


def visualize_subgraph(G: nx.Graph, nodes: list[str], height: str = "400px") -> str:
    """
    Render the subgraph induced by `nodes` as an interactive pyvis HTML string.

    Color nodes by type: product=blue, brand=green, category=orange, feature=purple.
    """
    raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §9")
