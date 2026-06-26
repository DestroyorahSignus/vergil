# retrieval/graph_retriever.py
"""Graph-based retrieval: subgraph extraction around seed product nodes."""
import networkx as nx


def graph_expand(G: nx.Graph, seed_ids: list[str], edge_types: list[str] | None = None) -> set:
    """
    Expand a set of seed product nodes via 1-hop graph neighbors.

    Follows high-signal edges (bought_together, also_bought, similar_to, has_brand)
    to discover related products that pure vector similarity would miss.
    """
    raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §6.1")


def graph_score(G: nx.Graph, candidates: set, vector_scores: dict) -> list[tuple]:
    """
    Score graph-expanded candidates by combining their vector score with a
    graph-proximity bonus keyed on edge type.
    """
    raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §6.1")
