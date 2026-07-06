# retrieval/graph_retriever.py
"""Graph-based retrieval: subgraph extraction around seed product nodes."""
import networkx as nx

from ..graph.traversal import extract_subgraph, product_neighbors
from .hybrid import EDGE_BONUS


def expand_products(G: nx.Graph, product_ids: list[str], hops: int = 1) -> set:
    """Return product node ids within `hops` of the seed products.

    Thin wrapper over traversal.extract_subgraph (so it inherits the
    hub-guard) that keeps only PRODUCT nodes and drops the seeds themselves —
    the "related products vector similarity would miss" set.
    """
    subgraph = extract_subgraph(G, product_ids, hops=hops)
    seeds = set(product_ids)
    return {
        n for n in subgraph.nodes
        if subgraph.nodes[n].get("type") == "product" and n not in seeds
    }


# --- Back-compat thin wrappers (pre-existing stub names) ---------------------

def graph_expand(G: nx.Graph, seed_ids: list[str], edge_types: list[str] | None = None) -> set:
    """
    Expand a set of seed product nodes via 1-hop graph neighbors.

    Follows high-signal edges (bought_together, also_bought, similar_to, has_brand)
    to discover related products that pure vector similarity would miss.
    """
    if edge_types is None:
        edge_types = ["bought_together", "also_bought", "similar_to", "has_brand"]
    expanded = set()
    for seed in seed_ids:
        for neighbor, _ in product_neighbors(G, seed, edge_types=edge_types):
            expanded.add(neighbor)
    return expanded


def graph_score(G: nx.Graph, candidates: set, vector_scores: dict) -> list[tuple]:
    """
    Score graph-expanded candidates by combining their vector score with a
    graph-proximity bonus keyed on edge type.

    Each candidate scores vector_scores.get(id, 0.0) plus the best EDGE_BONUS
    among its edges to vector-scored nodes. Returns (id, score) sorted desc.
    """
    scored = []
    for candidate in candidates:
        base = float(vector_scores.get(candidate, 0.0))
        bonus = 0.0
        if candidate in G:
            for neighbor in G.neighbors(candidate):
                if neighbor not in vector_scores:
                    continue
                edge_type = G.edges[candidate, neighbor].get("type", "related")
                bonus = max(bonus, EDGE_BONUS.get(edge_type, 0.0))
        scored.append((candidate, base + bonus))
    return sorted(scored, key=lambda x: x[1], reverse=True)
