# retrieval/hybrid.py
"""Hybrid graph + vector retrieval: local, global, and multi-hop search."""
import networkx as nx


def local_search(query: str, G: nx.Graph, colbert, product_index, top_k: int = 10):
    """
    For specific queries like "best noise cancelling headphones under $200".

    1. ColBERT retrieves top-50 products by text similarity
    2. For each retrieved product, expand via graph (1-hop neighbors):
       - bought_together products (high signal)
       - same-brand products
       - products sharing 2+ features
    3. Rerank the expanded set by relevance to query

    Products found via graph get a bonus based on edge type:
    bought_together: +0.3, also_bought: +0.2, similar_to: +0.1
    Then sort by combined score.
    """
    raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §6.1")


def global_search(query: str, community_summaries, summary_embeddings, model, top_k: int = 5):
    """
    For broad queries like "What are the trends in smart home devices?"
    or "Compare audio brands in the mid-range segment."

    This is what basic RAG CANNOT do — it requires aggregated knowledge.

    1. Embed the query
    2. Find top-K most relevant community summaries by cosine similarity
    3. Return the summaries + their key products as context
    """
    raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §6.2")


def multi_hop_search(query: str, G: nx.Graph, entities: list[str], max_hops: int = 2):
    """
    For relational queries like:
    - "What accessories from Sony work with the WH-1000XM5?"
    - "Find products that people buy together with this camera AND this lens"
    - "What's the cheapest product from the same brand that has noise cancelling?"

    These REQUIRE graph traversal — vector similarity alone cannot answer them.

    1. Extract entities from query (brand names, product names, features)
    2. Find matching nodes in the graph
    3. Traverse edges (BFS up to max_hops)
    4. Filter/score the discovered products by the query's constraints

    Returns results with path information (for citations), sorted by hop distance.
    """
    raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §6.3")


def _describe_path(G, path):
    """Convert a node path to a human-readable string."""
    parts = []
    for i, node in enumerate(path):
        node_data = G.nodes[node]
        name = node_data.get("name", node)[:40]
        if i < len(path) - 1:
            edge_data = G.edges[path[i], path[i+1]]
            edge_type = edge_data.get("type", "related")
            parts.append(f"{name} --[{edge_type}]-->")
        else:
            parts.append(name)
    return " ".join(parts)
