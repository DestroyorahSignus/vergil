# retrieval/hybrid.py
"""Hybrid graph + vector retrieval: local, global, and multi-hop search."""
import weakref

import networkx as nx

from ..graph.traversal import describe_path, extract_subgraph

# Edge-type bonus for graph-expanded candidates (see local_search docstring).
# bought_together is the highest-signal edge but is RARE in Amazon-2023
# (co-purchase fields are mostly empty — §3 DATA REALITY), so in practice the
# expansion runs on similar_to + same-brand.
EDGE_BONUS = {
    "bought_together": 0.30,
    "similar_to": 0.15,
    "has_brand": 0.10,   # same-brand sibling, reached through the brand node
}

# A graph-discovered product has no vector score of its own; it inherits its
# seed's vector score at this discount, plus the edge bonus above.
_GRAPH_SEED_DISCOUNT = 0.5

# How many sibling products to pull when hopping through a brand node.
_SIBLINGS_PER_HUB = 10

# Entity-linking caps for multi_hop_search.
_MATCHES_PER_ENTITY = 3
_MAX_MATCHED_NODES = 10
_FUZZ_CUTOFF = 80

# Per-graph cache of the entity-linking name index (see _get_name_index).
_NAME_INDEX_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _get_name_index(G) -> tuple[list[tuple], list[str]]:
    """(node_id, lowercased-name) pairs for product/brand nodes, built once per graph.

    multi_hop_search used to rebuild this by iterating every node in the ~65K-node
    graph on EVERY query; caching it per graph object makes entity linking pay only
    the per-entity match cost. WeakKeyDictionary ties the cache entry's lifetime to
    the graph itself (no stale index if a new graph is loaded)."""
    cached = _NAME_INDEX_CACHE.get(G)
    if cached is None:
        name_index = [
            (node_id, str(data.get("name", "")).lower())
            for node_id, data in G.nodes(data=True)
            if data.get("type") in ("product", "brand") and data.get("name")
        ]
        cached = (name_index, [name for _, name in name_index])
        _NAME_INDEX_CACHE[G] = cached
    return cached


def local_search(query: str, G: nx.Graph, vector_index, top_k: int = 10) -> list[dict]:
    """Vector retrieve -> 1-hop graph expand -> (optional) ColBERT rerank.

    For specific queries like "best noise cancelling headphones under $200".

    1. `vector_index` retrieves the top-50 products by embedding similarity.
    2. Each hit is expanded 1 hop through the graph:
       - bought_together neighbors (bonus +0.30 — high signal but rare, the
         Amazon-2023 co-purchase fields are mostly empty)
       - similar_to neighbors (bonus +0.15 — semantic siblings)
       - same-brand siblings via the brand node (bonus +0.10, capped at
         _SIBLINGS_PER_HUB per brand so mega-brands don't flood the pool)
    3. Scoring: a vector hit keeps its cosine score; if it is ALSO reachable
       via a graph edge it gains that edge's bonus (best bonus only, applied
       once). A graph-only discovery scores seed_score * 0.5 + bonus, so it
       can rank but never far above the seed that surfaced it.
    4. The pooled candidates go through `maybe_colbert_rerank` — on any
       reranker failure the pool order is kept as-is (identity fallback).

    Returns top_k dicts: {product_id, name, score, source, via_edge}
    (+ price/rating when present on the graph node, and rerank_score when the
    ColBERT rerank ran).
    """
    from .colbert_index import maybe_colbert_rerank

    vector_results = vector_index.search(query, top_k=50)

    # candidate pool: product_id -> {base, bonus, source, via_edge}
    pool: dict[str, dict] = {}

    def _upsert(pid: str, base: float, bonus: float, source: str, via_edge: str | None):
        entry = pool.get(pid)
        if entry is None:
            pool[pid] = {"base": base, "bonus": bonus, "source": source, "via_edge": via_edge}
            return
        # Vector evidence always wins as the source/base; bonuses keep the best edge.
        if source == "vector":
            entry["source"] = "vector"
        entry["base"] = max(entry["base"], base)
        if bonus > entry["bonus"]:
            entry["bonus"] = bonus
            entry["via_edge"] = via_edge

    # Step 1: vector hits.
    for product_id, score in vector_results:
        if product_id not in G:
            continue
        _upsert(product_id, float(score), 0.0, "vector", None)

    # Step 2: 1-hop graph expansion. similar_to / bought_together are direct
    # product<->product edges; same-brand siblings are reached by hopping
    # THROUGH the brand node (2 physical hops, 1 semantic hop).
    for product_id, score in vector_results:
        if product_id not in G:
            continue
        graph_base = float(score) * _GRAPH_SEED_DISCOUNT
        for neighbor in G.neighbors(product_id):
            edge_type = G.edges[product_id, neighbor].get("type", "related")
            neighbor_type = G.nodes[neighbor].get("type")
            if neighbor_type == "product" and edge_type in ("bought_together", "similar_to"):
                _upsert(neighbor, graph_base, EDGE_BONUS[edge_type], "graph", edge_type)
            elif neighbor_type == "brand":
                pulled = 0
                for sibling in G.neighbors(neighbor):
                    if sibling == product_id or G.nodes[sibling].get("type") != "product":
                        continue
                    _upsert(sibling, graph_base, EDGE_BONUS["has_brand"], "graph", "has_brand")
                    pulled += 1
                    if pulled >= _SIBLINGS_PER_HUB:
                        break

    # Step 3: materialize + sort by combined score.
    candidates = []
    for pid, entry in pool.items():
        node = G.nodes[pid]
        candidates.append({
            "product_id": pid,
            "name": node.get("name", pid),
            "score": entry["base"] + entry["bonus"],
            "source": entry["source"],
            "via_edge": entry["via_edge"],
            "price": node.get("price"),
            "rating": node.get("rating"),
            "text": node.get("description") or node.get("name", ""),
        })
    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Step 4: optional ColBERT rerank over a bounded pool (identity fallback).
    rerank_pool = candidates[: max(top_k * 3, 30)]
    results = maybe_colbert_rerank(query, rerank_pool, top_k=top_k)
    for result in results:
        result.pop("text", None)  # internal rerank field, not part of the contract
    return results


def global_search(query: str, community_summaries, summary_embeddings, encoder,
                  top_k: int = 5) -> list[dict]:
    """Cosine retrieval over community summaries (broad/exploratory queries).

    For queries like "What are the trends in smart home devices?" or "Compare
    audio brands in the mid-range segment" — what basic RAG cannot do, since
    it requires knowledge AGGREGATED across many products.

    1. Embed the query with the same encoder that embedded the summaries.
    2. Cosine similarity against `summary_embeddings` (rows re-normalized
       defensively; the stored bge-small matrix is already normalized).
    3. Return the top-K summaries with their key brands + sample products.

    Returns dicts: {community_id, summary, key_brands, num_products, score,
    sample_product_ids}.
    """
    import numpy as np

    query_emb = np.asarray(encoder.encode([query]), dtype="float32")[0]
    query_norm = np.linalg.norm(query_emb)
    if query_norm > 0:
        query_emb = query_emb / query_norm

    embeddings = np.asarray(summary_embeddings, dtype="float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    similarities = (embeddings / norms) @ query_emb

    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        summary = community_summaries[int(idx)]
        results.append({
            "community_id": summary.get("community_id", int(idx)),
            "summary": summary.get("summary", ""),
            "key_brands": summary.get("key_brands", []),
            "num_products": summary.get("num_products", 0),
            "score": float(similarities[idx]),
            "sample_product_ids": list(summary.get("product_ids", []))[:5],
        })
    return results


def multi_hop_search(query: str, G: nx.Graph, entities: list[str], max_hops: int = 2,
                     vector_index=None) -> dict:
    """Entity-link -> hub-guarded BFS -> score products (relational queries).

    For queries that REQUIRE graph traversal — vector similarity alone cannot
    answer them:
    - "What accessories from Sony work with the WH-1000XM5?"
    - "What's the cheapest product from the same brand that has noise cancelling?"

    1. Entity linking: rapidfuzz token_set_ratio >= 80 against product+brand
       node names (fuzzy handles "WH1000XM5" vs "WH-1000XM5"; exact-substring
       hits are preferred over fuzzy ones). Matches are capped
       (_MATCHES_PER_ENTITY per entity, _MAX_MATCHED_NODES total) so a generic
       entity can't seed the whole catalog.
    2. BFS <= max_hops from the matched seeds via extract_subgraph, whose
       hub-guard caps neighbors-per-node so brand/category hubs don't explode.
    3. Collect discovered product nodes and score them against the query —
       embedding cosine when `vector_index` is given, else rapidfuzz
       token_set_ratio on the product name.
    4. Each product carries `paths`: the human-readable shortest path from its
       CLOSEST seed (the citation/reasoning chain), via shortest-path search
       inside the extracted subgraph.

    Returns {"discovered": [ {product_id, name, score, hop_distance, paths} ],
    "matched_nodes": [ {node_id, name, type} ], "note": str | None}, with
    "discovered" sorted closest-first (ties broken by score). When NO entity
    matches a graph node, "discovered" is [] and "note" says why — the
    pipeline uses that to fall back to local_search.
    """
    entities = [e for e in (entities or []) if isinstance(e, str) and e.strip()]
    if not entities:
        return {"discovered": [], "matched_nodes": [],
                "note": "no entities extracted from the query"}

    from rapidfuzz import fuzz, process

    # Step 1: entity linking against product + brand node names only
    # (category/feature hubs as seeds would drown the BFS in generic nodes).
    # The index is cached per graph (WeakKeyDictionary — auto-evicted when the
    # graph is gc'd), so the 65K-node scan happens ONCE, not per query.
    name_index, names_only = _get_name_index(G)

    matched: list[str] = []
    for entity in entities:
        needle = entity.lower().strip()
        # Prefer exact-substring hits (model numbers, brand names).
        substring_hits = [i for i, name in enumerate(names_only) if needle in name]
        if substring_hits:
            # Shortest names first — the tightest match, not a keyword-stuffed title.
            substring_hits.sort(key=lambda i: len(names_only[i]))
            matched.extend(name_index[i][0] for i in substring_hits[:_MATCHES_PER_ENTITY])
            continue
        fuzzy_hits = process.extract(
            needle, names_only, scorer=fuzz.token_set_ratio,
            limit=_MATCHES_PER_ENTITY, score_cutoff=_FUZZ_CUTOFF,
        )
        matched.extend(name_index[idx][0] for (_, _, idx) in fuzzy_hits)

    matched = list(dict.fromkeys(matched))[:_MAX_MATCHED_NODES]  # dedup, keep order, cap
    if not matched:
        return {"discovered": [], "matched_nodes": [],
                "note": "no graph nodes matched the query entities"}

    matched_info = [
        {"node_id": n, "name": G.nodes[n].get("name", n), "type": G.nodes[n].get("type")}
        for n in matched
    ]

    # Step 2: hub-guarded BFS around the seeds.
    subgraph = extract_subgraph(G, matched, hops=max_hops, max_nodes=1000)
    seed_set = set(matched)
    product_nodes = [
        n for n in subgraph.nodes
        if subgraph.nodes[n].get("type") == "product" and n not in seed_set
    ]
    if not product_nodes:
        return {"discovered": [], "matched_nodes": matched_info,
                "note": "graph traversal found no products near the matched entities"}

    # Step 3: score discovered products against the query.
    if vector_index is not None:
        texts = [
            subgraph.nodes[n].get("description") or subgraph.nodes[n].get("name", "")
            for n in product_nodes
        ]
        query_emb = vector_index.encode([query])[0]
        scores = (vector_index.encode(texts) @ query_emb).tolist()
    else:
        query_lower = query.lower()
        scores = [
            fuzz.token_set_ratio(query_lower, str(subgraph.nodes[n].get("name", "")).lower()) / 100.0
            for n in product_nodes
        ]

    # Step 4: attach the citation path from the closest seed.
    discovered = []
    for node_id, score in zip(product_nodes, scores):
        best_path = None
        for seed in matched:
            try:
                path = nx.shortest_path(subgraph, seed, node_id)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if best_path is None or len(path) < len(best_path):
                best_path = path
        if best_path is None:
            continue
        discovered.append({
            "product_id": node_id,
            "name": subgraph.nodes[node_id].get("name", node_id),
            "score": float(score),
            "hop_distance": len(best_path) - 1,
            "paths": [describe_path(subgraph, best_path)],
        })

    # Closest first, best-scoring within the same hop distance.
    discovered.sort(key=lambda d: (d["hop_distance"], -d["score"]))
    return {"discovered": discovered[:20], "matched_nodes": matched_info, "note": None}
