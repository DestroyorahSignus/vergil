# graph/summarizer.py
import networkx as nx

COMMUNITY_SUMMARY_PROMPT = """You are analyzing a cluster of related products from an e-commerce catalog.

Here are the products in this cluster:
{product_list}

Here are the relationships between them:
{edge_list}

Provide a concise summary (3-5 sentences) of this product cluster:
1. What is the main theme/category of this cluster?
2. What are the key brands represented?
3. What are the common features/use cases?
4. What types of products are frequently bought together?

Summary:"""


def summarize_communities(G: nx.Graph, communities: list[list[str]], llm) -> list[dict]:
    """
    Generate a natural-language summary for each community using the LLM.

    This is the expensive step — batch it. For 50K products / ~500 communities,
    expect ~500 LLM calls. At ~0.5s each on T4 with Qwen2.5-7B-Q4, that's ~4 min.

    For training on A100, this is even faster. Or pre-compute on CPU with a smaller model.
    """
    summaries = []
    for i, community in enumerate(communities):
        # Collect product node names for this community.
        product_names = [
            G.nodes[n].get("name", n)
            for n in community
            if G.nodes.get(n, {}).get("type") == "product"
        ]

        # Tiny/empty clusters make the LLM hallucinate — emit a deterministic stub
        # (same dict shape) and skip the LLM call entirely.
        if len(product_names) < 3:
            summaries.append({
                "community_id": i,
                "summary": "Small cluster: " + "; ".join(product_names[:3]),
                "num_products": len(product_names),
                "key_brands": _get_top_brands(G, community),
                "product_ids": community,
            })
            continue

        # Get product info for this community
        product_list = []
        for node_id in community[:30]:  # cap at 30 products per summary
            node_data = G.nodes[node_id]
            if node_data.get("type") == "product":
                product_list.append(f"- {node_data['name']} (Brand: {_get_brand(G, node_id)})")

        # Get internal edges
        subgraph = G.subgraph(community)
        edge_list = []
        for u, v, data in subgraph.edges(data=True):
            edge_list.append(f"- {G.nodes[u]['name'][:50]} --[{data.get('type', 'related')}]--> {G.nodes[v]['name'][:50]}")

        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            product_list="\n".join(product_list[:20]),
            edge_list="\n".join(edge_list[:20])
        )

        summary = llm.generate(prompt, max_tokens=300)
        summaries.append({
            "community_id": i,
            "summary": summary,
            "num_products": len([n for n in community if G.nodes[n].get("type") == "product"]),
            "key_brands": _get_top_brands(G, community),
            "product_ids": community,
        })

    return summaries


def embed_summaries(summaries: list[dict], model) -> tuple:
    """
    Encode community summaries with the bi-encoder for vector retrieval.
    These enable "global search" — finding relevant communities for broad queries.
    """
    texts = [s["summary"] for s in summaries]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings  # shape: (num_communities, embedding_dim)


def _get_brand(G: nx.Graph, product_id: str) -> str:
    """Return the brand NAME of a product node (via its has_brand edge), or 'Unknown'."""
    for neighbor in G.neighbors(product_id):
        if G.nodes.get(neighbor, {}).get("type") == "brand":
            return G.nodes[neighbor].get("name", "Unknown")
    return "Unknown"


def _get_top_brands(G: nx.Graph, community: list[str], top_n: int = 5) -> list[str]:
    """Return the most-connected brands among the nodes in a community."""
    brands = {}
    for node in community:
        if G.nodes.get(node, {}).get("type") == "brand":
            brands[G.nodes[node].get("name", node)] = sum(1 for _ in G.neighbors(node))
    return sorted(brands.items(), key=lambda x: x[1], reverse=True)[:top_n]
