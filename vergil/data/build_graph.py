# data/build_graph.py
import faiss
import networkx as nx
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

# VERGIL ships its own default encoder so the repo is fully standalone.
# A consumer (e.g. SPARDA) may inject a stronger fine-tuned model instead.
DEFAULT_ENCODER = "BAAI/bge-small-en-v1.5"   # off-the-shelf, ~0.13GB, no external repo


def _as_list(v) -> list:
    """Normalize a parquet list-field to a plain Python list.

    pyarrow/pandas return list columns (features, categories, description,
    bought_together, ...) as numpy arrays. Using them in a boolean context
    (`if features:`, `field or []`) raises "truth value of an array ... is
    ambiguous". Always funnel such fields through this first.
    """
    if v is None:
        return []
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return list(v)
    if hasattr(v, "tolist"):          # pandas/other array-likes
        try:
            return list(v.tolist())
        except Exception:
            return []
    return [v]                         # scalar → single-element list


def build_product_graph(meta_df: pd.DataFrame) -> nx.Graph:
    """
    Build a heterogeneous knowledge graph from Amazon metadata.

    Node counts (approximate for 50K Electronics subsample):
    - ~50,000 product nodes
    - ~5,000 brand nodes
    - ~500 category nodes
    - ~2,000 feature nodes (extracted)

    Edge counts:
    - bought_together: ~100K+ edges (high-value co-purchase signal)
    - has_brand: ~50K edges (every product → its brand)
    - in_category: ~50K+ edges (products → category path nodes)
    - has_feature: ~80K edges (extracted features)
    - similar_to: ~250K edges (embedding similarity, computed later)
    """
    G = nx.Graph()

    # Precompute the set of valid ASINs ONCE — membership in a pandas .values array
    # is O(N) per check, making the per-row co-purchase loops O(N^2). A set is O(1).
    valid_asins = set(meta_df["parent_asin"])

    for _, row in meta_df.iterrows():
        asin = row["parent_asin"]

        # Add product node
        G.add_node(asin, type="product", name=str(row.get("title") or ""),
                   description=_build_description(row),
                   price=row.get("price"), rating=row.get("average_rating"))

        # Add brand node + edge
        brand = _extract_brand(row)
        if brand:
            # Canonicalize so storefront fragments ("Visit the X Store", "X Store",
            # "by X") collapse to one brand node. Keep the prettiest display name.
            brand_key = _canon_brand(brand)
            brand_id = f"brand:{brand_key}"
            if not G.has_node(brand_id):
                G.add_node(brand_id, type="brand", name=brand)
            else:
                # Keep the prettiest (shortest, title-ish) display name we've seen.
                existing = G.nodes[brand_id].get("name", brand)
                if len(brand) < len(existing):
                    G.nodes[brand_id]["name"] = brand
            G.add_edge(asin, brand_id, type="has_brand")

        # Add category nodes + edges (hierarchical). `categories` is a list of
        # category PATHS (each path a list of strings); both levels may be numpy arrays.
        for cat_path in _as_list(row.get("categories")):
            cats = _as_list(cat_path)
            for i, cat in enumerate(cats):
                cat = str(cat)
                cat_id = f"cat:{cat.lower()}"
                G.add_node(cat_id, type="category", name=cat)
                G.add_edge(asin, cat_id, type="in_category")
                # Category hierarchy edges
                if i > 0:
                    parent_id = f"cat:{str(cats[i-1]).lower()}"
                    G.add_edge(parent_id, cat_id, type="category_parent")

        # Add co-purchase edges (bought_together)
        for related_asin in _as_list(row.get("bought_together")):
            if related_asin in valid_asins:
                G.add_edge(asin, related_asin, type="bought_together", weight=2.0)

        # Add also_bought / also_viewed edges (weaker signal)
        # also_buy/also_view absent in Amazon-2023 Electronics (no-op); kept for forward-compat
        for related_asin in _as_list(row.get("also_buy"))[:10]:  # cap at 10
            if related_asin in valid_asins:
                G.add_edge(asin, related_asin, type="also_bought", weight=1.0)

    return G


def _build_description(row) -> str:
    """Combine title + features + description into a rich text blob.

    features/description are parquet list columns (numpy arrays) — normalize via
    _as_list and coerce each element to str before joining.
    """
    parts = [str(row.get("title") or "")]
    features = _as_list(row.get("features"))
    if features:
        parts.append(" | ".join(str(f) for f in features[:5]))  # top 5 bullet points
    desc = _as_list(row.get("description"))
    if desc:
        parts.append((" ".join(str(d) for d in desc))[:500])  # truncate long descriptions
    return " ".join(p for p in parts if p)


def _extract_brand(row) -> str | None:
    """Extract brand from details dict or store field."""
    details = row.get("details")
    if not isinstance(details, dict):
        details = {}
    brand = details.get("Brand") or details.get("Manufacturer") or row.get("store")
    return str(brand).strip() if brand else None


def _canon_brand(s: str) -> str:
    """Canonical brand key: strip Amazon storefront cruft so fragments collapse.

    Handles "Visit the X Store", "X Store", leading "by "; lowercases and
    collapses internal whitespace. Used only for the brand NODE id (dedup key);
    the human-readable display name is kept separately.
    """
    import re

    s = (s or "").strip()
    # "Visit the X Store" / "Visit X Store" -> X
    s = re.sub(r"^visit\s+(the\s+)?", "", s, flags=re.IGNORECASE)
    # trailing " Store" (storefront suffix) -> drop
    s = re.sub(r"\s+store$", "", s, flags=re.IGNORECASE)
    # leading "by " -> drop
    s = re.sub(r"^by\s+", "", s, flags=re.IGNORECASE)
    # lowercase + collapse whitespace
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return s


def extract_features(G: nx.Graph, product_nodes: list[dict]):
    """
    Extract product features from titles/descriptions.
    Use simple keyword extraction (TF-IDF top terms or KeyBERT).
    These become feature nodes in the graph.

    Examples:
    - "Sony WH-1000XM5 Wireless Noise Cancelling Headphones"
      → features: ["wireless", "noise cancelling", "headphones", "over-ear"]
    - "Anker 65W USB-C Charger"
      → features: ["usb-c", "fast charging", "65w", "portable"]

    Option A: Simple — use sklearn TfidfVectorizer, take top terms per product.
    Option B: Better — use KeyBERT with a small sentence-transformer.
    Option C: Best — use the Qwen2.5 LLM to extract structured features (offline, batch).

    Go with Option A for speed. Option C is a nice "future work" mention.
    """
    raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §4.3")


def add_similarity_edges(G: nx.Graph, encoder=None, threshold: float = 0.85, top_k: int = 5):
    """
    Encode all product descriptions with a bi-encoder, find pairs with cosine > threshold,
    add 'similar_to' edges.

    Args:
        encoder: a SentenceTransformer (or anything with .encode). If None, VERGIL loads
                 its own DEFAULT_ENCODER — keeping the repo standalone. SPARDA passes
                 DANTE's fine-tuned bi-encoder here for higher-quality edges.

    This connects products that are semantically related but NOT in the co-purchase data.
    Only add edges between products of DIFFERENT brands (same-brand connections are
    already captured by has_brand edges).
    """
    if encoder is None:
        encoder = SentenceTransformer(DEFAULT_ENCODER)

    product_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "product"]
    if not product_nodes:
        return
    ids = [n for n, _ in product_nodes]
    texts = [d.get("description", "") or d.get("name", "") for _, d in product_nodes]

    try:
        embeddings = encoder.encode(
            texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # CUDA OOM at batch_size=256 — retry once with a smaller batch.
            embeddings = encoder.encode(
                texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
            )
        else:
            raise

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    # top_k + 1 because the nearest neighbour of each product is itself.
    scores, indices = index.search(embeddings, top_k + 1)

    for i, pid in enumerate(ids):
        pid_brand = _get_brand(G, pid)
        for j in range(1, top_k + 1):
            if scores[i][j] >= threshold:
                neighbor_id = ids[indices[i][j]]
                # Only cross-brand edges — same-brand links already exist via has_brand.
                if _get_brand(G, neighbor_id) != pid_brand:
                    G.add_edge(pid, neighbor_id, type="similar_to", weight=float(scores[i][j]))


def _get_brand(G: nx.Graph, product_id: str):
    """Return the brand NODE id linked to a product via its has_brand edge (or None)."""
    for neighbor in G.neighbors(product_id):
        if G.edges[product_id, neighbor].get("type") == "has_brand":
            return neighbor
    return None
