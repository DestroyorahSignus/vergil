# retrieval/colbert_index.py
"""Vector index + optional ColBERT rerank over product descriptions.

Two layers, both VERGIL's own (no dependency on DANTE):

1. `VectorIndex` — a FAISS IndexFlatIP over normalized product-description
   embeddings. The encoder is INJECTED (any object with `.encode`); by default
   VERGIL loads its own bge-small so the repo stays standalone. Embeddings can
   be persisted/loaded as .npy so the Modal build stage caches the expensive
   encode once.

2. `maybe_colbert_rerank` — late-interaction MaxSim rerank of a small
   candidate list via `rerankers` (answerai-colbert-small-v1). It NEVER
   crashes the pipeline: any import/load/scoring failure degrades to an
   identity pass-through (mirrors DANTE's colbert_reranker pattern).
"""

# Same default encoder as data/build_graph.py — kept in sync so query
# embeddings live in the same space as the graph's similar_to edges.
DEFAULT_ENCODER = "BAAI/bge-small-en-v1.5"
COLBERT_MODEL = "answerdotai/answerai-colbert-small-v1"

# Module-level reranker cache: load once, and if it ever fails, stop retrying
# (a broken install would otherwise pay the failed-load cost on every query).
_COLBERT_RERANKER = None
_COLBERT_FAILED = False


class VectorIndex:
    """Dense retriever: FAISS inner-product index over product descriptions.

    Embeddings are L2-normalized, so inner product == cosine similarity.
    """

    def __init__(self, encoder=None, model_name: str = DEFAULT_ENCODER):
        """Create an (empty) index.

        Args:
            encoder: any object with `.encode(list[str]) -> array`
                (e.g. a SentenceTransformer). If None, VERGIL lazily loads
                `model_name` on first use — keeping the repo standalone while
                letting SPARDA inject DANTE's fine-tuned bi-encoder.
            model_name: HF id of the default encoder to lazy-load.
        """
        self.model_name = model_name
        self._encoder = encoder
        self.index = None                     # faiss.IndexFlatIP, set by build/load
        self.product_ids: list[str] = []

    @property
    def encoder(self):
        """The injected or lazily-loaded sentence encoder."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def encode(self, texts: list[str], batch_size: int = 256):
        """Encode texts to a float32 L2-normalized (N, dim) matrix."""
        import numpy as np

        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 1000,
        )
        return np.asarray(embeddings, dtype="float32")

    def build(self, product_ids: list[str], documents: list[str],
              embeddings_path: str | None = None) -> "VectorIndex":
        """Index product descriptions for retrieval.

        Args:
            product_ids: ids aligned 1:1 with `documents`.
            documents: description text per product.
            embeddings_path: optional .npy cache. If the file exists and its
                row count matches `product_ids`, encoding is SKIPPED and the
                cached matrix is used; otherwise we encode and write the cache
                (this is what lets the Modal stage pay the encode cost once).

        Returns self, so `VectorIndex().build(...)` chains.
        """
        import os

        import faiss
        import numpy as np

        embeddings = None
        if embeddings_path and os.path.exists(embeddings_path):
            cached = np.load(embeddings_path)
            if cached.shape[0] == len(product_ids):
                embeddings = np.asarray(cached, dtype="float32")
            # else: stale cache (catalog changed) — fall through and re-encode.
        if embeddings is None:
            embeddings = self.encode(documents)
            if embeddings_path:
                np.save(embeddings_path, embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.product_ids = list(product_ids)
        return self

    def load(self, product_ids: list[str], embeddings_path: str) -> "VectorIndex":
        """Load a prebuilt embedding matrix — no documents, no encoding.

        Raises ValueError if the cached matrix and `product_ids` disagree on
        length (a silent mismatch would return wrong product ids from search).
        """
        import faiss
        import numpy as np

        embeddings = np.asarray(np.load(embeddings_path), dtype="float32")
        if embeddings.shape[0] != len(product_ids):
            raise ValueError(
                f"embeddings at {embeddings_path} have {embeddings.shape[0]} rows "
                f"but {len(product_ids)} product_ids were given"
            )
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.product_ids = list(product_ids)
        return self

    def search(self, query: str, top_k: int = 50) -> list[tuple]:
        """Return the top-k (product_id, cosine_score) results for a query."""
        if self.index is None:
            raise RuntimeError("VectorIndex is empty — call build() or load() first")
        query_emb = self.encode([query])
        k = min(top_k, len(self.product_ids))
        scores, indices = self.index.search(query_emb, k)
        return [
            (self.product_ids[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]


def maybe_colbert_rerank(query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
    """ColBERT (MaxSim) rerank of candidate dicts — with identity fallback.

    Args:
        query: the user query.
        candidates: dicts already sorted by the retrieval score; text is taken
            from the first present of "text" / "description" / "name".
        top_k: how many to return (default: all).

    Returns the candidates reordered by ColBERT relevance, each copy annotated
    with "rerank_score". On ANY failure (rerankers not installed, model load
    error, transformers-version breakage, scoring exception) it returns
    `candidates[:top_k]` UNCHANGED — the pipeline never crashes, it just falls
    back to the retrieval-score order (DANTE's colbert_reranker pattern).
    """
    global _COLBERT_RERANKER, _COLBERT_FAILED

    if top_k is None:
        top_k = len(candidates)
    if not candidates or _COLBERT_FAILED:
        return candidates[:top_k]

    try:
        if _COLBERT_RERANKER is None:
            from rerankers import Reranker
            _COLBERT_RERANKER = Reranker(COLBERT_MODEL, model_type="colbert", verbose=0)

        texts = [
            str(c.get("text") or c.get("description") or c.get("name") or "")
            for c in candidates
        ]
        ranked = _COLBERT_RERANKER.rank(
            query=query, docs=texts, doc_ids=list(range(len(texts)))
        )
        reranked = []
        for result in ranked.results:
            doc_id = getattr(result, "doc_id", None)
            if doc_id is None:
                doc_id = result.document.doc_id
            candidate = dict(candidates[int(doc_id)])
            candidate["rerank_score"] = float(result.score)
            reranked.append(candidate)
        return reranked[:top_k]
    except Exception:
        _COLBERT_FAILED = True  # don't pay the failed-load cost again
        return candidates[:top_k]
