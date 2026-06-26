# retrieval/colbert_index.py
"""ColBERT index over product descriptions (VERGIL's own retriever — no DANTE)."""


class ColBERTIndex:
    """Late-interaction neural retriever over product descriptions via RAGatouille."""

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        """Load the ColBERT model used to build and query the product index."""
        raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §2/§6.1")

    def build(self, product_ids: list[str], documents: list[str], index_name: str = "vergil_products"):
        """Index product descriptions for retrieval."""
        raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §2/§6.1")

    def search(self, query: str, k: int = 50) -> list[tuple]:
        """Return the top-k (product_id, score) results for a query."""
        raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §6.1")
