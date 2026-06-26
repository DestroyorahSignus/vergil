# generation/rag_pipeline.py
from .prompts import (
    LOCAL_QA_PROMPT,
    GLOBAL_QA_PROMPT,
    MULTI_HOP_QA_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
)
from ..retrieval.hybrid import local_search, global_search, multi_hop_search


class VergilRAG:
    def __init__(self, graph, colbert, llm, community_summaries, summary_embeddings, encoder):
        self.graph = graph
        self.colbert = colbert
        self.llm = llm
        self.summaries = community_summaries
        self.summary_embs = summary_embeddings
        self.encoder = encoder

    def answer(self, query: str) -> dict:
        """
        Route query to the best retrieval strategy, then generate.

        Routing heuristic:
        - Contains comparison words ("compare", "vs", "difference", "trend") → global search
        - Contains relational words ("works with", "compatible", "from same brand",
          "accessories for") → multi-hop search
        - Otherwise → local search

        A smarter approach: use the LLM to classify the query type. But the heuristic
        is good enough for a demo and avoids an extra LLM call.

        Returns a dict with keys: answer, query_type, sources, retrieval_method.
        """
        raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §7.3")

    def _classify_query(self, query: str) -> str:
        q = query.lower()
        global_keywords = ["compare", "vs", "versus", "trend", "overview", "popular",
                           "best brands", "market", "landscape"]
        multi_hop_keywords = ["works with", "compatible", "accessories", "from same brand",
                              "bought together", "pair with", "goes with", "along with"]
        if any(kw in q for kw in global_keywords):
            return "global"
        if any(kw in q for kw in multi_hop_keywords):
            return "multi_hop"
        return "local"

    def _extract_entities(self, query: str) -> list[str]:
        response = self.llm.generate(
            ENTITY_EXTRACTION_PROMPT.format(query=query),
            max_tokens=100, temperature=0.0
        )
        try:
            import json
            return json.loads(response)
        except Exception:
            # Fallback: simple noun extraction
            return [w for w in query.split() if len(w) > 3 and w[0].isupper()]

    def _format_communities(self, context) -> str:
        """Format global-search community results into prompt context text."""
        raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §7.3")

    def _format_products(self, context) -> str:
        """Format retrieved/discovered products into prompt context text."""
        raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §7.3")

    def _format_paths(self, context) -> str:
        """Format multi-hop traversal paths into prompt context text."""
        raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §7.3")

    def _format_graph_context(self, context) -> str:
        """Format local-search graph relationships into prompt context text."""
        raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §7.3")
