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
        self.colbert = colbert   # VectorIndex (name kept for API stability)
        self.llm = llm
        self.summaries = community_summaries
        self.summary_embs = summary_embeddings
        self.encoder = encoder

    def answer(self, query: str) -> dict:
        """
        Route query to the best retrieval strategy, then generate.

        Routing heuristic (see _classify_query):
        - Contains comparison words ("compare", "vs", "trend", ...) → global search
        - Contains relational words ("works with", "compatible", "from same brand",
          "accessories", ...) → multi-hop search
        - Otherwise → local search

        A smarter approach: use the LLM to classify the query type. But the heuristic
        is good enough for a demo and avoids an extra LLM call.

        Multi-hop degrades gracefully: when entity linking finds nothing in the
        graph (empty "discovered"), we fall back to local_search and flag it via
        "fallback": true in the returned dict.

        Returns a dict with keys: answer, query_type, sources, retrieval_method,
        fallback. Multi-hop sources carry `paths` (graph citation chains).
        """
        query_type = self._classify_query(query)
        fallback = False

        if query_type == "global":
            context = global_search(query, self.summaries, self.summary_embs, self.encoder)
            prompt = GLOBAL_QA_PROMPT.format(
                community_summaries=self._format_communities(context),
                query=query,
            )
        elif query_type == "multi_hop":
            entities = self._extract_entities(query)
            context = multi_hop_search(
                query, self.graph, entities, vector_index=self.colbert
            )
            if not context.get("discovered"):
                # No entity matched a graph node (or nothing near the seeds) —
                # answer from plain local search rather than an empty context.
                fallback = True
                context = local_search(query, self.graph, self.colbert)
                prompt = LOCAL_QA_PROMPT.format(
                    product_context=self._format_products(context),
                    graph_context=self._format_graph_context(context),
                    query=query,
                )
            else:
                prompt = MULTI_HOP_QA_PROMPT.format(
                    source_entities=", ".join(entities) if entities else "(none)",
                    discovered_products=self._format_products(context),
                    paths=self._format_paths(context),
                    query=query,
                )
        else:  # local
            context = local_search(query, self.graph, self.colbert)
            prompt = LOCAL_QA_PROMPT.format(
                product_context=self._format_products(context),
                graph_context=self._format_graph_context(context),
                query=query,
            )

        answer = self.llm.generate(prompt, max_tokens=600)

        return {
            "answer": answer,
            "query_type": query_type,
            "sources": context,
            "retrieval_method": "local" if fallback else query_type,
            "fallback": fallback,
        }

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
        blocks = []
        for community in context:
            brands = community.get("key_brands") or []
            # key_brands entries are (name, degree) pairs from the summarizer,
            # but tolerate plain strings (summaries.json round-trips as lists).
            brand_names = ", ".join(
                str(b[0]) if isinstance(b, (list, tuple)) and b else str(b)
                for b in brands[:5]
            )
            header = (
                f"### Cluster {community.get('community_id', '?')} "
                f"({community.get('num_products', '?')} products"
                + (f"; key brands: {brand_names}" if brand_names else "")
                + ")"
            )
            blocks.append(f"{header}\n{str(community.get('summary', '')).strip()}")
        return "\n\n".join(blocks) or "(no relevant product clusters found)"

    def _format_products(self, context) -> str:
        """Format retrieved/discovered products into prompt context text."""
        products = context.get("discovered", []) if isinstance(context, dict) else context
        lines = []
        for product in products[:15]:
            name = product.get("name") or product.get("product_id", "?")
            extras = [f"score: {float(product.get('score', 0.0)):.3f}"]
            if product.get("price") not in (None, "", "None"):
                extras.append(f"price: ${product['price']}")
            if product.get("rating") not in (None, "", "None"):
                extras.append(f"rating: {product['rating']}")
            if product.get("hop_distance") is not None:
                extras.append(f"{product['hop_distance']} hop(s) from query entity")
            lines.append(f"- {name} ({'; '.join(extras)})")
        return "\n".join(lines) or "(no products retrieved)"

    def _format_paths(self, context) -> str:
        """Format multi-hop traversal paths into prompt context text."""
        products = context.get("discovered", []) if isinstance(context, dict) else context
        lines = []
        for product in products[:15]:
            for path in (product.get("paths") or [])[:2]:
                lines.append(f"- {path}")
        return "\n".join(lines) or "(no traversal paths)"

    def _format_graph_context(self, context) -> str:
        """Format local-search graph relationships into prompt context text."""
        products = context.get("discovered", []) if isinstance(context, dict) else context
        lines = [
            f"- {product.get('name') or product.get('product_id', '?')} "
            f"— found via graph edge '{product.get('via_edge')}'"
            for product in products
            if product.get("via_edge")
        ]
        return "\n".join(lines) or "(all results came from direct vector retrieval)"
