# vergil/__init__.py — the ONLY surface other repos import
from .generation.rag_pipeline import VergilRAG          # full GraphRAG pipeline
from .data.build_graph import build_product_graph, add_similarity_edges
from .graph.community import detect_communities
from .graph.summarizer import summarize_communities, embed_summaries
from .retrieval.hybrid import local_search, global_search, multi_hop_search

__all__ = [
    "VergilRAG", "build_product_graph", "detect_communities",
    "summarize_communities", "embed_summaries",
    "local_search", "global_search", "multi_hop_search", "add_similarity_edges",
]
