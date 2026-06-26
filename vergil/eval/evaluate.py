# eval/evaluate.py
"""Evaluation: faithfulness, relevance, and multi-hop accuracy; GraphRAG vs vanilla ablation."""


def evaluate_query(rag, query: str, query_type: str) -> dict:
    """
    Run a single test query through the pipeline and score the answer on a 1-5 scale
    for faithfulness + relevance + completeness.
    """
    raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §8.2")


def run_ablation(rag, vanilla_rag, test_queries: list[dict]) -> dict:
    """
    Run the GraphRAG-vs-vanilla-RAG ablation over the curated test queries and
    produce the comparison table (local / global / multi-hop / comparison).
    """
    raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §8.2")


def graph_statistics(G) -> dict:
    """
    Compute graph stats for the README: node/edge counts by type, community counts
    (L0, L1), average community size, density, average path length, top-10 brands.
    """
    raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §8.3")
