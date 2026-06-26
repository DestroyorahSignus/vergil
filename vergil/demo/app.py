# demo/app.py
"""Gradio chat UI with a graph-visualization panel and source citations."""


def create_demo(rag_pipeline, graph):
    """
    Build the Gradio Blocks app: a chat interface that answers product queries via the
    VERGIL RAG pipeline (with a retrieval-method badge + source citations) alongside an
    interactive pyvis visualization of the relevant subgraph.

    Example queries to seed:
    - "What accessories from Sony work with the WH-1000XM5?"
    - "Compare the smart home ecosystems — Alexa vs Google Home"
    - "Find USB-C chargers from Anker that people buy with MacBooks"
    - "What are the trends in wireless earbuds?"
    """
    raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §9")
