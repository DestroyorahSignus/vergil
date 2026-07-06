# demo/app.py
"""Gradio chat UI with a graph-visualization panel and source citations.

gradio (and pyvis, via graph_viz) are imported lazily inside create_demo / __main__
so this module imports fine on machines without them installed.
"""

EXAMPLE_QUERIES = [
    "What accessories from Sony work with the WH-1000XM5?",
    "Compare the smart home ecosystems — Alexa vs Google Home",
    "Find USB-C chargers from Anker that people buy with MacBooks",
    "What are the trends in wireless earbuds?",
]


def _format_answer(result: dict) -> str:
    """Answer text with a retrieval-method badge and source citations."""
    method = str(result.get("retrieval_method") or result.get("query_type") or "?")
    formatted = f"**[{method.upper()} search]**\n\n{result.get('answer') or '(no answer)'}"

    sources = result.get("sources") or []
    if sources:
        formatted += "\n\n---\n**Sources:**\n"
        for src in sources[:5]:
            if isinstance(src, dict):
                name = src.get("name") or src.get("title") or src.get("id")
                if name:
                    formatted += f"- {name}\n"
            elif isinstance(src, str):
                formatted += f"- {src}\n"

    paths = result.get("paths") or result.get("traversal_paths") or []
    if paths:
        formatted += "\n**Graph paths:**\n"
        for p in paths[:5]:
            formatted += f"- `{p}`\n"
    return formatted


def create_demo(rag_pipeline, graph):
    """Build the Gradio Blocks app: chat (route badge + citations) alongside an
    interactive pyvis view of the subgraph relevant to the last query."""
    import gradio as gr

    from .graph_viz import query_subgraph_nodes, visualize_subgraph

    def chat_fn(query, history):
        history = history or []
        if not (query or "").strip():
            return history, ""
        try:
            result = rag_pipeline.answer(query)
            reply = _format_answer(result)
        except Exception as e:
            reply = f"**[ERROR]** {e!r}"
        history = history + [(query, reply)]
        return history, ""  # clear the textbox

    def viz_fn(query):
        if not (query or "").strip():
            return ""
        try:
            entities = rag_pipeline._extract_entities(query)
            nodes = query_subgraph_nodes(graph, entities)
            if not nodes:
                return "<p>No matching nodes found in the graph for this query.</p>"
            html = visualize_subgraph(graph, nodes)
            # sandboxed iframe so pyvis' scripts don't fight gradio's DOM
            escaped = html.replace('"', "&quot;")
            return (f'<iframe srcdoc="{escaped}" style="width:100%;height:420px;'
                    f'border:none;" sandbox="allow-scripts"></iframe>')
        except Exception as e:
            return f"<p>Graph rendering failed: {e!r}</p>"

    with gr.Blocks(title="VERGIL") as demo:
        gr.Markdown("# 🔮 VERGIL — GraphRAG Product Discovery")
        gr.Markdown("Ask anything about electronics products. Try multi-hop queries — "
                    "the badge shows which retrieval route answered you.")

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=400)
                query = gr.Textbox(
                    placeholder="What accessories work with the Sony WH-1000XM5?",
                    label="Your question",
                )
                btn = gr.Button("Ask", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown("### Knowledge graph\nproduct=blue, brand=green, "
                            "category=orange, feature=purple")
                graph_html = gr.HTML(label="Relevant subgraph")

        btn.click(chat_fn, [query, chatbot], [chatbot, query])
        btn.click(viz_fn, [query], graph_html)
        query.submit(chat_fn, [query, chatbot], [chatbot, query])
        query.submit(viz_fn, [query], graph_html)

        gr.Examples(EXAMPLE_QUERIES, query)

    return demo


def _load_pipeline(artifacts_dir: str = "artifacts"):
    """Load build artifacts and assemble a VergilRAG (mirrors the Modal rag_eval
    stage, local paths). Heavy: needs torch/transformers/sentence-transformers."""
    import json
    import os
    import pickle

    import numpy as np
    from sentence_transformers import SentenceTransformer

    from ..generation.llm import QwenLLM
    from ..generation.rag_pipeline import VergilRAG
    from ..retrieval.colbert_index import VectorIndex

    with open(os.path.join(artifacts_dir, "graph.pkl"), "rb") as f:
        G = pickle.load(f)
    with open(os.path.join(artifacts_dir, "summaries.json")) as f:
        summaries = json.load(f)
    summary_embs = np.load(os.path.join(artifacts_dir, "summary_embeddings.npy"))

    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    # build() uses/refreshes the .npy embedding cache itself (row-count checked).
    products = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "product"]
    index = VectorIndex(encoder=encoder).build(
        [n for n, _ in products],
        [d.get("description") or d.get("name", "") for _, d in products],
        embeddings_path=os.path.join(artifacts_dir, "product_embeddings.npy"),
    )

    llm = QwenLLM()  # transformers backend; pass a GGUF path + backend="llama_cpp" on T4
    rag = VergilRAG(G, index, llm, summaries, summary_embs, encoder)
    return rag, G


if __name__ == "__main__":
    rag, G = _load_pipeline()
    create_demo(rag, G).launch()
