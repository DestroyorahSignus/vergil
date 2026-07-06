"""VERGIL — isolated Modal build job for the product knowledge graph + GraphRAG index.

================================================================================
ISOLATION CONTRACT (read this)
================================================================================
Fully self-contained; shares NOTHING with any other project on your Modal workspace:

  * App name      : "vergil-build"         (its own app, no collisions)
  * Volume        : "vergil-artifacts"     (its own storage, created on demand)
  * Secrets       : NONE attached          <-- cannot read the Mongo/company secret
  * No MongoDB, no external services, no reference to any other folder/app.

Modal auth is account-level, so this reuses your account + GPU quota only. Running
or deleting this job's volume has zero effect on the finetunev1 / company work.

================================================================================
WHAT THIS DOES (nothing is *trained* — VERGIL has no trainable model)
================================================================================
  data       (CPU) download Amazon Reviews 2023 Electronics metadata, take top 50K
  graph      (CPU) build the NetworkX knowledge graph + TF-IDF feature edges
  enrich     (GPU) add embedding "similar_to" edges (bge-small encoder)
  community  (CPU) Leiden community detection (L0 + L1)
  summarize  (GPU) write LLM community summaries (Qwen3-4B-Instruct-2507) + embed them
  rag_eval   (GPU) GraphRAG-vs-vanilla ablation over TEST_QUERIES (not part of
             `--stage all`; run explicitly after the build)

The two GPU stages are short bursts: bge-small is tiny, and Qwen here just runs
inference to write ~hundreds of short summaries. (On Modal we run Qwen via
`transformers` on the A100 — the Q4 GGUF / llama-cpp path in the plan is only for
the Kaggle T4 *inference* demo, not for this build.)

It calls VERGIL's OWN package functions (build_product_graph, add_similarity_edges,
detect_communities, summarize_communities, embed_summaries) and only fills the gaps
the package leaves as stubs (the data download and the LLM) inline here.

================================================================================
HOW TO RUN
================================================================================
    pip install modal && modal token new          # one-time auth

    # smoke test on a tiny slice (fast, cheap):
    modal run modal_build.py --stage all --limit 2000

    # full build:
    modal run modal_build.py --stage all

    # individual stages (each reads the previous stage's output from the volume):
    modal run modal_build.py --stage data
    modal run modal_build.py --stage graph
    modal run modal_build.py --stage enrich
    modal run modal_build.py --stage community
    modal run modal_build.py --stage summarize

    # GraphRAG-vs-vanilla eval (after the build; writes /artifacts/rag_eval.json):
    modal run --detach modal_build.py --stage rag_eval

================================================================================
GET THE ARTIFACTS ONTO YOUR PC  (then push to HF Hub / Kaggle, NOT git)
================================================================================
    modal volume get vergil-artifacts /graph.pkl              ./artifacts/graph.pkl
    modal volume get vergil-artifacts /communities.pkl        ./artifacts/communities.pkl
    modal volume get vergil-artifacts /summaries.json         ./artifacts/summaries.json
    modal volume get vergil-artifacts /summary_embeddings.npy ./artifacts/summary_embeddings.npy

================================================================================
CLEAN UP WHEN DONE  (safe — only touches THIS volume)
================================================================================
    modal volume ls     vergil-artifacts /
    modal volume delete vergil-artifacts            # back up first!

================================================================================
IF A STAGE FAILS (CONTINGENCIES)
================================================================================
GENERAL RULE: every stage is idempotent and writes its output to the volume, so the
universal fallback is "fix the cause and re-run just that `--stage`." Later stages
read the earlier stage's pickle/parquet from the volume, so you never restart from 0.

  * Amazon parquet 404 / shard path or count changed
        → `python -c "from huggingface_hub import list_repo_files as l; print([f for f in
          l('McAuley-Lab/Amazon-Reviews-2023', repo_type='dataset') if 'raw_meta_Electronics' in f])"`
          and adjust AMAZON_PARQUET. NEVER use the loading script (dead on datasets>=4).
  * Graph too SPARSE (few edges → tiny/empty communities)
        → bought_together is empty by design (audited); lower SIM_THRESHOLD 0.85→0.80
          and/or raise top_k in add_similarity_edges so brand/category/feature/similar_to
          carry the graph.
  * cdlib / leidenalg / python-igraph install or import failure
        → fall back to networkx `greedy_modularity_communities` (lower quality, no C deps);
          wrap detect_communities accordingly.
  * One giant community (blob)
        → raise LEIDEN_RESOLUTION (1.0→1.5→2.0) and/or cap similar_to edges.
  * Qwen OOM on the A100
        → set LLM_MODEL to a smaller Qwen3 (e.g. "Qwen/Qwen3-1.7B") or load in
          4-bit. Summaries are short; the default Qwen3-4B is already ~half the
          VRAM of the old Qwen2.5-7B, so OOM is unlikely.
  * Summaries empty / hallucinated
        → lower temperature, tighten the prompt, skip <3-product clusters, re-run
          `--stage summarize` (graph + communities are already on the volume).
  * Modal preemption / timeout
        → re-run the failed `--stage`; everything prior is persisted on the volume.
  * Inference (Kaggle T4) OOM later
        → Qwen 3B Q4 GGUF, lazy-load models, faiss-cpu, fewer candidates.
================================================================================
"""

import modal

# ---- App + storage (both private to this project) ---------------------------
app = modal.App("vergil-build")
vol = modal.Volume.from_name("vergil-artifacts", create_if_missing=True)
ARTIFACTS = "/artifacts"

# Image carries every dep VERGIL's package imports at load time (networkx, cdlib,
# leidenalg, sentence-transformers, sklearn, rapidfuzz, faiss) plus transformers
# for the Qwen summary/eval passes. NOTE: llama-cpp is intentionally NOT installed —
# the package's llm.py guards that import (llama_cpp is the Kaggle-T4 GGUF route);
# on Modal QwenLLM/QwenSummarizer run via transformers. `add_local_python_source
# ("vergil")` bundles the local package so the remote functions can call its real code.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Shared ML stack pinned to the versions DANTE validated on Modal 2026-06-28
        # (reproducibility — see VERGIL_BUILD_PLAN.md §11). The graph-specific deps
        # below stay at >= until a VERGIL --stage data/graph smoke captures their
        # resolved versions (don't hard-pin untested versions).
        #
        # 4.x STACK RATIONALE (aligned 2026-07-06): Qwen3-4B-Instruct-2507 needs
        # transformers>=4.51 (satisfied), and the ColBERT reranker (rerankers[transformers])
        # requires the 4.x line — transformers 5.x removed the internals it hooks.
        # This transformers 4.57.6 + sentence-transformers 4.1.0 + numpy 2.2.6
        # combo is the one DANTE validated on Modal 2026-06-28. Note transformers
        # 4.57 already supports the `dtype=` from_pretrained kwarg (torch_dtype is
        # its deprecated alias), so _QwenSummarizer/QwenLLM work unchanged.
        "torch==2.12.1",
        "transformers==4.57.6",
        "accelerate==1.14.0",
        "sentence-transformers==4.1.0",
        "rerankers[transformers]==0.10.0",
        "datasets==5.0.0",
        "pandas==3.0.3",
        "numpy==2.2.6",
        "faiss-cpu==1.14.3",
        # --- graph stack: pin after a VERGIL smoke captures resolved versions ---
        "scikit-learn>=1.3",
        "networkx>=3.2",
        "cdlib>=0.4.0",
        "leidenalg>=0.10.0",
        "python-igraph>=0.11.0",
        "rapidfuzz>=3.6.0",
    )
    .env({"HF_HOME": f"{ARTIFACTS}/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("vergil")
)

# ---- Config (mirror configs/default.yaml) -----------------------------------
# Amazon-2023's HF loading script is dead (datasets>=4 removed trust_remote_code),
# so we read the published parquet shards directly. 10 shards, ~161K rows each.
AMAZON_REPO = "McAuley-Lab/Amazon-Reviews-2023"
AMAZON_PARQUET = [
    f"hf://datasets/{AMAZON_REPO}/raw_meta_Electronics/full-{i:05d}-of-00010.parquet"
    for i in range(10)
]
SUBSAMPLE_N = 50_000
ENCODER_NAME = "BAAI/bge-small-en-v1.5"        # VERGIL's standalone default encoder
# Generator LLM (research 2026-07-06): Qwen3-4B-Instruct-2507 is a strict upgrade
# over Qwen2.5-7B-Instruct — Apache-2.0 + ungated, natively NON-thinking instruct
# (no <think> blocks / no enable_thinking flag needed), supported by transformers
# >=4.51 so fine on the pinned 4.57.6. Benchmarks at/above Qwen3-8B and above
# Qwen2.5-7B while using ~half the VRAM (~2.5GB Q4 GGUF — good for the Kaggle T4
# story). Same standard apply_chat_template + .generate() path, no code changes.
LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LEIDEN_RESOLUTION = 4.0  # 1.0 gave 25 mega-communities (largest 8,187/50K); 4.0 targets ~100-400 finer ones
SIM_THRESHOLD = 0.85

GRAPH_PKL = f"{ARTIFACTS}/graph.pkl"
META_PARQUET = f"{ARTIFACTS}/electronics_meta.parquet"
COMMUNITIES_PKL = f"{ARTIFACTS}/communities.pkl"
SUMMARIES_JSON = f"{ARTIFACTS}/summaries.json"
SUMMARY_EMB_NPY = f"{ARTIFACTS}/summary_embeddings.npy"
PRODUCT_EMB_NPY = f"{ARTIFACTS}/product_embeddings.npy"
RAG_EVAL_JSON = f"{ARTIFACTS}/rag_eval.json"


# ---- The two pieces the package leaves as stubs, implemented inline ----------
def _add_tfidf_features(G, top_k_per_product: int = 5):
    """Add has_feature edges via TF-IDF keywords (mirrors vergil.data.build_graph
    .extract_features, which ships as a stub). Kept here so the build is runnable."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    products = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "product"]
    if not products:
        return
    texts = [d.get("description", "") or d.get("name", "") for _, d in products]
    ids = [n for n, _ in products]
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    matrix = tfidf.fit_transform(texts)
    names = tfidf.get_feature_names_out()
    for i, pid in enumerate(ids):
        row = matrix[i].toarray().ravel()
        for idx in row.argsort()[-top_k_per_product:][::-1]:
            if row[idx] > 0:
                feat = names[idx]
                fid = f"feat:{feat}"
                G.add_node(fid, type="feature", name=feat)
                G.add_edge(pid, fid, type="has_feature", weight=float(row[idx]))


class _QwenSummarizer:
    """Minimal LLM wrapper with the .generate(prompt, max_tokens) shape that
    vergil.graph.summarizer.summarize_communities expects. Uses transformers on the
    A100 (not llama-cpp). Injected by argument — no edit to the package needed."""

    def __init__(self, model_id: str = LLM_MODEL):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id)
        # `dtype=` verified supported on transformers 4.57.x (torch_dtype is the
        # deprecated alias there) — no change needed for the 5.x -> 4.57.6 pin move.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda"
        )
        self.model.eval()

    def generate(self, prompt: str, max_tokens: int = 300, temperature: float = 0.1) -> str:
        import torch

        # Standard chat template — works identically for Qwen3-4B-Instruct-2507.
        # That model is NON-thinking by default (no <think> blocks emitted), so no
        # enable_thinking / thinking-parse handling is required here.
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=temperature > 0, temperature=max(temperature, 1e-4),
                pad_token_id=self.tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


# ---- Stages ------------------------------------------------------------------
@app.function(image=image, volumes={ARTIFACTS: vol}, timeout=90 * 60, cpu=8.0, memory=32768)
def prepare_data(limit: int = 0):
    """Download Amazon Reviews 2023 Electronics metadata, keep the top-N most-reviewed."""
    import os
    from datasets import load_dataset

    os.makedirs(f"{ARTIFACTS}/hf", exist_ok=True)
    print(f"[data] loading Electronics meta from {len(AMAZON_PARQUET)} parquet shards ...")
    # Parquet shards, NOT the (removed) loading script — no trust_remote_code.
    ds = load_dataset("parquet", data_files=AMAZON_PARQUET, split="train")
    print(f"[data] raw products: {len(ds):,}")
    # NOTE: bought_together / also_buy / also_view are unavailable in this dataset
    # (audited 2026-06-26). The graph is built on brand + category + feature +
    # similar_to edges; build_product_graph's co-purchase loops no-op harmlessly.

    ds = ds.filter(
        lambda r: bool(r.get("title")) and r.get("rating_number") is not None,
        num_proc=8,
    )
    n = limit if (limit and limit > 0) else SUBSAMPLE_N
    ds = ds.sort("rating_number", reverse=True).select(range(min(n, len(ds))))
    print(f"[data] keeping top {len(ds):,} by rating_number")

    ds.to_pandas().to_parquet(META_PARQUET)
    vol.commit()
    print(f"[data] saved -> {META_PARQUET}")
    return {"products": len(ds)}


@app.function(image=image, volumes={ARTIFACTS: vol}, timeout=60 * 60, cpu=8.0, memory=32768)
def build_graph():
    """Build the NetworkX knowledge graph (real package code) + TF-IDF feature edges."""
    import pickle
    import pandas as pd
    from vergil.data.build_graph import build_product_graph

    df = pd.read_parquet(META_PARQUET)
    print(f"[graph] building from {len(df):,} products ...")
    G = build_product_graph(df)
    _add_tfidf_features(G)

    with open(GRAPH_PKL, "wb") as f:
        pickle.dump(G, f)
    vol.commit()
    print(f"[graph] nodes={G.number_of_nodes():,} edges={G.number_of_edges():,} -> {GRAPH_PKL}")
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}


@app.function(image=image, volumes={ARTIFACTS: vol}, gpu="A100", timeout=60 * 60)
def enrich_edges():
    """Add embedding 'similar_to' edges using VERGIL's own encoder (real package code)."""
    import pickle
    from sentence_transformers import SentenceTransformer
    from vergil.data.build_graph import add_similarity_edges

    with open(GRAPH_PKL, "rb") as f:
        G = pickle.load(f)
    before = G.number_of_edges()
    encoder = SentenceTransformer(ENCODER_NAME, device="cuda")
    add_similarity_edges(G, encoder=encoder, threshold=SIM_THRESHOLD)

    with open(GRAPH_PKL, "wb") as f:
        pickle.dump(G, f)
    vol.commit()
    print(f"[enrich] added {G.number_of_edges() - before:,} similar_to edges")

    # Connectivity summary — early detection of a sparse / fragmented graph.
    import networkx as nx
    from collections import Counter

    edge_types = Counter(d.get("type", "?") for _, _, d in G.edges(data=True))
    n_nodes = G.number_of_nodes()
    if n_nodes:
        largest_cc = max((len(c) for c in nx.connected_components(G)), default=0)
        frac = largest_cc / n_nodes
    else:
        largest_cc, frac = 0, 0.0
    print(f"[enrich] edges by type: {dict(sorted(edge_types.items()))}")
    print(f"[enrich] nodes={n_nodes:,}  largest_cc={largest_cc:,} ({frac:.1%} of nodes)")
    return {"edges": G.number_of_edges()}


@app.function(image=image, volumes={ARTIFACTS: vol}, timeout=60 * 60, cpu=8.0, memory=32768)
def detect_communities_stage():
    """Leiden community detection, L0 + L1 (real package code)."""
    import pickle
    from vergil.graph.community import detect_communities

    with open(GRAPH_PKL, "rb") as f:
        G = pickle.load(f)
    comms = detect_communities(G, resolution=LEIDEN_RESOLUTION)

    with open(COMMUNITIES_PKL, "wb") as f:
        pickle.dump(comms, f)
    vol.commit()
    print(f"[community] L0={len(comms['level_0'])}  L1={len(comms['level_1'])} -> {COMMUNITIES_PKL}")
    return {"l0": len(comms["level_0"]), "l1": len(comms["level_1"])}


@app.function(image=image, volumes={ARTIFACTS: vol}, gpu="A100", timeout=2 * 60 * 60)
def summarize():
    """Write LLM community summaries (real package code, Qwen injected) and embed them."""
    import json
    import pickle
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from vergil.graph.summarizer import summarize_communities, embed_summaries

    with open(GRAPH_PKL, "rb") as f:
        G = pickle.load(f)
    with open(COMMUNITIES_PKL, "rb") as f:
        comms = pickle.load(f)

    print(f"[summarize] loading {LLM_MODEL} ...")
    llm = _QwenSummarizer()
    print(f"[summarize] summarizing {len(comms['level_0'])} L0 communities ...")
    summaries = summarize_communities(G, comms["level_0"], llm)

    encoder = SentenceTransformer(ENCODER_NAME, device="cuda")
    embs = embed_summaries(summaries, encoder)

    with open(SUMMARIES_JSON, "w") as f:
        json.dump(summaries, f)
    np.save(SUMMARY_EMB_NPY, np.asarray(embs))
    vol.commit()
    print(f"[summarize] {len(summaries)} summaries -> {SUMMARIES_JSON} (+ embeddings)")
    return {"summaries": len(summaries)}


@app.function(image=image, volumes={ARTIFACTS: vol}, gpu="A100-80GB", timeout=2 * 60 * 60)
def rag_eval():
    """End-to-end GraphRAG-vs-vanilla ablation over the built artifacts.

    Loads graph/communities/summaries/summary-embeddings from the volume, builds a
    bge-small VectorIndex over the product descriptions (cached to the volume),
    stands up QwenLLM (transformers backend) + VergilRAG, runs run_rag_ablation,
    writes /artifacts/rag_eval.json, and prints the per-type table + graph stats.
    """
    import json
    import os
    import pickle

    import numpy as np
    from sentence_transformers import SentenceTransformer

    from vergil.eval.evaluate import (
        format_ablation_table, format_graph_stats, graph_statistics, run_rag_ablation,
    )
    from vergil.generation.llm import QwenLLM
    from vergil.generation.rag_pipeline import VergilRAG
    from vergil.retrieval.colbert_index import VectorIndex

    with open(GRAPH_PKL, "rb") as f:
        G = pickle.load(f)
    with open(COMMUNITIES_PKL, "rb") as f:
        comms = pickle.load(f)
    with open(SUMMARIES_JSON) as f:
        summaries = json.load(f)
    summary_embs = np.load(SUMMARY_EMB_NPY)

    encoder = SentenceTransformer(ENCODER_NAME, device="cuda")

    # Product vector index (baseline retriever + VergilRAG's local-search index).
    # build() itself uses/refreshes the .npy embedding cache on the volume
    # (row-count checked against product_ids), so re-runs skip the encode pass.
    products = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "product"]
    cached = os.path.exists(PRODUCT_EMB_NPY)
    print(f"[rag_eval] indexing {len(products):,} products "
          f"({'cached embeddings' if cached else 'encoding — cache miss'}) ...")
    index = VectorIndex(encoder=encoder).build(
        [n for n, _ in products],
        [d.get("description") or d.get("name", "") for _, d in products],
        embeddings_path=PRODUCT_EMB_NPY,
    )
    if not cached:
        vol.commit()
        print(f"[rag_eval] cached -> {PRODUCT_EMB_NPY}")

    print(f"[rag_eval] loading {LLM_MODEL} (transformers backend) ...")
    llm = QwenLLM(LLM_MODEL, backend="transformers")
    rag = VergilRAG(G, index, llm, summaries, summary_embs, encoder)

    results = run_rag_ablation(rag, index, llm)

    with open(RAG_EVAL_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    vol.commit()
    print(f"[rag_eval] results -> {RAG_EVAL_JSON}")

    print(format_ablation_table(results))
    print(format_graph_stats(graph_statistics(G, communities=comms)))
    return {
        "queries": len(results["per_query"]),
        "by_type": {t: a["n"] for t, a in results["by_type"].items()},
    }


@app.local_entrypoint()
def main(stage: str = "all", limit: int = 0):
    """Orchestrate. stage: all | data | graph | enrich | community | summarize | rag_eval.

    NOTE: `all` = the 5 BUILD stages only; rag_eval is run explicitly afterwards
    (it needs V2's retrieval/pipeline code and burns extra A100 minutes):
        modal run --detach modal_build.py --stage rag_eval
    """
    order = ["data", "graph", "enrich", "community", "summarize"]
    if stage != "all" and stage != "rag_eval" and stage not in order:
        raise SystemExit(f"unknown stage {stage!r}; use all|{'|'.join(order)}|rag_eval")
    todo = order if stage == "all" else [stage]

    if "data" in todo:
        print("== data =="); print(prepare_data.remote(limit=limit))
    if "graph" in todo:
        print("== graph =="); print(build_graph.remote())
    if "enrich" in todo:
        print("== enrich =="); print(enrich_edges.remote())
    if "community" in todo:
        print("== community =="); print(detect_communities_stage.remote())
    if "summarize" in todo:
        print("== summarize =="); print(summarize.remote())
    if "rag_eval" in todo:
        print("== rag_eval =="); print(rag_eval.remote())
