# VERGIL — GraphRAG Product Knowledge Graph + Hybrid Retrieval Q&A

VERGIL is a **Graph-enhanced Retrieval-Augmented Generation system** for e-commerce
product discovery and Q&A. It builds a **product knowledge graph** from Amazon Reviews
2023 metadata (products → brands → categories → co-purchase → attribute edges), runs
**Microsoft-style GraphRAG** (entity/relation extraction → Leiden community detection →
hierarchical community summaries), and combines **graph traversal** with a **ColBERT
neural retriever** for hybrid search. A **Qwen2.5-7B-Instruct Q4** (or 3B if VRAM tight)
generates grounded, multi-hop answers with citations. The demo shows product Q&A that
basic vector RAG literally cannot answer (e.g., "What accessories from the same brand
work with this camera?" requires graph reasoning).

VERGIL is **standalone** — it brings its own off-the-shelf embedding model
(`BAAI/bge-small-en-v1.5`) for similarity edges and its own ColBERT for product
retrieval. It has **zero dependency on DANTE**; a downstream consumer (SPARDA) may inject
a stronger fine-tuned bi-encoder via the `encoder=` argument.

## Install

```bash
pip install git+https://github.com/DestroyorahSignus/vergil.git
```

## Retrieval modes

- **Local search** — specific product queries (e.g. "best noise cancelling headphones
  under $200"). ColBERT retrieves candidates, then the graph expands via co-purchase /
  same-brand / shared-feature neighbors and reranks.
- **Global search** — broad/exploratory queries (e.g. "trends in wireless audio"). Routes
  over the hierarchical community summaries; this is what vanilla vector RAG cannot do.
- **Multi-hop search** — relational queries (e.g. "what accessories from Sony work with
  the WH-1000XM5?"). Entity-links the query to graph nodes, BFS-traverses up to N hops,
  and returns discovered products with their reasoning paths as citations.

## Ablation — GraphRAG vs vanilla RAG

| Query type | Vanilla RAG (ColBERT only) | VERGIL (graph + ColBERT) | Δ |
|---|---|---|---|
| Local (product search) | | | should be similar |
| Global (market overview) | | | VERGIL wins big |
| Multi-hop (relational) | | | VERGIL wins huge — vanilla literally can't answer |
| Comparison (brand vs brand) | | | VERGIL wins |

Scoring: manual 1-5 evaluation on faithfulness + relevance + completeness.

## How to run

_TODO: end-to-end build steps (download → graph → communities → RAG → demo). See
`VERGIL_BUILD_PLAN.md` §10 for the build sequence._

## Production note

This project uses **NetworkX in-memory** for the knowledge graph, which is correct at the
50K-product demo scale. For production you would move the graph backend to **Neo4j** or
**Amazon Neptune**; the rest of the architecture (vector index + LLM) is unchanged.
