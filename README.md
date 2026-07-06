# VERGIL — GraphRAG Product Knowledge Graph + Hybrid Retrieval Q&A

VERGIL is a **Graph-enhanced Retrieval-Augmented Generation system** for e-commerce
product discovery and Q&A. It builds a **product knowledge graph** from Amazon Reviews
2023 (Electronics) metadata, runs **Microsoft-style GraphRAG** (graph construction →
Leiden community detection → hierarchical community summaries), and combines **graph
traversal** with a **ColBERT neural retriever** for hybrid search. A
**Qwen3-4B-Instruct-2507** generator produces grounded answers with **provenance
citations** — including multi-hop reasoning paths that plain vector RAG cannot produce.

VERGIL is **standalone** — it brings its own embedding model (`BAAI/bge-small-en-v1.5`)
for similarity edges + community search and its own ColBERT for product retrieval. It has
**zero dependency on DANTE**; a downstream consumer (SPARDA) can inject a stronger
fine-tuned bi-encoder via the `encoder=` argument.

## The knowledge graph (full 50K-product build)

| | |
|---|---|
| **Nodes** | **65,570** — product 50,000 · brand 10,051 · feature 4,722 · category 797 |
| **Edges** | **569,734** — has_feature 249,484 · in_category 209,289 · similar_to 60,962 · has_brand 49,999 |
| **Communities** | **73** at level 0, **13** at level 1 (Leiden, resolution 4.0) |
| **Avg product degree** | 12.61 |
| **Top brands by degree** | Amazon Basics, Samsung, Apple, Sony, Logitech, SanDisk … |

Edges are derived from structured metadata (`store`→brand, `categories`, `features`) plus
a `bge-small` k-NN `similar_to` layer. (Amazon-2023's `bought_together` field is empty in
the public release, so co-purchase edges are not used — an honest data-reality call; the
`similar_to` layer plays the "related products" role instead.)

## Retrieval modes (routed by query type)

- **Local search** — specific product queries (e.g. *"4K webcam for streaming"*). ColBERT
  retrieves candidates, then the graph expands via same-brand / shared-feature / similar
  neighbors and reranks.
- **Global search** — broad/exploratory queries (e.g. *"trends in wireless audio"*). Routes
  over the hierarchical **community summaries** — the thing vanilla vector RAG cannot do,
  because no single chunk contains a market-level synthesis.
- **Multi-hop search** — relational queries (e.g. *"accessories from Sony compatible with
  the WH-1000XM5?"*). Fuzzy entity-links the query to graph nodes, traverses the graph, and
  returns discovered products **with their reasoning paths as citations**.

Routing is a fast keyword heuristic (comparison words → global; relational cues like
"works with" / "same brand" / "buy together" → multi-hop; else local). *Future work: an
LLM classifier for ambiguous queries.*

## Evaluation — GraphRAG vs vanilla RAG

12 hand-authored queries (3 local / 4 global / 5 multi-hop) run through both a **vanilla
vector-only** pipeline and the **routed VERGIL** pipeline (generator held constant).

**Routing accuracy: 10/12 (83%).** Local 3/3, global 4/4, multi-hop 3/5 — the two
multi-hop "misses" are defensible label calls (an attribute-filter query correctly went
local; a "JBL *vs* Sonos" query went global on the comparison cue).

Both pipelines return non-empty, cited answers on every query, so the differentiator is
**capability and provenance, not a scalar score** (there is no LLM-judge; grade
`side_by_side` manually):

- **Global** queries: VERGIL grounds the answer in community summaries spanning hundreds of
  products; vanilla can only quote whatever few chunks it retrieved.
- **Multi-hop** queries: VERGIL returns **graph path citations** vanilla structurally
  cannot.

### Flagship example — *"What accessories from Sony are compatible with the WH-1000XM5?"*

- **Vanilla RAG:** *"There are no accessories listed in the provided product information
  that are specifically compatible with the Sony WH-1000XM5…"* — a dead end.
- **VERGIL (multi-hop):** entity-links **Sony** + **WH-1000XM5**, traverses the graph, and
  returns discovered products **with the paths that found them**:

  ```
  Sony --[has_brand]--> Sony WH-1000XM4 Wireless Noise Cancelling Headphones
  Sony --[has_brand]--> Sony WH-1000X Noise Cancelling Headphones
  Sony --[has_brand]--> Sony MDREX14AP In-Ear Earbud Headphones
  ```

  VERGIL is also honest about the graph's limits — it states the KG has no explicit
  *compatible-with* edges (only `has_brand` / `has_feature` / `in_category` / `similar_to`),
  so it surfaces same-brand candidates rather than hallucinating compatibility.

Latency: vanilla ~10–20 s, VERGIL ~22–25 s (graph traversal + expansion adds a few seconds;
accuracy/provenance over speed by design).

## Build the graph + GraphRAG index on Modal

`modal_build.py` is a self-contained Modal job that builds every artifact VERGIL needs: the
knowledge graph, similarity edges, Leiden communities, and the LLM community summaries +
their embeddings. Nothing is *trained* (VERGIL has no trainable model) — the GPU stages are
short bursts (bge-small encoding, Qwen3-4B writing summaries via `transformers`). Fully
isolated: its own app (`vergil-build`) and volume (`vergil-artifacts`), **no secrets**, no
database, no shared state with any other project.

```bash
pip install modal && modal token new              # one-time auth

modal run modal_build.py --stage all --limit 2000 # quick smoke test
modal run modal_build.py --stage all              # full build
```

Stages (each reads the previous stage's output from the volume): `data` (CPU) → `graph`
(CPU) → `enrich` (GPU) → `community` (CPU) → `summarize` (GPU) → `rag_eval` (GPU,
GraphRAG-vs-vanilla ablation). Run one at a time with `--stage <name>`.

Pull artifacts (then push to HF Hub / Kaggle — not git):

```bash
modal volume get vergil-artifacts /graph.pkl              ./artifacts/graph.pkl
modal volume get vergil-artifacts /communities.pkl        ./artifacts/communities.pkl
modal volume get vergil-artifacts /summaries.json         ./artifacts/summaries.json
modal volume get vergil-artifacts /summary_embeddings.npy ./artifacts/summary_embeddings.npy
modal volume get vergil-artifacts /rag_eval.json          ./artifacts/rag_eval.json
```

Clean up (only touches this volume): `modal volume delete vergil-artifacts`.

## Install (as a library)

```bash
pip install git+https://github.com/DestroyorahSignus/vergil.git
```

## Production note

VERGIL uses **NetworkX in-memory** for the knowledge graph, which is correct at the
50K-product demo scale. For production you would move the graph backend to **Neo4j** or
**Amazon Neptune**; the rest of the architecture (vector index + community summaries + LLM)
is unchanged.
