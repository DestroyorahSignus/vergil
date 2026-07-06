# eval/evaluate.py
"""GraphRAG-vs-vanilla ablation harness + graph statistics.

The ablation is deliberately honest and simple (VERGIL_BUILD_PLAN.md §8.2):

* VANILLA baseline — pure vector RAG: ``vector_index.search`` top-10 products
  stuffed into ``LOCAL_QA_PROMPT`` (no graph context), one LLM call.
* VERGIL — ``rag.answer(query)``: routed local / global / multi-hop GraphRAG.

Automatic scoring stays mechanical (NO LLM-judge):
  answer_nonempty — the model produced text;
  cites_sources   — >=1 retrieved product / community / path is actually
                    referenced in the answer text;
  used_graph      — (multi_hop only) VERGIL's traversal returned non-empty paths.

Everything subjective (faithfulness / completeness, 1-5) is left to manual
grading over the returned ``side_by_side`` list.
"""

import time

from ..generation.prompts import LOCAL_QA_PROMPT
from .test_queries import TEST_QUERIES


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _product_text(G, pid) -> str:
    """One prompt line for a product node (name, truncated description, price)."""
    d = G.nodes.get(pid, {}) if G is not None else {}
    name = d.get("name") or str(pid)
    desc = (d.get("description") or "")[:300]
    price = d.get("price")
    price_s = f" | price: {price}" if price not in (None, "", "None") else ""
    return f"- {name}{price_s}\n  {desc}".rstrip()


def _hit_ids(hits) -> list:
    """Normalize vector_index.search output ((id, score) tuples or dicts) to ids."""
    ids = []
    for h in hits:
        if isinstance(h, dict):
            ids.append(h.get("id") or h.get("product_id") or h.get("node_id"))
        elif isinstance(h, (tuple, list)) and len(h):
            ids.append(h[0])
        else:
            ids.append(h)
    return [i for i in ids if i is not None]


def _source_items(sources) -> list:
    """Normalize an answer()'s sources to a flat list. Local/global search return
    a list of dicts; multi_hop returns {"discovered": [...], ...}."""
    if isinstance(sources, dict):
        return list(sources.get("discovered") or [])
    return list(sources or [])


def _source_names(sources, G=None) -> list[str]:
    """Citable names out of a heterogeneous sources context: product names,
    community key-brands + cluster ids, plain node ids/strings."""
    names = []
    for src in _source_items(sources):
        if isinstance(src, dict):
            if "community_id" in src or "key_brands" in src:  # community dict
                names.append(f"Cluster {src.get('community_id', '?')}")
                for b in src.get("key_brands") or []:
                    names.append(str(b[0]) if isinstance(b, (list, tuple)) and b else str(b))
            else:  # product dict
                name = src.get("name") or src.get("product_id") or src.get("id")
                if name:
                    names.append(str(name))
        elif isinstance(src, str):
            if G is not None and src in G.nodes:
                names.append(str(G.nodes[src].get("name") or src))
            else:
                names.append(src)
    return [n for n in names if n]


_CITE_STOPWORDS = {"the", "a", "an", "new", "best", "with", "for", "and", "pro", "mini"}


def _cites(answer: str, names: list[str]) -> bool:
    """True if >=1 source name is plausibly referenced in the answer.

    Product titles are long and LLMs quote partial names, so try progressively
    shorter word-prefixes of each name (6 -> 2 words), then fall back to a
    distinctive leading token (brand / model number)."""
    if not answer or not names:
        return False
    ans = " ".join(answer.lower().split())
    for name in names:
        norm = " ".join(str(name).lower().split())
        if not norm:
            continue
        words = norm.split()
        for k in (6, 4, 3, 2):
            prefix = " ".join(words[:k])
            if len(prefix) >= 8 and prefix in ans:
                return True
        first = words[0]
        if len(first) >= 4 and first not in _CITE_STOPWORDS and first in ans:
            return True
    return False


def _extract_paths(result: dict) -> list:
    """Pull traversal paths out of a VergilRAG answer dict. Multi-hop sources
    carry them per discovered product (product["paths"]); tolerate top-level
    keys too, defensively."""
    for key in ("paths", "traversal_paths", "graph_paths"):
        if result.get(key):
            return list(result[key])
    paths = []
    for src in _source_items(result.get("sources")):
        if isinstance(src, dict):
            paths.extend(src.get("paths") or [])
            if src.get("path"):
                paths.append(src["path"])
    return paths


# ---------------------------------------------------------------------------
# vanilla baseline
# ---------------------------------------------------------------------------

def vanilla_answer(query: str, vector_index, llm, G=None, top_k: int = 10) -> dict:
    """Pure vector RAG: top-k product hits -> LOCAL_QA_PROMPT (no graph context)."""
    t0 = time.time()
    hits = vector_index.search(query, top_k)
    ids = _hit_ids(hits)[:top_k]
    product_context = "\n".join(_product_text(G, pid) for pid in ids) or "(no products retrieved)"
    prompt = LOCAL_QA_PROMPT.format(
        product_context=product_context,
        graph_context="(none — vanilla vector-RAG baseline, no knowledge graph)",
        query=query,
    )
    answer = llm.generate(prompt, max_tokens=512, temperature=0.1)
    names = [str((G.nodes.get(pid, {}) if G is not None else {}).get("name") or pid) for pid in ids]
    return {
        "answer": answer,
        "sources": names,
        "latency_s": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# the ablation
# ---------------------------------------------------------------------------

def run_rag_ablation(rag, vector_index, llm, queries: list[dict] = TEST_QUERIES) -> dict:
    """Run every test query through BOTH systems and score mechanically.

    Args:
        rag: a built ``VergilRAG`` (routed GraphRAG pipeline).
        vector_index: ``VectorIndex`` over product descriptions (the baseline
            retriever; also usually the same index ``rag`` uses for local search).
        llm: ``QwenLLM`` (shared by both sides — the LLM is controlled, only
            retrieval differs).
        queries: list of {"query": str, "type": "local"|"global"|"multi_hop"}.

    Returns:
        {"per_query": [...], "by_type": {...}, "side_by_side": [...]}
        — JSON-serializable; ``format_ablation_table`` renders "by_type".
    """
    G = getattr(rag, "graph", None)
    per_query, side_by_side = [], []

    for i, q in enumerate(queries):
        query, qtype = q["query"], q["type"]
        print(f"[ablation] {i + 1}/{len(queries)} ({qtype}): {query}")

        # -- vanilla ----------------------------------------------------------
        try:
            van = vanilla_answer(query, vector_index, llm, G=G)
        except Exception as e:  # keep the run alive; record the failure
            van = {"answer": "", "sources": [], "latency_s": 0.0, "error": repr(e)}

        # -- vergil (routed) ----------------------------------------------------
        t0 = time.time()
        try:
            res = rag.answer(query)
        except Exception as e:
            res = {"answer": "", "query_type": "error", "sources": [],
                   "retrieval_method": "error", "error": repr(e)}
        ver_latency = round(time.time() - t0, 2)

        ver_answer = res.get("answer") or ""
        ver_names = _source_names(res.get("sources"), G=G)
        paths = _extract_paths(res)

        row = {
            "query": query,
            "type": qtype,
            "route": res.get("retrieval_method") or res.get("query_type") or "?",
            "vanilla": {
                "answer_nonempty": bool(van["answer"].strip()),
                "cites_sources": _cites(van["answer"], van["sources"]),
                "latency_s": van["latency_s"],
                "n_sources": len(van["sources"]),
                **({"error": van["error"]} if van.get("error") else {}),
            },
            "vergil": {
                "answer_nonempty": bool(ver_answer.strip()),
                "cites_sources": _cites(ver_answer, ver_names)
                                 or _cites(ver_answer, [str(p) for p in paths]),
                "latency_s": ver_latency,
                "n_sources": len(ver_names),
                **({"error": res["error"]} if res.get("error") else {}),
            },
        }
        if qtype == "multi_hop":
            row["vergil"]["used_graph"] = bool(paths)
        per_query.append(row)

        side_by_side.append({
            "query": query,
            "type": qtype,
            "route": row["route"],
            "vanilla_answer": van["answer"],
            "vanilla_sources": van["sources"][:10],
            "vergil_answer": ver_answer,
            "vergil_sources": ver_names[:10],
            "vergil_paths": [str(p) for p in paths][:10],
        })

    return {
        "per_query": per_query,
        "by_type": _aggregate(per_query),
        "side_by_side": side_by_side,
    }


def _aggregate(per_query: list[dict]) -> dict:
    """Per-query-type aggregates for both systems."""
    by_type: dict = {}
    for row in per_query:
        agg = by_type.setdefault(row["type"], {
            "n": 0,
            "vanilla": {"nonempty": 0, "cites": 0, "latency_s": 0.0},
            "vergil": {"nonempty": 0, "cites": 0, "latency_s": 0.0, "used_graph": 0},
            "routes": {},
        })
        agg["n"] += 1
        agg["routes"][row["route"]] = agg["routes"].get(row["route"], 0) + 1
        for side in ("vanilla", "vergil"):
            agg[side]["nonempty"] += int(row[side]["answer_nonempty"])
            agg[side]["cites"] += int(row[side]["cites_sources"])
            agg[side]["latency_s"] += row[side]["latency_s"]
        if "used_graph" in row["vergil"]:
            agg["vergil"]["used_graph"] += int(row["vergil"]["used_graph"])

    for qtype, agg in by_type.items():
        n = max(agg["n"], 1)
        for side in ("vanilla", "vergil"):
            agg[side]["nonempty_rate"] = round(agg[side].pop("nonempty") / n, 3)
            agg[side]["cites_rate"] = round(agg[side].pop("cites") / n, 3)
            agg[side]["avg_latency_s"] = round(agg[side].pop("latency_s") / n, 2)
        if qtype == "multi_hop":
            agg["vergil"]["used_graph_rate"] = round(agg["vergil"].pop("used_graph") / n, 3)
        else:
            agg["vergil"].pop("used_graph", None)
    return by_type


def format_ablation_table(results: dict) -> str:
    """Printable per-type table from run_rag_ablation()'s return dict."""
    by_type = results["by_type"]
    header = (
        f"{'query type':<12} {'n':>2} | {'van nonempty':>12} {'van cites':>9} {'van lat(s)':>10} "
        f"| {'vgl nonempty':>12} {'vgl cites':>9} {'vgl lat(s)':>10} {'used_graph':>10} | routes"
    )
    lines = [header, "-" * len(header)]
    for qtype in ("local", "global", "multi_hop"):
        if qtype not in by_type:
            continue
        a = by_type[qtype]
        van, ver = a["vanilla"], a["vergil"]
        ug = f"{ver['used_graph_rate']:.2f}" if "used_graph_rate" in ver else "-"
        routes = ",".join(f"{k}:{v}" for k, v in sorted(a["routes"].items()))
        lines.append(
            f"{qtype:<12} {a['n']:>2} | {van['nonempty_rate']:>12.2f} {van['cites_rate']:>9.2f} "
            f"{van['avg_latency_s']:>10.2f} | {ver['nonempty_rate']:>12.2f} {ver['cites_rate']:>9.2f} "
            f"{ver['avg_latency_s']:>10.2f} {ug:>10} | {routes}"
        )
    lines.append("")
    lines.append("(cites/nonempty/used_graph are mechanical checks; grade answer quality "
                 "manually from results['side_by_side'] — no LLM-judge.)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# graph statistics (README + rag_eval stage GRAPH STATS block)
# ---------------------------------------------------------------------------

def graph_statistics(G, communities: dict | None = None) -> dict:
    """Node/edge counts by type, community counts, top-10 brands by degree,
    average product degree. ``communities`` is the detect_communities() dict
    ({"level_0": [...], "level_1": [...]}), optional."""
    from collections import Counter

    node_types = Counter(d.get("type", "?") for _, d in G.nodes(data=True))
    edge_types = Counter(d.get("type", "?") for _, _, d in G.edges(data=True))

    brand_degrees = sorted(
        ((G.nodes[n].get("name", n), G.degree(n))
         for n, d in G.nodes(data=True) if d.get("type") == "brand"),
        key=lambda x: -x[1],
    )
    product_degrees = [G.degree(n) for n, d in G.nodes(data=True) if d.get("type") == "product"]

    stats = {
        "nodes_total": G.number_of_nodes(),
        "edges_total": G.number_of_edges(),
        "nodes_by_type": dict(sorted(node_types.items())),
        "edges_by_type": dict(sorted(edge_types.items())),
        "top_10_brands_by_degree": brand_degrees[:10],
        "avg_product_degree": round(sum(product_degrees) / max(len(product_degrees), 1), 2),
    }
    if communities:
        stats["communities_l0"] = len(communities.get("level_0", []))
        stats["communities_l1"] = len(communities.get("level_1", []))
    return stats


def format_graph_stats(stats: dict) -> str:
    """Printable GRAPH STATS block."""
    lines = ["=" * 60, "GRAPH STATS", "=" * 60,
             f"nodes: {stats['nodes_total']:,}   edges: {stats['edges_total']:,}",
             f"nodes by type: {stats['nodes_by_type']}",
             f"edges by type: {stats['edges_by_type']}"]
    if "communities_l0" in stats:
        lines.append(f"communities: L0={stats['communities_l0']}  L1={stats['communities_l1']}")
    lines.append(f"avg product degree: {stats['avg_product_degree']}")
    lines.append("top-10 brands by degree:")
    for name, deg in stats["top_10_brands_by_degree"]:
        lines.append(f"  {deg:>6,}  {name}")
    lines.append("=" * 60)
    return "\n".join(lines)
