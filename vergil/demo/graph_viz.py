# demo/graph_viz.py
"""Pyvis interactive visualization of a relevant subgraph.

pyvis is imported lazily inside visualize_subgraph so this module (and the whole
vergil package) imports fine on machines without pyvis installed.
"""
import networkx as nx

# Node colors by type (plan §9: product=blue, brand=green, category=orange, feature=purple)
TYPE_COLORS = {
    "product": "#4a90d9",   # blue
    "brand": "#3cb371",     # green
    "category": "#f5a623",  # orange
    "feature": "#9b59b6",   # purple
}
_DEFAULT_COLOR = "#95a5a6"  # grey for anything unexpected

MAX_NODES = 150  # keep the pyvis physics sim responsive in the browser


def visualize_subgraph(G: nx.Graph, nodes: list[str], height: str = "400px") -> str:
    """Render the subgraph induced by `nodes` as an interactive pyvis HTML string.

    Nodes are colored by type; edges are labeled with their edge type on hover.
    Unknown node ids are ignored. Returns a full standalone HTML document string
    (drop it into a gradio `gr.HTML` component or an iframe).
    """
    try:
        from pyvis.network import Network
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "visualize_subgraph requires pyvis (pip install pyvis>=0.3.2)"
        ) from e

    present = [n for n in nodes if n in G.nodes][:MAX_NODES]
    sub = G.subgraph(present)

    net = Network(height=height, width="100%", notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)

    for n in sub.nodes:
        d = G.nodes[n]
        ntype = d.get("type", "?")
        name = str(d.get("name") or n)
        net.add_node(
            n,
            label=name[:40],
            title=f"[{ntype}] {name}",
            color=TYPE_COLORS.get(ntype, _DEFAULT_COLOR),
            size=18 if ntype == "product" else 12,
        )
    for u, v, d in sub.edges(data=True):
        net.add_edge(u, v, title=str(d.get("type", "related")), color="#c0c0c0")

    # generate_html returns the standalone document (pyvis >= 0.3) — no temp file.
    return net.generate_html()


def query_subgraph_nodes(G: nx.Graph, entities: list[str], max_seeds: int = 5,
                         max_neighbors: int = 15) -> list[str]:
    """Find graph nodes matching the query entities (case-insensitive substring on
    node names), plus a capped set of 1-hop neighbors — the node set the demo
    visualizes for a query. Pure networkx, no pyvis needed."""
    seeds = []
    wanted = [e.lower() for e in entities if e and len(str(e)) >= 3]
    if wanted:
        for n, d in G.nodes(data=True):
            name = str(d.get("name") or "").lower()
            if name and any(w in name for w in wanted):
                seeds.append(n)
                if len(seeds) >= max_seeds:
                    break

    nodes = list(seeds)
    for s in seeds:
        for i, nb in enumerate(G.neighbors(s)):
            if i >= max_neighbors:
                break
            nodes.append(nb)
    # de-dupe, preserve order
    seen, out = set(), []
    for n in nodes:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out
