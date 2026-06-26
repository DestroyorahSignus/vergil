# generation/prompts.py

LOCAL_QA_PROMPT = """You are a helpful e-commerce product assistant. Answer the user's question
using ONLY the provided product information and knowledge graph context. If you cannot answer
from the context, say so.

## Retrieved products:
{product_context}

## Knowledge graph relationships:
{graph_context}

## User question:
{query}

Provide a clear, helpful answer. Cite specific products by name when referencing them.
If comparing products, use a structured format.

Answer:"""

GLOBAL_QA_PROMPT = """You are a helpful e-commerce analyst. Answer the user's question using
the provided community summaries from the product knowledge graph.

## Relevant product clusters:
{community_summaries}

## User question:
{query}

Provide a comprehensive answer based on the aggregated knowledge. Reference specific clusters
and brands when relevant.

Answer:"""

MULTI_HOP_QA_PROMPT = """You are a helpful product discovery assistant. The user asked a question
that required traversing the product knowledge graph. Here are the products discovered through
graph traversal, with the reasoning paths that connect them to the user's query.

## Query entities found in graph:
{source_entities}

## Discovered products (via graph traversal):
{discovered_products}

## Traversal paths:
{paths}

## User question:
{query}

Explain how these products relate to the user's query, using the graph paths as reasoning chains.
Recommend the most relevant products and explain why.

Answer:"""

ENTITY_EXTRACTION_PROMPT = """Extract product-related entities from this e-commerce query.
Return ONLY a JSON list of entities (brand names, product names, features, categories).

Query: "{query}"

Entities (JSON list):"""
