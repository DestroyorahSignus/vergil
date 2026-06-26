# eval/test_queries.py

TEST_QUERIES = [
    # LOCAL queries (should work with vector RAG too)
    {"query": "best wireless noise cancelling headphones under $300", "type": "local"},
    {"query": "USB-C charger for MacBook Pro", "type": "local"},
    {"query": "4K webcam for streaming", "type": "local"},

    # GLOBAL queries (require community summaries — basic RAG fails here)
    {"query": "What are the major smart home ecosystems and how do they compare?", "type": "global"},
    {"query": "Overview of the portable power bank market by brand", "type": "global"},
    {"query": "What trends exist in wireless audio products?", "type": "global"},

    # MULTI-HOP queries (require graph traversal — basic RAG CANNOT answer these)
    {"query": "What accessories from Sony are compatible with the WH-1000XM5?", "type": "multi_hop"},
    {"query": "Find chargers from the same brand as this laptop that also work with phones", "type": "multi_hop"},
    {"query": "What do people usually buy together with a Canon EOS R camera?", "type": "multi_hop"},
    {"query": "Products from Anker that have USB-C and are under $50", "type": "multi_hop"},

    # COMPARISON queries (require both graph + aggregation)
    {"query": "Compare Sony vs Bose noise cancelling headphones", "type": "global"},
    {"query": "JBL vs Sonos for home speakers — which ecosystem has more accessories?", "type": "multi_hop"},
]
