# data/download_amazon.py


def download_amazon_metadata(category: str = "Electronics", output_dir: str = "data/raw"):
    """
    Download Amazon Reviews 2023 metadata for one category.
    HuggingFace: McAuley-Lab/Amazon-Reviews-2023

    Metadata includes:
    - main_category, title, description, features (bullet points)
    - price, average_rating, rating_number
    - store (brand/seller)
    - categories (list of category paths)
    - details (dict with brand, model, dimensions, etc.)
    - bought_together (list of ASINs) ← THIS IS THE GOLD for graph edges
    - also_buy, also_view (related products)

    Filter: keep products with title AND at least one of:
    bought_together, also_buy, brand, categories. Drop products with empty/null titles.
    Save metadata as parquet. ALSO download a small sample of reviews (3 per product)
    for description enrichment and save as parquet.
    """
    raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §3.2")
