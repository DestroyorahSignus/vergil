# graph/schema.py
from enum import Enum
from dataclasses import dataclass


class NodeType(Enum):
    PRODUCT = "product"
    BRAND = "brand"
    CATEGORY = "category"
    FEATURE = "feature"       # extracted from bullet points/descriptions


class EdgeType(Enum):
    BOUGHT_TOGETHER = "bought_together"    # from Amazon metadata
    ALSO_BOUGHT = "also_bought"            # from Amazon metadata
    ALSO_VIEWED = "also_viewed"            # from Amazon metadata
    HAS_BRAND = "has_brand"                # product → brand
    IN_CATEGORY = "in_category"            # product → category
    HAS_FEATURE = "has_feature"            # product → feature (e.g., "wireless", "noise-cancelling")
    SIMILAR_TO = "similar_to"              # dense embedding similarity > threshold (computed)
    CATEGORY_PARENT = "category_parent"    # category hierarchy


@dataclass
class Node:
    id: str
    type: NodeType
    name: str
    description: str = ""
    metadata: dict = None  # price, rating, etc.


@dataclass
class Edge:
    source: str
    target: str
    type: EdgeType
    weight: float = 1.0
