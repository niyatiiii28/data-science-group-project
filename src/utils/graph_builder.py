"""Build hierarchical product graph using NetworkX."""

import json
import logging
from pathlib import Path

import networkx as nx
import pandas as pd

from config.constants import HIERARCHY_DECAY_FACTOR

logger = logging.getLogger("semantic_search")


def build_hierarchy_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed acyclic graph: ROOT → Department → Category → Subcategory → ProductType → Product.

    Each node stores: level, name, count(products under it).
    Each product leaf stores: product_id, product_name, and row index.
    """
    G = nx.DiGraph()
    G.add_node("ROOT", level=0, name="ROOT", count=len(df))

    for idx, row in df.iterrows():
        dept = str(row.get("department", "Other")).strip() or "Other"
        cat = str(row.get("category", "General")).strip() or "General"
        subcat = str(row.get("subcategory", "General")).strip() or "General"
        ptype = str(row.get("product_type", "General")).strip() or "General"
        pid = str(row.get("product_id", f"P{idx}"))

        # Department node
        dept_node = f"dept:{dept}"
        if not G.has_node(dept_node):
            G.add_node(dept_node, level=1, name=dept, count=0)
            G.add_edge("ROOT", dept_node)
        G.nodes[dept_node]["count"] = G.nodes[dept_node].get("count", 0) + 1

        # Category node
        cat_node = f"cat:{dept}:{cat}"
        if not G.has_node(cat_node):
            G.add_node(cat_node, level=2, name=cat, count=0)
            G.add_edge(dept_node, cat_node)
        G.nodes[cat_node]["count"] = G.nodes[cat_node].get("count", 0) + 1

        # Subcategory node
        subcat_node = f"subcat:{dept}:{cat}:{subcat}"
        if not G.has_node(subcat_node):
            G.add_node(subcat_node, level=3, name=subcat, count=0)
            G.add_edge(cat_node, subcat_node)
        G.nodes[subcat_node]["count"] = G.nodes[subcat_node].get("count", 0) + 1

        # ProductType node
        ptype_node = f"ptype:{dept}:{cat}:{subcat}:{ptype}"
        if not G.has_node(ptype_node):
            G.add_node(ptype_node, level=4, name=ptype, count=0)
            G.add_edge(subcat_node, ptype_node)
        G.nodes[ptype_node]["count"] = G.nodes[ptype_node].get("count", 0) + 1

        # Product leaf
        prod_node = f"prod:{pid}"
        G.add_node(
            prod_node,
            level=5,
            name=str(row.get("product_name", "")),
            product_id=pid,
            row_index=idx,
            count=1,
        )
        G.add_edge(ptype_node, prod_node)

    stats = {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "departments": sum(1 for n, d in G.nodes(data=True) if d.get("level") == 1),
        "categories": sum(1 for n, d in G.nodes(data=True) if d.get("level") == 2),
        "subcategories": sum(1 for n, d in G.nodes(data=True) if d.get("level") == 3),
        "product_types": sum(1 for n, d in G.nodes(data=True) if d.get("level") == 4),
        "products": sum(1 for n, d in G.nodes(data=True) if d.get("level") == 5),
    }
    logger.info("Hierarchy graph built: %s", stats)
    return G


def get_nodes_at_level(G: nx.DiGraph, level: int) -> list[str]:
    """Get all node IDs at a given hierarchy level."""
    return [n for n, d in G.nodes(data=True) if d.get("level") == level]


def get_children(G: nx.DiGraph, node: str) -> list[str]:
    """Get direct children of a node."""
    return list(G.successors(node))


def get_leaf_products(G: nx.DiGraph, node: str) -> list[str]:
    """Get all product leaf nodes under a given node (BFS)."""
    leaves = []
    for descendant in nx.descendants(G, node):
        if G.nodes[descendant].get("level") == 5:
            leaves.append(descendant)
    return leaves


def export_hierarchy_json(G: nx.DiGraph, filepath: str | Path) -> None:
    """Export the hierarchy as an inspectable JSON tree."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    def _build_tree(node: str) -> dict:
        data = G.nodes[node]
        tree = {
            "name": data.get("name", node),
            "level": data.get("level", 0),
            "count": data.get("count", 0),
        }
        children = get_children(G, node)
        if children and data.get("level", 0) < 4:  # Don't expand individual products
            tree["children"] = [_build_tree(c) for c in children]
        elif data.get("level", 0) == 4:
            tree["products"] = len(children)
        return tree

    tree = _build_tree("ROOT")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    logger.info("Exported hierarchy JSON → %s", filepath)


def save_graph(G: nx.DiGraph, filepath: str | Path) -> None:
    import pickle
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(G, f)
    logger.info("Saved hierarchy graph → %s", filepath)


def load_graph(filepath: str | Path) -> nx.DiGraph:
    import pickle
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Graph file not found: {filepath}")
    with open(filepath, "rb") as f:
        G = pickle.load(f)
    logger.info("Loaded hierarchy graph from %s", filepath)
    return G
