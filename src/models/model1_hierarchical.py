"""Model 1: Hierarchical Graph-Based Search with score decay."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from src.models.base import BaseSearchModel
from src.core.schemas import SearchResult
from src.utils.embeddings import encode_texts, encode_single, save_embeddings, load_embeddings
from src.utils.graph_builder import (
    build_hierarchy_graph,
    get_nodes_at_level,
    get_children,
    get_leaf_products,
    export_hierarchy_json,
    save_graph,
    load_graph,
)
from config.constants import HIERARCHY_DECAY_FACTOR

logger = logging.getLogger("semantic_search")


class HierarchicalSearchModel(BaseSearchModel):
    """Tree-based traversal with embedding similarity at each hierarchy level."""

    model_id = "model1_hierarchical"

    def __init__(self):
        self.graph: Optional[nx.DiGraph] = None
        self.df: Optional[pd.DataFrame] = None
        self.node_embeddings: dict[str, np.ndarray] = {}
        self._ready = False

    def build_index(self, df: pd.DataFrame) -> None:
        logger.info("[Model 1] Building hierarchical index from %d products", len(df))
        self.df = df.reset_index(drop=True)

        # Build graph
        self.graph = build_hierarchy_graph(df)

        # Generate embeddings for non-product nodes (levels 1-4)
        self._build_node_embeddings()

        # Generate embeddings for product nodes (level 5)
        self._build_product_embeddings()

        self._ready = True
        logger.info("[Model 1] Index built: %d node embeddings", len(self.node_embeddings))

    def _build_node_embeddings(self) -> None:
        """Embed hierarchy names at levels 1-4."""
        nodes = []
        texts = []
        for node, data in self.graph.nodes(data=True):
            level = data.get("level", 0)
            if 1 <= level <= 4:
                nodes.append(node)
                texts.append(data.get("name", ""))

        if texts:
            embeddings = encode_texts(texts, show_progress=False)
            for node, emb in zip(nodes, embeddings):
                self.node_embeddings[node] = emb

    def _build_product_embeddings(self) -> None:
        """Embed product names + descriptions at level 5."""
        prod_nodes = get_nodes_at_level(self.graph, 5)
        texts = []
        for pn in prod_nodes:
            data = self.graph.nodes[pn]
            row_idx = data.get("row_index")
            if row_idx is not None and row_idx < len(self.df):
                row = self.df.iloc[row_idx]
                text = f"{row['product_name']}. {row.get('category', '')}. {row.get('features', '')}"
            else:
                text = data.get("name", "")
            texts.append(text)

        if texts:
            embeddings = encode_texts(texts, show_progress=True)
            for node, emb in zip(prod_nodes, embeddings):
                self.node_embeddings[node] = emb

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self._ready:
            raise RuntimeError("Model 1 not initialized. Call build_index() first.")

        query_emb = encode_single(query)
        results = self._traverse_hierarchy(query_emb, top_k)
        return results

    def _traverse_hierarchy(self, query_emb: np.ndarray, top_k: int) -> list[SearchResult]:
        """Navigate hierarchy top-down, scoring at each level."""
        decay = HIERARCHY_DECAY_FACTOR
        candidate_scores: dict[str, float] = {}  # prod_node → cumulative score

        # Score departments (level 1)
        dept_nodes = get_nodes_at_level(self.graph, 1)
        dept_scores = self._score_nodes(query_emb, dept_nodes)
        top_depts = sorted(dept_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        for dept_node, dept_score in top_depts:
            # Score categories under this department (level 2)
            cat_nodes = get_children(self.graph, dept_node)
            cat_scores = self._score_nodes(query_emb, cat_nodes)
            top_cats = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)[:5]

            for cat_node, cat_score in top_cats:
                # Score subcategories (level 3)
                subcat_nodes = get_children(self.graph, cat_node)
                subcat_scores = self._score_nodes(query_emb, subcat_nodes)
                top_subcats = sorted(subcat_scores.items(), key=lambda x: x[1], reverse=True)[:5]

                for subcat_node, subcat_score in top_subcats:
                    # Score product types (level 4)
                    ptype_nodes = get_children(self.graph, subcat_node)
                    ptype_scores = self._score_nodes(query_emb, ptype_nodes)
                    top_ptypes = sorted(ptype_scores.items(), key=lambda x: x[1], reverse=True)[:3]

                    for ptype_node, ptype_score in top_ptypes:
                        # Score products (level 5)
                        prod_nodes = get_children(self.graph, ptype_node)
                        prod_scores = self._score_nodes(query_emb, prod_nodes)

                        for prod_node, prod_score in prod_scores.items():
                            # Hierarchical score with decay
                            hier_score = (
                                dept_score * (decay ** 4)
                                + cat_score * (decay ** 3)
                                + subcat_score * (decay ** 2)
                                + ptype_score * decay
                                + prod_score
                            ) / 5.0

                            if prod_node not in candidate_scores or hier_score > candidate_scores[prod_node]:
                                candidate_scores[prod_node] = hier_score

        # Sort and return top-K
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for rank, (prod_node, score) in enumerate(sorted_candidates[:top_k], 1):
            data = self.graph.nodes[prod_node]
            row_idx = data.get("row_index")
            if row_idx is not None and row_idx < len(self.df):
                row = self.df.iloc[row_idx]
                results.append(SearchResult(
                    product_id=str(row["product_id"]),
                    product_name=str(row["product_name"]),
                    department=str(row.get("department", "")),
                    category=str(row.get("category", "")),
                    brand=str(row.get("brand", "")),
                    price=float(row.get("price", 0)),
                    rating=float(row.get("rating", 0)),
                    reviews_count=int(row.get("reviews_count", 0)),
                    description=str(row.get("description", ""))[:200],
                    confidence_score=round(min(max(score, 0.0), 1.0), 4),
                    rank=rank,
                ))
        return results

    def _score_nodes(self, query_emb: np.ndarray, nodes: list[str]) -> dict[str, float]:
        """Compute cosine similarity between query and a list of nodes."""
        scores = {}
        for node in nodes:
            if node in self.node_embeddings:
                sim = cosine_similarity(
                    query_emb.reshape(1, -1),
                    self.node_embeddings[node].reshape(1, -1),
                )[0][0]
                scores[node] = float(max(sim, 0.0))
        return scores

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_graph(self.graph, directory / "product_hierarchy.pkl")
        with open(directory / "node_embeddings.pkl", "wb") as f:
            pickle.dump(self.node_embeddings, f)
        logger.info("[Model 1] Saved to %s", directory)

    def load(self, directory: str | Path) -> None:
        directory = Path(directory)
        self.graph = load_graph(directory / "product_hierarchy.pkl")
        with open(directory / "node_embeddings.pkl", "rb") as f:
            self.node_embeddings = pickle.load(f)
        self._ready = True
        logger.info("[Model 1] Loaded from %s", directory)

    @property
    def is_ready(self) -> bool:
        return self._ready
