"""Model 3: Enhanced Semantic Search with enriched descriptions + metadata re-ranking."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.models.model2_flat_semantic import FlatSemanticSearchModel
from src.core.schemas import SearchResult
from src.utils.embeddings import encode_texts, encode_single, save_embeddings, load_embeddings
from config.constants import RERANK_WEIGHTS

logger = logging.getLogger("semantic_search")


class EnhancedSemanticSearchModel(FlatSemanticSearchModel):
    """Extends Model 2: uses enriched descriptions + metadata re-ranking."""

    model_id = "model3_enhanced"

    def _create_descriptions(self) -> list[str]:
        """Use enriched descriptions if available, else fall back to combined text."""
        descriptions = []
        for _, row in self.df.iterrows():
            enriched = str(row.get("enriched_description", "")).strip()
            if enriched and len(enriched) > 50:
                # Use enriched description + key metadata
                text = (
                    f"{row.get('product_name', '')}. "
                    f"{enriched}. "
                    f"Category: {row.get('category', '')}. "
                    f"Brand: {row.get('brand', '')}. "
                    f"Tags: {row.get('tags', '')}."
                )
            else:
                # Fallback to Model 2 style
                from src.utils.text_processing import extract_combined_text
                text = extract_combined_text(row.to_dict())
            descriptions.append(text)
        return descriptions

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search with over-fetch + metadata re-ranking."""
        if not self._ready:
            raise RuntimeError("Model 3 not initialized. Call build_index() first.")

        # Over-fetch 5x candidates for re-ranking
        fetch_k = min(top_k * 5, self.faiss_index.total_vectors)
        query_emb = encode_single(query)
        distances, indices = self.faiss_index.search(query_emb, fetch_k)

        # Score candidates with combined semantic + metadata signals
        candidates = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.df):
                continue
            row = self.df.iloc[idx]
            semantic_score = self._distance_to_confidence(dist)
            combined_score = self._compute_reranked_score(semantic_score, row)
            candidates.append((idx, combined_score, semantic_score))

        # Sort by combined score
        candidates.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, combined_score, _) in enumerate(candidates[:top_k], 1):
            row = self.df.iloc[idx]
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
                confidence_score=round(min(max(combined_score, 0.0), 1.0), 4),
                rank=rank,
            ))
        return results

    @staticmethod
    def _compute_reranked_score(semantic_score: float, row: pd.Series) -> float:
        """Combine semantic similarity with metadata signals."""
        w = RERANK_WEIGHTS

        # Normalize rating to 0-1
        rating = float(row.get("rating", 0))
        rating_score = rating / 5.0

        # Normalize reviews (log scale, cap at 1.0)
        reviews = int(row.get("reviews_count", 0))
        reviews_score = min(np.log1p(reviews) / np.log1p(1000), 1.0) if reviews > 0 else 0.0

        # Price relevance: having a price is a positive signal (product is real/available)
        price = float(row.get("price", 0))
        price_score = min(price / 10000, 1.0) if price > 0 else 0.3

        # Stock bonus
        stock = str(row.get("stock_status", "")).lower()
        stock_score = 1.0 if "in stock" in stock else 0.2

        combined = (
            w["semantic_similarity"] * semantic_score
            + w["rating_score"] * rating_score
            + w["reviews_score"] * reviews_score
            + w["price_relevance"] * price_score
            + w["stock_bonus"] * stock_score
        )
        return combined

    def save(self, directory: str | Path) -> None:
        import pickle
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.faiss_index.save(directory / "model3.faiss")
        save_embeddings(self.embeddings, directory / "model3_embeddings.npy")
        with open(directory / "products_cache.pkl", "wb") as f:
            pickle.dump(self.df, f)
        logger.info("[Model 3] Saved to %s", directory)

    def load(self, directory: str | Path) -> None:
        import pickle
        directory = Path(directory)
        self.faiss_index.load(directory / "model3.faiss")
        self.embeddings = load_embeddings(directory / "model3_embeddings.npy")
        with open(directory / "products_cache.pkl", "rb") as f:
            self.df = pickle.load(f)
        self._ready = True
        logger.info("[Model 3] Loaded from %s", directory)
