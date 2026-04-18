"""Model 2: Flat Semantic Search using FAISS with basic product descriptions."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.models.base import BaseSearchModel
from src.core.schemas import SearchResult
from src.utils.embeddings import encode_texts, encode_single, save_embeddings, load_embeddings
from src.utils.faiss_index import FAISSIndex
from src.utils.text_processing import extract_combined_text

logger = logging.getLogger("semantic_search")


class FlatSemanticSearchModel(BaseSearchModel):
    """Basic semantic search: encode product fields → FAISS → top-K."""

    model_id = "model2_flat_semantic"

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.faiss_index = FAISSIndex()
        self.embeddings: Optional[np.ndarray] = None
        self._ready = False

    def build_index(self, df: pd.DataFrame) -> None:
        logger.info("[Model 2] Building flat semantic index from %d products", len(df))
        self.df = df.reset_index(drop=True)

        # Create combined descriptions
        descriptions = self._create_descriptions()

        # Generate embeddings
        self.embeddings = encode_texts(descriptions, show_progress=True)
        logger.info("[Model 2] Embeddings shape: %s", self.embeddings.shape)

        # Build FAISS index
        self.faiss_index.build(self.embeddings)

        self._ready = True
        logger.info("[Model 2] Index built successfully")

    def _create_descriptions(self) -> list[str]:
        """Combine product fields into single description strings."""
        descriptions = []
        for _, row in self.df.iterrows():
            text = extract_combined_text(row.to_dict())
            descriptions.append(text)
        return descriptions

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self._ready:
            raise RuntimeError("Model 2 not initialized. Call build_index() first.")

        query_emb = encode_single(query)
        distances, indices = self.faiss_index.search(query_emb, top_k)

        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
            if idx < 0 or idx >= len(self.df):
                continue

            # Convert L2 distance to confidence score (0-1)
            confidence = self._distance_to_confidence(dist)

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
                confidence_score=round(confidence, 4),
                rank=rank,
            ))
        return results

    @staticmethod
    def _distance_to_confidence(distance: float) -> float:
        """Convert L2 distance to a 0-1 confidence score.

        For normalized embeddings, L2 distance ranges ~[0, 2].
        confidence = 1 - (distance / 2), clamped to [0, 1].
        """
        confidence = 1.0 - (distance / 2.0)
        return max(0.0, min(1.0, confidence))

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.faiss_index.save(directory / "model2.faiss")
        save_embeddings(self.embeddings, directory / "model2_embeddings.npy")
        with open(directory / "products_cache.pkl", "wb") as f:
            pickle.dump(self.df, f)
        logger.info("[Model 2] Saved to %s", directory)

    def load(self, directory: str | Path) -> None:
        directory = Path(directory)
        self.faiss_index.load(directory / "model2.faiss")
        self.embeddings = load_embeddings(directory / "model2_embeddings.npy")
        with open(directory / "products_cache.pkl", "rb") as f:
            self.df = pickle.load(f)
        self._ready = True
        logger.info("[Model 2] Loaded from %s", directory)

    @property
    def is_ready(self) -> bool:
        return self._ready
