"""FAISS index management: build, save, load, search."""

import logging
from pathlib import Path

import numpy as np
import faiss

from config.settings import settings
from config.constants import FAISS_IVF_THRESHOLD
from src.core.exceptions import IndexNotBuiltError

logger = logging.getLogger("semantic_search")


class FAISSIndex:
    """Manages a FAISS index for vector similarity search."""

    def __init__(self, dimension: int = None):
        self.dimension = dimension or settings.embedding_dimension
        self.index: faiss.Index | None = None
        self.is_trained = False

    def build(self, embeddings: np.ndarray, use_ivf: bool | None = None) -> None:
        """Build index from embedding matrix.

        Args:
            embeddings: (N, D) float32 array
            use_ivf: Force IVF if True, FlatL2 if False, auto if None
        """
        n_vectors, dim = embeddings.shape
        self.dimension = dim
        embeddings = embeddings.astype(np.float32)

        if use_ivf is None:
            use_ivf = n_vectors >= FAISS_IVF_THRESHOLD

        if use_ivf:
            nlist = min(settings.faiss_nlist, n_vectors // 10)
            nlist = max(nlist, 1)
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.index.train(embeddings)
            self.index.nprobe = settings.faiss_nprobe
            logger.info("Built IVFFlat index: %d vectors, nlist=%d", n_vectors, nlist)
        else:
            self.index = faiss.IndexFlatL2(dim)
            logger.info("Built FlatL2 index: %d vectors", n_vectors)

        self.index.add(embeddings)
        self.is_trained = True
        logger.info("Index contains %d vectors", self.index.ntotal)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Search the index.

        Args:
            query_embedding: (D,) or (1, D) float32 array
            top_k: number of nearest neighbors

        Returns:
            (distances, indices) each of shape (1, top_k)
        """
        if self.index is None:
            raise IndexNotBuiltError("FAISS index has not been built. Call build() first.")

        query = query_embedding.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        top_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query, top_k)
        return distances, indices

    def save(self, filepath: str | Path) -> None:
        if self.index is None:
            raise IndexNotBuiltError("Cannot save — index not built.")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(filepath))
        logger.info("Saved FAISS index (%d vectors) → %s", self.index.ntotal, filepath)

    def load(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"FAISS index not found: {filepath}")
        self.index = faiss.read_index(str(filepath))
        self.dimension = self.index.d
        self.is_trained = True
        logger.info("Loaded FAISS index: %d vectors from %s", self.index.ntotal, filepath)

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal if self.index else 0
