"""SentenceTransformer embedding service with batching and caching."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import settings
from config.constants import EMBEDDING_BATCH_SIZE
from src.core.exceptions import EmbeddingError

logger = logging.getLogger("semantic_search")

_model_instance = None


def get_embedding_model():
    """Lazy-load the SentenceTransformer model (singleton)."""
    global _model_instance
    if _model_instance is None:
        from sentence_transformers import SentenceTransformer

        model_name = settings.embedding_model_name
        logger.info("Loading embedding model: %s", model_name)
        _model_instance = SentenceTransformer(model_name)

        # Auto-detect GPU
        device = _model_instance.device
        logger.info("Embedding model loaded on device: %s", device)
    return _model_instance


def encode_texts(
    texts: list[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
    normalize: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode a list of texts into embedding vectors.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim) as float32
    """
    if not texts:
        return np.array([], dtype=np.float32).reshape(0, settings.embedding_dimension)

    model = get_embedding_model()
    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        embeddings = embeddings.astype(np.float32)
        logger.info("Encoded %d texts → shape %s", len(texts), embeddings.shape)
        return embeddings
    except Exception as e:
        raise EmbeddingError(f"Failed to encode {len(texts)} texts: {e}") from e


def encode_single(text: str, normalize: bool = True) -> np.ndarray:
    """Encode a single text string. Returns shape (embedding_dim,)."""
    result = encode_texts([text], batch_size=1, normalize=normalize, show_progress=False)
    return result[0]


def save_embeddings(embeddings: np.ndarray, filepath: str | Path) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, embeddings)
    logger.info("Saved embeddings: shape %s → %s", embeddings.shape, filepath)


def load_embeddings(filepath: str | Path) -> np.ndarray:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Embeddings file not found: {filepath}")
    embeddings = np.load(filepath)
    logger.info("Loaded embeddings: shape %s from %s", embeddings.shape, filepath)
    return embeddings
