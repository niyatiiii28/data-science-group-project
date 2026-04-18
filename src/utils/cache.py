"""LRU in-memory + pickle disk cache for embeddings and results."""

import hashlib
import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("semantic_search")

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "_cache"


class DiskCache:
    """Simple pickle-based disk cache keyed by string hashes."""

    def __init__(self, cache_dir: str | Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{h}.pkl"

    def get(self, key: str) -> Optional[Any]:
        path = self._key_to_path(key)
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def set(self, key: str, value: Any) -> None:
        path = self._key_to_path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f)

    def has(self, key: str) -> bool:
        return self._key_to_path(key).exists()

    def clear(self) -> None:
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink()
        logger.info("Disk cache cleared: %s", self.cache_dir)


# Singleton cache instance
disk_cache = DiskCache()
