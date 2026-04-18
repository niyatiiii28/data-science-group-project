"""Abstract base class for all search models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd

from src.core.schemas import SearchResult


class BaseSearchModel(ABC):
    """Interface contract that all 4 models must implement."""

    model_id: str = "base"

    @abstractmethod
    def build_index(self, df: pd.DataFrame) -> None:
        """Build the search index from processed product data."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Execute a search and return ranked results."""

    @abstractmethod
    def save(self, directory: str | Path) -> None:
        """Persist model artifacts to disk."""

    @abstractmethod
    def load(self, directory: str | Path) -> None:
        """Load model artifacts from disk."""

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Whether the model is loaded and ready to serve queries."""
