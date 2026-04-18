"""Unit tests for FAISS index manager."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest
from src.utils.faiss_index import FAISSIndex
from src.core.exceptions import IndexNotBuiltError


class TestFAISSIndex:
    def test_build_flat(self, sample_embeddings):
        idx = FAISSIndex(dimension=384)
        idx.build(sample_embeddings, use_ivf=False)
        assert idx.total_vectors == 10
        assert idx.is_trained

    def test_search_returns_correct_shape(self, sample_embeddings):
        idx = FAISSIndex(dimension=384)
        idx.build(sample_embeddings, use_ivf=False)
        query = sample_embeddings[0]
        distances, indices = idx.search(query, top_k=3)
        assert distances.shape == (1, 3)
        assert indices.shape == (1, 3)

    def test_search_top1_is_itself(self, sample_embeddings):
        idx = FAISSIndex(dimension=384)
        idx.build(sample_embeddings, use_ivf=False)
        query = sample_embeddings[3]
        _, indices = idx.search(query, top_k=1)
        assert indices[0][0] == 3

    def test_search_without_build_raises(self):
        idx = FAISSIndex(dimension=384)
        with pytest.raises(IndexNotBuiltError):
            idx.search(np.zeros(384, dtype=np.float32), top_k=1)

    def test_save_and_load(self, sample_embeddings, tmp_path):
        idx = FAISSIndex(dimension=384)
        idx.build(sample_embeddings, use_ivf=False)
        filepath = tmp_path / "test.faiss"
        idx.save(filepath)

        idx2 = FAISSIndex()
        idx2.load(filepath)
        assert idx2.total_vectors == 10
        assert idx2.is_trained
