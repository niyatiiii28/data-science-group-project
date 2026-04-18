"""Unit tests for evaluation metrics."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    f1_score,
    compute_all_metrics,
)


class TestPrecisionAtK:
    def test_perfect(self):
        assert precision_at_k({"A", "B"}, ["A", "B", "C"], 2) == 1.0

    def test_none_relevant(self):
        assert precision_at_k({"A"}, ["B", "C", "D"], 3) == 0.0

    def test_partial(self):
        assert precision_at_k({"A", "B"}, ["A", "C", "B"], 3) == 2 / 3

    def test_empty_retrieved(self):
        assert precision_at_k({"A"}, [], 5) == 0.0


class TestRecallAtK:
    def test_perfect(self):
        assert recall_at_k({"A", "B"}, ["A", "B", "C"], 3) == 1.0

    def test_partial(self):
        assert recall_at_k({"A", "B", "C"}, ["A", "D"], 2) == 1 / 3

    def test_empty_relevant(self):
        assert recall_at_k(set(), ["A", "B"], 2) == 0.0


class TestMRR:
    def test_first_position(self):
        assert mean_reciprocal_rank({"A"}, ["A", "B", "C"]) == 1.0

    def test_second_position(self):
        assert mean_reciprocal_rank({"B"}, ["A", "B", "C"]) == 0.5

    def test_not_found(self):
        assert mean_reciprocal_rank({"D"}, ["A", "B", "C"]) == 0.0


class TestNDCG:
    def test_perfect_ordering(self):
        score = ndcg_at_k({"A", "B"}, ["A", "B", "C"], 3)
        assert score == 1.0

    def test_zero_relevance(self):
        score = ndcg_at_k({"D"}, ["A", "B", "C"], 3)
        assert score == 0.0


class TestF1:
    def test_perfect(self):
        assert f1_score(1.0, 1.0) == 1.0

    def test_zero(self):
        assert f1_score(0.0, 0.0) == 0.0

    def test_balanced(self):
        assert round(f1_score(0.8, 0.6), 4) == round(2 * 0.8 * 0.6 / (0.8 + 0.6), 4)


class TestComputeAll:
    def test_returns_all_keys(self):
        result = compute_all_metrics({"A"}, ["A", "B"], k=2)
        assert "precision_at_k" in result
        assert "recall_at_k" in result
        assert "mrr" in result
        assert "ndcg" in result
        assert "f1_score" in result
