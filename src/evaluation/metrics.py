"""Evaluation metrics: Precision@K, Recall@K, MRR, NDCG, F1 — vectorized with NumPy."""

import numpy as np


def precision_at_k(relevant: set, retrieved: list[str], k: int) -> float:
    """Fraction of top-K results that are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(top_k)


def recall_at_k(relevant: set, retrieved: list[str], k: int) -> float:
    """Fraction of relevant items found in top-K."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(relevant: set, retrieved: list[str]) -> float:
    """Reciprocal of the rank of the first relevant result."""
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant: set, retrieved: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    top_k = retrieved[:k]
    dcg = 0.0
    for i, r in enumerate(top_k):
        rel = 1.0 if r in relevant else 0.0
        dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG: all relevant items at top
    ideal_rels = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_all_metrics(
    relevant: set, retrieved: list[str], k: int = 5
) -> dict[str, float]:
    """Compute all metrics for a single query."""
    p = precision_at_k(relevant, retrieved, k)
    r = recall_at_k(relevant, retrieved, k)
    return {
        "precision_at_k": round(p, 4),
        "recall_at_k": round(r, 4),
        "mrr": round(mean_reciprocal_rank(relevant, retrieved), 4),
        "ndcg": round(ndcg_at_k(relevant, retrieved, k), 4),
        "f1_score": round(f1_score(p, r), 4),
    }


def average_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    """Average metrics across multiple queries."""
    if not metrics_list:
        return {"precision_at_k": 0, "recall_at_k": 0, "mrr": 0, "ndcg": 0, "f1_score": 0}

    keys = metrics_list[0].keys()
    return {
        key: round(np.mean([m[key] for m in metrics_list]), 4)
        for key in keys
    }
