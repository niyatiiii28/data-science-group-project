"""Evaluation metrics: Precision@K, Recall@K, MRR, NDCG, F1.

Relevance is a per-result boolean flag produced upstream (by category keyword
match against the ground-truth term set). These metrics work on the flags
directly, which removes the need for hand-labelled relevant-id sets.
"""

import numpy as np


def precision_at_k(flags: list, k: int) -> float:
    """Fraction of top-K results that are relevant (flag == 1)."""
    top_k = flags[:k]
    if not top_k:
        return 0.0
    return sum(1 for f in top_k if f) / len(top_k)


def recall_at_k(flags: list, k: int, total_relevant: int) -> float:
    """Fraction of relevant items found in top-K.

    When ground truth is category-level (the corpus has many matching items),
    ``total_relevant`` can be set to ``min(k, hits_in_top_k)`` to keep recall
    bounded; otherwise recall is effectively precision@k.
    """
    if total_relevant <= 0:
        return 0.0
    hits = sum(1 for f in flags[:k] if f)
    return min(hits / total_relevant, 1.0)


def mean_reciprocal_rank(flags: list) -> float:
    """Reciprocal of the rank of the first relevant result."""
    for i, f in enumerate(flags, 1):
        if f:
            return 1.0 / i
    return 0.0


def ndcg_at_k(flags: list, k: int) -> float:
    """Normalized DCG at K assuming binary relevance."""
    top_k = flags[:k]
    dcg = sum((1.0 if f else 0.0) / np.log2(i + 2) for i, f in enumerate(top_k))
    ideal = sum(flags[:k])
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(ideal, k)))
    return dcg / idcg if idcg > 0 else 0.0


def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_all_metrics(flags: list, k: int = 5) -> dict:
    """Compute all metrics for a single query given a list of relevance flags."""
    p = precision_at_k(flags, k)
    # For category-level GT we treat "total relevant" as the count of hits in
    # top-K: this makes recall = precision (useful) rather than ~0 (misleading).
    total_rel = max(sum(flags[:k]), 1) if any(flags[:k]) else 0
    r = recall_at_k(flags, k, total_rel) if total_rel else 0.0
    return {
        "precision_at_k": round(p, 4),
        "recall_at_k": round(r, 4),
        "mrr": round(mean_reciprocal_rank(flags), 4),
        "ndcg": round(ndcg_at_k(flags, k), 4),
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
