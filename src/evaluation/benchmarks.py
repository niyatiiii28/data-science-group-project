"""Benchmark runner: evaluates all models on all test query sets."""

import logging
import time
from datetime import datetime

from src.models.orchestrator import SearchOrchestrator
from src.evaluation.metrics import compute_all_metrics, average_metrics
from src.evaluation.test_queries import ALL_QUERY_SETS
from src.core.schemas import EvaluationResult, EvaluationReport

logger = logging.getLogger("semantic_search")


def run_benchmarks(
    orchestrator: SearchOrchestrator,
    top_k: int = 5,
) -> EvaluationReport:
    """Run all loaded models against all test query sets."""
    results = []
    loaded = orchestrator.get_loaded_models()
    logger.info("Running benchmarks on models: %s", loaded)

    for model_id in loaded:
        for query_type, queries in ALL_QUERY_SETS.items():
            metrics_list = []
            total_time = 0.0

            for query_str, relevant_ids in queries:
                start = time.perf_counter()
                try:
                    response = orchestrator.search(query_str, model_id=model_id, top_k=top_k)
                    retrieved_ids = [r.product_id for r in response.results]
                except Exception as e:
                    logger.warning("Model %s failed on query '%s': %s", model_id, query_str, e)
                    retrieved_ids = []

                elapsed = (time.perf_counter() - start) * 1000
                total_time += elapsed

                metrics = compute_all_metrics(relevant_ids, retrieved_ids, k=top_k)
                metrics_list.append(metrics)

            avg = average_metrics(metrics_list)
            avg_time = total_time / len(queries) if queries else 0.0

            result = EvaluationResult(
                model_id=model_id,
                query_type=query_type,
                precision_at_k=avg["precision_at_k"],
                recall_at_k=avg["recall_at_k"],
                mrr=avg["mrr"],
                ndcg=avg["ndcg"],
                f1_score=avg["f1_score"],
                avg_query_time_ms=round(avg_time, 2),
            )
            results.append(result)
            logger.info(
                "[Eval] %s / %s: P@%d=%.3f R@%d=%.3f MRR=%.3f NDCG=%.3f F1=%.3f (%.1fms)",
                model_id, query_type, top_k, avg["precision_at_k"],
                top_k, avg["recall_at_k"], avg["mrr"], avg["ndcg"],
                avg["f1_score"], avg_time,
            )

    report = EvaluationReport(
        timestamp=datetime.now().isoformat(),
        results=results,
    )
    return report


def format_report(report: EvaluationReport) -> str:
    """Format a benchmark report as a readable text table."""
    lines = [
        "=" * 90,
        "EVALUATION REPORT",
        f"Timestamp: {report.timestamp}",
        "=" * 90,
        f"{'Model':<25} {'Type':<10} {'P@K':>6} {'R@K':>6} {'MRR':>6} {'NDCG':>6} {'F1':>6} {'Time':>8}",
        "-" * 90,
    ]
    for r in report.results:
        lines.append(
            f"{r.model_id:<25} {r.query_type:<10} "
            f"{r.precision_at_k:>6.3f} {r.recall_at_k:>6.3f} "
            f"{r.mrr:>6.3f} {r.ndcg:>6.3f} {r.f1_score:>6.3f} "
            f"{r.avg_query_time_ms:>7.1f}ms"
        )
    lines.append("=" * 90)
    return "\n".join(lines)
