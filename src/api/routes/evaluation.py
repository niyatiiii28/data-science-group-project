"""Evaluation API endpoint — trigger benchmarks from the UI."""

from flask import Blueprint, jsonify, current_app

eval_bp = Blueprint("evaluation", __name__)


@eval_bp.route("/evaluate", methods=["POST"])
def run_evaluation():
    """Run benchmark suite across all loaded models."""
    orchestrator = current_app.config.get("ORCHESTRATOR")
    if not orchestrator:
        return jsonify({"error": "Search engine not initialized"}), 503

    from src.evaluation.benchmarks import run_benchmarks
    report = run_benchmarks(orchestrator)
    return jsonify(report.model_dump())
