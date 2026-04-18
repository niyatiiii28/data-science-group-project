"""Health and readiness check endpoints."""

from flask import Blueprint, jsonify, current_app

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health():
    """Liveness probe — is the server running?"""
    return jsonify({"status": "ok"})


@health_bp.route("/ready", methods=["GET"])
def ready():
    """Readiness probe — are models loaded and ready?"""
    orchestrator = current_app.config.get("ORCHESTRATOR")
    if not orchestrator:
        return jsonify({"status": "not_ready", "models_loaded": [], "total_products": 0}), 503

    loaded = orchestrator.get_loaded_models()
    total = len(orchestrator.df) if orchestrator.df is not None else 0

    if not loaded:
        return jsonify({"status": "not_ready", "models_loaded": [], "total_products": total}), 503

    return jsonify({
        "status": "ready",
        "models_loaded": loaded,
        "total_products": total,
    })
