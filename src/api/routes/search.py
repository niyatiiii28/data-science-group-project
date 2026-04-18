"""Search API routes."""

from flask import Blueprint, request, jsonify, current_app

search_bp = Blueprint("search", __name__)


@search_bp.route("/search", methods=["POST"])
def search():
    """Search across products using the default or specified model.

    Request JSON:
        {
            "query": "comfortable office shoes",
            "model_id": "model3_enhanced" (optional),
            "top_k": 5 (optional)
        }
    """
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query is required"}), 400
    if len(query) > 500:
        return jsonify({"error": "Query too long (max 500 chars)"}), 400

    model_id = data.get("model_id")
    top_k = min(int(data.get("top_k", 5)), 50)

    orchestrator = current_app.config.get("ORCHESTRATOR")
    if not orchestrator:
        return jsonify({"error": "Search engine not initialized"}), 503

    response = orchestrator.search(query, model_id=model_id, top_k=top_k)
    return jsonify(response.model_dump())


@search_bp.route("/search/<model_id>", methods=["POST"])
def search_with_model(model_id: str):
    """Search using a specific model."""
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query is required"}), 400

    top_k = min(int(data.get("top_k", 5)), 50)

    orchestrator = current_app.config.get("ORCHESTRATOR")
    if not orchestrator:
        return jsonify({"error": "Search engine not initialized"}), 503

    response = orchestrator.search(query, model_id=model_id, top_k=top_k)
    return jsonify(response.model_dump())


@search_bp.route("/search/ensemble", methods=["POST"])
def search_ensemble():
    """Run all models and merge results."""
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query is required"}), 400

    top_k = min(int(data.get("top_k", 5)), 50)

    orchestrator = current_app.config.get("ORCHESTRATOR")
    if not orchestrator:
        return jsonify({"error": "Search engine not initialized"}), 503

    response = orchestrator.search_ensemble(query, top_k=top_k)
    return jsonify(response.model_dump())


@search_bp.route("/models", methods=["GET"])
def list_models():
    """List available search models."""
    orchestrator = current_app.config.get("ORCHESTRATOR")
    if not orchestrator:
        return jsonify({"models": []})

    return jsonify({
        "models": orchestrator.get_loaded_models(),
    })
