"""Global error handlers returning structured JSON."""

import logging
from flask import Flask, jsonify

from src.core.exceptions import (
    SemanticSearchError,
    InvalidModelError,
    ModelNotLoadedError,
    DataValidationError,
    QueryTooLongError,
)

logger = logging.getLogger("semantic_search")


def register_error_handlers(app: Flask) -> None:

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "Bad Request", "message": str(e)}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not Found", "message": str(e)}), 404

    @app.errorhandler(422)
    def unprocessable(e):
        return jsonify({"error": "Unprocessable Entity", "message": str(e)}), 422

    @app.errorhandler(429)
    def rate_limited(e):
        return jsonify({"error": "Rate Limited", "message": "Too many requests"}), 429

    @app.errorhandler(500)
    def internal_error(e):
        logger.error("Internal error: %s", e, exc_info=True)
        return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred"}), 500

    @app.errorhandler(InvalidModelError)
    def handle_invalid_model(e):
        return jsonify({"error": "Invalid Model", "message": str(e)}), 400

    @app.errorhandler(ModelNotLoadedError)
    def handle_model_not_loaded(e):
        return jsonify({"error": "Model Not Loaded", "message": str(e)}), 503

    @app.errorhandler(DataValidationError)
    def handle_validation_error(e):
        return jsonify({"error": "Data Validation Error", "message": str(e)}), 422

    @app.errorhandler(QueryTooLongError)
    def handle_query_too_long(e):
        return jsonify({"error": "Query Too Long", "message": str(e)}), 400
