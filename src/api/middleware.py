"""Middleware: request timing, request IDs, rate limiting."""

import time
import uuid
import logging

from flask import Flask, request, g

logger = logging.getLogger("semantic_search")


def register_middleware(app: Flask) -> None:
    """Attach before/after request hooks."""

    @app.before_request
    def before_request():
        g.request_id = str(uuid.uuid4())[:8]
        g.start_time = time.perf_counter()

    @app.after_request
    def after_request(response):
        # Timing header
        elapsed = (time.perf_counter() - g.get("start_time", 0)) * 1000
        response.headers["X-Request-ID"] = g.get("request_id", "")
        response.headers["X-Response-Time-Ms"] = f"{elapsed:.2f}"

        # Log request
        logger.info(
            "request",
            extra={
                "request_id": g.get("request_id"),
                "method": request.method,
                "path": request.path,
                "status": response.status_code,
                "time_ms": round(elapsed, 2),
            },
        )
        return response

    # Optional: rate limiting (only if flask-limiter is installed)
    try:
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address
        from config.settings import settings

        Limiter(
            app=app,
            key_func=get_remote_address,
            default_limits=[settings.rate_limit_default],
            storage_uri="memory://",
        )
    except ImportError:
        logger.warning("flask-limiter not installed, rate limiting disabled")
