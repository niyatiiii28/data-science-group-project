"""Flask application factory."""

import logging
from flask import Flask
from flask_cors import CORS

from config.settings import settings

logger = logging.getLogger("semantic_search")


def create_app() -> Flask:
    """Application factory: creates and configures the Flask app."""
    app = Flask(
        __name__,
        template_folder=str(settings.base_dir / "templates"),
        static_folder=str(settings.base_dir / "templates" / "static"),
    )
    app.config["SECRET_KEY"] = settings.secret_key

    # CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register middleware
    from src.api.middleware import register_middleware
    register_middleware(app)

    # Register error handlers
    from src.api.error_handlers import register_error_handlers
    register_error_handlers(app)

    # Register route blueprints
    from src.api.routes.search import search_bp
    from src.api.routes.products import products_bp
    from src.api.routes.health import health_bp
    from src.api.routes.evaluation import eval_bp

    app.register_blueprint(search_bp, url_prefix="/api")
    app.register_blueprint(products_bp, url_prefix="/api")
    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(eval_bp, url_prefix="/api")

    # Register frontend route
    @app.route("/")
    def index():
        from flask import render_template
        return render_template("index.html")

    @app.route("/compare")
    def compare():
        from flask import render_template
        return render_template("compare.html")

    @app.route("/about")
    def about():
        from flask import render_template
        return render_template("about.html")

    logger.info("Flask app created successfully")
    return app
