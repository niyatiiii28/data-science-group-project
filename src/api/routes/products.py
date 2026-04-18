"""Product browsing API routes."""

from flask import Blueprint, request, jsonify, current_app

products_bp = Blueprint("products", __name__)


@products_bp.route("/products", methods=["GET"])
def list_products():
    """List products with pagination and optional filtering.

    Query params: page, per_page, department, category, brand, min_price, max_price
    """
    orchestrator = current_app.config.get("ORCHESTRATOR")
    if not orchestrator or orchestrator.df is None:
        return jsonify({"error": "Data not loaded"}), 503

    df = orchestrator.df

    # Filters
    dept = request.args.get("department")
    cat = request.args.get("category")
    brand = request.args.get("brand")
    min_price = request.args.get("min_price", type=float)
    max_price = request.args.get("max_price", type=float)

    if dept:
        df = df[df["department"].str.lower() == dept.lower()]
    if cat:
        df = df[df["category"].str.lower() == cat.lower()]
    if brand:
        df = df[df["brand"].str.lower() == brand.lower()]
    if min_price is not None:
        df = df[df["price"] >= min_price]
    if max_price is not None:
        df = df[df["price"] <= max_price]

    # Pagination
    page = max(int(request.args.get("page", 1)), 1)
    per_page = min(int(request.args.get("per_page", 20)), 100)
    total = len(df)
    start = (page - 1) * per_page
    end = start + per_page

    products = df.iloc[start:end].to_dict(orient="records")

    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
        "products": products,
    })


@products_bp.route("/products/<product_id>", methods=["GET"])
def get_product(product_id: str):
    """Get a single product by ID."""
    orchestrator = current_app.config.get("ORCHESTRATOR")
    if not orchestrator or orchestrator.df is None:
        return jsonify({"error": "Data not loaded"}), 503

    df = orchestrator.df
    match = df[df["product_id"] == product_id]

    if match.empty:
        return jsonify({"error": f"Product {product_id} not found"}), 404

    return jsonify(match.iloc[0].to_dict())


@products_bp.route("/products/stats", methods=["GET"])
def product_stats():
    """Get dataset statistics."""
    orchestrator = current_app.config.get("ORCHESTRATOR")
    if not orchestrator or orchestrator.df is None:
        return jsonify({"error": "Data not loaded"}), 503

    df = orchestrator.df
    stats = {
        "total_products": len(df),
        "departments": df["department"].nunique(),
        "categories": df["category"].nunique(),
        "brands": df["brand"].nunique(),
        "avg_price": round(df["price"].mean(), 2),
        "avg_rating": round(df["rating"].mean(), 2),
        "department_counts": df["department"].value_counts().head(10).to_dict(),
        "top_brands": df["brand"].value_counts().head(10).to_dict(),
    }
    return jsonify(stats)
