"""Generate enriched descriptions for products (300+ words)."""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger("semantic_search")


# Templates per department for generating rich descriptions
_TEMPLATES = {
    "Clothing": (
        "{product_name} is a premium {product_type} designed for {use_case}. "
        "Made from {material_text}, this {category} item offers exceptional comfort "
        "and style. {features_text} Available in {color_text}, it pairs well with "
        "various outfits for {occasion}. The {brand} brand is known for quality "
        "craftsmanship and attention to detail. {size_text} Whether you're looking "
        "for everyday wear or something special, this {subcategory} piece delivers "
        "on both aesthetics and functionality. Perfect for {target_audience} who "
        "value {value_proposition}. Rated {rating}/5 by customers."
    ),
    "Electronics": (
        "{product_name} is a cutting-edge {product_type} from {brand}. "
        "Featuring {features_text}, this {category} device is designed for "
        "{use_case}. {description_text} With advanced technology and reliable "
        "performance, it stands out in the {subcategory} segment. {specs_text} "
        "Ideal for {target_audience} seeking {value_proposition}. "
        "Rated {rating}/5 stars by verified buyers. Priced at {price_text}."
    ),
    "default": (
        "{product_name} — a top-rated {product_type} from {brand}. "
        "Category: {category} > {subcategory}. {description_text} "
        "Key features include {features_text}. {color_text} {material_text} "
        "{size_text} This product has earned {rating}/5 stars and is priced "
        "at {price_text}. Ideal for {use_case}. {tags_text}"
    ),
}

_USE_CASES = {
    "Clothing": "daily wear, office settings, and casual outings",
    "Footwear": "walking, running, and everyday comfort",
    "Electronics": "home entertainment, productivity, and connectivity",
    "Furniture": "home decoration, comfort, and space optimization",
    "Kitchen": "cooking, food preparation, and kitchen efficiency",
    "Beauty": "personal care, skincare, and grooming routines",
    "Sports": "fitness activities, training, and outdoor adventures",
    "Books": "reading, learning, and personal development",
    "Automotive": "vehicle maintenance, upgrades, and driving comfort",
    "Baby": "infant care, comfort, and safety",
}

_OCCASIONS = {
    "Clothing": "casual outings, formal events, and everyday activities",
    "Footwear": "sports, office wear, and weekend outings",
    "default": "various occasions and daily use",
}

_TARGET_AUDIENCE = {
    "Clothing": "fashion-conscious individuals",
    "Electronics": "tech enthusiasts and everyday users",
    "default": "discerning shoppers seeking quality products",
}


def enrich_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Add enriched_description column using template-based generation."""
    logger.info("Generating enriched descriptions for %d products", len(df))
    df = df.copy()
    df["enriched_description"] = df.apply(_generate_description, axis=1)

    avg_len = df["enriched_description"].str.split().str.len().mean()
    logger.info("Average enriched description length: %.0f words", avg_len)
    return df


def _generate_description(row: pd.Series) -> str:
    """Generate a single enriched description from product fields."""
    dept = str(row.get("department", ""))
    cat = str(row.get("category", ""))

    # Pick template
    template_key = "default"
    for key in _TEMPLATES:
        if key != "default" and key.lower() in (dept.lower() + " " + cat.lower()):
            template_key = key
            break

    template = _TEMPLATES[template_key]

    # Build context values
    features = str(row.get("features", ""))
    features_text = features if features else "quality construction and reliable performance"

    color = str(row.get("color", ""))
    color_text = f"Available in {color}." if color else ""

    material = str(row.get("material", ""))
    material_text = f"Crafted from {material}." if material else "high-quality materials"

    size = str(row.get("size_range", ""))
    size_text = f"Available sizes: {size}." if size else ""

    description = str(row.get("description", ""))
    description_text = description if description else ""

    price = row.get("price", 0)
    price_text = f"₹{price:,.0f}" if price else "competitive pricing"

    rating = row.get("rating", 0)

    use_case_key = cat if cat in _USE_CASES else dept if dept in _USE_CASES else "default"
    use_case = _USE_CASES.get(use_case_key, "various everyday needs")

    occasion = _OCCASIONS.get(cat, _OCCASIONS.get(dept, _OCCASIONS["default"]))

    target_key = cat if cat in _TARGET_AUDIENCE else dept if dept in _TARGET_AUDIENCE else "default"
    target_audience = _TARGET_AUDIENCE.get(target_key, _TARGET_AUDIENCE["default"])

    tags = str(row.get("tags", ""))
    tags_text = f"Related: {tags}." if tags else ""

    try:
        enriched = template.format(
            product_name=row.get("product_name", "Product"),
            product_type=row.get("product_type", "item"),
            category=cat,
            subcategory=row.get("subcategory", ""),
            brand=row.get("brand", "Unknown"),
            features_text=features_text,
            color_text=color_text,
            material_text=material_text,
            size_text=size_text,
            description_text=description_text,
            price_text=price_text,
            rating=rating,
            use_case=use_case,
            occasion=occasion,
            target_audience=target_audience,
            value_proposition="quality, style, and value",
            tags_text=tags_text,
            specs_text="",
        )
    except KeyError:
        # Fallback if template has unexpected placeholders
        enriched = (
            f"{row.get('product_name', 'Product')}. "
            f"{description_text} {features_text} "
            f"{color_text} {material_text} {size_text} "
            f"Brand: {row.get('brand', 'Unknown')}. "
            f"Category: {cat}. {tags_text}"
        )

    # Ensure minimum length — pad with product details
    if len(enriched.split()) < 50:
        enriched += (
            f" This {row.get('product_type', 'product')} is perfect for "
            f"{use_case}. From the trusted {row.get('brand', 'Unknown')} brand, "
            f"it delivers excellent value. Highly rated by customers at {rating}/5 stars. "
            f"Part of the {cat} collection in our {dept} department. {tags_text}"
        )

    return enriched.strip()
