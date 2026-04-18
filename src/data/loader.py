"""Dataset loader with validation — handles Flipkart CSV format."""

import ast
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config.constants import CATEGORY_TREE_SEPARATOR, MAX_HIERARCHY_DEPTH
from src.core.exceptions import DataLoadError, DataValidationError

logger = logging.getLogger("semantic_search")


def load_flipkart_dataset(filepath: str | Path) -> pd.DataFrame:
    """Load the raw Flipkart CSV and transform to unified schema."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise DataLoadError(f"Dataset not found: {filepath}")

    logger.info("Loading Flipkart dataset from %s", filepath)
    df = pd.read_csv(filepath, encoding="utf-8")

    logger.info("Raw dataset: %d rows, %d columns", len(df), len(df.columns))
    required = {"product_name", "product_category_tree"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    df = _transform_flipkart_to_unified(df)
    logger.info("Transformed dataset: %d rows, %d columns", len(df), len(df.columns))
    return df


def _transform_flipkart_to_unified(df: pd.DataFrame) -> pd.DataFrame:
    """Map Flipkart columns to the unified 18-column schema."""
    out = pd.DataFrame()

    # product_id
    if "pid" in df.columns:
        out["product_id"] = df["pid"].astype(str)
    else:
        out["product_id"] = [f"P{str(i).zfill(5)}" for i in range(1, len(df) + 1)]

    out["product_name"] = df["product_name"].fillna("Unknown Product").astype(str)

    # Parse category tree → department, category, subcategory, product_type
    hierarchy = df["product_category_tree"].apply(_parse_category_tree)
    out["department"] = hierarchy.apply(lambda x: x[0])
    out["category"] = hierarchy.apply(lambda x: x[1])
    out["subcategory"] = hierarchy.apply(lambda x: x[2])
    out["product_type"] = hierarchy.apply(lambda x: x[3])

    out["description"] = df.get("description", pd.Series("", index=df.index)).fillna("")

    # Parse product_specifications → features, color, material, size_range
    specs = df.get("product_specifications", pd.Series("", index=df.index)).apply(
        _parse_specifications
    )
    out["features"] = specs.apply(lambda x: x.get("features", ""))
    out["color"] = specs.apply(lambda x: x.get("color", ""))
    out["material"] = specs.apply(lambda x: x.get("material", ""))
    out["size_range"] = specs.apply(lambda x: x.get("size_range", ""))

    # Price
    out["price"] = _parse_price_column(df)

    # Rating
    for col in ["product_rating", "overall_rating", "rating"]:
        if col in df.columns:
            out["rating"] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            break
    else:
        out["rating"] = 0.0

    out["reviews_count"] = 0  # Flipkart data doesn't have this directly

    # Tags from category tree words
    out["tags"] = hierarchy.apply(
        lambda x: ", ".join(set(w.lower() for level in x for w in level.split() if len(w) > 2))
    )

    out["brand"] = df.get("brand", pd.Series("Unknown", index=df.index)).fillna("Unknown")
    out["stock_status"] = "In Stock"
    out["keywords"] = out["product_name"].str.lower() + ", " + out["tags"]

    return out


def _parse_category_tree(tree_str: str) -> list[str]:
    """Parse 'A >> B >> C >> D >> E' into [department, category, subcategory, product_type]."""
    if pd.isna(tree_str) or not isinstance(tree_str, str):
        return ["Other", "General", "General", "General"]

    # Some entries are wrapped in list-like brackets
    tree_str = tree_str.strip().strip("[]\"\' ")

    parts = [p.strip() for p in tree_str.split(CATEGORY_TREE_SEPARATOR)]
    parts = [p for p in parts if p]

    # Pad to at least 4 levels
    while len(parts) < MAX_HIERARCHY_DEPTH - 1:
        parts.append(parts[-1] if parts else "General")

    return parts[:4]


def _parse_specifications(spec_str) -> dict:
    """Extract features, color, material, size from product_specifications JSON/dict."""
    result = {"features": "", "color": "", "material": "", "size_range": ""}

    if pd.isna(spec_str) or not spec_str:
        return result

    spec_str = str(spec_str).strip()

    # Try parsing as Python literal (Flipkart format)
    parsed = None
    try:
        parsed = ast.literal_eval(spec_str)
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(spec_str)
        except (json.JSONDecodeError, TypeError):
            return result

    if not parsed:
        return result

    # Flipkart specs are usually: {"product_specification": [{"key": "k", "value": "v"}, ...]}
    specs_list = []
    if isinstance(parsed, dict):
        specs_list = parsed.get("product_specification", [])
        if not specs_list:
            # Flatten dict values
            specs_list = [{"key": k, "value": v} for k, v in parsed.items()]
    elif isinstance(parsed, list):
        specs_list = parsed

    features_parts = []
    color_keys = {"color", "colour", "shade"}
    material_keys = {"material", "fabric", "composition"}
    size_keys = {"size", "dimensions", "size_range"}

    for item in specs_list:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip().lower()
        value = str(item.get("value", "")).strip()

        if not key or not value or value.lower() == "nan":
            continue

        if key in color_keys:
            result["color"] = value
        elif key in material_keys:
            result["material"] = value
        elif key in size_keys:
            result["size_range"] = value
        else:
            features_parts.append(value)

    result["features"] = ", ".join(features_parts[:10])
    return result


def _parse_price_column(df: pd.DataFrame) -> pd.Series:
    """Extract best available price column."""
    for col in ["discounted_price", "retail_price", "price"]:
        if col in df.columns:
            return (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.strip()
                .apply(lambda x: float(x) if x and x != "nan" else 0.0)
            )
    return pd.Series(0.0, index=df.index)


def load_processed_dataset(filepath: str | Path) -> pd.DataFrame:
    """Load an already-processed CSV in unified schema."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise DataLoadError(f"Processed dataset not found: {filepath}")

    df = pd.read_csv(filepath, encoding="utf-8")
    df = df.fillna("")
    logger.info("Loaded processed dataset: %d rows", len(df))
    return df


def save_processed_dataset(df: pd.DataFrame, filepath: str | Path) -> None:
    """Save processed DataFrame to CSV."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False, encoding="utf-8")
    logger.info("Saved processed dataset: %d rows → %s", len(df), filepath)
