"""Text cleaning, deduplication, and normalization pipeline."""

import logging
import re
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger("semantic_search")


class DataPreprocessor:
    """Pipeline: clean → deduplicate → normalize → validate."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._original_count = len(df)

    def run_full_pipeline(self) -> pd.DataFrame:
        logger.info("Starting preprocessing pipeline on %d rows", len(self.df))
        self.clean_text_fields()
        self.normalize_prices()
        self.normalize_ratings()
        self.remove_duplicates()
        self.remove_empty_products()
        self.fill_missing_values()
        self.validate_schema()
        logger.info(
            "Pipeline complete: %d → %d rows (removed %d)",
            self._original_count,
            len(self.df),
            self._original_count - len(self.df),
        )
        return self.df

    def clean_text_fields(self) -> None:
        text_cols = [
            "product_name", "description", "features", "tags",
            "brand", "keywords", "department", "category",
            "subcategory", "product_type",
        ]
        for col in text_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).apply(_clean_text)

    def normalize_prices(self) -> None:
        if "price" in self.df.columns:
            self.df["price"] = pd.to_numeric(self.df["price"], errors="coerce").fillna(0.0)
            # Cap extreme outliers (beyond 99.9th percentile)
            cap = self.df["price"].quantile(0.999)
            if cap > 0:
                self.df.loc[self.df["price"] > cap * 3, "price"] = cap

    def normalize_ratings(self) -> None:
        if "rating" in self.df.columns:
            self.df["rating"] = pd.to_numeric(self.df["rating"], errors="coerce").fillna(0.0)
            self.df["rating"] = self.df["rating"].clip(0.0, 5.0)

    def remove_duplicates(self) -> None:
        before = len(self.df)
        # Exact duplicates by product_id
        if "product_id" in self.df.columns:
            self.df = self.df.drop_duplicates(subset=["product_id"], keep="first")

        # Near-duplicates: same name + same category
        self.df = self.df.drop_duplicates(
            subset=["product_name", "category"], keep="first"
        )
        removed = before - len(self.df)
        if removed:
            logger.info("Removed %d duplicate rows", removed)

    def remove_empty_products(self) -> None:
        before = len(self.df)
        self.df = self.df[
            self.df["product_name"].str.strip().str.len() > 2
        ]
        removed = before - len(self.df)
        if removed:
            logger.info("Removed %d empty/invalid product rows", removed)

    def fill_missing_values(self) -> None:
        defaults = {
            "department": "Other",
            "category": "General",
            "subcategory": "General",
            "product_type": "General",
            "description": "",
            "features": "",
            "brand": "Unknown",
            "color": "",
            "material": "",
            "size_range": "",
            "tags": "",
            "keywords": "",
            "stock_status": "In Stock",
        }
        for col, default in defaults.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(default).replace("nan", default)

    def validate_schema(self) -> None:
        required_cols = {"product_id", "product_name", "department", "category"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Schema validation failed — missing columns: {missing}")

    def get_stats(self) -> dict:
        return {
            "total_products": len(self.df),
            "departments": self.df["department"].nunique(),
            "categories": self.df["category"].nunique(),
            "brands": self.df["brand"].nunique(),
            "avg_price": round(self.df["price"].mean(), 2),
            "avg_rating": round(self.df["rating"].mean(), 2),
            "products_with_description": int((self.df["description"].str.len() > 10).sum()),
            "products_with_features": int((self.df["features"].str.len() > 5).sum()),
        }


def _clean_text(text: str) -> str:
    """Clean a single text field."""
    if not text or text.lower() in ("nan", "none", "null"):
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Normalize unicode
    text = text.encode("ascii", "ignore").decode("ascii")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
