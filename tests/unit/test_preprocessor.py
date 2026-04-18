"""Unit tests for the data preprocessor."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from src.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    def test_removes_duplicates(self, sample_df):
        # Add a duplicate
        dup_df = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
        proc = DataPreprocessor(dup_df)
        result = proc.run_full_pipeline()
        assert len(result) == 10  # duplicate removed

    def test_removes_empty_products(self, sample_df):
        sample_df.loc[0, "product_name"] = ""
        proc = DataPreprocessor(sample_df)
        result = proc.run_full_pipeline()
        assert len(result) < 10

    def test_normalizes_ratings(self, sample_df):
        sample_df.loc[0, "rating"] = 7.0  # Invalid
        proc = DataPreprocessor(sample_df)
        result = proc.run_full_pipeline()
        assert result["rating"].max() <= 5.0

    def test_fills_missing_brands(self, sample_df):
        sample_df.loc[0, "brand"] = None
        proc = DataPreprocessor(sample_df)
        result = proc.run_full_pipeline()
        assert result.iloc[0]["brand"] == "Unknown"

    def test_schema_validation(self, sample_df):
        proc = DataPreprocessor(sample_df)
        result = proc.run_full_pipeline()
        assert "product_id" in result.columns
        assert "product_name" in result.columns

    def test_get_stats(self, sample_df):
        proc = DataPreprocessor(sample_df)
        proc.run_full_pipeline()
        stats = proc.get_stats()
        assert stats["total_products"] == 10
        assert stats["departments"] > 0
