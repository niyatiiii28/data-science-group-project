"""Pydantic data models for products, queries, and results."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class Product(BaseModel):
    product_id: str
    product_name: str
    department: str = ""
    category: str = ""
    subcategory: str = ""
    product_type: str = ""
    description: str = ""
    features: str = ""
    price: float = 0.0
    rating: float = 0.0
    reviews_count: int = 0
    tags: str = ""
    color: str = ""
    size_range: str = ""
    material: str = ""
    brand: str = ""
    stock_status: str = "In Stock"
    keywords: str = ""
    enriched_description: str = ""

    @field_validator("price", mode="before")
    @classmethod
    def coerce_price(cls, v):
        if isinstance(v, str):
            v = v.replace(",", "").replace("₹", "").strip()
            return float(v) if v else 0.0
        return float(v) if v else 0.0

    @field_validator("rating", mode="before")
    @classmethod
    def coerce_rating(cls, v):
        try:
            val = float(v) if v else 0.0
            return max(0.0, min(5.0, val))
        except (ValueError, TypeError):
            return 0.0

    @field_validator("reviews_count", mode="before")
    @classmethod
    def coerce_reviews(cls, v):
        try:
            return int(float(v)) if v else 0
        except (ValueError, TypeError):
            return 0


class SearchFilters(BaseModel):
    department: Optional[str] = None
    category: Optional[str] = None
    min_price: Optional[float] = Field(default=None, ge=0)
    max_price: Optional[float] = Field(default=None, ge=0)
    min_rating: Optional[float] = Field(default=None, ge=0, le=5)
    brand: Optional[str] = None


class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    model_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Optional[SearchFilters] = None





class SearchResult(BaseModel):
    product_id: str
    product_name: str
    department: str = ""
    category: str = ""
    brand: str = ""
    price: float = 0.0
    rating: float = 0.0
    reviews_count: int = 0
    description: str = ""
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    rank: int = 0


class SearchResponse(BaseModel):
    query: str
    model_id: str
    total_results: int
    results: list[SearchResult]
    query_time_ms: float


class EvaluationResult(BaseModel):
    model_id: str
    query_type: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float
    f1_score: float
    avg_query_time_ms: float


class EvaluationReport(BaseModel):
    timestamp: str
    results: list[EvaluationResult]


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    total_products: int
