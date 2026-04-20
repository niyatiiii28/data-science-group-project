"""Multi-model orchestrator: routes queries, runs ensemble, deduplicates results."""

import logging
import re
import time
from typing import Optional

import pandas as pd

from src.models.base import BaseSearchModel
from src.models.model1_hierarchical import HierarchicalSearchModel
from src.models.model2_flat_semantic import FlatSemanticSearchModel
from src.models.model3_enhanced import EnhancedSemanticSearchModel
from src.models.model4_llm import LLMEnhancedSearchModel
from src.core.schemas import SearchResult, SearchResponse
from src.core.exceptions import InvalidModelError, ModelNotLoadedError
from config.constants import ModelID

logger = logging.getLogger("semantic_search")


class SearchOrchestrator:
    """Loads all models, routes queries, and optionally ensembles results."""

    def __init__(self):
        self.models: dict[str, BaseSearchModel] = {}
        self.df: Optional[pd.DataFrame] = None

    @staticmethod
    def _query_constraints(query: str) -> dict:
        """Extract lightweight lexical intent constraints from query text."""
        q = query.lower().strip()

        # Gender intent
        men_pat = r"\b(men|mens|men's|male|gents|gentlemen|boy)\b"
        women_pat = r"\b(women|womens|women's|female|ladies|girl)\b"
        gender = None
        if re.search(men_pat, q):
            gender = "men"
        elif re.search(women_pat, q):
            gender = "women"

        # Product-type intent for common ambiguous apparel terms
        wants_tshirt = bool(re.search(r"\b(t[\s\-]?shirt|tee|tees)\b", q))
        wants_kids = bool(re.search(r"\b(kid|kids|child|children|boy|girl|infant|baby|toddler)\b", q))

        return {
            "gender": gender,
            "wants_tshirt": wants_tshirt,
            "wants_kids": wants_kids,
        }

    @staticmethod
    def _intent_adjusted_score(result: SearchResult, base_score: float, constraints: dict) -> float:
        """Apply deterministic boosts/penalties for explicit intent (gender/type)."""
        text = " ".join([
            str(result.product_name or ""),
            str(result.department or ""),
            str(result.category or ""),
            str(result.description or ""),
        ]).lower()

        score = base_score

        has_men = bool(re.search(r"\b(men|men's|mens|male|gents|boy)\b", text))
        has_women = bool(re.search(r"\b(women|women's|womens|female|ladies|girl)\b", text))
        has_kids = bool(re.search(r"\b(kid|kids|child|children|boy|girl|infant|baby|toddler)\b", text))

        gender = constraints.get("gender")
        if gender == "men":
            if has_women and not has_men:
                score *= 0.35
            elif has_men:
                score *= 1.20
        elif gender == "women":
            if has_men and not has_women:
                score *= 0.35
            elif has_women:
                score *= 1.20

        if gender in {"men", "women"} and has_kids and not constraints.get("wants_kids"):
            score *= 0.60

        if constraints.get("wants_tshirt"):
            has_tshirt = bool(re.search(r"\b(t[\s\-]?shirt|tee|polo neck|round neck)\b", text))
            has_bra = bool(re.search(r"\bbra\b", text))
            if has_bra:
                score *= 0.30
            if has_tshirt:
                score *= 1.15

        return max(0.0, min(score, 1.0))

    def _apply_query_intent_rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Re-rank results with deterministic intent constraints."""
        constraints = self._query_constraints(query)
        if not constraints["gender"] and not constraints["wants_tshirt"]:
            return results

        rescored: list[tuple[float, SearchResult]] = []
        for r in results:
            adj = self._intent_adjusted_score(r, float(r.confidence_score), constraints)
            r.confidence_score = round(adj, 4)
            rescored.append((adj, r))

        rescored.sort(key=lambda x: x[0], reverse=True)
        reranked = []
        for rank, (_, r) in enumerate(rescored, 1):
            r.rank = rank
            reranked.append(r)
        return reranked

    def initialize_models(self, df: pd.DataFrame, enriched_df: Optional[pd.DataFrame] = None) -> None:
        """Build indices for all 4 models."""
        self.df = df
        logger.info("Initializing all models with %d products", len(df))

        # Model 1: Hierarchical
        m1 = HierarchicalSearchModel()
        m1.build_index(df)
        self.models[ModelID.HIERARCHICAL] = m1

        # Model 2: Flat semantic (basic descriptions)
        m2 = FlatSemanticSearchModel()
        m2.build_index(df)
        self.models[ModelID.FLAT_SEMANTIC] = m2

        # Model 3: Enhanced semantic (enriched descriptions)
        m3 = EnhancedSemanticSearchModel()
        data_for_m3 = enriched_df if enriched_df is not None else df
        m3.build_index(data_for_m3)
        self.models[ModelID.ENHANCED_SEMANTIC] = m3

        # Model 4: LLM-enhanced (reuses Model 3's index)
        m4 = LLMEnhancedSearchModel()
        m4.df = m3.df
        m4.faiss_index = m3.faiss_index
        m4.embeddings = m3.embeddings
        m4._ready = True
        self.models[ModelID.LLM_ENHANCED] = m4

        logger.info("All %d models initialized", len(self.models))

    def search(
        self, query: str, model_id: Optional[str] = None, top_k: int = 5
    ) -> SearchResponse:
        """Route query to a specific model or run all."""
        start = time.perf_counter()
        constraints = self._query_constraints(query)
        constrained = bool(constraints["gender"] or constraints["wants_tshirt"])
        fetch_k = min(top_k * 8, 50) if constrained else top_k

        if model_id:
            if model_id not in self.models:
                raise InvalidModelError(f"Unknown model: {model_id}. Available: {list(self.models.keys())}")
            model = self.models[model_id]
            if not model.is_ready:
                raise ModelNotLoadedError(f"Model {model_id} is not loaded")
            results = model.search(query, fetch_k)
            results = self._apply_query_intent_rerank(query, results)[:top_k]
            used_model = model_id
        else:
            # Default to Model 3 (best balance of quality and speed)
            used_model = ModelID.ENHANCED_SEMANTIC
            model = self.models.get(used_model)
            if model and model.is_ready:
                results = model.search(query, fetch_k)
                results = self._apply_query_intent_rerank(query, results)[:top_k]
            else:
                raise ModelNotLoadedError("No models available")

        elapsed_ms = (time.perf_counter() - start) * 1000

        return SearchResponse(
            query=query,
            model_id=used_model,
            total_results=len(results),
            results=results,
            query_time_ms=round(elapsed_ms, 2),
        )

    def search_ensemble(self, query: str, top_k: int = 5) -> SearchResponse:
        """Run all models and merge/deduplicate results by weighted voting."""
        start = time.perf_counter()

        model_weights = {
            ModelID.HIERARCHICAL: 0.20,
            ModelID.FLAT_SEMANTIC: 0.20,
            ModelID.ENHANCED_SEMANTIC: 0.35,
            ModelID.LLM_ENHANCED: 0.25,
        }
        constraints = self._query_constraints(query)
        constrained = bool(constraints["gender"] or constraints["wants_tshirt"])
        per_model_k = min(top_k * 8, 50) if constrained else (top_k * 2)

        # Collect results from all ready models
        product_scores: dict[str, dict] = {}  # product_id → {score, result}

        for model_id, weight in model_weights.items():
            model = self.models.get(model_id)
            if not model or not model.is_ready:
                continue

            try:
                results = model.search(query, per_model_k)
                for result in results:
                    pid = result.product_id
                    adjusted = self._intent_adjusted_score(result, float(result.confidence_score), constraints)
                    weighted_score = adjusted * weight
                    if pid in product_scores:
                        product_scores[pid]["score"] += weighted_score
                        product_scores[pid]["votes"] += 1
                    else:
                        product_scores[pid] = {
                            "score": weighted_score,
                            "result": result,
                            "votes": 1,
                        }
            except Exception as e:
                logger.warning("Model %s failed during ensemble: %s", model_id, e)

        # Sort by combined score, break ties by vote count
        sorted_products = sorted(
            product_scores.values(),
            key=lambda x: (x["score"], x["votes"]),
            reverse=True,
        )

        # Re-rank
        results = []
        for rank, item in enumerate(sorted_products[:top_k], 1):
            r = item["result"]
            r.confidence_score = round(min(item["score"], 1.0), 4)
            r.rank = rank
            results.append(r)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return SearchResponse(
            query=query,
            model_id="ensemble",
            total_results=len(results),
            results=results,
            query_time_ms=round(elapsed_ms, 2),
        )

    def get_loaded_models(self) -> list[str]:
        return [mid for mid, m in self.models.items() if m.is_ready]

    def save_all(self, base_dir: str) -> None:
        from pathlib import Path
        base = Path(base_dir)
        model_dirs = {
            ModelID.HIERARCHICAL: "model1",
            ModelID.FLAT_SEMANTIC: "model2",
            ModelID.ENHANCED_SEMANTIC: "model3",
            ModelID.LLM_ENHANCED: "model4",
        }
        for model_id, dirname in model_dirs.items():
            model = self.models.get(model_id)
            if model and model.is_ready:
                model.save(base / dirname)

    def load_all(self, base_dir: str, df: pd.DataFrame, enriched_df: Optional[pd.DataFrame] = None) -> None:
        from pathlib import Path
        base = Path(base_dir)

        # Model 1
        m1 = HierarchicalSearchModel()
        m1.df = df.reset_index(drop=True)
        try:
            m1.load(base / "model1")
            self.models[ModelID.HIERARCHICAL] = m1
        except Exception as e:
            logger.warning("Could not load Model 1: %s", e)

        # Model 2
        m2 = FlatSemanticSearchModel()
        try:
            m2.load(base / "model2")
            self.models[ModelID.FLAT_SEMANTIC] = m2
        except Exception as e:
            logger.warning("Could not load Model 2: %s", e)

        # Model 3
        m3 = EnhancedSemanticSearchModel()
        try:
            m3.load(base / "model3")
            self.models[ModelID.ENHANCED_SEMANTIC] = m3
        except Exception as e:
            logger.warning("Could not load Model 3: %s", e)

        # Model 4 (reuses Model 3)
        if ModelID.ENHANCED_SEMANTIC in self.models:
            m4 = LLMEnhancedSearchModel()
            m3_ref = self.models[ModelID.ENHANCED_SEMANTIC]
            m4.df = m3_ref.df
            m4.faiss_index = m3_ref.faiss_index
            m4.embeddings = m3_ref.embeddings
            m4._ready = True
            self.models[ModelID.LLM_ENHANCED] = m4

        logger.info("Loaded %d models from %s", len(self.models), base_dir)
