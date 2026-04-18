"""Model 4: LLM-Enhanced Search — refines vague queries via LLM, then uses Model 3."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from src.models.model3_enhanced import EnhancedSemanticSearchModel
from src.core.schemas import SearchResult
from src.core.exceptions import LLMError
from config.settings import settings

logger = logging.getLogger("semantic_search")


class LLMEnhancedSearchModel(EnhancedSemanticSearchModel):
    """Extends Model 3: refines queries with LLM before searching."""

    model_id = "model4_llm"

    def __init__(self):
        super().__init__()
        self._llm_client = None
        self._system_prompt = self._load_system_prompt()

    @staticmethod
    def _load_system_prompt() -> str:
        """Load system prompt from models_config.yaml."""
        config_path = Path(settings.base_dir) / "config" / "models_config.yaml"
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config.get("model4_llm", {}).get(
                "system_prompt",
                "Convert this vague e-commerce query to precise product search terms. "
                "Return ONLY a refined search query string.",
            )
        except Exception:
            return (
                "Convert this vague e-commerce query to precise product search terms. "
                "Return ONLY a refined search query string."
            )

    def _get_llm_client(self):
        if self._llm_client is None:
            from openai import OpenAI
            api_key = settings.openai_api_key
            if not api_key:
                raise LLMError(
                    "OPENAI_API_KEY not set. Set it in .env or environment variables."
                )
            self._llm_client = OpenAI(
                base_url=settings.openai_base_url,
                api_key=api_key,
            )
        return self._llm_client

    def enhance_query(self, query: str) -> str:
        """Use LLM to refine a vague/conversational query."""
        try:
            client = self._get_llm_client()
            response = client.chat.completions.create(
                model=settings.llm_model_name,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=150,
                temperature=0.3,
                timeout=10,
            )
            refined = response.choices[0].message.content.strip()
            logger.info("[Model 4] Query refined: '%s' → '%s'", query, refined)
            return refined
        except LLMError:
            raise
        except Exception as e:
            logger.warning("[Model 4] LLM call failed: %s. Falling back to original query.", e)
            return query

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Enhance query with LLM, then use Model 3 search."""
        if not self._ready:
            raise RuntimeError("Model 4 not initialized. Call build_index() first.")

        try:
            enhanced_query = self.enhance_query(query)
        except LLMError:
            logger.warning("[Model 4] LLM unavailable, falling back to Model 3 search")
            enhanced_query = query

        # Use parent (Model 3) search with enhanced query
        return super().search(enhanced_query, top_k)

    def search_with_fallback(self, query: str, top_k: int = 5) -> tuple[list[SearchResult], str]:
        """Search and return both results and the refined query."""
        try:
            enhanced_query = self.enhance_query(query)
        except LLMError:
            enhanced_query = query

        results = super().search(enhanced_query, top_k)
        return results, enhanced_query

    def save(self, directory: str | Path) -> None:
        # Model 4 reuses Model 3's index — just save a marker
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        import json
        config = {
            "reuses_model3_index": True,
            "llm_model": settings.llm_model_name,
            "system_prompt": self._system_prompt,
        }
        with open(directory / "api_config.json", "w") as f:
            json.dump(config, f, indent=2)
        logger.info("[Model 4] Config saved to %s", directory)

    def load(self, directory: str | Path) -> None:
        # Load Model 3's index
        model3_dir = Path(directory).parent / "model3"
        super().load(model3_dir)
        logger.info("[Model 4] Loaded Model 3 index from %s", model3_dir)

    @property
    def is_ready(self) -> bool:
        return self._ready
