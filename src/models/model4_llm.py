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


# Rule-based intent → product-vocabulary expansion. Used as a deterministic
# fallback when the LLM is unavailable (quota exceeded, no key, network error).
# Keys are substring triggers; values are space-joined expansions that steer the
# embedding toward the correct product cluster.
_INTENT_EXPANSIONS = {
    "mens tshirt": "men men's clothing t-shirt tshirt tee round neck polo neck",
    "men tshirt": "men men's clothing t-shirt tshirt tee round neck polo neck",
    "men t-shirt": "men men's clothing t-shirt tshirt tee round neck polo neck",
    "mens t-shirt": "men men's clothing t-shirt tshirt tee round neck polo neck",
    "men tee": "men men's clothing t-shirt tshirt tee",
    "for men tshirt": "men men's clothing t-shirt tshirt tee",
    "watch movies": "television tv led lcd smart monitor home theatre projector",
    "watch videos": "television tv monitor tablet laptop projector",
    "listen to music": "headphone earphone speaker bluetooth headset audio",
    "listen music": "headphone earphone speaker bluetooth headset audio",
    "take pictures": "camera dslr mirrorless lens photography smartphone",
    "take photos": "camera dslr mirrorless lens photography",
    "record video": "camera camcorder gopro action camera dslr",
    "play games": "gaming console controller joystick gamepad",
    "read book": "kindle ereader tablet",
    "read digitally": "kindle ereader tablet",
    "cook food": "cookware pan pot kitchen utensil knife",
    "prepare food": "cookware pan pot kitchen utensil knife chopping board",
    "store food": "container jar storage box airtight",
    "clean clothes": "washing machine detergent laundry",
    "wash clothes": "washing machine detergent laundry",
    "stay cool": "air conditioner fan cooler",
    "stay warm": "heater blanket jacket sweater",
    "sleep better": "bed mattress pillow bedsheet",
    "work out": "gym fitness dumbbell yoga mat treadmill sports",
    "exercise": "gym fitness dumbbell yoga mat treadmill sports shoes",
    "run outdoors": "running shoes sports shoes athletic footwear",
    "walk comfortably": "shoes sneakers footwear casual",
    "carry laptop": "laptop bag backpack messenger",
    "carry things": "bag backpack handbag tote",
    "keep money": "wallet purse cardholder",
    "organize money": "wallet purse cardholder",
    "wear to office": "formal shirt trouser blazer tie formal shoe",
    "wear to work": "formal shirt trouser blazer tie formal shoe",
    "wear casually": "t-shirt jeans casual shirt",
    "feet for exercise": "sports shoes running shoes athletic footwear",
    "on my feet": "shoes footwear sneakers sandal",
    "for my daughter": "girls kids children",
    "for my son": "boys kids children",
    "for my baby": "baby infant toddler",
    "for kids": "kids children boys girls",
    "charge phone": "mobile charger power bank cable adapter",
    "cut hair": "trimmer shaver clipper",
    "shave": "trimmer shaver razor",
    "protect phone": "mobile case cover screen guard tempered glass",
    "smell good": "perfume deodorant cologne fragrance",
    "look good": "cosmetics makeup beauty skincare",
}


def _rule_based_expand(query: str) -> str:
    """Deterministic query expansion. Appends product vocabulary for matching intents."""
    ql = query.lower()
    expansions: list[str] = []
    for trigger, terms in _INTENT_EXPANSIONS.items():
        if trigger in ql:
            expansions.append(terms)
    if not expansions:
        return query
    return query + " " + " ".join(expansions)


class LLMEnhancedSearchModel(EnhancedSemanticSearchModel):
    """Extends Model 3: refines queries with LLM before searching.

    If the LLM call fails (quota, auth, network), a deterministic intent-based
    expansion is used so the model still improves on vague queries.
    """

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
        """Refine a vague/conversational query via LLM, with rule-based fallback."""
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
            logger.info("[Model 4] Query refined (LLM): '%s' -> '%s'", query, refined)
            return refined
        except LLMError:
            raise
        except Exception as e:
            expanded = _rule_based_expand(query)
            if expanded != query:
                logger.warning(
                    "[Model 4] LLM failed (%s); using rule-based expansion: '%s' -> '%s'",
                    e, query, expanded,
                )
            else:
                logger.warning("[Model 4] LLM failed (%s); no rule match, using raw query.", e)
            return expanded

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Enhance query (LLM or rule-based), then use Model 3 search."""
        if not self._ready:
            raise RuntimeError("Model 4 not initialized. Call build_index() first.")

        try:
            enhanced_query = self.enhance_query(query)
        except LLMError:
            enhanced_query = _rule_based_expand(query)
            logger.warning(
                "[Model 4] LLM unavailable; rule-based expansion: '%s' -> '%s'",
                query, enhanced_query,
            )

        return super().search(enhanced_query, top_k)

    def search_with_fallback(self, query: str, top_k: int = 5) -> tuple[list[SearchResult], str]:
        """Search and return both results and the refined query."""
        try:
            enhanced_query = self.enhance_query(query)
        except LLMError:
            enhanced_query = _rule_based_expand(query)

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
