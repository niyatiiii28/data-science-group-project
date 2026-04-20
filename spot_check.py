"""Spot-check: inspect search results for a vague query."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import settings
from src.core.logging_config import setup_logging
from src.data.loader import load_processed_dataset
from src.models.orchestrator import SearchOrchestrator
from src.evaluation.test_queries import is_relevant

logger = setup_logging()

# Test query and expected relevant terms
TEST_QUERY = "something to watch movies on"
RELEVANT_TERMS = {"television", "tv", "home theatre", "projector", "monitor"}

def main():
    processed_path = settings.processed_data_path / "products_cleaned.csv"
    if not processed_path.exists():
        logger.error("No processed data found. Run preprocessing first.")
        sys.exit(1)

    df = load_processed_dataset(processed_path)
    enriched_path = settings.processed_data_path / "products_enriched.csv"
    enriched_df = load_processed_dataset(enriched_path) if enriched_path.exists() else df

    # Load orchestrator
    orchestrator = SearchOrchestrator()
    try:
        orchestrator.load_all(str(settings.models_path), df, enriched_df)
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        sys.exit(1)

    print("\n" + "="*80)
    print(f"SPOT-CHECK: Query '{TEST_QUERY}'")
    print(f"Expected relevant terms: {RELEVANT_TERMS}")
    print("="*80)

    # Run each model
    for model_id in ["model2_flat_semantic", "model3_enhanced", "model4_llm"]:
        print(f"\n[{model_id}]")
        try:
            response = orchestrator.search(TEST_QUERY, model_id=model_id, top_k=5)
            print(f"Query time: {response.query_time_ms}ms\n")
            
            for i, result in enumerate(response.results, 1):
                is_rel = is_relevant(result.model_dump(), RELEVANT_TERMS)
                rel_marker = "RELEVANT" if is_rel else "NOT RELEVANT"
                print(f"  {i}. {result.product_name}")
                print(f"     Dept: {result.department} | Cat: {result.category}")
                print(f"     Score: {result.confidence_score} | {rel_marker}")
        except Exception as e:
            logger.error(f"Model {model_id} failed: {e}")

    print("\n" + "="*80)
    print("ENSEMBLE RESULTS")
    print("="*80)
    try:
        response = orchestrator.search_ensemble(TEST_QUERY, top_k=5)
        print(f"Query time: {response.query_time_ms}ms\n")
        
        for i, result in enumerate(response.results, 1):
            is_rel = is_relevant(result.model_dump(), RELEVANT_TERMS)
            rel_marker = "RELEVANT" if is_rel else "NOT RELEVANT"
            print(f"  {i}. {result.product_name}")
            print(f"     Dept: {result.department} | Cat: {result.category}")
            print(f"     Score: {result.confidence_score} | {rel_marker}")
    except Exception as e:
        logger.error(f"Ensemble failed: {e}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()

