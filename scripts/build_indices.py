"""CLI script: Build all FAISS indices and model artifacts from processed data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from src.core.logging_config import setup_logging
from src.data.loader import load_processed_dataset
from src.data.enricher import enrich_descriptions
from src.models.orchestrator import SearchOrchestrator
from src.utils.graph_builder import export_hierarchy_json, build_hierarchy_graph

logger = setup_logging()


def main():
    processed_path = settings.processed_data_path / "products_cleaned.csv"
    enriched_path = settings.processed_data_path / "products_enriched.csv"

    if not processed_path.exists():
        logger.error("No processed data found at %s. Run data preprocessing first.", processed_path)
        sys.exit(1)

    logger.info("Loading processed dataset")
    df = load_processed_dataset(processed_path)

    # Generate enriched if not exists
    if enriched_path.exists():
        enriched_df = load_processed_dataset(enriched_path)
    else:
        enriched_df = enrich_descriptions(df)
        from src.data.loader import save_processed_dataset
        save_processed_dataset(enriched_df, enriched_path)

    # Export hierarchy JSON
    logger.info("Exporting hierarchy JSON")
    G = build_hierarchy_graph(df)
    export_hierarchy_json(G, settings.processed_data_path / "hierarchy.json")

    # Build all model indices
    logger.info("Building all model indices")
    orchestrator = SearchOrchestrator()
    orchestrator.initialize_models(df, enriched_df)
    orchestrator.save_all(str(settings.models_path))

    logger.info("All indices built and saved to %s", settings.models_path)


if __name__ == "__main__":
    main()
