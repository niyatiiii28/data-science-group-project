"""Main entry point: loads data, initializes models, starts Flask server."""

import sys
import logging
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import settings
from src.core.logging_config import setup_logging
from src.data.loader import load_flipkart_dataset, load_processed_dataset, save_processed_dataset
from src.data.preprocessor import DataPreprocessor
from src.data.enricher import enrich_descriptions
from src.models.orchestrator import SearchOrchestrator
from src.api.app import create_app

logger = setup_logging()


def initialize_engine():
    """Load data, preprocess, build models, return orchestrator."""
    processed_path = settings.processed_data_path / "products_cleaned.csv"
    enriched_path = settings.processed_data_path / "products_enriched.csv"

    # Check for pre-processed data
    if processed_path.exists():
        logger.info("Loading pre-processed dataset")
        df = load_processed_dataset(processed_path)
    else:
        # Look for raw Flipkart CSV
        raw_files = list(settings.raw_data_path.glob("*.csv"))
        if not raw_files:
            logger.error(
                "No dataset found! Place the Flipkart CSV in %s",
                settings.raw_data_path,
            )
            raise FileNotFoundError(
                f"No CSV files found in {settings.raw_data_path}. "
                "Download the Flipkart Products dataset from Kaggle and place it there."
            )

        raw_path = raw_files[0]
        logger.info("Processing raw dataset: %s", raw_path)
        df = load_flipkart_dataset(raw_path)

        # Preprocess
        preprocessor = DataPreprocessor(df)
        df = preprocessor.run_full_pipeline()
        stats = preprocessor.get_stats()
        logger.info("Preprocessing stats: %s", stats)

        # Save processed
        save_processed_dataset(df, processed_path)

    # Enriched descriptions
    if enriched_path.exists():
        enriched_df = load_processed_dataset(enriched_path)
    else:
        enriched_df = enrich_descriptions(df)
        save_processed_dataset(enriched_df, enriched_path)

    # Initialize all models
    orchestrator = SearchOrchestrator()

    # Try loading pre-built indices first
    models_dir = settings.models_path
    try:
        orchestrator.load_all(str(models_dir), df, enriched_df)
        if orchestrator.get_loaded_models():
            logger.info("Loaded pre-built models: %s", orchestrator.get_loaded_models())
            orchestrator.df = df
            return orchestrator
    except Exception as e:
        logger.info("No pre-built models found, building from scratch: %s", e)

    # Build from scratch
    orchestrator.initialize_models(df, enriched_df)
    orchestrator.save_all(str(models_dir))

    return orchestrator


def main():
    logger.info("Starting Semantic Search Engine")

    orchestrator = initialize_engine()

    app = create_app()
    app.config["ORCHESTRATOR"] = orchestrator

    logger.info(
        "Server ready — %d products, %d models loaded",
        len(orchestrator.df),
        len(orchestrator.get_loaded_models()),
    )

    app.run(
        host=settings.flask_host,
        port=settings.flask_port,
        debug=settings.flask_debug,
    )


if __name__ == "__main__":
    main()