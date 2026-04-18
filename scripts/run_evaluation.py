"""CLI script: Run evaluation benchmarks and generate report."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from src.core.logging_config import setup_logging
from src.data.loader import load_processed_dataset
from src.models.orchestrator import SearchOrchestrator
from src.evaluation.benchmarks import run_benchmarks, format_report

logger = setup_logging()


def main():
    processed_path = settings.processed_data_path / "products_cleaned.csv"
    if not processed_path.exists():
        logger.error("No processed data found. Run preprocessing first.")
        sys.exit(1)

    df = load_processed_dataset(processed_path)

    enriched_path = settings.processed_data_path / "products_enriched.csv"
    enriched_df = load_processed_dataset(enriched_path) if enriched_path.exists() else df

    # Load models
    orchestrator = SearchOrchestrator()
    try:
        orchestrator.load_all(str(settings.models_path), df, enriched_df)
    except Exception:
        logger.info("Loading models from scratch")
        orchestrator.initialize_models(df, enriched_df)

    # Run benchmarks
    report = run_benchmarks(orchestrator, top_k=5)

    # Print formatted report
    text_report = format_report(report)
    print(text_report)

    # Save to results/
    results_dir = settings.base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "evaluation_report.txt", "w") as f:
        f.write(text_report)

    with open(results_dir / "evaluation_report.json", "w") as f:
        json.dump(report.model_dump(), f, indent=2)

    logger.info("Results saved to %s", results_dir)


if __name__ == "__main__":
    main()
