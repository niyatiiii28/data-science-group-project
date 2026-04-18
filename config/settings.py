"""Application settings loaded from environment variables."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # --- Paths ---
    base_dir: Path = BASE_DIR
    raw_data_path: Path = Field(default=BASE_DIR / "data" / "raw")
    processed_data_path: Path = Field(default=BASE_DIR / "data" / "processed")
    evaluation_data_path: Path = Field(default=BASE_DIR / "data" / "evaluation")
    models_path: Path = Field(default=BASE_DIR / "models")
    logs_path: Path = Field(default=BASE_DIR / "logs")

    # --- Embedding Model ---
    embedding_model_name: str = "all-MiniLM-L12-v2"
    embedding_dimension: int = 384

    # --- FAISS ---
    faiss_index_type: str = "IVFFlat"
    faiss_nlist: int = 100
    faiss_nprobe: int = 10

    # --- Flask ---
    flask_host: str = "0.0.0.0"
    flask_port: int = 5000
    flask_debug: bool = True
    secret_key: str = "change-this-in-production"

    # --- Rate Limiting ---
    rate_limit_default: str = "100/minute"

    # --- LLM (Model 4) ---
    openai_api_key: str = ""
    openai_base_url: str = "https://openrouter.ai/api/v1"
    llm_model_name: str = "openai/gpt-3.5-turbo"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Search ---
    default_top_k: int = 5
    max_top_k: int = 50

    model_config = {
        "env_file": str(BASE_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
