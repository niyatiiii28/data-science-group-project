# Data Science Group Project

A Python-based semantic search application for e-commerce product discovery, built around a Flask web app and vector search pipeline. The project processes a Flipkart product dataset, enriches product descriptions, and enables semantic product search using embeddings and similarity search.

## Overview

This repository contains a full stack for an intelligent product search engine that:

- loads and preprocesses product data,
- generates embeddings for product titles and descriptions,
- indexes products using FAISS,
- exposes a REST API and web interface for search,
- supports evaluation and comparison workflows.

The application is designed for data science and ML experimentation in a product search context, with a focus on semantic matching rather than keyword-only retrieval.

## Tech Stack

- Python
- Flask
- FAISS
- sentence-transformers
- scikit-learn
- pandas / NumPy
- PyTorch
- OpenAI-compatible LLM integration
- Docker / Docker Compose

## Project Structure

```text
.
├── app.py                     # Application entry point
├── Dockerfile                 # Container definition
├── docker-compose.yml         # Local multi-container setup
├── Makefile                   # Common project commands
├── requirements.txt           # Python dependencies
├── .env.example               # Sample environment configuration
├── config/
│   └── settings.py            # Application configuration
├── data/
│   ├── raw/                   # Raw dataset files
│   ├── processed/             # Cleaned and processed data
│   └── evaluation/            # Evaluation data
├── models/                    # Saved model artifacts and indexes
├── logs/                      # Runtime logs
├── results/                   # Output artifacts
├── scripts/                   # Utility scripts
├── src/
│   ├── api/                   # Flask app and API routes
│   ├── core/                  # Logging and shared utilities
│   ├── data/                  # Dataset loading and preprocessing
│   ├── models/                # Search orchestration and model logic
│   └── ...
├── templates/                 # HTML frontend templates
├── tests/                     # Test suite
├── evaluation_output.txt      # Evaluation output sample
├── spot_check.py              # Spot-check scripts
├── spot_check_output.txt      # Spot-check results
└── flipkart_com-ecommerce_sample.csv
```

## Features

- Semantic product search using embeddings
- Saved FAISS indices and model artifacts
- Data cleaning and enrichment pipeline
- Config-driven deployment with environment variables
- Flask API endpoints for product and search functionality
- Web UI templates for browsing and comparing products
- Docker setup for easier local deployment

## Prerequisites

Before running the project, make sure you have:

- Python 3.10+
- pip
- Docker (optional, for containerized setup)
- A dataset file in `data/raw/` (for example, a Flipkart product CSV)

## Setup

1. Clone the repository

```bash
git clone https://github.com/niyatiiii28/data-science-group-project.git
cd data-science-group-project
```

2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or .venv\Scripts\activate  # Windows
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Configure environment variables

```bash
cp .env.example .env
```

Update `.env` as needed for values such as API keys, model settings, and runtime configuration.

5. Add dataset

Place your product dataset CSV file in the `data/raw/` directory. The app expects a dataset similar to the Flipkart product catalog and will preprocess it automatically on first run.

## Run the Application

### Local Python run

```bash
python app.py
```

Then open:

- http://localhost:5000
- API endpoints under `/api`

### Docker run

```bash
docker compose up --build
```

## API Usage

The app exposes search and product-related endpoints via Flask. Typical usage includes:

- health checks
- product listing
- semantic search queries
- evaluation and comparison tasks

See the `src/api` modules for route definitions and request handling.

## Model Workflow

On startup, the app:

1. loads or creates the processed dataset,
2. enriches product descriptions,
3. initializes the search orchestrator,
4. builds model artifacts if they don't already exist,
5. serves the application and search API.

## Notes

- The project is configured to use a local `.env` file for environment variables.
- Some features may require external API keys or downloaded models depending on the selected embedding and LLM configuration.
- Large datasets can take time to preprocess and index on first run.

## License

This project does not currently declare a license file. If you plan to share or distribute it publicly, consider adding an appropriate open-source license.

## Contributing

If you are collaborating on this project, follow the repository workflow and keep environment settings and generated artifacts isolated from source control where appropriate.

## Authors

This repository is part of a data science group project focused on semantic search and product discovery.
