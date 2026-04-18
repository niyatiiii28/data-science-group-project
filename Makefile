.PHONY: setup install generate-data build-index run test evaluate docker clean

# Setup virtualenv and install deps
setup:
	python -m venv venv
	./venv/Scripts/pip install -r requirements.txt
	./venv/Scripts/python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

# Install deps only
install:
	pip install -r requirements.txt

# Build all model indices from processed data
build-index:
	python scripts/build_indices.py

# Run the Flask server
run:
	python app.py

# Run tests
test:
	pytest tests/ -v --tb=short

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Run evaluation benchmarks
evaluate:
	python scripts/run_evaluation.py

# Docker build and run
docker:
	docker-compose up --build

# Clean generated files
clean:
	rm -rf models/model1/*.pkl models/model1/*.npy
	rm -rf models/model2/*.faiss models/model2/*.npy models/model2/*.pkl
	rm -rf models/model3/*.faiss models/model3/*.npy models/model3/*.pkl
	rm -rf models/model4/*.json
	rm -rf logs/*.log
	rm -rf __pycache__ src/__pycache__ **/__pycache__
