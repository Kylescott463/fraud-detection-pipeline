.PHONY: install format lint typecheck train serve test report clean bootstrap kaggle-setup kaggle-test download eda splits evaluate tune help

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install dependencies"
	@echo "  bootstrap    - Bootstrap Python and Kaggle environment"
	@echo "  kaggle-setup - Set up Kaggle credentials"
	@echo "  kaggle-test  - Test Kaggle CLI and API"
	@echo "  download     - Download Credit Card Fraud dataset"
	@echo "  eda          - Generate EDA report"
	@echo "  splits       - Create train/test splits"
	@echo "  evaluate     - Evaluate models and select best"
	@echo "  tune         - Tune hyperparameters with Optuna"
	@echo "  serve        - Start FastAPI server"
	@echo "  format       - Format code with black"
	@echo "  lint         - Lint code with ruff"
	@echo "  typecheck    - Type check with mypy"
	@echo "  train        - Train models"
	@echo "  serve        - Start FastAPI server"
	@echo "  test         - Run tests"
	@echo "  report       - Generate reports"
	@echo "  clean        - Clean build artifacts"

# Install dependencies
install:
	python3 -m pip install -r requirements.txt
	python3 -m pip install -e .

# Bootstrap Python and Kaggle environment
bootstrap:
	@echo "🚀 Bootstrapping Python and Kaggle environment..."
	@if [ "$(OS)" = "Windows_NT" ]; then \
		echo "Windows detected. Please run:"; \
		echo "powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1"; \
	else \
		bash scripts/bootstrap.sh; \
	fi

# Set up Kaggle credentials (idempotent)
kaggle-setup:
	@echo "🔐 Setting up Kaggle credentials..."
	@if [ -f "kaggle.json" ]; then \
		mkdir -p ~/.kaggle; \
		mv kaggle.json ~/.kaggle/kaggle.json; \
		chmod 600 ~/.kaggle/kaggle.json; \
		echo "✅ Moved kaggle.json to ~/.kaggle/kaggle.json"; \
	else \
		echo "❌ kaggle.json not found in project root"; \
		echo "💡 Please place kaggle.json in the project root and re-run"; \
		exit 1; \
	fi

# Test Kaggle CLI and API
kaggle-test:
	@echo "🔍 Testing Kaggle CLI and API..."
	python3 scripts/verify_kaggle.py

# Download dataset
download:
	@echo "📥 Downloading Credit Card Fraud dataset..."
	python3 scripts/download_data.py --config fraud_pipeline/configs/default.yaml

# Generate EDA report
eda:
	@echo "📊 Generating EDA report..."
	python3 scripts/eda_report.py --config fraud_pipeline/configs/default.yaml

# Create data splits
splits:
	@echo "✂️  Creating train/test splits..."
	python3 scripts/make_splits.py --config fraud_pipeline/configs/default.yaml

# Evaluate models
evaluate:
	@echo "🎯 Evaluating models and selecting best..."
	python3 scripts/evaluate.py --config fraud_pipeline/configs/default.yaml

# Tune hyperparameters
tune:
	@echo "🔧 Tuning hyperparameters with Optuna..."
	python3 scripts/tune.py --config fraud_pipeline/configs/default.yaml --trials 50

# Start API server
serve:
	@echo "🚀 Starting FastAPI server..."
	python3 scripts/serve.py --config fraud_pipeline/configs/default.yaml

# Format code
format:
	black fraud_pipeline/ scripts/ tests/
	ruff --fix fraud_pipeline/ scripts/ tests/

# Lint code
lint:
	ruff check fraud_pipeline/ scripts/ tests/

# Type check
typecheck:
	mypy fraud_pipeline/ scripts/

# Train models
train:
	@echo "🤖 Training fraud detection models..."
	python3 scripts/train.py --config fraud_pipeline/configs/default.yaml

# Start FastAPI server
serve:
	python scripts/serve.py --config fraud_pipeline/configs/default.yaml

# Run tests
test:
	pytest tests/ -v --cov=fraud_pipeline --cov-report=term-missing

# Run monitoring
monitor:
	@echo "🔍 Running data drift monitoring..."
	python3 scripts/monitor.py --input data/raw/creditcard.csv --config fraud_pipeline/configs/default.yaml

# Generate reports
report:
	python scripts/generate_report.py --config fraud_pipeline/configs/default.yaml

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
