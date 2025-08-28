# Credit Card Fraud Detection Pipeline

[![Live Demo](https://img.shields.io/badge/Streamlit-Demo-brightgreen?logo=streamlit)](https://fraud-detection-pipeline-kylescott463.streamlit.app/)

A production-ready fraud detection system using machine learning, built with Python, FastAPI, and MLflow.

## How to use (for non-technical folks)
1. Click the **Live Demo** button at the top.
2. In the app, choose **Single Prediction** to test one transaction, or go to **Batch Scoring** and **Upload CSV** (or click **Download Sample CSV** inside the app).
3. Click **Download Scored CSV** to save results.

**Policy:** The model's threshold was chosen from the Precision-Recall curve to catch **≥90% of fraud**.

## Run locally
```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

## Features

- **Data Processing**: Automated data ingestion and preprocessing
- **Model Training**: Multiple ML models with hyperparameter optimization
- **Model Serving**: FastAPI-based REST API for real-time predictions
- **Experiment Tracking**: MLflow integration for model versioning
- **Monitoring**: Comprehensive logging and metrics
- **Testing**: Unit tests and integration tests

## Project Structure

```
fraud_pipeline/
├── fraud_pipeline/          # Main package
│   ├── __init__.py
│   ├── configs/            # Configuration files
│   │   └── default.yaml
│   ├── data/               # Data processing modules
│   ├── models/             # Model training and serving
│   └── utils/              # Utility functions
├── scripts/                # CLI scripts
├── notebooks/              # Jupyter notebooks
├── data/                   # Data storage
│   ├── raw/               # Raw data files
│   ├── processed/         # Processed data
│   ├── models/            # Trained models
│   └── reports/           # Generated reports
├── tests/                 # Test files
├── pyproject.toml         # Project configuration
├── requirements.txt       # Dependencies
├── Makefile              # Build automation
└── README.md             # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd fraud_detection_pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install
```

### 2. Pre-Stage 2: Kaggle Bootstrap

Before downloading data, you need to set up the Kaggle CLI and credentials:

#### macOS/Linux:
```bash
# Bootstrap Python and Kaggle environment
make bootstrap

# Test Kaggle setup
make kaggle-test
```

**Note:** You'll need to download `kaggle.json` from your Kaggle account settings and place it in the project root. See `kaggle.json.example` for the expected format.

#### Windows:
```powershell
# Bootstrap Python and Kaggle environment
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1

# Test Kaggle setup
python scripts/verify_kaggle.py
```

**Note:** You'll need to download `kaggle.json` from your Kaggle account settings and place it in the project root. See `kaggle.json.example` for the expected format.

**Note:** If PATH was updated during bootstrap, restart your shell or run:
- macOS/Linux: `source ~/.zshrc` or `source ~/.bashrc`
- Windows: Restart PowerShell

### 3. Configure Environment

```bash
# Copy environment template
cp env.example .env

# Edit .env with your Kaggle credentials
# KAGGLE_USERNAME=your_username
# KAGGLE_KEY=your_api_key
```

### 4. Download Data

```bash
# Download credit card fraud dataset
make download

# Or run manually
python scripts/download_data.py --config fraud_pipeline/configs/default.yaml
```

### 5. Generate EDA Report

```bash
# Generate comprehensive EDA report
make eda

# Or run manually
python scripts/eda_report.py --config fraud_pipeline/configs/default.yaml
```

The EDA report includes:
- 📊 Dataset overview and summary statistics
- 📈 Class distribution analysis with visualizations
- 🔥 Correlation heatmap for top features
- 💰 Amount and Time distribution histograms
- 🔍 Data quality and leakage checks
- 📋 Complete data types summary

**Report Location:** `data/reports/eda.html`

### 6. Create Data Splits

```bash
# Create train/test splits with preprocessing
make splits

# Or run manually
python scripts/make_splits.py --config fraud_pipeline/configs/default.yaml
```

The preprocessing pipeline includes:
- 📊 Feature engineering (V1-V28 + scaled Amount/Time)
- ✂️ Stratified 80/20 train/test split
- ⚖️ Imbalanced handling (none/class_weight/under/smote)
- 💾 Saved artifacts (parquet files + transformers)

**Configuration Options:**
- `sampling.strategy`: `none`, `class_weight`, `under`, `smote`
- `sampling.include_time`: Include scaled Time feature
- `seed`: Random seed for reproducibility

**Output Files:**
- `data/processed/X_train.parquet` - Training features
- `data/processed/X_test.parquet` - Test features  
- `data/processed/y_train.parquet` - Training targets
- `data/processed/y_test.parquet` - Test targets
- `data/processed/transformers/standard_scaler.joblib` - Fitted scaler
- `data/processed/transformers/sampler.joblib` - Fitted sampler (if used)

### 7. Train Models

```bash
# Train all baseline models
make train

# Or run manually
python3 scripts/train.py --config fraud_pipeline/configs/default.yaml
```

The training pipeline includes:
- 🤖 **Baseline Models**: LogisticRegression, RandomForest, XGBoost (if available)
- 🔧 **Sklearn Pipelines**: Scaler + Classifier with optional sampling
- 📊 **Cross-Validation**: 5-fold StratifiedKFold for robust evaluation
- 📈 **Comprehensive Metrics**: PR-AUC, ROC-AUC, Precision@k, Recall@k, F1, Confusion Matrix
- 📝 **MLflow Integration**: Complete experiment tracking and model logging
- 💾 **Model Persistence**: Saved pipelines for inference

**Models Trained:**
- **Logistic Regression**: Linear model with liblinear solver
- **Random Forest**: Ensemble with 100 trees, max_depth=10
- **XGBoost**: Gradient boosting (if OpenMP runtime available)

**Metrics Tracked:**
- **Primary**: PR-AUC (Precision-Recall Area Under Curve)
- **Secondary**: ROC-AUC, Precision, Recall, F1-Score
- **Business**: Precision@10%, Recall@10%, Precision@5%, Recall@5%
- **Validation**: 5-fold CV with mean ± std reporting

**Output Files:**
- `data/models/{model_name}.joblib` - Trained pipelines
- `data/models/{model_name}_confusion_matrix.csv` - Confusion matrices
- `data/models/{model_name}_classification_report.csv` - Detailed reports
- `mlflow.db` - MLflow tracking database

**MLflow Tracking:**
- **Experiment**: `fraud_detection`
- **Tracking URI**: `sqlite:///mlflow.db`
- **Logged**: Parameters, metrics, artifacts, model files

```bash
# Train all models
make train

# Or run with custom config
python scripts/train.py --config fraud_pipeline/configs/default.yaml
```

### 8. Evaluate Models and Select Best

```bash
# Evaluate all models and select the best one
make evaluate

# Or run with custom config
python3 scripts/evaluate.py --config fraud_pipeline/configs/default.yaml
```

This stage:
- **Loads trained models** and evaluates them on the held-out test set
- **Selects best model** by PR-AUC (Precision-Recall Area Under Curve)
- **Optimizes threshold** to achieve ≥90% recall while maximizing precision
- **Performs probability calibration** using Platt scaling if it improves Brier score
- **Generates comprehensive reports** including:
  - Model performance comparison table
  - PR and ROC curves for all models
  - Confusion matrix at optimal threshold
  - Final evaluation report in Markdown
  - Optimal threshold saved for serving

#### Output Files
- `data/reports/test_metrics.json` - Complete evaluation metrics
- `data/reports/final_report.md` - Comprehensive evaluation report
- `data/reports/model_curves.png` - PR and ROC curves for all models
- `data/reports/best_model_pr_curve.png` - PR curve with optimal threshold
- `data/models/threshold.json` - Optimal threshold for serving

#### Example Results
```
🏆 BEST MODEL SELECTION
Selected Model: random_forest
PR-AUC Score: 0.9920

⚙️  THRESHOLD OPTIMIZATION
Optimal Threshold: 0.2461
Precision at Threshold: 1.0000
Recall at Threshold: 0.9082
F1-Score at Threshold: 0.9519
```

### 9. Tune Hyperparameters

```bash
# Tune hyperparameters for best models
make tune

# Or run with custom config and trials
python3 scripts/tune.py --config fraud_pipeline/configs/default.yaml --trials 50
```

This stage:
- **Uses Optuna** for hyperparameter optimization (fallback to RandomizedSearchCV if not available)
- **Tunes top models** from evaluation stage (Random Forest, Logistic Regression)
- **Maximizes PR-AUC** via 3-fold stratified cross-validation
- **Comprehensive search spaces**:
  - **Logistic Regression**: C, penalty (l1/l2/elasticnet), l1_ratio, solver
  - **Random Forest**: n_estimators, max_depth, max_features, min_samples_split, min_samples_leaf
- **Logs results to MLflow** with parameters, metrics, and artifacts
- **Saves best model** as `data/models/best_model.joblib`

#### Output Files
- `data/models/best_model.joblib` - Best tuned model pipeline
- `data/models/best_{model_name}.joblib` - Individual best models
- `data/models/tuning_summary.json` - Tuning results summary
- MLflow runs with complete optimization history

#### Example Results
```
🏆 BEST MODEL SELECTION
Selected Model: random_forest
Test PR-AUC: 0.6991

📊 RANDOM_FOREST
Best CV PR-AUC: 0.7417
Best Parameters:
  n_estimators: 200
  max_depth: None
  max_features: log2
  min_samples_split: 2
  min_samples_leaf: 1
```

### 10. Start API Server

```bash
# Start the FastAPI server
make serve

# Or run with custom config
python3 scripts/serve.py --config fraud_pipeline/configs/default.yaml --host 0.0.0.0 --port 8000
```

The API server provides:
- **Health check**: `GET /health`
- **Single prediction**: `POST /predict`
- **Batch prediction**: `POST /predict_batch`
- **Interactive docs**: `http://localhost:8000/docs`

#### Example API Usage

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"Time": 0.0, "Amount": 149.62, "V1": -1.36, ...}'

# Batch prediction
curl -X POST "http://localhost:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d '{"transactions": [{"Time": 0.0, "Amount": 149.62, ...}]}'
```

#### Example Response

```json
{
  "transaction_id": 0,
  "probability": 0.22,
  "decision": 0,
  "threshold": 0.2461,
  "contributing_fields": {
    "V14": 0.1718,
    "V12": 0.1150,
    "V10": 0.1047,
    "V17": 0.0917,
    "V4": 0.0880
  }
}
```

### 11. Batch Scoring

```bash
# Score a batch of transactions from CSV
python3 scripts/score_batch.py \
  --input data/raw/creditcard.csv \
  --output data/reports/batch_scored.parquet \
  --config fraud_pipeline/configs/default.yaml
```

This creates a Parquet file with original data plus:
- `fraud_probability`: Model prediction probability
- `fraud_decision`: Binary decision (0/1)
- `threshold`: Threshold used for decision

### 12. Docker Deployment

```bash
# Build Docker image
docker build -t fraud-detection-api .

# Run container
docker run -p 8000:8000 fraud-detection-api

# Or with custom config
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  fraud-detection-api
```

### 13. Make Predictions

```bash
# Test the API
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [1.0, 2.0, 3.0, ...]}'
```

## Configuration

The pipeline uses YAML configuration files located in `fraud_pipeline/configs/`. Key settings:

- **Data paths**: Raw data, processed data, and model storage locations
- **Model parameters**: Hyperparameters for each ML model
- **Sampling strategy**: How to handle imbalanced data (SMOTE, undersampling, etc.)
- **Serving threshold**: Probability threshold for fraud detection
- **MLflow settings**: Experiment tracking configuration

### Default Configuration

```yaml
# Random seed for reproducibility
seed: 42

# Sampling strategy for imbalanced data
sampling:
  strategy: "smote"  # Options: smote, undersample, oversample, none

# Model serving threshold
serving:
  threshold: 0.5
  min_probability: 0.1
```

## Available Commands

### Development

```bash
make format      # Format code with black and ruff
make lint        # Lint code with ruff
make typecheck   # Type check with mypy
make test        # Run tests
```

### Pipeline Operations

```bash
make bootstrap   # Bootstrap Python and Kaggle environment
make kaggle-test # Test Kaggle CLI and API
make download    # Download Credit Card Fraud dataset
make eda         # Generate EDA report
make splits      # Create train/test splits
make train       # Train all models
make evaluate    # Evaluate models and select best
make tune        # Tune hyperparameters with Optuna
make serve       # Start API server
make report      # Generate performance reports
```

### Maintenance

```bash
make clean       # Clean build artifacts
make install     # Install dependencies
```

## API Endpoints

### Health Check
- `GET /health` - Check API status

### Predictions
- `POST /predict` - Make fraud predictions
- `POST /predict_batch` - Batch predictions

### Model Management
- `GET /models` - List available models
- `GET /models/{model_name}` - Get model info
- `POST /models/{model_name}/predict` - Predict with specific model

## EDA Report

The EDA (Exploratory Data Analysis) report provides comprehensive insights into the Credit Card Fraud dataset:

### Key Insights

- **Dataset Size:** 284,807 transactions with 31 features
- **Class Imbalance:** 99.83% legitimate vs 0.17% fraudulent (1:577.9 ratio)
- **Data Quality:** 1,081 duplicate rows detected
- **Memory Usage:** ~144MB CSV, ~70MB Parquet (50% compression)

### Generated Visualizations

1. **Class Distribution Bar Plot** - Shows the severe class imbalance
2. **Correlation Heatmap** - Top 15 features correlated with fraud detection
3. **Amount Histograms** - Transaction amount distribution by class
4. **Time Histograms** - Transaction timing patterns by class

### Data Quality Checks

- ✅ No missing values detected
- ✅ No perfect correlations between features
- ✅ No zero variance features
- ⚠️ 1,081 duplicate rows (0.38% of dataset)

### Report Access

```bash
# Generate the report
make eda

# View in browser
open data/reports/eda.html
```

## Testing

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=fraud_pipeline --cov-report=html
```

## Monitoring

The pipeline includes data drift monitoring to detect when the distribution of incoming data changes significantly from the training data.

### Running Monitoring

```bash
# Monitor data drift
make monitor

# Or run directly
python scripts/monitor.py --input data/raw/creditcard.csv --config fraud_pipeline/configs/default.yaml

# Monitor with custom parameters
python scripts/monitor.py --input new_data.csv --alpha 0.01 --verbose
```

### Monitoring Report

The monitoring script generates:
- **HTML Report**: `data/reports/monitoring.html`
- **Drift Plots**: `data/reports/plots/drift_*.png`
- **Console Summary**: Drift statistics and recommendations

### Interpreting Drift Results

- **Low Drift (< 10%)**: Model appears stable, continue monitoring
- **Moderate Drift (10-20%)**: Monitor closely, consider retraining soon
- **High Drift (> 20%)**: Consider retraining the model

### Drift Detection Method

Uses Kolmogorov-Smirnov (KS) test to compare feature distributions:
- Compares new data vs training data for each numeric feature
- Significance level α = 0.05 (configurable)
- Reports KS statistic and p-value for each feature

## Production Checklist

Before deploying to production, ensure:

### ✅ Code Quality
- [ ] All tests pass: `make test`
- [ ] Code is formatted: `make format`
- [ ] Linting passes: `make lint`
- [ ] Type checking passes: `make typecheck`

### ✅ Data Pipeline
- [ ] Data ingestion works: `make download`
- [ ] EDA report generated: `make eda`
- [ ] Train/test splits created: `make splits`
- [ ] Models trained successfully: `make train`

### ✅ Model Performance
- [ ] Model evaluation completed: `make evaluate`
- [ ] Hyperparameter tuning done: `make tune`
- [ ] Best model selected and saved
- [ ] Threshold optimized for business requirements

### ✅ API & Deployment
- [ ] API starts successfully: `make serve`
- [ ] Health endpoint responds: `curl http://localhost:8000/health`
- [ ] Predictions work: Test with sample data
- [ ] Docker build succeeds: `docker build -t fraud-detection .`

### ✅ Monitoring Setup
- [ ] Monitoring script works: `make monitor`
- [ ] Drift detection configured
- [ ] Alerting thresholds set
- [ ] Monitoring reports accessible

### ✅ Security & Performance
- [ ] Input validation working
- [ ] Rate limiting configured (if needed)
- [ ] Authentication/authorization (if required)
- [ ] Performance benchmarks met
- [ ] Error handling tested

### ✅ Documentation
- [ ] README updated
- [ ] API documentation complete
- [ ] Deployment instructions clear
- [ ] Troubleshooting guide available

## Development

### Adding New Models

1. Add model configuration to `fraud_pipeline/configs/default.yaml`
2. Implement model class in `fraud_pipeline/models/`
3. Add tests in `tests/test_models.py`
4. Update training script if needed

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement changes with tests
3. Run quality checks: `make format lint typecheck test`
4. Submit pull request

## Troubleshooting

### Common Issues

1. **Import Error**: Run `make install` to install the package
2. **Kaggle API Error**: Check your credentials in `.env`
3. **Port Already in Use**: Change port in config or kill existing process
4. **Memory Issues**: Reduce batch size or model complexity

### Logs

- Application logs: `logs/app.log`
- MLflow logs: `mlflow.db`
- Model artifacts: `data/models/`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks
5. Submit pull request

## Live Demo

▶ **[Live Demo](https://your-streamlit-url-here)** - Interactive fraud detection demo

### Local Demo Setup

```bash
# Install demo dependencies
pip install -r app/requirements.txt

# Run Streamlit demo
streamlit run app/streamlit_app.py
```

The demo provides:
- **Single Prediction**: Enter transaction details for real-time fraud detection
- **Batch Scoring**: Upload CSV files for bulk transaction analysis
- **Threshold Policy**: Shows current threshold (≥90% recall) from PR curve optimization

Threshold chosen from PR curve to achieve ≥90% recall (see `data/reports/final_report.md`).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
