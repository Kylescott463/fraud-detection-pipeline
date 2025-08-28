"""Setup script for fraud detection pipeline."""

from setuptools import setup, find_packages

setup(
    name="fraud-pipeline",
    version="0.1.0",
    description="Credit Card Fraud Detection Pipeline",
    author="Fraud Detection Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "imbalanced-learn>=0.9.0",
        "matplotlib>=3.5.0",
        "mlflow>=1.28.0",
        "pydantic>=1.10.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.20.0",
        "joblib>=1.2.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "requests>=2.28.0",
        "xgboost>=1.6.0",
        "optuna>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "ruff>=0.0.270",
            "mypy>=1.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ]
    },
)
