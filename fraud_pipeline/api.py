"""FastAPI application for fraud detection serving."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from .config import load_config
from .utils.logger import setup_logger

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Credit Card Fraud Detection API using machine learning",
    version="1.0.0"
)

# Global variables for model and config
model = None
threshold = None
config = None
logger = None
scaler = None


class Transaction(BaseModel):
    """Transaction data model."""
    
    Time: float = Field(..., description="Time in seconds relative to first transaction")
    Amount: float = Field(..., description="Transaction amount")
    V1: float = Field(..., description="PCA component V1")
    V2: float = Field(..., description="PCA component V2")
    V3: float = Field(..., description="PCA component V3")
    V4: float = Field(..., description="PCA component V4")
    V5: float = Field(..., description="PCA component V5")
    V6: float = Field(..., description="PCA component V6")
    V7: float = Field(..., description="PCA component V7")
    V8: float = Field(..., description="PCA component V8")
    V9: float = Field(..., description="PCA component V9")
    V10: float = Field(..., description="PCA component V10")
    V11: float = Field(..., description="PCA component V11")
    V12: float = Field(..., description="PCA component V12")
    V13: float = Field(..., description="PCA component V13")
    V14: float = Field(..., description="PCA component V14")
    V15: float = Field(..., description="PCA component V15")
    V16: float = Field(..., description="PCA component V16")
    V17: float = Field(..., description="PCA component V17")
    V18: float = Field(..., description="PCA component V18")
    V19: float = Field(..., description="PCA component V19")
    V20: float = Field(..., description="PCA component V20")
    V21: float = Field(..., description="PCA component V21")
    V22: float = Field(..., description="PCA component V22")
    V23: float = Field(..., description="PCA component V23")
    V24: float = Field(..., description="PCA component V24")
    V25: float = Field(..., description="PCA component V25")
    V26: float = Field(..., description="PCA component V26")
    V27: float = Field(..., description="PCA component V27")
    V28: float = Field(..., description="PCA component V28")
    
    @validator('Amount')
    def validate_amount(cls, v):
        if v < 0:
            raise ValueError('Amount must be non-negative')
        return v
    
    @validator('Time')
    def validate_time(cls, v):
        if v < 0:
            raise ValueError('Time must be non-negative')
        return v


class PredictionResponse(BaseModel):
    """Prediction response model."""
    
    transaction_id: int
    probability: float = Field(..., ge=0.0, le=1.0, description="Fraud probability")
    decision: int = Field(..., ge=0, le=1, description="Fraud decision (0=legitimate, 1=fraud)")
    threshold: float = Field(..., description="Threshold used for decision")
    contributing_fields: Dict[str, float] = Field(..., description="Top contributing features")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    
    transactions: List[Transaction] = Field(..., min_items=1, max_items=1000)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    
    predictions: List[PredictionResponse]
    summary: Dict[str, Any] = Field(..., description="Summary statistics")


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    model_version: str = Field(..., description="Model version")
    threshold: float = Field(..., description="Current threshold")
    model_type: str = Field(..., description="Model type")
    features: List[str] = Field(..., description="Expected features")


def load_model_and_config():
    """Load the trained model and configuration."""
    global model, threshold, config, logger, scaler
    
    logger = setup_logger("fraud_api")
    
    try:
        # Load configuration
        config_path = os.getenv("CONFIG_PATH", "fraud_pipeline/configs/default.yaml")
        config = load_config(config_path)
        logger.info(f"Configuration loaded from: {config_path}")
        
        # Load model
        model_path = Path(config.data.models_path) / "best_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"Model loaded from: {model_path}")
        
        # Load scaler
        scaler_path = Path(config.data.processed_path) / "transformers" / "standard_scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from: {scaler_path}")
        else:
            scaler = None
            logger.warning("Scaler not found, using raw features")
        
        # Load threshold
        threshold_path = Path(config.data.models_path) / "threshold.json"
        if threshold_path.exists():
            with open(threshold_path, 'r') as f:
                threshold_data = json.load(f)
                threshold = threshold_data["threshold"]
                logger.info(f"Threshold loaded: {threshold}")
        else:
            threshold = config.serving.threshold
            logger.info(f"Using default threshold: {threshold}")
        
        logger.info("Model and configuration loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model and configuration: {e}")
        raise


def get_feature_importance(transaction_data: pd.DataFrame) -> Dict[str, float]:
    """Get feature importance for a transaction.
    
    Args:
        transaction_data: Transaction features as DataFrame
        
    Returns:
        Dictionary of feature names and their importance scores
    """
    try:
        # Get feature names
        feature_names = transaction_data.columns.tolist()
        
        # Get model feature importance if available
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
        elif hasattr(model.named_steps['classifier'], 'coef_'):
            # For linear models, use absolute coefficients
            importances = np.abs(model.named_steps['classifier'].coef_[0])
        else:
            # Fallback: use random importance
            importances = np.random.rand(len(feature_names))
        
        # Create feature importance dictionary
        feature_importance = dict(zip(feature_names, importances))
        
        # Sort by importance and return top 5
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:5])
        
    except Exception as e:
        logger.warning(f"Error getting feature importance: {e}")
        return {"Amount": 1.0, "Time": 0.5}


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    load_model_and_config()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get model type
    model_type = type(model.named_steps['classifier']).__name__
    
    # Get expected features
    features = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    
    return HealthResponse(
        status="healthy",
        model_version="1.0.0",
        threshold=threshold,
        model_type=model_type,
        features=features
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(transaction: Transaction):
    """Predict fraud for a single transaction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert transaction to DataFrame
        transaction_dict = transaction.dict()
        transaction_df = pd.DataFrame([transaction_dict])
        
        # Apply preprocessing to match the training data format
        if scaler is not None:
            # Scale Amount and Time features separately (as done in Stage 4)
            amount_scaled = scaler.transform(transaction_df[['Amount']].values.reshape(-1, 1))
            transaction_df['Amount_scaled'] = amount_scaled.flatten()
            
            # For Time, we need to check if it was included in training
            if hasattr(model.named_steps['scaler'], 'feature_names_in_') and 'Time_scaled' in model.named_steps['scaler'].feature_names_in_:
                time_scaled = scaler.transform(transaction_df[['Time']].values.reshape(-1, 1))
                transaction_df['Time_scaled'] = time_scaled.flatten()
            
            # Drop original Amount and Time columns
            transaction_df = transaction_df.drop(['Amount', 'Time'], axis=1)
        else:
            # If no scaler, create dummy scaled columns
            transaction_df['Amount_scaled'] = transaction_df['Amount']
            if hasattr(model.named_steps['scaler'], 'feature_names_in_') and 'Time_scaled' in model.named_steps['scaler'].feature_names_in_:
                transaction_df['Time_scaled'] = transaction_df['Time']
            transaction_df = transaction_df.drop(['Amount', 'Time'], axis=1)
        
        # Get prediction probability
        probability = model.predict_proba(transaction_df)[0, 1]
        
        # Make decision based on threshold
        decision = 1 if probability >= threshold else 0
        
        # Get contributing fields
        contributing_fields = get_feature_importance(transaction_df)
        
        return PredictionResponse(
            transaction_id=0,  # Single transaction
            probability=float(probability),
            decision=decision,
            threshold=threshold,
            contributing_fields=contributing_fields
        )
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict fraud for multiple transactions."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert transactions to DataFrame
        transactions_data = [t.dict() for t in request.transactions]
        transactions_df = pd.DataFrame(transactions_data)
        
        # Apply preprocessing to match the training data format
        if scaler is not None:
            # Scale Amount and Time features separately (as done in Stage 4)
            amount_scaled = scaler.transform(transactions_df[['Amount']].values.reshape(-1, 1))
            transactions_df['Amount_scaled'] = amount_scaled.flatten()
            
            # For Time, we need to check if it was included in training
            if hasattr(model.named_steps['scaler'], 'feature_names_in_') and 'Time_scaled' in model.named_steps['scaler'].feature_names_in_:
                time_scaled = scaler.transform(transactions_df[['Time']].values.reshape(-1, 1))
                transactions_df['Time_scaled'] = time_scaled.flatten()
            
            # Drop original Amount and Time columns
            transactions_df = transactions_df.drop(['Amount', 'Time'], axis=1)
        else:
            # If no scaler, create dummy scaled columns
            transactions_df['Amount_scaled'] = transactions_df['Amount']
            if hasattr(model.named_steps['scaler'], 'feature_names_in_') and 'Time_scaled' in model.named_steps['scaler'].feature_names_in_:
                transactions_df['Time_scaled'] = transactions_df['Time']
            transactions_df = transactions_df.drop(['Amount', 'Time'], axis=1)
        
        # Get prediction probabilities
        probabilities = model.predict_proba(transactions_df)[:, 1]
        
        # Make decisions based on threshold
        decisions = (probabilities >= threshold).astype(int)
        
        # Create predictions
        predictions = []
        for i, (prob, decision) in enumerate(zip(probabilities, decisions)):
            # Get contributing fields for this transaction
            transaction_df = transactions_df.iloc[[i]]
            contributing_fields = get_feature_importance(transaction_df)
            
            predictions.append(PredictionResponse(
                transaction_id=i,
                probability=float(prob),
                decision=int(decision),
                threshold=threshold,
                contributing_fields=contributing_fields
            ))
        
        # Create summary
        summary = {
            "total_transactions": len(predictions),
            "fraud_count": sum(p.decision for p in predictions),
            "fraud_rate": sum(p.decision for p in predictions) / len(predictions),
            "avg_probability": np.mean([p.probability for p in predictions]),
            "min_probability": min(p.probability for p in predictions),
            "max_probability": max(p.probability for p in predictions)
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error making batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict_batch",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
