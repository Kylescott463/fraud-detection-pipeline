"""Tests for API endpoints."""

import pytest
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from fraud_pipeline.api import app


class TestAPI:
    """Test cases for API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    @patch('fraud_pipeline.api.model')
    @patch('fraud_pipeline.api.threshold')
    def test_health_endpoint_success(self, mock_threshold, mock_model):
        """Test health endpoint when model is loaded."""
        # Mock model and threshold
        mock_classifier = Mock()
        mock_classifier.__class__.__name__ = 'LogisticRegression'
        mock_model.named_steps = {'classifier': mock_classifier}
        mock_threshold.return_value = 0.5
        
        # Make request
        response = self.client.get("/health")
        
        # Assertions
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_version"] == "1.0.0"
        assert data["threshold"] == 0.5
        assert data["model_type"] == "LogisticRegression"
        assert "features" in data
        assert len(data["features"]) == 30  # Time, Amount, V1-V28
    
    @patch('fraud_pipeline.api.model')
    def test_health_endpoint_no_model(self, mock_model):
        """Test health endpoint when model is not loaded."""
        # Mock model as None
        mock_model = None
        
        # Make request
        response = self.client.get("/health")
        
        # Assertions
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data
        assert "Model not loaded" in data["detail"]
    
    @patch('fraud_pipeline.api.model')
    @patch('fraud_pipeline.api.threshold')
    def test_health_endpoint_model_metadata(self, mock_threshold, mock_model):
        """Test health endpoint returns correct model metadata."""
        # Mock model with different classifier
        mock_classifier = Mock()
        mock_classifier.__class__.__name__ = 'RandomForestClassifier'
        mock_model.named_steps = {'classifier': mock_classifier}
        mock_threshold.return_value = 0.7
        
        # Make request
        response = self.client.get("/health")
        
        # Assertions
        assert response.status_code == 200
        
        data = response.json()
        assert data["model_type"] == "RandomForestClassifier"
        assert data["threshold"] == 0.7
    
    def test_health_endpoint_features(self):
        """Test health endpoint returns correct feature list."""
        with patch('fraud_pipeline.api.model') as mock_model:
            mock_classifier = Mock()
            mock_classifier.__class__.__name__ = 'LogisticRegression'
            mock_model.named_steps = {'classifier': mock_classifier}
            
            with patch('fraud_pipeline.api.threshold', 0.5):
                response = self.client.get("/health")
                
                assert response.status_code == 200
                data = response.json()
                
                # Check features
                features = data["features"]
                assert "Time" in features
                assert "Amount" in features
                
                # Check V1-V28 features
                for i in range(1, 29):
                    assert f"V{i}" in features
                
                assert len(features) == 30  # Time + Amount + V1-V28
    
    def test_predict_endpoint_no_model(self):
        """Test predict endpoint when model is not loaded."""
        with patch('fraud_pipeline.api.model', None):
            # Create test transaction
            transaction = {
                "Time": 0.0,
                "Amount": 100.0,
                "V1": 0.1, "V2": 0.2, "V3": 0.3, "V4": 0.4, "V5": 0.5,
                "V6": 0.6, "V7": 0.7, "V8": 0.8, "V9": 0.9, "V10": 1.0,
                "V11": 1.1, "V12": 1.2, "V13": 1.3, "V14": 1.4, "V15": 1.5,
                "V16": 1.6, "V17": 1.7, "V18": 1.8, "V19": 1.9, "V20": 2.0,
                "V21": 2.1, "V22": 2.2, "V23": 2.3, "V24": 2.4, "V25": 2.5,
                "V26": 2.6, "V27": 2.7, "V28": 2.8
            }
            
            response = self.client.post("/predict", json=transaction)
            
            assert response.status_code == 503
            data = response.json()
            assert "detail" in data
            assert "Model not loaded" in data["detail"]
    
    def test_predict_endpoint_invalid_input(self):
        """Test predict endpoint with invalid input."""
        # Test with missing required fields
        invalid_transaction = {
            "Time": 0.0,
            "Amount": 100.0
            # Missing V1-V28 fields
        }
        
        response = self.client.post("/predict", json=invalid_transaction)
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_negative_amount(self):
        """Test predict endpoint with negative amount."""
        transaction = {
            "Time": 0.0,
            "Amount": -100.0,  # Negative amount should fail validation
            "V1": 0.1, "V2": 0.2, "V3": 0.3, "V4": 0.4, "V5": 0.5,
            "V6": 0.6, "V7": 0.7, "V8": 0.8, "V9": 0.9, "V10": 1.0,
            "V11": 1.1, "V12": 1.2, "V13": 1.3, "V14": 1.4, "V15": 1.5,
            "V16": 1.6, "V17": 1.7, "V18": 1.8, "V19": 1.9, "V20": 2.0,
            "V21": 2.1, "V22": 2.2, "V23": 2.3, "V24": 2.4, "V25": 2.5,
            "V26": 2.6, "V27": 2.7, "V28": 2.8
        }
        
        response = self.client.post("/predict", json=transaction)
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_negative_time(self):
        """Test predict endpoint with negative time."""
        transaction = {
            "Time": -1.0,  # Negative time should fail validation
            "Amount": 100.0,
            "V1": 0.1, "V2": 0.2, "V3": 0.3, "V4": 0.4, "V5": 0.5,
            "V6": 0.6, "V7": 0.7, "V8": 0.8, "V9": 0.9, "V10": 1.0,
            "V11": 1.1, "V12": 1.2, "V13": 1.3, "V14": 1.4, "V15": 1.5,
            "V16": 1.6, "V17": 1.7, "V18": 1.8, "V19": 1.9, "V20": 2.0,
            "V21": 2.1, "V22": 2.2, "V23": 2.3, "V24": 2.4, "V25": 2.5,
            "V26": 2.6, "V27": 2.7, "V28": 2.8
        }
        
        response = self.client.post("/predict", json=transaction)
        
        assert response.status_code == 422  # Validation error
    
    def test_api_documentation_endpoints(self):
        """Test that API documentation endpoints are accessible."""
        # Test OpenAPI schema
        response = self.client.get("/openapi.json")
        assert response.status_code == 200
        
        # Test docs endpoint
        response = self.client.get("/docs")
        assert response.status_code == 200
        
        # Test redoc endpoint
        response = self.client.get("/redoc")
        assert response.status_code == 200
    
    def test_api_root_endpoint(self):
        """Test API root endpoint."""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_api_endpoints_list(self):
        """Test that API returns correct list of endpoints."""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        endpoints = data.get("endpoints", {})
        
        # Check that expected endpoints are listed
        assert "health" in endpoints
        assert "predict" in endpoints
        assert "predict_batch" in endpoints
        assert "docs" in endpoints
