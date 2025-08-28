#!/usr/bin/env python3
"""
Fraud Detection Demo - Streamlit Application
Production-quality demo for non-technical reviewers.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path for model loading
sys.path.insert(0, str(Path(__file__).parent.parent))

# Constants
MODEL_PATH = Path("data/models/best_model.joblib")
THRESHOLD_PATH = Path("data/models/threshold.json")
REQUIRED_COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


@st.cache_resource
def load_artifacts() -> Tuple[object, float]:
    """Load model and threshold from saved artifacts."""
    try:
        model = joblib.load(MODEL_PATH)
        
        with open(THRESHOLD_PATH, "r") as f:
            threshold_data = json.load(f)
            threshold = threshold_data["threshold"]
        
        return model, threshold
    except FileNotFoundError as e:
        st.error(f"Model artifacts not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        st.stop()


def validate_input_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate input DataFrame has required columns."""
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    return True, ""


def predict_single(model: object, threshold: float, inputs: Dict) -> Tuple[float, int]:
    """Make single prediction and return probability and decision."""
    # Create feature array
    features = np.array([[inputs[col] for col in REQUIRED_COLUMNS]])
    
    # Get probability
    prob = model.predict_proba(features)[0][1]  # Probability of fraud class
    
    # Apply threshold
    decision = 1 if prob >= threshold else 0
    
    return prob, decision


def score_batch(model: object, threshold: float, df: pd.DataFrame) -> pd.DataFrame:
    """Score batch of transactions and return DataFrame with predictions."""
    # Select only required columns
    features = df[REQUIRED_COLUMNS].values
    
    # Get probabilities
    probs = model.predict_proba(features)[:, 1]  # Probability of fraud class
    
    # Apply threshold
    decisions = (probs >= threshold).astype(int)
    
    # Create result DataFrame
    result = df.copy()
    result["fraud_probability"] = probs
    result["fraud_decision"] = decisions
    
    return result


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Fraud Detection Demo",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Fraud Detection Demo")
    st.markdown("---")
    
    # Load artifacts
    model, threshold = load_artifacts()
    
    # Display threshold info
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info(f"**Threshold:** {threshold:.4f} | **Policy:** ≥90% recall")
    with col2:
        st.success("✅ Model loaded successfully")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Scoring"])
    
    with tab1:
        st.header("Single Transaction Prediction")
        st.markdown("Enter transaction details to get fraud probability and decision.")
        
        # Input form
        with st.form("single_prediction"):
            col1, col2 = st.columns(2)
            
            with col1:
                time = st.number_input("Time", min_value=0, value=0, step=1)
                amount = st.number_input("Amount", min_value=0.0, value=0.0, step=0.01)
            
            # V1-V28 inputs in a compact grid
            v_inputs = {}
            cols = st.columns(4)
            for i, v_name in enumerate([f"V{i}" for i in range(1, 29)]):
                col_idx = i % 4
                with cols[col_idx]:
                    v_inputs[v_name] = st.number_input(
                        v_name, 
                        value=0.0, 
                        step=0.01,
                        format="%.4f"
                    )
            
            submitted = st.form_submit_button("Predict Fraud", type="primary")
            
            if submitted:
                # Prepare inputs
                inputs = {"Time": time, "Amount": amount, **v_inputs}
                
                # Make prediction
                prob, decision = predict_single(model, threshold, inputs)
                
                # Display results
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Fraud Probability", f"{prob:.4f}")
                
                with col2:
                    decision_text = "🚨 FRAUD" if decision == 1 else "✅ LEGITIMATE"
                    st.metric("Decision", decision_text)
                
                with col3:
                    threshold_status = "Above" if prob >= threshold else "Below"
                    st.metric("Threshold Status", threshold_status)
                
                # Show inputs for verification
                with st.expander("Input Values"):
                    st.json(inputs)
    
    with tab2:
        st.header("Batch Transaction Scoring")
        st.markdown("Upload a CSV file with transaction data to score multiple transactions.")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="CSV must contain columns: Time, Amount, V1, V2, ..., V28"
        )
        
        if uploaded_file is not None:
            try:
                # Load CSV
                df = pd.read_csv(uploaded_file)
                
                # Validate columns
                is_valid, error_msg = validate_input_data(df)
                
                if not is_valid:
                    st.error(error_msg)
                    st.info(f"Required columns: {', '.join(REQUIRED_COLUMNS)}")
                else:
                    # Show preview
                    st.subheader("Data Preview")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Score batch
                    if st.button("Score Transactions", type="primary"):
                        with st.spinner("Scoring transactions..."):
                            scored_df = score_batch(model, threshold, df)
                        
                        # Show results
                        st.subheader("Scoring Results")
                        st.dataframe(scored_df.head(), use_container_width=True)
                        
                        # Download button
                        csv = scored_df.to_csv(index=False)
                        st.download_button(
                            label="Download Scored CSV",
                            data=csv,
                            file_name="fraud_scores.csv",
                            mime="text/csv"
                        )
                        
                        # Summary stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Transactions", len(scored_df))
                        with col2:
                            fraud_count = scored_df["fraud_decision"].sum()
                            st.metric("Fraudulent", fraud_count)
                        with col3:
                            fraud_rate = fraud_count / len(scored_df) * 100
                            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                
            except Exception as e:
                st.error(f"Error processing file: {e}")


if __name__ == "__main__":
    main()
