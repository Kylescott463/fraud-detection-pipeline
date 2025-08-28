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


@st.cache_data
def _generate_sample_df() -> pd.DataFrame:
    """Generate a 50-row realistic synthetic dataset."""
    np.random.seed(42)  # Deterministic for demo consistency
    
    # Time: random ints in [0, 172800)
    time_values = np.random.randint(0, 172800, 50)
    
    # Amount: log-normal-ish positive floats
    amount_values = np.round(np.random.lognormal(mean=4.5, sigma=1.2, size=50), 2)
    
    # V1-V28: standard normal with some fraud-like patterns
    v_values = np.random.normal(0, 1, (50, 28))
    
    # Add some fraud-like patterns to a minority of rows
    fraud_indices = np.random.choice(50, size=8, replace=False)  # ~16% fraud rate
    v_values[fraud_indices] += np.random.normal(0, 0.5, (8, 28))  # Slightly larger variance
    
    # Create DataFrame with exact column order
    data = {"Time": time_values, "Amount": amount_values}
    for i in range(1, 29):
        data[f"V{i}"] = v_values[:, i-1]
    
    return pd.DataFrame(data)


def _score_df(model: object, threshold: float, df: pd.DataFrame) -> pd.DataFrame:
    """Score DataFrame and return results with summary stats."""
    scored_df = score_batch(model, threshold, df)
    
    # Display results
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
    
    return scored_df


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Fraud Detection Demo",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Fraud Detection Demo")
    
    # Instruction banner
    st.markdown("---")
    st.markdown("### 📋 How to use")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1)** Try one transaction in Single Prediction")
    with col2:
        st.markdown("**2)** Or go to Batch Scoring and Upload CSV (or use our Sample CSV)")
    with col3:
        st.markdown("**3)** Click Download Scored CSV to save the results")
    
    st.info("Threshold chosen from the PR curve to achieve ≥90% recall.")
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
        
        # Sample dataset buttons
        col1, col2 = st.columns(2)
        with col1:
            sample_df = _generate_sample_df()
            csv_data = sample_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample CSV (50 rows)",
                data=csv_data,
                file_name="sample_transactions.csv",
                mime="text/csv"
            )
        with col2:
            use_sample = st.button("🎯 Use Sample Dataset", type="secondary")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="CSV must contain columns: Time, Amount, V1, V2, ..., V28"
        )
        
        # Handle data source (uploaded file takes precedence)
        df = None
        data_source = None
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                data_source = "uploaded"
                if use_sample:
                    st.info("📁 Using uploaded file (sample dataset ignored)")
            except Exception as e:
                st.error(f"Error processing file: {e}")
        elif use_sample:
            df = sample_df
            data_source = "sample"
            st.success("✅ Sample dataset loaded (50 rows)")
        
        # Process data if available
        if df is not None:
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
                        _score_df(model, threshold, df)


if __name__ == "__main__":
    main()
