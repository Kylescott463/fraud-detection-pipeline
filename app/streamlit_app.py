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
METRICS_PATH = Path("data/reports/test_metrics.json")
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


def load_test_metrics() -> Optional[Dict]:
    """Load test metrics if available."""
    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None


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


def _generate_template_csv() -> str:
    """Generate template CSV with headers only."""
    template_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    return template_df.to_csv(index=False)


def _generate_scenario_vector(scenario: str) -> List[float]:
    """Generate deterministic V1-V28 vector based on scenario."""
    np.random.seed(42)  # Base seed for consistency
    
    # Scenario-specific seeds
    scenario_seeds = {
        "Typical Purchase": 100,
        "High Amount": 200,
        "Rapid Swipes": 300,
        "Anomalous Pattern": 400
    }
    
    np.random.seed(scenario_seeds.get(scenario, 100))
    return list(np.random.normal(0, 1, 28))


def _score_df(model: object, threshold: float, df: pd.DataFrame) -> pd.DataFrame:
    """Score DataFrame and return results with summary stats."""
    scored_df = score_batch(model, threshold, df)
    
    # Display results
    st.subheader("Scoring Results")
    st.dataframe(scored_df.head(), use_container_width=True)
    
    # Summary stats
    fraud_count = scored_df["fraud_decision"].sum()
    fraud_rate = fraud_count / len(scored_df) * 100
    
    st.success(f"Flagged {fraud_count} of {len(scored_df)} transactions ({fraud_rate:.1f}%).")
    
    return scored_df


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Fraud Detection Demo",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Fraud Detection Demo")
    
    # How to use banner
    st.markdown("---")
    st.markdown("### 📋 How to use")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1)** Try Single Prediction")
    with col2:
        st.markdown("**2)** Or go to Batch Scoring → Upload CSV (or click Download Sample CSV)")
    with col3:
        st.markdown("**3)** Click Download Scored CSV to save results")
    
    st.info("Threshold chosen from the Precision-Recall curve to catch ≥90% of fraud.")
    st.markdown("---")
    
    # Load artifacts
    model, threshold = load_artifacts()
    
    # Load test metrics for model card
    test_metrics = load_test_metrics()
    
    # Sidebar with model card
    with st.sidebar:
        st.header("📊 Model Card")
        
        if test_metrics:
            metrics_at_threshold = test_metrics.get("metrics_at_threshold", {})
            st.metric("Threshold", f"{threshold:.4f}")
            st.metric("Recall", f"{metrics_at_threshold.get('recall', 0):.3f}")
            st.metric("Precision", f"{metrics_at_threshold.get('precision', 0):.3f}")
            
            # Best model metrics
            best_model = test_metrics.get("best_model", "unknown")
            if best_model in test_metrics.get("test_metrics", {}):
                model_metrics = test_metrics["test_metrics"][best_model]
                st.metric("PR-AUC", f"{model_metrics.get('pr_auc', 0):.3f}")
                st.metric("ROC-AUC", f"{model_metrics.get('roc_auc', 0):.3f}")
        
        st.info("**Policy:** ≥90% recall target")
        
        st.markdown("---")
        st.markdown("### ⚠️ Limitations")
        st.markdown("• V1-V28 features are encoded (not human-interpretable)")
        st.markdown("• Trained on imbalanced Kaggle dataset")
        st.markdown("• Results best when new data resembles training distribution")
    
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
        
        # Simple vs Advanced toggle
        mode = st.radio(
            "Mode",
            ["Simple", "Advanced"],
            horizontal=True,
            key="prediction_mode"
        )
        
        if mode == "Simple":
            # Simple mode inputs
            with st.form("simple_prediction"):
                col1, col2 = st.columns(2)
                
                with col1:
                    time = st.number_input(
                        "Time", 
                        min_value=0, 
                        value=1000, 
                        step=1,
                        help="Seconds since dataset start. Example: 1000 ≈ 16 minutes."
                    )
                    amount = st.number_input(
                        "Amount", 
                        min_value=0.0, 
                        value=100.0, 
                        step=0.01,
                        help="Transaction amount in USD. Try $1–$5000."
                    )
                
                with col2:
                    scenario = st.selectbox(
                        "Scenario",
                        ["Typical Purchase", "High Amount", "Rapid Swipes", "Anomalous Pattern"],
                        help="Choose a transaction scenario to generate realistic V-features"
                    )
                
                submitted = st.form_submit_button("Predict Fraud", type="primary")
                
                if submitted:
                    # Generate V-features based on scenario
                    v_values = _generate_scenario_vector(scenario)
                    v_inputs = {f"V{i}": v_values[i-1] for i in range(1, 29)}
                    
                    # Store in session state for reference
                    st.session_state["simple_v"] = v_inputs
                    
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
                    
                    # Show generated V-features
                    with st.expander("Generated V-Features"):
                        st.json(v_inputs)
        
        else:
            # Advanced mode - existing V1-V28 inputs
            with st.form("advanced_prediction"):
                col1, col2 = st.columns(2)
                
                with col1:
                    time = st.number_input("Time", min_value=0, value=0, step=1)
                    amount = st.number_input("Amount", min_value=0.0, value=0.0, step=0.01)
                
                # V1-V28 inputs in expander
                with st.expander("Encoded Features (V1..V28)"):
                    st.caption("Encoded (PCA-like) features; not required in Simple mode.")
                    v_inputs = {}
                    cols = st.columns(4)
                    for i, v_name in enumerate([f"V{i}" for i in range(1, 29)]):
                        col_idx = i % 4
                        with cols[col_idx]:
                            v_inputs[v_name] = st.number_input(
                                v_name, 
                                value=0.0, 
                                step=0.01,
                                format="%.4f",
                                key=f"advanced_{v_name}"
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
        
        # Initialize session state
        if "batch_source" not in st.session_state:
            st.session_state["batch_source"] = None
        if "batch_df" not in st.session_state:
            st.session_state["batch_df"] = None
        
        # Dataset options
        col1, col2, col3 = st.columns(3)
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
            template_csv = _generate_template_csv()
            st.download_button(
                label="📋 Download Template CSV (headers only)",
                data=template_csv,
                file_name="template_transactions.csv",
                mime="text/csv"
            )
        with col3:
            if st.button("🎯 Use Sample Dataset", type="secondary", key="use_sample"):
                st.session_state["batch_source"] = "sample"
                st.session_state["batch_df"] = sample_df
                st.rerun()
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="CSV must contain columns: Time, Amount, V1, V2, ..., V28"
        )
        
        # Handle uploaded file (takes precedence over sample)
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                st.session_state["batch_source"] = "upload"
                st.session_state["batch_df"] = uploaded_df
                if st.session_state.get("batch_source") == "sample":
                    st.info("📁 Using uploaded file. Click 'Use Sample Dataset' again to switch.")
            except Exception as e:
                st.error(f"Error processing file: {e}")
        
        # Show preview and schema check if data is available
        if st.session_state["batch_df"] is not None:
            df = st.session_state["batch_df"]
            source = st.session_state["batch_source"]
            
            # Schema check
            st.subheader("📋 Schema Check")
            is_valid, error_msg = validate_input_data(df)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if is_valid:
                    st.success("✅ Required columns present")
                else:
                    st.error("❌ Missing columns")
            with col2:
                extra_cols = set(df.columns) - set(REQUIRED_COLUMNS)
                if extra_cols:
                    st.info(f"↩️ {len(extra_cols)} extra columns ignored")
                else:
                    st.info("↩️ No extra columns")
            with col3:
                st.info(f"🔢 {len(df)} rows")
            with col4:
                # Calculate predicted positive rate at current threshold
                try:
                    features = df[REQUIRED_COLUMNS].values
                    probs = model.predict_proba(features)[:, 1]
                    positive_rate = (probs >= threshold).mean() * 100
                    st.info(f"📊 {positive_rate:.1f}% predicted positive")
                except:
                    st.info("📊 Rate calculation failed")
            
            if not is_valid:
                st.error(error_msg)
                st.info(f"Required columns: {', '.join(REQUIRED_COLUMNS)}")
            else:
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Initialize session state for scored results
                if "scored_df" not in st.session_state:
                    st.session_state["scored_df"] = None
                
                # Scoring form
                with st.form(key="batch_form"):
                    submitted = st.form_submit_button("Score transactions", type="primary")
                    
                    if submitted:
                        with st.spinner("Scoring transactions..."):
                            st.session_state["scored_df"] = _score_df(model, threshold, df)
                
                # Download button (outside form)
                if st.session_state["scored_df"] is not None:
                    scored_df = st.session_state["scored_df"]
                    csv = scored_df.to_csv(index=False)
                    st.download_button(
                        label="Download Scored CSV",
                        data=csv,
                        file_name="fraud_scores.csv",
                        mime="text/csv"
                    )
        else:
            st.info("Load a dataset (upload or sample) first.")


if __name__ == "__main__":
    main()
