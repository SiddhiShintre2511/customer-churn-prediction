"""Streamlit dashboard for customer churn prediction."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
DATA_PATH = ROOT / "data" / "customer_churn.csv"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")
st.title("📉 Customer Churn Prediction")
st.write("Predict whether a customer may leave a service using a trained classification model.")

if not MODEL_PATH.exists():
    st.warning("Model not found. First run `python train_model.py` in the project terminal.")
    st.stop()

model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)
features = [c for c in data.columns if c != "Churn" and c not in ["CustomerID", "customerID", "customer_id"]]
example = data[features].iloc[0]

st.subheader("Predict churn for a customer")
left, right = st.columns(2)
input_values = {}
for idx, col in enumerate(features):
    val = example[col]
    target_col = left if idx % 2 == 0 else right
    if pd.api.types.is_numeric_dtype(data[col]):
        numeric = pd.to_numeric(data[col], errors="coerce").dropna()
        lo = float(numeric.min()) if not numeric.empty else 0.0
        hi = float(numeric.max()) if not numeric.empty else max(lo + 1, 1.0)
        default = float(val) if pd.notna(val) else lo
        if lo == hi:
            hi = lo + 1.0
        input_values[col] = target_col.number_input(col, min_value=lo, max_value=hi, value=min(max(default, lo), hi))
    else:
        options = data[col].dropna().astype(str).unique().tolist()
        if not options:
            options = ["Unknown"]
        default = str(val) if pd.notna(val) and str(val) in options else options[0]
        input_values[col] = target_col.selectbox(col, options, index=options.index(default))

if st.button("Predict Customer Churn", type="primary"):
    row = pd.DataFrame([input_values], columns=features)
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1]) if hasattr(model, "predict_proba") else None
    if prediction == 1:
        st.error("Prediction: This customer may churn.")
    else:
        st.success("Prediction: This customer is likely to stay.")
    if probability is not None:
        st.metric("Estimated churn probability", f"{probability:.1%}")
        st.caption("This is a model estimate, not a guarantee. Validate decisions with real business context.")

st.divider()
st.subheader("Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(data))
c2.metric("Features", len(features))
churn_rate = data["Churn"].astype(str).str.lower().isin(["yes", "1", "true"]).mean()
c3.metric("Sample churn rate", f"{churn_rate:.1%}")
st.dataframe(data.head(10), use_container_width=True, hide_index=True)

if METRICS_PATH.exists():
    st.subheader("Model evaluation")
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)
    st.write("Selected model:", metrics.get("selected_model", "Not recorded"))
    st.json(metrics.get("test_metrics", {}))
