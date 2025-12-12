# app.py
import streamlit as st
import pandas as pd
import os
from pathlib import Path
from joblib import load
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rf_insurance_model.joblib")
ENC_PATH = os.path.join(BASE_DIR, "encoders.joblib")  # optional

st.set_page_config(page_title="Insurance Policy Response Prediction", layout="wide")
st.title("📊 Insurance Policy Response Prediction")
st.write("Random Forest model to predict if a customer will be interested in the policy.")

# ---------- Load model (cached) ----------
@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at: {path}. "
            "Run your training script to create and save the model first."
        )
    return load(path)

# ---------- Try loading model and optional encoders ----------
try:
    model = load_model(MODEL_PATH)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

encoders = None
if os.path.exists(ENC_PATH):
    try:
        encoders = load(ENC_PATH)
    except Exception:
        st.warning("Found encoders file but failed to load it — proceeding without encoders.")

st.sidebar.header("Customer Details")

# ---------- Inputs (must match training columns) ----------
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

age = st.sidebar.slider("Age", 18, 85, 30)

driving_license = st.sidebar.selectbox(
    "Driving License",
    [1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

region_code = st.sidebar.number_input(
    "Region Code", min_value=0.0, max_value=999.0, value=26.0, step=1.0
)

previously_insured = st.sidebar.selectbox(
    "Previously Insured",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

vehicle_age = st.sidebar.selectbox(
    "Vehicle Age",
    ["< 1 Year", "1-2 Year", "> 2 Years"]
)

vehicle_damage = st.sidebar.selectbox(
    "Vehicle Damage",
    ["Yes", "No"]
)

annual_premium = st.sidebar.number_input(
    "Annual Premium", min_value=0.0, max_value=1_000_000.0,
    value=30000.0, step=500.0
)

policy_sales_channel = st.sidebar.number_input(
    "Policy Sales Channel", min_value=1.0, max_value=1000.0,
    value=26.0, step=1.0
)

vintage = st.sidebar.slider(
    "Vintage (days with company)", 0, 1000, 150
)

# ---------- Prepare input dataframe ----------
FEATURES = [
    "Gender", "Age", "Driving_License", "Region_Code", "Previously_Insured",
    "Vehicle_Age", "Vehicle_Damage", "Annual_Premium",
    "Policy_Sales_Channel", "Vintage"
]

def build_input_df():
    df = pd.DataFrame([{
        "Gender": gender,
        "Age": int(age),
        "Driving_License": int(driving_license),
        "Region_Code": float(region_code),
        "Previously_Insured": int(previously_insured),
        "Vehicle_Age": vehicle_age,
        "Vehicle_Damage": vehicle_damage,
        "Annual_Premium": float(annual_premium),
        "Policy_Sales_Channel": float(policy_sales_channel),
        "Vintage": int(vintage)
    }])
    # Ensure column order
    return df[FEATURES].copy()

# ---------- Predict ----------
if st.button("Predict Response"):
    input_df = build_input_df()

    # If encoders are available, apply them to matching columns
    if encoders:
        try:
            for col, le in encoders.items():
                if col in input_df.columns:
                    val = input_df.at[0, col]
                    # If unseen, append to classes_ so transform won't error
                    if val not in le.classes_:
                        le.classes_ = np.append(le.classes_, val)
                    input_df[col] = le.transform([val])
        except Exception as e:
            st.warning(f"Error applying encoders: {e}. Proceeding with raw values.")

    # Some models require numeric dtype — ensure numeric columns are numeric
    numeric_cols = ["Age", "Driving_License", "Region_Code", "Previously_Insured",
                    "Annual_Premium", "Policy_Sales_Channel", "Vintage"]
    for c in numeric_cols:
        if c in input_df.columns:
            input_df[c] = pd.to_numeric(input_df[c], errors="coerce").fillna(0)

    # Predict (handle models without predict_proba)
    try:
        prediction = int(model.predict(input_df)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    probability = None
    if hasattr(model, "predict_proba"):
        try:
            probability = float(model.predict_proba(input_df)[0][1])
        except Exception:
            probability = None

    # ---------- Show result ----------
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("✅ Customer is LIKELY INTERESTED in the policy.")
    else:
        st.info("ℹ️ Customer is NOT LIKELY INTERESTED in the policy.")

    if probability is not None:
        st.write(f"**Probability (positive class):** {probability:.2%}")
        st.progress(min(max(probability, 0.0), 1.0))
    else:
        st.write("Probability score not available for this model.")
