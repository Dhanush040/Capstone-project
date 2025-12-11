# app.py
import streamlit as st
import pandas as pd
import os
from joblib import load

st.set_page_config(page_title="Insurance Response Prediction")

# ---------- Load model safely ----------
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "rf_insurance_model.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}. "
            f"Run main_rf.py first to train and save the model."
        )

    return load(model_path)


st.title("📊 Insurance Policy Response Prediction")
st.write("Random Forest model to predict if a customer will be interested in the policy.")

# Try loading model and show a friendly error if it fails
try:
    model = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

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
    "Region Code", min_value=0.0, max_value=52.0, value=26.0, step=1.0
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
    "Annual Premium", min_value=1000.0, max_value=100000.0,
    value=30000.0, step=500.0
)

policy_sales_channel = st.sidebar.number_input(
    "Policy Sales Channel", min_value=1.0, max_value=163.0,
    value=26.0, step=1.0
)

vintage = st.sidebar.slider(
    "Vintage (days with company)", 0, 300, 150
)

# ---------- Predict ----------
if st.button("Predict Response"):
    input_df = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Driving_License": driving_license,
        "Region_Code": region_code,
        "Previously_Insured": previously_insured,
        "Vehicle_Age": vehicle_age,
        "Vehicle_Damage": vehicle_damage,
        "Annual_Premium": annual_premium,
        "Policy_Sales_Channel": policy_sales_channel,
        "Vintage": vintage
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            f" Customer is **LIKELY INTERESTED** in the policy.\n\n"
            f"**Probability: {probability:.2%}**"
        )
    else:
        st.info(
            f" Customer is **NOT LIKELY INTERESTED** in the policy.\n\n"
            f"**Probability: {probability:.2%}**"
        )
