 # CUSTOMER SEGMENTATION APP (CLEAN VERSION)

import streamlit as st
import numpy as np
import joblib

# Load model & scaler

model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title(" Customer Segmentation App")

# Sidebar Inputs
st.sidebar.header("Enter Customer Data")

credit_limit = st.sidebar.slider("Avg Credit Limit", 0, 100000, 50000)
cards = st.sidebar.slider("Total Credit Cards", 0, 10, 3)
bank = st.sidebar.slider("Bank Visits", 0, 20, 2)
online = st.sidebar.slider("Online Visits", 0, 20, 5)
calls = st.sidebar.slider("Calls Made", 0, 20, 1)

# Predict Button
if st.button("Predict Segment"):

    # Feature Engineering
    total_visits = bank + online + calls
    online_ratio = online / (total_visits + 1)
    credit_per_card = credit_limit / (cards + 1)
    engagement_score = total_visits * cards

    # Input Data
    data = np.array([[
        credit_limit,
        cards,
        bank,
        online,
        calls,
        total_visits,
        online_ratio,
        credit_per_card,
        engagement_score
    ]])

    # Scaling
    data_scaled = scaler.transform(data)

    # Prediction
    prediction = model.predict(data_scaled)

    # Output
    st.success(f"Predicted Customer Segment: {prediction[0]}")