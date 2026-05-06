import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------------
# Load trained model (pipeline: preprocessor + linear regression)
# ------------------------------------------------------------------
model = joblib.load("shoes_sales_model.joblib")

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Footwear Sales Demand Predictor",
    layout="centered"
)

# ------------------------------------------------------------------
# App title and description
# ------------------------------------------------------------------
st.title("👟 Footwear Sales Demand Predictor")
st.write(
    "This application predicts the expected number of units sold "
    "based on product characteristics, pricing, and market inputs."
)

# ------------------------------------------------------------------
# User input widgets
# ------------------------------------------------------------------
brand = st.selectbox(
    "Brand",
    ["Adidas", "Nike", "Puma", "Reebok", "New Balance", "Skechers"]
)

shoe_type = st.selectbox(
    "Shoe Type",
    ["Sneakers", "Running", "Boots", "Casual", "Formal", "Sports"]
)

price_band = st.selectbox(
    "Price Band",
    ["Low", "Mid", "High", "Premium"]
)

country = st.selectbox(
    "Country",
    ["Germany", "USA", "UK", "UAE", "India", "Saudi Arabia"]
)

sales_channel = st.selectbox(
    "Sales Channel",
    ["Online", "Mall", "Retail Store"]
)

quarter = st.selectbox(
    "Quarter",
    [1, 2, 3, 4]
)

# ------------------------------------------------------------------
# Create input DataFrame (must match training features)
# ------------------------------------------------------------------
input_df = pd.DataFrame(
    {
        "Brand": [brand],
        "Shoe_Type": [shoe_type],
        "Price_Band": [price_band],
        "Country": [country],
        "Sales_Channel": [sales_channel],
        "Quarter": [quarter],
    }
)

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if st.button("Predict Units Sold"):
    prediction = model.predict(input_df)[0]
    st.success(f"✅ Predicted Units Sold: {prediction:.2f}")
