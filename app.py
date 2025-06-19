import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import joblib

# Load the dataset (optional preview)
@st.cache_data
def load_data():
    return pd.read_csv("final_load_weather_data.csv")

# ✅ Correct: Load model directly, not from a dict
@st.cache_resource
def load_model():
    return joblib.load("xgb_model_results.pkl")

# Main UI
st.title("⚡ Load Forecasting with Weather Conditions")

df = load_data()
model = load_model()

# Show a preview of data
if st.checkbox("Show raw dataset"):
    st.dataframe(df.head())

st.subheader("Enter weather parameters to predict load")

features = [
    'forecast_load', 'temp', 'temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed',
    'wind_deg', 'rain_1h', 'rain_3h', 'snow_3h', 'clouds_all', 'weather_id',
    'lag_1', 'lag_2', 'lag_3', 'rolling_mean_3'
]


input_data = {}
for feature in features:
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Predict
if st.button("Predict Load"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    st.success(f"Predicted Load: {prediction[0]:.2f} MW")
