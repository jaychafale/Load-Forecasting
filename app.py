import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from PIL import Image

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("final_load_weather_data.csv")

# Load the trained XGBoost model
@st.cache_resource
def load_model():
    return joblib.load("xgb_model_results.pkl")

# Load the energy source recommender model
@st.cache_resource
def load_energy_model():
    return joblib.load("energy_recommender.pkl")

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🔬 Load Predictor", "📊 Dashboard"])

# Load models and data
model = load_model()
energy_model = load_energy_model()
df = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# 🔬 Predictor Page
# ─────────────────────────────────────────────────────────────────────────────
if page == "🔬 Load Predictor":
    st.title("⚡ Load Forecasting with Weather Conditions")

    if st.checkbox("📂 Show raw dataset"):
        st.dataframe(df.head())

    st.subheader("📥 Enter Weather Parameters")

    # 5 Inputs shown to user
    features_user = ['temp', 'humidity', 'wind_speed', 'dew_point', 'solar_radiation']
    user_input = {}
    for f in features_user:
        user_input[f] = st.number_input(f"{f.capitalize()}", value=0.0)

    # Predict Button
    if st.button("🔮 Predict Load"):
        try:
            # Extract inputs
            temp = user_input['temp']
            humidity = user_input['humidity']
            wind_speed = user_input['wind_speed']
            dew_point = user_input['dew_point']
            solar_radiation = user_input['solar_radiation']

            # Auto-generate remaining model-required features
            input_full = {
                'forecast_load': 0,
                'temp': temp,
                'temp_min': temp - 1.5,
                'temp_max': temp + 1.5,
                'pressure': 1013,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'wind_deg': 180,
                'rain_1h': 0.0,
                'rain_3h': 0.0,
                'snow_3h': 0.0,
                'clouds_all': 30,
                'weather_id': 800,
                'lag_1': temp - 2,
                'lag_2': temp - 3,
                'lag_3': temp - 4,
                'rolling_mean_3': temp - 1
            }

            input_df = pd.DataFrame([input_full])
            prediction = model.predict(input_df)
            st.success(f"📈 Predicted Load: **{prediction[0]:.2f} MW**")

            # Predict best energy source using RandomForest
            energy_input = pd.DataFrame([{
                'temp': temp,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'dew_point': dew_point,
                'solar_radiation': solar_radiation
            }])
            recommended = energy_model.predict(energy_input)[0]

            icons = {
                "solar": "☀️",
                "wind": "🌬",
                "both": "⚡",
                "none": "🔋"
            }
            st.info(f"{icons[recommended]} **Recommended Source:** {recommended.capitalize()} energy")

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

# ─────────────────────────────────────────────────────────────────────────────
# 📊 Dashboard Page
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Dashboard":
    st.title("📊 Power BI Dashboard Snapshot")

    try:
        image = Image.open("image.jpg")
        st.image(image, caption="Static Dashboard View", use_column_width=True)

        with open("image.jpg", "rb") as file:
            st.download_button(
                label="📥 Download Dashboard Image",
                data=file,
                file_name="powerbi_dashboard.jpg",
                mime="image/jpeg"
            )
    except Exception:
        st.error("🚫 Failed to load dashboard image.")
