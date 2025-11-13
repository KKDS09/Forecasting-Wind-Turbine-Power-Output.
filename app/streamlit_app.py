# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# --- Load saved objects ---
best_model = joblib.load("models/wind_power_model.pkl")          # best model (XGBoost or Linear)
# preprocess_input = joblib.load("models/preprocess_input.pkl")    # preprocessing function
metadata = joblib.load("models/model_metadata.pkl")              # features + metrics

# --- Preprocessing function ---
def preprocess_input(df):
    df = df.copy()
    df['Wind_Speed_Cubed'] = df['Wind Speed (m/s)'] ** 3
    df['Dir_Sin'] = np.sin(np.radians(df['Wind Direction (°)']))
    df['Dir_Cos'] = np.cos(np.radians(df['Wind Direction (°)']))
    df['Rolling_Speed'] = df['Wind Speed (m/s)']  # for single-point prediction
    return df[['Wind Speed (m/s)', 'Wind_Speed_Cubed', 'Dir_Sin', 'Dir_Cos', 'Rolling_Speed']]

# --- App Title ---
st.title("Wind Turbine Power Prediction System")
st.markdown("Enter operational parameters to estimate turbine power output.")

# --- User Inputs ---
wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, step=0.1)
wind_direction = st.number_input("Wind Direction (°)", min_value=0.0, max_value=360.0, step=1.0)

# --- Prediction ---
if st.button("Predict Power Output"):
    input_df = pd.DataFrame({
        'Wind Speed (m/s)': [wind_speed],
        'Wind Direction (°)': [wind_direction]
    })

    # Apply preprocessing
    final_features = preprocess_input(input_df)
    prediction = max(0,best_model.predict(final_features)[0])

    # Display prediction
    st.subheader(f"Predicted Power Output: {prediction:.2f} kW")

    # Status based on prediction with optional color formatting
    if prediction < 300:
        st.markdown("<span style='color:red'>LOW → Possible Underperformance</span>", unsafe_allow_html=True)
    elif prediction < 1500:
        st.markdown("<span style='color:orange'>NORMAL → Stable Operation</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:green'>HIGH → Peak Performance</span>", unsafe_allow_html=True)

# --- Optional: Model Info ---
with st.expander("Model Information"):
    st.write("Features used:", metadata["features"])
    st.write("Performance Metrics:")
    for model_name, metrics in metadata["metrics"].items():
        st.write(f"{model_name}: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, R²={metrics['R2']:.3f}")

