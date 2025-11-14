# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# --- Load saved objects ---
best_model = joblib.load("models/wind_power_model.pkl") # best model (XGBoost or Linear)
# preprocess_input = joblib.load("models/preprocess_input.pkl") # preprocessing function
metadata = joblib.load("models/model_metadata.pkl") # features + metrics

# --- Preprocessing function ---
def preprocess_input(df):
    df = df.copy()
    df['Wind_Speed_Cubed'] = df['Wind Speed (m/s)'] ** 3
    df['Dir_Sin'] = np.sin(np.radians(df['Wind Direction (°)']))
    df['Dir_Cos'] = np.cos(np.radians(df['Wind Direction (°)']))
    df['Rolling_Speed'] = df['Wind Speed (m/s)'] # for single-point prediction
    return df[['Wind Speed (m/s)', 'Wind_Speed_Cubed', 'Dir_Sin', 'Dir_Cos', 'Rolling_Speed']]

# --- Sidebar for Model Info ---
with st.sidebar:
    st.header("Model Details")
    st.write("Features used:", metadata["features"])
    st.write("Performance Metrics:")
    for model_name, metrics in metadata["metrics"].items():
        st.metric(label=f"{model_name} R²", value=f"{metrics['R2']:.3f}")
        st.write(f"RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}")

# --- App Title ---
st.title("Wind Turbine Power Prediction System")
st.image("turbine_header.jpg", caption="Wind Turbine Power Forecasting", use_column_width=True)
st.markdown("Enter operational parameters to estimate turbine power output.")

# --- User Inputs with Columns ---
col1, col2 = st.columns(2)
with col1:
    wind_speed = st.slider("Wind Speed (m/s)", min_value=0.0, max_value=25.0, value=10.0, step=0.1)
with col2:
    wind_direction = st.slider("Wind Direction (°)", min_value=0.0, max_value=360.0, value=180.0, step=1.0)

# --- Prediction ---
if st.button("Predict Power Output"):
    input_df = pd.DataFrame({
        'Wind Speed (m/s)': [wind_speed],
        'Wind Direction (°)': [wind_direction]
    })
    # Apply preprocessing
    final_features = preprocess_input(input_df)
    prediction = max(0, best_model.predict(final_features)[0])
    
    # Simple theoretical power calculation (defaults: air_density=1.225, swept_area=4000, efficiency=0.4)
    theoretical = 0.5 * 1.225 * 4000 * (wind_speed ** 3) * 0.4 / 1000  # in kW
    
    # Display prediction
    st.subheader(f"Predicted Power Output: {prediction:.2f} kW")
    
    # Status based on prediction with optional color formatting
    if prediction < 300:
        st.markdown("<span style='color:red'>LOW → Possible Underperformance</span>", unsafe_allow_html=True)
    elif prediction < 1500:
        st.markdown("<span style='color:orange'>NORMAL → Stable Operation</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:green'>HIGH → Peak Performance</span>", unsafe_allow_html=True)
    
    # Simple Bar Plot: Predicted vs Theoretical
    categories = ['Predicted', 'Theoretical']
    values = [prediction, theoretical]
    colors = ['#1f77b4', '#2ca02c']  # Blue and green for consistency
    bars = ax.bar(categories, values, color=colors)
    ax.set_ylabel('Power (kW)')
    ax.set_title('Power Output Comparison')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f'{value:.0f}', ha='center', va='bottom')
    plt.tight_layout()
    st.pyplot(fig)

# --- Optional: Image Placeholder (add your turbine image here if available) ---
# st.image("turbine_image.jpg", caption="Wind Turbine Example", use_column_width=True)
