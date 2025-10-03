import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import streamlit as st

# Step 1: Dummy Dataset Banaein (5 sources ka data - real data replace karo)
data = {
    'sun_temperature': np.random.uniform(20, 50, 500),  # Sunlight temperature in Celsius
    'water_flow': np.random.uniform(1, 10, 500),       # Water flow in m3/s
    'wind_speed': np.random.uniform(5, 25, 500),       # Wind speed in m/s
    'plastic_mass': np.random.uniform(100, 1000, 500), # Plastic mass in kg
    'geothermal_delta_t': np.random.uniform(50, 200, 500), # Geothermal delta T in Celsius
    'electricity_output': np.random.uniform(100, 1000, 500)  # Electricity output in kW (simulated)
}  # Real data: Use P = A * r * H * PR (solar), η * ρ * g * h * Q (hydro) formulas
df = pd.DataFrame(data)
# Dataset save karne ke liye alag se file mein rakho (GitHub pe CSV upload karo)

# Step 2: Model Train Karein (Random Forest Regression)
X = df.drop('electricity_output', axis=1)
y = df['electricity_output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Model save karne ke liye alag se file mein rakho (GitHub pe model.pkl upload karo)

# Step 3: Streamlit App (Yeh part Streamlit Cloud pe chalega)
# Load model (GitHub pe upload kiya hua model load hoga)
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

# Main app
st.title("Electricity Generation Predictor")
st.write("Enter values for 5 sources to predict electricity generation in kW.")

model = load_model()

# Inputs
sun_temp = st.slider("Sunlight Temperature (Celsius)", 20, 50, 30)
water_flow = st.slider("Water Flow (m³/s)", 1, 10, 5)
wind_speed = st.slider("Wind Speed (m/s)", 5, 25, 10)
plastic_mass = st.slider("Plastic Mass (kg)", 100, 1000, 500)
geothermal_delta = st.slider("Geothermal Delta T (Celsius)", 50, 200, 100)

# Prediction
input_data = np.array([[sun_temp, water_flow, wind_speed, plastic_mass, geothermal_delta]])
prediction = model.predict(input_data)[0]

st.subheader(f"Predicted Electricity Generation: {prediction:.2f} kW")
