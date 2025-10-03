# Step 1: Environment Setup - Libraries install karein aur Drive mount karein
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Folder banaein
import os
folder_path = '/content/drive/MyDrive/electricity_app'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print("Folder bana diya: /content/drive/MyDrive/electricity_app")

# Step 2: Dummy Dataset Banaein (5 sources ka data - real data replace karo)
import pandas as pd
import numpy as np
data = {
    'sun_temperature': np.random.uniform(20, 50, 500),  # Sunlight temperature in Celsius
    'water_flow': np.random.uniform(1, 10, 500),       # Water flow in m3/s
    'wind_speed': np.random.uniform(5, 25, 500),       # Wind speed in m/s
    'plastic_mass': np.random.uniform(100, 1000, 500), # Plastic mass in kg
    'geothermal_delta_t': np.random.uniform(50, 200, 500), # Geothermal delta T in Celsius
    'electricity_output': np.random.uniform(100, 1000, 500)  # Electricity output in kW (simulated)
}  # Real data: Use P = A * r * H * PR (solar), η * ρ * g * h * Q (hydro) formulas
df = pd.DataFrame(data)
df.to_csv('/content/drive/MyDrive/electricity_app/electricity_db.csv', index=False)
print("Dataset saved: /content/drive/MyDrive/electricity_app/electricity_db.csv")

# Step 3: Model Train Karein (Random Forest Regression)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('/content/drive/MyDrive/electricity_app/electricity_db.csv')
X = df.drop('electricity_output', axis=1)
y = df['electricity_output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model R2 Score: {accuracy*100:.2f}%")

with open('/content/drive/MyDrive/electricity_app/model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved: /content/drive/MyDrive/electricity_app/model.pkl")

# Step 4: Streamlit App Banaein (app.py)

import streamlit as st
import pickle
import numpy as np

# Load model
@st.cache_resource
def load_model():
    with open('/mount/src/electricity-generation-predictor/model.pkl', 'rb') as f:
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

# Step 5: Requirements File Banaein
streamlit
scikit-learn
pandas
numpy

# Step 6: Files Download Karo (GitHub ke liye)
from google.colab import files
files.download('/content/drive/MyDrive/electricity_app/app.py')
files.download('/content/drive/MyDrive/electricity_app/model.pkl')
files.download('/content/drive/MyDrive/electricity_app/electricity_db.csv')
files.download('/content/drive/MyDrive/electricity_app/requirements.txt')
