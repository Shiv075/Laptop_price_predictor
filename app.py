#app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model and dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("💻 Laptop Price Predictor:")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of Laptop
laptop_type = st.selectbox('Type', df['TypeName'].unique())

# RAM
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the laptop (in kg)', min_value=0.5, max_value=5.0, step=0.1)

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS Display
ips = st.selectbox('IPS Display', ['No', 'Yes'])

# Screen Size
screen_size = st.number_input('Screen Size (in inches)', min_value=10.0, max_value=20.0, step=0.1)

# Resolution
resolution = st.selectbox(
    'Screen Resolution',
    ['1920x1080', '1366x768', '1600x900', '3840x2160',
     '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440']
)

# CPU
cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())

# HDD
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048, 4096])

# SSD
ssd = st.selectbox('SSD (in GB)', [0,8,32,64,128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())

# OS
os = st.selectbox('Operating System', df['os'].unique())


# Helper function to calculate PPI
def calculate_ppi(resolution, screen_size):
    x_res, y_res = map(int, resolution.split('x'))
    ppi = ((x_res ** 2 + y_res ** 2) ** 0.5) / screen_size
    return ppi


if st.button('Predict Price'):
    # Convert categorical to numerical values
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0
    ppi = calculate_ppi(resolution, screen_size)

    # Create a DataFrame for prediction
    query = pd.DataFrame({
        'Company': [company],
        'TypeName': [laptop_type],
        'Ram': [ram],
        'Weight': [weight],
        'Touchscreen': [touchscreen_val],
        'Ips': [ips_val],
        'ppi': [ppi],
        'Cpu brand': [cpu],
        'HDD': [hdd],
        'SSD': [ssd],
        'Gpu brand': [gpu],
        'os': [os]
    })

    # Predict
    predicted_log_price = pipe.predict(query)[0]
    predicted_price = np.exp(predicted_log_price)

    st.markdown(f"### 💰 Predicted Laptop Price: ₹ {int(predicted_price):,}")
    st.success("Prediction complete ✅")




