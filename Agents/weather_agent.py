# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 12:13:15 2025

@author: FMT COMPUTERS
"""

import requests
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import xgboost
import streamlit as st
from datetime import datetime, timedelta
import warnings
import os
import matplotlib.pyplot as plt
import pytz

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Islamabad Weather Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for a modern dashboard look
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .card {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 15px;
        margin: 10px;
        text-align: center;
    }
    .header {
        background-color: #2c3e50;
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
    .metric {
        font-size: 20px;
        font-weight: bold;
        color: #2c3e50;
    }
    .subheader {
        color: #7f8c8d;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Open-Meteo API endpoint for Islamabad (latitude: 33.6844, longitude: 73.0479)
latitude = 33.6844
longitude = 73.0479
url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,cloud_cover,uv_index&forecast_days=1&timezone=auto"

# Fetch current weather data
@st.cache_data(ttl=3600)  # Cache for 1 hour to reduce API calls
def fetch_weather_data():
    try:
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"Failed to fetch data from Open-Meteo: {response.status_code}")
            return None
        return response.json()
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

data = fetch_weather_data()
if data is None:
    st.stop()

# Extract hourly data with None checks
hours = data['hourly'].get('time', [])
temperatures = data['hourly'].get('temperature_2m', [])
precipitations = data['hourly'].get('precipitation', [])
humidities = data['hourly'].get('relative_humidity_2m', [])
wind_speeds = data['hourly'].get('wind_speed_10m', [])
cloud_covers = data['hourly'].get('cloud_cover', [])
uv_indices = data['hourly'].get('uv_index', [])

# Get current time dynamically in PKT
pk_time = pytz.timezone('Asia/Karachi')
current_time = pk_time.localize(datetime.now()).replace(minute=0, second=0, microsecond=0)  # Round to nearest hour

# Find the index of the closest hour in the API data
def find_closest_hour(api_hours, current_time):
    api_times = [datetime.fromisoformat(t.replace('Z', '+00:00')).astimezone(pk_time) for t in api_hours]
    return min(range(len(api_times)), key=lambda i: abs(api_times[i] - current_time))

closest_hour_idx = find_closest_hour(hours, current_time)
if closest_hour_idx >= len(temperatures):
    closest_hour_idx = len(temperatures) - 1  # Fallback to last available hour

# Current weather based on the closest hour
current_weather = {
    'Time': current_time.strftime('%Y-%m-%d %H:%M'),
    'Temperature (°C)': temperatures[closest_hour_idx] if temperatures and len(temperatures) > closest_hour_idx else "N/A",
    'Precipitation (mm)': precipitations[closest_hour_idx] if precipitations and len(precipitations) > closest_hour_idx else "N/A",
    'Humidity (%)': humidities[closest_hour_idx] if humidities and len(humidities) > closest_hour_idx else "N/A",
    'Wind Speed (km/h)': wind_speeds[closest_hour_idx] if wind_speeds and len(wind_speeds) > closest_hour_idx and wind_speeds[closest_hour_idx] is not None else "N/A",
    'Cloud Cover (%)': cloud_covers[closest_hour_idx] if cloud_covers and len(cloud_covers) > closest_hour_idx else "N/A",
    'UV Index': uv_indices[closest_hour_idx] if uv_indices and len(uv_indices) > closest_hour_idx and uv_indices[closest_hour_idx] is not None else "N/A"
}

# Create DataFrame for the next 8 hours starting from the closest hour
start_idx = closest_hour_idx
end_idx = min(closest_hour_idx + 8, len(temperatures))
df = pd.DataFrame({
    'time': [pk_time.localize(datetime.fromisoformat(t.replace('Z', '+00:00'))) for t in hours[start_idx:end_idx]],
    'temperature_2m (°C)': [t if t is not None else 0.0 for t in temperatures[start_idx:end_idx]],
    'precipitation (mm)': [p if p is not None else 0.0 for p in precipitations[start_idx:end_idx]],
    'relative_humidity_2m (%)': [h if h is not None else 0.0 for h in humidities[start_idx:end_idx]],
    'wind_speed_10m (km/h)': [w if w is not None else 0.0 for w in wind_speeds[start_idx:end_idx]],
    'cloud_cover (%)': [c if c is not None else 0.0 for c in cloud_covers[start_idx:end_idx]],
    'uv_index': [u if u is not None else 0.0 for u in uv_indices[start_idx:end_idx]]
})

# Load the saved model from .pt file
@st.cache_resource
def load_model():
    model_path = r"C:\Users\FMT COMPUTERS\Downloads\Code Paper 3\xgboost_weather_modelfinal.pt"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    try:
        state_dict = torch.load(model_path)
        booster_raw = state_dict['booster']
        booster = xgboost.Booster()
        booster.load_model(booster_raw)
        return booster
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

booster = load_model()
if booster is None:
    st.stop()

# Prepare features and predict
features = df[[
    'temperature_2m (°C)', 'precipitation (mm)', 'relative_humidity_2m (%)',
    'wind_speed_10m (km/h)', 'cloud_cover (%)', 'uv_index'
]]
scaler = StandardScaler()
scaler.fit(features)
scaled_features = scaler.transform(features)
dmatrix = xgboost.DMatrix(scaled_features)
probabilities = booster.predict(dmatrix)

# Define label mapping
label_mapping = {0: 'Normal', 1: 'Rain', 2: 'Heavy Rain', 3: 'Heatwave'}
predicted_labels = [label_mapping[np.argmax(probs)] for probs in probabilities]
df['predicted_label'] = predicted_labels

# Dashboard Layout
st.markdown(f'<div class="header"><h1>Islamabad Weather Dashboard</h1><p>Updated: {current_time.strftime("%Y-%m-%d %H:%M:%S PKT")}</p></div>', unsafe_allow_html=True)

# Current Weather Section
col1, col2, col3 = st.columns(3)
with col1:
    temp = current_weather['Temperature (°C)']
    display_temp = f"{temp:.1f}" if isinstance(temp, (int, float)) else "N/A"
    st.markdown(f'<div class="card"><p class="subheader">Temperature</p><p class="metric">{display_temp} °C</p></div>', unsafe_allow_html=True)
with col2:
    precip = current_weather['Precipitation (mm)']
    display_precip = f"{precip:.1f}" if isinstance(precip, (int, float)) else "N/A"
    st.markdown(f'<div class="card"><p class="subheader">Precipitation</p><p class="metric">{display_precip} mm</p></div>', unsafe_allow_html=True)
with col3:
    humid = current_weather['Humidity (%)']
    display_humid = f"{humid:.1f}" if isinstance(humid, (int, float)) else "N/A"
    st.markdown(f'<div class="card"><p class="subheader">Humidity</p><p class="metric">{display_humid} %</p></div>', unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)
with col4:
    wind_speed = current_weather['Wind Speed (km/h)']
    display_wind_speed = f"{wind_speed:.1f}" if isinstance(wind_speed, (int, float)) else "N/A"
    st.markdown(f'<div class="card"><p class="subheader">Wind Speed</p><p class="metric">{display_wind_speed} km/h</p></div>', unsafe_allow_html=True)
with col5:
    cloud_cover = current_weather['Cloud Cover (%)']
    display_cloud_cover = f"{cloud_cover:.1f}" if isinstance(cloud_cover, (int, float)) else "N/A"
    st.markdown(f'<div class="card"><p class="subheader">Cloud Cover</p><p class="metric">{display_cloud_cover} %</p></div>', unsafe_allow_html=True)
with col6:
    uv_index = current_weather['UV Index']
    display_uv_index = f"{uv_index:.1f}" if isinstance(uv_index, (int, float)) else "N/A"
    st.markdown(f'<div class="card"><p class="subheader">UV Index</p><p class="metric">{display_uv_index}</p></div>', unsafe_allow_html=True)

# Prediction Section
col7, col8 = st.columns([1, 2])

with col7:
    st.subheader("8-Hour Predictions")
    st.dataframe(df[['time', 'temperature_2m (°C)', 'precipitation (mm)', 'predicted_label']])

with col8:
    st.subheader("Temperature and Prediction Visualization")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['time'], df['temperature_2m (°C)'], label='Temperature (°C)', color='#2c3e50', marker='o')
    for i, label in enumerate(df['predicted_label']):
        color = '#e74c3c' if label == 'Rain' else '#3498db' if label == 'Normal' else '#e67e22' if label == 'Heavy Rain' else '#f1c40f'
        ax.text(df['time'][i], df['temperature_2m (°C)'][i] + 0.5, label, ha='center', color=color)
    ax.set_title('Hourly Temperature and Weather Prediction', fontsize=14)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    st.pyplot(fig)
