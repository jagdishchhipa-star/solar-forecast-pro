import streamlit as st
import pandas as pd
import requests
import joblib
import pvlib
from pvlib.location import Location
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Pro Solar Forecaster", layout="wide")
st.title("☀️ Pro Solar Forecaster (Version 3.0)")

# Sidebar
st.sidebar.header("Settings")
lat = st.sidebar.number_input("Latitude", value=26.9124, format="%.4f") # Jaipur
lon = st.sidebar.number_input("Longitude", value=75.7873, format="%.4f")
capacity = st.sidebar.number_input("System Capacity (kW)", value=5.0)
tilt = st.sidebar.slider("Panel Tilt (°)", 0, 90, 26)
azimuth = st.sidebar.slider("Azimuth (180=South)", 0, 360, 180)

# Load Model
try:
    model = joblib.load('solar_model.pkl')
except:
    st.error("Model not found. Run 'python train_model.py' in terminal first.")
    st.stop()

def get_solar_data(lat, lon):
    # Fetching weather in UTC to avoid time-shift bugs
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,shortwave_radiation,direct_normal_irradiance,diffuse_radiation&timezone=UTC&forecast_days=1"
    res = requests.get(url).json()
    
    df = pd.DataFrame({
        'time_utc': pd.to_datetime(res['hourly']['time'], utc=True),
        'temp': res['hourly']['temperature_2m'],
        'ghi': res['hourly']['shortwave_radiation'],
        'dni': res['hourly']['direct_normal_irradiance'],
        'dhi': res['hourly']['diffuse_radiation']
    })
    return df

if st.button("Run Prediction"):
    with st.spinner("Analyzing atmosphere and sun geometry..."):
        # 1. Get Weather (UTC)
        df = get_solar_data(lat, lon)
        
        # 2. Physics Engine (PVLib)
        # Calculate exactly how much sun hits the TILTED panels
        loc = Location(lat, lon)
        solpos = loc.get_solarposition(df['time_utc'])
        
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            solar_zenith=solpos['apparent_zenith'],
            solar_azimuth=solpos['azimuth'],
            dni=df['dni'],
            ghi=df['ghi'],
            dhi=df['dhi']
        )
        
        # Use the Plane-of-Array (POA) as the input for the AI
        df['poa_global'] = poa['poa_global'].fillna(0)
        
        # 3. AI Prediction
        # We use 'poa_global' instead of 'ghi' for better accuracy on tilted roofs
        inputs = df[['poa_global', 'temp']]
        inputs.columns = ['ghi', 'temp'] # Match model feature names
        
        df['power_kw'] = model.predict(inputs) * capacity
        
        # 4. Convert to India Time for the User
        df['time_ist'] = df['time_utc'].dt.tz_convert('Asia/Kolkata')
        
        # --- DISPLAY ---
        total_kwh = df['power_kw'].sum()
        specific_yield = total_kwh / capacity
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Generation", f"{total_kwh:.2f} kWh")
        col2.metric("Specific Yield", f"{specific_yield:.2f} kWh/kW")
        col3.metric("Peak Power", f"{df['power_kw'].max():.2f} kW")
        
        # Chart
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['time_ist'], df['power_kw'], color='#FFC107', lw=3)
        ax.fill_between(df['time_ist'], df['power_kw'], color='#FFC107', alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%I %p', tz='Asia/Kolkata'))
        ax.set_ylabel("Power (kW)")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)
        
        # Data Table
        with st.expander("View Raw Calculation Table"):
            st.dataframe(df[['time_ist', 'ghi', 'poa_global', 'power_kw']])