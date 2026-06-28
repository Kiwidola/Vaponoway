import os
import joblib
import pandas as pd
import pydeck as pdk
import streamlit as st
import plotly.express as px
from datetime import timedelta

# --- CONFIGURATION ---
MODEL_PATH = "models/rf_model_unified.joblib"
MODEL_NAME = "Aurafarm AI"
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT8Oho84O3uIYEEYE2iNub7I5Ktv4mTUteMkdBR4NpBTlJZS0tY2VFXmqM-_XlGIgSaeUIR7VjpnWSZ/pub?output=csv"

# Fixed coordinates for the facility
MY_LAT = 18.5847
MY_LON = 99.0256

st.set_page_config(page_title=MODEL_NAME, layout="wide")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.warning(f"Model file not found at {MODEL_PATH}.")
        return None

my_model = load_model()

# --- 2. DATA LOADING ---
@st.cache_data(ttl=30)
def load_sensor_data():
    try:
        df = pd.read_csv(SHEET_URL)
        column_mapping = {
            "Unnamed: 0": "Timestamp",
            "tvoc": "TVOC",
            "eco2": "eCO2",
            "temp": "Temp",
            "humidity": "Humidity",
            "ch0": "CH0",
            "ch3": "CH3",
            "mq135": "MQ135",
            "2.5": "PM2.5",
            "10": "PM10",
        }
        df = df.rename(columns=column_mapping)
        if "Unnamed: 1" in df.columns:
            df = df.drop(columns=["Unnamed: 1"])
        if "Timestamp" in df.columns:
            df["Display_Time"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
            df["Sort_Time"] = df["Display_Time"] + pd.Timedelta(hours=7)
            df = df.dropna(subset=["Display_Time"])
            df = df.sort_values(by="Sort_Time", ascending=False)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

df = load_sensor_data()

# --- 3. DASHBOARD UI ---
st.title(f"{MODEL_NAME} Dashboard")
if df.empty:
    st.warning("No data found.")
    st.stop()

latest = df.iloc[0].to_frame().T

# --- 4. RUN MODEL PREDICTION ---
prediction = None
mapping_dict = {"TVOC": "col_2", "eCO2": "col_3", "Temp": "col_4", "Humidity": "col_5", "PM2.5": "col_6", "CH0": "col_7", "CH3": "col_8", "MQ135": "col_9"}

if my_model:
    features = latest[["TVOC", "eCO2", "Temp", "Humidity", "PM2.5", "CH0", "CH3", "MQ135"]].rename(columns=mapping_dict)
    try:
        prediction = my_model.predict(features)[0]
        if prediction == 1:
            st.error("VAPE DETECTED: AI Model indicates vape particles!")
        else:
            st.success("AIR QUALITY: Clean.")
    except Exception as e:
        st.error(f"Prediction error: {e}")
else:
    st.info("Prediction system offline.")

# --- 5. VAPE DETECTION HISTORY ---
if my_model:
    st.subheader("Detection History")
    hist_features = df[["TVOC", "eCO2", "Temp", "Humidity", "PM2.5", "CH0", "CH3", "MQ135"]].rename(columns=mapping_dict)
    df["is_vape"] = my_model.predict(hist_features)
    vape_rows = df[df["is_vape"] == 1].copy()
    if not vape_rows.empty:
        vape_rows = vape_rows.sort_values("Display_Time")
        vape_rows["block"] = (vape_rows["Display_Time"].diff() > pd.Timedelta(minutes=5)).cumsum()
        for _, group in reversed(list(vape_rows.groupby("block"))):
            start_time = group["Display_Time"].min().strftime("%H:%M")
            end_time = group["Display_Time"].max().strftime("%H:%M")
            date_str = group["Display_Time"].min().strftime("%Y-%m-%d")
            time_range = f"at {start_time}" if start_time == end_time else f"from {start_time} to {end_time}"
            st.markdown(f"<div style='color: #ff4b4b; font-weight: bold; padding: 10px; border-left: 5px solid #ff4b4b; background-color: #fff0f0; margin-bottom: 10px; border-radius: 4px;'>Vape Detected: {date_str} {time_range}</div>", unsafe_allow_html=True)
    else:
        st.info("No vape events detected in the available data.")

# --- METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Temp", f"{latest['Temp'].values[0]} °C")
col2.metric("Humidity", f"{latest['Humidity'].values[0]} %")
col3.metric("TVOC", f"{latest['TVOC'].values[0]} ppb")
col4.metric("PM 2.5", f"{latest['PM2.5'].values[0]} μg/m³")
st.caption(f"Last updated (Sensor Time): {latest['Display_Time'].values[0]}")
st.divider()

# --- ALL TIME TRENDS ---
st.subheader("All Time Sensor Trends")
chart_data = df.sort_values(by="Sort_Time", ascending=True)
chart_data = chart_data.set_index("Sort_Time")
numeric_cols = chart_data.select_dtypes(include="number").columns
chart_data = (chart_data[numeric_cols].resample("1min").mean().interpolate(method="time"))

def plot_interactive(df, cols):
    fig = px.line(df, y=cols)
    max_time = df.index.max()
    min_time = max_time - timedelta(hours=24)
    fig.update_xaxes(rangeslider_visible=True, range=[min_time, max_time], fixedrange=False)
    fig.update_layout(hovermode="x unified", dragmode="pan")
    return fig

tab1, tab2, tab3 = st.tabs(["Particles", "Air Quality", "Climate"])
with tab1:
    st.plotly_chart(plot_interactive(chart_data, ["PM2.5", "PM10", "MQ135"]), use_container_width=True)
with tab2:
    st.plotly_chart(plot_interactive(chart_data, ["TVOC", "eCO2"]), use_container_width=True)
with tab3:
    st.plotly_chart(plot_interactive(chart_data, ["Temp", "Humidity"]), use_container_width=True)

# --- 6. MAP ---
st.divider()
st.subheader("Map")

live_state = 1 if ('prediction' in locals() and prediction == 1) else 0

# Define a single sensor
sensor_data = {
    'sensor_id': ['SN-01 (Main Location)'],
    'latitude': [MY_LAT],
    'longitude': [MY_LON],
    'vape_detected': [live_state],
    'air_quality': ['Poor (Vape)' if live_state == 1 else 'Good']
}
mock_sensors = pd.DataFrame(sensor_data)
mock_sensors["color"] = mock_sensors["vape_detected"].map({1: [255, 75, 75, 255], 0: [0, 204, 102, 255]})

col_map, col_text = st.columns([2, 1])
with col_map:
    view_state = pdk.ViewState(latitude=MY_LAT, longitude=MY_LON, zoom=17, pitch=0)
    layer = pdk.Layer(
        "ScatterplotLayer", 
        data=mock_sensors, 
        get_position=["longitude", "latitude"], 
        get_fill_color="color", 
        get_radius=10, 
        radius_units="meters", 
        pickable=True
    )
    st.pydeck_chart(pdk.Deck(
        layers=[layer], 
        initial_view_state=view_state, 
        tooltip={"text": "Sensor: {sensor_id}\nStatus: {air_quality}"}
    ))

with col_text:
    st.write("### Live Sensor Data")
    # Display the actual latest row of data from your CSV
    st.dataframe(latest.T.rename(columns={latest.index[0]: "Current Value"}), use_container_width=True)
    
    # Status indicator
    color = "#ff4b4b" if live_state == 1 else "#00cc66"
    st.markdown(f"**Status:** <span style='color:{color}; font-weight:bold;'>{mock_sensors.iloc[0]['air_quality']}</span>", unsafe_allow_html=True)
