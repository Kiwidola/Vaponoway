import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import timedelta

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH  = "models/rf_model_unified.joblib"
MODEL_NAME  = "Aurafarm AI"
SHEET_URL   = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vT8Oho84O3uIYEEYE2iNub7I5Ktv4mTUteMkdBR4NpBTlJZS0tY2VFXmqM-_XlGIgSaeUIR7VjpnWSZ"
    "/pub?output=csv"
)

FEATURE_COLS   = ["TVOC", "eCO2", "Temp", "Humidity", "PM2.5", "CH0", "CH3", "MQ135"]
MAPPING_DICT   = {
    "TVOC": "col_2", "eCO2": "col_3", "Temp": "col_4", "Humidity": "col_5",
    "PM2.5": "col_6", "CH0": "col_7", "CH3": "col_8", "MQ135": "col_9",
}
COLUMN_RENAME  = {
    "Unnamed: 0": "Timestamp",
    "tvoc": "TVOC", "eco2": "eCO2", "temp": "Temp", "humidity": "Humidity",
    "ch0": "CH0", "ch3": "CH3", "mq135": "MQ135", "2.5": "PM2.5", "10": "PM10",
}

st.set_page_config(page_title=MODEL_NAME, layout="wide")

# ─────────────────────────────────────────────
# 1. LOAD MODEL & DATA
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try: return joblib.load(MODEL_PATH)
        except Exception: return None
    return None

# Auto-refresh disabled by removing ttl
@st.cache_data
def load_sensor_data():
    try:
        df = pd.read_csv(SHEET_URL)
        df = df.rename(columns=COLUMN_RENAME)
        if "Unnamed: 1" in df.columns: df = df.drop(columns=["Unnamed: 1"])
        if "Timestamp" in df.columns:
            df["Display_Time"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
            df = df.dropna(subset=["Display_Time"]).sort_values("Display_Time", ascending=False)
        return df
    except Exception: return pd.DataFrame()

my_model = load_model()
df = load_sensor_data()

# ─────────────────────────────────────────────
# 2. DASHBOARD UI
# ─────────────────────────────────────────────
st.title(f"{MODEL_NAME} Dashboard")
if df.empty:
    st.warning("No data found.")
    st.stop()

latest = df.iloc[0]

# ─────────────────────────────────────────────
# 3. RUN MODEL PREDICTION
# ─────────────────────────────────────────────
prediction = None
if my_model:
    try:
        latest_features = latest[FEATURE_COLS].to_frame().T.rename(columns=MAPPING_DICT)
        prediction = int(my_model.predict(latest_features)[0])
        if prediction == 1:
            st.error("VAPE DETECTED: AI Model indicates vape particles!")
        else:
            st.success("AIR QUALITY: Clean.")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# ─────────────────────────────────────────────
# 4. METRICS
# ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Temp", f"{latest['Temp']} °C")
col2.metric("Humidity", f"{latest['Humidity']} %")
col3.metric("TVOC", f"{latest['TVOC']} ppb")
col4.metric("PM 2.5", f"{latest['PM2.5']} μg/m³")
st.caption(f"Last updated: {latest['Display_Time']}")
st.divider()

# ─────────────────────────────────────────────
# 5. STATUS DISPLAY
# ─────────────────────────────────────────────
st.subheader("Current Status")
live_state = 1 if prediction == 1 else 0
status_text = 'Poor (Vape)' if live_state == 1 else 'Good'
color = "#ff4b4b" if live_state == 1 else "#00cc66"

st.markdown(f"### System Status: <span style='color:{color};'>{status_text}</span>", unsafe_allow_html=True)
st.write("---")
st.write("#### Detailed Sensor Readings")
cols_to_show = ["Timestamp", "TVOC", "eCO2", "Temp", "Humidity", "PM2.5"]
st.dataframe(latest[cols_to_show].T.rename(columns={latest.name: "Value"}), use_container_width=True)

# ─────────────────────────────────────────────
# 6. TRENDS
# ─────────────────────────────────────────────
st.divider()
st.subheader("All Time Sensor Trends")
chart_data = df.sort_values(by="Display_Time", ascending=True).set_index("Display_Time")
numeric_cols = chart_data.select_dtypes(include="number").columns
chart_data = (chart_data[numeric_cols].resample("1min").mean().interpolate(method="time"))

tab1, tab2, tab3 = st.tabs(["Particles", "Air Quality", "Climate"])
with tab1: st.line_chart(chart_data[["PM2.5", "PM10", "MQ135"]])
with tab2: st.line_chart(chart_data[["TVOC", "eCO2"]])
with tab3: st.line_chart(chart_data[["Temp", "Humidity"]])
