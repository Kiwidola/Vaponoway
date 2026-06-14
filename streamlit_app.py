import streamlit as st
import pandas as pd

# Set page configuration
st.set_page_config(page_title="VapeRadar Dashboard", page_icon="🚭", layout="wide")

# Header
st.title("🚭 VapeRadar Dashboard")
st.markdown("**Real-Time Vape Detection Monitoring**")
st.divider()

# --- Summary Cards ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total Sensors", value="12")
with col2:
    st.metric(label="Active Sensors", value="11")
with col3:
    st.metric(label="Vape Alerts Today", value="7")
with col4:
    st.metric(label="Campus Air Clarity", value="87%")

st.divider()

# --- Sensor Locations & Details ---
st.subheader("📍 Sensor Locations")

# Data for our sensors
sensors = [
    {
        "id": "A01", "color": "🟢", "loc": "Building A - Floor 1", "clarity": 96, 
        "status": "Clean Air", "voc": "120 ppb", "pm": "8 μg/m³", "co2": "420 ppm", "time": "09:12"
    },
    {
        "id": "A02", "color": "🟡", "loc": "Building A - Floor 2", "clarity": 78, 
        "status": "Slightly Abnormal", "voc": "340 ppb", "pm": "18 μg/m³", "co2": "700 ppm", "time": "11:03"
    },
    {
        "id": "B01", "color": "🔴", "loc": "Library Entrance", "clarity": 51, 
        "status": "High Risk", "voc": "640 ppb", "pm": "54 μg/m³", "co2": "980 ppm", "time": "10:15"
    },
    {
        "id": "C01", "color": "🟣", "loc": "Bathroom Block C", "clarity": 12, 
        "status": "Severe Detection", "voc": "945 ppb", "pm": "142 μg/m³", "co2": "1520 ppm", "time": "13:28"
    }
]

# Create a grid layout for the sensors
sensor_cols = st.columns(4)

# Recreating the "click for details" modal feature using Streamlit Expanders
for i, sensor in enumerate(sensors):
    with sensor_cols[i]:
        with st.expander(f"{sensor['color']} Sensor {sensor['id']} - {sensor['clarity']}%", expanded=False):
            st.markdown(f"**Location:** {sensor['loc']}")
            st.markdown(f"**Status:** {sensor['status']}")
            st.divider()
            st.markdown("### Sensor Readings")
            st.markdown(f"**VOC:** {sensor['voc']}")
            st.markdown(f"**PM2.5:** {sensor['pm']}")
            st.markdown(f"**CO₂:** {sensor['co2']}")
            st.markdown(f"**Last Detection:** {sensor['time']}")

# Legend
st.caption("🟢 90-100% Clean Air | 🟡 70-89% Slightly Abnormal | 🔴 40-69% High Risk | 🟣 0-39% Severe Detection")
st.divider()

# --- Detection History ---
st.subheader("⚠ Detection History")

# Create a Pandas DataFrame for the table
history_data = {
    "Time": ["09:42", "10:15", "11:03", "13:28"],
    "Sensor": ["A01", "B01", "C01", "C01"],
    "Air Clarity": ["63%", "51%", "27%", "12%"],
    "Status": ["High Risk", "High Risk", "Severe Detection", "Severe Detection"]
}
df_history = pd.DataFrame(history_data)

# Display the table
st.dataframe(df_history, use_container_width=True, hide_index=True)

st.divider()

# --- Hotspot Map ---
st.subheader("🗺 Vape Hotspot Map")
st.markdown("View live hotspots and historical detection areas.")
if st.button("Open Hotspot Map", type="primary"):
    st.info("Map interface would load here. (Requires mapping library integration like Folium or PyDeck).")
