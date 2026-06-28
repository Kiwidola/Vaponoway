import os
import joblib
import pandas as pd
import pydeck as pdk
import streamlit as st

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
BASE_LAT    = 18.5847
BASE_LON    = 99.0256
REFRESH_MS  = 30_000
HOURS_BACK  = 24

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

# ─────────────────────────────────────────────
# PAGE CONFIG & DARK THEME CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title=MODEL_NAME, layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700;800&display=swap');

    /* ── Dark canvas ── */
    html, body, [data-testid="stAppViewContainer"] {
        background: #0a0d14 !important;
        color: #e2e8f0;
    }
    [data-testid="stHeader"]  { background: transparent !important; }
    [data-testid="stSidebar"] { background: #0f1320 !important; }
    .block-container { padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1320px; }

    /* ── Typography ── */
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        color: #f1f5f9;
        letter-spacing: -0.4px;
    }

    /* ── Eyebrow label ── */
    .eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .eyebrow::before {
        content: "";
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #22d3ee;
        box-shadow: 0 0 8px #22d3ee;
    }

    /* ── Card containers ── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #111827 !important;
        border: 1px solid #1e2a3a !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
        padding: 24px 26px !important;
        margin-bottom: 20px !important;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #0f1825;
        border: 1px solid #1e2d40;
        border-radius: 12px;
        padding: 16px 18px;
    }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.72rem !important; font-weight: 600 !important; letter-spacing: 0.5px; }
    [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.5rem !important; font-family: 'JetBrains Mono', monospace !important; }
    [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

    /* ── Hero status pill ── */
    .status-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        padding: 14px 52px;
        min-width: 300px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.2px;
        white-space: nowrap;
        font-family: 'Inter', sans-serif;
    }
    .status-vape {
        background: rgba(239,68,68,0.12);
        border: 1.5px solid rgba(239,68,68,0.5);
        color: #fca5a5;
        box-shadow: 0 0 24px rgba(239,68,68,0.15);
    }
    .status-clean {
        background: rgba(34,211,238,0.08);
        border: 1.5px solid rgba(34,211,238,0.35);
        color: #67e8f9;
        box-shadow: 0 0 24px rgba(34,211,238,0.12);
    }
    .status-offline {
        background: rgba(100,116,139,0.12);
        border: 1.5px solid #334155;
        color: #94a3b8;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    .dvn-scroller { background: #0f1825 !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #0f1320;
        border-radius: 10px;
        gap: 4px;
        padding: 4px;
        border: 1px solid #1e2a3a;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #64748b;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"] p { color: inherit !important; }
    .stTabs [aria-selected="true"] {
        background: #1e2a3a !important;
        color: #e2e8f0 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.3);
    }

    /* ── Node status cards ── */
    .node-card {
        background: #0f1825;
        border: 1px solid #1e2d40;
        border-radius: 10px;
        padding: 11px 16px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.84rem;
        color: #cbd5e1;
    }
    .node-id {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #94a3b8;
    }
    .pill-red   { color: #f87171; font-weight: 700; }
    .pill-green { color: #34d399; font-weight: 700; }
    .pill-dot-red   { display:inline-block; width:7px; height:7px; border-radius:50%; background:#ef4444; box-shadow:0 0 6px #ef4444; margin-right:6px; }
    .pill-dot-green { display:inline-block; width:7px; height:7px; border-radius:50%; background:#10b981; box-shadow:0 0 6px #10b981; margin-right:6px; }

    /* ── Map legend ── */
    .map-legend {
        display: flex;
        gap: 20px;
        margin-top: 12px;
        font-size: 0.78rem;
        color: #64748b;
        font-family: 'JetBrains Mono', monospace;
    }
    .legend-dot {
        display: inline-block;
        width: 10px; height: 10px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
    }

    /* ── Misc ── */
    hr { border-color: #1e2a3a; }
    p, li { color: #94a3b8; }
    [data-testid="stCaptionContainer"] { color: #475569 !important; }
    .stAlert { border-radius: 10px; }

    /* ── Section divider label ── */
    .section-sub {
        font-size: 0.78rem;
        color: #475569;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    return None

my_model = load_model()

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_sensor_data():
    try:
        df = pd.read_csv(SHEET_URL)
        df = df.rename(columns=COLUMN_RENAME)
        if "Unnamed: 1" in df.columns:
            df = df.drop(columns=["Unnamed: 1"])
        if "Timestamp" in df.columns:
            df["Display_Time"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
            df["Sort_Time"]    = df["Display_Time"] + pd.Timedelta(hours=7)
            df = df.dropna(subset=["Display_Time"]).sort_values("Sort_Time", ascending=False)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

df = load_sensor_data()

if df.empty:
    st.warning("No data available. Check your Google Sheets connection.")
    st.stop()

latest   = df.iloc[0]
previous = df.iloc[1] if len(df) > 1 else latest

# ─────────────────────────────────────────────
# RUN MODEL
# ─────────────────────────────────────────────
prediction   = None
confidence   = None
all_features = df[FEATURE_COLS].rename(columns=MAPPING_DICT)

if my_model:
    try:
        latest_features = latest[FEATURE_COLS].to_frame().T.rename(columns=MAPPING_DICT)
        prediction = int(my_model.predict(latest_features)[0])
        if hasattr(my_model, "predict_proba"):
            proba      = my_model.predict_proba(latest_features)[0]
            confidence = float(proba[prediction]) * 100
        df["is_vape"] = my_model.predict(all_features)
    except Exception as e:
        st.error(f"Prediction error: {e}")

# ─────────────────────────────────────────────
# SENSOR POSITIONS  (used in both map layers)
# ─────────────────────────────────────────────
live_state = 1 if prediction == 1 else 0

mock_sensors = pd.DataFrame({
    "sensor_id":    ["SN-01", "SN-02", "SN-03", "SN-04"],
    "label":        ["Main Lobby", "East Restroom", "Breakroom", "Stairwell B"],
    "latitude":     [BASE_LAT + 0.0004, BASE_LAT + 0.0004, BASE_LAT - 0.0005, BASE_LAT + 0.0002],
    "longitude":    [BASE_LON,          BASE_LON - 0.0006,  BASE_LON - 0.0002, BASE_LON + 0.0005],
    "vape_detected":[live_state, 0, 0, 0],
})
mock_sensors["status_text"] = mock_sensors["vape_detected"].map(
    {1: "⚠ Vape Detected", 0: "✅ Clean"}
)
mock_sensors["fill_color"]  = mock_sensors["vape_detected"].map(
    {1: [239, 68, 68, 230], 0: [34, 211, 238, 200]}
)
mock_sensors["ring_color"]  = mock_sensors["vape_detected"].map(
    {1: [239, 68, 68, 80],  0: [34, 211, 238, 60]}
)

# ─────────────────────────────────────────────
# HERO CARD
# ─────────────────────────────────────────────
with st.container(border=True):
    st.markdown(
        "<div style='text-align:center;font-size:0.7rem;font-family:JetBrains Mono,monospace;"
        "letter-spacing:2px;color:#334155;text-transform:uppercase;margin-bottom:6px'>"
        "AURAFARM AI · FACILITY AIR QUALITY MONITOR</div>"
        "<div style='text-align:center;font-size:2rem;font-weight:800;line-height:1.1;"
        "color:#f1f5f9;font-family:Inter,sans-serif;margin-bottom:20px'>Vapo noWay</div>",
        unsafe_allow_html=True,
    )

    if prediction is None:
        st.markdown(
            "<div style='text-align:center'><span class='status-pill status-offline'>"
            "⚠ Detection Offline</span></div>",
            unsafe_allow_html=True,
        )
    elif prediction == 1:
        conf_str = f" · {confidence:.0f}% confidence" if confidence else ""
        st.markdown(
            f"<div style='text-align:center'>"
            f"<span class='status-pill status-vape'>🚨 Vape Detected{conf_str}</span><br>"
            f"<div style='color:#475569;font-size:0.72rem;font-family:JetBrains Mono,monospace;"
            f"margin-top:8px'>Last updated {latest['Display_Time'].strftime('%H:%M:%S')}</div></div>",
            unsafe_allow_html=True,
        )
    else:
        conf_str = f" · {confidence:.0f}% confidence" if confidence else ""
        st.markdown(
            f"<div style='text-align:center'>"
            f"<span class='status-pill status-clean'>✦ Air Quality Clean{conf_str}</span><br>"
            f"<div style='color:#475569;font-size:0.72rem;font-family:JetBrains Mono,monospace;"
            f"margin-top:8px'>Last updated {latest['Display_Time'].strftime('%H:%M:%S')}</div></div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────
def safe_delta(col):
    try:
        return round(float(latest[col]) - float(previous[col]), 2)
    except Exception:
        return None

with st.container(border=True):
    st.markdown("<div class='eyebrow'>Live Sensor Readings</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("TVOC",     f"{latest['TVOC']} ppb",    delta=safe_delta("TVOC"),     delta_color="inverse")
    c2.metric("PM 2.5",   f"{latest['PM2.5']} μg/m³", delta=safe_delta("PM2.5"),    delta_color="inverse")
    c3.metric("eCO₂",     f"{latest['eCO2']} ppm",    delta=safe_delta("eCO2"),     delta_color="inverse")
    c4.metric("MQ7",      f"{latest['CH0']}",         delta=safe_delta("CH0"),      delta_color="inverse")
    c5.metric("MQ135",    f"{latest['MQ135']}",       delta=safe_delta("MQ135"),    delta_color="inverse")
    c6.metric("Temp",     f"{latest['Temp']} °C",     delta=safe_delta("Temp"))
    c7.metric("Humidity", f"{latest['Humidity']} %",  delta=safe_delta("Humidity"))

# ─────────────────────────────────────────────
# DETECTION HISTORY  +  IMPROVED FACILITY MAP
# ─────────────────────────────────────────────
col_hist, col_map = st.columns([1, 1], gap="large")

with col_hist:
    with st.container(border=True):
        st.markdown("<div class='eyebrow'>Detection History</div>", unsafe_allow_html=True)

        if my_model and "is_vape" in df.columns:
            vape_rows = df[df["is_vape"] == 1].copy()

            if not vape_rows.empty:
                vape_rows = vape_rows.sort_values("Display_Time")
                vape_rows["block"] = (
                    vape_rows["Display_Time"].diff() > pd.Timedelta(minutes=5)
                ).cumsum()

                event_rows = []
                for _, grp in vape_rows.groupby("block"):
                    event_rows.append({
                        "Date":       grp["Display_Time"].min().strftime("%Y-%m-%d"),
                        "Start":      grp["Display_Time"].min().strftime("%H:%M"),
                        "End":        grp["Display_Time"].max().strftime("%H:%M"),
                        "Duration":   str(grp["Display_Time"].max() - grp["Display_Time"].min()).split(".")[0],
                        "Peak TVOC":  f"{grp['TVOC'].max():.0f} ppb",
                        "Peak PM2.5": f"{grp['PM2.5'].max():.1f} μg/m³",
                    })

                events_df = pd.DataFrame(event_rows[::-1])
                st.dataframe(
                    events_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Date":       st.column_config.TextColumn("Date"),
                        "Start":      st.column_config.TextColumn("Start"),
                        "End":        st.column_config.TextColumn("End"),
                        "Duration":   st.column_config.TextColumn("Duration"),
                        "Peak TVOC":  st.column_config.TextColumn("Peak TVOC"),
                        "Peak PM2.5": st.column_config.TextColumn("Peak PM2.5"),
                    },
                )
                st.caption(f"{len(events_df)} detection event(s) in available data.")
            else:
                st.markdown(
                    "<div style='color:#475569;padding:32px 0;text-align:center;"
                    "font-family:JetBrains Mono,monospace;font-size:0.82rem'>"
                    "No vape events detected in the available data.</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div style='color:#475569;padding:32px 0;text-align:center;"
                "font-family:JetBrains Mono,monospace;font-size:0.82rem'>"
                "Detection system offline — history unavailable.</div>",
                unsafe_allow_html=True,
            )

with col_map:
    with st.container(border=True):
        st.markdown("<div class='eyebrow'>Facility Sensor Network</div>", unsafe_allow_html=True)

        # Outer glow ring layer (pulsing effect approximation)
        ring_layer = pdk.Layer(
            "ScatterplotLayer",
            data=mock_sensors,
            get_position=["longitude", "latitude"],
            get_fill_color="ring_color",
            get_radius=18,
            radius_units="meters",
            radius_min_pixels=14,
            radius_max_pixels=28,
            pickable=False,
            stroked=False,
        )

        # Inner dot layer
        dot_layer = pdk.Layer(
            "ScatterplotLayer",
            data=mock_sensors,
            get_position=["longitude", "latitude"],
            get_fill_color="fill_color",
            get_radius=7,
            radius_units="meters",
            radius_min_pixels=6,
            radius_max_pixels=14,
            pickable=True,
            stroked=True,
            get_line_color=[255, 255, 255, 60],
            line_width_min_pixels=1,
        )

        # Label text layer
        text_layer = pdk.Layer(
            "TextLayer",
            data=mock_sensors,
            get_position=["longitude", "latitude"],
            get_text="sensor_id",
            get_size=11,
            get_color=[226, 232, 240, 220],
            get_anchor="middle",
            get_alignment_baseline="'bottom'",
            get_pixel_offset=[0, -18],
            font_family="JetBrains Mono, monospace",
            font_weight=600,
            pickable=False,
        )

        view_state = pdk.ViewState(
            latitude=BASE_LAT,
            longitude=BASE_LON,
            zoom=17,
            pitch=40,        # slight 3D tilt for depth
            bearing=0,
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[ring_layer, dot_layer, text_layer],
                initial_view_state=view_state,
                map_style="mapbox://styles/mapbox/dark-v11",
                tooltip={
                    "html": (
                        "<div style='font-family:JetBrains Mono,monospace;font-size:12px;"
                        "background:#111827;color:#e2e8f0;padding:10px 14px;border-radius:8px;"
                        "border:1px solid #1e2a3a;line-height:1.6'>"
                        "<b style='color:#67e8f9'>{sensor_id}</b><br>"
                        "<span style='color:#94a3b8'>{label}</span><br>"
                        "<span style='color:#f1f5f9'>{status_text}</span>"
                        "</div>"
                    ),
                    "style": {"backgroundColor": "transparent", "color": "transparent"},
                },
            ),
            height=320,
        )

        # Map legend
        st.markdown(
            "<div class='map-legend'>"
            "<span><span class='legend-dot' style='background:#22d3ee;box-shadow:0 0 6px #22d3ee'></span>Clean</span>"
            "<span><span class='legend-dot' style='background:#ef4444;box-shadow:0 0 6px #ef4444'></span>Vape Detected</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Node status list
        st.markdown(
            "<div style='margin-top:16px;font-size:0.7rem;font-family:JetBrains Mono,monospace;"
            "color:#475569;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px'>"
            "Node Status</div>",
            unsafe_allow_html=True,
        )
        for _, row in mock_sensors.iterrows():
            if row["vape_detected"]:
                pill = "<span class='pill-red'><span class='pill-dot-red'></span>Vape Detected</span>"
            else:
                pill = "<span class='pill-green'><span class='pill-dot-green'></span>Clean</span>"
            st.markdown(
                f"<div class='node-card'>"
                f"<div><span class='node-id'>{row['sensor_id']}</span>"
                f"<span style='color:#64748b;margin-left:8px;font-size:0.78rem'>{row['label']}</span></div>"
                f"{pill}"
                f"</div>",
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────
# TREND CHARTS
# ─────────────────────────────────────────────
chart_data = df.sort_values("Sort_Time", ascending=True).copy()
cutoff     = chart_data["Sort_Time"].max() - pd.Timedelta(hours=HOURS_BACK)
chart_data = chart_data[chart_data["Sort_Time"] >= cutoff].set_index("Sort_Time")
numeric_cols = chart_data.select_dtypes(include="number").columns
chart_data   = chart_data[numeric_cols].resample("1min").mean().interpolate(method="time")

if my_model and "is_vape" in df.columns:
    vape_overlay = (
        df[df["Sort_Time"] >= cutoff]
        .sort_values("Sort_Time")
        .set_index("Sort_Time")[["is_vape"]]
        .resample("1min")
        .max()
        .rename(columns={"is_vape": "⚠ Vape Event"})
    )
    chart_data = chart_data.join(vape_overlay, how="left")

with st.container(border=True):
    st.markdown(
        f"<div class='eyebrow'>Sensor Trends · Last {HOURS_BACK} Hours</div>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Particles", "Air Quality", "Climate", "All Sensors"])

    with tab1:
        cols_p = [c for c in ["PM2.5", "PM10", "MQ135"] if c in chart_data.columns]
        if "⚠ Vape Event" in chart_data.columns:
            cols_p.append("⚠ Vape Event")
        st.line_chart(chart_data[cols_p])

    with tab2:
        cols_a = [c for c in ["TVOC", "eCO2"] if c in chart_data.columns]
        if "⚠ Vape Event" in chart_data.columns:
            cols_a.append("⚠ Vape Event")
        st.line_chart(chart_data[cols_a])

    with tab3:
        cols_c = [c for c in ["Temp", "Humidity"] if c in chart_data.columns]
        st.line_chart(chart_data[cols_c])

    with tab4:
        display_cols = [c for c in FEATURE_COLS if c in chart_data.columns]
        st.line_chart(chart_data[display_cols])

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;color:#334155;font-size:0.72rem;"
    "font-family:JetBrains Mono,monospace;padding:12px 0;letter-spacing:0.5px'>"
    "AURAFARM AI · AUTO-REFRESHES EVERY 30 s · SENSOR DATA VIA GOOGLE SHEETS"
    "</div>",
    unsafe_allow_html=True,
)
