import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk

# ============================
# Config
# ============================
st.set_page_config(page_title="Pixl Parking Dashboard", layout="wide")
st.title("Pixl Parking — Cost & Coverage Dashboard")

LOCAL_SEGMENTS = os.path.join("data", "curb_segments.geojson")

# CARTO basemaps (no token required)
CARTO_STYLES = {
    "CARTO Positron (light)": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    "CARTO Voyager (streets)": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    "CARTO Dark Matter (dark)": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
}

# Map colors (coverage-class visualization)
COLOR_MAP = {
    "transit": [0, 150, 255],        # blue
    "opportunistic": [255, 165, 0],   # orange
    "active": [220, 0, 90],          # magenta/red
    "none": [160, 160, 160],         # grey
}

# ============================
# Sidebar: SLA + deployment + costs
# ============================
st.sidebar.header("SLA")
sla_minutes = st.sidebar.slider("Target refresh SLA (minutes)", 5, 60, 15, step=1)

st.sidebar.divider()
st.sidebar.header("Deployment Mix")

use_transit = st.sidebar.checkbox("Use Transit data feed", value=True)
num_opportunistic = st.sidebar.slider("Opportunistic drivers", 0, 500, 80)
num_active = st.sidebar.slider("Active drivers", 0, 80, 12)

st.sidebar.divider()
st.sidebar.header("Cost assumptions (editable)")

equip_transit_upfront = st.sidebar.number_input(
    "Transit integration equipment (one-time, $)", min_value=0, value=25000, step=1000
)
equip_annual_maint = st.sidebar.number_input(
    "Annual maintenance (transit integration, $/yr)", min_value=0, value=5000, step=500
)

active_driver_cost_hr = st.sidebar.number_input("Active driver cost ($/hr)", min_value=0.0, value=35.0, step=1.0)
active_driver_hours_day = st.sidebar.number_input("Active driver hours/day", min_value=0.0, value=6.0, step=0.5)

opportunistic_cost_hr = st.sidebar.number_input("Opportunistic cost ($/hr)", min_value=0.0, value=20.0, step=1.0)
opportunistic_hours_day = st.sidebar.number_input("Opportunistic hours/day", min_value=0.0, value=4.0, step=0.5)

st.sidebar.divider()
st.sidebar.header("Coverage assumptions (editable)")

transit_total_hits_per_hour = st.sidebar.number_input(
    "Transit hits/hour across paid network (if enabled)",
    min_value=0.0, value=900.0, step=50.0,
    help="Total vehicle-tags per hour produced by buses over the paid network. Placeholder until real GTFS overlap is wired."
)
opp_hits_per_driver_per_hour = st.sidebar.number_input(
    "Opportunistic hits/driver/hour", min_value=0.0, value=60.0, step=5.0
)
active_hits_per_driver_per_hour = st.sidebar.number_input(
    "Active hits/driver/hour", min_value=0.0, value=140.0, step=10.0
)

transit_coverage_share = st.sidebar.slider(
    "Transit coverage share of segments (placeholder)",
    0.0, 1.0, 0.40, 0.05,
    help="Placeholder for fraction of segments transit touches at least once (e.g., ~285/708 ≈ 0.40)."
)

st.sidebar.divider()
st.sidebar.header("Map")
basemap = st.sidebar.selectbox("Basemap", list(CARTO_STYLES.keys()), index=0)

show_transit = st.sidebar.checkbox("Show Transit layer", value=True)
show_opportunistic = st.sidebar.checkbox("Show Opportunistic layer", value=True)
show_active = st.sidebar.checkbox("Show Active layer", value=True)

# ============================
# Load curb segments (local GeoJSON)
# ============================
if not os.path.exists(LOCAL_SEGMENTS):
    st.error(f"Missing curb segments file: {LOCAL_SEGMENTS}")
    st.stop()

@st.cache_data(ttl=3600, show_spinner=True)
def load_segments(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.notna()].copy()
    try:
        gdf = gdf.to_crs("EPSG:4326")
    except Exception:
        gdf.set_crs("EPSG:4326", inplace=True, allow_override=True)
    if "seg_id" not in gdf.columns:
        gdf["seg_id"] = np.arange(len(gdf))
    return gdf

seg_gdf = load_segments(LOCAL_SEGMENTS)

# Stable join key (useful later)
if {"block_id", "begin_measure", "end_measure"}.issubset(seg_gdf.columns):
    seg_gdf["segment_key"] = (
        seg_gdf["block_id"].astype(str) + "_" +
        seg_gdf["begin_measure"].astype(str) + "_" +
        seg_gdf["end_measure"].astype(str)
    )
else:
    seg_gdf["segment_key"] = seg_gdf["seg_id"].astype(str)

N = len(seg_gdf)

# ============================
# Cost model (simple, clear)
# ============================
daily_active_cost = num_active * active_driver_cost_hr * active_driver_hours_day
daily_opp_cost = num_opportunistic * opportunistic_cost_hr * opportunistic_hours_day
annual_ops_cost = (daily_active_cost + daily_opp_cost) * 365 + (equip_annual_maint if use_transit else 0)
upfront_cost = equip_transit_upfront if use_transit else 0

# ============================
# SLA model (placeholder but responsive)
# ============================
rng = np.random.default_rng(123)
activity_factor = rng.uniform(0.6, 1.6, size=N)  # stable heterogeneity per segment

# Transit coverage placeholder: fixed subset of segments
rng2 = np.random.default_rng(999)
transit_flag = np.zeros(N, dtype=int)
if use_transit and transit_coverage_share > 0:
    k = int(round(transit_coverage_share * N))
    idx = rng2.choice(np.arange(N), size=min(k, N), replace=False)
    transit_flag[idx] = 1

# Total hits/hour by source
total_transit_hits_hr = transit_total_hits_per_hour if use_transit else 0.0
total_opp_hits_hr = num_opportunistic * opp_hits_per_driver_per_hour
total_active_hits_hr = num_active * active_hits_per_driver_per_hour

# Per-segment rates (hits/hour per segment)
n_transit_segments = max(1, transit_flag.sum())
lambda_transit_hr = np.where(transit_flag == 1, total_transit_hits_hr / n_transit_segments, 0.0)
lambda_opp_hr = total_opp_hits_hr / max(1, N)
lambda_active_hr = total_active_hits_hr / max(1, N)

lambda_total_hr = (lambda_transit_hr + lambda_opp_hr + lambda_active_hr) * activity_factor

expected_refresh_min = np.where(lambda_total_hr > 0, 60.0 / lambda_total_hr, np.inf)

pct_meeting_sla = float(np.mean(expected_refresh_min <= sla_minutes) * 100.0)
finite = expected_refresh_min[np.isfinite(expected_refresh_min)]
median_refresh = float(np.median(finite)) if len(finite) else np.inf
p90_refresh = float(np.quantile(finite, 0.90)) if len(finite) else np.inf

# Dominant source for visualization
dominant = np.argmax(
    np.vstack([lambda_transit_hr, np.full(N, lambda_opp_hr), np.full(N, lambda_active_hr)]),
    axis=0
)
coverage_class = np.array(["none"] * N, dtype=object)
coverage_class[lambda_total_hr > 0] = np.where(dominant[lambda_total_hr > 0] == 0, "transit",
                                    np.where(dominant[lambda_total_hr > 0] == 1, "opportunistic", "active"))

# Visibility toggles
visible = {"none"}
if show_transit: visible.add("transit")
if show_opportunistic: visible.add("opportunistic")
if show_active: visible.add("active")

seg_gdf["coverage_class"] = coverage_class
seg_gdf["expected_refresh_min"] = expected_refresh_min
seg_gdf["transit_flag"] = transit_flag

map_gdf = seg_gdf[seg_gdf["coverage_class"].isin(visible)].copy()
map_gdf["line_color"] = map_gdf["coverage_class"].map(COLOR_MAP)

# ============================
# Top dashboard cards
# ============================
st.write("### KPIs")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Segments in network", f"{N:,}")
with c2:
    st.metric(f"SLA coverage (≤ {sla_minutes} min)", f"{pct_meeting_sla:.1f}%")
    st.progress(min(1.0, pct_meeting_sla / 100.0))
with c3:
    st.metric("Median refresh", f"{median_refresh:.1f} min" if np.isfinite(median_refresh) else "∞")
with c4:
    st.metric("P90 refresh", f"{p90_refresh:.1f} min" if np.isfinite(p90_refresh) else "∞")

st.write("### Cost snapshot")
c5, c6, c7 = st.columns(3)
with c5:
    st.metric("Daily active cost", f"${daily_active_cost:,.0f}")
with c6:
    st.metric("Daily opportunistic cost", f"${daily_opp_cost:,.0f}")
with c7:
    st.metric("Annual ops cost", f"${annual_ops_cost:,.0f}")

if use_transit:
    st.info(f"Transit enabled: upfront integration **${upfront_cost:,.0f}**, annual maintenance included above.")
else:
    st.info("Transit disabled.")

# ============================
# Map
# ============================
st.subheader("Curb segments map")

center = map_gdf.geometry.unary_union.centroid
center_lat, center_lon = float(center.y), float(center.x)

zoom = st.slider("Zoom", 10, 18, 13)
view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=0)

geojson = json.loads(map_gdf.to_json())

halo_layer = pdk.Layer(
    "GeoJsonLayer",
    data=geojson,
    pickable=False,
    stroked=True,
    filled=False,
    get_line_color=[255, 255, 255],
    get_line_width=6,
    line_width_units="pixels",
)

main_layer = pdk.Layer(
    "GeoJsonLayer",
    data=geojson,
    pickable=True,
    stroked=True,
    filled=False,
    get_line_color="properties.line_color",
    get_line_width=3,
    line_width_units="pixels",
)

tooltip = {
    "text": (
        "coverage: {coverage_class}\n"
        "expected refresh: {expected_refresh_min} min\n"
        "block_id: {block_id}\n"
        "seg_id: {seg_id}"
    )
}

st.pydeck_chart(
    pdk.Deck(
        layers=[halo_layer, main_layer],
        initial_view_state=view,
        map_style=CARTO_STYLES[basemap],
        tooltip=tooltip,
    )
)

# ============================
# SLA distribution
# ============================
st.write("### SLA distribution (quick)")
if len(finite) == 0:
    st.warning("No refresh events (all rates are zero). Increase active/opportunistic, or enable transit.")
else:
    bins = [0, 5, 10, 15, 20, 30, 60, 120, np.inf]
    labels = ["≤5", "5–10", "10–15", "15–20", "20–30", "30–60", "60–120", "120+"]
    cats = pd.cut(finite, bins=bins, labels=labels, include_lowest=True, right=True)
    dist = cats.value_counts().sort_index()
    st.write((dist / dist.sum() * 100).round(1).to_dict())

with st.expander("Debug"):
    st.write("Segment columns:", list(seg_gdf.columns))
    st.write("Transit-covered segments (placeholder):", int(transit_flag.sum()))
    st.dataframe(map_gdf.drop(columns=["geometry"], errors="ignore").head(25))