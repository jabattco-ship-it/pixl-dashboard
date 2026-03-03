import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="Pixl Parking Dashboard", layout="wide")
st.title("Pixl Parking — Cost & Coverage Dashboard")
st.markdown(
    """
**How to use this demo (30 seconds):**
1. Adjust the SLA target (frequency a parked car is tagged) in the left sidebar.
2. Increase/decrease Active and Opportunistic drivers.
3. Toggle Transit on/off.
4. Watch how cost changes relative to % of segments meeting the SLA.

This models how a city can choose the lowest-cost deployment mix that satisfies a refresh-time target.
"""
)
LOCAL_GEOJSON = os.path.join("data", "curb_segments.geojson")

CARTO_STYLES = {
    "CARTO Positron (light)": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    "CARTO Voyager (streets)": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    "CARTO Dark Matter (dark)": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
}

COLOR_MAP = {
    "transit": [0, 150, 255],
    "opportunistic": [255, 165, 0],
    "active": [220, 0, 90],
    "none": [160, 160, 160],
}

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("SLA")
sla_minutes = st.sidebar.slider("Target refresh SLA (minutes)", 5, 60, 15, 1)

st.sidebar.divider()
st.sidebar.header("Deployment Mix")
use_transit = st.sidebar.checkbox("Use Transit data feed", value=True)
num_opportunistic = st.sidebar.slider("Opportunistic drivers", 0, 500, 80)
num_active = st.sidebar.slider("Active drivers", 0, 80, 12)

st.sidebar.divider()
st.sidebar.header("Cost assumptions")
equip_transit_upfront = st.sidebar.number_input("Transit integration (one-time, $)", 0, value=25000, step=1000)
equip_annual_maint = st.sidebar.number_input("Transit annual maintenance ($/yr)", 0, value=5000, step=500)
active_driver_cost_hr = st.sidebar.number_input("Active driver cost ($/hr)", 0.0, value=35.0, step=1.0)
active_driver_hours_day = st.sidebar.number_input("Active driver hours/day", 0.0, value=6.0, step=0.5)
opportunistic_cost_hr = st.sidebar.number_input("Opportunistic cost ($/hr)", 0.0, value=20.0, step=1.0)
opportunistic_hours_day = st.sidebar.number_input("Opportunistic hours/day", 0.0, value=4.0, step=0.5)

st.sidebar.divider()
st.sidebar.header("Coverage assumptions")
transit_total_hits_per_hour = st.sidebar.number_input(
    "Transit hits/hour across network (if enabled)", 0.0, value=900.0, step=50.0
)
opp_hits_per_driver_per_hour = st.sidebar.number_input("Opportunistic hits/driver/hour", 0.0, value=60.0, step=5.0)
active_hits_per_driver_per_hour = st.sidebar.number_input("Active hits/driver/hour", 0.0, value=140.0, step=10.0)
transit_coverage_share = st.sidebar.slider("Transit coverage share of segments", 0.0, 1.0, 0.40, 0.05)

st.sidebar.divider()
st.sidebar.header("Map")
basemap = st.sidebar.selectbox("Basemap", list(CARTO_STYLES.keys()), index=0)
show_transit = st.sidebar.checkbox("Show Transit layer", value=True)
show_opportunistic = st.sidebar.checkbox("Show Opportunistic layer", value=True)
show_active = st.sidebar.checkbox("Show Active layer", value=True)

# ----------------------------
# Load GeoJSON (no GeoPandas)
# ----------------------------
if not os.path.exists(LOCAL_GEOJSON):
    st.error(f"Missing file: {LOCAL_GEOJSON} (make sure it exists in the repo)")
    st.stop()

@st.cache_data(ttl=3600, show_spinner=True)
def load_geojson(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

gj = load_geojson(LOCAL_GEOJSON)
features = gj.get("features", [])
if not features:
    st.error("GeoJSON has no features.")
    st.stop()

# Add seg_id if missing, and extract some fields into properties we can use
for i, feat in enumerate(features):
    props = feat.setdefault("properties", {})
    props.setdefault("seg_id", i)
    # segment_key is useful later
    if all(k in props for k in ["block_id", "begin_measure", "end_measure"]):
        props.setdefault(
            "segment_key",
            f'{props.get("block_id")}_{props.get("begin_measure")}_{props.get("end_measure")}'
        )
    else:
        props.setdefault("segment_key", str(props["seg_id"]))

N = len(features)

# Compute a simple center (average of all coordinate points)
def iter_coords(geom):
    if not geom:
        return
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if coords is None:
        return
    if gtype == "Point":
        yield coords
    elif gtype in ("LineString",):
        for c in coords:
            yield c
    elif gtype in ("MultiLineString",):
        for line in coords:
            for c in line:
                yield c
    elif gtype in ("Polygon",):
        for ring in coords:
            for c in ring:
                yield c
    elif gtype in ("MultiPolygon",):
        for poly in coords:
            for ring in poly:
                for c in ring:
                    yield c

xs, ys = [], []
for feat in features:
    for c in iter_coords(feat.get("geometry", {})):
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            xs.append(float(c[0]))
            ys.append(float(c[1]))

if xs and ys:
    center_lon = float(np.mean(xs))
    center_lat = float(np.mean(ys))
else:
    center_lat, center_lon = 49.8951, -97.1384  # fallback downtown

# ----------------------------
# Cost model
# ----------------------------
daily_active_cost = num_active * active_driver_cost_hr * active_driver_hours_day
daily_opp_cost = num_opportunistic * opportunistic_cost_hr * opportunistic_hours_day
annual_ops_cost = (daily_active_cost + daily_opp_cost) * 365 + (equip_annual_maint if use_transit else 0)
upfront_cost = equip_transit_upfront if use_transit else 0

# ----------------------------
# SLA model (placeholder, responsive)
# ----------------------------
rng = np.random.default_rng(123)
activity_factor = rng.uniform(0.6, 1.6, size=N)

rng2 = np.random.default_rng(999)
transit_flag = np.zeros(N, dtype=int)
if use_transit and transit_coverage_share > 0:
    k = int(round(transit_coverage_share * N))
    idx = rng2.choice(np.arange(N), size=min(k, N), replace=False)
    transit_flag[idx] = 1

total_transit_hits_hr = transit_total_hits_per_hour if use_transit else 0.0
total_opp_hits_hr = num_opportunistic * opp_hits_per_driver_per_hour
total_active_hits_hr = num_active * active_hits_per_driver_per_hour

n_transit_segments = max(1, int(transit_flag.sum()))
lambda_transit_hr = np.where(transit_flag == 1, total_transit_hits_hr / n_transit_segments, 0.0)
lambda_opp_hr = total_opp_hits_hr / max(1, N)
lambda_active_hr = total_active_hits_hr / max(1, N)

lambda_total_hr = (lambda_transit_hr + lambda_opp_hr + lambda_active_hr) * activity_factor
expected_refresh_min = np.where(lambda_total_hr > 0, 60.0 / lambda_total_hr, np.inf)

pct_meeting_sla = float(np.mean(expected_refresh_min <= sla_minutes) * 100.0)
finite = expected_refresh_min[np.isfinite(expected_refresh_min)]
median_refresh = float(np.median(finite)) if len(finite) else np.inf
p90_refresh = float(np.quantile(finite, 0.90)) if len(finite) else np.inf

# Dominant source visualization
dominant = np.argmax(
    np.vstack([lambda_transit_hr, np.full(N, lambda_opp_hr), np.full(N, lambda_active_hr)]),
    axis=0
)
coverage_class = np.array(["none"] * N, dtype=object)
mask = lambda_total_hr > 0
coverage_class[mask] = np.where(dominant[mask] == 0, "transit",
                        np.where(dominant[mask] == 1, "opportunistic", "active"))

visible = {"none"}
if show_transit: visible.add("transit")
if show_opportunistic: visible.add("opportunistic")
if show_active: visible.add("active")

# Write coverage fields back into GeoJSON properties for deck.gl styling/tooltips
for i, feat in enumerate(features):
    props = feat["properties"]
    props["coverage_class"] = str(coverage_class[i])
    props["expected_refresh_min"] = float(expected_refresh_min[i]) if np.isfinite(expected_refresh_min[i]) else None
    props["transit_flag"] = int(transit_flag[i])
    props["line_color"] = COLOR_MAP.get(props["coverage_class"], COLOR_MAP["none"])

# Filter features for visibility
gj_visible = {
    "type": "FeatureCollection",
    "features": [f for f in features if f["properties"]["coverage_class"] in visible],
}

# ----------------------------
# KPIs
# ----------------------------
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
d1, d2, d3 = st.columns(3)
with d1:
    st.metric("Daily active cost", f"${daily_active_cost:,.0f}")
with d2:
    st.metric("Daily opportunistic cost", f"${daily_opp_cost:,.0f}")
with d3:
    st.metric("Annual ops cost", f"${annual_ops_cost:,.0f}")

st.info(
    f"Transit {'enabled' if use_transit else 'disabled'} — upfront **${upfront_cost:,.0f}** "
    f"({('annual maintenance included' if use_transit else 'no maintenance')})."
)

# ----------------------------
# Map
# ----------------------------
st.subheader("Curb segments map")

zoom = st.slider("Zoom", 10, 18, 13)
view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=0)

halo_layer = pdk.Layer(
    "GeoJsonLayer",
    data=gj_visible,
    pickable=False,
    stroked=True,
    filled=False,
    get_line_color=[255, 255, 255],
    get_line_width=6,
    line_width_units="pixels",
)

main_layer = pdk.Layer(
    "GeoJsonLayer",
    data=gj_visible,
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

# ----------------------------
# SLA distribution (quick)
# ----------------------------
st.write("### SLA distribution (quick)")
if len(finite) == 0:
    st.warning("No refresh events (all rates are zero). Increase active/opportunistic, or enable transit.")
else:
    bins = [0, 5, 10, 15, 20, 30, 60, 120, np.inf]
    labels = ["≤5", "5–10", "10–15", "15–20", "20–30", "30–60", "60–120", "120+"]
    cats = pd.cut(finite, bins=bins, labels=labels, include_lowest=True, right=True)
    dist = cats.value_counts().sort_index()
    st.write((dist / dist.sum() * 100).round(1).to_dict())
