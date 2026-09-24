"""Real-time India energy decision-support dashboard & powerhouse dispatch simulator.

Two dispatch simulators:
  1. MAIN — real-time dispatch on the live-weather forecast anchored at now(),
     across four horizon tabs (Hourly 24h / Daily 7d / Weekly 4w / Monthly 12mo).
  2. SCENARIO — a what-if sandbox: pick a season, festival day, day type and
     weather condition; the SAME trained models forecast a representative day and
     the SAME battery hierarchy dispatches it.

Each simulator presents 3 core zoomable charts (Bounds / Generation Breakdown / Weather Context),
an animated Powerplant Management diagram, and a point-wise SHAP-backed explanation.
"""
from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.operations import (DEFAULT_DEMAND_SCALE, DEFAULT_SOLAR_SCALE,  # noqa: E402
                                        DEFAULT_WIND_SCALE, cached_facility_frame,
                                        dispatch_scenario, impact, load_cached_hourly, to_facility)
from src.forecasting import scenario as scen  # noqa: E402
from src.features.calendar_features import SEASONS  # noqa: E402
from src.explainability.explain_dispatch import explain, explain_from_features  # noqa: E402
from app.visuals import powerhouse_html, weather_chart, demand_breakdown_chart  # noqa: E402

st.set_page_config(page_title="India Energy Decision Support", page_icon="⚡", layout="wide")

FUTURE_FEATURES = ROOT / "data" / "processed" / "forecasts" / "future_features.csv"
TABS = {
    "Hourly · next 24 h": {"block": 1, "n": 24, "picker_label": "Select Hour of Day"},
    "Daily · next 7 days": {"block": 24, "n": 7, "picker_label": "Select Day of Week"},
    "Weekly · next 4 weeks": {"block": 168, "n": 4, "picker_label": "Select Week of Month"},
    "Monthly · next 12 months": {"block": 730, "n": 12, "picker_label": "Select Month of Year"},
}
COLORS = {"Demand": "#e4572e", "Renewable supply": "#2ca02c", "Backup needed": "#1f77b4"}


# ---------------- cached data builders ----------------
@st.cache_data(show_spinner=False)
def build_dispatch(demand_scale, solar_scale, wind_scale, capacity, soc0, power):
    facility = cached_facility_frame(demand_scale, solar_scale, wind_scale)
    return dispatch_scenario(facility, capacity, soc0, power)


@st.cache_data(show_spinner=False)
def load_weather():
    if not FUTURE_FEATURES.exists():
        return pd.DataFrame()
    cols = ["datetime", "temperature_2m_c", "relative_humidity_pct", "cloud_cover_pct",
            "precipitation_mm", "solar_radiation_w_m2", "wind_speed_10m_kmh"]
    ff = pd.read_csv(FUTURE_FEATURES, parse_dates=["datetime"])
    return ff[[c for c in cols if c in ff.columns]]


@st.cache_data(show_spinner=False)
def build_scenario(season, festival, weather, weekend, demand_scale, solar_scale,
                   wind_scale, capacity, soc0, power):
    frame, feats = scen.build_scenario_frame(season, festival, weather, weekend,
                                             return_features=True)
    facility = to_facility(frame, demand_scale, solar_scale, wind_scale)
    disp = dispatch_scenario(facility, capacity, soc0, power)
    disp["season"] = frame["season"].to_numpy()
    disp["festival_name"] = frame["festival_name"].to_numpy()
    for c in ["temperature_2m_c", "relative_humidity_pct", "cloud_cover_pct",
              "precipitation_mm", "solar_radiation_w_m2", "wind_speed_10m_kmh"]:
        if c in frame:
            disp[c] = frame[c].to_numpy()
    return disp, feats


def aggregate_blocks(disp, block, n):
    if block == 1:
        out = disp.iloc[:n].copy()
        out["period"] = out["datetime"]
        return out
    rows = []
    for i in range(n):
        a = i * block
        chunk = disp.iloc[a:a + block]
        if chunk.empty:
            continue
        mean = chunk.mean(numeric_only=True)
        mean["period"] = chunk["datetime"].iloc[0]
        mean["soc_expected_kwh"] = chunk["soc_expected_kwh"].iloc[-1]
        if "renewable_stored_cum_expected_kwh" in chunk:
            mean["renewable_stored_cum_expected_kwh"] = chunk["renewable_stored_cum_expected_kwh"].iloc[-1]
        rows.append(mean)
    return pd.DataFrame(rows)


def aggregate_weather(block, n):
    wx = load_weather()
    if wx.empty:
        return wx
    wx = wx.iloc[:block * n]
    if block == 1:
        return wx.rename(columns={"datetime": "period"})
    rows = []
    for i in range(n):
        chunk = wx.iloc[i * block:(i + 1) * block]
        if chunk.empty:
            continue
        m = chunk.mean(numeric_only=True)
        m["period"] = chunk["datetime"].iloc[0]
        rows.append(m)
    return pd.DataFrame(rows)


def rng(lo, exp, up):
    return f"{exp:,.0f}  ({lo:,.0f}–{up:,.0f})"


def table(view):
    return pd.DataFrame({
        "Period": pd.to_datetime(view["period"]).dt.strftime("%Y-%m-%d %H:%M"),
        "Demand kW": [rng(r.demand_lower_kw, r.demand_expected_kw, r.demand_upper_kw) for r in view.itertuples()],
        "Renewable supply kW": [rng(r.renewable_lower_kw, r.renewable_expected_kw, r.renewable_upper_kw) for r in view.itertuples()],
        "Renewable stored kW": [f"{getattr(r, 'renewable_stored_expected_kw', 0.0):,.0f}" for r in view.itertuples()],
        "Battery SOC kWh": [f"{getattr(r, 'soc_expected_kwh', 0.0):,.0f}" for r in view.itertuples()],
        "Storage discharge kW": [rng(r.battery_discharge_lower_kw, r.battery_discharge_expected_kw, r.battery_discharge_upper_kw) for r in view.itertuples()],
        "Backup needed kW": [rng(r.backup_lower_kw, r.backup_expected_kw, r.backup_upper_kw) for r in view.itertuples()],
    })


def bounds_chart(view):
    frames = []
    for series, key in [("Demand", "demand"), ("Renewable supply", "renewable"), ("Backup needed", "backup")]:
        frames.append(pd.DataFrame({
            "period": pd.to_datetime(view["period"]), "series": series,
            "expected": view[f"{key}_expected_kw"].to_numpy(),
            "lower": view[f"{key}_lower_kw"].to_numpy(),
            "upper": view[f"{key}_upper_kw"].to_numpy()}))
    long = pd.concat(frames, ignore_index=True)
    color = alt.Color("series:N", scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
                      legend=alt.Legend(title="Series"))
    base = alt.Chart(long).encode(x=alt.X("period:T", title="Time Horizon"), color=color)
    area = base.mark_area(opacity=0.16).encode(y=alt.Y("lower:Q", title="Power (kW)"), y2="upper:Q")
    line = base.mark_line(strokeWidth=2.5).encode(y="expected:Q")
    return (area + line).resolve_scale(color="shared").properties(height=320).interactive()


def sc_dict(row, dt, tariff, ef):
    demand = float(row.demand_expected_kw)
    renewable = float(row.renewable_expected_kw)
    backup = float(row.backup_expected_kw)
    backup_without = max(0.0, demand - renewable)
    saved = backup_without - backup
    return {"datetime": str(dt), "demand_kw": demand,
            "solar_kw": float(row.solar_expected_kw), "wind_kw": float(row.wind_expected_kw),
            "renewable_kw": renewable, "renewable_used_kw": float(row.renewable_used_expected_kw),
            "battery_charge_kw": float(getattr(row, "renewable_stored_expected_kw", row.battery_charge_expected_kw)),
            "battery_discharge_kw": float(row.battery_discharge_expected_kw),
            "backup_kw": backup, "curtailed_kw": float(row.curtailed_expected_kw),
            "soc_kwh": float(row.soc_expected_kwh), "backup_without_storage_kw": backup_without,
            "cost_saved_rs": saved * tariff, "co2_avoided_kg": saved * ef}


def powerhouse(sc, capacity):
    components.html(powerhouse_html(sc, capacity, sc["soc_kwh"]), height=410, scrolling=False)


def metrics_row(imp, demand_kwh, renewable_kwh):
    st.markdown("""
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.80rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    k = st.columns(7)
    k[0].metric("Demand energy", f"{demand_kwh:,.0f} kWh")
    k[1].metric("Renewable energy", f"{renewable_kwh:,.0f} kWh")
    k[2].metric("Stored to battery", f"{imp.get('stored_kwh', 0.0):,.0f} kWh")
    k[3].metric("Discharged from battery", f"{imp.get('discharged_kwh', 0.0):,.0f} kWh")
    k[4].metric("Backup energy", f"{imp['backup_with_storage_kwh']:,.0f} kWh")
    k[5].metric("Cost saved", f"₹{imp['cost_savings_rs']:,.0f}")
    k[6].metric("CO₂ avoided", f"{imp['co2_avoided_kg']:,.1f} kg")


# ---------------- sidebar (reference assumptions) ----------------
st.sidebar.header("⚡ Reference Assumptions")
anchor = load_cached_hourly().datetime.iloc[0]
st.sidebar.caption(f"Forecast anchored at **{anchor}** IST (live weather + learned "
                   "season/festival patterns).")
with st.sidebar.expander("Facility sizing & battery", expanded=False):
    demand_scale = st.slider("Demand scale (kW per national MW)", 0.01, 0.15, DEFAULT_DEMAND_SCALE, 0.01)
    approx_load_kw = demand_scale * 180000
    st.caption(f"💡 Approx Facility Peak Load: **{approx_load_kw:,.0f} kW** ({approx_load_kw/1000:,.1f} MW)")

    solar_scale = st.slider("Solar array size (× plant kW)", 1.0, 30.0, DEFAULT_SOLAR_SCALE, 0.5)
    approx_solar_kw = solar_scale * 700
    st.caption(f"💡 Approx Midday Solar Peak: **{approx_solar_kw:,.0f} kW** ({approx_solar_kw/1000:,.1f} MW)")

    wind_scale = st.slider("Wind array size (× plant kW)", 10.0, 500.0, DEFAULT_WIND_SCALE, 10.0)
    approx_wind_kw = wind_scale * 25
    st.caption(f"💡 Approx Wind Peak Capacity: **{approx_wind_kw:,.0f} kW** ({approx_wind_kw/1000:,.1f} MW)")

    capacity = st.slider("Battery capacity (kWh)", 500, 20000, 5000, 500)
    soc0 = st.slider("Initial battery SOC (%)", 0, 100, 50)
    power = st.slider("Battery power limit (kW)", 100, 5000, 1000, 100)
with st.sidebar.expander("Tariff & emission factor", expanded=False):
    tariff = st.slider("Backup tariff (₹/kWh)", 2.0, 15.0, 6.52, 0.01)
    factor = st.slider("Grid emission factor (kg CO₂/kWh)", 0.2, 1.2, 0.710, 0.001)

st.title("⚡ India Real-Time Energy Forecast & Dispatch Simulator")
st.caption("Live forecasting, energy balance, and powerhouse dispatch simulator with SHAP explainability & what-if scenario testing.")

# =====================================================================
# SIMULATOR 1 — MAIN REAL-TIME DISPATCH
# =====================================================================
st.header("🟢 Main Dispatch Simulator — Real-Time Forecast")
disp = build_dispatch(demand_scale, solar_scale, wind_scale, capacity, soc0, power)

for (label, cfg), tab in zip(TABS.items(), st.tabs(list(TABS.keys()))):
    with tab:
        view = aggregate_blocks(disp, cfg["block"], cfg["n"])
        horizon = disp.iloc[:cfg["block"] * cfg["n"]]
        imp = impact(horizon, tariff, factor)
        imp["stored_kwh"] = float(horizon["renewable_stored_expected_kw"].sum()) \
            if "renewable_stored_expected_kw" in horizon else 0.0
        imp["discharged_kwh"] = float(horizon["battery_discharge_expected_kw"].sum()) \
            if "battery_discharge_expected_kw" in horizon else 0.0

        metrics_row(imp, horizon.demand_expected_kw.sum(), horizon.renewable_expected_kw.sum())

        st.markdown("---")
        st.subheader("📋 Forecast Data Table (Expected values with Lower–Upper prediction bounds)")
        st.dataframe(table(view), hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 3 Core Interactive Visualizations")

        st.markdown("#### 1️⃣ Real-Time Bounds & Energy Balance (Demand vs Renewable vs Backup)")
        st.altair_chart(bounds_chart(view), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 2️⃣ Generation & Load Breakdown (Demand vs Solar vs Wind)")
            bc = demand_breakdown_chart(view, time_col="period", height=320)
            if bc is not None:
                st.altair_chart(bc, use_container_width=True)
        with c2:
            st.markdown("#### 3️⃣ Combined Weather Context Attributes")
            wx = aggregate_weather(cfg["block"], cfg["n"])
            wc = weather_chart(wx.rename(columns={"period": "datetime"}), height=320)
            if wc is not None:
                st.altair_chart(wc, use_container_width=True)
            else:
                st.info("Weather context unavailable.")

        st.markdown("---")
        st.subheader("🏭 Live Powerhouse Dispatch — Select Period")
        periods = pd.to_datetime(view["period"]).dt.strftime("%Y-%m-%d %H:%M").tolist()

        is_hourly = (cfg["n"] == 24 and cfg["block"] == 1)
        default_idx = 6 if is_hourly else 0

        import time
        time_key = f"select_time_{label}"
        pick_key = f"pick_{label}"

        now_t = time.time()
        if time_key not in st.session_state:
            st.session_state[time_key] = now_t

        if pick_key not in st.session_state:
            st.session_state[pick_key] = default_idx

        # Selective auto-shift back to running live hour without reloading whole dashboard
        elapsed = now_t - st.session_state[time_key]
        if is_hourly and st.session_state[pick_key] != default_idx:
            if elapsed >= 30:
                st.session_state[pick_key] = default_idx
                st.session_state[time_key] = now_t
                st.rerun()

        def format_period_label(i):
            p_str = periods[i]
            if is_hourly:
                if i == 6:
                    return f"🔴 {p_str} (LIVE RUNNING HOUR)"
                elif i < 6:
                    return f"⏮️ {p_str} (-{6 - i}h past)"
                else:
                    return f"⏭️ {p_str} (+{i - 6}h future forecast)"
            return f"{p_str} (Period {i + 1} of {len(view)})"

        def on_picker_change():
            st.session_state[time_key] = time.time()

        pick = st.selectbox(
            f"{cfg['picker_label']}",
            range(len(view)),
            format_func=format_period_label,
            key=pick_key,
            on_change=on_picker_change,
        )

        if is_hourly and pick != default_idx:
            rem = max(1, int(30 - (time.time() - st.session_state[time_key])))
            st.caption(f"⏱️ Temporarily inspecting Period {pick + 1}. Auto-shifting back to Running Live Hour in {rem}s...")

            @st.fragment(run_every=max(1, rem))
            def auto_hour_shifter():
                if time.time() - st.session_state[time_key] >= 30:
                    st.session_state[pick_key] = default_idx
                    st.session_state[time_key] = time.time()
                    st.rerun()

            auto_hour_shifter()

        row = view.iloc[pick]
        dt = pd.to_datetime(row.period)
        sc = sc_dict(row, dt, tariff, factor)
        powerhouse(sc, capacity)

        st.subheader("🔍 Explainability (Point-wise SHAP Attributions)")
        ex = explain(dt, sc, tariff, factor)
        for line in ex["lines"]:
            st.markdown(f"- {line}")

# =====================================================================
# SIMULATOR 2 — SCENARIO WHAT-IF DISPATCH
# =====================================================================
st.markdown("---")
st.header("🟡 Scenario Dispatch Simulator — What-If Sandbox")
st.caption("Build a hypothetical day (season · festival · day type · weather). The trained models "
           "forecast it and the same battery hierarchy dispatches it.")

cc = st.columns(4)
season = cc[0].selectbox("Season", SEASONS, index=1)
festival = cc[1].selectbox("Festival day", scen.festival_choices(), index=0)
weather = cc[2].selectbox("Weather condition", scen.PRESET_NAMES, index=0)
weekend = cc[3].selectbox("Day type", ["Weekday", "Weekend"], index=0) == "Weekend"

sdisp, sfeats = build_scenario(season, festival, weather, weekend, demand_scale, solar_scale,
                               wind_scale, capacity, soc0, power)
lab = scen.scenario_labels(season, festival, weather, weekend)
if festival != "No festival":
    st.markdown(f"**Selected Scenario:** {lab['season']} · {lab['festival']} ({lab['date']}) · {lab['weather']} · {lab['day_type']}")
else:
    st.markdown(f"**Selected Scenario:** {lab['season']} · {lab['festival']} · {lab['weather']} · {lab['day_type']}")

SCEN_TABS = {
    "Hourly · 24 h": {"block": 1, "n": 24, "picker_label": "Select Hour of Scenario Day", "fmt": "%H:%M"},
    "Daily · 7 days": {"block": 24, "n": 7, "picker_label": "Select Day of Scenario Week", "fmt": "%Y-%m-%d (%a)"},
    "Weekly · 4 weeks": {"block": 168, "n": 4, "picker_label": "Select Week of Scenario Month", "fmt": "Week %U (%Y-%m-%d)"},
    "Monthly · 12 months": {"block": 730, "n": 12, "picker_label": "Select Month of Scenario Year", "fmt": "%B %Y"},
}

for (slab, scfg), stab in zip(SCEN_TABS.items(), st.tabs(list(SCEN_TABS.keys()))):
    with stab:
        sview = aggregate_blocks(sdisp, scfg["block"], scfg["n"])
        shorizon = sdisp.iloc[:scfg["block"] * scfg["n"]]
        simp = impact(shorizon, tariff, factor)
        simp["stored_kwh"] = float(shorizon["renewable_stored_expected_kw"].sum()) \
            if "renewable_stored_expected_kw" in shorizon else 0.0
        simp["discharged_kwh"] = float(shorizon["battery_discharge_expected_kw"].sum()) \
            if "battery_discharge_expected_kw" in shorizon else 0.0

        metrics_row(simp, shorizon.demand_expected_kw.sum(), shorizon.renewable_expected_kw.sum())

        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.subheader("1️⃣ Scenario Demand & Renewable Generation Breakdown")
            sbc = demand_breakdown_chart(sview, time_col="period", height=320)
            if sbc is not None:
                st.altair_chart(sbc, use_container_width=True)
        with c2:
            st.subheader("2️⃣ Scenario Weather Attributes")
            swx = aggregate_weather(scfg["block"], scfg["n"])
            wc = weather_chart(sdisp if scfg["block"] == 1 else swx.rename(columns={"period": "datetime"}), height=320)
            if wc is not None:
                st.altair_chart(wc, use_container_width=True)

        st.subheader("🏭 What-If Powerhouse Dispatch")
        speriods = pd.to_datetime(sview["period"]).dt.strftime(scfg["fmt"]).tolist()
        spick = st.selectbox(f"{scfg['picker_label']}", range(len(sview)),
                             format_func=lambda i: speriods[i], index=0, key=f"scen_pick_{slab}")
        srow = sview.iloc[spick]
        sdt = pd.to_datetime(srow.period)
        ssc = sc_dict(srow, sdt, tariff, factor)
        powerhouse(ssc, capacity)

        st.subheader("🔍 Explainability (Point-wise SHAP Attributions)")
        frow = sfeats.iloc[min(spick * scfg["block"], len(sfeats) - 1)].to_dict()
        frow["datetime"] = sdt
        sex = explain_from_features(frow, ssc, tariff, factor)
        for line in sex["lines"]:
            st.markdown(f"- {line}")
