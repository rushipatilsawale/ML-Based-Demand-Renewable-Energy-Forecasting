"""Real-time India energy decision-support dashboard.

Four tabs (Hourly / Daily / Weekly / Monthly). Each shows a ranged table, a bounds
chart, a scenario-based dispatch simulator, and a point-wise SHAP-backed
explanation of the dispatch, CO2, and cost calculations.
"""
from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.operations import (cached_facility_frame, dispatch_scenario,  # noqa: E402
                                        impact, load_cached_hourly)
from src.explainability.explain_dispatch import explain  # noqa: E402

st.set_page_config(page_title="India Energy Decision Support", page_icon="⚡", layout="wide")

TABS = {
    "Hourly · next 24 h": {"block": 1, "n": 24},
    "Daily · next 7 days": {"block": 24, "n": 7},
    "Weekly · next 4 weeks": {"block": 168, "n": 4},
    "Monthly · next 12 months": {"block": 730, "n": 12},
}
COLORS = {"Demand": "#e4572e", "Renewable supply": "#2ca02c", "Backup needed": "#1f77b4"}


@st.cache_data(show_spinner=False)
def build_dispatch(demand_scale, solar_scale, wind_scale, capacity, soc0, power):
    facility = cached_facility_frame(demand_scale, solar_scale, wind_scale)
    return dispatch_scenario(facility, capacity, soc0, power)


def aggregate_blocks(disp, block, n):
    if block == 1:
        out = disp.iloc[:n].copy()
        out["period"] = out["datetime"]
        out["_idx"] = range(len(out))
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
        mean["_idx"] = a
        rows.append(mean)
    return pd.DataFrame(rows)


def rng(lo, exp, up):
    return f"{exp:,.0f}  ({lo:,.0f}–{up:,.0f})"


def table(view):
    return pd.DataFrame({
        "Period": pd.to_datetime(view["period"]).dt.strftime("%Y-%m-%d %H:%M"),
        "Demand kW": [rng(r.demand_lower_kw, r.demand_expected_kw, r.demand_upper_kw) for r in view.itertuples()],
        "Renewable supply kW": [rng(r.renewable_lower_kw, r.renewable_expected_kw, r.renewable_upper_kw) for r in view.itertuples()],
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
    base = alt.Chart(long).encode(x=alt.X("period:T", title="Real-time horizon"), color=color)
    area = base.mark_area(opacity=0.16).encode(y=alt.Y("lower:Q", title="kW"), y2="upper:Q")
    line = base.mark_line(strokeWidth=2).encode(y="expected:Q")
    return (area + line).resolve_scale(color="shared").properties(height=380).interactive()


def row_scenario(row, tariff, ef):
    demand = float(row.demand_expected_kw)
    renewable = float(row.renewable_expected_kw)
    backup = float(row.backup_expected_kw)
    backup_without = max(0.0, demand - renewable)
    saved = backup_without - backup
    return {"datetime": str(pd.to_datetime(row.period)), "demand_kw": demand,
            "solar_kw": float(row.solar_expected_kw), "wind_kw": float(row.wind_expected_kw),
            "renewable_kw": renewable, "renewable_used_kw": float(row.renewable_used_expected_kw),
            "battery_charge_kw": float(row.battery_charge_expected_kw),
            "battery_discharge_kw": float(row.battery_discharge_expected_kw),
            "backup_kw": backup, "curtailed_kw": float(row.curtailed_expected_kw),
            "soc_kwh": float(row.soc_expected_kwh), "backup_without_storage_kw": backup_without,
            "cost_saved_rs": saved * tariff, "co2_avoided_kg": saved * ef}


# ---------------- sidebar ----------------
st.sidebar.header("⚡ Real-time controls")
anchor = load_cached_hourly().datetime.iloc[0]
st.sidebar.caption(f"Forecast anchored at **{anchor}** IST (live weather + learned patterns). "
                   "Re-run `src/models/realtime_forecast.py` to refresh.")
demand_scale = st.sidebar.slider("Facility demand scale (kW per national MW)", 0.01, 0.15, 0.05, 0.01)
solar_scale = st.sidebar.slider("Facility solar scale", 1.0, 15.0, 5.0, 0.5)
wind_scale = st.sidebar.slider("Facility wind scale", 1.0, 15.0, 1.0, 0.5)
capacity = st.sidebar.slider("Battery capacity (kWh)", 500, 20000, 5000, 500)
soc0 = st.sidebar.slider("Initial battery SOC (%)", 0, 100, 50)
power = st.sidebar.slider("Battery power limit (kW)", 100, 5000, 1000, 100)
tariff = st.sidebar.slider("Backup fuel/grid tariff (₹/kWh)", 2.0, 15.0, 6.52, 0.01)
factor = st.sidebar.slider("Grid emission factor (kg CO₂/kWh)", 0.2, 1.2, 0.710, 0.001)

st.title("⚡ India Real-Time Energy Forecast & Dispatch")
disp = build_dispatch(demand_scale, solar_scale, wind_scale, capacity, soc0, power)

tab_labels = list(TABS.keys())
for (label, cfg), tab in zip(TABS.items(), st.tabs(tab_labels)):
    with tab:
        view = aggregate_blocks(disp, cfg["block"], cfg["n"])
        horizon = disp.iloc[:cfg["block"] * cfg["n"]]
        imp = impact(horizon, tariff, factor)

        k = st.columns(5)
        k[0].metric("Demand energy", f"{horizon.demand_expected_kw.sum():,.0f} kWh")
        k[1].metric("Renewable energy", f"{horizon.renewable_expected_kw.sum():,.0f} kWh")
        k[2].metric("Backup energy", f"{imp['backup_with_storage_kwh']:,.0f} kWh")
        k[3].metric("Cost saved", f"₹{imp['cost_savings_rs']:,.0f}")
        k[4].metric("CO₂ avoided", f"{imp['co2_avoided_kg']:,.1f} kg")

        st.subheader("Forecast table (expected with lower–upper range)")
        st.dataframe(table(view), hide_index=True, use_container_width=True)

        st.subheader("Bounds chart")
        st.altair_chart(bounds_chart(view), use_container_width=True)

        st.subheader("Scenario-based dispatch simulator")
        periods = pd.to_datetime(view["period"]).dt.strftime("%Y-%m-%d %H:%M").tolist()
        pick = st.selectbox("Select a scenario time", range(len(view)),
                            format_func=lambda i: periods[i], key=f"pick_{label}")
        row = view.iloc[pick]
        sc = row_scenario(row, tariff, factor)
        c = st.columns(6)
        c[0].metric("Demand", f"{sc['demand_kw']:,.0f} kW")
        c[1].metric("Renewable used", f"{sc['renewable_used_kw']:,.0f} kW")
        c[2].metric("Battery discharge", f"{sc['battery_discharge_kw']:,.0f} kW")
        c[3].metric("Backup needed", f"{sc['backup_kw']:,.0f} kW")
        c[4].metric("Cost saved", f"₹{sc['cost_saved_rs']:,.0f}")
        c[5].metric("CO₂ avoided", f"{sc['co2_avoided_kg']:,.1f} kg")

        st.subheader("Explanation (point-wise, SHAP-backed)")
        ex = explain(pd.to_datetime(row.period), sc, tariff, factor)
        for line in ex["lines"]:
            st.markdown(f"- {line}")
