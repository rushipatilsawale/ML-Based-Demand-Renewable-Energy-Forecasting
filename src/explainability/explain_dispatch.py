"""SHAP explainability for the real-time dispatch, CO2, and cost calculations.

This does NOT explain model selection. For a chosen real-time hour it attributes
*why the demand and renewable forecasts took the values that drive dispatch*,
then walks through the dispatch / CO2-avoided / cost-saved arithmetic in plain,
point-wise language.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

ROOT = Path(__file__).resolve().parents[2]
PACK_PATH = ROOT / "models" / "realtime_forecasters.joblib"
FUTURE_FEATURES = ROOT / "data" / "processed" / "forecasts" / "future_features.csv"
HISTORY = ROOT / "data" / "processed" / "forecast_features.csv"

LABELS = {
    "hour_sin": "time of day", "hour_cos": "time of day",
    "day_of_week_sin": "day of week", "day_of_week_cos": "day of week",
    "month_sin": "month/season", "month_cos": "month/season",
    "is_weekend": "weekend", "is_night": "nighttime",
    "temperature_2m_c": "temperature", "relative_humidity_pct": "humidity",
    "cloud_cover_pct": "cloud cover", "precipitation_mm": "precipitation",
    "solar_radiation_w_m2": "solar radiation", "wind_speed_10m_kmh": "wind speed",
    "wind_speed_10m_ms": "wind speed",
    "national_demand_lag_1h": "demand 1h ago", "national_demand_lag_24h": "demand yesterday",
    "national_demand_lag_168h": "demand last week",
    "national_demand_rolling_mean_24h": "24h average demand",
    "national_demand_rolling_std_24h": "24h demand volatility",
    "national_demand_rolling_mean_168h": "weekly average demand",
    "national_demand_rolling_std_168h": "weekly demand volatility",
    "solar_generation_lag_1h": "solar 1h ago", "solar_generation_lag_24h": "solar yesterday",
    "solar_generation_lag_168h": "solar last week",
    "solar_generation_rolling_mean_24h": "24h average solar",
    "solar_generation_rolling_std_24h": "24h solar volatility",
    "solar_generation_rolling_mean_168h": "weekly average solar",
    "solar_generation_rolling_std_168h": "weekly solar volatility",
    "wind_power_potential_lag_1h": "wind 1h ago", "wind_power_potential_lag_24h": "wind yesterday",
    "wind_power_potential_lag_168h": "wind last week",
    "wind_power_potential_rolling_mean_24h": "24h average wind",
    "wind_power_potential_rolling_std_24h": "24h wind volatility",
    "wind_power_potential_rolling_mean_168h": "weekly average wind",
    "wind_power_potential_rolling_std_168h": "weekly wind volatility",
}

_pack = None
_explainers = {}
_feats = None


def _label(col):
    return LABELS.get(col, col.replace("_", " "))


def _time_of_day(hour):
    if 5 <= hour < 9:
        return "early morning ramp-up"
    if 9 <= hour < 12:
        return "late morning"
    if 12 <= hour < 15:
        return "mid-day (solar peak)"
    if 15 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 22:
        return "evening demand peak"
    return "overnight low-demand period"


def context_lines(row):
    """Plain-language 'why' for one hour from its season / festival / weather /
    time-of-day context. `row` is any mapping carrying the label + weather keys
    (a future_features.csv row for real-time, or a scenario frame row for what-if).

    This complements the SHAP drivers: SHAP quantifies each feature's push on the
    forecast; these lines translate the same context into an operator narrative.
    """
    if row is None:
        return []
    ts = pd.to_datetime(row.get("datetime"))
    hour = int(ts.hour) if pd.notna(ts) else 12
    season = row.get("season")
    festival = row.get("festival_name")
    lines = []

    ctx = [str(season) + " season"] if season and str(season) != "nan" else []
    ctx.append(_time_of_day(hour))
    if festival and str(festival) not in ("nan", "No festival", "None"):
        ctx.append(f"{festival} festival day")
    if ctx:
        lines.append("Context: " + ", ".join(ctx) + ".")

    def num(key):
        v = row.get(key)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    reasons = []
    cloud = num("cloud_cover_pct")
    rad = num("solar_radiation_w_m2")
    temp = num("temperature_2m_c")
    hum = num("relative_humidity_pct")
    precip = num("precipitation_mm")
    wind = num("wind_speed_10m_kmh")

    if rad is not None:
        if rad >= 700:
            reasons.append(f"strong solar radiation ({rad:,.0f} W/m²) drives high mid-day solar generation")
        elif rad <= 150 and 6 <= hour <= 18:
            reasons.append(f"weak solar radiation ({rad:,.0f} W/m²) limits solar generation")
    if cloud is not None and cloud >= 70:
        reasons.append(f"heavy cloud cover ({cloud:,.0f}%) suppresses solar output")
    if temp is not None and temp >= 35:
        reasons.append(f"high temperature ({temp:,.0f}°C) raises cooling-driven demand")
    elif temp is not None and temp <= 15:
        reasons.append(f"cool temperature ({temp:,.0f}°C) lowers cooling load")
    if hum is not None and hum >= 75:
        reasons.append(f"high humidity ({hum:,.0f}%) adds to perceived heat and load")
    if precip is not None and precip >= 1.0:
        reasons.append(f"rainfall ({precip:,.1f} mm) keeps demand indoor-heavy and solar low")
    if wind is not None and wind >= 25:
        reasons.append(f"strong wind ({wind:,.0f} km/h) lifts wind generation")
    if festival and str(festival) not in ("nan", "No festival", "None"):
        reasons.append(f"{festival} typically shifts demand (festive lighting/cooking, offices closed)")
    if season and str(season) != "nan":
        reasons.append(f"{season} seasonal pattern")

    if reasons:
        lines.append("Why demand/renewable look like this: " + "; ".join(reasons) + ".")
    return lines


def _context_row(timestamp):
    """Fetch the persisted future_features row (labels + weather) for a timestamp."""
    feats = _load_features()
    if feats is None:
        return None
    row = feats.loc[feats.datetime == pd.Timestamp(timestamp)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()



def _load_pack():
    global _pack
    if _pack is None:
        _pack = joblib.load(PACK_PATH)
    return _pack


def _load_features():
    global _feats
    if _feats is None and FUTURE_FEATURES.exists():
        _feats = pd.read_csv(FUTURE_FEATURES, parse_dates=["datetime"])
    return _feats


def _explainer(name):
    if name in _explainers:
        return _explainers[name]
    info = _load_pack()["models"][name]
    model = info["model"]
    # The selected demand model is linear_regression (solar/wind are tree-based),
    # so detect any linear estimator by coef_ rather than hard-coding Ridge.
    is_linear = isinstance(model, (LinearRegression, Ridge, Lasso, ElasticNet))
    if is_linear:
        hist = pd.read_csv(HISTORY).tail(500)[info["features"]]
        ex = shap.LinearExplainer(model, hist)
    else:
        ex = shap.TreeExplainer(model)
    _explainers[name] = ex
    return ex


def drivers(timestamp, name, top_n=3):
    """Top SHAP feature drivers for one target's forecast at a given timestamp."""
    feats = _load_features()
    if feats is None:
        return []
    row = feats.loc[feats.datetime == pd.Timestamp(timestamp)]
    if row.empty:
        return []
    info = _load_pack()["models"][name]
    x = row[info["features"]]
    values = np.asarray(_explainer(name).shap_values(x)).ravel()
    order = np.argsort(-np.abs(values))[:top_n]
    return [(_label(info["features"][i]), float(values[i])) for i in order]


def drivers_from_row(feature_row, name, top_n=3):
    """Top SHAP drivers for one target from an explicit feature mapping.

    Used by the scenario (what-if) simulator, whose representative dates are not
    present in the persisted real-time future_features.csv.
    """
    if feature_row is None:
        return []
    info = _load_pack()["models"][name]
    try:
        x = pd.DataFrame([{f: feature_row[f] for f in info["features"]}])
    except KeyError:
        return []
    values = np.asarray(_explainer(name).shap_values(x)).ravel()
    order = np.argsort(-np.abs(values))[:top_n]
    return [(_label(info["features"][i]), float(values[i])) for i in order]


def explain_from_features(feature_row, sc, tariff=6.52, emission_factor=0.710, top_n=3):
    """Full explanation for a scenario hour using an explicit feature row for both
    the SHAP drivers and the context narrative (no future_features.csv lookup)."""
    demand_dr = drivers_from_row(feature_row, "demand", top_n)
    is_solar_zero = sc.get("solar_kw", 0.0) <= 0.001
    solar_dr = [] if is_solar_zero else drivers_from_row(feature_row, "solar", top_n)

    def fmt(dr, is_solar=False):
        if is_solar and is_solar_zero:
            return "nighttime (solar radiation is 0 W/m²)"
        return ", ".join(f"{lbl} ({v:+,.0f})" for lbl, v in dr) or "no single dominant factor"

    backup_without = max(0.0, sc["demand_kw"] - sc["renewable_kw"])
    lines = context_lines(feature_row) + [
        f"Demand: {sc['demand_kw']:,.0f} kW expected at {sc['datetime']}. Main drivers: {fmt(demand_dr)}.",
        f"Renewable supply: {sc['renewable_kw']:,.0f} kW (solar {sc['solar_kw']:,.0f} kW + wind {sc['wind_kw']:,.0f} kW). Solar drivers: {fmt(solar_dr, is_solar=True)}.",
        f"Renewable used directly: {sc['renewable_used_kw']:,.0f} kW = min(renewable supply, demand).",
        f"Renewable stored to battery: {sc['battery_charge_kw']:,.0f} kW of surplus parked this hour (auto-filled, never manual).",
        f"Storage: discharged {sc['battery_discharge_kw']:,.0f} kW, ending SOC {sc['soc_kwh']:,.0f} kWh.",
        f"Backup needed: {sc['backup_kw']:,.0f} kW = demand - renewable used - battery discharge.",
        f"Curtailed surplus: {sc['curtailed_kw']:,.0f} kW (renewable above demand when battery is full).",
        f"Dispatch order: renewable first, then battery discharge, then backup/grid for the remainder.",
        f"CO2 avoided: {sc['co2_avoided_kg']:,.1f} kg = (backup without storage {backup_without:,.0f} - backup with storage {sc['backup_kw']:,.0f}) kWh x {emission_factor} kg/kWh.",
        f"Cost saved: Rs {sc['cost_saved_rs']:,.0f} = avoided backup kWh x Rs {tariff}/kWh.",
    ]
    return {"demand_drivers": demand_dr, "solar_drivers": solar_dr,
            "context": feature_row, "lines": lines}


def explain(timestamp, sc, tariff=6.52, emission_factor=0.710, top_n=3, context=None):
    """Point-wise plain-language explanation for one real-time scenario hour.

    `sc` is the dict returned by src.forecasting.operations.scenario().
    `context` (optional) is a mapping with season/festival/weather keys used for
    the narrative; when omitted it is read from the persisted future_features row
    at `timestamp` (the real-time path). The scenario simulator passes its own row.
    """
    demand_dr = drivers(timestamp, "demand", top_n)
    is_solar_zero = sc.get("solar_kw", 0.0) <= 0.001
    solar_dr = [] if is_solar_zero else drivers(timestamp, "solar", top_n)

    def fmt(dr, is_solar=False):
        if is_solar and is_solar_zero:
            return "nighttime (solar radiation is 0 W/m²)"
        return ", ".join(f"{lbl} ({v:+,.0f})" for lbl, v in dr) or "no single dominant factor"

    ctx_row = context if context is not None else _context_row(timestamp)
    backup_without = max(0.0, sc["demand_kw"] - sc["renewable_kw"])
    lines = context_lines(ctx_row) + [
        f"Demand: {sc['demand_kw']:,.0f} kW expected at {sc['datetime']}. Main drivers: {fmt(demand_dr)}.",
        f"Renewable supply: {sc['renewable_kw']:,.0f} kW (solar {sc['solar_kw']:,.0f} kW + wind {sc['wind_kw']:,.0f} kW). Solar drivers: {fmt(solar_dr, is_solar=True)}.",
        f"Renewable used directly: {sc['renewable_used_kw']:,.0f} kW = min(renewable supply, demand).",
        f"Renewable stored to battery: {sc['battery_charge_kw']:,.0f} kW of surplus parked this hour (auto-filled, never manual).",
        f"Storage: discharged {sc['battery_discharge_kw']:,.0f} kW, ending SOC {sc['soc_kwh']:,.0f} kWh.",
        f"Backup needed: {sc['backup_kw']:,.0f} kW = demand - renewable used - battery discharge.",
        f"Curtailed surplus: {sc['curtailed_kw']:,.0f} kW (renewable above demand when battery is full).",
        f"Dispatch order: renewable first, then battery discharge, then backup/grid for the remainder.",
        f"CO2 avoided: {sc['co2_avoided_kg']:,.1f} kg = (backup without storage {backup_without:,.0f} - backup with storage {sc['backup_kw']:,.0f}) kWh x {emission_factor} kg/kWh.",
        f"Cost saved: Rs {sc['cost_saved_rs']:,.0f} = avoided backup kWh x Rs {tariff}/kWh.",
    ]
    return {"demand_drivers": demand_dr, "solar_drivers": solar_dr,
            "context": ctx_row, "lines": lines}
