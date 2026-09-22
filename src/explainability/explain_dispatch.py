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
from sklearn.linear_model import Ridge

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
    if isinstance(model, Ridge):
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


def explain(timestamp, sc, tariff=6.52, emission_factor=0.710, top_n=3):
    """Point-wise plain-language explanation for one real-time scenario hour.

    `sc` is the dict returned by src.forecasting.operations.scenario().
    """
    demand_dr = drivers(timestamp, "demand", top_n)
    solar_dr = drivers(timestamp, "solar", top_n)

    def fmt(dr):
        return ", ".join(f"{lbl} ({v:+,.0f})" for lbl, v in dr) or "no single dominant factor"

    backup_without = max(0.0, sc["demand_kw"] - sc["renewable_kw"])
    lines = [
        f"Demand: {sc['demand_kw']:,.0f} kW expected at {sc['datetime']}. Main drivers: {fmt(demand_dr)}.",
        f"Renewable supply: {sc['renewable_kw']:,.0f} kW (solar {sc['solar_kw']:,.0f} kW + wind {sc['wind_kw']:,.0f} kW). Solar drivers: {fmt(solar_dr)}.",
        f"Renewable used directly: {sc['renewable_used_kw']:,.0f} kW = min(renewable supply, demand).",
        f"Storage: charged {sc['battery_charge_kw']:,.0f} kW, discharged {sc['battery_discharge_kw']:,.0f} kW, ending SOC {sc['soc_kwh']:,.0f} kWh.",
        f"Backup needed: {sc['backup_kw']:,.0f} kW = demand - renewable used - battery discharge.",
        f"Curtailed surplus: {sc['curtailed_kw']:,.0f} kW (renewable above demand when battery is full).",
        f"Dispatch order: renewable first, then battery discharge, then backup/grid for the remainder.",
        f"CO2 avoided: {sc['co2_avoided_kg']:,.1f} kg = (backup without storage {backup_without:,.0f} - backup with storage {sc['backup_kw']:,.0f}) kWh x {emission_factor} kg/kWh.",
        f"Cost saved: Rs {sc['cost_saved_rs']:,.0f} = avoided backup kWh x Rs {tariff}/kWh.",
    ]
    return {"demand_drivers": demand_dr, "solar_drivers": solar_dr, "lines": lines}
