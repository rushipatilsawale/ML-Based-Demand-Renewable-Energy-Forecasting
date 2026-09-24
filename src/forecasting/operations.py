"""Real-time scenario dispatch, energy balance, and CO2/cost impact engine.

Consumes the trained recursive forecast (src/models/realtime_forecast.py), scales
national demand and plant renewables into a facility/microgrid frame, runs the
battery dispatch hierarchy per bound (lower / expected / upper), and quantifies
backup-fuel cost and CO2 avoided by forecast-driven dispatch versus a no-storage
counterfactual.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.realtime_forecast import forecast_hourly

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "processed" / "forecasts"

LEVELS = ("lower", "expected", "upper")
# Facility-scale factors (audit "Microgrid Scale Mode"); user-adjustable sliders.
# Solar is sized so a clear midday genuinely over-produces vs facility demand and
# the surplus auto-fills the battery (the storage story only exists if renewable
# can exceed demand). Demand x0.05 -> ~8 MW facility; solar x14 -> ~9 MWp array.
DEFAULT_DEMAND_SCALE = 0.05   # national MW -> facility kW
DEFAULT_SOLAR_SCALE = 14.0    # plant kW -> facility solar kW
DEFAULT_WIND_SCALE = 200.0    # plant kW -> facility wind kW (sized for ~3 MW wind array)
ETA = 0.9                     # round-trip charge/discharge efficiency


def historical_data():
    df = pd.read_csv(ROOT / "data/processed/aligned_hourly_dataset.csv", parse_dates=["datetime"])
    return df.sort_values("datetime")


def to_facility(hourly, demand_scale=DEFAULT_DEMAND_SCALE, solar_scale=DEFAULT_SOLAR_SCALE,
                wind_scale=DEFAULT_WIND_SCALE):
    """Scale a national-unit hourly forecast (MW/kW) into facility kW with bounds."""
    out = pd.DataFrame({"datetime": pd.to_datetime(hourly["datetime"]).to_numpy()})
    for b in LEVELS:
        out[f"demand_{b}_kw"] = hourly[f"demand_mw_{b}"].to_numpy(float) * demand_scale
        out[f"solar_{b}_kw"] = hourly[f"solar_kw_{b}"].to_numpy(float) * solar_scale
        out[f"wind_{b}_kw"] = hourly[f"wind_kw_{b}"].to_numpy(float) * wind_scale
        out[f"renewable_{b}_kw"] = out[f"solar_{b}_kw"] + out[f"wind_{b}_kw"]
    return out


def load_cached_hourly():
    """Read the precomputed real-time hourly forecast cache (national units)."""
    return pd.read_csv(CACHE_DIR / "hourly_forecast.csv", parse_dates=["datetime"])


def cached_facility_frame(demand_scale=DEFAULT_DEMAND_SCALE, solar_scale=DEFAULT_SOLAR_SCALE,
                          wind_scale=DEFAULT_WIND_SCALE):
    df = load_cached_hourly()
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).floor("h")
    start_time = now - pd.Timedelta(hours=6)
    if start_time in df["datetime"].values:
        idx = df[df["datetime"] == start_time].index[0]
        df = df.iloc[idx:].reset_index(drop=True)
    return to_facility(df, demand_scale, solar_scale, wind_scale)


def forecast_frame(hours=24, start=None, live=True,
                   demand_scale=DEFAULT_DEMAND_SCALE, solar_scale=DEFAULT_SOLAR_SCALE,
                   wind_scale=DEFAULT_WIND_SCALE):
    """Live recompute path: fresh recursive forecast scaled to facility kW."""
    hourly = forecast_hourly(hours=hours, start=start, live=live)
    return to_facility(hourly, demand_scale, solar_scale, wind_scale)


def _dispatch_level(frame, level, capacity_kwh, initial_soc_pct, power_kw):
    soc = capacity_kwh * initial_soc_pct / 100.0
    demand = frame[f"demand_{level}_kw"].to_numpy(float)
    renewable = np.maximum(0.0, frame[f"renewable_{level}_kw"].to_numpy(float))
    n = len(frame)
    used = np.zeros(n)
    charge = np.zeros(n)
    discharge = np.zeros(n)
    backup = np.zeros(n)
    curtailed = np.zeros(n)
    soc_trace = np.zeros(n)
    for i in range(n):
        d, r = demand[i], renewable[i]
        direct = min(r, d)
        surplus = max(0.0, r - direct)
        c = min(surplus, power_kw, (capacity_kwh - soc) / ETA)
        soc += c * ETA
        deficit = max(0.0, d - direct)
        dis = min(deficit, power_kw, soc * ETA)
        soc -= dis / ETA
        used[i], charge[i], discharge[i] = direct, c, dis
        backup[i] = max(0.0, deficit - dis)
        curtailed[i] = max(0.0, surplus - c)
        soc_trace[i] = soc
    # Renewable storage: `charge` is exactly the renewable surplus parked in the
    # battery each hour (auto-filled, never manual). Cumulative tracks the running
    # total stored so hourly/daily/weekly/monthly records can report it directly.
    stored_cum = np.cumsum(charge)
    return pd.DataFrame({
        f"renewable_used_{level}_kw": used, f"battery_charge_{level}_kw": charge,
        f"battery_discharge_{level}_kw": discharge, f"backup_{level}_kw": backup,
        f"curtailed_{level}_kw": curtailed, f"soc_{level}_kwh": soc_trace,
        f"renewable_stored_{level}_kw": charge,
        f"renewable_stored_cum_{level}_kwh": stored_cum})


def dispatch_scenario(frame, capacity_kwh=5000, initial_soc_pct=50, power_kw=1000):
    """Run the dispatch hierarchy for all three bounds and merge into one frame."""
    parts = [_dispatch_level(frame, lv, capacity_kwh, initial_soc_pct, power_kw) for lv in LEVELS]
    return pd.concat([frame.reset_index(drop=True), *parts], axis=1)


def impact(frame, tariff=6.52, emission_factor=0.710, level="expected"):
    """Backup-fuel cost and CO2 avoided by storage dispatch vs no-storage.

    Counterfactual keeps direct renewable use but has no battery, so any deficit
    is met by backup/grid. Hourly kW over 1h == kWh.
    """
    demand = frame[f"demand_{level}_kw"].to_numpy(float)
    renewable = np.maximum(0.0, frame[f"renewable_{level}_kw"].to_numpy(float))
    backup_with = frame[f"backup_{level}_kw"].to_numpy(float)
    backup_without = np.maximum(0.0, demand - renewable)
    saved_kwh = float((backup_without - backup_with).sum())
    return {
        "backup_without_storage_kwh": float(backup_without.sum()),
        "backup_with_storage_kwh": float(backup_with.sum()),
        "energy_saved_kwh": saved_kwh,
        "cost_without_storage_rs": float(backup_without.sum() * tariff),
        "cost_with_storage_rs": float(backup_with.sum() * tariff),
        "cost_savings_rs": float(saved_kwh * tariff),
        "co2_without_storage_kg": float(backup_without.sum() * emission_factor),
        "co2_with_storage_kg": float(backup_with.sum() * emission_factor),
        "co2_avoided_kg": float(saved_kwh * emission_factor),
    }


def scenario(hour_frame, index=0, tariff=6.52, emission_factor=0.710):
    """Dispatch + impact breakdown for a single selected real-time hour."""
    row = hour_frame.iloc[index]
    demand = float(row.demand_expected_kw)
    solar = float(row.solar_expected_kw)
    wind = float(row.wind_expected_kw)
    renewable = float(row.renewable_expected_kw)
    used = float(row.renewable_used_expected_kw)
    charge = float(row.battery_charge_expected_kw)
    discharge = float(row.battery_discharge_expected_kw)
    backup = float(row.backup_expected_kw)
    curtailed = float(row.curtailed_expected_kw)
    backup_without = max(0.0, demand - renewable)
    saved = backup_without - backup
    return {
        "datetime": str(row.datetime), "demand_kw": demand, "solar_kw": solar, "wind_kw": wind,
        "renewable_kw": renewable, "renewable_used_kw": used, "battery_charge_kw": charge,
        "battery_discharge_kw": discharge, "backup_kw": backup, "curtailed_kw": curtailed,
        "soc_kwh": float(row.soc_expected_kwh), "backup_without_storage_kw": backup_without,
        "cost_saved_rs": saved * tariff, "co2_avoided_kg": saved * emission_factor,
    }
