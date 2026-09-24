"""Scenario ("what-if") dispatch engine — the SECOND dispatch simulator.

The main simulator (src.forecasting.operations) keeps running the real-time,
live-weather forecast anchored at now(). This module is the what-if sandbox: the
operator picks a season, a festival day, a day type, and a weather condition
(sunny / cloudy / rainy / windy / clear night) and we build a representative
24-hour day, run it through the SAME trained model pack the real-time engine
uses, then dispatch it through the SAME battery hierarchy.

Everything is derived, never hard-coded demand:
  * calendar features (season one-hot, is_festival, festival_name) come from the
    shared src.features.calendar_features module the models trained on;
  * weather comes from an editable preset, with a diurnal solar-radiation curve
    and a diurnal temperature curve so a sunny afternoon genuinely over-produces
    and the surplus auto-fills the battery;
  * lag / rolling features come from the historical month-by-hour analog so the
    autoregressive inputs are realistic for that season and hour.

The result is a facility-scaled hourly frame with lower/expected/upper bounds,
ready for operations.dispatch_scenario and the animated powerhouse view.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.calendar_features import (SEASONS, festival_lookup,  # noqa: E402
                                            festival_name_for, season_name)
from src.models.realtime_forecast import (LAGS, WEATHER, WINDOWS, Z,  # noqa: E402
                                          _calendar, load_history, load_pack)

# Representative month for each IMD season (used when no festival is selected).
SEASON_MONTH = {"Winter": 1, "Summer": 5, "Monsoon": 7, "Post-monsoon": 10}

# Editable weather presets. `radiation_peak` scales the diurnal solar curve;
# `temp_day`/`temp_night` drive a simple diurnal temperature curve.
WEATHER_PRESETS = {
    "Sunny / clear": {"radiation_peak": 900, "cloud_cover_pct": 5, "relative_humidity_pct": 25,
                      "precipitation_mm": 0.0, "wind_speed_10m_kmh": 8, "temp_day": 40, "temp_night": 27},
    "Partly cloudy": {"radiation_peak": 600, "cloud_cover_pct": 40, "relative_humidity_pct": 45,
                      "precipitation_mm": 0.0, "wind_speed_10m_kmh": 12, "temp_day": 34, "temp_night": 25},
    "Cloudy / overcast": {"radiation_peak": 220, "cloud_cover_pct": 85, "relative_humidity_pct": 65,
                          "precipitation_mm": 0.2, "wind_speed_10m_kmh": 14, "temp_day": 30, "temp_night": 24},
    "Rainy / monsoon": {"radiation_peak": 130, "cloud_cover_pct": 95, "relative_humidity_pct": 88,
                        "precipitation_mm": 9.0, "wind_speed_10m_kmh": 18, "temp_day": 28, "temp_night": 24},
    "Windy": {"radiation_peak": 700, "cloud_cover_pct": 25, "relative_humidity_pct": 40,
              "precipitation_mm": 0.0, "wind_speed_10m_kmh": 38, "temp_day": 33, "temp_night": 24},
    "Clear cool night": {"radiation_peak": 0, "cloud_cover_pct": 5, "relative_humidity_pct": 55,
                         "precipitation_mm": 0.0, "wind_speed_10m_kmh": 10, "temp_day": 22, "temp_night": 15},
}
PRESET_NAMES = list(WEATHER_PRESETS)


def festival_choices():
    """Sorted unique festival names the operator can pick (plus 'No festival')."""
    names = sorted(set(festival_lookup().values()))
    return ["No festival", *names]


def _festival_date(name, prefer_year=2026):
    """A representative calendar date for a festival name (defaults to 2026)."""
    from src.features.calendar_features import LUNAR_FESTIVALS, FIXED_HOLIDAY_MONTH_DAY
    if name in LUNAR_FESTIVALS:
        dates = [pd.Timestamp(d) for d in LUNAR_FESTIVALS[name]]
        for d in dates:
            if d.year == prefer_year:
                return d
        return dates[len(dates) // 2]
    if name in FIXED_HOLIDAY_MONTH_DAY:
        month, day = FIXED_HOLIDAY_MONTH_DAY[name]
        return pd.Timestamp(year=prefer_year, month=month, day=day)
    return None


def scenario_day_date(season="Summer", festival="No festival"):
    """The representative calendar date for a scenario (festival wins if given)."""
    if festival and festival != "No festival":
        d = _festival_date(festival)
        if d is not None:
            return d
    month = SEASON_MONTH.get(season, 5)
    return pd.Timestamp(year=2026, month=month, day=15)


def _weather_for_hour(preset, hour):
    """Diurnal weather for one hour from a preset (solar + temperature curves)."""
    p = WEATHER_PRESETS[preset]
    # Solar radiation: bell curve across ~06:00-18:00, zero at night.
    if 6 <= hour <= 18:
        rad = p["radiation_peak"] * max(0.0, np.sin(np.pi * (hour - 6) / 12.0))
    else:
        rad = 0.0
    # Temperature: coolest ~05:00, warmest ~15:00.
    frac = 0.5 - 0.5 * np.cos(2 * np.pi * (hour - 5) / 24.0)
    temp = p["temp_night"] + (p["temp_day"] - p["temp_night"]) * frac
    kmh = p["wind_speed_10m_kmh"]
    return {
        "temperature_2m_c": float(temp),
        "relative_humidity_pct": float(p["relative_humidity_pct"]),
        "cloud_cover_pct": float(p["cloud_cover_pct"]),
        "precipitation_mm": float(p["precipitation_mm"]),
        "solar_radiation_w_m2": float(rad),
        "wind_speed_10m_kmh": float(kmh),
        "wind_speed_10m_ms": float(kmh / 3.6),
    }


def _analog_stats(hist):
    """Historical month-by-hour mean/std per target stem for lag/rolling seeding."""
    stats = {}
    for stem, col in [("national_demand", "national_demand_mw"),
                      ("solar_generation", "solar_generation_kw"),
                      ("wind_power_potential", "wind_power_potential_kw")]:
        if col not in hist:
            continue
        g = hist.groupby([hist.datetime.dt.month, hist.datetime.dt.hour])[col]
        stats[stem] = {"mean": g.mean(), "std": g.std().fillna(0.0)}
    return stats


def _analog_value(stats, stem, month, hour, kind):
    try:
        val = float(stats[stem][kind].loc[(month, hour)])
    except (KeyError, TypeError):
        val = 0.0
    return val if np.isfinite(val) else 0.0


def build_scenario_frame(season="Summer", festival="No festival", weather="Sunny / clear",
                         weekend=False, days=365, hist=None, pack=None, return_features=False):
    """Predict a representative multi-day scenario horizon (default 365 days = 8760h)
    and return the national-unit hourly bound frame.
    """
    pack = pack or load_pack()
    hist = hist if hist is not None else load_history()
    start_date = scenario_day_date(season, festival)
    if weekend:
        # shift to the nearest Saturday for a genuine weekend calendar encoding
        start_date = start_date + pd.Timedelta(days=(5 - start_date.dayofweek) % 7)
    stats = _analog_stats(hist)
    models = pack["models"]
    feature_union = sorted({f for info in models.values() for f in info["features"]})

    records, feature_rows = [], []
    for d_idx in range(days):
        cur_date = start_date + pd.Timedelta(days=d_idx)
        for hour in range(24):
            ts = pd.Timestamp(year=cur_date.year, month=cur_date.month, day=cur_date.day, hour=hour)
        row = _calendar(ts)
        row.update(_weather_for_hour(weather, hour))
        for name, info in models.items():
            stem = info["target"].replace("_mw", "").replace("_kw", "")
            mean = _analog_value(stats, stem, ts.month, ts.hour, "mean")
            std = _analog_value(stats, stem, ts.month, ts.hour, "std")
            for lag in LAGS:
                row[f"{stem}_lag_{lag}h"] = mean
            for w in WINDOWS:
                row[f"{stem}_rolling_mean_{w}h"] = mean
                row[f"{stem}_rolling_std_{w}h"] = std
            feats = info["features"]
            pred = float(info["model"].predict(np.array([[row[f] for f in feats]], float))[0])
            if name == "solar" and row["solar_radiation_w_m2"] <= 0:
                pred = 0.0
            pred = max(0.0, pred)
            margin = Z * info["residual_std"]
            unit = "mw" if name == "demand" else "kw"
            row[f"{name}_{unit}_expected"] = pred
            row[f"{name}_{unit}_lower"] = max(0.0, pred - margin)
            row[f"{name}_{unit}_upper"] = pred + margin
        value_cols = [c for c in row if c.endswith(("_lower", "_expected", "_upper"))]
        records.append({"datetime": ts, **{c: row[c] for c in value_cols},
                        "season": row.get("season"), "festival_name": row.get("festival_name"),
                        **{c: row[c] for c in WEATHER}})
        if return_features:
            feature_rows.append({"datetime": ts, **{c: row[c] for c in feature_union},
                                 "season": row.get("season"), "festival_name": row.get("festival_name"),
                                 **{c: row[c] for c in WEATHER}})
    frame = pd.DataFrame(records)
    if return_features:
        return frame, pd.DataFrame(feature_rows)
    return frame


def scenario_labels(season, festival, weather, weekend):
    """Human context for the scenario header/narrative."""
    date = scenario_day_date(season, festival)
    return {
        "date": date.strftime("%Y-%m-%d"),
        "season": season_name(date.month),
        "festival": festival_name_for(date) if festival == "No festival" else festival,
        "weather": weather,
        "day_type": "Weekend" if weekend else "Weekday",
    }
