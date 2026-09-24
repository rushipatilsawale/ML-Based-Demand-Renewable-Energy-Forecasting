"""Create validated leakage-safe features for demand and renewable forecasting."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.calendar_features import (  # noqa: E402
    LABEL_COLUMNS,
    NUMERIC_CALENDAR_FEATURES,
    add_calendar_features,
)

INPUT = ROOT / "data" / "processed" / "aligned_hourly_dataset.csv"
OUTPUT = ROOT / "data" / "processed" / "forecast_features.csv"
MANIFEST = ROOT / "data" / "processed" / "feature_manifest.json"

TARGETS = ["national_demand_mw", "solar_generation_kw", "wind_power_potential_kw"]
LAGS = [1, 24, 168]
WINDOWS = [24, 168]


def build_features():
    df = pd.read_csv(INPUT, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    if df.datetime.duplicated().any() or not df.datetime.is_monotonic_increasing:
        raise ValueError("Aligned input must contain unique chronological timestamps.")
    # Cyclical calendar features preserve proximity across periodic boundaries.
    for column, period in [("hour", 24), ("day_of_week", 7), ("month", 12)]:
        df[f"{column}_sin"] = np.sin(2 * np.pi * df[column] / period)
        df[f"{column}_cos"] = np.cos(2 * np.pi * df[column] / period)
    df["is_night"] = ((df.hour < 6) | (df.hour >= 18)).astype("int8")
    # Season (IMD 4-season) and Indian-festival flags come from the shared calendar
    # module so the real-time engine encodes future rows identically. Numeric
    # one-hot/binary columns become model features; season/festival_name stay as
    # human-readable labels for the dashboard and SHAP narrative.
    add_calendar_features(df)
    # Every historical target is shifted before lag/rolling calculation.
    for target in TARGETS:
        shifted = df[target].shift(1)
        stem = target.replace("_mw", "").replace("_kw", "")
        for lag in LAGS:
            df[f"{stem}_lag_{lag}h"] = df[target].shift(lag)
        for window in WINDOWS:
            df[f"{stem}_rolling_mean_{window}h"] = shifted.rolling(window, min_periods=window).mean()
            df[f"{stem}_rolling_std_{window}h"] = shifted.rolling(window, min_periods=window).std()
    # Future weather must be supplied by the operational weather-feed layer;
    # these historical columns train the relationship without target leakage.
    weather_features = ["temperature_2m_c", "relative_humidity_pct", "cloud_cover_pct", "precipitation_mm", "solar_radiation_w_m2", "wind_speed_10m_kmh", "wind_speed_10m_ms"]
    required = ["datetime", *TARGETS, *weather_features]
    if not set(required).issubset(df.columns):
        raise ValueError("Required demand, renewable, or weather columns are missing.")
    features = df.dropna().reset_index(drop=True)
    if features.isna().any().any() or features.datetime.duplicated().any():
        raise ValueError("Feature output has missing values or duplicate timestamps.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT, index=False)
    manifest = {"input": str(INPUT.name), "output": str(OUTPUT.name), "rows": len(features),
                "start": str(features.datetime.min()), "end": str(features.datetime.max()),
                "target_columns": TARGETS, "weather_features": weather_features,
                "calendar_features": NUMERIC_CALENDAR_FEATURES, "label_columns": LABEL_COLUMNS,
                "lags_hours": LAGS, "rolling_windows_hours": WINDOWS,
                "leakage_control": "All rolling values use target.shift(1); lag values use prior timestamps only."}
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Feature engineering complete: {len(features):,} rows, {len(features.columns)} columns.")
    return features


if __name__ == "__main__":
    build_features()
