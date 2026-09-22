"""Recursive multi-horizon forecasting for demand, solar, and wind.

The engine anchors at the current real-world hour and rolls forward one step at a
time, feeding each prediction back as the lag/rolling input for the next
(autoregressive). Future weather comes from a pluggable layer: live Open-Meteo
values where available (~16 days), blended with month-by-hour climatological
normals beyond that. Every horizon carries a widening 95% interval.

Note: historical demand ends 2024-04-30, so the autoregressive seed (last 168h)
comes from the latest observed state; there is no live national-demand feed.
"""
import sys
import warnings
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Feature order is fixed and validated against each model's stored feature list,
# so the numpy fast-path predict is safe; silence the names warning it triggers.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.live_weather import TIMEZONE, fetch_live_weather  # noqa: E402

MODEL_PATH = ROOT / "models" / "realtime_forecasters.joblib"
HISTORY = ROOT / "data" / "processed" / "forecast_features.csv"
OUT_DIR = ROOT / "data" / "processed" / "forecasts"

LAGS = [1, 24, 168]
WINDOWS = [24, 168]
WEATHER = ["temperature_2m_c", "relative_humidity_pct", "cloud_cover_pct",
           "precipitation_mm", "solar_radiation_w_m2", "wind_speed_10m_kmh",
           "wind_speed_10m_ms"]
UNITS = {"demand": "mw", "solar": "kw", "wind": "kw"}
VALUE_COLS = [f"{n}_{UNITS[n]}_{b}" for n in UNITS for b in ("lower", "expected", "upper")]
ALPHA = 0.015   # interval widening per forecast hour
Z = 1.96        # 95% confidence
HOURS_12M = 24 * 365


def load_pack():
    return joblib.load(MODEL_PATH)


def load_history():
    return pd.read_csv(HISTORY, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)


def default_start():
    """Current real-world hour (IST, tz-naive) — forecasts begin the next hour."""
    return pd.Timestamp.now(tz=TIMEZONE).tz_localize(None).floor("h")


def _climatology(hist, future_ts):
    normals = hist.groupby([hist.datetime.dt.month, hist.datetime.dt.hour])[WEATHER].mean()
    idx = pd.MultiIndex.from_arrays([future_ts.month, future_ts.hour])
    return pd.DataFrame(normals.loc[idx].to_numpy(), index=future_ts, columns=WEATHER)


def future_weather(future_ts, hist, live=True):
    """Live weather where available, climatological normals elsewhere."""
    wx = _climatology(hist, future_ts)
    if live:
        live_df = fetch_live_weather()
        if len(live_df):
            live_df = live_df.set_index("datetime")[WEATHER]
            common = wx.index.intersection(live_df.index)
            if len(common):
                wx.loc[common] = live_df.loc[common].to_numpy()
    return wx


def _calendar(ts):
    return {
        "is_weekend": float(ts.dayofweek >= 5), "is_night": float(ts.hour < 6 or ts.hour >= 18),
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24), "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
        "day_of_week_sin": np.sin(2 * np.pi * ts.dayofweek / 7),
        "day_of_week_cos": np.cos(2 * np.pi * ts.dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * ts.month / 12), "month_cos": np.cos(2 * np.pi * ts.month / 12),
    }


def forecast_hourly(hours=HOURS_12M, start=None, pack=None, hist=None, live=True,
                    return_features=False):
    """Recursively forecast `hours` steps starting from the next real-world hour.

    If `return_features` is True, also returns the per-step feature matrix used by
    the models so SHAP can explain any selected hour without re-running recursion.
    """
    pack = pack or load_pack()
    hist = hist if hist is not None else load_history()
    start = start or default_start()
    future_ts = pd.date_range(start + pd.Timedelta(hours=1), periods=hours, freq="h")
    wx = future_weather(future_ts, hist, live=live)

    seed_len = max(LAGS + WINDOWS)
    buffers = {name: deque(hist[info["target"]].tail(seed_len).to_numpy(float), maxlen=seed_len)
               for name, info in pack["models"].items()}
    models = pack["models"]
    wx_vals = wx.to_numpy(float)
    feature_union = sorted({f for info in models.values() for f in info["features"]})

    records, feature_rows = [], []
    for i in range(hours):
        ts = future_ts[i]
        row = _calendar(ts)
        for c, v in zip(WEATHER, wx_vals[i]):
            row[c] = v
        margin_factor = np.sqrt(1 + ALPHA * i)

        for name, info in models.items():
            stem = info["target"].replace("_mw", "").replace("_kw", "")
            buf = buffers[name]
            for lag in LAGS:
                row[f"{stem}_lag_{lag}h"] = buf[-lag]
            arr = np.fromiter(buf, float, len(buf))
            for w in WINDOWS:
                window = arr[-w:]
                row[f"{stem}_rolling_mean_{w}h"] = window.mean()
                row[f"{stem}_rolling_std_{w}h"] = window.std(ddof=1)

            feats = info["features"]
            pred = float(info["model"].predict(np.array([[row[f] for f in feats]], float))[0])
            if name == "solar" and row["solar_radiation_w_m2"] <= 0:
                pred = 0.0
            pred = max(0.0, pred)
            buf.append(pred)

            margin = Z * info["residual_std"] * margin_factor
            unit = UNITS[name]
            row[f"{name}_{unit}_expected"] = pred
            row[f"{name}_{unit}_lower"] = max(0.0, pred - margin)
            row[f"{name}_{unit}_upper"] = pred + margin

        records.append({"datetime": ts, **{c: row[c] for c in VALUE_COLS}})
        if return_features:
            feature_rows.append({"datetime": ts, **{c: row[c] for c in feature_union}})

    hourly = pd.DataFrame(records)
    if return_features:
        return hourly, pd.DataFrame(feature_rows)
    return hourly


def aggregate(hourly, freq):
    """Collapse hourly bounds to a period average (mean power over the period)."""
    indexed = hourly.set_index("datetime")
    period = indexed[VALUE_COLS].resample(freq).mean().dropna(how="all")
    return period.reset_index().rename(columns={"datetime": "period_start"})


def build_forecasts(hours=HOURS_12M, start=None, live=True):
    hourly = forecast_hourly(hours=hours, start=start, live=live)
    return {"hourly": hourly, "daily": aggregate(hourly, "D"),
            "weekly": aggregate(hourly, "W-MON"), "monthly": aggregate(hourly, "ME")}


def write_forecasts(hours=HOURS_12M, start=None, live=True, out_dir=OUT_DIR):
    pack = load_pack()
    hist = load_history()
    start = start or default_start()
    hourly, features = forecast_hourly(hours=hours, start=start, live=live,
                                       pack=pack, hist=hist, return_features=True)
    forecasts = {"hourly": hourly, "daily": aggregate(hourly, "D"),
                 "weekly": aggregate(hourly, "W-MON"), "monthly": aggregate(hourly, "ME")}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in forecasts.items():
        path = out_dir / f"{name}_forecast.csv"
        frame.to_csv(path, index=False)
        print(f"{name:>8}: {len(frame):>5} rows -> {path.relative_to(ROOT)}")
    features.to_csv(out_dir / "future_features.csv", index=False)
    print(f"features: {len(features):>5} rows -> forecasts/future_features.csv")
    print(f"anchored at real-world start: {hourly.datetime.iloc[0]} ({TIMEZONE}), live={live}")
    return forecasts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate real-time multi-horizon forecasts.")
    parser.add_argument("--hours", type=int, default=HOURS_12M, help="forecast horizon in hours")
    parser.add_argument("--offline", action="store_true", help="skip live weather, use climatology")
    args = parser.parse_args()
    write_forecasts(hours=args.hours, live=not args.offline)
