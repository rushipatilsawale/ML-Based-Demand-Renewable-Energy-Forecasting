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
from src.features.calendar_features import LABEL_COLUMNS, calendar_features_for_timestamp  # noqa: E402

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
    """Current real-world hour minus 6h (IST, tz-naive) — provides -6h past to +18h future window."""
    return pd.Timestamp.now(tz=TIMEZONE).tz_localize(None).floor("h") - pd.Timedelta(hours=6)


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
    row = {
        "is_weekend": float(ts.dayofweek >= 5), "is_night": float(ts.hour < 6 or ts.hour >= 18),
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24), "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
        "day_of_week_sin": np.sin(2 * np.pi * ts.dayofweek / 7),
        "day_of_week_cos": np.cos(2 * np.pi * ts.dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * ts.month / 12), "month_cos": np.cos(2 * np.pi * ts.month / 12),
    }
    # Season one-hot, is_festival, and the human labels (season, festival_name) so
    # live rows match the training encoding and the dashboard can name the festival.
    row.update(calendar_features_for_timestamp(ts))
    return row


def forecast_hourly(hours=HOURS_12M, start=None, pack=None, hist=None, live=True,
                    return_features=False):
    """Recursively forecast `hours` steps starting from the next real-world hour.

    If `return_features` is True, also returns the per-step feature matrix used by
    the models so SHAP can explain any selected hour without re-running recursion.
    """
    warnings.filterwarnings("ignore")
    pack = pack or load_pack()
    hist = hist if hist is not None else load_history()
    start = start or default_start()
    future_ts = pd.date_range(start + pd.Timedelta(hours=1), periods=hours, freq="h")
    wx = future_weather(future_ts, hist, live=live)

    seed_len = max(LAGS + WINDOWS)
    models = pack["models"]
    feature_union = sorted({f for info in models.values() for f in info["features"]})

    # Pre-build calendar features and weather matrix for all hours vectorized
    cal_rows = [_calendar(ts) for ts in future_ts]
    cal_df = pd.DataFrame(cal_rows, index=future_ts)
    for c in WEATHER:
        cal_df[c] = wx[c].to_numpy(float)

    # Setup fast-path history buffers and running stats tracking
    buffers = {}
    running_stats = {}
    for name, info in models.items():
        stem = info["target"].replace("_mw", "").replace("_kw", "")
        buf = np.zeros(seed_len + hours, dtype=float)
        seed_vals = hist[info["target"]].tail(seed_len).to_numpy(float)
        buf[:seed_len] = seed_vals
        buffers[name] = buf

        # Pre-compute initial sums and sum-of-squares for each rolling window
        rstat = {}
        for w in WINDOWS:
            win = buf[seed_len - w : seed_len]
            rstat[w] = {"sum": float(win.sum()), "sq": float((win**2).sum())}
        running_stats[name] = rstat

    # Prepare model info fast data structures
    fast_models = {}
    for name, info in models.items():
        model = info["model"]
        feats = info["features"]
        stem = info["target"].replace("_mw", "").replace("_kw", "")
        is_linear = hasattr(model, "coef_") and hasattr(model, "intercept_")
        coef = np.asarray(model.coef_, dtype=float) if is_linear else None
        intercept = float(model.intercept_) if is_linear else 0.0
        fast_models[name] = {
            "model": model, "feats": feats, "stem": stem, "is_linear": is_linear,
            "coef": coef, "intercept": intercept,
            "residual_std": float(info["residual_std"]),
            "unit": UNITS[name],
            "x_buf": np.zeros((1, len(feats)), dtype=float),
        }

    records, feature_rows = [], []
    cal_dicts = cal_df.to_dict("records")
    alpha_sqrt = np.sqrt(1 + ALPHA * np.arange(hours))

    for i in range(hours):
        ts = future_ts[i]
        row = cal_dicts[i]
        pos = seed_len + i
        margin_factor = alpha_sqrt[i]

        for name, mdata in fast_models.items():
            stem = mdata["stem"]
            buf = buffers[name]
            rstat = running_stats[name]

            # Fast lag retrieval
            for lag in LAGS:
                row[f"{stem}_lag_{lag}h"] = buf[pos - lag]

            # Fast O(1) rolling mean and std
            for w in WINDOWS:
                if i > 0:
                    add_val = buf[pos - 1]
                    rem_val = buf[pos - w - 1]
                    st = rstat[w]
                    st["sum"] += add_val - rem_val
                    st["sq"] += add_val * add_val - rem_val * rem_val
                st = rstat[w]
                mean_val = st["sum"] / w
                var_val = max(0.0, (st["sq"] - (st["sum"] ** 2) / w) / (w - 1))
                row[f"{stem}_rolling_mean_{w}h"] = mean_val
                row[f"{stem}_rolling_std_{w}h"] = float(np.sqrt(var_val))

            feats = mdata["feats"]
            if mdata["is_linear"]:
                x_vec = np.fromiter((row[f] for f in feats), float, len(feats))
                pred = float(np.dot(x_vec, mdata["coef"]) + mdata["intercept"])
            else:
                x_buf = mdata["x_buf"]
                for k, f in enumerate(feats):
                    x_buf[0, k] = float(row[f])
                pred = float(mdata["model"].predict(x_buf)[0])

            if name == "solar" and row.get("solar_radiation_w_m2", 0.0) <= 0:
                pred = 0.0
            pred = max(0.0, pred)
            buf[pos] = pred

            margin = Z * mdata["residual_std"] * margin_factor
            unit = mdata["unit"]
            row[f"{name}_{unit}_expected"] = pred
            row[f"{name}_{unit}_lower"] = max(0.0, pred - margin)
            row[f"{name}_{unit}_upper"] = pred + margin

        records.append({"datetime": ts, **{c: row[c] for c in VALUE_COLS}})
        if return_features:
            labels = {c: row[c] for c in LABEL_COLUMNS if c in row}
            feature_rows.append({"datetime": ts, **{c: row[c] for c in feature_union}, **labels})

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
    print(f"Generating recursive forecast for horizon = {hours} hours...")
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
