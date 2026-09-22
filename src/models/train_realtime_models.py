"""Train demand, solar, and wind forecasters with multi-model auto-selection.

Each target is fit with several scikit-learn regressors on a chronological
train split; the algorithm with the lowest validation RMSE is selected and
persisted for the recursive multi-horizon forecasting engine.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "forecast_features.csv"
MODEL_PATH = ROOT / "models" / "realtime_forecasters.joblib"
REPORT = ROOT / "reports" / "realtime_model_metrics.csv"

CALENDAR = ["hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
            "month_sin", "month_cos", "is_weekend", "is_night"]
WEATHER = ["temperature_2m_c", "relative_humidity_pct", "cloud_cover_pct",
           "precipitation_mm", "solar_radiation_w_m2", "wind_speed_10m_kmh",
           "wind_speed_10m_ms"]
TARGETS = {"demand": "national_demand_mw",
           "solar": "solar_generation_kw",
           "wind": "wind_power_potential_kw"}
TEST_FRACTION = 0.2
RANDOM_STATE = 42


def candidates():
    return {
        "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                               n_jobs=-1, random_state=RANDOM_STATE),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=RANDOM_STATE),
    }


def target_features(columns, target):
    stem = target.replace("_mw", "").replace("_kw", "")
    history = [c for c in columns
               if c.startswith(f"{stem}_lag_") or c.startswith(f"{stem}_rolling_")]
    return CALENDAR + WEATHER + history


def train():
    df = pd.read_csv(DATA, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    cut = int(len(df) * (1 - TEST_FRACTION))
    train_df, test_df = df.iloc[:cut], df.iloc[cut:]

    pack_models, metric_rows = {}, []
    for name, target in TARGETS.items():
        features = target_features(df.columns, target)
        X_tr, y_tr = train_df[features], train_df[target]
        X_te, y_te = test_df[features], test_df[target].to_numpy()

        best = None
        for algo, model in candidates().items():
            model.fit(X_tr, y_tr)
            pred = np.maximum(0.0, model.predict(X_te))
            resid = y_te - pred
            rmse = float(mean_squared_error(y_te, pred) ** 0.5)
            metric_rows.append({
                "target": name, "algorithm": algo,
                "mae": float(mean_absolute_error(y_te, pred)), "rmse": rmse,
                "mape_pct": float((np.abs(resid) / np.maximum(y_te, 1.0)).mean() * 100),
                "test_rows": len(y_te), "selected": 0})
            if best is None or rmse < best["rmse"]:
                best = {"algo": algo, "model": model, "rmse": rmse,
                        "residual_std": float(resid.std())}

        for row in metric_rows:
            if row["target"] == name and row["algorithm"] == best["algo"]:
                row["selected"] = 1
        pack_models[name] = {"model": best["model"], "target": target, "features": features,
                             "algorithm": best["algo"], "residual_std": best["residual_std"]}
        print(f"{name:>6}: selected {best['algo']} (RMSE {best['rmse']:.2f})")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": pack_models, "feature_columns": list(df.columns),
                 "trained_through": str(train_df.datetime.iloc[-1]),
                 "data_end": str(df.datetime.iloc[-1])}, MODEL_PATH)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(REPORT, index=False)
    return metrics


if __name__ == "__main__":
    train()
