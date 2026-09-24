"""Shared configuration, splitting, evaluation and model registry for Phase 5.

Every modelling stage (baseline, advanced ML, comparison, selection) imports from
here so they all evaluate on the *same* chronological split with the *same* metrics
and the *same* feature definitions. That is what makes the cross-stage comparison
fair and the final selection trustworthy.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.calendar_features import NUMERIC_CALENDAR_FEATURES  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS_DIR = ROOT / "models"

FEATURES_CSV = PROCESSED / "forecast_features.csv"
MODEL_PACK = MODELS_DIR / "realtime_forecasters.joblib"

# Three forecasting targets, each trained and evaluated independently.
TARGETS = {
    "demand": "national_demand_mw",
    "solar": "solar_generation_kw",
    "wind": "wind_power_potential_kw",
}

CALENDAR = ["hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
            "month_sin", "month_cos", "is_weekend", "is_night",
            *NUMERIC_CALENDAR_FEATURES]
WEATHER = ["temperature_2m_c", "relative_humidity_pct", "cloud_cover_pct",
           "precipitation_mm", "solar_radiation_w_m2", "wind_speed_10m_kmh",
           "wind_speed_10m_ms"]

# Chronological (past -> future) holdout; simulates real deployment.
TEST_FRACTION = 0.2
RANDOM_STATE = 42

# Seasonal-naive baselines: prediction(t) = actual(t - lag).
NAIVE_LAGS = {"naive_1h": 1, "naive_24h": 24, "naive_168h": 168, "naive_720h": 720}

# Models that can be deployed into the recursive real-time engine (they consume the
# engineered feature vector). Naive baselines are benchmarks only -- they have no
# feature vector, so they are excluded from deployment/selection.
DEPLOYABLE_MODELS = {"linear_regression", "ridge", "random_forest",
                     "gradient_boosting", "xgboost"}


def target_features(columns, target):
    """Calendar + weather + the target's own lag/rolling history (leakage-safe)."""
    stem = target.replace("_mw", "").replace("_kw", "")
    history = [c for c in columns
               if c.startswith(f"{stem}_lag_") or c.startswith(f"{stem}_rolling_")]
    return CALENDAR + WEATHER + history


def load_features():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def chronological_split(df, test_fraction=TEST_FRACTION):
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def evaluate(y_true, y_pred):
    """MAE / RMSE / MAPE plus residual std (used for the 95% forecast intervals)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(0.0, np.asarray(y_pred, dtype=float))
    resid = y_true - y_pred
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mape_pct": float((np.abs(resid) / np.maximum(y_true, 1.0)).mean() * 100),
        "residual_std": float(resid.std()),
    }


def model_registry():
    """Fresh (unfitted) estimator instances keyed by algorithm name.

    A single registry is used both to train candidates and to refit the selected
    winner on the full dataset during the selection stage, so the deployed model is
    always built from exactly the same configuration that was evaluated.
    """
    from xgboost import XGBRegressor
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                               n_jobs=-1, random_state=RANDOM_STATE),
        "gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=RANDOM_STATE),
        "xgboost": XGBRegressor(
            n_estimators=300, learning_rate=0.08, max_depth=6, subsample=0.9,
            colsample_bytree=0.9, reg_lambda=1.0, tree_method="hist",
            n_jobs=-1, random_state=RANDOM_STATE),
        }


def advanced_ml_models():
    """The three advanced ML candidates requested for the ML stage."""
    registry = model_registry()
    return {name: registry[name] for name in ("random_forest", "gradient_boosting", "xgboost")}


def save_figure(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=160)
    import matplotlib.pyplot as plt
    plt.close(fig)
