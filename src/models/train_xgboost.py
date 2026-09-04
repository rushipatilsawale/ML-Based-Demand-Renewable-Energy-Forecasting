import os

import joblib
import pandas as pd
from xgboost import XGBRegressor


TRAIN_FILE = "data/processed/ml_train.csv"
TEST_FILE = "data/processed/ml_test.csv"
MODEL_FILE = "models/xgboost.pkl"

TARGET = "national_demand_mw"

FEATURES = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_lag_168h",
    "demand_rolling_mean_24h",
    "demand_rolling_std_24h",
    "demand_rolling_mean_168h",
    "demand_rolling_std_168h",
    "temperature_2m_c",
    "relative_humidity_pct",
    "cloud_cover_pct",
    "precipitation_mm",
    "wind_speed_10m_kmh",
    "solar_radiation_w_m2",
]


def train_xgboost():
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    print("XGBoost training completed.")
    print("=" * 50)
    print(f"Training rows : {len(X_train)}")
    print(f"Testing rows  : {len(X_test)}")
    print(f"Features      : {len(FEATURES)}")
    print(f"Model saved   : {MODEL_FILE}")
    print(f"Predictions   : {len(predictions)}")


if __name__ == "__main__":
    train_xgboost()