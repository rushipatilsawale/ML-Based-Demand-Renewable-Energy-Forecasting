import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


INPUT_FILE = "data/processed/solar_hourly.csv"

TRAIN_FILE = "data/processed/solar_train.csv"
TEST_FILE = "data/processed/solar_test.csv"

MODEL_DIR = "models/solar"
MODEL_FILE = "models/solar/solar_linear_regression.pkl"

METRICS_FILE = "reports/solar_metrics.csv"


def load_data():
    """Load processed solar hourly data."""

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["datetime"] = pd.to_datetime(df["datetime"])

    df = df.sort_values(
        by=["datetime", "plant_id"]
    ).reset_index(drop=True)

    return df


def create_features(df):
    """Create basic time-series features for solar forecasting."""

    df = df.copy()

    # Time features
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    # Cyclical hour features
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # Lag features
    df["solar_lag_1h"] = (
        df.groupby("plant_id")["solar_generation_ac"]
        .shift(1)
    )

    df["solar_lag_24h"] = (
        df.groupby("plant_id")["solar_generation_ac"]
        .shift(24)
    )

    # Rolling mean
    df["solar_rolling_24h"] = (
        df.groupby("plant_id")["solar_generation_ac"]
        .transform(
            lambda x: x.shift(1).rolling(24).mean()
        )
    )

    return df


def split_data(df):
    """Perform chronological train-test split."""

    df = df.sort_values(
        by=["datetime", "plant_id"]
    ).reset_index(drop=True)

    split_date = df["datetime"].quantile(0.80)

    train = df[
        df["datetime"] <= split_date
    ].copy()

    test = df[
        df["datetime"] > split_date
    ].copy()

    return train, test


def main():

    print("Starting Phase 10 solar forecasting...")

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    df = load_data()

    print("\nLoaded dataset:")
    print(df.shape)

    # ---------------------------------------------------------
    # 2. Create features
    # ---------------------------------------------------------

    df = create_features(df)

    # Remove rows created by lag/rolling operations
    df = df.dropna().reset_index(drop=True)

    print("\nDataset after feature creation:")
    print(df.shape)

    # ---------------------------------------------------------
    # 3. Train-test split
    # ---------------------------------------------------------

    train, test = split_data(df)

    print("\nTrain shape:")
    print(train.shape)

    print("\nTest shape:")
    print(test.shape)

    # ---------------------------------------------------------
    # 4. Define features and target
    # ---------------------------------------------------------

    features = [
        "hour",
        "day_of_week",
        "month",
        "hour_sin",
        "hour_cos",
        "solar_lag_1h",
        "solar_lag_24h",
        "solar_rolling_24h"
    ]

    target = "solar_generation_ac"

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    # ---------------------------------------------------------
    # 5. Train Linear Regression model
    # ---------------------------------------------------------

    print("\nTraining Linear Regression model...")

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    # ---------------------------------------------------------
    # 6. Predictions
    # ---------------------------------------------------------

    predictions = model.predict(X_test)

    # Solar generation cannot be negative
    predictions = np.maximum(
        predictions,
        0
    )

    # ---------------------------------------------------------
    # 7. Evaluation
    # ---------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    non_zero_actual = y_test != 0

    if non_zero_actual.sum() > 0:

        mape = (
            np.mean(
                np.abs(
                    (
                        y_test[non_zero_actual]
                        - predictions[non_zero_actual]
                    )
                    / y_test[non_zero_actual]
                )
            )
            * 100
        )

    else:
        mape = np.nan

    print("\nSolar Model Performance:")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.4f}%")

    # ---------------------------------------------------------
    # 8. Save train/test datasets
    # ---------------------------------------------------------

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    train.to_csv(
        TRAIN_FILE,
        index=False
    )

    test.to_csv(
        TEST_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # 9. Save model
    # ---------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    import joblib

    joblib.dump(
        model,
        MODEL_FILE
    )

    # ---------------------------------------------------------
    # 10. Save metrics
    # ---------------------------------------------------------

    os.makedirs(
        "reports",
        exist_ok=True
    )

    metrics = pd.DataFrame(
        [
            {
                "model": "Linear Regression",
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape
            }
        ]
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # 11. Final information
    # ---------------------------------------------------------

    print("\nGenerated files:")
    print(TRAIN_FILE)
    print(TEST_FILE)
    print(MODEL_FILE)
    print(METRICS_FILE)

    print("\nPHASE 10 SOLAR FORECASTING: COMPLETED")


if __name__ == "__main__":
    main()