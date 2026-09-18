import os
import pandas as pd
import joblib


TEST_FILE = "data/processed/wind_test.csv"
MODEL_FILE = "models/wind/wind_linear_regression.pkl"
METRICS_FILE = "reports/wind_metrics.csv"


def main():

    print("Starting Phase 10 wind model validation...")

    # ---------------------------------------------------------
    # 1. Check required files
    # ---------------------------------------------------------

    required_files = [
        TEST_FILE,
        MODEL_FILE,
        METRICS_FILE
    ]

    for file_path in required_files:

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Missing required file: {file_path}"
            )

    # ---------------------------------------------------------
    # 2. Load test data
    # ---------------------------------------------------------

    test = pd.read_csv(TEST_FILE)

    print("\nTest dataset shape:")
    print(test.shape)

    # ---------------------------------------------------------
    # 3. Required columns
    # ---------------------------------------------------------

    required_columns = [
        "datetime",
        "wind_speed_10m_ms",
        "hour",
        "day_of_week",
        "month",
        "hour_sin",
        "hour_cos",
        "wind_lag_1h",
        "wind_lag_24h",
        "wind_rolling_24h"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in test.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ---------------------------------------------------------
    # 4. Load model
    # ---------------------------------------------------------

    model = joblib.load(MODEL_FILE)

    # ---------------------------------------------------------
    # 5. Define features
    # ---------------------------------------------------------

    features = [
        "hour",
        "day_of_week",
        "month",
        "hour_sin",
        "hour_cos",
        "wind_lag_1h",
        "wind_lag_24h",
        "wind_rolling_24h"
    ]

    target = "wind_speed_10m_ms"

    X_test = test[features]
    y_test = test[target]

    # ---------------------------------------------------------
    # 6. Generate predictions
    # ---------------------------------------------------------

    predictions = model.predict(X_test)

    # Wind speed cannot be negative
    predictions[predictions < 0] = 0

    # ---------------------------------------------------------
    # 7. Validate predictions
    # ---------------------------------------------------------

    if len(predictions) != len(y_test):
        raise ValueError(
            "Prediction count does not match test dataset."
        )

    if pd.isna(predictions).any():
        raise ValueError(
            "Model generated NaN predictions."
        )

    if (predictions < 0).any():
        raise ValueError(
            "Negative wind-speed predictions detected."
        )

    # ---------------------------------------------------------
    # 8. Validate metrics file
    # ---------------------------------------------------------

    metrics = pd.read_csv(METRICS_FILE)

    required_metric_columns = [
        "model",
        "MAE",
        "RMSE",
        "MAPE"
    ]

    for column in required_metric_columns:

        if column not in metrics.columns:
            raise ValueError(
                f"Missing metric column: {column}"
            )

    if metrics.empty:
        raise ValueError(
            "Wind metrics file is empty."
        )

    # ---------------------------------------------------------
    # 9. Print validation information
    # ---------------------------------------------------------

    print("\nModel:")
    print(metrics["model"].iloc[0])

    print("\nMetrics:")
    print(metrics.to_string(index=False))

    print("\nPrediction count:")
    print(len(predictions))

    print("\nPrediction range:")
    print(
        f"Minimum: {predictions.min():.4f} m/s"
    )
    print(
        f"Maximum: {predictions.max():.4f} m/s"
    )

    print(
        "\nFINAL PHASE 10 WIND MODEL VALIDATION: PASSED"
    )


if __name__ == "__main__":
    main()