import os
import numpy as np
import pandas as pd
import joblib


SOLAR_TEST_FILE = "data/processed/solar_test.csv"
SOLAR_MODEL_FILE = "models/solar/solar_linear_regression.pkl"
SOLAR_OUTPUT_FILE = "data/processed/solar_uncertainty.csv"

WIND_TEST_FILE = "data/processed/wind_test.csv"
WIND_MODEL_FILE = "models/wind/wind_linear_regression.pkl"
WIND_OUTPUT_FILE = "data/processed/wind_uncertainty.csv"

CONFIDENCE_LEVEL = 0.95


def calculate_intervals(
    df,
    model,
    features,
    target
):
    X = df[features]
    y_actual = df[target]

    predictions = model.predict(X)

    # Prevent physically impossible negative predictions
    predictions = np.maximum(predictions, 0)

    # Test-set residuals
    residuals = y_actual.values - predictions

    # Residual standard deviation
    residual_std = np.std(
        residuals,
        ddof=1
    )

    # Approximate 95% prediction interval
    z_value = 1.96

    margin = z_value * residual_std

    lower_bound = np.maximum(
        predictions - margin,
        0
    )

    upper_bound = predictions + margin

    result = pd.DataFrame({
        "datetime": df["datetime"],
        "actual": y_actual,
        "prediction": predictions,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "confidence_level": CONFIDENCE_LEVEL
    })

    return result, residual_std


def process_solar():

    if not os.path.exists(SOLAR_TEST_FILE):
        raise FileNotFoundError(
            f"Missing file: {SOLAR_TEST_FILE}"
        )

    if not os.path.exists(SOLAR_MODEL_FILE):
        raise FileNotFoundError(
            f"Missing file: {SOLAR_MODEL_FILE}"
        )

    df = pd.read_csv(SOLAR_TEST_FILE)

    model = joblib.load(SOLAR_MODEL_FILE)

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

    result, residual_std = calculate_intervals(
        df,
        model,
        features,
        target
    )

    os.makedirs(
        os.path.dirname(SOLAR_OUTPUT_FILE),
        exist_ok=True
    )

    result.to_csv(
        SOLAR_OUTPUT_FILE,
        index=False
    )

    return result, residual_std


def process_wind():

    if not os.path.exists(WIND_TEST_FILE):
        raise FileNotFoundError(
            f"Missing file: {WIND_TEST_FILE}"
        )

    if not os.path.exists(WIND_MODEL_FILE):
        raise FileNotFoundError(
            f"Missing file: {WIND_MODEL_FILE}"
        )

    df = pd.read_csv(WIND_TEST_FILE)

    model = joblib.load(WIND_MODEL_FILE)

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

    result, residual_std = calculate_intervals(
        df,
        model,
        features,
        target
    )

    os.makedirs(
        os.path.dirname(WIND_OUTPUT_FILE),
        exist_ok=True
    )

    result.to_csv(
        WIND_OUTPUT_FILE,
        index=False
    )

    return result, residual_std


def main():

    print(
        "Starting Phase 11 uncertainty analysis..."
    )

    solar_result, solar_std = process_solar()

    wind_result, wind_std = process_wind()

    print("\nSolar uncertainty:")
    print(
        f"Residual standard deviation: "
        f"{solar_std:.4f}"
    )

    print("\nWind uncertainty:")
    print(
        f"Residual standard deviation: "
        f"{wind_std:.4f}"
    )

    print("\nGenerated files:")
    print(SOLAR_OUTPUT_FILE)
    print(WIND_OUTPUT_FILE)

    print(
        "\nPHASE 11 UNCERTAINTY CALCULATION: COMPLETED"
    )


if __name__ == "__main__":
    main()