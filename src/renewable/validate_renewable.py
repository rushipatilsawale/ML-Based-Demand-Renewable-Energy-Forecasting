import os
import pandas as pd
import joblib


SOLAR_FILE = "data/processed/solar_hourly.csv"
SOLAR_TRAIN_FILE = "data/processed/solar_train.csv"
SOLAR_TEST_FILE = "data/processed/solar_test.csv"
SOLAR_MODEL_FILE = "models/solar/solar_linear_regression.pkl"
SOLAR_METRICS_FILE = "reports/solar_metrics.csv"

WIND_FILE = "data/raw/renewable/nasa_power_wind_hourly.csv"
WIND_TRAIN_FILE = "data/processed/wind_train.csv"
WIND_TEST_FILE = "data/processed/wind_test.csv"
WIND_MODEL_FILE = "models/wind/wind_linear_regression.pkl"
WIND_METRICS_FILE = "reports/wind_metrics.csv"

WIND_POWER_FILE = "data/processed/wind_power_estimates.csv"
RENEWABLE_EVALUATION_FILE = "reports/renewable_evaluation.csv"
RENEWABLE_FIGURE_FILE = "reports/figures/renewable_forecasting.png"


def check_files(files):
    for file_path in files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Missing required file: {file_path}"
            )


def check_metrics(file_path, source_name):
    metrics = pd.read_csv(file_path)

    required_columns = [
        "model",
        "MAE",
        "RMSE",
        "MAPE"
    ]

    for column in required_columns:
        if column not in metrics.columns:
            raise ValueError(
                f"Missing {source_name} metric column: {column}"
            )

    if metrics.empty:
        raise ValueError(
            f"{source_name} metrics file is empty."
        )

    return metrics


def main():

    print("Starting FINAL Phase 10 renewable validation...")

    required_files = [
        SOLAR_FILE,
        SOLAR_TRAIN_FILE,
        SOLAR_TEST_FILE,
        SOLAR_MODEL_FILE,
        SOLAR_METRICS_FILE,
        WIND_FILE,
        WIND_TRAIN_FILE,
        WIND_TEST_FILE,
        WIND_MODEL_FILE,
        WIND_METRICS_FILE,
        WIND_POWER_FILE,
        RENEWABLE_EVALUATION_FILE,
        RENEWABLE_FIGURE_FILE
    ]

    check_files(required_files)

    # --------------------------------------------------
    # 1. Solar validation
    # --------------------------------------------------

    solar = pd.read_csv(SOLAR_FILE)

    required_solar_columns = [
        "datetime",
        "plant_id",
        "solar_generation_ac"
    ]

    for column in required_solar_columns:
        if column not in solar.columns:
            raise ValueError(
                f"Missing solar column: {column}"
            )

    if solar["solar_generation_ac"].isna().any():
        raise ValueError(
            "Missing solar generation values detected."
        )

    solar_metrics = check_metrics(
        SOLAR_METRICS_FILE,
        "solar"
    )

    joblib.load(SOLAR_MODEL_FILE)

    print("Solar validation: PASSED")

    # --------------------------------------------------
    # 2. Wind validation
    # --------------------------------------------------

    wind = pd.read_csv(WIND_FILE)

    required_wind_columns = [
        "datetime",
        "wind_speed_10m_ms",
        "latitude",
        "longitude"
    ]

    for column in required_wind_columns:
        if column not in wind.columns:
            raise ValueError(
                f"Missing wind column: {column}"
            )

    if wind["wind_speed_10m_ms"].isna().any():
        raise ValueError(
            "Missing wind-speed values detected."
        )

    if (wind["wind_speed_10m_ms"] < 0).any():
        raise ValueError(
            "Negative wind-speed values detected."
        )

    wind_metrics = check_metrics(
        WIND_METRICS_FILE,
        "wind"
    )

    joblib.load(WIND_MODEL_FILE)

    print("Wind validation: PASSED")

    # --------------------------------------------------
    # 3. Wind power validation
    # --------------------------------------------------

    wind_power = pd.read_csv(WIND_POWER_FILE)

    required_power_columns = [
        "datetime",
        "wind_speed_10m_ms",
        "actual_power_fraction",
        "estimated_wind_power_kw"
    ]

    for column in required_power_columns:
        if column not in wind_power.columns:
            raise ValueError(
                f"Missing wind-power column: {column}"
            )

    if wind_power.isna().any().any():
        raise ValueError(
            "Missing values detected in wind-power estimates."
        )

    if (
        (wind_power["actual_power_fraction"] < 0).any()
        or
        (wind_power["actual_power_fraction"] > 1).any()
    ):
        raise ValueError(
            "Invalid wind power fractions detected."
        )

    if (wind_power["estimated_wind_power_kw"] < 0).any():
        raise ValueError(
            "Negative estimated wind power detected."
        )

    print("Wind power validation: PASSED")

    # --------------------------------------------------
    # 4. Renewable evaluation validation
    # --------------------------------------------------

    evaluation = pd.read_csv(
        RENEWABLE_EVALUATION_FILE
    )

    required_evaluation_columns = [
        "energy_source",
        "model",
        "MAE",
        "RMSE",
        "MAPE"
    ]

    for column in required_evaluation_columns:
        if column not in evaluation.columns:
            raise ValueError(
                f"Missing evaluation column: {column}"
            )

    if evaluation.empty:
        raise ValueError(
            "Renewable evaluation file is empty."
        )

    expected_sources = {
        "Solar",
        "Wind Speed"
    }

    actual_sources = set(
        evaluation["energy_source"]
    )

    if not expected_sources.issubset(actual_sources):
        raise ValueError(
            "Solar and Wind Speed evaluations are not both present."
        )

    print("Renewable evaluation validation: PASSED")

    # --------------------------------------------------
    # 5. Final summary
    # --------------------------------------------------

    print("\nSolar model:")
    print(solar_metrics.to_string(index=False))

    print("\nWind model:")
    print(wind_metrics.to_string(index=False))

    print("\nWind power summary:")
    print(
        f"Mean estimated power: "
        f"{wind_power['estimated_wind_power_kw'].mean():.2f} kW"
    )

    print(
        f"Maximum estimated power: "
        f"{wind_power['estimated_wind_power_kw'].max():.2f} kW"
    )

    print("\nRenewable evaluation:")
    print(evaluation.to_string(index=False))

    print("\nFinal outputs verified:")
    print("- Solar dataset")
    print("- Solar forecasting model")
    print("- Solar metrics")
    print("- Wind dataset")
    print("- Wind forecasting model")
    print("- Wind metrics")
    print("- Wind power estimation")
    print("- Renewable evaluation")
    print("- Renewable forecasting figure")

    print(
        "\nFINAL PHASE 10 VALIDATION: PASSED"
    )


if __name__ == "__main__":
    main()