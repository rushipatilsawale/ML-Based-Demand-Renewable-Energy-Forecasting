import os
import numpy as np
import pandas as pd

INPUT_FILE = "data/processed/wind_test.csv"
OUTPUT_FILE = "data/processed/wind_power_estimates.csv"

# Representative normalized turbine power-curve assumptions
CUT_IN_SPEED = 3.0       # m/s
RATED_SPEED = 12.0       # m/s
CUT_OUT_SPEED = 25.0     # m/s

REFERENCE_CAPACITY_KW = 1000.0


def calculate_power_fraction(wind_speed):
    """
    Convert wind speed into normalized turbine power
    using a representative power-curve shape.

    Output range: 0.0 to 1.0
    """

    if wind_speed < CUT_IN_SPEED:
        return 0.0

    if wind_speed >= CUT_OUT_SPEED:
        return 0.0

    if wind_speed >= RATED_SPEED:
        return 1.0

    # Simplified cubic relationship between cut-in and rated speed
    fraction = (
        (wind_speed**3 - CUT_IN_SPEED**3)
        / (RATED_SPEED**3 - CUT_IN_SPEED**3)
    )

    return float(np.clip(fraction, 0.0, 1.0))


def main():

    print("Starting wind power estimation...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "datetime",
        "wind_speed_10m_ms"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    df["datetime"] = pd.to_datetime(df["datetime"])

    # Actual observed wind speed
    df["actual_power_fraction"] = (
        df["wind_speed_10m_ms"]
        .apply(calculate_power_fraction)
    )

    # Forecast wind speed
    if "predicted_wind_speed_ms" in df.columns:
        df["forecast_power_fraction"] = (
            df["predicted_wind_speed_ms"]
            .apply(calculate_power_fraction)
        )

    # Convert normalized power to reference kW
    df["estimated_wind_power_kw"] = (
        df["actual_power_fraction"]
        * REFERENCE_CAPACITY_KW
    )

    if "forecast_power_fraction" in df.columns:
        df["forecast_wind_power_kw"] = (
            df["forecast_power_fraction"]
            * REFERENCE_CAPACITY_KW
        )

    output_columns = [
        "datetime",
        "wind_speed_10m_ms",
        "actual_power_fraction",
        "estimated_wind_power_kw"
    ]

    if "predicted_wind_speed_ms" in df.columns:
        output_columns.insert(
            2,
            "predicted_wind_speed_ms"
        )
        output_columns.extend([
            "forecast_power_fraction",
            "forecast_wind_power_kw"
        ])

    output = df[output_columns].copy()

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nWind power estimation completed.")

    print("\nOutput shape:")
    print(output.shape)

    print("\nPower statistics:")
    print(
        output[
            [
                "actual_power_fraction",
                "estimated_wind_power_kw"
            ]
        ].describe()
    )

    print("\nOutput file:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()