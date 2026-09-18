import os
import pandas as pd

INPUT_FILE = "data/processed/wind_power_estimates.csv"


def main():

    print("Starting wind power estimation validation...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Missing required file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "datetime",
        "wind_speed_10m_ms",
        "actual_power_fraction",
        "estimated_wind_power_kw"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Check datetime
    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    if df["datetime"].isna().any():
        raise ValueError(
            "Invalid datetime values detected."
        )

    # Check missing values
    if df[required_columns].isna().any().any():
        raise ValueError(
            "Missing values detected in required columns."
        )

    # Check wind speed
    if (df["wind_speed_10m_ms"] < 0).any():
        raise ValueError(
            "Negative wind-speed values detected."
        )

    # Power fraction must be between 0 and 1
    if (
        (df["actual_power_fraction"] < 0).any()
        or
        (df["actual_power_fraction"] > 1).any()
    ):
        raise ValueError(
            "Power fraction outside 0-1 range."
        )

    # Estimated power must be non-negative
    if (df["estimated_wind_power_kw"] < 0).any():
        raise ValueError(
            "Negative estimated wind-power values detected."
        )

    # Verify power calculation
    expected_power = (
        df["actual_power_fraction"] * 1000.0
    )

    if not (
        (df["estimated_wind_power_kw"] - expected_power)
        .abs()
        < 1e-6
    ).all():
        raise ValueError(
            "Estimated wind-power calculation is incorrect."
        )

    print("\nDataset shape:")
    print(df.shape)

    print("\nWind speed range:")
    print(
        f"Minimum: {df['wind_speed_10m_ms'].min():.4f} m/s"
    )
    print(
        f"Maximum: {df['wind_speed_10m_ms'].max():.4f} m/s"
    )

    print("\nEstimated power range:")
    print(
        f"Minimum: {df['estimated_wind_power_kw'].min():.2f} kW"
    )
    print(
        f"Maximum: {df['estimated_wind_power_kw'].max():.2f} kW"
    )

    print("\nRequired columns verified.")
    print("No missing values.")
    print("No negative wind speeds.")
    print("Power fraction range verified.")
    print("Power calculation verified.")

    print(
        "\nFINAL WIND POWER ESTIMATION VALIDATION: PASSED"
    )


if __name__ == "__main__":
    main()