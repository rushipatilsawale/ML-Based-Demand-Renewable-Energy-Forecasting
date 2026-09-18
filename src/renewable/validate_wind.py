import os
import pandas as pd


INPUT_FILE = "data/raw/renewable/nasa_power_wind_hourly.csv"


def main():

    print("Starting Phase 10 wind dataset validation...")

    # ---------------------------------------------------------
    # 1. Check file exists
    # ---------------------------------------------------------

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Missing wind dataset: {INPUT_FILE}"
        )

    # ---------------------------------------------------------
    # 2. Load dataset
    # ---------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print("\nDataset shape:")
    print(df.shape)

    # ---------------------------------------------------------
    # 3. Required columns
    # ---------------------------------------------------------

    required_columns = [
        "datetime",
        "wind_speed_10m_ms",
        "latitude",
        "longitude"
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

    # ---------------------------------------------------------
    # 4. Convert datetime
    # ---------------------------------------------------------

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    if df["datetime"].isna().any():
        raise ValueError(
            "Invalid datetime values detected."
        )

    # ---------------------------------------------------------
    # 5. Check missing values
    # ---------------------------------------------------------

    missing_values = df.isna().sum()

    print("\nMissing values:")
    print(missing_values)

    if missing_values.any():
        raise ValueError(
            "Missing values detected in wind dataset."
        )

    # ---------------------------------------------------------
    # 6. Check duplicate timestamps
    # ---------------------------------------------------------

    duplicate_count = df["datetime"].duplicated().sum()

    print("\nDuplicate timestamps:")
    print(duplicate_count)

    if duplicate_count > 0:
        raise ValueError(
            "Duplicate datetime records detected."
        )

    # ---------------------------------------------------------
    # 7. Check chronological order
    # ---------------------------------------------------------

    if not df["datetime"].is_monotonic_increasing:
        raise ValueError(
            "Dataset is not sorted chronologically."
        )

    # ---------------------------------------------------------
    # 8. Check wind speed values
    # ---------------------------------------------------------

    negative_wind = (
        df["wind_speed_10m_ms"] < 0
    ).sum()

    print("\nNegative wind-speed values:")
    print(negative_wind)

    if negative_wind > 0:
        raise ValueError(
            "Negative wind-speed values detected."
        )

    # ---------------------------------------------------------
    # 9. Check geographic consistency
    # ---------------------------------------------------------

    latitude_count = df["latitude"].nunique()
    longitude_count = df["longitude"].nunique()

    print("\nUnique latitude values:")
    print(latitude_count)

    print("Unique longitude values:")
    print(longitude_count)

    if latitude_count != 1 or longitude_count != 1:
        raise ValueError(
            "Multiple geographic locations detected."
        )

    # ---------------------------------------------------------
    # 10. Check hourly intervals
    # ---------------------------------------------------------

    time_difference = (
        df["datetime"]
        .diff()
        .dropna()
    )

    non_hourly = (
        time_difference
        != pd.Timedelta(hours=1)
    ).sum()

    print("\nNon-consecutive hourly intervals:")
    print(non_hourly)

    # ---------------------------------------------------------
    # 11. Dataset summary
    # ---------------------------------------------------------

    print("\nDate range:")
    print(df["datetime"].min())
    print(df["datetime"].max())

    print("\nWind speed statistics:")
    print(
        df["wind_speed_10m_ms"].describe()
    )

    print("\nLocation:")
    print(
        f"Latitude : {df['latitude'].iloc[0]}"
    )
    print(
        f"Longitude: {df['longitude'].iloc[0]}"
    )

    # ---------------------------------------------------------
    # 12. Final validation
    # ---------------------------------------------------------

    print(
        "\nFINAL PHASE 10 WIND DATA VALIDATION: PASSED"
    )


if __name__ == "__main__":
    main()