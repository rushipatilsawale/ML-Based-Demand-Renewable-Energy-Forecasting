import os
import pandas as pd


INPUT_FILE = "data/processed/solar_hourly.csv"

REQUIRED_COLUMNS = [
    "datetime",
    "plant_id",
    "solar_generation_ac"
]


def main():

    print("Starting Phase 10 solar dataset validation...")

    # 1. Check file exists
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Missing file: {INPUT_FILE}"
        )

    # 2. Load dataset
    df = pd.read_csv(INPUT_FILE)

    print(f"\nDataset shape: {df.shape}")

    # 3. Check required columns
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # 4. Convert datetime
    df["datetime"] = pd.to_datetime(df["datetime"])

    # 5. Check missing values
    missing_values = df[REQUIRED_COLUMNS].isna().sum()

    if missing_values.any():
        raise ValueError(
            f"Missing values detected:\n{missing_values}"
        )

    # 6. Check duplicate rows
    duplicates = df.duplicated(
        subset=["datetime", "plant_id"]
    ).sum()

    if duplicates > 0:
        raise ValueError(
            f"Duplicate datetime/plant records detected: {duplicates}"
        )

    # 7. Check negative generation
    negative_generation = (
        df["solar_generation_ac"] < 0
    ).sum()

    if negative_generation > 0:
        raise ValueError(
            f"Negative solar generation values detected: "
            f"{negative_generation}"
        )

    # 8. Check plant IDs
    plants = sorted(df["plant_id"].unique())

    print(f"\nPlant IDs: {plants}")

    if len(plants) != 2:
        raise ValueError(
            f"Expected 2 solar plants, found {len(plants)}"
        )

    # 9. Check datetime ordering
    if not df["datetime"].is_monotonic_increasing:
        raise ValueError(
            "Dataset is not sorted chronologically."
        )

    # 10. Check hourly frequency within each plant
    for plant_id in plants:

        plant_data = (
            df[df["plant_id"] == plant_id]
            .sort_values("datetime")
        )

        intervals = (
            plant_data["datetime"]
            .diff()
            .dropna()
        )

        invalid_intervals = (
            intervals != pd.Timedelta(hours=1)
        ).sum()

        # Gaps are allowed because the raw dataset contains
        # some missing timestamps.
        print(
            f"Plant {plant_id}: "
            f"{len(plant_data)} hourly records, "
            f"{invalid_intervals} non-consecutive intervals"
        )

    # 11. Print summary
    print("\nDate range:")
    print(df["datetime"].min())
    print(df["datetime"].max())

    print("\nSolar generation statistics:")
    print(
        df["solar_generation_ac"]
        .describe()
    )

    print("\nMissing values:")
    print(
        df[REQUIRED_COLUMNS]
        .isna()
        .sum()
    )

    print("\nFINAL PHASE 10 SOLAR VALIDATION: PASSED")


if __name__ == "__main__":
    main()