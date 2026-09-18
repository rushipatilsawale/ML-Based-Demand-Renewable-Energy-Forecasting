import os
import pandas as pd


PLANT_1_FILE = "data/raw/renewable/Plant_1_Generation_Data.csv"
PLANT_2_FILE = "data/raw/renewable/Plant_2_Generation_Data.csv"

OUTPUT_FILE = "data/processed/solar_hourly.csv"


def load_generation_data(file_path):
    """Load and prepare plant generation data."""

    df = pd.read_csv(file_path)

    df["DATE_TIME"] = pd.to_datetime(
        df["DATE_TIME"],
        dayfirst=True
    )

    return df


def prepare_plant_data(df):
    """Aggregate inverter-level AC power to plant-level hourly generation."""

    # Keep only the columns required for forecasting
    df = df[
        [
            "DATE_TIME",
            "PLANT_ID",
            "SOURCE_KEY",
            "AC_POWER"
        ]
    ].copy()

    # Aggregate all inverters at the same timestamp
    plant_generation = (
        df.groupby(["DATE_TIME", "PLANT_ID"])["AC_POWER"]
        .sum()
        .reset_index()
    )

    # Convert 15-minute generation data to hourly generation
    hourly_generation = (
        plant_generation
        .set_index("DATE_TIME")
        .groupby("PLANT_ID")["AC_POWER"]
        .resample("1h")
        .mean()
        .reset_index()
    )

    return hourly_generation


def main():

    print("Starting Phase 10 solar data preparation...")

    # Check input files
    if not os.path.exists(PLANT_1_FILE):
        raise FileNotFoundError(
            f"Missing file: {PLANT_1_FILE}"
        )

    if not os.path.exists(PLANT_2_FILE):
        raise FileNotFoundError(
            f"Missing file: {PLANT_2_FILE}"
        )

    # Load datasets
    print("\nLoading Plant 1 generation data...")
    plant_1 = load_generation_data(PLANT_1_FILE)

    print("Loading Plant 2 generation data...")
    plant_2 = load_generation_data(PLANT_2_FILE)

    print(f"Plant 1 rows: {len(plant_1)}")
    print(f"Plant 2 rows: {len(plant_2)}")

    # Prepare each plant
    print("\nAggregating Plant 1 inverter generation...")
    plant_1_hourly = prepare_plant_data(plant_1)

    print("Aggregating Plant 2 inverter generation...")
    plant_2_hourly = prepare_plant_data(plant_2)

    # Combine both plants
    solar_hourly = pd.concat(
        [
            plant_1_hourly,
            plant_2_hourly
        ],
        ignore_index=True
    )

    # Rename columns
    solar_hourly = solar_hourly.rename(
        columns={
            "DATE_TIME": "datetime",
            "PLANT_ID": "plant_id",
            "AC_POWER": "solar_generation_ac"
        }
    )

    # Sort chronologically
    solar_hourly = solar_hourly.sort_values(
    by=["datetime", "plant_id"]
).reset_index(drop=True)

    # ---------------------------------------------------------
    # Handle missing hourly solar generation values
    # ---------------------------------------------------------
    # Missing observations are interpolated within each plant.
    # We do not replace missing values with zero.

    missing_before = solar_hourly["solar_generation_ac"].isna().sum()

    print(
        f"\nMissing solar generation values before interpolation: "
        f"{missing_before}"
    )

    solar_hourly["solar_generation_ac"] = (
        solar_hourly
        .groupby("plant_id")["solar_generation_ac"]
        .transform(
            lambda x: x.interpolate(method="linear")
        )
    )

    missing_after = solar_hourly["solar_generation_ac"].isna().sum()

    print(
        f"Missing solar generation values after interpolation: "
        f"{missing_after}"
    )

    # Safety check
    if missing_after > 0:
        raise ValueError(
            f"Missing solar generation values remain after "
            f"interpolation: {missing_after}"
        )

    # Create output directory
    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    # Save output
    solar_hourly.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nSaved hourly solar dataset:")
    print(OUTPUT_FILE)

    print("\nFinal shape:")
    print(solar_hourly.shape)

    print("\nColumns:")
    print(solar_hourly.columns.tolist())

    print("\nDate range:")
    print(solar_hourly["datetime"].min())
    print(solar_hourly["datetime"].max())

    print("\nPlant-wise records:")
    print(
        solar_hourly["plant_id"]
        .value_counts()
        .sort_index()
    )

    print("\nMissing values:")
    print(solar_hourly.isna().sum())

    print("\nPhase 10 solar data preparation completed.")


if __name__ == "__main__":
    main()