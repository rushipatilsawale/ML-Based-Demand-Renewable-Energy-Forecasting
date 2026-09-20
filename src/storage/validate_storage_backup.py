import os
import pandas as pd


INPUT_FILE = "data/processed/storage_backup_simulation.csv"


def main():

    print(
        "Starting Phase 12 storage vs backup validation..."
    )

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Missing file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "datetime",
        "solar_prediction_kw",
        "demand_kw",
        "battery_soc_kwh",
        "renewable_used_kw",
        "storage_discharge_kw",
        "backup_with_storage_kw",
        "backup_only_kw"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError(
            "Storage simulation output is empty."
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    numeric_columns = [
        "solar_prediction_kw",
        "demand_kw",
        "battery_soc_kwh",
        "renewable_used_kw",
        "storage_discharge_kw",
        "backup_with_storage_kw",
        "backup_only_kw"
    ]

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "Missing numeric values detected."
        )

    if (df[numeric_columns] < 0).any().any():
        raise ValueError(
            "Negative simulation values detected."
        )

    if (df["battery_soc_kwh"] > 5000).any():
        raise ValueError(
            "Battery SOC exceeds configured capacity."
        )

    if (
        df["backup_with_storage_kw"]
        > df["backup_only_kw"] + 1e-9
    ).any():
        raise ValueError(
            "Storage scenario requires more backup "
            "than backup-only scenario."
        )

    total_backup_only = (
        df["backup_only_kw"].sum()
    )

    total_backup_with_storage = (
        df["backup_with_storage_kw"].sum()
    )

    backup_reduction = (
        total_backup_only
        - total_backup_with_storage
    )

    print("\nSimulation shape:")
    print(df.shape)

    print("\nTotal backup energy:")
    print(
        f"Backup-only: "
        f"{total_backup_only:.2f} kWh"
    )

    print(
        f"With storage: "
        f"{total_backup_with_storage:.2f} kWh"
    )

    print(
        f"Backup reduction: "
        f"{backup_reduction:.2f} kWh"
    )

    print(
        "\nFINAL PHASE 12 STORAGE VALIDATION: PASSED"
    )


if __name__ == "__main__":
    main()