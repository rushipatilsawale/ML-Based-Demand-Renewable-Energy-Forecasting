import os
import pandas as pd


INPUT_FILE = "data/processed/cost_co2_impact.csv"


def main():

    print(
        "Starting Phase 13 cost and CO2 validation..."
    )

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Missing file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "datetime",
        "backup_only_kwh",
        "backup_with_storage_kwh",
        "cost_without_storage_rs",
        "cost_with_storage_rs",
        "cost_savings_rs",
        "co2_without_storage_kg",
        "co2_with_storage_kg",
        "co2_reduction_kg"
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
            "Cost and CO2 output is empty."
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    numeric_columns = [
        "backup_only_kwh",
        "backup_with_storage_kwh",
        "cost_without_storage_rs",
        "cost_with_storage_rs",
        "cost_savings_rs",
        "co2_without_storage_kg",
        "co2_with_storage_kg",
        "co2_reduction_kg"
    ]

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "Missing numeric values detected."
        )

    if (df[numeric_columns] < 0).any().any():
        raise ValueError(
            "Negative cost or CO2 values detected."
        )

    if (
        df["backup_with_storage_kwh"]
        > df["backup_only_kwh"] + 1e-9
    ).any():
        raise ValueError(
            "Storage scenario requires more backup "
            "than backup-only scenario."
        )

    if (
        df["cost_with_storage_rs"]
        > df["cost_without_storage_rs"] + 1e-9
    ).any():
        raise ValueError(
            "Storage scenario has higher backup cost."
        )

    if (
        df["co2_with_storage_kg"]
        > df["co2_without_storage_kg"] + 1e-9
    ).any():
        raise ValueError(
            "Storage scenario has higher CO2 emissions."
        )

    print("\nOutput shape:")
    print(df.shape)

    print(
        "\nTotal cost without storage:"
    )
    print(
        f"Rs. "
        f"{df['cost_without_storage_rs'].sum():.2f}"
    )

    print(
        "\nTotal cost with storage:"
    )
    print(
        f"Rs. "
        f"{df['cost_with_storage_rs'].sum():.2f}"
    )

    print(
        "\nTotal cost savings:"
    )
    print(
        f"Rs. "
        f"{df['cost_savings_rs'].sum():.2f}"
    )

    print(
        "\nTotal CO2 without storage:"
    )
    print(
        f"{df['co2_without_storage_kg'].sum():.2f} kg"
    )

    print(
        "\nTotal CO2 with storage:"
    )
    print(
        f"{df['co2_with_storage_kg'].sum():.2f} kg"
    )

    print(
        "\nTotal CO2 reduction:"
    )
    print(
        f"{df['co2_reduction_kg'].sum():.2f} kg"
    )

    print(
        "\nFINAL PHASE 13 COST AND CO2 VALIDATION: PASSED"
    )


if __name__ == "__main__":
    main()