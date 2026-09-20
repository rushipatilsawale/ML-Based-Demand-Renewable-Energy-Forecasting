import os
import pandas as pd


INPUT_FILE = "data/processed/storage_backup_simulation.csv"

OUTPUT_FILE = "data/processed/cost_co2_impact.csv"


# Scenario assumptions
ELECTRICITY_COST_PER_KWH = 6.52

GRID_EMISSION_FACTOR_KG_CO2_PER_KWH = 0.716


def load_simulation_data():

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Missing file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    return df


def calculate_impact(df):

    result = df.copy()

    # Backup energy is represented by the backup
    # requirement in each hourly simulation step.
    result["backup_only_kwh"] = (
        result["backup_only_kw"]
    )

    result["backup_with_storage_kwh"] = (
        result["backup_with_storage_kw"]
    )

    # Electricity cost
    result["cost_without_storage_rs"] = (
        result["backup_only_kwh"]
        * ELECTRICITY_COST_PER_KWH
    )

    result["cost_with_storage_rs"] = (
        result["backup_with_storage_kwh"]
        * ELECTRICITY_COST_PER_KWH
    )

    result["cost_savings_rs"] = (
        result["cost_without_storage_rs"]
        - result["cost_with_storage_rs"]
    )

    # CO2 emissions
    result["co2_without_storage_kg"] = (
        result["backup_only_kwh"]
        * GRID_EMISSION_FACTOR_KG_CO2_PER_KWH
    )

    result["co2_with_storage_kg"] = (
        result["backup_with_storage_kwh"]
        * GRID_EMISSION_FACTOR_KG_CO2_PER_KWH
    )

    result["co2_reduction_kg"] = (
        result["co2_without_storage_kg"]
        - result["co2_with_storage_kg"]
    )

    return result


def main():

    print(
        "Starting Phase 13 cost and CO2 impact analysis..."
    )

    df = load_simulation_data()

    result = calculate_impact(df)

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nAnalysis completed.")

    print("\nScenario assumptions:")
    print(
        f"Electricity cost: "
        f"Rs. {ELECTRICITY_COST_PER_KWH:.2f}/kWh"
    )

    print(
        f"Grid emission factor: "
        f"{GRID_EMISSION_FACTOR_KG_CO2_PER_KWH:.3f} "
        f"kg CO2/kWh"
    )

    print("\nOutput:")
    print(OUTPUT_FILE)

    print(
        "\nPHASE 13 COST AND CO2 CALCULATION: COMPLETED"
    )


if __name__ == "__main__":
    main()