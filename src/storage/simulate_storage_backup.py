import os
import numpy as np
import pandas as pd


SOLAR_FILE = "data/processed/solar_uncertainty.csv"

OUTPUT_FILE = "data/processed/storage_backup_simulation.csv"


# Scenario-based battery assumptions
BATTERY_CAPACITY_KWH = 5000.0
INITIAL_SOC_KWH = 2500.0

MAX_CHARGE_KW = 1000.0
MAX_DISCHARGE_KW = 1000.0

CHARGE_EFFICIENCY = 0.90
DISCHARGE_EFFICIENCY = 0.90


def load_solar_data():

    if not os.path.exists(SOLAR_FILE):
        raise FileNotFoundError(
            f"Missing file: {SOLAR_FILE}"
        )

    solar = pd.read_csv(SOLAR_FILE)

    solar["datetime"] = pd.to_datetime(
        solar["datetime"]
    )

    solar = solar.sort_values(
        "datetime"
    ).reset_index(drop=True)

    return solar


def create_demand_scenario(df):

    hour = df["datetime"].dt.hour

    demand = (
        1800
        + 600 * np.sin(
            2 * np.pi * (hour - 7) / 24
        )
        + 300 * (
            (hour >= 18)
            & (hour <= 22)
        )
    )

    return demand.astype(float)


def simulate_storage(df):

    battery_soc = INITIAL_SOC_KWH

    results = []

    for _, row in df.iterrows():

        renewable_energy = max(
            row["prediction"],
            0
        )

        demand = row["demand_kw"]

        renewable_used = min(
            renewable_energy,
            demand
        )

        surplus = max(
            renewable_energy - demand,
            0
        )

        deficit = max(
            demand - renewable_energy,
            0
        )

        # Charge battery using renewable surplus
        charge_input = min(
            surplus,
            MAX_CHARGE_KW,
            (
                BATTERY_CAPACITY_KWH
                - battery_soc
            ) / CHARGE_EFFICIENCY
        )

        battery_soc += (
            charge_input
            * CHARGE_EFFICIENCY
        )

        # Remaining demand deficit
        remaining_deficit = deficit

        # Discharge battery
        available_discharge = min(
            MAX_DISCHARGE_KW,
            battery_soc * DISCHARGE_EFFICIENCY
        )

        battery_discharge = min(
            remaining_deficit,
            available_discharge
        )

        battery_soc -= (
            battery_discharge
            / DISCHARGE_EFFICIENCY
        )

        remaining_deficit -= (
            battery_discharge
        )

        backup_with_storage = max(
            remaining_deficit,
            0
        )

        backup_only = max(
            demand - renewable_energy,
            0
        )

        results.append({
            "datetime": row["datetime"],
            "solar_prediction_kw":
                renewable_energy,
            "demand_kw":
                demand,
            "battery_soc_kwh":
                battery_soc,
            "renewable_used_kw":
                renewable_used,
            "storage_discharge_kw":
                battery_discharge,
            "backup_with_storage_kw":
                backup_with_storage,
            "backup_only_kw":
                backup_only
        })

    return pd.DataFrame(results)


def main():

    print(
        "Starting Phase 12 storage vs backup simulation..."
    )

    solar = load_solar_data()

    solar["demand_kw"] = (
        create_demand_scenario(solar)
    )

    result = simulate_storage(solar)

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSimulation completed.")

    print("\nSimulation shape:")
    print(result.shape)

    print("\nBattery configuration:")
    print(
        f"Capacity: "
        f"{BATTERY_CAPACITY_KWH:.0f} kWh"
    )

    print(
        f"Initial SOC: "
        f"{INITIAL_SOC_KWH:.0f} kWh"
    )

    print(
        f"Charge efficiency: "
        f"{CHARGE_EFFICIENCY:.2f}"
    )

    print(
        f"Discharge efficiency: "
        f"{DISCHARGE_EFFICIENCY:.2f}"
    )

    print("\nOutput:")
    print(OUTPUT_FILE)

    print(
        "\nPHASE 12 STORAGE SIMULATION: COMPLETED"
    )


if __name__ == "__main__":
    main()