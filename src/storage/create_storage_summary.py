import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


INPUT_FILE = "data/processed/storage_backup_simulation.csv"

SUMMARY_FILE = "reports/storage_backup_summary.csv"
FIGURE_FILE = "reports/figures/storage_backup_comparison.png"


def main():

    print(
        "Creating Phase 12 storage vs backup summary..."
    )

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Missing file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    total_backup_only = (
        df["backup_only_kw"].sum()
    )

    total_backup_with_storage = (
        df["backup_with_storage_kw"].sum()
    )

    total_storage_discharge = (
        df["storage_discharge_kw"].sum()
    )

    backup_reduction = (
        total_backup_only
        - total_backup_with_storage
    )

    if total_backup_only > 0:
        backup_reduction_percent = (
            backup_reduction
            / total_backup_only
            * 100
        )
    else:
        backup_reduction_percent = 0.0

    summary = pd.DataFrame({
        "metric": [
            "Total Backup Without Storage (kWh)",
            "Total Backup With Storage (kWh)",
            "Total Storage Discharge (kWh)",
            "Backup Reduction (kWh)",
            "Backup Reduction (%)",
            "Battery Capacity (kWh)"
        ],
        "value": [
            total_backup_only,
            total_backup_with_storage,
            total_storage_discharge,
            backup_reduction,
            backup_reduction_percent,
            5000.0
        ]
    })

    os.makedirs(
        os.path.dirname(SUMMARY_FILE),
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(FIGURE_FILE),
        exist_ok=True
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # Backup comparison plot
    plt.figure(figsize=(12, 5))

    plt.plot(
        df["datetime"],
        df["backup_only_kw"],
        label="Backup Only"
    )

    plt.plot(
        df["datetime"],
        df["backup_with_storage_kw"],
        label="Backup With Storage"
    )

    plt.xlabel("Datetime")
    plt.ylabel("Backup Requirement (kW)")
    plt.title(
        "Storage vs Backup Requirement"
    )

    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_FILE,
        dpi=150
    )

    plt.close()

    print("\nStorage summary:")
    print(summary.to_string(index=False))

    print("\nGenerated files:")
    print(SUMMARY_FILE)
    print(FIGURE_FILE)

    print(
        "\nPHASE 12 STORAGE SUMMARY: COMPLETED"
    )


if __name__ == "__main__":
    main()