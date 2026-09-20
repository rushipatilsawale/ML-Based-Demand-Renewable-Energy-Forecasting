import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


INPUT_FILE = "data/processed/cost_co2_impact.csv"

SUMMARY_FILE = "reports/cost_co2_summary.csv"

FIGURE_FILE = "reports/figures/cost_co2_impact.png"


def main():

    print(
        "Creating Phase 13 cost and CO2 summary..."
    )

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Missing file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    total_cost_without_storage = (
        df["cost_without_storage_rs"].sum()
    )

    total_cost_with_storage = (
        df["cost_with_storage_rs"].sum()
    )

    total_cost_savings = (
        df["cost_savings_rs"].sum()
    )

    total_co2_without_storage = (
        df["co2_without_storage_kg"].sum()
    )

    total_co2_with_storage = (
        df["co2_with_storage_kg"].sum()
    )

    total_co2_reduction = (
        df["co2_reduction_kg"].sum()
    )

    if total_cost_without_storage > 0:

        cost_savings_percent = (
            total_cost_savings
            / total_cost_without_storage
            * 100
        )

    else:

        cost_savings_percent = 0.0

    if total_co2_without_storage > 0:

        co2_reduction_percent = (
            total_co2_reduction
            / total_co2_without_storage
            * 100
        )

    else:

        co2_reduction_percent = 0.0

    summary = pd.DataFrame({

        "metric": [

            "Total Cost Without Storage (Rs.)",

            "Total Cost With Storage (Rs.)",

            "Cost Savings (Rs.)",

            "Cost Savings (%)",

            "Total CO2 Without Storage (kg)",

            "Total CO2 With Storage (kg)",

            "CO2 Reduction (kg)",

            "CO2 Reduction (%)"

        ],

        "value": [

            total_cost_without_storage,

            total_cost_with_storage,

            total_cost_savings,

            cost_savings_percent,

            total_co2_without_storage,

            total_co2_with_storage,

            total_co2_reduction,

            co2_reduction_percent

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

    # Cost comparison
    plt.figure(figsize=(8, 5))

    plt.bar(
        [
            "Without Storage",
            "With Storage"
        ],
        [
            total_cost_without_storage,
            total_cost_with_storage
        ]
    )

    plt.ylabel("Cost (Rs.)")

    plt.title(
        "Electricity Backup Cost: Storage vs No Storage"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_FILE,
        dpi=150
    )

    plt.close()

    print("\nCost and CO2 summary:")
    print(
        summary.to_string(index=False)
    )

    print("\nGenerated files:")
    print(SUMMARY_FILE)
    print(FIGURE_FILE)

    print(
        "\nPHASE 13 COST AND CO2 SUMMARY: COMPLETED"
    )


if __name__ == "__main__":
    main()