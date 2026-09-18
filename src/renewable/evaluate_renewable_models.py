import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

SOLAR_METRICS_FILE = "reports/solar_metrics.csv"
WIND_METRICS_FILE = "reports/wind_metrics.csv"
WIND_POWER_FILE = "data/processed/wind_power_estimates.csv"

OUTPUT_FILE = "reports/renewable_evaluation.csv"
FIGURE_FILE = "reports/figures/renewable_forecasting.png"


def main():

    print("Starting Phase 10 renewable model evaluation...")

    required_files = [
        SOLAR_METRICS_FILE,
        WIND_METRICS_FILE,
        WIND_POWER_FILE
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Missing required file: {file_path}"
            )

    solar_metrics = pd.read_csv(SOLAR_METRICS_FILE)
    wind_metrics = pd.read_csv(WIND_METRICS_FILE)
    wind_power = pd.read_csv(WIND_POWER_FILE)

    # Validate metric columns
    metric_columns = ["model", "MAE", "RMSE", "MAPE"]

    for column in metric_columns:
        if column not in solar_metrics.columns:
            raise ValueError(
                f"Missing solar metric column: {column}"
            )

        if column not in wind_metrics.columns:
            raise ValueError(
                f"Missing wind metric column: {column}"
            )

    # Renewable model evaluation table
    solar_row = solar_metrics.iloc[0]
    wind_row = wind_metrics.iloc[0]

    evaluation = pd.DataFrame({
        "energy_source": [
            "Solar",
            "Wind Speed"
        ],
        "model": [
            solar_row["model"],
            wind_row["model"]
        ],
        "MAE": [
            solar_row["MAE"],
            wind_row["MAE"]
        ],
        "RMSE": [
            solar_row["RMSE"],
            wind_row["RMSE"]
        ],
        "MAPE": [
            solar_row["MAPE"],
            wind_row["MAPE"]
        ]
    })

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(FIGURE_FILE),
        exist_ok=True
    )

    evaluation.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # Wind power summary
    wind_power_summary = {
        "mean_wind_power_kw":
            wind_power["estimated_wind_power_kw"].mean(),

        "maximum_wind_power_kw":
            wind_power["estimated_wind_power_kw"].max(),

        "minimum_wind_power_kw":
            wind_power["estimated_wind_power_kw"].min(),

        "mean_power_fraction":
            wind_power["actual_power_fraction"].mean()
    }

    # Save figure
    plt.figure(figsize=(12, 5))

    plt.plot(
        wind_power["datetime"],
        wind_power["estimated_wind_power_kw"]
    )

    plt.xlabel("Datetime")
    plt.ylabel("Estimated Wind Power (kW)")
    plt.title(
        "Estimated Wind Power Potential"
    )

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        FIGURE_FILE,
        dpi=150
    )

    plt.close()

    print("\nRenewable model evaluation:")
    print(evaluation.to_string(index=False))

    print("\nWind power summary:")

    for key, value in wind_power_summary.items():
        print(f"{key}: {value:.4f}")

    print("\nGenerated files:")
    print(OUTPUT_FILE)
    print(FIGURE_FILE)

    print(
        "\nPHASE 10 RENEWABLE MODEL EVALUATION: COMPLETED"
    )


if __name__ == "__main__":
    main()