import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SOLAR_FILE = "data/processed/solar_uncertainty.csv"
WIND_FILE = "data/processed/wind_uncertainty.csv"

SUMMARY_FILE = "reports/uncertainty_summary.csv"
FIGURE_FILE = "reports/figures/uncertainty_intervals.png"


def calculate_summary(df, energy_source):

    interval_width = (
        df["upper_bound"] - df["lower_bound"]
    )

    coverage = (
        (df["actual"] >= df["lower_bound"])
        & (df["actual"] <= df["upper_bound"])
    ).mean()

    return {
        "energy_source": energy_source,
        "confidence_level": df["confidence_level"].iloc[0],
        "mean_prediction": df["prediction"].mean(),
        "mean_interval_width": interval_width.mean(),
        "interval_coverage": coverage
    }


def main():

    print("Creating Phase 11 uncertainty summary...")

    if not os.path.exists(SOLAR_FILE):
        raise FileNotFoundError(
            f"Missing file: {SOLAR_FILE}"
        )

    if not os.path.exists(WIND_FILE):
        raise FileNotFoundError(
            f"Missing file: {WIND_FILE}"
        )

    solar = pd.read_csv(SOLAR_FILE)
    wind = pd.read_csv(WIND_FILE)

    solar_summary = calculate_summary(
        solar,
        "Solar"
    )

    wind_summary = calculate_summary(
        wind,
        "Wind Speed"
    )

    summary = pd.DataFrame([
        solar_summary,
        wind_summary
    ])

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

    # Solar uncertainty visualization
    plt.figure(figsize=(12, 5))

    plt.plot(
        solar["datetime"],
        solar["actual"],
        label="Actual"
    )

    plt.plot(
        solar["datetime"],
        solar["prediction"],
        label="Prediction"
    )

    plt.fill_between(
        range(len(solar)),
        solar["lower_bound"],
        solar["upper_bound"],
        alpha=0.2,
        label="95% Prediction Interval"
    )

    plt.xlabel("Test Observation")
    plt.ylabel("Solar Generation AC")
    plt.title(
        "Solar Forecast Uncertainty"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_FILE,
        dpi=150
    )

    plt.close()

    print("\nUncertainty summary:")
    print(summary.to_string(index=False))

    print("\nGenerated files:")
    print(SUMMARY_FILE)
    print(FIGURE_FILE)

    print(
        "\nPHASE 11 UNCERTAINTY SUMMARY: COMPLETED"
    )


if __name__ == "__main__":
    main()