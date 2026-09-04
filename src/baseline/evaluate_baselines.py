import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


INPUT_FILE = "data/processed/baseline_predictions.csv"
METRICS_FILE = "reports/baseline_metrics.csv"
FIGURE_FILE = "reports/figures/baseline_comparison.png"

TARGET = "national_demand_mw"

MODELS = {
    "Naive 24h": "naive_24h",
    "Naive 168h": "naive_168h",
    "Linear Regression": "linear_regression",
}


def calculate_mape(actual, predicted):
    return (
        abs((actual - predicted) / actual)
        .replace([float("inf"), -float("inf")], pd.NA)
        .dropna()
        .mean()
        * 100
    )


def evaluate():
    df = pd.read_csv(INPUT_FILE)

    metrics = []

    for model_name, prediction_column in MODELS.items():

        evaluation_data = df[
            [TARGET, prediction_column]
        ].dropna()

        actual = evaluation_data[TARGET]
        predicted = evaluation_data[prediction_column]

        mae = mean_absolute_error(actual, predicted)

        rmse = mean_squared_error(
            actual,
            predicted,
        ) ** 0.5

        mape = calculate_mape(actual, predicted)

        metrics.append(
            {
                "model": model_name,
                "MAE_MW": mae,
                "RMSE_MW": rmse,
                "MAPE_percent": mape,
            }
        )

    metrics_df = pd.DataFrame(metrics)

    metrics_df = metrics_df.sort_values(
        "RMSE_MW"
    ).reset_index(drop=True)

    os.makedirs("reports/figures", exist_ok=True)

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    # Plot RMSE comparison
    plt.figure(figsize=(10, 6))

    plt.bar(
        metrics_df["model"],
        metrics_df["RMSE_MW"],
    )

    plt.xlabel("Baseline Model")
    plt.ylabel("RMSE (MW)")
    plt.title("Baseline Forecasting RMSE Comparison")

    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(FIGURE_FILE, dpi=300)
    plt.close()

    print("\nBaseline Evaluation Results")
    print("=" * 50)

    print(metrics_df.to_string(index=False))

    print("\nBest baseline:")
    print(metrics_df.iloc[0]["model"])

    print(f"\nMetrics saved to: {METRICS_FILE}")
    print(f"Figure saved to : {FIGURE_FILE}")


if __name__ == "__main__":
    evaluate()