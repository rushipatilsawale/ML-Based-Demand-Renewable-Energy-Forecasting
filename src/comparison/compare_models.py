import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


METRICS_FILE = "reports/time_series_metrics.csv"

OUTPUT_METRICS = "reports/performance_comparison.csv"
OUTPUT_FIGURE = "reports/figures/performance_comparison.png"


REQUIRED_MODELS = [
    "naive_24h",
    "naive_168h",
    "linear_regression",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "arima",
    "sarima"
]


def main():

    print("Starting Phase 7 performance comparison...")

    # --------------------------------------------------
    # 1. Load model metrics
    # --------------------------------------------------

    metrics = pd.read_csv(METRICS_FILE)

    print("\nLoaded metrics:")
    print(metrics)

    # --------------------------------------------------
    # 2. Validate required columns
    # --------------------------------------------------

    required_columns = [
        "model",
        "MAE",
        "RMSE",
        "MAPE"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in metrics.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing metric columns: {missing_columns}"
        )

    # --------------------------------------------------
    # 3. Validate required models
    # --------------------------------------------------

    missing_models = [
        model
        for model in REQUIRED_MODELS
        if model not in metrics["model"].values
    ]

    if missing_models:
        raise ValueError(
            f"Missing models in comparison: {missing_models}"
        )

    # --------------------------------------------------
    # 4. Keep required models
    # --------------------------------------------------

    comparison = metrics[
        metrics["model"].isin(REQUIRED_MODELS)
    ].copy()

    # --------------------------------------------------
    # 5. Check metric values
    # --------------------------------------------------

    if comparison[
        ["MAE", "RMSE", "MAPE"]
    ].isnull().any().any():

        raise ValueError(
            "Missing metric values detected."
        )

    if (
        comparison[
            ["MAE", "RMSE", "MAPE"]
        ] < 0
    ).any().any():

        raise ValueError(
            "Negative metric values detected."
        )

    # --------------------------------------------------
    # 6. Sort models by RMSE
    # --------------------------------------------------

    comparison = comparison.sort_values(
        by="RMSE",
        ascending=True
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 7. Save consolidated comparison
    # --------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_METRICS),
        exist_ok=True
    )

    comparison.to_csv(
        OUTPUT_METRICS,
        index=False
    )

    print(
        f"\nSaved comparison metrics to: "
        f"{OUTPUT_METRICS}"
    )

    # --------------------------------------------------
    # 8. Identify best models
    # --------------------------------------------------

    best_mae = comparison.loc[
        comparison["MAE"].idxmin(),
        "model"
    ]

    best_rmse = comparison.loc[
        comparison["RMSE"].idxmin(),
        "model"
    ]

    best_mape = comparison.loc[
        comparison["MAPE"].idxmin(),
        "model"
    ]

    print("\nBest model by MAE:")
    print(best_mae)

    print("\nBest model by RMSE:")
    print(best_rmse)

    print("\nBest model by MAPE:")
    print(best_mape)

    # --------------------------------------------------
    # 9. Create RMSE comparison plot
    # --------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FIGURE),
        exist_ok=True
    )

    plt.figure(figsize=(12, 6))

    plt.bar(
        comparison["model"].astype(str),
        comparison["RMSE"]
    )

    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.title("Forecasting Model Performance Comparison")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURE,
        dpi=300
    )

    plt.close()

    print(
        f"\nSaved comparison figure to: "
        f"{OUTPUT_FIGURE}"
    )

    # --------------------------------------------------
    # 10. Final comparison
    # --------------------------------------------------

    print("\nFinal Performance Comparison:")

    print(
        comparison.to_string(index=False)
    )

    print(
        "\nPhase 7 performance comparison completed."
    )


if __name__ == "__main__":
    main()