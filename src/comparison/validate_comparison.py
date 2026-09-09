import os
import pandas as pd


METRICS_FILE = "reports/performance_comparison.csv"

FIGURE_FILE = "reports/figures/performance_comparison.png"


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

    print("Starting Phase 7 performance comparison validation...")

    # --------------------------------------------------
    # 1. Check output files
    # --------------------------------------------------

    print("\nChecking generated files...")

    if not os.path.exists(METRICS_FILE):
        raise FileNotFoundError(
            f"Missing file: {METRICS_FILE}"
        )

    print(
        f"{METRICS_FILE}: PASSED"
    )

    if not os.path.exists(FIGURE_FILE):
        raise FileNotFoundError(
            f"Missing file: {FIGURE_FILE}"
        )

    print(
        f"{FIGURE_FILE}: PASSED"
    )

    # --------------------------------------------------
    # 2. Load comparison metrics
    # --------------------------------------------------

    comparison = pd.read_csv(
        METRICS_FILE
    )

    required_columns = [
        "model",
        "MAE",
        "RMSE",
        "MAPE"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in comparison.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print("\nMetric columns: PASSED")

    # --------------------------------------------------
    # 3. Check model coverage
    # --------------------------------------------------

    missing_models = [
        model
        for model in REQUIRED_MODELS
        if model not in comparison["model"].values
    ]

    if missing_models:
        raise ValueError(
            f"Missing models: {missing_models}"
        )

    print("Model coverage: PASSED")

    # --------------------------------------------------
    # 4. Check model count
    # --------------------------------------------------

    if len(comparison) != len(REQUIRED_MODELS):
        raise ValueError(
            "Unexpected number of models in comparison."
        )

    print("Model count: PASSED")

    # --------------------------------------------------
    # 5. Check duplicate models
    # --------------------------------------------------

    if comparison["model"].duplicated().any():
        raise ValueError(
            "Duplicate models detected."
        )

    print("Duplicate model check: PASSED")

    # --------------------------------------------------
    # 6. Check metric values
    # --------------------------------------------------

    if comparison[
        ["MAE", "RMSE", "MAPE"]
    ].isnull().any().any():

        raise ValueError(
            "Missing metric values detected."
        )

    print("Missing metric check: PASSED")

    if (
        comparison[["MAE", "RMSE", "MAPE"]] < 0
    ).any().any():

        raise ValueError(
            "Negative metric values detected."
        )

    print("Metric value check: PASSED")

    # --------------------------------------------------
    # 7. Check RMSE ordering
    # --------------------------------------------------

    expected_rmse = comparison[
        "RMSE"
    ].sort_values(
        ascending=True
    ).reset_index(drop=True)

    actual_rmse = comparison[
        "RMSE"
    ].reset_index(drop=True)

    if not actual_rmse.equals(expected_rmse):
        raise ValueError(
            "Models are not sorted by RMSE."
        )

    print("RMSE ordering: PASSED")

    # --------------------------------------------------
    # 8. Identify best model
    # --------------------------------------------------

    best_model = comparison.iloc[0]["model"]

    print("\nBest model based on RMSE:")
    print(best_model)

    # --------------------------------------------------
    # 9. Final validation
    # --------------------------------------------------

    print(
        "\nFINAL PHASE 7 VALIDATION: PASSED"
    )


if __name__ == "__main__":
    main()