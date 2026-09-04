import os

import pandas as pd


REQUIRED_MODELS = [
    "models/random_forest.pkl",
    "models/gradient_boosting.pkl",
    "models/xgboost.pkl",
]

REQUIRED_FILES = [
    "data/processed/ml_train.csv",
    "data/processed/ml_test.csv",
    "reports/ml_model_metrics.csv",
    "data/processed/ml_predictions.csv",
    "reports/figures/ml_model_comparison.png",
]


def validate_models():
    passed = True

    print("ML Model Validation")
    print("=" * 50)

    for file in REQUIRED_MODELS + REQUIRED_FILES:
        exists = os.path.exists(file)

        print(
            f"{file:<55}: "
            f"{'PASSED' if exists else 'FAILED'}"
        )

        if not exists:
            passed = False

    # Validate metrics
    if os.path.exists("reports/ml_model_metrics.csv"):
        metrics = pd.read_csv(
            "reports/ml_model_metrics.csv"
        )

        required_models = {
            "Naive 24h",
            "Naive 168h",
            "Linear Regression",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
        }

        actual_models = set(metrics["model"])

        models_present = required_models.issubset(
            actual_models
        )

        print(
            f"{'All required models evaluated':<55}: "
            f"{'PASSED' if models_present else 'FAILED'}"
        )

        if not models_present:
            passed = False

        metric_columns = {
            "MAE_MW",
            "RMSE_MW",
            "MAPE_percent",
        }

        columns_present = metric_columns.issubset(
            set(metrics.columns)
        )

        print(
            f"{'Required metrics present':<55}: "
            f"{'PASSED' if columns_present else 'FAILED'}"
        )

        if not columns_present:
            passed = False

    print("\n" + "=" * 50)

    if passed:
        print("FINAL ML MODEL VALIDATION: PASSED")
    else:
        print("FINAL ML MODEL VALIDATION: FAILED")


if __name__ == "__main__":
    validate_models()