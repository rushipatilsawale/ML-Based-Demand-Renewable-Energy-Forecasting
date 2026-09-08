import os
import pandas as pd


MODEL_FILES = [
    "models/arima.pkl",
    "models/sarima.pkl"
]

REQUIRED_FILES = [
    "data/processed/ts_train.csv",
    "data/processed/ts_test.csv",
    "data/processed/ts_predictions.csv",
    "reports/time_series_metrics.csv"
]


def main():

    print("Starting Phase 6 time-series validation...\n")

    # --------------------------------------------------
    # 1. Check model files
    # --------------------------------------------------

    print("Checking model artifacts...")

    for file in MODEL_FILES:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Missing model: {file}")

    print("ARIMA model: PASSED")
    print("SARIMA model: PASSED")

    # --------------------------------------------------
    # 2. Check required generated files
    # --------------------------------------------------

    print("\nChecking generated files...")

    for file in REQUIRED_FILES:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Missing file: {file}")
        print(f"{file}: PASSED")

    # --------------------------------------------------
    # 3. Validate train/test data
    # --------------------------------------------------

    train = pd.read_csv("data/processed/ts_train.csv")
    test = pd.read_csv("data/processed/ts_test.csv")

    train["datetime"] = pd.to_datetime(train["datetime"])
    test["datetime"] = pd.to_datetime(test["datetime"])

    required_columns = [
        "datetime",
        "national_demand_mw"
    ]

    for column in required_columns:
        if column not in train.columns:
            raise ValueError(
                f"Missing column in training data: {column}"
            )

        if column not in test.columns:
            raise ValueError(
                f"Missing column in testing data: {column}"
            )

    print("\nTime-series columns: PASSED")

    # Check chronology
    if not train["datetime"].is_monotonic_increasing:
        raise ValueError("Training data is not chronological.")

    if not test["datetime"].is_monotonic_increasing:
        raise ValueError("Testing data is not chronological.")

    print("Training chronology: PASSED")
    print("Testing chronology: PASSED")

    # Check duplicates
    if train["datetime"].duplicated().any():
        raise ValueError("Duplicate timestamps in training data.")

    if test["datetime"].duplicated().any():
        raise ValueError("Duplicate timestamps in testing data.")

    print("Duplicate timestamp check: PASSED")

    # --------------------------------------------------
    # 4. Validate predictions
    # --------------------------------------------------

    predictions = pd.read_csv(
        "data/processed/ts_predictions.csv"
    )

    predictions["datetime"] = pd.to_datetime(
        predictions["datetime"]
    )

    required_prediction_columns = [
        "datetime",
        "actual",
        "arima",
        "sarima"
    ]

    for column in required_prediction_columns:
        if column not in predictions.columns:
            raise ValueError(
                f"Missing prediction column: {column}"
            )

    print("\nPrediction columns: PASSED")

    # Check ARIMA/SARIMA missing values
    if predictions["arima"].isna().any():
        raise ValueError("ARIMA predictions contain missing values.")

    if predictions["sarima"].isna().any():
        raise ValueError("SARIMA predictions contain missing values.")

    print("ARIMA predictions: PASSED")
    print("SARIMA predictions: PASSED")

    # --------------------------------------------------
    # 5. Validate metrics
    # --------------------------------------------------

    metrics = pd.read_csv(
        "reports/time_series_metrics.csv"
    )

    required_metric_columns = [
        "model",
        "MAE",
        "RMSE",
        "MAPE"
    ]

    for column in required_metric_columns:
        if column not in metrics.columns:
            raise ValueError(
                f"Missing metric column: {column}"
            )

    print("\nMetric columns: PASSED")

    required_models = [
        "naive_24h",
        "naive_168h",
        "linear_regression",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "arima",
        "sarima"
    ]

    missing_models = [
        model for model in required_models
        if model not in metrics["model"].values
    ]

    if missing_models:
        raise ValueError(
            f"Missing models in evaluation: {missing_models}"
        )

    print("All required models evaluated: PASSED")

    # Check metric values
    if metrics[["MAE", "RMSE", "MAPE"]].isna().any().any():
        raise ValueError("Metrics contain missing values.")

    if (metrics[["MAE", "RMSE", "MAPE"]] < 0).any().any():
        raise ValueError("Metrics contain negative values.")

    print("Metric values: PASSED")

    # --------------------------------------------------
    # 6. Final validation
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("FINAL TIME-SERIES VALIDATION: PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()