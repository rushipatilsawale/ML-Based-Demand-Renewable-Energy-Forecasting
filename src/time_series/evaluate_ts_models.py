import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error


TEST_FILE = "data/processed/ts_test.csv"
BASELINE_FILE = "data/processed/baseline_predictions.csv"
ML_PREDICTIONS_FILE = "data/processed/ml_predictions.csv"

ARIMA_MODEL = "models/arima.pkl"
SARIMA_MODEL = "models/sarima.pkl"

OUTPUT_PREDICTIONS = "data/processed/ts_predictions.csv"
OUTPUT_METRICS = "reports/time_series_metrics.csv"


def calculate_mape(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)

    mask = actual != 0

    return np.mean(
        np.abs((actual[mask] - predicted[mask]) / actual[mask])
    ) * 100


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mape = calculate_mape(actual, predicted)

    return mae, rmse, mape


def main():

    # --------------------------------------------------
    # 1. Load test data
    # --------------------------------------------------

    test = pd.read_csv(TEST_FILE)

    test["datetime"] = pd.to_datetime(test["datetime"])

    test = test.sort_values("datetime").reset_index(drop=True)

    print("Test rows:", len(test))
    print(
        "Test period:",
        test["datetime"].min(),
        "to",
        test["datetime"].max()
    )

    # --------------------------------------------------
    # 2. Load ARIMA model
    # --------------------------------------------------

    print("\nGenerating ARIMA forecast...")

    arima_model = joblib.load(ARIMA_MODEL)

    arima_forecast = np.asarray(
        arima_model.forecast(steps=len(test))
    )

    # --------------------------------------------------
    # 3. Load SARIMA model
    # --------------------------------------------------

    print("Generating SARIMA forecast...")

    sarima_model = joblib.load(SARIMA_MODEL)

    sarima_forecast = np.asarray(
        sarima_model.forecast(steps=len(test))
    )

    # --------------------------------------------------
    # 4. Load Phase 4 baseline predictions
    # --------------------------------------------------

    baseline = pd.read_csv(BASELINE_FILE)

    baseline["datetime"] = pd.to_datetime(
        baseline["datetime"]
    )

    baseline = baseline[
        baseline["datetime"].isin(test["datetime"])
    ].copy()

    # --------------------------------------------------
    # 5. Load Phase 5 ML predictions
    # --------------------------------------------------

    ml = pd.read_csv(ML_PREDICTIONS_FILE)

    ml["datetime"] = pd.to_datetime(
        ml["datetime"]
    )

    ml = ml[
        ml["datetime"].isin(test["datetime"])
    ].copy()

    # --------------------------------------------------
    # 6. Create common prediction dataframe
    # --------------------------------------------------

    predictions = test[
        ["datetime", "national_demand_mw"]
    ].copy()

    predictions.rename(
        columns={
            "national_demand_mw": "actual"
        },
        inplace=True
    )

    # --------------------------------------------------
    # 6A. Phase 4 Baseline Models
    # --------------------------------------------------

    baseline_column_mapping = {
        "Naive 24h": "naive_24h",
        "Naive 168h": "naive_168h",
        "Linear Regression": "linear_regression",
        "naive_24h": "naive_24h",
        "naive_168h": "naive_168h",
        "linear_regression": "linear_regression"
    }

    for source_column, target_column in baseline_column_mapping.items():

        if source_column in baseline.columns:

            predictions = predictions.merge(
                baseline[
                    ["datetime", source_column]
                ].rename(
                    columns={
                        source_column: target_column
                    }
                ),
                on="datetime",
                how="left"
            )

    # --------------------------------------------------
    # 6B. Phase 5 Machine Learning Models
    # --------------------------------------------------

    ml_column_mapping = {
        "Random Forest": "random_forest",
        "Gradient Boosting": "gradient_boosting",
        "XGBoost": "xgboost",
        "random_forest": "random_forest",
        "gradient_boosting": "gradient_boosting",
        "xgboost": "xgboost"
    }

    for source_column, target_column in ml_column_mapping.items():

        if source_column in ml.columns:

            predictions = predictions.merge(
                ml[
                    ["datetime", source_column]
                ].rename(
                    columns={
                        source_column: target_column
                    }
                ),
                on="datetime",
                how="left"
            )

    # --------------------------------------------------
    # 6C. Phase 6 Time-Series Models
    # --------------------------------------------------

    predictions["arima"] = arima_forecast
    predictions["sarima"] = sarima_forecast

    # --------------------------------------------------
    # 7. Save all predictions
    # --------------------------------------------------

    predictions.to_csv(
        OUTPUT_PREDICTIONS,
        index=False
    )

    # --------------------------------------------------
    # 8. Evaluate all models
    # --------------------------------------------------

    model_columns = [
        "naive_24h",
        "naive_168h",
        "linear_regression",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "arima",
        "sarima"
    ]

    results = []

    for model_name in model_columns:

        if model_name not in predictions.columns:
            continue

        valid = predictions[
            ["actual", model_name]
        ].dropna()

        if len(valid) == 0:
            continue

        mae, rmse, mape = calculate_metrics(
            valid["actual"],
            valid[model_name]
        )

        results.append({
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        })

    metrics = pd.DataFrame(results)

    # --------------------------------------------------
    # 9. Sort and save metrics
    # --------------------------------------------------

    metrics = metrics.sort_values(
        "RMSE"
    ).reset_index(drop=True)

    metrics.to_csv(
        OUTPUT_METRICS,
        index=False
    )

    # --------------------------------------------------
    # 10. Display results
    # --------------------------------------------------

    print("\nTime-Series Model Evaluation:")
    print(metrics.to_string(index=False))

    print("\nBest model based on RMSE:")
    print(metrics.iloc[0]["model"])


if __name__ == "__main__":
    main()