import os

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


TEST_FILE = "data/processed/ml_test.csv"
BASELINE_FILE = "data/processed/baseline_predictions.csv"

METRICS_FILE = "reports/ml_model_metrics.csv"
PREDICTIONS_FILE = "data/processed/ml_predictions.csv"
FIGURE_FILE = "reports/figures/ml_model_comparison.png"

TARGET = "national_demand_mw"

FEATURES = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_lag_168h",
    "demand_rolling_mean_24h",
    "demand_rolling_std_24h",
    "demand_rolling_mean_168h",
    "demand_rolling_std_168h",
    "temperature_2m_c",
    "relative_humidity_pct",
    "cloud_cover_pct",
    "precipitation_mm",
    "wind_speed_10m_kmh",
    "solar_radiation_w_m2",
]


MODELS = {
    "Random Forest": "models/random_forest.pkl",
    "Gradient Boosting": "models/gradient_boosting.pkl",
    "XGBoost": "models/xgboost.pkl",
}


def calculate_mape(actual, predicted):
    actual = pd.Series(actual)
    predicted = pd.Series(predicted)

    mask = actual != 0

    return (
        (abs((actual[mask] - predicted[mask]) / actual[mask])).mean()
        * 100
    )


def evaluate_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)

    rmse = mean_squared_error(
        actual,
        predicted,
    ) ** 0.5

    mape = calculate_mape(actual, predicted)

    return mae, rmse, mape


def evaluate_all_models():
    test = pd.read_csv(TEST_FILE)
    baseline = pd.read_csv(BASELINE_FILE)

    X_test = test[FEATURES]
    actual = test[TARGET]

    predictions = test[
        [
            "datetime",
            TARGET,
        ]
    ].copy()

    metrics = []

    # -------------------------
    # Phase 4 Baselines
    # -------------------------

    baseline = baseline[
        baseline["datetime"].isin(test["datetime"])
    ].copy()

    baseline = baseline.sort_values("datetime").reset_index(drop=True)

    for model_name, column in {
        "Naive 24h": "naive_24h",
        "Naive 168h": "naive_168h",
        "Linear Regression": "linear_regression",
    }.items():

        predicted = baseline[column]

        valid = predicted.notna()

        mae, rmse, mape = evaluate_model(
            actual[valid],
            predicted[valid],
        )

        metrics.append(
            {
                "model": model_name,
                "MAE_MW": mae,
                "RMSE_MW": rmse,
                "MAPE_percent": mape,
            }
        )

        predictions[model_name] = predicted.values

    # -------------------------
    # Phase 5 ML Models
    # -------------------------

    for model_name, model_file in MODELS.items():

        model = joblib.load(model_file)

        predicted = model.predict(X_test)

        mae, rmse, mape = evaluate_model(
            actual,
            predicted,
        )

        metrics.append(
            {
                "model": model_name,
                "MAE_MW": mae,
                "RMSE_MW": rmse,
                "MAPE_percent": mape,
            }
        )

        predictions[model_name] = predicted

    # -------------------------
    # Save results
    # -------------------------

    metrics_df = pd.DataFrame(metrics)

    metrics_df = metrics_df.sort_values(
        "RMSE_MW"
    ).reset_index(drop=True)

    os.makedirs("reports/figures", exist_ok=True)

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # -------------------------
    # RMSE comparison
    # -------------------------

    plt.figure(figsize=(11, 6))

    plt.bar(
        metrics_df["model"],
        metrics_df["RMSE_MW"],
    )

    plt.xlabel("Model")
    plt.ylabel("RMSE (MW)")
    plt.title("Baseline vs Machine Learning Model Performance")

    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(
        FIGURE_FILE,
        dpi=300,
    )

    plt.close()

    # -------------------------
    # Console output
    # -------------------------

    print("\nModel Evaluation Results")
    print("=" * 70)

    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nBest model:")
    print(metrics_df.iloc[0]["model"])

    print(
        f"\nMetrics saved to: {METRICS_FILE}"
    )

    print(
        f"Predictions saved to: {PREDICTIONS_FILE}"
    )

    print(
        f"Figure saved to: {FIGURE_FILE}"
    )


if __name__ == "__main__":
    evaluate_all_models()