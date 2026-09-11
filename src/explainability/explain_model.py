import os
import joblib
import pandas as pd
import shap
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


MODEL_FILE = "models/linear_regression.pkl"

TEST_FILE = "data/processed/ml_test.csv"

OUTPUT_SUMMARY = "reports/explainability_summary.csv"

OUTPUT_SHAP_SUMMARY = "reports/figures/shap_summary.png"

OUTPUT_SHAP_BAR = "reports/figures/shap_bar.png"

TARGET_COLUMN = "national_demand_mw"

DATETIME_COLUMN = "datetime"


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


def main():

    print("Starting Phase 9 explainability analysis...")

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Missing model file: {MODEL_FILE}"
        )

    model = joblib.load(MODEL_FILE)

    print(f"\nLoaded model: {MODEL_FILE}")

    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(
            f"Missing test file: {TEST_FILE}"
        )

    test_data = pd.read_csv(TEST_FILE)

    print(f"Loaded test data: {TEST_FILE}")

    if TARGET_COLUMN not in test_data.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in test_data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features in test data: {missing_features}"
        )

    X_test = test_data[FEATURES]

    if X_test.empty:
        raise ValueError(
            "No features available for SHAP analysis."
        )

    print(
        f"\nNumber of features: {X_test.shape[1]}"
    )

    sample_size = min(
        1000,
        len(X_test)
    )

    X_sample = X_test.sample(
        n=sample_size,
        random_state=42
    )

    print(
        f"SHAP sample size: {sample_size}"
    )

    print("\nCreating SHAP LinearExplainer...")

    explainer = shap.LinearExplainer(
        model,
        X_sample
    )

    shap_values = explainer(
        X_sample
    ).values

    mean_abs_shap = (
        abs(shap_values)
        .mean(axis=0)
    )

    explainability_summary = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs_shap
    })

    explainability_summary = (
        explainability_summary
        .sort_values(
            by="mean_abs_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )

    os.makedirs(
        os.path.dirname(OUTPUT_SUMMARY),
        exist_ok=True
    )

    explainability_summary.to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    print(
        f"\nSaved SHAP summary to: "
        f"{OUTPUT_SUMMARY}"
    )

    os.makedirs(
        os.path.dirname(OUTPUT_SHAP_SUMMARY),
        exist_ok=True
    )

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_SHAP_SUMMARY,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved SHAP summary plot to: "
        f"{OUTPUT_SHAP_SUMMARY}"
    )

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_SHAP_BAR,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved SHAP bar plot to: "
        f"{OUTPUT_SHAP_BAR}"
    )

    print("\nTop features by mean absolute SHAP value:")

    print(
        explainability_summary
        .head(10)
        .to_string(index=False)
    )

    print(
        "\nPhase 9 explainability analysis completed."
    )


if __name__ == "__main__":
    main()