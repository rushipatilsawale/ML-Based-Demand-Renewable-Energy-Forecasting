import pandas as pd

from sklearn.linear_model import LinearRegression


INPUT_FILE = "data/processed/featured_dataset.csv"
OUTPUT_FILE = "data/processed/baseline_predictions.csv"

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


def create_regression_predictions():
    df = pd.read_csv(INPUT_FILE, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Time-based split: first 80% training, last 20% testing
    split_index = int(len(df) * 0.80)

    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    model = LinearRegression()
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)

    regression_results = test[
        [
            "datetime",
            TARGET,
        ]
    ].copy()

    regression_results["linear_regression"] = test_predictions

    # Merge with existing naive predictions
    naive_results = pd.read_csv(
        OUTPUT_FILE,
        parse_dates=["datetime"],
    )

    naive_results = naive_results[
        naive_results["datetime"].isin(regression_results["datetime"])
    ]

    final_results = regression_results.merge(
        naive_results[
            [
                "datetime",
                "naive_24h",
                "naive_168h",
            ]
        ],
        on="datetime",
        how="left",
    )

    final_results.to_csv(OUTPUT_FILE, index=False)

    print("Linear regression baseline created successfully.")
    print(f"Training rows: {len(train)}")
    print(f"Testing rows : {len(test)}")
    print(f"Output       : {OUTPUT_FILE}")


if __name__ == "__main__":
    create_regression_predictions()