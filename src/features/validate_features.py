import pandas as pd


INPUT_PATH = "data/processed/featured_dataset.csv"


df = pd.read_csv(INPUT_PATH, parse_dates=["datetime"])

print("\n===== FEATURE DATASET VALIDATION =====")

checks = {}

checks["No missing values"] = df.isna().sum().sum() == 0
checks["No duplicate timestamps"] = df["datetime"].duplicated().sum() == 0
checks["Chronological order"] = df["datetime"].is_monotonic_increasing

required_features = [
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
    "demand_rolling_std_168h"
]

checks["Required features"] = all(
    feature in df.columns for feature in required_features
)

for check, result in checks.items():
    print(f"{check:<30}: {'PASSED' if result else 'FAILED'}")

print("\nRows    :", len(df))
print("Columns :", len(df.columns))
print("Start   :", df["datetime"].min())
print("End     :", df["datetime"].max())

if all(checks.values()):
    print("\nFINAL FEATURE VALIDATION: PASSED")
else:
    print("\nFINAL FEATURE VALIDATION: FAILED")
    raise ValueError("Feature validation failed.")