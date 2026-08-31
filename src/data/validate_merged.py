from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_merged_dataset.csv"
)


# ---------------------------------------------------------
# Expected structure
# ---------------------------------------------------------

EXPECTED_COLUMNS = [
    "datetime",
    "national_demand_mw",
    "north_demand_mw",
    "west_demand_mw",
    "east_demand_mw",
    "south_demand_mw",
    "north_east_demand_mw",
    "hour",
    "day",
    "month",
    "year",
    "day_of_week",
    "is_weekend",
    "temperature_2m_c",
    "relative_humidity_pct",
    "cloud_cover_pct",
    "precipitation_mm",
    "wind_speed_10m_kmh",
    "solar_radiation_w_m2",
]


# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

print("=" * 60)
print("FINAL MERGED DATASET VALIDATION")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

df["datetime"] = pd.to_datetime(df["datetime"])


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

checks = []

# Row count
checks.append(
    ("Expected row count", len(df) == 46728)
)

# Columns
checks.append(
    ("Expected columns", list(df.columns) == EXPECTED_COLUMNS)
)

# Missing values
checks.append(
    ("No missing values", df.isna().sum().sum() == 0)
)

# Duplicate timestamps
checks.append(
    (
        "No duplicate timestamps",
        df["datetime"].duplicated().sum() == 0,
    )
)

# Sorted timestamps
checks.append(
    (
        "Chronological order",
        df["datetime"].is_monotonic_increasing,
    )
)

# Expected start
checks.append(
    (
        "Correct start date",
        df["datetime"].min()
        == pd.Timestamp("2019-01-01 00:00:00"),
    )
)

# Expected end
checks.append(
    (
        "Correct end date",
        df["datetime"].max()
        == pd.Timestamp("2024-04-30 23:00:00"),
    )
)

# Hourly continuity
time_diff = df["datetime"].diff().dropna()

checks.append(
    (
        "Hourly continuity",
        (time_diff == pd.Timedelta(hours=1)).all(),
    )
)


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print("\n--- VALIDATION CHECKS ---")

all_passed = True

for name, passed in checks:

    status = "PASSED" if passed else "FAILED"

    print(f"{name:<30}: {status}")

    if not passed:
        all_passed = False


# ---------------------------------------------------------
# Dataset information
# ---------------------------------------------------------

print("\n--- DATASET INFORMATION ---")

print(f"Rows       : {len(df)}")
print(f"Columns    : {len(df.columns)}")
print(f"Start      : {df['datetime'].min()}")
print(f"End        : {df['datetime'].max()}")


# ---------------------------------------------------------
# Final result
# ---------------------------------------------------------

print("\n" + "=" * 60)

if all_passed:
    print("FINAL VALIDATION: PASSED")
else:
    print("FINAL VALIDATION: FAILED")

print("=" * 60)