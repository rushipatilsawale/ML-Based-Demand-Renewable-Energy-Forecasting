from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# 1. Define project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "hourlyLoadDataIndia.xlsx"


# ---------------------------------------------------------
# 2. Load the raw demand dataset
# ---------------------------------------------------------

print("=" * 60)
print("DEMAND DATASET INSPECTION")
print("=" * 60)

print(f"\nLoading file:\n{RAW_FILE}")

df = pd.read_excel(RAW_FILE)


# ---------------------------------------------------------
# 3. Basic information
# ---------------------------------------------------------

print("\n--- BASIC INFORMATION ---")

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nColumns:")
for column in df.columns:
    print(f"  - {column}")


# ---------------------------------------------------------
# 4. Validate required columns
# ---------------------------------------------------------

required_columns = [
    "datetime",
    "National Hourly Demand",
    "Northen Region Hourly Demand",
    "Western Region Hourly Demand",
    "Eastern Region Hourly Demand",
    "Southern Region Hourly Demand",
    "North-Eastern Region Hourly Demand",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

print("\n--- COLUMN VALIDATION ---")

if missing_columns:
    print("ERROR: Missing required columns:")
    for column in missing_columns:
        print(f"  - {column}")
else:
    print("All required columns are present.")


# ---------------------------------------------------------
# 5. Datetime validation
# ---------------------------------------------------------

print("\n--- DATETIME VALIDATION ---")

df["datetime"] = pd.to_datetime(df["datetime"])

print(f"Start time : {df['datetime'].min()}")
print(f"End time   : {df['datetime'].max()}")

duplicate_count = df["datetime"].duplicated().sum()

print(f"Duplicate timestamps : {duplicate_count}")


# ---------------------------------------------------------
# 6. Missing-value check
# ---------------------------------------------------------

print("\n--- MISSING VALUE CHECK ---")

missing_values = df.isna().sum()

print(missing_values)

total_missing = missing_values.sum()

print(f"\nTotal missing values: {total_missing}")


# ---------------------------------------------------------
# 7. Hourly continuity check
# ---------------------------------------------------------

print("\n--- TIME FREQUENCY CHECK ---")

time_difference = df["datetime"].sort_values().diff().dropna()

expected_interval = pd.Timedelta(hours=1)

incorrect_intervals = (time_difference != expected_interval).sum()

print(f"Expected interval : {expected_interval}")
print(f"Incorrect intervals: {incorrect_intervals}")


# ---------------------------------------------------------
# 8. Demand statistics
# ---------------------------------------------------------

print("\n--- NATIONAL DEMAND STATISTICS ---")

print(
    df["National Hourly Demand"]
    .describe()
    .round(2)
)


# ---------------------------------------------------------
# 9. Final validation result
# ---------------------------------------------------------

print("\n" + "=" * 60)

if (
    not missing_columns
    and total_missing == 0
    and duplicate_count == 0
    and incorrect_intervals == 0
):
    print("VALIDATION RESULT: PASSED")
else:
    print("VALIDATION RESULT: REVIEW REQUIRED")

print("=" * 60)