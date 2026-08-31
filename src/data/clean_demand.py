from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# 1. Define project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "hourlyLoadDataIndia.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "demand_cleaned.csv"


# ---------------------------------------------------------
# 2. Load raw dataset
# ---------------------------------------------------------

print("=" * 60)
print("DEMAND DATA CLEANING")
print("=" * 60)

print(f"\nLoading:\n{RAW_FILE}")

df = pd.read_excel(RAW_FILE)


# ---------------------------------------------------------
# 3. Standardize datetime
# ---------------------------------------------------------

df["datetime"] = pd.to_datetime(df["datetime"])

df = df.sort_values("datetime").reset_index(drop=True)


# ---------------------------------------------------------
# 4. Rename columns
# ---------------------------------------------------------

column_mapping = {
    "National Hourly Demand": "national_demand_mw",
    "Northen Region Hourly Demand": "north_demand_mw",
    "Western Region Hourly Demand": "west_demand_mw",
    "Eastern Region Hourly Demand": "east_demand_mw",
    "Southern Region Hourly Demand": "south_demand_mw",
    "North-Eastern Region Hourly Demand": "north_east_demand_mw",
}

df = df.rename(columns=column_mapping)


# ---------------------------------------------------------
# 5. Add calendar features
# ---------------------------------------------------------

df["hour"] = df["datetime"].dt.hour

df["day"] = df["datetime"].dt.day

df["month"] = df["datetime"].dt.month

df["year"] = df["datetime"].dt.year

df["day_of_week"] = df["datetime"].dt.dayofweek

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)


# ---------------------------------------------------------
# 6. Final column ordering
# ---------------------------------------------------------

columns = [
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
]

df = df[columns]


# ---------------------------------------------------------
# 7. Final validation
# ---------------------------------------------------------

print("\n--- FINAL VALIDATION ---")

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

print(f"Missing values: {df.isna().sum().sum()}")

print(f"Duplicate timestamps: {df['datetime'].duplicated().sum()}")


# ---------------------------------------------------------
# 8. Save processed dataset
# ---------------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print(f"\nProcessed dataset saved to:")
print(OUTPUT_FILE)

print("\nFirst 5 rows:")
print(df.head().to_string(index=False))

print("\n" + "=" * 60)
print("DEMAND CLEANING COMPLETED")
print("=" * 60)