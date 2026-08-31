from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# 1. Define project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEMAND_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demand_cleaned.csv"
)

WEATHER_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "weather_hourly.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_merged_dataset.csv"
)


# ---------------------------------------------------------
# 2. Load datasets
# ---------------------------------------------------------

print("=" * 60)
print("DATASET MERGING")
print("=" * 60)

print("\nLoading demand dataset...")
demand = pd.read_csv(DEMAND_FILE)

print("Loading weather dataset...")
weather = pd.read_csv(WEATHER_FILE)


# ---------------------------------------------------------
# 3. Convert datetime columns
# ---------------------------------------------------------

demand["datetime"] = pd.to_datetime(demand["datetime"])

weather["datetime"] = pd.to_datetime(weather["datetime"])


# ---------------------------------------------------------
# 4. Sort datasets
# ---------------------------------------------------------

demand = demand.sort_values("datetime").reset_index(drop=True)

weather = weather.sort_values("datetime").reset_index(drop=True)


# ---------------------------------------------------------
# 5. Validate timestamp uniqueness
# ---------------------------------------------------------

print("\n--- TIMESTAMP VALIDATION ---")

print(
    "Demand duplicate timestamps:",
    demand["datetime"].duplicated().sum()
)

print(
    "Weather duplicate timestamps:",
    weather["datetime"].duplicated().sum()
)


# ---------------------------------------------------------
# 6. Merge
# ---------------------------------------------------------

print("\nMerging datasets...")

merged = demand.merge(
    weather,
    on="datetime",
    how="left",
    validate="one_to_one",
)


# ---------------------------------------------------------
# 7. Validate merged dataset
# ---------------------------------------------------------

print("\n--- MERGED DATA VALIDATION ---")

print(f"Demand rows  : {len(demand)}")
print(f"Weather rows : {len(weather)}")
print(f"Merged rows  : {len(merged)}")

print(
    "Merged duplicate timestamps:",
    merged["datetime"].duplicated().sum()
)

print(
    "Total missing values:",
    merged.isna().sum().sum()
)


# ---------------------------------------------------------
# 8. Validate expected weather coverage
# ---------------------------------------------------------

weather_columns = [
    "temperature_2m_c",
    "relative_humidity_pct",
    "cloud_cover_pct",
    "precipitation_mm",
    "wind_speed_10m_kmh",
    "solar_radiation_w_m2",
]

weather_missing = merged[weather_columns].isna().sum().sum()

print(
    "Missing weather values:",
    weather_missing
)


# ---------------------------------------------------------
# 9. Save final dataset
# ---------------------------------------------------------

merged.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFinal dataset saved to:")

print(OUTPUT_FILE)


# ---------------------------------------------------------
# 10. Display final structure
# ---------------------------------------------------------

print("\nFinal shape:")
print(merged.shape)

print("\nFinal columns:")

for column in merged.columns:
    print(f"  - {column}")


print("\nFirst 5 rows:")

print(
    merged.head().to_string(index=False)
)


print("\n" + "=" * 60)

if (
    len(merged) == len(demand)
    and merged["datetime"].duplicated().sum() == 0
    and merged.isna().sum().sum() == 0
):
    print("MERGE VALIDATION: PASSED")
else:
    print("MERGE VALIDATION: REVIEW REQUIRED")

print("=" * 60)