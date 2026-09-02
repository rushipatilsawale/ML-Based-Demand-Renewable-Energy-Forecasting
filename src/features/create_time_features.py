import pandas as pd


INPUT_PATH = "data/processed/final_merged_dataset.csv"
OUTPUT_PATH = "data/processed/time_features.csv"


df = pd.read_csv(INPUT_PATH, parse_dates=["datetime"])

df["hour_sin"] = __import__("numpy").sin(
    2 * __import__("numpy").pi * df["hour"] / 24
)

df["hour_cos"] = __import__("numpy").cos(
    2 * __import__("numpy").pi * df["hour"] / 24
)

df["month_sin"] = __import__("numpy").sin(
    2 * __import__("numpy").pi * df["month"] / 12
)

df["month_cos"] = __import__("numpy").cos(
    2 * __import__("numpy").pi * df["month"] / 12
)

df["day_of_week_sin"] = __import__("numpy").sin(
    2 * __import__("numpy").pi * df["day_of_week"] / 7
)

df["day_of_week_cos"] = __import__("numpy").cos(
    2 * __import__("numpy").pi * df["day_of_week"] / 7
)

df.to_csv(OUTPUT_PATH, index=False)

print("Time features created successfully.")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")
print(f"Saved   : {OUTPUT_PATH}")