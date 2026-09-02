import pandas as pd


INPUT_PATH = "data/processed/rolling_features.csv"
OUTPUT_PATH = "data/processed/featured_dataset.csv"


df = pd.read_csv(INPUT_PATH, parse_dates=["datetime"])

df = df.sort_values("datetime").reset_index(drop=True)

# Remove rows where lag/rolling features are unavailable.
df = df.dropna().reset_index(drop=True)

df.to_csv(OUTPUT_PATH, index=False)

print("\n===== FEATURE ENGINEERING COMPLETE =====")
print(f"Rows       : {len(df)}")
print(f"Columns    : {len(df.columns)}")
print(f"Start      : {df['datetime'].min()}")
print(f"End        : {df['datetime'].max()}")
print(f"Missing    : {df.isna().sum().sum()}")
print(f"Saved      : {OUTPUT_PATH}")