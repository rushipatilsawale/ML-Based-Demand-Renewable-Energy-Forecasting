import pandas as pd


INPUT_PATH = "data/processed/lag_features.csv"
OUTPUT_PATH = "data/processed/rolling_features.csv"


df = pd.read_csv(INPUT_PATH, parse_dates=["datetime"])

df = df.sort_values("datetime").reset_index(drop=True)

# Shift first so the current target is never used.
previous_demand = df["national_demand_mw"].shift(1)

df["demand_rolling_mean_24h"] = (
    previous_demand.rolling(window=24).mean()
)

df["demand_rolling_std_24h"] = (
    previous_demand.rolling(window=24).std()
)

df["demand_rolling_mean_168h"] = (
    previous_demand.rolling(window=168).mean()
)

df["demand_rolling_std_168h"] = (
    previous_demand.rolling(window=168).std()
)

df.to_csv(OUTPUT_PATH, index=False)

print("Rolling features created successfully.")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")
print(f"Saved   : {OUTPUT_PATH}")