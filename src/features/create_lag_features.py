import pandas as pd


INPUT_PATH = "data/processed/time_features.csv"
OUTPUT_PATH = "data/processed/lag_features.csv"


df = pd.read_csv(INPUT_PATH, parse_dates=["datetime"])

df = df.sort_values("datetime").reset_index(drop=True)

# Previous hour
df["demand_lag_1h"] = df["national_demand_mw"].shift(1)

# Same hour previous day
df["demand_lag_24h"] = df["national_demand_mw"].shift(24)

# Same hour previous week
df["demand_lag_168h"] = df["national_demand_mw"].shift(168)

df.to_csv(OUTPUT_PATH, index=False)

print("Lag features created successfully.")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")
print(f"Saved   : {OUTPUT_PATH}")