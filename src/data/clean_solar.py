"""Validate and standardise the project solar-potential source to hourly CSV."""

import json
from pathlib import Path

import pandas as pd


INPUT_RAW_JSON = Path("data/raw/solar_pvgis_delhi_2019_2024.json")
OUTPUT_PROCESSED_CSV = Path("data/processed/solar_hourly.csv")
START = pd.Timestamp("2019-01-01 00:00:00")
END = pd.Timestamp("2024-04-30 23:00:00")
PEAK_CAPACITY_KW = 1000.0


def clean_solar_data(input_path=INPUT_RAW_JSON, output_path=OUTPUT_PROCESSED_CSV):
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    df = pd.DataFrame({"datetime": payload["datetime"], "solar_generation_kw": payload["pv_power_kw"]})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["solar_generation_kw"] = pd.to_numeric(df["solar_generation_kw"], errors="coerce").clip(lower=0)
    # PVGIS returned ten isolated nulls in the 2024 archive.  Interpolate only
    # those short gaps and record the flag so this treatment remains visible.
    df["solar_imputed"] = df["solar_generation_kw"].isna().astype("int8")
    df["solar_generation_kw"] = df["solar_generation_kw"].interpolate(limit_direction="both")
    expected = pd.date_range(START, END, freq="h")
    if not df["datetime"].equals(pd.Series(expected, name="datetime")):
        raise ValueError("Solar timestamps are not an exact continuous hourly project window.")
    if df.isna().any().any() or df["datetime"].duplicated().any():
        raise ValueError("Solar input has missing or duplicate values.")
    if df["solar_generation_kw"].max() > PEAK_CAPACITY_KW * 1.05:
        raise ValueError("Solar generation exceeds the configured facility capacity.")
    df["solar_irradiance_proxy_w_m2"] = (df["solar_generation_kw"] / PEAK_CAPACITY_KW * 1000).round(3)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Validated and saved {len(df):,} hourly solar records to {output_path}")
    return df


if __name__ == "__main__":
    clean_solar_data()
