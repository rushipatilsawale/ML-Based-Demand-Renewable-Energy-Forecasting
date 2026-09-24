"""Create the one authoritative, timestamp-aligned hourly project dataset."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
START, END = pd.Timestamp("2019-01-01 00:00:00"), pd.Timestamp("2024-04-30 23:00:00")


def _read(path):
    frame = pd.read_csv(path)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    return frame.sort_values("datetime").drop_duplicates("datetime")


def build_aligned_dataset():
    """Inner-join the four retained sources and fail fast on alignment defects."""
    # demand_cleaned is a reproducible preprocessing artifact already derived
    # from the retained XLSX source; it avoids adding an Excel-only runtime.
    demand = _read(OUT / "demand_cleaned.csv")
    demand = demand[[c for c in demand.columns if c == "datetime" or c.endswith("_demand_mw")]]
    weather = _read(RAW / "weather_hourly.csv")
    wind = _read(RAW / "renewable" / "nasa_power_wind_hourly.csv")[["datetime", "wind_speed_10m_ms"]]
    solar = _read(OUT / "solar_hourly.csv")

    expected = pd.date_range(START, END, freq="h")
    for name, frame in {"demand": demand, "weather": weather, "wind": wind, "solar": solar}.items():
        frame = frame[(frame.datetime >= START) & (frame.datetime <= END)]
        if not frame.datetime.reset_index(drop=True).equals(pd.Series(expected, name="datetime")):
            raise ValueError(f"{name} does not provide one continuous hourly record for the common project window.")

    merged = demand.merge(weather, on="datetime", validate="one_to_one")
    merged = merged.merge(wind, on="datetime", validate="one_to_one")
    merged = merged.merge(solar, on="datetime", validate="one_to_one")
    merged = merged[(merged.datetime >= START) & (merged.datetime <= END)].copy()
    if len(merged) != len(expected) or merged.isna().any().any() or merged.datetime.duplicated().any():
        raise ValueError("Aligned dataset failed row-count, null, or duplicate validation.")

    merged["hour"] = merged.datetime.dt.hour
    merged["day_of_week"] = merged.datetime.dt.dayofweek
    merged["month"] = merged.datetime.dt.month
    merged["year"] = merged.datetime.dt.year
    merged["is_weekend"] = (merged.day_of_week >= 5).astype("int8")
    merged["wind_power_potential_kw"] = wind_power_curve(merged["wind_speed_10m_ms"])
    OUT.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT / "aligned_hourly_dataset.csv", index=False)
    pd.DataFrame([{
        "common_start": START, "common_end": END, "hourly_rows": len(merged),
        "missing_values": int(merged.isna().sum().sum()), "duplicate_timestamps": int(merged.datetime.duplicated().sum()),
        "solar_source": "PVGIS 6 modelled 1 MWp Delhi potential", "wind_source": "NASA POWER wind resource converted by 1 MW turbine curve",
    }]).to_csv(OUT / "data_quality_summary.csv", index=False)
    return merged


def wind_power_curve(speed_10m):
    """1 MW reference turbine at 80m hub height (Hellmann power law alpha=0.143).
    Cut-in: 3 m/s, rated: 12 m/s, cut-out: 25 m/s.
    """
    speed_80m = pd.Series(speed_10m, dtype=float) * (8.0 ** 0.143)
    s = speed_80m
    return pd.Series(0.0, index=s.index).mask((s >= 3.0) & (s < 12.0), 1000.0 * ((s - 3.0) / 9.0) ** 3).mask((s >= 12.0) & (s < 25.0), 1000.0)


if __name__ == "__main__":
    print(f"Created aligned dataset with {len(build_aligned_dataset()):,} hourly records.")
