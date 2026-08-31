from pathlib import Path

import requests
import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "weather_hourly.csv"
)


# ---------------------------------------------------------
# Dataset time range
# ---------------------------------------------------------

START_DATE = "2019-01-01"
END_DATE = "2024-04-30"


# ---------------------------------------------------------
# Representative location
# ---------------------------------------------------------

LATITUDE = 28.6139
LONGITUDE = 77.2090


# ---------------------------------------------------------
# Open-Meteo API
# ---------------------------------------------------------

URL = "https://archive-api.open-meteo.com/v1/archive"

PARAMS = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": ",".join(
        [
            "temperature_2m",
            "relative_humidity_2m",
            "cloud_cover",
            "precipitation",
            "wind_speed_10m",
            "shortwave_radiation",
        ]
    ),
    "timezone": "Asia/Kolkata",
}


# ---------------------------------------------------------
# Request data
# ---------------------------------------------------------

print("=" * 60)
print("WEATHER DATA ACQUISITION")
print("=" * 60)

print("\nRequesting hourly weather data...")
print(f"Location: {LATITUDE}, {LONGITUDE}")
print(f"Period: {START_DATE} → {END_DATE}")

response = requests.get(URL, params=PARAMS, timeout=60)

response.raise_for_status()

data = response.json()


# ---------------------------------------------------------
# Convert API response to DataFrame
# ---------------------------------------------------------

weather = pd.DataFrame(data["hourly"])


# ---------------------------------------------------------
# Standardize timestamp
# ---------------------------------------------------------

weather["datetime"] = pd.to_datetime(weather["time"])

weather = weather.drop(columns=["time"])

weather = weather.sort_values("datetime").reset_index(drop=True)


# ---------------------------------------------------------
# Rename columns
# ---------------------------------------------------------

weather = weather.rename(
    columns={
        "temperature_2m": "temperature_2m_c",
        "relative_humidity_2m": "relative_humidity_pct",
        "cloud_cover": "cloud_cover_pct",
        "precipitation": "precipitation_mm",
        "wind_speed_10m": "wind_speed_10m_kmh",
        "shortwave_radiation": "solar_radiation_w_m2",
    }
)


# ---------------------------------------------------------
# Validate
# ---------------------------------------------------------

print("\n--- WEATHER DATA VALIDATION ---")

print(f"Rows: {len(weather)}")

print(f"Start: {weather['datetime'].min()}")

print(f"End: {weather['datetime'].max()}")

print(
    f"Duplicate timestamps: "
    f"{weather['datetime'].duplicated().sum()}"
)

print(
    f"Missing values: "
    f"{weather.isna().sum().sum()}"
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

weather.to_csv(OUTPUT_FILE, index=False)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nFirst 5 rows:")
print(weather.head().to_string(index=False))

print("\n" + "=" * 60)
print("WEATHER DATA ACQUISITION COMPLETED")
print("=" * 60)