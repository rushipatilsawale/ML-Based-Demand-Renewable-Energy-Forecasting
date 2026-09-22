"""Live weather feed for real-time forecasting (Open-Meteo, Delhi).

Returns the same seven weather columns the models were trained on, so the
recursive engine can consume live future weather directly. Best-effort: any
network/API failure returns an empty frame and the caller falls back to
climatological normals. Open-Meteo forecasts a maximum of 16 days ahead.
"""
import pandas as pd
import requests

API_URL = "https://api.open-meteo.com/v1/forecast"
DELHI_LAT, DELHI_LON = 28.6139, 77.2090
TIMEZONE = "Asia/Kolkata"
MAX_DAYS = 16

WEATHER_COLUMNS = ["temperature_2m_c", "relative_humidity_pct", "cloud_cover_pct",
                   "precipitation_mm", "solar_radiation_w_m2", "wind_speed_10m_kmh",
                   "wind_speed_10m_ms"]


def fetch_live_weather(days=MAX_DAYS, lat=DELHI_LAT, lon=DELHI_LON, timeout=30):
    params = {
        "latitude": lat, "longitude": lon, "timezone": TIMEZONE,
        "forecast_days": min(days, MAX_DAYS),
        "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,precipitation,"
                  "shortwave_radiation,wind_speed_10m",
    }
    try:
        resp = requests.get(API_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
    except (requests.RequestException, KeyError, ValueError):
        return pd.DataFrame(columns=["datetime", *WEATHER_COLUMNS])

    df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "temperature_2m_c": hourly["temperature_2m"],
        "relative_humidity_pct": hourly["relative_humidity_2m"],
        "cloud_cover_pct": hourly["cloud_cover"],
        "precipitation_mm": hourly["precipitation"],
        "solar_radiation_w_m2": hourly["shortwave_radiation"],
        "wind_speed_10m_kmh": hourly["wind_speed_10m"],
    })
    df["wind_speed_10m_ms"] = df["wind_speed_10m_kmh"] / 3.6
    return df.dropna(subset=["datetime"]).reset_index(drop=True)


if __name__ == "__main__":
    live = fetch_live_weather()
    print(f"live weather rows: {len(live)}")
    if len(live):
        print(live.head(3).to_string(index=False))
