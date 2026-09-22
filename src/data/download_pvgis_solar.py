"""Download hourly Delhi PV-generation potential from the official PVGIS 6 API.

The result is a modelled 1 MWp photovoltaic facility, not metered plant output.
It is hourly, India-focused, and shares the complete project window with the
Indian load, weather, and wind-resource inputs.
"""

import json
from pathlib import Path

import pandas as pd
import requests


API_ENDPOINT = "https://photovoltaic-geographic-information-system.ec.europa.eu/api/v6/power/broadband"
LOCATION = {"name": "Delhi, India", "latitude": 28.7041, "longitude": 77.1025}
PV_CONFIG = {"peak_power_kw": 1000.0, "system_efficiency": 0.86, "tilt_deg": 28.0, "orientation_deg": 180.0}
START = "2019-01-01 00:00:00"
END = "2024-04-30 23:00:00"
# PVGIS v6 treats end_time as an exclusive boundary for hourly values.
API_END = "2024-05-01 00:00:00"
OUTPUT_RAW_JSON = Path("data/raw/solar_pvgis_delhi_2019_2024.json")
OUTPUT_METADATA_JSON = Path("data/raw/solar_pvgis_metadata.json")


def fetch_pvgis_data():
    """Return an hourly power series in IST and auditable request metadata."""
    params = {
        "latitude": LOCATION["latitude"], "longitude": LOCATION["longitude"],
        "start_time": START, "end_time": API_END, "frequency": "Hourly",
        "timezone": "Asia/Kolkata", "irradiance_source": "ERA5",
        "surface_tilt": PV_CONFIG["tilt_deg"], "surface_orientation": PV_CONFIG["orientation_deg"],
        "peak-power": PV_CONFIG["peak_power_kw"], "system_efficiency": PV_CONFIG["system_efficiency"],
    }
    response = requests.get(API_ENDPOINT, params=params, timeout=180)
    response.raise_for_status()
    power = response.json().get("power")
    if not isinstance(power, list):
        raise ValueError("PVGIS response did not contain an hourly 'power' list.")
    index = pd.date_range(START, END, freq="h")
    if len(power) != len(index):
        raise ValueError(f"PVGIS returned {len(power)} values; expected {len(index)} hourly values.")
    return index, power, params


def main():
    index, power, params = fetch_pvgis_data()
    OUTPUT_RAW_JSON.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "source": "European Commission JRC PVGIS 6 API",
        "classification": "modelled PV generation potential for a 1 MWp Delhi facility",
        "timezone": "Asia/Kolkata",
        "datetime": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in index],
        "pv_power_kw": power,
    }
    metadata = {"source": raw["source"], "api_endpoint": API_ENDPOINT, "location": LOCATION,
                "pv_configuration": PV_CONFIG, "request_parameters": params, "start": START,
                "end": END, "frequency": "hourly", "rows": len(power),
                "classification": raw["classification"]}
    OUTPUT_RAW_JSON.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    OUTPUT_METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Downloaded {len(power):,} hourly solar records to {OUTPUT_RAW_JSON}")


if __name__ == "__main__":
    main()
