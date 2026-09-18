import os
import requests
import pandas as pd


# ---------------------------------------------------------
# NASA POWER configuration
# ---------------------------------------------------------

LATITUDE = 28.7041
LONGITUDE = 77.1025

START_DATE = "20190101"
END_DATE = "20240430"

PARAMETER = "WS10M"

OUTPUT_FILE = "data/raw/renewable/nasa_power_wind_hourly.csv"

API_URL = (
    "https://power.larc.nasa.gov/api/temporal/hourly/point"
)


def download_wind_data():

    print("Starting NASA POWER wind data download...")

    params = {
        "parameters": PARAMETER,
        "community": "RE",
        "longitude": LONGITUDE,
        "latitude": LATITUDE,
        "start": START_DATE,
        "end": END_DATE,
        "format": "JSON",
        "time-standard": "UTC"
    }

    print("\nRequesting NASA POWER data...")
    print(f"Latitude : {LATITUDE}")
    print(f"Longitude: {LONGITUDE}")
    print(f"Parameter: {PARAMETER}")
    print(f"Period   : {START_DATE} to {END_DATE}")

    response = requests.get(
        API_URL,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    # -----------------------------------------------------
    # Extract hourly parameter data
    # -----------------------------------------------------

    parameter_data = data["properties"]["parameter"][PARAMETER]

    df = pd.DataFrame(
        list(parameter_data.items()),
        columns=["datetime", "wind_speed_10m_ms"]
    )

    # -----------------------------------------------------
    # Convert datetime
    # -----------------------------------------------------

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        format="%Y%m%d%H"
    )

    # Convert values to numeric
    df["wind_speed_10m_ms"] = pd.to_numeric(
        df["wind_speed_10m_ms"],
        errors="coerce"
    )

    # -----------------------------------------------------
    # Add location information
    # -----------------------------------------------------

    df["latitude"] = LATITUDE
    df["longitude"] = LONGITUDE

    # -----------------------------------------------------
    # Sort data
    # -----------------------------------------------------

    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Create output directory
    # -----------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    # -----------------------------------------------------
    # Save raw downloaded data
    # -----------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nWind data downloaded successfully.")

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nDate range:")
    print(df["datetime"].min())
    print(df["datetime"].max())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nOutput file:")
    print(OUTPUT_FILE)

    print("\nNASA POWER WIND DOWNLOAD: COMPLETED")


if __name__ == "__main__":
    download_wind_data()