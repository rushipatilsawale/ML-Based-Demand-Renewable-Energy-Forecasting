import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_merged_dataset.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    return df


def analyze_weather(df):
    print("=" * 60)
    print("WEATHER ANALYSIS")
    print("=" * 60)

    weather_columns = [
        "temperature_2m_c",
        "relative_humidity_pct",
        "cloud_cover_pct",
        "precipitation_mm",
        "wind_speed_10m_kmh",
        "solar_radiation_w_m2",
    ]

    print("\n--- WEATHER STATISTICS ---")
    print(df[weather_columns].describe().round(2).to_string())

    print("\n--- AVERAGE WEATHER VARIABLES ---")
    print(df[weather_columns].mean().round(2).to_string())

    print("\n--- WEATHER EXTREMES ---")

    for column in weather_columns:
        minimum = df[column].min()
        maximum = df[column].max()

        print(f"{column}")
        print(f"  Minimum : {minimum}")
        print(f"  Maximum : {maximum}")

    # Temperature
    hourly_temperature = df.groupby("hour")["temperature_2m_c"].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(hourly_temperature.index, hourly_temperature.values)
    plt.title("Average Temperature by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Temperature (°C)")
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output = FIGURES_DIR / "average_temperature_by_hour.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {output}")

    # Solar radiation
    hourly_solar = df.groupby("hour")["solar_radiation_w_m2"].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(hourly_solar.index, hourly_solar.values)
    plt.title("Average Solar Radiation by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Solar Radiation (W/m²)")
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output = FIGURES_DIR / "average_solar_radiation_by_hour.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output}")

    # Wind speed
    hourly_wind = df.groupby("hour")["wind_speed_10m_kmh"].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(hourly_wind.index, hourly_wind.values)
    plt.title("Average Wind Speed by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Wind Speed (km/h)")
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output = FIGURES_DIR / "average_wind_speed_by_hour.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output}")

    print("\n" + "=" * 60)
    print("WEATHER ANALYSIS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    df = load_data()
    analyze_weather(df)