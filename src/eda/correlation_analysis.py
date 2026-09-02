import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_merged_dataset.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    demand = "national_demand_mw"

    weather_columns = [
        "temperature_2m_c",
        "relative_humidity_pct",
        "cloud_cover_pct",
        "precipitation_mm",
        "wind_speed_10m_kmh",
        "solar_radiation_w_m2",
    ]

    correlation = df[[demand] + weather_columns].corr()

    print("\n--- DEMAND VS WEATHER CORRELATION ---")

    demand_correlation = correlation[demand].drop(demand).sort_values(
        ascending=False
    )

    print(demand_correlation.round(4).to_string())

    print("\n--- COMPLETE CORRELATION MATRIX ---")
    print(correlation.round(4).to_string())

    # Plot demand-weather correlations
    plt.figure(figsize=(10, 6))

    demand_correlation.plot(kind="bar")

    plt.title("National Demand vs Weather Variables")
    plt.xlabel("Weather Variable")
    plt.ylabel("Pearson Correlation")
    plt.xticks(rotation=45, ha="right")
    plt.axhline(0, linewidth=0.8)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output = FIGURES_DIR / "demand_weather_correlation.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {output}")

    # Complete matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation, aspect="auto")

    plt.colorbar(label="Correlation")

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(correlation.index)),
        correlation.index
    )

    plt.title("Demand and Weather Correlation Matrix")
    plt.tight_layout()

    output = FIGURES_DIR / "correlation_matrix.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output}")

    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()