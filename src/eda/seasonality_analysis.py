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
    print("SEASONALITY AND TREND ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])

    df["date"] = df["datetime"].dt.date

    # Daily average demand
    daily_demand = (
        df.groupby("date")["national_demand_mw"]
        .mean()
    )

    print("\n--- DAILY DEMAND TREND ---")
    print(f"Minimum daily average : {daily_demand.min():.2f} MW")
    print(f"Maximum daily average : {daily_demand.max():.2f} MW")
    print(f"Mean daily average    : {daily_demand.mean():.2f} MW")

    plt.figure(figsize=(14, 5))
    plt.plot(
        pd.to_datetime(daily_demand.index),
        daily_demand.values
    )

    plt.title("Daily Average National Electricity Demand")
    plt.xlabel("Date")
    plt.ylabel("Demand (MW)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output = FIGURES_DIR / "daily_demand_trend.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {output}")

    # Monthly average demand
    monthly = (
        df.groupby(["year", "month"])["national_demand_mw"]
        .mean()
        .reset_index()
    )

    monthly["period"] = pd.to_datetime(
        monthly["year"].astype(str)
        + "-"
        + monthly["month"].astype(str)
        + "-01"
    )

    plt.figure(figsize=(14, 5))
    plt.plot(
        monthly["period"],
        monthly["national_demand_mw"]
    )

    plt.title("Monthly Average National Electricity Demand")
    plt.xlabel("Date")
    plt.ylabel("Demand (MW)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output = FIGURES_DIR / "monthly_demand_trend.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output}")

    # Seasonal demand
    seasonal = (
        df.groupby("month")["national_demand_mw"]
        .mean()
    )

    print("\n--- SEASONAL DEMAND ---")
    print(seasonal.round(2).to_string())

    print("\n" + "=" * 60)
    print("SEASONALITY AND TREND ANALYSIS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()