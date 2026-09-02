import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_merged_dataset.csv"
)

FIGURES_PATH = PROJECT_ROOT / "reports" / "figures"

FIGURES_PATH.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():
    """Load the final merged dataset."""

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["datetime"]
    )

    return df


# ---------------------------------------------------------
# HOURLY DEMAND ANALYSIS
# ---------------------------------------------------------

def analyze_hourly_demand(df):

    hourly = (
        df.groupby("hour")["national_demand_mw"]
        .mean()
    )

    print("\n--- AVERAGE DEMAND BY HOUR ---")

    print(
        hourly.round(2).to_string()
    )

    peak_hour = hourly.idxmax()
    peak_value = hourly.max()

    lowest_hour = hourly.idxmin()
    lowest_value = hourly.min()

    print(
        f"\nPeak average demand hour   : "
        f"{peak_hour}:00 ({peak_value:.2f} MW)"
    )

    print(
        f"Lowest average demand hour : "
        f"{lowest_hour}:00 ({lowest_value:.2f} MW)"
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        hourly.index,
        hourly.values,
        marker="o"
    )

    plt.title("Average National Electricity Demand by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Demand (MW)")
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output = FIGURES_PATH / "average_demand_by_hour.png"

    plt.savefig(output, dpi=150)

    plt.close()

    print(f"\nSaved: {output}")


# ---------------------------------------------------------
# DAY OF WEEK ANALYSIS
# ---------------------------------------------------------

def analyze_day_of_week(df):

    daily = (
        df.groupby("day_of_week")["national_demand_mw"]
        .mean()
    )

    day_names = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday"
    }

    daily.index = daily.index.map(day_names)

    print("\n--- AVERAGE DEMAND BY DAY OF WEEK ---")

    print(
        daily.round(2).to_string()
    )

    plt.figure(figsize=(10, 5))

    plt.bar(
        daily.index,
        daily.values
    )

    plt.title("Average National Electricity Demand by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Average Demand (MW)")
    plt.xticks(rotation=30)

    plt.tight_layout()

    output = FIGURES_PATH / "average_demand_by_day.png"

    plt.savefig(output, dpi=150)

    plt.close()

    print(f"\nSaved: {output}")


# ---------------------------------------------------------
# WEEKDAY VS WEEKEND
# ---------------------------------------------------------

def analyze_weekday_weekend(df):

    comparison = (
        df.groupby("is_weekend")["national_demand_mw"]
        .mean()
    )

    weekday = comparison.get(0)
    weekend = comparison.get(1)

    print("\n--- WEEKDAY VS WEEKEND ---")

    print(f"Weekday average : {weekday:.2f} MW")
    print(f"Weekend average : {weekend:.2f} MW")

    difference = weekend - weekday

    print(
        f"Difference      : {difference:.2f} MW"
    )


# ---------------------------------------------------------
# MONTHLY DEMAND ANALYSIS
# ---------------------------------------------------------

def analyze_monthly_demand(df):

    monthly = (
        df.groupby("month")["national_demand_mw"]
        .mean()
    )

    print("\n--- AVERAGE DEMAND BY MONTH ---")

    print(
        monthly.round(2).to_string()
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        monthly.index,
        monthly.values,
        marker="o"
    )

    plt.title("Average National Electricity Demand by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Demand (MW)")
    plt.xticks(range(1, 13))
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output = FIGURES_PATH / "average_demand_by_month.png"

    plt.savefig(output, dpi=150)

    plt.close()

    print(f"\nSaved: {output}")


# ---------------------------------------------------------
# YEARLY TREND
# ---------------------------------------------------------

def analyze_yearly_demand(df):

    yearly = (
        df.groupby("year")["national_demand_mw"]
        .mean()
    )

    print("\n--- AVERAGE DEMAND BY YEAR ---")

    print(
        yearly.round(2).to_string()
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        yearly.index,
        yearly.values,
        marker="o"
    )

    plt.title("Average National Electricity Demand by Year")
    plt.xlabel("Year")
    plt.ylabel("Average Demand (MW)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output = FIGURES_PATH / "average_demand_by_year.png"

    plt.savefig(output, dpi=150)

    plt.close()

    print(f"\nSaved: {output}")


# ---------------------------------------------------------
# REGIONAL DEMAND
# ---------------------------------------------------------

def analyze_regional_demand(df):

    regions = {
        "North": "north_demand_mw",
        "West": "west_demand_mw",
        "East": "east_demand_mw",
        "South": "south_demand_mw",
        "North-East": "north_east_demand_mw"
    }

    regional_means = {}

    for region, column in regions.items():

        regional_means[region] = df[column].mean()

    regional = pd.Series(regional_means)

    print("\n--- AVERAGE REGIONAL DEMAND ---")

    print(
        regional.round(2).to_string()
    )

    plt.figure(figsize=(10, 5))

    plt.bar(
        regional.index,
        regional.values
    )

    plt.title("Average Electricity Demand by Region")
    plt.xlabel("Region")
    plt.ylabel("Average Demand (MW)")

    plt.tight_layout()

    output = FIGURES_PATH / "average_regional_demand.png"

    plt.savefig(output, dpi=150)

    plt.close()

    print(f"\nSaved: {output}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("DEMAND PATTERN ANALYSIS")
    print("=" * 60)

    df = load_data()

    analyze_hourly_demand(df)
    analyze_day_of_week(df)
    analyze_weekday_weekend(df)
    analyze_monthly_demand(df)
    analyze_yearly_demand(df)
    analyze_regional_demand(df)

    print("\n" + "=" * 60)
    print("DEMAND ANALYSIS COMPLETED")
    print("=" * 60)