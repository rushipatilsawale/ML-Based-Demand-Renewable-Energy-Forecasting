"""Advanced pattern & weather-factor EDA for the aligned India demand-renewable dataset.

This is a dedicated analysis layer on top of ``run_aligned_eda.py``. It exists to
answer the questions that matter for feature engineering and model training:

* How does demand move hour-by-hour, and how does renewable generation move with it?
  (Demand is in MW, renewable in kW -- a ~1000x scale gap. Every demand-vs-renewable
  plot here uses twin axes so the renewable curve is actually visible instead of
  being crushed against zero.)
* How do weekday vs weekend profiles differ?
* How does demand and renewable behave across the Indian meteorological seasons
  (winter / summer / monsoon / post-monsoon)?
* Do Indian festival days shift demand away from their month's normal baseline?
* How do weather factors -- humidity, temperature, cloud cover, wind speed --
  drive BOTH demand and renewable generation?

Outputs: figure gallery in ``reports/figures/patterns/`` and statistical tables in
``reports/patterns/``. Read-only over the aligned dataset; nothing here mutates inputs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Season mapping and the Indian festival calendar come from the same shared module
# the feature pipeline uses, so EDA insights and model features never drift apart.
from src.features.calendar_features import (  # noqa: E402
    FIXED_HOLIDAY_MONTH_DAY,
    LUNAR_FESTIVALS,
    SEASON_BY_MONTH,
    SEASONS as SEASON_ORDER,
)

INPUT = ROOT / "data" / "processed" / "aligned_hourly_dataset.csv"
REPORTS = ROOT / "reports"
TABLES = REPORTS / "patterns"
FIGURES = REPORTS / "figures" / "patterns"

DEMAND_COLOR = "#e4572e"
SOLAR_COLOR = "#f2a71b"
WIND_COLOR = "#1f77b4"
COMBINED_COLOR = "#2ca02c"


def _twin_axis(x, left_y, right_y, title, left_label, right_label, name,
               left_color=DEMAND_COLOR, right_colors=None, xticks=None, xtick_labels=None):
    """Render a demand (left, MW) vs renewable (right, kW) chart on twin axes."""
    fig, left_ax = plt.subplots(figsize=(12, 5))
    left_ax.plot(x, left_y, color=left_color, linewidth=2.0, label=left_label)
    left_ax.set_ylabel(left_label, color=left_color)
    left_ax.tick_params(axis="y", labelcolor=left_color)
    left_ax.grid(alpha=.25)
    right_ax = left_ax.twinx()
    right_colors = right_colors or [COMBINED_COLOR] * len(right_y.columns)
    for column, color in zip(right_y.columns, right_colors):
        right_ax.plot(x, right_y[column], color=color, linewidth=1.8, linestyle="--", label=column)
    right_ax.set_ylabel(right_label)
    if xticks is not None:
        left_ax.set_xticks(xticks)
    if xtick_labels is not None:
        left_ax.set_xticks(range(len(xtick_labels)))
        left_ax.set_xticklabels(xtick_labels)
    lines = left_ax.get_lines() + right_ax.get_lines()
    left_ax.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=8)
    left_ax.set_title(title)
    fig.tight_layout()
    plt.savefig(FIGURES / name, dpi=160)
    plt.close(fig)


def _bars(frame, value_col, title, ylabel, name, color=DEMAND_COLOR, rotate=0):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(frame)), frame[value_col], color=color)
    ax.set_xticks(range(len(frame)))
    ax.set_xticklabels(frame.index, rotation=rotate, ha="right" if rotate else "center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=.25, axis="y")
    fig.tight_layout()
    plt.savefig(FIGURES / name, dpi=160)
    plt.close(fig)


def _heatmap(matrix, title, name, xlabel, ylabel, cmap="YlOrRd"):
    fig, ax = plt.subplots(figsize=(13, 6))
    image = ax.imshow(matrix.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(matrix.columns)), [int(c) for c in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    plt.savefig(FIGURES / name, dpi=160)
    plt.close(fig)


def load_dataset():
    df = pd.read_csv(INPUT, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date
    df["combined_renewable_kw"] = df["solar_generation_kw"] + df["wind_power_potential_kw"]
    df["season"] = df["month"].map(SEASON_BY_MONTH)
    df["day_type"] = np.where(df["is_weekend"] == 1, "Weekend", "Weekday")
    return df


def hourly_analysis(df):
    profile = df.groupby("hour").agg(
        national_demand_mw=("national_demand_mw", "mean"),
        solar_generation_kw=("solar_generation_kw", "mean"),
        wind_power_potential_kw=("wind_power_potential_kw", "mean"),
        combined_renewable_kw=("combined_renewable_kw", "mean"),
        demand_std=("national_demand_mw", "std"),
        renewable_std=("combined_renewable_kw", "std"),
    )
    profile.round(3).to_csv(TABLES / "hourly_profile.csv")
    renewable = profile[["solar_generation_kw", "wind_power_potential_kw", "combined_renewable_kw"]]
    _twin_axis(
        profile.index, profile["national_demand_mw"], renewable,
        "Average hourly demand vs renewable generation (all hours)",
        "Demand (MW)", "Renewable (kW)", "avg_hourly_demand_renewable.png",
        right_colors=[SOLAR_COLOR, WIND_COLOR, COMBINED_COLOR],
        xticks=range(0, 24, 2),
    )
    return profile


def weekday_weekend_analysis(df):
    hourly = df.groupby(["day_type", "hour"]).agg(
        national_demand_mw=("national_demand_mw", "mean"),
        combined_renewable_kw=("combined_renewable_kw", "mean"),
    ).reset_index()
    pivot_demand = hourly.pivot(index="hour", columns="day_type", values="national_demand_mw")
    pivot_renew = hourly.pivot(index="hour", columns="day_type", values="combined_renewable_kw")
    hourly.round(3).to_csv(TABLES / "weekday_weekend_hourly.csv", index=False)

    day_summary = df.groupby("day_type")["national_demand_mw"].agg(["mean", "std", "min", "max"]).round(2)
    day_summary.to_csv(TABLES / "weekday_weekend_summary.csv")

    fig, ax = plt.subplots(figsize=(12, 5))
    for column in pivot_demand.columns:
        ax.plot(pivot_demand.index, pivot_demand[column], linewidth=2.0, label=f"{column} demand")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Demand (MW)")
    ax.set_title("Hourly demand: weekday vs weekend"); ax.set_xticks(range(0, 24, 2))
    ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); plt.savefig(FIGURES / "weekday_vs_weekend_demand.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for column in pivot_renew.columns:
        ax.plot(pivot_renew.index, pivot_renew[column], linewidth=2.0, label=f"{column} renewable")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Renewable (kW)")
    ax.set_title("Hourly renewable generation: weekday vs weekend"); ax.set_xticks(range(0, 24, 2))
    ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); plt.savefig(FIGURES / "weekday_vs_weekend_renewable.png", dpi=160); plt.close(fig)
    return hourly


def season_analysis(df):
    season = df.groupby("season").agg(
        national_demand_mw=("national_demand_mw", "mean"),
        solar_generation_kw=("solar_generation_kw", "mean"),
        wind_power_potential_kw=("wind_power_potential_kw", "mean"),
        combined_renewable_kw=("combined_renewable_kw", "mean"),
        temperature_2m_c=("temperature_2m_c", "mean"),
        relative_humidity_pct=("relative_humidity_pct", "mean"),
        cloud_cover_pct=("cloud_cover_pct", "mean"),
        precipitation_mm=("precipitation_mm", "mean"),
    ).reindex(SEASON_ORDER)
    season.round(3).to_csv(TABLES / "season_profile.csv")
    renewable = season[["solar_generation_kw", "wind_power_potential_kw", "combined_renewable_kw"]]
    _twin_axis(
        range(len(season)), season["national_demand_mw"].values, renewable,
        "Indian seasonal demand vs renewable generation",
        "Demand (MW)", "Renewable (kW)", "season_demand_renewable.png",
        right_colors=[SOLAR_COLOR, WIND_COLOR, COMBINED_COLOR],
        xtick_labels=list(season.index),
    )
    return season


def festival_analysis(df):
    festival_dates = {}
    for name, dates in LUNAR_FESTIVALS.items():
        for value in dates:
            festival_dates[pd.Timestamp(value).date()] = name
    years = sorted(df["year"].unique())
    for name, (month, day) in FIXED_HOLIDAY_MONTH_DAY.items():
        for year in years:
            festival_dates[pd.Timestamp(year, month, day).date()] = name

    daily = df.groupby("date").agg(
        national_demand_mw=("national_demand_mw", "mean"),
        month=("month", "first"),
    )
    daily["festival"] = pd.Series({d: festival_dates.get(d) for d in daily.index})
    baseline = daily[daily["festival"].isna()].groupby("month")["national_demand_mw"].mean()
    festival_days = daily.dropna(subset=["festival"]).copy()
    festival_days["baseline_mw"] = festival_days["month"].map(baseline)
    festival_days["deviation_pct"] = (
        (festival_days["national_demand_mw"] - festival_days["baseline_mw"])
        / festival_days["baseline_mw"] * 100
    )
    festival_days.round(3).to_csv(TABLES / "festival_daily_demand.csv")

    by_festival = festival_days.groupby("festival")["deviation_pct"].agg(["mean", "count"]).round(2)
    by_festival = by_festival[by_festival["count"] >= 1].sort_values("mean")
    by_festival.to_csv(TABLES / "festival_demand_effect.csv")

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#c0392b" if v < 0 else "#2ca02c" for v in by_festival["mean"]]
    ax.barh(by_festival.index, by_festival["mean"], color=colors)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_xlabel("Demand deviation from same-month non-festival baseline (%)")
    ax.set_title("Indian festival effect on national electricity demand")
    ax.grid(alpha=.25, axis="x")
    fig.tight_layout(); plt.savefig(FIGURES / "festival_demand_effect.png", dpi=160); plt.close(fig)
    return by_festival


def _band_table(df, bin_col, labels, value_cols):
    agg = {col: (col, "mean") for col in value_cols}
    agg["observations"] = (bin_col, "size")
    grouped = df.groupby(bin_col, observed=False).agg(**agg).reindex(labels).round(3)
    return grouped


def weather_demand_analysis(df):
    humidity_bins = pd.cut(df["relative_humidity_pct"], [0, 20, 40, 60, 80, 100],
                           labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])
    temp_bins = pd.cut(df["temperature_2m_c"], [-np.inf, 10, 15, 20, 25, 30, 35, 40, np.inf],
                       labels=["<10C", "10-15C", "15-20C", "20-25C", "25-30C", "30-35C", "35-40C", ">40C"])
    df = df.assign(humidity_band=humidity_bins, temp_band=temp_bins)

    humidity = _band_table(df, "humidity_band", ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
                           ["national_demand_mw", "combined_renewable_kw", "solar_generation_kw"])
    humidity.to_csv(TABLES / "weather_humidity_bands.csv")
    temperature = _band_table(df, "temp_band",
                              ["<10C", "10-15C", "15-20C", "20-25C", "25-30C", "30-35C", "35-40C", ">40C"],
                              ["national_demand_mw", "combined_renewable_kw"])
    temperature.to_csv(TABLES / "weather_temperature_bands.csv")

    _bars(humidity, "national_demand_mw", "Mean demand by humidity band", "Demand (MW)",
          "humidity_band_demand.png", rotate=0)
    _bars(temperature, "national_demand_mw", "Mean demand by temperature band", "Demand (MW)",
          "temperature_band_demand.png", rotate=30)
    return humidity, temperature


def weather_renewable_analysis(df):
    cloud_bins = pd.cut(df["cloud_cover_pct"], [-0.01, 20, 40, 60, 80, 100],
                        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])
    wind_bins = pd.cut(df["wind_speed_10m_ms"], [-0.01, 3, 6, 9, 12, 25, np.inf],
                       labels=["<3 (cut-in)", "3-6", "6-9", "9-12", "12-25 (rated)", ">25 (cut-out)"])
    df = df.assign(cloud_band=cloud_bins, wind_band=wind_bins)

    cloud = _band_table(df, "cloud_band", ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
                        ["solar_generation_kw", "solar_radiation_w_m2"])
    cloud.to_csv(TABLES / "weather_cloud_solar.csv")
    wind = _band_table(df, "wind_band",
                       ["<3 (cut-in)", "3-6", "6-9", "9-12", "12-25 (rated)", ">25 (cut-out)"],
                       ["wind_power_potential_kw", "wind_speed_10m_ms"])
    wind.to_csv(TABLES / "weather_wind_power.csv")

    _bars(cloud, "solar_generation_kw", "Mean solar generation by cloud-cover band",
          "Solar (kW)", "cloud_band_solar.png", color=SOLAR_COLOR)
    _bars(wind, "wind_power_potential_kw", "Mean wind power by wind-speed band (1 MW turbine curve)",
          "Wind power (kW)", "wind_band_power.png", color=WIND_COLOR, rotate=20)
    return cloud, wind


def month_hour_matrices(df):
    demand_mh = df.pivot_table(index="month", columns="hour", values="national_demand_mw", aggfunc="mean").round(2)
    solar_mh = df.pivot_table(index="month", columns="hour", values="solar_generation_kw", aggfunc="mean").round(2)
    demand_mh.to_csv(TABLES / "month_hour_demand_matrix.csv")
    solar_mh.to_csv(TABLES / "month_hour_solar_matrix.csv")
    _heatmap(demand_mh, "Mean demand by month and hour (MW)", "month_hour_demand_heatmap.png",
             "Hour of day", "Month")
    _heatmap(solar_mh, "Mean solar generation by month and hour (kW)", "month_hour_solar_heatmap.png",
             "Hour of day", "Month", cmap="YlGnBu")
    return demand_mh, solar_mh


def regional_demand_analysis(df):
    regions = ["national_demand_mw", "north_demand_mw", "west_demand_mw",
               "east_demand_mw", "south_demand_mw", "north_east_demand_mw"]
    summary = df[regions].agg(["mean", "std", "min", "max"]).T.round(2)
    summary["share_pct"] = (summary["mean"] / summary.loc["national_demand_mw", "mean"] * 100).round(2)
    summary.to_csv(TABLES / "regional_demand_summary.csv")

    hourly = df.groupby("hour")[regions].mean().round(2)
    hourly.to_csv(TABLES / "regional_demand_hourly.csv")

    fig, ax = plt.subplots(figsize=(12, 5))
    for column in regions:
        linewidth = 2.6 if column == "national_demand_mw" else 1.5
        ax.plot(hourly.index, hourly[column], linewidth=linewidth, label=column.replace("_demand_mw", ""))
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Demand (MW)")
    ax.set_title("Average hourly demand by region"); ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout(); plt.savefig(FIGURES / "regional_demand_hourly.png", dpi=160); plt.close(fig)
    return summary


def yearly_trend_analysis(df):
    yearly = df.groupby("year").agg(
        national_demand_mw=("national_demand_mw", "mean"),
        solar_generation_kw=("solar_generation_kw", "mean"),
        wind_power_potential_kw=("wind_power_potential_kw", "mean"),
        combined_renewable_kw=("combined_renewable_kw", "mean"),
        temperature_2m_c=("temperature_2m_c", "mean"),
    ).round(3)
    yearly.to_csv(TABLES / "yearly_trend.csv")
    _twin_axis(
        yearly.index.values, yearly["national_demand_mw"].values,
        yearly[["solar_generation_kw", "wind_power_potential_kw", "combined_renewable_kw"]],
        "Yearly trend: demand vs renewable generation",
        "Demand (MW)", "Renewable (kW)", "yearly_trend.png",
        right_colors=[SOLAR_COLOR, WIND_COLOR, COMBINED_COLOR],
        xtick_labels=[str(y) for y in yearly.index],
    )
    return yearly


def summary_statistics(df):
    metrics = ["national_demand_mw", "solar_generation_kw", "wind_power_potential_kw",
               "combined_renewable_kw", "temperature_2m_c", "relative_humidity_pct",
               "cloud_cover_pct", "wind_speed_10m_ms"]
    rows = []
    for metric in metrics:
        series = df[metric]
        rows.append({
            "metric": metric, "mean": series.mean(), "std": series.std(),
            "min": series.min(), "p25": series.quantile(.25), "median": series.median(),
            "p75": series.quantile(.75), "max": series.max(),
            "pct_hours_zero": float((series == 0).mean() * 100),
        })
    summary = pd.DataFrame(rows).round(3)
    summary.to_csv(TABLES / "statistical_summary.csv", index=False)
    return summary


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    hourly_analysis(df)
    weekday_weekend_analysis(df)
    regional_demand_analysis(df)
    yearly_trend_analysis(df)
    season_analysis(df)
    festival_analysis(df)
    weather_demand_analysis(df)
    weather_renewable_analysis(df)
    month_hour_matrices(df)
    summary = summary_statistics(df)
    print(f"Advanced EDA complete: {len(df):,} hourly records.")
    print(f"  Figures -> {FIGURES}")
    print(f"  Tables  -> {TABLES}")
    print("Renewable sanity (mean kW): "
          f"solar={df['solar_generation_kw'].mean():.1f}, "
          f"wind={df['wind_power_potential_kw'].mean():.1f}, "
          f"combined={df['combined_renewable_kw'].mean():.1f}")


if __name__ == "__main__":
    main()
