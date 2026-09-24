"""Generate reproducible EDA for the aligned India demand-renewable dataset."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "aligned_hourly_dataset.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
FACILITY_SCALE_KW_PER_MW = 0.06


def save_plot(frame, columns, title, ylabel, name):
    ax = frame[columns].plot(figsize=(12, 5), linewidth=1.8)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel("")
    ax.grid(alpha=.25); plt.tight_layout(); plt.savefig(FIGURES / name, dpi=160); plt.close()


def save_dual_axis_plot(frame, left_columns, right_columns, title, left_label, right_label, name):
    """Plot MW-scale demand and kW-scale renewable on separate axes.

    Demand (~1.5e5 MW) and renewable (~1e2 kW) differ by ~1000x; on a shared
    axis the renewable series is crushed against zero. Twin axes keep both
    readable so the renewable pattern is actually visible.
    """
    fig, left_ax = plt.subplots(figsize=(12, 5))
    left_ax.plot(frame.index, frame[left_columns], linewidth=1.8, color="#e4572e")
    left_ax.set_ylabel(left_label, color="#e4572e")
    left_ax.tick_params(axis="y", labelcolor="#e4572e")
    left_ax.grid(alpha=.25)
    right_ax = left_ax.twinx()
    right_ax.plot(frame.index, frame[right_columns], linewidth=1.8)
    right_ax.set_ylabel(right_label)
    lines = left_ax.get_lines() + right_ax.get_lines()
    left_ax.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=8)
    left_ax.set_title(title)
    fig.tight_layout(); plt.savefig(FIGURES / name, dpi=160); plt.close(fig)


def run_eda():
    REPORTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(INPUT, parse_dates=["datetime"]).sort_values("datetime")
    df["combined_renewable_kw"] = df["solar_generation_kw"] + df["wind_power_potential_kw"]
    df["facility_demand_kw"] = df["national_demand_mw"] * FACILITY_SCALE_KW_PER_MW
    df["energy_balance_kw"] = df["combined_renewable_kw"] - df["facility_demand_kw"]

    # Forecast/dashboard-ready aggregation records.
    hourly = df.set_index("datetime")
    daily = hourly.resample("D").mean(numeric_only=True)
    weekly = hourly.resample("W-MON", label="left", closed="left").mean(numeric_only=True)
    monthly = hourly.resample("MS").mean(numeric_only=True)
    for name, frame in {"hourly": hourly, "daily": daily, "weekly": weekly, "monthly": monthly}.items():
        frame.reset_index().to_csv(REPORTS / f"{name}_eda_records.csv", index=False)

    # Demand and renewable profiles used to inspect forecast seasonality.
    hourly_profile = df.groupby("hour")[["national_demand_mw", "solar_generation_kw", "wind_power_potential_kw", "combined_renewable_kw"]].mean()
    weekday_profile = df.groupby("day_of_week")[["national_demand_mw", "combined_renewable_kw"]].mean()
    monthly_profile = df.groupby("month")[["national_demand_mw", "combined_renewable_kw", "energy_balance_kw"]].mean()
    hourly_profile.to_csv(REPORTS / "hourly_pattern_profile.csv")
    weekday_profile.to_csv(REPORTS / "weekday_pattern_profile.csv")
    monthly_profile.to_csv(REPORTS / "monthly_pattern_profile.csv")

    variables = ["national_demand_mw", "temperature_2m_c", "relative_humidity_pct", "cloud_cover_pct", "precipitation_mm", "solar_radiation_w_m2", "wind_speed_10m_kmh", "wind_speed_10m_ms", "solar_generation_kw", "wind_power_potential_kw", "combined_renewable_kw"]
    corr = df[variables].corr(method="pearson")
    corr.to_csv(REPORTS / "correlation_matrix.csv")
    pd.DataFrame({"metric": ["rows", "missing_values", "duplicate_timestamps", "start", "end", "facility_scale_kw_per_mw", "mean_national_demand_mw", "mean_combined_renewable_kw", "mean_energy_balance_kw"],
                  "value": [len(df), int(df.isna().sum().sum()), int(df.datetime.duplicated().sum()), df.datetime.min(), df.datetime.max(), FACILITY_SCALE_KW_PER_MW, df.national_demand_mw.mean(), df.combined_renewable_kw.mean(), df.energy_balance_kw.mean()]}).to_csv(REPORTS / "eda_summary.csv", index=False)

    save_dual_axis_plot(hourly_profile, ["national_demand_mw"], ["solar_generation_kw", "wind_power_potential_kw", "combined_renewable_kw"], "Average hourly demand and renewable generation", "Demand (MW)", "Renewable (kW)", "hourly_demand_renewable_profile.png")
    save_dual_axis_plot(weekday_profile, ["national_demand_mw"], ["combined_renewable_kw"], "Average weekday demand and combined renewable supply", "Demand (MW)", "Renewable (kW)", "weekday_demand_renewable_profile.png")
    save_dual_axis_plot(monthly_profile, ["national_demand_mw"], ["combined_renewable_kw", "energy_balance_kw"], "Monthly demand, renewable supply and facility energy balance", "Demand (MW)", "Renewable / balance (kW)", "monthly_energy_patterns.png")
    save_plot(daily, ["national_demand_mw"], "Daily mean national demand", "MW", "daily_demand.png")
    save_dual_axis_plot(weekly, ["national_demand_mw"], ["combined_renewable_kw"], "Weekly mean demand and renewable supply", "Demand (MW)", "Renewable (kW)", "weekly_demand_renewable.png")
    fig, ax = plt.subplots(figsize=(11, 9)); image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(variables)), variables, rotation=75, ha="right"); ax.set_yticks(range(len(variables)), variables)
    fig.colorbar(image, ax=ax, label="Pearson correlation"); ax.set_title("Demand, weather, solar and wind correlation matrix")
    plt.tight_layout(); plt.savefig(FIGURES / "demand_weather_renewable_correlation.png", dpi=160); plt.close()
    print(f"EDA complete: {len(df):,} aligned hourly records; outputs in {REPORTS}")


if __name__ == "__main__":
    run_eda()
