"""Merge baseline and ML metrics into one ranking per target.

All models -- seasonal-naive benchmarks, linear regression, random forest, gradient
boosting and XGBoost -- are compared on the SAME chronological test split using MAE,
RMSE and MAPE. RMSE is the primary ranking metric (it penalises large errors most).

Outputs:
    reports/performance_comparison.csv        (ranked within each target by RMSE)
    reports/figures/performance_comparison.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.models.modeling_common import REPORTS, TARGETS, save_figure

BASELINE_METRICS = REPORTS / "baseline_metrics.csv"
ML_METRICS = REPORTS / "ml_model_metrics.csv"


def build_comparison():
    baseline = pd.read_csv(BASELINE_METRICS).assign(category="baseline")
    ml = pd.read_csv(ML_METRICS).assign(category="ml")
    combined = pd.concat([baseline, ml], ignore_index=True)
    combined = combined.sort_values(["target", "rmse"]).reset_index(drop=True)
    combined["rank_in_target"] = combined.groupby("target").cumcount() + 1
    cols = ["target", "rank_in_target", "category", "model",
            "rmse", "mae", "mape_pct", "residual_std", "test_rows", "deployable"]
    combined = combined[cols].round(4)

    REPORTS.mkdir(parents=True, exist_ok=True)
    combined.to_csv(REPORTS / "performance_comparison.csv", index=False)

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(17, 5.5))
    palette = {"baseline": "#7f7f7f", "ml": "#2ca02c"}
    for ax, name in zip(axes, TARGETS):
        panel = combined[combined.target == name].sort_values("rmse")
        colors = [palette[c] for c in panel["category"]]
        ax.bar(panel["model"], panel["rmse"], color=colors)
        ax.set_title(f"{name}: model RMSE ranking")
        ax.set_ylabel("RMSE (lower is better)")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(alpha=.25, axis="y")
    save_figure(fig, "performance_comparison.png")

    print(combined.to_string(index=False))
    return combined


if __name__ == "__main__":
    build_comparison()
