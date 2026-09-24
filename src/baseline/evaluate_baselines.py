"""Evaluate all baseline models on the shared chronological test split.

Produces:
    data/processed/baseline_predictions.csv   (long: datetime, target, model, actual, predicted)
    reports/baseline_metrics.csv              (per target+model: mae, rmse, mape_pct, residual_std)
    reports/figures/baseline_comparison.png   (RMSE per model, one panel per target)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.models.modeling_common import (
    NAIVE_LAGS, PROCESSED, REPORTS, TARGETS, TEST_FRACTION, chronological_split,
    evaluate, load_features, save_figure,
)
from src.baseline.naive_baseline import naive_test_predictions
from src.baseline.regression_baseline import fit_predict_linear


def run_baselines():
    df = load_features()
    train_df, test_df = chronological_split(df)
    cut = len(train_df)

    pred_rows, metric_rows = [], []
    for name, target in TARGETS.items():
        # Seasonal-naive benchmarks (1h / 24h / weekly / monthly).
        for model_name, lag in NAIVE_LAGS.items():
            actual, predicted, idx = naive_test_predictions(df, target, lag, cut)
            metrics = evaluate(actual, predicted)
            metric_rows.append({"target": name, "model": model_name, **metrics,
                                "test_rows": len(actual), "deployable": 0})
            for ts, a, p in zip(df["datetime"].iloc[idx], actual, predicted):
                pred_rows.append({"datetime": ts, "target": name, "model": model_name,
                                  "actual": a, "predicted": p})

        # Linear-regression benchmark (deployable).
        predicted, _ = fit_predict_linear(train_df, test_df, target)
        actual = test_df[target].to_numpy()
        metrics = evaluate(actual, predicted)
        metric_rows.append({"target": name, "model": "linear_regression", **metrics,
                            "test_rows": len(actual), "deployable": 1})
        for ts, a, p in zip(test_df["datetime"], actual, predicted):
            pred_rows.append({"datetime": ts, "target": name, "model": "linear_regression",
                              "actual": a, "predicted": p})

    predictions = pd.DataFrame(pred_rows)
    metrics = pd.DataFrame(metric_rows).round(4)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PROCESSED / "baseline_predictions.csv", index=False)
    metrics.to_csv(REPORTS / "baseline_metrics.csv", index=False)

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(16, 5))
    for ax, name in zip(axes, TARGETS):
        panel = metrics[metrics.target == name].sort_values("rmse")
        ax.bar(panel["model"], panel["rmse"], color="#1f77b4")
        ax.set_title(f"{name} baseline RMSE")
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=.25, axis="y")
    save_figure(fig, "baseline_comparison.png")

    print(f"Baselines evaluated on {cut}->{len(df)} test window.")
    print(metrics.to_string(index=False))
    return metrics


if __name__ == "__main__":
    run_baselines()
