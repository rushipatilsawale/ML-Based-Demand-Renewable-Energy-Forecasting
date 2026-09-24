"""Train the three advanced ML models per target and evaluate on the held-out future.

Candidates (see modeling_common.advanced_ml_models):
    random_forest      - bagged decision trees, averaged
    gradient_boosting  - HistGradientBoosting, sequential trees correcting residuals
    xgboost            - optimised gradient boosting, strong on tabular/time-series

Each target (demand, solar, wind) is fit independently on ml_train.csv and scored on
ml_test.csv with MAE / RMSE / MAPE. The stage does NOT pick a winner or persist a
deployment model -- that is the selection stage's job.

Outputs:
    data/processed/ml_predictions.csv     (long: datetime, target, model, actual, predicted)
    reports/ml_model_metrics.csv          (per target+model metrics, deployable=1)
    reports/figures/ml_model_comparison.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.modeling_common import (
    PROCESSED, REPORTS, TARGETS, advanced_ml_models, evaluate, save_figure, target_features,
)


def train_ml_models():
    train_df = pd.read_csv(PROCESSED / "ml_train.csv", parse_dates=["datetime"])
    test_df = pd.read_csv(PROCESSED / "ml_test.csv", parse_dates=["datetime"])

    pred_rows, metric_rows = [], []
    for name, target in TARGETS.items():
        features = target_features(train_df.columns, target)
        X_tr, y_tr = train_df[features], train_df[target]
        X_te = test_df[features]
        y_te = test_df[target].to_numpy()

        for model_name, model in advanced_ml_models().items():
            model.fit(X_tr, y_tr)
            pred = np.maximum(0.0, model.predict(X_te))
            metrics = evaluate(y_te, pred)
            metric_rows.append({"target": name, "model": model_name, **metrics,
                                "test_rows": len(y_te), "deployable": 1})
            for ts, a, p in zip(test_df["datetime"], y_te, pred):
                pred_rows.append({"datetime": ts, "target": name, "model": model_name,
                                  "actual": a, "predicted": p})
            print(f"{name:>6} / {model_name:<16} RMSE {metrics['rmse']:.3f} "
                  f"MAE {metrics['mae']:.3f} MAPE {metrics['mape_pct']:.2f}%")

    predictions = pd.DataFrame(pred_rows)
    metrics = pd.DataFrame(metric_rows).round(4)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PROCESSED / "ml_predictions.csv", index=False)
    metrics.to_csv(REPORTS / "ml_model_metrics.csv", index=False)

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(15, 5))
    for ax, name in zip(axes, TARGETS):
        panel = metrics[metrics.target == name].sort_values("rmse")
        ax.bar(panel["model"], panel["rmse"], color="#2ca02c")
        ax.set_title(f"{name} ML model RMSE")
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=.25, axis="y")
    save_figure(fig, "ml_model_comparison.png")

    print("\nML model metrics:")
    print(metrics.to_string(index=False))
    return metrics


if __name__ == "__main__":
    train_ml_models()
