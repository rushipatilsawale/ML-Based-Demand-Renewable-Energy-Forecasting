"""Select the best deployable model per target and persist the real-time model pack.

Selection criteria (objective, measured on the held-out future):
    primary    RMSE  (lower is better)
    secondary  MAE   (lower is better)
    secondary  MAPE  (lower is better)

Only feature-based models are deployable (naive baselines have no feature vector and
are excluded). The winning algorithm for each target is refit on the FULL dataset and
persisted, together with its exact feature list and held-out residual std (which the
recursive engine turns into widening 95% forecast intervals), into
models/realtime_forecasters.joblib -- the pack src/models/realtime_forecast.py loads.

Outputs:
    reports/best_model.csv
    reports/figures/best_model_comparison.png
    models/realtime_forecasters.joblib
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import pandas as pd

from src.models.modeling_common import (
    MODELS_DIR, MODEL_PACK, REPORTS, TARGETS, load_features, model_registry,
    save_figure, target_features,
)

COMPARISON = REPORTS / "performance_comparison.csv"


def select_and_deploy():
    comparison = pd.read_csv(COMPARISON)
    df = load_features()
    registry = model_registry()

    pack_models, best_rows = {}, []
    for name, target in TARGETS.items():
        panel = comparison[(comparison.target == name) & (comparison.deployable == 1)]
        if panel.empty:
            raise ValueError(f"No deployable candidate for target '{name}'.")
        # For wind, exclude degenerate zero-predicting models (e.g. random_forest with near-zero MAE)
        if name == "wind":
            active_panel = panel[panel["mae"] > 0.05]
            if not active_panel.empty:
                panel = active_panel
        winner = panel.sort_values(["rmse", "mae", "mape_pct"]).iloc[0]

        features = target_features(df.columns, target)
        from sklearn.base import clone
        model = clone(registry[winner["model"]])
        model.fit(df[features], df[target])

        pack_models[name] = {
            "model": model, "target": target, "features": features,
            "algorithm": winner["model"], "residual_std": float(winner["residual_std"]),
        }
        best_rows.append({
            "target": name, "selected_model": winner["model"], "category": winner["category"],
            "rmse": winner["rmse"], "mae": winner["mae"], "mape_pct": winner["mape_pct"],
            "residual_std": winner["residual_std"], "n_features": len(features),
        })
        print(f"{name:>6}: selected {winner['model']} (RMSE {winner['rmse']:.3f}, "
              f"MAE {winner['mae']:.3f}, MAPE {winner['mape_pct']:.2f}%)")

    best = pd.DataFrame(best_rows).round(4)
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best.to_csv(REPORTS / "best_model.csv", index=False)

    pack = {"models": pack_models, "feature_columns": list(df.columns),
            "trained_through": str(df.datetime.iloc[-1]), "data_end": str(df.datetime.iloc[-1])}
    joblib.dump(pack, MODEL_PACK)
    print(f"\nPersisted real-time model pack -> {MODEL_PACK.relative_to(MODELS_DIR.parent)}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(best["target"], best["rmse"], color="#e4572e")
    for i, row in best.iterrows():
        ax.text(i, row["rmse"], f"{row['selected_model']}\nRMSE {row['rmse']:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Held-out RMSE")
    ax.set_title("Selected best model per target")
    ax.grid(alpha=.25, axis="y")
    save_figure(fig, "best_model_comparison.png")
    return best


if __name__ == "__main__":
    select_and_deploy()
