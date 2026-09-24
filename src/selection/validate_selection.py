"""Validate the selection artifact and the deployed real-time model pack.

Checks: best_model.csv covers all three targets with a deployable winner; the joblib
pack loads, contains every target with model/features/algorithm/residual_std; each
stored model can predict on a correctly-ordered feature row built from its own
feature list (this is exactly what the recursive engine does at inference time).
"""
import sys

import joblib
import numpy as np
import pandas as pd

from src.models.modeling_common import (
    DEPLOYABLE_MODELS, MODEL_PACK, REPORTS, TARGETS, load_features,
)


def validate_selection():
    errors = []
    best_path = REPORTS / "best_model.csv"
    if not best_path.exists():
        print("SELECTION VALIDATION: FAILED (best_model.csv missing)")
        sys.exit(1)
    best = pd.read_csv(best_path)

    for name, target in TARGETS.items():
        row = best[best.target == name]
        if row.empty:
            errors.append(f"best_model.csv missing target '{name}'")
            continue
        if row.iloc[0]["selected_model"] not in DEPLOYABLE_MODELS:
            errors.append(f"{name}: selected model is not deployable")

    if not MODEL_PACK.exists():
        errors.append("realtime_forecasters.joblib missing")
    else:
        pack = joblib.load(MODEL_PACK)
        df = load_features()
        sample = df.tail(1)
        for name, target in TARGETS.items():
            info = pack["models"].get(name)
            if not info:
                errors.append(f"pack missing target '{name}'")
                continue
            for key in ("model", "features", "algorithm", "residual_std"):
                if key not in info:
                    errors.append(f"pack[{name}] missing '{key}'")
            missing_feats = [f for f in info["features"] if f not in sample.columns]
            if missing_feats:
                errors.append(f"pack[{name}] features absent from dataset: {missing_feats[:5]}")
                continue
            try:
                pred = info["model"].predict(sample[info["features"]].to_numpy(float))
                if not np.isfinite(pred).all():
                    errors.append(f"pack[{name}] produced non-finite prediction")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"pack[{name}] predict failed: {exc}")

    if errors:
        print("SELECTION VALIDATION: FAILED")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"SELECTION VALIDATION: PASSED ({len(best)} targets deployed)")
    print(best.to_string(index=False))
    return best


if __name__ == "__main__":
    validate_selection()
