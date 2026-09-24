"""Validate the performance-comparison artifact.

Checks: every target is present, metrics are finite and non-negative, and rows are
correctly ordered by RMSE ascending within each target (rank_in_target consistent).
"""
import sys

import numpy as np
import pandas as pd

from src.models.modeling_common import REPORTS, TARGETS


def validate_comparison():
    path = REPORTS / "performance_comparison.csv"
    df = pd.read_csv(path)
    errors = []

    for target in TARGETS:
        panel = df[df.target == target]
        if panel.empty:
            errors.append(f"missing target: {target}")
            continue
        metric_cols = ["rmse", "mae", "mape_pct", "residual_std"]
        if not np.isfinite(panel[metric_cols].to_numpy(float)).all():
            errors.append(f"{target}: non-finite metric values")
        if (panel[metric_cols] < 0).any().any():
            errors.append(f"{target}: negative metric values")
        rmse = panel.sort_values("rank_in_target")["rmse"].to_numpy()
        if not np.all(np.diff(rmse) >= -1e-9):
            errors.append(f"{target}: rows are not ordered by ascending RMSE")

    if errors:
        print("COMPARISON VALIDATION: FAILED")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"COMPARISON VALIDATION: PASSED ({len(df)} rows, {df.target.nunique()} targets)")
    return df


if __name__ == "__main__":
    validate_comparison()
