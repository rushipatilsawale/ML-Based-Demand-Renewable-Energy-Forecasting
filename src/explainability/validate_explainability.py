import os
import pandas as pd


SUMMARY_FILE = "reports/explainability_summary.csv"

SHAP_SUMMARY_FILE = "reports/figures/shap_summary.png"

SHAP_BAR_FILE = "reports/figures/shap_bar.png"


REQUIRED_COLUMNS = [
    "feature",
    "mean_abs_shap"
]


def main():

    print(
        "Starting Phase 9 explainability validation..."
    )

    # --------------------------------------------------
    # 1. Check generated files
    # --------------------------------------------------

    print("\nChecking generated files...")

    if not os.path.exists(SUMMARY_FILE):
        raise FileNotFoundError(
            f"Missing file: {SUMMARY_FILE}"
        )

    print(
        f"{SUMMARY_FILE}: PASSED"
    )

    if not os.path.exists(SHAP_SUMMARY_FILE):
        raise FileNotFoundError(
            f"Missing file: {SHAP_SUMMARY_FILE}"
        )

    print(
        f"{SHAP_SUMMARY_FILE}: PASSED"
    )

    if not os.path.exists(SHAP_BAR_FILE):
        raise FileNotFoundError(
            f"Missing file: {SHAP_BAR_FILE}"
        )

    print(
        f"{SHAP_BAR_FILE}: PASSED"
    )

    # --------------------------------------------------
    # 2. Load summary
    # --------------------------------------------------

    summary = pd.read_csv(
        SUMMARY_FILE
    )

    # --------------------------------------------------
    # 3. Check columns
    # --------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in summary.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print(
        "\nRequired columns: PASSED"
    )

    # --------------------------------------------------
    # 4. Check feature count
    # --------------------------------------------------

    if len(summary) == 0:
        raise ValueError(
            "Explainability summary is empty."
        )

    print(
        "Feature count: PASSED"
    )

    # --------------------------------------------------
    # 5. Check duplicate features
    # --------------------------------------------------

    if summary["feature"].duplicated().any():
        raise ValueError(
            "Duplicate features detected."
        )

    print(
        "Duplicate feature check: PASSED"
    )

    # --------------------------------------------------
    # 6. Check SHAP values
    # --------------------------------------------------

    if summary[
        "mean_abs_shap"
    ].isnull().any():

        raise ValueError(
            "Missing SHAP importance values detected."
        )

    if (
        summary["mean_abs_shap"] < 0
    ).any():

        raise ValueError(
            "Negative SHAP importance values detected."
        )

    print(
        "SHAP value check: PASSED"
    )

    # --------------------------------------------------
    # 7. Check sorting
    # --------------------------------------------------

    expected = (
        summary["mean_abs_shap"]
        .sort_values(
            ascending=False
        )
        .reset_index(drop=True)
    )

    actual = (
        summary["mean_abs_shap"]
        .reset_index(drop=True)
    )

    if not actual.equals(expected):
        raise ValueError(
            "Features are not sorted by SHAP importance."
        )

    print(
        "SHAP importance ordering: PASSED"
    )

    # --------------------------------------------------
    # 8. Final validation
    # --------------------------------------------------

    print(
        "\nTop feature:"
    )

    print(
        summary.iloc[0]["feature"]
    )

    print(
        "\nFINAL PHASE 9 VALIDATION: PASSED"
    )


if __name__ == "__main__":
    main()