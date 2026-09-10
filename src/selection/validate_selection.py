import os
import pandas as pd


INPUT_FILE = "reports/performance_comparison.csv"

OUTPUT_FILE = "reports/best_model.csv"

FIGURE_FILE = "reports/figures/best_model_comparison.png"


REQUIRED_COLUMNS = [
    "model",
    "MAE",
    "RMSE",
    "MAPE",
    "RMSE_Rank",
    "MAE_Rank",
    "MAPE_Rank"
]


def main():

    print("Starting Phase 8 best model selection validation...")

    # --------------------------------------------------
    # 1. Check generated files
    # --------------------------------------------------

    print("\nChecking generated files...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}"
        )

    print(
        f"{INPUT_FILE}: PASSED"
    )

    if not os.path.exists(OUTPUT_FILE):
        raise FileNotFoundError(
            f"Missing output file: {OUTPUT_FILE}"
        )

    print(
        f"{OUTPUT_FILE}: PASSED"
    )

    if not os.path.exists(FIGURE_FILE):
        raise FileNotFoundError(
            f"Missing figure: {FIGURE_FILE}"
        )

    print(
        f"{FIGURE_FILE}: PASSED"
    )

    # --------------------------------------------------
    # 2. Load files
    # --------------------------------------------------

    comparison = pd.read_csv(INPUT_FILE)

    selected = pd.read_csv(OUTPUT_FILE)

    # --------------------------------------------------
    # 3. Validate output columns
    # --------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in selected.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print("\nOutput columns: PASSED")

    # --------------------------------------------------
    # 4. Validate exactly one selected model
    # --------------------------------------------------

    if len(selected) != 1:
        raise ValueError(
            "Best model output must contain exactly one model."
        )

    print("Selected model count: PASSED")

    # --------------------------------------------------
    # 5. Validate selected model exists
    # --------------------------------------------------

    selected_model = selected.iloc[0]["model"]

    if selected_model not in comparison["model"].values:
        raise ValueError(
            "Selected model does not exist in comparison data."
        )

    print("Selected model existence: PASSED")

    # --------------------------------------------------
    # 6. Validate metric values
    # --------------------------------------------------

    if selected[
        ["MAE", "RMSE", "MAPE"]
    ].isnull().any().any():

        raise ValueError(
            "Missing metric values detected."
        )

    if (
        selected[
            ["MAE", "RMSE", "MAPE"]
        ] < 0
    ).any().any():

        raise ValueError(
            "Negative metric values detected."
        )

    print("Metric value check: PASSED")

    # --------------------------------------------------
    # 7. Validate best RMSE
    # --------------------------------------------------

    best_rmse = comparison["RMSE"].min()

    selected_rmse = selected.iloc[0]["RMSE"]

    if selected_rmse != best_rmse:
        raise ValueError(
            "Selected model does not have the best RMSE."
        )

    print("Best RMSE check: PASSED")

    # --------------------------------------------------
    # 8. Validate ranking
    # --------------------------------------------------

    if selected.iloc[0]["RMSE_Rank"] != 1:
        raise ValueError(
            "Selected model does not have RMSE rank 1."
        )

    print("RMSE ranking check: PASSED")

    # --------------------------------------------------
    # 9. Final validation
    # --------------------------------------------------

    print(
        "\nSelected Best Model:"
    )

    print(
        selected_model
    )

    print(
        "\nFINAL PHASE 8 VALIDATION: PASSED"
    )


if __name__ == "__main__":
    main()