import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


INPUT_FILE = "reports/performance_comparison.csv"

OUTPUT_FILE = "reports/best_model.csv"

OUTPUT_FIGURE = "reports/figures/best_model_comparison.png"


REQUIRED_COLUMNS = [
    "model",
    "MAE",
    "RMSE",
    "MAPE"
]


def main():

    print("Starting Phase 8 best model selection...")

    # --------------------------------------------------
    # 1. Load performance comparison
    # --------------------------------------------------

    comparison = pd.read_csv(INPUT_FILE)

    print("\nLoaded performance comparison:")
    print(comparison)

    # --------------------------------------------------
    # 2. Validate required columns
    # --------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in comparison.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------
    # 3. Validate metric values
    # --------------------------------------------------

    if comparison[
        ["MAE", "RMSE", "MAPE"]
    ].isnull().any().any():

        raise ValueError(
            "Missing metric values detected."
        )

    if (
        comparison[
            ["MAE", "RMSE", "MAPE"]
        ] < 0
    ).any().any():

        raise ValueError(
            "Negative metric values detected."
        )

    # --------------------------------------------------
    # 4. Rank models
    # --------------------------------------------------

    comparison["RMSE_Rank"] = (
        comparison["RMSE"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    comparison["MAE_Rank"] = (
        comparison["MAE"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    comparison["MAPE_Rank"] = (
        comparison["MAPE"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    # --------------------------------------------------
    # 5. Select best model
    # --------------------------------------------------

    comparison = comparison.sort_values(
        by=[
            "RMSE",
            "MAE",
            "MAPE"
        ],
        ascending=[
            True,
            True,
            True
        ]
    ).reset_index(drop=True)

    best_model = comparison.iloc[0]

    # --------------------------------------------------
    # 6. Create best model result
    # --------------------------------------------------

    selected_model = pd.DataFrame({
        "model": [best_model["model"]],
        "MAE": [best_model["MAE"]],
        "RMSE": [best_model["RMSE"]],
        "MAPE": [best_model["MAPE"]],
        "RMSE_Rank": [best_model["RMSE_Rank"]],
        "MAE_Rank": [best_model["MAE_Rank"]],
        "MAPE_Rank": [best_model["MAPE_Rank"]]
    })

    # --------------------------------------------------
    # 7. Save best model
    # --------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    selected_model.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved best model to: {OUTPUT_FILE}"
    )

    # --------------------------------------------------
    # 8. Create comparison visualization
    # --------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FIGURE),
        exist_ok=True
    )

    top_models = comparison.head(5)

    plt.figure(figsize=(10, 6))

    plt.bar(
        top_models["model"].astype(str),
        top_models["RMSE"]
    )

    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.title("Top Forecasting Models by RMSE")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURE,
        dpi=300
    )

    plt.close()

    print(
        f"Saved best model figure to: {OUTPUT_FIGURE}"
    )

    # --------------------------------------------------
    # 9. Display selection
    # --------------------------------------------------

    print("\nSelected Best Model:")
    print(
        f"Model: {best_model['model']}"
    )

    print(
        f"MAE: {best_model['MAE']}"
    )

    print(
        f"RMSE: {best_model['RMSE']}"
    )

    print(
        f"MAPE: {best_model['MAPE']}"
    )

    print(
        f"RMSE Rank: {best_model['RMSE_Rank']}"
    )

    print(
        "\nPhase 8 best model selection completed."
    )


if __name__ == "__main__":
    main()