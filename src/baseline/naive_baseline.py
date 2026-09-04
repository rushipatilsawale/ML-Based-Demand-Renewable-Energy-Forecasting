import pandas as pd


INPUT_FILE = "data/processed/featured_dataset.csv"
OUTPUT_FILE = "data/processed/baseline_predictions.csv"

TARGET = "national_demand_mw"


def create_naive_predictions():
    df = pd.read_csv(INPUT_FILE, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Previous day same hour
    df["naive_24h"] = df[TARGET].shift(24)

    # Previous week same hour
    df["naive_168h"] = df[TARGET].shift(168)

    # Keep required columns
    predictions = df[
        [
            "datetime",
            TARGET,
            "naive_24h",
            "naive_168h",
        ]
    ].copy()

    predictions.to_csv(OUTPUT_FILE, index=False)

    print("Naive baseline predictions created successfully.")
    print(f"Rows: {len(predictions)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_naive_predictions()