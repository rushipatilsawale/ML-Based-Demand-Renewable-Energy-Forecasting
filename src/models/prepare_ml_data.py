import os
import pandas as pd


INPUT_FILE = "data/processed/featured_dataset.csv"

TRAIN_FILE = "data/processed/ml_train.csv"
TEST_FILE = "data/processed/ml_test.csv"

TARGET = "national_demand_mw"


def prepare_ml_data():
    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["datetime"],
    )

    df = df.sort_values("datetime").reset_index(drop=True)

    # Chronological 80/20 split
    split_index = int(len(df) * 0.80)

    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    os.makedirs("data/processed", exist_ok=True)

    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)

    print("ML data preparation completed.")
    print("=" * 50)

    print(f"Total rows : {len(df)}")
    print(f"Train rows : {len(train)}")
    print(f"Test rows  : {len(test)}")

    print("\nTraining period:")
    print(f"{train['datetime'].min()} → {train['datetime'].max()}")

    print("\nTesting period:")
    print(f"{test['datetime'].min()} → {test['datetime'].max()}")

    print("\nTarget:")
    print(TARGET)

    print(f"\nTraining data saved to: {TRAIN_FILE}")
    print(f"Testing data saved to : {TEST_FILE}")


if __name__ == "__main__":
    prepare_ml_data()