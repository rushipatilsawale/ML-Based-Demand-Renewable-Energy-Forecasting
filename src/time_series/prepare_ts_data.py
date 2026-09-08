import pandas as pd

INPUT_FILE = "data/processed/featured_dataset.csv"

TRAIN_OUTPUT = "data/processed/ts_train.csv"
TEST_OUTPUT = "data/processed/ts_test.csv"

TARGET = "national_demand_mw"

TRAIN_RATIO = 0.80


def main():
    df = pd.read_csv(INPUT_FILE)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Keep only the time-series target
    ts_data = df[["datetime", TARGET]].copy()

    # Chronological 80/20 split
    split_index = int(len(ts_data) * TRAIN_RATIO)

    train = ts_data.iloc[:split_index].copy()
    test = ts_data.iloc[split_index:].copy()

    train.to_csv(TRAIN_OUTPUT, index=False)
    test.to_csv(TEST_OUTPUT, index=False)

    print("Time-series data preparation completed.")
    print(f"Total rows : {len(ts_data)}")
    print(f"Train rows : {len(train)}")
    print(f"Test rows  : {len(test)}")

    print("\nTraining period:")
    print(train["datetime"].min(), "to", train["datetime"].max())

    print("\nTesting period:")
    print(test["datetime"].min(), "to", test["datetime"].max())

    print("\nGenerated:")
    print(TRAIN_OUTPUT)
    print(TEST_OUTPUT)


if __name__ == "__main__":
    main()