"""Split the engineered feature dataset into chronological train/test for the ML stage.

Why chronological (not random): forecasting must simulate real deployment -- the model
trains on the past and is tested on the future. A random split would leak future
observations into training and overstate accuracy.

Outputs:
    data/processed/ml_train.csv   (first 80%)
    data/processed/ml_test.csv    (last 20%)
"""
from src.models.modeling_common import PROCESSED, chronological_split, load_features


def prepare_ml_data():
    df = load_features()
    train_df, test_df = chronological_split(df)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED / "ml_train.csv", index=False)
    test_df.to_csv(PROCESSED / "ml_test.csv", index=False)
    print(f"ml_train.csv: {len(train_df):,} rows (through {train_df.datetime.iloc[-1]})")
    print(f"ml_test.csv : {len(test_df):,} rows (from {test_df.datetime.iloc[0]})")
    return train_df, test_df


if __name__ == "__main__":
    prepare_ml_data()
