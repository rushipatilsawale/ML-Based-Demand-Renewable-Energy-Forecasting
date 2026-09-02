from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_merged_dataset.csv"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():
    """Load the final merged dataset."""
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    return df


# ---------------------------------------------------------
# BASIC DATASET INFORMATION
# ---------------------------------------------------------

def inspect_dataset(df):
    """Display basic information about the dataset."""

    print("=" * 60)
    print("BASIC EDA - FINAL MERGED DATASET")
    print("=" * 60)

    print("\n--- DATASET SHAPE ---")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\n--- COLUMNS ---")
    for column in df.columns:
        print(f"  - {column}")

    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    print("\n--- MISSING VALUES ---")
    print(df.isna().sum())

    print("\n--- DUPLICATE ROWS ---")
    print(df.duplicated().sum())

    print("\n--- DATE RANGE ---")
    print(f"Start : {df['datetime'].min()}")
    print(f"End   : {df['datetime'].max()}")

    print("\n--- NUMERICAL SUMMARY ---")
    print(df.describe().round(2).to_string())


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    df = load_data()
    inspect_dataset(df)

    print("\n" + "=" * 60)
    print("BASIC EDA COMPLETED")
    print("=" * 60)