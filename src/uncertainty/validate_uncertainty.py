import os
import pandas as pd


SOLAR_FILE = "data/processed/solar_uncertainty.csv"
WIND_FILE = "data/processed/wind_uncertainty.csv"


def validate_file(file_path, name):

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Missing {name} file: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = [
        "datetime",
        "actual",
        "prediction",
        "lower_bound",
        "upper_bound",
        "confidence_level"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{name}: Missing columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError(
            f"{name}: Output file is empty."
        )

    if df["datetime"].isna().any():
        raise ValueError(
            f"{name}: Invalid datetime values."
        )

    numeric_columns = [
        "actual",
        "prediction",
        "lower_bound",
        "upper_bound",
        "confidence_level"
    ]

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            f"{name}: Missing numeric values."
        )

    if (df["prediction"] < 0).any():
        raise ValueError(
            f"{name}: Negative predictions detected."
        )

    if (df["lower_bound"] < 0).any():
        raise ValueError(
            f"{name}: Negative lower bounds detected."
        )

    if (df["upper_bound"] < df["lower_bound"]).any():
        raise ValueError(
            f"{name}: Invalid prediction interval."
        )

    if not (df["confidence_level"] == 0.95).all():
        raise ValueError(
            f"{name}: Confidence level must be 0.95."
        )

    print(f"{name} validation: PASSED")
    print(f"Rows: {len(df)}")
    print(
        f"Average interval width: "
        f"{(df['upper_bound'] - df['lower_bound']).mean():.4f}"
    )


def main():

    print("Starting Phase 11 uncertainty validation...\n")

    validate_file(
        SOLAR_FILE,
        "Solar uncertainty"
    )

    print()

    validate_file(
        WIND_FILE,
        "Wind uncertainty"
    )

    print(
        "\nFINAL PHASE 11 UNCERTAINTY VALIDATION: PASSED"
    )


if __name__ == "__main__":
    main()