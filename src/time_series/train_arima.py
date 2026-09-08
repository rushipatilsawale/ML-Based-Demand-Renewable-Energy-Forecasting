import pandas as pd
import joblib
from statsmodels.tsa.arima.model import ARIMA

INPUT_FILE = "data/processed/ts_train.csv"
MODEL_OUTPUT = "models/arima.pkl"

TARGET = "national_demand_mw"

# ARIMA parameters
ORDER = (2, 1, 2)


def main():
    df = pd.read_csv(INPUT_FILE)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    # Set datetime as time-series index
    series = df.set_index("datetime")[TARGET]

    print("Training ARIMA model...")
    print(f"Training rows: {len(series)}")
    print(f"Training period: {series.index.min()} to {series.index.max()}")

    # Train ARIMA
    model = ARIMA(series, order=ORDER)
    fitted_model = model.fit()

    # Save trained model
    joblib.dump(fitted_model, MODEL_OUTPUT)

    print("\nARIMA training completed.")
    print(f"ARIMA order: {ORDER}")
    print(f"Model saved: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()