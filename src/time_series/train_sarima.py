import pandas as pd
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX

INPUT_FILE = "data/processed/ts_train.csv"
MODEL_OUTPUT = "models/sarima.pkl"

TARGET = "national_demand_mw"

# SARIMA parameters
ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 24)

# Use the most recent 90 days of hourly data
TRAIN_HOURS = 24 * 90


def main():
    df = pd.read_csv(INPUT_FILE)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    # Keep only the target time series
    series = df.set_index("datetime")[TARGET]

    # Use recent data to make SARIMA computationally practical
    if len(series) > TRAIN_HOURS:
        series = series.iloc[-TRAIN_HOURS:]

    print("Training SARIMA model...")
    print(f"Training rows: {len(series)}")
    print(f"Training period: {series.index.min()} to {series.index.max()}")
    print(f"SARIMA order: {ORDER}")
    print(f"Seasonal order: {SEASONAL_ORDER}")

    model = SARIMAX(
        series,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_model = model.fit(
        disp=False,
        maxiter=50
    )

    joblib.dump(fitted_model, MODEL_OUTPUT)

    print("\nSARIMA training completed.")
    print(f"Model saved: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()