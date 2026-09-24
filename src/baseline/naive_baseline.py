"""Seasonal-naive baselines: prediction(t) = actual(t - lag).

Four horizons are benchmarked so the ML models are compared against persistence at
the natural seasonal cycles of electricity demand:
    naive_1h    -> previous hour            (persistence)
    naive_24h   -> same hour yesterday      (daily seasonality)
    naive_168h  -> same hour last week      (weekly seasonality)
    naive_720h  -> same hour ~30 days ago   (monthly seasonality)

These are benchmarks only: they have no feature vector, so they are never deployed
into the recursive real-time engine (see modeling_common.DEPLOYABLE_MODELS).
"""
import pandas as pd


def naive_predictions(df, target, lag):
    """Return a full-length prediction series: each row predicts target.shift(lag)."""
    return df[target].shift(lag)


def naive_test_predictions(df, target, lag, cut):
    """Slice the naive predictions to the chronological test window (rows >= cut)."""
    pred = naive_predictions(df, target, lag).iloc[cut:]
    actual = df[target].iloc[cut:]
    frame = pd.DataFrame({"actual": actual, "predicted": pred}).dropna()
    return frame["actual"].to_numpy(), frame["predicted"].to_numpy(), frame.index
