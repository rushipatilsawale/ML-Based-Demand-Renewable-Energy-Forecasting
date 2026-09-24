"""Linear-regression baseline: a simple feature-based ML benchmark.

Learns a linear relationship between the engineered features (calendar incl. season
and festival, weather, and the target's own lag/rolling history) and each target. It
is deployable (it consumes the feature vector), so unlike the naive baselines it can
be selected for the real-time engine if it wins on RMSE.
"""
from src.models.modeling_common import model_registry, target_features


def fit_predict_linear(train_df, test_df, target):
    features = target_features(train_df.columns, target)
    model = model_registry()["linear_regression"]
    model.fit(train_df[features], train_df[target])
    pred = model.predict(test_df[features])
    return pred, features
