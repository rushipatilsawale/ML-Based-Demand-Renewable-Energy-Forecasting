# Complete Project Overview

## Scope

An end-to-end, **real-time** electricity-demand and renewable-energy forecasting and decision-support system for India. It learns from four aligned historical datasets, trains and auto-selects a best model per target, forecasts forward from the current moment, simulates dispatch, quantifies CO₂ / cost impact, explains those calculations with SHAP, and presents everything in a 4-tab Streamlit dashboard.

## Data pipeline

Raw inputs: Hourly Load India (demand), Open-Meteo Delhi weather, NASA POWER `WS10M` wind resource, and PVGIS 6 modelled 1 MWp Delhi solar potential. `build_aligned_dataset.py` inner-joins all four on an exact common hourly index (2019-01-01 → 2024-04-30), adds calendar fields, and converts wind speed to a bounded 1 MW turbine-power potential. Result: **46,728 rows × 23 columns, zero missing values, zero duplicate timestamps** (`aligned_hourly_dataset.csv` + `data_quality_summary.csv`).

## Features & EDA

`run_aligned_eda.py` produces hourly/daily/weekly/monthly demand and renewable pattern records, correlations and figures. `build_forecast_features.py` adds cyclical calendar features, lag features (1/24/168 h) and rolling statistics (24/168 h) for all three targets, applying `shift(1)` before every rolling window to prevent target leakage → `forecast_features.csv` (46,560 × 51) + `feature_manifest.json`.

## Models

`train_realtime_models.py` trains three scikit-learn candidates (Ridge, RandomForest, HistGradientBoosting) per target and keeps the lowest-RMSE model on a chronological split. Selected: **demand → Ridge (RMSE 3253.33)**, **solar → HistGradientBoosting (RMSE 28.66)**, **wind → RandomForest (RMSE 0.186)**. The model, feature list, algorithm name and residual std are persisted in `models/realtime_forecasters.joblib`; metrics in `reports/realtime_model_metrics.csv`.

## Real-time forecasting

`live_weather.py` pulls the live Open-Meteo Delhi forecast (next 16 days). `realtime_forecast.py` anchors at the current IST hour, blends live weather with month×hour climatological normals beyond 16 days, and runs a recursive autoregressive projection (lags feed forward) for 8,760 hours with widening 95% intervals. Outputs are cached to `forecasts/hourly|daily|weekly|monthly_forecast.csv` plus `future_features.csv` for SHAP. Demand is seeded from the last 168 observed hours because there is no live demand feed (history ends 2024-04-30) — a documented limitation.

## Dispatch, cost & CO₂

`operations.py` scales national values to a facility/microgrid scenario and runs a 3-level (lower/expected/upper) dispatch simulation: direct renewable use → battery charge → battery discharge → grid backup → curtailment, with exact energy conservation and SOC bounds. Impact compares a no-storage counterfactual against with-storage to derive cost savings (× tariff) and avoided CO₂ (× emission factor).

## Explainability

`explain_dispatch.py` uses SHAP (LinearExplainer for Ridge, TreeExplainer for trees) over the persisted `future_features.csv` to explain the **dispatch, CO₂ and cost calculations** for a selected timestamp — not model selection.

## Dashboard

`app/dashboard.py` reads the cached forecasts and model pack (it never retrains or re-recurses on startup) and renders exactly four tabs — Hourly (next 24 h), Daily (next 7 days), Weekly (next 4 weeks), Monthly (next 12 months). Each tab shows a ranged table (demand / renewable supply / storage / backup), one bounds chart with colour-coded upper/lower bands, a scenario dispatch simulator, and a point-wise SHAP explanation block.

## Run

From the repository root, using the local `.venv`:

```bash
.venv/Scripts/python.exe src/data/build_aligned_dataset.py
.venv/Scripts/python.exe src/eda/run_aligned_eda.py
.venv/Scripts/python.exe src/features/build_forecast_features.py
.venv/Scripts/python.exe src/models/train_realtime_models.py
.venv/Scripts/python.exe src/models/realtime_forecast.py     # ~10 min, anchored at NOW
.venv/Scripts/streamlit.exe run app/dashboard.py
```

If the forecast cache and model pack already exist, the final streamlit command is sufficient.
