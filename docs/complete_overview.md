# Complete Project Overview

## Scope

An end-to-end, **real-time** electricity-demand and renewable-energy forecasting and decision-support system for India. It learns from four aligned historical datasets, trains and auto-selects a best model per target, forecasts forward from the current moment, simulates dispatch, quantifies CO₂ / cost impact, explains those calculations with SHAP, and presents everything in a 4-tab Streamlit dashboard.

## Data pipeline

Raw inputs: Hourly Load India (demand), Open-Meteo Delhi weather, NASA POWER `WS10M` wind resource, and PVGIS 6 modelled 1 MWp Delhi solar potential. `build_aligned_dataset.py` inner-joins all four on an exact common hourly index (2019-01-01 → 2024-04-30), adds calendar fields, and converts wind speed to a bounded 1 MW turbine-power potential. Result: **46,728 rows × 23 columns, zero missing values, zero duplicate timestamps** (`aligned_hourly_dataset.csv` + `data_quality_summary.csv`).

## Features & EDA

`run_aligned_eda.py` produces hourly/daily/weekly/monthly demand and renewable pattern records (dual-axis plots so the MW demand and kW renewable scales are both legible), correlations and figures. `pattern_analysis.py` adds the deeper context the dashboard and SHAP rely on: weekday/weekend, regional and yearly demand trends, IMD four-season patterns, the Indian festival calendar, and weather-vs-demand and weather-vs-renewable relationships → `reports/patterns/*.csv` + `reports/figures/patterns/*.png`. `build_forecast_features.py` adds cyclical calendar features, **season one-hot + festival flag** (from the shared `src/features/calendar_features.py`), lag features (1/24/168 h) and rolling statistics (24/168 h) for all three targets, applying `shift(1)` before every rolling window to prevent target leakage → `forecast_features.csv` (46,560 × 58) + `feature_manifest.json`.

## Models

Modelling runs as a four-stage pipeline, all stages sharing one chronological 80/20 split and one metric set (`src/models/modeling_common.py`):

1. **Baseline** (`src/baseline/evaluate_baselines.py`) — seasonal-naive at 1 h / 24 h / 168 h / 720 h plus a linear-regression benchmark, per target → `reports/baseline_metrics.csv`.
2. **Advanced ML** (`src/models/prepare_ml_data.py` → `train_ml_models.py`) — RandomForest, HistGradientBoosting and XGBoost per target → `reports/ml_model_metrics.csv`.
3. **Comparison** (`src/comparison/compare_models.py`) — every model ranked by RMSE within each target → `reports/performance_comparison.csv`.
4. **Selection** (`src/selection/select_best_model.py`) — the best *deployable* (feature-based) model per target is chosen by RMSE→MAE→MAPE, refit on the full dataset, and persisted with its feature list, algorithm name and held-out residual std (which drives the widening 95% intervals) → `models/realtime_forecasters.joblib`, `reports/best_model.csv`.

Selected on the current data: **demand → linear_regression (RMSE 3237.13)**, **solar → xgboost (RMSE 28.39)**, **wind → xgboost (RMSE 6.44)**. Naive baselines are benchmarks only — they have no feature vector, so they are excluded from deployment.

## Real-time forecasting

`live_weather.py` pulls the live Open-Meteo Delhi forecast (next 16 days). `realtime_forecast.py` anchors at the current IST hour, blends live weather with month×hour climatological normals beyond 16 days, and runs a recursive autoregressive projection (lags feed forward) for 8,760 hours with widening 95% intervals. Outputs are cached to `forecasts/hourly|daily|weekly|monthly_forecast.csv` plus `future_features.csv` for SHAP. Demand is seeded from the last 168 observed hours because there is no live demand feed (history ends 2024-04-30) — a documented limitation.

## Dispatch, cost & CO₂

`operations.py` scales national values to a facility/microgrid scenario and runs a 3-level (lower/expected/upper) dispatch simulation: direct renewable use → **battery auto-fill from renewable surplus** → battery discharge → grid backup → curtailment, with exact energy conservation and SOC bounds. The renewable surplus stored each hour is exposed as a first-class `renewable_stored_*_kw` column (plus a running cumulative) so hourly/daily/weekly/monthly records report storage directly. Facility sizing defaults to a solar-heavy microgrid (demand ×0.05 ≈ 8 MW, solar ×14 ≈ 9 MWp, wind ×200 ≈ 3 MW) so a clear midday genuinely over-produces and charges the battery. Impact compares a no-storage counterfactual against with-storage to derive cost savings (× tariff) and avoided CO₂ (× emission factor).

`scenario.py` is the **second, what-if dispatch simulator**: the operator picks a season, festival day, day type and weather preset (sunny / cloudy / rainy / windy / clear night), and the module builds a representative 24-hour day — calendar features from the shared festival/season module, a diurnal solar-radiation and temperature curve, and historical month-by-hour analogs for the lag/rolling inputs — then runs it through the *same* trained model pack and the *same* battery hierarchy. The main real-time simulator keeps running the live forecast independently.

## Explainability

`explain_dispatch.py` uses SHAP (LinearExplainer for the linear demand model, TreeExplainer for the tree-based solar/wind models) over the persisted `future_features.csv` to explain the **dispatch, CO₂ and cost calculations** for a selected timestamp — not model selection. On top of the SHAP drivers it adds a plain-language **context narrative** (season, festival, time-of-day, and weather conditions such as cloud cover, heat, humidity, rain or wind) that answers "why is demand high/low right now". `explain_from_features` provides the same explanation for scenario what-if hours whose timestamps are not in the real-time cache.

## Dashboard

`app/dashboard.py` reads the cached forecasts and model pack (it never retrains or re-recurses on startup) and presents **two dispatch simulators**. The **Main** simulator renders four horizon tabs — Hourly (next 24 h), Daily (next 7 days), Weekly (next 4 weeks), Monthly (next 12 months) — each with a ranged table (demand / renewable supply / **renewable stored** / SOC / storage discharge / backup), a bounds chart, a weather-context chart, an **animated powerhouse flow diagram** (solar + wind sources, a filling/draining battery container, backup source, with flows that animate on the actual dispatch numbers), and a point-wise SHAP + context explanation. The **Scenario** simulator adds season/festival/weather/day-type selectors, an hour-by-hour festival-shape chart, the same animated powerhouse and explanation for the what-if day. Facility sizing, battery spec, tariff and emission factor live in collapsible sidebar reference-assumption expanders (no scenario sidebar).

## Run

From the repository root, using the local `.venv`:

```bash
.venv/Scripts/python.exe -m src.data.build_aligned_dataset
.venv/Scripts/python.exe -m src.eda.run_aligned_eda
.venv/Scripts/python.exe -m src.eda.pattern_analysis
.venv/Scripts/python.exe -m src.features.build_forecast_features
.venv/Scripts/python.exe -m src.baseline.evaluate_baselines
.venv/Scripts/python.exe -m src.models.prepare_ml_data
.venv/Scripts/python.exe -m src.models.train_ml_models
.venv/Scripts/python.exe -m src.comparison.compare_models
.venv/Scripts/python.exe -m src.selection.select_best_model
.venv/Scripts/python.exe -m src.models.realtime_forecast --hours 8760   # ~10 min, anchored at NOW
.venv/Scripts/streamlit.exe run app/dashboard.py
```

If the forecast cache and model pack already exist, the final streamlit command is sufficient.
