# ML-Based Demand & Renewable Energy Forecasting
## Full Project Execution Plan (as delivered)

This plan reflects the system that was actually built. The original brief allowed a 3-tier FastAPI + React + PostgreSQL architecture; the delivered system is a streamlined single-tier Python pipeline with a Streamlit dashboard and file-based artefacts (no separate API/DB tier), which keeps the project reproducible on one machine.

---

## 1. Project Overview

A real-time ML system that forecasts short-term electricity demand and renewable (solar/wind) supply with confidence ranges, simulates a storage-vs-backup dispatch decision per scenario, calculates the resulting cost / CO₂ impact, explains those calculations with SHAP, and presents everything on a live interactive dashboard.

Four connected parts (delivered as Python modules rather than separate services):

- **ML models** — per-target demand / solar / wind forecasters with widening 95% intervals.
- **Forecast engine** — recursive, anchored at the current IST hour, live-weather blended.
- **Dispatch + impact** — a live real-time simulator plus a what-if scenario simulator, renewable-battery storage accounting, and a CO₂ / backup-cost calculator.
- **Dashboard** — Streamlit UI with a Main real-time simulator (4 horizon tabs) and a Scenario what-if simulator, animated powerhouse view, ranged tables, bounds/weather/festival charts and SHAP + context explanations.

---

## 2. System Architecture (delivered)

```text
Raw data (demand + weather + wind + solar)
   → build_aligned_dataset.py (clean, align hourly, validate)
   → build_forecast_features.py (leakage-safe lags/rolling/calendar + season/festival)
   → evaluate_baselines.py (naive 1h/24h/168h/720h + linear regression)  → baseline_metrics.csv
   → prepare_ml_data.py → train_ml_models.py (RandomForest · GradientBoosting · XGBoost)  → ml_model_metrics.csv
   → compare_models.py (rank all by RMSE)  → performance_comparison.csv
   → select_best_model.py (best deployable per target, refit + persist)  → models/realtime_forecasters.joblib
   → realtime_forecast.py (recursive, anchored at NOW, live weather) → forecasts/*.csv
   → operations.py (facility scaling, dispatch, impact, renewable storage)
   → scenario.py (what-if day: season · festival · weather)
   → explain_dispatch.py (SHAP + context on dispatch/CO₂/cost)
   → app/dashboard.py (Streamlit: Main real-time tabs + Scenario simulator)
```

| Layer | Responsibility | Tech used |
|---|---|---|
| Data pipeline | Acquire, clean, align, validate the four sources | pandas, requests (Open-Meteo, NASA POWER, PVGIS) |
| ML model | Train + auto-select demand/solar/wind forecasters, persist residual std | scikit-learn (LinearRegression, Ridge, RandomForest, HistGradientBoosting), xgboost, joblib |
| Forecast engine | Real-time recursive multi-horizon projection with intervals | numpy, pandas, live Open-Meteo feed |
| Dispatch + impact | Storage-vs-backup simulation (live + what-if), renewable storage, cost & CO₂ | numpy, pandas |
| Explainability | SHAP drivers + season/festival/weather context narrative | shap |
| Dashboard | Interactive two-simulator UI with animated powerhouse | Streamlit, Altair |

> Storage of results is file-based (CSV + joblib) under `data/processed/` and `models/`. No SQL database or REST API tier was required at this scale.

---

## 3. Datasets — what was collected

| Category | Source used | Notes |
|---|---|---|
| Electricity demand | Kaggle — Hourly Load India | National + regional demand, 2019-01-01 → 2024-04-30, hourly |
| Weather | Open-Meteo API (Delhi 28.6139, 77.2090) | Temperature, humidity, cloud cover, precipitation, radiation, wind speed |
| Wind resource | NASA POWER `WS10M` (Delhi) | Wind speed → converted to power via 1 MW turbine curve |
| Solar potential | PVGIS 6 (1 MWp Delhi) | Modelled hourly potential, converted to kW |
| Emission factor | CEA India reference (0.710 kg CO₂/kWh) | Configurable scenario value |

All sources are aligned to one common hourly index (Delhi/India), so demand, weather, wind and solar line up exactly.

---

## 4. Step-by-step development process (delivered)

1. **Data collection & cleaning** — inspect, clean and align the four sources; validate continuity, nulls, duplicates → `aligned_hourly_dataset.csv`.
2. **EDA** — hourly/daily/weekly/monthly demand and renewable patterns, correlations, figures.
3. **Feature engineering** — cyclical calendar, lags (1/24/168 h), rolling (24/168 h), `shift(1)` before rolling to prevent leakage.
4. **Model training & selection** — three candidates per target, auto-select lowest validation RMSE on a chronological split.
5. **Real-time forecasting** — anchor at the current IST hour, blend live Open-Meteo weather (16 d) with climatology, recurse 12 months, widen 95% intervals, aggregate to daily/weekly/monthly, cache to CSV.
6. **Dispatch simulator & impact** — facility scaling, renewable→battery(auto-fill)→grid dispatch across all three bound levels with a `renewable_stored` column, plus a what-if scenario simulator; no-storage counterfactual for cost & CO₂.
7. **SHAP explainability** — explain the dispatch / CO₂ / cost calculations for a selected timestamp, plus a season/festival/weather/time-of-day context narrative.
8. **Dashboard** — two dispatch simulators: Main real-time across four horizon tabs (Hourly / Daily / Weekly / Monthly) and a Scenario what-if; each with a ranged table (incl. renewable-stored + SOC), bounds/weather/festival charts, an animated powerhouse, and a point-wise explanation.
9. **Cleanup & docs** — remove obsolete modules, refresh README and docs.

---

## 5. Forecast horizons per tab

| Tab | Window | Aggregation |
|---|---|---|
| Hourly | next 24 h | 1-hour blocks |
| Daily | next 7 days | 24-hour blocks |
| Weekly | next 4 weeks | 168-hour blocks |
| Monthly | next 12 months | ~730-hour blocks |

---

## 6. Tech stack summary (delivered)

| Layer | Tools |
|---|---|
| Language | Python 3.13 |
| ML / data | pandas, numpy, scikit-learn, xgboost, joblib, SHAP |
| Live data | requests (Open-Meteo, NASA POWER, PVGIS) |
| Dashboard | Streamlit, Altair |
| Plots (EDA) | matplotlib |
| Version control | Git / GitHub |

Not used (present in the original brief but out of scope for the delivered single-tier system): FastAPI, React, PostgreSQL, Prophet, LightGBM. (XGBoost IS used — it is one of the three advanced ML candidates and is the selected solar model.)

---

## 7. Final deliverables checklist

- ✔ Cleaned, aligned 4-source hourly dataset (46,728 rows, validated)
- ✔ Leakage-safe engineered feature set (46,560 × 58, incl. season + festival)
- ✔ Auto-selected best model per target with recorded metrics
- ✔ Real-time recursive multi-horizon forecast (now → +12 months) with 95% intervals
- ✔ Live dispatch simulator (storage vs backup) across three bound levels with renewable-storage column
- ✔ What-if scenario dispatch simulator (season · festival · weather · day type)
- ✔ CO₂ / backup-cost impact calculator (no-storage counterfactual)
- ✔ SHAP + context explainability of the dispatch / CO₂ / cost calculations
- ✔ Interactive two-simulator Streamlit dashboard with animated powerhouse
- ✔ Refreshed README and docs; obsolete modules removed
