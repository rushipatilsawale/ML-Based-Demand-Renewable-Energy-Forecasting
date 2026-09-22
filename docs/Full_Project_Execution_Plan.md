# ML-Based Demand & Renewable Energy Forecasting
## Full Project Execution Plan (as delivered)

This plan reflects the system that was actually built. The original brief allowed a 3-tier FastAPI + React + PostgreSQL architecture; the delivered system is a streamlined single-tier Python pipeline with a Streamlit dashboard and file-based artefacts (no separate API/DB tier), which keeps the project reproducible on one machine.

---

## 1. Project Overview

A real-time ML system that forecasts short-term electricity demand and renewable (solar/wind) supply with confidence ranges, simulates a storage-vs-backup dispatch decision per scenario, calculates the resulting cost / CO₂ impact, explains those calculations with SHAP, and presents everything on a live interactive dashboard.

Four connected parts (delivered as Python modules rather than separate services):

- **ML models** — per-target demand / solar / wind forecasters with widening 95% intervals.
- **Forecast engine** — recursive, anchored at the current IST hour, live-weather blended.
- **Dispatch + impact** — scenario simulator and CO₂ / backup-cost calculator.
- **Dashboard** — 4-tab Streamlit UI with ranged tables, bounds charts, scenario simulator and SHAP explanations.

---

## 2. System Architecture (delivered)

```text
Raw data (demand + weather + wind + solar)
   → build_aligned_dataset.py (clean, align hourly, validate)
   → build_forecast_features.py (leakage-safe lags/rolling/calendar)
   → train_realtime_models.py (auto-select best per target)  → models/realtime_forecasters.joblib
   → realtime_forecast.py (recursive, anchored at NOW, live weather) → forecasts/*.csv
   → operations.py (facility scaling, dispatch scenario, impact)
   → explain_dispatch.py (SHAP on dispatch/CO₂/cost)
   → app/dashboard.py (Streamlit, 4 tabs)
```

| Layer | Responsibility | Tech used |
|---|---|---|
| Data pipeline | Acquire, clean, align, validate the four sources | pandas, requests (Open-Meteo, NASA POWER, PVGIS) |
| ML model | Train + auto-select demand/solar/wind forecasters, persist residual std | scikit-learn (Ridge, RandomForest, HistGradientBoosting), joblib |
| Forecast engine | Real-time recursive multi-horizon projection with intervals | numpy, pandas, live Open-Meteo feed |
| Dispatch + impact | Storage-vs-backup simulation, cost & CO₂ | numpy, pandas |
| Explainability | SHAP drivers of dispatch / CO₂ / cost | shap |
| Dashboard | Interactive 4-tab UI | Streamlit, Altair |

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
6. **Dispatch simulator & impact** — facility scaling, renewable→battery→grid dispatch across all three bound levels, no-storage counterfactual for cost & CO₂.
7. **SHAP explainability** — explain the dispatch / CO₂ / cost calculations for a selected timestamp.
8. **Dashboard** — exactly four tabs (Hourly / Daily / Weekly / Monthly), each with a ranged table, one bounds chart, a scenario simulator and a point-wise explanation.
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
| ML / data | pandas, numpy, scikit-learn, joblib, SHAP |
| Live data | requests (Open-Meteo, NASA POWER, PVGIS) |
| Dashboard | Streamlit, Altair |
| Plots (EDA) | matplotlib |
| Version control | Git / GitHub |

Not used (present in the original brief but out of scope for the delivered single-tier system): FastAPI, React, PostgreSQL, Prophet, XGBoost/LightGBM.

---

## 7. Final deliverables checklist

- ✔ Cleaned, aligned 4-source hourly dataset (46,728 rows, validated)
- ✔ Leakage-safe engineered feature set (46,560 × 51)
- ✔ Auto-selected best model per target with recorded metrics
- ✔ Real-time recursive multi-horizon forecast (now → +12 months) with 95% intervals
- ✔ Scenario dispatch simulator (storage vs backup) across three bound levels
- ✔ CO₂ / backup-cost impact calculator (no-storage counterfactual)
- ✔ SHAP explainability of the dispatch / CO₂ / cost calculations
- ✔ Interactive 4-tab Streamlit dashboard
- ✔ Refreshed README and docs; obsolete modules removed
