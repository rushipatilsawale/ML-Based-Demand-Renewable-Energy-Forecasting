# Project Progress

Status of the redesigned, real-time forecasting system. The legacy 14-phase linear build (separate baseline / time-series / comparison / selection / uncertainty / storage / impact modules) was consolidated into the streamlined pipeline below.

## Phase status

| # | Phase | Key artefact(s) | Status |
|--:|---|---|---|
| 1 | Data acquisition (demand, weather, wind, solar) | `data/raw/…`, `demand_cleaned.csv`, `solar_hourly.csv`, `nasa_power_wind_hourly.csv` | ✅ Completed |
| 2 | Preprocessing & timestamp alignment | `aligned_hourly_dataset.csv` (46,728 × 23), `data_quality_summary.csv` | ✅ Completed |
| 3 | EDA & pattern analysis (hourly/daily/weekly/monthly) | `reports/*_eda_records.csv`, `*_pattern_profile.csv`, `figures/` | ✅ Completed |
| 4 | Leakage-safe feature engineering | `forecast_features.csv` (46,560 × 51), `feature_manifest.json` | ✅ Completed |
| 5 | Model training & auto-selection (3 targets × 3 candidates) | `models/realtime_forecasters.joblib`, `reports/realtime_model_metrics.csv` | ✅ Completed |
| 6 | Live weather feed (Open-Meteo, real-time anchor) | `src/data/live_weather.py` | ✅ Completed |
| 7 | Real-time recursive multi-horizon forecast (now → +12 months) | `forecasts/hourly|daily|weekly|monthly_forecast.csv`, `future_features.csv` | ✅ Completed |
| 8 | Scenario dispatch simulator (3 bound levels) | `src/forecasting/operations.py` | ✅ Completed |
| 9 | CO₂ avoided & backup-cost impact | `operations.impact()` | ✅ Completed |
| 10 | SHAP explainability of dispatch / CO₂ / cost | `src/explainability/explain_dispatch.py` | ✅ Completed |
| 11 | 4-tab dashboard (Hourly · Daily · Weekly · Monthly) | `app/dashboard.py` | ✅ Completed |
| 12 | Structure cleanup & documentation refresh | this docs set, rewritten `README.md` | ✅ Completed |

## Selected models

| Target | Model | RMSE | MAE | MAPE |
|---|---|---:|---:|---:|
| Demand | LinearRegression | 3237.13 | 2485.34 | 1.36% |
| Solar | XGBoost | 28.39 | 11.87 | 22.3% |
| Wind | XGBoost | 6.44 | 0.43 | 1.11% |

## Verified behaviour

- Aligned dataset: 46,728 continuous hourly rows, 0 nulls, 0 duplicate timestamps.
- Dispatch energy conservation residual: 0.0 kW; SOC bounded within `[0, capacity]`.
- Real-time anchor: forecast starts at the current IST hour (cached run anchored to live time in IST).
- Live weather feed verified reachable (Open-Meteo, status 200).
- Dashboard renders all four tabs with ranged table, bounds chart, scenario simulator, 7 metric summary cards, animated powerhouse topology, and SHAP explanation.

## Modular pipeline structure

Active stages: `src/data/`, `src/eda/`, `src/features/`, `src/baseline/`, `src/models/`, `src/comparison/`, `src/selection/`, `src/forecasting/`, `src/explainability/`.
Legacy unintegrated scripts (standalone ARIMA/SARIMA, old single-target explainers, static storage scripts) have been clean-merged into this modular flow.

## Known limitations

- No live demand feed — history ends 2024-04-30; demand is projected from learned patterns seeded by the last 168 observed hours.
- Solar is PVGIS-modelled potential, not metered output; wind is speed converted through a representative turbine curve.
- Delhi weather is a representative signal for an India-wide demand series.
- Autoregressive demand drifts downward over the full 12-month horizon; near-term (hourly/daily) values are the operationally meaningful range.
- Tariff (₹6.52/kWh) and emission factor (0.710 kg CO₂/kWh) are configurable scenario assumptions.
