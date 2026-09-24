# ML-Based Demand & Renewable Energy Forecasting

A real-time, multi-horizon electricity-demand and renewable-energy forecasting system for India, extended into **two dispatch simulators** (a live real-time one and a what-if scenario one), renewable-battery storage accounting, CO₂ / backup-cost impact analysis, SHAP + context explainability of those calculations, and a Streamlit decision-support dashboard with an animated powerhouse view.

The system **learns from four aligned historical datasets**, trains a best model per target, and then **forecasts forward from the actual current moment** (anchored to the present hour in IST), producing hourly / daily / weekly / monthly projections with lower and upper bounds.

---

## 1. What this project does (plain language)

1. **Acquire** four datasets — electricity demand, weather, wind resource, solar potential.
2. **Preprocess** — inspect, clean, align all four on a common hourly timestamp, run EDA, and engineer leakage-safe features (hourly / daily / weekly / monthly patterns).
3. **Train** several candidate models per target and **auto-select the best** by validation RMSE.
4. **Forecast in real time** — starting at *now*, recursively project demand, solar and wind forward 12 months, with widening 95% intervals.
5. **Simulate dispatch** — renewable use → **battery auto-fill from surplus** → battery discharge → grid backup, with the renewable energy stored each hour tracked as its own column.
6. **Quantify impact** — CO₂ avoided and backup-fuel cost saved versus a no-storage counterfactual.
7. **Explain with SHAP + context** — explain the *dispatch, CO₂ and cost* calculations (not model selection), plus a plain-language "why" (season, festival, time-of-day, weather).
8. **Dashboard** — **two dispatch simulators**: a Main real-time simulator across **4 horizon tabs** (Hourly, Daily, Weekly, Monthly) and a Scenario what-if simulator; each with a ranged table (incl. renewable-stored + SOC), bounds/weather/festival charts, an **animated powerhouse** flow, and a point-wise explanation block.

---

## 2. System architecture

```text
        ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
        │ Demand (Kaggle│  │ Weather       │  │ Wind (NASA    │  │ Solar (PVGIS 6│
        │ hourly India) │  │ (Open-Meteo)  │  │ POWER WS10M)  │  │ 1 MWp Delhi)  │
        └───────┬───────┘  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
                └──────────────────┴───────┬───────────┴──────────────────┘
                                           ▼
                       build_aligned_dataset.py  (inner-join on hourly timestamp)
                                           │  → aligned_hourly_dataset.csv (46,728 × 23)
                                           ▼
                       run_aligned_eda.py  → reports/*.csv + figures
                                           ▼
                       build_forecast_features.py (leakage-safe lags/rolling/calendar + season/festival)
                                           │  → forecast_features.csv (46,560 × 58)
                                           ▼
                       BASELINE  evaluate_baselines.py (naive 1h/24h/168h/720h + linear regression)
                                           │  → reports/baseline_metrics.csv
                                           ▼
                       ADVANCED ML  prepare_ml_data.py → train_ml_models.py (RandomForest · GradientBoosting · XGBoost)
                                           │  → reports/ml_model_metrics.csv
                                           ▼
                       COMPARE  compare_models.py (rank all by RMSE) → reports/performance_comparison.csv
                                           ▼
                       SELECT  select_best_model.py (best deployable per target, refit on full data)
                                           │  → models/realtime_forecasters.joblib
                                           │  → reports/best_model.csv
                                           ▼
        live_weather.py (Open-Meteo, next 16 d) ──┐
                                                 ▼
                       realtime_forecast.py (recursive, anchored at NOW IST)
                                           │  → forecasts/hourly|daily|weekly|monthly_forecast.csv
                                           │  → forecasts/future_features.csv
                                           ▼
                       operations.py (facility scaling · dispatch · impact · renewable storage)
                                           ▼
                       scenario.py (what-if day: season · festival · weather)
                                           ▼
                       explain_dispatch.py (SHAP + context on dispatch / CO₂ / cost)
                                           ▼
                       app/dashboard.py (Main real-time tabs + Scenario simulator)
```

**Design rule:** the dashboard only *reads* cached CSVs and the persisted model pack. The expensive 12-month recursion is run once as a batch step; the UI never retrains or re-recurses on startup.

---

## 3. Project structure (current, cleaned)

```text
ML-Based-Demand-Renewable-Energy-Forecasting/
├── app/
│   ├── dashboard.py                  # two-simulator Streamlit UI
│   └── visuals.py                    # animated powerhouse + weather/festival charts
├── data/
│   ├── raw/                          # acquired sources (demand xlsx, weather, renewable/)
│   └── processed/
│       ├── demand_cleaned.csv
│       ├── solar_hourly.csv
│       ├── aligned_hourly_dataset.csv      # authoritative merged hourly table
│       ├── data_quality_summary.csv
│       ├── forecast_features.csv           # leakage-safe training features
│       ├── feature_manifest.json
│       └── forecasts/
│           ├── hourly_forecast.csv         # 8,760 h (12 months) lower/expected/upper
│           ├── daily_forecast.csv
│           ├── weekly_forecast.csv
│           ├── monthly_forecast.csv
│           └── future_features.csv         # features persisted for SHAP
├── models/
│   └── realtime_forecasters.joblib         # selected model + feature list per target
├── reports/
│   ├── realtime_model_metrics.csv          # MAE/RMSE/MAPE per candidate, selected flag
│   ├── *_eda_records.csv, *_pattern_profile.csv, correlation_matrix.csv
│   └── figures/
├── src/
│   ├── data/        inspect_demand · clean_demand · fetch_weather ·
│   │                download_pvgis_solar · clean_solar · build_aligned_dataset · live_weather
│   ├── renewable/   download_wind_data
│   ├── eda/         run_aligned_eda · pattern_analysis
│   ├── features/    build_forecast_features · calendar_features
│   ├── baseline/    naive_baseline · regression_baseline · evaluate_baselines
│   ├── models/      modeling_common · prepare_ml_data · train_ml_models · realtime_forecast
│   ├── comparison/  compare_models · validate_comparison
│   ├── selection/   select_best_model · validate_selection
│   ├── forecasting/ operations · scenario
│   └── explainability/ explain_dispatch
├── docs/            project_audit · progress · decisions · complete_overview ·
│                    Full_Project_Execution_Plan · Dataset_Research_Report
├── README.md
└── requirements.txt
```

---

## 4. File-by-file guide (purpose · tech stack · output)

| File | Purpose | Tech stack | Output |
|---|---|---|---|
| `src/data/inspect_demand.py` | Inspect raw demand workbook (shape, types, range, nulls, duplicates) | pandas | console report |
| `src/data/clean_demand.py` | Clean demand, parse datetimes, sort, add calendar columns | pandas | `data/processed/demand_cleaned.csv` |
| `src/data/fetch_weather.py` | Download Delhi historical hourly weather | requests, Open-Meteo API | `data/raw/weather_hourly.csv` |
| `src/data/download_pvgis_solar.py` | Download PVGIS 6 modelled 1 MWp Delhi solar potential | requests, PVGIS API | `data/raw/…solar…` |
| `src/data/clean_solar.py` | Normalise solar to hourly `solar_generation_kw` | pandas | `data/processed/solar_hourly.csv` |
| `src/renewable/download_wind_data.py` | Download NASA POWER `WS10M` hourly wind speed (Delhi) | requests, NASA POWER API | `data/raw/renewable/nasa_power_wind_hourly.csv` |
| `src/data/build_aligned_dataset.py` | Inner-join all four sources on hourly timestamp; validate continuity/nulls/duplicates; convert wind speed → power via 1 MW turbine curve | pandas | `aligned_hourly_dataset.csv`, `data_quality_summary.csv` |
| `src/eda/run_aligned_eda.py` | Hourly/daily/weekly/monthly demand & renewable patterns, correlations, dual-axis figures | pandas, matplotlib | `reports/*_eda_records.csv`, `*_pattern_profile.csv`, `figures/` |
| `src/eda/pattern_analysis.py` | Advanced patterns: hourly, weekday/weekend, regional, yearly, IMD season, Indian-festival effect, weather-band (humidity/temp/cloud/wind) vs demand & renewable | pandas, matplotlib | `reports/patterns/*.csv`, `reports/figures/patterns/*.png` |
| `src/features/calendar_features.py` | Single source of truth for IMD season + Indian festival calendar; emits season one-hot, `is_festival`, and `season`/`festival_name` labels | pandas | used by features, training, real-time engine |
| `src/features/build_forecast_features.py` | Cyclical calendar + season/festival + lag (1/24/168 h) + rolling (24/168 h) features; `shift(1)` before rolling to prevent leakage | numpy, pandas | `forecast_features.csv` (46,560 × 58), `feature_manifest.json` |
| `src/models/modeling_common.py` | Shared targets, feature groups, chronological split, MAE/RMSE/MAPE evaluation, and the model registry (LinearRegression, Ridge, RandomForest, HistGradientBoosting, XGBoost) | scikit-learn, xgboost | imported by all modelling stages |
| `src/baseline/evaluate_baselines.py` | Baseline benchmarks: seasonal-naive 1h/24h/168h/720h + linear regression, per target on the shared test split | scikit-learn, pandas | `baseline_predictions.csv`, `reports/baseline_metrics.csv`, `figures/baseline_comparison.png` |
| `src/models/prepare_ml_data.py` | Chronological 80/20 split of the feature dataset | pandas | `ml_train.csv`, `ml_test.csv` |
| `src/models/train_ml_models.py` | Train RandomForest · HistGradientBoosting · XGBoost per target; evaluate MAE/RMSE/MAPE | scikit-learn, xgboost | `ml_predictions.csv`, `reports/ml_model_metrics.csv`, `figures/ml_model_comparison.png` |
| `src/comparison/compare_models.py` | Merge baseline + ML metrics and rank every model by RMSE per target | pandas, matplotlib | `reports/performance_comparison.csv`, `figures/performance_comparison.png` |
| `src/selection/select_best_model.py` | Pick the best deployable model per target (RMSE→MAE→MAPE), refit on full data, persist the real-time pack | scikit-learn, xgboost, joblib | `reports/best_model.csv`, `figures/best_model_comparison.png`, `models/realtime_forecasters.joblib` |
| `src/data/live_weather.py` | Fetch live Delhi weather (next 16 days) to anchor the real-time forecast; falls back to climatology on failure | requests, Open-Meteo API | in-memory DataFrame |
| `src/models/realtime_forecast.py` | Recursive autoregressive multi-horizon forecast anchored at the current IST hour; live weather blended with month×hour normals; widening 95% intervals; aggregate to daily/weekly/monthly | numpy, pandas, joblib | `forecasts/hourly|daily|weekly|monthly_forecast.csv`, `future_features.csv` |
| `src/forecasting/operations.py` | Scale national MW → facility kW; run the 3-level (lower/expected/upper) dispatch simulator with renewable-battery auto-fill and a `renewable_stored_*` column; compute cost & CO₂ impact vs no-storage counterfactual | numpy, pandas | in-memory frames/dicts used by the dashboard |
| `src/forecasting/scenario.py` | What-if dispatch engine: build a representative 24 h day from a chosen season / festival / day-type / weather preset and forecast it with the same model pack | numpy, pandas, joblib | in-memory scenario frame + feature matrix |
| `src/explainability/explain_dispatch.py` | SHAP explanations of the dispatch / CO₂ / cost calculations for a chosen timestamp, plus a season/festival/weather/time-of-day context narrative | shap (LinearExplainer / TreeExplainer), joblib | explanation dicts rendered in the UI |
| `app/visuals.py` | Animated powerhouse HTML/CSS flow diagram and weather / festival-hourly Altair charts | altair | HTML + charts rendered in the UI |
| `app/dashboard.py` | Two dispatch simulators: Main real-time (Hourly/Daily/Weekly/Monthly tabs) + Scenario what-if; ranged tables (incl. renewable-stored + SOC), bounds/weather/festival charts, animated powerhouse, point-wise explanation | streamlit, altair, pandas | web UI |

---

## 5. Datasets

| Dataset | Source | Coverage | Role |
|---|---|---|---|
| Electricity demand | Kaggle — Hourly Load India | 2019-01-01 → 2024-04-30, hourly | Demand target (`national_demand_mw`) + regional columns |
| Weather | Open-Meteo (Delhi 28.6139, 77.2090) | same window, hourly | Forecast features (temp, humidity, cloud, precipitation, radiation, wind speed) |
| Wind resource | NASA POWER `WS10M` (Delhi) | same window, hourly | Converted to `wind_power_potential_kw` via 1 MW turbine curve |
| Solar potential | PVGIS 6, 1 MWp Delhi | same window, hourly | `solar_generation_kw` (modelled potential, not metered output) |

All four are inner-joined into **`aligned_hourly_dataset.csv` — 46,728 hourly rows × 23 columns**, with zero missing values and zero duplicate timestamps over the common window.

---

## 6. Models & selection

Each target trains three candidates and keeps the one with the lowest validation RMSE (chronological split, 9,312 test rows):

| Target | Selected model | RMSE | MAE | MAPE |
|---|---|---:|---:|---:|
| Demand (`national_demand_mw`) | **LinearRegression** | 3237.13 | 2485.34 | 1.36% |
| Solar (`solar_generation_kw`) | **XGBoost** | 28.39 | 11.87 | 22.3% |
| Wind (`wind_power_potential_kw`) | **XGBoost** | 6.44 | 0.43 | 1.11% |

Scikit-learn and XGBoost estimators are used (LinearRegression, XGBRegressor). The selected model, its exact feature list, algorithm name and residual std are persisted together in `models/realtime_forecasters.joblib`.

---

## 7. Real-time forecasting (how "from now" works)

- **Anchor:** `pd.Timestamp.now(tz="Asia/Kolkata").floor("h")` — the forecast starts at the current IST hour (the cached run is anchored `2026-09-22 07:00`).
- **Weather:** live Open-Meteo Delhi forecast for the next 16 days; beyond that, month×hour climatological normals from the aligned history.
- **Demand seed:** historical demand ends 2024-04-30, so the autoregressive buffers are seeded from the last 168 observed hours. *(Documented limitation: there is no live demand feed; demand patterns are learned, then projected.)*
- **Recursion:** each predicted hour feeds its own lag/rolling features forward (lags 1/24/168 h, rolling 24/168 h).
- **Intervals:** `margin = 1.96 × residual_std × sqrt(1 + 0.015 × (h−1))` → 95% bounds widen with horizon. Solar is forced to 0 when climatological radiation ≤ 0.
- **Horizon:** 8,760 hours (12 months) cached once, then aggregated to daily (366), weekly (53) and monthly (13).

> **Long-horizon note:** because demand is projected autoregressively for a full year from a 2024 seed, the monthly *expected* demand drifts downward over the 12-month horizon. The near-term (hourly/daily) forecast is the operationally meaningful range; treat far-month values as trend indicators, not precise levels.

---

## 8. Dispatch simulator, cost & CO₂

`operations.py` scales national values to a facility/microgrid (defaults: demand ×0.05 → kW, **solar ×14.0**, **wind ×200.0**; battery 5,000 kWh / 1,000 kW; round-trip efficiency 0.9). Solar is sized so a clear midday genuinely over-produces and the surplus **auto-fills the battery** (the storage story only exists if renewable can exceed demand). For each hour and each bound level:

```text
renewable = solar + wind
used      = min(renewable, demand)
surplus   = renewable − used            → charges battery (≤ power, ≤ free capacity)
deficit   = demand − used               → battery discharges (≤ power, ≤ available energy)
backup    = max(0, deficit − discharge) → grid / diesel backup
curtailed = max(0, surplus − charge)
```

The surplus stored each hour is exposed as `renewable_stored_<level>_kw` (plus a running `renewable_stored_cum_<level>_kwh`), so hourly/daily/weekly/monthly records report renewable storage directly.

`scenario.py` is the **second, what-if simulator**: pick a season, festival day, day type and weather preset (sunny / cloudy / rainy / windy / clear night); it builds a representative 24 h day — calendar features from the shared festival/season module, a diurnal solar-radiation and temperature curve, and historical month-by-hour analogs for the lag/rolling inputs — then forecasts it with the *same* model pack and dispatches it with the *same* battery hierarchy. The main real-time simulator runs independently of it.

**Impact** compares a no-storage counterfactual (renewable direct use only) against with-storage:

- `cost_savings = (backup_without_storage − backup_with_storage) × tariff` (default ₹6.52/kWh)
- `co2_avoided  = (backup_without_storage − backup_with_storage) × emission_factor` (default 0.710 kg CO₂/kWh)

Energy conservation is enforced exactly (verified 0.0 kW residual) and SOC stays within `[0, capacity]`.

---

## 9. SHAP explainability (of the calculations)

`explain_dispatch.py` explains **why the dispatch / CO₂ / cost numbers came out as they did** for a selected timestamp — *not* why a model was chosen. It loads the persisted model pack and `future_features.csv`, uses `LinearExplainer` for the linear demand model (500-row background) and `TreeExplainer` for the tree models, and returns the top drivers plus a point-wise narrative. On top of the SHAP drivers it adds a **context narrative** (season, festival, time-of-day, and weather such as cloud cover, heat, humidity, rain or wind) that answers "why is demand high/low right now". `explain_from_features` provides the same for scenario what-if hours not present in the real-time cache:

```text
Context:          Monsoon season, evening demand peak, Diwali festival day
Why:              heavy cloud cover suppresses solar; high temperature raises cooling demand …
Demand:           …
Renewable supply: …
Renewable used:   …
Renewable stored: … (surplus auto-filled into the battery)
Storage:          …
Backup:           …
Curtailed:        …
Dispatch order:   renewable → battery → grid
CO₂ avoided:      …
Cost saved:       …
```

---

## 10. Dashboard — two dispatch simulators

**Main real-time simulator** (4 horizon tabs):

| Tab | Window | Table | Charts | Simulator | Explanation |
|---|---|---|---|---|---|
| **Hourly** | next 24 h | demand / renewable supply / **renewable stored** / SOC / discharge / backup, in ranges | bounds chart + weather-context chart | pick an hour → animated powerhouse | point-wise SHAP + context block |
| **Daily** | next 7 days | same, daily ranges | same | pick a day | same |
| **Weekly** | next 4 weeks | same, weekly ranges | same | pick a week | same |
| **Monthly** | next 12 months | same, monthly ranges | same | pick a month | same |

**Scenario what-if simulator** (below the tabs): season / festival / weather / day-type selectors, an hour-by-hour demand-vs-renewable festival-shape chart, a scenario weather chart, the scenario table (with renewable-stored + SOC), the animated powerhouse for a chosen scenario hour, and a point-wise SHAP + context explanation.

The **animated powerhouse** (`app/visuals.py`) shows solar + wind sources, a filling/draining renewable-battery container with SOC %, and the backup source; flow arrows animate only when that hour's dispatch actually charges, discharges, or draws backup. Bound colours: demand `#e4572e`, renewable `#2ca02c`, backup `#1f77b4`. Facility sizing, battery spec, tariff and emission factor live in collapsible sidebar reference-assumption expanders (there is no scenario sidebar).

---

## 11. How to run

All commands run from the project root. The project uses the local virtual environment `.venv` (shap/streamlit are installed there, not in the global Python).

**Windows (Git Bash):**

```bash
# 0) one-time: create + install the environment
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

# 1) build the aligned dataset (after the raw sources are present)
.venv/Scripts/python.exe src/data/build_aligned_dataset.py

# 2) EDA + features
.venv/Scripts/python.exe src/eda/run_aligned_eda.py
.venv/Scripts/python.exe src/features/build_forecast_features.py

# 3) baseline benchmarks (naive 1h/24h/168h/720h + linear regression)
.venv/Scripts/python.exe -m src.baseline.evaluate_baselines

# 4) advanced ML models (RandomForest · GradientBoosting · XGBoost)
.venv/Scripts/python.exe -m src.models.prepare_ml_data
.venv/Scripts/python.exe -m src.models.train_ml_models

# 5) compare all models, then select + deploy the best per target
.venv/Scripts/python.exe -m src.comparison.compare_models
.venv/Scripts/python.exe -m src.comparison.validate_comparison
.venv/Scripts/python.exe -m src.selection.select_best_model
.venv/Scripts/python.exe -m src.selection.validate_selection

# 6) generate the real-time 12-month forecast cache (anchored at NOW; ~10 min)
.venv/Scripts/python.exe -m src.models.realtime_forecast --hours 8760

# 7) launch the dashboard
.venv/Scripts/streamlit.exe run app/dashboard.py
```

Then open the URL printed in the terminal (normally `http://localhost:8501`).

**PowerShell / CMD** equivalents use `.venv\Scripts\python.exe` and `.venv\Scripts\streamlit.exe`.

> Steps 1–4 are batch preparation. If `data/processed/forecasts/*.csv` and `models/realtime_forecasters.joblib` already exist, you can skip straight to step 5 — the dashboard reads the cache and does not retrain.

---

## 12. Key assumptions & limitations (read before review)

- **Delhi weather** is a representative hourly signal for an India-wide demand series; it does not claim to represent all of India.
- **Solar** is PVGIS-modelled potential for a 1 MWp Delhi system, not metered plant output.
- **Wind** is NASA POWER wind *speed* converted to power through a representative 1 MW turbine curve (cut-in 3, rated 12, cut-out 25 m/s), not measured generation.
- **Demand has no live feed** — history ends 2024-04-30; the real-time forecast projects learned patterns forward from the last observed state.
- **Facility scaling** (demand ×0.05, **solar ×14.0**, **wind ×200.0**) maps national values to a solar-heavy microgrid so a clear midday can over-produce and charge the battery; adjust via the sidebar expanders.
- **Tariff ₹6.52/kWh and emission factor 0.710 kg CO₂/kWh** are configurable scenario/reference values, not universal constants.
- **Long-horizon demand drift** — see §7.

---

## 13. Tech stack

Python 3.13 · pandas · numpy · scikit-learn (LinearRegression, Ridge, RandomForest, HistGradientBoosting) · xgboost · joblib · SHAP · Streamlit · Altair · matplotlib · requests (Open-Meteo, NASA POWER, PVGIS).

---

## 14. Documentation

- `docs/project_audit.md` — the REUSE → EXTEND → REFACTOR blueprint that drove this redesign.
- `docs/progress.md` — phase status against the new goal.
- `docs/decisions.md` — technical decisions and rationale.
- `docs/complete_overview.md` — end-to-end narrative.
- `docs/Full_Project_Execution_Plan.md` — execution plan.
- `docs/Dataset_Research_Report.md` — dataset sourcing research.

---

## 15. License

Developed for academic and research purposes.
