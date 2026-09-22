# ML-Based Demand & Renewable Energy Forecasting

A real-time, multi-horizon electricity-demand and renewable-energy forecasting system for India, extended into a scenario-based dispatch simulator, CO₂ / backup-cost impact analysis, SHAP explainability of those calculations, and a 4-tab Streamlit decision-support dashboard.

The system **learns from four aligned historical datasets**, trains a best model per target, and then **forecasts forward from the actual current moment** (anchored to the present hour in IST), producing hourly / daily / weekly / monthly projections with lower and upper bounds.

---

## 1. What this project does (plain language)

1. **Acquire** four datasets — electricity demand, weather, wind resource, solar potential.
2. **Preprocess** — inspect, clean, align all four on a common hourly timestamp, run EDA, and engineer leakage-safe features (hourly / daily / weekly / monthly patterns).
3. **Train** several candidate models per target and **auto-select the best** by validation RMSE.
4. **Forecast in real time** — starting at *now*, recursively project demand, solar and wind forward 12 months, with widening 95% intervals.
5. **Simulate dispatch** — for any selected hour/scenario, decide renewable use → battery charge/discharge → grid backup.
6. **Quantify impact** — CO₂ avoided and backup-fuel cost saved versus a no-storage counterfactual.
7. **Explain with SHAP** — explain the *dispatch, CO₂ and cost* calculations (not model selection).
8. **Dashboard** — exactly **4 tabs** (Hourly, Daily, Weekly, Monthly), each with a ranged table, one bounds chart, a scenario dispatch simulator, and a point-wise explanation block.

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
                       build_forecast_features.py (leakage-safe lags/rolling/calendar)
                                           │  → forecast_features.csv (46,560 × 51)
                                           ▼
                       train_realtime_models.py (3 targets × 3 candidates, auto-select)
                                           │  → models/realtime_forecasters.joblib
                                           │  → reports/realtime_model_metrics.csv
                                           ▼
        live_weather.py (Open-Meteo, next 16 d) ──┐
                                                 ▼
                       realtime_forecast.py (recursive, anchored at NOW IST)
                                           │  → forecasts/hourly|daily|weekly|monthly_forecast.csv
                                           │  → forecasts/future_features.csv
                                           ▼
                       operations.py (facility scaling · dispatch scenario · impact)
                                           ▼
                       explain_dispatch.py (SHAP on dispatch / CO₂ / cost)
                                           ▼
                       app/dashboard.py (4 tabs: Hourly · Daily · Weekly · Monthly)
```

**Design rule:** the dashboard only *reads* cached CSVs and the persisted model pack. The expensive 12-month recursion is run once as a batch step; the UI never retrains or re-recurses on startup.

---

## 3. Project structure (current, cleaned)

```text
ML-Based-Demand-Renewable-Energy-Forecasting/
├── app/
│   └── dashboard.py                  # 4-tab Streamlit UI
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
│   ├── eda/         run_aligned_eda
│   ├── features/    build_forecast_features
│   ├── models/      train_realtime_models · realtime_forecast
│   ├── forecasting/ operations
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
| `src/eda/run_aligned_eda.py` | Hourly/daily/weekly/monthly demand & renewable patterns, correlations, figures | pandas, matplotlib | `reports/*_eda_records.csv`, `*_pattern_profile.csv`, `figures/` |
| `src/features/build_forecast_features.py` | Cyclical calendar + lag (1/24/168 h) + rolling (24/168 h) features; `shift(1)` before rolling to prevent leakage | numpy, pandas | `forecast_features.csv`, `feature_manifest.json` |
| `src/models/train_realtime_models.py` | Train Ridge / RandomForest / HistGradientBoosting per target (demand, solar, wind); auto-select best by test RMSE; persist pack | scikit-learn, joblib | `models/realtime_forecasters.joblib`, `reports/realtime_model_metrics.csv` |
| `src/data/live_weather.py` | Fetch live Delhi weather (next 16 days) to anchor the real-time forecast; falls back to climatology on failure | requests, Open-Meteo API | in-memory DataFrame |
| `src/models/realtime_forecast.py` | Recursive autoregressive multi-horizon forecast anchored at the current IST hour; live weather blended with month×hour normals; widening 95% intervals; aggregate to daily/weekly/monthly | numpy, pandas, joblib | `forecasts/hourly|daily|weekly|monthly_forecast.csv`, `future_features.csv` |
| `src/forecasting/operations.py` | Scale national MW → facility kW; run the 3-level (lower/expected/upper) dispatch simulator; compute cost & CO₂ impact vs no-storage counterfactual | numpy, pandas | in-memory frames/dicts used by the dashboard |
| `src/explainability/explain_dispatch.py` | SHAP explanations of the dispatch / CO₂ / cost calculations for a chosen timestamp | shap (LinearExplainer / TreeExplainer), joblib | explanation dicts rendered in the UI |
| `app/dashboard.py` | 4-tab interactive dashboard (Hourly, Daily, Weekly, Monthly): ranged table, bounds chart, scenario dispatch simulator, point-wise explanation | streamlit, altair, pandas | web UI |

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
| Demand (`national_demand_mw`) | **Ridge** | 3253.33 | 2490.50 | 1.36% |
| Solar (`solar_generation_kw`) | **HistGradientBoosting** | 28.66 | 12.33 | 24.2% |
| Wind (`wind_power_potential_kw`) | **RandomForest** | 0.186 | 0.0088 | 0.004% |

Only scikit-learn estimators are used (Ridge, RandomForestRegressor, HistGradientBoostingRegressor). The selected model, its exact feature list, algorithm name and residual std are persisted together in `models/realtime_forecasters.joblib`.

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

`operations.py` scales national values to a facility/microgrid (defaults: demand ×0.05 → kW, solar ×5.0, wind ×1.0; battery 5,000 kWh / 1,000 kW; round-trip efficiency 0.9). For each hour and each bound level:

```text
renewable = solar + wind
used      = min(renewable, demand)
surplus   = renewable − used            → charges battery (≤ power, ≤ free capacity)
deficit   = demand − used               → battery discharges (≤ power, ≤ available energy)
backup    = max(0, deficit − discharge) → grid / diesel backup
curtailed = max(0, surplus − charge)
```

**Impact** compares a no-storage counterfactual (renewable direct use only) against with-storage:

- `cost_savings = (backup_without_storage − backup_with_storage) × tariff` (default ₹6.52/kWh)
- `co2_avoided  = (backup_without_storage − backup_with_storage) × emission_factor` (default 0.710 kg CO₂/kWh)

Energy conservation is enforced exactly (verified 0.0 kW residual) and SOC stays within `[0, capacity]`.

---

## 9. SHAP explainability (of the calculations)

`explain_dispatch.py` explains **why the dispatch / CO₂ / cost numbers came out as they did** for a selected timestamp — *not* why a model was chosen. It loads the persisted model pack and `future_features.csv`, uses `LinearExplainer` for Ridge (500-row background) and `TreeExplainer` for the tree models, and returns the top drivers plus a point-wise narrative:

```text
Demand:           …
Renewable supply: …
Renewable used:   …
Storage:          …
Backup:           …
Curtailed:        …
Dispatch order:   renewable → battery → grid
CO₂ avoided:      …
Cost saved:       …
```

---

## 10. Dashboard — exactly 4 tabs

| Tab | Window | Table | Chart | Simulator | Explanation |
|---|---|---|---|---|---|
| **Hourly** | next 24 h | demand / renewable supply / storage / backup, in ranges | one bounds chart (upper+lower, colour-coded) | pick an hour → dispatch scenario | point-wise SHAP block |
| **Daily** | next 7 days | same, daily ranges | same | pick a day | same |
| **Weekly** | next 4 weeks | same, weekly ranges | same | pick a week | same |
| **Monthly** | next 12 months | same, monthly ranges | same | pick a month | same |

Bound colours: demand `#e4572e`, renewable `#2ca02c`, backup `#1f77b4`. Sidebar sliders control demand/solar/wind scale, battery capacity/SOC/power, tariff and emission factor.

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

# 3) train + auto-select best model per target
.venv/Scripts/python.exe src/models/train_realtime_models.py

# 4) generate the real-time 12-month forecast cache (anchored at NOW; ~10 min)
.venv/Scripts/python.exe src/models/realtime_forecast.py

# 5) launch the dashboard
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
- **Facility scaling** (demand ×0.05, solar ×5.0, wind ×1.0) maps national values to a microgrid-sized scenario; adjust via the sidebar.
- **Tariff ₹6.52/kWh and emission factor 0.710 kg CO₂/kWh** are configurable scenario/reference values, not universal constants.
- **Long-horizon demand drift** — see §7.

---

## 13. Tech stack

Python 3.13 · pandas · numpy · scikit-learn (Ridge, RandomForest, HistGradientBoosting) · joblib · SHAP · Streamlit · Altair · matplotlib · requests (Open-Meteo, NASA POWER, PVGIS).

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
