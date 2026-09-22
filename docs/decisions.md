# Technical Decisions

Rationale behind the redesigned real-time forecasting system.

## Data

| Decision | Reason |
|---|---|
| Inner-join all four sources on an exact hourly index | Guarantees demand, weather, wind and solar refer to the same timestamp; fails fast on any gap or duplicate. |
| Single authoritative `aligned_hourly_dataset.csv` | One reproducible source of truth replaces the earlier chained merge/validate artefacts. |
| Delhi weather as the representative signal | Provides a consistent hourly series for an India-wide demand target without claiming nationwide representation. |
| PVGIS 6 modelled 1 MWp solar potential | Gives a complete, gap-free hourly solar series for the full window; explicitly labelled modelled potential, not metered output. |
| NASA POWER wind speed → 1 MW turbine curve | The raw data is meteorological wind speed, not generation; a documented power curve (cut-in 3, rated 12, cut-out 25 m/s) converts it to `wind_power_potential_kw`. |

## Features

| Decision | Reason |
|---|---|
| Cyclical encoding of hour / weekday / month | Preserves proximity across periodic boundaries (23:00 ≈ 00:00). |
| Lags 1 / 24 / 168 h and rolling 24 / 168 h | Captures autoregressive, daily and weekly structure for all three targets. |
| `shift(1)` before every rolling window | Prevents target leakage — rolling statistics never see the current hour's target. |
| Build features for demand, solar and wind together | One consistent feature frame trains all targets and is persisted for SHAP. |

## Models

| Decision | Reason |
|---|---|
| Three candidates per target (Ridge, RandomForest, HistGradientBoosting) | Lets the data pick the model rather than assuming one algorithm fits all targets. |
| Auto-select by validation RMSE on a chronological split | RMSE penalises large errors; chronological split simulates real past→future deployment. |
| scikit-learn only | No xgboost / statsmodels / lightgbm in the environment; sklearn covers the required estimators. |
| Persist model + feature list + algorithm + residual std in one joblib pack | The forecaster and SHAP explainer must use the exact training features and residual scale. |

## Real-time forecasting

| Decision | Reason |
|---|---|
| Anchor at `now` in Asia/Kolkata, floored to the hour | The user requirement is forecasting "from this moment onwards". |
| Recursive autoregressive projection | Multi-horizon forecasts feed their own predictions into the lag/rolling features. |
| Live Open-Meteo weather for 16 days, then month×hour climatology | Live feed covers the near term; climatology extends smoothly to 12 months without a second API. |
| Seed demand from last 168 observed hours | No live demand feed exists; the most recent week is the best available autoregressive state. |
| Widening 95% intervals `1.96·σ·√(1+0.015·(h−1))` | Uncertainty must grow with horizon; residual std comes from the persisted pack. |
| Cache the 8,760-hour run; dashboard only reads CSVs | The recursion is slow (~10 min); the UI must stay instant and never retrain. |

## Dispatch, cost & CO₂

| Decision | Reason |
|---|---|
| Facility scaling (demand ×0.05, solar ×5.0, wind ×1.0) | Maps national MW to a microgrid-sized scenario the battery can actually serve. |
| Dispatch order renewable → battery → grid | Reflects real merit-order: use free renewable first, then storage, then paid backup. |
| Simulate all three bound levels | Lower/expected/upper dispatch gives a risk range, not a single number. |
| No-storage counterfactual for impact | Isolates the benefit attributable to storage: avoided backup × tariff and × emission factor. |
| Configurable tariff (₹6.52/kWh) and emission factor (0.710 kg CO₂/kWh) | Scenario/reference values that a reviewer can adjust. |
| Allocate each dispatch array separately | An earlier aliasing bug (one shared zeros array) broke conservation; separate arrays conserve energy exactly. |

## Explainability

| Decision | Reason |
|---|---|
| SHAP explains dispatch / CO₂ / cost, not model selection | Explicit user requirement — explain the calculations the dashboard presents. |
| LinearExplainer for Ridge, TreeExplainer for trees | Correct explainer per selected algorithm. |
| Explain from persisted `future_features.csv` | Guarantees the explanation uses the same features the forecast used, at the chosen timestamp. |

## Dashboard

| Decision | Reason |
|---|---|
| Exactly four tabs (Hourly / Daily / Weekly / Monthly) | Explicit user requirement, replacing the earlier 9-view layout. |
| One bounds chart per tab with colour-coded bands | Demand / renewable / backup upper-lower bounds in distinct colours. |
| Ranged table (demand / renewable supply / storage / backup) | Shows the interval, not just a point estimate. |
| Scenario simulator + point-wise explanation per tab | Selecting an hour/day/week/month recomputes that scenario and its SHAP narrative. |
| `st.tabs([...])` API | The singular `st.tab` does not exist in the installed Streamlit version. |
