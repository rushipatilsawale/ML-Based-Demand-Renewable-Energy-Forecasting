# Comprehensive Project Audit & Target Architecture Blueprint

**Project**: ML-Based Demand & Renewable Energy Forecasting and Real-Time Energy Management System  
**Audit Date**: September 2026  
**Auditor**: Antigravity Technical Audit Agent  
**Status**: Pre-Implementation Deep Audit (No Code Modified)

---

## 1. Current Architecture

The current repository represents a completed 14-phase machine-learning pipeline that progresses linearly from raw data ingestion to a static Streamlit dashboard. 

```text
[Raw Demand & Weather]
         ↓
Phase 1: Data Ingestion & Merging (final_merged_dataset.csv)
         ↓
Phase 2: Exploratory Data Analysis (reports/figures/*.png)
         ↓
Phase 3: Feature Engineering (featured_dataset.csv: cyclical, lags, rolling)
         ↓
Phase 4: Baseline Models (Naive 24h/168h, Linear Regression)
         ↓
Phase 5: Machine Learning Models (Random Forest, Gradient Boosting, XGBoost)
         ↓
Phase 6: Time-Series Models (ARIMA, SARIMA)
         ↓
Phase 7: Performance Comparison (reports/performance_comparison.csv)
         ↓
Phase 8: Best Model Selection (Linear Regression selected by test RMSE)
         ↓
Phase 9: SHAP Explainability (LinearExplainer on 19 features)
         ↓
Phase 10: Renewable Forecasting (Solar: AC power; Wind: NASA wind speed → power curve)
         ↓
Phase 11: Uncertainty Analysis (Residual 95% intervals for solar & wind test sets)
         ↓
Phase 12: Storage Simulation (5 MWh BESS vs. Solar vs. Synthetic sine wave demand)
         ↓
Phase 13: Cost & CO₂ Analysis (₹6.52/kWh, 0.716 kg CO₂/kWh applied to Phase 12 backup)
         ↓
Phase 14: Final Dashboard (Streamlit app reading static CSVs)
```

### End-to-End Operational Flow
1. **Demand Data Cleaning & Integration**: Hourly Indian national and regional electricity demand (2019-01-01 to 2024-04-30) was cleaned and merged with hourly Delhi weather variables into `data/processed/final_merged_dataset.csv` (46,728 rows × 19 columns).
2. **Feature Engineering**: Generated cyclical encodings (`hour_sin`, `month_cos`, etc.), 3 demand lags (1h, 24h, 168h), and 4 rolling features (24h/168h mean and std) in `data/processed/featured_dataset.csv`.
3. **Model Evaluation & Selection**: Models were trained on the first 80% (37,248 hours) and tested on the last 20% (9,312 hours: April 2023 – April 2024). Linear Regression achieved the best 1-step RMSE (3,325 MW, 1.38% MAPE) and was selected.
4. **Renewables Pipeline**: 
   - Solar: 34 days of inverter AC power from two plants in India (May 15 – June 17, 2020) were aggregated to plant hourly generation (`solar_hourly.csv`) and modeled with Linear Regression using temporal/lag features.
   - Wind: NASA POWER hourly wind speed for Delhi (2019–2024) was modeled with Linear Regression, and test predictions were converted to estimated wind-power potential using a representative 1,000 kW turbine power curve.
5. **Downstream Simulation**:
   - Residual-based 95% confidence intervals were generated for solar and wind test periods.
   - Battery storage (5,000 kWh) was simulated using solar test predictions and an artificial mathematical demand curve (`1800 + 600*sin(...) + 300*(...)`).
   - Cost and emissions reductions were computed on the resulting battery backup requirements.
   - A multi-page Streamlit application (`app/dashboard.py`) was created to visualize these static CSV outputs.

---

## 2. Existing Components That Are Correct (Keep & Reuse)

The project has established several high-quality, scientifically sound components that must be preserved and reused:

1. **Demand & Weather Preprocessing**:
   - `src/data/inspect_demand.py` and `src/data/clean_demand.py`: Correctly parse and validate 46,728 continuous hourly records without missing values or duplicate timestamps.
   - `src/data/merge_data.py` and `src/data/validate_merged.py`: Cleanly integrate historical load with 6 weather variables (temperature, relative humidity, cloud cover, precipitation, wind speed, solar radiation).
2. **Foundational Feature Engineering**:
   - `src/features/create_time_features.py`: Robust cyclical trigonometric encodings ($2\pi \cdot t / T$) for diurnal, weekly, and annual cycles.
   - `src/features/create_lag_features.py` and `src/features/create_rolling_features.py`: Correctly shift targets to prevent future data leakage.
3. **Machine Learning Model Implementations**:
   - `src/models/train_random_forest.py`, `train_gradient_boosting.py`, `train_xgboost.py`: Properly configured tree ensembles saved in `models/`.
   - `src/baseline/regression_baseline.py`: Linear Regression benchmark.
4. **SHAP Explainability**:
   - `src/explainability/explain_model.py`: Rigorous global feature importance analysis showing feature weights (lags, solar radiation, diurnal cycles).
5. **Wind Resource Processing & Power Curve**:
   - `src/renewable/download_wind_data.py`: NASA POWER hourly wind speed dataset (2019–2024).
   - `src/renewable/estimate_wind_power.py`: Standard cubic power curve with cut-in (3 m/s), rated (12 m/s), and cut-out (25 m/s) speeds for a 1,000 kW reference turbine.
6. **Physical Battery Storage Equations**:
   - `src/storage/simulate_storage_backup.py`: The underlying state-of-charge (SOC) tracking math with charging/discharging efficiency losses ($\eta_c = 0.9, \eta_d = 0.9$) and power limits ($P_{\max} = 1000\text{ kW}$) is physically sound.
7. **Economic & Emissions Mathematical Formulations**:
   - `src/impact/calculate_cost_co2.py`: The formulas for cost ($\text{Backup kWh} \times \text{Tariff}$) and emissions ($\text{Backup kWh} \times \text{Emission Factor}$) are correct and adhere to CEA baseline benchmarks.
8. **Streamlit UI Layout & Styling**:
   - `app/dashboard.py`: Page configuration, metrics styling, and navigation architecture form an excellent visual foundation.

---

## 3. Missing Components (Gaps Relative to Master Goal)

To realize the intended decision-support system, the following components must be built:

1. **Multi-Time-Scale Trend Learning & Analysis**:
   - An explicit analytical engine that analyzes and summarizes how electricity demand changes across **Hourly (diurnal peaks/valleys)**, **Daily (weekday vs. weekend)**, **Weekly**, **Monthly (summer vs. winter peaks)**, and **Yearly (annual growth from 2019 to 2024)**.
2. **Forward Multi-Horizon Forecasting Engine**:
   - Current models only evaluated 1-step ahead test predictions with ground-truth lags.
   - Missing: A multi-horizon forecasting engine that takes the **latest available historical state** and forecasts:
     - **Hourly (Next 1 to 24 Hours)**
     - **Daily (Next 7 Days)**
     - **Weekly & Monthly (Next 30 Days)**
3. **Demand Prediction Ranges (Uncertainty Bounds)**:
   - Phase 11 only calculated prediction intervals for solar and wind.
   - **Demand prediction intervals (Lower Bound, Expected Forecast, Upper Bound)** are completely missing!
4. **Weather Integration for Solar Forecasting**:
   - `prepare_solar_data.py` ignored raw solar weather sensor data (`Plant_1_Weather_Sensor_Data.csv` containing irradiance and ambient temperature).
   - Missing: Solar models trained directly on solar irradiance and temperature regressors.
5. **Combined Renewable Forecasting (Solar + Wind)**:
   - Missing: A unified renewable forecasting stream that combines solar generation and estimated wind power potential with confidence ranges across hourly, daily, weekly, and monthly scales.
6. **Multi-Resource Data Alignment**:
   - Missing: A scientifically aligned dataset where Demand, Weather, Solar, and Wind coexist (specifically during the May 15 – June 17, 2020 window) to test integrated dispatch.
7. **Real-Time / Latest-Available Forecasting Pipeline**:
   - A modular pipeline: `Ingest Latest State → Generate Lag/Cyclical Features → Predict Multi-Horizon Demand & Renewables → Estimate Uncertainty → Pass to Dispatch`.
8. **Forecast-Driven Energy Balance & Dispatch Simulator**:
   - Real-time comparison: $\text{Energy Balance}_t = \text{Renewable Forecast}_t - \text{Demand Forecast}_t$.
   - Dispatch hierarchy: Renewable $\rightarrow$ BESS Discharge $\rightarrow$ Grid/Backup, plus recording **Curtailed / Surplus Renewable Energy**.
9. **Dynamic Cost & Avoided CO₂ Engine**:
   - Dynamic calculation reacting to user-adjustable tariffs, grid emission factors, and forecast horizons.
10. **Decision-Focused Dashboard Redesign**:
    - Replacing static CSV views with interactive forecast horizon selectors, what-if sliders, and future dispatch/curtailment charts.

---

## 4. Incorrect or Misaligned Components (Audit & Reuse Strategy)

| Component | Current Implementation | Flaw / Misalignment | Refactor / Reuse Strategy |
| :--- | :--- | :--- | :--- |
| **Storage Demand Input** | `create_demand_scenario()` in `simulate_storage_backup.py` used `1800 + 600*sin(...) + 300*(...)` | Synthetic mathematical equation completely decoupled from real demand models | **Refactor**: Remove synthetic equation; feed actual forecasted demand (scaled to facility capacity). |
| **Renewable Input to BESS** | Solar generation only | Ignored wind power potential completely | **Extend**: Combine predicted solar generation (kW) + estimated wind power potential (kW) into total renewable supply. |
| **Forecasting Method** | Static 1-step test evaluation using ground-truth `demand_lag_1h` | Cannot forecast into future horizons where future demand is unknown | **Refactor**: Implement iterative autoregressive multi-step forecasting updating lags from previous predictions. |
| **Solar Feature Set** | Calendar features + solar lags only | Ignored raw solar sensor data (`IRRADIATION`, `AMBIENT_TEMPERATURE`) | **Refactor**: Merge weather sensor data into solar preprocessing; train weather-driven solar model. |
| **Wind Terminology & Use** | Meteorological wind speed used as generation proxy | Wind speed is in m/s; cannot be directly subtracted from electricity demand (kW/MW) | **Clarify & Maintain**: Always convert wind speed $\rightarrow$ turbine power curve $\rightarrow$ estimated wind-power potential (kW); label clearly. |
| **Dashboard Operation** | Reads static hardcoded CSVs generated during development | Does not run forecasts, cannot adjust horizons, cannot evaluate what-if scenarios | **Refactor**: Import forecasting, dispatch, and impact modules dynamically into Streamlit with interactive controls. |

---

## 5. Data Compatibility Audit

```text
Dataset Overview:
┌────────────────────────┬──────────────────────┬─────────────┬───────────┬──────────────┬────────────────────────┐
│ Dataset Name           │ Date Range           │ Frequency   │ Rows      │ Missing Data │ Classification         │
├────────────────────────┼──────────────────────┼─────────────┼───────────┼──────────────┼────────────────────────┤
│ Hourly Load India      │ 2019-01-01 to 2024-04│ Hourly      │ 46,728    │ 0 (Clean)    │ Recorded Demand (MW)   │
│ Delhi Weather          │ 2019-01-01 to 2024-04│ Hourly      │ 46,728    │ 0 (Clean)    │ Recorded Weather       │
│ NASA POWER Wind Speed  │ 2019-01-01 to 2024-04│ Hourly      │ 46,728    │ 0 (Clean)    │ Resource Proxy (m/s)   │
│ Solar Plant Generation │ 2020-05-15 to 2020-06│ 15-Minute   │ 136,476*  │ Resampled    │ Recorded Gen (kW)      │
│ Solar Weather Sensors  │ 2020-05-15 to 2020-06│ 15-Minute   │ 6,441*    │ Resampled    │ Recorded Solar Weather │
└────────────────────────┴──────────────────────┴─────────────┴───────────┴──────────────┴────────────────────────┘
* Across Plant 1 and Plant 2.
```

### Key Compatibility Findings:
1. **Full 5-Year Temporal Overlap (Jan 2019 – Apr 2024)**:
   - National Demand, Delhi Weather, and NASA POWER Wind Speed share identical hourly timestamps across 46,728 hours.
   - Long-term demand pattern learning, 5-year trend analysis, and wind resource forecasting can be performed over this entire period.
2. **Synchronized 34-Day Overlap (May 15 – June 17, 2020)**:
   - Solar Plant 1 and Plant 2 generation and weather sensor data cover 34 days (816 hours).
   - During this exact 34-day window, Demand, Weather, Wind Speed, Solar Generation, and Solar Irradiance **all concurrently exist**.
   - This provides a scientifically legitimate, synchronized multi-resource window for end-to-end dispatch simulation without fabricating timestamps.
3. **Capacity & Unit Scaling Alignment**:
   - National demand is in **MW** (mean ~160,000 MW, peak ~237,000 MW).
   - Solar plant AC generation is in **kW** (sum of inverters peak ~30,000 kW = 30 MW).
   - Wind power curve reference turbine is **1,000 kW (1 MW)**.
   - In battery storage and dispatch, combining 160,000 MW demand with 30 MW solar and 1 MW wind on a 5 MWh battery would mean renewables are a rounding error ($<0.02\%$).
   - **Resolution**: Provide a **Facility / Microgrid Scale Mode** (e.g. 10,000 kW peak demand, 3,000 kW solar, 1,000 kW wind, 5,000 kWh battery) with interactive scaling sliders, alongside a Regional Grid Percentage view.

---

## 6. Model Audit

| Model | Target Variable | Features Used | Training Window | Forecasting Capabilities | Multi-Horizon Reusability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** (`models/linear_regression.pkl`) | `national_demand_mw` | 19 features (cyclical, lags 1/24/168h, rolling 24/168h, weather) | 2019-01 to 2023-04 (37,248 hrs) | Best 1-step RMSE (3,325 MW, 1.38% MAPE) | Highly reusable; can be deployed autoregressively for 24h, 7d, 30d horizons. |
| **XGBoost** (`models/xgboost.pkl`) | `national_demand_mw` | 19 features (same as above) | 2019-01 to 2023-04 (37,248 hrs) | Strong nonlinear benchmark (RMSE 4,777 MW) | Highly reusable for multi-step ensemble comparison. |
| **Random Forest** (`models/random_forest.pkl`) | `national_demand_mw` | 19 features | 2019-01 to 2023-04 | High accuracy (RMSE 4,203 MW) | Reusable; large file size (257 MB). |
| **Gradient Boosting** (`models/gradient_boosting.pkl`) | `national_demand_mw` | 19 features | 2019-01 to 2023-04 | Solid performance (RMSE 4,212 MW) | Reusable. |
| **ARIMA(2,1,2)** (`models/arima.pkl`) | `national_demand_mw` | Univariate demand | 2019-01 to 2023-04 | RMSE 23,292 MW | Reusable for classical baseline comparison. |
| **SARIMA(1,1,1)(1,1,1,24)** (`models/sarima.pkl`) | `national_demand_mw` | Univariate demand (90 days) | Recent 90 days of train set | RMSE 138,561 MW (poor test fit) | Deprecate as primary forecaster; keep as benchmark. |
| **Solar Regressor** (`models/solar/solar_linear_regression.pkl`) | `solar_generation_ac` | Hour, day of week, month, hour sin/cos, solar lags 1/24h, rolling 24h | May 15 – June 10, 2020 (80%) | MAE 117.8 kW, RMSE 275.6 kW | Needs retraining with `IRRADIATION` and `AMBIENT_TEMPERATURE`. |
| **Wind Regressor** (`models/wind/wind_linear_regression.pkl`) | `wind_speed_10m_ms` | Hour, day of week, month, hour sin/cos, wind lags 1/24h, rolling 24h | 2019-01 to 2023-04 (80%) | MAE 0.52 m/s, RMSE 0.69 m/s | Highly reusable; predict wind speed $\rightarrow$ apply turbine power curve. |

---

## 7. Uncertainty Audit

### Current Status:
- Phase 11 implemented residual-based 95% prediction intervals:
  $$\text{Interval} = \hat{y} \pm 1.96 \times \sigma_{\text{residuals}}$$
- Applied only to solar and wind test sets (`solar_uncertainty.csv`, `wind_uncertainty.csv`).
- **Demand uncertainty was completely omitted**.

### Required Enhancements:
1. **Demand Uncertainty**:
   - Compute test-set residual standard deviation for the selected demand model ($\sigma_{\text{demand}} \approx 3,325\text{ MW}$).
   - For multi-step horizons, compute horizon-dependent standard errors:
     $$\text{Margin}_h = 1.96 \times \sigma_{\text{demand}} \times \sqrt{1 + \alpha \cdot (h-1)}$$
     where $h$ is forecast step in hours, providing realistic widening fan charts for 24h, 7d, and 30d forecasts.
2. **Solar Uncertainty**:
   - Apply non-negative lower bound clipping ($\max(0, \text{Lower})$) and nighttime zeroing when solar radiation is zero.
3. **Wind Uncertainty**:
   - Propagate wind speed intervals $[\hat{v}_{\text{lower}}, \hat{v}_{\text{expected}}, \hat{v}_{\text{upper}}]$ through the monotonic power curve to produce exact $[\hat{P}_{\text{lower}}, \hat{P}_{\text{expected}}, \hat{P}_{\text{upper}}]$ in kW.

---

## 8. Dispatch Audit

### Current Status:
- In `simulate_storage_backup.py`, dispatch logic used:
  ```python
  # Synthetic demand:
  demand = 1800 + 600 * np.sin(2 * np.pi * (hour - 7) / 24) + 300 * ((hour >= 18) & (hour <= 22))
  ```
  and subtracted only solar generation. Wind was excluded.

### Target Dispatch Logic:
1. **Input Streams**:
   - $\hat{D}_t$: Forecasted demand (kW/MW)
   - $\hat{S}_t$: Forecasted solar generation (kW/MW)
   - $\hat{W}_t$: Forecasted wind power potential (kW/MW)
   - $\hat{R}_t = \hat{S}_t + \hat{W}_t$: Total renewable supply
2. **Dispatch Hierarchy**:
   - **Direct Renewable Utilization**:
     $$\text{Renewable Used}_t = \min(\hat{R}_t, \hat{D}_t)$$
   - **Surplus Renewable Energy**:
     $$\text{Surplus}_t = \max(0, \hat{R}_t - \hat{D}_t)$$
   - **Deficit Demand**:
     $$\text{Deficit}_t = \max(0, \hat{D}_t - \hat{R}_t)$$
   - **Battery Charging** (from Surplus up to $P_{\max}^{\text{chg}}$ and capacity limit):
     $$\text{Charge Input}_t = \min\left(\text{Surplus}_t, P_{\max}^{\text{chg}}, \frac{C_{\text{battery}} - \text{SOC}_t}{\eta_c}\right)$$
     $$\text{SOC}_{t+1} = \text{SOC}_t + (\text{Charge Input}_t \times \eta_c)$$
   - **Battery Discharging** (to meet Deficit up to $P_{\max}^{\text{dis}}$ and available SOC):
     $$\text{Discharge Avail}_t = \min(P_{\max}^{\text{dis}}, \text{SOC}_t \times \eta_d)$$
     $$\text{Storage Discharge}_t = \min(\text{Deficit}_t, \text{Discharge Avail}_t)$$
     $$\text{SOC}_{t+1} = \text{SOC}_t - \frac{\text{Storage Discharge}_t}{\eta_d}$$
   - **Remaining Backup / Grid Requirement**:
     $$\text{Grid Backup}_t = \max(0, \text{Deficit}_t - \text{Storage Discharge}_t)$$
   - **Curtailed Energy** (Surplus remaining when battery is full):
     $$\text{Curtailed}_t = \max(0, \text{Surplus}_t - \text{Charge Input}_t)$$

---

## 9. Dashboard Audit

### Existing Features to Preserve:
- Streamlit application framework (`st.set_page_config`, sidebar radio navigation).
- KPI metric card presentation (`st.metric`).
- Responsive tabular views (`st.dataframe`).
- Project Limitations documentation section.

### Components Requiring Redesign:
1. **Dynamic Execution**: Replace static `pd.read_csv("data/processed/*.csv")` with dynamic function calls that execute forecasting and dispatch models on demand.
2. **Navigation Structure**:
   - **1. Overview**: Executive summary with latest state, upcoming forecast totals, and dispatch summary.
   - **2. Historical Trend Analysis**: Multi-scale views (Hourly, Daily, Weekly, Monthly, Yearly trends 2019–2024).
   - **3. Demand Forecast**: Multi-horizon tabs (Hourly 1–24h, Daily 7d, Weekly, Monthly) with lower, expected, and upper fan charts.
   - **4. Renewable Forecast**: Breakdown of Solar and Wind potential with confidence intervals.
   - **5. Weather Impact & Sensitivity**: Temperature, radiation, humidity, and cloud cover effects on demand and generation.
   - **6. Energy Dispatch & Balance**: Stacked area chart showing Demand vs. Renewable Used, Battery Discharge, and Grid Backup.
   - **7. Battery / Storage Operations**: Dynamic SOC tracking, charging power, discharging power, and curtailment.
   - **8. Cost & CO₂ Impact**: Financial savings and avoided grid emissions under configurable tariffs.
   - **9. Explainability & Model Lineage**: SHAP feature importance, model specifications, and data classification flags.
3. **Interactive Sidebar Controls**:
   - Forecast Horizon selector (Next 24h, Next 7d, Next 30d).
   - Battery Capacity (kWh) and Initial SOC (%) sliders.
   - Grid Electricity Tariff (₹/kWh) and Emission Factor (kg CO₂/kWh) sliders.
   - Facility Demand Scale slider.

---

## 10. Recommended Target Architecture

```text
                           ┌────────────────────────────────────────────────────────┐
                           │                     RAW DATASETS                       │
                           │  • Hourly Load India (46,728 hrs, 2019-2024)          │
                           │  • Delhi Weather (46,728 hrs, 2019-2024)              │
                           │  • Solar Generation & Sensors (May-June 2020)          │
                           │  • NASA POWER Hourly Wind Resource (2019-2024)        │
                           └───────────────────────────┬────────────────────────────┘
                                                       │
                                           ┌───────────▼───────────┐
                                           │   PREPROCESSING &     │
                                           │  MULTI-RESOURCE ALIGN │
                                           │  • Solar Sensor Merge │
                                           │  • Synchronized Window│
                                           └───────────┬───────────┘
                                                       │
                                    ┌──────────────────┴──────────────────┐
                                    │                                     │
                        ┌───────────▼───────────┐             ┌───────────▼───────────┐
                        │   HISTORICAL TREND    │             │  WEATHER INFLUENCE    │
                        │    PATTERN ENGINE     │             │    ANALYSIS (SHAP)    │
                        │  • Hourly, Daily, Wk, │             │  • Temperature / GHI  │
                        │    Monthly, Yearly    │             │  • Humidity / Wind    │
                        └───────────┬───────────┘             └───────────┬───────────┘
                                    │                                     │
                                    └──────────────────┬──────────────────┘
                                                       │
                                           ┌───────────▼───────────┐
                                           │   FEATURE GENERATION  │
                                           │  • Cyclical Encoding  │
                                           │  • Recursive Lags     │
                                           │  • Rolling Windows    │
                                           └───────────┬───────────┘
                                                       │
                                    ┌──────────────────┴──────────────────┐
                                    │                                     │
                        ┌───────────▼───────────┐             ┌───────────▼───────────┐
                        │    DEMAND FORECAST    │             │  RENEWABLE FORECAST   │
                        │        ENGINE         │             │        ENGINE         │
                        │  • Multi-Horizon Reg  │             │  • Weather-Solar Reg  │
                        │  • Autoregressive Lags│             │  • Wind Speed → Power │
                        └───────────┬───────────┘             └───────────┬───────────┘
                                    │                                     │
                                    └──────────────────┬──────────────────┘
                                                       │
                                           ┌───────────▼───────────┐
                                           │  PREDICTION INTERVALS │
                                           │  • Lower / Exp / Upper│
                                           │  • Fan Chart Margins  │
                                           └───────────┬───────────┘
                                                       │
                                           ┌───────────▼───────────┐
                                           │  REAL-TIME DISPATCH   │
                                           │       SIMULATOR       │
                                           │  • Demand vs. Renew   │
                                           │  • BESS SOC Tracking  │
                                           │  • Backup & Curtail   │
                                           └───────────┬───────────┘
                                                       │
                                           ┌───────────▼───────────┐
                                           │  DYNAMIC COST & CO₂   │
                                           │       ANALYZER        │
                                           │  • Configurable Tariff│
                                           │  • Avoided Emissions  │
                                           └───────────┬───────────┘
                                                       │
                                           ┌───────────▼───────────┐
                                           │  INTERACTIVE DECISION │
                                           │       DASHBOARD       │
                                           │  • 9 Focused Views    │
                                           │  • What-If Controls   │
                                           └───────────────────────┘
```

---

## 11. Revised Phase-by-Phase Plan

The plan follows the strict principle: **REUSE $\rightarrow$ EXTEND $\rightarrow$ REFACTOR**.

```text
Phase 1: Solar Sensor Integration & Synchronized Dataset Creation
Phase 2: Multi-Time-Scale Historical Pattern Learning Engine
Phase 3: Multi-Horizon Demand Forecasting Engine with Uncertainty
Phase 4: Weather-Driven Renewable Forecasting Engine with Uncertainty
Phase 5: Real-Time Energy Balance & Integrated Dispatch Simulator
Phase 6: Dynamic Cost & Avoided CO₂ Impact Analyzer
Phase 7: Interactive Decision-Support Dashboard Redesign
Phase 8: Comprehensive Automated Testing & End-to-End System Validation
Phase 9: Documentation Synchronization (README, progress.md, decisions.md)
```

### Detailed Phase Specifications:

#### Phase 1: Solar Sensor Integration & Synchronized Dataset Creation
* **Objective**: Enrich solar data with actual weather sensors (irradiance, temperature) and create the synchronized multi-resource dataset (May 15 – June 17, 2020) where Demand, Weather, Solar, and Wind coexist.
* **Components Reused**: `src/renewable/prepare_solar_data.py`, `src/data/clean_demand.py`.
* **New Components**: `src/data/create_synchronized_dataset.py`, `src/renewable/merge_solar_sensors.py`.
* **Expected Inputs**: `Plant_1_Generation_Data.csv`, `Plant_1_Weather_Sensor_Data.csv`, `Plant_2_*`, `final_merged_dataset.csv`, `nasa_power_wind_hourly.csv`.
* **Expected Outputs**: `data/processed/solar_weather_hourly.csv`, `data/processed/synchronized_scenario_dataset.csv`.
* **Validation**: Continuity, non-null check, aligned timestamps across all 4 resources.

#### Phase 2: Multi-Time-Scale Historical Pattern Learning Engine
* **Objective**: Learn and quantify electricity demand patterns across Hourly, Daily, Weekly, Monthly, and Yearly scales (2019–2024), and extract seasonal and weather-demand correlation coefficients.
* **Components Reused**: `src/eda/demand_analysis.py`, `src/eda/seasonality_analysis.py`.
* **New Components**: `src/forecasting/pattern_learning.py`.
* **Expected Inputs**: `data/processed/final_merged_dataset.csv`.
* **Expected Outputs**: `reports/historical_pattern_summary.csv`, seasonal demand profile matrices.
* **Validation**: Quantified diurnal peak hours, weekend reduction percentages, monthly peak factors, annual growth rates.

#### Phase 3: Multi-Horizon Demand Forecasting Engine with Uncertainty
* **Objective**: Build the forward multi-horizon forecasting engine (Hourly 1–24h, Daily 7d, Monthly 30d) that recursively predicts demand from the latest historical state and computes 95% prediction intervals (lower, expected, upper).
* **Components Reused**: `models/linear_regression.pkl`, `models/xgboost.pkl`, feature creation functions.
* **New Components**: `src/forecasting/multi_horizon_demand.py`.
* **Expected Inputs**: Trained demand model, latest historical observation window.
* **Expected Outputs**: Structured DataFrame with `[datetime, lower_demand, predicted_demand, upper_demand]` for 24h, 7d, and 30d horizons.
* **Validation**: Autoregressive update consistency, verification that `lower <= expected <= upper`.

#### Phase 4: Weather-Driven Renewable Forecasting Engine with Uncertainty
* **Objective**: Retrain solar model with irradiance and ambient temperature, convert wind speed forecasts to wind-power potential via power curve, and generate combined renewable forecasts with uncertainty ranges.
* **Components Reused**: `src/renewable/estimate_wind_power.py`, `models/wind/wind_linear_regression.pkl`.
* **New Components**: `src/renewable/train_weather_solar_model.py`, `src/forecasting/multi_horizon_renewable.py`.
* **Expected Inputs**: `solar_weather_hourly.csv`, `wind_power_estimates.csv`.
* **Expected Outputs**: `models/solar/solar_weather_regressor.pkl`, structured renewable forecast DataFrame with `[datetime, lower_solar, expected_solar, upper_solar, lower_wind, expected_wind, upper_wind, lower_total_re, expected_total_re, upper_total_re]`.
* **Validation**: Zero solar at night, non-negative wind power, power curve bounds [0, 1000 kW].

#### Phase 5: Real-Time Energy Balance & Integrated Dispatch Simulator
* **Objective**: Replace synthetic demand simulation with physical dispatch logic driven by predicted demand and predicted renewables.
* **Components Reused**: Battery SOC calculation logic from `src/storage/simulate_storage_backup.py`.
* **New Components**: `src/storage/dispatch_simulator.py`.
* **Expected Inputs**: Forecasted demand, forecasted renewables (solar + wind), battery parameters ($C_{\text{bat}}, \text{SOC}_0, P_{\max}^{\text{chg}}, P_{\max}^{\text{dis}}, \eta_c, \eta_d$).
* **Expected Outputs**: Time-series dispatch records: `[datetime, demand, re_used, battery_soc, battery_charge, battery_discharge, grid_backup, curtailed_re]`.
* **Validation**: Conservation of energy at every timestamp: $\text{Demand} = \text{RE Used} + \text{Battery Discharge} + \text{Grid Backup}$.

#### Phase 6: Dynamic Cost & Avoided CO₂ Impact Analyzer
* **Objective**: Implement dynamic economic and environmental analysis evaluating backup costs, storage savings, and avoided grid emissions under user-configurable tariffs.
* **Components Reused**: Formulas in `src/impact/calculate_cost_co2.py`.
* **New Components**: `src/impact/dynamic_impact_calculator.py`.
* **Expected Inputs**: Dispatch results, electricity tariff (₹/kWh), grid emission factor (kg CO₂/kWh).
* **Expected Outputs**: Hourly and aggregated metrics: Cost Without Storage, Cost With Storage, Cost Savings, CO₂ Without Storage, CO₂ With Storage, Avoided CO₂.
* **Validation**: Non-negative costs and emissions; cost savings equal to avoided backup energy $\times$ tariff.

#### Phase 7: Interactive Decision-Support Dashboard Redesign
* **Objective**: Redesign `app/dashboard.py` into a dynamic 9-section application connecting live model forecasting with what-if scenario controls.
* **Components Reused**: Streamlit page layout, styling, metric cards, limitations section.
* **New Components**: Modular page views in `app/dashboard.py` and `app/components/`.
* **Expected Inputs**: Validated forecasting, dispatch, and impact modules.
* **Expected Outputs**: Responsive web dashboard running at `http://localhost:8501`.
* **Validation**: All 9 sections load cleanly without runtime exceptions; sliders dynamically recalculate dispatch and savings.

#### Phase 8: Comprehensive Automated Testing & End-to-End System Validation
* **Objective**: Build a pytest test suite covering forecasting engines, dispatch conservation laws, uncertainty intervals, and economic calculations.
* **Components Reused**: Validation scripts in `src/**/validate_*.py`.
* **New Components**: `tests/test_forecasting.py`, `tests/test_dispatch.py`, `tests/test_impact.py`.
* **Expected Outputs**: Test results reporting 100% passing tests.
* **Validation**: Automated execution with `pytest tests/`.

#### Phase 9: Documentation Synchronization
* **Objective**: Synchronize `README.md`, `docs/progress.md`, and `docs/decisions.md` to document the completed decision-support architecture.
* **Expected Outputs**: Accurate, unified documentation reflecting actual codebase capabilities.

---

## 12. Conclusion & First Phase Readiness

The audit is complete. No source code has been altered. All existing assets have been cataloged for maximum reuse. 

**First Step Upon Approval**:
Execute **Phase 1: Solar Sensor Integration & Synchronized Dataset Creation** on a dedicated feature branch (`feature/solar-sensor-integration`).
