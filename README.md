# ML-Based Demand & Renewable Energy Forecasting

An end-to-end machine learning and time-series forecasting project for electricity demand and renewable energy forecasting, with the long-term goal of supporting energy planning, renewable integration, storage decisions, and environmental impact analysis.

---

## Project Overview

Electricity demand varies continuously with time, weather, seasonal conditions, and human activity.

This project develops a forecasting system that combines historical electricity demand, weather information, machine learning, and advanced time-series techniques to forecast future energy requirements.

The project progressively expands from electricity demand forecasting toward renewable energy forecasting and energy-management analysis.

---

# Project Pipeline

```text
Data Acquisition
       ↓
Data Cleaning & Integration
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Baseline Forecasting
       ↓
Machine Learning Models
       ↓
Advanced Time-Series Models
       ↓
Performance Comparison
       ↓
Best Model Selection
       ↓
Explainability
       ↓
Renewable Energy Forecasting
       ↓
Uncertainty / Confidence Analysis
       ↓
Storage vs Backup Simulation
       ↓
Cost & CO₂ Impact
       ↓
Dashboard & Final System
```

---

# Objectives

* Forecast electricity demand using historical data.
* Analyze the effect of weather and temporal patterns on electricity demand.
* Develop machine-learning-based forecasting models.
* Compare machine learning and advanced time-series approaches.
* Select the best-performing forecasting model.
* Provide model explainability.
* Extend the system toward solar and wind energy forecasting.
* Estimate forecasting uncertainty and confidence.
* Simulate storage versus backup energy decisions.
* Analyze potential cost and CO₂ impacts.
* Develop a final dashboard for visualization and decision support.

---

# Dataset

## Electricity Demand

The primary demand dataset contains hourly electricity demand data for India.

### Period

```text
2019-01-01 → 2024-04-30
```

### Records

```text
46,728 hourly records
```

### Demand Variables

* National demand
* Northern region demand
* Western region demand
* Eastern region demand
* Southern region demand
* North-Eastern region demand

---

## Weather

Historical hourly weather data was integrated with the electricity demand data.

### Weather Variables

* Temperature
* Relative humidity
* Cloud cover
* Precipitation
* Wind speed
* Solar radiation

The weather data was obtained for Delhi using historical hourly weather information.

---

# Phase 1 — Data Acquisition, Cleaning & Integration

**Status: ✅ Completed**

Phase 1 established the validated demand-weather dataset used by the subsequent analysis and forecasting stages.

## Demand Data Processing

Raw demand data was inspected and cleaned.

Scripts:

```text
src/data/inspect_demand.py
src/data/clean_demand.py
```

Generated:

```text
data/processed/demand_cleaned.csv
```

Basic temporal features were also created:

* Hour
* Day
* Month
* Year
* Day of week
* Weekend indicator

## Weather Data

Historical hourly weather data was acquired and integrated with the demand data.

Weather variables:

* Temperature
* Relative humidity
* Cloud cover
* Precipitation
* Wind speed
* Solar radiation

## Dataset Integration

Script:

```text
src/data/merge_data.py
```

Generated:

```text
data/processed/final_merged_dataset.csv
```

## Final Dataset

```text
Rows    : 46,728
Columns : 19
Start   : 2019-01-01 00:00:00
End     : 2024-04-30 23:00:00
```

## Validation

Script:

```text
src/data/validate_merged.py
```

Validation confirmed:

* Correct row count
* Correct columns
* No missing values
* No duplicate timestamps
* Correct chronological order
* Correct start date
* Correct end date
* Continuous hourly timestamps

---

# Phase 2 — Exploratory Data Analysis

**Status: ✅ Completed**

EDA was performed on the validated Phase 1 merged dataset.

## EDA Areas

* Basic dataset analysis
* Electricity demand analysis
* Hourly demand patterns
* Daily demand patterns
* Monthly demand patterns
* Yearly demand trends
* Weekday vs weekend analysis
* Regional demand analysis
* Weather analysis
* Demand-weather correlation analysis
* Seasonality analysis

## EDA Scripts

```text
src/eda/__init__.py
src/eda/basic_eda.py
src/eda/demand_analysis.py
src/eda/weather_analysis.py
src/eda/correlation_analysis.py
src/eda/seasonality_analysis.py
```

## Important EDA Findings

Average national electricity demand:

```text
160,487.07 MW
```

Highest average demand hour:

```text
11:00
```

Average demand at 11:00:

```text
173,085.39 MW
```

Lowest average demand hour:

```text
03:00
```

Average demand at 03:00:

```text
143,526.45 MW
```

Average weekday demand:

```text
161,412.76 MW
```

Average weekend demand:

```text
158,171.17 MW
```

## Weather Statistics

Average temperature:

```text
24.05 °C
```

Average relative humidity:

```text
62.61 %
```

Average cloud cover:

```text
31.94 %
```

Average precipitation:

```text
0.08 mm
```

Average wind speed:

```text
9.73 km/h
```

Average solar radiation:

```text
202.42 W/m²
```

## EDA Visualizations

Generated visualizations are stored in:

```text
reports/figures/
```

The figures include:

* Average demand by hour
* Average demand by day
* Average demand by month
* Average demand by year
* Regional demand comparison
* Temperature analysis
* Solar radiation analysis
* Wind speed analysis
* Demand-weather correlation
* Correlation matrix
* Daily demand trend
* Monthly demand trend

---

# Phase 3 — Feature Engineering

**Status: ✅ Completed**

Feature engineering was performed using the validated Phase 1 dataset and findings from Phase 2 EDA.

## Time-Based Features

Cyclical time features were created:

```text
hour_sin
hour_cos
month_sin
month_cos
day_of_week_sin
day_of_week_cos
```

These represent recurring temporal patterns in a machine-learning-friendly form.

## Lag Features

Historical demand features were created:

```text
demand_lag_1h
demand_lag_24h
demand_lag_168h
```

These represent:

* Previous-hour demand
* Previous-day demand
* Previous-week demand

## Rolling Features

Rolling demand statistics were created:

```text
demand_rolling_mean_24h
demand_rolling_std_24h
demand_rolling_mean_168h
demand_rolling_std_168h
```

Rolling calculations use shifted demand values to avoid target leakage.

## Feature Engineering Scripts

```text
src/features/__init__.py
src/features/create_time_features.py
src/features/create_lag_features.py
src/features/create_rolling_features.py
src/features/create_features.py
src/features/validate_features.py
```

## Feature Datasets

Intermediate datasets:

```text
data/processed/time_features.csv
data/processed/lag_features.csv
data/processed/rolling_features.csv
```

Final feature dataset:

```text
data/processed/featured_dataset.csv
```

The final feature dataset was validated for:

* Missing values
* Duplicate timestamps
* Chronological ordering
* Required engineered features

---

# Phase 4 — Baseline Forecasting

**Status: ✅ Completed**

Baseline forecasting was implemented to establish benchmark performance before training machine-learning and advanced time-series models.

## Baseline Models

### Seasonal Naive — 24 Hours

Uses demand from the previous day as the prediction.

```text
naive_24h = demand(t - 24)
```

### Seasonal Naive — 168 Hours

Uses demand from the previous week as the prediction.

```text
naive_168h = demand(t - 168)
```

### Linear Regression

A linear regression model was implemented using temporal, lag, rolling, and weather features.

## Baseline Scripts

```text
src/baseline/__init__.py
src/baseline/naive_baseline.py
src/baseline/regression_baseline.py
src/baseline/evaluate_baselines.py
```

## Evaluation Metrics

The baseline models were evaluated using:

* MAE — Mean Absolute Error
* RMSE — Root Mean Squared Error
* MAPE — Mean Absolute Percentage Error

## Generated Artifacts

```text
data/processed/baseline_predictions.csv
reports/baseline_metrics.csv
reports/figures/baseline_comparison.png
```

---

# Phase 5 — Machine Learning Models

**Status: ✅ Completed**

Machine-learning regression models were trained for national electricity demand forecasting using the engineered features developed in Phase 3.

## ML Data Preparation

The feature dataset was divided chronologically:

```text
80% → Training
20% → Testing
```

A chronological split was used to preserve the time-series structure and prevent future observations from being randomly mixed into the training data.

Script:

```text
src/models/prepare_ml_data.py
```

Generated:

```text
data/processed/ml_train.csv
data/processed/ml_test.csv
```

---

## Machine Learning Models

Three regression models were implemented.

### Random Forest Regressor

A tree-based ensemble model capable of capturing nonlinear relationships between demand, temporal features, lag features, rolling statistics, and weather variables.

Script:

```text
src/models/train_random_forest.py
```

Generated model:

```text
models/random_forest.pkl
```

### Gradient Boosting Regressor

A sequential boosting model that builds an ensemble of weak learners to improve prediction performance.

Script:

```text
src/models/train_gradient_boosting.py
```

Generated model:

```text
models/gradient_boosting.pkl
```

### XGBoost Regressor

A gradient-boosted tree model used as a strong nonlinear forecasting benchmark.

Script:

```text
src/models/train_xgboost.py
```

Generated model:

```text
models/xgboost.pkl
```

---

## ML Features

The models use:

### Cyclical Features

```text
hour_sin
hour_cos
month_sin
month_cos
day_of_week_sin
day_of_week_cos
```

### Lag Features

```text
demand_lag_1h
demand_lag_24h
demand_lag_168h
```

### Rolling Features

```text
demand_rolling_mean_24h
demand_rolling_std_24h
demand_rolling_mean_168h
demand_rolling_std_168h
```

### Weather Features

```text
temperature_2m_c
relative_humidity_pct
cloud_cover_pct
precipitation_mm
wind_speed_10m_kmh
solar_radiation_w_m2
```

---

## ML Evaluation

The Phase 5 models were compared against the Phase 4 baseline models:

```text
Naive 24h
Naive 168h
Linear Regression
Random Forest
Gradient Boosting
XGBoost
```

Evaluation metrics:

* MAE
* RMSE
* MAPE

Script:

```text
src/models/evaluate_ml_models.py
```

Generated:

```text
data/processed/ml_predictions.csv
reports/ml_model_metrics.csv
reports/figures/ml_model_comparison.png
```

---

## Phase 5 Validation

Script:

```text
src/models/validate_models.py
```

Validation checks include:

* Training dataset availability
* Testing dataset availability
* Model artifact availability
* Prediction output availability
* Evaluation metric availability
* Required model coverage
* Performance comparison output

**Phase 5 completed successfully.**

---

# Phase 6 — Advanced Time-Series Models

Classical time-series models were added to complement the machine-learning forecasting models.

Implemented models:

- ARIMA
- SARIMA

ARIMA configuration:

```text
(2,1,2)

SARIMA configuration:
(1,1,1)(1,1,1,24)

The SARIMA model uses a 24-hour seasonal period to capture daily hourly demand patterns.

Generated artifacts:
models/arima.pkl
models/sarima.pkl
data/processed/ts_train.csv
data/processed/ts_test.csv
data/processed/ts_predictions.csv
reports/time_series_metrics.csv

ARIMA and SARIMA are evaluated alongside the Phase 4 baseline models and Phase 5 machine-learning models using MAE, RMSE, and MAPE.

---
```
# Phase 7 — Performance Comparison

A unified performance comparison was created across all forecasting approaches implemented in Phases 4, 5, and 6.

## Compared Models

- Naive 24h
- Naive 168h
- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost
- ARIMA
- SARIMA

## Evaluation Metrics

- MAE
- RMSE
- MAPE

## Generated Artifacts

```text
reports/performance_comparison.csv
reports/figures/performance_comparison.png
```
---

# Phase 8 — Best Model Selection

The best forecasting model was selected using the consolidated performance results generated in Phase 7.

### Selection Criteria

- **RMSE** — primary criterion
- **MAE** — secondary criterion
- **MAPE** — secondary criterion

Lower metric values indicate better forecasting performance.

### Generated Artifacts

```text
reports/best_model.csv
reports/figures/best_model_comparison.png
```
# Project Progress

| Phase | Description                              | Status      |
| ----: | ---------------------------------------- | ----------- |
|     1 | Data Acquisition, Cleaning & Integration | ✅ Completed |
|     2 | Exploratory Data Analysis                | ✅ Completed |
|     3 | Feature Engineering                      | ✅ Completed |
|     4 | Baseline Forecasting                     | ✅ Completed |
|     5 | Machine Learning Models                  | ✅ Completed |
|     6 | Advanced Time-Series Models              | ✅ Completed |
|     7 | Performance Comparison                   | ✅ Completed |
|     8 | Best Model Selection                     | ✅ Completed |
|     9 | Explainability                           | ⏳ Next      |
|    10 | Renewable Energy Forecasting             | ⏳ Pending   |
|    11 | Uncertainty / Confidence Analysis        | ⏳ Pending   |
|    12 | Storage vs Backup Simulation             | ⏳ Pending   |
|    13 | Cost & CO₂ Impact Analysis               | ⏳ Pending   |
|    14 | Dashboard & Final System                 | ⏳ Pending   |

---

# Current Status

The following phases have been successfully completed:

```text
Phase 1 — Data Acquisition, Cleaning & Integration
Phase 2 — Exploratory Data Analysis
Phase 3 — Feature Engineering
Phase 4 — Baseline Forecasting
Phase 5 — Machine Learning Models
```

The current next stage is:

```text
Phase 6 — Advanced Time-Series Models
```

Phase 6 has **not yet been implemented**.

---

# Future Renewable Energy Components

The later stages of the project will incorporate renewable-energy-related components including:

* Solar energy forecasting
* Wind energy forecasting
* Renewable supply analysis
* Uncertainty and confidence estimation
* Storage versus backup simulation
* Cost analysis
* CO₂ impact analysis

Actual solar and wind generation data and CO₂ calculations are intentionally reserved for their planned later stages rather than being forced into the Phase 1 demand-weather dataset.

---

# Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost
* Joblib
* Statsmodels
* Time-Series Forecasting
* Machine Learning
* Explainable AI
* Data Visualization

Additional libraries will be introduced as required by later phases.

---

# Project Structure

```text
ML-Based-Demand-Renewable-Energy-Forecasting/

│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── progress.md
│   ├── decisions.md
│   ├── Dataset_Research_Report.md
│   ├── Full_Project_Execution_Plan.md
│   └── PROJECT_CONTEXT.md
│
├── models/
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   ├── baseline_metrics.csv
│   └── ml_model_metrics.csv
│
├── src/
│   │
│   ├── data/
│   │   ├── inspect_demand.py
│   │   ├── clean_demand.py
│   │   ├── merge_data.py
│   │   └── validate_merged.py
│   │
│   ├── eda/
│   │   ├── __init__.py
│   │   ├── basic_eda.py
│   │   ├── demand_analysis.py
│   │   ├── weather_analysis.py
│   │   ├── correlation_analysis.py
│   │   └── seasonality_analysis.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── create_time_features.py
│   │   ├── create_lag_features.py
│   │   ├── create_rolling_features.py
│   │   ├── create_features.py
│   │   └── validate_features.py
│   │
│   ├── baseline/
│   │   ├── __init__.py
│   │   ├── naive_baseline.py
│   │   ├── regression_baseline.py
│   │   └── evaluate_baselines.py
│   │
│   └── models/
│       ├── __init__.py
│       ├── prepare_ml_data.py
│       ├── train_random_forest.py
│       ├── train_gradient_boosting.py
│       ├── train_xgboost.py
│       ├── evaluate_ml_models.py
│       └── validate_models.py
│
├── README.md
└── .gitignore
```

---

# Development Approach

The project is being implemented incrementally, one phase at a time.

Each phase follows this workflow:

1. Implementation
2. Validation
3. Documentation
4. Git commit
5. GitHub push
6. Pull Request
7. Merge into `main`

This approach keeps the project reproducible, organized, and traceable throughout development.

---

# Documentation

Project progress:

```text
docs/progress.md
```

Technical decisions:

```text
docs/decisions.md
```

Dataset research:

```text
docs/Dataset_Research_Report.md
```

Full execution plan:

```text
docs/Full_Project_Execution_Plan.md
```

Project context:

```text
docs/PROJECT_CONTEXT.md
```

---

# Model Artifacts

Trained machine-learning models are generated under:

```text
models/
```

The `.pkl` model artifacts are generated outputs and are excluded from normal Git tracking because some trained models exceed GitHub's standard file-size limit.

They can be regenerated using the corresponding training scripts under:

```text
src/models/
```

---

# License

This project is developed for academic and research purposes.
