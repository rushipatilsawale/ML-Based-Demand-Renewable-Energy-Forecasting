# Project Decisions

## ML-Based Demand & Renewable Energy Forecasting

This document records important technical and project-scope decisions made during implementation.

---

# Phase 1 — Data Acquisition, Cleaning & Integration

## Decision 1 — Primary Demand Dataset

The historical India electricity demand dataset was selected as the primary demand source.

The dataset contains hourly national and regional electricity demand from:

```text
2019-01-01
to
2024-04-30
```

The national demand variable is selected as the primary forecasting target.

---

## Decision 2 — Weather Data

Historical hourly weather data was integrated with the electricity demand data.

The selected weather variables are:

* Temperature
* Relative humidity
* Cloud cover
* Precipitation
* Wind speed
* Solar radiation

---

## Decision 3 — Common Time Resolution

The project uses **hourly resolution** as the primary time resolution.

All integrated datasets must therefore use compatible hourly timestamps.

---

## Decision 4 — Final Phase 1 Dataset

The Phase 1 integrated dataset contains electricity demand and weather information.

Final dataset:

```text
data/processed/final_merged_dataset.csv
```

Dimensions:

```text
46,728 rows
19 columns
```

The dataset passed all validation checks.

---

## Decision 5 — Time-Based Features

The following basic time features were created during preprocessing:

* hour
* day
* month
* year
* day of week
* weekend indicator

Additional forecasting features will be created during Phase 3.

---

# Phase 2 — Exploratory Data Analysis

## Decision 6 — EDA Dataset

EDA is performed on:

```text
data/processed/final_merged_dataset.csv
```

This ensures that all exploratory analysis is performed on the validated integrated dataset.

---

## Decision 7 — Demand Analysis

National electricity demand is the primary target variable for the initial forecasting stages.

Demand was analyzed across:

* Hour
* Day
* Month
* Year
* Weekday/weekend
* Region

This was done to identify temporal patterns and seasonality before feature engineering.

---

## Decision 8 — Weather Analysis

Weather variables were analyzed individually and against electricity demand.

The objective is to determine which environmental variables may provide useful predictive information for demand forecasting.

---

## Decision 9 — Correlation Analysis

Correlation analysis was performed between national electricity demand and the available weather variables.

Correlation is used as an exploratory tool and will not be treated as the sole criterion for selecting forecasting features.

---

## Decision 10 — Seasonality

Hourly, daily, monthly, and yearly patterns are being considered when designing forecasting features.

The observed temporal structure will guide the creation of lag, rolling-window, calendar, and seasonal features during Phase 3.

---

# Renewable Energy Scope

## Decision 11 — Solar and Wind Generation

Actual renewable electricity generation data is **not being forced into the Phase 1 demand-weather merged dataset**.

Solar and wind supply forecasting remain part of the later project stages according to the overall project roadmap.

The existing solar radiation and wind-speed variables are retained as weather/environmental variables.

---

## Decision 12 — CO₂ Analysis

CO₂ impact calculations are retained for the later system-level stage.

They will be incorporated after forecasting and energy-management components are developed rather than artificially adding CO₂ calculations to the Phase 1 dataset.

---

# Modeling Decisions

## Decision 13 — Feature Engineering Before Modeling

Feature engineering will be completed before baseline and machine-learning forecasting models are implemented.

The feature-engineering stage will focus on extracting predictive information from:

* Historical demand
* Weather
* Calendar information
* Temporal patterns
* Lagged demand
* Rolling statistics

---

## Decision 14 — Evaluation

Forecasting models will be evaluated using appropriate time-series evaluation methods.

Random train-test splitting will be avoided for the primary forecasting workflow because it can introduce temporal leakage.

---

# Phase 3 — Feature Engineering Decisions

## Decision 15 — Cyclical Time Encoding

Cyclical encoding was selected for hour, month, and day-of-week variables so that temporal relationships are represented continuously.

## Decision 16 — Demand Lag Features

Three demand lags were selected:

- 1 hour
- 24 hours
- 168 hours

These represent short-term, daily, and weekly demand dependencies.

## Decision 17 — Rolling Features

24-hour and 168-hour rolling mean and standard deviation features were created.

Rolling calculations use shifted demand values so the current target value is not included.

## Decision 18 — Leakage Prevention

Feature engineering must prevent future information from entering the predictors.

The current demand target is therefore excluded from its own lag and rolling calculations.

## Decision 19 — Feature Dataset

The engineered dataset is stored separately from the original merged dataset:

`data/processed/featured_dataset.csv`

The original Phase 1 dataset remains unchanged.

---

# Phase 4 — Baseline Forecasting Decisions

## Decision 20 — Seasonal Naive Baselines

Seasonal naive forecasting was selected as the primary simple benchmark.

Two seasonal periods were used:

* 24 hours for daily seasonality
* 168 hours for weekly seasonality

This provides simple benchmarks that future forecasting models must outperform.

## Decision 21 — Linear Regression Baseline

Linear Regression was selected as a simple machine-learning baseline before introducing more advanced models.

The model uses engineered temporal, lag, rolling, and weather features.

## Decision 22 — Chronological Train-Test Split

An 80/20 chronological split was selected instead of a random split.

This preserves the temporal structure of the forecasting problem and prevents future observations from entering the training data.

## Decision 23 — Evaluation Metrics

The forecasting baselines are evaluated using:

* MAE
* RMSE
* MAPE

RMSE is particularly useful for identifying larger forecasting errors.

## Decision 24 — Baseline Artifacts

Baseline predictions are stored separately from the feature dataset:

`data/processed/baseline_predictions.csv`

Evaluation results are stored in:

`reports/baseline_metrics.csv`

This keeps the original feature dataset unchanged.

---

# Phase 5 — Machine Learning Model Decisions

## Decision 25 — Chronological Data Split

An 80/20 chronological train-test split was retained for machine-learning forecasting.

Random splitting was avoided because forecasting models must predict future observations using information available in the past.

## Decision 26 — Random Forest

Random Forest Regressor was selected as a tree-based ensemble benchmark capable of modeling nonlinear relationships between demand, temporal features, and weather variables.

## Decision 27 — Gradient Boosting

Gradient Boosting Regressor was selected as a sequential boosting model for comparison with Random Forest.

## Decision 28 — XGBoost

XGBoost Regressor was selected as an additional gradient-boosted tree model and provides a strong benchmark for nonlinear forecasting relationships.

## Decision 29 — Common Evaluation Framework

All baseline and machine-learning models are evaluated using the same:

* MAE
* RMSE
* MAPE

This allows direct and consistent comparison.

## Decision 30 — Model Artifacts

Trained models are stored separately in the `models/` directory.

Prediction results and evaluation metrics are stored separately from the original feature dataset.

---

