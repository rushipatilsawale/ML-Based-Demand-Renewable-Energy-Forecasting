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

- Temperature
- Relative humidity
- Cloud cover
- Precipitation
- Wind speed
- Solar radiation

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

- hour
- day
- month
- year
- day of week
- weekend indicator

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

- Hour
- Day
- Month
- Year
- Weekday/weekend
- Region

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

- Historical demand
- Weather
- Calendar information
- Temporal patterns
- Lagged demand
- Rolling statistics

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

- 24 hours for daily seasonality
- 168 hours for weekly seasonality

This provides simple benchmarks that future forecasting models must outperform.

## Decision 21 — Linear Regression Baseline

Linear Regression was selected as a simple machine-learning baseline before introducing more advanced models.

The model uses engineered temporal, lag, rolling, and weather features.

## Decision 22 — Chronological Train-Test Split

An 80/20 chronological split was selected instead of a random split.

This preserves the temporal structure of the forecasting problem and prevents future observations from entering the training data.

## Decision 23 — Evaluation Metrics

The forecasting baselines are evaluated using:

- MAE
- RMSE
- MAPE

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

- MAE
- RMSE
- MAPE

This allows direct and consistent comparison.

## Decision 30 — Model Artifacts

Trained models are stored separately in the `models/` directory.

Prediction results and evaluation metrics are stored separately from the original feature dataset.

---

# Phase 6 — Advanced Time-Series Model Decisions

## Decision 31 — Dedicated Time-Series Dataset

A separate dataset containing `datetime` and `national_demand_mw` was created for classical time-series models.

This keeps statistical time-series modeling separate from the feature-heavy machine-learning pipeline.

## Decision 32 — ARIMA

ARIMA was selected as a classical statistical forecasting benchmark.

Configuration:

```text
(2,1,2)
```

## Decision 33 — Daily SARIMA Seasonality

SARIMA was selected to model both non-seasonal and seasonal demand behavior.

A seasonal period of 24 hours was selected because the dataset contains hourly electricity demand and daily demand patterns were identified during EDA.

Configuration:

```text
(1,1,1)(1,1,1,24)
```

## Decision 34 — SARIMA Training Window

SARIMA was trained using the most recent 90 days of the training dataset.

This decision was made because fitting a seasonal SARIMA model with 24-hour seasonality on the complete 37,248-hour training dataset was computationally expensive.

## Decision 35 — Common Evaluation Framework

ARIMA and SARIMA were evaluated using the same metrics used in previous phases:

- MAE
- RMSE
- MAPE

They were also compared with the Phase 4 and Phase 5 forecasting models.

## Decision 36 — Time-Series Artifacts

Time-series models are stored separately in:

```text
models/
```

Forecasts are stored in:

```text
data/processed/
```

Evaluation metrics are stored in:

```text
reports/
```

This maintains separation between models, predictions, and evaluation results.

---

# Phase 7 - Performance Comparison Decisions

## Decision 37 — Unified Model Performance Comparison

A unified performance comparison was created to evaluate forecasting models developed across Phases 4, 5, and 6.

The comparison includes:

- Naive 24h
- Naive 168h
- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost
- ARIMA
- SARIMA

---

## Decision 38 — Standard Evaluation Metrics

The project uses the following common metrics for comparing forecasting models:

- MAE
- RMSE
- MAPE

Using the same metrics across all models ensures consistent evaluation.

---

## Decision 39 — RMSE-Based Model Ranking

RMSE was selected as the primary ranking metric for the consolidated performance comparison.

Models are sorted in ascending RMSE order, where a lower RMSE indicates better forecasting performance.

The final model-selection decision will be made during Phase 8 using the consolidated results.

---

## Decision 40 — Consolidated Performance Artifacts

The following artifacts were generated for Phase 7:

```text
reports/performance_comparison.csv
reports/figures/performance_comparison.png

---

```

# Phase 8 - Best Model Selection

## Decision 41 — Best Model Selection

The best forecasting model is selected from the consolidated Phase 7 comparison.

RMSE is used as the primary selection criterion because it penalizes larger forecasting errors more strongly.

MAE and MAPE are retained as supporting evaluation metrics.

---

## Decision 42 — Model Ranking

Models are ranked using:

1. RMSE
2. MAE
3. MAPE

Lower values indicate better forecasting performance.

The model with the lowest RMSE is selected as the primary forecasting model.

---

## Decision 43 — Best Model Artifact

The selected forecasting model and its evaluation metrics are stored in:

```text
reports/best_model.csv
```

---

# Phase 9 Decisions — Explainability

## Decision 44 — Use SHAP

SHAP was selected as the explainability method because it provides feature-level contribution analysis for individual predictions and global feature importance.

## Decision 45 — Explain the selected best model

The model selected in Phase 8 was used for explainability rather than selecting a different model.

Selected model:

- Linear Regression

## Decision 46 — Use LinearExplainer

Because the selected model is Linear Regression, SHAP LinearExplainer was used instead of TreeExplainer.

## Decision 47 — Use the original training features

The explainability pipeline uses the same 19 features used to train the Linear Regression model to prevent feature-dimension mismatch.

## Decision 48 — Generate global importance plots

Mean absolute SHAP values were used to rank feature importance and generate:

- SHAP summary plot
- SHAP bar plot

---

# Phase 10 Decisions — Renewable Energy Forecasting

## Decision 49 — Use separate renewable data sources

Solar and wind resources were handled using separate datasets because a single synchronized Indian renewable-generation dataset covering the complete project period was not available for this implementation.

## Decision 50 — Use Indian solar generation data

The selected solar dataset contains generation data from two Indian solar plants.

AC_POWER was used as the solar generation variable.

The inverter-level generation records were aggregated to plant-level hourly generation.

## Decision 51 — Use NASA POWER for wind resource forecasting

NASA POWER hourly wind-speed data was selected as a wind-resource proxy.

The data represents meteorological wind speed rather than measured turbine generation.

Delhi coordinates were used to maintain geographic consistency with the project's India-focused demand and weather analysis.

## Decision 52 — Forecast wind speed rather than measured wind generation

Because the selected wind dataset provides meteorological wind speed rather than turbine SCADA generation, the model forecasts wind speed.

The forecast is subsequently converted into estimated wind-power potential.

## Decision 53 — Use a representative turbine power curve

A normalized representative turbine power curve was used to convert wind speed into estimated power potential.

The reference capacity is 1000 kW and is used only for normalization.

The resulting values must not be interpreted as measured generation from an actual 1 MW turbine installation.

## Decision 54 — Use Linear Regression

Linear Regression was used for both solar and wind forecasting to maintain a consistent baseline modelling approach within the renewable forecasting phase.

## Decision 55 — Maintain chronological validation

An 80/20 chronological train-test split was used for renewable forecasting models to prevent future observations from being used to train the models.

## Decision 56 — Document renewable data limitations

The solar dataset covers approximately one month rather than the complete 2019–2024 demand period.

The wind dataset represents meteorological wind speed and is not measured Indian wind-generation data.

Therefore, Phase 10 results are presented as renewable forecasting and estimated renewable-power potential rather than a complete historical renewable-generation forecast for India.

## Decision 57 — Continue renewable analysis in later phases

The outputs generated in Phase 10 will be used as inputs for subsequent uncertainty analysis, storage-versus-backup simulation, and cost and CO₂ impact analysis.

---

# Phase 11 Decisions — Uncertainty / Confidence Analysis

## Decision 58 — Use prediction intervals

Uncertainty was represented using prediction intervals around the model's point predictions.

## Decision 59 — Use residual-based uncertainty

The test-set residual standard deviation was used to estimate prediction uncertainty.

## Decision 60 — Use 95% confidence level

A 95% prediction interval was selected for the uncertainty analysis.

## Decision 61 — Apply non-negative bounds

Solar generation and wind-speed predictions cannot be negative, so lower prediction bounds were clipped at zero.

## Decision 62 — Evaluate interval coverage

Interval coverage was calculated by checking whether the actual observation falls between the lower and upper prediction bounds.

## Decision 63 — Apply uncertainty analysis to both renewable resources

The same uncertainty-analysis framework was applied to:

- Solar generation
- Wind speed

The wind results represent uncertainty in predicted wind speed rather than measured wind generation.

---

# Phase 12 Decisions — Storage vs Backup Simulation

## Decision 64 — Use scenario-based simulation

A scenario-based simulation was selected because the Phase 10 solar and wind datasets are not temporally synchronized with the complete historical demand dataset.

## Decision 65 — Use solar forecast as renewable input

Solar forecast data was used as the renewable-energy input because it represents renewable generation in kW.

Wind-speed predictions were not directly added to renewable energy because wind speed is not equivalent to electrical power generation.

## Decision 66 — Use a battery storage model

A battery model was implemented with:

- 5000 kWh capacity
- 2500 kWh initial SOC
- 1000 kW maximum charge rate
- 1000 kW maximum discharge rate
- 90% charge efficiency
- 90% discharge efficiency

## Decision 67 — Compare storage against backup-only operation

Two scenarios were evaluated:

- Backup without storage
- Backup after renewable energy and battery storage

This allows the reduction in backup requirement due to storage to be quantified.

## Decision 68 — Use a scenario-based demand profile

A synthetic demand profile was used for the simulation because the renewable forecasting period does not align with the complete historical demand dataset.

The resulting analysis is therefore a scenario study rather than a historical demand-storage reconstruction.

---

# Phase 13 Decisions — Cost & CO₂ Impact Analysis

## Decision 69 — Use Phase 12 backup results

The Phase 12 backup requirements were used as the basis for cost and CO₂ calculations.

## Decision 70 — Use a configurable electricity cost

An electricity cost of ₹6.52/kWh was used as a scenario/reference assumption.

The value is configurable and should not be interpreted as a universal electricity tariff.

## Decision 71 — Use a grid emission factor

A grid emission factor of 0.716 kg CO₂/kWh was used to estimate emissions associated with backup electricity.

## Decision 72 — Calculate cost from backup energy

Backup energy was multiplied by the electricity cost to calculate:

- Cost without storage
- Cost with storage
- Cost savings

## Decision 73 — Calculate CO₂ from backup energy

Backup energy was multiplied by the grid emission factor to calculate:

- CO₂ emissions without storage
- CO₂ emissions with storage
- CO₂ reduction

## Decision 74 — Compare storage and backup scenarios

The primary comparison is between the backup-only scenario and the storage scenario.

## Decision 75 — Preserve Phase 12 limitations

The cost and CO₂ analysis remains scenario-based because the underlying Phase 12 simulation uses a synthetic demand profile and solar forecast data.

Therefore, the results should not be interpreted as a complete historical India-wide economic or emissions assessment.

---
