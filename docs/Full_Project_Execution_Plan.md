ML-Based Demand & Renewable Energy Forecasting
Full Project Execution Plan
Team Development Guide — Datasets, Process, Architecture & Deliverables
1. Project Overview
We are building a full-stack ML system that predicts short-term electricity demand and renewable (solar/wind) supply, estimates forecast confidence, simulates a storage-vs-backup dispatch decision, and calculates the resulting cost/CO2 impact — all shown on a live interactive dashboard.
The final deliverable has four connected parts:
•	ML Model — forecasts demand & supply with confidence ranges (probabilistic forecasting)
•	Backend (API) — serves predictions, runs the dispatch simulation and CO2 calculation, talks to the database
•	Database — stores historical data, weather data, predictions, and simulation logs
•	Frontend (Dashboard) — lets a user view forecasts, change what-if scenarios, and see the impact results
2. System Architecture
High-level data flow:
Raw Data (load + weather) → Data Pipeline (cleaning, feature engineering) → ML Model (stored on backend) → Backend API → Database (stores results) → Frontend Dashboard (displays to user)
2.1 Component Breakdown
Layer	Responsibility	Suggested Tech
ML Model	Train & serve demand/supply forecasts, uncertainty ranges, feature importance (SHAP)	Python, XGBoost/LightGBM, Prophet, scikit-learn, SHAP
Backend	Expose REST APIs for predictions, run dispatch simulation logic, run CO2 calculation, connect DB to frontend	FastAPI (Python)
Database	Store historical load/weather data, model predictions, simulation results, user scenario inputs	PostgreSQL (or SQLite for simplicity)
Frontend	Interactive dashboard: view forecasts, adjust what-if scenario sliders, see dispatch decision + CO2/cost impact	React + Recharts, or Streamlit for a faster build
 
Team tip: if timeline is tight, Streamlit can combine frontend + backend calls into one simpler app. If you want a 'proper' 3-tier architecture (more impressive to show), use FastAPI + React as separate services talking over REST API.
 
3. Datasets — What to Collect and Where to Get Them
3.1 Electricity Demand / Load Data
•	Kaggle — Hourly Energy Consumption (PJM Interconnection, USA): clean hourly load data, easiest to start with. Search 'PJM Hourly Energy Consumption' on kaggle.com.
•	EIA Open Data (US Energy Information Administration): eia.gov/opendata — free API, real hourly demand data by region.
•	ENTSO-E Transparency Platform (Europe): transparency.entsoe.eu — free registration, hourly load data across European countries.
•	Grid-India / POSOCO (India): posoco.in / reports.grid-india.in — daily/hourly grid demand reports for Indian states if you want an India-based case study.
3.2 Weather Data (for demand + solar/wind supply features)
•	Open-Meteo API: open-meteo.com — free, no API key required, historical + forecast weather data (temperature, cloud cover, wind speed).
•	NASA POWER API: power.larc.nasa.gov — free solar radiation and wind speed data, good for solar/wind supply modeling.
•	OpenWeatherMap: openweathermap.org/api — free tier available, good for real-time + historical weather.
3.3 Solar / Wind Generation Data
•	Kaggle — Solar Power Generation Data: search 'Solar Power Generation Data' — includes plant generation + weather sensor readings.
•	Kaggle — Wind Turbine SCADA Data: search 'Wind Turbine Scada Dataset' — real turbine output data useful for wind supply forecasting.
•	NREL (National Renewable Energy Laboratory, USA): nrel.gov/grid/solar-resource, wind resource data — good supplementary datasets.
3.4 Emission Factors (for the CO2 Calculator)
•	EPA eGRID (US): epa.gov/egrid — standard CO2-per-MWh emission factors by region.
•	CEA India (Central Electricity Authority): cea.nic.in — publishes India's grid emission factor reports (CO2/kWh) — use this if working with Indian data.
Pick ONE region consistently (e.g., one US grid region, or one Indian state) so demand data, weather data, and emission factors all line up. Mixing regions will break your feature alignment.
 
4. Step-by-Step Development Process
Phase 1 — Data Collection & Cleaning (Week 1)
•	Download historical load data and matching weather data for the same region and time range
•	Align timestamps (hourly), handle missing values, remove outliers
•	Merge load + weather + calendar (holiday/weekday) data into a single clean dataset
Phase 2 — Exploratory Data Analysis / EDA (Week 1-2)
•	Plot demand patterns by hour, day of week, season
•	Check correlation between weather variables and demand/supply
•	Identify seasonality and trend components
Phase 3 — Baseline Model (Week 2)
•	Build a naive baseline (e.g., 'same as last week same hour')
•	Build a simple regression model as a second baseline
•	Record baseline error (MAPE/RMSE) — this is what all later models must beat
Phase 4 — Core Forecasting Model (Week 3-4)
•	Feature engineer: lag features, rolling averages, cyclical time encoding
•	Train XGBoost/LightGBM or Prophet for point forecasts
•	Add quantile regression (or LightGBM quantile loss) for confidence-range (probabilistic) forecasts
•	Evaluate against baseline on a held-out future time period (never shuffle time-series data)
•	Add SHAP to explain which features drive each prediction
Phase 5 — Dispatch Simulator & CO2 Calculator (Week 4-5)
•	Define simple dispatch rules: if predicted renewable supply < predicted demand, decide storage release vs backup power, based on confidence level
•	Convert backup power used into cost and CO2 emissions using regional emission factors
•	Test simulator against a few realistic scenarios (sunny day, cloudy day, heatwave)
Phase 6 — Backend API (Week 5)
•	Build FastAPI endpoints that serve model predictions and simulation results (see Section 6)
•	Connect backend to the database to store/retrieve historical and simulated data
Phase 7 — Frontend Dashboard (Week 5-6)
•	Build the dashboard UI: forecast charts, confidence bands, what-if scenario controls, dispatch/CO2 results panel
•	Connect frontend to backend APIs
Phase 8 — Integration, Testing & Polish (Week 6-7)
•	Test the full flow end-to-end: change a scenario on frontend → backend recalculates → dashboard updates
•	Fix bugs, polish UI, write documentation
Phase 9 — Final Report & Presentation (Week 7-8)
•	Write the final report (problem, related work/gaps, methodology, results, novelty, conclusion)
•	Prepare slides and a live demo of the dashboard
 
5. Team Roles & Responsibilities
Role	Owns	Deliverables
Data & Baseline Lead	Data collection, cleaning, feature engineering, naive + regression baseline	Clean merged dataset, baseline model + error scores, EDA plots
ML Model Lead	Core forecasting model, probabilistic (confidence range) forecasting, SHAP explainability	Trained model, evaluation report, SHAP plots
Backend / Simulation Lead	Dispatch simulator logic, CO2 calculator, FastAPI endpoints, database integration	Working API + database, dispatch/CO2 logic module
Frontend / Dashboard Lead	Interactive dashboard UI, what-if scenario controls, connecting to backend	Live dashboard app
 
For a 3-person team: merge the Backend/Simulation and Frontend/Dashboard roles into one person, or pair up for those two phases since they happen close together in the timeline.
 
6. Database Schema (Suggested)
6.1 historical_data
•	timestamp, region, demand_mw, solar_mw, wind_mw, temperature, cloud_cover, wind_speed, is_holiday
6.2 predictions
•	timestamp, region, predicted_demand, predicted_demand_lower, predicted_demand_upper, predicted_supply, predicted_supply_lower, predicted_supply_upper, model_version
6.3 dispatch_logs
•	timestamp, region, scenario_name, demand_forecast, supply_forecast, storage_used_mw, backup_used_mw, decision_confidence
6.4 impact_results
•	timestamp, region, backup_mw_avoided, cost_saved, co2_avoided_tons
SQLite is fine for course-project scale; move to PostgreSQL only if you want the 'production-grade' architecture to be part of your evaluation criteria.
7. Backend API Endpoints (Suggested)
Endpoint	Method	Purpose
/predict/demand	GET	Return demand forecast + confidence range for a given time window
/predict/supply	GET	Return solar/wind supply forecast + confidence range
/simulate/dispatch	POST	Accept a what-if scenario input, return dispatch decision (storage/backup)
/impact/summary	GET	Return cost saved + CO2 avoided for a given scenario/time range
/data/historical	GET	Return historical data for charting on the dashboard
 
8. Frontend Dashboard — Pages/Sections
•	Overview Page — current demand vs. supply forecast chart with confidence bands
•	What-If Scenario Panel — sliders/inputs to adjust weather assumptions (e.g., cloud cover, temperature) and re-run the forecast
•	Dispatch Decision Panel — shows recommended action (use storage / use backup) for the selected scenario
•	Impact Summary Panel — displays cost saved and CO2 avoided for the current scenario vs. a 'no forecasting' baseline
•	Model Insights Panel (optional but impressive) — SHAP-based chart showing which features are driving the current prediction
9. Tech Stack Summary
Layer	Tools
Language	Python (model + backend), JavaScript (frontend, if using React)
ML/Data	pandas, numpy, scikit-learn, XGBoost/LightGBM, Prophet, SHAP
Backend	FastAPI, uvicorn
Database	SQLite (simple) or PostgreSQL (production-style)
Frontend	React + Recharts/Chart.js, OR Streamlit (faster, all-in-one)
Version Control	GitHub — one repo, branch per feature, merge via pull requests
 
10. Timeline / Milestones (8-Week Plan)
Week	Milestone
Week 1	Data collected, cleaned, and merged. Region + datasets finalized.
Week 2	EDA complete. Naive + regression baseline built and scored.
Week 3-4	Core forecasting model + probabilistic (confidence range) forecasting + SHAP done.
Week 4-5	Dispatch simulator + CO2 calculator built and tested on sample scenarios.
Week 5	Backend API built and connected to database.
Week 5-6	Frontend dashboard built and connected to backend.
Week 6-7	Full integration testing, bug fixes, UI polish.
Week 7-8	Final report + slides + live demo rehearsal.
11. Final Deliverables Checklist
•	✔ Cleaned, merged dataset (load + weather + calendar features) for the chosen region
•	✔ Naive baseline + simple regression baseline with recorded error scores
•	✔ Core ML model with probabilistic (confidence range) forecasts, beating baseline
•	✔ SHAP-based explainability plots for the model
•	✔ Dispatch simulator logic (storage vs. backup decision engine)
•	✔ CO2 / cost impact calculator
•	✔ Working backend API (FastAPI) connected to a database
•	✔ Interactive frontend dashboard with what-if scenario controls
•	✔ Final written report (problem, research gap analysis, methodology, results, novelty)
•	✔ Slide deck + live demo for presentation day
