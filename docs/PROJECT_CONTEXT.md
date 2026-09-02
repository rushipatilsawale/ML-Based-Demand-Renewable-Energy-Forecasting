Project
ML-Based Demand & Renewable Energy Forecasting — full-stack system that forecasts short-term electricity demand and solar/wind supply with confidence ranges, simulates a storage-vs-backup dispatch decision, and calculates cost/CO2 impact, shown on a live dashboard.

Region chosen: India (national + North/West/East/South/North-East grids).

Architecture
Raw Data → Data Pipeline (cleaning, feature engineering) → ML Model → Backend API (FastAPI) → Database (SQLite/PostgreSQL) → Frontend Dashboard (React+Recharts or Streamlit)

Datasets in use
Demand: Kaggle "Hourly Load India — Electrical Load Forecasting" (46,728 hourly rows, Jan 2019–Apr 2024, national + 5 regions)
Weather: Open-Meteo API (temperature, humidity, cloud cover, precipitation, wind speed, solar radiation)
CO2 factor: CEA CO2 Baseline Database — 0.710 tCO2/MWh (FY 2024-25)
Solar/wind supply data: not yet integrated (planned — Kaggle Solar Power Generation Data / SolarGeneration Karnataka, or NASA POWER as proxy)
Final merged dataset (confirmed via EDA)
46,728 rows × 19 columns, 2019-01-01 to 2024-04-30, 0 missing values, 0 duplicates
Columns: datetime, national_demand_mw, north/west/east/south/north_east_demand_mw, hour, day, month, year, day_of_week, is_weekend, temperature_2m_c, relative_humidity_pct, cloud_cover_pct, precipitation_mm, wind_speed_10m_kmh, solar_radiation_w_m2
National demand range: 95,336.56–237,361.97 MW, mean 160,487.07 MW
Peak demand hour: 11:00 (173,085 MW avg); lowest: 03:00 (143,526 MW avg)
Repo structure so far
src/
  eda/
    __init__.py
    basic_eda.py            ✅ done
    demand_analysis.py      🚧 in progress (Matplotlib Agg backend fix applied)
data/
  processed/final_merged_dataset.csv
reports/
  figures/
Progress checklist (update this as you go)
 Phase 1 — Data collection, cleaning, weather merge, final validation
 Phase 2a — Basic EDA
 Phase 2b — Demand pattern analysis (hourly/weekly/monthly/yearly/regional) — script working after Agg backend + Path import fix
 Phase 2c — Weather analysis (correlate weather vars with demand)
 Phase 2d — Correlation analysis
 Phase 2e — Renewable/solar-wind analysis (needs supply dataset integration)
 Phase 2f — EDA conclusions writeup
 Phase 3 — Baseline models (naive + regression)
 Phase 4 — Core forecasting model (XGBoost/LightGBM/Prophet + quantile regression + SHAP)
 Phase 5 — Dispatch simulator + CO2 calculator
 Phase 6 — Backend API (FastAPI)
 Phase 7 — Frontend dashboard
 Phase 8 — Integration testing & polish
 Phase 9 — Final report & presentation
Conventions
Use .venv virtual environment; record every new package in requirements.txt immediately after installing (pip freeze > requirements.txt or add manually).
Matplotlib always uses the Agg backend (no GUI available) — save figures to reports/figures/, never plt.show().
Time-series: never shuffle when splitting train/test; hold out a future time window.
One region only (India) — don't mix in other countries' data.