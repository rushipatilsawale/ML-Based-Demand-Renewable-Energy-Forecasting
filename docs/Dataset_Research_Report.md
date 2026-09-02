Dataset Research Report
ML-Based Demand & Renewable Energy Forecasting
Curated datasets, APIs and sources for demand, solar, wind, and emissions data
How to Use This Document
We recommend picking ONE consistent region/country so demand data, weather data, and emission factors all align in time and geography. Two ready-made combos are suggested at the end of this document (Section 6) — an India-based combo and a US/PJM-based combo — pick whichever is easier for your team to access and matches your pitch.
1. Electricity Demand / Load Datasets
1.1 Hourly Load India — Electrical Load Forecasting (Recommended for India-based project)
https://www.kaggle.com/datasets/shubhamvashisht/hourly-load-india-electrical-load-forecasting
•	Source: Kaggle
•	Coverage: National + regional (Northern, Western, Southern, Eastern grids) hourly electricity load for India, Jan 2019 - Apr 2024
•	Size: 46,728 hourly entries
•	Includes matching temperature data as a supplementary file
•	Why it fits: directly usable for short-term load forecasting (STLF); India-specific, so it lines up with CEA emission factor data (see Section 4)
1.2 Hourly Energy Consumption — PJM Interconnection (Recommended for US-based project)
https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
•	Source: Kaggle
•	Coverage: Multiple US grid regions (AEP, COMED, DAYTON, etc.) — the AEP_hourly file alone has 121,273 hourly records
•	Why it fits: very clean, widely used in tutorials/papers, easy to pair with EIA data for cross-verification
1.3 EIA Open Data (US Energy Information Administration)
https://www.eia.gov/opendata/
•	Source: US Government (free API, registration required for key)
•	Coverage: Real hourly electricity demand by US grid region, updated continuously
•	Why it fits: authoritative, real-time-capable, good if you want to demo 'live' data pulling in your dashboard
1.4 ENTSO-E Transparency Platform (Europe)
https://transparency.entsoe.eu/
•	Source: European grid operators consortium (free, registration required)
•	Coverage: Hourly load data across European countries
•	Why it fits: good alternative if you want a European case study or want to test regional generalization
1.5 Grid-India / POSOCO Reports
https://posoco.in/
•	Source: Government of India — Power System Operation Corporation
•	Coverage: Daily/hourly grid demand reports for Indian states and national grid
•	Why it fits: official Indian source, useful to validate or supplement the Kaggle India dataset (1.1)
 
2. Weather Data (for demand + solar/wind supply features)
2.1 Open-Meteo API (Recommended)
https://open-meteo.com/
•	Source: Open-source, aggregates national weather services (NOAA, DWD, ECMWF, etc.)
•	Cost: Completely free, no API key needed, ~10,000 requests/day
•	Coverage: Historical weather back to 1940, plus 16-day forecasts, for any lat/long globally
•	Variables: temperature, cloud cover, wind speed, precipitation, solar radiation (GHI), humidity
•	Why it fits: easiest to integrate — no signup, no key, works immediately. Best choice for a course project timeline
2.2 NASA POWER API
https://power.larc.nasa.gov/
•	Source: NASA (free, no key required)
•	Coverage: Solar radiation and wind speed data, especially strong for solar/wind resource modeling
•	Why it fits: purpose-built for renewable energy applications, good secondary source for solar/wind supply features
2.3 OpenWeatherMap
https://openweathermap.org/api
•	Source: Commercial (free tier available with signup)
•	Coverage: Real-time + historical weather data
•	Why it fits: good backup option if Open-Meteo lacks a specific variable you need
3. Solar & Wind Generation Datasets
3.1 Solar Power Generation Data (Two Indian Plants)
https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
•	Source: Kaggle
•	Coverage: 34 days of generation + sensor data (irradiation, ambient/module temperature) from two solar power plants in India, 15-minute intervals
•	Why it fits: real plant-level data with matching weather sensors — ideal for the supply-forecasting half of the project
3.2 SolarGeneration — Hassan, Karnataka, India
https://www.kaggle.com/datasets/arunkanagolkar/solargeneration
•	Source: Kaggle, collected from an operational solar plant near Hassan, Karnataka
•	Includes: GHI, DNI, DHI (solar radiation components), temperature, wind speed, and PV energy output
•	Why it fits: another India-based option if you want to keep the whole project India-focused alongside dataset 1.1
3.3 Wind Turbine SCADA Dataset (Turkey)
https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset
•	Source: Kaggle
•	Coverage: 10-minute interval data — timestamp, active power output (kW), wind speed, theoretical power curve, wind direction
•	Why it fits: the most widely used and well-documented wind SCADA dataset; used in the Scientific Reports 2025 ensemble wind-forecasting paper referenced in your literature review
3.4 Wind Power Generation — Germany (4 Grid Operators)
https://www.kaggle.com/datasets/jorgesandoval/wind-power-generation
•	Source: Kaggle, aggregated from 50Hertz, Amprion, TenneT TSO, TransnetBW
•	Coverage: 15-minute interval wind generation data, Aug 2019 - Sep 2020
•	Why it fits: good if you want grid-scale (not single-turbine) wind supply data to match against national demand data
 
4. CO2 Emission Factor Data (for the Impact Calculator)
4.1 CEA CO2 Baseline Database for the Indian Power Sector (Recommended if using India data)
https://cea.nic.in/
•	Source: Central Electricity Authority (CEA), Ministry of Power, Government of India
•	Latest version: Version 21.0 (Nov 2025), covering FY 2024-25
•	Key figure: India's grid-wide weighted average emission factor is 0.710 tCO2/MWh for FY 2024-25 (declined from 0.774 tCO2/MWh in FY 2013-14)
•	Why it fits: this is the authoritative, most current number to convert 'MW of backup power avoided' into 'tons of CO2 avoided' in your impact calculator
4.2 EPA eGRID (US Emission Factors)
https://www.epa.gov/egrid
•	Source: US Environmental Protection Agency
•	Coverage: CO2-per-MWh emission factors broken down by US grid region
•	Why it fits: use this instead of CEA if your demand/supply datasets are US-based (e.g., PJM or EIA data)
5. Summary Table — All Sources at a Glance
Category	Region	Source
Demand/Load	India	Hourly Load India (Kaggle)

Demand/Load	USA	PJM Hourly Energy Consumption (Kaggle)

Demand/Load	USA	EIA Open Data API

Demand/Load	Europe	ENTSO-E Transparency Platform

Demand/Load	India	Grid-India / POSOCO

Weather	Global	Open-Meteo API

Weather	Global	NASA POWER API

Weather	Global	OpenWeatherMap API

Solar Supply	India	Solar Power Generation Data (Kaggle)

Solar Supply	India	SolarGeneration - Karnataka (Kaggle)

Wind Supply	Turkey	Wind Turbine SCADA Dataset (Kaggle)

Wind Supply	Germany	Wind Power Generation (Kaggle)

CO2 Factors	India	CEA CO2 Baseline Database

CO2 Factors	USA	EPA eGRID

 
6. Recommended Combos (Pick One)
Combo A — India-Focused (Recommended)
•	Demand: Hourly Load India (Kaggle) — Section 1.1
•	Weather: Open-Meteo API, queried for the same Indian region/city — Section 2.1
•	Solar supply: Solar Power Generation Data or SolarGeneration Karnataka — Section 3.1 / 3.2
•	Wind supply: use NASA POWER wind speed data as a proxy if a matching Indian wind SCADA dataset isn't available, or supplement with the Turkey SCADA dataset for methodology only
•	CO2 factor: CEA CO2 Baseline Database, 0.710 tCO2/MWh — Section 4.1
Why this combo: everything lines up geographically and can be tied back to your pitch's India-relevant framing, and CEA gives you a precise, citable, current emission factor.
Combo B — US-Focused (Alternative)
•	Demand: PJM Hourly Energy Consumption (Kaggle) or EIA Open Data API — Section 1.2 / 1.3
•	Weather: Open-Meteo API, queried for the PJM region — Section 2.1
•	Solar/Wind supply: use NASA POWER for the same region as a proxy supply signal
•	CO2 factor: EPA eGRID — Section 4.2
Why this combo: PJM data is extremely clean and widely benchmarked in research (including the IISE PG&E 2025 paper from your literature review), making it easy to compare your results against published baselines.
7. Practical Notes Before You Start
•	Check date ranges overlap: your demand, weather, and solar/wind datasets must cover the same time period, or you'll have to trim to the common overlap
•	Match resolution: some datasets are hourly, others are 10-15 minute intervals — resample everything to a single common resolution (hourly is simplest)
•	Keep raw and processed data separate: save raw downloads as-is, do all cleaning/merging in a separate processed folder/script so it's reproducible
•	Version-control the data pipeline script (not the raw data files) on GitHub so the whole team works from the same cleaned dataset
