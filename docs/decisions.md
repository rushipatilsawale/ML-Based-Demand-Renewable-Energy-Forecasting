# Project Decisions

## Decision Log

This document records important technical decisions made during the development of the project and the reasoning behind them.

---

## D001 — Use Git Feature Branches

**Decision:**
Development work will be performed on feature branches and merged into `main` after completion.

**Reason:**
This keeps `main` stable and creates a clear development history.

Current branch used for Phase 1:

`feature/data-pipeline`

---

## D002 — Preserve Raw Data

**Decision:**
Original datasets will remain unchanged in `data/raw/`.

**Reason:**
Raw data should always be preserved so that the processing pipeline remains reproducible and traceable.

---

## D003 — Create Processed Datasets

**Decision:**
Cleaned and merged datasets will be generated under `data/processed/`.

**Reason:**
Separating raw and processed data prevents accidental modification of source data and makes the pipeline easier to understand.

---

## D004 — Use Datetime as the Integration Key

**Decision:**
Demand and weather observations will be aligned using `datetime`.

**Reason:**
Both datasets contain hourly observations covering the same period. Timestamp alignment provides a natural key for integrating demand and weather information.

---

## D005 — Use One-to-One Merge Validation

**Decision:**
The demand and weather datasets are merged using a one-to-one relationship.

**Reason:**
Each timestamp should correspond to exactly one demand observation and one weather observation. Pandas merge validation is used to detect unexpected duplication.

---

## D006 — Demand as the Primary Dataset

**Decision:**
The demand dataset is treated as the primary dataset during integration.

**Reason:**
Electricity demand is the forecasting target, while weather variables act as external/exogenous predictors.

A left join ensures that demand observations remain the reference set.

---

## D007 — Initial Representative Weather Location

**Decision:**
A representative Delhi location is used for the initial weather dataset.

**Reason:**
The current phase focuses on establishing a reproducible end-to-end data pipeline. Spatially distributed weather data can be evaluated and added in a later iteration.

**Important limitation:**
Delhi weather should not be interpreted as India's complete weather profile.

---

## D008 — Exclude Large Datasets from Git

**Decision:**
Raw and processed datasets are excluded from Git tracking.

**Reason:**

* Avoid unnecessarily large repository size.
* Keep source data separate from code.
* Allow datasets to be reproduced through the pipeline.
* Prevent accidental commits of large generated files.

---

## D009 — Validate Every Major Pipeline Stage

**Decision:**
Validation scripts are used for the demand dataset and final merged dataset.

**Reason:**
Automated validation reduces the risk of silently introducing missing values, duplicate timestamps, incorrect intervals, or structural errors before ML modeling.

Current validation scripts:

* `src/data/inspect_demand.py`
* `src/data/validate_merged.py`
