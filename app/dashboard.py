import os
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ML-Based Demand & Renewable Energy Forecasting",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

SOLAR_UNCERTAINTY_FILE = (
    "data/processed/solar_uncertainty.csv"
)

WIND_UNCERTAINTY_FILE = (
    "data/processed/wind_uncertainty.csv"
)

STORAGE_FILE = (
    "data/processed/storage_backup_simulation.csv"
)

COST_CO2_FILE = (
    "data/processed/cost_co2_impact.csv"
)

UNCERTAINTY_SUMMARY_FILE = (
    "reports/uncertainty_summary.csv"
)

STORAGE_SUMMARY_FILE = (
    "reports/storage_backup_summary.csv"
)

COST_CO2_SUMMARY_FILE = (
    "reports/cost_co2_summary.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_csv(path):

    if not os.path.exists(path):
        return None

    return pd.read_csv(path)


def format_rupees(value):

    return f"₹{value:,.2f}"


def format_number(value):

    return f"{value:,.2f}"


# ============================================================
# LOAD DATA
# ============================================================

solar = load_csv(SOLAR_UNCERTAINTY_FILE)
wind = load_csv(WIND_UNCERTAINTY_FILE)

storage = load_csv(STORAGE_FILE)
cost_co2 = load_csv(COST_CO2_FILE)

uncertainty_summary = load_csv(
    UNCERTAINTY_SUMMARY_FILE
)

storage_summary = load_csv(
    STORAGE_SUMMARY_FILE
)

cost_co2_summary = load_csv(
    COST_CO2_SUMMARY_FILE
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "⚡ ML-Based Demand & Renewable Energy Forecasting"
)

st.markdown(
    """
    ### Final Energy Forecasting & Impact Dashboard

    This dashboard brings together the forecasting,
    renewable-energy, uncertainty, storage, backup,
    cost, and CO₂ analysis developed throughout the project.
    """
)


st.info(
    "The dashboard presents validated project outputs. "
    "Storage, cost, and CO₂ results are scenario-based."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Forecasting",
        "Renewable Energy",
        "Uncertainty",
        "Storage vs Backup",
        "Cost & CO₂",
        "Project Limitations"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Project Phases",
            "14"
        )

    with col2:

        st.metric(
            "Forecasting",
            "Completed"
        )

    with col3:

        st.metric(
            "Renewable Analysis",
            "Completed"
        )

    with col4:

        st.metric(
            "Impact Analysis",
            "Completed"
        )

    st.subheader("End-to-End Pipeline")

    st.markdown(
        """
        **Demand Data**

        ↓

        **Data Cleaning & EDA**

        ↓

        **Feature Engineering**

        ↓

        **Demand Forecasting**

        ↓

        **Model Comparison & Selection**

        ↓

        **SHAP Explainability**

        ↓

        **Solar & Wind Forecasting**

        ↓

        **Prediction Uncertainty**

        ↓

        **Battery Storage Simulation**

        ↓

        **Cost & CO₂ Impact**

        ↓

        **Final Dashboard**
        """
    )


# ============================================================
# FORECASTING
# ============================================================

elif page == "Forecasting":

    st.header("Forecasting Results")

    if solar is None:

        st.error(
            "Solar forecasting output is unavailable."
        )

    else:

        st.subheader(
            "Solar Forecast"
        )

        solar_display = solar.copy()

        solar_display["datetime"] = pd.to_datetime(
            solar_display["datetime"]
        )

        st.line_chart(
            solar_display,
            x="datetime",
            y="prediction"
        )

        st.subheader(
            "Solar Prediction Data"
        )

        st.dataframe(
            solar_display.head(100),
            width="stretch"
        )

    if wind is not None:

        st.subheader(
            "Wind-Speed Forecast"
        )

        wind_display = wind.copy()

        wind_display["datetime"] = pd.to_datetime(
            wind_display["datetime"]
        )

        st.line_chart(
            wind_display,
            x="datetime",
            y="prediction"
        )

        st.dataframe(
            wind_display.head(100),
            width="stretch"
        )


# ============================================================
# RENEWABLE ENERGY
# ============================================================

elif page == "Renewable Energy":

    st.header(
        "Renewable Energy Forecasting"
    )

    col1, col2 = st.columns(2)

    if solar is not None:

        with col1:

            st.metric(
                "Solar Records",
                f"{len(solar):,}"
            )

            st.metric(
                "Average Solar Prediction",
                f"{solar['prediction'].mean():,.2f} kW"
            )

    if wind is not None:

        with col2:

            st.metric(
                "Wind Records",
                f"{len(wind):,}"
            )

            st.metric(
                "Average Wind Prediction",
                f"{wind['prediction'].mean():,.2f} m/s"
            )

    st.subheader(
        "Renewable Data"
    )

    if solar is not None:

        st.write(
            "Solar generation forecast"
        )

        st.dataframe(
            solar.head(100),
            width="stretch"
        )

    if wind is not None:

        st.write(
            "Wind-speed forecast"
        )

        st.dataframe(
            wind.head(100),
            width="stretch"
        )


# ============================================================
# UNCERTAINTY
# ============================================================

elif page == "Uncertainty":

    st.header(
        "Forecast Uncertainty"
    )

    if uncertainty_summary is not None:

        st.subheader(
            "Uncertainty Summary"
        )

        st.dataframe(
            uncertainty_summary,
            width="stretch"
        )

    col1, col2 = st.columns(2)

    if solar is not None:

        with col1:

            st.subheader(
                "Solar Prediction Interval"
            )

            solar_uncertainty = solar.copy()

            solar_uncertainty["datetime"] = (
                pd.to_datetime(
                    solar_uncertainty["datetime"]
                )
            )

            st.line_chart(
                solar_uncertainty,
                x="datetime",
                y=[
                    "prediction",
                    "lower_bound",
                    "upper_bound"
                ]
            )

    if wind is not None:

        with col2:

            st.subheader(
                "Wind Prediction Interval"
            )

            wind_uncertainty = wind.copy()

            wind_uncertainty["datetime"] = (
                pd.to_datetime(
                    wind_uncertainty["datetime"]
                )
            )

            st.line_chart(
                wind_uncertainty,
                x="datetime",
                y=[
                    "prediction",
                    "lower_bound",
                    "upper_bound"
                ]
            )


# ============================================================
# STORAGE VS BACKUP
# ============================================================

elif page == "Storage vs Backup":

    st.header(
        "Storage vs Backup"
    )

    if storage_summary is not None:

        summary = storage_summary.set_index(
            "metric"
        )["value"]

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Backup Without Storage",
                f"{summary['Total Backup Without Storage (kWh)']:,.2f} kWh"
            )

        with col2:

            st.metric(
                "Backup With Storage",
                f"{summary['Total Backup With Storage (kWh)']:,.2f} kWh"
            )

        with col3:

            st.metric(
                "Backup Reduction",
                f"{summary['Backup Reduction (%)']:.2f}%"
            )

    if storage is not None:

        st.subheader(
            "Backup Requirement Comparison"
        )

        storage_display = storage.copy()

        storage_display["datetime"] = (
            pd.to_datetime(
                storage_display["datetime"]
            )
        )

        st.line_chart(
            storage_display,
            x="datetime",
            y=[
                "backup_only_kw",
                "backup_with_storage_kw"
            ]
        )

        st.subheader(
            "Battery State of Charge"
        )

        st.line_chart(
            storage_display,
            x="datetime",
            y="battery_soc_kwh"
        )


# ============================================================
# COST & CO2
# ============================================================

elif page == "Cost & CO₂":

    st.header(
        "Cost & CO₂ Impact"
    )

    if cost_co2_summary is not None:

        summary = cost_co2_summary.set_index(
            "metric"
        )["value"]

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Cost Without Storage",
                format_rupees(
                    summary[
                        "Total Cost Without Storage (Rs.)"
                    ]
                )
            )

        with col2:

            st.metric(
                "Cost With Storage",
                format_rupees(
                    summary[
                        "Total Cost With Storage (Rs.)"
                    ]
                )
            )

        with col3:

            st.metric(
                "Cost Savings",
                format_rupees(
                    summary[
                        "Cost Savings (Rs.)"
                    ]
                )
            )

        col4, col5, col6 = st.columns(3)

        with col4:

            st.metric(
                "CO₂ Without Storage",
                f"{summary['Total CO2 Without Storage (kg)']:,.2f} kg"
            )

        with col5:

            st.metric(
                "CO₂ With Storage",
                f"{summary['Total CO2 With Storage (kg)']:,.2f} kg"
            )

        with col6:

            st.metric(
                "CO₂ Reduction",
                f"{summary['CO2 Reduction (%)']:.2f}%"
            )

        st.subheader(
            "Cost & CO₂ Summary"
        )

        st.dataframe(
            cost_co2_summary,
            width="stretch"
        )

    if cost_co2 is not None:

        st.subheader(
            "Cost Savings Over Simulation"
        )

        cost_display = cost_co2.copy()

        cost_display["datetime"] = (
            pd.to_datetime(
                cost_display["datetime"]
            )
        )

        st.line_chart(
            cost_display,
            x="datetime",
            y="cost_savings_rs"
        )


# ============================================================
# LIMITATIONS
# ============================================================

elif page == "Project Limitations":

    st.header(
        "Project Limitations"
    )

    st.warning(
        "These limitations should be considered when "
        "interpreting the final dashboard."
    )

    st.markdown(
        """
        ### Renewable Data

        - The solar dataset covers approximately one month.
        - Wind data represents meteorological wind speed,
          not measured turbine generation.
        - Wind power was therefore treated as estimated
          wind-power potential.

        ### Storage Simulation

        - The storage simulation uses solar forecast data.
        - Solar and wind datasets are not temporally synchronized.
        - Demand in the storage simulation is scenario-based,
          not measured historical demand.

        ### Cost & CO₂

        - Electricity cost is a scenario/reference assumption.
        - Grid emission factor is a reference factor.
        - Cost and CO₂ results inherit the assumptions
          of the storage simulation.

        ### Overall Interpretation

        The system demonstrates an end-to-end ML-based
        forecasting and energy-impact analysis workflow.

        The storage, cost, and CO₂ outputs should be interpreted
        as scenario analysis rather than a complete historical
        India-wide grid simulation.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ML-Based Demand & Renewable Energy Forecasting | "
    "Final Phase 14 Dashboard"
)