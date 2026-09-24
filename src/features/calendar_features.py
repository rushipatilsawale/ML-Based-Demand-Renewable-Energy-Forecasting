"""Single source of truth for calendar-driven features: IMD season and Indian festivals.

Both the historical feature builder and the real-time forecasting engine import from
here, so training rows and live/future rows always encode season and festivals the
same way. Without that shared definition, SHAP could explain a historical festival
effect but the future rows would carry blank season/festival values and the live
explanation would silently break.

Numeric features produced (safe for every sklearn model):
    season_winter, season_summer, season_monsoon, season_post_monsoon  (one-hot, 0/1)
    is_festival                                                        (0/1)
Human-readable labels produced for the dashboard / SHAP narrative (NOT model inputs):
    season         e.g. "Monsoon"
    festival_name  e.g. "Diwali" or "None"
"""

from datetime import date
from functools import lru_cache

import pandas as pd

# Indian Meteorological Department four-season convention, keyed by month number.
SEASON_BY_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-monsoon", 11: "Post-monsoon",
}
SEASONS = ["Winter", "Summer", "Monsoon", "Post-monsoon"]

# Lunar-festival civil dates. 2019-2024 fall inside the historical window; 2025-2027
# extend coverage so the real-time engine (which anchors at the current real-world
# hour) can flag festivals in the forecast window. These are approximate observed
# dates -- edit freely if a specific year needs correction.
LUNAR_FESTIVALS = {
    "Diwali": ["2019-10-27", "2020-11-14", "2021-11-04", "2022-10-24", "2023-11-12",
               "2025-10-20", "2026-11-08", "2027-10-29"],
    "Holi": ["2019-03-21", "2020-03-10", "2021-03-29", "2022-03-18", "2023-03-08", "2024-03-25",
             "2025-03-14", "2026-03-04", "2027-03-22"],
    "Eid al-Fitr": ["2019-06-05", "2020-05-25", "2021-05-14", "2022-05-03", "2023-04-22", "2024-04-10",
                    "2025-03-31", "2026-03-20", "2027-03-10"],
    "Eid al-Adha": ["2019-08-12", "2020-08-01", "2021-07-21", "2022-07-10", "2023-06-29",
                    "2025-06-07", "2026-05-27", "2027-05-17"],
    "Dussehra": ["2019-10-08", "2020-10-25", "2021-10-15", "2022-10-05", "2023-10-24",
                 "2025-10-01", "2026-10-20", "2027-10-09"],
    "Raksha Bandhan": ["2019-08-15", "2020-08-03", "2021-08-22", "2022-08-11", "2023-08-30",
                       "2025-08-09", "2026-08-28", "2027-08-17"],
    "Maha Shivratri": ["2019-03-04", "2020-02-21", "2021-03-11", "2022-03-01", "2023-02-18", "2024-03-08",
                       "2025-02-26", "2026-02-15", "2027-02-04"],
    "Guru Nanak Jayanti": ["2019-11-12", "2020-11-30", "2021-11-19", "2022-11-08", "2023-11-27",
                           "2025-11-05", "2026-11-24", "2027-11-13"],
    "Buddha Purnima": ["2019-05-18", "2020-05-07", "2021-05-26", "2022-05-16", "2023-05-05",
                       "2025-05-12", "2026-05-01", "2027-05-20"],
    "Mahavir Jayanti": ["2019-04-17", "2020-04-06", "2021-04-25", "2022-04-14", "2023-04-04", "2024-04-21",
                        "2025-04-10", "2026-03-31", "2027-04-18"],
}
# Fixed-date national holidays are generated for every year in the covered range.
FIXED_HOLIDAY_MONTH_DAY = {
    "New Year": (1, 1),
    "Republic Day": (1, 26),
    "Labour Day": (5, 1),
    "Independence Day": (8, 15),
    "Gandhi Jayanti": (10, 2),
    "Christmas": (12, 25),
}
_FIXED_HOLIDAY_YEARS = range(2018, 2031)

SEASON_ONEHOT = [f"season_{s.lower().replace('-', '_')}" for s in SEASONS]
NUMERIC_CALENDAR_FEATURES = SEASON_ONEHOT + ["is_festival"]
LABEL_COLUMNS = ["season", "festival_name"]
# Not "None": pandas.read_csv parses the literal "None" back as NaN on round-trip.
NO_FESTIVAL = "No festival"


def season_name(month):
    return SEASON_BY_MONTH[int(month)]


@lru_cache(maxsize=1)
def festival_lookup():
    """Map every festival date -> festival name across the covered years."""
    lookup = {}
    for name, dates in LUNAR_FESTIVALS.items():
        for value in dates:
            lookup[pd.Timestamp(value).date()] = name
    for name, (month, day) in FIXED_HOLIDAY_MONTH_DAY.items():
        for year in _FIXED_HOLIDAY_YEARS:
            lookup[date(year, month, day)] = name
    return lookup


def festival_name_for(ts):
    return festival_lookup().get(pd.Timestamp(ts).date(), NO_FESTIVAL)


def add_calendar_features(df, datetime_col="datetime"):
    """Add season one-hot, is_festival, and human labels to a DataFrame in place."""
    stamp = pd.to_datetime(df[datetime_col])
    df["season"] = stamp.dt.month.map(season_name)
    for slug, season in zip(SEASON_ONEHOT, SEASONS):
        df[slug] = (df["season"] == season).astype("int8")
    df["festival_name"] = [festival_name_for(ts) for ts in stamp]
    df["is_festival"] = (df["festival_name"] != NO_FESTIVAL).astype("int8")
    return df


def calendar_features_for_timestamp(ts):
    """Return numeric calendar features + labels for a single timestamp.

    Used by the real-time engine so future rows carry exactly the same season and
    festival encoding the models were trained on.
    """
    season = season_name(pd.Timestamp(ts).month)
    festival = festival_name_for(ts)
    values = {slug: float(season == s) for slug, s in zip(SEASON_ONEHOT, SEASONS)}
    values["is_festival"] = float(festival != NO_FESTIVAL)
    values["season"] = season
    values["festival_name"] = festival
    return values
