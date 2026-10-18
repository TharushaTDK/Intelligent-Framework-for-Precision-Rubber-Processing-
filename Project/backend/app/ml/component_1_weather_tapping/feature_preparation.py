from __future__ import annotations

import math
from datetime import datetime
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np

import pandas as pd


# ============================================================
# COMPONENT 1 FEATURE PREPARATION HELPERS
# ============================================================
#
# Small, model-facing helpers shared by the stateless
# Component 1 prediction service. Everything here reproduces
# the AUTHORITATIVE research behaviour documented in:
#
#   ml-research/component_1_weather_tapping/src/common.py
#
# Nothing in this module refits, retunes or alters any model
# artifact.
#


# ============================================================
# RESEARCH MISSING-VALUE POLICY
# ============================================================
#
# Equivalent to common.py clean_basic():
#
#   - inf / -inf observations are treated as missing
#     (common.py: "df.replace([np.inf, -np.inf], np.nan)")
#
#   - out-of-range observations are converted to missing in
#     the processed copy, and the fitted pipelines then
#     median-impute missing numeric values internally.
#
# The bounds below are copied EXACTLY from clean_basic().
# No new physical limits are introduced. Features that
# clean_basic() did not range-check are passed through
# unchanged (missing stays missing).
# ============================================================


NON_NEGATIVE_MM_OR_KMH_FEATURES = (
    "rainfall_last_6h_mm",
    "forecast_rain_next_6h_mm",
    "rainfall_last_3days_mm",
    "wind_speed_kmh",
)

PERCENT_0_100_FEATURES = (
    "soil_moisture_pct",
    "relative_humidity_pct",
)

# Inclusive bounds, matching pandas Series.between(lo, hi).
INCLUSIVE_RANGE_FEATURES: dict[str, tuple[float, float]] = {
    "soil_ph": (0.0, 14.0),
    "temperature_c": (-5.0, 60.0),
    "latex_drc_pct": (0.0, 100.0),
}


def prepare_numeric_feature(
    feature_name: str,
    value: float | None,
) -> float:
    """
    Apply the documented research missing-value policy to one
    numeric feature value.

    - None (omitted observation) -> missing (NaN)
    - inf / -inf -> missing (NaN), matching clean_basic()
    - out-of-range per clean_basic() -> missing (NaN)
    - everything else -> unchanged

    The fitted pipeline performs median imputation for any
    missing numeric value; this function never invents a
    reading and never substitutes an arbitrary zero.
    """

    if value is None:
        return math.nan

    numeric = float(value)

    if not math.isfinite(numeric):
        # clean_basic(): inf / -inf -> NaN
        return math.nan

    if feature_name in NON_NEGATIVE_MM_OR_KMH_FEATURES:
        # clean_basic(): negative rainfall / wind -> NaN
        if numeric < 0.0:
            return math.nan

        return numeric

    if feature_name in PERCENT_0_100_FEATURES:
        # clean_basic(): outside [0, 100] -> NaN
        if not 0.0 <= numeric <= 100.0:
            return math.nan

        return numeric

    if feature_name in INCLUSIVE_RANGE_FEATURES:
        low, high = INCLUSIVE_RANGE_FEATURES[feature_name]

        if not low <= numeric <= high:
            return math.nan

        return numeric

    # tree_count, days_since_last_tap,
    # previous_collection_yield_litres and any other numeric
    # feature were NOT range-checked by clean_basic().
    return numeric


# ============================================================
# CATEGORICAL NORMALIZATION
# ============================================================


def normalize_tapping_system(
    value: str | None,
) -> str | None:
    """
    Normalize tapping_system exactly as training did.

    common.py clean_basic():

        df["tapping_system"] = (
            df["tapping_system"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

    estate_id is intentionally NOT normalized here: the
    research pipeline never normalized it, so the runtime
    must not invent extra behaviour for it either.
    """

    if value is None:
        return None

    return value.strip().upper()


def normalize_estate_id(
    value: str | None,
) -> str | None:
    """
    estate_id is a categorical feature passed through as-is.

    The research pipeline did not transform estate_id; the
    fitted OneHotEncoder handles unseen categories with
    handle_unknown="ignore" (verified in Phase 0/1).
    """

    if value is None:
        return None

    return value


# ============================================================
# TIMESTAMP-DERIVED CALENDAR FEATURES
# ============================================================
#
# Training semantics (verified):
#
#   common.py clean_basic():
#
#       ts = pd.to_datetime(df["timestamp_local"],
#                           errors="coerce")
#       df["month"] = ts.dt.month
#       df["day_of_year"] = ts.dt.dayofyear
#
# timestamp_local is a NAIVE local wall-clock string in the
# frozen synthetic dataset (e.g. "2025-01-02 08:00"). No
# timezone database reference, UTC conversion, or daylight
# saving logic exists anywhere in the research project.
# month and day_of_year were therefore plain calendar
# arithmetic on the observation-local wall clock.
#
# The runtime reproduces this by converting the client's
# timezone-aware timestamp EXPLICITLY to the configured
# observation timezone (never silently to UTC) and reading
# the calendar fields from that wall clock.
# ============================================================


def derive_calendar_features(
    observation_timestamp: datetime,
    observation_timezone: str,
) -> tuple[int, int]:
    """
    Derive (month, day_of_year) for one observation.

    observation_timestamp MUST be timezone-aware (explicit
    UTC offset). It is converted explicitly, using zoneinfo,
    to observation_timezone - the timezone whose wall clock
    defines the estate-local calendar - and the month and
    day_of_year are read from the converted wall clock,
    exactly matching training's naive-local semantics.

    Raises ValueError for naive timestamps or unknown
    timezone names.
    """

    if (
        observation_timestamp.tzinfo is None
        or observation_timestamp.utcoffset() is None
    ):
        raise ValueError(
            "observation_timestamp must be timezone-aware; "
            "the runtime never assumes a timezone silently."
        )

    try:
        observation_zone = ZoneInfo(observation_timezone)
    except Exception as exc:
        raise ValueError(
            f"Unknown observation timezone: "
            f"{observation_timezone!r}."
        ) from exc

    local_wall_clock = observation_timestamp.astimezone(
        observation_zone
    )

    month = local_wall_clock.month

    # Equivalent to pandas ts.dt.dayofyear on the naive
    # local wall clock.
    day_of_year = local_wall_clock.timetuple().tm_yday

    return month, day_of_year


# ============================================================
# FEATURE FRAME CONSTRUCTION
# ============================================================


def build_feature_frame(
    ordered_features: Sequence[str],
    numeric_values: Mapping[str, float],
    categorical_values: Mapping[str, str | None],
) -> pd.DataFrame:
    """
    Build the single-row DataFrame passed to a fitted
    pipeline, with EXACTLY the ordered feature names taken
    from the verified runtime manifest.

    The fitted pipeline's embedded preprocessing (median /
    most-frequent imputation, scaling, one-hot encoding)
    runs inside predict(); this function only assembles the
    raw feature row.
    """

    row: dict[str, object] = {}

    for feature_name in ordered_features:
        if feature_name in categorical_values:
            row[feature_name] = categorical_values[
                feature_name
            ]

        else:
            row[feature_name] = numeric_values[feature_name]

    frame = pd.DataFrame([row], columns=list(ordered_features))

    return frame
