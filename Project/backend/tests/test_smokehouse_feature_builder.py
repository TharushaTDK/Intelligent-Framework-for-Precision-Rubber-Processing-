from datetime import datetime, timezone

import pytest

from backend.app.ml.component_3_smokehouse.feature_builder import (
    EXPECTED_FEATURES,
    HistoricalReading,
    build_feature_row,
)


# ============================================================
# SHARED TIMESTAMP HELPER
# ============================================================


def utc_datetime(
    hour: int,
    minute: int,
) -> datetime:
    """
    Create a timezone-aware UTC datetime for the test batch.
    """

    return datetime(
        2026,
        10,
        15,
        hour,
        minute,
        tzinfo=timezone.utc,
    )


# ============================================================
# TEST 1 — EXACT FEATURE CONTRACT
# ============================================================


def test_feature_builder_returns_exact_17_features():
    """
    The runtime feature builder must always return exactly the
    finalized 17 model features in the correct order.
    """

    result = build_feature_row(
        smokehouse_id="SH01",
        layer_id="L1",
        temperature_c=40.0,
        humidity_pct=80.0,
        recorded_at=utc_datetime(0, 10),
        batch_started_at=utc_datetime(0, 0),
        number_of_sheets=40,
        layer_area_m2=8.5,
        history=[],
    )

    assert result.features.shape == (
        1,
        17,
    )

    assert list(
        result.features.columns
    ) == list(
        EXPECTED_FEATURES
    )


# ============================================================
# TEST 2 — FIRST READING / NO HISTORY
# ============================================================


def test_first_reading_has_expected_missing_history_features():
    """
    For the first reading there is no previous sensor history.

    Therefore:
    - rate features are missing
    - exact 30-minute change features are missing
    - exact 60-minute change features are missing
    - rolling std values are 0 because only the current value
      exists in the rolling window
    """

    result = build_feature_row(
        smokehouse_id="SH01",
        layer_id="L1",
        temperature_c=40.0,
        humidity_pct=80.0,
        recorded_at=utc_datetime(0, 10),
        batch_started_at=utc_datetime(0, 0),
        number_of_sheets=40,
        layer_area_m2=8.5,
        history=[],
    )

    row = result.features.iloc[0]

    assert (
        result.historical_reading_count
        == 0
    )

    assert (
        result.missing_numeric_features
        == (
            "Temperature_Rate_per_10min",
            "Humidity_Rate_per_10min",
            "Temperature_Change_30m",
            "Humidity_Change_30m",
            "Temperature_Change_60m",
            "Humidity_Change_60m",
        )
    )

    assert row["Smokehouse_ID"] == "SH01"
    assert row["Layer_ID"] == "L1"

    assert row["Temperature_C"] == pytest.approx(
        40.0
    )

    assert row["Humidity_%"] == pytest.approx(
        80.0
    )

    assert row[
        "Elapsed_Drying_Hours"
    ] == pytest.approx(
        10 / 60
    )

    assert row[
        "Temperature_RollStd_30m"
    ] == pytest.approx(
        0.0
    )

    assert row[
        "Humidity_RollStd_30m"
    ] == pytest.approx(
        0.0
    )

    assert row[
        "Temperature_RollStd_60m"
    ] == pytest.approx(
        0.0
    )

    assert row[
        "Humidity_RollStd_60m"
    ] == pytest.approx(
        0.0
    )


# ============================================================
# TEST 3 — RATE NORMALIZATION
# ============================================================


def test_rate_is_normalized_to_10_minutes():
    """
    Rate features must use the actual time gap between the
    latest previous reading and the current reading.

    Previous:
        00:20 -> 42 C / 78 %

    Current:
        00:40 -> 46 C / 74 %

    Actual gap = 20 minutes.

    Temperature:
        (46 - 42) / 20 * 10 = +2

    Humidity:
        (74 - 78) / 20 * 10 = -2
    """

    history = [
        HistoricalReading(
            recorded_at=utc_datetime(
                0,
                20,
            ),
            temperature_c=42.0,
            humidity_pct=78.0,
        ),
    ]

    result = build_feature_row(
        smokehouse_id="SH01",
        layer_id="L1",
        temperature_c=46.0,
        humidity_pct=74.0,
        recorded_at=utc_datetime(0, 40),
        batch_started_at=utc_datetime(0, 0),
        number_of_sheets=40,
        layer_area_m2=8.5,
        history=history,
    )

    row = result.features.iloc[0]

    assert row[
        "Temperature_Rate_per_10min"
    ] == pytest.approx(
        2.0
    )

    assert row[
        "Humidity_Rate_per_10min"
    ] == pytest.approx(
        -2.0
    )


# ============================================================
# TEST 4 — EXACT 30 / 60 MINUTE LAGS + ROLLING WINDOWS
# ============================================================


def test_exact_lags_and_rolling_windows_with_irregular_history():
    """
    Validate the finalized temporal logic using irregular
    observation timing.

    Historical readings:

        00:10 -> 40 C / 80 %
        00:20 -> 42 C / 78 %
        00:40 -> 46 C / 74 %

    Current:

        01:10 -> 52 C / 68 %

    Expected:

    Rate:
        latest previous = 00:40
        gap = 30 minutes

        temperature:
        (52 - 46) / 30 * 10 = +2

        humidity:
        (68 - 74) / 30 * 10 = -2

    Exact 30-minute lag:
        01:10 - 30 min = 00:40

        temperature = +6
        humidity = -6

    Exact 60-minute lag:
        01:10 - 60 min = 00:10

        temperature = +12
        humidity = -12
    """

    history = [
        HistoricalReading(
            recorded_at=utc_datetime(
                0,
                10,
            ),
            temperature_c=40.0,
            humidity_pct=80.0,
        ),
        HistoricalReading(
            recorded_at=utc_datetime(
                0,
                20,
            ),
            temperature_c=42.0,
            humidity_pct=78.0,
        ),
        HistoricalReading(
            recorded_at=utc_datetime(
                0,
                40,
            ),
            temperature_c=46.0,
            humidity_pct=74.0,
        ),
    ]

    result = build_feature_row(
        smokehouse_id="SH01",
        layer_id="L1",
        temperature_c=52.0,
        humidity_pct=68.0,
        recorded_at=utc_datetime(1, 10),
        batch_started_at=utc_datetime(0, 0),
        number_of_sheets=40,
        layer_area_m2=8.5,
        history=history,
    )

    row = result.features.iloc[0]

    assert (
        result.historical_reading_count
        == 3
    )

    assert (
        result.missing_numeric_features
        == ()
    )

    # --------------------------------------------------------
    # Rate features
    # --------------------------------------------------------

    assert row[
        "Temperature_Rate_per_10min"
    ] == pytest.approx(
        2.0
    )

    assert row[
        "Humidity_Rate_per_10min"
    ] == pytest.approx(
        -2.0
    )

    # --------------------------------------------------------
    # Exact 30-minute lag
    # --------------------------------------------------------

    assert row[
        "Temperature_Change_30m"
    ] == pytest.approx(
        6.0
    )

    assert row[
        "Humidity_Change_30m"
    ] == pytest.approx(
        -6.0
    )

    # --------------------------------------------------------
    # Exact 60-minute lag
    # --------------------------------------------------------

    assert row[
        "Temperature_Change_60m"
    ] == pytest.approx(
        12.0
    )

    assert row[
        "Humidity_Change_60m"
    ] == pytest.approx(
        -12.0
    )

    # --------------------------------------------------------
    # 30-minute rolling population std
    #
    # Window:
    #   00:40 -> 46
    #   01:10 -> 52
    #
    # Population std = 3
    # --------------------------------------------------------

    assert row[
        "Temperature_RollStd_30m"
    ] == pytest.approx(
        3.0
    )

    assert row[
        "Humidity_RollStd_30m"
    ] == pytest.approx(
        3.0
    )

    # --------------------------------------------------------
    # 60-minute rolling population std
    #
    # Values:
    #   temperature = [40, 42, 46, 52]
    #   humidity    = [80, 78, 74, 68]
    # --------------------------------------------------------

    assert row[
        "Temperature_RollStd_60m"
    ] == pytest.approx(
        4.58257569495584
    )

    assert row[
        "Humidity_RollStd_60m"
    ] == pytest.approx(
        4.58257569495584
    )
