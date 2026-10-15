from datetime import datetime, timezone

import numpy as np
import pytest

from backend.app.ml.component_3_smokehouse.feature_builder import (
    EXPECTED_FEATURES,
    HistoricalReading,
)
from backend.app.ml.component_3_smokehouse.model_loader import (
    get_models,
)
from backend.app.services.smokehouse_service import (
    SmokehouseServiceError,
    build_fire_risk_guidance,
    predict_smokehouse_state,
)


# ============================================================
# TEST HELPERS
# ============================================================


def utc_datetime(
    hour: int,
    minute: int,
) -> datetime:
    return datetime(
        2026,
        10,
        15,
        hour,
        minute,
        tzinfo=timezone.utc,
    )


def complete_history() -> list[HistoricalReading]:
    """
    Six previous readings at exact 10-minute intervals.

    00:00 -> 40 C / 80 %
    00:10 -> 42 C / 78 %
    00:20 -> 44 C / 76 %
    00:30 -> 46 C / 74 %
    00:40 -> 48 C / 72 %
    00:50 -> 50 C / 70 %

    Current reading used by the tests:

    01:00 -> 52 C / 68 %
    """

    values = [
        (0, 40.0, 80.0),
        (10, 42.0, 78.0),
        (20, 44.0, 76.0),
        (30, 46.0, 74.0),
        (40, 48.0, 72.0),
        (50, 50.0, 70.0),
    ]

    return [
        HistoricalReading(
            recorded_at=utc_datetime(
                0,
                minute,
            ),
            temperature_c=temperature,
            humidity_pct=humidity,
        )
        for minute, temperature, humidity in values
    ]


# ============================================================
# TEST 1 — FINALIZED MODEL ARTIFACTS LOAD
# ============================================================


def test_finalized_models_load_and_are_cached():
    """
    Both finalized Component 3 model artifacts must load
    successfully.

    get_models() is cached, so repeated calls should return
    the same loaded model container.
    """

    first = get_models()
    second = get_models()

    assert first is second

    assert hasattr(
        first.drying_time,
        "predict",
    )

    assert hasattr(
        first.fire_risk,
        "predict",
    )

    assert hasattr(
        first.fire_risk,
        "predict_proba",
    )


# ============================================================
# TEST 2 — COMPLETE END-TO-END INFERENCE
# ============================================================


def test_complete_history_runs_both_finalized_models():
    """
    Validate one complete runtime inference using:

    - exact finalized 17-feature contract
    - finalized drying-time model
    - finalized ordinal fire-risk model
    - classifier-derived operator guidance

    This is a runtime regression test, not a model-accuracy test.
    """

    result = predict_smokehouse_state(
        smokehouse_id="SH01",
        layer_id="L1",
        temperature_c=52.0,
        humidity_pct=68.0,
        recorded_at=utc_datetime(1, 0),
        batch_started_at=utc_datetime(0, 0),
        number_of_sheets=40,
        layer_area_m2=8.5,
        history=complete_history(),
    )

    # --------------------------------------------------------
    # Feature contract
    # --------------------------------------------------------

    assert result.feature_row.shape == (
        1,
        17,
    )

    assert list(
        result.feature_row.columns
    ) == list(
        EXPECTED_FEATURES
    )

    assert (
        result.historical_reading_count
        == 6
    )

    assert (
        result.missing_numeric_features
        == ()
    )

    # --------------------------------------------------------
    # Regression output
    # --------------------------------------------------------

    assert np.isfinite(
        result.remaining_drying_time_hours
    )

    assert (
        result.remaining_drying_time_hours
        == pytest.approx(
            94.78255591267066,
            rel=1e-10,
            abs=1e-10,
        )
    )

    # --------------------------------------------------------
    # Fire-risk output
    # --------------------------------------------------------

    assert result.fire_risk_label in {
        "Normal",
        "Warning",
        "Critical",
    }

    probabilities = np.array(
        [
            result.normal_probability,
            result.warning_probability,
            result.critical_probability,
        ],
        dtype=float,
    )

    assert np.all(
        np.isfinite(probabilities)
    )

    assert np.all(
        probabilities >= 0
    )

    assert np.all(
        probabilities <= 1
    )

    assert float(
        probabilities.sum()
    ) == pytest.approx(
        1.0,
        abs=1e-6,
    )

    # --------------------------------------------------------
    # Guidance integration
    # --------------------------------------------------------

    assert result.guidance == (
        build_fire_risk_guidance(
            result.fire_risk_label
        )
    )


# ============================================================
# TEST 3 — FIRST READING STILL SUPPORTS INFERENCE
# ============================================================


def test_first_reading_with_missing_history_still_predicts():
    """
    A newly started layer has no previous sensor observations.

    The feature builder therefore creates missing temporal
    values, and the finalized pipeline's internal numeric
    imputation must still allow inference to complete.
    """

    result = predict_smokehouse_state(
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

    assert np.isfinite(
        result.remaining_drying_time_hours
    )

    assert result.fire_risk_label in {
        "Normal",
        "Warning",
        "Critical",
    }

    assert (
        result.normal_probability
        + result.warning_probability
        + result.critical_probability
    ) == pytest.approx(
        1.0,
        abs=1e-6,
    )

    assert result.guidance == (
        build_fire_risk_guidance(
            result.fire_risk_label
        )
    )


# ============================================================
# TEST 4 — FIRE-RISK GUIDANCE MAPPING
# ============================================================


@pytest.mark.parametrize(
    ("fire_risk_label", "expected_guidance"),
    [
        (
            "Normal",
            (
                "Normal fire-risk condition. "
                "Continue monitoring smokehouse conditions."
            ),
        ),
        (
            "Warning",
            (
                "Warning fire-risk condition. "
                "Check smokehouse conditions before making any "
                "fire-management adjustment."
            ),
        ),
        (
            "Critical",
            (
                "Critical fire-risk condition. "
                "Immediate operator inspection of smokehouse "
                "conditions is required."
            ),
        ),
    ],
)
def test_fire_risk_guidance_mapping(
    fire_risk_label: str,
    expected_guidance: str,
):
    """
    Each finalized classifier label must map to exactly one
    operator-facing status message.

    These are explanatory messages only. They are not the
    future Add Fire / Hold Fire / Reduce Fire decision rules.
    """

    assert (
        build_fire_risk_guidance(
            fire_risk_label
        )
        == expected_guidance
    )


# ============================================================
# TEST 5 — UNKNOWN GUIDANCE LABEL IS REJECTED
# ============================================================


def test_unknown_fire_risk_guidance_label_is_rejected():
    """
    Guidance must never silently accept an unknown classifier
    label.
    """

    with pytest.raises(
        SmokehouseServiceError,
        match="unknown fire-risk label",
    ):
        build_fire_risk_guidance(
            "Unknown"
        )
