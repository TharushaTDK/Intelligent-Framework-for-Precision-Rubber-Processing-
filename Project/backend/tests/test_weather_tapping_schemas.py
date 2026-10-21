import math
from datetime import datetime, timedelta, timezone

import pytest

from pydantic import ValidationError

from backend.app.schemas.weather_tapping import (
    DrcPredictionRequest,
    DrcPredictionResponse,
    PriceEstimationRequest,
    PriceEstimationResponse,
    TappingSuitabilityRequest,
    TappingSuitabilityResponse,
)


# ============================================================
# TEST HELPERS
# ============================================================


COLOMBO_OFFSET = timezone(timedelta(hours=5, minutes=30))

AWARE_TIMESTAMP = datetime(
    2025,
    1,
    2,
    8,
    0,
    tzinfo=COLOMBO_OFFSET,
)


# ============================================================
# VALID REQUESTS
# ============================================================


def test_valid_drc_request_accepts_optional_fields():
    """
    A DRC request with only an aware timestamp must be
    valid; every sensor field defaults to None (observation
    missing).
    """

    request = DrcPredictionRequest(
        observation_timestamp=AWARE_TIMESTAMP,
    )

    assert request.temperature_c is None
    assert request.estate_id is None
    assert request.visible_latex_condition is None


def test_valid_drc_request_accepts_all_fields():
    """
    A fully populated DRC request must validate with all
    fields preserved exactly as supplied.
    """

    request = DrcPredictionRequest(
        observation_timestamp=AWARE_TIMESTAMP,
        temperature_c=27.5,
        relative_humidity_pct=82.0,
        rainfall_last_6h_mm=4.2,
        forecast_rain_next_6h_mm=0.0,
        rainfall_last_3days_mm=18.5,
        wind_speed_kmh=7.3,
        soil_moisture_pct=41.0,
        soil_ph=5.8,
        visible_latex_condition="Watery",
        estate_id="Estate 01",
    )

    assert request.estate_id == "Estate 01"
    assert request.visible_latex_condition == "Watery"


def test_valid_tapping_request_accepts_optional_fields():
    """
    A tapping-suitability request with only an aware
    timestamp must be valid.
    """

    request = TappingSuitabilityRequest(
        observation_timestamp=AWARE_TIMESTAMP,
    )

    assert request.soil_ph is None
    assert request.forecast_rain_next_6h_mm is None


def test_valid_price_request_accepts_optional_fields():
    """
    A price-estimation request with only an aware timestamp
    must be valid.
    """

    request = PriceEstimationRequest(
        observation_timestamp=AWARE_TIMESTAMP,
    )

    assert request.latex_drc_pct is None


# ============================================================
# TIMESTAMP-TIMEZONE VALIDATION
# ============================================================


def test_naive_observation_timestamp_is_rejected():
    """
    A timestamp without an explicit UTC offset must be
    rejected: the backend never assumes a timezone
    silently and never treats naive input as UTC.
    """

    with pytest.raises(ValidationError):
        DrcPredictionRequest(
            observation_timestamp=datetime(2025, 1, 2, 8, 0),
        )


def test_utc_offset_timestamps_are_accepted():
    """
    Timestamps carrying an explicit UTC offset (both Z and
    numeric forms) must be accepted.
    """

    utc_request = DrcPredictionRequest(
        observation_timestamp=datetime(
            2025,
            1,
            2,
            2,
            30,
            tzinfo=timezone.utc,
        ),
    )

    assert utc_request.observation_timestamp.utcoffset() == (
        timedelta(0)
    )

    colombo_request = TappingSuitabilityRequest(
        observation_timestamp=AWARE_TIMESTAMP,
    )

    assert colombo_request.observation_timestamp.utcoffset() == (
        timedelta(hours=5, minutes=30)
    )


@pytest.mark.parametrize(
    ("request_type",),
    [
        (DrcPredictionRequest,),
        (TappingSuitabilityRequest,),
        (PriceEstimationRequest,),
    ],
)
def test_all_request_types_reject_naive_timestamps(
    request_type,
):
    """
    Every request contract must reject naive timestamps.
    """

    with pytest.raises(ValidationError):
        request_type(
            observation_timestamp=datetime(2025, 1, 2, 8, 0),
        )


# ============================================================
# NON-FINITE NUMERIC INPUT REJECTION
# ============================================================


@pytest.mark.parametrize(
    ("field_name", "bad_value", "request_type"),
    [
        ("temperature_c", math.inf, DrcPredictionRequest),
        ("temperature_c", -math.inf, DrcPredictionRequest),
        ("temperature_c", math.nan, DrcPredictionRequest),
        ("soil_ph", math.nan, DrcPredictionRequest),
        (
            "rainfall_last_3days_mm",
            math.inf,
            DrcPredictionRequest,
        ),
        (
            "forecast_rain_next_6h_mm",
            math.nan,
            TappingSuitabilityRequest,
        ),
        ("latex_drc_pct", math.inf, PriceEstimationRequest),
        ("soil_moisture_pct", math.nan, PriceEstimationRequest),
    ],
)
def test_non_finite_numeric_values_are_rejected(
    field_name: str,
    bad_value: float,
    request_type,
):
    """
    Explicit NaN or infinity in any numeric field must be
    rejected at the contract boundary; only genuine
    "observation missing" (None) may reach the missing-value
    policy.
    """

    with pytest.raises(ValidationError):
        request_type(
            observation_timestamp=AWARE_TIMESTAMP,
            **{field_name: bad_value},
        )


# ============================================================
# UNEXPECTED-FIELD REJECTION (estate_name IS NOT A FEATURE)
# ============================================================


def test_estate_name_is_rejected_as_unexpected_field():
    """
    estate_name is not a model feature; supplying it must
    be rejected as an unexpected field rather than silently
    ignored or used.
    """

    with pytest.raises(ValidationError):
        DrcPredictionRequest(
            observation_timestamp=AWARE_TIMESTAMP,
            estate_name="Rubber Estate A",
        )


def test_unknown_extra_fields_are_rejected():
    """
    Any unknown extra field must be rejected (extra=forbid)
    on every request contract.
    """

    with pytest.raises(ValidationError):
        DrcPredictionRequest(
            observation_timestamp=AWARE_TIMESTAMP,
            some_future_field=1,
        )

    with pytest.raises(ValidationError):
        TappingSuitabilityRequest(
            observation_timestamp=AWARE_TIMESTAMP,
            estate_id="Estate 01",  # not a tapping feature
        )

    with pytest.raises(ValidationError):
        PriceEstimationRequest(
            observation_timestamp=AWARE_TIMESTAMP,
            estate_id="Estate 01",  # not a price feature
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("tree_count", 150.0),
        ("tapping_system", "S/2"),
        ("days_since_last_tap", 2.0),
        ("previous_collection_yield_litres", 38.0),
    ],
)
def test_fields_dropped_in_the_retrain_are_rejected(
    field_name: str,
    value,
):
    """
    The retrained models no longer use these old synthetic
    fields, so sending them must fail loudly instead of being
    quietly ignored.
    """

    with pytest.raises(ValidationError):
        DrcPredictionRequest(
            observation_timestamp=AWARE_TIMESTAMP,
            **{field_name: value},
        )


# ============================================================
# LATEX CONDITION VALIDATION
# ============================================================


@pytest.mark.parametrize(
    ("condition",),
    [
        ("",),
        ("   ",),
        ("watery",),
        ("Very watery",),
    ],
)
def test_unknown_latex_condition_is_rejected(condition: str):
    """
    Only the three labels seen in training are accepted;
    anything else would be ignored by the encoder and give a
    misleading DRC.
    """

    with pytest.raises(ValidationError):
        DrcPredictionRequest(
            observation_timestamp=AWARE_TIMESTAMP,
            visible_latex_condition=condition,
        )


def test_empty_estate_id_is_rejected():
    """
    Empty or whitespace-only estate_id must be rejected as
    malformed input.
    """

    with pytest.raises(ValidationError):
        DrcPredictionRequest(
            observation_timestamp=AWARE_TIMESTAMP,
            estate_id="  ",
        )


# ============================================================
# IDENTIFIER VALUES ARE NOT REWRITTEN BY VALIDATION
# ============================================================


def test_identifier_validation_does_not_rewrite_values():
    """
    Validation rejects malformed identifiers but never
    rewrites well-formed ones: category normalization
    happens later under the documented training policy.
    """

    request = DrcPredictionRequest(
        observation_timestamp=AWARE_TIMESTAMP,
        estate_id=" Estate 01 ",
    )

    assert request.estate_id == " Estate 01 "


# ============================================================
# RESPONSE CONTRACTS
# ============================================================


def test_drc_response_units_and_target_names():
    """
    The DRC response must expose the fixed target name and
    unit for latex dry-rubber content.
    """

    response = DrcPredictionResponse(
        model_name="drc_prediction",
        model_class="GradientBoostingRegressor",
        runtime_artifact_sha256="a" * 64,
        original_artifact_sha256="b" * 64,
        dataset_provenance="synthetic; not real-world evidence",
        observation_month=1,
        observation_day_of_year=2,
        predicted_latex_drc_pct=34.5,
    )

    assert response.target_name == "latex_drc_pct"
    assert response.unit == "percent"


def test_price_response_units_and_target_names():
    """
    The price response must expose the fixed target name and
    unit for LKR-per-kg price estimation.
    """

    response = PriceEstimationResponse(
        model_name="price_estimation",
        model_class="RandomForestRegressor",
        runtime_artifact_sha256="a" * 64,
        original_artifact_sha256="b" * 64,
        dataset_provenance="synthetic; not real-world evidence",
        observation_month=1,
        observation_day_of_year=2,
        predicted_price_lkr_per_kg=870.0,
    )

    assert response.target_name == "price_lkr_per_kg"
    assert response.unit == "LKR per kg"


def test_tapping_response_labels_and_proxy_flag():
    """
    The tapping response must expose only the three fitted
    labels, the exact class order, and the proxy marker.
    """

    response = TappingSuitabilityResponse(
        model_name="tapping_advisory",
        model_class="GradientBoostingClassifier",
        runtime_artifact_sha256="a" * 64,
        original_artifact_sha256="b" * 64,
        dataset_provenance="synthetic; not real-world evidence",
        observation_month=1,
        observation_day_of_year=2,
        predicted_label="Moderate",
        class_order=["High", "Low", "Moderate"],
        probabilities={
            "high": 0.1,
            "low": 0.2,
            "moderate": 0.7,
        },
        is_rule_derived_proxy=True,
        research_warning="Rule-derived proxy labels.",
    )

    assert response.target_name == (
        "tapping_suitability_proxy"
    )
    assert response.predicted_label == "Moderate"
    assert response.class_order == [
        "High",
        "Low",
        "Moderate",
    ]
    assert response.is_rule_derived_proxy is True


def test_tapping_response_rejects_unknown_labels():
    """
    The tapping response must reject labels outside the
    three fitted classes.
    """

    with pytest.raises(ValidationError):
        TappingSuitabilityResponse(
            model_name="tapping_advisory",
            model_class="GradientBoostingClassifier",
            runtime_artifact_sha256="a" * 64,
            original_artifact_sha256="b" * 64,
            dataset_provenance="synthetic",
            observation_month=1,
            observation_day_of_year=2,
            predicted_label="Excellent",
            class_order=["High", "Low", "Moderate"],
            probabilities={
                "high": 0.1,
                "low": 0.2,
                "moderate": 0.7,
            },
            is_rule_derived_proxy=True,
            research_warning="...",
        )
