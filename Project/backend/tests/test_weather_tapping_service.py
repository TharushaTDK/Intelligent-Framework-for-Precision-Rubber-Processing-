import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.ml.component_1_weather_tapping import (
    model_loader,
)
from backend.app.ml.component_1_weather_tapping.model_loader import (
    EXPECTED_DRC_PIPELINE_SHA256,
    EXPECTED_PRICE_PIPELINE_SHA256,
    EXPECTED_TAPPING_PIPELINE_SHA256,
    TAPPING_CLASSES_IN_PROBABILITY_ORDER,
    ModelLoadError,
    get_models,
)

from backend.app.schemas.weather_tapping import (
    DrcPredictionRequest,
    PriceEstimationRequest,
    TappingSuitabilityRequest,
)

from backend.app.services import weather_tapping_service
from backend.app.services.weather_tapping_service import (
    OBSERVATION_TIMEZONE,
    WeatherTappingInputError,
    WeatherTappingModelError,
    predict_drc,
    predict_price_estimation,
    predict_tapping_suitability,
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


DRC_FEATURES = [
    "temperature_c",
    "relative_humidity_pct",
    "rainfall_last_6h_mm",
    "forecast_rain_next_6h_mm",
    "rainfall_last_3days_mm",
    "wind_speed_kmh",
    "soil_moisture_pct",
    "soil_ph",
    "estate_id",
    "tree_count",
    "tapping_system",
    "days_since_last_tap",
    "previous_collection_yield_litres",
    "month",
    "day_of_year",
]

PRICE_FEATURES = [
    "latex_drc_pct",
    "temperature_c",
    "relative_humidity_pct",
    "rainfall_last_6h_mm",
    "soil_moisture_pct",
    "soil_ph",
    "tree_count",
    "tapping_system",
    "month",
    "day_of_year",
]

TAPPING_FEATURES = [
    "temperature_c",
    "relative_humidity_pct",
    "rainfall_last_6h_mm",
    "forecast_rain_next_6h_mm",
    "rainfall_last_3days_mm",
    "wind_speed_kmh",
    "soil_moisture_pct",
    "soil_ph",
    "tree_count",
    "tapping_system",
    "days_since_last_tap",
    "month",
    "day_of_year",
]


STUB_MANIFEST = {
    "models": [
        {
            "name": "drc_prediction",
            "model_class": "GradientBoostingRegressor",
            "runtime_sha256": "1" * 64,
            "original_artifact": {"sha256": "2" * 64},
            "features_in_pipeline_order": DRC_FEATURES,
            "dataset_provenance": (
                "synthetic; not real-world evidence"
            ),
        },
        {
            "name": "price_estimation",
            "model_class": "RandomForestRegressor",
            "runtime_sha256": "3" * 64,
            "original_artifact": {"sha256": "4" * 64},
            "features_in_pipeline_order": PRICE_FEATURES,
            "dataset_provenance": (
                "synthetic; not real-world evidence"
            ),
        },
        {
            "name": "tapping_advisory",
            "model_class": "GradientBoostingClassifier",
            "runtime_sha256": "5" * 64,
            "original_artifact": {"sha256": "6" * 64},
            "features_in_pipeline_order": TAPPING_FEATURES,
            "dataset_provenance": (
                "synthetic; not real-world evidence"
            ),
        },
    ],
}


class RecordingRegressionPipeline:
    """
    Fake regression pipeline that records the feature frame
    it receives and returns a fixed prediction.
    """

    def __init__(self, prediction):
        self.prediction = prediction
        self.frames = []

    def predict(self, frame):
        self.frames.append(frame.copy())
        return np.asarray(self.prediction, dtype=float)


class RecordingClassificationPipeline:
    """
    Fake classification pipeline that records the feature
    frame it receives and returns a fixed encoded label and
    probability matrix.
    """

    def __init__(self, encoded_label, probabilities):
        self.encoded_label = encoded_label
        self.probabilities = probabilities
        self.frames = []

    def predict(self, frame):
        self.frames.append(frame.copy())
        return np.asarray(self.encoded_label)

    def predict_proba(self, frame):
        return np.asarray(self.probabilities, dtype=float)


class FakeLabelEncoder:
    """
    Minimal inverse_transform compatible with the fitted
    LabelEncoder: columns 0, 1, 2 map to High, Low,
    Moderate.
    """

    def inverse_transform(self, encoded):
        labels = ["High", "Low", "Moderate"]
        return np.array(
            [labels[int(index)] for index in encoded]
        )


def build_fake_models(
    drc_prediction=(32.0,),
    price_prediction=(850.0,),
    tapping_encoded=(1,),
    tapping_probabilities=((0.1, 0.7, 0.2),),
):
    """
    Build a fake models container shaped exactly like
    model_loader.Component1Models for service-level tests.
    """

    drc_pipeline = RecordingRegressionPipeline(
        drc_prediction
    )
    price_pipeline = RecordingRegressionPipeline(
        price_prediction
    )
    tapping_pipeline = RecordingClassificationPipeline(
        tapping_encoded,
        tapping_probabilities,
    )

    models = SimpleNamespace(
        manifest=STUB_MANIFEST,
        drc=drc_pipeline,
        price=price_pipeline,
        tapping=SimpleNamespace(
            pipeline=tapping_pipeline,
            label_encoder=FakeLabelEncoder(),
            classes=("High", "Low", "Moderate"),
            warning=(
                "Rule-derived proxy labels with ~30% "
                "simulated label noise (seed 340); NOT "
                "expert-verified tapping advice."
            ),
        ),
    )

    return models, drc_pipeline, price_pipeline, tapping_pipeline


@pytest.fixture
def fake_models(monkeypatch):
    """
    Patch the service's model source with recording fakes
    and return them with their pipelines.
    """

    (
        models,
        drc_pipeline,
        price_pipeline,
        tapping_pipeline,
    ) = build_fake_models()

    monkeypatch.setattr(
        weather_tapping_service,
        "get_models",
        lambda: models,
    )

    return models, drc_pipeline, price_pipeline, tapping_pipeline


# ============================================================
# VALID PREDICTION REQUESTS
# ============================================================


def test_valid_drc_request_produces_response(
    fake_models,
):
    """
    A fully populated valid DRC request must produce a
    response with the fixed target name, unit, and the stub
    model identifiers.
    """

    models, drc_pipeline, _, _ = fake_models

    response = predict_drc(
        DrcPredictionRequest(
            observation_timestamp=AWARE_TIMESTAMP,
            temperature_c=27.5,
            relative_humidity_pct=82.0,
            rainfall_last_6h_mm=4.2,
            forecast_rain_next_6h_mm=0.0,
            rainfall_last_3days_mm=18.5,
            wind_speed_kmh=7.3,
            soil_moisture_pct=41.0,
            soil_ph=5.8,
            estate_id="EST-001",
            tree_count=150.0,
            tapping_system="S/2",
            days_since_last_tap=2.0,
            previous_collection_yield_litres=38.0,
        )
    )

    assert response.target_name == "latex_drc_pct"
    assert response.unit == "percent"
    assert response.predicted_latex_drc_pct == 32.0
    assert response.model_name == "drc_prediction"
    assert response.runtime_artifact_sha256 == "1" * 64
    assert response.original_artifact_sha256 == "2" * 64
    assert response.observation_month == 1
    assert response.observation_day_of_year == 2


def test_valid_tapping_request_produces_response(
    fake_models,
):
    """
    A valid tapping-suitability request must produce a
    response carrying the fitted label, the explicit class
    order, and the proxy marker.
    """

    models, _, _, tapping_pipeline = fake_models

    response = predict_tapping_suitability(
        TappingSuitabilityRequest(
            observation_timestamp=AWARE_TIMESTAMP,
            temperature_c=27.5,
            relative_humidity_pct=82.0,
            rainfall_last_6h_mm=4.2,
            forecast_rain_next_6h_mm=0.0,
            rainfall_last_3days_mm=18.5,
            wind_speed_kmh=7.3,
            soil_moisture_pct=41.0,
            soil_ph=5.8,
            tree_count=150.0,
            tapping_system="S/2",
            days_since_last_tap=2.0,
        )
    )

    assert response.target_name == (
        "tapping_suitability_proxy"
    )
    assert response.predicted_label == "Low"
    assert response.class_order == [
        "High",
        "Low",
        "Moderate",
    ]
    assert response.is_rule_derived_proxy is True
    assert response.research_warning.startswith(
        "Rule-derived proxy"
    )


def test_valid_price_request_produces_response(
    fake_models,
):
    """
    A valid price-estimation request must produce a response
    with the fixed target name and LKR-per-kg unit.
    """

    models, _, price_pipeline, _ = fake_models

    response = predict_price_estimation(
        PriceEstimationRequest(
            observation_timestamp=AWARE_TIMESTAMP,
            latex_drc_pct=34.5,
            temperature_c=27.5,
            relative_humidity_pct=82.0,
            rainfall_last_6h_mm=4.2,
            soil_moisture_pct=41.0,
            soil_ph=5.8,
            tree_count=150.0,
            tapping_system="S/2",
        )
    )

    assert response.target_name == "price_lkr_per_kg"
    assert response.unit == "LKR per kg"
    assert response.predicted_price_lkr_per_kg == 850.0
    assert response.model_name == "price_estimation"
    assert response.runtime_artifact_sha256 == "3" * 64
    assert response.original_artifact_sha256 == "4" * 64


def test_request_with_only_timestamp_produces_response(
    fake_models,
):
    """
    A request with every optional observation omitted (None)
    must still produce a response: missing values flow to
    the pipeline's imputation path.
    """

    models, drc_pipeline, _, _ = fake_models

    response = predict_drc(
        DrcPredictionRequest(
            observation_timestamp=AWARE_TIMESTAMP,
        )
    )

    frame = drc_pipeline.frames[0]

    for column in [
        "temperature_c",
        "relative_humidity_pct",
        "rainfall_last_6h_mm",
        "forecast_rain_next_6h_mm",
        "rainfall_last_3days_mm",
        "wind_speed_kmh",
        "soil_moisture_pct",
        "soil_ph",
        "tree_count",
        "days_since_last_tap",
        "previous_collection_yield_litres",
    ]:
        assert math.isnan(frame.loc[0, column])

    assert frame.loc[0, "estate_id"] is None
    assert frame.loc[0, "tapping_system"] is None


# ============================================================
# EXACT FEATURE ORDER AND estate_name ABSENCE
# ============================================================


def test_drc_feature_frame_uses_exact_manifest_order(
    fake_models,
):
    """
    The feature frame sent to the DRC pipeline must contain
    EXACTLY the manifest-ordered feature columns - never
    estate_name and never any engineered extra column.
    """

    models, drc_pipeline, _, _ = fake_models

    predict_drc(
        DrcPredictionRequest(
            observation_timestamp=AWARE_TIMESTAMP,
            temperature_c=27.5,
        )
    )

    frame = drc_pipeline.frames[0]

    assert list(frame.columns) == DRC_FEATURES
    assert "estate_name" not in frame.columns
    assert frame.shape == (1, len(DRC_FEATURES))


def test_price_feature_frame_uses_exact_manifest_order(
    fake_models,
):
    """
    The feature frame sent to the price pipeline must
    contain EXACTLY the 10 manifest-ordered columns, with no
    estate_id and no estate_name.
    """

    models, _, price_pipeline, _ = fake_models

    predict_price_estimation(
        PriceEstimationRequest(
            observation_timestamp=AWARE_TIMESTAMP,
        )
    )

    frame = price_pipeline.frames[0]

    assert list(frame.columns) == PRICE_FEATURES
    assert "estate_id" not in frame.columns
    assert "estate_name" not in frame.columns


def test_tapping_feature_frame_uses_exact_manifest_order(
    fake_models,
):
    """
    The feature frame sent to the tapping pipeline must
    contain EXACTLY the 13 manifest-ordered columns, with no
    estate_id, estate_name or collection-yield feature.
    """

    models, _, _, tapping_pipeline = fake_models

    predict_tapping_suitability(
        TappingSuitabilityRequest(
            observation_timestamp=AWARE_TIMESTAMP,
        )
    )

    frame = tapping_pipeline.frames[0]

    assert list(frame.columns) == TAPPING_FEATURES
    assert "estate_id" not in frame.columns
    assert "estate_name" not in frame.columns
    assert (
        "previous_collection_yield_litres"
        not in frame.columns
    )


# ============================================================
# MISSING-VALUE AND NORMALIZATION POLICY
# ============================================================


def test_out_of_range_value_equals_missing_value_path(
    fake_models,
):
    """
    A sensor value outside the documented research range
    must reach the pipeline exactly like an omitted
    observation (converted to missing / NaN, then
    median-imputed inside the fitted pipeline) - never
    clamped and never substituted with zero.
    """

    models, drc_pipeline, _, _ = fake_models

    predict_drc(
        DrcPredictionRequest(
            observation_timestamp=AWARE_TIMESTAMP,
            temperature_c=-100.0,
        )
    )

    predict_drc(
        DrcPredictionRequest(
            observation_timestamp=AWARE_TIMESTAMP,
        )
    )

    out_of_range_frame = drc_pipeline.frames[0]
    omitted_frame = drc_pipeline.frames[1]

    assert math.isnan(out_of_range_frame.loc[0, "temperature_c"])
    assert math.isnan(omitted_frame.loc[0, "temperature_c"])
    assert out_of_range_frame.equals(omitted_frame)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("rainfall_last_6h_mm", -5.0),
        ("forecast_rain_next_6h_mm", -1.0),
        ("rainfall_last_3days_mm", -0.001),
        ("wind_speed_kmh", -10.0),
        ("relative_humidity_pct", 150.0),
        ("relative_humidity_pct", -0.1),
        ("soil_moisture_pct", 101.0),
        ("soil_ph", 15.0),
        ("soil_ph", -1.0),
        ("temperature_c", 61.0),
        ("temperature_c", -6.0),
    ],
)
def test_research_range_policy_applies_per_field(
    fake_models,
    field_name: str,
    bad_value: float,
):
    """
    Each field's documented clean_basic() range rule must
    convert out-of-range values to missing (NaN) before the
    row reaches the pipeline.
    """

    models, drc_pipeline, _, _ = fake_models

    predict_drc(
        DrcPredictionRequest(
            observation_timestamp=AWARE_TIMESTAMP,
            **{field_name: bad_value},
        )
    )

    frame = drc_pipeline.frames[0]

    assert math.isnan(frame.loc[0, field_name])


def test_in_range_values_pass_through_unchanged(
    fake_models,
):
    """
    In-range values must reach the pipeline unchanged - no
    rounding, scaling or rewriting in the service layer.
    """

    models, drc_pipeline, _, _ = fake_models

    predict_drc(
        DrcPredictionRequest(
            observation_timestamp=AWARE_TIMESTAMP,
            temperature_c=27.5,
            soil_ph=5.8,
            relative_humidity_pct=82.0,
        )
    )

    frame = drc_pipeline.frames[0]

    assert frame.loc[0, "temperature_c"] == 27.5
    assert frame.loc[0, "soil_ph"] == 5.8
    assert frame.loc[0, "relative_humidity_pct"] == 82.0


def test_tapping_system_is_normalized_like_training(
    fake_models,
):
    """
    tapping_system must be trimmed and upper-cased exactly
    as training's clean_basic() did; estate_id must NOT be
    normalized (training never transformed it).
    """

    models, drc_pipeline, _, _ = fake_models

    predict_drc(
        DrcPredictionRequest(
            observation_timestamp=AWARE_TIMESTAMP,
            estate_id=" est-001 ",
            tapping_system=" s/2 ",
        )
    )

    frame = drc_pipeline.frames[0]

    assert frame.loc[0, "tapping_system"] == "S/2"
    assert frame.loc[0, "estate_id"] == " est-001 "


def test_unknown_categories_are_passed_through(
    fake_models,
):
    """
    Unknown categories (unseen at fit time) must be passed
    through unchanged; the fitted one-hot encoder's
    unknown-ignore behaviour handles them inside the
    pipeline.
    """

    models, drc_pipeline, _, _ = fake_models

    predict_drc(
        DrcPredictionRequest(
            observation_timestamp=AWARE_TIMESTAMP,
            estate_id="EST-NEW-999",
            tapping_system="D5",
        )
    )

    frame = drc_pipeline.frames[0]

    assert frame.loc[0, "estate_id"] == "EST-NEW-999"
    assert frame.loc[0, "tapping_system"] == "D5"


# ============================================================
# TIMESTAMP SEMANTICS AND CALENDAR FEATURE DERIVATION
# ============================================================


def test_observation_timezone_default_is_colombo():
    """
    The configured observation timezone must remain the
    documented default Asia/Colombo - separate from the
    pure feature helpers and never silently replaced by
    UTC.
    """

    assert OBSERVATION_TIMEZONE == "Asia/Colombo"


def test_colombo_timestamp_derives_local_calendar_features(
    fake_models,
):
    """
    A 2025-01-02T08:00+05:30 observation must derive
    month=1 and day_of_year=2 from the Colombo wall clock
    (equivalent to training's naive-local derivation).
    """

    models, drc_pipeline, _, _ = fake_models

    response = predict_drc(
        DrcPredictionRequest(
            observation_timestamp=datetime(
                2025,
                1,
                2,
                8,
                0,
                tzinfo=COLOMBO_OFFSET,
            ),
        )
    )

    frame = drc_pipeline.frames[0]

    assert response.observation_month == 1
    assert response.observation_day_of_year == 2
    assert frame.loc[0, "month"] == 1.0
    assert frame.loc[0, "day_of_year"] == 2.0


def test_utc_timestamp_is_explicitly_converted_to_observation_local(
    fake_models,
):
    """
    A UTC timestamp whose Colombo wall clock falls on the
    NEXT calendar day must derive month/day_of_year from
    the observation-local calendar, NOT from UTC.
    """

    models, drc_pipeline, _, _ = fake_models

    response = predict_drc(
        DrcPredictionRequest(
            observation_timestamp=datetime(
                2025,
                1,
                1,
                19,
                30,
                tzinfo=timezone.utc,
            ),
        )
    )

    frame = drc_pipeline.frames[0]

    # 2025-01-01T19:30Z -> 2025-01-02T01:00+05:30
    assert response.observation_month == 1
    assert response.observation_day_of_year == 2
    assert frame.loc[0, "month"] == 1.0
    assert frame.loc[0, "day_of_year"] == 2.0


def test_utc_midnight_before_colombo_new_year_boundary(
    fake_models,
):
    """
    A timestamp that is New Year's Eve in UTC but already
    New Year's Day in Colombo must derive the Colombo
    calendar features (never UTC).
    """

    models, drc_pipeline, _, _ = fake_models

    response = predict_drc(
        DrcPredictionRequest(
            observation_timestamp=datetime(
                2024,
                12,
                31,
                20,
                0,
                tzinfo=timezone.utc,
            ),
        )
    )

    # 2024-12-31T20:00Z -> 2025-01-01T01:30+05:30
    assert response.observation_month == 1
    assert response.observation_day_of_year == 1


def test_leap_day_calendar_features(
    fake_models,
):
    """
    A leap-day observation must derive day_of_year=60 for
    29 February in a leap year, matching pandas dayofyear.
    """

    models, drc_pipeline, _, _ = fake_models

    response = predict_drc(
        DrcPredictionRequest(
            observation_timestamp=datetime(
                2024,
                2,
                29,
                12,
                0,
                tzinfo=COLOMBO_OFFSET,
            ),
        )
    )

    assert response.observation_month == 2
    assert response.observation_day_of_year == 60


def test_month_boundary_midnight_before_and_after(
    fake_models,
):
    """
    Observations on either side of a month boundary must
    derive different months from the observation-local wall
    clock.
    """

    models, drc_pipeline, _, _ = fake_models

    response_before = predict_drc(
        DrcPredictionRequest(
            observation_timestamp=datetime(
                2025,
                3,
                31,
                23,
                0,
                tzinfo=COLOMBO_OFFSET,
            ),
        )
    )

    response_after = predict_drc(
        DrcPredictionRequest(
            observation_timestamp=datetime(
                2025,
                4,
                1,
                0,
                30,
                tzinfo=COLOMBO_OFFSET,
            ),
        )
    )

    assert (
        response_before.observation_month,
        response_before.observation_day_of_year,
    ) == (3, 90)

    assert (
        response_after.observation_month,
        response_after.observation_day_of_year,
    ) == (4, 91)


# ============================================================
# TAPPING CLASS ORDER AND PROBABILITY MAPPING
# ============================================================


def test_probability_columns_are_mapped_to_labels_explicitly(
    monkeypatch,
):
    """
    predict_proba() column 0/1/2 must map to High / Low /
    Moderate exactly, as recorded in the trusted class
    order.
    """

    models, _, _, _ = build_fake_models(
        tapping_encoded=(0,),
        tapping_probabilities=((0.6, 0.3, 0.1),),
    )

    monkeypatch.setattr(
        weather_tapping_service,
        "get_models",
        lambda: models,
    )

    response = predict_tapping_suitability(
        TappingSuitabilityRequest(
            observation_timestamp=AWARE_TIMESTAMP,
        )
    )

    assert response.predicted_label == "High"
    assert response.probabilities.high == 0.6
    assert response.probabilities.low == 0.3
    assert response.probabilities.moderate == 0.1
    assert response.class_order == [
        "High",
        "Low",
        "Moderate",
    ]


def test_probability_mapping_low_class(
    fake_models,
):
    """
    An encoded label of 1 with probability column 1 highest
    must decode to Low with correctly mapped probabilities.
    """

    models, _, _, tapping_pipeline = fake_models

    response = predict_tapping_suitability(
        TappingSuitabilityRequest(
            observation_timestamp=AWARE_TIMESTAMP,
        )
    )

    # default fake: encoded=1, proba (0.1, 0.7, 0.2)
    assert response.predicted_label == "Low"
    assert response.probabilities.high == 0.1
    assert response.probabilities.low == 0.7
    assert response.probabilities.moderate == 0.2


def test_probability_mapping_moderate_class(
    monkeypatch,
):
    """
    An encoded label of 2 with probability column 2 highest
    must decode to Moderate with correctly mapped
    probabilities.
    """

    models, _, _, _ = build_fake_models(
        tapping_encoded=(2,),
        tapping_probabilities=((0.05, 0.15, 0.8),),
    )

    monkeypatch.setattr(
        weather_tapping_service,
        "get_models",
        lambda: models,
    )

    response = predict_tapping_suitability(
        TappingSuitabilityRequest(
            observation_timestamp=AWARE_TIMESTAMP,
        )
    )

    assert response.predicted_label == "Moderate"
    assert response.probabilities.moderate == 0.8
    assert response.class_order == [
        "High",
        "Low",
        "Moderate",
    ]
    assert (
        tuple(TAPPING_CLASSES_IN_PROBABILITY_ORDER)
        == ("High", "Low", "Moderate")
    )


# ============================================================
# BAD MODEL OUTPUTS ARE REJECTED
# ============================================================


def test_wrong_probability_shape_raises_model_error(
    monkeypatch,
):
    """
    A probability matrix with the wrong shape must raise a
    controlled model error, not be silently reshaped or
    interpreted.
    """

    (
        models,
        _,
        _,
        tapping_pipeline,
    ) = build_fake_models(
        tapping_probabilities=((0.1, 0.2, 0.3, 0.4),),
    )

    monkeypatch.setattr(
        weather_tapping_service,
        "get_models",
        lambda: models,
    )

    with pytest.raises(
        WeatherTappingModelError,
        match="probability shape",
    ):
        predict_tapping_suitability(
            TappingSuitabilityRequest(
                observation_timestamp=AWARE_TIMESTAMP,
            )
        )


def test_wrong_prediction_shape_raises_model_error(
    monkeypatch,
):
    """
    A regression pipeline returning a two-row prediction for
    a single-row request must raise a controlled model
    error.
    """

    (
        models,
        _,
        _,
        tapping_pipeline,
    ) = build_fake_models(
        drc_prediction=(30.0, 31.0),
    )

    monkeypatch.setattr(
        weather_tapping_service,
        "get_models",
        lambda: models,
    )

    with pytest.raises(
        WeatherTappingModelError,
        match="unexpected prediction shape",
    ):
        predict_drc(
            DrcPredictionRequest(
                observation_timestamp=AWARE_TIMESTAMP,
            )
        )


def test_non_finite_prediction_raises_model_error(
    monkeypatch,
):
    """
    A non-finite regression output must be rejected with a
    controlled model error - never passed through to the
    caller.
    """

    (
        models,
        _,
        _,
        tapping_pipeline,
    ) = build_fake_models(
        drc_prediction=(math.nan,),
    )

    monkeypatch.setattr(
        weather_tapping_service,
        "get_models",
        lambda: models,
    )

    with pytest.raises(
        WeatherTappingModelError,
        match="non-finite",
    ):
        predict_drc(
            DrcPredictionRequest(
                observation_timestamp=AWARE_TIMESTAMP,
            )
        )


def test_non_finite_probabilities_raise_model_error(
    monkeypatch,
):
    """
    Non-finite probability outputs must be rejected with a
    controlled model error.
    """

    (
        models,
        _,
        _,
        tapping_pipeline,
    ) = build_fake_models(
        tapping_probabilities=((0.1, math.nan, 0.9),),
    )

    monkeypatch.setattr(
        weather_tapping_service,
        "get_models",
        lambda: models,
    )

    with pytest.raises(
        WeatherTappingModelError,
        match="non-finite",
    ):
        predict_tapping_suitability(
            TappingSuitabilityRequest(
                observation_timestamp=AWARE_TIMESTAMP,
            )
        )


def test_out_of_range_probabilities_raise_model_error(
    monkeypatch,
):
    """
    Probabilities outside [0, 1] must be rejected with a
    controlled model error.
    """

    (
        models,
        _,
        _,
        tapping_pipeline,
    ) = build_fake_models(
        tapping_probabilities=((0.1, 1.5, -0.6),),
    )

    monkeypatch.setattr(
        weather_tapping_service,
        "get_models",
        lambda: models,
    )

    with pytest.raises(
        WeatherTappingModelError,
        match=r"outside \[0, 1\]",
    ):
        predict_tapping_suitability(
            TappingSuitabilityRequest(
                observation_timestamp=AWARE_TIMESTAMP,
            )
        )


def test_pipeline_value_error_is_wrapped_as_model_error(
    monkeypatch,
):
    """
    A ValueError raised during pipeline inference (for
    example the fitted pipelines' own non-finite rejection)
    must be wrapped as a controlled model error - never
    disguised as an input error.
    """

    class ExplodingPipeline:
        def predict(self, frame):
            raise ValueError("Input X contains infinity.")

    models = SimpleNamespace(
        manifest=STUB_MANIFEST,
        drc=ExplodingPipeline(),
        price=ExplodingPipeline(),
        tapping=SimpleNamespace(
            pipeline=ExplodingPipeline(),
            label_encoder=FakeLabelEncoder(),
            classes=("High", "Low", "Moderate"),
            warning="...",
        ),
    )

    monkeypatch.setattr(
        weather_tapping_service,
        "get_models",
        lambda: models,
    )

    with pytest.raises(
        WeatherTappingModelError,
        match="rejected the feature row",
    ):
        predict_drc(
            DrcPredictionRequest(
                observation_timestamp=AWARE_TIMESTAMP,
            )
        )


# ============================================================
# MODEL LOAD FAILURE HANDLING
# ============================================================


def test_model_load_failure_is_wrapped_without_paths(
    monkeypatch,
):
    """
    A ModelLoadError from the verified loader must surface
    as a controlled WeatherTappingModelError whose message
    contains no file paths, .pkl references, or secrets.
    """

    def raise_load_error():
        raise ModelLoadError(
            r"C:\secret\location\drc_pipeline.pkl "
            "integrity check failed."
        )

    monkeypatch.setattr(
        weather_tapping_service,
        "get_models",
        raise_load_error,
    )

    with pytest.raises(
        WeatherTappingModelError,
    ) as exc_info:
        predict_drc(
            DrcPredictionRequest(
                observation_timestamp=AWARE_TIMESTAMP,
            )
        )

    message = str(exc_info.value)

    assert ".pkl" not in message
    assert "\\" not in message
    assert "/" not in message
    assert "integrity check failed" not in message


def test_unexpected_loader_failure_is_wrapped_as_model_error(
    monkeypatch,
):
    """
    Any unexpected exception from the loader must be wrapped
    as a model error - never disguised as an input error.
    """

    def raise_unexpected():
        raise RuntimeError("disk offline")

    monkeypatch.setattr(
        weather_tapping_service,
        "get_models",
        raise_unexpected,
    )

    with pytest.raises(
        WeatherTappingModelError,
    ):
        predict_price_estimation(
            PriceEstimationRequest(
                observation_timestamp=AWARE_TIMESTAMP,
            )
        )


# ============================================================
# REAL FINALIZED MODELS - END-TO-END SERVICE SMOKE
# ============================================================


def test_real_drc_service_end_to_end():
    """
    The full DRC service path through the real verified
    artifacts must produce a finite prediction and the real
    trusted hashes in the response.
    """

    response = predict_drc(
        DrcPredictionRequest(
            observation_timestamp=AWARE_TIMESTAMP,
            temperature_c=27.5,
            relative_humidity_pct=82.0,
            rainfall_last_6h_mm=4.2,
            forecast_rain_next_6h_mm=0.0,
            rainfall_last_3days_mm=18.5,
            wind_speed_kmh=7.3,
            soil_moisture_pct=41.0,
            soil_ph=5.8,
            estate_id="EST-001",
            tree_count=150.0,
            tapping_system="S/2",
            days_since_last_tap=2.0,
            previous_collection_yield_litres=38.0,
        )
    )

    assert math.isfinite(response.predicted_latex_drc_pct)
    assert response.runtime_artifact_sha256 == (
        EXPECTED_DRC_PIPELINE_SHA256
    )
    assert response.original_artifact_sha256 == (
        "19475009a289d9c099a352cac94409a84cfa9a265cfac3f8cf44e5dd5bcb0290"
    )
    assert response.dataset_provenance == (
        "synthetic; not real-world evidence"
    )


def test_real_price_service_end_to_end():
    """
    The full price service path through the real verified
    artifacts must produce a finite LKR/kg prediction.
    """

    response = predict_price_estimation(
        PriceEstimationRequest(
            observation_timestamp=AWARE_TIMESTAMP,
            latex_drc_pct=34.5,
            temperature_c=27.5,
            relative_humidity_pct=82.0,
            rainfall_last_6h_mm=4.2,
            soil_moisture_pct=41.0,
            soil_ph=5.8,
            tree_count=150.0,
            tapping_system="S/2",
        )
    )

    assert math.isfinite(response.predicted_price_lkr_per_kg)
    assert response.runtime_artifact_sha256 == (
        EXPECTED_PRICE_PIPELINE_SHA256
    )


def test_real_tapping_service_end_to_end():
    """
    The full tapping service path through the real verified
    artifacts must produce a fitted label with valid
    probabilities and the explicit proxy warning.
    """

    response = predict_tapping_suitability(
        TappingSuitabilityRequest(
            observation_timestamp=AWARE_TIMESTAMP,
            temperature_c=27.5,
            relative_humidity_pct=82.0,
            rainfall_last_6h_mm=4.2,
            forecast_rain_next_6h_mm=0.0,
            rainfall_last_3days_mm=18.5,
            wind_speed_kmh=7.3,
            soil_moisture_pct=41.0,
            soil_ph=5.8,
            tree_count=150.0,
            tapping_system="S/2",
            days_since_last_tap=2.0,
        )
    )

    assert response.predicted_label in (
        "High",
        "Low",
        "Moderate",
    )
    assert response.class_order == [
        "High",
        "Low",
        "Moderate",
    ]
    assert response.is_rule_derived_proxy is True
    assert response.research_warning
    assert (
        response.probabilities.high
        + response.probabilities.low
        + response.probabilities.moderate
    ) == pytest.approx(1.0, abs=1e-6)
    assert response.runtime_artifact_sha256 == (
        EXPECTED_TAPPING_PIPELINE_SHA256
    )


def test_real_all_missing_service_end_to_end():
    """
    The real pipelines must produce finite predictions for
    a request with every optional observation omitted
    (median-imputation path).
    """

    drc_response = predict_drc(
        DrcPredictionRequest(
            observation_timestamp=AWARE_TIMESTAMP,
        )
    )

    price_response = predict_price_estimation(
        PriceEstimationRequest(
            observation_timestamp=AWARE_TIMESTAMP,
        )
    )

    assert math.isfinite(
        drc_response.predicted_latex_drc_pct
    )
    assert math.isfinite(
        price_response.predicted_price_lkr_per_kg
    )
