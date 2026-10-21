from __future__ import annotations

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from fastapi.testclient import TestClient

import backend.app.api.routes.weather_tapping as weather_tapping_routes
from backend.app.main import app
from backend.app.ml.component_1_weather_tapping.model_loader import (
    EXPECTED_DRC_PIPELINE_SHA256,
    EXPECTED_PRICE_PIPELINE_SHA256,
    EXPECTED_TAPPING_PIPELINE_SHA256,
)

from backend.app.schemas.weather_tapping import (
    DrcPredictionResponse,
    PriceEstimationResponse,
    TappingClassProbabilities,
    TappingSuitabilityResponse,
)

from backend.app.services.weather_provider import (
    WeatherProviderDataError,
    WeatherProviderError,
    WeatherTimestampRangeError,
)

from backend.app.services.weather_tapping_service import (
    WeatherTappingInputError,
    WeatherTappingModelError,
)

from backend.tests.test_weather_provider import (
    make_payload,
    make_transport,
)

client = TestClient(app)


# ============================================================
# PAYLOAD HELPERS
# ============================================================


def valid_drc_payload() -> dict:
    return {
        "observation_timestamp": "2025-01-02T08:00:00+05:30",
        "temperature_c": 27.5,
        "relative_humidity_pct": 82.0,
        "rainfall_last_6h_mm": 4.2,
        "forecast_rain_next_6h_mm": 0.0,
        "rainfall_last_3days_mm": 18.5,
        "wind_speed_kmh": 7.3,
        "soil_moisture_pct": 41.0,
        "soil_ph": 5.8,
        "visible_latex_condition": "Normal",
        "estate_id": "Estate 01",
    }


def valid_tapping_payload() -> dict:
    return {
        "observation_timestamp": "2025-01-02T08:00:00+05:30",
        "temperature_c": 27.5,
        "relative_humidity_pct": 82.0,
        "rainfall_last_6h_mm": 4.2,
        "forecast_rain_next_6h_mm": 0.0,
        "rainfall_last_3days_mm": 18.5,
        "wind_speed_kmh": 7.3,
        "soil_moisture_pct": 41.0,
        "soil_ph": 5.8,
    }


def valid_price_payload() -> dict:
    return {
        "observation_timestamp": "2025-01-02T08:00:00+05:30",
        "latex_drc_pct": 34.5,
        "temperature_c": 27.5,
        "relative_humidity_pct": 82.0,
        "rainfall_last_6h_mm": 4.2,
        "soil_moisture_pct": 41.0,
        "soil_ph": 5.8,
    }


# ============================================================
# SUCCESSFUL RESPONSES — REAL FINALIZED MODELS
# ============================================================


def test_drc_estimation_success():
    """
    POST /drc-estimation must return 200 with a finite
    prediction and the trusted runtime artifact hash.
    """

    response = client.post(
        "/api/v1/weather-tapping/drc-estimation",
        json=valid_drc_payload(),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["target_name"] == "latex_drc_pct"
    assert body["unit"] == "percent"
    assert math.isfinite(body["predicted_latex_drc_pct"])
    assert (
        body["runtime_artifact_sha256"]
        == EXPECTED_DRC_PIPELINE_SHA256
    )
    assert body["observation_month"] == 1
    assert body["observation_day_of_year"] == 2


def test_tapping_advisory_success():
    """
    POST /tapping-advisory must return 200 with a fitted
    label, the exact class order, valid probabilities, and
    the proxy disclaimer kept visible.
    """

    response = client.post(
        "/api/v1/weather-tapping/tapping-advisory",
        json=valid_tapping_payload(),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["predicted_label"] in (
        "High",
        "Low",
        "Moderate",
    )
    assert body["class_order"] == [
        "High",
        "Low",
        "Moderate",
    ]

    probabilities = body["probabilities"]

    assert (
        probabilities["high"]
        + probabilities["low"]
        + probabilities["moderate"]
    ) == pytest.approx(1.0, abs=1e-6)

    assert body["is_rule_derived_proxy"] is True
    assert body["research_warning"]
    assert (
        body["runtime_artifact_sha256"]
        == EXPECTED_TAPPING_PIPELINE_SHA256
    )


def test_price_estimation_success():
    """
    POST /price-estimation must return 200 with a finite
    LKR-per-kg estimate and the trusted runtime hash.
    """

    response = client.post(
        "/api/v1/weather-tapping/price-estimation",
        json=valid_price_payload(),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["target_name"] == "price_lkr_per_kg"
    assert body["unit"] == "LKR per kg"
    assert math.isfinite(body["predicted_price_lkr_per_kg"])
    assert (
        body["runtime_artifact_sha256"]
        == EXPECTED_PRICE_PIPELINE_SHA256
    )


def test_all_missing_observations_still_succeed():
    """
    Requests with only an explicit timestamp must still
    return 200: missing observations flow to the pipeline's
    median-imputation path.
    """

    response = client.post(
        "/api/v1/weather-tapping/drc-estimation",
        json={"observation_timestamp": "2025-01-02T08:00:00+05:30"},
    )

    assert response.status_code == 200
    assert math.isfinite(
        response.json()["predicted_latex_drc_pct"]
    )


# ============================================================
# RAW PREDICTION PRESERVATION (MOCKED SERVICE)
# ============================================================


def test_raw_prediction_value_is_preserved(monkeypatch):
    """
    The API must pass the raw model output through unchanged
    - no clamping, rounding or rewriting.
    """

    expected = 34.567890123456

    def fake_predict_drc(request):
        return DrcPredictionResponse(
            model_name="drc_prediction",
            model_class="GradientBoostingRegressor",
            runtime_artifact_sha256="1" * 64,
            original_artifact_sha256="2" * 64,
            dataset_provenance=(
                "synthetic; not real-world evidence"
            ),
            observation_month=1,
            observation_day_of_year=2,
            predicted_latex_drc_pct=expected,
        )

    monkeypatch.setattr(
        weather_tapping_routes,
        "predict_drc",
        fake_predict_drc,
    )

    response = client.post(
        "/api/v1/weather-tapping/drc-estimation",
        json=valid_drc_payload(),
    )

    assert response.status_code == 200
    assert (
        response.json()["predicted_latex_drc_pct"]
        == expected
    )


def test_tapping_probabilities_and_class_order_preserved(
    monkeypatch,
):
    """
    The API must preserve the probability-to-class mapping
    and the exact fitted class order.
    """

    def fake_predict_tapping(request):
        return TappingSuitabilityResponse(
            model_name="tapping_advisory",
            model_class="GradientBoostingClassifier",
            runtime_artifact_sha256="1" * 64,
            original_artifact_sha256="2" * 64,
            dataset_provenance=(
                "synthetic; not real-world evidence"
            ),
            observation_month=1,
            observation_day_of_year=2,
            predicted_label="High",
            class_order=["High", "Low", "Moderate"],
            probabilities=TappingClassProbabilities(
                high=0.6,
                low=0.3,
                moderate=0.1,
            ),
            is_rule_derived_proxy=True,
            research_warning=(
                "Rule-derived proxy labels; NOT "
                "expert-verified tapping advice."
            ),
        )

    monkeypatch.setattr(
        weather_tapping_routes,
        "predict_tapping_suitability",
        fake_predict_tapping,
    )

    response = client.post(
        "/api/v1/weather-tapping/tapping-advisory",
        json=valid_tapping_payload(),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["predicted_label"] == "High"
    assert body["class_order"] == [
        "High",
        "Low",
        "Moderate",
    ]
    assert body["probabilities"] == {
        "high": 0.6,
        "low": 0.3,
        "moderate": 0.1,
    }
    assert "expert-verified" in body["research_warning"]


def test_price_raw_value_is_preserved(monkeypatch):
    """
    The price endpoint must pass the raw LKR estimate
    through unchanged.
    """

    expected = 849.987654321

    def fake_predict_price(request):
        return PriceEstimationResponse(
            model_name="price_estimation",
            model_class="RandomForestRegressor",
            runtime_artifact_sha256="1" * 64,
            original_artifact_sha256="2" * 64,
            dataset_provenance=(
                "synthetic; not real-world evidence"
            ),
            observation_month=1,
            observation_day_of_year=2,
            predicted_price_lkr_per_kg=expected,
        )

    monkeypatch.setattr(
        weather_tapping_routes,
        "predict_price_estimation",
        fake_predict_price,
    )

    response = client.post(
        "/api/v1/weather-tapping/price-estimation",
        json=valid_price_payload(),
    )

    assert response.status_code == 200
    assert (
        response.json()["predicted_price_lkr_per_kg"]
        == expected
    )


# ============================================================
# VALIDATION ERRORS
# ============================================================


def test_naive_timestamp_returns_422():
    """
    A timestamp without an explicit offset must be rejected
    with 422 before any model runs.
    """

    payload = valid_drc_payload()
    payload["observation_timestamp"] = "2025-01-02T08:00:00"

    response = client.post(
        "/api/v1/weather-tapping/drc-estimation",
        json=payload,
    )

    assert response.status_code == 422


def test_estate_name_returns_422():
    """
    estate_name is not a model feature; supplying it must
    be rejected with 422.
    """

    payload = valid_drc_payload()
    payload["estate_name"] = "Rubber Estate A"

    response = client.post(
        "/api/v1/weather-tapping/drc-estimation",
        json=payload,
    )

    assert response.status_code == 422


def test_non_finite_value_returns_422():
    """
    Explicit infinity in a numeric field must be rejected
    with 422.
    """

    payload = valid_tapping_payload()
    payload["temperature_c"] = "Infinity"

    response = client.post(
        "/api/v1/weather-tapping/tapping-advisory",
        json=payload,
    )

    assert response.status_code == 422


def test_missing_observation_timestamp_returns_422():
    """
    A payload without the required timestamp must be
    rejected with 422.
    """

    response = client.post(
        "/api/v1/weather-tapping/price-estimation",
        json={"latex_drc_pct": 34.5},
    )

    assert response.status_code == 422


# ============================================================
# SERVICE ERRORS
# ============================================================


def test_service_input_error_returns_422(monkeypatch):
    """
    A service-layer input error must map to 422, not 500.
    """

    def fail_input(request):
        raise WeatherTappingInputError(
            "observation_timestamp must be timezone-aware."
        )

    monkeypatch.setattr(
        weather_tapping_routes,
        "predict_drc",
        fail_input,
    )

    response = client.post(
        "/api/v1/weather-tapping/drc-estimation",
        json=valid_drc_payload(),
    )

    assert response.status_code == 422
    assert (
        "timezone-aware"
        in response.json()["detail"]
    )


def test_model_error_returns_500_not_422(monkeypatch):
    """
    Model-loading and inference failures must map to 500 -
    never disguised as client-side 422 validation problems.
    """

    def fail_model(request):
        raise WeatherTappingModelError(
            "The Component 1 prediction models could not be "
            "loaded or failed integrity verification."
        )

    monkeypatch.setattr(
        weather_tapping_routes,
        "predict_price_estimation",
        fail_model,
    )

    response = client.post(
        "/api/v1/weather-tapping/price-estimation",
        json=valid_price_payload(),
    )

    assert response.status_code == 500
    assert "integrity verification" in (
        response.json()["detail"]
    )


def test_unexpected_internal_error_returns_500_generic(
    monkeypatch,
):
    """
    An unexpected non-service exception must produce a
    generic 500 without leaking internal details.
    """

    def explode(request):
        raise RuntimeError(
            r"secret internal path C:\models\secret.pkl"
        )

    monkeypatch.setattr(
        weather_tapping_routes,
        "predict_tapping_suitability",
        explode,
    )

    response = client.post(
        "/api/v1/weather-tapping/tapping-advisory",
        json=valid_tapping_payload(),
    )

    assert response.status_code == 500
    assert (
        response.json()["detail"]
        == "An unexpected internal error occurred."
    )
    assert "secret" not in response.text
    assert ".pkl" not in response.text


# ============================================================
# MODELS ENDPOINT
# ============================================================


def test_models_endpoint_returns_verified_metadata():
    """
    GET /models must return verified metadata for all three
    models, with no local file paths or internal details.
    """

    response = client.get(
        "/api/v1/weather-tapping/models",
    )

    assert response.status_code == 200

    body = response.json()

    assert body["component"] == (
        "component_1_weather_tapping"
    )
    assert body["observation_timezone"] == "Asia/Colombo"

    model_names = [m["name"] for m in body["models"]]

    assert model_names == [
        "drc_prediction",
        "price_estimation",
        "tapping_advisory",
    ]

    for model in body["models"]:
        assert len(model["runtime_artifact_sha256"]) == 64
        assert len(model["original_artifact_sha256"]) == 64
        assert model["feature_count"] == len(
            model["features_in_pipeline_order"]
        )
        assert model["dataset_provenance"]
        assert model["target_warning"]

    drc = body["models"][0]
    price = body["models"][1]
    tapping = body["models"][2]

    assert drc["feature_count"] == 12
    assert "visible_latex_condition" in drc[
        "features_in_pipeline_order"
    ]
    assert price["feature_count"] == 8
    assert "estate_id" not in price[
        "features_in_pipeline_order"
    ]
    assert tapping["feature_count"] == 10
    assert tapping["classes_in_probability_order"] == [
        "High",
        "Low",
        "Moderate",
    ]
    assert tapping["target_warning"]

    assert len(body["known_limitations"]) > 0

    # no internal paths or artifacts leak
    assert ".pkl" not in response.text
    assert "\\" not in response.text
    assert "C:" not in response.text


def test_models_endpoint_no_path_leak_on_load_failure(
    monkeypatch,
):
    """
    A manifest-load failure must produce a generic 500
    without leaking file paths from the loader.
    """

    def fail_manifest():
        raise RuntimeError(
            "C:\\backend\\artifacts\\manifest.json not found"
        )

    monkeypatch.setattr(
        weather_tapping_routes,
        "get_manifest",
        fail_manifest,
    )

    response = client.get(
        "/api/v1/weather-tapping/models",
    )

    assert response.status_code == 500
    assert "artifacts" not in response.text
    assert "\\" not in response.json()["detail"]


# ============================================================
# WEATHER SNAPSHOT ENDPOINT
# ============================================================


def valid_snapshot_payload() -> dict:
    return {
        "latitude": 6.9271,
        "longitude": 79.8612,
        "observation_timestamp": "2025-06-16T08:00:00+05:30",
    }


def post_snapshot_with_mock(
    monkeypatch,
    payload: dict,
    observation: datetime,
) -> object:
    """POST the snapshot endpoint with the real provider code
    path served by a mock transport (no network)."""

    transport = make_transport(payload)

    real_fetch = weather_tapping_routes.fetch_weather_snapshot

    def fetch_with_mock_transport(
        latitude, longitude, observation_timestamp
    ):
        return real_fetch(
            latitude=latitude,
            longitude=longitude,
            observation_timestamp=observation_timestamp,
            transport=transport,
        )

    monkeypatch.setattr(
        weather_tapping_routes,
        "fetch_weather_snapshot",
        fetch_with_mock_transport,
    )

    return client.post(
        "/api/v1/weather-tapping/weather-snapshot",
        json={
            "latitude": 6.9271,
            "longitude": 79.8612,
            "observation_timestamp": observation.isoformat(),
        },
    )


def recent_observation_hour() -> datetime:
    """An on-the-hour timestamp 2h in the past, relative to
    the real clock, so derived day counts stay in range."""

    tz = ZoneInfo("Asia/Colombo")

    return (
        datetime.now(tz) - timedelta(hours=2)
    ).replace(minute=0, second=0, microsecond=0)


def test_weather_snapshot_success(monkeypatch):
    """The real provider code path, served by a mock
    transport (no network), must return 200 with the mapped
    values and the unverified window conventions."""

    observation = recent_observation_hour()

    payload = make_payload(
        observation.replace(tzinfo=None),
        precip_past=2.0,
        precip_future=0.5,
    )

    response = post_snapshot_with_mock(
        monkeypatch,
        payload,
        observation,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["provider"] == "Open-Meteo"
    assert body["observation_timezone"] == "Asia/Colombo"
    assert body["rainfall_last_6h_mm"] == pytest.approx(12.0)
    assert body["rainfall_last_3days_mm"] == pytest.approx(144.0)
    assert body["forecast_rain_next_6h_mm"] == pytest.approx(3.0)

    conventions = body["data_quality"][
        "rainfall_window_conventions"
    ]

    assert len(conventions) == 3
    assert all(
        entry["verification"] == "unverified_runtime_convention"
        for entry in conventions
    )

    quality = body["data_quality"]

    assert quality["grid_cell_latitude"] == pytest.approx(6.92)
    assert quality["grid_cell_longitude"] == pytest.approx(79.85)
    assert quality["grid_cell_latitude"] != 6.9271
    assert quality["grid_cell_longitude"] != 79.8612


def test_weather_snapshot_absent_grid_coordinates_serialized_as_null(
    monkeypatch,
):
    """When the provider payload carries no grid-cell
    coordinates, the JSON body keeps both keys with explicit
    null values - present, distinguishable from the requested
    coordinates, and never fabricated."""

    observation = recent_observation_hour()

    payload = make_payload(
        observation.replace(tzinfo=None),
        grid=None,
    )

    response = post_snapshot_with_mock(
        monkeypatch,
        payload,
        observation,
    )

    assert response.status_code == 200

    quality = response.json()["data_quality"]

    assert "grid_cell_latitude" in quality
    assert "grid_cell_longitude" in quality
    assert quality["grid_cell_latitude"] is None
    assert quality["grid_cell_longitude"] is None
    assert quality["grid_cell_latitude"] != 6.9271
    assert quality["grid_cell_longitude"] != 79.8612


def test_weather_snapshot_provider_error_returns_503(
    monkeypatch,
):
    """Transport failures map to 503 with the fixed public
    detail - never the exception text, URL or parameters."""

    def fail_provider(latitude, longitude, observation_timestamp):
        raise WeatherProviderError(
            "Weather provider request timed out. "
            "https://api.open-meteo.com/v1/forecast?past_days=100"
        )

    monkeypatch.setattr(
        weather_tapping_routes,
        "fetch_weather_snapshot",
        fail_provider,
    )

    response = client.post(
        "/api/v1/weather-tapping/weather-snapshot",
        json=valid_snapshot_payload(),
    )

    assert response.status_code == 503
    assert (
        response.json()["detail"]
        == "Weather data is currently unavailable."
    )

    lowered = response.text.lower()

    assert "open-meteo" not in lowered
    assert "past_days" not in lowered
    assert "httpx" not in lowered
    assert "timed out" not in lowered


def test_weather_snapshot_data_error_returns_502(monkeypatch):
    """Provider data problems map to 502 with the fixed
    public detail - provider reasons stay server-side."""

    def fail_provider(latitude, longitude, observation_timestamp):
        raise WeatherProviderDataError(
            "weather provider returned HTTP 400 "
            "(reason: invalid hourly)"
        )

    monkeypatch.setattr(
        weather_tapping_routes,
        "fetch_weather_snapshot",
        fail_provider,
    )

    response = client.post(
        "/api/v1/weather-tapping/weather-snapshot",
        json=valid_snapshot_payload(),
    )

    assert response.status_code == 502
    assert (
        response.json()["detail"]
        == "Weather provider returned incomplete or "
        "invalid data."
    )
    assert "reason" not in response.text
    assert "400" not in response.text


def test_weather_snapshot_unsupported_time_returns_clear_422(monkeypatch):
    """A time more than 7 days ahead (or too far back) is the
    caller's input, so it's a 422 with a plain explanation - not
    the 502 'provider returned invalid data'."""

    def fail_provider(latitude, longitude, observation_timestamp):
        raise WeatherTimestampRangeError(
            "observation_timestamp is outside the supported provider range"
        )

    monkeypatch.setattr(
        weather_tapping_routes,
        "fetch_weather_snapshot",
        fail_provider,
    )

    response = client.post(
        "/api/v1/weather-tapping/weather-snapshot",
        json=valid_snapshot_payload(),
    )

    assert response.status_code == 422
    assert "7 days in the future" in response.json()["detail"]
    assert "provider" not in response.json()["detail"]


def test_weather_snapshot_validation_returns_422():
    """Naive timestamps, out-of-range coordinates and extra
    fields are rejected with 422 before any provider call."""

    naive = valid_snapshot_payload()
    naive["observation_timestamp"] = "2025-06-16T08:00:00"

    assert (
        client.post(
            "/api/v1/weather-tapping/weather-snapshot",
            json=naive,
        ).status_code
        == 422
    )

    out_of_range = valid_snapshot_payload()
    out_of_range["latitude"] = 100.0

    assert (
        client.post(
            "/api/v1/weather-tapping/weather-snapshot",
            json=out_of_range,
        ).status_code
        == 422
    )

    extra_field = valid_snapshot_payload()
    extra_field["estate_name"] = "Rubber Estate A"

    assert (
        client.post(
            "/api/v1/weather-tapping/weather-snapshot",
            json=extra_field,
        ).status_code
        == 422
    )


def test_weather_snapshot_unexpected_error_returns_500_generic(
    monkeypatch,
):
    """Unexpected failures produce the generic 500 without
    leaking internal details."""

    def explode(latitude, longitude, observation_timestamp):
        raise RuntimeError(
            r"secret internal path C:\cache\weather "
            "open-meteo.com?past_days=92"
        )

    monkeypatch.setattr(
        weather_tapping_routes,
        "fetch_weather_snapshot",
        explode,
    )

    response = client.post(
        "/api/v1/weather-tapping/weather-snapshot",
        json=valid_snapshot_payload(),
    )

    assert response.status_code == 500
    assert (
        response.json()["detail"]
        == "An unexpected internal error occurred."
    )
    assert "secret" not in response.text
    assert "past_days" not in response.text
    assert ".pkl" not in response.text


def test_prediction_endpoints_reject_snapshot_fields():
    """The snapshot endpoint stays standalone: prediction
    endpoints must keep rejecting its latitude/longitude
    fields."""

    payload = valid_drc_payload()
    payload["latitude"] = 6.9271
    payload["longitude"] = 79.8612

    assert (
        client.post(
            "/api/v1/weather-tapping/drc-estimation",
            json=payload,
        ).status_code
        == 422
    )


# ============================================================
# COMPONENT 3 ROUTES REMAIN REGISTERED
# ============================================================


def test_component3_routes_remain_registered():
    """
    Registering the Component 1 router must not remove or
    break any Component 3 route or the health endpoint.
    """

    paths = set()

    for route in app.routes:
        if hasattr(route, "path"):
            paths.add(route.path)

        # included routers keep the real APIRoutes on
        # original_router
        inner = getattr(route, "original_router", None)

        if inner is not None:
            for sub in inner.routes:
                if hasattr(sub, "path"):
                    paths.add(sub.path)

    assert "/health" in paths
    assert "/api/v1/smokehouse/batches" in paths
    assert "/api/v1/smokehouse/readings" in paths
    assert "/api/v1/smokehouse/predict-preview" in paths

    # all three Component 1 prediction endpoints + catalog
    assert (
        "/api/v1/weather-tapping/drc-estimation" in paths
    )
    assert (
        "/api/v1/weather-tapping/tapping-advisory"
        in paths
    )
    assert (
        "/api/v1/weather-tapping/price-estimation"
        in paths
    )
    assert "/api/v1/weather-tapping/models" in paths
    assert (
        "/api/v1/weather-tapping/weather-snapshot" in paths
    )


def test_health_endpoint_still_works():
    """
    The shared health endpoint must remain unchanged.
    """

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
