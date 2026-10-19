from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
import pytest

from backend.app.schemas.weather_provider import (
    WeatherSnapshotRequest,
    WeatherSnapshotResponse,
)
import backend.app.services.weather_provider as weather_provider
from backend.app.services.weather_provider import (
    WeatherProviderDataError,
    WeatherProviderError,
    WeatherTimestampRangeError,
    fetch_weather_snapshot,
)


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    # Retries pause between attempts; tests don't need to wait.
    monkeypatch.setattr(weather_provider, "_sleep", lambda seconds: None)


TZ = ZoneInfo("Asia/Colombo")

OBS = datetime(2025, 6, 16, 8, 0, tzinfo=TZ)
NOW = datetime(2025, 6, 16, 8, 0, tzinfo=TZ)

DEFAULT_OBS_NAIVE = datetime(2025, 6, 16, 8, 0)

REQUEST_LATITUDE = 6.9271
REQUEST_LONGITUDE = 79.8612


# ============================================================
# MOCK PROVIDER PAYLOAD / TRANSPORT HELPERS
# ============================================================


def make_payload(
    observation_hour: datetime = DEFAULT_OBS_NAIVE,
    *,
    precip_past: float = 1.0,
    precip_future: float = 5.0,
    temperature: float | None = 25.5,
    humidity: float = 71.0,
    wind: float = 8.5,
    timezone_echo: str = "Asia/Colombo",
    grid: tuple[float, float] | None = (6.92, 79.85),
    drop: tuple[str, ...] = (),
    null_precip_index: int | None = None,
    null_temperature: bool = False,
    truncate_precipitation: bool = False,
    remove_index: int | None = None,
    precipitation_unit: str = "mm",
) -> dict:
    """
    Build an Open-Meteo-shaped payload covering exactly
    t-71..t+6 (78 hourly buckets). Index 71 is the
    observation hour; 72..77 are the strictly-future hours.
    """

    hours = [
        observation_hour - timedelta(hours=71 - index)
        for index in range(78)
    ]

    precipitation = [
        precip_past if index <= 71 else precip_future
        for index in range(78)
    ]

    if null_precip_index is not None:
        precipitation[null_precip_index] = None

    temperatures = [temperature] * 78

    if null_temperature:
        temperatures[71] = None

    time_values = [hour.strftime("%Y-%m-%dT%H:%M") for hour in hours]
    arrays = {
        "temperature_2m": temperatures,
        "relative_humidity_2m": [humidity] * 78,
        "wind_speed_10m": [wind] * 78,
        "precipitation": precipitation,
    }

    if truncate_precipitation:
        arrays["precipitation"] = arrays["precipitation"][:-1]

    if remove_index is not None:
        time_values.pop(remove_index)
        for key in arrays:
            arrays[key].pop(remove_index)

    hourly = {"time": time_values, **arrays}

    for variable in drop:
        hourly.pop(variable, None)

    payload = {
        "timezone": timezone_echo,
        "hourly_units": {
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "wind_speed_10m": "km/h",
            "precipitation": precipitation_unit,
        },
        "hourly": hourly,
    }

    if grid is not None:
        payload["latitude"], payload["longitude"] = grid

    return payload


def make_transport(
    payload: dict | None = None,
    *,
    status: int = 200,
    content: bytes | None = None,
    error: type[Exception] | None = None,
    capture: list | None = None,
) -> httpx.MockTransport:
    """MockTransport that records requests and can serve any
    payload, raw content, HTTP status or raised transport
    error."""

    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(request)

        if error is not None:
            raise error("mock failure", request=request)

        if content is not None:
            return httpx.Response(status, content=content)

        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


def fetch_snapshot(
    transport: httpx.MockTransport | None = None,
    observation_timestamp: datetime = OBS,
    now: datetime | None = NOW,
) -> WeatherSnapshotResponse:
    return fetch_weather_snapshot(
        latitude=REQUEST_LATITUDE,
        longitude=REQUEST_LONGITUDE,
        observation_timestamp=observation_timestamp,
        transport=transport,
        now=now,
    )


# ============================================================
# SUCCESS MAPPING
# ============================================================


def test_success_maps_instant_values_and_rainfall_windows():
    """A valid payload must map instant values at t and the
    three agreed window sums exactly."""

    payload = make_payload(
        precip_past=1.0,
        precip_future=5.0,
        temperature=25.5,
        humidity=71.0,
        wind=8.5,
    )

    response = fetch_snapshot(make_transport(payload))

    assert response.provider == "Open-Meteo"
    assert response.observation_timestamp == OBS
    assert response.observation_timezone == "Asia/Colombo"
    assert response.temperature_c == 25.5
    assert response.relative_humidity_pct == 71.0
    assert response.wind_speed_kmh == 8.5

    # t-5..t: 6 buckets x 1.0 mm; t-71..t: 72 x 1.0 mm;
    # t+1..t+6: 6 x 5.0 mm
    assert response.rainfall_last_6h_mm == pytest.approx(6.0)
    assert response.rainfall_last_3days_mm == pytest.approx(72.0)
    assert response.forecast_rain_next_6h_mm == pytest.approx(30.0)


def test_data_quality_windows_and_unverified_conventions():
    """data_quality must expose the exact window bounds and
    label every rainfall convention as unverified."""

    response = fetch_snapshot(make_transport(make_payload()))

    quality = response.data_quality

    assert quality.historical_window_start == OBS - timedelta(
        hours=71
    )
    assert quality.historical_window_end == OBS
    assert quality.forecast_window_end == OBS + timedelta(hours=6)
    assert quality.missing_hourly_values == []

    conventions = quality.rainfall_window_conventions

    assert [c.feature for c in conventions] == [
        "rainfall_last_6h_mm",
        "rainfall_last_3days_mm",
        "forecast_rain_next_6h_mm",
    ]

    assert all(
        c.verification == "unverified_runtime_convention"
        for c in conventions
    )

    # the past-timestamp caveat is documented on the forecast
    # window, and no claim of training verification exists
    forecast_convention = conventions[2]

    assert "best-match" in forecast_convention.window
    assert "as-issued" in forecast_convention.window
    assert "verified against training" not in (
        forecast_convention.window
    )


def test_grid_cell_coordinates_reported_from_provider():
    """Provider grid-cell coordinates are returned as-is,
    distinguishable from the requested coordinates, with both
    keys always present in the serialized payload."""

    payload = make_payload(grid=(6.92, 79.85))

    response = fetch_snapshot(make_transport(payload))
    quality = response.data_quality

    assert quality.grid_cell_latitude == pytest.approx(6.92)
    assert quality.grid_cell_longitude == pytest.approx(79.85)
    assert quality.grid_cell_latitude != REQUEST_LATITUDE
    assert quality.grid_cell_longitude != REQUEST_LONGITUDE

    dump = quality.model_dump()

    assert dump["grid_cell_latitude"] == pytest.approx(6.92)
    assert dump["grid_cell_longitude"] == pytest.approx(79.85)


def test_grid_cell_coordinates_absent_serialized_as_null():
    """When the provider returns no grid-cell coordinates the
    fields stay None and both keys remain present in the
    serialized payload (explicit null, matching the existing
    /models and Component 3 null convention) - never
    fabricated from the request."""

    payload = make_payload(grid=None)

    response = fetch_snapshot(make_transport(payload))
    quality = response.data_quality

    assert quality.grid_cell_latitude is None
    assert quality.grid_cell_longitude is None

    dump = response.model_dump()["data_quality"]

    assert "grid_cell_latitude" in dump
    assert "grid_cell_longitude" in dump
    assert dump["grid_cell_latitude"] is None
    assert dump["grid_cell_longitude"] is None


# ============================================================
# OUTBOUND REQUEST PARAMETERS
# ============================================================


def test_request_parameters_units_variables_and_timezone():
    """The outbound Forecast API request must carry the exact
    coordinates, the four weather variables, unit overrides,
    Asia/Colombo and the derived day counts - and nothing
    else."""

    capture: list[httpx.Request] = []
    payload = make_payload()

    fetch_snapshot(make_transport(payload, capture=capture))

    assert len(capture) == 1

    request = capture[0]
    params = request.url.params

    assert request.url.path == "/v1/forecast"
    assert params["latitude"] == str(REQUEST_LATITUDE)
    assert params["longitude"] == str(REQUEST_LONGITUDE)
    assert params["hourly"] == (
        "temperature_2m,relative_humidity_2m,wind_speed_10m,"
        "precipitation"
    )
    assert params["temperature_unit"] == "celsius"
    assert params["wind_speed_unit"] == "kmh"
    assert params["precipitation_unit"] == "mm"
    assert params["timezone"] == "Asia/Colombo"

    # history_start = 2025-06-13 09:00 -> 3 days back;
    # forecast ends the same day -> 1 forecast day
    assert params["past_days"] == "3"
    assert params["forecast_days"] == "1"

    assert set(params) == {
        "latitude",
        "longitude",
        "hourly",
        "temperature_unit",
        "wind_speed_unit",
        "precipitation_unit",
        "timezone",
        "past_days",
        "forecast_days",
    }


def test_no_soil_or_estate_variables_requested_or_returned():
    """The snapshot never requests soil/estate variables and
    the response contract contains no such fields."""

    capture: list[httpx.Request] = []

    fetch_snapshot(make_transport(make_payload(), capture=capture))

    assert "soil" not in capture[0].url.params["hourly"]
    assert "estate" not in capture[0].url.params["hourly"]

    assert set(WeatherSnapshotResponse.model_fields) == {
        "observation_timestamp",
        "observation_timezone",
        "temperature_c",
        "relative_humidity_pct",
        "wind_speed_kmh",
        "rainfall_last_6h_mm",
        "rainfall_last_3days_mm",
        "forecast_rain_next_6h_mm",
        "provider",
        "data_quality",
    }


# ============================================================
# TIMEZONE CONVERSION AND HOUR ALIGNMENT
# ============================================================


def test_utc_timestamp_converted_and_floored_to_colombo_hour():
    """A UTC timestamp must convert to Asia/Colombo and floor
    to its containing local hour (18:45Z = 00:15 Colombo ->
    00:00)."""

    observation = datetime(2025, 6, 15, 18, 45, tzinfo=timezone.utc)
    now = datetime(2025, 6, 16, 1, 0, tzinfo=TZ)
    payload = make_payload(datetime(2025, 6, 16, 0, 0))

    response = fetch_snapshot(
        make_transport(payload),
        observation_timestamp=observation,
        now=now,
    )

    assert response.observation_timestamp == datetime(
        2025,
        6,
        16,
        0,
        0,
        tzinfo=TZ,
    )


def test_naive_timestamp_rejected_before_any_http_call():
    """The service must refuse naive timestamps outright."""

    with pytest.raises(ValueError):
        fetch_weather_snapshot(
            latitude=REQUEST_LATITUDE,
            longitude=REQUEST_LONGITUDE,
            observation_timestamp=datetime(2025, 6, 16, 8, 0),
            transport=make_transport(make_payload()),
            now=NOW,
        )


# ============================================================
# HISTORICAL / FORECAST WINDOW SEPARATION
# ============================================================


def test_historical_and_forecast_buckets_are_separated():
    """Future precipitation must never leak into the
    historical sums, and historical precipitation never into
    the forecast sum."""

    past = fetch_snapshot(
        make_transport(
            make_payload(precip_past=1.0, precip_future=5.0)
        )
    )
    future_changed = fetch_snapshot(
        make_transport(
            make_payload(precip_past=1.0, precip_future=99.0)
        )
    )

    assert past.rainfall_last_6h_mm == pytest.approx(6.0)
    assert past.rainfall_last_3days_mm == pytest.approx(72.0)
    assert past.forecast_rain_next_6h_mm == pytest.approx(30.0)

    # changing only future buckets leaves history untouched
    assert future_changed.rainfall_last_6h_mm == pytest.approx(6.0)
    assert future_changed.rainfall_last_3days_mm == pytest.approx(
        72.0
    )
    assert future_changed.forecast_rain_next_6h_mm == pytest.approx(
        594.0
    )


# ============================================================
# PROVIDER RESPONSE REJECTION
# ============================================================


def test_missing_hourly_variable_rejected():
    payload = make_payload(drop=("precipitation",))

    with pytest.raises(WeatherProviderDataError):
        fetch_snapshot(make_transport(payload))


def test_null_precipitation_bucket_in_window_rejected():
    payload = make_payload(null_precip_index=10)

    with pytest.raises(WeatherProviderDataError):
        fetch_snapshot(make_transport(payload))


def test_negative_precipitation_rejected():
    payload = make_payload(precip_past=-0.5)

    with pytest.raises(WeatherProviderDataError):
        fetch_snapshot(make_transport(payload))


def test_null_temperature_at_observation_hour_rejected():
    payload = make_payload(null_temperature=True)

    with pytest.raises(WeatherProviderDataError):
        fetch_snapshot(make_transport(payload))


def test_array_length_mismatch_rejected():
    payload = make_payload(truncate_precipitation=True)

    with pytest.raises(WeatherProviderDataError):
        fetch_snapshot(make_transport(payload))


def test_missing_hour_bucket_rejected():
    """Consistent arrays with a whole hour removed must still
    be rejected because a required bucket is absent."""

    payload = make_payload(remove_index=10)

    with pytest.raises(WeatherProviderDataError):
        fetch_snapshot(make_transport(payload))


def test_wrong_units_rejected():
    payload = make_payload(precipitation_unit="inch")

    with pytest.raises(WeatherProviderDataError):
        fetch_snapshot(make_transport(payload))


def test_wrong_timezone_echo_rejected():
    payload = make_payload(timezone_echo="UTC")

    with pytest.raises(WeatherProviderDataError):
        fetch_snapshot(make_transport(payload))


def test_invalid_json_rejected():
    transport = make_transport(content=b"<html>not json</html>")

    with pytest.raises(WeatherProviderDataError):
        fetch_snapshot(transport)


def test_provider_http_error_rejected_without_leaking_details():
    transport = make_transport(
        {"reason": "latitude is out of bounds"},
        status=400,
    )

    with pytest.raises(WeatherProviderDataError) as exc_info:
        fetch_snapshot(transport)

    message = str(exc_info.value)

    assert "400" in message
    assert "reason" not in message
    assert "api.open-meteo.com" not in message


# ============================================================
# TRANSPORT FAILURES
# ============================================================


def test_timeout_raises_provider_error():
    transport = make_transport(error=httpx.ReadTimeout)

    with pytest.raises(WeatherProviderError) as exc_info:
        fetch_snapshot(transport)

    assert not isinstance(exc_info.value, WeatherProviderDataError)
    assert "timed out" in str(exc_info.value)
    assert "api.open-meteo.com" not in str(exc_info.value)


def test_connection_error_raises_provider_error():
    transport = make_transport(error=httpx.ConnectError)

    with pytest.raises(WeatherProviderError) as exc_info:
        fetch_snapshot(transport)

    assert not isinstance(exc_info.value, WeatherProviderDataError)
    assert "connect" in str(exc_info.value).lower()


# ============================================================
# RETRIES FOR A BRIEFLY UNAVAILABLE PROVIDER
# ============================================================


def make_sequence_transport(steps: list, capture: list) -> httpx.MockTransport:
    """
    Serves one step per request, in order. A step is either an
    HTTP status (served with a valid payload when it is 200) or an
    exception class to raise.
    """

    remaining = list(steps)

    def handler(request: httpx.Request) -> httpx.Response:
        capture.append(request)
        step = remaining.pop(0)

        if isinstance(step, type) and issubclass(step, Exception):
            raise step("mock failure", request=request)

        if step == 200:
            return httpx.Response(200, json=make_payload())

        return httpx.Response(step, json={"reason": "busy"})

    return httpx.MockTransport(handler)


def test_brief_503_is_retried_and_then_succeeds():
    # This is what happened in the browser: one 503, then fine.
    capture = []
    response = fetch_snapshot(make_sequence_transport([503, 200], capture))

    assert len(capture) == 2
    assert response.temperature_c is not None


def test_several_overloaded_replies_in_a_row_still_recover():
    capture = []
    response = fetch_snapshot(make_sequence_transport([503, 503, 503, 200], capture))

    assert len(capture) == 4
    assert response.temperature_c is not None


def test_connection_error_is_retried_and_then_succeeds():
    capture = []
    response = fetch_snapshot(
        make_sequence_transport([httpx.ConnectError, 200], capture)
    )

    assert len(capture) == 2
    assert response.provider


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_provider_still_busy_after_retries_is_unavailable_not_bad_data(status):
    attempts = len(weather_provider.RETRY_DELAYS_SECONDS) + 1
    capture = []
    transport = make_sequence_transport([status] * attempts, capture)

    with pytest.raises(WeatherProviderError) as exc_info:
        fetch_snapshot(transport)

    # Unavailable (API 503), not "invalid data" (API 502).
    assert not isinstance(exc_info.value, WeatherProviderDataError)
    assert len(capture) == attempts
    message = str(exc_info.value)
    assert "temporarily unavailable" in message
    assert "busy" not in message
    assert "api.open-meteo.com" not in message


def test_client_error_status_is_not_retried():
    capture = []
    transport = make_sequence_transport([400], capture)

    with pytest.raises(WeatherProviderDataError):
        fetch_snapshot(transport)

    assert len(capture) == 1


def test_timeout_is_not_retried():
    capture = []
    transport = make_sequence_transport([httpx.ReadTimeout], capture)

    with pytest.raises(WeatherProviderError):
        fetch_snapshot(transport)

    assert len(capture) == 1


# ============================================================
# PRE-FLIGHT RANGE REJECTIONS (NO HTTP CALL)
# ============================================================


def test_history_beyond_supported_range_rejected_before_http():
    capture: list[httpx.Request] = []
    transport = make_transport(make_payload(), capture=capture)
    observation = NOW - timedelta(days=100)

    with pytest.raises(WeatherProviderDataError):
        fetch_snapshot(
            transport,
            observation_timestamp=observation,
        )

    assert capture == []


def test_future_observation_beyond_7_days_rejected_before_http():
    capture: list[httpx.Request] = []
    transport = make_transport(make_payload(), capture=capture)
    observation = NOW + timedelta(days=7, hours=1)

    with pytest.raises(WeatherTimestampRangeError):
        fetch_snapshot(
            transport,
            observation_timestamp=observation,
        )

    assert capture == []


def test_history_range_error_is_the_timestamp_range_type():
    with pytest.raises(WeatherTimestampRangeError):
        fetch_snapshot(
            make_transport(make_payload()),
            observation_timestamp=NOW - timedelta(days=100),
        )


def future_snapshot(hours_ahead: int) -> WeatherSnapshotResponse:
    observation = NOW + timedelta(hours=hours_ahead)
    return fetch_snapshot(
        make_transport(make_payload(observation.replace(tzinfo=None))),
        observation_timestamp=observation,
    )


def test_current_hour_is_not_flagged_as_forecast():
    quality = future_snapshot(0).data_quality

    assert quality.observation_in_future is False
    assert quality.forecast_hours_in_history_window == 0


def test_a_few_hours_ahead_counts_the_forecast_hours():
    quality = future_snapshot(2).data_quality

    assert quality.observation_in_future is True
    assert quality.forecast_hours_in_history_window == 2


@pytest.mark.parametrize("hours_ahead", [72, 7 * 24])
def test_days_ahead_is_accepted_and_fully_forecast(hours_ahead):
    response = future_snapshot(hours_ahead)

    assert response.data_quality.observation_in_future is True
    assert response.data_quality.forecast_hours_in_history_window == 72
    assert response.temperature_c is not None


# ============================================================
# REQUEST SCHEMA VALIDATION
# ============================================================


def test_request_schema_rejects_naive_timestamp():
    with pytest.raises(ValueError):
        WeatherSnapshotRequest(
            latitude=6.9,
            longitude=79.8,
            observation_timestamp=datetime(2025, 6, 16, 8, 0),
        )


def test_request_schema_rejects_out_of_range_coordinates():
    with pytest.raises(ValueError):
        WeatherSnapshotRequest(
            latitude=100.0,
            longitude=79.8,
            observation_timestamp=OBS,
        )

    with pytest.raises(ValueError):
        WeatherSnapshotRequest(
            latitude=6.9,
            longitude=200.0,
            observation_timestamp=OBS,
        )


def test_request_schema_rejects_extra_fields():
    with pytest.raises(ValueError):
        WeatherSnapshotRequest(
            latitude=6.9,
            longitude=79.8,
            observation_timestamp=OBS,
            estate_name="Rubber Estate A",
        )
