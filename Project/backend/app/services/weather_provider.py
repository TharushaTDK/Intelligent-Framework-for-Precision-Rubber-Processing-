from __future__ import annotations

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

from backend.app.core.config import settings

from backend.app.schemas.weather_provider import (
    RainfallWindowConvention,
    WeatherDataQuality,
    WeatherSnapshotResponse,
)

from backend.app.services.weather_tapping_service import (
    OBSERVATION_TIMEZONE,
)


# The four hourly variables the snapshot needs - nothing else
# (no soil, estate or unrelated variables are requested).
HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
)

EXPECTED_HOURLY_UNITS = {
    "temperature_2m": "°C",
    "relative_humidity_2m": "%",
    "wind_speed_10m": "km/h",
    "precipitation": "mm",
}

# Required coverage: t-71..t (72 historical buckets) plus
# t+1..t+6 (6 strictly-future buckets).
HISTORICAL_HOURS = 72
FORECAST_HOURS = 6

# Open-Meteo Forecast API hard limits (past_days/forecast_days).
MAX_PAST_DAYS = 92
MAX_FORECAST_DAYS = 16

# How far into the future an observation timestamp may sit.
FUTURE_OBSERVATION_GRACE = timedelta(hours=FORECAST_HOURS)

# Runtime window arithmetic agreed for this component. These
# are unverified_runtime_convention: the research code has no
# window definitions to compare against.
WINDOW_CONVENTIONS = (
    RainfallWindowConvention(
        feature="rainfall_last_6h_mm",
        window=(
            "sum of hourly precipitation buckets t-5 through "
            "t (inclusive, 6 buckets, Asia/Colombo "
            "containing-hour aligned)"
        ),
        verification="unverified_runtime_convention",
    ),
    RainfallWindowConvention(
        feature="rainfall_last_3days_mm",
        window=(
            "sum of hourly precipitation buckets t-71 through "
            "t (inclusive, 72 buckets, Asia/Colombo "
            "containing-hour aligned)"
        ),
        verification="unverified_runtime_convention",
    ),
    RainfallWindowConvention(
        feature="forecast_rain_next_6h_mm",
        window=(
            "sum of hourly precipitation buckets t+1 through "
            "t+6 (strictly after the observation hour); for "
            "past observation timestamps this is the "
            "provider's current best-match series, not an "
            "as-issued forecast"
        ),
        verification="unverified_runtime_convention",
    ),
)


class WeatherProviderError(RuntimeError):
    """
    Transport-level failure (timeout, connection) talking to
    the weather provider. Messages carry no URLs, parameters
    or provider payloads.
    """


class WeatherProviderDataError(WeatherProviderError):
    """
    The provider response is unusable: unexpected status,
    invalid JSON, wrong units/timezone, missing hourly values
    or an unsupported timestamp range.
    """


def _to_observation_timezone(
    timestamp: datetime,
) -> datetime:
    """
    Convert an aware timestamp to Asia/Colombo. Naive input
    is rejected - the backend never assumes a timezone.
    """

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(
            "observation_timestamp must be timezone-aware"
        )

    return timestamp.astimezone(ZoneInfo(OBSERVATION_TIMEZONE))


def _aligned_hour(timestamp: datetime) -> datetime:
    """Floor an aware timestamp to its containing local hour."""

    return timestamp.replace(
        minute=0,
        second=0,
        microsecond=0,
    )


def _coverage(
    observation_timestamp: datetime,
    now: datetime,
) -> tuple[datetime, datetime, datetime, int, int]:
    """
    Derive the hour-aligned observation time, required window
    bounds and the past_days/forecast_days request parameters.

    Raises WeatherProviderDataError when the timestamp is
    outside the Forecast API's supported envelope, before any
    HTTP request is made.
    """

    if observation_timestamp > now + FUTURE_OBSERVATION_GRACE:
        raise WeatherProviderDataError(
            "observation_timestamp is outside the supported "
            "provider range"
        )

    observation_hour = _aligned_hour(observation_timestamp)
    history_start = observation_hour - timedelta(
        hours=HISTORICAL_HOURS - 1,
    )
    forecast_end = observation_hour + timedelta(
        hours=FORECAST_HOURS,
    )

    today = now.date()
    past_days = (today - history_start.date()).days

    if past_days > MAX_PAST_DAYS:
        raise WeatherProviderDataError(
            "observation_timestamp is outside the supported "
            "provider range"
        )

    past_days = max(past_days, 0)
    forecast_days = (
        (forecast_end.date() - today).days + 1
    )

    if forecast_days > MAX_FORECAST_DAYS:
        raise WeatherProviderDataError(
            "observation_timestamp is outside the supported "
            "provider range"
        )

    return (
        observation_hour,
        history_start,
        forecast_end,
        max(forecast_days, 1),
        past_days,
    )


def _request_forecast(
    latitude: float,
    longitude: float,
    past_days: int,
    forecast_days: int,
    transport: httpx.BaseTransport | None,
) -> dict:
    """
    Call the configured Forecast API and return the decoded
    JSON payload. Transport failures raise
    WeatherProviderError; status/JSON problems raise
    WeatherProviderDataError.
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(HOURLY_VARIABLES),
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "timezone": OBSERVATION_TIMEZONE,
        "past_days": past_days,
        "forecast_days": forecast_days,
    }

    timeout = httpx.Timeout(
        connect=settings.weather_provider_connect_timeout_seconds,
        read=settings.weather_provider_read_timeout_seconds,
        write=settings.weather_provider_read_timeout_seconds,
        pool=settings.weather_provider_connect_timeout_seconds,
    )

    try:
        with httpx.Client(
            timeout=timeout,
            transport=transport,
        ) as client:
            response = client.get(
                settings.open_meteo_base_url,
                params=params,
            )
    except httpx.TimeoutException as exc:
        raise WeatherProviderError(
            "Weather provider request timed out."
        ) from exc
    except httpx.TransportError as exc:
        raise WeatherProviderError(
            "Could not connect to the weather provider."
        ) from exc

    if response.status_code != 200:
        raise WeatherProviderDataError(
            f"weather provider returned HTTP "
            f"{response.status_code}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise WeatherProviderDataError(
            "weather provider returned invalid JSON"
        ) from exc

    if not isinstance(payload, dict):
        raise WeatherProviderDataError(
            "weather provider returned an unexpected payload"
        )

    return payload


def _validated_hourly(
    payload: dict,
) -> dict[datetime, dict[str, float]]:
    """
    Validate payload structure, units and timezone, then map
    every hourly bucket to an Asia/Colombo-aware hour key.

    Raises WeatherProviderDataError on any structural
    problem; values are checked later only for the required
    window hours.
    """

    if payload.get("timezone") != OBSERVATION_TIMEZONE:
        raise WeatherProviderDataError(
            "weather provider returned the wrong timezone"
        )

    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise WeatherProviderDataError(
            "weather provider response has no hourly section"
        )

    units = payload.get("hourly_units")
    if not isinstance(units, dict) or any(
        units.get(variable) != expected
        for variable, expected in EXPECTED_HOURLY_UNITS.items()
    ):
        raise WeatherProviderDataError(
            "weather provider returned unexpected units"
        )

    times = hourly.get("time")
    if not isinstance(times, list) or not times:
        raise WeatherProviderDataError(
            "weather provider response has no hourly timestamps"
        )

    series: dict[str, list] = {}

    for variable in HOURLY_VARIABLES:
        values = hourly.get(variable)
        if not isinstance(values, list) or len(values) != len(
            times
        ):
            raise WeatherProviderDataError(
                f"weather provider returned an inconsistent "
                f"{variable} array"
            )
        series[variable] = values

    parsed: dict[datetime, dict[str, float]] = {}
    timezone = ZoneInfo(OBSERVATION_TIMEZONE)

    for index, raw_time in enumerate(times):
        try:
            hour = datetime.fromisoformat(raw_time)
        except (TypeError, ValueError) as exc:
            raise WeatherProviderDataError(
                "weather provider returned an unparsable "
                "hourly timestamp"
            ) from exc

        if hour.tzinfo is None:
            hour = hour.replace(tzinfo=timezone)
        else:
            hour = hour.astimezone(timezone)

        parsed[hour] = {
            variable: series[variable][index]
            for variable in HOURLY_VARIABLES
        }

    return parsed


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _required_values(
    parsed: dict[datetime, dict[str, float]],
    observation_hour: datetime,
) -> dict:
    """
    Extract the instant values at t and the precipitation
    totals for the three windows. Every required hour must be
    present with finite values (precipitation additionally
    non-negative) - never zero-filled, never partially summed.
    """

    history_start = observation_hour - timedelta(
        hours=HISTORICAL_HOURS - 1,
    )
    forecast_end = observation_hour + timedelta(
        hours=FORECAST_HOURS,
    )

    precipitation: dict[datetime, float] = {}
    hour = history_start

    while hour <= forecast_end:
        row = parsed.get(hour)

        if row is None:
            raise WeatherProviderDataError(
                "weather provider response is missing a "
                "required hourly bucket"
            )

        value = row["precipitation"]

        if not _is_finite_number(value) or value < 0:
            raise WeatherProviderDataError(
                "weather provider returned an invalid "
                "precipitation value in the required window"
            )

        precipitation[hour] = float(value)
        hour += timedelta(hours=1)

    instant = parsed.get(observation_hour)

    if instant is None:
        raise WeatherProviderDataError(
            "weather provider response is missing the "
            "observation hour"
        )

    for variable, name in (
        ("temperature_2m", "temperature"),
        ("relative_humidity_2m", "humidity"),
        ("wind_speed_10m", "wind speed"),
    ):
        if not _is_finite_number(instant[variable]):
            raise WeatherProviderDataError(
                f"weather provider returned an invalid "
                f"{name} value at the observation hour"
            )

    return {
        "temperature_c": float(instant["temperature_2m"]),
        "relative_humidity_pct": float(
            instant["relative_humidity_2m"]
        ),
        "wind_speed_kmh": float(instant["wind_speed_10m"]),
        "rainfall_last_6h_mm": math.fsum(
            precipitation[
                observation_hour - timedelta(hours=offset)
            ]
            for offset in range(5, -1, -1)
        ),
        "rainfall_last_3days_mm": math.fsum(
            precipitation[
                observation_hour - timedelta(hours=offset)
            ]
            for offset in range(71, -1, -1)
        ),
        "forecast_rain_next_6h_mm": math.fsum(
            precipitation[
                observation_hour + timedelta(hours=offset)
            ]
            for offset in range(1, 7)
        ),
    }


def _grid_coordinate(
    payload: dict,
    key: str,
) -> float | None:
    """Provider-returned grid-cell coordinate, or None if
    absent or unusable - never fabricated."""

    value = payload.get(key)

    if not _is_finite_number(value):
        return None

    return float(value)


def fetch_weather_snapshot(
    latitude: float,
    longitude: float,
    observation_timestamp: datetime,
    transport: httpx.BaseTransport | None = None,
    now: datetime | None = None,
) -> WeatherSnapshotResponse:
    """
    Fetch one validated weather snapshot from the configured
    Open-Meteo Forecast API.

    `now` exists so tests can pin the clock deterministically.
    Past observation timestamps inside the supported range are
    allowed; forecast_rain_next_6h_mm for such timestamps
    reflects the provider's current best-match series, not an
    as-issued historical forecast.
    """

    observation_local = _to_observation_timezone(
        observation_timestamp,
    )

    if now is None:
        now = datetime.now(ZoneInfo(OBSERVATION_TIMEZONE))
    elif now.tzinfo is None or now.utcoffset() is None:
        now = now.replace(tzinfo=ZoneInfo(OBSERVATION_TIMEZONE))
    else:
        now = now.astimezone(ZoneInfo(OBSERVATION_TIMEZONE))

    (
        observation_hour,
        history_start,
        forecast_end,
        forecast_days,
        past_days,
    ) = _coverage(observation_local, now)

    payload = _request_forecast(
        latitude,
        longitude,
        past_days,
        forecast_days,
        transport,
    )

    parsed = _validated_hourly(payload)
    values = _required_values(parsed, observation_hour)

    return WeatherSnapshotResponse(
        observation_timestamp=observation_hour,
        observation_timezone=OBSERVATION_TIMEZONE,
        temperature_c=values["temperature_c"],
        relative_humidity_pct=values["relative_humidity_pct"],
        wind_speed_kmh=values["wind_speed_kmh"],
        rainfall_last_6h_mm=values["rainfall_last_6h_mm"],
        rainfall_last_3days_mm=values[
            "rainfall_last_3days_mm"
        ],
        forecast_rain_next_6h_mm=values[
            "forecast_rain_next_6h_mm"
        ],
        provider="Open-Meteo",
        data_quality=WeatherDataQuality(
            historical_window_start=history_start,
            historical_window_end=observation_hour,
            forecast_window_end=forecast_end,
            missing_hourly_values=[],
            rainfall_window_conventions=list(
                WINDOW_CONVENTIONS
            ),
            grid_cell_latitude=_grid_coordinate(
                payload,
                "latitude",
            ),
            grid_cell_longitude=_grid_coordinate(
                payload,
                "longitude",
            ),
        ),
    )
