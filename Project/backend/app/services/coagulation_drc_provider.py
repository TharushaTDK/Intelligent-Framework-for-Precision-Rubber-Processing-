from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from backend.app.schemas.weather_tapping import DrcPredictionRequest
from backend.app.services.advisory_record_service import (
    AdvisoryRecordStorageError,
    get_advisory_record,
)
from backend.app.services.weather_provider import WeatherProviderError, fetch_weather_snapshot
from backend.app.services.weather_tapping_service import WeatherTappingServiceError, predict_drc

# the same weather values the Component 1 dashboard sends to its DRC model
WEATHER_FIELDS = (
    "temperature_c",
    "relative_humidity_pct",
    "rainfall_last_6h_mm",
    "forecast_rain_next_6h_mm",
    "rainfall_last_3days_mm",
    "wind_speed_kmh",
)


class EstateDrcUnavailableError(RuntimeError):
    pass


class EstateNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class EstateDrc:
    estate_id: str
    drc: float
    source: str
    observation_timestamp: datetime | None = None


def location_label(latitude: float, longitude: float) -> str:
    return f"{latitude:.4f}, {longitude:.4f}"


def get_location_drc(db: Session, location_id: int) -> EstateDrc:
    # a location is a saved Component 1 advisory, its DRC estimate is used as it is
    try:
        record = get_advisory_record(db, location_id)
    except AdvisoryRecordStorageError as exc:
        raise EstateDrcUnavailableError(
            "Component 1 saved locations are currently unavailable."
        ) from exc

    if record is None:
        raise EstateNotFoundError(f"No saved Component 1 location with id {location_id}.")

    return EstateDrc(
        estate_id=location_label(record.latitude, record.longitude),
        drc=record.predicted_latex_drc_pct,
        source="component_1",
        observation_timestamp=record.observation_timestamp,
    )


def get_point_drc(
    latitude: float,
    longitude: float,
    observation_timestamp: datetime | None = None,
) -> EstateDrc:
    # map pin -> Component 1 weather snapshot -> Component 1 DRC model
    when = observation_timestamp or datetime.now(timezone.utc)
    try:
        weather = fetch_weather_snapshot(
            latitude=latitude,
            longitude=longitude,
            observation_timestamp=when,
        )
    except WeatherProviderError as exc:
        raise EstateDrcUnavailableError(
            "Weather for this location is not available right now, so the DRC could not be estimated."
        ) from exc

    try:
        result = predict_drc(
            DrcPredictionRequest(
                observation_timestamp=weather.observation_timestamp,
                **{name: getattr(weather, name) for name in WEATHER_FIELDS},
            )
        )
    except WeatherTappingServiceError as exc:
        raise EstateDrcUnavailableError(
            "The DRC could not be estimated for this location."
        ) from exc

    return EstateDrc(
        estate_id=location_label(latitude, longitude),
        drc=result.predicted_latex_drc_pct,
        source="component_1",
        observation_timestamp=weather.observation_timestamp,
    )
