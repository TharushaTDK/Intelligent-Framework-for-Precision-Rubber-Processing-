from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Contracts for saving and reading Component 1 advisories.
#
# A saved advisory holds the location, the weather the models used
# and what the models returned, so Component 2 can reuse the DRC and
# location. All model values are estimates, never measurements.


TappingLabel = Literal["High", "Low", "Moderate"]

LocationSource = Literal["map", "gps", "typed"]


class AdvisoryRecordCreate(BaseModel):
    """Body for POST /advisory-records."""

    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(ge=-90.0, le=90.0, allow_inf_nan=False)
    longitude: float = Field(ge=-180.0, le=180.0, allow_inf_nan=False)
    location_source: LocationSource | None = Field(
        default=None,
        description="How the location was chosen on the dashboard.",
    )

    observation_timestamp: datetime = Field(
        description=(
            "Hour the advisory is for, exactly as returned by the "
            "weather-snapshot endpoint. Must include a timezone offset."
        ),
    )
    forecast_based: bool = Field(
        default=False,
        description="True when the weather was a forecast (future hour).",
    )

    temperature_c: float = Field(allow_inf_nan=False)
    relative_humidity_pct: float = Field(allow_inf_nan=False)
    wind_speed_kmh: float = Field(allow_inf_nan=False)
    rainfall_last_6h_mm: float = Field(allow_inf_nan=False)
    rainfall_last_3days_mm: float = Field(allow_inf_nan=False)
    forecast_rain_next_6h_mm: float = Field(allow_inf_nan=False)

    predicted_latex_drc_pct: float = Field(
        allow_inf_nan=False,
        description="DRC estimate (percent) from POST /drc-estimation.",
    )
    drc_model_name: str = Field(min_length=1, max_length=100)
    drc_runtime_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    tapping_label: TappingLabel | None = None
    tapping_probability_high: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    tapping_probability_low: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    tapping_probability_moderate: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)

    predicted_price_lkr_per_kg: float | None = Field(default=None, allow_inf_nan=False)

    @field_validator("observation_timestamp")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        # Same rule as every other Component 1 timestamp: never guess
        # the timezone.
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "observation_timestamp must include an explicit "
                "timezone offset (for example 2026-10-19T21:00:00+05:30)."
            )
        return value


class AdvisoryRecordResponse(AdvisoryRecordCreate):
    """One saved advisory, as stored."""

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: int
    created_at: datetime


class AdvisoryRecordList(BaseModel):
    """Saved advisories, newest first."""

    count: int
    records: list[AdvisoryRecordResponse]
