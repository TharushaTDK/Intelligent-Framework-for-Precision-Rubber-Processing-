from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


# Standalone weather-snapshot contracts. The snapshot endpoint
# fetches observed/forecast hourly weather from the configured
# provider; it never runs a model and is never chained into the
# DRC, tapping or price endpoints.
#
# The three rainfall totals below follow rolling-hour windows
# whose exact definition was NOT recoverable from the research
# code (no window or rolling feature code exists there). They
# are labelled unverified_runtime_convention everywhere they
# appear - never as verified training behaviour.


class WeatherSnapshotRequest(BaseModel):
    """Standalone request for one provider weather snapshot."""

    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(
        ge=-90.0,
        le=90.0,
        allow_inf_nan=False,
        description="WGS84 latitude of the observation point.",
    )
    longitude: float = Field(
        ge=-180.0,
        le=180.0,
        allow_inf_nan=False,
        description="WGS84 longitude of the observation point.",
    )
    observation_timestamp: datetime = Field(
        description=(
            "Observation time with an explicit UTC offset. "
            "Converted to Asia/Colombo and floored to the "
            "containing hour. Allowed from about 89 days in the "
            "past up to 7 days in the future; future times are "
            "flagged as forecast-based in data_quality."
        ),
    )

    @field_validator("observation_timestamp")
    @classmethod
    def _require_timezone_aware(
        cls,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "observation_timestamp must include an "
                "explicit timezone offset (for example "
                "2025-01-02T08:00:00+05:30 or "
                "2025-01-02T02:30:00Z). Naive timestamps "
                "are rejected: the backend never assumes a "
                "timezone silently."
            )
        return value


class RainfallWindowConvention(BaseModel):
    """
    Describes one rainfall window. verification is always
    'unverified_runtime_convention': the window reproduces the
    agreed runtime arithmetic only and is not verified against
    training preprocessing (no window definitions exist in the
    research code).
    """

    feature: str
    window: str
    verification: Literal["unverified_runtime_convention"]


class WeatherDataQuality(BaseModel):
    """Coverage and provenance metadata for one snapshot."""

    historical_window_start: datetime
    historical_window_end: datetime
    forecast_window_end: datetime
    missing_hourly_values: list[str] = Field(
        default_factory=list,
    )
    rainfall_window_conventions: list[RainfallWindowConvention]
    grid_cell_latitude: float | None = Field(
        default=None,
        description=(
            "Provider-returned grid-cell centre latitude, "
            "which may differ from the requested latitude. "
            "None when the provider did not return one - "
            "never fabricated."
        ),
    )
    grid_cell_longitude: float | None = Field(
        default=None,
        description=(
            "Provider-returned grid-cell centre longitude, "
            "which may differ from the requested longitude. "
            "None when the provider did not return one - "
            "never fabricated."
        ),
    )
    observation_in_future: bool = Field(
        default=False,
        description=(
            "True when the observation hour is after the current "
            "hour. Every weather value is then a forecast, "
            "including the rainfall_last_6h_mm / "
            "rainfall_last_3days_mm windows; the models were "
            "trained on observed weather, so predictions from it "
            "are forecast-based and not validated."
        ),
    )
    forecast_hours_in_history_window: int = Field(
        default=0,
        ge=0,
        le=72,
        description=(
            "How many of the 72 hourly buckets in the t-71..t "
            "rainfall window lie after the current hour, i.e. "
            "are forecasts rather than observations."
        ),
    )


class WeatherSnapshotResponse(BaseModel):
    """
    Hourly weather at the hour-aligned observation time.

    rainfall_last_6h_mm / rainfall_last_3days_mm reproduce
    unverified_runtime_convention windows t-5..t and t-71..t.
    forecast_rain_next_6h_mm covers t+1..t+6; for past
    observation timestamps it reflects the provider's current
    best-match series for those hours, not the forecast as it
    was originally issued at that time.
    """

    observation_timestamp: datetime
    observation_timezone: str
    temperature_c: float = Field(allow_inf_nan=False)
    relative_humidity_pct: float = Field(allow_inf_nan=False)
    wind_speed_kmh: float = Field(allow_inf_nan=False)
    rainfall_last_6h_mm: float = Field(
        allow_inf_nan=False,
        description=(
            "Sum of hourly precipitation buckets t-5..t "
            "(inclusive, 6 buckets). Unverified runtime "
            "convention."
        ),
    )
    rainfall_last_3days_mm: float = Field(
        allow_inf_nan=False,
        description=(
            "Sum of hourly precipitation buckets t-71..t "
            "(inclusive, 72 buckets). Unverified runtime "
            "convention."
        ),
    )
    forecast_rain_next_6h_mm: float = Field(
        allow_inf_nan=False,
        description=(
            "Sum of hourly precipitation buckets t+1..t+6 "
            "(strictly after the observation hour). Unverified "
            "runtime convention; for past timestamps this is "
            "the provider's current best-match series, not an "
            "as-issued historical forecast."
        ),
    )
    provider: str
    data_quality: WeatherDataQuality
