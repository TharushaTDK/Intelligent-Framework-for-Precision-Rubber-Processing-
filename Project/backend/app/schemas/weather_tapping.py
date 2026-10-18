from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)


# ============================================================
# COMPONENT 1 REQUEST CONTRACTS
# ============================================================
#
# Three independent stateless prediction contracts:
#
#   1. DRC prediction            (target: latex_drc_pct)
#   2. Tapping suitability proxy (target: tapping_suitability_proxy)
#   3. Rubber price estimation   (target: price_lkr_per_kg)
#
# Clients never supply engineered ML features. month and
# day_of_year are derived by the backend from the explicit
# observation timestamp, exactly as training did from the
# naive observation-local wall clock.
#
# Optional numeric fields default to None = "observation
# missing". Missing values (and values the research cleaning
# policy considered out of range) are median-imputed inside
# the fitted pipeline; the backend never invents readings.
#
# estate_name is NOT a model feature and is rejected as an
# unexpected field. Collection-time prediction does not
# exist in this component.
# ============================================================


# ============================================================
# SHARED VALIDATION HELPERS
# ============================================================


def _require_timezone_aware_datetime(
    value: datetime,
    field_name: str,
) -> datetime:
    """
    Require the observation timestamp to carry an explicit
    UTC offset.

    The backend never silently assumes UTC and never treats
    a naive timestamp as if it were UTC. The supplied offset
    defines the observation-local wall clock, which the
    service converts explicitly (never silently) to the
    configured observation timezone before deriving month
    and day_of_year.
    """

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            f"{field_name} must include an explicit timezone "
            "offset (for example 2025-01-02T08:00:00+05:30 "
            "or 2025-01-02T02:30:00Z). Naive timestamps are "
            "rejected: the backend never assumes a timezone "
            "silently."
        )

    return value


def _reject_empty_identifier(
    value: object,
    field_name: str,
) -> object:
    """
    Reject empty or whitespace-only identifier strings
    WITHOUT rewriting the value.

    Category values are passed through exactly as supplied
    (tapping_system is normalized later using the documented
    training policy, not here).
    """

    if not isinstance(value, str):
        return value

    if not value.strip():
        raise ValueError(
            f"{field_name} must not be empty or "
            "whitespace-only."
        )

    return value


# ============================================================
# DRC PREDICTION REQUEST
# ============================================================


class DrcPredictionRequest(BaseModel):
    """
    Stateless request for one latex dry-rubber-content (DRC)
    prediction.

    Ordered features (from the verified runtime manifest):

        temperature_c
        relative_humidity_pct
        rainfall_last_6h_mm
        forecast_rain_next_6h_mm
        rainfall_last_3days_mm
        wind_speed_kmh
        soil_moisture_pct
        soil_ph
        estate_id
        tree_count
        tapping_system
        days_since_last_tap
        previous_collection_yield_litres
        month        (derived from observation_timestamp)
        day_of_year  (derived from observation_timestamp)
    """

    model_config = ConfigDict(extra="forbid")

    observation_timestamp: datetime = Field(
        description=(
            "Explicit timestamp of the observation, with an "
            "explicit UTC offset. The month and day_of_year "
            "model features are derived from its wall clock "
            "in the configured observation timezone "
            "(default Asia/Colombo), exactly as training "
            "derived them from observation-local time."
        ),
    )

    temperature_c: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description=(
            "Air temperature in degrees Celsius. "
            "Research policy: values outside [-5, 60] are "
            "treated as missing and median-imputed."
        ),
    )

    relative_humidity_pct: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description=(
            "Relative humidity in percent. "
            "Research policy: values outside [0, 100] are "
            "treated as missing and median-imputed."
        ),
    )

    rainfall_last_6h_mm: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description=(
            "Rainfall over the last 6 hours in millimetres. "
            "Research policy: negative values are treated as "
            "missing and median-imputed."
        ),
    )

    forecast_rain_next_6h_mm: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description=(
            "Forecast rainfall for the next 6 hours in "
            "millimetres. Research policy: negative values "
            "are treated as missing and median-imputed."
        ),
    )

    rainfall_last_3days_mm: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description=(
            "Rainfall over the last 3 days in millimetres. "
            "Research policy: negative values are treated as "
            "missing and median-imputed."
        ),
    )

    wind_speed_kmh: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description=(
            "Wind speed in kilometres per hour. "
            "Research policy: negative values are treated as "
            "missing and median-imputed."
        ),
    )

    soil_moisture_pct: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description=(
            "Soil moisture in percent. "
            "Research policy: values outside [0, 100] are "
            "treated as missing and median-imputed."
        ),
    )

    soil_ph: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description=(
            "Soil pH. Research policy: values outside "
            "[0, 14] are treated as missing and "
            "median-imputed."
        ),
    )

    estate_id: str | None = Field(
        default=None,
        max_length=50,
        description=(
            "Estate identifier as a categorical feature "
            "(for example EST-001). Unseen categories are "
            "handled by the fitted one-hot encoder's "
            "unknown-ignore behaviour. estate_name is never "
            "used as a model feature."
        ),
    )

    tree_count: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description=(
            "Number of trees on the estate block. "
            "No research range rule applies."
        ),
    )

    tapping_system: str | None = Field(
        default=None,
        max_length=20,
        description=(
            "Tapping system category (for example S/2, D2, "
            "D3, D4). Normalized like training: trimmed and "
            "upper-cased. Missing values are "
            "most-frequency-imputed inside the pipeline."
        ),
    )

    days_since_last_tap: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description=(
            "Days elapsed since the last tap. "
            "No research range rule applies."
        ),
    )

    previous_collection_yield_litres: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description=(
            "Previous collection yield in litres. "
            "No research range rule applies."
        ),
    )

    @field_validator("observation_timestamp")
    @classmethod
    def validate_observation_timestamp_timezone(
        cls,
        value: datetime,
        info: ValidationInfo,
    ) -> datetime:
        return _require_timezone_aware_datetime(
            value,
            info.field_name,
        )

    @field_validator("estate_id", "tapping_system")
    @classmethod
    def validate_identifier_strings(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> object:
        return _reject_empty_identifier(
            value,
            info.field_name,
        )


# ============================================================
# TAPPING SUITABILITY PROXY REQUEST
# ============================================================


class TappingSuitabilityRequest(BaseModel):
    """
    Stateless request for one tapping-suitability PROXY
    prediction.

    The target is a rule-derived research proxy with
    simulated label noise - NOT expert-verified tapping
    advice.

    Ordered features (from the verified runtime manifest):

        temperature_c
        relative_humidity_pct
        rainfall_last_6h_mm
        forecast_rain_next_6h_mm
        rainfall_last_3days_mm
        wind_speed_kmh
        soil_moisture_pct
        soil_ph
        tree_count
        tapping_system
        days_since_last_tap
        month        (derived from observation_timestamp)
        day_of_year  (derived from observation_timestamp)

    estate_id and estate_name are NOT features of this
    model, and collection-time prediction does not exist.
    """

    model_config = ConfigDict(extra="forbid")

    observation_timestamp: datetime = Field(
        description=(
            "Explicit timestamp of the observation, with an "
            "explicit UTC offset."
        ),
    )

    temperature_c: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    relative_humidity_pct: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    rainfall_last_6h_mm: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    forecast_rain_next_6h_mm: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    rainfall_last_3days_mm: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    wind_speed_kmh: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    soil_moisture_pct: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    soil_ph: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    tree_count: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    tapping_system: str | None = Field(
        default=None,
        max_length=20,
    )

    days_since_last_tap: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    @field_validator("observation_timestamp")
    @classmethod
    def validate_observation_timestamp_timezone(
        cls,
        value: datetime,
        info: ValidationInfo,
    ) -> datetime:
        return _require_timezone_aware_datetime(
            value,
            info.field_name,
        )

    @field_validator("tapping_system")
    @classmethod
    def validate_tapping_system_string(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> object:
        return _reject_empty_identifier(
            value,
            info.field_name,
        )


# ============================================================
# PRICE ESTIMATION REQUEST
# ============================================================


class PriceEstimationRequest(BaseModel):
    """
    Stateless request for one rubber price estimation.

    Ordered features (from the verified runtime manifest):

        latex_drc_pct
        temperature_c
        relative_humidity_pct
        rainfall_last_6h_mm
        soil_moisture_pct
        soil_ph
        tree_count
        tapping_system
        month        (derived from observation_timestamp)
        day_of_year  (derived from observation_timestamp)

    latex_drc_pct may be a measured value or a value
    produced by the DRC prediction operation; the caller
    supplies it explicitly - the service never chains the
    models implicitly.
    """

    model_config = ConfigDict(extra="forbid")

    observation_timestamp: datetime = Field(
        description=(
            "Explicit timestamp of the observation, with an "
            "explicit UTC offset."
        ),
    )

    latex_drc_pct: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description=(
            "Latex dry-rubber content in percent. "
            "Research policy: values outside [0, 100] are "
            "treated as missing and median-imputed."
        ),
    )

    temperature_c: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    relative_humidity_pct: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    rainfall_last_6h_mm: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    soil_moisture_pct: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    soil_ph: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    tree_count: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    tapping_system: str | None = Field(
        default=None,
        max_length=20,
    )

    @field_validator("observation_timestamp")
    @classmethod
    def validate_observation_timestamp_timezone(
        cls,
        value: datetime,
        info: ValidationInfo,
    ) -> datetime:
        return _require_timezone_aware_datetime(
            value,
            info.field_name,
        )

    @field_validator("tapping_system")
    @classmethod
    def validate_tapping_system_string(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> object:
        return _reject_empty_identifier(
            value,
            info.field_name,
        )


# ============================================================
# RESPONSE CONTRACTS
# ============================================================


TappingLabel = Literal["High", "Low", "Moderate"]


class ModelIdentifierFields(BaseModel):
    """
    Identifiers of the exact finalized model that produced a
    prediction.

    runtime_artifact_sha256 is the trusted hash of the
    scikit-learn 1.9.1-compatible runtime artifact; the
    original_artifact_sha256 is the hash of the untouched
    research artifact it was re-serialized from.
    """

    model_name: str

    model_class: str

    runtime_artifact_sha256: str

    original_artifact_sha256: str

    dataset_provenance: str


class CalendarFeatureFields(BaseModel):
    """
    Calendar features derived from observation_timestamp in
    the configured observation timezone (never silently in
    UTC), matching training's observation-local semantics.
    """

    observation_month: int = Field(
        ge=1,
        le=12,
    )

    observation_day_of_year: int = Field(
        ge=1,
        le=366,
    )


# ============================================================
# DRC PREDICTION RESPONSE
# ============================================================


class DrcPredictionResponse(
    ModelIdentifierFields,
    CalendarFeatureFields,
):
    """
    Raw DRC prediction produced by the finalized regression
    pipeline.

    The value is preserved exactly as produced by the model
    (no clamping or rounding). The research validation
    tolerance for this task is +/- 1.0 DRC percentage
    point; that tolerance is documentation of research
    uncertainty, not a runtime adjustment.
    """

    target_name: Literal["latex_drc_pct"] = (
        "latex_drc_pct"
    )

    unit: Literal["percent"] = "percent"

    predicted_latex_drc_pct: float = Field(
        allow_inf_nan=False,
    )


# ============================================================
# TAPPING SUITABILITY PROXY RESPONSE
# ============================================================


class TappingClassProbabilities(BaseModel):
    """
    Probability for each fitted class, mapped explicitly to
    the fitted label names.

    The fitted predict_proba() column order is exactly:

        column 0 -> High
        column 1 -> Low
        column 2 -> Moderate
    """

    high: float = Field(
        ge=0,
        le=1,
        allow_inf_nan=False,
    )

    low: float = Field(
        ge=0,
        le=1,
        allow_inf_nan=False,
    )

    moderate: float = Field(
        ge=0,
        le=1,
        allow_inf_nan=False,
    )


class TappingSuitabilityResponse(
    ModelIdentifierFields,
    CalendarFeatureFields,
):
    """
    Tapping-suitability PROXY prediction.

    This is a rule-derived research proxy trained on
    synthetic labels with approximately 30% simulated label
    noise (fixed seed 340). It is NOT expert-verified
    tapping advice and must not be presented as agronomic
    fact or field-validated guidance.
    """

    target_name: Literal["tapping_suitability_proxy"] = (
        "tapping_suitability_proxy"
    )

    predicted_label: TappingLabel

    class_order: list[TappingLabel] = Field(
        description=(
            "Exact fitted class order; predict_proba() "
            "columns follow this order."
        ),
    )

    probabilities: TappingClassProbabilities

    is_rule_derived_proxy: bool = True

    research_warning: str


# ============================================================
# PRICE ESTIMATION RESPONSE
# ============================================================


class PriceEstimationResponse(
    ModelIdentifierFields,
    CalendarFeatureFields,
):
    """
    Raw price estimation produced by the finalized
    regression pipeline.

    The value is preserved exactly as produced by the model
    (no clamping or rounding). The research validation
    tolerance for this task is +/- 9.0 LKR per kg; that
    tolerance is documentation of research uncertainty, not
    a runtime adjustment.
    """

    target_name: Literal["price_lkr_per_kg"] = (
        "price_lkr_per_kg"
    )

    unit: Literal["LKR per kg"] = "LKR per kg"

    predicted_price_lkr_per_kg: float = Field(
        allow_inf_nan=False,
    )
