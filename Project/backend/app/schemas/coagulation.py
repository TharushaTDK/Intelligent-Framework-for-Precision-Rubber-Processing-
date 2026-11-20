from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CoagulationOutput = Literal["water_acid", "coagulation_time"]
RecommendationStatus = Literal["recommended", "in_progress", "completed", "cancelled"]
StageKey = Literal["liquid", "thickening", "setting", "firm"]


def _require_timezone(value: datetime | None) -> datetime | None:
    if value is not None and (value.tzinfo is None or value.utcoffset() is None):
        raise ValueError("Timestamp must include timezone information, e.g. 'Z' or '+05:30'.")
    return value


class _CoagulationInput(BaseModel):
    latex_volume_L: float = Field(gt=0, allow_inf_nan=False, description="Latex batch volume in litres.")
    outputs: list[CoagulationOutput] = Field(
        min_length=1,
        description="What the farmer wants: water_acid, coagulation_time or both.",
    )

    @field_validator("outputs")
    @classmethod
    def _no_duplicate_outputs(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("outputs must not contain duplicates.")
        return value


class _MapPoint(BaseModel):
    latitude: float | None = Field(default=None, ge=-90, le=90, allow_inf_nan=False)
    longitude: float | None = Field(default=None, ge=-180, le=180, allow_inf_nan=False)


class CoagulationRecommendationCreate(_CoagulationInput, _MapPoint):
    location_id: int | None = Field(
        default=None,
        ge=1,
        description="Saved Component 1 location (advisory id). Its DRC is used and its coordinates become the estate.",
    )
    estate_id: str | None = Field(
        default=None,
        max_length=50,
        description="Only used with a manual DRC when no location is chosen.",
    )
    drc: float | None = Field(
        default=None,
        gt=0,
        le=100,
        allow_inf_nan=False,
        description="Manual dry rubber content in %. Overrides the Component 1 DRC.",
    )
    observation_timestamp: datetime | None = Field(
        default=None,
        description="Weather hour for a map point, as returned by /location-drc. Defaults to now.",
    )

    @field_validator("estate_id", mode="before")
    @classmethod
    def _clean_estate_id(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("estate_id must not be empty.")
        return value

    @field_validator("observation_timestamp")
    @classmethod
    def _aware(cls, value: datetime | None) -> datetime | None:
        return _require_timezone(value)

    @model_validator(mode="after")
    def _location_or_manual_drc(self) -> "CoagulationRecommendationCreate":
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be given together.")
        if self.latitude is not None and self.location_id is not None:
            raise ValueError("Give either a map point or a location_id, not both.")
        has_location = self.location_id is not None or self.latitude is not None
        if not has_location and (self.estate_id is None or self.drc is None):
            raise ValueError("Pick a map point or location_id, or give estate_id and drc for a manual DRC.")
        return self


class LocationDrcRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)


class LocationDrcResponse(BaseModel):
    latitude: float
    longitude: float
    location: str
    drc: float
    drc_source: str
    observation_timestamp: datetime


class CoagulationPreviewRequest(_CoagulationInput):
    # preview takes DRC directly so the models can be checked without Component 1
    drc: float = Field(gt=0, le=100, allow_inf_nan=False, description="Dry rubber content in %.")


class WaterAcidResult(BaseModel):
    water_amount_L: float = Field(allow_inf_nan=False)
    prepared_1pct_formic_acid_solution_mL: float = Field(allow_inf_nan=False)


class CoagulationGuidance(BaseModel):
    output: Literal["water", "acid", "coagulation_time"]
    section: str
    title: str
    guidance_heading: str = "Guidance for the farmer"
    value: float
    unit: Literal["L", "mL", "hours"]
    display_value: str
    guidance: list[str] = Field(description="Guidance points shown one by one in the app.")


class CoagulationPreviewResponse(BaseModel):
    drc: float
    latex_volume_L: float
    water_acid: WaterAcidResult | None = None
    coagulation_time_hours: float | None = Field(default=None, allow_inf_nan=False)
    outside_training_range: bool
    notes: list[str]
    guidance: list[CoagulationGuidance]


class CoagulationStatusUpdate(BaseModel):
    status: Literal["in_progress", "completed", "cancelled"]
    changed_at: datetime | None = Field(default=None, description="Defaults to the server time.")

    @field_validator("changed_at")
    @classmethod
    def _aware(cls, value: datetime | None) -> datetime | None:
        return _require_timezone(value)


class CoagulationAlert(BaseModel):
    alert_at: datetime
    type: Literal["progress", "due", "overdue"]
    message: str
    is_past: bool


class CoagulationStage(BaseModel):
    key: StageKey
    number: int = Field(ge=1, le=4)
    label: str
    description: str


class CoagulationStep(BaseModel):
    number: int
    title: str
    detail: str


class AcidStockUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stock_mL: float = Field(ge=0, le=100000, allow_inf_nan=False, description="Prepared 1% acid solution on hand.")


class AcidStockAdd(BaseModel):
    model_config = ConfigDict(extra="forbid")

    added_mL: float = Field(gt=0, le=100000, allow_inf_nan=False, description="Newly prepared 1% acid solution.")


class AcidStockResponse(BaseModel):
    stock_mL: float | None = Field(default=None, description="None until the farmer enters the stock.")
    updated_at: datetime | None = None
    average_batch_acid_mL: float | None = None
    batches_left: int | None = None
    low_stock: bool
    message: str


class CoagulationRecommendationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    estate_id: str
    latex_volume_L: float
    drc: float
    drc_source: str
    outputs: list[CoagulationOutput]
    water_acid: WaterAcidResult | None = None
    coagulation_time_hours: float | None = None
    outside_training_range: bool
    notes: list[str]
    guidance: list[CoagulationGuidance]
    water_acid_model_version: str
    coagulation_time_model_version: str | None = None
    status: RecommendationStatus
    created_at: datetime
    started_at: datetime | None = None
    expected_completion_at: datetime | None = None
    completed_at: datetime | None = None
    actual_coagulation_time_hours: float | None = None
    acid_used_mL: float | None = None
    steps: list[CoagulationStep] = []
    # live timer values, worked out when the record is read
    time_remaining_hours: float | None = None
    is_due: bool | None = None
    alert_interval_minutes: int | None = None
    next_alert_at: datetime | None = None
    alerts: list[CoagulationAlert] = []
    progress_pct: float | None = None
    stage: CoagulationStage | None = None


class CoagulationHistoryResponse(BaseModel):
    items: list[CoagulationRecommendationResponse]
