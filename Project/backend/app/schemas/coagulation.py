from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CoagulationOutput = Literal["water_acid", "coagulation_time"]
RecommendationStatus = Literal["recommended", "in_progress", "completed", "cancelled"]


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


class CoagulationRecommendationCreate(_CoagulationInput):
    estate_id: str = Field(max_length=50, description="Estate identifier from Component 1.")
    # manual DRC until the Component 1 DRC service is connected
    drc: float | None = Field(
        default=None,
        gt=0,
        le=100,
        allow_inf_nan=False,
        description="Dry rubber content in %. If left out, the estate DRC from Component 1 is used.",
    )

    @field_validator("estate_id", mode="before")
    @classmethod
    def _clean_estate_id(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("estate_id must not be empty.")
        return value


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
    # live timer values, worked out when the record is read
    time_remaining_hours: float | None = None
    is_due: bool | None = None
    alert_interval_minutes: int | None = None
    next_alert_at: datetime | None = None
    alerts: list[CoagulationAlert] = []


class CoagulationHistoryResponse(BaseModel):
    items: list[CoagulationRecommendationResponse]
