"""Public contracts for Component 4 quality assessment."""
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AcousticUploadMetadata(BaseModel):
    filename: str = Field(default="recording.wav", min_length=1, max_length=160)
    sheet_id: str | None = Field(default=None, min_length=1, max_length=64)
    condition_id: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("filename", "sheet_id", "condition_id", mode="before")
    @classmethod
    def clean_text(cls, value):
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Identifier/filename cannot be blank.")
        return value


class AcousticMeasurementResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    measurement_id: UUID
    created_at: datetime
    filename: str
    sheet_id: str | None = None
    condition_id: str | None = None
    source_sha256: str
    processing_status: Literal["COMPLETED"] = "COMPLETED"
    technical_status: Literal["TECHNICAL_PASSED", "TECHNICAL_REJECTED"]
    quality_status: Literal["VALID", "INVALID"]
    classifier_status: str
    decision_source: Literal["TECHNICAL_RULES", "ML_CLASSIFIER"]
    rejection_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rule_version: str
    model_version: str | None = None
    model_status: Literal["PRELIMINARY_DEMO"] = "PRELIMINARY_DEMO"
    invalid_probability_uncalibrated: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    candidate_count: int = Field(default=0, ge=0)
    onset_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    segment_duration_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    peak_percent_full_scale: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    near_full_scale_fraction: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    moisture_percent: None = None
    guidance: str
