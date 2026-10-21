"""Database model for Component 4 acoustic measurements."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.session import Base


class AcousticMeasurement(Base):
    """One completed recording-quality assessment.

    WAV bytes remain in file storage. PostgreSQL stores the
    recording reference, identifiers, assessment and provenance.
    """

    __tablename__ = "acoustic_measurements"

    __table_args__ = (
        CheckConstraint(
            "quality_status IN ('VALID', 'INVALID')",
            name="ck_acoustic_measurements_quality",
        ),
        CheckConstraint(
            "technical_status IN "
            "('TECHNICAL_PASSED', 'TECHNICAL_REJECTED')",
            name="ck_acoustic_measurements_technical",
        ),
        CheckConstraint(
            "decision_source IN ('TECHNICAL_RULES', 'ML_CLASSIFIER')",
            name="ck_acoustic_measurements_decision",
        ),
        CheckConstraint(
            "invalid_probability_uncalibrated IS NULL OR "
            "invalid_probability_uncalibrated BETWEEN 0 AND 1",
            name="ck_acoustic_measurements_invalid_score",
        ),
        CheckConstraint(
            "candidate_count >= 0",
            name="ck_acoustic_measurements_candidates",
        ),
        CheckConstraint(
            "moisture_percent IS NULL OR "
            "moisture_percent BETWEEN 0 AND 100",
            name="ck_acoustic_measurements_moisture",
        ),
    )

    # Identification and recording provenance
    measurement_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    filename: Mapped[str] = mapped_column(String(160), nullable=False)

    # Server-generated basename, not a client-supplied filesystem path.
    raw_file_name: Mapped[str] = mapped_column(
        String(80),
        nullable=False,
        unique=True,
    )

    sheet_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    condition_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    source_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )

    # Assessment status
    processing_status: Mapped[str] = mapped_column(
        String(32), nullable=False
    )
    technical_status: Mapped[str] = mapped_column(
        String(32), nullable=False
    )
    quality_status: Mapped[str] = mapped_column(
        String(16), nullable=False
    )
    classifier_status: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    decision_source: Mapped[str] = mapped_column(
        String(32), nullable=False
    )

    rejection_reasons: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    warnings: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )

    # Processing and model versions
    rule_version: Mapped[str] = mapped_column(
        String(100), nullable=False
    )
    model_version: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )
    model_status: Mapped[str] = mapped_column(
        String(32), nullable=False
    )

    # Recording diagnostics
    invalid_probability_uncalibrated: Mapped[float | None] = (
        mapped_column(Float, nullable=True)
    )
    candidate_count: Mapped[int] = mapped_column(
        Integer, nullable=False
    )
    onset_seconds: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    segment_duration_seconds: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    peak_percent_full_scale: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    near_full_scale_fraction: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )

    # Reserved for the future moisture model; currently always NULL.
    moisture_percent: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )

    guidance: Mapped[str] = mapped_column(Text, nullable=False)