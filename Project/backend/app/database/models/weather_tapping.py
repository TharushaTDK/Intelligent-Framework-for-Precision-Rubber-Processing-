from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.session import Base


# ============================================================
# COMPONENT 1 SAVED ADVISORY
# ============================================================


class TappingAdvisory(Base):
    """
    One saved Component 1 advisory: where it was for, when, the
    weather it used and what the models returned.

    Saved so later components (Component 2 needs the DRC and the
    location) can pick the values up instead of re-running the
    models. Every model value here is an ESTIMATE from the
    Component 1 models, never a field measurement.
    """

    __tablename__ = "tapping_advisories"

    __table_args__ = (
        CheckConstraint(
            "latitude BETWEEN -90 AND 90",
            name="ck_tapping_advisories_latitude",
        ),
        CheckConstraint(
            "longitude BETWEEN -180 AND 180",
            name="ck_tapping_advisories_longitude",
        ),
        CheckConstraint(
            "location_source IS NULL OR location_source IN ('map', 'gps', 'typed')",
            name="ck_tapping_advisories_location_source",
        ),
        CheckConstraint(
            "tapping_label IS NULL OR tapping_label IN ('High', 'Low', 'Moderate')",
            name="ck_tapping_advisories_tapping_label",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # When the record was saved (set by the database).
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    # ---------------- location ----------------

    latitude: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    longitude: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    # How the farmer chose the location on the dashboard.
    location_source: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
    )

    # ---------------- weather used ----------------

    # Hour the advisory is for, as returned by the weather API.
    observation_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    # True when that hour was in the future, i.e. the weather and
    # everything built on it was forecast-based.
    forecast_based: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    temperature_c: Mapped[float] = mapped_column(Float, nullable=False)
    relative_humidity_pct: Mapped[float] = mapped_column(Float, nullable=False)
    wind_speed_kmh: Mapped[float] = mapped_column(Float, nullable=False)
    rainfall_last_6h_mm: Mapped[float] = mapped_column(Float, nullable=False)
    rainfall_last_3days_mm: Mapped[float] = mapped_column(Float, nullable=False)
    forecast_rain_next_6h_mm: Mapped[float] = mapped_column(Float, nullable=False)

    # ---------------- model results ----------------

    # Estimated dry rubber content (percent) from the DRC model.
    predicted_latex_drc_pct: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    # Which exact DRC model produced it.
    drc_model_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    drc_runtime_artifact_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    # Tapping and price are optional: they may fail on their own
    # while the DRC estimate still exists.
    tapping_label: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
    )

    tapping_probability_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    tapping_probability_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    tapping_probability_moderate: Mapped[float | None] = mapped_column(Float, nullable=True)

    predicted_price_lkr_per_kg: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
